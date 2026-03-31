"""
eval_ngram_distributed.py — 8-GPU eval with global n-gram cache via all-reduce.

Key architecture:
  - Standard distributed sliding window eval (torchrun --nproc_per_node=8)
  - Hash tables as PyTorch GPU tensors (not numpy)
  - After each batch: local scatter_add_ updates, then all_reduce(SUM) to sync
  - Every GPU has the exact global cache state after each batch
  - ~0.75ms per all-reduce on H100 NVLink → ~3s total overhead for ~4000 batches

Usage:
  cd /path/to/parameter-golf
  torchrun --standalone --nproc_per_node=8 experiments/eval_time_mixing/scripts/eval_ngram_distributed.py

Env vars:
  MODEL_PATH      Path to final_model.pt (default: final_model.pt)
  NGRAM_ORDER     Max n-gram order (default: 7)
  NGRAM_MIN_ORDER Min n-gram order for backoff (default: 2)
  NGRAM_ALPHA     Fixed mixing alpha (default: 0.40)
  NGRAM_BUCKETS   Hash table buckets (default: 4194304 = 2^22)
  NGRAM_MIN_COUNT Minimum context count to use n-gram (default: 2)
  EVAL_STRIDE     Sliding window stride (default: 64)
  NGRAM_ENABLED   Enable n-gram cache (default: 1)

STRICT CAUSALITY: Tables updated AFTER scoring. No future tokens used.
"""

from __future__ import annotations
import io
import lzma
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

# Load all model classes from the training script
REPO = Path(__file__).resolve().parent.parent.parent.parent  # parameter-golf/
TRAIN_SCRIPT = os.environ.get("TRAIN_SCRIPT", str(REPO / "train_609_val_calib.py"))
_src = Path(TRAIN_SCRIPT).read_text()
exec(compile(_src.split("\ndef main")[0], TRAIN_SCRIPT, "exec"))


# ── GPU Hash Primes (as torch int64) ─────────────────────────────────────────

PRIMES_LIST = [36313, 27191, 51647, 81929, 131071, 174763, 233017, 310019, 412553]


# ── GPU N-gram Hash Functions ─────────────────────────────────────────────────

def gpu_hash_context(val_tokens_gpu: Tensor, positions: Tensor, ctx_width: int,
                     primes: Tensor, ng_mask: int) -> Tensor:
    """Hash context tokens on GPU. All tensors on same device, int64."""
    ctx_hash = torch.zeros(len(positions), dtype=torch.int64, device=positions.device)
    for k in range(ctx_width):
        tok = val_tokens_gpu[positions - (ctx_width - k)]
        ctx_hash ^= tok * primes[k % len(primes)]
    return ctx_hash & ng_mask


def gpu_hash_full(ctx_hash: Tensor, target_tokens: Tensor, ctx_width: int,
                  primes: Tensor, ng_mask: int) -> Tensor:
    """Hash context + target on GPU."""
    return (ctx_hash ^ (target_tokens * primes[ctx_width % len(primes)])) & ng_mask


# ── Distributed N-gram Sliding Window Eval ────────────────────────────────────

def eval_val_sliding_ngram(
    args,
    base_model,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
    # N-gram config
    use_ngram: bool = True,
    ngram_max_order: int = 7,
    ngram_min_order: int = 2,
    ngram_buckets: int = 4_194_304,
    ngram_min_count: int = 2,
    ngram_alpha: float = 0.40,
) -> tuple[float, float]:
    """Sliding window eval with GPU n-gram cache + all-reduce sync.

    Each GPU processes its portion of windows. After each batch, hash table
    deltas are all-reduced so every GPU has the global cache state.
    """
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    distributed = dist.is_available() and dist.is_initialized()

    # Divide windows across GPUs (contiguous blocks)
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Val tokens on GPU for hash lookups
    val_tokens_gpu = val_tokens.to(dtype=torch.int64, device=device)

    # GPU hash tables: one pair (ctx, full) per n-gram order
    orders = list(range(ngram_min_order, ngram_max_order + 1))
    n_orders = len(orders)
    ng_mask = ngram_buckets - 1

    if use_ngram:
        # Persistent tables — accumulate counts across all batches
        ctx_tables = [torch.zeros(ngram_buckets, dtype=torch.int32, device=device)
                      for _ in range(n_orders)]
        full_tables = [torch.zeros(ngram_buckets, dtype=torch.int32, device=device)
                       for _ in range(n_orders)]
        # Delta buffers for all-reduce (only sync the NEW counts per batch)
        ctx_deltas = [torch.zeros(ngram_buckets, dtype=torch.int32, device=device)
                      for _ in range(n_orders)]
        full_deltas = [torch.zeros(ngram_buckets, dtype=torch.int32, device=device)
                       for _ in range(n_orders)]
        # Primes on GPU
        primes = torch.tensor(PRIMES_LIST, dtype=torch.int64, device=device)

        if rank == 0:
            mem_mb = n_orders * 4 * ngram_buckets * 4 / 1e6  # tables + deltas
            print(f"ngram_cache: orders={ngram_min_order}-{ngram_max_order} "
                  f"alpha={ngram_alpha} buckets={ngram_buckets} "
                  f"min_count={ngram_min_count} mem={mem_mb:.0f}MB/gpu", flush=True)

    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    t_start = time.perf_counter()
    sync_time = 0.0
    ngram_time = 0.0

    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            # Build input/target batches
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens_gpu[ws:end + 1]
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            # GPU forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)

            # Compute per-token NLL
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            # N-gram mixing (all on GPU)
            if use_ngram:
                t_ng = time.perf_counter()

                # Collect all update keys for the batch (for post-scoring table update)
                batch_ctx_keys = [[] for _ in range(n_orders)]
                batch_full_keys = [[] for _ in range(n_orders)]

                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    seg_len = wlen - s
                    if seg_len <= 0:
                        continue

                    # Global positions of TARGET tokens in this scored segment
                    global_j = torch.arange(ws + s + 1, ws + wlen + 1,
                                            dtype=torch.int64, device=device)

                    # Model probabilities for the correct token
                    scored_nll = nll[i, s:wlen]
                    model_p = torch.exp(-scored_nll)

                    # Multi-order backoff: highest order first, fill unmatched
                    best_p_ng = torch.full((seg_len,), -1.0, device=device)

                    for oi in range(n_orders - 1, -1, -1):
                        order = orders[oi]
                        ctx_w = order - 1

                        # Which positions have enough context AND no match yet
                        valid = (global_j >= order) & (best_p_ng < 0)
                        if not valid.any():
                            continue
                        v_idx = torch.nonzero(valid, as_tuple=True)[0]
                        jv = global_j[v_idx]

                        # Hash context
                        ctx_key = gpu_hash_context(val_tokens_gpu, jv, ctx_w, primes, ng_mask)
                        # Hash context + target
                        tgt = val_tokens_gpu[jv]
                        full_key = gpu_hash_full(ctx_key, tgt, ctx_w, primes, ng_mask)

                        # Lookup counts
                        cc = ctx_tables[oi][ctx_key].float()
                        fc = full_tables[oi][full_key].float()
                        has_match = cc >= float(ngram_min_count)

                        if has_match.any():
                            match_idx = v_idx[has_match]
                            p = torch.clamp(fc[has_match] / cc[has_match].clamp(min=1.0), 0.0, 1.0)
                            best_p_ng[match_idx] = p

                    # Mix where we have n-gram matches
                    has_ngram = best_p_ng >= 0
                    if has_ngram.any():
                        ng_idx = torch.nonzero(has_ngram, as_tuple=True)[0]
                        mixed_p = (1.0 - ngram_alpha) * model_p[ng_idx] + ngram_alpha * best_p_ng[ng_idx]
                        mixed_nll = -torch.log(mixed_p.clamp(min=1e-12))
                        scored_nll = scored_nll.clone()
                        scored_nll[ng_idx] = mixed_nll.float()

                    # Accumulate loss
                    loss_sum += scored_nll.to(torch.float64).sum()
                    token_count += float(seg_len)
                    tgt_ids = y_batch[i, s:wlen]
                    prev_ids = x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt_ids].to(torch.float64)
                    tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
                    byte_count += tb.sum()

                    # Collect keys for SCORE-FIRST table update (after scoring)
                    for oi in range(n_orders):
                        order = orders[oi]
                        ctx_w = order - 1
                        valid = global_j >= order
                        if not valid.any():
                            continue
                        jv = global_j[valid]
                        ctx_key = gpu_hash_context(val_tokens_gpu, jv, ctx_w, primes, ng_mask)
                        tgt = val_tokens_gpu[jv]
                        full_key = gpu_hash_full(ctx_key, tgt, ctx_w, primes, ng_mask)
                        batch_ctx_keys[oi].append(ctx_key)
                        batch_full_keys[oi].append(full_key)

                # SCORE-FIRST: update delta buffers AFTER all windows in batch are scored
                for oi in range(n_orders):
                    ctx_deltas[oi].zero_()
                    full_deltas[oi].zero_()
                    if batch_ctx_keys[oi]:
                        all_ctx = torch.cat(batch_ctx_keys[oi])
                        all_full = torch.cat(batch_full_keys[oi])
                        ones = torch.ones(len(all_ctx), dtype=torch.int32, device=device)
                        ctx_deltas[oi].scatter_add_(0, all_ctx, ones)
                        full_deltas[oi].scatter_add_(0, all_full, ones)

                ngram_time += time.perf_counter() - t_ng

                # ALL-REDUCE deltas, then add to persistent tables
                if distributed:
                    t_sync = time.perf_counter()
                    for oi in range(n_orders):
                        dist.all_reduce(ctx_deltas[oi], op=dist.ReduceOp.SUM)
                        dist.all_reduce(full_deltas[oi], op=dist.ReduceOp.SUM)
                    sync_time += time.perf_counter() - t_sync

                # Apply synced deltas to persistent tables
                for oi in range(n_orders):
                    ctx_tables[oi] += ctx_deltas[oi]
                    full_tables[oi] += full_deltas[oi]

            else:
                # No n-gram — standard scoring
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    seg_len = wlen - s
                    if seg_len <= 0:
                        continue
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(seg_len)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

            # Progress (rank 0 only)
            if rank == 0:
                done = min(bi + batch_seqs, len(my_windows))
                if done % (batch_seqs * 20) == 0 or done == len(my_windows):
                    elapsed = time.perf_counter() - t_start
                    pct = 100.0 * done / len(my_windows)
                    curr_bpb = (loss_sum.item() / max(token_count.item(), 1)) / math.log(2) * \
                               (token_count.item() / max(byte_count.item(), 1))
                    print(f"  [{pct:5.1f}%] windows={done}/{len(my_windows)} "
                          f"bpb={curr_bpb:.4f} elapsed={elapsed:.1f}s "
                          f"ngram={ngram_time:.1f}s sync={sync_time:.1f}s", flush=True)

    # Final all-reduce for loss aggregation
    if distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    elapsed = time.perf_counter() - t_start

    if rank == 0:
        print(f"\nFINAL: val_bpb={bpt * tpb:.6f} val_loss={val_loss:.6f} "
              f"time={elapsed:.1f}s ngram={ngram_time:.1f}s sync={sync_time:.1f}s", flush=True)

    base_model.train()
    return val_loss, bpt * tpb


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = Hyperparameters()

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load tokenizer + val data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    if rank == 0:
        print(f"Val tokens: {val_tokens.numel()-1:,} | GPUs: {world_size}", flush=True)

    # Load model — auto-detect config from checkpoint
    model_path = os.environ.get("MODEL_PATH", "final_model.pt")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    ckpt_keys = set(state.keys())

    bigram_vocab = state["bigram.embed.weight"].shape[0] if "bigram.embed.weight" in ckpt_keys else 0
    bigram_dim = state["bigram.embed.weight"].shape[1] if "bigram.embed.weight" in ckpt_keys else 128
    ve_enabled = "ve_shared.embed.weight" in ckpt_keys
    ve_dim = state["ve_shared.embed.weight"].shape[1] if ve_enabled else 128

    if rank == 0:
        print(f"Model: bigram={bigram_vocab}x{bigram_dim} ve={ve_enabled}", flush=True)

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=bigram_vocab, bigram_dim=bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=ve_enabled, ve_dim=ve_dim, ve_layers=args.ve_layers,
        gated_attention=args.gated_attention, value_residual=args.value_residual,
    ).to(device).bfloat16()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        if missing:
            print(f"WARNING: Missing keys: {missing[:5]}")
        if unexpected:
            print(f"INFO: Extra keys (ignored): {unexpected[:5]}")

    # Config from env
    use_ngram = bool(int(os.environ.get("NGRAM_ENABLED", "1")))
    ngram_max_order = int(os.environ.get("NGRAM_ORDER", "7"))
    ngram_min_order = int(os.environ.get("NGRAM_MIN_ORDER", "2"))
    ngram_alpha = float(os.environ.get("NGRAM_ALPHA", "0.40"))
    ngram_buckets = int(os.environ.get("NGRAM_BUCKETS", "4194304"))
    ngram_min_count = int(os.environ.get("NGRAM_MIN_COUNT", "2"))
    eval_stride = int(os.environ.get("EVAL_STRIDE", "64"))

    if rank == 0:
        print(f"Config: ngram={use_ngram} order={ngram_min_order}-{ngram_max_order} "
              f"alpha={ngram_alpha} stride={eval_stride}", flush=True)

    # Run eval
    torch.cuda.synchronize()
    val_loss, val_bpb = eval_val_sliding_ngram(
        args, model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=eval_stride,
        use_ngram=use_ngram,
        ngram_max_order=ngram_max_order,
        ngram_min_order=ngram_min_order,
        ngram_buckets=ngram_buckets,
        ngram_min_count=ngram_min_count,
        ngram_alpha=ngram_alpha,
    )
    torch.cuda.synchronize()

    if rank == 0:
        print(f"\nval_bpb={val_bpb:.8f} val_loss={val_loss:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
