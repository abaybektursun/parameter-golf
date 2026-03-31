"""
Eval-time PPM exponential tilt.

Phase 1: C++ PPMHintGenerator processes all tokens causally, outputs
         (hint_token, beta) per scored position using PPMC backoff.
Phase 2: GPU forward pass → logits. For each position:
         p_hint = softmax(logits)[hint_token]
         Z = 1 + p_hint * (exp(β) - 1)
         nll' = nll + log(Z) - β * (target == hint)
         This is a valid probability distribution: Σ p'(y) = 1 always.

Usage:
    python eval_context_mixer.py \
        --code train_gpt_mlp35_mixed.py \
        --model final_model.int6.ptz \
        --val-pattern './data/datasets/fineweb10B_sp1024/fineweb_val_*.bin' \
        --tokenizer ./data/tokenizers/fineweb_1024_bpe.model
"""
from __future__ import annotations
import argparse
import io
import math
import time
import glob
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F


# ── Data loading ────────────────────────────────────────────────────────────

def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=offset)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def build_sentencepiece_luts(sp, vocab_size):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros(table_size, dtype=np.int16)
    has_leading_space = np.zeros(table_size, dtype=np.uint8)
    is_boundary = np.ones(table_size, dtype=np.uint8)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary[token_id] = 0
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space[token_id] = 1
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary


# ── Model loading ───────────────────────────────────────────────────────────

def _decompressors():
    try:
        import brotli
        yield "brotli", brotli.decompress
    except ImportError:
        pass
    import lzma
    yield "lzma", lzma.decompress
    try:
        import zstandard
        yield "zstd", zstandard.ZstdDecompressor().decompress
    except ImportError:
        pass
    import zlib
    yield "zlib", zlib.decompress


def load_model(args, tg, device):
    model = tg.GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims,
        ln_scale=True, ve_enabled=True,
        ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()

    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for m in model.modules():
        if isinstance(m, tg.CastedLinear):
            m.float()
    tg.restore_low_dim_params_to_fp32(model)

    print(f"Loading {args.model}...")
    with open(args.model, "rb") as f:
        blob = f.read()
    decompressed = None
    for name, fn in _decompressors():
        try:
            decompressed = fn(blob)
            print(f"  Decompressed with {name}: {len(blob)} -> {len(decompressed)} bytes")
            break
        except Exception:
            continue
    assert decompressed is not None, "Could not decompress checkpoint"

    quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    unbanked = tg._unbank_state_dict(sd_cpu, args.num_layers)
    deq = tg.dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked)
    model.load_state_dict(tg._rebank_state_dict(deq, args.num_layers, sd_cpu), strict=True)
    model.eval()
    print("Model loaded.")
    return model


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PPM exponential tilt eval")
    parser.add_argument("--code", default="train_gpt_mlp35_mixed.py")
    parser.add_argument("--model", default="final_model.int6.ptz")
    parser.add_argument("--val-pattern",
                        default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer",
                        default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-seqs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=0)
    # PPM config
    parser.add_argument("--max-order", type=int, default=8)
    parser.add_argument("--token-table-bits", type=int, default=22)
    parser.add_argument("--within-table-bits", type=int, default=21)
    parser.add_argument("--word-order", type=int, default=4)
    parser.add_argument("--word-table-bits", type=int, default=20)
    # Boost config
    parser.add_argument("--base-beta", type=float, default=2.625)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--within-beta", type=float, default=0.75)
    parser.add_argument("--within-threshold", type=float, default=0.45)
    parser.add_argument("--word-beta", type=float, default=0.75)
    parser.add_argument("--word-threshold", type=float, default=0.65)
    parser.add_argument("--agree-bonus", type=float, default=0.5)
    parser.add_argument("--p-hint-max", type=float, default=0.1,
                        help="Only apply tilt when p_neural(hint) < this value")
    # Model config
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=11)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=float, default=3.5)
    parser.add_argument("--logit-softcap", type=float, default=30.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--qk-gain-init", type=float, default=1.5)
    parser.add_argument("--bigram-vocab-size", type=int, default=3072)
    parser.add_argument("--bigram-dim", type=int, default=112)
    parser.add_argument("--xsa-last-n", type=int, default=11)
    parser.add_argument("--rope-dims", type=int, default=16)
    parser.add_argument("--ve-dim", type=int, default=128)
    parser.add_argument("--ve-layers", default="9,10")
    args = parser.parse_args()

    device = torch.device(args.device)

    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gpt", args.code)
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)

    # Load val tokens
    val_files = sorted(glob.glob(args.val_pattern))
    assert val_files, f"No val files: {args.val_pattern}"
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    if args.max_tokens > 0:
        val_tokens = val_tokens[:args.max_tokens + 1]
    total_tokens = val_tokens.numel() - 1
    print(f"Val tokens: {total_tokens:,}")

    # LUTs
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    base_bytes_np, has_ls_np, is_bnd_np = build_sentencepiece_luts(sp, args.vocab_size)

    # Build byte-counting LUTs on GPU for fast lookup
    base_bytes_gpu = torch.tensor(base_bytes_np, dtype=torch.int16, device=device)
    has_ls_gpu = torch.tensor(has_ls_np, dtype=torch.bool, device=device)
    is_bnd_gpu = torch.tensor(is_bnd_np, dtype=torch.bool, device=device)

    # Load model
    model = load_model(args, tg, device)

    # Init PPM hint generator
    from fused_expert_ext import PPMHintGenerator
    ppm = PPMHintGenerator(
        max_order=args.max_order,
        token_table_bits=args.token_table_bits,
        within_table_bits=args.within_table_bits,
        word_order=args.word_order,
        word_table_bits=args.word_table_bits,
    )
    ppm.configure(args.base_beta, args.conf_threshold,
                  args.within_beta, args.within_threshold,
                  args.word_beta, args.word_threshold,
                  args.agree_bonus)

    val_np = val_tokens.numpy().astype(np.int64)
    ppm.set_tokens(val_np)
    ppm.set_luts(base_bytes_np, has_ls_np, is_bnd_np)

    # Sliding windows
    seq_len = args.seq_len
    stride = args.stride
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    num_windows = len(window_starts)
    print(f"Windows: {num_windows:,} (stride={stride}, seq_len={seq_len})")

    # ── Phase 1: Pre-compute PPM hints for all scored positions ─────────
    # Process ALL tokens causally (not just scored ones) so PPM sees
    # the full prefix. Output hints only for scored positions.
    print(f"\nPhase 1: PPM hint generation ({total_tokens:,} tokens)...")
    t1 = time.perf_counter()

    # Build scored position set
    scored_set = set()
    max_scored = 0
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        for j in range(wlen - s):
            gp = ws + s + j + 1
            if gp > max_scored:
                scored_set.add(gp)
                max_scored = gp

    # Process all tokens, collect hints for scored positions
    scored_positions = sorted(scored_set)
    scored_idx = {p: i for i, p in enumerate(scored_positions)}
    num_scored = len(scored_positions)

    all_hints = np.full(num_scored, -1, dtype=np.int32)
    all_betas = np.zeros(num_scored, dtype=np.float64)

    # Process in chunks through PPM
    chunk_size = 65536
    for ci in range(1, total_tokens + 1, chunk_size):
        end = min(ci + chunk_size, total_tokens + 1)
        positions = np.arange(ci, end, dtype=np.int64)
        hints_buf = np.zeros(len(positions), dtype=np.int32)
        betas_buf = np.zeros(len(positions), dtype=np.float64)
        ppm.get_hints_batch(positions, hints_buf, betas_buf)

        # Copy hints for scored positions
        for j, gp in enumerate(range(ci, end)):
            if gp in scored_idx:
                idx = scored_idx[gp]
                all_hints[idx] = hints_buf[j]
                all_betas[idx] = betas_buf[j]

    phase1_time = time.perf_counter() - t1
    hint_rate = np.sum(all_hints >= 0) / num_scored * 100
    print(f"Phase 1 done: {num_scored:,} scored, {hint_rate:.1f}% have hints, "
          f"{phase1_time:.1f}s ({total_tokens/phase1_time:,.0f} tok/s)")

    # ── Phase 2: GPU forward + exponential tilt ─────────────────────────
    print(f"\nPhase 2: GPU forward + exponential tilt ({num_windows} windows)...")
    t2 = time.perf_counter()

    # Pre-convert hints to GPU tensors (indexed by scored position)
    positions_np = np.array(scored_positions, dtype=np.int64)
    hints_torch = torch.tensor(all_hints, dtype=torch.int64)
    betas_torch = torch.tensor(all_betas, dtype=torch.float64)

    baseline_loss_sum = 0.0
    tilted_loss_sum = 0.0
    byte_sum = 0.0
    token_count = 0
    max_scored_so_far = 0  # for dedup
    # Diagnostics
    n_tilted = 0        # positions where tilt was applied
    n_tilt_hit = 0      # tilt applied AND target == hint
    n_tilt_miss = 0     # tilt applied AND target != hint
    sum_gain_hit = 0.0  # total NLL reduction from hits
    sum_cost_miss = 0.0 # total NLL increase from misses

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for bi in range(0, num_windows, args.batch_seqs):
            batch_ws = window_starts[bi:bi + args.batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            logits = model.forward_logits(x_batch)  # (bsz, seq_len, vocab)
            log_probs = F.log_softmax(logits.float(), dim=-1)  # (bsz, seq_len, vocab)

            # Per-position scoring with exponential tilt
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)

                targets = y_batch[i, s:wlen]
                prevs = x_batch[i, s:wlen]

                # Baseline NLL per position
                target_log_probs = log_probs[i, s:wlen].gather(
                    1, targets.unsqueeze(1)).squeeze(1)
                baseline_nll = -target_log_probs

                seg_len = wlen - s
                for j in range(seg_len):
                    gp = ws + s + j + 1
                    if gp <= max_scored_so_far:
                        continue
                    max_scored_so_far = gp

                    b_nll = baseline_nll[j].double().item()
                    t_nll = b_nll

                    idx = scored_idx.get(gp)
                    if idx is not None:
                        hint = int(all_hints[idx])
                        beta = float(all_betas[idx])
                        if hint >= 0 and beta > 0:
                            p_hint = log_probs[i, s + j, hint].exp().double().item()
                            if p_hint < args.p_hint_max:
                                Z = 1.0 + p_hint * (math.exp(beta) - 1.0)
                                log_Z = math.log(Z)
                                if targets[j].item() == hint:
                                    t_nll = b_nll - beta + log_Z
                                    n_tilted += 1; n_tilt_hit += 1
                                    sum_gain_hit += (b_nll - t_nll)
                                else:
                                    t_nll = b_nll + log_Z
                                    n_tilted += 1; n_tilt_miss += 1
                                    sum_cost_miss += (t_nll - b_nll)

                    baseline_loss_sum += b_nll
                    tilted_loss_sum += t_nll

                    tgt = targets[j].item()
                    prev = prevs[j].item()
                    tb = float(base_bytes_np[tgt])
                    if has_ls_np[tgt] and not is_bnd_np[prev]:
                        tb += 1.0
                    byte_sum += tb
                    token_count += 1

            if bi % (200 * args.batch_seqs) == 0:
                elapsed = time.perf_counter() - t2
                if token_count > 0 and byte_sum > 0:
                    tpb = token_count / byte_sum
                    b_bpb = (baseline_loss_sum / token_count / math.log(2.0)) * tpb
                    t_bpb = (tilted_loss_sum / token_count / math.log(2.0)) * tpb
                    print(f"  batch {bi//args.batch_seqs}/{(num_windows+args.batch_seqs-1)//args.batch_seqs}"
                          f" | base:{b_bpb:.6f} tilted:{t_bpb:.6f} delta:{t_bpb-b_bpb:+.6f}"
                          f" | {elapsed:.1f}s")

    phase2_time = time.perf_counter() - t2

    # Final results
    tpb = token_count / byte_sum
    base_bpb = (baseline_loss_sum / token_count / math.log(2.0)) * tpb
    tilt_bpb = (tilted_loss_sum / token_count / math.log(2.0)) * tpb

    print(f"\n{'=' * 72}")
    print(f"RESULTS: PPM Exponential Tilt")
    print(f"  max_order={args.max_order}, base_beta={args.base_beta}")
    print(f"  conf_threshold={args.conf_threshold}, p_hint_max={args.p_hint_max}")
    print(f"  stride={stride}, seq_len={seq_len}")
    print(f"{'=' * 72}")
    print(f"Neural only:  val_bpb = {base_bpb:.8f}")
    print(f"PPM tilted:   val_bpb = {tilt_bpb:.8f}")
    print(f"Delta:        {tilt_bpb - base_bpb:+.8f} BPB")
    print(f"Hint rate:    {hint_rate:.1f}%")
    print(f"Tilt applied: {n_tilted:,} positions ({n_tilted/token_count*100:.1f}%)")
    if n_tilted > 0:
        print(f"  Hits:  {n_tilt_hit:,} ({n_tilt_hit/n_tilted*100:.1f}%), avg gain: {sum_gain_hit/max(n_tilt_hit,1):.4f} nats")
        print(f"  Miss:  {n_tilt_miss:,} ({n_tilt_miss/n_tilted*100:.1f}%), avg cost: {sum_cost_miss/max(n_tilt_miss,1):.4f} nats")
        print(f"  Net:   {sum_gain_hit - sum_cost_miss:+.2f} nats ({(sum_gain_hit-sum_cost_miss)/token_count:+.6f} nats/tok)")
    print(f"Tokens scored: {token_count:,}")
    print(f"Bytes scored:  {byte_sum:,.0f}")
    print(f"Phase 1 (PPM): {phase1_time:.1f}s")
    print(f"Phase 2 (GPU): {phase2_time:.1f}s")
    print(f"Total:         {phase1_time + phase2_time:.1f}s")


if __name__ == "__main__":
    main()
