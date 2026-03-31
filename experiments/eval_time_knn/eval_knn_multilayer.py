"""
Multi-layer kNN-LM: store hidden states from ALL transformer layers.

Instead of using only the final layer's hidden states for kNN retrieval,
concatenate hidden states from all 11 layers into a 5632-dim key (11 × 512).
Different layers capture different abstraction levels — retrieval quality
should improve over single-layer.

With 4×H100 PCIe (81GB each), we have ~320GB free VRAM across 3 GPUs.
Store fits on one GPU: 2M entries × 5632 dim × fp16 = ~21.5GB.

Usage:
    PYTHONUNBUFFERED=1 python eval_knn_multilayer.py \
        --model final_model.int6.ptz --device cuda:1 \
        --store-size 2000000 --k 64 --lam 0.015
"""
from __future__ import annotations
import argparse
import io
import lzma
import math
import time
import glob
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor


class GPUDatastore:
    def __init__(self, max_size, dim, device):
        self.keys = torch.zeros(max_size, dim, dtype=torch.float16, device=device)
        self.vals = torch.zeros(max_size, dtype=torch.int64, device=device)
        self.size = 0
        self.max_size = max_size

    def add(self, hidden, next_tokens):
        n = min(hidden.shape[0], self.max_size - self.size)
        if n <= 0:
            return
        # L2-normalize before storing to keep distances in fp16 range
        normed = F.normalize(hidden[:n].float(), dim=-1)
        self.keys[self.size:self.size + n] = normed.half()
        self.vals[self.size:self.size + n] = next_tokens[:n]
        self.size += n

    def search(self, queries, k=64, T=1.0, vocab_size=1024):
        # Queries are also normalized; distance = 2 - 2*cosine_sim, stays in [0, 4]
        q = F.normalize(queries.float(), dim=-1).half()
        store = self.keys[:self.size]
        # Cosine similarity via dot product (both unit-normed)
        sims = q @ store.T  # (N, store_size), values in [-1, 1]
        _, idx = sims.topk(min(k, self.size), dim=-1)
        weights = F.softmax(sims.gather(1, idx).float() / T, dim=-1)
        probs = torch.zeros(q.shape[0], vocab_size, device=q.device)
        probs.scatter_add_(1, self.vals[idx], weights)
        return probs


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=offset)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def forward_all_layers(model, input_ids):
    """Return (list_of_layer_hiddens, logits).
    Each layer hidden is shape (bsz, seq_len, model_dim).
    Also returns the final hidden (post-norm) and logits."""
    n = model.num_layers
    x = model.tok_emb(input_ids)
    if model.bigram is not None:
        x = x + model.bigram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = model.smear(x)
    x0 = x
    v0 = None
    skips = []
    ve_cache = {}
    layer_outputs = []

    for i in range(model.num_encoder_layers):
        ve = model._get_ve(i, input_ids, ve_cache)
        x, raw_v = model.blocks[i](x, x0,
            model.qo_bank[i], model.kv_bank[i], model.kv_bank[n + i],
            model.qo_bank[n + i], model.mlp_up_bank[i], model.mlp_down_bank[i],
            v_embed=ve, v0=v0)
        if v0 is None and raw_v is not None:
            v0 = raw_v
        skips.append(x)
        layer_outputs.append(x)

    for i in range(model.num_decoder_layers):
        bi = model.num_encoder_layers + i
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        ve = model._get_ve(bi, input_ids, ve_cache)
        x, _ = model.blocks[bi](x, x0,
            model.qo_bank[bi], model.kv_bank[bi], model.kv_bank[n + bi],
            model.qo_bank[n + bi], model.mlp_up_bank[bi], model.mlp_down_bank[bi],
            v_embed=ve, v0=v0)
        layer_outputs.append(x)

    hidden = model.final_norm(x)
    if model.tie_embeddings:
        logits_proj = F.linear(hidden, model.tok_emb.weight)
    else:
        logits_proj = model.lm_head(hidden)
    logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)
    return layer_outputs, hidden, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model.int6.ptz")
    parser.add_argument("--val-pattern", default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-seqs", type=int, default=4)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--lam", type=float, default=0.015)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--store-size", type=int, default=2_000_000)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--layers", default="all",
                        help="Comma-separated layer indices or 'all' for all 11 layers")
    # Model config
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=11)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=float, default=3.0)
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

    # Parse layer selection
    if args.layers == "all":
        selected_layers = list(range(args.num_layers))
    else:
        selected_layers = [int(x) for x in args.layers.split(",")]

    concat_dim = len(selected_layers) * args.model_dim
    print(f"Multi-layer kNN: using layers {selected_layers}")
    print(f"Concatenated key dim: {concat_dim} ({len(selected_layers)} layers × {args.model_dim})")
    store_bytes = args.store_size * concat_dim * 2  # fp16
    print(f"Store memory: {store_bytes / 1e9:.1f} GB")

    # Import model classes
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gpt", "train_gpt.py")
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)

    # Load validation tokens
    val_files = sorted(glob.glob(args.val_pattern))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    usable = ((val_tokens.numel() - 1) // args.seq_len) * args.seq_len
    val_tokens = val_tokens[:usable + 1]
    print(f"val tokens: {val_tokens.numel() - 1}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Build and load model
    model = tg.GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim, xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale=True,
        ve_enabled=True, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for m in model.modules():
        if isinstance(m, tg.CastedLinear):
            m.float()
    tg.restore_low_dim_params_to_fp32(model)

    print(f"Loading model from {args.model}...")
    with open(args.model, "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    unbanked_template = tg._unbank_state_dict(sd_cpu, args.num_layers)
    deq_unbanked = tg.dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_template)
    deq_state = tg._rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    model.load_state_dict(deq_state, strict=True)
    model.eval()
    print("Model loaded.")

    # Sliding window eval with multi-layer kNN
    seq_len = args.seq_len
    stride = args.stride
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]

    store = GPUDatastore(args.store_size, concat_dim, device)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    baseline_loss_sum = torch.zeros((), device=device, dtype=torch.float64)

    t0 = time.perf_counter()
    num_windows = len(window_starts)
    print(f"Total windows: {num_windows}, stride={stride}, seq_len={seq_len}")
    print(f"kNN config: k={args.k}, λ={args.lam}, T={args.temperature}, store={args.store_size}")
    print(f"Layers: {selected_layers}, key_dim={concat_dim}")

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

            layer_outputs, final_hidden, logits = forward_all_layers(model, x_batch)

            # Concatenate selected layers into multi-layer key
            selected = [layer_outputs[l] for l in selected_layers]
            multi_hidden = torch.cat(selected, dim=-1)  # (bsz, seq_len, concat_dim)

            nll_all = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)

                scored_targets = y_batch[i, s:wlen]
                scored_multi = multi_hidden[i, s:wlen].float()
                scored_logits = logits[i, s:wlen].float()

                baseline_nll = nll_all[i, s:wlen].to(torch.float64)
                baseline_loss_sum += baseline_nll.sum()

                model_probs = F.softmax(scored_logits, dim=-1)

                if store.size >= args.k:
                    knn_probs = store.search(scored_multi, k=args.k, T=args.temperature, vocab_size=args.vocab_size)
                    final_probs = (1 - args.lam) * model_probs + args.lam * knn_probs
                else:
                    final_probs = model_probs

                nll = -torch.log(final_probs[torch.arange(len(scored_targets), device=device), scored_targets].clamp(min=1e-12))
                loss_sum += nll.to(torch.float64).sum()
                token_count += float(wlen - s)

                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

                # Add ALL window positions to store AFTER scoring
                store.add(multi_hidden[i, :wlen].float(), y_batch[i, :wlen])

            if (bi // args.batch_seqs) % 500 == 0 and token_count.item() > 0:
                elapsed = time.perf_counter() - t0
                curr_loss = (loss_sum / token_count).item()
                curr_bpb = (curr_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
                base_loss = (baseline_loss_sum / token_count).item()
                base_bpb = (base_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
                print(f"  window {bi}/{num_windows} | store:{store.size}/{store.max_size} | "
                      f"knn_bpb:{curr_bpb:.6f} base_bpb:{base_bpb:.6f} delta:{curr_bpb-base_bpb:.6f} | "
                      f"{elapsed:.0f}s")

    elapsed = time.perf_counter() - t0
    val_loss_knn = (loss_sum / token_count).item()
    val_loss_base = (baseline_loss_sum / token_count).item()
    bpb_knn = (val_loss_knn / math.log(2.0)) * (token_count.item() / byte_count.item())
    bpb_base = (val_loss_base / math.log(2.0)) * (token_count.item() / byte_count.item())

    print(f"\n{'='*80}")
    print(f"RESULTS: Multi-layer kNN (layers={selected_layers})")
    print(f"  k={args.k}, λ={args.lam}, T={args.temperature}, store={args.store_size}")
    print(f"  key_dim={concat_dim}")
    print(f"{'='*80}")
    print(f"Baseline (no kNN):  val_loss={val_loss_base:.8f}  val_bpb={bpb_base:.8f}")
    print(f"kNN-augmented:      val_loss={val_loss_knn:.8f}  val_bpb={bpb_knn:.8f}")
    print(f"Delta:              val_loss={val_loss_knn - val_loss_base:+.8f}  val_bpb={bpb_knn - bpb_base:+.8f}")
    print(f"Store filled: {store.size}/{store.max_size}")
    print(f"Eval time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
