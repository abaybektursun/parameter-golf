"""
Sliding window logit averaging (geometric mean in prob space).

Instead of scoring each position from only the max-context window,
average NLLs across ALL overlapping windows that cover each position.

With stride=64 and seq_len=2048, each interior position is covered by
up to 2048/64 = 32 overlapping windows. Averaging reduces variance.

Geometric mean in prob space = arithmetic mean in log-prob space = mean of NLLs.

Storage: 62M positions × 8B (float64) × 2 arrays ≈ 750MB. Trivial.

Usage:
    PYTHONUNBUFFERED=1 python eval_logit_avg.py --model final_model.int6.ptz --device cuda:2
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model.int6.ptz")
    parser.add_argument("--val-pattern", default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-seqs", type=int, default=32)
    parser.add_argument("--device", default="cuda:2")
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

    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gpt", "train_gpt.py")
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)

    # Load validation tokens
    val_files = sorted(glob.glob(args.val_pattern))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    total_tokens = val_tokens.numel() - 1
    print(f"val tokens: {total_tokens}")

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
    compiled_logits = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    print("Model loaded and compiled.")

    # Allocate per-position accumulators on CPU
    nll_sum = np.zeros(total_tokens, dtype=np.float64)
    nll_count = np.zeros(total_tokens, dtype=np.int32)

    seq_len = args.seq_len
    stride = args.stride
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    num_windows = len(window_starts)

    # Also compute standard max-context scoring for comparison
    std_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    std_token_count = torch.zeros((), device=device, dtype=torch.float64)
    std_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    t0 = time.perf_counter()
    print(f"Total windows: {num_windows}, stride={stride}, seq_len={seq_len}")
    print(f"Max coverage per position: {seq_len // stride} windows")

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

            logits = compiled_logits(x_batch)

            # Per-token NLL for ALL positions in each window
            nll_all = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len).cpu().numpy().astype(np.float64)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                # Accumulate NLL for ALL positions in this window
                global_start = ws  # position of first predicted token (y[0] predicts token at ws+1)
                nll_sum[global_start:global_start + wlen] += nll_all[i, :wlen]
                nll_count[global_start:global_start + wlen] += 1

                # Standard scoring (max-context only, same as baseline)
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = torch.tensor(nll_all[i, s:wlen], dtype=torch.float64, device=device)
                std_loss_sum += scored_nll.sum()
                std_token_count += float(wlen - s)

                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                std_byte_count += tb.sum()

            if bi % (500 * args.batch_seqs) == 0:
                elapsed = time.perf_counter() - t0
                covered = np.sum(nll_count > 0)
                pct = bi / num_windows * 100
                print(f"  batch {bi//args.batch_seqs}/{num_windows//args.batch_seqs} ({pct:.1f}%) | "
                      f"covered:{covered}/{total_tokens} | {elapsed:.0f}s")

    elapsed = time.perf_counter() - t0

    # Compute averaged BPB
    covered_mask = nll_count > 0
    avg_nll = np.where(covered_mask, nll_sum / np.maximum(nll_count, 1), 0.0)
    num_covered = int(np.sum(covered_mask))

    # Compute BPB for covered positions
    # Need byte counts for covered positions
    covered_targets = val_tokens[1:].numpy().astype(np.int64)  # target tokens
    covered_prevs = val_tokens[:-1].numpy().astype(np.int64)  # previous tokens

    base_bytes_np = base_bytes_lut.cpu().numpy().astype(np.float64)
    has_leading_space_np = has_leading_space_lut.cpu().numpy()
    is_boundary_np = is_boundary_token_lut.cpu().numpy()

    byte_counts = base_bytes_np[covered_targets]
    byte_counts += (has_leading_space_np[covered_targets] & ~is_boundary_np[covered_prevs]).astype(np.float64)

    avg_loss = np.sum(avg_nll[covered_mask]) / num_covered
    avg_bits = avg_loss / math.log(2.0)
    total_bytes = np.sum(byte_counts[covered_mask])
    tokens_per_byte = num_covered / total_bytes
    avg_bpb = avg_bits * tokens_per_byte

    # Standard (max-context) BPB for comparison
    std_loss = (std_loss_sum / std_token_count).item()
    std_bits = std_loss / math.log(2.0)
    std_tpb = std_token_count.item() / std_byte_count.item()
    std_bpb = std_bits * std_tpb

    # Coverage stats
    max_count = int(np.max(nll_count))
    mean_count = np.mean(nll_count[covered_mask])

    print(f"\n{'='*80}")
    print(f"RESULTS: Sliding Window Logit Averaging")
    print(f"  stride={stride}, seq_len={seq_len}")
    print(f"{'='*80}")
    print(f"Standard (max-context):  val_loss={std_loss:.8f}  val_bpb={std_bpb:.8f}")
    print(f"Averaged (all windows):  val_loss={avg_loss:.8f}  val_bpb={avg_bpb:.8f}")
    print(f"Delta:                   val_loss={avg_loss - std_loss:+.8f}  val_bpb={avg_bpb - std_bpb:+.8f}")
    print(f"Coverage: {num_covered}/{total_tokens} positions ({100*num_covered/total_tokens:.1f}%)")
    print(f"Mean windows/position: {mean_count:.1f}, max: {max_count}")
    print(f"Eval time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
