"""
SelfExtend: extended context via position ID remapping.

The model's RoPE was calibrated for train_seq_len=1024. SelfExtend allows
attending over longer sequences (e.g., 4096, 8192) without RoPE distortion:

- Tokens within group_window (e.g., 1024): exact position IDs
- Tokens beyond group_window: floor(position / group_size) positions
  (approximate but in-distribution)

This gives the model real K,V content from a wider context while keeping
all position embeddings within the trained range.

We monkey-patch the Rotary.forward to remap positions, then run sliding
window eval with larger seq_len.

Usage:
    PYTHONUNBUFFERED=1 python eval_selfextend.py \
        --model final_model.int6.ptz --device cuda:3 \
        --eval-seq-len 4096 --group-size 4 --group-window 1024
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


def selfextend_positions(seq_len, group_window, group_size):
    """Generate remapped position IDs for SelfExtend.

    Returns position tensor of shape (seq_len,) where:
    - Positions 0..group_window-1: exact (0, 1, 2, ...)
    - Positions group_window..seq_len-1: floor(pos / group_size)

    This keeps all position IDs within the trained range while allowing
    the model to attend over a much wider context.
    """
    positions = torch.arange(seq_len, dtype=torch.long)
    # For distant tokens, compress positions
    far_mask = positions >= group_window
    positions[far_mask] = group_window + (positions[far_mask] - group_window) // group_size
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model.int6.ptz")
    parser.add_argument("--val-pattern", default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--eval-seq-len", type=int, default=4096,
                        help="Extended eval sequence length (longer than training)")
    parser.add_argument("--train-seq-len", type=int, default=2048,
                        help="Original training sequence length")
    parser.add_argument("--group-window", type=int, default=1024,
                        help="Positions within this range use exact IDs")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Position compression factor for distant tokens")
    parser.add_argument("--batch-seqs", type=int, default=8)
    parser.add_argument("--device", default="cuda:3")
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
    print("Model loaded.")

    # Monkey-patch Rotary to use SelfExtend position remapping
    remapped_positions = selfextend_positions(
        args.eval_seq_len, args.group_window, args.group_size
    ).to(device)
    print(f"SelfExtend: seq_len={args.eval_seq_len}, group_window={args.group_window}, "
          f"group_size={args.group_size}")
    print(f"Max remapped position: {remapped_positions.max().item()} "
          f"(original max would be {args.eval_seq_len - 1})")

    original_rotary_forward = tg.Rotary.forward

    def selfextend_rotary_forward(self, seq_len, device, dtype):
        """Rotary forward with remapped positions for SelfExtend."""
        rd = self.rope_dims
        inv_freq = self.inv_freq.to(device)

        # Use remapped positions instead of raw 0..seq_len-1
        positions = remapped_positions[:seq_len].float()

        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    # Patch all rotary modules
    for block in model.blocks:
        block.attn.rotary.forward = lambda seq_len, device, dtype, r=block.attn.rotary: selfextend_rotary_forward(r, seq_len, device, dtype)

    # Sliding window eval with extended context
    seq_len = args.eval_seq_len
    stride = args.stride
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    num_windows = len(window_starts)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    t0 = time.perf_counter()
    print(f"Total windows: {num_windows}, stride={stride}, seq_len={seq_len}")

    # Also run baseline (train_seq_len) for comparison
    baseline_seq_len = args.train_seq_len
    baseline_windows = [ws for ws in range(0, total_tokens, stride)
                        if min(ws + baseline_seq_len, total_tokens) - ws >= 1]

    # Run extended eval
    with torch.inference_mode():
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

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)

                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            if bi % (200 * args.batch_seqs) == 0:
                elapsed = time.perf_counter() - t0
                if token_count.item() > 0:
                    curr_loss = (loss_sum / token_count).item()
                    curr_bpb = (curr_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
                    print(f"  window {bi}/{num_windows} | bpb:{curr_bpb:.6f} | {elapsed:.0f}s")

    elapsed = time.perf_counter() - t0
    val_loss = (loss_sum / token_count).item()
    bpb = (val_loss / math.log(2.0)) * (token_count.item() / byte_count.item())

    # Reference: baseline BPB from the training log
    baseline_bpb = 1.11525021  # from our 4xH100 baseline run

    print(f"\n{'='*80}")
    print(f"RESULTS: SelfExtend eval")
    print(f"  eval_seq_len={args.eval_seq_len}, group_window={args.group_window}, "
          f"group_size={args.group_size}")
    print(f"{'='*80}")
    print(f"Baseline (seq_len={baseline_seq_len}):  val_bpb={baseline_bpb:.8f}")
    print(f"SelfExtend (seq_len={seq_len}):         val_loss={val_loss:.8f}  val_bpb={bpb:.8f}")
    print(f"Delta:                                  val_bpb={bpb - baseline_bpb:+.8f}")
    print(f"Eval time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
