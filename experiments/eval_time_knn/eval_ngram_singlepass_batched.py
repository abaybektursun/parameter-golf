"""
Single-pass n-gram + neural log-linear blend — BATCHED neural forward pass.

Legal single-pass: neural forward passes are batched (32 windows at a time),
then scored positions are blended with n-gram LEFT-TO-RIGHT sequentially.
This is legal because neural logits don't depend on n-gram state.

Usage:
    PYTHONUNBUFFERED=1 python eval_ngram_singlepass_batched.py \
        --model final_model.int6.ptz --device cuda:0 \
        --alpha 0.02 --max-order 6
"""
from __future__ import annotations
import argparse
import io
import lzma
import math
import time
import glob
from pathlib import Path
from collections import defaultdict

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
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=num_tokens, offset=offset).astype(np.uint16, copy=False))


class FastNGram:
    __slots__ = ['max_order', 'vocab_size', 'counts', 'totals', 'context_buf', 'lambdas']

    def __init__(self, max_order=6, vocab_size=1024):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.counts = [defaultdict(dict) for _ in range(max_order + 1)]
        self.totals = [defaultdict(int) for _ in range(max_order + 1)]
        self.context_buf = []
        raw = [1.0] + [2.0 ** i for i in range(max_order)]
        s = sum(raw)
        self.lambdas = [w / s for w in raw]

    def _context_hash(self, order):
        if order == 0:
            return 0
        buf = self.context_buf
        n = len(buf)
        if order > n:
            return None
        h = 2166136261
        for i in range(order):
            h = ((h ^ buf[n - order + i]) * 16777619) & 0xFFFFFFFF
        return h

    def update(self, token):
        token = int(token)
        for order in range(min(self.max_order, len(self.context_buf)) + 1):
            ch = self._context_hash(order)
            if ch is None:
                continue
            inner = self.counts[order][ch]
            inner[token] = inner.get(token, 0) + 1
            self.totals[order][ch] += 1
        self.context_buf.append(token)

    def predict_log_probs(self):
        probs = np.full(self.vocab_size, self.lambdas[0] / self.vocab_size, dtype=np.float64)
        for order in range(1, min(self.max_order, len(self.context_buf)) + 1):
            ch = self._context_hash(order)
            if ch is None:
                continue
            total = self.totals[order].get(ch, 0)
            if total == 0:
                probs += self.lambdas[order] / self.vocab_size
                continue
            lam = self.lambdas[order]
            inner = self.counts[order].get(ch, {})
            for tok, c in inner.items():
                probs[tok] += lam * c / total
        probs = np.maximum(probs, 1e-30)
        probs /= probs.sum()
        return np.log(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model.int6.ptz")
    parser.add_argument("--val-pattern", default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-seqs", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
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

    val_files = sorted(glob.glob(args.val_pattern))
    val_tokens_full = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    if args.max_tokens > 0:
        val_tokens_full = val_tokens_full[:args.max_tokens + 1]
    total_tokens = val_tokens_full.numel() - 1
    print(f"val tokens: {total_tokens}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    base_bytes_np = base_bytes_lut.cpu().numpy()
    has_ls_np = has_leading_space_lut.cpu().numpy()
    is_bound_np = is_boundary_token_lut.cpu().numpy()

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

    print(f"Loading {args.model}...")
    with open(args.model, "rb") as f:
        quant_state = torch.load(io.BytesIO(lzma.decompress(f.read())), map_location="cpu")
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    unbanked = tg._unbank_state_dict(sd_cpu, args.num_layers)
    deq = tg.dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked)
    model.load_state_dict(tg._rebank_state_dict(deq, args.num_layers, sd_cpu), strict=True)
    model.eval()
    compiled_logits = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    print("Model loaded and compiled.")

    ngram = FastNGram(max_order=args.max_order, vocab_size=args.vocab_size)
    alpha = args.alpha
    print(f"N-gram blend (single-pass, batched neural, log-linear)")
    print(f"  max_order={args.max_order}, α={alpha}, batch_seqs={args.batch_seqs}")

    seq_len = args.seq_len
    stride = args.stride
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    num_windows = len(window_starts)

    baseline_loss_sum = 0.0
    blend_loss_sum = 0.0
    byte_sum = 0.0
    token_count = 0

    first_input = int(val_tokens_full[0].item())
    ngram.context_buf.append(first_input)

    t0 = time.perf_counter()
    print(f"Eval: {num_windows} windows, stride={stride}, seq_len={seq_len}")

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for bi in range(0, num_windows, args.batch_seqs):
            batch_ws = window_starts[bi:bi + args.batch_seqs]
            bsz = len(batch_ws)

            # Batched neural forward pass
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens_full[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            logits = compiled_logits(x_batch)
            neural_log_probs = F.log_softmax(logits.float(), dim=-1)

            # Sequential n-gram blending for scored positions (left-to-right)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)

                # Get neural log-probs for scored positions
                nlp = neural_log_probs[i, s:wlen].cpu().numpy()  # (scored, vocab)
                targets = y_batch[i, s:wlen].cpu().numpy().astype(np.int64)
                prevs = x_batch[i, s:wlen].cpu().numpy().astype(np.int64)

                for j in range(len(targets)):
                    target = int(targets[j])
                    prev = int(prevs[j])

                    baseline_nll = -float(nlp[j, target])

                    # N-gram log-probs
                    ngram_lp = ngram.predict_log_probs()

                    # Log-linear blend
                    blended_lp = (1.0 - alpha) * nlp[j] + alpha * ngram_lp
                    blended_lp -= blended_lp.max()
                    blended_p = np.exp(blended_lp)
                    blended_p /= blended_p.sum()
                    blend_nll = -math.log(max(blended_p[target], 1e-30))

                    tb = float(base_bytes_np[target])
                    if has_ls_np[target] and not is_bound_np[prev]:
                        tb += 1.0

                    baseline_loss_sum += baseline_nll
                    blend_loss_sum += blend_nll
                    byte_sum += tb
                    token_count += 1

                    ngram.update(target)

            if bi % (500 * args.batch_seqs) == 0 and token_count > 0:
                elapsed = time.perf_counter() - t0
                tpb = token_count / byte_sum
                bl = (baseline_loss_sum / token_count / math.log(2.0)) * tpb
                bn = (blend_loss_sum / token_count / math.log(2.0)) * tpb
                print(f"  win {bi}/{num_windows} | base:{bl:.6f} blend:{bn:.6f} "
                      f"delta:{bn-bl:+.7f} | {token_count} tok | {elapsed:.0f}s ({token_count/elapsed:.0f} tok/s)")

    elapsed = time.perf_counter() - t0
    tpb = token_count / byte_sum
    base_bpb = (baseline_loss_sum / token_count / math.log(2.0)) * tpb
    blend_bpb = (blend_loss_sum / token_count / math.log(2.0)) * tpb

    print(f"\n{'='*80}")
    print(f"RESULTS: Single-Pass Batched N-gram + Neural Log-Linear Blend")
    print(f"  α={alpha}, max_order={args.max_order}, stride={stride}")
    print(f"{'='*80}")
    print(f"Neural only:        val_bpb={base_bpb:.8f}")
    print(f"Blended:            val_bpb={blend_bpb:.8f}")
    print(f"Delta (blend-base): {blend_bpb - base_bpb:+.8f} BPB")
    print(f"Tokens scored: {token_count}")
    print(f"Total time: {elapsed:.1f}s ({token_count/elapsed:.0f} tokens/s)")


if __name__ == "__main__":
    main()
