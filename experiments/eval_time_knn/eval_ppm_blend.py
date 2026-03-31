"""
Log-space PPM + neural blending at eval time.

PPMC (Prediction by Partial Matching, method C) builds a trie causally
from already-scored tokens. At each position, PPM produces a distribution
over 1024 tokens. This is blended with the neural model's logits in
log-odds space (not probability space), then converted back.

Log-odds blending: logit_mix = (1-α) * logit_neural + α * logit_ppm
This avoids the catastrophic failure of probability-space blending where
a single P≈0 from PPM kills the mixture.

Usage:
    PYTHONUNBUFFERED=1 python eval_ppm_blend.py \
        --model final_model.int6.ptz --device cuda:0 \
        --alpha 0.01 --max-order 8
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


# ---- PPM Trie ----

class PPMNode:
    """Trie node for PPM. Stores counts of following tokens."""
    __slots__ = ['counts', 'total', 'num_distinct', 'children']

    def __init__(self):
        self.counts = {}      # token -> count
        self.total = 0
        self.num_distinct = 0
        self.children = {}    # token -> PPMNode


class PPMModel:
    """PPMC model with exclusion. Operates on 1024-token BPE alphabet."""

    def __init__(self, max_order=8, vocab_size=1024):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.root = PPMNode()  # order-0 node
        self.context_buf = []  # rolling context buffer

    def update(self, token):
        """Add a token to the model. Call AFTER scoring this position."""
        token = int(token)

        # Update counts at all orders
        node = self.root
        # Order 0: just count token frequencies
        if token not in node.counts:
            node.num_distinct += 1
        node.counts[token] = node.counts.get(token, 0) + 1
        node.total += 1

        # Orders 1..max_order
        ctx = self.context_buf
        for order in range(1, min(self.max_order, len(ctx)) + 1):
            ctx_token = ctx[-order]
            if ctx_token not in node.children:
                node.children[ctx_token] = PPMNode()
            node = node.children[ctx_token]

            if token not in node.counts:
                node.num_distinct += 1
            node.counts[token] = node.counts.get(token, 0) + 1
            node.total += 1

        self.context_buf.append(token)

    def predict(self, device):
        """Produce log-probabilities over vocab_size tokens using PPMC with exclusion.
        Returns tensor of shape (vocab_size,) on the given device."""
        probs = np.zeros(self.vocab_size, dtype=np.float64)
        excluded = set()
        remaining_mass = 1.0

        ctx = self.context_buf

        # Try orders from highest to 0
        for order in range(min(self.max_order, len(ctx)), -1, -1):
            # Navigate to the right node
            node = self.root
            if order > 0:
                found = True
                for i in range(order, 0, -1):
                    ctx_tok = ctx[-i]
                    if ctx_tok not in node.children:
                        found = False
                        break
                    node = node.children[ctx_tok]
                if not found:
                    continue

            if node.total == 0:
                continue

            # Collect counts excluding already-predicted tokens
            order_total = 0
            order_distinct = 0
            order_counts = {}
            for tok, cnt in node.counts.items():
                if tok not in excluded:
                    order_counts[tok] = cnt
                    order_total += cnt
                    order_distinct += 1

            if order_total == 0:
                continue

            # PPMC escape: P(esc) = d / (n + d)
            escape_prob = order_distinct / (order_total + order_distinct)
            order_mass = remaining_mass * (1.0 - escape_prob)

            for tok, cnt in order_counts.items():
                probs[tok] += order_mass * cnt / order_total

            excluded.update(order_counts.keys())
            remaining_mass *= escape_prob

        # Distribute remaining mass uniformly over unseen tokens
        num_unseen = self.vocab_size - len(excluded)
        if num_unseen > 0 and remaining_mass > 0:
            per_unseen = remaining_mass / num_unseen
            for t in range(self.vocab_size):
                if t not in excluded:
                    probs[t] += per_unseen

        # Convert to log-probs, clamp to avoid log(0)
        log_probs = np.log(np.maximum(probs, 1e-30))
        return torch.tensor(log_probs, dtype=torch.float32, device=device)


# ---- Helpers ----

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


def logits_to_logodds(logits):
    """Convert logits to log-odds: log(p / (1-p)) for each class.
    Input: raw logits (unnormalized). Output: log-odds."""
    log_probs = F.log_softmax(logits, dim=-1)
    # log_odds = log(p) - log(1-p) = log_probs - log(1 - exp(log_probs))
    # Use log1p(-exp(log_probs)) for numerical stability
    log_one_minus_p = torch.log1p(-torch.exp(log_probs.clamp(max=-1e-7)))
    return log_probs - log_one_minus_p


def logodds_to_probs(logodds):
    """Convert log-odds back to probabilities: sigmoid then normalize."""
    # sigmoid(logodds) = 1 / (1 + exp(-logodds))
    probs = torch.sigmoid(logodds)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model.int6.ptz")
    parser.add_argument("--val-pattern", default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-seqs", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.01, help="PPM blending weight in log-odds space")
    parser.add_argument("--max-order", type=int, default=8, help="Maximum PPM context order")
    parser.add_argument("--max-tokens", type=int, default=0, help="Limit eval to first N tokens (0=all)")
    parser.add_argument("--device", default="cuda:0")
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

    # Load val tokens
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

    # Load model
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
    print("Model loaded.")

    # Initialize PPM
    ppm = PPMModel(max_order=args.max_order, vocab_size=args.vocab_size)
    alpha = args.alpha
    print(f"PPM config: max_order={args.max_order}, α={alpha}")

    # Phase 1: Precompute neural logits for all scored positions via sliding window
    # Store as per-position logit vectors (only for scored positions, in order)
    seq_len = args.seq_len
    stride = args.stride
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    num_windows = len(window_starts)

    # Build map: global_position -> (neural_logits, target_token, prev_token)
    # Process windows in order, storing scored positions
    print(f"Phase 1: Precomputing neural logits ({num_windows} windows, stride={stride})...")
    t_phase1 = time.perf_counter()

    # Scored positions in order
    scored_positions = []  # (global_pos, neural_logits_row_idx)
    all_neural_logits = []  # list of tensors, each (num_scored_in_batch, vocab_size)
    all_targets = []
    all_prevs = []

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
                chunk = val_tokens_full[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            logits = model.forward_logits(x_batch)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                # Store logits for scored positions
                scored_logits = logits[i, s:wlen].float().cpu()
                scored_targets_batch = y_batch[i, s:wlen].cpu()
                scored_prevs_batch = x_batch[i, s:wlen].cpu()
                for j in range(wlen - s):
                    global_pos = ws + s + j
                    scored_positions.append(global_pos)
                all_neural_logits.append(scored_logits)
                all_targets.append(scored_targets_batch)
                all_prevs.append(scored_prevs_batch)

            if bi % (100 * args.batch_seqs) == 0:
                print(f"  phase1 batch {bi//args.batch_seqs}/{num_windows//args.batch_seqs} | {time.perf_counter()-t_phase1:.1f}s")

    # Concatenate
    all_neural_logits = torch.cat(all_neural_logits, dim=0)  # (total_scored, vocab)
    all_targets = torch.cat(all_targets, dim=0)               # (total_scored,)
    all_prevs = torch.cat(all_prevs, dim=0)

    num_scored = len(scored_positions)
    phase1_time = time.perf_counter() - t_phase1
    print(f"Phase 1 done: {num_scored} scored positions in {phase1_time:.1f}s")
    print(f"Neural logits stored: {all_neural_logits.shape} ({all_neural_logits.numel() * 4 / 1e9:.1f} GB)")

    # Phase 2: Sequential PPM blending
    print(f"Phase 2: Sequential PPM blending (α={alpha})...")
    t_phase2 = time.perf_counter()

    baseline_loss_sum = 0.0
    blend_loss_sum = 0.0
    ppm_loss_sum = 0.0
    byte_sum = 0.0
    token_count = 0

    # Feed the very first token into PPM context (token[0] is the first input token)
    first_input_token = int(val_tokens_full[0].item())
    ppm.context_buf.append(first_input_token)

    base_bytes_np = base_bytes_lut.cpu().numpy()
    has_ls_np = has_leading_space_lut.cpu().numpy()
    is_bound_np = is_boundary_token_lut.cpu().numpy()

    for idx in range(num_scored):
        neural_logits = all_neural_logits[idx]  # (vocab_size,)
        target = int(all_targets[idx].item())
        prev = int(all_prevs[idx].item())

        # Baseline neural NLL
        neural_log_probs = F.log_softmax(neural_logits, dim=-1)
        baseline_nll = -neural_log_probs[target].item()

        # PPM prediction
        ppm_log_probs = ppm.predict(torch.device('cpu'))  # (vocab_size,)

        # PPM-only NLL
        ppm_nll = -ppm_log_probs[target].item()

        # Log-odds blending
        neural_logodds = logits_to_logodds(neural_logits.unsqueeze(0)).squeeze(0)
        ppm_logodds = logits_to_logodds(ppm_log_probs.exp().unsqueeze(0)).squeeze(0)

        blended_logodds = (1.0 - alpha) * neural_logodds + alpha * ppm_logodds
        blended_probs = logodds_to_probs(blended_logodds.unsqueeze(0)).squeeze(0)
        blend_nll = -torch.log(blended_probs[target].clamp(min=1e-30)).item()

        # BPB bytes
        tb = float(base_bytes_np[target])
        if has_ls_np[target] and not is_bound_np[prev]:
            tb += 1.0

        baseline_loss_sum += baseline_nll
        blend_loss_sum += blend_nll
        ppm_loss_sum += ppm_nll
        byte_sum += tb
        token_count += 1

        # Update PPM with this token (strict causality: update AFTER scoring)
        ppm.update(target)

        if idx % 100000 == 0 and idx > 0:
            elapsed = time.perf_counter() - t_phase2
            bl = baseline_loss_sum / token_count
            bn = blend_loss_sum / token_count
            pn = ppm_loss_sum / token_count
            tpb = token_count / byte_sum
            base_bpb = (bl / math.log(2.0)) * tpb
            blend_bpb = (bn / math.log(2.0)) * tpb
            ppm_bpb = (pn / math.log(2.0)) * tpb
            print(f"  pos {idx}/{num_scored} | base_bpb:{base_bpb:.6f} blend_bpb:{blend_bpb:.6f} "
                  f"ppm_bpb:{ppm_bpb:.4f} delta:{blend_bpb-base_bpb:+.6f} | "
                  f"ppm_nodes:{len(ppm.context_buf)} | {elapsed:.0f}s "
                  f"({idx/elapsed:.0f} tok/s)")

    phase2_time = time.perf_counter() - t_phase2

    # Final results
    bl = baseline_loss_sum / token_count
    bn = blend_loss_sum / token_count
    pn = ppm_loss_sum / token_count
    tpb = token_count / byte_sum
    base_bpb = (bl / math.log(2.0)) * tpb
    blend_bpb = (bn / math.log(2.0)) * tpb
    ppm_bpb = (pn / math.log(2.0)) * tpb

    print(f"\n{'='*80}")
    print(f"RESULTS: Log-space PPM + Neural Blend")
    print(f"  α={alpha}, max_order={args.max_order}, stride={stride}")
    print(f"{'='*80}")
    print(f"Neural only:     val_bpb={base_bpb:.8f}")
    print(f"PPM only:        val_bpb={ppm_bpb:.8f}")
    print(f"Blended:         val_bpb={blend_bpb:.8f}")
    print(f"Delta (blend-base): {blend_bpb - base_bpb:+.8f} BPB")
    print(f"Tokens scored: {token_count}")
    print(f"Phase 1 (neural): {phase1_time:.1f}s")
    print(f"Phase 2 (PPM blend): {phase2_time:.1f}s")
    print(f"PPM throughput: {token_count/phase2_time:.0f} tokens/s")
    print(f"Total: {phase1_time + phase2_time:.1f}s")


if __name__ == "__main__":
    main()
