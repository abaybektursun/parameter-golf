"""Attention head analysis: classify 88 heads by function (Olsson et al. 2022).

Monkey-patches flash_attn with standard PyTorch attention to capture per-head
attention weight matrices. Computes induction, previous-token, positional, and
entropy scores for each head.
"""
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

import sys, os
from common import make_args, load_model, load_validation_tokens

DEVICE = "cuda:0"
SEQ_LEN = 2048
NUM_EVAL_SEQS = 32
NUM_LAYERS = 11
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 64

# Global capture buffer — each forward pass appends one [1, H, T, T] tensor per layer
attn_capture = []


def manual_attention(q, k, v, causal=True):
    """Drop-in replacement for flash_attn_3_func that captures attention weights.

    q: [B, T, H, D]     (8 Q heads)
    k: [B, T, Hkv, D]   (4 KV heads)
    v: [B, T, Hkv, D]   (4 KV heads)
    Returns: [B, T, H, D]
    """
    B, T, H, D = q.shape
    Hkv = k.size(2)

    # Expand KV heads for GQA: 4 -> 8
    k_exp = k.repeat_interleave(H // Hkv, dim=2)
    v_exp = v.repeat_interleave(H // Hkv, dim=2)

    # [B, H, T, D] layout for matmul
    q_t = q.transpose(1, 2).float()
    k_t = k_exp.transpose(1, 2).float()
    v_t = v_exp.transpose(1, 2).float()

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(D)

    # Causal mask
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    scores.masked_fill_(mask[None, None], float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)  # [B, H, T, T]
    attn_capture.append(attn_weights.detach())

    y = torch.matmul(attn_weights, v_t)
    return y.transpose(1, 2).to(q.dtype)  # [B, T, H, D]


def compute_induction_pairs(tokens):
    """Find all (query_pos, target_pos) pairs for induction score.

    For position i, if tokens[i-1] appeared earlier at position j (j < i-1),
    the induction target is j+1.
    """
    tok = tokens.numpy()
    # Build position lists per token
    token_positions = {}
    for pos in range(len(tok)):
        tid = tok[pos]
        if tid not in token_positions:
            token_positions[tid] = []
        token_positions[tid].append(pos)

    query_pos = []
    target_pos = []
    for i in range(2, len(tok)):
        prefix_id = tok[i - 1]
        positions = token_positions.get(prefix_id)
        if positions is None:
            continue
        for j in positions:
            if j >= i - 1:
                break
            if j + 1 < i:
                query_pos.append(i)
                target_pos.append(j + 1)
    return np.array(query_pos, dtype=np.int64), np.array(target_pos, dtype=np.int64)


def load_model_fast(args, device):
    """Load model skipping expensive orthogonal init (we overwrite with pretrained weights)."""
    import train_gpt_mlp35_mixed as train_gpt
    from train_gpt_mlp35_mixed import GPT, CastedLinear, restore_low_dim_params_to_fp32
    from common import WEIGHTS_PATH

    # Skip _init_weights entirely — we're loading pretrained weights
    original_init = GPT._init_weights
    GPT._init_weights = lambda self: None

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()

    GPT._init_weights = original_init  # restore

    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    sd = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def main():
    args = make_args()
    device = torch.device(DEVICE)

    # Monkey-patch flash attention BEFORE loading model
    import train_gpt_mlp35_mixed as train_gpt
    train_gpt.flash_attn_3_func = manual_attention

    print("Loading model (skipping init)...", flush=True)
    model = load_model_fast(args, device)
    print("Model loaded!", flush=True)

    val_tokens = load_validation_tokens(args.val_files, SEQ_LEN)

    num_seqs = min((val_tokens.numel() - 1) // SEQ_LEN, NUM_EVAL_SEQS)

    # Accumulators (float64)
    prev_token_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    induction_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    induction_count_total = 0
    bos_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    entropy_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    self_attn_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    offset_sums = np.zeros((5, NUM_LAYERS, NUM_HEADS), dtype=np.float64)

    total_positions = 0  # for averaging prev_token, bos, entropy, self_attn
    total_induction_pairs = 0

    t0 = time.time()
    print(f"Attention head analysis: {num_seqs} sequences of length {SEQ_LEN}")
    print(f"Model: {NUM_LAYERS} layers, {NUM_HEADS} Q heads, {NUM_KV_HEADS} KV heads")

    with torch.inference_mode():
        for seq_idx in range(num_seqs):
            inputs = val_tokens[seq_idx * SEQ_LEN : seq_idx * SEQ_LEN + SEQ_LEN].unsqueeze(0).to(device).long()

            attn_capture.clear()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model.forward_logits(inputs)

            assert len(attn_capture) == NUM_LAYERS, f"Expected {NUM_LAYERS} captures, got {len(attn_capture)}"

            # Precompute induction pairs for this sequence
            seq_tokens = inputs[0].cpu()
            q_pos, t_pos = compute_induction_pairs(seq_tokens)
            total_induction_pairs += len(q_pos)
            T = SEQ_LEN

            for li in range(NUM_LAYERS):
                aw = attn_capture[li][0].cpu().numpy()  # [H, T, T] float32

                # 1. Previous-token score: attn[h, t, t-1] for t=1..T-1
                #    Extract diagonal at offset -1: for each h, sum aw[h, t, t-1]
                for h in range(NUM_HEADS):
                    prev_token_sum[li, h] += np.trace(aw[h], offset=-1)

                # 2. BOS score: attn[h, :, 0]
                bos_sum[li] += aw[:, :, 0].sum(axis=1)  # [H]

                # 3. Entropy: -sum(p * log(p)) per position, averaged
                #    For causal attention, row t has nonzero entries [0..t]
                #    Using full row is ok since masked entries are 0 (softmax output)
                p_safe = np.clip(aw, 1e-10, 1.0)
                ent_per_pos = -(aw * np.log(p_safe)).sum(axis=-1)  # [H, T]
                entropy_sum[li] += ent_per_pos.sum(axis=-1)  # [H]

                # 4. Self-attention: attn[h, t, t]
                diag_idx = np.arange(T)
                self_attn_sum[li] += aw[:, diag_idx, diag_idx].sum(axis=-1)  # [H]

                # 5. Offset scores (1-5)
                for d in range(1, 6):
                    # attn[h, t, t-d] for t=d..T-1 — this is trace at offset -d
                    for h in range(NUM_HEADS):
                        offset_sums[d - 1, li, h] += np.trace(aw[h], offset=-d)

                # 6. Induction score: attn[h, query_pos, target_pos]
                if len(q_pos) > 0:
                    induction_sum[li] += aw[:, q_pos, t_pos].sum(axis=-1)  # [H]

            total_positions += T
            attn_capture.clear()

            elapsed = time.time() - t0
            print(f"  seq {seq_idx + 1}/{num_seqs}, "
                  f"induction_pairs={total_induction_pairs:,}, "
                  f"{elapsed:.1f}s")

    elapsed = time.time() - t0

    # Compute final scores
    # prev_token: summed over (T-1) positions per seq × num_seqs
    n_prev = (SEQ_LEN - 1) * num_seqs
    n_total = SEQ_LEN * num_seqs
    n_offsets = [(SEQ_LEN - d) * num_seqs for d in range(1, 6)]

    prev_score = prev_token_sum / n_prev
    bos_score_arr = bos_sum / n_total
    ent_score = entropy_sum / n_total
    self_score = self_attn_sum / n_total
    off_scores = np.zeros((5, NUM_LAYERS, NUM_HEADS))
    for d in range(5):
        off_scores[d] = offset_sums[d] / n_offsets[d]
    ind_score = np.divide(induction_sum, total_induction_pairs, out=np.zeros_like(induction_sum)) if total_induction_pairs > 0 else np.zeros_like(induction_sum)

    # Classification
    def classify(layer, head):
        if ind_score[layer, head] > 0.02:
            return "induction"
        if prev_score[layer, head] > 0.1:
            return "previous_token"
        if bos_score_arr[layer, head] > 0.1:
            return "positional"
        return "other"

    # Report
    print("\n" + "=" * 120)
    print("ATTENTION HEAD ANALYSIS (Olsson et al. 2022)")
    print("=" * 120)
    print(f"\nTotal tokens analyzed: {num_seqs * SEQ_LEN:,}")
    print(f"Sequences: {num_seqs}, length: {SEQ_LEN}")
    print(f"Repeated bigram instances found: {total_induction_pairs:,}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    header = (f"{'Layer':>5} {'Head':>5} {'PrevTok':>8} {'Induction':>10} {'BOS':>8} "
              f"{'Entropy':>8} {'SelfAttn':>9} "
              + " ".join([f"{'Off'+str(d+1):>6}" for d in range(5)])
              + f" {'Classification':>16}")
    print(f"\n{header}")
    print("-" * 120)

    classifications = {}
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            cls = classify(layer, head)
            classifications[(layer, head)] = cls
            off_str = " ".join([f"{off_scores[d, layer, head]:>6.4f}" for d in range(5)])
            print(f"  L{layer:<3} H{head:<3} {prev_score[layer, head]:>8.4f} "
                  f"{ind_score[layer, head]:>10.4f} "
                  f"{bos_score_arr[layer, head]:>8.4f} {ent_score[layer, head]:>8.4f} "
                  f"{self_score[layer, head]:>9.4f} "
                  f"{off_str} {cls:>16}")

    # Summary
    induction_heads = sorted([(l, h) for (l, h), c in classifications.items() if c == "induction"])
    prev_token_heads = sorted([(l, h) for (l, h), c in classifications.items() if c == "previous_token"])
    positional_heads = sorted([(l, h) for (l, h), c in classifications.items() if c == "positional"])
    other_heads = sorted([(l, h) for (l, h), c in classifications.items() if c == "other"])

    print(f"\nSummary:")
    print(f"  Induction heads ({len(induction_heads)}): {', '.join(f'L{l}H{h}' for l, h in induction_heads)}")
    print(f"  Previous-token heads ({len(prev_token_heads)}): {', '.join(f'L{l}H{h}' for l, h in prev_token_heads)}")
    print(f"  Positional heads ({len(positional_heads)}): {', '.join(f'L{l}H{h}' for l, h in positional_heads)}")
    print(f"  Other/mixed heads ({len(other_heads)}): {', '.join(f'L{l}H{h}' for l, h in other_heads)}")

    enc_layers = range(5)
    dec_layers = range(5, 11)
    enc_ind = sum(1 for l, h in induction_heads if l in enc_layers)
    dec_ind = sum(1 for l, h in induction_heads if l in dec_layers)
    enc_prev = sum(1 for l, h in prev_token_heads if l in enc_layers)
    dec_prev = sum(1 for l, h in prev_token_heads if l in dec_layers)
    print(f"\n  Encoder (L0-4): {enc_ind} induction, {enc_prev} previous-token")
    print(f"  Decoder (L5-10): {dec_ind} induction, {dec_prev} previous-token")

    print("\n" + "=" * 120)
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
