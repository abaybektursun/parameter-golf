"""Token-level confidence calibration analysis.

For each token prediction, compare model's predicted probability to actual outcome.
Produces calibration curves, ECE, and loss-by-confidence breakdown.
"""
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

import sys, os
from common import (
    make_args, load_model, load_validation_tokens, build_sentencepiece_luts,
    TOKENIZER_PATH,
)

DEVICE = "cuda:0"
SEQ_LEN = 2048
BATCH_SEQS = 32
NUM_BINS = 20


def main():
    args = make_args()
    device = torch.device(DEVICE)
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    model = load_model(args, device)
    val_tokens = load_validation_tokens(args.val_files, SEQ_LEN)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Token type LUT (reuse from loss decomp): 0=word, 1=punct, 2=number, 3=whitespace, 4=other
    TYPE_NAMES = ["word", "punct", "number", "whitespace", "other"]
    type_lut = torch.zeros(args.vocab_size, dtype=torch.long, device=device)
    for tid in range(args.vocab_size):
        if sp.is_byte(tid) or sp.is_control(tid) or sp.is_unknown(tid):
            type_lut[tid] = 4
            continue
        piece = sp.id_to_piece(tid)
        text = piece.replace("\u2581", "")
        if not text:
            type_lut[tid] = 3
        elif text.isdigit() or all(c.isdigit() or c in ".,+-" for c in text):
            type_lut[tid] = 2
        elif all(c in ".,;:!?\"'()[]{}<>/-_@#$%^&*+=|\\~`" for c in text):
            type_lut[tid] = 1
        elif text.isalpha() or text.replace("'", "").isalpha():
            type_lut[tid] = 0
        else:
            type_lut[tid] = 4

    num_seqs = (val_tokens.numel() - 1) // SEQ_LEN

    # Accumulators for calibration bins (binned by max predicted probability)
    bin_edges = torch.linspace(0, 1, NUM_BINS + 1)
    bin_correct = np.zeros(NUM_BINS, dtype=np.float64)    # sum of top-1 correct
    bin_confidence = np.zeros(NUM_BINS, dtype=np.float64)  # sum of max probabilities
    bin_loss = np.zeros(NUM_BINS, dtype=np.float64)        # sum of NLL
    bin_count = np.zeros(NUM_BINS, dtype=np.int64)         # count

    # Accumulators for bins by P(correct token)
    pcorr_bin_loss = np.zeros(NUM_BINS, dtype=np.float64)
    pcorr_bin_sum = np.zeros(NUM_BINS, dtype=np.float64)   # sum of p_correct
    pcorr_bin_count = np.zeros(NUM_BINS, dtype=np.int64)

    # Per-type calibration (5 types × NUM_BINS)
    type_bin_correct = np.zeros((5, NUM_BINS), dtype=np.float64)
    type_bin_confidence = np.zeros((5, NUM_BINS), dtype=np.float64)
    type_bin_count = np.zeros((5, NUM_BINS), dtype=np.int64)

    # Entropy and rank stats
    entropy_sum = 0.0
    rank_counts = np.zeros(20, dtype=np.int64)  # rank of correct token: 1,2,...,20+

    total_tokens = 0
    total_correct = 0
    total_loss = 0.0
    total_bytes = 0.0

    t0 = time.time()
    print(f"Confidence calibration on {num_seqs} sequences of length {SEQ_LEN}")

    with torch.inference_mode():
        for batch_start in range(0, num_seqs, BATCH_SEQS):
            batch_end = min(batch_start + BATCH_SEQS, num_seqs)
            bs = batch_end - batch_start
            inputs = torch.stack([
                val_tokens[i * SEQ_LEN : i * SEQ_LEN + SEQ_LEN]
                for i in range(batch_start, batch_end)
            ]).to(device).long()
            targets = torch.stack([
                val_tokens[i * SEQ_LEN + 1 : i * SEQ_LEN + SEQ_LEN + 1]
                for i in range(batch_start, batch_end)
            ]).to(device).long()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(inputs)

            # Compute in float32 for precision
            logits_f = logits.float()
            probs = F.softmax(logits_f, dim=-1)                           # (B, T, V)
            log_probs = F.log_softmax(logits_f, dim=-1)                   # (B, T, V)

            # Per-token metrics
            p_correct = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
            nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # (B, T)
            p_max, pred = probs.max(dim=-1)                                  # (B, T)
            top1_correct = (pred == targets).float()                         # (B, T)
            entropy = -(probs * log_probs).sum(dim=-1)                       # (B, T)

            # Rank of correct token
            ranks = (probs > p_correct.unsqueeze(-1)).sum(dim=-1) + 1        # (B, T) — 1-indexed

            # Byte counts for BPB
            byt = base_bytes_lut[targets].float()
            byt += (has_leading_space_lut[targets] & ~is_boundary_token_lut[inputs]).float()

            # Move to CPU for binning
            p_max_np = p_max.cpu().numpy().ravel()
            p_correct_np = p_correct.cpu().numpy().ravel()
            correct_np = top1_correct.cpu().numpy().ravel()
            nll_np = nll.cpu().numpy().ravel()
            entropy_np = entropy.cpu().numpy().ravel()
            ranks_np = ranks.cpu().numpy().ravel()
            types_np = type_lut[targets].cpu().numpy().ravel()

            n = bs * SEQ_LEN

            # Bin by max predicted probability (confidence)
            conf_bins = np.digitize(p_max_np, bin_edges.numpy()) - 1
            conf_bins = np.clip(conf_bins, 0, NUM_BINS - 1)
            for b in range(NUM_BINS):
                mask = conf_bins == b
                bin_correct[b] += correct_np[mask].sum()
                bin_confidence[b] += p_max_np[mask].sum()
                bin_loss[b] += nll_np[mask].sum()
                bin_count[b] += mask.sum()

            # Bin by P(correct token)
            pcorr_bins = np.digitize(p_correct_np, bin_edges.numpy()) - 1
            pcorr_bins = np.clip(pcorr_bins, 0, NUM_BINS - 1)
            for b in range(NUM_BINS):
                mask = pcorr_bins == b
                pcorr_bin_loss[b] += nll_np[mask].sum()
                pcorr_bin_sum[b] += p_correct_np[mask].sum()
                pcorr_bin_count[b] += mask.sum()

            # Per-type calibration bins
            for ti in range(5):
                tmask = types_np == ti
                if not tmask.any():
                    continue
                cb = conf_bins[tmask]
                for b in range(NUM_BINS):
                    bmask = cb == b
                    type_bin_correct[ti, b] += correct_np[tmask][bmask].sum()
                    type_bin_confidence[ti, b] += p_max_np[tmask][bmask].sum()
                    type_bin_count[ti, b] += bmask.sum()

            # Rank histogram (last bin is rank >= 20)
            for r in range(19):
                rank_counts[r] += (ranks_np == r + 1).sum()
            rank_counts[19] += (ranks_np >= 20).sum()

            entropy_sum += entropy_np.sum()
            total_tokens += n
            total_correct += correct_np.sum()
            total_loss += nll_np.sum()
            total_bytes += byt.sum().item()

            if (batch_start // BATCH_SEQS) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  batch {batch_start}/{num_seqs}, "
                      f"acc={total_correct/total_tokens:.4f}, "
                      f"BPB={total_loss/max(total_bytes,1)/math.log(2):.4f}, "
                      f"{elapsed:.0f}s")

    elapsed = time.time() - t0
    overall_acc = total_correct / total_tokens
    overall_bpb = total_loss / total_bytes / math.log(2)
    mean_entropy = entropy_sum / total_tokens

    # --- Compute ECE ---
    ece = 0.0
    for b in range(NUM_BINS):
        if bin_count[b] == 0:
            continue
        acc_b = bin_correct[b] / bin_count[b]
        conf_b = bin_confidence[b] / bin_count[b]
        ece += (bin_count[b] / total_tokens) * abs(acc_b - conf_b)

    # --- Report ---
    print("\n" + "=" * 80)
    print("TOKEN-LEVEL CONFIDENCE CALIBRATION REPORT")
    print("=" * 80)
    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Overall top-1 accuracy: {overall_acc:.4%}")
    print(f"Overall BPB: {overall_bpb:.6f}")
    print(f"Mean prediction entropy: {mean_entropy:.4f} nats ({mean_entropy/math.log(2):.4f} bits)")
    print(f"Expected Calibration Error (ECE): {ece:.6f}")
    print(f"Time: {elapsed:.0f}s")

    # Calibration table (binned by confidence = max predicted probability)
    print(f"\n{'Bin':>12} {'Count':>12} {'% Tokens':>10} {'Avg Conf':>10} {'Accuracy':>10} {'Gap':>10} {'Avg Loss':>10} {'% Loss':>10}")
    print("-" * 94)
    for b in range(NUM_BINS):
        if bin_count[b] == 0:
            continue
        lo, hi = bin_edges[b].item(), bin_edges[b+1].item()
        acc_b = bin_correct[b] / bin_count[b]
        conf_b = bin_confidence[b] / bin_count[b]
        avg_loss = bin_loss[b] / bin_count[b]
        pct_tokens = 100 * bin_count[b] / total_tokens
        pct_loss = 100 * bin_loss[b] / total_loss
        gap = acc_b - conf_b
        print(f"  [{lo:.2f},{hi:.2f}) {bin_count[b]:>12,} {pct_tokens:>9.2f}% {conf_b:>10.4f} {acc_b:>10.4f} {gap:>+10.4f} {avg_loss:>10.4f} {pct_loss:>9.2f}%")

    # Loss by P(correct) table
    print(f"\n{'P(correct)':>12} {'Count':>12} {'% Tokens':>10} {'Avg P(corr)':>12} {'Avg Loss':>10} {'% Loss':>10}")
    print("-" * 76)
    for b in range(NUM_BINS):
        if pcorr_bin_count[b] == 0:
            continue
        lo, hi = bin_edges[b].item(), bin_edges[b+1].item()
        avg_p = pcorr_bin_sum[b] / pcorr_bin_count[b]
        avg_loss = pcorr_bin_loss[b] / pcorr_bin_count[b]
        pct_tokens = 100 * pcorr_bin_count[b] / total_tokens
        pct_loss = 100 * pcorr_bin_loss[b] / total_loss
        print(f"  [{lo:.2f},{hi:.2f}) {pcorr_bin_count[b]:>12,} {pct_tokens:>9.2f}% {avg_p:>12.4f} {avg_loss:>10.4f} {pct_loss:>9.2f}%")

    # Rank histogram
    print(f"\n{'Rank':>6} {'Count':>14} {'% Tokens':>10} {'Cumulative':>10}")
    print("-" * 44)
    cum = 0.0
    for r in range(20):
        pct = 100 * rank_counts[r] / total_tokens
        cum += pct
        label = f"  {r+1}" if r < 19 else "  20+"
        print(f"{label:>6} {rank_counts[r]:>14,} {pct:>9.2f}% {cum:>9.2f}%")

    # Per-type ECE
    print(f"\n{'Type':<12} {'ECE':>10} {'Accuracy':>10}")
    print("-" * 34)
    for ti, tname in enumerate(TYPE_NAMES):
        type_ece = 0.0
        type_total = type_bin_count[ti].sum()
        type_correct_total = type_bin_correct[ti].sum()
        if type_total == 0:
            continue
        for b in range(NUM_BINS):
            if type_bin_count[ti, b] == 0:
                continue
            acc_b = type_bin_correct[ti, b] / type_bin_count[ti, b]
            conf_b = type_bin_confidence[ti, b] / type_bin_count[ti, b]
            type_ece += (type_bin_count[ti, b] / type_total) * abs(acc_b - conf_b)
        type_acc = type_correct_total / type_total
        print(f"{tname:<12} {type_ece:>10.6f} {type_acc:>10.4%}")

    # Over/underconfidence summary
    overconf_loss = 0.0
    underconf_loss = 0.0
    overconf_count = 0
    underconf_count = 0
    for b in range(NUM_BINS):
        if bin_count[b] == 0:
            continue
        acc_b = bin_correct[b] / bin_count[b]
        conf_b = bin_confidence[b] / bin_count[b]
        if conf_b > acc_b:
            overconf_loss += bin_loss[b]
            overconf_count += bin_count[b]
        else:
            underconf_loss += bin_loss[b]
            underconf_count += bin_count[b]

    print(f"\nOverconfident bins (confidence > accuracy):")
    print(f"  Tokens: {overconf_count:,} ({100*overconf_count/total_tokens:.1f}%)")
    print(f"  Loss:   {overconf_loss:.0f} nats ({100*overconf_loss/total_loss:.1f}%)")
    print(f"Underconfident bins (confidence ≤ accuracy):")
    print(f"  Tokens: {underconf_count:,} ({100*underconf_count/total_tokens:.1f}%)")
    print(f"  Loss:   {underconf_loss:.0f} nats ({100*underconf_loss/total_loss:.1f}%)")

    print("\n" + "=" * 80)
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
