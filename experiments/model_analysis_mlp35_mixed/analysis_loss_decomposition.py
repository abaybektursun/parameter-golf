"""Loss decomposition: break BPB down by token frequency, type, position."""
import math
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

from common import (
    make_args, load_model, load_validation_tokens, build_sentencepiece_luts,
    TOKENIZER_PATH, DATA_PATH,
)

DEVICE = "cuda:0"
SEQ_LEN = 2048
BATCH_SEQS = 16


def main():
    args = make_args()
    device = torch.device(DEVICE)
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    model = load_model(args, device)

    val_tokens = load_validation_tokens(args.val_files, SEQ_LEN)
    num_seqs = (val_tokens.numel() - 1) // SEQ_LEN
    print(f"Val tokens: {val_tokens.numel() - 1}, sequences: {num_seqs}")

    # --- Build per-vocab LUTs ---
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Top-100 frequency mask
    token_counts = torch.zeros(args.vocab_size, dtype=torch.long)
    for t in val_tokens.tolist():
        token_counts[t] += 1
    is_top100 = torch.zeros(args.vocab_size, dtype=torch.bool, device=device)
    is_top100[token_counts.argsort(descending=True)[:100]] = True

    # Token type LUT: 0=word, 1=punct, 2=number, 3=whitespace, 4=other
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

    # --- Accumulators ---
    acc = {k: 0.0 for k in [
        "total_loss", "total_bytes",
        "freq_top100_loss", "freq_top100_bytes",
        "freq_tail_loss", "freq_tail_bytes",
        "pos_first100_loss", "pos_first100_bytes",
        "pos_later_loss", "pos_later_bytes",
    ]}
    type_loss = np.zeros(5, dtype=np.float64)
    type_bytes = np.zeros(5, dtype=np.float64)
    entropy_sum, entropy_count = 0.0, 0

    pos_first100 = torch.zeros(SEQ_LEN, dtype=torch.bool, device=device)
    pos_first100[:100] = True

    # --- Main eval loop ---
    with torch.inference_mode():
        for batch_start in range(0, num_seqs, BATCH_SEQS):
            batch_end = min(batch_start + BATCH_SEQS, num_seqs)
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

            log_probs = F.log_softmax(logits.float(), dim=-1)
            loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
            entropy = -(logits.float().softmax(-1) * log_probs).sum(-1)      # (B, T)

            # Byte counts — matches eval_val: leading space only if prev is not boundary
            byt = base_bytes_lut[targets].float()
            byt += (has_leading_space_lut[targets] & ~is_boundary_token_lut[inputs]).float()
            valid = byt > 0

            # Totals (all tokens in loss, all bytes in denominator)
            acc["total_loss"] += loss.sum().item()
            acc["total_bytes"] += byt.sum().item()

            # Frequency
            top = is_top100[targets]
            acc["freq_top100_loss"] += (loss * top).sum().item()
            acc["freq_top100_bytes"] += (byt * top).sum().item()
            acc["freq_tail_loss"] += (loss * ~top).sum().item()
            acc["freq_tail_bytes"] += (byt * ~top).sum().item()

            # Position
            p = pos_first100[None, :].expand_as(loss)
            acc["pos_first100_loss"] += (loss * p).sum().item()
            acc["pos_first100_bytes"] += (byt * p).sum().item()
            acc["pos_later_loss"] += (loss * ~p).sum().item()
            acc["pos_later_bytes"] += (byt * ~p).sum().item()

            # Token type
            types = type_lut[targets]
            for ti in range(5):
                m = types == ti
                type_loss[ti] += (loss * m).sum().item()
                type_bytes[ti] += (byt * m).sum().item()

            # Entropy (only for tokens that represent bytes)
            entropy_sum += (entropy * valid).sum().item()
            entropy_count += valid.sum().item()

            if (batch_start // BATCH_SEQS) % 50 == 0:
                bpb = acc["total_loss"] / max(acc["total_bytes"], 1) / math.log(2)
                print(f"  batch {batch_start}/{num_seqs}, running BPB: {bpb:.4f}")

    # --- Report ---
    LOG2 = math.log(2)
    print("\n" + "=" * 70)
    print("LOSS DECOMPOSITION REPORT")
    print("=" * 70)
    overall_bpb = acc["total_loss"] / acc["total_bytes"] / LOG2
    print(f"\nOverall BPB: {overall_bpb:.6f} (total_bytes: {acc['total_bytes']:.0f})")
    print(f"Mean prediction entropy: {entropy_sum / max(entropy_count, 1):.4f} nats")

    print(f"\n{'Bucket':<25} {'BPB':>10} {'% Bytes':>10} {'% Loss':>10}")
    print("-" * 55)

    rows = [
        ("freq_top100", acc["freq_top100_loss"], acc["freq_top100_bytes"]),
        ("freq_tail", acc["freq_tail_loss"], acc["freq_tail_bytes"]),
        ("pos_first100", acc["pos_first100_loss"], acc["pos_first100_bytes"]),
        ("pos_later", acc["pos_later_loss"], acc["pos_later_bytes"]),
    ]
    for ti, name in enumerate(TYPE_NAMES):
        if type_bytes[ti] > 0:
            rows.append((f"type_{name}", type_loss[ti], type_bytes[ti]))

    for name, l, b in sorted(rows):
        if b > 0:
            print(f"{name:<25} {l / b / LOG2:>10.4f} {100 * b / acc['total_bytes']:>9.1f}% {100 * l / acc['total_loss']:>9.1f}%")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Compare % Bytes vs % Loss.")
    print("If a bucket has 5% of bytes but 30% of loss, focus there.")
    print("=" * 70)


if __name__ == "__main__":
    main()
