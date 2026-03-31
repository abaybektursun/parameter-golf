"""Token-level interpretability: loss heatmap, top-k predictions, side-by-side generation.

Runs one forward pass over 8 validation sequences to collect per-token loss and top-5
predictions, then generates continuations from real text prompts. Writes JSON directly.
"""
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path

import sys, os
from common import make_args, load_model, load_validation_tokens, TOKENIZER_PATH

DEVICE = "cuda:0"
SEQ_LEN = 2048
NUM_SEQS = 8
PROMPT_LEN = 50
GEN_LEN = 200
TEMPERATURE = 0.8
TOP_K_DISPLAY = 5
LOSS_PERCENTILE = 90

OUT_PATH = Path(__file__).resolve().parent / "results" / "token_interpretability.json"


def decode_token(sp, tid):
    """Convert token ID to display string."""
    if sp.is_byte(tid):
        piece = sp.id_to_piece(tid)       # e.g. "<0x0A>"
        byte_val = int(piece[3:-1], 16)
        if byte_val == 0x0A:
            return "\n"
        if 0x20 <= byte_val < 0x7F:
            return chr(byte_val)
        return piece                       # non-printable: keep <0xHH>
    if sp.is_control(tid) or sp.is_unknown(tid):
        return sp.id_to_piece(tid)
    return sp.id_to_piece(tid).replace("\u2581", " ")


def main():
    args = make_args()
    device = torch.device(DEVICE)
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    model = load_model(args, device)
    val_tokens = load_validation_tokens(args.val_files, SEQ_LEN)

    vocab_texts = [decode_token(sp, tid) for tid in range(args.vocab_size)]

    num_avail = (val_tokens.numel() - 1) // SEQ_LEN
    assert num_avail >= NUM_SEQS, f"Need {NUM_SEQS} sequences but only {num_avail} available"

    t0 = time.time()

    # === Phase 1: Forward pass — per-token loss + top-k predictions ===
    print(f"Phase 1: Forward pass on {NUM_SEQS} x {SEQ_LEN} tokens...")

    inputs_batch = torch.stack([
        val_tokens[i * SEQ_LEN : i * SEQ_LEN + SEQ_LEN]
        for i in range(NUM_SEQS)
    ]).to(device).long()

    targets_batch = torch.stack([
        val_tokens[i * SEQ_LEN + 1 : i * SEQ_LEN + SEQ_LEN + 1]
        for i in range(NUM_SEQS)
    ]).to(device).long()

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward_logits(inputs_batch)

        logits_f = logits.float()
        log_probs = F.log_softmax(logits_f, dim=-1)
        probs = F.softmax(logits_f, dim=-1)

        per_token_loss = -log_probs.gather(-1, targets_batch.unsqueeze(-1)).squeeze(-1)
        top5_probs, top5_ids = probs.topk(TOP_K_DISPLAY, dim=-1)

    per_token_loss_np = per_token_loss.cpu().numpy()
    top5_probs_np = top5_probs.cpu().numpy()
    top5_ids_np = top5_ids.cpu().numpy()
    inputs_np = inputs_batch.cpu().numpy()
    targets_np = targets_batch.cpu().numpy()
    all_losses_flat = per_token_loss_np.flatten()

    loss_threshold = float(np.percentile(all_losses_flat, LOSS_PERCENTILE))
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Loss threshold (P{LOSS_PERCENTILE}): {loss_threshold:.4f} nats")
    print(f"  Mean: {all_losses_flat.mean():.4f}, Median: {np.median(all_losses_flat):.4f}")

    # === Phase 2: Autoregressive generation from real prompts ===
    print(f"\nPhase 2: Generating {GEN_LEN} tokens from {PROMPT_LEN}-token prompts...")
    t1 = time.time()

    prompts = torch.stack([
        val_tokens[i * SEQ_LEN : i * SEQ_LEN + PROMPT_LEN]
        for i in range(NUM_SEQS)
    ]).to(device).long()

    real_cont_np = np.stack([
        val_tokens[i * SEQ_LEN + PROMPT_LEN : i * SEQ_LEN + PROMPT_LEN + GEN_LEN].numpy()
        for i in range(NUM_SEQS)
    ])

    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    tokens = prompts.clone()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for step in range(GEN_LEN):
            logits_gen = model.forward_logits(tokens)
            next_logits = logits_gen[:, -1, :].float()
            next_probs = torch.softmax(next_logits / TEMPERATURE, dim=-1)
            next_tok = torch.multinomial(next_probs, 1, generator=rng)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (step + 1) % 50 == 0:
                print(f"  {step + 1}/{GEN_LEN} tokens generated")

    gen_np = tokens[:, PROMPT_LEN:].cpu().numpy()
    prompts_np = prompts.cpu().numpy()
    print(f"  Done in {time.time() - t1:.1f}s")

    # === Phase 3: Assemble JSON ===
    print("\nPhase 3: Writing JSON...")

    output = {
        "metadata": {
            "num_sequences": NUM_SEQS,
            "seq_len": SEQ_LEN,
            "prompt_len": PROMPT_LEN,
            "gen_len": GEN_LEN,
            "temperature": TEMPERATURE,
            "top_k_display": TOP_K_DISPLAY,
            "loss_percentile": LOSS_PERCENTILE,
            "loss_threshold_nats": round(loss_threshold, 4),
            "mean_loss_nats": round(float(all_losses_flat.mean()), 4),
            "median_loss_nats": round(float(np.median(all_losses_flat)), 4),
        },
        "sequences": [],
    }

    for si in range(NUM_SEQS):
        # Token array: position 0 = first input (no loss), positions 1..SEQ_LEN = targets with loss
        tok_list = []

        # Position 0: context-only (no loss for the first token)
        tok_list.append({"t": vocab_texts[int(inputs_np[si, 0])], "id": int(inputs_np[si, 0])})

        # Positions 1..SEQ_LEN
        for pos in range(SEQ_LEN):
            tid = int(targets_np[si, pos])
            loss = float(per_token_loss_np[si, pos])
            entry = {"t": vocab_texts[tid], "id": tid, "l": round(loss, 4)}

            if loss >= loss_threshold:
                entry["top5"] = [
                    {
                        "t": vocab_texts[int(top5_ids_np[si, pos, k])],
                        "id": int(top5_ids_np[si, pos, k]),
                        "p": round(float(top5_probs_np[si, pos, k]), 4),
                    }
                    for k in range(TOP_K_DISPLAY)
                ]
            tok_list.append(entry)

        # Generation data
        gen_obj = {
            "prompt": [{"t": vocab_texts[int(prompts_np[si, i])], "id": int(prompts_np[si, i])}
                       for i in range(PROMPT_LEN)],
            "real":   [{"t": vocab_texts[int(real_cont_np[si, i])], "id": int(real_cont_np[si, i])}
                       for i in range(GEN_LEN)],
            "model":  [{"t": vocab_texts[int(gen_np[si, i])], "id": int(gen_np[si, i])}
                       for i in range(GEN_LEN)],
        }

        output["sequences"].append({"idx": si, "tokens": tok_list, "gen": gen_obj})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f)

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"\nWrote {OUT_PATH} ({size_kb:.0f} KB)")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
