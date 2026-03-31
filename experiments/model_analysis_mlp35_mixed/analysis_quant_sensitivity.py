"""Quantization sensitivity analysis: quantize one matrix at a time to int6, measure BPB delta."""
import math
import time
import torch
import torch.nn.functional as F
import sentencepiece as spm

from common import (
    make_args, load_model, load_validation_tokens, build_sentencepiece_luts,
    TOKENIZER_PATH,
)

DEVICE = "cuda:0"
SEQ_LEN = 2048
BATCH_SEQS = 32  # larger batch for speed on H100


def quantize_dequantize_int6(weight, clip_range=31):
    """Quantize weight to int6 and dequantize back, returning the noisy weight."""
    t32 = weight.float()
    best_recon, best_err = None, float("inf")
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
        recon = q.float() * s.float()[:, None]
        err = (t32 - recon).pow(2).mean().item()
        if err < best_err:
            best_recon, best_err = recon, err
    return best_recon.to(weight.dtype)


def eval_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """Evaluate BPB on the full validation set, matching eval_val byte counting."""
    num_seqs = (val_tokens.numel() - 1) // SEQ_LEN
    total_loss = 0.0
    total_bytes = 0.0

    with torch.inference_mode():
        for batch_start in range(0, num_seqs, BATCH_SEQS):
            batch_end = min(batch_start + BATCH_SEQS, num_seqs)
            inputs = torch.stack([
                val_tokens[i * SEQ_LEN : i * SEQ_LEN + SEQ_LEN]
                for i in range(batch_start, batch_end)
            ]).to(DEVICE).long()
            targets = torch.stack([
                val_tokens[i * SEQ_LEN + 1 : i * SEQ_LEN + SEQ_LEN + 1]
                for i in range(batch_start, batch_end)
            ]).to(DEVICE).long()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(inputs)

            log_probs = F.log_softmax(logits.float(), dim=-1)
            loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            byt = base_bytes_lut[targets].float()
            byt += (has_leading_space_lut[targets] & ~is_boundary_token_lut[inputs]).float()

            total_loss += loss.sum().item()
            total_bytes += byt.sum().item()

    return total_loss / total_bytes / math.log(2)


MATRIX_TYPES = ["Q", "K", "V", "Out", "MLP_up", "MLP_down"]


def get_matrix(model, n, layer, mtype):
    if mtype == "Q":        return model.qo_bank.data[layer]
    if mtype == "Out":      return model.qo_bank.data[n + layer]
    if mtype == "K":        return model.kv_bank.data[layer]
    if mtype == "V":        return model.kv_bank.data[n + layer]
    if mtype == "MLP_up":   return model.mlp_up_bank.data[layer]
    if mtype == "MLP_down": return model.mlp_down_bank.data[layer]


def set_matrix(model, n, layer, mtype, value):
    if mtype == "Q":        model.qo_bank.data[layer] = value
    elif mtype == "Out":    model.qo_bank.data[n + layer] = value
    elif mtype == "K":      model.kv_bank.data[layer] = value
    elif mtype == "V":      model.kv_bank.data[n + layer] = value
    elif mtype == "MLP_up": model.mlp_up_bank.data[layer] = value
    elif mtype == "MLP_down": model.mlp_down_bank.data[layer] = value


def main():
    args = make_args()
    device = torch.device(DEVICE)
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    model = load_model(args, device)
    val_tokens = load_validation_tokens(args.val_files, SEQ_LEN)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    n = args.num_layers  # 11

    # Save originals
    orig_qo = model.qo_bank.data.clone()
    orig_kv = model.kv_bank.data.clone()
    orig_mlp_up = model.mlp_up_bank.data.clone()
    orig_mlp_down = model.mlp_down_bank.data.clone()

    def restore_all():
        model.qo_bank.data.copy_(orig_qo)
        model.kv_bank.data.copy_(orig_kv)
        model.mlp_up_bank.data.copy_(orig_mlp_up)
        model.mlp_down_bank.data.copy_(orig_mlp_down)

    # --- Baseline ---
    t0 = time.time()
    print("Evaluating baseline (full precision bfloat16)...")
    baseline_bpb = eval_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"Baseline BPB: {baseline_bpb:.6f}  ({time.time() - t0:.1f}s)")

    # --- Per-matrix sensitivity (66 evals) ---
    print("\n--- Per-matrix sensitivity (one matrix quantized at a time) ---")
    results = []
    for layer in range(n):
        for mtype in MATRIX_TYPES:
            orig_w = get_matrix(model, n, layer, mtype).clone()
            quant_w = quantize_dequantize_int6(orig_w)
            set_matrix(model, n, layer, mtype, quant_w)

            bpb = eval_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            delta = bpb - baseline_bpb
            results.append((layer, mtype, bpb, delta))
            print(f"  L{layer:2d} {mtype:<10s} BPB={bpb:.6f}  Δ={delta:+.6f}")

            restore_all()

    # --- Per-layer aggregate (11 evals) ---
    print("\n--- Per-layer aggregate (all 6 matrices of one layer quantized) ---")
    layer_results = []
    for layer in range(n):
        for mtype in MATRIX_TYPES:
            quant_w = quantize_dequantize_int6(get_matrix(model, n, layer, mtype).clone())
            set_matrix(model, n, layer, mtype, quant_w)

        bpb = eval_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        delta = bpb - baseline_bpb
        layer_results.append((layer, bpb, delta))
        print(f"  Layer {layer:2d}: BPB={bpb:.6f}  Δ={delta:+.6f}")

        restore_all()

    # --- Full model quantized (1 eval) ---
    print("\n--- Full model (all layers quantized) ---")
    for layer in range(n):
        for mtype in MATRIX_TYPES:
            quant_w = quantize_dequantize_int6(get_matrix(model, n, layer, mtype).clone())
            set_matrix(model, n, layer, mtype, quant_w)
    full_bpb = eval_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    full_delta = full_bpb - baseline_bpb
    print(f"  Full model int6: BPB={full_bpb:.6f}  Δ={full_delta:+.6f}")
    restore_all()

    # --- Summary tables ---
    print("\n" + "=" * 90)
    print("QUANTIZATION SENSITIVITY ANALYSIS — PER-MATRIX BPB DELTA")
    print("=" * 90)
    print(f"\nBaseline BPB (full precision bfloat16): {baseline_bpb:.6f}")
    print(f"Full model int6 (no GPTQ): BPB={full_bpb:.6f}  Δ={full_delta:+.6f}")

    print(f"\n{'Layer':<8}", end="")
    for mt in MATRIX_TYPES:
        print(f"{mt:>12}", end="")
    print(f"{'ALL':>12}")
    print("-" * (8 + 12 * 7))

    for layer in range(n):
        print(f"L{layer:<7d}", end="")
        for mt in MATRIX_TYPES:
            r = next(x for x in results if x[0] == layer and x[1] == mt)
            print(f"{r[3]:>+12.6f}", end="")
        lr = next(x for x in layer_results if x[0] == layer)
        print(f"{lr[2]:>+12.6f}")

    # Column totals (sum of isolated deltas — NOT simultaneous quantization)
    print(f"\n{'Sum iso.:':<8}", end="")
    for mt in MATRIX_TYPES:
        total = sum(x[3] for x in results if x[1] == mt)
        print(f"{total:>+12.6f}", end="")
    print()

    # Top-10 most sensitive
    sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
    print(f"\nTop 10 most sensitive matrices:")
    for layer, mtype, bpb, delta in sorted_results[:10]:
        print(f"  L{layer} {mtype:<10s}  Δ={delta:+.6f}")

    print(f"\nTop 10 least sensitive matrices:")
    for layer, mtype, bpb, delta in sorted_results[-10:]:
        print(f"  L{layer} {mtype:<10s}  Δ={delta:+.6f}")

    print("\n" + "=" * 90)
    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
