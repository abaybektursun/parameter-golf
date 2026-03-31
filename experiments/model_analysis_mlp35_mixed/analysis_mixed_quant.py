"""Mixed-precision quant sensitivity: int5 vs int6 per matrix + optimal bit allocation for 16MB."""
import math
import time
import json
import torch
import torch.nn.functional as F
import sentencepiece as spm

from common import (
    make_args, load_model, load_validation_tokens, build_sentencepiece_luts,
    TOKENIZER_PATH,
)

DEVICE = "cuda:0"
SEQ_LEN = 2048
BATCH_SEQS = 32
ARTIFACT_BUDGET_BYTES = 16 * 1024 * 1024  # 16 MB


def quantize_dequantize(weight, bits):
    """Quantize weight to int-N and dequantize back. bits=5 -> clip_range=15, bits=6 -> clip_range=31."""
    clip_range = (1 << (bits - 1)) - 1
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
    """Evaluate BPB on the full validation set."""
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
    n = args.num_layers

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

    # --- Per-matrix sensitivity at int5 AND int6 ---
    results = []
    for layer in range(n):
        for mtype in MATRIX_TYPES:
            nparams = get_matrix(model, n, layer, mtype).numel()
            row = {"layer": layer, "type": mtype, "params": nparams}

            for bits in [5, 6]:
                orig_w = get_matrix(model, n, layer, mtype).clone()
                quant_w = quantize_dequantize(orig_w, bits)
                set_matrix(model, n, layer, mtype, quant_w)
                bpb = eval_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
                delta = bpb - baseline_bpb
                row[f"bpb_int{bits}"] = bpb
                row[f"delta_int{bits}"] = delta
                restore_all()

            row["upgrade_benefit"] = row["delta_int5"] - row["delta_int6"]
            row["upgrade_cost_bits"] = nparams  # 1 extra bit per param
            row["benefit_per_bit"] = row["upgrade_benefit"] / nparams if nparams > 0 else 0.0
            results.append(row)
            print(f"  L{layer:2d} {mtype:<10s}  int5 d={row['delta_int5']:+.6f}  int6 d={row['delta_int6']:+.6f}  upgrade={row['upgrade_benefit']:+.6f}")

    # --- Greedy knapsack: optimal bit allocation under 16MB ---
    bank_params = sum(r["params"] for r in results)
    all_params = sum(p.numel() for p in model.parameters())
    non_bank_params = all_params - bank_params

    # Start with everything at int5, compute base artifact size
    scale_overhead_bits = sum(get_matrix(model, n, r["layer"], r["type"]).shape[0] * 16 for r in results)
    base_artifact_bits = bank_params * 5 + non_bank_params * 6 + scale_overhead_bits
    base_artifact_bytes = base_artifact_bits / 8

    print(f"\n--- KNAPSACK ALLOCATION ---")
    print(f"Total bank params: {bank_params:,}")
    print(f"Non-bank params: {non_bank_params:,}")
    print(f"All-int5 artifact: {base_artifact_bytes / 1024 / 1024:.2f} MB")
    print(f"Budget: {ARTIFACT_BUDGET_BYTES / 1024 / 1024:.2f} MB")

    remaining_budget_bits = (ARTIFACT_BUDGET_BYTES * 8) - base_artifact_bits

    # Sort by benefit_per_bit descending
    ranked = sorted(results, key=lambda r: r["benefit_per_bit"], reverse=True)
    allocation = {}
    total_upgrade_benefit = 0.0
    upgraded_count = 0

    for r in ranked:
        key = f"L{r['layer']}_{r['type']}"
        cost = r["upgrade_cost_bits"]
        if cost <= remaining_budget_bits:
            allocation[key] = 6
            remaining_budget_bits -= cost
            total_upgrade_benefit += r["upgrade_benefit"]
            upgraded_count += 1
        else:
            allocation[key] = 5

    final_artifact_bytes = (ARTIFACT_BUDGET_BYTES * 8 - remaining_budget_bits) / 8
    int5_bpb = baseline_bpb + sum(r["delta_int5"] for r in results)
    estimated_mixed_bpb = int5_bpb - total_upgrade_benefit

    print(f"\nOptimal allocation: {upgraded_count}/{len(results)} matrices at int6, rest at int5")
    print(f"Estimated artifact: {final_artifact_bytes / 1024 / 1024:.2f} MB")
    print(f"All-int5 BPB (isolated sum): {int5_bpb:.6f}")
    print(f"Mixed BPB (estimated): {estimated_mixed_bpb:.6f}")
    print(f"Upgrade benefit: {total_upgrade_benefit:.6f} BPB")

    # --- Report ---
    print("\n" + "=" * 100)
    print("MIXED-PRECISION QUANTIZATION ANALYSIS")
    print("=" * 100)
    print(f"\nBaseline BPB: {baseline_bpb:.6f}")

    header = "Layer".ljust(8) + "Type".ljust(10) + "Params".rjust(10) + "int5 d".rjust(12) + "int6 d".rjust(12) + "Upgrade".rjust(12) + "Alloc".rjust(6)
    print(f"\n{header}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x["layer"], MATRIX_TYPES.index(x["type"]))):
        key = f"L{r['layer']}_{r['type']}"
        alloc = allocation[key]
        print(f"L{r['layer']:<7d} {r['type']:<10s} {r['params']:>10,} {r['delta_int5']:>+12.6f} {r['delta_int6']:>+12.6f} {r['upgrade_benefit']:>+12.6f} {'int' + str(alloc):>6}")

    # Top upgrades
    print(f"\nTop 10 most valuable upgrades (int5 -> int6):")
    for r in ranked[:10]:
        key = f"L{r['layer']}_{r['type']}"
        print(f"  {key:<16s}  upgrade={r['upgrade_benefit']:+.6f}  benefit/Mbit={r['benefit_per_bit']*1e6:+.4f}")

    # Summary by type
    type_header = "Type".ljust(10) + "Sum int5 d".rjust(12) + "Sum int6 d".rjust(12) + "Sum upgrade".rjust(12)
    print(f"\n{type_header}")
    print("-" * 50)
    for mt in MATRIX_TYPES:
        s5 = sum(r["delta_int5"] for r in results if r["type"] == mt)
        s6 = sum(r["delta_int6"] for r in results if r["type"] == mt)
        print(f"{mt:<10} {s5:>+12.6f} {s6:>+12.6f} {s5 - s6:>+12.6f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save JSON
    json_out = {
        "baseline_bpb": baseline_bpb,
        "artifact_budget_mb": ARTIFACT_BUDGET_BYTES / 1024 / 1024,
        "all_int5_artifact_mb": base_artifact_bytes / 1024 / 1024,
        "estimated_mixed_bpb": estimated_mixed_bpb,
        "allocation": allocation,
        "per_matrix": results,
    }
    with open("results_mixed_quant.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Results saved to results_mixed_quant.json")


if __name__ == "__main__":
    main()
