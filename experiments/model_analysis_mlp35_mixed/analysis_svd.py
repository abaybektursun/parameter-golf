"""SVD analysis: singular value spectra of weight matrices to find effective rank per layer."""
import json
import os
import numpy as np
import torch

from common import make_args, WEIGHTS_PATH


def effective_rank(sv, threshold=0.99):
    """Number of singular values needed to capture `threshold` of total variance (sum of sv^2)."""
    sv2 = sv ** 2
    return int((sv2.cumsum(0) < threshold * sv2.sum()).sum().item()) + 1


def stable_rank(sv):
    """||W||_F^2 / ||W||_2^2 = sum(sv^2) / max(sv)^2."""
    sv2 = sv ** 2
    return sv2.sum().item() / sv2[0].item()


def main():
    args = make_args()
    sd = torch.load(WEIGHTS_PATH, map_location="cpu")
    n = args.num_layers  # 11

    # Bank index → readable name
    def bank_label(bank_name, idx):
        if bank_name == "qo_bank":
            return f"L{idx}_Q" if idx < n else f"L{idx - n}_Out"
        if bank_name == "kv_bank":
            return f"L{idx}_K" if idx < n else f"L{idx - n}_V"
        if bank_name == "mlp_up_bank":
            return f"L{idx}_MLP_up"
        if bank_name == "mlp_down_bank":
            return f"L{idx}_MLP_down"
        return f"{bank_name}[{idx}]"

    results = []
    sv_curves = {}  # label -> full SV array for JSON export

    # Banked weights (Q, K, V, Out, MLP_up, MLP_down per layer)
    for bank_name in ["qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank"]:
        bank = sd.get(bank_name)
        if bank is None:
            continue
        for idx in range(bank.shape[0]):
            W = bank[idx].float()
            sv = torch.linalg.svdvals(W)
            label = bank_label(bank_name, idx)
            t = label.split("_", 1)[1] if "_" in label else "other"
            sv_curves[label] = {"type": t, "layer": idx if idx < n else idx - n, "sv": sv.tolist()}
            results.append({
                "label": label,
                "shape": f"{W.shape[0]}x{W.shape[1]}",
                "min_dim": min(W.shape),
                "eff_rank_99": effective_rank(sv, 0.99),
                "eff_rank_95": effective_rank(sv, 0.95),
                "stable_rank": stable_rank(sv),
                "sv_max": sv[0].item(),
                "sv_min": sv[-1].item(),
                "cond": sv[0].item() / max(sv[-1].item(), 1e-10),
            })

    # Non-banked 2D weights (tok_emb, bigram, VE, etc.)
    for name, param in sd.items():
        if "bank" in name or param.ndim != 2 or min(param.shape) < 16:
            continue
        sv = torch.linalg.svdvals(param.float())
        sv_curves[name] = {"type": "other", "layer": -1, "sv": sv.tolist()}
        results.append({
            "label": name,
            "shape": f"{param.shape[0]}x{param.shape[1]}",
            "min_dim": min(param.shape),
            "eff_rank_99": effective_rank(sv, 0.99),
            "eff_rank_95": effective_rank(sv, 0.95),
            "stable_rank": stable_rank(sv),
            "sv_max": sv[0].item(),
            "sv_min": sv[-1].item(),
            "cond": sv[0].item() / max(sv[-1].item(), 1e-10),
        })

    # --- Per-matrix table ---
    print("=" * 90)
    print("SVD ANALYSIS OF WEIGHT MATRICES")
    print("=" * 90)
    print(f"\n{'Matrix':<20} {'Shape':<12} {'Rank99':>7} {'Rank95':>7} {'StblRk':>7} {'SV_max':>8} {'SV_min':>8} {'Cond#':>10}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x["label"]):
        print(f"{r['label']:<20} {r['shape']:<12} {r['eff_rank_99']:>7} {r['eff_rank_95']:>7} "
              f"{r['stable_rank']:>7.1f} {r['sv_max']:>8.3f} {r['sv_min']:>8.5f} {r['cond']:>10.0f}")

    # --- Summary by matrix type ---
    print("\n" + "=" * 90)
    print("SUMMARY BY MATRIX TYPE")
    print("=" * 90)

    type_groups = {}
    for r in results:
        label = r["label"]
        if "_Q" in label: t = "Q"
        elif "_K" in label: t = "K"
        elif "_V" in label: t = "V"
        elif "_Out" in label: t = "Out"
        elif "_MLP_up" in label: t = "MLP_up"
        elif "_MLP_down" in label: t = "MLP_down"
        else: t = "other"
        type_groups.setdefault(t, []).append(r)

    print(f"\n{'Type':<12} {'Count':>5} {'Avg Rank99':>11} {'Avg Rank95':>11} {'Avg StblRk':>11} {'Rank99/dim':>11} {'Avg Cond#':>11}")
    print("-" * 75)
    for t in ["Q", "K", "V", "Out", "MLP_up", "MLP_down", "other"]:
        g = type_groups.get(t, [])
        if not g:
            continue
        print(f"{t:<12} {len(g):>5} "
              f"{np.mean([r['eff_rank_99'] for r in g]):>11.1f} "
              f"{np.mean([r['eff_rank_95'] for r in g]):>11.1f} "
              f"{np.mean([r['stable_rank'] for r in g]):>11.1f} "
              f"{np.mean([r['eff_rank_99'] / r['min_dim'] for r in g]):>10.1%} "
              f"{np.mean([r['cond'] for r in g]):>11.0f}")

    print("\n" + "=" * 90)
    print("INTERPRETATION:")
    print("- High Cond# = fragile under quantization (give more bits in GPTQ)")
    print("- High Rank99/dim = capacity fully utilized")
    print("- High stable rank = energy spread across many directions (not dominated by top SVs)")
    print("=" * 90)

    # Export full SV curves and summary to JSON
    sv_json = {"curves": sv_curves}
    with open("results/sv_curves.json", "w") as f:
        json.dump(sv_json, f)
    print(f"\nSV curves saved to results/sv_curves.json ({len(sv_curves)} matrices)")

    summary = []
    for r in sorted(results, key=lambda x: x["label"]):
        summary.append({k: v for k, v in r.items()})
    with open("results/svd_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"SVD summary saved to results/svd_summary.json")


if __name__ == "__main__":
    main()
