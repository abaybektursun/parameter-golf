"""Parse svd_results.txt for summary + load weights for full SV curves."""
import json, re, sys
from pathlib import Path

TXT_PATH = Path(__file__).resolve().parent.parent.parent / "weights/svd_results.txt"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent.parent / "weights/final_model.pt"
SUMMARY_OUT = Path(__file__).resolve().parent.parent / "public/data/svd_summary.json"
CURVES_OUT = Path(__file__).resolve().parent.parent / "public/data/sv_curves.json"

# --- Parse the text summary ---
text = TXT_PATH.read_text()
matrices = []

for line in text.splitlines():
    # L0_Q  512x512  380  292  47.0  12.978  0.00118  10958
    m = re.match(
        r"(\S+)\s+(\d+)x(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        line.strip(),
    )
    if not m:
        continue
    label = m.group(1)
    rows, cols = int(m.group(2)), int(m.group(3))

    # Determine type and layer
    if "_Q" in label: mtype = "Q"
    elif "_K" in label: mtype = "K"
    elif "_V" in label: mtype = "V"
    elif "_Out" in label: mtype = "Out"
    elif "_MLP_up" in label: mtype = "MLP_up"
    elif "_MLP_down" in label: mtype = "MLP_down"
    else: mtype = "other"

    layer_m = re.match(r"L(\d+)_", label)
    layer = int(layer_m.group(1)) if layer_m else -1

    matrices.append({
        "label": label,
        "layer": layer,
        "type": mtype,
        "shape": [rows, cols],
        "eff_rank_99": int(m.group(4)),
        "eff_rank_95": int(m.group(5)),
        "stable_rank": float(m.group(6)),
        "sv_max": float(m.group(7)),
        "sv_min": float(m.group(8)),
        "condition_number": float(m.group(9)),
    })

SUMMARY_OUT.write_text(json.dumps({"matrices": matrices}, indent=2))
print(f"Wrote {len(matrices)} matrices to {SUMMARY_OUT}")

# --- Extract full SV curves from weights ---
import torch

print(f"Loading weights from {WEIGHTS_PATH}...")
sd = torch.load(WEIGHTS_PATH, map_location="cpu")
n = 11  # num_layers

bank_map = {
    "qo_bank": lambda idx: f"L{idx}_Q" if idx < n else f"L{idx - n}_Out",
    "kv_bank": lambda idx: f"L{idx}_K" if idx < n else f"L{idx - n}_V",
    "mlp_up_bank": lambda idx: f"L{idx}_MLP_up",
    "mlp_down_bank": lambda idx: f"L{idx}_MLP_down",
}

curves = {}

for bank_name, label_fn in bank_map.items():
    bank = sd.get(bank_name)
    if bank is None:
        continue
    for idx in range(bank.shape[0]):
        label = label_fn(idx)
        sv = torch.linalg.svdvals(bank[idx].float()).tolist()
        # Determine type
        if "_Q" in label: mtype = "Q"
        elif "_K" in label: mtype = "K"
        elif "_V" in label: mtype = "V"
        elif "_Out" in label: mtype = "Out"
        elif "_MLP_up" in label: mtype = "MLP_up"
        elif "_MLP_down" in label: mtype = "MLP_down"
        else: mtype = "other"

        layer_m = re.match(r"L(\d+)_", label)
        layer = int(layer_m.group(1)) if layer_m else -1

        curves[label] = {"type": mtype, "layer": layer, "sv": sv}
        print(f"  {label}: {len(sv)} singular values")

# Non-banked 2D weights
for name, param in sd.items():
    if "bank" in name or param.ndim != 2 or min(param.shape) < 16:
        continue
    sv = torch.linalg.svdvals(param.float()).tolist()
    curves[name] = {"type": "other", "layer": -1, "sv": sv}
    print(f"  {name}: {len(sv)} singular values")

CURVES_OUT.write_text(json.dumps({"curves": curves}, indent=2))
print(f"Wrote {len(curves)} curves to {CURVES_OUT}")
