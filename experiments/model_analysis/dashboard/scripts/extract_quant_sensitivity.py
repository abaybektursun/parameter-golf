"""Extract quantization sensitivity results into JSON for the dashboard."""
import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "weights/quant_sensitivity_results.txt"
OUT = Path(__file__).resolve().parent.parent / "public/data/quant_sensitivity.json"

def main():
    text = RESULTS.read_text()

    # Parse baseline
    m = re.search(r"Baseline BPB.*?:\s*([\d.]+)", text)
    baseline_bpb = float(m.group(1))

    # Parse full model
    m = re.search(r"Full model int6.*?BPB=([\d.]+)\s+.*?=([\+\-\d.]+)", text)
    full_model_bpb = float(m.group(1))
    full_model_delta = float(m.group(2))

    # Parse per-matrix results
    per_matrix = []
    for m in re.finditer(r"L\s*(\d+)\s+(Q|K|V|Out|MLP_up|MLP_down)\s+BPB=([\d.]+)\s+.*?=([\+\-\d.]+)", text):
        per_matrix.append({
            "layer": int(m.group(1)),
            "type": m.group(2),
            "bpb": float(m.group(3)),
            "delta": float(m.group(4)),
        })

    # Parse per-layer aggregate
    per_layer = []
    for m in re.finditer(r"Layer\s+(\d+):\s+BPB=([\d.]+)\s+.*?=([\+\-\d.]+)", text):
        per_layer.append({
            "layer": int(m.group(1)),
            "bpb": float(m.group(2)),
            "delta": float(m.group(3)),
        })

    out = {
        "baseline_bpb": baseline_bpb,
        "full_model_int6_bpb": full_model_bpb,
        "full_model_int6_delta": full_model_delta,
        "per_matrix": per_matrix,
        "per_layer": per_layer,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
