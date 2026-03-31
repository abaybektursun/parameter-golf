"""Parse loss_decomp_results.txt into structured JSON."""
import json, re
from pathlib import Path

TXT_PATH = Path(__file__).resolve().parent.parent.parent / "weights/loss_decomp_results.txt"
OUT_PATH = Path(__file__).resolve().parent.parent / "public/data/loss_decomposition.json"

text = TXT_PATH.read_text()

# Overall BPB
m = re.search(r"Overall BPB: ([\d.]+) \(total_bytes: ([\d.]+)\)", text)
overall_bpb = float(m.group(1))
total_bytes = float(m.group(2))

# Mean entropy
m = re.search(r"Mean prediction entropy: ([\d.]+)", text)
mean_entropy = float(m.group(1))

# Bucket rows: name  BPB  %Bytes  %Loss
buckets = []
dimension_map = {
    "freq_top100": ("frequency", "top-100"),
    "freq_tail": ("frequency", "tail"),
    "pos_first100": ("position", "first 100 tokens"),
    "pos_later": ("position", "later tokens"),
    "type_word": ("type", "word"),
    "type_number": ("type", "number"),
    "type_punct": ("type", "punctuation"),
    "type_whitespace": ("type", "whitespace"),
    "type_other": ("type", "other"),
}

for line in text.splitlines():
    m = re.match(r"(\w+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%", line.strip())
    if m:
        key = m.group(1)
        dim, name = dimension_map.get(key, ("unknown", key))
        buckets.append({
            "dimension": dim,
            "name": name,
            "bpb": float(m.group(2)),
            "pct_bytes": float(m.group(3)),
            "pct_loss": float(m.group(4)),
        })

output = {
    "overall_bpb": overall_bpb,
    "total_bytes": total_bytes,
    "mean_entropy_nats": mean_entropy,
    "buckets": buckets,
}

OUT_PATH.write_text(json.dumps(output, indent=2))
print(f"Wrote {len(buckets)} buckets to {OUT_PATH}")
