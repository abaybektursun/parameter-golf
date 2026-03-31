"""Parse logit_lens_results.txt into structured JSON."""
import json, re
from pathlib import Path

TXT_PATH = Path(__file__).resolve().parent.parent.parent / "weights/logit_lens_results.txt"
OUT_PATH = Path(__file__).resolve().parent.parent / "public/data/logit_lens.json"

text = TXT_PATH.read_text()

# 11 layers: encoder = 0-4 (5 layers), decoder = 5-10 (6 layers)
NUM_ENCODER = 5
NUM_DECODER = 6

probe_points = []
for line in text.splitlines():
    # Match: embed  18.3498  26.4732  0.5%
    # or: layer_0  7.3965  10.6708  7.7%  -15.8023
    m = re.match(r"(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s*([-+]?[\d.]*)", line.strip())
    if m:
        label = m.group(1)
        loss_nats = float(m.group(2))
        bits_per_tok = float(m.group(3))
        top1_acc = float(m.group(4)) / 100.0
        delta_str = m.group(5)
        delta = float(delta_str) if delta_str else None

        # Determine layer index and role
        if label == "embed":
            layer_idx = -1
            is_encoder = True
            is_skip = False
        elif label == "final":
            layer_idx = 11
            is_encoder = False
            is_skip = False
        else:
            layer_idx = int(label.split("_")[1])
            is_encoder = layer_idx < NUM_ENCODER
            is_skip = not is_encoder and layer_idx < NUM_ENCODER + NUM_DECODER

        probe_points.append({
            "label": label,
            "layer_idx": layer_idx,
            "loss_nats": loss_nats,
            "bits_per_token": bits_per_tok,
            "top1_accuracy": top1_acc,
            "delta_bits_per_token": delta,
            "is_encoder": is_encoder,
            "is_decoder": not is_encoder and label != "embed",
        })

# U-Net skip connections: encoder layer i connects to decoder layer (num_encoder + i)
# with num_skip = min(num_encoder, num_decoder) = 5
skip_connections = [
    {"from_encoder": NUM_ENCODER - 1 - i, "to_decoder": NUM_ENCODER + i}
    for i in range(min(NUM_ENCODER, NUM_DECODER))
]

output = {
    "probe_points": probe_points,
    "architecture": {
        "num_layers": 11,
        "num_encoder_layers": NUM_ENCODER,
        "num_decoder_layers": NUM_DECODER,
        "skip_connections": skip_connections,
    },
}

OUT_PATH.write_text(json.dumps(output, indent=2))
print(f"Wrote {len(probe_points)} probe points to {OUT_PATH}")
