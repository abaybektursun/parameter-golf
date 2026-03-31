"""Parse train.log into structured JSON for the dashboard."""
import json, re
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "weights/train.log"
OUT_PATH = Path(__file__).resolve().parent.parent / "public/data/training_log.json"

steps = []
events = []

for line in LOG_PATH.read_text().splitlines():
    # Training step: step:100/7000 train_loss:3.2226 train_time:34136ms step_avg:341.36ms
    m = re.search(r"step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms", line)
    if m:
        steps.append({
            "step": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "val_loss": None,
            "val_bpb": None,
            "train_time_ms": int(m.group(3)),
        })
        continue

    # Validation: step:1000/7000 val_loss:2.2058 val_bpb:1.3064 train_time:...
    m = re.search(r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line)
    if m:
        step_num = int(m.group(1))
        # Merge into existing step or create new
        existing = next((s for s in steps if s["step"] == step_num), None)
        if existing:
            existing["val_loss"] = float(m.group(2))
            existing["val_bpb"] = float(m.group(3))
        else:
            steps.append({
                "step": step_num,
                "train_loss": None,
                "val_loss": float(m.group(2)),
                "val_bpb": float(m.group(3)),
                "train_time_ms": 0,
            })
        continue

    # Events
    if "swa:start" in line:
        m = re.search(r"swa:start step:(\d+)", line)
        if m:
            events.append({"step": int(m.group(1)), "type": "swa", "label": "SWA starts"})

    if "late_qat:enabled" in line:
        m = re.search(r"late_qat:enabled step:(\d+)", line)
        if m:
            events.append({"step": int(m.group(1)), "type": "qat", "label": "Late QAT enabled"})

    if "ema:applying" in line:
        events.append({"step": 7000, "type": "ema", "label": "EMA weights applied"})

# Deduplicate and sort
seen = set()
unique_steps = []
for s in sorted(steps, key=lambda x: x["step"]):
    if s["step"] not in seen:
        seen.add(s["step"])
        unique_steps.append(s)

# Add warmdown start event (step 3000 = 7000 - 4000)
events.insert(0, {"step": 3000, "type": "warmdown", "label": "Warmdown begins"})

output = {
    "metadata": {
        "total_steps": 7000,
        "step_avg_ms": 343.75,
        "final_val_bpb": 1.1338,
        "model_params": 27067484,
        "hardware": "2×H100 SXM",
        "seed": 314,
    },
    "events": events,
    "steps": unique_steps,
}

OUT_PATH.write_text(json.dumps(output, indent=2))
print(f"Wrote {len(unique_steps)} steps, {len(events)} events to {OUT_PATH}")
