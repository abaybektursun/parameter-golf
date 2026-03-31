#!/usr/bin/env python3
"""Validate all downloaded 8xH100 training log files."""
import os
import sys
import json
from pathlib import Path

BASE = Path("/Users/abaybektursun/projects/parameter-golf/8xh100_logs")

# Validation criteria
MIN_SIZE = 200  # bytes - real logs are at least a few hundred bytes
TRAINING_MARKERS = [
    "step", "loss", "val_loss", "val_bpb", "bpb",
    "train_loss", "step_avg", "wallclock", "tokens",
    "training", "epoch", "lr", "learning_rate",
]
BAD_MARKERS = [
    "<html", "<!DOCTYPE", "404", "Not Found",
    "403 Forbidden", "Access denied", "rate limit",
    "API rate limit", "Repository not found",
]

results = {"valid": [], "invalid": [], "suspicious": []}

for pr_dir in sorted(BASE.iterdir()):
    if not pr_dir.is_dir() or pr_dir.name.startswith(".") or pr_dir.name == "__pycache__":
        continue
    if not pr_dir.name.startswith("PR_"):
        continue

    pr_num = pr_dir.name.split("_")[1]

    for log_file in sorted(pr_dir.iterdir()):
        if log_file.is_dir():
            continue
        if log_file.name.endswith(".sh") or log_file.name.endswith(".py"):
            continue

        fpath = str(log_file)
        fname = log_file.name
        fsize = log_file.stat().st_size

        entry = {
            "pr": pr_num,
            "file": fname,
            "path": fpath,
            "size": fsize,
            "issues": [],
        }

        # Check 1: Size
        if fsize == 0:
            entry["issues"].append("EMPTY (0 bytes)")
            results["invalid"].append(entry)
            continue
        if fsize < MIN_SIZE:
            entry["issues"].append(f"TINY ({fsize} bytes)")

        # Read content
        try:
            with open(log_file, "r", errors="replace") as f:
                content = f.read(50000)  # first 50KB
        except Exception as e:
            entry["issues"].append(f"READ_ERROR: {e}")
            results["invalid"].append(entry)
            continue

        content_lower = content.lower()

        # Check 2: HTML/error page
        for bad in BAD_MARKERS:
            if bad.lower() in content_lower[:2000]:
                entry["issues"].append(f"BAD_CONTENT: contains '{bad}'")

        # Check 3: Training markers
        marker_count = sum(1 for m in TRAINING_MARKERS if m in content_lower)
        entry["marker_count"] = marker_count

        if marker_count == 0:
            entry["issues"].append("NO_TRAINING_MARKERS")

        # Check 4: Looks like actual step-by-step log
        lines = content.split("\n")
        entry["line_count"] = len(lines)

        has_step_pattern = False
        numeric_lines = 0
        for line in lines[:200]:
            ll = line.lower()
            if ("step" in ll and any(c.isdigit() for c in ll)):
                has_step_pattern = True
            if any(c.isdigit() for c in line) and ("loss" in ll or "bpb" in ll or "step" in ll):
                numeric_lines += 1

        entry["has_step_pattern"] = has_step_pattern
        entry["numeric_data_lines"] = numeric_lines

        # Check 5: Extract key metrics if possible
        last_lines = "\n".join(lines[-30:]).lower()
        if "val_bpb" in last_lines or "val_loss" in last_lines:
            entry["has_final_metrics"] = True
        else:
            entry["has_final_metrics"] = False

        # Classify
        if entry["issues"]:
            if any("BAD_CONTENT" in i for i in entry["issues"]):
                results["invalid"].append(entry)
            elif marker_count == 0:
                results["invalid"].append(entry)
            else:
                results["suspicious"].append(entry)
        elif marker_count < 2 and not has_step_pattern:
            entry["issues"].append("LOW_MARKERS_NO_STEPS")
            results["suspicious"].append(entry)
        else:
            results["valid"].append(entry)

# Print report
print("=" * 80)
print(f"VALIDATION REPORT")
print(f"=" * 80)
print(f"Valid:      {len(results['valid'])} files")
print(f"Suspicious: {len(results['suspicious'])} files")
print(f"Invalid:    {len(results['invalid'])} files")
print(f"Total:      {len(results['valid']) + len(results['suspicious']) + len(results['invalid'])} files")
print()

if results["invalid"]:
    print("=" * 80)
    print("INVALID FILES (should be removed)")
    print("=" * 80)
    for e in results["invalid"]:
        print(f"  PR #{e['pr']} | {e['file']} | {e['size']} bytes | {', '.join(e['issues'])}")
    print()

if results["suspicious"]:
    print("=" * 80)
    print("SUSPICIOUS FILES (need manual review)")
    print("=" * 80)
    for e in results["suspicious"]:
        markers = e.get("marker_count", 0)
        lines = e.get("line_count", 0)
        print(f"  PR #{e['pr']} | {e['file']} | {e['size']}B | {lines} lines | {markers} markers | {', '.join(e['issues'])}")
    print()

print("=" * 80)
print("VALID FILES SUMMARY")
print("=" * 80)
# Group by PR
from collections import defaultdict
by_pr = defaultdict(list)
for e in results["valid"]:
    by_pr[e["pr"]].append(e)

for pr in sorted(by_pr.keys()):
    files = by_pr[pr]
    total_size = sum(f["size"] for f in files)
    total_lines = sum(f.get("line_count", 0) for f in files)
    has_final = any(f.get("has_final_metrics") for f in files)
    fnames = [f["file"] for f in files]
    print(f"  PR #{pr} | {len(files)} file(s) | {total_size:,}B | {total_lines:,} lines | final_metrics={'YES' if has_final else 'NO'} | {', '.join(fnames)}")

# Write JSON for programmatic use
with open(BASE / "validation_report.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nJSON report saved to: {BASE}/validation_report.json")
