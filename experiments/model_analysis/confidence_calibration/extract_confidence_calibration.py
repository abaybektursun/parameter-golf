"""Extract confidence calibration results into JSON for the dashboard."""
import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "confidence_calibration_results.txt"
OUT = Path(__file__).resolve().parent.parent / "dashboard/public/data/confidence_calibration.json"

def main():
    text = RESULTS.read_text()

    # Parse summary stats
    total_tokens = int(re.search(r"Total tokens:\s*([\d,]+)", text).group(1).replace(",", ""))
    overall_acc = float(re.search(r"Overall top-1 accuracy:\s*([\d.]+)%", text).group(1)) / 100
    overall_bpb = float(re.search(r"Overall BPB:\s*([\d.]+)", text).group(1))
    entropy_nats = float(re.search(r"Mean prediction entropy:\s*([\d.]+) nats", text).group(1))
    entropy_bits = float(re.search(r"\(([\d.]+) bits\)", text).group(1))
    ece = float(re.search(r"Expected Calibration Error.*?:\s*([\d.]+)", text).group(1))

    # Parse calibration table (binned by confidence)
    calibration_bins = []
    for m in re.finditer(
        r"\[([\d.]+),([\d.]+)\)\s+([\d,]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([+\-][\d.]+)\s+([\d.]+)\s+([\d.]+)%",
        text[:text.index("P(correct)")]
    ):
        calibration_bins.append({
            "bin_lo": float(m.group(1)),
            "bin_hi": float(m.group(2)),
            "count": int(m.group(3).replace(",", "")),
            "pct_tokens": float(m.group(4)),
            "avg_confidence": float(m.group(5)),
            "accuracy": float(m.group(6)),
            "gap": float(m.group(7)),
            "avg_loss": float(m.group(8)),
            "pct_loss": float(m.group(9)),
        })

    # Parse P(correct) table
    pcorrect_section = text[text.index("P(correct)"):text.index("Rank")]
    pcorrect_bins = []
    for m in re.finditer(
        r"\[([\d.]+),([\d.]+)\)\s+([\d,]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%",
        pcorrect_section,
    ):
        pcorrect_bins.append({
            "bin_lo": float(m.group(1)),
            "bin_hi": float(m.group(2)),
            "count": int(m.group(3).replace(",", "")),
            "pct_tokens": float(m.group(4)),
            "avg_p_correct": float(m.group(5)),
            "avg_loss": float(m.group(6)),
            "pct_loss": float(m.group(7)),
        })

    # Parse rank histogram
    rank_histogram = []
    for m in re.finditer(r"^\s+(\d+\+?)\s+([\d,]+)\s+([\d.]+)%\s+([\d.]+)%", text, re.MULTILINE):
        rank_histogram.append({
            "rank": m.group(1),
            "count": int(m.group(2).replace(",", "")),
            "pct": float(m.group(3)),
            "cumulative": float(m.group(4)),
        })

    # Parse per-type ECE
    type_ece = []
    for m in re.finditer(r"^(\w+)\s+([\d.]+)\s+([\d.]+)%", text[text.index("Type"):], re.MULTILINE):
        type_ece.append({
            "type": m.group(1),
            "ece": float(m.group(2)),
            "accuracy": float(m.group(3)) / 100,
        })

    # Parse over/underconfidence
    overconf_pct_tokens = float(re.search(r"Overconfident.*?Tokens:.*?\(([\d.]+)%\)", text, re.DOTALL).group(1))
    overconf_pct_loss = float(re.search(r"Overconfident.*?Loss:.*?\(([\d.]+)%\)", text, re.DOTALL).group(1))
    underconf_pct_tokens = float(re.search(r"Underconfident.*?Tokens:.*?\(([\d.]+)%\)", text, re.DOTALL).group(1))
    underconf_pct_loss = float(re.search(r"Underconfident.*?Loss:.*?\(([\d.]+)%\)", text, re.DOTALL).group(1))

    out = {
        "total_tokens": total_tokens,
        "overall_accuracy": overall_acc,
        "overall_bpb": overall_bpb,
        "mean_entropy_nats": entropy_nats,
        "mean_entropy_bits": entropy_bits,
        "ece": ece,
        "calibration_bins": calibration_bins,
        "pcorrect_bins": pcorrect_bins,
        "rank_histogram": rank_histogram,
        "type_ece": type_ece,
        "overconfident_pct_tokens": overconf_pct_tokens,
        "overconfident_pct_loss": overconf_pct_loss,
        "underconfident_pct_tokens": underconf_pct_tokens,
        "underconfident_pct_loss": underconf_pct_loss,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
