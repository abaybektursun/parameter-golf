"""Extract attention head analysis results into JSON for the dashboard."""
import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "attention_heads_results.txt"
OUT = Path(__file__).resolve().parent.parent / "dashboard/public/data/attention_heads.json"


def main():
    text = RESULTS.read_text()

    # Parse summary stats
    total_tokens = int(re.search(r"Total tokens analyzed:\s*([\d,]+)", text).group(1).replace(",", ""))
    num_sequences = int(re.search(r"Sequences:\s*(\d+)", text).group(1))
    seq_len = int(re.search(r"length:\s*(\d+)", text).group(1))
    bigram_pairs = int(re.search(r"Repeated bigram instances found:\s*([\d,]+)", text).group(1).replace(",", ""))

    # Parse per-head table
    heads = []
    for m in re.finditer(
        r"L(\d+)\s+H(\d+)\s+"
        r"([\d.]+)\s+"     # PrevTok
        r"([\d.]+)\s+"     # Induction
        r"([\d.]+)\s+"     # BOS
        r"([\d.]+)\s+"     # Entropy
        r"([\d.]+)\s+"     # SelfAttn
        r"([\d.]+)\s+"     # Off1
        r"([\d.]+)\s+"     # Off2
        r"([\d.]+)\s+"     # Off3
        r"([\d.]+)\s+"     # Off4
        r"([\d.]+)\s+"     # Off5
        r"(\w+)",          # Classification
        text,
    ):
        heads.append({
            "layer": int(m.group(1)),
            "head": int(m.group(2)),
            "prev_token_score": float(m.group(3)),
            "induction_score": float(m.group(4)),
            "bos_score": float(m.group(5)),
            "entropy": float(m.group(6)),
            "self_attn_score": float(m.group(7)),
            "offset_scores": [float(m.group(8 + i)) for i in range(5)],
            "classification": m.group(13),
        })

    # Parse summary
    ind_match = re.search(r"Induction heads \((\d+)\):\s*(.*)", text)
    prev_match = re.search(r"Previous-token heads \((\d+)\):\s*(.*)", text)
    pos_match = re.search(r"Positional heads \((\d+)\):\s*(.*)", text)
    other_match = re.search(r"Other/mixed heads \((\d+)\):\s*(.*)", text)

    def parse_head_list(match):
        if not match or not match.group(2).strip():
            return []
        return [h.strip() for h in match.group(2).split(",") if h.strip()]

    summary = {
        "induction_count": int(ind_match.group(1)) if ind_match else 0,
        "prev_token_count": int(prev_match.group(1)) if prev_match else 0,
        "positional_count": int(pos_match.group(1)) if pos_match else 0,
        "other_count": int(other_match.group(1)) if other_match else 0,
        "induction_heads": parse_head_list(ind_match),
        "prev_token_heads": parse_head_list(prev_match),
        "positional_heads": parse_head_list(pos_match),
    }

    # Encoder/decoder breakdown
    enc_match = re.search(r"Encoder.*?:\s*(\d+) induction,\s*(\d+) previous-token", text)
    dec_match = re.search(r"Decoder.*?:\s*(\d+) induction,\s*(\d+) previous-token", text)
    if enc_match:
        summary["encoder_induction"] = int(enc_match.group(1))
        summary["encoder_prev_token"] = int(enc_match.group(2))
    if dec_match:
        summary["decoder_induction"] = int(dec_match.group(1))
        summary["decoder_prev_token"] = int(dec_match.group(2))

    out = {
        "total_tokens": total_tokens,
        "num_sequences": num_sequences,
        "seq_len": seq_len,
        "repeated_bigrams_found": bigram_pairs,
        "heads": heads,
        "summary": summary,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {len(heads)} heads)")


if __name__ == "__main__":
    main()
