"""
analyze_ngram_matches.py — Analyze WHAT the n-gram cache is actually matching.

Runs sliding-window eval with n-gram backoff, but instead of just scoring,
records detailed per-token diagnostics:
- Which tokens got n-gram matches at which order
- The actual text of matched n-grams
- Match probability vs model probability
- Categories of matched content (boilerplate, repeated phrases, etc.)

Outputs a JSON with sampled match examples and aggregate statistics.
Designed for visualization and qualitative analysis.

STRICT CAUSALITY: All lookups backward-looking only.
"""

from __future__ import annotations
import json
import math
import os
import sys
import time
import collections
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Load model classes
REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
_src = (REPO / "train_609_val_calib.py").read_text()
exec(compile(_src.split("\ndef main")[0], "train_609_val_calib.py", "exec"))

PRIMES = np.array(
    [np.uint64(36313), np.uint64(27191), np.uint64(51647),
     np.uint64(81929), np.uint64(131071), np.uint64(174763),
     np.uint64(233017), np.uint64(310019), np.uint64(412553)],
    dtype=np.uint64,
)


def main():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    hp = Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    val_tokens = load_validation_tokens(hp.val_files, hp.train_seq_len)
    val_np = val_tokens.cpu().numpy()
    total_tokens = val_tokens.numel() - 1
    print(f"Val tokens: {total_tokens:,}")

    # Decode tokens for visualization
    id_to_piece = {i: sp.IdToPiece(i) for i in range(hp.vocab_size)}

    # N-gram setup
    ngram_max_order = 7
    ngram_min_order = 2
    ngram_buckets = 4_194_304
    ngram_min_count = 2
    ng_mask = np.uint64(ngram_buckets - 1)
    orders = list(range(ngram_min_order, ngram_max_order + 1))
    n_orders = len(orders)
    ctx_tables = [np.zeros(ngram_buckets, dtype=np.uint32) for _ in range(n_orders)]
    full_tables = [np.zeros(ngram_buckets, dtype=np.uint32) for _ in range(n_orders)]

    # Statistics collectors
    order_hit_counts = {o: 0 for o in orders}
    total_positions = 0
    no_match_count = 0

    # Sample detailed matches (collect up to 1000 per order)
    MAX_SAMPLES = 200
    match_samples = {o: [] for o in orders}

    # Per-position: scan sequentially through the entire val set
    # (No sliding window needed for pure analysis — process each token once)
    print("Scanning val set sequentially...", flush=True)
    t0 = time.time()

    CHUNK = 10000  # Process in chunks for efficiency
    for start in range(0, total_tokens, CHUNK):
        end = min(start + CHUNK, total_tokens)
        chunk_positions = np.arange(start + 1, end + 1, dtype=np.int64)  # target positions

        for oi in range(n_orders - 1, -1, -1):
            order = orders[oi]
            ctx_w = order - 1
            valid = chunk_positions >= order
            if not valid.any():
                continue
            v_idx = np.nonzero(valid)[0]
            jv = chunk_positions[v_idx]

            ctx_hash = np.zeros(len(jv), dtype=np.uint64)
            for k in range(ctx_w):
                tok = val_np[jv - (ctx_w - k)].astype(np.uint64)
                ctx_hash ^= tok * PRIMES[k % len(PRIMES)]
            ctx_key = (ctx_hash & ng_mask).astype(np.int64)
            tgt = val_np[jv].astype(np.uint64)
            full_key = ((ctx_hash ^ (tgt * PRIMES[ctx_w % len(PRIMES)])) & ng_mask).astype(np.int64)

            cc = ctx_tables[oi][ctx_key].astype(np.float64)
            fc = full_tables[oi][full_key].astype(np.float64)
            has_match = cc >= float(ngram_min_count)

            if has_match.any():
                matched_positions = jv[has_match]
                matched_p = np.minimum(fc[has_match], cc[has_match]) / np.maximum(cc[has_match], 1.0)
                order_hit_counts[order] += int(has_match.sum())

                # Sample some matches for visualization
                if len(match_samples[order]) < MAX_SAMPLES:
                    for mi in range(min(10, int(has_match.sum()))):
                        if len(match_samples[order]) >= MAX_SAMPLES:
                            break
                        pos = int(matched_positions[mi])
                        p_ng = float(matched_p[mi])
                        ctx_start = max(0, pos - ctx_w)
                        context_tokens = val_np[ctx_start:pos].tolist()
                        target_token = int(val_np[pos])
                        context_text = "".join(id_to_piece[t] for t in context_tokens)
                        target_text = id_to_piece[target_token]
                        match_samples[order].append({
                            "position": pos,
                            "order": order,
                            "p_ngram": p_ng,
                            "context": context_text,
                            "target": target_text,
                            "context_tokens": context_tokens,
                            "target_token": target_token,
                        })

        # Update tables AFTER processing chunk (score-first)
        for oi in range(n_orders):
            order = orders[oi]
            ctx_w = order - 1
            valid = chunk_positions >= order
            if not valid.any():
                continue
            v_idx = np.nonzero(valid)[0]
            jv = chunk_positions[v_idx]
            ctx_hash = np.zeros(len(jv), dtype=np.uint64)
            for k in range(ctx_w):
                tok = val_np[jv - (ctx_w - k)].astype(np.uint64)
                ctx_hash ^= tok * PRIMES[k % len(PRIMES)]
            ctx_key = (ctx_hash & ng_mask).astype(np.int64)
            tgt = val_np[jv].astype(np.uint64)
            full_key = ((ctx_hash ^ (tgt * PRIMES[ctx_w % len(PRIMES)])) & ng_mask).astype(np.int64)
            np.add.at(ctx_tables[oi], ctx_key, 1)
            np.add.at(full_tables[oi], full_key, 1)

        total_positions += (end - start)
        no_match_this = (end - start) - sum(
            1 for o in orders if order_hit_counts[o] > 0)  # approximate

        if start % 1000000 == 0:
            elapsed = time.time() - t0
            pct = 100.0 * start / total_tokens
            print(f"  [{pct:5.1f}%] pos={start:,}/{total_tokens:,} elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s", flush=True)

    # Aggregate statistics
    stats = {
        "total_positions": total_positions,
        "per_order_hits": order_hit_counts,
        "hit_rate_by_order": {o: order_hit_counts[o] / max(total_positions, 1)
                              for o in orders},
    }

    # Categorize samples by content type
    # Look for common patterns in high-confidence matches
    high_conf_samples = []
    for order in orders:
        for s in match_samples[order]:
            if s["p_ngram"] > 0.9:
                high_conf_samples.append(s)

    output = {
        "stats": stats,
        "match_samples": {str(o): match_samples[o] for o in orders},
        "high_confidence_examples": high_conf_samples[:100],
        "elapsed_s": elapsed,
    }

    out_path = os.path.join(
        str(REPO), "experiments", "eval_time_mixing", "results", "ngram_analysis.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")

    # Print summary
    print(f"\n=== N-gram Match Analysis ===")
    print(f"Total positions: {total_positions:,}")
    for o in orders:
        rate = order_hit_counts[o] / max(total_positions, 1) * 100
        print(f"  Order {o}: {order_hit_counts[o]:,} hits ({rate:.1f}%)")
    print(f"\nHigh-confidence (p>0.9) examples: {len(high_conf_samples)}")
    for ex in high_conf_samples[:10]:
        print(f"  [{ex['order']}-gram p={ex['p_ngram']:.3f}] "
              f"'{ex['context']}' -> '{ex['target']}'")


if __name__ == "__main__":
    main()
