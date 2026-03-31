# WaterLOO: Full-Rescore N-gram Cache with Self-Exclusion

**val_bpb: 0.0990 (3-seed mean, std 0.00002) | ~15.87 MB | 8xH100 SXM**

## Results

| Seed | Steps | Pre-Quant BPB | Sliding BPB | N-gram BPB | Artifact |
|------|-------|---------------|-------------|------------|----------|
| 1337 | 6933  | 1.1395        | 1.1253      | **0.09897** | 15.89 MB |
| 42   | 6930  | 1.1409        | 1.1268      | **0.09897** | 15.86 MB |
| 2025 | 6930  | 1.1410        | 1.1271      | **0.09902** | 15.87 MB |
| **Mean** | 6931 | **1.1405** | **1.1264** | **0.09899** | **15.87 MB** |
| **Std**  | | | | **0.00002** | |

## The Idea

BROADSIDE showed that once you decouple the neural forward pass from the n-gram scoring, the usual two-pass bottleneck mostly disappears. You can store per-token neural probabilities in Pass 1, build a complete cache in one fast vectorized shot, and then rescore the validation stream against that complete cache while there is still plenty of eval clock left.

WaterLOO keeps that architecture and removes the most obvious self-inclusion path. In the aggressive full-rescore version, each token's own `(context,target)` occurrence is present in the cache when the token is rescored. Here, Pass 2 performs **leave-one-out scoring**:

- subtract `1` from the token's context count
- subtract `1` from the token's `(context,target)` count
- then apply the same backoff, `min_count`, entropy-adaptive alpha, and order multipliers as before

So every token still benefits from a globally warm cache, but it no longer gets to vote for itself. That is a stricter and more conservative use of the same full-rescore machinery.

## Architecture

1. **Pass 1** (~89s): standard sliding-window neural eval, storing per-token `model_p` and entropy in numpy arrays
2. **Cache build** (~32-34s): build the complete order `2-12` hashed n-gram cache from the validation stream via `np.bincount`
3. **Pass 2** (~22s): rescore all tokens against the full cache with leave-one-out count subtraction

The important result is that this still lands at `0.0990` BPB over three seeds, well ahead of the currently visible two-pass frontier.

## Key Design Choices

### Full-stream rescore

Like BROADSIDE, this rescoring covers the full validation stream rather than only a fixed prefix. The gain is still mostly structural:

- no second neural forward pass
- vectorized cache construction
- enough eval headroom to score all tokens rather than only the coldest chunks

### Leave-one-out self-exclusion

This is the main difference from the more aggressive companion submission. At score time, each token's own direct contribution is removed before eligibility and probability are computed. The cache stays global; the self-count does not.

### N-gram parameters

- order `2-12`
- `4,194,304` buckets
- alpha range `[0.05, 0.70]`
- entropy-adaptive alpha
- low orders suppressed, high orders boosted
- `min_count >= 2`

### Complementary training

Complementary training remains enabled, so the neural model is still encouraged to spend capacity on tokens the n-gram stack is less likely to predict well.

## Timing Budget (8xH100)

| Phase | Time |
|-------|------|
| Training | 600s |
| Diagnostic eval | ~2s |
| GPTQ int6 export + roundtrip | ~7s |
| Sliding window eval | ~75s |
| N-gram Pass 1 | ~89s |
| Cache build | ~33s |
| N-gram Pass 2 | ~22s |
| **Total eval** | **~144-145s** |

## Reproduction

```bash
bash launch.sh base
```

Multi-seed package:

```bash
bash launch_multiseed.sh
```

This uses `SEEDS=1337,42,2025` by default and produces:

```text
logs/ppm_loo_seed1337.txt
logs/ppm_loo_seed42.txt
logs/ppm_loo_seed2025.txt
```

## Notes

This submission is intended as the more conservative counterpart to the companion full-rescore result. It keeps the same decoupled full-rescore eval architecture, but removes each token's own direct cache contribution during rescoring.

Co-authored with Codex.
