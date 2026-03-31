# Optimal Eval-Time Prediction Mixing — Experiment Plan

**Goal**: Write a small research paper on optimal eval-time prediction model mixing for fixed-corpus LM scoring, validated on the Parameter Golf (FineWeb val, 62M tokens, 1024-vocab BPE).

**Base model**: Our PR #549 stack at ~1.1194 BPB (or the `train_gpt.py` in `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`). Train once, reuse checkpoint for all eval experiments.

**Cost model**: Each eval run ~5 min on 8xH100. Budget: ~65 eval runs = ~5.5 GPU-hours.

---

## CRITICAL CONSTRAINT: STRICT CAUSALITY

**You CANNOT access validation tokens you have not yet scored.**

Every technique in this plan MUST satisfy:
1. **Score-first**: Each token is scored by the model BEFORE it enters any cache or statistical model.
2. **Backward-looking only**: N-gram tables, suffix trees, PPM models, etc. are built exclusively from already-scored tokens. Future tokens are invisible.
3. **No oracle selection**: Mixing weights, gating decisions, etc. must depend only on the model's own output (e.g. its entropy), NEVER on the ground-truth target token.
4. **No min(NLL) tricks**: You cannot score a token with multiple models and pick whichever gave lower loss after the fact.

Any experiment that violates these constraints produces illegitimate results and will be discarded. When in doubt, score first, update after.

---

## Workspace Layout

```
experiments/eval_time_mixing/
  PLAN.md           -- this file
  LOG.md            -- chronological experiment log (runs, results, observations)
  SURPRISES.md      -- non-obvious findings, failed hypotheses, unexpected results
  scripts/          -- runnable experiment scripts
  results/          -- raw outputs, CSVs, logs from runs
  references/       -- saved PR code snippets, relevant papers
```

---

## Experiments

### EXP-0: Baseline Reproduction
**Status**: DONE
**Purpose**: Reproduce existing n-gram approaches to validate our eval harness.

| Run   | Config                                    | Expected BPB |
|-------|-------------------------------------------|-------------|
| 0a    | Neural only (no cache, stride=64)         | ~1.12       |
| 0b    | Fixed 7-gram, alpha=0.40, no backoff      | ~1.03       |
| 0c    | Backoff 2-7, alpha=0.40                   | ~0.98       |
| 0d    | Backoff 2-7, entropy-adaptive alpha       | ~0.97       |
| 0e    | Backoff 2-9, order-adaptive entropy (#788) | ~0.91       |

**Outputs**: Table 1 in the paper.

---

### EXP-1: N-gram Order Scaling Law
**Status**: QUEUED (after EXP-0 and EXP-2)
**Purpose**: Characterize diminishing returns as max order increases.

Sweep max order K in {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20}. Backoff K->2, entropy-adaptive alpha.

**Measurements**: BPB, per-order hit rate, eval time, peak memory.

**Expected**: Saturation around order 9-12. The shape of this curve is publishable.

---

### EXP-2: Logistic vs Linear Mixing (CORE CLAIM)
**Status**: RUNNING
**Purpose**: Test the paper's central theoretical claim — logistic mixing eliminates the entropy floor of linear interpolation.

| Variant | Formula |
|---------|---------|
| 2a: Linear (current) | `p_mix = (1-a)*p_model + a*p_ngram` |
| 2b: Logistic | `p_mix = sigma(w1*logit(p_model) + w2*logit(p_ngram))`, w1=1-a, w2=a |
| 2c: Logistic + online SGD | Same as 2b, update w1,w2 via SGD per segment |
| 2d: Geometric mean | `p_mix proportional to p_model^(1-a) * p_ngram^a` (renormalized) |

**Additional analysis**: Bucket tokens by p_ngram confidence ([0-0.2], [0.2-0.5], [0.5-0.8], [0.8-0.95], [0.95-1.0]) and compare BPB penalty per bucket. This directly validates the entropy-floor argument from Section 4.

---

### EXP-3: PPM Escape Probabilities vs Naive MLE
**Status**: NOT STARTED
**Purpose**: Test whether proper escape estimation beats naive `count(ctx+tgt)/count(ctx)`.

| Variant | Method |
|---------|--------|
| 3a: Current MLE | `p = full_count / ctx_count` |
| 3b: PPMC | Moffat exclusion, `escape = d/(d+n)` |
| 3c: PPMD | `escape = d/(2n)` |
| 3d: PPM+exclusion | 3b + exclude symbols already accounted at higher orders |

**Implementation**: Needs a `unique_successors[ctx_key]` counter — increment only when `full_table[full_key]` transitions 0->1.

---

### EXP-4: Full-Distribution vs Point-Probability Mixing
**Status**: NOT STARTED
**Purpose**: Current code only mixes p(y_true). Test building full V=1024 distribution.

| Variant | What's mixed |
|---------|-------------|
| 4a: Point (current) | Only p_ngram(y_true) |
| 4b: Full distribution | All 1024 probabilities mixed, then score |
| 4c: Top-K (K=32) | Build n-gram dist for top-32 successors, rest uniform |

**Causality note**: Point mixing (4a) only looks up the true target token's count — this is technically using the ground-truth label to select WHICH hash bucket to query. This is legal because it doesn't affect the model's prediction (it only post-hoc adjusts the score), but full-distribution mixing (4b) is the theoretically cleaner approach since it builds the n-gram distribution WITHOUT knowing the true target.

---

### EXP-5: Hash Collision Impact
**Status**: NOT STARTED
**Purpose**: Quantify collision penalty. Current: 4M buckets.

Sweep buckets: 1M, 4M, 16M, 64M, 256M.
Reference: Python dict (collision-free, slow).

---

### EXP-6: Stride Decomposition
**Status**: NOT STARTED
**Purpose**: Cleanly separate overlap benefit from n-gram benefit.

Sweep stride {64, 128, 256, 512, 1024, 2048}, with and without cache.
Compute: delta_overlap, delta_ngram, total.

---

### EXP-7: Document-Boundary Analysis
**Status**: NOT STARTED
**Purpose**: Cross-document vs within-document n-gram utility.

| Variant | Cache Policy |
|---------|-------------|
| 7a: Global (current) | Never reset |
| 7b: Per-document | Reset at document boundaries |
| 7c: Decaying | Multiply counts by 0.95 at boundaries |

---

### EXP-8: Suffix-Based Longest Match
**Status**: NOT STARTED
**Purpose**: Test unbounded context matching vs fixed-order n-grams.

Implementation: Rolling hash (Rabin-Karp) for 8-token windows, then greedy forward extension.

Measurements: Match length distribution histogram, BPB vs order-limited approaches.

**Causality note**: The suffix structure is built ONLY from already-scored tokens. At position t, we search for the longest match of the context (tokens before t) within the history of tokens scored before t. The token at position t itself must NOT be used in the lookup key (it hasn't been scored yet) — we match the CONTEXT preceding t, then predict what follows.

---

### EXP-9: Online Weight Learning (PAQ-style)
**Status**: NOT STARTED
**Purpose**: Test learned weights vs fixed sigmoid formula.

| Variant | Update Rule |
|---------|------------|
| 9a: Fixed alpha=0.40 | None |
| 9b: Entropy-adaptive (current) | Sigmoid formula |
| 9c: Online SGD (scalar) | Single alpha, grad descent |
| 9d: Online SGD (per-order) | Separate alpha per order |
| 9e: Logistic + per-order SGD | PAQ-style full system |

Learning rates: {0.001, 0.01, 0.1}.

**Causality note**: Weight updates use the prediction error from the CURRENT segment (which has already been scored). The updated weights apply to the NEXT segment. This is strictly causal — we learn from past mistakes to improve future predictions.

---

### EXP-10: FineWeb Empirical Entropy Estimation
**Status**: NOT STARTED
**Purpose**: Establish theoretical ceiling.

Compress val set with gzip-9, bzip2-9, xz-9, zstd-22. Convert to BPB.
If feasible: ppmd on 1M-byte subset, cmix on 10K subset.

---

### EXP-11: 8-GPU Global Cache with All-Reduce Sync (COMPETITION CRITICAL)
**Status**: IMPLEMENTED — ready to deploy
**Purpose**: Achieve global-cache BPB (~0.49) within the 600s eval budget on 8xH100.

**Architecture:**
1. Standard 8-GPU distributed sliding window eval (torchrun --nproc_per_node=8)
2. Hash tables as **PyTorch tensors on GPU** (not numpy on CPU)
3. After EVERY batch, **all-reduce (SUM)** the hash table increments across all 8 GPUs
4. After sync, every GPU has the exact global cache state
5. Flash Attention 3: `pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291`

**Cost estimate:**
- Hash tables: 6 orders × 2 tables × 4M buckets × 4 bytes = 192MB per GPU
- All-reduce per sync: 192MB over NVLink (~900 GB/s on H100 SXM) = <0.3ms
- ~3800 batches → ~3800 syncs = ~1.1s total all-reduce overhead
- Forward pass: 606s ÷ 8 GPUs = ~76s
- N-gram on GPU: ~30-50s (vs 470s on CPU numpy)
- **Expected total: ~130s** (well within 600s)

**Implementation plan:**
- Replace numpy `ctx_tables`/`full_tables` with `torch.zeros(buckets, dtype=torch.int32, device=device)`
- Replace `np.add.at(table, keys, 1)` with `table.scatter_add_(0, keys, ones)` or `table[keys] += 1`
- Hash computation: pure torch integer ops (already vectorizable)
- After scoring + updating local tables each batch:
  `dist.all_reduce(ctx_delta, op=dist.ReduceOp.SUM)`
  `dist.all_reduce(full_delta, op=dist.ReduceOp.SUM)`
- Keep delta buffers to avoid re-syncing accumulated counts

**Key constraint:** Single pass. Each token scored exactly once. Cache updated after scoring. All-reduce syncs the updates so all GPUs see the global picture.

---

## Priority Order

| Priority | Experiments | Rationale |
|----------|------------|-----------|
| **P-CRITICAL** | **11** | **Competition submission: 8-GPU global cache under 600s** |
| P0 | 0, 2, 1 | Core paper: baselines, main claim, scaling law |
| P1 | 3, 8, 5 | Advanced techniques: PPM, suffix matching, collision analysis |
| P2 | 6, 9, 4 | Supporting evidence: stride decomposition, online learning, full-dist |
| P3 | 7, 10 | Characterization: document boundaries, entropy ceiling |
