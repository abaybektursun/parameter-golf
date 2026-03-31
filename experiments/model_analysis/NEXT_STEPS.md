# What to try next

Current model: 11-layer transformer, 512-dim, 27.1M params, int6 GPTQ → 16 MB, **1.11 BPB**.

This document lists interventions ordered by expected leverage. Each section states the problem, the evidence, the intervention, and what we expect.

---

## 1. Reallocate parameters from attention to MLP

### The problem

MLP is the model's knowledge store. It's at 94.4% SVD utilization — fully packed. It accounts for 80% of quantization damage. Every dimension carries signal.

Attention is over-provisioned. Q matrices sit at 72.6% utilization. Only 2 out of 88 heads are induction heads. 59 heads are classified as "other/mixed" — diffuse, unspecialized.

The model allocates 32% of its parameters (8.65M) to attention and 64% (17.3M) to MLP. The analysis says attention is under-delivering relative to its cost.

### The evidence

- SVD: MLP utilization 94.4% vs Q utilization 72.6%
- Quant sensitivity: MLP accounts for 80% of int6 damage; Q matrices are nearly insensitive
- Logit lens: Layer 10 contributes only -0.47 bpt. Layers 3-5 increase loss (skip prep)
- Attention heads: 2/88 induction, 22/88 previous-token, 59/88 unspecialized

### What to try

**Option A — MLP-only layers.** Remove attention from the 5 least productive layers (3, 4, 5, 9, 10 per logit lens). Saves ~3.9M params. Redistribute to wider MLP across all 11 layers (~22% MLP capacity increase). Keep attention in layers 0, 1, 2, 6, 7, 8 where the logit lens shows it matters.

**Option B — MQA (1 KV head).** Reduces kv_dim from 256 to 64. Saves ~2.2M params across 11 layers. Less disruptive than removing attention entirely. Can combine with Option A.

**Option C — Replace attention with a recurrent mechanism.** Mamba-style selective state spaces or RWKV-style linear attention. Much smaller parameter footprint for cross-token mixing. Frees 5-6M params for MLP (30-35% increase). Bonus: infinite context instead of a 2048-token sliding window. Downside: unproven at 27M params, requires significant engineering.

### What to expect

The analysis predicts wider MLP helps. But the training dynamics caveat is real: attention heads that appear useless at inference may stabilize gradient flow during training (Behnke & Heafield 2020, Michel et al. 2019). Option A is the cleanest test because the logit lens identifies exactly which layers to cut. One training run answers the question.

---

## 2. Mixed-precision quantization

### The problem

Int6 GPTQ costs 0.008 BPB. MLP absorbs 80% of that damage (0.0063 BPB), attention only 20% (0.0016 BPB). Uniform int6 treats all matrices equally. This is wasteful.

### The constraint

The byte budget is tight. Total: 27.07M params × 4.72 bits/param = 16 MB at uniform int6. MLP alone is 17.3M params — at int8 that's 17.3 MB, exceeding the entire budget. You cannot simply upgrade MLP to int8.

### What to try

**Surgical mixed-precision.** Downgrade attention (Q, K, V, Out) to int4. This buys headroom. Then upgrade only MLP_down (the single most sensitive matrix type, responsible for 50% of all quant damage) to int7 or int8. MLP_up stays at int5-int6.

The math: attention at int4 = 8.65M × 4 / 8 = 4.33 MB. Other params at int6 = ~0.65 MB. Remaining for MLP: 16 - 4.33 - 0.65 = 10.97 MB = 87.8M bits for 17.3M MLP params = 5.07 bits/param average for MLP. Allocate ~6 bits to MLP_down and ~4.5 bits to MLP_up.

### What to expect

Recovery of 0.003-0.005 of the 0.008 BPB quantization gap. No retraining required — this is a post-training quantization change. Can be combined with any architecture change from intervention #1.

---

## 3. Perplexity-based training data curation

### The problem

The model has 33-60M bits of knowledge capacity. Exposing it to hundreds of billions of random FineWeb tokens causes capacity dilution — the model overwrites useful patterns with noise.

The epiplexity framework (Finzi et al. 2026, arXiv:2601.03220) formalizes this: data has a learnable component (epiplexity) and an irreducible component (time-bounded entropy). Training on high time-bounded entropy tokens — random strings, hashes, session IDs, unique URLs — burns capacity on noise the model can never predict.

### The evidence

- 28% of tokens have P(correct) < 5% and contribute 70% of total loss
- Confidence calibration shows the model knows it's clueless on these tokens (ECE 0.24%)
- The model is well-calibrated overall but numbers (8.3% ECE) and "other" tokens (5.7% ECE) are poorly calibrated — these categories overlap with high-entropy content

### What to try

**Step 1 — Score FineWeb with a reference model.** Use the current 27.1M model (or a slightly larger open-source model) to compute per-token perplexity across the training corpus.

**Step 2 — Build a coreset.** Discard the lowest-perplexity data (trivial boilerplate the model already predicts perfectly) and the highest-perplexity data (chaotic/random content the model can never predict). Retain the middle — the "Goldilocks zone" of learnable content.

**Step 3 — Retrain on the coreset.** A 5-10B token coreset of high-epiplexity text forces every gradient step to reinforce a useful pattern rather than fighting irreducible noise.

### What to expect

The literature on data pruning (Sorscher et al. 2022, Marion et al. 2023) shows capacity-constrained models benefit substantially from intelligent data selection. Expected gain: uncertain but potentially 0.01-0.03 BPB. Requires one scoring pass over the training data plus one training run.

---

## 4. Loss truncation on unpredictable tokens

### The problem

Same root cause as #3 but applied during training rather than data selection. When the model encounters a genuinely random token (hash, UUID, random alphanumeric), the gradient signal says "you should have predicted this" — but no amount of learning will help. The gradient updates push the model's weights in a random direction, diluting capacity.

### What to try

**Truncated loss.** During training, clip the per-token loss at a threshold (e.g., 95th percentile of the loss distribution). Tokens with loss above the threshold contribute zero gradient. The model stops trying to learn the unlearnable.

This is simpler than data curation (#3) because it requires no preprocessing — just a one-line change to the training loop.

### What to expect

Small but real gain. The model conserves capacity for tokens it can actually predict. Combines well with #3 (curated data with loss truncation removes noise at both the data level and the gradient level). One training run to test.

---

## 5. Vocabulary size experiment

### The problem

The 1024-token BPE vocabulary fragments words into tiny character shards. Frequent tokens become universal building blocks that appear in thousands of unrelated contexts, driving up their conditional entropy. The loss decomposition confirms: top-100 tokens have 1.18 BPB vs tail at 1.10 BPB. The model struggles with frequent tokens precisely because they carry so little semantic signal.

### The tradeoff

Larger vocabulary = shorter sequences (less fragmentation) + more semantic tokens, but more embedding parameters. With tied embeddings:

| Vocab size | Embedding params | Cost at int6 | Sequence length reduction |
|------------|-----------------|--------------|--------------------------|
| 1024       | 524K            | 0.31 MB      | baseline                 |
| 2048       | 1.05M           | 0.62 MB      | ~15-20% shorter          |
| 4096       | 2.10M           | 1.24 MB      | ~25-30% shorter          |

Going from 1024 to 2048 costs ~0.31 MB (one extra embedding matrix worth), freeing the model from ~15-20% of sequence positions. The question: does the shorter sequence compensate for the parameter cost?

### What to try

Train the same architecture with 2048 BPE tokens. Keep total parameter budget constant (slightly narrower MLP to compensate for the larger embedding). Compare BPB.

### What to expect

Unknown. This is the most speculative intervention on the list. The IsoFLOPs literature suggests small models are sensitive to the vocab/network ratio, but nobody has published results at exactly this scale. One training run to test.

---

## What not to try

**Eval-time n-gram mixing.** Every implementation we've examined either produces invalid probability distributions (single-token lookup, entropy expert) or hurts performance when made valid. PR #700's Hedge Mixer drops to ~1.3 BPB with the invalid entropy expert removed. Valid n-gram blending dilutes the neural model's predictions.

**Test-time training.** 25 configurations tested, all negative or negligible. The 62M token causal budget is insufficient to meaningfully shift pre-trained weights. Causal TTT on same-distribution data is functionally equivalent to training longer — and the model was already trained on far more data.

**kNN-LM / non-parametric memory.** PR #738 showed a modest gain but used invalid selective mixing. The model's 512-dim embeddings (from a 16 MB model) are too low-quality for reliable nearest-neighbor retrieval. Dense retrieval quality degrades severely with poor representations.

**Temperature scaling / calibration tricks.** ECE is 0.24%. The model is nearly perfectly calibrated. The bottleneck is raw accuracy, not confidence.

---

## Priority order

| # | Intervention | Requires retraining | Expected BPB gain | Confidence |
|---|-------------|--------------------|--------------------|------------|
| 1 | MLP-only layers (remove attn from 5 layers) | Yes | 0.01-0.03 | Medium |
| 2 | Mixed-precision GPTQ (int4 attn, int7 MLP_down) | No | 0.003-0.005 | High |
| 3 | Perplexity-based data curation | Yes (scoring + training) | 0.01-0.03 | Medium |
| 4 | Loss truncation | Yes (one-line change) | 0.005-0.01 | Medium |
| 5 | Vocabulary 2048 | Yes | Unknown | Low |

Interventions #1 and #3 can be combined in a single training run (new architecture trained on curated data with loss truncation). Intervention #2 is applied post-training and stacks on top of anything else.

The maximum plausible gain from combining all of the above: **0.03-0.08 BPB**, bringing the model from 1.11 toward **1.03-1.08 BPB**. The theoretical floor for web text compression is somewhere below 0.664 BPB (Chinchilla 70B on enwik9), but a 16 MB model will never approach that — the capacity gap is too large.
