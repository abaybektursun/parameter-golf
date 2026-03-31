# Deep Research: Optimal Architectures for Extreme Parameter-Constrained Language Modeling

## 1. Introduction

The OpenAI "Parameter Golf" challenge: train the best language model where compressed artifact (model weights + code) ≤ 16MB (16,000,000 bytes), training ≤ 10 min on 8×H100 GPUs. Metric: bits-per-byte (BPB) on FineWeb validation (lower = better). Baseline: 1.2244 BPB.

**Current SOTA: 1.1248 BPB** (PR #315: 11L Partial RoPE + LN Scale + Late QAT + XSA + EMA). Mean across 3 seeds: 1.1250.

---

## 2. Leaderboard (as of March 21, 2026)

| Score (BPB) | PR | Technique Stack | Notes |
|-------------|-----|-----------------|-------|
| **1.1248** | **#315** | **PR #287 + Partial RoPE (16/64) + LN Scale + Late QAT** | **NEW SOTA.** 3-seed mean: 1.1250 |
| 1.1254 | #338 | PR #315 + TTT (SGD, 3 epochs) | TTT adds only ~0.002 on this base |
| 1.1271 | #287 | PR #198 + XSA (last 4 layers) + EMA (β=0.997) | Previous SOTA |
| 1.1303 | #254 | PR #198 + full-weight TTT (SGD, 3 epochs) | TTT on SWA base |
| 1.1320 | #332 | **12L** + Gradient-Guided Quant (Int5/6/7) + Partial RoPE + LN Scale | Deepest model, mixed quant |
| 1.1354 | #290 | XSA + TTT + BatchOpt (no FA3) | |
| 1.1377 | #64 | XSA + EMA + TTT (lr=1e-4) | Low-LR TTT still negative |
| 1.1403 | #274 | Stride-32 + Warmdown/Muon tuning | stride-32 proven |
| 1.1436 | #303 | XSA + EMA + TTT (lr=2e-3) | **TTT negative interaction confirmed** |
| 1.1450 | #327 | TrigramHash + Partial RoPE (50%) + Per-Head Temp + stride-32 | Novel techniques |

---

## 3. The Proven SOTA Stack (PR #198 base, used by #254 and #265)

### Full Architecture (verified from code)
- 11-layer transformer, 512 dim, 8 heads, 4 KV heads (GQA)
- 3× MLP (hidden=1536) with **relu²** activation (`relu(x).square()`)
- **U-Net skip connections**: encoder/decoder split with learnable per-layer per-dim skip weights
- **SmearGate**: per-dim sigmoid gate blending current + previous token (~512 params)
- **resid_mix (Value Residual Learning)**: per-layer mixing between current residual `x` and original embedding `x0`
- Int6 QAT via STE (per-row symmetric, [-31,31]) on MLP + attention weights
- FP16 tied embeddings (exempt from quantization)
- Bigram hash embedding (2048 buckets, 128-dim)
- **Logit softcap** at 30.0: `30 * tanh(logits / 30)`
- **MTP infrastructure in code but DISABLED** (`MTP_NUM_HEADS=0`). Heads discarded at export.
- **Stochastic Weight Averaging** (~8 checkpoints, every 200 steps when LR < 0.5)
- Muon optimizer (5-step Newton-Schulz, momentum=0.99, WD=0.04) + AdamW for 1D params
- Orthogonal + muP-scaled initialization
- Seq_len 2048, NTK-aware RoPE, Flash Attention 3
- Sliding window eval (stride=64)
- zstd-22 compression
- 26.8M params → 15.7MB artifact, 7,412 steps in 600s (~81ms/step)

---

## 4. EMPIRICALLY PROVEN New Techniques (from PRs)

### Test-Time Training — CONFIRMED SOTA (PR #254, 1.1303 BPB)

Full-weight SGD adaptation on the validation data BEFORE scoring. This is the single biggest improvement discovered.

**How it works (PR #254 FarnsworthEngine):**
- After training completes and model is quantized+dequantized, run SGD on the full val set
- SGD: lr=0.002, momentum=0.9, **3 epochs** over val data
- **Freeze first 2 blocks** for stability (only adapt layers 2-10)
- Gradient clipping at 1.0
- Takes ~43s on 8×H100 (well within 10-min eval budget)
- Then run sliding window eval (stride=64) on TTT-adapted weights (~86s)
- Total eval: ~129s

**Measured impact:**

| Seed | Pre-TTT BPB | Post-TTT Sliding BPB | TTT Gain |
|------|-------------|---------------------|----------|
| 1337 | 1.1447 | **1.1303** | 0.0144 |
| 42 | 1.1449 | **1.1312** | 0.0137 |
| 7 | 1.1453 | **1.1323** | 0.0130 |
| **Mean** | | **1.1313** | **0.014** |

**Key detail:** TTT is NOT the same as the earlier LoRA TTT (PR #77, which only gained 0.003 BPB). Full-weight SGD with frozen early layers is 4-5× more effective than LoRA adaptation.

**Verdict: MUST USE. ~0.014 BPB for free (eval-time only, zero artifact cost).**

### Exclusive Self-Attention (XSA) — CONFIRMED SOTA (PR #287, 1.1271 BPB)

Removes self-attention bias in the deepest layers. Based on arXiv:2603.09078.

**How it works:**
- Standard attention has a bias where output vectors become too similar to value vectors of the same token (self-attention concentration increases with depth)
- XSA projects out the self-value component: `y = y - dot(y, normalize(v)) * normalize(v)`
- GQA-aware implementation uses reshape+broadcast instead of `repeat_interleave` (zero extra memory)
- Overhead: only ~2-3ms/step

**XSA layer count matters:**
- PR #265: XSA on last **3** layers → 1.1307 BPB
- PR #287: XSA on last **4** layers → **1.1271 BPB** (new SOTA)

**Paper: arXiv:2603.09078 (Exclusive Self Attention, 2026)**

**Verdict: MUST USE on last 4 layers. Part of new SOTA.**

### EMA — CORRECTED: Works at β=0.997 (PR #287, part of new SOTA)

**Our earlier debunking was wrong.** PR #201 tested EMA at β=0.995 and got worse results. But PR #287 uses EMA at **β=0.997** and it's part of the new SOTA (1.1271).

| PR | Method | Decay | BPB | vs PR #198 |
|----|--------|-------|-----|------------|
| #198 | Discrete SWA (8 checkpoints) | N/A | 1.1318 | baseline |
| #201 | LAWA-EMA | β=0.995 | 1.1551 | -0.023 WORSE |
| #287 | EMA (every step) | **β=0.997** | **1.1271** | **+0.005 BETTER** |

The β value is critical. 0.995 over-smooths; 0.997 preserves enough weight structure while still averaging out quantization noise.

**Verdict: USE EMA at β=0.997, replacing SWA. Part of new SOTA.**

### Partial RoPE — CONFIRMED SOTA (PR #315, 1.1248 BPB)

Apply rotary position embeddings to only a fraction of the head dimensions. Remaining dims use position-free attention.

- PR #315: **16 of 64 dims** (25%) → 1.1248 (new SOTA)
- PR #327: 32 of 64 dims (50%) → 1.1450 (different base)
- Zero new parameters. Pure architectural change.
- **Why it works:** Position-free dims enable content-based similarity matching regardless of distance, improving generalization. The 25% RoPE dims still provide enough positional signal.
- **Verdict: MUST USE. Part of new SOTA. `ROPE_DIMS=16`**

### LN Scale — CONFIRMED SOTA (PR #315, 1.1248 BPB)

Scale RMSNorm outputs by `1/sqrt(layer_idx + 1)`. Damps deeper layers' contributions.

- Zero new parameters. Pure scaling change.
- Stabilizes training in deep quantized networks by preventing later layers from dominating the residual stream.
- **Verdict: MUST USE. Part of new SOTA. `LN_SCALE=1`**

### Late QAT — CONFIRMED at 11L, HARMFUL at 12L

Activate Int6 STE fake-quantization only in the final ~4% of training (when lr_scale < 0.1). Based on arXiv:2505.14302.

- PR #315 (11L): **Cuts int6 degradation by 3×** with no cost to pre-quant quality
- PR #332 (12L): **Late QAT HURTS** — 1.1361 vs 1.1321 without. The ~7ms/step overhead at 12 layers costs ~770 training steps, and the lost model quality exceeds the quantization improvement.
- **Verdict: USE at 11L. Skip at 12L. `LATE_QAT=1 QAT_THRESHOLD=0.1`**

### Gradient-Guided Adaptive Quantization (PR #332, 1.1320 BPB)

Instead of uniform Int6, use gradient sensitivity to assign per-tensor bit-width:
- Top 10% most sensitive tensors: Int7 (63 levels)
- Middle 70%: Int6 (31 levels)
- Bottom 20%: Int5 (15 levels)
- Saves ~1MB vs uniform Int6 → enables a **12th layer** (MLP narrowed to 1408)

PR #332: 12 layers, 27.6M params, 1.1320 BPB (mean across 3 seeds: 1.1320)

**Verdict: Interesting direction. 12L at 1.1320 beats old 11L SOTA (1.1271) but trails new 11L SOTA (1.1248). Depth vs width tradeoff not yet clearly resolved.**

### TTT on PR #315 Base — MARGINAL (PR #338, 1.1254)

PR #338 adds TTT (SGD, lr=0.002, 3 epochs, freeze 2 blocks) to PR #315's stack:
- PR #315 without TTT: mean 1.1250
- PR #338 with TTT: mean 1.1256

**TTT makes it ~0.001 WORSE on this base.** The Partial RoPE + LN Scale + Late QAT stack has further reduced the headroom for eval-time adaptation. TTT's contextual gains are even more redundant now.

### PPM-C Eval-Time Context Mixer (PR #283) — PROMISING, UNTESTED ON SOTA

Classical Prediction by Partial Matching mixed with neural model predictions at eval time.

- 95% neural + 5% PPM-C (order-2) = ~0.015 BPB improvement on 2K subset
- Zero learned parameters, zero artifact cost, ~60 lines of code
- Only tested on baseline (1.2244), not on current SOTA stack
- **Verdict: NEEDS TESTING on SOTA. If the 0.015 BPP gain holds, this is huge and free.**

### Stride-32 Eval — CONFIRMED (PR #274, 1.1403 BPB)

PR #274 switched from stride-64 to stride-32 on the SOTA base and gained ~0.003 BPB. Each scored token now has even more overlapping context. Eval takes longer but eval time is free.

**Verdict: USE stride-32. Free ~0.003 BPB.**

### Backout Module (PR #295, 1.1477 BPB)

Learned residual subtraction: `output = output - λ * x_mid` where x_mid is the cached residual stream at the middle layer. Prevents deep layer corruption from contaminating the LM head's access to clean mid-level features.

- Part of PR #295's stack (1.1477) alongside Int5/Int6 mixed quant + BigramHash(10240)
- Hard to isolate Backout's individual contribution
- **Verdict: NEEDS ISOLATED ABLATION. Interesting but unproven in isolation.**

### Stride-OGD (PR #241) — NOVEL EVAL TRICK

Online gradient descent on a 1024-dim vocab bias vector, updated every stride (64 tokens).
- Zero artifact cost (bias is computed at eval time)
- 16× faster feedback than TTT LoRA
- Working code provided but no BPB numbers on SOTA stack yet
- **Verdict: NEEDS TESTING. Could stack on top of full-weight TTT.**

---

## 5. EMPIRICALLY DEBUNKED Recommendations

### EMA at β=0.995 — DEBUNKED, but β=0.997 WORKS (PR #201 vs PR #287)

PR #201 tested EMA at β=0.995: 1.1551 BPB (0.023 worse). We incorrectly concluded EMA was dead.

**PR #287 then proved EMA at β=0.997 BEATS SWA: 1.1271 BPB (new SOTA).**

The β value is extremely sensitive. 0.995 decays too fast (over-smooths). 0.997 is the sweet spot.

**Verdict: EMA WORKS at β=0.997. Part of the new SOTA.**

### SWA Under Default Warmdown — NUANCED (PR #199)

PR #199 ran controlled ablations:
- SWA (73 snapshots) under default warmdown: **no improvement** (+0.0004 BPB)
- SWA only helps when warmdown is long enough to produce diverse snapshots
- PR #198 uses 3000-step warmdown (40%) — this IS long enough and SWA helps there
- **Key insight: SWA effectiveness depends on warmdown length. Short warmdown → SWA is useless.**

### Doc-Isolated Eval — HARMFUL at stride=64 (PR #199)

| | Flat-stream | Doc-isolated | Delta |
|--|---|---|---|
| Post-quant BPB | **1.1933** | 1.2015 | +0.0086 (WORSE) |

At stride=64, tokens already have 960+ context. Doc isolation forces start-of-document tokens to lose ALL context (window resets), which hurts more than cleaner boundaries help.

**Verdict: Do NOT use doc-isolated eval with stride=64.**

### SWA Can Reverse Quantization Gap (PR #238)

With enough SWA checkpoints (84), post-quant BPB is actually LOWER than pre-quant (1.5164 vs 1.5536). SWA smoothing eliminates quantization-sensitive outliers so effectively that quantization acts as beneficial regularization.

### Int5 MLP — RISKY, NET NEGATIVE WITHOUT TTT (PR #219, #238, #264)

**PR #219 (12L Int5-MLP + Int6-Attn):** 1.1541 BPB — worse than 11L Int6 (1.1318)
- 12 layers but 107ms/step (vs 81ms) → only 5,590 steps (vs 7,412)
- The extra layer doesn't compensate for fewer training steps + quantization noise

**PR #238:** Int5 is "catastrophic for undertrained models" — quant gap explodes from 0.3 to 1.4 BPB (4.5×)

**PR #264 (Int5-MLP + TTT):** 1.1455 BPB — better than pure Int5 but still worse than SOTA
- TTT recovers 0.005 BPB on Int5 model (1.1507 → 1.1455)

**Verdict: Int5 MLP saves 2.16MB and enables 12 layers, but slower steps + quantization noise make it net negative. Only viable if TTT can recover enough. Currently NOT worth it — standard Int6 at 11 layers + TTT (1.1303) beats 12L Int5 + TTT (1.1455) by 0.015 BPP.**

### Curriculum Learning — MODEST (PR #242)

PR #242 (Crystal Curriculum, TF-IDF scoring):
- 0.009 BPB improvement over baseline on 1×H100
- BUT: 48 fewer training steps due to batch scoring overhead
- Only tested on older baseline, not current SOTA

**Verdict: Modest gains, real overhead. Low priority vs TTT/XSA.**

### Depth Recurrence — PENDING (PR #268)

2 prelude + 1 shared (looped 7×) + 2 coda = 5 unique blocks, 11 effective depth. No compute results yet.

### Linearized Neural Memory + TTT (PR #182)

Titans-style causal linear attention memory + LoRA TTT:
- 1.1844 with TTT, 1.2163 without (0.032 BPB TTT gain on 10L model)
- Not competitive with 11L SOTA but confirms TTT works across architectures

---

## 6. Techniques Confirmed Optimal (Do Not Change)

| Technique | Why Optimal | Evidence |
|-----------|------------|---------|
| relu² activation | 33% fewer MLP params than SwiGLU, >90% sparsity | Round 2 research |
| GQA-4 (4 KV heads) | d=512 needs head diversity; MQA destroys it | Round 2 research |
| Logit softcap 30.0 | Protects Int6 from outlier spikes | Gemma 2 (arXiv:2408.00118) |
| x0 residual mixing | Prevents attention concentration in narrow transformers | ResFormer (arXiv:2410.17897) |
| ~40% warmdown (trapezoidal) | Required for EMA + flat basin for quantization | Round 2 research + PR #199 |
| zstd-22 | Near-optimal for quantized weights | Exp 003 (LZMA worse) |
| Int6 QAT (uniform) | Int5 causes more harm than depth benefit | PR #219, #238, #264 |
| FP16 tied embeddings | Most quantization-sensitive tensor | All top PRs |
| **EMA (β=0.997)** | **Beats SWA by 0.005 BPB; β=0.995 fails but 0.997 works** | **PR #287, #315 (SOTA)** |
| **XSA on last 4 layers** | **Removes self-attention bias, zero new params, ~2ms/step** | **PR #287, #315 (SOTA)** |
| **Partial RoPE (16/64 dims)** | **Position-free dims improve generalization. Zero params.** | **PR #315 (SOTA): ~0.002 BPB gain** |
| **LN Scale (1/√(layer+1))** | **Damps deep layer contributions, stabilizes quantized training** | **PR #315 (SOTA)** |
| **Late QAT (4% of training, 11L only)** | **Cuts Int6 degradation 3× with no pre-quant cost** | **PR #315 (SOTA). Harmful at 12L (PR #332)** |
| Flat-stream eval (not doc-isolated) | Doc-isolation hurts 0.009 BPB at stride=64 | PR #199 |

---

## 7. Eliminated Approaches (Full List)

| Approach | Status | Evidence |
|----------|--------|---------|
| EMA at β=0.995 | DEBUNKED (but β=0.997 works!) | PR #201 vs PR #287 |
| Doc-isolated eval | DEBUNKED | PR #199: 0.009 BPB worse at stride=64 |
| Int5 MLP for depth | NET NEGATIVE | PR #219: 12L Int5 (1.1541) < 11L Int6 (1.1318) |
| BTT Matrices | ELIMINATED | Muon incompatible, wall-clock worse |
| LZMA/brotli | DEBUNKED | Exp 003: LZMA worse than zlib on quant weights |
| MQA (1 KV head) | ELIMINATED | Destroys expressivity at d=512 |
| BitNet b1.58 | ELIMINATED | No H100 support, conflicts Int6 |
| Mamba/SSM | ELIMINATED | Requires full rewrite, worse at small scale |
| minGRU/minLSTM | ELIMINATED | Complete architectural rewrite |
| Neuroevolution | ELIMINATED | Incompatible with 10-min constraint |
| Hypernetworks/KANs | ELIMINATED | Too slow to train |
| AQLM | ELIMINATED | Can't train codebooks from scratch |
| ALBERT weight sharing | ELIMINATED | Scales negatively with compute |
| **MTP (k=1)** | **CONFIRMED HARMFUL** | **+0.013 BPB worse, tested empirically. Capacity collapse at 26M params.** |
| Full-weight TTT at low LR on XSA+EMA | CONFIRMED HARMFUL | PR #64: lr=1e-4 still +0.011 worse (1.1377 vs 1.1271) |
| Full-weight TTT on PR #315 base | MARGINAL/NEGATIVE | PR #338: mean 1.1256 vs PR #315 mean 1.1250 (+0.001 worse) |
| **Bias-Only TTT (1D params only)** | **CONFIRMED NEGATIVE** | **Tested: 1.1273 vs 1.1272 without TTT (+0.0001 worse). Same direction as full TTT, just smaller.** |
| Late QAT at 12 layers | HARMFUL | PR #332: costs ~770 steps, net +0.004 worse (1.1361 vs 1.1321) |
| Knowledge Distillation | UNCERTAIN | V=1024 makes overhead trivial, but no one has tested |

---

## 8. Recommended Action Plan (Updated with Deep Research Round 3)

### CRITICAL UPDATE: XSA+TTT is a NEGATIVE INTERACTION (PR #303)

PR #303 proved XSA + EMA + full-weight TTT = 1.1436 (0.016 BPB WORSE than XSA + EMA alone). Full-weight SGD destroys the smooth EMA weight surface and redundantly targets the same local context signal as XSA.

### THE SOLUTION: Bias-Only TTT (BitFit-style) — Deep Research Top Recommendation

The deep research identified the resolution to the XSA+TTT paradox: **freeze all 2D matrices, adapt only 1D parameters** (biases, normalization scales, SmearGate gates, resid_mix vectors).

**Why this works:**
- Full-weight TTT fails because it warps the topological geometry of the EMA-smoothed parameter space
- Bias-only adaptation is a **pure affine translation** — it shifts activation hyperplanes without rotating or scaling them
- The EMA-smoothed 2D matrices (where quantization equilibrium lives) remain untouched
- SmearGate (~512 params) and resid_mix (~512 params per layer) are temporal routing parameters — adapting them per-document recalibrates the model's context window without touching the semantic core
- Total adapted params: <2,000 across the entire network
- Learning rate must be much lower than standard TTT: ~1e-4 to 1e-5 (vs 0.002)

**Expected gain: ~0.012 BPB** (research estimate). This would push 1.1271 → ~1.115.

**Implementation:** Modify the TTT eval loop to set `requires_grad=False` on all 2D parameters, leaving only biases, norms, SmearGate, and resid_mix trainable.

### WARNING: MTP CONFIRMED HARMFUL — Empirically Tested

The deep research warned MTP would cause capacity collapse at 26M params. **We tested it and the research was right:**

| Metric | MTP k=1 | No MTP (prev run) | PR #287 |
|--------|---------|-------------------|---------|
| Sliding BPB s64 | 1.1398 | 1.1272 | 1.1271 |

**+0.013 worse.** MTP cost ~87 steps (7081 vs 7168) due to slower step time, and the auxiliary loss competed with the primary loss for gradient capacity in the tight 600s budget.

**Do NOT enable MTP_NUM_HEADS=1.** The code infrastructure exists but it's a trap at this model scale.

**Alternatives that might work at small scale:**
1. **Curriculum MTP**: Disable MTP for first 50% of training, gradually activate in second half after core representations stabilize — avoids early capacity saturation
2. **Token Order Prediction (TOP)**: Instead of predicting exact future tokens, use a learning-to-rank loss to order upcoming tokens by proximity. Much lighter than MTP. Needs only a single lightweight head, ~50 lines of code. Expected: ~0.005 BPB.

### Updated Action Plan (Single Branch — XSA + EMA + Non-Destructive Adaptation)

| Rank | Technique | Expected BPB Gain | Confidence | EMA/XSA Safe? | Effort |
|------|-----------|-------------------|------------|---------------|--------|
| ~~1~~ | ~~Bias-Only TTT~~ | ~~0.012~~ | ~~DEAD~~ | ~~YES~~ | **TESTED: +0.0001 worse. Research was wrong — even affine-only adaptation can't help XSA+EMA models.** |
| **2** | **PPM-C probability ensembling** (classical n-gram blended with neural probs at eval) | ~0.007 | 65% | **YES** — doesn't touch weights at all | High: needs optimized Trie implementation |
| **3** | **Token Order Prediction** (auxiliary training loss, discarded at export — NOT MTP) | ~0.005 | 60% | **YES** — training signal only | Moderate: ~50 lines, dual-loss training |
| **4** | **Stride-16 eval** (instead of stride-64) | ~0.004 | 85% | **YES** — pure eval config | Trivial: 1 env var |
| **5** | **2:4 Structured Sparsity** (50% weights zeroed → massive compression → 15-16 layers) | ~0.008 | 50% | **YES** — training-time change | High: rewrite QAT + hyperparameter recalibration |
| **6** | **Late QAT** (activate Int6 STE at 85% of training, not from start) | ~0.003? | 55% | **YES** — training-time change | Low: add step threshold |

### The Path to Beat 1.1271

**Step 1 (immediate, highest confidence):** Reduce eval stride to 16. One env var: `EVAL_STRIDE=16`. Expected: ~1.1231.

**Step 2 (highest leverage):** Implement Bias-Only TTT on PR #287. Freeze all 2D matrices, adapt only SmearGate + resid_mix + biases + norms via SGD at lr=1e-4. Expected: ~1.115.

**Step 3 (probability space):** Add PPM-C mixer. Build an order-5 PPM-C Trie at eval time, blend with neural probs (dynamic α based on n-gram match confidence). Expected: ~1.108.

**Step 4 (training improvement):** Replace the disabled MTP with Token Order Prediction (TOP) auxiliary loss. Denser gradient signal without capacity collapse. Expected: ~1.103.

**Best realistic case: ~1.10-1.11 BPB** (Bias-Only TTT + PPM-C + stride-16 + TOP all working on XSA+EMA base).

### CRITICAL: XSA + TTT Are REDUNDANT (PR #303)

**PR #303 tested XSA + EMA + TTT and it's WORSE than XSA + EMA alone:**

| Configuration | val_bpb | Delta |
|---|---|---|
| XSA + EMA (PR #287) | **1.1280** | baseline |
| XSA + EMA + TTT (PR #303) | 1.1436 | **+0.016 WORSE** |
| TTT only (PR #254) | 1.1313 | +0.003 worse |

**Why:** XSA and TTT both target local context modeling. XSA removes self-information from attention; TTT adapts weights to local val patterns. Stacking them double-counts the same signal. Worse: TTT's SGD updates **disrupt the smooth EMA weight landscape** that the model was trained to produce.

**This means the frontier splits into two incompatible branches:**
1. **XSA + EMA branch** (1.1271) — training-time improvements only
2. **TTT branch** (1.1303) — eval-time adaptation on SWA-based models

They cannot be combined naively. PR #302 tried a decay-prior TTT variant (pulls weights back toward original) and got 1.1520 — still not competitive.

### Competitive Landscape (Who's Working on What)

**PR #296 (1.1645 BPB):** Reptile meta-learning for TTT — 0.011 BPB improvement over naive TTT (10× better). But this was tested on non-XSA models. Unknown if Reptile helps the XSA+EMA branch.

**PR #297 (1.1629 BPB):** Late STE QAT — activate QAT at 85% of training. Avoids corrupting Muon momentum early. Only tested on 9L.

**PR #302 (1.1520 BPB):** Online causal TTT with decay prior + Reptile meta-learning + XSA. Sophisticated TTT variant but still trails SOTA. Suggests TTT needs fundamental rethinking to work with XSA+EMA.

### What Might Still Work (as of March 20, 2026)

| Combination | Status | Notes |
|-------------|--------|-------|
| XSA + EMA + stride-32 (no TTT) | Untested | Stride-32 adds ~0.003 (PR #274). Safe to combine. |
| XSA + EMA + MTP training | Untested | MTP is a training signal, orthogonal to XSA/EMA |
| XSA + EMA + PPM-C mixer | Untested | PPM-C is eval-time probability blending, not weight adaptation |
| TTT branch + stride-32 | Untested | Should be safe — stride only changes eval windowing |
| XSA + EMA + Late QAT | Untested | Late QAT is training-time, should be compatible |
| XSA + EMA + modified TTT (decay prior, frozen layers) | PR #302 tried, 1.1520 | Not competitive yet |
| **XSA (5+ layers?)** | Untested | PR #287 uses 4 layers. More might help. |

### Depth Recurrence Status
~30 PRs have tried depth recurrence. Best result: PR #187 at 1.1709 (encoder blocks run 2×). All recurrence approaches are 0.04+ BPB behind the non-recurrence SOTA. Key failure modes: speed penalty (fewer training steps), specialization loss (shared layers can't differentiate). PR #296 found 13 layers beats 10 layers despite 23% fewer steps — depth matters, but through unique layers + Int6 compression, not weight sharing.
