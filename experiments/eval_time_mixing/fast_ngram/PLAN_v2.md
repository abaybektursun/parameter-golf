# Plan: Single-Pass Causal N-gram + Neural LM

## 0. Lessons Learned (What Failed and Why)

### 0.1 Two-phase architecture → Causality violation
Plan v2 proposed: Phase 1 (all n-gram), Phase 2 (all GPU). This structure is illegal
under competition rules. PRs #986, #1026 were rejected for this pattern. The eval must
be a single left-to-right pass. PR #1145 does this correctly: one interleaved loop.

### 0.2 PPM logistic mixer → Hash collisions produce garbage probabilities
Hash-based PPM (flat arrays, no stored keys) cannot produce valid probability
distributions. Collisions inflate pair_count, giving P_ppm(target) >> true value.
The logistic mixer amplified this error → -0.21 BPB (catastrophic).

### 0.3 Exponential tilt with naive hints → PPM redundant with neural model
Switched to exponential tilt (always normalized). But the PPM and neural model
learn the SAME patterns from the SAME token stream. When one is uncertain, so is
the other. At positions where hints would matter most (neural p < 0.05), hint
accuracy is 0.7%. Result: +0.048 BPB (still hurts).

### 0.4 The mechanistic insight that changes everything
Two internal studies (loss decomposition + SVD/quant sensitivity) reveal WHERE
the neural model fails: numbers (2.09 BPB), punctuation (2.38 BPB), whitespace
(3.01 BPB). N-gram has NO advantage in any of these — they're inherently
unpredictable from local context.

**The n-gram's value is not being a better language model. It's providing
information the neural model CANNOT access due to architectural constraints.**

## 1. The Neural Model's Exploitable Structural Limitations

### 1.1 Fixed weights → can't adapt to the current document

The model sees "Johannesen" at position 200, assigns p=0.001 (rare name, low
training prior). Sees it again at position 800 — still p=0.001 (same weights).
An online n-gram at position 800 assigns p=0.3 (seen this exact context before
in THIS document). This is a real signal the neural model CANNOT produce.

**N-gram design implication:** The n-gram's value comes from ONLINE ADAPTATION
to the validation stream. Patterns that repeat within the same document or across
nearby documents are the primary signal. The n-gram is a memorizer of recent
context, not a general language model.

### 1.2 2048-token window → blind to long-range repetitions

A phrase at position 1000 and again at position 5000: the neural model at 5000
has zero memory of position 1000 (outside the 2048-token window). The n-gram's
causal state has NO window limit — it remembers everything from position 0.

**N-gram design implication:** Beyond-window repetitions (>2048 tokens apart) are
a signal ONLY the n-gram can capture. Use moderate orders (3-6) for these — order
16 requires 16 consecutive tokens to match, which almost never happens across
2000+ token gaps. Order 4 can match "in the United States" from 3000 tokens ago.

### 1.3 BigramHash 3072 already handles orders 1-2

The model has a built-in BigramHashEmbedding (3072 vocab, 112 dim). This captures
bigram statistics during the forward pass. Our n-gram at orders 1-2 is COMPLETELY
REDUNDANT with this built-in feature.

**N-gram design implication:** Skip orders 1-2 entirely. Start PPM backoff at
order 3. Free the table capacity for orders 3-16 where the model has no built-in
n-gram support.

### 1.4 Generalization vs memorization

The neural model learns smooth distributions: "after context X, tokens A/B/C are
likely." For an exact local repetition (same 8-token context seen 5 times in this
document), the n-gram says "token A appeared all 5 times" with near-certainty.
The neural model still spreads mass across A/B/C. The n-gram's SHARPNESS on
exact matches is the complementary signal.

**N-gram design implication:** Only fire when the n-gram is very confident
(threshold ≥ 0.8). When unsure, stay silent. Most positions (~90-95%) should
get NO hint. The n-gram is a surgical specialist, not a general-purpose model.

## 2. The Surgical Specialist N-gram Architecture

### 2.1 What the n-gram should do

```
Old thinking: "Build a better probability model and mix it with neural"
New thinking: "Find positions where the n-gram KNOWS something the neural
              model CAN'T know, and only intervene there"
```

The n-gram fires on three types of signal:

**Signal A — Document-specific exact repetitions (order 8-16):**
A phrase, name, or pattern that appeared earlier in the current document.
The neural model treats each occurrence identically (fixed weights).
The n-gram memorizes and predicts confidently on repeat occurrences.

**Signal B — Beyond-window repetitions (order 3-6):**
Patterns that repeat with >2048 token gaps. The neural model's window can't
bridge this. The n-gram's unlimited causal memory can.
Use moderate orders — long enough to be discriminative, short enough to match
across large gaps.

**Signal C — Subword completion (within-word PPM, order 1-3):**
After the BPE subword "believ", the continuation is almost certainly "able"
or "ed" or "ing". The neural model distributes mass broadly. A within-word
PPM (reset at word boundaries) can be very sharp. This is the one place
where short-order n-grams beat the neural model — but only within words.

### 2.2 What the n-gram should NOT do

- Don't predict at orders 1-2 (redundant with BigramHash)
- Don't fire when confidence < 0.8 (neural model is probably better)
- Don't try to produce calibrated probabilities (hash collisions corrupt them)
- Don't fire on every position (most tokens are not repetitive)
- Don't compete with the neural model on general language modeling

### 2.3 Per-order confidence thresholds + minimum count

Higher orders match rarely but when they match, predictions are very sharp.
Lower orders match often but predictions are noisy. Two filters are needed:

1. **Confidence threshold:** top_count / ctx_count must exceed this
2. **Minimum ctx_count:** the context must have been seen at least this many
   times. With hash collisions (noise floor ~15 per 4M bucket), a single
   occurrence is unreliable. Require count ≥ 3 at all orders.

| Order | Threshold | Min count | Why |
|-------|-----------|-----------|-----|
| 3-4 | 0.90 | 5 | Common matches, need high bar + repetition |
| 5-8 | 0.80 | 3 | Good discriminative power |
| 9-12 | 0.70 | 2 | Rare matches, inherently sharp |
| 13-16 | 0.60 | 2 | Very rare, near-certain when they occur |

**Backoff floor: order 8.** The PPM tries orders 16, 15, ..., 8. If none match
above their threshold, produce NO HINT. Do NOT back off to orders 3-7 for hint
selection — the diagnostic showed low-order hints hurt. Orders 3-7 are still
maintained in the PPM state (for the PPMC escape probability math) but they don't
produce hints on their own. Only orders 8-16 produce hints.

Exception: within-word PPM (orders 1-3, reset at word boundaries) CAN produce
hints because within-word patterns are inherently sharp (BPE completion is
mechanical, not statistical).

## 3. Single-Pass Interleaved Architecture

### 3.1 The legal eval structure (matching PR 1145)

```
for each chunk of ~131K tokens:
    C++ n-gram processes chunk's tokens causally (score-first, update-after)
    for each batch of 32 windows within this chunk:
        GPU forward pass → neural logits + NLL
        GPU tilt: boost hint token in neural distribution
        Accumulate loss + bytes
```

Single left-to-right pass. N-gram state at position t uses only tokens 0..t-1.
GPU forward is causal (flash attention with causal=True). Legality matches PR #1145.

### 3.2 The exponential tilt (always normalized)

The n-gram produces hint_token h. The GPU applies:
```
p'(h) = exp(β) · p(h) / Z
p'(a) = p(a) / Z    for a ≠ h
Z = 1 + p(h) · (exp(β) - 1)
```
Σ p'(a) = 1 always. Hash collisions can only affect WHICH token gets boosted,
not the validity of the resulting distribution.

For scoring target token y:
```
if y == h:  mixed_nll = neural_nll - β + log(Z)   (hint was correct → lower NLL)
if y != h:  mixed_nll = neural_nll + log(Z)        (hint was wrong → slightly higher NLL)
```

### 3.3 Adaptive β within the single pass

After each chunk, adapt β based on hit rate:
```
hit_rate_ema = 0.99 * hit_rate_ema + 0.01 * chunk_hit_rate
β = base_β * clamp(hit_rate_ema / target_hit_rate, 0.3, 3.0)
```
Each chunk's adaptation depends only on PREVIOUS chunks. Legal.

### 3.4 GPU-side computation (two gathers, no softmax)

```python
logit_hint = logits.gather(-1, hint_tokens.unsqueeze(-1)).squeeze(-1)
logit_target = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
logsumexp = nll + logit_target   # reuse cross_entropy's internal computation
log_p_hint = logit_hint - logsumexp
p_hint = log_p_hint.exp()
Z = 1.0 + p_hint * (betas.exp() - 1.0)
mixed_nll = nll + has_hint * (Z.log() - betas * (targets == hints).float())
```

Two gathers + elementwise ops. <0.1ms on GPU. No full softmax needed.

## 4. C++ Implementation: ContextMixer Class

### 4.1 Key design decisions

- **PPM orders 3-16** (skip 1-2, redundant with BigramHash)
- **Flat hash tables** (no stored keys — we only need argmax, not probabilities)
- **Per-order confidence thresholds** (section 2.3)
- **Within-word PPM** orders 1-3 (reset at word boundaries)
- **Word-start model** (word-level bigram)
- **CPU byte counting** (eliminate GPU LUT round-trips)
- **One method: `get_hints_batch(positions, out_hints, out_betas)`**
  Processes positions sequentially, returns hint_token + beta per position.
  Updates all model state causally after each position.

### 4.2 Table sizing

Orders 3-16 have very different match rates. High orders (10-16) almost never
match because 10+-grams are nearly always unique in 62M tokens. Don't waste
memory on them. Use tiered allocation:

**Tier 1 — Orders 3-7 (PPMC state, no direct hints):**
These orders maintain PPM escape statistics but don't produce hints (backoff
floor is order 8). They need ctx + distinct tables but NOT large pair tables.
- ctx tables: 1M buckets (20 bits) × uint16 = 2 MB per order
- pair tables: 1M buckets (20 bits) × uint16 = 2 MB per order
- distinct tables: 1M buckets (20 bits) × uint16 = 2 MB per order
- Per order: 6 MB × 5 orders = 30 MB

**Tier 2 — Orders 8-12 (hint producers, moderate match rate):**
These need larger pair tables for argmax quality.
- pair tables: 4M buckets (22 bits) × uint16 = 8 MB per order
- ctx + distinct: 1M each = 4 MB per order
- Per order: 12 MB × 5 orders = 60 MB

**Tier 3 — Orders 13-16 (hint producers, rare matches):**
Share a SINGLE set of tables across all 4 orders (mix order number into hash).
Almost no entries — shared table avoids wasting memory.
- Shared pair table: 2M buckets (21 bits) × uint16 = 4 MB
- Shared ctx + distinct: 512K each = 2 MB
- Total tier 3: 6 MB

Plus within-word (3 orders × 6 MB) + word-start (6 MB) = 24 MB.
**Total: ~120 MB.** Fits in Xeon L3 (120 MB). Hot working set (tiers 1-2)
fits in ~90 MB.

Runtime adaptive: detect L3 size via /sys/devices/system/cpu/cpu0/cache/index3/size.
On EPYC Genoa (32 MB L3 per CCD): reduce to 18-bit pair tables (1M buckets).
On Xeon 8462Y+ (120 MB L3): use full 22-bit pair tables.

### 4.3 Compile flags

```cmake
target_compile_options(fused_expert_ext PRIVATE
    -O3 -march=native -funroll-loops -fno-math-errno -ffinite-math-only)
```
`-march=native` auto-enables AVX-512 on both Sapphire Rapids and Genoa.
`-fno-math-errno -ffinite-math-only` give ~2× faster exp/log.

### 4.4 Hugepages

```cpp
#ifdef __linux__
madvise(table_ptr, table_bytes, MADV_HUGEPAGE);  // hint, no failure mode
#endif
```
Reduces TLB misses from ~95% to ~0% on Linux. No-op on macOS.

## 5. Timing

### 5.1 Per-chunk breakdown (the real pipeline)

Within PR 1145's chunk-based structure, the n-gram processes ALL tokens in the
chunk BEFORE the GPU processes that chunk's windows. This is NOT overlapped:

```
Per chunk (~131K tokens, ~64 GPU batches):
  C++ n-gram (serial, all 131K tokens):  ~65ms  (131K × 0.5µs)
  GPU batches (64 × 12ms):               ~768ms (1×H100)
  GPU tilt per batch:                     ~0.1ms × 64 = ~6.4ms
  ─────────────────────────────────────
  Per chunk total:                        ~839ms
```

The n-gram's 65ms per chunk is SERIAL overhead — not hidden behind GPU.
Over ~474 chunks: 65ms × 474 = ~31s of n-gram serial time.

Total = GPU forward + n-gram serial + tilt overhead:
  1×H100: 364s + 31s + 3s = ~398s
  8×H100: 45s + 31s + 0.4s = ~76s

### 5.2 Total eval time

| GPUs | Neural eval | N-gram overhead | Total | Budget |
|------|-------------|-----------------|-------|--------|
| 1 | 364s | ~32s (8.8%) | ~396s | 66% |
| 2 | 182s | ~16s | ~198s | 33% |
| 4 | 91s | ~8s | ~99s | 17% |
| 8 | 45s | ~4s | ~49s | 8% |

All within 600s. Freed budget → more training steps or SLOT.

## 6. Expected BPB Gain

| Signal source | Est. BPB gain | Positions affected |
|---------------|---------------|--------------------|
| Document-specific repetitions (order 8-16) | 0.001-0.003 | ~2-5% |
| Beyond-window repetitions (order 3-6) | 0.0005-0.001 | ~1-3% |
| Subword completion (within-word) | 0.0005-0.001 | ~3-5% |
| **Total** | **0.002-0.005** | **~5-10%** |

Conservative. PR 1145 gets 0.003-0.004 with a simpler approach. Our PPM backoff
with per-order thresholds should match or slightly exceed this.

The main value is SPEED: 481s → ~50s frees ~430s of budget.

## 7. Competition Legality

| Requirement | Status |
|---|---|
| N-gram state at position t uses only tokens 0..t-1 | ✓ Score-first, update-after |
| Neural model is causal (causal attention mask) | ✓ flash_attn causal=True |
| Single left-to-right pass | ✓ Interleaved chunk-based loop |
| No batch statistics from future positions | ✓ Each chunk independent |
| Tilt distribution properly normalized | ✓ Z normalizes by construction |
| No external data during eval | ✓ Only val_tokens + model weights |
| Adaptive β uses only past results | ✓ EMA from previous chunks |

Matches PR #1145's accepted legality argument.

## 8. Implementation Files

| File | Purpose |
|------|---------|
| `fused_expert_blend.cpp` | ContextMixer class (PPM 3-16 + within-word + word-start) |
| `CMakeLists.txt` | Build with -O3 -march=native optimization flags |
| `test_fused.py` | Correctness + performance tests |
| `eval_fused.py` | Single-pass interleaved eval driver |

## 9. Key References

- [PPM escape formulas](https://en.wikipedia.org/wiki/Prediction_by_partial_matching)
- [Context mixing](https://en.wikipedia.org/wiki/Context_mixing)
- [PAQ (Matt Mahoney)](https://mattmahoney.net/dc/paq.html)
- [cmix](https://www.byronknoll.com/cmix.html)
- [Secondary estimation: PPM SEE to PAQ APM](http://cbloomrants.blogspot.com/2018/05/secondary-estimation-from-ppmz-see-to.html)
- experiments/model_analysis/weights/loss_decomp_results.txt
- experiments/model_analysis/weights/svd_results.txt
- experiments/model_analysis/weights/quant_sensitivity_results.txt
- experiments/model_analysis/NEXT_STEPS.md
