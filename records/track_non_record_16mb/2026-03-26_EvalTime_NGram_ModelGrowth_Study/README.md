# Non-Record: Eval-Time N-gram Mixing and the Unbounded Model Growth Problem

**Author:** abaybektursun | **Date:** 2026-03-26 | **Track:** Non-record study

This submission is not a leaderboard entry. It is a study of eval-time n-gram caching — a technique that reduces BPB from 1.11 to 0.38 while preserving strict causality, costing zero artifact bytes, but growing the effective model to 17x the artifact limit at eval time. We present results, explain why this creates a dilemma for the competition, and propose rule clarifications.

---

## Results

All runs use the PR #549 base model (~1.1194 BPB, 11L/512d, ~16MB artifact). Single GPU, stride=64, FineWeb val (62M tokens).

| Config | BPB | Eval-time state | Effective model | Time |
|--------|----:|----------------:|----------------:|-----:|
| Neural only (int6 quantized, leaderboard) | 1.1142 | 0 MB | 16 MB | 606s |
| Neural only (float, pre-quant) | 1.1109 | 0 MB | 16 MB | 606s |
| Pure n-gram, no neural model | 1.0615 | 192 MB | 192 MB | 535s |
| Fixed 7-gram, alpha=0.40 | 0.5234 | 192 MB | 208 MB | 824s |
| Backoff 2-7, alpha=0.40 | 0.4923 | 192 MB | 208 MB | 1079s |
| Backoff 2-7, entropy-adaptive alpha | 0.6535 | 192 MB | 208 MB | 1114s |
| **Backoff 2-9, order-adaptive entropy** | **0.3779** | **256 MB** | **272 MB** | **1234s** |

The n-gram cache alone — with no neural model — beats the 27M-parameter transformer (1.06 vs 1.11 BPB). Combined, it cuts BPB by 66%.

### 8-GPU results with all-reduce sync (EXP-11)

These results fit within the 600s competition eval budget. All-reduce sync cost: 1.6–2.0s total.

| Config | BPB | Time | Cache | Sync cost |
|--------|----:|-----:|-------|-----------|
| Neural only (8-GPU) | 1.1130 | 110s | None | — |
| Backoff 2-7, α=0.40 | 0.4941 | 401s | Global (all-reduce) | 1.6s |
| Backoff 2-9, α=0.40 | 0.4548 | 500s | Global (all-reduce) | 1.9s |
| Backoff 2-7, **α=0.80** | **0.3942** | 939s | Global (all-reduce) | ~2.0s |

Alpha sweep (8-GPU, backoff 2-7): α=0.20 → 0.6180, α=0.40 → 0.4941, α=0.60 → 0.4263, α=0.80 → 0.3942. Higher alpha is monotonically better — the opposite of PR #727's finding. With a global cache, the n-gram is reliable enough that the model should defer to it more, not less.

### What the n-gram cache is

After each token is scored by the neural model, the token and its preceding context are inserted into hash tables. When a future token's context matches a previously seen n-gram, the cached frequency estimate is mixed with the neural prediction:

```
p_mix = (1 - alpha) * p_neural + alpha * p_ngram
```

The tables are built exclusively from already-scored tokens. No future tokens are accessed. Strict causality is preserved.

### What the n-gram cache costs

| Config | Hash table memory | Formula |
|--------|------------------:|---------|
| Orders 2-7 (6 orders) | 192 MB | 6 orders x 2 tables x 4M buckets x 4 bytes |
| Orders 2-9 (8 orders) | 256 MB | 8 orders x 2 tables x 4M buckets x 4 bytes |
| Orders 2-9, 64M buckets | 4,096 MB | 8 orders x 2 tables x 64M buckets x 4 bytes |

None of this counts toward the 16MB artifact limit. The tables are empty at the start of evaluation and grow as tokens are scored. By the end of evaluation, the model that is doing the actual prediction is 16MB of neural weights plus 256MB of hash tables — **272 MB total**.

---

## The Dilemma

The competition constrains the artifact to 16MB. The intent is clear: force creative compression of model knowledge into a small footprint. But eval-time techniques like n-gram caching, TTT, and LoRA adaptation grow the effective model far beyond 16MB during evaluation — legally, because the rules only constrain the artifact, not the eval-time state.

This creates a gap between what the competition measures and what matters in practice.

### Four dimensions of the gap

|  | Competition | Real-world inference |
|--|-------------|---------------------|
| **Corpus** | Fixed 62M tokens, scored in one pass | Streaming queries, each independent |
| **Time budget** | 600 seconds for the entire corpus | < 100ms per token, real-time |
| **Hardware** | 8x H100 80GB (640 GB VRAM) | Often 1 GPU, sometimes CPU |
| **Model size** | 16 MB artifact; eval-time state unconstrained | Total model must fit deployment target |

Each dimension matters:

**1. Inference time.** The competition allows 600 seconds to score 62M tokens. The n-gram cache exploits this by doing O(K) hash lookups per token across K orders, plus table updates after scoring. On a single GPU, our best config takes 1234s — already over budget. On 8 GPUs with all-reduce sync (EXP-11, implemented but not yet deployed), we estimate ~130s. In real-world inference, you serve one token at a time with a latency budget measured in milliseconds. There is no batch of 62M tokens to amortize over.

**2. Inference hardware.** The competition provides 8x H100 with 640GB of combined VRAM. The hash tables (256 MB per GPU, synced via all-reduce) are negligible relative to this. In deployment, models run on single GPUs, edge devices, or CPUs. The 256MB of hash tables alone exceeds the 16MB artifact by 16x.

**3. Competition setup.** The artifact limit constrains what you ship. But the n-gram cache ships nothing — it materializes at eval time from the scored tokens themselves. The 16MB limit was designed to constrain model capacity. The n-gram cache circumvents this by building an unbounded statistical model during evaluation, limited only by the number of hash buckets allocated.

**4. Real-world evaluation.** In production, a language model scores individual prompts. Each query arrives independently. There is no corpus-level repetition to exploit. The n-gram cache's power comes entirely from within-corpus repetition — repeated documents, boilerplate, subword completion patterns, common phrases. This is **compression**, not **language modeling**. It works because FineWeb val has structure that repeats across its 62M tokens. On a stream of independent queries, the cache starts empty for each request and provides no benefit.

### The core tension

The competition implicitly asks: **given N bytes of model, how well can you compress natural language?**

Eval-time caching answers a different question: **given N bytes of model plus unbounded eval-time memory, how well can you compress a specific fixed corpus?**

These are different problems. The second has a much lower floor — any corpus with internal repetition can be compressed toward its empirical entropy by memorizing seen patterns. Our results show the gap is enormous: 1.11 BPB (neural only) vs 0.38 BPB (neural + cache). The cache contributes 2/3 of the total compression, yet costs zero artifact bytes.

---

## What's already legal and where the line blurs

The competition already permits eval-time model growth through several mechanisms:

| Technique | Eval-time state growth | Legality status |
|-----------|----------------------:|----|
| Sliding window eval (stride < seq_len) | KV cache, ~20 MB | Uncontroversial |
| Test-time training (score-first TTT) | LoRA deltas, ~2 MB | Approved (PRs #549, #548) |
| Per-document LoRA TTT (8 epochs) | LoRA deltas, ~2 MB | Approved (PR #596, 0.62 BPB) |
| N-gram cache (backoff 2-7) | Hash tables, 192 MB | Under review |
| N-gram cache (backoff 2-9, 64M buckets) | Hash tables, 4 GB | Under review |

TTT and LoRA adaptation are already approved. They also grow the model at eval time (LoRA weights are not in the artifact), though the growth is modest (~2 MB). The n-gram cache follows the same principle — build state from scored tokens — but at 100x the scale.

The question is not whether causality is preserved (it is), but whether unbounded eval-time model growth is in the spirit of the 16MB constraint.

---

## Proposal

We suggest the competition consider one or more of the following clarifications:

**Option A: Cap eval-time state.** Define a total memory budget for eval-time state (e.g., artifact + eval-time state <= 32 MB or 64 MB). This directly constrains the effective model size and aligns the competition with deployment realities.

**Option B: Per-token compute budget.** Instead of a wall-clock limit for the entire corpus, define a per-token compute budget (e.g., max FLOPs per token). This prevents techniques that amortize expensive corpus-level operations.

**Option C: Evaluate on independent documents.** Score each document independently with a fresh model state (no carry-over between documents). This eliminates cross-document repetition exploitation while still allowing within-document TTT and caching.

**Option D: Accept eval-time growth, but measure it.** Keep current rules but require submissions to report their peak eval-time state size alongside val_bpb. This makes the tradeoff transparent: "0.38 BPB at 272 MB effective model" tells a different story than "0.38 BPB at 16 MB."

We believe **Option A** or **Option D** would be the simplest to implement and the least disruptive to existing submissions.

---

## Surprising findings

1. **Global cache vs partitioned cache:** On 8 GPUs with independent caches (as in PRs #727, #788), each GPU sees 1/8 of the tokens. This degrades BPB from ~0.49 (global) to ~0.97 (partitioned) — a 0.48 BPB gap from cache fragmentation alone. Our EXP-11 implementation solves this with all-reduce sync of hash table deltas across GPUs, giving every GPU the global cache state.

2. **Entropy-adaptive alpha hurts with strong caches:** The sigmoid-gated alpha from PR #727 (which reduces n-gram weight when the neural model is confident) gives 0.65 BPB — 0.16 BPB *worse* than fixed alpha=0.40 (0.49 BPB). With a global cache, the n-gram is often more reliable than the neural model, and the entropy gate is too conservative.

3. **N-gram alone beats the neural model:** Pure n-gram (no neural model at all) achieves 1.06 BPB vs 1.11 BPB for the neural model. A zero-parameter frequency table built from scored tokens predicts FineWeb better than a 27M-parameter transformer.

4. **Three compression phenomena:** The n-gram cache captures (a) deterministic BPE subword completion (orders 2-4), (b) common English collocations (orders 4-6), and (c) verbatim document repetition (orders 6+). Only (c) is corpus-specific.

---

## Reproduction

All scripts are in `experiments/eval_time_mixing/scripts/`:

```bash
# Single-GPU experiments (EXP-0, requires 1xH100 + trained model)
python3 experiments/eval_time_mixing/scripts/eval_ngram.py \
    --model final_model.pt --exp backoff_7

# 8-GPU distributed with global cache (EXP-11)
NGRAM_ENABLED=1 NGRAM_ORDER=9 NGRAM_ALPHA=0.40 \
    torchrun --standalone --nproc_per_node=8 \
    experiments/eval_time_mixing/scripts/eval_ngram_distributed.py

# N-gram match analysis (qualitative)
python3 experiments/eval_time_mixing/scripts/analyze_ngram_matches.py
```

Base model: `train_609_val_calib.py` from `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`.

## Credits

N-gram cache concept and initial implementations: [PR #727](https://github.com/openai/parameter-golf/pull/727), [PR #779](https://github.com/openai/parameter-golf/pull/779), [PR #788](https://github.com/openai/parameter-golf/pull/788). Competition design and infrastructure: OpenAI.
