# Surprises and Learnings

Non-obvious findings that emerged during this research.

---

## 2026-03-25: Initial PR Review

0. **Causality is the hard constraint**: You can ONLY use validation tokens you have already scored. This is the single most important rule. Every technique — n-gram tables, suffix trees, PPM, PAQ mixing — must be strictly backward-looking. Score first, update after. No peeking at future tokens, no oracle selection based on the true target.

1. **The n-gram cache is absurdly effective**: PR #727 shows the cache alone gives -0.16 BPB improvement (1.127 -> 0.967). That's larger than ALL neural architecture improvements combined (baseline 1.224 -> our SOTA 1.119 = -0.105 BPB). A zero-parameter eval-time trick beats months of architecture work.

2. **PR #779 ablation is damning**: In their 0.6683 BPB submission, TTT contributes essentially nothing. Base model 1.1363, TTT-only 1.1369 (WORSE), mixer-only 0.6712, full system 0.6663. The BackoffNgramMixer is 99% of the improvement.

3. **All implementations mix in probability space, not logit space**: Every PR does `p_mix = (1-a)*p_model + a*p_ngram`. The research doc's Section 4 argues logistic mixing is strictly superior. Nobody has tested this yet — this could be our core novel contribution.

4. **Point-probability mixing is a shortcut**: All PRs only compute p_ngram for the TRUE target token. They never build a full distribution over V=1024. This means the "mixing" is really just adjusting the NLL of the correct answer — it's not a proper probabilistic ensemble. This is both a limitation and possibly the reason it works so well (you're directly optimizing the scoring metric).

5. **Hash collision rate is unknown**: 4M buckets with 62M tokens across 6-8 orders means hundreds of millions of insertions. Nobody has measured the false positive/collision rate. This could be a silent performance killer.

6. **PR #788's order-adaptive entropy gating is clever**: Instead of a single sigmoid threshold, the center shifts with match order. A 9-gram match triggers at low entropy (center=1.25) while a 2-gram needs high entropy (center=3.0). This is a simple but elegant heuristic.

7. **Nobody has tried PPM or suffix trees**: Despite decades of compression literature showing PPM >> n-gram counting and suffix trees enabling unbounded context, all competition entries use the simplest possible approach. Low-hanging fruit for a paper.

## 2026-03-26: EXP-0 First Results

8. **Single-GPU global cache is MUCH better than 8-GPU partitioned caches**: Our fixed 7-gram (alpha=0.40, no backoff) on single GPU gave **0.5234 BPB**. PR #727 reported ~1.03 for the same config on 8xH100. The difference: on 8 GPUs, each GPU has its own independent cache seeing only 1/8 of the data. On single GPU, the cache sees everything. This is a ~0.50 BPB gap just from cache coverage. This is a critical finding — the "shared-nothing" parallelism in the PRs massively degrades cache effectiveness.

9. **"Scoring a fixed corpus" vs "modeling language" is the core insight**: The n-gram cache exploits repetition within the FineWeb val corpus — this is compression, not language modeling. The neural model generalizes; the n-gram memorizes. The research question is: is this improvement purely an artifact of this specific dataset's repetition structure? Or would it generalize to any web text corpus? This motivates EXP-7 (document boundaries) and a new experiment measuring the exact repetition rate of the corpus.

10. **N-gram overhead: practical only for scoring, not for inference**: The n-gram cache adds 36-103% eval time overhead (CPU numpy bottleneck). In production, it requires O(N) memory scaling with scored tokens, O(K) per-token lookups, and cannot be batched. It's a corpus compression trick, not an inference technique. But crucially, the tables cost zero artifact bytes — they're built dynamically from scored data, exploiting the gap between the 16MB artifact limit and the 640GB eval-time VRAM.

11. **Entropy-adaptive alpha HURTS with a strong global cache**: backoff_7 (fixed α=0.40) gives 0.4923 BPB, but backoff_7_ent (entropy-adaptive) gives 0.6535 BPB — 0.16 BPB WORSE. The entropy-adaptive sigmoid reduces alpha when the model is confident (low entropy), which makes sense on 8-GPU partitioned caches where n-gram stats are sparse. But with a global cache, the n-gram is often MORE reliable than the model, and the entropy gate is being too conservative — it defers to the model when the n-gram should dominate. This means the optimal alpha depends on cache coverage, and the "one-size-fits-all" sigmoid formula from PR #727 is actually counterproductive in high-coverage regimes.

## 2026-03-26: EXP-11 Results (8-GPU All-Reduce)

12. **8-GPU all-reduce sync works perfectly**: The all-reduce of hash table deltas costs only 1.6-2.0s total across ~4000 batches. The 8-GPU BPB (0.4941) matches single-GPU (0.4923) within 0.002 — confirming global cache quality is preserved. The total eval time (401s for backoff_7) fits well within the 600s competition budget.

13. **Higher alpha is strictly better for global cache**: Alpha sweep on 8-GPU shows monotonic improvement: α=0.20 (0.6180) → α=0.40 (0.4941) → α=0.60 (0.4263) → α=0.80 (0.3942). The n-gram cache is so reliable with global coverage that trusting it more always helps. This is the opposite of the PR findings where α=0.40 was optimal (because their partitioned caches were less reliable).

14. **Logistic mixing is WORSE than linear**: mix_logistic (1-GPU) gives 0.7544 BPB vs linear (0.4923). The research doc's Section 4 claimed logistic mixing would eliminate the "entropy floor" penalty. In practice, when n-gram confidence is low (early in corpus, rare contexts), the logistic transform amplifies the low n-gram probability downward, hurting the mixture. Linear mixing is more robust to low-confidence n-gram estimates.

15. **Hash collisions HELP, not hurt**: Bucket sweep results are backwards from expectations. 1M buckets (0.5793 BPB) beats 4M (0.4923) beats 64M (1.0629) beats 256M (1.1123). With 256M buckets, the hash table is so sparse that most n-grams have count=1 and fail the `min_count >= 2` threshold. Collisions at 4M buckets are actually BENEFICIAL — they merge similar n-grams together, artificially boosting counts above the threshold. The hash table is functioning as a lossy count-min sketch, not as an exact lookup. This fundamentally changes our understanding of the mechanism.

16. **Sliding window overlap and n-gram cache are orthogonal**: Stride decomposition shows the n-gram delta is ~0.62 BPB regardless of stride (64: -0.619, 256: -0.620, 2048: -0.636). The overlap from sliding windows contributes only ~0.03 BPB (stride 64 vs 2048 without cache). They operate on completely different information sources.

17. **GPU contention caused misleading timing**: Some alpha sweep runs showed 2-3x timing variation (α=0.40: 401s vs α=0.50: 1295s). Root cause: a competing single-GPU experiment (mix_geometric) was sharing GPU 7 with the 8-GPU torchrun. Since NCCL all-reduce waits for the slowest GPU, one contended GPU dragged all 8 to a crawl. Lesson: 8-GPU distributed jobs must have EXCLUSIVE access to all GPUs. The clean timing (401s for backoff_7, 500s for backoff_9) is reliable.
