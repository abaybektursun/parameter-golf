# Neurosymbolic experiments

Current model: 11-layer transformer, 512-dim, 27.1M params, int6 GPTQ → 16 MB, **1.11 BPB**.

These experiments combine symbolic/algorithmic methods with the neural model. Every experiment produces valid probability distributions over the full 1024-token vocabulary. Ordered by leverage.

---

## 1. Log-space blending of neural + PPM at eval time

### Why this is different from what failed

The Hedge Mixer blends in probability space: `p_mix = Σ wk · pk(y)`. When a naive n-gram expert assigns near-zero probability to the correct token, the mixture gets dragged toward zero — catastrophic log-loss. This is why removing the (invalid) entropy expert from PR #700 dropped performance to ~1.3 BPB. The valid n-gram experts actively hurt.

PAQ/CMIX context mixing blends in log-odds space: `logit_mix = Σ wi · log(pi / (1-pi))`, then sigmoid back to probability. This has two critical properties:

1. A confident-but-wrong expert doesn't cause catastrophic loss — its logit is large but finite, and other experts can offset it.
2. Weights can go negative, effectively inverting a confidently wrong expert.

The Midicoth system (Tacconelli 2026, arXiv:2603.08771) adds a further refinement: Tweedie denoising of count-based predictions before blending. Raw PPM counts are treated as noisy estimates; a James-Stein shrinkage correction pulls extreme predictions toward uniform based on the variance of that specific context. This prevents the overconfidence problem that plagues raw count models.

### What to build

```
For each token position:
  1. Neural model produces logits over 1024 tokens → softmax → neural_probs
  2. PPM tree (built causally from scored tokens) produces distribution
     over 1024 tokens via escape mechanism (PPMC or PPMD)
  3. Apply Tweedie shrinkage to PPM distribution (optional, test with and without)
  4. Convert both distributions to log-odds
  5. Blend: logit_mix = (1-α) · logit_neural + α · logit_ppm
     where α is updated online via SGD on log-loss (Nacrith-style adaptive bias)
  6. Sigmoid → normalize → valid 1024-token distribution
```

The PPM tree starts empty. As tokens are scored, they're inserted into the tree. This is strictly causal — the tree only contains already-scored tokens. The tree lives in VRAM (estimated ~280 MB for order-12 over 62M tokens), well within the 640 GB budget. The tree costs zero artifact bytes.

### Key implementation detail: 1024-token alphabet

PPM traditionally operates on bytes (256 symbols). With 1024 BPE tokens, the tree has wider branching factor. Two options:

**Option A — Native 1024-ary PPM.** Each tree node branches into up to 1024 children. Memory-hungry but conceptually simple. The escape mechanism allocates mass to unseen tokens at each order.

**Option B — Binary tree decomposition.** Encode each 1024-token prediction as 10 binary decisions (2^10 = 1024). Run 10 binary PPM models in sequence. Each binary model uses standard PPM with 2-symbol alphabet — fast, well-studied, memory-efficient. This is what Midicoth does for bytes (8 binary decisions for 256 symbols). The 1024-token version needs 10 decisions. The resulting distribution is automatically valid.

### Compute budget

PPM tree traversal is O(order) per token. At order 12 with 1024-ary branching: ~12 pointer lookups per token. Modern C++ implementations run at tens of MB/s on CPU. For 62M tokens in 600 seconds, we need ~103K tokens/second. PPM easily handles this — the neural forward pass is the bottleneck, not PPM.

### What to expect

Unknown. This is the first experiment to try because it tests the core hypothesis: does log-space blending with a principled symbolic predictor help, where probability-space blending hurt? If this doesn't beat 1.11 BPB, the remaining experiments are less likely to work. If it does, it opens a new direction.

---

## 2. PPM-guided loss reweighting during training

### The idea

Run a lightweight PPM model alongside the neural training loop. At each token, the PPM model produces its own prediction. Compare PPM's confidence to the neural model's loss:

- **PPM confident, PPM correct:** This token is structurally predictable (boilerplate, template, BPE completion). The neural model doesn't need to learn this — a symbolic method handles it for free. **Downweight the neural loss.**
- **PPM uncertain, neural loss high:** This token requires semantic understanding that only the neural model can provide. **Full weight on neural loss.**
- **PPM confident, PPM wrong:** Rare. PPM's escape mechanism usually prevents high confidence on wrong predictions. If it happens, ignore — neural loss stays at full weight.

This steers the neural model's 16 MB of capacity toward tokens where it has a unique advantage: long-range semantic reasoning, world knowledge, contextual inference. The symbolic PPM handles the rest at eval time for free.

### Implementation

```python
# During training, alongside the neural forward pass:
ppm_prob = ppm_model.predict(context, all_tokens=True)  # [1024] distribution
ppm_confidence = ppm_prob[target_token]

# Scale the neural loss inversely to PPM confidence
weight = 1.0 - beta * ppm_confidence  # beta ∈ [0, 1] controls strength
weight = max(weight, min_weight)       # floor to prevent zero gradient
loss = weight * cross_entropy(neural_logits, target)
```

This is a one-paragraph change to the training loop. The PPM model runs on CPU in parallel — it doesn't touch the GPU.

### Interaction with experiment #1

If the eval-time PPM blend (experiment #1) works, training with PPM loss reweighting makes the neural model complementary to PPM by design. The neural model learns to be good where PPM is bad, and PPM covers where the neural model is bad. The blend at eval time exploits this complementarity.

If experiment #1 doesn't work (PPM can't help at eval time), this training intervention still helps — it focuses capacity on high-epiplexity tokens regardless of whether PPM is used at eval.

### What to expect

0.005-0.02 BPB improvement. The epiplexity framework (Finzi et al. 2026) says capacity-constrained models benefit from focusing on learnable signal. This directly implements that principle. Requires one training run.

---

## 3. Hard-coded induction heads

### The problem

The model has 2 induction heads out of 88. Induction heads perform a purely algorithmic operation: given the pattern `A B ... A`, predict `B`. This is exact string matching and copying — a symbolic task that the neural model struggles to learn with only 2 heads dedicated to it.

FineWeb has significant cross-document repetition (per-dump deduplication leaves cross-dump duplicates intact). Induction heads are exactly what's needed for this data. The model is under-equipped for the task.

### The idea

Replace 4-8 learned attention heads with parameter-free deterministic copy heads. These heads don't use Q/K/V projection weights — they compute exact token-ID matching between the current context and past context, then copy the token that followed the match.

```
Hard-coded induction head at position t:
  1. Look at token[t-1] (the "A" in "A B ... A → B")
  2. Scan positions 0..t-2 for exact matches of token[t-1]
  3. For each match at position j, record token[j+1] as a candidate
  4. Build a distribution over candidates (weighted by recency or frequency)
  5. Smooth with a uniform floor over all 1024 tokens
  6. Output: valid 1024-token distribution
```

This distribution gets mixed with the neural model's output (via the log-space blending from experiment #1, or via a learned gate within the architecture).

### Parameter savings

If these hard-coded heads replace learned heads, the Q/K/V/O weights for those heads are freed. With GQA (8 Q heads, 4 KV heads), the savings depend on which heads are replaced. If we replace 4 Q heads but keep their shared KV heads, we save 4 × 64 × 512 × 2 (Q and O projections) × 11 layers = ~2.9M params. These go straight to MLP.

Alternatively, the hard-coded heads can be added alongside the learned heads (no parameter savings but better induction capability). This trades compute for accuracy.

### Caveat

The hard-coded head produces a distribution based purely on token-ID matching. It doesn't use the model's learned representations. This means it can't generalize — it only copies exact matches, not semantic near-matches. Whether exact copying is sufficient for FineWeb's repetition patterns is an empirical question.

### What to expect

Uncertain. The value depends on how much of FineWeb's val set contains exact repeated n-grams that the current 2 induction heads miss. If the repetition rate is high (plausible given per-dump dedup), this could be significant. If most repetition is semantic rather than exact, it won't help.

Test cheaply: before building the hard-coded heads, run a simple analysis on the val set. Count how many tokens at position t are exact copies of the token following a previous occurrence of token[t-1]. If this is >5% of tokens, the intervention has headroom.

---

## 4. Symbolic BPE completion prior

### The observation

With a 1024-token BPE vocabulary, many tokens are character fragments. The sequence of fragments that complete a word is highly deterministic. If the model has emitted "prog", the next token is overwhelmingly likely to be "ram" or "ress" or "ressive". This is pure orthographic structure — no semantic reasoning needed.

The loss decomposition shows top-100 frequent tokens (mostly fragments) have 1.18 BPB — higher than tail tokens at 1.10 BPP. The model wastes parameters learning spelling rules.

### The idea

Pre-compute a deterministic BPE completion table from the tokenizer. For every possible prefix state (current token + partial word context), enumerate the valid BPE continuations and their frequencies in the training data.

At eval time, this table provides a valid prior distribution over the next token based purely on the BPE fragment state. Blend this prior with the neural model's semantic prediction in log-space.

```
If current token is a word fragment (not a complete word):
  prior = bpe_completion_table[current_token]  # precomputed, valid dist over 1024
  logit_blend = (1-γ) · logit_neural + γ · logit_prior
else:
  logit_blend = logit_neural  # no prior, pure neural
```

### Size in the artifact

The BPE completion table has at most 1024 × 1024 entries (each token's distribution over possible next tokens). At 2 bytes per entry: 2 MB. This fits in the 16 MB artifact. The neural model loses ~2 MB of parameters (~0.85M params at int6), but gains a perfect BPE completion oracle that handles an entire class of predictions with zero error.

### What to expect

Small but nearly certain gain on fragment-to-fragment transitions. The model currently wastes MLP capacity on learning `"prog" → "ram"` type patterns. Offloading this to a lookup table frees MLP for semantic prediction. The gain depends on what fraction of tokens are mid-word fragments — likely 30-50% of all positions.

---

## 5. Binary tree prediction (10-depth binary classifier cascade)

### The idea

Instead of predicting one of 1024 tokens directly (a 1024-way classification), decompose each prediction into 10 binary decisions along a Huffman or frequency-ordered binary tree. The first decision splits the vocabulary into two halves (most frequent vs least frequent). Each subsequent decision refines the prediction.

This is how PAQ/Midicoth handle large alphabets. The advantage: each binary decision is a simpler problem that can be solved by a simpler model. A capacity-starved model might be better at making 10 easy binary decisions than one hard 1024-way decision.

### Integration with the architecture

Replace the final `lm_head` (a 512 → 1024 linear layer, 524K params with tied embeddings) with a 10-layer binary decision tree. Each node is a small binary classifier (512 → 1 linear layer, 512 params). Total: 1024 nodes × 512 params = 524K params — exactly the same as the current lm_head.

The tree structure encodes frequency information: common tokens are reached in fewer decisions (shorter path), rare tokens take more decisions. This is a learned Huffman coding built into the architecture.

### What to expect

Theoretically neutral in parameter count. The question is whether the inductive bias of hierarchical binary decisions helps the capacity-starved model. If the model's 512-dim hidden state contains the information for coarse category decisions (is this a function word vs a content word?) but struggles with fine-grained distinctions (which specific content word?), the tree structure lets it allocate capacity to the right level of granularity.

Speculative. One training run to test.

---

## Experiment dependency graph

```
Experiment 1 (log-space PPM blend at eval)
    ↑
    Combines naturally with
    ↓
Experiment 2 (PPM loss reweighting during training)
    → Makes the neural model complementary to PPM by design

Experiment 3 (hard-coded induction heads)
    → Independent. Can combine with any other experiment.
    → Test the repetition rate first (cheap analysis).

Experiment 4 (BPE completion prior)
    → Independent. Can combine with any other experiment.
    → Smallest expected gain but highest confidence.

Experiment 5 (binary tree prediction)
    → Independent. Architectural change, requires retraining.
    → Most speculative.
```

### Recommended execution order

1. **Experiment 1** first — tests the core hypothesis (log-space blending helps). If this fails, experiments 2 and 4 still stand on their own. If this works, it unlocks the full pipeline.

2. **Experiment 4** second — nearly zero risk, can be built and tested in hours. Pre-compute the BPE table, add the blend, measure BPB.

3. **Repetition analysis** for experiment 3 — before building hard-coded induction heads, count exact n-gram repetitions in the val set. This takes one eval pass, no training.

4. **Experiment 2** in the next training run — combine PPM loss reweighting with whatever architectural changes are planned (MLP rebalancing from NEXT_STEPS.md).

5. **Experiment 5** only if experiments 1-4 show promise — the binary tree is the most speculative and requires the most engineering.

---

## References

- Tacconelli 2026a, "Nacrith: Neural Lossless Compression via Ensemble Context Modeling" — arXiv:2602.19626
- Tacconelli 2026b, "Micro-Diffusion Compression: Binary Tree Tweedie Denoising" — arXiv:2603.08771
- Finzi et al. 2026, "From Entropy to Epiplexity" — arXiv:2601.03220
- Cleary & Witten 1984, "Data Compression Using Adaptive Coding and Partial String Matching" (PPM)
- Veness et al. 2017, "Online Learning with Gated Linear Networks" (GLN, formalizing PAQ context mixing)
