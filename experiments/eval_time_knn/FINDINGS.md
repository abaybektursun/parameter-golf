# kNN-LM & N-gram Audit Experiments — Complete Findings

## 1. N-gram BPB Scores Are Invalid (THE key result)

**Base model:** 1×H100 single-GPU, base train_gpt.py, 600s wallclock → val_bpb=1.2711

| Experiment | BPB | Distribution Sum | Valid? |
|---|---|---|---|
| Baseline (no cache) | 1.2711 | 1.0 | Yes |
| N-gram backoff 2-7, α=0.40 | 0.5422 | **277.0** | **No** |
| N-gram (after full-vocab normalization) | ~3.73 | 1.0 | Yes, but far worse than baseline |

**What this proves:** The n-gram "improvement" is entirely a measurement artifact. The blended distribution sums to 277, not 1. After normalization, the n-gram actively hurts. One `assert abs(probs.sum() - 1.0) < 1e-4` in the eval harness catches this.

**How the audit works:** The `backoff_7_audit` config runs normal n-gram mixing (producing the "claimed" BPB), then separately computes P(cache_bin) for ALL 1024 tokens, blends the full distributions, and reports (a) the sum, (b) the normalized NLL. No changes to the n-gram implementation — just measuring what the eval harness should check.

## 2. Five Competition PRs Are Invalid (community analysis)

### Pure n-gram hash PRs: sum ≈ 410. Unfixable.
Single-token hash lookup is the mechanism. P(cache_bin) ≈ 1.0 for every token due to collision aggregation. No way to fix this while keeping the approach.

### Hedge Mixer PRs (#700, #745, #849, #953): sum ≈ 2-6. Fixable.
The n-gram experts (unigram, bigram, trigram, 4-gram) are actually correct — Laplace-smoothed count tables over all 1024 tokens. Valid distributions. The mixture of valid distributions is valid.

But Expert 4 ("entropy") computes Shannon entropy H(p_neural) = -Σ p·log(p) — a scalar that doesn't depend on the target token. Mixed via logsumexp as if it were a log-probability. This makes the mixture sum to ~2-6 instead of 1.

**Fix:** Remove the entropy expert. The remaining mixture (neural + smoothed n-grams) is fully valid. These PRs did the hard part right and broke validity with one bogus expert.

### kNN-LM PR #738: Partially valid.
The kNN part produces a valid full-vocab distribution (scatter_add_ + softmax over distances). But:
1. N-gram part only looks up correct token — same bug as other n-gram PRs
2. Selective mixing: when cache doesn't help, skips mixing entirely (keeps full p_model instead of reduced (1-λ)*p_model)
3. Confidence gate peeks at the target token — oracle-like behavior

## 3. kNN-LM: Valid Distributions, Tiny Improvement

kNN-LM stores hidden states during eval, retrieves k nearest neighbors via L2 distance, and produces valid probability distributions (softmax over distances → scatter into vocab → sum=1).

### Results

**Best config: λ=0.015, k=64, T=1.0 → +0.0008 BPB improvement with valid distributions.**

#### Phase 1: k sweep (λ=0.01, T=1.0)
| k | Improvement |
|---|---|
| 4 | +0.0006 |
| 8 | +0.0007 |
| 16 | +0.0007 |
| 32 | +0.0007 |
| 64 | +0.0007 |
| 128 | +0.0007 |

k doesn't matter — ceiling is λ-limited, not k-limited.

#### Phase 2: Fine λ sweep (k=64, T=1.0)
| λ | Improvement |
|---|---|
| 0.002 | +0.0003 |
| 0.005 | +0.0006 |
| 0.008 | +0.0007 |
| 0.010 | +0.0007 |
| **0.015** | **+0.0008** |
| 0.020 | +0.0007 |
| 0.030 | +0.0005 |
| 0.050 | -0.0007 |
| 0.100 | -0.0057 |
| 0.250 | -0.0389 |

Clear bell curve. Peak at λ=0.015. Higher λ overwhelms the model with noise from weak 512-dim embeddings.

#### Temperature sweep (λ=0.01)
| T | Improvement |
|---|---|
| 1.0 | +0.0007 |
| 10.0 | +0.0001 |
| 100.0 | +0.0000 |

Low T wins — peaky kNN distribution (trust nearest neighbor) beats flat (all neighbors equal).

#### Cosine normalization
| Config | Improvement |
|---|---|
| k=8, λ=0.01, norm | +0.0002 |

L2 distance outperforms cosine for this model. Normalization hurts.

### Interpretation

kNN-LM works, but barely. A 16MB model's 512-dim embeddings are too degraded for nearest-neighbor retrieval to provide significant signal. Khandelwal et al. 2020 showed +31% perplexity reduction with GPT-2 (774M params, 1600-dim). Our model is 17M params with 512-dim — the embedding space is too noisy for retrieval beyond ~1.5% interpolation weight.

**This is still a valid result for the paper:**
- kNN-LM IS the right way to do eval-time model growth (valid distributions, strict causality)
- With a stronger base model (or future models), kNN-LM could provide meaningful gains
- Even +0.0008 BPB proves the mechanism works — it's just limited by embedding quality
- The 2GB eval-time state (per GPU) for +0.0008 BPB illustrates the cost/benefit of unbounded state

## 4. Critical Bug: Model Weights Must Stay float32

**Discovery:** Loading model as `.bfloat16()` (casting all weights to bf16) destroys performance:
- float32 weights: val_loss=2.24, working correctly
- bfloat16 weights: val_loss=3.12, ~40% worse

**Root cause:** The model was trained with `torch.autocast` (bf16 compute, float32 master weights). The weights have precision that bf16 cannot represent. Casting weights to bf16 rounds them destructively.

**Fix:** Keep model weights float32. Use `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` for bf16 compute during the forward pass only.

**Impact:** The 1×H100 kNN result (model_bpb=2.17, knn_bpb=2.09, delta=+0.07) was computed with corrupted bf16 weights. The delta was real (kNN helped the degraded model) but both absolute BPB values were wrong. Corrected results use float32 weights throughout.

## 5. Implementation Notes

### GPU-native kNN (no FAISS)
Pure torch matmul for L2 distances: `dists = ||q||² + ||s||² - 2*q·s`. Faster and simpler than FAISS on modern GPUs.
- FAISS GPU had cublas compatibility issues (assertion failure with CUDA 12.8)
- FAISS CPU was impractical (estimated 67 hours for 2M store)
- Torch matmul on H100: ~2ms per batch for 2M-store search

### Distributed kNN
8-GPU data parallel: contiguous window chunks per GPU, local datastores, all-reduce final metrics. No cross-GPU kNN communication needed.
- 8×RTX PRO 6000 Blackwell (96GB each): ~7 min per full eval
- Each GPU stores 2M vectors (2 GB fp16)

### Datastore capacity
With 96GB per card, could store ALL 62M hidden states (64 GB fp16) per GPU. Would enable full-corpus retrieval with potentially much better results. Not tested due to time constraints.

## Hardware Used

| Machine | GPU | VRAM | Use |
|---|---|---|---|
| vast.ai #1 | 1×H100 80GB SXM | 80 GB | Training, n-gram eval, n-gram audit |
| vast.ai #2 | 1×H100 80GB SXM | 80 GB | Single-GPU kNN-LM (bf16 bug found) |
| vast.ai #3 | 1×H100 80GB SXM | 80 GB | Single-GPU kNN-LM (float32 corrected) |
| Dedicated | 8×RTX PRO 6000 Blackwell | 768 GB | Distributed kNN-LM sweep |

## Files

```
experiments/eval_time_knn/
├── FINDINGS.md              ← this file
├── scripts/
│   ├── eval_knn.py          # Single-GPU kNN-LM (pure torch)
│   ├── eval_knn_distributed.py  # 8-GPU with --normalize flag
│   ├── sweep_knn.sh         # v1 sweep: λ × T × normalize
│   └── sweep_knn_v2.sh      # v2 sweep: k, fine-λ, cosine
├── checkpoints/
│   └── final_model_1gpu.pt  # 67MB float32 (base train_gpt.py, 1×H100)
├── results/
│   ├── baseline_1gpu.json
│   ├── backoff_7_1gpu.json
│   ├── backoff_7_audit_1gpu.json    # THE key result: sum=277
│   ├── knn_1xH100_*.json
│   ├── knn_8xRTXPRO6000_*.json
│   └── all_sweep_results.txt
└── logs/
    ├── train_1gpu.log
    ├── knn_*.log
    └── sweep_*.log
```

Also see: `experiments/eval_time_mixing/` for the n-gram eval scripts (eval_ngram.py with audit mode).
