# Optimization Opportunities

Ranked by expected impact and effort. Derived from two deep research reports,
due diligence verification, first-principles FLOP/memory analysis, and
cross-referencing with the modded-nanogpt speedrun codebase.

Baseline: ~52ms/step, ~11,500 steps in 10min, ~27% hardware efficiency.
59% of step time is overhead (not useful compute).

---

## Tier 1: High Impact, Low Effort

### 1. `mode="max-autotune"` in torch.compile
**Status:** Not currently used. One-line change.
**What:** Enables CUDA graph capture (eliminates kernel launch overhead entirely
for static shapes) and deeper Triton autotuning grid search.
**Why it matters:** Our shapes are fully static (65536 tokens, fixed dims). CUDA
graphs collapse the entire fwd+bwd into a single GPU-side replay — no Python
dispatch, no CPU-side kernel launches. This directly attacks the ~30ms of
unaccounted overhead.
**Change:**
```python
# Current:
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
# New:
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True, mode="max-autotune")
```
**Risk:** Low. If it breaks, remove it.
**Source:** Research report #2, verified against PyTorch 2.8 docs.

### 2. Fused Cross-Entropy + Logit Softcap
**Status:** Not used. Available via Liger-Kernel or modded-nanogpt's `FusedSoftcappedCrossEntropy`.
**What:** Replaces the current 3-step sequence (matmul → tanh softcap → cross_entropy)
with a single kernel that never materializes the (65536, 1024) = 268MB logit tensor.
Computes softmax+NLL per row entirely in SRAM (vocab=1024 = 4KB per row, trivial).
**Why it matters:** Eliminates ~1.6GB of HBM traffic per step (forward + backward).
Removes 3-4 kernel launches. Proven in modded-nanogpt speedrun records.
**Implementation paths:**
- `pip install liger-kernel` → `LigerFusedLinearCrossEntropyLoss`
- Port `FusedSoftcappedCrossEntropy` from modded-nanogpt's triton_kernels
- Adapt for our specific softcap: `30 * tanh(logits / 30)`
**Risk:** Low. Well-tested in production. Need to verify compatibility with our
tied embedding (tok_emb.weight used as projection).
**Source:** Both research reports, validated by modded-nanogpt codebase.

### 3. Fused ReLU² MLP (forward + backward)
**Status:** Not used. Available via modded-nanogpt's `FusedLinearReLUSquareFunction`.
**What:** Fuses `proj(relu(fc(x))²)` into a single operation that avoids
materializing the 130MB intermediate tensor per layer. Critically, handles the
backward pass by recomputing activations on the fly instead of storing them.
**Why it matters:** Eliminates 2.34GB of HBM traffic across 9 layers (130MB × 2 × 9).
The recomputation costs ~1.2 TFLOPS of extra compute — trivial at 650 TFLOPS/s.
**Important:** Research report #2 incorrectly claimed torch.compile makes this
redundant. It handles forward epilogue fusion but NOT backward recomputation.
modded-nanogpt has this kernel for a reason.
**Risk:** Medium. Need to adapt to our architecture (992 hidden, not standard).
**Source:** modded-nanogpt codebase, verified by our memory traffic analysis.

---

## Tier 2: Medium Impact, Low-Medium Effort

### 4. Batch Size Warmup Schedule
**Status:** Not used. Config + minor code change.
**What:** Start training with smaller batch (131K tokens), step up to 262K at
~1 minute, then 524K at ~3 minutes. Scale LR proportionally (√batch factor).
**Why it matters:** Early in training, gradient signal is dense and high-variance.
Smaller batches = more weight updates = faster early exploration. Our 524K batch
is near the critical batch size (~250K-550K), so early steps are wasted on
redundant gradient information.
**Implementation:** Step-function (not continuous) to avoid torch.compile
recompilation. Only 2 recompilations, cleanly cached.
**Risk:** Low-medium. LR scaling with batch needs tuning.
**Source:** Research report #2, validated by modded-nanogpt practices.

### 5. Triton Autotune Configuration
**Status:** Not used. Environment variable changes.
**What:** Force deeper autotuning grid search and Triton-only GEMM backends.
```python
import torch._inductor.config as config
config.max_autotune = True
config.max_autotune_gemm = True
```
**Why it matters:** Default autotune searches a narrow space to minimize compile
time. Spending 20-45 extra seconds at startup to find better tile sizes / warp
counts could save microseconds per kernel across 11,500 steps.
**Risk:** Low. Increases initial compile time. May regress if Triton configs are
worse than cuBLAS for our specific matrix sizes.
**Source:** Research report #2.

### 6. Lookahead/SNOO Outer Optimizer + LAWA
**Status:** Implemented in experiment 002, not yet tested.
**What:** Slow-weights wrapper with Nesterov momentum (H=30 steps) + checkpoint
averaging in final 25% of training.
**Why it matters:** Finds wider, flatter minima that generalize better AND survive
int8 quantization with less degradation. MuLoCo (Muon + Local SGD) shown to be
especially quantization-resilient.
**Risk:** Medium. Hyperparameters (outer_lr, H) need tuning.
**Source:** Research report #1 (DiLoCo/SNOO literature).

---

## Tier 3: Medium Impact, Medium-High Effort

### 7. Profile Before Optimizing (TORCH_COMPILE_DEBUG + nsys)
**Status:** Not done. Essential prerequisite for Tier 1 kernel work.
**What:** Two-step profiling:
1. `TORCH_COMPILE_DEBUG=1` — inspect generated Triton kernels, verify which
   fusions actually happened (RMSNorm+RoPE? relu² epilogue?).
2. `nsys profile --trace=cuda,nvtx` — get exact temporal breakdown of the 52ms
   step across forward, backward, optimizer, all-reduce, and gaps.
**Why it matters:** Without this, we're guessing where the time goes. The 30ms
of "unaccounted overhead" could be kernel launch gaps (fixable by CUDA graphs),
unfused element-wise chains, or something else entirely.
**Risk:** None. Pure information gathering.
**Source:** Research report #2.

### 8. Verify torch.compile Fusions (RMSNorm + RoPE)
**Status:** Unknown. Research report #2 claims PyTorch 2.8 fuses these automatically.
**What:** Check `TORCH_COMPILE_DEBUG` output to see if F.rms_norm on Q/K followed
by RoPE generates a single fused Triton kernel or separate kernels.
**Why it matters:** If NOT fused, this is 2 extra read/writes of Q and K tensors
per layer (64MB each × 2 × 9 layers = 2.3GB of traffic). If fused, no action needed.
**Risk:** None to check. Custom kernel needed only if compiler misses it.
**Source:** Research report #2 (unverified claim).

### 9. Existing Fused Kernels from `kernels` Package
**Status:** The `kernels` package is already in requirements.txt.
**What:** The Kernel Hub may have pre-built fused kernels loadable at runtime.
modded-nanogpt's triton_kernels module exports exactly the kernels we need.
Investigate what's available without writing anything from scratch.
**Risk:** Low. Import and test.
**Source:** requirements.txt, modded-nanogpt codebase.

---

## Tier 4: Speculative / Needs Empirical Validation

### 10. Sequence Length 4096
**Status:** Not tested. Trivial env var change.
**What:** Double context from 2048 to 4096. Same total tokens per step (65536).
**Why it matters:** 1024→2048 was a proven win. But 2048→4096 increases attention
FLOPs by 2x (attention is O(N²)), total step compute by ~53%, estimated step
time from ~52ms to ~63ms. Loses ~2,100 steps (11,500→9,400).
**Open question:** Does per-step quality improvement outweigh 18% fewer steps?
At d=64, FlashAttention is memory-bound, making the wall-clock penalty worse
than the FLOP ratio suggests.
**Risk:** Medium. Easy to test but likely negative for our model size + step budget.
**Source:** Both reports agree it's risky. Must test empirically.

### 11. Mixed FP8 (Linear Layers Only)
**Status:** Not tested. Non-trivial code changes.
**What:** FP8 for attention projections and MLP linears only. Keep bf16 for
norms, softmax, softcap, embeddings. Use PyTorch 2.8's torch.float8 APIs.
**Why it matters:** Could get ~1.5x throughput on the GEMM-heavy portions.
Research report #2 rejected FP8 entirely, but based on a false premise
(claimed modded-nanogpt doesn't use FP8 — it does).
**Open questions:** Does FP8 degrade quality enough to negate the extra steps?
Does the tied embedding work with mixed scaling? Does our tanh softcap survive?
**Risk:** High. Small models are more sensitive to precision loss.
**Source:** Research report #2 (rejected, but on incorrect grounds).

### 12. Larger Batch Size (2x = 1M tokens)
**Status:** Not tested. Trivial env var change.
**What:** Double batch to 1,048,576 tokens. Step time increases sub-linearly
(fixed overheads don't scale), but total tokens stays roughly constant.
**Why it matters:** Only useful if B_crit > 524K, meaning our current batch
is suboptimal. Unlikely based on analysis, but trivial to test.
**Risk:** Low (easy to test), but probably no gain.
**Source:** Both reports agree this is near-saturated.

---

## Rejected

### Custom NVSHMEM / P2P All-Reduce
All-reduce is ~2-3ms for 20MB on NVLink. Savings: ~2ms/step. Requires months
of custom CUDA. Not worth it.

### Persistent Mega-Kernels / Whole-Block Fusion
Entire transformer block in one kernel. Months of CUDA development.
Impractical for challenge timeline.

### L2 Cache Pinning
Model fits in L2 (19.2MB < 50MB), but no Python-level API to exploit this.
Requires custom CUDA. torch.compile may already benefit from natural L2 residency.

### Full FP8 Training (All Components)
Logit softcap (tanh), relu² outliers, and tied embeddings all have documented
FP8 failure modes. Not worth the risk for a 9.6M parameter model.
