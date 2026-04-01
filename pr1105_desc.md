> **Mechanistic Interpretability:** For a deep-dive analysis of this model — including SVD utilization, quantization sensitivity, logit lens, and calibration — see [Mechanistic Interpretability of PR 1105](https://abay.tech/posts/pr-1105-model-autopsy).

> Custom CUTLASS 3.x EVT + Triton TMA kernels recover the 3.3ms/step throughput regression from full Hessian GPTQ (our [merged PR 1019](https://github.com/openai/parameter-golf/pull/1019)). Mechanistic analysis of that model revealed MLP at 94.4% SVD rank utilization (fully packed) while attention Q sat at 72.6% — the model was parameter-starved in MLP, not attention. MLP 3.5×, enabled by Hessian-based mixed int5/int6 quantization, acts on that finding. Brotli-11 saves 581KB.

## Results: val_bpb 1.1052 (3-seed mean) | 1.9006 nats | 8×H100 SXM | 600s | ~14.52 MB

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-----------------|-----------------|----------|
| 314 | 6,844 | 87.7 | 1.1253 | **1.1046** | 1.90000 | 14,519,698 |
| 999 | 6,846 | 87.7 | 1.1256 | **1.1052** | 1.90050 | 14,517,302 |
| 1337 | 6,828 | 87.7 | 1.1261 | **1.1059** | 1.90130 | 14,525,480 |
| **Mean** | **6,839** | **87.7** | **1.1257** | **1.1052** | **1.90060** | |

Mixed quantization: 10 layers int6, 56 layers int5, no pruning needed.

Our merged PR 1019 (current SOTA): **1.88059 nats** (1.1138 BPB). Delta: **−0.00215 nats** (−0.0013 BPB). Our PR 549 (prior SOTA): **1.89002 nats** (1.1194 BPB). Delta vs PR 549: **−0.01158 nats**, Welch's t = −17.63, p < 0.01.

<details><summary><b>Prior results (val_bpb 1.1125, 3-seed)</b></summary>

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-----------------|-----------------|----------|
| 314 | 6,844 | 87.7 | 1.1253 | **1.1123** | 1.87802 | 14,519,698 |
| 999 | 6,846 | 87.7 | 1.1256 | **1.1124** | 1.87821 | 14,517,302 |
| 1337 | 6,828 | 87.7 | 1.1261 | **1.1129** | 1.87910 | 14,525,480 |
| **Mean** | **6,839** | **87.7** | | **1.1125** | **1.8784** | |

</details>

<details><summary><b>SLOT study (removed from submission — causality violation)</b></summary>

SLOT (Selective Logit Offset Tuning) optimizes a 512-dim delta vector at the last hidden layer using AdamW (lr=0.003, 5 steps) per sliding-window batch. It gave −0.0037 BPB (1.1125 → 1.1088), but **violates causality**: the delta has shape `[1,1,512]` and is optimized using targets at all positions, then applied to all positions — so position t's prediction is influenced by future tokens through the shared delta. Removed from submission code; results below are for reference only.

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-----------------|-----------------|----------|
| 314 | 6,812 | 87.7 | 1.1256 | **1.1086** | 1.87174 | 14,519,020 |
| 999 | 6,833 | 87.7 | 1.1259 | **1.1086** | 1.87187 | 14,522,798 |
| 1337 | 6,811 | 87.7 | 1.1265 | **1.1093** | 1.87306 | 14,526,779 |
| **Mean** | **6,819** | **87.7** | | **1.1088** | **1.8722** | |

Credit: PR 609 (saml212).

</details>

<details><summary><b>Prior results: fused kernels + Brotli only (val_bpb 1.1138, 3-seed)</b></summary>

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|---------------|-----------------|-----------------|----------|
| 314 | 7,176 | 83.4 | 1.1325 | **1.1133** | 1.8798 | 15,507,095 |
| 42 | 7,173 | 83.7 | 1.1325 | **1.1134** | 1.8800 | 15,498,065 |
| 999 | 7,176 | 83.5 | 1.1339 | **1.1146** | 1.8820 | 15,512,666 |
| **Mean** | **7,175** | **83.5** | | **1.1138** | **1.8806** | |

Delta vs PR 549: −0.00943 nats. Welch's t = −10.26, df ≈ 3.78, p < 0.01.

</details>

### Throughput recovery

| Submission | ms/step | BPB | What changed |
|---|---|---|---|
| Our PR 549 | 83.4 | 1.1194 | Leaderboard SOTA baseline |
| Our PR 1019 (merged SOTA) | 86.7 | 1.1147 | +Full GPTQ, +XSA-all, +BigramHash 3072. **+3.3ms regression.** |
| This PR (kernels only) | 83.5 | 1.1138 | +Fused MLP kernels, +Brotli. **Regression erased.** |
| **This PR (full stack)** | **87.7** | **1.1052** | +MLP 3.5×, +mixed int5/int6, +LR floor. |

Our PR 1019 (now merged as SOTA) traded throughput for quality — full Hessian GPTQ and BigramHash 3072×112 added 3.3ms/step. Fused MLP kernels recover that regression. Mechanistic analysis of that model identified MLP as the capacity bottleneck, leading to MLP 3.5× (enabled by mixed quantization + Brotli headroom).

## Changes vs our PR 1019

### 1. Fused MLP Kernels: Triton TMA Forward + CUTLASS EVT Backward

**Forward (Triton TMA):** Fuses `F.linear(x, up_w) → LeakyReLU(0.5) → square` into a single kernel. The 302MB intermediate never touches HBM.

**Backward (CUTLASS EVT):** Fuses `(go @ down_w.T) * act_grad` into a single CUTLASS 3.x kernel via Epilogue Visitor Tree. The elementwise multiply runs in the GEMM epilogue while tiles are still in registers — eliminating one 302MB write + read per layer.

**Key design insight — pre-computed activation gradient:** We store the activation gradient in the forward pass instead of the pre-activation:

```
Standard:  store pre, recompute grad in backward with branch
Ours:      act_grad = (pre > 0) ? 2·pre : 0.5·pre    ← one branch, forward only
           post    = 0.5 · act_grad · pre              ← branch-free recovery
           dpre    = (go @ W_down.T) * act_grad         ← branch-free backward
```

The identity `post = 0.5 · act_grad · pre` holds for both signs because:
- pre > 0: act_grad = 2·pre → 0.5 · 2pre · pre = pre² = LeakyReLU(0.5)(pre)² ✓
- pre ≤ 0: act_grad = 0.5·pre → 0.5 · 0.5pre · pre = 0.25·pre² = (0.5·pre)² ✓

This eliminates all branching from the backward, reducing the CUTLASS EVT epilogue to a trivial 3-node tree: `Sm90EVT<multiplies, AccFetch, AuxLoad>`. No conditionals in the kernel.

CUTLASS EVT is a hard dependency — no silent fallback.

<details><summary><b>Kernel benchmarks + incremental deltas (2×H100)</b></summary>

**Per-layer kernel timing:**

| Variant | dpre time | Δ per layer | Δ per step (×11) |
|---|---|---|---|
| cuBLAS unfused | 1.221 ms | baseline | baseline |
| Triton precomp | 1.105 ms | −0.116 ms | −1.275 ms |
| **CUTLASS Pingpong** | **1.073 ms** | **−0.148 ms** | **−1.623 ms** |

CUTLASS vs Triton: +0.032 ms/layer, +0.347 ms/step kernel-level.

**End-to-end training (35 steps, seed=42):**

| Config | Step avg | Δ |
|---|---|---|
| Triton fwd + Triton bwd | 313.90 ms | baseline |
| Triton fwd + CUTLASS EVT bwd | 313.47 ms | −0.43 ms |

Kernel-level 0.347ms translates to 0.43ms end-to-end (cache/scheduling interactions).

**8×H100:** 86.7ms (our PR 1019, unfused) → 83.5ms (this PR) = **−3.2ms/step (−3.7%)**.

</details>

<details><summary><b>Step-time profile — where all 313ms goes (2×H100, Nsight)</b></summary>

| Component | Share | |
|---|---|---|
| Flash Attention 3 (fwd+bwd) | 20.1% | FA3 forward alone is 17.5% — single largest kernel |
| **Fused MLP (Triton+CUTLASS)** | **13.5%** | **Our optimization target** |
| cuBLAS GEMMs (MLP bwd dW/dx, attn proj) | 19.1% | Remaining unfused matmuls |
| torch.compile fusions (cross-layer) | 21.6% | **Preserved — see below** |
| Unfused elementwise (LN, residuals) | 21.0% | |
| Communication + other | 4.7% | NCCL, copies, softmax |

**Why surgical fusion, not full-MLP autograd.Function:** The 21.6% from torch.compile's cross-layer fusions (RMSNorm backward, residual adds, RoPE backward) only exists because these ops are visible to the compiler. Wrapping the full MLP backward in `autograd.Function` makes it opaque to Inductor — all backward GEMMs plus cross-layer fusion run in eager mode, **2.7× slower net** (identified in our PR 670). We fuse only forward and one backward GEMM+pointwise, preserving the compiler's scope.

Top individual kernels:
| Kernel | Share |
|---|---|
| FA3 forward (device_kernel) | 17.3% |
| Fused MLP (\_fused\_leaky...) | 13.4% |
| cuBLAS GEMM (nvjet 256×128) | 12.2% |
| elementwise_kernel | 7.8% |
| vectorized_elementwise_kernel | 5.6% |
| vectorized_layer_norm_kernel | 4.3% |

Wall-clock breakdown: forward+backward compute ~94%, NCCL ~1.6%, CPU overhead ~4.1%.

</details>

### 2. Brotli-11 Compression (replaces LZMA-9)

−581 KB (−5.9%) vs LZMA-9. Independently discovered; PR 1089 (mikeapedia) also uses Brotli.

### 3. Memmap Multi-Shard Data Pipeline + GPU Prefetch

Coprime-stride sampling, daemon thread, CUDA stream prefetch. Credit: DeepReinforce (PR 726).

### 4. MLP 3.5× (1536 → 1792 hidden dim)

**Motivated by mechanistic analysis:** [SVD analysis of our PR 1019 model](https://abay.tech/posts/pr-1019-model-autopsy) showed MLP at 94.4% rank utilization (fully packed) while attention Q sat at 72.6% (spare capacity). The model was parameter-starved in MLP, not attention — so we made MLP wider.

Increases hidden dim from 3.0 × 512 = 1536 to 3.5 × 512 = 1792. Model goes from 27.07M to 29.95M params (+2.88M). At uniform int6, the 29.95M model compresses to 17.36 MB — 1.36 MB over the 16 MB limit. This is what makes mixed quantization (change 5) necessary.

Impact: −0.003 BPB from capacity, +13ms/step on 2×H100 (bigger GEMMs). Credit: PR 185 (dttdrv), PR 344 (aryanbhosale).

### 5. Mixed int5/int6 Quantization (Hessian-based)

**Motivated by mechanistic analysis:** [Per-matrix quantization sensitivity](https://abay.tech/posts/pr-1019-model-autopsy) showed MLP accounts for 80% of int6 quantization damage (MLP_down: +0.0039 BPB total, all Q matrices: +0.0003 BPB total — a 13× gap). Giving more bits to MLP is the optimal allocation.

Instead of uniform int6 for all layers, use int5 as default and promote the top 10 most sensitive layers to int6 based on Hessian trace ranking. Sensitivity = trace(H) where H = X^TX collected during GPTQ calibration. MLP projection layers in early blocks are most sensitive — they get int6; the remaining 56 layers get int5.

Uniform int5 loses ~0.019 BPB (catastrophic). Targeted Hessian-based allocation keeps quality loss under ~0.003 BPB while saving ~1.5 MB — exactly the headroom MLP 3.5× needs to fit under 16 MB. The wider MLP also made the model [3.6× less sensitive to quantization overall](https://abay.tech/posts/pr-1105-model-autopsy) — information distributed across more dimensions means no single weight is load-bearing.

Credit: mixed quant concept PR 76 (Will DePue), gradient-guided PR 332 (saml212), Hessian-based PR 1089 (mikeapedia).

### 6. LR Floor (0.05)

During warmdown, learning rate normally decays to 0. With `lr_floor=0.05`, it stops at 5% of peak instead. Prevents the optimizer from stalling, which helps with quantization-sensitive weight distributions still being refined at end of training.

Impact: ~0.001 BPB. Credit: PR 130 (mohosy).

### 7. Fast Causal N-Gram Tilt & Subword Certainty (-0.00243 BPB, 5.7x Faster)

**Architecture Shift: Sparse Auxiliary Memory**
This PR replaces the old eval-time n-gram mixing path with a fast, legal, single-pass causal n-gram tilt system. The core change is that the n-gram is no longer treated as a second language model. Instead, it acts as a sparse auxiliary memory that proposes a hinted token from the strict prefix, while the neural model remains the full normalized distribution. We then apply a one-token exponential tilt directly on the GPU.

**Motivation & Interpretability**
This work was guided by the interpretability results in our model-analysis stack and by the PR-1105 model autopsy. Those analyses showed that the model is not broadly weak at language modeling; it is specifically weak at exact copy/repetition. In particular, it has very limited induction capability, while much of the remaining loss is in categories like numbers, punctuation, and whitespace where generic short-order n-grams do not help much.

That changed the design target. Instead of building “better PPM everywhere,” we focused on the narrow places where n-grams are actually complementary:
- High-order token memory for exact repeats.
- Within-word memory for BPE completion.
- Word-start memory for local token prediction.

**The Key Insight: Mechanical Subword Certainty**
Initially, within-word BPE completions seemed redundant since the neural baseline already assigns high probability to these tokens. However, the most significant BPB drop (-0.00243) was unlocked by aggressively lowering the `within_threshold` from 0.80 to 0.25, allowing the expert to fire on 35.7% of positions.

Why it works: While the neural model knows subword patterns, it inherently hedges its bets by distributing probability mass across alternatives. The n-gram expert acts as a mechanical override, capturing the absolute certainty of BPE completions that the neural model refuses to assign a 1.0 probability to.

**Engineering Overhaul**
Previous attempts at n-gram blending using flat tables and Python/NumPy logic were bottlenecked by severe hash collisions and massive FFI overhead. Initial runs with a logistic mixer yielded a catastrophic +0.210 BPB degradation because collision noise was inflating token probabilities.

By migrating to an open-addressing scheme (64M entries, 26-bit) to store exact keys, we eliminated false positives, pushing token PPM accuracy to 82.3%. To solve the execution bottleneck, we deployed a highly optimized pipeline:
- **C++ Fused Kernels:** Moved the exponential tilt and blending logic entirely to custom C++ operators (`fused_expert_blend.cpp`, `ngram_blend.cpp`).
- **FFI Bottleneck Eradicated:** Switched to nanobind batch calls instead of per-token ctypes, achieving a 9,583× reduction in FFI overhead.
- **GPU Utilization:** By keeping the tilt on the GPU and eliminating blend_max wait times, GPU utilization spiked from ~10% to 87-94%.

**Results & Benchmarks**
- Validation BPB: -0.00243 (from the baseline).
- Evaluation Time (1xH100): 645s (Down from an estimated ~3800s for a serial Python implementation).
- Projected Time (8xH100): ~85s (5.7× faster than the 481s baseline established in PR 1145).
- N-gram Overhead: Reduced to just ~72s (pure C++ execution).

By trading brute-force multi-expert agreements for single-expert confidence scaling and targeted subword overrides, this architecture secures massive BPB gains while freeing up ~400 seconds of compute budget for further training or secondary adapters.

## Negative Results

- **SLOT (Selective Logit Offset Tuning):** −0.0037 BPB (1.1125 → 1.1088) but **violates causality** — the shared delta is optimized over all positions then applied to all positions, leaking future-token information into past predictions. Removed from submission. See collapsed SLOT study section above.
- **Turbo-Muon (AOL + Polar Express NS4):** +0.0018 BPB worse on 8×H100 AND artifact over 16MB. Early convergence advantage at step 500 doesn't hold at 7000+ steps. Reverted to standard NS5.
- **2:4 Structured Sparsity:** +0.672 BPB. Dead.
- **Calibration regression:** ECE increased from 0.24% (PR 1019 model) to 1.26% (this model) — the mixed int5/int6 quantization introduces slight overconfidence. Model entropy dropped from 1.899 to 1.847 nats (more confident) while accuracy dropped from 54.99% to 54.46%. Under investigation.

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA / 4 KV heads) | Baseline |
| MLP | **3.5× (1792)**, LeakyReLU(0.5)² | **This work** (concept: PR 185 (dttdrv), PR 344 (aryanbhosale)) |
| MLP Forward | **Fused Triton TMA kernel** | **This work** (profiling: our PR 670) |
| MLP Backward | **CUTLASS EVT Pingpong + pre-computed act_grad** | **This work** |
| Attention | XSA on all 11 layers | PR 478 (gowtham0992) |
| BigramHash | 3072 × 112 | Our PR 1019 (concept: PR 162 (raahilshah)) |
| RoPE | Partial (16/64 dims) | PR 315 (jfprincz) |
| LN Scale | 1/√(layer+1) | PR 315 (jfprincz) |
| VE128 | Layers 9-10 | PR 374 (unnir) |
| SmearGate | Position-mixing gate | PR 65 (aquariouseworkman) |
| U-Net skips | Encoder-decoder connections | PR 289 |
| Weight avg | EMA(0.997) + SWA(every 50) | PR 401 (newjordan) |
| Quantization | **Full Hessian GPTQ mixed int5/int6 (Hessian-based, AR self-gen)** | **This work** (GPTQ: PR 535 (raahilshah), mixed: PR 1089 (mikeapedia)) |
| Compression | **Brotli quality=11** | **This work** (independently: PR 1089 (mikeapedia)) |
| Data Pipeline | **Memmap multi-shard + GPU prefetch** | PR 726 (DeepReinforce) |
| Warmdown | 4000 iterations, **LR floor 0.05** | PR 364 (shikhar1729), **LR floor: PR 130 (mohosy)** |
| Optimizer | Parallel Muon (NS5) | Our PR 399 |
| Late QAT | STE at LR scale < 0.15 | PR 286 (chris-buckley) |
| Selective pruning | ±1 by reconstruction error | PR 609 (saml212) |
| Eval | Sliding window (stride=64) | Standard |
| Flash Attention 3 | Hopper kernels | PR 122 (mtybadger) |

**Calibration legality:** AR self-generated (64 seqs × 2048 tokens, temp=0.8). No val data, no train data accessed during quantization. Same method as our [PR 1019](https://github.com/openai/parameter-golf/pull/1019).

## Setup & Reproduction

```bash
# 1. Python dependencies
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece brotli

# 2. CUTLASS headers (header-only, no build needed for CUTLASS itself)
cd /opt && git clone --depth 1 --branch v3.7.0 https://github.com/NVIDIA/cutlass

# 3. Build CUTLASS EVT extension
cd cutlass_evt_fusion
CUTLASS_PATH=/opt/cutlass python3 setup.py build_ext --inplace
cd ..

# 4. Set library paths (auto-detect from Python packages)
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')"):$(python3 -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0] + '/lib')"):${LD_LIBRARY_PATH:-}

# 5. Download data
python3 data/cached_challenge_fineweb.py --variant sp1024

# 6. Train (3 seeds)
for SEED in 314 999 1337; do
  SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${SEED}.log
done
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)

