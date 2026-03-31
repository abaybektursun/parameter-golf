# Parameter-Golf-Kernels: Complete Experiment Summary

Source: `parameter-golf-kernels` repository. 100+ experiments, 4 merged PRs (#399, #473, #549, #593).

---

## Best Result: PR #593 — 1.1170 BPB (#1 Non-TTT Leaderboard)

11L/512d/8h/4kv GQA, LeakyReLU(0.5)², BIGRAM=1536, Full Hessian GPTQ int6, Parallel Muon + Parameter Banking, 83.5ms/step, 7185 steps, 15.96MB artifact.

Progression: #399 (1.1247) → #473 (1.1220) → #549 (1.1194) → #593 (1.1170)

---

## Major Wins

### 1. Full Hessian-Aware GPTQ (−0.0048 BPB) — Largest Single Win

Int6 quantization becomes a quality **gain** with Hessian-aware error compensation. Collect H = X^T X via forward hooks on 256 calibration batches, then Cholesky-based error compensation with activation ordering. Post-GPTQ 1.1170 **improves** on pre-quant 1.1218 by 0.0048 BPB.

Key insight: GPTQ already compensates for quantization error, leaving nothing for TTT to recover — explains why TTT gain disappears when combined with Full GPTQ.

### 2. LeakyReLU(0.5)² (−0.003 BPB)

Replace `torch.relu` with `F.leaky_relu(negative_slope=0.5)` in MLP. Preserves negative gradient flow. Tradeoff: weights ~100KB less compressible, forced BIGRAM down from 3072 to 1536.

Variants tested: LeakyReLU(0.3)² — same result. Star-ReLU (learned scale/bias) — worse, +6ms overhead.

### 3. Parameter Banking + Parallel Muon (−7ms/step, 3.4% speedup)

**Banking:** Replace 66 individual `nn.Linear` with 4 contiguous 3D `nn.Parameter` tensors (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`). Newton-Schulz goes from 66 sequential small GEMMs (19.7ms) to 3 batched `torch.bmm` calls (1.31ms) — 15× faster.

**Parallel Muon:** Reduce-Scatter → local NS5 on shard → async All-Gather. Bank N all-gather overlaps with bank N+1 NS compute. DDP incompatible with banking; manual comm scheduling required.

Net: ~82ms/step vs ~89ms baseline = ~770 extra steps in 600s.

Limitation: speed gains don't help SWA-based submissions. Tested on PR #414 (SWA-based): banked=1.1237 vs baseline 1.1233 = +0.0004 worse. Only helps EMA-only configs.

---

## Exhaustive GEMM Profiling: cuBLAS is Optimal

12+ hours testing, 20+ experiments, 6 approaches. **No lossless kernel-level speedup exists at d=512.**

| Approach | Result |
|----------|--------|
| CUTLASS 3.x/4.x TMA+WGMMA | 3-10% slower |
| CUTLASS Stream-K | 1-4% slower |
| cuBLASLt exhaustive search | Identical to default |
| Triton autotune | 9-19% slower |
| QKV fusion (with GQA) | 3-17% slower |
| FP8 training | No speedup (memory-bound at d=512) |

Root cause: K=512 pipeline depth = 512/64 = 8 iterations, insufficient to hide TMA latency (~200 cycles). Practical ceiling ~60% roofline; cuBLAS at 48% is near hardware limit.

**H100 memory compression artifact:** Synthetic benchmarks (cudaMemset, BlockFillRandom) inflate by 25-50%. Initial CUTLASS "speedups" were 15-53% on synthetic data but evaporated on real `torch.randn` data. Rule: always benchmark with `torch.randn` or actual training tensors.

---

## torch.compile Already Handles Fusion

| Fusion | Status |
|--------|--------|
| Cross-entropy + softcap(tanh) + backward | Single 0.43ms Triton kernel |
| LeakyReLU² + residual backward | Fused |
| RMSNorm backward + residual + RoPE backward | Fused |
| Custom Triton fused norm+residual | 0.136ms — ties with torch.compile exactly |

Custom fused GEMM+activation: forward 1.82× faster, but `torch.autograd.Function` backward runs in eager mode (2-3× slower). Net fwd+bwd: 2.7× SLOWER. torch.compile treats cuBLAS as atomic; can't fuse GEMM epilogue. Would require CUTLASS EVT (C++ CUDA).

**Lesson: don't write custom Triton unless solving something Inductor fundamentally can't express.**

---

## Legal TTT: All 22 Experiments Worse Than Non-TTT

Score-first constraint: every token scored under `inference_mode()` BEFORE model adapts. Per 32K chunk: Phase 1 Score, Phase 2 Train (3 epochs SGD+momentum).

Best legal TTT: 1.1177 BPB. Non-TTT baseline: 1.1171 BPB. TTT gain disappears when combined with Full GPTQ (GPTQ already compensates for quantization error).

Timing: legal TTT adds 400-1600s to eval. Only 3-epoch configs (~400s) fit within eval budget.

---

## BigramHash Scaling

| BIGRAM | BPB | Artifact |
|--------|-----|----------|
| 1536 | 1.1171 | 15.88MB ✓ |
| 2048 | 1.1172 | 16.08MB ✗ |
| 2560 | ~1.1161 | 15.99MB (marginal) |
| 3072 | 1.1163 | 16.07-16.19MB ✗ |
| 4096 | 1.1165 | 16.19MB ✗ |

Each +1K buckets ≈ +256KB compressed artifact. Hash embeddings are high-entropy and compress poorly.

---

## What Didn't Work

| Approach | Why |
|----------|-----|
| Polar Express optimizer | Weights compress 190KB worse → over 16MB |
| QKV fusion | GQA split creates non-contiguous views, overhead > GEMM savings |
| FP8 forward | Already memory-bound at d=512 |
| Custom fused CE | torch.compile already does it |
| Custom GEMM kernels | CUTLASS/Triton all slower than cuBLAS |
| torch.library.custom_op | Makes kernels opaque to Inductor, breaks fusion |
| NorMuon all-reduce | Fixes shard-NS quality penalty (+0.0005) but adds 2.5ms overhead |
| LAWA averaging | 1.1259 vs SWA 1.1235 |
| TrigramHash | Too many collisions at 1536 buckets |
| Int5 quantization | Loses 0.019 BPB; requires working QAT |
| Value Residual + Gated Attention | +6ms overhead, breaks torch.compile with banking |
| Batch size 1M tokens | 108ms/step, fewer steps → +0.0028 BPB worse |
| Batch size 524K/655K | OOM in GPTQ/eval |
| Hessian all-reduce across GPUs | Only −0.0002 BPB — not worth complexity |
| AdamW TTT | Overshoots, overfits to current chunk |

---

## Quantization & Compression

**Int6 GPTQ pipeline:**
1. Collect H = X^T X via forward hooks (256 batches)
2. Activation ordering (actorder) permutation
3. Cholesky error compensation, block_size=128
4. Multi-percentile clip search (0.9990, 0.9995, 0.9999, 0.99999, 1.0)

Post-training pipeline: ~150-200s total (GPTQ calibration ~90s, quant ~5s, eval ~90s). All untimed.

Late QAT: improves compression ~1-2% but triggers torch.compile recompilation → OOM on larger models.

---

## Operational Lessons

1. **Stale torch.compile cache:** Clear `/tmp/torchinductor_root`, `~/.cache/torch_compile`, `~/.triton/cache` when switching architectures. 14× slowdown otherwise.
2. **Thermal throttling:** H100 degrades after 350s sustained 8-GPU load (82ms → 130ms+).
3. **Stale GPU processes:** Always kill all Python and verify `nvidia-smi` shows 0 MiB before runs.
4. **FineWeb shards are homogeneous:** KL divergence 0.000171-0.001042 between shards. Data selection/ordering is not a viable optimization path (~0.00005 BPB).
5. **1ms ≈ 0.006 BPB:** The rough conversion factor for step-time improvements.

---

## Competitive Intelligence

| PR | BPB | Key Differentiator |
|----|-----|--------------------|
| #606 | 1.1162 | Soft-Round QAT + int5 + 33.6M params + AdamW TTT |
| #569 | 1.1175 | Full GPTQ + VRL (Value Residual Learning) |
| #545 | ~1.12 | Int5 GPTQ, 33.6M params, MLP 1792, BIGRAM 8192 |

Competitive gaps: (1) Soft-Round QAT works with torch.compile unlike our STE approach, (2) larger models (33.6M+ at int5 vs our 26.8M at int6), (3) AdamW TTT on larger models recovers more quant damage.

---

## Open Opportunities

1. **Fix Late QAT under torch.compile** — free BPB, use `torch.compiler.allow_in_graph` or pre-enable from step 0
2. **Soft-Round QAT** — tanh-based differentiable rounding with alpha annealing, works with torch.compile
3. **Scale to 33.6M+ params** — fit in 16MB via int5 + better compression
4. **d=768+ architecture** — 10-13% better GEMM utilization (more pipeline depth)
5. **Coprime-stride data pipeline** — PR #1060 technique, more diverse batches (−0.0025 BPB)
6. **Parallel GPTQ** — split layers across GPUs, save ~4s post-training

---

## Time Breakdown (85ms baseline → 82ms optimized)

| Phase | Baseline | Optimized | Notes |
|-------|----------|-----------|-------|
| Forward (BF16 GEMMs + FA3) | ~28ms | ~28ms | cuBLAS optimal, no headroom |
| Backward (2× GEMMs + FA3 bwd) | ~42ms | ~42ms | Hardware limit, no savings found |
| Optimizer (NS5) | ~10ms | ~1.3ms | Batched banking = 15× faster |
| Comm + EMA | ~5ms | ~3ms | Parallel Muon overlap |
| **Total** | **~85ms** | **~82ms** | **−3.5%** |

Theoretical floor ~38ms requires FP8 backward (lossy), wider model, or better comm — none viable under current constraints.

---

*Source: parameter-golf-kernels repository, 100+ experiments conducted March 2026.*
