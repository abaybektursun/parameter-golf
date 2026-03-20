# Experiment: Kernel Speedrun — Maximize Training Throughput with Custom Kernels

## Hypothesis

The baseline wastes ~59% of step time on overhead (kernel launches, HBM traffic, unfused ops). Custom Triton/CUDA kernels and compile-time optimizations can cut step time from ~52ms to ~25-35ms on 8xH100, yielding 40-100% more training steps in the same 10-minute budget. More steps = lower val_bpb.

## Plan

- Use autoresearch agent loop but optimized for **speed** (ms/step), not val_bpb directly
- Short benchmark runs (~100 steps post-warmup, ~60s each) to iterate fast
- Full 10-min validation run only when a speed improvement is confirmed
- Target optimizations from `experiments/optimization_opportunities.md`:
  1. `mode="max-autotune"` in torch.compile (CUDA graphs)
  2. Fused cross-entropy + logit softcap (eliminate 268MB logit tensor)
  3. Fused ReLU² MLP (eliminate 2.34GB HBM traffic across 9 layers)
  4. Triton autotune config (deeper search for better tile sizes)
  5. Batch size warmup schedule
  6. Custom Triton kernels for RMSNorm+RoPE fusion
  7. Any other kernel-level optimizations

## Success Metric

- Baseline: ~52ms/step on 8xH100 (~27% MFU)
- Target: <35ms/step (>40% MFU)
- Constraint: val_bpb must not regress more than 1% vs baseline at same step count

## Observations

| Run | Seed | Steps | ms/step | MFU% | val_bpb | Artifact Size | Notes |
|-----|------|-------|---------|------|---------|---------------|-------|
|     |      |       |         |      |         |               |       |

## Post-mortem

(pending results)
