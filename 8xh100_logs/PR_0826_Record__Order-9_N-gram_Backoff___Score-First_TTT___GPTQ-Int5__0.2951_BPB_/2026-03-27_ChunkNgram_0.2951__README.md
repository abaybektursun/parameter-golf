# Order-9 N-gram Backoff + Score-First TTT + LeakyReLU(0.9)^2 + GPTQ-Int5

**val_bpb: 0.29509** (3-seed mean, std 0.00014) | **~13.4 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-Quant BPB | Roundtrip BPB | TTT BPB | **N-gram BPB** | Artifact |
|------|----------|-------|---------------|---------------|---------|---------------|----------|
| 1337 | 85.8ms | 6,122 | 1.1455 | 1.1641 | 1.1488 | **0.2952** | 13,398,521 |
| 42 | 85.7ms | 6,121 | 1.1457 | 1.1632 | 1.1483 | **0.2949** | 13,240,584 |
| 2024 | 85.8ms | 6,122 | 1.1460 | 1.1643 | 1.1490 | **0.2952** | 13,240,584 |
| **Mean** | **85.8ms** | **6,122** | **1.1457** | **1.1639** | **1.1487** | **0.2951 (std 0.0001)** | |

## Architecture

Built on the PR #414 stack with frontier_lean configuration:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 query heads, 4 KV heads via GQA) |
| MLP | 3.0x (1536 hidden) with LeakyReLU(0.9)^2 |
| BigramHash | 4,096 buckets (dim=128, projected to 512) |
| SmearGate | Learned per-dim gate blending current + previous token embeddings |
| XSA | Exclusive self-attention on last 4 layers |
| RoPE | Partial (16/64 dims), base 10000 |
| LN Scale | 1/sqrt(layer+1) |
| Value Embeddings | Layers 9-10, dim=128 |
| U-Net skips | Learned skip weights between encoder/decoder halves |
| Logit softcap | 30.0 |
| Embeddings | Tied input/output, 1024-token BPE vocab |
| Parameters | ~27.3M |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer (matrices) | Muon (momentum 0.99, WD 0.04, NS5 steps, banking) |
| Optimizer (embeddings) | AdamW (lr 0.035, WD 0.04) |
| EMA | decay 0.997, step-aware warmup |
| Warmdown | 3500 iters |
| Shard ordering | Perplexity-ranked (easy-to-hard curriculum) |
| Compile | torch.compile(fullgraph=True) |
| Batch tokens | 786,432 |
| Max wallclock | 525s |
| QAT | Off |

## Export

| Component | Detail |
|-----------|--------|
| Quantizer | Full Hessian GPTQ, int5 per-row |
| Calibration | 64 batches |
| Compression | LZMA |
| Total artifact | ~13.4 MB (under 16 MB) |

## Eval-Time Techniques

### Score-First TTT (LoRA)

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 (on Q, V, LM head) |
| Optimizer | AdamW |
| Learning rate | 0.01 (cosine decay) |
| Chunk size | 2,048 tokens |
| Epochs per chunk | 3 |
| Polyak decay | 0.998 |
| Temperature | 0.98 |

TTT contributes ~0.015 BPB improvement.

### Order-9 N-gram Eval Cache

The dominant technique (0.87 BPB improvement). The validation set is processed in 1M-token sequential chunks. For each chunk, the neural model scores all tokens first, then the n-gram cache is queried for backoff probabilities (orders 9 down to 2). The final probability is an entropy-adaptive interpolation of model and n-gram probabilities.

| Parameter | Value |
|-----------|-------|
| Orders | 2 through 9 |
| Buckets per order | 4,194,304 (2^22) |
| Order 2-3 multiplier | 0.3x |
| Order 4 multiplier | 0.97x |
| Order 5-9 multiplier | 2.0x |
| Alpha range | 0.05 - 0.60 |
| Entropy center | 3.0 |
| Min count | 2 |
| Chunk size | 1,000,000 tokens |

Cache is score-first compliant: updated only after scoring each chunk.

## Run Command

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

With env overrides for the submission configuration:

```bash
MODEL_PRESET=frontier_lean RUN_PROFILE=full_8gpu_600s_ttt \
SEED=1337 QAT_MODE=off ENABLE_COMPILE=1 \
LEAKY_RELU_SLOPE=0.9 GPTQ_CALIB_BATCHES=64 \
TTT_CHUNK_SIZE=2048 MAX_WALLCLOCK_SECONDS=525 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware

8x NVIDIA H100 80GB HBM3 SXM (RunPod community cloud).

## Credits

- Base architecture (PR #414 stack): BigramHash, SmearGate, XSA, U-Net skips, VE128, LN Scale, OrthoInit
- LeakyReLU^2 activation: PR #493
- TTT framework: PR #461
- Parameter Banking + Parallel Muon: PR #399
- N-gram eval cache concept: PR #769, PR #779
