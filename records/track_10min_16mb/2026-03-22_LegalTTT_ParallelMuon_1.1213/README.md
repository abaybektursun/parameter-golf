# Legal Score-First TTT + Parallel Muon + Parameter Banking

**val_bpb: 1.1213** (legal TTT, seed 1337) | **15.84 MB** | 8×H100 SXM, 600s training + 400s TTT eval

## Key Result

| Metric | Value |
|--------|-------|
| Pre-TTT int6 sliding bpb (stride=64) | 1.1234 |
| **Post-TTT bpb (legal score-first)** | **1.1213** |
| TTT improvement | **-0.0021** |
| Training time | 600s (7,278 steps @ 82.3ms) |
| TTT eval time | 400s |
| Artifact | 15,841,722 bytes |

## Legal TTT Protocol (from PR #461)

Every validation token is **scored BEFORE any weight update** that could use it:

```
for each 32K-token chunk of val data:
    Phase 1 — SCORE: sliding window eval under torch.inference_mode()
              Record per-token NLL. This is the official score.
    Phase 2 — TRAIN: SGD(lr=0.002, momentum=0.9) for 3 epochs
              Freeze first 2 blocks. Grad clip 1.0. Cosine LR decay.
              Model adapts, improving predictions for FUTURE chunks only.
```

Scoring under `inference_mode()` guarantees no gradient computation or weight mutation during scoring. The chunk ordering ensures strict causal legality.

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | SGD + momentum(0.9) |
| Learning rate | 0.002 (cosine decay across chunks) |
| Epochs per chunk | 3 |
| Frozen blocks | First 2 of 11 |
| Gradient clip | 1.0 |

## Training Architecture

Built on PR #414's stack with Parameter Banking + Parallel Muon optimizer:

- 11L, 512d, 8H/4KV, MLP 3× (relu²)
- XSA on last 4 layers, Partial RoPE (16/64 dims), LN Scale
- SmearGate, BigramHash(2048), VE128 on layers 9-10
- EMA(0.997) + Tight SWA(every 50)
- GPTQ-lite int6 quantization + lzma compression
- **Parameter Banking**: 4 contiguous 3D banks replace 66 nn.Linear weights
- **Parallel Muon**: No DDP for banks. Post-backward reduce-scatter → local NS → all-gather

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=2 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **TTT recipe**: PR #461 by @anantdgoel — legal score-first TTT with SGD+momentum, selective freezing
- **Base model**: PR #414 by @signalrush — GPTQ-lite, VE128, Tight SWA, warmdown=3500
- **Optimizer**: Parameter Banking + Parallel Muon (arXiv:2511.07464)
