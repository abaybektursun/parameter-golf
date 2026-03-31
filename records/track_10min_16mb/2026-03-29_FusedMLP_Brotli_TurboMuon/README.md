# Record: Fused Triton MLP + Brotli Compression + Turbo-Muon

Continuation of PR 1019. Three hardware/systems optimizations stacked on top of our previous best (1.1147 BPB).

## Changes vs PR 1019

### 1. Fused Triton MLP Kernel (forward-only)
Fuses `F.linear(x, up_w) -> LeakyReLU(0.5) -> square` into a single Triton TMA kernel. The 302MB intermediate activation per layer never touches HBM. Backward uses explicit cuBLAS matmuls (avoids Inductor bypass issue from PR 670).

- Inspired by approach in PR 1072 (vimeto)
- Validated: -8ms/step on 2xH100 (-2.5%), projects to ~-17ms on 8xH100

### 2. Brotli-11 Compression (replaces LZMA-9)
Drop-in replacement. Brotli quality=11 saves 581 KB (-5.9%) vs LZMA preset=9 on int6 quantized weights. Byte-shuffle tested and found to provide no additional benefit.

- Independently discovered; also used in PR 1089 (mikeapedia)
- Frees headroom for more BigramHash buckets or mixed bit allocation

### 3. Turbo-Muon (AOL + Polar Express + NS4)
Replaces standard 5-iteration Newton-Schulz with AOL-preconditioned 4-iteration variant using Polar Express per-iteration optimal coefficients (Amsel et al., arXiv:2505.16932). Post-NS row/col L2 normalization for stability.

- From PR 1089 (mikeapedia)
- Drop-in replacement for `zeropower_via_newtonschulz5()` inside existing Parallel Muon
- Neutral throughput on 2xH100 (AOL cost ~= 1 NS iteration saved), but better convergence quality (-0.044 nats train loss at step 500)

### 4. Memmap Multi-Shard Data Pipeline + GPU Prefetch
Coprime-stride sampling across multiple shards with daemon thread CPU batch building and CUDA stream prefetch. Better data diversity per batch.

- From PR 726 (DeepReinforce)
- Already validated in our stack

## Architecture (unchanged from PR 1019)
- 11L/512d, 8 heads, 4 KV heads, GQA
- LeakyReLU(0.5)^2, MLP 3x (1536)
- XSA on all 11 layers
- BigramHash 3072 x dim=112
- Partial RoPE 16/64, LN 1/sqrt(layer+1)
- VE128 layers 9-10, SmearGate, U-Net skips
- EMA(0.997) + SWA(every 50), Late QAT (STE at scale<0.15)
- Full Hessian GPTQ int6 (AR self-gen calibration)
- Sliding window eval stride=64

## 2xH100 Validation Results (PENDING 8xH100)

| Metric | PR 1019 baseline | This work | Delta |
|---|---|---|---|
| Step avg (2xH100) | 327.51ms | ~319ms | -8ms (-2.5%) |
| Artifact size | 10.09 MB | ~9.5 MB (Brotli) | -0.6 MB |

8xH100 results pending.

## Run command
```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Requirements
```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece brotli
```
