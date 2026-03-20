# Experiment: Lookahead/SNOO + LAWA (on top of experiment 001)

## Research Background

Deep research was conducted on advanced distributed optimization strategies for ultra-constrained
LM pretraining. The full report is summarized here as it motivates every design choice.

### The sharp minima problem with DDP

Standard synchronous DDP with large batch sizes (our 524K tokens/step) has a well-documented
tendency to converge into sharp, narrow minima. This is quantifiable via Hessian eigenvalue
analysis and SAM metrics. Sharp minima generalize poorly to held-out validation data.

The key insight from the DiLoCo (Distributed Low-Communication) literature (Douillard et al.,
2023) and Local SGD research: allowing optimization trajectories to diverge before averaging
acts as a geometric regularizer that filters out high-frequency gradient noise and avoids
high-curvature regions. The resulting models land in wider, flatter basins.

### Why Lookahead/SNOO instead of full DiLoCo

Full DiLoCo (K=8 independent workers) would split our batch across independent models, reducing
each worker to 65K tokens/step. Research shows this can actually degrade performance for small
(~10M param) models where the gradient variance from 65K tokens is already high.

The SNOO (Step-K Nesterov Outer Optimizer) variant is mathematically equivalent to single-worker
DiLoCo (K=1). It wraps the entire DDP group as one worker:
- Preserves full 524K-token batch size (maximum throughput per step)
- Still gets flat-minima-seeking regularization from the slow/fast weight interpolation
- Research on 160M LLaMA variants shows 1.5-2.5x convergence speedup over standard AdamW/Muon

### Why Nesterov for the outer optimizer

The outer optimizer aggregates pseudo-gradients (delta between fast and slow weights). Nesterov
momentum anticipates the trajectory by evaluating at the projected future position. Research
shows it significantly outperforms plain SGD or Adam for this role because it stabilizes the
updates when the fast weights have diverged substantially over H inner steps.

### MuLoCo: Muon + Local SGD synergy

Using Muon as the inner optimizer (our existing setup) creates what the literature calls
"MuLoCo." The Muon optimizer produces pseudo-gradients that are more directionally correct than
AdamW, especially as worker count increases. Key finding: MuLoCo completely avoids the
performance degradation sometimes seen with AdamW-based DiLoCo at K=8 scale.

Critical for our challenge: MuLoCo pseudo-gradients are extremely resilient to quantization.
Research shows 8-bit and even 2-bit Muon maintains mathematical stability, avoiding the
gradient collapse seen with compressed AdamW. Since our final artifact is int8+zlib compressed,
an optimizer natively resilient to low-precision states prevents representation collapse during
post-training quantization.

### LAWA (Latest Weight Averaging)

Standard practice: rely on LR decay to settle into a minimum. In our brutally truncated
10-minute window, standard decay forces premature suboptimal convergence.

LAWA (Kaddour et al., 2022) demonstrates that averaging checkpoints along the training
trajectory significantly accelerates convergence, especially when paired with higher learning
rates. The high LR enables aggressive landscape exploration; the averaging synthesizes a model
capturing that broad exploration.

Key finding: LAWA substantially outperforms EMA and SWA because it deliberately leverages the
high-variance checkpoints from before LR decay. A FIFO buffer of checkpoints sampled with
spacing during the final ~25% of training captures both the diverse high-LR phase and the
early-warmdown convergence phase.

### Why higher LR (0.032 -> 0.05)

The Lookahead wrapper acts as a variance dampener: the slow weights smooth out the high-LR
oscillations of the fast weights. This makes higher LRs safe that would otherwise cause
instability. The FP16 embed submission independently validated MATRIX_LR=0.06 with
WARMDOWN_ITERS=3600. We use 0.05 as a conservative middle ground (between seq2048's 0.032
and fp16_embed's 0.06), relying on Lookahead to manage the extra variance.

### Hyperparameter choices

| Parameter | Value | Source |
|---|---|---|
| LOOKAHEAD_H | 30 | DiLoCo paper: H=30-50 optimal for 10-min training constraints |
| OUTER_LR | 0.7 | DiLoCo paper standard; acts as scaling factor for pseudo-gradient |
| OUTER_MOMENTUM | 0.9 | DiLoCo paper standard for Nesterov outer optimizer |
| LAWA_START_FRAC | 0.75 | Captures ~7 pre-warmdown + ~13 warmdown checkpoints |
| LAWA_INTERVAL | 150 | Gives ~19 checkpoints in the LAWA window, good spread |
| LAWA_MAX_CKPTS | 20 | ~768MB CPU memory, trivial vs 80GB available |
| MATRIX_LR | 0.05 | Midpoint between validated 0.032 and 0.06, Lookahead-stabilized |
| SCALAR_LR | 0.05 | Matched to MATRIX_LR |

## Hypothesis

Wrapping our existing DDP+Muon training in a Lookahead/SNOO outer optimizer (H=30, Nesterov
momentum) will guide the model toward wider, flatter minima that generalize better AND survive
int8 quantization with less degradation. Pairing with LAWA checkpoint averaging during the
final 25% of training will further smooth the final weights. The higher LR (0.05 vs 0.032)
is safe under Lookahead and enables more aggressive exploration that LAWA can then consolidate.

Expected outcome: post-quant val_bpb < 1.16 (vs experiment 001's expected ~1.17).

## Plan

Build on experiment 001's `train_gpt.py`. Three additions:

### 1. Lookahead/SNOO outer optimizer
- Maintain `slow_params` (copy of all model params) and `outer_velocity` (zeros)
- Every H=30 inner DDP steps:
  - Compute pseudo-gradient: `g = slow_params - fast_params`
  - Nesterov update: `v = momentum * v + g; slow = slow - outer_lr * (g + momentum * v)`
  - Reset fast params to slow params
- Re-sync slow_params after the compile warmup phase

### 2. LAWA checkpoint averaging
- Starting at 75% wall-clock elapsed, save slow_params to CPU every 150 steps
- FIFO buffer capped at 20 snapshots
- After training, simple element-wise average of all snapshots -> final model
- If no LAWA checkpoints (e.g. short run), fall back to slow_params

### 3. Higher learning rate
- MATRIX_LR: 0.032 -> 0.05
- SCALAR_LR: 0.032 -> 0.05

### Algorithm flow
```
for step in range(iterations):
    # Standard DDP forward/backward/optimizer step (unchanged)
    loss = model(x, y)
    loss.backward()
    optimizers.step()

    # Lookahead outer update every H steps
    if step % H == 0:
        for each param:
            g = slow - fast
            velocity = momentum * velocity + g
            slow -= outer_lr * (g + momentum * velocity)   # Nesterov
            fast = slow                                     # reset

    # LAWA: save slow_params snapshot in final 25% of training
    if elapsed > 75% and step % 150 == 0:
        lawa_buffer.append(copy of slow_params)

# After training
final_weights = average(lawa_buffer)  # or slow_params if no LAWA
quantize + sliding window eval
```

### Success metric
Post-quant val_bpb < 1.16 (improvement over experiment 001's expected ~1.17)

### Run command (8xH100)
```bash
RUN_ID=exp002_lookahead_lawa \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  experiments/002_lookahead_snoo_lawa/train_gpt.py
```

All defaults baked in. Tunable via env vars: LOOKAHEAD_H, OUTER_LR, OUTER_MOMENTUM,
LAWA_START_FRAC, LAWA_INTERVAL, LAWA_MAX_CKPTS, MATRIX_LR, SCALAR_LR.

### Risks
- Higher LR might destabilize despite Lookahead (mitigated: easily reverted via env var)
- LAWA averaging might dilute fine-grained features learned late in training
- Lookahead overhead (~9.6M param copy every 30 steps) should be negligible but untested
- The Nesterov outer optimizer can overshoot (outer_lr=0.7 is aggressive); may need tuning

## Observations

| Run | Seed | Steps | Pre-quant BPB | Post-quant BPB | Artifact Size | Notes |
|-----|------|-------|---------------|----------------|---------------|-------|
|     |      |       |               |                |               |       |

## Post-mortem

(pending results)
