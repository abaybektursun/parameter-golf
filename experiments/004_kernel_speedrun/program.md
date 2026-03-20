# Kernel Speedrun — Autonomous Speed Optimization

Autonomous experiment loop for maximizing training throughput via custom Triton/CUDA kernels and compile-time optimizations.

## Goal

**Minimize ms/step** (maximize tokens/sec and MFU%) without degrading model quality. Every ms saved per step = more training steps in the 10-minute budget = lower final val_bpb.

This is NOT about hyperparameter tuning. This is about making the same computation run faster.

## Starting point

Your working copy (`train_gpt.py`) is the **current best model from the 8xH100 autoresearch run**:
- 9 layers, 512 dim, 8 heads, 2 KV heads (GQA), MLP 2x
- seq_len=2048, tied embeddings, vocab=1024
- matrix_lr=0.05, tied_embed_lr=0.03784, qk_gain_init=1.7, warmdown=100
- Baseline step time on 8xH100: ~52ms/step, ~27% MFU

**Do NOT change the model architecture or hyperparameters.** Only optimize how fast the same computation executes.

## Setup

1. You are on branch `autoresearch/kernels` (or create it if it doesn't exist).
2. Your working copy is `experiments/004_kernel_speedrun/train_gpt.py`.
3. Results go in `experiments/004_kernel_speedrun/results.tsv`.
4. Read `experiments/optimization_opportunities.md` — it has a prioritized list of kernel optimizations with expected impact.

## How speed benchmarking works

**You do NOT need 10-minute runs to measure speed.**

After torch.compile warmup (the script does 20 warmup steps), step time is stable. You only need ~100 steady-state steps to get reliable ms/step. Use this for fast iteration:

```bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
ITERATIONS=150 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=120 \
  torchrun --standalone --nproc_per_node=$NUM_GPUS \
  experiments/004_kernel_speedrun/train_gpt.py > run.log 2>&1
```

This runs ~150 steps (20 warmup + ~130 real) in about 60-90 seconds. Extract speed:

```bash
grep "step_avg:" run.log | tail -1
```

The `step_avg` value (in ms) is your metric. Lower = better.

**Only run a full 10-minute validation run** when you've confirmed a speed improvement and want to verify quality isn't degraded:

```bash
torchrun --standalone --nproc_per_node=$NUM_GPUS \
  experiments/004_kernel_speedrun/train_gpt.py > run.log 2>&1
```

## What you CAN modify

- `experiments/004_kernel_speedrun/train_gpt.py` — the only file you edit
- You can add Triton kernels inline (as `@triton.jit` functions) or import from the `kernels` package (already in requirements.txt)
- You can write custom CUDA extensions inline using `torch.utils.cpp_extension.load_inline`
- You can change torch.compile options, SDP backends, memory allocation strategies
- You can rewrite any module (MLP, Attention, RMSNorm, etc.) with fused implementations

## What you CANNOT modify

- The model's mathematical behavior (same architecture, same forward pass semantics)
- The evaluation harness (`eval_val`)
- Dependencies beyond `requirements.txt` (but `kernels`, `torch`, `triton` are already available)

## Logging results

TSV with 7 columns:

```
commit	ms_per_step	mfu_percent	val_bpb	artifact_bytes	status	description
```

- `ms_per_step`: steady-state average from `step_avg` (use 0.0 for crashes)
- `mfu_percent`: from training log (use 0.0 for crashes)
- `val_bpb`: from full run only (use `-` for speed-only benchmarks)
- `artifact_bytes`: from full run only (use `-` for speed-only benchmarks)
- `status`: `keep`, `discard`, or `crash`
- `description`: what kernel/optimization was tried

Example:

```
commit	ms_per_step	mfu_percent	val_bpb	artifact_bytes	status	description
a1b2c3d	52.1	27.0	-	-	keep	baseline speed measurement
b2c3d4e	45.3	31.1	-	-	keep	mode=max-autotune in torch.compile
c3d4e5f	38.7	36.4	1.2100	15200000	keep	fused cross-entropy + softcap (full run)
d4e5f6g	0.0	0.0	-	-	crash	custom triton RMSNorm (shape mismatch)
```

## The experiment loop

LOOP FOREVER:

1. Read `experiments/004_kernel_speedrun/results.tsv` and `learnings.md` to understand what's been tried.
2. Read `experiments/optimization_opportunities.md` for the prioritized optimization list.
3. Pick the next optimization to try. Start with Tier 1 (highest impact, lowest effort):
   - `mode="max-autotune"` in torch.compile
   - Fused cross-entropy + logit softcap (Triton kernel)
   - Fused ReLU² MLP (Triton kernel with backward recomputation)
4. Edit `experiments/004_kernel_speedrun/train_gpt.py`.
5. git commit.
6. Run a **speed benchmark** (150 steps, ~60-90 seconds):
   ```bash
   NUM_GPUS=$(nvidia-smi -L | wc -l)
   ITERATIONS=150 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=120 \
     torchrun --standalone --nproc_per_node=$NUM_GPUS \
     experiments/004_kernel_speedrun/train_gpt.py > run.log 2>&1
   ```
7. Extract `step_avg` from the log. If it crashes, check `tail -50 run.log`.
8. Log to results.tsv.
9. If ms/step improved: keep. If not: `git reset --hard` and revert.
10. After confirming speed gains, do a **full 10-min validation run** to verify quality:
    ```bash
    torchrun --standalone --nproc_per_node=$NUM_GPUS \
      experiments/004_kernel_speedrun/train_gpt.py > run.log 2>&1
    ```
11. Push + S3 sync: `git push origin HEAD` and `aws s3 sync . s3://fuelos-autoresearch/latest-1xa100/ --profile fuelos --exclude 'data/*' --exclude '.git/*' --exclude '__pycache__/*' --exclude '.venv/*'`
12. Update `experiments/004_kernel_speedrun/learnings.md` with what you learned.

## Key references

- `experiments/optimization_opportunities.md` — full analysis with FLOP/memory estimates
- modded-nanogpt's `triton_kernels` module has battle-tested implementations of:
  - `FusedSoftcappedCrossEntropy` (cross-entropy + tanh softcap in one kernel)
  - `FusedLinearReLUSquareFunction` (MLP forward+backward with activation recomputation)
- The `kernels` package (already installed) may have pre-built fused kernels

## Writing Triton kernels

You can write Triton kernels inline in train_gpt.py:

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # ... compute ...
    tl.store(output_ptr + offsets, result, mask=mask)
```

Key Triton tips:
- Use `tl.constexpr` for block sizes to enable compile-time optimization
- Fuse element-wise chains (norm + rope, softcap + cross-entropy) to avoid HBM round-trips
- For backward passes, consider activation recomputation instead of storing intermediates
- Test numerical accuracy: compare output against the original implementation with `torch.allclose`

## NEVER STOP

The human is away. Keep iterating. If one optimization fails, try the next. If you exhaust the tier list, look for new fusion opportunities in the code. Profile with `torch.cuda.Event` timestamps to find the next bottleneck. The loop runs until interrupted.
