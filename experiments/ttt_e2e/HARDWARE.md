# Hardware Alignment & Implementation Efficiency

## 1. Tensor Core Alignment

Prime MLP dimensions: [512, r] and [r, 512].

H100 tensor cores (bf16): optimal with M, N, K that are multiples of 64 or 128.

| r | 512×r alignment | cuBLAS tile fit | Verdict |
|---|---|---|---|
| 128 | 128 = 2^7 | Perfect (128-wide tiles) | Best |
| 192 | 192 = 64×3 | Good (64-wide tiles) | OK |
| 256 | 256 = 2^8 | Perfect (128/256-wide tiles) | Best |
| 320 | 320 = 64×5 | Acceptable | OK |

Recommendation: r=128 or r=256 for cleanest tensor core utilization.

## 2. Arithmetic Intensity Analysis

For up_proj GEMM: [M, 512] × [512, r] → [M, r] where M = batch*seq

Arithmetic intensity = (2 × M × 512 × r) / (2 × (M×512 + 512×r + M×r))

With M=1024, r=256:
AI = 2 × 1024 × 512 × 256 / (2 × (524K + 131K + 262K))
   = 268M / 1.83M ≈ 146 FLOPs/byte

H100 break-even: ~990 TFLOPS / 3.35 TB/s ≈ 295 FLOPs/byte

Prime MLP is **memory-bandwidth bound** (AI 146 < 295). The GEMMs are too
small to saturate compute. This means:
- Kernel fusion helps (reduces HBM round-trips)
- But the prime MLP is NOT the bottleneck — it's tiny compared to main MLP
- Don't over-optimize the prime MLP kernels

For comparison, main MLP up_proj (512×1792, M=1024): AI ≈ 286 (nearly compute-bound).

## 3. The Critical Eval Backward Optimization

The current fused backward (FusedLeakyReLUSqMLP) ALWAYS computes weight gradients:

```python
def backward(ctx, grad_output):
    dW2 = go.T @ post      # grad for down_w — EXPENSIVE, [512,M]×[M,1792]
    dpre = fused(go, down_w.T, pre)  # fused grad input — needed
    dW1 = dpre.T @ x_flat  # grad for up_w — EXPENSIVE, [1792,M]×[M,512]
    dx = dpre @ up_w       # grad for input — needed
    return dx, dW1, dW2
```

During eval TTT, we only need `dx` (to propagate to prime MLP). `dW1` and `dW2`
are WASTED — they compute gradients for frozen main MLP weights.

These two wasted GEMMs are the LARGEST operations in the backward (1792-dim hidden).
Skipping them saves ~50% of backward cost per suffix layer.

### Solution: `needs_input_grad` guard

```python
def backward(ctx, grad_output):
    x_flat, up_w, down_w, pre, post = ctx.saved_tensors
    go = grad_output.view(-1, grad_output.shape[-1])
    dpre = _fused_leaky_relu_sq(go, down_w.T.contiguous(), aux=pre)
    dx = (dpre @ up_w).view(grad_output.shape) if ctx.needs_input_grad[0] else None
    dW1 = dpre.T @ x_flat if ctx.needs_input_grad[1] else None
    dW2 = go.T @ post if ctx.needs_input_grad[2] else None
    return dx, dW1, dW2
```

Combined with `requires_grad=False` on all bank weights during eval, this
automatically skips the expensive weight GEMMs. The fused forward (Triton TMA)
and fused dpre (CUTLASS EVT) still run — only the unnecessary work is skipped.

### Alternative: non-fused path during eval

```python
# Standard autograd automatically skips weight grads when requires_grad=False
x = F.leaky_relu(F.linear(x, up_w), negative_slope=0.5)  # up_w.requires_grad=False
return F.linear(x.square(), down_w)  # down_w.requires_grad=False
```

Simpler but loses the fused forward. Acceptable for initial implementation.

## 4. torch.compile Compatibility

### Current pattern (works well)

Bank weights are passed as ARGUMENTS to Block.forward, not stored as attributes:
```python
def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w, ...):
```

torch.compile captures the computation graph with tensor argument shapes.
Changing tensor VALUES (but not shapes) between calls requires no recompilation.

### Prime MLP weights: two options

**Option A: Module parameters (simpler)**
```python
class Block(nn.Module):
    def __init__(self, ..., prime_dim=0):
        if prime_dim > 0:
            self.prime_up = nn.Parameter(torch.empty(prime_dim, dim))
            self.prime_down = nn.Parameter(torch.zeros(dim, prime_dim))
```
In-place SGD updates (`p.data -= lr * g`) don't change tensor identity, so
torch.compile's captured references remain valid. No recompilation.

**Option B: Passed as arguments (matches bank pattern)**
```python
def forward(self, x, x0, ..., up_w, down_w, prime_up_w=None, prime_down_w=None):
```
More explicit, easier to reason about in FOMAML (detach/clone for inner loop).

Recommendation: Option B for FOMAML training (need detach/clone flexibility),
Option A for eval TTT (simpler in-place updates). Can convert between them.

### The inference_mode problem

Current eval uses `torch.inference_mode()` — disables autograd entirely.
For TTT eval, we CANNOT use inference_mode. Instead:

```python
# Set requires_grad=False on everything except prime params
for p in model.parameters():
    p.requires_grad_(False)
for block in suffix_blocks:
    block.prime_up.requires_grad_(True)
    block.prime_down.requires_grad_(True)

# Don't use inference_mode — use autocast only
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = forward(x_batch)
```

torch.compile MAY need separate compilation for the TTT path (with grad) vs
the non-TTT path (with inference_mode). Two compiled versions is fine — they
share the same underlying kernels.

## 5. Eval-Time Data Flow

### Current eval flow (embarrassingly parallel)

```
windows distributed across 8 GPUs, any order
each GPU: batch of 32 windows → forward → score → accumulate
all_reduce(loss_sum, token_count, byte_count)
```

### TTT eval flow (sequential mini-batches)

```
for each mini-batch of b=1024 tokens (sequential):
    create 16 sliding windows (b/stride = 1024/64)
    distribute 16 windows across 8 GPUs (2 per GPU)
    each GPU: forward 2 windows → score → loss → backward (prime only)
    all_reduce(prime_gradients)  [786K params × 4 bytes = 3.14 MB, NVLink: <0.01ms]
    SGD update prime params (all GPUs identical)
```

### Latency estimate

Per TTT update (one mini-batch of b=1024 tokens):
- Forward per GPU: 2 windows × 2048 tokens = 4096 tokens ≈ 0.5-1.5ms
- Backward per GPU (3 suffix layers, prime grads only): ~0.3-0.8ms
- All-reduce: ~0.01ms
- Total per update: ~1-2.5ms

Total TTT eval: 58K updates × 2ms = 116s ≈ 2 minutes compute
Plus scoring/accumulation overhead: ~1-2 minutes
Plus data loading: ~0.5 minutes
Estimated total: 3.5-5 minutes (fits 10 min budget)

### The low-utilization concern

2 windows per GPU per TTT update = 4096 tokens per GPU. This is low for an
H100 (underutilized SMs and memory bandwidth).

Mitigations (if needed):
1. Increase b (fewer updates, more tokens per batch, but coarser adaptation)
2. Process multiple consecutive mini-batches' FORWARD passes together, then
   backward separately (requires keeping computation graphs)
3. Accept the low utilization — the total time still fits budget

Start with the simple approach. Optimize only if profiling shows we're near
the 10-minute limit.

## 6. Multi-GPU Synchronization

### Training (FOMAML)

Standard data parallelism. Each GPU processes different sequences.
Inner loop: prime gradients are LOCAL per GPU (different data).
Outer loop: all-reduce outer gradients (standard DDP behavior).

No special handling needed — existing DDP infrastructure works.

### Eval (TTT)

All GPUs must have IDENTICAL prime weights at all times. Achieved by:
1. All GPUs process the same mini-batch (different windows from it)
2. All-reduce prime gradients before update
3. All GPUs apply identical SGD step

Prime gradient all-reduce: 786K params × 4 bytes = 3.14 MB per sync.
H100 SXM NVLink: 900 GB/s → 3.14 MB / 900 GB/s = 3.5 microseconds.
Over 58K syncs: 0.2 seconds total. Negligible.

## 7. Memory Budget

### Training (FOMAML from scratch)

Additional memory vs standard training:
- Prime MLP weights: 786K × 4 bytes (fp32 for optimizer) = 3.14 MB
- Prime optimizer state: 786K × 4 bytes (SGD has no state) = 0 MB
  (Or 786K × 8 bytes if using Adam for inner loop = 6.28 MB)
- Inner loop: detached prime weight copies = 786K × 2 bytes (bf16) = 1.57 MB
- Cached prefix output: batch × seq × 512 × 2 bytes
  With batch=8, seq=2048: 8 × 2048 × 512 × 2 = 16 MB

Total additional: ~20 MB. Negligible on 80 GB H100.

### Eval (TTT)

Additional memory vs standard eval:
- Computation graph for backward through 3 suffix layers + one window:
  ~3 × (window_tokens × 512 × few_tensors × 2 bytes) ≈ 30 MB per window
- Accumulated prime gradients: 786K × 4 bytes = 3.14 MB

Total additional: ~35 MB. Negligible.

## 8. Artifact Size

### Prime MLP weight storage

| Precision | Size (r=256, 3 layers) | Notes |
|---|---|---|
| bf16 | 1.57 MB | Direct, no quality loss |
| int8 | 0.79 MB | Dequant at eval start, inner loop adapts away from quant error |
| bf16 + Brotli-11 | ~1.0-1.3 MB | Depends on weight entropy |
| int8 + Brotli-11 | ~0.5-0.7 MB | Best compression |

Prime MLP RMSNorm: parameter-free (our RMSNorm has no learned scale).
Prime scale parameter: 512 × 4 bytes × 3 layers = 6 KB. Negligible.

Recommendation: bf16 + Brotli-11 for initial experiments. Switch to int8 if
budget is tight. The meta-learned initialization is a STARTING POINT — int8
quantization error gets corrected by the first few TTT steps.

## 9. Implementation Phases

### Phase 0: Minimal viable TTT (no retraining, ~1 day)

Modify eval_val_sliding to add score-first TTT:
1. Add PrimeMLP module to Block (zero-init down_proj)
2. Remove inference_mode, set requires_grad=False on non-prime params
3. After each b tokens scored: backward → SGD update prime params
4. Use non-fused MLP path for suffix layers (simple, correct)

Goal: validate machinery, measure overhead, check if naive adapter TTT has signal.

### Phase 1: FOMAML training (~3 days)

Modify training loop:
1. Add prime MLPs from step 1
2. Inner loop: forward suffix → loss → grad → update prime (detached)
3. Outer loop: forward with updated primes → loss → backward all
4. Cache prefix outputs between inner and outer forward

Goal: meta-learned prime MLPs, measure BPB improvement.

### Phase 2: Optimize (~2 days)

1. Add needs_input_grad to FusedLeakyReLUSqMLP backward
2. Compile separate TTT eval path (with grad support)
3. Profile and optimize the eval TTT loop
4. Tune b, inner_lr, weight_decay, r

Goal: fit eval within 10-minute budget with optimal BPB.
