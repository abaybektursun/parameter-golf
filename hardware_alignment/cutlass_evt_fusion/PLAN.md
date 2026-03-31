# CUTLASS EVT Backward MLP Fusion — Implementation Plan

## What We're Doing

Fuse `matmul(go, down_w.T) * activation_grad(pre)` into a single CUTLASS 3.x kernel
using the Epilogue Visitor Tree (EVT). This eliminates one 302MB write + one 302MB read
per layer per backward pass (the intermediate `dpre_matmul` before pointwise multiply).

Activation gradient: `where(pre > 0, 2*pre, 0.5*pre)` (derivative of LeakyReLU(0.5)²).

## Why This Might Work Despite Previous Failures

**What failed (#670):** Wrapped the entire MLP forward+backward in `torch.autograd.Function`.
This made the whole backward opaque to Inductor. The 3 standard cuBLAS matmuls
(`dW2 = go.T @ post`, `dW1 = dpre.T @ x_flat`, `dx = dpre @ up_w`) plus all cross-layer
fusion (RMSNorm backward, residual, RoPE backward) ran in eager mode. Backward was 2.7x
slower, net 46% slower.

**What's different now:** The CUTLASS EVT kernel replaces ONE operation (GEMM + pointwise).
It doesn't touch the autograd boundary. The existing Triton forward-only fusion already
uses `torch.autograd.Function` with explicit cuBLAS matmuls in backward — and that works
(validated -8ms/step). We're replacing the Triton backward kernel call at line 129:

```python
# Current (train_gpt_fused_mlp.py:129):
dpre = _fused_leaky_relu_sq(go, down_w.T.contiguous(), aux=pre)
```

with:

```python
# New:
dpre = cutlass_gemm_act_grad(go, down_w.T.contiguous(), pre)
```

Same call site, same interface. The `torch.autograd.Function` backward already runs in
eager mode (that's how the Triton version works). We're not trying to make it Inductor-
visible — we're trying to make the eager backward FASTER by fusing the GEMM epilogue.

## Critical Constraint: cuBLAS is Already Fast

Previous profiling showed CUTLASS GEMM is 3-10% slower than cuBLAS at d=512 (K=512,
pipeline depth 8, insufficient to hide TMA latency). The EVT fusion must overcome this
penalty with the bandwidth savings.

**The math (per layer, 8-GPU, M=98304):**

- GEMM `(98304, 512) @ (512, 1536)`: ~103 GFLOP, at 48% roofline = ~0.22ms
- Pointwise (read 302MB pre + read 302MB dpre + write 302MB dpre): ~0.18ms at 3.35 TB/s
- Unfused total: ~0.40ms
- CUTLASS EVT (GEMM+epilogue): GEMM ~0.23ms (5% penalty) + epilogue ~free = ~0.23ms
- **Savings per layer: ~0.17ms. x11 layers = ~1.9ms per backward pass.**

At 1ms ~ 0.006 BPB, that's ~0.011 BPB. Modest but real.

**Kill condition:** If CUTLASS GEMM penalty exceeds ~80% of pointwise savings (~0.15ms
per layer), the kernel is net-negative. Profile the standalone CUTLASS GEMM first before
building the EVT.

## Architecture

```
hardware_alignment/cutlass_evt_fusion/
├── PLAN.md                    (this file)
├── csrc/
│   ├── gemm_act_grad.cu       (CUTLASS kernel: EVT tree + launcher)
│   └── torch_binding.cpp      (PyTorch C++ extension registration)
├── setup.py                   (torch.utils.cpp_extension build)
├── test_correctness.py        (numerical parity with Triton version)
├── bench_standalone.py        (isolated GEMM benchmark: CUTLASS vs cuBLAS)
└── bench_e2e.py               (full backward pass benchmark)
```

## Implementation Steps

### Step 0: Kill-Gate — Standalone CUTLASS GEMM Benchmark

Before writing any EVT code, benchmark a vanilla CUTLASS 3.x GEMM (no custom epilogue)
against cuBLAS at our exact dimensions:

- A: (98304, 512) bf16, row-major
- B: (1536, 512) bf16, row-major (transposed: B.T is (512, 1536))
- C: (98304, 1536) bf16, row-major

Use the CUTLASS profiler or a minimal Python binding. If CUTLASS is >15% slower, stop.
The EVT fusion cannot overcome a >15% GEMM penalty.

**How:** Clone CUTLASS 3.x on the H100. Build the profiler. Run:
```
./tools/profiler/cutlass_profiler --operation=gemm \
  --m=98304 --n=1536 --k=512 \
  --A=bf16:row --B=bf16:col --C=bf16:row \
  --kernels=sm90 --warmup=5 --iterations=100
```
Compare against `torch.mm` + `torch.cuda.Event` timing.

Acceptable: within 5% of cuBLAS. Marginal: 5-10%. Kill: >10%.

### Step 1: Custom Activation Gradient Functor

```cpp
template <typename T>
struct LeakyReLUSqGrad {
    CUTLASS_HOST_DEVICE T operator()(T const &dpre_val, T const &pre_val) const {
        T two = T(2.0f);
        T half = T(0.5f);
        T zero = T(0.0f);
        T multiplier = (pre_val > zero) ? two : half;
        return dpre_val * (pre_val * multiplier);
    }
};

template <typename T, int N>
struct LeakyReLUSqGrad<cutlass::Array<T, N>> {
    CUTLASS_HOST_DEVICE cutlass::Array<T, N> operator()(
        cutlass::Array<T, N> const &dpre_vec,
        cutlass::Array<T, N> const &pre_vec) const {
        cutlass::Array<T, N> result;
        LeakyReLUSqGrad<T> scalar_op;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = scalar_op(dpre_vec[i], pre_vec[i]);
        }
        return result;
    }
};
```

The ternary compiles to PTX `sel` (predicated assignment), no warp divergence.
`CUTLASS_PRAGMA_UNROLL` forces unrolling since N is compile-time known.

### Step 2: EVT Tree Definition

```cpp
using namespace cutlass::epilogue::fusion;

using ElementAcc    = float;          // accumulate in FP32
using ElementCompute = float;         // epilogue math in FP32
using ElementOutput = cutlass::bfloat16_t;
using ElementAux    = cutlass::bfloat16_t;  // the saved 'pre' tensor

// Leaf: load 'pre' tensor from HBM via TMA
using AuxLoad = Sm90AuxLoad<
    /* Stages */ 2,
    /* EpiTile */ cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAux,
    /* StrideAux */ cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
    /* SmemLayout */ cutlass::layout::NoPermute,
    /* CopyOp */ cute::SM90_TMA_LOAD,
    /* Alignment */ 8>;  // bf16: 8 elements = 16 bytes (TMA requirement)

// Compute node: applies LeakyReLUSqGrad(acc_fragment, aux_fragment)
using Compute = Sm90Compute<
    LeakyReLUSqGrad,
    ElementOutput,
    ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

// Tree: root = Compute(child0 = AccFetch, child1 = AuxLoad)
using EVT = Sm90EVT<Compute, Sm90AccFetch, AuxLoad>;
```

### Step 3: CollectiveBuilder + Kernel Type

```cpp
using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute,
    ElementOutput, cutlass::layout::RowMajor, /* AlignOut */ 8,
    ElementOutput, cutlass::layout::RowMajor, /* AlignOut */ 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EVT
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementOutput, cutlass::layout::RowMajor, /* AlignA */ 8,
    ElementOutput, cutlass::layout::ColumnMajor, /* AlignB */ 8,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename CollectiveEpilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,  // ProblemShape: M, N, K, L
    CollectiveMainloop,
    CollectiveEpilogue>;

using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

**Tile shape rationale:** 128x256x64 matches the existing Triton kernel's block sizes.
Cluster shape 1x1x1 is simplest; can tune later if profiling shows benefit.

**Layout note:** A (go) is row-major. B (down_w.T) is column-major (transposed row-major
weight). Pre (aux) is row-major, same layout as A's output dimension.

### Step 4: Host Launcher

```cpp
void launch_gemm_act_grad(
    void const* ptr_go,        // (M, K) bf16 row-major
    void const* ptr_down_w_t,  // (K, N) bf16 col-major (= down_w transposed)
    void const* ptr_pre,       // (M, N) bf16 row-major
    void* ptr_dpre,            // (M, N) bf16 row-major — output
    int M, int N, int K,
    cudaStream_t stream)
{
    using StrideA = cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>;
    using StrideB = cutlass::gemm::TagToStrideB_t<cutlass::layout::ColumnMajor>;
    using StrideC = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;

    auto prob_shape = cute::make_shape(M, N, K, 1);

    // EVT arguments — post-order: children before node op
    typename EVT::Arguments evt_args {
        {},                          // Sm90AccFetch: no args
        {                            // Sm90AuxLoad: pointer + stride
            static_cast<ElementAux const*>(ptr_pre),
            ElementAux(0),           // out-of-bounds default
            StrideC{N, cute::Int<1>{}, 0}
        },
        {}                           // Sm90Compute (LeakyReLUSqGrad): no args
    };

    typename GemmOp::Arguments args {
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {   // Mainloop args
            static_cast<ElementOutput const*>(ptr_go),
            StrideA{K, cute::Int<1>{}, 0},
            static_cast<ElementOutput const*>(ptr_down_w_t),
            StrideB{cute::Int<1>{}, N, 0},
        },
        {   // Epilogue args
            evt_args,
            static_cast<ElementOutput*>(ptr_dpre),
            StrideC{N, cute::Int<1>{}, 0},
        }
    };

    GemmOp gemm_op;
    size_t workspace_size = GemmOp::get_workspace_size(args);
    // Allocate workspace if needed (typically 0 for this config)
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    auto status = gemm_op.initialize(args, workspace, stream);
    assert(status == cutlass::Status::kSuccess);

    status = gemm_op.run(stream);
    assert(status == cutlass::Status::kSuccess);

    if (workspace) cudaFree(workspace);
}
```

**Critical detail:** The stride for AuxLoad must exactly match the physical layout of the
`pre` tensor from PyTorch. PyTorch contiguous row-major bf16 tensor of shape (M, N) has
stride (N, 1) in elements. The CuTe stride is {N, Int<1>{}, 0} where the third element
is the batch stride (unused, L=1).

**TMA alignment:** N=1536 elements * 2 bytes = 3072 bytes. 3072 / 16 = 192, cleanly
divisible. Inner dimension alignment is satisfied.

### Step 5: PyTorch C++ Binding

```cpp
// torch_binding.cpp
#include <torch/extension.h>

void launch_gemm_act_grad(
    void const*, void const*, void const*, void*, int, int, int, cudaStream_t);

at::Tensor gemm_act_grad(at::Tensor go, at::Tensor down_w_t, at::Tensor pre) {
    TORCH_CHECK(go.is_cuda() && go.is_contiguous());
    TORCH_CHECK(down_w_t.is_cuda() && down_w_t.is_contiguous());
    TORCH_CHECK(pre.is_cuda() && pre.is_contiguous());
    TORCH_CHECK(go.scalar_type() == at::kBFloat16);
    TORCH_CHECK(pre.stride(-1) == 1, "pre inner dim must be contiguous for TMA");

    int M = go.size(0);
    int K = go.size(1);
    int N = down_w_t.size(1);

    at::Tensor dpre = at::empty({M, N}, go.options());

    launch_gemm_act_grad(
        go.data_ptr(), down_w_t.data_ptr(), pre.data_ptr(), dpre.data_ptr(),
        M, N, K,
        at::cuda::getCurrentCUDAStream());

    return dpre;
}

TORCH_LIBRARY(cutlass_evt, m) {
    m.def("gemm_act_grad(Tensor go, Tensor down_w_t, Tensor pre) -> Tensor");
}

TORCH_LIBRARY_IMPL(cutlass_evt, CUDA, m) {
    m.impl("gemm_act_grad", &gemm_act_grad);
}
```

### Step 6: Build System

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

CUTLASS_PATH = os.environ.get("CUTLASS_PATH", "/opt/cutlass")

setup(
    name="cutlass_evt_fusion",
    ext_modules=[
        CUDAExtension(
            name="cutlass_evt_fusion",
            sources=[
                "csrc/gemm_act_grad.cu",
                "csrc/torch_binding.cpp",
            ],
            include_dirs=[
                f"{CUTLASS_PATH}/include",
                f"{CUTLASS_PATH}/tools/util/include",
            ],
            extra_compile_args={
                "nvcc": [
                    "-std=c++17",
                    "-arch=sm_90a",    # Hopper native (required for WGMMA/TMA)
                    "-O3",
                    "--use_fast_math",
                    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

**CUTLASS installation on the H100:**
```bash
cd /opt && git clone --depth 1 --branch v3.7.0 https://github.com/NVIDIA/cutlass.git
export CUTLASS_PATH=/opt/cutlass
```

Header-only library, no build needed for CUTLASS itself.

### Step 7: Integration into train_gpt_fused_mlp.py

Minimal change. Replace the backward Triton call:

```python
# Before (line 129):
dpre = _fused_leaky_relu_sq(go, down_w.T.contiguous(), aux=pre)

# After:
try:
    import cutlass_evt_fusion
    HAS_CUTLASS_EVT = True
except ImportError:
    HAS_CUTLASS_EVT = False

# In backward:
if HAS_CUTLASS_EVT:
    dpre = torch.ops.cutlass_evt.gemm_act_grad(go, down_w.T.contiguous(), pre)
else:
    dpre = _fused_leaky_relu_sq(go, down_w.T.contiguous(), aux=pre)
```

The rest of the backward (dW2, dW1, dx) stays exactly the same.

### Step 8: Testing

**Correctness (test_correctness.py):**
```python
# Reference: unfused PyTorch
dpre_ref = (go @ down_w.T) * torch.where(pre > 0, 2.0 * pre, 0.5 * pre)

# CUTLASS EVT
dpre_evt = torch.ops.cutlass_evt.gemm_act_grad(go, down_w_t, pre)

# Must pass at bf16 tolerance
assert torch.allclose(dpre_ref, dpre_evt, atol=1e-2, rtol=1e-2)
```

Test at multiple shapes:
- Production: M=98304, K=512, N=1536
- Small: M=128, K=512, N=1536 (single tile)
- Alignment edge: M=98305 (not multiple of 128)

**Performance (bench_standalone.py):**
```python
# Unfused baseline
t_unfused = benchmark(lambda: (go @ down_w_t) * act_grad(pre))

# CUTLASS EVT
t_fused = benchmark(lambda: torch.ops.cutlass_evt.gemm_act_grad(go, down_w_t, pre))

# Must be faster
print(f"Speedup: {t_unfused / t_fused:.2f}x")
```

**End-to-end (bench_e2e.py):**
Full backward pass timing with and without the CUTLASS kernel, 100 iterations.

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CUTLASS GEMM >10% slower than cuBLAS | Medium | Fatal | Step 0 kill-gate |
| TMA alignment fault on pre tensor | Low | Blocks | N=1536 is 16-byte aligned; assert in binding |
| Register spill from EVT aux load | Medium | Perf loss | Profile with Nsight Compute; reduce tile size if needed |
| SMEM overflow (mainloop + aux staging) | Low | Build fail | StageCountAutoCarveout handles this; reduce stages if needed |
| CUTLASS API changes between versions | Low | Delays | Pin to v3.7.0 |
| torch.compile recompilation from new op | Low | OOM | Op is inside eager backward, not traced |

## Execution Order

1. **[1 hour] Step 0:** CUTLASS GEMM benchmark. If >10% slower, stop.
2. **[2-3 hours] Steps 1-4:** Write the kernel (.cu file).
3. **[1 hour] Steps 5-6:** PyTorch binding + build system.
4. **[1 hour] Step 7-8:** Integration + correctness test.
5. **[1 hour] Profiling:** Nsight Compute for register spill, SMEM usage, L2 behavior.

Total: ~6-8 hours of focused work on H100, assuming Step 0 passes.

## Decision Points

- **After Step 0:** If CUTLASS GEMM is >10% slower, abandon. If 5-10%, proceed with caution.
- **After Step 8 correctness:** If numerical parity fails, check stride/layout assumptions.
- **After Step 8 performance:** If <1ms savings across 11 layers, not worth the maintenance burden. Minimum viable win: 1.5ms/backward.
