"""Standalone benchmark: CUTLASS EVT fused vs unfused cuBLAS + pointwise."""

import torch
import cutlass_evt_fusion


def benchmark(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    M, K, N = 98304, 512, 1536
    print(f"Shape: M={M}, K={K}, N={N}")
    print(f"go: ({M}, {K}) bf16  down_w: ({K}, {N}) bf16  act_grad: ({M}, {N}) bf16\n")

    go = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    down_w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    act_grad = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # Unfused baseline: cuBLAS GEMM + pointwise multiply
    def unfused():
        acc = go @ down_w
        return acc * act_grad

    # CUTLASS EVT fused
    def fused():
        return torch.ops.cutlass_evt.gemm_mul(go, down_w, act_grad)

    # cuBLAS GEMM only (to measure GEMM penalty)
    def cublas_gemm_only():
        return go @ down_w

    t_unfused = benchmark(unfused)
    t_fused = benchmark(fused)
    t_gemm = benchmark(cublas_gemm_only)

    print(f"cuBLAS GEMM only:  {t_gemm:.3f} ms")
    print(f"Unfused (GEMM+mul): {t_unfused:.3f} ms")
    print(f"CUTLASS EVT fused: {t_fused:.3f} ms")
    print(f"Speedup:           {t_unfused / t_fused:.2f}x")
    print(f"Savings per call:  {t_unfused - t_fused:.3f} ms")
    print(f"Savings x11 layers: {(t_unfused - t_fused) * 11:.3f} ms")


if __name__ == "__main__":
    main()
