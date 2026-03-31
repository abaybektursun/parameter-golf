"""Benchmark: CUTLASS EVT fused vs PR #1105 Triton TMA fused vs unfused cuBLAS."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import cutlass_evt_fusion

# --- PR #1105 Triton TMA kernel (backward mode) ---

@triton.jit
def _fused_leaky_relu_sq_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 M, N, K,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr,
                                 FORWARD: tl.constexpr):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c0 = acc0.to(dtype)
        if not FORWARD:
            c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = c0 * tl.where(c0_pre > 0, 2.0 * c0_pre, 0.5 * c0_pre)
        c_desc.store([offs_am_c, offs_bn_c], c0)
        if FORWARD:
            c0_post = tl.where(c0 > 0, c0, 0.5 * c0)
            c0_post = c0_post * c0_post
            aux_desc.store([offs_am_c, offs_bn_c], c0_post)
        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = c1 * tl.where(c1_pre > 0, 2.0 * c1_pre, 0.5 * c1_pre)
        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        if FORWARD:
            c1_post = tl.where(c1 > 0, c1, 0.5 * c1)
            c1_post = c1_post * c1_post
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)


def triton_fused_backward(go, down_w_t, pre):
    """PR #1105 Triton TMA kernel in backward mode."""
    M, K = go.shape
    N, K2 = down_w_t.shape
    c = torch.empty((M, N), device=go.device, dtype=go.dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 64
    a_desc = TensorDescriptor.from_tensor(go, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(down_w_t, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(pre, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    def grid(META):
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),)
    _fused_leaky_relu_sq_kernel[grid](
        a_desc, b_desc, c_desc, aux_desc, M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1, NUM_SMS=NUM_SMS, FORWARD=False,
        num_stages=3, num_warps=8)
    return c


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
    return start.elapsed_time(end) / iters


def main():
    M, K, N = 98304, 512, 1536
    print(f"Shape: M={M}, K={K}, N={N}")
    print(f"Operation: dpre = matmul(go, down_w) * activation_grad(pre)\n")

    go = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    down_w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    down_w_t = down_w.T.contiguous()  # (N, K) for Triton
    pre = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    act_grad = torch.where(pre > 0, 2.0 * pre, 0.5 * pre)  # pre-computed for CUTLASS

    # 1. Unfused cuBLAS baseline
    def unfused():
        acc = go @ down_w
        return acc * torch.where(pre > 0, 2.0 * pre, 0.5 * pre)

    # 2. Unfused cuBLAS with pre-computed act_grad
    def unfused_precomputed():
        acc = go @ down_w
        return acc * act_grad

    # 3. PR #1105 Triton TMA fused (computes act_grad inline)
    def triton_fused():
        return triton_fused_backward(go, down_w_t, pre)

    # 4. CUTLASS EVT fused (uses pre-computed act_grad)
    def cutlass_fused():
        return torch.ops.cutlass_evt.gemm_mul(go, down_w, act_grad)

    # 5. cuBLAS GEMM only (lower bound)
    def cublas_gemm_only():
        return go @ down_w

    t_unfused = benchmark(unfused)
    t_unfused_pre = benchmark(unfused_precomputed)
    t_triton = benchmark(triton_fused)
    t_cutlass = benchmark(cutlass_fused)
    t_gemm = benchmark(cublas_gemm_only)

    print(f"cuBLAS GEMM only:           {t_gemm:.3f} ms  (lower bound)")
    print(f"Unfused (GEMM+actgrad+mul): {t_unfused:.3f} ms")
    print(f"Unfused (GEMM+mul precomp): {t_unfused_pre:.3f} ms")
    print(f"PR#1105 Triton TMA fused:   {t_triton:.3f} ms")
    print(f"CUTLASS EVT fused:          {t_cutlass:.3f} ms")
    print()
    print(f"--- Speedup vs unfused (GEMM+actgrad+mul) ---")
    print(f"Triton TMA:  {t_unfused / t_triton:.2f}x  (saves {t_unfused - t_triton:.3f} ms)")
    print(f"CUTLASS EVT: {t_unfused / t_cutlass:.2f}x  (saves {t_unfused - t_cutlass:.3f} ms)")
    print()
    print(f"--- Head to head ---")
    winner = "CUTLASS" if t_cutlass < t_triton else "Triton"
    print(f"Triton:  {t_triton:.3f} ms")
    print(f"CUTLASS: {t_cutlass:.3f} ms")
    print(f"Winner:  {winner} ({abs(t_triton - t_cutlass):.3f} ms faster)")
    print()
    print(f"--- Per-step savings (x11 layers) ---")
    print(f"Triton TMA:  {(t_unfused - t_triton) * 11:.3f} ms/step")
    print(f"CUTLASS EVT: {(t_unfused - t_cutlass) * 11:.3f} ms/step")


if __name__ == "__main__":
    main()
