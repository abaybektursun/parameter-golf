"""Benchmark: Does torch.compile fuse mm * tensor into a single kernel?
Then: triton_op + register_autograd full MLP backward vs current best."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# ============================================================
# Part 1: Can Inductor fuse mm+multiply?
# ============================================================

def unfused_dpre(go, down_w, act_grad):
    return torch.mm(go, down_w) * act_grad

compiled_dpre = torch.compile(unfused_dpre, mode="max-autotune")


# ============================================================
# Part 2: Current best Triton kernel (precomp act_grad)
# ============================================================

@triton.jit
def _precomp_kernel(a_desc, b_desc, c_desc, aux_desc,
                    M, N, K,
                    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                    NUM_SMS: tl.constexpr, FORWARD: tl.constexpr):
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
            c0_ag = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = c0 * c0_ag
            c_desc.store([offs_am_c, offs_bn_c], c0)
        if FORWARD:
            c0_ag = tl.where(c0 > 0, 2.0 * c0, 0.5 * c0)
            c_desc.store([offs_am_c, offs_bn_c], c0_ag)
            c0_post = 0.5 * c0_ag * c0
            aux_desc.store([offs_am_c, offs_bn_c], c0_post)
        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_ag = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = c1 * c1_ag
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        if FORWARD:
            c1_ag = tl.where(c1 > 0, 2.0 * c1, 0.5 * c1)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_ag)
            c1_post = 0.5 * c1_ag * c1
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)


def triton_fused_bwd(go, down_w_t, act_grad):
    M, K = go.shape
    N = down_w_t.shape[0]
    c = torch.empty((M, N), device=go.device, dtype=go.dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    BSM, BSN, BSK = 128, 256, 64
    a_desc = TensorDescriptor.from_tensor(go, [BSM, BSK])
    b_desc = TensorDescriptor.from_tensor(down_w_t, [BSN, BSK])
    c_desc = TensorDescriptor.from_tensor(c, [BSM, BSN // 2])
    aux_desc = TensorDescriptor.from_tensor(act_grad, [BSM, BSN // 2])
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, BSM) * triton.cdiv(N, BSN)),)
    _precomp_kernel[grid](a_desc, b_desc, c_desc, aux_desc, M, N, K,
                          BLOCK_SIZE_M=BSM, BLOCK_SIZE_N=BSN, BLOCK_SIZE_K=BSK,
                          GROUP_SIZE_M=1, NUM_SMS=NUM_SMS, FORWARD=False,
                          num_stages=3, num_warps=8)
    return c


# ============================================================
# Part 3: Full backward comparison
# ============================================================

def full_backward_triton(go, up_w, down_w, down_w_t, act_grad, post, x_flat):
    """Current best: Triton fused backward."""
    dW2 = go.T @ post
    dpre = triton_fused_bwd(go, down_w_t, act_grad)
    dW1 = dpre.T @ x_flat
    dx = dpre @ up_w
    return dx, dW1, dW2


def full_backward_inductor(go, up_w, down_w, act_grad, post, x_flat):
    """Inductor-visible backward: standard ops that compile can fuse."""
    dW2 = go.T @ post
    dpre = torch.mm(go, down_w) * act_grad
    dW1 = dpre.T @ x_flat
    dx = dpre @ up_w
    return dx, dW1, dW2

compiled_full_backward = torch.compile(full_backward_inductor, mode="max-autotune")


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    M, K, N = 98304, 512, 1536
    print(f"Shape: M={M}, K={K}, N={N}\n")

    go = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    down_w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    down_w_t = down_w.T.contiguous()
    up_w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    x_flat = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    act_grad = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    post = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # ---- Part 1: dpre only ----
    print("=== Part 1: dpre = mm(go, down_w) * act_grad ===")

    # Correctness
    ref = torch.mm(go, down_w) * act_grad
    tri = triton_fused_bwd(go, down_w_t, act_grad)
    comp = compiled_dpre(go, down_w, act_grad)
    print(f"Triton vs ref: max_err={( ref - tri).abs().max().item():.4f}")
    print(f"Compiled vs ref: max_err={(ref - comp).abs().max().item():.4f}")

    t_cublas_unfused = bench(lambda: torch.mm(go, down_w) * act_grad)
    t_triton = bench(lambda: triton_fused_bwd(go, down_w_t, act_grad))
    t_compiled = bench(lambda: compiled_dpre(go, down_w, act_grad))

    print(f"\ncuBLAS + mul (unfused):     {t_cublas_unfused:.3f} ms")
    print(f"Triton TMA fused (precomp): {t_triton:.3f} ms")
    print(f"torch.compile max-autotune: {t_compiled:.3f} ms")

    # ---- Part 2: full backward ----
    print(f"\n=== Part 2: Full MLP backward (all 4 GEMMs) ===")

    # Warmup compiled version (triggers compilation)
    print("Compiling full backward (this may take a minute)...")
    for _ in range(3):
        compiled_full_backward(go, up_w, down_w, act_grad, post, x_flat)
    torch.cuda.synchronize()
    print("Compilation done.")

    t_triton_full = bench(lambda: full_backward_triton(go, up_w, down_w, down_w_t, act_grad, post, x_flat))
    t_compiled_full = bench(lambda: compiled_full_backward(go, up_w, down_w, act_grad, post, x_flat))

    print(f"\nTriton fused backward:      {t_triton_full:.3f} ms")
    print(f"Compiled Inductor backward: {t_compiled_full:.3f} ms")
    delta = t_triton_full - t_compiled_full
    winner = "Compiled" if delta > 0 else "Triton"
    print(f"Winner: {winner} ({abs(delta):.3f} ms faster)")
    print(f"Per step (x11): {delta * 11:+.3f} ms")


if __name__ == "__main__":
    main()
