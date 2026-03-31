"""Benchmark + correctness: modified Triton kernel with pre-computed act_grad vs original."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# --- ORIGINAL kernel (PR #1105) ---

@triton.jit
def _orig_kernel(a_desc, b_desc, c_desc, aux_desc,
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


# --- MODIFIED kernel (pre-computed act_grad) ---

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


def _launch(kernel, a, b, aux, forward):
    M, K = a.shape
    N = b.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if forward:
        aux = torch.empty((M, N), device=a.device, dtype=a.dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    BSM, BSN, BSK = 128, 256, 64
    a_desc = TensorDescriptor.from_tensor(a, [BSM, BSK])
    b_desc = TensorDescriptor.from_tensor(b, [BSN, BSK])
    c_desc = TensorDescriptor.from_tensor(c, [BSM, BSN // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BSM, BSN // 2])
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, BSM) * triton.cdiv(N, BSN)),)
    kernel[grid](a_desc, b_desc, c_desc, aux_desc, M, N, K,
                 BLOCK_SIZE_M=BSM, BLOCK_SIZE_N=BSN, BLOCK_SIZE_K=BSK,
                 GROUP_SIZE_M=1, NUM_SMS=NUM_SMS, FORWARD=forward,
                 num_stages=4 if forward else 3, num_warps=8)
    return (c, aux) if forward else c


def benchmark(fn, warmup=20, iters=100):
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
    torch.manual_seed(42)
    go = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    up_w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    down_w_t = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # --- Correctness: forward ---
    print("=== Forward correctness ===")
    orig_ag, orig_post = _launch(_orig_kernel, go, up_w, None, forward=True)
    new_ag, new_post = _launch(_precomp_kernel, go, up_w, None, forward=True)

    # Original forward: c=pre, aux=post
    # Modified forward: c=act_grad, aux=post
    # Verify post is identical
    post_err = (orig_post - new_post).abs().max().item()
    print(f"Post match: max_err={post_err:.6f} {'PASS' if post_err < 0.01 else 'FAIL'}")

    # Verify act_grad = where(pre > 0, 2*pre, 0.5*pre)
    expected_ag = torch.where(orig_ag > 0, 2.0 * orig_ag, 0.5 * orig_ag)
    ag_err = (expected_ag - new_ag).abs().max().item()
    print(f"Act_grad correct: max_err={ag_err:.6f} {'PASS' if ag_err < 0.01 else 'FAIL'}")

    # --- Correctness: backward ---
    print("\n=== Backward correctness ===")
    pre = orig_ag  # original kernel stores pre
    act_grad = new_ag  # modified kernel stores act_grad

    orig_dpre = _launch(_orig_kernel, go, down_w_t, pre, forward=False)
    new_dpre = _launch(_precomp_kernel, go, down_w_t, act_grad, forward=False)

    bwd_err = (orig_dpre - new_dpre).abs().max().item()
    bwd_mean = (orig_dpre - new_dpre).abs().mean().item()
    print(f"Backward match: max_err={bwd_err:.6f} mean_err={bwd_mean:.6f} {'PASS' if bwd_err < 1.0 else 'FAIL'}")

    # --- Benchmark ---
    print(f"\n=== Benchmark: M={M}, K={K}, N={N} ===")

    # Forward
    t_orig_fwd = benchmark(lambda: _launch(_orig_kernel, go, up_w, None, forward=True))
    t_new_fwd = benchmark(lambda: _launch(_precomp_kernel, go, up_w, None, forward=True))
    print(f"\nForward:")
    print(f"  Original:  {t_orig_fwd:.3f} ms")
    print(f"  Precomp:   {t_new_fwd:.3f} ms")
    print(f"  Delta:     {t_new_fwd - t_orig_fwd:+.3f} ms")

    # Backward
    t_orig_bwd = benchmark(lambda: _launch(_orig_kernel, go, down_w_t, pre, forward=False))
    t_new_bwd = benchmark(lambda: _launch(_precomp_kernel, go, down_w_t, act_grad, forward=False))
    print(f"\nBackward:")
    print(f"  Original (inline act_grad):  {t_orig_bwd:.3f} ms")
    print(f"  Precomp (just multiply):     {t_new_bwd:.3f} ms")
    print(f"  Delta:                       {t_new_bwd - t_orig_bwd:+.3f} ms")

    # Net per step
    fwd_delta = (t_new_fwd - t_orig_fwd)
    bwd_delta = (t_new_bwd - t_orig_bwd)
    total_delta = (fwd_delta + bwd_delta) * 11
    print(f"\nNet per step (x11 layers):")
    print(f"  Forward delta:  {fwd_delta * 11:+.3f} ms")
    print(f"  Backward delta: {bwd_delta * 11:+.3f} ms")
    print(f"  Total:          {total_delta:+.3f} ms")


if __name__ == "__main__":
    main()
