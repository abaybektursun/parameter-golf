"""Final benchmark: CUTLASS EVT backward vs Triton precomp backward vs cuBLAS unfused.
Full MLP backward with all 4 GEMMs."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import cutlass_evt_fusion


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
        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_ag = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = c1 * c1_ag
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)


def triton_bwd(go, down_w_t, act_grad):
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
    M, D, D_MLP = 98304, 512, 1536
    print(f"Full MLP backward: M={M}, D={D}, D_MLP={D_MLP}\n")

    x_flat = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    up_w = torch.randn(D_MLP, D, device="cuda", dtype=torch.bfloat16)
    down_w = torch.randn(D, D_MLP, device="cuda", dtype=torch.bfloat16)
    down_w_t = down_w.T.contiguous()
    go = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    act_grad = torch.randn(M, D_MLP, device="cuda", dtype=torch.bfloat16)
    post = torch.randn(M, D_MLP, device="cuda", dtype=torch.bfloat16)

    # 1. cuBLAS unfused (baseline)
    def bwd_unfused():
        dW2 = go.T @ post
        dpre = (go @ down_w) * act_grad
        dW1 = dpre.T @ x_flat
        dx = dpre @ up_w
        return dx, dW1, dW2

    # 2. Triton precomp backward
    def bwd_triton():
        dW2 = go.T @ post
        dpre = triton_bwd(go, down_w_t, act_grad)
        dW1 = dpre.T @ x_flat
        dx = dpre @ up_w
        return dx, dW1, dW2

    # 3. CUTLASS EVT backward
    def bwd_cutlass():
        dW2 = go.T @ post
        dpre = torch.ops.cutlass_evt.gemm_mul(go, down_w, act_grad)
        dW1 = dpre.T @ x_flat
        dx = dpre @ up_w
        return dx, dW1, dW2

    # Correctness
    ref = bwd_unfused()
    tri = bwd_triton()
    cut = bwd_cutlass()
    for i, name in enumerate(["dx", "dW1", "dW2"]):
        te = (ref[i] - tri[i]).abs().max().item()
        ce = (ref[i] - cut[i]).abs().max().item()
        print(f"{name}: Triton err={te:.2f}  CUTLASS err={ce:.2f}")

    # Benchmark
    t_unfused = bench(bwd_unfused)
    t_triton = bench(bwd_triton)
    t_cutlass = bench(bwd_cutlass)

    print(f"\n{'Method':<30} {'Time':>8}  {'vs unfused':>10}  {'per step (x11)':>14}")
    print(f"{'='*70}")
    print(f"{'cuBLAS unfused':<30} {t_unfused:>7.3f}ms  {'baseline':>10}  {'':>14}")
    print(f"{'Triton precomp':<30} {t_triton:>7.3f}ms  {t_unfused-t_triton:>+9.3f}ms  {(t_unfused-t_triton)*11:>+13.3f}ms")
    print(f"{'CUTLASS EVT':<30} {t_cutlass:>7.3f}ms  {t_unfused-t_cutlass:>+9.3f}ms  {(t_unfused-t_cutlass)*11:>+13.3f}ms")
    print(f"\n{'CUTLASS vs Triton':<30} {t_triton-t_cutlass:>+7.3f}ms/layer  {(t_triton-t_cutlass)*11:>+13.3f}ms/step")


if __name__ == "__main__":
    main()
