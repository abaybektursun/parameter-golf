"""End-to-end backward pass benchmark with and without CUTLASS EVT kernel."""

import torch


def backward_unfused(go, up_w, down_w, act_grad, post, x_flat):
    """Standard unfused backward pass."""
    dW2 = go.T @ post
    dpre_matmul = go @ down_w  # (M,K) @ (K,N) -> (M,N)
    dpre = dpre_matmul * act_grad
    dW1 = dpre.T @ x_flat
    dx = dpre @ up_w
    return dx, dW1, dW2


def backward_fused(go, up_w, down_w, act_grad, post, x_flat):
    """Backward pass with CUTLASS EVT fused kernel."""
    import cutlass_evt_fusion
    dW2 = go.T @ post
    dpre = torch.ops.cutlass_evt.gemm_mul(go, down_w, act_grad)
    dW1 = dpre.T @ x_flat
    dx = dpre @ up_w
    return dx, dW1, dW2


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
    M, D, D_MLP = 98304, 512, 1536
    print(f"Shape: M={M}, D={D}, D_MLP={D_MLP}")

    x_flat = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    up_w = torch.randn(D_MLP, D, device="cuda", dtype=torch.bfloat16)
    down_w = torch.randn(D, D_MLP, device="cuda", dtype=torch.bfloat16)

    # Forward to get act_grad, post
    pre = x_flat @ up_w.T
    act_grad = torch.where(pre > 0, 2.0 * pre, 0.5 * pre)
    post = torch.where(pre > 0, pre, 0.5 * pre) ** 2
    go = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)

    t_unfused = benchmark(lambda: backward_unfused(go, up_w, down_w, act_grad, post, x_flat))
    t_fused = benchmark(lambda: backward_fused(go, up_w, down_w, act_grad, post, x_flat))

    print(f"\nUnfused backward: {t_unfused:.3f} ms")
    print(f"Fused backward:   {t_fused:.3f} ms")
    print(f"Speedup:          {t_unfused / t_fused:.2f}x")
    print(f"Savings:          {t_unfused - t_fused:.3f} ms")
    print(f"Savings x11 layers: {(t_unfused - t_fused) * 11:.3f} ms")


if __name__ == "__main__":
    main()
