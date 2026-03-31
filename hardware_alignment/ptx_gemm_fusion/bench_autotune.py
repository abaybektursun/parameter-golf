"""Autotune: benchmark all CUTLASS kernel configs vs cuBLAS baseline."""

import torch
import ptx_gemm_fusion


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
    print(f"Autotune: M={M}, K={K}, N={N}\n")

    go = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    dw = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    ag = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # Reference
    ref = (go @ dw) * ag

    configs = [
        ("v0: Cooperative 128x256",         torch.ops.ptx_gemm.gemm_mul_v0),
        ("v1: Pingpong 128x256",            torch.ops.ptx_gemm.gemm_mul_v1),
        ("v2: Pingpong 128x128",            torch.ops.ptx_gemm.gemm_mul_v2),
        ("v3: Pingpong 256x128",            torch.ops.ptx_gemm.gemm_mul_v3),
        ("v4: Pingpong 128x256 cluster2x1", torch.ops.ptx_gemm.gemm_mul_v4),
        ("v5: CoopPingpong 128x256",        torch.ops.ptx_gemm.gemm_mul_v5),
    ]

    # cuBLAS baselines
    t_cublas = bench(lambda: go @ dw)
    t_unfused = bench(lambda: (go @ dw) * ag)
    print(f"{'cuBLAS GEMM only':<40} {t_cublas:.3f} ms")
    print(f"{'cuBLAS GEMM + separate mul':<40} {t_unfused:.3f} ms")
    print()

    best_name, best_time = None, 999.0
    for name, fn in configs:
        # Correctness check
        out = fn(go, dw, ag)
        err = (ref - out).abs().max().item()
        correct = err < 5.0  # bf16 tolerance

        if correct:
            t = bench(lambda: fn(go, dw, ag))
            overhead = t - t_cublas
            print(f"{name:<40} {t:.3f} ms  (GEMM overhead: {overhead:+.3f} ms)  err={err:.1f}  PASS")
            if t < best_time:
                best_time = t
                best_name = name
        else:
            print(f"{name:<40} FAIL  err={err:.1f}")

    print(f"\n{'='*60}")
    print(f"Best:     {best_name} at {best_time:.3f} ms")
    print(f"cuBLAS:   {t_cublas:.3f} ms")
    print(f"Gap:      {best_time - t_cublas:.3f} ms ({(best_time/t_cublas - 1)*100:.0f}% overhead)")
    print(f"vs unfused: {t_unfused - best_time:+.3f} ms/layer = {(t_unfused - best_time)*11:+.3f} ms/step")


if __name__ == "__main__":
    main()
