"""Numerical parity test: CUTLASS EVT fused GEMM*mul vs unfused PyTorch."""

import torch
import cutlass_evt_fusion


def reference_gemm_mul(go, down_w, act_grad):
    """Unfused reference: (go @ down_w.T) * act_grad.
    go:       (M, K)
    down_w:   (K, N) — passed directly, NOT transposed
    act_grad: (M, N)
    """
    acc = go @ down_w  # (M,K) @ (K,N) -> (M,N)
    return acc * act_grad


def test_shape(M, K, N, label=""):
    torch.manual_seed(42)
    go = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    down_w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    act_grad = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    dpre_ref = reference_gemm_mul(go, down_w, act_grad)
    dpre_evt = torch.ops.cutlass_evt.gemm_mul(go, down_w, act_grad)

    max_err = (dpre_ref - dpre_evt).abs().max().item()
    mean_err = (dpre_ref - dpre_evt).abs().mean().item()
    passed = torch.allclose(dpre_ref, dpre_evt, atol=1e-2, rtol=1e-2)

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label} M={M} K={K} N={N}  max_err={max_err:.6f}  mean_err={mean_err:.6f}")
    return passed


if __name__ == "__main__":
    all_passed = True
    all_passed &= test_shape(98304, 512, 1536, "production")
    all_passed &= test_shape(128, 512, 1536, "single-tile")
    all_passed &= test_shape(98305, 512, 1536, "non-aligned-M")
    all_passed &= test_shape(256, 512, 1536, "small")
    all_passed &= test_shape(1024, 512, 1536, "medium")

    if all_passed:
        print("\nAll tests passed.")
    else:
        print("\nSome tests FAILED.")
        exit(1)
