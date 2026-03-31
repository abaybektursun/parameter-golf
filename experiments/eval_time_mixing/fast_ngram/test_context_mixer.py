"""
Tests for ContextMixer (PPM + linear interpolation).
Verifies: PPM backoff, blending improvement, byte counting, performance.

Run: python test_context_mixer.py
"""
import time
import numpy as np


def make_luts(vocab_size=1024):
    base_bytes = np.ones(vocab_size, dtype=np.int16)
    has_leading_space = np.zeros(vocab_size, dtype=np.uint8)
    has_leading_space[:256] = 1
    is_boundary = np.zeros(vocab_size, dtype=np.uint8)
    is_boundary[0] = 1
    return base_bytes, has_leading_space, is_boundary


def test_ppm_backoff():
    """PPM learns repetitive patterns — early NLL > late NLL."""
    from fused_expert_ext import ContextMixer

    np.random.seed(42)
    n_tokens = 50_000
    vocab_size = 1024
    pattern = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    tokens = np.tile(pattern, n_tokens // len(pattern) + 1)[:n_tokens]

    mixer = ContextMixer(max_order=8, table_bits=20,
                         within_table_bits=18, word_table_bits=18, alpha=0.5)
    mixer.set_tokens(tokens)
    mixer.set_luts(*make_luts(vocab_size))

    positions = np.arange(1, n_tokens, dtype=np.int64)
    neural_nll = np.full(len(positions), 2.0)  # mediocre neural

    mixed_nll, _ = mixer.process_batch(positions, neural_nll)

    early = np.mean(mixed_nll[10:100])
    late = np.mean(mixed_nll[-5000:])
    print(f"  Early avg NLL={early:.4f}, late avg NLL={late:.4f}")
    assert late < early, f"Late ({late:.4f}) should < early ({early:.4f})"
    assert late < 2.0, f"Mixed ({late:.4f}) should beat neural (2.0)"
    print("  PPM backoff PASSED")


def test_blend_bounded():
    """Mixed BPB should be between PPM-only and neural-only (no divergence)."""
    from fused_expert_ext import ContextMixer

    np.random.seed(99)
    n_tokens = 50_000
    vocab_size = 1024
    tokens = np.random.randint(0, vocab_size, size=n_tokens).astype(np.int64)

    mixer = ContextMixer(max_order=4, table_bits=18,
                         within_table_bits=16, word_table_bits=16, alpha=0.02)
    mixer.set_tokens(tokens)
    mixer.set_luts(*make_luts(vocab_size))

    positions = np.arange(1, n_tokens, dtype=np.int64)
    neural_nll = np.random.RandomState(99).uniform(0.5, 5.0, size=len(positions))

    mixed_nll, _ = mixer.process_batch(positions, neural_nll)

    avg_neural = np.mean(neural_nll)
    avg_mixed = np.mean(mixed_nll)
    # With α=0.02, mixed should be very close to neural but slightly different
    print(f"  Neural avg NLL={avg_neural:.4f}, mixed avg NLL={avg_mixed:.4f}")
    print(f"  Ratio mixed/neural: {avg_mixed/avg_neural:.6f}")
    # Mixed NLL should not diverge (should stay within 2x of neural)
    assert avg_mixed < avg_neural * 2.0, \
        f"Mixed ({avg_mixed:.4f}) diverged beyond 2x neural ({avg_neural:.4f})"
    # With small alpha on random data, should be very close to neural
    assert abs(avg_mixed - avg_neural) < avg_neural * 0.1, \
        f"With α=0.02 on random data, mixed should be within 10% of neural"
    print("  Blend bounded PASSED")


def test_byte_counting():
    """Byte counting matches expected values."""
    from fused_expert_ext import ContextMixer

    vocab_size = 1024
    mixer = ContextMixer(max_order=4, table_bits=16,
                         within_table_bits=16, word_table_bits=16)

    base_bytes = np.array([3, 1, 2, 1, 5] + [1] * (vocab_size - 5), dtype=np.int16)
    has_leading_space = np.array([0, 1, 0, 1, 0] + [0] * (vocab_size - 5), dtype=np.uint8)
    is_boundary = np.array([1, 0, 0, 0, 0] + [0] * (vocab_size - 5), dtype=np.uint8)
    mixer.set_luts(base_bytes, has_leading_space, is_boundary)

    targets = np.array([1, 1, 2], dtype=np.int64)
    prev = np.array([0, 2, 3], dtype=np.int64)
    total = mixer.compute_bytes(targets, prev)
    expected = 1.0 + 2.0 + 2.0
    assert abs(total - expected) < 1e-10, f"Expected {expected}, got {total}"
    print(f"  Byte counting: {total} (expected {expected}) PASSED")


def test_performance():
    """Benchmark throughput for 1M tokens."""
    from fused_expert_ext import ContextMixer

    n_tokens = 1_000_000
    vocab_size = 1024
    tokens = np.random.RandomState(42).randint(
        0, vocab_size, size=n_tokens).astype(np.int64)

    mixer = ContextMixer(max_order=8, table_bits=20,
                         within_table_bits=18, word_table_bits=18, alpha=0.02)
    mixer.set_tokens(tokens)
    mixer.set_luts(*make_luts(vocab_size))

    positions = np.arange(1, n_tokens, dtype=np.int64)
    neural_nll = np.random.RandomState(42).uniform(0.5, 5.0, size=len(positions))

    # Warmup
    mixer.reset()
    mixer.process_batch(positions[:1000], neural_nll[:1000])
    mixer.reset()

    # Timed run
    batch_size = 2048
    t0 = time.perf_counter()
    offset = 0
    n_batches = 0
    while offset < len(positions):
        end = min(offset + batch_size, len(positions))
        mixer.process_batch(positions[offset:end], neural_nll[offset:end])
        offset = end
        n_batches += 1
    elapsed = time.perf_counter() - t0

    tps = n_tokens / elapsed
    projected_62m = elapsed * 62
    per_batch_ms = elapsed / n_batches * 1000

    print(f"\n  Performance ({n_tokens:,} tokens, order=8, α=0.02):")
    print(f"    Elapsed:              {elapsed:.3f}s ({tps:,.0f} tok/s)")
    print(f"    Projected 62M tokens: {projected_62m:.1f}s")
    print(f"    Per-batch (2048 tok): {per_batch_ms:.2f}ms")
    print(f"    Budget (12ms GPU):    {per_batch_ms / 12 * 100:.1f}% utilized")

    return {
        "elapsed_s": elapsed,
        "tokens_per_sec": tps,
        "projected_62m_s": projected_62m,
        "per_batch_ms": per_batch_ms,
    }


if __name__ == "__main__":
    for name, fn in [
        ("test_ppm_backoff", test_ppm_backoff),
        ("test_blend_bounded", test_blend_bounded),
        ("test_byte_counting", test_byte_counting),
        ("test_performance", test_performance),
    ]:
        print(f"\n{name}:")
        fn()

    print("\n=== ALL TESTS PASSED ===")
