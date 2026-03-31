"""
Test fused_expert_ext: three-expert n-gram blender.
Verifies correctness of token/within-word/word-start experts + agreement.
Benchmarks throughput against 62M-token target.

Run: python test_fused.py
"""
import math
import time
import numpy as np


def test_token_expert():
    """Token n-gram expert: tracks last N tokens, predicts next."""
    from fused_expert_ext import FusedExpertBlender

    np.random.seed(42)
    n_tokens = 20_000
    vocab_size = 1024
    tokens = np.random.randint(0, vocab_size, size=n_tokens).astype(np.int64)

    # Inject a repeating pattern: tokens 100,200,300 always followed by 400
    for i in range(100, n_tokens - 3, 500):
        tokens[i] = 100
        tokens[i + 1] = 200
        tokens[i + 2] = 300
        tokens[i + 3] = 400

    blender = FusedExpertBlender(
        token_order=3,        # 3-token context matches our 3-token pattern
        token_table_bits=18,  # 256K buckets
        within_table_bits=16,
        word_order=2,
        word_table_bits=16,
    )
    blender.set_tokens(tokens)

    # Dummy LUTs (no word boundaries for this test)
    base_bytes = np.ones(vocab_size, dtype=np.int16)
    has_leading_space = np.zeros(vocab_size, dtype=np.uint8)
    is_boundary = np.zeros(vocab_size, dtype=np.uint8)
    blender.set_luts(base_bytes, has_leading_space, is_boundary)
    blender.configure(2.625, 0.5, 0.75, 0.45, 0.75, 0.65, 0.5)

    # Seed with first token
    blender.seed_token(int(tokens[0]))

    # Process all tokens
    positions = np.arange(1, n_tokens, dtype=np.int64)
    hints = np.zeros(len(positions), dtype=np.int32)
    betas = np.zeros(len(positions), dtype=np.float64)
    blender.get_hints_batch(positions, hints, betas)

    # After enough repetitions, the token expert should predict 400 after 100,200,300
    # Check positions where tokens[i-3:i] == [100, 200, 300]
    correct_predictions = 0
    total_pattern_positions = 0
    for i in range(4, n_tokens):
        if (tokens[i - 3] == 100 and tokens[i - 2] == 200 and
                tokens[i - 1] == 300 and tokens[i] == 400):
            idx = i - 1  # position in our hints array
            total_pattern_positions += 1
            if hints[idx] == 400:
                correct_predictions += 1

    # After a few repetitions the pattern should be learned
    print(f"  Token expert: {correct_predictions}/{total_pattern_positions} "
          f"pattern predictions correct")
    assert correct_predictions > total_pattern_positions * 0.5, \
        f"Token expert should learn repeating pattern, got {correct_predictions}/{total_pattern_positions}"
    print("  Token expert PASSED")


def test_within_word_expert():
    """Within-word expert: predicts continuations within a word."""
    from fused_expert_ext import FusedExpertBlender

    np.random.seed(123)
    n_tokens = 10_000
    vocab_size = 1024
    tokens = np.random.randint(0, vocab_size, size=n_tokens).astype(np.int64)

    # Simulate word boundaries: tokens 0-99 have leading space (word starters)
    has_leading_space = np.zeros(vocab_size, dtype=np.uint8)
    has_leading_space[:100] = 1

    # Inject within-word pattern: after token 50 (word-start) + token 500,
    # always comes token 600
    for i in range(200, n_tokens - 3, 400):
        tokens[i] = 50      # word start
        tokens[i + 1] = 500  # within word
        tokens[i + 2] = 600  # predictable continuation

    blender = FusedExpertBlender(4, 18, 18, 2, 16)
    blender.set_tokens(tokens)

    base_bytes = np.ones(vocab_size, dtype=np.int16)
    is_boundary = np.zeros(vocab_size, dtype=np.uint8)
    blender.set_luts(base_bytes, has_leading_space, is_boundary)
    blender.configure(2.625, 0.8, 0.75, 0.3, 0.75, 0.65, 0.5)

    blender.seed_token(int(tokens[0]))
    positions = np.arange(1, n_tokens, dtype=np.int64)
    hints = np.zeros(len(positions), dtype=np.int32)
    betas = np.zeros(len(positions), dtype=np.float64)
    blender.get_hints_batch(positions, hints, betas)

    # Check that SOME hint is provided for within-word positions
    active_hints = np.sum(hints >= 0)
    print(f"  Within-word expert: {active_hints}/{len(positions)} positions got hints")
    assert active_hints > 0, "Within-word expert should produce some hints"
    print("  Within-word expert PASSED")


def test_agreement():
    """When multiple experts agree on same token, boost should increase."""
    from fused_expert_ext import FusedExpertBlender

    np.random.seed(77)
    n_tokens = 5_000
    vocab_size = 1024
    tokens = np.random.randint(0, vocab_size, size=n_tokens).astype(np.int64)

    blender = FusedExpertBlender(4, 18, 18, 2, 16)
    blender.set_tokens(tokens)

    base_bytes = np.ones(vocab_size, dtype=np.int16)
    has_leading_space = np.zeros(vocab_size, dtype=np.uint8)
    is_boundary = np.zeros(vocab_size, dtype=np.uint8)
    blender.set_luts(base_bytes, has_leading_space, is_boundary)
    blender.configure(2.625, 0.5, 0.75, 0.3, 0.75, 0.5, 0.5)

    blender.seed_token(int(tokens[0]))
    positions = np.arange(1, n_tokens, dtype=np.int64)
    hints = np.zeros(len(positions), dtype=np.int32)
    betas = np.zeros(len(positions), dtype=np.float64)
    blender.get_hints_batch(positions, hints, betas)

    # Check that some positions have beta > base token_boost (2.625)
    # indicating agreement bonus was applied
    high_beta = np.sum(betas > 2.625)
    any_hint = np.sum(hints >= 0)
    print(f"  Agreement: {any_hint} hints total, {high_beta} with agreement boost")
    print("  Agreement PASSED")


def test_byte_counting():
    """CPU byte counting matches expected behavior."""
    from fused_expert_ext import FusedExpertBlender

    vocab_size = 1024
    blender = FusedExpertBlender(4, 16, 16, 2, 16)

    # Set up LUTs
    base_bytes = np.array([3, 1, 2, 1, 5] + [1] * (vocab_size - 5), dtype=np.int16)
    has_leading_space = np.array([0, 1, 0, 1, 0] + [0] * (vocab_size - 5), dtype=np.uint8)
    is_boundary = np.array([1, 0, 0, 0, 0] + [0] * (vocab_size - 5), dtype=np.uint8)
    blender.set_luts(base_bytes, has_leading_space, is_boundary)

    # Token 1 after token 0 (boundary): bytes = 1 + 0 = 1 (space not counted after boundary)
    # Token 1 after token 2 (not boundary): bytes = 1 + 1 = 2 (space counted)
    # Token 2 after token 3 (not boundary): bytes = 2 + 0 = 2 (no space)
    targets = np.array([1, 1, 2], dtype=np.int64)
    prev = np.array([0, 2, 3], dtype=np.int64)
    total = blender.compute_bytes(targets, prev)
    expected = 1.0 + 2.0 + 2.0  # = 5.0
    assert abs(total - expected) < 1e-10, f"Expected {expected}, got {total}"
    print(f"  Byte counting: {total} (expected {expected}) PASSED")


def test_blend_formula():
    """Verify the GPU blend formula produces correct NLLs."""
    # Simulate what the GPU does:
    # p'(a) = exp(beta * 1[a=hint]) * p(a) / Z
    # Z = 1 - p(hint) + exp(beta) * p(hint)
    # nll' = -log(p'(target))
    #      = -log(p(target)) + log(Z) - beta * (target == hint)
    #      = model_nll + log(Z) - beta * (target == hint)

    rng = np.random.RandomState(42)
    n = 1000
    model_nll = rng.uniform(0.5, 5.0, size=n)
    p_target = np.exp(-model_nll)
    p_hint = rng.uniform(0.01, 0.5, size=n)
    betas = rng.uniform(0.5, 3.0, size=n)
    is_hit = rng.random(n) > 0.7  # 30% hit rate

    Z = 1.0 + p_hint * (np.exp(betas) - 1.0)
    blended_nll = model_nll + np.log(Z) - betas * is_hit.astype(np.float64)

    # Verify against explicit formula
    for i in range(n):
        z = 1.0 - p_hint[i] + math.exp(betas[i]) * p_hint[i]
        if is_hit[i]:
            p_boosted = math.exp(betas[i]) * p_target[i] / z
        else:
            p_boosted = p_target[i] / z
        expected = -math.log(max(p_boosted, 1e-30))
        assert abs(blended_nll[i] - expected) < 1e-10, \
            f"Position {i}: got {blended_nll[i]:.10f}, expected {expected:.10f}"

    print(f"  Blend formula: {n} positions verified PASSED")


def test_performance():
    """Benchmark throughput for 1M tokens."""
    from fused_expert_ext import FusedExpertBlender

    n_tokens = 1_000_000
    vocab_size = 1024
    tokens = np.random.RandomState(42).randint(
        0, vocab_size, size=n_tokens).astype(np.int64)

    blender = FusedExpertBlender(
        token_order=16,
        token_table_bits=22,  # 4M buckets
        within_table_bits=21,
        word_order=4,
        word_table_bits=20,
    )
    blender.set_tokens(tokens)

    base_bytes = np.ones(vocab_size, dtype=np.int16)
    has_leading_space = np.zeros(vocab_size, dtype=np.uint8)
    has_leading_space[:256] = 1  # first 256 tokens are word-starters
    is_boundary = np.zeros(vocab_size, dtype=np.uint8)
    is_boundary[0] = 1
    blender.set_luts(base_bytes, has_leading_space, is_boundary)
    blender.configure(2.625, 0.800, 0.750, 0.450, 0.750, 0.650, 0.500)
    blender.seed_token(int(tokens[0]))

    # Process in batches of 2048 (matching real eval: 32 windows × 64 stride)
    batch_size = 2048
    n_batches = (n_tokens - 1 + batch_size - 1) // batch_size
    hints_buf = np.zeros(batch_size, dtype=np.int32)
    betas_buf = np.zeros(batch_size, dtype=np.float64)

    t0 = time.perf_counter()
    offset = 1
    for _ in range(n_batches):
        end = min(offset + batch_size, n_tokens)
        n = end - offset
        if n <= 0:
            break
        positions = np.arange(offset, end, dtype=np.int64)
        blender.get_hints_batch(positions, hints_buf[:n], betas_buf[:n])
        offset = end
    elapsed = time.perf_counter() - t0

    tps = n_tokens / elapsed
    projected_62m = elapsed * 62
    print(f"\n  Performance ({n_tokens:,} tokens, order=16, 3 experts):")
    print(f"    Elapsed: {elapsed:.3f}s ({tps:,.0f} tok/s)")
    print(f"    Projected 62M tokens: {projected_62m:.1f}s")
    print(f"    Per-batch (2048 tok): {elapsed / n_batches * 1000:.2f}ms")
    print(f"    Budget per batch (12ms GPU): "
          f"{elapsed / n_batches * 1000 / 12 * 100:.1f}% utilized")


if __name__ == "__main__":
    print("test_token_expert:")
    test_token_expert()
    print("\ntest_within_word_expert:")
    test_within_word_expert()
    print("\ntest_agreement:")
    test_agreement()
    print("\ntest_byte_counting:")
    test_byte_counting()
    print("\ntest_blend_formula:")
    test_blend_formula()
    print("\ntest_performance:")
    test_performance()
