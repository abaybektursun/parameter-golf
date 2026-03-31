"""
Test fast_ngram_ext C++ extension against Python reference implementation.

Run: python test_blend.py
"""
import time
import numpy as np

# ── Python reference (extracted from eval_ngram.py) ──────────────────────────

PRIMES = np.array(
    [36313, 27191, 51647, 81929, 131071, 174763, 233017, 310019, 412553],
    dtype=np.uint64,
)


def hash_context(val_np, positions, ctx_width, primes, ng_mask):
    ctx_hash = np.zeros(len(positions), dtype=np.uint64)
    for k in range(ctx_width):
        tok = val_np[positions - (ctx_width - k)].astype(np.uint64)
        ctx_hash ^= tok * primes[k % len(primes)]
    return ctx_hash


def hash_full(ctx_hash, target_tokens, ctx_width, primes, ng_mask):
    tgt = target_tokens.astype(np.uint64)
    return ctx_hash ^ (tgt * primes[ctx_width % len(primes)])


def py_process_stride(val_np, positions, model_nll, ctx_tables, full_tables,
                      orders, min_count, ng_mask, alpha, mixing_fn):
    """Reference Python: one stride of lookup + mix + update."""
    n_seg = len(positions)
    n_orders = len(orders)
    seg_model_p = np.exp(-model_nll)

    best_p_ng = np.full(n_seg, -1.0)

    for oi in range(n_orders - 1, -1, -1):
        order = orders[oi]
        ctx_w = order - 1
        valid = (positions >= order) & (best_p_ng < 0)
        if not valid.any():
            continue
        v_idx = np.nonzero(valid)[0]
        jv = positions[v_idx]

        ctx_hash = hash_context(val_np, jv, ctx_w, PRIMES, ng_mask)
        ctx_key = (ctx_hash & ng_mask).astype(np.int64)
        tgt_np = val_np[jv]
        full_key = (hash_full(ctx_hash, tgt_np, ctx_w, PRIMES, ng_mask)
                    & ng_mask).astype(np.int64)

        cc = ctx_tables[oi][ctx_key].astype(np.float64)
        fc = full_tables[oi][full_key].astype(np.float64)
        has_match = cc >= float(min_count)

        if has_match.any():
            p = np.minimum(fc, cc) / np.maximum(cc, 1.0)
            p = np.clip(p, 0.0, 1.0)
            fill_idx = v_idx[has_match]
            best_p_ng[fill_idx] = p[has_match]

    has_ngram = best_p_ng >= 0
    if has_ngram.any():
        ng_idx = np.nonzero(has_ngram)[0]
        pm = seg_model_p[ng_idx]
        pn = best_p_ng[ng_idx]

        if mixing_fn == 0:
            mixed_p = (1.0 - alpha) * pm + alpha * pn
        elif mixing_fn == 1:
            eps = 1e-7
            pm_c = np.clip(pm, eps, 1.0 - eps)
            pn_c = np.clip(pn, eps, 1.0 - eps)
            lm = np.log(pm_c / (1.0 - pm_c))
            ln = np.log(pn_c / (1.0 - pn_c))
            combined = (1.0 - alpha) * lm + alpha * ln
            mixed_p = 1.0 / (1.0 + np.exp(-combined))
        else:
            eps = 1e-12
            log_mix = ((1.0 - alpha) * np.log(np.clip(pm, eps, 1.0))
                       + alpha * np.log(np.clip(pn, eps, 1.0)))
            mixed_p = np.exp(log_mix)

        seg_model_p[ng_idx] = mixed_p

    out_nll = -np.log(np.clip(seg_model_p, 1e-12, 1.0))

    for oi in range(n_orders):
        order = orders[oi]
        ctx_w = order - 1
        valid = positions >= order
        if not valid.any():
            continue
        v_idx = np.nonzero(valid)[0]
        jv = positions[v_idx]
        ctx_hash = hash_context(val_np, jv, ctx_w, PRIMES, ng_mask)
        ctx_key = (ctx_hash & ng_mask).astype(np.int64)
        tgt_np = val_np[jv]
        full_key = (hash_full(ctx_hash, tgt_np, ctx_w, PRIMES, ng_mask)
                    & ng_mask).astype(np.int64)
        np.add.at(ctx_tables[oi], ctx_key, 1)
        np.add.at(full_tables[oi], full_key, 1)

    return out_nll


# ── Tests ────────────────────────────────────────────────────────────────────

def test_correctness():
    """Verify C++ matches Python for fixed alpha, all three mixing functions."""
    from fast_ngram_ext import NGramBlender

    np.random.seed(42)
    vocab_size = 1024
    n_tokens = 10_000
    val_np = np.random.randint(0, vocab_size, size=n_tokens).astype(np.int64)

    min_order, max_order = 2, 7
    ngram_buckets = 1 << 20
    min_count = 2
    alpha = 0.40
    stride = 64

    orders = list(range(min_order, max_order + 1))
    n_orders = len(orders)
    ng_mask = np.uint64(ngram_buckets - 1)

    for mixing_fn, name in [(0, "linear"), (1, "logistic"), (2, "geometric")]:
        py_ctx = [np.zeros(ngram_buckets, dtype=np.uint32)
                  for _ in range(n_orders)]
        py_full = [np.zeros(ngram_buckets, dtype=np.uint32)
                   for _ in range(n_orders)]

        blender = NGramBlender(min_order, max_order, ngram_buckets, min_count)
        blender.set_tokens(val_np)
        blender.configure_alpha(0, alpha, 0.05, 0.55, 2.0, 4.0, 3.0, 0.25)
        blender.set_mixing_fn(mixing_fn)

        rng = np.random.RandomState(99)
        max_err = 0.0
        n_strides = (n_tokens - 1) // stride

        for si in range(n_strides):
            start = si * stride + 1
            end = min(start + stride, n_tokens)
            positions = np.arange(start, end, dtype=np.int64)
            model_nll = rng.uniform(0.5, 5.0, size=len(positions))
            entropy = np.empty(0, dtype=np.float64)

            py_out = py_process_stride(
                val_np, positions, model_nll.copy(),
                py_ctx, py_full, orders, min_count, ng_mask, alpha, mixing_fn,
            )
            cpp_out = np.asarray(blender.process_stride(
                positions, model_nll, entropy,
            ))

            err = np.max(np.abs(py_out - cpp_out))
            max_err = max(max_err, err)
            assert err < 1e-10, (
                f"Mismatch at stride {si} ({name}): max_err={err:.2e}\n"
                f"  py:  {py_out[:5]}\n  cpp: {cpp_out[:5]}"
            )

        print(f"  {name:10s} PASSED  (max error: {max_err:.2e}, "
              f"{n_strides} strides)")


def test_batch_correctness():
    """Verify process_batch matches process_stride (same results)."""
    from fast_ngram_ext import NGramBlender

    np.random.seed(42)
    n_tokens = 10_000
    val_np = np.random.randint(0, 1024, size=n_tokens).astype(np.int64)

    min_order, max_order = 2, 7
    ngram_buckets = 1 << 20
    min_count = 2
    alpha = 0.40
    stride = 64
    batch_seqs = 32

    rng = np.random.RandomState(99)
    n_strides = (n_tokens - 1) // stride

    # --- Run with process_stride ---
    blender_s = NGramBlender(min_order, max_order, ngram_buckets, min_count)
    blender_s.set_tokens(val_np)
    blender_s.configure_alpha(0, alpha, 0.05, 0.55, 2.0, 4.0, 3.0, 0.25)
    blender_s.set_mixing_fn(0)

    stride_results = []
    rng_s = np.random.RandomState(99)
    for si in range(n_strides):
        start = si * stride + 1
        end = min(start + stride, n_tokens)
        positions = np.arange(start, end, dtype=np.int64)
        model_nll = rng_s.uniform(0.5, 5.0, size=len(positions))
        out = np.asarray(blender_s.process_stride(
            positions, model_nll, np.empty(0, dtype=np.float64)))
        stride_results.append(out.copy())

    # --- Run with process_batch ---
    blender_b = NGramBlender(min_order, max_order, ngram_buckets, min_count)
    blender_b.set_tokens(val_np)
    blender_b.configure_alpha(0, alpha, 0.05, 0.55, 2.0, 4.0, 3.0, 0.25)
    blender_b.set_mixing_fn(0)

    batch_results = []
    rng_b = np.random.RandomState(99)

    for bi in range(0, n_strides, batch_seqs):
        batch_end = min(bi + batch_seqs, n_strides)
        all_pos = []
        all_nll = []
        seg_lens = []

        for si in range(bi, batch_end):
            start = si * stride + 1
            end = min(start + stride, n_tokens)
            positions = np.arange(start, end, dtype=np.int64)
            model_nll = rng_b.uniform(0.5, 5.0, size=len(positions))
            all_pos.append(positions)
            all_nll.append(model_nll)
            seg_lens.append(len(positions))

        cat_pos = np.concatenate(all_pos)
        cat_nll = np.concatenate(all_nll)
        seg_lens_arr = np.array(seg_lens, dtype=np.int32)

        mixed = np.asarray(blender_b.process_batch(
            cat_pos, seg_lens_arr, cat_nll, np.empty(0, dtype=np.float64)))

        offset = 0
        for sl in seg_lens:
            batch_results.append(mixed[offset:offset + sl].copy())
            offset += sl

    # Compare
    max_err = 0.0
    for i, (s_out, b_out) in enumerate(zip(stride_results, batch_results)):
        err = np.max(np.abs(s_out - b_out))
        max_err = max(max_err, err)
        assert err < 1e-15, f"Batch mismatch at stride {i}: {err:.2e}"

    print(f"  batch vs stride PASSED  (max error: {max_err:.2e}, "
          f"{n_strides} strides)")


def test_performance():
    """Benchmark: Python vs C++ stride vs C++ batch."""
    from fast_ngram_ext import NGramBlender

    n_tokens = 100_000
    vocab_size = 1024
    min_order, max_order = 2, 7
    ngram_buckets = 1 << 22
    min_count = 2
    alpha = 0.40
    stride = 64
    batch_seqs = 32

    orders = list(range(min_order, max_order + 1))
    n_orders = len(orders)
    ng_mask = np.uint64(ngram_buckets - 1)

    val_np = np.random.RandomState(42).randint(
        0, vocab_size, size=n_tokens).astype(np.int64)
    n_strides = (n_tokens - 1) // stride

    # Pre-generate model NLLs
    rng = np.random.RandomState(7)
    all_nll_data = [rng.uniform(0.5, 5.0, size=min(stride, n_tokens - 1 - s * stride))
                    for s in range(n_strides)]

    empty_ent = np.empty(0, dtype=np.float64)

    # -- Python --
    py_ctx = [np.zeros(ngram_buckets, dtype=np.uint32)
              for _ in range(n_orders)]
    py_full = [np.zeros(ngram_buckets, dtype=np.uint32)
               for _ in range(n_orders)]

    t0 = time.perf_counter()
    for si in range(n_strides):
        start = si * stride + 1
        end = min(start + stride, n_tokens)
        positions = np.arange(start, end, dtype=np.int64)
        py_process_stride(
            val_np, positions, all_nll_data[si].copy(),
            py_ctx, py_full, orders, min_count, ng_mask, alpha, 0,
        )
    py_time = time.perf_counter() - t0

    # -- C++ stride --
    blender = NGramBlender(min_order, max_order, ngram_buckets, min_count)
    blender.set_tokens(val_np)
    blender.configure_alpha(0, alpha, 0.05, 0.55, 2.0, 4.0, 3.0, 0.25)
    blender.set_mixing_fn(0)

    t0 = time.perf_counter()
    for si in range(n_strides):
        start = si * stride + 1
        end = min(start + stride, n_tokens)
        positions = np.arange(start, end, dtype=np.int64)
        blender.process_stride(positions, all_nll_data[si], empty_ent)
    cpp_stride_time = time.perf_counter() - t0

    # -- C++ batch --
    blender2 = NGramBlender(min_order, max_order, ngram_buckets, min_count)
    blender2.set_tokens(val_np)
    blender2.configure_alpha(0, alpha, 0.05, 0.55, 2.0, 4.0, 3.0, 0.25)
    blender2.set_mixing_fn(0)

    t0 = time.perf_counter()
    for bi in range(0, n_strides, batch_seqs):
        batch_end = min(bi + batch_seqs, n_strides)
        all_pos = []
        all_nll_batch = []
        seg_lens = []

        for si in range(bi, batch_end):
            start = si * stride + 1
            end = min(start + stride, n_tokens)
            all_pos.append(np.arange(start, end, dtype=np.int64))
            all_nll_batch.append(all_nll_data[si])
            seg_lens.append(end - start)

        cat_pos = np.concatenate(all_pos)
        cat_nll = np.concatenate(all_nll_batch)
        seg_lens_arr = np.array(seg_lens, dtype=np.int32)
        blender2.process_batch(cat_pos, seg_lens_arr, cat_nll, empty_ent)
    cpp_batch_time = time.perf_counter() - t0

    print(f"\n  Benchmark ({n_tokens:,} tokens, stride={stride}, batch={batch_seqs}):")
    print(f"    Python:       {py_time:.3f}s  ({n_tokens / py_time:>10,.0f} tok/s)")
    print(f"    C++ stride:   {cpp_stride_time:.3f}s  ({n_tokens / cpp_stride_time:>10,.0f} tok/s)")
    print(f"    C++ batch:    {cpp_batch_time:.3f}s  ({n_tokens / cpp_batch_time:>10,.0f} tok/s)")
    print(f"    Stride speedup over Python: {py_time / cpp_stride_time:.1f}x")
    print(f"    Batch  speedup over Python: {py_time / cpp_batch_time:.1f}x")
    print(f"    Batch  speedup over stride: {cpp_stride_time / cpp_batch_time:.2f}x")


def test_scale():
    """Simulate 62M-token scale to project real eval time."""
    from fast_ngram_ext import NGramBlender

    n_tokens = 1_000_000
    stride = 64
    batch_seqs = 32
    val_np = np.random.RandomState(42).randint(
        0, 1024, size=n_tokens).astype(np.int64)
    n_strides = (n_tokens - 1) // stride

    empty_ent = np.empty(0, dtype=np.float64)
    rng = np.random.RandomState(7)

    blender = NGramBlender(2, 7, 1 << 22, 2)
    blender.set_tokens(val_np)
    blender.configure_alpha(0, 0.40, 0.05, 0.55, 2.0, 4.0, 3.0, 0.25)
    blender.set_mixing_fn(0)

    t0 = time.perf_counter()
    for bi in range(0, n_strides, batch_seqs):
        batch_end = min(bi + batch_seqs, n_strides)
        all_pos = []
        all_nll = []
        seg_lens = []
        for si in range(bi, batch_end):
            start = si * stride + 1
            end = min(start + stride, n_tokens)
            all_pos.append(np.arange(start, end, dtype=np.int64))
            all_nll.append(rng.uniform(0.5, 5.0, size=end - start))
            seg_lens.append(end - start)
        cat_pos = np.concatenate(all_pos)
        cat_nll = np.concatenate(all_nll)
        seg_lens_arr = np.array(seg_lens, dtype=np.int32)
        blender.process_batch(cat_pos, seg_lens_arr, cat_nll, empty_ent)
    elapsed = time.perf_counter() - t0

    projected_62m = elapsed * (62_000_000 / n_tokens)
    print(f"\n  Scale test ({n_tokens:,} tokens):")
    print(f"    Elapsed: {elapsed:.3f}s ({n_tokens / elapsed:,.0f} tok/s)")
    print(f"    Projected 62M tokens: {projected_62m:.1f}s")


if __name__ == "__main__":
    print("test_correctness:")
    test_correctness()
    print("\ntest_batch_correctness:")
    test_batch_correctness()
    print("\ntest_performance:")
    test_performance()
    print("\ntest_scale:")
    test_scale()
