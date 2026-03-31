"""Compare compression: LZMA-9 vs Brotli-11 vs Brotli-11+byte-shuffle on int6 GPTQ artifact."""
import io
import time
import zlib
import lzma
import numpy as np
import torch

# --- Byte-shuffle functions (from PR #1089) ---
_BSHF_MAGIC = b"BSHF"

def _byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off:dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data: bytes) -> bytes:
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()

# --- Load the existing LZMA artifact ---
print("Loading existing LZMA artifact...")
with open("final_model.int6.ptz", "rb") as f:
    lzma_blob = f.read()
print(f"  LZMA compressed size: {len(lzma_blob):,} bytes ({len(lzma_blob)/1024/1024:.2f} MB)")

# Decompress to get raw data
t0 = time.time()
raw = lzma.decompress(lzma_blob)
t_lzma_decomp = time.time() - t0
print(f"  LZMA decompressed size: {len(raw):,} bytes ({len(raw)/1024/1024:.2f} MB)")
print(f"  LZMA decompress time: {t_lzma_decomp:.2f}s")

# Verify roundtrip
state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
print(f"  State dict keys: {list(state.keys())[:5]}...")

# --- Test 1: Re-compress with LZMA-9 (sanity) ---
t0 = time.time()
lzma9 = lzma.compress(raw, preset=9)
t_lzma = time.time() - t0
print(f"\n[LZMA-9] {len(lzma9):,} bytes ({len(lzma9)/1024/1024:.2f} MB) in {t_lzma:.1f}s")

# --- Test 2: Brotli-11 (no shuffle) ---
import brotli
t0 = time.time()
brotli11 = brotli.compress(raw, quality=11)
t_brotli = time.time() - t0
print(f"[Brotli-11] {len(brotli11):,} bytes ({len(brotli11)/1024/1024:.2f} MB) in {t_brotli:.1f}s")
delta = len(brotli11) - len(lzma9)
print(f"  vs LZMA-9: {delta:+,} bytes ({delta/len(lzma9)*100:+.2f}%)")

# --- Test 3: Byte-shuffle + Brotli-11 ---
t0 = time.time()
shuffled = _byte_shuffle(raw, stride=2)
brotli11_shuf = brotli.compress(shuffled, quality=11)
t_brotli_shuf = time.time() - t0
print(f"[Brotli-11+shuffle] {len(brotli11_shuf):,} bytes ({len(brotli11_shuf)/1024/1024:.2f} MB) in {t_brotli_shuf:.1f}s")
delta = len(brotli11_shuf) - len(lzma9)
print(f"  vs LZMA-9: {delta:+,} bytes ({delta/len(lzma9)*100:+.2f}%)")

# --- Test 4: Byte-shuffle + LZMA-9 ---
t0 = time.time()
lzma9_shuf = lzma.compress(shuffled, preset=9)
t_lzma_shuf = time.time() - t0
print(f"[LZMA-9+shuffle] {len(lzma9_shuf):,} bytes ({len(lzma9_shuf)/1024/1024:.2f} MB) in {t_lzma_shuf:.1f}s")
delta = len(lzma9_shuf) - len(lzma9)
print(f"  vs LZMA-9: {delta:+,} bytes ({delta/len(lzma9)*100:+.2f}%)")

# --- Test 5: Brotli quality sweep ---
print(f"\nBrotli quality sweep:")
for q in [6, 8, 9, 10, 11]:
    t0 = time.time()
    blob = brotli.compress(shuffled, quality=q)
    dt = time.time() - t0
    print(f"  q={q:2d}: {len(blob):,} bytes ({len(blob)/1024/1024:.2f} MB) in {dt:.1f}s")

# --- Test 6: Roundtrip verification ---
print("\nRoundtrip verification...")
decompressed = brotli.decompress(brotli11_shuf)
unshuffled = _byte_unshuffle(decompressed)
assert unshuffled == raw, "ROUNDTRIP FAILED!"
print("  Brotli+shuffle roundtrip: OK")

# --- Summary ---
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
results = [
    ("LZMA-9", len(lzma9), t_lzma),
    ("Brotli-11", len(brotli11), t_brotli),
    ("LZMA-9+shuffle", len(lzma9_shuf), t_lzma_shuf),
    ("Brotli-11+shuffle", len(brotli11_shuf), t_brotli_shuf),
]
results.sort(key=lambda x: x[1])
for name, size, dt in results:
    saved = len(lzma9) - size
    print(f"  {name:20s}: {size/1024/1024:.3f} MB  ({saved/1024:+.1f} KB vs LZMA-9)  [{dt:.1f}s]")

code_size = 124789  # from fused MLP script
print(f"\nWith code ({code_size:,} bytes):")
for name, size, dt in results:
    total = size + code_size
    print(f"  {name:20s}: {total/1024/1024:.3f} MB total submission")
