# Experiment: Custom Binary Serialization (drop torch.save)

## Research Background

Analysis of the current serialization pipeline revealed that `torch.save` wraps the
quantized model in a pickle + ZIP archive format. This adds ~515 KB of overhead:
pickle protocol framing, ZIP entry headers (~100 bytes × 184 tensor entries), repeated
string keys across 4 dicts (`quantized`, `scales`, `dtypes`, `qmeta`), and dtype metadata.

The current artifact budget:
- Cap: 16,000,000 bytes (code + model)
- Code: ~57,000 bytes
- Model (torch.save + zlib): ~15,816,489 bytes
- Headroom: **~124 KB**

Compression benchmarks on 16MB of unique int8 weight data (non-repeating, gaussian):
- zlib level=9: 90.7% of original (current approach)
- lzma preset=6: 91.1% of original (**worse** than zlib, +65 KB, 12x slower)
- zstd, brotli: not tested but expected similar to zlib on high-entropy int8 data

Conclusion: the compression algorithm is near-optimal. The waste is in the **container
format**, not the compressor. A custom binary format with a minimal header reduces
serialization overhead from ~515 KB to ~3 KB, saving ~500 KB.

## Hypothesis

Replacing `torch.save` + pickle + ZIP with a raw binary format (tiny header + packed
tensor bytes) will save ~500 KB in the compressed artifact. This reclaims 4x our current
headroom, enabling either:
- MLP_HIDDEN back to 1024 (fixes Triton kernel tile alignment at 128-divisible)
- ~500K more int8 parameters elsewhere in the model
- Breathing room for additional code (fused kernel imports, ~5 KB each)

The decompression code is ~30 lines of Python, costing <2 KB of code budget.

## Plan

### Binary format spec

```
Header (fixed):
  magic:        4 bytes  "PG01"
  num_tensors:  2 bytes  uint16

Per tensor (variable):
  name_len:     1 byte   uint8
  name:         N bytes  utf-8
  ndim:         1 byte   uint8
  shape:        ndim * 4 bytes  (uint32 per dim)
  dtype_code:   1 byte   uint8  (0=int8, 1=float16, 2=float32)
  data:         product(shape) * element_size  bytes

Footer:
  none
```

Total per-tensor overhead: ~30 bytes average (vs ~300 bytes in torch.save ZIP).
92 tensors × 30 bytes = ~2,760 bytes total metadata.

### Implementation

Two functions replacing `quantize_state_dict_int8` serialization and `dequantize_state_dict_int8` deserialization:

**`serialize_model(state_dict) -> bytes`**
1. Run existing int8 quantization (unchanged)
2. Pack all tensors (quantized int8 + fp16 scales + fp16/fp32 passthrough) into
   the binary format above
3. zlib.compress(level=9) the packed bytes
4. Return compressed blob

**`deserialize_model(blob) -> state_dict`**
1. zlib.decompress
2. Parse header, iterate tensors, reconstruct state_dict
3. Dequantize int8 tensors using scales (same math as current)
4. Return state_dict ready for `model.load_state_dict()`

### What stays the same
- Quantization logic (int8 per-row, fp16 scales, passthrough for small tensors)
- zlib compression (proven optimal for this data)
- Model architecture, training loop, eval pipeline
- The `tok_emb.weight` fp16 passthrough from experiment 001

### What changes
- `torch.save(quant_obj, buf)` → `serialize_model(quant_obj)`
- `torch.load(...)` → `deserialize_model(blob)`
- Remove dependency on pickle for model serialization

### Verification
- Bit-exact roundtrip: `deserialize(serialize(state_dict))` must produce identical
  tensors to the current `dequantize(quantize(state_dict))` pipeline
- Same val_bpb after roundtrip (must match current numbers exactly)
- Artifact size must be < current by ~400-500 KB

### Success metric
- Artifact size reduction of ≥400 KB vs current torch.save + zlib
- Zero degradation in val_bpb (this is a lossless format change)
- MLP_HIDDEN can increase from 992 to 1024 within the 16MB budget

### Run command (8xH100)
```bash
RUN_ID=exp003_custom_serial \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_HIDDEN=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  experiments/003_custom_serialization/train_gpt.py
```

### Risks
- Off-by-one in binary packing → corrupted weights (mitigated: bit-exact roundtrip test)
- zlib compresses the custom format differently than the ZIP format (could be slightly
  better or worse — ZIP has its own deflate internally, so removing that layer and
  applying zlib directly to raw bytes should be at least as good)

## Observations

| Run | Seed | Steps | Pre-quant BPB | Post-quant BPB | Artifact Size | Notes |
|-----|------|-------|---------------|----------------|---------------|-------|
|     |      |       |               |                |               |       |

## Post-mortem

(pending results)
