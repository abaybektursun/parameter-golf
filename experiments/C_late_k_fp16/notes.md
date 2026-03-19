# Experiment C: Late-K FP16 Passthrough

## Hypothesis
Key projection weights in the last 2 transformer layers are disproportionately
sensitive to quantization. Keeping them in fp16 reduces quant noise at a small
artifact cost (~64KB per key matrix). (Validated: PRs #99, #114)

## Technique
In the quantization pipeline, pattern-match tensor names:
- blocks.{N-2}.attn.c_k.weight → fp16 passthrough
- blocks.{N-1}.attn.c_k.weight → fp16 passthrough
Everything else follows experiment A's int6 schema.

## Why Keys Specifically
Key projections in late layers determine attention routing. Small quantization
errors in K compound through softmax, while V/Q/O errors are smoothly averaged.

## Size Cost
- 2 × (512 × 256) × 2 bytes = 512 KB fp16 (for 4 KV heads)
  Actually: kv_dim = num_kv_heads * head_dim = 4 * 64 = 256
  c_k shape: (256, 512) → 256 × 512 × 2 bytes = 256 KB per layer
  2 layers → 512 KB total
- Must verify artifact stays under 16 MB

## Changes from Experiment A
- Modified quantization to detect late-layer c_k and passthrough as fp16

## Run Command
```bash
RUN_ID=exp_C_latek \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Success Metric
- post-quant val_bpb < experiment A's post-quant val_bpb
- artifact_bytes < 16,000,000

## Results
| Seed | pre-quant | post-quant | quant_gap | artifact_bytes | status |
|------|-----------|------------|-----------|----------------|--------|
|      |           |            |           |                |        |
