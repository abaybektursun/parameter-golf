# Experiment D: 8192 Vocabulary Tokenizer

## Hypothesis
Larger vocabulary compresses text more efficiently, directly improving the
tokens-per-byte multiplier in the BPB formula. PR #78 achieved 1.186 with
8192 vocab using int6/int8 but WITHOUT sliding window eval or MLP 3x.
Adding the full technique stack to 8192 vocab could break 1.15 BPB.

## Technique
- Use fineweb_8192_bpe.model tokenizer (from HuggingFace sproos/parameter-golf-tokenizers)
- Re-tokenized FineWeb shards: fineweb10B_sp8192
- VOCAB_SIZE=8192 → embedding table grows from 524K to 4.2M params
- Trade off: drop from 10 to 8 layers to fit under 16 MB
- Keep all other meta stack techniques (int6, zstd, fp16 embed, sliding window, tuned Muon)

## Size Budget
- 8 layers × 512dim × MLP 3x = ~18.9M params (estimate)
- Embedding: 8192 × 512 = 4.2M params (fp16 passthrough = 8 MB)
  Wait — 8 MB for embedding alone is too much.
  Need: int8 embedding (not fp16) OR reduced model dim
- PR #78 approach: int8 embedding, int6 weights → 14.8 MB artifact

## Data Prep Required
```bash
# Download 8192-vocab tokenized data
python3 data/cached_challenge_fineweb.py --variant sp8192
```

## Changes from Experiment A
- VOCAB_SIZE=8192, NUM_LAYERS=8
- DATA_PATH, TOKENIZER_PATH point to sp8192 variants
- Embedding quantized as int8 (too large for fp16 passthrough)

## Run Command
```bash
RUN_ID=exp_D_vocab8k \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
NUM_LAYERS=8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Success Metric
- post-quant val_bpb < 1.16
- artifact_bytes < 16,000,000

## Results
| Seed | pre-quant | post-quant | artifact_bytes | status |
|------|-----------|------------|----------------|--------|
|      |           |            |                |        |
