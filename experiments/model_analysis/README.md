# Model Analysis — PR #1019 (AR Self-Gen GPTQ + XSA-all + BigramHash 3072)

Three analyses of the pre-quantization model weights:

1. **Loss decomposition** — BPB broken down by token frequency, type, and position
2. **Logit lens** — per-layer prediction quality (residual stream → unembedding)
3. **SVD** — singular value spectra and effective rank of each weight matrix

## Weights

`weights/final_model.pt` — pre-quantization EMA weights (106MB), trained on 2×H100 SXM
with the same config as PR #1019 (seed 314, 7000 steps, step-based warmdown).
Post-EMA val_bpb: 1.1338 (PR original: 1.1354 on 8×H100, ~6927 steps).

## Run

```bash
# From this directory:
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 python3 analysis_svd.py           # CPU only
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 python3 analysis_logit_lens.py    # 1 GPU
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 python3 analysis_loss_decomposition.py  # 1 GPU

# Override paths via env vars:
WEIGHTS_PATH=... DATA_PATH=... TOKENIZER_PATH=... python3 analysis_svd.py
```

Requires: `torch`, `sentencepiece`, `flash_attn_3`, `numpy`.

## Pre-computed Results

See `weights/*.txt` for results from the 2×H100 SXM run.

Note: `loss_decomp_results.txt` was generated before a byte-counting bugfix
(leading-space bytes were unconditionally added; should be conditional on
previous token not being a boundary token). The impact is small (<0.001 BPB)
and relative bucket proportions are unaffected.
