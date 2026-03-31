# Parameter Golf

## FA3 (Flash Attention 3) Setup

Never compile FA3 from source. Install prebuilt wheels:

```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2110
```

Requires PyTorch 2.11.0+cu128. The wheel URL pattern is `cu{CUDA}_torch{VERSION}`.

Quick test: `python -c "from flash_attn_interface import flash_attn_func"`

### Available PyTorch + FA3 combinations (as of 2026-03-30):
- **PyTorch 2.11.0** (latest stable, released 2026-03-23): FA3 wheels for cu128, cu130
- PyTorch 2.10.0: FA3 wheels for cu126, cu128, cu129, cu130
- PyTorch 2.9.1: FA3 wheels for cu128

PyTorch 2.11.0 features FlashAttention-4 backend for FlexAttention on Hopper GPUs.

## GPU Instance Setup (vast.ai)

Use interruptible (spot) H100 SXM instances. Setup pattern:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2110
pip install sentencepiece
```

### Previous setup (PyTorch 2.9.1) — still works:
```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece
```

## Data

- Repo: `willdepueoai/parameter-golf` on HuggingFace (dataset)
- Tokenizer: `datasets/tokenizers/fineweb_1024_bpe.model`
- Val shard: `data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin` (118MB)
