"""Shared utilities for model analysis scripts."""
import os, sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Resolve import path for train_gpt.py
_RECORD_DIR = Path(__file__).resolve().parent.parent.parent / "records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072"
_SCRIPT_DIR = Path(__file__).resolve().parent
for p in [str(_SCRIPT_DIR), str(_RECORD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from train_gpt import (  # noqa: E402
    Hyperparameters, GPT, CastedLinear, RMSNorm,
    restore_low_dim_params_to_fp32,
    build_sentencepiece_luts, load_validation_tokens,
)

# Defaults: local paths (override via env vars for remote)
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", str(_SCRIPT_DIR / "weights/final_model.pt"))
DATA_PATH = os.environ.get("DATA_PATH", str(_RECORD_DIR.parent.parent.parent / "data/datasets/fineweb10B_sp1024"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", str(_RECORD_DIR.parent.parent.parent / "data/tokenizers/fineweb_1024_bpe.model"))


def make_args():
    """Build Hyperparameters matching the PR #1019 model config."""
    args = Hyperparameters()
    args.data_path = DATA_PATH
    args.val_files = os.path.join(DATA_PATH, "fineweb_val_*.bin")
    args.tokenizer_path = TOKENIZER_PATH
    args.bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
    args.bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))
    args.xsa_last_n = 11
    args.rope_dims = 16
    args.ln_scale = True
    args.ve_enabled = True
    args.ve_dim = 128
    args.ve_layers = "9,10"
    return args


def load_model(args, device):
    """Instantiate GPT and load pre-quantization weights."""
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    sd = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model
