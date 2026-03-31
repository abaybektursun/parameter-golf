"""Shared utilities for model analysis scripts (EXP10: MLP 3.5x + mixed quant)."""
import os, sys, types
from pathlib import Path

# --- CUTLASS EVT stub (must run before importing train_gpt) ---
if "cutlass_evt_fusion" not in sys.modules:
    sys.modules["cutlass_evt_fusion"] = types.ModuleType("cutlass_evt_fusion")
import torch
_cutlass_evt_lib = torch.library.Library("cutlass_evt", "DEF")
_cutlass_evt_lib.define("gemm_mul(Tensor go, Tensor down_w, Tensor act_grad) -> Tensor")

import torch.nn.functional as F

# Resolve import path for train_gpt.py
_BASE_DIR = Path("/root/parameter-golf")
sys.path.insert(0, str(_BASE_DIR))

from train_gpt_mlp35_mixed import (
    Hyperparameters, GPT, CastedLinear, RMSNorm,
    restore_low_dim_params_to_fp32,
    build_sentencepiece_luts, load_validation_tokens,
)

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", str(_BASE_DIR / "final_model.pt"))
DATA_PATH = os.environ.get("DATA_PATH", str(_BASE_DIR / "data/datasets/fineweb10B_sp1024"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", str(_BASE_DIR / "data/tokenizers/fineweb_1024_bpe.model"))


def make_args():
    """Build Hyperparameters matching EXP10 (MLP 3.5x, BigramHash 3072x112)."""
    args = Hyperparameters()
    args.data_path = DATA_PATH
    args.val_files = os.path.join(DATA_PATH, "fineweb_val_*.bin")
    args.tokenizer_path = TOKENIZER_PATH
    # EXP10 was trained with these explicit overrides
    args.mlp_mult = float(os.environ.get("MLP_MULT", 3.5))
    args.bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
    args.bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))
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
