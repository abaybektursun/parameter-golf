"""
Mixed-precision GPTQ: int4 attention, int7 MLP_down, int6 MLP_up.

From NEXT_STEPS.md analysis:
- MLP accounts for 80% of quantization damage
- MLP_down is the single most sensitive matrix (50% of quant damage)
- Attention Q matrices are nearly insensitive (72.6% SVD utilization)

Strategy: steal bits from attention, give to MLP_down.
  - Attention (Q, K, V, Out): int4 (clip_range=7)
  - MLP_up: int5 (clip_range=15) or int6 (clip_range=31)
  - MLP_down: int8 (clip_range=127)
  - Embeddings/small params: unchanged

Usage:
    PYTHONUNBUFFERED=1 python eval_mixed_precision.py \
        --model-pt final_model.pt --device cuda:0
"""
from __future__ import annotations
import argparse
import io
import lzma
import math
import time
import glob
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=num_tokens, offset=offset).astype(np.uint16, copy=False))


def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        if ".proj." in name or "mlp_down" in name or "mlp.proj" in name:
            return "mlp_down"
        return "mlp_up"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_per_row(t, clip_range):
    """Quantize to ±clip_range with per-row scales and percentile search."""
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s, best_err
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    err = (t32 - q.float() * scale.float()).pow(2).mean().item()
    return q, scale, err


def quantize_gptq(weight, hessian, clip_range, block_size=128):
    """Full GPTQ with variable clip_range."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        q, s, err = quantize_per_row(t32, clip_range)
        return q, s
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    best_q, best_scale, best_err = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, inv_perm]
    return best_q, best_scale


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix",
    "resid_mixes", "q_gain", "skip_weight", "skip_weights", "smear",
    "dtg_gate", "ve_layer_scales", "ve_shared.scale", "attn_gate", "vr_lambda",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pt", default="final_model.pt")
    parser.add_argument("--val-pattern", default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-bits", type=int, default=4, help="Bits for attention weights")
    parser.add_argument("--mlp-up-bits", type=int, default=6, help="Bits for MLP up weights")
    parser.add_argument("--mlp-down-bits", type=int, default=8, help="Bits for MLP down weights")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--target-mb", type=float, default=15.9)
    args = parser.parse_args()

    device = torch.device(args.device)

    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gpt", "train_gpt.py")
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)

    # Set env vars to match the memmap model's config
    import os
    os.environ.setdefault("BIGRAM_VOCAB_SIZE", "3072")
    os.environ.setdefault("BIGRAM_DIM", "112")
    os.environ.setdefault("WARMDOWN_ITERS", "4000")
    os.environ.setdefault("XSA_LAST_N", "11")
    hp = tg.Hyperparameters()

    # Bit width -> clip range mapping
    bits_to_clip = {4: 7, 5: 15, 6: 31, 7: 63, 8: 127}
    attn_clip = bits_to_clip[args.attn_bits]
    mlp_up_clip = bits_to_clip[args.mlp_up_bits]
    mlp_down_clip = bits_to_clip[args.mlp_down_bits]

    print(f"Mixed-precision GPTQ:")
    print(f"  Attention: int{args.attn_bits} (clip_range={attn_clip})")
    print(f"  MLP up:    int{args.mlp_up_bits} (clip_range={mlp_up_clip})")
    print(f"  MLP down:  int{args.mlp_down_bits} (clip_range={mlp_down_clip})")

    # Load full-precision model
    print(f"Loading {args.model_pt}...")
    export_sd = torch.load(args.model_pt, map_location="cpu")
    print(f"Loaded {sum(t.numel() for t in export_sd.values())} params")

    # Build model for eval
    model = tg.GPT(
        vocab_size=hp.vocab_size, num_layers=hp.num_layers, model_dim=hp.model_dim,
        num_heads=hp.num_heads, num_kv_heads=hp.num_kv_heads, mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings, tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap, rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init, bigram_vocab_size=hp.bigram_vocab_size,
        bigram_dim=hp.bigram_dim, xsa_last_n=hp.xsa_last_n,
        rope_dims=hp.rope_dims, ln_scale=hp.ln_scale,
        ve_enabled=hp.ve_enabled, ve_dim=hp.ve_dim, ve_layers=hp.ve_layers,
    ).to(device).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for m in model.modules():
        if isinstance(m, tg.CastedLinear):
            m.float()
    tg.restore_low_dim_params_to_fp32(model)

    # Unbank for quantization
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = tg._unbank_state_dict(sd_cpu, hp.num_layers)

    # Collect Hessians via AR self-gen (same as the original pipeline)
    model.load_state_dict(export_sd, strict=False)
    print("Generating AR calibration data for GPTQ...")
    t_gen = time.perf_counter()
    ar_tokens = tg.generate_autoregressive_calib(
        model, device, num_seqs=64, seq_len=hp.train_seq_len,
        vocab_size=hp.vocab_size, temperature=0.8, batch_size=8, seed=hp.seed,
    )
    print(f"Generated {len(ar_tokens)} sequences in {time.perf_counter()-t_gen:.1f}s")

    # Build hessian model
    hessian_model = tg._HessianGPT(
        vocab_size=hp.vocab_size, num_layers=hp.num_layers, model_dim=hp.model_dim,
        num_heads=hp.num_heads, num_kv_heads=hp.num_kv_heads, mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings, logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base, qk_gain_init=hp.qk_gain_init,
        bigram_vocab_size=hp.bigram_vocab_size, bigram_dim=hp.bigram_dim,
        xsa_last_n=hp.xsa_last_n, rope_dims=hp.rope_dims, ln_scale=hp.ln_scale,
        ve_enabled=hp.ve_enabled, ve_dim=hp.ve_dim, ve_layers=hp.ve_layers,
    ).to(device).bfloat16()
    for m in hessian_model.modules():
        if isinstance(m, tg.CastedLinear):
            m.float()
    tg.restore_low_dim_params_to_fp32(hessian_model)
    hessian_model.load_state_dict(
        {k: v.to(device) for k, v in unbanked_sd.items() if k in hessian_model.state_dict()},
        strict=False,
    )
    print("Collecting Hessians...")
    hessians = tg.collect_hessians_from_tokens(hessian_model, ar_tokens, device)
    print(f"Collected Hessians for {len(hessians)} layers")
    del ar_tokens, hessian_model
    torch.cuda.empty_cache()

    # Mixed-precision quantization
    print("Quantizing with mixed precision...")
    result = {}
    meta = {}
    param_bits = {}

    for name, tensor in unbanked_sd.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)

        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue

        H = hessians.get(name)

        if cat == "attn":
            cr = attn_clip
            bits_label = args.attn_bits
        elif cat == "mlp_down":
            cr = mlp_down_clip
            bits_label = args.mlp_down_bits
        elif cat == "mlp_up":
            cr = mlp_up_clip
            bits_label = args.mlp_up_bits
        else:
            cr = 31  # int6 default for embed/other
            bits_label = 6

        if H is not None:
            q, s = quantize_gptq(t, H, cr)
        else:
            q, s, _ = quantize_per_row(t, cr)

        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = {"type": f"int{bits_label}"}
        param_bits[name] = (t.numel(), bits_label)
        print(f"  {name}: {t.shape} -> int{bits_label} (clip={cr})")

    # Compress
    quant_buf = io.BytesIO()
    torch.save({"w": result, "m": meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=9)

    code_bytes = Path("train_gpt.py").read_bytes()
    total_size = len(quant_blob) + len(code_bytes)
    print(f"\nCompressed model: {len(quant_blob)} bytes")
    print(f"Code: {len(code_bytes)} bytes")
    print(f"Total: {total_size} bytes ({total_size/1024/1024:.2f} MiB)")

    if total_size > args.target_mb * 1024 * 1024:
        print(f"WARNING: Exceeds {args.target_mb} MB target!")

    # Param budget breakdown
    attn_params = sum(n for name, (n, b) in param_bits.items() if _classify_param(name) == "attn")
    mlp_up_params = sum(n for name, (n, b) in param_bits.items() if _classify_param(name) == "mlp_up")
    mlp_down_params = sum(n for name, (n, b) in param_bits.items() if _classify_param(name) == "mlp_down")
    other_params = sum(n for name, (n, b) in param_bits.items() if _classify_param(name) not in ("attn", "mlp_up", "mlp_down"))
    print(f"\nParam breakdown:")
    print(f"  Attention: {attn_params:,} params × int{args.attn_bits} = {attn_params*args.attn_bits/8/1e6:.2f} MB")
    print(f"  MLP up:    {mlp_up_params:,} params × int{args.mlp_up_bits} = {mlp_up_params*args.mlp_up_bits/8/1e6:.2f} MB")
    print(f"  MLP down:  {mlp_down_params:,} params × int{args.mlp_down_bits} = {mlp_down_params*args.mlp_down_bits/8/1e6:.2f} MB")
    print(f"  Other:     {other_params:,} params")

    # Dequantize and evaluate
    print("\nDequantizing for eval...")
    deq_unbanked = tg.dequantize_mixed_int6(result, meta, unbanked_sd)
    deq_state = tg._rebank_state_dict(deq_unbanked, hp.num_layers, sd_cpu)

    eval_model = tg.GPT(
        vocab_size=hp.vocab_size, num_layers=hp.num_layers, model_dim=hp.model_dim,
        num_heads=hp.num_heads, num_kv_heads=hp.num_kv_heads, mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings, tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap, rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init, bigram_vocab_size=hp.bigram_vocab_size,
        bigram_dim=hp.bigram_dim, xsa_last_n=hp.xsa_last_n,
        rope_dims=hp.rope_dims, ln_scale=hp.ln_scale,
        ve_enabled=hp.ve_enabled, ve_dim=hp.ve_dim, ve_layers=hp.ve_layers,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, tg.CastedLinear):
            m.float()
    tg.restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    eval_model.eval()

    # Load val tokens
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, hp.vocab_size, device
    )
    val_files = sorted(glob.glob(args.val_pattern))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    usable = ((val_tokens.numel() - 1) // hp.train_seq_len) * hp.train_seq_len
    val_tokens = val_tokens[:usable + 1]
    print(f"Val tokens: {val_tokens.numel() - 1}")

    # Sliding window eval
    print("Running sliding window eval...")
    t_eval = time.perf_counter()
    sw_val_loss, sw_val_bpb = tg.eval_val_sliding(
        hp, eval_model, 0, 1, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.stride,
    )
    eval_time = time.perf_counter() - t_eval

    # Baseline comparison
    baseline_bpb = 1.11197530  # memmap model uniform int6

    print(f"\n{'='*80}")
    print(f"RESULTS: Mixed-Precision GPTQ")
    print(f"  Attn=int{args.attn_bits}, MLP_up=int{args.mlp_up_bits}, MLP_down=int{args.mlp_down_bits}")
    print(f"{'='*80}")
    print(f"Baseline (uniform int6):  val_bpb={baseline_bpb:.8f}")
    print(f"Mixed-precision:          val_loss={sw_val_loss:.8f}  val_bpb={sw_val_bpb:.8f}")
    print(f"Delta:                    {sw_val_bpb - baseline_bpb:+.8f} BPB")
    print(f"Artifact size: {total_size} bytes ({total_size/1024/1024:.2f} MiB)")
    print(f"Eval time: {eval_time:.1f}s")


if __name__ == "__main__":
    main()
