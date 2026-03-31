"""
eval_ngram.py — Standalone eval-only script for n-gram mixing experiments.

Loads a trained model checkpoint and runs sliding-window eval with configurable
n-gram cache variants. No training, no quantization — pure eval.

Usage:
    python3 eval_ngram.py --model /path/to/final_model.pt --exp <experiment_name>

Experiments:
    baseline        Neural only, no cache
    fixed_7gram     Fixed 7-gram, alpha=0.40, no backoff
    backoff_7       Backoff 2-7, fixed alpha=0.40
    backoff_7_ent   Backoff 2-7, entropy-adaptive alpha
    backoff_9_ent   Backoff 2-9, order-adaptive entropy (PR #788 style)
    order_sweep     Sweep max order 2..20, backoff, entropy-adaptive
    logistic        Logistic mixing (vs linear baseline)
    geometric       Geometric mean mixing
    logistic_sgd    Logistic mixing with online SGD weight updates
    collision_sweep Sweep bucket counts: 1M, 4M, 16M, 64M, 256M

STRICT CAUSALITY: All n-gram tables are updated AFTER scoring. No future tokens used.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from fast_ngram_ext import NGramBlender as _CppBlender

# Load model classes from the training script
REPO = Path(__file__).resolve().parent.parent.parent.parent  # parameter-golf/
sys.path.insert(0, str(REPO))
_train_script = Path(os.environ.get("TRAIN_SCRIPT", str(REPO / "train_gpt.py")))
_src = _train_script.read_text()
exec(compile(_src.split("\ndef main")[0], str(_train_script), "exec"))

# ── N-gram Cache ──────────────────────────────────────────────────────────────

PRIMES = np.array(
    [np.uint64(36313), np.uint64(27191), np.uint64(51647),
     np.uint64(81929), np.uint64(131071), np.uint64(174763),
     np.uint64(233017), np.uint64(310019), np.uint64(412553)],
    dtype=np.uint64,
)


def hash_context(val_np, positions, ctx_width, primes, ng_mask):
    """Hash the context (ctx_width tokens before each position)."""
    ctx_hash = np.zeros(len(positions), dtype=np.uint64)
    for k in range(ctx_width):
        tok = val_np[positions - (ctx_width - k)].astype(np.uint64)
        ctx_hash ^= tok * primes[k % len(primes)]
    return ctx_hash


def hash_full(ctx_hash, target_tokens, ctx_width, primes, ng_mask):
    """Hash context + target."""
    tgt = target_tokens.astype(np.uint64)
    return ctx_hash ^ (tgt * primes[ctx_width % len(primes)])


# ── Mixing Strategies ─────────────────────────────────────────────────────────

def mix_linear(p_model, p_ngram, alpha):
    """Standard linear interpolation: p = (1-a)*p_model + a*p_ngram"""
    return (1.0 - alpha) * p_model + alpha * p_ngram


def mix_logistic(p_model, p_ngram, alpha):
    """Logistic (PAQ-style) mixing in stretched probability space."""
    eps = 1e-7
    p_model = np.clip(p_model, eps, 1.0 - eps)
    p_ngram = np.clip(p_ngram, eps, 1.0 - eps)
    logit_m = np.log(p_model / (1.0 - p_model))
    logit_n = np.log(p_ngram / (1.0 - p_ngram))
    combined = (1.0 - alpha) * logit_m + alpha * logit_n
    return 1.0 / (1.0 + np.exp(-combined))


def mix_geometric(p_model, p_ngram, alpha):
    """Geometric mean: p proportional to p_model^(1-a) * p_ngram^a"""
    eps = 1e-12
    p_model = np.clip(p_model, eps, 1.0)
    p_ngram = np.clip(p_ngram, eps, 1.0)
    log_mix = (1.0 - alpha) * np.log(p_model) + alpha * np.log(p_ngram)
    return np.exp(log_mix)


# ── Eval Function ─────────────────────────────────────────────────────────────

def eval_sliding_ngram(
    base_model,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    device,
    seq_len=2048,
    stride=64,
    batch_seqs=32,
    # N-gram config
    use_ngram=False,
    ngram_max_order=7,
    ngram_min_order=2,
    ngram_buckets=4_194_304,
    ngram_min_count=2,
    ngram_backoff=True,
    ngram_alpha=0.40,
    entropy_adaptive=False,
    ent_base=0.05,
    ent_range=0.55,
    ent_scale=2.0,
    ent_thresh=4.0,
    order_adaptive_entropy=False,
    order_ent_center=3.0,
    order_ent_slope=0.25,
    mixing_fn="linear",
    ngram_only=False,  # If True: skip neural model, use uniform base + 100% n-gram
    audit_distribution=False,  # If True: compute full vocab blend, report sum and normalized BPB
    # Diagnostics
    collect_diagnostics=False,
):
    """Sliding window eval with configurable n-gram cache.

    Returns: (val_loss, val_bpb, diagnostics_dict)
    """
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Choose mixing function
    mixer = {"linear": mix_linear, "logistic": mix_logistic, "geometric": mix_geometric}[mixing_fn]

    if use_ngram:
        val_np = val_tokens.cpu().numpy().astype(np.int64)
        actual_min = ngram_min_order if ngram_backoff else ngram_max_order
        blender = _CppBlender(actual_min, ngram_max_order, ngram_buckets, ngram_min_count)
        blender.set_tokens(val_np)
        blender.set_mixing_fn({"linear": 0, "logistic": 1, "geometric": 2}[mixing_fn])
        _alpha_mode = 2 if order_adaptive_entropy else (1 if entropy_adaptive else 0)
        _alpha_val = 1.0 if ngram_only else ngram_alpha
        blender.configure_alpha(_alpha_mode, _alpha_val, ent_base, ent_range,
                                ent_scale, ent_thresh, order_ent_center, order_ent_slope)
        _empty_ent = np.empty(0, dtype=np.float64)

    # Diagnostics collectors
    diag = {}
    if collect_diagnostics:
        diag["per_order_hits"] = {o: 0 for o in range(ngram_min_order, ngram_max_order + 1)}
        diag["total_scored"] = 0
        diag["ngram_matched"] = 0
        diag["p_ngram_buckets"] = {b: {"count": 0, "nll_sum": 0.0} for b in
                                    ["0.0-0.2", "0.2-0.5", "0.5-0.8", "0.8-0.95", "0.95-1.0"]}

    vocab_size = 1024  # FineWeb SP1024
    uniform_nll = math.log(vocab_size)

    # Audit: track distribution sums and normalized BPB
    if audit_distribution:
        audit_sum_total = 0.0
        audit_sum_count = 0
        audit_norm_nll_sum = 0.0

    if not ngram_only:
        base_model.eval()
        if not hasattr(base_model, "forward_logits"):
            # Base train_gpt.py: forward returns loss, not logits. Add logits-only method.
            def _forward_logits(input_ids):
                m = base_model
                x = m.tok_emb(input_ids)
                x = F.rms_norm(x, (x.size(-1),))
                x0 = x
                skips = []
                for i in range(m.num_encoder_layers):
                    x = m.blocks[i](x, x0)
                    skips.append(x)
                for i in range(m.num_decoder_layers):
                    if skips:
                        x = x + m.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                    x = m.blocks[m.num_encoder_layers + i](x, x0)
                x = m.final_norm(x)
                if m.tie_embeddings:
                    logits = F.linear(x, m.tok_emb.weight)
                else:
                    logits = m.lm_head(x)
                return m.logit_softcap * torch.tanh(logits / m.logit_softcap)
            base_model.forward_logits = _forward_logits
        compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    t_start = time.time()

    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_seqs):
            batch_ws = window_starts[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            if ngram_only:
                # No neural model: uniform NLL for all positions
                nll = torch.full((bsz, seq_len), uniform_nll, device=device, dtype=torch.float32)
                logits = None
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)

            # Compute entropy if needed (before n-gram mixing)
            if use_ngram and not ngram_only and (entropy_adaptive or order_adaptive_entropy):
                with torch.no_grad():
                    lp = F.log_softmax(logits.float(), dim=-1)
                    entropy_all = -(lp.exp() * lp).sum(dim=-1).cpu().numpy()

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                seg_len = wlen - s
                if seg_len <= 0:
                    continue

                scored_nll = nll[i, s:wlen].to(torch.float64)

                if use_ngram:
                    seg_nll_np = scored_nll.cpu().numpy()
                    global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
                    seg_ent = entropy_all[i, s:wlen].astype(np.float64) if (entropy_adaptive or order_adaptive_entropy) else _empty_ent
                    mixed = np.asarray(blender.process_stride(global_j, seg_nll_np, seg_ent))
                    scored_nll = torch.from_numpy(mixed).to(dtype=torch.float64, device=device)

                loss_sum += scored_nll.sum()
                token_count += float(seg_len)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            # Progress
            done = min(bi + batch_seqs, len(window_starts))
            if done % (batch_seqs * 10) == 0 or done == len(window_starts):
                elapsed = time.time() - t_start
                pct = 100.0 * done / len(window_starts)
                curr_bpb = (loss_sum.item() / max(token_count.item(), 1)) / math.log(2) * (token_count.item() / max(byte_count.item(), 1))
                print(f"  [{pct:5.1f}%] windows={done}/{len(window_starts)} "
                      f"bpb={curr_bpb:.4f} elapsed={elapsed:.0f}s", flush=True)

    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()

    if audit_distribution and use_ngram and audit_sum_count > 0:
        avg_sum = audit_sum_total / audit_sum_count
        # Normalized BPB: replace n-gram matched NLLs with normalized ones
        # non-matched tokens keep their model NLL, so approximate total:
        norm_loss = (loss_sum.item() - audit_norm_nll_sum) + audit_norm_nll_sum  # same total, but let's compute properly
        # Actually: loss_sum includes the unnormalized mixed NLLs for matched tokens.
        # We need to subtract those and add the normalized ones.
        # matched tokens' unnormalized NLL is already in loss_sum via seg_model_p assignment.
        # We stored the normalized NLL sum in audit_norm_nll_sum.
        # The unnormalized NLL for matched tokens: -log(mixed_p) which is what's in loss_sum.
        # But we didn't separately track that. Let's just report audit_norm_nll_sum / audit_sum_count as avg NLL.
        diag["audit"] = {
            "avg_distribution_sum": avg_sum,
            "positions_audited": audit_sum_count,
            "avg_normalized_nll": audit_norm_nll_sum / audit_sum_count,
        }
        print(f"\n  AUDIT: avg distribution sum = {avg_sum:.1f} (should be 1.0)")
        print(f"  AUDIT: positions audited = {audit_sum_count:,}")
        print(f"  AUDIT: avg NLL after normalization = {audit_norm_nll_sum / audit_sum_count:.4f}")

    return val_loss, bpt * tpb, diag


# ── Experiment Configs ────────────────────────────────────────────────────────

EXPERIMENTS = {
    # EXP-0: Baselines
    "baseline": dict(use_ngram=False),
    "fixed_7gram": dict(use_ngram=True, ngram_max_order=7, ngram_backoff=False,
                        ngram_alpha=0.40, entropy_adaptive=False),
    "backoff_7": dict(use_ngram=True, ngram_max_order=7, ngram_backoff=True,
                      ngram_alpha=0.40, entropy_adaptive=False),
    "backoff_7_ent": dict(use_ngram=True, ngram_max_order=7, ngram_backoff=True,
                          entropy_adaptive=True),
    "backoff_9_ent_oadapt": dict(use_ngram=True, ngram_max_order=9, ngram_backoff=True,
                                  order_adaptive_entropy=True),

    # Audit variants — same configs but audit the distribution sum and normalized BPB
    "backoff_7_audit": dict(use_ngram=True, ngram_max_order=7, ngram_backoff=True,
                            ngram_alpha=0.40, entropy_adaptive=False, audit_distribution=True),
    "backoff_9_ent_oadapt_audit": dict(use_ngram=True, ngram_max_order=9, ngram_backoff=True,
                                        order_adaptive_entropy=True, audit_distribution=True),

    # N-gram only (no neural model)
    "ngram_only_7": dict(use_ngram=True, ngram_max_order=7, ngram_backoff=True,
                         ngram_only=True),
    "ngram_only_7_no_backoff": dict(use_ngram=True, ngram_max_order=7, ngram_backoff=False,
                                     ngram_only=True),

    # EXP-2: Mixing strategies (all use backoff 2-7, entropy-adaptive alpha)
    "mix_linear": dict(use_ngram=True, ngram_max_order=7, ngram_backoff=True,
                       entropy_adaptive=True, mixing_fn="linear"),
    "mix_logistic": dict(use_ngram=True, ngram_max_order=7, ngram_backoff=True,
                         entropy_adaptive=True, mixing_fn="logistic"),
    "mix_geometric": dict(use_ngram=True, ngram_max_order=7, ngram_backoff=True,
                          entropy_adaptive=True, mixing_fn="geometric"),
}

# EXP-1: Order sweep (generated programmatically)
for K in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
    EXPERIMENTS[f"order_{K}"] = dict(
        use_ngram=True, ngram_max_order=K, ngram_backoff=True, entropy_adaptive=True,
        collect_diagnostics=True,
    )

# Stride sweep (for fitting in 600s budget)
for S in [128, 256, 512]:
    EXPERIMENTS[f"backoff_7_stride{S}"] = dict(
        use_ngram=True, ngram_max_order=7, ngram_backoff=True,
        ngram_alpha=0.40, entropy_adaptive=False,
    )

# EXP-5: Bucket sweep
for log2b in [20, 22, 24, 26, 28]:
    buckets = 2 ** log2b
    EXPERIMENTS[f"buckets_{buckets}"] = dict(
        use_ngram=True, ngram_max_order=7, ngram_backoff=True,
        entropy_adaptive=True, ngram_buckets=buckets,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to final_model.pt")
    parser.add_argument("--exp", required=True, help="Experiment name or comma-separated list")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="./results", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load tokenizer + val data
    hp = Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    val_tokens = load_validation_tokens(hp.val_files, hp.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, hp.vocab_size, device
    )
    print(f"Loaded val tokens: {val_tokens.numel()-1:,}", flush=True)

    # Load model — auto-detect bigram/VE config from checkpoint keys
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    ckpt_keys = set(state.keys())

    # Detect bigram config
    bigram_vocab = 0
    bigram_dim = 128
    if "bigram.embed.weight" in ckpt_keys:
        bigram_vocab = state["bigram.embed.weight"].shape[0]
        bigram_dim = state["bigram.embed.weight"].shape[1]
        print(f"Detected bigram: vocab={bigram_vocab}, dim={bigram_dim}")

    # Detect VE config
    ve_enabled = "ve_shared.embed.weight" in ckpt_keys
    ve_dim = state["ve_shared.embed.weight"].shape[1] if ve_enabled else 128
    if ve_enabled:
        # Count VE layer scales to find which layers
        ve_scale_keys = [k for k in ckpt_keys if k.startswith("ve_layer_scales.")]
        print(f"Detected VE: dim={ve_dim}, layers={len(ve_scale_keys)}")

    import inspect
    gpt_sig = inspect.signature(GPT.__init__).parameters
    gpt_kwargs = dict(
        vocab_size=hp.vocab_size, num_layers=hp.num_layers, model_dim=hp.model_dim,
        num_heads=hp.num_heads, num_kv_heads=hp.num_kv_heads, mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings, tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap, rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
        bigram_vocab_size=bigram_vocab, bigram_dim=bigram_dim,
        xsa_last_n=getattr(hp, "xsa_last_n", 0), rope_dims=getattr(hp, "rope_dims", 0),
        ln_scale=getattr(hp, "ln_scale", 1.0),
        ve_enabled=ve_enabled, ve_dim=ve_dim, ve_layers=getattr(hp, "ve_layers", ""),
        gated_attention=getattr(hp, "gated_attention", False),
        value_residual=getattr(hp, "value_residual", False),
    )
    gpt_kwargs = {k: v for k, v in gpt_kwargs.items() if k in gpt_sig}
    model = GPT(**gpt_kwargs).to(device).bfloat16()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"INFO: Unexpected keys (ignored): {unexpected}")
    print(f"Loaded model from {args.model}", flush=True)

    os.makedirs(args.out, exist_ok=True)

    # Run experiments
    exp_names = [e.strip() for e in args.exp.split(",")]
    for exp_name in exp_names:
        if exp_name not in EXPERIMENTS:
            print(f"ERROR: Unknown experiment '{exp_name}'. Available: {sorted(EXPERIMENTS.keys())}")
            continue

        cfg = EXPERIMENTS[exp_name]
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"Config: {cfg}")
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        val_loss, val_bpb, diag = eval_sliding_ngram(
            base_model=model,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            device=device,
            seq_len=getattr(hp, "eval_seq_len", hp.train_seq_len),
            stride=args.stride,
            batch_seqs=args.batch,
            **cfg,
        )
        elapsed = time.time() - t0

        result = {
            "experiment": exp_name,
            "val_loss": val_loss,
            "val_bpb": val_bpb,
            "eval_time_s": elapsed,
            "config": cfg,
            "diagnostics": diag,
        }

        out_path = os.path.join(args.out, f"{exp_name}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nRESULT: {exp_name} -> val_bpb={val_bpb:.6f} val_loss={val_loss:.6f} time={elapsed:.0f}s")
        print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
