"""Test mixed int6/int7 bit allocation using Hessian sensitivity.
Re-quantize the trained model with int7 for most-sensitive layers, measure BPB + artifact size."""
import sys, os, io, math, glob, time
import numpy as np
import brotli
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, '/root/parameter-golf')
import importlib.util
spec = importlib.util.spec_from_file_location("train_mod", "/root/parameter-golf/train_gpt_turbo_muon.py")
train_mod = importlib.util.module_from_spec(spec)
train_mod.__name__ = "train_mod"
spec.loader.exec_module(train_mod)

device = torch.device("cuda", 0)
args = train_mod.Hyperparameters()
args.bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
args.bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))
args.xsa_last_n = 11

# Load tokenizer + LUTs
sp = spm.SentencePieceProcessor(args.tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = train_mod.build_sentencepiece_luts(sp, args.vocab_size, device)

# Load val tokens
val_files = sorted(glob.glob(args.val_files))
from pathlib import Path
val_tokens_list = []
for vf in val_files:
    header = np.fromfile(vf, dtype=np.dtype("<i4"), count=256)
    num_tokens = int(header[2])
    tokens = np.memmap(vf, mode='r', dtype=np.dtype("<u2"), offset=1024, shape=(num_tokens,))
    val_tokens_list.append(torch.from_numpy(np.array(tokens)))
val_tokens = torch.cat(val_tokens_list).to(dtype=torch.int64, device=device)
print(f"val_tokens: {val_tokens.numel()}")

# Build model
model = train_mod.GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
    mlp_mult=int(args.mlp_mult), tie_embeddings=args.tie_embeddings,
    tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
    rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
).to(device)

# Load weights
print("Loading model...")
state = torch.load("/root/parameter-golf/final_model.pt", map_location=device, weights_only=False)
model.load_state_dict(state, strict=False)
model.to(torch.bfloat16)
train_mod.restore_low_dim_params_to_fp32(model)

# Unbank for GPTQ
sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
unbanked_sd = train_mod._unbank_state_dict(sd_cpu, args.num_layers)

# AR generation uses banked model; Hessian collection uses unbanked model
print("Generating AR calibration data...")
ar_tokens = train_mod.generate_autoregressive_calib(
    model, device, num_seqs=64, seq_len=2048,
    vocab_size=args.vocab_size, temperature=0.8, seed=42)
print(f"Generated {len(ar_tokens)} sequences")

print("Building Hessian model...")
hessian_model = train_mod._HessianGPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
    mlp_mult=int(args.mlp_mult), tie_embeddings=args.tie_embeddings,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
).to(device).bfloat16()
hessian_model.load_state_dict(unbanked_sd, strict=False)
train_mod.restore_low_dim_params_to_fp32(hessian_model)

print("Collecting Hessians...")
hessians = train_mod.collect_hessians_from_tokens(hessian_model, ar_tokens, device)
print(f"Collected hessians for {len(hessians)} layers")
del hessian_model, ar_tokens
torch.cuda.empty_cache()

# Rank layers by Hessian trace (sensitivity)
print("\n=== Layer Sensitivity (Hessian trace) ===")
sensitivity = {}
for name, H in hessians.items():
    tr = H.diag().sum().item()
    sensitivity[name] = tr

# Sort by sensitivity
ranked = sorted(sensitivity.items(), key=lambda x: -x[1])
for name, tr in ranked[:20]:
    cat = "MLP" if "mlp" in name else "ATN" if "attn" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name or "out_proj" in name else "OTH"
    print(f"  {cat} {name}: trace={tr:.1f}")

# Helper: quantize with specific clip_range per layer
def mixed_quantize(state_dict, hessians, bit_allocation):
    """bit_allocation: dict mapping layer name -> clip_range (31=int6, 63=int7)"""
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in train_mod.CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        cat = train_mod._classify_param(name)
        if cat in {"mlp", "attn"} and t.ndim >= 1:
            cr = bit_allocation.get(name, 31)  # default int6
            H = hessians.get(name)
            if H is not None:
                q, s = train_mod.quantize_int6_gptq(t, hessian=H, clip_range=cr)
            else:
                q, s = train_mod.quantize_int6_per_row(t, clip_range=cr)
            bit_label = "int7" if cr == 63 else "int6"
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": bit_label}
        else:
            q, s = train_mod.quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta

def measure_artifact(result, meta):
    buf = io.BytesIO()
    torch.save({"w": result, "m": meta}, buf)
    raw = buf.getvalue()
    compressed = brotli.compress(raw, quality=11)
    return len(compressed)

def eval_roundtrip(result, meta, unbanked_sd):
    deq = train_mod.dequantize_mixed_int6(result, meta, unbanked_sd)
    rebanked = train_mod._rebank_state_dict(deq, args.num_layers, sd_cpu)
    model.load_state_dict(rebanked, strict=False)
    model.to(torch.bfloat16)
    train_mod.restore_low_dim_params_to_fp32(model)
    model.eval()
    # Quick eval (chunked, not sliding window)
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    seq_len = 2048
    num_seqs = (val_tokens.numel() - 1) // seq_len
    with torch.inference_mode():
        for i in range(0, num_seqs, 32):
            bsz = min(32, num_seqs - i)
            x_list, y_list = [], []
            for j in range(i, i + bsz):
                s = j * seq_len
                x_list.append(val_tokens[s:s+seq_len])
                y_list.append(val_tokens[s+1:s+seq_len+1])
            x = torch.stack(x_list).to(device)
            y = torch.stack(y_list).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            total_loss += loss.item() * bsz * seq_len
            total_tokens += bsz * seq_len
            token_bytes = base_bytes_lut[y] + (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).float()
            total_bytes += token_bytes.sum().item()
    val_loss = total_loss / total_tokens
    bpb = (val_loss / math.log(2)) * (total_tokens / total_bytes)
    return val_loss, bpb

# --- Test configurations ---
code_bytes = 127280

print("\n=== Configuration 1: Uniform int6 (baseline) ===")
alloc_int6 = {name: 31 for name in hessians}
r6, m6 = mixed_quantize(unbanked_sd, hessians, alloc_int6)
sz6 = measure_artifact(r6, m6)
print(f"  Artifact: {sz6/1024/1024:.2f} MB, total: {(sz6+code_bytes)/1024/1024:.2f} MB")
loss6, bpb6 = eval_roundtrip(r6, m6, unbanked_sd)
print(f"  val_loss: {loss6:.4f}  val_bpb: {bpb6:.4f}")

print("\n=== Configuration 2: int7 for top-10 most sensitive layers ===")
top10 = [name for name, _ in ranked[:10]]
alloc_top10 = {name: (63 if name in top10 else 31) for name in hessians}
print(f"  int7 layers: {[n for n in top10]}")
r7_10, m7_10 = mixed_quantize(unbanked_sd, hessians, alloc_top10)
sz7_10 = measure_artifact(r7_10, m7_10)
print(f"  Artifact: {sz7_10/1024/1024:.2f} MB, total: {(sz7_10+code_bytes)/1024/1024:.2f} MB")
loss7_10, bpb7_10 = eval_roundtrip(r7_10, m7_10, unbanked_sd)
print(f"  val_loss: {loss7_10:.4f}  val_bpb: {bpb7_10:.4f}")
print(f"  Delta vs int6: {bpb7_10 - bpb6:+.4f} BPB, {(sz7_10-sz6)/1024:+.1f} KB")

print("\n=== Configuration 3: int7 for ALL MLP layers ===")
alloc_mlp7 = {name: (63 if "mlp" in name else 31) for name in hessians}
r_mlp7, m_mlp7 = mixed_quantize(unbanked_sd, hessians, alloc_mlp7)
sz_mlp7 = measure_artifact(r_mlp7, m_mlp7)
print(f"  Artifact: {sz_mlp7/1024/1024:.2f} MB, total: {(sz_mlp7+code_bytes)/1024/1024:.2f} MB")
loss_mlp7, bpb_mlp7 = eval_roundtrip(r_mlp7, m_mlp7, unbanked_sd)
print(f"  val_loss: {loss_mlp7:.4f}  val_bpb: {bpb_mlp7:.4f}")
print(f"  Delta vs int6: {bpb_mlp7 - bpb6:+.4f} BPB, {(sz_mlp7-sz6)/1024:+.1f} KB")

print("\n=== Configuration 4: int7 for ALL layers ===")
alloc_all7 = {name: 63 for name in hessians}
r_all7, m_all7 = mixed_quantize(unbanked_sd, hessians, alloc_all7)
sz_all7 = measure_artifact(r_all7, m_all7)
print(f"  Artifact: {sz_all7/1024/1024:.2f} MB, total: {(sz_all7+code_bytes)/1024/1024:.2f} MB")
loss_all7, bpb_all7 = eval_roundtrip(r_all7, m_all7, unbanked_sd)
print(f"  val_loss: {loss_all7:.4f}  val_bpb: {bpb_all7:.4f}")
print(f"  Delta vs int6: {bpb_all7 - bpb6:+.4f} BPB, {(sz_all7-sz6)/1024:+.1f} KB")

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY (16MB budget = {16*1024*1024} bytes)")
print(f"{'='*60}")
configs = [
    ("Uniform int6", sz6, bpb6),
    ("Top-10 int7", sz7_10, bpb7_10),
    ("All MLP int7", sz_mlp7, bpb_mlp7),
    ("All int7", sz_all7, bpb_all7),
]
for name, sz, bpb in configs:
    total = sz + code_bytes
    fits = "OK" if total <= 16*1024*1024 else "OVER"
    print(f"  {name:15s}: {total/1024/1024:.2f} MB [{fits}]  BPB={bpb:.4f}  delta={bpb-bpb6:+.4f}")
