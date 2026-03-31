"""Test 2:4 structured sparsity on trained model.
Post-training prune MLP weights to 2:4 pattern, measure BPB degradation."""
import sys
import os
import io
import math
import glob
import torch
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm

# Import model definition from our turbo-muon script
sys.path.insert(0, '/root/parameter-golf')
import importlib.util
spec = importlib.util.spec_from_file_location("train_mod", "/root/parameter-golf/train_gpt_turbo_muon.py")
train_mod = importlib.util.module_from_spec(spec)
# Patch __name__ so if __name__ == "__main__" guard prevents main() from running
train_mod.__name__ = "train_mod"
spec.loader.exec_module(train_mod)

Hyperparameters = train_mod.Hyperparameters
GPT = train_mod.GPT
dequantize_state_dict_int8 = train_mod.dequantize_state_dict_int8
build_sentencepiece_luts = train_mod.build_sentencepiece_luts

device = torch.device("cuda", 0)


def apply_24_sparsity(tensor):
    """Apply 2:4 structured sparsity: for every 4 consecutive elements, zero the 2 smallest."""
    if tensor.ndim != 2:
        return tensor
    t = tensor.clone()
    M, N = t.shape
    # Pad N to multiple of 4 if needed
    pad = (4 - N % 4) % 4
    if pad > 0:
        t_padded = F.pad(t, (0, pad))
    else:
        t_padded = t
    # Reshape to groups of 4
    t_groups = t_padded.reshape(M, -1, 4)
    # Find the 2 smallest magnitude indices per group
    _, indices = t_groups.abs().topk(2, dim=-1, largest=False)
    # Zero them out
    mask = torch.ones_like(t_groups, dtype=torch.bool)
    mask.scatter_(-1, indices, False)
    t_groups = t_groups * mask
    # Reshape back
    t_out = t_groups.reshape(M, -1)
    if pad > 0:
        t_out = t_out[:, :N]
    return t_out


def count_sparsity(state_dict):
    """Count total zeros and params in 2D tensors."""
    total = 0
    zeros = 0
    for name, t in state_dict.items():
        if t.ndim == 2:
            total += t.numel()
            zeros += (t == 0).sum().item()
    return zeros, total


def eval_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, seq_len=2048):
    """Simple single-GPU BPB evaluation."""
    model.eval()
    total_tokens = val_tokens.numel() - 1
    num_seqs = total_tokens // seq_len
    val_loss_sum = 0.0
    val_token_count = 0
    val_byte_count = 0.0

    with torch.inference_mode():
        for i in range(0, num_seqs, 32):  # batch of 32 sequences
            batch_end = min(i + 32, num_seqs)
            bsz = batch_end - i
            x_list = []
            y_list = []
            for j in range(i, batch_end):
                start = j * seq_len
                x_list.append(val_tokens[start:start + seq_len])
                y_list.append(val_tokens[start + 1:start + seq_len + 1])
            x = torch.stack(x_list).to(device)
            y = torch.stack(y_list).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)

            val_loss_sum += loss.item() * bsz * seq_len
            val_token_count += bsz * seq_len

            # Byte counting
            tgt_ids = y
            prev_ids = x
            token_bytes = base_bytes_lut[tgt_ids]
            token_bytes = token_bytes + (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).float()
            val_byte_count += token_bytes.sum().item()

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss / math.log(2)
    tokens_per_byte = val_token_count / val_byte_count
    bpb = bits_per_token * tokens_per_byte
    return val_loss, bpb


def main():
    args = Hyperparameters()
    args.bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
    args.bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))
    args.xsa_last_n = 11

    # Load tokenizer and build LUTs
    sp = spm.SentencePieceProcessor(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device)

    # Load validation tokens
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
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        mlp_mult=int(args.mlp_mult), tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device)

    # Load trained weights
    print("Loading trained model...")
    state = torch.load("/root/parameter-golf/final_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(torch.bfloat16)
    train_mod.restore_low_dim_params_to_fp32(model)

    # Eval baseline (no sparsity)
    print("\n=== BASELINE (dense) ===")
    val_loss, val_bpb = eval_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"val_loss: {val_loss:.4f}  val_bpb: {val_bpb:.4f}")
    baseline_bpb = val_bpb

    # Apply 2:4 sparsity to all bank weights (MLP and attention)
    print("\n=== APPLYING 2:4 SPARSITY ===")
    sd = model.state_dict()
    pruned_count = 0
    for name in list(sd.keys()):
        if any(k in name for k in ['qo_bank', 'kv_bank', 'mlp_up_bank', 'mlp_down_bank']):
            original = sd[name]
            pruned = apply_24_sparsity(original.reshape(-1, original.shape[-1]))
            sd[name] = pruned.reshape(original.shape)
            pruned_count += 1
            z = (sd[name] == 0).sum().item()
            t = sd[name].numel()
            print(f"  {name}: {z}/{t} zeros ({100*z/t:.1f}%)")

    model.load_state_dict(sd, strict=False)
    print(f"\nPruned {pruned_count} bank tensors to 2:4 pattern")

    zeros, total = count_sparsity(sd)
    print(f"Total 2D sparsity: {zeros}/{total} ({100*zeros/total:.1f}%)")

    # Eval after sparsity
    print("\n=== AFTER 2:4 SPARSITY ===")
    val_loss, val_bpb = eval_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"val_loss: {val_loss:.4f}  val_bpb: {val_bpb:.4f}")
    print(f"BPB degradation: {val_bpb - baseline_bpb:+.4f}")

    # Also test MLP-only sparsity (less aggressive)
    print("\n=== MLP-ONLY 2:4 SPARSITY ===")
    sd2 = torch.load("/root/parameter-golf/final_model.pt", map_location=device, weights_only=False)
    for name in list(sd2.keys()):
        if any(k in name for k in ['mlp_up_bank', 'mlp_down_bank']):
            original = sd2[name]
            sd2[name] = apply_24_sparsity(original.reshape(-1, original.shape[-1])).reshape(original.shape)
    model.load_state_dict(sd2, strict=False)
    model.to(torch.bfloat16)
    train_mod.restore_low_dim_params_to_fp32(model)

    val_loss, val_bpb = eval_bpb(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"val_loss: {val_loss:.4f}  val_bpb: {val_bpb:.4f}")
    print(f"BPB degradation (MLP-only): {val_bpb - baseline_bpb:+.4f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"  Baseline BPB:           {baseline_bpb:.4f}")
    print(f"  All-banks 2:4 BPB:      {val_bpb:.4f} (need to re-eval)")
    print(f"  Threshold for viability: < +0.02 BPB degradation")


if __name__ == "__main__":
    main()
