"""Answer Q1: Does autoregressive self-generation close the 0.0020 GPTQ calibration gap?

Loads pre-trained checkpoint, runs GPTQ with 3 calibration methods on the same weights:
1. Val data (control — should reproduce ~1.1145)
2. Autoregressive self-generation (the missing v1 experiment)
3. Random tokens (control — should reproduce ~1.1165)

Reports sliding BPB for each.

Usage: torchrun --standalone --nproc_per_node=8 answer_q1.py
"""
import io, lzma, math, os, sys, time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
import sentencepiece as spm

sys.path.insert(0, "/root/parameter-golf")
from train_609_val_calib import (
    Hyperparameters, GPT, _HessianGPT, CastedLinear,
    _unbank_state_dict, _rebank_state_dict,
    mixed_quantize_int6, dequantize_mixed_int6,
    collect_hessians,
    eval_val_sliding,
    load_validation_tokens, build_sentencepiece_luts,
    DistributedTokenLoader, restore_low_dim_params_to_fp32,
)


def generate_autoregressive(model, device, num_seqs=64, seq_len=2048,
                            vocab_size=1024, temperature=0.8, batch_size=8, seed=42):
    """Generate sequences autoregressively from the model's learned distribution."""
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - batch_start)
            tokens = torch.randint(0, vocab_size, (bs, 1), device=device, generator=rng)
            for pos in range(seq_len - 1):
                logits = model.forward_logits(tokens)
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens = torch.cat([tokens, next_tok], dim=1)
                if pos % 256 == 0:
                    print(f"  gen batch {batch_start//batch_size+1}/{num_seqs//batch_size} pos {pos}/{seq_len-1}", flush=True)
            for i in range(bs):
                all_tokens.append(tokens[i:i+1])
    return all_tokens


def collect_hessians_from_tokens(hessian_model, token_seqs, device):
    """Collect H = X^T X from pre-generated token sequences."""
    hessians = {}
    hooks = []
    for name, module in hessian_model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')
            def make_hook(pname):
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pname] += (x.T @ x).cpu()
                return hook_fn
            h = module.register_forward_hook(make_hook(param_name))
            hooks.append(h)
    hessian_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in token_seqs:
            x = seq[:, :-1].to(device)
            y = seq[:, 1:].to(device)
            hessian_model(x, y)
    for h in hooks:
        h.remove()
    num_batches = len(token_seqs)
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    return hessians


def collect_hessians_random(hessian_model, args, device, num_batches=64, seed=42):
    """Collect H = X^T X from random token forward passes."""
    hessians = {}
    hooks = []
    for name, module in hessian_model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')
            def make_hook(pname):
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pname] += (x.T @ x).cpu()
                return hook_fn
            h = module.register_forward_hook(make_hook(param_name))
            hooks.append(h)
    hessian_model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(num_batches):
            x = torch.randint(0, args.vocab_size, (1, args.train_seq_len), device=device, generator=rng)
            y = torch.randint(0, args.vocab_size, (1, args.train_seq_len), device=device, generator=rng)
            hessian_model(x, y)
    for h in hooks:
        h.remove()
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    return hessians


def quantize_prune_eval(hessians, unbanked_sd, sd_cpu, args, device, rank, world_size,
                        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                        code, label):
    """Quantize with given Hessians → selective prune → serialize → dequantize → sliding eval."""
    master = (rank == 0)
    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians)

    # Selective ±1 pruning (same logic as production)
    target_mb = float(os.environ.get("TARGET_MB", "15.9"))
    code_bytes_est = len(code.encode("utf-8"))
    ones_info = []
    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type") == "int6"):
            continue
        qk, sk = name + ".q", name + ".scale"
        if qk not in quant_result or sk not in quant_result:
            continue
        q, s = quant_result[qk], quant_result[sk]
        if s.ndim > 0:
            ones_mask = (q.abs() == 1)
            if ones_mask.any():
                row_idx = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
                flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
                errors = s.float()[row_idx].pow(2)
                for fi, err in zip(flat_idx.tolist(), errors.tolist()):
                    ones_info.append((qk, fi, err))
    if ones_info:
        ones_info.sort(key=lambda x: x[2])
        def _try_prune(n):
            tmp = {k: v.clone() for k, v in quant_result.items()}
            for i in range(min(n, len(ones_info))):
                tmp[ones_info[i][0]].view(-1)[ones_info[i][1]] = 0
            buf = io.BytesIO()
            torch.save({"w": tmp, "m": quant_meta}, buf)
            return len(lzma.compress(buf.getvalue(), preset=9)) + code_bytes_est, tmp
        no_sz, _ = _try_prune(0)
        target_bytes = int(target_mb * 1024 * 1024)
        if no_sz > target_bytes:
            full_sz, _ = _try_prune(len(ones_info))
            if full_sz > target_bytes:
                _, quant_result = _try_prune(len(ones_info))
            else:
                lo, hi = 0, len(ones_info)
                while lo < hi:
                    mid = (lo + hi) // 2
                    sz, _ = _try_prune(mid)
                    if sz <= target_bytes:
                        hi = mid
                    else:
                        lo = mid + 1
                _, quant_result = _try_prune(lo)

    # Serialize → decompress → dequantize
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = lzma.compress(quant_buf.getvalue(), preset=9)
    artifact_size = len(quant_blob) + code_bytes_est
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)

    # Build eval model
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)

    # Sliding window eval
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sw_loss, sw_bpb = eval_val_sliding(
        args, eval_model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.eval_stride, eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    if master:
        print(f"[{label}] sliding_bpb={sw_bpb:.5f}  artifact={artifact_size}  eval_time={time.perf_counter()-t0:.1f}s", flush=True)

    del eval_model, deq_state, deq_unbanked, quant_result, quant_meta
    torch.cuda.empty_cache()
    return sw_bpb


def main():
    args = Hyperparameters()
    code = Path("/root/parameter-golf/train_609_val_calib.py").read_text(encoding="utf-8")

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, 8 // world_size)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = (rank == 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def log0(msg):
        if master:
            print(msg, flush=True)

    # Load tokenizer & val tokens
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Load checkpoint
    log0("Loading final_model.pt...")
    export_sd = torch.load("/root/parameter-golf/final_model.pt", map_location="cpu")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)

    # ── STEP 1: Generate autoregressive tokens on rank 0 ──
    log0("Generating autoregressive tokens (64 seqs × 2048 tokens, temp=0.8)...")
    ar_path = "/tmp/ar_tokens.pt"
    if master:
        gen_model = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            mtp_num_heads=0, mtp_loss_weight=0.0,
            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
            xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
            ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        ).to(device).bfloat16()
        gen_model.load_state_dict({k: v.to(device) for k, v in export_sd.items()}, strict=False)
        t0 = time.perf_counter()
        ar_tokens_list = generate_autoregressive(
            gen_model, device, num_seqs=64, seq_len=2048,
            vocab_size=args.vocab_size, temperature=0.8, batch_size=8, seed=42,
        )
        gen_time = time.perf_counter() - t0
        log0(f"Generated {len(ar_tokens_list)} sequences in {gen_time:.1f}s")
        ar_stacked = torch.cat(ar_tokens_list, dim=0)  # [64, 2048]
        torch.save(ar_stacked, ar_path)
        del gen_model
        torch.cuda.empty_cache()
    if distributed:
        dist.barrier()
    ar_stacked = torch.load(ar_path, map_location="cpu")
    ar_tokens = [ar_stacked[i:i+1] for i in range(ar_stacked.shape[0])]

    # ── Helper: build fresh hessian model ──
    def build_hessian_model():
        hm = _HessianGPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, logit_softcap=args.logit_softcap,
            rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
            xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
            ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        ).to(device).bfloat16()
        for m in hm.modules():
            if isinstance(m, CastedLinear):
                m.float()
        restore_low_dim_params_to_fp32(hm)
        hm.load_state_dict(
            {k: v.to(device) for k, v in unbanked_sd.items() if k in hm.state_dict()},
            strict=False,
        )
        return hm

    eval_args = (unbanked_sd, sd_cpu, args, device, rank, world_size,
                 val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, code)

    # ── EXPERIMENT 1: Val-calibrated (control) ──
    log0("\n" + "="*60)
    log0("EXPERIMENT 1: Val-calibrated GPTQ")
    log0("="*60)
    hm = build_hessian_model()
    t0 = time.perf_counter()
    calib_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)
    h_val = collect_hessians(hm, calib_loader, args, device, grad_accum_steps, num_batches=64)
    log0(f"Hessians collected: {len(h_val)} layers, {time.perf_counter()-t0:.1f}s")
    del hm; torch.cuda.empty_cache()
    bpb_val = quantize_prune_eval(h_val, *eval_args, "VAL-CALIB")
    del h_val

    # ── EXPERIMENT 2: Autoregressive self-generation ──
    log0("\n" + "="*60)
    log0("EXPERIMENT 2: Autoregressive self-generated GPTQ")
    log0("="*60)
    hm = build_hessian_model()
    t0 = time.perf_counter()
    h_ar = collect_hessians_from_tokens(hm, ar_tokens, device)
    log0(f"Hessians collected: {len(h_ar)} layers, {time.perf_counter()-t0:.1f}s")
    del hm; torch.cuda.empty_cache()
    bpb_ar = quantize_prune_eval(h_ar, *eval_args, "AUTOREGRESSIVE")
    del h_ar

    # ── EXPERIMENT 3: Random tokens (control) ──
    log0("\n" + "="*60)
    log0("EXPERIMENT 3: Random-token GPTQ")
    log0("="*60)
    hm = build_hessian_model()
    t0 = time.perf_counter()
    h_rand = collect_hessians_random(hm, args, device, num_batches=64, seed=42)
    log0(f"Hessians collected: {len(h_rand)} layers, {time.perf_counter()-t0:.1f}s")
    del hm; torch.cuda.empty_cache()
    bpb_rand = quantize_prune_eval(h_rand, *eval_args, "RANDOM")
    del h_rand

    # ── Summary ──
    if master:
        print("\n" + "="*60, flush=True)
        print("RESULTS", flush=True)
        print("="*60, flush=True)
        print(f"Val-calibrated:      {bpb_val:.5f}", flush=True)
        print(f"Autoregressive:      {bpb_ar:.5f}", flush=True)
        print(f"Random tokens:       {bpb_rand:.5f}", flush=True)
        print(f"", flush=True)
        print(f"Gap (val - AR):      {bpb_val - bpb_ar:+.5f}", flush=True)
        print(f"Gap (val - random):  {bpb_val - bpb_rand:+.5f}", flush=True)
        print(f"Gap (AR - random):   {bpb_ar - bpb_rand:+.5f}", flush=True)
        print("="*60, flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
