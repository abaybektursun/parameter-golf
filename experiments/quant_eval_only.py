"""Quant-eval-only: loads pre-trained checkpoint, runs selfgen GPTQ calibration + eval.
Skips all training. Tests different selfgen configs on the same trained weights."""
from __future__ import annotations
import io, lzma, math, os, sys, time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# Import everything from the training script
sys.path.insert(0, "/root/parameter-golf")
from train_selfgen_v3 import (
    Hyperparameters, GPT, _HessianGPT, CastedLinear,
    _unbank_state_dict, _rebank_state_dict,
    mixed_quantize_int6, dequantize_mixed_int6,
    collect_hessians, collect_hessians_selfgen,
    eval_val, eval_val_sliding,
    load_validation_tokens, build_sentencepiece_luts,
    DistributedTokenLoader, restore_low_dim_params_to_fp32,
)
import sentencepiece as spm

def main():
    args = Hyperparameters()
    code = Path("/root/parameter-golf/train_selfgen_v3.py").read_text(encoding="utf-8")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def log0(msg, console=True):
        if master_process and console:
            print(msg, flush=True)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Load pre-trained checkpoint
    log0(f"Loading checkpoint: final_model.pt")
    export_sd = torch.load("/root/parameter-golf/final_model.pt", map_location="cpu")

    # Build banked model for Gibbs generation (if selfgen)
    banked_model = None
    if args.selfgen_gptq:
        banked_model = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            mtp_num_heads=0, mtp_loss_weight=0.0,
            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
            xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
            ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        ).to(device).bfloat16()
        banked_model.load_state_dict({k: v.to(device) for k, v in export_sd.items()}, strict=False)
        log0("Loaded banked model for Gibbs generation")

    # Unbank
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)

    # Build hessian model
    hessian_model = _HessianGPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in hessian_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(hessian_model)
    hessian_model.load_state_dict(
        {k: v.to(device) for k, v in unbanked_sd.items() if k in hessian_model.state_dict()},
        strict=False,
    )

    # Collect Hessians
    if args.selfgen_gptq:
        log0(f"selfgen: batches={args.selfgen_batches} bs={args.selfgen_batch_size} "
             f"gibbs={args.selfgen_gibbs_rounds} temp={args.selfgen_temperature}")
        hessians, total_t, gibbs_t, hessian_t, total_tok = collect_hessians_selfgen(
            hessian_model, banked_model, args, device,
            num_batches=args.selfgen_batches, batch_size=args.selfgen_batch_size,
            gibbs_rounds=args.selfgen_gibbs_rounds, temperature=args.selfgen_temperature,
            seed=args.seed)
        log0(f"selfgen: {len(hessians)} layers, {total_tok} tokens, "
             f"gibbs={gibbs_t:.1f}s hessian={hessian_t:.1f}s total={total_t:.1f}s")
        del banked_model
    else:
        log0(f"val-calibrated: {args.gptq_calib_batches} batches")
        calib_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)
        hessians = collect_hessians(hessian_model, calib_loader, args, device, grad_accum_steps,
                                    num_batches=args.gptq_calib_batches)
        log0(f"val-calibrated: {len(hessians)} layers")

    del hessian_model
    torch.cuda.empty_cache()

    # Quantize
    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians)

    # Selective pruning
    target_mb = float(os.environ.get("TARGET_MB", "15.9"))
    code_bytes_est = len(code.encode("utf-8"))
    ones_info = []
    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type") == "int6"): continue
        qk, sk = name + ".q", name + ".scale"
        if qk not in quant_result or sk not in quant_result: continue
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
            buf = io.BytesIO(); torch.save({"w": tmp, "m": quant_meta}, buf)
            return len(lzma.compress(buf.getvalue(), preset=9)) + code_bytes_est, tmp
        no_sz, _ = _try_prune(0)
        target_bytes = int(target_mb * 1024 * 1024)
        log0(f"prune: {len(ones_info)} candidates, unpruned={no_sz/(1024*1024):.2f}MB")
        if no_sz <= target_bytes:
            log0("prune: fits, no pruning needed")
        else:
            full_sz, _ = _try_prune(len(ones_info))
            if full_sz > target_bytes:
                _, quant_result = _try_prune(len(ones_info))
            else:
                lo, hi = 0, len(ones_info)
                while lo < hi:
                    mid = (lo + hi) // 2
                    sz, _ = _try_prune(mid)
                    if sz <= target_bytes: hi = mid
                    else: lo = mid + 1
                log0(f"prune: {lo}/{len(ones_info)} values")
                _, quant_result = _try_prune(lo)

    # Serialize
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = lzma.compress(quant_buf.getvalue(), preset=9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        log0(f"artifact: {len(quant_blob)} bytes + code {code_bytes_est} = {len(quant_blob)+code_bytes_est}")
    if distributed:
        dist.barrier()

    # Load and eval
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob_disk)), map_location="cpu")
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)

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

    # Roundtrip eval
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(f"roundtrip val_bpb:{q_val_bpb:.4f} time:{1000*(time.perf_counter()-t0):.0f}ms")

    # Sliding window eval
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(f"sliding val_bpb:{sw_bpb:.4f} stride:{args.eval_stride} time:{1000*(time.perf_counter()-t0):.0f}ms")
        log0(f"sliding_exact val_bpb:{sw_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
