#!/usr/bin/env python3
"""Patch train_609_val_calib.py → train_quant_exp.py with Qronos iterative Hessian + CDQuant refinement."""

with open("/root/parameter-golf/train_609_val_calib.py") as f:
    lines = f.readlines()

code = "".join(lines)

# --- Patch 1: Add env vars to Hyperparameters ---
code = code.replace(
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n',
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n'
    '    qronos_iters = int(os.environ.get("QRONOS_ITERS", "1"))  # >1 = iterative Hessian re-collection\n'
    '    cdquant_passes = int(os.environ.get("CDQUANT_PASSES", "0"))  # >0 = coordinate descent refinement\n',
    1
)

# --- Patch 2: Add cdquant_refine function after quantize_int6_gptq ---
CDQUANT_FN = r'''
def cdquant_refine(weight, Q, scale, hessian, clip_range=31, num_passes=3):
    """CDQuant-style coordinate descent refinement after GPTQ.
    Re-quantizes each column optimally given all other columns' current quantized state."""
    W = weight.float()
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(H.shape[0]), torch.arange(H.shape[0])] += damp
    S = scale.float()
    Q_work = Q.clone()
    rows, cols = W.shape
    for pass_idx in range(num_passes):
        changed = 0
        res = W - Q_work.float() * S[:, None]
        RH = res @ H  # precompute [rows, cols]
        for j in range(cols):
            target = W[:, j] + (RH[:, j] - res[:, j] * H[j, j]) / H[j, j]
            q_new = torch.clamp(torch.round(target / S), -clip_range, clip_range).to(torch.int8)
            if not torch.equal(q_new, Q_work[:, j]):
                old_res_j = res[:, j].clone()
                Q_work[:, j] = q_new
                new_res_j = W[:, j] - q_new.float() * S
                delta_res = new_res_j - old_res_j
                res[:, j] = new_res_j
                RH += delta_res.unsqueeze(1) * H[j, :].unsqueeze(0)  # rank-1 update
                changed += 1
        if changed == 0:
            break
    return Q_work, scale

'''

code = code.replace(
    '\ndef _quantize_int6_percentile',
    CDQUANT_FN + 'def _quantize_int6_percentile',
    1
)

# --- Patch 3: Modify mixed_quantize_int6 to accept cdquant_passes ---
code = code.replace(
    'def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str], hessians: dict[str, Tensor] | None = None):',
    'def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str], hessians: dict[str, Tensor] | None = None, cdquant_passes: int = 0):',
    1
)

# --- Patch 4: Add CDQuant refinement after GPTQ quantization in mixed_quantize_int6 ---
code = code.replace(
    '            result[name + ".q"] = q\n'
    '            result[name + ".scale"] = s\n'
    '            meta[name] = {"type": "int6"}',

    '            if cdquant_passes > 0 and H is not None and t.ndim == 2:\n'
    '                q, s = cdquant_refine(t, q, s, H, clip_range=cr, num_passes=cdquant_passes)\n'
    '            result[name + ".q"] = q\n'
    '            result[name + ".scale"] = s\n'
    '            meta[name] = {"type": "int6"}',
    1
)

# --- Patch 5: Replace GPTQ section in main() with iterative version ---
OLD_GPTQ_BLOCK = (
    '    # Unbank 3D tensors into individual 2D tensors for quantization\n'
    '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
    '    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)\n'
    '    # Full GPTQ: collect Hessians via a temporary non-banked model\n'
    '    log0(f"gptq:building non-banked model for Hessian collection...")\n'
    '    hessian_model = _HessianGPT(\n'
    '        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,\n'
    '        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,\n'
    '        tie_embeddings=args.tie_embeddings, logit_softcap=args.logit_softcap,\n'
    '        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,\n'
    '        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,\n'
    '        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,\n'
    '        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,\n'
    '    ).to(device).bfloat16()\n'
    '    for m in hessian_model.modules():\n'
    '        if isinstance(m, CastedLinear):\n'
    '            m.float()\n'
    '    restore_low_dim_params_to_fp32(hessian_model)\n'
    '    # Load unbanked weights into the non-banked model\n'
    '    hessian_model.load_state_dict(\n'
    '        {k: v.to(device) for k, v in unbanked_sd.items() if k in hessian_model.state_dict()},\n'
    '        strict=False,\n'
    '    )\n'
    '    log0(f"gptq:calibrating with {args.gptq_calib_batches} batches (using val data)...")\n'
    '    calib_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)\n'
    '    hessians = collect_hessians(hessian_model, calib_loader, args, device, grad_accum_steps,\n'
    '                                num_batches=args.gptq_calib_batches)\n'
    '    log0(f"gptq:collected hessians for {len(hessians)} layers")\n'
    '    del hessian_model\n'
    '    torch.cuda.empty_cache()\n'
    '    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians)'
)

NEW_GPTQ_BLOCK = (
    '    # Unbank 3D tensors into individual 2D tensors for quantization\n'
    '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
    '    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)\n'
    '    if master_process and int(os.environ.get("SAVE_PREQUANT", "0")):\n'
    '        torch.save(unbanked_sd, "prequant_sd.pt")\n'
    '        log0("Saved prequant_sd.pt")\n'
    '    # Iterative GPTQ Hessian collection (Qronos-inspired when QRONOS_ITERS > 1)\n'
    '    hessians = None\n'
    '    prev_qr = prev_qm = None\n'
    '    for qr_iter in range(args.qronos_iters):\n'
    '        log0(f"gptq:iter {qr_iter+1}/{args.qronos_iters} building non-banked model...")\n'
    '        hessian_model = _HessianGPT(\n'
    '            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,\n'
    '            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,\n'
    '            tie_embeddings=args.tie_embeddings, logit_softcap=args.logit_softcap,\n'
    '            rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,\n'
    '            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,\n'
    '            xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,\n'
    '            ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,\n'
    '        ).to(device).bfloat16()\n'
    '        for m in hessian_model.modules():\n'
    '            if isinstance(m, CastedLinear):\n'
    '                m.float()\n'
    '        restore_low_dim_params_to_fp32(hessian_model)\n'
    '        if qr_iter == 0:\n'
    '            hessian_model.load_state_dict(\n'
    '                {k: v.to(device) for k, v in unbanked_sd.items() if k in hessian_model.state_dict()},\n'
    '                strict=False,\n'
    '            )\n'
    '        else:\n'
    '            prev_deq = dequantize_mixed_int6(prev_qr, prev_qm, unbanked_sd)\n'
    '            hessian_model.load_state_dict(\n'
    '                {k: v.to(device) for k, v in prev_deq.items() if k in hessian_model.state_dict()},\n'
    '                strict=False,\n'
    '            )\n'
    '            del prev_deq\n'
    '        log0(f"gptq:iter {qr_iter+1} calibrating with {args.gptq_calib_batches} batches (val data)...")\n'
    '        calib_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)\n'
    '        hessians = collect_hessians(hessian_model, calib_loader, args, device, grad_accum_steps,\n'
    '                                    num_batches=args.gptq_calib_batches)\n'
    '        log0(f"gptq:iter {qr_iter+1} collected hessians for {len(hessians)} layers")\n'
    '        del hessian_model\n'
    '        torch.cuda.empty_cache()\n'
    '        if qr_iter < args.qronos_iters - 1:\n'
    '            prev_qr, prev_qm = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians)\n'
    '            log0(f"gptq:iter {qr_iter+1} intermediate quantization done")\n'
    '    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians, cdquant_passes=args.cdquant_passes)\n'
    '    if args.cdquant_passes > 0:\n'
    '        log0(f"gptq:applied CDQuant refinement with {args.cdquant_passes} passes")'
)

count = code.count(OLD_GPTQ_BLOCK)
if count != 1:
    print(f"ERROR: Found {count} matches for OLD_GPTQ_BLOCK (expected 1)")
    # Debug: try to find partial matches
    for i, line in enumerate(OLD_GPTQ_BLOCK.split('\n')[:5]):
        if line and line in code:
            print(f"  line {i} found: {line[:80]}")
        else:
            print(f"  line {i} NOT FOUND: {line[:80]}")
    exit(1)

code = code.replace(OLD_GPTQ_BLOCK, NEW_GPTQ_BLOCK, 1)

with open("/root/parameter-golf/train_quant_exp.py", "w") as f:
    f.write(code)

print(f"Created train_quant_exp.py ({len(code)} bytes, {code.count(chr(10))} lines)")
print("Env vars: QRONOS_ITERS (default 1), CDQUANT_PASSES (default 0), SAVE_PREQUANT (default 0)")
