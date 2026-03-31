#!/usr/bin/env python3
"""Patch train_609_val_calib.py → train_trio_exp.py:
1. Spectral Conditioning Init (SPEC_LAMBDA)
2. Non-uniform quantization grid (NONUNIFORM_QUANT)
3. SLOT additive bias eval (SLOT_ENABLED)
"""

with open("/root/parameter-golf/train_609_val_calib.py") as f:
    code = f.read()

# --- Patch 1: Add env vars to Hyperparameters ---
code = code.replace(
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n',
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n'
    '    spec_lambda = float(os.environ.get("SPEC_LAMBDA", "0"))  # Spectral Conditioning Init\n'
    '    nonuniform_quant = bool(int(os.environ.get("NONUNIFORM_QUANT", "0")))\n'
    '    slot_enabled = bool(int(os.environ.get("SLOT_ENABLED", "0")))\n'
    '    slot_lr = float(os.environ.get("SLOT_LR", 0.01))\n'
    '    slot_steps = int(os.environ.get("SLOT_STEPS", 3))\n'
    '    slot_chunk_tokens = int(os.environ.get("SLOT_CHUNK_TOKENS", 32768))\n',
    1
)

# --- Patch 2: Add non-uniform quantization + SLOT functions before GPTQ section ---
NEW_FUNCTIONS = r'''
def quantize_nonuniform_gptq(weight, hessian=None, num_levels=63, block_size=128):
    """Non-uniform GPTQ: quantile centroids instead of uniform int6 grid."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _quantize_int6_percentile(t32, 31)  # fallback
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
    # Per-row scale (same as uniform)
    row_clip = t32.abs().amax(dim=1)
    s = (row_clip / 31.0).clamp_min(1.0 / 31.0).to(torch.float16)
    sf = s.float()
    # Compute quantile centroids on normalized weights
    W_norm = (t32 / sf[:, None]).reshape(-1)
    percentiles = torch.linspace(0.5 / num_levels, 1 - 0.5 / num_levels, num_levels)
    centroids = torch.quantile(W_norm, percentiles).contiguous()
    # Boundaries for fast searchsorted
    boundaries = ((centroids[:-1] + centroids[1:]) / 2).contiguous()
    # GPTQ with non-uniform grid
    Q = torch.zeros(rows, cols, dtype=torch.uint8)
    W_work = W.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        count = i2 - i1
        W1 = W_work[:, i1:i2].clone()
        Q1 = torch.zeros(rows, count, dtype=torch.uint8)
        Err1 = torch.zeros(rows, count)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            w_norm = w / sf
            idx = torch.searchsorted(boundaries, w_norm).clamp(0, num_levels - 1)
            Q1[:, i] = idx.to(torch.uint8)
            recon = centroids[idx] * sf
            err = (w - recon) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        if i2 < cols:
            W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    Q = Q[:, inv_perm]
    return Q, s, centroids


def mixed_quantize_nonuniform(state_dict, int6_cats, hessians=None):
    """Non-uniform quantization variant of mixed_quantize_int6."""
    num_layers_total = max(
        (int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")), default=0,
    ) + 1
    result = {}
    meta = {}
    all_centroids = {}
    for name, tensor in state_dict.items():
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
        if cat in int6_cats and t.ndim >= 1:
            H = hessians.get(name) if hessians else None
            if H is not None and t.ndim == 2:
                q, s, centroids = quantize_nonuniform_gptq(t, hessian=H)
                all_centroids[name] = centroids.to(torch.float16)
                result[name + ".q"] = q
                result[name + ".scale"] = s
                meta[name] = {"type": "nonuniform"}
            else:
                q, s = quantize_int6_per_row(t, clip_range=31)
                result[name + ".q"] = q
                result[name + ".scale"] = s
                meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta, all_centroids


def dequantize_nonuniform(result, meta, template_sd, all_centroids):
    """Dequantize non-uniform quantized weights."""
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if isinstance(info, dict) and info.get("type") == "nonuniform":
            centroids = all_centroids[name].float()
            w = centroids[q.long()] * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
            out[name] = w.to(orig_dtype)
        else:
            if s.ndim > 0:
                out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
            else:
                out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


def eval_val_sliding_slot(
    args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride, batch_seqs=32, log0=print,
):
    """Legal score-first SLOT: optimize a 512-dim additive bias on final hidden states.
    Score each chunk first (inference_mode), then optimize bias on scored chunk."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    slot_chunk = args.slot_chunk_tokens
    model_dim = args.model_dim

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + slot_chunk - 1) // slot_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // slot_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    # SLOT bias: single learnable vector
    slot_bias = torch.zeros(model_dim, device=device, dtype=torch.float32, requires_grad=True)
    slot_opt = torch.optim.AdamW([slot_bias], lr=args.slot_lr)

    log0(f"slot_sliding:start chunks={num_chunks} chunk_tokens={slot_chunk} "
         f"lr={args.slot_lr} steps={args.slot_steps} bias_dim={model_dim}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    # Hook to add bias to final hidden states
    bias_hook = None
    hook_handle = None

    def add_bias_hook(module, input, output):
        return output + slot_bias.to(output.dtype)

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * slot_chunk
        chunk_end = min((ci + 1) * slot_chunk, total_tokens)

        # --- Phase 1: SCORE (inference_mode, with current bias) ---
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        # Register hook on final_norm to add bias AFTER normalization
        hook_handle = base_model.final_norm.register_forward_hook(add_bias_hook)

        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        hook_handle.remove()

        # --- Phase 2: TRAIN bias on already-scored chunk (legal) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.slot_steps > 0:
            hook_handle = base_model.final_norm.register_forward_hook(add_bias_hook)
            base_model.eval()  # model stays eval, only bias trains
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _step in range(args.slot_steps):
                    for bs in range(0, min(my_chunk_seqs, 4), 4):
                        be = min(bs + 4, my_chunk_seqs)
                        start_tok = chunk_start + (my_seq_s + bs) * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        slot_opt.zero_grad()
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = base_model.forward_logits(x)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)).float(),
                            y.reshape(-1),
                        )
                        loss.backward()
                        if world_size > 1 and slot_bias.grad is not None:
                            dist.all_reduce(slot_bias.grad, op=dist.ReduceOp.AVG)
                        slot_opt.step()
            hook_handle.remove()

        if rank == 0 and (ci % 100 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  slot_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} bias_norm={slot_bias.norm().item():.4f} time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    log0(f"slot_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb

'''

code = code.replace(
    '\n# --- GPTQ-lite int6 quantization ---',
    NEW_FUNCTIONS + '# --- GPTQ-lite int6 quantization ---',
    1
)

# --- Patch 3: Add Spectral Init after model creation in main() ---
# Find where the model is created and add spectral conditioning
SPEC_INIT_CODE = (
    '    # Spectral Conditioning Init: add lambda*I to QKV at init\n'
    '    if args.spec_lambda > 0:\n'
    '        with torch.no_grad():\n'
    '            lam = args.spec_lambda\n'
    '            n = args.num_layers\n'
    '            dq = min(model.qo_bank.shape[1], model.qo_bank.shape[2])\n'
    '            eye_q = lam * torch.eye(dq, device=device, dtype=model.qo_bank.dtype)\n'
    '            for i in range(n):\n'
    '                model.qo_bank.data[i, :dq, :dq] += eye_q  # Q weights\n'
    '            dk = min(model.kv_bank.shape[1], model.kv_bank.shape[2])\n'
    '            eye_k = lam * torch.eye(dk, device=device, dtype=model.kv_bank.dtype)\n'
    '            for i in range(n):\n'
    '                model.kv_bank.data[i, :dk, :dk] += eye_k  # K weights\n'
    '                model.kv_bank.data[n + i, :dk, :dk] += eye_k  # V weights\n'
    '            log0(f"spectral_init: lambda={lam} applied to Q({dq}), K({dk}), V({dk})")\n'
)

# Insert after model.to(device) and before the optimizer setup
# Find the "model = GPT(" line and the subsequent optimizer setup
code = code.replace(
    '    # zeropower_via_newtonschulz5 runs eagerly with bmm -- do NOT compile\n',
    '    # zeropower_via_newtonschulz5 runs eagerly with bmm -- do NOT compile\n'
    + SPEC_INIT_CODE,
    1
)

# Wait, that's too early - the model isn't created yet. Let me find a better insertion point.
# Undo that
code = code.replace(
    '    # zeropower_via_newtonschulz5 runs eagerly with bmm -- do NOT compile\n'
    + SPEC_INIT_CODE,
    '    # zeropower_via_newtonschulz5 runs eagerly with bmm -- do NOT compile\n',
    1
)

# Insert spectral init right before optimizer creation (after model is fully set up)
code = code.replace(
    '    optimizer_tok = torch.optim.AdamW(\n',
    SPEC_INIT_CODE + '    optimizer_tok = torch.optim.AdamW(\n',
    1
)

# --- Patch 4: Add non-uniform quantization path in main() ---
# Replace the quantization call to support non-uniform
OLD_QUANT = '    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians)'

NEW_QUANT = (
    '    if args.nonuniform_quant:\n'
    '        log0("Using non-uniform quantization grid")\n'
    '        quant_result, quant_meta, all_centroids = mixed_quantize_nonuniform(unbanked_sd, {"mlp", "attn"}, hessians=hessians)\n'
    '    else:\n'
    '        quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians)\n'
    '        all_centroids = None'
)

count = code.count(OLD_QUANT)
if count != 1:
    print(f"ERROR: Found {count} matches for OLD_QUANT (expected 1)")
    exit(1)
code = code.replace(OLD_QUANT, NEW_QUANT, 1)

# --- Patch 5: Fix serialization for non-uniform (add centroids to saved state) ---
OLD_SERIALIZE = '    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)'
NEW_SERIALIZE = '    torch.save({"w": quant_result, "m": quant_meta, "c": all_centroids}, quant_buf)'
code = code.replace(OLD_SERIALIZE, NEW_SERIALIZE, 1)

# Fix deserialization
OLD_DESER = '    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)'
NEW_DESER = (
    '    if args.nonuniform_quant:\n'
    '        deq_unbanked = dequantize_nonuniform(quant_state["w"], quant_state["m"], unbanked_sd, quant_state.get("c", {}))\n'
    '    else:\n'
    '        deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)'
)
code = code.replace(OLD_DESER, NEW_DESER, 1)

# --- Patch 6: Add SLOT eval after sliding window eval ---
OLD_EVAL_END = (
    '    if distributed:\n'
    '        dist.destroy_process_group()\n'
    'if __name__ == "__main__":\n'
    '    main()'
)

NEW_EVAL_END = (
    '    # --- SLOT eval ---\n'
    '    if args.slot_enabled:\n'
    '        log0("=== Starting SLOT additive bias eval ===")\n'
    '        torch.cuda.synchronize()\n'
    '        t_slot = time.perf_counter()\n'
    '        slot_val_loss, slot_val_bpb = eval_val_sliding_slot(\n'
    '            args, eval_model, rank, world_size, device,\n'
    '            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n'
    '            stride=args.eval_stride, log0=log0,\n'
    '        )\n'
    '        torch.cuda.synchronize()\n'
    '        log0(\n'
    '            f"final_slot val_loss:{slot_val_loss:.4f} val_bpb:{slot_val_bpb:.4f} "\n'
    '            f"eval_time:{1000.0 * (time.perf_counter() - t_slot):.0f}ms"\n'
    '        )\n'
    '        log0(f"final_slot_exact val_loss:{slot_val_loss:.8f} val_bpb:{slot_val_bpb:.8f}")\n'
    '    if distributed:\n'
    '        dist.destroy_process_group()\n'
    'if __name__ == "__main__":\n'
    '    main()'
)

count = code.count(OLD_EVAL_END)
if count != 1:
    print(f"ERROR: Found {count} matches for OLD_EVAL_END (expected 1)")
    exit(1)
code = code.replace(OLD_EVAL_END, NEW_EVAL_END, 1)

# --- Patch 7: Fix selective pruning for non-uniform ---
# The existing pruning checks q.abs() == 1 which doesn't work for uint8 indices
# Add a guard: skip pruning if non-uniform
OLD_PRUNE_START = '    # NOVEL: Selective ±1 pruning by reconstruction error'
NEW_PRUNE_START = '    # NOVEL: Selective ±1 pruning by reconstruction error (skip for non-uniform)\n    if not args.nonuniform_quant:'
# Need to indent the whole pruning block... this is tricky with string replacement.
# Instead, let's just wrap it in a simple if:
code = code.replace(
    '    # NOVEL: Selective ±1 pruning by reconstruction error\n'
    '    # Sort ±1 quantized values by their reconstruction error (scale²),\n'
    '    # prune least-impactful first until artifact fits target size.\n'
    '    target_mb = float(os.environ.get("TARGET_MB", "15.9"))',
    '    # NOVEL: Selective ±1 pruning by reconstruction error\n'
    '    # Sort ±1 quantized values by their reconstruction error (scale²),\n'
    '    # prune least-impactful first until artifact fits target size.\n'
    '    target_mb = float(os.environ.get("TARGET_MB", "15.9"))\n'
    '    if args.nonuniform_quant:\n'
    '        log0(f"nonuniform_quant: skipping ±1 pruning (not applicable)")\n'
    '        ones_info = []\n'
    '    else:\n'
    '        target_mb = target_mb  # keep variable in scope',
    1
)

with open("/root/parameter-golf/train_trio_exp.py", "w") as f:
    f.write(code)

print(f"Created train_trio_exp.py ({len(code)} bytes, {code.count(chr(10))} lines)")
print("Env vars: SPEC_LAMBDA, NONUNIFORM_QUANT, SLOT_ENABLED, SLOT_LR, SLOT_STEPS")
