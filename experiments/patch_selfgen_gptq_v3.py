#!/usr/bin/env python3
"""Patch: self-generated GPTQ v3 — Gibbs-refined tokens + large batch count.
3 rounds of forward→sample→replace to approximate model's stationary distribution.
Then collect Hessians with 512 batches for robust estimation."""

with open("/root/parameter-golf/train_609_val_calib.py") as f:
    code = f.read()

# --- Patch 1: Add env vars ---
code = code.replace(
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n',
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n'
    '    selfgen_gptq = bool(int(os.environ.get("SELFGEN_GPTQ", "0")))\n'
    '    selfgen_batches = int(os.environ.get("SELFGEN_BATCHES", 256))\n'
    '    selfgen_batch_size = int(os.environ.get("SELFGEN_BS", 8))\n'
    '    selfgen_gibbs_rounds = int(os.environ.get("SELFGEN_GIBBS", 3))\n'
    '    selfgen_temperature = float(os.environ.get("SELFGEN_TEMP", 1.0))\n',
    1
)

# --- Patch 2: Add Gibbs-refined Hessian collection ---
SELFGEN_FN = r'''
def collect_hessians_selfgen(hessian_model, banked_model, args, device, num_batches=256,
                              batch_size=8, gibbs_rounds=3, temperature=1.0, seed=42):
    """Collect Hessians via Gibbs-refined self-generated tokens.
    1. Start with random tokens
    2. Run gibbs_rounds of forward→sample→replace (using banked model with forward_logits)
    3. Collect Hessians on final-round forward pass (using hessian model)
    Fully self-contained — no train or val data accessed."""
    import time as _time
    t0 = _time.perf_counter()
    vocab = args.vocab_size
    seq_len = args.train_seq_len
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Phase 1: Gibbs refinement to get realistic tokens
    t_gibbs_start = _time.perf_counter()
    all_x = []
    all_y = []
    banked_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for bi in range(num_batches):
            x = torch.randint(0, vocab, (batch_size, seq_len), device=device, generator=rng)
            for _round in range(gibbs_rounds):
                logits = banked_model.forward_logits(x)
                probs = torch.softmax(logits.float() / temperature, dim=-1)
                x = torch.multinomial(probs.reshape(-1, vocab), 1, generator=rng).reshape(batch_size, seq_len)
            # Use final refined tokens for Hessian collection
            # Create shifted y for the hessian model's forward
            y = torch.cat([x[:, 1:], torch.randint(0, vocab, (batch_size, 1), device=device, generator=rng)], dim=1)
            all_x.append(x)
            all_y.append(y)
    t_gibbs = _time.perf_counter() - t_gibbs_start

    # Phase 2: Collect Hessians using the non-banked hessian model
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
    t_hessian_start = _time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for x, y in zip(all_x, all_y):
            hessian_model(x, y)
    t_hessian = _time.perf_counter() - t_hessian_start
    for h in hooks:
        h.remove()
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    total = _time.perf_counter() - t0
    total_tokens = num_batches * batch_size * seq_len
    return hessians, total, t_gibbs, t_hessian, total_tokens

'''

code = code.replace(
    '\n# --- GPTQ-lite int6 quantization ---',
    SELFGEN_FN + '# --- GPTQ-lite int6 quantization ---',
    1
)

# --- Patch 3: Generate tokens BEFORE unbanking, collect Hessians AFTER ---
OLD_UNBANK = (
    '    # Unbank 3D tensors into individual 2D tensors for quantization\n'
    '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
    '    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)'
)

NEW_UNBANK = (
    '    # For selfgen GPTQ: load export weights into banked model for Gibbs generation\n'
    '    if args.selfgen_gptq:\n'
    '        base_model.load_state_dict(export_sd, strict=False)\n'
    '    # Unbank 3D tensors into individual 2D tensors for quantization\n'
    '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
    '    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)'
)

count = code.count(OLD_UNBANK)
if count != 1:
    print(f"ERROR: Found {count} matches for OLD_UNBANK")
    exit(1)
code = code.replace(OLD_UNBANK, NEW_UNBANK, 1)

# --- Patch 4: Replace calibration ---
OLD_CALIB = (
    '    log0(f"gptq:calibrating with {args.gptq_calib_batches} batches (using val data)...")\n'
    '    calib_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)\n'
    '    hessians = collect_hessians(hessian_model, calib_loader, args, device, grad_accum_steps,\n'
    '                                num_batches=args.gptq_calib_batches)\n'
    '    log0(f"gptq:collected hessians for {len(hessians)} layers")'
)

NEW_CALIB = (
    '    if args.selfgen_gptq:\n'
    '        log0(f"gptq:selfgen Gibbs-refined calibration: {args.selfgen_batches} batches, "\n'
    '             f"bs={args.selfgen_batch_size}, gibbs={args.selfgen_gibbs_rounds}, temp={args.selfgen_temperature}")\n'
    '        hessians, total_t, gibbs_t, hessian_t, total_tok = collect_hessians_selfgen(\n'
    '            hessian_model, base_model, args, device,\n'
    '            num_batches=args.selfgen_batches, batch_size=args.selfgen_batch_size,\n'
    '            gibbs_rounds=args.selfgen_gibbs_rounds, temperature=args.selfgen_temperature,\n'
    '            seed=args.seed)\n'
    '        log0(f"gptq:selfgen done: {len(hessians)} layers, {total_tok} tokens, "\n'
    '             f"gibbs={gibbs_t:.1f}s hessian={hessian_t:.1f}s total={total_t:.1f}s")\n'
    '    else:\n'
    '        log0(f"gptq:calibrating with {args.gptq_calib_batches} batches (using val data)...")\n'
    '        calib_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)\n'
    '        hessians = collect_hessians(hessian_model, calib_loader, args, device, grad_accum_steps,\n'
    '                                    num_batches=args.gptq_calib_batches)\n'
    '        log0(f"gptq:collected hessians for {len(hessians)} layers")'
)

count = code.count(OLD_CALIB)
if count != 1:
    print(f"ERROR: Found {count} matches for OLD_CALIB")
    exit(1)
code = code.replace(OLD_CALIB, NEW_CALIB, 1)

with open("/root/parameter-golf/train_selfgen_v3.py", "w") as f:
    f.write(code)

print(f"Created train_selfgen_v3.py ({len(code)} bytes)")
