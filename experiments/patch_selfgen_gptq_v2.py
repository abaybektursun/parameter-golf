#!/usr/bin/env python3
"""Patch: self-generated proxy data for GPTQ calibration (v2: random token forward passes).
Random tokens through the model give representative activations for Hessian estimation.
No autoregressive generation — single forward passes, instant."""

with open("/root/parameter-golf/train_609_val_calib.py") as f:
    code = f.read()

# --- Patch 1: Add env var ---
code = code.replace(
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n',
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n'
    '    selfgen_gptq = bool(int(os.environ.get("SELFGEN_GPTQ", "0")))\n'
    '    selfgen_batches = int(os.environ.get("SELFGEN_BATCHES", 64))\n',
    1
)

# --- Patch 2: Add collect_hessians_selfgen ---
SELFGEN_FN = r'''
def collect_hessians_selfgen(hessian_model, args, device, num_batches=64, seed=42):
    """Collect Hessians from random-token forward passes.
    No train or val data accessed — fully self-contained."""
    import time as _time
    t0 = _time.perf_counter()
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
            # Random tokens — same shape as real calibration batches
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
    elapsed = _time.perf_counter() - t0
    return hessians, elapsed

'''

code = code.replace(
    '\n# --- GPTQ-lite int6 quantization ---',
    SELFGEN_FN + '# --- GPTQ-lite int6 quantization ---',
    1
)

# --- Patch 3: Replace calibration in main() ---
OLD_CALIB = (
    '    log0(f"gptq:calibrating with {args.gptq_calib_batches} batches (using val data)...")\n'
    '    calib_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)\n'
    '    hessians = collect_hessians(hessian_model, calib_loader, args, device, grad_accum_steps,\n'
    '                                num_batches=args.gptq_calib_batches)\n'
    '    log0(f"gptq:collected hessians for {len(hessians)} layers")'
)

NEW_CALIB = (
    '    if args.selfgen_gptq:\n'
    '        log0(f"gptq:selfgen calibration with {args.selfgen_batches} random-token batches...")\n'
    '        hessians, selfgen_time = collect_hessians_selfgen(\n'
    '            hessian_model, args, device, num_batches=args.selfgen_batches, seed=args.seed)\n'
    '        log0(f"gptq:collected hessians for {len(hessians)} layers (selfgen, {selfgen_time:.1f}s)")\n'
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

with open("/root/parameter-golf/train_selfgen_v2.py", "w") as f:
    f.write(code)

print(f"Created train_selfgen_v2.py ({len(code)} bytes)")
