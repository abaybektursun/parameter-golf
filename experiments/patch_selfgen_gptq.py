#!/usr/bin/env python3
"""Patch: self-generated proxy data for GPTQ calibration.
Model generates its own tokens with fixed seeds, then collects Hessians on those activations.
Touches neither train nor val data during quantization."""

with open("/root/parameter-golf/train_609_val_calib.py") as f:
    code = f.read()

# --- Patch 1: Add env var ---
code = code.replace(
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n',
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n'
    '    selfgen_gptq = bool(int(os.environ.get("SELFGEN_GPTQ", "0")))\n'
    '    selfgen_batches = int(os.environ.get("SELFGEN_BATCHES", 64))\n'
    '    selfgen_seq_len = int(os.environ.get("SELFGEN_SEQ_LEN", 2048))\n',
    1
)

# --- Patch 2: Add self-generation function before GPTQ section ---
SELFGEN_FN = r'''
def generate_proxy_data(model, device, num_batches=64, seq_len=2048, vocab_size=1024, seed=42):
    """Generate token sequences from the model itself for GPTQ calibration.
    Uses fixed seed for reproducibility. No train or val data accessed."""
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_idx in range(num_batches):
            # Start with random prompt tokens (fixed seed)
            prompt = torch.randint(0, vocab_size, (1, 1), device=device, generator=rng)
            tokens = [prompt]
            # Autoregressive generation
            for pos in range(seq_len - 1):
                x = torch.cat(tokens, dim=1)
                logits = model.forward_logits(x)
                next_logit = logits[:, -1, :]  # [1, vocab]
                # Temperature sampling with fixed seed
                probs = torch.softmax(next_logit / 0.8, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens.append(next_tok)
            full_seq = torch.cat(tokens, dim=1)  # [1, seq_len]
            all_tokens.append(full_seq)
    return all_tokens


def collect_hessians_selfgen(hessian_model, proxy_tokens, device):
    """Collect Hessians from self-generated proxy data."""
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
    num_batches = len(proxy_tokens)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in proxy_tokens:
            x = seq[:, :-1].to(device)
            y = seq[:, 1:].to(device)
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

'''

code = code.replace(
    '\n# --- GPTQ-lite int6 quantization ---',
    SELFGEN_FN + '# --- GPTQ-lite int6 quantization ---',
    1
)

# --- Patch 3: Replace Hessian collection in main() to optionally use self-generated data ---
# --- Patch 3: Generate proxy data BEFORE unbanking (using banked model with forward_logits) ---
# Then pass tokens to hessian model for Hessian collection

OLD_UNBANK = (
    '    # Unbank 3D tensors into individual 2D tensors for quantization\n'
    '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
    '    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)'
)

NEW_UNBANK = (
    '    # Self-generated proxy data for GPTQ (before unbanking, using banked model)\n'
    '    selfgen_tokens = None\n'
    '    if args.selfgen_gptq:\n'
    '        log0(f"gptq:generating {args.selfgen_batches} proxy sequences (self-generated)...")\n'
    '        # Load export weights into banked model for generation\n'
    '        base_model.load_state_dict(export_sd, strict=False)\n'
    '        selfgen_tokens = generate_proxy_data(\n'
    '            base_model, device, num_batches=args.selfgen_batches,\n'
    '            seq_len=args.selfgen_seq_len, vocab_size=args.vocab_size, seed=args.seed,\n'
    '        )\n'
    '        log0(f"gptq:generated {len(selfgen_tokens)} proxy sequences")\n'
    '    # Unbank 3D tensors into individual 2D tensors for quantization\n'
    '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
    '    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)'
)

OLD_CALIB = (
    '    log0(f"gptq:calibrating with {args.gptq_calib_batches} batches (using val data)...")\n'
    '    calib_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)\n'
    '    hessians = collect_hessians(hessian_model, calib_loader, args, device, grad_accum_steps,\n'
    '                                num_batches=args.gptq_calib_batches)\n'
    '    log0(f"gptq:collected hessians for {len(hessians)} layers")'
)

NEW_CALIB = (
    '    if args.selfgen_gptq and selfgen_tokens is not None:\n'
    '        log0(f"gptq:collecting hessians from self-generated proxy data...")\n'
    '        hessians = collect_hessians_selfgen(hessian_model, selfgen_tokens, device)\n'
    '        log0(f"gptq:collected hessians for {len(hessians)} layers (self-generated)")\n'
    '        del selfgen_tokens\n'
    '    else:\n'
    '        log0(f"gptq:calibrating with {args.gptq_calib_batches} batches (using val data)...")\n'
    '        calib_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)\n'
    '        hessians = collect_hessians(hessian_model, calib_loader, args, device, grad_accum_steps,\n'
    '                                    num_batches=args.gptq_calib_batches)\n'
    '        log0(f"gptq:collected hessians for {len(hessians)} layers")'
)

# Apply unbank patch (generate proxy data before unbanking)
count = code.count(OLD_UNBANK)
if count != 1:
    print(f"ERROR: Found {count} matches for OLD_UNBANK")
    exit(1)
code = code.replace(OLD_UNBANK, NEW_UNBANK, 1)

# Apply calibration patch (use selfgen tokens for Hessians)
count = code.count(OLD_CALIB)
if count != 1:
    print(f"ERROR: Found {count} matches for OLD_CALIB")
    for i, line in enumerate(OLD_CALIB.split('\n')[:3]):
        print(f"  line {i}: {'FOUND' if line in code else 'MISSING'}: {line[:80]}")
    exit(1)
code = code.replace(OLD_CALIB, NEW_CALIB, 1)

with open("/root/parameter-golf/train_selfgen_gptq.py", "w") as f:
    f.write(code)

print(f"Created train_selfgen_gptq.py ({len(code)} bytes, {code.count(chr(10))} lines)")
