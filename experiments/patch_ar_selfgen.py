"""Patch: Replace val-data GPTQ calibration with autoregressive self-generated calibration.
Reads train_gpt.py from the record submission, patches the GPTQ section, writes train_ar_selfgen.py."""

import sys

src = "/root/parameter-golf/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py"
with open(src) as f:
    code = f.read()

# --- Patch 1: Add AR generation + hessian collection functions before GPTQ section ---
AR_FUNCTIONS = '''
def generate_autoregressive_calib(model, device, num_seqs=64, seq_len=2048,
                                   vocab_size=1024, temperature=0.8, batch_size=8, seed=42):
    """Generate sequences autoregressively from the model for GPTQ calibration.
    No external data accessed — fully self-contained."""
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

'''

# Insert AR functions before GPTQ section
anchor = '\n# --- GPTQ-lite int6 quantization ---'
assert anchor in code, "Could not find GPTQ anchor"
code = code.replace(anchor, AR_FUNCTIONS + anchor, 1)

# --- Patch 2: Replace val-data calibration with AR self-gen ---
OLD_CALIB = (
    '    log0(f"gptq:calibrating with {args.gptq_calib_batches} batches (using val data)...")\n'
    '    calib_loader = DistributedTokenLoader(args.val_files, rank, world_size, device)\n'
    '    hessians = collect_hessians(hessian_model, calib_loader, args, device, grad_accum_steps,\n'
    '                                num_batches=args.gptq_calib_batches)\n'
    '    log0(f"gptq:collected hessians for {len(hessians)} layers")'
)

NEW_CALIB = (
    '    # Autoregressive self-generated calibration (no external data)\n'
    '    log0("gptq:generating autoregressive calibration data (64 seqs x 2048 tokens, temp=0.8)...")\n'
    '    base_model.load_state_dict(export_sd, strict=False)\n'
    '    t_gen = time.perf_counter()\n'
    '    ar_tokens = generate_autoregressive_calib(\n'
    '        base_model, device, num_seqs=64, seq_len=args.train_seq_len,\n'
    '        vocab_size=args.vocab_size, temperature=0.8, batch_size=8, seed=args.seed,\n'
    '    )\n'
    '    log0(f"gptq:generated {len(ar_tokens)} sequences in {time.perf_counter()-t_gen:.1f}s")\n'
    '    log0("gptq:collecting hessians from autoregressive data...")\n'
    '    hessians = collect_hessians_from_tokens(hessian_model, ar_tokens, device)\n'
    '    log0(f"gptq:collected hessians for {len(hessians)} layers (AR self-gen)")\n'
    '    del ar_tokens'
)

count = code.count(OLD_CALIB)
assert count == 1, f"Expected 1 match for OLD_CALIB, found {count}"
code = code.replace(OLD_CALIB, NEW_CALIB, 1)

out = "/root/parameter-golf/train_ar_selfgen.py"
with open(out, "w") as f:
    f.write(code)
print(f"Created {out} ({len(code)} bytes, {code.count(chr(10))} lines)")
