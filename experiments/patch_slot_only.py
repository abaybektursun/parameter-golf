#!/usr/bin/env python3
"""Minimal patch: add SLOT additive bias eval to train_609_val_calib.py.
Only adds eval-time code — does NOT touch training loop or torch.compile."""

with open("/root/parameter-golf/train_609_val_calib.py") as f:
    code = f.read()

# --- Patch 1: Add SLOT env vars ---
code = code.replace(
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n',
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n'
    '    slot_enabled = bool(int(os.environ.get("SLOT_ENABLED", "0")))\n'
    '    slot_lr = float(os.environ.get("SLOT_LR", 0.01))\n'
    '    slot_steps = int(os.environ.get("SLOT_STEPS", 3))\n'
    '    slot_chunk_tokens = int(os.environ.get("SLOT_CHUNK_TOKENS", 32768))\n',
    1
)

# --- Patch 2: Add SLOT eval function (after eval_val_sliding, before GPTQ section) ---
SLOT_FN = r'''
def eval_val_sliding_slot(
    args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride, batch_seqs=32, log0=print,
):
    """Legal score-first SLOT: 512-dim additive bias on final hidden states.
    Score chunk under inference_mode, then optimize bias on scored chunk."""
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

    slot_bias = torch.zeros(model_dim, device=device, dtype=torch.float32, requires_grad=True)
    slot_opt = torch.optim.AdamW([slot_bias], lr=args.slot_lr)

    log0(f"slot:start chunks={num_chunks} chunk_tokens={slot_chunk} "
         f"lr={args.slot_lr} steps={args.slot_steps} bias_dim={model_dim}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    def add_bias_hook(module, input, output):
        return output + slot_bias.to(output.dtype)

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * slot_chunk
        chunk_end = min((ci + 1) * slot_chunk, total_tokens)

        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        # --- Phase 1: SCORE (inference_mode) ---
        base_model.eval()
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

        # --- Phase 2: TRAIN bias on scored chunk (legal) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.slot_steps > 0:
            hook_handle = base_model.final_norm.register_forward_hook(add_bias_hook)
            base_model.eval()
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
            log0(f"  slot [{ci+1}/{num_chunks}] bpb={rbpb:.6f} bias_norm={slot_bias.norm().item():.4f} time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    log0(f"slot:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb

'''

code = code.replace(
    '\n# --- GPTQ-lite int6 quantization ---',
    SLOT_FN + '# --- GPTQ-lite int6 quantization ---',
    1
)

# --- Patch 3: Add SLOT call after sliding window eval ---
OLD_EVAL_END = (
    '    if distributed:\n'
    '        dist.destroy_process_group()\n'
    'if __name__ == "__main__":\n'
    '    main()'
)

NEW_EVAL_END = (
    '    if args.slot_enabled:\n'
    '        log0("=== SLOT additive bias eval ===")\n'
    '        torch.cuda.synchronize()\n'
    '        t_slot = time.perf_counter()\n'
    '        slot_loss, slot_bpb = eval_val_sliding_slot(\n'
    '            args, eval_model, rank, world_size, device,\n'
    '            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n'
    '            stride=args.eval_stride, log0=log0,\n'
    '        )\n'
    '        torch.cuda.synchronize()\n'
    '        log0(f"final_slot val_loss:{slot_loss:.4f} val_bpb:{slot_bpb:.4f} time:{1000*(time.perf_counter()-t_slot):.0f}ms")\n'
    '        log0(f"final_slot_exact val_loss:{slot_loss:.8f} val_bpb:{slot_bpb:.8f}")\n'
    '    if distributed:\n'
    '        dist.destroy_process_group()\n'
    'if __name__ == "__main__":\n'
    '    main()'
)

count = code.count(OLD_EVAL_END)
if count != 1:
    print(f"ERROR: Found {count} matches for OLD_EVAL_END")
    exit(1)
code = code.replace(OLD_EVAL_END, NEW_EVAL_END, 1)

with open("/root/parameter-golf/train_slot_exp.py", "w") as f:
    f.write(code)

print(f"Created train_slot_exp.py ({len(code)} bytes, {code.count(chr(10))} lines)")
