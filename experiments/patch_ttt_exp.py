#!/usr/bin/env python3
"""Patch train_609_val_calib.py → train_ttt_exp.py: add legal score-first TTT experiments."""

with open("/root/parameter-golf/train_609_val_calib.py") as f:
    code = f.read()

# --- Patch 1: Add TTT env vars to Hyperparameters ---
code = code.replace(
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n',
    '    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))\n'
    '    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))\n'
    '    ttt_lr = float(os.environ.get("TTT_LR", 0.002))\n'
    '    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))\n'
    '    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))\n'
    '    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))\n'
    '    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))\n'
    '    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))\n'
    '    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))\n'
    '    ttt_param_mode = os.environ.get("TTT_PARAM_MODE", "all")  # all, mlp_down, mlp_all, last_N\n',
    1
)

# --- Patch 2: Add eval_val_sliding_ttt function before the GPTQ section ---
TTT_FN = r'''
def eval_val_sliding_ttt(
    args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride, batch_seqs=32, log0=print,
):
    """Legal score-first TTT: score each chunk under inference_mode, then train on it.
    Every token graded BEFORE any weight update that could use it."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    # Pre-compute all window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]

    # Assign each window to a chunk
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    # Select TTT parameters based on mode
    param_mode = args.ttt_param_mode
    ttt_params = []
    for name, p in base_model.named_parameters():
        include = False
        if param_mode == "all":
            # Freeze first N blocks, unfreeze rest
            freeze = False
            for bi in range(min(args.ttt_freeze_blocks, 11)):
                if f"blocks.{bi}." in name or (f"_bank" in name and False):
                    pass  # bank params handled below
            # For banked model: freeze based on block index within bank
            if "_bank" in name:
                include = True  # banks span all layers, can't easily freeze subsets
            elif any(f"blocks.{bi}." in name for bi in range(min(args.ttt_freeze_blocks, 11))):
                include = False
            else:
                include = True
        elif param_mode == "mlp_down":
            include = "mlp_down_bank" in name
        elif param_mode == "mlp_all":
            include = "mlp_down_bank" in name or "mlp_up_bank" in name
        elif param_mode.startswith("last_"):
            n_last = int(param_mode.split("_")[1])
            # For banked params, always include (they span all layers)
            # For individual block params, only include if block >= num_layers - n_last
            num_layers = 11
            threshold = num_layers - n_last
            if "_bank" in name:
                include = True  # TODO: could slice bank but complex
            elif "blocks." in name:
                block_idx = int(name.split(".")[1])
                include = block_idx >= threshold
            else:
                include = True  # skip_weights, tok_emb, etc.
        else:
            include = True

        if include:
            p.requires_grad_(True)
            ttt_params.append(p)
        else:
            p.requires_grad_(False)

    unfrozen_count = sum(p.numel() for p in ttt_params)
    frozen_count = sum(p.numel() for p in base_model.parameters() if not p.requires_grad)
    log0(f"ttt_sliding:start mode={param_mode} chunks={num_chunks} chunk_tokens={ttt_chunk} "
         f"windows={len(window_starts)} stride={stride} "
         f"lr={args.ttt_lr} epochs={args.ttt_epochs} "
         f"unfrozen={unfrozen_count} frozen={frozen_count}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        # --- Phase 1: SCORE (inference_mode — no gradients, no weight mutation) ---
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
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

        # --- Phase 2: TRAIN on already-scored chunk (legal) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
         f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb

'''

code = code.replace(
    '\n# --- GPTQ-lite int6 quantization ---',
    TTT_FN + '# --- GPTQ-lite int6 quantization ---',
    1
)

# --- Patch 3: Add TTT call after the sliding window eval in main() ---
# Find the sliding window eval section and add TTT after it
OLD_EVAL_END = (
    '    if distributed:\n'
    '        dist.destroy_process_group()\n'
    'if __name__ == "__main__":\n'
    '    main()'
)

NEW_EVAL_END = (
    '    # --- TTT eval (legal score-first) ---\n'
    '    if args.ttt_enabled:\n'
    '        log0("=== Starting legal score-first TTT ===")\n'
    '        torch.cuda.synchronize()\n'
    '        t_ttt = time.perf_counter()\n'
    '        ttt_val_loss, ttt_val_bpb = eval_val_sliding_ttt(\n'
    '            args, eval_model, rank, world_size, device,\n'
    '            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n'
    '            stride=args.eval_stride, log0=log0,\n'
    '        )\n'
    '        torch.cuda.synchronize()\n'
    '        log0(\n'
    '            f"final_ttt val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "\n'
    '            f"mode:{args.ttt_param_mode} eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"\n'
    '        )\n'
    '        log0(f"final_ttt_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f}")\n'
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

with open("/root/parameter-golf/train_ttt_exp.py", "w") as f:
    f.write(code)

print(f"Created train_ttt_exp.py ({len(code)} bytes, {code.count(chr(10))} lines)")
print("TTT env vars: TTT_ENABLED, TTT_LR, TTT_EPOCHS, TTT_CHUNK_TOKENS, TTT_PARAM_MODE")
print("TTT_PARAM_MODE options: all, mlp_down, mlp_all, last_N (e.g. last_3)")
