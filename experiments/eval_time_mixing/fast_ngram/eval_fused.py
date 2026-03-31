"""
Pipelined n-gram + neural LM eval.
Precompute-all approach: n-gram + indices computed in threads overlapping model load + compile.
Precomputed indices eliminate Python loop overhead in the main eval loop.
"""
from __future__ import annotations
import argparse, io, math, time, glob, threading
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F


def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    return torch.from_numpy(
        np.fromfile(file, dtype="<u2", count=int(header[2]),
                    offset=256 * np.dtype("<i4").itemsize)
        .astype(np.uint16, copy=False))


def build_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size())
    sz = max(sp_vs, vocab_size)
    bb = np.zeros(sz, dtype=np.int16); ls = np.zeros(sz, dtype=np.bool_)
    bd = np.ones(sz, dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        bd[tid] = False
        if sp.is_byte(tid): bb[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): ls[tid] = True; piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.float64, device=device),
            torch.tensor(ls, dtype=torch.bool, device=device),
            torch.tensor(bd, dtype=torch.bool, device=device))


def _decompressors():
    try:
        import brotli; yield "brotli", brotli.decompress
    except ImportError: pass
    import lzma; yield "lzma", lzma.decompress
    try:
        import zstandard; yield "zstd", zstandard.ZstdDecompressor().decompress
    except ImportError: pass
    import zlib; yield "zlib", zlib.decompress


def load_model(args, tg, device):
    model = tg.GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims,
        ln_scale=True, ve_enabled=True,
        ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for m in model.modules():
        if isinstance(m, tg.CastedLinear): m.float()
    tg.restore_low_dim_params_to_fp32(model)
    print(f"Loading {args.model}...")
    with open(args.model, "rb") as f: blob = f.read()
    for name, fn in _decompressors():
        try: dec = fn(blob); print(f"  Decompressed with {name}"); break
        except: continue
    qs = torch.load(io.BytesIO(dec), map_location="cpu")
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ub = tg._unbank_state_dict(sd, args.num_layers)
    dq = tg.dequantize_mixed_int6(qs["w"], qs["m"], ub)
    model.load_state_dict(tg._rebank_state_dict(dq, args.num_layers, sd), strict=True)
    model.eval()
    print("Model loaded.")
    return model


def precompute_batch_indices(all_windows, total_tokens, seq_len, stride, batch_seqs):
    n_batches = (len(all_windows) + batch_seqs - 1) // batch_seqs
    est = total_tokens + seq_len
    all_bi = np.zeros(est, dtype=np.int64)
    all_si = np.zeros(est, dtype=np.int64)
    all_gp = np.zeros(est, dtype=np.int64)
    batch_starts = []
    batch_score_ranges = []
    max_scored = 0
    flat_off = 0
    for bi in range(n_batches):
        idx = bi * batch_seqs
        batch_ws = all_windows[idx:idx + batch_seqs]
        batch_starts.append(batch_ws)
        score_start = flat_off
        for i, ws in enumerate(batch_ws):
            end = min(ws + seq_len, total_tokens)
            wl = end - ws
            s = 0 if ws == 0 else max(wl - stride, 0)
            if wl - s <= 0: continue
            gp_start = ws + s + 1
            gp_end = ws + wl
            if gp_end <= max_scored: continue
            n = gp_end - gp_start + 1
            all_bi[flat_off:flat_off+n] = i
            all_si[flat_off:flat_off+n] = np.arange(s, wl)
            all_gp[flat_off:flat_off+n] = np.arange(gp_start, gp_end + 1)
            flat_off += n
        batch_score_ranges.append((score_start, flat_off))
        if flat_off > score_start:
            max_scored = int(all_gp[flat_off - 1])
    return all_bi[:flat_off], all_si[:flat_off], all_gp[:flat_off], batch_starts, batch_score_ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="train_gpt_mlp35_mixed.py")
    parser.add_argument("--model", default="final_model.int6.ptz")
    parser.add_argument("--val-pattern", default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-seqs", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--base-beta", type=float, default=1.0)
    parser.add_argument("--agree-bonus", type=float, default=0.5)
    parser.add_argument("--within-threshold", type=float, default=0.25)
    parser.add_argument("--within-beta", type=float, default=0.55)
    parser.add_argument("--word-threshold", type=float, default=0.80)
    parser.add_argument("--word-beta", type=float, default=0.50)
    parser.add_argument("--open-table-bits", type=int, default=26)
    parser.add_argument("--token-threshold-scale", type=float, default=1.0)
    parser.add_argument("--order-stride", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=11)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=float, default=3.5)
    parser.add_argument("--logit-softcap", type=float, default=30.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--qk-gain-init", type=float, default=1.5)
    parser.add_argument("--bigram-vocab-size", type=int, default=3072)
    parser.add_argument("--bigram-dim", type=int, default=112)
    parser.add_argument("--xsa-last-n", type=int, default=11)
    parser.add_argument("--rope-dims", type=int, default=16)
    parser.add_argument("--ve-dim", type=int, default=128)
    parser.add_argument("--ve-layers", default="9,10")
    args = parser.parse_args()
    device = torch.device(args.device)
    t_wall = time.perf_counter()

    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gpt", args.code)
    tg = importlib.util.module_from_spec(spec); spec.loader.exec_module(tg)

    val_files = sorted(glob.glob(args.val_pattern)); assert val_files
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    if args.max_tokens > 0: val_tokens = val_tokens[:args.max_tokens + 1]
    total_tokens = val_tokens.numel() - 1
    print(f"Val tokens: {total_tokens:,}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    bb_lut, ls_lut, bd_lut = build_luts(sp, args.vocab_size, device)

    from fused_expert_ext import ContextMixer
    val_np = val_tokens.numpy().astype(np.int64)
    ngram = ContextMixer(
        base_beta=args.base_beta, agree_bonus=args.agree_bonus,
        within_threshold=args.within_threshold, within_beta=args.within_beta,
        word_threshold=args.word_threshold, word_beta=args.word_beta,
        open_table_bits=args.open_table_bits,
        token_threshold_scale=args.token_threshold_scale,
        order_stride=args.order_stride)
    ngram.set_tokens(val_np)
    ngram.set_luts(bb_lut.cpu().to(torch.int16).numpy(),
                   ls_lut.cpu().numpy().astype(np.uint8),
                   bd_lut.cpu().numpy().astype(np.uint8))

    seq_len, stride = args.seq_len, args.stride
    all_windows = [ws for ws in range(0, total_tokens, stride)
                   if min(ws + seq_len, total_tokens) - ws >= 1]

    # ── Start CPU precompute threads, then load model + compile on GPU ────
    all_hints = np.zeros(total_tokens + 1, dtype=np.int32)
    all_betas = np.zeros(total_tokens + 1, dtype=np.float64)
    positions = np.arange(1, total_tokens + 1, dtype=np.int64)
    idx_result = [None]

    def do_ngram():
        ngram.get_hints_batch(positions, all_hints[1:], all_betas[1:])
    def do_indices():
        idx_result[0] = precompute_batch_indices(
            all_windows, total_tokens, seq_len, stride, args.batch_seqs)

    ngram_thread = threading.Thread(target=do_ngram, daemon=True)
    idx_thread = threading.Thread(target=do_indices, daemon=True)
    ngram_thread.start()
    idx_thread.start()

    # GPU: load model + compile (overlaps with CPU threads)
    val_gpu = val_tokens.to(device=device, dtype=torch.int64)
    model = load_model(args, tg, device)
    compiled_logits = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    xb_static = torch.zeros(args.batch_seqs, seq_len, dtype=torch.int64, device=device)
    yb_static = torch.zeros(args.batch_seqs, seq_len, dtype=torch.int64, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(3): compiled_logits(xb_static)
    torch.cuda.synchronize()

    # Wait for CPU threads
    idx_thread.join()
    ngram_thread.join()
    all_bi_np, all_si_np, all_gp_np, batch_starts, batch_score_ranges = idx_result[0]
    n_batches = len(batch_starts)

    # Upload everything to GPU
    all_hints_gpu = torch.from_numpy(all_hints.astype(np.int64)).to(device)
    all_betas_gpu = torch.from_numpy(all_betas).to(device=device, dtype=torch.float64)
    all_bi_gpu = torch.from_numpy(all_bi_np).to(device)
    all_si_gpu = torch.from_numpy(all_si_np).to(device)
    all_gp_gpu = torch.from_numpy(all_gp_np).to(device)
    offsets_gpu = torch.arange(seq_len, device=device)

    print(f"Windows: {len(all_windows):,}, batches: {n_batches}")
    print(f"Setup: {time.perf_counter() - t_wall:.1f}s")

    gpu_loss = torch.zeros(1, dtype=torch.float64, device=device)
    gpu_tilt_loss = torch.zeros(1, dtype=torch.float64, device=device)
    gpu_bytes = torch.zeros(1, dtype=torch.float64, device=device)
    gpu_tokens = torch.zeros(1, dtype=torch.float64, device=device)
    gpu_tilted = torch.zeros(1, dtype=torch.float64, device=device)
    gpu_hits = torch.zeros(1, dtype=torch.float64, device=device)
    max_scored = 0
    t0 = time.perf_counter()

    with torch.inference_mode():
        for bi in range(n_batches):
            batch_ws = batch_starts[bi]
            bsz = len(batch_ws)
            sc_start, sc_end = batch_score_ranges[bi]
            if sc_end <= sc_start: continue

            ws_tensor = torch.tensor(batch_ws, device=device, dtype=torch.int64)
            indices = ws_tensor.unsqueeze(1) + offsets_gpu.unsqueeze(0)
            indices.clamp_(max=total_tokens)
            xb = xb_static[:bsz]; yb = yb_static[:bsz]
            xb[:] = val_gpu[indices]
            yb[:] = val_gpu[(indices + 1).clamp_(max=total_tokens)]

            bi_g = all_bi_gpu[sc_start:sc_end]
            si_g = all_si_gpu[sc_start:sc_end]
            gp_g = all_gp_gpu[sc_start:sc_end]
            hints_gpu = all_hints_gpu[gp_g]
            betas_gpu = all_betas_gpu[gp_g]

            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(xb)

            flat_logits = logits[bi_g, si_g].float()
            flat_targets = yb[bi_g, si_g]
            flat_prevs = xb[bi_g, si_g]
            flat_nll = F.cross_entropy(flat_logits, flat_targets, reduction="none").to(torch.float64)

            safe_hints = hints_gpu.clamp(min=0)
            logit_target = flat_logits.gather(-1, flat_targets.unsqueeze(-1)).squeeze(-1).to(torch.float64)
            logit_hint = flat_logits.gather(-1, safe_hints.unsqueeze(-1)).squeeze(-1).to(torch.float64)
            logsumexp = flat_nll + logit_target
            p_hint = (logit_hint - logsumexp).exp().clamp(0.0, 1.0)

            has_hint = (hints_gpu >= 0).to(torch.float64)
            Z = 1.0 + p_hint * (betas_gpu.exp() - 1.0)
            is_hit = (flat_targets == hints_gpu).to(torch.float64)
            mixed_nll = flat_nll + has_hint * (Z.log() - betas_gpu * is_hit)

            valid = gp_g > max_scored
            max_scored = int(gp_g[-1].item())
            v = valid.to(torch.float64)

            tb = bb_lut[flat_targets] + (ls_lut[flat_targets] & ~bd_lut[flat_prevs]).to(torch.float64)
            gpu_loss += (flat_nll * v).sum()
            gpu_tilt_loss += (mixed_nll * v).sum()
            gpu_bytes += (tb * v).sum()
            gpu_tokens += v.sum()
            gpu_tilted += (has_hint * v).sum()
            gpu_hits += (has_hint * is_hit * v).sum()

            if bi % 500 == 0:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                tc = gpu_tokens.item()
                if tc > 0:
                    bs = gpu_bytes.item()
                    tpb = tc / bs if bs > 0 else 1.0
                    b = (gpu_loss.item() / tc / math.log(2.0)) * tpb
                    t = (gpu_tilt_loss.item() / tc / math.log(2.0)) * tpb
                    print(f"  {bi/n_batches*100:5.1f}% | base:{b:.6f} tilt:{t:.6f} delta:{t-b:+.6f} | {elapsed:.0f}s")

    torch.cuda.synchronize()
    loop_time = time.perf_counter() - t0
    wall_time = time.perf_counter() - t_wall
    tc = gpu_tokens.item(); bs = gpu_bytes.item(); tpb = tc / bs
    base_bpb = (gpu_loss.item() / tc / math.log(2.0)) * tpb
    tilt_bpb = (gpu_tilt_loss.item() / tc / math.log(2.0)) * tpb
    nt = int(gpu_tilted.item()); nh = int(gpu_hits.item())
    print(f"\n{'='*72}")
    print(f"RESULTS  base_beta={args.base_beta}, stride={stride}, seq_len={seq_len}")
    print(f"{'='*72}")
    print(f"Neural only:  val_bpb = {base_bpb:.8f}")
    print(f"Tilted:       val_bpb = {tilt_bpb:.8f}")
    print(f"Delta:        {tilt_bpb - base_bpb:+.8f} BPB")
    print(f"Tokens: {int(tc):,} | Bytes: {bs:,.0f}")
    if nt > 0:
        print(f"Tilted: {nt:,} ({nt/tc*100:.1f}%) | Hits: {nh:,} ({nh/nt*100:.1f}%)")
    print(f"Loop: {loop_time:.1f}s | Wall: {wall_time:.1f}s")

if __name__ == "__main__":
    main()
