#!/usr/bin/env python3
"""Distributed kNN-LM eval on 8 GPUs.

Each GPU processes 1/8 of windows, builds a local datastore, searches locally.
All-reduce final NLL sums for BPB. No cross-GPU kNN communication needed —
each GPU's local store is sufficient for demonstrating valid-distribution improvement.

Usage:
    torchrun --standalone --nproc_per_node=8 eval_knn_distributed.py --model final_model.pt
"""

import os, sys, math, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
_train_script = Path(os.environ.get("TRAIN_SCRIPT", str(REPO / "train_gpt.py")))
_src = _train_script.read_text()
exec(compile(_src.split("\ndef main")[0], str(_train_script), "exec"))


class GPUDatastore:
    """Fixed-size GPU datastore. All tensors on device."""

    def __init__(self, max_size, dim, vocab_size, device, normalize=False):
        self.max_size = max_size
        self.dim = dim
        self.vocab_size = vocab_size
        self.device = device
        self.normalize = normalize
        self.keys = torch.zeros(max_size, dim, dtype=torch.float16, device=device)
        self.vals = torch.zeros(max_size, dtype=torch.int64, device=device)
        self.size = 0

    def add(self, hidden_states, next_tokens):
        n = hidden_states.shape[0]
        space = self.max_size - self.size
        if space <= 0:
            return
        n = min(n, space)
        h = hidden_states[:n].half()
        if self.normalize:
            h = F.normalize(h, dim=-1)
        self.keys[self.size:self.size + n] = h
        self.vals[self.size:self.size + n] = next_tokens[:n]
        self.size += n

    def search(self, queries, k, temperature):
        n_queries = queries.shape[0]
        store = self.keys[:self.size]
        q = queries.half()
        if self.normalize:
            q = F.normalize(q, dim=-1)
        q_norm = (q * q).sum(dim=-1, keepdim=True)
        s_norm = (store * store).sum(dim=-1, keepdim=True).T
        dots = q @ store.T
        dists = q_norm + s_norm - 2 * dots
        actual_k = min(k, self.size)
        neg_dists_topk, indices = (-dists).topk(actual_k, dim=-1)
        topk_dists = -neg_dists_topk
        weights = F.softmax(-topk_dists.float() / temperature, dim=-1)
        neighbor_tokens = self.vals[indices]
        knn_probs = torch.zeros(n_queries, self.vocab_size, device=self.device)
        knn_probs.scatter_add_(1, neighbor_tokens, weights)
        return knn_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--lam", type=float, default=0.25)
    parser.add_argument("--max-store", type=int, default=2_000_000)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize hidden states (cosine similarity)")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="./results")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    hp = Hyperparameters()
    seq_len = getattr(hp, "eval_seq_len", hp.train_seq_len)
    model_dim = hp.model_dim
    vocab_size = hp.vocab_size

    # Load model on each GPU
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    import inspect
    gpt_sig = inspect.signature(GPT.__init__).parameters
    gpt_kwargs = dict(
        vocab_size=vocab_size, num_layers=hp.num_layers, model_dim=model_dim,
        num_heads=hp.num_heads, num_kv_heads=hp.num_kv_heads, mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings, tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap, rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
    )
    gpt_kwargs = {k: v for k, v in gpt_kwargs.items() if k in gpt_sig}
    model = GPT(**gpt_kwargs).to(device)  # keep float32 weights, use autocast for bf16 compute
    model.load_state_dict(state, strict=False)
    model.eval()

    @torch.no_grad()
    def forward_hidden(input_ids):
        x = model.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(model.num_encoder_layers):
            x = model.blocks[i](x, x0)
            skips.append(x)
        for i in range(model.num_decoder_layers):
            if skips:
                x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = model.blocks[model.num_encoder_layers + i](x, x0)
        x = model.final_norm(x)
        if model.tie_embeddings:
            logits = F.linear(x, model.tok_emb.weight)
        else:
            logits = model.lm_head(x)
        logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
        return x, logits

    # Load val data
    val_tokens = load_validation_tokens(hp.val_files, seq_len)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, hp.vocab_size, device
    )
    total_tokens_val = val_tokens.numel() - 1

    # Window assignment: each GPU gets contiguous chunk for better local kNN quality
    all_windows = [ws for ws in range(0, total_tokens_val, args.stride)
                   if min(ws + seq_len, total_tokens_val) - ws >= 1]
    n_total = len(all_windows)
    chunk = n_total // world_size
    start = rank * chunk
    end = n_total if rank == world_size - 1 else start + chunk
    my_windows = all_windows[start:end]

    if rank == 0:
        store_mb = args.max_store * model_dim * 2 / 1e6
        print(f"kNN-LM distributed: {world_size} GPUs, k={args.k} T={args.temperature} lam={args.lam}")
        print(f"Total windows: {n_total:,}, per GPU: ~{len(my_windows):,}")
        print(f"Datastore per GPU: max {args.max_store:,} vectors (~{store_mb:.0f} MB)")

    datastore = GPUDatastore(args.max_store, model_dim, vocab_size, device, normalize=args.normalize)

    model_nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    knn_nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    t0 = time.time()
    n_my_windows = len(my_windows)

    with torch.inference_mode():
        for bi in range(0, n_my_windows, args.batch):
            batch_ws = my_windows[bi:bi + args.batch]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []

            for i, ws in enumerate(batch_ws):
                wend = min(ws + seq_len, total_tokens_val)
                wlen = wend - ws
                wlens.append(wlen)
                chunk_tok = val_tokens[ws:wend + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk_tok[:-1]
                y_batch[i, :wlen] = chunk_tok[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden, logits = forward_hidden(x_batch)

            all_hidden = []
            all_targets = []
            all_logits = []

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - args.stride, 0)
                seg_len = wlen - s
                if seg_len <= 0:
                    continue

                all_hidden.append(hidden[i, s:wlen, :])
                all_targets.append(y_batch[i, s:wlen])
                all_logits.append(logits[i, s:wlen, :])

                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            if not all_hidden:
                continue

            all_hidden_t = torch.cat(all_hidden, dim=0).float()
            all_targets_t = torch.cat(all_targets, dim=0)
            all_logits_t = torch.cat(all_logits, dim=0).float()
            total_scored = all_hidden_t.shape[0]

            seg_model_nll = F.cross_entropy(all_logits_t, all_targets_t, reduction="none")
            model_nll_sum += seg_model_nll.sum()
            token_count += total_scored

            if datastore.size >= args.k:
                knn_probs = datastore.search(all_hidden_t, args.k, args.temperature)
                model_probs = F.softmax(all_logits_t, dim=-1)
                final_probs = (1 - args.lam) * model_probs + args.lam * knn_probs
                final_nll = -torch.log(
                    final_probs[torch.arange(total_scored, device=device), all_targets_t].clamp(min=1e-12)
                )
                knn_nll_sum += final_nll.sum()
            else:
                knn_nll_sum += seg_model_nll.sum()

            datastore.add(all_hidden_t, all_targets_t)

            done = min(bi + args.batch, n_my_windows)
            if rank == 0 and (done % (args.batch * 50) == 0 or done >= n_my_windows):
                elapsed = time.time() - t0
                tc = token_count.item()
                bc = max(byte_count.item(), 1)
                tpb = tc / bc
                mbpb = (model_nll_sum.item() / tc) / math.log(2) * tpb
                kbpb = (knn_nll_sum.item() / tc) / math.log(2) * tpb
                store_mb = datastore.size * model_dim * 2 / 1e6
                rate = done / elapsed
                eta = (n_my_windows - done) / rate if rate > 0 else 0
                print(f"  [{done/n_my_windows*100:5.1f}%] model={mbpb:.4f} knn={kbpb:.4f} "
                      f"delta={mbpb-kbpb:+.4f} store={datastore.size:,}({store_mb:.0f}MB) "
                      f"{elapsed:.0f}s eta={eta:.0f}s")

    # All-reduce metrics across GPUs
    dist.all_reduce(model_nll_sum)
    dist.all_reduce(knn_nll_sum)
    dist.all_reduce(token_count)
    dist.all_reduce(byte_count)

    if rank == 0:
        tc = token_count.item()
        bc = byte_count.item()
        tpb = tc / bc
        model_bpb = (model_nll_sum.item() / tc) / math.log(2) * tpb
        knn_bpb = (knn_nll_sum.item() / tc) / math.log(2) * tpb
        total_store = datastore.size * world_size  # approx total across GPUs
        total_store_mb = total_store * model_dim * 2 / 1e6
        elapsed = time.time() - t0

        print(f"\n{'='*60}")
        print(f"  GPUs:                {world_size}")
        print(f"  Model-only BPB:      {model_bpb:.4f}")
        print(f"  kNN-LM BPB:          {knn_bpb:.4f}")
        print(f"  Improvement:         {model_bpb - knn_bpb:+.4f} BPB")
        print(f"  Store per GPU:       {datastore.size:,} vectors")
        print(f"  Total store:         ~{total_store:,} vectors (~{total_store_mb:.0f} MB)")
        print(f"  Distribution valid:  YES (both components sum to 1)")
        print(f"  Causality:           STRICT (score-first)")
        print(f"  Wall time:           {elapsed:.0f}s")
        print(f"{'='*60}")

        results = {
            "model_bpb": model_bpb,
            "knn_bpb": knn_bpb,
            "improvement_bpb": model_bpb - knn_bpb,
            "gpus": world_size,
            "store_per_gpu": datastore.size,
            "store_per_gpu_mb": datastore.size * model_dim * 2 / 1e6,
            "total_store_vectors": total_store,
            "total_store_mb": total_store_mb,
            "k": args.k,
            "temperature": args.temperature,
            "lambda": args.lam,
            "distribution_valid": True,
            "causality": "score-first",
            "eval_time_s": elapsed,
        }

        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, f"knn_dist_{world_size}gpu_k{args.k}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
