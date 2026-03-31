#!/usr/bin/env python3
"""kNN-LM eval: eval-time model growth with valid probability distributions.

GPU-optimized: all tensors stay on GPU. No FAISS — pure torch matmul for
distance computation. H100 does 990 TFLOPS bf16, so brute-force kNN on
GPU vastly outperforms any CPU approach.

Unlike n-gram caching, kNN-LM produces valid distributions that sum to 1:
  - Store hidden states from already-scored tokens on GPU
  - At each step, L2 distance via torch.cdist, top-k nearest neighbors
  - softmax(-distances/T) scattered into vocab → valid distribution
  - Interpolate with model distribution → also valid, sums to 1
"""

import os, sys, math, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
_train_script = Path(os.environ.get("TRAIN_SCRIPT", str(REPO / "train_gpt.py")))
_src = _train_script.read_text()
exec(compile(_src.split("\ndef main")[0], str(_train_script), "exec"))


class GPUDatastore:
    """Fixed-size GPU datastore for kNN-LM. All tensors on GPU."""

    def __init__(self, max_size, dim, vocab_size, device):
        self.max_size = max_size
        self.dim = dim
        self.vocab_size = vocab_size
        self.device = device
        self.keys = torch.zeros(max_size, dim, dtype=torch.float16, device=device)
        self.vals = torch.zeros(max_size, dtype=torch.int64, device=device)
        self.size = 0

    def add(self, hidden_states, next_tokens):
        """Add hidden states and their next-token IDs. Both must be on GPU."""
        n = hidden_states.shape[0]
        space = self.max_size - self.size
        if space <= 0:
            return
        n = min(n, space)
        self.keys[self.size:self.size + n] = hidden_states[:n].half()
        self.vals[self.size:self.size + n] = next_tokens[:n]
        self.size += n

    def search(self, queries, k, temperature):
        """Batch kNN search + distribution construction. All on GPU.

        Args:
            queries: [N, dim] float16/float32 on GPU
            k: number of neighbors
            temperature: softmax temperature

        Returns:
            knn_probs: [N, vocab_size] valid probability distribution
        """
        n_queries = queries.shape[0]
        store = self.keys[:self.size]  # [S, dim]

        # L2 distance via matmul: ||q - s||^2 = ||q||^2 + ||s||^2 - 2*q·s
        # More efficient than cdist for large stores
        q = queries.half()  # [N, dim]
        q_norm = (q * q).sum(dim=-1, keepdim=True)  # [N, 1]
        s_norm = (store * store).sum(dim=-1, keepdim=True).T  # [1, S]
        dots = q @ store.T  # [N, S]
        dists = q_norm + s_norm - 2 * dots  # [N, S]

        # Top-k nearest (smallest distance)
        actual_k = min(k, self.size)
        neg_dists_topk, indices = (-dists).topk(actual_k, dim=-1)  # [N, k]
        topk_dists = -neg_dists_topk  # [N, k] positive distances

        # Softmax over distances → neighbor weights
        weights = F.softmax(-topk_dists.float() / temperature, dim=-1)  # [N, k]

        # Scatter into vocab distribution
        neighbor_tokens = self.vals[indices]  # [N, k]
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
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="./results")
    args = parser.parse_args()

    device = torch.device("cuda")
    hp = Hyperparameters()
    seq_len = getattr(hp, "eval_seq_len", hp.train_seq_len)
    model_dim = hp.model_dim
    vocab_size = hp.vocab_size

    # Load model
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
    print(f"Model loaded: {model_dim}d, {hp.num_layers}L")

    # Forward: returns (hidden_states, logits) — no torch.compile to avoid tuple issues
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

    # Load validation data
    val_tokens = load_validation_tokens(hp.val_files, seq_len)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, hp.vocab_size, device
    )
    n_tokens = val_tokens.numel() - 1
    print(f"Val tokens: {n_tokens:,}")

    # GPU datastore
    # Memory: max_store * dim * 2 bytes (fp16) + max_store * 8 bytes (int64)
    store_mem_gb = args.max_store * (model_dim * 2 + 8) / 1e9
    print(f"Datastore: max {args.max_store:,} vectors, ~{store_mem_gb:.1f} GB GPU RAM")
    datastore = GPUDatastore(args.max_store, model_dim, vocab_size, device)

    # Eval
    model_nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    knn_nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    total_tokens_val = val_tokens.numel() - 1
    window_starts = list(range(0, total_tokens_val, args.stride))
    window_starts = [ws for ws in window_starts if min(ws + seq_len, total_tokens_val) - ws >= 1]
    n_windows = len(window_starts)

    print(f"kNN-LM: k={args.k} T={args.temperature} lam={args.lam}")
    print(f"Windows: {n_windows:,}, stride={args.stride}, batch={args.batch}")

    t0 = time.time()

    with torch.inference_mode():
        for bi in range(0, n_windows, args.batch):
            batch_ws = window_starts[bi:bi + args.batch]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens_val)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden, logits = forward_hidden(x_batch)

            # Collect all scored positions for batch kNN search
            all_hidden = []
            all_targets = []
            all_model_logits = []
            seg_info = []  # (start_idx, count) for reassembly

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - args.stride, 0)
                seg_len = wlen - s
                if seg_len <= 0:
                    seg_info.append((len(all_targets), 0))
                    continue

                seg_info.append((sum(si[1] for si in seg_info) if seg_info else 0, seg_len))
                all_hidden.append(hidden[i, s:wlen, :])
                all_targets.append(y_batch[i, s:wlen])
                all_model_logits.append(logits[i, s:wlen, :])

                # Byte counting
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            if not all_hidden:
                continue

            all_hidden_t = torch.cat(all_hidden, dim=0).float()  # [total_scored, dim]
            all_targets_t = torch.cat(all_targets, dim=0)        # [total_scored]
            all_logits_t = torch.cat(all_model_logits, dim=0).float()  # [total_scored, vocab]
            total_scored = all_hidden_t.shape[0]

            # Model NLL
            seg_model_nll = F.cross_entropy(all_logits_t, all_targets_t, reduction="none")
            model_nll_sum += seg_model_nll.sum()
            token_count += total_scored

            # kNN search (single batched call for all scored positions)
            if datastore.size >= args.k:
                knn_probs = datastore.search(all_hidden_t, args.k, args.temperature)
                model_probs = F.softmax(all_logits_t, dim=-1)

                # Interpolate: both sum to 1
                final_probs = (1 - args.lam) * model_probs + args.lam * knn_probs
                final_nll = -torch.log(
                    final_probs[torch.arange(total_scored, device=device), all_targets_t].clamp(min=1e-12)
                )
                knn_nll_sum += final_nll.sum()
            else:
                knn_nll_sum += seg_model_nll.sum()

            # Add to datastore AFTER scoring (strict causality)
            datastore.add(all_hidden_t, all_targets_t)

            # Progress
            done = min(bi + args.batch, n_windows)
            if done % (args.batch * 50) == 0 or done >= n_windows:
                elapsed = time.time() - t0
                tc = token_count.item()
                bc = max(byte_count.item(), 1)
                tpb = tc / bc
                mbpb = (model_nll_sum.item() / tc) / math.log(2) * tpb
                kbpb = (knn_nll_sum.item() / tc) / math.log(2) * tpb
                store_mb = datastore.size * model_dim * 2 / 1e6
                rate = done / elapsed
                eta = (n_windows - done) / rate if rate > 0 else 0
                print(f"  [{done/n_windows*100:5.1f}%] model={mbpb:.4f} knn={kbpb:.4f} "
                      f"delta={mbpb-kbpb:+.4f} store={datastore.size:,}({store_mb:.0f}MB) "
                      f"{elapsed:.0f}s eta={eta:.0f}s")

    tc = token_count.item()
    bc = byte_count.item()
    tpb = tc / bc
    model_bpb = (model_nll_sum.item() / tc) / math.log(2) * tpb
    knn_bpb = (knn_nll_sum.item() / tc) / math.log(2) * tpb
    store_mb = datastore.size * model_dim * 2 / 1e6

    print(f"\n{'='*60}")
    print(f"  Model-only BPB:      {model_bpb:.4f}")
    print(f"  kNN-LM BPB:          {knn_bpb:.4f}")
    print(f"  Improvement:         {model_bpb - knn_bpb:+.4f} BPB")
    print(f"  Datastore:           {datastore.size:,} vectors ({store_mb:.0f} MB)")
    print(f"  Distribution valid:  YES (both components sum to 1)")
    print(f"  Causality:           STRICT (score-first)")
    print(f"{'='*60}")

    results = {
        "model_bpb": model_bpb,
        "knn_bpb": knn_bpb,
        "improvement_bpb": model_bpb - knn_bpb,
        "datastore_vectors": datastore.size,
        "datastore_mb": store_mb,
        "k": args.k,
        "temperature": args.temperature,
        "lambda": args.lam,
        "distribution_valid": True,
        "causality": "score-first",
        "eval_time_s": time.time() - t0,
    }

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"knn_k{args.k}_T{args.temperature}_lam{args.lam}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
