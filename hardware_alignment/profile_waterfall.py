"""Wrap the training step with torch.profiler to get per-component breakdown.
Run via: torchrun --nproc_per_node=2 --standalone train_gpt_turbo_muon.py
Then post-process the trace.

Simpler approach: just run profiler on the fwd+bwd of a single step."""

import torch
import torch.distributed as dist
import os, sys

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    sys.path.insert(0, "/root")
    import train_gpt_turbo_muon as tgt

    H = tgt.Hyperparameters
    model = tgt.GPT(
        vocab_size=H.vocab_size, num_layers=H.num_layers, model_dim=H.model_dim,
        num_heads=H.num_heads, num_kv_heads=H.num_kv_heads, mlp_mult=int(H.mlp_mult),
        tie_embeddings=H.tie_embeddings, tied_embed_init_std=H.tied_embed_init_std,
        logit_softcap=H.logit_softcap, rope_base=H.rope_base, qk_gain_init=H.qk_gain_init,
        bigram_vocab_size=H.bigram_vocab_size, bigram_dim=H.bigram_dim,
        xsa_last_n=H.xsa_last_n, rope_dims=H.rope_dims, ln_scale=H.ln_scale,
        ve_enabled=H.ve_enabled, ve_dim=H.ve_dim, ve_layers=H.ve_layers,
    ).cuda()

    # No torch.compile — eager mode for clean kernel profiling
    B = H.train_batch_tokens // (H.train_seq_len * world_size)
    if rank == 0:
        print(f"Batch/GPU: {B}, seq_len: {H.train_seq_len}, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Warmup (torch.compile + triton compilation)
    for i in range(8):
        x = torch.randint(0, H.vocab_size, (B, H.train_seq_len + 1), device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        model.zero_grad(set_to_none=True)
        if rank == 0 and i % 2 == 0:
            print(f"  warmup {i}/8")
    torch.cuda.synchronize()
    if rank == 0:
        print("Warmup done, profiling 3 steps...")

    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
    ) as prof:
        for step in range(4):
            x = torch.randint(0, H.vocab_size, (B, H.train_seq_len + 1), device="cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x[:, :-1], x[:, 1:])
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            prof.step()

    if rank == 0:
        events = prof.key_averages()
        cuda_events = []
        for e in events:
            if e.cuda_time_total > 0:
                cuda_events.append((e.key, e.cuda_time_total / 1e3, e.count))
        cuda_events.sort(key=lambda x: -x[1])
        total_ms = sum(x[1] for x in cuda_events)

        # Per-kernel top 30
        print(f"\n{'='*90}")
        print(f"TOP CUDA KERNELS (total over 3 profiled steps)")
        print(f"{'='*90}")
        print(f"{'Kernel':<65} {'Total ms':>8} {'Cnt':>5} {'%':>5}")
        print("-" * 90)
        for name, ms, cnt in cuda_events[:30]:
            print(f"{name[:63]:<65} {ms:>8.2f} {cnt:>5} {ms/total_ms*100:>4.1f}%")

        # Category breakdown
        cats = {"GEMM": 0, "FlashAttn": 0, "Triton": 0, "NCCL": 0,
                "Elementwise": 0, "Copy": 0, "Other": 0}
        for name, ms, cnt in cuda_events:
            nl = name.lower()
            if any(k in nl for k in ["gemm", "cublas", "sm90_xmma", "sm80_xmma", "cutlass"]):
                cats["GEMM"] += ms
            elif any(k in nl for k in ["flash", "fmha"]):
                cats["FlashAttn"] += ms
            elif "triton" in nl:
                cats["Triton"] += ms
            elif any(k in nl for k in ["nccl", "allreduce", "reducescatter", "allgather"]):
                cats["NCCL"] += ms
            elif any(k in nl for k in ["elementwise", "vectorized", "pointwise", "reduce_kernel", "softmax", "norm", "cast", "fill", "zero"]):
                cats["Elementwise"] += ms
            elif any(k in nl for k in ["copy", "transpose", "permute", "memcpy", "memset"]):
                cats["Copy"] += ms
            else:
                cats["Other"] += ms

        print(f"\n{'='*50}")
        print(f"CATEGORY BREAKDOWN (per step = total/3)")
        print(f"{'='*50}")
        print(f"{'Category':<25} {'Per-step ms':>12} {'%':>6}")
        print("-" * 50)
        for cat, ms in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"{cat:<25} {ms/3:>12.2f} {ms/total_ms*100:>5.1f}%")
        print(f"{'TOTAL per step':<25} {total_ms/3:>12.2f}")
        print(f"\nNote: this is fwd+bwd only (no optimizer, no data loading)")

        prof.export_chrome_trace("/root/trace.json")
        print("Trace: /root/trace.json")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
