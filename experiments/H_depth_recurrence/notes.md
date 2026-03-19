# Experiment H: Depth Recurrence with LoRA Differentiation

## Hypothesis
5 physical blocks looped 6× = 30 virtual layers. Per-virtual-layer rank-3
LoRA adapters on Q and V let shared blocks specialize. Simulates much deeper
network without storing extra full layers. (Concept: PR #103; negative result
for naive recurrence: PR #31 got 1.2663, worse than baseline)

## Why LoRA Differentiation is Critical
- Naive weight sharing (PR #31): 1.2663 BPB (WORSE than baseline 1.2244)
- Massive quant gap with shared layers: 0.13 BPB (PR #102 ablation)
- Per-virtual-layer LoRA fixes both: 30 × rank-3 adapters = ~307K params
  but each virtual layer can specialize

## Architecture
- model_dim=768 (wider to fill budget with fewer physical blocks)
- 5 unique blocks, 6 passes each = 30 virtual depth
- num_heads=12, num_kv_heads=4
- mlp_mult=2
- LoRA rank=3 on Q and V per virtual layer (30 pairs)
- Per-virtual-layer scale parameter
- Encoder-decoder skip: first 15 virtual layers = encoder, last 15 = decoder

## Forward Pass
```
for i in range(30):
    block = blocks[i % 5]  # cycle physical blocks
    q = block.c_q(x) + lora_q[i](x)
    v = block.c_v(x) + lora_v[i](x)
    k = block.c_k(x)  # no LoRA on keys
    ...
```

## Risk Assessment
- HIGH RISK: No validated 8xH100 run exists
- The 6× forward pass makes each step ~6× slower
- With 768 dim, each step is even heavier
- May not converge in 10 minutes
- Quant gap for shared blocks is a known problem

## Changes from Experiment A
- Fundamentally different architecture
- GPT class rewritten with recurrence + LoRA
- Wider model (768 vs 512)
- Fewer physical blocks but deeper virtual depth

## Run Command
```bash
RUN_ID=exp_H_recur \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Success Metric
- post-quant val_bpb < 1.16
- artifact_bytes < 16,000,000
- Finishes within 600s wallclock

## Results
| Seed | pre-quant | post-quant | quant_gap | artifact_bytes | steps | status |
|------|-----------|------------|-----------|----------------|-------|--------|
|      |           |            |           |                |       |        |
