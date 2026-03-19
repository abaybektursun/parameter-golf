# Experiment G: LoRA Test-Time Training

## Hypothesis
Adapting rank-8 LoRA adapters per-document at eval time lets the model
temporarily specialize to each document's distribution. Exploits the
underutilized 10-minute eval budget. (Validated: PR #77, -0.003 BPB on
weak baseline; could be stronger on meta stack base)

## Technique
At eval time, for each document:
1. Initialize rank-8 LoRA adapters on Q and V projections (every block)
2. Split document into 256-token chunks within sliding windows
3. For each chunk: SCORE it first, then take 1 Adam gradient step
4. Reset LoRA + optimizer between documents (no cross-contamination)

## Architecture
- BatchedLinearLoRA: separate A/B per batch element
  - A: [bsz, rank, in_features] (kaiming init)
  - B: [bsz, out_features, rank] (zero init)
- Applied to: c_q and c_v on every block (NOT c_k, NOT MLP)
- Optimizer: Adam(lr=0.01, betas=(0.9, 0.95))

## Key Details
- Score-then-train ordering prevents data leakage
- Last chunk of each document is scored only (never trained on)
- Document boundaries found via BOS tokens
- Documents sorted by length for batching efficiency
- TTT_BATCH_SIZE=64 documents processed in parallel

## Eval Budget
- Baseline eval: ~16s
- Sliding window eval: ~70s
- TTT + sliding window: estimated ~300-400s (within 10-min limit)

## Changes from Experiment A
- Add BatchedLinearLoRA module
- Add BatchedTTTLoRA wrapper
- Add eval_val_ttt_lora() function
- Modified final evaluation to use TTT

## Run Command
```bash
RUN_ID=exp_G_ttt \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Success Metric
- post-quant val_bpb < experiment A's val_bpb (sliding window only)
- Eval time < 600 seconds

## Results
| Seed | sliding_only_bpb | ttt_bpb | ttt_delta | eval_time_s | status |
|------|------------------|---------|-----------|-------------|--------|
|      |                  |         |           |             |        |
