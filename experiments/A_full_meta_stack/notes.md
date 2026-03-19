# Experiment A: Full Meta Stack

## Hypothesis
Combining the 5 converged techniques from the top leaderboard submissions
will bring us from baseline 1.2244 to ~1.16 BPB. This is now table stakes.

## Techniques (all validated across 5+ independent PRs)
1. **Int6 per-row quantization** [-31, +31] stored in int8 bytes
2. **zstd-22 compression** (replaces zlib-9; exploits zero high bits in int6)
3. **MLP 3x expansion** (hidden=1536, funded by int6 savings)
4. **10 transformer layers** (extra layer fits due to int6)
5. **FP16 tied embedding passthrough** (most quant-sensitive tensor)
6. **Sliding window eval** (stride=64, each token gets ~960 context)
7. **Tuned Muon hyperparameters**:
   - matrix_lr=0.02, scalar_lr=0.02, tied_embed_lr=0.03
   - momentum=0.99, warmup from 0.92 over 1500 steps
   - warmdown=3000 iters
   - grad_clip=0.3
8. **TRAIN_SEQ_LEN=2048** (matches sliding window eval distribution)
9. **TRAIN_BATCH_TOKENS=786432** (larger batch for seq2048)

## Size Budget (estimated)
- 10 layers × 512dim × MLP 3x = ~21.8M params
- Int6 + zstd-22 → ~15.0-15.5 MB model
- FP16 embedding: 1024 × 512 × 2 bytes = ~1 MB (passthrough)
- Code: ~60 KB
- Total: ~15.1-15.6 MB (under 16 MB)

## Run Command
```bash
RUN_ID=exp_A_meta \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Success Metric
- post-quant val_bpb < 1.165

## Results
| Seed | pre-quant val_bpb | post-quant val_bpb | artifact_bytes | status |
|------|-------------------|--------------------|----------------|--------|
|      |                   |                    |                |        |
