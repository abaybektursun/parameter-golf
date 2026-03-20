# Experiment: Combine Seq2048 + FP16 Embed + Sliding Window Eval

## Hypothesis

The top 3 improvements on the leaderboard are orthogonal (training, quantization, evaluation). Combining all three should yield ~1.17 BPB, beating the SOTA of 1.206 by ~0.036.

## Plan

- Start from baseline `train_gpt.py`
- Apply seq2048 changes: `TRAIN_SEQ_LEN=2048`, `TIED_EMBED_LR=0.04`, `MATRIX_LR=0.032`, `SCALAR_LR=0.032`
- Apply FP16 embed changes: keep `tok_emb.weight` in fp16 during quantization, shrink `MLP_HIDDEN=992` to stay under 16MB
- Apply sliding window eval: add `forward_logits()` method, `eval_val_sliding()` function, `EVAL_STRIDE=64`
- Success metric: post-quant val_bpb < 1.201 (beating SOTA by >0.005 nats)

## Observations

| Run | Seed | Steps | Pre-quant BPB | Post-quant BPB | Artifact Size | Notes |
|-----|------|-------|---------------|----------------|---------------|-------|
|     |      |       |               |                |               |       |

## Post-mortem

(pending results)
