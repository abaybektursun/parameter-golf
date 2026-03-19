# Experiment: Autoresearch — Autonomous Agent Loop for Parameter Golf

## Hypothesis

An autonomous AI agent running a tight experiment loop (modify train_gpt.py -> train -> measure -> keep/discard -> repeat) can systematically explore the hyperparameter and architecture space faster than manual iteration. Over ~100 experiments overnight, the agent should find improvements that beat the current SOTA of 1.206 val_bpb.

## Plan

- Adapt Karpathy's autoresearch `program.md` to parameter-golf constraints:
  - Target: lowest post-quant val_bpb under 16MB artifact + 10 min on 8xH100s
  - Agent modifies only `train_gpt.py` (or a working copy)
  - Each experiment: full train run, quantize, measure post-quant val_bpb
  - Keep if improved, revert if not
- Launch the agent with `program.md` as its instruction set
- Agent logs all experiments to `results.tsv`
- Success metric: post-quant val_bpb < 1.206 (beating current SOTA)

## Observations

| Run | Seed | Steps | Pre-quant BPB | Post-quant BPB | Artifact Size | Notes |
|-----|------|-------|---------------|----------------|---------------|-------|
|     |      |       |               |                |               |       |

## Post-mortem

(pending results)
