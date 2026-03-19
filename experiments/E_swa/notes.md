# Experiment E: Stochastic Weight Averaging

## Hypothesis
Averaging 7 checkpoints during the warmdown phase produces smoother weights
that survive quantization better. Almost zero engineering cost for ~0.003-0.005
BPB improvement. (Validated: PR #89)

## Technique
During warmdown, when LR multiplier drops below 0.5:
1. Save model state dict every 200 steps
2. After training, compute uniform average of all saved checkpoints
3. Quantize the averaged model

## Config
- SWA_ENABLED=1 (default)
- SWA_START_FRAC=0.5 (start at 50% through warmdown)
- SWA_EVERY=200 (checkpoint every 200 steps)
- Expected: ~7 checkpoints with warmdown=3000

## Changes from Experiment A
- Add SWA collection logic in training loop
- Add averaging before quantization

## Run Command
```bash
RUN_ID=exp_E_swa \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Success Metric
- post-quant val_bpb < experiment A's post-quant val_bpb
- Quant gap should be smaller than experiment A

## Results
| Seed | pre-quant | post-quant | quant_gap | swa_checkpoints | status |
|------|-----------|------------|-----------|-----------------|--------|
|      |           |            |           |                 |        |
