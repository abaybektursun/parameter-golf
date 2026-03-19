# Experiment B: Straight-Through Estimator Quantization-Aware Training

## Hypothesis
Adding fake int6 quantize→dequantize in the forward pass during training
teaches the model weight distributions that survive post-training quantization.
Reduces quant gap from ~0.01 BPB to ~0.002 BPB. (Validated: PR #89)

## Technique
In CastedLinear.forward(), during training:
1. Compute per-row scale = quantile(|w|, 99.99984%) / 31
2. Fake-quantize: w_q = round(clamp(w / scale, -31, 31)) * scale
3. STE trick: w_out = w + (w_q - w).detach()
   - Forward sees w_q (quantized weights)
   - Backward flows through w (full precision)

## Tradeoff
~50ms/step overhead. With ~11,000 steps in 10 min, this costs ~550s of
additional compute → fewer total steps. The quality gain must outweigh.
PR #102 disabled STE because "54% step overhead outweighs quant gap reduction."
PR #89 kept it and achieved 1.1622 with SWA compensation.

## Changes from Experiment A
- Modified CastedLinear.forward() with STE fake-quantize
- ENV: STE_ENABLED=1 (default on), STE_ENABLED=0 to disable

## Run Command
```bash
RUN_ID=exp_B_ste \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Success Metric
- post-quant val_bpb < experiment A's post-quant val_bpb
- quant gap (pre-quant minus post-quant) < 0.003

## Results
| Seed | pre-quant | post-quant | quant_gap | artifact_bytes | status |
|------|-----------|------------|-----------|----------------|--------|
|      |           |            |           |                |        |
