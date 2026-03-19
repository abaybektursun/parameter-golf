# Experiment F: SmearGate + Multi-Token Prediction Auxiliary Head

## Hypothesis
Two cheap additions that improve training signal without artifact cost:
1. SmearGate: bigram info for 512 params (validated: PR #102, ~0.003 BPB)
2. MTP head: richer gradients during training, excluded from artifact (validated: PR #88)

## SmearGate
```python
class SmearGate(nn.Module):
    def __init__(self, dim):
        self.gate = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        g = sigmoid(gate)[None, None, :]
        x_prev = cat([zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
```
- Applied after embedding + RMSNorm, before transformer blocks
- 512 parameters, initialized at sigmoid(0) = 0.5
- Optimized with Adam at SCALAR_LR
- Passthrough in quantization (control tensor)

## MTP Head
- 1 auxiliary linear head: model_dim → vocab_size (512 × 1024 = 524K params)
- Predicts token at position i+2 from hidden state at position i
- Loss: main_loss + 0.1 * mtp_loss
- Training only (guarded by self.training)
- Excluded from artifact at export (filtered by name)

## Changes from Experiment A
- Add SmearGate module
- Add MTP head and modified loss
- Export function strips MTP weights

## Run Command
```bash
RUN_ID=exp_F_smear_mtp \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Success Metric
- post-quant val_bpb < experiment A's post-quant val_bpb

## Results
| Seed | pre-quant | post-quant | artifact_bytes | status |
|------|-----------|------------|----------------|--------|
|      |           |            |                |        |
