---
name: train-8xh100
description: Launch and monitor training runs on the 8×H100 server for parameter-golf experiments.
user_invocable: true
---

# Train on 8×H100 Server

**Server:** `root@86.38.238.38` — 8×H100 80GB SXM (IP changes frequently — always confirm with the user before connecting)
**SSH:** `ssh root@86.38.238.38`
**Stack:** PyTorch 2.9.1+cu128, CUDA 12.8, flash-attn 2.8.3 (FA3)

## Data & Scripts

- **Data:** `/root/parameter-golf/data/datasets/fineweb10B_sp1024/`
- **Best script (val-calibrated GPTQ):** `/root/parameter-golf/train_609_val_calib.py`
- **Other scripts:** `/root/parameter-golf/train_609.py`, `/root/parameter-golf/train_609_lzma9.py`
- **Experiment logs:** 157+ logs at `/root/log_*.txt`

## Pre-flight

Always clean up before launching a run:

```bash
# Kill stale python processes
ssh root@86.38.238.38 'ps aux | grep python | grep -v grep | grep -v unattended | awk "{print \$2}" | xargs -r kill -9'

# Verify GPUs are clean
ssh root@86.38.238.38 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

All 8 GPUs should show near-zero memory usage before launching.

## Launch Training

```bash
ssh root@86.38.238.38 'cd /root/parameter-golf && \
nohup env \
  BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
  WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=64 \
  TARGET_MB=15.9 SEED=314 \
  torchrun --standalone --nproc_per_node=8 train_609_val_calib.py \
  > /root/log_DESCRIPTION.txt 2>&1 &'
```

Replace `DESCRIPTION` with a meaningful name (e.g., `log_val_calib_seed314.txt`).

## Monitor

```bash
# Tail the log
ssh root@86.38.238.38 'tail -50 /root/log_DESCRIPTION.txt'

# Check GPU utilization
ssh root@86.38.238.38 'nvidia-smi'
```

## File Transfers

Always use `scp`, never `ssh cat` (vast.ai SSH banner corrupts stdout):

```bash
# Download a log
scp root@86.38.238.38:/root/log_DESCRIPTION.txt .

# Download an artifact
scp root@86.38.238.38:/root/parameter-golf/artifact.bin .

# Upload a script
scp local_script.py root@86.38.238.38:/root/parameter-golf/
```

## Key Rules

- **Never kill running training jobs** — GPU time is expensive, always let runs finish
- Use `scp` for file transfers, never `ssh cat`
- Use `nohup ... > /root/log_name.txt 2>&1 &` for background runs
- Use `uv` for Python packages, never `pip`
- Never do hyperparameter sweeps — only fundamental mechanistic/mathematical work
