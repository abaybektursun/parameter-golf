# Autoresearch System Architecture

## Overview

An autonomous AI agent (OpenAI Codex CLI) runs on a disposable spot GPU instance, continuously modifying `train_gpt.py`, training, evaluating, and keeping or discarding changes — with all state synced to GitHub and S3.

## Components

### Local (your Mac)
- **Repo**: `/Users/abaybektursun/projects/parameter-golf` — fork at `abaybektursun/parameter-golf`, remote via SSH
- **`init_remote.sh`** — one-shot setup script for any new GPU instance (gitignored, contains credentials)
- **S3 bucket**: `s3://fuelos-autoresearch/` (AWS profile `fuelos`) — backup of all remote state

### Remote (spot GPU instance)
- **Repo**: `/root/parameter-golf` — cloned from fork with PAT in URL (enables `git push`)
- **Codex CLI** — runs `codex exec --dangerously-bypass-approvals-and-sandbox` with `gpt-5.3-codex` at `xhigh` reasoning effort
- **Config**: `/root/.codex/config.toml` — model, reasoning effort, summaries
- **Auth**: `/root/.codex/auth.json` — OpenAI API key
- **Cron jobs**:
  - `* * * * *` — Sync repo to S3 (excludes `data/`, `.git/`, `.venv/`, `__pycache__/`)
  - `* * * * *` — Sync `/root/codex.log` to S3
  - `*/5 * * * *` — Watchdog: restarts Codex agent if it's not running (see `experiments/002_autoresearch/watchdog.sh`)

### Persistence (survives spot termination)
| What | Where | How |
|------|-------|-----|
| Code + git history | GitHub fork (`autoresearch/<tag>` branch) | Agent does `git push origin HEAD` after each successful experiment |
| Model artifacts, logs, results.tsv | `s3://fuelos-autoresearch/latest/` | Cron every 2 min + agent push after success |
| Codex agent log | `s3://fuelos-autoresearch/logs/codex.log` | Cron every 2 min |

## File Layout

```
experiments/002_autoresearch/
├── program.md          # Agent instructions (the "brain" — human edits this)
├── train_gpt.py        # Agent's working copy (agent edits this)
├── notes.md            # Experiment hypothesis + results table
├── results.tsv         # Tab-separated experiment log (untracked by git)
├── watchdog.sh         # Cron script that restarts Codex agent if it dies
├── daemon_loop.py      # Python daemon for automated hyperparameter sweeps
├── daemon_supervisor.sh # Keeps daemon_loop.py alive
└── SYSTEM.md           # This file
```

## Experiment Loop (defined in program.md)

```
LOOP FOREVER:
  1. Pick an idea (architecture, optimizer, quantization, hyperparams)
  2. Edit experiments/002_autoresearch/train_gpt.py
  3. git commit
  4. torchrun --standalone --nproc_per_node=$NUM_GPUS experiments/002_autoresearch/train_gpt.py > run.log 2>&1
  5. Extract: grep "final_int8_zlib_roundtrip_exact|Total submission size" run.log
  6. If crash → fix or revert, log "crash" in results.tsv
  7. Log results to results.tsv
  8. If improved AND artifact < 16MB:
     → keep commit
     → git push origin HEAD
     → aws s3 sync to S3
  9. If worse → git reset --hard
```

## Constraints (Parameter Golf challenge)
- **Artifact**: `final_model.int8.ptz` + code ≤ 16 MB
- **Compute**: 10 min wall clock on 8xH100 (leaderboard); agent uses whatever GPUs available
- **Metric**: post-quantization `val_bpb` (lower = better)
- **Eval**: `eval_val()` function in train_gpt.py is read-only ground truth

## Init Script (`init_remote.sh`)

Deploys to any fresh GPU instance in one command:
```bash
scp init_remote.sh root@<IP>:/root/ && ssh root@<IP> bash /root/init_remote.sh
```

Installs: AWS CLI v2, Node.js 22, Codex CLI, uv, Python 3.12 venv, PyTorch (cu128), training data, cron jobs, git+AWS+OpenAI auth.

## Launching the Agent

Manually:
```bash
ssh root@<IP>
cd /root/parameter-golf && source .venv/bin/activate
export OPENAI_API_KEY="..."
nohup codex exec --dangerously-bypass-approvals-and-sandbox \
  "Read experiments/002_autoresearch/program.md and kick off the experiment loop." \
  > /root/codex.log 2>&1 &
```

Or just let the watchdog cron handle it — it checks every 5 minutes and restarts the agent if it's not running:
```bash
# Install watchdog cron (copy watchdog.sh to /root/ first)
cp experiments/002_autoresearch/watchdog.sh /root/watchdog.sh
chmod +x /root/watchdog.sh
# Add to crontab alongside S3 sync jobs:
(crontab -l 2>/dev/null; echo "*/5 * * * * /root/watchdog.sh") | crontab -
```

## Monitoring

```bash
# Live agent log
ssh root@<IP> "tail -50 /root/codex.log"

# From S3 (if instance is dead)
aws s3 cp s3://fuelos-autoresearch/logs/codex.log - --profile fuelos | tail -50

# Experiment results (1xA100)
aws s3 cp s3://fuelos-autoresearch/latest/experiments/002_autoresearch/results.tsv - --profile fuelos

# Experiment results (8xH100)
aws s3 cp s3://fuelos-autoresearch/latest-8xh100/experiments/002_autoresearch/results.tsv - --profile fuelos

# Dashboard (pulls both sources, auto-refreshes)
cd /Users/abaybektursun/projects/parameter-golf && python3 dashboard.py
# Then open http://localhost:9090

# GPU utilization
ssh root@<IP> nvidia-smi
```
