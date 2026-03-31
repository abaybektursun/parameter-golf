# System Architecture

## Overview
Each machine runs one agent (Codex CLI) that autonomously experiments on `train_gpt.py`. Agents coordinate through shared DynamoDB tables to avoid duplicate work and build on each other's results.

## Repo Structure
```
parameter-golf/
├── agent/
│   ├── program.md       # Agent instructions (the brain)
│   ├── helpers.py       # DB, SOTA detection, log parsing, notifications
│   ├── watchdog.sh      # Cron: restart agent if dead
│   └── setup_db.py      # One-time: create DynamoDB tables
├── design.md            # Architecture doc
├── deep_research_architectures.md  # Knowledge base
├── train_gpt.py         # The single mutable file
├── data/                # Tokenizers and datasets
├── records/             # Reference SOTA implementations
├── init_remote.sh       # Bootstrap any new machine
└── requirements.txt
```

## Infrastructure
- **Database**: DynamoDB (tables: pg_sota, pg_experiments, pg_runs, pg_agents)
- **AWS profile**: `fuelos`
- **Logs/artifacts**: S3 `s3://fuelos-autoresearch/`
- **Notifications**: ntfy.sh
- **SOTA source**: PRs on `openai/parameter-golf`

## Deploying a New Agent
```bash
# 1. Clone repo
git clone git@github.com:abaybektursun/parameter-golf.git
cd parameter-golf

# 2. Setup venv + deps (using uv, never pip)
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# FA3 pre-built wheel for H100 — no compilation needed
uv pip install flash-attn-3 --extra-index-url https://download.pytorch.org/whl/cu128

# 3. Configure AWS credentials (fuelos profile)
aws configure --profile fuelos

# 4. Set env vars
export AGENT_ID="$(hostname)-$(date +%s)"
export PROVIDER="vast.ai"  # or runpod, aws, etc.
export NTFY_TOPIC="parameter-golf-alerts"

# 5. Install cron
chmod +x agent/watchdog.sh
(crontab -l 2>/dev/null; echo "*/5 * * * * AGENT_ID=$AGENT_ID /root/parameter-golf/agent/watchdog.sh") | crontab -

# 6. Start agent (or let watchdog do it)
agent/watchdog.sh
```

## Monitoring
```bash
# Agent log
tail -f /root/codex.log

# Training log (latest)
tail -f $(ls -t logs/*.txt | head -1)

# Is agent running?
pgrep -fa codex

# Is training running?
pgrep -fa train_gpt
```

## Cron Jobs
```
*/5 * * * * /root/parameter-golf/agent/watchdog.sh    # Restart dead agent
```
