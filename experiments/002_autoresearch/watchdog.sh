#!/bin/bash
if pgrep -f "codex exec" > /dev/null; then
  exit 0
fi

cd /root/parameter-golf
source .venv/bin/activate
export OPENAI_API_KEY="$OPENAI_API_KEY"

BEST=$(grep "keep" experiments/002_autoresearch/results.tsv | awk -F"\t" "{print \$3}" | sort -n | head -1)
TRIES=$(wc -l < experiments/002_autoresearch/results.tsv)
BRANCH=$(git branch --show-current)

nohup /usr/bin/codex exec --dangerously-bypass-approvals-and-sandbox "You are continuing an autoresearch experiment loop. Read experiments/002_autoresearch/program.md for full instructions.

Current state:
- Branch: $BRANCH
- Best post-quant val_bpb so far: $BEST
- $TRIES experiments completed so far (see results.tsv)
- Check results.tsv and git log to see what has been tried

CRITICAL: LOOP FOREVER. Never stop. The human is asleep." > /root/codex.log 2>&1 &
