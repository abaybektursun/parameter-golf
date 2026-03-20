#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/parameter-golf"
LOG="$ROOT/experiments/002_autoresearch/daemon_supervisor.log"
cd "$ROOT"
while true; do
  if ! pgrep -f "python3 experiments/002_autoresearch/daemon_loop.py" >/dev/null; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] daemon missing; starting" >> "$LOG"
    setsid -f bash -lc "cd $ROOT && exec python3 experiments/002_autoresearch/daemon_loop.py >> experiments/002_autoresearch/daemon.log 2>&1"
    sleep 2
  fi
  sleep 30
done
