#!/bin/bash
# Cron watchdog: sends heartbeat, restarts agent if dead.
# Install: */5 * * * * $HOME/parameter-golf/agent/watchdog.sh

REPO_DIR="${REPO_DIR:-$HOME/parameter-golf}"
LOG_FILE="${LOG_FILE:-$HOME/agent.log}"

cd "$REPO_DIR"
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
unset ANTHROPIC_API_KEY  # force OAuth auth, not API key

# Stable agent ID
AGENT_ID_FILE="$HOME/.agent_id"
if [ ! -f "$AGENT_ID_FILE" ]; then
  echo "$(hostname)-$(cat /proc/sys/kernel/random/uuid | head -c 8)" > "$AGENT_ID_FILE"
fi
export AGENT_ID="$(cat $AGENT_ID_FILE)"
export PROVIDER="${PROVIDER:-local}"
export NTFY_TOPIC="${NTFY_TOPIC:-parameter-golf-alerts}"

# Always send heartbeat (offloads this from the agent)
python3 -c "
import sys, os; sys.path.insert(0, '$REPO_DIR')
os.environ['AGENT_ID'] = '$AGENT_ID'
from agent.helpers import heartbeat, detect_machine_config
machine = detect_machine_config()
heartbeat('$AGENT_ID', machine)
" 2>/dev/null

# Restart agent if dead
if pgrep -f "claude.*program.md" > /dev/null; then
  exit 0
fi

nohup claude --dangerously-skip-permissions \
  -p "You are a parameter-golf optimization agent. Read agent/program.md for your full instructions.

Agent ID: $AGENT_ID
CRITICAL: LOOP FOREVER. Never stop." \
  --verbose --output-format stream-json > "$LOG_FILE" 2>&1 &
