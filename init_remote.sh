#!/bin/bash
# Parameter Golf — Remote GPU Instance Init Script
# Usage: scp init_remote.sh root@<IP>:/root/ && ssh root@<IP> bash /root/init_remote.sh
set -euo pipefail

# --- Credentials (loaded from .env) ---
source "$(dirname "$0")/.env"

# --- System deps ---
apt-get update && apt-get install -y unzip python3.12-dev
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install
rm -rf /tmp/awscliv2.zip /tmp/aws

# --- AWS credentials (fuelos profile) ---
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[fuelos]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
cat > ~/.aws/config << EOF
[profile fuelos]
region = us-east-1
EOF

# --- Clone repo ---
cd /root
git clone https://${GH_TOKEN}@github.com/abaybektursun/parameter-golf.git
cd parameter-golf

# --- Git identity ---
git config user.email "abaybektursun@users.noreply.github.com"
git config user.name "abaybektursun"

# --- Node.js + Codex CLI ---
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
npm i -g @openai/codex

# --- Install uv ---
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# --- Python deps ---
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

# --- Verify CUDA ---
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# --- Download training data ---
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# --- Codex CLI auth ---
mkdir -p /root/.codex
cat > /root/.codex/auth.json << EOF
{
  "OPENAI_API_KEY": "$OPENAI_API_KEY"
}
EOF

# --- S3 sync cron (every 2 min) ---
(crontab -l 2>/dev/null; echo "*/2 * * * * cd /root/parameter-golf && /usr/local/bin/aws s3 sync . s3://fuelos-autoresearch/latest/ --profile fuelos --exclude 'data/*' --exclude '.git/*' --exclude '__pycache__/*' --exclude '.venv/*'"; echo "*/2 * * * * /usr/local/bin/aws s3 cp /root/codex.log s3://fuelos-autoresearch/logs/codex.log --profile fuelos") | crontab -

# --- Summary ---
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo ""
echo "=== Setup complete ==="
echo "GPUs: $NUM_GPUS"
echo "Repo: /root/parameter-golf"
echo "S3 backup: aws s3 sync . s3://fuelos-autoresearch/ --profile fuelos --exclude 'data/*' --exclude '.git/*' --exclude '__pycache__/*'"
echo "Train: torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py"
echo "Autoresearch: OPENAI_API_KEY=\$OPENAI_API_KEY codex exec --dangerously-bypass-approvals-and-sandbox 'Read experiments/002_autoresearch/program.md and kick off the experiment loop'"
