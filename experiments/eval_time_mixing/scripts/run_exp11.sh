#!/bin/bash
# EXP-11: 8-GPU Global Cache with All-Reduce Sync
# COMPETITION CRITICAL — must complete under 600s
set -e
cd /root/parameter-golf

# Install flash attention 3 if not present
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null || \
    pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

SCRIPT="experiments/eval_time_mixing/scripts/eval_ngram_distributed.py"
LOG_DIR="experiments/eval_time_mixing/logs"
mkdir -p "$LOG_DIR"

echo "=== EXP-11: 8-GPU N-gram Eval === $(date)"

# Run 1: Baseline (no n-gram, 8 GPU) — timing reference
echo "--- Run 1: 8-GPU baseline (no n-gram) ---"
NGRAM_ENABLED=0 \
    torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
    2>&1 | tee "$LOG_DIR/exp11_baseline_8gpu.log"

# Run 2: 8-GPU with n-gram backoff 2-7, fixed alpha=0.40
echo "--- Run 2: 8-GPU + n-gram backoff 2-7 ---"
NGRAM_ENABLED=1 NGRAM_ORDER=7 NGRAM_ALPHA=0.40 \
    torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
    2>&1 | tee "$LOG_DIR/exp11_backoff7_8gpu.log"

# Run 3: 8-GPU with n-gram backoff 2-9, fixed alpha=0.40
echo "--- Run 3: 8-GPU + n-gram backoff 2-9 ---"
NGRAM_ENABLED=1 NGRAM_ORDER=9 NGRAM_ALPHA=0.40 \
    torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
    2>&1 | tee "$LOG_DIR/exp11_backoff9_8gpu.log"

# Run 4: Alpha sweep on backoff 2-7
for ALPHA in 0.20 0.30 0.50 0.60 0.80; do
    echo "--- Run 4: backoff 2-7 alpha=$ALPHA ---"
    NGRAM_ENABLED=1 NGRAM_ORDER=7 NGRAM_ALPHA=$ALPHA \
        torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
        2>&1 | tee "$LOG_DIR/exp11_backoff7_alpha${ALPHA}_8gpu.log"
done

echo "=== EXP-11 COMPLETE === $(date)"
