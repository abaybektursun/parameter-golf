#!/bin/bash
# Run 8 high-leverage experiments in parallel, one per GPU
# Each experiment tests a fundamentally different idea (not a sweep)
set -e
cd /root/parameter-golf

SCRIPT="experiments/eval_time_mixing/scripts/eval_ngram.py"
MODEL="final_model.pt"
LOGS="experiments/eval_time_mixing/logs"
mkdir -p "$LOGS" experiments/eval_time_mixing/results/{exp5,exp6_s256,exp6_s2048,exp1}

echo "=== 8 PARALLEL EXPERIMENTS === $(date)"

# GPU 0: EXP-5 — Hash collision impact (1M buckets = high collision)
CUDA_VISIBLE_DEVICES=0 nohup python3 "$SCRIPT" --model "$MODEL" \
    --exp buckets_1048576 --out experiments/eval_time_mixing/results/exp5 \
    > "$LOGS/par_buckets_1M.log" 2>&1 &

# GPU 1: EXP-5 — Hash collision impact (64M buckets = low collision)
CUDA_VISIBLE_DEVICES=1 nohup python3 "$SCRIPT" --model "$MODEL" \
    --exp buckets_67108864 --out experiments/eval_time_mixing/results/exp5 \
    > "$LOGS/par_buckets_64M.log" 2>&1 &

# GPU 2: EXP-6 — Stride=256 WITHOUT cache (isolate overlap contribution)
CUDA_VISIBLE_DEVICES=2 nohup python3 "$SCRIPT" --model "$MODEL" \
    --exp baseline --stride 256 --out experiments/eval_time_mixing/results/exp6_s256 \
    > "$LOGS/par_baseline_s256.log" 2>&1 &

# GPU 3: EXP-6 — Stride=256 WITH cache (measure pure n-gram delta at same stride)
CUDA_VISIBLE_DEVICES=3 nohup python3 "$SCRIPT" --model "$MODEL" \
    --exp backoff_7 --stride 256 --out experiments/eval_time_mixing/results/exp6_s256 \
    > "$LOGS/par_backoff7_s256.log" 2>&1 &

# GPU 4: EXP-5 — Hash collision impact (256M buckets = near-zero collision)
CUDA_VISIBLE_DEVICES=4 nohup python3 "$SCRIPT" --model "$MODEL" \
    --exp buckets_268435456 --out experiments/eval_time_mixing/results/exp5 \
    > "$LOGS/par_buckets_256M.log" 2>&1 &

# GPU 5: EXP-6 — Stride=2048 WITHOUT cache (zero overlap, pure single-pass)
CUDA_VISIBLE_DEVICES=5 nohup python3 "$SCRIPT" --model "$MODEL" \
    --exp baseline --stride 2048 --out experiments/eval_time_mixing/results/exp6_s2048 \
    > "$LOGS/par_baseline_s2048.log" 2>&1 &

# GPU 6: EXP-6 — Stride=2048 WITH cache (n-gram with zero overlap)
CUDA_VISIBLE_DEVICES=6 nohup python3 "$SCRIPT" --model "$MODEL" \
    --exp backoff_7 --stride 2048 --out experiments/eval_time_mixing/results/exp6_s2048 \
    > "$LOGS/par_backoff7_s2048.log" 2>&1 &

# GPU 7: EXP-1 gap fill — order_7 with fixed alpha (to compare with existing results)
CUDA_VISIBLE_DEVICES=7 nohup python3 "$SCRIPT" --model "$MODEL" \
    --exp order_7 --out experiments/eval_time_mixing/results/exp1 \
    > "$LOGS/par_order_7.log" 2>&1 &

echo "8 experiments launched. Monitor with:"
echo "  for i in 0 1 2 3 4 5 6 7; do echo \"GPU\$i: \$(tail -1 $LOGS/par_*.log 2>/dev/null | head -1)\"; done"
