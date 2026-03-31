#!/bin/bash
# kNN-LM hyperparameter sweep on 8 GPUs
# Runs multiple (lambda, temperature, normalize) configs sequentially

SCRIPT="experiments/eval_time_knn/scripts/eval_knn_distributed.py"
MODEL="/root/parameter-golf/final_model.pt"
COMMON="--k 64 --max-store 2000000 --stride 64 --batch 32 --out /root/results"

cd /root/parameter-golf

for NORM in "" "--normalize"; do
    for LAM in 0.01 0.05 0.10; do
        for TEMP in 1.0 10.0 100.0; do
            NORM_TAG=""
            if [ -n "$NORM" ]; then NORM_TAG="_norm"; fi
            TAG="knn_lam${LAM}_T${TEMP}${NORM_TAG}"
            echo "=== $TAG ==="
            PYTHONUNBUFFERED=1 TRAIN_SCRIPT=/root/parameter-golf/train_gpt.py \
                /root/venv/bin/torchrun --standalone --nproc_per_node=8 \
                $SCRIPT --model $MODEL --temperature $TEMP --lam $LAM $NORM $COMMON \
                2>&1 | grep -E "RESULT|model=|BPB|delta|Improvement|STARTED|ERROR" | tail -5
            echo ""
        done
    done
done
echo "=== SWEEP DONE ==="
ls -la /root/results/knn_dist_*.json
