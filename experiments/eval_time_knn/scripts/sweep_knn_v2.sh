#!/bin/bash
# Focused kNN-LM sweep: vary k and fine-tune lambda around the sweet spot
# Best so far: k=64, lambda=0.01, T=1.0 -> +0.0007 BPB

SCRIPT="experiments/eval_time_knn/scripts/eval_knn_distributed.py"
MODEL="/root/parameter-golf/final_model.pt"
COMMON="--max-store 2000000 --stride 64 --batch 32 --out /root/results"

cd /root/parameter-golf

# Phase 1: Vary k with best lambda/T
for K in 4 8 16 32 128; do
    TAG="knn_k${K}_lam0.01_T1.0"
    echo "=== $TAG ==="
    PYTHONUNBUFFERED=1 TRAIN_SCRIPT=/root/parameter-golf/train_gpt.py \
        /root/venv/bin/torchrun --standalone --nproc_per_node=8 \
        $SCRIPT --model $MODEL --k $K --temperature 1.0 --lam 0.01 $COMMON \
        2>&1 | grep -E "Improvement|BPB:|delta" | tail -3
    echo ""
done

# Phase 2: Fine lambda with best k (will use k=64 unless Phase 1 finds better)
for LAM in 0.002 0.005 0.008 0.015 0.02 0.03; do
    TAG="knn_k64_lam${LAM}_T1.0"
    echo "=== $TAG ==="
    PYTHONUNBUFFERED=1 TRAIN_SCRIPT=/root/parameter-golf/train_gpt.py \
        /root/venv/bin/torchrun --standalone --nproc_per_node=8 \
        $SCRIPT --model $MODEL --k 64 --temperature 1.0 --lam $LAM $COMMON \
        2>&1 | grep -E "Improvement|BPB:|delta" | tail -3
    echo ""
done

# Phase 3: Best configs with cosine normalization
for K in 8 64; do
    for LAM in 0.01 0.05; do
        TAG="knn_k${K}_lam${LAM}_T1.0_norm"
        echo "=== $TAG ==="
        PYTHONUNBUFFERED=1 TRAIN_SCRIPT=/root/parameter-golf/train_gpt.py \
            /root/venv/bin/torchrun --standalone --nproc_per_node=8 \
            $SCRIPT --model $MODEL --k $K --temperature 1.0 --lam $LAM --normalize $COMMON \
            2>&1 | grep -E "Improvement|BPB:|delta" | tail -3
        echo ""
    done
done

echo "=== SWEEP V2 DONE ==="
