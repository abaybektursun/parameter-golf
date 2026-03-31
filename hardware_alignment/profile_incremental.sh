#!/bin/bash
# Profile incremental improvements on 2xH100
# Runs each config for 30 steps and captures step times

set -e

export DATA_PATH=/root/data/datasets/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/root/data/datasets/tokenizers/fineweb_1024_bpe.model
export SEED=42
export MAX_WALLCLOCK_SECONDS=9999
export ITERATIONS=30
export VAL_LOSS_EVERY=9999
export TRAIN_LOG_EVERY=5

echo "============================================"
echo "Config 1: Baseline (fused_mlp - original PR 1105 Triton fwd+bwd)"
echo "============================================"
# Use the original fused_mlp script (before our act_grad precomp changes)
# Actually train_gpt_fused_mlp.py already has our changes.
# To get a true baseline we'd need the original. Let's just time what we have
# and compare the kernel-level benchmarks.

cd /root
torchrun --nproc_per_node=2 --standalone train_gpt_fused_mlp.py 2>&1 | grep -E "step|Step|ms|time|bpb|BPB" | head -30

echo ""
echo "============================================"
echo "Config 2: Our best (turbo_muon - Triton fwd + CUTLASS EVT bwd)"
echo "============================================"

torchrun --nproc_per_node=2 --standalone train_gpt_turbo_muon.py 2>&1 | grep -E "step|Step|ms|time|bpb|BPB" | head -30

echo ""
echo "============================================"
echo "Kernel-level benchmarks"
echo "============================================"
cd /root/cutlass_evt_fusion
python bench_final.py 2>&1 | grep -v Warning | grep -v cpu
