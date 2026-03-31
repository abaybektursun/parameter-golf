#!/bin/bash
# EXP-0: Baseline Reproduction
# Run all 5 baseline configs sequentially
set -e
cd /root/parameter-golf

MODEL="/root/parameter-golf/final_model.pt"
OUT="/root/parameter-golf/experiments/eval_time_mixing/results/exp0"
SCRIPT="/root/parameter-golf/experiments/eval_time_mixing/scripts/eval_ngram.py"

mkdir -p "$OUT"

echo "=== EXP-0: Baseline Reproduction ==="
echo "Started: $(date)"

# 0a: Neural only
python3 "$SCRIPT" --model "$MODEL" --exp baseline --out "$OUT" 2>&1 | tee "$OUT/baseline.log"

# 0b: Fixed 7-gram
python3 "$SCRIPT" --model "$MODEL" --exp fixed_7gram --out "$OUT" 2>&1 | tee "$OUT/fixed_7gram.log"

# 0c: Backoff 2-7, fixed alpha
python3 "$SCRIPT" --model "$MODEL" --exp backoff_7 --out "$OUT" 2>&1 | tee "$OUT/backoff_7.log"

# 0d: Backoff 2-7, entropy-adaptive
python3 "$SCRIPT" --model "$MODEL" --exp backoff_7_ent --out "$OUT" 2>&1 | tee "$OUT/backoff_7_ent.log"

# 0e: Backoff 2-9, order-adaptive entropy
python3 "$SCRIPT" --model "$MODEL" --exp backoff_9_ent_oadapt --out "$OUT" 2>&1 | tee "$OUT/backoff_9_ent_oadapt.log"

echo "=== EXP-0 COMPLETE: $(date) ==="
