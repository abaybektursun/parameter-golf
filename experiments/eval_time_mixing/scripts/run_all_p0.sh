#!/bin/bash
# P0 Experiments — sequential, one at a time, no pipes
cd /root/parameter-golf

MODEL="/root/parameter-golf/final_model.pt"
SCRIPT="/root/parameter-golf/experiments/eval_time_mixing/scripts/eval_ngram.py"
RESULTS="/root/parameter-golf/experiments/eval_time_mixing/results"
LOG="/root/parameter-golf/experiments/eval_time_mixing/logs/run_p0.log"

echo "=== P0 START: $(date) ===" >> "$LOG"

# List of experiments: "name:dir"
EXPS=(
    "fixed_7gram:exp0"
    "backoff_7:exp0"
    "backoff_7_ent:exp0"
    "backoff_9_ent_oadapt:exp0"
    "mix_linear:exp2"
    "mix_logistic:exp2"
    "mix_geometric:exp2"
    "order_2:exp1"
    "order_3:exp1"
    "order_5:exp1"
    "order_7:exp1"
    "order_9:exp1"
    "order_12:exp1"
    "order_15:exp1"
    "order_20:exp1"
)

for entry in "${EXPS[@]}"; do
    IFS=':' read -r name dir <<< "$entry"
    outdir="$RESULTS/$dir"
    mkdir -p "$outdir"

    # Skip if result already exists
    if [ -f "$outdir/$name.json" ]; then
        echo "SKIP $name (already done)" >> "$LOG"
        continue
    fi

    echo "$(date '+%H:%M:%S') START $name" >> "$LOG"
    python3 "$SCRIPT" --model "$MODEL" --exp "$name" --out "$outdir" >> "$LOG" 2>&1
    status=$?
    echo "$(date '+%H:%M:%S') DONE  $name (exit=$status)" >> "$LOG"
    echo "" >> "$LOG"
done

echo "=== P0 COMPLETE: $(date) ===" >> "$LOG"

# Summary
echo "" >> "$LOG"
echo "=== RESULTS ===" >> "$LOG"
for dir in exp0 exp1 exp2; do
    for f in "$RESULTS"/$dir/*.json; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .json)
        bpb=$(python3 -c "import json; d=json.load(open('$f')); print(f'{d[\"val_bpb\"]:.6f}')")
        echo "  [$dir] $name: $bpb" >> "$LOG"
    done
done
