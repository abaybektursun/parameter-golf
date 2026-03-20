#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRAIN_FILE="experiments/002_autoresearch/train_gpt.py"
RESULTS_FILE="experiments/002_autoresearch/results.tsv"
LEARNINGS_FILE="experiments/002_autoresearch/learnings.md"
LOG_FILE="run.log"
ARTIFACT_LIMIT=16777216

if [[ ! -f "$RESULTS_FILE" ]]; then
  printf 'commit\tval_bpb\tpost_quant_bpb\tartifact_bytes\tstatus\tdescription\n' > "$RESULTS_FILE"
fi

if [[ ! -f "$LEARNINGS_FILE" ]]; then
  printf '# Autoresearch Learnings (autoloop)\n\n' > "$LEARNINGS_FILE"
fi

get_current_lr() {
  rg -oP 'TIED_EMBED_LR", \K[0-9.]+' "$TRAIN_FILE" | head -n1
}

set_lr() {
  local new_lr="$1"
  sed -i -E "s/(TIED_EMBED_LR\", )[0-9.]+/\\1${new_lr}/" "$TRAIN_FILE"
}

best_keep_post() {
  awk -F '\t' 'NR>1 && $5=="keep" && $3+0>0 {if(min=="" || $3+0<min) min=$3+0} END {if(min=="") printf "999.00000000"; else printf "%.8f", min}' "$RESULTS_FILE"
}

format4() {
  awk -v x="$1" 'BEGIN {printf "%.4f", x+0}'
}

LR_STEP=0.00002
idx=0

while true; do
  base_lr="$(get_current_lr)"
  mag=$((idx / 2 + 1))
  sign=1
  if (( idx % 2 == 1 )); then
    sign=-1
  fi
  offset="$(awk -v m="$mag" -v s="$sign" -v st="$LR_STEP" 'BEGIN {printf "%.5f", m * s * st}')"
  idx=$((idx + 1))

  cand_lr="$(awk -v b="$base_lr" -v o="$offset" 'BEGIN {v=b+o; if (v < 0.001) v=0.001; if (v > 0.2) v=0.2; printf "%.5f", v}')"
  if [[ "$cand_lr" == "$base_lr" ]]; then
    continue
  fi
  if rg -q "\\b${cand_lr}\\b" "$RESULTS_FILE"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] skipping already-tested cand_lr=$cand_lr"
    continue
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] base_lr=$base_lr cand_lr=$cand_lr offset=$offset"

  set_lr "$cand_lr"
  if git diff --quiet -- "$TRAIN_FILE"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] no train file diff for cand_lr=$cand_lr, skipping"
    continue
  fi

  git add "$TRAIN_FILE"
  git commit -m "exp: set tied_embed_lr to $cand_lr with kv2 setup" -- "$TRAIN_FILE"
  exp_commit="$(git rev-parse --short HEAD)"

  NUM_GPUS="$(nvidia-smi -L | wc -l)"
  run_rc=0
  timeout 930 torchrun --standalone --nproc_per_node="$NUM_GPUS" experiments/002_autoresearch/train_gpt.py > "$LOG_FILE" 2>&1 || run_rc=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] run_exit_code=$run_rc commit=$exp_commit"

  pre_raw="$(rg -oP 'step:[0-9]+/20000 val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "$LOG_FILE" | tail -n1 || true)"
  post_raw="$(rg -oP 'final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "$LOG_FILE" | tail -n1 || true)"
  artifact_raw="$(rg -oP 'Total submission size int8\+zlib: \K[0-9]+' "$LOG_FILE" | tail -n1 || true)"

  status=""
  if [[ -z "$post_raw" || -z "$artifact_raw" || -z "$pre_raw" ]]; then
    pre_fmt="0.0000"
    post_fmt="0.0000"
    artifact_raw="0"
    status="crash"
  else
    pre_fmt="$(format4 "$pre_raw")"
    post_fmt="$(format4 "$post_raw")"
    best_post="$(best_keep_post)"
    better="$(awk -v p="$post_raw" -v b="$best_post" 'BEGIN {print (p+0 < b+0) ? 1 : 0}')"
    under_limit="$(awk -v a="$artifact_raw" -v l="$ARTIFACT_LIMIT" 'BEGIN {print (a+0 <= l+0) ? 1 : 0}')"
    if [[ "$better" == "1" && "$under_limit" == "1" ]]; then
      status="keep"
    else
      status="discard"
    fi
  fi

  description="autoloop tied_embed_lr=$cand_lr (base=$base_lr offset=$offset)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$exp_commit" "$pre_fmt" "$post_fmt" "$artifact_raw" "$status" "$description" >> "$RESULTS_FILE"

  echo "- $(date -u +%Y-%m-%dT%H:%M:%SZ) \\`$exp_commit\\`: tried \\`TIED_EMBED_LR=$cand_lr\\` from base \\`$base_lr\\` (offset \\`$offset\\`) -> pre/post \\`$pre_fmt\\`/\\`$post_fmt\\`, artifact \\`$artifact_raw\\`, status \\`$status\\`." >> "$LEARNINGS_FILE"

  if [[ "$status" == "keep" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] KEEP commit=$exp_commit post=$post_fmt bytes=$artifact_raw"
    git push origin HEAD
    aws s3 sync . s3://fuelos-autoresearch/latest/ --profile fuelos --exclude 'data/*' --exclude '.git/*' --exclude '__pycache__/*' --exclude '.venv/*'
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $status commit=$exp_commit post=$post_fmt bytes=$artifact_raw restoring base_lr=$base_lr"
    set_lr "$base_lr"
    git add "$TRAIN_FILE"
    git commit -m "revert: tied_embed_lr $cand_lr experiment (restore $base_lr)" -- "$TRAIN_FILE"
  fi

done
