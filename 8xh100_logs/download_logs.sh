#!/bin/bash
# Download 8xH100 training logs from parameter-golf PRs
# Usage: ./download_logs.sh <start_pr> <end_pr>
# Or: ./download_logs.sh (downloads all)

REPO="openai/parameter-golf"
BASE_DIR="/Users/abaybektursun/projects/parameter-golf/8xh100_logs"

# All confirmed 8xH100 PR numbers
ALL_PRS=(
  21 39 41 42 44 45 48 50 53 59 61 63 64 65 71 74 75 78 81 88 89 92 95 96
  102 107 114 116 122 123 136 137 139 141 145 148 155 156 160 161 162 164 169 170 172 173 176 179 181 184 186 191 192 194 196 197 198 200
  201 204 205 206 207 209 212 213 219 221 222 230 236 237 244 251 252 256 273 274 275 286 287 288 289 290 295 297
  302 304 305 307 309 310 315 317 331 332 333 338 348 349 351 352 354 355 359 360 362 368 369 370 375 379 385 389 394 397 398 399
  401 406 414 416 418 420 421 429 433 443 447 448 450 453 455 457 465 468 469 470 473 474 476 477 478 480 481 483 484 486 489 490 492 499 506 507 510 512 516 517 535 538 543 546 547 548 549 552
)

download_pr() {
  local pr_num=$1

  # Get PR metadata
  local pr_info=$(gh pr view "$pr_num" --repo "$REPO" --json title,headRefName,headRepository,headRepositoryOwner,files 2>/dev/null)
  if [ -z "$pr_info" ]; then
    echo "SKIP PR #$pr_num: could not fetch info"
    return
  fi

  local title=$(echo "$pr_info" | python3 -c "import json,sys,re; d=json.load(sys.stdin); t=d['title'][:80]; print(re.sub(r'[^a-zA-Z0-9._-]','_',t))" 2>/dev/null)
  local owner=$(echo "$pr_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('headRepositoryOwner',{}).get('login',''))" 2>/dev/null)
  local branch=$(echo "$pr_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('headRefName',''))" 2>/dev/null)
  local repo_name=$(echo "$pr_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('headRepository',{}).get('name','parameter-golf'))" 2>/dev/null)

  # Get log files from the PR
  local log_files=$(echo "$pr_info" | python3 -c "
import json, sys
d = json.load(sys.stdin)
files = d.get('files', [])
for f in files:
    p = f['path']
    if p.endswith('.log') or p.endswith('.txt'):
        # Filter to likely training logs
        low = p.lower()
        if 'train' in low or 'log' in low or 'seed' in low or 'ablation' in low or 'run' in low or 'verify' in low or 'baseline' in low or 'eval' in low:
            print(p)
" 2>/dev/null)

  if [ -z "$log_files" ]; then
    echo "SKIP PR #$pr_num: no log files found in file list"
    return
  fi

  # Create directory
  local pr_dir=$(printf "%s/PR_%04d_%s" "$BASE_DIR" "$pr_num" "$title")
  mkdir -p "$pr_dir"

  # Download each log file
  local count=0
  while IFS= read -r filepath; do
    [ -z "$filepath" ] && continue
    local filename=$(basename "$filepath")
    # If there are subdirectories in the path, flatten but keep context
    local subdir=$(dirname "$filepath" | sed 's|/|__|g')
    local dest_name="${filename}"

    # If multiple log files might have same name, prefix with parent dir
    if [ "$subdir" != "." ]; then
      local parent=$(basename "$(dirname "$filepath")")
      if [ "$parent" != "." ] && [ "$parent" != "records" ]; then
        dest_name="${parent}__${filename}"
      fi
    fi

    # Download via GitHub API raw content
    local url="https://raw.githubusercontent.com/${owner}/${repo_name}/${branch}/${filepath}"
    if curl -sL -f -o "$pr_dir/$dest_name" "$url" 2>/dev/null; then
      count=$((count + 1))
    else
      # Try alternate: use gh api
      gh api "repos/${owner}/${repo_name}/contents/${filepath}?ref=${branch}" --jq '.content' 2>/dev/null | base64 -d > "$pr_dir/$dest_name" 2>/dev/null
      if [ -s "$pr_dir/$dest_name" ]; then
        count=$((count + 1))
      else
        rm -f "$pr_dir/$dest_name"
      fi
    fi
  done <<< "$log_files"

  if [ $count -eq 0 ]; then
    rmdir "$pr_dir" 2>/dev/null
    echo "FAIL PR #$pr_num: downloaded 0 files"
  else
    echo "OK   PR #$pr_num: downloaded $count log file(s) -> $pr_dir"
  fi
}

# Determine which PRs to process
if [ -n "$1" ] && [ -n "$2" ]; then
  TARGET_PRS=()
  for pr in "${ALL_PRS[@]}"; do
    if [ "$pr" -ge "$1" ] && [ "$pr" -le "$2" ]; then
      TARGET_PRS+=("$pr")
    fi
  done
else
  TARGET_PRS=("${ALL_PRS[@]}")
fi

echo "Downloading logs for ${#TARGET_PRS[@]} PRs..."

for pr in "${TARGET_PRS[@]}"; do
  download_pr "$pr"
done

echo "Done."
