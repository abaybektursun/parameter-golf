#!/bin/bash
# Download logs + supplements for new 8xH100 PRs
REPO="openai/parameter-golf"
BASE_DIR="/Users/abaybektursun/projects/parameter-golf/8xh100_logs"

# New confirmed 8xH100 PRs with log files
NEW_PRS=(555 562 563 564 568 569 570 577 578 579 583 585 586 589)

download_pr_full() {
  local pr_num=$1
  local pr_info=$(gh pr view "$pr_num" --repo "$REPO" --json title,headRefName,headRepositoryOwner,headRepository,files 2>/dev/null)
  if [ -z "$pr_info" ]; then
    echo "SKIP PR #$pr_num: could not fetch"
    return
  fi

  local title=$(echo "$pr_info" | python3 -c "import json,sys,re; d=json.load(sys.stdin); t=d['title'][:80]; print(re.sub(r'[^a-zA-Z0-9._-]','_',t))" 2>/dev/null)
  local owner=$(echo "$pr_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('headRepositoryOwner',{}).get('login',''))" 2>/dev/null)
  local branch=$(echo "$pr_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('headRefName',''))" 2>/dev/null)
  local repo_name=$(echo "$pr_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('headRepository',{}).get('name','parameter-golf'))" 2>/dev/null)

  local pr_dir=$(printf "%s/PR_%04d_%s" "$BASE_DIR" "$pr_num" "$title")
  mkdir -p "$pr_dir"

  # Get all target files (logs + train_gpt.py + README.md + submission.json)
  local targets=$(echo "$pr_info" | python3 -c "
import json, sys
d = json.load(sys.stdin)
files = d.get('files', [])
for f in files:
    p = f['path']
    bn = p.rsplit('/', 1)[-1] if '/' in p else p
    if bn.endswith('.log') or bn.endswith('.txt'):
        low = p.lower()
        if 'train' in low or 'log' in low or 'seed' in low or 'run' in low or 'eval' in low:
            print(p)
    elif bn in ('train_gpt.py', 'README.md', 'submission.json'):
        print(p)
" 2>/dev/null)

  if [ -z "$targets" ]; then
    echo "SKIP PR #$pr_num: no target files"
    rmdir "$pr_dir" 2>/dev/null
    return
  fi

  local count=0
  while IFS= read -r filepath; do
    [ -z "$filepath" ] && continue
    local filename=$(basename "$filepath")
    local parent=$(basename "$(dirname "$filepath")")
    local dest_name="${filename}"
    if [ "$parent" != "." ] && [ "$parent" != "records" ]; then
      dest_name="${parent}__${filename}"
    fi

    [ -f "$pr_dir/$dest_name" ] && { count=$((count + 1)); continue; }

    local url="https://raw.githubusercontent.com/${owner}/${repo_name}/${branch}/${filepath}"
    if curl -sL -f -o "$pr_dir/$dest_name" "$url" 2>/dev/null; then
      if head -1 "$pr_dir/$dest_name" 2>/dev/null | grep -qi "<!DOCTYPE\|<html"; then
        rm -f "$pr_dir/$dest_name"
      else
        count=$((count + 1))
      fi
    else
      rm -f "$pr_dir/$dest_name" 2>/dev/null
    fi
  done <<< "$targets"

  if [ $count -eq 0 ]; then
    rmdir "$pr_dir" 2>/dev/null
    echo "FAIL PR #$pr_num: 0 files"
  else
    echo "OK   PR #$pr_num: $count file(s) -> $(basename "$pr_dir")"
  fi
}

echo "Downloading ${#NEW_PRS[@]} new PRs..."
for pr in "${NEW_PRS[@]}"; do
  download_pr_full "$pr"
done
echo "Done."
