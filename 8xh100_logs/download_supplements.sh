#!/bin/bash
# Download train_gpt.py, README.md, submission.json for each existing PR directory
REPO="openai/parameter-golf"
BASE_DIR="/Users/abaybektursun/projects/parameter-golf/8xh100_logs"

download_supplements() {
  local pr_dir="$1"
  local pr_num=$(echo "$pr_dir" | sed 's/PR_0*\([0-9]*\)_.*/\1/')

  # Get PR metadata
  local pr_info=$(gh pr view "$pr_num" --repo "$REPO" --json headRefName,headRepositoryOwner,headRepository,files 2>/dev/null)
  if [ -z "$pr_info" ]; then
    echo "SKIP PR #$pr_num: could not fetch"
    return
  fi

  local owner=$(echo "$pr_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('headRepositoryOwner',{}).get('login',''))" 2>/dev/null)
  local branch=$(echo "$pr_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('headRefName',''))" 2>/dev/null)
  local repo_name=$(echo "$pr_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('headRepository',{}).get('name','parameter-golf'))" 2>/dev/null)

  # Find the target files from the PR's changed files
  local targets=$(echo "$pr_info" | python3 -c "
import json, sys
d = json.load(sys.stdin)
files = d.get('files', [])
for f in files:
    p = f['path']
    bn = p.rsplit('/', 1)[-1] if '/' in p else p
    if bn in ('train_gpt.py', 'README.md', 'submission.json'):
        print(p)
" 2>/dev/null)

  if [ -z "$targets" ]; then
    echo "SKIP PR #$pr_num: no supplement files in PR"
    return
  fi

  local count=0
  while IFS= read -r filepath; do
    [ -z "$filepath" ] && continue
    local filename=$(basename "$filepath")
    # Prefix with parent dir to avoid collisions
    local parent=$(basename "$(dirname "$filepath")")
    local dest_name="${filename}"
    if [ "$parent" != "." ] && [ "$parent" != "records" ]; then
      dest_name="${parent}__${filename}"
    fi

    # Skip if already exists
    if [ -f "${BASE_DIR}/${pr_dir}/${dest_name}" ]; then
      count=$((count + 1))
      continue
    fi

    local url="https://raw.githubusercontent.com/${owner}/${repo_name}/${branch}/${filepath}"
    if curl -sL -f -o "${BASE_DIR}/${pr_dir}/${dest_name}" "$url" 2>/dev/null; then
      # Verify it's not an error page
      if head -1 "${BASE_DIR}/${pr_dir}/${dest_name}" 2>/dev/null | grep -qi "<!DOCTYPE\|<html\|404"; then
        rm -f "${BASE_DIR}/${pr_dir}/${dest_name}"
      else
        count=$((count + 1))
      fi
    else
      rm -f "${BASE_DIR}/${pr_dir}/${dest_name}" 2>/dev/null
    fi
  done <<< "$targets"

  echo "OK   PR #$pr_num: $count supplement file(s)"
}

cd "$BASE_DIR"

# Process range from args, or all
if [ -n "$1" ] && [ -n "$2" ]; then
  for pr_dir in PR_*; do
    pr_num=$(echo "$pr_dir" | sed 's/PR_0*\([0-9]*\)_.*/\1/')
    if [ "$pr_num" -ge "$1" ] && [ "$pr_num" -le "$2" ]; then
      download_supplements "$pr_dir"
    fi
  done
else
  for pr_dir in PR_*; do
    download_supplements "$pr_dir"
  done
fi

echo "Done."
