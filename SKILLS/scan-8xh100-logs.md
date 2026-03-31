---
name: scan-8xh100-logs
description: Incrementally scan openai/parameter-golf PRs for 8xH100 training logs, download valid ones with supplements.
user_invocable: true
---

# Scan & Download 8xH100 Logs

## Sequence

### 1. Find where we left off

```bash
BASE_DIR="/Users/abaybektursun/projects/parameter-golf/8xh100_logs"
mkdir -p "$BASE_DIR"
LAST_PR=$(ls -d "$BASE_DIR"/PR_* 2>/dev/null | sed 's/.*PR_0*\([0-9]*\)_.*/\1/' | sort -n | tail -1)
LAST_PR=${LAST_PR:-0}
```

### 2. Bulk-fetch new PR numbers (single API call)

```bash
gh pr list --repo openai/parameter-golf --state all --limit 2000 \
  --json number -q "[.[] | select(.number > $LAST_PR)] | .[].number" \
  | sort -n > /tmp/new_pr_numbers.txt
```

### 3. Scan new PRs and identify 8xH100 submissions

Use this Python scanner. It handles special characters in PR bodies that break shell-based parsing.

Run it in **parallel batches** (split the PR list into 5 chunks, run concurrently) for speed.

```python
import subprocess, json

REPO = "openai/parameter-golf"

POSITIVE = ["8xh100", "8×h100", "8x h100", "world_size:8", "nproc_per_node=8"]
NEGATIVE = ["1xh100", "1×h100", "4xh100", "h200", "a100", "a800", "rtx",
            "apple", "mlx", "kaggle", "m4 air", "pending compute", "awaiting h100"]

def scan_pr(num):
    result = subprocess.run(
        ["gh", "pr", "view", str(num), "--repo", REPO,
         "--json", "title,body,state,files,headRefName,headRepositoryOwner,headRepository"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)

def is_8xh100(d):
    """Determine if PR is a real 8xH100 submission with downloadable logs."""
    title = d.get("title", "")
    body = d.get("body", "") or ""
    files = d.get("files", [])
    combined = (body + " " + title).lower()

    # Must have log files
    log_files = [f["path"] for f in files
                 if f["path"].endswith((".log", ".txt"))
                 and any(k in f["path"].lower() for k in ("train","log","seed","run","eval"))]
    if not log_files:
        return False, "no log files"

    # Check for 8xH100 evidence
    has_positive = any(s in combined for s in POSITIVE)
    # track_10min_16mb in file paths is strong signal
    if not has_positive:
        has_positive = any("track_10min_16mb" in f["path"] for f in files)
    if not has_positive:
        return False, "no 8xH100 evidence"

    # Check for disqualifying negatives
    # A negative only disqualifies if the PR is ABOUT that hardware, not just mentioning it
    found_neg = [n for n in NEGATIVE if n in combined]
    for neg in found_neg:
        # If the title explicitly says this is a different GPU, skip
        if neg in title.lower() and not any(p in title.lower() for p in POSITIVE):
            return False, f"title says {neg}"
        # A800/A100 as primary hardware (not just a comparison)
        if neg in ("a800", "a100") and neg in title.lower():
            return False, f"primary hardware is {neg}"

    return True, log_files

# Main
pr_numbers = [int(x.strip()) for x in open("/tmp/new_pr_numbers.txt") if x.strip()]
confirmed = []

for num in pr_numbers:
    d = scan_pr(num)
    if d is None:
        continue
    ok, info = is_8xh100(d)
    if ok:
        confirmed.append(num)
        print(f">>> PR #{num}: {d['title']}")
        print(f"    logs: {info}")
    # Optionally print skips for review:
    # else:
    #     print(f"    PR #{num}: SKIP ({info})")

print(f"\nConfirmed: {confirmed}")
# Write for step 4
with open("/tmp/confirmed_prs.txt", "w") as f:
    for n in confirmed:
        f.write(f"{n}\n")
```

### 4. Download — use this function

Run in **parallel batches** (5 concurrent shells).

```bash
REPO="openai/parameter-golf"
BASE_DIR="/Users/abaybektursun/projects/parameter-golf/8xh100_logs"

download_pr_full() {
  local pr_num=$1
  local pr_info=$(gh pr view "$pr_num" --repo "$REPO" \
    --json title,headRefName,headRepositoryOwner,headRepository,files 2>/dev/null)
  [ -z "$pr_info" ] && { echo "SKIP PR #$pr_num: could not fetch"; return; }

  local title=$(echo "$pr_info" | python3 -c "
import json,sys,re; d=json.load(sys.stdin)
print(re.sub(r'[^a-zA-Z0-9._-]','_',d['title'][:80]))" 2>/dev/null)
  local owner=$(echo "$pr_info" | python3 -c "
import json,sys; print(json.load(sys.stdin).get('headRepositoryOwner',{}).get('login',''))" 2>/dev/null)
  local branch=$(echo "$pr_info" | python3 -c "
import json,sys; print(json.load(sys.stdin).get('headRefName',''))" 2>/dev/null)
  local repo_name=$(echo "$pr_info" | python3 -c "
import json,sys; print(json.load(sys.stdin).get('headRepository',{}).get('name','parameter-golf'))" 2>/dev/null)

  local pr_dir=$(printf "%s/PR_%04d_%s" "$BASE_DIR" "$pr_num" "$title")
  mkdir -p "$pr_dir"

  local targets=$(echo "$pr_info" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for f in d.get('files', []):
    p = f['path']
    bn = p.rsplit('/', 1)[-1] if '/' in p else p
    if bn.endswith('.log') or bn.endswith('.txt'):
        low = p.lower()
        if 'train' in low or 'log' in low or 'seed' in low or 'run' in low or 'eval' in low:
            print(p)
    elif bn in ('train_gpt.py', 'README.md', 'submission.json'):
        print(p)
" 2>/dev/null)

  [ -z "$targets" ] && { rmdir "$pr_dir" 2>/dev/null; echo "SKIP PR #$pr_num: no files"; return; }

  local count=0
  while IFS= read -r filepath; do
    [ -z "$filepath" ] && continue
    local filename=$(basename "$filepath")
    local parent=$(basename "$(dirname "$filepath")")
    local dest_name="$filename"
    [ "$parent" != "." ] && [ "$parent" != "records" ] && dest_name="${parent}__${filename}"
    [ -f "$pr_dir/$dest_name" ] && { count=$((count + 1)); continue; }

    local url="https://raw.githubusercontent.com/${owner}/${repo_name}/${branch}/${filepath}"
    if curl -sL -f -o "$pr_dir/$dest_name" "$url" 2>/dev/null; then
      if head -1 "$pr_dir/$dest_name" 2>/dev/null | grep -qi '<!DOCTYPE\|<html'; then
        rm -f "$pr_dir/$dest_name"
      else
        count=$((count + 1))
      fi
    else
      rm -f "$pr_dir/$dest_name" 2>/dev/null
    fi
  done <<< "$targets"

  [ $count -eq 0 ] && { rmdir "$pr_dir" 2>/dev/null; echo "FAIL PR #$pr_num: 0 files"; return; }
  echo "OK   PR #$pr_num: $count files"
}

# Download all confirmed PRs
while read pr_num; do
  download_pr_full "$pr_num"
done < /tmp/confirmed_prs.txt
```

### 5. Validate downloaded files

For each new PR directory, check every log file:
- **DELETE if:** 0 bytes, HTML error page, or crashed run with zero training data
- **KEEP if:** contains training metrics — `step`, `loss`, `val_bpb`, `val_loss`, `[RESULT]`, `score=`
- **Don't be fooled by:** "404" or "rate limit" appearing inside otherwise valid logs — these are often process IDs, port numbers, or benign warnings. Read the actual content before deleting.
- Orchestration logs using `[STAGE]/[RESULT]` format are valid wrapper output

### 6. Print stats

```bash
cd "$BASE_DIR"
echo "PR directories: $(ls -d PR_* | wc -l)"
echo "Log files: $(find PR_* -type f \( -name '*.log' -o -name '*.txt' \) | wc -l)"
echo "train_gpt.py: $(find PR_* -type f -name '*train_gpt*' | wc -l)"
echo "submission.json: $(find PR_* -type f -name '*submission*' | wc -l)"
echo "Total size: $(du -sh . | cut -f1)"
```

## Notes

- **Parallelism:** Split PR lists into batches and run 5 concurrent shells. This is 5x faster and stays within GitHub rate limits.
- **Rate limits:** GitHub allows 5000 authenticated requests/hour. Each PR costs ~2 calls (view + curl). Monitor with `gh api rate_limit`.
- **Deleted branches:** Some closed PRs have deleted branches. The curl will fail silently — that's fine, the function handles it.
- **Duplicates:** Some authors resubmit the same run as a new PR (e.g., fixing formatting). The folder naming includes PR number so duplicates coexist harmlessly, but note them in the report if spotted.
- **Output structure:** Each folder is `PR_{NNNN}_{title}/` with files prefixed by their submission subdirectory name for disambiguation.
