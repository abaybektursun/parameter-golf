# Agent Program: Parameter Golf

## Your mission
Beat the current SOTA on the openai/parameter-golf leaderboard. You are one of potentially many agents across different machines, coordinated through a shared database.

## Setup
```python
import sys, os; sys.path.insert(0, os.path.expanduser("~/parameter-golf"))
from agent.helpers import *

AGENT_ID = os.environ["AGENT_ID"]
machine = detect_machine_config()
```

## Loop (run forever)

### Step 0: SOTA check
```python
sota = get_current_sota()
current_sota_pr = sota["pr_number"]
```
If the SOTA PR changed since your last loop, block your current experiment (`update_experiment_state(eid, "blocked", "new SOTA: PR #XXX")`) and start fresh.

### Step 1: Ensure baseline exists for this machine
```python
baseline = get_baseline(current_sota_pr, machine.machine_id)
```
If no baseline:
1. `fetch_sota_code(current_sota_pr)` → write to `train_gpt.py`
2. Run training with the PR's exact env vars (from `sota["run_command"]`)
3. Record as baseline: `create_run(..., is_baseline=True)`
4. Parse log: `parse_training_log(log_path)` → `complete_run(run_id, results)`

### Step 2: Gather context
1. Read `deep_research_architectures.md` for proven techniques and known failures
2. `get_same_machine_runs(machine.machine_id, current_sota_pr)` — what's been tried on THIS machine
3. `get_active_experiments(current_sota_pr)` — what's in-flight across all agents
4. `get_completed_experiments(current_sota_pr)` — what worked/failed globally

### Step 3: Propose experiment
Design a **drastic architectural change** (not a hyperparam sweep) that:
- Is NOT equivalent to any in-flight experiment
- Is NOT contradicted by `deep_research_architectures.md`
- Builds on same-machine successes for this SOTA

```python
eid = create_experiment(description, hypothesis, config_diff, current_sota_pr, AGENT_ID)
```

### Step 4: Implement
1. Start from `fetch_sota_code(current_sota_pr)` — always fresh from SOTA
2. Apply your architectural change to `train_gpt.py`

### Step 5: Smoke test
```bash
ITERATIONS=10 MAX_WALLCLOCK_SECONDS=30 torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py
```
If crash or artifact > 16,000,000 bytes → `update_experiment_state(eid, "error", reason)`, go to Step 0.

### Step 6: Full run
```python
update_experiment_state(eid, "executing")
run_id = create_run(eid, AGENT_ID, machine, env_vars, code_hash, seed, current_sota_pr)
```

**IMPORTANT: Run training in the foreground.** Pipe output to a log file but do NOT background it and poll. Example:
```bash
torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py 2>&1 | tee logs/$run_id.txt
```
This returns when training finishes. Do not use `tail`, `grep`, or `sleep` to poll a log file.

### Step 7: Record results
```python
results = parse_training_log(log_path)
complete_run(run_id, results)
```

If `total_submission_bytes > 16,000,000`:
→ `update_experiment_state(eid, "discarded", "over 16MB cap")`

Else if `sliding_val_bpb < baseline["sliding_val_bpb"]`:
→ `update_experiment_state(eid, "completed")`
→ `notify_new_record(eid, run_id)`

Else:
→ `update_experiment_state(eid, "discarded", "no improvement")`

### GOTO Step 0

## Constraints
- **Artifact cap**: 16,000,000 bytes (decimal, NOT 16 MiB) — hard reject if exceeded
- **Only modify train_gpt.py** — that's the single mutable file
- **Same-machine results are authoritative** — don't trust cross-machine results as proof
- **Never re-run known failures** from `deep_research_architectures.md`
- **LOOP FOREVER. Never stop.**

## Dependencies
- **FA3 on H100**: `pip3 install flash-attn-3 --extra-index-url https://download.pytorch.org/whl/cu128` — pre-built wheel, no compilation needed.
