# Multi-Agent Experiment Coordination

## Overview
Multiple agents on different machines run experiments in parallel, coordinated through a shared database. Each agent optimizes against the current SOTA from the official repo.

## Infrastructure
- **Database**: AWS DynamoDB (profile: `fuelos`)
- **Knowledge base**: `deep_research_architectures.md` (in repo, synced via git)
- **Logs/artifacts**: S3 `s3://fuelos-autoresearch/`
- **SOTA source**: PRs on `openai/parameter-golf` (official repo)
- **Notifications**: ntfy.sh (push to phone via `curl ntfy.sh/<topic>`)

---

## Agent Runtime

Each agent is a **Claude Code** instance (`claude --dangerously-skip-permissions`) with:
- A `program.md` that defines its mission and loop (`agent/program.md`)
- A `watchdog.sh` cron job that restarts the agent if it dies
- Access to the helper functions library (Python module in the repo)
- The single mutable file: `train_gpt.py` (the only file the challenge allows changing)

Agents fetch the SOTA PR's `train_gpt.py` directly via `gh` API, modify it, and run it.

---

## SOTA Tracking

### How agents identify SOTA
1. List open + recently merged PRs on `openai/parameter-golf` via `gh`
2. Parse PR titles/bodies for `val_bpb` scores under challenge rules (≤16,000,000 bytes artifact, ≤10min 8xH100)
3. Lowest valid `val_bpb` = current SOTA
4. Extract `train_gpt.py` and run command from that PR

### Table: `sota`

| Field | Type | Key | Description |
|-------|------|-----|-------------|
| `pr_number` | Number | PK | PR number on openai/parameter-golf |
| `val_bpb` | Number | | Reported BPB |
| `artifact_bytes` | Number | | Reported artifact size |
| `run_command` | String | | Exact env vars + torchrun invocation |
| `code_hash` | String | | Hash of the PR's train_gpt.py |
| `detected_at` | String | | ISO timestamp |
| `detected_by` | String | | agent_id |

### SOTA transition
When an agent detects a new SOTA PR:
1. Write to `sota` table
2. Block current experiment: reason `"new SOTA: PR #XXX"`
3. Check `runs` for baseline with (sota_pr=XXX, machine_id=MINE, is_baseline=true)
   - Exists → use it, start experimenting
   - Missing → run the PR's exact code/config as baseline first

---

## Table: `experiments`

| Field | Type | Key | Description |
|-------|------|-----|-------------|
| `experiment_id` | String | PK | UUID |
| `agent_id` | String | | Owning agent |
| `sota_pr` | Number | | Which SOTA PR this builds on |
| `state` | String | GSI-PK | `proposed` / `implementing` / `executing` / `completed` / `discarded` / `blocked` / `error` |
| `state_reason` | String | | Why (for `blocked`, `error`, `discarded`) |
| `description` | String | | What the experiment is — must describe a drastic architectural change, not a hyperparam tweak |
| `hypothesis` | String | | Why it should work |
| `config_diff` | String | | What changed vs SOTA baseline |
| `parent_experiment_id` | String | | Builds on which prior experiment |
| `created_at` | String | | ISO timestamp |
| `updated_at` | String | | ISO timestamp |

**GSI**: `state-index` (PK: `state`, SK: `updated_at`)

### Experiment scope
Experiments must introduce **drastic architectural changes** — not hyperparameter sweeps. The description field captures what the change is in natural language.

---

## Table: `runs`

| Field | Type | Key | Description |
|-------|------|-----|-------------|
| `run_id` | String | PK | UUID (matches log file) |
| `experiment_id` | String | GSI-PK | FK to experiments |
| `agent_id` | String | | |
| `sota_pr` | Number | | Which SOTA PR this is based on |
| `is_baseline` | Boolean | | True if unmodified SOTA reproduction |
| **Machine** | | | |
| `machine_id` | String | GSI-PK | Hash of (gpu_name, gpu_count, cpu_model, ram_gb) |
| `gpu_name` | String | | e.g. "NVIDIA H100 80GB HBM3" |
| `gpu_count` | Number | | |
| `gpu_memory_gb` | Number | | Per-GPU |
| `cpu_model` | String | | |
| `cpu_count` | Number | | Logical cores |
| `ram_gb` | Number | | |
| `nvlink` | String | | Topology if available |
| `machine_ip` | String | | |
| `provider` | String | | vast.ai, runpod, aws |
| **Config** | | | |
| `env_vars` | Map | | All training env vars |
| `code_hash` | String | | Git commit or file hash |
| `seed` | Number | | |
| **Results — Timing** | | | |
| `step_count` | Number | | |
| `step_avg_ms` | Number | | |
| `train_time_ms` | Number | | |
| `eval_time_ms` | Number | | |
| **Results — Quality** | | | |
| `val_bpb` | Number | | Pre-quant |
| `roundtrip_val_bpb` | Number | | Post-quant |
| `sliding_val_bpb` | Number | | Submission score |
| `sliding_stride` | Number | | 32 or 64 |
| **Results — Artifact** | | | |
| `artifact_bytes` | Number | | |
| `total_submission_bytes` | Number | | Must be ≤ 16,000,000 |
| `model_params` | Number | | |
| `peak_memory_mib` | Number | | |
| **Meta** | | | |
| `log_s3_path` | String | | |
| `status` | String | | `running` / `completed` / `crashed` / `oom` |
| `error_message` | String | | |
| `created_at` | String | | |
| `completed_at` | String | | |

**GSIs**:
- `experiment-index` (PK: `experiment_id`, SK: `created_at`)
- `machine-index` (PK: `machine_id`, SK: `sliding_val_bpb`)

### `machine_id`
```python
machine_id = sha256(json.dumps({
    "gpu_name": ..., "gpu_count": ..., "cpu_model": ..., "ram_gb": ...
}, sort_keys=True).encode()).hexdigest()[:16]
```

---

## Table: `agents`

| Field | Type | Key | Description |
|-------|------|-----|-------------|
| `agent_id` | String | PK | UUID |
| `machine_id` | String | | |
| `machine_ip` | String | | |
| `provider` | String | | |
| `last_heartbeat` | String | | ISO timestamp, updated every loop iteration |
| `current_experiment_id` | String | | What it's working on |
| `status` | String | | `active` / `dead` |

### Heartbeat
- Agent writes `last_heartbeat` at the start of every loop iteration
- Other agents check: if `last_heartbeat` > 10 minutes old and status=`active`, mark agent as `dead` and its experiment as `blocked` (reason: `"agent dead"`)
- The dead agent's experiment becomes available for others to pick up

---

## Notifications: Beating Baseline

When an experiment's `sliding_val_bpb` beats the same-machine baseline for the current SOTA:

```bash
curl -d "NEW RECORD: {bpb} BPB (baseline: {baseline}, delta: {delta})
Experiment: {description}
Machine: {gpu_name} x{gpu_count}
Agent: {agent_id}
Artifact: {bytes} bytes" ntfy.sh/<topic>
```

Fires from `notify_new_record(experiment_id, run_id)` helper function.

---

## Agent Loop

```
LOOP:
  0. Heartbeat: update agents table with current timestamp
  1. Check SOTA: query openai/parameter-golf PRs
     IF new SOTA detected:
       → write to sota table
       → block current experiment (reason: "new SOTA: PR #XXX")
       → check runs for baseline (sota_pr, machine_id, is_baseline=true)
       → if missing: fetch_sota_code(), run as-is, record baseline
  2. Read deep_research_architectures.md
  3. Query runs WHERE machine_id = MINE AND sota_pr = CURRENT
  4. Query experiments WHERE state IN (implementing, executing, proposed) AND sota_pr = CURRENT
  5. Query experiments WHERE state IN (completed, discarded, error) AND sota_pr = CURRENT
  6. Propose experiment:
     - Not equivalent to anything in-flight
     - Not contradicted by deep_research_architectures.md
     - Builds on same-machine successes
     - Must be a drastic architectural change, not a hyperparam sweep
  7. Write experiment: state=implementing
  8. Fetch SOTA train_gpt.py, apply changes
  9. Smoke test: ITERATIONS=10 MAX_WALLCLOCK_SECONDS=30
     IF crash or artifact > 16,000,000 → state=error, GOTO 0
  10. Update experiment: state=executing, write run: status=running
  11. Train (full run)
  12. Parse log, update run with results
  13. IF total_submission_bytes > 16,000,000 → discard (reason: "over cap")
  14. IF sliding_val_bpb < baseline → notify_new_record(), update experiment: state=completed
      ELSE → update experiment: state=discarded
  15. If novel finding → update deep_research_architectures.md
  16. GOTO 0
```

## Decision Rules
- **SOTA check first**: always check for new SOTA before starting new work
- **Baseline required**: no experiments without a same-machine baseline for the current SOTA PR
- **Same-machine results are authoritative**: only trust runs from identical `machine_id`
- **Cross-machine results are informational**: worth trying, not proof
- **deep_research_architectures.md is prior knowledge**: don't re-run known failures
- **Old SOTA experiments are stale**: when SOTA changes, in-flight experiments for old SOTA get blocked
- **Artifact cap is hard**: total_submission_bytes > 16,000,000 = automatic discard

---

## Helper Functions

Python module agents import. All DB functions use boto3 with profile `fuelos`.

```
# --- Machine ---
detect_machine_config() → MachineConfig
  nvidia-smi, /proc/cpuinfo, free -g, nvidia-smi topo.
  Returns gpu_name, gpu_count, gpu_memory_gb, cpu_model, cpu_count, ram_gb, nvlink, machine_id.

# --- SOTA ---
get_current_sota() → SotaRecord
  gh pr list on openai/parameter-golf, parse for lowest valid val_bpb.

check_new_sota(current_pr: int) → SotaRecord | None
  Returns new SOTA if changed, writes to sota table.

fetch_sota_code(pr_number: int) → str
  Downloads train_gpt.py from PR via gh API.

# --- DB reads ---
get_baseline(sota_pr: int, machine_id: str) → RunRecord | None
get_same_machine_runs(machine_id: str, sota_pr: int) → list[RunRecord]
get_active_experiments(sota_pr: int) → list[ExperimentRecord]
get_completed_experiments(sota_pr: int) → list[ExperimentRecord]

# --- DB writes ---
create_experiment(description, hypothesis, config_diff, sota_pr, agent_id) → experiment_id
update_experiment_state(experiment_id, state, reason=None)
create_run(experiment_id, agent_id, machine_config, env_vars, code_hash, seed, sota_pr, is_baseline) → run_id
complete_run(run_id, results: dict)
fail_run(run_id, error_message, status="crashed")

# --- Heartbeat ---
heartbeat(agent_id)
check_dead_agents(timeout_minutes=10) → list[agent_id]

# --- Log parsing ---
parse_training_log(log_path: str) → dict
  Extracts step_count, step_avg_ms, val_bpb, sliding_val_bpb, artifact_bytes, etc.

# --- Notifications ---
notify_new_record(experiment_id, run_id)
  Sends push notification via ntfy.sh with experiment details and BPB delta.
```
