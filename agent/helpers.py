"""Helper functions for multi-agent experiment coordination."""

import hashlib
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import boto3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "parameter-golf-alerts")
AWS_PROFILE = os.environ.get("AWS_PROFILE", "fuelos")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
OFFICIAL_REPO = "openai/parameter-golf"

_session = boto3.Session(profile_name=AWS_PROFILE)
_ddb = _session.resource("dynamodb", region_name=AWS_REGION)

_t_sota = _ddb.Table("pg_sota")
_t_experiments = _ddb.Table("pg_experiments")
_t_runs = _ddb.Table("pg_runs")
_t_agents = _ddb.Table("pg_agents")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MachineConfig:
    gpu_name: str
    gpu_count: int
    gpu_memory_gb: int
    cpu_model: str
    cpu_count: int
    ram_gb: int
    nvlink: str
    machine_ip: str
    provider: str
    machine_id: str = ""

    def __post_init__(self):
        self.machine_id = hashlib.sha256(json.dumps({
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "cpu_model": self.cpu_model,
            "ram_gb": self.ram_gb,
        }, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Machine detection
# ---------------------------------------------------------------------------

def detect_machine_config() -> MachineConfig:
    gpu_name = subprocess.check_output(
        "nvidia-smi --query-gpu=name --format=csv,noheader | head -1",
        shell=True, text=True,
    ).strip()
    gpu_count = int(subprocess.check_output(
        "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l",
        shell=True, text=True,
    ).strip())
    gpu_mem = int(float(subprocess.check_output(
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1",
        shell=True, text=True,
    ).strip()) / 1024)
    cpu_model = subprocess.check_output(
        "grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2",
        shell=True, text=True,
    ).strip()
    cpu_count = int(subprocess.check_output("nproc", shell=True, text=True).strip())
    ram_gb = int(subprocess.check_output(
        "free -g | awk '/Mem:/{print $2}'", shell=True, text=True,
    ).strip())
    nvlink = subprocess.check_output(
        "nvidia-smi topo -m 2>/dev/null | head -3 || echo 'N/A'",
        shell=True, text=True,
    ).strip()
    machine_ip = subprocess.check_output(
        "hostname -I | awk '{print $1}'", shell=True, text=True,
    ).strip()
    provider = os.environ.get("PROVIDER", "unknown")

    return MachineConfig(
        gpu_name=gpu_name, gpu_count=gpu_count, gpu_memory_gb=gpu_mem,
        cpu_model=cpu_model, cpu_count=cpu_count, ram_gb=ram_gb,
        nvlink=nvlink, machine_ip=machine_ip, provider=provider,
    )


# ---------------------------------------------------------------------------
# SOTA
# ---------------------------------------------------------------------------

def get_current_sota() -> dict:
    """Query openai/parameter-golf PRs, find lowest valid val_bpb."""
    raw = subprocess.check_output(
        f"gh pr list --repo {OFFICIAL_REPO} --state all --limit 50 "
        f"--json number,title,body",
        shell=True, text=True,
    )
    prs = json.loads(raw)
    best_pr, best_bpb = None, 999.0
    for pr in prs:
        body = pr.get("body", "") or ""
        title = pr.get("title", "") or ""
        # Look for val_bpb in title or body
        for text in [title, body]:
            for m in re.finditer(r"val_bpb[:\s]*([0-9]+\.[0-9]+)", text):
                bpb = float(m.group(1))
                if bpb < best_bpb:
                    best_bpb = bpb
                    best_pr = pr["number"]
        # Also match patterns like "1.1271" in title with "Record" keyword
        if "record" in title.lower():
            for m in re.finditer(r"(\d\.\d{4})", title):
                bpb = float(m.group(1))
                if bpb < best_bpb:
                    best_bpb = bpb
                    best_pr = pr["number"]

    if best_pr is None:
        return {}

    # Get run command from PR body
    pr_data = json.loads(subprocess.check_output(
        f"gh pr view {best_pr} --repo {OFFICIAL_REPO} --json body,title",
        shell=True, text=True,
    ))
    run_cmd = ""
    body = pr_data.get("body", "") or ""
    cmd_match = re.search(r"```bash\n(.*?)```", body, re.DOTALL)
    if cmd_match:
        run_cmd = cmd_match.group(1).strip()

    # Extract artifact bytes from PR body
    artifact_bytes = None
    art_match = re.search(r"(?:artifact|submission)[:\s]*([0-9,]+)\s*bytes", body, re.IGNORECASE)
    if art_match:
        artifact_bytes = int(art_match.group(1).replace(",", ""))

    # Get code hash from PR head SHA
    head_sha = subprocess.check_output(
        f"gh api repos/{OFFICIAL_REPO}/pulls/{best_pr} --jq '.head.sha'",
        shell=True, text=True,
    ).strip()

    return {
        "pr_number": best_pr,
        "val_bpb": best_bpb,
        "run_command": run_cmd,
        "title": pr_data.get("title", ""),
        "artifact_bytes": artifact_bytes,
        "code_hash": head_sha,
    }


def check_new_sota(current_pr: int) -> Optional[dict]:
    """Returns new SOTA if changed, writes to DB. None otherwise."""
    sota = get_current_sota()
    if not sota or sota["pr_number"] == current_pr:
        return None
    now = datetime.now(timezone.utc).isoformat()
    item = {
        "pr_number": sota["pr_number"],
        "val_bpb": Decimal(str(sota["val_bpb"])),
        "run_command": sota["run_command"],
        "detected_at": now,
        "detected_by": os.environ.get("AGENT_ID", "unknown"),
    }
    if sota.get("artifact_bytes") is not None:
        item["artifact_bytes"] = sota["artifact_bytes"]
    if sota.get("code_hash"):
        item["code_hash"] = sota["code_hash"]
    _t_sota.put_item(Item=item)
    return sota


def fetch_sota_code(pr_number: int) -> str:
    """Download train_gpt.py from a PR's submission directory."""
    # Find the train_gpt.py under records/ (the submission file, not the base)
    files_raw = subprocess.check_output(
        f"gh api repos/{OFFICIAL_REPO}/pulls/{pr_number}/files --paginate "
        f"--jq '.[].filename'",
        shell=True, text=True,
    )
    target = None
    for fname in files_raw.strip().split("\n"):
        if fname.endswith("train_gpt.py") and "records/" in fname:
            target = fname
            break
    if not target:
        # Fallback: any train_gpt.py in the PR
        for fname in files_raw.strip().split("\n"):
            if fname.endswith("train_gpt.py"):
                target = fname
                break
    if not target:
        return ""

    # Get file content from the PR's head commit
    head_sha = subprocess.check_output(
        f"gh api repos/{OFFICIAL_REPO}/pulls/{pr_number} --jq '.head.sha'",
        shell=True, text=True,
    ).strip()
    content = subprocess.check_output(
        f"gh api 'repos/{OFFICIAL_REPO}/contents/{target}?ref={head_sha}' --jq '.content'",
        shell=True, text=True,
    ).strip()
    import base64
    return base64.b64decode(content).decode("utf-8")


# ---------------------------------------------------------------------------
# DB reads
# ---------------------------------------------------------------------------

def get_baseline(sota_pr: int, machine_id: str) -> Optional[dict]:
    resp = _t_runs.query(
        IndexName="machine-index",
        KeyConditionExpression="machine_id = :mid",
        FilterExpression="sota_pr = :pr AND is_baseline = :t",
        ExpressionAttributeValues={
            ":pr": sota_pr, ":mid": machine_id, ":t": True,
        },
    )
    items = resp.get("Items", [])
    return items[0] if items else None


def get_same_machine_runs(machine_id: str, sota_pr: int) -> list[dict]:
    resp = _t_runs.query(
        IndexName="machine-index",
        KeyConditionExpression="machine_id = :mid",
        FilterExpression="sota_pr = :pr",
        ExpressionAttributeValues={":mid": machine_id, ":pr": sota_pr},
    )
    return resp.get("Items", [])


def get_active_experiments(sota_pr: int) -> list[dict]:
    results = []
    for state in ("proposed", "implementing", "executing"):
        resp = _t_experiments.query(
            IndexName="state-index",
            KeyConditionExpression="#s = :s",
            FilterExpression="sota_pr = :pr",
            ExpressionAttributeNames={"#s": "state"},
            ExpressionAttributeValues={":s": state, ":pr": sota_pr},
        )
        results.extend(resp.get("Items", []))
    return results


def get_completed_experiments(sota_pr: int) -> list[dict]:
    results = []
    for state in ("completed", "discarded", "error"):
        resp = _t_experiments.query(
            IndexName="state-index",
            KeyConditionExpression="#s = :s",
            FilterExpression="sota_pr = :pr",
            ExpressionAttributeNames={"#s": "state"},
            ExpressionAttributeValues={":s": state, ":pr": sota_pr},
        )
        results.extend(resp.get("Items", []))
    return results


# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------

def create_experiment(description: str, hypothesis: str, config_diff: str,
                      sota_pr: int, agent_id: str,
                      parent_experiment_id: str = "") -> str:
    eid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    _t_experiments.put_item(Item={
        "experiment_id": eid,
        "agent_id": agent_id,
        "sota_pr": sota_pr,
        "state": "implementing",
        "state_reason": "",
        "description": description,
        "hypothesis": hypothesis,
        "config_diff": config_diff,
        "parent_experiment_id": parent_experiment_id,
        "created_at": now,
        "updated_at": now,
    })
    return eid


def update_experiment_state(experiment_id: str, state: str, reason: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    _t_experiments.update_item(
        Key={"experiment_id": experiment_id},
        UpdateExpression="SET #s = :s, state_reason = :r, updated_at = :t",
        ExpressionAttributeNames={"#s": "state"},
        ExpressionAttributeValues={":s": state, ":r": reason, ":t": now},
    )


def create_run(experiment_id: str, agent_id: str, machine: MachineConfig,
               env_vars: dict, code_hash: str, seed: int,
               sota_pr: int, is_baseline: bool = False) -> str:
    rid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    _t_runs.put_item(Item={
        "run_id": rid,
        "experiment_id": experiment_id,
        "agent_id": agent_id,
        "sota_pr": sota_pr,
        "is_baseline": is_baseline,
        "machine_id": machine.machine_id,
        "gpu_name": machine.gpu_name,
        "gpu_count": machine.gpu_count,
        "gpu_memory_gb": machine.gpu_memory_gb,
        "cpu_model": machine.cpu_model,
        "cpu_count": machine.cpu_count,
        "ram_gb": machine.ram_gb,
        "nvlink": machine.nvlink,
        "machine_ip": machine.machine_ip,
        "provider": machine.provider,
        "env_vars": env_vars,
        "code_hash": code_hash,
        "seed": seed,
        "status": "running",
        "created_at": now,
    })
    return rid


def complete_run(run_id: str, results: dict):
    now = datetime.now(timezone.utc).isoformat()
    update_expr = "SET #st = :st, completed_at = :t"
    attr_names = {"#st": "status"}
    attr_vals = {":st": "completed", ":t": now}
    for k, v in results.items():
        if v is None:
            continue
        safe_key = k.replace("-", "_")
        placeholder = f"#k_{safe_key}"
        attr_names[placeholder] = safe_key
        update_expr += f", {placeholder} = :{safe_key}"
        attr_vals[f":{safe_key}"] = Decimal(str(v)) if isinstance(v, float) else v
    _t_runs.update_item(
        Key={"run_id": run_id},
        UpdateExpression=update_expr,
        ExpressionAttributeNames=attr_names,
        ExpressionAttributeValues=attr_vals,
    )


def fail_run(run_id: str, error_message: str, status: str = "crashed"):
    now = datetime.now(timezone.utc).isoformat()
    _t_runs.update_item(
        Key={"run_id": run_id},
        UpdateExpression="SET #st = :st, error_message = :e, completed_at = :t",
        ExpressionAttributeNames={"#st": "status"},
        ExpressionAttributeValues={":st": status, ":e": error_message, ":t": now},
    )


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def heartbeat(agent_id: str, machine: MachineConfig, current_experiment_id: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    _t_agents.put_item(Item={
        "agent_id": agent_id,
        "machine_id": machine.machine_id,
        "machine_ip": machine.machine_ip,
        "provider": machine.provider,
        "last_heartbeat": now,
        "current_experiment_id": current_experiment_id,
        "status": "active",
    })


def check_dead_agents(timeout_minutes: int = 10) -> list[str]:
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)).isoformat()
    resp = _t_agents.scan(
        FilterExpression="#st = :active AND last_heartbeat < :cutoff",
        ExpressionAttributeNames={"#st": "status"},
        ExpressionAttributeValues={":active": "active", ":cutoff": cutoff},
    )
    dead = []
    for agent in resp.get("Items", []):
        aid = agent["agent_id"]
        _t_agents.update_item(
            Key={"agent_id": aid},
            UpdateExpression="SET #st = :dead",
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={":dead": "dead"},
        )
        eid = agent.get("current_experiment_id", "")
        if eid:
            update_experiment_state(eid, "blocked", f"agent {aid} dead")
        dead.append(aid)
    return dead


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_training_log(log_path: str) -> dict:
    with open(log_path) as f:
        text = f.read()

    def find(pattern, typ=float, last=False, pattern_flags=0):
        if last:
            matches = re.findall(pattern, text, pattern_flags)
            return typ(matches[-1]) if matches else None
        m = re.search(pattern, text, pattern_flags)
        return typ(m.group(1)) if m else None

    # Find last step line for step_count and step_avg
    step_lines = re.findall(r"step:(\d+)/\d+.*step_avg:([0-9.]+)ms", text)
    step_count = int(step_lines[-1][0]) if step_lines else None
    step_avg_ms = float(step_lines[-1][1]) if step_lines else None

    # Train time
    train_time_match = re.findall(r"train_time:([0-9.]+)ms", text)
    train_time_ms = float(train_time_match[-1]) if train_time_match else None

    return {
        "step_count": step_count,
        "step_avg_ms": step_avg_ms,
        "train_time_ms": train_time_ms,
        "val_bpb": find(r"^step:\d+/\d+ val_loss:[0-9.]+ val_bpb:([0-9.]+)", last=True, pattern_flags=re.MULTILINE),
        "roundtrip_val_bpb": find(r"final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)"),
        "sliding_val_bpb": find(r"final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)"),
        "sliding_stride": find(r"stride:(\d+)", int),
        "artifact_bytes": find(r"Serialized model int6\+\w+: (\d+) bytes", int),
        "total_submission_bytes": find(r"Total submission size int6\+\w+: (\d+) bytes", int),
        "model_params": find(r"model_params:(\d+)", int),
        "peak_memory_mib": find(r"peak memory allocated: (\d+) MiB", int),
        "eval_time_ms": find(r"eval_time:(\d+)ms"),
    }


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def notify_new_record(experiment_id: str, run_id: str):
    exp = _t_experiments.get_item(Key={"experiment_id": experiment_id}).get("Item", {})
    run = _t_runs.get_item(Key={"run_id": run_id}).get("Item", {})
    baseline = get_baseline(run.get("sota_pr", 0), run.get("machine_id", ""))
    baseline_bpb = float(baseline["sliding_val_bpb"]) if baseline else 0.0
    run_bpb = float(run.get("sliding_val_bpb", 0))
    delta = run_bpb - baseline_bpb

    msg = (
        f"NEW RECORD: {run_bpb:.4f} BPB "
        f"(baseline: {baseline_bpb:.4f}, delta: {delta:+.4f})\n"
        f"Experiment: {exp.get('description', '?')}\n"
        f"Machine: {run.get('gpu_name', '?')} x{run.get('gpu_count', '?')}\n"
        f"Artifact: {run.get('total_submission_bytes', '?')} bytes"
    )
    subprocess.run(
        ["curl", "-s", "-d", msg, f"https://ntfy.sh/{NTFY_TOPIC}"],
        capture_output=True,
    )
