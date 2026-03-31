"""
Smoke Test Round 2: Multi-agent coordination & edge cases.
Tests the coordination layer without requiring GPU training.

Usage: python agent/smoke_test_round2.py
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

# Ensure agent module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.helpers import (
    MachineConfig,
    _t_agents,
    _t_experiments,
    _t_runs,
    _t_sota,
    check_dead_agents,
    complete_run,
    create_experiment,
    create_run,
    fail_run,
    get_active_experiments,
    get_baseline,
    get_completed_experiments,
    get_current_sota,
    get_same_machine_runs,
    heartbeat,
    notify_new_record,
    parse_training_log,
    update_experiment_state,
)

PASS = 0
FAIL = 0
TEST_PREFIX = "smoke2_"

# Two simulated machines
MACHINE_A = MachineConfig(
    gpu_name="NVIDIA H100 80GB HBM3", gpu_count=8, gpu_memory_gb=80,
    cpu_model="AMD EPYC 9534", cpu_count=128, ram_gb=512,
    nvlink="NVSwitch", machine_ip="10.0.0.1", provider="vast.ai",
)
MACHINE_B = MachineConfig(
    gpu_name="NVIDIA H100 80GB HBM3", gpu_count=8, gpu_memory_gb=80,
    cpu_model="Intel Xeon w9-3595X", cpu_count=96, ram_gb=768,
    nvlink="NVSwitch", machine_ip="10.0.0.2", provider="aws",
)

AGENT_A = f"{TEST_PREFIX}agent_A"
AGENT_B = f"{TEST_PREFIX}agent_B"

# Track created resources for cleanup
created_experiments = []
created_runs = []
created_agents = []
created_sota_prs = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {detail}")


def cleanup():
    """Remove all test data from DynamoDB."""
    print("\n--- CLEANUP ---")
    for eid in created_experiments:
        _t_experiments.delete_item(Key={"experiment_id": eid})
    for rid in created_runs:
        _t_runs.delete_item(Key={"run_id": rid})
    for aid in created_agents:
        _t_agents.delete_item(Key={"agent_id": aid})
    for pr in created_sota_prs:
        _t_sota.delete_item(Key={"pr_number": pr})
    print(f"Cleaned up: {len(created_experiments)} experiments, {len(created_runs)} runs, "
          f"{len(created_agents)} agents, {len(created_sota_prs)} SOTA entries")


# ==========================================================================
# TEST 1: Live SOTA detection
# ==========================================================================
def test_sota_detection():
    print("\n=== TEST 1: Live SOTA detection ===")
    sota = get_current_sota()
    check("SOTA returns data", bool(sota), "empty result")
    if sota:
        check("SOTA has pr_number", "pr_number" in sota)
        check("SOTA has val_bpb", "val_bpb" in sota)
        check("SOTA val_bpb is reasonable", 1.0 < sota["val_bpb"] < 1.3,
              f"got {sota.get('val_bpb')}")
        check("SOTA has run_command", bool(sota.get("run_command")),
              "empty run command")
        print(f"  Current SOTA: PR#{sota['pr_number']} val_bpb={sota['val_bpb']}")


# ==========================================================================
# TEST 2: Two-agent conflict detection
# ==========================================================================
def test_conflict_detection():
    print("\n=== TEST 2: Two-agent conflict detection ===")
    SOTA_PR = 99901  # Fake SOTA PR for testing

    # Agent A creates an experiment
    eid_a = create_experiment(
        description=f"{TEST_PREFIX}XSA last 6 layers",
        hypothesis="More XSA layers improve BPB",
        config_diff="XSA_LAST_N=6",
        sota_pr=SOTA_PR, agent_id=AGENT_A,
    )
    created_experiments.append(eid_a)
    check("Agent A creates experiment", bool(eid_a))

    # Agent B queries active experiments (should see Agent A's)
    active = get_active_experiments(SOTA_PR)
    agent_a_exps = [e for e in active if e["agent_id"] == AGENT_A]
    check("Agent B sees Agent A's active experiment", len(agent_a_exps) >= 1,
          f"found {len(agent_a_exps)}")

    # Agent B checks description to avoid duplicating
    descriptions = [e["description"] for e in active]
    has_xsa = any("XSA" in d for d in descriptions)
    check("Agent B can detect XSA experiment in-flight", has_xsa,
          f"descriptions: {descriptions}")

    # Agent B creates a DIFFERENT experiment
    eid_b = create_experiment(
        description=f"{TEST_PREFIX}Partial RoPE 8 dims",
        hypothesis="Fewer RoPE dims improve generalization",
        config_diff="ROPE_DIMS=8",
        sota_pr=SOTA_PR, agent_id=AGENT_B,
    )
    created_experiments.append(eid_b)
    check("Agent B creates different experiment", bool(eid_b))

    # Both should now be active
    active = get_active_experiments(SOTA_PR)
    our_active = [e for e in active if e["experiment_id"] in (eid_a, eid_b)]
    check("Both experiments active simultaneously", len(our_active) == 2,
          f"found {len(our_active)}")

    # Clean up states
    update_experiment_state(eid_a, "discarded", "smoke test cleanup")
    update_experiment_state(eid_b, "discarded", "smoke test cleanup")


# ==========================================================================
# TEST 3: Full experiment lifecycle
# ==========================================================================
def test_full_lifecycle():
    print("\n=== TEST 3: Full experiment lifecycle ===")
    SOTA_PR = 99902

    # Create
    eid = create_experiment(
        description=f"{TEST_PREFIX}Stride-16 sliding eval",
        hypothesis="Finer stride improves eval accuracy",
        config_diff="EVAL_STRIDE=16",
        sota_pr=SOTA_PR, agent_id=AGENT_A,
    )
    created_experiments.append(eid)

    # Verify initial state
    exp = _t_experiments.get_item(Key={"experiment_id": eid})["Item"]
    check("Initial state is 'implementing'", exp["state"] == "implementing")

    # Transition: implementing → executing
    update_experiment_state(eid, "executing", "smoke test passed")
    exp = _t_experiments.get_item(Key={"experiment_id": eid})["Item"]
    check("Transition to 'executing'", exp["state"] == "executing")

    # Create a run
    rid = create_run(
        experiment_id=eid, agent_id=AGENT_A, machine=MACHINE_A,
        env_vars={"EVAL_STRIDE": "16"}, code_hash="abc123", seed=42,
        sota_pr=SOTA_PR,
    )
    created_runs.append(rid)
    check("Run created", bool(rid))

    # Verify run in DB
    run = _t_runs.get_item(Key={"run_id": rid})["Item"]
    check("Run links to experiment", run["experiment_id"] == eid)
    check("Run has machine_id", run["machine_id"] == MACHINE_A.machine_id)
    check("Run status is 'running'", run["status"] == "running")

    # Complete the run with results
    results = {
        "step_count": 9000,
        "step_avg_ms": 83.5,
        "train_time_ms": 580000,
        "val_bpb": 1.1260,
        "sliding_val_bpb": 1.1235,
        "sliding_stride": 16,
        "artifact_bytes": 15500000,
        "total_submission_bytes": 15700000,
        "model_params": 26829913,
        "peak_memory_mib": 42000,
    }
    complete_run(rid, results)
    run = _t_runs.get_item(Key={"run_id": rid})["Item"]
    check("Run status is 'completed'", run["status"] == "completed")
    check("sliding_val_bpb recorded", float(run["sliding_val_bpb"]) == 1.1235,
          f"got {run.get('sliding_val_bpb')}")
    check("artifact_bytes recorded", int(run["artifact_bytes"]) == 15500000)

    # Transition: executing → completed
    update_experiment_state(eid, "completed", "beat baseline")
    exp = _t_experiments.get_item(Key={"experiment_id": eid})["Item"]
    check("Final state is 'completed'", exp["state"] == "completed")

    # Query completed experiments
    completed = get_completed_experiments(SOTA_PR)
    our = [e for e in completed if e["experiment_id"] == eid]
    check("Appears in completed experiments", len(our) == 1)


# ==========================================================================
# TEST 4: Concurrent baselines on different machines
# ==========================================================================
def test_concurrent_baselines():
    print("\n=== TEST 4: Concurrent baselines (different machines) ===")
    SOTA_PR = 99903

    # Machine A baseline
    rid_a = create_run(
        experiment_id="baseline", agent_id=AGENT_A, machine=MACHINE_A,
        env_vars={}, code_hash="sota315", seed=42,
        sota_pr=SOTA_PR, is_baseline=True,
    )
    created_runs.append(rid_a)
    complete_run(rid_a, {"sliding_val_bpb": 1.1271, "step_count": 9000,
                         "total_submission_bytes": 15600000})

    # Machine B baseline
    rid_b = create_run(
        experiment_id="baseline", agent_id=AGENT_B, machine=MACHINE_B,
        env_vars={}, code_hash="sota315", seed=42,
        sota_pr=SOTA_PR, is_baseline=True,
    )
    created_runs.append(rid_b)
    complete_run(rid_b, {"sliding_val_bpb": 1.1265, "step_count": 9000,
                         "total_submission_bytes": 15600000})

    # Each machine queries its own baseline
    bl_a = get_baseline(SOTA_PR, MACHINE_A.machine_id)
    bl_b = get_baseline(SOTA_PR, MACHINE_B.machine_id)
    check("Machine A finds its baseline", bl_a is not None)
    check("Machine B finds its baseline", bl_b is not None)
    if bl_a and bl_b:
        check("Baselines are different runs", bl_a["run_id"] != bl_b["run_id"])
        check("Machine A baseline BPB", float(bl_a["sliding_val_bpb"]) == 1.1271)
        check("Machine B baseline BPB", float(bl_b["sliding_val_bpb"]) == 1.1265)

    # Same-machine runs query
    runs_a = get_same_machine_runs(MACHINE_A.machine_id, SOTA_PR)
    runs_b = get_same_machine_runs(MACHINE_B.machine_id, SOTA_PR)
    check("Machine A sees only its runs", all(r["machine_id"] == MACHINE_A.machine_id for r in runs_a))
    check("Machine B sees only its runs", all(r["machine_id"] == MACHINE_B.machine_id for r in runs_b))


# ==========================================================================
# TEST 5: Log parsing edge cases
# ==========================================================================
def test_log_parsing():
    print("\n=== TEST 5: Log parsing edge cases ===")

    # Case A: Normal complete log (real format: train lines have train_loss, val lines have val_loss)
    normal_log = """step:100/9000 train_loss:3.456 train_time:8500ms step_avg:85.2ms
step:100/9000 val_loss:2.100 val_bpb:1.3500 train_time:8500ms step_avg:85.2ms
step:1000/9000 train_loss:2.100 train_time:83500ms step_avg:83.5ms
step:1000/9000 val_loss:1.800 val_bpb:1.2100 train_time:83500ms step_avg:83.5ms
step:9000/9000 train_loss:1.500 train_time:580000ms step_avg:83.5ms
step:9000/9000 val_loss:1.200 val_bpb:1.1271 train_time:580000ms step_avg:83.5ms
Serialized model int6+zstd: 15500000 bytes
Total submission size int6+zstd: 15700000 bytes
model_params:26829913
peak memory allocated: 42000 MiB
final_int6_roundtrip_exact val_loss:1.190 val_bpb:1.1290
final_int6_sliding_window_exact val_loss:1.180 val_bpb:1.1248 stride:64
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(normal_log)
        normal_path = f.name

    r = parse_training_log(normal_path)
    check("Normal: step_count=9000", r["step_count"] == 9000)
    check("Normal: step_avg_ms=83.5", r["step_avg_ms"] == 83.5)
    check("Normal: val_bpb=1.1271 (last step, not sliding)", r["val_bpb"] == 1.1271)
    check("Normal: sliding_val_bpb=1.1248", r["sliding_val_bpb"] == 1.1248)
    check("Normal: artifact_bytes=15500000", r["artifact_bytes"] == 15500000)
    check("Normal: total_submission_bytes=15700000", r["total_submission_bytes"] == 15700000)
    check("Normal: model_params", r["model_params"] == 26829913)
    check("Normal: peak_memory_mib", r["peak_memory_mib"] == 42000)
    os.unlink(normal_path)

    # Case B: Truncated log (OOM crash mid-training)
    oom_log = """step:100/9000 val_loss:2.100 val_bpb:1.3500 train_time:8500ms step_avg:85.2ms
step:500/9000 val_loss:1.900 val_bpb:1.2500 train_time:42000ms step_avg:84.0ms
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(oom_log)
        oom_path = f.name

    r = parse_training_log(oom_path)
    check("OOM: step_count=500 (partial)", r["step_count"] == 500)
    check("OOM: val_bpb captured (last seen)", r["val_bpb"] == 1.2500)
    check("OOM: no sliding_val_bpb", r["sliding_val_bpb"] is None)
    check("OOM: no artifact_bytes", r["artifact_bytes"] is None)
    os.unlink(oom_path)

    # Case C: Empty log
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        empty_path = f.name

    r = parse_training_log(empty_path)
    check("Empty: step_count is None", r["step_count"] is None)
    check("Empty: val_bpb is None", r["val_bpb"] is None)
    check("Empty: doesn't crash", True)
    os.unlink(empty_path)

    # Case D: Log with sliding window lines that should NOT be matched as step val_bpb
    tricky_log = """step:9000/9000 val_loss:1.200 val_bpb:1.1271 train_time:580000ms step_avg:83.5ms
final_int6_sliding_window_exact val_loss:1.180 val_bpb:1.1248 stride:64
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(tricky_log)
        tricky_path = f.name

    r = parse_training_log(tricky_path)
    check("Tricky: val_bpb=1.1271 (step line, NOT sliding line)", r["val_bpb"] == 1.1271,
          f"got {r['val_bpb']}")
    check("Tricky: sliding_val_bpb=1.1248", r["sliding_val_bpb"] == 1.1248)
    os.unlink(tricky_path)


# ==========================================================================
# TEST 6: Dead agent cascade
# ==========================================================================
def test_dead_agent_cascade():
    print("\n=== TEST 6: Dead agent cascade ===")
    SOTA_PR = 99904

    # Agent A registers and creates experiment
    eid = create_experiment(
        description=f"{TEST_PREFIX}Dead agent test",
        hypothesis="Testing dead agent detection",
        config_diff="none",
        sota_pr=SOTA_PR, agent_id=AGENT_A,
    )
    created_experiments.append(eid)
    update_experiment_state(eid, "executing", "running full train")

    # Agent A heartbeats with a stale timestamp (15 min ago)
    stale_time = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    _t_agents.put_item(Item={
        "agent_id": AGENT_A,
        "machine_id": MACHINE_A.machine_id,
        "machine_ip": MACHINE_A.machine_ip,
        "provider": MACHINE_A.provider,
        "last_heartbeat": stale_time,
        "current_experiment_id": eid,
        "status": "active",
    })
    created_agents.append(AGENT_A)

    # Agent B is alive and healthy
    heartbeat(AGENT_B, MACHINE_B, current_experiment_id="")
    created_agents.append(AGENT_B)

    # Check for dead agents (10 min timeout)
    dead = check_dead_agents(timeout_minutes=10)
    check("Agent A detected as dead", AGENT_A in dead, f"dead list: {dead}")
    check("Agent B NOT detected as dead", AGENT_B not in dead)

    # Verify Agent A's experiment is blocked
    exp = _t_experiments.get_item(Key={"experiment_id": eid})["Item"]
    check("Dead agent's experiment blocked", exp["state"] == "blocked",
          f"state: {exp['state']}")
    check("Block reason mentions agent", AGENT_A in exp.get("state_reason", ""),
          f"reason: {exp.get('state_reason')}")

    # Verify agent status updated
    agent_a = _t_agents.get_item(Key={"agent_id": AGENT_A})["Item"]
    check("Agent A status is 'dead'", agent_a["status"] == "dead")

    agent_b = _t_agents.get_item(Key={"agent_id": AGENT_B})["Item"]
    check("Agent B status is 'active'", agent_b["status"] == "active")


# ==========================================================================
# TEST 7: Artifact cap enforcement (logic check)
# ==========================================================================
def test_artifact_cap():
    print("\n=== TEST 7: Artifact cap enforcement ===")
    SOTA_PR = 99905
    CAP = 16_000_000

    # Simulate experiment with over-cap artifact
    eid = create_experiment(
        description=f"{TEST_PREFIX}Over-cap experiment",
        hypothesis="Testing cap enforcement",
        config_diff="NUM_LAYERS=13",
        sota_pr=SOTA_PR, agent_id=AGENT_A,
    )
    created_experiments.append(eid)
    update_experiment_state(eid, "executing")

    rid = create_run(
        experiment_id=eid, agent_id=AGENT_A, machine=MACHINE_A,
        env_vars={}, code_hash="test", seed=42,
        sota_pr=SOTA_PR,
    )
    created_runs.append(rid)

    # Complete with over-cap artifact
    complete_run(rid, {
        "sliding_val_bpb": 1.1200,  # Great BPB but too large
        "total_submission_bytes": 17_200_000,
        "artifact_bytes": 16_800_000,
        "step_count": 9000,
    })

    # Agent decision: check cap
    run = _t_runs.get_item(Key={"run_id": rid})["Item"]
    total_bytes = int(run["total_submission_bytes"])
    over_cap = total_bytes > CAP

    check("Detects artifact over cap", over_cap, f"total_bytes={total_bytes}")

    if over_cap:
        update_experiment_state(eid, "discarded", f"over 16MB cap ({total_bytes} bytes)")

    exp = _t_experiments.get_item(Key={"experiment_id": eid})["Item"]
    check("Experiment discarded", exp["state"] == "discarded")
    check("Reason mentions cap", "16MB" in exp["state_reason"],
          f"reason: {exp['state_reason']}")

    # Under-cap experiment should pass
    eid2 = create_experiment(
        description=f"{TEST_PREFIX}Under-cap experiment",
        hypothesis="Testing cap enforcement (pass)",
        config_diff="NUM_LAYERS=11",
        sota_pr=SOTA_PR, agent_id=AGENT_A,
    )
    created_experiments.append(eid2)
    update_experiment_state(eid2, "executing")

    rid2 = create_run(
        experiment_id=eid2, agent_id=AGENT_A, machine=MACHINE_A,
        env_vars={}, code_hash="test2", seed=42,
        sota_pr=SOTA_PR,
    )
    created_runs.append(rid2)

    complete_run(rid2, {
        "sliding_val_bpb": 1.1240,
        "total_submission_bytes": 15_700_000,
        "artifact_bytes": 15_500_000,
        "step_count": 9000,
    })

    run2 = _t_runs.get_item(Key={"run_id": rid2})["Item"]
    under_cap = int(run2["total_submission_bytes"]) <= CAP
    check("Under-cap passes", under_cap)


# ==========================================================================
# TEST 8: SOTA transition mid-experiment
# ==========================================================================
def test_sota_transition():
    print("\n=== TEST 8: SOTA transition mid-experiment ===")

    OLD_SOTA_PR = 99906
    NEW_SOTA_PR = 99907

    # Insert old SOTA
    _t_sota.put_item(Item={
        "pr_number": OLD_SOTA_PR,
        "val_bpb": Decimal("1.1271"),
        "run_command": "torchrun train_gpt.py",
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "detected_by": AGENT_A,
    })
    created_sota_prs.append(OLD_SOTA_PR)

    # Agent A starts experiment on old SOTA
    eid = create_experiment(
        description=f"{TEST_PREFIX}SOTA transition test",
        hypothesis="Testing transition handling",
        config_diff="test",
        sota_pr=OLD_SOTA_PR, agent_id=AGENT_A,
    )
    created_experiments.append(eid)
    update_experiment_state(eid, "executing")

    # New SOTA appears
    _t_sota.put_item(Item={
        "pr_number": NEW_SOTA_PR,
        "val_bpb": Decimal("1.1200"),
        "run_command": "torchrun train_gpt.py --new",
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "detected_by": AGENT_B,
    })
    created_sota_prs.append(NEW_SOTA_PR)

    # Agent A's loop: detect SOTA changed
    # Simulate: agent knows current_sota_pr = OLD_SOTA_PR
    # It queries pg_sota for latest
    resp = _t_sota.scan()
    all_sota = resp.get("Items", [])
    # Filter to our test SOTAs
    test_sota = [s for s in all_sota if int(s["pr_number"]) in (OLD_SOTA_PR, NEW_SOTA_PR)]
    latest = min(test_sota, key=lambda s: float(s["val_bpb"]))
    new_pr = int(latest["pr_number"])

    check("Detects new SOTA PR", new_pr == NEW_SOTA_PR, f"got PR#{new_pr}")
    check("New SOTA is better", float(latest["val_bpb"]) < 1.1271)

    # Block current experiment
    if new_pr != OLD_SOTA_PR:
        update_experiment_state(eid, "blocked", f"SOTA changed to PR#{new_pr}")

    exp = _t_experiments.get_item(Key={"experiment_id": eid})["Item"]
    check("Old experiment blocked on SOTA change", exp["state"] == "blocked")
    check("Block reason mentions new PR", str(NEW_SOTA_PR) in exp["state_reason"])


# ==========================================================================
# TEST 9: fail_run
# ==========================================================================
def test_fail_run():
    print("\n=== TEST 9: Run failure handling ===")
    SOTA_PR = 99908

    rid = create_run(
        experiment_id="fail-test", agent_id=AGENT_A, machine=MACHINE_A,
        env_vars={}, code_hash="test", seed=42,
        sota_pr=SOTA_PR,
    )
    created_runs.append(rid)

    fail_run(rid, "torch.cuda.OutOfMemoryError: CUDA out of memory", status="oom")
    run = _t_runs.get_item(Key={"run_id": rid})["Item"]
    check("Failed run status is 'oom'", run["status"] == "oom")
    check("Error message captured", "OutOfMemoryError" in run["error_message"])
    check("completed_at set", "completed_at" in run)


# ==========================================================================
# TEST 10: Notification (live ntfy.sh)
# ==========================================================================
def test_notification():
    print("\n=== TEST 10: Notification delivery ===")
    SOTA_PR = 99909

    # Create baseline for notification comparison
    bl_rid = create_run(
        experiment_id="notif-baseline", agent_id=AGENT_A, machine=MACHINE_A,
        env_vars={}, code_hash="base", seed=42,
        sota_pr=SOTA_PR, is_baseline=True,
    )
    created_runs.append(bl_rid)
    complete_run(bl_rid, {"sliding_val_bpb": 1.1271, "total_submission_bytes": 15600000})

    # Create experiment + run that beats baseline
    eid = create_experiment(
        description=f"{TEST_PREFIX}Notification test experiment",
        hypothesis="Testing ntfy delivery",
        config_diff="test",
        sota_pr=SOTA_PR, agent_id=AGENT_A,
    )
    created_experiments.append(eid)

    rid = create_run(
        experiment_id=eid, agent_id=AGENT_A, machine=MACHINE_A,
        env_vars={}, code_hash="test", seed=42,
        sota_pr=SOTA_PR,
    )
    created_runs.append(rid)
    complete_run(rid, {"sliding_val_bpb": 1.1235, "total_submission_bytes": 15700000,
                       "gpu_name": "NVIDIA H100 80GB HBM3", "gpu_count": 8})

    # Send notification
    notify_new_record(eid, rid)
    print("  (Check ntfy.sh/parameter-golf-alerts for notification)")
    check("notify_new_record didn't crash", True)


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST ROUND 2: Multi-agent coordination & edge cases")
    print("=" * 60)

    tests = [
        test_sota_detection,
        test_conflict_detection,
        test_full_lifecycle,
        test_concurrent_baselines,
        test_log_parsing,
        test_dead_agent_cascade,
        test_artifact_cap,
        test_sota_transition,
        test_fail_run,
        test_notification,
    ]

    for test_fn in tests:
        test_fn()

    cleanup()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print(f"{'=' * 60}")

    sys.exit(1 if FAIL > 0 else 0)
