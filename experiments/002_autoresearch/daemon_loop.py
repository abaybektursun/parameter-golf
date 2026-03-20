#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = ROOT / "experiments/002_autoresearch/train_gpt.py"
RESULTS_FILE = ROOT / "experiments/002_autoresearch/results.tsv"
LEARNINGS_FILE = ROOT / "experiments/002_autoresearch/learnings.md"
RUN_LOG = ROOT / "run.log"
ARTIFACT_LIMIT = 16_777_216


@dataclass(frozen=True)
class ParamSpec:
    key: str
    kind: str
    step: float
    min_value: float
    max_value: float
    precision: int = 5


PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("TIED_EMBED_LR", "float", 0.00002, 0.001, 0.2, 5),
    ParamSpec("MATRIX_LR", "float", 0.001, 0.005, 0.2, 5),
    ParamSpec("SCALAR_LR", "float", 0.001, 0.005, 0.2, 5),
    ParamSpec("WARMDOWN_ITERS", "int", 20.0, 50.0, 2000.0, 0),
    ParamSpec("QK_GAIN_INIT", "float", 0.05, 0.5, 3.0, 5),
)


def ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def run(cmd: list[str], check: bool = True, capture: bool = False, cwd: Path | None = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture,
    )


def ensure_files() -> None:
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text("commit\tval_bpb\tpost_quant_bpb\tartifact_bytes\tstatus\tdescription\n", encoding="utf-8")
    if not LEARNINGS_FILE.exists():
        LEARNINGS_FILE.write_text("# Autoresearch Learnings\n\n", encoding="utf-8")


def read_train_file() -> str:
    return TRAIN_FILE.read_text(encoding="utf-8")


def format_value(spec: ParamSpec, value: float) -> str:
    if spec.kind == "int":
        return str(int(round(value)))
    return f"{value:.{spec.precision}f}"


def get_current_value(spec: ParamSpec) -> float:
    text = TRAIN_FILE.read_text(encoding="utf-8")
    m = re.search(rf'{spec.key}",\s*([0-9.]+)', text)
    if not m:
        raise RuntimeError(f"could not parse {spec.key}")
    return float(m.group(1))


def set_value(spec: ParamSpec, value: float) -> None:
    rendered = format_value(spec, value)
    text = TRAIN_FILE.read_text(encoding="utf-8")
    new_text, n = re.subn(rf'({spec.key}",\s*)[0-9.]+', rf"\g<1>{rendered}", text, count=1)
    if n != 1:
        raise RuntimeError(f"failed to update {spec.key}")
    TRAIN_FILE.write_text(new_text, encoding="utf-8")


def already_tested(spec: ParamSpec, value: float) -> bool:
    rendered = format_value(spec, value)
    key_lc = spec.key.lower()
    for i, line in enumerate(RESULTS_FILE.read_text(encoding="utf-8").splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        status = parts[4].strip()
        # Allow retrying infrastructure crash entries; skip only successful or measured runs.
        if status == "crash":
            continue

        line_lc = line.lower()
        desc_lc = parts[5].lower()
        if f"{key_lc}={rendered.lower()}" in desc_lc:
            return True
        # Backward compatibility with legacy free-form descriptions.
        if key_lc in line_lc and rendered in line:
            return True
    return False


def best_keep_post() -> float:
    best = None
    for i, line in enumerate(RESULTS_FILE.read_text(encoding="utf-8").splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        status = parts[4]
        try:
            post = float(parts[2])
        except ValueError:
            continue
        if status == "keep" and post > 0.0:
            if best is None or post < best:
                best = post
    return best if best is not None else 999.0


def parse_run_metrics() -> tuple[str, str, int, str]:
    text = RUN_LOG.read_text(encoding="utf-8", errors="ignore") if RUN_LOG.exists() else ""
    pre = re.findall(r"step:[0-9]+/20000 val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    post = re.findall(r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    art = re.findall(r"Total submission size int8\+zlib: ([0-9]+)", text)
    if not pre or not post or not art:
        return "0.0000", "0.0000", 0, "crash"

    pre_raw = float(pre[-1])
    post_raw = float(post[-1])
    artifact = int(art[-1])

    status = "discard"
    if post_raw < best_keep_post() and artifact <= ARTIFACT_LIMIT:
        status = "keep"

    return f"{pre_raw:.4f}", f"{post_raw:.4f}", artifact, status


def append_result(commit: str, pre: str, post: str, artifact: int, status: str, desc: str) -> None:
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{commit}\t{pre}\t{post}\t{artifact}\t{status}\t{desc}\n")


def append_learning(line: str) -> None:
    with LEARNINGS_FILE.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def commit_train_change(message: str) -> str:
    run(["git", "add", str(TRAIN_FILE)])
    run(["git", "commit", "-m", message, "--", str(TRAIN_FILE)])
    out = run(["git", "rev-parse", "--short", "HEAD"], capture=True)
    return out.stdout.strip()


def quantize_candidate(spec: ParamSpec, value: float) -> float:
    if spec.kind == "int":
        return float(int(round(value)))
    return round(value, spec.precision)


def current_values() -> dict[str, float]:
    text = read_train_file()
    out: dict[str, float] = {}
    for spec in PARAM_SPECS:
        m = re.search(rf'{spec.key}",\s*([0-9.]+)', text)
        if not m:
            raise RuntimeError(f"could not parse {spec.key}")
        out[spec.key] = float(m.group(1))
    return out


def choose_candidate(values: dict[str, float]) -> tuple[ParamSpec, float, float]:
    for mag in range(1, 20_001):
        for spec in PARAM_SPECS:
            base = values[spec.key]
            for sign in (1.0, -1.0):
                raw = base + sign * mag * spec.step
                raw = max(spec.min_value, min(spec.max_value, raw))
                cand = quantize_candidate(spec, raw)
                base_q = quantize_candidate(spec, base)
                if cand == base_q:
                    continue
                if already_tested(spec, cand):
                    continue
                return spec, cand, base_q
    raise RuntimeError("no untested candidate found in configured search space")


def commit_label(spec: ParamSpec) -> str:
    return spec.key.lower()


def log_label(spec: ParamSpec) -> str:
    return spec.key


def pretty_value(spec: ParamSpec, value: float) -> str:
    return format_value(spec, quantize_candidate(spec, value))


def apply_candidate(spec: ParamSpec, candidate: float) -> bool:
    set_value(spec, candidate)
    diff = subprocess.run(["git", "diff", "--quiet", "--", str(TRAIN_FILE)], cwd=str(ROOT))
    return diff.returncode != 0


def experiment_description(spec: ParamSpec, candidate: float, base: float) -> str:
    return f"daemon {spec.key}={pretty_value(spec, candidate)} (base={pretty_value(spec, base)})"


def learning_line(commit: str, spec: ParamSpec, candidate: float, base: float, pre: str, post: str, artifact: int, status: str) -> str:
    return (
        f"- {ts()} `{commit}`: tried `{spec.key}={pretty_value(spec, candidate)}` from "
        f"`{pretty_value(spec, base)}` -> pre/post `{pre}/{post}`, artifact `{artifact}`, status `{status}`."
    )


def revert_candidate(spec: ParamSpec, candidate: float, base: float) -> None:
    set_value(spec, base)
    commit_train_change(
        f"revert: {commit_label(spec)} {pretty_value(spec, candidate)} experiment "
        f"(restore {pretty_value(spec, base)})"
    )


def main() -> None:
    os.chdir(ROOT)
    ensure_files()

    while True:
        try:
            values = current_values()
            spec, candidate, base = choose_candidate(values)
            log(
                f"base_{commit_label(spec)}={pretty_value(spec, base)} "
                f"cand_{commit_label(spec)}={pretty_value(spec, candidate)}"
            )

            changed = apply_candidate(spec, candidate)
            if not changed:
                log(f"no diff for {commit_label(spec)}={pretty_value(spec, candidate)}, skipping")
                continue

            exp_commit = commit_train_change(
                f"exp: set {commit_label(spec)} to {pretty_value(spec, candidate)} (daemon sweep)"
            )
            log(
                f"run commit={exp_commit} "
                f"{commit_label(spec)}={pretty_value(spec, candidate)}"
            )

            num_gpus = int(run(["bash", "-lc", "nvidia-smi -L | wc -l"], capture=True).stdout.strip())
            with RUN_LOG.open("w", encoding="utf-8") as lf:
                rc = subprocess.run(
                    [
                        "timeout",
                        "930",
                        "torchrun",
                        "--standalone",
                        f"--nproc_per_node={num_gpus}",
                        "experiments/002_autoresearch/train_gpt.py",
                    ],
                    cwd=str(ROOT),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).returncode
            log(f"run_exit_code={rc} commit={exp_commit}")

            pre, post, artifact, status = parse_run_metrics()
            desc = experiment_description(spec, candidate, base)
            append_result(exp_commit, pre, post, artifact, status, desc)
            append_learning(learning_line(exp_commit, spec, candidate, base, pre, post, artifact, status))

            if status == "keep":
                log(f"KEEP commit={exp_commit} post={post} artifact={artifact}")
                run(["git", "push", "origin", "HEAD"], check=False)
                run(
                    [
                        "aws",
                        "s3",
                        "sync",
                        ".",
                        "s3://fuelos-autoresearch/latest/",
                        "--profile",
                        "fuelos",
                        "--exclude",
                        "data/*",
                        "--exclude",
                        ".git/*",
                        "--exclude",
                        "__pycache__/*",
                        "--exclude",
                        ".venv/*",
                    ],
                    check=False,
                )
            else:
                revert_candidate(spec, candidate, base)
                log(
                    f"{status.upper()} commit={exp_commit} "
                    f"restored_{commit_label(spec)}={pretty_value(spec, base)}"
                )

        except Exception as e:
            log(f"loop_error={type(e).__name__}: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
