#!/usr/bin/env python3
"""The sole r10 Phase-C executor: exact14 -> hidden opener -> conditional Stage-B.

No command-line flag selects Stage-B.  The supervisor creates one O_EXCL lock,
runs the fixed Stage-A cells, invokes the frozen static opener with stdout
suppressed, independently recomputes its severe-negative decision, and only
then runs the fixed 28-cell Stage-B schedule.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    CellKey, PHASE_ID, PROTOCOL_ID, cell_paths, claim_cell, stage_a_paths,
    validate_stage_a_cell_directory_set, verify_cell_exact, write_failed_once,
    write_json_exclusive, write_started,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r10_static import (  # noqa: E402
    MANIFEST_ENV, R10_CELL_ROOT, R10_RECEIPT_ROOT, RUN_LOCK, STATIC_MANIFEST,
    cell_schedule, create_run_lock, manifest_metadata, validate_static_manifest,
)


PIPELINE = ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r10_cell_pipeline.py"
OPENER = ROOT / "sua_exploration/scripts/open_m2_native_post33_phase_c_v5_r10_static.py"
STATUS = R10_RECEIPT_ROOT / "control/supervisor_terminal_status.json"
DATA_ROOT = ROOT / "SPINT-main/data/000953"
ACTIVE_PROCESSES: dict[CellKey, subprocess.Popen[bytes]] = {}
ACTIVE_LOCK = threading.Lock()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _review(path: Path) -> None:
    """Require a tiny immutable external review before any GPU child starts."""
    _require(path.is_file() and not path.is_symlink(), "r10 independent review is missing")
    _require((path.stat().st_mode & 0o777) == 0o444, "r10 independent review must be 0444")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "m2_post33_phase_c_v5_r10_independent_launch_review_v1",
        "status": "APPROVED_FOR_STATIC_R10_LAUNCH",
        "manifest": manifest_metadata(),
    }
    _require(payload == expected, "r10 independent review does not bind this manifest")


def _owner(key: CellKey) -> str:
    return f"r10-static-fold{key.fold}-seed{key.seed}-{key.arm}"


def _pipeline_command(key: CellKey, owner_token: str) -> list[str]:
    return [
        sys.executable, str(PIPELINE),
        "--cell-root", str(R10_CELL_ROOT.resolve()),
        "--workspace-root", str(ROOT.resolve()),
        "--data-root", str(DATA_ROOT.resolve(strict=True)),
        "--arm", key.arm, "--fold", str(key.fold), "--seed", str(key.seed),
        "--owner-token", owner_token,
    ]


def _terminate_other_active(except_key: CellKey) -> None:
    with ACTIVE_LOCK:
        active = list(ACTIVE_PROCESSES.items())
    for key, process in active:
        if key != except_key and process.poll() is None:
            process.terminate()


def _run_cell(key: CellKey, abort: threading.Event) -> None:
    if abort.is_set():
        raise RuntimeError("r10 sibling shard failed before this cell could start")
    owner_token = _owner(key)
    claim_cell(R10_CELL_ROOT, key, owner_token=owner_token)
    write_started(R10_CELL_ROOT, key, owner_token=owner_token)
    env = dict(os.environ)
    env[MANIFEST_ENV] = str(STATIC_MANIFEST.resolve())
    env["CUDA_VISIBLE_DEVICES"] = str(key.fold % 2)
    process = subprocess.Popen(
        _pipeline_command(key, owner_token), cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    with ACTIVE_LOCK:
        ACTIVE_PROCESSES[key] = process
    while process.poll() is None:
        if abort.is_set():
            process.terminate()
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            pass
    with ACTIVE_LOCK:
        ACTIVE_PROCESSES.pop(key, None)
    if process.returncode != 0:
        write_failed_once(
            R10_CELL_ROOT, key, owner_token=owner_token,
            failure_kind="child_nonzero_or_sibling_abort", return_code=process.returncode,
        )
        abort.set()
        _terminate_other_active(key)
        raise RuntimeError(f"r10 cell failed: {key.arm}/fold{key.fold}/seed{key.seed}")
    try:
        verify_cell_exact(R10_CELL_ROOT, key)
    except BaseException:
        paths = cell_paths(R10_CELL_ROOT, key)
        if not paths["failed"].exists() and not paths["completed"].exists():
            write_failed_once(
                R10_CELL_ROOT, key, owner_token=owner_token,
                failure_kind="zero_exit_without_exact_completion", return_code=0,
            )
        abort.set()
        _terminate_other_active(key)
        raise


def _run_stage(keys: tuple[CellKey, ...]) -> None:
    """Two fixed GPU shards: ordered within GPU, concurrent across GPUs."""
    abort = threading.Event()
    shards = tuple(tuple(key for key in keys if key.fold % 2 == gpu) for gpu in (0, 1))
    _require(all(shard for shard in shards), "r10 fixed GPU shard unexpectedly empty")
    def run_shard(shard: tuple[CellKey, ...]) -> None:
        for key in shard:
            if abort.is_set():
                return
            _run_cell(key, abort)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="r10-gpu") as pool:
        futures = [pool.submit(run_shard, shard) for shard in shards]
        done, _ = wait(futures)
        errors = [future.exception() for future in done if future.exception() is not None]
    if errors:
        raise errors[0]


def _verify_exact_stage_a() -> None:
    """Score-blind Stage-A cardinality/completion/failure check."""
    validate_stage_a_cell_directory_set(R10_CELL_ROOT)
    for key in cell_schedule()["stage_a"]:
        paths = cell_paths(R10_CELL_ROOT, key)
        _require(not paths["failed"].exists(), "r10 Stage-A contains a failed cell")
        verify_cell_exact(R10_CELL_ROOT, key)


def _hidden_opener(*, opening: str) -> Mapping[str, Any]:
    """Invoke the fixed r10 opener without forwarding an endpoint or R²."""
    process = subprocess.run(
        [sys.executable, str(OPENER), "--cell-root", str(R10_CELL_ROOT.resolve()), "--opening", opening],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    _require(process.returncode == 0, f"r10 hidden {opening} opener failed")
    if opening == "stage_a":
        decision_path = stage_a_paths(R10_CELL_ROOT)["decision"]
        _require(decision_path.is_file(), "r10 Stage-A opener did not write decision")
        return json.loads(decision_path.read_text(encoding="utf-8"))
    return {"opened": "full"}


def _recompute_severe_negative(decision: Mapping[str, Any]) -> bool:
    rows = decision.get("paired_rows")
    _require(isinstance(rows, list) and len(rows) == 7, "r10 hidden Stage-A decision has wrong paired rows")
    deltas: list[float] = []
    for fold, row in enumerate(rows):
        _require(isinstance(row, Mapping) and row.get("fold") == fold, "r10 decision row identity mismatch")
        delta = row.get("delta")
        _require(isinstance(delta, (int, float)) and not isinstance(delta, bool), "r10 decision delta invalid")
        deltas.append(float(delta))
    mean42 = sum(deltas) / len(deltas)
    pos42 = sum(value > 0 for value in deltas)
    severe = mean42 <= -0.03 or pos42 <= 1
    _require(decision.get("mean42") == mean42 and decision.get("pos42") == pos42, "r10 opener aggregate mismatch")
    _require(decision.get("severe_negative_triggered") is severe, "r10 opener severe-negative mismatch")
    expected = "seed42_severe_negative_futility_stop" if severe else "continue_without_positive_claim"
    _require(decision.get("decision") == expected and decision.get("positive_claim_made") is False, "r10 opener branch mismatch")
    return severe


def _terminal_status(state: str) -> None:
    _require(not STATUS.exists(), "r10 terminal status is write-once")
    write_json_exclusive(STATUS, {
        "schema": "m2_post33_phase_c_v5_r10_supervisor_terminal_status_v1",
        "state": state,
        "manifest": manifest_metadata(),
        "score_or_r2_printed": False,
        "manual_branch_selection": False,
    })
    os.chmod(STATUS, 0o600)


def _dry_run() -> dict[str, Any]:
    manifest = validate_static_manifest(require_lock=False)
    return {
        "schema": "m2_post33_phase_c_v5_r10_supervisor_dry_run_v1",
        "status": "PASS_STATIC_NO_GPU",
        "manifest": manifest_metadata(),
        "stage_a_cell_count": len(cell_schedule()["stage_a"]),
        "stage_b_cell_count": len(cell_schedule()["stage_b"]),
        "cell_root_created": R10_CELL_ROOT.exists(),
        "run_lock_created": RUN_LOCK.exists(),
        "gpu_used": False,
        "formal_data_opened": False,
        "score_data_opened": False,
        "manual_stage_b_option_present": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--independent-review", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.execute:
        print(json.dumps(_dry_run(), sort_keys=True))
        return 0
    _require(args.independent_review is not None, "r10 --execute requires immutable independent review")
    _review(args.independent_review)
    create_run_lock(supervisor_pid=os.getpid())
    _run_stage(cell_schedule()["stage_a"])
    _verify_exact_stage_a()
    decision = _hidden_opener(opening="stage_a")
    if _recompute_severe_negative(decision):
        _terminal_status("STOP_SEVERE_NEGATIVE_AFTER_EXACT14")
        return 0
    _run_stage(cell_schedule()["stage_b"])
    _hidden_opener(opening="full")
    _terminal_status("COMPLETED_FULL_MATRIX_HIDDEN_OPENING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
