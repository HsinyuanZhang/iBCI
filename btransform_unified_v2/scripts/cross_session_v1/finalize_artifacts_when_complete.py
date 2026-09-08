#!/usr/bin/env python3
"""Wait for the 33-cell continuation ledger, then produce review artifacts.

This wrapper never starts, restarts, kills, or scores a training cell.  Only
after the continuation controller has naturally exited with a complete grid
does it verify artifacts, atomically install the completed ledger as canonical,
and invoke the existing source-pairing audit and figure collector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


ARMS = ("Z_NONE", "B_ACTIVITY_ONLY", "D_JOINT")
FOLDS = {
    "m1": {"ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928"},
    "m2": {"source7_ext4"},
    "h1": {"19250101", "19250108", "19250113", "19250115", "19250119", "19250120"},
}
ACTIVE = {"PENDING", "PREPARING", "TRAINING", "SCORING"}
LIVE_DETAIL = ACTIVE - {"PENDING"}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
SOURCE_AUDITOR = SCRIPT_DIR / "audit_source_pairing.py"
FIGURE_COLLECTOR = SCRIPT_DIR / "summarize_figure.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required JSON is missing: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON must be an object: {path}")
    return value


def parse_json_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON must be an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_bytes(path: Path, payload: bytes) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def canonical_cells(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    cells = ledger.get("cells")
    require(ledger.get("schema") == "cross_session_calibration_program_v1", "continuation ledger schema mismatch")
    require(ledger.get("primary_cell_count") == 33 and isinstance(cells, list) and len(cells) == 33,
            "continuation ledger must enumerate exactly 33 cells")
    seen: set[tuple[str, str, str, int]] = set()
    for cell in cells:
        require(isinstance(cell, dict), "continuation cell is not an object")
        task, fold, arm, seed = cell.get("dataset"), cell.get("fold"), cell.get("arm"), cell.get("seed")
        require(task in FOLDS and fold in FOLDS[task] and arm in ARMS and seed == 42,
                f"invalid canonical cell identity: {cell}")
        key = (task, fold, arm, seed)
        require(key not in seen, f"duplicate continuation cell: {key}")
        seen.add(key)
    expected = {(task, fold, arm, 42) for task, folds in FOLDS.items() for fold in folds for arm in ARMS}
    require(seen == expected, "continuation ledger has missing or unexpected canonical cells")
    return cells


def proc_state(pid: int, expected_script: str) -> str:
    """Return absent, exiting, or live after guarding against PID reuse."""
    require(pid > 0, f"invalid controller PID: {pid}")
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return "absent"
    try:
        raw = cmdline.read_bytes()
    except FileNotFoundError:
        return "absent"
    argv = [part for part in raw.decode(errors="replace").split("\x00") if part]
    if not argv:
        return "exiting"
    require(any(Path(part).name == expected_script for part in argv),
            f"PID {pid} exists but is not {expected_script}: {argv}")
    return "live"


def heartbeat_summary(cell: dict[str, Any]) -> dict[str, Any] | None:
    run = cell.get("run")
    if not isinstance(run, str):
        return None
    path = Path(run) / "heartbeat.json"
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"state": "unreadable"}
    if not isinstance(row, dict):
        return {"state": "invalid"}
    return {key: row[key] for key in ("event", "epoch", "step", "elapsed_seconds") if key in row}


def progress_snapshot(cells: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    live: list[dict[str, Any]] = []
    for cell in cells:
        status = cell.get("status")
        counts[str(status)] = counts.get(str(status), 0) + 1
        if status in LIVE_DETAIL:
            live.append({"cell": [cell["dataset"], cell["fold"], cell["arm"]], "pid": cell.get("pid"),
                         "stage": cell.get("stage"), "heartbeat": heartbeat_summary(cell)})
    return {"counts": dict(sorted(counts.items())), "live": live}


def check_source_seal(seal: dict[str, Any]) -> None:
    files = seal.get("files")
    require(isinstance(files, dict) and files, "execution source seal lacks file hashes")
    for raw, expected in files.items():
        path = Path(raw)
        require(path.is_file() and isinstance(expected, str) and len(expected) == 64, f"invalid sealed entry: {path}")
        require(sha(path) == expected, f"sealed source drift: {path}")


def fixed_inputs(root: Path) -> dict[str, Any]:
    seal_path = root / "execution_source_seal.json"
    handoff_path = root / "prior_queue_completion_handoff_audit.json"
    result = {
        "execution_source_seal": {"path": str(seal_path), "sha256": sha(seal_path)},
        "source_auditor": {"path": str(SOURCE_AUDITOR), "sha256": sha(SOURCE_AUDITOR)},
        "figure_collector": {"path": str(FIGURE_COLLECTOR), "sha256": sha(FIGURE_COLLECTOR)},
        "pipeline_wrapper": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
        "handoff": {"path": str(handoff_path), "sha256": sha(handoff_path)},
    }
    check_source_seal(load_json(seal_path))
    return result


def recheck_fixed_inputs(fixed: dict[str, Any]) -> None:
    for label in ("execution_source_seal", "source_auditor", "figure_collector", "pipeline_wrapper", "handoff"):
        entry = fixed[label]
        path = Path(entry["path"])
        require(path.is_file() and sha(path) == entry["sha256"], f"fixed input drift before execution: {label}")
    check_source_seal(load_json(Path(fixed["execution_source_seal"]["path"])))


def verify_handoff(root: Path, fixed: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    original_path = root / "program.json"
    original_bytes = original_path.read_bytes()
    handoff = load_json(Path(fixed["handoff"]["path"]))
    require(handoff.get("status") == "PASSED", "prior queue handoff audit did not pass")
    require(sha(original_path) == handoff.get("prior_program_sha256"), "canonical original program SHA differs from handoff audit")
    prior_pid = handoff.get("prior_controller_pid")
    require(isinstance(prior_pid, int) and proc_state(prior_pid, "run_program_overlap.py") == "absent",
            "prior overlap controller has not naturally exited")
    return handoff, original_bytes


def wait_for_completion(root: Path, poll_seconds: int, fixed: dict[str, Any]) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    continuation = root / "program_continuation.json"
    last: str | None = None
    last_report = 0.0
    while True:
        ledger_bytes = continuation.read_bytes()
        ledger = parse_json_bytes(continuation, ledger_bytes)
        cells = canonical_cells(ledger)
        require(not any(cell.get("status") == "FAILED" for cell in cells), "continuation ledger contains a FAILED cell")
        unknown = {cell.get("status") for cell in cells} - ACTIVE - {"COMPLETED"}
        require(not unknown, f"continuation ledger has unexpected status values: {unknown}")
        controller_pid = ledger.get("queue_pid")
        require(isinstance(controller_pid, int), "continuation ledger lacks controller queue_pid")
        controller = proc_state(controller_pid, "run_program_continuation.py")
        snapshot = progress_snapshot(cells)
        status_stamp = json.dumps({"controller": controller, "counts": snapshot["counts"],
                                   "live": [{key: row.get(key) for key in ("cell", "pid", "stage")} for row in snapshot["live"]]}, sort_keys=True)
        moment = time.monotonic()
        if status_stamp != last or moment - last_report >= 60:
            print(json.dumps({"event": "waiting", "time": now(), "controller": controller, **snapshot}, sort_keys=True), flush=True)
            last, last_report = status_stamp, moment
        complete = all(cell.get("status") == "COMPLETED" for cell in cells)
        if complete:
            if controller != "absent":
                time.sleep(poll_seconds)
                continue
            require(ledger.get("status") == "PRIMARY_GRID_COMPLETED_AWAITING_FIGURE",
                    "terminal continuation ledger lacks PRIMARY_GRID_COMPLETED_AWAITING_FIGURE status")
            recheck_fixed_inputs(fixed)
            handoff, original = verify_handoff(root, fixed)
            return ledger, ledger_bytes, {"handoff": handoff, "original_program_bytes": original}
        if controller == "absent":
            # An atomic ledger replacement can race a controller's final exit.
            # Re-read once; only an unchanged incomplete snapshot is terminal.
            if continuation.read_bytes() != ledger_bytes:
                continue
            raise RuntimeError("continuation controller disappeared before the grid completed")
        time.sleep(poll_seconds)


def validate_score_artifacts(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sys.path.insert(0, str(SCRIPT_DIR))
    import summarize_figure  # Existing strict artifact reader; import only after completion.
    checked: list[dict[str, Any]] = []
    for cell in cells:
        extracted = summarize_figure.extract(cell)
        expected = cell.get("score_sha256")
        require(isinstance(expected, str) and extracted.get("receipt_sha256") == expected,
                f"continuation score receipt SHA mismatch: {cell['dataset']}/{cell['fold']}/{cell['arm']}")
        checked.append({"cell": [cell["dataset"], cell["fold"], cell["arm"]], "score_receipt_sha256": expected})
    return checked


def tree_hashes(directory: Path) -> dict[str, str]:
    require(directory.is_dir(), f"expected output directory is missing: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    require(files, f"output directory has no files: {directory}")
    return {str(path.resolve()): sha(path) for path in files}


def run_checked(command: list[str], cwd: Path) -> None:
    print(json.dumps({"event": "subprocess", "time": now(), "command": command}, sort_keys=True), flush=True)
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(WORKSPACE_ROOT), *([old] if old else [])])
    subprocess.run(command, cwd=cwd, env=env, check=True)


def merge_and_finalize(root: Path, ledger: dict[str, Any], continuation_bytes: bytes, context: dict[str, Any], fixed: dict[str, Any]) -> None:
    prior = root / "prior_queue_program.json"
    canonical = root / "program.json"
    pairing = root / "source_pairing_audit.json"
    figure = root / "final_figure"
    merge_receipt = root / "ledger_merge_receipt.json"
    receipt = root / "post_training_pipeline_receipt.json"
    for path in (prior, pairing, figure, merge_receipt, receipt):
        require(not path.exists(), f"refusing to overwrite existing finalization artifact: {path}")
    # No ledger write occurs until source seals, handoff, and actual score artifacts
    # have all been verified.  Copy exact bytes rather than reserializing either ledger.
    original_bytes = context["original_program_bytes"]
    require(sha(canonical) == context["handoff"].get("prior_program_sha256"), "canonical original changed before merge")
    merge_started_at = now()
    atomic_bytes(prior, original_bytes)
    archived_at = now()
    atomic_bytes(canonical, continuation_bytes)
    canonical_installed_at = now()
    merge = {
        "schema": "cross_session_ledger_merge_v1", "status": "COMPLETED", "merge_started_at": merge_started_at,
        "prior_program_archived_at": archived_at, "canonical_program_installed_at": canonical_installed_at,
        "prior_queue_program": {"path": str(prior), "sha256": sha(prior)},
        "canonical_program": {"path": str(canonical), "sha256": sha(canonical)},
        "continuation_program": {"path": str(root / "program_continuation.json"), "sha256": sha(root / "program_continuation.json")},
        "original_program_sha256_before_merge": hashlib.sha256(original_bytes).hexdigest(),
    }
    require(merge["prior_queue_program"]["sha256"] == merge["original_program_sha256_before_merge"], "prior program archival byte drift")
    require(merge["canonical_program"]["sha256"] == hashlib.sha256(continuation_bytes).hexdigest(), "canonical merge byte drift")
    atomic_json(merge_receipt, merge)
    run_checked([sys.executable, str(SOURCE_AUDITOR), "--root", str(root), "--out", str(pairing)], PROJECT_ROOT)
    audit = load_json(pairing)
    require(audit.get("status") == "PASS", "source pairing audit did not pass")
    run_checked([sys.executable, str(FIGURE_COLLECTOR), "--root", str(root), "--out", str(figure)], PROJECT_ROOT)
    output_hashes = tree_hashes(figure)
    final = {
        "schema": "cross_session_post_training_pipeline_v1", "status": "ARTIFACTS_READY_FOR_REVIEW",
        "created_at": now(),
        "scope": "Artifacts are ready for root visual inspection and paper integration; this status is not paper completion or goal completion.",
        "fixed_inputs_at_start": fixed,
        "ledger_merge_receipt": {"path": str(merge_receipt), "sha256": sha(merge_receipt)},
        "prior_queue_program": merge["prior_queue_program"], "canonical_program": merge["canonical_program"],
        "continuation_program": merge["continuation_program"],
        "source_pairing_audit": {"path": str(pairing), "sha256": sha(pairing)},
        "final_figure_files_sha256": output_hashes,
        "verified_score_receipts": context["score_checks"],
        "no_training_restart_no_paper_write_no_push": True,
    }
    atomic_json(receipt, final)
    print(json.dumps({"event": "artifacts_ready_for_review", "time": now(), "receipt": str(receipt)}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="absolute cross_session_v1 results directory")
    parser.add_argument("--poll-seconds", type=int, default=15, help="wait interval from 1 through 30 seconds")
    args = parser.parse_args()
    require(args.root.is_absolute() and args.root.is_dir(), "--root must be an existing absolute results directory")
    require(1 <= args.poll_seconds <= 30, "--poll-seconds must be in [1,30]")
    root = args.root.resolve()
    fixed = fixed_inputs(root)
    ledger, continuation_bytes, context = wait_for_completion(root, args.poll_seconds, fixed)
    context["score_checks"] = validate_score_artifacts(canonical_cells(ledger))
    # The continuation ledger must not change after it was observed terminal.
    require((root / "program_continuation.json").read_bytes() == continuation_bytes,
            "continuation ledger changed after terminal observation")
    recheck_fixed_inputs(fixed)
    merge_and_finalize(root, ledger, continuation_bytes, context, fixed)


if __name__ == "__main__":
    main()
