#!/usr/bin/env python3
"""Fail-closed RT-to-H-LS early-v2 source-training handoff.

The default invocation is a read-only dry run.  The explicit execution path
waits for the two named RT tmux workers to disappear, binds one immutable
15-fold RT aggregate, proves both requested GPUs and the RT writer path are
idle, runs the existing early-v2 two-GPU executor, and terminalizes the five
fixed epoch-49 source checkpoints with the existing no-target audit.

This module never opens target data and never calls the post-upstream binder or
either evaluator.  It deliberately performs no cleanup: claims, run
directories, logs, and failure evidence remain in place after every failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES,
    EARLY_LAUNCH_SCHEMA,
    EARLY_LAUNCH_STATUS,
    EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS,
    read_immutable_json,
    sha256_file,
)


RT_WORKERS = ("rt_xls_v2_gpu0_desc", "rt_xls_v2_gpu1_asc")
HANDOFF_TMUX = "h1_hls_early_v2_after_rt_handoff"
EXECUTOR = ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_source_executor.py"
TERMINAL_AUDIT = ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_source_terminal_audit.py"
CHECKPOINT_RELATIVE = Path("checkpoints/fixed_epoch50/epoch_049.ckpt")
CLAIM_FILENAME = "H1_HLS_EARLY_V2_AFTER_RT_HANDOFF_v1.claim.json"
LOG_FILENAME = "H1_HLS_EARLY_V2_AFTER_RT_HANDOFF_v1.events.jsonl"
SUBPROCESS_LOG_FILENAME = "H1_HLS_EARLY_V2_AFTER_RT_HANDOFF_v1.subprocess.log"
TERMINAL_FILENAME = "H1_HLS_EARLY_V2_AFTER_RT_HANDOFF_v1.terminal.json"
HANDOFF_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_after_rt_handoff_v1"
HANDOFF_CLAIM_STATUS = "CLAIMED_WAITING_FOR_RT_THEN_SOURCE_ONLY"
HANDOFF_PASS_STATUS = "PASS_H1_HLS_EARLY_V2_AFTER_RT_SOURCE_TERMINALIZED_NO_TARGET"
HANDOFF_FAIL_STATUS = "FAILED_H1_HLS_EARLY_V2_AFTER_RT_PRESERVED_FORENSIC_STATE"
WRITER_CLAIM_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_writer_claim_v1"
WRITER_CLAIM_STATUS = "CLAIMED_BEFORE_SOURCE_PROCESS_START"


class AfterRtHandoffError(RuntimeError):
    """A fail-closed handoff invariant was not satisfied."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AfterRtHandoffError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive_json(path: Path, body: Mapping[str, Any]) -> str:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(body), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise AfterRtHandoffError(f"refusing repeated immutable handoff output: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except BaseException:
        # Keep even a partial exclusive file as collision/forensic evidence.
        raise
    return _sha256(path)


def _append_event(log_path: Path, event: str, **fields: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time_unix": time.time(), "event": event, **fields}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(value) for value in command], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )


def _run_streamed(command: Sequence[str], *, log: Path, label: str) -> int:
    """Run a long GPU command with stdout/stderr streamed directly to disk."""

    rendered = [str(value) for value in command]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"label": label, "command": rendered, "event": "START"},
                                sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        result = subprocess.run(
            rendered, cwd=ROOT, text=True, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
        handle.write(json.dumps({"label": label, "returncode": result.returncode,
                                 "event": "EXIT"}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return int(result.returncode)


def _session_exists(name: str) -> bool:
    result = _run(("tmux", "has-session", "-t", f"={name}"))
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise AfterRtHandoffError(
        f"tmux inspection failed for {name}: rc={result.returncode}: {result.stderr.strip()}"
    )


def _shared_deadline(*, max_wait_seconds: float,
                     deadline_monotonic: float | None) -> float | None:
    _need(max_wait_seconds >= 0, "maximum wait cannot be negative")
    if deadline_monotonic is not None:
        return deadline_monotonic
    return None if max_wait_seconds == 0 else time.monotonic() + max_wait_seconds


def _wait_poll(*, poll_seconds: float, deadline_monotonic: float | None,
               timeout_message: str) -> None:
    _need(poll_seconds >= 0, "poll seconds cannot be negative")
    delay = poll_seconds
    if deadline_monotonic is not None:
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise AfterRtHandoffError(timeout_message)
        delay = min(delay, remaining)
    time.sleep(delay)


def wait_for_rt_workers(
    *, workers: Sequence[str] = RT_WORKERS, poll_seconds: float = 30.0,
    max_wait_seconds: float = 0.0, deadline_monotonic: float | None = None,
    event_log: Path | None = None,
) -> None:
    """Wait only for exact RT worker names; zero timeout means unbounded."""

    deadline = _shared_deadline(max_wait_seconds=max_wait_seconds,
                                deadline_monotonic=deadline_monotonic)
    while True:
        alive = [name for name in workers if _session_exists(name)]
        if not alive:
            if event_log is not None:
                _append_event(event_log, "rt_workers_cleanly_absent", workers=list(workers))
            return
        if event_log is not None:
            _append_event(event_log, "waiting_for_rt_workers", alive=alive)
        _wait_poll(
            poll_seconds=poll_seconds, deadline_monotonic=deadline,
            timeout_message=f"RT workers did not cleanly disappear before shared timeout: {alive}",
        )


def _fold_count(body: Mapping[str, Any]) -> int | None:
    for key in ("fold_count", "n_folds", "num_folds", "completed_fold_count"):
        value = body.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for key in ("folds", "per_fold", "fold_results", "results_by_fold", "rows"):
        value = body.get(key)
        if isinstance(value, (list, tuple, dict)):
            return len(value)
    aggregate = body.get("aggregate")
    if isinstance(aggregate, Mapping):
        return _fold_count(aggregate)
    return None


def verify_rt_aggregate(
    path: Path, *, pass_status: str, expected_sha256: str | None = None,
) -> dict[str, Any]:
    supplied = Path(path)
    _need(not supplied.is_symlink(), f"RT 15-fold aggregate cannot be a symlink: {supplied}")
    candidate = supplied.resolve()
    _need(candidate.is_file(), f"RT 15-fold aggregate is missing: {candidate}")
    _need(stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          f"RT 15-fold aggregate must be immutable mode 0444: {candidate}")
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AfterRtHandoffError(f"RT aggregate is not readable JSON: {candidate}") from error
    _need(isinstance(body, Mapping), "RT aggregate must be a JSON object")
    _need(body.get("status") == pass_status,
          f"RT aggregate status is not the exact admitted PASS status: {body.get('status')!r}")
    _need(_fold_count(body) == 15, "RT aggregate does not prove an exact 15-fold result")
    digest = _sha256(candidate)
    if expected_sha256 is not None:
        _need(len(expected_sha256) == 64 and all(char in "0123456789abcdef" for char in expected_sha256),
              "expected RT aggregate SHA-256 must be 64 lowercase hex characters")
        _need(digest == expected_sha256, "RT aggregate mutated or differs from the expected SHA-256")
    return {"path": str(candidate), "sha256": digest, "status": pass_status, "fold_count": 15}


def wait_for_rt_aggregate(
    path: Path, *, pass_status: str, expected_sha256: str | None = None,
    poll_seconds: float = 30.0, max_wait_seconds: float = 0.0,
    deadline_monotonic: float | None = None, event_log: Path | None = None,
) -> dict[str, Any]:
    """Wait for absence only; any aggregate that appears is judged exactly once.

    A producer may atomically publish the final aggregate just after its tmux
    worker exits.  Absence is therefore waitable.  A present symlink, mutable
    file, malformed JSON, wrong status/fold count, or wrong SHA is terminal and
    cannot be replaced under this handoff claim.
    """

    deadline = _shared_deadline(max_wait_seconds=max_wait_seconds,
                                deadline_monotonic=deadline_monotonic)
    candidate = Path(path)
    while True:
        if os.path.lexists(str(candidate)):
            admitted = verify_rt_aggregate(
                candidate, pass_status=pass_status, expected_sha256=expected_sha256,
            )
            if event_log is not None:
                _append_event(event_log, "rt_aggregate_admitted", **admitted)
            return admitted
        if event_log is not None:
            _append_event(event_log, "waiting_for_rt_aggregate", aggregate=str(candidate.resolve()))
        _wait_poll(
            poll_seconds=poll_seconds, deadline_monotonic=deadline,
            timeout_message=f"RT 15-fold aggregate did not appear before shared timeout: {candidate.resolve()}",
        )


def _gpu_compute_owners(gpu_devices: Sequence[str]) -> dict[str, list[str]]:
    _need(len(gpu_devices) == 2 and len(set(gpu_devices)) == 2,
          "handoff requires two distinct physical GPU devices")
    occupied: dict[str, list[str]] = {}
    for device in gpu_devices:
        result = _run((
            "nvidia-smi", "-i", str(device),
            "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits",
        ))
        _need(result.returncode == 0,
              f"cannot inspect GPU {device}: rc={result.returncode}: {result.stderr.strip()}")
        owners = [line.strip() for line in result.stdout.splitlines()
                  if line.strip() and "no running processes" not in line.lower()]
        if owners:
            occupied[str(device)] = owners
    return occupied


def assert_gpus_idle(gpu_devices: Sequence[str]) -> None:
    occupied = _gpu_compute_owners(gpu_devices)
    _need(not occupied, f"GPU compute owner(s) still exist: {occupied}")


def _rt_train_writers(markers: Sequence[str]) -> list[str]:
    result = _run(("ps", "-eo", "pid=,args="))
    _need(result.returncode == 0, f"cannot inspect RT train writers: {result.stderr.strip()}")
    own_pids = {os.getpid(), os.getppid()}
    found: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if pid_text.isdigit() and int(pid_text) in own_pids:
            continue
        lowered = command.lower()
        if any(marker.lower() in lowered for marker in markers) and any(
            token in lowered for token in ("train.py", "trainer", "lightning", "xls_v2")
        ):
            found.append(stripped)
    return found


def assert_no_rt_train_writer(markers: Sequence[str]) -> None:
    found = _rt_train_writers(markers)
    _need(not found, f"RT train writer process still exists: {found}")


def wait_for_gpu_and_rt_writer_release(
    *, gpu_devices: Sequence[str], rt_writer_markers: Sequence[str],
    poll_seconds: float = 30.0, max_wait_seconds: float = 0.0,
    deadline_monotonic: float | None = None, event_log: Path | None = None,
) -> None:
    """Wait for RT's short CUDA/finalizer tail under the shared deadline."""

    deadline = _shared_deadline(max_wait_seconds=max_wait_seconds,
                                deadline_monotonic=deadline_monotonic)
    while True:
        owners = _gpu_compute_owners(gpu_devices)
        writers = _rt_train_writers(rt_writer_markers)
        if not owners and not writers:
            if event_log is not None:
                _append_event(event_log, "gpu_and_rt_writer_gate_passed",
                              gpus=list(gpu_devices))
            return
        if event_log is not None:
            _append_event(event_log, "waiting_for_gpu_and_rt_writer_release",
                          gpu_compute_owners=owners, rt_writers=writers)
        _wait_poll(
            poll_seconds=poll_seconds, deadline_monotonic=deadline,
            timeout_message=("GPU/RT finalizer release did not complete before shared timeout: "
                             f"gpu_compute_owners={owners}, rt_writers={writers}"),
        )


def _terminal_path(terminal_root: Path, date: str) -> Path:
    return terminal_root / f"H1_HLS_EARLY_V2_{date}_SOURCE_E49_TERMINAL_v1.json"


def build_plan(
    *, launch_receipt: Path, run_root: Path, claim_root: Path, terminal_root: Path,
    state_root: Path, rt_aggregate: Path, rt_aggregate_pass_status: str,
    rt_aggregate_sha256: str | None, gpu0_device: str, gpu1_device: str,
    python_executable: str, handoff_tmux: str = HANDOFF_TMUX,
) -> dict[str, Any]:
    launch_path, launch, launch_sha = read_immutable_json(
        launch_receipt, schema=EARLY_LAUNCH_SCHEMA, status=EARLY_LAUNCH_STATUS,
    )
    _need(tuple(launch.get("fixed_grid", ())) == DATES,
          "early-v2 launch receipt does not bind the exact five-date grid")
    source_rows = launch.get("source_preflights")
    _need(isinstance(source_rows, Mapping) and tuple(source_rows) == DATES,
          "early-v2 launch receipt lacks the ordered five source preflights")
    run_root, claim_root = Path(run_root).resolve(), Path(claim_root).resolve()
    terminal_root, state_root = Path(terminal_root).resolve(), Path(state_root).resolve()
    dates: dict[str, Any] = {}
    for date in DATES:
        row = source_rows[date]
        _need(isinstance(row, Mapping) and isinstance(row.get("path"), str),
              f"early-v2 launch source preflight is malformed: {date}")
        run_dir = run_root / date
        writer_claim = claim_root / f"H1_HLS_EARLY_V2_{date}_WRITER_CLAIM_v1.json"
        terminal = _terminal_path(terminal_root, date)
        for label, candidate in (("run", run_dir), ("writer claim", writer_claim), ("terminal", terminal)):
            _need(not os.path.lexists(str(candidate)),
                  f"existing {label} blocks repeated H-LS handoff for {date}: {candidate}")
        checkpoint = run_dir / CHECKPOINT_RELATIVE
        dates[date] = {
            "run_dir": str(run_dir), "writer_claim": str(writer_claim),
            "source_preflight": str(Path(row["path"]).resolve()),
            "checkpoint": str(checkpoint), "terminal_receipt": str(terminal),
        }
    claim = state_root / CLAIM_FILENAME
    event_log = state_root / LOG_FILENAME
    subprocess_log = state_root / SUBPROCESS_LOG_FILENAME
    handoff_terminal = state_root / TERMINAL_FILENAME
    for label, candidate in (
        ("handoff claim", claim), ("handoff event log", event_log),
        ("handoff subprocess log", subprocess_log), ("handoff terminal", handoff_terminal),
    ):
        _need(not os.path.lexists(str(candidate)), f"existing {label} blocks repeated handoff: {candidate}")
    executor_command = [
        python_executable, str(EXECUTOR), "--launch-receipt", str(launch_path),
        "--run-root", str(run_root), "--claim-root", str(claim_root),
        "--gpu0-device", str(gpu0_device), "--gpu1-device", str(gpu1_device),
        "--python-executable", python_executable, "--execute-source-training",
    ]
    return {
        "schema": HANDOFF_SCHEMA, "status": "DRY_RUN_NOT_EXECUTED",
        "launch_receipt": {"path": str(launch_path), "sha256": launch_sha},
        "rt_gate": {"workers": list(RT_WORKERS), "aggregate": str(Path(rt_aggregate).resolve()),
                    "pass_status": rt_aggregate_pass_status,
                    "expected_sha256": rt_aggregate_sha256},
        "handoff_tmux": handoff_tmux, "gpus": [str(gpu0_device), str(gpu1_device)],
        "paths": {"claim": str(claim), "event_log": str(event_log),
                  "subprocess_log": str(subprocess_log), "terminal": str(handoff_terminal)},
        "dates": dates, "executor_command": executor_command,
        "scope": {"target_opened": 0, "binder_called": False, "evaluator_called": False,
                  "cleanup_on_failure": False},
    }


def _record_subprocess(log: Path, label: str, command: Sequence[str], result: subprocess.CompletedProcess[str]) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"label": label, "command": list(command), "returncode": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _check_post_executor_rows(plan: Mapping[str, Any]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for date in DATES:
        row = plan["dates"][date]
        run_dir, expected = Path(row["run_dir"]), Path(row["checkpoint"])
        matches = [path.resolve() for path in run_dir.rglob("epoch_049.ckpt")
                   if path.is_file() and not path.is_symlink()]
        _need(len(matches) == 1, f"{date} must contain exactly one fixed e49 checkpoint; found {len(matches)}")
        _need(matches[0] == expected.resolve(), f"{date} fixed e49 checkpoint is outside canonical path")
        claim_path, claim, _claim_sha = read_immutable_json(
            row["writer_claim"], schema=WRITER_CLAIM_SCHEMA, status=WRITER_CLAIM_STATUS,
        )
        _need(claim.get("outer_date") == date and Path(str(claim.get("run_dir", ""))).resolve() == run_dir.resolve(),
              f"{date} writer claim does not bind its canonical run")
        _need(str(claim_path) == str(Path(row["writer_claim"]).resolve()),
              f"{date} writer claim path drift")
        checkpoints[date] = matches[0]
    return checkpoints


def execute(
    plan: Mapping[str, Any], *, poll_seconds: float, max_wait_seconds: float,
    rt_writer_markers: Sequence[str], python_executable: str,
) -> dict[str, Any]:
    paths = {key: Path(value) for key, value in plan["paths"].items()}
    _need(not _session_exists(str(plan["handoff_tmux"])),
          f"same-name handoff tmux already exists: {plan['handoff_tmux']}")
    claim_body = {
        "schema": HANDOFF_SCHEMA, "status": HANDOFF_CLAIM_STATUS,
        "launch_receipt": plan["launch_receipt"], "rt_gate": plan["rt_gate"],
        "gpus": plan["gpus"], "dates": list(DATES),
        "scope": plan["scope"],
    }
    claim_sha = _write_exclusive_json(paths["claim"], claim_body)
    try:
        _append_event(paths["event_log"], "handoff_claimed", claim_sha256=claim_sha)
        shared_deadline = _shared_deadline(max_wait_seconds=max_wait_seconds,
                                           deadline_monotonic=None)
        wait_for_rt_workers(poll_seconds=poll_seconds, deadline_monotonic=shared_deadline,
                            event_log=paths["event_log"])
        aggregate = wait_for_rt_aggregate(
            Path(plan["rt_gate"]["aggregate"]), pass_status=str(plan["rt_gate"]["pass_status"]),
            expected_sha256=plan["rt_gate"].get("expected_sha256"),
            poll_seconds=poll_seconds, deadline_monotonic=shared_deadline,
            event_log=paths["event_log"],
        )
        wait_for_gpu_and_rt_writer_release(
            gpu_devices=plan["gpus"], rt_writer_markers=rt_writer_markers,
            poll_seconds=poll_seconds, deadline_monotonic=shared_deadline,
            event_log=paths["event_log"],
        )

        # Re-hash immediately before source training so a post-admission mutation fails closed.
        verify_rt_aggregate(Path(aggregate["path"]), pass_status=aggregate["status"],
                            expected_sha256=aggregate["sha256"])
        executor_command = list(plan["executor_command"])
        executor_returncode = _run_streamed(
            executor_command, log=paths["subprocess_log"], label="early_v2_source_executor",
        )
        _need(executor_returncode == 0,
              f"early-v2 source executor failed: rc={executor_returncode}; see {paths['subprocess_log']}")
        _append_event(paths["event_log"], "five_date_source_executor_completed")

        # Bind the RT completion evidence across the long GPU section as well.
        verify_rt_aggregate(Path(aggregate["path"]), pass_status=aggregate["status"],
                            expected_sha256=aggregate["sha256"])
        checkpoints = _check_post_executor_rows(plan)
        terminals: dict[str, Any] = {}
        for date in DATES:
            row = plan["dates"][date]
            command = [
                python_executable, str(TERMINAL_AUDIT), "--checkpoint", str(checkpoints[date]),
                "--source-preflight", row["source_preflight"], "--writer-claim", row["writer_claim"],
                "--output", row["terminal_receipt"],
            ]
            for flag in ("--checkpoint", "--source-preflight", "--writer-claim", "--output"):
                _need(command.count(flag) == 1,
                      f"early-v2 terminal audit argument must occur exactly once: {flag}")
            result = _run(command)
            _record_subprocess(paths["subprocess_log"], f"source_terminal_audit_{date}", command, result)
            _need(result.returncode == 0,
                  f"early-v2 terminal audit failed for {date}: rc={result.returncode}: {result.stderr.strip()}")
            terminal_path, terminal, terminal_sha = read_immutable_json(
                row["terminal_receipt"], schema=EARLY_TERMINAL_SCHEMA, status=EARLY_TERMINAL_STATUS,
            )
            _need(terminal.get("outer_date") == date,
                  f"early-v2 terminal receipt date drift: {date}")
            terminals[date] = {
                "path": str(terminal_path), "sha256": terminal_sha,
                "checkpoint": str(checkpoints[date]), "checkpoint_sha256": sha256_file(checkpoints[date]),
            }
            _append_event(paths["event_log"], "source_terminalized", outer_date=date,
                          terminal_sha256=terminal_sha)
        terminal_body = {
            "schema": HANDOFF_SCHEMA, "status": HANDOFF_PASS_STATUS,
            "claim": {"path": str(paths["claim"]), "sha256": claim_sha},
            "launch_receipt": plan["launch_receipt"], "rt_aggregate": aggregate,
            "gpus": plan["gpus"], "source_terminals": terminals,
            "scope": {"target_opened": 0, "target_bytes_read": 0, "binder_called": False,
                      "evaluator_called": False, "cleanup_performed": False},
        }
        terminal_sha = _write_exclusive_json(paths["terminal"], terminal_body)
        _append_event(paths["event_log"], "handoff_completed", terminal_sha256=terminal_sha)
        return {"status": HANDOFF_PASS_STATUS, "terminal": str(paths["terminal"]),
                "terminal_sha256": terminal_sha, "source_terminals": terminals}
    except BaseException as error:
        try:
            _append_event(paths["event_log"], "handoff_failed", error_type=type(error).__name__, error=str(error))
            if not os.path.lexists(str(paths["terminal"])):
                _write_exclusive_json(paths["terminal"], {
                    "schema": HANDOFF_SCHEMA, "status": HANDOFF_FAIL_STATUS,
                    "claim": {"path": str(paths["claim"]), "sha256": claim_sha},
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "preservation": {"claims_deleted": False, "runs_deleted": False,
                                     "logs_deleted": False, "cleanup_performed": False},
                    "scope": {"target_opened": 0, "binder_called": False, "evaluator_called": False},
                })
        finally:
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--claim-root", required=True, type=Path)
    parser.add_argument("--terminal-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--rt-aggregate", required=True, type=Path)
    parser.add_argument("--rt-aggregate-pass-status", required=True)
    parser.add_argument("--rt-aggregate-sha256")
    parser.add_argument("--gpu0-device", default="0")
    parser.add_argument("--gpu1-device", default="1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--handoff-tmux", default=HANDOFF_TMUX)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=0.0,
                        help="0 waits indefinitely for the exact RT worker sessions to disappear")
    parser.add_argument("--rt-writer-marker", action="append", default=None)
    parser.add_argument("--execute-after-rt", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = build_plan(
        launch_receipt=args.launch_receipt, run_root=args.run_root, claim_root=args.claim_root,
        terminal_root=args.terminal_root, state_root=args.state_root,
        rt_aggregate=args.rt_aggregate, rt_aggregate_pass_status=args.rt_aggregate_pass_status,
        rt_aggregate_sha256=args.rt_aggregate_sha256,
        gpu0_device=args.gpu0_device, gpu1_device=args.gpu1_device,
        python_executable=args.python_executable, handoff_tmux=args.handoff_tmux,
    )
    if not args.execute_after_rt:
        print(json.dumps(plan, sort_keys=True))
        return
    markers = args.rt_writer_marker or ["rt_xls_v2", "falcon_rt"]
    print(json.dumps(execute(plan, poll_seconds=args.poll_seconds,
                             max_wait_seconds=args.max_wait_seconds,
                             rt_writer_markers=markers,
                             python_executable=args.python_executable), sort_keys=True))


if __name__ == "__main__":
    main()
