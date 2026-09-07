#!/usr/bin/env python3
"""Fail-closed post-training audit/package/smoke for H1 all-source H-C.

The watcher binds one fresh, fixed epoch-49 checkpoint to the immutable launch
receipt and launch marker, waits for the tmux process and GPU1 to release, then
performs a CPU-only package/smoke transaction.  It opens exactly the 13 public
held-in-calib and 14 public held-out-calib NWBs; no minival, query, formal
label, EvalAI, target optimizer, or backward path is present.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = ROOT / "pilot_artifacts/h1_carrierid_all_source_official_v1"
ASSET = ART / "H1_CARRIERID_ALL_SOURCE_ASSET_PREFLIGHT_v1.json"
LAUNCH = ART / "H1_CARRIERID_ALL_SOURCE_LAUNCH_RECEIPT_v1.json"
EXECUTION = ART / "H1_CARRIERID_ALL_SOURCE_GPU1_EXECUTION_LAUNCH_v1.json"
START = ART / "logs/h1_all_source_gpu1_start.txt"
TRAIN_LOG = ART / "logs/h1_all_source_gpu1.log"
RUNS = ROOT / "logs/h1_carrierid_all_source_official/runs"
# v1 artifacts remain immutable historical outputs.  The active watcher
# publishes into distinct v5 names so a rerun can never replace them.
PREFLIGHT_V1 = ART / "H1_CARRIERID_ALL_SOURCE_PACKAGE_PREFLIGHT_v1.json"
PAYLOAD_V1 = ART / "h1_carrierid_all_source_official_payload_v1.pkl"
SMOKE_V1 = ART / "H1_CARRIERID_ALL_SOURCE_RUNTIME_SMOKE_v1.json"
AUDIT_V1 = ART / "H1_CARRIERID_ALL_SOURCE_TERMINAL_AUDIT_v1.json"
PREFLIGHT = ART / "H1_CARRIERID_ALL_SOURCE_PACKAGE_PREFLIGHT_v5.json"
PAYLOAD = ART / "h1_carrierid_all_source_official_payload_v5.pkl"
SMOKE = ART / "H1_CARRIERID_ALL_SOURCE_RUNTIME_SMOKE_v5.json"
AUDIT = ART / "H1_CARRIERID_ALL_SOURCE_TERMINAL_AUDIT_v5.json"
STABILITY = ART / "recovery_package_v1/H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_M3_STABILITY_AUDIT_v5_r4.json"
STABILITY_SHA256 = "4b6adb58460a48cb791a56ed92efd1eb1f2ab961c5f43dd366777267da3a7277"
LOCK = ART / ".h1_all_source_posttrain_watch.lock"
TMUX_SESSION = "h1_all_source_gpu1"
GUARD_SESSION = "h1_all_source_gpu1_watch"
POLL_SECONDS = 30.0
EXPECTED_BEHAVIOR_DIM = 7
EXPECTED_PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"

EXPECTED_HELDIN_SESSIONS = (
    "ses-19250101T111740", "ses-19250101T112404",
    "ses-19250108T110520", "ses-19250108T111022", "ses-19250108T111455",
    "ses-19250113T120811", "ses-19250113T121303",
    "ses-19250115T110633", "ses-19250115T111328",
    "ses-19250119T113543", "ses-19250119T114045",
    "ses-19250120T115044", "ses-19250120T115537",
)
EXPECTED_HELDOUT_SESSIONS = (
    "ses-19250126T113454", "ses-19250126T114029",
    "ses-19250127T120333", "ses-19250127T120826",
    "ses-19250129T112555", "ses-19250129T113059",
    "ses-19250202T113958", "ses-19250202T114452",
    "ses-19250203T113515", "ses-19250203T114018",
    "ses-19250206T112219", "ses-19250206T112712",
    "ses-19250209T111826", "ses-19250209T112327",
)
FINAL_ARTIFACTS = (PREFLIGHT, PAYLOAD, SMOKE, AUDIT)


@dataclass(frozen=True)
class ExecutionBinding:
    nonce: str
    start_time_ns: int
    execution_sha256: str
    preexisting_run_dirs: tuple[str, ...]


def _stamp(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


def _expect(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {value!r}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is not a mapping")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON receipt: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON receipt is not a mapping: {path}")
    return value


def _expected_command() -> list[str]:
    return [
        EXPECTED_PYTHON,
        str(ROOT / "src/train.py"),
        "experiment=h1_carrierid_all_source_official",
        f"official_candidate.asset_manifest_path={ASSET}",
        "ckpt_path=null",
        "test=false",
    ]


def _expected_command_text() -> str:
    return "CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 " + " ".join(_expected_command())


def _parse_start_marker(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"start marker must be a regular file: {path}")
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise RuntimeError(f"start marker is not immutable mode 0444: {path}")
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            raise RuntimeError(f"malformed START marker line: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in rows:
            raise RuntimeError(f"duplicate/empty START marker key: {key!r}")
        rows[key] = value
    expected_keys = {
        "nonce", "start_time_ns", "start_time", "workdir", "tmux_session",
        "guard_session", "command", "stdout_stderr_log", "execution_receipt",
    }
    if set(rows) != expected_keys:
        raise RuntimeError(
            f"START marker keys drifted: observed={sorted(rows)} expected={sorted(expected_keys)}"
        )
    return rows


def _validate_execution_binding() -> ExecutionBinding:
    """Bind immutable execution receipt, START nonce and pre-launch run snapshot."""

    if not EXECUTION.is_file() or EXECUTION.is_symlink():
        raise FileNotFoundError(f"immutable execution receipt missing: {EXECUTION}")
    if stat.S_IMODE(EXECUTION.stat().st_mode) != 0o444:
        raise RuntimeError(f"execution receipt is not immutable mode 0444: {EXECUTION}")
    receipt = _load_json(EXECUTION)
    expected_receipt_keys = {
        "schema", "status", "nonce", "start_time_ns", "start_time",
        "tmux_session", "guard_session", "workdir", "expected_command",
        "expected_command_text", "asset_manifest", "prepared_launch_receipt",
        "prepared_candidate_config_sha256", "current_code_sha256",
        "launch_guard_sha256", "preexisting_all_source_run_dirs",
        "expected_new_run_root", "h_c0_gpu1_terminal_checkpoints", "scope",
    }
    if set(receipt) != expected_receipt_keys:
        raise RuntimeError(
            f"execution receipt keys drifted: observed={sorted(receipt)} "
            f"expected={sorted(expected_receipt_keys)}"
        )
    _expect(receipt.get("schema"), "h1_carrierid_all_public_source_gpu1_execution_launch_v1",
            "execution.schema")
    _expect(receipt.get("status"), "PASS_ALL_SOURCE_GPU1_LAUNCH_RESERVED", "execution.status")
    nonce = receipt.get("nonce")
    if not isinstance(nonce, str) or len(nonce) != 32 or any(c not in "0123456789abcdef" for c in nonce):
        raise RuntimeError(f"execution nonce is malformed: {nonce!r}")
    start_time_ns = receipt.get("start_time_ns")
    if not isinstance(start_time_ns, int) or start_time_ns <= 0:
        raise RuntimeError(f"execution start_time_ns is malformed: {start_time_ns!r}")
    _expect(receipt.get("tmux_session"), TMUX_SESSION, "execution.tmux_session")
    _expect(receipt.get("guard_session"), GUARD_SESSION, "execution.guard_session")
    _expect(receipt.get("workdir"), str(ROOT), "execution.workdir")
    _expect(receipt.get("expected_command"), _expected_command(), "execution.expected_command")
    _expect(receipt.get("expected_command_text"), _expected_command_text(),
            "execution.expected_command_text")
    _expect(receipt.get("expected_new_run_root"), str(RUNS.resolve()),
            "execution.expected_new_run_root")
    asset = _mapping(receipt.get("asset_manifest"), "execution.asset_manifest")
    _expect(asset.get("path"), str(ASSET.resolve()), "execution.asset path")
    _expect(asset.get("sha256"), _sha256_file(ASSET), "execution.asset SHA")
    prepared = _mapping(receipt.get("prepared_launch_receipt"),
                        "execution.prepared_launch_receipt")
    _expect(prepared.get("path"), str(LAUNCH.resolve()), "execution.prepared path")
    _expect(prepared.get("sha256"), _sha256_file(LAUNCH), "execution.prepared SHA")
    prepared_body = _load_json(LAUNCH)
    _expect(receipt.get("prepared_candidate_config_sha256"),
            _mapping(prepared_body.get("candidate"), "prepared.candidate").get("config_sha256"),
            "execution.prepared candidate config SHA")
    _expect(receipt.get("current_code_sha256"), prepared_body.get("code_sha256"),
            "execution/prepared code SHA")
    guard_path = ROOT / "scripts/h1_carrierid_all_source_official_gpu1_launch_guard.py"
    _expect(receipt.get("launch_guard_sha256"), _sha256_file(guard_path),
            "execution.launch guard SHA")
    scope = _mapping(receipt.get("scope"), "execution.scope")
    for key, expected in {
        "formal_test_labels_opened": 0, "target_optimizer_steps": 0,
        "target_backward_steps": 0, "evalai_submission_authorized": False,
    }.items():
        _expect(scope.get(key), expected, f"execution.scope.{key}")
    hc0 = receipt.get("h_c0_gpu1_terminal_checkpoints")
    if not isinstance(hc0, list) or len(hc0) != 2 or any(not isinstance(row, Mapping) for row in hc0):
        raise RuntimeError("execution H-C0 terminal-checkpoint evidence is malformed")
    _expect({row.get("date") for row in hc0}, {"19250113", "19250119"},
            "execution H-C0 dates")
    for row in hc0:
        for key in ("checkpoint_sha256", "config_sha256", "pair_sha256",
                    "phase2_source_binding_sha256", "phase1_source_manifest_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError(f"execution H-C0 {row.get('date')} {key} is malformed")

    preexisting = receipt.get("preexisting_all_source_run_dirs")
    if not isinstance(preexisting, list) or any(not isinstance(name, str) for name in preexisting):
        raise RuntimeError("execution preexisting run-dir snapshot is malformed")
    if preexisting != sorted(preexisting) or len(preexisting) != len(set(preexisting)):
        raise RuntimeError("execution preexisting run-dir snapshot is unsorted or duplicated")
    for name in preexisting:
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1 or name in {"", ".", ".."}:
            raise RuntimeError(f"unsafe preexisting run-dir entry: {name!r}")

    start = _parse_start_marker(START)
    _expect(start.get("nonce"), nonce, "START/execution nonce")
    _expect(start.get("start_time_ns"), str(start_time_ns), "START/execution start_time_ns")
    _expect(start.get("start_time"), receipt.get("start_time"), "START/execution start_time")
    _expect(start.get("workdir"), str(ROOT), "START.workdir")
    _expect(start.get("tmux_session"), TMUX_SESSION, "START.tmux_session")
    _expect(start.get("guard_session"), GUARD_SESSION, "START.guard_session")
    _expect(start.get("command"), _expected_command_text(), "START.command")
    _expect(start.get("stdout_stderr_log"), str(TRAIN_LOG), "START.stdout_stderr_log")
    _expect(start.get("execution_receipt"), str(EXECUTION), "START.execution_receipt")
    if EXECUTION.stat().st_mtime_ns < start_time_ns:
        raise RuntimeError("execution receipt mtime predates recorded reservation time")
    if START.stat().st_mtime_ns < EXECUTION.stat().st_mtime_ns:
        raise RuntimeError("START marker predates execution receipt publication")
    return ExecutionBinding(
        nonce=nonce,
        start_time_ns=start_time_ns,
        execution_sha256=_sha256_file(EXECUTION),
        preexisting_run_dirs=tuple(preexisting),
    )


@contextmanager
def _exclusive_watch_lock() -> Iterator[None]:
    ART.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another H1 all-source post-training watcher is active") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _tmux_state() -> str:
    """Return absent/running/exited_zero and reject every command/status error."""

    try:
        # ``=NAME`` asks tmux for an exact target-session name.  Without the
        # equals prefix, the launch guard's ``h1_all_source_gpu1_watch`` name
        # would be accepted as a prefix for the target session.
        exists = subprocess.run(
            ["tmux", "has-session", "-t", f"={TMUX_SESSION}"],
            text=True, capture_output=True, check=False,
        )
    except OSError as exc:
        raise RuntimeError("tmux status command failed") from exc
    if exists.returncode == 1:
        return "absent"
    if exists.returncode != 0:
        raise RuntimeError(
            f"tmux has-session failed rc={exists.returncode}: {exists.stderr.strip()}"
        )
    panes = subprocess.run(
        ["tmux", "list-panes", "-t", f"={TMUX_SESSION}",
         "-F", "#{pane_dead}|#{pane_dead_status}"],
        text=True, capture_output=True, check=False,
    )
    if panes.returncode != 0:
        raise RuntimeError(
            f"tmux list-panes failed rc={panes.returncode}: {panes.stderr.strip()}"
        )
    rows = [line.strip().split("|", 1) for line in panes.stdout.splitlines() if line.strip()]
    if not rows or any(len(row) != 2 or row[0] not in {"0", "1"} for row in rows):
        raise RuntimeError(f"malformed tmux pane state: {panes.stdout!r}")
    if any(dead == "0" for dead, _status in rows):
        return "running"
    try:
        statuses = [int(status) for _dead, status in rows]
    except ValueError as exc:
        raise RuntimeError(f"malformed tmux pane exit status: {rows}") from exc
    if any(status != 0 for status in statuses):
        raise RuntimeError(f"all-source tmux pane exited nonzero: {statuses}")
    return "exited_zero"


def _gpu1_busy() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", "1", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
    except OSError as exc:
        raise RuntimeError("nvidia-smi GPU-release query failed") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi GPU-release query failed rc={result.returncode}: {result.stderr.strip()}"
        )
    pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not pid.isdigit() for pid in pids):
        raise RuntimeError(f"malformed nvidia-smi compute PID output: {pids}")
    return bool(pids)


def _new_run_dirs(binding: ExecutionBinding) -> tuple[Path, ...]:
    if not RUNS.exists():
        return ()
    if not RUNS.is_dir() or RUNS.is_symlink():
        raise RuntimeError(f"all-source run root is not a regular directory: {RUNS}")
    preexisting = set(binding.preexisting_run_dirs)
    new: list[Path] = []
    for candidate in sorted(RUNS.iterdir(), key=lambda path: path.name):
        if candidate.is_symlink():
            raise RuntimeError(f"all-source run-dir symlink is forbidden: {candidate}")
        if not candidate.is_dir() or candidate.name in preexisting:
            continue
        if candidate.stat().st_mtime_ns < binding.start_time_ns:
            raise RuntimeError(f"new run dir predates execution reservation: {candidate}")
        new.append(candidate.resolve())
    if len(new) > 1:
        raise RuntimeError(f"ambiguous post-reservation run dirs: {[path.name for path in new]}")
    return tuple(new)


def _fresh_terminal_candidates(binding: ExecutionBinding) -> tuple[Path, ...]:
    runs = _new_run_dirs(binding)
    if not runs:
        return ()
    run = runs[0]
    candidate = run / "checkpoints/fixed_epoch50/epoch_049.ckpt"
    if candidate.is_symlink():
        raise RuntimeError(f"terminal checkpoint symlink is forbidden: {candidate}")
    if not candidate.is_file():
        return ()
    if candidate.stat().st_mtime_ns < binding.start_time_ns:
        raise RuntimeError(f"terminal checkpoint predates execution reservation: {candidate}")
    return (candidate.resolve(),)


def _audit_training_log(start_time_ns: int, *, require_exists: bool) -> None:
    if not TRAIN_LOG.is_file():
        if require_exists:
            raise FileNotFoundError(f"all-source training log missing: {TRAIN_LOG}")
        return
    if TRAIN_LOG.is_symlink() or TRAIN_LOG.stat().st_mtime_ns < start_time_ns:
        raise RuntimeError("all-source training log is a symlink or predates start marker")
    text = TRAIN_LOG.read_text(encoding="utf-8", errors="replace").lower()
    fatal = (
        "traceback (most recent call last)", "cuda error", "out of memory",
        "segmentation fault", "process killed",
    )
    matches = [token for token in fatal if token in text]
    if matches:
        raise RuntimeError(f"all-source training log contains terminal error marker(s): {matches}")


def _wait_for_training_completion(*, poll_seconds: float = POLL_SECONDS) -> ExecutionBinding:
    _stamp("waiting for nonce-bound all-source execution receipt and START marker")
    while not EXECUTION.is_file() or not START.is_file():
        time.sleep(poll_seconds)
    binding = _validate_execution_binding()

    observed = False
    while True:
        state = _tmux_state()
        if state == "running":
            if not observed:
                _stamp("all-source tmux process observed; waiting for process exit")
            observed = True
            time.sleep(poll_seconds)
            continue
        if state == "exited_zero" or observed:
            break
        _audit_training_log(binding.start_time_ns, require_exists=False)
        # Allows a watcher restart after tmux auto-destroyed, but only when a
        # fresh terminal checkpoint exists. _checkpoint later requires it to
        # be unique, so an old or ambiguous run can never be selected.
        if _fresh_terminal_candidates(binding):
            _stamp("tmux absent; fresh terminal e49 observed")
            break
        time.sleep(poll_seconds)

    _audit_training_log(binding.start_time_ns, require_exists=True)
    idle_observations = 0
    while idle_observations < 2:
        if _gpu1_busy():
            idle_observations = 0
            time.sleep(poll_seconds)
            continue
        idle_observations += 1
        if idle_observations < 2:
            time.sleep(min(poll_seconds, 5.0))
    return binding


def _session_from_path(path: Path) -> str:
    marker = "_ses-"
    if marker not in path.stem:
        raise RuntimeError(f"cannot parse H1 calibration session: {path}")
    return path.stem[path.stem.index(marker) + 1 :]


def _allowlist(
    data_root: Path | None = None,
    heldin_sessions: Sequence[str] = EXPECTED_HELDIN_SESSIONS,
    heldout_sessions: Sequence[str] = EXPECTED_HELDOUT_SESSIONS,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    """Return the exact ordered 13+14 public calibration NWBs."""

    root = (data_root or (ROOT / "data/000954")).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"H1 data root missing: {root}")
    groups = (
        ("sub-HumanPitt-held-in-calib", tuple(heldin_sessions)),
        ("sub-HumanPitt-held-out-calib", tuple(heldout_sessions)),
    )
    outputs: list[tuple[Path, ...]] = []
    forbidden = ("minival", "query", "test_ecephys", "formal", "private")
    for dirname, expected in groups:
        directory = (root / dirname).resolve()
        try:
            directory.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"calibration directory escapes H1 root: {directory}") from exc
        if not directory.is_dir():
            raise FileNotFoundError(f"H1 calibration directory missing: {directory}")
        observed: dict[str, Path] = {}
        for candidate in sorted(directory.glob("*.nwb")):
            if candidate.is_symlink() or not candidate.is_file():
                raise RuntimeError(f"calibration allowlist rejects symlink/non-file: {candidate}")
            resolved = candidate.resolve()
            try:
                resolved.relative_to(directory)
            except ValueError as exc:
                raise RuntimeError(f"calibration file escapes role directory: {candidate}") from exc
            session = _session_from_path(candidate)
            if candidate.name != f"{dirname}_{session}.nwb":
                raise RuntimeError(f"noncanonical H1 calibration filename: {candidate.name}")
            if session in observed:
                raise RuntimeError(f"duplicate H1 calibration session: {session}")
            if any(token in resolved.as_posix().lower() for token in forbidden):
                raise RuntimeError(f"forbidden endpoint path in calibration allowlist: {resolved}")
            observed[session] = resolved
        if set(observed) != set(expected) or len(observed) != len(expected):
            raise RuntimeError(
                f"H1 {dirname} exact-session allowlist drift: "
                f"observed={sorted(observed)} expected={sorted(expected)}"
            )
        outputs.append(tuple(observed[session] for session in expected))
    heldin, heldout = outputs
    paths = heldin + heldout
    if len(paths) != 27 or len(set(paths)) != 27:
        raise RuntimeError("H1 calibration allowlist must contain 27 unique files")
    return paths, heldin, heldout


def _checkpoint(binding: ExecutionBinding) -> Path:
    candidates = _fresh_terminal_candidates(binding)
    if not candidates:
        raise FileNotFoundError(f"no current-run terminal e49 checkpoint under {RUNS}")
    if len(candidates) != 1:
        raise RuntimeError(f"ambiguous current-run e49 checkpoints: {[str(path) for path in candidates]}")
    checkpoint = candidates[0]
    if checkpoint.stat().st_size <= 0:
        raise RuntimeError(f"terminal checkpoint is empty: {checkpoint}")
    run_dir = checkpoint.parent.parent.parent
    _expect(_new_run_dirs(binding), (run_dir.resolve(),), "execution-bound fresh run dir")
    if run_dir.parent.resolve() != RUNS.resolve():
        raise RuntimeError(f"terminal run is not a direct child of {RUNS}: {checkpoint}")
    if not (run_dir / ".hydra/config.yaml").is_file():
        raise FileNotFoundError(f"terminal run lacks resolved Hydra config: {run_dir}")
    return checkpoint


def _config_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _validate_launch_receipt(asset_manifest_sha256: str) -> dict[str, Any]:
    if stat.S_IMODE(LAUNCH.stat().st_mode) != 0o444:
        raise RuntimeError(f"launch receipt is not immutable mode 0444: {LAUNCH}")
    receipt = _load_json(LAUNCH)
    _expect(receipt.get("schema"), "h1_carrierid_all_public_source_official_launch_receipt_v1",
            "launch.schema")
    _expect(receipt.get("status"), "PASS_ALL_SOURCE_HC_PREPARED_NOT_LAUNCHED", "launch.status")
    _expect(receipt.get("planned_command"), _expected_command(), "launch.planned_command")
    candidate = _mapping(receipt.get("candidate"), "launch.candidate")
    _expect(candidate.get("asset_manifest_path"), str(ASSET.resolve()), "launch.asset path")
    _expect(candidate.get("asset_manifest_sha256"), asset_manifest_sha256, "launch.asset SHA")
    _expect(candidate.get("source_sessions"), list(EXPECTED_HELDIN_SESSIONS), "launch.sessions")
    for key, expected in {"fresh_seed": 42, "epochs": 50, "checkpoint_epoch_zero_based": 49}.items():
        _expect(candidate.get(key), expected, f"launch.{key}")
    scope = _mapping(receipt.get("scope"), "launch.scope")
    for key, expected in {
        "public_held_in_calibration_recordings": 13,
        "minival_recordings_opened": 0, "formal_test_labels_opened": 0,
        "target_optimizer_steps": 0, "target_backward_steps": 0,
        "trainer_constructed_or_launched": False,
        "cuda_constructed_or_launched": False,
        "evalai_submission_authorized": False,
    }.items():
        _expect(scope.get(key), expected, f"launch.scope.{key}")
    code = _mapping(receipt.get("code_sha256"), "launch.code_sha256")
    code_paths = {
        "launcher": ROOT / "scripts/h1_carrierid_all_source_official_launcher.py",
        "data": ROOT / "src/data/h1_carrierid_all_source_official.py",
        "model": ROOT / "src/models/h1_carrierid_all_source_official_module.py",
        "experiment": ROOT / "configs/experiment/h1_carrierid_all_source_official.yaml",
        "data_config": ROOT / "configs/data/falcon_h1_carrierid_all_source_official.yaml",
        "model_config": ROOT / "configs/model/falcon_h1_carrierid_all_source_official.yaml",
    }
    for key, path in code_paths.items():
        _expect(code.get(key), _sha256_file(path), f"launch code SHA {key}")
    return receipt


def _validate_resolved_config(resolved: Mapping[str, Any]) -> tuple[Mapping[str, Any], Path]:
    for key, expected in {
        "protocol_id": "h1_carrierid_all_public_source_official_candidate_v1",
        "task_name": "h1_carrierid_all_source_official", "seed": 42,
        "train": True, "test": False, "ckpt_path": None, "logger": False,
    }.items():
        _expect(resolved.get(key), expected, f"config.{key}")
    official = _mapping(resolved.get("official_candidate"), "config.official_candidate")
    for key, expected in {
        "training_scope": "all_13_public_held_in_calibration_recordings",
        "calibration_support_trials": 4, "formal_test_labels_opened": 0,
        "target_optimizer_steps": 0, "target_backward_steps": 0,
        "evalai_submission_authorized": False,
    }.items():
        _expect(official.get(key), expected, f"config.official_candidate.{key}")
    _expect(_config_path(official.get("asset_manifest_path")), ASSET.resolve(),
            "config.official_candidate.asset_manifest_path")

    data = _mapping(resolved.get("data"), "config.data")
    for key, expected in {
        "_target_": "src.data.h1_carrierid_all_source_official.H1CarrierIdAllSourceDataModule",
        "task": "h1", "batch_size": 32, "window_size": 700,
        "calibration_n_trials": 4, "max_trial_length": 1024, "seed": 42,
        "fixed_epochs": 50, "num_workers": 0, "pin_memory": False,
    }.items():
        _expect(data.get(key), expected, f"config.data.{key}")
    _expect(_config_path(data.get("asset_manifest_path")), ASSET.resolve(),
            "config.data.asset_manifest_path")
    data_dir = _config_path(data.get("data_dir"))
    _expect(data_dir, (ROOT / "data/000954").resolve(), "config.data.data_dir")

    model = _mapping(resolved.get("model"), "config.model")
    for key, expected in {
        "_target_": "src.models.h1_carrierid_all_source_official_module.H1CarrierIdAllSourceLitModule",
        "task": "h1", "fixed_seed": 42, "decode_last_timestep_only": True,
        "predict_scaled_behavior": True, "behavior_scaling_factor": 20.0,
        "clean_teacher": True,
    }.items():
        _expect(model.get(key), expected, f"config.model.{key}")
    net = _mapping(model.get("net"), "config.model.net")
    for key, expected in {
        "_target_": "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
        "carrier_dim": 4, "carrier_trial_length": 1024, "zero_carrier": False,
        "model_dim": 1024, "num_covariates": EXPECTED_BEHAVIOR_DIM, "window_size": 700,
    }.items():
        _expect(net.get(key), expected, f"config.model.net.{key}")
    trainer = _mapping(resolved.get("trainer"), "config.trainer")
    for key, expected in {
        "accelerator": "gpu", "devices": 1, "min_epochs": 50, "max_epochs": 50,
        "precision": "32-true", "enable_checkpointing": True,
        "limit_val_batches": 0, "num_sanity_val_steps": 0,
    }.items():
        _expect(trainer.get(key), expected, f"config.trainer.{key}")
    fixed = _mapping(
        _mapping(resolved.get("callbacks"), "config.callbacks").get("fixed_epoch50"),
        "config.callbacks.fixed_epoch50",
    )
    for key, expected in {
        "monitor": None, "save_top_k": -1, "save_last": False,
        "every_n_epochs": 50, "auto_insert_metric_name": False,
    }.items():
        _expect(fixed.get(key), expected, f"config.callbacks.fixed_epoch50.{key}")
    return data, data_dir


def _force_cpu_only() -> None:
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported before CPU-only post-training boundary")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _validate_smoke_output(output: np.ndarray, *, behavior_dim: int = EXPECTED_BEHAVIOR_DIM) -> None:
    value = np.asarray(output)
    if value.shape != (1, behavior_dim) or not bool(np.isfinite(value).all()):
        raise RuntimeError(f"H1 runtime output must be finite [1,{behavior_dim}], got {value.shape}")


def _validate_v5_payload_receipts(
    receipts: Any, paths: Sequence[Path], heldin: Sequence[Path], heldout: Sequence[Path],
) -> dict[str, list[int]]:
    """Require one exact, non-padded TrialNum receipt for every 13+14 record."""

    if not isinstance(receipts, list) or len(receipts) != 27:
        raise RuntimeError("v5 payload calibration receipts must contain exactly 27 rows")
    expected_heldin = {path.resolve() for path in heldin}
    shapes: dict[str, list[int]] = {}
    for index, (row, expected_path) in enumerate(zip(receipts, paths)):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"v5 payload receipt {index} is malformed")
        if Path(str(row.get("path", ""))).resolve() != expected_path.resolve():
            raise RuntimeError(f"v5 payload receipt {index} path drift")
        support_m = row.get("support_m")
        expected_m = 4 if expected_path.resolve() in expected_heldin else 3
        if type(support_m) is not int or support_m != expected_m:
            raise RuntimeError(f"v5 payload receipt {index} support_m must be {expected_m}")
        values = row.get("support_trial_numbers")
        if not isinstance(values, list) or len(values) != expected_m:
            raise RuntimeError(f"v5 payload receipt {index} TrialNum length drift")
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"v5 payload receipt {index} TrialNum values malformed") from exc
        if len(set(numeric)) != expected_m:
            raise RuntimeError(f"v5 payload receipt {index} contains padding/duplication")
        if row.get("identity_shape", [expected_m, 1024, 176]) != [expected_m, 1024, 176]:
            raise RuntimeError(f"v5 payload receipt {index} identity shape drift")
        if row.get("carrier_shape", [176, 4]) != [176, 4]:
            raise RuntimeError(f"v5 payload receipt {index} carrier shape drift")
        if row.get("padding_or_duplication") not in (None, False):
            raise RuntimeError(f"v5 payload receipt {index} records padding/duplication")
        tag = str(row.get("dataset_tag", ""))
        if not tag:
            raise RuntimeError(f"v5 payload receipt {index} dataset tag missing")
        shapes[tag] = [expected_m, 1024, 176]
    return shapes


def _publish_bundle(staged_to_final: Sequence[tuple[Path, Path]]) -> None:
    """Publish an already validated bundle without replacement."""

    for staged, final in staged_to_final:
        if not staged.is_file():
            raise FileNotFoundError(f"staged artifact missing: {staged}")
        if final.exists():
            raise FileExistsError(f"refusing to overwrite post-training artifact: {final}")
    created: list[Path] = []
    try:
        for staged, final in staged_to_final:
            os.link(staged, final)
            created.append(final)
    except Exception:
        for final in reversed(created):
            try:
                final.unlink()
            except FileNotFoundError:
                pass
        raise


def _main_locked() -> None:
    existing = [str(path) for path in FINAL_ARTIFACTS if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")
    if not STABILITY.is_file() or STABILITY.is_symlink() or _sha256_file(STABILITY) != STABILITY_SHA256:
        raise RuntimeError("canonical v5_r4 stability receipt is missing or SHA-bound incorrectly")
    execution = _wait_for_training_completion()
    checkpoint = _checkpoint(execution)
    _stamp(f"current-run terminal checkpoint: {checkpoint}")

    # Hide CUDA before importing torch or any model/runtime module.  The
    # inherited SPINT runtime otherwise chooses CUDA whenever it is available.
    _force_cpu_only()
    import torch
    from omegaconf import OmegaConf
    from falcon_challenge.config import FalconConfig, FalconTask
    from src.data.h1_baseline_datamodule import (
        H1_BASELINE_HELDIN_SESSIONS, H1_BASELINE_HELDOUT_SESSIONS,
    )
    from src.data.h1_carrierid_all_source_official import (
        H1CarrierIdAllSourceDataModule, H1_HELDIN_SESSIONS,
        assert_deployment_calibration_path, load_all_source_assets,
    )
    from src.h1_m4_cce_contract import canonical_sha256, sha256_file, write_immutable_json
    from src.models.h1_carrierid_all_source_official_module import ALL_SOURCE_CHECKPOINT_SCHEMA
    from scripts.h1_carrierid_all_source_official_package import (
        export_payload, prepare_package_preflight, validate_checkpoint_metadata,
    )
    from third_party.falcon_challenge.h1_carrierid_all_source_decoder import (
        H1CarrierIdAllSourceDecoder, H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5,
        validate_carrier_payload,
    )
    from third_party.falcon_challenge.spint_decoder import CPU_Unpickler

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only watcher unexpectedly sees CUDA")
    _expect(tuple(H1_HELDIN_SESSIONS), EXPECTED_HELDIN_SESSIONS, "all-source sessions")
    _expect(tuple(H1_BASELINE_HELDIN_SESSIONS), EXPECTED_HELDIN_SESSIONS, "baseline held-in")
    _expect(tuple(H1_BASELINE_HELDOUT_SESSIONS), EXPECTED_HELDOUT_SESSIONS, "baseline held-out")
    paths, heldin, heldout = _allowlist()
    for path in paths:
        _expect(assert_deployment_calibration_path(path), path, f"deployment path {path}")

    assets = load_all_source_assets(ASSET)
    launch = _validate_launch_receipt(assets.manifest_sha256)
    checkpoint_sha = sha256_file(checkpoint)
    body = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _expect(sha256_file(checkpoint), checkpoint_sha, "checkpoint SHA after load")
    if not isinstance(body, dict):
        raise RuntimeError("terminal checkpoint is not a mapping")
    _expect(body.get("epoch"), 49, "checkpoint top-level epoch")
    metadata = validate_checkpoint_metadata(body, asset_manifest_sha256=assets.manifest_sha256)
    for key, expected in {
        "schema": ALL_SOURCE_CHECKPOINT_SCHEMA, "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_no_formal_selection",
        "checkpoint_warm_start": False, "target_optimizer_steps": 0,
        "target_backward_steps": 0, "formal_test_labels_opened": 0,
    }.items():
        _expect(metadata.get(key), expected, f"checkpoint metadata {key}")
    initial_sha = metadata.get("initial_state_sha256")
    if not isinstance(initial_sha, str) or len(initial_sha) != 64:
        raise RuntimeError("checkpoint initial-state SHA is missing or malformed")

    run_dir = checkpoint.parent.parent.parent.resolve()
    config_path = run_dir / ".hydra/config.yaml"
    config_sha = sha256_file(config_path)
    _expect(metadata.get("config_sha256"), config_sha, "checkpoint/config SHA")
    resolved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    data_cfg, data_dir = _validate_resolved_config(_mapping(resolved, "resolved config"))
    module = H1CarrierIdAllSourceDataModule(
        task=data_cfg["task"], data_dir=str(data_dir), asset_manifest_path=str(ASSET.resolve()),
        batch_size=data_cfg["batch_size"], window_size=data_cfg["window_size"],
        calibration_n_trials=data_cfg["calibration_n_trials"],
        max_trial_length=data_cfg["max_trial_length"], seed=data_cfg["seed"],
        fixed_epochs=data_cfg["fixed_epochs"], num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )
    module.setup("fit")
    binding = module.all_source_manifest()
    binding_sha = canonical_sha256(binding)
    _expect(metadata.get("source_binding_sha256"), binding_sha, "checkpoint/source binding SHA")
    state = body.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("terminal checkpoint state_dict missing")
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"state_dict entry is not tensor: {name}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"nonfinite terminal state tensor: {name}")
    tensor_count = len(state)
    _stamp(f"metadata/source/config/state audit PASS ({tensor_count} tensors)")

    with tempfile.TemporaryDirectory(prefix=".posttrain_stage_", dir=ART) as temporary:
        stage = Path(temporary)
        staged_preflight = stage / PREFLIGHT.name
        staged_payload = stage / PAYLOAD.name
        staged_smoke = stage / SMOKE.name
        staged_audit = stage / AUDIT.name
        preflight = prepare_package_preflight(
            checkpoint=checkpoint, asset_manifest=ASSET,
            calibration_files=paths, output=staged_preflight,
        )
        _expect(preflight.get("schema"), "h1_carrierid_all_public_source_package_preflight_v1",
                "package preflight schema")
        _expect(preflight.get("status"), "PASS_PACKAGE_PREPARED_NOT_EXPORTED_NOT_SUBMITTED",
                "package preflight status")
        _expect(preflight["checkpoint"]["sha256"], checkpoint_sha,
                "package preflight checkpoint SHA")
        export = export_payload(
            checkpoint=checkpoint, asset_manifest=ASSET,
            calibration_files=paths, output=staged_payload,
        )
        _expect(sha256_file(checkpoint), checkpoint_sha, "checkpoint SHA after export")
        payload_sha = sha256_file(staged_payload)
        with staged_payload.open("rb") as handle:
            packaged = CPU_Unpickler(handle).load()
        expected_payload_keys = {
            "decoder", "task", "window_size", "behavior_scaling_factor",
            "interpolate_trials", "interpolate_trials_kind", "calib_trial_features",
            "calib_carriers", "smooth_calibration", "carrier_payload_schema",
            "carrier_asset_manifest_sha256", "carrier_transform_sha256",
            "carrier_normalizer_sha256", "source_checkpoint_metadata",
            "calibration_receipts", "deployment_contract",
        }
        if not isinstance(packaged, dict) or set(packaged) != expected_payload_keys:
            raise RuntimeError("payload top-level contract drift")
        task_config = FalconConfig(task=FalconTask.h1)
        carriers = validate_carrier_payload(packaged, expected_task=FalconTask.h1)
        expected_tags = {task_config.hash_dataset(path.stem) for path in paths}
        _expect(set(carriers), expected_tags, "payload dataset tags")
        _expect(packaged.get("carrier_payload_schema"), H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5,
                "payload schema")
        _expect(packaged.get("carrier_asset_manifest_sha256"), assets.manifest_sha256,
                "payload asset SHA")
        _expect(packaged.get("carrier_transform_sha256"), assets.plan.transform_sha256,
                "payload transform SHA")
        _expect(packaged.get("carrier_normalizer_sha256"), assets.normalizer.normalizer_sha256,
                "payload normalizer SHA")
        _expect(packaged.get("source_checkpoint_metadata"), dict(metadata),
                "payload checkpoint metadata")
        identity_shapes: dict[str, list[int]] = {}
        for tag, value in packaged["calib_trial_features"].items():
            identity = np.asarray(value)
            if identity.ndim != 3 or identity.shape[0] not in (3, 4) \
                    or identity.shape[1:] != (1024, 176) or not bool(np.isfinite(identity).all()):
                raise RuntimeError(f"payload identity shape/finite mismatch: {tag} {identity.shape}")
            identity_shapes[tag] = list(identity.shape)
        receipts = export.get("datasets")
        receipt_shapes = _validate_v5_payload_receipts(receipts, paths, heldin, heldout)
        if set(receipt_shapes) != set(packaged["calib_trial_features"]):
            raise RuntimeError("v5 payload receipt/tag roster drift")
        for row, path in zip(receipts, paths):
            _expect(row.get("input_sha256"), sha256_file(path), f"payload input SHA {path}")
        for tag, shape in receipt_shapes.items():
            _expect(identity_shapes.get(tag), shape, f"v5 payload identity shape {tag}")

        decoder = H1CarrierIdAllSourceDecoder(
            task_config=task_config, model_path=str(staged_payload), batch_size=1
        )
        smoke_rows = []
        for split, split_paths in (("heldin", heldin), ("heldout", heldout)):
            for path in split_paths:
                decoder.reset([path])
                _expect(decoder.device.type, "cpu", "runtime smoke device")
                output = decoder.predict(np.zeros((1, 176), dtype=np.float32))
                _validate_smoke_output(output)
                smoke_rows.append({
                    "split": split, "path": str(path), "dataset_tag": task_config.hash_dataset(path.stem),
                    "output_shape": list(output.shape), "finite": True,
                })
        smoke_body = {
            "schema": "h1_carrierid_all_public_source_recovery_package_smoke_v5",
            "status": "PASS_RECOVERY_CPU_SMOKE_NO_QUERY_OPENED",
            "payload": {"path": str(PAYLOAD), "sha256": payload_sha},
            "counts": {"heldin": 13, "heldout": 14, "total": 27},
            "scope": {
                "query_recordings_opened": 0,
                "formal_test_labels_opened": 0, "target_optimizer_steps": 0,
                "target_backward_steps": 0, "cuda_used": False,
                "evalai_accessed_or_submitted": False,
            },
            "rows": smoke_rows,
            "submission_authorized": False,
        }
        write_immutable_json(staged_smoke, smoke_body)
        audit_body = {
            "schema": "h1_carrierid_all_public_source_terminal_audit_v5",
            "status": "PASS_TERMINAL_METADATA_SOURCE_CONFIG_STATE_PAYLOAD_SMOKE_V5",
            "execution_provenance": {
                "nonce_bound_launch_verified": True,
                "execution_receipt": str(EXECUTION),
                "execution_receipt_sha256": execution.execution_sha256,
                "start_marker": str(START),
                "start_marker_sha256": sha256_file(START),
            },
            "launch": {"path": str(LAUNCH), "sha256": sha256_file(LAUNCH),
                       "schema": launch["schema"], "status": launch["status"]},
            "execution": {
                "path": str(EXECUTION), "sha256": execution.execution_sha256,
                "nonce": execution.nonce, "start_time_ns": execution.start_time_ns,
                "preexisting_all_source_run_dirs": list(execution.preexisting_run_dirs),
                "fresh_run_dir": checkpoint.parent.parent.parent.name,
            },
            "start_marker": {
                "path": str(START), "sha256": sha256_file(START),
                "nonce": execution.nonce, "start_time_ns": execution.start_time_ns,
            },
            "checkpoint": {
                "path": str(checkpoint), "sha256": checkpoint_sha,
                "mode": oct(stat.S_IMODE(checkpoint.stat().st_mode)),
                "metadata": dict(metadata), "state_tensor_count": tensor_count,
                "state_finite": True,
            },
            "config": {"path": str(config_path), "sha256": config_sha},
            "source": {
                "binding_sha256_checkpoint": metadata["source_binding_sha256"],
                "binding_sha256_reconstructed": binding_sha,
                "asset_manifest_sha256": assets.manifest_sha256,
                "source_sessions": list(H1_HELDIN_SESSIONS),
            },
            "calibration_allowlist": {
                "counts": {"heldin": len(heldin), "heldout": len(heldout), "total": len(paths)},
                "heldin": [str(path) for path in heldin], "heldout": [str(path) for path in heldout],
                "files": [str(path) for path in paths],
                "dataset_tags": [task_config.hash_dataset(path.stem) for path in paths],
                "held_in_count": len(heldin), "held_out_count": len(heldout),
                "held_in_sessions": list(EXPECTED_HELDIN_SESSIONS),
                "held_out_sessions": list(EXPECTED_HELDOUT_SESSIONS),
                "paths": [str(path) for path in paths], "only_calib_nwb": True,
                "minival_query_test_labels_opened": False,
            },
            "package_preflight": {
                "path": str(PREFLIGHT), "sha256": sha256_file(staged_preflight),
                "schema": preflight["schema"], "status": preflight["status"],
            },
            "payload": {
                "path": str(PAYLOAD), "sha256": payload_sha,
                "schema": packaged["carrier_payload_schema"],
                "dataset_count": len(carriers), "identity_shapes": identity_shapes,
                "shape": {"carrier": [176, 4], "identity": "per_record_m_3_or_4"},
                "calibration_receipts": [dict(row) for row in receipts],
            },
            "runtime_smoke": {
                "path": str(SMOKE), "sha256": sha256_file(staged_smoke),
                "status": smoke_body["status"], "output_dim": EXPECTED_BEHAVIOR_DIM,
            },
            "scope": {
                "source_only": True,
                "calibration_recordings_indexed": 27,
                "calibration_recordings_opened_for_export": 27,
                "held_in_smoked": 13, "held_out_smoked": 14,
                "minival_recordings_opened": 0, "query_recordings_opened": 0,
                "formal_test_labels_opened": 0, "target_optimizer_steps": 0,
                "target_backward_steps": 0, "evalai_accessed_or_submitted": False,
                "cuda_used": False, "cuda_used_for_audit_export_smoke": False,
            },
            "stability_audit": {
                "path": str(STABILITY), "sha256": STABILITY_SHA256,
                "schema": "h1_carrierid_all_public_source_recovery_package_m3_stability_audit_v5",
                "status": "PASS_RECOVERY_M3_FIRST4_PUBLIC_CALIBRATION_STABILITY_NO_QUERY_OPENED",
            },
            "submission": {"authorized": False, "requires_independent_human_review": True},
            "code_sha256": {
                **dict(preflight.get("code_sha256", {})),
                "deployment_data": sha256_file(ROOT / "src/data/h1_carrierid_all_source_deployment.py"),
            },
        }
        write_immutable_json(staged_audit, audit_body)
        _expect(sha256_file(EXECUTION), execution.execution_sha256,
                "execution receipt final SHA")
        _expect(_parse_start_marker(START).get("nonce"), execution.nonce,
                "START final nonce")
        _expect(_new_run_dirs(execution), (checkpoint.parent.parent.parent.resolve(),),
                "fresh run-dir final binding")
        _expect(sha256_file(checkpoint), checkpoint_sha, "checkpoint final SHA")
        _expect(sha256_file(config_path), config_sha, "config final SHA")
        _publish_bundle((
            (staged_preflight, PREFLIGHT), (staged_payload, PAYLOAD),
            (staged_smoke, SMOKE), (staged_audit, AUDIT),
        ))
    _stamp(f"post-training audit/package/smoke PASS audit_sha={_sha256_file(AUDIT)}")


def main() -> None:
    with _exclusive_watch_lock():
        _main_locked()


if __name__ == "__main__":
    main()
