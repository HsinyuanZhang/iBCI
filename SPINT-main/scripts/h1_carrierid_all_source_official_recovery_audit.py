#!/usr/bin/env python3
"""Recover an append-only source-training audit after an unbound launch.

The 2026-08-10 all-source run has a legacy START marker and training log but
no nonce-bound execution receipt.  This utility deliberately does *not* make
up a nonce or rewrite the marker.  After the source-only trainer has exited,
it binds the original marker/log, immutable public-source asset receipt,
resolved config, terminal checkpoint, and (when still available) a process
snapshot into a new immutable recovery receipt.  It never imports a target
dataset, opens an NWB, instantiates a Trainer/model, or submits EvalAI.

The output is append-only: an existing output path is a hard error.  A passing
receipt therefore means "source-only terminal evidence recovered despite a
provenance gap", not "the launch was nonce-bound".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART = ROOT / "pilot_artifacts/h1_carrierid_all_source_official_v1"
ASSET = ART / "H1_CARRIERID_ALL_SOURCE_ASSET_PREFLIGHT_v1.json"
PREPARED_LAUNCH = ART / "H1_CARRIERID_ALL_SOURCE_LAUNCH_RECEIPT_v1.json"
EXECUTION = ART / "H1_CARRIERID_ALL_SOURCE_GPU1_EXECUTION_LAUNCH_v1.json"
START = ART / "logs/h1_all_source_gpu1_start.txt"
TRAIN_LOG = ART / "logs/h1_all_source_gpu1.log"
DEFAULT_OUTPUT = ART / "H1_CARRIERID_ALL_SOURCE_RECOVERY_AUDIT_v1.json"

SCHEMA = "h1_carrierid_all_public_source_gpu1_recovery_audit_v1"
PASS_STATUS = "PASS_SOURCE_ONLY_RECOVERY_WITH_PROVENANCE_GAP"
EXPECTED_PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
EXPECTED_EPOCH = 49
EXPECTED_EPOCHS = 50
EXPECTED_BATCHES_PER_EPOCH = 4133
EXPECTED_GLOBAL_STEP = EXPECTED_EPOCHS * EXPECTED_BATCHES_PER_EPOCH
EXPECTED_LIGHTNING_VERSION = "2.4.0"

CODE_PATHS = {
    "launcher": ROOT / "scripts/h1_carrierid_all_source_official_launcher.py",
    "data": ROOT / "src/data/h1_carrierid_all_source_official.py",
    "model": ROOT / "src/models/h1_carrierid_all_source_official_module.py",
    "experiment": ROOT / "configs/experiment/h1_carrierid_all_source_official.yaml",
    "data_config": ROOT / "configs/data/falcon_h1_carrierid_all_source_official.yaml",
    "model_config": ROOT / "configs/model/falcon_h1_carrierid_all_source_official.yaml",
    "train": ROOT / "src/train.py",
}


class RecoveryAuditError(ValueError):
    """A source-only recovery gate failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryAuditError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444


def _json_object(path: Path, *, immutable: bool = False) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(), f"receipt is not a regular file: {path}")
    if immutable:
        _need(_immutable(path), f"receipt is not immutable mode 0444: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryAuditError(f"receipt JSON is unreadable: {path}") from error
    _need(isinstance(value, dict), f"receipt is not a JSON object: {path}")
    return value


def _parse_start_marker(path: Path) -> dict[str, str]:
    """Read the legacy marker while proving that it was not nonce-bound."""

    # This marker predates the nonce-bound guard and is known to be mode 0664;
    # do not chmod or otherwise mutate it during recovery.  Its observed mode
    # is recorded in the append-only receipt below.
    _need(path.is_file() and not path.is_symlink(), f"legacy START marker is not a regular file: {path}")
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        _need("=" in line and line, f"malformed START marker line: {line!r}")
        key, value = line.split("=", 1)
        _need(bool(key) and key not in rows, f"duplicate/empty START marker key: {key!r}")
        rows[key] = value
    required = {"start_time", "workdir", "tmux_session", "command", "stdout_stderr_log"}
    _need(required.issubset(rows), f"START marker lacks required keys: {sorted(required - set(rows))}")
    forbidden_nonce_keys = {"nonce", "start_time_ns", "execution_receipt"}
    _need(not forbidden_nonce_keys.intersection(rows), "START marker unexpectedly contains nonce-bound fields")
    rows["_observed_mode"] = str(stat.S_IMODE(path.stat().st_mode))
    return rows


def _command_binding(command: str) -> dict[str, Any]:
    """Parse the exact marker command and its environment prefix."""

    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise RecoveryAuditError(f"START command is not shell-parseable: {error}") from error
    environment: dict[str, str] = {}
    while tokens and "=" in tokens[0] and tokens[0].split("=", 1)[0].replace("_", "a").isalnum():
        key, value = tokens.pop(0).split("=", 1)
        environment[key] = value
    # The legacy marker ran from ``workdir`` and therefore records the script
    # as the relative argv token ``src/train.py`` (the prepared receipt keeps
    # the absolute path).  Preserve that exact argv while binding it to the
    # known workdir in the caller.
    _need(tokens[:2] == [EXPECTED_PYTHON, "src/train.py"], "START command executable drift")
    expected_suffix = [
        "experiment=h1_carrierid_all_source_official",
        f"official_candidate.asset_manifest_path={ASSET.resolve()}",
        "ckpt_path=null",
        "test=false",
    ]
    _need(tokens[2:] == expected_suffix, "START command argv drift")
    _need(environment.get("CUDA_VISIBLE_DEVICES") == "1", "START command CUDA binding drift")
    _need(environment.get("PYTHONNOUSERSITE") == "1", "START command Python isolation drift")
    return {"environment": environment, "argv": tokens, "raw": command}


def _process_snapshot(pid: int | None, *, expected_argv: list[str]) -> dict[str, Any]:
    """Capture /proc evidence if a PID remains; never invent post-exit data."""

    if pid is None:
        return {"requested_pid": None, "status": "not_requested", "argv": None, "start_time_ticks": None}
    proc = Path("/proc") / str(pid)
    cmdline = proc / "cmdline"
    stat_path = proc / "stat"
    if not cmdline.is_file() or not stat_path.is_file():
        return {
            "requested_pid": int(pid), "status": "unavailable_after_termination",
            "argv": None, "start_time_ticks": None,
        }
    raw = cmdline.read_bytes().split(b"\0")
    argv = [part.decode("utf-8", "replace") for part in raw if part]
    _need(argv == expected_argv, f"PID {pid} argv does not match the legacy START command")
    stat_text = stat_path.read_text(encoding="utf-8")
    _need(")" in stat_text, f"PID {pid} /proc stat is malformed")
    after_comm = stat_text.rsplit(")", 1)[1].split()
    _need(len(after_comm) > 19 and after_comm[0] != "Z", f"PID {pid} is still running or zombie")
    try:
        start_ticks = int(after_comm[19])
    except ValueError as error:
        raise RecoveryAuditError(f"PID {pid} /proc start time is malformed") from error
    return {"requested_pid": int(pid), "status": "observed_exited", "argv": argv, "start_time_ticks": start_ticks}


def _validate_assets() -> dict[str, Any]:
    body = _json_object(ASSET, immutable=True)
    _need(body.get("schema") == "h1_carrierid_all_public_source_assets_v1", "all-source asset schema drift")
    _need(body.get("status") == "PASS_ALL_PUBLIC_HELDIN_ASSETS_FROZEN_NO_GPU_NO_FORMAL", "asset status drift")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping), "asset scope is malformed")
    for key, expected in {
        "cuda_used": False, "evalai_accessed": False, "formal_test_labels_opened": 0,
        "held_out_query_recordings_opened": 0, "minival_recordings_opened": 0,
        "trainer_constructed": False, "held_in_calibration_recordings_opened": 13,
    }.items():
        _need(scope.get(key) == expected, f"asset scope {key} drift")
    asset_file = (ASSET.parent / str(body.get("asset_file", ""))).resolve()
    _need(asset_file.parent == ASSET.parent.resolve(), "asset array path escapes receipt directory")
    _need(_immutable(asset_file), f"asset array is not immutable mode 0444: {asset_file}")
    asset_sha = _sha256_file(asset_file)
    _need(asset_sha == body.get("asset_file_sha256"), "asset array SHA drift")
    source_files = body.get("source_files")
    _need(isinstance(source_files, list) and len(source_files) == 13, "all-source file list must contain 13 rows")
    observed_sources: list[dict[str, Any]] = []
    forbidden = ("held-out", "minival", "query", "formal", "test_ecephys", "private")
    for row in source_files:
        _need(isinstance(row, Mapping), "all-source source-file row is malformed")
        path = Path(str(row.get("path", ""))).resolve()
        _need(path.is_file() and not path.is_symlink(), f"all-source source file is missing: {path}")
        _need(not any(token in path.as_posix().lower() for token in forbidden), f"forbidden source path: {path}")
        actual = _sha256_file(path)
        _need(actual == row.get("sha256"), f"all-source source file SHA drift: {path}")
        observed_sources.append({"session": row.get("session"), "path": str(path), "sha256": actual})
    return {
        "path": str(ASSET.resolve()), "sha256": _sha256_file(ASSET),
        "asset_file": {"path": str(asset_file), "sha256": asset_sha},
        "source_files": observed_sources,
        "source_sessions": body.get("source_sessions"),
        "source_manifest": body,
    }


def _validate_prepared_launch(*, asset_sha256: str) -> dict[str, Any]:
    body = _json_object(PREPARED_LAUNCH, immutable=True)
    _need(body.get("schema") == "h1_carrierid_all_public_source_official_launch_receipt_v1", "prepared launch schema drift")
    _need(body.get("status") == "PASS_ALL_SOURCE_HC_PREPARED_NOT_LAUNCHED", "prepared launch status drift")
    candidate = body.get("candidate")
    _need(isinstance(candidate, Mapping), "prepared launch candidate is malformed")
    _need(candidate.get("asset_manifest_path") == str(ASSET.resolve()), "prepared candidate asset path drift")
    _need(candidate.get("asset_manifest_sha256") == asset_sha256, "prepared candidate asset SHA drift")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping), "prepared launch scope is malformed")
    for key, expected in {
        "minival_recordings_opened": 0, "formal_test_labels_opened": 0,
        "target_optimizer_steps": 0, "target_backward_steps": 0,
        "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
        "evalai_submission_authorized": False,
    }.items():
        _need(scope.get(key) == expected, f"prepared launch scope {key} drift")
    return {
        "path": str(PREPARED_LAUNCH.resolve()),
        "sha256": _sha256_file(PREPARED_LAUNCH),
        "candidate": dict(candidate),
        "code_sha256": dict(body.get("code_sha256", {})),
    }


def _validate_code_hashes(claimed: Mapping[str, Any]) -> dict[str, Any]:
    current: dict[str, str] = {}
    for key, path in CODE_PATHS.items():
        _need(path.is_file() and not path.is_symlink(), f"source code/config file is missing: {path}")
        current[key] = _sha256_file(path)
        if key in claimed:
            _need(claimed[key] == current[key], f"source code SHA drift: {key}")
    return {"prepared": dict(claimed), "current": current, "all_match": all(claimed.get(k) == v for k, v in current.items() if k in claimed)}


def _finite_state(value: Any, *, label: str) -> None:
    if torch.is_tensor(value):
        _need(bool(torch.isfinite(value).all()), f"{label} contains nonfinite tensor values")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _finite_state(child, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_state(child, label=f"{label}[{index}]")


def _counter(value: Any, *, label: str, expected: int, fields: tuple[str, ...]) -> None:
    _need(isinstance(value, Mapping), f"{label} is missing")
    for key in fields:
        _need(type(value.get(key)) is int and value.get(key) == expected, f"{label}.{key} drift")


def _validate_checkpoint(checkpoint: Path, *, start_mtime_ns: int, asset_sha256: str) -> dict[str, Any]:
    _need(checkpoint.is_file() and not checkpoint.is_symlink(), f"terminal checkpoint is not a regular file: {checkpoint}")
    _need(checkpoint.stat().st_mtime_ns >= start_mtime_ns, "terminal checkpoint predates legacy START marker")
    run_dir = checkpoint.parent.parent.parent.resolve()
    config_path = run_dir / ".hydra/config.yaml"
    _need(config_path.is_file() and not config_path.is_symlink(), f"terminal resolved config is missing: {config_path}")
    config_sha = _sha256_file(config_path)
    checkpoint_sha_before = _sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_sha_after = _sha256_file(checkpoint)
    _need(checkpoint_sha_before == checkpoint_sha_after, "terminal checkpoint changed while loading")
    _need(isinstance(payload, Mapping), "terminal checkpoint is not a mapping")
    _need(type(payload.get("epoch")) is int and payload.get("epoch") == EXPECTED_EPOCH, "terminal epoch is not e49")
    _need(type(payload.get("global_step")) is int and payload.get("global_step") == EXPECTED_GLOBAL_STEP, "terminal global_step drift")
    _need(payload.get("pytorch-lightning_version") == EXPECTED_LIGHTNING_VERSION, "terminal Lightning version drift")
    optimizer_states = payload.get("optimizer_states")
    _need(isinstance(optimizer_states, list) and len(optimizer_states) == 1, "terminal optimizer-state count drift")
    _finite_state(optimizer_states, label="optimizer_states")
    _need(payload.get("lr_schedulers") == [], "terminal checkpoint unexpectedly contains scheduler state")
    loops = payload.get("loops")
    _need(isinstance(loops, Mapping) and isinstance(loops.get("fit_loop"), Mapping), "terminal fit-loop state missing")
    fit = loops["fit_loop"]
    epoch_progress = fit.get("epoch_progress")
    _need(isinstance(epoch_progress, Mapping), "terminal epoch progress missing")
    _counter(epoch_progress.get("total"), label="epoch_progress.total", expected=EXPECTED_EPOCHS, fields=("ready", "started", "processed"))
    _counter(epoch_progress.get("current"), label="epoch_progress.current", expected=EXPECTED_EPOCHS, fields=("ready", "started", "processed"))
    _counter(epoch_progress["total"], label="epoch_progress.total", expected=EXPECTED_EPOCH, fields=("completed",))
    _counter(epoch_progress["current"], label="epoch_progress.current", expected=EXPECTED_EPOCH, fields=("completed",))
    _need(isinstance(fit.get("epoch_loop.state_dict"), Mapping), "terminal epoch-loop state missing")
    _need(fit["epoch_loop.state_dict"].get("_batches_that_stepped") == EXPECTED_GLOBAL_STEP, "terminal stepped-batch count drift")
    batch_progress = fit.get("epoch_loop.batch_progress")
    _need(isinstance(batch_progress, Mapping), "terminal batch progress missing")
    for scope, expected in (("total", EXPECTED_GLOBAL_STEP), ("current", EXPECTED_BATCHES_PER_EPOCH)):
        _counter(batch_progress.get(scope), label=f"batch_progress.{scope}", expected=expected, fields=("ready", "completed", "started", "processed"))
    _need(batch_progress.get("is_last_batch") is True, "terminal batch is not last")
    automatic = fit.get("epoch_loop.automatic_optimization.optim_progress")
    _need(isinstance(automatic, Mapping) and isinstance(automatic.get("optimizer"), Mapping), "terminal optimizer progress missing")
    optimizer = automatic["optimizer"]
    step = optimizer.get("step")
    _need(isinstance(step, Mapping), "terminal optimizer step progress missing")
    for scope, expected in (("total", EXPECTED_GLOBAL_STEP), ("current", EXPECTED_BATCHES_PER_EPOCH)):
        _counter(step.get(scope), label=f"optimizer.step.{scope}", expected=expected, fields=("ready", "completed"))
    zero_grad = optimizer.get("zero_grad")
    _need(isinstance(zero_grad, Mapping), "terminal zero-grad progress missing")
    for scope, expected in (("total", EXPECTED_GLOBAL_STEP), ("current", EXPECTED_BATCHES_PER_EPOCH)):
        _counter(zero_grad.get(scope), label=f"optimizer.zero_grad.{scope}", expected=expected, fields=("ready", "completed", "started"))
    scheduler_progress = fit.get("epoch_loop.scheduler_progress")
    _need(isinstance(scheduler_progress, Mapping), "terminal scheduler progress missing")
    for scope in ("total", "current"):
        _counter(scheduler_progress.get(scope), label=f"scheduler.{scope}", expected=0, fields=("ready", "completed"))
    state_dict = payload.get("state_dict")
    _need(isinstance(state_dict, Mapping) and bool(state_dict), "terminal state_dict is empty")
    for name, tensor in state_dict.items():
        _need(torch.is_tensor(tensor), f"terminal state_dict entry is not a tensor: {name}")
        _need(bool(torch.isfinite(tensor).all()), f"terminal state_dict entry is nonfinite: {name}")
    metadata = payload.get("h1_carrierid_all_source_official")
    _need(isinstance(metadata, Mapping), "all-source terminal metadata is missing")
    for key, expected in {
        "schema": "h1_carrierid_all_public_source_terminal_checkpoint_v1", "fresh_seed": 42,
        "checkpoint_epoch_zero_based": EXPECTED_EPOCH, "epochs_completed": EXPECTED_EPOCHS,
        "selected_by": "fixed_terminal_epoch_no_validation_no_formal_selection",
        "checkpoint_warm_start": False, "target_optimizer_steps": 0,
        "target_backward_steps": 0, "formal_test_labels_opened": 0,
        "asset_manifest_sha256": asset_sha256, "config_sha256": config_sha,
    }.items():
        _need(metadata.get(key) == expected, f"terminal metadata {key} drift")
    return {
        "path": str(checkpoint.resolve()), "sha256": checkpoint_sha_before, "bytes": checkpoint.stat().st_size,
        "run_dir": str(run_dir), "config": {"path": str(config_path), "sha256": config_sha},
        "epoch": int(payload["epoch"]), "epochs_completed": EXPECTED_EPOCHS,
        "global_step": int(payload["global_step"]), "batches_per_epoch": EXPECTED_BATCHES_PER_EPOCH,
        "optimizer_states": len(optimizer_states), "lr_schedulers": 0,
        "state_dict_tensors": len(state_dict), "state_finite": True,
        "metadata": dict(metadata),
    }


def _write_append_only(path: Path, value: Mapping[str, Any]) -> str:
    output = Path(path).resolve()
    _need(not output.exists() and not output.is_symlink(), f"refusing to overwrite recovery audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(_immutable(output), f"recovery audit is not immutable mode 0444: {output}")
    return _sha256_file(output)


def recover(
    *, checkpoint: Path, output: Path = DEFAULT_OUTPUT, pid: int | None = None,
    start_marker: Path = START, train_log: Path = TRAIN_LOG,
) -> dict[str, Any]:
    """Build one immutable recovery receipt; never replace an existing one."""

    _need(not EXECUTION.exists() and not EXECUTION.is_symlink(),
          "nonce-bound execution receipt exists; refuse provenance-gap recovery")
    marker = _parse_start_marker(start_marker)
    binding = _command_binding(marker["command"])
    _need(Path(marker["workdir"]).resolve() == ROOT.resolve(), "legacy START workdir drift")
    _need(marker["tmux_session"] == "h1_all_source_gpu1", "legacy START tmux session drift")
    _need(Path(marker["stdout_stderr_log"]).resolve() == train_log.resolve(), "legacy START log path drift")
    _need(train_log.is_file() and not train_log.is_symlink(), f"training log is missing: {train_log}")
    _need(train_log.stat().st_mtime_ns >= start_marker.stat().st_mtime_ns, "training log predates START marker")
    process = _process_snapshot(pid, expected_argv=binding["argv"])
    _need(process["status"] != "running", "recovery audit cannot run while training PID is live")
    assets = _validate_assets()
    prepared = _validate_prepared_launch(asset_sha256=assets["sha256"])
    code = _validate_code_hashes(prepared["code_sha256"])
    terminal = _validate_checkpoint(checkpoint.resolve(), start_mtime_ns=start_marker.stat().st_mtime_ns, asset_sha256=assets["sha256"])
    body: dict[str, Any] = {
        "schema": SCHEMA, "status": PASS_STATUS,
        "provenance_gap": {
            "execution_receipt_path": str(EXECUTION.resolve()), "execution_receipt_present": False,
            "nonce_bound_launch_verified": False, "nonce_reconstructed": False,
            "legacy_start_marker_path": str(start_marker.resolve()),
            "legacy_start_marker_sha256": _sha256_file(start_marker),
            "legacy_start_marker_mode": stat.S_IMODE(start_marker.stat().st_mode),
            "legacy_start_marker_has_nonce": False,
            "interpretation": "source-only terminal evidence recovered; original launch nonce receipt was missing and was not fabricated",
        },
        "start_marker": {**marker, "command_binding": binding},
        "process": process,
        "cuda": {"visible_devices_from_start_command": binding["environment"].get("CUDA_VISIBLE_DEVICES"), "source_training_gpu": 1},
        "training_log": {"path": str(train_log.resolve()), "sha256": _sha256_file(train_log), "bytes": train_log.stat().st_size},
        "prepared_launch": prepared,
        "asset_evidence": {key: value for key, value in assets.items() if key != "source_manifest"},
        "code_evidence": code,
        "terminal_checkpoint": terminal,
        "scope": {
            "source_recordings_only": True, "target_recordings_opened": 0,
            "formal_test_labels_opened": 0, "target_optimizer_steps": 0,
            "target_backward_steps": 0, "evalai_submission_authorized": False,
            "target_evaluation_authorized_by_this_receipt": False,
        },
    }
    body["audit_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    body["output"] = {"path": str(Path(output).resolve())}
    body["output"]["sha256"] = _write_append_only(Path(output), body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--start-marker", type=Path, default=START)
    parser.add_argument("--train-log", type=Path, default=TRAIN_LOG)
    args = parser.parse_args()
    print(json.dumps(recover(
        checkpoint=args.checkpoint, output=args.output, pid=args.pid,
        start_marker=args.start_marker, train_log=args.train_log,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
