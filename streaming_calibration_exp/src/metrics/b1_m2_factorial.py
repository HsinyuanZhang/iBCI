"""Pure-Python contract utilities for the B1 matched 2×2 factorial.

This module intentionally performs no Torch, CUDA, NWB, or model import.  It
defines the only accepted cell lattice and validates post-training score
receipts.  A practical Stage-P result gates the *predeclared* Stage-F folds;
it cannot select an alternative fold, descriptor, loss weight, or epoch.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCREEN_ID = "m2_carrier_distillation_interaction_v2"
SCHEMA_VERSION = 1
CARRIERS = ("t4", "z4")
LOSS_MODES = ("task_plus_y", "task_plus_y_plus_E")
SEEDS = (42, 43, 44)
EPOCH_WINDOW = tuple(range(5, 13))
STAGE_P_FOLDS = (0,)
STAGE_F_FOLDS = (1, 2, 3)
STAGE_P_PRACTICAL_INTERACTION = 0.03
FINAL_PRACTICAL_INTERACTION = 0.03
CROSSED_BOOTSTRAP_SEED = 20260813
CROSSED_BOOTSTRAP_DRAWS = 20_000

# B1 source binding is a reviewed runtime closure, not a repository snapshot.
# The generic artifact writer intentionally records the whole project tree;
# adding an unrelated experiment or test must not invalidate an already
# running B1 cell.  These entry points plus the Hydra-instantiated targets are
# recursively closed over local Python imports by ``source_bindings`` below.
B1_RUNTIME_PYTHON_ENTRYPOINTS = (
    "streaming_calibration_exp/src/train.py",
    "streaming_calibration_exp/src/metrics/b1_m2_factorial.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "streaming_calibration_exp/src/data/b1_m2_matched_z4_datamodule.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/callbacks/override_epoch_step.py",
    "streaming_calibration_exp/scripts/preflight_b1_m2_factorial.py",
    "streaming_calibration_exp/scripts/run_b1_m2_factorial.py",
    "streaming_calibration_exp/scripts/execute_b1_m2_factorial_cell.py",
    "streaming_calibration_exp/scripts/bind_b1_m2_factorial_post_training.py",
    "streaming_calibration_exp/scripts/score_b1_m2_factorial_epochs.py",
    "streaming_calibration_exp/scripts/aggregate_b1_m2_factorial.py",
)

# Hydra composes these files for every B1 arm.  Listing the configuration
# closure explicitly is preferable to hashing all unrelated experiment YAMLs:
# each entry either contributes to train.yaml's selected defaults or is one of
# the four frozen factorial overrides.
B1_RUNTIME_CONFIG_FILES = (
    "streaming_calibration_exp/configs/train.yaml",
    "streaming_calibration_exp/configs/data/falcon_m2.yaml",
    "streaming_calibration_exp/configs/data/falcon_m2_b1_matched_z4.yaml",
    "streaming_calibration_exp/configs/model/_streaming_base.yaml",
    "streaming_calibration_exp/configs/model/streaming_b3s_t4.yaml",
    "streaming_calibration_exp/configs/callbacks/b1_epoch_window.yaml",
    "streaming_calibration_exp/configs/logger/tensorboard.yaml",
    "streaming_calibration_exp/configs/trainer/default.yaml",
    "streaming_calibration_exp/configs/trainer/gpu.yaml",
    "streaming_calibration_exp/configs/paths/default.yaml",
    "streaming_calibration_exp/configs/extras/default.yaml",
    "streaming_calibration_exp/configs/hydra/default.yaml",
    "streaming_calibration_exp/configs/experiment/b1_v2_m2_t4_task_plus_y.yaml",
    "streaming_calibration_exp/configs/experiment/b1_v2_m2_t4_task_plus_y_plus_E.yaml",
    "streaming_calibration_exp/configs/experiment/b1_v2_m2_z4_task_plus_y.yaml",
    "streaming_calibration_exp/configs/experiment/b1_v2_m2_z4_task_plus_y_plus_E.yaml",
)

# These are deliberately a *bounded scientific configuration* rather than an
# opaque hash of an artifact directory.  Paths used only to place logs are not
# scientific degrees of freedom; all data/model/optimization/callback fields
# are.  Both preflight and the post-training scorer use this exact projection.
SCIENCE_CONFIG_KEYS = (
    "seed", "train", "test", "ckpt_path", "no_early_stopping",
    "require_baseline_validation", "data", "model", "trainer", "callbacks",
    "optimizer", "scheduler", "baseline_metrics_path",
)


class B1ContractError(ValueError):
    """Raised for an incomplete or scientifically incomparable B1 matrix."""


@dataclass(frozen=True)
class CellSpec:
    stage: str
    fold: int
    seed: int
    carrier: str
    loss_mode: str

    @property
    def key(self) -> str:
        return f"{self.stage}/f{self.fold}/s{self.seed}/{self.carrier}/{self.loss_mode}"


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def science_config_projection(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete non-runtime B1 science configuration.

    The projection is intentionally exact for the selected top-level fields;
    no individual hyperparameter can silently disappear from the comparison.
    Hydra/log paths and task naming are excluded because they only determine
    file placement.  ``json`` round-tripping makes the result immutable and
    rejects non-serializable config values instead of silently stringifying
    them.
    """
    if not isinstance(cfg, Mapping):
        raise B1ContractError("Science config must be a mapping")
    # Optimizer/scheduler may be model-owned and therefore absent from this
    # repository's Hydra root config.  Preserve that absence as an explicit
    # null in the projection rather than requiring a fictitious config group.
    missing = [key for key in SCIENCE_CONFIG_KEYS if key not in cfg and key not in {"optimizer", "scheduler"}]
    if missing:
        raise B1ContractError(f"Science config missing required keys: {missing}")
    selected = {key: cfg.get(key) for key in SCIENCE_CONFIG_KEYS}
    try:
        return json.loads(canonical_json_bytes(selected))
    except (TypeError, ValueError) as exc:
        raise B1ContractError("Science config is not canonical JSON") from exc


def science_config_sha256(cfg: Mapping[str, Any]) -> str:
    return sha256_payload(science_config_projection(cfg))


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically create body + digest sidecar, refusing every overwrite."""
    path = path.resolve()
    sidecar = Path(f"{path}.sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"B1 immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload) + b"\n"
    digest = hashlib.sha256(body).hexdigest()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    try:
        fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(f"{digest}  {path.name}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    return digest


def write_future_launch_receipt(
    path: Path,
    *,
    spec: CellSpec,
    official_preflight_sha256: str,
    command: Iterable[str],
    explicit_log_dir: Path,
    artifact_parent: Path,
    run_id_prefix: str,
    future_post_training_binding: Path,
    future_score_receipt: Path,
    interpreter: Path,
    script: Path,
    working_dir: Path,
    cuda_visible_devices: str,
    execution_start_receipt: Path,
    execution_completion_receipt: Path,
    execution_log: Path,
) -> str:
    """Create the O_EXCL receipt a later authorized launcher must use.

    This helper does not launch anything.  It makes collision/freshness a
    fail-closed contract: no pre-existing explicit run directory, launch
    receipt, or score receipt is allowed.  The current runner never calls it.
    """
    if any(item is None for item in (
        future_post_training_binding,
        future_score_receipt,
        execution_start_receipt,
        execution_completion_receipt,
        execution_log,
    )):
        raise B1ContractError(
            "Every B1 launch contract requires post-training-binding, score, start, completion, and execution-log paths"
        )
    execution_paths = (execution_start_receipt, execution_completion_receipt, execution_log)
    immutable_outputs = (future_post_training_binding, future_score_receipt)
    if (explicit_log_dir.exists() or artifact_parent.exists()
            or any(item.exists() or Path(f"{item}.sha256").exists() for item in immutable_outputs)
            or path.exists() or Path(f"{path}.sha256").exists()
            or any(item.exists() or Path(f"{item}.sha256").exists() for item in execution_paths)):
        raise B1ContractError("B1 future launch freshness check failed")
    interpreter = interpreter.resolve()
    script = script.resolve()
    working_dir = working_dir.resolve()
    if not interpreter.is_absolute() or not script.is_absolute() or not working_dir.is_absolute():
        raise B1ContractError("B1 future launch execution context must be absolute")
    if not isinstance(cuda_visible_devices, str) or not cuda_visible_devices.isdecimal():
        raise B1ContractError("B1 one-cell CUDA_VISIBLE_DEVICES must name exactly one decimal GPU index")
    command = list(command)
    if not command or command[:2] != [str(interpreter), str(script)]:
        raise B1ContractError("B1 future launch command must begin with its bound interpreter and train script")
    environment_contract = execution_environment_contract(cuda_visible_devices)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt_kind": "b1_m2_factorial_future_launch_contract",
        "screen_id": SCREEN_ID,
        "cell": spec.key,
        "official_preflight_sha256": official_preflight_sha256,
        "command": command,
        "unique_explicit_log_dir": str(explicit_log_dir.resolve()),
        "unique_artifact_parent": str(artifact_parent.resolve()),
        "run_id_prefix": run_id_prefix,
        "future_post_training_binding": str(future_post_training_binding.resolve()),
        "future_score_receipt": str(future_score_receipt.resolve()),
        "execution_context": {
            "interpreter": str(interpreter),
            "script": str(script),
            "working_dir": str(working_dir),
        },
        "cuda_visible_devices": cuda_visible_devices,
        "environment_contract": environment_contract,
        "execution_receipts": {
            "start": str(execution_start_receipt.resolve()),
            "completion": str(execution_completion_receipt.resolve()),
            "stdout_stderr_log": str(execution_log.resolve()),
        },
        "o_excl": True,
        "training_started_by_this_helper": False,
    }
    return write_immutable_json(path, payload)


def execution_environment_contract(cuda_visible_devices: str) -> dict[str, str]:
    """Return the exact B1-owned subprocess environment fields.

    The executor may preserve unrelated inherited environment variables, but
    these three fields are immutable scientific/execution provenance and are
    checked again by the binder and scorer.
    """
    if not isinstance(cuda_visible_devices, str) or not cuda_visible_devices.isdecimal():
        raise B1ContractError("B1 one-cell CUDA_VISIBLE_DEVICES must name exactly one decimal GPU index")
    return {
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONUNBUFFERED": "1",
    }


def write_execution_start_receipt(
    path: Path,
    *,
    future_launch_receipt: Path,
    cuda_visible_devices: str,
) -> str:
    """Record the one-cell subprocess intent immediately before invoking it.

    This is intentionally distinct from the pre-execution contract, whose
    ``training_started_by_this_helper`` field is false.  The start receipt is
    written *before* process creation, so it truthfully marks the invocation
    as attempted but does not claim a subprocess actually started.  Only the
    completion receipt may set ``subprocess_started=true`` and report an exit
    code.
    """
    launch, launch_sha = load_verified_immutable_json(future_launch_receipt)
    if launch.get("receipt_kind") != "b1_m2_factorial_future_launch_contract":
        raise B1ContractError("B1 execution start requires a future launch contract")
    if not cuda_visible_devices.isdecimal() or cuda_visible_devices != launch.get("cuda_visible_devices"):
        raise B1ContractError("B1 execution start CUDA device differs from immutable contract")
    receipts = launch.get("execution_receipts")
    if (not isinstance(receipts, Mapping)
            or any(not isinstance(receipts.get(key), str) or not receipts.get(key)
                   for key in ("start", "completion", "stdout_stderr_log"))
            or Path(str(receipts.get("start"))).resolve() != path.resolve()):
        raise B1ContractError("B1 execution start path differs from immutable contract")
    expected_environment = execution_environment_contract(cuda_visible_devices)
    if launch.get("environment_contract") != expected_environment:
        raise B1ContractError("B1 execution start environment differs from immutable contract")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt_kind": "b1_m2_factorial_execution_start",
        "screen_id": SCREEN_ID,
        "cell": launch.get("cell"),
        "future_launch_receipt_path": str(future_launch_receipt.resolve()),
        "future_launch_receipt_sha256": launch_sha,
        "command": launch.get("command"),
        "working_dir": launch.get("execution_context", {}).get("working_dir"),
        "cuda_visible_devices": cuda_visible_devices,
        "environment_contract": expected_environment,
        "execution_attempted": True,
        "subprocess_started": False,
        "subprocess_invocation_pending_at_receipt_write": True,
        "subprocess_exit_code": None,
        "training_completed_successfully": False,
    }
    return write_immutable_json(path, payload)


def write_execution_completion_receipt(
    path: Path,
    *,
    future_launch_receipt: Path,
    execution_start_receipt: Path,
    invoked_command: Iterable[str],
    invoked_working_dir: Path,
    invoked_environment: Mapping[str, str],
    subprocess_started: bool,
    exit_code: int,
) -> str:
    """Bind the actual completed invocation to launch + pre-spawn receipts."""
    future, future_sha = load_verified_immutable_json(future_launch_receipt)
    if future.get("receipt_kind") != "b1_m2_factorial_future_launch_contract":
        raise B1ContractError("B1 execution completion requires a future launch contract")
    start, start_sha = load_verified_immutable_json(execution_start_receipt)
    if start.get("receipt_kind") != "b1_m2_factorial_execution_start":
        raise B1ContractError("B1 execution completion requires an execution-start receipt")
    if (Path(str(start.get("future_launch_receipt_path", ""))).resolve() != future_launch_receipt.resolve()
            or start.get("future_launch_receipt_sha256") != future_sha):
        raise B1ContractError("B1 execution start does not bind the supplied future launch contract")
    receipts = future.get("execution_receipts")
    if (not isinstance(receipts, Mapping)
            or Path(str(receipts.get("start", ""))).resolve() != execution_start_receipt.resolve()
            or Path(str(receipts.get("completion", ""))).resolve() != path.resolve()):
        raise B1ContractError("B1 execution completion path differs from immutable contract")
    command = list(invoked_command)
    working_dir = invoked_working_dir.resolve()
    expected_environment = future.get("environment_contract")
    actual_environment = {
        key: invoked_environment.get(key)
        for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTHONUNBUFFERED")
    }
    if (command != future.get("command") or command != start.get("command")
            or str(working_dir) != future.get("execution_context", {}).get("working_dir")
            or str(working_dir) != start.get("working_dir")
            or actual_environment != expected_environment
            or actual_environment != start.get("environment_contract")
            or actual_environment.get("CUDA_VISIBLE_DEVICES") != future.get("cuda_visible_devices")
            or future.get("cell") != start.get("cell")):
        raise B1ContractError("B1 actual invocation command/context/device/cell differs from immutable chain")
    if subprocess_started is not True:
        raise B1ContractError("B1 completion cannot claim an invocation that did not start")
    if not isinstance(exit_code, int):
        raise B1ContractError("B1 subprocess exit code must be an integer")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt_kind": "b1_m2_factorial_execution_completion",
        "screen_id": SCREEN_ID,
        "cell": future.get("cell"),
        "future_launch_receipt_path": str(future_launch_receipt.resolve()),
        "future_launch_receipt_sha256": future_sha,
        "execution_start_receipt_path": str(execution_start_receipt.resolve()),
        "execution_start_receipt_sha256": start_sha,
        "actual_invoked_command": command,
        "actual_working_dir": str(working_dir),
        "actual_cuda_visible_devices": actual_environment["CUDA_VISIBLE_DEVICES"],
        "actual_environment_contract": actual_environment,
        "execution_attempted": True,
        "subprocess_started": True,
        "subprocess_exit_code": exit_code,
        "training_completed_successfully": exit_code == 0,
        "post_training_binding_completed_by_this_receipt": False,
        "scoring_completed_by_this_receipt": False,
    }
    return write_immutable_json(path, payload)


def validate_execution_chain(
    *,
    future_launch_receipt: Path,
    execution_start_receipt: Path,
    execution_completion_receipt: Path,
    require_success: bool = True,
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any], str]:
    """Independently reopen and verify the immutable launch execution chain."""
    launch, launch_sha = load_verified_immutable_json(future_launch_receipt)
    start, start_sha = load_verified_immutable_json(execution_start_receipt)
    completion, completion_sha = load_verified_immutable_json(execution_completion_receipt)
    if launch.get("receipt_kind") != "b1_m2_factorial_future_launch_contract":
        raise B1ContractError("B1 chain launch kind drift")
    receipts = launch.get("execution_receipts")
    required_paths = {
        "start": execution_start_receipt.resolve(),
        "completion": execution_completion_receipt.resolve(),
    }
    if (not isinstance(receipts, Mapping)
            or any(not isinstance(receipts.get(key), str) or Path(str(receipts[key])).resolve() != expected
                   for key, expected in required_paths.items())
            or not isinstance(receipts.get("stdout_stderr_log"), str)
            or not receipts.get("stdout_stderr_log")):
        raise B1ContractError("B1 chain launch lacks mandatory start/completion/log bindings")
    expected_environment = execution_environment_contract(str(launch.get("cuda_visible_devices", "")))
    expected_context = launch.get("execution_context")
    if (not isinstance(expected_context, Mapping)
            or launch.get("environment_contract") != expected_environment
            or any(not isinstance(expected_context.get(key), str) or not Path(str(expected_context[key])).is_absolute()
                   for key in ("interpreter", "script", "working_dir"))):
        raise B1ContractError("B1 chain launch execution context/environment malformed")
    if (start.get("receipt_kind") != "b1_m2_factorial_execution_start"
            or start.get("future_launch_receipt_path") != str(future_launch_receipt.resolve())
            or start.get("future_launch_receipt_sha256") != launch_sha
            or start.get("cell") != launch.get("cell")
            or start.get("command") != launch.get("command")
            or start.get("working_dir") != expected_context.get("working_dir")
            or start.get("cuda_visible_devices") != launch.get("cuda_visible_devices")
            or start.get("environment_contract") != expected_environment
            or start.get("execution_attempted") is not True
            or start.get("subprocess_started") is not False
            or start.get("subprocess_exit_code") is not None):
        raise B1ContractError("B1 chain execution-start provenance drift")
    if (completion.get("receipt_kind") != "b1_m2_factorial_execution_completion"
            or completion.get("future_launch_receipt_path") != str(future_launch_receipt.resolve())
            or completion.get("future_launch_receipt_sha256") != launch_sha
            or completion.get("execution_start_receipt_path") != str(execution_start_receipt.resolve())
            or completion.get("execution_start_receipt_sha256") != start_sha
            or completion.get("cell") != launch.get("cell")
            or completion.get("actual_invoked_command") != launch.get("command")
            or completion.get("actual_working_dir") != expected_context.get("working_dir")
            or completion.get("actual_cuda_visible_devices") != launch.get("cuda_visible_devices")
            or completion.get("actual_environment_contract") != expected_environment
            or completion.get("execution_attempted") is not True
            or completion.get("subprocess_started") is not True
            or not isinstance(completion.get("subprocess_exit_code"), int)
            or completion.get("training_completed_successfully") != (completion.get("subprocess_exit_code") == 0)):
        raise B1ContractError("B1 chain execution-completion provenance drift")
    if require_success and (completion.get("subprocess_exit_code") != 0
                            or completion.get("training_completed_successfully") is not True):
        raise B1ContractError("B1 chain subprocess did not complete successfully")
    if require_success and not Path(str(receipts["stdout_stderr_log"])).is_file():
        raise B1ContractError("B1 chain successful subprocess lacks its committed execution log")
    return launch, launch_sha, start, start_sha, completion, completion_sha


def write_post_training_binding(
    path: Path,
    *,
    future_launch_receipt: Path,
    artifact: Path,
    checkpoint_run_dir: Path,
    interpreter: Path,
    script: Path,
    working_dir: Path,
    execution_completion_receipt: Path,
) -> str:
    """O_EXCL bind the actual artifact and Hydra checkpoint directory after a run.

    This is intentionally a provenance binder, not an execution assertion. It
    refuses to claim that the pre-launch receipt itself proves training ran.
    The scorer requires this post-training binding before it will combine the
    otherwise-separate artifact root and Hydra checkpoint run directory.
    """
    if execution_completion_receipt is None:
        raise B1ContractError("Every B1 post-training binding requires a successful completion receipt")
    launch, launch_sha = load_verified_immutable_json(future_launch_receipt)
    if launch.get("receipt_kind") != "b1_m2_factorial_future_launch_contract":
        raise B1ContractError("B1 post-training binding requires a future launch contract")
    expected_log = Path(str(launch.get("unique_explicit_log_dir", ""))).resolve()
    expected_parent = Path(str(launch.get("unique_artifact_parent", ""))).resolve()
    expected_binding = Path(str(launch.get("future_post_training_binding", ""))).resolve()
    if path.resolve() != expected_binding:
        raise B1ContractError("B1 post-training binding path differs from future launch contract")
    artifact = artifact.resolve()
    checkpoint_run_dir = checkpoint_run_dir.resolve()
    if checkpoint_run_dir != expected_log or artifact.parent != expected_parent:
        raise B1ContractError("B1 artifact/checkpoint paths do not match committed launch contract")
    prefix = str(launch.get("run_id_prefix", ""))
    if not prefix or not artifact.name.startswith(prefix):
        raise B1ContractError("B1 artifact run-id stem does not match committed launch contract")
    if not expected_parent.is_dir():
        raise B1ContractError("B1 committed artifact parent is absent")
    matching_artifacts = sorted(
        child.resolve() for child in expected_parent.iterdir()
        if child.is_dir() and child.name.startswith(prefix)
    )
    if matching_artifacts != [artifact]:
        raise B1ContractError(
            "B1 committed artifact parent must contain exactly one matching run-id directory equal to supplied artifact"
        )
    required_artifact = ("resolved_config.yaml", "source_manifest.json", "split_manifest.json", "teacher_metadata.json")
    if any(not (artifact / name).is_file() for name in required_artifact):
        raise B1ContractError("B1 post-training artifact metadata is incomplete")
    expected_context = launch.get("execution_context")
    if not isinstance(expected_context, Mapping):
        raise B1ContractError("B1 future launch lacks immutable execution context")
    actual_context = {
        "interpreter": str(interpreter.resolve()),
        "script": str(script.resolve()),
        "working_dir": str(working_dir.resolve()),
    }
    if actual_context != dict(expected_context):
        raise B1ContractError("B1 post-training execution context differs from future launch contract")
    execution_receipts = launch.get("execution_receipts")
    if (not isinstance(execution_receipts, Mapping)
            or not isinstance(execution_receipts.get("start"), str) or not execution_receipts.get("start")
            or not isinstance(execution_receipts.get("completion"), str) or not execution_receipts.get("completion")
            or not isinstance(execution_receipts.get("stdout_stderr_log"), str)
            or not execution_receipts.get("stdout_stderr_log")):
        raise B1ContractError("B1 future launch lacks mandatory execution receipt/log paths")
    start_path = Path(str(execution_receipts["start"]))
    expected_completion = Path(str(execution_receipts["completion"]))
    if expected_completion.resolve() != execution_completion_receipt.resolve():
        raise B1ContractError("B1 execution completion path differs from future launch contract")
    (_launch, verified_launch_sha, _start, start_sha, _completion, completion_sha) = validate_execution_chain(
        future_launch_receipt=future_launch_receipt,
        execution_start_receipt=start_path,
        execution_completion_receipt=execution_completion_receipt,
        require_success=True,
    )
    if verified_launch_sha != launch_sha:
        raise B1ContractError("B1 execution chain launch SHA differs during binding")
    epoch_dir = checkpoint_run_dir / "checkpoints" / "epoch_ckpts"
    epochs = {str(epoch): sha256_file(epoch_dir / f"epoch_{epoch - 1:03d}.ckpt") for epoch in EPOCH_WINDOW}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt_kind": "b1_m2_factorial_post_training_path_binding",
        "screen_id": SCREEN_ID,
        "cell": launch.get("cell"),
        "future_launch_receipt_path": str(future_launch_receipt.resolve()),
        "future_launch_receipt_sha256": launch_sha,
        "official_preflight_sha256": launch.get("official_preflight_sha256"),
        "artifact": str(artifact),
        "checkpoint_run_dir": str(checkpoint_run_dir),
        "epochs004_011_sha256": epochs,
        "execution_context": actual_context,
        "execution_start_receipt_path": str(start_path.resolve()),
        "execution_start_receipt_sha256": start_sha,
        "execution_completion_receipt_path": str(execution_completion_receipt.resolve()),
        "execution_completion_receipt_sha256": completion_sha,
        "execution_claim": "paths_and_completed_epoch_files_bound_after_training; launch receipt alone did not prove execution",
    }
    return write_immutable_json(path, payload)


def load_verified_immutable_json(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    sidecar = Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise B1ContractError(f"Missing immutable body/sidecar: {path}")
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    expected = sidecar.read_text(encoding="ascii").strip().split(maxsplit=1)
    if not expected or expected[0] != digest:
        raise B1ContractError(f"Immutable digest mismatch: {path}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise B1ContractError(f"Invalid JSON receipt: {path}") from exc
    if not isinstance(payload, dict):
        raise B1ContractError(f"Receipt root must be an object: {path}")
    return payload, digest


def cells_for_stage(stage: str) -> tuple[CellSpec, ...]:
    if stage == "P":
        folds = STAGE_P_FOLDS
    elif stage == "F":
        folds = STAGE_F_FOLDS
    else:
        raise B1ContractError(f"Unknown B1 stage: {stage!r}")
    return tuple(
        CellSpec(stage, fold, seed, carrier, loss_mode)
        for fold in folds
        for seed in SEEDS
        for carrier in CARRIERS
        for loss_mode in LOSS_MODES
    )


def expected_config_name(carrier: str, loss_mode: str) -> str:
    if carrier not in CARRIERS or loss_mode not in LOSS_MODES:
        raise B1ContractError(f"Bad B1 factor level carrier={carrier} loss={loss_mode}")
    suffix = "task_plus_y" if loss_mode == "task_plus_y" else "task_plus_y_plus_E"
    return f"b1_v2_m2_{carrier}_{suffix}"


def expected_preflight_cell(preflight: Mapping[str, Any], spec: CellSpec) -> Mapping[str, Any]:
    """Find exactly one declared cell in an immutable official preflight."""
    if preflight.get("screen_id") != SCREEN_ID:
        raise B1ContractError("Wrong B1 screen ID in preflight")
    rows = preflight.get("stage_p_cells" if spec.stage == "P" else "stage_f_cells")
    if not isinstance(rows, list):
        raise B1ContractError(f"Preflight lacks Stage-{spec.stage} cell lattice")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("cell") == spec.key]
    if len(matches) != 1:
        raise B1ContractError(f"Preflight has no unique declaration for {spec.key}")
    cell = matches[0]
    for key in ("experiment", "science_config_sha256", "science_config"):
        if key not in cell:
            raise B1ContractError(f"Preflight cell {spec.key} missing {key}")
    if cell["experiment"] != expected_config_name(spec.carrier, spec.loss_mode):
        raise B1ContractError(f"Preflight experiment drift for {spec.key}")
    return cell


def _relative_under_project(path: str) -> str | None:
    prefix = "streaming_calibration_exp/"
    return path[len(prefix):] if path.startswith(prefix) else None


def expected_b1_source_manifest_bindings(bindings: Mapping[str, Any]) -> dict[str, str]:
    """Relevant source subset that ``train.py`` must record.

    ``write_source_manifest`` records paths relative to
    ``streaming_calibration_exp`` and intentionally includes a broader tree
    than B1 uses.  This maps the reviewed runtime closure to that artifact
    convention; unrelated manifest rows are forensic extras, not dependencies.
    """
    files = bindings.get("files") if isinstance(bindings, Mapping) else None
    if not isinstance(files, Mapping):
        raise B1ContractError("Implementation bindings lack file digest map")
    expected: dict[str, str] = {}
    for root_relative, digest in files.items():
        rel = _relative_under_project(str(root_relative))
        if rel is not None:
            expected[rel] = str(digest)
    if not expected:
        raise B1ContractError("No source-manifest-eligible B1 bindings")
    if bindings.get("artifact_source_manifest_sha256") != sha256_payload(expected):
        raise B1ContractError("B1 artifact source-manifest binding digest drift")
    return expected


def validate_exact_artifact_source_manifest(
    observed: Mapping[str, Any], bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the reviewed subset and report, but do not reject, extras.

    The complete artifact file remains byte-bound by
    ``source_manifest_sha256`` in every score receipt.  This function answers
    the separate scientific question: were all B1 runtime/config dependencies
    present with their reviewed bytes?
    """
    expected = expected_b1_source_manifest_bindings(bindings)
    actual = {str(path): str(digest) for path, digest in observed.items()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    drift = sorted(path for path in set(expected).intersection(actual) if expected[path] != actual[path])
    if missing or drift:
        raise B1ContractError(
            f"B1 relevant artifact source manifest drift: missing={missing[:5]} changed={drift[:5]}"
        )
    relevant_actual = {path: actual[path] for path in sorted(expected)}
    extras = {path: actual[path] for path in extra}
    return {
        "policy": "required_recursive_runtime_subset_exact_unrelated_extras_recorded",
        "required_count": len(expected),
        "required_manifest_sha256": sha256_payload(relevant_actual),
        "extra_count": len(extras),
        "extra_manifest_sha256": sha256_payload(extras),
        # Keep paths for a human-readable provenance audit; their digests are
        # already committed by the full artifact source_manifest byte hash.
        "extra_paths": extra,
    }


def validate_runtime_dependency_bindings(bindings: Mapping[str, Any]) -> None:
    """Validate the independently hashed recursive B1 runtime closure."""
    runtime = bindings.get("runtime_dependency_manifest") if isinstance(bindings, Mapping) else None
    if not isinstance(runtime, Mapping) or not runtime:
        raise B1ContractError("B1 implementation bindings lack runtime dependency manifest")
    normalized = {str(path): str(digest) for path, digest in runtime.items()}
    if bindings.get("runtime_dependency_manifest_sha256") != sha256_payload(normalized):
        raise B1ContractError("B1 runtime dependency manifest digest drift")
    required = {
        "streaming_calibration_exp/src/train.py",
        "streaming_calibration_exp/src/metrics/b1_m2_factorial.py",
        "streaming_calibration_exp/src/data/falcon_datamodule.py",
        "streaming_calibration_exp/src/data/b1_m2_matched_z4_datamodule.py",
        "streaming_calibration_exp/src/models/streaming_calibration_module.py",
        "streaming_calibration_exp/src/models/components/streaming_spint.py",
        "streaming_calibration_exp/configs/train.yaml",
        "streaming_calibration_exp/configs/model/streaming_b3s_t4.yaml",
        "streaming_calibration_exp/configs/data/falcon_m2.yaml",
        "streaming_calibration_exp/configs/data/falcon_m2_b1_matched_z4.yaml",
        "streaming_calibration_exp/src/data/validation_protocol.py",
        "streaming_calibration_exp/third_party/catalyst/distributed_sampler.py",
        "streaming_calibration_exp/third_party/falcon_challenge/filtering.py",
    }
    missing = sorted(required - set(normalized))
    if missing:
        raise B1ContractError(f"B1 runtime dependency closure missing reviewed imports: {missing}")


def validate_live_source_bindings(root: Path, expected: Mapping[str, Any]) -> None:
    """Fail closed if any reviewed source/config/runtime dependency changed."""
    validate_runtime_dependency_bindings(expected)
    live = source_bindings(root)
    if dict(expected) != live:
        expected_runtime = dict(expected.get("runtime_dependency_manifest") or {})
        live_runtime = dict(live.get("runtime_dependency_manifest") or {})
        missing = sorted(set(expected_runtime) - set(live_runtime))
        extra = sorted(set(live_runtime) - set(expected_runtime))
        changed = sorted(
            path for path in set(expected_runtime).intersection(live_runtime)
            if expected_runtime[path] != live_runtime[path]
        )
        raise B1ContractError(
            "B1 live recursive runtime/config dependency closure differs from immutable bindings: "
            f"missing={missing[:5]} extra={extra[:5]} changed={changed[:5]}"
        )


def validate_preflight_payload(preflight: Mapping[str, Any]) -> None:
    """Validate the immutable protocol lattice before a runner/scorer trusts it."""
    if preflight.get("schema_version") != SCHEMA_VERSION or preflight.get("screen_id") != SCREEN_ID:
        raise B1ContractError("Official B1 preflight identity drift")
    if preflight.get("status") != "CPU_PREFLIGHT_READY_GPU_NOT_AUTHORIZED":
        raise B1ContractError("B1 preflight is not the reviewed no-launch state")
    bindings = preflight.get("implementation_bindings")
    if not isinstance(bindings, Mapping) or preflight.get("implementation_bindings_sha256") != sha256_payload(bindings):
        raise B1ContractError("Official B1 implementation-binding digest drift")
    validate_runtime_dependency_bindings(bindings)
    # Also validates that the artifact-writer-compatible source tree has its
    # own digest and cannot silently omit a runtime file.
    expected_b1_source_manifest_bindings(bindings)
    teacher = bindings.get("teacher")
    if not isinstance(teacher, Mapping):
        raise B1ContractError("Official B1 preflight lacks teacher-byte binding")
    for field in ("checkpoint", "teacher_hydra_config"):
        entry = teacher.get(field)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str) or len(str(entry["sha256"])) != 64:
            raise B1ContractError(f"Official B1 preflight teacher {field} binding missing")
    for stage, expected in (("P", cells_for_stage("P")), ("F", cells_for_stage("F"))):
        rows = preflight.get("stage_p_cells" if stage == "P" else "stage_f_cells")
        if not isinstance(rows, list) or {str(row.get("cell")) for row in rows if isinstance(row, Mapping)} != {cell.key for cell in expected}:
            raise B1ContractError(f"Official B1 Stage-{stage} lattice drift")
        for cell in expected:
            declared = expected_preflight_cell(preflight, cell)
            science = declared.get("science_config")
            if not isinstance(science, Mapping) or science.get("train") is not True or science.get("test") is not False:
                raise B1ContractError(f"Official B1 {cell.key} train/test projection drift")
            if declared.get("science_config_sha256") != sha256_payload(science):
                raise B1ContractError(f"Official B1 {cell.key} science projection digest drift")
    expansion = preflight.get("stage_f_expansion_manifest")
    if not isinstance(expansion, Mapping):
        raise B1ContractError("Official B1 Stage-F expansion manifest missing")
    unhashed_expansion = {key: value for key, value in expansion.items() if key != "sha256"}
    if expansion.get("sha256") != sha256_payload(unhashed_expansion):
        raise B1ContractError("Official B1 Stage-F expansion manifest digest drift")
    if expansion.get("stage") != "F" or expansion.get("predeclared_folds") != list(STAGE_F_FOLDS):
        raise B1ContractError("Official B1 Stage-F expansion folds drift")
    if expansion.get("expected_cell_keys") != [cell.key for cell in cells_for_stage("F")]:
        raise B1ContractError("Official B1 Stage-F expansion lattice drift")


def _finite(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise B1ContractError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise B1ContractError(f"{label} is non-finite: {value!r}")
    return parsed


def _score_one_cell(payload: Mapping[str, Any], spec: CellSpec) -> tuple[str, float]:
    """Return the sole fixed-LOSO session and fixed epoch-window R² score."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise B1ContractError(f"{spec.key}: schema version drift")
    for key, expected in {
        "screen_id": SCREEN_ID,
        "stage": spec.stage,
        "fold": spec.fold,
        "seed": spec.seed,
        "carrier": spec.carrier,
        "loss_mode": spec.loss_mode,
        "development_scope_only": True,
        "formal_test_or_external_heldout_opened": False,
        "target_session_backward_updates": False,
        "target_session_decoder_weight_updates": False,
        "target_session_weight_updates_during_b1": False,
    }.items():
        if payload.get(key) != expected:
            raise B1ContractError(f"{spec.key}: {key} drift: {payload.get(key)!r} != {expected!r}")
    teacher = payload.get("teacher_provenance")
    expected_teacher = {
        "during_B1_target_weight_updates": False,
        "legacy_frozen_teacher_pretraining_included_B1_validation_session": True,
        "clean_teacher_target_exclusion": False,
        "interpretation": "internal_development_only",
    }
    if not isinstance(teacher, Mapping) or dict(teacher) != expected_teacher:
        raise B1ContractError(f"{spec.key}: teacher provenance drift")
    query = payload.get("query_provenance")
    expected_query = {
        "support_direction_labels_used_for_carrier": True,
        "query_behavior_loaded_for_validation_scoring": True,
        "query_behavior_used_for_gradient_updates": False,
        "query_behavior_used_for_carrier_fit": False,
        "query_behavior_used_for_normalizer_fit": False,
        "query_behavior_used_for_checkpoint_selection": False,
        # Stage-P's aggregate is the frozen routing criterion for the
        # predeclared Stage-F continuation.  It is therefore a legitimate
        # selection use of its query behavior, but never a training/carrier/
        # normalizer/checkpoint-selection use.  Stage-F has no further gate.
        "query_behavior_used_for_stage_f_continuation_gate": spec.stage == "P",
        "external_heldout_opened": False,
    }
    if not isinstance(query, Mapping) or dict(query) != expected_query:
        raise B1ContractError(f"{spec.key}: query provenance drift")
    provenance = payload.get("carrier_provenance")
    if not isinstance(provenance, Mapping):
        raise B1ContractError(f"{spec.key}: carrier provenance missing")
    for key, expected in {
        "carrier_fit_executed": True,
        "calibration_direction_labels_read": True,
        "target_session_carrier_fit_executed": True,
        "target_session_direction_labels_used_for_carrier": True,
        "target_session_query_labels_used_for_carrier": False,
        "source_normalizer_fit_only": True,
    }.items():
        if provenance.get(key) != expected:
            raise B1ContractError(f"{spec.key}: carrier provenance {key} drift")
    expected_model_visible = "standardized_t4" if spec.carrier == "t4" else "all_zero_mask_after_standardized_t4"
    if provenance.get("model_visible_carrier") != expected_model_visible:
        raise B1ContractError(f"{spec.key}: matched carrier visibility drift")
    artifact = payload.get("source_artifact")
    if not isinstance(artifact, Mapping):
        raise B1ContractError(f"{spec.key}: source artifact binding missing")
    required_artifact_fields = (
        "resolved_config_sha256", "science_config_sha256", "source_manifest_sha256",
        "split_manifest_sha256", "teacher_metadata_sha256", "checkpoint_bundle",
        "normalizer_binding", "split_semantics", "post_training_path_binding_sha256", "execution_context", "checkpoint_run_dir",
    )
    if any(key not in artifact for key in required_artifact_fields):
        raise B1ContractError(f"{spec.key}: incomplete source artifact science binding")
    checkpoints = artifact.get("checkpoint_bundle")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {str(epoch) for epoch in EPOCH_WINDOW}:
        raise B1ContractError(f"{spec.key}: every epoch 5--12 checkpoint must be bound")
    for epoch in EPOCH_WINDOW:
        ckpt = checkpoints[str(epoch)]
        if not isinstance(ckpt, Mapping) or not isinstance(ckpt.get("sha256"), str) or len(str(ckpt["sha256"])) != 64:
            raise B1ContractError(f"{spec.key}: epoch {epoch} checkpoint SHA missing")
        if ckpt.get("stored_epoch") != epoch - 1 or ckpt.get("logical_epoch") != epoch:
            raise B1ContractError(f"{spec.key}: epoch {epoch} checkpoint epoch binding drift")
    loss = payload.get("loss")
    if not isinstance(loss, Mapping):
        raise B1ContractError(f"{spec.key}: loss binding missing")
    expected_loss = (
        {"lambda_y": 1.0, "lambda_E": 0.0}
        if spec.loss_mode == "task_plus_y"
        else {"lambda_y": 1.0, "lambda_E": 0.1}
    )
    for key, expected in expected_loss.items():
        if _finite(loss.get(key), label=f"{spec.key}/{key}") != expected:
            raise B1ContractError(f"{spec.key}: {key} drift")
    if tuple(payload.get("epoch_window") or ()) != EPOCH_WINDOW:
        raise B1ContractError(f"{spec.key}: must score fixed logical epochs 5--12")
    per_epoch = payload.get("per_epoch_session_r2")
    if not isinstance(per_epoch, Mapping) or set(per_epoch) != {str(epoch) for epoch in EPOCH_WINDOW}:
        raise B1ContractError(f"{spec.key}: epoch receipt is incomplete")
    expected_session = None
    epoch_scores: list[float] = []
    for epoch in EPOCH_WINDOW:
        row = per_epoch[str(epoch)]
        if not isinstance(row, Mapping) or len(row) != 1:
            raise B1ContractError(
                f"{spec.key}: Stage-{spec.stage} one-fold cell must contain exactly one LOSO validation session"
            )
        session, value = next(iter(row.items()))
        if expected_session is None:
            expected_session = str(session)
        elif str(session) != expected_session:
            raise B1ContractError(f"{spec.key}: validation session changed across epochs")
        epoch_scores.append(_finite(value, label=f"{spec.key}/epoch{epoch}/{session}"))
    assert expected_session is not None
    return expected_session, sum(epoch_scores) / len(epoch_scores)


def aggregate_stage(stage: str, receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Fail closed, then aggregate the declared factorial scores.

    ``receipts`` must be keyed by ``CellSpec.key``.  Stage P has one fixed
    session and therefore reports seed consistency only.  Stage F aggregates
    exactly the predeclared folds 1,2,3 together with the immutable Stage-P
    fold 0 only in the separate final aggregator.
    """
    cells = cells_for_stage(stage)
    expected_keys = {cell.key for cell in cells}
    if set(receipts) != expected_keys:
        missing = sorted(expected_keys - set(receipts))
        extra = sorted(set(receipts) - expected_keys)
        raise B1ContractError(f"Stage {stage} receipt lattice mismatch; missing={missing}, extra={extra}")
    scores: dict[tuple[int, int, str, str], float] = {}
    sessions: dict[tuple[int, int], str] = {}
    shared_preflight_sha: str | None = None
    shared_bindings: Mapping[str, Any] | None = None
    for cell in cells:
        receipt = receipts[cell.key]
        session, score = _score_one_cell(receipt, cell)
        official = receipt.get("official_preflight")
        if not isinstance(official, Mapping):
            raise B1ContractError(f"{cell.key}: official-preflight binding missing")
        bindings = official.get("implementation_bindings")
        if not isinstance(bindings, Mapping) or official.get("implementation_bindings_sha256") != sha256_payload(bindings):
            raise B1ContractError(f"{cell.key}: official-preflight bindings drift")
        preflight_sha = receipt.get("official_preflight_sha256")
        if not isinstance(preflight_sha, str) or len(preflight_sha) != 64:
            raise B1ContractError(f"{cell.key}: official preflight SHA missing")
        if official.get("full_preflight_sha256") != preflight_sha:
            raise B1ContractError(f"{cell.key}: embedded preflight SHA drift")
        if shared_preflight_sha is None:
            shared_preflight_sha, shared_bindings = preflight_sha, bindings
        elif preflight_sha != shared_preflight_sha or dict(bindings) != dict(shared_bindings or {}):
            raise B1ContractError(f"{cell.key}: cross-cell official-preflight drift")
        pair = (cell.fold, cell.seed)
        if pair in sessions and sessions[pair] != session:
            raise B1ContractError(f"{cell.key}: carrier/loss arms disagree on validation session")
        sessions[pair] = session
        scores[(cell.fold, cell.seed, cell.carrier, cell.loss_mode)] = score
    interaction_by_fold_seed: dict[str, float] = {}
    for fold in (STAGE_P_FOLDS if stage == "P" else STAGE_F_FOLDS):
        for seed in SEEDS:
            interaction = (
                scores[(fold, seed, "t4", "task_plus_y")]
                - scores[(fold, seed, "z4", "task_plus_y")]
                - scores[(fold, seed, "t4", "task_plus_y_plus_E")]
                + scores[(fold, seed, "z4", "task_plus_y_plus_E")]
            )
            interaction_by_fold_seed[f"f{fold}_s{seed}"] = interaction
    values = tuple(interaction_by_fold_seed.values())
    mean = sum(values) / len(values)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "stage": stage,
        "cell_count": len(cells),
        "epoch_window": list(EPOCH_WINDOW),
        "score_definition": "unweighted mean of the sole fixed-LOSO validation-session R2 over logical epochs 5--12",
        "validation_session_by_fold_seed": {f"f{fold}_s{seed}": session for (fold, seed), session in sorted(sessions.items())},
        "interaction_by_fold_seed": interaction_by_fold_seed,
        "mean_interaction": mean,
        "all_interactions_positive": all(value > 0.0 for value in values),
        "official_preflight_sha256": shared_preflight_sha,
        "official_preflight_screen_id": SCREEN_ID,
        "expected_cell_keys": sorted(expected_keys),
        "implementation_bindings_sha256": sha256_payload(shared_bindings or {}),
    }
    if stage == "P":
        result["stage_f_predeclared_gate"] = {
            "threshold": STAGE_P_PRACTICAL_INTERACTION,
            "mean_interaction_at_least_threshold": mean >= STAGE_P_PRACTICAL_INTERACTION,
            "all_three_seed_interactions_positive": all(value > 0.0 for value in values),
            "stage_f_authorized_by_evidence": (
                mean >= STAGE_P_PRACTICAL_INTERACTION and all(value > 0.0 for value in values)
            ),
            "if_false": "STOP_B1_NO_STAGE_F",
        }
    return result


def validate_stage_p_aggregate(
    payload: Mapping[str, Any], *, official_preflight_sha256: str | None = None,
) -> None:
    """Recompute and validate the immutable Stage-P routing aggregate.

    Stage P has exactly three seed interactions on one fixed LOSO session.  It
    is a routing screen only: this helper proves its lattice, finite values,
    reported mean, and frozen continuation gate before either the runner or
    the Stage-F aggregator can trust it.
    """
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("screen_id") != SCREEN_ID:
        raise B1ContractError("Stage-P aggregate identity drift")
    if payload.get("stage") != "P":
        raise B1ContractError("Stage-P aggregate stage drift")
    if official_preflight_sha256 is not None and payload.get("official_preflight_sha256") != official_preflight_sha256:
        raise B1ContractError("Stage-P aggregate official-preflight drift")
    expected_keys = {f"f0_s{seed}" for seed in SEEDS}
    interactions = payload.get("interaction_by_fold_seed")
    if not isinstance(interactions, Mapping) or set(interactions) != expected_keys:
        raise B1ContractError("Stage-P aggregate interaction lattice drift")
    values = tuple(_finite(interactions[key], label=f"Stage-P/{key}") for key in sorted(expected_keys))
    mean = sum(values) / len(values)
    if not math.isclose(_finite(payload.get("mean_interaction"), label="Stage-P mean"), mean, rel_tol=0.0, abs_tol=1e-12):
        raise B1ContractError("Stage-P aggregate mean does not equal its exact three interactions")
    if payload.get("all_interactions_positive") is not all(value > 0.0 for value in values):
        raise B1ContractError("Stage-P aggregate sign summary drift")
    expected_cells = {cell.key for cell in cells_for_stage("P")}
    if payload.get("expected_cell_keys") != sorted(expected_cells):
        raise B1ContractError("Stage-P aggregate expected cell lattice drift")
    receipts = payload.get("cell_receipt_sha256")
    if not isinstance(receipts, Mapping) or set(receipts) != expected_cells:
        raise B1ContractError("Stage-P aggregate cell-receipt lattice drift")
    if any(not isinstance(value, str) or len(value) != 64 for value in receipts.values()):
        raise B1ContractError("Stage-P aggregate cell-receipt SHA malformed")
    gate = payload.get("stage_f_predeclared_gate")
    if not isinstance(gate, Mapping):
        raise B1ContractError("Stage-P aggregate continuation gate missing")
    expected_authorized = mean >= STAGE_P_PRACTICAL_INTERACTION and all(value > 0.0 for value in values)
    expected_gate = {
        "threshold": STAGE_P_PRACTICAL_INTERACTION,
        "mean_interaction_at_least_threshold": mean >= STAGE_P_PRACTICAL_INTERACTION,
        "all_three_seed_interactions_positive": all(value > 0.0 for value in values),
        "stage_f_authorized_by_evidence": expected_authorized,
        "if_false": "STOP_B1_NO_STAGE_F",
    }
    if dict(gate) != expected_gate:
        raise B1ContractError("Stage-P aggregate continuation gate drift")


def make_cell_score_receipt(
    *,
    spec: CellSpec,
    preflight: Mapping[str, Any],
    run_artifact: Path,
    per_epoch_session_r2: Mapping[str, Mapping[str, float]],
    artifact_binding: Mapping[str, Any],
    checkpoint_bundle: Mapping[str, Any],
    official_preflight_sha256: str,
) -> dict[str, Any]:
    """Build a strictly validated, immutable-ready score receipt.

    The evaluator supplies the measured R² values, but this function prevents
    it from silently changing a B1 factor level, the epoch window, or the
    preflight implementation bindings.  The source training artifact is only
    *bound* by path/SHA here; it is never treated as a best-checkpoint result.
    """
    validate_preflight_payload(preflight)
    declared = expected_preflight_cell(preflight, spec)
    bindings = preflight.get("implementation_bindings")
    if not isinstance(bindings, Mapping):
        raise B1ContractError("B1 preflight implementation bindings missing")
    if preflight.get("implementation_bindings_sha256") != sha256_payload(bindings):
        raise B1ContractError("B1 preflight implementation binding digest drift")
    if not isinstance(official_preflight_sha256, str) or len(official_preflight_sha256) != 64:
        raise B1ContractError("B1 scorer missing official preflight immutable SHA")
    if artifact_binding.get("science_config_sha256") != declared.get("science_config_sha256"):
        raise B1ContractError("B1 artifact science config does not equal official preflight cell")
    if set(checkpoint_bundle) != {str(epoch) for epoch in EPOCH_WINDOW}:
        raise B1ContractError("B1 checkpoint bundle must bind every logical epoch 5--12")
    if set(per_epoch_session_r2) != {str(epoch) for epoch in EPOCH_WINDOW}:
        raise B1ContractError("B1 scorer must receive exactly epochs 5--12")
    # Verify a single, fixed validation session across the stored epoch scores.
    session = None
    normalized_scores: dict[str, dict[str, float]] = {}
    for epoch in EPOCH_WINDOW:
        row = per_epoch_session_r2[str(epoch)]
        if not isinstance(row, Mapping) or len(row) != 1:
            raise B1ContractError("B1 each epoch must have exactly one fixed-LOSO session score")
        name, value = next(iter(row.items()))
        if session is None:
            session = str(name)
        elif str(name) != session:
            raise B1ContractError("B1 validation session drift across fixed epochs")
        normalized_scores[str(epoch)] = {str(name): _finite(value, label=f"epoch{epoch}/{name}")}
    if not run_artifact.is_dir():
        raise B1ContractError(f"B1 source artifact is not a directory: {run_artifact}")
    resolved_config = run_artifact / "resolved_config.yaml"
    source_manifest = run_artifact / "source_manifest.json"
    split_manifest = run_artifact / "split_manifest.json"
    for path in (resolved_config, source_manifest, split_manifest):
        if not path.is_file():
            raise B1ContractError(f"B1 source artifact missing required binding: {path.name}")
    loss = (
        {"mode": "task_plus_y", "lambda_y": 1.0, "lambda_E": 0.0}
        if spec.loss_mode == "task_plus_y"
        else {"mode": "task_plus_y_plus_E", "lambda_y": 1.0, "lambda_E": 0.1}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_kind": "b1_m2_factorial_cell_score",
        "screen_id": SCREEN_ID,
        "stage": spec.stage,
        "fold": spec.fold,
        "seed": spec.seed,
        "carrier": spec.carrier,
        "loss_mode": spec.loss_mode,
        "development_scope_only": True,
        "formal_test_or_external_heldout_opened": False,
        "target_session_backward_updates": False,
        "target_session_decoder_weight_updates": False,
        "target_session_weight_updates_during_b1": False,
        "teacher_provenance": {
            "during_B1_target_weight_updates": False,
            "legacy_frozen_teacher_pretraining_included_B1_validation_session": True,
            "clean_teacher_target_exclusion": False,
            "interpretation": "internal_development_only",
        },
        "query_provenance": {
            "support_direction_labels_used_for_carrier": True,
            "query_behavior_loaded_for_validation_scoring": True,
            "query_behavior_used_for_gradient_updates": False,
            "query_behavior_used_for_carrier_fit": False,
            "query_behavior_used_for_normalizer_fit": False,
            "query_behavior_used_for_checkpoint_selection": False,
            "query_behavior_used_for_stage_f_continuation_gate": spec.stage == "P",
            "external_heldout_opened": False,
        },
        "epoch_window": list(EPOCH_WINDOW),
        "per_epoch_session_r2": normalized_scores,
        "loss": loss,
        "carrier_provenance": {
            "carrier_fit_executed": True,
            "calibration_direction_labels_read": True,
            "target_session_carrier_fit_executed": True,
            "target_session_direction_labels_used_for_carrier": True,
            "target_session_query_labels_used_for_carrier": False,
            "source_normalizer_fit_only": True,
            "model_visible_carrier": "standardized_t4" if spec.carrier == "t4" else "all_zero_mask_after_standardized_t4",
        },
        "official_preflight": {
            "implementation_bindings": dict(bindings),
            "implementation_bindings_sha256": preflight["implementation_bindings_sha256"],
            "full_preflight_sha256": official_preflight_sha256,
        },
        "source_artifact": {**dict(artifact_binding), "path": str(run_artifact.resolve()), "checkpoint_bundle": dict(checkpoint_bundle)},
        "checkpoint_selection": "none; all fixed logical epochs 5--12 are forward-scored",
    }


def aggregate_final(stage_p: Mapping[str, Any], stage_f: Mapping[str, Any]) -> dict[str, Any]:
    """Return Stage-F confirmatory primary plus a clearly descriptive P+F sensitivity."""
    if (stage_p.get("screen_id") != SCREEN_ID or stage_f.get("screen_id") != SCREEN_ID
            or stage_p.get("stage") != "P" or stage_f.get("stage") != "F"):
        raise B1ContractError("Final B1 aggregate requires Stage P plus Stage F receipts")
    gate = stage_p.get("stage_f_predeclared_gate")
    if not isinstance(gate, Mapping) or gate.get("stage_f_authorized_by_evidence") is not True:
        raise B1ContractError("Stage F is not authorized by the immutable Stage-P gate")
    validate_stage_p_aggregate(stage_p, official_preflight_sha256=str(stage_f.get("official_preflight_sha256")))
    if stage_p.get("official_preflight_sha256") != stage_f.get("official_preflight_sha256"):
        raise B1ContractError("Stage P/F official-preflight drift")
    if stage_p.get("official_preflight_screen_id") != SCREEN_ID or stage_f.get("official_preflight_screen_id") != SCREEN_ID:
        raise B1ContractError("Stage P/F official-preflight screen drift")
    if stage_p.get("expected_cell_keys") != sorted(cell.key for cell in cells_for_stage("P")):
        raise B1ContractError("Stage-P receipt lattice provenance drift")
    if stage_f.get("expected_cell_keys") != sorted(cell.key for cell in cells_for_stage("F")):
        raise B1ContractError("Stage-F receipt lattice provenance drift")
    for label, stage_payload, expected_cells in (
        ("P", stage_p, cells_for_stage("P")),
        ("F", stage_f, cells_for_stage("F")),
    ):
        lattice = stage_payload.get("cell_receipt_sha256")
        if not isinstance(lattice, Mapping) or set(lattice) != {cell.key for cell in expected_cells}:
            raise B1ContractError(f"Stage-{label} immutable cell-receipt lattice drift")
        if any(not isinstance(value, str) or len(value) != 64 for value in lattice.values()):
            raise B1ContractError(f"Stage-{label} cell-receipt SHA is malformed")
    stage_p_values = dict(stage_p.get("interaction_by_fold_seed") or {})
    stage_f_values = dict(stage_f.get("interaction_by_fold_seed") or {})
    if set(stage_p_values).intersection(stage_f_values):
        raise B1ContractError("Stage P/F fold overlap")
    expected_f_keys = {f"f{fold}_s{seed}" for fold in STAGE_F_FOLDS for seed in SEEDS}
    if set(stage_f_values) != expected_f_keys:
        raise B1ContractError("Stage-F confirmatory interaction lattice is incomplete")
    values = tuple(_finite(stage_f_values[key], label=f"Stage-F/{key}") for key in sorted(expected_f_keys))
    sessions = dict(stage_f.get("validation_session_by_fold_seed") or {})
    if set(sessions) != expected_f_keys:
        raise B1ContractError("Stage-F confirmatory validation-session lattice is incomplete")
    session_by_fold: dict[int, str] = {}
    for fold in STAGE_F_FOLDS:
        names = {str(sessions[f"f{fold}_s{seed}"]) for seed in SEEDS}
        if len(names) != 1:
            raise B1ContractError(f"Final B1 fold {fold} validation session changes across seeds")
        session_by_fold[fold] = next(iter(names))
    if len(set(session_by_fold.values())) != 3:
        raise B1ContractError("Stage-F primary needs three distinct LOSO validation sessions")
    ordered_folds = STAGE_F_FOLDS
    matrix = np.asarray(
        [[_finite(stage_f_values[f"f{fold}_s{seed}"], label=f"f{fold}_s{seed}") for seed in SEEDS] for fold in ordered_folds],
        dtype=np.float64,
    )
    rng = np.random.default_rng(CROSSED_BOOTSTRAP_SEED)
    # A draw samples the three confirmatory session rows and three seed columns independently.
    # The sampled session row is shared over every selected seed within a draw,
    # preserving the requested crossed resampling structure.
    sampled_sessions = rng.integers(0, len(ordered_folds), size=(CROSSED_BOOTSTRAP_DRAWS, len(ordered_folds)))
    sampled_seeds = rng.integers(0, len(SEEDS), size=(CROSSED_BOOTSTRAP_DRAWS, len(SEEDS)))
    bootstrap = matrix[sampled_sessions[:, :, None], sampled_seeds[:, None, :]].mean(axis=(1, 2))
    session_means = {session_by_fold[fold]: float(matrix[index].mean()) for index, fold in enumerate(ordered_folds)}
    seed_means = {str(seed): float(matrix[:, index].mean()) for index, seed in enumerate(SEEDS)}
    mean_interaction = float(matrix.mean())
    terminal_gate = {
        "mean_interaction_at_least": FINAL_PRACTICAL_INTERACTION,
        "mean_interaction_at_least_threshold": mean_interaction >= FINAL_PRACTICAL_INTERACTION,
        "all_three_session_means_positive": all(value > 0.0 for value in session_means.values()),
        "all_three_seed_means_positive": all(value > 0.0 for value in seed_means.values()),
    }
    terminal_gate["terminal_pass"] = bool(
        terminal_gate["mean_interaction_at_least_threshold"]
        and terminal_gate["all_three_session_means_positive"]
        and terminal_gate["all_three_seed_means_positive"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "stage": "F_confirmatory",
        "interaction_by_fold_seed": stage_f_values,
        "mean_interaction": mean_interaction,
        "all_interactions_positive": all(value > 0.0 for value in values),
        "inference_unit": "crossed 3 predeclared Stage-F LOSO sessions × 3 fixed seeds",
        "note": "Stage P is routing-only; no fold/seed selection occurred after Stage-P. Stage-F folds were predeclared.",
        "session_means": session_means,
        "seed_means": seed_means,
        "crossed_bootstrap": {
            "kind": "two_way_crossed_session_and_seed_resampling",
            "rng_seed": CROSSED_BOOTSTRAP_SEED,
            "draws": CROSSED_BOOTSTRAP_DRAWS,
            "ci_95_percentile_descriptive_only": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "no_p_value_or_small_n_null_claim": True,
        },
        "terminal_gate": terminal_gate,
        "p_plus_f_descriptive_sensitivity_only": {
            "label": "descriptive_sensitivity_never_terminal",
            "interaction_by_fold_seed": {**stage_p_values, **stage_f_values},
            "mean_interaction": float(np.mean([
                _finite(value, label=f"P_plus_F/{key}")
                for key, value in sorted({**stage_p_values, **stage_f_values}.items())
            ])),
            "terminal_gate_applied": False,
        },
    }


def _file_binding(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise B1ContractError(f"Missing B1 implementation binding: {relative}")
    return {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _local_python_module_index(root: Path) -> dict[str, Path]:
    """Index importable local modules without importing application code."""
    project = root / "streaming_calibration_exp"
    index: dict[str, Path] = {}
    for base_name in ("src", "third_party"):
        base = project / base_name
        if not base.is_dir():
            raise B1ContractError(f"Missing B1 local Python tree: streaming_calibration_exp/{base_name}")
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(project).with_suffix("")
            parts = list(relative.parts)
            if parts[-1] == "__init__":
                parts.pop()
            if parts:
                index[".".join(parts)] = path
    return index


def _module_name_for_path(path: Path, *, index: Mapping[str, Path]) -> str | None:
    resolved = path.resolve()
    for module, candidate in index.items():
        if candidate.resolve() == resolved:
            return module
    # B1 scripts are executable entry points rather than importable modules.
    # They contain no relative imports, so ``None`` is sufficient here.
    return None


def _import_requests(path: Path, *, module_name: str | None) -> set[str]:
    """Return import names referenced by one Python file using AST only."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise B1ContractError(f"Cannot parse B1 runtime dependency {path}") from exc
    requests: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            requests.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base = node.module or ""
        if node.level:
            if module_name is None:
                raise B1ContractError(f"Relative import in non-module B1 entry point: {path}")
            package = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
            try:
                base = importlib.util.resolve_name("." * node.level + base, package)
            except (ImportError, ValueError) as exc:
                raise B1ContractError(f"Cannot resolve local import in {path}") from exc
        if base:
            requests.add(base)
        for alias in node.names:
            if alias.name != "*" and base:
                requests.add(f"{base}.{alias.name}")
    return requests


def _add_module_and_packages(module: str, *, index: Mapping[str, Path], output: set[Path]) -> None:
    """Add a local module and every import-executed package ``__init__``."""
    parts = module.split(".")
    for length in range(1, len(parts) + 1):
        candidate = ".".join(parts[:length])
        path = index.get(candidate)
        if path is not None:
            output.add(path)


def _runtime_python_closure(root: Path) -> set[Path]:
    """Compute B1's recursive local-import closure from reviewed entry points.

    Hydra targets do not appear as Python imports, so their modules are listed
    as entry points above.  Once seeded, every eager local ``src`` and
    ``third_party`` import is followed recursively.  A newly introduced true
    dependency therefore changes the importing file *and* enters the next
    preflight closure, while an unreferenced experiment file remains outside.
    """
    root = root.resolve()
    index = _local_python_module_index(root)
    discovered: set[Path] = set()
    for relative in B1_RUNTIME_PYTHON_ENTRYPOINTS:
        path = (root / relative).resolve()
        if not path.is_file():
            raise B1ContractError(f"Missing B1 Python entry point: {relative}")
        discovered.add(path)

    queue = list(sorted(discovered))
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        module_name = _module_name_for_path(path, index=index)
        additions: set[Path] = set()
        for request in _import_requests(path, module_name=module_name):
            # The full request may be an imported attribute.  Only names in
            # the local module index are dependencies; external packages are
            # environment provenance, not mutable repository source.
            _add_module_and_packages(request, index=index, output=additions)
        for dependency in additions:
            if dependency not in discovered:
                discovered.add(dependency)
                queue.append(dependency)
    return discovered


def source_bindings(root: Path) -> dict[str, Any]:
    """Hash B1's explicit recursive runtime/configuration dependency closure.

    The artifact writer records a broad repository snapshot.  ``files`` is the
    scientifically relevant subset of that snapshot; the scorer requires this
    subset byte-for-byte and records unrelated extras separately.  Vendored
    ``third_party`` imports are additionally included in the runtime manifest
    even though the generic artifact writer deliberately excludes that tree.
    """
    root = root.resolve()
    project_rel = "streaming_calibration_exp/"
    python_paths = _runtime_python_closure(root)
    config_paths = {(root / relative).resolve() for relative in B1_RUNTIME_CONFIG_FILES}
    missing_configs = sorted(
        path.relative_to(root).as_posix() for path in config_paths if not path.is_file()
    )
    if missing_configs:
        raise B1ContractError(f"Missing B1 runtime configuration: {missing_configs}")
    all_relevant = python_paths | config_paths
    third_party_manifest = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(all_relevant)
        if path.relative_to(root).as_posix().startswith(f"{project_rel}third_party/")
    }
    artifact_manifest = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(all_relevant)
        if not path.relative_to(root).as_posix().startswith(f"{project_rel}third_party/")
    }
    runtime_manifest = {**artifact_manifest, **third_party_manifest}
    teacher_checkpoint = _file_binding(
        root,
        "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt",
    )
    teacher_config = _file_binding(
        root,
        "SPINT-main/logs/train/runs/2026-07-07-16-05-16/.hydra/config.yaml",
    )
    return {
        "binding_scope_schema": "b1_recursive_runtime_closure_v2",
        "python_entrypoints": list(B1_RUNTIME_PYTHON_ENTRYPOINTS),
        "hydra_config_closure": list(B1_RUNTIME_CONFIG_FILES),
        "files": artifact_manifest,
        "artifact_source_manifest_sha256": sha256_payload({
            _relative_under_project(path): digest for path, digest in artifact_manifest.items()
        }),
        "runtime_dependency_manifest": runtime_manifest,
        "runtime_dependency_manifest_sha256": sha256_payload(runtime_manifest),
        "teacher": {
            "checkpoint": teacher_checkpoint,
            "teacher_hydra_config": teacher_config,
            "checkpoint_deserialized": False,
            "binding_method": "sha256_bytes_only",
        },
    }
