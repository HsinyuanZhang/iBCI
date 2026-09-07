"""Phase-C v4 fail-closed contracts for native-M2 post-33 confirmation.

The module never imports a scorer and never interprets an endpoint value.  It
seals per-cell artifacts through opaque score commitments, exact filesystem
sets, canonical containment, content hashes, and write-once state changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
PHASE_ID = "PHASE_C_V4"
ARMS = ("spint", "t4")
SEEDS = (42, 43, 44)
FOLDS = {
    0: "ses-2020-10-19-Run1",
    1: "ses-2020-10-19-Run2",
    2: "ses-2020-10-20-Run1",
    3: "ses-2020-10-20-Run2",
    4: "ses-2020-10-27-Run1",
    5: "ses-2020-10-27-Run2",
    6: "ses-2020-10-28-Run1",
}
EXPECTED_CELLS = len(ARMS) * len(FOLDS) * len(SEEDS)
EXPECTED_PAIRS = len(FOLDS) * len(SEEDS)
EXPECTED_DECODER_TENSORS = 31
EPOCHS = {"spint": tuple(range(35)), "t4": tuple(range(12))}
SELECTOR_SCHEMAS = {
    "spint": "m2_post33_source_selector_records_v4",
    "t4": "m2_post33_t4_source_selector_records_v4",
}
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_bytes_exclusive(path: str | Path, data: bytes) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        cursor = 0
        while cursor < len(data):
            cursor += os.write(fd, data[cursor:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return target


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return write_bytes_exclusive(path, _json_bytes(payload))


def copy_file_exclusive(source: str | Path, destination: str | Path) -> Path:
    source_path = require_canonical_regular_file(source)
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(destination_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                cursor = 0
                while cursor < len(chunk):
                    cursor += os.write(fd, chunk[cursor:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return destination_path


def file_metadata(path: str | Path) -> dict[str, Any]:
    canonical = require_canonical_regular_file(path)
    return {
        "canonical_path": str(canonical),
        "size_bytes": canonical.stat().st_size,
        "sha256": sha256_file(canonical),
    }


@dataclass(frozen=True)
class SelectedCheckpointSnapshot:
    """A private, byte-bound evaluator restore copy of the selected source ckpt.

    The source checkpoint remains in the owned run tree so its later seal can
    name the canonical training artifact.  Evaluation never restores that
    mutable pathname directly: it restores only the pinned
    ``/proc/self/fd/<restore_fd>`` view of the snapshot inode.  The inode
    identity is retained as an additional replacement guard for the named
    temporary file; the source is always checked against its selector metadata
    by file descriptor rather than by a check-then-open pathname sequence.
    """

    canonical_selected_checkpoint: Path
    snapshot_path: Path
    size_bytes: int
    sha256: str
    snapshot_device: int
    snapshot_inode: int
    restore_fd: int

    @property
    def trainer_checkpoint_path(self) -> Path:
        """The non-replaceable Linux FD path that Trainer is allowed to restore."""
        return Path("/proc/self/fd") / str(self.restore_fd)


@dataclass(frozen=True)
class DeploymentConstantsSnapshot:
    """Immutable selector-bound deployment constants for a test datamodule."""

    canonical_deployment_constants: Path
    payload_bytes: bytes
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CellKey:
    protocol_id: str
    arm: str
    fold: int
    seed: int

    def __post_init__(self) -> None:
        if self.protocol_id != PROTOCOL_ID:
            raise ValueError("protocol_id mismatch")
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if isinstance(self.fold, bool) or self.fold not in FOLDS:
            raise ValueError("fold must be an integer in [0, 6]")
        if isinstance(self.seed, bool) or self.seed not in SEEDS:
            raise ValueError(f"seed must be one of {SEEDS}")

    @property
    def outer_session(self) -> str:
        return FOLDS[self.fold]

    @property
    def source_sessions(self) -> tuple[str, ...]:
        return tuple(session for fold, session in FOLDS.items() if fold != self.fold)

    @property
    def relative_cell(self) -> Path:
        return Path("cells") / f"arm-{self.arm}" / f"fold-{self.fold}" / f"seed-{self.seed}"

    def identity(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "phase_id": PHASE_ID,
            "arm": self.arm,
            "fold": self.fold,
            "seed": self.seed,
            "outer_session": self.outer_session,
            "source_sessions": list(self.source_sessions),
        }


def matrix_keys() -> tuple[CellKey, ...]:
    return tuple(
        CellKey(PROTOCOL_ID, arm, fold, seed)
        for arm in ARMS
        for fold in FOLDS
        for seed in SEEDS
    )


def _require_non_symlink_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a non-symlink directory: {path}")
    return path


def validate_stage_a_cell_directory_set(root: str | Path) -> None:
    """Require the exact fourteen seed-42 cell roots before any score opening.

    We intentionally inspect the three structural levels only. Each expected
    seed-42 directory has its own run/control/sealed tree, which is validated
    by verify_cell_exact; scanning below that level here would incorrectly
    confuse normal cell artifacts with foreign cell identities.
    """
    phase = phase_root(root).resolve(strict=True)
    cells = _require_non_symlink_directory(phase / "cells", label="Stage-A cells root")
    expected_arms = {f"arm-{arm}" for arm in ARMS}
    observed_arms = {entry.name for entry in cells.iterdir()}
    if observed_arms != expected_arms:
        raise ValueError("Stage-A arm directory exact set mismatch")
    for arm in ARMS:
        arm_dir = _require_non_symlink_directory(cells / f"arm-{arm}", label="Stage-A arm")
        expected_folds = {f"fold-{fold}" for fold in FOLDS}
        observed_folds = {entry.name for entry in arm_dir.iterdir()}
        if observed_folds != expected_folds:
            raise ValueError("Stage-A fold directory exact set mismatch")
        for fold in FOLDS:
            fold_dir = _require_non_symlink_directory(
                arm_dir / f"fold-{fold}", label="Stage-A fold"
            )
            expected_seeds = {"seed-42"}
            observed_seeds = {entry.name for entry in fold_dir.iterdir()}
            if observed_seeds != expected_seeds:
                raise ValueError("Stage-A seed directory exact set mismatch")
            _require_non_symlink_directory(
                fold_dir / "seed-42", label="Stage-A seed cell"
            )


def validate_portable_transfer_manifest(
    manifest_path: str | Path,
    *,
    workspace_root: str | Path,
    data_root: str | Path,
    cell_root: str | Path,
) -> Mapping[str, Any]:
    path = require_canonical_regular_file(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "m2_post33_phase_c_portable_transfer_manifest_v4":
        raise ValueError("portable transfer manifest schema mismatch")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("phase_id") != PHASE_ID:
        raise ValueError("portable transfer protocol/phase mismatch")
    expected_paths = {
        "workspace_root": str(Path(workspace_root).resolve(strict=True)),
        "data_root": str(Path(data_root).resolve(strict=True)),
        "absolute_cell_root": str(Path(cell_root).resolve()),
    }
    for field, expected in expected_paths.items():
        if manifest.get(field) != expected:
            raise ValueError(f"portable transfer {field} mismatch")
    closures = manifest.get("hash_closures")
    if not isinstance(closures, list) or len(closures) != 3:
        raise ValueError("portable transfer needs exact program, deep-audit, and data-audit closures")
    if {closure.get("role") for closure in closures if isinstance(closure, Mapping)} != {
        "phase_c_program_receipt", "deep_source_audit_receipt", "phase_a_data_audit"
    }:
        raise ValueError("portable transfer closure roles mismatch")
    for closure in closures:
        if not isinstance(closure, Mapping) or set(closure) != {
            "role", "canonical_path", "size_bytes", "sha256"
        }:
            raise ValueError("portable transfer closure exact keys mismatch")
        file = require_canonical_regular_file(closure.get("canonical_path", ""))
        if file.stat().st_size != closure.get("size_bytes") or sha256_file(file) != closure.get("sha256"):
            raise ValueError("portable transfer hash closure drift")
    closure_by_role = {str(closure["role"]): closure for closure in closures}
    from sua_exploration.mc_maze.m2_native_post33_cost_v4 import (
        validate_deep_source_audit_receipt,
    )
    from sua_exploration.mc_maze.m2_native_post33_program_v4 import (
        validate_phase_c_program_receipt,
    )
    program_path = closure_by_role["phase_c_program_receipt"]["canonical_path"]
    program = validate_phase_c_program_receipt(program_path)
    deep_metadata = closure_by_role["deep_source_audit_receipt"]
    if program.get("deep_source_audit_receipt") != {
        key: deep_metadata[key] for key in ("canonical_path", "size_bytes", "sha256")
    }:
        raise ValueError("portable transfer program/deep-audit closure mismatch")
    deep_path = deep_metadata["canonical_path"]
    deep = json.loads(require_canonical_regular_file(deep_path).read_text(encoding="utf-8"))
    audit_metadata = deep.get("source_batch_audit") if isinstance(deep, Mapping) else None
    if not isinstance(audit_metadata, Mapping):
        raise ValueError("portable transfer deep-audit source-batch binding missing")
    validate_deep_source_audit_receipt(
        deep, source_batch_audit_path=audit_metadata.get("canonical_path", ""), deep_verify=False
    )
    if manifest.get("same_absolute_paths_required_on_all_hosts") is not True:
        raise ValueError("portable transfer must require identical absolute paths")
    return manifest


def validate_shard_manifest(
    shard_path: str | Path,
    *,
    portable_manifest_path: str | Path,
    cell_root: str | Path,
) -> Mapping[str, Any]:
    path = require_canonical_regular_file(shard_path)
    shard = json.loads(path.read_text(encoding="utf-8"))
    if shard.get("schema") != "m2_post33_phase_c_shard_manifest_v4":
        raise ValueError("shard manifest schema mismatch")
    if shard.get("protocol_id") != PROTOCOL_ID or shard.get("phase_id") != PHASE_ID:
        raise ValueError("shard protocol/phase mismatch")
    if shard.get("arms_in_order") != ["spint", "t4"] or shard.get("paired_same_host_required") is not True:
        raise ValueError("shard must run paired SPINT then T4 on the same host")
    if shard.get("absolute_cell_root") != str(Path(cell_root).resolve()):
        raise ValueError("shard absolute cell root mismatch")
    folds = shard.get("fold_allowlist")
    seeds = shard.get("seed_allowlist")
    if (
        not isinstance(folds, list)
        or not folds
        or len(set(folds)) != len(folds)
        or any(isinstance(fold, bool) or fold not in FOLDS for fold in folds)
    ):
        raise ValueError("shard fold allowlist invalid")
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or seed not in SEEDS for seed in seeds)
    ):
        raise ValueError("shard seed allowlist invalid")
    gpu_id = shard.get("gpu_id")
    if not isinstance(gpu_id, str) or not gpu_id or any(char.isspace() for char in gpu_id):
        raise ValueError("shard GPU id invalid")
    host_id = shard.get("host_id")
    if not isinstance(host_id, str) or not host_id or any(char.isspace() for char in host_id):
        raise ValueError("shard host id invalid")
    portable = require_canonical_regular_file(portable_manifest_path)
    if shard.get("portable_transfer_manifest_sha256") != sha256_file(portable):
        raise ValueError("shard portable-transfer manifest substitution")
    return shard


def validate_gpu_authorization(
    authorization_path: str | Path,
    *,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
) -> Mapping[str, Any]:
    del authorization_path, phase_c_program_receipt_path, portable_manifest_path
    raise PermissionError(
        "unsigned JSON authorization is forbidden; use the detached-Ed25519 Phase-C verifier"
    )


def phase_root(root: str | Path) -> Path:
    return Path(root).resolve() / PROTOCOL_ID / PHASE_ID


_SYNTHETIC_TEST_SENTINEL = ".m2_post33_phase_c_v4_synthetic_fixture.json"
_SYNTHETIC_TEST_ROOTS: set[Path] = set()
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _validate_synthetic_fixture_root_location(root: str | Path) -> Path:
    """Return a disposable fixture root, rejecting every workspace location.

    The relaxed synthetic paths are deliberately meaningful only below the
    system temporary directory.  Checking both conditions is intentional:
    test runners can be configured to put temporary directories below a
    workspace, and a workspace can in turn be placed below ``/tmp``.  Neither
    layout may turn a production-like Phase-C root into a synthetic fixture.
    """
    root_path = Path(root).resolve()
    try:
        root_path.relative_to(_WORKSPACE_ROOT)
    except ValueError:
        pass
    else:
        raise PermissionError("synthetic fixture root must not be inside the workspace")
    temporary_root = Path(tempfile.gettempdir()).resolve()
    try:
        root_path.relative_to(temporary_root)
    except ValueError as exc:
        raise PermissionError(
            "synthetic fixture root must be below the system temp directory"
        ) from exc
    return root_path


def _activate_synthetic_test_fixture(root: str | Path) -> None:
    """Register one temporary test fixture root for private relaxed validators.

    This intentionally has no production caller. The tests-only support module
    creates the O_EXCL sentinel first; only roots below the system temp
    directory can be registered. A private function accepting
    allow_synthetic=True therefore cannot be used to write a production-like
    worktree root unless that root has first been explicitly registered as a
    disposable test fixture.
    """
    root_path = _validate_synthetic_fixture_root_location(root)
    sentinel = require_canonical_regular_file(root_path / _SYNTHETIC_TEST_SENTINEL)
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    if payload != {
        "schema": "m2_post33_phase_c_v4_synthetic_fixture_sentinel",
        "absolute_fixture_root": str(root_path),
    }:
        raise PermissionError("synthetic fixture sentinel substitution")
    _SYNTHETIC_TEST_ROOTS.add(root_path)


def _require_synthetic_test_fixture(root: str | Path) -> None:
    root_path = _validate_synthetic_fixture_root_location(root)
    if root_path not in _SYNTHETIC_TEST_ROOTS:
        raise PermissionError(
            "synthetic relaxed path requires a tests-only registered fixture root"
        )
    sentinel = require_canonical_regular_file(root_path / _SYNTHETIC_TEST_SENTINEL)
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    if payload != {
        "schema": "m2_post33_phase_c_v4_synthetic_fixture_sentinel",
        "absolute_fixture_root": str(root_path),
    }:
        raise PermissionError("synthetic fixture sentinel changed after registration")


def cell_paths(root: str | Path, key: CellKey) -> dict[str, Path]:
    cell = phase_root(root) / key.relative_cell
    control = cell / "control"
    run = cell / "run"
    sealed = cell / "sealed"
    return {
        "phase_root": phase_root(root),
        "cell_dir": cell,
        "control": control,
        "owner": control / "ownership.json",
        "started": control / "status.started.json",
        "completed": control / "status.completed.json",
        "failed": control / "status.failed.json",
        "run": run,
        "hydra": run / "hydra",
        "checkpoints": run / "checkpoints",
        "selector_records": run / "selector_records.json",
        "resolved_config": run / "resolved_config.yaml",
        "deployment_constants_run": run / "deployment_constants.json",
        "decoder_lifecycle_stages": run / "decoder_lifecycle_stages",
        "decoder_lifecycle_evidence_run": run / "decoder_lifecycle_evidence.json",
        "outer_runtime_evidence_run": run / "outer_runtime_evidence.json",
        "source_cost_evidence_run": run / "source_cost_evidence.json",
        "deployment_cost_evidence_run": run / "deployment_cost_evidence.json",
        "execution_capability_evidence_run": run / "execution_capability_evidence.json",
        "opaque_payload_run": run / "endpoint_payload.from_evaluator.json",
        "score_commitment_run": run / "score_commitment.from_evaluator.json",
        "secondary_artifact_root": run / "secondary_artifacts",
        "secondary_artifacts": run / "secondary_artifacts" / "canonical",
        "sealed": sealed,
        "selected_checkpoint": sealed / "selected.ckpt",
        "deployment_constants": sealed / "deployment_constants.json",
        "completion_receipt": sealed / "completion_receipt.json",
        "cost_binding": sealed / "cost_binding.json",
        "score_commitment": sealed / "score_commitment.json",
        "opaque_payload": sealed / "opaque_endpoint_payload.json",
        "run_manifest": sealed / "run_manifest.json",
        "decoder_lifecycle": sealed / "decoder_lifecycle.json",
        "paired_spint_completion": sealed / "paired_spint_completion_receipt.json",
        "outer_runtime_evidence": sealed / "outer_runtime_evidence.json",
        "source_cost_evidence": sealed / "source_cost_evidence.json",
        "deployment_cost_evidence": sealed / "deployment_cost_evidence.json",
        "result": sealed / "result.json",
    }


_TRAINING_MODEL_TARGETS = {
    "spint": "src.models.falcon_post33_confirm_v4_module.M2Post33ExactOuterFalconLitModuleV4",
    "t4": "src.models.streaming_post33_exact_t4_v4_module.M2Post33ExactPairedT4LitModuleV4",
}
_TRAINING_DATA_TARGETS = {
    "spint": "src.data.falcon_post33_confirm_v4_datamodule.M2Post33ConfirmSPINTDataModuleV4",
    "t4": "src.data.falcon_post33_confirm_v4_datamodule.M2Post33ConfirmT4DataModuleV4",
}
_TRAINING_CALLBACK_TARGETS = {
    "spint": {
        "override_epoch_step": "src.callbacks.override_epoch_step.OverrideEpochStepCallback",
        "model_summary": "lightning.pytorch.callbacks.RichModelSummary",
        "source_selector_v4": "src.callbacks.post33_source_selector_v4.Post33SourceSelectorV4",
        "source_cost_runtime_v4": "src.callbacks.post33_source_cost_runtime_v4.Post33SourceCostRuntimeV4",
    },
    "t4": {
        "source_selector_v4": "src.callbacks.post33_t4_source_selector_v4.Post33T4SourceSelectorV4",
        "decoder_lifecycle_train_v4": "src.callbacks.decoder_lifecycle_phase_c_v4.DecoderLifecycleTrainStagesV4",
        "source_cost_runtime_v4": "src.callbacks.post33_source_cost_runtime_v4.Post33SourceCostRuntimeV4",
    },
}


def _training_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase-C training {label} must be a mapping")
    return value


def _training_exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"Phase-C training {label} exact key set mismatch")


def _training_exact_path(value: Any, expected: Path, label: str) -> None:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Phase-C training {label} must be a path string")
    raw = Path(value)
    if not raw.is_absolute() or raw.is_symlink() or str(raw) != str(expected):
        raise ValueError(f"Phase-C training {label} path substitution")


def _training_workspace_path(value: Any, expected: Path, label: str) -> None:
    """Compare a workspace-relative config path without accepting aliases."""
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Phase-C training {label} must be a path string")
    raw = Path(value)
    lexical = Path(os.path.abspath(raw))
    if raw.is_symlink():
        raise ValueError(f"Phase-C training {label} symlink substitution")
    canonical = lexical.resolve()
    expected_canonical = expected.resolve()
    if canonical != expected_canonical or str(lexical) != str(canonical):
        raise ValueError(f"Phase-C training {label} path substitution")


def _training_exact_callback(
    callbacks: Mapping[str, Any],
    *,
    name: str,
    target: str,
    fields: Mapping[str, Any],
) -> None:
    callback = _training_mapping(callbacks.get(name), f"callback {name}")
    _training_exact_keys(callback, {"_target_", *fields}, f"callback {name}")
    if callback.get("_target_") != target:
        raise ValueError(f"Phase-C training callback {name} target substitution")
    for field, expected in fields.items():
        if callback.get(field) != expected:
            raise ValueError(f"Phase-C training callback {name} {field} substitution")


_PHASE_C_TRAINING_WRAPPER_ARGUMENT_KEYS = {
    "spint": frozenset(
        {
            "experiment",
            "data.loso_fold",
            "seed",
            "cell_owner_token",
            "cell_paths.cell_dir",
            "cell_paths.owner",
            "cell_paths.hydra",
            "cell_paths.selector_records",
            "cell_paths.checkpoints",
            "cell_paths.resolved_config",
            "cell_paths.deployment_constants",
            "cell_paths.source_cost_evidence",
            "cell_paths.cost_supplement",
        }
    ),
    "t4": frozenset(
        {
            "experiment",
            "data.loso_fold",
            "seed",
            "cell_owner_token",
            "cell_paths.cell_dir",
            "cell_paths.owner",
            "cell_paths.hydra",
            "cell_paths.selector_records",
            "cell_paths.checkpoints",
            "cell_paths.resolved_config",
            "cell_paths.deployment_constants",
            "cell_paths.source_cost_evidence",
            "cell_paths.cost_supplement",
            "cell_paths.decoder_lifecycle_stages",
            "cell_paths.secondary_artifact_root",
            "cell_paths.secondary_artifacts",
            "model.paired_spint_completion_receipt",
        }
    ),
}


def validate_phase_c_training_wrapper_argv(argv: Sequence[str], *, arm: str) -> dict[str, str]:
    """Fail closed before ``@hydra.main`` can construct any Hydra callback.

    The production cell pipeline supplies one exact set of ordinary overrides.
    Accepting arbitrary Hydra control-plane keys here would let Hydra build
    plugin callbacks before the later capability and semantic-plan guards run.
    This raw-``argv`` gate intentionally runs immediately before ``main()``.
    """
    if arm not in _PHASE_C_TRAINING_WRAPPER_ARGUMENT_KEYS:
        raise ValueError("Phase-C raw argv arm is invalid")
    expected = _PHASE_C_TRAINING_WRAPPER_ARGUMENT_KEYS[arm]
    expected_experiment = f"m2_native_post33_confirm_v4_{arm}"
    observed: dict[str, str] = {}
    for argument in argv:
        if argument in {"-m", "--multirun"} or argument.startswith(
            ("-m=", "--multirun=")
        ):
            raise ValueError("Phase-C raw argv forbids Hydra multirun")
        if argument.startswith("-"):
            raise ValueError("Phase-C raw argv forbids Hydra command-line controls")
        if "=" not in argument:
            raise ValueError("Phase-C raw argv requires exact key=value overrides")
        name, value = argument.split("=", 1)
        normalized = name.lstrip("+~")
        if normalized == "hydra" or normalized.startswith("hydra."):
            raise ValueError("Phase-C raw argv forbids Hydra control-plane overrides")
        if name not in expected or not value or name in observed:
            raise ValueError("Phase-C raw argv exact override set mismatch")
        observed[name] = value
    if set(observed) != set(expected):
        raise ValueError("Phase-C raw argv exact override set mismatch")
    if observed["experiment"] != expected_experiment:
        raise ValueError("Phase-C raw argv experiment substitution")
    return observed


def require_phase_c_training_wrapper_pre_hydra_gate(
    argv: Sequence[str], *, arm: str
) -> dict[str, str]:
    """Validate raw launcher identity/capability before Hydra can create anything.

    This gate intentionally obtains the signed execution capability before the
    decorated function is invoked.  It prevents an unauthenticated direct
    wrapper invocation from letting Hydra create an output directory or build
    a callback before the ordinary in-main guard sees the config.
    """
    overrides = validate_phase_c_training_wrapper_argv(argv, arm=arm)
    raw_cell = Path(overrides["cell_paths.cell_dir"])
    environment_cell_value = os.environ.get("M2_POST33_PHASE_C_CELL_DIR")
    if not environment_cell_value:
        raise PermissionError("missing required Phase-C cell environment: M2_POST33_PHASE_C_CELL_DIR")
    environment_cell_raw = Path(environment_cell_value)
    cell = require_canonical_directory(raw_cell)
    environment_cell = require_canonical_directory(environment_cell_raw)
    if (
        str(raw_cell) != str(cell)
        or str(environment_cell_raw) != str(environment_cell)
        or cell != environment_cell
    ):
        raise PermissionError("Phase-C raw argv cell directory differs from capability environment")
    try:
        root = cell.parents[5]
    except IndexError as exc:
        raise PermissionError("Phase-C raw argv cell directory is outside the canonical layout") from exc
    try:
        key = CellKey(
            PROTOCOL_ID,
            arm,
            int(overrides["data.loso_fold"]),
            int(overrides["seed"]),
        )
    except (TypeError, ValueError) as exc:
        raise PermissionError("Phase-C raw argv fold/seed is invalid") from exc
    if cell_paths(root, key)["cell_dir"] != cell:
        raise PermissionError("Phase-C raw argv cell identity/layout substitution")
    from sua_exploration.mc_maze.m2_native_post33_authorization_v4 import (
        require_cell_execution_capability_from_environment,
    )

    require_cell_execution_capability_from_environment(root=root, key=key)
    return overrides


def validate_phase_c_training_plan(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    key: CellKey,
    authorized_cost_supplement: str | Path,
    project_root: str | Path,
) -> None:
    """Reject every runtime override outside the fixed Phase-C training plan.

    The top-level pipeline validates an authorization capability, but Hydra can
    still accept command-line overrides after that capability has been issued.
    This preflight freezes the scientific/provenance-critical training surface
    before either historical trainer can instantiate data, models, callbacks,
    loggers, or a ``Trainer``.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Phase-C resolved training config must be a mapping")
    paths = cell_paths(root, key)
    supplement = require_canonical_regular_file(authorized_cost_supplement)
    project = require_canonical_directory(project_root)

    for field, expected in {
        "protocol_id": key.protocol_id,
        "phase_id": PHASE_ID,
        "arm": key.arm,
        "seed": key.seed,
    }.items():
        if config.get(field) != expected:
            raise ValueError(f"Phase-C training {field} substitution")
    if config.get("train") is not True or config.get("test") is not False:
        raise ValueError("Phase-C training must be fit-only")
    if config.get("ckpt_path") is not None:
        raise ValueError("Phase-C training forbids ckpt_path/resume")
    if config.get("logger") is not False:
        raise ValueError("Phase-C training requires logger=false")
    if config.get("optimized_metric") != "val_source/r2_equal_session_mean":
        raise ValueError("Phase-C training optimized metric substitution")
    extras = _training_mapping(config.get("extras"), "extras")
    expected_extras = {
        "ignore_warnings": False,
        "enforce_tags": False,
        "print_config": False,
    }
    _training_exact_keys(extras, set(expected_extras), "extras")
    if dict(extras) != expected_extras:
        raise ValueError("Phase-C training extras substitution")

    trainer = _training_mapping(config.get("trainer"), "trainer")
    expected_trainer = {
        "_target_": "lightning.pytorch.trainer.Trainer",
        "default_root_dir": str(paths["run"]),
        "min_epochs": 1,
        "max_epochs": len(EPOCHS[key.arm]),
        "gradient_clip_val": 0.0,
        "accelerator": "gpu",
        "devices": 1,
        "check_val_every_n_epoch": 1,
        "deterministic": False,
        "log_every_n_steps": 1 if key.arm == "spint" else 10,
        "precision": "32-true",
        "num_sanity_val_steps": 2,
        "enable_checkpointing": False,
    }
    _training_exact_keys(trainer, set(expected_trainer), "trainer")
    for field, expected in expected_trainer.items():
        if trainer.get(field) != expected:
            raise ValueError(f"Phase-C training trainer.{field} substitution")

    cell_config = _training_mapping(config.get("cell_paths"), "cell_paths")
    expected_cell_paths: dict[str, Path] = {
        "cell_dir": paths["cell_dir"],
        "owner": paths["owner"],
        "hydra": paths["hydra"],
        "selector_records": paths["selector_records"],
        "checkpoints": paths["checkpoints"],
        "resolved_config": paths["resolved_config"],
        "deployment_constants": paths["deployment_constants_run"],
        "source_cost_evidence": paths["source_cost_evidence_run"],
        "cost_supplement": supplement,
    }
    if key.arm == "t4":
        expected_cell_paths.update(
            {
                "decoder_lifecycle_stages": paths["decoder_lifecycle_stages"],
                "secondary_artifact_root": paths["secondary_artifact_root"],
                "secondary_artifacts": paths["secondary_artifacts"],
            }
        )
    _training_exact_keys(cell_config, set(expected_cell_paths), "cell_paths")
    for field, expected in expected_cell_paths.items():
        _training_exact_path(cell_config.get(field), expected, f"cell_paths.{field}")

    paths_config = _training_mapping(config.get("paths"), "paths")
    expected_path_keys = {
        "root_dir", "data_dir", "log_dir", "output_dir", "work_dir",
    }
    if key.arm == "t4":
        expected_path_keys.update(
            {"artifact_dir", "teacher_ckpt_path", "m1_teacher_ckpt_path"}
        )
    _training_exact_keys(paths_config, expected_path_keys, "paths")
    _training_workspace_path(paths_config.get("root_dir"), project, "paths.root_dir")
    _training_workspace_path(paths_config.get("log_dir"), project / "logs", "paths.log_dir")
    _training_workspace_path(paths_config.get("work_dir"), project, "paths.work_dir")
    _training_exact_path(paths_config.get("output_dir"), paths["run"], "paths.output_dir")
    data_root = project / "data"
    if key.arm == "t4":
        data_root = project.parent / "SPINT-main" / "data"
    _training_workspace_path(paths_config.get("data_dir"), data_root, "paths.data_dir")
    if key.arm == "t4":
        _training_exact_path(
            paths_config.get("artifact_dir"),
            paths["secondary_artifact_root"],
            "paths.artifact_dir",
        )
        _training_workspace_path(
            paths_config.get("teacher_ckpt_path"),
            data_root.parent / "logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt",
            "paths.teacher_ckpt_path",
        )
        _training_workspace_path(
            paths_config.get("m1_teacher_ckpt_path"),
            data_root.parent / "logs/train/runs/2026-07-21-19-11-01/checkpoints/best_ckpt/epoch_019.ckpt",
            "paths.m1_teacher_ckpt_path",
        )

    data = _training_mapping(config.get("data"), "data")
    data_expected: dict[str, Any] = {
        "_target_": _TRAINING_DATA_TARGETS[key.arm],
        "deployment_constants_path": str(paths["deployment_constants_run"]),
        "seed": key.seed,
        "task": "m2",
        "validation_protocol": "loso",
        "loso_fold": key.fold,
        "calibration_n_trials": 33,
        "heldin_query_start_trial": 33,
        "heldin_query_end_trial": None,
        "query_start_trial": 0,
        "random_calibration": False,
        "include_heldout_in_fit": False,
        "include_heldout_in_test": False,
        "batch_size": 32,
        "window_size": 50,
        "smooth_calibration": False,
        "max_trial_length": 100,
        "standardize_covariates": False,
        "use_intertrials": True,
        "use_calib_intertrials": False,
        "trial_feature_type": "raw",
        "remove_still_times": False,
        "remove_calib_still_times": False,
        "use_calib_active_segments": False,
        "calib_n_active_segments": 1,
        "interpolate_trials": True,
        "interpolate_trials_kind": "cubic",
        "pad_value": -1.0,
        "num_workers": 0,
        "pin_memory": False,
    }
    if key.arm == "t4":
        data_expected.update(
            {
                "sampler_seed": key.seed,
                "balance_session_batches": False,
                "reshuffle_train_sampler_each_epoch": False,
                "side_feature_group": "t4",
                "side_feature_shuffle_seed": key.seed,
            }
        )
    _training_exact_keys(data, {*data_expected, "data_dir"}, "data")
    for field, expected in data_expected.items():
        if data.get(field) != expected:
            raise ValueError(f"Phase-C training data.{field} substitution")
    _training_workspace_path(data.get("data_dir"), data_root / "000953", "data.data_dir")

    model = _training_mapping(config.get("model"), "model")
    if key.arm == "spint":
        expected_net = {
            "_target_": "src.models.components.spint.SpintModel",
            "model_dim": 512,
            "num_covariates": 2,
            "window_size": 50,
            "num_heads": 64,
            "num_layers": 1,
            "num_id_layers": 3,
            "use_learnable_id": True,
            "learnable_id_type": "mlp",
            "learnable_rep": True,
            "dropout_rate": 0.0,
            "dynamic_dropout": True,
            "dynamic_dropout_low": 0.0,
            "dynamic_dropout_high": 1.0,
            "tf_drop_rate": 0.1,
            "readin_layer_type": "mlp",
        }
        expected_model = {
            "_target_": _TRAINING_MODEL_TARGETS["spint"],
            "task": "m2",
            "loso_fold": key.fold,
            "decode_last_timestep_only": True,
            "predict_scaled_behavior": True,
            "behavior_scaling_factor": 5.0,
            "scheduler": None,
            "scheduler_monitor": "val_source/r2_equal_session_mean",
            "compile": False,
            "clean_teacher": False,
        }
        _training_exact_keys(model, {*expected_model, "optimizer", "net"}, "model")
        for field, expected in expected_model.items():
            if model.get(field) != expected:
                raise ValueError(f"Phase-C training model.{field} substitution")
        optimizer = _training_mapping(model.get("optimizer"), "model.optimizer")
        expected_optimizer = {
            "_target_": "torch.optim.Adam", "_partial_": True,
            "lr": 5e-5, "weight_decay": 0.0,
        }
        _training_exact_keys(optimizer, set(expected_optimizer), "model.optimizer")
        if dict(optimizer) != expected_optimizer:
            raise ValueError("Phase-C training SPINT optimizer substitution")
        net = _training_mapping(model.get("net"), "model.net")
        _training_exact_keys(net, set(expected_net), "model.net")
        if dict(net) != expected_net:
            raise ValueError("Phase-C training SPINT model width/depth substitution")
    else:
        paired = cell_paths(root, CellKey(PROTOCOL_ID, "spint", key.fold, key.seed))[
            "completion_receipt"
        ]
        expected_model = {
            "_target_": _TRAINING_MODEL_TARGETS["t4"],
            "task": "m2",
            "variant": "B3S",
            "paired_spint_completion_receipt": str(paired),
            "phase_c_t4_owner_path": str(paths["owner"]),
            "phase_c_t4_owner_token": config.get("cell_owner_token"),
            "loso_fold": key.fold,
            "seed": key.seed,
            "teacher_receipt_path": None,
            "require_clean_teacher_receipt": False,
            "window_size": 50,
            "trial_length": 100,
            "pad_value": -1.0,
            "freeze_decoder": True,
            "encoder_warmstart_path": None,
            "freeze_encoder_base": False,
            "tune_encoder_fusion": False,
            "fusion_mean_lr_scale": 1.0,
            "loss_mode": "task_plus_y_plus_E",
            "lambda_y": 1.0,
            "lambda_E": 0.1,
            "decode_last_timestep_only": True,
            "predict_scaled_behavior": True,
            "behavior_scaling_factor": 5.0,
            "id_hidden_dim": 128,
            "hidden_dim": 64,
            "num_emas": 4,
            "num_filters": 4,
            "kernel_size": 5,
            "learnable_ema_alpha": False,
            "side_dim": 4,
            "electrode_embed_dim": 0,
            "num_electrodes": 0,
            "scheduler": None,
            "compile": False,
            "neuron_dropout_mode": "none",
            "neuron_dropout_p_low": 0.0,
            "neuron_dropout_p_high": 0.3,
            "neuron_dropout_block_size": 4,
            "neuron_dropout_warmup_epochs": 10,
            "support_prediction_consistency_weight": 0.0,
        }
        _training_exact_keys(model, {*expected_model, "optimizer"}, "model")
        for field, expected in expected_model.items():
            if model.get(field) != expected:
                raise ValueError(f"Phase-C training model.{field} substitution")
        optimizer = _training_mapping(model.get("optimizer"), "model.optimizer")
        expected_optimizer = {
            "_target_": "torch.optim.Adam", "_partial_": True,
            "lr": 1e-4, "weight_decay": 0.0,
        }
        _training_exact_keys(optimizer, set(expected_optimizer), "model.optimizer")
        if dict(optimizer) != expected_optimizer:
            raise ValueError("Phase-C training T4 optimizer substitution")

    callbacks = _training_mapping(config.get("callbacks"), "callbacks")
    expected_callback_targets = _TRAINING_CALLBACK_TARGETS[key.arm]
    _training_exact_keys(callbacks, set(expected_callback_targets), "callbacks")
    selector_fields = {
        "checkpoint_dir": str(paths["checkpoints"]),
        "selector_records_path": str(paths["selector_records"]),
        "deployment_constants_path": str(paths["deployment_constants_run"]),
        "cell_owner_path": str(paths["owner"]),
        "owner_token": config.get("cell_owner_token"),
        "fold": key.fold,
        "seed": key.seed,
    }
    _training_exact_callback(
        callbacks,
        name="source_selector_v4",
        target=expected_callback_targets["source_selector_v4"],
        fields=selector_fields,
    )
    source_cost_fields = {
        "output_path": str(paths["source_cost_evidence_run"]),
        "resolved_config_path": str(paths["resolved_config"]),
        "cell_owner_path": str(paths["owner"]),
        "owner_token": config.get("cell_owner_token"),
        "cost_supplement_path": str(supplement),
        "arm": key.arm,
        "fold": key.fold,
        "seed": key.seed,
    }
    _training_exact_callback(
        callbacks,
        name="source_cost_runtime_v4",
        target=expected_callback_targets["source_cost_runtime_v4"],
        fields=source_cost_fields,
    )
    if key.arm == "spint":
        _training_exact_callback(
            callbacks,
            name="override_epoch_step",
            target=expected_callback_targets["override_epoch_step"],
            fields={},
        )
        _training_exact_callback(
            callbacks,
            name="model_summary",
            target=expected_callback_targets["model_summary"],
            fields={"max_depth": 1},
        )
        if config.get("clean_teacher_mode") is not False:
            raise ValueError("Phase-C SPINT training control substitution")
    else:
        lifecycle_fields = {
            "stage_dir": str(paths["decoder_lifecycle_stages"]),
            "cell_owner_path": str(paths["owner"]),
            "owner_token": config.get("cell_owner_token"),
            "fold": key.fold,
            "seed": key.seed,
        }
        _training_exact_callback(
            callbacks,
            name="decoder_lifecycle_train_v4",
            target=expected_callback_targets["decoder_lifecycle_train_v4"],
            fields=lifecycle_fields,
        )
        paired = cell_paths(root, CellKey(PROTOCOL_ID, "spint", key.fold, key.seed))[
            "completion_receipt"
        ]
        _training_exact_path(
            model.get("paired_spint_completion_receipt"),
            paired,
            "model.paired_spint_completion_receipt",
        )
        _training_exact_path(
            model.get("phase_c_t4_owner_path"), paths["owner"], "model.phase_c_t4_owner_path"
        )
        if model.get("phase_c_t4_owner_token") != config.get("cell_owner_token"):
            raise ValueError("Phase-C training model owner-token substitution")
        if model.get("seed") != key.seed or model.get("variant") != "B3S":
            raise ValueError("Phase-C training T4 model substitution")
        if (
            config.get("no_early_stopping") is not True
            or config.get("require_baseline_validation") is not False
            or config.get("run_id") != "canonical"
        ):
            raise ValueError("Phase-C T4 training control substitution")


_SELECTED_CHECKPOINT_SNAPSHOT_PREFIX = ".m2-post33-phase-c-v4-selected-"
_SELECTED_CHECKPOINT_COPY_CHUNK_BYTES = 1 << 20


def _no_follow_open_flags(flags: int) -> int:
    """Return Linux/POSIX flags that make a file-descriptor boundary explicit."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Phase-C requires O_NOFOLLOW for selected checkpoint snapshots")
    result = flags | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        result |= os.O_CLOEXEC
    return result


def _open_regular_readonly_nofollow(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    """Open one final path component without following it, then prove regularity."""
    try:
        fd = os.open(str(path), _no_follow_open_flags(os.O_RDONLY))
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened without following a substituted path") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular file")
        return fd, info
    except BaseException:
        os.close(fd)
        raise


def _stream_fd_size_and_sha256(fd: int) -> tuple[int, str]:
    """Hash exactly the bytes read from an already-open regular-file descriptor."""
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size_bytes = 0
    while True:
        chunk = os.read(fd, _SELECTED_CHECKPOINT_COPY_CHUNK_BYTES)
        if not chunk:
            break
        size_bytes += len(chunk)
        digest.update(chunk)
    return size_bytes, digest.hexdigest()


def _write_all(fd: int, data: bytes) -> None:
    cursor = 0
    while cursor < len(data):
        cursor += os.write(fd, data[cursor:])


def _bound_file_metadata(
    metadata: Any, *, expected_path: Path, label: str
) -> tuple[int, str]:
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "canonical_path", "size_bytes", "sha256"
    }:
        raise ValueError(f"{label} metadata mismatch")
    size_bytes = metadata.get("size_bytes")
    digest = metadata.get("sha256")
    if metadata.get("canonical_path") != str(expected_path):
        raise ValueError(f"{label} canonical-path mismatch")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError(f"{label} size metadata is invalid")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label} SHA-256 metadata is invalid")
    return size_bytes, digest


def capture_deployment_constants_snapshot(
    source_file: str | Path,
    *,
    selector_metadata: Mapping[str, Any],
) -> DeploymentConstantsSnapshot:
    """Read selector-bound constants through one O_NOFOLLOW descriptor.

    The returned ``bytes`` object is the only constants representation the
    evaluator datamodule may consume during ``setup(test)``.  The source path
    is revalidated separately before endpoint/seal work, but never re-opened
    to determine the outer normalizer.
    """
    source = require_canonical_regular_file(source_file)
    size_bytes, digest = _bound_file_metadata(
        selector_metadata, expected_path=source, label="deployment constants"
    )
    fd, before = _open_regular_readonly_nofollow(
        source, label="deployment constants source"
    )
    try:
        if before.st_size != size_bytes:
            raise ValueError("deployment constants source changed before snapshot capture")
        chunks: list[bytes] = []
        observed_size = 0
        observed_digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, _SELECTED_CHECKPOINT_COPY_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
            observed_size += len(chunk)
            observed_digest.update(chunk)
        after = os.fstat(fd)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_size != size_bytes
            or observed_size != size_bytes
            or observed_digest.hexdigest() != digest
        ):
            raise ValueError("deployment constants source bytes changed during snapshot capture")
        return DeploymentConstantsSnapshot(
            canonical_deployment_constants=source,
            payload_bytes=b"".join(chunks),
            size_bytes=size_bytes,
            sha256=digest,
        )
    finally:
        os.close(fd)


def validate_deployment_constants_snapshot(snapshot: DeploymentConstantsSnapshot) -> None:
    """Prove the in-memory constants object still equals selector-bound bytes."""
    if not isinstance(snapshot, DeploymentConstantsSnapshot):
        raise TypeError("Phase-C deployment constants snapshot binding is invalid")
    if (
        len(snapshot.payload_bytes) != snapshot.size_bytes
        or hashlib.sha256(snapshot.payload_bytes).hexdigest() != snapshot.sha256
    ):
        raise ValueError("deployment constants snapshot bytes changed")


def validate_deployment_constants_origin(snapshot: DeploymentConstantsSnapshot) -> None:
    """Revalidate the canonical on-disk constants without consuming its bytes."""
    if not isinstance(snapshot, DeploymentConstantsSnapshot):
        raise TypeError("Phase-C deployment constants snapshot binding is invalid")
    fd, before = _open_regular_readonly_nofollow(
        snapshot.canonical_deployment_constants, label="deployment constants source"
    )
    try:
        _validate_fd_against_selected_checkpoint_binding(
            fd,
            before=before,
            size_bytes=snapshot.size_bytes,
            sha256=snapshot.sha256,
            label="deployment constants source",
        )
    finally:
        os.close(fd)


def _validate_fd_against_selected_checkpoint_binding(
    fd: int,
    *,
    before: os.stat_result,
    size_bytes: int,
    sha256: str,
    label: str,
    expected_identity: tuple[int, int] | None = None,
    require_private_snapshot_mode: bool = False,
) -> None:
    """Verify bytes, size, and (for snapshots) immutable private-file identity.

    ``fd`` is deliberately the one descriptor whose bytes are streamed: no
    pre-open ``Path.stat`` or ``Path.open`` result is trusted for this boundary.
    """
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if before.st_size != size_bytes:
        raise ValueError(f"{label} size changed")
    if expected_identity is not None and (
        before.st_dev,
        before.st_ino,
    ) != expected_identity:
        raise ValueError(f"{label} was replaced")
    if require_private_snapshot_mode:
        if (
            before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_nlink != 1
        ):
            raise ValueError(f"{label} is not a private read-only snapshot")
    actual_size, actual_sha256 = _stream_fd_size_and_sha256(fd)
    after = os.fstat(fd)
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_size != size_bytes
        or actual_size != size_bytes
        or actual_sha256 != sha256
    ):
        raise ValueError(f"{label} bytes changed")
    if expected_identity is not None and (after.st_dev, after.st_ino) != expected_identity:
        raise ValueError(f"{label} was replaced")
    if require_private_snapshot_mode and (
        after.st_uid != os.geteuid()
        or stat.S_IMODE(after.st_mode) != 0o400
        or after.st_nlink != 1
    ):
        raise ValueError(f"{label} is not a private read-only snapshot")


def _snapshot_temp_directory() -> Path:
    """Use only the host system's canonical temporary directory for restores."""
    return require_canonical_directory(Path(tempfile.gettempdir()))


def _safe_unlink_selected_checkpoint_snapshot(
    snapshot_path: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Remove only our still-identical temporary inode; never unlink a swap."""
    temp_dir = _snapshot_temp_directory()
    if snapshot_path.parent != temp_dir or not snapshot_path.name.startswith(
        _SELECTED_CHECKPOINT_SNAPSHOT_PREFIX
    ):
        return
    if not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("Phase-C requires O_DIRECTORY for selected checkpoint cleanup")
    directory_fd = os.open(
        str(temp_dir), _no_follow_open_flags(os.O_RDONLY | os.O_DIRECTORY)
    )
    try:
        try:
            observed = os.stat(
                snapshot_path.name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return
        if (
            stat.S_ISREG(observed.st_mode)
            and (observed.st_dev, observed.st_ino) == expected_identity
        ):
            os.unlink(snapshot_path.name, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def create_pinned_file_snapshot(
    source_file: str | Path,
    *,
    expected_metadata: Mapping[str, Any],
    label: str,
) -> SelectedCheckpointSnapshot:
    """Make a private O_EXCL, O_NOFOLLOW FD-pinned copy of one bound file.

    This is intentionally a descriptor-to-descriptor copy.  Hashing and byte
    counting happen while the source descriptor is read, so a source pathname
    replacement between selector validation and this copy cannot silently
    become the restore checkpoint.
    """
    source = require_canonical_regular_file(source_file)
    size_bytes, digest = _bound_file_metadata(
        expected_metadata, expected_path=source, label=label
    )
    source_fd, source_before = _open_regular_readonly_nofollow(
        source, label=f"{label} source"
    )
    temp_dir = _snapshot_temp_directory()
    if not hasattr(os, "O_DIRECTORY"):
        os.close(source_fd)
        raise RuntimeError(f"Phase-C requires O_DIRECTORY for {label} snapshots")
    directory_fd = os.open(
        str(temp_dir), _no_follow_open_flags(os.O_RDONLY | os.O_DIRECTORY)
    )
    destination_fd: int | None = None
    pinned_restore_fd: int | None = None
    snapshot_path: Path | None = None
    snapshot_identity: tuple[int, int] | None = None
    try:
        # Validate the source via the descriptor that will actually be copied,
        # rather than relying on the preceding lexical regular-file check.
        if not stat.S_ISREG(source_before.st_mode) or source_before.st_size != size_bytes:
            raise ValueError(f"{label} source changed before snapshot copy")
        for _ in range(32):
            filename = f"{_SELECTED_CHECKPOINT_SNAPSHOT_PREFIX}{secrets.token_hex(24)}.ckpt"
            try:
                destination_fd = os.open(
                    filename,
                    _no_follow_open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
                    0o600,
                    dir_fd=directory_fd,
                )
                snapshot_path = temp_dir / filename
                created = os.fstat(destination_fd)
                if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
                    raise ValueError(f"{label} snapshot creation is not private")
                snapshot_identity = (created.st_dev, created.st_ino)
                break
            except FileExistsError:
                continue
        if destination_fd is None or snapshot_path is None or snapshot_identity is None:
            raise RuntimeError(f"could not allocate private {label} snapshot")

        source_digest = hashlib.sha256()
        copied_bytes = 0
        os.lseek(source_fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(source_fd, _SELECTED_CHECKPOINT_COPY_CHUNK_BYTES)
            if not chunk:
                break
            copied_bytes += len(chunk)
            source_digest.update(chunk)
            _write_all(destination_fd, chunk)
        source_after = os.fstat(source_fd)
        if (
            not stat.S_ISREG(source_after.st_mode)
            or source_after.st_size != size_bytes
            or copied_bytes != size_bytes
            or source_digest.hexdigest() != digest
        ):
            raise ValueError(f"{label} source bytes changed during snapshot copy")
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o400)
        snapshot_info = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(snapshot_info.st_mode)
            or (snapshot_info.st_dev, snapshot_info.st_ino) != snapshot_identity
            or snapshot_info.st_uid != os.geteuid()
            or stat.S_IMODE(snapshot_info.st_mode) != 0o400
            or snapshot_info.st_nlink != 1
            or snapshot_info.st_size != size_bytes
        ):
            raise ValueError(f"{label} snapshot privacy/identity mismatch")
        # Keep a read-only descriptor to the inode itself.  The Trainer gets
        # this ``/proc/self/fd`` view, not ``snapshot_path``; an unlink/replace
        # of the named temp entry can therefore never redirect the restore.
        try:
            pinned_restore_fd = os.open(
                f"/proc/self/fd/{destination_fd}",
                (os.O_RDONLY | (os.O_CLOEXEC if hasattr(os, "O_CLOEXEC") else 0)),
            )
        except OSError as exc:
            raise RuntimeError(f"Phase-C requires a readable /proc/self/fd {label} pin") from exc
        pinned_info = os.fstat(pinned_restore_fd)
        if (
            not stat.S_ISREG(pinned_info.st_mode)
            or (pinned_info.st_dev, pinned_info.st_ino) != snapshot_identity
            or pinned_info.st_size != size_bytes
            or stat.S_IMODE(pinned_info.st_mode) != 0o400
        ):
            raise ValueError(f"{label} pinned FD identity mismatch")
        snapshot = SelectedCheckpointSnapshot(
            canonical_selected_checkpoint=source,
            snapshot_path=snapshot_path,
            size_bytes=size_bytes,
            sha256=digest,
            snapshot_device=snapshot_identity[0],
            snapshot_inode=snapshot_identity[1],
            restore_fd=pinned_restore_fd,
        )
        # Ownership transfers to the returned snapshot.  The finally block
        # still closes the write-only creation descriptor.
        pinned_restore_fd = None
        return snapshot
    except BaseException:
        if snapshot_path is not None and snapshot_identity is not None:
            _safe_unlink_selected_checkpoint_snapshot(
                snapshot_path, expected_identity=snapshot_identity
            )
        raise
    finally:
        if pinned_restore_fd is not None:
            os.close(pinned_restore_fd)
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(directory_fd)
        os.close(source_fd)


def create_selected_checkpoint_snapshot(
    source_checkpoint: str | Path,
    *,
    selector_metadata: Mapping[str, Any],
) -> SelectedCheckpointSnapshot:
    """Create the evaluator's selector-bound selected-student restore copy."""
    return create_pinned_file_snapshot(
        source_checkpoint,
        expected_metadata=selector_metadata,
        label="selected checkpoint",
    )


def validate_pinned_file_snapshot(
    snapshot: SelectedCheckpointSnapshot, *, label: str
) -> None:
    """Revalidate the named snapshot and the pinned inode used by a loader."""
    if not isinstance(snapshot, SelectedCheckpointSnapshot):
        raise TypeError(f"Phase-C {label} snapshot binding is invalid")
    temp_dir = _snapshot_temp_directory()
    if (
        snapshot.snapshot_path.parent != temp_dir
        or not snapshot.snapshot_path.name.startswith(_SELECTED_CHECKPOINT_SNAPSHOT_PREFIX)
    ):
        raise ValueError(f"{label} snapshot escaped the system temporary directory")
    # Check the published temporary name too.  This detects an attempted
    # replacement before any datamodule/Trainer work.  The actual restore
    # remains safe even if an attacker wins after this check because it uses
    # the independent pinned descriptor below.
    name_fd, name_before = _open_regular_readonly_nofollow(
        snapshot.snapshot_path, label=f"{label} snapshot"
    )
    try:
        _validate_fd_against_selected_checkpoint_binding(
            name_fd,
            before=name_before,
            size_bytes=snapshot.size_bytes,
            sha256=snapshot.sha256,
            label=f"{label} snapshot",
            expected_identity=(snapshot.snapshot_device, snapshot.snapshot_inode),
            require_private_snapshot_mode=True,
        )
    finally:
        os.close(name_fd)
    try:
        pinned_before = os.fstat(snapshot.restore_fd)
    except OSError as exc:
        raise ValueError(f"{label} pinned snapshot FD is unavailable") from exc
    _validate_fd_against_selected_checkpoint_binding(
        snapshot.restore_fd,
        before=pinned_before,
        size_bytes=snapshot.size_bytes,
        sha256=snapshot.sha256,
        label=f"{label} pinned snapshot",
        expected_identity=(snapshot.snapshot_device, snapshot.snapshot_inode),
        require_private_snapshot_mode=True,
    )


def validate_pinned_file_origin(snapshot: SelectedCheckpointSnapshot, *, label: str) -> None:
    """Revalidate the canonical source before endpoint/completion/seal work."""
    if not isinstance(snapshot, SelectedCheckpointSnapshot):
        raise TypeError(f"Phase-C {label} snapshot binding is invalid")
    fd, before = _open_regular_readonly_nofollow(
        snapshot.canonical_selected_checkpoint, label=f"{label} source"
    )
    try:
        _validate_fd_against_selected_checkpoint_binding(
            fd,
            before=before,
            size_bytes=snapshot.size_bytes,
            sha256=snapshot.sha256,
            label=f"{label} source",
        )
    finally:
        os.close(fd)


def release_pinned_file_snapshot(snapshot: SelectedCheckpointSnapshot) -> None:
    """Best-effort cleanup that cannot unlink an attacker-replaced temp path."""
    if not isinstance(snapshot, SelectedCheckpointSnapshot):
        raise TypeError("Phase-C pinned file snapshot binding is invalid")
    # Do not close an FD that may have been closed/reused externally.  A valid
    # matching inode is the only descriptor this snapshot ever owns.
    try:
        pinned = os.fstat(snapshot.restore_fd)
    except OSError:
        pinned = None
    if pinned is not None and (pinned.st_dev, pinned.st_ino) == (
        snapshot.snapshot_device,
        snapshot.snapshot_inode,
    ):
        os.close(snapshot.restore_fd)
    _safe_unlink_selected_checkpoint_snapshot(
        snapshot.snapshot_path,
        expected_identity=(snapshot.snapshot_device, snapshot.snapshot_inode),
    )


def validate_selected_checkpoint_snapshot(snapshot: SelectedCheckpointSnapshot) -> None:
    validate_pinned_file_snapshot(snapshot, label="selected checkpoint")


def validate_selected_checkpoint_origin(snapshot: SelectedCheckpointSnapshot) -> None:
    validate_pinned_file_origin(snapshot, label="selected checkpoint")


def release_selected_checkpoint_snapshot(snapshot: SelectedCheckpointSnapshot) -> None:
    release_pinned_file_snapshot(snapshot)


def prepare_phase_c_evaluator_handoff(
    *,
    root: str | Path,
    key: CellKey,
    cost_supplement_path: str | Path,
) -> tuple[
    Mapping[str, Any],
    SelectedCheckpointSnapshot,
    DeploymentConstantsSnapshot,
    Path,
    bytes,
]:
    """Freeze selected restore bytes and exact training-config bytes for evaluation.

    The evaluator is intentionally separate from source fit.  It therefore
    parses only the bound in-memory resolved-config bytes and restores only a
    private copy of the selector-bound checkpoint, never a mutable run-tree
    path.  Both evaluator workers revalidate that copy after Lightning restore
    and revalidate the canonical selected source before endpoint/seal work.
    """
    paths = cell_paths(root, key)
    cell = require_canonical_directory(paths["cell_dir"])
    run = require_canonical_directory(paths["run"], within=cell)
    selector_path = require_canonical_regular_file(paths["selector_records"], within=run)
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    selected = validate_selector_payload(selector, key, run_dir=run)
    checkpoint = require_canonical_regular_file(
        Path(str(selected["checkpoint_path"])), within=run / "checkpoints"
    )

    config_path = require_canonical_regular_file(paths["resolved_config"], within=run)
    # Retain the bytes that will be parsed below.  Re-reading this path after
    # source-cost validation would reintroduce a narrow replace-after-check
    # window; instead compare this immutable snapshot against the fit-end
    # metadata and let the evaluator parse only this snapshot.
    config_bytes = config_path.read_bytes()
    source_cost_path = require_canonical_regular_file(
        paths["source_cost_evidence_run"], within=run
    )
    source_cost = json.loads(source_cost_path.read_text(encoding="utf-8"))
    supplement = require_canonical_regular_file(cost_supplement_path)
    from sua_exploration.mc_maze.m2_native_post33_cost_v4 import (
        validate_source_cost_evidence,
    )

    validate_source_cost_evidence(source_cost, key, cost_supplement_path=supplement)
    bound_config = source_cost.get("resolved_config")
    if not isinstance(bound_config, Mapping) or set(bound_config) != {
        "canonical_path", "size_bytes", "sha256"
    }:
        raise ValueError("source cost evidence resolved-config metadata mismatch")
    if (
        bound_config.get("canonical_path") != str(config_path)
        or bound_config.get("size_bytes") != len(config_bytes)
        or bound_config.get("sha256") != hashlib.sha256(config_bytes).hexdigest()
    ):
        raise ValueError("Phase-C evaluator resolved config changed after source fit")
    snapshot = create_selected_checkpoint_snapshot(
        checkpoint, selector_metadata=selector["selected_checkpoint"]
    )
    try:
        constants_snapshot = capture_deployment_constants_snapshot(
            paths["deployment_constants_run"],
            selector_metadata=selector["deployment_constants"],
        )
    except BaseException:
        release_selected_checkpoint_snapshot(snapshot)
        raise
    return selected, snapshot, constants_snapshot, config_path, config_bytes


def matrix_paths(root: str | Path) -> dict[str, Path]:
    directory = phase_root(root) / "matrix"
    return {
        "directory": directory,
        "manifest": directory / "score_sealed_matrix_manifest.json",
        "completed": directory / "status.completed.json",
        "opened_aggregate": directory / "opened_full_aggregate.json",
    }


def stage_a_paths(root: str | Path) -> dict[str, Path]:
    directory = phase_root(root) / "stage_a"
    return {
        "directory": directory,
        "manifest": directory / "score_sealed_stage_a_manifest.json",
        "completed": directory / "status.completed.json",
        "decision": directory / "opened_stage_a_decision.json",
        "decision_signature": directory / "opened_stage_a_decision.sig",
    }


def authorization_claim_inventory(root: str | Path, *, stage: str | None = None) -> list[dict[str, Any]]:
    directory = phase_root(root) / "authorization_claims"
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("authorization claim root must be a non-symlink directory")
    rows = []
    host_dirs = sorted(directory.iterdir())
    if not host_dirs:
        raise ValueError("empty authorization claim root is forbidden")
    for host_dir in host_dirs:
        if host_dir.is_symlink() or not host_dir.is_dir() or re.fullmatch(r"[A-Za-z0-9_.-]+", host_dir.name) is None:
            raise ValueError("authorization claim host directory invalid")
        files = sorted(host_dir.iterdir())
        if not files:
            raise ValueError("empty authorization claim host directory forbidden")
        for path in files:
            canonical = require_canonical_regular_file(path, within=directory)
            claim = json.loads(canonical.read_text(encoding="utf-8"))
            if claim.get("schema") != "m2_post33_phase_c_authorization_nonce_claim_v4":
                raise ValueError("authorization nonce claim schema mismatch")
            if claim.get("protocol_id") != PROTOCOL_ID or claim.get("phase_id") != PHASE_ID:
                raise ValueError("authorization nonce claim protocol mismatch")
            if claim.get("host_id") != host_dir.name:
                raise ValueError("authorization nonce claim host substitution")
            nonce = claim.get("single_use_nonce")
            if not isinstance(nonce, str) or path.name != f"{nonce}.json" or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
                raise ValueError("authorization nonce claim path/nonce mismatch")
            if claim.get("absolute_cell_root") != str(Path(root).resolve()):
                raise ValueError("authorization nonce claim cell-root substitution")
            for field in ("authorization", "signature", "shard_manifest"):
                metadata = claim.get(field)
                if not isinstance(metadata, Mapping):
                    raise ValueError("authorization nonce claim missing bound file")
                _metadata_matches(Path(metadata.get("canonical_path", "")), metadata)
            if stage is None or claim.get("stage") == stage:
                rows.append({"claim": file_metadata(canonical), "payload": claim})
    nonces = [row["payload"]["single_use_nonce"] for row in rows]
    if len(nonces) != len(set(nonces)):
        raise ValueError("duplicate authorization nonce claim")
    return rows


def require_canonical_regular_file(path: str | Path, *, within: str | Path | None = None) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"symlink is forbidden: {raw}")
    canonical = raw.resolve(strict=True)
    if not canonical.is_file():
        raise ValueError(f"regular file required: {canonical}")
    if str(raw) != str(canonical):
        raise ValueError(f"path must already be canonical: {raw}")
    if within is not None:
        canonical.relative_to(Path(within).resolve(strict=True))
    return canonical


def require_canonical_directory(path: str | Path, *, within: str | Path | None = None) -> Path:
    """Require a lexically canonical, non-symlink directory before resolving it.

    ``Path.resolve`` intentionally follows symlinks.  It is therefore the
    wrong first operation for a provenance boundary: after resolution there is
    no evidence that the caller supplied a symlinked run/checkpoint directory.
    Keep the lexical path long enough to reject that substitution first.
    """
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"directory symlink is forbidden: {raw}")
    canonical = raw.resolve(strict=True)
    if not canonical.is_dir():
        raise ValueError(f"directory required: {canonical}")
    if str(raw) != str(canonical):
        raise ValueError(f"directory path must already be canonical: {raw}")
    if within is not None:
        canonical.relative_to(require_canonical_directory(within))
    return canonical


def require_no_symlink_components(path: Path, *, stop: Path) -> None:
    # Preserve the lexical path while walking it.  Resolving ``path`` first
    # would erase evidence that an intermediate component was a symlink.
    canonical_stop = stop.resolve(strict=True)
    lexical_path = Path(os.path.abspath(path))
    lexical_path.relative_to(canonical_stop)
    canonical_path = lexical_path.resolve(strict=True)
    canonical_path.relative_to(canonical_stop)
    current = canonical_stop
    for part in lexical_path.relative_to(canonical_stop).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink component forbidden: {current}")


def claim_cell(root: str | Path, key: CellKey, *, owner_token: str) -> dict[str, Path]:
    if not owner_token or any(char.isspace() for char in owner_token):
        raise ValueError("owner_token must be non-empty and whitespace-free")
    paths = cell_paths(root, key)
    write_json_exclusive(
        paths["owner"],
        {
            "schema": "m2_post33_phase_c_cell_ownership_v4",
            **key.identity(),
            "owner_token": owner_token,
        },
    )
    return paths


def _validate_owner(paths: Mapping[str, Path], key: CellKey, owner_token: str) -> None:
    owner = json.loads(require_canonical_regular_file(paths["owner"]).read_text(encoding="utf-8"))
    if set(owner) != {
        "schema", "protocol_id", "phase_id", "arm", "fold", "seed",
        "outer_session", "source_sessions", "owner_token",
    }:
        raise ValueError("ownership exact key set mismatch")
    if owner.get("schema") != "m2_post33_phase_c_cell_ownership_v4":
        raise ValueError("ownership schema mismatch")
    for field, expected in key.identity().items():
        if owner.get(field) != expected:
            raise ValueError(f"ownership {field} mismatch")
    if owner.get("owner_token") != owner_token:
        raise PermissionError("owner_token mismatch")


def write_started(root: str | Path, key: CellKey, *, owner_token: str) -> Path:
    paths = cell_paths(root, key)
    _validate_owner(paths, key, owner_token)
    return write_json_exclusive(
        paths["started"],
        {
            "schema": "m2_post33_phase_c_cell_status_v4",
            **key.identity(),
            "state": "started",
            "owner_token": owner_token,
        },
    )


def write_failed_once(
    root: str | Path,
    key: CellKey,
    *,
    owner_token: str,
    failure_kind: str,
    return_code: int | None,
) -> Path:
    paths = cell_paths(root, key)
    _validate_owner(paths, key, owner_token)
    if paths["completed"].exists():
        raise RuntimeError("cannot fail an already completed cell")
    payload = {
        "schema": "m2_post33_phase_c_cell_status_v4",
        **key.identity(),
        "state": "failed",
        "owner_token": owner_token,
        "failure_kind": failure_kind,
        "return_code": return_code,
    }
    try:
        return write_json_exclusive(paths["failed"], payload)
    except FileExistsError:
        existing = json.loads(paths["failed"].read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("failed status already exists with different content")
        return paths["failed"]


def _write_completed(
    root: str | Path,
    key: CellKey,
    *,
    owner_token: str,
    result: Path,
    result_metadata: Mapping[str, Any] | None = None,
) -> Path:
    paths = cell_paths(root, key)
    _validate_owner(paths, key, owner_token)
    if paths["failed"].exists():
        raise RuntimeError("cannot complete a failed cell")
    if result_metadata is None:
        bound_result = file_metadata(result)
    else:
        if set(result_metadata) != {"canonical_path", "size_bytes", "sha256"}:
            raise ValueError("completed result metadata schema mismatch")
        if result_metadata.get("canonical_path") != str(result):
            raise ValueError("completed result metadata path mismatch")
        size = result_metadata.get("size_bytes")
        digest = result_metadata.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("completed result metadata values are invalid")
        bound_result = dict(result_metadata)
    return write_json_exclusive(
        paths["completed"],
        {
            "schema": "m2_post33_phase_c_cell_status_v4",
            **key.identity(),
            "state": "completed",
            "owner_token": owner_token,
            "result": bound_result,
        },
    )


def validate_selector_payload(
    payload: Mapping[str, Any],
    key: CellKey,
    *,
    run_dir: str | Path,
) -> Mapping[str, Any]:
    """Validate a selector against the owning cell's exact checkpoint tree.

    A checkpoint basename such as ``epoch_002.ckpt`` is not a provenance
    boundary: a foreign Hydra output or secondary artifact can carry the same
    basename.  Every selector record must therefore name the one canonical
    file at ``run/checkpoints/epoch_NNN.ckpt`` of this exact cell.
    """
    run = require_canonical_directory(run_dir)
    checkpoint_dir = require_canonical_directory(run / "checkpoints", within=run)
    if set(payload) != {
        "schema", "policy", "selected_epoch", "selected_checkpoint_path",
        "selected_checkpoint", "deployment_constants", "records",
    }:
        raise ValueError("selector top-level exact key set mismatch")
    if payload.get("schema") != SELECTOR_SCHEMAS[key.arm]:
        raise ValueError("selector schema mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("selector records must be a list")
    expected_epochs = EPOCHS[key.arm]
    epochs = [record.get("epoch") for record in records]
    if tuple(sorted(epochs)) != expected_epochs or len(set(epochs)) != len(expected_epochs):
        raise ValueError("selector epochs do not match the exact arm budget")
    for record in records:
        expected_keys = {
            "epoch",
            "metric_name",
            "metric_value",
            "metric_scope",
            "source_sessions",
            "source_totals",
            "outer_session",
            "outer_total",
            "checkpoint_path",
        }
        if set(record) != expected_keys:
            raise ValueError("selector record keys mismatch")
        if record["metric_name"] != "val_source/r2_equal_session_mean":
            raise ValueError("selector metric name substitution")
        if record["metric_scope"] != "exact_six_outer_train_source_sessions_only":
            raise ValueError("selector metric scope substitution")
        value = record["metric_value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("selector metric must be finite")
        if tuple(record["source_sessions"]) != key.source_sessions:
            raise ValueError("selector source session substitution")
        totals = record["source_totals"]
        if set(totals) != set(key.source_sessions) or any(
            isinstance(total, bool) or not isinstance(total, int) or total <= 2
            for total in totals.values()
        ):
            raise ValueError("selector source totals invalid")
        if record["outer_session"] != key.outer_session or record["outer_total"] != 0:
            raise ValueError("outer session entered selector")
        epoch = record["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError("selector epoch must be an integer")
        raw_checkpoint = record["checkpoint_path"]
        if not isinstance(raw_checkpoint, str):
            raise ValueError("selector checkpoint path must be a string")
        checkpoint = Path(raw_checkpoint)
        expected_lexical = checkpoint_dir / f"epoch_{epoch:03d}.ckpt"
        # Reject an epoch-name symlink before ``resolve`` can erase the
        # substitution.  A matching basename alone is not a checkpoint
        # provenance boundary.
        if expected_lexical.is_symlink():
            raise ValueError("selector canonical epoch checkpoint symlink is forbidden")
        expected_checkpoint = require_canonical_regular_file(
            expected_lexical, within=checkpoint_dir
        )
        if str(checkpoint) != str(expected_checkpoint):
            raise ValueError("selector checkpoint is outside the exact canonical epoch path")
        require_canonical_regular_file(checkpoint, within=checkpoint_dir)
    selected = max(records, key=lambda row: (float(row["metric_value"]), -int(row["epoch"])))
    if payload.get("selected_epoch") != selected["epoch"]:
        raise ValueError("selector selected_epoch mismatch")
    if payload.get("selected_checkpoint_path") != selected["checkpoint_path"]:
        raise ValueError("selector selected checkpoint mismatch")
    selected_checkpoint = require_canonical_regular_file(
        Path(str(selected["checkpoint_path"])), within=checkpoint_dir
    )
    selected_metadata = payload.get("selected_checkpoint")
    if not isinstance(selected_metadata, Mapping) or set(selected_metadata) != {
        "canonical_path", "size_bytes", "sha256"
    }:
        raise ValueError("selector selected checkpoint metadata mismatch")
    if selected_metadata != file_metadata(selected_checkpoint):
        raise ValueError("selector selected checkpoint bytes changed after source fit")
    deployment_constants = require_canonical_regular_file(
        run / "deployment_constants.json", within=run
    )
    deployment_metadata = payload.get("deployment_constants")
    if not isinstance(deployment_metadata, Mapping) or set(deployment_metadata) != {
        "canonical_path", "size_bytes", "sha256"
    }:
        raise ValueError("selector deployment constants metadata mismatch")
    if deployment_metadata != file_metadata(deployment_constants):
        raise ValueError("selector deployment constants changed after source fit")
    if payload.get("policy") != "max_finite_equal_session_mean_then_earlier_epoch":
        raise ValueError("selector policy mismatch")
    return selected


def validate_score_commitment(
    payload: Mapping[str, Any], key: CellKey, *, expected_payload_path: str | Path | None = None
) -> None:
    required = {
        "schema",
        "protocol_id",
        "phase_id",
        "arm",
        "fold",
        "seed",
        "opaque_payload_canonical_path",
        "opaque_payload_sha256",
        "opaque_payload_size_bytes",
        "execution_capability_evidence",
        "value_disclosed",
        "aggregation_permitted",
    }
    if set(payload) != required:
        raise ValueError("score commitment keys must be exact")
    if payload["schema"] != "opaque_endpoint_score_commitment_v4":
        raise ValueError("score commitment schema mismatch")
    for field in ("protocol_id", "phase_id", "arm", "fold", "seed"):
        if payload[field] != key.identity()[field]:
            raise ValueError(f"score commitment {field} substitution")
    digest = payload["opaque_payload_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("opaque payload SHA-256 invalid")
    size = payload["opaque_payload_size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("opaque payload size must be positive")
    if payload["value_disclosed"] is not False or payload["aggregation_permitted"] is not False:
        raise ValueError("Phase-C score commitment must stay opaque/non-aggregating")
    committed_path = payload["opaque_payload_canonical_path"]
    if not isinstance(committed_path, str) or not Path(committed_path).is_absolute():
        raise ValueError("opaque payload path must be canonical absolute")
    if expected_payload_path is not None:
        expected = require_canonical_regular_file(expected_payload_path)
        if committed_path != str(expected):
            raise ValueError("opaque payload path substitution")
        if expected.stat().st_size != size or sha256_file(expected) != digest:
            raise ValueError("opaque payload commitment byte/hash substitution")
        expected_capability = require_canonical_regular_file(
            expected.parent / "execution_capability_evidence.json", within=expected.parent
        )
        if payload.get("execution_capability_evidence") != file_metadata(expected_capability):
            raise ValueError("score commitment execution capability substitution")


_COMMON_RUN_FILES = frozenset(
    {
        "resolved_config.yaml",
        "selector_records.json",
        "deployment_constants.json",
        "source_cost_evidence.json",
        "execution_capability_evidence.json",
        "endpoint_payload.from_evaluator.json",
        "score_commitment.from_evaluator.json",
        "deployment_cost_evidence.json",
    }
)

_T4_SECONDARY_CANONICAL_FILES = frozenset(
    {
        "secondary_artifacts/canonical/resolved_config.yaml",
        "secondary_artifacts/canonical/environment.txt",
        "secondary_artifacts/canonical/git_state.txt",
        "secondary_artifacts/canonical/source_manifest.json",
        "secondary_artifacts/canonical/hardware_cost.json",
        "secondary_artifacts/canonical/split_manifest.json",
        "secondary_artifacts/canonical/metrics_summary.csv",
        "secondary_artifacts/canonical/metrics_per_session.csv",
    }
)


def _expected_run_tree(key: CellKey) -> tuple[frozenset[str], frozenset[str]]:
    """Return the closed pre-seal run tree for one arm.

    This is deliberately an allowlist rather than a discovered inventory.
    Phase-C training disables the TensorBoard logger and Hydra bookkeeping
    output, leaving only these fixed-name training/evaluation artifacts.  The
    T4 secondary directory is a fixed legacy-training compatibility surface,
    not an open-ended artifact sink.
    """
    files = set(_COMMON_RUN_FILES)
    files.update(f"checkpoints/epoch_{epoch:03d}.ckpt" for epoch in EPOCHS[key.arm])
    directories = {"checkpoints"}
    if key.arm == "t4":
        files.update(
            {
                "decoder_lifecycle_stages/pretrain.json",
                "decoder_lifecycle_stages/posttrain.json",
                "decoder_lifecycle_stages/reload.json",
                "decoder_lifecycle_stages/prequery.json",
                "decoder_lifecycle_evidence.json",
                "outer_runtime_evidence.json",
            }
        )
        files.update(_T4_SECONDARY_CANONICAL_FILES)
        directories.update(
            {
                "decoder_lifecycle_stages",
                "secondary_artifacts",
                "secondary_artifacts/canonical",
                "secondary_artifacts/canonical/checkpoints",
            }
        )
    return frozenset(files), frozenset(directories)


def build_run_manifest(run_dir: str | Path, key: CellKey) -> dict[str, Any]:
    """Hash a prevalidated, arm-specific closed run tree.

    The previous implementation recursively hashed whatever happened to be
    present, which made an arbitrary pre-existing artifact legitimate merely
    by including it in the manifest.  Compare against the exact path sets
    before hashing a single file instead.
    """
    run = require_canonical_directory(run_dir)
    expected_files, expected_directories = _expected_run_tree(key)
    files: dict[str, dict[str, Any]] = {}
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"run symlink forbidden: {path}")
        if path.is_dir():
            observed_directories.add(path.relative_to(run).as_posix())
            continue
        if not path.is_file():
            raise ValueError(f"non-regular run artifact: {path}")
        relative = path.relative_to(run).as_posix()
        observed_files.add(relative)
    if observed_directories != set(expected_directories):
        raise ValueError("run directory exact set mismatch")
    if observed_files != set(expected_files):
        raise ValueError("run file exact set mismatch")
    for relative in sorted(expected_files):
        path = require_canonical_regular_file(run / relative, within=run)
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema": "m2_post33_phase_c_run_manifest_v4",
        "arm": key.arm,
        "run_dir": str(run),
        "file_count": len(files),
        "files": files,
        "directory_count": len(expected_directories),
    }


def verify_run_manifest(payload: Mapping[str, Any], run_dir: str | Path, key: CellKey) -> None:
    expected_keys = {
        "schema", "arm", "run_dir", "file_count", "files", "directory_count",
    }
    if set(payload) != expected_keys:
        raise ValueError("run manifest exact top-level key set mismatch")
    if payload.get("schema") != "m2_post33_phase_c_run_manifest_v4":
        raise ValueError("run manifest schema mismatch")
    run = require_canonical_directory(run_dir)
    if payload.get("run_dir") != str(run):
        raise ValueError("run manifest directory substitution")
    if payload.get("arm") != key.arm:
        raise ValueError("run manifest arm substitution")
    observed = build_run_manifest(run, key)
    if payload.get("files") != observed["files"] or payload.get("file_count") != observed["file_count"]:
        raise ValueError("run exact-set/hash mismatch (missing, extra, or substituted file)")
    if payload.get("directory_count") != observed["directory_count"]:
        raise ValueError("run directory count substitution")


def _metadata_matches(path: Path, metadata: Mapping[str, Any], *, canonical_required: bool = True) -> None:
    canonical = require_canonical_regular_file(path)
    if canonical_required and metadata.get("canonical_path") != str(canonical):
        raise ValueError("canonical path substitution")
    if metadata.get("size_bytes") != canonical.stat().st_size or metadata.get("sha256") != sha256_file(canonical):
        raise ValueError("file size/SHA substitution")


def _validate_cost_receipt(cost: Mapping[str, Any]) -> None:
    if cost.get("schema") != "m2_post33_paired_arm_cost_receipt_v4":
        raise ValueError("global cost receipt schema mismatch")
    if cost.get("protocol_id") != PROTOCOL_ID or cost.get("phase_id") != PHASE_ID:
        raise ValueError("cost receipt protocol/phase mismatch")
    if set(cost.get("arms", {})) != set(ARMS):
        raise ValueError("cost receipt arms mismatch")
    if cost.get("score_data_accessed") is not False or cost.get("gpu_used") is not False:
        raise ValueError("cost receipt is not score-free CPU evidence")


def validate_deployment_constants(payload: Mapping[str, Any], key: CellKey) -> None:
    if set(payload) != {
        "schema", "protocol_id", "phase_id", "arm", "fold", "seed",
        "t4_normalizer", "source_stage_access_evidence",
    }:
        raise ValueError("deployment constants exact keys mismatch")
    for field, expected in {
        "protocol_id": key.protocol_id, "phase_id": PHASE_ID, "arm": key.arm,
        "fold": key.fold, "seed": key.seed,
    }.items():
        if payload.get(field) != expected:
            raise ValueError(f"deployment constants {field} mismatch")
    expected_access = {
        "stage": "fit", "source_files_opened": 12,
        "outer_calibration_files_opened": 0, "formal_files_opened": 0,
        "outer_directional_label_accesses": 0, "outer_query_batch_calls": 0,
        "scorer_calls": 0,
    }
    if key.arm == "t4":
        expected_access.update({
            "outer_descriptor_fit_invocations": 0,
            "outer_calibration_claims": 0,
        })
    if payload["source_stage_access_evidence"] != expected_access:
        raise ValueError("deployment constants source-stage isolation failed")
    normalizer = payload["t4_normalizer"]
    if key.arm == "spint":
        if normalizer is not None:
            raise ValueError("SPINT deployment constants cannot contain a T4 normalizer")
        return
    if not isinstance(normalizer, Mapping) or set(normalizer) != {
        "mean", "std", "source_sessions", "support_trials"
    }:
        raise ValueError("T4 deployment normalizer keys mismatch")
    mean, std = normalizer["mean"], normalizer["std"]
    if (
        not isinstance(mean, list) or not isinstance(std, list)
        or len(mean) != 4 or len(std) != 4
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in mean + std)
        or any(float(v) <= 0 for v in std)
        or normalizer["source_sessions"] != list(key.source_sessions)
        or normalizer["support_trials"] != 33
    ):
        raise ValueError("T4 deployment normalizer values mismatch")


def _validate_decoder_evidence(
    evidence: Mapping[str, Any],
    key: CellKey,
    *,
    synthetic_fixture_root: str | Path | None,
) -> None:
    if evidence.get("schema") != "m2_post33_decoder_lifecycle_evidence_v4":
        raise ValueError("decoder lifecycle schema mismatch")
    for field in ("protocol_id", "phase_id", "arm", "fold", "seed"):
        if evidence.get(field) != key.identity()[field]:
            raise ValueError(f"decoder evidence {field} substitution")
    if key.arm != "t4":
        raise ValueError("decoder lifecycle evidence belongs only to T4")
    if not isinstance(evidence.get("synthetic_proof"), bool):
        raise ValueError("decoder synthetic_proof must be explicit boolean")
    if evidence["synthetic_proof"]:
        if synthetic_fixture_root is None:
            raise ValueError("synthetic decoder evidence cannot finalize production")
        # The root-capability check lives here as well as at the finalizer
        # boundary, so this relaxed evidence validator cannot be reused as a
        # standalone production bypass through a private import.
        _require_synthetic_test_fixture(synthetic_fixture_root)
    if evidence.get("stages") != ["pretrain", "posttrain", "reload", "prequery"]:
        raise ValueError("decoder lifecycle stages mismatch")
    if evidence.get("tensor_count") != EXPECTED_DECODER_TENSORS:
        raise ValueError("decoder lifecycle is not 31/31")
    if evidence.get("requires_grad_parameter_count") != 0:
        raise ValueError("decoder requires_grad closure failed")
    if evidence.get("optimizer_intersection_count") != 0:
        raise ValueError("decoder optimizer closure failed")
    if evidence.get("updated_tensor_count") != 0 or evidence.get("bit_exact") is not True:
        raise ValueError("decoder byte closure failed")
    snapshots = evidence.get("snapshots")
    if not isinstance(snapshots, Mapping) or set(snapshots) != set(evidence["stages"]):
        raise ValueError("decoder snapshots missing/reordered")
    reference = snapshots["pretrain"]
    for stage in evidence["stages"]:
        if snapshots[stage] != reference:
            raise ValueError(f"decoder substitution at {stage}")
    rows = reference.get("tensors") if isinstance(reference, Mapping) else None
    if not isinstance(rows, list) or len(rows) != EXPECTED_DECODER_TENSORS:
        raise ValueError("decoder snapshot tensor cardinality mismatch")
    names = []
    for row in rows:
        if set(row) != {"name", "shape", "dtype", "num_bytes", "sha256"}:
            raise ValueError("decoder tensor descriptor keys mismatch")
        if not isinstance(row["name"], str) or not row["name"]:
            raise ValueError("decoder tensor name invalid")
        if re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None:
            raise ValueError("decoder tensor hash invalid")
        names.append(row["name"])
    if len(set(names)) != EXPECTED_DECODER_TENSORS:
        raise ValueError("decoder tensor name substitution/duplication")


def validate_t4_outer_runtime_evidence(evidence: Mapping[str, Any], key: CellKey) -> None:
    if key.arm != "t4" or evidence.get("schema") != "m2_post33_t4_outer_runtime_evidence_v4":
        raise ValueError("T4 outer runtime evidence schema/arm mismatch")
    for field in ("protocol_id", "phase_id", "arm", "fold", "seed"):
        if evidence.get(field) != key.identity()[field]:
            raise ValueError(f"T4 outer runtime {field} substitution")
    if evidence.get("outer_session") != key.outer_session:
        raise ValueError("T4 outer runtime session substitution")
    total = evidence.get("metric_total")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 2:
        raise ValueError("T4 outer metric total must be integer >2")
    if (
        evidence.get("metric_finite") is not True
        or evidence.get("metric_value_disclosed") is not False
        or evidence.get("non_outer_sessions_observed") != 0
    ):
        raise ValueError("T4 exact-one finite outer metric evidence failed")
    query = evidence.get("query_window_audit")
    if not isinstance(query, Mapping) or (
        query.get("support_trials") != 33
        or query.get("query_start_trial") != 33
        or query.get("window_size") != 50
        or query.get("eligible_windows", 0) <= 0
        or query.get("full_window_disjoint") is not True
    ):
        raise ValueError("T4 post33 complete-history query evidence failed")
    raw_start = query.get("raw_query_start_bin")
    padded_start = query.get("minimum_window_start_padded_bin")
    if not isinstance(raw_start, int) or not isinstance(padded_start, int) or padded_start - raw_start != 49:
        raise ValueError("T4 post33 window does not bind the complete 50-bin history")
    if evidence.get("neural_support_trials") != 33 or evidence.get("direction_design_rank") != 3:
        raise ValueError("T4 support/rank evidence failed")
    directional = evidence.get("directional_label_support_trials")
    unlabeled = evidence.get("unlabeled_centre_or_rest_trials")
    if not isinstance(directional, int) or directional < 3 or not isinstance(unlabeled, int) or directional + unlabeled != 33:
        raise ValueError("T4 directional/unlabeled support counts invalid")
    balance = evidence.get("direction_balance_min_over_max")
    if not isinstance(balance, (int, float)) or isinstance(balance, bool) or not (0 < float(balance) <= 1):
        raise ValueError("T4 direction balance invalid")
    required_zero = (
        "target_calibration_optimizer_steps",
        "target_calibration_backward_calls",
        "target_calibration_updated_parameter_tensors",
    )
    if any(evidence.get(field) != 0 for field in required_zero):
        raise ValueError("T4 target calibration was not zero-backprop")
    required_false = (
        "centre_rest_assigned_artificial_direction",
        "query_targets_used_for_calibration",
        "query_targets_used_for_normalization",
        "query_targets_used_for_selection",
    )
    if any(evidence.get(field) is not False for field in required_false):
        raise ValueError("T4 label/query isolation evidence failed")


@dataclass
class _SealedDirectoryWriter:
    """One finalizer's O_NOFOLLOW dirfd-anchored sealed artifact writer."""

    cell: Path
    path: Path
    cell_fd: int
    fd: int
    device: int
    inode: int

    def _assert_bound_path(self) -> None:
        """Prove the ``sealed`` name still names the opened child directory.

        This never calls ``Path.resolve`` on ``cell/sealed``: even a concurrent
        name replacement is observed relative to the already-open cell dirfd
        without following a possible symlink.
        """
        try:
            named = os.stat("sealed", dir_fd=self.cell_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ValueError("sealed directory was removed after creation") from exc
        opened = os.fstat(self.fd)
        if (
            not stat.S_ISDIR(named.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or (named.st_dev, named.st_ino) != (self.device, self.inode)
            or (opened.st_dev, opened.st_ino) != (self.device, self.inode)
        ):
            raise ValueError("sealed directory was replaced after creation")

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or Path(name).name != name:
            raise ValueError("sealed artifact filename is invalid")

    def write_bytes(self, name: str, data: bytes) -> Path:
        self._validate_name(name)
        self._assert_bound_path()
        fd = os.open(
            name,
            _no_follow_open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
            0o600,
            dir_fd=self.fd,
        )
        try:
            _write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        self._assert_bound_path()
        return self.path / name

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        return self.write_bytes(name, _json_bytes(payload))

    def file_metadata(self, name: str) -> dict[str, Any]:
        """Hash a sealed child solely through this writer's anchored dirfd."""
        self._validate_name(name)
        self._assert_bound_path()
        fd = os.open(name, _no_follow_open_flags(os.O_RDONLY), dir_fd=self.fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("sealed artifact must be a regular file")
            size_bytes, digest = _stream_fd_size_and_sha256(fd)
            after = os.fstat(fd)
            if (
                not stat.S_ISREG(after.st_mode)
                or after.st_size != size_bytes
                or (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino)
            ):
                raise ValueError("sealed artifact bytes changed while hashing")
        finally:
            os.close(fd)
        self._assert_bound_path()
        return {
            "canonical_path": str(self.path / name),
            "size_bytes": size_bytes,
            "sha256": digest,
        }

    def copy_verified_file(self, source: str | Path, name: str) -> Path:
        """Copy exactly one descriptor-bound source while hashing/counting it."""
        self._validate_name(name)
        self._assert_bound_path()
        source_path = require_canonical_regular_file(source)
        expected = file_metadata(source_path)
        source_fd, before = _open_regular_readonly_nofollow(
            source_path, label=f"sealed copy source {name}"
        )
        destination_fd: int | None = None
        try:
            if before.st_size != expected["size_bytes"]:
                raise ValueError("sealed copy source changed before open")
            destination_fd = os.open(
                name,
                _no_follow_open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
                0o600,
                dir_fd=self.fd,
            )
            digest = hashlib.sha256()
            copied_bytes = 0
            while True:
                chunk = os.read(source_fd, _SELECTED_CHECKPOINT_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                copied_bytes += len(chunk)
                digest.update(chunk)
                _write_all(destination_fd, chunk)
            after = os.fstat(source_fd)
            if (
                not stat.S_ISREG(after.st_mode)
                or after.st_size != expected["size_bytes"]
                or copied_bytes != expected["size_bytes"]
                or digest.hexdigest() != expected["sha256"]
            ):
                raise ValueError("sealed copy source bytes changed during copy")
            os.fsync(destination_fd)
        finally:
            if destination_fd is not None:
                os.close(destination_fd)
            os.close(source_fd)
        self._assert_bound_path()
        return self.path / name

    def close(self) -> None:
        try:
            os.close(self.fd)
        finally:
            os.close(self.cell_fd)


def _create_empty_sealed_directory(cell_dir: str | Path) -> _SealedDirectoryWriter:
    """Atomically create the finalizer-only sealed directory beneath one cell FD.

    A preexisting entry is always rejected.  In particular, this makes a
    ``sealed -> foreign`` symlink fail before the first copy/write rather than
    letting a resolve/mkdir helper create artifacts outside the owned cell.
    """
    cell = require_canonical_directory(cell_dir)
    if not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("Phase-C finalization requires O_DIRECTORY")
    cell_fd: int | None = os.open(
        str(cell), _no_follow_open_flags(os.O_RDONLY | os.O_DIRECTORY)
    )
    sealed_fd: int | None = None
    try:
        try:
            os.mkdir("sealed", 0o700, dir_fd=cell_fd)
        except FileExistsError as exc:
            raise FileExistsError("sealed directory must not preexist before finalization") from exc
        sealed_fd = os.open(
            "sealed",
            _no_follow_open_flags(os.O_RDONLY | os.O_DIRECTORY),
            dir_fd=cell_fd,
        )
        info = os.fstat(sealed_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_nlink < 2:
            raise ValueError("sealed directory creation did not produce a private directory")
        named = os.stat("sealed", dir_fd=cell_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named.st_mode)
            or (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino)
        ):
            raise ValueError("sealed directory changed while being opened")
        # Do not resolve this path: every sealed operation below is anchored to
        # ``sealed_fd`` and ``cell_fd``.  The lexical path is only report data
        # for receipt fields once the anchored identity check succeeds.
        sealed = cell / "sealed"
        if any(os.listdir(sealed_fd)):
            raise ValueError("new sealed directory is unexpectedly nonempty")
        writer = _SealedDirectoryWriter(
            cell=cell,
            path=sealed,
            cell_fd=cell_fd,
            fd=sealed_fd,
            device=info.st_dev,
            inode=info.st_ino,
        )
        sealed_fd = None
        cell_fd = None
        return writer
    finally:
        if sealed_fd is not None:
            os.close(sealed_fd)
        if cell_fd is not None:
            os.close(cell_fd)


def _finalize_cell_score_sealed(
    *,
    root: str | Path,
    key: CellKey,
    owner_token: str,
    score_commitment_path: str | Path,
    opaque_payload_path: str | Path,
    global_cost_receipt_path: str | Path,
    cost_supplement_path: str | Path,
    source_cost_evidence_path: str | Path,
    deployment_cost_evidence_path: str | Path,
    decoder_lifecycle_path: str | Path | None = None,
    paired_spint_completion_path: str | Path | None = None,
    outer_runtime_evidence_path: str | Path | None = None,
    allow_synthetic_decoder_evidence: bool = False,
) -> Path:
    """Finalize one cell without opening, parsing, or aggregating endpoint values."""
    if allow_synthetic_decoder_evidence:
        _require_synthetic_test_fixture(root)
    paths = cell_paths(root, key)
    _validate_owner(paths, key, owner_token)
    if not paths["started"].is_file() or paths["failed"].exists() or paths["completed"].exists():
        raise RuntimeError("cell is not in exactly the started, non-terminal state")
    run = require_canonical_directory(paths["run"], within=paths["cell_dir"])
    selector_file = require_canonical_regular_file(paths["selector_records"], within=run)
    selector_payload = json.loads(selector_file.read_text(encoding="utf-8"))
    selected = validate_selector_payload(selector_payload, key, run_dir=run)
    selected_source = require_canonical_regular_file(selected["checkpoint_path"], within=run)
    deployment_constants_source = require_canonical_regular_file(
        paths["deployment_constants_run"], within=run
    )
    deployment_constants_payload = json.loads(
        deployment_constants_source.read_text(encoding="utf-8")
    )
    validate_deployment_constants(deployment_constants_payload, key)

    opaque_payload_source = require_canonical_regular_file(opaque_payload_path, within=run)
    commitment_source = require_canonical_regular_file(score_commitment_path, within=run)
    commitment = json.loads(commitment_source.read_text(encoding="utf-8"))
    validate_score_commitment(commitment, key, expected_payload_path=opaque_payload_source)
    cost_source = require_canonical_regular_file(global_cost_receipt_path)
    cost = json.loads(cost_source.read_text(encoding="utf-8"))
    _validate_cost_receipt(cost)
    from sua_exploration.mc_maze.m2_native_post33_cost_v4 import (
        validate_cost_supplement,
        validate_deployment_cost_evidence,
        validate_source_cost_evidence,
    )
    supplement_source = require_canonical_regular_file(cost_supplement_path)
    supplement = json.loads(supplement_source.read_text(encoding="utf-8"))
    validate_cost_supplement(supplement)
    if supplement["base_cost_receipt"] != file_metadata(cost_source):
        raise ValueError("cost supplement does not bind the supplied immutable base receipt")
    source_cost_source = require_canonical_regular_file(source_cost_evidence_path, within=run)
    source_cost = json.loads(source_cost_source.read_text(encoding="utf-8"))
    validate_source_cost_evidence(source_cost, key, cost_supplement_path=supplement_source)
    deployment_cost_source = require_canonical_regular_file(deployment_cost_evidence_path, within=run)
    deployment_cost = json.loads(deployment_cost_source.read_text(encoding="utf-8"))
    validate_deployment_cost_evidence(deployment_cost, key)

    decoder_evidence = None
    paired_spint = None
    if key.arm == "t4":
        if decoder_lifecycle_path is None or paired_spint_completion_path is None or outer_runtime_evidence_path is None:
            raise ValueError("T4 finalization requires decoder, paired SPINT, and outer runtime evidence")
        decoder_source = require_canonical_regular_file(decoder_lifecycle_path)
        decoder_evidence = json.loads(decoder_source.read_text(encoding="utf-8"))
        _validate_decoder_evidence(
            decoder_evidence,
            key,
            synthetic_fixture_root=(root if allow_synthetic_decoder_evidence else None),
        )
        paired_key = CellKey(PROTOCOL_ID, "spint", key.fold, key.seed)
        expected_paired_source = require_canonical_regular_file(
            cell_paths(root, paired_key)["completion_receipt"]
        )
        paired_source = require_canonical_regular_file(paired_spint_completion_path)
        if paired_source != expected_paired_source:
            raise ValueError("T4 paired SPINT receipt is not the same-root paired cell")
        paired_spint = json.loads(paired_source.read_text(encoding="utf-8"))
        if paired_spint.get("schema") != "m2_post33_phase_c_completion_receipt_v4":
            raise ValueError("paired SPINT completion schema mismatch")
        for field, expected in paired_key.identity().items():
            if paired_spint.get(field) != expected:
                raise ValueError("paired SPINT completion substitution")
        outer_runtime_source = require_canonical_regular_file(outer_runtime_evidence_path)
        outer_runtime = json.loads(outer_runtime_source.read_text(encoding="utf-8"))
        validate_t4_outer_runtime_evidence(outer_runtime, key)
    elif decoder_lifecycle_path is not None or paired_spint_completion_path is not None or outer_runtime_evidence_path is not None:
        raise ValueError("SPINT finalization forbids T4-only evidence")

    # All validation precedes the first sealed write.  In particular, do not
    # create a partially sealed cell when the arm-specific run tree is not
    # exactly the committed schema.
    run_manifest = build_run_manifest(run, key)
    sealed_writer = _create_empty_sealed_directory(paths["cell_dir"])
    result_path: Path | None = None
    result_metadata: dict[str, Any] | None = None
    try:
        sealed_writer.copy_verified_file(selected_source, "selected.ckpt")
        selected_copy_metadata = sealed_writer.file_metadata("selected.ckpt")
        sealed_writer.copy_verified_file(
            deployment_constants_source, "deployment_constants.json"
        )
        deployment_constants_copy_metadata = sealed_writer.file_metadata(
            "deployment_constants.json"
        )
        sealed_writer.write_json("run_manifest.json", run_manifest)
        sealed_writer.write_json("score_commitment.json", commitment)
        sealed_writer.copy_verified_file(
            opaque_payload_source, "opaque_endpoint_payload.json"
        )
        cost_binding = {
            "schema": "m2_post33_cell_cost_binding_v4",
            **key.identity(),
            "global_cost_receipt": file_metadata(cost_source),
            "cost_supplement": file_metadata(supplement_source),
            "source_cost_evidence": file_metadata(source_cost_source),
            "deployment_cost_evidence": file_metadata(deployment_cost_source),
            "arm_cost_sha256": sha256_json(cost["arms"][key.arm]),
        }
        sealed_writer.write_json("cost_binding.json", cost_binding)
        sealed_writer.write_json("source_cost_evidence.json", source_cost)
        sealed_writer.write_json("deployment_cost_evidence.json", deployment_cost)
        if decoder_evidence is not None:
            sealed_writer.write_json("decoder_lifecycle.json", decoder_evidence)
            assert paired_spint is not None
            sealed_writer.write_json("paired_spint_completion_receipt.json", paired_spint)
            sealed_writer.write_json("outer_runtime_evidence.json", outer_runtime)

        completion = {
            "schema": "m2_post33_phase_c_completion_receipt_v4",
            **key.identity(),
            "selected_epoch": selected["epoch"],
            "selector_policy": selector_payload["policy"],
            "selector_records": file_metadata(selector_file),
            "selected_checkpoint": selected_copy_metadata,
            "deployment_constants": deployment_constants_copy_metadata,
            "resolved_config": file_metadata(paths["resolved_config"]),
        }
        sealed_writer.write_json("completion_receipt.json", completion)
        result_refs = {
            "completion_receipt": sealed_writer.file_metadata("completion_receipt.json"),
            "selected_checkpoint": selected_copy_metadata,
            "deployment_constants": deployment_constants_copy_metadata,
            "score_commitment": sealed_writer.file_metadata("score_commitment.json"),
            "opaque_endpoint_payload": sealed_writer.file_metadata(
                "opaque_endpoint_payload.json"
            ),
            "cost_binding": sealed_writer.file_metadata("cost_binding.json"),
            "run_manifest": sealed_writer.file_metadata("run_manifest.json"),
            "source_cost_evidence": sealed_writer.file_metadata("source_cost_evidence.json"),
            "deployment_cost_evidence": sealed_writer.file_metadata(
                "deployment_cost_evidence.json"
            ),
        }
        if key.arm == "t4":
            result_refs["decoder_lifecycle"] = sealed_writer.file_metadata(
                "decoder_lifecycle.json"
            )
            result_refs["paired_spint_completion"] = sealed_writer.file_metadata(
                "paired_spint_completion_receipt.json"
            )
            result_refs["outer_runtime_evidence"] = sealed_writer.file_metadata(
                "outer_runtime_evidence.json"
            )
        result = {
            "schema": "m2_post33_score_sealed_cell_result_v4",
            **key.identity(),
            "score_value_disclosed": False,
            "score_aggregation_performed": False,
            "references": result_refs,
        }
        result_path = sealed_writer.write_json("result.json", result)
        result_metadata = sealed_writer.file_metadata("result.json")
        sealed_writer._assert_bound_path()
    finally:
        sealed_writer.close()
    if result_path is None or result_metadata is None:
        raise RuntimeError("sealed result was not created")
    _write_completed(
        root,
        key,
        owner_token=owner_token,
        result=result_path,
        result_metadata=result_metadata,
    )
    return result_path


def _expected_sealed_names(key: CellKey) -> set[str]:
    common = {
        "selected.ckpt",
        "deployment_constants.json",
        "completion_receipt.json",
        "cost_binding.json",
        "score_commitment.json",
        "opaque_endpoint_payload.json",
        "run_manifest.json",
        "result.json",
        "source_cost_evidence.json",
        "deployment_cost_evidence.json",
    }
    if key.arm == "t4":
        common |= {
            "decoder_lifecycle.json",
            "paired_spint_completion_receipt.json",
            "outer_runtime_evidence.json",
        }
    return common


def _verify_cell_exact(
    root: str | Path, key: CellKey, *, allow_synthetic: bool
) -> dict[str, Any]:
    if allow_synthetic:
        _require_synthetic_test_fixture(root)
    paths = cell_paths(root, key)
    phase = paths["phase_root"].resolve(strict=True)
    require_no_symlink_components(paths["cell_dir"], stop=phase)
    cell = paths["cell_dir"].resolve(strict=True)
    cell.relative_to(phase)
    top = {entry.name for entry in cell.iterdir()}
    if top != {"control", "run", "sealed"}:
        raise ValueError("cell top-level exact set mismatch")
    control_names = {entry.name for entry in paths["control"].iterdir()}
    if control_names != {"ownership.json", "status.started.json", "status.completed.json"}:
        raise ValueError("completed control exact set mismatch")
    for name in control_names:
        require_canonical_regular_file(paths["control"] / name, within=cell)
    owner = json.loads(paths["owner"].read_text(encoding="utf-8"))
    _validate_owner(paths, key, owner["owner_token"])
    started = json.loads(paths["started"].read_text(encoding="utf-8"))
    completed = json.loads(paths["completed"].read_text(encoding="utf-8"))
    exact_status_keys = {
        "schema", "protocol_id", "phase_id", "arm", "fold", "seed",
        "outer_session", "source_sessions", "state", "owner_token",
    }
    if set(started) != exact_status_keys or started.get("state") != "started":
        raise ValueError("started status schema/exact set mismatch")
    if set(completed) != exact_status_keys | {"result"} or completed.get("state") != "completed":
        raise ValueError("completed status substitution")
    for status in (started, completed):
        if status.get("schema") != "m2_post33_phase_c_cell_status_v4":
            raise ValueError("cell status schema mismatch")
        for field, expected in key.identity().items():
            if status.get(field) != expected:
                raise ValueError("cell status identity substitution")
        if status.get("owner_token") != owner["owner_token"]:
            raise ValueError("cell status owner substitution")

    sealed_names = {entry.name for entry in paths["sealed"].iterdir()}
    if sealed_names != _expected_sealed_names(key):
        raise ValueError("sealed exact set mismatch (missing or extra artifact)")
    for name in sealed_names:
        require_canonical_regular_file(paths["sealed"] / name, within=cell)
    run_manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    verify_run_manifest(run_manifest, paths["run"], key)
    selector_payload = json.loads(paths["selector_records"].read_text(encoding="utf-8"))
    selected = validate_selector_payload(selector_payload, key, run_dir=paths["run"])
    selected_source = require_canonical_regular_file(selected["checkpoint_path"], within=paths["run"])
    commitment = json.loads(paths["score_commitment"].read_text(encoding="utf-8"))
    validate_score_commitment(commitment, key, expected_payload_path=paths["opaque_payload_run"])
    if (
        paths["opaque_payload"].stat().st_size != commitment["opaque_payload_size_bytes"]
        or sha256_file(paths["opaque_payload"]) != commitment["opaque_payload_sha256"]
    ):
        raise ValueError("sealed opaque payload differs from evaluator commitment")
    cost_binding = json.loads(paths["cost_binding"].read_text(encoding="utf-8"))
    if cost_binding.get("schema") != "m2_post33_cell_cost_binding_v4":
        raise ValueError("cost binding schema mismatch")
    if set(cost_binding) != {
        "schema", "protocol_id", "phase_id", "arm", "fold", "seed",
        "outer_session", "source_sessions", "global_cost_receipt",
        "cost_supplement", "source_cost_evidence", "deployment_cost_evidence",
        "arm_cost_sha256",
    }:
        raise ValueError("cost binding exact key set mismatch")
    for field, expected in key.identity().items():
        if cost_binding.get(field) != expected:
            raise ValueError("cost binding cell substitution")
    cost_metadata = cost_binding.get("global_cost_receipt")
    if not isinstance(cost_metadata, Mapping):
        raise ValueError("cost binding missing global cost receipt")
    cost_path = require_canonical_regular_file(cost_metadata.get("canonical_path", ""))
    _metadata_matches(cost_path, cost_metadata)
    cost_payload = json.loads(cost_path.read_text(encoding="utf-8"))
    _validate_cost_receipt(cost_payload)
    if cost_binding.get("arm_cost_sha256") != sha256_json(cost_payload["arms"][key.arm]):
        raise ValueError("arm cost binding substitution")
    from sua_exploration.mc_maze.m2_native_post33_cost_v4 import (
        validate_cost_supplement,
        validate_deployment_cost_evidence,
        validate_source_cost_evidence,
    )
    supplement_metadata = cost_binding.get("cost_supplement")
    if not isinstance(supplement_metadata, Mapping):
        raise ValueError("cost binding missing supplement")
    supplement_path = require_canonical_regular_file(supplement_metadata.get("canonical_path", ""))
    _metadata_matches(supplement_path, supplement_metadata)
    supplement_payload = json.loads(supplement_path.read_text(encoding="utf-8"))
    validate_cost_supplement(supplement_payload)
    if supplement_payload["base_cost_receipt"] != cost_metadata:
        raise ValueError("cost supplement/base binding mismatch")
    for field, sealed_path, run_path in (
        ("source_cost_evidence", paths["source_cost_evidence"], paths["source_cost_evidence_run"]),
        ("deployment_cost_evidence", paths["deployment_cost_evidence"], paths["deployment_cost_evidence_run"]),
    ):
        metadata = cost_binding.get(field)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"cost binding missing {field}")
        _metadata_matches(run_path, metadata)
        if sealed_path.read_bytes() != run_path.read_bytes():
            raise ValueError(f"sealed {field} differs from run evidence")
    source_cost_payload = json.loads(paths["source_cost_evidence"].read_text(encoding="utf-8"))
    deployment_cost_payload = json.loads(paths["deployment_cost_evidence"].read_text(encoding="utf-8"))
    validate_source_cost_evidence(
        source_cost_payload, key, cost_supplement_path=supplement_path
    )
    validate_deployment_cost_evidence(deployment_cost_payload, key)

    completion = json.loads(paths["completion_receipt"].read_text(encoding="utf-8"))
    if completion.get("schema") != "m2_post33_phase_c_completion_receipt_v4":
        raise ValueError("completion receipt schema mismatch")
    for field, expected in key.identity().items():
        if completion.get(field) != expected:
            raise ValueError("completion receipt cell substitution")
    if completion.get("selected_epoch") != selected["epoch"]:
        raise ValueError("completion selected epoch substitution")
    if completion.get("selector_policy") != selector_payload["policy"]:
        raise ValueError("completion selector policy substitution")
    _metadata_matches(paths["selected_checkpoint"], completion["selected_checkpoint"])
    _metadata_matches(paths["deployment_constants"], completion["deployment_constants"])
    constants = json.loads(paths["deployment_constants"].read_text(encoding="utf-8"))
    validate_deployment_constants(constants, key)
    if sha256_file(paths["selected_checkpoint"]) != sha256_file(selected_source):
        raise ValueError("sealed selected checkpoint differs from selected source")
    _metadata_matches(paths["selector_records"], completion["selector_records"])
    _metadata_matches(paths["resolved_config"], completion["resolved_config"])

    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    if result.get("schema") != "m2_post33_score_sealed_cell_result_v4":
        raise ValueError("result schema mismatch")
    for field, expected in key.identity().items():
        if result.get(field) != expected:
            raise ValueError("result cell substitution")
    if result.get("score_value_disclosed") is not False or result.get("score_aggregation_performed") is not False:
        raise ValueError("result disclosed/aggregated a score")
    expected_refs = {
        "completion_receipt": paths["completion_receipt"],
        "selected_checkpoint": paths["selected_checkpoint"],
        "deployment_constants": paths["deployment_constants"],
        "score_commitment": paths["score_commitment"],
        "opaque_endpoint_payload": paths["opaque_payload"],
        "cost_binding": paths["cost_binding"],
        "run_manifest": paths["run_manifest"],
        "source_cost_evidence": paths["source_cost_evidence"],
        "deployment_cost_evidence": paths["deployment_cost_evidence"],
    }
    if key.arm == "t4":
        expected_refs.update(
            {
                "decoder_lifecycle": paths["decoder_lifecycle"],
                "paired_spint_completion": paths["paired_spint_completion"],
                "outer_runtime_evidence": paths["outer_runtime_evidence"],
            }
        )
        evidence = json.loads(paths["decoder_lifecycle"].read_text(encoding="utf-8"))
        _validate_decoder_evidence(
            evidence,
            key,
            synthetic_fixture_root=(root if allow_synthetic else None),
        )
        paired = json.loads(paths["paired_spint_completion"].read_text(encoding="utf-8"))
        paired_key = CellKey(PROTOCOL_ID, "spint", key.fold, key.seed)
        for field, expected in paired_key.identity().items():
            if paired.get(field) != expected:
                raise ValueError("paired SPINT substitution")
        actual_paired = cell_paths(root, paired_key)["completion_receipt"]
        if paths["paired_spint_completion"].read_bytes() != actual_paired.read_bytes():
            raise ValueError("sealed paired SPINT receipt differs from same-root source")
        outer_runtime = json.loads(paths["outer_runtime_evidence"].read_text(encoding="utf-8"))
        validate_t4_outer_runtime_evidence(outer_runtime, key)
    if set(result.get("references", {})) != set(expected_refs):
        raise ValueError("result reference exact set mismatch")
    for name, path in expected_refs.items():
        _metadata_matches(path, result["references"][name])
    _metadata_matches(paths["result"], completed["result"])
    return {
        "cell": key.identity(),
        "result": file_metadata(paths["result"]),
        "score_commitment_sha256": sha256_file(paths["score_commitment"]),
        "synthetic_decoder_evidence": (
            bool(json.loads(paths["decoder_lifecycle"].read_text())["synthetic_proof"])
            if key.arm == "t4"
            else False
        ),
    }


def _finalize_matrix_score_sealed(
    root: str | Path, *, allow_synthetic: bool = False
) -> Path:
    """Finalize exactly 42 terminal cells without opening endpoint payloads."""
    if allow_synthetic:
        _require_synthetic_test_fixture(root)
    stage_a = _verify_stage_a_exact(root, allow_synthetic=allow_synthetic)
    if stage_a["decision_present"] is not True:
        raise ValueError("full matrix cannot seal before the Stage-A decision")
    from sua_exploration.mc_maze.m2_native_post33_openers_v4 import _validate_stage_a_decision
    _validate_stage_a_decision(root, require_continue=True, allow_synthetic=allow_synthetic)
    phase = phase_root(root).resolve(strict=True)
    cells_root = phase / "cells"
    expected_cell_dirs = {str((phase / key.relative_cell).resolve()) for key in matrix_keys()}
    observed_seed_dirs = {
        str(path.resolve())
        for path in cells_root.glob("arm-*/fold-*/seed-*")
        if path.is_dir()
    }
    if observed_seed_dirs != expected_cell_dirs:
        raise ValueError("matrix cell directory cardinality/exact set mismatch")
    rows = []
    for key in matrix_keys():
        rows.append(_verify_cell_exact(root, key, allow_synthetic=allow_synthetic))
    if len(rows) != EXPECTED_CELLS:
        raise RuntimeError("matrix cardinality is not 42")
    if not allow_synthetic and any(row["synthetic_decoder_evidence"] for row in rows):
        raise ValueError("production matrix cannot contain synthetic decoder evidence")
    by_identity = {
        (row["cell"]["arm"], row["cell"]["fold"], row["cell"]["seed"]): row
        for row in rows
    }
    pairs = []
    for fold in FOLDS:
        for seed in SEEDS:
            spint = by_identity[("spint", fold, seed)]
            t4 = by_identity[("t4", fold, seed)]
            pairs.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "spint_result_sha256": spint["result"]["sha256"],
                    "t4_result_sha256": t4["result"]["sha256"],
                    "endpoint_values_opened": False,
                    "score_delta_computed": False,
                }
            )
    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError("paired matrix cardinality is not 21")
    authorization_claims = authorization_claim_inventory(root)
    if not allow_synthetic and {row["payload"].get("stage") for row in authorization_claims} != {
        "stage_a", "stage_b"
    }:
        raise ValueError("production matrix requires Stage-A and Stage-B authorization claims")
    if not allow_synthetic:
        from sua_exploration.mc_maze.m2_native_post33_authorization_v4 import validate_claim_coverage
        validate_claim_coverage(root, stage="stage_a")
        validate_claim_coverage(root, stage="stage_b")
    manifest = {
        "schema": "m2_post33_score_sealed_matrix_manifest_v4",
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "cell_count": EXPECTED_CELLS,
        "paired_cell_count": EXPECTED_PAIRS,
        "endpoint_values_opened": False,
        "score_aggregation_performed": False,
        "cells": rows,
        "pairs": pairs,
        "authorization_nonce_claims": authorization_claims,
    }
    destinations = matrix_paths(root)
    if destinations["directory"].exists():
        raise FileExistsError("matrix finalization is write-once")
    manifest_path = write_json_exclusive(destinations["manifest"], manifest)
    write_json_exclusive(
        destinations["completed"],
        {
            "schema": "m2_post33_phase_c_matrix_status_v4",
            "protocol_id": PROTOCOL_ID,
            "phase_id": PHASE_ID,
            "state": "completed",
            "manifest": file_metadata(manifest_path),
        },
    )
    return manifest_path


def _finalize_stage_a_score_sealed(
    root: str | Path, *, allow_synthetic: bool = False
) -> Path:
    """Seal the exact 14 seed-42 cells before any Stage-A delta can be opened."""
    if allow_synthetic:
        _require_synthetic_test_fixture(root)
    validate_stage_a_cell_directory_set(root)
    stage_keys = tuple(
        CellKey(PROTOCOL_ID, arm, fold, 42) for arm in ARMS for fold in FOLDS
    )
    rows = [
        _verify_cell_exact(root, key, allow_synthetic=allow_synthetic)
        for key in stage_keys
    ]
    if len(rows) != 14:
        raise RuntimeError("Stage A cardinality is not exactly 14")
    by_identity = {
        (row["cell"]["arm"], row["cell"]["fold"]): row for row in rows
    }
    pairs = []
    for fold in FOLDS:
        spint = by_identity[("spint", fold)]
        t4 = by_identity[("t4", fold)]
        pairs.append(
            {
                "fold": fold,
                "seed": 42,
                "spint_result_sha256": spint["result"]["sha256"],
                "t4_result_sha256": t4["result"]["sha256"],
                "endpoint_values_opened": False,
                "score_delta_computed": False,
            }
        )
    if len(pairs) != 7:
        raise RuntimeError("Stage A paired cardinality is not exactly seven")
    authorization_claims = authorization_claim_inventory(root, stage="stage_a")
    if not allow_synthetic and not authorization_claims:
        raise ValueError("production Stage A requires a claimed signed authorization")
    if not allow_synthetic:
        from sua_exploration.mc_maze.m2_native_post33_authorization_v4 import validate_claim_coverage
        validate_claim_coverage(root, stage="stage_a")
    manifest = {
        "schema": "m2_post33_score_sealed_stage_a_manifest_v4",
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "seed": 42,
        "cell_count": 14,
        "paired_cell_count": 7,
        "endpoint_values_opened": False,
        "score_delta_computed": False,
        "cells": rows,
        "pairs": pairs,
        "authorization_nonce_claims": authorization_claims,
    }
    destinations = stage_a_paths(root)
    if destinations["directory"].exists():
        raise FileExistsError("Stage A finalization is write-once")
    manifest_path = write_json_exclusive(destinations["manifest"], manifest)
    write_json_exclusive(
        destinations["completed"],
        {
            "schema": "m2_post33_phase_c_stage_a_status_v4",
            "protocol_id": PROTOCOL_ID,
            "phase_id": PHASE_ID,
            "state": "completed",
            "manifest": file_metadata(manifest_path),
        },
    )
    return manifest_path


def _verify_stage_a_exact(
    root: str | Path, *, allow_synthetic: bool
) -> dict[str, Any]:
    if allow_synthetic:
        _require_synthetic_test_fixture(root)
    # This verifier remains valid after a signed Stage-A continue decision has
    # legitimately allowed seed-43/44 cells to be created.  The live exact-14
    # boundary is therefore enforced by Stage-A finalization and the Stage-A
    # opener immediately before its first score read, not here.
    paths = stage_a_paths(root)
    directory = paths["directory"].resolve(strict=True)
    names = {entry.name for entry in directory.iterdir()}
    base_names = {
        "score_sealed_stage_a_manifest.json",
        "status.completed.json",
    }
    if names not in (
        base_names,
        base_names | {"opened_stage_a_decision.json"},
        base_names | {"opened_stage_a_decision.json", "opened_stage_a_decision.sig"},
    ):
        raise ValueError("Stage A artifact exact set mismatch")
    if "opened_stage_a_decision.json" in names:
        require_canonical_regular_file(paths["decision"], within=directory)
    if "opened_stage_a_decision.sig" in names:
        require_canonical_regular_file(paths["decision_signature"], within=directory)
    manifest = json.loads(require_canonical_regular_file(paths["manifest"]).read_text(encoding="utf-8"))
    status = json.loads(require_canonical_regular_file(paths["completed"]).read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "m2_post33_score_sealed_stage_a_manifest_v4"
        or manifest.get("cell_count") != 14
        or len(manifest.get("cells", [])) != 14
        or manifest.get("paired_cell_count") != 7
        or len(manifest.get("pairs", [])) != 7
        or manifest.get("endpoint_values_opened") is not False
        or manifest.get("score_delta_computed") is not False
        or not isinstance(manifest.get("authorization_nonce_claims"), list)
    ):
        raise ValueError("Stage A manifest/cardinality/score-sealing mismatch")
    expected_keys = tuple(
        CellKey(PROTOCOL_ID, arm, fold, 42) for arm in ARMS for fold in FOLDS
    )
    observed = [
        _verify_cell_exact(root, key, allow_synthetic=allow_synthetic)
        for key in expected_keys
    ]
    if manifest["cells"] != observed:
        raise ValueError("Stage A cell substitution")
    by_identity = {
        (row["cell"]["arm"], row["cell"]["fold"]): row for row in observed
    }
    expected_pairs = [
        {
            "fold": fold,
            "seed": 42,
            "spint_result_sha256": by_identity[("spint", fold)]["result"]["sha256"],
            "t4_result_sha256": by_identity[("t4", fold)]["result"]["sha256"],
            "endpoint_values_opened": False,
            "score_delta_computed": False,
        }
        for fold in FOLDS
    ]
    if manifest["pairs"] != expected_pairs:
        raise ValueError("Stage A pair ordering/substitution")
    if manifest["authorization_nonce_claims"] != authorization_claim_inventory(root, stage="stage_a"):
        raise ValueError("Stage A authorization claim substitution")
    if not allow_synthetic:
        from sua_exploration.mc_maze.m2_native_post33_authorization_v4 import validate_claim_coverage
        validate_claim_coverage(root, stage="stage_a")
    if set(status) != {"schema", "protocol_id", "phase_id", "state", "manifest"} or (
        status.get("schema") != "m2_post33_phase_c_stage_a_status_v4"
        or status.get("protocol_id") != PROTOCOL_ID
        or status.get("phase_id") != PHASE_ID
        or status.get("state") != "completed"
    ):
        raise ValueError("Stage A status schema/identity/exact set mismatch")
    _metadata_matches(paths["manifest"], status.get("manifest", {}))
    return {
        "status": "PASS_EXACT_SCORE_SEALED_STAGE_A_V4", "cells": 14, "pairs": 7,
        "decision_present": "opened_stage_a_decision.json" in names,
        "decision_signature_present": "opened_stage_a_decision.sig" in names,
    }


def _verify_matrix_exact(
    root: str | Path, *, allow_synthetic: bool
) -> dict[str, Any]:
    if allow_synthetic:
        _require_synthetic_test_fixture(root)
    phase = phase_root(root).resolve(strict=True)
    phase_names = {entry.name for entry in phase.iterdir()}
    # An opening capability is consumed before either delayed-score opener
    # reads an endpoint payload.  Its nonce claims are therefore part of the
    # legitimate Phase-C topology after the Stage-A opening, alongside the
    # execution nonce claims written by workers.  Keep the set closed while
    # allowing that auditable, protocol-owned directory.
    if phase_names not in (
        {"cells", "stage_a", "matrix"},
        {"cells", "stage_a", "matrix", "authorization_claims"},
        {"cells", "stage_a", "matrix", "opening_authorization_claims"},
        {
            "cells", "stage_a", "matrix", "authorization_claims",
            "opening_authorization_claims",
        },
    ):
        raise ValueError("phase-root exact set mismatch")
    stage_a = _verify_stage_a_exact(root, allow_synthetic=allow_synthetic)
    if stage_a["decision_present"] is not True:
        raise ValueError("full matrix verification requires the Stage-A decision")
    from sua_exploration.mc_maze.m2_native_post33_openers_v4 import _validate_stage_a_decision
    _validate_stage_a_decision(root, require_continue=True, allow_synthetic=allow_synthetic)
    manifest_path = matrix_paths(root)["manifest"]
    status_path = matrix_paths(root)["completed"]
    matrix_dir = manifest_path.parent
    matrix_names = {entry.name for entry in matrix_dir.iterdir()}
    base_matrix_names = {
        "score_sealed_matrix_manifest.json",
        "status.completed.json",
    }
    if matrix_names not in (
        base_matrix_names, base_matrix_names | {"opened_full_aggregate.json"}
    ):
        raise ValueError("matrix artifact exact set mismatch")
    if "opened_full_aggregate.json" in matrix_names:
        require_canonical_regular_file(matrix_paths(root)["opened_aggregate"], within=matrix_dir)
    manifest = json.loads(require_canonical_regular_file(manifest_path, within=phase).read_text(encoding="utf-8"))
    status = json.loads(require_canonical_regular_file(status_path, within=phase).read_text(encoding="utf-8"))
    if manifest.get("schema") != "m2_post33_score_sealed_matrix_manifest_v4":
        raise ValueError("matrix schema mismatch")
    if manifest.get("cell_count") != EXPECTED_CELLS or len(manifest.get("cells", [])) != EXPECTED_CELLS:
        raise ValueError("matrix cell cardinality mismatch")
    if manifest.get("paired_cell_count") != EXPECTED_PAIRS or len(manifest.get("pairs", [])) != EXPECTED_PAIRS:
        raise ValueError("matrix pair cardinality mismatch")
    if not isinstance(manifest.get("authorization_nonce_claims"), list):
        raise ValueError("matrix authorization claim inventory missing")
    if manifest.get("endpoint_values_opened") is not False or manifest.get("score_aggregation_performed") is not False:
        raise ValueError("matrix opened/aggregated endpoint values")
    observed = [
        _verify_cell_exact(root, key, allow_synthetic=allow_synthetic)
        for key in matrix_keys()
    ]
    if manifest["cells"] != observed:
        raise ValueError("matrix manifest cell substitution")
    by_identity = {
        (row["cell"]["arm"], row["cell"]["fold"], row["cell"]["seed"]): row
        for row in observed
    }
    expected_pairs = [
        {
            "fold": fold,
            "seed": seed,
            "spint_result_sha256": by_identity[("spint", fold, seed)]["result"]["sha256"],
            "t4_result_sha256": by_identity[("t4", fold, seed)]["result"]["sha256"],
            "endpoint_values_opened": False,
            "score_delta_computed": False,
        }
        for fold in FOLDS
        for seed in SEEDS
    ]
    if manifest["pairs"] != expected_pairs:
        raise ValueError("matrix pair ordering/substitution")
    if manifest["authorization_nonce_claims"] != authorization_claim_inventory(root):
        raise ValueError("matrix authorization claim substitution")
    if not allow_synthetic:
        from sua_exploration.mc_maze.m2_native_post33_authorization_v4 import validate_claim_coverage
        validate_claim_coverage(root, stage="stage_a")
        validate_claim_coverage(root, stage="stage_b")
    if set(status) != {"schema", "protocol_id", "phase_id", "state", "manifest"} or (
        status.get("schema") != "m2_post33_phase_c_matrix_status_v4"
        or status.get("protocol_id") != PROTOCOL_ID
        or status.get("phase_id") != PHASE_ID
        or status.get("state") != "completed"
    ):
        raise ValueError("matrix status schema/identity/exact set mismatch")
    _metadata_matches(manifest_path, status.get("manifest", {}))
    return {
        "status": "PASS_EXACT_SCORE_SEALED_MATRIX_V4",
        "cell_count": EXPECTED_CELLS,
        "pair_count": EXPECTED_PAIRS,
        "endpoint_values_opened": False,
        "opened_aggregate_present": "opened_full_aggregate.json" in matrix_names,
    }


# Production entrypoints deliberately do not expose an allow_synthetic switch.
# The private implementations above are reachable only from the explicit
# tests-only support module; production finalization and verification always
# reject synthetic decoder evidence and require signed claim coverage.
def finalize_cell_score_sealed(
    *,
    root: str | Path,
    key: CellKey,
    owner_token: str,
    score_commitment_path: str | Path,
    opaque_payload_path: str | Path,
    global_cost_receipt_path: str | Path,
    cost_supplement_path: str | Path,
    source_cost_evidence_path: str | Path,
    deployment_cost_evidence_path: str | Path,
    decoder_lifecycle_path: str | Path | None = None,
    paired_spint_completion_path: str | Path | None = None,
    outer_runtime_evidence_path: str | Path | None = None,
) -> Path:
    return _finalize_cell_score_sealed(
        root=root,
        key=key,
        owner_token=owner_token,
        score_commitment_path=score_commitment_path,
        opaque_payload_path=opaque_payload_path,
        global_cost_receipt_path=global_cost_receipt_path,
        cost_supplement_path=cost_supplement_path,
        source_cost_evidence_path=source_cost_evidence_path,
        deployment_cost_evidence_path=deployment_cost_evidence_path,
        decoder_lifecycle_path=decoder_lifecycle_path,
        paired_spint_completion_path=paired_spint_completion_path,
        outer_runtime_evidence_path=outer_runtime_evidence_path,
        allow_synthetic_decoder_evidence=False,
    )


def verify_cell_exact(root: str | Path, key: CellKey) -> dict[str, Any]:
    return _verify_cell_exact(root, key, allow_synthetic=False)


def finalize_stage_a_score_sealed(root: str | Path) -> Path:
    return _finalize_stage_a_score_sealed(root, allow_synthetic=False)


def verify_stage_a_exact(root: str | Path) -> dict[str, Any]:
    return _verify_stage_a_exact(root, allow_synthetic=False)


def finalize_matrix_score_sealed(root: str | Path) -> Path:
    return _finalize_matrix_score_sealed(root, allow_synthetic=False)


def verify_matrix_exact(root: str | Path) -> dict[str, Any]:
    return _verify_matrix_exact(root, allow_synthetic=False)


def require_same_root_paired_spint_teacher(
    root: str | Path,
    key: CellKey,
    receipt_path: str | Path,
) -> Path:
    """Return the only sealed SPINT teacher allowed for a T4 cell.

    The T4 model constructor reads a completion receipt while it is being
    instantiated.  Production entrypoints call this guard *before* they
    construct a datamodule, model, or ``Trainer`` so a foreign-root receipt
    cannot silently select a different SPINT teacher with the same fold/seed.
    """
    if key.arm != "t4":
        raise ValueError("paired SPINT teacher is reserved for a T4 cell")
    paired_key = CellKey(PROTOCOL_ID, "spint", key.fold, key.seed)
    expected_receipt = require_canonical_regular_file(
        cell_paths(root, paired_key)["completion_receipt"]
    )
    supplied_receipt = require_canonical_regular_file(receipt_path)
    if supplied_receipt != expected_receipt:
        raise PermissionError("paired SPINT teacher receipt is not the same-root paired cell")

    # Validate the entire completed SPINT cell rather than trusting just a
    # superficially well-formed receipt.  This stays score-blind and verifies
    # its selected checkpoint/provenance binding before T4 can load weights.
    verify_cell_exact(root, paired_key)
    receipt = json.loads(expected_receipt.read_text(encoding="utf-8"))
    if receipt.get("schema") != "m2_post33_phase_c_completion_receipt_v4":
        raise ValueError("paired SPINT completion schema mismatch")
    for field, expected in paired_key.identity().items():
        if receipt.get(field) != expected:
            raise ValueError("paired SPINT completion identity substitution")
    selected = receipt.get("selected_checkpoint")
    if not isinstance(selected, Mapping):
        raise ValueError("paired SPINT completion lacks selected checkpoint metadata")
    checkpoint = require_canonical_regular_file(
        selected.get("canonical_path", ""),
        within=cell_paths(root, paired_key)["sealed"],
    )
    _metadata_matches(checkpoint, selected)
    return checkpoint


def bind_same_root_paired_spint_teacher(
    root: str | Path,
    key: CellKey,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Return a byte-bound same-root teacher binding for a T4 model lifecycle.

    The ordinary path-returning guard is useful to launchers.  A model can
    survive until a later ``setup`` call, however, so it additionally needs
    immutable receipt/checkpoint metadata to re-check immediately before and
    after the historical loader dereferences the checkpoint.
    """
    checkpoint = require_same_root_paired_spint_teacher(root, key, receipt_path)
    paired_key = CellKey(PROTOCOL_ID, "spint", key.fold, key.seed)
    receipt = require_canonical_regular_file(
        cell_paths(root, paired_key)["completion_receipt"]
    )
    return {
        "schema": "m2_post33_phase_c_same_root_teacher_binding_v4",
        "t4_cell": key.identity(),
        "paired_spint_cell": paired_key.identity(),
        "receipt": file_metadata(receipt),
        "selected_checkpoint": file_metadata(checkpoint),
    }


def bind_same_root_paired_spint_teacher_from_t4_owner(
    *,
    t4_owner_path: str | Path,
    t4_owner_token: str,
    receipt_path: str | Path,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    """Derive the T4 root from an owned cell, never from a model caller.

    Generic training/model construction must not be able to choose an
    arbitrary root merely because a foreign receipt has matching fold/seed
    metadata.  The canonical T4 ownership path names the only acceptable
    cell; this function reconstructs its root from the fixed Phase-C layout,
    validates ownership, and then creates the same-root binding.
    """
    owner = require_canonical_regular_file(t4_owner_path)
    if owner.name != "ownership.json" or owner.parent.name != "control":
        raise PermissionError("T4 teacher binding owner path is not a Phase-C ownership file")
    cell = owner.parent.parent
    try:
        root = cell.parents[5]
    except IndexError as exc:
        raise PermissionError("T4 teacher binding owner path is outside Phase-C layout") from exc
    key = CellKey(PROTOCOL_ID, "t4", fold, seed)
    paths = cell_paths(root, key)
    if paths["cell_dir"] != cell or paths["owner"] != owner:
        raise PermissionError("T4 teacher binding owner/cell layout substitution")
    _validate_owner(paths, key, t4_owner_token)
    return bind_same_root_paired_spint_teacher(root, key, receipt_path)
