"""Minimal static control plane for the fresh Phase-C v5/r10 recovery.

This module intentionally has no signer, socket, live process authority, or
score-opening entry point.  A sealed prelaunch manifest names the exact 42
cells and current source hashes.  The single supervisor is the only component
allowed to create the O_EXCL run lock; workers merely verify that fixed state
before Hydra or an evaluator can start.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    ARMS,
    FOLDS,
    PHASE_ID,
    PROTOCOL_ID,
    CellKey,
    cell_paths,
    file_metadata,
    require_canonical_directory,
    validate_phase_c_training_wrapper_argv,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[2]
R10_RECEIPT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r10_launch_receipts_20260805"
R10_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r10_cells_20260805"
PRELAUNCH_DIR = R10_RECEIPT_ROOT / "prelaunch"
STATIC_MANIFEST = PRELAUNCH_DIR / "static_manifest.json"
OVERRIDE = PRELAUNCH_DIR / "owner_control_plane_simplification_override.json"
RUN_LOCK = R10_RECEIPT_ROOT / "control/supervisor_run_lock.json"
MANIFEST_ENV = "M2_POST33_PHASE_C_V5_R10_STATIC_MANIFEST"
CELL_ENV = "M2_POST33_PHASE_C_CELL_DIR"

MANIFEST_SCHEMA = "m2_post33_phase_c_v5_r10_static_prelaunch_manifest_v1"
LOCK_SCHEMA = "m2_post33_phase_c_v5_r10_supervisor_run_lock_v1"
EXECUTION_EVIDENCE_SCHEMA = "m2_post33_phase_c_v5_r10_static_execution_evidence_v1"


class StaticControlError(PermissionError):
    """The static r10 contract was violated before any scientific work."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticControlError(message)


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


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"r10 {label} missing/noncanonical")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticControlError(f"r10 {label} is not valid JSON") from exc
    _require(isinstance(payload, Mapping), f"r10 {label} must be a mapping")
    return payload


def _expected_cells() -> dict[str, tuple[CellKey, ...]]:
    stage_a = tuple(
        CellKey(PROTOCOL_ID, arm, fold, 42)
        for fold in FOLDS for arm in ARMS
    )
    stage_b = tuple(
        CellKey(PROTOCOL_ID, arm, fold, seed)
        for seed in (43, 44) for fold in FOLDS for arm in ARMS
    )
    return {"stage_a": stage_a, "stage_b": stage_b}


def _key_record(key: CellKey) -> dict[str, Any]:
    return {"arm": key.arm, "fold": key.fold, "seed": key.seed}


def cell_schedule() -> dict[str, tuple[CellKey, ...]]:
    return _expected_cells()


def manifest_metadata() -> dict[str, Any]:
    return file_metadata(STATIC_MANIFEST)


def validate_static_manifest(*, require_lock: bool = False) -> Mapping[str, Any]:
    """Validate immutable r10 state and all source hash closures.

    The data root is checked only as a directory identity.  This function never
    opens a formal neural file, a checkpoint, or an endpoint payload.
    """
    _require(STATIC_MANIFEST.exists(), "r10 static prelaunch manifest is absent")
    _require((STATIC_MANIFEST.stat().st_mode & 0o777) == 0o444, "r10 manifest must be 0444")
    payload = _read_json(STATIC_MANIFEST, label="static manifest")
    expected_keys = {
        "schema", "status", "protocol_id", "phase_id", "workspace_root", "data_root",
        "r10_cell_root", "arms_in_order", "stage_a_cells", "stage_b_cells", "gpu_by_fold",
        "epoch_selection_rule", "severe_negative_rule", "source_map", "source_map_sha256",
        "source_map_entry_count", "cost_receipt", "cost_supplement", "r9_retirement",
        "r9_retirement_seal", "owner_control_plane_simplification_override",
        "score_payload_mode", "formal_data_opened_by_prelaunch", "score_data_opened_by_prelaunch",
    }
    _require(set(payload) == expected_keys, "r10 static manifest exact-key mismatch")
    _require(payload.get("schema") == MANIFEST_SCHEMA, "r10 manifest schema mismatch")
    _require(payload.get("status") == "PREPARED_NOT_EXECUTED", "r10 manifest status mismatch")
    _require(payload.get("protocol_id") == PROTOCOL_ID and payload.get("phase_id") == PHASE_ID, "r10 protocol/phase mismatch")
    _require(payload.get("workspace_root") == str(ROOT.resolve()), "r10 workspace substitution")
    _require(payload.get("data_root") == str((ROOT / "SPINT-main/data/000953").resolve(strict=True)), "r10 data-root substitution")
    _require(payload.get("r10_cell_root") == str(R10_CELL_ROOT.resolve()), "r10 cell-root substitution")
    _require(payload.get("arms_in_order") == list(ARMS), "r10 arm order mismatch")
    schedule = _expected_cells()
    _require(payload.get("stage_a_cells") == [_key_record(key) for key in schedule["stage_a"]], "r10 Stage-A scope mismatch")
    _require(payload.get("stage_b_cells") == [_key_record(key) for key in schedule["stage_b"]], "r10 Stage-B scope mismatch")
    _require(
        payload.get("epoch_selection_rule")
        == "max_finite_equal_session_mean_then_earlier_epoch;spint_epochs_0_through_34;t4_epochs_0_through_11;paired_spint_decoder_31_tensors_bit_exact",
        "r10 epoch rule drift",
    )
    _require(payload.get("severe_negative_rule") == "(mean42 <= -0.03) OR (pos42 <= 1)", "r10 futility rule drift")
    _require(payload.get("score_payload_mode") == "opaque_payload_0600_no_stdout", "r10 payload mode drift")
    _require(payload.get("formal_data_opened_by_prelaunch") is False and payload.get("score_data_opened_by_prelaunch") is False, "r10 prelaunch scope drift")
    rows = payload.get("source_map")
    _require(isinstance(rows, list) and len(rows) == payload.get("source_map_entry_count"), "r10 source-map cardinality mismatch")
    _require(sha256_json(rows) == payload.get("source_map_sha256"), "r10 source-map digest mismatch")
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == {"relative_path", "canonical_path", "size_bytes", "sha256"}, "r10 source-map row schema")
        relative = row.get("relative_path")
        _require(isinstance(relative, str) and relative and not relative.startswith("/"), "r10 source relative path invalid")
        source = (ROOT / relative).resolve(strict=True)
        _require(str(source) == row.get("canonical_path"), "r10 source canonical path substitution")
        observed = file_metadata(source)
        _require({key: observed[key] for key in ("canonical_path", "size_bytes", "sha256")} == {key: row[key] for key in ("canonical_path", "size_bytes", "sha256")}, "r10 source hash drift")
    _require(OVERRIDE.is_file() and not OVERRIDE.is_symlink() and (OVERRIDE.stat().st_mode & 0o777) == 0o444, "r10 owner override must be immutable")
    _require(payload.get("owner_control_plane_simplification_override") == file_metadata(OVERRIDE), "r10 owner override binding drift")
    for field in ("r9_retirement", "r9_retirement_seal", "cost_receipt", "cost_supplement"):
        metadata = payload.get(field)
        _require(isinstance(metadata, Mapping), f"r10 {field} metadata absent")
        source = Path(str(metadata.get("canonical_path", "")))
        _require(source.is_file() and not source.is_symlink(), f"r10 {field} unavailable")
        observed = file_metadata(source)
        _require(dict(metadata) == observed, f"r10 {field} hash drift")
    if require_lock:
        _validate_run_lock()
    return payload


def _validate_run_lock() -> Mapping[str, Any]:
    _require(RUN_LOCK.is_file() and not RUN_LOCK.is_symlink(), "r10 execution requires the supervisor O_EXCL lock")
    _require((RUN_LOCK.stat().st_mode & 0o777) == 0o600, "r10 run lock must be 0600")
    lock = _read_json(RUN_LOCK, label="supervisor run lock")
    expected = {"schema", "status", "manifest", "cell_root", "score_opened_before_lock", "supervisor_pid"}
    _require(set(lock) == expected and lock.get("schema") == LOCK_SCHEMA and lock.get("status") == "RUNNING", "r10 run-lock schema/state mismatch")
    _require(lock.get("manifest") == manifest_metadata(), "r10 run-lock manifest mismatch")
    _require(lock.get("cell_root") == str(R10_CELL_ROOT.resolve()), "r10 run-lock root mismatch")
    _require(lock.get("score_opened_before_lock") is False, "r10 lock scope mismatch")
    return lock


def create_run_lock(*, supervisor_pid: int) -> Path:
    """Create the only mutable control record with O_EXCL, before a cell root."""
    manifest = validate_static_manifest(require_lock=False)
    _require(not R10_CELL_ROOT.exists(), "r10 fresh cell root already exists")
    RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": LOCK_SCHEMA,
        "status": "RUNNING",
        "manifest": file_metadata(STATIC_MANIFEST),
        "cell_root": str(R10_CELL_ROOT.resolve()),
        "score_opened_before_lock": False,
        "supervisor_pid": int(supervisor_pid),
    }
    del manifest
    write_json_exclusive(RUN_LOCK, payload)
    os.chmod(RUN_LOCK, 0o600)
    return RUN_LOCK


def _key_is_scheduled(key: CellKey) -> bool:
    return key in set(_expected_cells()["stage_a"]) | set(_expected_cells()["stage_b"])


def _expected_gpu(payload: Mapping[str, Any], key: CellKey) -> str:
    mapping = payload.get("gpu_by_fold")
    _require(isinstance(mapping, Mapping), "r10 gpu map absent")
    value = mapping.get(str(key.fold))
    _require(isinstance(value, str) and value in {"0", "1"}, "r10 gpu assignment invalid")
    return value


def _static_execution_evidence(root: Path, key: CellKey, manifest: Mapping[str, Any]) -> Path:
    destination = cell_paths(root, key)["execution_capability_evidence_run"]
    expected = {
        "schema": EXECUTION_EVIDENCE_SCHEMA,
        **key.identity(),
        "static_manifest": file_metadata(STATIC_MANIFEST),
        "run_lock": file_metadata(RUN_LOCK),
        "score_opened_before_execution": False,
        "authorization_mechanism": "static_manifest_single_supervisor",
    }
    del manifest
    if destination.exists():
        existing = _read_json(destination, label="static execution evidence")
        _require(existing == expected, "r10 execution evidence substitution")
        return destination
    return write_json_exclusive(destination, expected)


def require_cell_execution_capability(*, root: str | Path, key: CellKey, **unused: Any) -> Mapping[str, Any]:
    """Compatibility-shaped static gate used by r10 worker shims.

    Extra named arguments are deliberately ignored so the frozen r9 worker
    command-line ABI can be reused without an authentication side channel.
    """
    del unused
    manifest = validate_static_manifest(require_lock=True)
    cell_root = Path(root).resolve()
    _require(cell_root == R10_CELL_ROOT.resolve(), "r10 worker root differs from fresh r10 root")
    _require(_key_is_scheduled(key), "r10 worker cell lies outside the static 42-cell schedule")
    _require(os.environ.get(MANIFEST_ENV) == str(STATIC_MANIFEST.resolve()), "r10 worker manifest environment mismatch")
    raw_cell = os.environ.get(CELL_ENV)
    _require(raw_cell is not None and Path(raw_cell).resolve() == cell_paths(cell_root, key)["cell_dir"], "r10 worker cell environment mismatch")
    observed_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    _require(observed_gpu == _expected_gpu(manifest, key), "r10 worker CUDA_VISIBLE_DEVICES differs from static assignment")
    evidence = _static_execution_evidence(cell_root, key, manifest)
    return {
        "_validated_cost_supplement_path": manifest["cost_supplement"]["canonical_path"],
        "_execution_capability_evidence_path": str(evidence),
        "authorization_mechanism": "static_manifest_single_supervisor",
    }


def require_cell_execution_capability_from_environment(*, root: str | Path, key: CellKey) -> Mapping[str, Any]:
    return require_cell_execution_capability(root=root, key=key)


def require_r10_training_wrapper_pre_hydra_gate(argv: Sequence[str], *, arm: str) -> dict[str, str]:
    overrides = validate_phase_c_training_wrapper_argv(argv, arm=arm)
    raw_cell = Path(overrides["cell_paths.cell_dir"])
    environment_cell = Path(os.environ.get(CELL_ENV, ""))
    cell = require_canonical_directory(raw_cell)
    _require(environment_cell.exists() and require_canonical_directory(environment_cell) == cell, "r10 pre-Hydra cell env mismatch")
    try:
        root = cell.parents[5]
    except IndexError as exc:
        raise StaticControlError("r10 pre-Hydra cell path is malformed") from exc
    key = CellKey(PROTOCOL_ID, arm, int(overrides["data.loso_fold"]), int(overrides["seed"]))
    _require(cell_paths(root, key)["cell_dir"] == cell, "r10 pre-Hydra key/layout mismatch")
    require_cell_execution_capability(root=root, key=key)
    return overrides


def verify_r10_static_pre_hydra_only(argv: Sequence[str], *, arm: str) -> bool:
    if list(argv) != ["--verify-r10-static-pre-hydra-only"]:
        return False
    _require(arm in ARMS, "r10 smoke arm invalid")
    validate_static_manifest(require_lock=False)
    print(json.dumps({
        "schema": "m2_post33_phase_c_v5_r10_static_pre_hydra_smoke_v1",
        "status": "PASS_STATIC_MANIFEST_BEFORE_HYDRA",
        "arm": arm,
        "cell_root_created": R10_CELL_ROOT.exists(),
        "run_lock_created": RUN_LOCK.exists(),
        "hydra_initialized": False,
        "formal_data_accessed": False,
        "cuda_initialized": False,
        "score_or_r2_accessed": False,
    }, sort_keys=True))
    return True


# Names retained solely for the small import shim that executes the audited r9
# workers under r10's static guard.  They are not r9 authorization functions.
require_r9_training_wrapper_pre_hydra_gate = require_r10_training_wrapper_pre_hydra_gate


def verify_r9_training_wrapper_capability_only(argv: Sequence[str], *, arm: str) -> bool:
    """Translate the frozen r9 one-flag ABI to r10's score-free smoke flag."""
    if list(argv) == ["--verify-r9-pre-hydra-capability-only"]:
        return verify_r10_static_pre_hydra_only(["--verify-r10-static-pre-hydra-only"], arm=arm)
    return False
