"""Append-only source closure for the v5/r9 device-repair rollover.

The r9 root is deliberately a new authorization lineage, not a modification of
the retired r6d tree.  It preserves the Phase-C data/cell contract while
pinning the new full v5 evaluators, the explicit device-preparation helper,
and the r9 worker plumbing in one CPU-verifiable source map.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from sua_exploration.mc_maze.m2_native_post33_cost_v4 import (
    validate_deep_source_audit_receipt,
    validate_source_batch_audit,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    PHASE_ID,
    PROTOCOL_ID,
    file_metadata,
    require_canonical_regular_file,
    sha256_file,
    sha256_json,
)
from sua_exploration.mc_maze.m2_native_post33_program_v4 import (
    PROGRAM_EXPECTED_CLOSURE,
    validate_upstream_eof_canonicalization_receipt,
)


ROOT = Path(__file__).resolve().parents[2]
ROLLOVER_ID = "PHASE_C_V5_R9_DEVICE_REPAIR"
PROGRAM_SCHEMA = "m2_post33_phase_c_v5_r9_program_receipt_v1"
ANCHOR_SCHEMA = "m2_post33_phase_c_v5_r9_public_anchor_v1"
R9_RECEIPT_ROOT = (
    ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_launch_receipts_20260805"
)
R9_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_cells_20260805"
PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v5_r9_root_ed25519_public.pem"
PUBLIC_ANCHOR = R9_RECEIPT_ROOT / "authority/public_anchor.json"
R6D_RETIREMENT = (
    ROOT
    / "sua_exploration/results/m2_native_post33_phase_c_v5_r7_device_recovery_20260805"
    / "r6d_engineering_failure_retirement.json"
)
R6D_RETIREMENT_SEAL = R6D_RETIREMENT.with_name("r6d_engineering_failure_retirement.seal.json")
R7_PRELAUNCH_RETIREMENT = (
    ROOT
    / "sua_exploration/results/m2_native_post33_phase_c_v5_r7_launch_receipts_20260805"
    / "retirement/r7_prelaunch_matrix_contract_retirement.json"
)
R8_PRE_HYDRA_RETIREMENT = (
    ROOT
    / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_device_recovery_20260805"
    / "r8_pre_hydra_abi_engineering_failure_retirement.json"
)
R8_PRE_HYDRA_RETIREMENT_SEAL = R8_PRE_HYDRA_RETIREMENT.with_name(
    "r8_pre_hydra_abi_engineering_failure_retirement.seal.json"
)
EOF_RECEIPT = (
    ROOT
    / "sua_exploration/results/m2_native_post33_phase_c_v4_upstream_canonicalization_20260805"
    / "upstream_eof_canonicalization.json"
)
DEEP_SOURCE_AUDIT = (
    ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_deep_source_20260805/deep_source_audit.json"
)
PHASE_A_DATA_AUDIT = (
    ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_scorefree_audit_20260804/audit.json"
)


# The v4 closure supplies the complete frozen model/config/source graph.  r9
# removes every old executable authority path and substitutes only audited r9
# worker paths.  We retain ``program_v4`` because it provides the immutable
# upstream EOF proof validator used by this CPU-only closure, not as a runtime
# authorization route.
_REMOVE_FROM_V4_CLOSURE = {
    "SPINT-main/src/train_post33_phase_c_v4.py",
    "SPINT-main/src/evaluate_post33_phase_c_v4.py",
    "streaming_calibration_exp/src/train_post33_phase_c_v4.py",
    "streaming_calibration_exp/src/evaluate_post33_phase_c_v4.py",
    "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py",
    "sua_exploration/mc_maze/m2_native_post33_openers_v4.py",
}
_REMOVE_FROM_V4_CLOSURE.update(
    path for path in PROGRAM_EXPECTED_CLOSURE if path.startswith("sua_exploration/scripts/")
)
R9_RUNTIME_SOURCES = frozenset(
    {
        "SPINT-main/src/train_post33_phase_c_v5_r9.py",
        "SPINT-main/src/evaluate_post33_phase_c_v5.py",
        "SPINT-main/src/evaluate_post33_phase_c_v5_r9.py",
        "streaming_calibration_exp/src/train_post33_phase_c_v5_r9.py",
        "streaming_calibration_exp/src/evaluate_post33_phase_c_v5.py",
        "streaming_calibration_exp/src/evaluate_post33_phase_c_v5_r9.py",
        "sua_exploration/mc_maze/m2_native_post33_deployment_device_prep_v5.py",
        "sua_exploration/mc_maze/m2_native_post33_phase_c_v5_r9_program.py",
        "sua_exploration/mc_maze/m2_native_post33_authorization_v5_r9.py",
        "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v5_r9.py",
        "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r9_cell_pipeline.py",
        "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r9_matrix.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v5_r9_prepare_stage_a.py",
        "sua_exploration/scripts/verify_m2_native_post33_phase_c_v5_r9.py",
    }
)
PROGRAM_SOURCE_PATHS = tuple(
    sorted((set(PROGRAM_EXPECTED_CLOSURE) - _REMOVE_FROM_V4_CLOSURE) | R9_RUNTIME_SOURCES)
)


def _metadata(path: str | Path) -> dict[str, Any]:
    return file_metadata(path)


def _read_json(path: str | Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    source = require_canonical_regular_file(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return source, payload


def validate_r6d_retirement() -> dict[str, Any]:
    """Bind the old lineage's sealed retirement without reading old cell data."""
    receipt = require_canonical_regular_file(R6D_RETIREMENT, within=ROOT)
    seal = require_canonical_regular_file(R6D_RETIREMENT_SEAL, within=ROOT)
    if (receipt.stat().st_mode & 0o777) != 0o444 or (seal.stat().st_mode & 0o777) != 0o444:
        raise PermissionError("r6d retirement receipt/seal must remain immutable mode 0444")
    _, retirement = _read_json(receipt, "r6d retirement receipt")
    _, sealed = _read_json(seal, "r6d retirement seal")
    if (
        retirement.get("schema")
        != "m2_post33_phase_c_v5_r7_r6d_engineering_failure_retirement_v1"
        or retirement.get("r6d_reuse_forbidden") is not True
        or retirement.get("formal_gpu_rerun_authorized_by_this_receipt") is not False
        or retirement.get("fresh_r7_root_nonce_selector_program_and_authorization_required") is not True
    ):
        raise PermissionError("r6d retirement does not require a fresh r9 lineage")
    expected_target = _metadata(receipt)
    if (
        sealed.get("schema") != "m2_post33_phase_c_v5_r7_r6d_retirement_receipt_seal_v1"
        or sealed.get("target") != expected_target
        or sealed.get("target_content_bytes_unchanged") is not True
        or sealed.get("target_mode_after") != "0444"
    ):
        raise PermissionError("r6d retirement seal mismatch")
    return {"retirement": expected_target, "seal": _metadata(seal)}


def validate_r7_prelaunch_retirement() -> dict[str, Any]:
    """Require the signed-but-unlaunched r7 lineage to be explicitly dead."""
    receipt = require_canonical_regular_file(R7_PRELAUNCH_RETIREMENT, within=ROOT)
    if (receipt.stat().st_mode & 0o777) != 0o444:
        raise PermissionError("r7 prelaunch retirement must remain immutable mode 0444")
    _, payload = _read_json(receipt, "r7 prelaunch retirement")
    expected = {
        "schema": "m2_post33_phase_c_v5_r7_prelaunch_matrix_contract_retirement_v1",
        "status": "RETIRED_NEVER_LAUNCH",
        "cell_root_exists": False,
        "selector_records_materialized": False,
        "authorization_nonce_claim_materialized": False,
        "worker_started": False,
        "cuda_initialized": False,
        "gpu_used": False,
        "formal_data_accessed": False,
        "score_data_accessed": False,
        "r7_capabilities_must_never_be_launched": True,
        "fresh_successor_root_key_program_auth_nonce_selector_required": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise PermissionError("r7 prelaunch retirement scope/immutability mismatch")
    return {"retirement": _metadata(receipt)}


def validate_r8_pre_hydra_retirement() -> dict[str, Any]:
    """Bind the sealed r8 ABI failure without reopening r8 result artifacts.

    r8 consumed two nonce claims but never reached Hydra, data construction,
    CUDA, or an endpoint.  Its immutable retirement receipt is therefore the
    only permitted r8 input to the fresh r9 authority lineage.
    """
    receipt = require_canonical_regular_file(R8_PRE_HYDRA_RETIREMENT, within=ROOT)
    seal = require_canonical_regular_file(R8_PRE_HYDRA_RETIREMENT_SEAL, within=ROOT)
    if (receipt.stat().st_mode & 0o777) != 0o444 or (seal.stat().st_mode & 0o777) != 0o444:
        raise PermissionError("r8 pre-Hydra retirement receipt/seal must remain immutable mode 0444")
    _, payload = _read_json(receipt, "r8 pre-Hydra retirement receipt")
    _, sealed = _read_json(seal, "r8 pre-Hydra retirement seal")
    expected = {
        "schema": "m2_post33_phase_c_v5_r9_r8_pre_hydra_abi_engineering_failure_retirement_v1",
        "status": "RETIRED_NEVER_RELAUNCH",
        "retired_lineage": "PHASE_C_V5_R8_DEVICE_REPAIR",
        "endpoint_score_or_r2_opened": False,
        "hydra_initialized": False,
        "data_module_constructed": False,
        "cuda_initialized": False,
        "gpu_compute_started": False,
        "formal_data_accessed": False,
        "score_data_accessed": False,
        "r8_capabilities_must_never_be_relaunched": True,
        "fresh_r9_root_key_program_auth_nonce_selector_required": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise PermissionError("r8 pre-Hydra retirement scope/immutability mismatch")
    if (
        sealed.get("schema") != "m2_post33_phase_c_v5_r9_r8_pre_hydra_abi_retirement_seal_v1"
        or sealed.get("target") != _metadata(receipt)
        or sealed.get("target_content_bytes_unchanged") is not True
        or sealed.get("target_mode_after") != "0444"
        or sealed.get("endpoint_score_or_r2_opened") is not False
    ):
        raise PermissionError("r8 pre-Hydra retirement seal mismatch")
    return {"retirement": _metadata(receipt), "seal": _metadata(seal)}


def validate_public_anchor() -> dict[str, Any]:
    """Validate the fixed r9 public key path and append-only anchor receipt."""
    pem = require_canonical_regular_file(PUBLIC_KEY, within=ROOT)
    anchor_path, anchor = _read_json(PUBLIC_ANCHOR, "r9 public anchor")
    expected_keys = {"schema", "rollover_id", "protocol_id", "phase_id", "public_key", "private_key_serialized_or_disk_persisted", "gpu_used", "formal_data_accessed", "score_data_accessed"}
    if set(anchor) != expected_keys or anchor.get("schema") != ANCHOR_SCHEMA:
        raise PermissionError("r9 public anchor schema mismatch")
    if (
        anchor.get("rollover_id") != ROLLOVER_ID
        or anchor.get("protocol_id") != PROTOCOL_ID
        or anchor.get("phase_id") != PHASE_ID
        or anchor.get("public_key") != _metadata(pem)
        or anchor.get("private_key_serialized_or_disk_persisted") is not False
        or any(anchor.get(field) is not False for field in ("gpu_used", "formal_data_accessed", "score_data_accessed"))
    ):
        raise PermissionError("r9 public anchor identity/scope mismatch")
    return {"anchor": _metadata(anchor_path), "public_key": _metadata(pem)}


def source_rows() -> list[dict[str, Any]]:
    """Return a deterministic current hash map; no raw dataset is opened."""
    rows: list[dict[str, Any]] = []
    for relative in PROGRAM_SOURCE_PATHS:
        source = require_canonical_regular_file(ROOT / relative, within=ROOT)
        rows.append({"relative_path": relative, **_metadata(source)})
    return rows


def _validate_upstream_receipts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    eof_path, eof = _read_json(EOF_RECEIPT, "upstream EOF canonicalization receipt")
    validate_upstream_eof_canonicalization_receipt(eof)
    deep_path, deep = _read_json(DEEP_SOURCE_AUDIT, "deep source audit receipt")
    audit_meta = deep.get("source_batch_audit")
    if not isinstance(audit_meta, Mapping):
        raise ValueError("deep source audit lacks source-batch audit metadata")
    validate_deep_source_audit_receipt(
        deep, source_batch_audit_path=audit_meta.get("canonical_path", ""), deep_verify=False
    )
    phase_a_path, phase_a = _read_json(PHASE_A_DATA_AUDIT, "Phase-A data audit")
    execution_scope = phase_a.get("execution_scope")
    if (
        phase_a.get("protocol_id") != PROTOCOL_ID
        or phase_a.get("status") != "PASS_SCORE_FREE_CPU_AUDIT"
        or not isinstance(execution_scope, Mapping)
        or any(execution_scope.get(field) not in {0, False} for field in (
            "evalai_calls", "formal_sua_files_opened", "formal_sua_paths_resolved",
            "gpu_used", "new_endpoint_r2_values_read", "scorer_modules_imported",
            "training_started",
        ))
    ):
        raise ValueError("Phase-A data audit is not the sealed score-free scope")
    return _metadata(eof_path), _metadata(deep_path), _metadata(phase_a_path)


def build_r9_program_receipt() -> dict[str, Any]:
    """Build the unsigned, source-only r9 receipt after the public anchor exists."""
    anchor = validate_public_anchor()
    retirement = validate_r6d_retirement()
    r7_retirement = validate_r7_prelaunch_retirement()
    r8_retirement = validate_r8_pre_hydra_retirement()
    eof, deep, phase_a = _validate_upstream_receipts()
    rows = source_rows()
    return {
        "schema": PROGRAM_SCHEMA,
        "rollover_id": ROLLOVER_ID,
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "absolute_workspace_root": str(ROOT.resolve()),
        "r9_receipt_root": str(R9_RECEIPT_ROOT.resolve()),
        "r9_cell_root": str(R9_CELL_ROOT.resolve()),
        "public_anchor": anchor,
        "r6d_retirement": retirement,
        "r7_prelaunch_retirement": r7_retirement,
        "r8_pre_hydra_retirement": r8_retirement,
        "phase_a_b_eof_canonicalization": eof,
        "deep_source_audit_receipt": deep,
        "phase_a_data_audit": phase_a,
        "source_map": rows,
        "source_map_sha256": sha256_json(rows),
        "source_map_entry_count": len(rows),
        "gpu_used": False,
        "formal_data_accessed": False,
        "score_data_accessed": False,
    }


def validate_r9_program_receipt(path: str | Path) -> Mapping[str, Any]:
    """Fail closed on source, trust-anchor, or old-lineage drift."""
    receipt_path, receipt = _read_json(path, "r9 program receipt")
    required = {
        "schema", "rollover_id", "protocol_id", "phase_id", "absolute_workspace_root",
        "r9_receipt_root", "r9_cell_root", "public_anchor", "r6d_retirement", "r7_prelaunch_retirement",
        "r8_pre_hydra_retirement",
        "phase_a_b_eof_canonicalization", "deep_source_audit_receipt", "phase_a_data_audit",
        "source_map", "source_map_sha256", "source_map_entry_count", "gpu_used",
        "formal_data_accessed", "score_data_accessed",
    }
    if set(receipt) != required or receipt.get("schema") != PROGRAM_SCHEMA:
        raise ValueError("r9 program receipt schema/exact-set mismatch")
    if (
        receipt.get("rollover_id") != ROLLOVER_ID
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("phase_id") != PHASE_ID
        or receipt.get("absolute_workspace_root") != str(ROOT.resolve())
        or receipt.get("r9_receipt_root") != str(R9_RECEIPT_ROOT.resolve())
        or receipt.get("r9_cell_root") != str(R9_CELL_ROOT.resolve())
        or any(receipt.get(field) is not False for field in ("gpu_used", "formal_data_accessed", "score_data_accessed"))
    ):
        raise ValueError("r9 program receipt identity/scope mismatch")
    if receipt.get("public_anchor") != validate_public_anchor():
        raise ValueError("r9 program receipt public-anchor drift")
    if receipt.get("r6d_retirement") != validate_r6d_retirement():
        raise ValueError("r9 program receipt retirement binding drift")
    if receipt.get("r7_prelaunch_retirement") != validate_r7_prelaunch_retirement():
        raise ValueError("r9 program receipt r7-retirement binding drift")
    if receipt.get("r8_pre_hydra_retirement") != validate_r8_pre_hydra_retirement():
        raise ValueError("r9 program receipt r8-retirement binding drift")
    eof, deep, phase_a = _validate_upstream_receipts()
    if (
        receipt.get("phase_a_b_eof_canonicalization") != eof
        or receipt.get("deep_source_audit_receipt") != deep
        or receipt.get("phase_a_data_audit") != phase_a
    ):
        raise ValueError("r9 program receipt upstream closure drift")
    del receipt_path
    return receipt
