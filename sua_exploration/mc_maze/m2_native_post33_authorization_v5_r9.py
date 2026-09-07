"""Fresh, Stage-A-only detached-Ed25519 capability contract for v5/r9.

This module deliberately does not accept any r6d anchor, nonce, evaluator, or
cell root.  It authorizes exactly the repaired v5 worker lineage and only the
seed-42 Stage-A shards; it contains no Stage-B or score-opening capability.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import socket
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from sua_exploration.mc_maze.m2_native_post33_cost_v4 import validate_cost_supplement
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    ARMS,
    CellKey,
    FOLDS,
    PHASE_ID,
    PROTOCOL_ID,
    cell_paths,
    file_metadata,
    phase_root,
    require_canonical_directory,
    require_canonical_regular_file,
    sha256_file,
    validate_phase_c_training_wrapper_argv,
    write_json_exclusive,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r9_program import (
    PUBLIC_KEY,
    R9_CELL_ROOT,
    R9_RECEIPT_ROOT,
    ROLLOVER_ID,
    validate_public_anchor,
    validate_r6d_retirement,
    validate_r7_prelaunch_retirement,
    validate_r8_pre_hydra_retirement,
    validate_r9_program_receipt,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v5_r9.py"
AUTH_SCHEMA = "m2_post33_phase_c_v5_r9_gpu_authorization_v1"
ENVELOPE_SCHEMA = "m2_post33_phase_c_v5_r9_signed_authorization_envelope_v1"
PORTABLE_SCHEMA = "m2_post33_phase_c_v5_r9_portable_manifest_v1"
SHARD_SCHEMA = "m2_post33_phase_c_v5_r9_shard_manifest_v1"
CLAIM_SCHEMA = "m2_post33_phase_c_v5_r9_authorization_nonce_claim_v1"
CAPABILITY_EVIDENCE_SCHEMA = "m2_post33_phase_c_v5_r9_execution_capability_evidence_v1"
SELECTOR_POLICY_SCHEMA = "m2_post33_phase_c_v5_r9_fresh_selector_policy_v1"
MAX_VALIDITY_SECONDS = 30 * 3600
CAPABILITY_SCOPE_CELL_EXECUTION = "cell_execution"
CAPABILITY_ENVIRONMENT = {
    "authorization_path": "M2_POST33_PHASE_C_V5_R9_AUTHORIZATION",
    "signature_path": "M2_POST33_PHASE_C_V5_R9_AUTHORIZATION_SIGNATURE",
    "phase_c_program_receipt_path": "M2_POST33_PHASE_C_V5_R9_PROGRAM_RECEIPT",
    "portable_manifest_path": "M2_POST33_PHASE_C_V5_R9_PORTABLE_MANIFEST",
    "shard_manifest_path": "M2_POST33_PHASE_C_V5_R9_SHARD_MANIFEST",
    "cost_supplement_path": "M2_POST33_PHASE_C_V5_R9_COST_SUPPLEMENT",
}
SPINT_SELECTOR_SOURCE = ROOT / "SPINT-main/src/callbacks/post33_source_selector_v4.py"
T4_SELECTOR_SOURCE = ROOT / "streaming_calibration_exp/src/callbacks/post33_t4_source_selector_v4.py"


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise PermissionError(f"r9 authorization {label} must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PermissionError(f"r9 authorization {label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _read_json(path: str | Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    source = require_canonical_regular_file(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError(f"r9 {label} is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise PermissionError(f"r9 {label} must be a mapping")
    return source, payload


def _verify_metadata(metadata: Any, expected: str | Path, label: str) -> Path:
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "canonical_path", "size_bytes", "sha256"
    }:
        raise PermissionError(f"r9 {label} metadata exact set mismatch")
    target = require_canonical_regular_file(expected)
    if dict(metadata) != file_metadata(target):
        raise PermissionError(f"r9 {label} substitution/hash drift")
    return target


def selector_policy() -> dict[str, Any]:
    return {
        "schema": SELECTOR_POLICY_SCHEMA,
        "selector_records_reuse_forbidden": True,
        "selector_created_only_by_authorized_training": True,
        "selector_sources": {
            "spint": file_metadata(SPINT_SELECTOR_SOURCE),
            "t4": file_metadata(T4_SELECTOR_SOURCE),
        },
    }


def validate_observed_host(shard: Mapping[str, Any], *, observed_host: str | None = None) -> str:
    observed = observed_host if observed_host is not None else socket.gethostname()
    if not isinstance(observed, str) or not observed or shard.get("host_id") != observed:
        raise PermissionError("r9 signed shard host_id differs from observed hostname")
    return observed


def validate_observed_gpu(
    shard: Mapping[str, Any], *, observed_visible_devices: str | None = None
) -> dict[str, Any]:
    raw = observed_visible_devices if observed_visible_devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(raw, str) or not raw:
        raise PermissionError("r9 signed GPU requires exactly one visible physical device")
    visible = [part.strip() for part in raw.split(",")]
    if len(visible) != 1 or not visible[0] or raw != visible[0] or visible[0] != shard.get("gpu_id"):
        raise PermissionError("r9 CUDA_VISIBLE_DEVICES does not equal signed physical GPU")
    return {
        "cuda_visible_devices": raw,
        "logical_cuda_device_index": 0,
        "physical_gpu_id": visible[0],
    }


def validate_r9_portable_transfer_manifest(
    manifest_path: str | Path,
    *,
    workspace_root: str | Path,
    data_root: str | Path,
    cell_root: str | Path,
) -> Mapping[str, Any]:
    path, manifest = _read_json(manifest_path, "portable manifest")
    required = {
        "schema", "rollover_id", "protocol_id", "phase_id", "workspace_root",
        "data_root", "absolute_cell_root", "same_absolute_paths_required_on_all_hosts",
        "program", "deep_source_audit", "phase_a_data_audit", "r6d_retirement",
        "r7_prelaunch_retirement", "r8_pre_hydra_retirement",
        "selector_policy",
    }
    if set(manifest) != required or manifest.get("schema") != PORTABLE_SCHEMA:
        raise PermissionError("r9 portable manifest schema/exact-set mismatch")
    if (
        manifest.get("rollover_id") != ROLLOVER_ID
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("phase_id") != PHASE_ID
        or manifest.get("workspace_root") != str(Path(workspace_root).resolve(strict=True))
        or manifest.get("data_root") != str(Path(data_root).resolve(strict=True))
        or manifest.get("absolute_cell_root") != str(Path(cell_root).resolve())
        or manifest.get("same_absolute_paths_required_on_all_hosts") is not True
        or manifest.get("selector_policy") != selector_policy()
    ):
        raise PermissionError("r9 portable manifest identity/path/selector mismatch")
    program = _verify_metadata(manifest.get("program"), manifest.get("program", {}).get("canonical_path", ""), "program")
    validated_program = validate_r9_program_receipt(program)
    for field, program_field in (
        ("deep_source_audit", "deep_source_audit_receipt"),
        ("phase_a_data_audit", "phase_a_data_audit"),
    ):
        metadata = manifest.get(field)
        if metadata != validated_program.get(program_field):
            raise PermissionError(f"r9 portable {field} does not match program receipt")
        _verify_metadata(metadata, metadata.get("canonical_path", ""), field)
    if manifest.get("r6d_retirement") != validated_program.get("r6d_retirement"):
        raise PermissionError("r9 portable r6d-retirement binding mismatch")
    if manifest.get("r7_prelaunch_retirement") != validated_program.get("r7_prelaunch_retirement"):
        raise PermissionError("r9 portable r7-retirement binding mismatch")
    if manifest.get("r8_pre_hydra_retirement") != validated_program.get("r8_pre_hydra_retirement"):
        raise PermissionError("r9 portable r8-pre-Hydra-retirement binding mismatch")
    validate_r6d_retirement()
    validate_r7_prelaunch_retirement()
    validate_r8_pre_hydra_retirement()
    del path
    return manifest


def validate_r9_shard_manifest(
    shard_path: str | Path,
    *,
    portable_manifest_path: str | Path,
    cell_root: str | Path,
) -> Mapping[str, Any]:
    _, shard = _read_json(shard_path, "shard manifest")
    required = {
        "schema", "rollover_id", "protocol_id", "phase_id", "host_id", "gpu_id",
        "arms_in_order", "paired_same_host_required", "absolute_cell_root",
        "fold_allowlist", "seed_allowlist", "portable_manifest_sha256",
    }
    if set(shard) != required or shard.get("schema") != SHARD_SCHEMA:
        raise PermissionError("r9 shard manifest schema/exact-set mismatch")
    folds, seeds = shard.get("fold_allowlist"), shard.get("seed_allowlist")
    if (
        shard.get("rollover_id") != ROLLOVER_ID
        or shard.get("protocol_id") != PROTOCOL_ID
        or shard.get("phase_id") != PHASE_ID
        or shard.get("arms_in_order") != list(ARMS)
        or shard.get("paired_same_host_required") is not True
        or shard.get("absolute_cell_root") != str(Path(cell_root).resolve())
        or not isinstance(folds, list)
        or sorted(folds) != folds
        or len(set(folds)) != len(folds)
        or any(isinstance(fold, bool) or fold not in FOLDS for fold in folds)
        or seeds != [42]
        or not isinstance(shard.get("host_id"), str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(shard.get("host_id")))
        or not isinstance(shard.get("gpu_id"), str)
        or not shard.get("gpu_id")
        or any(ch.isspace() for ch in str(shard.get("gpu_id")))
    ):
        raise PermissionError("r9 shard scope/identity mismatch")
    portable = require_canonical_regular_file(portable_manifest_path)
    if shard.get("portable_manifest_sha256") != sha256_file(portable):
        raise PermissionError("r9 shard portable-manifest substitution")
    return shard


def _verify_signature(authorization_path: str | Path, signature_path: str | Path) -> Mapping[str, Any]:
    authorization_file = require_canonical_regular_file(authorization_path)
    signature_file = require_canonical_regular_file(signature_path)
    _, envelope = _read_json(authorization_file, "authorization envelope")
    if set(envelope) != {"schema", "authorization"} or envelope.get("schema") != ENVELOPE_SCHEMA:
        raise PermissionError("r9 authorization envelope mismatch")
    auth = envelope.get("authorization")
    if not isinstance(auth, Mapping):
        raise PermissionError("r9 authorization body must be a mapping")
    return auth


def verify_signed_authorization(
    authorization_path: str | Path,
    signature_path: str | Path,
    *,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cell_root: str | Path,
    cost_supplement_path: str | Path | None = None,
    now: datetime | None = None,
) -> Mapping[str, Any]:
    """Verify the new anchor plus an exact, 14-cell-only Stage-A scope."""
    auth = _verify_signature(authorization_path, signature_path)
    required = {
        "schema", "rollover_id", "protocol_id", "phase_id", "status", "authorization_id",
        "single_use_nonce", "issued_at", "expires_at", "stage", "capability_scope",
        "host_id", "gpu_id", "absolute_cell_root", "fold_allowlist", "seed_allowlist",
        "arms_in_order", "phase_c_program_receipt", "portable_manifest", "shard_manifest",
        "evaluator", "cost_receipt", "cost_supplement", "public_key", "selector_policy",
    }
    if set(auth) != required or auth.get("schema") != AUTH_SCHEMA:
        raise PermissionError("r9 authorization schema/exact-set mismatch")
    if (
        auth.get("rollover_id") != ROLLOVER_ID
        or auth.get("protocol_id") != PROTOCOL_ID
        or auth.get("phase_id") != PHASE_ID
        or auth.get("status") != "GO"
        or auth.get("stage") != "stage_a"
        or auth.get("capability_scope") != CAPABILITY_SCOPE_CELL_EXECUTION
        or auth.get("absolute_cell_root") != str(Path(cell_root).resolve())
        or auth.get("arms_in_order") != list(ARMS)
        or auth.get("seed_allowlist") != [42]
        or auth.get("selector_policy") != selector_policy()
    ):
        raise PermissionError("r9 authorization scope/identity mismatch")
    if not isinstance(auth.get("authorization_id"), str) or not auth["authorization_id"]:
        raise PermissionError("r9 authorization id missing")
    nonce = auth.get("single_use_nonce")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise PermissionError("r9 authorization nonce must be 256-bit lowercase hex")
    issued, expires = _parse_utc(auth.get("issued_at"), "issued_at"), _parse_utc(auth.get("expires_at"), "expires_at")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not (issued <= current <= expires) or (expires - issued).total_seconds() > MAX_VALIDITY_SECONDS:
        raise PermissionError("r9 authorization expired/not-yet-valid/exceeds 30h")
    program = _verify_metadata(auth.get("phase_c_program_receipt"), phase_c_program_receipt_path, "program receipt")
    validate_r9_program_receipt(program)
    portable = _verify_metadata(auth.get("portable_manifest"), portable_manifest_path, "portable manifest")
    validate_r9_portable_transfer_manifest(
        portable,
        workspace_root=ROOT,
        data_root=ROOT / "SPINT-main/data/000953",
        cell_root=cell_root,
    )
    shard_file = _verify_metadata(auth.get("shard_manifest"), shard_manifest_path, "shard manifest")
    shard = validate_r9_shard_manifest(shard_file, portable_manifest_path=portable, cell_root=cell_root)
    if (
        auth.get("host_id") != shard.get("host_id")
        or auth.get("gpu_id") != shard.get("gpu_id")
        or auth.get("fold_allowlist") != shard.get("fold_allowlist")
        or auth.get("seed_allowlist") != shard.get("seed_allowlist")
        or auth.get("arms_in_order") != shard.get("arms_in_order")
    ):
        raise PermissionError("r9 authorization/shard scope mismatch")
    _verify_metadata(auth.get("evaluator"), EVALUATOR, "outer evaluator")
    _verify_metadata(auth.get("public_key"), PUBLIC_KEY, "public key")
    cost = _verify_metadata(auth.get("cost_receipt"), auth.get("cost_receipt", {}).get("canonical_path", ""), "cost receipt")
    expected_supplement = cost_supplement_path or auth.get("cost_supplement", {}).get("canonical_path", "")
    supplement = _verify_metadata(auth.get("cost_supplement"), expected_supplement, "cost supplement")
    _, supplement_payload = _read_json(supplement, "cost supplement")
    validate_cost_supplement(supplement_payload)
    return {
        **dict(auth),
        "_validated_program_path": str(program),
        "_validated_portable_path": str(portable),
        "_validated_shard_path": str(shard_file),
        "_validated_cost_receipt_path": str(cost),
        "_validated_cost_supplement_path": str(supplement),
    }


def execution_capability_environment(
    *,
    authorization_path: str | Path,
    signature_path: str | Path,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cost_supplement_path: str | Path,
) -> dict[str, str]:
    inputs = {
        "authorization_path": authorization_path,
        "signature_path": signature_path,
        "phase_c_program_receipt_path": phase_c_program_receipt_path,
        "portable_manifest_path": portable_manifest_path,
        "shard_manifest_path": shard_manifest_path,
        "cost_supplement_path": cost_supplement_path,
    }
    return {
        CAPABILITY_ENVIRONMENT[name]: str(require_canonical_regular_file(path))
        for name, path in inputs.items()
    }


def _execution_capability_inputs_from_environment() -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for field, variable in CAPABILITY_ENVIRONMENT.items():
        value = os.environ.get(variable)
        if not value:
            raise PermissionError(f"missing r9 capability environment: {variable}")
        inputs[field] = require_canonical_regular_file(value)
    return inputs


def claim_authorization_nonce(
    *,
    root: str | Path,
    authorization: Mapping[str, Any],
    authorization_path: str | Path,
    signature_path: str | Path,
    shard_manifest_path: str | Path,
) -> Path:
    if Path(root).resolve() != R9_CELL_ROOT.resolve():
        raise PermissionError("r9 capability may only claim its fresh cell root")
    host = authorization.get("host_id")
    nonce = authorization.get("single_use_nonce")
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", host):
        raise PermissionError("r9 nonce-claim hostname is unsafe")
    destination = phase_root(root) / "authorization_claims" / host / f"{nonce}.json"
    return write_json_exclusive(
        destination,
        {
            "schema": CLAIM_SCHEMA,
            "rollover_id": ROLLOVER_ID,
            "protocol_id": PROTOCOL_ID,
            "phase_id": PHASE_ID,
            "authorization_id": authorization["authorization_id"],
            "single_use_nonce": nonce,
            "stage": "stage_a",
            "capability_scope": CAPABILITY_SCOPE_CELL_EXECUTION,
            "host_id": host,
            "gpu_id": authorization["gpu_id"],
            "absolute_cell_root": str(Path(root).resolve()),
            "authorization": file_metadata(authorization_path),
            "signature": file_metadata(signature_path),
            "shard_manifest": file_metadata(shard_manifest_path),
            "phase_c_program_receipt": file_metadata(authorization["_validated_program_path"]),
            "portable_manifest": file_metadata(authorization["_validated_portable_path"]),
            "cost_supplement": file_metadata(authorization["_validated_cost_supplement_path"]),
        },
    )


def verify_authorization_nonce_claim(
    *,
    root: str | Path,
    authorization: Mapping[str, Any],
    authorization_path: str | Path,
    signature_path: str | Path,
    shard_manifest_path: str | Path,
) -> Path:
    host, nonce = authorization["host_id"], authorization["single_use_nonce"]
    path = require_canonical_regular_file(phase_root(root) / "authorization_claims" / host / f"{nonce}.json")
    _, claim = _read_json(path, "nonce claim")
    expected = {
        "schema": CLAIM_SCHEMA,
        "rollover_id": ROLLOVER_ID,
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "authorization_id": authorization["authorization_id"],
        "single_use_nonce": nonce,
        "stage": "stage_a",
        "capability_scope": CAPABILITY_SCOPE_CELL_EXECUTION,
        "host_id": host,
        "gpu_id": authorization["gpu_id"],
        "absolute_cell_root": str(Path(root).resolve()),
        "authorization": file_metadata(authorization_path),
        "signature": file_metadata(signature_path),
        "shard_manifest": file_metadata(shard_manifest_path),
        "phase_c_program_receipt": file_metadata(authorization["_validated_program_path"]),
        "portable_manifest": file_metadata(authorization["_validated_portable_path"]),
        "cost_supplement": file_metadata(authorization["_validated_cost_supplement_path"]),
    }
    if claim != expected:
        raise PermissionError("r9 nonce claim substitution or wrong bound files")
    return path


def _write_or_verify_execution_evidence(
    *,
    root: str | Path,
    key: CellKey,
    authorization: Mapping[str, Any],
    authorization_path: str | Path,
    signature_path: str | Path,
    shard_manifest_path: str | Path,
    claim_path: str | Path,
    gpu_observation: Mapping[str, Any],
) -> Path:
    destination = cell_paths(root, key)["execution_capability_evidence_run"]
    expected = {
        "schema": CAPABILITY_EVIDENCE_SCHEMA,
        "rollover_id": ROLLOVER_ID,
        **key.identity(),
        "authorization_id": authorization["authorization_id"],
        "single_use_nonce": authorization["single_use_nonce"],
        "stage": "stage_a",
        "capability_scope": CAPABILITY_SCOPE_CELL_EXECUTION,
        "observed_host_id": authorization["host_id"],
        "observed_gpu": dict(gpu_observation),
        "absolute_cell_root": str(Path(root).resolve()),
        "authorization": file_metadata(authorization_path),
        "signature": file_metadata(signature_path),
        "shard_manifest": file_metadata(shard_manifest_path),
        "nonce_claim": file_metadata(claim_path),
    }
    if destination.exists():
        _, existing = _read_json(destination, "execution capability evidence")
        if existing != expected:
            raise PermissionError("r9 execution capability evidence substitution")
        return destination
    return write_json_exclusive(destination, expected)


def require_cell_execution_capability(
    *,
    root: str | Path,
    key: CellKey,
    authorization_path: str | Path,
    signature_path: str | Path,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cost_supplement_path: str | Path,
) -> Mapping[str, Any]:
    if Path(root).resolve() != R9_CELL_ROOT.resolve():
        raise PermissionError("r9 execution may only use the fresh r9 cell root")
    authorization = verify_signed_authorization(
        authorization_path,
        signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
        cell_root=root,
    )
    shard = validate_r9_shard_manifest(
        shard_manifest_path, portable_manifest_path=portable_manifest_path, cell_root=root
    )
    validate_observed_host(shard)
    observation = validate_observed_gpu(shard)
    if (
        key.arm not in shard["arms_in_order"]
        or key.fold not in shard["fold_allowlist"]
        or key.seed not in shard["seed_allowlist"]
    ):
        raise PermissionError("cell is outside r9 signed shard allowlist")
    claim = verify_authorization_nonce_claim(
        root=root,
        authorization=authorization,
        authorization_path=authorization_path,
        signature_path=signature_path,
        shard_manifest_path=shard_manifest_path,
    )
    evidence = _write_or_verify_execution_evidence(
        root=root,
        key=key,
        authorization=authorization,
        authorization_path=authorization_path,
        signature_path=signature_path,
        shard_manifest_path=shard_manifest_path,
        claim_path=claim,
        gpu_observation=observation,
    )
    return {
        **authorization,
        "_validated_nonce_claim_path": str(claim),
        "_execution_capability_evidence_path": str(evidence),
        "_observed_gpu": observation,
    }


def require_cell_execution_capability_from_environment(
    *, root: str | Path, key: CellKey
) -> Mapping[str, Any]:
    return require_cell_execution_capability(root=root, key=key, **_execution_capability_inputs_from_environment())


def require_r9_training_wrapper_pre_hydra_gate(
    argv: Sequence[str], *, arm: str
) -> dict[str, str]:
    """Bind the raw wrapper gate directly to the fresh r9 capability ABI.

    The historical helper is intentionally not reused here: it imports the v4
    authorization module internally and therefore cannot verify an r9
    capability environment.  This function keeps the frozen argv/layout checks
    but terminates the call graph at r9's verifier before Hydra can initialize.
    """
    overrides = validate_phase_c_training_wrapper_argv(argv, arm=arm)
    raw_cell = Path(overrides["cell_paths.cell_dir"])
    environment_cell_value = os.environ.get("M2_POST33_PHASE_C_CELL_DIR")
    if not environment_cell_value:
        raise PermissionError(
            "missing required Phase-C cell environment: M2_POST33_PHASE_C_CELL_DIR"
        )
    environment_cell_raw = Path(environment_cell_value)
    cell = require_canonical_directory(raw_cell)
    environment_cell = require_canonical_directory(environment_cell_raw)
    if (
        str(raw_cell) != str(cell)
        or str(environment_cell_raw) != str(environment_cell)
        or cell != environment_cell
    ):
        raise PermissionError(
            "Phase-C raw argv cell directory differs from capability environment"
        )
    try:
        root = cell.parents[5]
    except IndexError as exc:
        raise PermissionError(
            "Phase-C raw argv cell directory is outside the canonical layout"
        ) from exc
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
    require_cell_execution_capability_from_environment(root=root, key=key)
    return overrides


def verify_r9_training_wrapper_capability_only(
    argv: Sequence[str], *, arm: str
) -> bool:
    """CPU-only signed-envelope integration check for the real wrapper import.

    The exact one-flag path verifies the r9 environment, program, portable
    manifest, shard, signature, host, arm, and allowlists.  It deliberately
    performs no nonce claim, cell-root creation, execution-evidence write,
    Hydra initialization, data access, or CUDA observation.  Returning ``False``
    leaves the ordinary production argv path unchanged.
    """
    if list(argv) != ["--verify-r9-pre-hydra-capability-only"]:
        return False
    if arm not in ARMS:
        raise PermissionError("r9 wrapper capability-only arm is invalid")
    inputs = _execution_capability_inputs_from_environment()
    authorization = verify_signed_authorization(
        inputs["authorization_path"],
        inputs["signature_path"],
        phase_c_program_receipt_path=inputs["phase_c_program_receipt_path"],
        portable_manifest_path=inputs["portable_manifest_path"],
        shard_manifest_path=inputs["shard_manifest_path"],
        cost_supplement_path=inputs["cost_supplement_path"],
        cell_root=R9_CELL_ROOT,
    )
    shard = validate_r9_shard_manifest(
        inputs["shard_manifest_path"],
        portable_manifest_path=inputs["portable_manifest_path"],
        cell_root=R9_CELL_ROOT,
    )
    validate_observed_host(shard)
    if arm not in shard["arms_in_order"] or authorization["arms_in_order"] != list(ARMS):
        raise PermissionError("r9 wrapper arm is outside the signed capability")
    print(
        json.dumps(
            {
                "schema": "m2_post33_phase_c_v5_r9_wrapper_capability_only_v1",
                "status": "PASS_SIGNED_R9_CAPABILITY_BEFORE_HYDRA",
                "arm": arm,
                "authorization_id": authorization["authorization_id"],
                "host_id": shard["host_id"],
                "fold_allowlist": shard["fold_allowlist"],
                "seed_allowlist": shard["seed_allowlist"],
                "cell_root_created": R9_CELL_ROOT.exists(),
                "nonce_claimed": False,
                "hydra_initialized": False,
                "data_accessed": False,
                "cuda_initialized": False,
                "gpu_used": False,
                "score_or_r2_accessed": False,
            },
            sort_keys=True,
        )
    )
    return True
