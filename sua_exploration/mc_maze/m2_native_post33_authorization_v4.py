"""Detached-Ed25519, scoped, expiring, single-use Phase-C authorization."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import socket
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    ARMS,
    CellKey,
    PHASE_ID,
    PROTOCOL_ID,
    SEEDS,
    FOLDS,
    cell_paths,
    file_metadata,
    phase_root,
    require_canonical_regular_file,
    sha256_file,
    stage_a_paths,
    validate_shard_manifest,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6d_root_ed25519_public.pem"
PUBLIC_KEY_SHA256 = "3d5bb3b3e679762ce0fcf345cbbfc2bdf85a83e76e520939c4c4e02120a6535d"
EVALUATOR = ROOT / "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v4.py"
MAX_VALIDITY_SECONDS = 30 * 3600
CAPABILITY_SCOPE_CELL_EXECUTION = "cell_execution"
CAPABILITY_SCOPE_OPENING = "opening"
OPENING_STAGES = ("stage_a_opening", "full_opening")
CAPABILITY_ENVIRONMENT = {
    "authorization_path": "M2_POST33_PHASE_C_AUTHORIZATION",
    "signature_path": "M2_POST33_PHASE_C_AUTHORIZATION_SIGNATURE",
    "phase_c_program_receipt_path": "M2_POST33_PHASE_C_PROGRAM_RECEIPT",
    "portable_manifest_path": "M2_POST33_PHASE_C_PORTABLE_MANIFEST",
    "shard_manifest_path": "M2_POST33_PHASE_C_SHARD_MANIFEST",
    "cost_supplement_path": "M2_POST33_PHASE_C_COST_SUPPLEMENT",
}


def validate_observed_host(shard: Mapping[str, Any], *, observed_host: str | None = None) -> str:
    observed = observed_host if observed_host is not None else socket.gethostname()
    if not isinstance(observed, str) or not observed:
        raise PermissionError("observed canonical hostname is unavailable")
    if shard.get("host_id") != observed:
        raise PermissionError("signed shard host_id differs from observed canonical hostname")
    return observed


def validate_observed_gpu(
    shard: Mapping[str, Any], *, observed_visible_devices: str | None = None
) -> dict[str, Any]:
    """Validate the logical CUDA device's signed physical identity without CUDA I/O.

    A process with exactly one entry in CUDA_VISIBLE_DEVICES sees that physical
    device as logical device zero.  Requiring the sole token to equal the
    signed GPU identity prevents an inherited empty, multi-device, or
    mismatched mapping before a Trainer/evaluator can initialize CUDA.
    """
    raw = observed_visible_devices
    if raw is None:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(raw, str) or not raw:
        raise PermissionError("signed GPU requires one visible physical device")
    visible = [token.strip() for token in raw.split(",")]
    # The signed shard identifier has no whitespace.  Keep the process-side
    # representation canonical as well: accepting a cosmetic rewrite here
    # would make the recorded physical mapping ambiguous across launchers.
    if len(visible) != 1 or not visible[0] or raw != visible[0]:
        raise PermissionError("signed GPU requires exactly one visible device")
    expected = shard.get("gpu_id")
    if visible[0] != expected:
        raise PermissionError("signed GPU id differs from CUDA_VISIBLE_DEVICES physical mapping")
    return {
        "cuda_visible_devices": raw,
        "logical_cuda_device_index": 0,
        "physical_gpu_id": visible[0],
    }


def validate_cell_scope(
    *, arm: str, fold: int, seed: int, shard: Mapping[str, Any], authorization: Mapping[str, Any]
) -> None:
    if (
        arm not in shard.get("arms_in_order", [])
        or fold not in shard.get("fold_allowlist", [])
        or seed not in shard.get("seed_allowlist", [])
    ):
        raise PermissionError("cell is outside the signed shard allowlist")
    if (
        shard.get("arms_in_order") != authorization.get("arms_in_order")
        or shard.get("fold_allowlist") != authorization.get("fold_allowlist")
        or shard.get("seed_allowlist") != authorization.get("seed_allowlist")
    ):
        raise PermissionError("cell shard/authorization scope mismatch")


def execution_capability_environment(
    *,
    authorization_path: str | Path,
    signature_path: str | Path,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cost_supplement_path: str | Path,
) -> dict[str, str]:
    """Return the exact signed paths inherited by protected subprocesses."""
    inputs = {
        "authorization_path": authorization_path,
        "signature_path": signature_path,
        "phase_c_program_receipt_path": phase_c_program_receipt_path,
        "portable_manifest_path": portable_manifest_path,
        "shard_manifest_path": shard_manifest_path,
        "cost_supplement_path": cost_supplement_path,
    }
    return {
        CAPABILITY_ENVIRONMENT[name]: str(Path(path).resolve(strict=True))
        for name, path in inputs.items()
    }


def _execution_capability_inputs_from_environment() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for field, variable in CAPABILITY_ENVIRONMENT.items():
        value = os.environ.get(variable)
        if not value:
            raise PermissionError(f"missing required Phase-C capability environment: {variable}")
        paths[field] = require_canonical_regular_file(value)
    return paths


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise PermissionError(f"authorization {label} must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PermissionError(f"authorization {label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _verify_metadata(metadata: Any, expected: str | Path, label: str) -> Path:
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "canonical_path", "size_bytes", "sha256"
    }:
        raise PermissionError(f"authorization {label} metadata exact set mismatch")
    target = require_canonical_regular_file(expected)
    if metadata != file_metadata(target):
        raise PermissionError(f"authorization {label} substitution/hash drift")
    return target


def _verify_signed_authorization_with_anchor(
    authorization_path: str | Path,
    signature_path: str | Path,
    *,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cell_root: str | Path,
    cost_supplement_path: str | Path | None = None,
    now: datetime | None = None,
    public_key_path: str | Path,
    expected_public_key_sha256: str,
) -> Mapping[str, Any]:
    authorization_file = require_canonical_regular_file(authorization_path)
    signature_file = require_canonical_regular_file(signature_path)
    public_key_file = require_canonical_regular_file(public_key_path)
    if sha256_file(public_key_file) != expected_public_key_sha256:
        raise PermissionError("Phase-C root trust anchor hash mismatch")
    key = serialization.load_pem_public_key(public_key_file.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("Phase-C trust anchor is not Ed25519")
    try:
        signature = base64.b64decode(signature_file.read_bytes(), validate=True)
        key.verify(signature, authorization_file.read_bytes())
    except (InvalidSignature, ValueError) as exc:
        raise PermissionError("invalid detached Ed25519 Phase-C authorization") from exc

    envelope = json.loads(authorization_file.read_text(encoding="utf-8"))
    if set(envelope) != {"schema", "authorization"} or envelope.get("schema") != (
        "m2_post33_phase_c_signed_authorization_envelope_v4"
    ):
        raise PermissionError("Phase-C signed authorization envelope mismatch")
    auth = envelope.get("authorization")
    required = {
        "schema", "protocol_id", "phase_id", "status", "authorization_id",
        "single_use_nonce", "issued_at", "expires_at", "stage", "capability_scope",
        "host_id", "gpu_id",
        "absolute_cell_root", "fold_allowlist", "seed_allowlist", "arms_in_order",
        "phase_c_program_receipt", "portable_manifest", "shard_manifest",
        "evaluator", "cost_receipt", "cost_supplement", "public_key",
    }
    if not isinstance(auth, Mapping):
        raise PermissionError("Phase-C authorization body is not a mapping")
    stage = auth.get("stage")
    expected_keys = required if stage in {"stage_a", "stage_a_opening"} else required | {
        "stage_a_decision", "stage_a_decision_signature"
    }
    if set(auth) != expected_keys:
        raise PermissionError("Phase-C authorization exact key set mismatch")
    if (
        auth.get("schema") != "m2_post33_phase_c_gpu_authorization_v4"
        or auth.get("protocol_id") != PROTOCOL_ID
        or auth.get("phase_id") != PHASE_ID
        or auth.get("status") != "GO"
    ):
        raise PermissionError("Phase-C authorization protocol/status mismatch")
    scope = auth.get("capability_scope")
    if scope not in {CAPABILITY_SCOPE_CELL_EXECUTION, CAPABILITY_SCOPE_OPENING}:
        raise PermissionError("Phase-C authorization capability scope is invalid")
    nonce = auth.get("single_use_nonce")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise PermissionError("Phase-C authorization nonce must be 256-bit lowercase hex")
    auth_id = auth.get("authorization_id")
    if not isinstance(auth_id, str) or not auth_id or any(char.isspace() for char in auth_id):
        raise PermissionError("Phase-C authorization id invalid")
    issued = _parse_utc(auth.get("issued_at"), "issued_at")
    expires = _parse_utc(auth.get("expires_at"), "expires_at")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not (issued <= current <= expires) or (expires - issued).total_seconds() > MAX_VALIDITY_SECONDS:
        raise PermissionError("Phase-C authorization expired/not-yet-valid/exceeds 30h")

    program = _verify_metadata(
        auth.get("phase_c_program_receipt"), phase_c_program_receipt_path, "program receipt"
    )
    portable = _verify_metadata(
        auth.get("portable_manifest"), portable_manifest_path, "portable manifest"
    )
    shard_file = _verify_metadata(
        auth.get("shard_manifest"), shard_manifest_path, "shard manifest"
    )
    _verify_metadata(auth.get("evaluator"), EVALUATOR, "evaluator")
    cost_metadata = auth.get("cost_receipt")
    if not isinstance(cost_metadata, Mapping):
        raise PermissionError("authorization cost receipt missing")
    cost = _verify_metadata(
        cost_metadata, cost_metadata.get("canonical_path", ""), "cost receipt"
    )
    supplement = _verify_metadata(
        auth.get("cost_supplement"),
        cost_supplement_path or auth.get("cost_supplement", {}).get("canonical_path", ""),
        "cost supplement",
    )
    _verify_metadata(auth.get("public_key"), public_key_file, "public key")
    shard = validate_shard_manifest(
        shard_file, portable_manifest_path=portable, cell_root=cell_root
    )
    if auth.get("absolute_cell_root") != str(Path(cell_root).resolve()):
        raise PermissionError("authorization cell root substitution")
    for field in ("host_id", "gpu_id"):
        if auth.get(field) != shard.get(field):
            raise PermissionError(f"authorization {field} scope expansion/substitution")
    if auth.get("fold_allowlist") != shard.get("fold_allowlist"):
        raise PermissionError("authorization fold scope expansion/substitution")
    if auth.get("seed_allowlist") != shard.get("seed_allowlist"):
        raise PermissionError("authorization seed scope expansion/substitution")
    if auth.get("arms_in_order") != list(ARMS) or auth.get("arms_in_order") != shard.get("arms_in_order"):
        raise PermissionError("authorization arm scope expansion/substitution")
    seeds = set(auth["seed_allowlist"])
    if scope == CAPABILITY_SCOPE_CELL_EXECUTION:
        expected_stage = "stage_a" if seeds == {42} else "stage_b" if seeds <= {43, 44} else None
    else:
        expected_stage = (
            "stage_a_opening"
            if seeds == {42}
            else "full_opening"
            if seeds == set(SEEDS)
            else None
        )
    if expected_stage is None or auth.get("stage") != expected_stage:
        raise PermissionError("authorization stage/seed scope mismatch")
    if expected_stage in {"stage_b", "full_opening"}:
        _verify_metadata(
            auth.get("stage_a_decision"), stage_a_paths(cell_root)["decision"],
            "Stage-A continue decision",
        )
        _verify_metadata(
            auth.get("stage_a_decision_signature"), stage_a_paths(cell_root)["decision_signature"],
            "Stage-A continue decision signature",
        )
    return {
        **dict(auth), "_validated_cost_receipt_path": str(cost),
        "_validated_cost_supplement_path": str(supplement),
        "_validated_program_path": str(program),
        "_validated_portable_path": str(portable),
        "_validated_shard_path": str(shard_file),
    }


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
    """Production verifier with a non-caller-selectable project trust anchor."""
    authorization = _verify_signed_authorization_with_anchor(
        authorization_path,
        signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cell_root=cell_root,
        cost_supplement_path=cost_supplement_path,
        now=now,
        public_key_path=PUBLIC_KEY,
        expected_public_key_sha256=PUBLIC_KEY_SHA256,
    )
    # Metadata matching alone is insufficient: a signed authorization must
    # transit through the exact Phase-C source/EOF/deep-audit closure before it
    # becomes a production capability.  The test-only verifier intentionally
    # calls the lower-level anchor helper directly and is kept outside this
    # production API surface.
    from sua_exploration.mc_maze.m2_native_post33_cost_v4 import validate_cost_supplement
    from sua_exploration.mc_maze.m2_native_post33_program_v4 import (
        validate_phase_c_program_receipt,
    )
    program = validate_phase_c_program_receipt(authorization["_validated_program_path"])
    supplement_path = Path(authorization["_validated_cost_supplement_path"])
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    validate_cost_supplement(supplement)
    if program.get("deep_source_audit_receipt") != supplement.get("deep_source_audit_receipt"):
        raise PermissionError("authorization program/cost deep-source closure mismatch")
    return authorization


def _validate_opening_scope(
    authorization: Mapping[str, Any],
    *,
    opening_stage: str,
) -> None:
    """Reject a shard capability that is not the exact delayed-opening scope."""
    if opening_stage not in OPENING_STAGES:
        raise ValueError("opening stage is invalid")
    if (
        authorization.get("capability_scope") != CAPABILITY_SCOPE_OPENING
        or authorization.get("stage") != opening_stage
        or authorization.get("arms_in_order") != list(ARMS)
        or authorization.get("fold_allowlist") != list(FOLDS)
    ):
        raise PermissionError("opening capability scope expansion/substitution")
    expected_seeds = [42] if opening_stage == "stage_a_opening" else list(SEEDS)
    if authorization.get("seed_allowlist") != expected_seeds:
        raise PermissionError("opening capability seed scope mismatch")


def consume_opening_authorization_nonce(
    *,
    root: str | Path,
    opening_stage: str,
    authorization_path: str | Path,
    signature_path: str | Path,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cost_supplement_path: str | Path,
) -> tuple[Mapping[str, Any], Path]:
    """Verify and atomically consume a dedicated delayed-opening capability.

    The claim is intentionally separate from execution claims: opening a
    score-bearing payload is not an incidental continuation of a training
    shard.  The exclusive create happens before the caller can read a payload.
    """
    authorization = verify_signed_authorization(
        authorization_path,
        signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
        cell_root=root,
    )
    _validate_opening_scope(authorization, opening_stage=opening_stage)
    return authorization, _write_opening_authorization_claim(
        root=root,
        opening_stage=opening_stage,
        authorization=authorization,
        authorization_path=authorization_path,
        signature_path=signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
    )


def _write_opening_authorization_claim(
    *,
    root: str | Path,
    opening_stage: str,
    authorization: Mapping[str, Any],
    authorization_path: str | Path,
    signature_path: str | Path,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cost_supplement_path: str | Path,
) -> Path:
    """Write an already-validated opening claim; used by the test-only helper."""
    nonce = authorization["single_use_nonce"]
    destination = (
        phase_root(root)
        / "opening_authorization_claims"
        / opening_stage
        / f"{nonce}.json"
    )
    claim = {
        "schema": "m2_post33_phase_c_opening_nonce_claim_v4",
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "opening_stage": opening_stage,
        "authorization_id": authorization["authorization_id"],
        "single_use_nonce": nonce,
        "host_id": authorization["host_id"],
        "gpu_id": authorization["gpu_id"],
        "absolute_cell_root": authorization["absolute_cell_root"],
        "authorization": file_metadata(authorization_path),
        "signature": file_metadata(signature_path),
        "shard_manifest": file_metadata(shard_manifest_path),
        "phase_c_program_receipt": file_metadata(phase_c_program_receipt_path),
        "portable_manifest": file_metadata(portable_manifest_path),
        "cost_supplement": file_metadata(cost_supplement_path),
    }
    return write_json_exclusive(destination, claim)


def claim_authorization_nonce(
    *,
    root: str | Path,
    authorization: Mapping[str, Any],
    authorization_path: str | Path,
    signature_path: str | Path,
    shard_manifest_path: str | Path,
) -> Path:
    if authorization.get("capability_scope") != CAPABILITY_SCOPE_CELL_EXECUTION:
        raise PermissionError("only cell-execution capabilities can create execution claims")
    host_id = authorization["host_id"]
    if re.fullmatch(r"[A-Za-z0-9_.-]+", host_id) is None:
        raise PermissionError("authorization host id is unsafe for nonce claim")
    nonce = authorization["single_use_nonce"]
    destination = phase_root(root) / "authorization_claims" / host_id / f"{nonce}.json"
    return write_json_exclusive(
        destination,
        {
            "schema": "m2_post33_phase_c_authorization_nonce_claim_v4",
            "protocol_id": PROTOCOL_ID,
            "phase_id": PHASE_ID,
            "authorization_id": authorization["authorization_id"],
            "single_use_nonce": nonce,
            "stage": authorization["stage"],
            "capability_scope": authorization["capability_scope"],
            "host_id": host_id,
            "gpu_id": authorization["gpu_id"],
            "absolute_cell_root": authorization["absolute_cell_root"],
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
    host_id = authorization["host_id"]
    nonce = authorization["single_use_nonce"]
    path = phase_root(root) / "authorization_claims" / host_id / f"{nonce}.json"
    claim_file = require_canonical_regular_file(path, within=phase_root(root))
    claim = json.loads(claim_file.read_text(encoding="utf-8"))
    expected = {
        "schema": "m2_post33_phase_c_authorization_nonce_claim_v4",
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "authorization_id": authorization["authorization_id"],
        "single_use_nonce": nonce,
        "stage": authorization["stage"],
        "capability_scope": authorization["capability_scope"],
        "host_id": host_id,
        "gpu_id": authorization["gpu_id"],
        "absolute_cell_root": authorization["absolute_cell_root"],
        "authorization": file_metadata(authorization_path),
        "signature": file_metadata(signature_path),
        "shard_manifest": file_metadata(shard_manifest_path),
        "phase_c_program_receipt": file_metadata(authorization["_validated_program_path"]),
        "portable_manifest": file_metadata(authorization["_validated_portable_path"]),
        "cost_supplement": file_metadata(authorization["_validated_cost_supplement_path"]),
    }
    if claim != expected:
        raise PermissionError("authorization nonce claim substitution or wrong bound files")
    return claim_file


def _execution_capability_evidence(
    *,
    root: str | Path,
    key: CellKey,
    authorization: Mapping[str, Any],
    authorization_path: str | Path,
    signature_path: str | Path,
    shard_manifest_path: str | Path,
    claim_path: str | Path,
    gpu_observation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "m2_post33_phase_c_execution_capability_evidence_v4",
        **key.identity(),
        "authorization_id": authorization["authorization_id"],
        "single_use_nonce": authorization["single_use_nonce"],
        "stage": authorization["stage"],
        "capability_scope": authorization["capability_scope"],
        "observed_host_id": authorization["host_id"],
        "observed_gpu": dict(gpu_observation),
        "absolute_cell_root": str(Path(root).resolve()),
        "authorization": file_metadata(authorization_path),
        "signature": file_metadata(signature_path),
        "shard_manifest": file_metadata(shard_manifest_path),
        "nonce_claim": file_metadata(claim_path),
    }


def _write_or_verify_execution_capability_evidence(
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
    paths = cell_paths(root, key)
    expected = _execution_capability_evidence(
        root=root,
        key=key,
        authorization=authorization,
        authorization_path=authorization_path,
        signature_path=signature_path,
        shard_manifest_path=shard_manifest_path,
        claim_path=claim_path,
        gpu_observation=gpu_observation,
    )
    destination = paths["execution_capability_evidence_run"]
    if destination.exists():
        existing = require_canonical_regular_file(destination, within=paths["run"])
        observed = json.loads(existing.read_text(encoding="utf-8"))
        if observed != expected:
            raise PermissionError("execution capability evidence substitution")
        return existing
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
    """Fail closed before a production callable may construct data or CUDA."""
    authorization = verify_signed_authorization(
        authorization_path,
        signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
        cell_root=root,
    )
    if authorization.get("capability_scope") != CAPABILITY_SCOPE_CELL_EXECUTION:
        raise PermissionError("cell execution requires a cell_execution capability")
    shard = validate_shard_manifest(
        shard_manifest_path,
        portable_manifest_path=portable_manifest_path,
        cell_root=root,
    )
    validate_observed_host(shard)
    gpu_observation = validate_observed_gpu(shard)
    validate_cell_scope(
        arm=key.arm,
        fold=key.fold,
        seed=key.seed,
        shard=shard,
        authorization=authorization,
    )
    claim = verify_authorization_nonce_claim(
        root=root,
        authorization=authorization,
        authorization_path=authorization_path,
        signature_path=signature_path,
        shard_manifest_path=shard_manifest_path,
    )
    evidence = _write_or_verify_execution_capability_evidence(
        root=root,
        key=key,
        authorization=authorization,
        authorization_path=authorization_path,
        signature_path=signature_path,
        shard_manifest_path=shard_manifest_path,
        claim_path=claim,
        gpu_observation=gpu_observation,
    )
    return {
        **authorization,
        "_validated_nonce_claim_path": str(claim),
        "_execution_capability_evidence_path": str(evidence),
        "_observed_gpu": dict(gpu_observation),
    }


def require_cell_execution_capability_from_environment(
    *, root: str | Path, key: CellKey
) -> Mapping[str, Any]:
    """Production trainer entrypoints obtain signed paths only from this ABI."""
    inputs = _execution_capability_inputs_from_environment()
    return require_cell_execution_capability(root=root, key=key, **inputs)


def revalidate_bound_claim(
    root: str | Path,
    claim_path: str | Path,
) -> Mapping[str, Any]:
    claim_file = require_canonical_regular_file(claim_path, within=phase_root(root))
    claim = json.loads(claim_file.read_text(encoding="utf-8"))
    if claim.get("schema") != "m2_post33_phase_c_authorization_nonce_claim_v4":
        raise PermissionError("bound nonce claim schema mismatch")
    bound: dict[str, Path] = {}
    for field in ("authorization", "signature", "shard_manifest"):
        metadata = claim.get(field)
        if not isinstance(metadata, Mapping):
            raise PermissionError("bound nonce claim metadata missing")
        path = require_canonical_regular_file(metadata.get("canonical_path", ""))
        if metadata != file_metadata(path):
            raise PermissionError("bound nonce claim file hash substitution")
        bound[field] = path
    envelope = json.loads(bound["authorization"].read_text(encoding="utf-8"))
    auth = envelope.get("authorization") if isinstance(envelope, Mapping) else None
    if not isinstance(auth, Mapping):
        raise PermissionError("bound authorization envelope invalid")
    historical_validation_time = _parse_utc(auth.get("issued_at"), "issued_at") + timedelta(seconds=1)
    validated = verify_signed_authorization(
        bound["authorization"], bound["signature"],
        phase_c_program_receipt_path=auth.get("phase_c_program_receipt", {}).get("canonical_path", ""),
        portable_manifest_path=auth.get("portable_manifest", {}).get("canonical_path", ""),
        shard_manifest_path=bound["shard_manifest"], cell_root=root,
        cost_supplement_path=auth.get("cost_supplement", {}).get("canonical_path", ""),
        now=historical_validation_time,
    )
    verify_authorization_nonce_claim(
        root=root, authorization=validated,
        authorization_path=bound["authorization"], signature_path=bound["signature"],
        shard_manifest_path=bound["shard_manifest"],
    )
    return validated


def validate_claim_coverage(
    root: str | Path,
    *,
    stage: str,
) -> list[Mapping[str, Any]]:
    if stage not in {"stage_a", "stage_b"}:
        raise ValueError("claim coverage stage invalid")
    claim_root = phase_root(root) / "authorization_claims"
    if not claim_root.is_dir():
        raise PermissionError("signed authorization claims are missing")
    validated = []
    coverage: list[tuple[int, int]] = []
    for claim_path in sorted(claim_root.glob("*/*.json")):
        auth = revalidate_bound_claim(
            root, claim_path,
        )
        if auth["stage"] != stage:
            continue
        validated.append(auth)
        coverage.extend(
            (fold, seed)
            for fold in auth["fold_allowlist"]
            for seed in auth["seed_allowlist"]
        )
    validate_coverage_pairs(stage=stage, coverage=coverage)
    return validated


def validate_coverage_pairs(*, stage: str, coverage: list[tuple[int, int]]) -> None:
    if stage not in {"stage_a", "stage_b"}:
        raise ValueError("claim coverage stage invalid")
    expected_seeds = (42,) if stage == "stage_a" else (43, 44)
    expected = {(fold, seed) for fold in range(7) for seed in expected_seeds}
    if set(coverage) != expected or len(coverage) != len(expected):
        raise PermissionError(
            f"{stage} signed shard coverage is missing, duplicated, overlapping, or expanded"
        )
    cell_paths,
