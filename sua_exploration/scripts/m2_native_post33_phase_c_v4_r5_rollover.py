#!/usr/bin/env python3
"""Append-only r5 Phase-C capability renewal with an in-memory signer only."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import secrets
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sua_exploration.mc_maze import m2_native_post33_authorization_v4 as authorization
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    file_metadata,
    sha256_file,
    write_bytes_exclusive,
    write_json_exclusive,
)
from sua_exploration.mc_maze.m2_native_post33_program_v4 import (
    build_phase_c_program_receipt,
    validate_phase_c_program_receipt,
)


ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3"
R4 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r4"
R5 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r5"
CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r3"
EOF_RECEIPT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_upstream_canonicalization_20260805/upstream_eof_canonicalization.json"
DEEP_SOURCE_AUDIT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_deep_source_20260805/deep_source_audit.json"
R5_PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r5_root_ed25519_public.pem"
ALLOWED_SOURCE_DELTAS = {"sua_exploration/mc_maze/m2_native_post33_authorization_v4.py"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "+00:00")


def _source_rows(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = receipt.get("source_map")
    if not isinstance(rows, list):
        raise ValueError("program source map is missing")
    output = {str(row.get("relative_path")): dict(row) for row in rows if isinstance(row, Mapping)}
    if len(output) != len(rows):
        raise ValueError("program source map is malformed/non-unique")
    return output


def _validate_program_semantics(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    for field in (
        "schema", "protocol_id", "phase_id", "absolute_workspace_root",
        "score_data_accessed", "formal_data_accessed", "gpu_used",
        "phase_a_b_eof_canonicalization", "deep_source_audit_receipt",
    ):
        if before.get(field) != after.get(field):
            raise ValueError(f"r5 changed sealed program semantic field: {field}")
    old_rows, new_rows = _source_rows(before), _source_rows(after)
    if set(old_rows) != set(new_rows):
        raise ValueError("r5 changed source closure membership")
    deltas = {
        path: {"r3": old_rows[path], "r5": new_rows[path]}
        for path in sorted(old_rows) if old_rows[path] != new_rows[path]
    }
    if set(deltas) != ALLOWED_SOURCE_DELTAS:
        raise ValueError("r5 source delta is not limited to the fixed trust-anchor constants")
    return {
        "source_closure_membership_equal": True,
        "only_fixed_trust_anchor_constant_source_delta": deltas,
        "all_program_semantic_fields_equal": True,
    }


def _portable(r3: Mapping[str, Any], program: Path) -> dict[str, Any]:
    payload = dict(r3)
    closures = [dict(row) for row in r3["hash_closures"]]
    for row in closures:
        if row.get("role") == "phase_c_program_receipt":
            row.clear()
            row.update({"role": "phase_c_program_receipt", **file_metadata(program)})
    payload["hash_closures"] = closures
    return payload


def _shard(r3: Mapping[str, Any], portable: Path) -> dict[str, Any]:
    payload = dict(r3)
    payload["portable_transfer_manifest_sha256"] = sha256_file(portable)
    return payload


def _auth(r3: Mapping[str, Any], *, program: Path, portable: Path, shard: Path, issued: datetime, expires: datetime) -> dict[str, Any]:
    body = dict(r3["authorization"])
    original_id = str(body["authorization_id"])
    body["authorization_id"] = original_id.replace("_r3-", "_r5-")
    if body["authorization_id"] == original_id:
        raise ValueError("r3 authorization lacks required rollover id marker")
    body["single_use_nonce"] = secrets.token_hex(32)
    body["issued_at"] = _time(issued)
    body["expires_at"] = _time(expires)
    body["phase_c_program_receipt"] = file_metadata(program)
    body["portable_manifest"] = file_metadata(portable)
    body["shard_manifest"] = file_metadata(shard)
    body["public_key"] = file_metadata(R5_PUBLIC_KEY)
    return {"schema": "m2_post33_phase_c_signed_authorization_envelope_v4", "authorization": body}


def _validate_auth_semantics(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    old, new = before["authorization"], after["authorization"]
    if set(old) != set(new):
        raise ValueError("r5 authorization key-set changed")
    permitted = {
        "authorization_id", "single_use_nonce", "issued_at", "expires_at",
        "phase_c_program_receipt", "portable_manifest", "shard_manifest", "public_key",
    }
    changed = {key for key in old if old[key] != new[key]}
    if changed != permitted:
        raise ValueError("r5 authorization changed a protected field")
    for field in (
        "schema", "protocol_id", "phase_id", "status", "stage", "capability_scope",
        "host_id", "gpu_id", "absolute_cell_root", "fold_allowlist", "seed_allowlist",
        "arms_in_order", "evaluator", "cost_receipt", "cost_supplement",
    ):
        if old[field] != new[field]:
            raise ValueError(f"r5 changed protected authorization scope: {field}")
    return {"changed_authorization_metadata_only": sorted(changed)}


def _write_r4_failed_incident() -> Path:
    """Append a permanent no-execution incident marker; never rewrite r4 data."""
    destination = R4 / "r4_failed_pre_gpu_security_policy_relaxation.json"
    if destination.exists():
        raise FileExistsError("r4 failure incident marker already exists")
    r4_auths = sorted((R4 / "auth").glob("*.json"))
    if len(r4_auths) != 3 or CELL_ROOT.exists():
        raise ValueError("cannot attest r4 incident after a cell root or auth-set drift")
    payload = {
        "schema": "m2_post33_phase_c_failed_pre_gpu_incident_v1",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "status": "FAILED_PRE_GPU_SECURITY_POLICY_RELAXATION",
        "must_not_use_r4_authorizations": True,
        "failure_reason": {
            "max_validity_seconds_was_relaxed": True,
            "multiple_trust_anchors_were_accepted": True,
            "required_policy": "single_fixed_anchor_and_MAX_VALIDITY_SECONDS_30h",
        },
        "r4_authorizations": [file_metadata(path) for path in r4_auths],
        "r3_result_root": str(CELL_ROOT),
        "r3_result_root_exists": False,
        "gpu_used": False,
        "score_data_accessed": False,
        "endpoint_opened": False,
        "cell_opened": False,
        "authorization_nonce_claimed": False,
    }
    return write_json_exclusive(destination, payload)


def _preflight(private_key: Ed25519PrivateKey, *, issued: datetime, expires: datetime) -> None:
    if R5.exists():
        raise FileExistsError(f"append-only r5 receipt root already exists: {R5}")
    if not R3.is_dir() or not R4.is_dir() or CELL_ROOT.exists():
        raise FileNotFoundError("r3/r4 history or the required absent r3 result root is invalid")
    if authorization.MAX_VALIDITY_SECONDS != 30 * 3600:
        raise PermissionError("r5 refuses a relaxed Phase-C authorization maximum")
    if authorization.PUBLIC_KEY != R5_PUBLIC_KEY:
        raise PermissionError("r5 production verifier is not pinned to the r5 fixed anchor")
    public = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    if R5_PUBLIC_KEY.read_bytes() != public or sha256_file(R5_PUBLIC_KEY) != authorization.PUBLIC_KEY_SHA256:
        raise PermissionError("in-memory signer does not match the single r5 fixed public anchor")
    if not (issued <= datetime.now(timezone.utc) <= expires):
        raise PermissionError("r5 issuance time is not currently valid")
    if expires - issued != timedelta(hours=29) or expires - issued > timedelta(seconds=authorization.MAX_VALIDITY_SECONDS):
        raise PermissionError("r5 validity must be exactly 29 hours and within the 30-hour maximum")


def build_r5_rollover(private_key: Ed25519PrivateKey) -> dict[str, Any]:
    """Build r5 from an ephemeral private key that is never serialized."""
    issued = datetime.now(timezone.utc).replace(microsecond=0)
    expires = issued + timedelta(hours=29)
    _preflight(private_key, issued=issued, expires=expires)
    r3_program_path = R3 / "program/phase_c_program_r3.json"
    r3_portable_path = R3 / "manifest/portable_r3.json"
    r3_cost = R3 / "cost/base_cost_receipt_r3.json"
    r3_supplement = R3 / "cost/cost_supplement_r3.json"
    r3_program, r3_portable = _read_json(r3_program_path), _read_json(r3_portable_path)
    if r3_portable["absolute_cell_root"] != str(CELL_ROOT):
        raise ValueError("r3 does not bind the designated absent result root")
    r3_shards = {
        "gpu0": _read_json(R3 / "manifest/shard_stage_a_gpu0_r3.json"),
        "gpu1": _read_json(R3 / "manifest/shard_stage_a_gpu1_r3.json"),
        "opening": _read_json(R3 / "manifest/shard_stage_a_opening_r3.json"),
    }
    r3_auths = {
        "gpu0": _read_json(R3 / "auth/stage_a_execution_gpu0_r3.json"),
        "gpu1": _read_json(R3 / "auth/stage_a_execution_gpu1_r3.json"),
        "opening": _read_json(R3 / "auth/stage_a_opening_r3.json"),
    }
    r5_program_payload = build_phase_c_program_receipt(
        eof_canonicalization_receipt_path=EOF_RECEIPT,
        deep_source_audit_receipt_path=DEEP_SOURCE_AUDIT,
    )
    program = R5 / "program/phase_c_program_r5.json"
    portable = R5 / "manifest/portable_r5.json"
    shards = {
        "gpu0": R5 / "manifest/shard_stage_a_gpu0_r5.json",
        "gpu1": R5 / "manifest/shard_stage_a_gpu1_r5.json",
        "opening": R5 / "manifest/shard_stage_a_opening_r5.json",
    }
    auths = {
        "gpu0": R5 / "auth/stage_a_execution_gpu0_r5.json",
        "gpu1": R5 / "auth/stage_a_execution_gpu1_r5.json",
        "opening": R5 / "auth/stage_a_opening_r5.json",
    }

    # The failed r4 facts are frozen before any r5 authorization is written.
    incident = _write_r4_failed_incident()
    write_json_exclusive(program, r5_program_payload)
    validate_phase_c_program_receipt(program)
    write_json_exclusive(portable, _portable(r3_portable, program))
    r5_shards = {name: _shard(value, portable) for name, value in r3_shards.items()}
    for name, path in shards.items():
        write_json_exclusive(path, r5_shards[name])
    r5_auths = {
        name: _auth(r3_auths[name], program=program, portable=portable, shard=shards[name], issued=issued, expires=expires)
        for name in auths
    }
    nonces = [value["authorization"]["single_use_nonce"] for value in r5_auths.values()]
    if len(nonces) != 3 or len(set(nonces)) != 3:
        raise RuntimeError("r5 requires exactly three distinct 256-bit nonces")
    for name, path in auths.items():
        write_json_exclusive(path, r5_auths[name])
        write_bytes_exclusive(path.with_suffix(".sig"), base64.b64encode(private_key.sign(path.read_bytes())))

    r5_program = _read_json(program)
    proof = {
        "schema": "m2_post33_phase_c_r5_rollover_semantic_equality_v1",
        "protocol_id": r3_program["protocol_id"],
        "phase_id": r3_program["phase_id"],
        "r3_receipt_root": str(R3),
        "r4_failed_incident": file_metadata(incident),
        "r5_receipt_root": str(R5),
        "r3_result_root": str(CELL_ROOT),
        "r3_result_root_exists_at_r5": False,
        "single_fixed_trust_anchor": {
            "algorithm": "Ed25519",
            "public_key": file_metadata(R5_PUBLIC_KEY),
            "source_pinned_sha256": authorization.PUBLIC_KEY_SHA256,
            "multiple_anchors_accepted": False,
            "private_key_persisted": False,
            "detached_signatures_only": True,
        },
        "authorization_policy": {
            "max_validity_seconds": authorization.MAX_VALIDITY_SECONDS,
            "issued_at": _time(issued),
            "expires_at": _time(expires),
            "validity_hours": 29,
        },
        "program": _validate_program_semantics(r3_program, r5_program),
        "portable_manifest": {
            "only_program_receipt_closure_changed": True,
            "r3": file_metadata(r3_portable_path),
            "r5": file_metadata(portable),
        },
        "shards": {
            name: {
                "only_portable_digest_changed": True,
                "r3": file_metadata(R3 / f"manifest/shard_stage_a_{'opening' if name == 'opening' else name}_r3.json"),
                "r5": file_metadata(path),
                "host_id": r5_shards[name]["host_id"], "gpu_id": r5_shards[name]["gpu_id"],
                "fold_allowlist": r5_shards[name]["fold_allowlist"], "seed_allowlist": r5_shards[name]["seed_allowlist"],
            }
            for name, path in shards.items()
        },
        "cost_seals": {
            "reused_without_reserialization_because_not_anchor_dependent": True,
            "base_cost_receipt": file_metadata(r3_cost),
            "cost_supplement": file_metadata(r3_supplement),
        },
        "authorizations": {name: _validate_auth_semantics(r3_auths[name], r5_auths[name]) for name in auths},
        "forbidden_activity": {
            "gpu_used": False, "score_data_accessed": False, "endpoint_opened": False,
            "cell_opened": False, "result_root_created": False,
        },
    }
    proof_path = R5 / "r5_rollover_semantic_equality.json"
    write_json_exclusive(proof_path, proof)
    for name, path in auths.items():
        authorization.verify_signed_authorization(
            path, path.with_suffix(".sig"), phase_c_program_receipt_path=program,
            portable_manifest_path=portable, shard_manifest_path=shards[name],
            cost_supplement_path=r3_supplement, cell_root=CELL_ROOT,
            now=issued + timedelta(seconds=1),
        )
    return {
        "r4_failed_incident_sha256": sha256_file(incident),
        "r5_receipt_root": str(R5), "program_sha256": sha256_file(program),
        "portable_sha256": sha256_file(portable), "semantic_equality_sha256": sha256_file(proof_path),
        "authorization_sha256": {name: sha256_file(path) for name, path in auths.items()},
        "authorization_signature_sha256": {name: sha256_file(path.with_suffix(".sig")) for name, path in auths.items()},
    }

