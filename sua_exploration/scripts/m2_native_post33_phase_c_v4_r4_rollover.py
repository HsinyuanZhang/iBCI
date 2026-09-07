#!/usr/bin/env python3
"""Append-only, CPU-only r4 renewal of expired Phase-C Stage-A capabilities.

This helper deliberately accepts a live ``Ed25519PrivateKey`` object instead
of a filename or serialized value.  The caller is responsible for creating
that object in memory, so this repository never contains a new private key.
It does not create a cell root, start a trainer, read a score, or open an
endpoint payload.
"""
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
CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r3"
EOF_RECEIPT = (
    ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_upstream_canonicalization_20260805/upstream_eof_canonicalization.json"
)
DEEP_SOURCE_AUDIT = (
    ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_deep_source_20260805/deep_source_audit.json"
)
R4_PUBLIC_KEY = authorization.R4_ROLLOVER_PUBLIC_KEY
R4_PUBLIC_KEY_SHA256 = authorization.R4_ROLLOVER_PUBLIC_KEY_SHA256
ISSUED_AT = datetime(2026, 8, 4, 20, 30, tzinfo=timezone.utc)
EXPIRES_AT = datetime(2026, 8, 6, 19, 30, tzinfo=timezone.utc)
ALLOWED_SOURCE_DELTAS = frozenset(
    {
        "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py",
        "sua_exploration/mc_maze/m2_native_post33_openers_v4.py",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "+00:00")


def _source_map_by_path(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = receipt.get("source_map")
    if not isinstance(rows, list):
        raise ValueError("program receipt source map is missing")
    mapped = {str(row.get("relative_path")): dict(row) for row in rows if isinstance(row, Mapping)}
    if len(mapped) != len(rows):
        raise ValueError("program receipt source map is non-unique or malformed")
    return mapped


def _program_semantics(r3_program: Mapping[str, Any], r4_program: Mapping[str, Any]) -> dict[str, Any]:
    for field in (
        "schema", "protocol_id", "phase_id", "absolute_workspace_root",
        "score_data_accessed", "formal_data_accessed", "gpu_used",
        "phase_a_b_eof_canonicalization", "deep_source_audit_receipt",
    ):
        if r3_program.get(field) != r4_program.get(field):
            raise ValueError(f"r4 changed sealed program semantic field: {field}")
    before = _source_map_by_path(r3_program)
    after = _source_map_by_path(r4_program)
    if set(before) != set(after):
        raise ValueError("r4 program source closure changed membership")
    changed = {
        path: {"r3": before[path], "r4": after[path]}
        for path in sorted(before)
        if before[path] != after[path]
    }
    if set(changed) != ALLOWED_SOURCE_DELTAS:
        raise ValueError("r4 program has an unapproved source semantic delta")
    return {
        "source_closure_membership_equal": True,
        "approved_trust_policy_source_deltas": changed,
        "all_other_program_semantic_fields_equal": True,
    }


def _portable_payload(r3_portable: Mapping[str, Any], *, program: Path) -> dict[str, Any]:
    payload = dict(r3_portable)
    closures = [dict(row) for row in r3_portable["hash_closures"]]
    for row in closures:
        if row.get("role") == "phase_c_program_receipt":
            row.clear()
            row.update({"role": "phase_c_program_receipt", **file_metadata(program)})
    payload["hash_closures"] = closures
    return payload


def _shard_payload(r3_shard: Mapping[str, Any], *, portable: Path) -> dict[str, Any]:
    payload = dict(r3_shard)
    payload["portable_transfer_manifest_sha256"] = sha256_file(portable)
    return payload


def _authorization_payload(
    r3_envelope: Mapping[str, Any],
    *,
    program: Path,
    portable: Path,
    shard: Path,
) -> dict[str, Any]:
    body = dict(r3_envelope["authorization"])
    old_id = str(body["authorization_id"])
    body["authorization_id"] = old_id.replace("_r3-", "_r4-")
    if body["authorization_id"] == old_id:
        raise ValueError("r3 authorization id lacks its expected rollover marker")
    body["single_use_nonce"] = secrets.token_hex(32)
    body["issued_at"] = _timestamp(ISSUED_AT)
    body["expires_at"] = _timestamp(EXPIRES_AT)
    body["phase_c_program_receipt"] = file_metadata(program)
    body["portable_manifest"] = file_metadata(portable)
    body["shard_manifest"] = file_metadata(shard)
    body["public_key"] = file_metadata(R4_PUBLIC_KEY)
    return {
        "schema": "m2_post33_phase_c_signed_authorization_envelope_v4",
        "authorization": body,
    }


def _authorization_semantics(
    r3: Mapping[str, Any], r4: Mapping[str, Any]) -> dict[str, Any]:
    before = r3["authorization"]
    after = r4["authorization"]
    if set(before) != set(after):
        raise ValueError("r4 authorization key set drift")
    allowed = {
        "authorization_id", "single_use_nonce", "issued_at", "expires_at",
        "phase_c_program_receipt", "portable_manifest", "shard_manifest", "public_key",
    }
    changed = sorted(key for key in before if before[key] != after[key])
    if set(changed) != allowed:
        raise ValueError("r4 authorization has a non-rollover semantic delta")
    for field in ("protocol_id", "phase_id", "status", "stage", "capability_scope", "host_id", "gpu_id", "absolute_cell_root", "fold_allowlist", "seed_allowlist", "arms_in_order", "evaluator", "cost_receipt", "cost_supplement"):
        if before[field] != after[field]:
            raise ValueError(f"r4 authorization changed protected scope: {field}")
    return {"changed_authorization_metadata_only": changed}


def _preflight(private_key: Ed25519PrivateKey) -> None:
    if R4.exists():
        raise FileExistsError(f"append-only r4 receipt root already exists: {R4}")
    if CELL_ROOT.exists():
        raise FileExistsError("r3 cell root must remain absent before capability rollover")
    if not R3.is_dir():
        raise FileNotFoundError("immutable r3 receipt root is absent")
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if R4_PUBLIC_KEY.read_bytes() != public_bytes or sha256_file(R4_PUBLIC_KEY) != R4_PUBLIC_KEY_SHA256:
        raise PermissionError("in-memory rollover signer does not match the fixed r4 public anchor")
    if not (ISSUED_AT <= datetime.now(timezone.utc) <= EXPIRES_AT):
        raise PermissionError("r4 rollover authorization time window is not currently valid")
    if EXPIRES_AT - ISSUED_AT < timedelta(hours=36):
        raise ValueError("r4 expiry margin must be at least 36 HKT hours")
    if EXPIRES_AT - ISSUED_AT > timedelta(seconds=authorization.MAX_VALIDITY_SECONDS):
        raise ValueError("r4 expiry exceeds the production authorization maximum")


def build_r4_rollover(private_key: Ed25519PrivateKey) -> dict[str, Any]:
    """Build the r4 public receipts and detached signatures from a live key."""
    _preflight(private_key)
    r3_program_path = R3 / "program/phase_c_program_r3.json"
    r3_portable_path = R3 / "manifest/portable_r3.json"
    r3_cost = R3 / "cost/base_cost_receipt_r3.json"
    r3_supplement = R3 / "cost/cost_supplement_r3.json"
    r3_program = _read_json(r3_program_path)
    r3_portable = _read_json(r3_portable_path)
    r4_program_payload = build_phase_c_program_receipt(
        eof_canonicalization_receipt_path=EOF_RECEIPT,
        deep_source_audit_receipt_path=DEEP_SOURCE_AUDIT,
    )

    program_path = R4 / "program/phase_c_program_r4.json"
    portable_path = R4 / "manifest/portable_r4.json"
    shard_paths = {
        "gpu0": R4 / "manifest/shard_stage_a_gpu0_r4.json",
        "gpu1": R4 / "manifest/shard_stage_a_gpu1_r4.json",
        "opening": R4 / "manifest/shard_stage_a_opening_r4.json",
    }
    auth_paths = {
        "gpu0": R4 / "auth/stage_a_execution_gpu0_r4.json",
        "gpu1": R4 / "auth/stage_a_execution_gpu1_r4.json",
        "opening": R4 / "auth/stage_a_opening_r4.json",
    }
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

    # All validations and payload construction complete before the first
    # O_EXCL write.  Once writing begins the bundle is append-only.
    if r3_portable["absolute_cell_root"] != str(CELL_ROOT):
        raise ValueError("r3 portable manifest does not bind the designated absent cell root")
    # Program and portable metadata are not available until their O_EXCL
    # writes complete.  The authorization bodies are therefore constructed
    # after those prerequisite writes but before any signatures are emitted.
    write_json_exclusive(program_path, r4_program_payload)
    validate_phase_c_program_receipt(program_path)
    r4_portable_payload = _portable_payload(r3_portable, program=program_path)
    write_json_exclusive(portable_path, r4_portable_payload)
    r4_shards = {
        name: _shard_payload(payload, portable=portable_path)
        for name, payload in r3_shards.items()
    }
    for name, path in shard_paths.items():
        write_json_exclusive(path, r4_shards[name])
    r4_auths = {
        name: _authorization_payload(
            r3_auths[name], program=program_path, portable=portable_path, shard=shard_paths[name]
        )
        for name in auth_paths
    }
    nonces = [payload["authorization"]["single_use_nonce"] for payload in r4_auths.values()]
    if len(nonces) != len(set(nonces)):
        raise RuntimeError("r4 signer generated a duplicate authorization nonce")
    for name, path in auth_paths.items():
        write_json_exclusive(path, r4_auths[name])
        signature = base64.b64encode(private_key.sign(path.read_bytes()))
        write_bytes_exclusive(path.with_suffix(".sig"), signature)

    r4_program = _read_json(program_path)
    proof = {
        "schema": "m2_post33_phase_c_r4_rollover_semantic_equality_v1",
        "protocol_id": r3_program["protocol_id"],
        "phase_id": r3_program["phase_id"],
        "r3_receipt_root": str(R3),
        "r4_receipt_root": str(R4),
        "r3_cell_root": str(CELL_ROOT),
        "r3_cell_root_exists_at_rollover": False,
        "trust_anchor_rollover": {
            "r3_public_key": r3_auths["gpu0"]["authorization"]["public_key"],
            "r4_public_key": file_metadata(R4_PUBLIC_KEY),
            "algorithm": "Ed25519",
            "private_key_persisted": False,
            "detached_signatures_only": True,
        },
        "program": _program_semantics(r3_program, r4_program),
        "portable_manifest": {
            "only_phase_c_program_receipt_closure_changed": True,
            "r3": file_metadata(r3_portable_path),
            "r4": file_metadata(portable_path),
        },
        "shards": {
            name: {
                "only_portable_manifest_digest_changed": True,
                "r3": file_metadata(R3 / f"manifest/shard_stage_a_{'opening' if name == 'opening' else name}_r3.json"),
                "r4": file_metadata(path),
                "host_id": r4_shards[name]["host_id"],
                "gpu_id": r4_shards[name]["gpu_id"],
                "fold_allowlist": r4_shards[name]["fold_allowlist"],
                "seed_allowlist": r4_shards[name]["seed_allowlist"],
            }
            for name, path in shard_paths.items()
        },
        "cost_seals": {
            "reused_without_reserialization_because_not_trust-anchor_dependent": True,
            "base_cost_receipt": file_metadata(r3_cost),
            "cost_supplement": file_metadata(r3_supplement),
        },
        "authorizations": {
            name: _authorization_semantics(r3_auths[name], r4_auths[name])
            for name in auth_paths
        },
        "authorization_window": {
            "issued_at": _timestamp(ISSUED_AT),
            "expires_at": _timestamp(EXPIRES_AT),
            "validity_hours": (EXPIRES_AT - ISSUED_AT).total_seconds() / 3600,
            "minimum_hkt_margin_hours": 36,
        },
        "forbidden_activity": {
            "gpu_used": False,
            "score_data_accessed": False,
            "endpoint_opened": False,
            "cell_root_created": False,
        },
    }
    proof_path = R4 / "r4_rollover_semantic_equality.json"
    write_json_exclusive(proof_path, proof)
    # Verify each detached capability under the production fixed-anchor API at
    # issuance+one second.  This is CPU-only and does not consume a nonce.
    for name, path in auth_paths.items():
        authorization.verify_signed_authorization(
            path,
            path.with_suffix(".sig"),
            phase_c_program_receipt_path=program_path,
            portable_manifest_path=portable_path,
            shard_manifest_path=shard_paths[name],
            cost_supplement_path=r3_supplement,
            cell_root=CELL_ROOT,
            now=ISSUED_AT + timedelta(seconds=1),
        )
    return {
        "r4_receipt_root": str(R4),
        "program_sha256": sha256_file(program_path),
        "portable_sha256": sha256_file(portable_path),
        "semantic_equality_sha256": sha256_file(proof_path),
        "authorization_sha256": {name: sha256_file(path) for name, path in auth_paths.items()},
        "authorization_signature_sha256": {
            name: sha256_file(path.with_suffix(".sig")) for name, path in auth_paths.items()
        },
    }
