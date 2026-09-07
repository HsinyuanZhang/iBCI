#!/usr/bin/env python3
"""One-shot CPU-only r7 builder and in-memory Stage-A signer.

It writes a fresh public anchor, source program, manifests, and exactly two
seed-42 execution capabilities.  It does not create the r7 cell root, invoke
the matrix, initialize CUDA, serialize a private key, or mint any opening or
Stage-B authority.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import secrets
import socket
import sys
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_authorization_v5_r7 import (  # noqa: E402
    AUTH_SCHEMA,
    ENVELOPE_SCHEMA,
    EVALUATOR,
    PORTABLE_SCHEMA,
    SHARD_SCHEMA,
    selector_policy,
    verify_signed_authorization,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    ARMS,
    PHASE_ID,
    PROTOCOL_ID,
    file_metadata,
    require_canonical_regular_file,
    sha256_file,
    write_bytes_exclusive,
    write_json_exclusive,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r7_program import (  # noqa: E402
    ANCHOR_SCHEMA,
    DEEP_SOURCE_AUDIT,
    PHASE_A_DATA_AUDIT,
    PUBLIC_ANCHOR,
    PUBLIC_KEY,
    R7_CELL_ROOT,
    R7_RECEIPT_ROOT,
    ROLLOVER_ID,
    build_r7_program_receipt,
    source_rows,
    validate_r6d_retirement,
    validate_r7_program_receipt,
)


R3 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3"
COST_RECEIPT = R3 / "cost/base_cost_receipt_r3.json"
COST_SUPPLEMENT = R3 / "cost/cost_supplement_r3.json"
R6D = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6d"
SOURCE = Path(__file__).resolve()
PROGRAM_PATH = R7_RECEIPT_ROOT / "program/phase_c_program_v5_r7.json"
PORTABLE_PATH = R7_RECEIPT_ROOT / "manifest/portable_v5_r7.json"
SHARD_PATHS = {
    "gpu0": R7_RECEIPT_ROOT / "manifest/shard_stage_a_gpu0_v5_r7.json",
    "gpu1": R7_RECEIPT_ROOT / "manifest/shard_stage_a_gpu1_v5_r7.json",
}
AUTH_PATHS = {
    "gpu0": R7_RECEIPT_ROOT / "auth/stage_a_execution_gpu0_v5_r7.json",
    "gpu1": R7_RECEIPT_ROOT / "auth/stage_a_execution_gpu1_v5_r7.json",
}
SUMMARY_PATH = R7_RECEIPT_ROOT / "launch/stage_a_ready_not_launched.json"
PREPARE_RECEIPT = R7_RECEIPT_ROOT / "prelaunch/r7_stage_a_prepare_receipt.json"


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    source = require_canonical_regular_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return payload


def _assert_fresh_targets() -> None:
    for path in (R7_RECEIPT_ROOT, R7_CELL_ROOT, PUBLIC_KEY):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"r7 fresh target already exists: {path}")
    # No previously written selector can be carried forward because its tree
    # must not exist before the new matrix itself claims a cell.
    if R7_CELL_ROOT.exists():
        raise PermissionError("r7 selector/cell root unexpectedly pre-exists")


def _historical_r6d_nonces() -> set[str]:
    nonces: set[str] = set()
    for path in sorted((R6D / "auth").glob("*.json")):
        payload = _read_json(path, "historical r6d authorization")
        body = payload.get("authorization")
        if isinstance(body, Mapping) and isinstance(body.get("single_use_nonce"), str):
            nonces.add(str(body["single_use_nonce"]))
    if not nonces:
        raise PermissionError("r7 nonce audit found no historical r6d authorization inventory")
    return nonces


def _r6d_gpu_assignments() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ("gpu0", "gpu1"):
        payload = _read_json(
            R6D / "manifest" / f"shard_stage_a_{name}_r6d.json", f"r6d {name} shard"
        )
        folds = payload.get("fold_allowlist")
        if (
            not isinstance(folds, list)
            or payload.get("seed_allowlist") != [42]
            or payload.get("arms_in_order") != list(ARMS)
            or not isinstance(payload.get("host_id"), str)
            or not isinstance(payload.get("gpu_id"), str)
        ):
            raise PermissionError("historical r6d Stage-A shard is not a valid physical assignment template")
        result[name] = {
            "host_id": str(payload["host_id"]),
            "gpu_id": str(payload["gpu_id"]),
            "fold_allowlist": list(folds),
        }
    if sorted(result["gpu0"]["fold_allowlist"] + result["gpu1"]["fold_allowlist"]) != list(range(7)):
        raise PermissionError("r7 Stage-A physical assignment does not cover folds 0..6 exactly")
    if result["gpu0"]["host_id"] != result["gpu1"]["host_id"]:
        raise PermissionError("r7 Stage-A GPU assignment host mismatch")
    if socket.gethostname() != result["gpu0"]["host_id"]:
        raise PermissionError("r7 builder host differs from signed Stage-A host")
    return result


def _preflight() -> dict[str, Any]:
    _assert_fresh_targets()
    for source in (SOURCE, EVALUATOR, COST_RECEIPT, COST_SUPPLEMENT, DEEP_SOURCE_AUDIT, PHASE_A_DATA_AUDIT):
        require_canonical_regular_file(source, within=ROOT)
    # Hash the complete v5/r7 source closure before the first r7 artifact is
    # written.  This opens only local code/configuration files, not NWB data.
    rows = source_rows()
    if not rows:
        raise ValueError("r7 source closure is unexpectedly empty")
    retirement = validate_r6d_retirement()
    historical_nonces = _historical_r6d_nonces()
    assignments = _r6d_gpu_assignments()
    return {
        "source_map_entry_count": len(rows),
        "source_map_sha256": sha256_file(SOURCE),
        "retirement": retirement,
        "historical_r6d_nonce_count": len(historical_nonces),
        "assignments": assignments,
    }


def _write_public_anchor(private_key: Ed25519PrivateKey) -> dict[str, Any]:
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    write_bytes_exclusive(PUBLIC_KEY, public_bytes)
    os.chmod(PUBLIC_KEY, 0o444)
    anchor = {
        "schema": ANCHOR_SCHEMA,
        "rollover_id": ROLLOVER_ID,
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "public_key": file_metadata(PUBLIC_KEY),
        "private_key_serialized_or_disk_persisted": False,
        "gpu_used": False,
        "formal_data_accessed": False,
        "score_data_accessed": False,
    }
    write_json_exclusive(PUBLIC_ANCHOR, anchor)
    os.chmod(PUBLIC_ANCHOR, 0o444)
    return anchor


def _portable(program: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PORTABLE_SCHEMA,
        "rollover_id": ROLLOVER_ID,
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "workspace_root": str(ROOT.resolve()),
        "data_root": str((ROOT / "SPINT-main/data/000953").resolve(strict=True)),
        "absolute_cell_root": str(R7_CELL_ROOT.resolve()),
        "same_absolute_paths_required_on_all_hosts": True,
        "program": file_metadata(PROGRAM_PATH),
        "deep_source_audit": dict(program["deep_source_audit_receipt"]),
        "phase_a_data_audit": dict(program["phase_a_data_audit"]),
        "r6d_retirement": dict(program["r6d_retirement"]),
        "selector_policy": selector_policy(),
    }


def _shard(assignment: Mapping[str, Any], portable: Path) -> dict[str, Any]:
    return {
        "schema": SHARD_SCHEMA,
        "rollover_id": ROLLOVER_ID,
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "host_id": assignment["host_id"],
        "gpu_id": assignment["gpu_id"],
        "arms_in_order": list(ARMS),
        "paired_same_host_required": True,
        "absolute_cell_root": str(R7_CELL_ROOT.resolve()),
        "fold_allowlist": list(assignment["fold_allowlist"]),
        "seed_allowlist": [42],
        "portable_manifest_sha256": sha256_file(portable),
    }


def _authorization(
    *,
    name: str,
    nonce: str,
    issued: str,
    expires: str,
    shard: Path,
) -> dict[str, Any]:
    return {
        "schema": ENVELOPE_SCHEMA,
        "authorization": {
            "schema": AUTH_SCHEMA,
            "rollover_id": ROLLOVER_ID,
            "protocol_id": PROTOCOL_ID,
            "phase_id": PHASE_ID,
            "status": "GO",
            "authorization_id": f"m2-post33-phase-c-v5-r7-stage-a-{name}-20260805",
            "single_use_nonce": nonce,
            "issued_at": issued,
            "expires_at": expires,
            "stage": "stage_a",
            "capability_scope": "cell_execution",
            "host_id": _read_json(shard, "r7 shard")["host_id"],
            "gpu_id": _read_json(shard, "r7 shard")["gpu_id"],
            "absolute_cell_root": str(R7_CELL_ROOT.resolve()),
            "fold_allowlist": _read_json(shard, "r7 shard")["fold_allowlist"],
            "seed_allowlist": [42],
            "arms_in_order": list(ARMS),
            "phase_c_program_receipt": file_metadata(PROGRAM_PATH),
            "portable_manifest": file_metadata(PORTABLE_PATH),
            "shard_manifest": file_metadata(shard),
            "evaluator": file_metadata(EVALUATOR),
            "cost_receipt": file_metadata(COST_RECEIPT),
            "cost_supplement": file_metadata(COST_SUPPLEMENT),
            "public_key": file_metadata(PUBLIC_KEY),
            "selector_policy": selector_policy(),
        },
    }


def _sign(private_key: Ed25519PrivateKey, authorization_path: Path) -> Path:
    signature = base64.b64encode(private_key.sign(authorization_path.read_bytes()))
    destination = authorization_path.with_suffix(".sig")
    write_bytes_exclusive(destination, signature)
    os.chmod(destination, 0o444)
    return destination


def prepare() -> dict[str, Any]:
    preflight = _preflight()
    historical_nonces = _historical_r6d_nonces()
    # The private object never has a serialized representation; only this
    # process holds it until both detached signatures are written.
    private_key = Ed25519PrivateKey.generate()
    try:
        _write_public_anchor(private_key)
        program = build_r7_program_receipt()
        write_json_exclusive(PROGRAM_PATH, program)
        validate_r7_program_receipt(PROGRAM_PATH)
        portable = _portable(program)
        write_json_exclusive(PORTABLE_PATH, portable)
        assignments = preflight["assignments"]
        for name, path in SHARD_PATHS.items():
            write_json_exclusive(path, _shard(assignments[name], PORTABLE_PATH))
        issued_dt = datetime.now(timezone.utc).replace(microsecond=0)
        expires_dt = issued_dt + timedelta(hours=29)
        issued, expires = _time(issued_dt), _time(expires_dt)
        nonces = {name: secrets.token_hex(32) for name in AUTH_PATHS}
        if len(set(nonces.values())) != len(nonces) or set(nonces.values()) & historical_nonces:
            raise PermissionError("r7 nonce collision or prohibited r6d nonce reuse")
        signatures: dict[str, Path] = {}
        for name, path in AUTH_PATHS.items():
            write_json_exclusive(
                path,
                _authorization(name=name, nonce=nonces[name], issued=issued, expires=expires, shard=SHARD_PATHS[name]),
            )
            signatures[name] = _sign(private_key, path)
        # Signature verification also replays source/program/cost/portable and
        # exactly binds each fresh root, shard, outer evaluator and nonce.
        for name, path in AUTH_PATHS.items():
            verify_signed_authorization(
                path,
                signatures[name],
                phase_c_program_receipt_path=PROGRAM_PATH,
                portable_manifest_path=PORTABLE_PATH,
                shard_manifest_path=SHARD_PATHS[name],
                cost_supplement_path=COST_SUPPLEMENT,
                cell_root=R7_CELL_ROOT,
            )
        stage_a_cells = sum(
            len(_read_json(path, "r7 shard")["fold_allowlist"]) * len(ARMS)
            for path in SHARD_PATHS.values()
        )
        if stage_a_cells != 14:
            raise PermissionError("r7 Stage-A capability does not cover exactly 14 paired cells")
        summary = {
            "schema": "m2_post33_phase_c_v5_r7_stage_a_ready_not_launched_v1",
            "rollover_id": ROLLOVER_ID,
            "status": "READY_NOT_LAUNCHED_INDEPENDENT_REVIEW_REQUIRED",
            "program": file_metadata(PROGRAM_PATH),
            "portable_manifest": file_metadata(PORTABLE_PATH),
            "stage_a_capabilities": {
                name: {
                    "authorization": file_metadata(path),
                    "signature": file_metadata(signatures[name]),
                    "shard": file_metadata(SHARD_PATHS[name]),
                    "cell_count": len(_read_json(SHARD_PATHS[name], "r7 shard")["fold_allowlist"]) * 2,
                }
                for name, path in AUTH_PATHS.items()
            },
            "stage_a_total_cells": stage_a_cells,
            "cell_root_exists": R7_CELL_ROOT.exists(),
            "selector_records_materialized": False,
            "opening_capability_materialized": False,
            "stage_b_capability_materialized": False,
            "gpu_used": False,
            "formal_data_accessed": False,
            "score_data_accessed": False,
            "private_key_serialized_or_disk_persisted": False,
            "launch_requires_independent_review": True,
        }
        if summary["cell_root_exists"]:
            raise PermissionError("r7 builder must not create a cell/selector root")
        write_json_exclusive(SUMMARY_PATH, summary)
        receipt = {
            "schema": "m2_post33_phase_c_v5_r7_stage_a_prepare_receipt_v1",
            "builder_source": file_metadata(SOURCE),
            "preflight": preflight,
            "summary": file_metadata(SUMMARY_PATH),
            "public_anchor": file_metadata(PUBLIC_ANCHOR),
            "public_key": file_metadata(PUBLIC_KEY),
            "program": file_metadata(PROGRAM_PATH),
            "portable_manifest": file_metadata(PORTABLE_PATH),
            "stage_a_total_cells": 14,
            "nonces": {
                "count": 2,
                "pairwise_distinct": True,
                "not_reused_from_r6d": True,
                "values": nonces,
            },
            "cell_root_exists": False,
            "selector_records_materialized": False,
            "gpu_used": False,
            "formal_data_accessed": False,
            "score_data_accessed": False,
            "private_key_serialized_or_disk_persisted": False,
        }
        write_json_exclusive(PREPARE_RECEIPT, receipt)
        return {
            "prepare_receipt": file_metadata(PREPARE_RECEIPT),
            "summary": file_metadata(SUMMARY_PATH),
            "stage_a_total_cells": 14,
            "cell_root_exists": False,
            "gpu_used": False,
        }
    finally:
        # Drop the only private-key object regardless of a signing/check failure.
        del private_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="write/sign the new r7 Stage-A capability")
    parser.add_argument("--preflight-only", action="store_true", help="CPU-only source/history/fresh-root audit")
    args = parser.parse_args()
    if args.prepare == args.preflight_only:
        raise SystemExit("choose exactly one of --prepare or --preflight-only")
    if args.preflight_only:
        print(json.dumps(_preflight(), sort_keys=True, indent=2))
        return
    print(json.dumps(prepare(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
