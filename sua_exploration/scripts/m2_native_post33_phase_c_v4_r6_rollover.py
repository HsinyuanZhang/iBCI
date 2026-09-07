#!/usr/bin/env python3
"""Build the CPU-only, fresh-root r6b Phase-C Stage-A capability bundle.

This is intentionally an append-only recovery, not an evaluator salvage:

* r3 retains the failed fold-0/fold-1 cells and is never reused;
* r5 remains an archived capability bundle and source-map baseline;
* r6b binds a wholly absent result root and fresh single-use nonces;
* the only production closure changes from r5 are the two post-test
  ``model.eval()`` repairs and the one fixed public-anchor constant update.

The caller supplies a live ``Ed25519PrivateKey``.  No function in this module
accepts, reads, writes, or serializes private-key bytes.  It does not construct
a data module, open an endpoint, initialize CUDA, or create the result root.
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
    require_canonical_regular_file,
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
R5 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r5"
R6 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6b"
R3_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r3"
R6_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6b"
EOF_RECEIPT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_upstream_canonicalization_20260805/upstream_eof_canonicalization.json"
DEEP_SOURCE_AUDIT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_deep_source_20260805/deep_source_audit.json"
R6_PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6b_root_ed25519_public.pem"
ABORTED_R6_RECEIPT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6"
ABORTED_R6_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6"
ABORTED_R6_PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6_root_ed25519_public.pem"
ABORTED_R6_PUBLIC_KEY_SHA256 = "6bcd2de84cdb084162b1e4a53fac27cfbb950f26c9bc8f1f1c5976c424bd4c18"
REGRESSION_TEST = ROOT / "sua_exploration/tests/test_m2_native_post33_phase_c_v4.py"
LEGACY_SHARED_Z4_QUEUE = ROOT / "sua_exploration/scripts/queue_t4_paired_view_c1_shared_zero4_after_phase_c_v2.sh"
FROZEN_C1_BRIDGE = ROOT / "sua_exploration/docs/C1_TO_NATIVE_MUA_CLAIM_BRIDGE_AUDIT_20260804.md"
POSTSEAL_ANALYSIS_ADDENDUM = ROOT / "sua_exploration/docs/PHASE_C_ATTRIBUTION_POWER_ADDENDUM_20260805.md"

ALLOWED_PRODUCTION_SOURCE_DELTAS = frozenset(
    {
        "SPINT-main/src/evaluate_post33_phase_c_v4.py",
        "streaming_calibration_exp/src/evaluate_post33_phase_c_v4.py",
        "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py",
    }
)
R5_PUBLIC_KEY_LITERAL = (
    'sua_exploration/configs/m2_native_post33_phase_c_v4_r5_root_ed25519_public.pem'
)
R5_PUBLIC_KEY_SHA256_LITERAL = "cdaff17a44d5358682059a389fced1329a36841a70b76f96c3b065fb89c4bdb2"
R6_PUBLIC_KEY_LITERAL = (
    'sua_exploration/configs/m2_native_post33_phase_c_v4_r6b_root_ed25519_public.pem'
)
R6_PUBLIC_KEY_SHA256_LITERAL = "5e8b8bcfb04c249691b58f0f87fb7b3614d163eb8b8da3fab489a159c9bac1bb"


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


def _program_semantics(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    for field in (
        "schema", "protocol_id", "phase_id", "absolute_workspace_root",
        "score_data_accessed", "formal_data_accessed", "gpu_used",
        "phase_a_b_eof_canonicalization", "deep_source_audit_receipt",
    ):
        if before.get(field) != after.get(field):
            raise ValueError(f"r6b changed sealed program semantic field: {field}")
    old_rows, new_rows = _source_rows(before), _source_rows(after)
    if set(old_rows) != set(new_rows):
        raise ValueError("r6b changed source closure membership")
    deltas = {
        path: {"r5": old_rows[path], "r6b": new_rows[path]}
        for path in sorted(old_rows)
        if old_rows[path] != new_rows[path]
    }
    if set(deltas) != ALLOWED_PRODUCTION_SOURCE_DELTAS:
        raise ValueError(
            "r6b production source delta is not exactly the two evaluator repairs and fixed anchor"
        )
    return {
        "source_closure_membership_equal": True,
        "all_program_semantic_fields_equal": True,
        "only_allowed_production_source_deltas": deltas,
    }


def _validate_eval_mode_repair(relative_path: str) -> dict[str, Any]:
    path = require_canonical_regular_file(ROOT / relative_path, within=ROOT)
    source = path.read_text(encoding="utf-8")
    test_position = source.index("trainer.test(")
    eval_position = source.index("model.eval()", test_position)
    benchmark_position = source.index("profiler.benchmark_online_b1(", eval_position)
    if not test_position < eval_position < benchmark_position:
        raise ValueError(f"r6b {relative_path} does not restore eval mode before the benchmark")
    return {
        "source": file_metadata(path),
        "repair": "model.eval() after Trainer.test and before benchmark_online_b1",
        "trainer_test_position": test_position,
        "model_eval_position": eval_position,
        "benchmark_position": benchmark_position,
    }


def _validate_regression_test() -> dict[str, Any]:
    path = require_canonical_regular_file(REGRESSION_TEST, within=ROOT)
    source = path.read_text(encoding="utf-8")
    required = (
        "def test_workers_restore_eval_mode_before_cached_deployment_benchmark()",
        "SPINT_ROOT / \"src/evaluate_post33_phase_c_v4.py\"",
        "STREAMING_ROOT / \"src/evaluate_post33_phase_c_v4.py\"",
        "source.index(\"model.eval()\")",
        "source.index(\"profiler.benchmark_online_b1(\")",
        "assert eval_position < benchmark_position",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise ValueError(f"r6b eval-mode regression test is incomplete: {missing}")
    return {
        "test_file": file_metadata(path),
        "test_name": "test_workers_restore_eval_mode_before_cached_deployment_benchmark",
        "scope": "both Phase-C evaluator workers; requires model.eval before cached B=1 benchmark",
        "static_contract_verified": True,
    }


def _fixed_anchor_exact_diff(r5_program: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the authorization source changes only the two anchor literals.

    r5 records the old source digest but intentionally does not duplicate its
    source bytes.  Replacing exactly the r6b literals in the live source with
    the archived r5 literals reconstructs those sealed bytes in memory.
    """
    relative = "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py"
    old_row = _source_rows(r5_program).get(relative)
    if old_row is None:
        raise ValueError("r5 authorization source is absent from the archived closure")
    path = require_canonical_regular_file(ROOT / relative, within=ROOT)
    current = path.read_bytes()
    new_path = R6_PUBLIC_KEY_LITERAL.encode("utf-8")
    new_hash = R6_PUBLIC_KEY_SHA256_LITERAL.encode("ascii")
    if current.count(new_path) != 1 or current.count(new_hash) != 1:
        raise ValueError("r6b authorization source does not have exactly one fixed-anchor literal pair")
    reconstructed = current.replace(new_path, R5_PUBLIC_KEY_LITERAL.encode("utf-8"))
    reconstructed = reconstructed.replace(new_hash, R5_PUBLIC_KEY_SHA256_LITERAL.encode("ascii"))
    if sha256_file(path) == old_row["sha256"]:
        raise ValueError("r6b authorization source unexpectedly equals the r5 source")
    import hashlib

    if hashlib.sha256(reconstructed).hexdigest() != old_row["sha256"] or len(reconstructed) != old_row["size_bytes"]:
        raise ValueError("r6b authorization source has drift beyond the two fixed-anchor literals")
    if authorization.PUBLIC_KEY != R6_PUBLIC_KEY or authorization.PUBLIC_KEY_SHA256 != R6_PUBLIC_KEY_SHA256_LITERAL:
        raise PermissionError("production verifier is not pinned to the r6b fixed anchor")
    return {
        "relative_path": relative,
        "r5_sealed_source": old_row,
        "r6b_source": file_metadata(path),
        "proof_method": "replace exactly two r6b fixed-anchor literals in memory and recover r5 SHA-256",
        "r5_source_sha256_reconstructed_in_memory": hashlib.sha256(reconstructed).hexdigest(),
        "verifier_logic_changed": False,
        "runtime_gate_changed": False,
        "unified_diff": [
            f'-PUBLIC_KEY = ROOT / "{R5_PUBLIC_KEY_LITERAL}"',
            f'-PUBLIC_KEY_SHA256 = "{R5_PUBLIC_KEY_SHA256_LITERAL}"',
            f'+PUBLIC_KEY = ROOT / "{R6_PUBLIC_KEY_LITERAL}"',
            f'+PUBLIC_KEY_SHA256 = "{R6_PUBLIC_KEY_SHA256_LITERAL}"',
        ],
    }


def _portable(r5: Mapping[str, Any], program: Path) -> dict[str, Any]:
    payload = dict(r5)
    payload["absolute_cell_root"] = str(R6_CELL_ROOT.resolve())
    closures = [dict(row) for row in r5["hash_closures"]]
    for row in closures:
        if row.get("role") == "phase_c_program_receipt":
            row.clear()
            row.update({"role": "phase_c_program_receipt", **file_metadata(program)})
    payload["hash_closures"] = closures
    return payload


def _shard(r5: Mapping[str, Any], portable: Path) -> dict[str, Any]:
    payload = dict(r5)
    payload["absolute_cell_root"] = str(R6_CELL_ROOT.resolve())
    payload["portable_transfer_manifest_sha256"] = sha256_file(portable)
    return payload


def _auth(
    r5: Mapping[str, Any], *, program: Path, portable: Path, shard: Path,
    issued: datetime, expires: datetime,
) -> dict[str, Any]:
    body = dict(r5["authorization"])
    original_id = str(body["authorization_id"])
    body["authorization_id"] = original_id.replace("_r5-", "_r6b-")
    if body["authorization_id"] == original_id:
        raise ValueError("r5 authorization lacks the required rollover id marker")
    body["single_use_nonce"] = secrets.token_hex(32)
    body["issued_at"] = _time(issued)
    body["expires_at"] = _time(expires)
    body["absolute_cell_root"] = str(R6_CELL_ROOT.resolve())
    body["phase_c_program_receipt"] = file_metadata(program)
    body["portable_manifest"] = file_metadata(portable)
    body["shard_manifest"] = file_metadata(shard)
    body["public_key"] = file_metadata(R6_PUBLIC_KEY)
    return {"schema": "m2_post33_phase_c_signed_authorization_envelope_v4", "authorization": body}


def _validate_portable_semantics(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    permitted = {"absolute_cell_root", "hash_closures"}
    changed = {key for key in before if before[key] != after[key]}
    if changed != permitted or set(before) != set(after):
        raise ValueError("r6b portable manifest changed fields outside fresh root/program closure")
    old_closures = {str(row["role"]): row for row in before["hash_closures"]}
    new_closures = {str(row["role"]): row for row in after["hash_closures"]}
    for role in old_closures:
        if role != "phase_c_program_receipt" and old_closures[role] != new_closures[role]:
            raise ValueError("r6b changed a frozen non-program portable closure")
    return {
        "only_fresh_root_and_program_receipt_closure_changed": True,
        "changed_fields": sorted(changed),
    }


def _validate_shard_semantics(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    permitted = {"absolute_cell_root", "portable_transfer_manifest_sha256"}
    changed = {key for key in before if before[key] != after[key]}
    if changed != permitted or set(before) != set(after):
        raise ValueError("r6b shard changed fields outside fresh root/portable digest")
    for field in (
        "schema", "protocol_id", "phase_id", "host_id", "gpu_id", "fold_allowlist",
        "seed_allowlist", "arms_in_order", "paired_same_host_required",
    ):
        if before[field] != after[field]:
            raise ValueError(f"r6b shard changed protected field: {field}")
    return {"only_fresh_root_and_portable_digest_changed": True, "changed_fields": sorted(changed)}


def _validate_auth_semantics(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    old, new = before["authorization"], after["authorization"]
    if set(old) != set(new):
        raise ValueError("r6b authorization key-set changed")
    permitted = {
        "authorization_id", "single_use_nonce", "issued_at", "expires_at",
        "absolute_cell_root", "phase_c_program_receipt", "portable_manifest",
        "shard_manifest", "public_key",
    }
    changed = {key for key in old if old[key] != new[key]}
    if changed != permitted:
        raise ValueError("r6b authorization changed protected scope or evaluator binding")
    for field in (
        "schema", "protocol_id", "phase_id", "status", "stage", "capability_scope",
        "host_id", "gpu_id", "fold_allowlist", "seed_allowlist", "arms_in_order",
        "evaluator", "cost_receipt", "cost_supplement",
    ):
        if old[field] != new[field]:
            raise ValueError(f"r6b authorization changed protected field: {field}")
    return {"changed_authorization_metadata_only": sorted(changed)}


def _r3_immutable_failure_boundary() -> dict[str, Any]:
    """Bind the failed history by metadata only; never inspect opaque payloads."""
    expected = []
    for fold in (0, 1):
        base = (
            R3_CELL_ROOT / "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1" / "PHASE_C_V4"
            / "cells" / "arm-spint" / f"fold-{fold}" / "seed-42" / "control"
        )
        for name in ("ownership.json", "status.started.json", "status.failed.json"):
            expected.append(file_metadata(require_canonical_regular_file(base / name, within=R3_CELL_ROOT)))
    if not R3_CELL_ROOT.is_dir() or R3_CELL_ROOT.is_symlink():
        raise ValueError("r3 failed result root is no longer a canonical directory")
    return {
        "r3_result_root": str(R3_CELL_ROOT.resolve()),
        "r3_result_root_exists": True,
        "r3_failed_spint_control_receipts": expected,
        "r3_reuse_forbidden": True,
        "score_or_endpoint_payload_opened": False,
    }


def _aborted_r6_anchor_boundary() -> dict[str, Any]:
    """Bind the first r6 anchor that was never used after a pre-write abort."""
    if not ABORTED_R6_PUBLIC_KEY.is_file() or ABORTED_R6_PUBLIC_KEY.is_symlink():
        raise ValueError("burned r6 public anchor is missing/non-canonical")
    if sha256_file(ABORTED_R6_PUBLIC_KEY) != ABORTED_R6_PUBLIC_KEY_SHA256:
        raise ValueError("burned r6 public anchor digest drift")
    if ABORTED_R6_RECEIPT_ROOT.exists() or ABORTED_R6_CELL_ROOT.exists():
        raise ValueError("r6b refuses to overwrite or coexist with a materialized burned r6 root")
    return {
        "status": "ABORTED_PREWRITE_BURNED_ANCHOR",
        "public_key": file_metadata(ABORTED_R6_PUBLIC_KEY),
        "abort_reason": (
            "upstream Phase-A source closure rejected post-seal C1 bridge-document drift "
            "before the first r6 receipt, nonce, cell-root, GPU, endpoint, or score write"
        ),
        "r6_receipt_root_exists": False,
        "r6_cell_root_exists": False,
        "authorization_nonce_claimed": False,
        "gpu_used": False,
        "endpoint_opened": False,
        "score_data_accessed": False,
        "formal_data_accessed": False,
        "private_key_persisted": False,
        "r6_public_anchor_reused": False,
    }


def _frozen_c1_document_boundary() -> dict[str, Any]:
    """Bind the restored Phase-A document and its append-only analysis addendum."""
    phase_a = _read_json(
        ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_draft_prelaunch_v2_hash_hardened_20260804/draft_scorefree_receipt.json"
    )
    expected = phase_a.get("source_map", {}).get(
        "sua_exploration/docs/C1_TO_NATIVE_MUA_CLAIM_BRIDGE_AUDIT_20260804.md"
    )
    if not isinstance(expected, Mapping):
        raise ValueError("Phase-A sealed source map lacks the C1 bridge document")
    frozen = require_canonical_regular_file(FROZEN_C1_BRIDGE, within=ROOT)
    observed = file_metadata(frozen)
    expected_compact = {
        "size_bytes": expected.get("size_bytes"),
        "sha256": expected.get("sha256"),
    }
    if {"size_bytes": observed["size_bytes"], "sha256": observed["sha256"]} != expected_compact:
        raise ValueError("restored C1 bridge document does not exactly match the Phase-A sealed bytes")
    addendum = require_canonical_regular_file(POSTSEAL_ANALYSIS_ADDENDUM, within=ROOT)
    return {
        "frozen_phase_a_document": observed,
        "sealed_size_bytes": expected_compact["size_bytes"],
        "sealed_sha256": expected_compact["sha256"],
        "restored_exactly": True,
        "postseal_addendum": file_metadata(addendum),
        "addendum_is_outside_phase_c_runtime_source_closure": True,
    }


def _legacy_shared_z4_boundary() -> dict[str, Any]:
    """Make the pre-existing r5/r3-bound Z4 release gate non-transferable."""
    path = require_canonical_regular_file(LEGACY_SHARED_Z4_QUEUE, within=ROOT)
    source = path.read_text(encoding="utf-8")
    r3_suffix = "m2_native_post33_phase_c_v4_cells_20260805_r3"
    r6b_suffix = "m2_native_post33_phase_c_v4_cells_20260805_r6b"
    if r3_suffix not in source or r6b_suffix in source:
        raise ValueError("legacy shared-Z4 release wrapper root binding unexpectedly changed")
    return {
        "wrapper": file_metadata(path),
        "bound_phase_c_result_root": str(R3_CELL_ROOT.resolve()),
        "r6b_root_not_accepted": True,
        "r6b_authorizations_do_not_authorize_shared_z4_release": True,
    }


def _launch_argv(*, shard: Path, auth: Path, program: Path, portable: Path, supplement: Path) -> list[str]:
    return [
        "/home/xinyuan/miniconda3/envs/spint/bin/python", "-u",
        str(ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py"),
        "--execute",
        "--cell-root", str(R6_CELL_ROOT.resolve()),
        "--workspace-root", str(ROOT.resolve()),
        "--data-root", str((ROOT / "SPINT-main/data/000953").resolve(strict=True)),
        "--shard-manifest", str(shard.resolve(strict=True)),
        "--portable-manifest", str(portable.resolve(strict=True)),
        "--program-receipt", str(program.resolve(strict=True)),
        "--cost-supplement", str(supplement.resolve(strict=True)),
        "--authorization", str(auth.resolve(strict=True)),
        "--authorization-signature", str(auth.with_suffix(".sig").resolve(strict=True)),
    ]


def _preflight(private_key: Ed25519PrivateKey, *, issued: datetime, expires: datetime) -> dict[str, Any]:
    if R6.exists():
        raise FileExistsError(f"append-only r6b receipt root already exists: {R6}")
    if R6_CELL_ROOT.exists():
        raise FileExistsError("r6b fresh result root must be absent before any authorization is issued")
    if not R3.is_dir() or not R5.is_dir():
        raise FileNotFoundError("r3/r5 historical receipt roots are required")
    if authorization.MAX_VALIDITY_SECONDS != 30 * 3600:
        raise PermissionError("r6b refuses a relaxed Phase-C authorization maximum")
    if authorization.PUBLIC_KEY != R6_PUBLIC_KEY:
        raise PermissionError("r6b production verifier is not pinned to the r6b public anchor")
    public = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    if R6_PUBLIC_KEY.read_bytes() != public or sha256_file(R6_PUBLIC_KEY) != authorization.PUBLIC_KEY_SHA256:
        raise PermissionError("in-memory r6b signer does not match the fixed r6b public anchor")
    now = datetime.now(timezone.utc)
    if not (issued <= now <= expires):
        raise PermissionError("r6b issuance time is not currently valid")
    if expires - issued != timedelta(hours=29) or expires - issued > timedelta(seconds=authorization.MAX_VALIDITY_SECONDS):
        raise PermissionError("r6b validity must be exactly 29 hours and within the 30-hour maximum")
    return _r3_immutable_failure_boundary()


def build_r6b_rollover(private_key: Ed25519PrivateKey) -> dict[str, Any]:
    """Create and CPU-verify an r6b Stage-A package using an ephemeral signer."""
    issued = datetime.now(timezone.utc).replace(microsecond=0)
    expires = issued + timedelta(hours=29)
    r3_immutability = _preflight(private_key, issued=issued, expires=expires)
    aborted_r6 = _aborted_r6_anchor_boundary()
    frozen_c1_documents = _frozen_c1_document_boundary()
    legacy_shared_z4 = _legacy_shared_z4_boundary()

    r5_program_path = R5 / "program/phase_c_program_r5.json"
    r5_portable_path = R5 / "manifest/portable_r5.json"
    r3_cost = R3 / "cost/base_cost_receipt_r3.json"
    r3_supplement = R3 / "cost/cost_supplement_r3.json"
    r5_program, r5_portable = _read_json(r5_program_path), _read_json(r5_portable_path)
    r5_shards = {
        "gpu0": _read_json(R5 / "manifest/shard_stage_a_gpu0_r5.json"),
        "gpu1": _read_json(R5 / "manifest/shard_stage_a_gpu1_r5.json"),
        "opening": _read_json(R5 / "manifest/shard_stage_a_opening_r5.json"),
    }
    r5_auths = {
        "gpu0": _read_json(R5 / "auth/stage_a_execution_gpu0_r5.json"),
        "gpu1": _read_json(R5 / "auth/stage_a_execution_gpu1_r5.json"),
        "opening": _read_json(R5 / "auth/stage_a_opening_r5.json"),
    }
    if r5_portable["absolute_cell_root"] != str(R3_CELL_ROOT.resolve()):
        raise ValueError("r5 portable manifest no longer binds the archived r3 root")
    for payload in r5_auths.values():
        if payload["authorization"]["evaluator"] != file_metadata(authorization.EVALUATOR):
            raise ValueError("r6b refuses a changed evaluator frontdoor binding")

    r6_program_payload = build_phase_c_program_receipt(
        eof_canonicalization_receipt_path=EOF_RECEIPT,
        deep_source_audit_receipt_path=DEEP_SOURCE_AUDIT,
    )
    program_semantics = _program_semantics(r5_program, r6_program_payload)
    eval_repairs = {
        path: _validate_eval_mode_repair(path)
        for path in (
            "SPINT-main/src/evaluate_post33_phase_c_v4.py",
            "streaming_calibration_exp/src/evaluate_post33_phase_c_v4.py",
        )
    }
    regression_test = _validate_regression_test()
    anchor_diff = _fixed_anchor_exact_diff(r5_program)

    program = R6 / "program/phase_c_program_r6b.json"
    portable = R6 / "manifest/portable_r6b.json"
    shards = {
        "gpu0": R6 / "manifest/shard_stage_a_gpu0_r6b.json",
        "gpu1": R6 / "manifest/shard_stage_a_gpu1_r6b.json",
        "opening": R6 / "manifest/shard_stage_a_opening_r6b.json",
    }
    auths = {
        "gpu0": R6 / "auth/stage_a_execution_gpu0_r6b.json",
        "gpu1": R6 / "auth/stage_a_execution_gpu1_r6b.json",
        "opening": R6 / "auth/stage_a_opening_r6b.json",
    }

    # All source, trust, and historical-root checks have completed.  Every
    # write below is O_EXCL/append-only and targets only the r6b receipt root.
    write_json_exclusive(program, r6_program_payload)
    validate_phase_c_program_receipt(program)
    r6_portable = _portable(r5_portable, program)
    write_json_exclusive(portable, r6_portable)
    r6_shards = {name: _shard(payload, portable) for name, payload in r5_shards.items()}
    for name, path in shards.items():
        write_json_exclusive(path, r6_shards[name])
    r6_auths = {
        name: _auth(r5_auths[name], program=program, portable=portable, shard=shards[name], issued=issued, expires=expires)
        for name in auths
    }
    nonces = [value["authorization"]["single_use_nonce"] for value in r6_auths.values()]
    if len(nonces) != 3 or len(set(nonces)) != 3:
        raise RuntimeError("r6b requires exactly three distinct 256-bit nonces")
    for name, path in auths.items():
        write_json_exclusive(path, r6_auths[name])
        write_bytes_exclusive(path.with_suffix(".sig"), base64.b64encode(private_key.sign(path.read_bytes())))

    launch = {
        "schema": "m2_post33_phase_c_r6b_stage_a_launch_commands_v1",
        "status": "PREPARED_NOT_EXECUTED",
        "cell_root_must_be_absent_until_command_execution": str(R6_CELL_ROOT.resolve()),
        "commands": {
            "gpu0": {
                "fold_allowlist": r6_shards["gpu0"]["fold_allowlist"],
                "cuda_visible_devices": r6_shards["gpu0"]["gpu_id"],
                "argv": _launch_argv(shard=shards["gpu0"], auth=auths["gpu0"], program=program, portable=portable, supplement=r3_supplement),
            },
            "gpu1": {
                "fold_allowlist": r6_shards["gpu1"]["fold_allowlist"],
                "cuda_visible_devices": r6_shards["gpu1"]["gpu_id"],
                "argv": _launch_argv(shard=shards["gpu1"], auth=auths["gpu1"], program=program, portable=portable, supplement=r3_supplement),
            },
        },
        "opening_capability_issued_but_not_authorized_to_consume": file_metadata(auths["opening"]),
        "legacy_shared_z4_release_remains_blocked": legacy_shared_z4,
        "gpu_used": False,
        "score_data_accessed": False,
        "formal_data_accessed": False,
    }
    launch_path = R6 / "launch/stage_a_commands_r6b.json"
    write_json_exclusive(launch_path, launch)
    proof = {
        "schema": "m2_post33_phase_c_r6b_fresh_root_rollover_semantic_equality_v1",
        "protocol_id": r5_program["protocol_id"],
        "phase_id": r5_program["phase_id"],
        "r3_failed_history": r3_immutability,
        "aborted_prewrite_r6_anchor": aborted_r6,
        "frozen_c1_document_and_postseal_addendum": frozen_c1_documents,
        "legacy_shared_z4_release_boundary": legacy_shared_z4,
        "r5_archived_receipt_root": str(R5.resolve()),
        "r6b_receipt_root": str(R6.resolve()),
        "r6b_fresh_result_root": str(R6_CELL_ROOT.resolve()),
        "r6b_fresh_result_root_exists_at_issuance": False,
        "single_fixed_trust_anchor": {
            "algorithm": "Ed25519",
            "public_key": file_metadata(R6_PUBLIC_KEY),
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
        "program": program_semantics,
        "evaluator_eval_mode_repairs": eval_repairs,
        "regression_test": regression_test,
        "fixed_anchor_exact_source_diff": anchor_diff,
        "portable_manifest": {
            **_validate_portable_semantics(r5_portable, r6_portable),
            "r5": file_metadata(r5_portable_path),
            "r6b": file_metadata(portable),
        },
        "shards": {
            name: {
                **_validate_shard_semantics(r5_shards[name], r6_shards[name]),
                "r5": file_metadata(R5 / f"manifest/shard_stage_a_{'opening' if name == 'opening' else name}_r5.json"),
                "r6b": file_metadata(path),
                "host_id": r6_shards[name]["host_id"],
                "gpu_id": r6_shards[name]["gpu_id"],
                "fold_allowlist": r6_shards[name]["fold_allowlist"],
                "seed_allowlist": r6_shards[name]["seed_allowlist"],
            }
            for name, path in shards.items()
        },
        "cost_seals": {
            "reused_without_reserialization_because_protocol_costs_are_unchanged": True,
            "base_cost_receipt": file_metadata(r3_cost),
            "cost_supplement": file_metadata(r3_supplement),
        },
        "authorizations": {
            name: _validate_auth_semantics(r5_auths[name], r6_auths[name]) for name in auths
        },
        "launch_receipt": file_metadata(launch_path),
        "forbidden_activity": {
            "gpu_used": False,
            "score_data_accessed": False,
            "formal_data_accessed": False,
            "endpoint_opened": False,
            "r3_or_r5_history_rewritten": False,
            "r6b_result_root_created": False,
        },
    }
    proof_path = R6 / "r6b_fresh_root_rollover_semantic_equality.json"
    write_json_exclusive(proof_path, proof)
    for name, path in auths.items():
        if R6_CELL_ROOT.exists():
            raise RuntimeError("r6b fresh result root appeared before authorization verification")
        authorization.verify_signed_authorization(
            path,
            path.with_suffix(".sig"),
            phase_c_program_receipt_path=program,
            portable_manifest_path=portable,
            shard_manifest_path=shards[name],
            cost_supplement_path=r3_supplement,
            cell_root=R6_CELL_ROOT,
            now=issued + timedelta(seconds=1),
        )
    if R6_CELL_ROOT.exists():
        raise RuntimeError("r6b signer unexpectedly created the fresh result root")
    return {
        "r6b_receipt_root": str(R6),
        "r6b_fresh_cell_root": str(R6_CELL_ROOT),
        "program_sha256": sha256_file(program),
        "portable_sha256": sha256_file(portable),
        "semantic_equality_sha256": sha256_file(proof_path),
        "launch_receipt_sha256": sha256_file(launch_path),
        "authorization_sha256": {name: sha256_file(path) for name, path in auths.items()},
        "authorization_signature_sha256": {name: sha256_file(path.with_suffix(".sig")) for name, path in auths.items()},
    }
