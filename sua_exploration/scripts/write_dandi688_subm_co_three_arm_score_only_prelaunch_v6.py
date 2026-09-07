#!/usr/bin/env python3
"""Write/verify the blocked append-only external sub-M three-arm V6 package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sua_exploration.mc_maze import subm_co_three_arm_score_only_v6 as core  # noqa: E402


DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_score_only_prelaunch_v6"
V5_ROOT = ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_score_only_prelaunch_v5"
V5_PINS = {
    "blocked_prelaunch_draft.json": ("0d672e2cf89d0e7ef69af1cfa83cd6d45eccd0c9b888c4dbb2e0452d873412c0", 19_476),
    "receipt.json": ("56f1b9a5648005887f0118505dcec231311b1ec9c6e8511f1401bdf3cb17a60a", 1_585),
    "blocked_seal.json": ("bf7dd0458bb83386505c6b87b0c2792ce5603e0814ea209ab4d867cc23aa7562", 559),
}
PARITY_BUNDLE_SHA256 = "04ff3ead0f14446605e3ce54b4c37cf86f94689a2f354bb5c8aab4c7bb2f7de5"
SOURCE_PATHS = (
    "sua_exploration/docs/DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V6.md",
    "sua_exploration/mc_maze/subm_co_three_arm_score_only_v6.py",
    "sua_exploration/scripts/run_dandi688_subm_co_three_arm_score_only_v6.py",
    "sua_exploration/scripts/write_dandi688_subm_co_three_arm_score_only_prelaunch_v6.py",
    "sua_exploration/tests/test_dandi688_subm_co_three_arm_score_only_v6.py",
)


class StaticV6Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticV6Error(message)


def _pin_read(path: Path, digest: str, size: int, mode: str = "0444") -> bytes:
    try:
        observed = core.fd_read_regular(path, root=path.parent, expected_mode=mode, expected_sha256=digest, expected_bytes=size)
    except core.V6LedgerError as exc:
        raise StaticV6Error(str(exc)) from exc
    return observed.raw


def _canonical_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticV6Error(f"malformed {label}") from exc
    require(isinstance(value, dict) and raw == core.canonical_bytes(value), f"noncanonical {label}")
    return value


def _v5_predecessor() -> tuple[dict[str, Any], dict[str, Any]]:
    payloads = {}; pins = {}
    for name, (digest, size) in V5_PINS.items():
        path = V5_ROOT / name
        raw = _pin_read(path, digest, size)
        payloads[name] = _canonical_json(raw, f"V5 {name}")
        pins[name] = {"path": str(path), "sha256": digest, "bytes": size, "mode": "0444"}
    draft = payloads["blocked_prelaunch_draft.json"]
    require(draft.get("status") == "BLOCKED_MISSING_ZERO4_TERMINALS", "V5 predecessor status drift")
    require(draft.get("external_verified_authorization_grant") is None, "V5 predecessor unexpectedly authorized")
    return draft, pins


def _revalidate_existing_evidence(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in slots:
        closure = row.get("closure")
        if closure is None:
            continue
        path = ROOT / closure["path"]
        _pin_read(path, closure["sha256"], closure["bytes"], closure["mode"])
        result.append({"arm": row["arm"], "seed": row["seed"], "checkpoint_slot_binding_sha256": core.slot_binding(row), "closure": closure})
    require(len(result) == 6, "existing terminal evidence count drift")
    return result


def _source_map() -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"missing V6 source: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def build_blocked_draft() -> dict[str, Any]:
    predecessor, predecessor_pins = _v5_predecessor()
    old_contract = predecessor["contract"]
    slots = core.blocked_checkpoint_slots()
    core.validate_slots(slots, require_complete=False)
    existing = _revalidate_existing_evidence(slots)
    sources = _source_map()
    policy = {
        "schema": "dandi_000688_subm_three_arm_frozen_policy_v6",
        "frozen_predecessor_sha256": V5_PINS["blocked_prelaunch_draft.json"][0],
        "cohort": old_contract["cohort"],
        "query_counts": old_contract["query_window_count_by_asset_id"],
        "reviewed_checkpoint_slots": slots,
        "reviewed_checkpoint_slots_sha256": core.canonical_sha256(slots),
        "verified_closure_bundle_sha256": None,
        "source_snapshot_sha256": core.canonical_sha256(sources),
        "parity_bundle_sha256": PARITY_BUNDLE_SHA256,
        "runtime_identity": core.runtime_identity(), "cpu_policy": core.CPU_POLICY,
        "repository_root": str(ROOT), "closure_root": None,
        "output_parent": None, "external_parent": None, "claim_root": None,
        "public_key": None,
    }
    contract = core.build_expected_contract(policy)
    missing = [row for row in slots if row["closure"] is None]
    require([(row["arm"], row["seed"]) for row in missing] == [("shared_zero4", 42), ("shared_zero4", 43), ("shared_zero4", 44)], "zero4 blocker drift")
    return {
        "schema_version": 6,
        "kind": "dandi_000688_subm_three_arm_cryptographic_control_plane_blocked_v6",
        "status": "BLOCKED_MISSING_ZERO4_TERMINALS", "append_only": True,
        "v5_predecessor": predecessor_pins,
        "frozen_policy": policy, "contract": contract,
        "contract_sha256": contract["contract_sha256"],
        "missing_checkpoint_slots": missing,
        "existing_terminal_evidence_revalidated_without_checkpoint_open": existing,
        "verified_grant": None,
        "source_map": sources, "source_snapshot_sha256": policy["source_snapshot_sha256"],
        "repairs": {
            "grant_only_from_real_ed25519_verify_expiry_and_atomic_nonce_claim": True,
            "contract_exact_reconstruction_not_hash_only": True,
            "future_zero4_closure_and_authority_receipt_live_fd_verification": True,
            "full_verified_grant_sha256_through_artifact_cell_resume_aggregate": True,
            "existing_aggregate_exact_recomputation_from_270_cells": True,
            "safe_npz_and_internal_variance_weighted_r2_no_caller_r2": True,
            "fd_o_nofollow_fstat_hashing": True,
        },
        "operations_by_this_package": {
            "external_subm_nwb_files_opened": 0, "checkpoint_files_opened": 0,
            "torch_imports": 0, "model_forward_calls": 0, "external_r2_computations": 0,
            "gpu_used": False, "signatures_created": 0,
            "external_capability_created": False, "synthetic_fixtures_only": True,
        },
    }


def build_blocked_receipt() -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_blocked_draft()
    receipt = {
        "schema_version": 6, "kind": "dandi_000688_subm_three_arm_blocked_receipt_v6",
        "status": draft["status"], "append_only": True,
        "draft": {"path": "blocked_prelaunch_draft.json", "sha256": core.canonical_sha256(draft)},
        "contract_sha256": draft["contract_sha256"],
        "missing_checkpoint_slots": draft["missing_checkpoint_slots"],
        "not_an_authorization_or_verified_grant": True,
        "not_an_executable_prelaunch": True, "external_capability_created": False,
        "operations": draft["operations_by_this_package"],
    }
    return draft, receipt


def _write_immutable(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    raw = core.canonical_bytes(value)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc:
        raise StaticV6Error(f"append-only path exists: {path}") from exc
    try:
        os.write(descriptor, raw); os.fsync(descriptor); os.fchmod(descriptor, 0o444)
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o444 and metadata.st_size == len(raw), "immutable fd write failed")
    finally:
        os.close(descriptor)
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": "0444"}


def write_blocked_prelaunch(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists() and not output_dir.is_symlink(), "V6 blocked output already exists")
    draft, receipt = build_blocked_receipt()
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_pin = _write_immutable(output_dir / "blocked_prelaunch_draft.json", draft)
    require(draft_pin["sha256"] == receipt["draft"]["sha256"], "draft receipt mismatch")
    receipt_pin = _write_immutable(output_dir / "receipt.json", receipt)
    seal = {
        "schema_version": 6, "kind": "dandi_000688_subm_three_arm_blocked_seal_v6",
        "status": draft["status"], "append_only": True,
        "artifacts": [draft_pin, receipt_pin], "verified_grant_created": False,
        "executable_prelaunch_sealed": False, "external_capability_created": False,
    }
    seal_pin = _write_immutable(output_dir / "blocked_seal.json", seal)
    return {"output_dir": str(output_dir), "status": draft["status"], "draft_sha256": draft_pin["sha256"], "receipt_sha256": receipt_pin["sha256"], "blocked_seal_sha256": seal_pin["sha256"], "verified_grant_created": False, "executable_prelaunch_sealed": False, "external_capability_created": False}


def load_stored_blocked_prelaunch(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = output_dir.resolve(); names = ("blocked_prelaunch_draft.json", "receipt.json", "blocked_seal.json")
    payloads = []; pins = []
    for name in names:
        path = output_dir / name
        observed = core.fd_read_regular(path, root=output_dir, expected_mode="0444")
        payloads.append(_canonical_json(observed.raw, f"stored V6 {name}"))
        pins.append({"path": name, "sha256": observed.sha256, "bytes": observed.bytes, "mode": "0444"})
    draft, receipt, seal = payloads
    require(draft.get("status") == receipt.get("status") == seal.get("status") == "BLOCKED_MISSING_ZERO4_TERMINALS", "stored status drift")
    require(draft == build_blocked_draft(), "stored V6 draft drift")
    require(receipt.get("draft") == {"path": names[0], "sha256": pins[0]["sha256"]}, "stored receipt binding drift")
    require(seal.get("artifacts") == pins[:2] and seal.get("verified_grant_created") is False and seal.get("executable_prelaunch_sealed") is False and seal.get("external_capability_created") is False, "stored V6 seal gained capability")
    return {"status": draft["status"], "contract_sha256": draft["contract_sha256"], "missing_checkpoint_slots": draft["missing_checkpoint_slots"], "draft_sha256": pins[0]["sha256"], "receipt_sha256": pins[1]["sha256"], "blocked_seal_sha256": pins[2]["sha256"], "verified_grant_created": False, "external_capability_created": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--verify-only", action="store_true"); args = parser.parse_args(argv)
    try:
        result = load_stored_blocked_prelaunch(args.output_dir) if args.verify_only else write_blocked_prelaunch(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    except (StaticV6Error, core.V6Error) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
