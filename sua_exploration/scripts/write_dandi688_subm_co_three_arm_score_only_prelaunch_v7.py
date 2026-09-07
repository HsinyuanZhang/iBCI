#!/usr/bin/env python3
"""Write/verify the append-only, intentionally blocked V7 sub-M package.

This script is not a formal launcher.  It never calls the V7 trusted-root
installer, policy verifier, closure verifier, run-authorization verifier, or
ledger.  Consequently it cannot open an external NWB or a checkpoint, create
a complete policy/grant, perform a model forward/R² computation, or use a GPU.
"""
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
from sua_exploration.mc_maze import subm_co_three_arm_score_only_v7 as core  # noqa: E402


DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_score_only_prelaunch_v7"
V6_BLOCKED_DIR = ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_score_only_prelaunch_v6"
SOURCE_PATHS = (
    "sua_exploration/configs/dandi_000688_subm_v7_pinned_trust_anchor.json",
    "sua_exploration/docs/DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V7.md",
    "sua_exploration/mc_maze/subm_co_three_arm_score_only_v7.py",
    "sua_exploration/scripts/run_dandi688_subm_co_three_arm_score_only_v7.py",
    "sua_exploration/scripts/write_dandi688_subm_co_three_arm_score_only_prelaunch_v7.py",
    "sua_exploration/tests/test_dandi688_subm_co_three_arm_score_only_v7.py",
)


class StaticV7Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticV7Error(message)


def _source_map() -> dict[str, str]:
    source_map = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"missing V7 source: {relative}")
        source_map[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return source_map


def _pinned_blocked_anchor() -> dict[str, Any]:
    path = core.PINNED_TRUST_ANCHOR_PATH
    observed = core.fd_read_regular_at(
        path.parent, path.name, expected_mode=core.PINNED_TRUST_ANCHOR_MODE,
        expected_sha256=core.PINNED_TRUST_ANCHOR_SHA256,
        expected_bytes=core.PINNED_TRUST_ANCHOR_BYTES, max_bytes=64 * 1024,
    )
    try:
        payload = json.loads(observed.raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticV7Error("pinned anchor malformed") from exc
    require(isinstance(payload, dict) and observed.raw == core.canonical_bytes(payload), "pinned anchor noncanonical")
    return {"pin": {"path": str(path), "sha256": observed.sha256, "bytes": observed.bytes, "mode": observed.mode}, "payload": payload}


def _legacy_v6_reference() -> dict[str, Any]:
    """Pin only V6's blocked text receipt; never inspect a model/checkpoint."""
    receipt = V6_BLOCKED_DIR / "receipt.json"
    require(receipt.is_file() and not receipt.is_symlink(), "missing V6 blocked receipt")
    raw = receipt.read_bytes()
    try:
        payload = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticV7Error("V6 blocked receipt malformed") from exc
    require(isinstance(payload, dict) and raw == core.canonical_bytes(payload) and payload.get("status") == "BLOCKED_MISSING_ZERO4_TERMINALS", "V6 blocked predecessor drift")
    return {"path": str(receipt), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": f"0{stat.S_IMODE(receipt.stat().st_mode):03o}"}


def build_blocked_draft() -> dict[str, Any]:
    slots = core.blocked_checkpoint_slots()
    missing = [row for row in slots if row["closure"] is None]
    require([(row["arm"], row["seed"]) for row in missing] == [("shared_zero4", 42), ("shared_zero4", 43), ("shared_zero4", 44)], "zero4 terminal blocker drift")
    anchor = _pinned_blocked_anchor()
    require(anchor["payload"].get("status") == "BLOCKED_NO_ACTIVE_FORMAL_TRUST_ROOTS_V7", "V7 anchor unexpectedly active")
    sources = _source_map()
    return {
        "schema_version": 7,
        "kind": "dandi_000688_subm_three_arm_score_only_blocked_control_plane_v7",
        "status": "BLOCKED_MISSING_ZERO4_TERMINALS",
        "append_only": True,
        "v6_blocked_predecessor": _legacy_v6_reference(),
        "source_map": sources,
        "source_snapshot_sha256": core.canonical_sha256(sources),
        "pinned_formal_trust_anchor": anchor,
        "historical_checkpoint_slots": slots,
        "missing_checkpoint_slots": missing,
        "formal_v7_preconditions_not_satisfied": {
            "zero4_terminal_checkpoint_files_missing": [["shared_zero4", 42], ["shared_zero4", 43], ["shared_zero4", 44]],
            "independently_ed25519_signed_v7_terminal_closures_required_for_all_nine_slots": 9,
            "active_source_pinned_policy_checkpoint_run_auth_root_anchor_present": False,
            "complete_policy_signed_before_contract_present": False,
            "external_run_authorization_present": False,
        },
        "contract": None,
        "verified_policy": None,
        "verified_grant": None,
        "operations_by_this_package": {
            "external_subm_nwb_files_opened": 0,
            "real_checkpoint_files_opened": 0,
            "torch_imports": 0,
            "model_forward_calls": 0,
            "external_r2_computations": 0,
            "gpu_used": False,
            "complete_policies_created": 0,
            "signatures_created": 0,
            "verified_grants_created": 0,
            "synthetic_test_fixtures_only": True,
        },
        "v7_repairs": {
            "production_trusted_roots_are_source_pinned_not_runtime_payload": True,
            "runtime_policy_cannot_carry_public_key_or_root_path": True,
            "complete_policy_signature_precedes_contract_construction": True,
            "all_nine_independent_ed25519_closures_and_live_checkpoint_pins_required": True,
            "canonical_nonce_claim_and_dirfd_openat_ledger": True,
            "npy_header_first_archive_limits_and_frozen_torchmetrics_r2_semantics": True,
        },
    }


def build_blocked_receipt() -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_blocked_draft()
    receipt = {
        "schema_version": 7,
        "kind": "dandi_000688_subm_three_arm_blocked_receipt_v7",
        "status": draft["status"], "append_only": True,
        "draft": {"path": "blocked_prelaunch_draft.json", "sha256": core.canonical_sha256(draft)},
        "missing_checkpoint_slots": draft["missing_checkpoint_slots"],
        "formal_trust_anchor_active": False,
        "not_an_authorization_or_verified_grant": True,
        "not_an_executable_prelaunch": True,
        "operations": draft["operations_by_this_package"],
    }
    return draft, receipt


def write_blocked_prelaunch(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = Path(os.path.abspath(output_dir))
    require(not output_dir.exists() and not output_dir.is_symlink(), "V7 blocked output already exists")
    draft, receipt = build_blocked_receipt()
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    require(stat.S_ISDIR(output_dir.lstat().st_mode) and not output_dir.is_symlink(), "unsafe V7 output root")
    draft_pin = core._write_exclusive_at(output_dir, "blocked_prelaunch_draft.json", core.canonical_bytes(draft))
    require(draft_pin["sha256"] == receipt["draft"]["sha256"], "draft/receipt binding drift")
    receipt_pin = core._write_exclusive_at(output_dir, "receipt.json", core.canonical_bytes(receipt))
    seal = {
        "schema_version": 7, "kind": "dandi_000688_subm_three_arm_blocked_seal_v7",
        "status": draft["status"], "append_only": True,
        "artifacts": [draft_pin, receipt_pin],
        "formal_trust_anchor_active": False,
        "complete_policy_created": False, "verified_grant_created": False,
        "executable_prelaunch_sealed": False,
    }
    seal_pin = core._write_exclusive_at(output_dir, "blocked_seal.json", core.canonical_bytes(seal))
    return {
        "output_dir": str(output_dir), "status": draft["status"],
        "draft_sha256": draft_pin["sha256"], "receipt_sha256": receipt_pin["sha256"],
        "blocked_seal_sha256": seal_pin["sha256"],
        "complete_policy_created": False, "verified_grant_created": False,
        "executable_prelaunch_sealed": False,
    }


def load_stored_blocked_prelaunch(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = Path(os.path.abspath(output_dir))
    names = ("blocked_prelaunch_draft.json", "receipt.json", "blocked_seal.json")
    values: list[dict[str, Any]] = []; pins: list[dict[str, Any]] = []
    for name in names:
        value, observed = core._read_canonical_json_at(output_dir, name, expected_mode="0444", max_bytes=16 * 1024 * 1024)
        values.append(value); pins.append({"path": name, "sha256": observed.sha256, "bytes": observed.bytes, "mode": observed.mode})
    draft, receipt, seal = values
    require(draft == build_blocked_draft(), "stored V7 blocked draft/source anchor drift")
    require(receipt.get("status") == seal.get("status") == draft.get("status") == "BLOCKED_MISSING_ZERO4_TERMINALS", "stored V7 blocked status drift")
    require(receipt.get("draft") == {"path": names[0], "sha256": pins[0]["sha256"]}, "stored V7 receipt binding drift")
    require(seal.get("artifacts") == pins[:2] and seal.get("complete_policy_created") is False and seal.get("verified_grant_created") is False and seal.get("executable_prelaunch_sealed") is False, "stored V7 seal gained capability")
    return {
        "status": draft["status"], "draft_sha256": pins[0]["sha256"],
        "receipt_sha256": pins[1]["sha256"], "blocked_seal_sha256": pins[2]["sha256"],
        "complete_policy_created": False, "verified_grant_created": False,
        "executable_prelaunch_sealed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = load_stored_blocked_prelaunch(args.output_dir) if args.verify_only else write_blocked_prelaunch(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    except (StaticV7Error, core.V7Error) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
