#!/usr/bin/env python3
"""Seal a non-executable V8 quarantine receipt for external sub-M scoring.

This writer is deliberately unable to call a production transition.  It only
reads V8's exact blocked status and the V7 blocked receipt text, then writes
three immutable local provenance files.  It does not read external data or a
checkpoint, import Torch, create a policy/signature/grant, compute R², or use
a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sua_exploration.mc_maze import subm_co_three_arm_score_only_v8 as core  # noqa: E402


DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_score_only_prelaunch_v8"
V7_BLOCKED_RECEIPT = ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_score_only_prelaunch_v7/receipt.json"
SOURCE_PATHS = (
    "sua_exploration/configs/dandi_000688_subm_v8_pinned_production_anchor.json",
    "sua_exploration/docs/DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V8.md",
    "sua_exploration/mc_maze/subm_co_three_arm_score_only_v8.py",
    "sua_exploration/scripts/run_dandi688_subm_co_three_arm_score_only_v8.py",
    "sua_exploration/scripts/run_dandi688_subm_co_three_arm_score_only_v8_production.py",
    "sua_exploration/scripts/write_dandi688_subm_co_three_arm_score_only_prelaunch_v8.py",
    "sua_exploration/tests/test_dandi688_subm_co_three_arm_score_only_v8.py",
)


class StaticV8Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticV8Error(message)


def _canonical_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticV8Error(f"malformed {label}") from exc
    require(isinstance(payload, dict) and raw == core.canonical_bytes(payload), f"noncanonical {label}")
    return payload


def _source_map() -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"missing V8 source: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _v7_reference() -> dict[str, Any]:
    require(V7_BLOCKED_RECEIPT.is_file() and not V7_BLOCKED_RECEIPT.is_symlink(), "missing V7 blocked receipt")
    raw = V7_BLOCKED_RECEIPT.read_bytes(); payload = _canonical_json(raw, "V7 receipt")
    require(payload.get("status") == "BLOCKED_MISSING_ZERO4_TERMINALS" and payload.get("not_an_executable_prelaunch") is True, "V7 receipt no longer blocked")
    return {
        "path": str(V7_BLOCKED_RECEIPT), "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "mode": f"0{stat.S_IMODE(V7_BLOCKED_RECEIPT.stat().st_mode):03o}",
    }


def _anchor_pin() -> dict[str, Any]:
    # Production status captures and verifies the actual source-pinned anchor;
    # this writer receives only the resulting blocked facts, never root data.
    status = core.production_status()
    require(status["status"] == core.BLOCKED_STATUS and status["formal_root_chain_active"] is False, "V8 production status unexpectedly active")
    anchor = ROOT / "sua_exploration/configs/dandi_000688_subm_v8_pinned_production_anchor.json"
    raw = anchor.read_bytes(); payload = _canonical_json(raw, "V8 anchor")
    require(payload.get("status") == core.ANCHOR_STATUS, "V8 anchor status drift")
    return {"path": str(anchor), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": f"0{stat.S_IMODE(anchor.stat().st_mode):03o}"}


def build_blocked_draft() -> dict[str, Any]:
    source_map = _source_map()
    return {
        "schema_version": 8,
        "kind": "dandi_000688_subm_three_arm_score_only_quarantine_v8",
        "status": core.BLOCKED_STATUS,
        "append_only": True,
        "v7_blocked_predecessor": _v7_reference(),
        "source_map": source_map,
        "source_snapshot_sha256": core.canonical_sha256(source_map),
        "source_pinned_blocked_anchor": _anchor_pin(),
        "formal_matrix": core.formal_matrix_spec(),
        "p0_disposition": {
            "v7_independent_audit_status": "P0_ROOT_PROVENANCE_FORGERY_NOT_PASSED",
            "v8_does_not_claim_v7_pass": True,
            "production_transition_accepts_caller_roots_paths_keys_payloads": False,
            "active_formal_root_chain_present": False,
            "complete_policy_created": False,
            "checkpoint_closure_or_real_checkpoint_opened": False,
            "verified_grant_created": False,
        },
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
    }


def build_blocked_receipt() -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_blocked_draft()
    receipt = {
        "schema_version": 8,
        "kind": "dandi_000688_subm_three_arm_quarantine_receipt_v8",
        "status": draft["status"], "append_only": True,
        "draft": {"path": "blocked_prelaunch_draft.json", "sha256": core.canonical_sha256(draft)},
        "v7_p0_not_passed": True,
        "not_an_independent_pass": True,
        "not_an_authorization_or_verified_grant": True,
        "not_an_executable_prelaunch": True,
        "operations": draft["operations_by_this_package"],
    }
    return draft, receipt


def _write_immutable(root: Path, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw = core.canonical_bytes(payload); path = root / name
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc:
        raise StaticV8Error(f"duplicate V8 immutable file: {name}") from exc
    try:
        total = 0
        while total < len(raw):
            total += os.write(descriptor, raw[total:])
        os.fsync(descriptor); os.fchmod(descriptor, 0o444)
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o444 and metadata.st_size == len(raw), "V8 immutable writer failure")
    finally:
        os.close(descriptor)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": "0444"}


def write_blocked_prelaunch(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = Path(os.path.abspath(output_dir))
    require(not output_dir.exists() and not output_dir.is_symlink(), "V8 output already exists")
    draft, receipt = build_blocked_receipt()
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    draft_pin = _write_immutable(output_dir, "blocked_prelaunch_draft.json", draft)
    require(draft_pin["sha256"] == receipt["draft"]["sha256"], "V8 draft receipt binding drift")
    receipt_pin = _write_immutable(output_dir, "receipt.json", receipt)
    seal = {
        "schema_version": 8, "kind": "dandi_000688_subm_three_arm_quarantine_seal_v8",
        "status": core.BLOCKED_STATUS, "append_only": True,
        "artifacts": [draft_pin, receipt_pin],
        "v7_p0_not_passed": True, "not_an_independent_pass": True,
        "complete_policy_created": False, "verified_grant_created": False,
        "executable_prelaunch_sealed": False,
    }
    seal_pin = _write_immutable(output_dir, "blocked_seal.json", seal)
    return {
        "output_dir": str(output_dir), "status": core.BLOCKED_STATUS,
        "draft_sha256": draft_pin["sha256"], "receipt_sha256": receipt_pin["sha256"],
        "blocked_seal_sha256": seal_pin["sha256"],
        "v7_p0_not_passed": True, "not_an_independent_pass": True,
        "complete_policy_created": False, "verified_grant_created": False,
        "executable_prelaunch_sealed": False,
    }


def load_stored_blocked_prelaunch(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = Path(os.path.abspath(output_dir))
    names = ("blocked_prelaunch_draft.json", "receipt.json", "blocked_seal.json")
    values: list[dict[str, Any]] = []; pins: list[dict[str, Any]] = []
    for name in names:
        path = output_dir / name
        require(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444, f"stored V8 file unsafe: {name}")
        raw = path.read_bytes(); values.append(_canonical_json(raw, name)); pins.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": "0444"})
    draft, receipt, seal = values
    require(draft == build_blocked_draft(), "stored V8 draft/source/anchor drift")
    require(receipt.get("status") == seal.get("status") == core.BLOCKED_STATUS, "stored V8 blocked status drift")
    require(receipt.get("draft") == {"path": names[0], "sha256": pins[0]["sha256"]}, "stored V8 receipt binding drift")
    require(seal.get("artifacts") == pins[:2] and seal.get("v7_p0_not_passed") is True and seal.get("not_an_independent_pass") is True and seal.get("verified_grant_created") is False, "stored V8 seal capability/pass drift")
    return {
        "status": core.BLOCKED_STATUS, "draft_sha256": pins[0]["sha256"],
        "receipt_sha256": pins[1]["sha256"], "blocked_seal_sha256": pins[2]["sha256"],
        "v7_p0_not_passed": True, "not_an_independent_pass": True,
        "complete_policy_created": False, "verified_grant_created": False,
        "executable_prelaunch_sealed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        value = load_stored_blocked_prelaunch(args.output_dir) if args.verify_only else write_blocked_prelaunch(args.output_dir)
        print(json.dumps(value, indent=2, sort_keys=True)); return 0
    except (StaticV8Error, core.V8Error) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
