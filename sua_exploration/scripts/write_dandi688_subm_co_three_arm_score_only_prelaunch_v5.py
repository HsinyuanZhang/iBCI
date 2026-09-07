#!/usr/bin/env python3
"""Write/verify the blocked, append-only external sub-M three-arm V5 package."""
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

from sua_exploration.mc_maze import subm_co_three_arm_score_only_v5 as core  # noqa: E402


DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_score_only_prelaunch_v5"
V4_ROOT = ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_score_only_prelaunch_v4"
V4_PINS = {
    "blocked_prelaunch_draft.json": ("7f8d473a911b6aa4d999692862f0eaa4dde6980a36d472f63152e7665c04c602", 21_186),
    "receipt.json": ("a4e8ce776504bc1b64f63a34adbe17162a5cd5856ac3b5b2a907a3bbffa3f5fd", 1_578),
    "blocked_seal.json": ("1eaa12905d34af5b89228d08895914db9e4fbe8432d711732a69d6fb4c1e243d", 511),
}
V4_CONTRACT_SHA256 = "90cbd7f59d86909a7802f42e05b223d7f097b82dba5c8ff8293b7e84d22c6165"

SOURCE_PATHS = (
    "sua_exploration/docs/DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V5.md",
    "sua_exploration/mc_maze/subm_co_three_arm_score_only_v5.py",
    "sua_exploration/scripts/run_dandi688_subm_co_three_arm_score_only_v5.py",
    "sua_exploration/scripts/write_dandi688_subm_co_three_arm_score_only_prelaunch_v5.py",
    "sua_exploration/tests/test_dandi688_subm_co_three_arm_score_only_v5.py",
)


class StaticThreeArmV5Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticThreeArmV5Error(message)


def _safe_regular(path: Path, *, mode: int | None = None) -> Path:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe artifact: {path}")
    if mode is not None:
        require(stat.S_IMODE(path.lstat().st_mode) == mode, f"artifact mode drift: {path}")
    return path


def _load_canonical_json(path: Path) -> dict[str, Any]:
    raw = _safe_regular(path).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticThreeArmV5Error(f"malformed JSON: {path}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    require(raw == core.canonical_bytes(value), f"noncanonical JSON: {path}")
    return value


def _read_v4_predecessor() -> tuple[dict[str, Any], dict[str, Any]]:
    pins: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    for name, (digest, size) in V4_PINS.items():
        path = _safe_regular(V4_ROOT / name, mode=0o444)
        require(path.stat().st_size == size, f"V4 predecessor size drift: {name}")
        require(core.sha256_file(path) == digest, f"V4 predecessor digest drift: {name}")
        payloads[name] = _load_canonical_json(path)
        pins[name] = {"path": str(path), "sha256": digest, "bytes": size, "mode": "0444"}
    draft = payloads["blocked_prelaunch_draft.json"]
    require(draft.get("status") == "BLOCKED_MISSING_ZERO4_TERMINALS", "V4 status drift")
    old_contract = draft.get("authority", {}).get("contract")
    require(isinstance(old_contract, Mapping), "V4 contract missing")
    old_body = dict(old_contract); old_body.pop("contract_sha256", None)
    require(
        old_contract.get("contract_sha256") == V4_CONTRACT_SHA256
        and core.canonical_sha256(old_body) == V4_CONTRACT_SHA256,
        "V4 contract digest drift",
    )
    return dict(old_contract), pins


def _verify_existing_terminal_evidence(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for slot in slots:
        closure = slot.get("closure")
        if not isinstance(closure, Mapping):
            continue
        path = _safe_regular(ROOT / str(closure["path"]))
        require(path.stat().st_size == closure["bytes"], "terminal evidence size drift")
        require(core.sha256_file(path) == closure["sha256"], "terminal evidence digest drift")
        observed_mode = f"0{stat.S_IMODE(path.lstat().st_mode):03o}"
        require(observed_mode == closure["mode"], "terminal evidence mode drift")
        evidence.append({key: closure[key] for key in core.CLOSURE_KEYS})
    require(len(evidence) == 6, "known terminal evidence count drift")
    return evidence


def _source_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = _safe_regular(ROOT / relative)
        result[relative] = core.sha256_file(path)
    return result


def build_blocked_draft() -> dict[str, Any]:
    old_contract, predecessor = _read_v4_predecessor()
    slots = core.checkpoint_slots_v5()
    core.validate_checkpoint_slots(slots, require_complete=False)
    evidence = _verify_existing_terminal_evidence(slots)
    contract = core.build_contract_v5(
        cohort=old_contract["cohort"],
        query_counts=old_contract["query_window_count_by_asset_id"],
        checkpoint_slots=slots,
    )
    missing = core.missing_checkpoint_slots(slots)
    require(
        [(row["arm"], row["seed"]) for row in missing]
        == [("shared_zero4", 42), ("shared_zero4", 43), ("shared_zero4", 44)],
        "zero4 blocker drift",
    )
    sources = _source_map()
    return {
        "schema_version": 5,
        "kind": "dandi_000688_subm_three_arm_grant_bound_prelaunch_blocked_v5",
        "status": "BLOCKED_MISSING_ZERO4_TERMINALS",
        "append_only": True,
        "v4_predecessor": predecessor,
        "contract": contract,
        "contract_sha256": contract["contract_sha256"],
        "cohort_sha256": contract["cohort_sha256"],
        "query_map_sha256": contract["query_map_sha256"],
        "checkpoint_slots_sha256": contract["checkpoint_slots_sha256"],
        "missing_checkpoint_slots": missing,
        "known_terminal_evidence_revalidated": evidence,
        "external_verified_authorization_grant": None,
        "future_grant_must_bind": [
            "authorization_sha256", "nonce", "absolute output_root",
            "contract_sha256", "cohort_sha256", "query_map_sha256",
            "checkpoint_slots_sha256",
        ],
        "ledger_guarantees": {
            "same_grant_for_publish_resume_aggregate": True,
            "strict_exact_cell_schema": True,
            "finite_r2_and_exact_query_count": True,
            "artifact_path_hash_bytes_0444_no_symlink": True,
            "resume_revalidates_every_cell_and_artifact": True,
            "aggregate_accepts_no_caller_statistics": True,
            "aggregate_reconstructs_two_comparisons_by_two_views_from_270_cells": True,
            "bootstrap": core.BOOTSTRAP_POLICY,
        },
        "source_map": sources,
        "source_snapshot_sha256": core.canonical_sha256(sources),
        "operations_by_this_package": {
            "external_subm_nwb_files_opened": 0,
            "checkpoint_files_opened": 0,
            "torch_imports": 0,
            "model_forward_calls": 0,
            "r2_computations": 0,
            "gpu_used": False,
            "signatures_created": 0,
            "external_capability_created": False,
            "synthetic_fixtures_only": True,
        },
    }


def build_blocked_receipt() -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_blocked_draft()
    receipt = {
        "schema_version": 5,
        "kind": "dandi_000688_subm_three_arm_grant_bound_blocked_receipt_v5",
        "status": "BLOCKED_MISSING_ZERO4_TERMINALS",
        "append_only": True,
        "draft": {"path": "blocked_prelaunch_draft.json", "sha256": core.canonical_sha256(draft)},
        "contract_sha256": draft["contract_sha256"],
        "missing_terminal_slots": draft["missing_checkpoint_slots"],
        "not_an_authorization_grant": True,
        "not_an_executable_prelaunch": True,
        "external_capability_created": False,
        "operations": draft["operations_by_this_package"],
    }
    return draft, receipt


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = core.canonical_bytes(payload)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc:
        raise StaticThreeArmV5Error(f"append-only output exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    os.chmod(path, 0o444, follow_symlinks=False)
    require(stat.S_IMODE(path.lstat().st_mode) == 0o444, "immutable write failed")
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": "0444"}


def write_blocked_prelaunch(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists() and not output_dir.is_symlink(), "V5 blocked output already exists")
    draft, receipt = build_blocked_receipt()
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_pin = _write_immutable(output_dir / "blocked_prelaunch_draft.json", draft)
    require(draft_pin["sha256"] == receipt["draft"]["sha256"], "draft/receipt binding drift")
    receipt_pin = _write_immutable(output_dir / "receipt.json", receipt)
    seal = {
        "schema_version": 5,
        "kind": "dandi_000688_subm_three_arm_grant_bound_blocked_seal_v5",
        "status": "BLOCKED_MISSING_ZERO4_TERMINALS",
        "append_only": True,
        "artifacts": [draft_pin, receipt_pin],
        "executable_prelaunch_sealed": False,
        "external_verified_authorization_grant_created": False,
        "external_capability_created": False,
    }
    seal_pin = _write_immutable(output_dir / "blocked_seal.json", seal)
    return {
        "output_dir": str(output_dir), "status": seal["status"],
        "draft_sha256": draft_pin["sha256"], "receipt_sha256": receipt_pin["sha256"],
        "blocked_seal_sha256": seal_pin["sha256"],
        "executable_prelaunch_sealed": False, "external_capability_created": False,
    }


def load_stored_blocked_prelaunch(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    draft_path, receipt_path, seal_path = (
        output_dir / "blocked_prelaunch_draft.json", output_dir / "receipt.json", output_dir / "blocked_seal.json"
    )
    for path in (draft_path, receipt_path, seal_path):
        _safe_regular(path, mode=0o444)
    draft, receipt, seal = map(_load_canonical_json, (draft_path, receipt_path, seal_path))
    require(
        draft.get("status") == receipt.get("status") == seal.get("status")
        == "BLOCKED_MISSING_ZERO4_TERMINALS",
        "stored V5 status drift",
    )
    require(receipt.get("draft", {}).get("sha256") == core.sha256_file(draft_path), "stored draft binding drift")
    expected_pins = [
        {"path": draft_path.name, "sha256": core.sha256_file(draft_path), "bytes": draft_path.stat().st_size, "mode": "0444"},
        {"path": receipt_path.name, "sha256": core.sha256_file(receipt_path), "bytes": receipt_path.stat().st_size, "mode": "0444"},
    ]
    require(seal.get("artifacts") == expected_pins, "stored seal binding drift")
    require(
        draft.get("external_verified_authorization_grant") is None
        and receipt.get("not_an_authorization_grant") is True
        and seal.get("executable_prelaunch_sealed") is False
        and seal.get("external_capability_created") is False,
        "stored V5 package gained capability",
    )
    # Reconstruct from independently pinned V4 and local sources; a stored
    # draft cannot bless its own modified cohort or contract.
    require(draft == build_blocked_draft(), "stored V5 draft/content drift")
    return {
        "status": draft["status"], "contract_sha256": draft["contract_sha256"],
        "draft_sha256": expected_pins[0]["sha256"], "receipt_sha256": expected_pins[1]["sha256"],
        "blocked_seal_sha256": core.sha256_file(seal_path),
        "missing_checkpoint_slots": draft["missing_checkpoint_slots"],
        "executable_prelaunch_sealed": False, "external_capability_created": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = load_stored_blocked_prelaunch(args.output_dir) if args.verify_only else write_blocked_prelaunch(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (StaticThreeArmV5Error, core.ThreeArmV5Error) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
