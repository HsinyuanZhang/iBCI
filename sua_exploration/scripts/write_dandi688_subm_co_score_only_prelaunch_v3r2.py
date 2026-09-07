#!/usr/bin/env python3
"""Build an append-only, non-authorizing V3R2 external sub-M score policy."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.scripts import write_dandi688_subm_co_score_only_prelaunch_v3 as v3  # noqa: E402


DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v3r2"
OUTPUT_PARENT_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_score_only_v3r2_runs"
CLAIM_ROOT_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_score_only_v3r2_nonce_claims"
DOC = "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORE_ONLY_PROTOCOL_V3R2.md"
V3R2_SOURCE_PATHS = (
    DOC,
    "sua_exploration/mc_maze/subm_co_score_only_v3r2.py",
    "sua_exploration/scripts/run_dandi688_subm_co_score_only_v3r2.py",
    "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch_v3r2.py",
    "sua_exploration/scripts/prepare_dandi688_subm_co_score_only_authorization_v3r2.py",
    "sua_exploration/tests/test_dandi688_subm_co_score_only_v3r2.py",
)

SUPERSEDED_BANNER = (
    "> **SUPERSEDED — historical evidence only, not current authority.**\n"
    "> Current successor: [`DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md`]"
    "(DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md).\n"
    "> Preserved as append-only audit evidence; content unchanged.\n\n"
).encode("utf-8")

BANNER_ONLY_TRANSITIONS = {
    "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V3.md": {
        "historical_sha256": "f8c3da98d38c3d061f2d89d5514cf1d5453424b0a458d5440a4a64dd3c7bf802",
        "current_sha256": "d761115a7c9339e7c24522dc98cfdc5e0f98a4b74c4f33a0269365dc13f3b20d",
    },
    "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V4.md": {
        "historical_sha256": "2b175d7d7ab9886621f4eaf4b13d63cef0ffe9de72ddb91b93fd102be8def23b",
        "current_sha256": "87112abf97b3a449f5a85caedc400b48f169178e8dcfc1f4170607252d3b5af5",
    },
}


class StaticScoreV3R2Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticScoreV3R2Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticScoreV3R2Error(f"cannot parse {label}") from exc
    require(isinstance(value, dict), f"{label} root is not an object")
    return value


def _current_source_map(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        source = (root / relative).resolve()
        require(source.is_file() and not source.is_symlink(), f"missing V3R2 source: {relative}")
        result[relative] = sha256_file(source)
    return result


def _verify_banner_transition(root: Path, relative: str, transition: Mapping[str, str]) -> dict[str, Any]:
    source = (root / relative).resolve()
    require(source.is_file() and not source.is_symlink(), f"banner-transition source missing: {relative}")
    raw = source.read_bytes()
    require(len(SUPERSEDED_BANNER) == 276, "known superseded banner byte count drift")
    require(raw.startswith(SUPERSEDED_BANNER), f"known superseded banner absent: {relative}")
    current = hashlib.sha256(raw).hexdigest()
    historical = hashlib.sha256(raw[len(SUPERSEDED_BANNER):]).hexdigest()
    require(current == transition["current_sha256"], f"current bannered source drift: {relative}")
    require(historical == transition["historical_sha256"], f"historical body differs beyond banner: {relative}")
    return {
        "relative_path": relative,
        "canonical_path": str(source),
        "transition": "exact_276_byte_superseded_banner_prefix_only",
        "banner_bytes": len(SUPERSEDED_BANNER),
        "historical_body_sha256": historical,
        "current_full_file_sha256": current,
        "scientific_or_adapter_body_changed": False,
    }


def _v5r2_authority(root: Path) -> dict[str, Any]:
    run_root = (root / v3.V5R2_RUN_ROOT).resolve()
    payloads: dict[str, dict[str, Any]] = {}
    for name, expected in v3.V5R2_SUCCESS_ARTIFACTS.items():
        path = run_root / name
        require(path.is_file() and not path.is_symlink(), f"missing V5R2 parity artifact: {name}")
        require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"mutable V5R2 parity artifact: {name}")
        require(
            path.stat().st_size == expected["bytes"] and sha256_file(path) == expected["sha256"],
            f"V5R2 parity artifact hash/size drift: {name}",
        )
        payloads[name] = _read_json(path, f"V5R2 parity {name}")
    try:
        v3._verify_v5r2_success_semantics(
            payloads["parity_execution_receipt.json"],
            payloads["input_trace.json"],
            payloads["environment.json"],
            payloads["seal.json"],
        )
    except v3.StaticScoreV3Error as exc:
        raise StaticScoreV3R2Error(str(exc)) from exc

    historical = payloads["environment.json"].get("source_pins")
    require(isinstance(historical, Mapping) and historical, "V5R2 historical source snapshot missing")
    current: dict[str, str] = {}
    transitions: list[dict[str, Any]] = []
    transition_by_absolute = {
        str((root / relative).resolve()): (relative, record)
        for relative, record in BANNER_ONLY_TRANSITIONS.items()
    }
    for raw_path, historical_digest in historical.items():
        require(isinstance(raw_path, str) and isinstance(historical_digest, str), "V5R2 source pin schema drift")
        source = Path(raw_path)
        require(source.is_file() and not source.is_symlink(), f"V5R2 runtime source missing: {raw_path}")
        observed = sha256_file(source)
        if observed == historical_digest:
            current[raw_path] = observed
            continue
        transition = transition_by_absolute.get(str(source.resolve()))
        require(transition is not None, f"V5R2 runtime source drift is not an approved banner transition: {raw_path}")
        relative, record = transition
        require(historical_digest == record["historical_sha256"], f"V5R2 historical pin mismatch: {relative}")
        evidence = _verify_banner_transition(root, relative, record)
        transitions.append(evidence)
        current[raw_path] = evidence["current_full_file_sha256"]
    require(len(transitions) == 2, "V3R2 requires exactly two known banner-only transitions")
    return {
        "run_root": str(run_root),
        "artifacts": {
            name: {"path": str(run_root / name), **expected, "mode": "0444"}
            for name, expected in v3.V5R2_SUCCESS_ARTIFACTS.items()
        },
        "historical_runtime_source_snapshot": dict(historical),
        "current_runtime_source_snapshot": current,
        "banner_only_transitions": sorted(transitions, key=lambda row: row["relative_path"]),
        "status": payloads["parity_execution_receipt.json"]["status"],
        "r2": v3.EXPECTED_R2,
        "prediction_target_exact": True,
        "input_exact": True,
        "chronology_unchanged": True,
        "external_subm_scoring_performed": False,
    }


def _merge_source_snapshots(*snapshots: Mapping[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for snapshot in snapshots:
        for path, digest in snapshot.items():
            absolute = str(Path(path).resolve()) if Path(path).is_absolute() else str((ROOT / path).resolve())
            if absolute in merged:
                require(merged[absolute] == digest, f"conflicting current source pin: {absolute}")
            merged[absolute] = digest
    return merged


def static_authority(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    try:
        v2 = v3._v2_authority(root)
        v3_sources = v3._current_source_map(root)
    except v3.StaticScoreV3Error as exc:
        raise StaticScoreV3R2Error(str(exc)) from exc
    v5r2 = _v5r2_authority(root)
    v3r2_sources = _current_source_map(root, V3R2_SOURCE_PATHS)
    key = (root / v3.PUBLIC_KEY).resolve()
    require(key.is_file() and not key.is_symlink(), "dedicated public key missing")
    require(key.stat().st_size == v3.PUBLIC_KEY_BYTES and sha256_file(key) == v3.PUBLIC_KEY_SHA256, "dedicated public-key pin drift")
    return {
        "score_only_v2": v2,
        "v5r2_parity_execution": v5r2,
        "v3_historical_sources": v3_sources,
        "v3r2_sources": v3r2_sources,
        "source_snapshot": _merge_source_snapshots(
            v5r2["current_runtime_source_snapshot"], v2["sources"], v3_sources, v3r2_sources
        ),
        "public_key": {"path": str(key), "sha256": v3.PUBLIC_KEY_SHA256, "bytes": v3.PUBLIC_KEY_BYTES},
        "runtime_identity": v3._runtime_identity(),
        "cpu_policy": dict(v3.CPU_POLICY),
        "output_parent": str((root / OUTPUT_PARENT_RELATIVE).resolve()),
        "claim_root": str((root / CLAIM_ROOT_RELATIVE).resolve()),
        "deployment_budget": {
            "activity_identity_trials": 30,
            "t4_fit_pool_trials": 50,
            "query_start": "strictly_after_rewarded_trial_50",
            "chronological": True,
        },
        "claim_boundary": {
            "frozen_arms_only": ["shared_t4", "shared_ts4"],
            "supports": "correct-vs-shuffled attachment/content contrast only",
            "absolute_t4_over_spint_claim": "UNSUPPORTED_NO_SHARED_B0_CONTROL",
            "same_window_shared_b0_added": False,
            "same_dandiset_cross_animal_only": True,
            "native_threshold_crossing_mua_claim": False,
        },
        "execution_release_condition": "ONLY_AFTER_USER_CONFIRMS_NATIVE_M2_COMPLETE_AND_FRESH_ROOT_REVIEW",
    }


def _policy_base(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "public_key": dict(authority["public_key"]),
        "source_pins": dict(authority["source_snapshot"]),
        "runtime_identity": dict(authority["runtime_identity"]),
        "cpu_policy": dict(authority["cpu_policy"]),
        "output_parent": authority["output_parent"],
        "claim_root": authority["claim_root"],
    }


def build_draft(root: Path = ROOT) -> dict[str, Any]:
    authority = static_authority(root)
    return {
        "schema_version": "3r2",
        "kind": "dandi_000688_subm_co_score_only_cpu_prelaunch_draft_v3r2",
        "status": "BLOCKED_PENDING_NATIVE_M2_COMPLETION_AND_FRESH_SIGNATURE",
        "append_only": True,
        "authority": authority,
        "execution_policy_base": _policy_base(authority),
        "authorization_contract": {
            "authorization_schema": "dandi_000688_subm_co_score_only_execution_authorization_v3",
            "envelope_schema": "dandi_000688_subm_co_score_only_execution_authorization_envelope_v3",
            "permitted_action": "score_frozen_subm_matrix_via_v5_bridge_v3",
            "maximum_validity_seconds": 900,
            "fresh_nonce_required": True,
            "detached_ed25519_signature_required": True,
            "CPU_forward_R2_only": True,
            "CUDA": "FORBIDDEN",
            "normalizer_fitting": "FORBIDDEN",
            "optimizer_or_backward": "FORBIDDEN",
            "target_updates": "FORBIDDEN",
            "retry_or_overwrite": "FORBIDDEN",
        },
        "operations_by_this_prelaunch": {
            "external_subm_nwb_files_opened": 0,
            "checkpoint_files_opened": 0,
            "normalizer_files_opened": 0,
            "model_forward_calls": 0,
            "r2_computations": 0,
            "nonce_claims": 0,
            "authorization_created": False,
            "external_subm_scoring_performed": False,
        },
    }


def build_receipt(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_draft(root)
    receipt = {
        "schema_version": "3r2",
        "receipt_kind": "dandi_000688_subm_co_score_only_cpu_prelaunch_receipt_v3r2",
        "status": "STATIC_V3R2_READY_BUT_NOT_AUTHORIZED",
        "append_only": True,
        "draft": {
            "filename": "prelaunch_authorization_draft.json",
            "sha256": hashlib.sha256(canonical_bytes(draft)).hexdigest(),
        },
        "operations": dict(draft["operations_by_this_prelaunch"]),
        "blocked_until": [
            "user confirms Native-M2 experiment is complete",
            "root performs a fresh byte-level V3R2 review",
            "a short-lived canonical V3 detached capability is signed outside the workspace",
        ],
    }
    return draft, receipt


def _write_immutable(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, "immutable mode failed")
    return hashlib.sha256(raw).hexdigest()


def write_prelaunch(output_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), "V3R2 prelaunch root already exists")
    draft, receipt = build_receipt(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path = output_dir / "prelaunch_authorization_draft.json"
    receipt_path = output_dir / "receipt.json"
    seal_path = output_dir / "seal.json"
    draft_sha = _write_immutable(draft_path, draft)
    require(draft_sha == receipt["draft"]["sha256"], "V3R2 draft/receipt binding drift")
    receipt_sha = _write_immutable(receipt_path, receipt)
    seal = {
        "schema_version": "3r2",
        "kind": "dandi_000688_subm_co_score_only_cpu_prelaunch_seal_v3r2",
        "status": receipt["status"],
        "append_only": True,
        "artifacts": [
            {"path": draft_path.name, "sha256": draft_sha, "bytes": draft_path.stat().st_size, "mode": "0444"},
            {"path": receipt_path.name, "sha256": receipt_sha, "bytes": receipt_path.stat().st_size, "mode": "0444"},
        ],
        "future_execution_not_authorized_by_this_seal": True,
    }
    seal_sha = _write_immutable(seal_path, seal)
    return {
        "output_dir": str(output_dir),
        "draft": {"path": str(draft_path), "sha256": draft_sha},
        "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        "seal": {"path": str(seal_path), "sha256": seal_sha},
        "status": receipt["status"],
    }


def _authorization_bindings(
    authority: Mapping[str, Any], paths: tuple[Path, Path, Path], hashes: tuple[str, str, str]
) -> dict[str, Any]:
    draft_path, receipt_path, seal_path = paths
    draft_sha, receipt_sha, seal_sha = hashes
    return {
        "prelaunch_bundle_v3r2": {
            "draft": {"path": str(draft_path), "sha256": draft_sha, "bytes": draft_path.stat().st_size, "mode": "0444"},
            "receipt": {"path": str(receipt_path), "sha256": receipt_sha, "bytes": receipt_path.stat().st_size, "mode": "0444"},
            "seal": {"path": str(seal_path), "sha256": seal_sha, "bytes": seal_path.stat().st_size, "mode": "0444"},
        },
        "score_only_v2": dict(authority["score_only_v2"]),
        "v5r2_parity_execution": dict(authority["v5r2_parity_execution"]),
        "v3_historical_sources": dict(authority["v3_historical_sources"]),
        "v3r2_sources": dict(authority["v3r2_sources"]),
        "source_snapshot": dict(authority["source_snapshot"]),
        "public_key": dict(authority["public_key"]),
        "runtime_identity": dict(authority["runtime_identity"]),
        "cpu_policy": dict(authority["cpu_policy"]),
        "deployment_budget": dict(authority["deployment_budget"]),
        "claim_boundary": dict(authority["claim_boundary"]),
        "execution_release_condition": authority["execution_release_condition"],
    }


def load_stored_prelaunch(prelaunch_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    prelaunch_dir = prelaunch_dir.resolve()
    paths = tuple(prelaunch_dir / name for name in ("prelaunch_authorization_draft.json", "receipt.json", "seal.json"))
    for path in paths:
        require(
            path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444,
            f"stored V3R2 artifact unsafe: {path.name}",
        )
    draft = _read_json(paths[0], "V3R2 draft")
    receipt = _read_json(paths[1], "V3R2 receipt")
    seal = _read_json(paths[2], "V3R2 seal")
    hashes = tuple(sha256_file(path) for path in paths)
    require(draft.get("status") == "BLOCKED_PENDING_NATIVE_M2_COMPLETION_AND_FRESH_SIGNATURE", "V3R2 draft status drift")
    require(receipt.get("status") == seal.get("status") == "STATIC_V3R2_READY_BUT_NOT_AUTHORIZED", "V3R2 receipt/seal status drift")
    require(receipt.get("draft") == {"filename": paths[0].name, "sha256": hashes[0]}, "V3R2 draft receipt binding drift")
    require(
        seal.get("artifacts") == [
            {"path": paths[0].name, "sha256": hashes[0], "bytes": paths[0].stat().st_size, "mode": "0444"},
            {"path": paths[1].name, "sha256": hashes[1], "bytes": paths[1].stat().st_size, "mode": "0444"},
        ],
        "V3R2 seal binding drift",
    )
    authority = static_authority(root)
    require(draft.get("authority") == authority, "stored V3R2 authority drift")
    require(draft.get("execution_policy_base") == _policy_base(authority), "stored V3R2 policy-base drift")
    bindings = _authorization_bindings(authority, paths, hashes)
    policy_without_sha = {**dict(draft["execution_policy_base"]), "authorization_bindings": bindings}
    policy_sha = hashlib.sha256(canonical_bytes(policy_without_sha)).hexdigest()
    policy = {**policy_without_sha, "execution_policy_sha256": policy_sha}
    return {
        "draft_sha256": hashes[0],
        "receipt_sha256": hashes[1],
        "seal_sha256": hashes[2],
        "status": receipt["status"],
        "execution_policy": policy,
        "execution_policy_sha256": policy_sha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = static_authority(args.repo_root) if args.verify_only else write_prelaunch(args.output_dir, args.repo_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (StaticScoreV3R2Error, v3.StaticScoreV3Error) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
