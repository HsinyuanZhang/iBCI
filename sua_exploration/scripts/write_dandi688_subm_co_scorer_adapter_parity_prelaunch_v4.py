#!/usr/bin/env python3
"""Write the source-only, non-authorizing v4 CPU parity execution prelaunch.

This module reads only source/JSON/public-key/interpreter metadata. It never
imports the v3 helper, Torch, NumPy, PyNWB, a model/data owner, or any
checkpoint, normalizer NPZ, or NWB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_prelaunch_v4"
DOC = "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V4.md"
V3_ROOT = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_prelaunch_v3"
PUBLIC_KEY = "sua_exploration/configs/dandi688_subc_parity_v4_root_ed25519_public.pem"
PUBLIC_KEY_SHA256 = "a541b5aabc7922251e797da13727b97ceb4ed48de7e95d5b1c4031ba96c602a9"
PUBLIC_KEY_BYTES = 113
OUTPUT_PARENT_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_v4_runs"
CLAIM_ROOT_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_v4_nonce_claims"

V3_SOURCES = {
    "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V3.md": "f8c3da98d38c3d061f2d89d5514cf1d5453424b0a458d5440a4a64dd3c7bf802",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v3.py": "c7951beb13618b849c43a2d296a3b1550f0a644efb68e872567fd14368759a9b",
    "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v3.py": "e70da4456c3c78ebbf0c02c34533ea4b5b2ed0313a22aee5c9451948d6c9986f",
    "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v3.py": "fceeeaec8e72346f1dd41e849d74a00831f91df5c02d545b9d3fa1608a6a6f8d",
    "sua_exploration/tests/test_dandi688_subm_co_scorer_adapter_parity_v3.py": "2f19994b1104307b6cf3b7fe491f28328ff22d3e75f594c0332f7b2930fe87c5",
}
V3_ARTIFACTS = {
    "parity_protocol_draft.json": "96200f471c7a38277508071ab9c483e623defbc9038946468408cd495a4e6157",
    "receipt.json": "c52796b13f9e76f3dd98d62fa882158c6ae3ad4237f8896753741c259b383d8b",
    "seal.json": "e0088a63b97717b9d984409485da20fdaca8e378a01750733f4bb60b02191cb1",
}
V4_SOURCES = (
    DOC,
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_execution_v4.py",
    "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v4.py",
    "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v4.py",
    "sua_exploration/tests/test_dandi688_subm_co_scorer_adapter_parity_v4.py",
)
FIXTURE_PINS = {
    "checkpoint": {
        "path": "sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_t4_s44/epoch_ckpts/epoch_011.ckpt",
        "sha256": "a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6",
        "bytes": 64769167,
    },
    "teacher": {
        "path": "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt",
        "sha256": "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d",
        "bytes": 55195903,
    },
    "consumed_subc_nwb": {
        "path": "sua_exploration/data/dandi_000688/sub-C/sub-C_ses-CO-20151103_behavior+ecephys.nwb",
        "session": "sub-C_ses-CO-20151103",
        "sha256": "7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7",
        "bytes": 62145872,
    },
    "behavior_normalizer": {
        "path": "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/sua/behavior_stats/be50f588491c004f721e.npz",
        "sha256": "821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd",
        "bytes": 397,
    },
    "t4_normalizer": {
        "path": "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/sua/side_feature_stats/dd3da1f59700c1b96ab8.npz",
        "sha256": "32d32a7fe1b80a139571aae0ce3c3a1d802aec21c99b23a4cff72b8a60261701",
        "bytes": 614,
        "semantic_sha256": "ac5156097864110685e0b2fbfe314edcb747e69dc821c10451984a089be8a7a7",
    },
    "protocol": {
        "view": "sua",
        "support_trials": 50,
        "identity": "first_n30",
        "identity_trials": 30,
        "query": "trials[50:]",
        "loader_batch_size": 128,
        "loader_shuffle": False,
        "loader_num_workers": 0,
    },
}
CPU_POLICY = {
    "device": "cpu",
    "cuda_visible_devices": "",
    "torch_num_threads": 1,
    "torch_num_interop_threads": 1,
    "deterministic_algorithms": True,
    "tf32": False,
    "autocast": False,
}


class StaticParityV4Error(RuntimeError):
    """A source/prelaunch/public-key mismatch that prohibits v4 execution."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticParityV4Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticParityV4Error(f"cannot parse {label}") from exc
    require(isinstance(value, dict), f"{label} root is not an object")
    return value


def _pin_files(root: Path, relative_root: str, pins: Mapping[str, str], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative, expected in pins.items():
        path = (root / relative_root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise StaticParityV4Error(f"unsafe {label} path: {relative}") from exc
        require(path.is_file() and not path.is_symlink(), f"missing {label}: {relative}")
        observed = sha256_file(path)
        require(observed == expected, f"{label} SHA drift: {relative}")
        result[relative] = {
            "path": str(path.relative_to(root)),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return result


def _runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    require(executable.is_file() and not executable.is_symlink(), "SPINT Python executable missing or unsafe")
    return {
        "host": socket.gethostname(),
        "python": {
            "path": str(executable),
            "sha256": sha256_file(executable),
            "bytes": executable.stat().st_size,
            "implementation": getattr(sys.implementation, "name", ""),
            "version": sys.version,
        },
    }


def static_authority(root: Path = ROOT) -> dict[str, Any]:
    """Verify v3 closure, v4 sources, root public key, and static runtime pin."""
    root = root.resolve()
    v3_sources = _pin_files(root, "", V3_SOURCES, "v3 source")
    v3_artifacts = _pin_files(root, V3_ROOT, V3_ARTIFACTS, "v3 immutable artifact")
    for artifact in v3_artifacts.values():
        path = root / artifact["path"]
        require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"v3 artifact is not 0444: {path.name}")
    v3_draft = _read_json(root / V3_ROOT / "parity_protocol_draft.json", "v3 draft")
    v3_receipt = _read_json(root / V3_ROOT / "receipt.json", "v3 receipt")
    v3_seal = _read_json(root / V3_ROOT / "seal.json", "v3 seal")
    require(v3_draft.get("status") == "NOT_AUTHORIZED_FOR_PARITY_EXECUTION", "v3 draft unexpectedly authorizes")
    require(
        v3_receipt.get("status") == v3_seal.get("status")
        == "STATIC_PROTOCOL_AUDITED_V3_DATA_ADAPTER_PARITY_EXECUTION_NOT_AUTHORIZED",
        "v3 receipt/seal status drift",
    )
    require(
        v3_draft.get("authority", {}).get("parity_v2_disposition", {}).get("status")
        == "INSUFFICIENT_DATA_ADAPTER_PARITY_NON_AUTHORIZING",
        "v3 parity-v2 disposition drift",
    )
    v4_sources = _pin_files(
        root, "", {relative: sha256_file(root / relative) for relative in V4_SOURCES}, "v4 source"
    )
    key_path = (root / PUBLIC_KEY).resolve()
    require(key_path.is_file() and not key_path.is_symlink(), "dedicated parity-v4 root public key missing/unsafe")
    require(key_path.stat().st_size == PUBLIC_KEY_BYTES, "dedicated parity-v4 root public key byte drift")
    require(sha256_file(key_path) == PUBLIC_KEY_SHA256, "dedicated parity-v4 root public key SHA drift")
    source_snapshot = {
        **{str((root / relative).resolve()): expected for relative, expected in V3_SOURCES.items()},
        **{str((root / relative).resolve()): entry["sha256"] for relative, entry in v4_sources.items()},
    }
    return {
        "v3_sources": v3_sources,
        "v3_artifacts": v3_artifacts,
        "v4_sources": v4_sources,
        "fixture_pins": FIXTURE_PINS,
        "public_key": {"path": str(key_path), "sha256": PUBLIC_KEY_SHA256, "bytes": PUBLIC_KEY_BYTES},
        "runtime_identity": _runtime_identity(),
        "cpu_policy": CPU_POLICY,
        "source_snapshot": source_snapshot,
        "output_parent": str((root / OUTPUT_PARENT_RELATIVE).resolve()),
        "claim_root": str((root / CLAIM_ROOT_RELATIVE).resolve()),
    }


def _base_execution_policy(authority: Mapping[str, Any]) -> dict[str, Any]:
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
        "schema_version": 4,
        "kind": "dandi_000688_subc_cpu_parity_execution_prelaunch_draft_v4",
        "status": "NOT_AUTHORIZED_FOR_CPU_PARITY_EXECUTION",
        "append_only": True,
        "authority": authority,
        "execution_policy_base": _base_execution_policy(authority),
        "future_authorization": {
            "detached_signature": {
                "authorization_envelope_schema": "dandi_000688_subc_parity_execution_authorization_envelope_v4",
                "authorization_schema": "dandi_000688_subc_parity_execution_authorization_v4",
                "signature": "separate strict-base64 .sig, exactly 64 raw Ed25519 bytes over canonical authorization file bytes",
                "private_key_workspace_policy": "FORBIDDEN",
            },
            "maximum_validity_seconds": 900,
            "permitted_action": "concrete_parity_after_future_authorization",
            "one_time_nonce": "64 lowercase hexadecimal characters; O_CREAT|O_EXCL immutable claim after all validation",
            "output_root": "one new child below output_parent only; no retry or overwrite",
            "validation_order": [
                "stored prelaunch/source/public-key/runtime validation",
                "signature and authorization binding validation",
                "output freshness validation",
                "atomic nonce claim",
                "only then import v3 helper/Torch/data owners",
            ],
            "external_subm_scoring_permitted": False,
        },
        "operations_by_this_prelaunch": {
            "checkpoint_files_opened": 0,
            "normalizer_npz_files_opened": 0,
            "nwb_files_opened": 0,
            "v3_helper_imported": False,
            "torch_or_owner_imported": False,
            "model_forward_calls": 0,
            "external_subm_scoring_performed": False,
        },
    }


def build_receipt(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_draft(root)
    receipt = {
        "schema_version": 4,
        "receipt_kind": "dandi_000688_subc_cpu_parity_execution_prelaunch_receipt_v4",
        "status": "STATIC_V4_CPU_PARITY_EXECUTION_PACKAGE_NOT_AUTHORIZED",
        "append_only": True,
        "draft": {
            "filename": "parity_execution_prelaunch_draft.json",
            "sha256": hashlib.sha256(canonical_bytes(draft)).hexdigest(),
        },
        "operations": dict(draft["operations_by_this_prelaunch"]),
        "blocked_until": [
            "root signs a detached authorization with the dedicated parity-v4 private key outside the workspace",
            "the authorization is unexpired, source/pin/runtime-bound, one-time, and names one fresh v4 output root",
            "root separately reviews the resulting sealed consumed-sub-C compatibility receipt",
        ],
    }
    return draft, receipt


def _write_immutable(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"immutable mode failed: {path}")
    return hashlib.sha256(raw).hexdigest()


def write_prelaunch(output_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"v4 prelaunch root already exists: {output_dir}")
    draft, receipt = build_receipt(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path = output_dir / "parity_execution_prelaunch_draft.json"
    receipt_path = output_dir / "receipt.json"
    seal_path = output_dir / "seal.json"
    draft_sha = _write_immutable(draft_path, draft)
    require(draft_sha == receipt["draft"]["sha256"], "v4 draft/receipt binding drift")
    receipt_sha = _write_immutable(receipt_path, receipt)
    seal = {
        "schema_version": 4,
        "kind": "dandi_000688_subc_cpu_parity_execution_prelaunch_seal_v4",
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


def load_stored_prelaunch(prelaunch_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    """Load the immutable stored v4 prelaunch and build the signed execution policy."""
    prelaunch_dir = prelaunch_dir.resolve()
    paths = tuple(
        prelaunch_dir / name
        for name in ("parity_execution_prelaunch_draft.json", "receipt.json", "seal.json")
    )
    for path in paths:
        require(path.is_file() and not path.is_symlink(), f"stored v4 artifact missing/unsafe: {path.name}")
        require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"stored v4 artifact not 0444: {path.name}")
    draft, receipt, seal = (
        _read_json(paths[0], "stored v4 draft"),
        _read_json(paths[1], "stored v4 receipt"),
        _read_json(paths[2], "stored v4 seal"),
    )
    draft_sha, receipt_sha, seal_sha = (sha256_file(paths[0]), sha256_file(paths[1]), sha256_file(paths[2]))
    require(draft.get("status") == "NOT_AUTHORIZED_FOR_CPU_PARITY_EXECUTION", "stored v4 draft status drift")
    require(receipt.get("status") == seal.get("status") == "STATIC_V4_CPU_PARITY_EXECUTION_PACKAGE_NOT_AUTHORIZED", "stored v4 receipt/seal status drift")
    require(receipt.get("draft") == {"filename": paths[0].name, "sha256": draft_sha}, "stored v4 draft/receipt binding drift")
    require(
        seal.get("artifacts")
        == [
            {"path": paths[0].name, "sha256": draft_sha, "bytes": paths[0].stat().st_size, "mode": "0444"},
            {"path": paths[1].name, "sha256": receipt_sha, "bytes": paths[1].stat().st_size, "mode": "0444"},
        ],
        "stored v4 seal binding drift",
    )
    live = static_authority(root)
    require(draft.get("authority") == live, "stored v4 authority differs from live source/pin/runtime state")
    base = draft.get("execution_policy_base")
    require(isinstance(base, Mapping), "stored v4 execution policy base missing")
    bindings = {
        "prelaunch_bundle": {
            "draft": {"path": str(paths[0]), "sha256": draft_sha, "bytes": paths[0].stat().st_size, "mode": "0444"},
            "receipt": {"path": str(paths[1]), "sha256": receipt_sha, "bytes": paths[1].stat().st_size, "mode": "0444"},
            "seal": {"path": str(paths[2]), "sha256": seal_sha, "bytes": paths[2].stat().st_size, "mode": "0444"},
        },
        "v3_sources": {key: value["sha256"] for key, value in live["v3_sources"].items()},
        "v3_bundle": {key: value["sha256"] for key, value in live["v3_artifacts"].items()},
        "fixture_pins": live["fixture_pins"],
        "runtime_identity": live["runtime_identity"],
        "cpu_policy": live["cpu_policy"],
        "public_key": live["public_key"],
        "source_snapshot": live["source_snapshot"],
    }
    policy = {**dict(base), "authorization_bindings": bindings}
    policy_sha256 = hashlib.sha256(canonical_bytes(policy)).hexdigest()
    policy["execution_policy_sha256"] = policy_sha256
    return {
        "draft_sha256": draft_sha,
        "receipt_sha256": receipt_sha,
        "seal_sha256": seal_sha,
        "status": receipt["status"],
        "execution_policy": policy,
        "execution_policy_sha256": policy_sha256,
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
    except StaticParityV4Error as exc:
        print(f"FAIL_CLOSED: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
