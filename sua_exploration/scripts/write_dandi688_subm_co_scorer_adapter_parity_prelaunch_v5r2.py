#!/usr/bin/env python3
"""Write static-only V5R2 evidence for explicit V5-source closure."""
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
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_prelaunch_v5r2"
DOC = "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5R2.md"
V3_ROOT = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_prelaunch_v3"
V4_ROOT = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_prelaunch_v4"
V5_ROOT = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_prelaunch_v5"
INCIDENT = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_incident_v4_20260805_0654/receipt.json"
PUBLIC_KEY = "sua_exploration/configs/dandi688_subc_parity_v4_root_ed25519_public.pem"
PUBLIC_KEY_SHA256 = "a541b5aabc7922251e797da13727b97ceb4ed48de7e95d5b1c4031ba96c602a9"
PUBLIC_KEY_BYTES = 113
OUTPUT_PARENT_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_v5r2_runs"
CLAIM_ROOT_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_v5r2_nonce_claims"

V3_SOURCES = {
    "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V3.md": "f8c3da98d38c3d061f2d89d5514cf1d5453424b0a458d5440a4a64dd3c7bf802",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v3.py": "c7951beb13618b849c43a2d296a3b1550f0a644efb68e872567fd14368759a9b",
    "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v3.py": "e70da4456c3c78ebbf0c02c34533ea4b5b2ed0313a22aee5c9451948d6c9986f",
    "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v3.py": "fceeeaec8e72346f1dd41e849d74a00831f91df5c02d545b9d3fa1608a6a6f8d",
    "sua_exploration/tests/test_dandi688_subm_co_scorer_adapter_parity_v3.py": "2f19994b1104307b6cf3b7fe491f28328ff22d3e75f594c0332f7b2930fe87c5",
}
V4_SOURCES = {
    "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V4.md": "2b175d7d7ab9886621f4eaf4b13d63cef0ffe9de72ddb91b93fd102be8def23b",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_execution_v4.py": "4fb6aaf5b060a196e558b872f3df45f378ee91ebad8b1e0cc1bb6da064308167",
    "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v4.py": "6cd8f93b82b24ae01dcbd97471bbff550c0a66ad09ace3f3bdf9ce50cd8adf0a",
    "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v4.py": "17e535b12eca0c334c8c2f0ea5a9c9c491694088e5ae56d64b0e5e1d0d74ebac",
    "sua_exploration/tests/test_dandi688_subm_co_scorer_adapter_parity_v4.py": "e353e05071e6871b51134fb9ba84b6da61fc40819cc347ad6b5fdad4e959fb27",
}
# V5 is sealed. V5R2 names its full six-source closure directly instead of
# relying only on the older combined source_snapshot.
V5_SOURCES = {
    "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md": "48ef81af065d1270331a631b928640223dded4a1c639c81a666e81050dd2c28d",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py": "563f095bdfab38e5df4cd6087e9dac27f2fb58a81190e0b9ad920215fba7adb4",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_execution_v5.py": "e6074d344f0bb4f3d3027884b0248a22ece5861e49bb56b87fd3dc1921f8353c",
    "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v5.py": "4a36477615edcba692e0e63170510b568753b2881e75996bbbc36b6d9e94106f",
    "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v5.py": "209801edef7e452693157f7bdd0f829808240424440b1f2ddd20d9506966e24e",
    "sua_exploration/tests/test_dandi688_subm_co_scorer_adapter_parity_v5.py": "8893c0b914f66804424bfd2b9d46434ba95d31fcd48c165bcde09fa3c21d942d",
}
V3_ARTIFACTS = {
    "parity_protocol_draft.json": "96200f471c7a38277508071ab9c483e623defbc9038946468408cd495a4e6157",
    "receipt.json": "c52796b13f9e76f3dd98d62fa882158c6ae3ad4237f8896753741c259b383d8b",
    "seal.json": "e0088a63b97717b9d984409485da20fdaca8e378a01750733f4bb60b02191cb1",
}
V4_ARTIFACTS = {
    "parity_execution_prelaunch_draft.json": "2adf4aa1cff94fe9a4ee7fcc1ed95d8799d8967aa35501ac9716747fee7b81a6",
    "receipt.json": "54a7e02248ad44b611d6a11c9a81529f543fccfa1bd1deee3995e1c53b8fb1ca",
    "seal.json": "0129291ff87094a518ae14a2e80898680ad7aebbe9437782fa5d3b11f1f96408",
}
V5_ARTIFACTS = {
    "parity_execution_prelaunch_draft.json": "ddd5e799556aee08c8a501f973ddc33601ee079f82e0d0bfcfd26b41989c6785",
    "receipt.json": "be0588baf523f9ec9af2669400b286bbb1ec60229fb6a7f6fe7bd62a1b2c4691",
    "seal.json": "4e0b15bb2cc7cbd078f825181a55f6bcf585de515cbf3c7bda02bdfffa53104a",
}
V5_EXECUTION_POLICY_SHA256 = "54dd3af7a05958b5a391db7891b240e2c7ffda8beba520a550f04b031f8893f9"
INCIDENT_SHA256 = "4c676146c5855f91591aaa13fc1d0b312ebbb51138584bd39c02c2888f47d0bc"
V5R2_SOURCE_PATHS = (
    DOC,
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5r2.py",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_execution_v5r2.py",
    "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v5r2.py",
    "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v5r2.py",
    "sua_exploration/tests/test_dandi688_subm_co_scorer_adapter_parity_v5r2.py",
)
FIXTURE_PINS = {
    "checkpoint": {"path": "sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_t4_s44/epoch_ckpts/epoch_011.ckpt", "sha256": "a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6", "bytes": 64769167},
    "teacher": {"path": "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt", "sha256": "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d", "bytes": 55195903},
    "consumed_subc_nwb": {"path": "sua_exploration/data/dandi_000688/sub-C/sub-C_ses-CO-20151103_behavior+ecephys.nwb", "session": "sub-C_ses-CO-20151103", "sha256": "7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7", "bytes": 62145872},
    "schema_bridge": {"delegated_to_exact_v5_source_closure": True, "only_cast_fields": ["start", "stop"], "rebinning": False, "selection_changed_by_bridge": False, "t4_changed_by_bridge": False, "query_valid_starts_changed_by_bridge": False},
}
CPU_POLICY = {"device": "cpu", "cuda_visible_devices": "", "torch_num_threads": 1, "torch_num_interop_threads": 1, "deterministic_algorithms": True, "tf32": False, "autocast": False}


class StaticParityV5R2Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticParityV5R2Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticParityV5R2Error(f"cannot parse {label}") from exc
    require(isinstance(value, dict), f"{label} root not object")
    return value


def _source_map(root: Path, expected: Mapping[str, str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise StaticParityV5R2Error(f"unsafe {label}: {relative}") from exc
        require(path.is_file() and not path.is_symlink(), f"missing {label}: {relative}")
        observed = sha256_file(path)
        require(observed == digest, f"{label} SHA drift: {relative}")
        result[relative] = observed
    return result


def _current_source_map(root: Path, paths: tuple[str, ...], label: str) -> dict[str, str]:
    expected: dict[str, str] = {}
    for relative in paths:
        path = (root / relative).resolve()
        require(path.is_file() and not path.is_symlink(), f"missing {label}: {relative}")
        expected[relative] = sha256_file(path)
    return _source_map(root, expected, label)


def _artifact_map(root: Path, relative_root: str, expected: Mapping[str, str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, digest in expected.items():
        path = (root / relative_root / name).resolve()
        require(path.is_file() and not path.is_symlink(), f"missing {label}: {name}")
        require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"mutable {label}: {name}")
        observed = sha256_file(path)
        require(observed == digest, f"{label} SHA drift: {name}")
        result[name] = observed
    return result


def _runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    require(executable.is_file() and not executable.is_symlink(), "SPINT Python unsafe")
    return {"host": socket.gethostname(), "python": {"path": str(executable), "sha256": sha256_file(executable), "bytes": executable.stat().st_size, "implementation": getattr(sys.implementation, "name", ""), "version": sys.version}}


def _combined_source_snapshot(root: Path, *maps: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for source_map in maps:
        for relative, digest in source_map.items():
            absolute = str((root / relative).resolve())
            require(absolute not in result, f"duplicate source pin: {relative}")
            result[absolute] = digest
    return result


def _require_explicit_v5_closure(authority: Mapping[str, Any], root: Path) -> None:
    v5_sources = authority.get("v5_sources")
    snapshot = authority.get("source_snapshot")
    require(isinstance(v5_sources, Mapping), "explicit V5 source map missing")
    require(dict(v5_sources) == V5_SOURCES, "explicit V5 source map changed or incomplete")
    require(isinstance(snapshot, Mapping), "combined source snapshot missing")
    for relative, digest in V5_SOURCES.items():
        absolute = str((root / relative).resolve())
        require(snapshot.get(absolute) == digest, f"combined snapshot V5 mismatch: {relative}")


def static_authority(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    v3_sources = _source_map(root, V3_SOURCES, "V3 source")
    v4_sources = _source_map(root, V4_SOURCES, "V4 source")
    v5_sources = _source_map(root, V5_SOURCES, "V5 source")
    v5r2_sources = _current_source_map(root, V5R2_SOURCE_PATHS, "V5R2 source")
    v3_artifacts = _artifact_map(root, V3_ROOT, V3_ARTIFACTS, "V3 artifact")
    v4_artifacts = _artifact_map(root, V4_ROOT, V4_ARTIFACTS, "V4 artifact")
    v5_artifacts = _artifact_map(root, V5_ROOT, V5_ARTIFACTS, "V5 artifact")
    incident_path = root / INCIDENT
    require(incident_path.is_file() and not incident_path.is_symlink() and stat.S_IMODE(incident_path.stat().st_mode) == 0o444, "V4 incident missing/unsafe/mutable")
    require(sha256_file(incident_path) == INCIDENT_SHA256, "V4 incident SHA drift")
    incident = _read_json(incident_path, "V4 incident")
    require(incident.get("status") == "FAILED_CLOSED_AFTER_AUTHORIZATION_BEFORE_MODEL_FORWARD_SCHEMA_TYPE_MISMATCH", "V4 incident disposition drift")
    require(incident.get("failure", {}).get("model_or_r2_result") is False, "V4 incident unexpectedly has scientific result")
    key = (root / PUBLIC_KEY).resolve()
    require(key.is_file() and not key.is_symlink() and key.stat().st_size == PUBLIC_KEY_BYTES, "dedicated key missing/size drift")
    require(sha256_file(key) == PUBLIC_KEY_SHA256, "dedicated key SHA drift")
    authority = {
        "v3_sources": v3_sources, "v3_artifacts": v3_artifacts,
        "v4_sources": v4_sources, "v4_artifacts": v4_artifacts,
        "v5_sources": v5_sources, "v5_artifacts": v5_artifacts,
        "v5_execution_policy_sha256": V5_EXECUTION_POLICY_SHA256,
        "v5r2_sources": v5r2_sources,
        "v4_incident": {"path": INCIDENT, "sha256": INCIDENT_SHA256, "status": incident["status"]},
        "fixture_pins": FIXTURE_PINS,
        "public_key": {"path": str(key), "sha256": PUBLIC_KEY_SHA256, "bytes": PUBLIC_KEY_BYTES},
        "runtime_identity": _runtime_identity(), "cpu_policy": CPU_POLICY,
        "source_snapshot": _combined_source_snapshot(root, v3_sources, v4_sources, v5_sources, v5r2_sources),
        "output_parent": str((root / OUTPUT_PARENT_RELATIVE).resolve()),
        "claim_root": str((root / CLAIM_ROOT_RELATIVE).resolve()),
    }
    _require_explicit_v5_closure(authority, root)
    return authority


def _policy_base(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {"public_key": dict(authority["public_key"]), "source_pins": dict(authority["source_snapshot"]), "runtime_identity": dict(authority["runtime_identity"]), "cpu_policy": dict(authority["cpu_policy"]), "output_parent": authority["output_parent"], "claim_root": authority["claim_root"]}


def build_draft(root: Path = ROOT) -> dict[str, Any]:
    authority = static_authority(root)
    return {
        "schema_version": "5r2",
        "kind": "dandi_000688_subc_explicit_v5_source_closure_cpu_parity_execution_prelaunch_draft_v5r2",
        "status": "NOT_AUTHORIZED_FOR_EXPLICIT_V5_SOURCE_CLOSURE_CPU_PARITY_EXECUTION",
        "append_only": True, "authority": authority,
        "execution_policy_base": _policy_base(authority),
        "authorization_contract": {
            "authorization_schema": "dandi_000688_subc_parity_explicit_v5_source_closure_execution_authorization_v5r2",
            "envelope_schema": "dandi_000688_subc_parity_explicit_v5_source_closure_execution_authorization_envelope_v5r2",
            "permitted_action": "concrete_parity_after_future_authorization_v5r2",
            "maximum_validity_seconds": 900, "fresh_v5r2_nonce_required": True,
            "v4_v5_nonce_reuse": "FORBIDDEN",
            "import_order": "authorize+claim before V5R2/V5/V3/Torch/data imports",
            "explicit_v5_sources_required": True,
        },
        "operations_by_this_prelaunch": {
            "checkpoint_files_opened": 0, "normalizer_npz_files_opened": 0,
            "nwb_files_opened": 0, "v5r2_v5_v3_torch_imported": False,
            "model_forward_calls": 0, "external_subm_scoring_performed": False,
        },
    }


def build_receipt(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_draft(root)
    receipt = {
        "schema_version": "5r2",
        "receipt_kind": "dandi_000688_subc_explicit_v5_source_closure_cpu_parity_execution_prelaunch_receipt_v5r2",
        "status": "STATIC_V5R2_EXPLICIT_V5_SOURCE_CLOSURE_PACKAGE_NOT_AUTHORIZED",
        "append_only": True,
        "draft": {"filename": "parity_execution_prelaunch_draft.json", "sha256": hashlib.sha256(canonical_bytes(draft)).hexdigest()},
        "operations": dict(draft["operations_by_this_prelaunch"]),
        "blocked_until": ["root reviews V5R2 and signs a new detached V5R2 capability outside the workspace", "new authorization binds explicit V5 source closure, V5R2 source closure, predecessor bundles, incident, runtime, and a fresh nonce/output root"],
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
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, "immutable mode failed")
    return hashlib.sha256(raw).hexdigest()


def write_prelaunch(output_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), "V5R2 prelaunch root already exists")
    draft, receipt = build_receipt(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path = output_dir / "parity_execution_prelaunch_draft.json"
    receipt_path = output_dir / "receipt.json"
    seal_path = output_dir / "seal.json"
    draft_sha = _write_immutable(draft_path, draft)
    require(draft_sha == receipt["draft"]["sha256"], "draft receipt binding drift")
    receipt_sha = _write_immutable(receipt_path, receipt)
    seal = {
        "schema_version": "5r2", "kind": "dandi_000688_subc_explicit_v5_source_closure_cpu_parity_execution_prelaunch_seal_v5r2",
        "status": receipt["status"], "append_only": True,
        "artifacts": [
            {"path": draft_path.name, "sha256": draft_sha, "bytes": draft_path.stat().st_size, "mode": "0444"},
            {"path": receipt_path.name, "sha256": receipt_sha, "bytes": receipt_path.stat().st_size, "mode": "0444"},
        ],
        "future_execution_not_authorized_by_this_seal": True,
    }
    seal_sha = _write_immutable(seal_path, seal)
    return {"output_dir": str(output_dir), "draft": {"path": str(draft_path), "sha256": draft_sha}, "receipt": {"path": str(receipt_path), "sha256": receipt_sha}, "seal": {"path": str(seal_path), "sha256": seal_sha}, "status": receipt["status"]}


def _authorization_bindings(authority: Mapping[str, Any], paths: tuple[Path, Path, Path], hashes: tuple[str, str, str]) -> dict[str, Any]:
    draft_path, receipt_path, seal_path = paths
    draft_sha, receipt_sha, seal_sha = hashes
    return {
        "prelaunch_bundle": {
            "draft": {"path": str(draft_path), "sha256": draft_sha, "bytes": draft_path.stat().st_size, "mode": "0444"},
            "receipt": {"path": str(receipt_path), "sha256": receipt_sha, "bytes": receipt_path.stat().st_size, "mode": "0444"},
            "seal": {"path": str(seal_path), "sha256": seal_sha, "bytes": seal_path.stat().st_size, "mode": "0444"},
        },
        "v3_sources": dict(authority["v3_sources"]), "v3_bundle": dict(authority["v3_artifacts"]),
        "v4_sources": dict(authority["v4_sources"]), "v4_bundle": dict(authority["v4_artifacts"]),
        "v5_sources": dict(authority["v5_sources"]), "v5_bundle": dict(authority["v5_artifacts"]),
        "v5_execution_policy_sha256": authority["v5_execution_policy_sha256"],
        "v5r2_sources": dict(authority["v5r2_sources"]),
        "v4_incident": dict(authority["v4_incident"]), "fixture_pins": dict(authority["fixture_pins"]),
        "runtime_identity": dict(authority["runtime_identity"]), "cpu_policy": dict(authority["cpu_policy"]),
        "public_key": dict(authority["public_key"]), "source_snapshot": dict(authority["source_snapshot"]),
    }


def load_stored_prelaunch(prelaunch_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    prelaunch_dir = prelaunch_dir.resolve()
    paths = tuple(prelaunch_dir / name for name in ("parity_execution_prelaunch_draft.json", "receipt.json", "seal.json"))
    for path in paths:
        require(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444, f"stored V5R2 artifact unsafe: {path.name}")
    draft, receipt, seal = (_read_json(paths[0], "V5R2 draft"), _read_json(paths[1], "V5R2 receipt"), _read_json(paths[2], "V5R2 seal"))
    draft_sha, receipt_sha, seal_sha = (sha256_file(paths[0]), sha256_file(paths[1]), sha256_file(paths[2]))
    require(draft.get("status") == "NOT_AUTHORIZED_FOR_EXPLICIT_V5_SOURCE_CLOSURE_CPU_PARITY_EXECUTION", "V5R2 draft status drift")
    require(receipt.get("status") == seal.get("status") == "STATIC_V5R2_EXPLICIT_V5_SOURCE_CLOSURE_PACKAGE_NOT_AUTHORIZED", "V5R2 receipt/seal status drift")
    require(receipt.get("draft") == {"filename": paths[0].name, "sha256": draft_sha}, "V5R2 draft receipt binding drift")
    require(seal.get("artifacts") == [{"path": paths[0].name, "sha256": draft_sha, "bytes": paths[0].stat().st_size, "mode": "0444"}, {"path": paths[1].name, "sha256": receipt_sha, "bytes": paths[1].stat().st_size, "mode": "0444"}], "V5R2 seal binding drift")
    authority = static_authority(root)
    require(draft.get("authority") == authority, "stored V5R2 authority drift")
    _require_explicit_v5_closure(draft["authority"], root.resolve())
    require(draft.get("execution_policy_base") == _policy_base(authority), "stored V5R2 policy-base drift")
    bindings = _authorization_bindings(authority, paths, (draft_sha, receipt_sha, seal_sha))
    require(bindings["v5_sources"] == authority["v5_sources"], "explicit V5 source binding drift")
    policy = {**dict(draft["execution_policy_base"]), "authorization_bindings": bindings}
    policy_sha = hashlib.sha256(canonical_bytes(policy)).hexdigest()
    policy["execution_policy_sha256"] = policy_sha
    return {"draft_sha256": draft_sha, "receipt_sha256": receipt_sha, "seal_sha256": seal_sha, "status": receipt["status"], "execution_policy": policy, "execution_policy_sha256": policy_sha}


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
    except StaticParityV5R2Error as exc:
        print(f"FAIL_CLOSED: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
