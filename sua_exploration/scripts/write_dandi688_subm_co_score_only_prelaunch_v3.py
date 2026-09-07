#!/usr/bin/env python3
"""Write static-only V3 score-only prelaunch evidence pinned to V5R2 parity."""
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
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v3"
DOC = "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORE_ONLY_PROTOCOL_V3.md"
V2_ROOT = "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v2"
V5R2_RUN_ROOT = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_execution_v5r2_runs/run_20260805_072001_root"
PUBLIC_KEY = "sua_exploration/configs/dandi688_subc_parity_v4_root_ed25519_public.pem"
PUBLIC_KEY_SHA256 = "a541b5aabc7922251e797da13727b97ceb4ed48de7e95d5b1c4031ba96c602a9"
PUBLIC_KEY_BYTES = 113
OUTPUT_PARENT_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_score_only_v3_runs"
CLAIM_ROOT_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_score_only_v3_nonce_claims"

V2_SOURCES = {
    "sua_exploration/mc_maze/subm_co_score_only_v2.py": "213e5495b4fa8776967ea07bf57743db18619b74aea593bac1d17d5944308e4d",
    "sua_exploration/scripts/run_dandi688_subm_co_score_only_v2.py": "dc271dbe0b7ede31865f5c35890444b969296096f9107f960183079cc63a76c7",
    "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch_v2.py": "d8f4b37a091233c421bd052238c165c194bb31318ed2c8172e0049a444d2f58c",
    "sua_exploration/tests/test_dandi688_subm_co_score_only_v2.py": "944e5a35ca7c61d18655a4aa4e3c661252cd8cc9e97e687034a3c28f4d3ff339",
}
V2_ARTIFACTS = {
    "prelaunch_authorization_draft.json": "2a8455fff85e8a0644e5dbb1f9932f126db55b856842cc3207d3f59bf7c59eae",
    "receipt.json": "44c51dd5aa399636138f18a43ab6dbf44a2d2fe065ace3600a61e044e25f0868",
    "prelaunch_artifact_seal.json": "2b8a3591eb7b3d18b2587f38526cbe139c7b7d40686d8a041b992f07f13635ea",
}
V5R2_SUCCESS_ARTIFACTS = {
    "parity_execution_receipt.json": {"sha256": "faace6493fcf939e87bf0f5ad219f75df739434b7941fccc7f654c23d78c63e8", "bytes": 2554},
    "input_trace.json": {"sha256": "c3309f4d9e60de96517ea9c2cfad5d1f89c28398f3644cbb9a379d17779ae859", "bytes": 117385},
    "environment.json": {"sha256": "cf2bf2010ba326a3445d409902cb3343c05380ae5b29825d250a997d06345f02", "bytes": 6345},
    "seal.json": {"sha256": "b5f92c71a0e601fcdbd8866d4d7e1ec882ce9f285936431bfbe27aa7e9a03e18", "bytes": 628},
}
V3_SOURCE_PATHS = (
    DOC,
    "sua_exploration/mc_maze/subm_co_score_only_v3.py",
    "sua_exploration/scripts/run_dandi688_subm_co_score_only_v3.py",
    "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch_v3.py",
    "sua_exploration/tests/test_dandi688_subm_co_score_only_v3.py",
)
CPU_POLICY = {"device": "cpu", "cuda_visible_devices": "", "torch_num_threads": 1, "torch_num_interop_threads": 1, "deterministic_algorithms": True, "tf32": False, "autocast": False}
EXPECTED_R2 = 0.4910046458244324
EXPECTED_PREDICTION_DIGEST = {"bytes": 232712, "dtype": "float32", "finite": True, "sha256": "956d8c8c9ba7165cb9e5bcf5f4a971a6e038a525014bcc8ced838bf9bede5f40", "shape": [29089, 2]}
EXPECTED_TARGET_DIGEST = {"bytes": 232712, "dtype": "float32", "finite": True, "sha256": "637f346fdd07106bb2d7620803685dfc86b9c4a5c487ad3a9b105040171180ce", "shape": [29089, 2]}
EXPECTED_V5_BRIDGE_SHA256 = "563f095bdfab38e5df4cd6087e9dac27f2fb58a81190e0b9ad920215fba7adb4"


class StaticScoreV3Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticScoreV3Error(message)


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
        raise StaticScoreV3Error(f"cannot parse {label}") from exc
    require(isinstance(value, dict), f"{label} root not object")
    return value


def _source_map(root: Path, expected: Mapping[str, str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise StaticScoreV3Error(f"unsafe {label}: {relative}") from exc
        require(path.is_file() and not path.is_symlink(), f"missing {label}: {relative}")
        observed = sha256_file(path)
        require(observed == digest, f"{label} SHA drift: {relative}")
        result[relative] = observed
    return result


def _current_source_map(root: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for relative in V3_SOURCE_PATHS:
        path = (root / relative).resolve()
        require(path.is_file() and not path.is_symlink(), f"missing V3 source: {relative}")
        expected[relative] = sha256_file(path)
    return _source_map(root, expected, "V3 source")


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


def _verify_v5r2_success_semantics(
    receipt: Mapping[str, Any], input_trace: Mapping[str, Any], environment: Mapping[str, Any], seal: Mapping[str, Any]
) -> None:
    status = "PARITY_CONFIRMED_CONSUMED_SUBC_DEV_SESSION_PENDING_SEPARATE_ROOT_REVIEW_V5R2"
    require(receipt.get("schema") == "dandi_000688_subc_explicit_v5_source_closure_execution_receipt_v5r2", "V5R2 receipt schema drift")
    require(receipt.get("status") == status, "V5R2 receipt status drift")
    require(receipt.get("external_subm_scoring_performed") is False, "V5R2 parity receipt external scoring drift")
    reference, adapter = receipt.get("reference"), receipt.get("adapter")
    require(isinstance(reference, Mapping) and isinstance(adapter, Mapping), "V5R2 reference/adapter records missing")
    for record in (reference, adapter):
        require(record.get("r2") == EXPECTED_R2, "V5R2 parity R2 drift")
        require(record.get("prediction_digest") == EXPECTED_PREDICTION_DIGEST, "V5R2 prediction digest drift")
        require(record.get("target_digest") == EXPECTED_TARGET_DIGEST, "V5R2 target digest drift")
    require(reference == adapter, "V5R2 reference/adapter exact result mismatch")
    observer = receipt.get("observer")
    require(observer == {"forward_calls": 456, "input_comparisons": 1, "metric_computes": 2, "metric_updates": 456}, "V5R2 observer count drift")
    require(receipt.get("input_trace") == {"path": "input_trace.json", "sha256": V5R2_SUCCESS_ARTIFACTS["input_trace.json"]["sha256"]}, "V5R2 receipt/input hash binding drift")
    require(receipt.get("environment") == {"path": "environment.json", "sha256": V5R2_SUCCESS_ARTIFACTS["environment.json"]["sha256"]}, "V5R2 receipt/environment hash binding drift")
    shared = input_trace.get("shared_observer")
    bridge = input_trace.get("adapter_schema_bridge")
    require(isinstance(shared, Mapping) and shared.get("input_exact") is True, "V5R2 shared input exactness drift")
    require(isinstance(bridge, Mapping), "V5R2 bridge trace missing")
    require(bridge.get("raw_owner_chronology_sha256_before") == bridge.get("raw_owner_chronology_sha256_after"), "V5R2 raw chronology changed")
    require(bridge.get("only_start_stop_cast") is True and bridge.get("bin_recomputation") is False, "V5R2 bridge cast/bin rule drift")
    require(bridge.get("trial_selection_changed_by_schema_bridge") is False and bridge.get("t4_changed_by_schema_bridge") is False and bridge.get("query_valid_starts_changed_by_schema_bridge") is False, "V5R2 bridge semantic-change drift")
    require(bridge.get("support_trials") == 50 and bridge.get("identity_trials") == 30 and bridge.get("query_boundary") == "owner-loader valid_starts after first 50 rewarded trials", "V5R2 chronology contract drift")
    require(environment.get("schema") == "dandi_000688_subc_explicit_v5_source_closure_environment_v5r2", "V5R2 environment schema drift")
    require(environment.get("cpu_policy") == CPU_POLICY, "V5R2 CPU policy drift")
    require(environment.get("thread_environment", {}).get("CUDA_VISIBLE_DEVICES") == "", "V5R2 CUDA environment drift")
    source_pins = environment.get("source_pins")
    v5_sources = environment.get("v5_sources")
    require(isinstance(source_pins, Mapping) and isinstance(v5_sources, Mapping), "V5R2 source evidence missing")
    bridge_path = str((ROOT / "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py").resolve())
    require(source_pins.get(bridge_path) == EXPECTED_V5_BRIDGE_SHA256, "V5R2 V5 bridge source pin drift")
    require(v5_sources.get("sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py") == EXPECTED_V5_BRIDGE_SHA256, "V5R2 explicit V5 bridge source drift")
    require(seal.get("schema") == "dandi_000688_subc_explicit_v5_source_closure_execution_seal_v5r2" and seal.get("status") == status, "V5R2 seal schema/status drift")
    require(seal.get("external_subm_scoring_performed") is False, "V5R2 seal external scoring drift")
    require(seal.get("artifacts") == [
        {"path": "input_trace.json", "sha256": V5R2_SUCCESS_ARTIFACTS["input_trace.json"]["sha256"], "bytes": V5R2_SUCCESS_ARTIFACTS["input_trace.json"]["bytes"], "mode": "0444"},
        {"path": "environment.json", "sha256": V5R2_SUCCESS_ARTIFACTS["environment.json"]["sha256"], "bytes": V5R2_SUCCESS_ARTIFACTS["environment.json"]["bytes"], "mode": "0444"},
        {"path": "parity_execution_receipt.json", "sha256": V5R2_SUCCESS_ARTIFACTS["parity_execution_receipt.json"]["sha256"], "bytes": V5R2_SUCCESS_ARTIFACTS["parity_execution_receipt.json"]["bytes"], "mode": "0444"},
    ], "V5R2 seal artifact binding drift")


def _v5r2_success_authority(root: Path) -> dict[str, Any]:
    run_root = (root / V5R2_RUN_ROOT).resolve()
    payloads: dict[str, dict[str, Any]] = {}
    for name, expected in V5R2_SUCCESS_ARTIFACTS.items():
        path = run_root / name
        require(path.is_file() and not path.is_symlink(), f"missing V5R2 parity artifact: {name}")
        require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"mutable V5R2 parity artifact: {name}")
        require(path.stat().st_size == expected["bytes"] and sha256_file(path) == expected["sha256"], f"V5R2 parity artifact hash/size drift: {name}")
        payloads[name] = _read_json(path, f"V5R2 parity {name}")
    _verify_v5r2_success_semantics(
        payloads["parity_execution_receipt.json"], payloads["input_trace.json"], payloads["environment.json"], payloads["seal.json"]
    )
    source_pins = payloads["environment.json"]["source_pins"]
    require(isinstance(source_pins, Mapping) and source_pins, "V5R2 runtime source snapshot missing")
    for raw_path, digest in source_pins.items():
        require(isinstance(raw_path, str) and isinstance(digest, str), "V5R2 runtime source snapshot schema drift")
        path = Path(raw_path)
        require(path.is_file() and not path.is_symlink() and sha256_file(path) == digest, f"V5R2 runtime source drift: {raw_path}")
    return {
        "run_root": str(run_root),
        "artifacts": {name: {"path": str(run_root / name), **expected, "mode": "0444"} for name, expected in V5R2_SUCCESS_ARTIFACTS.items()},
        "runtime_source_snapshot": dict(source_pins),
        "status": payloads["parity_execution_receipt.json"]["status"],
        "r2": EXPECTED_R2,
        "prediction_target_exact": True,
        "input_exact": True,
        "chronology_unchanged": True,
        "external_subm_scoring_performed": False,
    }


def _v2_authority(root: Path) -> dict[str, Any]:
    v2_sources = _source_map(root, V2_SOURCES, "score-only V2 source")
    v2_artifacts = _artifact_map(root, V2_ROOT, V2_ARTIFACTS, "score-only V2 artifact")
    draft = _read_json(root / V2_ROOT / "prelaunch_authorization_draft.json", "score-only V2 draft")
    require(draft.get("kind") == "dandi_000688_subm_co_score_only_prelaunch_authorization_draft_v2" and draft.get("status") == "NOT_AUTHORIZED_FOR_SCORING", "score-only V2 draft schema/status drift")
    require(draft.get("runner_source_snapshot") == v2_sources, "score-only V2 source closure drift")
    matrix = draft.get("matrix")
    cohort = draft.get("frozen_common_cohort")
    require(isinstance(matrix, Mapping) and isinstance(cohort, list) and len(cohort) == 15, "frozen V2 cohort/matrix drift")
    require(matrix.get("N") == 15 and matrix.get("arms") == ["shared_t4", "shared_ts4"] and matrix.get("seeds") == [42, 43, 44] and matrix.get("views") == ["sua", "pseudo_mua"], "frozen V2 arm/seed/view matrix drift")
    require(matrix.get("sealed_cell_count") == 180 and matrix.get("query_windows_per_view_at_N15") == 708795, "frozen V2 matrix count drift")
    protocol = draft.get("score_only_protocol")
    require(isinstance(protocol, Mapping) and protocol.get("normalizer_fitting") == "FORBIDDEN" and protocol.get("optimizer_or_backward") == "FORBIDDEN", "frozen V2 score-only fence drift")
    return {"sources": v2_sources, "artifacts": v2_artifacts, "frozen_cohort": cohort, "matrix": dict(matrix), "scope_id": draft.get("scope_id"), "score_only_protocol": dict(protocol)}


def _merge_source_snapshots(*snapshots: Mapping[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for snapshot in snapshots:
        for path, digest in snapshot.items():
            absolute = str(Path(path).resolve()) if Path(path).is_absolute() else str((ROOT / path).resolve())
            if absolute in merged:
                require(merged[absolute] == digest, f"conflicting source snapshot pin: {absolute}")
            merged[absolute] = digest
    return merged


def static_authority(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    v2 = _v2_authority(root)
    v5r2 = _v5r2_success_authority(root)
    v3_sources = _current_source_map(root)
    key = (root / PUBLIC_KEY).resolve()
    require(key.is_file() and not key.is_symlink() and key.stat().st_size == PUBLIC_KEY_BYTES, "dedicated root key missing/size drift")
    require(sha256_file(key) == PUBLIC_KEY_SHA256, "dedicated root key SHA drift")
    return {
        "score_only_v2": v2,
        "v5r2_parity_execution": v5r2,
        "v3_sources": v3_sources,
        "source_snapshot": _merge_source_snapshots(v5r2["runtime_source_snapshot"], v2["sources"], v3_sources),
        "public_key": {"path": str(key), "sha256": PUBLIC_KEY_SHA256, "bytes": PUBLIC_KEY_BYTES},
        "runtime_identity": _runtime_identity(), "cpu_policy": CPU_POLICY,
        "output_parent": str((root / OUTPUT_PARENT_RELATIVE).resolve()),
        "claim_root": str((root / CLAIM_ROOT_RELATIVE).resolve()),
        "claim_boundary": {
            "frozen_arms_only": ["shared_t4", "shared_ts4"],
            "supports": "correct-vs-shuffled attachment/content contrast only",
            "absolute_t4_over_spint_claim": "UNSUPPORTED_NO_SHARED_B0_CONTROL",
            "same_window_shared_b0_added": False,
        },
    }


def _policy_base(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {"public_key": dict(authority["public_key"]), "source_pins": dict(authority["source_snapshot"]), "runtime_identity": dict(authority["runtime_identity"]), "cpu_policy": dict(authority["cpu_policy"]), "output_parent": authority["output_parent"], "claim_root": authority["claim_root"]}


def build_draft(root: Path = ROOT) -> dict[str, Any]:
    authority = static_authority(root)
    return {
        "schema_version": 3,
        "kind": "dandi_000688_subm_co_score_only_cpu_prelaunch_draft_v3",
        "status": "NOT_AUTHORIZED_FOR_FROZEN_SUBM_SCORE_ONLY_V3",
        "append_only": True, "authority": authority, "execution_policy_base": _policy_base(authority),
        "authorization_contract": {
            "authorization_schema": "dandi_000688_subm_co_score_only_execution_authorization_v3",
            "envelope_schema": "dandi_000688_subm_co_score_only_execution_authorization_envelope_v3",
            "permitted_action": "score_frozen_subm_matrix_via_v5_bridge_v3",
            "maximum_validity_seconds": 900, "fresh_v3_nonce_required": True,
            "CPU_forward_R2_only": True, "CUDA": "FORBIDDEN", "normalizer_fitting": "FORBIDDEN",
            "optimizer_or_backward": "FORBIDDEN", "target_updates": "FORBIDDEN",
            "future_fixture_owner_path": "V5 bridge_owner_chronology_for_c1_builder plus parity-proven C1 owners",
            "old_score_only_adapter_loader": "FORBIDDEN",
        },
        "operations_by_this_prelaunch": {
            "external_subm_nwb_files_opened": 0, "checkpoint_files_opened": 0,
            "normalizer_files_opened": 0, "model_forward_calls": 0, "r2_computations": 0,
            "v5_bridge_or_score_runtime_imported": False, "external_subm_scoring_performed": False,
        },
    }


def build_receipt(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_draft(root)
    receipt = {
        "schema_version": 3, "receipt_kind": "dandi_000688_subm_co_score_only_cpu_prelaunch_receipt_v3",
        "status": "STATIC_V3_SCORE_ONLY_PACKAGE_NOT_AUTHORIZED", "append_only": True,
        "draft": {"filename": "prelaunch_authorization_draft.json", "sha256": hashlib.sha256(canonical_bytes(draft)).hexdigest()},
        "operations": dict(draft["operations_by_this_prelaunch"]),
        "blocked_until": ["root reviews V3 and signs a fresh detached V3 capability outside the workspace", "future V3 score authorization binds V2 matrix, successful V5R2 parity evidence, CPU runtime, source closure, fresh nonce, output root, and exact external-NWB root"],
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
    require(not output_dir.exists(), "V3 prelaunch root already exists")
    draft, receipt = build_receipt(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path, receipt_path, seal_path = output_dir / "prelaunch_authorization_draft.json", output_dir / "receipt.json", output_dir / "seal.json"
    draft_sha = _write_immutable(draft_path, draft)
    require(draft_sha == receipt["draft"]["sha256"], "V3 draft/receipt binding drift")
    receipt_sha = _write_immutable(receipt_path, receipt)
    seal = {
        "schema_version": 3, "kind": "dandi_000688_subm_co_score_only_cpu_prelaunch_seal_v3",
        "status": receipt["status"], "append_only": True,
        "artifacts": [
            {"path": draft_path.name, "sha256": draft_sha, "bytes": draft_path.stat().st_size, "mode": "0444"},
            {"path": receipt_path.name, "sha256": receipt_sha, "bytes": receipt_path.stat().st_size, "mode": "0444"},
        ], "future_execution_not_authorized_by_this_seal": True,
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
        "score_only_v2_sources": dict(authority["score_only_v2"]["sources"]),
        "score_only_v2_bundle": dict(authority["score_only_v2"]["artifacts"]),
        "frozen_cohort": list(authority["score_only_v2"]["frozen_cohort"]),
        "frozen_matrix": dict(authority["score_only_v2"]["matrix"]),
        "v5r2_parity_execution": dict(authority["v5r2_parity_execution"]),
        "v3_sources": dict(authority["v3_sources"]), "source_snapshot": dict(authority["source_snapshot"]),
        "public_key": dict(authority["public_key"]), "runtime_identity": dict(authority["runtime_identity"]),
        "cpu_policy": dict(authority["cpu_policy"]), "claim_boundary": dict(authority["claim_boundary"]),
    }


def load_stored_prelaunch(prelaunch_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    prelaunch_dir = prelaunch_dir.resolve()
    paths = tuple(prelaunch_dir / name for name in ("prelaunch_authorization_draft.json", "receipt.json", "seal.json"))
    for path in paths:
        require(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444, f"stored V3 artifact unsafe: {path.name}")
    draft, receipt, seal = (_read_json(paths[0], "V3 draft"), _read_json(paths[1], "V3 receipt"), _read_json(paths[2], "V3 seal"))
    draft_sha, receipt_sha, seal_sha = (sha256_file(paths[0]), sha256_file(paths[1]), sha256_file(paths[2]))
    require(draft.get("status") == "NOT_AUTHORIZED_FOR_FROZEN_SUBM_SCORE_ONLY_V3", "V3 draft status drift")
    require(receipt.get("status") == seal.get("status") == "STATIC_V3_SCORE_ONLY_PACKAGE_NOT_AUTHORIZED", "V3 receipt/seal status drift")
    require(receipt.get("draft") == {"filename": paths[0].name, "sha256": draft_sha}, "V3 draft receipt binding drift")
    require(seal.get("artifacts") == [{"path": paths[0].name, "sha256": draft_sha, "bytes": paths[0].stat().st_size, "mode": "0444"}, {"path": paths[1].name, "sha256": receipt_sha, "bytes": paths[1].stat().st_size, "mode": "0444"}], "V3 seal binding drift")
    authority = static_authority(root)
    require(draft.get("authority") == authority, "stored V3 authority drift")
    require(draft.get("execution_policy_base") == _policy_base(authority), "stored V3 policy-base drift")
    bindings = _authorization_bindings(authority, paths, (draft_sha, receipt_sha, seal_sha))
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
    except StaticScoreV3Error as exc:
        print(f"FAIL_CLOSED: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
