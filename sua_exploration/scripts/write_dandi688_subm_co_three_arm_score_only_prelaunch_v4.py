#!/usr/bin/env python3
"""Write/verify a blocked three-arm V4 metadata-only preparation receipt."""
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

from sua_exploration.mc_maze.subm_co_three_arm_score_only_v4 import (  # noqa: E402
    ARMS,
    BOOTSTRAP_POLICY,
    COMPARISON_GATES,
    CPU_POLICY,
    EXPECTED_CELL_COUNT,
    EXPECTED_QUERY_WINDOWS_PER_VIEW,
    EXPECTED_SESSION_COUNT,
    SEEDS,
    VIEWS,
    build_three_arm_contract,
    canonical_bytes,
    canonical_sha256,
    checkpoint_slots_v4,
    missing_checkpoint_slots,
    runtime_identity,
    sha256_file,
    validate_checkpoint_slots,
)


DEFAULT_OUTPUT = (
    ROOT
    / "sua_exploration/results/dandi_000688_subm_co_three_arm_score_only_prelaunch_v4"
)
V2_ROOT = "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v2"
V5R2_ROOT = (
    "sua_exploration/results/"
    "dandi_000688_subm_co_scorer_adapter_parity_execution_v5r2_runs/"
    "run_20260805_072001_root"
)
ZERO4_PARITY_ROOT = (
    "sua_exploration/results/"
    "dandi_000688_subm_co_shared_zero4_adapter_parity_execution_v1"
)
SCOPE_MANIFEST = "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json"

V2_ARTIFACTS = {
    "prelaunch_authorization_draft.json": {
        "sha256": "2a8455fff85e8a0644e5dbb1f9932f126db55b856842cc3207d3f59bf7c59eae",
        "bytes": 10503,
    },
    "receipt.json": {
        "sha256": "44c51dd5aa399636138f18a43ab6dbf44a2d2fe065ace3600a61e044e25f0868",
        "bytes": 1926,
    },
    "prelaunch_artifact_seal.json": {
        "sha256": "2b8a3591eb7b3d18b2587f38526cbe139c7b7d40686d8a041b992f07f13635ea",
        "bytes": 1357,
    },
}
V5R2_ARTIFACTS = {
    "parity_execution_receipt.json": {
        "sha256": "faace6493fcf939e87bf0f5ad219f75df739434b7941fccc7f654c23d78c63e8",
        "bytes": 2554,
    },
    "input_trace.json": {
        "sha256": "c3309f4d9e60de96517ea9c2cfad5d1f89c28398f3644cbb9a379d17779ae859",
        "bytes": 117385,
    },
    "environment.json": {
        "sha256": "cf2bf2010ba326a3445d409902cb3343c05380ae5b29825d250a997d06345f02",
        "bytes": 6345,
    },
    "seal.json": {
        "sha256": "b5f92c71a0e601fcdbd8866d4d7e1ec882ce9f285936431bfbe27aa7e9a03e18",
        "bytes": 628,
    },
}
ZERO4_PARITY_ARTIFACTS = {
    "receipt.json": {
        "sha256": "226b705963342f1e85108d1ccf1da869b545b9176f1d3fe6b6eb2c75a1ea05e3",
        "bytes": 1665,
    },
    "input_parity_trace.json": {
        "sha256": "86a463d3c083759a2c0babc3682c7f7bc844f6a70e267314e0fa0dd4fa725223",
        "bytes": 273480,
    },
    "environment.json": {
        "sha256": "68e2788171c70e99f423570089cbef28ca46dbaf8d7cae7d18b40b86642d6170",
        "bytes": 933,
    },
    "seal.json": {
        "sha256": "e1397ace8cd575da4250cb287f75ac5651eb46fd424580ff1108b44f53e073da",
        "bytes": 839,
    },
}
SCOPE_MANIFEST_SHA256 = "68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55"
ADAPTER_SOURCE_PINS = {
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py":
        "563f095bdfab38e5df4cd6087e9dac27f2fb58a81190e0b9ad920215fba7adb4",
    "sua_exploration/mc_maze/paired_view_c1_shared_zero4.py":
        "ee754cb279fae1a72e88b51b1f11f94b22380e953dde972ef36eabcaf4295f88",
}

SOURCE_PATHS = (
    "sua_exploration/docs/DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V4.md",
    "sua_exploration/mc_maze/subm_co_three_arm_score_only_v4.py",
    "sua_exploration/scripts/run_dandi688_subm_co_three_arm_score_only_v4.py",
    "sua_exploration/scripts/write_dandi688_subm_co_three_arm_score_only_prelaunch_v4.py",
    "sua_exploration/tests/test_dandi688_subm_co_three_arm_score_only_v4.py",
)


class StaticThreeArmV4Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticThreeArmV4Error(message)


def _regular_under(root: Path, relative: str, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise StaticThreeArmV4Error(f"{label} escapes repository") from exc
    require(candidate.is_file() and not candidate.is_symlink(), f"missing/unsafe {label}: {relative}")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticThreeArmV4Error(f"cannot parse {label}") from exc
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def _artifact_bundle(
    root: Path, relative_root: str, expected: Mapping[str, Mapping[str, Any]], label: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payloads: dict[str, Any] = {}
    pins: dict[str, dict[str, Any]] = {}
    bundle_root = (root / relative_root).resolve()
    for name, pin in expected.items():
        path = bundle_root / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444,
            f"missing/unsafe/mutable {label}: {name}",
        )
        require(path.stat().st_size == pin["bytes"], f"{label} byte-size drift: {name}")
        require(sha256_file(path) == pin["sha256"], f"{label} SHA drift: {name}")
        payloads[name] = _read_json(path, f"{label} {name}")
        pins[name] = {
            "path": str(path),
            "sha256": pin["sha256"],
            "bytes": pin["bytes"],
            "mode": "0444",
        }
    return payloads, pins


def _source_map(root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(_regular_under(root, relative, "V4 source"))
        for relative in SOURCE_PATHS
    }


def _adapter_source_map(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in ADAPTER_SOURCE_PINS.items():
        digest = sha256_file(_regular_under(root, relative, "adapter source"))
        require(digest == expected, f"adapter source SHA drift: {relative}")
        observed[relative] = digest
    return observed


def _validate_v2(payloads: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    draft = payloads["prelaunch_authorization_draft.json"]
    matrix = draft.get("matrix")
    cohort = draft.get("frozen_common_cohort")
    require(draft.get("status") == "NOT_AUTHORIZED_FOR_SCORING", "V2 draft status drift")
    require(isinstance(matrix, Mapping), "V2 matrix missing")
    require(
        matrix.get("N") == 15
        and matrix.get("arms") == ["shared_t4", "shared_ts4"]
        and matrix.get("views") == list(VIEWS)
        and matrix.get("seeds") == list(SEEDS)
        and matrix.get("sealed_cell_count") == 180,
        "V2 matrix drift",
    )
    require(isinstance(cohort, list) and len(cohort) == 15, "V2 cohort drift")
    query_counts = matrix.get("query_window_count_by_asset_id")
    require(
        isinstance(query_counts, Mapping)
        and sum(query_counts.values()) == EXPECTED_QUERY_WINDOWS_PER_VIEW,
        "V2 query-count map drift",
    )
    return [dict(row) for row in cohort], {str(k): int(v) for k, v in query_counts.items()}


def _validate_v5r2(payloads: Mapping[str, Any], root: Path) -> None:
    receipt = payloads["parity_execution_receipt.json"]
    trace = payloads["input_trace.json"]
    seal = payloads["seal.json"]
    require(
        str(receipt.get("status", "")).startswith("PARITY_CONFIRMED_CONSUMED_SUBC"),
        "V5R2 parity status drift",
    )
    require(receipt.get("external_subm_scoring_performed") is False, "V5R2 external score drift")
    bridge = trace.get("adapter_schema_bridge") or trace.get("input_trace", {}).get("adapter_schema_bridge")
    if bridge is None:
        bridge = trace.get("shared_observer", {}).get("adapter", {}).get("adapter_schema_bridge")
    # The exact existing receipt has the bridge below input_trace; retain a
    # recursive semantic check without modifying that sealed layout.
    def find_bridge(value: Any) -> Mapping[str, Any] | None:
        if isinstance(value, Mapping):
            if value.get("only_start_stop_cast") is True and "support_trials" in value:
                return value
            for child in value.values():
                found = find_bridge(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_bridge(child)
                if found is not None:
                    return found
        return None
    bridge = bridge if isinstance(bridge, Mapping) else find_bridge(trace)
    require(isinstance(bridge, Mapping), "V5R2 chronology bridge evidence missing")
    require(
        bridge.get("only_start_stop_cast") is True
        and bridge.get("bin_recomputation") is False
        and bridge.get("support_trials") == 50
        and bridge.get("identity_trials") == 30
        and bridge.get("query_valid_starts_changed_by_schema_bridge") is False,
        "V5R2 chronology semantics drift",
    )
    require(seal.get("external_subm_scoring_performed") is False, "V5R2 seal score drift")
    environment = payloads["environment.json"]
    source_pins = environment.get("source_pins")
    bridge_path = str(
        (root / "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py").resolve()
    )
    require(
        isinstance(source_pins, Mapping)
        and source_pins.get(bridge_path)
        == ADAPTER_SOURCE_PINS[
            "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py"
        ],
        "V5R2 adapter source binding drift",
    )


def _validate_zero4_parity(payloads: Mapping[str, Any]) -> None:
    receipt = payloads["receipt.json"]
    trace = payloads["input_parity_trace.json"]
    seal = payloads["seal.json"]
    require(
        receipt.get("status") == "PASS_CONSUMED_SUBC_SHARED_ZERO4_INPUT_PARITY_NO_MODEL_NO_R2",
        "zero4 parity status drift",
    )
    require(receipt.get("external_subm_accessed") is False, "zero4 parity sub-M access drift")
    require(receipt.get("external_subm_scored") is False, "zero4 parity score drift")
    proofs = receipt.get("proofs")
    required_true = (
        "activity_first_n30",
        "bitwise_float32_n_by_4_zero",
        "channel_count_only_construction",
        "label_drop_prediction_input_invariant",
        "label_shuffle_prediction_input_invariant",
        "owner_valid_starts_exact",
        "sua_and_pseudo_mua_shapes_valid",
        "t4_comparator_pool_and_query_boundary_50",
    )
    required_zero = (
        "descriptor_t4_fit_calls",
        "descriptor_t4_normalizer_value_reads",
        "descriptor_t4_trial_rate_reads",
        "descriptor_target_direction_reads",
    )
    require(
        isinstance(proofs, Mapping)
        and all(proofs.get(key) is True for key in required_true)
        and all(proofs.get(key) == 0 for key in required_zero),
        "zero4 parity proof drift",
    )
    descriptor = trace.get("zero4_descriptor_contract")
    require(
        isinstance(descriptor, Mapping)
        and descriptor.get("construction_input") == ["channel_count"]
        and descriptor.get("positive_zero_bits_only") is True
        and descriptor.get("target_direction_label_reads_for_descriptor") == 0
        and descriptor.get("t4_trial_rate_reads_for_descriptor") == 0
        and descriptor.get("target_t4_rate_fit_calls") == 0
        and descriptor.get("source_t4_normalizer_value_reads") == 0,
        "zero4 descriptor evidence drift",
    )
    dependencies = trace.get("dependencies")
    require(
        isinstance(dependencies, Mapping)
        and dependencies.get("sua_exploration/mc_maze/paired_view_c1_shared_zero4.py")
        == ADAPTER_SOURCE_PINS[
            "sua_exploration/mc_maze/paired_view_c1_shared_zero4.py"
        ]
        and dependencies.get("sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py")
        == ADAPTER_SOURCE_PINS[
            "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py"
        ],
        "zero4 parity adapter source binding drift",
    )
    require(seal.get("external_scoring_capability_created") is False, "zero4 parity capability drift")


def _verify_known_checkpoint_metadata(root: Path, slots: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = _regular_under(root, SCOPE_MANIFEST, "scope manifest")
    require(sha256_file(manifest_path) == SCOPE_MANIFEST_SHA256, "scope manifest SHA drift")
    manifest = _read_json(manifest_path, "scope manifest")
    indexed: dict[tuple[str, int], str] = {}
    for arm in ("shared_t4", "shared_ts4"):
        rows = manifest.get("matched_mechanism_arms", {}).get(arm)
        require(isinstance(rows, list) and len(rows) == 3, f"scope manifest {arm} rows drift")
        for row in rows:
            indexed[(arm, int(row["seed"]))] = str(row["terminal_checkpoint"]["sha256"])
    for slot in slots:
        key = (slot["arm"], slot["seed"])
        if key in indexed:
            require(slot["sha256"] == indexed[key], f"checkpoint metadata drift: {key}")
    return {
        "path": str(manifest_path),
        "sha256": SCOPE_MANIFEST_SHA256,
        "known_terminal_hashes_verified_without_checkpoint_open": True,
    }


def static_authority(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    v2_payloads, v2_pins = _artifact_bundle(root, V2_ROOT, V2_ARTIFACTS, "V2 score-only")
    v5_payloads, v5_pins = _artifact_bundle(root, V5R2_ROOT, V5R2_ARTIFACTS, "V5R2 parity")
    zero_payloads, zero_pins = _artifact_bundle(
        root, ZERO4_PARITY_ROOT, ZERO4_PARITY_ARTIFACTS, "zero4 parity"
    )
    cohort, query_counts = _validate_v2(v2_payloads)
    _validate_v5r2(v5_payloads, root)
    _validate_zero4_parity(zero_payloads)
    adapter_sources = _adapter_source_map(root)
    slots = validate_checkpoint_slots(checkpoint_slots_v4(), require_complete=False)
    manifest_pin = _verify_known_checkpoint_metadata(root, slots)
    contract = build_three_arm_contract(
        cohort=cohort, query_counts=query_counts, checkpoint_slots=slots
    )
    missing = missing_checkpoint_slots(slots)
    require(
        [(row["arm"], row["seed"]) for row in missing]
        == [("shared_zero4", 42), ("shared_zero4", 43), ("shared_zero4", 44)],
        "missing-terminal blocker drift",
    )
    parity_bundle = {"v5r2": v5_pins, "zero4_v1": zero_pins}
    source_map = _source_map(root)
    return {
        "contract": contract,
        "contract_sha256": contract["contract_sha256"],
        "cohort_sha256": contract["cohort_sha256"],
        "checkpoint_slots": slots,
        "checkpoint_slots_sha256": contract["checkpoint_slots_sha256"],
        "missing_checkpoint_slots": missing,
        "v2_score_only": v2_pins,
        "v5r2_and_zero4_parity": parity_bundle,
        "parity_bundle_sha256": canonical_sha256(parity_bundle),
        "scope_manifest": manifest_pin,
        "adapter_source_pins": adapter_sources,
        "source_map": source_map,
        "source_snapshot_sha256": canonical_sha256(source_map),
        "runtime_identity": runtime_identity(),
        "cpu_policy": CPU_POLICY,
        "public_key": None,
        "authorization_signature": None,
        "output_parent": None,
        "claim_root": None,
        "external_root_parent": None,
    }


def build_blocked_draft(root: Path = ROOT) -> dict[str, Any]:
    authority = static_authority(root)
    return {
        "schema_version": 4,
        "kind": "dandi_000688_subm_three_arm_cpu_score_prelaunch_blocked_draft_v4",
        "status": "BLOCKED_MISSING_ZERO4_TERMINALS",
        "append_only": True,
        "authority": authority,
        "matrix": {
            "N": EXPECTED_SESSION_COUNT,
            "views": list(VIEWS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "cell_count": EXPECTED_CELL_COUNT,
        },
        "comparison_gates": COMPARISON_GATES,
        "bootstrap_policy": BOOTSTRAP_POLICY,
        "future_authorization_contract": {
            "schema": "dandi_000688_subm_three_arm_cpu_score_authorization_v4",
            "envelope_schema": "dandi_000688_subm_three_arm_cpu_score_authorization_envelope_v4",
            "maximum_validity_seconds": 900,
            "fresh_256_bit_nonce": True,
            "authorization_and_nonce_claim_before_torch_checkpoint_nwb_import": True,
            "current_public_key": None,
            "current_signature": None,
            "current_execution_policy": None,
        },
        "atomic_output_contract": {
            "per_cell_exclusive_0444": True,
            "resume_only_from_semantically_valid_complete_cells": True,
            "partial_or_unknown_cell_fails_closed": True,
            "duplicate_cell_fails_closed": True,
            "aggregate_only_after_exactly_270_complete_cells": True,
        },
        "operations_by_this_package": {
            "external_subm_nwb_files_opened": 0,
            "checkpoint_files_opened": 0,
            "normalizer_files_opened": 0,
            "torch_imports": 0,
            "model_forward_calls": 0,
            "r2_computations": 0,
            "gpu_used": False,
            "signatures_created": 0,
            "external_scoring_capability_created": False,
        },
    }


def build_blocked_receipt(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_blocked_draft(root)
    receipt = {
        "schema_version": 4,
        "kind": "dandi_000688_subm_three_arm_cpu_score_blocked_receipt_v4",
        "status": "BLOCKED_MISSING_ZERO4_TERMINALS",
        "append_only": True,
        "draft": {
            "path": "blocked_prelaunch_draft.json",
            "sha256": hashlib.sha256(canonical_bytes(draft)).hexdigest(),
        },
        "missing_terminal_slots": draft["authority"]["missing_checkpoint_slots"],
        "matrix_cell_count": EXPECTED_CELL_COUNT,
        "operations": dict(draft["operations_by_this_package"]),
        "not_an_executable_prelaunch": True,
        "blocked_until": [
            "shared_zero4 seeds 42/43/44 terminal epoch_011 checkpoints complete",
            "their terminal hashes and closure manifests receive independent review",
            "a later append-only package binds all nine hashes and a real public key",
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
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, "immutable mode failed")
    return hashlib.sha256(raw).hexdigest()


def write_blocked_prelaunch(output_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), "three-arm V4 blocked root already exists")
    draft, receipt = build_blocked_receipt(root)
    require(receipt["status"] == "BLOCKED_MISSING_ZERO4_TERMINALS", "blocked status drift")
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path = output_dir / "blocked_prelaunch_draft.json"
    receipt_path = output_dir / "receipt.json"
    seal_path = output_dir / "blocked_seal.json"
    draft_sha = _write_immutable(draft_path, draft)
    require(receipt["draft"]["sha256"] == draft_sha, "draft/receipt binding drift")
    receipt_sha = _write_immutable(receipt_path, receipt)
    seal = {
        "schema_version": 4,
        "kind": "dandi_000688_subm_three_arm_cpu_score_blocked_seal_v4",
        "status": receipt["status"],
        "append_only": True,
        "artifacts": [
            {
                "path": draft_path.name,
                "sha256": draft_sha,
                "bytes": draft_path.stat().st_size,
                "mode": "0444",
            },
            {
                "path": receipt_path.name,
                "sha256": receipt_sha,
                "bytes": receipt_path.stat().st_size,
                "mode": "0444",
            },
        ],
        "executable_prelaunch_sealed": False,
        "external_scoring_capability_created": False,
    }
    seal_sha = _write_immutable(seal_path, seal)
    return {
        "output_dir": str(output_dir),
        "status": receipt["status"],
        "draft_sha256": draft_sha,
        "receipt_sha256": receipt_sha,
        "blocked_seal_sha256": seal_sha,
        "executable_prelaunch_sealed": False,
    }


def load_stored_blocked_prelaunch(
    prelaunch_dir: Path, root: Path = ROOT
) -> dict[str, Any]:
    prelaunch_dir = prelaunch_dir.resolve()
    paths = (
        prelaunch_dir / "blocked_prelaunch_draft.json",
        prelaunch_dir / "receipt.json",
        prelaunch_dir / "blocked_seal.json",
    )
    for path in paths:
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444,
            f"stored blocked artifact unsafe: {path.name}",
        )
    draft, receipt, seal = (
        _read_json(paths[0], "stored V4 blocked draft"),
        _read_json(paths[1], "stored V4 blocked receipt"),
        _read_json(paths[2], "stored V4 blocked seal"),
    )
    hashes = tuple(sha256_file(path) for path in paths)
    require(
        draft.get("status")
        == receipt.get("status")
        == seal.get("status")
        == "BLOCKED_MISSING_ZERO4_TERMINALS",
        "stored V4 blocked status drift",
    )
    require(draft.get("authority") == static_authority(root), "stored V4 authority drift")
    require(
        receipt.get("draft") == {"path": paths[0].name, "sha256": hashes[0]},
        "stored V4 draft/receipt binding drift",
    )
    require(
        seal.get("artifacts")
        == [
            {
                "path": paths[0].name,
                "sha256": hashes[0],
                "bytes": paths[0].stat().st_size,
                "mode": "0444",
            },
            {
                "path": paths[1].name,
                "sha256": hashes[1],
                "bytes": paths[1].stat().st_size,
                "mode": "0444",
            },
        ],
        "stored V4 blocked seal binding drift",
    )
    require(
        receipt.get("not_an_executable_prelaunch") is True
        and seal.get("executable_prelaunch_sealed") is False
        and seal.get("external_scoring_capability_created") is False,
        "stored V4 blocked package gained capability",
    )
    return {
        "status": receipt["status"],
        "draft_sha256": hashes[0],
        "receipt_sha256": hashes[1],
        "blocked_seal_sha256": hashes[2],
        "authority": draft["authority"],
        "executable_prelaunch_sealed": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = (
            static_authority(args.repo_root)
            if args.verify_only
            else write_blocked_prelaunch(args.output_dir, args.repo_root)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except StaticThreeArmV4Error as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
