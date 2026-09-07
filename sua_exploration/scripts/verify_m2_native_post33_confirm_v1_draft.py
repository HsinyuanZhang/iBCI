#!/usr/bin/env python3
"""Fail-closed verifier for the non-authorizing M2 post-33 draft receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_draft_prelaunch_v2_hash_hardened_20260804/draft_scorefree_receipt.json"
OLD_DRAFT = ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_draft_prelaunch_20260804/draft_scorefree_receipt.json"
OLD_DRAFT_SHA256 = "dc6fd1be9a13e9a364dd359750cc4ffd7e1c3092505e8eb7c2120b93374feee5"
DRAFT_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1_DRAFT_V2_HASH_HARDENED"

EXPECTED_SCORE_FREE_SCOPE = {
    "gpu_used": False,
    "training_started": False,
    "new_endpoint_r2_values_read": 0,
    "scorer_modules_imported": 0,
    "formal_sua_paths_resolved": 0,
    "formal_sua_files_opened": 0,
    "evalai_calls": 0,
}
EXPECTED_LIVE_SCOPE = {
    "gpu_used": False,
    "training_started": False,
    "optimizer_steps": 0,
    "new_endpoint_r2_values_read": 0,
    "scorer_modules_imported": 0,
    "formal_sua_paths_resolved": 0,
    "evalai_calls": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_pinned_file(row: dict[str, Any]) -> Path:
    """Re-hash one currently present NWB and compare it with its pinned digest.

    Byte count alone is deliberately insufficient: a same-size content change
    must fail this helper and therefore the full draft verifier.
    """

    required = {"role", "session", "path", "size_bytes", "sha256", "sha256_reverified_now"}
    missing = sorted(required.difference(row))
    require(not missing, f"pinned input row missing fields: {missing}")
    require(row["role"] in {"heldin_calib", "heldin_minival"}, f"invalid input role {row['role']!r}")
    require(row["sha256_reverified_now"] is True, f"input was not SHA-reverified: {row['path']}")
    path = Path(row["path"]).resolve()
    require("held-out" not in str(path).lower(), f"forbidden held-out input {path}")
    require(path.is_file(), f"missing pinned input {path}")
    require(path.stat().st_size == row["size_bytes"], f"input byte-count drift {path}")
    pinned = row["sha256"]
    require(
        isinstance(pinned, str)
        and len(pinned) == 64
        and all(character in "0123456789abcdef" for character in pinned),
        f"invalid pinned SHA-256 {path}",
    )
    observed = sha256(path)
    require(observed == pinned, f"input SHA-256 drift {path}: expected {pinned}, found {observed}")
    return path


def verify(receipt_path: Path) -> dict[str, Any]:
    receipt_path = receipt_path.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt["protocol_id"] == "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1", "protocol ID")
    require(receipt["schema_version"] == 2 and receipt["draft_id"] == DRAFT_ID, "draft revision")
    require(receipt["status"] == "DRAFT_SCORE_FREE_PRELAUNCH_V2_NOT_AUTHORIZED", "draft status")
    require(
        receipt["authorization"]
        == {
            "gpu_launch_authorized": False,
            "final_launch_receipt_created": False,
            "root_review_required": True,
            "this_receipt_cannot_be_promoted_in_place": True,
        },
        "draft authorization must remain false",
    )
    require(OLD_DRAFT.is_file() and sha256(OLD_DRAFT) == OLD_DRAFT_SHA256, "superseded draft drift")
    require(
        receipt["operational_draft_supersession"]
        == {
            "supersedes_path": str(OLD_DRAFT.relative_to(ROOT)),
            "supersedes_sha256": OLD_DRAFT_SHA256,
            "old_draft_modified": False,
            "scientific_protocol_changed": False,
            "data_matrix_metric_or_gate_changed": False,
            "reason": (
                "verifier hardened to recompute all fourteen current NWB SHA-256 values, "
                "reject same-size content mutation, and validate complete audit/live zero-access scopes"
            ),
        },
        "operational draft supersession",
    )
    scope = receipt["execution_scope"]
    require(scope == EXPECTED_SCORE_FREE_SCOPE, "draft score-free execution scope drift")

    endpoint = receipt["endpoint_contract"]
    require(
        endpoint
        == {
            "task": "m2",
            "validation_protocol": "loso",
            "calibration_n_trials": 33,
            "heldin_query_start_trial": 33,
            "heldin_query_end_trial": None,
            "query_start_trial": 0,
            "random_calibration": False,
            "include_heldout_in_fit": False,
            "include_heldout_in_test": False,
            "window_size": 50,
            "query_source": "unique_outer_left_out_train_calib_heldin_session",
            "full_history_rule": "minimum_50_bin_window_start_at_or_after_trial33_boundary",
        },
        "endpoint contract drift",
    )
    folds = receipt["folds"]
    require(len(folds) == 7, "fold count")
    outer = []
    for fold in folds.values():
        require(len(fold["source_train_sessions"]) == 6, "source train count")
        require(fold["source_train_sessions"] == fold["source_normalizer_sessions"], "normalizer scope")
        require(
            fold["source_train_sessions"] == fold["source_checkpoint_selection_sessions"],
            "selection scope",
        )
        require(fold["outer_counts"] == {"train": 0, "normalizer": 0, "checkpoint_selection": 0, "post33_query": 1}, "outer role counts")
        require(fold["post33_query_sessions"] == [fold["outer_left_out_session"]], "unique query source")
        outer.append(fold["outer_left_out_session"])
    require(len(set(outer)) == 7, "outer folds are not unique")

    matrix = receipt["matrix"]
    require(matrix["arms"] == ["spint", "t4"] and matrix["seeds"] == [42, 43, 44], "matrix arms/seeds")
    require(matrix["full_terminal_arm_cells"] == 42, "full cell count")
    require(matrix["full_trainings"] == {"spint": 21, "t4": 21}, "training count")
    require(matrix["stage_a"]["terminal_arm_cells"] == 14, "Stage A count")
    require(matrix["stage_a"]["futility_rule"] == "(mean42 <= -0.03) OR (pos42 <= 1)", "futility rule")
    require(matrix["stage_b"]["additional_terminal_arm_cells"] == 28, "Stage B count")
    require(len(receipt["full_effectiveness_gates"]) == 6, "six effectiveness gates")

    decoder = receipt["decoder_contract"]
    require(decoder["tensor_count_expected"] == decoder["tensor_count_compared"] == 31, "decoder tensor count")
    require(decoder["bit_exact_required"] is True and decoder["frozen_during_t4_source_training"] is True, "decoder equality/freeze")
    target = receipt["target_calibration_contract"]
    require((target["optimizer_steps"], target["backward_calls"], target["updated_parameter_tensors"]) == (0, 0, 0), "target calibration updates")
    disclosure = receipt["label_information_disclosure"]
    require(disclosure["matched_neural_exposure"] is True, "neural exposure disclosure")
    require(disclosure["matched_label_information"] is False, "label asymmetry disclosure")
    require(all(value is False for value in receipt["no_rescue"].values()), "no-rescue contract")

    audit = ROOT / receipt["score_free_audit"]["path"]
    require(audit.is_file() and sha256(audit) == receipt["score_free_audit"]["sha256"], "audit binding")
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    require(audit_payload.get("status") == "PASS_SCORE_FREE_CPU_AUDIT", "audit status")
    require(audit_payload.get("execution_scope") == EXPECTED_SCORE_FREE_SCOPE, "audit execution scope")
    require(
        audit_payload.get("input_manifest", {}).get("role_counts")
        == {"heldin_calib": 7, "heldin_minival": 7},
        "audit role counts",
    )
    require(
        audit_payload.get("input_manifest", {}).get("all_sha256_reverified_now") is True,
        "audit did not fully reverify SHA-256",
    )
    require(receipt["score_free_audit"]["eligible_windows"] == 101_171, "eligible windows")
    live = ROOT / receipt["live_plumbing"]["path"]
    require(live.is_file() and sha256(live) == receipt["live_plumbing"]["sha256"], "live plumbing binding")
    live_payload = json.loads(live.read_text(encoding="utf-8"))
    require(live_payload.get("status") == "PASS_SCORE_FREE_LIVE_PLUMBING", "live plumbing status")
    require(live_payload.get("execution_scope") == EXPECTED_LIVE_SCOPE, "live plumbing execution scope")
    require(receipt["live_plumbing"]["probe_count"] == 14, "live plumbing probe count")
    require(
        receipt["live_plumbing"]["eligible_window_totals"]
        == {"spint": 101_171, "t4": 101_171},
        "live plumbing totals",
    )
    require(receipt["input_files"] == audit_payload.get("input_files"), "receipt/audit input rows differ")
    role_counts = {"heldin_calib": 0, "heldin_minival": 0}
    sessions_by_role = {"heldin_calib": set(), "heldin_minival": set()}
    for row in receipt["input_files"]:
        verify_pinned_file(row)
        role = row["role"]
        role_counts[role] += 1
        sessions_by_role[role].add(row["session"])
    require(role_counts == {"heldin_calib": 7, "heldin_minival": 7}, "input role counts")
    require(
        len(sessions_by_role["heldin_calib"]) == len(sessions_by_role["heldin_minival"]) == 7,
        "input sessions are not unique within role",
    )
    require(
        sessions_by_role["heldin_calib"] == sessions_by_role["heldin_minival"],
        "calibration/minival session sets differ",
    )

    for relative, frozen in receipt["source_map"].items():
        path = ROOT / relative
        require(path.is_file(), f"source missing: {relative}")
        require(path.stat().st_size == frozen["size_bytes"], f"source size drift: {relative}")
        require(sha256(path) == frozen["sha256"], f"source SHA drift: {relative}")

    checksum_path = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    if checksum_path.is_file():
        require(checksum_path.read_text().split()[0] == sha256(receipt_path), "receipt checksum")
    return {
        "status": "PASS_DRAFT_V2_HASH_HARDENED_NOT_AUTHORIZED",
        "draft_id": receipt["draft_id"],
        "protocol_id": receipt["protocol_id"],
        "source_count": len(receipt["source_map"]),
        "fold_count": len(folds),
        "terminal_cell_count": matrix["full_terminal_arm_cells"],
        "eligible_windows": receipt["score_free_audit"]["eligible_windows"],
        "gpu_launch_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
