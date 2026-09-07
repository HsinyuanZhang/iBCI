#!/usr/bin/env python3
"""Write the immutable, non-authorizing draft receipt for M2 post-33.

This writer is intentionally incapable of authorizing launch.  It records the
score-free data/plumbing closure and the prospective matrix/gates so root can
review them before a separate final receipt and authorization are created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
DEFAULT_AUDIT = SUA / "results/m2_native_t4_spint_post33_confirm_v1_scorefree_audit_20260804/audit.json"
DEFAULT_LIVE_PLUMBING = SUA / "results/m2_native_t4_spint_post33_confirm_v1_live_plumbing_20260804/live_plumbing.json"
DEFAULT_OUT = SUA / "results/m2_native_t4_spint_post33_confirm_v1_draft_prelaunch_v2_hash_hardened_20260804/draft_scorefree_receipt.json"
OLD_DRAFT = SUA / "results/m2_native_t4_spint_post33_confirm_v1_draft_prelaunch_20260804/draft_scorefree_receipt.json"
OLD_DRAFT_SHA256 = "dc6fd1be9a13e9a364dd359750cc4ffd7e1c3092505e8eb7c2120b93374feee5"
DRAFT_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1_DRAFT_V2_HASH_HARDENED"
OLD_PROTOCOL = SUA / "results/m2_heldin_postsupport_endpoint_v1/protocol_receipt.json"
OLD_PROTOCOL_SHA256 = "7673d360099775e37b199621956d099a9a2a7bafcbe7b074c3146978300f1c49"
BRIDGE_AUDIT = SUA / "docs/C1_TO_NATIVE_MUA_CLAIM_BRIDGE_AUDIT_20260804.md"

SOURCE_PATHS = (
    "SPINT-main/src/data/falcon_post33_confirm_v1_datamodule.py",
    "SPINT-main/src/data/falcon_datamodule.py",
    "SPINT-main/src/models/falcon_module.py",
    "SPINT-main/src/train.py",
    "SPINT-main/configs/train.yaml",
    "SPINT-main/configs/data/falcon_m2_post33_confirm_v1.yaml",
    "SPINT-main/configs/model/falcon_m2.yaml",
    "SPINT-main/configs/callbacks/post33_source_only.yaml",
    "SPINT-main/configs/experiment/m2_native_post33_confirm_v1_spint.yaml",
    "streaming_calibration_exp/src/data/falcon_post33_confirm_v1_datamodule.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "streaming_calibration_exp/src/data/falcon_t4_features.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/train.py",
    "streaming_calibration_exp/configs/train.yaml",
    "streaming_calibration_exp/configs/data/falcon_m2_post33_confirm_v1.yaml",
    "streaming_calibration_exp/configs/model/streaming_b3s_t4.yaml",
    "streaming_calibration_exp/configs/model/_streaming_base.yaml",
    "streaming_calibration_exp/configs/callbacks/post33_source_only.yaml",
    "streaming_calibration_exp/configs/experiment/m2_native_post33_confirm_v1_t4.yaml",
    "sua_exploration/mc_maze/m2_native_post33_confirm_v1.py",
    "sua_exploration/scripts/audit_m2_native_post33_confirm_v1_scorefree.py",
    "sua_exploration/scripts/audit_m2_native_post33_confirm_v1_live_plumbing.py",
    "sua_exploration/scripts/write_m2_native_post33_confirm_v1_draft_prelaunch.py",
    "sua_exploration/scripts/verify_m2_native_post33_confirm_v1_draft.py",
    "sua_exploration/tests/test_m2_native_post33_confirm_v1_plumbing.py",
    "sua_exploration/docs/M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1_DRAFT_PROTOCOL_20260804.md",
    "sua_exploration/docs/C1_TO_NATIVE_MUA_CLAIM_BRIDGE_AUDIT_20260804.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_map() -> dict[str, dict[str, Any]]:
    mapping = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        require(path.is_file(), f"missing draft source-map file: {relative}")
        mapping[relative] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return mapping


def load_contract():
    sys.path.insert(0, str(ROOT))
    from sua_exploration.mc_maze import m2_native_post33_confirm_v1 as contract

    return contract


def build(audit_path: Path, live_plumbing_path: Path = DEFAULT_LIVE_PLUMBING) -> dict[str, Any]:
    contract = load_contract()
    audit_path = audit_path.resolve()
    require(audit_path.is_file(), f"missing score-free audit {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("protocol_id") == contract.PROTOCOL_ID, "audit protocol ID mismatch")
    require(audit.get("status") == "PASS_SCORE_FREE_CPU_AUDIT", "score-free audit did not pass")
    scope = audit.get("execution_scope", {})
    require(
        scope
        == {
            "gpu_used": False,
            "training_started": False,
            "new_endpoint_r2_values_read": 0,
            "scorer_modules_imported": 0,
            "formal_sua_paths_resolved": 0,
            "formal_sua_files_opened": 0,
            "evalai_calls": 0,
        },
        "score-free execution scope mismatch",
    )
    require(audit["input_manifest"]["all_sha256_reverified_now"] is True, "full data SHA audit missing")
    require(audit["endpoint"]["totals"]["eligible_windows"] == 101_171, "window total mismatch")
    require(len(audit["folds"]) == 7, "seven-fold audit missing")
    c1_closure = audit["active_c1_v3r2_closure"]
    require(c1_closure["source_count"] == 34 and c1_closure["all_hashes_match"] is True, "C1 closure")
    for row in c1_closure["files"]:
        path = ROOT / row["path"]
        require(path.is_file() and sha256(path) == row["sha256"], f"active C1 drift: {row['path']}")
    require(sha256(OLD_PROTOCOL) == OLD_PROTOCOL_SHA256, "old v1 receipt drift")
    require(OLD_DRAFT.is_file() and sha256(OLD_DRAFT) == OLD_DRAFT_SHA256, "old draft drift")
    require(BRIDGE_AUDIT.is_file(), "bridge audit missing")
    live_plumbing_path = live_plumbing_path.resolve()
    require(live_plumbing_path.is_file(), f"missing live plumbing evidence {live_plumbing_path}")
    live = json.loads(live_plumbing_path.read_text(encoding="utf-8"))
    require(live.get("status") == "PASS_SCORE_FREE_LIVE_PLUMBING", "live plumbing did not pass")
    require(live.get("probe_count") == 14, "live plumbing must cover both sides and seven folds")
    require(
        live.get("eligible_window_totals") == {"spint": 101_171, "t4": 101_171},
        "live plumbing window totals mismatch",
    )
    matrix = contract.matrix_contract()
    require(matrix["full_terminal_arm_cells"] == 42, "full matrix must contain 42 arm-cells")
    require(matrix["stage_a"]["terminal_arm_cells"] == 14, "Stage A must contain 14 arm-cells")
    require(matrix["stage_b"]["additional_terminal_arm_cells"] == 28, "Stage B must contain 28 cells")

    return {
        "schema_version": 2,
        "protocol_id": contract.PROTOCOL_ID,
        "draft_id": DRAFT_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "DRAFT_SCORE_FREE_PRELAUNCH_V2_NOT_AUTHORIZED",
        "authorization": {
            "gpu_launch_authorized": False,
            "final_launch_receipt_created": False,
            "root_review_required": True,
            "this_receipt_cannot_be_promoted_in_place": True,
        },
        "execution_scope": scope,
        "scientific_scope": {
            "endpoint": "native_M2_outer_LOSO_chronological_post33_future_query",
            "claim_class": "prospective_internal_confirmation_not_external_hidden_test",
            "estimand": "supervised_T4_calibration_package_utility_minus_clean_local_SPINT",
            "independent_of_c1_verdict": True,
        },
        "operational_draft_supersession": {
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
        "supersession": {
            "path": str(OLD_PROTOCOL.relative_to(ROOT)),
            "sha256": OLD_PROTOCOL_SHA256,
            "file_modified": False,
            "scope_only": (
                "supersedes old v1 effectiveness use and target-session checkpoint selection; "
                "does not overwrite historical receipt"
            ),
        },
        "score_free_audit": {
            "path": str(audit_path.relative_to(ROOT)),
            "sha256": sha256(audit_path),
            "data_sha256_reverified": True,
            "eligible_windows": 101_171,
        },
        "live_plumbing": {
            "path": str(live_plumbing_path.relative_to(ROOT)),
            "sha256": sha256(live_plumbing_path),
            "probe_count": 14,
            "eligible_window_totals": {"spint": 101_171, "t4": 101_171},
            "gpu_used": False,
            "training_started": False,
            "new_endpoint_r2_values_read": 0,
        },
        "bridge_audit": {
            "path": str(BRIDGE_AUDIT.relative_to(ROOT)),
            "sha256": sha256(BRIDGE_AUDIT),
        },
        "active_c1_v3r2_noninterference": {
            "receipt": audit["active_c1_v3r2_closure"]["receipt"],
            "receipt_sha256": audit["active_c1_v3r2_closure"]["receipt_sha256"],
            "source_count": 34,
            "all_hashes_match_at_draft_write": True,
            "files_edited_by_post33_program": 0,
        },
        "endpoint_contract": {
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
        "folds": audit["folds"],
        "input_manifest": audit["input_manifest"],
        "input_files": audit["input_files"],
        "per_session_endpoint": audit["endpoint"],
        "matrix": matrix,
        "training_and_selection": {
            "spint": {
                "trainings": 21,
                "max_epochs": 35,
                "selection_metric": contract.SOURCE_SELECTION_METRIC,
                "selection_scope": "six_outer_train_source_sessions_only",
                "selection_mode": "max",
                "tie_break": "earlier_zero_based_epoch",
                "target_query_selection_windows": 0,
            },
            "t4": {
                "trainings": 21,
                "max_epochs": 12,
                "selection_metric": contract.SOURCE_SELECTION_METRIC,
                "selection_scope": "six_outer_train_source_sessions_only",
                "selection_mode": "max",
                "tie_break": "earlier_zero_based_epoch",
                "target_query_selection_windows": 0,
                "decoder_source": "paired_fold_seed_SPINT_selected_checkpoint_only",
            },
        },
        "label_information_disclosure": {
            "spint": "same first-33 neural support trials; zero target-direction labels",
            "t4": (
                "same first-33 neural support trials plus labels for eligible directional trials; "
                "normally 16 directional labels and 17 centre/rest exclusions"
            ),
            "matched_neural_exposure": True,
            "matched_label_information": False,
            "claim": "supervised_calibration_package_utility_not_equal_information_ablation",
        },
        "decoder_contract": {
            "tensor_count_expected": 31,
            "tensor_count_compared": 31,
            "bit_exact_required": True,
            "frozen_during_t4_source_training": True,
            "requires_grad_tensor_count_required": 0,
            "updated_tensor_count_required": 0,
            "mismatch_action": "fail_closed_no_score",
        },
        "target_calibration_contract": {
            "fit": "closed_form_rank3_cosine_[a,c,m,b]",
            "optimizer_steps": 0,
            "backward_calls": 0,
            "updated_parameter_tensors": 0,
            "query_labels_or_rates_used_for_fit": 0,
            "state": "cached_four_float_descriptor_per_native_channel_plus_frozen_activity_state",
        },
        "stage_a_futility": matrix["stage_a"],
        "stage_b": matrix["stage_b"],
        "full_effectiveness_gates": contract.effectiveness_gate_contract(),
        "no_rescue": {
            "retry": False,
            "seed_replacement": False,
            "extra_arm": False,
            "hyperparameter_or_epoch_scan": False,
            "m1_or_m24_fallback": False,
            "decoder_unfreeze": False,
            "quantization_rescue": False,
            "formal_sua": False,
            "evalai": False,
        },
        "future_runtime_closure_required": {
            "per_cell": [
                "receipt/source/config/data/checkpoint/normalizer hashes",
                "outer train/normalizer/checkpoint-selection counts all zero",
                "support label-rate alignment and rank-3 design",
                "minimum full-window start at or after trial-33 boundary",
                "target optimizer/backward/update counts all zero",
                "31/31 decoder bit equality and frozen state",
                "parameter/MAC/state/latency receipt",
            ],
            "stage_a_finalizer": "exactly_14_terminal_cells_then_joint_read_only",
            "full_finalizer": "exactly_42_terminal_cells_then_single_write_once_aggregate",
        },
        "deliberately_missing_before_final_authorization": [
            "production runner and scheduler",
            "per-cell closure writer/verifier",
            "Stage-A and full score-reading finalizers",
            "cost profiler receipt",
            "independent root launch authorization",
        ],
        "source_map": source_map(),
    }


def write(out: Path, audit: Path, live_plumbing: Path = DEFAULT_LIVE_PLUMBING) -> Path:
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build(audit, live_plumbing)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    checksum = out.with_suffix(out.suffix + ".sha256")
    with checksum.open("x", encoding="utf-8") as handle:
        handle.write(f"{sha256(out)}  {out.name}\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--live-plumbing", type=Path, default=DEFAULT_LIVE_PLUMBING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    path = write(args.out, args.audit, args.live_plumbing)
    print(path)
    print(sha256(path))


if __name__ == "__main__":
    main()
