from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import spec


FULL_SHA = "b8e5abcbe6cf7688fb5cb76c3bbd20b835b1e6332de1a59bf0ce2d692974f54a"
MB4_SHA = "152b76c149ae6c0cdc75c7320a3ba80863bac5f56abfaea87e58286262d6768a"
SEAL_SHA = "ef637fc64aaf317deb2111d2daaed2bc7c0aa00e0786a9bf09004b1d82ff2402"


def _synthetic_rows() -> list[dict[str, object]]:
    bindings = spec.compute_selection()["selection_preimage"]["candidate_fold_bindings"]
    sessions = {int(item["fold"]): item for item in bindings}
    rows: list[dict[str, object]] = []
    for seed in spec.EXPECTED_SEEDS:
        for fold in spec.SELECTED_FOLDS:
            for arm in (spec.FULL_ARM, spec.MB4_ARM):
                r2 = 0.25 + 0.01 * fold + (0.001 if arm == spec.FULL_ARM else 0.0)
                rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "arm": arm,
                        "target_session": sessions[fold]["target_session"],
                        "inner_validation_session": sessions[fold]["inner_validation_session"],
                        "r2_variance_weighted": r2,
                        "query_windows_evaluated": 100,
                        "selected_epoch": 10,
                        "selected_global_step": 1000,
                        "status": spec.EXPECTED_STATUS,
                        "model_state_unchanged": True,
                        "model_state_before_sha256": f"state-{seed}-{fold}",
                        "model_state_after_sha256": f"state-{seed}-{fold}",
                        "target_backpropagation": False,
                        "optimizer_present": False,
                        "source_split_manifest_sha256": f"split-{fold}",
                        "source_normalizer_sha256": f"norm-{fold}",
                        "checkpoint_sha256": f"checkpoint-{seed}-{fold}-{arm}",
                        "parameter_count": 1234,
                        "macs_per_decode_call": 5678,
                        "cached_state_bytes": 9012,
                        "initial_state_hash": f"init-{seed}-{fold}",
                    }
                )
    return rows


def test_immutable_anchor_hashes_and_selection_are_sealed() -> None:
    receipt = spec.build_draft_receipt()
    assert receipt["immutable_anchors"]["full_seed42"]["sha256"] == FULL_SHA
    assert receipt["immutable_anchors"]["mb4_seed42"]["sha256"] == MB4_SHA
    assert receipt["immutable_anchors"]["seed42_seal_marker"]["sha256"] == SEAL_SHA
    selection = receipt["selection"]
    assert selection["selection_preimage_sha256"] == "3e0463059d8a4042be4d3374c226178bfe40283d48823be8c9856b90f3de17ad"
    assert selection["selected_rank_order"] == [11, 3, 0, 12, 4]
    assert selection["selected_fold_ids"] == [0, 3, 4, 11, 12]
    assert len(selection["ranked_folds"]) == 15


def test_selection_preimage_contains_no_result_fields() -> None:
    preimage = spec.compute_selection()["selection_preimage"]
    encoded = json.dumps(preimage, sort_keys=True, separators=(",", ":")).lower()
    assert not any(term in encoded for term in ("r2", "score", "metric"))
    assert preimage["candidate_fold_ids"] == list(range(15))
    assert [int(item["fold"]) for item in preimage["candidate_fold_bindings"]] == list(range(15))


def test_draft_is_development_only_and_does_not_redefine_seed42_primary() -> None:
    receipt = spec.build_draft_receipt()
    assert receipt["status"] == "DRAFT_PREREGISTRATION_NOT_AUTHORIZED_NO_GPU"
    assert receipt["development_only"] is True
    assert receipt["formal_heldout_opened"] is False
    assert receipt["freeze"]["seed42_blinded"] is False
    assert receipt["freeze"]["seed43_44_outputs_known_at_freeze"] is False
    assert receipt["protocol"]["seeds"] == [43, 44]
    assert receipt["protocol"]["cells"] == 20
    assert receipt["resource_plan"]["no_gpu_launched_by_this_receipt"] is True


def test_complete_grid_aggregates_paired_nested_readout() -> None:
    result = spec.aggregate_cells(_synthetic_rows())
    assert result["status"] == "COMPLETE_DESCRIPTIVE_ONLY"
    assert result["label"] == "ROBUST_DIRECTIONAL"
    assert result["pair_count"] == 10
    assert result["cross_seed"]["both_seed_mean_positive"] is True
    assert result["cross_seed"]["each_seed_at_least_4_of_5_positive"] is True
    assert result["cross_seed"]["all_10_sign_counts"] == {"positive": 10, "negative": 0, "tie": 0}
    assert result["do_not_pool_seed_fold_as_independent"] is True
    assert result["inferential_significance_claim"] == "none"
    assert result["accounting"]["paired_equal_all_cells"] is True


def test_missing_or_failed_cells_fail_closed() -> None:
    rows = _synthetic_rows()
    with pytest.raises(spec.SpecError, match="incomplete annex grid"):
        spec.aggregate_cells(rows[:-1])
    rows[-1]["status"] = "FAILED"
    with pytest.raises(spec.SpecError, match="clean outer-evaluation pass"):
        spec.aggregate_cells(rows)


def test_pair_accounting_mismatch_fails_closed() -> None:
    rows = _synthetic_rows()
    rows[1]["macs_per_decode_call"] = 5679
    with pytest.raises(spec.SpecError, match="paired macs_per_decode_call mismatch"):
        spec.aggregate_cells(rows)


def test_fold_session_binding_mismatch_fails_closed() -> None:
    rows = _synthetic_rows()
    rows[0]["target_session"] = "wrong-target"
    with pytest.raises(spec.SpecError, match="fold/session binding mismatch"):
        spec.aggregate_cells(rows)


def test_runner_and_accounting_blockers_are_explicit() -> None:
    receipt = spec.build_draft_receipt()
    codes = {item["code"] for item in receipt["launch_blockers"]}
    assert "RUNNER_MISSING_AFC4_MB4_CLI" in codes
    assert "INITIAL_STATE_HASH_NOT_EXPOSED" in codes
    assert "ACCOUNTING_REQUIRED_AT_RUN_TIME" in codes
