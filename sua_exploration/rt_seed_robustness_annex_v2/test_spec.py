from __future__ import annotations

import json

import pytest

from . import spec


def _synthetic_rows() -> list[dict[str, object]]:
    bindings = {int(item["fold"]): item for item in spec.FOLD_BINDINGS}
    rows: list[dict[str, object]] = []
    for seed in spec.EXPECTED_SEEDS:
        for fold in spec.SELECTED_FOLDS:
            for arm in (spec.FULL_ARM, spec.MB4_ARM):
                rows.append(
                    {
                        "seed": seed, "fold": fold, "arm": arm,
                        "target_session": bindings[fold]["target_session"],
                        "inner_validation_session": bindings[fold]["inner_validation_session"],
                        "r2_variance_weighted": 0.25 + 0.01 * fold + (0.001 if arm == spec.FULL_ARM else 0.0),
                        "query_windows_evaluated": 100, "selected_epoch": 10, "selected_global_step": 1000,
                        "status": spec.EXPECTED_STATUS,
                        "target_model_state_unchanged": True,
                        "target_model_state_before_sha256": f"state-{seed}-{fold}",
                        "target_model_state_after_sha256": f"state-{seed}-{fold}",
                        "target_backpropagation": False, "target_optimizer_present": False,
                        # Raw hashes are arm-specific metadata and may differ;
                        # the frozen projections are the paired invariants.
                        "source_split_manifest_sha256": f"split-{seed}-{fold}-{arm}",
                        "source_normalizer_sha256": f"norm-{seed}-{fold}-{arm}",
                        "source_split_scope_sha256": f"scope-{fold}",
                        "source_normalizer_numeric_sha256": f"numeric-{fold}",
                        "checkpoint_sha256": f"checkpoint-{seed}-{fold}-{arm}",
                        "parameter_count": 1234, "macs_per_decode_call": 5678, "cached_state_bytes": 9012,
                        "initial_state_hash": f"init-{seed}-{fold}",
                        "implementation_snapshot_sha256": "a" * 64,
                        "paired_initial_state_equal_before_target_eval": True,
                        "paired_initial_state_receipt": f"/external/pair-{seed}-{fold}.json",
                        "target_forward_only_accounting": {
                            "parameter_count": 1234,
                            "macs_per_decode_call": 5678,
                            "cached_state_bytes": 9012,
                            "paired_full_mb4_equal_before_target_eval": True,
                        },
                    }
                )
    return rows


def test_selection_is_fixed_and_score_independent() -> None:
    selection = spec.compute_selection()
    assert selection["selected_fold_ids"] == [0, 3, 6, 9, 12]
    assert selection["method"] == "performance_independent_fold_id_modulo_v2"
    assert selection["score_bearing_anchor_hashes_used"] is False
    encoded = json.dumps(selection["selection_preimage"], sort_keys=True, separators=(",", ":")).lower()
    assert not any(term in encoded for term in ("r2", "score", "metric", "sha256"))


def test_draft_renames_target_evaluation_fields_and_supersedes_v1() -> None:
    receipt = spec.build_draft_receipt()
    assert receipt["schema"] == "rt_seed_robustness_annex_v2"
    assert receipt["supersedes"] == "rt_seed_robustness_annex_v1"
    required = set(receipt["per_cell_required_fields"])
    assert "target_model_state_unchanged" in required
    assert "target_optimizer_present" in required
    assert "model_state_unchanged" not in required
    assert "optimizer_present" not in required
    assert receipt["protocol"]["source_training_optimizer"].startswith("present")
    assert receipt["resource_plan"]["no_gpu_launched_by_this_receipt"] is True


def test_complete_grid_aggregates_paired_readout() -> None:
    result = spec.aggregate_cells(_synthetic_rows())
    assert result["status"] == "COMPLETE_DESCRIPTIVE_ONLY"
    assert result["label"] == "ROBUST_DIRECTIONAL"
    assert result["pair_count"] == 10
    assert result["cross_seed"]["all_10_sign_counts"] == {"positive": 10, "negative": 0, "tie": 0}
    assert result["accounting"]["target_optimizer_absent_all_cells"] is True


def test_raw_hashes_may_differ_when_scope_projections_match() -> None:
    rows = _synthetic_rows()
    raw_split = {row["source_split_manifest_sha256"] for row in rows}
    raw_norm = {row["source_normalizer_sha256"] for row in rows}
    assert len(raw_split) > 1 and len(raw_norm) > 1
    result = spec.aggregate_cells(rows)
    assert result["accounting"]["paired_scope_projection_equal_all_cells"] is True
    assert result["accounting"]["paired_numeric_normalizer_equal_all_cells"] is True


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("source_split_scope_sha256", "paired source_split_scope_sha256 mismatch"),
        ("source_normalizer_numeric_sha256", "paired source_normalizer_numeric_sha256 mismatch"),
    ],
)
def test_projection_drift_fails_closed(field: str, message: str) -> None:
    rows = _synthetic_rows()
    rows[0][field] = "drift"
    with pytest.raises(spec.SpecError, match=message):
        spec.aggregate_cells(rows)


def test_projection_hashes_exclude_only_arm_metadata_and_detect_scope_drift() -> None:
    binding = next(item for item in spec.FOLD_BINDINGS if int(item["fold"]) == 0)
    manifest = {
        field: (binding[field] if field in binding else {"fold": 0})
        for field in spec.SOURCE_SPLIT_SCOPE_FIELDS
    }
    manifest["outer_loso_fold"] = 0
    manifest["loso_fold"] = 0
    manifest["arm"] = "afc4_vel"
    other = dict(manifest, arm="afc4_mb4", requested_side_feature_group="afc4_mb4")
    assert spec.source_split_scope_sha256(manifest) == spec.source_split_scope_sha256(other)
    drift = dict(other, source_sessions=["different-session"])
    assert spec.source_split_scope_sha256(drift) != spec.source_split_scope_sha256(manifest)


def test_normalizer_projection_excludes_feature_group_but_detects_numeric_drift() -> None:
    normalizer = {
        "fit_scope": "inner_train_sessions_only",
        "fit_sessions": ["s1", "s2"],
        "excluded_inner_validation_session": "s3",
        "excluded_outer_target_session": "s4",
        "mean": [0.1, 0.2],
        "std": [1.0, 1.1],
        "feature_group": "afc4_vel",
    }
    other = dict(normalizer, feature_group="afc4_mb4")
    assert spec.source_normalizer_numeric_sha256(normalizer) == spec.source_normalizer_numeric_sha256(other)
    drift = dict(other, std=[1.0, 1.2])
    assert spec.source_normalizer_numeric_sha256(drift) != spec.source_normalizer_numeric_sha256(normalizer)


def test_legacy_ambiguous_target_fields_fail_closed() -> None:
    rows = _synthetic_rows()
    rows[0].pop("target_model_state_unchanged")
    rows[0].pop("target_optimizer_present")
    rows[0]["model_state_unchanged"] = True
    rows[0]["optimizer_present"] = False
    with pytest.raises(spec.SpecError, match="missing required fields"):
        spec.aggregate_cells(rows)


def test_missing_cell_fails_closed() -> None:
    with pytest.raises(spec.SpecError, match="incomplete annex grid"):
        spec.aggregate_cells(_synthetic_rows()[:-1])
