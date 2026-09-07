from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_t4_m30_mainline_preflight_v2.py"
SPEC = importlib.util.spec_from_file_location("t4_m30_preflight_v2", SCRIPT)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_m30_partition_has_one_shared_support_and_feature_fit_boundary() -> None:
    original = [100 + 3 * value for value in range(37)]
    partition = AUDIT.m30_partition(original)

    assert partition["activity_forward_support"]["original_trial_indices"] == original[:30]
    assert partition["t4_label_rate_fit"]["original_trial_indices"] == original[:30]
    assert partition["scored"]["original_trial_indices"] == original[30:]
    assert partition["support_feature_fit_equal"] is True
    assert partition["t4_fit_scored_overlap_original_trial_indices"] == []


def test_v2_binds_parameterized_pool_not_generic_default() -> None:
    evidence = AUDIT.source_and_launch_evidence()

    assert all(evidence["checked_contracts"].values())
    assert evidence["reconstructed_semantics"]["scored"] == "usable trials[30:end]"
    assert evidence["historical_source_byte_provenance"]["status"] == "not_persisted_in_historical_m30_artifacts"


def test_v2_binds_full_live_reference_matrix_without_claiming_ph4() -> None:
    identity = AUDIT.bind_reused_artifacts()

    assert len(identity["artifacts"]) == 9
    assert {row["feature_group"] for row in identity["artifacts"]} == {"none", "t4", "ts4"}
    assert all(row["declared_score_start"] == 30 for row in identity["artifacts"])
