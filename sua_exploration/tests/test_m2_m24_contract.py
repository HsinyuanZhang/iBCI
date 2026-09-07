"""Static fail-closed contracts for the M24 disjoint held-out source program."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_m24_runner_requires_corrected_v2_receipt_and_full_gate() -> None:
    runner = (ROOT / "sua_exploration/scripts/run_m2_m24_internal_one_arm.sh").read_text()
    assert "audit_m2_m24_heldin_v2.json" in runner
    assert "audit_m2_m24_correction_receipt_v1.json" in runner
    assert "mean_kreg_over_baseline_ratio" in runner
    assert "mean_kreg_over_w_only_null_median_ratio" in runner
    assert "min(s.get(\"W_A_B_correlations\", [0.0])) > .5" in runner
    assert "audit_m2_m24_heldin_v1.json\"" not in runner


def test_m24_aggregate_rejects_preprocess_drift_and_requires_four_arms() -> None:
    aggregate = (ROOT / "sua_exploration/scripts/aggregate_m2_m24_internal.py").read_text()
    assert 'GROUPS = {"f0"' in aggregate
    assert "preprocessing parity drift against T4" in aggregate
    assert 'estimator.get("calibration_trials") != M' in aggregate
    assert '"K4_minus_T4"' in aggregate
    assert '"K4_minus_KS4"' in aggregate
    assert '"K4_minus_F0"' in aggregate


def test_m24_configs_are_fixed_to_first24_heldin_only_sources() -> None:
    config_dir = ROOT / "streaming_calibration_exp/configs/experiment"
    for filename in (
        "b3_native_mua_f0_m2_m24_loso_internal.yaml",
        "b3s_t4_m2_m24_loso_internal.yaml",
        "b3s_k4_m2_m24_loso_internal.yaml",
        "b3s_ks4_m2_m24_loso_internal.yaml",
    ):
        config = (config_dir / filename).read_text()
        assert "calibration_n_trials: 24" in config or filename.endswith("ks4_m2_m24_loso_internal.yaml")
        assert "include_heldout_in_fit: false" in config
        assert "include_heldout_in_test: false" in config
        assert "query_start_trial: 0" in config or filename.endswith("ks4_m2_m24_loso_internal.yaml")


def test_m24_heldout_runner_is_test_only_and_verifies_source_content() -> None:
    runner = (ROOT / "sua_exploration/scripts/run_m2_m24_heldout_one_arm.sh").read_text()
    assert 'train=false test=true optimized_metric=null ckpt_path="$SOURCE_CKPT"' in runner
    assert "data.include_heldout_in_fit=false data.include_heldout_in_test=true data.query_start_trial=24" in runner
    assert "source checkpoint SHA drift before held-out launch" in runner
    assert "internal M24 source gate did not pass; held-out replay is prohibited" in runner
    assert "write_m2_m24_heldout_provenance.py" in runner


def test_m24_heldout_receipts_protect_every_arm_not_only_k4() -> None:
    provenance = (ROOT / "sua_exploration/scripts/write_m2_m24_heldout_provenance.py").read_text()
    aggregate = (ROOT / "sua_exploration/scripts/aggregate_m2_m24_heldout.py").read_text()
    for text in (provenance, aggregate):
        assert '"train": (cfg.get("train"), False)' in text or '"train": False' in text
        assert '"test": (cfg.get("test"), True)' in text or '"test": True' in text
        assert '"data.query_start_trial": (data.get("query_start_trial"), M)' in text or '"data.query_start_trial": M' in text
        assert "hidden_evalai_evaluated" in text
        assert "full temporal history overlaps support" in text or "temporal-history boundary is not exact" in text
    assert "wilcoxon(values, alternative=\"two-sided\", method=\"exact\")" in aggregate
    assert "K4_minus_T4" in aggregate and "K4_minus_KS4" in aggregate and "K4_minus_F0" in aggregate
    assert '"block_width_bins": 5' in provenance and '"behavior_lead_bins": 2' in provenance
    assert '"block_width_bins": 5' in aggregate and '"behavior_lead_bins": 2' in aggregate
    assert "held-out data mapping drifts from frozen source beyond the two permitted query overrides" in aggregate
    assert "held-out model mapping drifts from frozen source" in aggregate
    assert "held-out query-window layout drifts from F0" in aggregate
    assert '"raw_query_start_bin"' in aggregate and '"minimum_window_start_padded_bin"' in aggregate
