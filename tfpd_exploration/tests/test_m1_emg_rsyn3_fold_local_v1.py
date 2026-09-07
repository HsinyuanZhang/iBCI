"""CPU tests for the fold-local EMG-rSyn3 Stage-0 audit successor.

Sealed EMG-Syn3 FAIL and EMG-rSyn3 V1 PASS roots are read-only evidence.
No live GPU capability.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import syn3 as fold_syn3
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3


ROOT = Path(__file__).resolve().parents[2]
STAGE0_CLI = ROOT / "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_stage0.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
PARENT_FAIL = ROOT / "tfpd_exploration/results/m1_emg_syn3_fcm_v1/stage0"
RSYN3_PASS = ROOT / "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/stage0"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sealed_fail_and_rsyn3_pass_roots_remain_byte_identical() -> None:
    assert _sha(PARENT_FAIL / "decision.json") == plan.PARENT_FAIL_DECISION_SHA256
    assert _sha(PARENT_FAIL / "terminal.json") == plan.PARENT_FAIL_TERMINAL_SHA256
    assert _sha(RSYN3_PASS / "decision.json") == plan.RSYN3_PASS_DECISION_SHA256
    assert _sha(RSYN3_PASS / "terminal.json") == plan.RSYN3_PASS_TERMINAL_SHA256
    assert _sha(RSYN3_PASS / "fold_receipts.json") == plan.RSYN3_PASS_FOLD_RECEIPTS_SHA256
    assert _sha(RSYN3_PASS / "reliability_table.json") == plan.RSYN3_PASS_RELIABILITY_SHA256
    fail = json.loads((PARENT_FAIL / "decision.json").read_text())
    assert fail["decision"] == "FAIL"
    passed = json.loads((RSYN3_PASS / "decision.json").read_text())
    assert passed["decision"] == "PASS"
    assert oct((PARENT_FAIL / "decision.json").stat().st_mode & 0o777) == "0o444"
    assert oct((RSYN3_PASS / "decision.json").stat().st_mode & 0o777) == "0o444"


def test_successor_refuses_sealed_result_roots(tmp_path: Path) -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import receipts as fold_receipts

    with pytest.raises(Exception, match="sealed|parent|rsyn3_fcm_v1"):
        fold_receipts.refuse_sealed_roots(ROOT / "tfpd_exploration/results/m1_emg_syn3_fcm_v1")
    with pytest.raises(Exception, match="sealed|parent|rsyn3_fcm_v1"):
        fold_receipts.refuse_sealed_roots(ROOT / "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1")
    fold_receipts.refuse_sealed_roots(tmp_path / "m1_emg_rsyn3_fold_local_v1")


def test_relu_law_is_exactly_float64_maximum() -> None:
    assert plan.RECTIFIER_LAW["operator"] == "np.maximum(x_raw, 0.0)"
    assert plan.RECTIFIER_LAW["dtype"] == "float64"
    assert plan.RECTIFIER_LAW["threshold"] == 0.0
    assert plan.RECTIFIER_LAW["sweep"] == "forbidden"
    raw = np.array([[-0.11, 0.0, 1.2]], dtype=np.float32)
    out = rectify.relu_nonnegative_projection(raw)
    assert out.dtype == np.float64
    assert np.array_equal(out, np.maximum(np.asarray(raw, dtype=np.float64), 0.0))


def test_target_role_reads_only_support_trials() -> None:
    paths = fold_data.allowlisted_paths()
    target = fold_data.load_fold_session(paths["ses-20120924"], role="target")
    assert int(target.emg_trial_ids.min()) == 0
    assert int(target.emg_trial_ids.max()) == 9
    assert int(target.rate_trial_ids.max()) == 9
    assert target.signal_view["emg_trial_range"] == [0, 10]
    assert target.signal_view["neural_trial_range"] == [0, 10]
    assert target.signal_view["role"] == "target"
    assert target.signal_view["query_neural_or_emg_values_read"] is False
    assert target.signal_view["target_query_values_read"] is False


def test_source_role_reads_full_emg_and_support_neural() -> None:
    paths = fold_data.allowlisted_paths()
    source = fold_data.load_fold_session(paths["ses-20120924"], role="source")
    assert source.signal_view["role"] == "source"
    assert int(source.emg_trial_ids.max()) >= 10
    assert int(source.rate_trial_ids.max()) == 9
    assert source.signal_view["neural_trial_range"] == [0, 10]
    assert source.signal_view["emg_trial_range"][1] > 10


def test_fold_local_scope_does_not_preload_target_query() -> None:
    loaded = fold_data.load_fold_scope(fold=0)
    assert loaded["target"].session == "ses-20120924"
    assert set(loaded["sources"]) == {"ses-20120926", "ses-20120927", "ses-20120928"}
    assert int(loaded["target"].emg_trial_ids.max()) == 9
    for name, record in loaded["sources"].items():
        assert int(record.emg_trial_ids.max()) >= 10
        assert name != loaded["target"].session
    isolation = loaded["isolation"]
    assert isolation["target_emg_trial_range"] == [0, 10]
    assert isolation["target_neural_trial_range"] == [0, 10]
    assert isolation["target_query_values_read"] is False
    assert isolation["source_emg_policy"] == "full_session"
    assert isolation["source_neural_trial_range"] == [0, 10]


def test_dictionary_immutability_uses_real_target_support_emg() -> None:
    loaded = fold_data.load_fold_scope(fold=0)
    source_emg = np.concatenate(
        [rectify.relu_nonnegative_projection(record.emg) for record in loaded["sources"].values()],
        axis=0,
    )
    nmf = rsyn3.fit_source_nmf(source_emg)
    target_emg = rectify.relu_nonnegative_projection(loaded["target"].emg)
    probe = fold_syn3.dictionary_immutability_probe(nmf, target_emg)
    assert probe["source"] == "target_support_emg"
    assert probe["mutated"] is False
    assert int(probe["n_bins"]) == int(target_emg.shape[0])
    assert probe["n_bins"] != 8
    assert probe["dictionary_digest_before"] == probe["dictionary_digest_after"]


def test_pca_diagnostic_is_named_rectified_trial_mean_pca3() -> None:
    label = fold_syn3.RECTIFIED_TRIAL_MEAN_PCA3_LABEL
    assert label == "rectified trial-mean PCA3"
    assert "historical" not in label.lower()
    assert "signed pca" not in label.lower()
    bins = np.ones((12, 16), dtype=np.float64)
    rates = np.ones((12, 4), dtype=np.float64)
    ids = np.repeat(np.arange(6), 2)
    report = fold_syn3.rectified_trial_mean_pca3_reliability(bins, rates, ids)
    assert report["label"] == label
    assert report["bin_law"] == "mean_of_rectified_bins"
    source = inspect.getsource(fold_syn3.rectified_trial_mean_pca3_reliability)
    assert "historical signed" not in source.lower()


def test_trial_mean_pca3_uses_mean_of_rectified_bins_not_relu_of_means() -> None:
    ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    emg = np.zeros((12, 4), dtype=np.float64)
    emg[0] = [-1.0, 2.0, 0.0, 1.0]
    emg[1] = [3.0, 2.0, 0.0, 1.0]
    emg[2:] = 0.4
    rates = np.ones((12, 3), dtype=np.float64)
    raw_means = np.stack([emg[ids == trial].mean(axis=0) for trial in range(6)])
    rect_then_mean = np.stack([
        np.maximum(emg[ids == trial], 0.0).mean(axis=0) for trial in range(6)
    ])
    assert not np.allclose(np.maximum(raw_means, 0.0)[0], rect_then_mean[0])
    report = fold_syn3.rectified_trial_mean_pca3_reliability(emg, rates, ids)
    assert report["bin_law"] == "mean_of_rectified_bins"
    assert report["n_trials"] == 6


def test_ls4_disclosure_is_cyclic_tile_not_resampling() -> None:
    law = plan.LS4_LAW
    assert law["unequal_length"] == "cyclic_repeat_or_truncate"
    assert law["true_resampling"] is False
    assert law["enabled_in_stage1_pilot"] is False
    assert law["must_repair_before"] == "stage2_mechanism_controls"
    z = np.vstack([np.full((3, 3), 1.0), np.full((5, 3), 2.0)])
    rates = np.ones((8, 2), dtype=np.float64)
    ids = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    remapped, _ = rsyn3.derange_trial_association(z, rates, ids, "ses-20120924", 42)
    dest_n = 5
    source_n = 3
    take = np.arange(dest_n) % source_n
    assert take.tolist() == [0, 1, 2, 0, 1]
    assert remapped.shape == z.shape


def test_compare_sealed_pass_extracts_carrier_dictionary_coverage_splithalf() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import compare as fold_compare

    sealed = json.loads((RSYN3_PASS / "fold_receipts.json").read_text())
    keys = fold_compare.comparison_keys(sealed["0"])
    assert "dictionary_digest" in keys
    assert "rows" in keys
    assert len(keys["rows"]) == 4
    assert keys["rows"][0]["budget"] == 10
    assert "carrier_digest" in keys["rows"][0]
    assert "coverage" in keys["rows"][0]
    assert "split_half_rsyn3" in keys["rows"][0]


def test_public_cli_is_dry_and_cannot_mint_gpu_capability() -> None:
    dry = subprocess.run(
        [PYTHON, "-S", str(STAGE0_CLI), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "",
             "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(dry.stdout)
    assert payload["phase"] == "m1_emg_rsyn3_fold_local_v1"
    assert payload["imports_torch"] is False
    assert payload["creates_root_or_receipt"] is False
    assert payload["public_gpu_capability"] is False
    assert payload["gpu"]["gpu0_allowed"] is True
    denied = subprocess.run(
        [PYTHON, str(STAGE0_CLI), "--execute-gpu"],
        cwd=ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "",
             "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True,
    )
    assert denied.returncode != 0
    assert "capability" in (denied.stderr + denied.stdout).lower()


def test_plan_binds_disjoint_root_and_sealed_hashes() -> None:
    assert plan.RESULT_ROOT_RELATIVE == "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1"
    assert plan.STAGE0_ROOT_RELATIVE == (
        "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/stage0"
    )
    assert plan.RESULT_ROOT_RELATIVE != "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1"
    assert plan.DESIGN_SHA256 == _sha(ROOT / plan.DESIGN_RELATIVE)
    assert plan.WORKORDER_SHA256 == _sha(ROOT / plan.WORKORDER_RELATIVE)
    assert plan.STAGE1_ARMS == ("Z-Fix", "S-Fix", "S-Acyc")
    assert "S-LS4" not in plan.STAGE1_ARMS


def test_fold_local_stage0_pass_matches_sealed_v1_table() -> None:
    root = ROOT / plan.STAGE0_ROOT_RELATIVE
    decision = json.loads((root / "decision.json").read_text())
    comparison = json.loads((root / "sealed_pass_comparison.json").read_text())
    isolation = decision["query_isolation"]
    assert decision["decision"] == "PASS"
    assert decision["decoder_r2_computed"] is False
    assert decision["sealed_pass_comparison_all_equal"] is True
    assert comparison["all_equal"] is True
    assert isolation["target_query_values_read"] is False
    assert isolation["fold_local_target_emg_and_neural"] == [0, 10]
    assert decision["ls4"]["enabled_in_stage1_pilot"] is False
    assert decision["ls4"]["unequal_length"] == "cyclic_repeat_or_truncate"
    folds = json.loads((root / "fold_receipts.json").read_text())
    assert folds["0"]["dictionary_immutability"]["source"] == "target_support_emg"
    assert int(folds["0"]["dictionary_immutability"]["n_bins"]) == 636
    assert folds["0"]["pca"]["diagnostic_label"] == "rectified trial-mean PCA3"


def test_parent_fail_is_verified_before_successor_receipt_reserve() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import stage0 as fold_stage0

    source = inspect.getsource(fold_stage0.publish_stage0)
    verify_at = source.find("verify_sealed_roots")
    reserve_at = source.find("run_stage0")
    assert 0 <= verify_at < reserve_at


def test_zero4_target_fit_calls_are_always_zero() -> None:
    controls = rsyn3.build_controls(
        raw_syn3=np.arange(20, dtype=np.float64).reshape(5, 4),
        normalizer_mean=np.zeros(4),
        normalizer_scale=np.ones(4),
        session_name="ses-20120924",
        seed=42,
        n_units=5,
        target_fit_invoked=True,
    )
    fixed = fold_syn3.zero4_never_fits_target(controls)
    assert fixed["zero4_target_fit_calls"] == 0
    assert np.array_equal(fixed["Zero4"], np.zeros((5, 4)))


def test_fold_local_imports_injection_and_cdm_contracts() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import model as injection
    from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import score as cdm

    projection = injection.zero_linear(4, 8)
    hidden = np.ones((2, 8))
    carrier = np.arange(8, dtype=np.float64).reshape(2, 4)
    mask = np.array([1.0, 0.0])
    out = injection.apply_injection(hidden, carrier, projection, mask)
    assert np.array_equal(out[1], np.zeros(8))
    fifo = cdm.ActivityFIFO(support_trials=2)
    state = fifo.initialize([np.ones((2, 4)), np.full((2, 4), 2.0)])
    carrier0 = np.ones((3, 4))
    _, carrier_after, model_after = fifo.commit_completed(
        state, trial_activity=np.full((2, 4), 3.0), carrier=carrier0, model_digest="frozen",
    )
    assert np.array_equal(carrier_after, carrier0)
    assert model_after == "frozen"
