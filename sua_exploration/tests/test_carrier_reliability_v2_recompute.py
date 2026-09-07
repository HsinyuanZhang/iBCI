"""Tests for the V2 carrier-reliability recomputation.

These cover the parts that can be got wrong silently: the estimator actually used, the statistic's
algebra, the permutation p, the leave-one-out sweep, and the scope discipline that quarantined a
previous agent's receipt.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2
from sua_exploration.scripts import run_carrier_reliability_v2_recompute as runner


RECEIPT = ROOT / "sua_exploration/results/carrier_reliability_v2_20260814/receipt.json"


# --------------------------------------------------------------------------------------
# The estimator under test really is V2
# --------------------------------------------------------------------------------------

def test_runner_analyses_m4_because_m3_split_is_infeasible():
    assert runner.BUDGET == 4


def test_v2_constants_are_the_deployed_ones():
    assert (v2.CARRIER_DIM, v2.LATENT_DIM, v2.RIDGE_LAMBDA) == (5, 4, 3.0)
    assert (v1.CARRIER_DIM, v1.LATENT_DIM, v1.RIDGE_LAMBDA) == (4, 3, 0.1)


def test_v2_fitter_penalty_leaves_intercept_unpenalised_and_shrinks_four_slopes():
    """A pure-intercept response must come back with zero slopes and an unshrunk intercept."""
    generator = np.random.default_rng(0)
    z = generator.normal(size=(16, v2.LATENT_DIM))
    z -= z.mean(axis=0)  # orthogonalise the slopes against the intercept column
    response = np.tile(np.arange(v1.EXPECTED_NEURONS, dtype=np.float64), (16, 1))
    carrier = v2.fit_carrier_arrays(z, response)
    assert carrier.shape == (v1.EXPECTED_NEURONS, v2.CARRIER_DIM)
    assert np.allclose(carrier[:, : v2.LATENT_DIM], 0.0, atol=1e-9)
    assert np.allclose(carrier[:, v2.LATENT_DIM], np.arange(v1.EXPECTED_NEURONS))


def test_v2_fitter_refuses_fewer_than_eight_events():
    """This floor is why M=3 cannot be analysed; if it ever relaxes the protocol must be revisited."""
    generator = np.random.default_rng(1)
    z = generator.normal(size=(7, v2.LATENT_DIM))
    response = generator.normal(size=(7, v1.EXPECTED_NEURONS))
    with pytest.raises(v1.SparseEventEndpointError):
        v2.fit_carrier_arrays(z, response)


def test_v2_fitter_rejects_a_v1_shaped_three_column_design():
    generator = np.random.default_rng(2)
    z = generator.normal(size=(12, v1.LATENT_DIM))
    response = generator.normal(size=(12, v1.EXPECTED_NEURONS))
    with pytest.raises(v1.SparseEventEndpointError):
        v2.fit_carrier_arrays(z, response)


# --------------------------------------------------------------------------------------
# Statistic algebra
# --------------------------------------------------------------------------------------

def test_median_slope_cosine_is_one_for_identical_carriers():
    generator = np.random.default_rng(3)
    slopes = generator.normal(size=(v1.EXPECTED_NEURONS, v2.LATENT_DIM))
    result = runner._median_slope_cosine(slopes, slopes)
    assert result["status"] == "defined"
    assert result["defined_channels"] == v1.EXPECTED_NEURONS
    assert result["median_weight_cosine"] == pytest.approx(1.0)


def test_median_slope_cosine_is_minus_one_for_negated_carriers():
    generator = np.random.default_rng(4)
    slopes = generator.normal(size=(v1.EXPECTED_NEURONS, v2.LATENT_DIM))
    assert runner._median_slope_cosine(slopes, -slopes)["median_weight_cosine"] == pytest.approx(-1.0)


def test_median_slope_cosine_drops_zero_norm_channels():
    generator = np.random.default_rng(5)
    first = generator.normal(size=(v1.EXPECTED_NEURONS, v2.LATENT_DIM))
    second = first.copy()
    second[:10] = 0.0
    result = runner._median_slope_cosine(first, second)
    assert result["defined_channels"] == v1.EXPECTED_NEURONS - 10


def test_spearman_brown_steps_up_and_fixes_the_endpoints():
    assert runner.spearman_brown(0.0) == pytest.approx(0.0)
    assert runner.spearman_brown(1.0) == pytest.approx(1.0)
    assert runner.spearman_brown(0.06) > 0.06
    assert runner.spearman_brown(0.0598) == pytest.approx(2 * 0.0598 / 1.0598)


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------

def test_permutation_p_matches_scipy_rho_and_is_small_for_a_monotone_pair():
    x = list(range(13))
    y = [value * 2.0 + 1.0 for value in x]
    result = runner.spearman_with_permutation(x, y, permutations=5_000, seed=7)
    assert result["rho"] == pytest.approx(1.0)
    assert result["n"] == 13
    assert result["p_permutation"] < 0.01


def test_permutation_p_is_near_uniform_for_an_unrelated_pair():
    generator = np.random.default_rng(11)
    x = generator.normal(size=13)
    y = generator.normal(size=13)
    result = runner.spearman_with_permutation(x, y, permutations=20_000, seed=13)
    assert result["p_permutation"] > 0.05
    assert result["rho"] == pytest.approx(stats.spearmanr(x, y).statistic)


def test_permutation_p_is_reproducible_under_the_same_seed():
    generator = np.random.default_rng(17)
    x = generator.normal(size=13)
    y = x + generator.normal(size=13) * 0.5
    first = runner.spearman_with_permutation(x, y, permutations=10_000, seed=99)
    second = runner.spearman_with_permutation(x, y, permutations=10_000, seed=99)
    assert first == second


def test_permutation_p_never_reports_zero():
    x = list(range(13))
    result = runner.spearman_with_permutation(x, x, permutations=1_000, seed=3)
    assert result["p_permutation"] == pytest.approx(1.0 / 1001.0)


def test_spearman_rejects_a_constant_variable():
    with pytest.raises(runner.ReliabilityError):
        runner.spearman_with_permutation([1.0] * 13, list(range(13)), permutations=10, seed=1)


def test_leave_one_out_reports_thirteen_removals_and_bracket():
    generator = np.random.default_rng(23)
    x = generator.normal(size=13)
    y = x + generator.normal(size=13) * 0.4
    names = [f"s{index}" for index in range(13)]
    result = runner.leave_one_out_rho(x, y, names)
    assert len(result["per_removal"]) == 13
    assert all(row["n"] == 12 for row in result["per_removal"])
    assert result["minimum"] <= result["median"] <= result["maximum"]
    assert {row["removed_session"] for row in result["per_removal"]} == set(names)


def test_leave_one_out_detects_a_single_session_driving_the_correlation():
    """A correlation carried by one outlier must show a collapsing minimum."""
    x = np.array([0.0, 0.1, 0.05, 0.02, 0.08, 0.03, 0.09, 0.01, 0.07, 0.04, 0.06, 0.02, 10.0])
    y = np.array([0.5, 0.2, 0.9, 0.1, 0.4, 0.7, 0.3, 0.8, 0.6, 0.2, 0.5, 0.9, 99.0])
    names = [f"s{index}" for index in range(13)]
    result = runner.leave_one_out_rho(x, y, names)
    assert result["minimum"] < 0.4
    assert result["minimum_at_session"] == "s12"


# --------------------------------------------------------------------------------------
# Scope discipline: this is what quarantined a previous receipt
# --------------------------------------------------------------------------------------

def test_reject_path_scope_fails_closed_on_held_out():
    with pytest.raises(v1.SparseEventEndpointError):
        v1.reject_path_scope(
            ROOT / "SPINT-main/data/000954/sub-HumanPitt-held-out-calib/anything.nwb"
        )


def test_reject_path_scope_fails_closed_on_minival():
    with pytest.raises(v1.SparseEventEndpointError):
        v1.reject_path_scope(
            ROOT / "SPINT-main/data/000954/sub-HumanPitt-held-in-minival/anything.nwb"
        )


def test_runner_does_not_glob_for_nwb_files():
    """Discovery must go through index_heldin_calib, never a raw glob."""
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "index_heldin_calib" in source
    assert ".nwb" not in source
    assert "glob(" not in source


# --------------------------------------------------------------------------------------
# Receipt, when it exists
# --------------------------------------------------------------------------------------

@pytest.mark.skipif(not RECEIPT.exists(), reason="receipt not yet published")
class TestPublishedReceipt:
    @staticmethod
    def _body():
        return json.loads(RECEIPT.read_text(encoding="utf-8"))

    def test_receipt_matches_its_sidecar(self):
        sidecar = RECEIPT.with_suffix(RECEIPT.suffix + ".sha256")
        assert sidecar.read_text(encoding="ascii").split()[0] == v1.sha256_file(RECEIPT)

    def test_receipt_is_immutable(self):
        assert RECEIPT.stat().st_mode & 0o222 == 0

    def test_provenance_passed_all_thirteen_sessions(self):
        provenance = self._body()["provenance"]
        assert provenance["passed"] is True
        assert provenance["sessions_checked"] == 13
        assert provenance["sessions_fully_matched"] == 13
        for row in provenance["per_session"]:
            assert row["basis_sha256_matches"]
            assert row["carrier_sha256_matches"]
            assert row["input_sha256_matches"]

    def test_receipt_binds_inputs_and_environment(self):
        body = self._body()
        assert len(body["binding"]["inputs"]) == 13
        assert body["binding"]["v2_module_sha256"]
        assert body["binding"]["protocol_doc_sha256"]
        assert body["environment"]["numpy"]
        assert body["environment"]["thread_environment"]["OMP_NUM_THREADS"] == "1"

    def test_receipt_scope_is_clean(self):
        scope = self._body()["scope"]
        assert scope["cuda_used"] is False
        assert scope["held_out_nwbs_opened"] == 0
        assert scope["minival_nwbs_opened"] == 0
        assert scope["raw_glob_used"] is False
        assert scope["dense_velocity_series_opened"] is False
        assert scope["public_held_in_calibration_nwbs_opened"] == 13

    def test_receipt_reports_thirteen_sessions_with_defined_stability(self):
        body = self._body()
        assert len(body["per_session"]) == 13
        for row in body["per_session"].values():
            assert row["stability_v2"]["status"] == "defined"
            assert row["stability_v2"]["free_parameters_per_channel"] == 5

    def test_receipt_records_leave_one_out_for_all_thirteen(self):
        loo = self._body()["leave_one_out"]["v2_stability_vs_outcome"]
        assert len(loo["per_removal"]) == 13

    def test_receipt_states_m3_is_not_analysable(self):
        assert self._body()["m3_split_feasibility"]["analysable_at_m3"] is False

    def test_receipt_keeps_the_frozen_status_rather_than_overriding_it(self):
        body = self._body()
        assert body["status"].startswith(
            ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "STOP")
        )
        assert body["decision"]["adjudication"]
