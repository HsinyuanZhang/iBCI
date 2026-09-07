"""Synthetic no-data/no-CUDA tests for the M2 carrier alpha diagnostic.

The review-critical properties:

1. the alpha axis is a subset of the SEALED stage-2 mass grid and the anchor
   is the governing value; only alpha ever moves;
2. the verdict rule is exactly the pre-registered two-string law, including
   the epsilon band that never flips on float noise;
3. the monotonicity flags answer the pre-registered question;
4. every alpha > 0 row carries the binding leakage labels and no row is ever
   selection/deployment eligible;
5. the frozen rollout machinery, driven through THIS package's binding,
   satisfies the alpha = 0 exact no-op law and actually moves the carrier at
   alpha > 0 on a synthetic session (end-to-end, CPU numpy only);
6. the sealed stage-2 lookup fails closed on drift.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.support_anchored_t4_stage_p_v1 import gate as stage_p_gate
from src.support_anchored_t4_stage_p_v1 import plan as stage_p_plan
from src.support_anchored_t4_stage_p_v1 import replay as stage_p_replay

from src.cdm_p1_m2_local_v1 import replay as governing_replay
from src.cdm_p1_m2_local_v1.anchor import M2SupportAnchor, build_groups

from src.m2_carrier_alpha_diagnostic_v1 import physical, plan


# ---------------------------------------------------------------------------
# 1-4: the frozen laws.
# ---------------------------------------------------------------------------


def test_alpha_axis_is_subset_of_the_sealed_grid() -> None:
    assert plan.ALPHA_ANCHOR == 0.0
    assert set(plan.ALPHA_SCAN) <= set(stage_p_plan.ALPHA_M_CANDIDATES)
    assert 0.25 not in plan.ALPHA_SCAN  # only the two briefed points are run
    assert plan.ALPHA_ALL == (0.0, 0.125, 0.5)
    assert plan.SURFACES == ("within_post30", "external_official_query")
    assert plan.HELD_OUT_SURFACE == "external_official_query"
    assert plan.BUDGETS == (4, 10)


def test_governing_vectors_are_the_sealed_selection() -> None:
    for budget, vector in ((4, plan.GOVERNING_M4), (10, plan.GOVERNING_M10)):
        assert vector["alpha_M"] == 0.0
        assert vector["rho_M"] in stage_p_plan.RHO_M_CANDIDATES
        assert vector["c_M"] > 0.0
        thresholds = vector["thresholds"]
        assert thresholds["tau_d_rad"] in stage_p_plan.TAU_D_CANDIDATES
        assert thresholds["r_max_repetition_per_direction"] in stage_p_plan.R_MAX_CANDIDATES
        assert thresholds["d_min_distinct_directions"] in stage_p_plan.D_MIN_CANDIDATES
        assert (thresholds["max_pseudo_mass_relative_to_support_rows"]
                in stage_p_plan.MAX_PSEUDO_MASS_RELATIVE_CANDIDATES)
        assert plan.GOVERNING_BY_BUDGET[budget] == vector


def test_only_alpha_moves_in_the_hyperparameter_binding() -> None:
    for budget in plan.BUDGETS:
        vector = plan.GOVERNING_BY_BUDGET[budget]
        for alpha in plan.ALPHA_ALL:
            hp = physical._hp_with_alpha(
                vector, alpha, c_M=float(vector["c_M"]))
            assert hp.alpha_M == alpha
            assert hp.rho_M == vector["rho_M"]
            assert hp.c_M == vector["c_M"]
            assert hp.thresholds.payload() == {
                "tau_d_rad": vector["thresholds"]["tau_d_rad"],
                "r_max_repetition_per_direction":
                    vector["thresholds"]["r_max_repetition_per_direction"],
                "d_min_distinct_directions":
                    vector["thresholds"]["d_min_distinct_directions"],
                "max_pseudo_mass_relative_to_support_rows":
                    vector["thresholds"]["max_pseudo_mass_relative_to_support_rows"],
            }
        # the R family replays the selection grid's own c_M = None binding
        hp_r = physical._hp_with_alpha(vector, 0.125, c_M=None)
        assert hp_r.c_M is None and hp_r.alpha_M == 0.125


def test_verdict_rule_is_the_pre_registered_two_string_law() -> None:
    epsilon = float(plan.VERDICT_LAW["boundary_epsilon"])
    all_nonpositive = {"m4|alpha0.125": -0.004, "m4|alpha0.5": -0.009,
                       "m10|alpha0.125": -0.001, "m10|alpha0.5": -0.003}
    verdict = physical.evaluate_verdict(all_nonpositive, epsilon=epsilon)
    assert verdict["verdict"] == plan.VERDICT_MATCHED == "WITHIN_SELECTION_MATCHED_HELDOUT"
    assert verdict["any_alpha_positive_on_heldout"] is False
    assert all(row["counts_as_positive"] is False for row in verdict["rows"].values())

    some_positive = dict(all_nonpositive, **{"m10|alpha0.5": 0.0021})
    verdict = physical.evaluate_verdict(some_positive, epsilon=epsilon)
    assert verdict["verdict"] == plan.VERDICT_MISMATCH == "SELECTION_SURFACE_MISMATCH"
    assert verdict["rows"]["m10|alpha0.5"]["counts_as_positive"] is True

    # float noise inside the epsilon band never flips the verdict
    noisy = dict(all_nonpositive, **{"m4|alpha0.5": 1.0e-13})
    verdict = physical.evaluate_verdict(noisy, epsilon=epsilon)
    assert verdict["verdict"] == plan.VERDICT_MATCHED
    row = verdict["rows"]["m4|alpha0.5"]
    assert row["within_epsilon_band_of_zero"] is True
    assert row["counts_as_positive"] is False
    assert row["sign"] == "zero_band"

    exactly_zero = dict(all_nonpositive, **{"m10|alpha0.125": 0.0})
    verdict = physical.evaluate_verdict(exactly_zero, epsilon=epsilon)
    assert verdict["verdict"] == plan.VERDICT_MATCHED
    with pytest.raises(Exception):
        physical.evaluate_verdict({}, epsilon=epsilon)


def test_monotonicity_flags_answer_the_pre_registered_question() -> None:
    decreasing = {0.0: 0.45, 0.125: 0.44, 0.5: 0.43}
    flags = physical.evaluate_monotonicity(decreasing)
    assert flags["nonincreasing_in_alpha"] is True
    assert flags["anchor_is_maximum"] is True

    bump = {0.0: 0.45, 0.125: 0.43, 0.5: 0.46}
    flags = physical.evaluate_monotonicity(bump)
    assert flags["nonincreasing_in_alpha"] is False
    assert flags["anchor_is_maximum"] is False

    helps_at_low_alpha = {0.0: 0.45, 0.125: 0.46, 0.5: 0.44}
    flags = physical.evaluate_monotonicity(helps_at_low_alpha)
    assert flags["nonincreasing_in_alpha"] is False
    assert flags["anchor_is_maximum"] is False

    flat = {0.0: 0.45, 0.125: 0.45, 0.5: 0.45}
    flags = physical.evaluate_monotonicity(flat)
    assert flags["nonincreasing_in_alpha"] is True
    assert flags["anchor_is_maximum"] is True
    with pytest.raises(Exception):
        physical.evaluate_monotonicity({0.0: 0.45, 0.125: 0.44})


def test_leakage_labels_follow_the_binding_law() -> None:
    for alpha in plan.ALPHA_SCAN:
        labels = physical._leakage_labels(alpha)
        assert labels["target_label_leakage"] is True
        assert labels["checkpoint_selection_eligible"] is False
        assert labels["deployment_eligible"] is False
        assert labels["post_hoc_diagnostic"] is True
    anchor = physical._leakage_labels(plan.ALPHA_ANCHOR)
    assert anchor["target_label_leakage"] is False
    assert anchor["checkpoint_selection_eligible"] is False
    assert anchor["deployment_eligible"] is False
    assert anchor["post_hoc_diagnostic"] is True
    statement = plan.LEAKAGE_LAW["receipt_statement"]
    assert "never" in statement.lower() and "select" in statement.lower()


def test_cpu_and_receipt_laws_are_declared() -> None:
    assert plan.ENVIRONMENT_LAW["device"] == "cpu"
    assert "never touched" in plan.ENVIRONMENT_LAW["gpu_policy"] \
        or "no GPU" in plan.ENVIRONMENT_LAW["gpu_policy"]
    assert plan.ENVIRONMENT_LAW["dataloader_workers_max"] <= 2
    assert plan.ENVIRONMENT_LAW["torch_num_threads"] == 4
    assert plan.ANCHOR_TOLERANCE_R2 == 1.0e-5
    assert plan.RECEIPT_LAW["attempt_before_any_data_or_model_access"] is True
    assert "cdm_p1_m2_local_v1" in plan.RECEIPT_LAW["frozen_roots_never_modified"][0]
    deviations = {item["id"] for item in plan.DEVIATIONS}
    assert {"cpu_instead_of_gpu1", "tolerance_anchor_instead_of_bitwise",
            "selection_grid_binding"} <= deviations


def test_stage2_lookup_fails_closed() -> None:
    sealed = {"replay": {"hyperparameter_selection": {"m4": {"stage2_scored": [
        {"vector": {"rho_M": 0.5, "alpha_M": 0.125}, "mean_r2": 0.449,
         "per_session": {"s1": 0.4}},
        {"vector": {"rho_M": 1.0, "alpha_M": 0.125}, "mean_r2": 0.448,
         "per_session": {"s1": 0.39}},
    ]}}}}
    row = physical._stage2_scored(sealed, 4, 0.5, 0.125)
    assert row["mean_r2"] == 0.449
    with pytest.raises(physical.M2CarrierAlphaDiagnosticError):
        physical._stage2_scored(sealed, 4, 0.5, 0.5)
    with pytest.raises(physical.M2CarrierAlphaDiagnosticError):
        physical._governing_hp({"replay": {"hyperparameters": {"m4": {
            "rho_M": 1.0, "alpha_M": 0.0, "c_M": 0.2, "thresholds": {
                "tau_d_rad": 0.7853981633974483,
                "r_max_repetition_per_direction": 4,
                "d_min_distinct_directions": 4,
                "max_pseudo_mass_relative_to_support_rows": 1.0}}}}}, 4)


# ---------------------------------------------------------------------------
# 5: the end-to-end synthetic rollout (frozen machinery, this binding).
# ---------------------------------------------------------------------------


class _StubTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array, dtype=np.float32)

    def detach(self) -> "_StubTensor":
        return self

    def cpu(self) -> "_StubTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


@dataclass
class _StubDecoder:
    units: int
    fill: float = 0.0

    def identity(self, activity: np.ndarray, side: np.ndarray,
                 keep: Optional[np.ndarray] = None) -> _StubTensor:
        channels = (np.arange(activity.shape[2]) if keep is None
                    else np.asarray(keep, dtype=np.int64))
        # the identity content tracks the active carrier's modulation column,
        # so a committed movement changes the decode exactly like the real one
        fill = float(np.asarray(side, dtype=np.float32)[channels, 2].mean())
        return _StubTensor(np.full((channels.size, 50), fill, dtype=np.float32))

    def decode(self, neural: np.ndarray, identity: _StubTensor) -> np.ndarray:
        windows = int(neural.shape[0])
        # ~6 cm/s toward canonical direction 0 (+x), perturbed by the carrier
        amplitude = 0.06 + 1.0e-4 * float(identity.numpy().mean())
        return np.ascontiguousarray(
            np.tile(np.asarray([amplitude, 0.0], dtype=np.float32), (windows, 1)))

    def windows(self, neural: np.ndarray, starts: np.ndarray,
                keep: Optional[np.ndarray] = None) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.int64)
        indices = starts[:, None] + np.arange(50, dtype=np.int64)[None, :]
        block = np.ascontiguousarray(neural[indices], dtype=np.float32)
        if keep is None:
            return block
        return np.ascontiguousarray(block[:, :, np.asarray(keep, dtype=np.int64)])

    def clear_cache(self) -> None:
        return None


def _synthetic_session(*, units: int = 24, trials: int = 34):
    rng = np.random.default_rng(5)
    bins_per_trial = 200
    total_bins = trials * bins_per_trial
    neural = rng.poisson(0.5, size=(total_bins, units)).astype(np.float32)
    behavior = rng.normal(0.0, 0.5, size=(total_bins, 2)).astype(np.float32)
    dataset = SimpleNamespace(
        neural_data={"synthetic": neural},
        covariate_data={"synthetic": behavior},
        side_feature_mean=np.full(4, 0.25, dtype=np.float32),
        side_feature_std=np.full(4, 2.0, dtype=np.float32),
    )
    trials_list = []
    starts: list[np.ndarray] = []
    for position in range(trials):
        start_bin = position * bins_per_trial
        stop_bin = start_bin + bins_per_trial
        metric = np.ascontiguousarray(
            start_bin + 10 + 40 * np.arange(3, dtype=np.int64))
        causal = np.arange(start_bin, stop_bin - 49, dtype=np.int64)
        starts.append(metric)
        trials_list.append(governing_replay.QueryTrial(
            position=position, start_bin=start_bin, stop_bin=stop_bin,
            metric_starts=metric, causal_starts=np.ascontiguousarray(causal),
        ))
    starts_array = np.ascontiguousarray(np.concatenate(starts), dtype=np.int64)
    material = governing_replay.SessionMaterial(
        session="synthetic", surface="within_post30", dataset=dataset,
        starts=starts_array,
        targets=np.ascontiguousarray(behavior[starts_array + 49], dtype=np.float32),
        trials=tuple(trials_list),
        channel_ids=np.arange(units, dtype=np.int64),
    )
    return material, dataset


def _synthetic_carrier(units: int = 24, seed: int = 3):
    from src.calibration_budget_comparators_v1 import fit_ridge_t4

    rng = np.random.default_rng(seed)
    canonical = np.asarray(cdm_core.CANONICAL_DIRECTIONS_RAD)
    angles = np.asarray([canonical[index] + 0.03 * rng.standard_normal()
                         for index in (0, 2, 4, 6)])
    rates = rng.gamma(3.0, 0.4, size=(angles.size, units))
    raw, _evidence = fit_ridge_t4(
        rates, angles,
        normalized_lambda=governing_replay.plan.RIDGE_NORMALIZED_LAMBDA,
    )
    side = np.ascontiguousarray(
        (raw - np.full(4, 0.25)) / np.full(4, 2.0), dtype=np.float32)
    activity = np.ascontiguousarray(
        rng.poisson(0.5, size=(4, 30, units)).astype(np.float32))
    return {
        "budget": 4, "selected": np.arange(4, dtype=np.int64),
        "theta": angles, "rates": rates,
        "raw_t4": np.ascontiguousarray(raw, dtype=np.float32),
        "side": side, "activity": activity,
    }


def _synthetic_binding():
    material, _dataset = _synthetic_session()
    carrier = _synthetic_carrier()
    groups = build_groups(carrier["raw_t4"], material.channel_ids)
    anchor = M2SupportAnchor.from_labeled_support(
        groups=groups, support_trial_rates=carrier["rates"],
        support_angles_rad=carrier["theta"].tolist(),
        support_t4=carrier["raw_t4"],
    )
    assert anchor.support_coefficient_parity["rebuilt_rows_bitwise_equal"] is True
    return material, carrier, anchor


_PERMISSIVE = stage_p_gate.GateThresholds(
    tau_d=math.pi / 4.0, r_max=8, d_min=3, max_mass_relative=2.0,
)


def test_synthetic_rollout_alpha_zero_is_exact_noop() -> None:
    material, carrier, anchor = _synthetic_binding()
    decoder = _StubDecoder(units=material.channel_ids.size)
    hp = stage_p_replay.CellHyperparameters(
        rho_M=0.5, alpha_M=0.0, c_M=None, thresholds=_PERMISSIVE)
    rollout = physical._rollout(
        decoder, material, budget=4, anchor=anchor, carrier=carrier,
        hp=hp, row_spec=stage_p_replay.ROW_SPECS["P1"],
    )
    assert rollout["structural_noop"] is True
    assert rollout["final_carrier_sha256"] == rollout["initial_carrier_sha256"]
    assert all(receipt["cb"] == receipt["ca"] for receipt in rollout["receipts"])
    assert (np.max(rollout["movements"]) == 0.0) if rollout["movements"] else True
    assert rollout["committed_rows"] >= 1  # commits happen, movement is zero
    assert physical._causality_chain(rollout["receipts"]) is True


def test_synthetic_rollout_alpha_positive_moves_carrier_and_predictions() -> None:
    material, carrier, anchor = _synthetic_binding()
    hp_zero = stage_p_replay.CellHyperparameters(
        rho_M=0.5, alpha_M=0.0, c_M=None, thresholds=_PERMISSIVE)
    hp_half = stage_p_replay.CellHyperparameters(
        rho_M=0.5, alpha_M=0.5, c_M=None, thresholds=_PERMISSIVE)
    baseline = physical._rollout(
        _StubDecoder(units=material.channel_ids.size), material, budget=4,
        anchor=anchor, carrier=carrier, hp=hp_zero,
        row_spec=stage_p_replay.ROW_SPECS["P1"])
    moved = physical._rollout(
        _StubDecoder(units=material.channel_ids.size), material, budget=4,
        anchor=anchor, carrier=carrier, hp=hp_half,
        row_spec=stage_p_replay.ROW_SPECS["P1"])
    assert moved["committed_rows"] >= 1
    assert moved["structural_noop"] is False
    assert moved["final_carrier_sha256"] != moved["initial_carrier_sha256"]
    assert np.max(moved["movements"]) > 0.0
    assert not np.array_equal(moved["prediction"], baseline["prediction"])
    assert moved["prediction"].shape == baseline["prediction"].shape
    # the D-family row labelling on top of a moved rollout
    labels = physical._leakage_labels(0.5)
    assert labels["target_label_leakage"] is True
    assert labels["deployment_eligible"] is False


def test_synthetic_rollout_reproduces_selection_binding_shape() -> None:
    material, carrier, anchor = _synthetic_binding()
    hp = stage_p_replay.CellHyperparameters(
        rho_M=0.5, alpha_M=0.125, c_M=None, thresholds=_PERMISSIVE)
    rollout = physical._rollout(
        _StubDecoder(units=material.channel_ids.size), material, budget=4,
        anchor=anchor, carrier=carrier, hp=hp,
        row_spec=stage_p_replay.ROW_SPECS["P2"])
    assert rollout["cell"] == "F01m"
    assert rollout["prediction"].shape == material.targets.shape
    assert physical._causality_chain(rollout["receipts"]) is True
