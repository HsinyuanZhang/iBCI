"""Synthetic no-data/no-CUDA tests for the M2 Stage-O oracle headroom test.

The review-critical properties:

1. the Stage-O amendment hyperparameters are frozen (rho_M = 1.0, block = 1
   trial, w = 1, alpha_M = 0.5 primary, {0.125, 0.25, 1.0} non-governing);
2. the verdict rule is exactly the pre-registered two-string law, including
   the +0.01 delta floor, the >= 4/6 held-out breadth and the epsilon band
   that never flips on float noise;
3. the NULL-branch wall classifier separates commit-point scarcity from an
   informational ceiling;
4. the census counts eligible/with-window/accepted trials and the typed
   rejection reasons of the frozen direction gates;
5. the true-direction law maps covariate rows to the correct canonical snap
   under the frozen unit law and rejects degenerate movement;
6. every O2m row carries the binding leakage labels and no row is ever
   selection/deployment eligible;
7. the frozen Stage-O machinery, driven through THIS package's adapter and
   binding, satisfies the alpha = 0 exact no-op law (bitwise against the
   frozen F00m rollout) and actually moves the carrier at alpha > 0 on a
   synthetic session (end-to-end, CPU numpy only);
8. the anchor adapter supplies exactly the statistics view the frozen
   block-refit reads and the empty bank reproduces the support solve.
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
from src.support_anchored_t4_stage_o_v1 import block_refit as stage_o_block_refit
from src.support_anchored_t4_stage_o_v1 import trust_region

from src.cdm_p1_m2_local_v1 import replay as governing_replay
from src.cdm_p1_m2_local_v1.anchor import M2SupportAnchor, build_groups

from src.m2_carrier_oracle_v1 import physical, plan


# ---------------------------------------------------------------------------
# 1-4: the frozen laws.
# ---------------------------------------------------------------------------


def test_stage_o_amendment_hyperparameters_are_frozen() -> None:
    assert plan.RHO_M == 1.0
    assert plan.BLOCK_SIZE_COMPLETED_TRIALS == 1
    assert plan.EVIDENCE_WEIGHT_W == 1.0
    assert plan.ALPHA_M_PRIMARY == 0.5
    assert plan.ALPHA_M_SENSITIVITY == (0.125, 0.25, 1.0)
    assert plan.ALPHA_M_SENSITIVITY_GOVERNS is False
    assert plan.ALPHA_M_ALL == (0.5, 0.125, 0.25, 1.0)
    assert plan.ALPHA_M_M30_NOOP == 0.0
    assert plan.SURFACES == ("within_post30", "external_official_query")
    assert plan.BUDGETS == (4, 10)
    assert plan.M30 == 30
    assert plan.EXPECTED_WITHIN_SESSIONS == 7
    assert plan.EXPECTED_EXTERNAL_SESSIONS == 6
    assert plan.HELD_OUT_BREADTH_MIN == 4


def test_commit_law_is_always_commit_with_no_three_factor_semantics() -> None:
    law = plan.O2M_COMMIT_LAW
    assert law["commit_rule"] == "always-commit"
    assert law["movement_rule"] == "trust-region projection only"
    assert "NONE" in law["rejection_semantics"]
    assert "Stage P" in law["rejection_semantics"]
    assert "measurement_validity_still_required" in law
    assert plan.C_M_CALIBRATION["frozen_before_target_scoring"] is True
    assert "MEDIAN" in plan.C_M_CALIBRATION["rule"]


def test_verdict_rule_is_the_pre_registered_two_string_law() -> None:
    def delta(mean: float, positives: int, total: int = 6) -> dict:
        return {"equal_session_mean_delta": mean, "positive_sessions": positives,
                "session_count": total}

    # GO: m4 clears +0.01 with 5/6 breadth
    verdict = physical.evaluate_verdict(external_deltas={
        4: delta(0.0134, 5), 10: delta(-0.0021, 2)})
    assert verdict["verdict"] == plan.VERDICT_GO == "GO_M2_DENOISER_TARGET_EXISTS"
    assert verdict["go_budgets"] == ["m4"]
    assert verdict["rows"]["m4"]["meets_delta"] is True
    assert verdict["rows"]["m4"]["meets_breadth"] is True

    # delta clears but breadth 3/6 fails -> NULL
    verdict = physical.evaluate_verdict(external_deltas={
        4: delta(0.02, 3), 10: delta(-0.001, 2)})
    assert verdict["verdict"] == plan.VERDICT_NULL
    assert verdict["rows"]["m4"]["meets_breadth"] is False

    # breadth clears (4/6) but delta +0.009 misses -> NULL
    verdict = physical.evaluate_verdict(external_deltas={
        4: delta(0.009, 4), 10: delta(0.004, 4)})
    assert verdict["verdict"] == plan.VERDICT_NULL == "ORACLE_NULL__M2_CARRIER_CEILING_IS_ZERO"
    assert verdict["rows"]["m4"]["meets_delta"] is False
    assert verdict["rows"]["m10"]["meets_delta"] is False

    # exact boundary: +0.010 exactly with 4/6 -> GO
    verdict = physical.evaluate_verdict(external_deltas={
        4: delta(0.01, 4), 10: delta(-0.5, 0)})
    assert verdict["verdict"] == plan.VERDICT_GO

    # float noise just below the floor inside the 1e-12 band is disclosed,
    # never silently flipped
    verdict = physical.evaluate_verdict(external_deltas={
        4: delta(0.01 - 1.0e-13, 4), 10: delta(-0.5, 0)})
    assert verdict["rows"]["m4"]["within_epsilon_band_of_threshold"] is True
    assert verdict["verdict"] == plan.VERDICT_GO
    verdict = physical.evaluate_verdict(external_deltas={
        4: delta(0.01 - 1.0e-9, 4), 10: delta(-0.5, 0)})
    assert verdict["verdict"] == plan.VERDICT_NULL

    with pytest.raises(Exception):
        physical.evaluate_verdict(external_deltas={4: delta(0.02, 5)})


def test_wall_classifier_separates_the_two_walls() -> None:
    scarcity = physical.classify_wall(pooled_external_accepted={4: 9, 10: 11})
    assert scarcity["wall"] == "commit_point_scarcity_wall"
    assert scarcity["pooled_external_accepted_true_directions"] == {"m4": 9, "m10": 11}
    assert scarcity["scarcity_threshold_per_budget"] == 24

    information = physical.classify_wall(pooled_external_accepted={4: 30, 10: 8})
    assert information["wall"] == "no_information_in_true_directions"

    boundary = physical.classify_wall(pooled_external_accepted={4: 24, 10: 5})
    assert boundary["wall"] == "no_information_in_true_directions"

    with pytest.raises(Exception):
        physical.classify_wall(pooled_external_accepted={4: 10})


def test_census_counts_the_evidence_stream() -> None:
    receipts = [
        {"evidence_eligible": False, "causal_windows": 10, "reason": "frozen_carrier"},
        {"evidence_eligible": True, "causal_windows": 151, "reason": "always_commit"},
        {"evidence_eligible": True, "causal_windows": 151, "reason": "always_commit_zero_movement"},
        {"evidence_eligible": True, "causal_windows": 0,
         "reason": "measurement_rejected:no_complete_window_in_trial"},
        {"evidence_eligible": True, "causal_windows": 90,
         "reason": "measurement_rejected:low_displacement"},
        {"evidence_eligible": True, "causal_windows": 2,
         "reason": "measurement_rejected:movement_too_short"},
    ]
    census = physical.census_from_receipts(receipts)
    assert census["evidence_eligible_trials"] == 5
    assert census["with_complete_window_trials"] == 4
    assert census["accepted_true_directions"] == 2
    assert census["rejection_reasons"] == {
        "low_displacement": 1, "movement_too_short": 1,
        "no_complete_window_in_trial": 1,
    }
    assert census["oracle_acceptance_rate"] == pytest.approx(0.5)

    pooled = physical.pool_census({
        "s1": census,
        "s2": physical.census_from_receipts(receipts[:2]),
    })
    assert pooled["evidence_eligible_trials"] == 6
    assert pooled["with_complete_window_trials"] == 5
    assert pooled["accepted_true_directions"] == 3
    assert pooled["oracle_acceptance_rate"] == pytest.approx(3.0 / 5.0)


def test_decoded_census_reads_only_the_sealed_rows() -> None:
    sealed_rows = {
        ("external_official_query", "4", "ses-a"): {
            "committed_rows": 2,
            "measurement_reasons": {"measurement_rejected:low_displacement": 7},
            "receipts": [
                {"evidence_eligible": True, "causal_windows": 0, "reason": "x"},
                {"evidence_eligible": True, "causal_windows": 10, "reason": "y"},
                {"evidence_eligible": True, "causal_windows": 10, "reason": "z"},
            ],
        },
        ("within_post30", "4", "ses-w"): {
            "committed_rows": 9, "measurement_reasons": {}, "receipts": [],
        },
    }
    census = physical.decoded_census_from_sealed(sealed_rows, "external_official_query", 4)
    assert set(census["per_session"]) == {"ses-a"}
    assert census["pooled"]["committed_rows"] == 2
    assert census["pooled"]["with_complete_window_trials"] == 2
    assert census["pooled"]["decoded_acceptance_rate"] == pytest.approx(1.0)
    assert census["pooled"]["low_displacement_share_of_measurement_attempts"] == pytest.approx(3.5)
    assert "sealed" in census["source"]


def test_leakage_labels_follow_the_binding_law() -> None:
    for cell in ("O2m", "O2m_m30_noop"):
        labels = physical._leakage_labels(cell)
        assert labels["target_label_leakage"] is True
        assert labels["checkpoint_selection_eligible"] is False
        assert labels["deployment_eligible"] is False
        assert labels["post_hoc_diagnostic"] is True
    baseline = physical._leakage_labels("O0m")
    assert baseline["target_label_leakage"] is False
    assert baseline["checkpoint_selection_eligible"] is False
    assert baseline["deployment_eligible"] is False
    statement = plan.LEAKAGE_LAW["receipt_statement"]
    assert "never" in statement.lower() and "select" in statement.lower()
    assert plan.TRUE_DIRECTION_LAW["leakage"].startswith("reading the query stream")


def test_cpu_and_receipt_laws_are_declared() -> None:
    assert plan.ENVIRONMENT_LAW["device"] == "cpu"
    assert "no GPU" in plan.ENVIRONMENT_LAW["gpu_policy"]
    assert plan.ENVIRONMENT_LAW["dataloader_workers_max"] <= 2
    assert plan.ENVIRONMENT_LAW["dataloader_workers_actual"] == 0
    assert plan.ENVIRONMENT_LAW["torch_num_threads"] == 4
    assert plan.ANCHOR_TOLERANCE_R2 == 1.0e-5
    assert plan.RECEIPT_LAW["attempt_before_any_data_or_model_access"] is True
    assert "cdm_p1_m2_local_v1" in plan.RECEIPT_LAW["frozen_roots_never_modified"][0]
    deviations = {item["id"] for item in plan.DEVIATIONS}
    assert {"cpu_instead_of_any_gpu", "tolerance_anchor_instead_of_bitwise",
            "anchor_adapter_for_the_stage_o_laws", "o1m_not_run",
            "c_m_recalibrated_under_the_oracle_law"} <= deviations


# ---------------------------------------------------------------------------
# 5: the frozen true-direction law on synthetic covariates.
# ---------------------------------------------------------------------------


def _trial(start_bin: int, stop_bin: int, position: int,
           window_size: int = 50) -> governing_replay.QueryTrial:
    causal = np.arange(start_bin, stop_bin - window_size + 1, dtype=np.int64)
    metric = np.ascontiguousarray(start_bin + 10 + 40 * np.arange(3, dtype=np.int64))
    return governing_replay.QueryTrial(
        position=position, start_bin=start_bin, stop_bin=stop_bin,
        metric_starts=metric, causal_starts=np.ascontiguousarray(causal),
    )


def test_true_direction_law_snaps_canonical_directions() -> None:
    config = cdm_core.CDMDConfig(support_budget_m=4)
    size = 400
    for canonical, velocity in (
            (3, (0.01, 0.0)),       # +x (0 rad)     -> canonical 3
            (7, (-0.01, 0.0)),      # -x (pi rad)    -> canonical 7
            (4, (0.007, 0.007)),    # +45 deg        -> canonical 4
            (1, (0.0, -0.01)),      # -y (-pi/2)     -> canonical 1
    ):
        covariates = np.zeros((size, 2), dtype=np.float32)
        covariates[:, 0] = velocity[0]
        covariates[:, 1] = velocity[1]
        trial = _trial(50, 350, position=40)
        direction, meta = physical.true_direction_for_trial(covariates, trial, config=config)
        assert direction.accepted is True, f"canonical {canonical} rejected"
        assert direction.theta_index == canonical
        assert direction.canonical_distance_rad == pytest.approx(0.0, abs=1e-12)
        assert meta["true_rows"] == trial.causal_starts.size
    # a near-halfway angle snaps to the frozen nearest-canonical rule
    covariates = np.zeros((size, 2), dtype=np.float32)
    angle = math.pi / 8 + 0.01  # just past canonical 3 toward canonical 4
    covariates[:, 0] = 0.01 * math.cos(angle)
    covariates[:, 1] = 0.01 * math.sin(angle)
    direction, _meta = physical.true_direction_for_trial(
        covariates, _trial(50, 350, position=40), config=config)
    assert direction.accepted is True
    assert direction.theta_index in (3, 4)


def test_true_direction_law_applies_the_frozen_unit_and_quality_gates() -> None:
    config = cdm_core.CDMDConfig(support_budget_m=4)
    trial = _trial(50, 350, position=40)
    rows = trial.causal_starts.size
    # SI m/s rows are restored to the frozen cm/s view before the gates:
    # 1e-6 m/s over `rows` bins gives ~rows * 1e-6 * 100 * 0.02 cm << 0.05 cm.
    tiny = np.full((400, 2), 1.0e-6, dtype=np.float32)
    direction, _meta = physical.true_direction_for_trial(tiny, trial, config=config)
    assert direction.accepted is False
    assert direction.reason.value == "low_displacement"
    # the displacement arithmetic is exactly the frozen law's: sum(v)*dt in cm
    strong = np.zeros((400, 2), dtype=np.float32)
    strong[:, 0] = 0.01
    direction, _meta = physical.true_direction_for_trial(strong, trial, config=config)
    assert direction.accepted is True
    expected = float(rows * 0.01 * 100.0 * config.dt)
    assert direction.displacement_norm == pytest.approx(expected, rel=1e-6)
    with pytest.raises(Exception):
        physical.true_direction_for_trial(
            strong, governing_replay.QueryTrial(
                position=40, start_bin=0, stop_bin=10,
                metric_starts=np.asarray([], dtype=np.int64),
                causal_starts=np.asarray([], dtype=np.int64)),
            config=config)


# ---------------------------------------------------------------------------
# 6-8: the anchor adapter and the end-to-end synthetic oracle rollout.
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
    # TRUE covariates: a consistent +x movement every trial (the oracle target)
    behavior = np.zeros((total_bins, 2), dtype=np.float32)
    behavior[:, 0] = 0.01
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


def test_adapter_supplies_the_statistics_view_and_empty_bank_parity() -> None:
    _material, _carrier, anchor = _synthetic_binding()
    adapter = physical.StageOAnchorAdapter(anchor)
    assert adapter.digest == anchor.digest
    assert adapter.per_group is anchor.per_group
    assert np.array_equal(adapter.statistics.counts[0], anchor.per_group[0].counts)
    assert len(adapter.statistics.counts) == anchor.groups.group_count
    parity = stage_o_block_refit.empty_bank_refit_parity(adapter, rho_M=plan.RHO_M)
    # the float64 solve differs from the sealed carrier's float32-quantized
    # coefficients only at the quantization floor (the frozen quantization note)
    assert max(parity["max_abs_coefficient_difference_by_group"]) <= 1.0e-6
    assert max(parity["max_abs_coefficient_difference_by_group"]) > 0.0


def test_oracle_rollout_alpha_zero_is_the_bitwise_f00m_noop() -> None:
    material, carrier, anchor = _synthetic_binding()
    decoder = _StubDecoder(units=material.channel_ids.size)
    config = cdm_core.CDMDConfig(support_budget_m=4)
    frozen = governing_replay.rollout_session_f(
        decoder, material, budget=4,
        spec=governing_replay.CellSpec("F00m", "frozen_t4_per_trial", False),
        anchor=anchor, carrier=carrier, hp=None, config=config,
    )
    oracle = physical.rollout_session_oracle(
        decoder, material, budget=4, anchor=anchor, carrier=carrier,
        alpha_M=0.0, c_M=None, config=config,
    )
    assert oracle["structural_noop"] is True
    assert oracle["final_carrier_sha256"] == oracle["initial_carrier_sha256"]
    assert all(receipt["cb"] == receipt["ca"] for receipt in oracle["receipts"])
    assert oracle["committed_rows"] >= 1  # commits happen, movement is zero
    assert np.array_equal(oracle["prediction"], frozen["prediction"])  # bitwise
    assert physical._causality_chain(oracle["receipts"]) is True
    census = physical.census_from_receipts(oracle["receipts"])
    assert census["accepted_true_directions"] == oracle["committed_rows"]
    assert census["with_complete_window_trials"] == census["accepted_true_directions"]


def test_oracle_rollout_alpha_positive_moves_carrier_and_predictions() -> None:
    material, carrier, anchor = _synthetic_binding()
    config = cdm_core.CDMDConfig(support_budget_m=4)
    baseline = physical.rollout_session_oracle(
        _StubDecoder(units=material.channel_ids.size), material, budget=4,
        anchor=anchor, carrier=carrier, alpha_M=0.0, c_M=None, config=config)
    moved = physical.rollout_session_oracle(
        _StubDecoder(units=material.channel_ids.size), material, budget=4,
        anchor=anchor, carrier=carrier, alpha_M=0.5, c_M=None, config=config)
    assert moved["structural_noop"] is False
    assert moved["committed_rows"] == baseline["committed_rows"]  # alpha-invariant bank
    assert moved["final_carrier_sha256"] != moved["initial_carrier_sha256"]
    assert np.max(moved["movements"]) > 0.0
    assert not np.array_equal(moved["prediction"], baseline["prediction"])
    assert moved["prediction"].shape == baseline["prediction"].shape
    # the same bank produces the same D2 sequence regardless of alpha
    assert moved["d2_values"] == baseline["d2_values"]
    assert moved["bank_digest_chain"] == baseline["bank_digest_chain"]
    labels = physical._leakage_labels("O2m")
    assert labels["target_label_leakage"] is True
    assert labels["deployment_eligible"] is False


def test_oracle_rollout_trust_region_projection_binds_movement() -> None:
    material, carrier, anchor = _synthetic_binding()
    config = cdm_core.CDMDConfig(support_budget_m=4)
    decoder = _StubDecoder(units=material.channel_ids.size)
    unbounded = physical.rollout_session_oracle(
        decoder, material, budget=4, anchor=anchor, carrier=carrier,
        alpha_M=0.5, c_M=None, config=config)
    tiny_c_M = float(min(unbounded["d2_values"])) / 1000.0
    projected = physical.rollout_session_oracle(
        decoder, material, budget=4, anchor=anchor, carrier=carrier,
        alpha_M=0.5, c_M=tiny_c_M, config=config)
    assert projected["projected_count"] == projected["committed_rows"]
    assert projected["movements"] != unbounded["movements"]
    assert max(projected["movements"]) <= max(unbounded["movements"])
    # the D2 sequence is carrier-independent, so projection never changes it
    assert projected["d2_values"] == unbounded["d2_values"]


def test_c_m_calibration_is_the_pooled_within_median() -> None:
    values = [0.2, 0.4, 1.0, 3.0, 0.6]
    assert trust_region.calibrate_c_M(values) == pytest.approx(0.6)
    with pytest.raises(Exception):
        trust_region.calibrate_c_M([])
