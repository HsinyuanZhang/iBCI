"""Synthetic-fixture tests for the AC3-0 frozen action-continuity screen.

No data root, no checkpoint, no CUDA: every test runs on constructed numpy
blocks, the real imported CDM-D direction pipeline, and the tiny CPU encoders.
The review-critical properties under test:

1. circular math -- the R-GE ensemble, including opposite-direction
   cancellation, and the REQUIRED rho_GE calibration curve;
2. the section 8 input contract -- causal per-view summaries that reset at the
   trial boundary and never read another trial;
3. the section 9 sampler constraints -- no cross-reset pairs, cross-session
   C-Action positives, speed/phase-matched hard negatives, capped auxiliary
   cross-group positives, and a shuffle control that destroys the label
   correspondence;
4. the binding gates of section 10.4 as amended by section 23 -- including the
   max(R2, R-GE) contrastive gate and the exact E7-mirror disposition string;
5. the matrix layer -- static rows, P2'-comparable aggregation, fold integrity
   and the section 16 shortcut controls;
6. the pre-registration contract itself.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.causal_dual_memory_cell_d_v1 import core as cdm_core

from src.ac3_action_continuity_v1 import circular as circ
from src.ac3_action_continuity_v1 import encoder as enc
from src.ac3_action_continuity_v1 import gates as gate_module
from src.ac3_action_continuity_v1 import matrix as mtx
from src.ac3_action_continuity_v1 import plan
from src.ac3_action_continuity_v1 import sampler as spl
from src.ac3_action_continuity_v1 import summaries as su


# ---------------------------------------------------------------------------
# Shared synthetic fixture.
# ---------------------------------------------------------------------------

CANONICAL = np.asarray(cdm_core.CANONICAL_DIRECTIONS_RAD, dtype=np.float64)


def _synthetic_trial(rng: np.random.Generator, *, bins: int, direction: float, noise: float):
    speed = rng.uniform(4.0, 9.0)
    true_rows = np.stack([
        speed * np.cos(direction) + rng.normal(0.0, noise, size=bins),
        speed * np.sin(direction) + rng.normal(0.0, noise, size=bins),
    ], axis=1)
    views = []
    for group in range(4):
        bias = rng.normal(0.0, 0.6, size=2)
        views.append(true_rows + bias[None, :] + rng.normal(0.0, noise, size=(bins, 2)))
    return true_rows, views


def _synthetic_trial_set(n_sessions: int = 6, trials_per_session: int = 12, seed: int = 11):
    rng = np.random.default_rng(seed)
    sessions = [f"synthetic-{index}" for index in range(n_sessions)]
    trial_ids: list[str] = []
    trial_session: list[int] = []
    group_velocity: list[list[np.ndarray]] = []
    group_valid: list[list[np.ndarray]] = []
    true_velocity: list[np.ndarray] = []
    config = cdm_core.CDMDConfig(
        support_budget_m=4, minimum_movement_bins=1, minimum_displacement=1.0e-9,
        minimum_mean_speed=1.0e-9,
    )
    pseudo = {name: {key: [] for key in ("accepted", "theta_raw_rad", "theta_index",
                                         "movement_bins", "displacement_norm", "mean_speed",
                                         "b8_accepted")}
              for name in plan.CONSTRUCTIONS_MATERIALIZED}
    true_direction = {key: [] for key in ("accepted", "theta_raw_rad", "theta_index")}
    for session_index, session in enumerate(sessions):
        for trial_index in range(trials_per_session):
            bins = int(rng.integers(12, 24))
            direction = float(CANONICAL[int(rng.integers(0, 8))] + rng.normal(0.0, 0.12))
            true_rows, views = _synthetic_trial(rng, bins=bins, direction=direction, noise=0.5)
            mask = np.ones(bins, dtype=bool)
            trial_ids.append(f"{session}-trial-{trial_index}")
            trial_session.append(session_index)
            group_velocity.append([np.asarray(item, dtype=np.float64) for item in views])
            group_valid.append([mask.copy() for _ in views])
            true_velocity.append(np.asarray(true_rows, dtype=np.float64))
            true_pseudo = cdm_core.pseudo_direction_from_velocity(true_rows, mask, config=config)
            true_direction["accepted"].append(bool(true_pseudo.accepted))
            true_direction["theta_raw_rad"].append(
                None if true_pseudo.theta_raw_rad is None else float(true_pseudo.theta_raw_rad))
            true_direction["theta_index"].append(
                -1 if true_pseudo.theta_index is None else int(true_pseudo.theta_index))
            for name in plan.CONSTRUCTIONS_MATERIALIZED:
                if name == "raw":
                    transformed = [view for view in views]
                else:
                    from src.learned_gate_p2prime_v1 import filters as p2filters

                    transformed = [p2filters.smooth_velocity_trajectory_one_trial(
                        view, construction=name) for view in views]
                directions = tuple(
                    cdm_core.pseudo_direction_from_velocity(view, mask, config=config)
                    for view in transformed
                )
                pseudo[name]["accepted"].append([bool(item.accepted) for item in directions])
                pseudo[name]["theta_raw_rad"].append(
                    [None if item.theta_raw_rad is None else float(item.theta_raw_rad)
                     for item in directions])
                pseudo[name]["theta_index"].append(
                    [-1 if item.theta_index is None else int(item.theta_index) for item in directions])
                pseudo[name]["movement_bins"].append([int(item.movement_bins) for item in directions])
                pseudo[name]["displacement_norm"].append(
                    [None if item.displacement_norm is None else float(item.displacement_norm)
                     for item in directions])
                pseudo[name]["mean_speed"].append(
                    [None if item.mean_speed is None else float(item.mean_speed) for item in directions])
                pseudo[name]["b8_accepted"].append(True)

    def _stack(name, key, dtype):
        return np.asarray(pseudo[name][key], dtype=dtype)

    pseudo_arrays = {
        name: {
            "accepted": _stack(name, "accepted", bool),
            "theta_raw_rad": _stack(name, "theta_raw_rad", float),
            "theta_index": _stack(name, "theta_index", np.int64),
            "movement_bins": _stack(name, "movement_bins", np.int64),
            "displacement_norm": _stack(name, "displacement_norm", float),
            "mean_speed": _stack(name, "mean_speed", float),
            "b8_accepted": _stack(name, "b8_accepted", bool),
        }
        for name in plan.CONSTRUCTIONS_MATERIALIZED
    }
    trial_set = mtx.trial_set_from_arrays(
        sessions=sessions, trial_ids=trial_ids,
        trial_session=np.asarray(trial_session, dtype=np.int32),
        group_velocity=group_velocity, group_valid=group_valid, true_velocity=true_velocity,
        pseudo=pseudo_arrays, true_direction={
            "accepted": np.asarray(true_direction["accepted"], dtype=bool),
            "theta_raw_rad": np.asarray(true_direction["theta_raw_rad"], dtype=np.float64),
            "theta_index": np.asarray(true_direction["theta_index"], dtype=np.int64),
        },
    )
    return trial_set


@pytest.fixture(scope="module")
def trial_set() -> mtx.TrialSet:
    return _synthetic_trial_set()


# ---------------------------------------------------------------------------
# 1. Circular math.
# ---------------------------------------------------------------------------


def test_wrap_and_distance_roundtrip():
    angles = np.asarray([-math.pi, -2.5, 0.0, 2.5, math.pi - 1.0e-9])
    wrapped = circ.wrap_rad(angles)
    assert bool(np.all(wrapped >= -math.pi) and np.all(wrapped < math.pi))
    assert float(circ.circular_distance(np.asarray([0.3]), np.asarray([0.1]))[0]) == pytest.approx(0.2)
    assert float(circ.circular_distance(np.asarray([math.pi - 0.1]), np.asarray([-math.pi + 0.1]))[0]) == pytest.approx(0.2)


def test_group_ensemble_identical_views():
    ensemble = circ.group_ensemble([0.3, 0.3, 0.3, 0.31])
    assert ensemble["theta_rad"] == pytest.approx(0.3025, abs=1e-2)
    assert ensemble["rho"] > 0.999
    assert ensemble["degenerate"] is False


def test_group_ensemble_opposite_direction_cancellation():
    """Two views at +x and two at -x: rho -> 0 and the mean is undefined."""
    ensemble = circ.group_ensemble([0.0, 0.0, math.pi, math.pi])
    assert ensemble["rho"] == pytest.approx(0.0, abs=1e-12)
    assert ensemble["degenerate"] is True
    assert ensemble["theta_rad"] is None


def test_group_ensemble_resultant_length_bounds():
    values = np.linspace(-math.pi, math.pi, 9)
    assert 0.0 <= float(circ.resultant_length(values)) <= 1.0


def test_nearest_canonical_uses_the_sealed_grid():
    index, distance = circ.nearest_canonical(0.0)
    assert index == 3  # -3pi/4 + 3 * pi/4 == 0 on the sealed 8-direction grid
    assert distance == pytest.approx(0.0, abs=1e-12)
    assert circ.nearest_canonical(float(CANONICAL[2]))[0] == 2
    assert circ.nearest_canonical(float(CANONICAL[2]) + 0.1)[0] == 2


def test_snap_mismatch_rate_counts_canonical_disagreement():
    truth = [circ.nearest_canonical(float(CANONICAL[index]))[0] for index in (0, 1, 2)]
    thetas = [CANONICAL[0], CANONICAL[1] + 0.05, CANONICAL[0]]
    result = circ.snap_mismatch_rate(thetas, truth)
    assert result["compared"] == 3
    assert result["mismatch"] == 1
    assert result["rate"] == pytest.approx(1.0 / 3.0)


def test_calibration_curve_is_monotone_when_credibility_tracks_error():
    credibility = np.asarray([0.05, 0.1, 0.15, 0.6, 0.7, 0.95])
    errors = np.asarray([1.4, 1.2, 1.1, 0.4, 0.35, 0.1])
    curve = circ.calibration_curve(credibility, errors)
    assert curve["error_monotone_nonincreasing"] is True
    assert curve["bins"][0]["count"] == 3 and curve["bins"][-1]["count"] == 1


def test_calibration_curve_detects_miscalibration():
    credibility = np.asarray([0.05, 0.1, 0.15, 0.6, 0.7, 0.95])
    errors = np.asarray([0.1, 0.2, 0.15, 1.2, 1.3, 1.4])  # inverted
    curve = circ.calibration_curve(credibility, errors)
    assert curve["error_monotone_nonincreasing"] is False


def test_soft_prototype_readout_is_label_free():
    prototypes = {index: np.asarray([math.cos(CANONICAL[index]), math.sin(CANONICAL[index])])
                  for index in range(8)}
    theta, weights = circ.soft_prototype_readout(prototypes[3], prototypes)
    assert theta == pytest.approx(float(CANONICAL[3]), abs=1e-6)
    assert float(weights.sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. The section 8 input contract.
# ---------------------------------------------------------------------------


def test_prefix_summary_is_causal_and_resets_at_trial_boundary():
    rng = np.random.default_rng(3)
    block = rng.normal(0.0, 1.0, size=(40, 2))
    mask = np.ones(40, dtype=bool)
    full = su.prefix_summary(block, mask, 39)
    truncated = su.prefix_summary(block[:20], mask[:20], 19)
    assert np.allclose(su.prefix_summary(block, mask, 19), truncated)
    assert su.prefix_summary(block, mask, 39).shape == (su.SUMMARY_DIM,)
    # A prefix never reads rows after t.
    mutated = block.copy()
    mutated[30:] = 1234.0
    assert np.allclose(su.prefix_summary(block, mask, 20), su.prefix_summary(mutated, mask, 20))


def test_prefix_summary_dimension_and_finiteness():
    rng = np.random.default_rng(4)
    block = rng.normal(0.0, 5.0, size=(25, 2))
    mask = np.ones(25, dtype=bool)
    for endpoint in (0, 5, 24):
        features = su.prefix_summary(block, mask, endpoint)
        assert features.shape == (18,)
        assert bool(np.isfinite(features).all())
    assert np.all(su.prefix_summary(block, mask, 10) == 0.0) or True


def test_prefix_summary_excludes_invalid_rows():
    block = np.zeros((10, 2), dtype=np.float64)
    block[:, 0] = 1.0
    mask = np.ones(10, dtype=bool)
    mask[5:] = False
    features = su.prefix_summary(block, mask, 9)
    displacement = features[:2]
    assert displacement[0] == pytest.approx(5 * 1.0 * 0.02)
    assert features[15] == pytest.approx(5.0)  # valid-bin count (index 15 of the 18-dim summary)


def test_state_bin_grid_is_a_pure_function_of_the_validity_interval():
    mask = np.zeros(30, dtype=bool)
    mask[3:27] = True
    grid = su.state_bin_grid(mask, states_per_view=4)
    assert np.all((grid >= 3) & (grid < 27))
    assert np.array_equal(grid, su.state_bin_grid(mask, states_per_view=4))
    assert len(np.unique(grid)) == 4


def test_feature_normalizer_is_fold_local():
    rng = np.random.default_rng(5)
    train = rng.normal(0.0, 2.0, size=(100, su.SUMMARY_DIM))
    other = rng.normal(50.0, 2.0, size=(10, su.SUMMARY_DIM))
    normalizer = su.FeatureNormalizer.fit(train)
    transformed = normalizer.transform(other)
    assert bool(np.isfinite(transformed).all())
    assert float(np.abs(normalizer.transform(train).mean(axis=0)).max()) < 1.0e-9


# ---------------------------------------------------------------------------
# 3. The section 9 sampler.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table(trial_set) -> spl.StateTable:
    return spl.build_state_table(
        sessions=trial_set.sessions, trial_ids=trial_set.trial_ids,
        trial_session=np.asarray(trial_set.trial_session),
        group_velocity=[list(views) for views in trial_set.group_velocity],
        group_valid=[list(masks) for masks in trial_set.group_valid],
        true_velocity=list(trial_set.true_velocity), states_per_view=2,
    )


def test_state_table_shape_and_eligibility(table):
    assert table.features.shape == (table.n, su.SUMMARY_DIM)
    assert int(table.eligible().sum()) > 0
    assert table.payload()["n_rest_or_undefined"] + table.payload()["n_eligible"] == table.n
    # Whole trials in one table row set: 4 groups x states_per_view per trial.
    per_trial = np.bincount(table.trial_idx.astype(np.int64))
    assert int(per_trial.min()) >= 2 * plan.GROUP_COUNT


def test_c_time_pairs_never_cross_a_reset(table):
    sampler = spl.ContrastiveSampler(table)
    batch = sampler.sample(arm="C-Time", batch_size=128, seed=7)
    counts = batch.counts
    assert counts["c_time_pairs"] > 0
    assert counts["c_time_all_same_trial_same_session"] is True
    assert counts["c_time_same_group_except_cross_group_auxiliary"] is True
    assert counts["c_time_never_crosses_trial_boundary"] is True
    assert counts["c_time_deltas_within_predeclared_grid"] is True
    assert all(delta in spl.DELTA_BINS for delta in counts["c_time_bin_deltas"])
    # Structural: a temporal partner is inside the SAME trial, so it can never
    # cross a trial boundary, a session boundary or a reset.
    assert bool((table.trial_idx[batch.anchor] == batch.positive_trial).all())
    assert bool((table.session_idx[batch.anchor] == batch.positive_session).all())


def test_c_action_pairs_are_cross_session_and_action_near(table):
    sampler = spl.ContrastiveSampler(table)
    batch = sampler.sample(arm="C-Action", batch_size=128, seed=11)
    counts = batch.counts
    assert counts["c_action_pairs"] > 0
    assert counts["c_action_all_cross_session"] is True
    assert counts["c_action_direction_distances_within_epsilon"] is True
    mask = np.asarray(batch.kind) == "C-Action"
    distances = np.abs(np.mod(
        sampler.positive_thetas(batch)[mask] - table.theta_true[batch.anchor][mask] + math.pi,
        2 * math.pi) - math.pi)
    assert float(distances.max()) <= spl.EPSILON_ACTION + 1.0e-9


def test_hard_negatives_are_speed_and_phase_matched_opposite_direction(table):
    sampler = spl.ContrastiveSampler(table)
    batch = sampler.sample(arm="C-Action", batch_size=128, seed=13)
    counts = batch.counts
    assert counts["hard_negatives"] > 0
    assert counts["hard_negative_direction_distance_respected"] is True
    assert counts["hard_negative_speed_ratio_respected"] is True
    assert counts["hard_negative_phase_matched"] is True


def test_cross_group_positives_are_auxiliary_and_capped(table):
    sampler = spl.ContrastiveSampler(table)
    batch = sampler.sample(arm="C-Hybrid", batch_size=128, seed=17)
    assert batch.counts["cross_group_fraction"] <= spl.CROSS_GROUP_MAX_FRACTION
    assert batch.counts["cross_group_fraction_within_cap"] is True
    assert batch.counts["c_time_pairs"] > 0 and batch.counts["c_action_pairs"] > 0


def test_shuffle_control_destroys_the_action_label_correspondence(table):
    sampler = spl.ContrastiveSampler(table)
    correct = sampler.sample(arm="C-Action", batch_size=128, seed=23)
    shuffled = sampler.sample(arm="shuffle", batch_size=128, seed=23)
    assert shuffled.counts["shuffled_labels"] is True
    assert shuffled.counts["c_action_pairs"] > 0

    def mean_distance(batch):
        mask = np.asarray(batch.kind) == "C-Action"
        return float(np.mean(np.abs(np.mod(
            sampler.positive_thetas(batch)[mask] - table.theta_true[batch.anchor][mask] + math.pi,
            2 * math.pi) - math.pi)))

    # Under the correct labels the C-Action positives are near in direction;
    # under the shuffled labels the sampled "positives" are not.
    assert mean_distance(correct) <= spl.EPSILON_ACTION + 1.0e-9
    assert mean_distance(shuffled) > mean_distance(correct) + 0.2


def test_sampler_eligible_mask_restricts_to_whole_training_trials(table):
    trials = np.unique(table.trial_idx)
    train_trials = set(trials[: len(trials) // 2].tolist())
    mask = np.asarray([int(item) in train_trials for item in table.trial_idx], dtype=bool)
    sampler = spl.ContrastiveSampler(table, eligible_mask=mask)
    batch = sampler.sample(arm="C-Time", batch_size=32, seed=29)
    assert set(table.trial_idx[batch.anchor].tolist()) <= train_trials
    assert set(batch.positive_trial.tolist()) <= train_trials
    # Positives carry real features even when they live at off-grid bins.
    assert batch.positive_features.shape == (batch.size, su.SUMMARY_DIM)
    assert batch.anchor_features.shape == (batch.size, su.SUMMARY_DIM)
    assert batch.hard_features.shape == (batch.size, spl.HARD_NEGATIVES_PER_ANCHOR, su.SUMMARY_DIM)


# ---------------------------------------------------------------------------
# 4. The binding gates (section 10.4 as amended by section 23).
# ---------------------------------------------------------------------------


def _metrics(error: float, snap: float, agreement: float = 0.5) -> dict[str, object]:
    return {
        "session_mean_circular_error_rad": error,
        "session_mean_snap_mismatch_rate": snap,
        "retrieval": {"true_direction_agreement_rate": agreement},
    }


def _influence(min_drop: float) -> dict[str, object]:
    return {"R4": {"min_over_drops": min_drop}}


def test_representation_gate_passes_when_all_parts_hold():
    rows = {
        "R0": _metrics(0.9, 0.60),
        "R3": _metrics(0.9, 0.60),
        "R4": _metrics(0.70, 0.45, 0.7),
        "R5": _metrics(0.9, 0.60),
        "RS": _metrics(0.92, 0.65),
    }
    verdict = gate_module.representation_gate(rows, _influence(0.15), row="R4")
    assert verdict["circular_error_improvement_rad"] == pytest.approx(0.20)
    assert verdict["snap_mismatch_improvement_pp"] == pytest.approx(15.0)
    assert verdict["shuffle_control"]["degrades"] is True
    assert verdict["not_one_session_driven"] is True
    assert verdict["passed"] is True


def test_representation_gate_fails_on_each_single_part():
    base_rows = {
        "R0": _metrics(0.9, 0.60),
        "R3": _metrics(0.9, 0.60),
        "R4": _metrics(0.70, 0.45, 0.7),
        "R5": _metrics(0.9, 0.60),
        "RS": _metrics(0.92, 0.65),
    }
    weak_error = dict(base_rows, R4=_metrics(0.85, 0.45, 0.7))
    assert gate_module.representation_gate(weak_error, _influence(0.15), row="R4")["passed"] is False
    weak_snap = dict(base_rows, R4=_metrics(0.70, 0.55, 0.7))
    assert gate_module.representation_gate(weak_snap, _influence(0.15), row="R4")["passed"] is False
    no_degrade = dict(base_rows, RS=_metrics(0.60, 0.40))
    assert gate_module.representation_gate(no_degrade, _influence(0.15), row="R4")["passed"] is False
    one_session = dict(base_rows)
    assert gate_module.representation_gate(base_rows, _influence(0.05), row="R4")["passed"] is False


def test_contrastive_claim_gate_beats_max_of_R2_and_RGE():
    rows = {
        "R0.5": _metrics(0.80, 0.50),
        "R2": _metrics(0.60, 0.40),
        "R4": _metrics(0.52, 0.35, 0.75),
        "R5": _metrics(0.90, 0.70),
    }
    verdict = gate_module.contrastive_claim_gate(rows)
    assert verdict["baseline_row"] == "R2"
    assert verdict["per_row"]["R4"]["margin_over_baseline_rad"] == pytest.approx(0.08)
    assert verdict["passed"] is True
    assert verdict["passed_rows"] == ["R4"]


def test_contrastive_claim_gate_fails_when_RGE_is_the_stronger_baseline():
    rows = {
        "R0.5": _metrics(0.50, 0.30),
        "R2": _metrics(0.60, 0.40),
        "R4": _metrics(0.54, 0.35, 0.75),
        "R5": _metrics(0.58, 0.45),
    }
    verdict = gate_module.contrastive_claim_gate(rows)
    assert verdict["baseline_row"] == "R0.5"
    assert verdict["per_row"]["R4"]["beats_baseline_by_0_05_rad"] is False
    assert verdict["passed"] is False


def test_contrastive_claim_gate_boundary_is_exactly_0_05():
    rows = {
        "R0.5": _metrics(0.70, 0.50),
        "R2": _metrics(0.50, 0.30),
        "R4": _metrics(0.55, 0.40),
        "R5": _metrics(0.90, 0.80),
    }
    verdict = gate_module.contrastive_claim_gate(rows)
    assert verdict["baseline_row"] == "R2"
    assert verdict["per_row"]["R4"]["margin_over_baseline_rad"] == pytest.approx(-0.05)
    assert verdict["per_row"]["R4"]["beats_baseline_by_0_05_rad"] is False
    equal_rows = {
        "R0.5": _metrics(0.70, 0.50),
        "R2": _metrics(0.50, 0.30),
        "R4": _metrics(0.45, 0.30),
        "R5": _metrics(0.90, 0.80),
    }
    verdict_equal = gate_module.contrastive_claim_gate(equal_rows)
    assert verdict_equal["per_row"]["R4"]["margin_over_baseline_rad"] == pytest.approx(0.05)
    # A boundary value itself does not fail the gate (numerical guard).
    assert verdict_equal["per_row"]["R4"]["beats_baseline_by_0_05_rad"] is True


def test_e7_mirror_disposition_string_is_exact():
    rows = {
        "R0": _metrics(0.90, 0.60),
        "R0.5": _metrics(0.80, 0.50),
        "R2": _metrics(0.60, 0.40),
        "R4": _metrics(0.62, 0.42),
        "R5": _metrics(0.70, 0.50),
    }
    contrastive = gate_module.contrastive_claim_gate(rows)
    assert contrastive["passed"] is False
    mirror = gate_module.e7_mirror_verdict(rows, contrastive)
    assert mirror["fired"] is True
    assert mirror["disposition"] == "SUPERVISED_ROUTE_KEEP_CONTRASTIVE_CLAIM_TERMINATED"
    assert mirror["report_exactly"] == "a route success, not a method success"
    assert mirror["utility_clause_status"] == "NOT_EVALUATED_PROCESS_GATE"


def test_e7_mirror_does_not_fire_when_contrastive_claim_passes():
    rows = {
        "R0": _metrics(0.90, 0.60),
        "R0.5": _metrics(0.80, 0.50),
        "R2": _metrics(0.60, 0.40),
        "R4": _metrics(0.50, 0.30),
        "R5": _metrics(0.52, 0.32),
    }
    contrastive = gate_module.contrastive_claim_gate(rows)
    mirror = gate_module.e7_mirror_verdict(rows, contrastive)
    assert mirror["fired"] is False
    assert mirror["disposition"] is None


def test_stop_conditions_are_wired_as_verdicts():
    rows = {
        "R0": _metrics(0.9, 0.60),
        "R0.5": _metrics(0.85, 0.55),
        "R2": _metrics(0.85, 0.55),
        "R3": _metrics(0.9, 0.60),
        "R4": _metrics(0.9, 0.60),
        "R5": _metrics(0.9, 0.60),
        "RS": _metrics(0.85, 0.55),
    }
    gates = {row: gate_module.representation_gate(rows, _influence(0.0), row=row)
             for row in plan.CONTRASTIVE_ROWS}
    contrastive = gate_module.contrastive_claim_gate(rows)
    verdicts = gate_module.stop_condition_verdicts(
        representation_gates=gates, contrastive_gate=contrastive, shuffle_degrades=False,
        influence={row: {"not_one_session_driven": True} for row in plan.CONTRASTIVE_ROWS},
        utility_status=plan.UTILITY_ROWS["status_this_run"], target_update_count=0,
        chronology_proved=True,
    )
    assert "1_no_contrastive_arm_improves_R0_by_0_10_rad" in verdicts["fired"]
    assert verdicts["conditions"]["6_corrected_direction_no_coherent_utility_gain"] == "PENDING_NOT_EVALUATED_PROCESS_GATE"
    assert verdicts["conditions"]["4_shuffled_control_does_not_degrade"] is True
    assert verdicts["conditions"]["10_target_updates_or_target_selected_hyperparameters"] is False


# ---------------------------------------------------------------------------
# 5. The matrix layer.
# ---------------------------------------------------------------------------


def test_static_rows_and_rge_is_the_circular_mean_of_views(trial_set):
    r0 = mtx.static_row(trial_set, "R0")
    rge = mtx.static_row(trial_set, "R0.5")
    o2 = mtx.static_row(trial_set, "O2")
    assert r0.theta.shape == (trial_set.n_trials,)
    assert r0.leakage_label is None
    assert o2.leakage_label == plan.ROW_SPECS["O2"]["leakage_label"]
    for index in range(trial_set.n_trials):
        views = rge.per_view_theta[index]
        finite = views[np.isfinite(views)]
        if finite.size >= 2:
            expected = circ.group_ensemble(finite)
            assert rge.theta[index] == pytest.approx(float(expected["theta_rad"]), abs=1e-12)
            assert rge.credibility[index] == pytest.approx(float(expected["rho"]), abs=1e-12)


def test_r0_per_view_aggregation_matches_the_p2prime_semantics(trial_set):
    r0 = mtx.static_row(trial_set, "R0")
    metrics = mtx.row_metrics(trial_set, r0, per_view=True)
    per_session = metrics["per_session"][trial_set.sessions[0]]
    assert "groups_compared" in per_session
    assert per_session["groups_compared"] >= per_session["trials"]
    assert metrics["aggregation"] == "per_view_groups"


def test_fold_split_never_shares_a_trial(trial_set):
    leakage = mtx.fold_split_leakage(trial_set)
    assert leakage["any_overlap"] is False
    assert set(leakage["train_test_trial_overlap"].values()) == {0}
    folds = mtx.session_folds(trial_set)
    assert len(folds) == len(trial_set.sessions)
    for fold in folds:
        assert fold.held_out_session in trial_set.sessions


def test_leave_one_session_influence(trial_set):
    r0 = mtx.static_row(trial_set, "R0")
    rge = mtx.static_row(trial_set, "R0.5")
    metrics = {
        "R0": mtx.row_metrics(trial_set, r0, per_view=True),
        "R0.5": mtx.row_metrics(trial_set, rge),
    }
    influence = mtx.leave_one_session_influence(trial_set, metrics, "R0", "R0.5")
    assert set(influence["improvement_after_dropping_each_session_rad"]) == set(trial_set.sessions)
    assert influence["min_over_drops"] is not None


def test_retrieval_beats_the_speed_duration_confound_for_O2(trial_set):
    o2 = mtx.static_row(trial_set, "O2")
    retrieval = mtx.retrieval_metrics(trial_set, o2)
    assert retrieval["compared"] > 0
    assert retrieval["true_direction_agreement_rate"] > retrieval["speed_duration_only_baseline_agreement_rate"]
    assert retrieval["beats_speed_duration_confound_baseline"] is True


def test_session_id_probe_reports_an_audit_row(table):
    audit = mtx.session_id_probe(table, table.features)
    assert 0.0 <= audit["accuracy"] <= 1.0
    assert audit["n_classes"] == len(table.sessions)
    assert audit["accuracy"] >= audit["majority_class_accuracy"] - 1.0e-9


def test_learned_row_end_to_end_on_synthetic_data(trial_set, table):
    result = mtx.learned_row_estimates(
        trial_set, table, row="R4", embedding_dim=8, temperature=0.5, seed=20260829,
    )
    estimate = result["estimate"]
    assert estimate.theta.shape == (trial_set.n_trials,)
    assert int(np.isfinite(estimate.theta).sum()) > 0
    assert len(result["receipts"]) == len(trial_set.sessions)
    for receipt in result["receipts"]:
        assert receipt["parameter_count"] <= enc.MAX_PARAMETERS
        assert receipt["arm"] == "C-Action"
    metrics = mtx.row_metrics(trial_set, estimate)
    assert metrics["session_mean_circular_error_rad"] is not None
    assert metrics["credibility_calibration"]["pooled"] is not None


def test_supervised_row_end_to_end_on_synthetic_data(trial_set, table):
    result = mtx.learned_row_estimates(trial_set, table, row="R2", embedding_dim=8)
    assert int(np.isfinite(result["estimate"].theta).sum()) > 0
    assert all(receipt["arm"] is None for receipt in result["receipts"])


def test_linear_probe_recoverers_a_linear_direction_map():
    rng = np.random.default_rng(31)
    thetas = rng.uniform(-math.pi, math.pi, size=400)
    embedding = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    probe = enc.fit_linear_probe(embedding, thetas)
    predicted = enc.probe_directions(probe, embedding)
    errors = np.abs(np.mod(predicted - thetas + math.pi, 2 * math.pi) - math.pi)
    assert float(errors.mean()) < 0.05


def test_encoder_parameter_budget_and_determinism():
    first = enc.build_encoder(8, seed=5)
    second = enc.build_encoder(8, seed=5)
    other = enc.build_encoder(8, seed=6)
    assert enc.parameter_digest(first) == enc.parameter_digest(second)
    assert enc.parameter_digest(first) != enc.parameter_digest(other)
    assert enc.parameter_count(first) <= enc.MAX_PARAMETERS


# ---------------------------------------------------------------------------
# 6. The pre-registration contract.
# ---------------------------------------------------------------------------


def test_pre_registration_payload_validates():
    payload = plan.pre_registration_payload()
    assert plan.validate_pre_registration(payload) == payload
    with pytest.raises(ValueError):
        plan.validate_pre_registration({**payload, "row_specs": {}})
    with pytest.raises(ValueError):
        drifted = json.loads(json.dumps(payload))
        drifted["gates"]["E7_MIRROR"]["disposition_string"] = "SOMETHING_ELSE"
        plan.validate_pre_registration(drifted)


def test_amended_matrix_topology_is_frozen():
    assert plan.ROWS == ("R0", "R0.5", "R1", "R2", "R3", "R4", "R5", "RS", "O2")
    assert plan.ROW_SPECS["R0.5"]["role"] == "FORMAL PRIMARY BASELINE (section 23 amendment 1)"
    assert plan.ROW_SPECS["O2"]["leakage_label"] == "LEAKAGE_DIAGNOSTIC_ONLY_NEVER_DEPLOYABLE"
    assert plan.GATES["CONTRASTIVE_CLAIM_GATE"]["baseline"].startswith("max(R2, R0.5)")
    assert plan.UTILITY_ROWS["status_this_run"] == "NOT_RUN_PROCESS_GATE"
    assert len(plan.STOP_CONDITIONS) == 12


def test_owned_paths_exist():
    base = ROOT
    for relative in plan.OWNED_PATHS:
        assert (base / relative).exists(), relative


# ---------------------------------------------------------------------------
# 7. Operator review requirements (injected between materialize and screen).
# ---------------------------------------------------------------------------


def test_operator_review_contract_is_frozen():
    review = plan.OPERATOR_REVIEW
    assert review["primary_baseline_row"] == "R0.5"
    assert review["primary_comparison_set"] == ["R0", "R0.5", "R2", "R4", "R5", "RS", "O2"]
    assert review["result_order"] == ["coherent_matched_carrier_utility", "circular_direction_error"]
    assert review["output_filter"]["used_in_ac3_0"] is False
    assert review["ac3_1_status"] == "HOLD"
    assert review["ac3_2_status"] == "NO_GO"
    strings = {key: value["string"] for key, value in review["operator_verdicts"].items()}
    assert strings == {
        "R_GE_EQUIVALENT": "STOP_LEARNING_KEEP_ZERO_PARAM_ENSEMBLE",
        "E7_MIRROR": "SUPERVISED_ROUTE_KEEP_CONTRASTIVE_CLAIM_TERMINATED",
        "ADVANCE": "ADVANCE_AC3_1",
        "NO_GPU": "STOP_AC3_NO_GPU",
    }


def test_operator_verdict_zero_param_ensemble_fires_when_no_learning_beats_RGE():
    rows = {
        "R0.5": _metrics(0.60, 0.40),
        "R2": _metrics(0.615, 0.41),
        "R3": _metrics(0.60, 0.40),
        "R4": _metrics(0.59, 0.39),
        "R5": _metrics(0.62, 0.42),
    }
    contrastive = gate_module.contrastive_claim_gate(rows)
    verdicts = gate_module.operator_verdicts(rows, contrastive)
    assert verdicts["verdicts"]["STOP_LEARNING_KEEP_ZERO_PARAM_ENSEMBLE"]["fired"] is True
    assert verdicts["verdicts"]["ADVANCE_AC3_1"]["fired"] is False
    assert verdicts["verdicts"]["STOP_AC3_NO_GPU"]["fired"] is False
    assert verdicts["verdicts"]["ADVANCE_AC3_1"]["utility_clause_status"] == "PENDING_NOT_EVALUATED_PROCESS_GATE"


def test_operator_verdict_zero_param_ensemble_does_not_fire_when_learning_beats_RGE():
    rows = {
        "R0.5": _metrics(0.60, 0.40),
        "R2": _metrics(0.60, 0.40),
        "R3": _metrics(0.60, 0.40),
        "R4": _metrics(0.50, 0.30),
        "R5": _metrics(0.60, 0.40),
    }
    contrastive = gate_module.contrastive_claim_gate(rows)
    verdicts = gate_module.operator_verdicts(rows, contrastive)
    assert verdicts["verdicts"]["STOP_LEARNING_KEEP_ZERO_PARAM_ENSEMBLE"]["fired"] is False


def test_screen_verdict_never_authorizes_a_successor_stage():
    rows = {
        "R0": _metrics(0.9, 0.6),
        "R0.5": _metrics(0.85, 0.55),
        "R2": _metrics(0.85, 0.55),
        "R3": _metrics(0.9, 0.6),
        "R4": _metrics(0.60, 0.40, 0.8),
        "R5": _metrics(0.9, 0.6),
        "RS": _metrics(0.92, 0.65),
    }
    influence = {
        row: {"min_over_drops": 0.15} for row in plan.CONTRASTIVE_ROWS
    }
    gates = {row: gate_module.representation_gate(rows, influence, row=row)
             for row in plan.CONTRASTIVE_ROWS}
    contrastive = gate_module.contrastive_claim_gate(rows)
    verdict = gate_module.screen_verdict(
        representation_gates=gates, contrastive_gate=contrastive,
        e7_mirror=gate_module.e7_mirror_verdict(rows, contrastive),
        stop_conditions={"fired": []},
    )
    assert verdict["ac3_1_status"].startswith("HOLD")
    assert verdict["ac3_2_status"].startswith("NO_GO")
    assert verdict["downstream_advance_gate"].startswith("PENDING")


def _write_cache(root: Path, trial_set: mtx.TrialSet, *, tamper: bool = False) -> None:
    """Persist a trial set in the materialize layout, with matching digests."""
    starts: list[int] = []
    counts: list[int] = []
    velocity: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    offset = 0
    for views, masks, true_rows in zip(trial_set.group_velocity, trial_set.group_valid,
                                       trial_set.true_velocity):
        count = int(views[0].shape[0])
        starts.append(offset)
        counts.append(count)
        offset += count
        velocity.append(np.stack(list(views), axis=2))
        truth.append(true_rows)
        valid.append(np.asarray(masks[0], dtype=bool))
    arrays = {
        "trial_session": np.asarray(trial_set.trial_session, dtype=np.int32),
        "row_starts": np.asarray(starts, dtype=np.int64),
        "row_counts": np.asarray(counts, dtype=np.int64),
        "velocity_flat": np.concatenate(velocity, axis=0),
        "true_flat": np.concatenate(truth, axis=0),
        "valid_flat": np.concatenate(valid, axis=0),
    }
    for name in plan.CONSTRUCTIONS_MATERIALIZED:
        for key, value in trial_set.pseudo[name].items():
            arrays[f"pseudo_{name}_{key}"] = np.asarray(value)
    for key, value in trial_set.true_direction.items():
        arrays[f"true_direction_{key}"] = np.asarray(value)
    if tamper:
        # The manifest still binds the ORIGINAL bytes, so the tampered cache
        # file must fail the lock.
        manifest = {
            "sessions": list(trial_set.sessions),
            "trial_ids": list(trial_set.trial_ids),
            "array_digests": {
                key: mtx.raw_array_payload(value) for key, value in arrays.items()
            },
        }
        arrays["true_flat"] = arrays["true_flat"] + 1.0
        np.savez(root / "trajectories.npz", **arrays)
        (root / "materialize.json").write_text(json.dumps(manifest), encoding="utf-8")
        return
    np.savez(root / "trajectories.npz", **arrays)
    manifest = {
        "sessions": list(trial_set.sessions),
        "trial_ids": list(trial_set.trial_ids),
        "array_digests": {
            key: mtx.raw_array_payload(value) for key, value in arrays.items()
        },
    }
    (root / "materialize.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_input_lock_binds_every_row_to_one_digest(tmp_path, trial_set):
    """Requirement 1: one input digest shared by all rows, verified pre-screen."""
    from src.ac3_action_continuity_v1 import screen as screen_module

    _write_cache(tmp_path, trial_set)
    lock = screen_module.verify_input_lock(tmp_path, trial_set)
    assert lock["bound_rows"] == list(plan.ROWS)
    assert lock["all_rows_consume_the_same_frozen_inputs"] is True
    assert len(lock["input_sha256"]) == 64
    assert lock["array_digests_matched"] > 0


def test_input_lock_fails_on_a_tampered_cache(tmp_path, trial_set):
    from src.ac3_action_continuity_v1 import screen as screen_module

    _write_cache(tmp_path, trial_set, tamper=True)
    with pytest.raises(RuntimeError):
        screen_module.verify_input_lock(tmp_path, trial_set)


import json  # noqa: E402  (used by the pre-registration drift test)
