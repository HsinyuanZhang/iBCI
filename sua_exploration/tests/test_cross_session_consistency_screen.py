"""Synthetic, CPU-only contract tests for the CSCS cross-session consistency screen.

No NWB file is opened, no checkpoint is loaded, and no model is instantiated here: every
statistical guard is exercised on constructed fixtures.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import cross_session_consistency_screen as cscs


REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 20260813
DIM = 24


def _load_runner_module():
    script = REPO_ROOT / "sua_exploration/scripts/run_cross_session_consistency_screen.py"
    spec = importlib.util.spec_from_file_location("run_cross_session_consistency_screen", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trial(start: int, stop: int, direction_bin: int) -> dict:
    return {
        "start": start,
        "stop": stop,
        "target_dir": direction_bin * (math.pi / 4.0),
        "trial_index": start,
    }


def _session_trials(*, per_direction: int = 12, duration: int = 140, drop_direction: int | None = None) -> list[dict]:
    """30 calibration-prefix trials followed by a balanced post-30 query block."""
    trials = [_trial(index * duration, index * duration + duration, index % cscs.N_DIRECTIONS) for index in range(30)]
    cursor = 30 * duration
    for repeat in range(per_direction):
        for direction in range(cscs.N_DIRECTIONS):
            if direction == drop_direction:
                continue
            trials.append(_trial(cursor, cursor + duration, direction))
            cursor += duration
        _ = repeat
    return trials


# --------------------------------------------------------------------------------------
# Condition grid
# --------------------------------------------------------------------------------------
def test_direction_bin_maps_the_eight_center_out_targets():
    degrees = [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
    bins = sorted(cscs.direction_bin(math.radians(value)) for value in degrees)
    assert bins == list(range(8))


def test_direction_bin_rejects_an_off_grid_angle():
    with pytest.raises(cscs.CscsError):
        cscs.direction_bin(math.radians(30.0))


def test_selection_takes_the_chronologically_first_ten_per_direction():
    trials = _session_trials(per_direction=14)
    selection = cscs.select_condition_trials(trials)
    assert selection["complete_eight_direction_grid"] is True
    assert sorted(selection["selected_query_positions"]) == list(range(8))
    for direction, positions in selection["selected_query_positions"].items():
        assert len(positions) == cscs.TRIALS_PER_DIRECTION
        query = trials[cscs.EVALUATION_START_TRIAL_INDEX :]
        eligible = [i for i, t in enumerate(query) if cscs.direction_bin(t["target_dir"]) == direction]
        assert positions == eligible[: cscs.TRIALS_PER_DIRECTION]


def test_selection_never_reaches_into_the_calibration_prefix():
    trials = _session_trials(per_direction=12)
    selection = cscs.select_condition_trials(trials)
    plan = cscs.condition_window_plan(trials, selection)
    prefix_end = trials[cscs.EVALUATION_START_TRIAL_INDEX - 1]["stop"]
    assert int(plan["starts"].min()) >= prefix_end


def test_short_trials_are_excluded_and_can_make_a_session_incomplete():
    trials = _session_trials(per_direction=12, duration=140)
    for trial in trials[cscs.EVALUATION_START_TRIAL_INDEX :]:
        if cscs.direction_bin(trial["target_dir"]) == 3:
            trial["stop"] = trial["start"] + cscs.MIN_TRIAL_DURATION_BINS - 1
    selection = cscs.select_condition_trials(trials)
    assert selection["complete_eight_direction_grid"] is False
    assert 3 not in selection["selected_query_positions"]
    assert selection["eligible_counts"][3] == 0


def test_missing_direction_is_reported_not_silently_dropped():
    trials = _session_trials(per_direction=12, drop_direction=5)
    selection = cscs.select_condition_trials(trials)
    assert selection["complete_eight_direction_grid"] is False
    assert selection["eligible_counts"][5] == 0
    assert selection["available_directions"] == [0, 1, 2, 3, 4, 6, 7]


def test_window_plan_size_and_containment():
    trials = _session_trials(per_direction=12)
    selection = cscs.select_condition_trials(trials)
    plan = cscs.condition_window_plan(trials, selection)
    assert plan["starts"].size == 8 * cscs.TRIALS_PER_DIRECTION * cscs.N_PHASE_OFFSETS
    assert np.unique(cscs.condition_index(plan["direction_index"], plan["offset_index"])).size == cscs.N_CONDITIONS


def test_condition_means_split_halves_by_rank_parity():
    trials = _session_trials(per_direction=12)
    selection = cscs.select_condition_trials(trials)
    plan = cscs.condition_window_plan(trials, selection)
    values = np.zeros((plan["starts"].size, 3), dtype=np.float64)
    values[:, 0] = plan["trial_rank"]
    means = cscs.condition_means(values, plan)
    assert means["full"].shape == (cscs.N_CONDITIONS, 3)
    # ranks 0..9: even ranks mean 4.0, odd ranks mean 5.0, all ranks mean 4.5
    assert np.allclose(means["half_1"][:, 0], 4.0)
    assert np.allclose(means["half_2"][:, 0], 5.0)
    assert np.allclose(means["full"][:, 0], 4.5)


# --------------------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------------------
def test_fold_assignment_is_deterministic_and_stratified():
    ids = np.arange(cscs.N_CONDITIONS)
    folds = cscs.fold_assignment(ids)
    assert np.array_equal(folds, cscs.fold_assignment(ids))
    directions = ids // cscs.N_PHASE_OFFSETS
    for fold in range(cscs.CV_FOLDS):
        assert np.unique(directions[folds == fold]).size == cscs.N_DIRECTIONS


def test_cross_validated_r2_is_one_for_an_exact_affine_map():
    rng = np.random.default_rng(SEED)
    ids = np.arange(cscs.N_CONDITIONS)
    x = rng.normal(size=(cscs.N_CONDITIONS, 6))
    transform = rng.normal(size=(6, 6))
    y = x @ transform + rng.normal(size=6)
    cross = cscs.cross_validated_map_r2(x, y, cscs.fold_assignment(ids))
    assert cross == pytest.approx(1.0, abs=1e-8)


def test_cross_validated_r2_is_not_vacuously_one_for_unrelated_matrices():
    rng = np.random.default_rng(SEED + 1)
    ids = np.arange(cscs.N_CONDITIONS)
    x = rng.normal(size=(cscs.N_CONDITIONS, 6))
    y = rng.normal(size=(cscs.N_CONDITIONS, 6))
    assert cscs.cross_validated_map_r2(x, y, cscs.fold_assignment(ids)) < 0.2


def test_cross_validated_r2_refuses_a_saturated_fit():
    rng = np.random.default_rng(SEED + 2)
    ids = np.arange(20)
    x = rng.normal(size=(20, 32))
    y = rng.normal(size=(20, 4))
    with pytest.raises(cscs.CscsError):
        cscs.cross_validated_map_r2(x, y, cscs.fold_assignment(ids))


def test_source_pca_is_orthonormal_and_source_only():
    rng = np.random.default_rng(SEED + 3)
    matrices = {f"s{i}": rng.normal(size=(cscs.N_CONDITIONS, DIM)) for i in range(5)}
    basis = cscs.fit_source_pca(matrices, n_components=6, source_sessions=["s0", "s1", "s2"])
    assert basis.components.shape == (6, DIM)
    assert np.allclose(basis.components @ basis.components.T, np.eye(6), atol=1e-9)
    assert basis.fitted_on_sessions == ("s0", "s1", "s2")
    assert basis.project(matrices["s4"]).shape == (cscs.N_CONDITIONS, 6)
    assert 0.0 < basis.as_receipt()["cumulative_explained_variance_ratio"] <= 1.0 + 1e-9


def test_source_pca_rejects_a_missing_roster_session():
    rng = np.random.default_rng(SEED + 4)
    matrices = {"s0": rng.normal(size=(cscs.N_CONDITIONS, DIM))}
    with pytest.raises(cscs.CscsError):
        cscs.fit_source_pca(matrices, n_components=4, source_sessions=["s0", "missing"])


# --------------------------------------------------------------------------------------
# Null construction
# --------------------------------------------------------------------------------------
def _aligned_pair(consistency: float, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(cscs.N_CONDITIONS, DIM))
    rows_a, plan_a = cscs.synthetic_session_representation(
        n_units_seed=0, dimension=DIM, consistency=1.0, noise=noise, shared_basis=shared, rng=rng
    )
    rows_b, plan_b = cscs.synthetic_session_representation(
        n_units_seed=1, dimension=DIM, consistency=consistency, noise=noise, shared_basis=shared, rng=rng
    )
    means_a = cscs.condition_means(rows_a, plan_a)
    means_b = cscs.condition_means(rows_b, plan_b)
    return means_a, means_b


def test_null_is_near_zero_and_raw_exceeds_it_for_an_aligned_pair():
    means_a, means_b = _aligned_pair(consistency=1.0, noise=0.20, seed=SEED + 10)
    result = cscs.symmetric_pair_consistency(
        means_a["full"], means_b["full"], means_a["condition_ids"],
        session_a="A", session_b="B", seed_parts=["unit-test"], n_permutations=8,
    )
    assert result["raw"] > 0.7
    assert abs(result["null"]) < 0.25
    assert result["adv"] > 0.5


def test_unaligned_pair_has_near_zero_null_relative_advantage():
    means_a, means_b = _aligned_pair(consistency=0.0, noise=0.20, seed=SEED + 11)
    result = cscs.symmetric_pair_consistency(
        means_a["full"], means_b["full"], means_a["condition_ids"],
        session_a="A", session_b="B", seed_parts=["unit-test"], n_permutations=8,
    )
    assert result["adv"] < 0.15


def test_null_is_reproducible_and_independent_of_call_order():
    means_a, means_b = _aligned_pair(consistency=1.0, noise=0.3, seed=SEED + 12)
    kwargs = dict(seed_parts=["repro", 42, 5], n_permutations=6)
    first = cscs.pair_consistency(means_a["full"], means_b["full"], means_a["condition_ids"], **kwargs)
    _other = cscs.pair_consistency(means_b["full"], means_a["full"], means_a["condition_ids"], **kwargs)
    second = cscs.pair_consistency(means_a["full"], means_b["full"], means_a["condition_ids"], **kwargs)
    assert first == second


def test_direction_only_null_preserves_the_phase_axis():
    ids = np.arange(cscs.N_CONDITIONS)
    rng = np.random.Generator(np.random.PCG64(7))
    order = cscs.NULL_KINDS["direction_label_permutation"](ids, rng)
    assert np.array_equal(np.sort(order), np.arange(ids.size))
    assert np.array_equal(ids[order] % cscs.N_PHASE_OFFSETS, ids % cscs.N_PHASE_OFFSETS)
    assert not np.array_equal(ids[order] // cscs.N_PHASE_OFFSETS, ids // cscs.N_PHASE_OFFSETS)


def test_shared_condition_restriction_keeps_row_correspondence():
    ids_a = np.arange(cscs.N_CONDITIONS)
    ids_b = ids_a[ids_a // cscs.N_PHASE_OFFSETS != 5]
    keep = cscs.shared_condition_ids(ids_a, ids_b)
    matrix = np.arange(cscs.N_CONDITIONS * 2, dtype=np.float64).reshape(cscs.N_CONDITIONS, 2)
    restricted = cscs.restrict_to_conditions(matrix, ids_a, keep)
    assert restricted.shape[0] == keep.size
    assert np.array_equal(restricted[:, 0], keep * 2.0)


# --------------------------------------------------------------------------------------
# Headroom, association statistics, verdict
# --------------------------------------------------------------------------------------
def test_headroom_is_undefined_for_a_degenerate_ceiling():
    assert cscs.headroom_fraction(0.4, 0.02) is None
    assert cscs.headroom_fraction(0.4, 0.8) == pytest.approx(0.5)


def test_spearman_matches_a_hand_computed_monotone_case():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cscs.spearman(x, x**3) == pytest.approx(1.0)
    assert cscs.spearman(x, -(x**3)) == pytest.approx(-1.0)


def test_partial_spearman_removes_a_pure_confound():
    rng = np.random.default_rng(SEED + 20)
    control = rng.normal(size=200)
    x = control + rng.normal(size=200)
    y = control + rng.normal(size=200)
    # x and y share only the control, so the raw association is real and the partial one is not.
    assert cscs.spearman(x, y) > 0.3
    assert abs(cscs.partial_spearman(x, y, control)) < 0.15


def test_bootstrap_mean_spearman_reports_per_seed_and_an_interval():
    rng = np.random.default_rng(SEED + 21)
    base = rng.normal(size=13)
    consistency = {42: base + 0.05 * rng.normal(size=13), 43: base + 0.05 * rng.normal(size=13)}
    external = {42: base + 0.05 * rng.normal(size=13), 43: base + 0.05 * rng.normal(size=13)}
    result = cscs.bootstrap_mean_spearman_over_seeds(consistency, external, n_resamples=400)
    assert set(result["per_seed_spearman"]) == {"42", "43"}
    assert result["ci_lower_95"] <= result["mean_spearman"] <= result["ci_upper_95"]
    assert result["mean_spearman"] > 0.5


def test_paired_contrast_reports_sign_counts_and_an_interval():
    left = [0.5, 0.6, 0.7, 0.4]
    right = [0.3, 0.65, 0.5, 0.2]
    result = cscs.paired_contrast(left, right, n_resamples=400)
    assert result["positive_count"] == 3 and result["negative_count"] == 1
    assert result["ci_lower_95"] <= result["mean_difference"] <= result["ci_upper_95"]


def test_verdict_rule_fires_on_each_branch_independently():
    at_ceiling = cscs.evaluate_kill_criterion(headroom=0.05, mean_spearman=0.8, spearman_ci_lower=0.4)
    assert at_ceiling["verdict"] == "KILL" and at_ceiling["k1_no_null_space_left"]
    no_link = cscs.evaluate_kill_criterion(headroom=0.6, mean_spearman=0.1, spearman_ci_lower=-0.2)
    assert no_link["verdict"] == "KILL" and no_link["k2_consistency_does_not_predict_external"]
    straddles_zero = cscs.evaluate_kill_criterion(headroom=0.6, mean_spearman=0.5, spearman_ci_lower=-0.01)
    assert straddles_zero["verdict"] == "KILL" and straddles_zero["k2_consistency_does_not_predict_external"]
    proceed = cscs.evaluate_kill_criterion(headroom=0.6, mean_spearman=0.5, spearman_ci_lower=0.1)
    assert proceed["verdict"] == "PROCEED"
    undefined = cscs.evaluate_kill_criterion(headroom=None, mean_spearman=0.5, spearman_ci_lower=0.1)
    assert undefined["verdict"] == "PROCEED" and undefined["k1_undefined_ceiling"]


def test_sealed_formal_test_sessions_are_refused():
    with pytest.raises(cscs.CscsError):
        cscs.assert_sessions_not_sealed(["sub-C_ses-CO-20151113"], label="unit-test")
    cscs.assert_sessions_not_sealed(["sub-C_ses-CO-20131003"], label="unit-test")


# --------------------------------------------------------------------------------------
# Runner contract
# --------------------------------------------------------------------------------------
def test_runner_binds_the_frozen_protocol_and_refuses_cuda_and_user_site():
    runner = _load_runner_module()
    assert runner.PROTOCOL_PATH.is_file()
    assert runner.PROTOCOL_PATH.name == "CROSS_SESSION_CONSISTENCY_SCREEN_PROTOCOL_20260813.md"
    fingerprint = runner.environment_fingerprint()
    assert fingerprint["cuda_visible_devices"] == ""
    assert fingerprint["torch_cuda_available"] is False
    assert fingerprint["pythonnouserssite"] == "1"


def test_runner_yhat_hook_target_is_the_documented_readout_layer():
    runner = _load_runner_module()
    assert runner.YHAT_HOOK_MODULE_PATH == "student.decoder.fc_out"


def test_runner_hashes_its_own_implementation_and_every_sealed_dependency():
    runner = _load_runner_module()
    bindings = runner.implementation_bindings()
    assert set(bindings) == set(runner.IMPLEMENTATION_FILES)
    assert all(len(digest) == 64 for digest in bindings.values())
    for required in (
        "sua_exploration/mc_maze/cross_session_consistency_screen.py",
        "sua_exploration/scripts/run_cross_session_consistency_screen.py",
        "streaming_calibration_exp/src/models/components/streaming_spint.py",
    ):
        assert required in bindings


def test_runner_writes_an_immutable_receipt(tmp_path):
    runner = _load_runner_module()
    target = tmp_path / "receipt.json"
    body, sidecar, digest = runner.write_immutable_json(target, {"schema_version": 1, "value": 3})
    assert body.is_file() and sidecar.is_file()
    assert digest in sidecar.read_text(encoding="utf-8")
    assert (body.stat().st_mode & 0o222) == 0
    with pytest.raises(FileExistsError):
        runner.write_immutable_json(target, {"schema_version": 1, "value": 4})
