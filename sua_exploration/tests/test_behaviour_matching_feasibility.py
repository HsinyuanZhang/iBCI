"""Tests for the 2026-08-14 behaviour-matching feasibility estimator.

Everything here runs on synthetic arrays: the estimator module never opens an NWB.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.behaviour_matching import core, loaders


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# provenance / determinism
# ---------------------------------------------------------------------------


def test_protocol_document_exists() -> None:
    assert (REPO_ROOT / core.PROTOCOL_DOCUMENT).is_file()


def test_derived_seed_is_deterministic_and_role_sensitive() -> None:
    assert core.derived_seed("a", "b", "query") == core.derived_seed("a", "b", "query")
    assert core.derived_seed("a", "b", "query") != core.derived_seed("a", "b", "reference")
    assert core.derived_seed("a", "b", "query") != core.derived_seed("a", "c", "query")


def test_canonical_json_bytes_reject_nan() -> None:
    with pytest.raises(ValueError):
        core.canonical_json_bytes({"value": float("nan")})


# ---------------------------------------------------------------------------
# section 4: splits and subsampling
# ---------------------------------------------------------------------------


def test_chronological_halves_are_disjoint_and_ordered() -> None:
    reference, query = core.chronological_halves(101)
    assert reference.size == 50 and query.size == 51
    assert reference.max() < query.min()
    assert np.intersect1d(reference, query).size == 0


def test_subsample_is_nested_across_sizes() -> None:
    pool = np.arange(5000, dtype=np.int64)
    small = core.subsample(pool, 64, cohort="c", session="s", role="reference")
    large = core.subsample(pool, 1024, cohort="c", session="s", role="reference")
    assert small.size == 64 and large.size == 1024
    assert np.isin(small, large).all()


def test_subsample_truncates_to_pool_and_stays_deterministic() -> None:
    pool = np.arange(20, dtype=np.int64)
    drawn = core.subsample(pool, 1024, cohort="c", session="s", role="reference")
    assert np.array_equal(drawn, pool)
    again = core.subsample(np.arange(500, dtype=np.int64), 32, cohort="c", session="s", role="query")
    first = core.subsample(np.arange(500, dtype=np.int64), 32, cohort="c", session="s", role="query")
    assert np.array_equal(again, first)


# ---------------------------------------------------------------------------
# section 5: normalization
# ---------------------------------------------------------------------------


def test_grouped_zscore_shares_statistics_within_a_group() -> None:
    rng = np.random.default_rng(0)
    raw = np.concatenate(
        [rng.normal(3.0, 2.0, size=(400, 1)), rng.normal(-1.0, 5.0, size=(400, 1))], axis=1
    )
    windows = np.tile(raw, (1, 4))
    groups = loaders.representation_groups("win100")[:8]
    mean, sd = core.grouped_zscore_stats([windows], groups)
    assert np.allclose(mean[groups == 0], mean[groups == 0][0])
    assert np.allclose(sd[groups == 1], sd[groups == 1][0])
    assert mean[0] == pytest.approx(raw[:, 0].mean(), abs=1e-9)
    assert sd[1] == pytest.approx(raw[:, 1].std(), abs=1e-9)


def test_grouped_zscore_rejects_a_constant_dimension() -> None:
    constant = np.ones((32, 2), dtype=np.float64)
    with pytest.raises(core.BehaviourMatchingError):
        core.grouped_zscore_stats([constant], np.arange(2, dtype=np.int64))


def test_apply_zscore_rejects_nonfinite() -> None:
    values = np.array([[1.0, np.inf]])
    with pytest.raises(core.BehaviourMatchingError):
        core.apply_zscore(values, np.zeros(2), np.ones(2))


# ---------------------------------------------------------------------------
# nearest neighbours
# ---------------------------------------------------------------------------


def test_nn_distances_match_brute_force() -> None:
    rng = np.random.default_rng(1)
    query = rng.normal(size=(37, 5))
    reference = rng.normal(size=(211, 5))
    expected = np.min(np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2), axis=1)
    assert np.allclose(core.nn_distances(query, reference), expected, atol=1e-9)


def test_nn_distances_are_block_size_invariant() -> None:
    rng = np.random.default_rng(2)
    query = rng.normal(size=(20, 3))
    reference = rng.normal(size=(500, 3))
    assert np.allclose(
        core.nn_distances(query, reference, block=7), core.nn_distances(query, reference, block=4096)
    )


def test_nn_distance_is_zero_when_reference_contains_the_query() -> None:
    # The blocked |q|^2 + |r|^2 - 2 q.r form loses absolute precision exactly at a coincident
    # pair; the residual error is ~1e-7 in distance, six orders below any residual this
    # protocol reports, and the protocol's temporally disjoint halves make coincidence
    # vanishingly unlikely in the first place.
    rng = np.random.default_rng(3)
    query = rng.normal(size=(11, 4))
    reference = np.concatenate([rng.normal(size=(50, 4)), query], axis=0)
    assert np.allclose(core.nn_distances(query, reference), 0.0, atol=1e-6)


def test_nn_distances_reject_dimension_mismatch() -> None:
    with pytest.raises(core.BehaviourMatchingError):
        core.nn_distances(np.zeros((3, 2)), np.zeros((3, 5)))


def test_centroid_distance_matches_definition() -> None:
    query = np.array([[0.0, 0.0], [2.0, 0.0]])
    reference = np.array([[1.0, 0.0], [3.0, 0.0]])
    assert np.allclose(core.centroid_distances(query, reference), [2.0, 0.0])


def test_class_restricted_nn_only_uses_same_class_references() -> None:
    query = np.array([[0.0], [10.0]])
    query_labels = np.array([0, 1])
    reference = np.array([[5.0], [11.0]])
    reference_labels = np.array([0, 1])
    distances, match_rate = core.class_restricted_nn(query, query_labels, reference, reference_labels)
    assert match_rate == 1.0
    assert sorted(distances.tolist()) == [1.0, 5.0]


def test_class_restricted_nn_treats_unlabelled_samples_as_unmatchable() -> None:
    query = np.array([[0.0], [10.0]])
    query_labels = np.array([loaders.UNLABELLED, 3])
    reference = np.array([[1.0], [11.0]])
    reference_labels = np.array([loaders.UNLABELLED, 3])
    distances, match_rate = core.class_restricted_nn(query, query_labels, reference, reference_labels)
    assert match_rate == pytest.approx(0.5)
    assert distances.tolist() == [1.0]


def test_class_restricted_nn_reports_missing_classes() -> None:
    query = np.array([[0.0], [10.0]])
    query_labels = np.array([0, 7])
    reference = np.array([[5.0]])
    reference_labels = np.array([0])
    distances, match_rate = core.class_restricted_nn(query, query_labels, reference, reference_labels)
    assert match_rate == pytest.approx(0.5)
    assert distances.tolist() == [5.0]


# ---------------------------------------------------------------------------
# section 6/8: statistics and the predeclared decision rule
# ---------------------------------------------------------------------------


def test_distance_summary_reports_per_dimension_residual() -> None:
    summary = core.distance_summary(np.full(10, 4.0), dim=4)
    assert summary["median"] == pytest.approx(4.0)
    assert summary["median_per_dim"] == pytest.approx(2.0)


def test_spread_over_pairs_rejects_nonfinite() -> None:
    with pytest.raises(core.BehaviourMatchingError):
        core.spread_over_pairs([1.0, float("inf")])


@pytest.mark.parametrize(
    "median_ratio,p90_ratio,expected",
    [
        (1.0, 1.2, "TIGHT"),
        (1.25, 1.50, "TIGHT"),
        (1.10, 1.80, "MARGINAL"),
        (1.90, 4.00, "MARGINAL"),
        (2.01, 2.10, "LOOSE"),
    ],
)
def test_ratio_classification_matches_frozen_thresholds(median_ratio, p90_ratio, expected) -> None:
    assert core.classify_ratio(median_ratio, p90_ratio) == expected


@pytest.mark.parametrize(
    "value,expected", [(0.05, "ABS_TIGHT"), (0.10, "ABS_TIGHT"), (0.24, "ABS_MARGINAL"), (0.40, "ABS_LOOSE")]
)
def test_absolute_classification_matches_frozen_thresholds(value, expected) -> None:
    assert core.classify_absolute(value) == expected


def test_combined_verdict_requires_both_criteria() -> None:
    assert core.combined_verdict("TIGHT", "ABS_MARGINAL")["supports_behaviour_matched_pairing"]
    blocked = core.combined_verdict("TIGHT", "ABS_LOOSE")
    assert not blocked["supports_behaviour_matched_pairing"]
    assert blocked["failing_criteria"] == ["absolute_residual=ABS_LOOSE"]
    assert not core.combined_verdict("LOOSE", "ABS_TIGHT")["supports_behaviour_matched_pairing"]


# ---------------------------------------------------------------------------
# section 9: dimensionality
# ---------------------------------------------------------------------------


def test_participation_ratio_of_isotropic_data_approaches_the_dimension() -> None:
    rng = np.random.default_rng(4)
    report = core.dimensionality_report(rng.normal(size=(40000, 6)))
    assert report["participation_ratio"] == pytest.approx(6.0, rel=0.05)


def test_participation_ratio_collapses_for_a_rank_one_embedding() -> None:
    rng = np.random.default_rng(5)
    latent = rng.normal(size=(3000, 1))
    report = core.dimensionality_report(latent @ rng.normal(size=(1, 9)))
    assert report["participation_ratio"] == pytest.approx(1.0, abs=0.05)
    assert report["components_for_99pct"] == 1


# ---------------------------------------------------------------------------
# section 10: the -1/d scaling test
# ---------------------------------------------------------------------------


def test_slope_recovers_minus_one_over_d_on_uniform_synthetic_data() -> None:
    rng = np.random.default_rng(6)
    dim = 3
    query = rng.uniform(size=(400, dim))
    sizes = [64, 256, 1024, 4096, 16384]
    medians = [
        float(np.median(core.nn_distances(query, rng.uniform(size=(size, dim))))) / math.sqrt(dim)
        for size in sizes
    ]
    fit = core.fit_log_slope(sizes, medians)
    assert fit["implied_d_eff"] == pytest.approx(dim, rel=0.15)
    assert fit["r_squared"] > 0.99


def test_fit_log_slope_is_exact_on_a_clean_power_law() -> None:
    sizes = [10, 100, 1000]
    values = [size ** (-0.25) for size in sizes]
    fit = core.fit_log_slope(sizes, values)
    assert fit["slope"] == pytest.approx(-0.25, abs=1e-9)
    assert fit["implied_d_eff"] == pytest.approx(4.0, abs=1e-6)


def test_scaling_classification_needs_enough_sweep_points() -> None:
    thin = [{"cohort": "c", "representation": "r", "nominal_dim": 2, "participation_ratio": 2.0,
             "fit": {"points": 2, "implied_d_eff": 2.0}}]
    assert core.classify_scaling(thin)["verdict"] == "UNTESTABLE"


def test_scaling_classification_splits_agreement() -> None:
    rows = [
        {"cohort": "a", "representation": "r", "nominal_dim": 2, "participation_ratio": 2.0,
         "fit": {"points": 5, "implied_d_eff": 2.2}},
        {"cohort": "b", "representation": "r", "nominal_dim": 7, "participation_ratio": 7.0,
         "fit": {"points": 5, "implied_d_eff": 30.0}},
    ]
    verdict = core.classify_scaling(rows)
    assert verdict["verdict"] == "PARTIAL"
    assert verdict["agreeing"] == 1


# ---------------------------------------------------------------------------
# end-to-end behaviour of evaluate_pair on constructed distributions
# ---------------------------------------------------------------------------


def _view(name: str, query: np.ndarray, reference: np.ndarray, labels=None) -> core.SessionView:
    labels = labels or {}
    return core.SessionView(
        cohort="synthetic",
        session=name,
        representation="vel2",
        dim=int(query.shape[1]),
        query=query,
        reference=reference,
        query_labels={k: v[0] for k, v in labels.items()},
        reference_labels={k: v[1] for k, v in labels.items()},
    )


def test_identical_distributions_give_a_ratio_near_one() -> None:
    rng = np.random.default_rng(7)
    a = _view("a", rng.normal(size=(512, 2)), rng.normal(size=(1024, 2)))
    b = _view("b", rng.normal(size=(512, 2)), rng.normal(size=(1024, 2)))
    floor = core.within_session_floor(a)["floor"]["median"]
    row = core.evaluate_pair(a, b, within_floor_median=floor)
    assert row["ratio_cross_over_within"] == pytest.approx(1.0, abs=0.2)
    assert core.classify_ratio(row["ratio_cross_over_within"], row["ratio_cross_over_within"]) == "TIGHT"


def test_a_shifted_reference_distribution_is_detected_as_loose() -> None:
    rng = np.random.default_rng(8)
    a = _view("a", rng.normal(size=(512, 2)), rng.normal(size=(1024, 2)))
    b = _view("b", rng.normal(size=(512, 2)), rng.normal(size=(1024, 2)) + 10.0)
    floor = core.within_session_floor(a)["floor"]["median"]
    row = core.evaluate_pair(a, b, within_floor_median=floor)
    assert row["ratio_cross_over_within"] > 2.0
    assert core.classify_ratio(row["ratio_cross_over_within"], row["ratio_cross_over_within"]) == "LOOSE"


def test_within_floor_is_positive_for_a_temporally_disjoint_split() -> None:
    rng = np.random.default_rng(9)
    view = _view("a", rng.normal(size=(256, 2)), rng.normal(size=(512, 2)))
    assert core.within_session_floor(view)["floor"]["median"] > 0.0


def test_evaluate_pair_rejects_a_nonpositive_floor() -> None:
    rng = np.random.default_rng(10)
    a = _view("a", rng.normal(size=(16, 2)), rng.normal(size=(16, 2)))
    with pytest.raises(core.BehaviourMatchingError):
        core.evaluate_pair(a, a, within_floor_median=0.0)


def test_session_view_usability_honours_the_size_floor() -> None:
    small = _view("a", np.zeros((core.MIN_POOL - 1, 2)), np.zeros((core.M_REF, 2)))
    ok = _view("b", np.zeros((core.MIN_POOL, 2)), np.zeros((core.MIN_POOL, 2)))
    assert not small.usable()
    assert ok.usable()


# ---------------------------------------------------------------------------
# loaders: pure helpers only (no NWB is opened here)
# ---------------------------------------------------------------------------


def test_representation_dims_are_as_declared() -> None:
    assert loaders.representation_dim("vel2") == 2
    assert loaders.representation_dim("win100") == 100
    assert loaders.representation_dim("disp7") == 7
    with pytest.raises(core.BehaviourMatchingError):
        loaders.representation_groups("nope")


def test_win100_groups_alternate_by_velocity_channel() -> None:
    groups = loaders.representation_groups("win100")
    assert groups.size == 100
    assert groups[0] == 0 and groups[1] == 1 and groups[2] == 0
    assert int(np.unique(groups).size) == 2


def test_rt_eligible_starts_require_one_contiguous_accepted_segment() -> None:
    segment = np.array([0] * 6 + [-1] * 2 + [1] * 6, dtype=np.int64)
    eval_mask = np.ones(segment.size, dtype=bool)
    starts = loaders.rt_eligible_starts(segment, eval_mask, window_size=3)
    ends = starts + 2
    assert starts.size > 0
    assert np.all(segment[ends] >= 0)
    for start in starts:
        window = segment[start : start + 3]
        assert window.min() == window.max() and window[0] >= 0


def test_rt_eligible_starts_respect_the_eval_mask() -> None:
    segment = np.zeros(10, dtype=np.int64)
    eval_mask = np.zeros(10, dtype=bool)
    eval_mask[5] = True
    starts = loaders.rt_eligible_starts(segment, eval_mask, window_size=3)
    assert starts.tolist() == [3]


def test_rt_eligible_starts_never_span_a_segment_boundary() -> None:
    segment = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    eval_mask = np.ones(6, dtype=bool)
    starts = loaders.rt_eligible_starts(segment, eval_mask, window_size=3)
    assert starts.tolist() == [0, 3]


def test_rt_eligible_starts_handle_a_short_session() -> None:
    assert loaders.rt_eligible_starts(np.zeros(3, dtype=np.int64), np.ones(3, dtype=bool)).size == 0


def test_phase_quintile_spans_the_declared_range() -> None:
    offsets = np.arange(11, dtype=np.int64)
    quintiles = loaders._phase_quintile(offsets, np.full(11, 10))
    assert quintiles.min() == 0 and quintiles.max() == loaders.PHASE_QUANTILES - 1
    assert np.all(np.diff(quintiles) >= 0)


def test_session_samples_materializes_windows_in_lag_major_order() -> None:
    behavior = np.arange(2 * (loaders.WINDOW_SIZE + 10), dtype=np.float32).reshape(-1, 2)
    samples = loaders.SessionSamples(
        cohort="c", session="s", path=Path("/dev/null"), n_samples=2, labels={}, bindings={},
        behavior=behavior, starts=np.array([0, 5], dtype=np.int64),
    )
    window = samples.materialize("win100", np.array([0]))
    assert window.shape == (1, 100)
    assert np.allclose(window[0, :4], behavior[:2].reshape(-1))
    endpoint = samples.materialize("vel2", np.array([0, 1]))
    assert np.allclose(endpoint[0], behavior[loaders.WINDOW_SIZE - 1])
    assert np.allclose(endpoint[1], behavior[5 + loaders.WINDOW_SIZE - 1])


def test_frozen_source_manifest_resolves_to_27_sessions() -> None:
    paths = loaders.discover_centre_out_subc(REPO_ROOT)
    assert len(paths) == 27
    assert all(path.name.startswith("sub-C_ses-CO-") for path in paths)


def test_rt_discovery_is_bound_to_the_sealed_name_check() -> None:
    from sua_exploration.mc_maze import rt_classical_comparators as sealed

    paths = loaders.discover_rt_subc(REPO_ROOT)
    assert len(paths) == sealed.EXPECTED_FOLDS
    assert all("ses-RT-" in path.name for path in paths)
    with pytest.raises(sealed.RtClassicalComparatorError):
        sealed.session_name_from_nwb_path(Path("sub-C_ses-CO-20131003_behavior+ecephys.nwb"))


def test_h1_discovery_uses_the_sealed_heldin_index() -> None:
    paths = loaders.discover_h1_heldin(REPO_ROOT)
    assert len(paths) == 13
    assert all("sub-HumanPitt-held-in-calib" in str(path) for path in paths)
    assert not any("held-out" in str(path) or "minival" in str(path) for path in paths)
