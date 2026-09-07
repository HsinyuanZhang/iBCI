"""Focused no-NWB contracts for the isolated Step-2A T4@M preparation."""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.t4_cross_budget_features import (  # noqa: E402
    CrossBudgetFeatureBundle,
    T4_CROSS_BUDGET_CACHE_NAMESPACE,
    _bundle_arrays,
    _load_bundle,
    cross_budget_cache_payload,
)
from mc_maze.t4_cross_budget_audit import (  # noqa: E402
    CrossBudgetSessionAudit,
    descriptor_error_to_t50,
    deterministic_nonidentity_row_shuffle,
    disjoint_trial_count_partitions,
    load_frozen_source_development_manifest,
    nested_source_q_error_audit,
    split_half_t4_repeatability,
)
from mc_maze.t4_cross_budget_protocol import (  # noqa: E402
    RELIABILITY_FEATURE_NAMES,
    canonical_direction_indices,
    fit_t4_prefix,
    fit_t4_prefixes,
    thin_trial_counts_within_trial,
    within_trial_thinned_prefix_fit,
)
from mc_maze.unit_side_features import (  # noqa: E402
    CANONICAL_DIRECTIONS_RAD,
    _unit_tuning_features,
)


def _synthetic_trial_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directions = np.asarray([index % 8 for index in range(16)], dtype=np.int64)
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[int(index)] for index in directions])
    durations = np.ones(16, dtype=np.float64)
    # Two distinct units, with a small deterministic within-direction variation.  Existing T4
    # semantics must collapse it to equal per-direction means before the cosine fit.
    rates = np.stack(
        [
            8.0 + 2.0 * np.cos(theta) - 1.5 * np.sin(theta) + 0.2 * (np.arange(16) % 2),
            6.0 - 0.5 * np.cos(theta) + 3.0 * np.sin(theta) + 0.1 * (np.arange(16) % 2),
        ]
    )
    return rates, durations, directions


def test_actual_prefix_refit_matches_existing_equal_direction_t4_semantics():
    rates, durations, directions = _synthetic_trial_data()
    fit = fit_t4_prefix(rates, durations, directions, budget=16)
    assert fit.fit_defined and fit.design_rank == 3 and np.isfinite(fit.design_condition)
    expected = np.stack(
        [
            _unit_tuning_features(rates[unit], directions, list(range(8)))[0]
            for unit in range(rates.shape[0])
        ]
    )
    assert np.allclose(fit.t4, expected, atol=1.0e-6)
    assert fit.direction_counts.tolist() == [2] * 8
    assert fit.direction_balance == 1.0
    assert np.isfinite(fit.reliability).all()


def test_rank_deficient_prefix_is_undefined_not_zero_filled():
    rates = np.ones((3, 4), dtype=np.float64)
    durations = np.ones(4, dtype=np.float64)
    directions = np.asarray([0, 0, 1, 1], dtype=np.int64)
    fit = fit_t4_prefix(rates, durations, directions, budget=4)
    assert not fit.fit_defined
    assert fit.design_rank == 2
    assert np.isinf(fit.design_condition)
    assert np.isnan(fit.t4).all()
    assert np.isnan(fit.reliability).all()


def test_missing_direction_never_contaminates_labelled_fit_reliability():
    # First three rows form the rank-3 labelled design.  An enormous unlabelled rate must not
    # affect residual variance, labelled exposure, or modulation/residual.
    directions = np.asarray([0, 1, 2, -1], dtype=np.int64)
    durations = np.ones(4)
    rates_without_missing = np.asarray([[1.0, 2.0, 3.0]])
    low_missing = np.asarray([[1.0, 2.0, 3.0, 1.0e6]])
    reference = fit_t4_prefix(rates_without_missing, durations[:3], directions[:3], budget=3)
    observed = fit_t4_prefix(low_missing, durations, directions, budget=4)
    assert reference.fit_defined and observed.fit_defined
    assert np.allclose(observed.t4, reference.t4)
    # Labelled residual/spike/time/modulation coordinates are unchanged.  Total-prefix spike
    # count and time are deliberately separate, so the unlabelled row can affect only them.
    assert np.allclose(observed.reliability[:, [0, 1, 2, 5]], reference.reliability[:, [0, 1, 2, 5]])
    assert observed.reliability[0, 3] > reference.reliability[0, 3]
    assert observed.reliability[0, 4] > reference.reliability[0, 4]
    assert RELIABILITY_FEATURE_NAMES == (
        "log1p_labelled_fit_residual_variance", "log1p_labelled_spike_count",
        "log1p_labelled_time_exposure_s", "log1p_total_prefix_spike_count",
        "log1p_total_prefix_time_exposure_s", "log1p_modulation_to_residual", "rank_valid",
    )


def test_prefix_nesting_and_within_trial_thinning_keep_trial_label_design_fixed():
    rates, durations, directions = _synthetic_trial_data()
    fits = fit_t4_prefixes(rates, durations, directions, budgets=(10, 15, 16))
    assert list(fits) == [10, 15, 16]
    assert fits[10].direction_counts.sum() == 10
    assert fits[15].direction_counts.sum() == 15
    counts = np.rint(rates * durations[None, :]).astype(np.int64)
    # keep=1 must be exact and uses the same direction vector/order; no new coefficient noise.
    thinned = within_trial_thinned_prefix_fit(
        counts, durations, directions, budget=16, keep_probability=1.0, rng=np.random.default_rng(7)
    )
    direct = fit_t4_prefix(counts / durations[None, :], durations, directions, budget=16)
    assert np.array_equal(thinned.direction_counts, direct.direction_counts)
    assert np.array_equal(thinned.t4, direct.t4)
    half = thin_trial_counts_within_trial(
        counts, keep_probability=0.5, rng=np.random.default_rng(9)
    )
    assert half.shape == counts.shape
    assert np.all(half <= counts)


def test_direction_snapping_preserves_missing_label_and_cache_payload_is_versioned(tmp_path):
    directions = canonical_direction_indices([CANONICAL_DIRECTIONS_RAD[0], None, float("nan")])
    assert directions.tolist() == [0, -1, -1]
    source = tmp_path / "source.nwb"
    source.touch()
    payload = cross_budget_cache_payload(
        source,
        budgets=(10, 15, 20, 50),
        bin_size_ms=20,
        window_size=50,
        trial_result_filter="R",
        signal_view="sua",
    )
    assert payload["cache_namespace"] == T4_CROSS_BUDGET_CACHE_NAMESPACE
    assert payload["budgets"] == [10, 15, 20, 50]
    assert payload["signal_view"] == "sua"
    assert payload["fit_semantics_version"]
    changed = dict(payload); changed["budgets"] = [10, 20, 50]
    assert changed != payload
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_t4_prefixes(np.ones((1, 16)), np.ones(16), np.zeros(16, dtype=np.int64), budgets=(15, 10))


def test_cache_round_trip_retains_every_real_prefix_and_undefined_state(tmp_path):
    rates, durations, directions = _synthetic_trial_data()
    fits = fit_t4_prefixes(rates, durations, directions, budgets=(10, 16))
    source = tmp_path / "source.nwb"; source.touch()
    payload = cross_budget_cache_payload(source, budgets=(10, 16), bin_size_ms=20, window_size=50, trial_result_filter="R", signal_view="sua")
    bundle = CrossBudgetFeatureBundle(
        source_path=str(source), signal_view="sua", cache_key=__import__("mc_maze.multisession_datamodule", fromlist=["_cache_key"])._cache_key(payload), budgets=(10, 16), fits=fits,
        cache_payload=payload, trial_ordinals=np.arange(16), trial_start_times=np.arange(16, dtype=float), trial_stop_times=np.arange(16, dtype=float) + 0.5,
        trial_target_dirs_rad=np.asarray([CANONICAL_DIRECTIONS_RAD[index] for index in directions]), trial_direction_indices=directions,
    )
    path = tmp_path / "t4_cross_budget.npz"
    np.savez_compressed(path, **_bundle_arrays(bundle))
    loaded = _load_bundle(path, expected_payload=payload)
    assert loaded.budgets == (10, 16)
    assert loaded.cache_key == bundle.cache_key
    for budget in loaded.budgets:
        assert np.array_equal(loaded.fits[budget].t4, fits[budget].t4)
        assert np.array_equal(loaded.fits[budget].direction_counts, fits[budget].direction_counts)
    assert np.array_equal(loaded.trial_ordinals, np.arange(16))
    assert np.array_equal(loaded.trial_direction_indices, directions)
    mismatch = dict(payload); mismatch["window_size"] = 99
    with pytest.raises(ValueError, match="payload differs"):
        _load_bundle(path, expected_payload=mismatch)


def test_disjoint_count_partitions_conserve_counts_and_repeatability_is_summary_only():
    counts = np.full((3, 16), 20, dtype=np.int64)
    directions = np.asarray([index % 8 for index in range(16)], dtype=np.int64)
    parts = disjoint_trial_count_partitions(counts, seed=7)
    assert parts.shape == (2, 3, 16)
    assert np.array_equal(parts.sum(axis=0), counts)
    result = split_half_t4_repeatability(counts, np.ones(16), directions, budget=16, seeds=(11, 13))
    assert result["defined_replicates"] == 2
    assert result["partition"] == "disjoint_multinomial_two_way_rescaled_rate"
    assert "counts" not in result


def test_row_shuffle_and_nested_q_audit_are_target_free_at_leftout_application():
    rates, durations, directions = _synthetic_trial_data()
    # Make T50 distinct from low M while retaining rank-three coverage.
    rates = np.tile(rates, (1, 4))
    durations = np.ones(rates.shape[1]); directions = np.tile(directions, 4)
    base = fit_t4_prefixes(rates, durations, directions, budgets=(10, 15, 20, 50))
    sessions = []
    for index in range(3):
        shifted = rates + index * 0.2
        sessions.append(CrossBudgetSessionAudit(f"s{index}", fit_t4_prefixes(shifted, durations, directions, budgets=(10, 15, 20, 50))))
    shuffled, permutation = deterministic_nonidentity_row_shuffle(base[50].t4, session_name="s0", seed=42)
    assert not np.array_equal(permutation, np.arange(permutation.size))
    assert np.array_equal(np.sort(shuffled, axis=0), np.sort(base[50].t4, axis=0))
    assert descriptor_error_to_t50(base[10], base[50]).shape == (2,)
    audit = nested_source_q_error_audit(sessions, budgets=(10, 15, 20, 50))
    assert audit["formal_test_used"] is False
    assert audit["target_free_leftout_application"] is True
    assert audit["defined_session_count"] == 3


def test_frozen_source_manifest_keeps_formal_names_sealed_not_resolved(tmp_path):
    manifest = {
        "schema_version": 1,
        "session_splits": {
            "train": [f"train_{index}" for index in range(27)],
            "val": [f"val_{index}" for index in range(6)],
            "test": [f"test_{index}" for index in range(6)],
        },
    }
    path = tmp_path / "manifest.json"
    import json
    path.write_text(json.dumps(manifest))
    receipt = load_frozen_source_development_manifest(path)
    assert len(receipt["nested_source_session_names"]) == 27
    assert len(receipt["development_validation_names"]) == 6
    assert len(receipt["sealed_formal_test_names"]) == 6
    assert receipt["formal_test_paths_resolved"] is False


def _audit_script_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_sua_t4_cross_budget.py"
    spec = importlib.util.spec_from_file_location("step2a_cross_budget_runner", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_resolves_only_frozen_train_and_validation_never_formal_paths(tmp_path):
    module = _audit_script_module()
    base = tmp_path / "data" / "sub-C"; base.mkdir(parents=True)
    receipt = {
        "nested_source_session_names": tuple(f"train_{index}" for index in range(27)),
        "development_validation_names": tuple(f"val_{index}" for index in range(6)),
        "sealed_formal_test_names": tuple(f"test_{index}" for index in range(6)),
    }
    for name in receipt["nested_source_session_names"] + receipt["development_validation_names"]:
        (base / f"{name}_behavior+ecephys.nwb").touch()
    train, val = module.resolve_development_paths(receipt, tmp_path / "data")
    assert len(train) == 27 and len(val) == 6
    assert not any(path.name.startswith("test_") for path in list(train.values()) + list(val.values()))
    assert not (base / "test_0_behavior+ecephys.nwb").exists()


def test_runner_strict_json_has_no_nonfinite_array_leakage():
    module = _audit_script_module()
    value = module.strict_json({"count_like": np.asarray([1, 2]), "bad": np.asarray([np.nan, np.inf])})
    assert value == {"count_like": [1, 2], "bad": [None, None]}
