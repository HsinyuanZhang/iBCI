"""Tests for aggregate_attention_architecture_screen_v4.py.

Focus is the MEASUREMENT_PROTOCOL_V4 section 4.2b four-state verdict
(``classify_pair_verdict``, imported unmodified from aggregate_side_feature_ablation_v2 --
the single shared implementation every aggregator in this repo uses), covering all four
literal outcomes -- effective / effective_heterogeneous / ineffective / indeterminate -- plus
the boundary conditions between them, since conflating "indeterminate" with "ineffective" is
exactly the bug that voided attention_arch_screen_v3 (sua_exploration/docs/CURRENT_RESULTS.md
section H, section E retraction). Exhaustive coverage of the four-state cascade itself lives
in test_aggregate_side_feature_ablation_v2.py (where classify_pair_verdict is defined); the
tests here mainly pin that this module wires the SAME shared function/constants (not a
divergent copy) and exercises it correctly end to end. A smaller set of tests also covers the
section-2 consistency gate (unique run dirs, agreeing session splits, matched training
provenance) and one full synthetic 12-run pipeline exercising ``run_aggregation`` end to end,
with hand-designed inputs that produce three of the four verdicts simultaneously across the
three paired comparisons.

No GPU, no NWB data, no torch: the aggregator only reads JSON, so these tests only need
``pytest`` and the standard library.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_attention_architecture_screen_v4 as agg  # noqa: E402
import aggregate_side_feature_ablation_v2 as agg_v2  # noqa: E402


# ----------------------------------------------------------------------------------------
# classify_pair_verdict: the load-bearing four-state gate, imported unmodified from
# aggregate_side_feature_ablation_v2 (exhaustive cascade/boundary coverage lives there; these
# tests confirm this module wires the same shared function with its own (0.03 / 5) constants).
# ----------------------------------------------------------------------------------------
def _classify(**kwargs):
    kwargs.setdefault("effective_mean_delta_threshold", agg.EFFECTIVE_MEAN_DELTA_THRESHOLD)
    kwargs.setdefault("effective_min_positive_sessions", agg.EFFECTIVE_MIN_POSITIVE_SESSIONS)
    return agg.classify_pair_verdict(**kwargs)


def test_classify_pair_verdict_is_imported_from_v2_unmodified():
    assert agg.classify_pair_verdict is agg_v2.classify_pair_verdict


def test_effective_when_all_three_clauses_hold():
    verdict, decided_by = _classify(
        mean_delta=0.05,
        n_sessions_positive=6,
        n_sessions_total=6,
        per_seed_means=[0.04, 0.05, 0.06],
        sigma_delta_paired=0.01,
    )
    assert verdict == "effective"
    assert decided_by.startswith("effective:")


def test_effective_is_inclusive_at_exact_boundaries():
    # mean_delta exactly +0.03 and exactly 5/6 sessions positive: both boundaries use >=.
    verdict, _ = _classify(
        mean_delta=0.03,
        n_sessions_positive=5,
        n_sessions_total=6,
        per_seed_means=[0.001, 0.002, 0.003],
        sigma_delta_paired=100.0,  # irrelevant once effective is satisfied
    )
    assert verdict == "effective"


def test_not_effective_when_mean_delta_just_under_threshold():
    # mean_delta infinitesimally below +0.03 must not qualify as effective. With sigma=100
    # the mean+2sigma/mean-2sigma bounds are nowhere near threshold/zero, so this stays
    # indeterminate rather than tripping the ineffective or effective_heterogeneous clauses.
    verdict, _ = _classify(
        mean_delta=0.0299999,
        n_sessions_positive=6,
        n_sessions_total=6,
        per_seed_means=[0.03, 0.03, 0.03],
        sigma_delta_paired=100.0,
    )
    assert verdict == "indeterminate"


def test_not_effective_when_only_four_of_six_sessions_positive():
    # mean_delta and per-seed signs both pass, but positive-session count fails (4 < 5) --
    # with a huge sigma this is NOT effective_heterogeneous either (mean-2sigma < 0), so it
    # stays indeterminate.
    verdict, _ = _classify(
        mean_delta=0.04,
        n_sessions_positive=4,
        n_sessions_total=6,
        per_seed_means=[0.03, 0.04, 0.05],
        sigma_delta_paired=100.0,
    )
    assert verdict == "indeterminate"


def test_effective_heterogeneous_when_confidently_positive_but_sessions_inconsistent():
    # Same shape as the case above, but with a small (realistic) sigma: mean_delta=0.04,
    # sigma=0.005 -> mean-2sigma=0.03>0 (confidently positive), all seeds positive, but only
    # 4/6 sessions positive (< 5) -- the new 2026-07-27 state.
    verdict, decided_by = _classify(
        mean_delta=0.04,
        n_sessions_positive=4,
        n_sessions_total=6,
        per_seed_means=[0.03, 0.04, 0.05],
        sigma_delta_paired=0.005,
    )
    assert verdict == "effective_heterogeneous"
    assert decided_by.startswith("effective_heterogeneous:")


def test_not_effective_when_one_seed_mean_is_not_positive():
    # mean_delta and session count both pass, but one seed's own mean delta is <= 0.
    verdict, _ = _classify(
        mean_delta=0.05,
        n_sessions_positive=6,
        n_sessions_total=6,
        per_seed_means=[0.10, 0.10, -0.05],
        sigma_delta_paired=100.0,
    )
    assert verdict == "indeterminate"


def test_not_effective_when_seed_mean_is_exactly_zero():
    # "positive" is strict (> 0); a seed mean of exactly 0.0 does not count.
    verdict, _ = _classify(
        mean_delta=0.05,
        n_sessions_positive=6,
        n_sessions_total=6,
        per_seed_means=[0.10, 0.10, 0.0],
        sigma_delta_paired=100.0,
    )
    assert verdict == "indeterminate"


def test_ineffective_when_confidently_below_threshold():
    verdict, decided_by = _classify(
        mean_delta=-0.05,
        n_sessions_positive=0,
        n_sessions_total=6,
        per_seed_means=[-0.04, -0.05, -0.06],
        sigma_delta_paired=0.01,
    )
    assert verdict == "ineffective"
    assert decided_by.startswith("ineffective:")


def test_ineffective_boundary_is_strict_less_than():
    # mean_delta + 2*sigma_delta_paired exactly == threshold must NOT be ineffective (needs
    # <, not <=).
    verdict, _ = _classify(
        mean_delta=0.01,
        n_sessions_positive=3,
        n_sessions_total=6,
        per_seed_means=[0.01, 0.01, 0.01],
        sigma_delta_paired=0.01,  # mean + 2*sigma == 0.03 == threshold, not strictly less
    )
    assert verdict == "indeterminate"
    # Nudge just past the boundary: now it must flip to ineffective.
    verdict_past_boundary, _ = _classify(
        mean_delta=0.0099999,
        n_sessions_positive=3,
        n_sessions_total=6,
        per_seed_means=[0.01, 0.01, 0.01],
        sigma_delta_paired=0.01,
    )
    assert verdict_past_boundary == "ineffective"


def test_indeterminate_when_bounds_straddle_both_zero_and_threshold():
    verdict, _ = _classify(
        mean_delta=0.005,
        n_sessions_positive=4,
        n_sessions_total=6,
        per_seed_means=[0.002, 0.005, 0.008],
        sigma_delta_paired=0.02,  # mean-2sigma=-0.035<0, mean+2sigma=0.045>=0.03
    )
    assert verdict == "indeterminate"


def test_indeterminate_with_zero_sigma_and_zero_delta_does_not_crash():
    # Degenerate but legal: sigma_delta_paired == 0 and mean_delta == 0.
    verdict, _ = _classify(
        mean_delta=0.0,
        n_sessions_positive=0,
        n_sessions_total=6,
        per_seed_means=[0.0, 0.0, 0.0],
        sigma_delta_paired=0.0,
    )
    assert verdict == "indeterminate"


def test_verdict_is_always_one_of_the_four_literal_strings():
    # Never a bool, never any other spelling -- sweep a grid of plausible inputs.
    for mean_delta in (-0.2, -0.03, -0.005, 0.0, 0.005, 0.03, 0.2):
        for n_positive in (0, 2, 4, 5, 6):
            for sigma in (0.0, 0.001, 0.01, 0.05):
                for seed_means in ([mean_delta] * 3, [-1.0, mean_delta, 1.0]):
                    verdict, decided_by = _classify(
                        mean_delta=mean_delta,
                        n_sessions_positive=n_positive,
                        n_sessions_total=6,
                        per_seed_means=seed_means,
                        sigma_delta_paired=sigma,
                    )
                    assert verdict in agg_v2.VALID_VERDICTS
                    assert isinstance(verdict, str)
                    assert isinstance(decided_by, str) and decided_by


def test_classify_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma_delta_paired"):
        _classify(
            mean_delta=0.05, n_sessions_positive=5, n_sessions_total=6,
            per_seed_means=[0.1, 0.1, 0.1], sigma_delta_paired=-0.01,
        )


# ----------------------------------------------------------------------------------------
# sample_std
# ----------------------------------------------------------------------------------------
def test_sample_std_matches_statistics_stdev():
    import statistics

    values = [0.30, 0.34, 0.32]
    assert agg.sample_std(values) == pytest.approx(statistics.stdev(values))


def test_sample_std_of_constant_sequence_is_zero():
    assert agg.sample_std([0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_sample_std_requires_at_least_two_values():
    with pytest.raises(ValueError, match="at least 2"):
        agg.sample_std([0.5])


# ----------------------------------------------------------------------------------------
# sigma_delta_standard_error (M7 fix, ROADMAP.md "M"): pins the exact formula, including
# the ``/ sqrt(n_seeds)`` term that was previously missing. Before the fix, this aggregator
# used plain quadrature (``sqrt(sigma_a**2 + sigma_b**2)``, i.e. what
# ``sigma_delta_standard_error(sigma_a, sigma_b, n_seeds=1)`` returns here) as the SD of the
# n_seeds-seed MEAN delta -- that is the SD of a *single* seed's delta, ~1.73x too wide at
# n_seeds=3, which can only bias the three-state verdict toward "indeterminate".
# ----------------------------------------------------------------------------------------
def test_sigma_delta_standard_error_matches_hand_computed_value():
    import math

    result = agg.sigma_delta_standard_error(0.03, 0.04, 3)
    assert result == pytest.approx(0.05 / math.sqrt(3))


def test_sigma_delta_standard_error_divides_by_sqrt_n_seeds_not_n_seeds():
    # The pre-fix formula is exactly sigma_delta_standard_error(..., n_seeds=1).
    quadrature_only = agg.sigma_delta_standard_error(0.03, 0.04, 1)
    assert quadrature_only == pytest.approx(0.05)
    corrected = agg.sigma_delta_standard_error(0.03, 0.04, 3)
    assert corrected == pytest.approx(quadrature_only / 3 ** 0.5)
    assert corrected != pytest.approx(quadrature_only / 3)  # not the (wrong) /n_seeds form


def test_sigma_delta_standard_error_smaller_than_pre_fix_quadrature_for_n_seeds_gt_1():
    pre_fix = (0.024 ** 2 + 0.048 ** 2) ** 0.5
    post_fix = agg.sigma_delta_standard_error(0.024, 0.048, 3)
    assert post_fix < pre_fix
    assert post_fix == pytest.approx(pre_fix / 3 ** 0.5)


def test_sigma_delta_standard_error_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma_a and sigma_b"):
        agg.sigma_delta_standard_error(-0.01, 0.02, 3)


def test_sigma_delta_standard_error_rejects_non_positive_n_seeds():
    with pytest.raises(ValueError, match="n_seeds"):
        agg.sigma_delta_standard_error(0.01, 0.02, 0)


# ----------------------------------------------------------------------------------------
# sigma_delta_standard_error / sigma_delta_paired / implied_seed_correlation must be the
# SAME function objects as aggregate_side_feature_ablation_v2's (2026-07-26: this module used
# to keep its own copy-pasted sigma_delta_standard_error; it now imports all three from v2,
# matching aggregate_e3_tuning_ablation.py / aggregate_e4_encoder_variants.py, so there is
# exactly one implementation of the sigma_delta estimator in this repo).
# ----------------------------------------------------------------------------------------
def test_sigma_delta_standard_error_is_imported_from_v2_unmodified():
    assert agg.sigma_delta_standard_error is agg_v2.sigma_delta_standard_error


def test_sigma_delta_paired_is_imported_from_v2_unmodified():
    assert agg.sigma_delta_paired is agg_v2.sigma_delta_paired


def test_implied_seed_correlation_is_imported_from_v2_unmodified():
    assert agg.implied_seed_correlation is agg_v2.implied_seed_correlation


# ----------------------------------------------------------------------------------------
# Per-run helpers (no file I/O; pure dict transforms).
# ----------------------------------------------------------------------------------------
def _toy_payload(session_r2_by_epoch: dict[int, dict[str, float]]) -> dict:
    per_epoch = {
        str(epoch): {"per_session_r2": sessions, "mean_r2": agg.mean(list(sessions.values()))}
        for epoch, sessions in session_r2_by_epoch.items()
    }
    per_epoch_mean_r2 = {key: value["mean_r2"] for key, value in per_epoch.items()}
    return {
        "session_splits": {"val": sorted(next(iter(session_r2_by_epoch.values())).keys())},
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": per_epoch_mean_r2,
    }


def test_per_run_session_scores_averages_over_epoch_window():
    payload = _toy_payload({
        5: {"sA": 0.10, "sB": 0.20},
        6: {"sA": 0.20, "sB": 0.30},
        7: {"sA": 0.30, "sB": 0.40},
        8: {"sA": 0.40, "sB": 0.50},
        9: {"sA": 0.10, "sB": 0.20},
        10: {"sA": 0.20, "sB": 0.30},
        11: {"sA": 0.30, "sB": 0.40},
        12: {"sA": 0.40, "sB": 0.50},
    })
    scores = agg.per_run_session_scores(payload)
    assert scores["sA"] == pytest.approx((0.10 + 0.20 + 0.30 + 0.40) * 2 / 8)
    assert scores["sB"] == pytest.approx((0.20 + 0.30 + 0.40 + 0.50) * 2 / 8)


def test_per_run_within_window_std_uses_exactly_the_8_epoch_means():
    payload = _toy_payload({epoch: {"sA": 0.1 * (epoch - 4)} for epoch in agg.EXPECTED_EPOCH_WINDOW})
    # per_epoch_mean_r2 = {5: 0.1, 6: 0.2, ..., 12: 0.8}
    std = agg.per_run_within_window_std(payload)
    import statistics

    assert std == pytest.approx(statistics.stdev([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))


# ----------------------------------------------------------------------------------------
# Full synthetic 12-artifact pipeline through run_aggregation().
# ----------------------------------------------------------------------------------------
VAL_SESSIONS = [f"sub-X_ses-CO-2015110{i}" for i in range(1, 7)]
TRAIN_SESSIONS = [f"sub-X_ses-CO-2013100{i}" for i in range(1, 4)]
TEST_SESSIONS = [f"sub-X_ses-CO-2015120{i}" for i in range(1, 7)]
SESSION_SPLITS = {"train": TRAIN_SESSIONS, "val": VAL_SESSIONS, "test": TEST_SESSIONS}


def _epoch_values_with_exact_mean(target_mean: float) -> list[float]:
    """8 values alternating +/-0.01 around target_mean; mean is exactly target_mean and std
    is a known nonzero constant, so within_window_std is exercised without disturbing the
    per-seed variant_score (and therefore the hand-computed paired deltas below)."""
    wiggle = [-0.01, 0.01] * 4
    return [target_mean + delta for delta in wiggle]


def _write_artifact(
    results_dir: Path,
    checkpoints_dir: Path,
    variant: str,
    seed: int,
    target_mean: float,
    *,
    run_metadata_overrides: dict | None = None,
    run_dir_override: Path | None = None,
    session_splits: dict | None = None,
) -> Path:
    run_dir = run_dir_override or (checkpoints_dir / f"attention_arch_screen_v4_{variant.lower()}_dandi688_co_s{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_metadata_path = run_dir / "run_metadata.json"
    metadata_payload = {
        "status": "completed",
        "held_out_test_evaluated": False,
        "training": {"max_epochs": 12, "no_early_stopping": True, "checkpoint_every_epoch": True},
    }
    if run_metadata_overrides:
        training_overrides = run_metadata_overrides.pop("training", None)
        metadata_payload.update(run_metadata_overrides)
        if training_overrides:
            metadata_payload["training"].update(training_overrides)
    run_metadata_path.write_text(json.dumps(metadata_payload, sort_keys=True))

    splits = session_splits or SESSION_SPLITS
    epoch_values = _epoch_values_with_exact_mean(target_mean)
    per_epoch = {}
    per_epoch_mean_r2 = {}
    for epoch, value in zip(agg.EXPECTED_EPOCH_WINDOW, epoch_values):
        per_session_r2 = {session: value for session in splits["val"]}
        per_epoch[str(epoch)] = {
            "checkpoint_path": str(run_dir / "epoch_ckpts" / f"epoch_{epoch - 1:03d}.ckpt"),
            "checkpoint_sha256": "0" * 64,
            "per_session_r2": per_session_r2,
            "mean_r2": value,
        }
        per_epoch_mean_r2[str(epoch)] = value
    variant_score = agg.mean(list(per_epoch_mean_r2.values()))

    payload = {
        "schema_version": 1,
        "purpose": "epoch_window_deterministic_checkpoint_selection",
        "run_dir": str(run_dir),
        "run_metadata_path": str(run_metadata_path),
        "run_metadata_sha256": agg.sha256_file(run_metadata_path),
        "variant": variant,
        "seed": seed,
        "task": "CO",
        "protocol": {
            "total_epochs": 12,
            "epoch_window": agg.EXPECTED_EPOCH_WINDOW,
            "burn_in_epochs": 4,
            "selection_mode": "first",
            "calibration_n": 30,
            "pool_size": 50,
        },
        "epoch_list": agg.EXPECTED_EPOCH_WINDOW,
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": per_epoch_mean_r2,
        "variant_score": variant_score,
        "session_splits": splits,
        "calibration_trial_selection_uses_behavior_labels": False,
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
        "no_test_files_evaluated": True,
    }
    out_path = agg.artifact_path(results_dir, variant, seed)
    out_path.write_text(json.dumps(payload, sort_keys=True))
    return out_path


# Per-seed target means hand-chosen so that, simultaneously:
#   B15 - B3   -> effective     (mean_delta = +0.05, constant across seeds/sessions)
#   B15 - B15P -> indeterminate (mean_delta = +0.015; mean-2sigma<0 so not effective_
#                                 heterogeneous, mean+2sigma>=+0.03 so not ineffective either)
#   B15 - B15D -> ineffective   (mean_delta = -0.10, far outside measured sigma_delta)
# B15P's per-seed values are deliberately NOT a constant offset from B15's (unlike B3/B15D
# below): with a perfectly constant per-seed delta, sigma_delta_paired (2026-07-26 revision)
# is exactly 0, which would make ANY nonzero mean_delta resolve confidently one way or the
# other -- collapsing the three-simultaneous-verdicts design this fixture exists to exercise.
# The per-seed deltas here are +0.000/+0.020/+0.025 (mean +0.015, paired sigma ~0.0076, so
# 2*sigma~0.0152): mean-2*sigma ~ -0.0002 <= 0 (NOT confidently positive -- not
# effective_heterogeneous) and mean+2*sigma ~ +0.0302 >= +0.03 (NOT confidently excluding the
# threshold -- not ineffective either), so this pair stays "indeterminate" under the
# 2026-07-27 four-state clause (verified directly against classify_pair_verdict, not just
# against the old abs(mean)>2*sigma test this fixture originally targeted).
TARGET_MEANS = {
    ("B15", 42): 0.400, ("B15", 43): 0.420, ("B15", 44): 0.410,
    ("B3", 42): 0.350, ("B3", 43): 0.370, ("B3", 44): 0.360,
    ("B15P", 42): 0.400, ("B15P", 43): 0.400, ("B15P", 44): 0.385,
    ("B15D", 42): 0.500, ("B15D", 43): 0.520, ("B15D", 44): 0.510,
}


def _write_full_screen(tmp_path: Path, *, overrides: dict | None = None) -> tuple[Path, Path]:
    """Write all 12 valid v4 artifacts. ``overrides[(variant, seed)]`` may supply kwargs
    forwarded to _write_artifact for exactly that run (used by the negative tests below)."""
    overrides = overrides or {}
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints_dir.mkdir()
    for variant in agg.VARIANTS:
        for seed in agg.SEEDS:
            kwargs = dict(overrides.get((variant, seed), {}))
            _write_artifact(
                results_dir, checkpoints_dir, variant, seed, TARGET_MEANS[(variant, seed)], **kwargs
            )
    return results_dir, checkpoints_dir


def test_full_pipeline_produces_all_three_verdicts_simultaneously(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)

    payload = agg.run_aggregation(results_dir)

    assert payload["paired_deltas"]["B15_minus_B3"]["verdict"] == "effective"
    assert payload["paired_deltas"]["B15_minus_B15P"]["verdict"] == "indeterminate"
    assert payload["paired_deltas"]["B15_minus_B15D"]["verdict"] == "ineffective"

    # Point estimates (section 3 item 1) match the hand-designed constants.
    b15_b3 = payload["paired_deltas"]["B15_minus_B3"]
    assert b15_b3["mean_delta"] == pytest.approx(0.05)
    assert len(b15_b3["per_session_seed_mean"]) == 6
    assert set(b15_b3["per_session_seed_mean"]) == set(VAL_SESSIONS)
    assert all(value == pytest.approx(0.05) for value in b15_b3["per_session_seed_mean"].values())
    assert b15_b3["n_sessions_positive"] == 6
    assert b15_b3["all_seed_means_positive"] is True

    b15_b15p = payload["paired_deltas"]["B15_minus_B15P"]
    assert b15_b15p["mean_delta"] == pytest.approx(0.005)

    b15_b15d = payload["paired_deltas"]["B15_minus_B15D"]
    assert b15_b15d["mean_delta"] == pytest.approx(-0.10)
    assert b15_b15d["n_sessions_positive"] == 0

    # Uncertainty block (section 3): all three items present, all measured (nonzero,
    # nothing borrowed from the doc's priors of 0.0388 / 0.0112).
    uncertainty = payload["uncertainty"]
    assert set(uncertainty["within_window_std_per_run"]) == {
        f"{variant}_s{seed}" for variant in agg.VARIANTS for seed in agg.SEEDS
    }
    assert all(value > 0 for value in uncertainty["within_window_std_per_run"].values())
    assert set(uncertainty["across_seed_std_per_variant"]) == set(agg.VARIANTS)
    assert all(value > 0 for value in uncertainty["across_seed_std_per_variant"].values())
    expected_pairs = {"B15_minus_B3", "B15_minus_B15P", "B15_minus_B15D"}
    assert set(uncertainty["sigma_delta_paired_per_pair"]) == expected_pairs
    assert set(uncertainty["sigma_delta_unpaired_quadrature_per_pair"]) == expected_pairs
    assert set(uncertainty["implied_seed_correlation_per_pair"]) == expected_pairs

    # Absolute + per-seed variant scores (requirement 5).
    assert payload["variant_scores"]["B15"]["42"] == pytest.approx(0.400)
    assert payload["variant_scores"]["B15"]["mean"] == pytest.approx((0.40 + 0.42 + 0.41) / 3)

    # Never collapse indeterminate into a bool.
    for pair_payload in payload["paired_deltas"].values():
        assert pair_payload["verdict"] in ("effective", "ineffective", "indeterminate")
        assert not isinstance(pair_payload["verdict"], bool)


def test_pipeline_is_deterministic_across_repeated_aggregation(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)
    first = agg.run_aggregation(results_dir)
    second = agg.run_aggregation(results_dir)
    assert first == second


def test_full_pipeline_sigma_delta_matches_standard_error_helper(tmp_path):
    # End-to-end pin: BOTH sigma_delta estimates actually written into paired_deltas/
    # uncertainty by run_aggregation must equal sigma_delta_standard_error(...) /
    # sigma_delta_paired(...) applied to this same payload's own across_seed_std_per_variant /
    # per_seed_mean, and the verdict's ineffective clause must be gated on the PAIRED value
    # (2026-07-26 revision), not the unpaired quadrature (M7).
    results_dir, _ = _write_full_screen(tmp_path)
    payload = agg.run_aggregation(results_dir)
    across = payload["uncertainty"]["across_seed_std_per_variant"]
    for treatment, control in agg.PAIRS:
        pair_name = f"{treatment}_minus_{control}"
        pair = payload["paired_deltas"][pair_name]
        expected_unpaired = agg.sigma_delta_standard_error(across[treatment], across[control], len(agg.SEEDS))
        expected_paired = agg.sigma_delta_paired(list(pair["per_seed_mean"].values()))
        expected_rho = agg.implied_seed_correlation(
            sigma_a=across[treatment], sigma_b=across[control],
            per_seed_deltas=list(pair["per_seed_mean"].values()),
        )
        raw_quadrature = (across[treatment] ** 2 + across[control] ** 2) ** 0.5

        assert pair["sigma_delta_unpaired_quadrature"] == pytest.approx(expected_unpaired)
        assert pair["sigma_delta_paired"] == pytest.approx(expected_paired)
        assert pair["implied_seed_correlation"] == pytest.approx(expected_rho)
        assert payload["uncertainty"]["sigma_delta_unpaired_quadrature_per_pair"][pair_name] == pytest.approx(expected_unpaired)
        assert payload["uncertainty"]["sigma_delta_paired_per_pair"][pair_name] == pytest.approx(expected_paired)
        assert payload["uncertainty"]["implied_seed_correlation_per_pair"][pair_name] == pytest.approx(expected_rho)
        if raw_quadrature > 0:
            assert pair["sigma_delta_unpaired_quadrature"] < raw_quadrature

        assert pair["ineffective_abs_threshold"] == pytest.approx(2.0 * expected_paired)


# ----------------------------------------------------------------------------------------
# Section 2 consistency gate: negative tests.
# ----------------------------------------------------------------------------------------
def test_rejects_missing_artifact(tmp_path):
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints_dir.mkdir()
    for variant in agg.VARIANTS:
        for seed in agg.SEEDS:
            if (variant, seed) == ("B15", 44):
                continue  # simulate a still-running / crashed run
            _write_artifact(results_dir, checkpoints_dir, variant, seed, TARGET_MEANS[(variant, seed)])

    with pytest.raises(FileNotFoundError, match="epoch_window_b15_s44"):
        agg.run_aggregation(results_dir)


def test_rejects_mismatched_session_splits(tmp_path):
    bad_splits = dict(SESSION_SPLITS)
    bad_splits["val"] = list(VAL_SESSIONS[:-1]) + ["sub-X_ses-CO-99999999"]
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path, overrides={("B15", 44): {"session_splits": bad_splits}}
    )
    with pytest.raises(ValueError, match="session_splits disagree"):
        agg.run_aggregation(results_dir)


def test_rejects_two_runs_sharing_a_run_directory(tmp_path):
    # This is exactly v3 bug H.4: two (variant, seed) runs resolving to one run_dir.
    tmp_path_ckpts = tmp_path / "checkpoints"
    shared_dir = tmp_path_ckpts / "shared_run_dir"
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={
            ("B15", 43): {"run_dir_override": shared_dir},
            ("B15", 44): {"run_dir_override": shared_dir},
        },
    )
    with pytest.raises(ValueError, match="share a run directory"):
        agg.run_aggregation(results_dir)


def test_rejects_wrong_max_epochs(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path, overrides={("B3", 42): {"run_metadata_overrides": {"training": {"max_epochs": 20}}}}
    )
    with pytest.raises(ValueError, match="max_epochs"):
        agg.run_aggregation(results_dir)


def test_rejects_early_stopping_enabled(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={("B15P", 43): {"run_metadata_overrides": {"training": {"no_early_stopping": False}}}},
    )
    with pytest.raises(ValueError, match="no_early_stopping"):
        agg.run_aggregation(results_dir)


def test_rejects_tampered_run_metadata_after_evaluation(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(tmp_path)
    # Simulate the training run_metadata.json being rewritten after
    # eval_epoch_window_dandi688.py already hashed it.
    tampered_run_dir = checkpoints_dir / "attention_arch_screen_v4_b3_dandi688_co_s42"
    (tampered_run_dir / "run_metadata.json").write_text(
        json.dumps({
            "status": "completed",
            "held_out_test_evaluated": False,
            "training": {"max_epochs": 12, "no_early_stopping": True, "checkpoint_every_epoch": True},
            "tampered": True,
        })
    )
    with pytest.raises(ValueError, match="run_metadata_sha256 mismatch"):
        agg.run_aggregation(results_dir)


def test_rejects_incomplete_results_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        agg.run_aggregation(tmp_path / "nonexistent")
