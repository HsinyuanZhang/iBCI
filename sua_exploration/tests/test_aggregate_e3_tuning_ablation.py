"""Tests for aggregate_e3_tuning_ablation.py.

Structured like test_aggregate_side_feature_ablation_v2.py (verdict-logic unit tests, then a
full synthetic pipeline, then consistency-rejection paths), plus tests specific to what this
aggregator generalizes relative to v2: --seeds and --effective_mean_delta are required CLI
inputs rather than hardcoded module constants (SEEDS=(42,43,44),
EFFECTIVE_MEAN_DELTA_THRESHOLD=0.03), and the epoch window/total-epoch budget is validated by
internal self-consistency plus cross-artifact agreement rather than a hardcoded
EXPECTED_EPOCH_WINDOW=[5..12]. The full-pipeline fixtures below deliberately use a 4-seed,
20-epoch/burn-in-10 window -- neither number matches the frozen v2 screen's 3 seeds / 12
epochs / window 5-12 -- specifically so a regression to a hardcoded constant would be caught.

No GPU, no NWB data, no torch: like aggregate_side_feature_ablation_v2.py, this aggregator
only reads JSON, so these tests only need pytest and the standard library.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_e3_tuning_ablation as agg  # noqa: E402
import aggregate_side_feature_ablation_v2 as agg_v2  # noqa: E402


# ----------------------------------------------------------------------------------------
# pair_meets_effective_clause / pair_exceeds_ineffective_threshold: the per-pair predicates.
# Signatures differ from v2's: effective_mean_delta_threshold and n_seeds are now explicit
# arguments (no module-level EFFECTIVE_MEAN_DELTA_THRESHOLD/SEEDS to read from).
# ----------------------------------------------------------------------------------------
def test_pair_meets_effective_clause_true_case():
    assert agg.pair_meets_effective_clause(
        mean_delta=0.05, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=[0.04, 0.05, 0.06], effective_mean_delta_threshold=0.03, n_seeds=3,
    ) is True


def test_pair_meets_effective_clause_inclusive_boundaries():
    # mean_delta exactly equal to the threshold and exactly 5/6 positive: both use >=.
    assert agg.pair_meets_effective_clause(
        mean_delta=0.03, n_sessions_positive=5, n_sessions_total=6,
        per_seed_means=[0.001, 0.002, 0.003], effective_mean_delta_threshold=0.03, n_seeds=3,
    ) is True


def test_pair_meets_effective_clause_false_when_mean_delta_just_under_threshold():
    assert agg.pair_meets_effective_clause(
        mean_delta=0.0299999, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=[0.03, 0.03, 0.03], effective_mean_delta_threshold=0.03, n_seeds=3,
    ) is False


def test_pair_meets_effective_clause_false_when_four_of_six_positive():
    assert agg.pair_meets_effective_clause(
        mean_delta=0.04, n_sessions_positive=4, n_sessions_total=6,
        per_seed_means=[0.03, 0.04, 0.05], effective_mean_delta_threshold=0.03, n_seeds=3,
    ) is False


def test_pair_meets_effective_clause_false_when_one_seed_mean_not_positive():
    assert agg.pair_meets_effective_clause(
        mean_delta=0.05, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=[0.10, 0.10, 0.0], effective_mean_delta_threshold=0.03, n_seeds=3,
    ) is False


def test_pair_meets_effective_clause_rejects_wrong_session_total():
    with pytest.raises(ValueError, match="n_sessions_total"):
        agg.pair_meets_effective_clause(
            mean_delta=0.05, n_sessions_positive=5, n_sessions_total=5,
            per_seed_means=[0.1, 0.1, 0.1], effective_mean_delta_threshold=0.03, n_seeds=3,
        )


def test_pair_meets_effective_clause_rejects_wrong_seed_count():
    with pytest.raises(ValueError, match="per_seed_means"):
        agg.pair_meets_effective_clause(
            mean_delta=0.05, n_sessions_positive=5, n_sessions_total=6,
            per_seed_means=[0.1, 0.1], effective_mean_delta_threshold=0.03, n_seeds=3,
        )


def test_pair_meets_effective_clause_threshold_is_a_live_argument_not_hardcoded():
    # Same mean_delta/sessions/seed-means; only the threshold argument changes. If the
    # threshold were hardcoded (regressing to v2's module constant) this would not change.
    kwargs = dict(mean_delta=0.05, n_sessions_positive=6, n_sessions_total=6,
                  per_seed_means=[0.05, 0.05, 0.05], n_seeds=3)
    assert agg.pair_meets_effective_clause(effective_mean_delta_threshold=0.03, **kwargs) is True
    assert agg.pair_meets_effective_clause(effective_mean_delta_threshold=0.10, **kwargs) is False


def test_pair_meets_effective_clause_n_seeds_is_a_live_argument_not_hardcoded():
    # 5 per-seed means is valid when n_seeds=5 (not just the frozen screen's 3).
    assert agg.pair_meets_effective_clause(
        mean_delta=0.05, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=[0.05] * 5, effective_mean_delta_threshold=0.03, n_seeds=5,
    ) is True
    with pytest.raises(ValueError, match="per_seed_means"):
        agg.pair_meets_effective_clause(
            mean_delta=0.05, n_sessions_positive=6, n_sessions_total=6,
            per_seed_means=[0.05] * 5, effective_mean_delta_threshold=0.03, n_seeds=3,
        )


def test_pair_exceeds_ineffective_threshold_true_case():
    assert agg.pair_exceeds_ineffective_threshold(mean_delta=-0.05, sigma_delta=0.01) is True


def test_pair_exceeds_ineffective_threshold_boundary_is_strict():
    assert agg.pair_exceeds_ineffective_threshold(mean_delta=0.02, sigma_delta=0.01) is False
    assert agg.pair_exceeds_ineffective_threshold(mean_delta=0.0200001, sigma_delta=0.01) is True


def test_pair_exceeds_ineffective_threshold_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma_delta"):
        agg.pair_exceeds_ineffective_threshold(mean_delta=0.05, sigma_delta=-0.01)


# ----------------------------------------------------------------------------------------
# classify_tuning_group_verdict: the group-level three-state combination of two pairs
# (T4/T8 vs {F0, own TSx control}) -- same logic as aggregate_side_feature_ablation_v2's
# classify_feature_group_verdict, renamed for the E3 domain.
# ----------------------------------------------------------------------------------------
def test_group_effective_when_both_pairs_effective():
    assert agg.classify_tuning_group_verdict(
        pair_a_effective=True, pair_a_ineffective_threshold=False,
        pair_b_effective=True, pair_b_ineffective_threshold=False,
    ) == "effective"


def test_group_not_effective_when_only_one_pair_effective():
    verdict = agg.classify_tuning_group_verdict(
        pair_a_effective=True, pair_a_ineffective_threshold=False,
        pair_b_effective=False, pair_b_ineffective_threshold=False,
    )
    assert verdict != "effective"


def test_group_ineffective_when_both_pairs_confidently_resolved_negative():
    assert agg.classify_tuning_group_verdict(
        pair_a_effective=False, pair_a_ineffective_threshold=True,
        pair_b_effective=False, pair_b_ineffective_threshold=True,
    ) == "ineffective"


def test_group_indeterminate_when_only_one_pair_resolved_ineffective():
    verdict = agg.classify_tuning_group_verdict(
        pair_a_effective=False, pair_a_ineffective_threshold=True,
        pair_b_effective=False, pair_b_ineffective_threshold=False,
    )
    assert verdict == "indeterminate"
    verdict_swapped = agg.classify_tuning_group_verdict(
        pair_a_effective=False, pair_a_ineffective_threshold=False,
        pair_b_effective=False, pair_b_ineffective_threshold=True,
    )
    assert verdict_swapped == "indeterminate"


def test_group_indeterminate_when_neither_pair_resolved():
    assert agg.classify_tuning_group_verdict(
        pair_a_effective=False, pair_a_ineffective_threshold=False,
        pair_b_effective=False, pair_b_ineffective_threshold=False,
    ) == "indeterminate"


def test_group_verdict_is_always_one_of_the_three_literal_strings():
    for pair_a_effective in (True, False):
        for pair_a_ineffective in (True, False):
            for pair_b_effective in (True, False):
                for pair_b_ineffective in (True, False):
                    verdict = agg.classify_tuning_group_verdict(
                        pair_a_effective=pair_a_effective,
                        pair_a_ineffective_threshold=pair_a_ineffective,
                        pair_b_effective=pair_b_effective,
                        pair_b_ineffective_threshold=pair_b_ineffective,
                    )
                    assert verdict in ("effective", "ineffective", "indeterminate")
                    assert not isinstance(verdict, bool)
                    assert isinstance(verdict, str)


# ----------------------------------------------------------------------------------------
# sample_std / parse_seeds / sigma_delta_standard_error reuse.
# ----------------------------------------------------------------------------------------
def test_sample_std_matches_statistics_stdev():
    import statistics

    values = [0.20, 0.22, 0.21]
    assert agg.sample_std(values) == pytest.approx(statistics.stdev(values))


def test_sample_std_requires_at_least_two_values():
    with pytest.raises(ValueError, match="at least 2"):
        agg.sample_std([0.5])


def test_parse_seeds_basic():
    assert agg.parse_seeds("42,43,44") == [42, 43, 44]
    assert agg.parse_seeds(" 42 , 43 ") == [42, 43]


def test_parse_seeds_rejects_empty():
    with pytest.raises(ValueError, match="at least one seed"):
        agg.parse_seeds("")


def test_parse_seeds_rejects_non_integer():
    with pytest.raises(ValueError, match="comma-separated list of integers"):
        agg.parse_seeds("42,abc")


def test_parse_seeds_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        agg.parse_seeds("42,43,42")


def test_sigma_delta_standard_error_is_imported_from_v2_unmodified():
    # E3_E4_ENCODER_PROGRAM.md: "import or replicate it, do not regress it". This pins that
    # aggregate_e3_tuning_ablation genuinely reuses the same function object rather than
    # shadowing it with a divergent reimplementation.
    assert agg.sigma_delta_standard_error is agg_v2.sigma_delta_standard_error


def test_sigma_delta_standard_error_matches_hand_computed_value():
    import math

    result = agg.sigma_delta_standard_error(0.03, 0.04, 3)
    assert result == pytest.approx(0.05 / math.sqrt(3))


# ----------------------------------------------------------------------------------------
# sigma_delta_paired / implied_seed_correlation (2026-07-26 revision): same reuse contract as
# sigma_delta_standard_error above -- imported unmodified from v2, not reimplemented. The
# exhaustive edge-case coverage (n_seeds<2 raise, correlation formula, the synthetic
# different-verdicts case) lives in test_aggregate_side_feature_ablation_v2.py; here we only
# pin that this module genuinely reuses those same function objects plus one sanity value.
# ----------------------------------------------------------------------------------------
def test_sigma_delta_paired_is_imported_from_v2_unmodified():
    assert agg.sigma_delta_paired is agg_v2.sigma_delta_paired


def test_implied_seed_correlation_is_imported_from_v2_unmodified():
    assert agg.implied_seed_correlation is agg_v2.implied_seed_correlation


def test_sigma_delta_paired_matches_hand_computed_value():
    import math

    result = agg.sigma_delta_paired([0.01, 0.03, 0.05])
    assert result == pytest.approx(0.02 / math.sqrt(3))


def test_sigma_delta_paired_requires_at_least_two_seeds():
    with pytest.raises(ValueError, match="at least 2 seeds"):
        agg.sigma_delta_paired([0.05])


# ----------------------------------------------------------------------------------------
# Per-run helpers (no file I/O). Deliberately exercised with a NON-frozen epoch window
# (10, 20, ...) to prove they read the window from the artifact itself, not a hardcoded
# EXPECTED_EPOCH_WINDOW.
# ----------------------------------------------------------------------------------------
def _toy_payload(session_r2_by_epoch: dict[int, dict[str, float]]) -> dict:
    per_epoch = {
        str(epoch): {"per_session_r2": sessions, "mean_r2": agg.mean(list(sessions.values()))}
        for epoch, sessions in session_r2_by_epoch.items()
    }
    per_epoch_mean_r2 = {key: value["mean_r2"] for key, value in per_epoch.items()}
    epoch_list = sorted(session_r2_by_epoch.keys())
    return {
        "session_splits": {"val": sorted(next(iter(session_r2_by_epoch.values())).keys())},
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": per_epoch_mean_r2,
        "epoch_list": epoch_list,
    }


def test_per_run_session_scores_averages_over_the_artifacts_own_epoch_window():
    payload = _toy_payload({
        11: {"sA": 0.10, "sB": 0.20}, 12: {"sA": 0.20, "sB": 0.30},
        13: {"sA": 0.30, "sB": 0.40}, 14: {"sA": 0.40, "sB": 0.50},
    })
    scores = agg.per_run_session_scores(payload)
    assert scores["sA"] == pytest.approx((0.10 + 0.20 + 0.30 + 0.40) / 4)
    assert scores["sB"] == pytest.approx((0.20 + 0.30 + 0.40 + 0.50) / 4)


def test_per_run_within_window_std_uses_the_artifacts_own_epoch_list():
    payload = _toy_payload({epoch: {"sA": 0.1 * (epoch - 10)} for epoch in range(11, 21)})
    std = agg.per_run_within_window_std(payload)
    import statistics

    assert std == pytest.approx(statistics.stdev([0.1 * i for i in range(1, 11)]))


# ----------------------------------------------------------------------------------------
# Full synthetic pipeline through run_aggregation(). Deliberately uses 4 seeds (not v2's 3)
# and a 20-epoch/burn-in-10 window (not the frozen script's 12/5-12), so a regression to a
# hardcoded SEEDS/EXPECTED_EPOCH_WINDOW would be caught.
# ----------------------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45]
TOTAL_EPOCHS = 20
BURN_IN = 10
EPOCH_WINDOW = list(range(BURN_IN + 1, TOTAL_EPOCHS + 1))  # 11..20, 10 epochs
assert EPOCH_WINDOW == list(range(11, 21))
PROTOCOL = {
    "name": "fixed_epoch_window_deterministic_checkpoint_rule",
    "description": "test fixture",
    "total_epochs": TOTAL_EPOCHS,
    "epoch_window": EPOCH_WINDOW,
    "burn_in_epochs": BURN_IN,
    "selection_mode": "first",
    "calibration_n": 30,
    "pool_size": 50,
    "protocol_metric_source": (
        "select_gradient_free_protocol_dandi688.evaluate_fixed_protocol_over_validation_sessions"
    ),
}

VAL_SESSIONS = [f"sub-X_ses-CO-2015110{i}" for i in range(1, 7)]
TRAIN_SESSIONS = [f"sub-X_ses-CO-2013100{i}" for i in range(1, 4)]
TEST_SESSIONS = [f"sub-X_ses-CO-2015120{i}" for i in range(1, 7)]
SESSION_SPLITS = {"train": TRAIN_SESSIONS, "val": VAL_SESSIONS, "test": TEST_SESSIONS}

SIDE_DIM = {"none": 0, "t4": 4, "t8": 8, "ts4": 4, "ts8": 8}


def _epoch_values_with_exact_mean(target_mean: float, n: int) -> list[float]:
    """n values alternating +/-0.01 around target_mean (n even); the mean is exactly
    target_mean."""
    assert n % 2 == 0, "test fixture requires an even-length epoch window"
    wiggle = [-0.01, 0.01] * (n // 2)
    return [target_mean + delta for delta in wiggle]


def _write_artifact(
    results_dir: Path,
    checkpoints_dir: Path,
    group: str,
    seed: int,
    target_mean: float,
    *,
    run_metadata_overrides: dict | None = None,
    run_dir_override: Path | None = None,
    session_splits: dict | None = None,
    protocol: dict | None = None,
) -> Path:
    contract = agg.GROUP_CONTRACT[group]
    run_dir = run_dir_override or (
        checkpoints_dir / f"e3_tuning_ablation_{group.lower()}_dandi688_co_s{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    proto = protocol or PROTOCOL
    run_metadata_path = run_dir / "run_metadata.json"
    metadata_payload = {
        "status": "completed",
        "held_out_test_evaluated": False,
        "training": {
            "max_epochs": proto["total_epochs"],
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
        },
        "side_features": {
            "group": contract["side_features_group"],
            "pool_size": 50,
            "side_dim": SIDE_DIM[contract["side_features_group"]],
            "permutation_seed": seed if contract["side_features_group"] in ("ts4", "ts8") else None,
        },
    }
    if run_metadata_overrides:
        training_overrides = run_metadata_overrides.pop("training", None)
        side_overrides = run_metadata_overrides.pop("side_features", None)
        metadata_payload.update(run_metadata_overrides)
        if training_overrides:
            metadata_payload["training"].update(training_overrides)
        if side_overrides:
            metadata_payload["side_features"].update(side_overrides)
    run_metadata_path.write_text(json.dumps(metadata_payload, sort_keys=True))

    splits = session_splits or SESSION_SPLITS
    epoch_window = proto["epoch_window"]
    epoch_values = _epoch_values_with_exact_mean(target_mean, len(epoch_window))
    per_epoch = {}
    per_epoch_mean_r2 = {}
    for epoch, value in zip(epoch_window, epoch_values):
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
        "generated_by": "eval_epoch_window_generic_dandi688.py",
        "run_dir": str(run_dir),
        "run_metadata_path": str(run_metadata_path),
        "run_metadata_sha256": agg.sha256_file(run_metadata_path),
        "variant": contract["variant"],
        "seed": seed,
        "task": "CO",
        "protocol": dict(proto),
        "epoch_list": epoch_window,
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": per_epoch_mean_r2,
        "variant_score": variant_score,
        "session_splits": splits,
        "calibration_trial_selection_uses_behavior_labels": False,
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
        "no_test_files_evaluated": True,
    }
    out_path = agg.artifact_path(results_dir, group, seed)
    out_path.write_text(json.dumps(payload, sort_keys=True))
    return out_path


# Per-seed target means hand-chosen (same pattern as
# test_aggregate_side_feature_ablation_v2.py's F1/F2 fixture, extended to 4 seeds) so that,
# in ONE consistent dataset (F0 shared between both T4's and T8's comparisons):
#   T4 -> effective    (T4-F0 mean_delta=+0.06, T4-TS4 mean_delta=+0.04; both >= +0.03,
#                        6/6 sessions positive, all 4 seed means positive)
#   T8 -> ineffective  (T8-F0 and T8-TS8 both mean_delta=-0.10, far outside sigma_delta)
TARGET_MEANS = {
    ("F0", 42): 0.300, ("F0", 43): 0.320, ("F0", 44): 0.310, ("F0", 45): 0.305,
    ("T4", 42): 0.360, ("T4", 43): 0.380, ("T4", 44): 0.370, ("T4", 45): 0.365,
    ("T8", 42): 0.200, ("T8", 43): 0.220, ("T8", 44): 0.210, ("T8", 45): 0.205,
    ("TS4", 42): 0.320, ("TS4", 43): 0.340, ("TS4", 44): 0.330, ("TS4", 45): 0.325,
    ("TS8", 42): 0.300, ("TS8", 43): 0.320, ("TS8", 44): 0.310, ("TS8", 45): 0.305,
}
EFFECTIVE_MEAN_DELTA = 0.03


def _write_full_screen(tmp_path: Path, *, overrides: dict | None = None) -> tuple[Path, Path]:
    overrides = overrides or {}
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints_dir.mkdir()
    for group in agg.GROUPS:
        for seed in SEEDS:
            kwargs = dict(overrides.get((group, seed), {}))
            _write_artifact(
                results_dir, checkpoints_dir, group, seed, TARGET_MEANS[(group, seed)], **kwargs
            )
    return results_dir, checkpoints_dir


def test_full_pipeline_produces_effective_t4_and_ineffective_t8(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)

    payload = agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)

    assert payload["seeds"] == SEEDS
    assert payload["epoch_window"] == EPOCH_WINDOW
    assert payload["epoch_budget"] == TOTAL_EPOCHS
    assert payload["burn_in_epochs"] == BURN_IN
    assert payload["effective_mean_delta_threshold"] == EFFECTIVE_MEAN_DELTA

    assert payload["tuning_group_verdicts"]["T4"]["verdict"] == "effective"
    assert payload["tuning_group_verdicts"]["T8"]["verdict"] == "ineffective"

    t4_f0 = payload["paired_deltas"]["T4_minus_F0"]
    assert t4_f0["mean_delta"] == pytest.approx(0.06)
    assert t4_f0["n_sessions_positive"] == 6
    assert t4_f0["all_seed_means_positive"] is True
    assert t4_f0["meets_effective_clause"] is True
    assert set(t4_f0["per_session_seed_mean"]) == set(VAL_SESSIONS)

    t4_ts4 = payload["paired_deltas"]["T4_minus_TS4"]
    assert t4_ts4["mean_delta"] == pytest.approx(0.04)
    assert t4_ts4["meets_effective_clause"] is True

    t8_f0 = payload["paired_deltas"]["T8_minus_F0"]
    assert t8_f0["mean_delta"] == pytest.approx(-0.10)
    assert t8_f0["meets_effective_clause"] is False
    assert t8_f0["exceeds_ineffective_threshold"] is True

    t8_ts8 = payload["paired_deltas"]["T8_minus_TS8"]
    assert t8_ts8["mean_delta"] == pytest.approx(-0.10)
    assert t8_ts8["exceeds_ineffective_threshold"] is True

    # Never (T4, TS8) or (T8, TS4) -- dimension-matched pairs only.
    assert set(payload["paired_deltas"]) == {
        "T4_minus_F0", "T4_minus_TS4", "T8_minus_F0", "T8_minus_TS8",
    }

    uncertainty = payload["uncertainty"]
    assert set(uncertainty["within_window_std_per_run"]) == {
        f"{group}_s{seed}" for group in agg.GROUPS for seed in SEEDS
    }
    assert all(value > 0 for value in uncertainty["within_window_std_per_run"].values())

    assert payload["variant_scores"]["T4"]["42"] == pytest.approx(0.36)
    assert payload["variant_scores"]["F0"]["mean"] == pytest.approx((0.300 + 0.320 + 0.310 + 0.305) / 4)

    for group_payload in payload["tuning_group_verdicts"].values():
        assert group_payload["verdict"] in ("effective", "ineffective", "indeterminate")
        assert not isinstance(group_payload["verdict"], bool)


def test_pipeline_is_deterministic_across_repeated_aggregation(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)
    first = agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)
    second = agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)
    assert first == second


def test_full_pipeline_sigma_delta_matches_standard_error_helper(tmp_path):
    # End-to-end pin: BOTH sigma_delta estimates written into paired_deltas/uncertainty must
    # match sigma_delta_standard_error(...) / sigma_delta_paired(...) applied to this same
    # payload's own data, and the ineffective clause must be gated on the PAIRED value
    # (2026-07-26 revision), not the unpaired quadrature (M7).
    results_dir, _ = _write_full_screen(tmp_path)
    payload = agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)
    across = payload["uncertainty"]["across_seed_std_per_group"]
    for treatment, control in agg.PAIRS:
        pair_name = f"{treatment}_minus_{control}"
        pair = payload["paired_deltas"][pair_name]
        expected_unpaired = agg.sigma_delta_standard_error(across[treatment], across[control], len(SEEDS))
        expected_paired = agg.sigma_delta_paired(list(pair["per_seed_mean"].values()))
        expected_rho = agg.implied_seed_correlation(
            sigma_a=across[treatment], sigma_b=across[control],
            per_seed_deltas=list(pair["per_seed_mean"].values()),
        )
        assert pair["sigma_delta_unpaired_quadrature"] == pytest.approx(expected_unpaired)
        assert pair["sigma_delta_paired"] == pytest.approx(expected_paired)
        assert pair["implied_seed_correlation"] == pytest.approx(expected_rho)
        assert payload["uncertainty"]["sigma_delta_unpaired_quadrature_per_pair"][pair_name] == pytest.approx(expected_unpaired)
        assert payload["uncertainty"]["sigma_delta_paired_per_pair"][pair_name] == pytest.approx(expected_paired)
        assert payload["uncertainty"]["implied_seed_correlation_per_pair"][pair_name] == pytest.approx(expected_rho)
        assert pair["ineffective_abs_threshold"] == pytest.approx(2.0 * expected_paired)
        assert pair["exceeds_ineffective_threshold"] == agg.pair_exceeds_ineffective_threshold(
            mean_delta=pair["mean_delta"], sigma_delta=expected_paired
        )


def test_full_pipeline_threshold_is_a_live_cli_argument_not_hardcoded(tmp_path):
    # Same underlying data, only --effective_mean_delta changes: at 0.03 T4 is "effective"
    # (per test_full_pipeline_produces_effective_t4_and_ineffective_t8); raising the
    # threshold above T4's own +0.06/+0.04 deltas must flip it away from "effective". If the
    # threshold were hardcoded to 0.03 (a v2 regression), this would not change.
    results_dir, _ = _write_full_screen(tmp_path)
    low = agg.run_aggregation(results_dir, SEEDS, 0.03)
    high = agg.run_aggregation(results_dir, SEEDS, 0.07)
    assert low["tuning_group_verdicts"]["T4"]["verdict"] == "effective"
    assert high["tuning_group_verdicts"]["T4"]["verdict"] != "effective"
    assert low["effective_mean_delta_threshold"] == 0.03
    assert high["effective_mean_delta_threshold"] == 0.07


def test_full_pipeline_seeds_argument_selects_which_runs_are_aggregated(tmp_path):
    # All 4 seeds are on disk; aggregating over only 3 of them must use exactly those 3, not
    # silently discover and use all 4 (which would hide a --seeds mistake or a partially
    # complete screen).
    results_dir, _ = _write_full_screen(tmp_path)
    subset = [42, 43, 44]
    payload = agg.run_aggregation(results_dir, subset, EFFECTIVE_MEAN_DELTA)
    assert payload["seeds"] == subset
    assert set(payload["variant_scores"]["F0"]) == {"42", "43", "44", "mean"}
    expected_mean = agg.mean([TARGET_MEANS[("F0", s)] for s in subset])
    assert payload["variant_scores"]["F0"]["mean"] == pytest.approx(expected_mean)


def test_requesting_a_seed_not_on_disk_raises_file_not_found(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)
    with pytest.raises(FileNotFoundError, match="f0_s46"):
        agg.run_aggregation(results_dir, [42, 43, 44, 45, 46], EFFECTIVE_MEAN_DELTA)


def test_run_aggregation_rejects_non_positive_effective_mean_delta(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)
    with pytest.raises(ValueError, match="effective_mean_delta"):
        agg.run_aggregation(results_dir, SEEDS, 0.0)
    with pytest.raises(ValueError, match="effective_mean_delta"):
        agg.run_aggregation(results_dir, SEEDS, -0.01)


def test_run_aggregation_rejects_empty_seeds(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)
    with pytest.raises(ValueError, match="at least one seed"):
        agg.run_aggregation(results_dir, [], EFFECTIVE_MEAN_DELTA)


# ----------------------------------------------------------------------------------------
# Consistency gate: negative tests.
# ----------------------------------------------------------------------------------------
def test_rejects_missing_artifact(tmp_path):
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints_dir.mkdir()
    for group in agg.GROUPS:
        for seed in SEEDS:
            if (group, seed) == ("TS8", 45):
                continue  # simulate a still-running / crashed run
            _write_artifact(results_dir, checkpoints_dir, group, seed, TARGET_MEANS[(group, seed)])

    with pytest.raises(FileNotFoundError, match="ts8_s45"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_mismatched_session_splits(tmp_path):
    bad_splits = dict(SESSION_SPLITS)
    bad_splits["val"] = list(VAL_SESSIONS[:-1]) + ["sub-X_ses-CO-99999999"]
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path, overrides={("TS8", 45): {"session_splits": bad_splits}}
    )
    with pytest.raises(ValueError, match="session_splits disagree"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_mismatched_epoch_window_across_artifacts(tmp_path):
    # Specific to this aggregator's generalization: one run used a different burn_in (hence
    # a different epoch_window) than the rest -- M2 requires an IDENTICAL budget for every
    # group/seed, so this must be rejected even though each artifact is internally
    # self-consistent.
    other_protocol = dict(PROTOCOL, burn_in_epochs=8, epoch_window=list(range(9, 21)))
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path, overrides={("TS8", 45): {"protocol": other_protocol}}
    )
    with pytest.raises(ValueError, match="protocol .*disagrees"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_internally_inconsistent_epoch_window(tmp_path):
    # protocol.epoch_window does not equal burn_in_epochs+1..total_epochs for its OWN
    # recorded total_epochs/burn_in_epochs -- an artifact this self-contradictory must be
    # rejected before it ever reaches cross-artifact comparison.
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints_dir.mkdir()
    _write_artifact(results_dir, checkpoints_dir, "F0", 42, TARGET_MEANS[("F0", 42)])
    path = agg.artifact_path(results_dir, "F0", 42)
    payload = json.loads(path.read_text())
    payload["protocol"]["epoch_window"] = [1, 2, 3]  # inconsistent with burn_in/total_epochs
    path.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ValueError, match="is not burn_in_epochs"):
        agg.run_aggregation(results_dir, [42], EFFECTIVE_MEAN_DELTA)


def test_rejects_two_runs_sharing_a_run_directory(tmp_path):
    shared_dir = tmp_path / "checkpoints" / "shared_run_dir"
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={
            ("T4", 43): {"run_dir_override": shared_dir},
            ("T4", 44): {"run_dir_override": shared_dir},
        },
    )
    with pytest.raises(ValueError, match="share a run directory"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_wrong_max_epochs(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path, overrides={("F0", 42): {"run_metadata_overrides": {"training": {"max_epochs": 999}}}}
    )
    with pytest.raises(ValueError, match="max_epochs"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_early_stopping_enabled(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={("T8", 43): {"run_metadata_overrides": {"training": {"no_early_stopping": False}}}},
    )
    with pytest.raises(ValueError, match="no_early_stopping"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_missing_checkpoint_every_epoch(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={("T8", 43): {"run_metadata_overrides": {"training": {"checkpoint_every_epoch": False}}}},
    )
    with pytest.raises(ValueError, match="checkpoint_every_epoch"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_side_features_group_contract_mismatch(tmp_path):
    # A file named t4_s42.json whose run_metadata.json actually recorded side_features.group
    # "t8" (e.g. a mislabeled/misrouted artifact) must be rejected, not silently aggregated
    # as if it were T4.
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={("T4", 42): {"run_metadata_overrides": {"side_features": {"group": "t8"}}}},
    )
    with pytest.raises(ValueError, match="side_features.group"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_tampered_run_metadata_after_evaluation(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(tmp_path)
    tampered_run_dir = checkpoints_dir / "e3_tuning_ablation_f0_dandi688_co_s42"
    (tampered_run_dir / "run_metadata.json").write_text(
        json.dumps({
            "status": "completed",
            "held_out_test_evaluated": False,
            "training": {"max_epochs": TOTAL_EPOCHS, "no_early_stopping": True, "checkpoint_every_epoch": True},
            "side_features": {"group": "none", "pool_size": 50, "side_dim": 0, "permutation_seed": None},
            "tampered": True,
        })
    )
    with pytest.raises(ValueError, match="run_metadata_sha256 mismatch"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_incomplete_results_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        agg.run_aggregation(tmp_path / "nonexistent", SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_variant_mismatch_in_artifact(tmp_path):
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints_dir.mkdir()
    _write_artifact(results_dir, checkpoints_dir, "T4", 42, TARGET_MEANS[("T4", 42)])
    path = agg.artifact_path(results_dir, "T4", 42)
    payload = json.loads(path.read_text())
    payload["variant"] = "B15"  # T4 must be B3S
    path.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ValueError, match="variant mismatch"):
        agg.run_aggregation(results_dir, [42], EFFECTIVE_MEAN_DELTA)
