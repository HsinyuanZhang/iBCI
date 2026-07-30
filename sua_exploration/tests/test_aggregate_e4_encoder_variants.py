"""Tests for aggregate_e4_encoder_variants.py.

Structured like test_aggregate_e3_tuning_ablation.py / test_aggregate_side_feature_
ablation_v2.py (verdict-logic unit tests, then a full synthetic pipeline, then
consistency-rejection paths), with two E4-specific additions:

1. classify_variant_verdict is the single-pair degenerate case of E3's two-pair AND rule
   (E4 has no shuffled control -- charter section 4), so its own three-state coverage is
   simpler: one pair, one verdict.
2. deployment_cost_profile() is regression-tested against the exact parameter_count/
   mac_per_session/support_state_bytes numbers already independently verified for this repo
   (E3_E4_ENCODER_PROGRAM.md section 2.0, N=64/T=100/M=30):
       B3   params=18,034  mac/session=13,017,088  support_state=  16,384 B
       B3T  params=12,402  mac/session= 4,507,648  support_state=  16,384 B
       B3A  params=18,099  mac/session=13,139,968  support_state= 491,520 B  (30x B3)
   and the full pipeline test checks the deployment cost is actually attached next to each
   variant's R2 verdict, not just computed and discarded.

Unlike the other two aggregators' test files, this one is not torch-free: deployment_cost_
profile imports and instantiates (CPU-only, untrained, no checkpoint, no GPU) encoder
modules from streaming_calibration_exp. The verdict-logic and consistency-gate tests below
never touch that path and would still pass without torch installed; only the
deployment_cost_profile/full-pipeline tests need it, exactly like this aggregator module
itself.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_e4_encoder_variants as agg  # noqa: E402
import aggregate_side_feature_ablation_v2 as agg_v2  # noqa: E402


# ----------------------------------------------------------------------------------------
# pair_meets_effective_clause / pair_exceeds_ineffective_threshold: identical contract to
# aggregate_e3_tuning_ablation's functions of the same name (own tests since these are
# separate module-level function objects, to guard against the two copies diverging).
# ----------------------------------------------------------------------------------------
def test_pair_meets_effective_clause_true_case():
    assert agg.pair_meets_effective_clause(
        mean_delta=0.05, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=[0.04, 0.05, 0.06], effective_mean_delta_threshold=0.03, n_seeds=3,
    ) is True


def test_pair_meets_effective_clause_inclusive_boundaries():
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
    kwargs = dict(mean_delta=0.05, n_sessions_positive=6, n_sessions_total=6,
                  per_seed_means=[0.05, 0.05, 0.05], n_seeds=3)
    assert agg.pair_meets_effective_clause(effective_mean_delta_threshold=0.03, **kwargs) is True
    assert agg.pair_meets_effective_clause(effective_mean_delta_threshold=0.10, **kwargs) is False


def test_pair_exceeds_ineffective_threshold_true_case():
    assert agg.pair_exceeds_ineffective_threshold(mean_delta=-0.05, sigma_delta=0.01) is True


def test_pair_exceeds_ineffective_threshold_boundary_is_strict():
    assert agg.pair_exceeds_ineffective_threshold(mean_delta=0.02, sigma_delta=0.01) is False
    assert agg.pair_exceeds_ineffective_threshold(mean_delta=0.0200001, sigma_delta=0.01) is True


def test_pair_exceeds_ineffective_threshold_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma_delta"):
        agg.pair_exceeds_ineffective_threshold(mean_delta=0.05, sigma_delta=-0.01)


# ----------------------------------------------------------------------------------------
# classify_variant_verdict: single-pair three-state verdict (no AND-of-two-pairs -- E4 has
# no shuffled control, charter section 4). Exhaustive over all 4 (meets_effective,
# exceeds_ineffective_threshold) combinations.
# ----------------------------------------------------------------------------------------
def test_variant_effective_when_pair_meets_effective_clause():
    assert agg.classify_variant_verdict(meets_effective=True, exceeds_ineffective_threshold=False) == "effective"


def test_variant_effective_takes_priority_even_if_ineffective_threshold_also_true():
    # meets_effective and exceeds_ineffective_threshold are not mutually exclusive as
    # booleans (a large positive mean_delta can exceed 2*sigma too); effective must win.
    assert agg.classify_variant_verdict(meets_effective=True, exceeds_ineffective_threshold=True) == "effective"


def test_variant_ineffective_when_pair_exceeds_threshold_and_not_effective():
    assert agg.classify_variant_verdict(meets_effective=False, exceeds_ineffective_threshold=True) == "ineffective"


def test_variant_indeterminate_when_neither_resolved():
    assert agg.classify_variant_verdict(meets_effective=False, exceeds_ineffective_threshold=False) == "indeterminate"


def test_variant_verdict_is_always_one_of_the_three_literal_strings():
    for meets_effective in (True, False):
        for exceeds in (True, False):
            verdict = agg.classify_variant_verdict(
                meets_effective=meets_effective, exceeds_ineffective_threshold=exceeds
            )
            assert verdict in ("effective", "ineffective", "indeterminate")
            assert not isinstance(verdict, bool)
            assert isinstance(verdict, str)


# ----------------------------------------------------------------------------------------
# deployment_cost_profile: regression-pinned against the independently re-verified charter
# section 2.0 table (N=64, T=100, M=30). Real computation, no mocking -- CPU-only.
# ----------------------------------------------------------------------------------------
def test_deployment_cost_profile_b3_matches_charter_table():
    profile = agg.deployment_cost_profile("B3")
    assert profile["parameter_count"] == 18034
    assert profile["mac_per_session"] == 13017088
    assert profile["support_state_bytes"] == 16384
    assert profile["variant"] == "B3"


def test_deployment_cost_profile_b3t_matches_charter_table():
    profile = agg.deployment_cost_profile("B3T")
    assert profile["parameter_count"] == 12402
    assert profile["mac_per_session"] == 4507648
    assert profile["support_state_bytes"] == 16384
    assert profile["variant"] == "B3T"


def test_deployment_cost_profile_b3a_matches_charter_table():
    profile = agg.deployment_cost_profile("B3A")
    assert profile["parameter_count"] == 18099
    assert profile["mac_per_session"] == 13139968
    assert profile["support_state_bytes"] == 491520
    assert profile["variant"] == "B3A"


def test_deployment_cost_profile_b3a_support_state_is_30x_b3():
    b3 = agg.deployment_cost_profile("B3")
    b3a = agg.deployment_cost_profile("B3A")
    assert b3a["support_state_bytes"] / b3["support_state_bytes"] == pytest.approx(30.0)


def test_deployment_cost_profile_b3t_cheaper_than_b3_on_params_and_mac():
    b3 = agg.deployment_cost_profile("B3")
    b3t = agg.deployment_cost_profile("B3T")
    assert b3t["parameter_count"] < b3["parameter_count"]
    assert b3t["mac_per_session"] < b3["mac_per_session"]
    assert b3t["support_state_bytes"] == b3["support_state_bytes"]  # unchanged


def test_deployment_cost_profile_rejects_unknown_variant():
    with pytest.raises(ValueError, match="No cost profile"):
        agg.deployment_cost_profile("B15")


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


def test_parse_seeds_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        agg.parse_seeds("42,42")


def test_sigma_delta_standard_error_is_imported_from_v2_unmodified():
    assert agg.sigma_delta_standard_error is agg_v2.sigma_delta_standard_error


def test_sigma_delta_paired_is_imported_from_v2_unmodified():
    assert agg.sigma_delta_paired is agg_v2.sigma_delta_paired


def test_implied_seed_correlation_is_imported_from_v2_unmodified():
    assert agg.implied_seed_correlation is agg_v2.implied_seed_correlation


# ----------------------------------------------------------------------------------------
# Per-run helpers, using a NON-frozen epoch window to prove no hardcoded window.
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
    payload = _toy_payload({6: {"sA": 0.10}, 7: {"sA": 0.20}, 8: {"sA": 0.30}, 9: {"sA": 0.40}})
    scores = agg.per_run_session_scores(payload)
    assert scores["sA"] == pytest.approx((0.10 + 0.20 + 0.30 + 0.40) / 4)


def test_per_run_within_window_std_uses_the_artifacts_own_epoch_list():
    payload = _toy_payload({epoch: {"sA": 0.1 * (epoch - 5)} for epoch in range(6, 16)})
    std = agg.per_run_within_window_std(payload)
    import statistics

    assert std == pytest.approx(statistics.stdev([0.1 * i for i in range(1, 11)]))


# ----------------------------------------------------------------------------------------
# Full synthetic pipeline through run_aggregation(). Uses a 15-epoch/burn-in-5 window
# (neither the frozen script's 12/5-12 nor E3 test fixture's 20/11-20), so a regression to a
# hardcoded epoch window would be caught.
# ----------------------------------------------------------------------------------------
SEEDS = [42, 43, 44]
TOTAL_EPOCHS = 15
BURN_IN = 5
EPOCH_WINDOW = list(range(BURN_IN + 1, TOTAL_EPOCHS + 1))  # 6..15, 10 epochs
assert EPOCH_WINDOW == list(range(6, 16))
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


def _epoch_values_with_exact_mean(target_mean: float, n: int) -> list[float]:
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
        checkpoints_dir / f"e4_encoder_variants_{group.lower()}_dandi688_co_s{seed}"
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
            "side_dim": 0,
            "permutation_seed": None,
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


# Per-seed target means hand-chosen so that, in ONE consistent dataset (B3 shared control):
#   B3T -> effective    (B3T-B3 mean_delta=+0.05 for every seed)
#   B3A -> ineffective  (B3A-B3 mean_delta=-0.08 for every seed, far outside sigma_delta)
TARGET_MEANS = {
    ("B3", 42): 0.40, ("B3", 43): 0.42, ("B3", 44): 0.41,
    ("B3T", 42): 0.45, ("B3T", 43): 0.47, ("B3T", 44): 0.46,
    ("B3A", 42): 0.32, ("B3A", 43): 0.34, ("B3A", 44): 0.33,
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


def test_full_pipeline_produces_effective_b3t_and_ineffective_b3a(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)

    payload = agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)

    assert payload["seeds"] == SEEDS
    assert payload["epoch_window"] == EPOCH_WINDOW
    assert payload["epoch_budget"] == TOTAL_EPOCHS
    assert payload["burn_in_epochs"] == BURN_IN
    assert payload["effective_mean_delta_threshold"] == EFFECTIVE_MEAN_DELTA

    assert payload["variant_verdicts"]["B3T"]["verdict"] == "effective"
    assert payload["variant_verdicts"]["B3A"]["verdict"] == "ineffective"

    b3t_pair = payload["paired_deltas"]["B3T_minus_B3"]
    assert b3t_pair["mean_delta"] == pytest.approx(0.05)
    assert b3t_pair["n_sessions_positive"] == 6
    assert b3t_pair["all_seed_means_positive"] is True
    assert b3t_pair["meets_effective_clause"] is True

    b3a_pair = payload["paired_deltas"]["B3A_minus_B3"]
    assert b3a_pair["mean_delta"] == pytest.approx(-0.08)
    assert b3a_pair["meets_effective_clause"] is False
    assert b3a_pair["exceeds_ineffective_threshold"] is True

    # Never a third pair, never a shuffled control.
    assert set(payload["paired_deltas"]) == {"B3T_minus_B3", "B3A_minus_B3"}

    # The R2 verdict must never be read without the deployment cost beside it (charter
    # section 2.0): every variant_verdicts entry carries its own deployment_cost.
    for variant in ("B3T", "B3A"):
        cost = payload["variant_verdicts"][variant]["deployment_cost"]
        assert cost["support_state_bytes"] > 0
        assert cost["mac_per_session"] > 0
        assert cost["variant"] == variant

    b3a_cost = payload["variant_verdicts"]["B3A"]["deployment_cost"]
    assert b3a_cost["support_state_bytes"] == 491520
    b3a_ratio = payload["variant_verdicts"]["B3A"]["deployment_cost_vs_control"]
    assert b3a_ratio["support_state_bytes_ratio"] == pytest.approx(30.0)

    b3t_cost = payload["variant_verdicts"]["B3T"]["deployment_cost"]
    assert b3t_cost["mac_per_session"] == 4507648
    b3t_ratio = payload["variant_verdicts"]["B3T"]["deployment_cost_vs_control"]
    assert b3t_ratio["mac_per_session_ratio"] < 1.0  # B3T is cheaper in MAC than B3

    # Top-level deployment_cost table covers all three groups, including the control.
    assert set(payload["deployment_cost"]) == {"B3", "B3T", "B3A"}
    assert payload["deployment_cost"]["B3"]["parameter_count"] == 18034

    for group_payload in payload["variant_verdicts"].values():
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


# ----------------------------------------------------------------------------------------
# The whole point of the fix, wired end to end through run_aggregation (not just the pure
# helper functions): a synthetic B3T-B3 screen where the paired and unpaired estimators yield
# DIFFERENT three-state verdicts. Mirrors the real E4 finding (CURRENT_RESULTS.md section
# J.3c): mean_delta is small relative to each arm's OWN across-seed spread (quadrature can't
# resolve it -- stays "indeterminate"), but the per-seed delta is tight because both arms move
# together seed by seed (paired confidently resolves it as sub-threshold -- "ineffective").
# Numbers verified by hand: mean_delta=+0.02, 2*sigma_unpaired~=0.165 (>> delta, indeterminate
# under the old code), 2*sigma_paired~=0.0058 (<< delta, ineffective under the fixed code).
# ----------------------------------------------------------------------------------------
CORRELATED_SEEDS_TARGET_MEANS = {
    ("B3", 42): 0.40, ("B3", 43): 0.20, ("B3", 44): 0.30,  # across-seed std = 0.10: noisy arm
    ("B3T", 42): 0.425, ("B3T", 43): 0.220, ("B3T", 44): 0.315,  # per-seed deltas +.025/+.020/+.015
    ("B3A", 42): 0.40, ("B3A", 43): 0.20, ("B3A", 44): 0.30,  # identical to B3; not exercised here
}


def test_paired_sigma_resolves_an_effect_quadrature_leaves_indeterminate(tmp_path):
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints_dir.mkdir()
    for group in agg.GROUPS:
        for seed in SEEDS:
            _write_artifact(
                results_dir, checkpoints_dir, group, seed, CORRELATED_SEEDS_TARGET_MEANS[(group, seed)]
            )

    payload = agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)
    b3t_pair = payload["paired_deltas"]["B3T_minus_B3"]

    assert b3t_pair["mean_delta"] == pytest.approx(0.02)
    # Not effective: mean_delta (0.02) is well below the +0.03 bar -- unaffected by sigma.
    assert b3t_pair["meets_effective_clause"] is False

    # The unpaired (quadrature) estimate is wide because it ignores the shared-seed
    # correlation: 2*sigma is much bigger than the delta, so it CANNOT resolve this as
    # different from zero.
    assert 2 * b3t_pair["sigma_delta_unpaired_quadrature"] > abs(b3t_pair["mean_delta"])

    # The paired estimate is tight because the seed-level noise cancels in the paired
    # difference: 2*sigma IS smaller than the delta, so it confidently resolves this as a
    # real, nonzero (but sub-threshold) effect.
    assert 2 * b3t_pair["sigma_delta_paired"] < abs(b3t_pair["mean_delta"])
    assert b3t_pair["sigma_delta_paired"] < b3t_pair["sigma_delta_unpaired_quadrature"]
    assert b3t_pair["implied_seed_correlation"] > 0.9  # strongly positively correlated seeds

    # Recomputing what the OLD (pre-fix) code would have verdicted, by hand, from the same
    # recorded quadrature sigma: not effective, and not confidently resolved sub-threshold
    # either (2*sigma_unpaired > mean_delta) -> "indeterminate".
    old_exceeds_ineffective = agg.pair_exceeds_ineffective_threshold(
        mean_delta=b3t_pair["mean_delta"], sigma_delta=b3t_pair["sigma_delta_unpaired_quadrature"]
    )
    assert old_exceeds_ineffective is False
    old_verdict = agg.classify_variant_verdict(
        meets_effective=b3t_pair["meets_effective_clause"],
        exceeds_ineffective_threshold=old_exceeds_ineffective,
    )
    assert old_verdict == "indeterminate"

    # The actual (fixed) verdict this aggregator now produces, end to end, is "ineffective"
    # -- a DIFFERENT verdict from the exact same underlying data, purely because of which
    # sigma_delta feeds the gate. This is the whole point of the fix.
    assert payload["variant_verdicts"]["B3T"]["verdict"] == "ineffective"
    assert old_verdict != payload["variant_verdicts"]["B3T"]["verdict"]


def test_full_pipeline_threshold_is_a_live_cli_argument_not_hardcoded(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)
    low = agg.run_aggregation(results_dir, SEEDS, 0.03)
    high = agg.run_aggregation(results_dir, SEEDS, 0.06)
    assert low["variant_verdicts"]["B3T"]["verdict"] == "effective"
    assert high["variant_verdicts"]["B3T"]["verdict"] != "effective"


def test_full_pipeline_seeds_argument_selects_which_runs_are_aggregated(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)
    subset = [42, 43]
    payload = agg.run_aggregation(results_dir, subset, EFFECTIVE_MEAN_DELTA)
    assert payload["seeds"] == subset
    expected_mean = agg.mean([TARGET_MEANS[("B3", s)] for s in subset])
    assert payload["variant_scores"]["B3"]["mean"] == pytest.approx(expected_mean)


def test_requesting_a_seed_not_on_disk_raises_file_not_found(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)
    with pytest.raises(FileNotFoundError, match="b3_s99"):
        agg.run_aggregation(results_dir, [42, 43, 44, 99], EFFECTIVE_MEAN_DELTA)


def test_run_aggregation_rejects_non_positive_effective_mean_delta(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)
    with pytest.raises(ValueError, match="effective_mean_delta"):
        agg.run_aggregation(results_dir, SEEDS, 0.0)


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
            if (group, seed) == ("B3A", 44):
                continue  # simulate a still-running / crashed run
            _write_artifact(results_dir, checkpoints_dir, group, seed, TARGET_MEANS[(group, seed)])

    with pytest.raises(FileNotFoundError, match="b3a_s44"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_mismatched_session_splits(tmp_path):
    bad_splits = dict(SESSION_SPLITS)
    bad_splits["val"] = list(VAL_SESSIONS[:-1]) + ["sub-X_ses-CO-99999999"]
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path, overrides={("B3A", 44): {"session_splits": bad_splits}}
    )
    with pytest.raises(ValueError, match="session_splits disagree"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_mismatched_epoch_window_across_artifacts(tmp_path):
    other_protocol = dict(PROTOCOL, burn_in_epochs=3, epoch_window=list(range(4, 16)))
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path, overrides={("B3A", 44): {"protocol": other_protocol}}
    )
    with pytest.raises(ValueError, match="protocol .*disagrees"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_internally_inconsistent_epoch_window(tmp_path):
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints_dir.mkdir()
    _write_artifact(results_dir, checkpoints_dir, "B3", 42, TARGET_MEANS[("B3", 42)])
    path = agg.artifact_path(results_dir, "B3", 42)
    payload = json.loads(path.read_text())
    payload["protocol"]["epoch_window"] = [1, 2, 3]
    path.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ValueError, match="is not burn_in_epochs"):
        agg.run_aggregation(results_dir, [42], EFFECTIVE_MEAN_DELTA)


def test_rejects_two_runs_sharing_a_run_directory(tmp_path):
    shared_dir = tmp_path / "checkpoints" / "shared_run_dir"
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={
            ("B3T", 43): {"run_dir_override": shared_dir},
            ("B3T", 44): {"run_dir_override": shared_dir},
        },
    )
    with pytest.raises(ValueError, match="share a run directory"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_wrong_max_epochs(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path, overrides={("B3", 42): {"run_metadata_overrides": {"training": {"max_epochs": 999}}}}
    )
    with pytest.raises(ValueError, match="max_epochs"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_early_stopping_enabled(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={("B3A", 43): {"run_metadata_overrides": {"training": {"no_early_stopping": False}}}},
    )
    with pytest.raises(ValueError, match="no_early_stopping"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_missing_checkpoint_every_epoch(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={("B3A", 43): {"run_metadata_overrides": {"training": {"checkpoint_every_epoch": False}}}},
    )
    with pytest.raises(ValueError, match="checkpoint_every_epoch"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_side_features_group_contract_mismatch(tmp_path):
    # E4 requires side_features.group == "none" for every group -- a run that accidentally
    # used a real side-feature group must be rejected, not silently aggregated.
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={("B3T", 42): {"run_metadata_overrides": {"side_features": {"group": "t4"}}}},
    )
    with pytest.raises(ValueError, match="side_features.group"):
        agg.run_aggregation(results_dir, SEEDS, EFFECTIVE_MEAN_DELTA)


def test_rejects_tampered_run_metadata_after_evaluation(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(tmp_path)
    tampered_run_dir = checkpoints_dir / "e4_encoder_variants_b3_dandi688_co_s42"
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
    _write_artifact(results_dir, checkpoints_dir, "B3T", 42, TARGET_MEANS[("B3T", 42)])
    path = agg.artifact_path(results_dir, "B3T", 42)
    payload = json.loads(path.read_text())
    payload["variant"] = "B3A"
    path.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ValueError, match="variant mismatch"):
        agg.run_aggregation(results_dir, [42], EFFECTIVE_MEAN_DELTA)
