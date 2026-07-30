"""Tests for aggregate_side_feature_ablation_v2.py.

Focus is the four-state verdict logic (MEASUREMENT_PROTOCOL_V4.md section 4.2b, 2026-07-27
bug fix): ``pair_meets_effective_clause``, ``pair_meets_effective_heterogeneous_clause``,
``pair_exceeds_ineffective_threshold``, ``classify_pair_verdict`` (the per-pair four-state
verdict), and ``classify_group_verdict`` (the per-feature-group combination of a group's
dimension-matched pairs) -- covering all four literal outcomes (effective /
effective_heterogeneous / ineffective / indeterminate), the boundaries between them, the
AND/AND/OR asymmetry at group level (effective and effective_heterogeneous both require ALL
pairs; ineffective requires only ANY pair -- a 2026-07-27 change from the old all-pairs-AND
rule, which was over-conservative), and a regression pin of the real B3T-B3 measurement that
exposed the bug this revision fixes: mean +0.0324, paired 2*sigma 0.0203, all 3 seeds
positive, only 3/6 sessions positive -- confidently positive but session-heterogeneous, which
the pre-fix code mislabelled "ineffective" (reading "confidently nonzero" as bad for a
positive effect). Conflating "indeterminate" with "ineffective" is exactly the bug that voided
attention_arch_screen_v3 (sua_exploration/docs/CURRENT_RESULTS.md section H) and is called
out by name in UNIT_SIDE_FEATURE_ABLATION.md section 8.

A smaller set of tests covers the section-2/charter-section-8 consistency gate (unique run
dirs, agreeing session splits, matched training provenance including the side_features.group
contract), and one full synthetic 15-artifact pipeline exercising ``run_aggregation`` end to
end, hand-designed so F1 comes out "effective" and F2 comes out "ineffective" in the same
consistent dataset (F0 is shared between both groups' comparisons).

No GPU, no NWB data, no torch: the aggregator only reads JSON, so these tests only need
pytest and the standard library.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_side_feature_ablation_v2 as agg  # noqa: E402


# ----------------------------------------------------------------------------------------
# pair_meets_effective_clause: UNCHANGED by the 2026-07-27 fix (mean_delta >= threshold AND
# n_sessions_positive >= effective_min_positive_sessions AND all per-seed means positive).
# Signature now takes effective_mean_delta_threshold / effective_min_positive_sessions as
# explicit arguments (shared-function generalization) rather than reading v2's own module
# constants directly.
# ----------------------------------------------------------------------------------------
def test_pair_meets_effective_clause_true_case():
    assert agg.pair_meets_effective_clause(
        mean_delta=0.05, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=[0.04, 0.05, 0.06],
        effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
    ) is True


def test_pair_meets_effective_clause_inclusive_boundaries():
    # mean_delta exactly +0.03 and exactly 5/6 positive: both use >=.
    assert agg.pair_meets_effective_clause(
        mean_delta=0.03, n_sessions_positive=5, n_sessions_total=6,
        per_seed_means=[0.001, 0.002, 0.003],
        effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
    ) is True


def test_pair_meets_effective_clause_false_when_mean_delta_just_under():
    assert agg.pair_meets_effective_clause(
        mean_delta=0.0299999, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=[0.03, 0.03, 0.03],
        effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
    ) is False


def test_pair_meets_effective_clause_false_when_four_of_six_positive():
    assert agg.pair_meets_effective_clause(
        mean_delta=0.04, n_sessions_positive=4, n_sessions_total=6,
        per_seed_means=[0.03, 0.04, 0.05],
        effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
    ) is False


def test_pair_meets_effective_clause_false_when_one_seed_mean_not_positive():
    assert agg.pair_meets_effective_clause(
        mean_delta=0.05, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=[0.10, 0.10, 0.0],  # exactly zero is not positive
        effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
    ) is False


def test_pair_meets_effective_clause_rejects_wrong_session_total():
    with pytest.raises(ValueError, match="n_sessions_total"):
        agg.pair_meets_effective_clause(
            mean_delta=0.05, n_sessions_positive=5, n_sessions_total=0,
            per_seed_means=[0.1, 0.1, 0.1],
            effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
        )


def test_pair_meets_effective_clause_rejects_positive_count_out_of_range():
    with pytest.raises(ValueError, match="n_sessions_positive"):
        agg.pair_meets_effective_clause(
            mean_delta=0.05, n_sessions_positive=7, n_sessions_total=6,
            per_seed_means=[0.1, 0.1, 0.1],
            effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
        )


def test_pair_meets_effective_clause_rejects_empty_seed_means():
    with pytest.raises(ValueError, match="per_seed_means"):
        agg.pair_meets_effective_clause(
            mean_delta=0.05, n_sessions_positive=5, n_sessions_total=6,
            per_seed_means=[],
            effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
        )


# ----------------------------------------------------------------------------------------
# pair_meets_effective_heterogeneous_clause: the NEW 2026-07-27 clause (section 4.2b).
# Confidently positive (mean - 2*sigma > 0) AND all seeds positive AND the effective clause's
# session-consistency count specifically fails (n_sessions_positive < min) -- this is what a
# real, seed-replicated, session-heterogeneous effect looks like.
# ----------------------------------------------------------------------------------------
def test_effective_heterogeneous_clause_true_case_mirrors_real_b3t_minus_b3():
    # Real numbers (CURRENT_RESULTS.md section J.3c / e4_encoder_variants aggregate.json):
    # mean +0.0324, sigma_delta_paired ~0.0101 (2*sigma ~0.0203), 3/6 sessions positive.
    assert agg.pair_meets_effective_heterogeneous_clause(
        mean_delta=0.0324, sigma_delta_paired=0.0101,
        per_seed_means=[0.0516, 0.0284, 0.0171], n_sessions_positive=3,
        effective_min_positive_sessions=5,
    ) is True


def test_effective_heterogeneous_clause_false_when_not_confidently_positive():
    # mean - 2*sigma <= 0: cannot rule out a zero-or-negative effect.
    assert agg.pair_meets_effective_heterogeneous_clause(
        mean_delta=0.0324, sigma_delta_paired=0.02,  # 2*sigma = 0.04 > mean
        per_seed_means=[0.0516, 0.0284, 0.0171], n_sessions_positive=3,
        effective_min_positive_sessions=5,
    ) is False


def test_effective_heterogeneous_clause_false_when_a_seed_mean_is_not_positive():
    assert agg.pair_meets_effective_heterogeneous_clause(
        mean_delta=0.0324, sigma_delta_paired=0.0101,
        per_seed_means=[0.0516, 0.0284, -0.001], n_sessions_positive=3,
        effective_min_positive_sessions=5,
    ) is False


def test_effective_heterogeneous_clause_false_when_session_consistency_actually_holds():
    # If sessions ARE consistent (>= min), the effective clause's own session-consistency
    # sub-condition did not fail -- this is not the "session-heterogeneous" case at all (it
    # would be caught by pair_meets_effective_clause / classify_pair_verdict's first branch
    # if mean_delta also clears the threshold, or otherwise stay indeterminate/ineffective).
    assert agg.pair_meets_effective_heterogeneous_clause(
        mean_delta=0.0324, sigma_delta_paired=0.0101,
        per_seed_means=[0.0516, 0.0284, 0.0171], n_sessions_positive=5,
        effective_min_positive_sessions=5,
    ) is False


def test_effective_heterogeneous_clause_boundary_is_strict_greater_than():
    # mean - 2*sigma exactly == 0 must NOT count as confidently positive (needs >, not >=).
    assert agg.pair_meets_effective_heterogeneous_clause(
        mean_delta=0.02, sigma_delta_paired=0.01,  # mean - 2*sigma == 0.0 exactly
        per_seed_means=[0.02, 0.02, 0.02], n_sessions_positive=3,
        effective_min_positive_sessions=5,
    ) is False
    assert agg.pair_meets_effective_heterogeneous_clause(
        mean_delta=0.0200001, sigma_delta_paired=0.01,
        per_seed_means=[0.02, 0.02, 0.02], n_sessions_positive=3,
        effective_min_positive_sessions=5,
    ) is True


def test_effective_heterogeneous_clause_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma_delta_paired"):
        agg.pair_meets_effective_heterogeneous_clause(
            mean_delta=0.05, sigma_delta_paired=-0.01,
            per_seed_means=[0.1, 0.1, 0.1], n_sessions_positive=3,
            effective_min_positive_sessions=5,
        )


def test_effective_heterogeneous_clause_rejects_empty_seed_means():
    with pytest.raises(ValueError, match="per_seed_means"):
        agg.pair_meets_effective_heterogeneous_clause(
            mean_delta=0.05, sigma_delta_paired=0.01,
            per_seed_means=[], n_sessions_positive=3,
            effective_min_positive_sessions=5,
        )


# ----------------------------------------------------------------------------------------
# pair_exceeds_ineffective_threshold: THE 2026-07-27 BUG FIX. Old (wrong) clause was
# abs(mean_delta) > 2*sigma_delta ("confidently nonzero" -- backwards for a positive effect).
# New clause: mean_delta + 2*sigma_delta_paired < effective_mean_delta_threshold
# ("confidently excludes an effect of at least the threshold").
# ----------------------------------------------------------------------------------------
def test_pair_exceeds_ineffective_threshold_true_case():
    assert agg.pair_exceeds_ineffective_threshold(
        mean_delta=-0.05, sigma_delta_paired=0.01, effective_mean_delta_threshold=0.03,
    ) is True


def test_pair_exceeds_ineffective_threshold_boundary_is_strict():
    # mean_delta + 2*sigma exactly == threshold must NOT exceed (needs <, not <=).
    assert agg.pair_exceeds_ineffective_threshold(
        mean_delta=0.01, sigma_delta_paired=0.01, effective_mean_delta_threshold=0.03,
    ) is False  # 0.01 + 0.02 == 0.03
    assert agg.pair_exceeds_ineffective_threshold(
        mean_delta=0.0099999, sigma_delta_paired=0.01, effective_mean_delta_threshold=0.03,
    ) is True


def test_pair_exceeds_ineffective_threshold_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma_delta_paired"):
        agg.pair_exceeds_ineffective_threshold(
            mean_delta=0.05, sigma_delta_paired=-0.01, effective_mean_delta_threshold=0.03,
        )


def test_pair_exceeds_ineffective_threshold_fixes_the_backwards_old_clause():
    # THE bug this revision fixes: a confidently POSITIVE effect (B3T-B3-shaped: mean well
    # above zero, tight sigma) must NOT be "ineffective" merely because abs(mean) > 2*sigma.
    # Old (removed) clause would have said True here; the fixed clause correctly says False,
    # because mean+2sigma is nowhere near excluding the +0.03 threshold.
    mean_delta, sigma_delta_paired = 0.0324, 0.0101
    old_wrong_clause_result = abs(mean_delta) > 2.0 * sigma_delta_paired
    assert old_wrong_clause_result is True  # this is what used to (wrongly) fire
    assert agg.pair_exceeds_ineffective_threshold(
        mean_delta=mean_delta, sigma_delta_paired=sigma_delta_paired,
        effective_mean_delta_threshold=0.03,
    ) is False  # the fixed clause correctly does NOT call this ineffective


def test_pair_exceeds_ineffective_threshold_true_for_a_clearly_negative_mean_even_with_small_abs_margin():
    # A negative mean that is NOT "confidently different from zero" under the old clause
    # (abs(mean) barely below 2*sigma) can still be confidently EXCLUDED from reaching the
    # positive threshold under the new clause -- the old clause's blind spot for negative
    # effects. mean=-0.0324, sigma=0.0164: abs(mean)=0.0324 < 2*sigma=0.0329 (old clause says
    # NOT confidently nonzero), but mean+2*sigma=0.00046 is still far below threshold=0.03.
    mean_delta, sigma_delta_paired = -0.0324, 0.0164
    old_wrong_clause_result = abs(mean_delta) > 2.0 * sigma_delta_paired
    assert old_wrong_clause_result is False  # old clause would have said "not resolved"
    assert agg.pair_exceeds_ineffective_threshold(
        mean_delta=mean_delta, sigma_delta_paired=sigma_delta_paired,
        effective_mean_delta_threshold=0.03,
    ) is True  # new clause correctly excludes a >=+0.03 effect


# ----------------------------------------------------------------------------------------
# classify_pair_verdict: the four-state cascade (effective -> effective_heterogeneous ->
# ineffective -> indeterminate) built from the three predicates above, plus a decided_by
# string. THE single most load-bearing function in this module.
# ----------------------------------------------------------------------------------------
def _classify(**kwargs):
    kwargs.setdefault("effective_mean_delta_threshold", 0.03)
    kwargs.setdefault("effective_min_positive_sessions", 5)
    return agg.classify_pair_verdict(**kwargs)


def test_classify_effective_when_all_three_clauses_hold():
    verdict, decided_by = _classify(
        mean_delta=0.05, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=[0.04, 0.05, 0.06], sigma_delta_paired=0.01,
    )
    assert verdict == "effective"
    assert decided_by.startswith("effective:")
    assert "0.03" in decided_by or "0.0300" in decided_by


def test_classify_effective_heterogeneous_pins_real_b3t_minus_b3():
    # THE regression case this fix exists for (task requirement): real measured B3T-B3
    # (CURRENT_RESULTS.md section J.3c): mean +0.0324, per-seed +0.0516/+0.0284/+0.0171,
    # only 3/6 sessions positive. The pre-2026-07-27 code labelled this "ineffective"
    # (abs(0.0324) > 2*0.0101); the fix must resolve it to "effective_heterogeneous".
    deltas = [0.0516, 0.0284, 0.0171]
    mean_delta = agg.mean(deltas)
    sigma_paired = agg.sigma_delta_paired(deltas)
    assert mean_delta == pytest.approx(0.0324, abs=1e-4)
    assert sigma_paired == pytest.approx(0.0102, abs=1e-3)

    verdict, decided_by = _classify(
        mean_delta=mean_delta, n_sessions_positive=3, n_sessions_total=6,
        per_seed_means=deltas, sigma_delta_paired=sigma_paired,
    )
    assert verdict == "effective_heterogeneous"
    assert verdict != "ineffective"  # the bug this revision fixes
    assert decided_by.startswith("effective_heterogeneous:")
    assert "3/6" in decided_by


def test_classify_effective_heterogeneous_false_when_a_seed_is_negative():
    # Same mean/sigma/session-count shape as the B3T-B3 case, but one seed is negative:
    # must NOT be effective_heterogeneous (falls through toward ineffective/indeterminate).
    verdict, _ = _classify(
        mean_delta=0.0324, n_sessions_positive=3, n_sessions_total=6,
        per_seed_means=[0.0516, 0.0284, -0.05], sigma_delta_paired=0.0101,
    )
    assert verdict != "effective_heterogeneous"
    assert verdict != "effective"


def test_classify_ineffective_when_confidently_below_threshold():
    verdict, decided_by = _classify(
        mean_delta=-0.05, n_sessions_positive=0, n_sessions_total=6,
        per_seed_means=[-0.04, -0.05, -0.06], sigma_delta_paired=0.01,
    )
    assert verdict == "ineffective"
    assert decided_by.startswith("ineffective:")


def test_classify_ineffective_for_a_negative_mean_with_all_seeds_negative():
    # Real F1-F0 shape (side_feature_ablation_v2 aggregate.json): mean -0.0324, all seeds
    # negative, sigma_delta_paired 0.0164. Old clause: abs(-0.0324)=0.0324 vs
    # 2*0.0164=0.0329 -- NOT > threshold, so old code said "not ineffective" (a negative
    # effect the old clause failed to resolve). New clause: mean+2sigma=0.0004 < 0.03 --
    # confidently excludes a +0.03 effect -- ineffective.
    verdict, decided_by = _classify(
        mean_delta=-0.0324, n_sessions_positive=1, n_sessions_total=6,
        per_seed_means=[-0.0135, -0.0186, -0.0651], sigma_delta_paired=0.0164,
    )
    assert verdict == "ineffective"
    assert decided_by.startswith("ineffective:")


def test_classify_ineffective_via_f2_minus_fs2_real_numbers():
    # Real F2-FS2 pair (side_feature_ablation_v2 aggregate.json): mean -0.0367, sigma_delta_
    # paired 0.0265 -> mean+2sigma = +0.0163 < threshold 0.03. This is the exact pair the
    # task's decided_by example ("mean+2sigma=0.0163 < threshold=0.03") is drawn from.
    verdict, decided_by = _classify(
        mean_delta=-0.03665076496286525, n_sessions_positive=1, n_sessions_total=6,
        per_seed_means=[-0.08738490706309676, -0.024417744483798742, 0.001850356658299764],
        sigma_delta_paired=0.02647620419287064,
    )
    assert verdict == "ineffective"
    assert "0.0163" in decided_by


def test_classify_indeterminate_when_delta_small_relative_to_noise():
    # sigma must be large enough that BOTH mean-2sigma<=0 (not effective_heterogeneous) AND
    # mean+2sigma>=threshold (not ineffective) -- at sigma=0.01 (2*sigma=0.02) the upper bound
    # 0.005+0.02=0.025 is already < threshold=0.03, which (correctly, under the fixed clause)
    # resolves to "ineffective", not "indeterminate" -- exactly the behavior change this fix
    # makes: many small-but-not-quite-zero deltas that used to be "indeterminate" under the
    # old abs(mean)>2*sigma test now correctly resolve as confidently sub-threshold.
    verdict, decided_by = _classify(
        mean_delta=0.005, n_sessions_positive=4, n_sessions_total=6,
        per_seed_means=[0.002, 0.005, 0.008], sigma_delta_paired=0.02,
    )
    assert verdict == "indeterminate"
    assert decided_by.startswith("indeterminate:")


def test_classify_ineffective_not_indeterminate_for_small_delta_with_moderate_sigma():
    # The behavior-change case called out above, pinned explicitly: mean=0.005, sigma=0.01 --
    # small and not "confidently nonzero" (old clause's test), but mean+2sigma=0.025 is
    # confidently below threshold=0.03, so the fixed clause correctly resolves "ineffective".
    verdict, decided_by = _classify(
        mean_delta=0.005, n_sessions_positive=4, n_sessions_total=6,
        per_seed_means=[0.002, 0.005, 0.008], sigma_delta_paired=0.01,
    )
    assert verdict == "ineffective"
    old_wrong_clause_result = abs(0.005) > 2 * 0.01
    assert old_wrong_clause_result is False  # old clause would have left this "indeterminate"


def test_classify_indeterminate_via_b3a_minus_b3_real_numbers():
    # Real B3A-B3 pair (e4_encoder_variants aggregate.json): mean +0.0007 (~0), mixed-sign
    # seeds, sigma_delta_paired 0.0164 -> mean+2sigma=0.0335, just ABOVE threshold 0.03, so
    # the ineffective clause narrowly does NOT fire either -- stays indeterminate.
    verdict, _ = _classify(
        mean_delta=0.0007018150709983368, n_sessions_positive=3, n_sessions_total=6,
        per_seed_means=[0.0331963092709581, -0.011604847619310021, -0.01948601643865307],
        sigma_delta_paired=0.01640576443889119,
    )
    assert verdict == "indeterminate"


def test_classify_effective_heterogeneous_takes_priority_over_ineffective_when_both_clauses_hold():
    # Constructed edge case: sigma small enough relative to the threshold that a mean can be
    # simultaneously "confidently positive" (mean-2sigma>0) AND "confidently below threshold"
    # (mean+2sigma<threshold). mean=0.015, sigma=0.003: mean-2sigma=0.009>0;
    # mean+2sigma=0.021<0.03=threshold. Cascade order (matching section 4.2b's table) means
    # effective_heterogeneous wins.
    mean_delta, sigma = 0.015, 0.003
    assert (mean_delta - 2 * sigma) > 0.0
    assert (mean_delta + 2 * sigma) < 0.03
    verdict, decided_by = _classify(
        mean_delta=mean_delta, n_sessions_positive=2, n_sessions_total=6,
        per_seed_means=[0.014, 0.015, 0.016], sigma_delta_paired=sigma,
    )
    assert verdict == "effective_heterogeneous"
    assert decided_by.startswith("effective_heterogeneous:")


def test_classify_verdict_is_always_one_of_the_four_literal_strings():
    for mean_delta in (-0.2, -0.03, -0.005, 0.0, 0.005, 0.03, 0.2):
        for n_positive in (0, 2, 4, 5, 6):
            for sigma in (0.0, 0.001, 0.01, 0.05):
                for seed_means in ([mean_delta] * 3, [-1.0, mean_delta, 1.0]):
                    verdict, decided_by = _classify(
                        mean_delta=mean_delta, n_sessions_positive=n_positive,
                        n_sessions_total=6, per_seed_means=seed_means,
                        sigma_delta_paired=sigma,
                    )
                    assert verdict in agg.VALID_VERDICTS
                    assert not isinstance(verdict, bool)
                    assert isinstance(verdict, str)
                    assert isinstance(decided_by, str) and decided_by.startswith(verdict + ":")


def test_classify_effective_and_ineffective_are_mutually_exclusive():
    # By construction (sigma >= 0): mean_delta >= threshold implies mean_delta + 2*sigma >=
    # threshold, so a pair can never be both.
    import random

    rng = random.Random(0)
    for _ in range(200):
        mean_delta = rng.uniform(-0.2, 0.2)
        sigma = rng.uniform(0.0, 0.1)
        n_positive = rng.randint(0, 6)
        seed_means = [rng.uniform(-0.2, 0.2) for _ in range(3)]
        verdict, _ = _classify(
            mean_delta=mean_delta, n_sessions_positive=n_positive, n_sessions_total=6,
            per_seed_means=seed_means, sigma_delta_paired=sigma,
        )
        # Trivially true since classify_pair_verdict returns one verdict, but assert the
        # underlying clauses directly to document the invariant explicitly.
        is_eff = agg.pair_meets_effective_clause(
            mean_delta=mean_delta, n_sessions_positive=n_positive, n_sessions_total=6,
            per_seed_means=seed_means, effective_mean_delta_threshold=0.03,
            effective_min_positive_sessions=5,
        )
        is_ineff = agg.pair_exceeds_ineffective_threshold(
            mean_delta=mean_delta, sigma_delta_paired=sigma, effective_mean_delta_threshold=0.03,
        )
        assert not (is_eff and is_ineff)


# ----------------------------------------------------------------------------------------
# classify_group_verdict: the group-level four-state combination of a group's pairs. THE
# AND/AND/OR asymmetry (task requirement): effective and effective_heterogeneous both
# require ALL pairs; ineffective requires only ANY pair -- a 2026-07-27 change from the old
# all-pairs-AND-ineffective rule, which was over-conservative.
# ----------------------------------------------------------------------------------------
def test_group_effective_when_all_pairs_effective():
    eff = ("effective", "effective: ...")
    verdict, decided_by = agg.classify_group_verdict(
        pair_results={"a": eff, "b": eff}
    )
    assert verdict == "effective"
    assert decided_by.startswith("effective:")


def test_group_not_effective_when_only_one_pair_effective():
    # F_x beats F0 but not its own shuffled control (or vice versa): the charter is
    # explicit ALL pairs must pass, so this must NOT be "effective".
    verdict, _ = agg.classify_group_verdict(
        pair_results={
            "a": ("effective", "effective: ..."),
            "b": ("indeterminate", "indeterminate: ..."),
        }
    )
    assert verdict != "effective"


def test_group_effective_heterogeneous_when_all_pairs_effective_or_heterogeneous_with_at_least_one_heterogeneous():
    verdict, decided_by = agg.classify_group_verdict(
        pair_results={
            "a": ("effective", "effective: ..."),
            "b": ("effective_heterogeneous", "effective_heterogeneous: ..."),
        }
    )
    assert verdict == "effective_heterogeneous"
    assert decided_by.startswith("effective_heterogeneous:")

    # Also true when BOTH pairs are only effective_heterogeneous.
    verdict_both, _ = agg.classify_group_verdict(
        pair_results={
            "a": ("effective_heterogeneous", "effective_heterogeneous: ..."),
            "b": ("effective_heterogeneous", "effective_heterogeneous: ..."),
        }
    )
    assert verdict_both == "effective_heterogeneous"


def test_group_effective_heterogeneous_not_triggered_if_a_pair_is_ineffective_or_indeterminate():
    verdict_ineff, _ = agg.classify_group_verdict(
        pair_results={
            "a": ("effective_heterogeneous", "effective_heterogeneous: ..."),
            "b": ("ineffective", "ineffective: ..."),
        }
    )
    assert verdict_ineff != "effective_heterogeneous"

    verdict_indet, _ = agg.classify_group_verdict(
        pair_results={
            "a": ("effective_heterogeneous", "effective_heterogeneous: ..."),
            "b": ("indeterminate", "indeterminate: ..."),
        }
    )
    assert verdict_indet != "effective_heterogeneous"


def test_group_ineffective_when_both_pairs_ineffective():
    verdict, decided_by = agg.classify_group_verdict(
        pair_results={
            "a": ("ineffective", "ineffective: mean+2sigma=-0.03 < threshold=0.03"),
            "b": ("ineffective", "ineffective: mean+2sigma=0.01 < threshold=0.03"),
        }
    )
    assert verdict == "ineffective"


def test_group_ineffective_when_only_one_pair_ineffective_the_and_or_asymmetry():
    # THE task requirement: this is the AND/OR asymmetry. Old (three-state, pre-2026-07-27)
    # rule required ALL pairs ineffective (AND) -- a single ineffective pair alongside an
    # indeterminate one used to stay "indeterminate". New rule: ANY pair ineffective is
    # sufficient (OR) -- mirrors the real F2 group (F2-F0 stays indeterminate, F2-FS2
    # resolves ineffective; the group must now be "ineffective").
    pair_results = {
        "a": ("indeterminate", "indeterminate: ..."),
        "b": ("ineffective", "ineffective: mean+2sigma=0.0163 < threshold=0.03"),
    }
    new_verdict, new_decided_by = agg.classify_group_verdict(pair_results=pair_results)
    assert new_verdict == "ineffective"
    assert "b" in new_decided_by

    # Explicitly recompute what the OLD (pre-2026-07-27) all-pairs-AND rule would have
    # produced from the same per-pair booleans, to document the asymmetry is a real behavior
    # change and not just a renamed state.
    pair_a_old_ineffective = False  # "indeterminate" -> old exceeds_ineffective_threshold=False
    pair_b_old_ineffective = True  # "ineffective" -> old exceeds_ineffective_threshold=True
    old_verdict = "ineffective" if (pair_a_old_ineffective and pair_b_old_ineffective) else "indeterminate"
    assert old_verdict == "indeterminate"
    assert old_verdict != new_verdict


def test_group_ineffective_swapped_pair_order_still_ineffective():
    verdict, _ = agg.classify_group_verdict(
        pair_results={
            "a": ("ineffective", "ineffective: ..."),
            "b": ("indeterminate", "indeterminate: ..."),
        }
    )
    assert verdict == "ineffective"


def test_group_indeterminate_when_neither_pair_resolved():
    verdict, decided_by = agg.classify_group_verdict(
        pair_results={
            "a": ("indeterminate", "indeterminate: ..."),
            "b": ("indeterminate", "indeterminate: ..."),
        }
    )
    assert verdict == "indeterminate"
    assert decided_by.startswith("indeterminate:")


def test_group_verdict_matches_real_f1_group_both_pairs_ineffective_under_new_rule():
    # Real F1 group (side_feature_ablation_v2 aggregate.json): BOTH F1-F0 and F1-FS1
    # independently resolve "ineffective" under the fixed pair-level clause (unlike under
    # the old clause, where F1-F0's abs(mean) fell just short of 2*sigma). Group verdict must
    # be "ineffective" whether combined via AND or OR, since both individually qualify.
    f1_f0 = _classify(
        mean_delta=-0.03242079419497814, n_sessions_positive=1, n_sessions_total=6,
        per_seed_means=[-0.013503504917025566, -0.018612787360325456, -0.0651460903075834],
        sigma_delta_paired=0.016428988059875815,
    )
    f1_fs1 = _classify(
        mean_delta=-0.045695917991300426, n_sessions_positive=0, n_sessions_total=6,
        per_seed_means=[-0.052940525424977146, -0.03375956059123079, -0.05038766795769334],
        sigma_delta_paired=0.006013505391996546,
    )
    assert f1_f0[0] == "ineffective"
    assert f1_fs1[0] == "ineffective"
    verdict, _ = agg.classify_group_verdict(
        pair_results={"F1_minus_F0": f1_f0, "F1_minus_FS1": f1_fs1}
    )
    assert verdict == "ineffective"


def test_group_verdict_rejects_empty_pair_results():
    with pytest.raises(ValueError, match="non-empty"):
        agg.classify_group_verdict(pair_results={})


def test_group_verdict_rejects_unknown_verdict_string():
    with pytest.raises(ValueError, match="Unknown verdict"):
        agg.classify_group_verdict(
            pair_results={"a": ("effective", "..."), "b": ("bogus", "...")}
        )


def test_group_verdict_is_always_one_of_the_four_literal_strings():
    states = ("effective", "effective_heterogeneous", "ineffective", "indeterminate")
    for a in states:
        for b in states:
            verdict, decided_by = agg.classify_group_verdict(
                pair_results={"a": (a, f"{a}: ..."), "b": (b, f"{b}: ...")}
            )
            assert verdict in agg.VALID_VERDICTS
            assert not isinstance(verdict, bool)
            assert isinstance(verdict, str)
            assert isinstance(decided_by, str) and len(decided_by) > 0


def test_group_verdict_single_pair_group_degenerates_to_that_pairs_own_verdict():
    # e4_encoder_variants has no shuffled control -- a "group" of 1 pair must just reproduce
    # that pair's own verdict (used as a cross-check; e4 itself calls classify_pair_verdict
    # directly rather than routing through classify_group_verdict, but the degenerate case
    # must still hold for classify_group_verdict to be a true generalization).
    for state in ("effective", "effective_heterogeneous", "ineffective", "indeterminate"):
        verdict, _ = agg.classify_group_verdict(pair_results={"only": (state, f"{state}: ...")})
        assert verdict == state


# ----------------------------------------------------------------------------------------
# sample_std
# ----------------------------------------------------------------------------------------
def test_sample_std_matches_statistics_stdev():
    import statistics

    values = [0.20, 0.22, 0.21]
    assert agg.sample_std(values) == pytest.approx(statistics.stdev(values))


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
# sigma_delta_paired (2026-07-26 revision, MEASUREMENT_PROTOCOL_V4.md section 4.1): the
# second sigma_delta defect. sigma_delta_standard_error combines the two arms' across-seed
# SDs in quadrature, which assumes the arms' seed effects are independent -- they are not,
# since both arms share the same seed list. sigma_delta_paired instead computes the SE
# directly from the same-seed-paired per-seed mean deltas. Worked example pinned below is
# E4's real B3T-B3 pair (CURRENT_RESULTS.md section J.3c).
# ----------------------------------------------------------------------------------------
def test_sigma_delta_paired_matches_hand_computed_value():
    import math

    # Constant per-seed deltas: sample std is exactly 0, so sigma_delta_paired is exactly 0
    # regardless of how noisy each arm looks individually -- the cleanest possible
    # illustration that this estimator lives on the DIFFERENCE, not the two arms' own spread.
    assert agg.sigma_delta_paired([0.05, 0.05, 0.05]) == pytest.approx(0.0)

    # Non-degenerate case: deltas [0.01, 0.03, 0.05], mean 0.03, sample std (ddof=1) = 0.02.
    result = agg.sigma_delta_paired([0.01, 0.03, 0.05])
    assert result == pytest.approx(0.02 / math.sqrt(3))


def test_sigma_delta_paired_matches_real_b3t_minus_b3_worked_example():
    # CURRENT_RESULTS.md section J.3c / the sigma_delta fix task: B3 = [0.3341, 0.2886,
    # 0.3193], B3T = [0.3857, 0.3170, 0.3364] (seeds 42/43/44). Per-seed deltas
    # +0.0516/+0.0284/+0.0171 -> paired std 0.0175 -> paired SE ~= 0.0101, vs. the quadrature
    # estimate ~= 0.0244 (~2.4x larger). Loose tolerances: the doc's numbers are rounded to
    # 4dp.
    b3 = [0.3341, 0.2886, 0.3193]
    b3t = [0.3857, 0.3170, 0.3364]
    deltas = [t - c for t, c in zip(b3t, b3)]
    paired = agg.sigma_delta_paired(deltas)
    unpaired = agg.sigma_delta_standard_error(agg.sample_std(b3t), agg.sample_std(b3), 3)
    assert paired == pytest.approx(0.0101, abs=2e-4)
    assert unpaired == pytest.approx(0.0244, abs=2e-4)
    assert paired < unpaired / 2  # quadrature is roughly 2.4x too large here


def test_sigma_delta_paired_requires_at_least_two_seeds():
    with pytest.raises(ValueError, match="at least 2 seeds"):
        agg.sigma_delta_paired([0.05])


def test_sigma_delta_paired_rejects_empty_input():
    with pytest.raises(ValueError, match="at least 2 seeds"):
        agg.sigma_delta_paired([])


# ----------------------------------------------------------------------------------------
# implied_seed_correlation: purely diagnostic explanation of the gap between the paired and
# unpaired estimates. Worked example is the same real B3T-B3 pair (~0.90 implied
# correlation).
# ----------------------------------------------------------------------------------------
def test_implied_seed_correlation_matches_real_b3t_minus_b3_worked_example():
    b3 = [0.3341, 0.2886, 0.3193]
    b3t = [0.3857, 0.3170, 0.3364]
    deltas = [t - c for t, c in zip(b3t, b3)]
    rho = agg.implied_seed_correlation(
        sigma_a=agg.sample_std(b3t), sigma_b=agg.sample_std(b3), per_seed_deltas=deltas
    )
    assert rho == pytest.approx(0.90, abs=0.02)


def test_implied_seed_correlation_is_one_when_deltas_are_exactly_constant_and_sigmas_equal():
    # A constant per-seed delta means the two arms move in lockstep seed-by-seed -- perfect
    # positive correlation -- even though each arm individually has nonzero across-seed
    # spread. With equal sigma_a/sigma_b this must land exactly on rho=1 (the AM-GM equality
    # case; see the next test for what happens when sigma_a != sigma_b).
    rho = agg.implied_seed_correlation(sigma_a=0.02, sigma_b=0.02, per_seed_deltas=[0.1, 0.1, 0.1])
    assert rho == pytest.approx(1.0)


def test_implied_seed_correlation_can_exceed_one_for_unequal_sigmas_with_constant_delta():
    # Deliberately NOT clamped to [-1, 1] (see the docstring): with a constant delta but
    # UNEQUAL sigma_a/sigma_b, rho = (sigma_a**2+sigma_b**2)/(2*sigma_a*sigma_b), which by
    # AM-GM is >= 1 with equality iff sigma_a == sigma_b. This is small-n-noise overshoot
    # past the mathematically valid correlation range, not a bug -- it should be visible, not
    # silently clamped away.
    rho = agg.implied_seed_correlation(sigma_a=0.02, sigma_b=0.03, per_seed_deltas=[0.1, 0.1, 0.1])
    assert rho == pytest.approx((0.02 ** 2 + 0.03 ** 2) / (2 * 0.02 * 0.03))
    assert rho > 1.0


def test_implied_seed_correlation_is_zero_when_matching_unpaired_quadrature():
    # If the paired single-draw variance exactly equals sigma_a**2 + sigma_b**2 (i.e. the
    # paired and unpaired estimators would agree before the /sqrt(n_seeds) step), the implied
    # correlation must be exactly 0 -- the independence case sigma_delta_standard_error
    # implicitly assumes.
    sigma_a, sigma_b = 0.03, 0.04
    independent_single_draw_sd = (sigma_a ** 2 + sigma_b ** 2) ** 0.5
    deltas = [-independent_single_draw_sd, independent_single_draw_sd, 0.0]
    # sample_std(deltas) must equal independent_single_draw_sd for this to isolate rho=0;
    # verify the fixture before trusting the assertion below.
    assert agg.sample_std(deltas) == pytest.approx(independent_single_draw_sd)
    rho = agg.implied_seed_correlation(sigma_a=sigma_a, sigma_b=sigma_b, per_seed_deltas=deltas)
    assert rho == pytest.approx(0.0, abs=1e-9)


def test_implied_seed_correlation_requires_at_least_two_seeds():
    with pytest.raises(ValueError, match="at least 2 seeds"):
        agg.implied_seed_correlation(sigma_a=0.01, sigma_b=0.02, per_seed_deltas=[0.05])


def test_implied_seed_correlation_rejects_non_positive_sigma():
    with pytest.raises(ValueError, match="sigma_a, sigma_b > 0"):
        agg.implied_seed_correlation(sigma_a=0.0, sigma_b=0.02, per_seed_deltas=[0.05, 0.06])
    with pytest.raises(ValueError, match="sigma_a, sigma_b > 0"):
        agg.implied_seed_correlation(sigma_a=0.02, sigma_b=-0.01, per_seed_deltas=[0.05, 0.06])


# ----------------------------------------------------------------------------------------
# The whole point of the 2026-07-26 sigma_delta_paired fix (independent of the 2026-07-27
# four-state fix, but exercised through the same classify_pair_verdict/classify_group_verdict
# pipeline): a synthetic case where the paired and unpaired estimators yield DIFFERENT
# verdicts for the identical underlying data. Mirrors the real B3T-B3 finding
# (CURRENT_RESULTS.md section J.3c) with clean round numbers: mean_delta is small relative to
# each arm's OWN across-seed spread (quadrature can't resolve it -- stays "indeterminate"),
# but the per-seed delta is tight because both arms move together seed by seed (paired
# resolves it as confidently sub-threshold -- "ineffective").
# ----------------------------------------------------------------------------------------
def test_paired_and_unpaired_sigma_yield_different_verdicts_on_the_same_data():
    control = [0.40, 0.20, 0.30]  # across-seed std = 0.10 (noisy arm)
    treatment = [0.425, 0.220, 0.315]  # per-seed deltas +0.025/+0.020/+0.015: tight, all positive
    sigma_control = agg.sample_std(control)
    sigma_treatment = agg.sample_std(treatment)
    per_seed_deltas = [t - c for t, c in zip(treatment, control)]
    mean_delta = agg.mean(per_seed_deltas)
    assert mean_delta == pytest.approx(0.02)

    sigma_unpaired = agg.sigma_delta_standard_error(sigma_treatment, sigma_control, 3)
    sigma_paired = agg.sigma_delta_paired(per_seed_deltas)
    assert sigma_paired < sigma_unpaired  # paired is tighter, as expected from positive correlation

    # Neither meets the effective clause (mean_delta well below +0.03); that part is
    # unaffected by which sigma is used.
    meets_effective = agg.pair_meets_effective_clause(
        mean_delta=mean_delta, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=per_seed_deltas,
        effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
    )
    assert meets_effective is False

    # The ineffective clause -- mean_delta + 2*sigma_delta_paired < threshold -- flips
    # depending on which estimator feeds it: with the wide unpaired quadrature sigma, the
    # upper bound is nowhere near excluding +0.03; with the tight paired sigma, it is.
    unpaired_verdict, _ = agg.classify_pair_verdict(
        mean_delta=mean_delta, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=per_seed_deltas, sigma_delta_paired=sigma_unpaired,
        effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
    )
    paired_verdict, _ = agg.classify_pair_verdict(
        mean_delta=mean_delta, n_sessions_positive=6, n_sessions_total=6,
        per_seed_means=per_seed_deltas, sigma_delta_paired=sigma_paired,
        effective_mean_delta_threshold=0.03, effective_min_positive_sessions=5,
    )
    assert unpaired_verdict == "indeterminate"  # quadrature: too wide to resolve
    assert paired_verdict == "ineffective"  # paired: tight enough to confidently resolve

    # Wired through the group-level combination too (both pairs identical, as if F1's two
    # comparisons both looked like this).
    old_verdict, _ = agg.classify_group_verdict(
        pair_results={"a": (unpaired_verdict, "..."), "b": (unpaired_verdict, "...")}
    )
    new_verdict, _ = agg.classify_group_verdict(
        pair_results={"a": (paired_verdict, "..."), "b": (paired_verdict, "...")}
    )
    assert old_verdict == "indeterminate"
    assert new_verdict == "ineffective"
    assert old_verdict != new_verdict


# ----------------------------------------------------------------------------------------
# Per-run helpers (no file I/O; pure dict transforms) -- same contract as the attention
# screen aggregator's helpers of the same name.
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
        5: {"sA": 0.10, "sB": 0.20}, 6: {"sA": 0.20, "sB": 0.30},
        7: {"sA": 0.30, "sB": 0.40}, 8: {"sA": 0.40, "sB": 0.50},
        9: {"sA": 0.10, "sB": 0.20}, 10: {"sA": 0.20, "sB": 0.30},
        11: {"sA": 0.30, "sB": 0.40}, 12: {"sA": 0.40, "sB": 0.50},
    })
    scores = agg.per_run_session_scores(payload)
    assert scores["sA"] == pytest.approx((0.10 + 0.20 + 0.30 + 0.40) * 2 / 8)
    assert scores["sB"] == pytest.approx((0.20 + 0.30 + 0.40 + 0.50) * 2 / 8)


def test_per_run_within_window_std_uses_exactly_the_8_epoch_means():
    payload = _toy_payload({epoch: {"sA": 0.1 * (epoch - 4)} for epoch in agg.EXPECTED_EPOCH_WINDOW})
    std = agg.per_run_within_window_std(payload)
    import statistics

    assert std == pytest.approx(statistics.stdev([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))


# ----------------------------------------------------------------------------------------
# Full synthetic 15-artifact pipeline through run_aggregation().
# ----------------------------------------------------------------------------------------
VAL_SESSIONS = [f"sub-X_ses-CO-2015110{i}" for i in range(1, 7)]
TRAIN_SESSIONS = [f"sub-X_ses-CO-2013100{i}" for i in range(1, 4)]
TEST_SESSIONS = [f"sub-X_ses-CO-2015120{i}" for i in range(1, 7)]
SESSION_SPLITS = {"train": TRAIN_SESSIONS, "val": VAL_SESSIONS, "test": TEST_SESSIONS}


def _epoch_values_with_exact_mean(target_mean: float) -> list[float]:
    """8 values alternating +/-0.01 around target_mean; the mean is exactly target_mean
    (equal count of +/- 0.01) so within_window_std is exercised without disturbing the
    hand-computed paired deltas below."""
    wiggle = [-0.01, 0.01] * 4
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
) -> Path:
    contract = agg.GROUP_CONTRACT[group]
    run_dir = run_dir_override or (
        checkpoints_dir / f"side_feature_ablation_v2_{group.lower()}_dandi688_co_s{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_metadata_path = run_dir / "run_metadata.json"
    metadata_payload = {
        "status": "completed",
        "held_out_test_evaluated": False,
        "training": {"max_epochs": 12, "no_early_stopping": True, "checkpoint_every_epoch": True},
        "side_features": {
            "group": contract["side_features_group"],
            "pool_size": 50,
            "side_dim": {"none": 0, "f1": 3, "f2": 6, "fs1": 3, "fs2": 6}[contract["side_features_group"]],
            "permutation_seed": seed if contract["side_features_group"] in ("fs1", "fs2") else None,
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
        "variant": contract["variant"],
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
    out_path = agg.artifact_path(results_dir, group, seed)
    out_path.write_text(json.dumps(payload, sort_keys=True))
    return out_path


# Per-seed target means hand-chosen so that, simultaneously, in ONE consistent dataset
# (F0 is shared between both F1's and F2's comparisons):
#   F1 -> effective    (F1-F0 mean_delta=+0.06, F1-FS1 mean_delta=+0.04; both >= +0.03,
#                        6/6 sessions positive, all 3 seed means positive)
#   F2 -> ineffective   (F2-F0 and F2-FS2 both mean_delta=-0.10, both far outside their
#                        measured sigma_delta ~= 0.014)
TARGET_MEANS = {
    ("F0", 42): 0.30, ("F0", 43): 0.32, ("F0", 44): 0.31,
    ("F1", 42): 0.36, ("F1", 43): 0.38, ("F1", 44): 0.37,
    ("F2", 42): 0.20, ("F2", 43): 0.22, ("F2", 44): 0.21,
    ("FS1", 42): 0.32, ("FS1", 43): 0.34, ("FS1", 44): 0.33,
    ("FS2", 42): 0.30, ("FS2", 43): 0.32, ("FS2", 44): 0.31,
}


def _write_full_screen(tmp_path: Path, *, overrides: dict | None = None) -> tuple[Path, Path]:
    """Write all 15 valid v2 artifacts. ``overrides[(group, seed)]`` may supply kwargs
    forwarded to _write_artifact for exactly that run (used by the negative tests below)."""
    overrides = overrides or {}
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints_dir.mkdir()
    for group in agg.GROUPS:
        for seed in agg.SEEDS:
            kwargs = dict(overrides.get((group, seed), {}))
            _write_artifact(
                results_dir, checkpoints_dir, group, seed, TARGET_MEANS[(group, seed)], **kwargs
            )
    return results_dir, checkpoints_dir


def test_full_pipeline_produces_effective_f1_and_ineffective_f2(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)

    payload = agg.run_aggregation(results_dir)

    assert payload["feature_group_verdicts"]["F1"]["verdict"] == "effective"
    assert payload["feature_group_verdicts"]["F2"]["verdict"] == "ineffective"
    assert payload["feature_group_verdicts"]["F1"]["decided_by"].startswith("effective:")
    assert payload["feature_group_verdicts"]["F2"]["decided_by"].startswith("ineffective:")

    f1_f0 = payload["paired_deltas"]["F1_minus_F0"]
    assert f1_f0["mean_delta"] == pytest.approx(0.06)
    assert f1_f0["n_sessions_positive"] == 6
    assert f1_f0["all_seed_means_positive"] is True
    assert f1_f0["meets_effective_clause"] is True
    assert f1_f0["verdict"] == "effective"
    assert set(f1_f0["per_session_seed_mean"]) == set(VAL_SESSIONS)

    f1_fs1 = payload["paired_deltas"]["F1_minus_FS1"]
    assert f1_fs1["mean_delta"] == pytest.approx(0.04)
    assert f1_fs1["meets_effective_clause"] is True
    assert f1_fs1["verdict"] == "effective"

    f2_f0 = payload["paired_deltas"]["F2_minus_F0"]
    assert f2_f0["mean_delta"] == pytest.approx(-0.10)
    assert f2_f0["meets_effective_clause"] is False
    assert f2_f0["meets_effective_heterogeneous_clause"] is False
    assert f2_f0["exceeds_ineffective_threshold"] is True
    assert f2_f0["verdict"] == "ineffective"
    assert f2_f0["decided_by"].startswith("ineffective:")

    f2_fs2 = payload["paired_deltas"]["F2_minus_FS2"]
    assert f2_fs2["mean_delta"] == pytest.approx(-0.10)
    assert f2_fs2["exceeds_ineffective_threshold"] is True
    assert f2_fs2["verdict"] == "ineffective"

    # Never (F1, FS2) or (F2, FS1) -- dimension-matched pairs only.
    assert set(payload["paired_deltas"]) == {
        "F1_minus_F0", "F1_minus_FS1", "F2_minus_F0", "F2_minus_FS2",
    }

    # Uncertainty block: all measured, nothing borrowed from the doc's priors.
    uncertainty = payload["uncertainty"]
    assert set(uncertainty["within_window_std_per_run"]) == {
        f"{group}_s{seed}" for group in agg.GROUPS for seed in agg.SEEDS
    }
    assert all(value > 0 for value in uncertainty["within_window_std_per_run"].values())
    assert set(uncertainty["across_seed_std_per_group"]) == set(agg.GROUPS)
    expected_pairs = {"F1_minus_F0", "F1_minus_FS1", "F2_minus_F0", "F2_minus_FS2"}
    assert set(uncertainty["sigma_delta_paired_per_pair"]) == expected_pairs
    assert set(uncertainty["sigma_delta_unpaired_quadrature_per_pair"]) == expected_pairs
    assert set(uncertainty["implied_seed_correlation_per_pair"]) == expected_pairs

    # Absolute + per-seed variant scores (requirement 5).
    assert payload["variant_scores"]["F1"]["42"] == pytest.approx(0.36)
    assert payload["variant_scores"]["F0"]["mean"] == pytest.approx((0.30 + 0.32 + 0.31) / 3)

    # Never collapse a verdict into a bool, and never use anything but the four literal
    # strings (section 4.2b).
    for group_payload in payload["feature_group_verdicts"].values():
        assert group_payload["verdict"] in agg.VALID_VERDICTS
        assert not isinstance(group_payload["verdict"], bool)
        assert isinstance(group_payload["decided_by"], str) and group_payload["decided_by"]
    for pair_payload in payload["paired_deltas"].values():
        assert pair_payload["verdict"] in agg.VALID_VERDICTS
        assert not isinstance(pair_payload["verdict"], bool)
        assert isinstance(pair_payload["decided_by"], str) and pair_payload["decided_by"]


def test_pipeline_is_deterministic_across_repeated_aggregation(tmp_path):
    results_dir, _ = _write_full_screen(tmp_path)
    first = agg.run_aggregation(results_dir)
    second = agg.run_aggregation(results_dir)
    assert first == second


def test_full_pipeline_sigma_delta_matches_standard_error_helper(tmp_path):
    # End-to-end pin: BOTH sigma_delta estimates actually written into paired_deltas/
    # uncertainty by run_aggregation must equal sigma_delta_standard_error(...) /
    # sigma_delta_paired(...) applied to this same payload's own across_seed_std_per_group /
    # per_seed_mean, and the ineffective clause must be gated on the PAIRED value (2026-07-26
    # revision), not the unpaired quadrature (M7).
    results_dir, _ = _write_full_screen(tmp_path)
    payload = agg.run_aggregation(results_dir)
    across = payload["uncertainty"]["across_seed_std_per_group"]
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

        # The ineffective clause -- and therefore ineffective_abs_threshold -- must be gated
        # on the PAIRED estimate, not the unpaired quadrature.
        assert pair["ineffective_abs_threshold"] == pytest.approx(2.0 * expected_paired)
        assert pair["exceeds_ineffective_threshold"] == agg.pair_exceeds_ineffective_threshold(
            mean_delta=pair["mean_delta"], sigma_delta_paired=expected_paired,
            effective_mean_delta_threshold=agg.EFFECTIVE_MEAN_DELTA_THRESHOLD,
        )
        # And the four-state verdict itself must match classify_pair_verdict fed the same
        # PAIRED sigma (2026-07-27 revision).
        expected_verdict, expected_decided_by = agg.classify_pair_verdict(
            mean_delta=pair["mean_delta"],
            n_sessions_positive=pair["n_sessions_positive"],
            n_sessions_total=pair["n_sessions_total"],
            per_seed_means=list(pair["per_seed_mean"].values()),
            sigma_delta_paired=expected_paired,
            effective_mean_delta_threshold=agg.EFFECTIVE_MEAN_DELTA_THRESHOLD,
            effective_min_positive_sessions=agg.EFFECTIVE_MIN_POSITIVE_SESSIONS,
        )
        assert pair["verdict"] == expected_verdict
        assert pair["decided_by"] == expected_decided_by


# ----------------------------------------------------------------------------------------
# Consistency gate: negative tests.
# ----------------------------------------------------------------------------------------
def test_rejects_missing_artifact(tmp_path):
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints_dir.mkdir()
    for group in agg.GROUPS:
        for seed in agg.SEEDS:
            if (group, seed) == ("FS2", 44):
                continue  # simulate a still-running / crashed run
            _write_artifact(results_dir, checkpoints_dir, group, seed, TARGET_MEANS[(group, seed)])

    with pytest.raises(FileNotFoundError, match="fs2_s44"):
        agg.run_aggregation(results_dir)


def test_rejects_mismatched_session_splits(tmp_path):
    bad_splits = dict(SESSION_SPLITS)
    bad_splits["val"] = list(VAL_SESSIONS[:-1]) + ["sub-X_ses-CO-99999999"]
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path, overrides={("FS2", 44): {"session_splits": bad_splits}}
    )
    with pytest.raises(ValueError, match="session_splits disagree"):
        agg.run_aggregation(results_dir)


def test_rejects_two_runs_sharing_a_run_directory(tmp_path):
    # This is exactly v3 bug H.4: two (group, seed) runs resolving to one run_dir.
    shared_dir = tmp_path / "checkpoints" / "shared_run_dir"
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={
            ("F1", 43): {"run_dir_override": shared_dir},
            ("F1", 44): {"run_dir_override": shared_dir},
        },
    )
    with pytest.raises(ValueError, match="share a run directory"):
        agg.run_aggregation(results_dir)


def test_rejects_wrong_max_epochs(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path, overrides={("F0", 42): {"run_metadata_overrides": {"training": {"max_epochs": 20}}}}
    )
    with pytest.raises(ValueError, match="max_epochs"):
        agg.run_aggregation(results_dir)


def test_rejects_early_stopping_enabled(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={("F2", 43): {"run_metadata_overrides": {"training": {"no_early_stopping": False}}}},
    )
    with pytest.raises(ValueError, match="no_early_stopping"):
        agg.run_aggregation(results_dir)


def test_rejects_side_features_group_contract_mismatch(tmp_path):
    # A file named f1_s42.json whose run_metadata.json actually recorded side_features.group
    # "f2" (e.g. a mislabeled/misrouted artifact) must be rejected, not silently aggregated
    # as if it were F1 -- this is the dimension-matching guarantee the 2026-07-25 FS1/FS2
    # charter revision depends on.
    results_dir, checkpoints_dir = _write_full_screen(
        tmp_path,
        overrides={("F1", 42): {"run_metadata_overrides": {"side_features": {"group": "f2"}}}},
    )
    with pytest.raises(ValueError, match="side_features.group"):
        agg.run_aggregation(results_dir)


def test_rejects_tampered_run_metadata_after_evaluation(tmp_path):
    results_dir, checkpoints_dir = _write_full_screen(tmp_path)
    tampered_run_dir = checkpoints_dir / "side_feature_ablation_v2_f0_dandi688_co_s42"
    (tampered_run_dir / "run_metadata.json").write_text(
        json.dumps({
            "status": "completed",
            "held_out_test_evaluated": False,
            "training": {"max_epochs": 12, "no_early_stopping": True, "checkpoint_every_epoch": True},
            "side_features": {"group": "none", "pool_size": 50, "side_dim": 0, "permutation_seed": None},
            "tampered": True,
        })
    )
    with pytest.raises(ValueError, match="run_metadata_sha256 mismatch"):
        agg.run_aggregation(results_dir)


def test_rejects_incomplete_results_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        agg.run_aggregation(tmp_path / "nonexistent")
