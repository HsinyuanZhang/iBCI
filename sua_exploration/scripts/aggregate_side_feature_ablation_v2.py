"""Aggregate side_feature_ablation_v2 SUA results under MEASUREMENT_PROTOCOL_V4.

Implements sua_exploration/docs/UNIT_SIDE_FEATURE_ABLATION.md section 8 (predeclared
gates, itself pinned to MEASUREMENT_PROTOCOL_V4.md sections 2-4) over the 15
(group in {F0, F1, F2, FS1, FS2}) x (seed in {42, 43, 44}) epoch-window artifacts that
``eval_epoch_window_dandi688.py`` writes (M3 estimator; reused unmodified, not
reimplemented here) as ``results/side_feature_ablation_v2/{group_lower}_s{seed}.json``.

Group -> (variant, side_features.group) contract (charter section 6):
    F0  -> variant B3,  side_features none  (no side features; = the B3 baseline)
    F1  -> variant B3S, side_features f1    (p2p, noise_std, snr; 3 dims)
    F2  -> variant B3S, side_features f2    (F1 + pt_width, pt_ratio, repol_slope; 6 dims)
    FS1 -> variant B3S, side_features fs1   (F1's features, permuted along the unit axis)
    FS2 -> variant B3S, side_features fs2   (F2's features, permuted along the unit axis)

Four paired comparisons, each dimension-matched (charter section 6, 2026-07-25 revision --
F1 must never be compared against FS2, nor F2 against FS1: that would compare two different
post_pool architectures, fan_in 67 vs 70, with RNG streams diverging from the first layer
on -- exactly the defect that forced the single 6-dim "fs" control to be retired):
    F1_minus_F0, F1_minus_FS1   (F1's own content gate)
    F2_minus_F0, F2_minus_FS2   (F2's own content gate)

The four-state gate (MEASUREMENT_PROTOCOL_V4.md section 4.2b, 2026-07-27 bug fix) is
evaluated per PAIR first (``classify_pair_verdict``, defined here and imported unmodified by
every other aggregator in this repo), then combined per FEATURE GROUP (F1, F2), not per pair:

- a group is "effective" only if ALL of its pairs independently satisfy the effective clause
  (mean delta >= +0.03, >=5 of 6 sessions positive, all 3 per-seed means positive) --
  unchanged from the pre-2026-07-27 protocol;
- a group is "effective_heterogeneous" if ALL of its pairs are independently effective or
  effective_heterogeneous, with at least one only effective_heterogeneous (confidently
  positive and seed-consistent, but not session-consistent);
- a group is "ineffective" if ANY of its pairs is independently and confidently resolved as
  sub-threshold (mean delta + 2*sigma_delta_paired < +0.03) -- CHANGED 2026-07-27 from
  requiring ALL pairs ineffective, which was over-conservative: "if a variant has already
  been confidently excluded relative to its own permutation control, it cannot be effective"
  (section 4.2b);
- otherwise "indeterminate": this is exactly the state attention_arch_screen_v3 collapsed
  into a negative ``gate: false`` (CURRENT_RESULTS.md section H; section E retraction), which
  is the error this protocol exists to prevent.

The pre-2026-07-27 ineffective clause (``abs(mean delta) > 2*sigma_delta``) was itself a bug:
it means "confidently NONZERO", which is backwards for a positive effect -- it labelled
B3T-B3 (mean +0.0324, paired 2*sigma 0.0203, all 3 seeds positive) "ineffective". See
``classify_pair_verdict``'s docstring below for the full fix.

Hard data-isolation constraint (protocol section 6 / charter section 7): this script only
ever reads the JSON artifacts eval_epoch_window_dandi688.py already produced and the
run_metadata.json files they reference. It never opens an NWB file and never reads
spike/behavior/trial data, so it cannot violate the six-test-session isolation rule
regardless of what session names appear in session_splits.test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

GROUPS: tuple[str, ...] = ("F0", "F1", "F2", "FS1", "FS2")
SEEDS: tuple[int, ...] = (42, 43, 44)

# group -> (variant, side_features.group) contract each artifact + its referenced
# run_metadata.json must satisfy (UNIT_SIDE_FEATURE_ABLATION.md section 6).
GROUP_CONTRACT: dict[str, dict[str, str]] = {
    "F0": {"variant": "B3", "side_features_group": "none"},
    "F1": {"variant": "B3S", "side_features_group": "f1"},
    "F2": {"variant": "B3S", "side_features_group": "f2"},
    "FS1": {"variant": "B3S", "side_features_group": "fs1"},
    "FS2": {"variant": "B3S", "side_features_group": "fs2"},
}

# Dimension-matched pairs only -- never (F1, FS2) or (F2, FS1).
PAIRS: tuple[tuple[str, str], ...] = (
    ("F1", "F0"),
    ("F1", "FS1"),
    ("F2", "F0"),
    ("F2", "FS2"),
)
# Which two pairs feed each feature group's three-state verdict.
FEATURE_GROUP_PAIRS: dict[str, tuple[str, str]] = {
    "F1": ("F1_minus_F0", "F1_minus_FS1"),
    "F2": ("F2_minus_F0", "F2_minus_FS2"),
}

EXPECTED_EPOCH_WINDOW = [5, 6, 7, 8, 9, 10, 11, 12]
EXPECTED_PROTOCOL = {"selection_mode": "first", "calibration_n": 30, "pool_size": 50}
EXPECTED_TOTAL_EPOCHS = 12
EXPECTED_SESSION_TOTAL = 6

# Protocol section 4.2 / charter section 8 gate thresholds (verbatim; do not tune after
# the fact). The +0.03 threshold, the seed count, and the estimator window are UNCHANGED by
# the 2026-07-27 section 4.2b fix -- only the state definitions built from them changed.
EFFECTIVE_MEAN_DELTA_THRESHOLD = 0.03
EFFECTIVE_MIN_POSITIVE_SESSIONS = 5
INEFFECTIVE_SIGMA_MULTIPLE = 2.0

VERDICT_EFFECTIVE = "effective"
VERDICT_EFFECTIVE_HETEROGENEOUS = "effective_heterogeneous"
VERDICT_INEFFECTIVE = "ineffective"
VERDICT_INDETERMINATE = "indeterminate"
VALID_VERDICTS = (
    VERDICT_EFFECTIVE,
    VERDICT_EFFECTIVE_HETEROGENEOUS,
    VERDICT_INEFFECTIVE,
    VERDICT_INDETERMINATE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def sample_std(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1); requires at least 2 values."""
    if len(values) < 2:
        raise ValueError(f"sample_std requires at least 2 values, got {len(values)}")
    return float(statistics.stdev(values))


def sigma_delta_standard_error(sigma_a: float, sigma_b: float, n_seeds: int) -> float:
    """Standard error of the n_seeds-seed-MEAN paired delta between two independent groups.

    M7 fix (ROADMAP.md "M": ``sqrt(sigma_a**2 + sigma_b**2)`` alone is the SD of a
    *single-seed* delta (quadrature combination of each group's across-seed SD of
    variant_score) -- it is missing the ``/ sqrt(n_seeds)`` that turns a single-draw SD
    into the standard error of the n_seeds-seed mean this protocol actually reports and
    gates on. Omitting it makes the ineffective threshold ``2*sigma_delta`` too wide by a
    factor of ``sqrt(n_seeds)`` (~1.73x at n_seeds=3), which can only ever bias the
    three-state verdict toward "indeterminate", never toward a false "effective" or
    "ineffective" (MEASUREMENT_PROTOCOL_V4.md section 4.1, 2026-07-26 revision).
    """
    if sigma_a < 0 or sigma_b < 0:
        raise ValueError(f"sigma_a and sigma_b must be non-negative, got {sigma_a!r}, {sigma_b!r}")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    return ((sigma_a ** 2 + sigma_b ** 2) ** 0.5) / (n_seeds ** 0.5)


def sigma_delta_paired(per_seed_deltas: Sequence[float]) -> float:
    """Standard error of the n_seeds-seed-MEAN paired delta -- the PRIMARY sigma_delta
    estimator as of the 2026-07-26 revision (MEASUREMENT_PROTOCOL_V4.md section 4.1),
    superseding ``sigma_delta_standard_error`` (quadrature) for gating purposes.

    Second sigma_delta defect in this estimator (the first was M7's missing
    ``/ sqrt(n_seeds)``, fixed above): ``sigma_delta_standard_error`` combines the treatment
    and control arms' across-seed SDs in quadrature, which implicitly assumes the two arms'
    seed-level effects are statistically independent. They are not -- both arms are trained
    on the SAME seed list, so seed-level difficulty (initialization, data order, ...) is
    largely shared and cancels in the difference. Measured on E4's B3T-B3
    (CURRENT_RESULTS.md section J.3c): quadrature sigma_delta = 0.0244 vs paired sigma_delta
    = 0.0101 -- the quadrature estimate is ~2.4x too large there, an implied inter-arm seed
    correlation of ~0.90 (see ``implied_seed_correlation``). Because overstating sigma_delta
    only ever biases the three-state verdict toward "indeterminate" (never toward a false
    "effective" or "ineffective"), this defect has been silently hiding real effects.

    ``per_seed_deltas`` must already be the SAME-SEED-PAIRED per-seed mean deltas
    (``mean over sessions of (treatment[seed][session] - control[seed][session])``) -- exactly
    the ``per_seed_mean`` values ``run_aggregation`` already computes for the
    ``all_seed_means_positive`` check below, not two independent per-arm score lists.
    """
    n_seeds = len(per_seed_deltas)
    if n_seeds < 2:
        raise ValueError(
            "sigma_delta_paired requires at least 2 seeds (the paired std is undefined below "
            f"that -- refusing to silently fall back to another estimator), got {n_seeds}"
        )
    return sample_std(per_seed_deltas) / (n_seeds ** 0.5)


def implied_seed_correlation(
    *, sigma_a: float, sigma_b: float, per_seed_deltas: Sequence[float]
) -> float:
    """Correlation between the two arms' seed-level effects implied by the gap between the
    unpaired (quadrature) and paired single-seed-draw variances. Purely diagnostic --
    explains *why* ``sigma_delta_paired`` and ``sigma_delta_standard_error`` differ; no gate
    reads this value.

    Derivation: for arms A (treatment) and B (control) sharing one seed list,
    ``Var(A - B) = Var(A) + Var(B) - 2*rho*sigma_A*sigma_B`` at the single-seed-draw scale
    (i.e. before dividing by sqrt(n_seeds) to get a standard error of the mean). ``Var(A-B)``
    at that same scale is exactly the sample variance of ``per_seed_deltas``. Solving for rho:

        rho = (sigma_a**2 + sigma_b**2 - sample_var(per_seed_deltas)) / (2 * sigma_a * sigma_b)

    Not clamped to [-1, 1]: at n_seeds=3 the sample variance of ``per_seed_deltas`` is itself
    noisy enough that a small overshoot outside that range is informative (small-n noise), not
    a bug, and should stay visible rather than be silently hidden by clamping.
    """
    if sigma_a <= 0 or sigma_b <= 0:
        raise ValueError(
            f"implied_seed_correlation requires sigma_a, sigma_b > 0, got {sigma_a!r}, {sigma_b!r}"
        )
    n_seeds = len(per_seed_deltas)
    if n_seeds < 2:
        raise ValueError(f"implied_seed_correlation requires at least 2 seeds, got {n_seeds}")
    paired_single_draw_var = sample_std(per_seed_deltas) ** 2
    return (sigma_a ** 2 + sigma_b ** 2 - paired_single_draw_var) / (2 * sigma_a * sigma_b)


# ------------------------------------------------------------------------------------
# Section 4.2b (2026-07-27 bug fix) / 4.2c: the four-state verdict. THE SINGLE SHARED
# IMPLEMENTATION for every aggregator in this repo -- aggregate_attention_architecture_
# screen_v4.py, aggregate_e3_tuning_ablation.py and aggregate_e4_encoder_variants.py all
# import classify_pair_verdict / classify_group_verdict from here rather than keeping their
# own copies, exactly the discipline sigma_delta_paired's 2026-07-26 fix established for the
# uncertainty estimator, now extended to the verdict itself. Keep these pure (no I/O) -- this
# is the single most load-bearing piece of logic in this module.
#
# THE BUG THIS FIXES: the pre-2026-07-27 ineffective clause was
# ``abs(mean_delta) > 2*sigma_delta`` -- "the effect is confidently NONZERO". For a positive
# effect that reads backwards: it labels a confidently POSITIVE effect "ineffective" whenever
# it is also large relative to its own noise, which a real effect usually is. Measured
# example: B3T-B3 mean_delta=+0.0324, sigma_delta_paired=0.0101, 2*sigma=0.0203, all 3 seeds
# positive (3.2 SE from zero) -- confidently POSITIVE -- was judged "ineffective" under the
# old clause purely because abs(+0.0324) > 0.0203. That is the same class of confusion
# (reading "unresolved" as "negated"), just mirrored, that forced the v3 retraction
# (CURRENT_RESULTS.md section H).
#
# THE FIX: "ineffective" must mean "confidently EXCLUDES an effect of at least
# effective_mean_delta_threshold", not "confidently excludes zero":
#     ineffective  <=>  mean_delta + 2*sigma_delta_paired < effective_mean_delta_threshold
# This also surfaces a previously-missing fourth state: a pair whose mean is confidently
# positive (mean_delta - 2*sigma_delta_paired > 0) with every per-seed mean positive, but
# which fails only the >=5/6-sessions-positive consistency count, is neither a clean
# "effective" nor an unresolved "indeterminate" -- it is a real, seed-replicated,
# session-heterogeneous effect: "effective_heterogeneous".
#
# The ``effective`` clause itself, and the +0.03 / 5-of-6 / all-seeds-positive numbers, are
# UNCHANGED from the pre-2026-07-27 protocol -- this is a bug fix in the state definitions,
# not a moved goalpost (MEASUREMENT_PROTOCOL_V4.md section 4.2b).
# ------------------------------------------------------------------------------------
def pair_meets_effective_clause(
    *,
    mean_delta: float,
    n_sessions_positive: int,
    n_sessions_total: int,
    per_seed_means: Sequence[float],
    effective_mean_delta_threshold: float,
    effective_min_positive_sessions: int,
) -> bool:
    """One pair's effective clause -- UNCHANGED by the 2026-07-27 fix: mean_delta >=
    threshold AND n_sessions_positive >= effective_min_positive_sessions AND every per-seed
    mean positive."""
    if n_sessions_total < 1:
        raise ValueError(f"n_sessions_total must be >= 1, got {n_sessions_total}")
    if not (0 <= n_sessions_positive <= n_sessions_total):
        raise ValueError(
            f"n_sessions_positive ({n_sessions_positive}) must be between 0 and "
            f"n_sessions_total ({n_sessions_total})"
        )
    if not per_seed_means:
        raise ValueError("per_seed_means must be non-empty")
    return (
        mean_delta >= effective_mean_delta_threshold
        and n_sessions_positive >= effective_min_positive_sessions
        and all(value > 0.0 for value in per_seed_means)
    )


def pair_meets_effective_heterogeneous_clause(
    *,
    mean_delta: float,
    sigma_delta_paired: float,
    per_seed_means: Sequence[float],
    n_sessions_positive: int,
    effective_min_positive_sessions: int,
) -> bool:
    """New 2026-07-27 clause (MEASUREMENT_PROTOCOL_V4.md section 4.2b): the pair is
    confidently positive (mean_delta - 2*sigma_delta_paired > 0) with every per-seed mean
    positive, but fails the effective clause ONLY on the session-consistency count
    (n_sessions_positive < effective_min_positive_sessions) -- i.e. it is a real,
    seed-replicated effect that is not present in every validation session.

    Deliberately does NOT also require mean_delta >= effective_mean_delta_threshold: section
    4.2b's table condition is "confidently positive", not "confidently positive AND above the
    deployment threshold".
    """
    if sigma_delta_paired < 0:
        raise ValueError(f"sigma_delta_paired must be non-negative, got {sigma_delta_paired}")
    if not per_seed_means:
        raise ValueError("per_seed_means must be non-empty")
    confidently_positive = (mean_delta - INEFFECTIVE_SIGMA_MULTIPLE * sigma_delta_paired) > 0.0
    all_seeds_positive = all(value > 0.0 for value in per_seed_means)
    session_consistency_fails = n_sessions_positive < effective_min_positive_sessions
    return confidently_positive and all_seeds_positive and session_consistency_fails


def pair_exceeds_ineffective_threshold(
    *,
    mean_delta: float,
    sigma_delta_paired: float,
    effective_mean_delta_threshold: float,
) -> bool:
    """2026-07-27 fix: "ineffective" means "confidently EXCLUDES an effect of at least
    effective_mean_delta_threshold" -- mean_delta + 2*sigma_delta_paired <
    effective_mean_delta_threshold -- NOT the old (backwards, for positive effects)
    "confidently excludes zero" (abs(mean_delta) > 2*sigma_delta_paired)."""
    if sigma_delta_paired < 0:
        raise ValueError(f"sigma_delta_paired must be non-negative, got {sigma_delta_paired}")
    return (mean_delta + INEFFECTIVE_SIGMA_MULTIPLE * sigma_delta_paired) < effective_mean_delta_threshold


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def classify_pair_verdict(
    *,
    mean_delta: float,
    n_sessions_positive: int,
    n_sessions_total: int,
    per_seed_means: Sequence[float],
    sigma_delta_paired: float,
    effective_mean_delta_threshold: float,
    effective_min_positive_sessions: int,
) -> tuple[str, str]:
    """MEASUREMENT_PROTOCOL_V4.md section 4.2b's four-state verdict for ONE paired delta.
    THE single shared implementation every aggregator in this repo calls (imported, not
    copy-pasted) -- see the module-level comment above for the bug this fixes.

    Evaluated as an ordered cascade, matching the top-to-bottom order of section 4.2b's
    table -- effective, then effective_heterogeneous, then ineffective, then indeterminate:

    1. effective: mean_delta >= threshold AND >=effective_min_positive_sessions sessions
       positive AND every per-seed mean positive.
    2. effective_heterogeneous: confidently positive (mean_delta - 2*sigma > 0) AND every
       per-seed mean positive AND the ONLY reason (1) failed is the session-consistency count.
    3. ineffective: mean_delta + 2*sigma < threshold (confidently excludes an effect of at
       least effective_mean_delta_threshold).
    4. indeterminate: none of the above.

    (1) and (3) are mutually exclusive by construction (sigma >= 0: mean_delta >= threshold
    implies mean_delta + 2*sigma >= threshold). (2) and (3) are NOT always mutually
    exclusive -- for sigma small relative to the threshold, a mean can be simultaneously
    "confidently positive" and "confidently below threshold". The cascade order means (2)
    wins when both are true: "this is a real, seed-replicated effect, just not
    session-consistent" is the more decision-relevant fact, and matches section 4.2b's table
    order (effective_heterogeneous listed before ineffective).

    Returns ``(verdict, decided_by)``: ``verdict`` is always exactly one of VALID_VERDICTS
    (never a bool); ``decided_by`` names the clause AND the numbers that decided it (never
    just the verdict word), so a reader can see *why* a pair was resolved the way it was.
    """
    all_seeds_positive = all(value > 0.0 for value in per_seed_means)
    sessions_consistent = n_sessions_positive >= effective_min_positive_sessions
    lower_bound = mean_delta - INEFFECTIVE_SIGMA_MULTIPLE * sigma_delta_paired
    upper_bound = mean_delta + INEFFECTIVE_SIGMA_MULTIPLE * sigma_delta_paired

    if pair_meets_effective_clause(
        mean_delta=mean_delta,
        n_sessions_positive=n_sessions_positive,
        n_sessions_total=n_sessions_total,
        per_seed_means=per_seed_means,
        effective_mean_delta_threshold=effective_mean_delta_threshold,
        effective_min_positive_sessions=effective_min_positive_sessions,
    ):
        decided_by = (
            f"effective: mean={_fmt(mean_delta)} >= threshold={_fmt(effective_mean_delta_threshold)}, "
            f"{n_sessions_positive}/{n_sessions_total} sessions positive "
            f"(>= {effective_min_positive_sessions}), all {len(per_seed_means)} seeds positive"
        )
        return VERDICT_EFFECTIVE, decided_by

    if pair_meets_effective_heterogeneous_clause(
        mean_delta=mean_delta,
        sigma_delta_paired=sigma_delta_paired,
        per_seed_means=per_seed_means,
        n_sessions_positive=n_sessions_positive,
        effective_min_positive_sessions=effective_min_positive_sessions,
    ):
        decided_by = (
            f"effective_heterogeneous: mean-2sigma={_fmt(lower_bound)} > 0 (confidently "
            f"positive; sigma_delta_paired={_fmt(sigma_delta_paired)}), all "
            f"{len(per_seed_means)} seeds positive, but only {n_sessions_positive}/"
            f"{n_sessions_total} sessions positive (< {effective_min_positive_sessions}, "
            "session-consistency clause fails)"
        )
        return VERDICT_EFFECTIVE_HETEROGENEOUS, decided_by

    if pair_exceeds_ineffective_threshold(
        mean_delta=mean_delta,
        sigma_delta_paired=sigma_delta_paired,
        effective_mean_delta_threshold=effective_mean_delta_threshold,
    ):
        decided_by = (
            f"ineffective: mean+2sigma={_fmt(upper_bound)} < threshold="
            f"{_fmt(effective_mean_delta_threshold)} (sigma_delta_paired="
            f"{_fmt(sigma_delta_paired)})"
        )
        return VERDICT_INEFFECTIVE, decided_by

    decided_by = (
        "indeterminate: none of effective/effective_heterogeneous/ineffective satisfied "
        f"(mean={_fmt(mean_delta)}, mean-2sigma={_fmt(lower_bound)}, mean+2sigma="
        f"{_fmt(upper_bound)}, threshold={_fmt(effective_mean_delta_threshold)}, "
        f"{n_sessions_positive}/{n_sessions_total} sessions positive, "
        f"all_seed_means_positive={all_seeds_positive}, sessions_consistent={sessions_consistent})"
    )
    return VERDICT_INDETERMINATE, decided_by


def classify_group_verdict(*, pair_results: Mapping[str, tuple[str, str]]) -> tuple[str, str]:
    """Section 4.2b's group-level combination of one or more dimension-matched pairs (E3's
    T4/T8 and side-features' F1/F2 each combine 2 pairs; e4_encoder_variants' single-pair
    variants call this with a group of 1, which degenerates to returning that pair's own
    verdict wrapped with a group-shaped decided_by).

    ``pair_results`` maps pair_name -> (verdict, decided_by) as returned by
    ``classify_pair_verdict`` for each of the group's pairs.

    - effective: ALL pairs are pair-level 'effective' (AND) -- unchanged from the
      pre-2026-07-27 protocol.
    - effective_heterogeneous: ALL pairs are pair-level 'effective' or
      'effective_heterogeneous' (AND), with at least one 'effective_heterogeneous' (if every
      pair were fully 'effective' the first branch above would already have matched).
    - ineffective: ANY pair is pair-level 'ineffective' (OR). CHANGED 2026-07-27 from
      requiring ALL pairs ineffective, which was over-conservative: "if a variant has already
      been confidently excluded relative to its own permutation control, it cannot be
      effective" (MEASUREMENT_PROTOCOL_V4.md section 4.2b).
    - indeterminate: otherwise.

    These four branches are mutually exclusive by construction: per-pair verdicts are
    themselves mutually exclusive, so "any pair ineffective" can only be reached once the
    first two branches (which both require every pair to be effective or
    effective_heterogeneous) have already tested false.
    """
    if not pair_results:
        raise ValueError("pair_results must be non-empty")
    invalid = {name: verdict for name, (verdict, _) in pair_results.items() if verdict not in VALID_VERDICTS}
    if invalid:
        raise ValueError(f"Unknown verdict(s) in pair_results: {invalid}")

    names = list(pair_results)
    verdict_of = {name: pair_results[name][0] for name in names}

    if all(verdict_of[name] == VERDICT_EFFECTIVE for name in names):
        detail = "; ".join(f"{name} effective ({pair_results[name][1]})" for name in names)
        return VERDICT_EFFECTIVE, f"effective: all {len(names)} pair(s) independently effective -- {detail}"

    all_effective_or_heterogeneous = all(
        verdict_of[name] in (VERDICT_EFFECTIVE, VERDICT_EFFECTIVE_HETEROGENEOUS) for name in names
    )
    heterogeneous_names = [name for name in names if verdict_of[name] == VERDICT_EFFECTIVE_HETEROGENEOUS]
    if all_effective_or_heterogeneous and heterogeneous_names:
        detail = "; ".join(f"{name} {verdict_of[name]} ({pair_results[name][1]})" for name in names)
        return VERDICT_EFFECTIVE_HETEROGENEOUS, (
            f"effective_heterogeneous: all {len(names)} pair(s) effective or "
            f"effective_heterogeneous, {len(heterogeneous_names)}/{len(names)} only "
            f"effective_heterogeneous -- {detail}"
        )

    ineffective_names = [name for name in names if verdict_of[name] == VERDICT_INEFFECTIVE]
    if ineffective_names:
        detail = "; ".join(f"{name}: {pair_results[name][1]}" for name in ineffective_names)
        return VERDICT_INEFFECTIVE, (
            f"ineffective: {len(ineffective_names)}/{len(names)} pair(s) independently "
            f"resolved ineffective (any is sufficient) -- {detail}"
        )

    detail = "; ".join(f"{name}={verdict_of[name]} ({pair_results[name][1]})" for name in names)
    return VERDICT_INDETERMINATE, f"indeterminate: {detail}"


# ------------------------------------------------------------------------------------
# Artifact loading and validation (charter section 8 / protocol section 2).
# ------------------------------------------------------------------------------------
def artifact_path(results_dir: Path, group: str, seed: int) -> Path:
    return results_dir / f"{group.lower()}_s{seed}.json"


def load_artifact(path: Path, *, group: str, seed: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing side_feature_ablation_v2 artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_variant = GROUP_CONTRACT[group]["variant"]
    if payload.get("variant") != expected_variant:
        raise ValueError(
            f"{path}: variant mismatch for group {group}: expected {expected_variant!r}, "
            f"found {payload.get('variant')!r}"
        )
    if payload.get("seed") != seed:
        raise ValueError(
            f"{path}: seed mismatch, expected {seed!r}, found {payload.get('seed')!r}"
        )
    return payload


def validate_artifact_contract(payload: dict, *, path: Path) -> None:
    """Per-artifact checks against the fixed protocol constants (protocol section 2)."""
    if payload.get("epoch_list") != EXPECTED_EPOCH_WINDOW:
        raise ValueError(
            f"{path}: epoch_list must be {EXPECTED_EPOCH_WINDOW}, found {payload.get('epoch_list')}"
        )
    protocol = payload.get("protocol", {})
    if protocol.get("epoch_window") != EXPECTED_EPOCH_WINDOW:
        raise ValueError(f"{path}: protocol.epoch_window must be {EXPECTED_EPOCH_WINDOW}")
    if protocol.get("total_epochs") != EXPECTED_TOTAL_EPOCHS:
        raise ValueError(f"{path}: protocol.total_epochs must be {EXPECTED_TOTAL_EPOCHS}")
    observed_protocol = {key: protocol.get(key) for key in EXPECTED_PROTOCOL}
    if observed_protocol != EXPECTED_PROTOCOL:
        raise ValueError(
            f"{path}: fixed protocol mismatch: expected {EXPECTED_PROTOCOL}, found {observed_protocol}"
        )
    if payload.get("no_test_files_evaluated") is not True:
        raise ValueError(f"{path}: no_test_files_evaluated must be true")
    if payload.get("calibration_trial_selection_uses_behavior_labels") is not False:
        raise ValueError(f"{path}: calibration_trial_selection_uses_behavior_labels must be false")
    if payload.get("uses_behavior_labels_for_weight_updates") is not False:
        raise ValueError(f"{path}: uses_behavior_labels_for_weight_updates must be false")
    if payload.get("uses_backward_gradients") is not False:
        raise ValueError(f"{path}: uses_backward_gradients must be false")

    val_sessions = set(payload.get("session_splits", {}).get("val", []))
    if len(val_sessions) != EXPECTED_SESSION_TOTAL:
        raise ValueError(
            f"{path}: expected {EXPECTED_SESSION_TOTAL} validation sessions, found {len(val_sessions)}"
        )
    per_epoch = payload.get("per_epoch", {})
    if sorted(int(key) for key in per_epoch) != EXPECTED_EPOCH_WINDOW:
        raise ValueError(f"{path}: per_epoch keys must be exactly {EXPECTED_EPOCH_WINDOW}")
    for epoch in EXPECTED_EPOCH_WINDOW:
        epoch_sessions = set(per_epoch[str(epoch)].get("per_session_r2", {}))
        if epoch_sessions != val_sessions:
            raise ValueError(
                f"{path}: epoch {epoch} per_session_r2 sessions {sorted(epoch_sessions)} "
                f"!= validation sessions {sorted(val_sessions)}"
            )


def validate_training_provenance(payload: dict, *, group: str, path: Path) -> None:
    """Cross-check training.max_epochs / no_early_stopping / checkpoint_every_epoch AND
    the side_features.group contract against the sha256-pinned run_metadata.json the
    artifact references, rather than trusting the epoch-window JSON's own fields."""
    run_metadata_path = Path(payload["run_metadata_path"])
    if not run_metadata_path.is_file():
        raise FileNotFoundError(
            f"{path}: referenced run_metadata_path does not exist: {run_metadata_path}"
        )
    observed_hash = sha256_file(run_metadata_path)
    expected_hash = payload["run_metadata_sha256"]
    if observed_hash != expected_hash:
        raise ValueError(
            f"{path}: run_metadata_sha256 mismatch for {run_metadata_path}: expected "
            f"{expected_hash}, observed {observed_hash} (training metadata changed since "
            "evaluation -- re-run eval_epoch_window_dandi688.py)"
        )
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    training = run_metadata.get("training", {})
    side_features_meta = run_metadata.get("side_features") or {}
    expected_side_group = GROUP_CONTRACT[group]["side_features_group"]
    checks = {
        "status": (run_metadata.get("status"), "completed"),
        "held_out_test_evaluated": (run_metadata.get("held_out_test_evaluated"), False),
        "training.max_epochs": (training.get("max_epochs"), EXPECTED_TOTAL_EPOCHS),
        "training.no_early_stopping": (training.get("no_early_stopping"), True),
        "training.checkpoint_every_epoch": (training.get("checkpoint_every_epoch"), True),
        "side_features.group": (side_features_meta.get("group"), expected_side_group),
    }
    mismatches = {
        key: {"expected": expected, "observed": observed}
        for key, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(
            f"{path}: training provenance mismatch in {run_metadata_path}: {mismatches}"
        )


def validate_cross_artifact_consistency(artifacts: Mapping[tuple[str, int], dict]) -> dict:
    """Cross-artifact checks: session split agreement and run-directory uniqueness (v3
    bug H.4) across all 15 artifacts."""
    reference_key = next(iter(artifacts))
    reference_splits = artifacts[reference_key]["session_splits"]
    mismatched = [
        key for key, payload in artifacts.items() if payload["session_splits"] != reference_splits
    ]
    if mismatched:
        raise ValueError(
            f"session_splits disagree across v2 artifacts (reference={reference_key}): {mismatched}"
        )

    run_dir_owner: dict[str, tuple[str, int]] = {}
    duplicates: list[tuple[tuple[str, int], tuple[str, int], str]] = []
    for key, payload in artifacts.items():
        run_dir = payload["run_dir"]
        if run_dir in run_dir_owner:
            duplicates.append((run_dir_owner[run_dir], key, run_dir))
        else:
            run_dir_owner[run_dir] = key
    if duplicates:
        raise ValueError(
            f"Two or more v2 runs share a run directory (this was v3 bug H.4): {duplicates}"
        )

    return reference_splits


# ------------------------------------------------------------------------------------
# Per-run scoring helpers (identical in spirit to aggregate_attention_architecture_screen_v4.py).
# ------------------------------------------------------------------------------------
def per_run_session_scores(payload: dict) -> dict[str, float]:
    """8-epoch-window mean R2 for each validation session, for one (group, seed) run."""
    sessions = sorted(payload["session_splits"]["val"])
    per_epoch = payload["per_epoch"]
    return {
        session: mean([per_epoch[str(epoch)]["per_session_r2"][session] for epoch in EXPECTED_EPOCH_WINDOW])
        for session in sessions
    }


def per_run_within_window_std(payload: dict) -> float:
    """Within-window std (protocol section 3, item 2): std of this run's 8 per-epoch mean
    R2 values."""
    per_epoch_mean_r2 = payload["per_epoch_mean_r2"]
    values = [per_epoch_mean_r2[str(epoch)] for epoch in EXPECTED_EPOCH_WINDOW]
    return sample_std(values)


# ------------------------------------------------------------------------------------
# Top-level aggregation.
# ------------------------------------------------------------------------------------
def run_aggregation(results_dir: Path) -> dict:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    artifacts: dict[tuple[str, int], dict] = {}
    missing: list[str] = []
    for group in GROUPS:
        for seed in SEEDS:
            path = artifact_path(results_dir, group, seed)
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = load_artifact(path, group=group, seed=seed)
            validate_artifact_contract(payload, path=path)
            validate_training_provenance(payload, group=group, path=path)
            artifacts[(group, seed)] = payload
    if missing:
        raise FileNotFoundError(
            "Missing side_feature_ablation_v2 artifact(s); cannot aggregate until all 15 "
            "runs have completed training + eval_epoch_window_dandi688.py evaluation: "
            + ", ".join(missing)
        )

    reference_splits = validate_cross_artifact_consistency(artifacts)
    val_sessions = sorted(reference_splits["val"])

    session_scores = {key: per_run_session_scores(payload) for key, payload in artifacts.items()}
    within_window_std = {key: per_run_within_window_std(payload) for key, payload in artifacts.items()}

    # Integrity check: the per-session-then-mean-over-sessions score must reproduce the
    # artifact's own recorded variant_score.
    for key, payload in artifacts.items():
        recomputed = mean(list(session_scores[key].values()))
        recorded = payload["variant_score"]
        if abs(recomputed - recorded) > 1e-9:
            raise ValueError(
                f"{key}: recomputed variant_score {recomputed!r} does not match recorded "
                f"{recorded!r} in {artifact_path(results_dir, *key)}"
            )

    variant_scores = {
        group: {seed: artifacts[(group, seed)]["variant_score"] for seed in SEEDS}
        for group in GROUPS
    }
    variant_score_mean = {group: mean(list(variant_scores[group].values())) for group in GROUPS}
    across_seed_std = {
        group: sample_std(list(variant_scores[group].values())) for group in GROUPS
    }
    within_window_std_by_run = {
        f"{group}_s{seed}": within_window_std[(group, seed)] for group in GROUPS for seed in SEEDS
    }
    within_window_std_pooled_mean = mean(list(within_window_std_by_run.values()))

    paired_deltas: dict[str, dict] = {}
    for treatment, control in PAIRS:
        pair_name = f"{treatment}_minus_{control}"
        sigma_unpaired = sigma_delta_standard_error(
            across_seed_std[treatment], across_seed_std[control], len(SEEDS)
        )

        per_session_seed_mean: dict[str, float] = {}
        per_seed_values: dict[int, list[float]] = {seed: [] for seed in SEEDS}
        for session in val_sessions:
            session_deltas = []
            for seed in SEEDS:
                delta = session_scores[(treatment, seed)][session] - session_scores[(control, seed)][session]
                session_deltas.append(delta)
                per_seed_values[seed].append(delta)
            per_session_seed_mean[session] = mean(session_deltas)
        per_seed_mean = {seed: mean(values) for seed, values in per_seed_values.items()}
        per_seed_mean_values = list(per_seed_mean.values())

        # Primary estimator (2026-07-26 revision): paired directly on the same-seed deltas
        # rather than combined in quadrature from the two arms' independent across-seed SDs
        # (sigma_delta_paired's docstring has the measured evidence for why this matters).
        sigma_paired = sigma_delta_paired(per_seed_mean_values)
        seed_correlation = implied_seed_correlation(
            sigma_a=across_seed_std[treatment],
            sigma_b=across_seed_std[control],
            per_seed_deltas=per_seed_mean_values,
        )

        mean_delta = mean(list(per_session_seed_mean.values()))
        mean_delta_alt = mean(per_seed_mean_values)
        assert abs(mean_delta - mean_delta_alt) < 1e-9, (
            "grand mean must agree regardless of marginalization order "
            f"({mean_delta!r} vs {mean_delta_alt!r})"
        )

        n_positive = sum(1 for value in per_session_seed_mean.values() if value > 0.0)
        meets_effective = pair_meets_effective_clause(
            mean_delta=mean_delta,
            n_sessions_positive=n_positive,
            n_sessions_total=len(val_sessions),
            per_seed_means=per_seed_mean_values,
            effective_mean_delta_threshold=EFFECTIVE_MEAN_DELTA_THRESHOLD,
            effective_min_positive_sessions=EFFECTIVE_MIN_POSITIVE_SESSIONS,
        )
        meets_effective_heterogeneous = pair_meets_effective_heterogeneous_clause(
            mean_delta=mean_delta,
            sigma_delta_paired=sigma_paired,
            per_seed_means=per_seed_mean_values,
            n_sessions_positive=n_positive,
            effective_min_positive_sessions=EFFECTIVE_MIN_POSITIVE_SESSIONS,
        )
        # Ineffective clause is gated on the PAIRED estimate (primary as of the 2026-07-26
        # revision) -- NOT the unpaired quadrature, which is retained only for comparison
        # below. 2026-07-27: "ineffective" now means mean+2sigma < threshold (see
        # pair_exceeds_ineffective_threshold's docstring), not the old abs(mean)>2sigma.
        exceeds_ineffective = pair_exceeds_ineffective_threshold(
            mean_delta=mean_delta,
            sigma_delta_paired=sigma_paired,
            effective_mean_delta_threshold=EFFECTIVE_MEAN_DELTA_THRESHOLD,
        )
        verdict, decided_by = classify_pair_verdict(
            mean_delta=mean_delta,
            n_sessions_positive=n_positive,
            n_sessions_total=len(val_sessions),
            per_seed_means=per_seed_mean_values,
            sigma_delta_paired=sigma_paired,
            effective_mean_delta_threshold=EFFECTIVE_MEAN_DELTA_THRESHOLD,
            effective_min_positive_sessions=EFFECTIVE_MIN_POSITIVE_SESSIONS,
        )
        paired_deltas[pair_name] = {
            "treatment": treatment,
            "control": control,
            "per_session_seed_mean": per_session_seed_mean,
            "per_seed_mean": {str(seed): value for seed, value in per_seed_mean.items()},
            "mean_delta": mean_delta,
            "n_sessions_positive": n_positive,
            "n_sessions_total": len(val_sessions),
            "all_seed_means_positive": all(value > 0.0 for value in per_seed_mean_values),
            "sigma_delta_paired": sigma_paired,
            "sigma_delta_unpaired_quadrature": sigma_unpaired,
            "implied_seed_correlation": seed_correlation,
            "ineffective_abs_threshold": INEFFECTIVE_SIGMA_MULTIPLE * sigma_paired,
            "effective_mean_delta_threshold": EFFECTIVE_MEAN_DELTA_THRESHOLD,
            "effective_min_positive_sessions": EFFECTIVE_MIN_POSITIVE_SESSIONS,
            "meets_effective_clause": meets_effective,
            "meets_effective_heterogeneous_clause": meets_effective_heterogeneous,
            "exceeds_ineffective_threshold": exceeds_ineffective,
            "verdict": verdict,
            "decided_by": decided_by,
        }

    feature_group_verdicts: dict[str, dict] = {}
    for group, (pair_a_name, pair_b_name) in FEATURE_GROUP_PAIRS.items():
        pair_results = {
            pair_a_name: (paired_deltas[pair_a_name]["verdict"], paired_deltas[pair_a_name]["decided_by"]),
            pair_b_name: (paired_deltas[pair_b_name]["verdict"], paired_deltas[pair_b_name]["decided_by"]),
        }
        verdict, decided_by = classify_group_verdict(pair_results=pair_results)
        feature_group_verdicts[group] = {
            "content_pair_vs_f0": pair_a_name,
            "content_pair_vs_shuffled_control": pair_b_name,
            "verdict": verdict,
            "decided_by": decided_by,
        }

    return {
        "schema_version": 1,
        "purpose": "side_feature_ablation_v2_measurement_protocol_v4",
        "protocol_docs": [
            "sua_exploration/docs/UNIT_SIDE_FEATURE_ABLATION.md",
            "sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md",
        ],
        "screen_id": "side_feature_ablation_v2",
        "no_formal_test_sessions_evaluated": True,
        "groups": list(GROUPS),
        "seeds": list(SEEDS),
        "group_contract": GROUP_CONTRACT,
        "session_splits": reference_splits,
        "fixed_protocol": EXPECTED_PROTOCOL,
        "epoch_window": EXPECTED_EPOCH_WINDOW,
        "consistency_validated": True,
        "variant_scores": {
            group: {
                **{str(seed): variant_scores[group][seed] for seed in SEEDS},
                "mean": variant_score_mean[group],
            }
            for group in GROUPS
        },
        "uncertainty": {
            "definition": (
                "Measured directly from this screen's 15 artifacts per "
                "MEASUREMENT_PROTOCOL_V4.md section 3. None of these values reuse the "
                "doc's prior sigma_epoch=0.0388 / sigma_delta=0.0112 estimates, which were "
                "measured on attention_arch_screen_v3's best-checkpoint trajectories and "
                "are explicitly not to be carried over (section 3)."
            ),
            "within_window_std_per_run": within_window_std_by_run,
            "within_window_std_pooled_mean": within_window_std_pooled_mean,
            "across_seed_std_per_group": across_seed_std,
            "sigma_delta_paired_per_pair": {
                f"{treatment}_minus_{control}": paired_deltas[f"{treatment}_minus_{control}"]["sigma_delta_paired"]
                for treatment, control in PAIRS
            },
            "sigma_delta_unpaired_quadrature_per_pair": {
                f"{treatment}_minus_{control}": paired_deltas[f"{treatment}_minus_{control}"]["sigma_delta_unpaired_quadrature"]
                for treatment, control in PAIRS
            },
            "implied_seed_correlation_per_pair": {
                f"{treatment}_minus_{control}": paired_deltas[f"{treatment}_minus_{control}"]["implied_seed_correlation"]
                for treatment, control in PAIRS
            },
            "sigma_delta_method": (
                "PRIMARY, used by the ineffective clause / three-state verdict: "
                "sigma_delta_paired = stdev(per_seed_mean_deltas, ddof=1) / sqrt(n_seeds), "
                "computed directly from the same-seed-paired per-seed mean deltas ('per_seed_"
                "mean' above), not from the two arms' independent across-seed SDs. "
                "SECONDARY, retained for comparison only and NOT gated on: "
                "sigma_delta_unpaired_quadrature = sqrt(across_seed_std[treatment]**2 + "
                "across_seed_std[control]**2) / sqrt(n_seeds) (the M7-fixed quadrature "
                "combination; see sigma_delta_standard_error()). The quadrature form assumes "
                "the two arms' seed-level effects are statistically independent -- they are "
                "not, since both arms share the same seed list, so seed-level difficulty "
                "largely cancels in the paired difference. This made the quadrature estimate "
                "systematically too large (biasing the verdict toward 'indeterminate'); see "
                "implied_seed_correlation_per_pair for the measured inter-arm correlation "
                "that explains the gap between the two estimates (2026-07-26 revision, "
                "MEASUREMENT_PROTOCOL_V4.md section 4.1; see sigma_delta_paired())."
            ),
        },
        "paired_deltas": paired_deltas,
        "feature_group_verdicts": feature_group_verdicts,
        "source_artifacts": {
            f"{group}_s{seed}": str(artifact_path(results_dir, group, seed))
            for group in GROUPS
            for seed in SEEDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Defaults to sua_exploration/results/side_feature_ablation_v2",
    )
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    results_dir = args.results_dir or (root / "sua_exploration" / "results" / "side_feature_ablation_v2")

    payload = run_aggregation(results_dir)

    out_path = args.out_path or (results_dir / "aggregate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {group: data["verdict"] for group, data in payload["feature_group_verdicts"].items()},
        indent=2, sort_keys=True,
    ))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
