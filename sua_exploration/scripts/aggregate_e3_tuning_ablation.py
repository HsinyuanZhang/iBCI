"""Aggregate e3_tuning_ablation SUA results under MEASUREMENT_PROTOCOL_V4.

Implements sua_exploration/docs/E3_E4_ENCODER_PROGRAM.md sections 1 and 4 (directional
tuning side features), pinned to MEASUREMENT_PROTOCOL_V4.md sections 2-4, over the
(group in {F0, T4, T8, TS4, TS8}) x (seed in --seeds) epoch-window artifacts that
``eval_epoch_window_generic_dandi688.py`` writes (M3 estimator, generalized to a
CLI-supplied epoch budget; NOT the frozen ``eval_epoch_window_dandi688.py``, since E3's
epoch budget comes from E2 and is not guaranteed to be the frozen script's hardcoded 12) as
``results/e3_tuning_ablation/{group_lower}_s{seed}.json``.

Group -> (variant, side_features.group) contract (charter sections 1.2, 4):
    F0  -> variant B3,  side_features none  (no side features; = the B3 baseline)
    T4  -> variant B3S, side_features t4    (cosine-tuning fit [m*cosphi, m*sinphi, m, b])
    T8  -> variant B3S, side_features t8    (per-direction mean firing rate, 8 dims)
    TS4 -> variant B3S, side_features ts4   (T4's features, permuted along the unit axis)
    TS8 -> variant B3S, side_features ts8   (T8's features, permuted along the unit axis)

Four paired comparisons, each dimension-matched (charter section 1.2/4, mirroring the
F1/F2/FS1/FS2 2026-07-25 revision this same discipline is named after in
aggregate_side_feature_ablation_v2.py -- T4 must never be compared against TS8, nor T8
against TS4: that would compare two different post_pool architectures with RNG streams
diverging from the first layer on):
    T4_minus_F0, T4_minus_TS4   (T4's own content gate)
    T8_minus_F0, T8_minus_TS8   (T8's own content gate)

The four-state gate (MEASUREMENT_PROTOCOL_V4.md section 4.2b, 2026-07-27 bug fix) is
evaluated per pair first (``classify_pair_verdict``, imported unmodified from
aggregate_side_feature_ablation_v2 -- the single shared implementation every aggregator in
this repo uses), then combined per TUNING FEATURE GROUP (T4, T8), not per pair, via the same
module's ``classify_group_verdict``:

- a group is "effective" only if ALL of its pairs independently satisfy the effective clause
  (mean delta >= --effective_mean_delta, >=5 of 6 sessions positive, all per-seed means
  positive) -- unchanged from the pre-2026-07-27 protocol;
- a group is "effective_heterogeneous" if ALL of its pairs are independently effective or
  effective_heterogeneous, with at least one only effective_heterogeneous (confidently
  positive and seed-consistent, but not session-consistent);
- a group is "ineffective" if ANY of its pairs is independently and confidently resolved as
  sub-threshold (mean delta + 2*sigma_delta_paired < --effective_mean_delta) -- CHANGED
  2026-07-27 from requiring ALL pairs ineffective, which was over-conservative: "if a variant
  has already been confidently excluded relative to its own permutation control, it cannot be
  effective" (section 4.2b);
- otherwise "indeterminate": this is exactly the state attention_arch_screen_v3 collapsed
  into a negative ``gate: false`` (CURRENT_RESULTS.md section H; section E retraction), which
  is the error this protocol exists to prevent.

The pre-2026-07-27 ineffective clause (``abs(mean delta) > 2*sigma_delta``) was itself a bug
("confidently NONZERO" is backwards for a positive effect); see aggregate_side_feature_
ablation_v2.classify_pair_verdict's docstring for the full fix and the measured example
(B3T-B3) that exposed it.

Two generalizations relative to aggregate_side_feature_ablation_v2.py, both because E3 is
launched before E1/E2 have supplied numbers that v2's screen already had pinned
(E3_E4_ENCODER_PROGRAM.md section 0):

1. Epoch budget/window: v2 hardcodes EXPECTED_EPOCH_WINDOW=[5..12]/EXPECTED_TOTAL_EPOCHS=12
   because that screen's budget was already frozen. Here, every artifact is checked for
   INTERNAL self-consistency (its own epoch_list agrees with its own
   protocol.epoch_window/total_epochs/burn_in_epochs) and every artifact in one aggregation
   run is checked for EXTERNAL cross-artifact agreement (all runs used the identical
   window) -- which is the actual content of M2's "equal budget for every variant"
   requirement, not a comparison against a hardcoded absolute epoch count.
2. Seed count and the effective-mean-delta threshold are required CLI arguments
   (--seeds, --effective_mean_delta) rather than hardcoded module constants: E1's measured
   sigma_seed sets the seed count and, together with E2's epoch budget, the threshold value
   (charter section 0's rule: max(2*sigma_delta_measured, deployment floor)). This script
   does not compute or guess either.

sigma_delta_standard_error (M7 fix: the ``/ sqrt(n_seeds)`` term) is imported unmodified
from aggregate_side_feature_ablation_v2 rather than re-implemented, per
E3_E4_ENCODER_PROGRAM.md's instruction not to regress it.

Hard data-isolation constraint (protocol section 6 / charter section 3): this script only
ever reads the JSON artifacts eval_epoch_window_generic_dandi688.py already produced and the
run_metadata.json files they reference. It never opens an NWB file and never reads
spike/behavior/trial data, so it cannot violate the six-test-session isolation rule
regardless of what session names appear in session_splits.test.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dandi688_gradient_free_protocol import sha256_file  # noqa: E402
from aggregate_side_feature_ablation_v2 import (  # noqa: E402
    VALID_VERDICTS,
    VERDICT_EFFECTIVE,
    VERDICT_EFFECTIVE_HETEROGENEOUS,
    VERDICT_INDETERMINATE,
    VERDICT_INEFFECTIVE,
    classify_group_verdict,
    classify_pair_verdict,
    implied_seed_correlation,
    pair_exceeds_ineffective_threshold,
    pair_meets_effective_clause,
    pair_meets_effective_heterogeneous_clause,
    sigma_delta_paired,
    sigma_delta_standard_error,
)

GROUPS: tuple[str, ...] = ("F0", "T4", "T8", "TS4", "TS8")

# group -> (variant, side_features.group) contract each artifact + its referenced
# run_metadata.json must satisfy (E3_E4_ENCODER_PROGRAM.md section 1.2).
GROUP_CONTRACT: dict[str, dict[str, str]] = {
    "F0": {"variant": "B3", "side_features_group": "none"},
    "T4": {"variant": "B3S", "side_features_group": "t4"},
    "T8": {"variant": "B3S", "side_features_group": "t8"},
    "TS4": {"variant": "B3S", "side_features_group": "ts4"},
    "TS8": {"variant": "B3S", "side_features_group": "ts8"},
}

# Dimension-matched pairs only -- never (T4, TS8) or (T8, TS4).
PAIRS: tuple[tuple[str, str], ...] = (
    ("T4", "F0"),
    ("T4", "TS4"),
    ("T8", "F0"),
    ("T8", "TS8"),
)
# Which two pairs feed each tuning feature group's three-state verdict.
TUNING_GROUP_PAIRS: dict[str, tuple[str, str]] = {
    "T4": ("T4_minus_F0", "T4_minus_TS4"),
    "T8": ("T8_minus_F0", "T8_minus_TS8"),
}

# Fixed sub-protocol, unaffected by E1/E2 (MEASUREMENT_PROTOCOL_V4.md section 2.3): the
# forward-calibration evaluation itself, not the epoch budget.
EXPECTED_PROTOCOL = {"selection_mode": "first", "calibration_n": 30, "pool_size": 50}
# Fixed by --split_counts 27,6,6, which both runners must use verbatim (charter section 4)
# -- not an E1/E2 unknown.
EXPECTED_SESSION_TOTAL = 6

# Protocol section 4.2 gate constants that are NOT epoch-budget/seed-count dependent (only
# the mean-delta threshold itself is; see --effective_mean_delta).
EFFECTIVE_MIN_POSITIVE_SESSIONS = 5
INEFFECTIVE_SIGMA_MULTIPLE = 2.0

# VERDICT_EFFECTIVE / VERDICT_EFFECTIVE_HETEROGENEOUS / VERDICT_INEFFECTIVE /
# VERDICT_INDETERMINATE / VALID_VERDICTS are imported from aggregate_side_feature_ablation_v2
# above (single shared implementation; see that module's docstring for the 2026-07-27
# four-state fix).


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def sample_std(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1); requires at least 2 values."""
    if len(values) < 2:
        raise ValueError(f"sample_std requires at least 2 values, got {len(values)}")
    return float(statistics.stdev(values))


def parse_seeds(text: str) -> list[int]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("--seeds must contain at least one seed")
    try:
        seeds = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"--seeds must be a comma-separated list of integers, got {text!r}") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds must not contain duplicate values, got {text!r}")
    return seeds


# classify_pair_verdict / classify_group_verdict (section 4.2b's four-state gate, 2026-07-27
# bug fix) are imported unmodified from aggregate_side_feature_ablation_v2 above -- the single
# shared implementation every aggregator in this repo uses; see that module's docstring for
# the "ineffective" bug it fixes and the new "effective_heterogeneous" state. This module
# passes its own EFFECTIVE_MIN_POSITIVE_SESSIONS constant and the CLI-supplied
# effective_mean_delta as explicit arguments into those shared functions below (generalization
# 2 in the module docstring) rather than keeping a local copy of the verdict logic.


# ------------------------------------------------------------------------------------
# Artifact loading and validation.
# ------------------------------------------------------------------------------------
def artifact_path(results_dir: Path, group: str, seed: int) -> Path:
    return results_dir / f"{group.lower()}_s{seed}.json"


def load_artifact(path: Path, *, group: str, seed: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing e3_tuning_ablation artifact: {path}")
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
    """Per-artifact self-consistency checks (protocol section 2). Unlike
    aggregate_side_feature_ablation_v2.py, the epoch window/total-epoch budget is NOT
    compared against a hardcoded constant here -- E3's epoch budget comes from E2 and is not
    known at the time this aggregator is written (E3_E4_ENCODER_PROGRAM.md section 0).
    Instead each artifact is checked for internal self-consistency (its own epoch_list
    agrees with its own protocol.epoch_window, which agrees with its own
    protocol.burn_in_epochs/total_epochs); validate_cross_artifact_consistency separately
    checks that every artifact in the aggregation used the SAME window, which is the actual
    invariant M2 requires.
    """
    protocol = payload.get("protocol", {})
    total_epochs = protocol.get("total_epochs")
    burn_in = protocol.get("burn_in_epochs")
    epoch_window = protocol.get("epoch_window")
    if not isinstance(total_epochs, int) or not isinstance(burn_in, int):
        raise ValueError(f"{path}: protocol.total_epochs/burn_in_epochs must be integers")
    expected_window = list(range(burn_in + 1, total_epochs + 1))
    if epoch_window != expected_window:
        raise ValueError(
            f"{path}: protocol.epoch_window {epoch_window} is not burn_in_epochs+1.."
            f"total_epochs (expected {expected_window} from its own recorded "
            f"total_epochs={total_epochs}, burn_in_epochs={burn_in})"
        )
    if payload.get("epoch_list") != epoch_window:
        raise ValueError(
            f"{path}: epoch_list must equal protocol.epoch_window, found {payload.get('epoch_list')}"
        )

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
    if sorted(int(key) for key in per_epoch) != epoch_window:
        raise ValueError(f"{path}: per_epoch keys must be exactly {epoch_window}")
    for epoch in epoch_window:
        epoch_sessions = set(per_epoch[str(epoch)].get("per_session_r2", {}))
        if epoch_sessions != val_sessions:
            raise ValueError(
                f"{path}: epoch {epoch} per_session_r2 sessions {sorted(epoch_sessions)} "
                f"!= validation sessions {sorted(val_sessions)}"
            )


def validate_training_provenance(payload: dict, *, group: str, path: Path) -> None:
    """Cross-check training.max_epochs / no_early_stopping / checkpoint_every_epoch AND the
    side_features.group contract against the sha256-pinned run_metadata.json the artifact
    references, rather than trusting the epoch-window JSON's own fields.

    training.max_epochs is checked against THIS ARTIFACT's own protocol.total_epochs (not a
    hardcoded constant): eval_epoch_window_generic_dandi688.py already asserts this equality
    once before it ever runs; this is a second, independent check against the
    sha256-pinned metadata file, mirroring what aggregate_side_feature_ablation_v2.py does
    for its frozen 12-epoch case.
    """
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
            "evaluation -- re-run eval_epoch_window_generic_dandi688.py)"
        )
    run_metadata = json.loads(run_metadata_path.read_text())
    training = run_metadata.get("training", {})
    side_features_meta = run_metadata.get("side_features") or {}
    expected_side_group = GROUP_CONTRACT[group]["side_features_group"]
    expected_total_epochs = payload["protocol"]["total_epochs"]
    checks = {
        "status": (run_metadata.get("status"), "completed"),
        "held_out_test_evaluated": (run_metadata.get("held_out_test_evaluated"), False),
        "training.max_epochs": (training.get("max_epochs"), expected_total_epochs),
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


def validate_cross_artifact_consistency(artifacts: Mapping[tuple[str, int], dict]) -> tuple[dict, dict]:
    """Cross-artifact checks: session split agreement, epoch-window/protocol agreement (the
    actual content of M2's "equal budget for every variant" requirement -- not a comparison
    against a hardcoded epoch count), and run-directory uniqueness (v3 bug H.4)."""
    reference_key = next(iter(artifacts))
    reference_splits = artifacts[reference_key]["session_splits"]
    reference_protocol = artifacts[reference_key]["protocol"]

    mismatched_splits = [
        key for key, payload in artifacts.items() if payload["session_splits"] != reference_splits
    ]
    if mismatched_splits:
        raise ValueError(
            f"session_splits disagree across artifacts (reference={reference_key}): {mismatched_splits}"
        )

    mismatched_protocol = [
        key for key, payload in artifacts.items() if payload["protocol"] != reference_protocol
    ]
    if mismatched_protocol:
        raise ValueError(
            "protocol (epoch budget/window) disagrees across artifacts -- M2 requires every "
            f"group/seed to train the identical epoch budget (reference={reference_key} "
            f"protocol={reference_protocol}): {mismatched_protocol}"
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
            f"Two or more e3_tuning_ablation runs share a run directory (this was v3 bug H.4): {duplicates}"
        )

    return reference_splits, reference_protocol


# ------------------------------------------------------------------------------------
# Per-run scoring helpers. Unlike aggregate_side_feature_ablation_v2.py, these read the
# epoch window from the ARTIFACT ITSELF (payload["epoch_list"]) rather than a hardcoded
# module constant -- every artifact entering run_aggregation is already cross-validated
# (validate_cross_artifact_consistency) to share the identical window, so any one artifact's
# own epoch_list is equivalent to "the" window.
# ------------------------------------------------------------------------------------
def per_run_session_scores(payload: dict) -> dict[str, float]:
    """Epoch-window mean R2 for each validation session, for one (group, seed) run."""
    sessions = sorted(payload["session_splits"]["val"])
    per_epoch = payload["per_epoch"]
    epoch_window = payload["epoch_list"]
    return {
        session: mean([per_epoch[str(epoch)]["per_session_r2"][session] for epoch in epoch_window])
        for session in sessions
    }


def per_run_within_window_std(payload: dict) -> float:
    """Within-window std (protocol section 3, item 2): std of this run's per-epoch mean R2
    values."""
    per_epoch_mean_r2 = payload["per_epoch_mean_r2"]
    epoch_window = payload["epoch_list"]
    values = [per_epoch_mean_r2[str(epoch)] for epoch in epoch_window]
    return sample_std(values)


# ------------------------------------------------------------------------------------
# Top-level aggregation.
# ------------------------------------------------------------------------------------
def run_aggregation(results_dir: Path, seeds: Sequence[int], effective_mean_delta: float) -> dict:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if len(seeds) < 1:
        raise ValueError("seeds must contain at least one seed")
    if effective_mean_delta <= 0:
        raise ValueError(f"--effective_mean_delta must be positive, got {effective_mean_delta}")

    artifacts: dict[tuple[str, int], dict] = {}
    missing: list[str] = []
    for group in GROUPS:
        for seed in seeds:
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
            "Missing e3_tuning_ablation artifact(s); cannot aggregate until all "
            f"{len(GROUPS)}x{len(seeds)} runs have completed training + "
            "eval_epoch_window_generic_dandi688.py evaluation: " + ", ".join(missing)
        )

    reference_splits, reference_protocol = validate_cross_artifact_consistency(artifacts)
    val_sessions = sorted(reference_splits["val"])
    epoch_window = reference_protocol["epoch_window"]

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
        group: {seed: artifacts[(group, seed)]["variant_score"] for seed in seeds}
        for group in GROUPS
    }
    variant_score_mean = {group: mean(list(variant_scores[group].values())) for group in GROUPS}
    across_seed_std = {
        group: sample_std(list(variant_scores[group].values())) for group in GROUPS
    }
    within_window_std_by_run = {
        f"{group}_s{seed}": within_window_std[(group, seed)] for group in GROUPS for seed in seeds
    }
    within_window_std_pooled_mean = mean(list(within_window_std_by_run.values()))

    paired_deltas: dict[str, dict] = {}
    for treatment, control in PAIRS:
        pair_name = f"{treatment}_minus_{control}"
        sigma_unpaired = sigma_delta_standard_error(
            across_seed_std[treatment], across_seed_std[control], len(seeds)
        )

        per_session_seed_mean: dict[str, float] = {}
        per_seed_values: dict[int, list[float]] = {seed: [] for seed in seeds}
        for session in val_sessions:
            session_deltas = []
            for seed in seeds:
                delta = session_scores[(treatment, seed)][session] - session_scores[(control, seed)][session]
                session_deltas.append(delta)
                per_seed_values[seed].append(delta)
            per_session_seed_mean[session] = mean(session_deltas)
        per_seed_mean = {seed: mean(values) for seed, values in per_seed_values.items()}
        per_seed_mean_values = list(per_seed_mean.values())

        # Primary estimator (2026-07-26 revision): paired directly on the same-seed deltas
        # rather than combined in quadrature from the two groups' independent across-seed SDs
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
            effective_mean_delta_threshold=effective_mean_delta,
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
            effective_mean_delta_threshold=effective_mean_delta,
        )
        verdict, decided_by = classify_pair_verdict(
            mean_delta=mean_delta,
            n_sessions_positive=n_positive,
            n_sessions_total=len(val_sessions),
            per_seed_means=per_seed_mean_values,
            sigma_delta_paired=sigma_paired,
            effective_mean_delta_threshold=effective_mean_delta,
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
            "effective_mean_delta_threshold": effective_mean_delta,
            "effective_min_positive_sessions": EFFECTIVE_MIN_POSITIVE_SESSIONS,
            "meets_effective_clause": meets_effective,
            "meets_effective_heterogeneous_clause": meets_effective_heterogeneous,
            "exceeds_ineffective_threshold": exceeds_ineffective,
            "verdict": verdict,
            "decided_by": decided_by,
        }

    tuning_group_verdicts: dict[str, dict] = {}
    for group, (pair_a_name, pair_b_name) in TUNING_GROUP_PAIRS.items():
        pair_results = {
            pair_a_name: (paired_deltas[pair_a_name]["verdict"], paired_deltas[pair_a_name]["decided_by"]),
            pair_b_name: (paired_deltas[pair_b_name]["verdict"], paired_deltas[pair_b_name]["decided_by"]),
        }
        verdict, decided_by = classify_group_verdict(pair_results=pair_results)
        tuning_group_verdicts[group] = {
            "content_pair_vs_f0": pair_a_name,
            "content_pair_vs_shuffled_control": pair_b_name,
            "verdict": verdict,
            "decided_by": decided_by,
        }

    return {
        "schema_version": 1,
        "purpose": "e3_tuning_ablation_measurement_protocol_v4",
        "protocol_docs": [
            "sua_exploration/docs/E3_E4_ENCODER_PROGRAM.md",
            "sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md",
        ],
        "screen_id": "e3_tuning_ablation",
        "no_formal_test_sessions_evaluated": True,
        "groups": list(GROUPS),
        "seeds": list(seeds),
        "group_contract": GROUP_CONTRACT,
        "session_splits": reference_splits,
        "fixed_protocol": EXPECTED_PROTOCOL,
        "epoch_window": epoch_window,
        "epoch_budget": reference_protocol.get("total_epochs"),
        "burn_in_epochs": reference_protocol.get("burn_in_epochs"),
        "consistency_validated": True,
        "effective_mean_delta_threshold": effective_mean_delta,
        "effective_mean_delta_threshold_rule": (
            "max(2*sigma_delta_measured, deployment_relevance_floor) -- "
            "E3_E4_ENCODER_PROGRAM.md section 0 pre-registers this RULE but leaves the "
            "numeric VALUE unset until E1 (sigma_seed) and E2 (epoch budget) are known. "
            "The value used in this aggregation is supplied by the caller via "
            "--effective_mean_delta; this script does not compute or guess it."
        ),
        "variant_scores": {
            group: {
                **{str(seed): variant_scores[group][seed] for seed in seeds},
                "mean": variant_score_mean[group],
            }
            for group in GROUPS
        },
        "uncertainty": {
            "definition": (
                "Measured directly from this screen's own artifacts per "
                "MEASUREMENT_PROTOCOL_V4.md section 3. None of these values reuse any "
                "prior screen's sigma estimates -- section 3 requires sigma_delta to be "
                "measured fresh on this round's own data."
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
                "mean' above), not from the two groups' independent across-seed SDs. "
                "SECONDARY, retained for comparison only and NOT gated on: "
                "sigma_delta_unpaired_quadrature = sqrt(across_seed_std[treatment]**2 + "
                "across_seed_std[control]**2) / sqrt(n_seeds), divided by sqrt(n_seeds) to "
                "convert the single-seed-delta SD into the standard error of the n_seeds-seed "
                "mean delta (M7 fix; aggregate_side_feature_ablation_v2."
                "sigma_delta_standard_error(), imported here unmodified rather than "
                "re-implemented). The quadrature form assumes the two groups' seed-level "
                "effects are statistically independent -- they are not, since both groups "
                "share the same seed list, so seed-level difficulty largely cancels in the "
                "paired difference. This made the quadrature estimate systematically too "
                "large (biasing the verdict toward 'indeterminate'); see "
                "implied_seed_correlation_per_pair for the measured inter-arm correlation "
                "that explains the gap between the two estimates (2026-07-26 revision, "
                "MEASUREMENT_PROTOCOL_V4.md section 4.1; aggregate_side_feature_ablation_v2."
                "sigma_delta_paired(), imported here unmodified rather than re-implemented)."
            ),
        },
        "paired_deltas": paired_deltas,
        "tuning_group_verdicts": tuning_group_verdicts,
        "source_artifacts": {
            f"{group}_s{seed}": str(artifact_path(results_dir, group, seed))
            for group in GROUPS
            for seed in seeds
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seeds",
        type=str,
        required=True,
        help=(
            "Comma-separated seed list this screen actually ran, e.g. 42,43,44. Required, "
            "no default: must match --seeds as passed to run_e3_tuning_ablation.sh. E1's "
            "measured sigma_seed sets this count (E3_E4_ENCODER_PROGRAM.md section 0)."
        ),
    )
    parser.add_argument(
        "--effective_mean_delta",
        type=float,
        required=True,
        help=(
            "The mean-delta bound a pair must clear (>=) to be judged 'effective' "
            "(MEASUREMENT_PROTOCOL_V4.md section 4.2's '+0.03'-style gate). Required, no "
            "default: E3_E4_ENCODER_PROGRAM.md section 0 pre-registers the RULE "
            "(max(2*sigma_delta_measured, deployment floor)) but the VALUE is only "
            "computable after E1 (sigma_seed) and E2 (epoch budget) are known. This "
            "script refuses to guess it."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Defaults to sua_exploration/results/e3_tuning_ablation",
    )
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)

    root = Path(__file__).resolve().parents[2]
    results_dir = args.results_dir or (root / "sua_exploration" / "results" / "e3_tuning_ablation")

    payload = run_aggregation(results_dir, seeds, args.effective_mean_delta)

    out_path = args.out_path or (results_dir / "aggregate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {group: data["verdict"] for group, data in payload["tuning_group_verdicts"].items()},
        indent=2, sort_keys=True,
    ))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
