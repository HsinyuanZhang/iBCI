"""Aggregate e4_encoder_variants SUA results under MEASUREMENT_PROTOCOL_V4.

Implements sua_exploration/docs/E3_E4_ENCODER_PROGRAM.md sections 2, 2.0 and 4 (encoder
architecture variants), pinned to MEASUREMENT_PROTOCOL_V4.md sections 2-4, over the
(group in {B3, B3T, B3A}) x (seed in --seeds) epoch-window artifacts that
``eval_epoch_window_generic_dandi688.py`` writes (M3 estimator, generalized to a
CLI-supplied epoch budget; NOT the frozen ``eval_epoch_window_dandi688.py``, since E4's
epoch budget comes from E2 and is not guaranteed to be the frozen script's hardcoded 12) as
``results/e4_encoder_variants/{group_lower}_s{seed}.json``.

Group -> variant contract (charter section 2; side_features is "none" for every group --
E4 is architecture-only, no side-feature manipulation):
    B3  -> EarlyPoolEncoder (baseline / control)
    B3T -> TemporalBasisEarlyPoolEncoder (fixed raised-cosine temporal basis; charter 2.1)
    B3A -> TrialAttentionEarlyPoolEncoder (learned attention over the M calibration trials
           per unit; charter 2.2)

Two paired comparisons: B3T_minus_B3, B3A_minus_B3. Unlike E3 (and unlike
aggregate_side_feature_ablation_v2.py's F1/F2), there is NO shuffled control here and no
AND/OR-of-two-pairs group verdict: "E4's control is B3 itself (same seed, same budget); no
permutation control is needed -- architecture variants have no 'content vs width' confound"
(charter section 4). Each variant is therefore judged directly on its single pair's own
four-state verdict from ``classify_pair_verdict`` (MEASUREMENT_PROTOCOL_V4.md section 4.2b,
2026-07-27 bug fix -- imported unmodified from aggregate_side_feature_ablation_v2, the single
shared implementation every aggregator in this repo uses; see that module's docstring for the
"ineffective" bug it fixes and the new "effective_heterogeneous" state).

Deployment cost (charter section 2.0, 2026-07-26 independently re-verified): B3T is cheaper
than B3 on every axis (params -31%, MAC -65%, same 16,384 B support state), but B3A's
support state is 30x B3's (491,520 B vs 16,384 B) because it cannot reduce the M calibration
trials to an O(1) running accumulator -- it must retain every trial's per-unit features
until finalize_identity, breaking B3's streaming accumulation property. "Even if B3A wins on
R2, that 30x state cost must be weighed alongside it; if the gain is not decisive, B3T ...
is the more deployment-aligned direction" (charter section 2.0). This aggregator therefore
always records each variant's ``support_state_bytes``/``mac_per_session`` (and the full
``EncoderCostProfile``) from the ENCODERS' OWN ``cost_profile()`` method -- not
hand-transcribed numbers -- so the R2 verdict is never read without the deployment cost
beside it. This requires importing the (CPU-only, architecture-only, no trained weights, no
teacher checkpoint, no GPU) encoder classes from streaming_calibration_exp; every other
aggregator in this repo is torch-free, this one is not, and only for this reason.

Two generalizations relative to aggregate_side_feature_ablation_v2.py, both because E4 is
launched before E1/E2 have supplied numbers that v2's screen already had pinned
(E3_E4_ENCODER_PROGRAM.md section 0): epoch budget/window is validated by internal
self-consistency plus cross-artifact agreement rather than a hardcoded constant, and seed
count / effective-mean-delta threshold are required CLI arguments (--seeds,
--effective_mean_delta) rather than hardcoded module constants. See
aggregate_e3_tuning_ablation.py's module docstring for the full rationale -- both
generalizations are implemented identically here.

sigma_delta_standard_error (M7 fix: the ``/ sqrt(n_seeds)`` term) is imported unmodified
from aggregate_side_feature_ablation_v2 rather than re-implemented, per
E3_E4_ENCODER_PROGRAM.md's instruction not to regress it.

Hard data-isolation constraint (protocol section 6 / charter section 3): the result
aggregation half of this script only ever reads the JSON artifacts
eval_epoch_window_generic_dandi688.py already produced and the run_metadata.json files they
reference; it never opens an NWB file and never reads spike/behavior/trial data. The
deployment-cost half never touches data at all -- it instantiates untrained encoder modules
from fixed shape constants and reads off their parameter/MAC/state accounting.
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
    classify_pair_verdict,
    implied_seed_correlation,
    pair_exceeds_ineffective_threshold,
    pair_meets_effective_clause,
    pair_meets_effective_heterogeneous_clause,
    sigma_delta_paired,
    sigma_delta_standard_error,
)

GROUPS: tuple[str, ...] = ("B3", "B3T", "B3A")

# group -> (variant, side_features.group) contract each artifact + its referenced
# run_metadata.json must satisfy (E3_E4_ENCODER_PROGRAM.md section 2: no side features).
GROUP_CONTRACT: dict[str, dict[str, str]] = {
    "B3": {"variant": "B3", "side_features_group": "none"},
    "B3T": {"variant": "B3T", "side_features_group": "none"},
    "B3A": {"variant": "B3A", "side_features_group": "none"},
}

# B3 is the shared control for both variants; no shuffled control (charter section 4).
PAIRS: tuple[tuple[str, str], ...] = (
    ("B3T", "B3"),
    ("B3A", "B3"),
)
# Which single pair feeds each variant's four-state verdict.
VARIANT_PAIR: dict[str, str] = {
    "B3T": "B3T_minus_B3",
    "B3A": "B3A_minus_B3",
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

# ------------------------------------------------------------------------------------
# Deployment cost profile (charter section 2.0). Fixed shape constants matching this
# repo's training configuration (train_variant_dandi688.py: window_size=50, trial_length=100,
# hidden_dim=64) and the charter's own N=64/T=100/M=30 evaluation point. These are
# architecture/shape constants, not statistical estimates -- unlike --max_epochs/--seeds/
# --effective_mean_delta, there is nothing here for E1/E2 to determine, so they are safe to
# fix in code (and are exactly what reproduces the charter section 2.0 table; see
# test_aggregate_e4_encoder_variants.py's regression test against those published numbers).
# ------------------------------------------------------------------------------------
COST_PROFILE_NUM_NEURONS = 64
COST_PROFILE_TRIAL_LENGTH = 100
COST_PROFILE_NUM_TRIALS = 30
COST_PROFILE_HIDDEN_DIM = 64
COST_PROFILE_WINDOW_SIZE = 50


def _encoder_classes() -> dict[str, type]:
    """Import the three E4 encoder classes from streaming_calibration_exp.

    Deferred to call time (rather than a module-level import) so that every other function
    in this file -- in particular the pure verdict-logic functions the unit tests exercise
    most heavily -- stays importable even in an environment where torch is unavailable.
    CPU-only: these classes are plain nn.Module subclasses with a handful of nn.Linear
    layers; nothing here ever calls .cuda() or loads a checkpoint.
    """
    sce_root = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
    if str(sce_root) not in sys.path:
        sys.path.insert(0, str(sce_root))
    from src.models.components.streaming_encoders import (  # noqa: PLC0415
        EarlyPoolEncoder,
        TemporalBasisEarlyPoolEncoder,
        TrialAttentionEarlyPoolEncoder,
    )

    return {
        "B3": EarlyPoolEncoder,
        "B3T": TemporalBasisEarlyPoolEncoder,
        "B3A": TrialAttentionEarlyPoolEncoder,
    }


def deployment_cost_profile(variant: str) -> dict:
    """This variant's ``EncoderCostProfile`` at the charter's N=64/T=100/M=30 evaluation
    point, read directly off a freshly constructed (untrained) encoder -- never hand-copied
    numbers. Parameter/MAC/state accounting depends only on the module's architecture
    (layer shapes), not on learned weight values, so an untrained encoder gives the exact
    same profile as a trained checkpoint of the same variant.
    """
    encoder_classes = _encoder_classes()
    if variant not in encoder_classes:
        raise ValueError(f"No cost profile available for variant {variant!r}")
    encoder = encoder_classes[variant](
        trial_length=COST_PROFILE_TRIAL_LENGTH,
        window_size=COST_PROFILE_WINDOW_SIZE,
        hidden_dim=COST_PROFILE_HIDDEN_DIM,
    )
    profile = encoder.cost_profile(
        num_neurons=COST_PROFILE_NUM_NEURONS,
        trial_length=COST_PROFILE_TRIAL_LENGTH,
        num_trials=COST_PROFILE_NUM_TRIALS,
    )
    return {
        "variant": profile.variant,
        "parameter_count": profile.parameter_count,
        "weight_bytes": profile.weight_bytes,
        "trial_buffer_bytes": profile.trial_buffer_bytes,
        "support_state_bytes": profile.support_state_bytes,
        "peak_live_state_bytes": profile.peak_live_state_bytes,
        "mac_per_trial": profile.mac_per_trial,
        "mac_per_session": profile.mac_per_session,
        "requires_cubic_interpolation": profile.requires_cubic_interpolation,
        "requires_general_multiplier": profile.requires_general_multiplier,
        "requires_divider": profile.requires_divider,
        "cost_source": profile.cost_source,
        "cost_profile_shape": {
            "num_neurons": COST_PROFILE_NUM_NEURONS,
            "trial_length": COST_PROFILE_TRIAL_LENGTH,
            "num_trials": COST_PROFILE_NUM_TRIALS,
        },
        "cost_profile_reference": "E3_E4_ENCODER_PROGRAM.md section 2.0",
    }


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


# classify_pair_verdict (section 4.2b's four-state gate, 2026-07-27 bug fix) is imported
# unmodified from aggregate_side_feature_ablation_v2 above -- the single shared implementation
# every aggregator in this repo uses; see that module's docstring for the "ineffective" bug it
# fixes and the new "effective_heterogeneous" state. E4 has no shuffled control to combine
# against (charter section 4) -- each variant's verdict below IS its single pair's own
# classify_pair_verdict result, with no group-level AND/OR step (unlike E3's T4/T8 or v2's
# F1/F2, which each combine two pairs via classify_group_verdict).


# ------------------------------------------------------------------------------------
# Artifact loading and validation.
# ------------------------------------------------------------------------------------
def artifact_path(results_dir: Path, group: str, seed: int) -> Path:
    return results_dir / f"{group.lower()}_s{seed}.json"


def load_artifact(path: Path, *, group: str, seed: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing e4_encoder_variants artifact: {path}")
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
    """Per-artifact self-consistency checks (protocol section 2); see
    aggregate_e3_tuning_ablation.validate_artifact_contract for the full rationale for why
    the epoch window/total-epoch budget is validated by internal self-consistency here
    rather than against a hardcoded constant.
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
    side_features.group=="none" contract against the sha256-pinned run_metadata.json the
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
    """Cross-artifact checks: session split agreement, epoch-window/protocol agreement, and
    run-directory uniqueness (v3 bug H.4)."""
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
            f"Two or more e4_encoder_variants runs share a run directory (this was v3 bug H.4): {duplicates}"
        )

    return reference_splits, reference_protocol


# ------------------------------------------------------------------------------------
# Per-run scoring helpers (identical in spirit to aggregate_e3_tuning_ablation.py's).
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
            "Missing e4_encoder_variants artifact(s); cannot aggregate until all "
            f"{len(GROUPS)}x{len(seeds)} runs have completed training + "
            "eval_epoch_window_generic_dandi688.py evaluation: " + ", ".join(missing)
        )

    reference_splits, reference_protocol = validate_cross_artifact_consistency(artifacts)
    val_sessions = sorted(reference_splits["val"])
    epoch_window = reference_protocol["epoch_window"]

    session_scores = {key: per_run_session_scores(payload) for key, payload in artifacts.items()}
    within_window_std = {key: per_run_within_window_std(payload) for key, payload in artifacts.items()}

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

    deployment_cost = {group: deployment_cost_profile(GROUP_CONTRACT[group]["variant"]) for group in GROUPS}

    variant_verdicts: dict[str, dict] = {}
    for variant, pair_name in VARIANT_PAIR.items():
        pair = paired_deltas[pair_name]
        variant_verdicts[variant] = {
            "pair": pair_name,
            "verdict": pair["verdict"],
            "decided_by": pair["decided_by"],
            # Charter section 2.0: the R2 verdict must never be read without the deployment
            # cost beside it (B3A's 30x support-state growth in particular).
            "deployment_cost": deployment_cost[variant],
            "deployment_cost_vs_control": {
                "control": "B3",
                "support_state_bytes_ratio": (
                    deployment_cost[variant]["support_state_bytes"]
                    / deployment_cost["B3"]["support_state_bytes"]
                ),
                "mac_per_session_ratio": (
                    deployment_cost[variant]["mac_per_session"] / deployment_cost["B3"]["mac_per_session"]
                ),
                "parameter_count_ratio": (
                    deployment_cost[variant]["parameter_count"] / deployment_cost["B3"]["parameter_count"]
                ),
            },
        }

    return {
        "schema_version": 1,
        "purpose": "e4_encoder_variants_measurement_protocol_v4",
        "protocol_docs": [
            "sua_exploration/docs/E3_E4_ENCODER_PROGRAM.md",
            "sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md",
        ],
        "screen_id": "e4_encoder_variants",
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
        "deployment_cost": deployment_cost,
        "variant_verdicts": variant_verdicts,
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
            "no default: must match --seeds as passed to run_e4_encoder_variants.sh. E1's "
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
        help="Defaults to sua_exploration/results/e4_encoder_variants",
    )
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)

    root = Path(__file__).resolve().parents[2]
    results_dir = args.results_dir or (root / "sua_exploration" / "results" / "e4_encoder_variants")

    payload = run_aggregation(results_dir, seeds, args.effective_mean_delta)

    out_path = args.out_path or (results_dir / "aggregate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            variant: {
                "verdict": data["verdict"],
                "support_state_bytes": data["deployment_cost"]["support_state_bytes"],
                "mac_per_session": data["deployment_cost"]["mac_per_session"],
            }
            for variant, data in payload["variant_verdicts"].items()
        },
        indent=2, sort_keys=True,
    ))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
