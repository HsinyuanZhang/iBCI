"""Aggregate attention_arch_screen_v4 SUA results under MEASUREMENT_PROTOCOL_V4.

Implements sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md sections 2-6 exactly, over the
12 (variant in {B3, B15P, B15D, B15}) x (seed in {42, 43, 44}) epoch-window artifacts that
``eval_epoch_window_dandi688.py`` writes (M3 estimator; not reimplemented or modified here):

1. Consistency validation (section 2) before any scoring: all 12 artifacts must agree on
   the train/val/test session split, the fixed protocol (first/n=30/pool=50), the epoch
   window (5..12), and must each report ``max_epochs==12``, ``no_early_stopping==true``,
   ``checkpoint_every_epoch==true`` (cross-checked against the sha256-pinned
   run_metadata.json each artifact references, not just its own summary fields) and
   ``no_test_files_evaluated==true``. Also rejects if any two runs resolve to the same run
   directory (v3 bug H.4).
2. Paired deltas for (B15-B3, B15-B15P, B15-B15D): per-session deltas averaged over the 3
   seeds -> exactly 6 values per pair, keyed ``per_session_seed_mean`` (matching the v3
   aggregate's naming in results/attention_arch_screen_v3/aggregate.json).
3. Uncertainty measured from THIS run's 12 artifacts only (section 3) -- the prior
   sigma_epoch=0.0388 / sigma_delta=0.0112 estimates in the protocol doc are never reused:
   within-window std across the 8 epochs per run, across-seed std of the variant score per
   variant, and a combined sigma_delta per pair (quadrature of the two variants' own
   measured across-seed std).
4. Four-state verdict per pair (section 4.2b, 2026-07-27 bug fix): the literal strings
   "effective" / "effective_heterogeneous" / "ineffective" / "indeterminate", computed by
   ``classify_pair_verdict``, imported unmodified from aggregate_side_feature_ablation_v2 (the
   single shared implementation every aggregator in this repo uses -- see that module's
   docstring for the bug the 2026-07-27 revision fixes). "indeterminate" is never collapsed
   into a boolean false -- conflating those two is exactly the error that voided
   attention_arch_screen_v3 (section 4.3; CURRENT_RESULTS.md section H).
5. Each variant's absolute score (mean over seeds) and per-seed scores are also recorded.

Hard data-isolation constraint (protocol section 6): this script only ever reads the JSON
artifacts already produced by eval_epoch_window_dandi688.py and the run_metadata.json files
they reference. It never opens an NWB file and never reads spike/behavior/trial data, so it
cannot violate the six-test-session isolation rule regardless of what session names appear
in session_splits.test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_side_feature_ablation_v2 import (  # noqa: E402
    VALID_VERDICTS,
    VERDICT_EFFECTIVE,
    VERDICT_EFFECTIVE_HETEROGENEOUS,
    VERDICT_INDETERMINATE,
    VERDICT_INEFFECTIVE,
    classify_pair_verdict,
    implied_seed_correlation,
    sigma_delta_paired,
    sigma_delta_standard_error,
)

VARIANTS: tuple[str, ...] = ("B3", "B15P", "B15D", "B15")
SEEDS: tuple[int, ...] = (42, 43, 44)
CONTROLS: tuple[str, ...] = ("B3", "B15P", "B15D")
TREATMENT = "B15"
PAIRS: tuple[tuple[str, str], ...] = tuple((TREATMENT, control) for control in CONTROLS)

EXPECTED_EPOCH_WINDOW = [5, 6, 7, 8, 9, 10, 11, 12]
EXPECTED_PROTOCOL = {"selection_mode": "first", "calibration_n": 30, "pool_size": 50}
EXPECTED_TOTAL_EPOCHS = 12
EXPECTED_SESSION_TOTAL = 6

# Protocol section 4.2 gate thresholds (verbatim; do not tune after the fact).
EFFECTIVE_MEAN_DELTA_THRESHOLD = 0.03
EFFECTIVE_MIN_POSITIVE_SESSIONS = 5
INEFFECTIVE_SIGMA_MULTIPLE = 2.0


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


# sigma_delta_standard_error / sigma_delta_paired / implied_seed_correlation / VERDICT_* /
# classify_pair_verdict are all imported from aggregate_side_feature_ablation_v2 above (single
# shared implementation -- this module used to keep its own copy-pasted three-state
# classify_pair_verdict, which is exactly the kind of divergent duplicate the 2026-07-26
# sigma_delta_paired fix, and now the 2026-07-27 four-state classify_pair_verdict fix, must
# not repeat: see that module's docstring for the "ineffective" bug it fixes and the new
# "effective_heterogeneous" state). This module's own EFFECTIVE_MEAN_DELTA_THRESHOLD /
# EFFECTIVE_MIN_POSITIVE_SESSIONS constants above are still local (they are this screen's own
# frozen values, verbatim from the protocol) but are passed as explicit arguments into the
# shared classify_pair_verdict below, never hardcoded inside it.


# --------------------------------------------------------------------------------------
# Artifact loading and validation (section 2).
# --------------------------------------------------------------------------------------
def artifact_path(results_dir: Path, variant: str, seed: int) -> Path:
    return results_dir / f"epoch_window_{variant.lower()}_s{seed}.json"


def load_artifact(path: Path, *, variant: str, seed: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing v4 epoch-window artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("variant") != variant:
        raise ValueError(
            f"{path}: variant mismatch, expected {variant!r}, found {payload.get('variant')!r}"
        )
    if payload.get("seed") != seed:
        raise ValueError(
            f"{path}: seed mismatch, expected {seed!r}, found {payload.get('seed')!r}"
        )
    return payload


def validate_artifact_contract(payload: dict, *, path: Path) -> None:
    """Per-artifact checks against the fixed protocol constants (section 2)."""
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


def validate_training_provenance(payload: dict, *, path: Path) -> None:
    """Cross-check training.max_epochs / no_early_stopping / checkpoint_every_epoch against
    the sha256-pinned run_metadata.json the artifact references, rather than trusting the
    epoch-window JSON's own (much thinner) summary of that run."""
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
    checks = {
        "status": (run_metadata.get("status"), "completed"),
        "held_out_test_evaluated": (run_metadata.get("held_out_test_evaluated"), False),
        "training.max_epochs": (training.get("max_epochs"), EXPECTED_TOTAL_EPOCHS),
        "training.no_early_stopping": (training.get("no_early_stopping"), True),
        "training.checkpoint_every_epoch": (training.get("checkpoint_every_epoch"), True),
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
    """Cross-artifact checks that cannot be expressed as "matches a fixed constant":
    session split agreement and run-directory uniqueness (v3 bug H.4)."""
    reference_key = next(iter(artifacts))
    reference_splits = artifacts[reference_key]["session_splits"]
    mismatched = [
        key for key, payload in artifacts.items() if payload["session_splits"] != reference_splits
    ]
    if mismatched:
        raise ValueError(
            f"session_splits disagree across v4 artifacts (reference={reference_key}): {mismatched}"
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
            f"Two or more v4 runs share a run directory (this was v3 bug H.4): {duplicates}"
        )

    return reference_splits


# --------------------------------------------------------------------------------------
# Per-run scoring helpers.
# --------------------------------------------------------------------------------------
def per_run_session_scores(payload: dict) -> dict[str, float]:
    """8-epoch-window mean R2 for each validation session, for one (variant, seed) run.

    Averaging epochs 5..12 first per session, then averaging the 6 resulting per-session
    scores, is mathematically identical (by linearity) to the recorded ``variant_score``
    (mean over sessions of the per-epoch mean, then mean over epochs) -- this is checked
    explicitly in ``run_aggregation`` below as an artifact-integrity sanity check.
    """
    sessions = sorted(payload["session_splits"]["val"])
    per_epoch = payload["per_epoch"]
    return {
        session: mean([per_epoch[str(epoch)]["per_session_r2"][session] for epoch in EXPECTED_EPOCH_WINDOW])
        for session in sessions
    }


def per_run_within_window_std(payload: dict) -> float:
    """Within-window std (section 3, item 2): std of this run's 8 per-epoch mean R2 values."""
    per_epoch_mean_r2 = payload["per_epoch_mean_r2"]
    values = [per_epoch_mean_r2[str(epoch)] for epoch in EXPECTED_EPOCH_WINDOW]
    return sample_std(values)


# --------------------------------------------------------------------------------------
# Top-level aggregation (I/O-adjacent but returns a plain dict; main() just writes it).
# --------------------------------------------------------------------------------------
def run_aggregation(results_dir: Path) -> dict:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    artifacts: dict[tuple[str, int], dict] = {}
    missing: list[str] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = artifact_path(results_dir, variant, seed)
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = load_artifact(path, variant=variant, seed=seed)
            validate_artifact_contract(payload, path=path)
            validate_training_provenance(payload, path=path)
            artifacts[(variant, seed)] = payload
    if missing:
        raise FileNotFoundError(
            "Missing v4 epoch-window artifact(s); cannot aggregate until all 12 runs have "
            "completed training + eval_epoch_window_dandi688.py evaluation: " + ", ".join(missing)
        )

    reference_splits = validate_cross_artifact_consistency(artifacts)
    val_sessions = sorted(reference_splits["val"])

    session_scores = {key: per_run_session_scores(payload) for key, payload in artifacts.items()}
    within_window_std = {key: per_run_within_window_std(payload) for key, payload in artifacts.items()}

    # Integrity check: the per-session-then-mean-over-sessions score must reproduce the
    # artifact's own recorded variant_score (mean-over-epochs of mean-over-sessions); if
    # not, either this script or the upstream artifact has a bug -- fail loudly rather than
    # silently score from a self-inconsistent artifact.
    for key, payload in artifacts.items():
        recomputed = mean(list(session_scores[key].values()))
        recorded = payload["variant_score"]
        if abs(recomputed - recorded) > 1e-9:
            raise ValueError(
                f"{key}: recomputed variant_score {recomputed!r} does not match recorded "
                f"{recorded!r} in {artifact_path(results_dir, *key)}"
            )

    variant_scores = {
        variant: {seed: artifacts[(variant, seed)]["variant_score"] for seed in SEEDS}
        for variant in VARIANTS
    }
    variant_score_mean = {variant: mean(list(variant_scores[variant].values())) for variant in VARIANTS}
    across_seed_std = {
        variant: sample_std(list(variant_scores[variant].values())) for variant in VARIANTS
    }
    within_window_std_by_run = {
        f"{variant}_s{seed}": within_window_std[(variant, seed)] for variant in VARIANTS for seed in SEEDS
    }
    within_window_std_pooled_mean = mean(list(within_window_std_by_run.values()))

    paired_deltas: dict[str, dict] = {}
    sigma_delta_paired_per_pair: dict[str, float] = {}
    sigma_delta_unpaired_quadrature_per_pair: dict[str, float] = {}
    implied_seed_correlation_per_pair: dict[str, float] = {}
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
        # rather than combined in quadrature from the two variants' independent across-seed
        # SDs (sigma_delta_paired's docstring has the measured evidence for why this matters).
        sigma_paired = sigma_delta_paired(per_seed_mean_values)
        seed_correlation = implied_seed_correlation(
            sigma_a=across_seed_std[treatment],
            sigma_b=across_seed_std[control],
            per_seed_deltas=per_seed_mean_values,
        )
        sigma_delta_paired_per_pair[pair_name] = sigma_paired
        sigma_delta_unpaired_quadrature_per_pair[pair_name] = sigma_unpaired
        implied_seed_correlation_per_pair[pair_name] = seed_correlation

        mean_delta = mean(list(per_session_seed_mean.values()))
        mean_delta_alt = mean(per_seed_mean_values)
        assert abs(mean_delta - mean_delta_alt) < 1e-9, (
            "grand mean must agree regardless of marginalization order "
            f"({mean_delta!r} vs {mean_delta_alt!r})"
        )

        n_positive = sum(1 for value in per_session_seed_mean.values() if value > 0.0)
        # classify_pair_verdict's sigma_delta_paired is gated on the PAIRED estimate (primary
        # as of the 2026-07-26 revision) -- NOT the unpaired quadrature, which is retained
        # only for comparison. 2026-07-27: four-state verdict (see aggregate_side_feature_
        # ablation_v2.classify_pair_verdict's docstring for the ineffective-clause bug fix).
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
            "verdict": verdict,
            "decided_by": decided_by,
        }

    return {
        "schema_version": 1,
        "purpose": "attention_architecture_screen_v4_measurement_protocol_v4",
        "protocol_doc": "sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md",
        "screen_id": "attention_arch_screen_v4",
        "no_formal_test_sessions_evaluated": True,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "session_splits": reference_splits,
        "fixed_protocol": EXPECTED_PROTOCOL,
        "epoch_window": EXPECTED_EPOCH_WINDOW,
        "consistency_validated": True,
        "variant_scores": {
            variant: {
                **{str(seed): variant_scores[variant][seed] for seed in SEEDS},
                "mean": variant_score_mean[variant],
            }
            for variant in VARIANTS
        },
        "uncertainty": {
            "definition": (
                "Measured directly from this screen's 12 artifacts per "
                "MEASUREMENT_PROTOCOL_V4.md section 3. None of these values reuse the "
                "doc's prior sigma_epoch=0.0388 / sigma_delta=0.0112 estimates (section 4.1"
                ", which were measured on attention_arch_screen_v3's best-checkpoint "
                "trajectories and are explicitly not to be carried over, section 3)."
            ),
            "within_window_std_per_run": within_window_std_by_run,
            "within_window_std_pooled_mean": within_window_std_pooled_mean,
            "across_seed_std_per_variant": across_seed_std,
            "sigma_delta_paired_per_pair": sigma_delta_paired_per_pair,
            "sigma_delta_unpaired_quadrature_per_pair": sigma_delta_unpaired_quadrature_per_pair,
            "implied_seed_correlation_per_pair": implied_seed_correlation_per_pair,
            "sigma_delta_method": (
                "PRIMARY, used by the ineffective clause / three-state verdict: "
                "sigma_delta_paired = stdev(per_seed_mean_deltas, ddof=1) / sqrt(n_seeds), "
                "computed directly from the same-seed-paired per-seed mean deltas ('per_seed_"
                "mean' above), not from the two variants' independent across-seed SDs. "
                "SECONDARY, retained for comparison only and NOT gated on: "
                "sigma_delta_unpaired_quadrature = sqrt(across_seed_std[treatment]**2 + "
                "across_seed_std[control]**2) / sqrt(n_seeds) (the M7-fixed quadrature "
                "combination; see sigma_delta_standard_error()). The quadrature form assumes "
                "the two variants' seed-level effects are statistically independent -- they "
                "are not, since both variants share the same seed list, so seed-level "
                "difficulty largely cancels in the paired difference. This made the "
                "quadrature estimate systematically too large (biasing the verdict toward "
                "'indeterminate'); see implied_seed_correlation_per_pair for the measured "
                "inter-arm correlation that explains the gap between the two estimates "
                "(2026-07-26 revision, MEASUREMENT_PROTOCOL_V4.md section 4.1; see "
                "sigma_delta_paired())."
            ),
        },
        "paired_deltas": paired_deltas,
        "source_artifacts": {
            f"{variant}_s{seed}": str(artifact_path(results_dir, variant, seed))
            for variant in VARIANTS
            for seed in SEEDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Defaults to sua_exploration/results/attention_arch_screen_v4",
    )
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    results_dir = args.results_dir or (root / "sua_exploration" / "results" / "attention_arch_screen_v4")

    payload = run_aggregation(results_dir)

    out_path = args.out_path or (results_dir / "aggregate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({pair: data["verdict"] for pair, data in payload["paired_deltas"].items()}, indent=2, sort_keys=True))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
