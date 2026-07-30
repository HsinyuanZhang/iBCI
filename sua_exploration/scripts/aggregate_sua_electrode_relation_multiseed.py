"""Strict multi-seed decision aggregator for the SUA same-electrode relation screen.

Inputs are the single-seed strict aggregates emitted by
``aggregate_sua_electrode_relation_pilot.py``. This script does not trust those
summaries alone: it reopens every referenced epoch-window JSON and run metadata,
re-runs the single-arm contract checks, and verifies that the per-session values
match before computing the three pre-registered relation contrasts:

    REL - T4
    REL - REL-MS
    REL - REL-NG

Only validation artifacts are read. No NWB file is discovered or opened.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_side_feature_ablation_v2 import (  # noqa: E402
    EFFECTIVE_MIN_POSITIVE_SESSIONS,
    VERDICT_EFFECTIVE,
    VERDICT_EFFECTIVE_HETEROGENEOUS,
    VERDICT_INDETERMINATE,
    VERDICT_INEFFECTIVE,
    classify_group_verdict,
    classify_pair_verdict,
    pair_exceeds_ineffective_threshold,
    pair_meets_effective_clause,
    pair_meets_effective_heterogeneous_clause,
    sigma_delta_paired,
    sigma_delta_standard_error,
)
from aggregate_sua_electrode_relation_pilot import (  # noqa: E402
    EXPECTED,
    EXPECTED_EPOCHS,
    EXPECTED_SPLIT,
    read_json,
    validate_arm,
)


PAIRS: tuple[tuple[str, str, str], ...] = (
    ("relation", "t4", "REL_minus_T4"),
    ("relation", "membership_shuffle", "REL_minus_REL_MS"),
    ("relation", "no_group", "REL_minus_REL_NG"),
)
EXPECTED_SEEDS = (42, 43, 44)
EFFECTIVE_MEAN_DELTA = 0.03
EXPECTED_SESSION_TOTAL = 6


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError(f"sample_std requires at least two values, got {len(values)}")
    return float(statistics.stdev(values))


def _assert_nested_scores_equal(
    *,
    path: Path,
    observed: Mapping[str, Mapping[str, float]],
    expected: Mapping[str, Mapping[str, float]],
) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"{path}: arm set in aggregate does not match referenced artifacts")
    for arm in expected:
        if set(observed[arm]) != set(expected[arm]):
            raise ValueError(f"{path}: validation session set drifted for arm {arm}")
        for session, expected_value in expected[arm].items():
            observed_value = float(observed[arm][session])
            if abs(observed_value - expected_value) > 1e-12:
                raise ValueError(
                    f"{path}: stale/tampered score for {arm}/{session}: "
                    f"aggregate={observed_value}, artifact={expected_value}"
                )


def load_and_revalidate_seed_aggregate(
    path: Path,
) -> tuple[int, dict[str, dict[str, float]], str, dict]:
    path = path.expanduser().resolve()
    payload = read_json(path)
    if payload.get("purpose") not in {
        "strict_single_seed_sua_electrode_relation_pilot",
        # The seed-42 queue began before the generic-purpose text was patched.
        "strict_seed42_sua_electrode_relation_pilot",
    }:
        raise ValueError(f"{path}: unexpected single-seed aggregate purpose")
    contract = payload.get("aggregate_contract") or {}
    seed = contract.get("seed")
    if not isinstance(seed, int):
        raise ValueError(f"{path}: aggregate_contract.seed must be an integer")
    if contract.get("arms") != list(EXPECTED):
        raise ValueError(f"{path}: aggregate_contract.arms must be {list(EXPECTED)!r}")
    if contract.get("split_counts") != EXPECTED_SPLIT:
        raise ValueError(f"{path}: split_counts contract drifted")
    if contract.get("signal_view") != "sua":
        raise ValueError(f"{path}: signal_view must be sua")
    if contract.get("training_activity_calibration_n") != 10:
        raise ValueError(f"{path}: training activity calibration must be 10")
    if contract.get("evaluation_forward_calibration_n") != 30:
        raise ValueError(f"{path}: evaluation forward calibration must be 30")
    if contract.get("pool_size") != 50:
        raise ValueError(f"{path}: T4 pool size must be 50")
    if contract.get("epoch_window") != EXPECTED_EPOCHS:
        raise ValueError(f"{path}: epoch window must be {EXPECTED_EPOCHS}")
    if contract.get("formal_test_evaluated") is not False:
        raise ValueError(f"{path}: formal_test_evaluated must be false")

    provenance = payload.get("provenance") or {}
    if set(provenance) != set(EXPECTED):
        raise ValueError(f"{path}: provenance must contain exactly the four frozen arms")
    records: dict[str, dict[str, float]] = {}
    hashes: set[str] = set()
    for arm in EXPECTED:
        artifact_text = provenance[arm].get("artifact")
        if not isinstance(artifact_text, str):
            raise ValueError(f"{path}: missing artifact provenance for {arm}")
        artifact_path = Path(artifact_text).expanduser().resolve()
        artifact_payload, scores = validate_arm(
            artifact_path, arm, expected_seed=seed
        )
        records[arm] = scores
        artifact_hash = artifact_payload.get("train_val_manifest_sha256")
        if artifact_hash != provenance[arm].get("artifact_manifest_sha256"):
            raise ValueError(f"{path}: manifest hash provenance drifted for {arm}")
        hashes.add(artifact_hash)
    if len(hashes) != 1:
        raise ValueError(f"{path}: all four arms must share one manifest hash")

    summarized = payload.get("per_arm_per_session_epoch_window_mean_r2") or {}
    _assert_nested_scores_equal(path=path, observed=summarized, expected=records)
    return seed, records, next(iter(hashes)), {
        "single_seed_aggregate": str(path),
        "raw_artifacts": {
            arm: provenance[arm]["artifact"]
            for arm in EXPECTED
        },
    }


def compute_multiseed_decision(
    records_by_seed: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    effective_mean_delta: float = EFFECTIVE_MEAN_DELTA,
) -> dict:
    seeds = sorted(records_by_seed)
    if len(seeds) < 2:
        raise ValueError("multi-seed decision requires at least two seeds")
    if effective_mean_delta <= 0:
        raise ValueError("effective_mean_delta must be positive")

    reference_sessions = set(records_by_seed[seeds[0]]["t4"])
    if len(reference_sessions) != EXPECTED_SESSION_TOTAL:
        raise ValueError(
            f"expected {EXPECTED_SESSION_TOTAL} validation sessions, "
            f"found {len(reference_sessions)}"
        )
    for seed in seeds:
        if set(records_by_seed[seed]) != set(EXPECTED):
            raise ValueError(f"seed {seed}: arm set drifted")
        for arm in EXPECTED:
            if set(records_by_seed[seed][arm]) != reference_sessions:
                raise ValueError(f"seed {seed}/{arm}: validation session set drifted")

    arm_scores = {
        arm: {
            seed: mean(list(records_by_seed[seed][arm].values()))
            for seed in seeds
        }
        for arm in EXPECTED
    }
    arm_score_means = {
        arm: mean(list(per_seed.values()))
        for arm, per_seed in arm_scores.items()
    }
    arm_score_stds = {
        arm: sample_std(list(per_seed.values()))
        for arm, per_seed in arm_scores.items()
    }

    paired_deltas: dict[str, dict] = {}
    pair_verdict_inputs: dict[str, tuple[str, str]] = {}
    for treatment, control, pair_name in PAIRS:
        per_seed_mean: dict[int, float] = {}
        per_session_seed_mean: dict[str, float] = {}
        for seed in seeds:
            per_seed_mean[seed] = mean(
                [
                    records_by_seed[seed][treatment][session]
                    - records_by_seed[seed][control][session]
                    for session in sorted(reference_sessions)
                ]
            )
        for session in sorted(reference_sessions):
            per_session_seed_mean[session] = mean(
                [
                    records_by_seed[seed][treatment][session]
                    - records_by_seed[seed][control][session]
                    for seed in seeds
                ]
            )

        per_seed_values = list(per_seed_mean.values())
        mean_delta = mean(per_seed_values)
        if abs(mean_delta - mean(list(per_session_seed_mean.values()))) > 1e-12:
            raise AssertionError(f"{pair_name}: seed/session marginal means disagree")
        paired_se = sigma_delta_paired(per_seed_values)
        unpaired_se = sigma_delta_standard_error(
            arm_score_stds[treatment], arm_score_stds[control], len(seeds)
        )
        n_positive = sum(value > 0.0 for value in per_session_seed_mean.values())
        meets_effective = pair_meets_effective_clause(
            mean_delta=mean_delta,
            n_sessions_positive=n_positive,
            n_sessions_total=len(reference_sessions),
            per_seed_means=per_seed_values,
            effective_mean_delta_threshold=effective_mean_delta,
            effective_min_positive_sessions=EFFECTIVE_MIN_POSITIVE_SESSIONS,
        )
        meets_heterogeneous = pair_meets_effective_heterogeneous_clause(
            mean_delta=mean_delta,
            sigma_delta_paired=paired_se,
            per_seed_means=per_seed_values,
            n_sessions_positive=n_positive,
            effective_min_positive_sessions=EFFECTIVE_MIN_POSITIVE_SESSIONS,
        )
        excludes_threshold = pair_exceeds_ineffective_threshold(
            mean_delta=mean_delta,
            sigma_delta_paired=paired_se,
            effective_mean_delta_threshold=effective_mean_delta,
        )
        verdict, decided_by = classify_pair_verdict(
            mean_delta=mean_delta,
            n_sessions_positive=n_positive,
            n_sessions_total=len(reference_sessions),
            per_seed_means=per_seed_values,
            sigma_delta_paired=paired_se,
            effective_mean_delta_threshold=effective_mean_delta,
            effective_min_positive_sessions=EFFECTIVE_MIN_POSITIVE_SESSIONS,
        )
        paired_deltas[pair_name] = {
            "treatment": treatment,
            "control": control,
            "per_seed_mean_delta_r2": {
                str(seed): value for seed, value in per_seed_mean.items()
            },
            "per_session_seed_mean_delta_r2": per_session_seed_mean,
            "mean_delta_r2": mean_delta,
            "paired_se_across_seed_means": paired_se,
            "two_se_interval": [
                mean_delta - 2.0 * paired_se,
                mean_delta + 2.0 * paired_se,
            ],
            "unpaired_quadrature_se": unpaired_se,
            "positive_seed_count": sum(value > 0.0 for value in per_seed_values),
            "positive_session_count": n_positive,
            "session_count": len(reference_sessions),
            "effective_mean_delta_threshold": effective_mean_delta,
            "meets_effective_clause": meets_effective,
            "meets_effective_heterogeneous_clause": meets_heterogeneous,
            "confidently_excludes_effective_threshold": excludes_threshold,
            "verdict": verdict,
            "decided_by": decided_by,
        }
        pair_verdict_inputs[pair_name] = (verdict, decided_by)

    group_verdict, group_decided_by = classify_group_verdict(
        pair_results=pair_verdict_inputs
    )
    if group_verdict == VERDICT_EFFECTIVE:
        next_action = (
            "advance_to_preregistered_group_relative_amplitude_stage; keep the "
            "same real/shuffled/no-group controls and validation-only boundary"
        )
    elif group_verdict == VERDICT_INDETERMINATE:
        next_action = (
            "if more precision is scientifically required, run the complete four-arm "
            "matrix for seeds 45,46,47; never add seeds to selected arms only"
        )
    elif group_verdict == VERDICT_EFFECTIVE_HETEROGENEOUS:
        next_action = (
            "do not advance to amplitude yet; report the replicated but "
            "session-heterogeneous relation effect"
        )
    elif group_verdict == VERDICT_INEFFECTIVE:
        next_action = (
            "stop the same-electrode relation route and do not launch the "
            "relative-amplitude stage"
        )
    else:  # pragma: no cover - shared classifier guarantees the four states.
        raise AssertionError(f"unexpected group verdict: {group_verdict}")

    return {
        "seeds": seeds,
        "validation_sessions": sorted(reference_sessions),
        "arm_scores": {
            arm: {
                "per_seed": {
                    str(seed): value for seed, value in per_seed.items()
                },
                "mean": arm_score_means[arm],
                "across_seed_sample_std": arm_score_stds[arm],
            }
            for arm, per_seed in arm_scores.items()
        },
        "paired_deltas": paired_deltas,
        "relation_group_verdict": {
            "verdict": group_verdict,
            "decided_by": group_decided_by,
            "advance_to_relative_amplitude": group_verdict == VERDICT_EFFECTIVE,
            "next_action": next_action,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-aggregate",
        action="append",
        required=True,
        type=Path,
        help="Repeat once per seed; expected strict single-seed aggregate JSON.",
    )
    parser.add_argument(
        "--expected-seeds",
        default=",".join(str(seed) for seed in EXPECTED_SEEDS),
    )
    parser.add_argument(
        "--effective-mean-delta",
        type=float,
        default=EFFECTIVE_MEAN_DELTA,
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    expected_seeds = tuple(
        int(part.strip())
        for part in args.expected_seeds.split(",")
        if part.strip()
    )
    if len(expected_seeds) < 2 or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("--expected-seeds must contain at least two unique integers")

    records_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    provenance: dict[str, dict] = {}
    manifest_hashes: set[str] = set()
    for path in args.seed_aggregate:
        seed, records, manifest_hash, seed_provenance = (
            load_and_revalidate_seed_aggregate(path)
        )
        if seed in records_by_seed:
            raise ValueError(f"duplicate seed aggregate: {seed}")
        records_by_seed[seed] = records
        provenance[str(seed)] = seed_provenance
        manifest_hashes.add(manifest_hash)
    if tuple(sorted(records_by_seed)) != tuple(sorted(expected_seeds)):
        raise ValueError(
            f"expected seeds {sorted(expected_seeds)}, found "
            f"{sorted(records_by_seed)}"
        )
    if len(manifest_hashes) != 1:
        raise ValueError("all seeds/arms must use the same strict manifest SHA-256")

    decision = compute_multiseed_decision(
        records_by_seed,
        effective_mean_delta=args.effective_mean_delta,
    )
    output = {
        "schema_version": 1,
        "purpose": "strict_multiseed_sua_same_electrode_relation_decision",
        "created_at": datetime.now().astimezone().isoformat(),
        "no_formal_test_sessions_evaluated": True,
        "strict_manifest_sha256": next(iter(manifest_hashes)),
        "contract": {
            "arms": list(EXPECTED),
            "split_counts": EXPECTED_SPLIT,
            "training_activity_calibration_n": 10,
            "evaluation_forward_calibration_n": 30,
            "pool_size": 50,
            "epoch_window": EXPECTED_EPOCHS,
            "effective_mean_delta_threshold": args.effective_mean_delta,
            "effective_min_positive_sessions": EFFECTIVE_MIN_POSITIVE_SESSIONS,
        },
        "provenance": provenance,
        **decision,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Wrote strict multi-seed relation decision "
        f"({output['relation_group_verdict']['verdict']}): {args.out}"
    )


if __name__ == "__main__":
    main()
