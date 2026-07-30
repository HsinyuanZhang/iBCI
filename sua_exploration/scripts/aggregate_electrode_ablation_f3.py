#!/usr/bin/env python3
"""Aggregate electrode_ablation_f3 results (F3/FS3) under MEASUREMENT_PROTOCOL_V4.

F0 baseline artifacts are reused from ``results/e3_tuning_ablation/f0_s{seed}.json`` (same
B3/no-side-features baseline as E3). This screen does NOT re-run F1/F2.

Pairs (dimension-matched, UNIT_SIDE_FEATURE_ABLATION.md section 6):
    F3_minus_F0, F3_minus_FS3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_e3_tuning_ablation import (  # noqa: E402
    EFFECTIVE_MIN_POSITIVE_SESSIONS,
    EXPECTED_PROTOCOL,
    INEFFECTIVE_SIGMA_MULTIPLE,
    mean,
    per_run_session_scores,
    per_run_within_window_std,
    sample_std,
    validate_artifact_contract,
    validate_cross_artifact_consistency,
)
from aggregate_side_feature_ablation_v2 import (  # noqa: E402
    classify_group_verdict,
    classify_pair_verdict,
    implied_seed_correlation,
    pair_exceeds_ineffective_threshold,
    pair_meets_effective_clause,
    pair_meets_effective_heterogeneous_clause,
    sigma_delta_paired,
    sigma_delta_standard_error,
)
from dandi688_gradient_free_protocol import sha256_file  # noqa: E402

SCREEN_GROUPS: tuple[str, ...] = ("F3", "FS3")
BASELINE_GROUP = "F0"
PAIRS: tuple[tuple[str, str], ...] = (("F3", "F0"), ("F3", "FS3"))
FEATURE_GROUP_PAIRS: dict[str, tuple[str, str]] = {
    "F3": ("F3_minus_F0", "F3_minus_FS3"),
}

GROUP_CONTRACT: dict[str, dict[str, str]] = {
    "F0": {"variant": "B3", "side_features_group": "none"},
    "F3": {"variant": "B3S", "side_features_group": "f3"},
    "FS3": {"variant": "B3S", "side_features_group": "fs3"},
}


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds:
        raise ValueError("--seeds must list at least one integer seed")
    return seeds


def artifact_path(results_dir: Path, group: str, seed: int) -> Path:
    return results_dir / f"{group.lower()}_s{seed}.json"


def load_artifact(path: Path, *, group: str, seed: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact: {path}")
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


def validate_training_provenance(payload: dict, *, group: str, path: Path) -> None:
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
            f"{expected_hash}, observed {observed_hash}"
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


def load_group_artifacts(
    results_dir: Path,
    *,
    groups: Sequence[str],
    seeds: Sequence[int],
) -> dict[tuple[str, int], dict]:
    artifacts: dict[tuple[str, int], dict] = {}
    missing: list[str] = []
    for group in groups:
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
        raise FileNotFoundError(f"Missing required artifacts: {missing}")
    return artifacts


def run_aggregation(
    results_dir: Path,
    baseline_dir: Path,
    seeds: Sequence[int],
    effective_mean_delta: float,
) -> dict:
    screen_artifacts = load_group_artifacts(results_dir, groups=SCREEN_GROUPS, seeds=seeds)
    baseline_artifacts = load_group_artifacts(baseline_dir, groups=(BASELINE_GROUP,), seeds=seeds)
    artifacts = {**screen_artifacts, **baseline_artifacts}
    reference_splits, reference_protocol = validate_cross_artifact_consistency(artifacts)

    per_run_scores: dict[tuple[str, int], dict[str, float]] = {
        key: per_run_session_scores(payload) for key, payload in artifacts.items()
    }
    within_window_std: dict[tuple[str, int], float] = {
        key: per_run_within_window_std(payload) for key, payload in artifacts.items()
    }
    val_sessions = sorted(reference_splits["val"])

    paired_deltas: dict[str, dict] = {}
    for treatment, control in PAIRS:
        pair_name = f"{treatment}_minus_{control}"
        per_session_seed_mean: dict[str, float] = {}
        per_seed_mean: dict[int, float] = {}
        for session in val_sessions:
            deltas = [
                per_run_scores[(treatment, seed)][session] - per_run_scores[(control, seed)][session]
                for seed in seeds
            ]
            per_session_seed_mean[session] = mean(deltas)
        for seed in seeds:
            per_seed_mean[seed] = mean(
                per_run_scores[(treatment, seed)][session] - per_run_scores[(control, seed)][session]
                for session in val_sessions
            )
        per_seed_mean_values = list(per_seed_mean.values())
        mean_delta = mean(per_seed_mean_values)
        sigma_paired = sigma_delta_paired(per_seed_mean_values)
        sigma_unpaired = sigma_delta_standard_error(
            sample_std([within_window_std[(treatment, seed)] for seed in seeds]),
            sample_std([within_window_std[(control, seed)] for seed in seeds]),
            len(seeds),
        )
        seed_correlation = implied_seed_correlation(
            sigma_a=sample_std([within_window_std[(treatment, seed)] for seed in seeds]),
            sigma_b=sample_std([within_window_std[(control, seed)] for seed in seeds]),
            per_seed_deltas=per_seed_mean_values,
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

    group_verdicts: dict[str, dict] = {}
    for group, (pair_a_name, pair_b_name) in FEATURE_GROUP_PAIRS.items():
        pair_results = {
            pair_a_name: (paired_deltas[pair_a_name]["verdict"], paired_deltas[pair_a_name]["decided_by"]),
            pair_b_name: (paired_deltas[pair_b_name]["verdict"], paired_deltas[pair_b_name]["decided_by"]),
        }
        verdict, decided_by = classify_group_verdict(pair_results=pair_results)
        group_verdicts[group] = {
            "content_pair_vs_f0": pair_a_name,
            "content_pair_vs_shuffled_control": pair_b_name,
            "verdict": verdict,
            "decided_by": decided_by,
        }

    return {
        "schema_version": 1,
        "purpose": "electrode_ablation_f3_measurement_protocol_v4",
        "protocol_docs": [
            "sua_exploration/docs/UNIT_SIDE_FEATURE_ABLATION.md",
            "sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md",
        ],
        "screen_id": "electrode_ablation_f3",
        "no_formal_test_sessions_evaluated": True,
        "baseline_source": {
            "group": BASELINE_GROUP,
            "results_dir": str(baseline_dir),
            "note": "F0 reused from e3_tuning_ablation; F1/F2 not re-run",
        },
        "seeds": list(seeds),
        "protocol": reference_protocol,
        "fixed_protocol": EXPECTED_PROTOCOL,
        "session_splits": reference_splits,
        "group_contract": GROUP_CONTRACT,
        "paired_deltas": paired_deltas,
        "feature_group_verdicts": group_verdicts,
        "source_artifacts": {
            **{
                f"{group}_s{seed}": str(artifact_path(results_dir, group, seed))
                for group in SCREEN_GROUPS
                for seed in seeds
            },
            **{
                f"{BASELINE_GROUP}_s{seed}": str(artifact_path(baseline_dir, BASELINE_GROUP, seed))
                for seed in seeds
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--effective_mean_delta", type=float, required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    root = Path(__file__).resolve().parents[2]
    results_dir = args.results_dir or (root / "sua_exploration" / "results" / "electrode_ablation_f3")
    baseline_dir = args.baseline_dir or (root / "sua_exploration" / "results" / "e3_tuning_ablation")

    payload = run_aggregation(results_dir, baseline_dir, seeds, args.effective_mean_delta)
    out_path = args.out_path or (results_dir / "aggregate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {group: data["verdict"] for group, data in payload["feature_group_verdicts"].items()},
        indent=2,
        sort_keys=True,
    ))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
