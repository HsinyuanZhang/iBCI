#!/usr/bin/env python3
"""Fail-closed aggregate for the M_T4=15 uncertainty-shrunk T4 pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sua_confidence_film_t4_budget import (  # noqa: E402
    _per_session_epoch_mean,
    hierarchical_ci,
    summarize,
)


PILOT_ARMS = ("t4_m15", "t4w3_m15", "ts4w3_m15")
EXPECTED_GROUP = {
    "t4_m15": "t4",
    "t4w3_m15": "t4w3",
    "ts4w3_m15": "ts4w3",
    "t4_m50": "t4",
}
EXPECTED_POOL = {
    "t4_m15": 15,
    "t4w3_m15": 15,
    "ts4w3_m15": 15,
    "t4_m50": 50,
}
EPOCHS = list(range(5, 13))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    if not set(seeds).issubset({42, 43, 44}):
        raise argparse.ArgumentTypeError("seeds must be a subset of 42,43,44")
    return seeds


def validate_arm(
    path: Path,
    *,
    arm: str,
    seed: int,
) -> tuple[list[str], np.ndarray, str]:
    payload = _load(path)
    protocol = payload.get("protocol") or {}
    expected_pool = EXPECTED_POOL[arm]
    _require(f"{path}: variant", payload.get("variant"), "B3S")
    _require(f"{path}: seed", payload.get("seed"), seed)
    _require(f"{path}: split", payload.get("split_counts"), [27, 6, 6])
    _require(f"{path}: max units", payload.get("max_units_exclusive"), 100)
    _require(f"{path}: no test", payload.get("no_test_files_evaluated"), True)
    _require(f"{path}: no eval backward", payload.get("uses_backward_gradients"), False)
    _require(
        f"{path}: no eval label updates",
        payload.get("uses_behavior_labels_for_weight_updates"),
        False,
    )
    _require(
        f"{path}: labelled T4 support",
        payload.get("calibration_features_use_behavior_labels"),
        True,
    )
    _require(f"{path}: activity calibration", protocol.get("calibration_n"), 30)
    _require(f"{path}: common evaluation start", protocol.get("pool_size"), 50)
    _require(
        f"{path}: labelled feature budget",
        protocol.get("label_feature_calibration_n"),
        expected_pool,
    )
    _require(f"{path}: epoch window", protocol.get("epoch_window"), EPOCHS)
    _require(
        f"{path}: chronological label scope",
        payload.get("calibration_feature_label_scope"),
        f"chronological_rewarded_trials[0:{expected_pool}]",
    )

    metadata_path = Path(payload.get("run_metadata_path", ""))
    if not metadata_path.is_file():
        raise ValueError(f"{path}: missing run metadata {metadata_path}")
    metadata = _load(metadata_path)
    _require(f"{path}: metadata variant", metadata.get("variant"), "B3S")
    _require(f"{path}: metadata seed", metadata.get("seed"), seed)
    side = metadata.get("side_features") or {}
    _require(f"{path}: side group", side.get("group"), EXPECTED_GROUP[arm])
    _require(f"{path}: feature version", side.get("feature_version"), 1)
    _require(f"{path}: side pool", side.get("pool_size"), expected_pool)
    training = metadata.get("training") or {}
    _require(f"{path}: train activity budget", training.get("calibration_n_trials"), 30)
    _require(f"{path}: epochs", training.get("max_epochs"), 12)
    _require(f"{path}: no early stop", training.get("no_early_stopping"), True)
    _require(f"{path}: checkpoint each epoch", training.get("checkpoint_every_epoch"), True)
    _require(f"{path}: completed", metadata.get("status"), "completed")
    _require(f"{path}: formal unopened", metadata.get("held_out_test_evaluated"), False)
    _require(f"{path}: fresh fit", metadata.get("encoder_warmstart_path"), None)
    expected_permutation = seed if arm == "ts4w3_m15" else None
    _require(f"{path}: permutation receipt", side.get("permutation_seed"), expected_permutation)
    sessions, values = _per_session_epoch_mean(payload, path)
    return sessions, values, str(metadata_path.resolve())


def noninferiority_summary(
    treatment: np.ndarray,
    reference: np.ndarray,
    *,
    seeds: tuple[int, ...],
    sessions: list[str],
    margin: float = 0.03,
) -> dict:
    delta = treatment - reference
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    ci = hierarchical_ci(delta)
    stage0 = float(delta.mean()) >= -margin
    formal = bool(
        len(seeds) >= 3
        and stage0
        and ci[0] >= -margin
    )
    return {
        "margin_r2": margin,
        "mean_delta_r2": float(delta.mean()),
        "per_seed_mean_delta_r2": {
            str(seed): float(value) for seed, value in zip(seeds, seed_means)
        },
        "per_session_mean_delta_r2": {
            session: float(value)
            for session, value in zip(sessions, session_means)
        },
        "hierarchical_bootstrap_95ci": ci,
        "stage0_within_margin": stage0,
        "formal_within_margin": formal,
    }


def aggregate(
    result_dir: Path,
    reference_dir: Path,
    seeds: tuple[int, ...],
) -> dict:
    matrices: dict[str, np.ndarray] = {}
    artifacts: dict[str, dict[str, str]] = {
        arm: {} for arm in (*PILOT_ARMS, "t4_m50")
    }
    session_names: list[str] | None = None
    for arm in PILOT_ARMS:
        rows = []
        for seed in seeds:
            path = result_dir / f"{arm}_s{seed}.json"
            sessions, values, _metadata = validate_arm(
                path,
                arm=arm,
                seed=seed,
            )
            if session_names is None:
                session_names = sessions
            elif sessions != session_names:
                raise ValueError(f"{path}: validation session matrix differs")
            rows.append(values)
            artifacts[arm][str(seed)] = str(path.resolve())
        matrices[arm] = np.asarray(rows, dtype=np.float64)

    reference_rows = []
    for seed in seeds:
        path = reference_dir / f"t4m50_s{seed}.json"
        sessions, values, _metadata = validate_arm(
            path,
            arm="t4_m50",
            seed=seed,
        )
        if sessions != session_names:
            raise ValueError(f"{path}: M50 reference session matrix differs")
        reference_rows.append(values)
        artifacts["t4_m50"][str(seed)] = str(path.resolve())
    matrices["t4_m50"] = np.asarray(reference_rows, dtype=np.float64)
    assert session_names is not None

    mechanism = {
        "t4w3_m15_vs_t4_m15": summarize(
            matrices["t4w3_m15"],
            matrices["t4_m15"],
            seeds=seeds,
            sessions=session_names,
        ),
        "t4w3_m15_vs_ts4w3_m15": summarize(
            matrices["t4w3_m15"],
            matrices["ts4w3_m15"],
            seeds=seeds,
            sessions=session_names,
        ),
    }
    noninferiority = noninferiority_summary(
        matrices["t4w3_m15"],
        matrices["t4_m50"],
        seeds=seeds,
        sessions=session_names,
    )
    stage0 = bool(
        all(
            comparison["passes_stage0_descriptive_gates"]
            for comparison in mechanism.values()
        )
        and noninferiority["stage0_within_margin"]
    )
    formal = bool(
        all(
            comparison["passes_formal_effectiveness_gates"]
            for comparison in mechanism.values()
        )
        and noninferiority["formal_within_margin"]
    )
    if any(not math.isfinite(float(matrix.mean())) for matrix in matrices.values()):
        raise ValueError("non-finite arm score")
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "low_label_uncertainty_shrunk_t4_decoding_pilot",
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "protocol": {
            "M_activity": 30,
            "M_T4_pilot": 15,
            "M_T4_reference": 50,
            "common_evaluation_start": 50,
            "shrinkage": "wiener strength=3; frozen by train-only nested LOSO",
            "epochs": 12,
            "scored_epoch_window": EPOCHS,
            "seeds": list(seeds),
            "sessions": session_names,
            "formal_test_evaluated": False,
        },
        "artifacts": artifacts,
        "arm_mean_r2": {
            arm: float(matrix.mean()) for arm, matrix in matrices.items()
        },
        "mechanism_contrasts": mechanism,
        "m15_shrink_vs_m50_t4_noninferiority": noninferiority,
        "stage0_descriptive_mechanism_and_label_reduction_pass": stage0,
        "formal_effectiveness_eligible": len(seeds) >= 3,
        "formal_effectiveness_pass": formal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(42,))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.result_dir.expanduser().resolve(),
        args.reference_dir.expanduser().resolve(),
        args.seeds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "arm_mean_r2": result["arm_mean_r2"],
                "stage0_pass": result[
                    "stage0_descriptive_mechanism_and_label_reduction_pass"
                ],
                "formal_effectiveness_pass": result[
                    "formal_effectiveness_pass"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
