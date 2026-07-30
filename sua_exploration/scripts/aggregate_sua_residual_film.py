#!/usr/bin/env python3
"""Fail-closed aggregate for the train-audit-driven residual-only FiLM round."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sua_confidence_film_t4_budget import (
    summarize,
    validate_arm,
)


ARMS = (
    "t4_continuation",
    "film",
    "residual_film",
    "residual_shuffle",
    "residual_nofilm",
)
NEW_ARMS = ("residual_film", "residual_shuffle", "residual_nofilm")
CONTRASTS = {
    "residual_film_vs_t4_continuation": "t4_continuation",
    "residual_film_vs_full_confidence_film": "film",
    "residual_film_vs_residual_shuffle": "residual_shuffle",
    "residual_film_vs_residual_nofilm": "residual_nofilm",
}
REQUIRED = (
    "residual_film_vs_t4_continuation",
    "residual_film_vs_residual_shuffle",
    "residual_film_vs_residual_nofilm",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    if not set(seeds).issubset({42, 43, 44}):
        raise argparse.ArgumentTypeError("seeds must be a subset of 42,43,44")
    return seeds


def aggregate(result_dir: Path, seeds: tuple[int, ...]) -> dict:
    matrices: dict[str, np.ndarray] = {}
    artifacts: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    session_names: list[str] | None = None
    anchor_hashes: dict[str, str] = {}
    parameter_counts: dict[str, dict[str, int]] = {}

    for arm in ARMS:
        rows = []
        for seed in seeds:
            path = result_dir / f"{arm}_m50_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            sessions, values, warmstart, metadata_path = validate_arm(
                path, arm, 50, seed
            )
            if session_names is None:
                session_names = sessions
            elif session_names != sessions:
                raise ValueError(f"{path}: validation session matrix differs")
            previous = anchor_hashes.setdefault(str(seed), warmstart)
            if previous != warmstart:
                raise ValueError(
                    f"seed {seed}: all residual-round arms must share one anchor SHA"
                )
            metadata = _load(Path(metadata_path))
            profile = metadata.get("encoder_cost_profile_reference") or {}
            parameter_count = profile.get("parameter_count")
            if not isinstance(parameter_count, int):
                raise ValueError(f"{path}: encoder parameter receipt is missing")
            parameter_counts.setdefault(str(seed), {})[arm] = parameter_count
            if arm in NEW_ARMS:
                receipt = metadata.get("confidence_film") or {}
                if receipt.get("confidence_mask") != [True, False]:
                    raise ValueError(
                        f"{path}: residual-only confidence mask is not [true,false]"
                    )
                expected_additive = arm == "residual_nofilm"
                if receipt.get("additive_only") is not expected_additive:
                    raise ValueError(
                        f"{path}: residual additive-control receipt drifted"
                    )
                if receipt.get("parameter_matched_six_wide_context") is not True:
                    raise ValueError(
                        f"{path}: six-wide parameter-matched context receipt is missing"
                    )
            rows.append(values)
            artifacts[arm][str(seed)] = str(path.resolve())
        matrices[arm] = np.asarray(rows, dtype=np.float64)

    assert session_names is not None
    for seed, counts in parameter_counts.items():
        expected = counts["film"]
        for arm in ("residual_film", "residual_shuffle", "residual_nofilm"):
            if counts[arm] != expected:
                raise ValueError(
                    f"seed {seed}: {arm} parameter count {counts[arm]} "
                    f"!= full-confidence FiLM {expected}"
                )

    contrasts = {
        name: summarize(
            matrices["residual_film"],
            matrices[control],
            seeds=seeds,
            sessions=session_names,
        )
        for name, control in CONTRASTS.items()
    }
    stage0 = all(
        contrasts[name]["passes_stage0_descriptive_gates"]
        for name in REQUIRED
    )
    formal = all(
        contrasts[name]["passes_formal_effectiveness_gates"]
        for name in REQUIRED
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "residual_only_confidence_film_optimized_round",
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "protocol": {
            "M_activity": 30,
            "M_T4": 50,
            "common_evaluation_start": 50,
            "epochs": 12,
            "scored_epoch_window": list(range(5, 13)),
            "seeds": list(seeds),
            "sessions": session_names,
            "formal_test_evaluated": False,
            "optimization_basis": (
                "27-session train-only future-fit audit: residual variance "
                "predictive; geometry unsupported"
            ),
        },
        "selected_t4_anchor_sha256_by_seed": anchor_hashes,
        "artifacts": artifacts,
        "parameter_counts_by_seed": parameter_counts,
        "arm_mean_r2": {
            arm: float(values.mean()) for arm, values in matrices.items()
        },
        "contrasts": contrasts,
        "required_effectiveness_contrasts": list(REQUIRED),
        "stage0_descriptive_mechanism_pass": stage0,
        "formal_effectiveness_eligible": len(seeds) >= 3,
        "formal_effectiveness_pass": formal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(42,))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.result_dir.expanduser().resolve(), args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "arm_mean_r2": result["arm_mean_r2"],
                "stage0_descriptive_mechanism_pass": result[
                    "stage0_descriptive_mechanism_pass"
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
