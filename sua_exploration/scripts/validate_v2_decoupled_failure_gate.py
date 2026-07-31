#!/usr/bin/env python3
"""Recompute and validate the v2-failed gate before an oracle launch."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from aggregate_sua_decoupled_kv_v2 import aggregate


def _without_generation_time(payload: dict) -> dict:
    normalized = deepcopy(payload)
    normalized.pop("generated_at", None)
    return normalized


def validate_failure_gate(
    *,
    aggregate_path: Path,
    result_dir: Path,
    v1_result_dir: Path,
    seeds: tuple[int, ...],
) -> None:
    if seeds != (42,):
        raise ValueError(
            "the bounded oracle launch gate requires seed 42 only"
        )
    try:
        observed = json.loads(
            aggregate_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid v2 aggregate: {aggregate_path}"
        ) from exc
    expected = aggregate(result_dir, v1_result_dir, seeds)
    if _without_generation_time(observed) != _without_generation_time(
        expected
    ):
        raise ValueError(
            "stored v2 aggregate differs from strict recomputation"
        )
    stage0 = expected.get("stage0_descriptive_candidate_pass")
    if not isinstance(stage0, dict) or set(stage0) != {
        "kv2_e_t4",
        "kv2_e_only",
    }:
        raise ValueError("v2 aggregate has an invalid Stage-0 matrix")
    if any(value is not False for value in stage0.values()):
        raise ValueError(
            "refusing oracle launch because a v2 candidate passed"
        )
    if expected.get("formal_effectiveness_pass") is not False:
        raise ValueError(
            "seed-42 v2 aggregate cannot be formally effective"
        )
    if expected.get("no_test_files_evaluated") is not True:
        raise ValueError("v2 gate does not prove formal-test isolation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--v1-result-dir", type=Path, required=True
    )
    args = parser.parse_args()
    validate_failure_gate(
        aggregate_path=args.aggregate.expanduser().resolve(),
        result_dir=args.result_dir.expanduser().resolve(),
        v1_result_dir=args.v1_result_dir.expanduser().resolve(),
        seeds=(42,),
    )
    print("Validated strict v2 seed-42 failure gate.")


if __name__ == "__main__":
    main()
