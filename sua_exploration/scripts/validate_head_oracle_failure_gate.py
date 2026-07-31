#!/usr/bin/env python3
"""Recompute a failed exact-head oracle gate before launching M15."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from aggregate_sua_head_oracle import aggregate


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
            "the M15 handoff requires the seed-42 oracle screen"
        )
    try:
        observed = json.loads(
            aggregate_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid oracle aggregate: {aggregate_path}"
        ) from exc
    expected = aggregate(result_dir, v1_result_dir, seeds)
    if _without_generation_time(observed) != _without_generation_time(
        expected
    ):
        raise ValueError(
            "stored oracle aggregate differs from strict recomputation"
        )
    stage0 = expected.get("diagnostic_stage0_gates") or {}
    if stage0.get("pass") is not False:
        raise ValueError(
            "refusing M15 launch because the oracle diagnostic passed"
        )
    if expected.get("formal_effectiveness_pass") is not False:
        raise ValueError(
            "seed-42 oracle aggregate cannot be formally effective"
        )
    if expected.get("no_test_files_evaluated") is not True:
        raise ValueError(
            "oracle gate does not prove formal-test isolation"
        )


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
    print("Validated strict exact-head seed-42 failure gate.")


if __name__ == "__main__":
    main()
