#!/usr/bin/env python3
"""No-target evaluator readiness receipt for the future H1 H-C/H-LS grid.

There is intentionally no target loader in this preparation artifact.  A
future evaluator must be implemented as a separate isolated module and must
open a date only after this receipt binds all five verified H-LS source
terminals.  This prevents the half-completed H-S/H-C producer from being
modified or from leaking an outer target during plan preparation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    EVALUATOR_PREFLIGHT_SCHEMA, EVALUATOR_PREFLIGHT_STATUS, ROOT, UPSTREAM_AGGREGATE_DEFAULT,
    read_immutable_json, sha256_file, write_immutable_json,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_terminal_checker import CHECKER_SCHEMA, CHECKER_STATUS


class HlsFiveDateEvaluatorPreflightError(ValueError):
    pass


def prepare(*, source_terminal_aggregate: Path, output: Path) -> dict[str, Any]:
    """Publish target-closed readiness only; never evaluate or open a recording."""

    terminal_path, terminal, terminal_sha = read_immutable_json(
        source_terminal_aggregate, schema=CHECKER_SCHEMA, status=CHECKER_STATUS,
    )
    body: dict[str, Any] = {
        "schema": EVALUATOR_PREFLIGHT_SCHEMA, "status": EVALUATOR_PREFLIGHT_STATUS,
        "source_terminal_aggregate": {"path": str(terminal_path), "sha256": terminal_sha},
        "required_future_evaluator_contract": {
            "comparison": "H-C minus H-LS", "date_grid": terminal.get("fixed_grid"),
            "target_carrier_variants": {"H-C": "full correctly paired", "H-LS": "temporal velocity-label rotation"},
            "same_outer_date_support_and_query_neural_identity": True,
            "same_query_window_indices_per_paired_date": True,
            "forward_only": True, "target_optimizer_steps": 0, "target_backward_steps": 0,
            "state_sha256_before_after_equal": True, "r2_accumulator_dtype": "float64",
            "outer_date_inference_unit": True, "recordings_nested_and_reported": True,
        },
        "not_a_target_evaluator": True,
        "scope": {"nwb_opened": 0, "target_recordings_opened": 0, "target_bytes_read": 0,
                  "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
                  "target_optimizer_steps": 0, "target_backward_steps": 0},
        "code_sha256": {"target_evaluator_preflight": sha256_file(Path(__file__).resolve())},
    }
    path, digest = write_immutable_json(output, body)
    return {"status": EVALUATOR_PREFLIGHT_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-terminal-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(source_terminal_aggregate=args.source_terminal_aggregate, output=args.output),
                     sort_keys=True))


if __name__ == "__main__":
    main()
