#!/usr/bin/env python3
"""No-target code-closure receipt for the actual H-C/H-LS evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    DATES, EVALUATOR_PREFLIGHT_SCHEMA, EVALUATOR_PREFLIGHT_STATUS,
    read_immutable_json, sha256_file, write_immutable_json,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_terminal_checker import CHECKER_SCHEMA, CHECKER_STATUS


CLOSURE_SCHEMA = "h1_carrierid_date_lodo_hls_actual_target_evaluator_closure_v1"
CLOSURE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_ACTUAL_TARGET_EVALUATOR_CLOSURE_NOT_RUN"
CLOSURE_FILES = (
    "src/data/h1_carrierid_date_lodo_target.py",
    "src/data/h1_carrierid_date_lodo_hls_target.py",
    "src/data/h1_m4_eb_pilot.py",
    "src/models/components/h1_carrierid_spint.py",
    "src/models/h1_carrierid_date_lodo_phase2_module.py",
    "src/models/h1_carrierid_date_lodo_hls_module.py",
    "src/models/falcon_module.py",
    "scripts/h1_carrierid_date_lodo_hls_terminal_evaluate.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_contract.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_source_preflight.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_launch_receipt.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_source_executor.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_source_terminal_audit.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_post_upstream_binder.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_terminal_checker.py",
    "src/data/h1_carrierid_date_lodo_hls_early_v2.py",
    "configs/data/falcon_h1_carrierid_date_lodo_hls_early_v2.yaml",
    "configs/experiment/h1_carrierid_date_lodo_hls_early_v2.yaml",
)


class HlsEvaluatorClosureError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HlsEvaluatorClosureError(message)


def prepare(*, evaluator_preflight: Path, output: Path) -> dict[str, Any]:
    ready_path, ready, ready_sha = read_immutable_json(
        evaluator_preflight, schema=EVALUATOR_PREFLIGHT_SCHEMA, status=EVALUATOR_PREFLIGHT_STATUS,
    )
    row = ready.get("source_terminal_aggregate")
    _need(isinstance(row, Mapping), "H-LS readiness lacks source-terminal aggregate")
    terminal_path, terminal, terminal_sha = read_immutable_json(
        row.get("path", ""), schema=CHECKER_SCHEMA, status=CHECKER_STATUS,
    )
    _need(terminal_sha == row.get("sha256") and tuple(terminal.get("fixed_grid", ())) == DATES,
          "H-LS readiness/source-terminal aggregate drift")
    body = {
        "schema": CLOSURE_SCHEMA, "status": CLOSURE_STATUS,
        "evaluator_preflight": {"path": str(ready_path), "sha256": ready_sha},
        "source_terminal_aggregate": {"path": str(terminal_path), "sha256": terminal_sha},
        "fixed_grid": list(DATES),
        "comparison": "H-C minus H-LS",
        "code_sha256": {relative: sha256_file(ROOT / relative) for relative in CLOSURE_FILES},
        "scope": {"nwb_opened": 0, "target_recordings_opened": 0, "target_bytes_read": 0,
                  "checkpoint_loaded": False, "trainer_constructed": False,
                  "cuda_constructed": False, "target_evaluator_run": False},
    }
    written, digest = write_immutable_json(output, body)
    return {"status": CLOSURE_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(evaluator_preflight=args.evaluator_preflight, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
