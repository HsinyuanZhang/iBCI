#!/usr/bin/env python3
"""Publish the non-executing H1 five-date H-LS waiting plan.

It is intentionally a JSON-only planning operation.  It cannot open an NWB,
construct a Trainer/CUDA object, start tmux, or produce an H-LS checkpoint.
The hard gate is the immutable *complete* five-date H-S/H-C aggregate; no
partial date, H-C sign, or fold-0 H-LS result can select an outer date.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    DATES, NULL_AUDIT_DEFAULT, ROOT, UPSTREAM_AGGREGATE_DEFAULT, WAITING_SCHEMA, WAITING_STATUS,
    sha256_file, upstream_aggregate_state, validate_null_strength_audit, write_immutable_json,
)


DEFAULT_OUTPUT = (
    ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls_fivedate_waiting_plan/"
    "H1_CARRIERID_DATE_LODO_HLS_FIVEDATE_WAITING_PLAN_v1.json"
)


def build(*, output: Path, upstream_aggregate: Path = UPSTREAM_AGGREGATE_DEFAULT,
          null_strength_audit: Path = NULL_AUDIT_DEFAULT) -> dict[str, Any]:
    """Create the exact H-LS specification without data/trainer/GPU access."""

    audit_path, _audit, audit_sha = validate_null_strength_audit(null_strength_audit)
    aggregate = upstream_aggregate_state(upstream_aggregate)
    body = {
        "schema": WAITING_SCHEMA,
        "status": WAITING_STATUS,
        "purpose": (
            "Five-date separately-trained H-LS expansion of the fold-0 H-C minus H-LS temporal "
            "velocity-label-misalignment comparison. It tests whether the fold-0 pooled +0.0256 R2 "
            "and mixed recording signs generalize; it does not tune or select on those values."
        ),
        "current_gate": {
            "name": "immutable complete five-date H-S/H-C aggregate",
            "required_schema": "h1_carrierid_date_lodo_five_date_heldout_aggregate_v1",
            "required_status": "PASS_H1_CARRIERID_DATE_LODO_FIVE_DATE_SOURCE_DATE_SCREEN_COMPLETE_NO_ROUTE_SELECTED",
            "required_dates": list(DATES),
            "required_fields": {
                "all_five_date_receipts_present_and_validated": True,
                "route_prerequisite.status": "source/date screen complete",
                "route_prerequisite.automatic_route_selection": "FORBIDDEN",
            },
            "observed_at_plan_creation": aggregate,
            "bypass_from_partial_dates_or_fold0": "FORBIDDEN",
        },
        "control_semantics": {
            "name": "H-LS strong temporal velocity-label-misalignment control",
            "not_a_pure_label_deletion_control": True,
            "source_training": (
                "Keep source neural rates, source response windows, source M=4 support trials, schedule, and "
                "normalizer fixed. Deterministically rotate temporal velocity-label rows before refitting each "
                "four-dimensional carrier, then train a fresh H-LS consumer on those changed carriers."
            ),
            "deployment_evaluation": (
                "Use the same declared label-rotation carrier construction on each outer-date M=4 support block; "
                "do not change neural support/query identity or query-window start."
            ),
            "strong_null_audit": {
                "path": str(audit_path), "sha256": audit_sha,
                "required_sha256": "6404572da206e7664dc236b9d15005cc05b129b64b1541a841ca75a18f248409",
                "interpretation": "near-orthogonal temporal misalignment, not exchangeable label deletion",
            },
        },
        "future_complete_grid": {
            "outer_dates": list(DATES),
            "fold_or_date_selection": "FORBIDDEN; all five required or fail closed",
            "arms": ["H-C existing fresh source-only reference", "H-LS fresh separately-trained control"],
            "seed": 42,
            "source_roster": "exact Phase-1 source bundle for each outer date",
            "source_windows": "identical H-C/H-LS per outer date",
            "source_M4_schedule": "identical H-C/H-LS per outer date",
            "source_RMS_normalizer": "identical H-C/H-LS per outer date",
            "epochs": 50,
            "fixed_terminal_epoch_zero_based": 49,
            "checkpoint_warm_start": "FORBIDDEN",
            "deployment_target_optimizer_steps": 0,
            "deployment_target_backward_steps": 0,
            "deployment_model_state_update": "FORBIDDEN",
        },
        "primary_endpoint": {
            "quantity": "H-C minus H-LS held source-date R2",
            "inference_unit": "outer date; recordings remain nested and are reported individually",
            "required_reporting": [
                "all five datewise pooled deltas", "all eleven recording-level signed deltas",
                "five-date mean/median/paired standard error/bootstrap", "no post-hoc date subset",
            ],
            "permitted_interpretation": (
                "A positive complete-grid delta supports the necessity of correct temporal velocity-label pairing "
                "under this strong misalignment control. A mixed or null result limits that claim; it does not erase "
                "the separate H-C versus H-S performance result."
            ),
        },
        "execution_order_after_gate": [
            "root reviews an H-LS-specific source adapter/config/checkpoint provenance implementation; do not mutate the active H-S/H-C runtime",
            "one explicit CPU source-only preflight per all five dates, binding the existing exact Phase-1 bundle and source-only H-LS refit",
            "one immutable no-GPU launch receipt that lists all five dates; no date-at-a-time numerical selection",
            "fresh e49 H-LS source training, one run per date, with no target optimizer/backward steps",
            "source-only terminal checker over all five H-LS checkpoints before any outer-date target open",
            "a separate one-shot H-C/H-LS target evaluator preflight, then forward-only outer-date evaluation with state hashes before/after",
            "receipt-only five-date aggregation with date as inference unit",
        ],
        "current_scope": {
            "nwb_opened": 0, "target_recordings_opened": 0, "target_bytes_read": 0,
            "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
            "tmux_started": False, "checkpoint_created_or_loaded": False,
        },
        "code_sha256": {
            "contract": sha256_file(ROOT / "scripts/h1_carrierid_date_lodo_hls_fivedate_contract.py"),
            "waiting_preflight": sha256_file(Path(__file__).resolve()),
            "launcher": sha256_file(ROOT / "scripts/h1_carrierid_date_lodo_hls_fivedate_launch_receipt.py"),
            "terminal_checker": sha256_file(ROOT / "scripts/h1_carrierid_date_lodo_hls_fivedate_terminal_checker.py"),
            "target_evaluator_preflight": sha256_file(ROOT / "scripts/h1_carrierid_date_lodo_hls_fivedate_target_evaluator_preflight.py"),
        },
    }
    written, digest = write_immutable_json(output, body)
    return {"status": WAITING_STATUS, "receipt_path": str(written), "receipt_sha256": digest,
            "upstream_ready": bool(aggregate["ready"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--upstream-aggregate", type=Path, default=UPSTREAM_AGGREGATE_DEFAULT)
    parser.add_argument("--null-strength-audit", type=Path, default=NULL_AUDIT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build(output=args.output, upstream_aggregate=args.upstream_aggregate,
                           null_strength_audit=args.null_strength_audit), sort_keys=True))


if __name__ == "__main__":
    main()
