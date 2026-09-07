#!/usr/bin/env python3
"""Seal the post-evaluation H1 D-S4e/D-Q4e route disposition.

This is receipt-only bookkeeping after the already-authorized one-shot
leakage-diagnostic evaluator.  It never opens data, a checkpoint, CUDA, or a
Trainer.  Its output identifies only the *next preparation* from the frozen
decision table; it does not authorize any route's GPU work, select a model, or
upgrade D-Q4e into a deployable/paper-main result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_immutable_receipt,
    sha256_file,
    write_immutable_json,
)
from scripts.h1_carrierid_next_route_plan import SCHEMA as ROUTE_PLAN_SCHEMA
from scripts.h1_carrierid_next_route_plan import build_static_plan


DISPOSITION_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_route_disposition_v1"
DISPOSITION_STATUS = "PASS_H1_CARRIERID_H32_FRESH_D_S4E_D_Q4E_ROUTE_DISPOSITION_RECEIPT_ONLY"
CLAIM_BOUNDARY = "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT"
TERMINAL_PREFLIGHT_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_terminal_evaluator_preflight_v1"
TERMINAL_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_H32_FRESH_D_S4E_D_Q4E_TERMINAL_EVALUATOR_SOURCE_CLOSURE_NONLAUNCH"
TERMINAL_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_leakage_terminal_eval_v1"
TERMINAL_STATUS = "PASS_H1_CARRIERID_H32_FRESH_D_S4E_D_Q4E_LEAKAGE_DIAGNOSTIC_EVALUATED"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NormalizedV2ContractError(f"route disposition requires mapping {label}")
    return value


def route_from_terminal_evaluation(terminal: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a frozen evaluator result and map it to exactly one static exit."""

    if terminal.get("schema") != TERMINAL_SCHEMA or terminal.get("status") != TERMINAL_STATUS:
        raise NormalizedV2ContractError("route disposition requires the passing D-S4e/D-Q4e terminal evaluator receipt")
    if terminal.get("claim_boundary") != CLAIM_BOUNDARY:
        raise NormalizedV2ContractError("route disposition receipt lost D-Q4e leakage-only boundary")
    interpretation = _require_mapping(terminal.get("frozen_interpretation"), "frozen_interpretation")
    metrics = _require_mapping(terminal.get("metrics"), "metrics")
    s4 = _require_mapping(metrics.get("d_s4e"), "metrics.d_s4e")
    q4 = _require_mapping(metrics.get("d_q4e"), "metrics.d_q4e")
    delta = _require_mapping(metrics.get("d_q4e_minus_d_s4e"), "metrics.d_q4e_minus_d_s4e")
    try:
        observed_delta = float(q4["pooled_r2"]) - float(s4["pooled_r2"])
        recorded_delta = float(delta["pooled_r2"])
        interpreted_delta = float(interpretation["q4e_minus_s4e_pooled_r2"])
    except (KeyError, TypeError, ValueError) as error:
        raise NormalizedV2ContractError("route disposition terminal metric payload is incomplete") from error
    if observed_delta != recorded_delta or observed_delta != interpreted_delta:
        raise NormalizedV2ContractError("route disposition terminal pooled delta arithmetic drift")
    decision = interpretation.get("decision")
    static_plan = build_static_plan()
    if static_plan.get("schema") != ROUTE_PLAN_SCHEMA:
        raise NormalizedV2ContractError("route disposition static plan schema drift")
    routes = _require_mapping(static_plan.get("d_s4e_d_q4e_hypothesis_routes"), "static route table")
    if not isinstance(decision, str) or decision not in routes:
        raise NormalizedV2ContractError("route disposition terminal decision is absent from frozen static route table")
    route = _require_mapping(routes[decision], f"static route {decision}")
    next_preparation = route.get("next_preparation")
    if not isinstance(next_preparation, str) or not next_preparation:
        raise NormalizedV2ContractError("route disposition static route has no next preparation")
    return {
        "decision": decision,
        "next_preparation": next_preparation,
        "selection_boundary": route.get("selection_boundary"),
        "metrics_summary": {
            "d_s4e_pooled_r2": float(s4["pooled_r2"]),
            "d_q4e_pooled_r2": float(q4["pooled_r2"]),
            "d_q4e_minus_d_s4e_pooled_r2": observed_delta,
        },
    }


def write_route_disposition(*, terminal_evaluation: Path, terminal_preflight: Path, output: Path) -> dict[str, Any]:
    """Write one immutable route receipt, binding evaluator and no-target preflight."""

    preflight = assert_immutable_receipt(terminal_preflight, TERMINAL_PREFLIGHT_STATUS)
    if preflight.get("schema") != TERMINAL_PREFLIGHT_SCHEMA or preflight.get("launch", {}).get("authorized") is not False:
        raise NormalizedV2ContractError("route disposition requires immutable no-target terminal preflight")
    terminal = assert_immutable_receipt(terminal_evaluation, TERMINAL_STATUS)
    recorded_preflight = _require_mapping(terminal.get("terminal_preflight"), "terminal_evaluation.terminal_preflight")
    if Path(str(recorded_preflight.get("path", ""))).resolve() != terminal_preflight.resolve():
        raise NormalizedV2ContractError("route disposition evaluator used another terminal preflight path")
    if recorded_preflight.get("sha256") != sha256_file(terminal_preflight):
        raise NormalizedV2ContractError("route disposition evaluator terminal-preflight hash drift")
    route = route_from_terminal_evaluation(terminal)
    plan = build_static_plan()
    payload = {
        "schema": DISPOSITION_SCHEMA,
        "status": DISPOSITION_STATUS,
        "scope": {
            "source_or_target_recordings_opened": 0,
            "checkpoints_opened": 0,
            "cuda_constructed_or_launched": False,
            "trainer_constructed": False,
            "model_or_checkpoint_selected": False,
            "gpu_route_authorized": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "terminal_evaluation": {
            "path": str(terminal_evaluation.resolve()), "sha256": sha256_file(terminal_evaluation),
        },
        "terminal_preflight": {
            "path": str(terminal_preflight.resolve()), "sha256": sha256_file(terminal_preflight),
        },
        "static_route_plan": {"schema": plan["schema"], "sha256": sha256_file(Path(__file__).resolve().with_name("h1_carrierid_next_route_plan.py"))},
        "frozen_route": route,
        "next_boundary": {
            "authorization": "none; this is a preparation label only",
            "mandatory_before_any_new_gpu_route": "the static plan's independent source-only selection prerequisite",
            "source_only_selection_status": plan["source_only_selection_prerequisite"]["status"],
        },
    }
    path, digest = write_immutable_json(output, payload)
    return {"status": DISPOSITION_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-evaluation", required=True, type=Path)
    parser.add_argument("--terminal-preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write_route_disposition(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
