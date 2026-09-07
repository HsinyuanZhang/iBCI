#!/usr/bin/env python3
"""Prepare-only five-date H-LS early-v2 source launch receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES, EARLY_LAUNCH_SCHEMA, EARLY_LAUNCH_STATUS, EARLY_PREFLIGHT_SCHEMA,
    EARLY_PREFLIGHT_STATUS, STATIC_PARTITIONS, need, read_immutable_json,
    sha256_file, validate_waiting_plan, write_immutable_json,
)


def prepare(*, waiting_plan: Path, source_preflights: Mapping[str, Path],
            output: Path) -> dict[str, Any]:
    waiting_path, _waiting, waiting_sha = validate_waiting_plan(waiting_plan)
    need(tuple(source_preflights) == DATES,
         "early-v2 launch receipt requires exactly the ordered complete five-date grid")
    rows: dict[str, Any] = {}
    pair_shas: set[str] = set()
    for date in DATES:
        path, body, digest = read_immutable_json(
            source_preflights[date], schema=EARLY_PREFLIGHT_SCHEMA, status=EARLY_PREFLIGHT_STATUS,
        )
        controls, scope, policy = body.get("source_controls"), body.get("scope"), body.get("upstream_policy")
        need(body.get("outer_date") == date and body.get("arm") == "H-LS",
             f"early-v2 source preflight arm/date drift: {date}")
        need(isinstance(controls, Mapping)
             and controls.get("carrier_intervention") == "temporal_velocity_label_rotation"
             and controls.get("seed") == 42 and controls.get("epochs") == 50
             and controls.get("fixed_terminal_epoch_zero_based") == 49
             and controls.get("warm_start") is False,
             f"early-v2 source controls drift: {date}")
        need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
             and scope.get("target_bytes_read") == 0
             and scope.get("trainer_constructed_or_launched") is False
             and scope.get("cuda_constructed_or_launched") is False,
             f"early-v2 source preflight scope drift: {date}")
        need(isinstance(policy, Mapping)
             and policy.get("complete_hs_hc_aggregate_required_for_source_training") is False
             and policy.get("complete_hs_hc_aggregate_required_before_any_target_or_evaluation") is True
             and policy.get("post_upstream_binder_required") is True
             and policy.get("does_not_claim_v1_ready") is True,
             f"early-v2 source/upstream separation drift: {date}")
        need(body.get("waiting_plan", {}).get("sha256") == waiting_sha,
             f"early-v2 preflight binds another waiting plan: {date}")
        pair = body.get("matched_h_c_pair_preflight")
        need(isinstance(pair, Mapping) and isinstance(pair.get("sha256"), str),
             f"early-v2 preflight lacks matched pair: {date}")
        pair_shas.add(str(pair["sha256"]))
        rows[date] = {
            "path": str(path), "sha256": digest,
            "source_binding_sha256": body.get("source_binding_sha256"),
            "base_source_binding_sha256": body.get("matched_h_c_base_source_binding_sha256"),
            "pair_preflight": dict(pair),
        }
    need(len(pair_shas) == len(DATES),
         "early-v2 five dates must bind five distinct matched H-C pair preflights")
    assigned = tuple(date for owner in STATIC_PARTITIONS for date in STATIC_PARTITIONS[owner])
    need(len(assigned) == len(set(assigned)) == len(DATES) and set(assigned) == set(DATES),
         "early-v2 static GPU partitions overlap or omit a date")
    body = {
        "schema": EARLY_LAUNCH_SCHEMA, "status": EARLY_LAUNCH_STATUS,
        "route": "H1-HLS-EARLY-V2-SOURCE-ONLY",
        "not_a_runtime_launcher": True, "explicit_execute_required": True,
        "does_not_claim_upstream_complete": True,
        "target_gate": "CLOSED_UNTIL_POST_UPSTREAM_BINDER",
        "waiting_plan": {"path": str(waiting_path), "sha256": waiting_sha},
        "fixed_grid": list(DATES), "source_preflights": rows,
        "static_two_gpu_partitions": {owner: list(dates) for owner, dates in STATIC_PARTITIONS.items()},
        "training_contract": {
            "arm": "H-LS", "fresh_seed": 42, "epochs": 50,
            "fixed_terminal_epoch_zero_based": 49, "checkpoint_warm_start": False,
            "one_writer_per_outer_date": True, "run_directories_must_be_new": True,
            "target_optimizer_steps": 0, "target_backward_steps": 0,
        },
        "scope": {"nwb_opened": 0, "target_recordings_opened": 0, "target_bytes_read": 0,
                  "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
                  "tmux_started": False},
        "code_sha256": {"launch_receipt": sha256_file(Path(__file__).resolve())},
    }
    written, digest = write_immutable_json(output, body)
    return {"status": EARLY_LAUNCH_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waiting-plan", required=True, type=Path)
    parser.add_argument("--source-preflight", action="append", required=True, metavar="DATE=PATH")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    mapping: dict[str, Path] = {}
    for item in args.source_preflight:
        date, separator, value = str(item).partition("=")
        if not separator or date in mapping:
            raise SystemExit("--source-preflight requires unique DATE=PATH values")
        mapping[date] = Path(value)
    print(json.dumps(prepare(waiting_plan=args.waiting_plan, source_preflights=mapping,
                             output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
