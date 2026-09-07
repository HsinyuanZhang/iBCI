#!/usr/bin/env python3
"""No-GPU launch-receipt checker for the deferred H1 five-date H-LS grid.

This code can only publish an immutable *prepared, not launched* receipt
after every date has a passed source-only preflight.  It is deliberately not a
subprocess/tmux/GPU launcher; a later operator must implement and review the
H-LS source adapter separately rather than hot-patching the running H-S/H-C
producer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    DATES, LAUNCH_SCHEMA, LAUNCH_STATUS, ROOT, SOURCE_PREFLIGHT_SCHEMA, SOURCE_PREFLIGHT_STATUS,
    UPSTREAM_AGGREGATE_DEFAULT, WAITING_SCHEMA, WAITING_STATUS, read_immutable_json,
    require_complete_upstream, sha256_file, write_immutable_json,
)


class HlsFiveDateLaunchError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HlsFiveDateLaunchError(message)


def inspect_waiting_plan(*, waiting_plan: Path, upstream_aggregate: Path = UPSTREAM_AGGREGATE_DEFAULT) -> dict[str, Any]:
    """Read only receipts and return the current blocked/ready status."""

    plan_path, plan, plan_sha = read_immutable_json(waiting_plan, schema=WAITING_SCHEMA, status=WAITING_STATUS)
    aggregate = require_complete_upstream(upstream_aggregate)
    _need(tuple(plan.get("future_complete_grid", {}).get("outer_dates", ())) == DATES,
          "H-LS waiting plan lacks the full fixed five-date grid")
    return {
        "waiting_plan": {"path": str(plan_path), "sha256": plan_sha},
        "upstream_aggregate": {"path": str(aggregate[0]), "sha256": aggregate[1]},
        "ready_for_explicit_source_preflights": True,
        "nwb_opened": 0, "target_opened": 0, "trainer_constructed": False, "cuda_constructed": False,
    }


def prepare(*, waiting_plan: Path, upstream_aggregate: Path, source_preflights: Mapping[str, Path],
            explicit_route: str, output: Path) -> dict[str, Any]:
    """Publish a no-launch receipt only after 5/5 exact source preflights."""

    _need(explicit_route == "H1-HLS-FIVEDATE", "explicit route must be H1-HLS-FIVEDATE")
    state = inspect_waiting_plan(waiting_plan=waiting_plan, upstream_aggregate=upstream_aggregate)
    _need(tuple(source_preflights) == DATES, "H-LS launch receipt requires exactly all five date preflights")
    rows: dict[str, Any] = {}
    for date in DATES:
        path, body, digest = read_immutable_json(
            source_preflights[date], schema=SOURCE_PREFLIGHT_SCHEMA, status=SOURCE_PREFLIGHT_STATUS,
        )
        _need(body.get("outer_date") == date, f"H-LS source preflight date drift: {date}")
        scope = body.get("scope")
        controls = body.get("source_controls")
        _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
              and scope.get("target_bytes_read") == 0 and scope.get("cuda_constructed_or_launched") is False
              and scope.get("trainer_constructed_or_launched") is False, f"H-LS source preflight scope drift: {date}")
        _need(isinstance(controls, Mapping) and controls.get("carrier_intervention") == "temporal_velocity_label_rotation"
              and controls.get("same_h_c_source_windows") is True and controls.get("same_h_c_source_schedule") is True
              and controls.get("same_h_c_normalizer") is True and controls.get("seed") == 42
              and controls.get("epochs") == 50 and controls.get("fixed_terminal_epoch_zero_based") == 49,
              f"H-LS source controls drift: {date}")
        rows[date] = {"path": str(path), "sha256": digest, "source_binding_sha256": body.get("source_binding_sha256")}
    body = {
        "schema": LAUNCH_SCHEMA, "status": LAUNCH_STATUS,
        "route": explicit_route, "launch_authorized": False, "not_a_gpu_launcher": True,
        "waiting_plan": state["waiting_plan"], "upstream_aggregate": state["upstream_aggregate"],
        "source_preflights": rows,
        "fixed_grid": list(DATES),
        "training_contract": {
            "arm": "H-LS", "fresh_seed": 42, "epochs": 50, "fixed_terminal_epoch_zero_based": 49,
            "checkpoint_warm_start": False, "target_optimizer_steps": 0, "target_backward_steps": 0,
            "carrier_semantics": "strong temporal velocity-label misalignment; not pure label deletion",
        },
        "next_action": "root-reviewed explicit source-training executor; this receipt cannot execute a process",
        "scope": {"nwb_opened": 0, "target_opened": 0, "trainer_constructed_or_launched": False,
                  "cuda_constructed_or_launched": False, "tmux_started": False},
        "code_sha256": {"launcher": sha256_file(Path(__file__).resolve())},
    }
    path, digest = write_immutable_json(output, body)
    return {"status": LAUNCH_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waiting-plan", type=Path, required=True)
    parser.add_argument("--upstream-aggregate", type=Path, default=UPSTREAM_AGGREGATE_DEFAULT)
    parser.add_argument("--source-preflight", action="append", default=[], metavar="DATE=PATH")
    parser.add_argument("--explicit-route")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        print(json.dumps(inspect_waiting_plan(waiting_plan=args.waiting_plan,
                                              upstream_aggregate=args.upstream_aggregate), sort_keys=True))
        return
    mapping: dict[str, Path] = {}
    for item in args.source_preflight:
        date, separator, value = str(item).partition("=")
        if not separator or date in mapping:
            raise SystemExit("--source-preflight requires unique DATE=PATH entries")
        mapping[date] = Path(value)
    if args.explicit_route is None:
        raise SystemExit("--explicit-route is required when publishing a no-launch receipt")
    print(json.dumps(prepare(waiting_plan=args.waiting_plan, upstream_aggregate=args.upstream_aggregate,
                             source_preflights=mapping, explicit_route=args.explicit_route, output=args.output),
                     sort_keys=True))


if __name__ == "__main__":
    main()
