#!/usr/bin/env python3
"""Receipt-only 5/5 source-terminal checker for future H1 H-LS training.

This checker intentionally validates a standardized immutable *source-only*
terminal receipt rather than opening a checkpoint or any data.  The future
H-LS adapter/executor must create one such receipt per date after a real e49
checkpoint audit; its implementation is a root-review gate because the active
H-S/H-C production runtime must not be edited while it is running.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    DATES, LAUNCH_SCHEMA, LAUNCH_STATUS, SOURCE_TERMINAL_SCHEMA, SOURCE_TERMINAL_STATUS,
    UPSTREAM_AGGREGATE_DEFAULT, read_immutable_json, require_complete_upstream, sha256_file,
    write_immutable_json,
)


CHECKER_SCHEMA = "h1_carrierid_date_lodo_hls_fivedate_source_terminal_aggregate_v1"
CHECKER_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_FIVEDATE_SOURCE_TERMINALS_NO_TARGET"


class HlsFiveDateTerminalError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HlsFiveDateTerminalError(message)


def check(*, launch_receipt: Path, upstream_aggregate: Path, terminal_receipts: Mapping[str, Path],
          output: Path) -> dict[str, Any]:
    """Require all five H-LS terminals before a target evaluator can exist."""

    aggregate_path, aggregate_sha = require_complete_upstream(upstream_aggregate)
    launch_path, launch, launch_sha = read_immutable_json(launch_receipt, schema=LAUNCH_SCHEMA, status=LAUNCH_STATUS)
    _need(launch.get("upstream_aggregate", {}).get("sha256") == aggregate_sha,
          "H-LS launch receipt binds a different H-S/H-C aggregate")
    _need(tuple(launch.get("fixed_grid", ())) == DATES, "H-LS launch receipt has incomplete date grid")
    _need(tuple(terminal_receipts) == DATES, "H-LS terminal check requires exactly five dates")
    checked: dict[str, Any] = {}
    for date in DATES:
        path, body, digest = read_immutable_json(
            terminal_receipts[date], schema=SOURCE_TERMINAL_SCHEMA, status=SOURCE_TERMINAL_STATUS,
        )
        _need(body.get("outer_date") == date and body.get("arm") == "H-LS", f"H-LS terminal arm/date drift: {date}")
        source = body.get("source_controls")
        deployment = body.get("deployment_updates")
        _need(isinstance(source, Mapping) and source.get("carrier_intervention") == "temporal_velocity_label_rotation"
              and source.get("seed") == 42 and source.get("epochs") == 50
              and source.get("fixed_terminal_epoch_zero_based") == 49 and source.get("warm_start") is False,
              f"H-LS terminal source contract drift: {date}")
        _need(isinstance(deployment, Mapping) and deployment.get("target_optimizer_steps") == 0
              and deployment.get("target_backward_steps") == 0 and deployment.get("target_model_state_updated") is False,
              f"H-LS terminal target-update contract drift: {date}")
        _need(body.get("source_binding_sha256") == launch["source_preflights"][date]["source_binding_sha256"],
              f"H-LS terminal source binding drift: {date}")
        checked[date] = {"path": str(path), "sha256": digest,
                         "source_binding_sha256": body["source_binding_sha256"]}
    body = {
        "schema": CHECKER_SCHEMA, "status": CHECKER_STATUS,
        "upstream_aggregate": {"path": str(aggregate_path), "sha256": aggregate_sha},
        "launch_receipt": {"path": str(launch_path), "sha256": launch_sha},
        "terminals": checked, "fixed_grid": list(DATES),
        "scope": {"nwb_opened": 0, "target_recordings_opened": 0, "target_bytes_read": 0,
                  "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
                  "target_evaluator": "NOT_RUN"},
        "code_sha256": {"terminal_checker": sha256_file(Path(__file__).resolve())},
    }
    path, digest = write_immutable_json(output, body)
    return {"status": CHECKER_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--upstream-aggregate", type=Path, default=UPSTREAM_AGGREGATE_DEFAULT)
    parser.add_argument("--terminal", action="append", required=True, metavar="DATE=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping: dict[str, Path] = {}
    for item in args.terminal:
        date, separator, value = str(item).partition("=")
        if not separator or date in mapping:
            raise SystemExit("--terminal requires unique DATE=PATH entries")
        mapping[date] = Path(value)
    print(json.dumps(check(launch_receipt=args.launch_receipt, upstream_aggregate=args.upstream_aggregate,
                           terminal_receipts=mapping, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
