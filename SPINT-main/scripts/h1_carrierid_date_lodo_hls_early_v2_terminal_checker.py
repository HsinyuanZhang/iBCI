#!/usr/bin/env python3
"""No-target checker promoting post-bound early-v2 terminals to evaluation-ready state.

The output deliberately uses the common five-date source-terminal aggregate
schema/status consumed by the existing evaluator-readiness step.  Its route
metadata remains explicit, and the actual evaluator separately verifies each
early-v2 receipt and this post-upstream binder before target access.
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


from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES, EARLY_TERMINAL_SCHEMA, EARLY_TERMINAL_STATUS, POST_BINDER_SCHEMA,
    POST_BINDER_STATUS, need, read_immutable_json, sha256_file, write_immutable_json,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_terminal_checker import CHECKER_SCHEMA, CHECKER_STATUS


def check(*, post_upstream_binder: Path, output: Path) -> dict[str, Any]:
    binder_path, binder, binder_sha = read_immutable_json(
        post_upstream_binder, schema=POST_BINDER_SCHEMA, status=POST_BINDER_STATUS,
    )
    need(tuple(binder.get("fixed_grid", ())) == DATES
         and binder.get("all_five_dates_compatible") is True,
         "early-v2 terminal checker requires complete compatible binder")
    need(binder.get("code_sha256", {}).get("post_upstream_binder")
         == sha256_file(Path(__file__).with_name("h1_carrierid_date_lodo_hls_early_v2_post_upstream_binder.py")),
         "early-v2 post-upstream binder code changed before terminal checker")
    rows: dict[str, Any] = {}
    for date in DATES:
        bound = binder.get("dates", {}).get(date)
        need(isinstance(bound, Mapping), f"early-v2 binder lacks date: {date}")
        terminal_row = bound.get("early_terminal")
        need(isinstance(terminal_row, Mapping), f"early-v2 binder lacks terminal: {date}")
        terminal_path, terminal, terminal_sha = read_immutable_json(
            terminal_row.get("path", ""), schema=EARLY_TERMINAL_SCHEMA, status=EARLY_TERMINAL_STATUS,
        )
        need(terminal_sha == terminal_row.get("sha256") and terminal.get("outer_date") == date
             and terminal.get("arm") == "H-LS"
             and terminal.get("source_binding_sha256") == terminal_row.get("source_binding_sha256")
             and terminal.get("base_source_binding_sha256") == bound.get("base_source_binding_sha256"),
             f"early-v2 terminal changed after post-upstream binder: {date}")
        controls, updates, scope = terminal.get("source_controls"), terminal.get("deployment_updates"), terminal.get("scope")
        need(isinstance(controls, Mapping)
             and controls.get("carrier_intervention") == "temporal_velocity_label_rotation"
             and controls.get("seed") == 42 and controls.get("epochs") == 50
             and controls.get("fixed_terminal_epoch_zero_based") == 49
             and controls.get("warm_start") is False,
             f"early-v2 terminal source controls drift: {date}")
        need(isinstance(updates, Mapping) and updates.get("target_optimizer_steps") == 0
             and updates.get("target_backward_steps") == 0
             and updates.get("target_model_state_updated") is False
             and isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
             and scope.get("target_bytes_read") == 0,
             f"early-v2 terminal target scope/update drift: {date}")
        rows[date] = {
            "path": str(terminal_path), "sha256": terminal_sha,
            "terminal_schema": EARLY_TERMINAL_SCHEMA, "terminal_status": EARLY_TERMINAL_STATUS,
            "source_binding_sha256": terminal["source_binding_sha256"],
            "base_source_binding_sha256": terminal["base_source_binding_sha256"],
        }
    body = {
        "schema": CHECKER_SCHEMA, "status": CHECKER_STATUS,
        "route": "H1-HLS-EARLY-V2-POST-UPSTREAM",
        "required_target_execution": {
            "owner": "remote5070ti_original_hc_replay",
            "device_type": "cuda", "device_name_must_contain": "5070 Ti",
            "reason": "recompute H-C on the original 5070 Ti before the reproduction gate",
        },
        "upstream_aggregate": dict(binder["upstream_aggregate"]),
        "post_upstream_binder": {"path": str(binder_path), "sha256": binder_sha,
                                 "schema": POST_BINDER_SCHEMA, "status": POST_BINDER_STATUS},
        "terminals": rows, "fixed_grid": list(DATES),
        "scope": {"nwb_opened": 0, "target_recordings_opened": 0, "target_bytes_read": 0,
                  "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
                  "target_evaluator": "NOT_RUN"},
        "code_sha256": {"early_v2_terminal_checker": sha256_file(Path(__file__).resolve())},
    }
    written, digest = write_immutable_json(output, body)
    return {"status": CHECKER_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-upstream-binder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(check(post_upstream_binder=args.post_upstream_binder,
                           output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
