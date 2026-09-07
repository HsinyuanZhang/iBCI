#!/usr/bin/env python3
"""Receipt-only post-upstream binder for early-trained H1 H-LS terminals.

This is the hard boundary between early source training and any target route.
It cannot run until the complete immutable H-S/H-C five-date aggregate exists.
For every date it proves that the original sealed H-C evaluation, the matched
pair preflight, the Phase-1 base source binding, and the early H-LS terminal
all refer to the same source/date experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES, EARLY_LAUNCH_SCHEMA, EARLY_LAUNCH_STATUS, EARLY_PREFLIGHT_SCHEMA,
    EARLY_PREFLIGHT_STATUS, EARLY_TERMINAL_SCHEMA, EARLY_TERMINAL_STATUS,
    PAIR_SCHEMA, PAIR_STATUS, POST_BINDER_SCHEMA, POST_BINDER_STATUS,
    need, read_immutable_json, sha256_file, write_immutable_json,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    UPSTREAM_AGGREGATE_SCHEMA, UPSTREAM_AGGREGATE_STATUS,
)
from src.h1_m4_cce_contract import canonical_sha256


ORIGINAL_EVALUATION_SCHEMA = "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1"


def _read_original_evaluation(upstream: Mapping[str, Any], date: str) -> tuple[Path, dict[str, Any], str]:
    row = upstream.get("per_date", {}).get(date, {}).get("receipt")
    need(isinstance(row, Mapping), f"complete upstream aggregate lacks original evaluation: {date}")
    path = Path(str(row.get("path", ""))).resolve()
    need(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444,
         f"original H-C evaluation receipt is not immutable: {date}")
    digest = sha256_file(path)
    need(digest == row.get("sha256"), f"original H-C evaluation mutated after aggregate: {date}")
    body = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(body, dict) and body.get("schema") == ORIGINAL_EVALUATION_SCHEMA
         and body.get("status") == f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{date}_HS_HC_EVALUATED"
         and body.get("outer_date") == date,
         f"original H-C evaluation schema/status/date drift: {date}")
    return path, body, digest


def bind(*, upstream_aggregate: Path, early_launch_receipt: Path,
         early_terminals: Mapping[str, Path], output: Path) -> dict[str, Any]:
    upstream_path, upstream, upstream_sha = read_immutable_json(
        upstream_aggregate, schema=UPSTREAM_AGGREGATE_SCHEMA, status=UPSTREAM_AGGREGATE_STATUS,
    )
    need(tuple(upstream.get("required_outer_dates", ())) == DATES
         and upstream.get("all_five_date_receipts_present_and_validated") is True,
         "early-v2 post binder requires the complete exact five-date upstream aggregate")
    route = upstream.get("route_prerequisite")
    need(isinstance(route, Mapping) and route.get("status") == "source/date screen complete"
         and route.get("automatic_route_selection") == "FORBIDDEN",
         "early-v2 upstream aggregate is not the non-selecting source/date-complete gate")
    launch_path, launch, launch_sha = read_immutable_json(
        early_launch_receipt, schema=EARLY_LAUNCH_SCHEMA, status=EARLY_LAUNCH_STATUS,
    )
    need(tuple(launch.get("fixed_grid", ())) == DATES
         and launch.get("does_not_claim_upstream_complete") is True
         and launch.get("target_gate") == "CLOSED_UNTIL_POST_UPSTREAM_BINDER",
         "early-v2 launch receipt policy drift")
    need(tuple(early_terminals) == DATES,
         "early-v2 post binder requires exactly the ordered complete five-date terminal grid")

    rows: dict[str, Any] = {}
    for date in DATES:
        launch_row = launch.get("source_preflights", {}).get(date)
        need(isinstance(launch_row, Mapping), f"early-v2 launch lacks preflight row: {date}")
        preflight_path, preflight, preflight_sha = read_immutable_json(
            launch_row.get("path", ""), schema=EARLY_PREFLIGHT_SCHEMA, status=EARLY_PREFLIGHT_STATUS,
        )
        need(preflight_sha == launch_row.get("sha256") and preflight.get("outer_date") == date,
             f"early-v2 source preflight changed after launch: {date}")
        terminal_path, terminal, terminal_sha = read_immutable_json(
            early_terminals[date], schema=EARLY_TERMINAL_SCHEMA, status=EARLY_TERMINAL_STATUS,
        )
        need(terminal.get("outer_date") == date and terminal.get("arm") == "H-LS"
             and terminal.get("source_preflight", {}).get("path") == str(preflight_path)
             and terminal.get("source_preflight", {}).get("sha256") == preflight_sha
             and terminal.get("source_binding_sha256") == preflight.get("source_binding_sha256")
             and terminal.get("base_source_binding_sha256") == preflight.get("matched_h_c_base_source_binding_sha256"),
             f"early-v2 terminal/preflight/source binding drift: {date}")
        scope, updates = terminal.get("scope"), terminal.get("deployment_updates")
        need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
             and scope.get("target_bytes_read") == 0
             and isinstance(updates, Mapping) and updates.get("target_optimizer_steps") == 0
             and updates.get("target_backward_steps") == 0
             and updates.get("target_model_state_updated") is False,
             f"early-v2 terminal records target access/update: {date}")

        pair_row = preflight.get("matched_h_c_pair_preflight")
        need(isinstance(pair_row, Mapping), f"early-v2 preflight lacks matched pair: {date}")
        pair_path, pair, pair_sha = read_immutable_json(
            pair_row.get("path", ""), schema=PAIR_SCHEMA, status=PAIR_STATUS,
        )
        need(pair_sha == pair_row.get("sha256") and pair.get("outer_date") == date,
             f"matched H-C pair mutated or date drifted: {date}")
        base = preflight.get("matched_h_c_base_source_binding")
        need(isinstance(base, Mapping)
             and preflight.get("matched_h_c_base_source_binding_sha256") == canonical_sha256(base)
             and pair.get("source_binding") == base
             and pair.get("source_binding_sha256") == canonical_sha256(base),
             f"early-v2 matched pair/base source binding drift: {date}")

        original_path, original, original_sha = _read_original_evaluation(upstream, date)
        hc = original.get("checkpoints", {}).get("H-C")
        need(isinstance(hc, Mapping) and isinstance(hc.get("metadata"), Mapping),
             f"original H-C evaluation lacks checkpoint metadata: {date}")
        hc_meta = hc["metadata"]
        need(hc_meta.get("arm") == "H-C" and hc_meta.get("outer_date") == date
             and hc_meta.get("phase2_source_binding_sha256") == canonical_sha256(base)
             and hc_meta.get("phase1_source_manifest_sha256") == base.get("source_manifest_sha256")
             and hc_meta.get("phase1_preflight_sha256") == base.get("preflight_sha256")
             and hc_meta.get("target_optimizer_steps") == 0 and hc_meta.get("target_backward_steps") == 0
             and hc_meta.get("checkpoint_warm_start") is False,
             f"complete upstream H-C is incompatible with early-v2 matched source binding: {date}")
        need(original.get("source_manifest_sha256") == base.get("source_manifest_sha256"),
             f"original H-C evaluator/source manifest drift: {date}")
        rows[date] = {
            "early_source_preflight": {"path": str(preflight_path), "sha256": preflight_sha},
            "early_terminal": {"path": str(terminal_path), "sha256": terminal_sha,
                               "source_binding_sha256": terminal["source_binding_sha256"]},
            "matched_h_c_pair_preflight": {"path": str(pair_path), "sha256": pair_sha},
            "base_source_binding_sha256": canonical_sha256(base),
            "original_h_c_evaluation": {"path": str(original_path), "sha256": original_sha},
            "compatibility": {
                "pair_equals_early_base_source_binding": True,
                "upstream_h_c_equals_pair_source_binding": True,
                "phase1_source_manifest_equal": True,
                "phase1_preflight_equal": True,
                "early_terminal_target_scope_zero": True,
            },
        }
    body = {
        "schema": POST_BINDER_SCHEMA, "status": POST_BINDER_STATUS,
        "route": "H1-HLS-EARLY-V2-POST-UPSTREAM",
        "upstream_aggregate": {"path": str(upstream_path), "sha256": upstream_sha},
        "early_launch_receipt": {"path": str(launch_path), "sha256": launch_sha},
        "fixed_grid": list(DATES), "dates": rows,
        "all_five_dates_compatible": True,
        "target_gate_transition": "SOURCE_TERMINALS_MAY_ENTER_NO_TARGET_TERMINAL_CHECKER; TARGET_NOT_OPENED",
        "scope": {"nwb_opened": 0, "target_recordings_opened": 0, "target_bytes_read": 0,
                  "checkpoint_opened": 0, "trainer_constructed": False, "cuda_constructed": False,
                  "target_evaluator_run": False},
        "code_sha256": {"post_upstream_binder": sha256_file(Path(__file__).resolve())},
    }
    written, digest = write_immutable_json(output, body)
    return {"status": POST_BINDER_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-aggregate", required=True, type=Path)
    parser.add_argument("--early-launch-receipt", required=True, type=Path)
    parser.add_argument("--early-terminal", action="append", required=True, metavar="DATE=PATH")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    mapping: dict[str, Path] = {}
    for item in args.early_terminal:
        date, separator, value = str(item).partition("=")
        if not separator or date in mapping:
            raise SystemExit("--early-terminal requires unique DATE=PATH values")
        mapping[date] = Path(value)
    print(json.dumps(bind(upstream_aggregate=args.upstream_aggregate,
                          early_launch_receipt=args.early_launch_receipt,
                          early_terminals=mapping, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
