#!/usr/bin/env python3
"""Receipt-only preparation boundary for H1-EST4-SLODO source training.

It cannot start a process, open a recording/checkpoint, or inspect numerical
results.  A completed five-date H-S/H-C aggregate is an evidence prerequisite
only; a human must still explicitly name the EST4 route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_five_date_aggregate import (
    AGGREGATE_SCHEMA, AGGREGATE_STATUS, DATES, ROUTE_PREREQUISITE_STATUS,
)
from src.data.h1_carrierid_date_lodo_est4 import EST4_ARMS, EST4_PREFLIGHT_SCHEMA, EST4_PREFLIGHT_STATUS


ROUTE = "H1-EST4-SLODO"
LAUNCH_RECEIPT_SCHEMA = "h1_carrierid_date_lodo_est4_launch_receipt_v1"
LAUNCH_RECEIPT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_EST4_PREPARED_NOT_LAUNCHED"


class Est4LaunchReceiptError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4LaunchReceiptError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_json(path: Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    candidate = path.resolve()
    _need(candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          f"immutable mode-0444 receipt required: {candidate}")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == schema and body.get("status") == status,
          f"receipt schema/status drift: {candidate}")
    return candidate, body, _sha(candidate)


def prepare(*, five_date_aggregate: Path, est4_preflights: Mapping[str, Path], explicit_route: str, output: Path) -> dict[str, Any]:
    _need(explicit_route == ROUTE, "human must explicitly name H1-EST4-SLODO")
    _need(not output.exists() and not os.path.lexists(str(output)), "refusing to overwrite EST4 launch receipt")
    aggregate_path, aggregate, aggregate_sha = _immutable_json(
        five_date_aggregate, schema=AGGREGATE_SCHEMA, status=AGGREGATE_STATUS,
    )
    route = aggregate.get("route_prerequisite")
    _need(tuple(aggregate.get("required_outer_dates", ())) == DATES
          and aggregate.get("all_five_date_receipts_present_and_validated") is True
          and isinstance(route, Mapping) and route.get("status") == ROUTE_PREREQUISITE_STATUS
          and route.get("automatic_route_selection") == "FORBIDDEN",
          "EST4 launch receipt requires completed non-selecting 5/5 source screen")
    _need(tuple(est4_preflights) == DATES, "EST4 requires exactly five canonical source preflights")
    rows: dict[str, Any] = {}
    for date in DATES:
        path, body, digest = _immutable_json(est4_preflights[date], schema=EST4_PREFLIGHT_SCHEMA, status=EST4_PREFLIGHT_STATUS)
        controls, source = body.get("source_controls"), body.get("source_binding")
        _need(body.get("outer_date") == date and isinstance(controls, Mapping) and isinstance(source, Mapping),
              f"EST4 preflight date/source binding drift: {date}")
        _need(controls.get("all_arms") == list(EST4_ARMS)
              and controls.get("same_hs_hc_source_partition") is True
              and controls.get("same_source_windows") is True and controls.get("same_source_schedule") is True
              and controls.get("same_source_normalizer") is True and controls.get("target_score_selection") == "FORBIDDEN",
              f"EST4 preflight fails matched H-S/H-C control: {date}")
        scope = body.get("scope", {})
        _need(scope.get("target_recordings_opened") == 0 and scope.get("target_bytes_read") == 0
              and scope.get("cuda_constructed_or_launched") is False, f"EST4 preflight scope drift: {date}")
        rows[date] = {"path": str(path), "sha256": digest,
                      "source_binding_sha256": body.get("source_binding_sha256"),
                      "frozen_estimator_initialization": body.get("frozen_estimator_initialization")}
    payload = {
        "schema": LAUNCH_RECEIPT_SCHEMA, "status": LAUNCH_RECEIPT_STATUS,
        "route": ROUTE, "explicit_operator_route": explicit_route,
        "five_date_aggregate": {"path": str(aggregate_path), "sha256": aggregate_sha,
                                "source_date_screen_complete": True, "numeric_results_interpreted": False},
        "est4_source_preflights": rows, "proposed_arms_per_date": list(EST4_ARMS),
        "training_contract": {"fresh_seed": 42, "epochs": 50, "fixed_terminal_epoch_zero_based": 49,
                              "warm_start_forbidden": True, "target_score_selection": "FORBIDDEN",
                              "deployment_target_optimizer_steps": 0, "deployment_target_backward_steps": 0},
        "not_a_gpu_launcher": True, "launch_authorized": False,
        "scope": {"nwb_opened": False, "checkpoint_opened": False, "trainer_constructed_or_launched": False,
                  "gpu_constructed_or_launched": False, "target_data_opened": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(stat.S_IMODE(output.stat().st_mode) == 0o444, "EST4 launch receipt lost immutable mode")
    return {"status": LAUNCH_RECEIPT_STATUS, "receipt_path": str(output.resolve()), "receipt_sha256": _sha(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-date-aggregate", type=Path, required=True)
    parser.add_argument("--est4-preflight", action="append", metavar="DATE=PATH", required=True)
    parser.add_argument("--explicit-route", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping: dict[str, Path] = {}
    for item in args.est4_preflight:
        date, separator, path = item.partition("=")
        if not separator or date in mapping:
            raise SystemExit("each --est4-preflight must be one unique DATE=PATH")
        mapping[date] = Path(path)
    print(json.dumps(prepare(five_date_aggregate=args.five_date_aggregate, est4_preflights=mapping,
                             explicit_route=args.explicit_route, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

