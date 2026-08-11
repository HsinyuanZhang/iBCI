#!/usr/bin/env python3
"""Receipt-only preparation check for H1 CI32/CI64 source-date LODO.

This program neither opens a recording/checkpoint nor starts a subprocess.  A
human must explicitly name ``H1-CI64-SLODO`` after the immutable five-date
H-S/H-C aggregate is complete.  The aggregate can satisfy an evidence
prerequisite but can never choose CI64 automatically.
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
    AGGREGATE_SCHEMA,
    AGGREGATE_STATUS,
    DATES,
    ROUTE_PREREQUISITE_STATUS,
)


LAUNCH_RECEIPT_SCHEMA = "h1_carrierid_date_lodo_ci_launch_receipt_v1"
LAUNCH_RECEIPT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_PREPARED_NOT_LAUNCHED"
PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_ci_cpu_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_SOURCE_ONLY_NOT_LAUNCHED"
ROUTE = "H1-CI64-SLODO"


class CiLaunchReceiptError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CiLaunchReceiptError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_immutable_json(path: str | Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          f"immutable mode-0444 receipt required: {candidate}")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == schema and body.get("status") == status,
          f"receipt schema/status drift: {candidate}")
    return candidate, body, _sha(candidate)


def validate_five_date_aggregate(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    """Validate completion evidence without interpreting its numerical values."""

    candidate, aggregate, digest = _read_immutable_json(path, schema=AGGREGATE_SCHEMA, status=AGGREGATE_STATUS)
    _need(aggregate.get("all_five_date_receipts_present_and_validated") is True,
          "five-date aggregate is not complete")
    _need(tuple(aggregate.get("required_outer_dates", ())) == DATES,
          "five-date aggregate does not bind canonical date order")
    route = aggregate.get("route_prerequisite")
    _need(isinstance(route, Mapping) and route.get("status") == ROUTE_PREREQUISITE_STATUS,
          "five-date aggregate lacks source/date screen completion")
    _need(route.get("automatic_route_selection") == "FORBIDDEN",
          "five-date aggregate must not select CI64 automatically")
    return candidate, aggregate, digest


def prepare(*, five_date_aggregate: Path, ci_preflights: Mapping[str, Path], explicit_route: str, output: Path) -> dict[str, Any]:
    _need(explicit_route == ROUTE, "human must explicitly name H1-CI64-SLODO; no automatic route selection exists")
    _need(not output.exists() and not os.path.lexists(str(output)), f"refusing to overwrite launch receipt: {output}")
    aggregate_path, _aggregate, aggregate_sha = validate_five_date_aggregate(five_date_aggregate)
    _need(tuple(ci_preflights) == DATES, "CI launch receipt needs exactly five canonical date preflights")
    dates: dict[str, Any] = {}
    for date in DATES:
        path, receipt, digest = _read_immutable_json(ci_preflights[date], schema=PREFLIGHT_SCHEMA, status=PREFLIGHT_STATUS)
        _need(receipt.get("outer_date") == date, f"CI preflight outer date drift: {date}")
        controls = receipt.get("source_controls")
        _need(isinstance(controls, Mapping) and controls.get("all_arms") == ["CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS"],
              f"CI preflight controls drift: {date}")
        _need(controls.get("same_source_windows") is True and controls.get("same_source_schedule") is True
              and controls.get("same_source_normalizer") is True, f"CI pairing drift: {date}")
        _need(receipt.get("scope", {}).get("target_recordings_opened") == 0
              and receipt.get("scope", {}).get("cuda_constructed_or_launched") is False, f"CI preflight scope drift: {date}")
        dates[date] = {"path": str(path), "sha256": digest, "source_binding_sha256": receipt["source_binding_sha256"]}
    payload = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "status": LAUNCH_RECEIPT_STATUS,
        "route": ROUTE,
        "explicit_operator_route": explicit_route,
        "five_date_aggregate": {"path": str(aggregate_path), "sha256": aggregate_sha,
                                "source_date_screen_complete": True, "numeric_results_interpreted": False},
        "ci_source_preflights": dates,
        "proposed_arms_per_date": ["CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS"],
        "training_contract": {"fresh_seed": 42, "fixed_terminal_epoch_zero_based": 49, "epochs": 50,
                              "warm_start_forbidden": True, "H64": "PROHIBITED"},
        "not_a_gpu_launcher": True,
        "launch_authorized": False,
        "next_action": "separate explicit operator launch; this receipt cannot start training",
        "scope": {"nwb_opened": False, "checkpoint_opened": False, "trainer_constructed_or_launched": False,
                  "gpu_constructed_or_launched": False, "target_data_opened": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(stat.S_IMODE(output.stat().st_mode) == 0o444, "CI launch receipt lost immutable mode")
    return {"status": LAUNCH_RECEIPT_STATUS, "receipt_path": str(output.resolve()), "receipt_sha256": _sha(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-date-aggregate", type=Path, required=True)
    parser.add_argument("--ci-preflight", action="append", metavar="DATE=PATH", required=True)
    parser.add_argument("--explicit-route", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping: dict[str, Path] = {}
    for item in args.ci_preflight:
        date, separator, path = item.partition("=")
        if not separator or date in mapping:
            raise SystemExit("each --ci-preflight must be one unique DATE=PATH")
        mapping[date] = Path(path)
    print(json.dumps(prepare(five_date_aggregate=args.five_date_aggregate, ci_preflights=mapping,
                             explicit_route=args.explicit_route, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
