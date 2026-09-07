#!/usr/bin/env python3
"""Fail-closed B1 factorial aggregation from immutable cell score receipts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.metrics import b1_m2_factorial as core


def _parse_receipt(value: str) -> tuple[str, Path]:
    try:
        key, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("receipt must be CELL_KEY=PATH") from exc
    return key, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("P", "F"), required=True)
    parser.add_argument("--official-preflight", required=True, type=Path)
    parser.add_argument("--receipt", action="append", required=True, type=_parse_receipt)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage-p-aggregate", type=Path, default=None)
    args = parser.parse_args()
    preflight, preflight_digest = core.load_verified_immutable_json(args.official_preflight)
    core.validate_preflight_payload(preflight)
    if preflight["implementation_bindings"] != core.source_bindings(PROJECT.parent):
        raise SystemExit("B1 implementation changed after official preflight")
    receipts: dict[str, dict] = {}
    receipt_digests: dict[str, str] = {}
    for key, path in args.receipt:
        if key in receipts:
            raise SystemExit(f"Duplicate B1 cell receipt key: {key}")
        payload, digest = core.load_verified_immutable_json(path)
        receipts[key] = payload
        receipt_digests[key] = digest
    stage = core.aggregate_stage(args.stage, receipts)
    if stage.get("official_preflight_sha256") != preflight_digest:
        raise SystemExit("B1 cell receipts do not bind this official immutable preflight")
    stage["cell_receipt_sha256"] = receipt_digests
    if args.stage == "F":
        if args.stage_p_aggregate is None:
            raise SystemExit("Stage F aggregate requires --stage-p-aggregate")
        prior, prior_digest = core.load_verified_immutable_json(args.stage_p_aggregate)
        core.validate_stage_p_aggregate(prior, official_preflight_sha256=preflight_digest)
        if prior.get("official_preflight_sha256") != preflight_digest:
            raise SystemExit("Stage-P aggregate preflight does not match Stage-F official preflight")
        if prior.get("expected_cell_keys") != sorted(cell.key for cell in core.cells_for_stage("P")):
            raise SystemExit("Stage-P aggregate lattice drift")
        if set((prior.get("cell_receipt_sha256") or {})) != {cell.key for cell in core.cells_for_stage("P")}:
            raise SystemExit("Stage-P aggregate has incomplete receipt lattice")
        final = core.aggregate_final(prior, stage)
        final["stage_p_aggregate_sha256"] = prior_digest
        final["stage_f_aggregate"] = stage
        stage = final
    digest = core.write_immutable_json(args.out, stage)
    print(json.dumps({"receipt": str(args.out.resolve()), "sha256": digest, "stage": stage["stage"]}, indent=2))


if __name__ == "__main__":
    main()
