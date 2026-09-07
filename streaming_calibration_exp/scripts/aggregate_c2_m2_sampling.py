#!/usr/bin/env python3
"""Fail-closed C2 sampling-objective aggregation from immutable cell receipts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.metrics import c2_m2_sampling as core


def _parse_receipt(value: str) -> tuple[str, Path]:
    try:
        key, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("receipt must be CELL_KEY=PATH") from exc
    return key, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-preflight", required=True, type=Path)
    parser.add_argument("--receipt", action="append", required=True, type=_parse_receipt)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=core.CROSSED_BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    preflight, preflight_digest = core.load_verified_immutable_json(args.official_preflight)
    if preflight.get("screen_id") != core.SCREEN_ID:
        raise SystemExit("C2 preflight screen_id drift")
    if preflight.get("implementation_bindings") != core.source_bindings(PROJECT.parent):
        raise SystemExit("C2 implementation changed after official preflight")
    receipts: dict[str, dict] = {}
    receipt_digests: dict[str, str] = {}
    for key, path in args.receipt:
        if key in receipts:
            raise SystemExit(f"Duplicate C2 cell receipt key: {key}")
        payload, digest = core.load_verified_immutable_json(path)
        core.refuse_sealed_sessions(
            [str(payload.get("session_name", ""))],
            label=key,
        )
        receipts[key] = payload
        receipt_digests[key] = digest
    aggregate = core.aggregate_receipts(
        receipts,
        official_preflight_sha256=preflight_digest,
        bootstrap_draws=args.bootstrap_draws,
    )
    aggregate["cell_receipt_sha256"] = receipt_digests
    digest = core.write_immutable_json(args.out, aggregate)
    print(
        json.dumps(
            {
                "receipt": str(args.out.resolve()),
                "sha256": digest,
                "passed": aggregate["passed"],
                "classification": aggregate["classification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
