#!/usr/bin/env python3
"""Dry-only entry point for the PMC-D matched-score V3 successor.

The production route is intentionally not exposed by this command.  A later
root-reviewed in-process issuer must bind the exact SUBC/SUBM launch contract,
the immutable V2 input-stage failure graph, current provenance, V3 closure,
and a compatible device before invoking the physical wrapper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_score_module():
    root = Path(__file__).resolve().parents[2]
    package_root = root / "tfpd_exploration"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from src.posterior_marginalized_cell_d_v3 import matched_score

    return matched_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static PMC-D matched-score V3 successor plan")
    parser.add_argument("--dry-run", action="store_true", help="print the no-data/no-CUDA plan")
    parser.add_argument("--execute", action="store_true", help="rejected: no public execution capability")
    parser.add_argument("--root-reviewed", action="store_true", help="rejected: in-process capability only")
    args = parser.parse_args(argv)
    if args.execute or args.root_reviewed:
        parser.error("only --dry-run is supported; V3 requires a root-reviewed in-process capability")
    if argv not in (None, []) and not args.dry_run:
        parser.error("only --dry-run is supported")
    score = _load_score_module()
    sys.stdout.write(json.dumps(score.dry_plan(), sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
