#!/usr/bin/env python3
"""Static dry-plan entry point for the PMC-D matched scorer.

The public command never imports Torch, resolves a terminal/SWA pathname,
opens evaluation data, creates a result root, or initializes CUDA.  Physical
scoring requires a later in-process root-reviewed capability and is rejected
even when an operator passes an execution-looking flag.
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
    from src.posterior_marginalized_cell_d_v1 import matched_score

    return matched_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Static PMC-D matched-score plan; live scoring is deliberately unavailable."
    )
    parser.add_argument("--dry-run", action="store_true", help="print the no-data/no-CUDA plan")
    parser.add_argument("--execute", action="store_true", help="rejected: no public execution capability exists")
    parser.add_argument(
        "--root-reviewed", action="store_true",
        help="rejected: only a later in-process capability may score",
    )
    args = parser.parse_args(argv)
    if args.execute or args.root_reviewed:
        parser.error("only --dry-run is supported; PMC-D matched scoring is scaffold-only")
    if argv not in (None, []) and not args.dry_run:
        parser.error("only --dry-run is supported")
    score = _load_score_module()
    sys.stdout.write(json.dumps(score.dry_plan(), sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
