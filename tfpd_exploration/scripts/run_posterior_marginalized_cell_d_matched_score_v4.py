#!/usr/bin/env python3
"""Static/dry entry point for the PMC-D B128 matched-score V4 successor.

This command intentionally cannot mint the opaque in-process capability used
by :func:`execute_authorized_v4`.  Consequently even both public flags stop
before provenance, data, CUDA, or a result root can be reached.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FLAGS = {"--execute", "--root-reviewed"}


def _bootstrap(*, live: bool) -> None:
    """Install explicit route roots without relying on ambient PYTHONPATH."""
    project = str(ROOT / "tfpd_exploration")
    lane = str(ROOT / "tfpd_exploration/src")
    sys.path[:] = [item for item in sys.path if item not in {project, lane}]
    sys.path.insert(0, project)
    if live:
        sys.path.insert(0, lane)


def _contract():
    _bootstrap(live=False)
    from src.posterior_marginalized_cell_d_v4 import matched_score
    return matched_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="static PMC-D matched-score V4 plan")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and (args.execute or args.root_reviewed):
        parser.error("--dry-run cannot be combined with execution flags")
    if args.execute or args.root_reviewed:
        if not (args.execute and args.root_reviewed):
            parser.error("V4 requires both public flags and an in-process reviewed capability")
        _bootstrap(live=True)
        from src.posterior_marginalized_cell_d_v4 import matched_score_physical
        # The public process cannot construct this private object.  This call
        # is deliberately fail-closed before any root/provenance/data action.
        matched_score_physical.execute_authorized_v4(ROOT, capability=None)
        raise AssertionError("unreachable")
    if argv not in (None, []) and not args.dry_run:
        parser.error("only --dry-run is supported")
    print(json.dumps(_contract().dry_plan(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
