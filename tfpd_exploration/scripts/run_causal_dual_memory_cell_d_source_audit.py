#!/usr/bin/env python3
"""Static dry CLI for the future CDM-D strict-source safety audit.

This entrypoint imports only the standard-library-only lifecycle module.  It
cannot open source data, build a model, initialize CUDA, create output, score,
or launch an audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_dry_plan() -> dict:
    package_parent = str(ROOT / "tfpd_exploration")
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    from src.causal_dual_memory_cell_d_v1.lifecycle import dry_plan

    return dry_plan(ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CDM-D source-audit static dry/preflight plan")
    parser.add_argument("--dry-run", action="store_true", help="render the inert source-audit contract")
    parser.add_argument("--preflight", action="store_true", help="render the inert source-audit contract")
    parser.add_argument("--execute", action="store_true", help="always rejected; requires future root authorization")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("execution is not implemented; a later root-reviewed authorization is required")
    payload = _load_dry_plan()
    payload["requested_mode"] = "preflight" if args.preflight else "dry"
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
