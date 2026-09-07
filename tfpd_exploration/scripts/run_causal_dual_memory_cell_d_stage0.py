#!/usr/bin/env python3
"""Static dry/preflight CLI for CDM-D Stage 0.

The CLI intentionally imports only the standard-library-only plan module.  It
has no authority, data, output, model, CUDA, scorer, or launch path.
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
    # Do not import causal_dual_memory_cell_d_v1.core here: it is intentionally
    # numerical-runtime code, while this public entrypoint must import no Torch.
    from src.causal_dual_memory_cell_d_v1.plan import dry_plan

    return dry_plan(ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CDM-D Stage-0 static dry/preflight plan")
    parser.add_argument("--dry-run", action="store_true", help="render the inert Stage-0 contract")
    parser.add_argument("--preflight", action="store_true", help="render the inert Stage-0 contract")
    parser.add_argument("--execute", action="store_true", help="always rejected; Stage 0 has no execution route")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("execution is not implemented; a later root-reviewed work order is required")
    payload = _load_dry_plan()
    payload["requested_mode"] = "preflight" if args.preflight else "dry"
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
