#!/usr/bin/env python3
"""Inert public CLI for the M2 post-fusion variant screen.

No argument can mint the root-owned capability or trigger data, checkpoint,
Torch, CUDA, or result-root access.  A future audited operator entry point
calls ``driver.execute_screen`` in-process with an opaque capability.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "tfpd_exploration")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="explicit alias for the inert default")
    parser.add_argument("--execute", action="store_true", help="always refused by the public CLI")
    args = parser.parse_args(argv)
    from tfpd_exploration.src.m2_postfusion_variant_screen_v1 import plan

    if args.execute:
        print(json.dumps({"schema": plan.SCHEMA, "status": "REFUSED__ROOT_CAPABILITY_REQUIRED"}, sort_keys=True))
        return 2
    print(json.dumps(plan.dry_plan(REPO_ROOT), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
