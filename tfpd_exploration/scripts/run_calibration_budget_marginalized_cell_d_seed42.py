#!/usr/bin/env python3
"""Static dry CLI; reviewed physical launch is wired after Phase-1 gate."""
from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        parser.error("physical CBM-D execution requires the reviewed in-process route")
    if not args.dry_run:
        parser.error("use --dry-run")
    from src.calibration_budget_marginalized_cell_d_v1 import dry_plan
    print(json.dumps(dry_plan(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

