#!/usr/bin/env python3
"""Dry-by-default entry point for low-cost calibration diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tfpd_exploration"), str(ROOT / "tfpd_exploration/src"), str(ROOT / "sua_exploration")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-reviewed", action="store_true")
    args = parser.parse_args()
    from src import low_cost_calibration_v1 as route
    if not args.execute and not args.root_reviewed:
        print(json.dumps(route.dry_plan(), sort_keys=True))
        return
    if not (args.execute and args.root_reviewed):
        parser.error("execution requires both --execute and --root-reviewed")
    result = route.execute_reviewed(ROOT)
    print(json.dumps({key: result[key] for key in ("receipt_sha256", "terminal_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
