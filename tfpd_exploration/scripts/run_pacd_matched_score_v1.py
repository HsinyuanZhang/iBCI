#!/usr/bin/env python3
"""Inert public descriptor: no PACD producer literals exist yet."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paired_anchored_calibration_dropout_score_v1 import plan

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="emit the inert target-free descriptor (default)")
    parsed = parser.parse_args(argv)
    if parsed.execute:
        print("PACD score V1 has no mintable live producer binding", file=sys.stderr)
        return 2
    print(json.dumps({"schema": plan.SCHEMA + "_dry", "systems": plan.SYSTEM_ORDER,
                      "surfaces": plan.SURFACE_ORDER, "budgets": plan.BUDGET_ORDER,
                      "regimes": plan.REGIME_ORDER, "producer_literals_deferred": True,
                      "no_torch_import": True, "score_authorized": False}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
