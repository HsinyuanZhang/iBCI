#!/usr/bin/env python3
"""Inert public entry point; a root-owned in-process capability is required."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import plan

def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute: parser.error("execute is unavailable: no producer literals/capability")
    print(json.dumps({"schema": plan.SCHEMA, "status": "INERT_READY_REQUIRES_OPAQUE_CAPABILITY",
                      "authority": plan.validate_static(Path(__file__).resolve().parents[2]), "expected_rows": plan.EXPECTED_ROWS}, sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
