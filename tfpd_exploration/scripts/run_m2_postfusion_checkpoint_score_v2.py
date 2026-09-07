#!/usr/bin/env python3
"""Inert public entry point for the V2 Post-Fusion scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfpd_exploration.src.m2_postfusion_checkpoint_score_v2 import plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("execute is unavailable: root-owned opaque V2 capability required")
    print(json.dumps({"schema": plan.SCHEMA, "status": "INERT_READY_REQUIRES_OPAQUE_CAPABILITY",
                      "authority": plan.validate_static(Path(__file__).resolve().parents[2])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
