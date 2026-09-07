#!/usr/bin/env python3
"""Inert public APFG V1 CLI; it cannot train or mint a live capability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import plan
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.driver import static_admission


def main() -> int:
    parser = argparse.ArgumentParser(description="APFG V1 inert Stage-0 plan")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dry-run", action="store_true", help="accepted inert alias; execution is unavailable")
    arguments = parser.parse_args()
    payload = plan.dry_plan(arguments.repo_root)
    payload["admission"] = static_admission(arguments.repo_root)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
