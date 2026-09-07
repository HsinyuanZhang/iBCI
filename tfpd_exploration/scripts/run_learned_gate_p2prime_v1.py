#!/usr/bin/env python3
"""Dry by default; execute one P2' oracle-policy decomposition stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.learned_gate_p2prime_v1 import plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=1, choices=(0, 1))
    parser.add_argument(
        "--stage", default="all",
        choices=("all", "attempt", "a", "substudy", "cop", "m30", "finalize", "smoke"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan.dry_plan(), sort_keys=True))
        return 0
    from src.learned_gate_p2prime_v1.physical import run_stage

    stages = (
        ("attempt", "a", "substudy", "cop", "m30", "finalize")
        if args.stage == "all" else (args.stage,)
    )
    if args.smoke:
        stages = ("attempt", "smoke") if args.stage == "all" else (args.stage,)
    results = {}
    for stage in stages:
        results[stage] = run_stage(ROOT, gpu_index=args.gpu_index, stage=stage, smoke=args.smoke)
        print(json.dumps({stage: results[stage]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
