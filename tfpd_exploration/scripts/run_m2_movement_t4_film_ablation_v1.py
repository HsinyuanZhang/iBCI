#!/usr/bin/env python3
"""Run one seed of FiLM controls on M2 MOVE-T4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--execute", action="store_true"); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--batch-size", type=int, default=1024); args = parser.parse_args()
    from tfpd_exploration.src.m2_movement_t4_film_ablation_v1 import plan
    if not args.execute:
        print(json.dumps({"schema": plan.SCHEMA + "_dry", "status": "DRY", "seed": args.seed})); return
    from tfpd_exploration.src.m2_movement_t4_film_ablation_v1.physical import execute_seed, publish_failure
    try:
        result = execute_seed(ROOT, seed=args.seed, batch_size=args.batch_size)
    except BaseException as exc:
        publish_failure(ROOT, seed=args.seed, exc=exc); raise
    print(json.dumps({"status": result["status"], "seed": result["seed"], "external": {name: arm["matched"]["summaries"]["external_official_query"]["equal_session_mean"] for name, arm in result["arms"].items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

