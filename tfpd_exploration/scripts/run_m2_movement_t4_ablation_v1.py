#!/usr/bin/env python3
"""Execute the frozen M2 movement-window T4 score ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    from tfpd_exploration.src.m2_movement_t4_ablation_v1 import physical
    if not args.execute:
        print(json.dumps({"schema": physical.SCHEMA + "_dry", "status": "DRY", "window_bins": [physical.START_BIN, physical.STOP_BIN]}))
        return
    try:
        result = physical.execute(ROOT, batch_size=args.batch_size)
    except BaseException as exc:
        physical.publish_failure(ROOT, exc)
        raise
    print(json.dumps({"status": result["status"], "move_minus_whole": result["move_minus_whole"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

