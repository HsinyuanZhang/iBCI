#!/usr/bin/env python3
"""Run one frozen official M1 fold through the activity-headroom matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE",
            "folds": [0, 1, 2],
            "arms": ["STATIC_SUPPORT", "ROLLING_FIXED_M", "CAUSAL_GROWING_CAP30", "FULL_SESSION_ORACLE"],
        }, sort_keys=True))
        return
    if args.fold is None or args.output is None:
        parser.error("--execute requires --fold and --output")
    from tfpd_exploration.src.m1_h1_activity_headroom_v1.m1 import write_once
    from tfpd_exploration.src.m1_h1_activity_headroom_v1.m1_breadth import run_official_fold
    payload = run_official_fold(ROOT, fold=args.fold, device=args.device)
    path, digest = write_once(args.output, payload)
    print(json.dumps({"status": payload["status"], "path": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
