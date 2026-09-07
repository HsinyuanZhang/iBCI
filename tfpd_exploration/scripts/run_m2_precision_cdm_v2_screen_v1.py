#!/usr/bin/env python3
"""Dry by default; explicitly run the native M2 Precision-CDM V2 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT / "tfpd_exploration", ROOT):
    text = str(value)
    if text not in sys.path: sys.path.insert(0, text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if not args.execute:
        from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import plan
        print(json.dumps({"schema": plan.SCHEMA, "status": "DRY_NO_DATA_NO_TORCH_NO_GPU_NO_WRITE",
                          "seed": plan.SEED, "cells": list(plan.CELL_ORDER),
                          "result_root": str(plan.result_root(ROOT))}, sort_keys=True))
        return 0
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1.physical import execute
    result = execute(ROOT, gpu_index=args.gpu_index, batch_size=args.batch_size)
    print(json.dumps({"status": result["status"], "result_root": result["result_root"]}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
