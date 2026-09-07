#!/usr/bin/env python3
"""Dry by default; execute the activity-only matched quick screen explicitly."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.causal_dual_memory_cell_d_activity_only_quick_v1 import plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=1, choices=(0, 1))
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan.dry_plan(), sort_keys=True))
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from src.causal_dual_memory_cell_d_activity_only_quick_v1.physical import execute
    print(json.dumps(execute(ROOT, gpu_index=args.gpu_index), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

