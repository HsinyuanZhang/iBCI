#!/usr/bin/env python3
"""Dry by default; execute the frozen M2 activity-budget screen explicitly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "schema": "m2_t4_activity_budget_screen_v1_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "cells_per_surface": [
                "ridge_static_m30",
                "ridge_static_m10",
                "ridge_activity30_m10",
                "ridge_static_m4",
                "ridge_activity30_m4",
            ],
            "surfaces": ["within_post30", "external_official_query"],
        }, sort_keys=True))
        return
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical import execute

    payload = execute(REPO_ROOT, gpu_index=args.gpu_index, batch_size=args.batch_size)
    print(json.dumps({"status": payload["status"], "paired_contrasts": payload["paired_contrasts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
