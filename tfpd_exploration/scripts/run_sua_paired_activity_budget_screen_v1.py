#!/usr/bin/env python3
"""Dry by default; execute the frozen paired-view activity-budget screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    arguments = parser.parse_args()
    if not arguments.execute:
        print(json.dumps({
            "schema": "sua_paired_activity_budget_screen_v1_dry",
            "status": "DRY_NO_NWB_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "views": ["sua", "pseudo_mua"],
            "seeds": [42, 43, 44],
            "new_cells": [
                "ols_m10_activity10",
                "ridge_m4_activity4",
                "ridge_m4_activity30",
            ],
            "reused_reference": "ols_m10_activity30_reference",
        }, sort_keys=True))
        return
    from tfpd_exploration.src.sua_paired_activity_budget_screen_v1.physical import execute

    result = execute(
        REPO_ROOT, gpu_index=arguments.gpu_index, batch_size=arguments.batch_size
    )
    print(json.dumps({
        "status": result["status"],
        "elapsed_seconds": result["elapsed_seconds"],
        "new_forward_row_count": result["new_forward_row_count"],
        "paired_contrasts": result["paired_contrasts"],
        "paired_granularity_contrasts": result["paired_granularity_contrasts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

