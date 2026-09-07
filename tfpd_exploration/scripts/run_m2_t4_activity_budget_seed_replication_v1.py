#!/usr/bin/env python3
"""Dry by default; run M2 activity-budget seed replication explicitly."""

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
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "schema": "m2_t4_activity_budget_seed_replication_v1_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "seeds": [42, 43, 44],
            "new_forward_rows": 130,
            "reused_rows": 65,
        }, sort_keys=True))
        return
    from tfpd_exploration.src.m2_t4_activity_budget_seed_replication_v1.physical import execute

    payload = execute(REPO_ROOT, gpu_index=args.gpu_index, batch_size=args.batch_size)
    print(json.dumps({
        "status": payload["status"],
        "row_count": payload["row_count"],
        "across_seed_delta_summaries": payload["across_seed_delta_summaries"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
