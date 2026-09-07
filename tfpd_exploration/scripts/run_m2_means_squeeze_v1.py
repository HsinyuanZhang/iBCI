#!/usr/bin/env python3
"""Dry by default; execute the native-M33 FiLM squeeze explicitly."""

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
        from tfpd_exploration.src.m2_means_squeeze_v1 import plan

        print(json.dumps({
            "schema": plan.SCHEMA + "_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "evalai_push_this_cell": plan.EVALAI_PUSH_THIS_CELL,
            "shuffle_is_reject": plan.SHUFFLE_IS_REJECT,
            "p0_m33_external": plan.P0_M33_EXTERNAL,
            "means_m33_baseline": plan.MEANS_M33_BASELINE,
            "official_578221_heldout": plan.OFFICIAL_T4_578221_EXTERNAL,
            "configs": [row["name"] for row in plan.CONFIGS],
            "workorder": plan.WORKORDER_RELATIVE,
        }, sort_keys=True))
        return
    from tfpd_exploration.src.m2_means_squeeze_v1.physical import execute

    payload = execute(REPO_ROOT, gpu_index=args.gpu_index, batch_size=args.batch_size)
    print(json.dumps({
        "status": payload["status"],
        "submit": payload["submit"],
        "arms": {
            name: {
                "external_mean": body["external_mean"],
                "external_median": body["external_median"],
                "shuffle_external_mean": body["shuffle_external_mean"],
                "passes_floor": body["passes_floor"],
            }
            for name, body in payload["arms"].items()
        },
        "payload": payload["payload"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
