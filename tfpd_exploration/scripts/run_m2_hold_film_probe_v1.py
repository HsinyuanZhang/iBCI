#!/usr/bin/env python3
"""Dry by default; execute the M2 hold-vs-reach FiLM probe explicitly."""

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
        from tfpd_exploration.src.m2_hold_film_probe_v1 import plan

        print(json.dumps({
            "schema": plan.SCHEMA + "_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "checkpoint_sha256": plan.CHECKPOINT_SHA256,
            "arms": ["p0_zero_film_t4", "p1_trained_film", "p1_shuffle_score", "c2_train_on_shuffle"],
            "surfaces": ["within_post30", "external_official_query"],
            "activity_horizon": plan.ACTIVITY_HORIZON,
            "side_dim": plan.SIDE_DIM,
            "post_pool": "64+4",
            "tta": False,
            "evalai_push": False,
            "workorder": plan.WORKORDER_RELATIVE,
        }, sort_keys=True))
        return
    from tfpd_exploration.src.m2_hold_film_probe_v1.physical import execute

    payload = execute(REPO_ROOT, gpu_index=args.gpu_index, batch_size=args.batch_size)
    print(json.dumps({
        "status": payload["status"],
        "verdict": payload["verdict"],
        "arms": {
            name: body["summaries"]["external_official_query"]
            for name, body in payload["arms"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
