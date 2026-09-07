#!/usr/bin/env python3
"""Dry by default; execute P1 contrast optimization / EvalAI-prep explicitly."""

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
        from tfpd_exploration.src.m2_p1_contrast_opt_v1 import plan

        print(json.dumps({
            "schema": plan.SCHEMA + "_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "evalai_push": False,
            "official_578221_heldout": plan.OFFICIAL_T4_578221_EXTERNAL,
            "sealed_p1_external": plan.SEALED_P1_EXTERNAL,
            "masks": list(plan.MASKS),
            "workorder": plan.WORKORDER_RELATIVE,
        }, sort_keys=True))
        return
    from tfpd_exploration.src.m2_p1_contrast_opt_v1.physical import execute

    payload = execute(REPO_ROOT, gpu_index=args.gpu_index, batch_size=args.batch_size)
    print(json.dumps({
        "status": payload["status"],
        "submit": payload["submit"],
        "cost": payload["cost"],
        "interpretability": {
            name: {"external_mean": body["external_mean"], "minus_p0": body["minus_p0"], "keep": body["keeps_80pct_of_p1_gain"]}
            for name, body in payload["interpretability"].items()
        },
        "m33_p0": payload["m33"]["p0"]["summaries"]["external_official_query"]["equal_session_mean"],
        "m33_p1": payload["m33"]["p1_transfer"]["summaries"]["external_official_query"]["equal_session_mean"],
        "m33_means": payload["m33"]["means_retrain"]["summaries"]["external_official_query"]["equal_session_mean"],
        "payload": payload["payload"],
        "evalai_push": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
