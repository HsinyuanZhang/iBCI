#!/usr/bin/env python3
"""Dry by default; execute the contrast-only FiLM follow-up explicitly."""

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
        from tfpd_exploration.src.m2_contrast_only_squeeze_v1 import plan

        print(json.dumps({
            "schema": plan.SCHEMA + "_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "film_input": plan.FILM_INPUT,
            "configs": [row["name"] for row in plan.CONFIGS],
            "references": {
                "prior_contrast_only": plan.PRIOR_CONTRAST_ONLY_EXTERNAL,
                "winner_t4_plus_contrast": plan.WINNER_T4_PLUS_CONTRAST_EXTERNAL,
            },
            "workorder": plan.WORKORDER_RELATIVE,
        }, sort_keys=True))
        return
    from tfpd_exploration.src.m2_contrast_only_squeeze_v1.physical import execute

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
        "references": payload["references"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
