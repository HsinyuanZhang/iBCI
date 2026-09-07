#!/usr/bin/env python3
"""Run the GPU-accelerated C2/C3 posterior-input V4 score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "tfpd_exploration",
    ROOT / "tfpd_exploration/src",
    ROOT / "sua_exploration",
    ROOT / "streaming_calibration_exp/src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-reviewed", action="store_true")
    args = parser.parse_args()
    if args.execute != args.root_reviewed:
        parser.error("--execute and --root-reviewed must be supplied together")
    if not args.execute:
        print(json.dumps({
            "schema": "budget_matched_posterior_cal_aug_c2_c3_gpu_score_v4_plan",
            "status": "DRY_NO_DATA_NO_MODEL_NO_CUDA_NO_WRITE",
            "result_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
                "c2_c3_posterior_gpu_score_v4"
            ),
            "visible_device": "1",
            "expected_gpu_uuid": "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
            "gpu_batch_candidates": [1024, 512, 128],
            "parity_tolerances": {"max_abs_prediction": 2e-6, "absolute_r2": 2e-7},
            "execute_requires": ["--execute", "--root-reviewed"],
        }, sort_keys=True, indent=2))
        return 0
    from budget_matched_posterior_cal_aug_c3_v1.score_gpu_v4 import execute_reviewed
    print(json.dumps(execute_reviewed(ROOT), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
