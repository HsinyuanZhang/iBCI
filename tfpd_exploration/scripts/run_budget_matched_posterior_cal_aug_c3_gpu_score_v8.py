#!/usr/bin/env python3
"""Run V8 accelerated full C2/C3 score."""

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
            "schema": "budget_matched_posterior_cal_aug_c2_c3_gpu_score_v8_plan",
            "status": "DRY_NO_DATA_NO_MODEL_NO_CUDA_NO_WRITE",
            "result_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
                "c2_c3_posterior_gpu_score_v8"
            ),
            "v7_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
                "c2_c3_posterior_gpu_parity_v7"
            ),
            "engineering_parity_tolerances": {
                "max_abs_prediction": 1e-5,
                "absolute_r2": 1e-6,
                "selected_after_v7_diagnostic": True,
                "scientific_gates_unchanged": True,
            },
            "expected_row_count": 252,
            "execute_requires": ["--execute", "--root-reviewed"],
        }, sort_keys=True, indent=2))
        return 0
    from budget_matched_posterior_cal_aug_c3_v1 import score_gpu_v8 as v8

    print(json.dumps(v8.execute_reviewed(ROOT), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

