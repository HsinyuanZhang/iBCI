#!/usr/bin/env python3
"""Dry/execute CLI for the failure-bound C2/C3 score V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDERED_PATHS = (
    ROOT / "tfpd_exploration",
    ROOT / "tfpd_exploration/src",
    ROOT / "sua_exploration",
    ROOT / "streaming_calibration_exp/src",
)
for path in ORDERED_PATHS:
    while str(path) in sys.path:
        sys.path.remove(str(path))
for path in reversed(ORDERED_PATHS):
    sys.path.insert(0, str(path))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if args.execute != args.root_reviewed:
        parser.error("--execute and --root-reviewed must be supplied together")
    if not args.execute:
        print(json.dumps({
            "schema": "budget_matched_posterior_cal_aug_c2_c3_posterior_score_v2_plan",
            "status": "DRY_NO_DATA_NO_MODEL_NO_WRITE",
            "result_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
                "c2_c3_posterior_score_v2"
            ),
            "failed_v1_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
                "c2_c3_posterior_score_v1"
            ),
            "failed_v1_attempt_sha256": (
                "79ef7ef20e00c1cd15d7d711958da51d316fff3d7b7efdd2722b818e9b40b365"
            ),
            "failed_v1_failure_sha256": (
                "3d2dedce0b38938264ed912130f635c0a3666e90630b8eb776bff1d1ca0e9aef"
            ),
            "surfaces": ["within", "external"],
            "budgets": [4, 10, 30],
            "numerical_arms": ["c2", "c3_constant", "c3_real", "c3_real_q_shuffle"],
            "posterior_input_required": True,
            "namespace_repair": "independent models namespace",
            "review_drift_policy": "ACCEPTED_NON_NUMERIC_DRIFT",
        }, sort_keys=True, indent=2))
        return 0
    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")
    from budget_matched_posterior_cal_aug_c3_v1.score_v2 import execute_reviewed
    print(json.dumps(execute_reviewed(ROOT), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
