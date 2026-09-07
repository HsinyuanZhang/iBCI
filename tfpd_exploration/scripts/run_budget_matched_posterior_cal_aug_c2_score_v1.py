#!/usr/bin/env python3
"""Dry/execute entry point for C2 matched deployment scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
sys.path.insert(0, str(ROOT))

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({
            "schema": "budget_matched_posterior_cal_aug_c2_score_v1_plan",
            "status": "DRY_NO_DATA_NO_MODEL_NO_WRITE",
            "result_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c2_score_v1"
            ),
            "producer_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c2_full_v1"
            ),
            "surfaces": ["within", "external"],
            "budgets": [4, 10, 30],
            "review_drift_policy": "ACCEPTED_NON_NUMERIC_DRIFT",
        }, sort_keys=True))
        return 0
    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")
    from src.budget_matched_posterior_cal_aug_v1 import c2_score

    result = c2_score.execute_reviewed(REPO)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
