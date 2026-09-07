#!/usr/bin/env python3
"""Dry/execute CLI for preflight-complete C2/C3 score V3."""

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
            "schema": "budget_matched_posterior_cal_aug_c2_c3_posterior_score_v3_plan",
            "status": "DRY_NO_DATA_NO_MODEL_NO_WRITE",
            "result_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
                "c2_c3_posterior_score_v3"
            ),
            "failed_predecessors": ["c2_c3_posterior_score_v1", "c2_c3_posterior_score_v2"],
            "pre_attempt_checks": [
                "canonical SUBC_DATA_ROOT and SUBM_DATA_ROOT",
                "split src/models namespace origins",
                "strict execution closure and fresh V3 root",
            ],
            "review_drift_policy": "ACCEPTED_NON_NUMERIC_DRIFT",
        }, sort_keys=True, indent=2))
        return 0
    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")
    from budget_matched_posterior_cal_aug_c3_v1.score_v3 import execute_reviewed
    print(json.dumps(execute_reviewed(ROOT), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

