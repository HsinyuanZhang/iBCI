#!/usr/bin/env python3
"""Matched C3-Const/C3-Real full-training entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "streaming_calibration_exp",
    ROOT / "sua_exploration",
    ROOT / "tfpd_exploration/src",
    ROOT / "tfpd_exploration",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("constant", "real"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-reviewed", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=12 * 60 * 60)
    args = parser.parse_args()
    if args.execute != args.root_reviewed:
        parser.error("--execute and --root-reviewed must be supplied together")
    if not args.execute:
        result = (
            "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
            f"c3_{args.arm}_full_v1"
        )
        print(json.dumps({
            "schema": "budget_matched_posterior_cal_aug_c3_full_v1_plan",
            "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
            "arm": args.arm,
            "result_root_relative": result,
            "smoke_terminal_sha256": (
                "f9dc00b09bf12d917415f1a78fb1a9eec780246aa76f680a33ec58575f0f98d9"
            ),
            "epochs": 48,
            "steps_per_epoch": 33925,
            "total_optimizer_steps": 1628400,
            "expected_total_budget_counts": {"30": 542800, "10": 542800, "4": 542800},
            "train_batch_size": 32,
        }, sort_keys=True, indent=2))
        return 0
    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")
    from budget_matched_posterior_cal_aug_c3_v1.full import execute_reviewed

    print(json.dumps(execute_reviewed(
        ROOT, arm=args.arm, device_text=args.device, num_workers=args.num_workers,
        timeout_seconds=args.timeout_seconds,
    ), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
