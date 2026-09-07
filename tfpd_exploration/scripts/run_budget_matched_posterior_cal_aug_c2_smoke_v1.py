#!/usr/bin/env python3
"""Bounded C2 runtime smoke; no performance claim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Insert in reverse precedence order.  The final order makes
# ``tfpd_exploration/src`` win the historical top-level ``src`` namespace,
# while still exposing top-level ``budget_matched_*`` and ``mc_maze``.
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-reviewed", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.execute != args.root_reviewed:
        parser.error("--execute and --root-reviewed must be supplied together")
    if not args.execute:
        print(json.dumps({
            "schema": "budget_matched_posterior_cal_aug_c2_smoke_v1_plan",
            "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
            "result_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c2_smoke_v1"
            ),
            "source_authority_sha256": (
                "92f898aa828225b4805f2b23cc5f419f6c0f48cb6aaf54819ea884ab951a0791"
            ),
            "smoke_steps": 120,
            "budget_cycle": [30, 10, 4],
            "performance_claim": False,
        }, sort_keys=True, indent=2))
        return 0
    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")
    from budget_matched_posterior_cal_aug_v1.c2_smoke import execute_reviewed
    print(json.dumps(execute_reviewed(ROOT, device_text=args.device, num_workers=args.num_workers), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
