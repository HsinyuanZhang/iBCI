#!/usr/bin/env python3
"""Run the source-only E03 posterior-q / held-carrier error audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "tfpd_exploration/src", ROOT / "sua_exploration"):
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
            "schema": "budget_matched_posterior_q_error_source_v1_plan",
            "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
            "result_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
                "c3_q_error_source_v1"
            ),
            "source_session_count": 27,
            "budgets": [4, 10, 30],
            "held_reference_positions": [30, 50],
            "source_authority_sha256": (
                "92f898aa828225b4805f2b23cc5f419f6c0f48cb6aaf54819ea884ab951a0791"
            ),
            "execute_requires": ["--execute", "--root-reviewed"],
        }, sort_keys=True, indent=2))
        return 0
    from budget_matched_posterior_cal_aug_c3_v1.q_error_source import execute_reviewed
    print(json.dumps(execute_reviewed(ROOT), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
