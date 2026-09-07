#!/usr/bin/env python3
"""CPU-only V3 source audit after the V2 support-ID failure."""

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
            "schema": "budget_matched_posterior_cal_aug_source_audit_v3_plan",
            "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
            "result_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/source_audit_v3"
            ),
            "support_id_missing_direction_encoding": "canonical_json_null_no_imputation",
            "gpu_smoke_authorized": False,
        }, sort_keys=True, indent=2))
        return 0
    from budget_matched_posterior_cal_aug_v1.source_audit_v3 import execute_reviewed
    print(json.dumps(execute_reviewed(ROOT), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

