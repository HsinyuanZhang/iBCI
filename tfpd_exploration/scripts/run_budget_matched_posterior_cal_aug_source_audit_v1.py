#!/usr/bin/env python3
"""CPU-only source authority audit for E02/E03."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TFPD_SRC = ROOT / "tfpd_exploration/src"
SUA = ROOT / "sua_exploration"
for path in (str(TFPD_SRC), str(SUA)):
    if path not in sys.path:
        sys.path.insert(0, path)

from budget_matched_posterior_cal_aug_v1 import plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-reviewed", action="store_true")
    args = parser.parse_args()
    if args.execute != args.root_reviewed:
        parser.error("--execute and --root-reviewed must be supplied together")
    if not args.execute:
        # Keep the public dry boundary stdlib-only under ``python -S``.  The
        # executable module (NumPy + NWB parser) is imported only after both
        # mutating flags are present.
        print(json.dumps({
            "schema": "budget_matched_posterior_cal_aug_source_audit_v1_plan",
            "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
            "result_root_relative": (
                "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/source_audit"
            ),
            "budgets": [4, 10, 30],
            "strict_source_session_count": 27,
            "manifest_sha256": (
                "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
            ),
            "execute_requires": ["--execute", "--root-reviewed"],
            "gpu_smoke_authorized": False,
        }, sort_keys=True, indent=2))
        return 0
    from budget_matched_posterior_cal_aug_v1.source_audit import execute_reviewed
    result = execute_reviewed(ROOT)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
