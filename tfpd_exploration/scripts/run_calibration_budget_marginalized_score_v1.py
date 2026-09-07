#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("only the reviewed in-process scorer may execute")
    from src.calibration_budget_marginalized_score_v1 import dry_plan
    print(json.dumps(dry_plan(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

