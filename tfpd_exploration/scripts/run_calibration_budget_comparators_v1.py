#!/usr/bin/env python3
"""Static public entrypoint; reviewed execution is intentionally in-process."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src", ROOT / "sua_exploration"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.calibration_budget_comparators_v1 import dry_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit("reviewed comparator execution requires an in-process root capability")
    print(json.dumps(dry_plan(), sort_keys=True))


if __name__ == "__main__":
    main()
