#!/usr/bin/env python3
"""Inert CLI for the C2 Stage-0 CPU contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Deterministic local bootstrap for the additive package.  This executes before
# importing the package and is valid under ``python -S``; package ``__init__``
# and ``plan`` are both stdlib-only.
_TFPD_ROOT = Path(__file__).resolve().parents[1]
_SRC = _TFPD_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from budget_matched_posterior_cal_aug_v1 import plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        parser.error("Stage 0 is not executable; source audit and GPU launch are not authorized")
    print(json.dumps(plan.dry_payload(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
