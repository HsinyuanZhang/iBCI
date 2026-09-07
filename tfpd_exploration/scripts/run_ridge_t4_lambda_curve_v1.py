#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT / "tfpd_exploration" / "src", ROOT / "tfpd_exploration", ROOT / "sua_exploration"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from src.ridge_t4_lambda_curve_v1 import dry_plan


if __name__ == "__main__":
    print(json.dumps(dry_plan(), sort_keys=True))

