#!/usr/bin/env python3
"""Write seed42 numeric tables for Astra. Seed43 rows are appended after those runs finish."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("PYTHONNOUSERSITE", "1")


def main() -> int:
    from tfpd_exploration.src.m2_b_small_stability_v1.astra_pack import write_seed42_pack

    dest = write_seed42_pack()
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
