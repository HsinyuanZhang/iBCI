#!/usr/bin/env python3
"""Render Track-B v2's RT 15-fold source-only authority plan; write nothing."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_source_adapter as source  # noqa: E402


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    plan = source.build_rt_15fold_source_authority_plan()
    sys.stdout.buffer.write(base.canonical_json_bytes(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
