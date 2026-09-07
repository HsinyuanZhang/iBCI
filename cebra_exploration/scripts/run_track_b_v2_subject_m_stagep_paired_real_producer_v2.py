#!/usr/bin/env python3
"""Render the paired-real-producer v2 no-target review plan."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["PYTHONNOUSERSITE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_subject_m_stagep_paired_real_producer_v2 as producer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("v2 producer public CLI is a no-target review tripwire; use reviewed paired launcher")
    sys.stdout.write(json.dumps(producer.build_no_target_review_plan(), sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
