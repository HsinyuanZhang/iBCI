#!/usr/bin/env python3
"""Render the paired real-producer review plan; execution remains disabled."""
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

import track_b_v2_subject_m_stagep_paired_real_producer as producer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-have-independent-root-review", action="store_true")
    args = parser.parse_args(argv)
    if args.execute != args.i_have_independent_root_review:
        parser.error("execution requires both --execute and --i-have-independent-root-review")
    try:
        if args.execute:
            producer.refuse_execution_before_target()
        plan = producer.build_no_target_review_plan()
        sys.stdout.write(json.dumps(plan, sort_keys=True, indent=2) + "\n")
        return 0
    except producer.TrackBV2SubjectMRealProducerError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

