#!/usr/bin/env python3
"""Print a no-target Track-B v2 subject-M executor/scorer dry plan."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_subject_m_development_executor as executor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, choices=("sua", "pseudo_mua"))
    parser.add_argument("--outer-fold-id", required=True)
    parser.add_argument("--target-session-id", required=True,
                        help="Opaque ID only; this dry plan never accepts a target path or data.")
    args = parser.parse_args()
    try:
        payload = executor.build_subject_m_development_executor_dry_plan(
            dataset="subject_m", view=args.view, outer_fold_id=args.outer_fold_id,
            target_session_id=args.target_session_id,
        )
    except (base.TrackBV2ContractError, executor.TrackBV2SubjectMDevelopmentExecutorError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
