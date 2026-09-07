#!/usr/bin/env python3
"""Print a no-target Track-B v2 Subject-M one-cell successor preflight.

This CLI has no target-path, output, score, GPU, CEBRA, checkpoint, or
normalizer option.  ``--execute`` exists only as an explicit tripwire and is
always rejected in this review-only successor.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_subject_m_one_cell_successor as successor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, choices=("sua", "pseudo_mua"))
    parser.add_argument("--outer-fold-id", required=True)
    parser.add_argument("--target-session-id", required=True,
                        help="Opaque ID only; this command accepts no target path or data.")
    parser.add_argument("--seed", required=True, choices=(42, 43, 44), type=int)
    parser.add_argument("--execute", action="store_true",
                        help="Rejected: this successor has no target execution mode.")
    args = parser.parse_args()
    try:
        if args.execute:
            successor.refuse_subject_m_one_cell_execution(
                view=args.view, outer_fold_id=args.outer_fold_id,
                target_session_id=args.target_session_id, cebra_seed=args.seed,
            )
        payload = successor.build_subject_m_one_cell_preflight(
            view=args.view, outer_fold_id=args.outer_fold_id,
            target_session_id=args.target_session_id, cebra_seed=args.seed,
        )
    except (base.TrackBV2ContractError, successor.TrackBV2SubjectMOneCellSuccessorError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
