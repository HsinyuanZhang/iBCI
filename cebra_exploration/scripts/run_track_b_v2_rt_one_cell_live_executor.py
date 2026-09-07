#!/usr/bin/env python3
"""Render one canonical RT one-cell no-target preflight; execution is sealed off."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


os.environ["PYTHONNOUSERSITE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_rt_one_cell_live_executor as live  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-fold-index", type=int, required=True,
                        choices=(live.PILOT_OUTER_FOLD_INDEX,))
    parser.add_argument("--seed", type=int, required=True, choices=(live.PILOT_SEED,))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-have-root-review", action="store_true")
    args = parser.parse_args(argv)
    if args.execute != args.i_have_root_review:
        parser.error("execution requires both --execute and --i-have-root-review")
    try:
        if args.execute:
            live.execution_is_not_authorized_this_successor(
                outer_fold_index=args.outer_fold_index, seed=args.seed,
            )
            raise AssertionError("unreachable")
        payload = live.build_no_target_preflight(
            outer_fold_index=args.outer_fold_index, seed=args.seed,
        )
    except (base.TrackBV2ContractError, live.TrackBV2RTOneCellError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
