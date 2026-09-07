#!/usr/bin/env python3
"""Print one no-target Track-B v2 development target/query authority plan."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_development_target_authority as authority  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("subject_m", "rt"))
    parser.add_argument("--view", choices=("sua", "pseudo_mua"))
    parser.add_argument("--outer-fold-id", required=True)
    parser.add_argument("--target-session-id", required=True, help="Opaque ID only; no target data are opened.")
    args = parser.parse_args()
    if args.dataset == "subject_m" and args.view is None:
        parser.error("subject_m requires --view")
    if args.dataset == "rt" and args.view is not None:
        parser.error("rt has no --view")
    try:
        payload = authority.build_development_target_query_authority(
            dataset=args.dataset, view=args.view, outer_fold_id=args.outer_fold_id,
            target_session_id=args.target_session_id,
        )
    except (base.TrackBV2ContractError, authority.TrackBV2DevelopmentTargetAuthorityError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
