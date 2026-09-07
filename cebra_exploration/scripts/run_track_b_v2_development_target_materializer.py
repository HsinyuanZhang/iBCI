#!/usr/bin/env python3
"""Print a Track-B v2 target-materializer dry plan; never opens target data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_development_target_materializer as materializer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("subject_m", "rt"))
    parser.add_argument("--view", choices=("sua", "pseudo_mua"))
    parser.add_argument("--outer-fold-id", required=True)
    parser.add_argument("--target-session-id", required=True,
                        help="Opaque ID only; no target path or array is accepted.")
    args = parser.parse_args()
    if args.dataset == "subject_m" and args.view is None:
        parser.error("subject_m requires --view")
    if args.dataset == "rt" and args.view is not None:
        parser.error("rt has no --view")
    try:
        payload = materializer.build_development_target_materializer_dry_plan(
            dataset=args.dataset, view=args.view, outer_fold_id=args.outer_fold_id,
            target_session_id=args.target_session_id,
        )
    except (base.TrackBV2ContractError, materializer.TrackBV2DevelopmentTargetMaterializerError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
