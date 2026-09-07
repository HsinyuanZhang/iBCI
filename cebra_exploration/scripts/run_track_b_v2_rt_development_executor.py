#!/usr/bin/env python3
"""Print one no-target Track-B v2 RT executor/scorer dry plan."""
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
import track_b_v2_rt_development_executor as executor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-fold-id", required=True)
    parser.add_argument("--target-session-id", required=True,
                        help="Opaque RT session ID only; no target path or data is accepted.")
    args = parser.parse_args(argv)
    try:
        payload = executor.build_rt_development_executor_dry_plan(
            dataset="rt", view=None, outer_fold_id=args.outer_fold_id,
            target_session_id=args.target_session_id,
        )
    except (base.TrackBV2ContractError, executor.TrackBV2RTDevelopmentExecutorError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
