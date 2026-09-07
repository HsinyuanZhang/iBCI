#!/usr/bin/env python3
"""Render the paired Subject-M Stage-P live-executor plan or refuse execution.

This command has no target/session/date/path/GPU/geometry/checkpoint/decoder
arguments.  The only possible future pilot is the root-predeclared pair
``SUA 20140307 seed42`` followed by its paired ``pMUA 20140307 seed42``.
Default mode renders a no-data plan.  ``--execute`` rebuilds the exact current
Stage-P live admission and always refuses before a target path, Torch, CEBRA,
or a GPU is reachable; it is included to test the future admission boundary,
not to start a run.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


os.environ["PYTHONNOUSERSITE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_subject_m_stagep_live_executor as live  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="Rebuild live admission then deliberately refuse before target/GPU/CEBRA access.")
    parser.add_argument("--i-have-independent-root-review", action="store_true",
                        help="Required together with --execute; still cannot enable this scaffold.")
    args = parser.parse_args(argv)
    if args.execute != args.i_have_independent_root_review:
        parser.error("--execute requires --i-have-independent-root-review, and vice versa")
    try:
        if args.execute:
            live.refuse_paired_stagep_live_execution()
            raise AssertionError("unreachable")
        payload = live.build_unadmitted_paired_execution_plan()
    except (base.TrackBV2ContractError, live.TrackBV2SubjectMStagePLiveExecutorError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
