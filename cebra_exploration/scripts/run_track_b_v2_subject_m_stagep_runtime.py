#!/usr/bin/env python3
"""Render or refuse the predeclared Subject-M Stage-P primary runtime cell.

The command intentionally takes only a paired Stage-P view.  It cannot accept
a target date, target path, asset digest, output path, encoder geometry,
decoder, CEBRA seed, checkpoint, normalizer, GPU setting, or score override.
``--execute`` is an explicit hard tripwire: this turn does not open a target,
import/fit CEBRA, score, use a GPU, or mint a receipt.
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
import track_b_v2_subject_m_stagep_runtime as runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, choices=("sua", "pseudo_mua"),
                        help="Predeclared paired Stage-P view; date and seed are not caller-selectable.")
    parser.add_argument("--execute", action="store_true",
                        help="Always rejected in this no-target, pre-cost-review successor.")
    args = parser.parse_args()
    try:
        if args.execute:
            runtime.refuse_stagep_primary_execution(view=args.view)
        payload = runtime.build_stagep_primary_preflight(view=args.view)
    except (base.TrackBV2ContractError, runtime.TrackBV2SubjectMStagePRuntimeError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
