#!/usr/bin/env python3
"""Fail-closed Phase-C exact-set verifier with score values kept opaque."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey,
    PROTOCOL_ID,
    verify_cell_exact,
    verify_matrix_exact,
    verify_stage_a_exact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scope", choices=("cell", "stage-a", "full"), required=True)
    parser.add_argument("--arm", choices=("spint", "t4"))
    parser.add_argument("--fold", type=int, choices=range(7))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    if args.scope == "cell":
        if args.arm is None or args.fold is None or args.seed is None:
            parser.error("cell scope requires --arm, --fold, and --seed")
        report = verify_cell_exact(
            args.root,
            CellKey(PROTOCOL_ID, args.arm, args.fold, args.seed),
        )
        # Do not display a commitment hash or any evaluator payload metadata.
        report = {"status": "PASS_EXACT_SCORE_SEALED_CELL_V4", "cell": report["cell"]}
    elif args.scope == "stage-a":
        report = verify_stage_a_exact(args.root)
    else:
        report = verify_matrix_exact(args.root)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
