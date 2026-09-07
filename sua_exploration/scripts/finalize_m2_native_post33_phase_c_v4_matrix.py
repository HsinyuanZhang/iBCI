#!/usr/bin/env python3
"""Write-once score-sealed Stage-A or full Phase-C matrix finalizer."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    finalize_matrix_score_sealed,
    finalize_stage_a_score_sealed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scope", choices=("stage-a", "full"), required=True)
    args = parser.parse_args()
    if args.scope == "stage-a":
        output = finalize_stage_a_score_sealed(args.root)
    else:
        output = finalize_matrix_score_sealed(args.root)
    print(output)


if __name__ == "__main__":
    main()
