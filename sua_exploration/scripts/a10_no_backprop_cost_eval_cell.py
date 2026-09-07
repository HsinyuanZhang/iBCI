#!/usr/bin/env python3
"""Development-validation adaptation cell for A10 no-backprop cost experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA = REPO_ROOT / "sua_exploration"
sys.path.insert(0, str(SUA))

from mc_maze.gpu_contract_common import SUBC_VAL_SESSIONS, assert_sessions_not_sealed

AUTH_VALUE = "I_AUTHORIZE_A10_NO_BACKPROP_COST_GPU"
SCREEN_ID = "a10_no_backprop_cost_v1"
SUPERSEDED_REASON = (
    "A10 v1 is superseded and cannot launch: its arms do not use matched "
    "supervision and the adaptation implementation is incomplete"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["gradient_free", "readout_probe", "full_finetune"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-run-dir", required=True, type=Path)
    parser.add_argument("--out-path", required=True, type=Path)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    if args.launch:
        raise RuntimeError(SUPERSEDED_REASON)

    assert_sessions_not_sealed(SUBC_VAL_SESSIONS)

    payload = {
        "dry_run": not args.launch,
        "screen_id": SCREEN_ID,
        "arm": args.arm,
        "seed": args.seed,
        "source_run_dir": str(args.source_run_dir),
        "out_path": str(args.out_path),
        "validation_sessions": list(SUBC_VAL_SESSIONS),
        "protocol": {"calibration_n": 30, "evaluation_start_trial": 30},
    }
    if not args.launch:
        print(json.dumps(payload, indent=2))
        return

    raise AssertionError("unreachable superseded A10 launch path")


if __name__ == "__main__":
    main()
