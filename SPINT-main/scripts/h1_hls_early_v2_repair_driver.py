#!/usr/bin/env python3
"""Run the frozen H-LS replay after a pre-target import-path repair.

This driver does not alter the replay implementation or any data/checkpoint
binding.  It only supplies a fresh output root because the original attempt
left an immutable forensic terminal after stopping before target preflight.
The same transfer receipt, source checkpoints, support/query windows,
normalizer, manifest, fixed date order, and evaluator code are passed through
unchanged.  The caller must provide a Python wrapper that exports the repo
root on ``PYTHONPATH``; every child stage then runs through that wrapper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import h1_carrierid_date_lodo_hls_early_v2_remote5070_replay as replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-receipt", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--repair-root", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--aggregate-output", required=True, type=Path)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--gpu-device", default="0")
    args = parser.parse_args()
    plan = replay.build_plan(
        transfer_receipt=args.transfer_receipt,
        data_dir=args.data_dir,
        replay_root=args.repair_root,
        evaluation_dir=args.evaluation_dir,
        aggregate_output=args.aggregate_output,
        python_executable=args.python_executable,
    )
    print(json.dumps(replay.execute(plan, gpu_device=args.gpu_device), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
