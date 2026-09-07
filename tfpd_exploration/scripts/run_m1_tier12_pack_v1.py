#!/usr/bin/env python3
"""Public CLI for m1_tier12_pack_v1 (C1 + chunk-CDM + SWA + W32).

Default is dry. Live GPU execution:

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \\
      python tfpd_exploration/scripts/run_m1_tier12_pack_v1.py --dry-run

    PYTHONNOUSERSITE=1 \\
      python tfpd_exploration/scripts/run_m1_tier12_pack_v1.py \\
        --execute --gpu-authorized --gpu-index {0,1} --pack \\
        --mode {chunk_cdm,train,score} --arm {t0,c1,t0_swa,c1_swa,t0_w32,c1_w32}

--pack is optional: omit for single occupancy (refuses any compute app).
Wave A/B launches use --pack. --execute-gpu cannot mint a GPU capability.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=None, choices=(0, 1))
    parser.add_argument("--pack", action="store_true",
                        help="allow a second same-route occupant (PACK_LIMIT=2). "
                             "Optional: omit for single occupancy.")
    parser.add_argument("--mode", choices=("chunk_cdm", "train", "score"), default=None)
    parser.add_argument("--arm", choices=("t0", "c1", "t0_swa", "c1_swa", "t0_w32", "c1_w32"),
                        default=None)
    parser.add_argument("--no-held-in", action="store_true",
                        help="score 20120924 only (default scores held-in sessions after 20120924)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.execute_gpu:
        parser.error("cannot mint a GPU capability")
    if not arguments.execute:
        from tfpd_exploration.src.m1_tier12_pack_v1.plan import dry_plan

        payload = dry_plan()
        payload["cli"] = "run_m1_tier12_pack_v1.py"
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    if not arguments.gpu_authorized or arguments.gpu_index is None:
        parser.error("--execute requires --gpu-authorized --gpu-index")
    if arguments.mode is None or arguments.arm is None:
        parser.error("--execute requires --mode --arm")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("execute requires PYTHONNOUSERSITE=1")
    from tfpd_exploration.src.m1_tier12_pack_v1.execute import execute

    held_in = not bool(arguments.no_held_in)
    root = Path(__file__).resolve().parents[2]
    shas, terminal, failure = execute(
        root,
        mode=arguments.mode,
        arm=arguments.arm,
        gpu_index=int(arguments.gpu_index),
        gpu_authorized=True,
        pack=bool(arguments.pack),
        held_in=held_in,
    )
    print(json.dumps(
        {"terminal_sha256": terminal, "failure_sha256": failure, "bodies": shas},
        sort_keys=True, separators=(",", ":"),
    ))
    return 0 if terminal and failure is None else 2


if __name__ == "__main__":
    sys.exit(main())
