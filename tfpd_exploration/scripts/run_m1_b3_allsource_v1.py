#!/usr/bin/env python3
"""Public CLI for m1_b3_allsource_v1 (all-source B3 / B3S-rSyn3).

Default is dry. Live GPU training:

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \\
      python tfpd_exploration/scripts/run_m1_b3_allsource_v1.py --dry-run

    PYTHONNOUSERSITE=1 \\
      python tfpd_exploration/scripts/run_m1_b3_allsource_v1.py \\
        --execute --gpu-authorized --gpu-index 1 --pack \\
        --mode train --arm {b3,b3s_rsyn3,b3s_t4,b3s_t4_encoder,b3s_rsyn3_freeze,b3s_rsyn3_acyc}

Package (CPU, after train COMPLETE). TOP-4 arms reuse freeze/acyc checkpoints:

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \\
      python tfpd_exploration/scripts/run_m1_b3_allsource_v1.py \\
        --execute --mode package --arm b3s_rsyn3_freeze_top4

--pack is optional: omit for single occupancy. Wave-1/wave-2 launches use --pack on GPU 1.
--execute-gpu cannot mint a GPU capability.
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
    parser.add_argument("--mode", choices=("train", "package"), default=None)
    parser.add_argument(
        "--arm",
        choices=(
            "b3", "b3s_rsyn3", "b3s_t4", "b3s_t4_encoder",
            "b3s_rsyn3_freeze", "b3s_rsyn3_acyc",
            "b3s_rsyn3_freeze_top4", "b3s_rsyn3_acyc_top4",
        ),
        default=None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.execute_gpu:
        parser.error("cannot mint a GPU capability")
    if not arguments.execute:
        from tfpd_exploration.src.m1_b3_allsource_v1.plan import dry_plan

        payload = dry_plan()
        payload["cli"] = "run_m1_b3_allsource_v1.py"
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.mode is None or arguments.arm is None:
        parser.error("--execute requires --mode --arm")
    if arguments.mode == "train":
        if not arguments.gpu_authorized or arguments.gpu_index is None:
            parser.error("train requires --gpu-authorized --gpu-index")
        if arguments.gpu_index != 1:
            parser.error("this cell may only use GPU 1")
        if arguments.arm in ("b3s_rsyn3_freeze_top4", "b3s_rsyn3_acyc_top4"):
            parser.error("TOP-4 arms are package-only")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("execute requires PYTHONNOUSERSITE=1")
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tfpd_exploration.src.m1_b3_allsource_v1.execute import execute

    shas, terminal, failure = execute(
        root,
        mode=arguments.mode,
        arm=arguments.arm,
        gpu_index=arguments.gpu_index,
        gpu_authorized=bool(arguments.gpu_authorized),
        pack=bool(arguments.pack),
    )
    print(json.dumps(
        {"terminal_sha256": terminal, "failure_sha256": failure, "bodies": shas},
        sort_keys=True, separators=(",", ":"),
    ))
    return 0 if terminal and failure is None else 2


if __name__ == "__main__":
    sys.exit(main())
