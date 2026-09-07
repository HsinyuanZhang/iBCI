#!/usr/bin/env python3
"""CLI for m2_b_small_stability_v1.

--stage0 runs CPU gates only.
--train refuses unless the coordinator later sets M2_SMALL_TRAIN=1.
This process must not claim a GPU or launch 24-epoch S0/S1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("PYTHONNOUSERSITE", "1")


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_m2_b_small_stability_v1")
    parser.add_argument("--stage0", action="store_true", help="CPU Stage0 gates")
    parser.add_argument("--train", action="store_true", help="formal 24-epoch train (refused unless env)")
    parser.add_argument("--score-ext4", action="store_true", help="96-view ext-4 scan after both summaries exist")
    parser.add_argument("--lr-contrast", action="store_true", help="score N0/N1 1e-4 pair in the LR-contrast root")
    parser.add_argument("--cell", default="S1-SMALL-COS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if int(args.stage0) + int(args.train) + int(args.score_ext4) != 1:
        parser.error("specify exactly one of --stage0, --train, --score-ext4")
    if args.score_ext4:
        from tfpd_exploration.src.m2_b_small_stability_v1 import config as cfg
        from tfpd_exploration.src.m2_b_small_stability_v1.score import LR_CELLS, run_ext4_scan

        if args.lr_contrast:
            report = run_ext4_scan(
                seed=args.seed,
                root=cfg.LR_CONTRAST_RUN_ROOT,
                cells=LR_CELLS,
                primary=cfg.PRIMARY_CANDIDATE_LR,
            )
        else:
            report = run_ext4_scan(seed=args.seed)
        print(json.dumps({"routing": report["routing"], "primary": report["primary_source_pick"]}, indent=2))
        return 0
    if args.train:
        from tfpd_exploration.src.m2_b_small_stability_v1.launch import run_train

        run_train(cell=args.cell, seed=args.seed)
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from tfpd_exploration.src.m2_b_small_stability_v1 import config as cfg
    from tfpd_exploration.src.m2_b_small_stability_v1.stage0 import run_stage0

    report = run_stage0(root=cfg.ACTIVE_RUN_ROOT)
    print(json.dumps({"status": report["status"], "root": report["root"], "decoder_params": report["decoder_params"]}, indent=2))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
