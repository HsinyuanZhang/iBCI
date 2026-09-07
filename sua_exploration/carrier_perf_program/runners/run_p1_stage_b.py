#!/usr/bin/env python3
"""P1 Stage B CLI — FALCON held-in B vs kernel-smoothed C (CPU)."""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carrier_perf.gpu_guard import assert_no_gpu_flag, force_cpu, require_reviewed_cpu  # noqa: E402
from carrier_perf.p1_stage_b_audit import build_stage_b_audit, write_audit  # noqa: E402

DEFAULT_DATA_DIR = (
    ROOT.parent.parent / "SPINT-main" / "data" / "000953"
)


def main() -> None:
    force_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--ms", type=int, nargs="+", default=None)
    parser.add_argument("--eval-trials", type=int, default=16)
    parser.add_argument("--gpu", action="store_true", help="refused")
    args = parser.parse_args()
    if args.gpu:
        parser.error("GPU flags are refused by carrier_perf_program")
    assert_no_gpu_flag(args.gpu)

    modes = sum([bool(args.dry_run), bool(args.execute)])
    if modes != 1:
        parser.error("choose exactly one of --dry-run / --execute")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "carrier_perf_p1_stage_b_dry_run_v1",
                    "status": "dry_run",
                    "no_gpu": True,
                    "data_dir": str(args.data_dir),
                    "note": "Use --execute to audit held-in-calib NWBs only.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    require_reviewed_cpu(args.execute)
    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    ms = tuple(args.ms) if args.ms is not None else (10, 20, 30)
    audit = build_stage_b_audit(
        data_dir=args.data_dir,
        ms=ms,
        eval_trials=args.eval_trials,
        max_sessions=args.max_sessions,
    )
    path = write_audit(audit, args.output_dir)
    print(
        json.dumps(
            {
                "wrote": str(path),
                "status": audit["status"],
                "n_sessions": audit.get("n_sessions"),
                "stage_c_gpu_authorized_by_this_audit": audit.get(
                    "stage_c_gpu_authorized_by_this_audit"
                ),
                "aggregate_per_m": audit.get("aggregate_per_m"),
                "errors": audit.get("errors"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
