#!/usr/bin/env python3
"""P2 RT go-cue coverage audit CLI (CPU)."""
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
from carrier_perf.p2_rt_gocue_audit import build_rt_gocue_audit, write_audit  # noqa: E402

DEFAULT_DATA_ROOT = ROOT.parent / "data" / "dandi_000688"


def main() -> None:
    force_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", action="store_true", help="refused")
    args = parser.parse_args()
    assert_no_gpu_flag(args.gpu)
    if sum([args.dry_run, args.execute]) != 1:
        parser.error("choose exactly one of --dry-run / --execute")
    if args.dry_run:
        print(json.dumps({"schema": "rt_gocue_dry_run", "status": "dry_run", "no_gpu": True}, indent=2))
        return
    require_reviewed_cpu(args.execute)
    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    audit = build_rt_gocue_audit(data_root=args.data_root)
    path = write_audit(audit, args.output_dir)
    print(
        json.dumps(
            {
                "wrote": str(path),
                "status": audit["status"],
                "n_sessions": audit["n_sessions"],
                "n_eligible": audit["n_eligible"],
                "n_ineligible": audit["n_ineligible"],
                "eligible_sessions": audit["eligible_sessions"],
                "ineligible_sessions": audit["ineligible_sessions"],
                "all_sessions_have_single_target_dir": audit["all_sessions_have_single_target_dir"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
