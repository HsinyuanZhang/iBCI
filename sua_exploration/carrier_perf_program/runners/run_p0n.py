#!/usr/bin/env python3
"""P0N CLI — CPU only. Default: dry-run."""
from __future__ import annotations

import os
import sys

# Blank CUDA before any third-party import that might touch torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carrier_perf.gpu_guard import assert_no_gpu_flag, force_cpu, require_reviewed_cpu  # noqa: E402
from carrier_perf.p0n_noise_floor import (  # noqa: E402
    build_p0n_report,
    dry_run_payload,
    write_report,
)


def main() -> None:
    force_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print scaffold dry-run JSON")
    parser.add_argument("--execute", action="store_true", help="audit a real aggregate (CPU)")
    parser.add_argument("--aggregate", type=Path, help="path to aggregate_m2.json")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        help="path to native_mua_t4_v1/logs with m2_{arm}_f*_s*.log",
    )
    parser.add_argument("--output-dir", type=Path, help="fresh O_EXCL output directory")
    parser.add_argument("--gpu", action="store_true", help="refused")
    args = parser.parse_args()
    if args.gpu:
        parser.error("GPU flags are refused by carrier_perf_program")
    assert_no_gpu_flag(args.gpu)

    modes = sum([args.dry_run, args.execute])
    if modes != 1:
        parser.error("choose exactly one of --dry-run / --execute")

    if args.dry_run:
        print(json.dumps(dry_run_payload(), indent=2, sort_keys=True))
        return

    require_reviewed_cpu(args.execute)
    if args.aggregate is None or args.output_dir is None:
        parser.error("--execute requires --aggregate and --output-dir")
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    report = build_p0n_report(aggregate, logs_dir=args.logs_dir)
    path = write_report(report, args.output_dir)
    print(json.dumps({"wrote": str(path), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
