#!/usr/bin/env python3
"""P3 Stage A CLI — seal RS4/LS4 baseline receipt (CPU, no inference)."""
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
from carrier_perf.p3_baseline_receipt import build_p3_stage_a_receipt, write_receipt  # noqa: E402


def main() -> None:
    force_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
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
                    "schema": "carrier_perf_p3_stage_a_dry_run_v1",
                    "status": "dry_run",
                    "no_gpu": True,
                    "note": "Execute seals receipt against frozen aggregates; no ckpt forward.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    require_reviewed_cpu(args.execute)
    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    receipt = build_p3_stage_a_receipt(spint_root=ROOT.parent.parent)
    path = write_receipt(receipt, args.output_dir)
    print(
        json.dumps(
            {
                "wrote": str(path),
                "status": receipt["status"],
                "structural_claim_rs4_and_ls4_below_z4": receipt[
                    "structural_claim_rs4_and_ls4_below_z4"
                ],
                "exact_means_match_sealed_aggregates": receipt[
                    "exact_means_match_sealed_aggregates"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
