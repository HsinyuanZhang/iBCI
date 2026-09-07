#!/usr/bin/env python3
"""P4 CLI — harmonic dispersion strata (CPU)."""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUA = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SUA))

from carrier_perf.gpu_guard import assert_no_gpu_flag, force_cpu, require_reviewed_cpu  # noqa: E402
from carrier_perf.p4_harmonic_strata import build_p4_audit, write_audit  # noqa: E402

DEFAULT_MANIFEST = SUA / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
DEFAULT_DATA_ROOT = SUA / "data" / "dandi_000688"


def main() -> None:
    force_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--gpu", action="store_true", help="refused")
    args = parser.parse_args()
    assert_no_gpu_flag(args.gpu)
    if sum([args.dry_run, args.execute]) != 1:
        parser.error("choose exactly one of --dry-run / --execute")
    if args.dry_run:
        print(json.dumps({"schema": "p4_dry_run", "status": "dry_run", "no_gpu": True}, indent=2))
        return
    require_reviewed_cpu(args.execute)
    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    audit = build_p4_audit(
        manifest_path=args.manifest,
        data_root=args.data_root,
        max_sessions=args.max_sessions,
    )
    path = write_audit(audit, args.output_dir)
    print(
        json.dumps(
            {
                "wrote": str(path),
                "status": audit["status"],
                "n_sessions": audit["n_sessions"],
                "monotonic_gain_vs_dispersion": audit["monotonic_gain_vs_dispersion"],
                "transduction_consistent_harmonic_over_fund": audit[
                    "transduction_consistent_harmonic_over_fund"
                ],
                "gate_harmonic_gpu": audit["gate_harmonic_gpu"],
                "gate_reason": audit["gate_reason"],
                "stratum_stats": audit["stratum_stats"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
