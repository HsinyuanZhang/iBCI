#!/usr/bin/env python3
"""P1 Stage A CLI — synthetic dry-run or source-only NWB audit (CPU)."""
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
from carrier_perf.p1_estimators import sweep_m_synthetic  # noqa: E402
from carrier_perf.p1_stage_a_audit import build_audit, write_audit  # noqa: E402

DEFAULT_MANIFEST = SUA / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
DEFAULT_DATA_ROOT = SUA / "data" / "dandi_000688"


def main() -> None:
    force_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="synthetic estimators only")
    parser.add_argument("--execute", action="store_true", help="source-only 27-session audit")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, help="fresh O_EXCL output directory")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--ms", type=int, nargs="+", default=None)
    parser.add_argument("--eval-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--gpu", action="store_true", help="refused")
    args = parser.parse_args()
    if args.gpu:
        parser.error("GPU flags are refused by carrier_perf_program")
    assert_no_gpu_flag(args.gpu)

    modes = sum([bool(args.dry_run), bool(args.execute)])
    if modes != 1:
        parser.error("choose exactly one of --dry-run / --execute")

    if args.dry_run:
        rows = sweep_m_synthetic(seed=args.seed, noise_std=args.noise_std)
        payload = {
            "schema": "carrier_perf_p1_stage_a_synthetic_v1",
            "status": "synthetic_cpu_only_estimators_only",
            "no_gpu": True,
            "no_nwb_opened": True,
            "rows": rows,
            "note": "Use --execute for the source-only rate-MSE / PV-R2 audit.",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    require_reviewed_cpu(args.execute)
    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    ms = tuple(args.ms) if args.ms is not None else (10, 20, 30, 40, 50)
    audit = build_audit(
        manifest_path=args.manifest,
        data_root=args.data_root,
        ms=ms,
        eval_trials=args.eval_trials,
        max_sessions=args.max_sessions,
    )
    path = write_audit(audit, args.output_dir)
    summary = {
        "wrote": str(path),
        "status": audit["status"],
        "n_sessions": audit["n_sessions"],
        "overall": audit["overall"],
        "predeclared_outlet": audit["predeclared_outlet"],
        "aggregate_per_m": [
            {
                "M": a["M"],
                "transduction_consistent": a["transduction_consistent"],
                "preferred": a["preferred_estimator_if_consistent"],
                "rate_mse_A_minus_B_mean": a["rate_mse_A_minus_B"]["mean"],
                "pv_r2_A_minus_B_mean": a["pv_r2_A_minus_B"]["mean"],
            }
            for a in audit["aggregate_per_m"]
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
