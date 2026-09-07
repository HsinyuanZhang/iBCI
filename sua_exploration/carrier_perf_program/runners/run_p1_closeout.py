#!/usr/bin/env python3
"""P1 closeout CLI — bootstrap CI + SUA/FALCON fit-variance (CPU)."""
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
from carrier_perf.p1_closeout import build_p1_closeout, write_audit  # noqa: E402

DEFAULT_P1A = ROOT / "results" / "t4_estimator_equivalence_v1" / "audit.json"
DEFAULT_MANIFEST = SUA / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
DEFAULT_DATA_ROOT = SUA / "data" / "dandi_000688"
DEFAULT_FALCON = SUA.parent / "SPINT-main" / "data" / "000953"


def main() -> None:
    force_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--p1a-audit", type=Path, default=DEFAULT_P1A)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--falcon-data-dir", type=Path, default=DEFAULT_FALCON)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--gpu", action="store_true", help="refused")
    args = parser.parse_args()
    assert_no_gpu_flag(args.gpu)
    if sum([args.dry_run, args.execute]) != 1:
        parser.error("choose exactly one of --dry-run / --execute")
    if args.dry_run:
        print(json.dumps({"schema": "p1_closeout_dry_run", "status": "dry_run", "no_gpu": True}, indent=2))
        return
    require_reviewed_cpu(args.execute)
    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    audit = build_p1_closeout(
        p1a_audit_path=args.p1a_audit,
        manifest_path=args.manifest,
        data_root=args.data_root,
        falcon_data_dir=args.falcon_data_dir,
        max_sessions=args.max_sessions,
    )
    path = write_audit(audit, args.output_dir)
    fv = audit["fit_variance"]
    print(
        json.dumps(
            {
                "wrote": str(path),
                "status": audit["status"],
                "falcon_over_sua": fv["falcon_over_sua_mean_residual_var"],
                "w3_conditional_reopen_authorized": fv["w3_conditional_reopen_authorized"],
                "bootstrap_m30_mse": next(
                    (x for x in audit["bootstrap"]["per_m"] if x["M"] == 30), None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
