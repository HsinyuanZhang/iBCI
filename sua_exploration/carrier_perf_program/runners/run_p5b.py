#!/usr/bin/env python3
"""P5b CLI — fit-confidence gate / shrink (CPU, train-free)."""
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
from carrier_perf.p5b_confidence_gate import build_p5b_audit, write_audit  # noqa: E402

DEFAULT_MANIFEST = SUA / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
DEFAULT_DATA_ROOT = SUA / "data" / "dandi_000688"
DEFAULT_FALCON = SUA.parent / "SPINT-main" / "data" / "000953"


def main() -> None:
    force_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
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
        print(json.dumps({"schema": "p5b_dry_run", "status": "dry_run", "no_gpu": True}, indent=2))
        return
    require_reviewed_cpu(args.execute)
    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    audit = build_p5b_audit(
        manifest_path=args.manifest,
        data_root=args.data_root,
        falcon_data_dir=args.falcon_data_dir,
        max_sessions=args.max_sessions,
    )
    path = write_audit(audit, args.output_dir)
    print(
        json.dumps(
            {
                "wrote": str(path),
                "status": audit["status"],
                "gate_gpu_film_authorized_by_this_audit": audit[
                    "gate_gpu_film_authorized_by_this_audit"
                ],
                "sua_aggregates": audit["sua"]["aggregates"],
                "falcon_aggregates": audit["falcon"]["aggregates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
