#!/usr/bin/env python3
"""Write the source-only C1 teacher compatibility receipt.

This is the root gate for C1.  It establishes, without touching target or
sealed data and without training, whether a CO-native teacher can be
constructed and loaded into the existing student contract, and records exactly
what differs from the current MC-Maze decoder initialization.

It does not authorize a GPU run.  Teacher/target mismatch remains a plausible
contributor, not an isolated cause.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
SCE_ROOT = REPO_ROOT / "streaming_calibration_exp"
for path in (REPO_ROOT, SUA_ROOT, SCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mc_maze import c1_teacher_compatibility as compat
from mc_maze import c1_teacher_domain_ablation as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--teacher", type=Path, default=core.MC_MAZE_TEACHER_PATH)
    parser.add_argument(
        "--tiny-architecture",
        action="store_true",
        help="Debug/tests only: use a tiny SpintModel instead of the production architecture.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = core.RESULT_ROOT
    out_path = args.out.expanduser().resolve() if args.out is not None else core.compatibility_receipt_path(result_root)
    work_dir = args.work_dir.expanduser().resolve() if args.work_dir is not None else out_path.parent / "compatibility_work"
    tiny = None
    use_production = not args.tiny_architecture
    if args.tiny_architecture:
        tiny = {
            "model_dim": 32,
            "num_covariates": 2,
            "window_size": 50,
            "num_heads": 2,
            "num_layers": 1,
            "num_id_layers": 2,
            "use_learnable_id": True,
            "learnable_id_type": "mlp",
            "learnable_rep": True,
            "dropout_rate": 0.0,
            "dynamic_dropout": True,
            "dynamic_dropout_low": 0.0,
            "dynamic_dropout_high": 1.0,
            "tf_drop_rate": 0.1,
            "readin_layer_type": "mlp",
        }
    try:
        body, sidecar, digest, receipt = compat.write_compatibility_receipt(
            out_path,
            work_dir=work_dir,
            teacher_path=args.teacher,
            use_production_architecture=use_production,
            tiny_architecture=tiny,
        )
    except (core.C1ContractError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(
        core.pretty_json_bytes(
            {
                "status": receipt["status"],
                "interface_constructible": receipt["interface_constructible"],
                "trained_co_native_teacher_exists": receipt["trained_co_native_teacher_exists"],
                "gpu_authorized": False,
                "source_only": True,
                "receipt": str(body),
                "sidecar": str(sidecar),
                "receipt_sha256": digest,
                "hypothesis_status": receipt["hypothesis_status"],
            }
        ).decode("utf-8"),
        end="",
    )
    return 0 if receipt["interface_constructible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
