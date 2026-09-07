#!/usr/bin/env python3
"""Print the Track-B v2 actual-CEBRA CPU-control contract without executing it.

This runner has no data-path, checkpoint, or CEBRA import.  ``--execute`` is
intentionally fail-closed until a separately reviewed live integration owns the
canonical adapters, sealed reference authority, and source-only selector.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


# Set process isolation before importing a project module.  This script never
# calls a CUDA API; the blank setting is receipt-visible in the contract only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
SRC = SCRIPT_DIR.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base
import track_b_v2_live_contract as live


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=base.ALLOWED_DATASETS)
    parser.add_argument("--view", choices=base.SUBJECT_M_VIEWS, default=None)
    parser.add_argument("--output-dimension", type=int, default=8)
    parser.add_argument("--source-iterations", type=int, default=10000)
    parser.add_argument("--normalized-lambda", type=float, default=1.0e-2)
    parser.add_argument("--knn-output-dimension", type=int, default=None)
    parser.add_argument("--knn-source-iterations", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Always refused in this scaffold.")
    args = parser.parse_args()
    if args.execute:
        raise live.TrackBV2LiveContractError(
            "actual CEBRA CPU control execution is not authorised by this scaffold"
        )
    linear_geometry = base.CandidateGeometry(
        output_dimension=args.output_dimension,
        source_iterations=args.source_iterations,
        normalized_lambda=args.normalized_lambda,
    )
    knn_geometry = base.KnnCandidateGeometry(
        output_dimension=args.knn_output_dimension or args.output_dimension,
        source_iterations=args.knn_source_iterations or args.source_iterations,
    )
    payload = live.build_actual_cebra_cpu_control_runner_contract(
        dataset=args.dataset,
        view=args.view,
        linear_ridge_scoring_geometry=linear_geometry,
        knn_scoring_geometry=knn_geometry,
        execution_requested=False,
        device="cpu",
    )
    payload["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
