#!/usr/bin/env python3
"""Static no-execution entry point for CS-WG Stage 0.

This script deliberately imports only the standard library.  It cannot load
the numerical core, data, a model, a checkpoint, CUDA, or any future root;
the Stage-0 implementation is CPU/synthetic review material only.
"""
from __future__ import annotations

import argparse
import json


WORKORDER_SHA256 = "225fccec7588e28c25d1e4c3240066eb896fcd32c48e11236aaa8d45f8cb491d"
CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "stage0_cpu_synthetic_only",
        "workorder_sha256": WORKORDER_SHA256,
        "held_in_outer_loso_targets": ["20120924", "20120926", "20120927", "20120928"],
        "fixed_m1_source_decoder_recipe": {
            "calibration_budget": "M10",
            "calibration_trials": 10,
            "b3s_max_trial_length": 1024,
            "calibration_shape_per_row": [10, 1024, 64],
            "epochs": 20,
            "optimizer": "Adam",
            "lr": 1.0e-5,
            "weight_decay": 0.0,
            "scheduler": "None",
            "total_batch_size": 32,
            "final_bin_only_raw_output_mse": True,
        },
        "future_four_fold_matched_erm_plus_cswg_training_runs": 8,
        "historical_baseline_fold_wall_minutes": 68.8,
        "historical_reference_minutes_for_eight_runs_if_each_matches_baseline": 550.4,
        "actual_cswg_runtime_requires_measured_physical_receipts": True,
        "hidden_inner_grid_or_sweep_forbidden": True,
        "execution_authorized": False,
        "opens_source_or_target": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "trains": False,
        "scores": False,
        "launches": False,
        "future_execution_requirement": "root-reviewed source-only physical lifecycle not implemented in Stage 0",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print static Stage-0 contract only")
    parser.add_argument("--execute", action="store_true", help="always rejected at Stage 0")
    parser.add_argument("--train", action="store_true", help="always rejected at Stage 0")
    args = parser.parse_args(argv)
    if args.execute or args.train:
        parser.error("CS-WG Stage 0 is CPU/synthetic-only and has no execution route")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
