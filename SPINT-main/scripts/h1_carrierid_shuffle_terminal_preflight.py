#!/usr/bin/env python3
"""Write a no-target, nonlaunch source closure for H-C/H-RS/H-LS evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import sha256_file, write_immutable_json


PREFLIGHT_SCHEMA = "h1_carrierid_h32_rs_ls_terminal_evaluator_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_RS_LS_TERMINAL_EVALUATOR_SOURCE_CLOSURE_NONLAUNCH"


def write_preflight(output_path: str | Path) -> dict[str, str]:
    """Hash the entire terminal evaluator closure without instantiating any data module."""

    files = (
        "scripts/h1_carrierid_shuffle_terminal_preflight.py",
        "scripts/h1_carrierid_shuffle_terminal_evaluate.py",
        "scripts/h1_carrierid_shuffle_preflight.py",
        "scripts/h1_carrierid_evaluate.py",
        "scripts/h1_carrierid_paired_launcher.py",
        "src/data/h1_m4_eb_pilot.py",
        "src/data/h1_m4_eb_normalized_v2.py",
        "src/data/h1_carrierid_shuffle.py",
        "src/models/h1_carrierid_module.py",
        "src/models/h1_carrierid_shuffle_module.py",
        "src/models/components/h1_carrierid_spint.py",
        "src/h1_m4_eb_normalized_v2_contract.py",
        "configs/experiment/h1_carrierid.yaml",
        "configs/experiment/h1_carrierid_full.yaml",
        "configs/experiment/h1_carrierid_shuffle.yaml",
        "configs/experiment/h1_carrierid_rs.yaml",
        "configs/experiment/h1_carrierid_ls.yaml",
        "configs/data/falcon_h1_m4_eb_normalized_v2.yaml",
        "configs/data/falcon_h1_carrierid_shuffle.yaml",
        "configs/model/falcon_h1_carrierid.yaml",
        "configs/model/falcon_h1_carrierid_shuffle.yaml",
        "configs/callbacks/h1_carrierid_terminal.yaml",
        "tests/test_h1_carrierid_shuffle_contract.py",
        "tests/test_h1_carrierid_shuffle_terminal_contract.py",
    )
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "scope": {
            "target_opened": False,
            "target_enumerated": False,
            "minival_opened": False,
            "formal_opened": False,
            "evalai_opened": False,
            "cuda_launched": False,
            "checkpoint_selected": False,
        },
        "frozen_evaluator_contract": {
            "required_terminal_epoch": 49,
            "required_epochs": 50,
            "prediction_dtype": "float32",
            "r2_accumulator_dtype": "float64",
            "canonical_h_c_target_variant": "full",
            "h_rs_target_variant": "row",
            "h_ls_target_variant": "label",
            "eval_no_grad": True,
            "state_hash_before_after": True,
            "target_optimizer_or_backward": 0,
        },
        "source_sha256": {relative: sha256_file(ROOT / relative) for relative in files},
        "launch": {
            "authorized": False,
            "requires": "root audit plus both H-RS/H-LS epoch_049 checkpoints before target access",
        },
    }
    path, digest = write_immutable_json(output_path, receipt)
    return {"status": PREFLIGHT_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write_preflight(args.output_path), sort_keys=True))


if __name__ == "__main__":
    main()
