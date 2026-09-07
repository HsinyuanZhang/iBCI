#!/usr/bin/env python3
"""Write a static, no-target closure receipt for H1 D-S4/D-Q4 terminal evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import sha256_file, write_immutable_json
from scripts.h1_carrierid_distribution_terminal_checker import (
    TERMINAL_PREFLIGHT_SCHEMA,
    TERMINAL_PREFLIGHT_STATUS,
)


def write_preflight(output_path: str | Path) -> dict[str, str]:
    """Hash only code/config/test closure.  No DataModule or target is imported."""

    files = (
        "scripts/h1_carrierid_distribution_terminal_preflight.py",
        "scripts/h1_carrierid_distribution_terminal_checker.py",
        "scripts/h1_carrierid_distribution_terminal_evaluate.py",
        "src/data/h1_carrierid_distribution_target.py",
        "src/data/h1_carrierid_distribution.py",
        "src/models/h1_carrierid_distribution_module.py",
        "src/models/h1_carrierid_module.py",
        "src/models/components/h1_carrierid_spint.py",
        "src/data/h1_m4_eb_pilot.py",
        "src/h1_m4_eb_normalized_v2_contract.py",
        "scripts/h1_carrierid_distribution_preflight.py",
        "configs/data/falcon_h1_carrierid_distribution.yaml",
        "configs/model/falcon_h1_carrierid_distribution.yaml",
        "configs/experiment/h1_carrierid_distribution.yaml",
        "configs/experiment/h1_carrierid_distribution_s4.yaml",
        "configs/experiment/h1_carrierid_distribution_q4.yaml",
        "configs/callbacks/h1_carrierid_terminal.yaml",
        "tests/test_h1_carrierid_distribution_contract.py",
        "tests/test_h1_carrierid_distribution_terminal_contract.py",
        "docs/H1_CARRIERID_FRESH_MATCHED_DISTRIBUTION_PROTOCOL.md",
    )
    source_sha = {relative: sha256_file(ROOT / relative) for relative in files}
    receipt = {
        "schema": TERMINAL_PREFLIGHT_SCHEMA,
        "status": TERMINAL_PREFLIGHT_STATUS,
        "scope": {
            "source_or_target_recordings_opened": 0,
            "target_opened": False,
            "target_enumerated": False,
            "minival_or_formal_or_evalai_opened": False,
            "cuda_launched": False,
            "trainer_constructed": False,
            "checkpoint_created": False,
        },
        "contract": {
            "arms": ["D-S4", "D-Q4"],
            "checkpoint_requirement": "both independently trained fixed epoch49 checkpoints must pass checker before target open",
            "target_comparison_label": "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT",
            "one_shot_target_evaluation": True,
        },
        "source_sha256": source_sha,
        "launch": {"authorized": False, "target_evaluation_called": False},
    }
    path, digest = write_immutable_json(output_path, receipt)
    return {"status": TERMINAL_PREFLIGHT_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_preflight(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
