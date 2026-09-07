#!/usr/bin/env python3
"""Write the static/no-target terminal closure for H1 D-S4e/D-Q4e.

This is deliberately only a code/configuration receipt.  It neither imports a
DataModule nor enumerates an NWB directory.  The first permitted target access
is reserved for the separately invoked evaluator, after the two real fixed
epoch-49 checkpoints have passed the source-only pair checker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import sha256_file, write_immutable_json
from scripts.h1_carrierid_distribution_exposure_terminal_checker import (
    TERMINAL_PREFLIGHT_SCHEMA,
    TERMINAL_PREFLIGHT_STATUS,
)


# These gates are deliberately duplicated from the immutable source CPU receipt
# rather than estimated after target access.  D-Q4e remains a query-label
# leakage diagnostic even in the estimator-limited branch.
SEALED_H_C_POOLED_R2 = 0.5255107931
VALIDITY_MIN_D_S4E_R2 = 0.4955107931
DELTA_GATE = 0.03


def frozen_terminal_branch_gates() -> dict[str, object]:
    return {
        "sealed_h_c_pooled_r2": SEALED_H_C_POOLED_R2,
        "validity": {
            "all_required": True,
            "d_s4e_pooled_r2_min": VALIDITY_MIN_D_S4E_R2,
            "formula": "0.5255107931 - 0.03",
            "per_target_recording_requirement": "both D-S4e R2 values must be strictly > 0",
            "if_fail": "invalid_exposure_repair__q4e_minus_s4e_not_interpretable",
        },
        "only_after_validity_pass": {
            "estimator": {
                "all_required": True,
                "pooled_q4e_minus_s4e_min": DELTA_GATE,
                "per_target_recording_requirement": "both Q4e-S4e deltas must be strictly > 0",
                "decision": "estimator-limited evidence",
            },
            "consumer": {
                "all_required": True,
                "absolute_pooled_q4e_minus_s4e_strictly_below": DELTA_GATE,
                "no_session_sign_reversal_definition": "delta_session_0 * delta_session_1 >= 0",
                "decision": "consumer-limited evidence",
            },
            "otherwise": "unresolved",
        },
        "decision_order": ["validity", "estimator", "consumer", "unresolved"],
        "scope_limit": (
            "D-Q4e is a query-local leakage diagnostic, never a deployable "
            "target-time arm or paper-main selection endpoint."
        ),
    }


def write_preflight(output_path: str | Path) -> dict[str, str]:
    """Hash the complete evaluator closure without opening any recordings."""

    files = (
        "scripts/h1_carrierid_distribution_exposure_terminal_preflight.py",
        "scripts/h1_carrierid_distribution_exposure_terminal_checker.py",
        "scripts/h1_carrierid_distribution_exposure_terminal_evaluate.py",
        "scripts/h1_carrierid_distribution_exposure_preflight.py",
        "src/data/h1_carrierid_distribution_exposure.py",
        "src/models/h1_carrierid_distribution_exposure_module.py",
        "src/data/h1_carrierid_distribution_target.py",
        "src/data/h1_carrierid_distribution.py",
        "src/models/h1_carrierid_distribution_module.py",
        "src/models/h1_carrierid_module.py",
        "src/models/components/h1_carrierid_spint.py",
        "src/data/h1_m4_eb_pilot.py",
        "src/h1_m4_eb_normalized_v2_contract.py",
        "scripts/h1_carrierid_evaluate.py",
        "configs/data/falcon_h1_carrierid_distribution_exposure.yaml",
        "configs/model/falcon_h1_carrierid_distribution_exposure.yaml",
        "configs/experiment/h1_carrierid_distribution_exposure.yaml",
        "configs/experiment/h1_carrierid_distribution_exposure_s4.yaml",
        "configs/experiment/h1_carrierid_distribution_exposure_q4.yaml",
        "configs/callbacks/h1_carrierid_terminal.yaml",
        "tests/test_h1_carrierid_distribution_exposure_contract.py",
        "tests/test_h1_carrierid_distribution_exposure_terminal_contract.py",
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
            "arms": ["D-S4e", "D-Q4e"],
            "checkpoints": "two independently trained exact epoch49/global_step180500 checkpoints",
            "checkpoint_pair_requirement": "source-only pair checker must pass before target access",
            "target_comparison_label": "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT",
            "one_shot_target_evaluation": True,
            "validity_before_interpretation": True,
        },
        "frozen_validity_and_branch_gates": frozen_terminal_branch_gates(),
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
