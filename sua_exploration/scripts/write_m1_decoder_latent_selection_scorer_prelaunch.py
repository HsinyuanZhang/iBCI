#!/usr/bin/env python3
"""Immutable final-run preflight for M1 DLA selection scorer; never invokes scorer/data."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v2"
PREVIOUS = OUT / "selection_stage_prelaunch.json"
RUNNER = ROOT / "sua_exploration" / "scripts" / "m1_decoder_latent_selection_runner.py"
STAGE = ROOT / "sua_exploration" / "scripts" / "m1_decoder_latent_selection_stage.py"
TEST = ROOT / "sua_exploration" / "tests" / "test_m1_decoder_latent_selection_stage.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> Path:
    target = OUT / "selection_scorer_prelaunch.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    if prior.get("status") != "implementation_and_hash_preflight_passed_selection_execution_requires_root_review":
        raise ValueError("prior selection-stage prelaunch is not valid")
    source = RUNNER.read_text(encoding="utf-8")
    required = ["SOURCE_SESSIONS", "OUTER_LEFT_OUT", "_teacher_delta_for_source_only", "if name == OUTER_LEFT_OUT",
                "query_end_trial=210", "trial >= SELECTION[1]", "M1_DLA_ROOT_REVIEWED_EXECUTION", "refusing to overwrite immutable selection result",
                '"heldout_access": False', '"EvalAI_access": False', '"GPU_launch": False']
    if any(token not in source for token in required):
        raise ValueError("selection runner lacks one or more required actual-path boundaries")
    receipt = {"schema_version": "m1_dla_selection_scorer_prelaunch_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "scorer_implemented_not_executed_pending_root_launch", "scope": {"sources": ["ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928"],
                         "outer_left_out": "ses-20120926", "support": [0, 10], "query": [10, 210], "report_reading": False,
                         "heldout": False, "EvalAI": False, "GPU": False},
               "execution_contract": {"per_arm_cli": ["full", "rate_only", "label_shuffle", "rate_residualized_condition_only"],
                                      "all_arm_cli": True, "root_env_guard": "M1_DLA_ROOT_REVIEWED_EXECUTION=1",
                                      "output": "immutable result directory; no overwrite", "inner_candidates_per_arm": 12,
                                      "inner_metric": "three inner-validation frozen-decoder behavior Delta_R2 values", "outer_gate_runs_once": True,
                                      "outer_teacher_target": "forbidden by separate OuterSelectionSession construction branch"},
               "hashes": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in {"prior_prelaunch": PREVIOUS, "runner": RUNNER, "stage": STAGE, "tests": TEST}.items()},
               "not_executed": ["NWB_or_query_loading", "E0_or_teacher_construction", "inner_loso", "outer_gate", "bootstrap", "selection_result_write", "report_reading", "heldout_access", "EvalAI", "GPU"]}
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "selection_scorer_prelaunch.sha256").write_text(f"{sha256(target)}  selection_scorer_prelaunch.json\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run())
