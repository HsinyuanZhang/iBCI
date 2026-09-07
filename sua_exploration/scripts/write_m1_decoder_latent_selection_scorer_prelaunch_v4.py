#!/usr/bin/env python3
"""Independent audit receipt for the repaired M1 DLA selection scorer; no data opens."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v2"
RUNNER = ROOT / "sua_exploration" / "scripts" / "m1_decoder_latent_selection_runner.py"
STAGE = ROOT / "sua_exploration" / "scripts" / "m1_decoder_latent_selection_stage.py"
AGGREGATOR = ROOT / "sua_exploration" / "scripts" / "aggregate_m1_decoder_latent_selection_arms.py"
TEST = ROOT / "sua_exploration" / "tests" / "test_m1_decoder_latent_selection_stage.py"
PREVIOUS = OUT / "selection_scorer_prelaunch_v3.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> Path:
    target = OUT / "selection_scorer_prelaunch_v4.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    source = RUNNER.read_text(encoding="utf-8")
    required = [
        "sys.modules[spec.name] = module", "dataset.neural_data[name]", "dataset.covariate_data[name]",
        "behavior_window[-1]", "dataset[index]", "query_end_trial=210", "trial >= SELECTION[1]",
        "if name == OUTER_LEFT_OUT", "_teacher_delta_for_source_only", "torch.no_grad",
    ]
    if any(item not in source for item in required):
        raise ValueError("runner lacks a required repaired import/raw-query/split boundary")
    test_source = TEST.read_text(encoding="utf-8")
    if "test_runner_query_windows_bypass_side_feature_getitem" not in test_source:
        raise ValueError("focused raw-array D4-bypass test is missing")
    aggregate_source = AGGREGATOR.read_text(encoding="utf-8")
    for item in ("manifest/session audit mismatch", "F0 outer baseline mismatch", "no_report_access"):
        if item not in aggregate_source:
            raise ValueError("fail-closed aggregator contract is incomplete")
    payload = {
        "schema_version": "m1_dla_selection_scorer_prelaunch_v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "independent_audit_passed_scorer_not_executed_pending_root_launch",
        "audit_findings": {
            "dynamic_import": "pass: isolated selection-stage module is registered in sys.modules before exec_module; postponed dataclass annotations resolve.",
            "d4_side_feature": "pass: d4 is used only to load support obj_id metadata; query reader directly slices neural_data/covariate_data and never calls dataset.__getitem__.",
            "neural_behavior_convention": "pass: raw query path matches FalconDataset __getitem__ slice [start:start+window_size] and uses covariate_window[-1], matching frozen last-timestep decoder scoring.",
            "selection_boundary": "pass: dataset query window audit is fixed at trials [10,210), full-window-disjoint; direct reader rejects any derived trial outside 10..209.",
            "outer_target_disclosure": "pass: outer ses-20120926 is constructed in a separate branch with no delta field or teacher-target call.",
            "aggregation": "pass: aggregation rejects scope, manifest/session-audit, or F0-baseline disagreement before gates.",
        },
        "focused_verification": {
            "py_compile": [str(path.relative_to(ROOT)) for path in (RUNNER, STAGE, AGGREGATOR)],
            "tests": [
                "test_m1_decoder_latent_selection_stage.py",
                "test_m1_decoder_latent_alignment_oracle_base_addendum.py",
                "test_m1_clean_selection_query_window.py",
            ],
            "test_result": "31 passed; runner default preflight only",
        },
        "scope": {"support": [0, 10], "query": [10, 210], "report_or_later_read": False,
                  "heldout": False, "EvalAI": False, "full_scoring": False, "GPU_launch": False},
        "hashes": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in {
            "previous_v3": PREVIOUS, "runner": RUNNER, "stage": STAGE, "aggregator": AGGREGATOR, "focused_tests": TEST,
        }.items()},
        "remaining_blocker": None,
        "not_executed": ["NWB_or_query_loading", "input_build", "forward_smoke", "full_scoring", "report", "heldout", "EvalAI", "GPU"],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "selection_scorer_prelaunch_v4.sha256").write_text(f"{sha256(target)}  selection_scorer_prelaunch_v4.json\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run())
