#!/usr/bin/env python3
"""Print-only preparation for exact-query, forward-only B2-D1024 re-evaluation.

This is intentionally *not* a launcher.  It builds no model, touches no NWB,
imports neither Torch nor Lightning, and emits commands only.  A subsequent
one-shot evaluator invocation restores each sealed B2 checkpoint, uses no
optimizer/backward/checkpoint selection, and writes a new immutable outer
receipt.  The receipt's three strong query digests must subsequently equal the
verified Stage-2 T4d digests within fold before any T4d--B2 score is paired.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
COMPANION = ROOT / "sua_exploration/scripts/rt_sparse_t4d_vs_b2_d1024_companion.py"
STAGE2 = ROOT / "sua_exploration/scripts/rt_sparse_endpoint_stage2_terminal_verify.py"
EVALUATOR = ROOT / "streaming_calibration_exp/src/rt_clean_nested_loso_eval.py"
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/rt_sparse_t4d_b2_forward_reeval_v1"


class PreflightError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    _need(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(stage_manifest: Path, *, output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Return a no-side-effect preflight and command plan for folds with local checkpoints."""

    companion = _load_module(COMPANION, "rt_t4d_b2_companion_preflight")
    stage2 = _load_module(STAGE2, "rt_stage2_preflight")
    manifest = stage2._load_json(stage_manifest)
    stage2.validate_manifest_schema(manifest)
    surfaces = manifest["surfaces"]
    for name, path in {
        "outer_evaluator": EVALUATOR,
        "datamodule": ROOT / "streaming_calibration_exp/src/data/rt_nested_loso_datamodule.py",
        "falcon_dataset": ROOT / "streaming_calibration_exp/src/data/falcon_datamodule.py",
    }.items():
        _need(_sha(path) == surfaces[name]["sha256"], f"Stage2-bound {name} source SHA drift; no re-evaluation command is valid")
    rows = companion._legacy_b2_rows()
    plans: list[dict[str, Any]] = []
    for fold in companion.FOLDS:
        row = rows[fold]
        selection_path = companion._bound_legacy_file(row, "selection")
        config_path = companion._bound_legacy_file(row, "config")
        split_path = companion._bound_legacy_file(row, "split")
        selection = companion._json(selection_path)
        checkpoint = Path(str(selection.get("best_model_path", "")))
        available = checkpoint.is_file() and companion._sha(checkpoint) == selection.get("best_model_sha256")
        output = output_root / "outer" / f"f{fold:02d}_b2_d1024_zero4_forward_only.json"
        command = [
            "/home/xinyuan/miniconda3/envs/spint/bin/python", str(EVALUATOR),
            "--config", str(config_path), "--checkpoint", str(checkpoint),
            "--split-manifest", str(split_path), "--selection-receipt", str(selection_path),
            "--output", str(output), "--outer-fold", str(fold), "--device", "cpu",
        ]
        plans.append({
            "fold": fold, "available": available,
            "unavailable_reason": None if available else "selected sealed checkpoint absent or SHA mismatch",
            "checkpoint": {"path": str(checkpoint), "sha256": selection.get("best_model_sha256")},
            "output_must_not_exist": str(output),
            "cpu_command": command,
            "minimal_gpu_substitution": command[:-1] + ["cuda"],
        })
    return {
        "schema": "rt_sparse_t4d_b2_forward_reeval_preflight_v1",
        "status": "PASS_PRINT_ONLY_REEVAL_PREPARED_NOT_LAUNCHED",
        "purpose": "repair legacy B2 query-identity evidence by a new forward-only evaluation; never retrain B2",
        "stage2_manifest_sha256": _sha(stage_manifest),
        "bound_sources": {name: _sha(path) for name, path in {"outer_evaluator": EVALUATOR, "datamodule": ROOT / "streaming_calibration_exp/src/data/rt_nested_loso_datamodule.py", "falcon_dataset": ROOT / "streaming_calibration_exp/src/data/falcon_datamodule.py"}.items()},
        "requires_before_execution": [
            "Stage2 must first close 45/45 and pass its independent terminal verifier.",
            "Do not overwrite any historical B2 outer receipt or use it for this comparison.",
            "After each new receipt, prove state-before == state-after, no optimizer/backward, and exact three-digest equality with the corresponding verified T4d cell.",
            "Fold 0 cannot be re-evaluated until its sealed selected checkpoint is re-imported with SHA equality.",
        ],
        "cpu_practicality": {
            "historical_b2_mac_per_session": 5274009600,
            "available_checkpoint_folds": [plan["fold"] for plan in plans if plan["available"]],
            "unavailable_checkpoint_folds": [plan["fold"] for plan in plans if not plan["available"]],
            "assessment": "CPU is technically valid but likely hours for 14 B2-D1024 sessions plus NWB IO; prefer one GPU for forward-only evaluation if wall-clock matters. This is no-backprop inference, not a retraining job.",
        },
        "plans": plans,
        "non_interference": {"nwb_opened": False, "torch_imported": False, "gpu_opened": False, "trainer_started": False, "artifact_written": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.stage_manifest, output_root=args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except PreflightError as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(2) from error
