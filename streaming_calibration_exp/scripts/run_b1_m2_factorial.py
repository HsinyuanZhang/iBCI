#!/usr/bin/env python3
"""Print the only B1 factorial launch plan; refuse training in this scaffold.

The runner is intentionally dry-run only until an independently reviewed
post-training epoch scorer is bound.  It neither claims GPU authorization nor
permits an accidental best-validation replacement for the fixed 5--12 score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.metrics import b1_m2_factorial as core


RESULT_ROOT = PROJECT.parent / "sua_exploration/results/m2_carrier_distillation_interaction_v2"


def _load_official_preflight(path: Path) -> tuple[dict, str]:
    payload, digest = core.load_verified_immutable_json(path)
    core.validate_preflight_payload(payload)
    core.validate_live_source_bindings(PROJECT.parent, payload["implementation_bindings"])
    return payload, digest


def _cell_paths(cell: core.CellSpec) -> dict[str, Path]:
    # Explicit, unique, deterministic directories are part of the *future*
    # launch contract.  The runner below still refuses to launch.
    stem = f"stage{cell.stage}_fold{cell.fold}_seed{cell.seed}_{cell.carrier}_{cell.loss_mode}"
    return {
        "log_dir": RESULT_ROOT / "logs" / stem,
        "artifact_parent": RESULT_ROOT / "artifacts" / stem,
        "launch_receipt": RESULT_ROOT / "launch_receipts" / f"{stem}.json",
        "post_training_binding": RESULT_ROOT / "post_training_bindings" / f"{stem}.json",
        "score_receipt": RESULT_ROOT / "score_receipts" / f"{stem}.json",
    }


def _command(cell: core.CellSpec) -> list[str]:
    paths = _cell_paths(cell)
    return [
        str(Path(sys.executable).resolve()), str((PROJECT / "src/train.py").resolve()),
        f"experiment={core.expected_config_name(cell.carrier, cell.loss_mode)}",
        f"seed={cell.seed}", f"data.loso_fold={cell.fold}", "train=true", "test=false",
        f"run_id=b1_v2_{cell.stage}_f{cell.fold}_s{cell.seed}_{cell.carrier}_{cell.loss_mode}",
        f"hydra.run.dir={paths['log_dir']}",
        f"paths.artifact_dir={paths['artifact_parent']}",
        # Testing each raw epoch is delegated to the approved scorer.  Passing
        # test=false here prevents train.py's best-checkpoint test path.
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("P", "F"), required=True)
    parser.add_argument("--official-preflight", type=Path, required=True)
    parser.add_argument("--stage-p-aggregate", type=Path, default=None)
    parser.add_argument("--launch", action="store_true", help="always refused by this no-launch scaffold")
    args = parser.parse_args()
    preflight, preflight_sha = _load_official_preflight(args.official_preflight)
    if args.stage == "F":
        if args.stage_p_aggregate is None:
            raise SystemExit("Stage F requires an immutable Stage-P aggregate")
        aggregate, _digest = core.load_verified_immutable_json(args.stage_p_aggregate)
        try:
            core.validate_stage_p_aggregate(aggregate, official_preflight_sha256=preflight_sha)
        except core.B1ContractError as exc:
            raise SystemExit(f"Stage F blocked: invalid Stage-P aggregate: {exc}") from exc
        gate = aggregate.get("stage_f_predeclared_gate")
        if (aggregate.get("screen_id") != core.SCREEN_ID or aggregate.get("stage") != "P"
                or not isinstance(gate, dict) or gate.get("stage_f_authorized_by_evidence") is not True):
            raise SystemExit("Stage F blocked: Stage-P practical/3-seed gate did not pass")
        if aggregate.get("official_preflight_sha256") != preflight_sha:
            raise SystemExit("Stage F blocked: Stage-P aggregate was not scored against this official preflight")
        if aggregate.get("expected_cell_keys") != sorted(cell.key for cell in core.cells_for_stage("P")):
            raise SystemExit("Stage F blocked: Stage-P aggregate cell lattice drift")
        lattice = aggregate.get("cell_receipt_sha256")
        if not isinstance(lattice, dict) or set(lattice) != {cell.key for cell in core.cells_for_stage("P")}:
            raise SystemExit("Stage F blocked: Stage-P receipt lattice is incomplete")
        expansion = preflight["stage_f_expansion_manifest"]
        if expansion.get("expected_cell_keys") != [cell.key for cell in core.cells_for_stage("F")]:
            raise SystemExit("Stage F blocked: official preflight expansion manifest drift")
    if args.launch:
        raise SystemExit(
            "Refusing GPU launch: B1 scaffold is review-gated; do not substitute a "
            "best-validation checkpoint for epochs 5--12."
        )
    payload = {
        "screen_id": core.SCREEN_ID,
        "stage": args.stage,
        "official_preflight": str(args.official_preflight.resolve()),
        "official_preflight_sha256": preflight_sha,
        "gpu_launched": False,
        "future_launch_receipts_created": [],
        "legacy_prepare_launch_receipts_supported": False,
        "cells": [
            {
                "cell": cell.key,
                "command": _command(cell),
                "unique_explicit_log_dir": str(_cell_paths(cell)["log_dir"]),
                "unique_artifact_parent": str(_cell_paths(cell)["artifact_parent"]),
                "future_o_excl_launch_receipt": "must_be_created_only_by_execute_b1_m2_factorial_cell.py_with_explicit_cuda_device",
                "required_post_training_path_binding": str(_cell_paths(cell)["post_training_binding"]),
                "future_score_receipt": str(_cell_paths(cell)["score_receipt"]),
                "future_execution_context": {
                    "interpreter": str(Path(sys.executable).resolve()),
                    "script": str((PROJECT / "src/train.py").resolve()),
                    "working_dir": str(PROJECT.resolve()),
                },
                "freshness_rule": (
                    "executor must bind absent log/artifact/launch/start/completion/execution-log/score paths "
                    "in one immutable one-cell contract"
                ),
            }
            for cell in core.cells_for_stage(args.stage)
        ],
        "post_training_requirement": (
            "score and retain every logical epoch 5--12 with a dedicated evaluator; "
            "best-validation selection is prohibited"
        ),
        "known_scaffolding_blocker": (
            "This runner deliberately remains --launch-refusing.  A future authorized executor must create an "
            "O_EXCL launch receipt after freshness checks, retain epochs004..011, and bind the exact command, "
            "official preflight SHA, and unique explicit log directory before starting training."
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
