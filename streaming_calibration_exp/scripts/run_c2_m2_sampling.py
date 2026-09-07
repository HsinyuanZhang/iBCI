#!/usr/bin/env python3
"""Print the only C2 sampling-objective launch plan; refuse training.

This runner is intentionally dry-run only.  It neither claims GPU
authorization nor writes receipts, logs, or result artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.metrics import c2_m2_sampling as core

RESULT_ROOT = PROJECT.parent / "sua_exploration/results/m2_sampling_objective_c2_v1"
CONTRACT = PROJECT.parent / "sua_exploration/docs/C2_M2_SAMPLING_OBJECTIVE_CONTRACT_20260813.md"


def _load_official_preflight(path: Path) -> tuple[dict, str]:
    payload, digest = core.load_verified_immutable_json(path)
    if payload.get("screen_id") != core.SCREEN_ID:
        raise SystemExit("C2 preflight screen_id drift")
    if payload.get("gpu_authorized") is True:
        raise SystemExit("C2 preflight must not claim GPU authorization")
    live = core.source_bindings(PROJECT.parent)
    if payload.get("implementation_bindings") != live:
        raise SystemExit("C2 implementation changed after official preflight")
    return payload, digest


def _cell_paths(cell: core.CellSpec) -> dict[str, Path]:
    stem = f"{cell.sampling}_{cell.carrier}_fold{cell.fold}_seed{cell.seed}"
    return {
        "log_dir": RESULT_ROOT / "logs" / stem,
        "artifact_parent": RESULT_ROOT / "artifacts" / stem,
        "launch_receipt": RESULT_ROOT / "launch_receipts" / f"{stem}.json",
        "score_receipt": RESULT_ROOT / "score_receipts" / f"{stem}.json",
    }


def _command(cell: core.CellSpec) -> list[str]:
    paths = _cell_paths(cell)
    return [
        str(Path(sys.executable).resolve()),
        str((PROJECT / "src/train.py").resolve()),
        f"experiment={core.expected_config_name(cell.sampling, cell.carrier)}",
        f"seed={cell.seed}",
        f"data.loso_fold={cell.fold}",
        "train=true",
        "test=false",
        f"run_id=c2_v1_f{cell.fold}_s{cell.seed}_{cell.carrier}_{cell.sampling}",
        f"hydra.run.dir={paths['log_dir']}",
        f"paths.artifact_dir={paths['artifact_parent']}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-preflight", type=Path, required=True)
    parser.add_argument("--launch", action="store_true", help="always refused by this no-launch scaffold")
    args = parser.parse_args()
    preflight, preflight_sha = _load_official_preflight(args.official_preflight)
    if args.launch:
        raise SystemExit(
            "Refusing GPU launch: C2 sampling-objective scaffold is review-gated and "
            "authorizes no training. Equal-session interpolation remains default-off."
        )
    payload = {
        "screen_id": core.SCREEN_ID,
        "contract": str(CONTRACT.resolve()),
        "contract_sha256": core.sha256_file(CONTRACT),
        "official_preflight": str(args.official_preflight.resolve()),
        "official_preflight_sha256": preflight_sha,
        "preflight_status": preflight.get("status"),
        "gpu_launched": False,
        "gpu_authorized": False,
        "r2_native_loss": False,
        "existing_lever": dict(core.EXISTING_LEVER),
        "sealed_formal_test_sessions_refused": sorted(core.SEALED_FORMAL_TEST_SESSIONS),
        "cells": [
            {
                "cell": cell.key,
                "command": _command(cell),
                "unique_explicit_log_dir": str(_cell_paths(cell)["log_dir"]),
                "unique_artifact_parent": str(_cell_paths(cell)["artifact_parent"]),
                "future_score_receipt": str(_cell_paths(cell)["score_receipt"]),
                "balance_session_batches": cell.balance_session_batches,
            }
            for cell in core.expected_cells()
        ],
        "known_scaffolding_blocker": (
            "This runner deliberately remains --launch-refusing.  No receipt, log, "
            "or result artifact is created."
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
