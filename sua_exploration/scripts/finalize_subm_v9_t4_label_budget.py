#!/usr/bin/env python3
"""Authoritative local TorchMetrics-1.5.1 finalizer for the sub-M T4 budget curve."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "sua_exploration"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from sua_exploration.mc_maze import subm_v9_t4_label_budget as budget_core


VIEWS = ("sua", "pseudo_mua")
SEEDS = (42, 43, 44)
EXPECTED_SESSIONS = 15
EXPECTED_NEW_CELLS = EXPECTED_SESSIONS * len(VIEWS) * len(SEEDS) * len(budget_core.BUDGETS)
SCHEMA = "dandi_000688_subm_v9_t4_label_budget_torchmetrics151_v1"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def load_npz(path: Path, rows: int) -> tuple[np.ndarray, np.ndarray]:
    need(readonly(path), f"unsafe/missing artifact {path}")
    with np.load(path, allow_pickle=False) as archive:
        prediction = np.ascontiguousarray(archive["predictions"], dtype=np.float32)
        target = np.ascontiguousarray(archive["targets"], dtype=np.float32)
    need(prediction.shape == target.shape == (rows, 2), f"shape drift {path}")
    need(np.isfinite(prediction).all() and np.isfinite(target).all(), f"nonfinite {path}")
    return prediction, target


def new_paths(root: Path, asset: str, view: str, budget: int, seed: int) -> tuple[Path, Path]:
    artifact = root / "artifacts" / asset / view / f"m_{budget}" / f"seed_{seed}" / "predictions_targets.npz"
    commit = root / "commits" / asset / view / f"m_{budget}" / f"seed_{seed}.json"
    return artifact, commit


def ref_path(root: Path, asset: str, view: str, seed: int) -> Path:
    return root / "artifacts" / asset / view / "shared_t4" / f"seed_{seed}" / "predictions_targets.npz"


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-v9-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    reference = args.reference_v9_root.resolve()
    output = args.output.resolve() if args.output else root / "aggregate" / "endpoint_aggregate_torchmetrics151.json"

    import torchmetrics
    need(torchmetrics.__version__ == "1.5.1", f"requires TorchMetrics 1.5.1, got {torchmetrics.__version__}")
    from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu

    manifest = load_json(root / "run_manifest.json")
    need(manifest.get("expected_full_cells") == EXPECTED_NEW_CELLS, "budget manifest cell count drift")
    need(manifest.get("metric_computed") is False, "remote metric policy drift")
    v9_manifest = load_json(reference / "run_manifest.json")
    cohort = v9_manifest.get("contract", {}).get("cohort")
    need(isinstance(cohort, list) and len(cohort) == EXPECTED_SESSIONS, "V9 cohort missing")

    scores = {
        view: {budget: np.empty((EXPECTED_SESSIONS, len(SEEDS)), dtype=np.float64)
               for budget in (*budget_core.BUDGETS, budget_core.REFERENCE_BUDGET)}
        for view in VIEWS
    }
    cells: list[dict[str, Any]] = []
    for session_index, row in enumerate(cohort):
        asset = str(row["asset_id"])
        session = str(row["session_id"])
        rows = int(row["query_window_count"])
        for view in VIEWS:
            for seed_index, seed in enumerate(SEEDS):
                ref_prediction, ref_target = load_npz(ref_path(reference, asset, view, seed), rows)
                ref_r2 = float(recompute_torchmetrics_r2_cpu(ref_prediction, ref_target))
                scores[view][budget_core.REFERENCE_BUDGET][session_index, seed_index] = ref_r2
                for budget in budget_core.BUDGETS:
                    artifact, commit_path = new_paths(root, asset, view, budget, seed)
                    need(readonly(commit_path), f"unsafe/missing commit {commit_path}")
                    commit = load_json(commit_path)
                    need(commit.get("metric_computed") is False and commit.get("backward_called") is False, f"policy drift {commit_path}")
                    need(commit.get("artifact_sha256") == sha256_file(artifact), f"artifact SHA drift {artifact}")
                    prediction, target = load_npz(artifact, rows)
                    need(np.array_equal(target, ref_target), f"target differs from M50 {session}/{view}/M{budget}/s{seed}")
                    r2 = float(recompute_torchmetrics_r2_cpu(prediction, target))
                    scores[view][budget][session_index, seed_index] = r2
                    cells.append({
                        "asset_id": asset, "session_id": session, "view": view,
                        "budget": budget, "seed": seed, "query_window_count": rows, "r2": r2,
                    })
    need(len(cells) == EXPECTED_NEW_CELLS, "finalized cell topology drift")

    summary: dict[str, Any] = {}
    for view_index, view in enumerate(VIEWS):
        view_summary: dict[str, Any] = {}
        ref_grid = scores[view][budget_core.REFERENCE_BUDGET]
        for budget in (*budget_core.BUDGETS, budget_core.REFERENCE_BUDGET):
            grid = scores[view][budget]
            delta = grid - ref_grid
            session_means = delta.mean(axis=1)
            seed_means = grid.mean(axis=0)
            delta_seed_means = delta.mean(axis=0)
            bootstrap = budget_core.hierarchical_bootstrap(
                delta, seed=68_820_260_805 + view_index * 100 + budget
            )
            view_summary[str(budget)] = {
                "mean_r2": float(grid.mean()),
                "seed_mean_r2": {str(seed): float(seed_means[index]) for index, seed in enumerate(SEEDS)},
                "delta_vs_m50": float(delta.mean()),
                "seed_mean_delta_vs_m50": {str(seed): float(delta_seed_means[index]) for index, seed in enumerate(SEEDS)},
                "positive_session_delta_count": int(np.sum(session_means > 0.0)),
                "within_0_03_of_m50": budget_core.within_003_summary(delta),
                "hierarchical_bootstrap_delta_vs_m50": bootstrap,
            }
        eligible = [
            budget for budget in (*budget_core.BUDGETS, budget_core.REFERENCE_BUDGET)
            if view_summary[str(budget)]["within_0_03_of_m50"]["point_and_three_seed_stable"]
        ]
        view_summary["smallest_budget_within_0_03_all_seed_gate"] = min(eligible) if eligible else None
        summary[view] = view_summary

    payload = {
        "schema": SCHEMA,
        "status": "FULL_450_LOCAL_TORCHMETRICS_1_5_1_FINALIZED",
        "claim_boundary": {
            "scope": "post-hoc exploratory frozen-M50-trained V9 label-budget robustness",
            "query": "fixed strictly after trial 50; label-count effect, not earlier-start latency",
            "training": "no low-budget retraining; not a theoretical minimum-label claim",
        },
        "verified_new_cell_count": len(cells),
        "source_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "source_v9_manifest_sha256": sha256_file(reference / "run_manifest.json"),
        "summary": summary,
        "cells": cells,
    }
    need(not output.exists(), f"refusing to overwrite {output}")
    write_exclusive(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "sha256": sha256_file(output), "summary": summary,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
