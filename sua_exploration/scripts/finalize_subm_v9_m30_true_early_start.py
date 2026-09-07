#!/usr/bin/env python3
"""Local TorchMetrics-1.5.1 finalizer for sub-M M30 true-early-start artifacts.

The remote 5070 Ti must only have produced immutable forward artifacts.  This
command refuses to score before all 270 cells are present and validates the
preflight-bound target trace for every arm/seed.  It reports two different,
non-interchangeable quantities:

* the real deployment endpoint, scored over every post-trial-30 window; and
* the exact post-50 suffix replay, which has byte-identical V9 targets and
  therefore permits a paired M30-versus-M50 check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "sua_exploration"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from sua_exploration.mc_maze import subm_v9_m30_true_early_start as early
from sua_exploration.mc_maze import subm_v9_t4_label_budget as budget_core


SCHEMA = "dandi_000688_subm_v9_m30_true_early_start_torchmetrics151_v1"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def readonly(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444
    except OSError:
        return False


def read_canonical_json(path: Path) -> dict[str, Any]:
    need(readonly(path), f"missing/unsafe immutable JSON {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    need(isinstance(value, dict) and canonical_bytes(value) == raw, f"noncanonical JSON {path}")
    return value


def write_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    raw = canonical_bytes(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    need(readonly(path), f"failed to seal aggregate {path}")
    return hashlib.sha256(raw).hexdigest()


def artifact_path(root: Path, asset_id: str, view: str, arm: str, seed: int) -> Path:
    return root / "artifacts" / asset_id / view / arm / f"seed_{seed}" / "predictions_targets.npz"


def commit_path(root: Path, asset_id: str, view: str, arm: str, seed: int) -> Path:
    return root / "commits" / asset_id / view / arm / f"seed_{seed}.json"


def load_artifact(path: Path, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    need(readonly(path), f"missing/unsafe immutable artifact {path}")
    with np.load(path, allow_pickle=False) as archive:
        prediction = np.ascontiguousarray(archive["predictions"], dtype=np.float32)
        target = np.ascontiguousarray(archive["targets"], dtype=np.float32)
    need(prediction.shape == target.shape == (expected_rows, 2), f"artifact shape drift {path}")
    need(np.isfinite(prediction).all() and np.isfinite(target).all(), f"nonfinite artifact {path}")
    return prediction, target


def receipt_view(receipt: Mapping[str, Any], asset_id: str, view: str) -> dict[str, Any]:
    matches = [row for row in receipt["cohort"] if row["asset_id"] == asset_id]
    need(len(matches) == 1, f"receipt lacks asset {asset_id}")
    value = matches[0].get("views", {}).get(view)
    need(isinstance(value, dict), f"receipt lacks view {asset_id}/{view}")
    return value


def reference_artifact_path(root: Path, asset_id: str, view: str, arm: str, seed: int) -> Path:
    return root / "artifacts" / asset_id / view / arm / f"seed_{seed}" / "predictions_targets.npz"


def validate_complete_topology(root: Path, reference_v9_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Open only artifacts/commits and prove the 270-cell score scope is full."""
    manifest = read_canonical_json(root / "run_manifest.json")
    need(manifest.get("schema") == early.SCHEMA and manifest.get("status") == "FORWARD_ONLY_NO_METRIC", "run manifest identity drift")
    protocol = manifest.get("protocol")
    need(isinstance(protocol, dict) and protocol.get("expected_full_cells") == early.EXPECTED_CELLS, "run manifest topology drift")
    receipt_path = Path(str(manifest.get("preflight_receipt", ""))).expanduser()
    # A copied remote result points at a remote absolute receipt.  Resolve it
    # locally by also accepting the standard in-root copy, but never rewrite it.
    if not readonly(receipt_path):
        receipt_path = root / "preflight" / "preflight_receipt.json"
    receipt = read_canonical_json(receipt_path)
    need(sha256_file(receipt_path) == manifest.get("preflight_receipt_sha256"), "preflight receipt hash drift")
    need(receipt.get("schema") == early.SCHEMA and receipt.get("protocol", {}).get("expected_full_cells") == early.EXPECTED_CELLS, "receipt topology drift")
    need(receipt.get("metric_computed") is False and receipt.get("backward_called") is False, "receipt policy drift")
    reference_manifest = reference_v9_root / "run_manifest.json"
    need(readonly(reference_manifest), "missing/unsafe local V9 reference manifest")
    need(sha256_file(reference_manifest) == manifest.get("reference_v9_manifest_sha256"), "V9 reference manifest hash drift")

    cells: list[dict[str, Any]] = []
    target_by_session_view: dict[tuple[str, str], bytes] = {}
    for receipt_row in receipt["cohort"]:
        asset_id = str(receipt_row["asset_id"])
        session_id = str(receipt_row["session_id"])
        for view in early.VIEWS:
            evidence = receipt_view(receipt, asset_id, view)
            query = evidence["query"]
            rows = int(query["observed_query_window_count"])
            suffix_rows = int(query["post50_suffix_window_count"])
            need(0 < suffix_rows <= rows, f"bad post50 suffix count {asset_id}/{view}")
            for arm in early.ARMS:
                for seed in early.SEEDS:
                    artifact = artifact_path(root, asset_id, view, arm, seed)
                    commit = commit_path(root, asset_id, view, arm, seed)
                    payload = read_canonical_json(commit)
                    need(payload.get("schema") == early.SCHEMA, f"commit schema drift {commit}")
                    need(payload.get("metric_computed") is False and payload.get("backward_called") is False, f"commit policy drift {commit}")
                    need(
                        {
                            "asset_id": payload.get("asset_id"),
                            "session_id": payload.get("session_id"),
                            "view": payload.get("view"),
                            "arm": payload.get("arm"),
                            "seed": payload.get("seed"),
                        }
                        == {
                            "asset_id": asset_id,
                            "session_id": session_id,
                            "view": view,
                            "arm": arm,
                            "seed": seed,
                        },
                        f"commit key drift {commit}",
                    )
                    need(
                        payload.get("artifact") == str(artifact.relative_to(root))
                        and payload.get("artifact_bytes") == artifact.stat().st_size,
                        f"commit artifact path/byte drift {commit}",
                    )
                    need(payload.get("artifact_sha256") == sha256_file(artifact), f"artifact hash drift {artifact}")
                    need(payload.get("query_window_count") == rows, f"commit query count drift {commit}")
                    need(payload.get("query_target_sha256") == query["target_sha256"], f"commit target receipt drift {commit}")
                    prediction, target = load_artifact(artifact, rows)
                    need(payload.get("prediction_sha256") == sha_array(prediction), f"commit prediction hash drift {commit}")
                    need(hashlib.sha256(target.tobytes(order="C")).hexdigest() == payload.get("query_target_sha256"), f"target hash drift {artifact}")
                    peer = target_by_session_view.setdefault((asset_id, view), target.tobytes(order="C"))
                    need(peer == target.tobytes(order="C"), f"target differs across arm/seed {asset_id}/{view}")
                    reference = reference_artifact_path(reference_v9_root, asset_id, view, arm, seed)
                    reference_commit = commit_path(reference_v9_root, asset_id, view, arm, seed)
                    reference_payload = read_canonical_json(reference_commit)
                    reference_evidence = reference_payload.get("evidence")
                    need(isinstance(reference_evidence, dict), f"V9 reference checkpoint evidence missing {reference_commit}")
                    need(
                        payload.get("checkpoint_sha256") == reference_evidence.get("checkpoint_sha256")
                        and payload.get("checkpoint_path") == reference_evidence.get("checkpoint_path")
                        and payload.get("runtime_contract_sha256") == manifest.get("runtime_contract_sha256"),
                        f"checkpoint binding drift {commit}",
                    )
                    _ref_prediction, ref_target = load_artifact(reference, suffix_rows)
                    need(np.array_equal(target[-suffix_rows:], ref_target), f"M30 suffix target differs from V9 {asset_id}/{view}/{arm}/{seed}")
                    cells.append({
                        "asset_id": asset_id,
                        "session_id": session_id,
                        "view": view,
                        "arm": arm,
                        "seed": seed,
                        "query_window_count": rows,
                        "post50_suffix_window_count": suffix_rows,
                        "artifact": artifact,
                        "reference": reference,
                        "new_early_only_window_count": int(rows - suffix_rows),
                    })
    need(len(cells) == early.EXPECTED_CELLS, "incomplete or duplicated M30 topology")
    return manifest, receipt, cells


def contrast_summary(delta: np.ndarray, *, bootstrap_seed: int) -> dict[str, Any]:
    need(delta.shape == (early.EXPECTED_SESSIONS, len(early.SEEDS)), "contrast grid shape drift")
    session_means = delta.mean(axis=1, dtype=np.float64)
    seed_means = delta.mean(axis=0, dtype=np.float64)
    return {
        "mean_delta_r2": float(delta.mean(dtype=np.float64)),
        "seed_mean_delta_r2": {str(seed): float(seed_means[index]) for index, seed in enumerate(early.SEEDS)},
        "positive_session_count": int(np.sum(session_means > 0.0)),
        "hierarchical_bootstrap": budget_core.hierarchical_bootstrap(delta, seed=bootstrap_seed),
    }


def finalize(root: Path, reference_v9_root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    reference_v9_root = reference_v9_root.resolve()
    manifest, receipt, cells = validate_complete_topology(root, reference_v9_root)
    need(not output.exists(), f"refusing to overwrite aggregate {output}")

    # Only now can the unique designated local metric be opened.
    import torchmetrics
    need(torchmetrics.__version__ == "1.5.1", f"requires TorchMetrics 1.5.1, got {torchmetrics.__version__}")
    from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu

    grids = {
        view: {
            "all_post30": {arm: np.empty((early.EXPECTED_SESSIONS, len(early.SEEDS)), dtype=np.float64) for arm in early.ARMS},
            "new_early_only": {arm: np.empty((early.EXPECTED_SESSIONS, len(early.SEEDS)), dtype=np.float64) for arm in early.ARMS},
            "post50_suffix": {arm: np.empty((early.EXPECTED_SESSIONS, len(early.SEEDS)), dtype=np.float64) for arm in early.ARMS},
            "v9_post50": {arm: np.empty((early.EXPECTED_SESSIONS, len(early.SEEDS)), dtype=np.float64) for arm in early.ARMS},
        }
        for view in early.VIEWS
    }
    finalized_cells: list[dict[str, Any]] = []
    for cell in cells:
        session_index = next(index for index, row in enumerate(receipt["cohort"]) if row["asset_id"] == cell["asset_id"])
        seed_index = early.SEEDS.index(int(cell["seed"]))
        prediction, target = load_artifact(cell["artifact"], int(cell["query_window_count"]))
        ref_prediction, ref_target = load_artifact(cell["reference"], int(cell["post50_suffix_window_count"]))
        suffix_rows = int(cell["post50_suffix_window_count"])
        new_early_rows = int(cell["new_early_only_window_count"])
        need(new_early_rows > 0, "M30 endpoint has no newly available trial31--50 windows")
        all_post30_r2 = float(recompute_torchmetrics_r2_cpu(prediction, target))
        new_early_only_r2 = float(recompute_torchmetrics_r2_cpu(prediction[:new_early_rows], target[:new_early_rows]))
        suffix_r2 = float(recompute_torchmetrics_r2_cpu(prediction[-suffix_rows:], target[-suffix_rows:]))
        v9_r2 = float(recompute_torchmetrics_r2_cpu(ref_prediction, ref_target))
        need(
            math.isfinite(all_post30_r2) and math.isfinite(new_early_only_r2)
            and math.isfinite(suffix_r2) and math.isfinite(v9_r2),
            "nonfinite local R2",
        )
        view, arm = str(cell["view"]), str(cell["arm"])
        grids[view]["all_post30"][arm][session_index, seed_index] = all_post30_r2
        grids[view]["new_early_only"][arm][session_index, seed_index] = new_early_only_r2
        grids[view]["post50_suffix"][arm][session_index, seed_index] = suffix_r2
        grids[view]["v9_post50"][arm][session_index, seed_index] = v9_r2
        finalized_cells.append({
            **{key: value for key, value in cell.items() if key not in {"artifact", "reference"}},
            "r2_all_post30_true_early_start": all_post30_r2,
            "r2_new_early_only_trials31_to_50": new_early_only_r2,
            "r2_m30_on_matched_post50_suffix": suffix_r2,
            "r2_v9_m50_on_matched_post50_suffix": v9_r2,
        })

    summary: dict[str, Any] = {}
    for view_index, view in enumerate(early.VIEWS):
        by_view: dict[str, Any] = {"absolute": {}, "contrasts": {}}
        for arm in early.ARMS:
            all_post30_grid = grids[view]["all_post30"][arm]
            new_early_only_grid = grids[view]["new_early_only"][arm]
            suffix_grid = grids[view]["post50_suffix"][arm]
            v9_grid = grids[view]["v9_post50"][arm]
            by_view["absolute"][arm] = {
                "mean_r2_all_post30_true_early_start": float(all_post30_grid.mean(dtype=np.float64)),
                "seed_mean_r2_all_post30_true_early_start": {str(seed): float(all_post30_grid[:, index].mean()) for index, seed in enumerate(early.SEEDS)},
                "mean_r2_new_early_only_trials31_to_50": float(new_early_only_grid.mean(dtype=np.float64)),
                "seed_mean_r2_new_early_only_trials31_to_50": {str(seed): float(new_early_only_grid[:, index].mean()) for index, seed in enumerate(early.SEEDS)},
                "mean_r2_m30_on_matched_post50_suffix": float(suffix_grid.mean(dtype=np.float64)),
                "mean_r2_v9_m50_on_matched_post50_suffix": float(v9_grid.mean(dtype=np.float64)),
                "m30_minus_v9_m50_on_matched_post50_suffix": contrast_summary(
                    suffix_grid - v9_grid, bootstrap_seed=68_820_260_806 + view_index * 100 + early.ARMS.index(arm)
                ),
            }
        for endpoint, grid_key in (
            ("all_post30_true_early_start", "all_post30"),
            ("new_early_only_trials31_to_50", "new_early_only"),
            ("matched_post50_suffix", "post50_suffix"),
        ):
            t4 = grids[view][grid_key]["shared_t4"]
            for label, control in (("shared_t4_minus_shared_zero4", "shared_zero4"), ("shared_t4_minus_shared_ts4", "shared_ts4")):
                by_view["contrasts"][f"{endpoint}:{label}"] = contrast_summary(
                    t4 - grids[view][grid_key][control],
                    bootstrap_seed=68_820_260_900 + view_index * 100 + len(by_view["contrasts"]),
                )
        summary[view] = by_view
    payload = {
        "schema": SCHEMA,
        "status": "FULL_270_LOCAL_TORCHMETRICS_1_5_1_FINALIZED",
        "claim_boundary": {
            "scope": "post-hoc same-external-cohort frozen-V9-system M30 true-early-start latency characterization",
            "all_post30_true_early_start_endpoint": "all query windows from rewarded trials strictly after trial 30",
            "new_early_only_endpoint": "only newly available query windows in trials 31--50; never pooled with post50 windows",
            "matched_post50_suffix_endpoint": "exact V9 target-window suffix; use for M30-versus-V9-M50 paired replay only",
            "not_new_external_confirmation": True,
            "target_session_backpropagation": False,
        },
        "verified_cell_count": len(finalized_cells),
        "source_run_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "source_preflight_receipt_sha256": manifest["preflight_receipt_sha256"],
        "source_v9_manifest_sha256": sha256_file(reference_v9_root / "run_manifest.json"),
        "summary": summary,
        "cells": finalized_cells,
    }
    digest = write_exclusive(output, payload)
    return {"status": payload["status"], "output": str(output), "sha256": digest, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-v9-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "aggregate" / "endpoint_aggregate_torchmetrics151.json"
    print(json.dumps(finalize(root, args.reference_v9_root.resolve(), output), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
