#!/usr/bin/env python3
"""Independent, post-hoc archival-integrity verification for Priority A2b-v2.

This verifier deliberately reads only the completed A2b-v2 receipt, its completed
A2a-v2 reference receipt, and the twelve retained batch JSON files.  It does not
import or execute the receipt-bound A2b producer.  A passing result proves that
the *currently retained* files are internally consistent; it cannot establish
historical producer lineage, scheduling, or that the batch files existed before
the combined receipt was written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = REPO_ROOT / "sua_exploration/results/trial_level_ridge_v1"
DEFAULT_RECEIPT = RESULTS_DIR / "priority_a2_same_target_density_v2_receipt.json"
DEFAULT_A2A_RECEIPT = RESULTS_DIR / "priority_a2_weighting_control_v2_receipt.json"
DEFAULT_BATCH_DIR = RESULTS_DIR / "a2b_v2_batches"
DEFAULT_MANIFEST = REPO_ROOT / "sua_exploration/manifests/a2b_v2_posthoc_archival_integrity_20260811.json"
RUNNER_PATH = SCRIPT_DIR / "run_priority_a2_same_target_density_v2.py"
CORE_PATH = REPO_ROOT / "sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py"
VIEWS = ("sua", "pseudo_mua")
BUDGETS = (15, 30, 50)
SEEDS = (42, 43, 44)
FINITE_KS = (1, 2, 4, 8, 16)


class ArchivalIntegrityError(RuntimeError):
    """Raised when the retained A2b-v2 files cannot support an integrity check."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArchivalIntegrityError(f"missing required file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchivalIntegrityError(f"unreadable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ArchivalIntegrityError(f"top-level JSON is not an object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchivalIntegrityError(message)


def _a2a_dense_index(a2a: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    _require(a2a.get("schema") == "priority_a2a_weighting_control_v2", "A2a reference schema is not corrected v2")
    cells = a2a.get("cells")
    _require(isinstance(cells, list) and len(cells) == 90, "A2a reference must contain exactly 90 cells")
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for cell in cells:
        _require(isinstance(cell, dict), "A2a reference contains a malformed cell")
        key = (cell.get("asset_id"), cell.get("view"), cell.get("budget"))
        _require(isinstance(key[0], str) and key[0] and key[1] in VIEWS and key[2] in BUDGETS, "A2a reference has an invalid grid key")
        _require(key not in index, "A2a reference has duplicate grid cells")
        arm = cell.get("arms", {}).get("dense_equal_trial") if isinstance(cell.get("arms"), dict) else None
        _require(isinstance(arm, dict), "A2a reference cell lacks dense_equal_trial")
        index[key] = arm
    assets = {key[0] for key in index}
    expected = {(asset, view, budget) for asset in assets for view in VIEWS for budget in BUDGETS}
    _require(len(assets) == 15 and set(index) == expected, "A2a reference does not have the exact 15×2×3 grid")
    return index


def _cell_key(cell: dict[str, Any]) -> tuple[str, str, int, int, str]:
    asset, view, budget, seed, k = cell.get("asset_id"), cell.get("view"), cell.get("budget"), cell.get("mask_seed"), cell.get("K")
    _require(isinstance(asset, str) and asset, "A2b cell has invalid asset_id")
    _require(view in VIEWS and budget in BUDGETS and seed in SEEDS, "A2b cell has invalid view, budget, or mask seed")
    normalized_k = "all" if k == "all" else str(k)
    _require(normalized_k in {"1", "2", "4", "8", "16", "all"}, "A2b cell has invalid K")
    return asset, view, budget, seed, normalized_k


def verify_archival_integrity(
    *,
    receipt_path: Path = DEFAULT_RECEIPT,
    a2a_receipt_path: Path = DEFAULT_A2A_RECEIPT,
    batch_dir: Path = DEFAULT_BATCH_DIR,
) -> dict[str, Any]:
    """Verify the exact retained grid, A2a reproduction, and batch concatenation."""
    receipt = _load_json(receipt_path)
    a2a = _load_json(a2a_receipt_path)
    _require(receipt.get("schema") == "priority_a2b_same_target_density_v2", "A2b receipt schema mismatch")
    _require(receipt.get("status") == "COMPLETED_CPU_ONLY", "A2b receipt is not marked completed CPU-only")

    bindings = receipt.get("input_bindings")
    _require(isinstance(bindings, dict), "A2b receipt lacks input bindings")
    _require(bindings.get("runner_sha256") == _sha256_file(RUNNER_PATH), "current receipt-bound runner SHA does not match retained runner")
    _require(bindings.get("ridge_core_sha256") == _sha256_file(CORE_PATH), "current receipt-bound core SHA does not match retained core")
    a2a_binding = bindings.get("a2a_v2_receipt")
    _require(isinstance(a2a_binding, dict) and a2a_binding.get("sha256") == _sha256_file(a2a_receipt_path), "A2b does not bind the retained A2a receipt SHA")

    a2a_index = _a2a_dense_index(a2a)
    assets = {key[0] for key in a2a_index}
    cells = receipt.get("cells")
    _require(isinstance(cells, list) and len(cells) == 15 * 2 * 3 * 3 * 6, "A2b must contain exactly 15×2×3×3×6 = 1620 cells")
    typed_cells: list[dict[str, Any]] = []
    grid: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    for raw_cell in cells:
        _require(isinstance(raw_cell, dict), "A2b receipt contains a malformed cell")
        key = _cell_key(raw_cell)
        _require(key not in grid, f"A2b receipt has duplicate grid cell: {key}")
        grid[key] = raw_cell
        typed_cells.append(raw_cell)
    expected_grid = {(asset, view, budget, seed, str(k)) for asset in assets for view in VIEWS for budget in BUDGETS for seed in SEEDS for k in FINITE_KS}
    expected_grid |= {(asset, view, budget, seed, "all") for asset in assets for view in VIEWS for budget in BUDGETS for seed in SEEDS}
    _require(set(grid) == expected_grid, "A2b receipt grid has missing or extra cells")

    all_index: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for key, cell in grid.items():
        asset, view, budget, seed, k = key
        if k != "all":
            _require(cell.get("mask_seed_independent") is False, "finite-K cell incorrectly marked mask-seed independent")
            continue
        _require(cell.get("mask_seed_independent") is True, "K=all cell is not marked mask-seed independent")
        _require(cell.get("reused_from_mask_seed") == SEEDS[0] and cell.get("fit_prediction_reused_across_seeds") is True, "K=all reuse contract is incomplete")
        all_index.setdefault((asset, view, budget), []).append(cell)
    _require(len(all_index) == 90 and all(len(value) == 3 for value in all_index.values()), "K=all reproduction lacks exactly 90 cells replicated over three mask seeds")

    r2_exact = 0
    coeff_exact = 0
    prediction_exact = 0
    repeated_all_contracts = 0
    max_r2_diff = 0.0
    for base_key, repeated_cells in all_index.items():
        reference = a2a_index[base_key]
        canonical = next(cell for cell in repeated_cells if cell.get("mask_seed") == SEEDS[0])
        r2 = canonical.get("r2")
        ref_r2 = reference.get("r2")
        _require(isinstance(r2, (int, float)) and isinstance(ref_r2, (int, float)) and math.isfinite(float(r2)) and math.isfinite(float(ref_r2)), "K=all/A2a R² is invalid")
        max_r2_diff = max(max_r2_diff, abs(float(r2) - float(ref_r2)))
        _require(float(r2) == float(ref_r2), f"K=all R² is not exact for {base_key}")
        _require(canonical.get("coefficients_sha256") == reference.get("coefficients_sha256"), f"K=all coefficient SHA mismatch for {base_key}")
        r2_exact += 1
        coeff_exact += 1
        prediction_exact += int(canonical.get("prediction_sha256") == reference.get("prediction_sha256"))
        for cell in repeated_cells:
            _require(cell.get("r2") == canonical.get("r2") and cell.get("coefficients_sha256") == canonical.get("coefficients_sha256") and cell.get("prediction_sha256") == canonical.get("prediction_sha256"), f"K=all seed-reuse artifacts disagree for {base_key}")
        repeated_all_contracts += 1

    batch_names = receipt.get("batch_files")
    _require(isinstance(batch_names, list) and len(batch_names) == 12 and all(isinstance(name, str) for name in batch_names), "A2b receipt must name exactly 12 batch files")
    _require(len(set(batch_names)) == 12, "A2b receipt batch-file list contains duplicates")
    actual_names = sorted(path.name for path in batch_dir.glob("batch_*.json")) if batch_dir.is_dir() else []
    _require(sorted(batch_names) == actual_names, "retained batch directory does not exactly match the receipt batch list")
    concatenated_cells: list[Any] = []
    batch_rows: list[dict[str, Any]] = []
    expected_start = 0
    for name in batch_names:
        path = batch_dir / name
        batch = _load_json(path)
        start, end = batch.get("session_start"), batch.get("session_end")
        _require(isinstance(start, int) and isinstance(end, int) and start == expected_start and end > start, f"batch session ranges are not contiguous at {name}")
        expected_start = end
        batch_cells = batch.get("cells")
        _require(isinstance(batch_cells, list) and len(batch_cells) == (end - start) * 2 * 3 * 3 * 6, f"batch cell count is invalid for {name}")
        concatenated_cells.extend(batch_cells)
        batch_rows.append({"name": name, "sha256": _sha256_file(path), "bytes": path.stat().st_size, "session_start": start, "session_end": end, "cells": len(batch_cells)})
    _require(expected_start == 15, "batch ranges do not cover exactly the 15-session cohort")
    _require(_canonical_json_bytes(concatenated_cells) == _canonical_json_bytes(typed_cells), "final receipt cells are not byte-for-byte equal under canonical JSON to ordered batch-cell concatenation")

    final_cells_sha = hashlib.sha256(_canonical_json_bytes(typed_cells)).hexdigest()
    batch_cells_sha = hashlib.sha256(_canonical_json_bytes(concatenated_cells)).hexdigest()
    return {
        "schema": "priority_a2b_v2_posthoc_archival_integrity_v1",
        "status": "PASS",
        "attestation_scope": "Post-hoc archival integrity closure. It verifies internal consistency of currently retained artifacts, not producer lineage, scheduler history, or historical file-creation order.",
        "receipt": {"path": str(receipt_path), "sha256": _sha256_file(receipt_path)},
        "a2a_reference": {"path": str(a2a_receipt_path), "sha256": _sha256_file(a2a_receipt_path)},
        "runner_core_bindings": {"runner_sha256": bindings["runner_sha256"], "ridge_core_sha256": bindings["ridge_core_sha256"]},
        "grid": {"expected_cells": 1620, "observed_cells": len(typed_cells), "assets": len(assets), "views": len(VIEWS), "budgets": len(BUDGETS), "mask_seeds": len(SEEDS), "K_levels": len(FINITE_KS) + 1},
        "k_all_reproduction": {"unique_base_cells": len(all_index), "replicated_cells": sum(len(value) for value in all_index.values()), "r2_exact_matches": r2_exact, "coefficient_sha_exact_matches": coeff_exact, "prediction_sha_exact_matches": prediction_exact, "seed_reuse_contracts_checked": repeated_all_contracts, "max_r2_abs_diff": max_r2_diff},
        "batch_archival_integrity": {"n_batch_files": len(batch_rows), "batches": batch_rows, "receipt_cells_canonical_sha256": final_cells_sha, "ordered_batch_cells_canonical_sha256": batch_cells_sha, "canonical_cells_byte_for_byte_match": final_cells_sha == batch_cells_sha},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc archival-integrity closure for retained A2b-v2 artifacts")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--a2a-receipt", type=Path, default=DEFAULT_A2A_RECEIPT)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--write-manifest", type=Path, default=None, help="Write this post-hoc integrity report as a new manifest")
    args = parser.parse_args()
    report = verify_archival_integrity(receipt_path=args.receipt, a2a_receipt_path=args.a2a_receipt, batch_dir=args.batch_dir)
    if args.write_manifest is not None:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_bytes(_canonical_json_bytes(report))
        report = {**report, "written_manifest": {"path": str(args.write_manifest), "sha256": _sha256_file(args.write_manifest)}}
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
