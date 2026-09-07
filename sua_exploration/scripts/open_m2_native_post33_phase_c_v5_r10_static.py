#!/usr/bin/env python3
"""Frozen r10 delayed-score opener, callable only by the single supervisor.

It first seals static score-blind Stage-A/matrix manifests, then reads opaque
0600 payloads only after the exact cardinality barrier.  It never prints a
score, delta, aggregate, or decision value.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_evaluator_v4 import validate_endpoint_payload  # noqa: E402
from sua_exploration.mc_maze.m2_native_post33_openers_v4 import (  # noqa: E402
    compute_full_gates, stage_a_decision_from_deltas,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    ARMS, FOLDS, PHASE_ID, PROTOCOL_ID, SEEDS, CellKey, cell_paths, file_metadata,
    matrix_paths, stage_a_paths, validate_selector_payload, verify_cell_exact,
    write_json_exclusive,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r10_static import (  # noqa: E402
    R10_CELL_ROOT, STATIC_MANIFEST, manifest_metadata, validate_static_manifest,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_private(path: Path, payload: Mapping[str, Any]) -> Path:
    result = write_json_exclusive(path, payload)
    os.chmod(result, 0o600)
    return result


def _score(key: CellKey) -> float:
    """The only r10 endpoint reader; reachable only after static exactness."""
    verify_cell_exact(R10_CELL_ROOT, key)
    paths = cell_paths(R10_CELL_ROOT, key)
    selector = json.loads(paths["selector_records"].read_text(encoding="utf-8"))
    selected = validate_selector_payload(selector, key, run_dir=paths["run"])
    payload = json.loads(paths["opaque_payload"].read_text(encoding="utf-8"))
    return float(validate_endpoint_payload(payload, key=key, selected_checkpoint=selected["checkpoint_path"], resolved_config=paths["resolved_config"]))


def _seal_stage_a() -> Path:
    destinations = stage_a_paths(R10_CELL_ROOT)
    _require(not destinations["directory"].exists(), "r10 Stage-A static manifest is write-once")
    cells = []
    for arm in ARMS:
        for fold in FOLDS:
            cells.append(verify_cell_exact(R10_CELL_ROOT, CellKey(PROTOCOL_ID, arm, fold, 42)))
    pairs = [
        {"fold": fold, "seed": 42, "spint_result_sha256": cells[fold]["result"]["sha256"], "t4_result_sha256": cells[7 + fold]["result"]["sha256"], "endpoint_values_opened": False, "score_delta_computed": False}
        for fold in FOLDS
    ]
    manifest = {
        "schema": "m2_post33_phase_c_v5_r10_static_stage_a_manifest_v1",
        "protocol_id": PROTOCOL_ID, "phase_id": PHASE_ID,
        "static_manifest": manifest_metadata(), "cell_count": 14, "pair_count": 7,
        "cells": cells, "pairs": pairs, "endpoint_values_opened": False,
        "score_aggregation_performed": False,
    }
    manifest_path = _write_private(destinations["manifest"], manifest)
    _write_private(destinations["completed"], {
        "schema": "m2_post33_phase_c_v5_r10_static_stage_a_status_v1",
        "state": "completed", "manifest": file_metadata(manifest_path),
    })
    return manifest_path


def _open_stage_a() -> None:
    validate_static_manifest(require_lock=True)
    manifest_path = _seal_stage_a()
    paths = stage_a_paths(R10_CELL_ROOT)
    rows = []
    for fold, session in FOLDS.items():
        spint = _score(CellKey(PROTOCOL_ID, "spint", fold, 42))
        t4 = _score(CellKey(PROTOCOL_ID, "t4", fold, 42))
        rows.append({"fold": fold, "outer_session": session, "spint_r2": spint, "t4_r2": t4, "delta": t4 - spint})
    decision = stage_a_decision_from_deltas([row["delta"] for row in rows])
    _write_private(paths["decision"], {
        "schema": "m2_post33_phase_c_v5_r10_static_opened_stage_a_decision_v1",
        "protocol_id": PROTOCOL_ID, "phase_id": PHASE_ID,
        "absolute_cell_root": str(R10_CELL_ROOT.resolve()), "stage_a_manifest": file_metadata(manifest_path),
        "static_manifest": manifest_metadata(), "cell_count": 14, "pair_count": 7,
        "seed": 42, "paired_rows": rows, **decision,
    })


def _validate_stage_a_continue() -> None:
    path = stage_a_paths(R10_CELL_ROOT)["decision"]
    _require(path.is_file(), "r10 Stage-A decision missing")
    decision = json.loads(path.read_text(encoding="utf-8"))
    _require(decision.get("schema") == "m2_post33_phase_c_v5_r10_static_opened_stage_a_decision_v1", "r10 Stage-A decision schema mismatch")
    _require(decision.get("decision") == "continue_without_positive_claim" and decision.get("positive_claim_made") is False, "r10 Stage-A decision forbids full opening")


def _seal_matrix() -> Path:
    paths = matrix_paths(R10_CELL_ROOT)
    _require(not paths["directory"].exists(), "r10 matrix static manifest is write-once")
    cells = [verify_cell_exact(R10_CELL_ROOT, CellKey(PROTOCOL_ID, arm, fold, seed)) for arm in ARMS for fold in FOLDS for seed in SEEDS]
    by = {(row["cell"]["arm"], row["cell"]["fold"], row["cell"]["seed"]): row for row in cells}
    pairs = [
        {"fold": fold, "seed": seed, "spint_result_sha256": by[("spint", fold, seed)]["result"]["sha256"], "t4_result_sha256": by[("t4", fold, seed)]["result"]["sha256"], "endpoint_values_opened": False, "score_delta_computed": False}
        for fold in FOLDS for seed in SEEDS
    ]
    manifest = {
        "schema": "m2_post33_phase_c_v5_r10_static_matrix_manifest_v1",
        "protocol_id": PROTOCOL_ID, "phase_id": PHASE_ID, "static_manifest": manifest_metadata(),
        "cell_count": 42, "pair_count": 21, "cells": cells, "pairs": pairs,
        "endpoint_values_opened": False, "score_aggregation_performed": False,
    }
    output = _write_private(paths["manifest"], manifest)
    _write_private(paths["completed"], {"schema": "m2_post33_phase_c_v5_r10_static_matrix_status_v1", "state": "completed", "manifest": file_metadata(output)})
    return output


def _open_full() -> None:
    validate_static_manifest(require_lock=True)
    _validate_stage_a_continue()
    matrix_manifest = _seal_matrix()
    paths = matrix_paths(R10_CELL_ROOT)
    spint = np.empty((3, 7), dtype=np.float64)
    t4 = np.empty((3, 7), dtype=np.float64)
    rows = []
    for index, seed in enumerate(SEEDS):
        for fold, session in FOLDS.items():
            spint_value = _score(CellKey(PROTOCOL_ID, "spint", fold, seed))
            t4_value = _score(CellKey(PROTOCOL_ID, "t4", fold, seed))
            spint[index, fold], t4[index, fold] = spint_value, t4_value
            rows.append({"seed": seed, "fold": fold, "outer_session": session, "spint_r2": spint_value, "t4_r2": t4_value, "delta": t4_value - spint_value})
    _write_private(paths["opened_aggregate"], {
        "schema": "m2_post33_phase_c_v5_r10_static_opened_full_aggregate_v1",
        "protocol_id": PROTOCOL_ID, "phase_id": PHASE_ID, "absolute_cell_root": str(R10_CELL_ROOT.resolve()),
        "matrix_manifest": file_metadata(matrix_manifest), "stage_a_decision": file_metadata(stage_a_paths(R10_CELL_ROOT)["decision"]),
        "static_manifest": manifest_metadata(), "cell_count": 42, "pair_count": 21,
        "seed_by_session_rows": rows, **compute_full_gates(t4 - spint, spint, t4),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--opening", choices=("stage_a", "full"), required=True)
    args = parser.parse_args()
    _require(args.cell_root.resolve() == R10_CELL_ROOT.resolve(), "r10 opener root substitution")
    if args.opening == "stage_a":
        _open_stage_a()
    else:
        _open_full()


if __name__ == "__main__":
    main()
