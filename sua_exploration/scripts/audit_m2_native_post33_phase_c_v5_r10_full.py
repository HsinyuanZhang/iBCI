#!/usr/bin/env python3
"""Fail-closed, read-only audit of an already-opened native-M2 r10 result.

The aggregate must already exist before this program reads any cell or invokes
the static guard.  It never launches training, creates a result artifact, or
acts as a score-opening path.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_openers_v4 import compute_full_gates  # noqa: E402
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    ARMS, FOLDS, PHASE_ID, PROTOCOL_ID, SEEDS, CellKey, cell_paths,
    file_metadata, matrix_paths, phase_root, stage_a_paths, verify_cell_exact,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r10_static import (  # noqa: E402
    R10_CELL_ROOT, RUN_LOCK, STATIC_MANIFEST, cell_schedule, manifest_metadata,
    validate_static_manifest,
)

SCHEMA = "m2_post33_phase_c_v5_r10_full_result_audit_v1"
FULL_SCHEMA = "m2_post33_phase_c_v5_r10_static_opened_full_aggregate_v1"


class AuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AuditError(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or noncanonical")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label} is not JSON") from exc
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    value = float(value)
    require(math.isfinite(value), f"{label} is not finite")
    return value


def validate_rows_and_gates(
    rows: Any, *, stage_rows: Any = None,
) -> dict[str, Any]:
    """Pure aggregate-row/gate checker; no filesystem, model, or score reads."""
    require(isinstance(rows, list) and len(rows) == 21, "full aggregate needs 21 rows")
    if stage_rows is not None:
        require(isinstance(stage_rows, list) and len(stage_rows) == 7, "Stage-A needs 7 rows")
    spint, t4 = np.empty((3, 7)), np.empty((3, 7))
    for si, seed in enumerate(SEEDS):
        for fold, session in FOLDS.items():
            row = rows[si * 7 + fold]
            require(isinstance(row, dict) and set(row) == {"seed", "fold", "outer_session", "spint_r2", "t4_r2", "delta"}, "full row schema mismatch")
            require((row["seed"], row["fold"], row["outer_session"]) == (seed, fold, session), "full row order/identity mismatch")
            spint_value, t4_value, delta = (number(row[name], name) for name in ("spint_r2", "t4_r2", "delta"))
            require(row["delta"] == t4_value - spint_value, "full row delta mismatch")
            if seed == 42 and stage_rows is not None:
                require(row == {"seed": 42, **stage_rows[fold]}, "seed-42 row differs from Stage-A")
            spint[si, fold], t4[si, fold] = spint_value, t4_value
    return compute_full_gates(t4 - spint, spint, t4)


def audit(cell_root: str | Path | None = None, aggregate: str | Path | None = None) -> dict[str, Any]:
    root = Path(R10_CELL_ROOT if cell_root is None else cell_root)
    require(root.is_dir() and not root.is_symlink(), "r10 cell root missing or noncanonical")
    root = root.resolve(strict=True)
    require(root == Path(R10_CELL_ROOT).resolve(), "--cell-root differs from authoritative r10 root")
    paths, stage = matrix_paths(root), stage_a_paths(root)
    aggregate_path = paths["opened_aggregate"] if aggregate is None else Path(aggregate)
    require(aggregate_path.resolve() == paths["opened_aggregate"].resolve(), "--aggregate differs from authoritative r10 path")

    # The first result-bearing operation: no pre-score cell/static path is read.
    result = read_json(aggregate_path, "opened full aggregate")
    try:
        validate_static_manifest(require_lock=True)
    except Exception as exc:
        raise AuditError(f"static manifest/run lock invalid: {exc}") from exc
    static = manifest_metadata()
    require(static == file_metadata(STATIC_MANIFEST), "static manifest metadata mismatch")
    require(RUN_LOCK.is_file() and not RUN_LOCK.is_symlink(), "r10 run lock missing")

    schedule = cell_schedule()
    keys = tuple(schedule["stage_a"]) + tuple(schedule["stage_b"])
    require(len(keys) == 42 and len(set(keys)) == 42, "r10 schedule is not exact 42")
    expected_dirs = {cell_paths(root, key)["cell_dir"].resolve() for key in keys}
    observed_dirs = {path.resolve() for path in (phase_root(root) / "cells").glob("arm-*/fold-*/seed-*") if path.is_dir() and not path.is_symlink()}
    require(observed_dirs == expected_dirs, "r10 sealed cell directory set is not exact")
    verified = {(key.arm, key.fold, key.seed): verify_cell_exact(root, key) for key in keys}
    require(len(verified) == 42 and all(row.get("synthetic_decoder_evidence") is False for row in verified.values()), "r10 exact sealed-cell verification failed")

    matrix = read_json(paths["manifest"], "matrix manifest")
    status, decision = read_json(paths["completed"], "matrix status"), read_json(stage["decision"], "Stage-A decision")
    require(matrix.get("schema") == "m2_post33_phase_c_v5_r10_static_matrix_manifest_v1", "matrix schema mismatch")
    require(matrix.get("static_manifest") == static and matrix.get("cell_count") == 42 and matrix.get("pair_count") == 21, "matrix metadata/cardinality mismatch")
    require(isinstance(matrix.get("cells"), list) and len(matrix["cells"]) == 42 and isinstance(matrix.get("pairs"), list) and len(matrix["pairs"]) == 21, "matrix needs 42 cells and 21 pairs")
    require(status == {"schema": "m2_post33_phase_c_v5_r10_static_matrix_status_v1", "state": "completed", "manifest": file_metadata(paths["manifest"])}, "matrix status mismatch")
    require(decision.get("schema") == "m2_post33_phase_c_v5_r10_static_opened_stage_a_decision_v1" and decision.get("decision") == "continue_without_positive_claim" and decision.get("positive_claim_made") is False, "Stage-A does not permit full result")

    expected_pairs = [{"fold": fold, "seed": seed, "spint_result_sha256": verified[("spint", fold, seed)]["result"]["sha256"], "t4_result_sha256": verified[("t4", fold, seed)]["result"]["sha256"], "endpoint_values_opened": False, "score_delta_computed": False} for fold in FOLDS for seed in SEEDS]
    expected_cells = [verified[(arm, fold, seed)] for arm in ARMS for fold in FOLDS for seed in SEEDS]
    require(matrix["cells"] == expected_cells, "matrix sealed-cell bindings mismatch")
    require(matrix["pairs"] == expected_pairs, "matrix pair bindings mismatch")
    require(result.get("schema") == FULL_SCHEMA and result.get("protocol_id") == PROTOCOL_ID and result.get("phase_id") == PHASE_ID, "full aggregate identity mismatch")
    require(result.get("absolute_cell_root") == str(root) and result.get("cell_count") == 42 and result.get("pair_count") == 21 and result.get("matrix_manifest") == file_metadata(paths["manifest"]) and result.get("stage_a_decision") == file_metadata(stage["decision"]) and result.get("static_manifest") == static, "full aggregate metadata mismatch")
    summary = validate_rows_and_gates(result.get("seed_by_session_rows"), stage_rows=decision.get("paired_rows"))
    require(all(result.get(name) == value for name, value in summary.items()), "recomputed full gates differ from aggregate")
    return {"schema": SCHEMA, "status": "PASS_READ_ONLY_R10_FULL_AUDIT", "absolute_cell_root": str(root), "aggregate": file_metadata(aggregate_path), "static_manifest": static, "run_lock": file_metadata(RUN_LOCK), "cell_count": 42, "pair_count": 21, "all_six_pass": summary["all_six_pass"], "gates": summary["gates"]}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-root", type=Path, default=R10_CELL_ROOT)
    parser.add_argument("--aggregate", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload, code = audit(args.cell_root, args.aggregate), 0
    except AuditError as exc:
        payload, code = {"schema": SCHEMA, "status": "FAIL_CLOSED", "reason": str(exc)}, 2
    except Exception as exc:
        payload, code = {"schema": SCHEMA, "status": "FAIL_CLOSED", "reason": f"unexpected audit failure: {exc}"}, 3
    print(json.dumps(payload, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
