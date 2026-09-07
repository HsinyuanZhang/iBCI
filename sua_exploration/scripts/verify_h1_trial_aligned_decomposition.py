#!/usr/bin/env python3
"""Independent trial-aligned H1 decomposition verifier (CPU, sealed-primitive bound)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2

SCHEMA = "h1_trial_aligned_decomposition_independent_v1"
EXPECTED_NEURONS = 176
LATENT_DIM = 4
CARRIER_DIM = 5
RIDGE_LAMBDA = 3.0
MIN_SUPPORT_EVENTS = 8
NORM_FLOOR = 1.0e-12
REQUIRED_CELLS: tuple[tuple[int, int], ...] = ((1, 4), (2, 4), (3, 4), (4, 4), (3, 3))
SEALED_CROSSCHECK_CELLS: tuple[tuple[int, int], ...] = ((3, 3), (4, 4))
SCRIPT_PATH = Path(__file__).resolve()


class ReceiptExistsError(FileExistsError):
    """Raised when the output receipt already exists."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_support(events: Sequence[v1.MovementEvent], support_budget: int) -> tuple[v1.MovementEvent, ...]:
    return tuple(event for event in events if event.trial_index < support_budget)


def select_eval(events: Sequence[v1.MovementEvent], eval_threshold: int) -> tuple[v1.MovementEvent, ...]:
    return tuple(event for event in events if event.trial_index >= eval_threshold)


def fit_carrier_ridge(z: np.ndarray, response: np.ndarray) -> np.ndarray | None:
    z = np.asarray(z, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    n = z.shape[0]
    if n < MIN_SUPPORT_EVENTS:
        return None
    design = np.column_stack((np.ones(n, dtype=np.float64), z))
    if np.linalg.matrix_rank(design) != CARRIER_DIM:
        return None
    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0]) * (n * RIDGE_LAMBDA)
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    if not np.isfinite(coefficient).all():
        return None
    return np.column_stack((coefficient[1:].T, coefficient[0]))


def predict_from_carrier(carrier: np.ndarray, z_eval: np.ndarray) -> np.ndarray:
    carrier = np.asarray(carrier, dtype=np.float64)
    z_eval = np.asarray(z_eval, dtype=np.float64)
    return z_eval @ carrier[:, :LATENT_DIM].T + carrier[:, LATENT_DIM][None, :]


def r2_by_channel(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    total = np.sum(np.square(observed - observed.mean(axis=0, keepdims=True)), axis=0)
    residual = np.sum(np.square(observed - predicted), axis=0)
    output = np.full(observed.shape[1], np.nan, dtype=np.float64)
    defined = total > NORM_FLOOR
    output[defined] = 1.0 - residual[defined] / total[defined]
    return output


def event_arrays(
    events: Sequence[v1.MovementEvent], basis: v2.EndpointBasisV2,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = np.stack([event.displacement for event in events]).astype(np.float64)
    response = np.stack([event.log_rates for event in events]).astype(np.float64)
    return basis.transform(displacement), response


def compute_cell(
    session: v1.EventSession,
    basis: v2.EndpointBasisV2,
    support_budget: int,
    eval_threshold: int,
) -> dict[str, Any]:
    support = select_support(session.events, support_budget)
    eval_events = select_eval(session.events, eval_threshold)
    base = {
        "support_budget": support_budget,
        "eval_threshold": eval_threshold,
        "support_events": len(support),
        "eval_events": len(eval_events),
    }
    if len(support) < MIN_SUPPORT_EVENTS or not eval_events:
        return {**base, "status": "undefined", "median_r2_correct": None, "median_delta_intercept": None}
    z_support, y_support = event_arrays(support, basis)
    carrier = fit_carrier_ridge(z_support, y_support)
    if carrier is None:
        return {**base, "status": "undefined", "median_r2_correct": None, "median_delta_intercept": None}
    z_eval, observed = event_arrays(eval_events, basis)
    predicted = predict_from_carrier(carrier, z_eval)
    support_mean = np.mean(np.stack([event.log_rates for event in support]), axis=0)
    intercept_prediction = np.broadcast_to(support_mean[None, :], observed.shape)
    r_correct = r2_by_channel(observed, predicted)
    r_intercept = r2_by_channel(observed, intercept_prediction)
    delta_intercept = r_correct - r_intercept
    r2_defined = np.isfinite(r_correct)
    delta_defined = np.isfinite(r_correct) & np.isfinite(r_intercept)
    if not np.any(r2_defined) and not np.any(delta_defined):
        return {**base, "status": "undefined", "median_r2_correct": None, "median_delta_intercept": None}
    median_r2 = float(np.median(r_correct[r2_defined])) if np.any(r2_defined) else None
    median_delta = float(np.median(delta_intercept[delta_defined])) if np.any(delta_defined) else None
    return {
        **base,
        "status": "defined",
        "median_r2_correct": median_r2,
        "median_delta_intercept": median_delta,
    }


def cell_key(support_budget: int, eval_threshold: int) -> str:
    return f"M{support_budget}_E{eval_threshold}"


def session_decomposition(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None]:
    def delta(key: str) -> float | None:
        value = cells[key].get("median_delta_intercept")
        if value is None or not math.isfinite(float(value)):
            return None
        return float(value)

    c33 = delta(cell_key(3, 3))
    c34 = delta(cell_key(3, 4))
    c44 = delta(cell_key(4, 4))
    if c33 is None or c34 is None or c44 is None:
        return {"eval_effect": None, "support_effect": None, "total": None}
    eval_effect = c33 - c34
    support_effect = c34 - c44
    total = c33 - c44
    if abs(total - (eval_effect + support_effect)) > 1.0e-12:
        raise AssertionError(
            f"decomposition identity failed: total={total}, sum={eval_effect + support_effect}"
        )
    return {
        "eval_effect": eval_effect,
        "support_effect": support_effect,
        "total": total,
    }


def aggregate_quantities(rows: Mapping[str, float | None]) -> dict[str, Any]:
    values = [float(value) for value in rows.values() if value is not None and math.isfinite(float(value))]
    if not values:
        return {"defined_sessions": 0, "mean": None, "median": None, "positive": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "defined_sessions": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "positive": int(np.sum(array > 0)),
    }


def cell_table_summary(per_session_cells: Mapping[str, Mapping[str, Mapping[str, Any]]], cell: tuple[int, int]) -> dict[str, Any]:
    key = cell_key(cell[0], cell[1])
    deltas: list[float] = []
    for session in v1.H1_HELDIN_SESSIONS:
        row = per_session_cells[session][key]
        value = row.get("median_delta_intercept")
        if row.get("status") == "defined" and value is not None and math.isfinite(float(value)):
            deltas.append(float(value))
    if not deltas:
        return {"cell": list(cell), "defined_sessions": 0, "mean": None, "median": None, "positive": 0}
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "cell": list(cell),
        "defined_sessions": len(deltas),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "positive": int(np.sum(array > 0)),
    }


def compare_sealed_helper(
    sessions: Mapping[str, v1.EventSession],
    bases: Mapping[str, v2.EndpointBasisV2],
    independent_cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for session_name in v1.H1_HELDIN_SESSIONS:
        session = sessions[session_name]
        basis = bases[session_name]
        for support_budget, eval_threshold in SEALED_CROSSCHECK_CELLS:
            key = cell_key(support_budget, eval_threshold)
            ours = independent_cells[session_name][key]
            sealed = v2.forward_transfer(session, basis, budget=support_budget)
            for field in ("median_r2_correct", "median_delta_intercept"):
                independent_value = ours.get(field)
                sealed_value = sealed.get(field)
                if independent_value is None or sealed_value is None:
                    diff = None
                else:
                    diff = abs(float(independent_value) - float(sealed_value))
                    if diff is not None:
                        max_abs_diff = max(max_abs_diff, diff)
                comparisons.append({
                    "session": session_name,
                    "cell": [support_budget, eval_threshold],
                    "field": field,
                    "independent": independent_value,
                    "sealed_forward_transfer": sealed_value,
                    "abs_diff": diff,
                })
    return {"comparisons": comparisons, "max_abs_diff": max_abs_diff}


def run_computation(data_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    index = v1.index_heldin_calib(data_root)
    sessions = {name: v1.load_event_session(path) for name, path in index.items()}
    bases = {
        name: v2.fit_source_all_event_basis(sessions, outer_date=v1.session_date(name))
        for name in v1.H1_HELDIN_SESSIONS
    }

    per_session_cells: dict[str, dict[str, dict[str, Any]]] = {}
    per_session_decomposition: dict[str, dict[str, float | None]] = {}
    for session_name in v1.H1_HELDIN_SESSIONS:
        session = sessions[session_name]
        basis = bases[session_name]
        cells: dict[str, dict[str, Any]] = {}
        for support_budget, eval_threshold in REQUIRED_CELLS:
            cells[cell_key(support_budget, eval_threshold)] = compute_cell(
                session, basis, support_budget, eval_threshold,
            )
        per_session_cells[session_name] = cells
        per_session_decomposition[session_name] = session_decomposition(cells)

    decomposition_aggregates = {
        name: aggregate_quantities({session: per_session_decomposition[session][name] for session in v1.H1_HELDIN_SESSIONS})
        for name in ("eval_effect", "support_effect", "total")
    }
    five_cell_table = [cell_table_summary(per_session_cells, cell) for cell in REQUIRED_CELLS]
    sealed_comparison = compare_sealed_helper(sessions, bases, per_session_cells)

    runtime_seconds = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "cells": REQUIRED_CELLS,
        "per_session_cells": {
            session: {
                key: {
                    "median_r2_correct": row["median_r2_correct"],
                    "median_delta_intercept": row["median_delta_intercept"],
                    "status": row["status"],
                    "support_events": row["support_events"],
                    "eval_events": row["eval_events"],
                }
                for key, row in cells.items()
            }
            for session, cells in per_session_cells.items()
        },
        "per_session_decomposition": per_session_decomposition,
        "decomposition_aggregates": decomposition_aggregates,
        "five_cell_table": five_cell_table,
        "sealed_helper_comparison": sealed_comparison,
        "source_binding": {
            "sessions": list(v1.H1_HELDIN_SESSIONS),
            "dates": list(v1.H1_DATES),
            "files": [
                {
                    "session": name,
                    "path": str(sessions[name].path),
                    "input_sha256": sessions[name].input_sha256,
                }
                for name in v1.H1_HELDIN_SESSIONS
            ],
        },
        "implementation_binding": {
            "script_path": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
        },
        "runtime_seconds": runtime_seconds,
        "scope": {
            "cuda_used": False,
            "dense_velocity_series_opened": False,
        },
        "sessions": sessions,
        "bases": bases,
    }


def write_receipt(output: Path, body: Mapping[str, Any]) -> str:
    output = output.resolve()
    if output.exists():
        raise ReceiptExistsError(f"refusing to overwrite existing receipt: {output}")
    serializable = {key: value for key, value in body.items() if key not in {"sessions", "bases"}}
    encoded = (json.dumps(serializable, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    output.chmod(0o444)
    if stat.S_IMODE(output.stat().st_mode) != 0o444:
        raise OSError(f"failed to set receipt mode 0444: {output}")
    return hashlib.sha256(encoded).hexdigest()


def print_summary(body: Mapping[str, Any]) -> None:
    print("Five-cell median_delta_intercept summary:")
    print(f"{'Cell':<10} {'Defined':>8} {'Mean':>12} {'Median':>12} {'Positive':>8}")
    for row in body["five_cell_table"]:
        cell = tuple(row["cell"])
        mean = row["mean"]
        median = row["median"]
        mean_s = f"{mean:12.6f}" if mean is not None else f"{'n/a':>12}"
        median_s = f"{median:12.6f}" if median is not None else f"{'n/a':>12}"
        print(f"{str(cell):<10} {row['defined_sessions']:8d} {mean_s} {median_s} {row['positive']:8d}")

    print("\nDecomposition aggregates (median_delta_intercept):")
    for name in ("eval_effect", "support_effect", "total"):
        agg = body["decomposition_aggregates"][name]
        mean = agg["mean"]
        median = agg["median"]
        mean_s = f"{mean:.6f}" if mean is not None else "n/a"
        median_s = f"{median:.6f}" if median is not None else "n/a"
        print(f"  {name}: defined={agg['defined_sessions']}, mean={mean_s}, median={median_s}, positive={agg['positive']}")

    print(f"\nMax |independent - sealed forward_transfer| = {body['sealed_helper_comparison']['max_abs_diff']:.6e}")
    print(f"Runtime: {body['runtime_seconds']:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("SPINT-main/data/000954"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sua_exploration/results/h1_trial_aligned_sweep/independent_crosscheck.json"),
    )
    args = parser.parse_args()

    body = run_computation(args.data_root.resolve())
    receipt_digest = write_receipt(args.output, body)
    print_summary(body)
    print(f"\nReceipt: {args.output.resolve()}")
    print(f"Receipt sha256: {receipt_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
