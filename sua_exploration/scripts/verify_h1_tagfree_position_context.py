#!/usr/bin/env python3
"""Independent cross-check for H1 tag-free position-context carrier screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze.h1_event_carrier_design_screen import (
    FEATURE_SCALE_FLOOR,
    LATENT_SCALE_FLOOR,
    SOURCE_RIDGE_LAMBDA,
    TAG_NAMES,
    TARGET_RIDGE_LAMBDA,
    ContextEvent,
    ContextSession,
    load_context_sessions,
)

SCHEMA = "h1_tagfree_position_context_independent_crosscheck_v1"
SCRIPT_PATH = Path(__file__).resolve()
SEALED_RECEIPT_SHA256 = "74bbc01490432794546e7ca2fd4242fbed6f2a7ebdd65786f56035eaa49bfeb3"
EXPECTED_NEURONS = 176
RANK = 4
NORM_FLOOR = 1.0e-12
REFERENCE_ARMS = ("pca_delta_q4", "ser_context_q4")
BUDGETS = (3, 4)

CANDIDATE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("pca_delta_q4", "pca", "delta"),
    ("ser_context_q4", "source_encoding", "context"),
    ("ser_poscontext_q4", "source_encoding", "poscontext"),
    ("pca_poscontext_q4", "pca", "poscontext"),
    ("ser_startstop_q4", "source_encoding", "startstop"),
    ("ser_deltastart_q4", "source_encoding", "deltastart"),
)


@dataclass(frozen=True)
class LatentBasis:
    name: str
    basis_mode: str
    feature_family: str
    outer_date: str
    source_sessions: tuple[str, ...]
    active_mask: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    projection: np.ndarray
    latent_scale: np.ndarray
    energy_ratio: np.ndarray
    retained_energy_at_rank: float


class ReceiptExistsError(FileExistsError):
    """Raised when the output receipt already exists."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag_index_map() -> dict[str, int]:
    return {tag: index for index, tag in enumerate(TAG_NAMES)}


def build_onehot(tags: Sequence[str]) -> np.ndarray:
    lookup = tag_index_map()
    output = np.zeros((len(tags), len(TAG_NAMES)), dtype=np.float64)
    for row, tag in enumerate(tags):
        output[row, lookup[tag]] = 1.0
    return output


def build_raw_features(
    events: Sequence[ContextEvent],
    family: str,
    *,
    tags: Sequence[str] | None = None,
) -> np.ndarray:
    if not events:
        raise ValueError("carrier feature set is empty")
    if tags is None:
        tags = [event.base.tag for event in events]
    if len(tags) != len(events):
        raise ValueError("tag override length mismatch")

    delta = np.vstack([event.base.displacement for event in events]).astype(np.float64)
    midpoint = np.vstack([event.midpoint_state for event in events]).astype(np.float64)
    start = np.vstack([event.start_state for event in events]).astype(np.float64)
    stop = start + delta

    if family == "delta":
        output = delta
    elif family == "context":
        output = np.hstack((delta, midpoint, build_onehot(tags)))
    elif family == "poscontext":
        output = np.hstack((delta, midpoint))
    elif family == "startstop":
        output = np.hstack((start, stop))
    elif family == "deltastart":
        output = np.hstack((delta, start))
    else:
        raise ValueError(f"unknown feature family {family!r}")

    if not np.isfinite(output).all():
        raise ValueError(f"{family}: nonfinite raw feature")
    return output


def canonicalize_projection(projection: np.ndarray) -> np.ndarray:
    matrix = np.asarray(projection, dtype=np.float64).copy()
    for column in range(matrix.shape[1]):
        pivot = int(np.argmax(np.abs(matrix[:, column])))
        if matrix[pivot, column] < 0.0:
            matrix[:, column] *= -1.0
    return matrix


def standardize_columns(raw: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_mean = raw.mean(axis=0)
    raw_scale = raw.std(axis=0, ddof=0)
    active = raw_scale > floor
    mean = raw_mean[active]
    scale = raw_scale[active]
    standardized = (raw[:, active] - mean[None, :]) / scale[None, :]
    return active, mean, scale, standardized


def fit_pca_projection(standardized: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    _left, singular, right = np.linalg.svd(standardized, full_matrices=False)
    projection = right[:rank].T
    energy = np.square(singular) / np.square(singular).sum()
    return projection, energy


def fit_source_encoding_projection(
    standardized: np.ndarray,
    sessions: Mapping[str, ContextSession],
    source_names: Sequence[str],
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    blocks: list[np.ndarray] = []
    offset = 0
    for name in source_names:
        count = len(sessions[name].events)
        xs = standardized[offset : offset + count]
        response = np.vstack([event.base.log_rates for event in sessions[name].events]).astype(np.float64)
        response_mean = response.mean(axis=0, keepdims=True)
        response_scale = np.maximum(response.std(axis=0, keepdims=True, ddof=0), 1.0e-6)
        ys = (response - response_mean) / response_scale
        penalty = np.eye(xs.shape[1], dtype=np.float64) * (count * SOURCE_RIDGE_LAMBDA)
        coefficient = np.linalg.solve(xs.T @ xs + penalty, xs.T @ ys)
        fro = float(np.linalg.norm(coefficient, ord="fro"))
        coefficient /= max(fro, 1.0e-12)
        blocks.append(coefficient)
        offset += count
    stacked = np.hstack(blocks)
    left, singular, _right = np.linalg.svd(stacked, full_matrices=False)
    projection = left[:, :rank]
    energy = np.square(singular) / np.square(singular).sum()
    return projection, energy


def fit_latent_basis(
    sessions: Mapping[str, ContextSession],
    *,
    name: str,
    basis_mode: str,
    feature_family: str,
    outer_date: str,
) -> LatentBasis:
    source_names = tuple(
        session_name for session_name in v1.H1_HELDIN_SESSIONS if v1.session_date(session_name) != outer_date
    )
    pooled_events = [event for session_name in source_names for event in sessions[session_name].events]
    if not pooled_events:
        raise ValueError(f"{outer_date}: empty source event pool")

    raw = build_raw_features(pooled_events, feature_family)
    active, mean, scale, standardized = standardize_columns(raw, FEATURE_SCALE_FLOOR)
    if int(active.sum()) < RANK:
        raise ValueError(f"{name}: too few active source features")

    if basis_mode == "pca":
        projection, energy = fit_pca_projection(standardized, RANK)
    elif basis_mode == "source_encoding":
        projection, energy = fit_source_encoding_projection(standardized, sessions, source_names, RANK)
    else:
        raise ValueError(f"unknown basis mode {basis_mode!r}")

    projection = canonicalize_projection(projection)
    latent = standardized @ projection
    latent_scale = np.maximum(latent.std(axis=0, ddof=0), LATENT_SCALE_FLOOR)
    return LatentBasis(
        name=name,
        basis_mode=basis_mode,
        feature_family=feature_family,
        outer_date=outer_date,
        source_sessions=source_names,
        active_mask=np.asarray(active, dtype=bool),
        feature_mean=np.asarray(mean, dtype=np.float64),
        feature_scale=np.asarray(scale, dtype=np.float64),
        projection=np.asarray(projection, dtype=np.float64),
        latent_scale=np.asarray(latent_scale, dtype=np.float64),
        energy_ratio=np.asarray(energy, dtype=np.float64),
        retained_energy_at_rank=float(energy[:RANK].sum()),
    )


def transform_events(basis: LatentBasis, events: Sequence[ContextEvent], *, tags: Sequence[str] | None = None) -> np.ndarray:
    raw = build_raw_features(events, basis.feature_family, tags=tags)
    standardized = (raw[:, basis.active_mask] - basis.feature_mean[None, :]) / basis.feature_scale[None, :]
    latent = standardized @ basis.projection
    return latent / basis.latent_scale[None, :]


def solve_target_ridge(latent: np.ndarray, response: np.ndarray) -> np.ndarray:
    z = np.asarray(latent, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    n = z.shape[0]
    design = np.column_stack((np.ones(n, dtype=np.float64), z))
    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0]) * (n * TARGET_RIDGE_LAMBDA)
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return np.column_stack((coefficient[1:].T, coefficient[0]))


def predict_carrier(carrier: np.ndarray, latent: np.ndarray) -> np.ndarray:
    weights = carrier[:, :RANK]
    bias = carrier[:, RANK]
    return latent @ weights.T + bias[None, :]


def r2_by_channel(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    total = np.sum(np.square(observed - observed.mean(axis=0, keepdims=True)), axis=0)
    residual = np.sum(np.square(observed - predicted), axis=0)
    output = np.full(observed.shape[1], np.nan, dtype=np.float64)
    defined = total > NORM_FLOOR
    output[defined] = 1.0 - residual[defined] / total[defined]
    return output


def paired_summary(rows: Sequence[tuple[str, float]]) -> dict[str, Any]:
    values = np.asarray([float(value) for _name, value in rows], dtype=np.float64)
    remove = int(np.argmax(np.abs(values)))
    kept = np.delete(values, remove)
    return {
        "defined_sessions": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positive": int(np.sum(values > 0)),
        "zero": int(np.sum(values == 0)),
        "negative": int(np.sum(values < 0)),
        "leave_largest_absolute_out_mean": float(np.mean(kept)) if kept.size else None,
        "removed_session": rows[remove][0],
    }


def evaluate_session(
    session: ContextSession,
    basis: LatentBasis,
    *,
    budget: int,
) -> dict[str, float]:
    support = tuple(event for event in session.events if event.base.trial_index < budget)
    later = tuple(event for event in session.events if event.base.trial_index >= budget)
    if len(support) < RANK + 1 or len(later) < 4:
        raise ValueError(f"{session.base.session_name}: insufficient support/query events")

    z_support = transform_events(basis, support)
    z_later = transform_events(basis, later)
    y_support = np.vstack([event.base.log_rates for event in support]).astype(np.float64)
    y_later = np.vstack([event.base.log_rates for event in later]).astype(np.float64)

    carrier = solve_target_ridge(z_support, y_support)
    r_correct = r2_by_channel(y_later, predict_carrier(carrier, z_later))

    order, _manifest = v1.within_trial_label_shuffle(
        tuple(event.base for event in support),
        session=session.base.session_name,
        budget=budget,
    )
    label_carrier = solve_target_ridge(z_support[order], y_support)
    r_label = r2_by_channel(y_later, predict_carrier(label_carrier, z_later))

    support_mean = y_support.mean(axis=0)
    r_intercept = r2_by_channel(y_later, np.broadcast_to(support_mean[None, :], y_later.shape))

    defined = np.isfinite(r_correct) & np.isfinite(r_label) & np.isfinite(r_intercept)
    if not np.any(defined):
        raise ValueError(f"{session.base.session_name}: no defined forward channels")

    return {
        "median_r2_correct": float(np.median(r_correct[defined])),
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])),
    }


def compare_sealed_receipt(
    sealed: Mapping[str, Any],
    computed: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for budget in BUDGETS:
        budget_key = f"M{budget}"
        for arm in REFERENCE_ARMS:
            for session_name in v1.H1_HELDIN_SESSIONS:
                sealed_row = sealed["budgets"][budget_key]["candidates"][arm]["sessions"][session_name]
                ours = computed[budget_key][session_name][arm]
                for field in ("median_r2_correct", "median_delta_intercept"):
                    sealed_value = float(sealed_row[field])
                    independent_value = float(ours[field])
                    diff = abs(independent_value - sealed_value)
                    max_abs_diff = max(max_abs_diff, diff)
                    comparisons.append({
                        "arm": arm,
                        "budget": budget,
                        "field": field,
                        "session": session_name,
                        "independent": independent_value,
                        "sealed": sealed_value,
                        "abs_diff": diff,
                    })
    return {
        "comparisons": comparisons,
        "max_abs_diff": max_abs_diff,
        "reference_path": sealed.get("_path"),
        "reference_sha256": sealed.get("_sha256"),
    }


def run_computation(data_root: Path, sealed_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    sessions = load_context_sessions(data_root)

    bases: dict[str, dict[str, LatentBasis]] = {name: {} for name, _mode, _family in CANDIDATE_SPECS}
    retained_energy: dict[str, dict[str, float]] = {name: {} for name, _mode, _family in CANDIDATE_SPECS}
    for candidate_name, basis_mode, feature_family in CANDIDATE_SPECS:
        for outer_date in v1.H1_DATES:
            basis = fit_latent_basis(
                sessions,
                name=candidate_name,
                basis_mode=basis_mode,
                feature_family=feature_family,
                outer_date=outer_date,
            )
            bases[candidate_name][outer_date] = basis
            retained_energy[candidate_name][outer_date] = basis.retained_energy_at_rank

    per_session: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        f"M{budget}": {session: {} for session in v1.H1_HELDIN_SESSIONS}
        for budget in BUDGETS
    }
    for budget in BUDGETS:
        budget_key = f"M{budget}"
        for session_name in v1.H1_HELDIN_SESSIONS:
            outer_date = v1.session_date(session_name)
            session_rows: dict[str, dict[str, float]] = {}
            for candidate_name, _basis_mode, _feature_family in CANDIDATE_SPECS:
                session_rows[candidate_name] = evaluate_session(
                    sessions[session_name],
                    bases[candidate_name][outer_date],
                    budget=budget,
                )
            per_session[budget_key][session_name] = session_rows

    aggregates: dict[str, dict[str, Any]] = {}
    for budget in BUDGETS:
        budget_key = f"M{budget}"
        baseline_name = "pca_delta_q4"
        aggregates[budget_key] = {}
        for candidate_name, _basis_mode, _feature_family in CANDIDATE_SPECS:
            contrasts = [
                (
                    session_name,
                    per_session[budget_key][session_name][candidate_name]["median_r2_correct"]
                    - per_session[budget_key][session_name][baseline_name]["median_r2_correct"],
                )
                for session_name in v1.H1_HELDIN_SESSIONS
            ]
            aggregates[budget_key][candidate_name] = paired_summary(contrasts)

    sealed_body = json.loads(sealed_path.read_text(encoding="utf-8"))
    sealed_body["_path"] = str(sealed_path.resolve())
    sealed_body["_sha256"] = sha256_file(sealed_path)
    sealed_comparison = compare_sealed_receipt(sealed_body, per_session)

    runtime_seconds = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "per_session": per_session,
        "aggregates": aggregates,
        "retained_energy_at_rank": retained_energy,
        "sealed_comparison": sealed_comparison,
        "source_binding": {
            "sessions": list(v1.H1_HELDIN_SESSIONS),
            "dates": list(v1.H1_DATES),
            "files": [
                {
                    "session": session_name,
                    "path": str(sessions[session_name].base.path),
                    "sha256": sessions[session_name].base.input_sha256,
                }
                for session_name in v1.H1_HELDIN_SESSIONS
            ],
        },
        "implementation_binding": {
            "script_path": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
        },
        "runtime_seconds": runtime_seconds,
        "scope": {
            "cuda_used": False,
            "decoder_constructed": False,
            "trainer_constructed": False,
            "dense_velocity_series_opened": False,
            "dense_velocity_series_opened_by_carrier_screen": False,
            "public_held_in_calibration_nwbs_opened": 13,
            "held_out_nwbs_opened": 0,
            "minival_nwbs_opened": 0,
            "formal_test_labels_opened": 0,
            "target_session_optimizer_steps": 0,
            "target_session_backward_steps": 0,
        },
    }


def write_receipt(output: Path, body: Mapping[str, Any]) -> str:
    output = output.resolve()
    if output.exists():
        raise ReceiptExistsError(f"refusing to overwrite existing receipt: {output}")
    encoded = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    output.chmod(0o444)
    if stat.S_IMODE(output.stat().st_mode) != 0o444:
        raise OSError(f"failed to set receipt mode 0444: {output}")
    return hashlib.sha256(encoded).hexdigest()


def print_summary(body: Mapping[str, Any]) -> None:
    print("Contrast vs pca_delta_q4 (median_r2_correct):")
    print(f"{'Candidate':<24} {'Budget':<6} {'Mean':>12} {'Median':>12} {'Positive':>8}")
    for budget in BUDGETS:
        budget_key = f"M{budget}"
        for candidate_name, _basis_mode, _feature_family in CANDIDATE_SPECS:
            row = body["aggregates"][budget_key][candidate_name]
            print(
                f"{candidate_name:<24} {budget_key:<6} "
                f"{row['mean']:12.6f} {row['median']:12.6f} {row['positive']:8d}"
            )

    print("\nRetained energy at rank 4 (mean over outer dates):")
    for candidate_name, _basis_mode, _feature_family in CANDIDATE_SPECS:
        values = list(body["retained_energy_at_rank"][candidate_name].values())
        print(f"  {candidate_name}: mean={float(np.mean(values)):.6f}")

    sealed = body["sealed_comparison"]
    print(
        f"\nSealed receipt max |diff| (pca_delta_q4, ser_context_q4): "
        f"{sealed['max_abs_diff']:.6e}"
    )
    print(f"Sealed receipt: {sealed['reference_path']}")
    print(f"Sealed receipt sha256: {sealed['reference_sha256']}")
    print(f"Runtime: {body['runtime_seconds']:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("SPINT-main/data/000954"))
    parser.add_argument(
        "--sealed-receipt",
        type=Path,
        default=Path("sua_exploration/results/h1_event_carrier_design_screen_v1/source_screen.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sua_exploration/results/h1_tagfree_position_context/independent_crosscheck.json"),
    )
    args = parser.parse_args()

    sealed_path = args.sealed_receipt.resolve()
    receipt_sha = sha256_file(sealed_path)
    if receipt_sha != SEALED_RECEIPT_SHA256:
        print(f"warning: sealed receipt sha256 {receipt_sha} != expected {SEALED_RECEIPT_SHA256}", file=sys.stderr)

    body = run_computation(args.data_root.resolve(), sealed_path)
    receipt_digest = write_receipt(args.output, body)
    print_summary(body)
    print(f"\nReceipt: {args.output.resolve()}")
    print(f"Receipt sha256: {receipt_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
