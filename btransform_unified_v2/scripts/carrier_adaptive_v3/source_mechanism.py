#!/usr/bin/env python3
"""Retrospective source-covariance mechanism diagnostic for saved projections.

The diagnostic loads only four hash-bound source projection receipts and their
NPZ arrays.  It does not fit a projection, inspect NWB/model data, score a
target, or select a method.  It is intentionally retrospective: its role is
to describe the source geometry after inner results have been seen, not to
claim preregistered validation or downstream decoder causality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from statistics import AdaptiveProjection, load_projection

HERE = Path(__file__).resolve().parent
STATISTICS = HERE / "statistics.py"
DATASETS = ("m1", "h1")
METHODS = ("pca_wiener", "snr_wiener")


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_scalar(value: Any, context: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"nonfinite scalar: {context}")
    return result


def finite_array(value: np.ndarray, context: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.ndim in (1, 2) and np.isfinite(array).all(), f"nonfinite/invalid array: {context}")
    return array


def symmetric_covariance(value: np.ndarray, name: str) -> np.ndarray:
    array = finite_array(value, name)
    require(array.ndim == 2 and array.shape[0] == array.shape[1], f"square covariance required: {name}")
    require(np.array_equal(array, array.T), f"covariance must be exactly symmetric: {name}")
    return array


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else finite_scalar(numerator / denominator, "ratio")


def trace(value: np.ndarray) -> float:
    return finite_scalar(np.trace(value), "trace")


def participation_rank(value: np.ndarray) -> float | None:
    numerator = trace(value)
    denominator = trace(value @ value)
    return ratio(numerator * numerator, denominator)


def covariance_summary(signal: np.ndarray, noise: np.ndarray, noise_regularized: np.ndarray) -> dict[str, Any]:
    trace_signal, trace_noise = trace(signal), trace(noise)
    signal_eigenvalues = np.linalg.eigvalsh(signal)[::-1]
    noise_eigenvalues = np.linalg.eigvalsh(noise)[::-1]
    regularized_eigenvalues = np.linalg.eigvalsh(noise_regularized)[::-1]
    require(np.isfinite(signal_eigenvalues).all() and np.isfinite(noise_eigenvalues).all() and np.isfinite(regularized_eigenvalues).all(), "nonfinite covariance eigenspectrum")
    minimum = finite_scalar(regularized_eigenvalues[-1], "Nreg smallest eigenvalue")
    require(minimum > 0.0, "noise_regularized must be positive definite")
    return {
        "trace_signal": trace_signal,
        "trace_noise": trace_noise,
        "trace_signal_over_trace_noise": ratio(trace_signal, trace_noise),
        "signal_eigenvalues_descending": [finite_scalar(x, "signal eigenvalue") for x in signal_eigenvalues],
        "noise_eigenvalues_descending": [finite_scalar(x, "noise eigenvalue") for x in noise_eigenvalues],
        "signal_participation_rank": participation_rank(signal),
        "noise_participation_rank": participation_rank(noise),
        "noise_regularized_condition_number": finite_scalar(regularized_eigenvalues[0] / minimum, "Nreg condition number"),
    }


def transform_span(transform: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the nonzero column span using the requested SVD threshold."""
    u, singular_values, _ = np.linalg.svd(transform, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = float(np.finfo(np.float64).eps * max(transform.shape) * largest)
    rank = int(np.count_nonzero(singular_values > threshold)) if largest > 0.0 else 0
    return u[:, :rank], {
        "rank": rank,
        "svd_threshold": threshold,
        "singular_values_descending": [finite_scalar(value, "transform singular value") for value in singular_values],
        "rank_convention": "rank is zero when the largest singular value is zero; otherwise singular values strictly greater than eps*max(shape)*largest are retained",
    }


def span_summary(transform: np.ndarray, signal: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    q, summary = transform_span(transform)
    signal_trace, noise_trace = trace(signal), trace(noise)
    if q.shape[1] == 0:
        summary.update({
            "signal_energy_retained": None,
            "noise_energy_retained": None,
            "subspace_signal_trace": 0.0,
            "subspace_noise_trace": 0.0,
            "subspace_signal_over_noise": None,
            "zero_rank_convention": "all span-dependent ratios are null for a zero-rank transform",
        })
        return q, summary
    projected_signal = trace(q.T @ signal @ q)
    projected_noise = trace(q.T @ noise @ q)
    summary.update({
        "signal_energy_retained": ratio(projected_signal, signal_trace),
        "noise_energy_retained": ratio(projected_noise, noise_trace),
        "subspace_signal_trace": projected_signal,
        "subspace_noise_trace": projected_noise,
        "subspace_signal_over_noise": ratio(projected_signal, projected_noise),
    })
    return q, summary


def axis_summary(transform: np.ndarray, signal: np.ndarray, noise: np.ndarray) -> list[dict[str, Any]]:
    signal_variances = np.diag(transform.T @ signal @ transform)
    noise_variances = np.diag(transform.T @ noise @ transform)
    rows: list[dict[str, Any]] = []
    for index, column in enumerate(transform.T):
        norm = finite_scalar(np.linalg.norm(column), f"transform column norm {index}")
        signal_variance = finite_scalar(signal_variances[index], f"axis signal variance {index}")
        noise_variance = finite_scalar(noise_variances[index], f"axis noise variance {index}")
        if norm == 0.0:
            normalized_signal = None
            normalized_noise = None
        else:
            unit = column / norm
            normalized_signal = finite_scalar(unit @ signal @ unit, f"unit axis signal variance {index}")
            normalized_noise = finite_scalar(unit @ noise @ unit, f"unit axis noise variance {index}")
        rows.append({
            "axis": index,
            "transform_column_l2_norm": norm,
            "transform_signal_variance": signal_variance,
            "transform_noise_variance": noise_variance,
            "transform_axis_snr": ratio(signal_variance, noise_variance),
            "unit_direction_signal_variance": normalized_signal,
            "unit_direction_noise_variance": normalized_noise,
            "unit_direction_snr": None if normalized_signal is None or normalized_noise is None else ratio(normalized_signal, normalized_noise),
        })
    return rows


def principal_angles(q_left: np.ndarray, q_right: np.ndarray) -> dict[str, Any]:
    if q_left.shape[1] == 0 or q_right.shape[1] == 0:
        return {
            "cosines_descending": None,
            "degrees_ascending": None,
            "zero_rank_convention": "principal angles are null when either method has zero nonzero-column span",
        }
    cosines = np.linalg.svd(q_left.T @ q_right, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    degrees = np.degrees(np.arccos(cosines))
    return {
        "cosines_descending": [finite_scalar(value, "principal-angle cosine") for value in cosines],
        "degrees_ascending": [finite_scalar(value, "principal-angle degrees") for value in degrees],
    }


def metadata_summary(metadata: Any, method: str, dataset: str) -> dict[str, Any]:
    require(isinstance(metadata, dict), f"metadata object required: {dataset}/{method}")
    require(metadata.get("schema") == "carrier_adaptive_v3_projection_v1", f"projection schema drift: {dataset}/{method}")
    require(metadata.get("method") == method, f"projection method drift: {dataset}/{method}")
    sessions = metadata.get("sessions")
    require(isinstance(sessions, list) and sessions and all(isinstance(value, str) for value in sessions), f"projection session roster missing: {dataset}/{method}")
    keys = ("rho", "reliability", "basis", "eigenvalues", "effective_rank", "zeroaxes", "equal_weighting", "signal_covariance_interpretation")
    require(all(key in metadata for key in keys), f"projection metadata incomplete: {dataset}/{method}")
    optional = ("common_scale", "projected_signal", "projected_total", "snr_weights")
    return {key: metadata[key] for key in keys} | {"sessions": sessions} | {key: metadata[key] for key in optional if key in metadata}


def receipt_binding(receipt_path: Path, plan: AdaptiveProjection) -> dict[str, str]:
    receipt = json.loads(receipt_path.read_text())
    require(isinstance(receipt, dict) and isinstance(receipt.get("arrays"), str), f"array path absent from receipt: {receipt_path}")
    arrays = Path(receipt["arrays"])
    require(arrays.is_file(), f"projection arrays missing: {arrays}")
    # load_projection has already checked the receipt's arrays and individual-array hashes.
    return {
        "json_path": str(receipt_path.resolve()),
        "json_sha256": sha256(receipt_path),
        "npz_path": str(arrays.resolve()),
        "npz_sha256": sha256(arrays),
    }


def dataset_block(root: Path, dataset: str) -> dict[str, Any]:
    receipts = {method: root / "projections" / "inner" / dataset / f"{method}.json" for method in METHODS}
    plans = {method: load_projection(path) for method, path in receipts.items()}
    metadata = {method: metadata_summary(plans[method].metadata, method, dataset) for method in METHODS}
    pca, snr = plans["pca_wiener"], plans["snr_wiener"]
    for name in ("mean", "signal_cov", "noise_cov", "noise_regularized"):
        left, right = getattr(pca, name), getattr(snr, name)
        require(np.array_equal(left, right), f"shared source array mismatch: {dataset}:{name}")
    require(metadata["pca_wiener"]["sessions"] == metadata["snr_wiener"]["sessions"], f"shared source session roster mismatch: {dataset}")
    signal = symmetric_covariance(pca.signal_cov, f"{dataset}:signal_cov")
    noise = symmetric_covariance(pca.noise_cov, f"{dataset}:noise_cov")
    regularized = symmetric_covariance(pca.noise_regularized, f"{dataset}:noise_regularized")
    mean = finite_array(pca.mean, f"{dataset}:mean")
    require(mean.ndim == 1 and mean.size == signal.shape[0], f"mean/covariance geometry drift: {dataset}")
    transforms: dict[str, np.ndarray] = {}
    spans: dict[str, np.ndarray] = {}
    methods: dict[str, Any] = {}
    for method, plan in plans.items():
        transform = finite_array(plan.transform, f"{dataset}:{method}:transform")
        require(transform.shape == (mean.size, 4), f"transform geometry drift: {dataset}:{method}")
        transforms[method] = transform
        q, span = span_summary(transform, signal, noise)
        spans[method] = q
        methods[method] = {"metadata": metadata[method], "column_span": span, "axes": axis_summary(transform, signal, noise), "input": receipt_binding(receipts[method], plan)}
    return {
        "dataset": dataset,
        "source_sessions": metadata["pca_wiener"]["sessions"],
        "covariance": covariance_summary(signal, noise, regularized),
        "methods": methods,
        "principal_angles_pca_wiener_vs_snr_wiener": principal_angles(spans["pca_wiener"], spans["snr_wiener"]),
    }


def require_destination(root: Path, destination: Path) -> Path:
    resolved = destination.resolve()
    require(root in (resolved, *resolved.parents), f"--dest must be inside --root: {resolved}")
    require(resolved != root and resolved.suffix == ".json", "--dest must be a new JSON file below --root")
    require(not resolved.exists(), f"--dest already exists: {resolved}")
    require(resolved.parent.is_dir(), f"--dest parent must already exist: {resolved.parent}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="results/carrier_adaptive_v3")
    parser.add_argument("--dest", type=Path, required=True, help="new JSON path below --root")
    args = parser.parse_args()
    root = args.root.resolve()
    require(root.is_dir(), f"--root must be an existing directory: {root}")
    destination = require_destination(root, args.dest)
    require(STATISTICS.is_file(), f"missing frozen statistics implementation: {STATISTICS}")
    payload = {
        "schema": "carrier_adaptive_v3_source_mechanism_v1",
        "status": "PASSED",
        "retrospective_source_mechanism_diagnostic": True,
        "not_used_for_method_selection": True,
        "purpose": "post-inner-result source-geometry mechanism inspection; not preregistered validation",
        "interpretation_limits": [
            "Covariances use raw-feature Euclidean geometry.",
            "Signal and noise are empirical repeated-panel approximations, with the limitations recorded in the projection metadata.",
            "Transform columns can differ in scale; column-span quantities isolate direction from that scaling, and per-axis unit-direction quantities are reported separately.",
            "A larger PCA source signal-energy-retention fraction does not by itself establish more downstream task information.",
            "This statistical source diagnostic does not prove decoder causality.",
        ],
        "statistics_py": {"path": str(STATISTICS.resolve()), "sha256": sha256(STATISTICS)},
        "datasets": [dataset_block(root, dataset) for dataset in DATASETS],
    }
    json.dumps(payload, sort_keys=True, allow_nan=False)
    with destination.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
