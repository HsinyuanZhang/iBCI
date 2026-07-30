#!/usr/bin/env python3
"""Train-only descriptive audit for the cached T4 confidence coordinates.

This script never opens an NWB file.  It resolves only the sessions named in
the requested manifest split and summarizes the cached raw ``t4c`` vectors:

    [a, c, m, b, log residual variance, 0.5 log condition(C_ac)]

The audit is an input-eligibility check, not evidence that confidence improves
decoding.  In particular, the ``not_near_duplicate_of_t4`` flag only rejects a
simple linear-duplication failure mode before the controlled FiLM experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "t4_a",
    "t4_c",
    "t4_m",
    "t4_b",
    "log_residual_variance",
    "log_ac_covariance_shape",
)
T4_INDICES = tuple(range(4))
CONFIDENCE_INDICES = (4, 5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 3:
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return _finite_float(float(np.corrcoef(x, y)[0, 1]))


def _summary(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("summary requires a non-empty finite vector")
    quantiles = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "min": float(np.min(values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _load_cached_t4c(
    *,
    cache_dir: Path,
    session: str,
    pool_size: int,
    feature_version: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = sorted((cache_dir / "side_features").glob(f"{session}_*.npz"))
    matches: list[tuple[Path, np.ndarray, dict[str, Any]]] = []
    for path in candidates:
        with np.load(path, allow_pickle=False) as cache:
            if (
                str(cache["feature_group"].item()) != "t4c"
                or int(cache["pool_size"].item()) != pool_size
                or int(cache["feature_version"].item()) != feature_version
            ):
                continue
            features = cache["features"].astype(np.float64, copy=True)
            metadata = {
                "path": str(path.resolve()),
                "cache_key": str(cache["cache_key"].item()),
                "feature_version": int(cache["feature_version"].item()),
                "pool_size": int(cache["pool_size"].item()),
                "unit_count": int(features.shape[0]),
            }
            matches.append((path, features, metadata))
    if len(matches) != 1:
        raise RuntimeError(
            f"{session}: expected exactly one cached t4c@{pool_size} "
            f"version {feature_version} entry, "
            f"found {len(matches)}"
        )
    _, features, metadata = matches[0]
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"{session}: expected [units,6], got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"{session}: cached t4c contains non-finite values")
    return features, metadata


def _pairwise_correlations(matrix: np.ndarray) -> dict[str, float | None]:
    return {
        f"{FEATURE_NAMES[i]}__{FEATURE_NAMES[j]}": _safe_corr(
            matrix[:, i], matrix[:, j]
        )
        for i in CONFIDENCE_INDICES
        for j in T4_INDICES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--feature-version", type=int, default=2)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    sessions = list(manifest["session_splits"][args.split])
    if not sessions or len(sessions) != len(set(sessions)):
        raise ValueError(f"invalid or duplicate sessions in manifest split {args.split!r}")

    session_features: dict[str, np.ndarray] = {}
    cache_receipts: dict[str, dict[str, Any]] = {}
    for session in sessions:
        features, receipt = _load_cached_t4c(
            cache_dir=cache_dir,
            session=session,
            pool_size=args.pool_size,
            feature_version=args.feature_version,
        )
        session_features[session] = features
        cache_receipts[session] = receipt

    pooled = np.concatenate([session_features[name] for name in sessions], axis=0)
    centered = np.concatenate(
        [
            session_features[name] - np.mean(session_features[name], axis=0, keepdims=True)
            for name in sessions
        ],
        axis=0,
    )
    session_means = np.stack(
        [np.mean(session_features[name], axis=0) for name in sessions], axis=0
    )

    per_feature: dict[str, Any] = {}
    for index, name in enumerate(FEATURE_NAMES):
        within_stds = np.asarray(
            [np.std(session_features[session][:, index]) for session in sessions],
            dtype=np.float64,
        )
        per_feature[name] = {
            "pooled": _summary(pooled[:, index]),
            "within_session_std": _summary(within_stds),
            "nonconstant_session_count_at_1e-6": int(np.sum(within_stds > 1e-6)),
            "session_mean": _summary(session_means[:, index]),
        }

    raw_correlations = _pairwise_correlations(pooled)
    within_session_centered_correlations = _pairwise_correlations(centered)
    between_session_mean_correlations = _pairwise_correlations(session_means)
    within_values = [
        abs(value)
        for value in within_session_centered_correlations.values()
        if value is not None
    ]
    raw_values = [
        abs(value) for value in raw_correlations.values() if value is not None
    ]
    residual_nonconstant_all_sessions = (
        per_feature[FEATURE_NAMES[4]]["nonconstant_session_count_at_1e-6"]
        == len(sessions)
    )
    geometry_session_std = float(np.std(session_means[:, 5]))
    geometry_varies_across_sessions = geometry_session_std > 1e-6
    max_abs_within_corr = max(within_values) if within_values else None
    max_abs_raw_corr = max(raw_values) if raw_values else None
    not_near_duplicate = (
        max_abs_within_corr is not None
        and max_abs_within_corr < args.near_duplicate_threshold
    )

    standardized = (pooled - np.mean(pooled, axis=0, keepdims=True)) / np.maximum(
        np.std(pooled, axis=0, keepdims=True), 1e-12
    )
    singular_values = np.linalg.svd(standardized, full_matrices=False, compute_uv=False)

    result = {
        "schema_version": 1,
        "purpose": "train_only_t4c_descriptor_eligibility_audit",
        "interpretation_boundary": (
            "Descriptive input audit only; decoding value requires the matched "
            "confidence-FiLM controls."
        ),
        "protocol": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "split": args.split,
            "pool_size": args.pool_size,
            "feature_group": "t4c",
            "feature_version": args.feature_version,
            "feature_names": list(FEATURE_NAMES),
            "raw_nwb_opened": False,
            "test_session_opened": False,
            "near_duplicate_threshold": args.near_duplicate_threshold,
        },
        "sample": {
            "session_count": len(sessions),
            "unit_count": int(pooled.shape[0]),
            "sessions": sessions,
        },
        "per_feature": per_feature,
        "correlations": {
            "pooled_raw_confidence_vs_t4": raw_correlations,
            "within_session_centered_confidence_vs_t4": (
                within_session_centered_correlations
            ),
            "between_session_mean_confidence_vs_t4": (
                between_session_mean_correlations
            ),
            "max_abs_pooled_raw_confidence_vs_t4": max_abs_raw_corr,
            "max_abs_within_session_centered_confidence_vs_t4": max_abs_within_corr,
            "confidence_pair_pooled_raw": _safe_corr(pooled[:, 4], pooled[:, 5]),
            "confidence_pair_within_session_centered": _safe_corr(
                centered[:, 4], centered[:, 5]
            ),
        },
        "standardized_matrix": {
            "singular_values": [float(value) for value in singular_values],
            "numerical_rank_at_1e-8": int(np.linalg.matrix_rank(standardized, tol=1e-8)),
        },
        "eligibility": {
            "all_values_finite": True,
            "unit_residual_nonconstant_in_every_session": (
                residual_nonconstant_all_sessions
            ),
            "direction_geometry_varies_across_sessions": (
                geometry_varies_across_sessions
            ),
            "direction_geometry_session_mean_std": geometry_session_std,
            "not_near_duplicate_of_t4_within_sessions": not_near_duplicate,
            "eligible_for_controlled_film_screen": bool(
                residual_nonconstant_all_sessions
                and geometry_varies_across_sessions
                and not_near_duplicate
            ),
            "does_not_establish_performance_gain": True,
        },
        "cache_receipts": cache_receipts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["sample"], indent=2))
    print(json.dumps(result["correlations"], indent=2, sort_keys=True))
    print(json.dumps(result["eligibility"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
