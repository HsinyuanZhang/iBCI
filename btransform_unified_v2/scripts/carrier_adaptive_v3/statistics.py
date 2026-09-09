"""Repeated-panel covariance estimates and fixed four-axis adaptive projections.

The inputs to :func:`fit_projection` are source-only panel arrays.  Session and
unit contributions are deliberately averaged with equal weights, regardless of
their number of panels or samples.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


_RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
_OUTPUT_DIM = 4


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sym(value: np.ndarray) -> np.ndarray:
    return (value + value.T) * 0.5


def _psd(value: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_sym(value))
    return _sym((vectors * np.maximum(values, 0.0)) @ vectors.T)


def _floor(covariance: np.ndarray) -> float:
    dimension = covariance.shape[0]
    return max(1e-6 * float(np.trace(covariance)) / dimension, 1e-12)


def _regularize(noise: np.ndarray, rho: float) -> np.ndarray:
    """Shrink to spherical noise, then apply the specified eigenvalue floor."""
    dimension = noise.shape[0]
    target = np.eye(dimension) * (float(np.trace(noise)) / dimension)
    shrunken = _sym((1.0 - rho) * noise + rho * target)
    values, vectors = np.linalg.eigh(shrunken)
    return _sym((vectors * np.maximum(values, _floor(noise))) @ vectors.T)


def _signed_eigh(covariance: np.ndarray, descending: bool = True) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(_sym(covariance))
    order = np.argsort(values, kind="mergesort")
    if descending:
        order = order[::-1]
    values, vectors = values[order], vectors[:, order]
    # Eigenvectors have an arbitrary sign.  Fix it from the first largest
    # magnitude component; argmax's first-index rule is deterministic.
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0.0:
            vectors[:, column] *= -1.0
    return values, vectors


def _as_panels(panels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    _need(isinstance(panels, Mapping) and bool(panels), "panels must be a nonempty session mapping")
    result: dict[str, np.ndarray] = {}
    dimension: int | None = None
    for session, raw in panels.items():
        name = str(session)
        _need(name not in result, "duplicate session name after string conversion")
        value = np.asarray(raw, dtype=np.float64)
        _need(value.ndim == 3, f"{name}: panels must have shape [P,U,K]")
        p_count, units, width = value.shape
        _need(p_count >= 2 and units >= 1 and width >= 1, f"{name}: require P>=2, U>=1, K>=1")
        _need(np.isfinite(value).all(), f"{name}: panels contain nonfinite values")
        if dimension is None:
            dimension = width
        _need(width == dimension, "all sessions must share K")
        result[name] = np.ascontiguousarray(value)
    return result


def _session_statistics(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return equal-unit mean, repeated-panel noise, and its signal correction."""
    panel_count = value.shape[0]
    unit_means = value.mean(axis=0)                         # [U,K]
    centered = value - unit_means[None, :, :]
    unit_noise = np.einsum("puk,pul->ukl", centered, centered) / (panel_count - 1)
    noise = _sym(unit_noise.mean(axis=0))                   # equal unit weighting
    return unit_means, noise, unit_means.mean(axis=0), panel_count


def _fit_noise(noise_by_session: Mapping[str, np.ndarray]) -> tuple[float, dict[str, float | None], np.ndarray]:
    """Choose one source-only shrinkage coefficient by leave-session-out risk."""
    names = tuple(noise_by_session)
    dimension = next(iter(noise_by_session.values())).shape[0]
    scores: dict[str, float | None] = {}
    if len(names) == 1:
        # With no leave-session-out split, use the fully spherical prescribed
        # fallback and record that no LOO score exists.
        rho = 1.0
        scores = {str(candidate): None for candidate in _RHO_GRID}
    else:
        for rho in _RHO_GRID:
            losses = []
            for held in names:
                training = sum((noise_by_session[name] for name in names if name != held), np.zeros((dimension, dimension))) / (len(names) - 1)
                regularized = _regularize(training, rho)
                sign, logdet = np.linalg.slogdet(regularized)
                _need(sign > 0.0 and np.isfinite(logdet), "regularized noise must be positive definite")
                loss = float(logdet + np.trace(np.linalg.solve(regularized, noise_by_session[held])))
                losses.append(loss)
            scores[str(rho)] = float(np.mean(losses))
        rho = min(_RHO_GRID, key=lambda candidate: (scores[str(candidate)], candidate))
    pooled = sum(noise_by_session.values(), np.zeros((dimension, dimension))) / len(names)
    return rho, scores, _regularize(pooled, rho)


@dataclass(frozen=True)
class AdaptiveProjection:
    mean: np.ndarray
    transform: np.ndarray
    reconstruction: np.ndarray
    signal_cov: np.ndarray
    noise_cov: np.ndarray
    noise_regularized: np.ndarray
    metadata: Mapping[str, object]


def fit_projection(panels: Mapping[str, np.ndarray], method: str) -> AdaptiveProjection:
    """Fit a fixed four-output source projection from ``session -> [P,U,K]``."""
    _need(method in {"pca_wiener", "snr_wiener"}, "method must be pca_wiener or snr_wiener")
    source = _as_panels(panels)
    unit_means: dict[str, np.ndarray] = {}
    noise_by_session: dict[str, np.ndarray] = {}
    session_means: dict[str, np.ndarray] = {}
    panel_counts: dict[str, int] = {}
    for name, value in source.items():
        means, noise, session_mean, count = _session_statistics(value)
        unit_means[name], noise_by_session[name] = means, noise
        session_means[name], panel_counts[name] = session_mean, count

    # Each session gets one equal share, then each session's units do too.
    mean = sum((means.mean(axis=0) for means in unit_means.values()), np.zeros_like(next(iter(session_means.values())))) / len(source)
    cmean = sum((np.einsum("uk,ul->kl", means - mean, means - mean) / means.shape[0] for means in unit_means.values()), np.zeros((mean.size, mean.size))) / len(source)
    noise = sum(noise_by_session.values(), np.zeros_like(cmean)) / len(source)
    repeated_panel_noise_of_mean = sum((noise_by_session[name] / panel_counts[name] for name in source), np.zeros_like(cmean)) / len(source)
    signal = _psd(cmean - repeated_panel_noise_of_mean)
    rho, loo_scores, noise_regularized = _fit_noise(noise_by_session)
    dimension = mean.size
    transform = np.zeros((dimension, _OUTPUT_DIM), dtype=np.float64)
    reconstruction = np.zeros((dimension, dimension), dtype=np.float64)
    reliability = np.zeros(_OUTPUT_DIM, dtype=np.float64)
    eigenvalues = np.zeros(_OUTPUT_DIM, dtype=np.float64)
    active = min(dimension, _OUTPUT_DIM)

    if method == "pca_wiener":
        # PCA is explicitly based on the unshrunk repeated-panel noise.
        total = _psd(signal + noise)
        values, basis = _signed_eigh(total)
        chosen_values, chosen_basis = values[:active], basis[:, :active]
        projected_signal = np.einsum("ki,kl,li->i", chosen_basis, signal, chosen_basis)
        alpha = np.clip(projected_signal / np.maximum(chosen_values, _floor(total)), 0.0, 1.0)
        scale = float(np.sqrt(max(float(np.mean(chosen_values)), _floor(total)))) if active else 1.0
        transform[:, :active] = chosen_basis * alpha[None, :] / scale
        reconstruction = _sym((chosen_basis * alpha[None, :]) @ chosen_basis.T)
        reliability[:active], eigenvalues[:active] = alpha, chosen_values
        method_metadata = {"basis": "eigen(Csignal + raw N)", "common_scale": scale, "projected_signal": projected_signal.tolist(), "projected_total": chosen_values.tolist()}
    else:
        noise_values, noise_vectors = _signed_eigh(noise_regularized)
        inverse_sqrt = (noise_vectors * (1.0 / np.sqrt(noise_values))) @ noise_vectors.T
        whitened_signal = _psd(inverse_sqrt @ signal @ inverse_sqrt)
        values, vectors = _signed_eigh(whitened_signal)
        chosen_values, chosen_vectors = values[:active], vectors[:, :active]
        # Loading signs are part of the output contract.  Whitened-space signs
        # alone are insufficient because the returned loading is N^-1/2 V.
        final_loadings = inverse_sqrt @ chosen_vectors
        for column in range(active):
            pivot = int(np.argmax(np.abs(final_loadings[:, column])))
            if final_loadings[pivot, column] < 0.0:
                chosen_vectors[:, column] *= -1.0
        final_loadings = inverse_sqrt @ chosen_vectors
        weights = chosen_values / np.power(1.0 + chosen_values, 1.5)
        transform[:, :active] = final_loadings * weights[None, :]
        alpha = chosen_values / (1.0 + chosen_values)
        # This is the stated row-vector Wiener reconstruction.  It is not the
        # pseudoinverse of transform: that would undo the intended shrinkage.
        reconstruction = inverse_sqrt @ (chosen_vectors * alpha[None, :]) @ chosen_vectors.T @ (noise_vectors * np.sqrt(noise_values)) @ noise_vectors.T
        reliability[:active], eigenvalues[:active] = chosen_values / (1.0 + chosen_values), chosen_values
        method_metadata = {"basis": "eigen(Nreg^-1/2 Csignal Nreg^-1/2)", "snr_weights": weights.tolist()}

    metadata = {
        "schema": "carrier_adaptive_v3_projection_v1",
        "method": method,
        "rho": float(rho),
        "loo_scores": loo_scores,
        "eigenvalues": eigenvalues.tolist(),
        "reliability": reliability.tolist(),
        "effective_rank": int(np.count_nonzero(reliability > 1e-12)),
        "zeroaxes": [index for index, value in enumerate(reliability) if value <= 1e-12],
        "confidence_participation_ratio": float((reliability.sum() ** 2) / np.square(reliability).sum()) if np.any(reliability > 0.0) else 0.0,
        "sessions": list(source),
        "panel_counts": panel_counts,
        "equal_weighting": "equal session, equal unit; repeated-panel covariance uses sample denominator P-1",
        "signal_covariance_interpretation": "empirical PSD debiased approximation from repeated panels; correlated units, panel structure, drift, and coverage effects mean it is not asserted to be an unbiased biological covariance",
        **method_metadata,
    }
    return AdaptiveProjection(mean.copy(), transform, reconstruction, signal, noise, noise_regularized, metadata)


def project(plan: AdaptiveProjection, raw: np.ndarray) -> np.ndarray:
    """Center raw vectors and return their four projection axes as float32."""
    _need(isinstance(plan, AdaptiveProjection), "plan must be AdaptiveProjection")
    value = np.asarray(raw, dtype=np.float64)
    _need(value.ndim >= 1 and value.shape[-1] == plan.mean.size, "raw final dimension must match plan")
    _need(np.isfinite(value).all(), "raw contains nonfinite values")
    return np.asarray((value - plan.mean) @ plan.transform, dtype=np.float32)


def reconstruct(plan: AdaptiveProjection, raw: np.ndarray) -> np.ndarray:
    """Return the float64 rank-four Wiener prediction in original raw units."""
    _need(isinstance(plan, AdaptiveProjection), "plan must be AdaptiveProjection")
    value = np.asarray(raw, dtype=np.float64)
    _need(value.ndim >= 1 and value.shape[-1] == plan.mean.size, "raw final dimension must match plan")
    _need(np.isfinite(value).all(), "raw contains nonfinite values")
    return np.asarray(plan.mean + (value - plan.mean) @ plan.reconstruction, dtype=np.float64)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(item) for item in value]
    return value


def save_projection(plan: AdaptiveProjection, destination: Path) -> tuple[Path, Path]:
    """Write a new immutable NPZ and its JSON hash binding; never overwrite."""
    _need(isinstance(plan, AdaptiveProjection), "plan must be AdaptiveProjection")
    destination = Path(destination)
    receipt, arrays = destination.with_suffix(".json"), destination.with_suffix(".npz")
    if receipt.exists() or arrays.exists():
        raise FileExistsError("projection outputs must be new")
    payload = {"mean": plan.mean, "transform": plan.transform, "reconstruction": plan.reconstruction, "signal_cov": plan.signal_cov, "noise_cov": plan.noise_cov, "noise_regularized": plan.noise_regularized}
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, **payload)
    body = {**_jsonable(dict(plan.metadata)), "arrays": str(arrays.resolve()), "arrays_sha256": _sha_file(arrays), "array_sha256": {key: _array_sha(value) for key, value in payload.items()}, "implementation_sha256": _sha_file(Path(__file__).resolve())}
    receipt.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return receipt, arrays


def load_projection(receipt_path: Path) -> AdaptiveProjection:
    """Load only an NPZ whose receipt and individual array hashes agree."""
    receipt = Path(receipt_path)
    metadata = json.loads(receipt.read_text())
    arrays = Path(metadata["arrays"])
    _need(arrays.is_file() and metadata.get("arrays_sha256") == _sha_file(arrays), "projection array binding drift")
    with np.load(arrays, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in ("mean", "transform", "reconstruction", "signal_cov", "noise_cov", "noise_regularized")}
    for key, value in payload.items():
        _need(metadata.get("array_sha256", {}).get(key) == _array_sha(value), f"projection array binding drift: {key}")
    _need(payload["transform"].shape == (payload["mean"].size, _OUTPUT_DIM), "projection transform geometry")
    _need(payload["reconstruction"].shape == (payload["mean"].size, payload["mean"].size), "projection reconstruction geometry")
    return AdaptiveProjection(payload["mean"], payload["transform"], payload["reconstruction"], payload["signal_cov"], payload["noise_cov"], payload["noise_regularized"], metadata)
