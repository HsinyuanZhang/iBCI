"""Source-frozen EMG-Syn3 / PCA3 carriers, controls, and coverage."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import NMF, PCA

from . import plan


class Syn3Error(RuntimeError):
    """Fail closed for EMG-Syn3 constructibility."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Syn3Error(message)


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    _require(array.ndim >= 1 and np.isfinite(array).all(), "digest needs finite array")
    header = json.dumps(
        {"shape": list(array.shape), "dtype": str(array.dtype)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + array.tobytes()).hexdigest()


def _row_hash(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


@dataclass(frozen=True)
class SourceBasis:
    kind: str
    scale: np.ndarray
    dictionary: np.ndarray
    activations: np.ndarray
    order: tuple[int, ...]
    reconstruction_digest: str
    library: Mapping[str, object]
    extra: Mapping[str, object]


def positive_scale(emg: np.ndarray) -> np.ndarray:
    matrix = np.asarray(emg, dtype=np.float64)
    _require(matrix.ndim == 2 and matrix.size > 0 and np.isfinite(matrix).all(), "EMG scale input")
    rms = np.sqrt(np.mean(np.square(matrix), axis=0))
    return np.maximum(rms, plan.SCALE_FLOOR)


def apply_scale(emg: np.ndarray, scale: np.ndarray) -> np.ndarray:
    matrix = np.asarray(emg, dtype=np.float64)
    _require(matrix.ndim == 2 and scale.shape == (matrix.shape[1],), "scale broadcast")
    return matrix / scale


def _require_nonnegative(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    _require(array.ndim == 2 and np.isfinite(array).all(), "basis input must be finite 2-D")
    negative_fraction = float(np.mean(array < 0.0))
    _require(negative_fraction == 0.0, f"signed EMG rejects NNMF (negative_fraction={negative_fraction})")
    _require(float(np.min(array)) >= 0.0, "signed EMG rejects NNMF")
    return array


def fit_source_nmf(emg: np.ndarray) -> SourceBasis:
    raw = _require_nonnegative(emg)
    scale = positive_scale(raw)
    x = apply_scale(raw, scale)
    kwargs = dict(plan.NNMF_LAW["sklearn_kwargs"])
    model = NMF(**kwargs)
    activations = np.asarray(model.fit_transform(x), dtype=np.float64)
    dictionary = np.asarray(model.components_, dtype=np.float64)
    _require(int(getattr(model, "n_iter_", plan.NNMF_LAW["maximum_iterations"]))
             < int(plan.NNMF_LAW["maximum_iterations"]),
             "NNMF did not converge")
    _require(np.isfinite(activations).all() and np.isfinite(dictionary).all(), "NNMF nonfinite")
    _require(np.all(activations >= -1.0e-12) and np.all(dictionary >= -1.0e-12), "NNMF left the nonnegative cone")
    activations = np.maximum(activations, 0.0)
    dictionary = np.maximum(dictionary, 0.0)
    norms = np.linalg.norm(dictionary, axis=1)
    _require(np.all(norms > 0.0), "NNMF dictionary row vanished")
    dictionary = dictionary / norms[:, None]
    activations = activations * norms[None, :]
    energies = np.sum(np.square(activations), axis=0)
    order_keys = [(-float(energies[index]), _row_hash(dictionary[index]), index) for index in range(plan.RANK)]
    order = tuple(item[2] for item in sorted(order_keys))
    dictionary = dictionary[list(order)]
    activations = activations[:, list(order)]
    reconstruction = activations @ dictionary
    import sklearn
    return SourceBasis(
        kind="nnmf",
        scale=scale,
        dictionary=dictionary,
        activations=activations,
        order=order,
        reconstruction_digest=array_digest(reconstruction),
        library={"sklearn": sklearn.__version__, "estimator": "sklearn.decomposition.NMF"},
        extra={"n_iter": int(model.n_iter_), "reconstruction_err": float(model.reconstruction_err_)},
    )


def fit_source_pca(emg: np.ndarray) -> SourceBasis:
    raw = np.asarray(emg, dtype=np.float64)
    _require(raw.ndim == 2 and np.isfinite(raw).all(), "PCA input")
    scale = positive_scale(raw)
    x = apply_scale(raw, scale)
    mean = x.mean(axis=0)
    centered = x - mean
    pca = PCA(n_components=plan.RANK, svd_solver="full", random_state=plan.SEED)
    scores = np.asarray(pca.fit_transform(centered), dtype=np.float64)
    components = np.asarray(pca.components_, dtype=np.float64)
    anchors = np.argmax(np.abs(components), axis=1)
    for row, anchor in enumerate(anchors):
        if components[row, anchor] < 0.0:
            components[row] *= -1.0
            scores[:, row] *= -1.0
    reconstruction = (scores @ components) + mean
    return SourceBasis(
        kind="signed_pca",
        scale=scale,
        dictionary=components,
        activations=scores,
        order=tuple(range(plan.RANK)),
        reconstruction_digest=array_digest(reconstruction),
        library={"estimator": "sklearn.decomposition.PCA"},
        extra={
            "mean": mean,
            "singular_values": np.asarray(pca.singular_values_, dtype=np.float64),
            "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        },
    )


def nnls_activations(emg: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    matrix = np.asarray(emg, dtype=np.float64)
    h = np.asarray(dictionary, dtype=np.float64)
    _require(matrix.ndim == 2 and h.shape == (plan.RANK, matrix.shape[1]), "NNLS shapes")
    rows = np.zeros((matrix.shape[0], plan.RANK), dtype=np.float64)
    ht = h.T
    for index, row in enumerate(matrix):
        coef, _residual = nnls(ht, row)
        rows[index] = coef
    _require(np.isfinite(rows).all() and np.all(rows >= -1.0e-12), "NNLS produced nonfinite/negative")
    return np.maximum(rows, 0.0)


def project_basis(emg: np.ndarray, basis: SourceBasis) -> np.ndarray:
    scaled = apply_scale(emg, basis.scale)
    if basis.kind == "nnmf":
        return nnls_activations(scaled, basis.dictionary)
    mean = np.asarray(basis.extra["mean"], dtype=np.float64)
    return (scaled - mean) @ basis.dictionary.T


def fit_unit_ridge(scores: np.ndarray, rates: np.ndarray) -> tuple[np.ndarray, float]:
    z = np.asarray(scores, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64).reshape(-1)
    _require(z.ndim == 2 and z.shape[0] == r.shape[0] >= 1 and z.shape[1] == plan.RANK, "ridge shapes")
    n = float(z.shape[0])
    design = np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))
    gram = (design.T @ design) / n
    penalty = np.diag([0.0, plan.RIDGE_LAMBDA, plan.RIDGE_LAMBDA, plan.RIDGE_LAMBDA])
    rhs = (design.T @ r) / n
    beta = np.linalg.solve(gram + penalty, rhs)
    _require(np.isfinite(beta).all(), "ridge nonfinite")
    return beta[1:].copy(), float(beta[0])


def fit_all_units(scores: np.ndarray, rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(scores, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    _require(r.ndim == 2 and r.shape[0] == z.shape[0], "unit ridge rates")
    weights = np.zeros((r.shape[1], plan.RANK), dtype=np.float64)
    intercepts = np.zeros(r.shape[1], dtype=np.float64)
    for unit in range(r.shape[1]):
        weights[unit], intercepts[unit] = fit_unit_ridge(z, r[:, unit])
    return weights, intercepts


def carrier_from_encoding(weights: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    return np.column_stack((weights, intercepts))


def require_support_bins(trial_ids: np.ndarray, *, budget: int) -> None:
    ids = np.asarray(trial_ids)
    _require(ids.ndim == 1 and ids.size > 0, "trial ids")
    _require(int(np.min(ids)) >= 0 and int(np.max(ids)) < int(budget), "support boundary crossed")


def coverage_report(
    scores: np.ndarray, rates: np.ndarray, *, trial_ids: np.ndarray, budget: int,
) -> dict[str, object]:
    z = np.asarray(scores, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    ids = np.asarray(trial_ids)
    require_support_bins(ids, budget=budget)
    design = np.column_stack((np.ones(z.shape[0]), z))
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = float(max(design.shape) * np.finfo(np.float64).eps * singular[0])
    rank = int(np.sum(singular > tolerance))
    condition = float(np.inf) if rank < design.shape[1] else float(singular[0] / singular[-1])
    gram = (design.T @ design) / float(design.shape[0])
    eig = np.sort(np.linalg.eigvalsh(gram))
    occupancy = {str(trial): int(np.sum(ids == trial)) for trial in np.unique(ids)}
    return {
        "budget": int(budget),
        "n_trials": int(len(np.unique(ids))),
        "valid_bins": int(z.shape[0]),
        "valid_seconds": float(z.shape[0] * plan.BIN_SECONDS),
        "rejected_for_trial_count": False,
        "design_rank": rank,
        "design_condition": condition,
        "normalized_gram_eigenvalues": eig.tolist(),
        "smallest_eigenvalue": float(eig[0]),
        "per_synergy_dispersion": np.std(z, axis=0).tolist(),
        "trial_stratified_occupancy": occupancy,
        "finite_rates": bool(np.isfinite(r).all()),
    }


def trial_stratified_split_half(
    scores: np.ndarray, rates: np.ndarray, *, trial_ids: np.ndarray,
) -> dict[str, object]:
    z = np.asarray(scores, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    ids = np.asarray(trial_ids)
    unique = np.unique(ids)
    even = unique[unique % 2 == 0]
    odd = unique[unique % 2 == 1]
    if even.size == 0 or odd.size == 0:
        return {"weight_flattened_pearson": None, "intercept_pearson": None, "split": "unavailable"}
    even_mask = np.isin(ids, even)
    odd_mask = np.isin(ids, odd)
    w_even, b_even = fit_all_units(z[even_mask], r[even_mask])
    w_odd, b_odd = fit_all_units(z[odd_mask], r[odd_mask])

    def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
        x = left.reshape(-1)
        y = right.reshape(-1)
        x, y = x - x.mean(), y - y.mean()
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        return None if denom <= 1.0e-12 else float(x.dot(y) / denom)

    return {
        "even_trials": even.tolist(),
        "odd_trials": odd.tolist(),
        "weight_flattened_pearson": _pearson(w_even, w_odd),
        "intercept_pearson": _pearson(b_even, b_odd),
    }


def normalize_carriers(raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(raw, dtype=np.float64) - mean) / scale


def deterministic_row_permutation(n_units: int, *, session_name: str, seed: int) -> np.ndarray:
    payload = f"m1-emg-syn3-rs4-v1:{int(seed)}:{session_name}:{int(n_units)}".encode()
    generator = np.random.RandomState(int.from_bytes(hashlib.sha256(payload).digest()[:4], "little"))
    order = generator.permutation(int(n_units))
    if np.array_equal(order, np.arange(int(n_units))):
        order = np.roll(order, 1)
    return order.astype(np.int64)


def deterministic_trial_derangement(n_trials: int, *, session_name: str, seed: int) -> np.ndarray:
    _require(n_trials >= 2, "LS4 needs at least two trials")
    payload = f"m1-emg-syn3-ls4-v1:{int(seed)}:{session_name}:{int(n_trials)}".encode()
    shift = 1 + int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (n_trials - 1)
    order = np.roll(np.arange(n_trials, dtype=np.int64), shift)
    _require(not np.any(order == np.arange(n_trials)), "LS4 is not a complete derangement")
    return order


def derange_trial_association(
    scores: np.ndarray, rates: np.ndarray, trial_ids: np.ndarray, session_name: str, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(scores, dtype=np.float64).copy()
    r = np.asarray(rates, dtype=np.float64)
    ids = np.asarray(trial_ids)
    unique = np.unique(ids)
    order = deterministic_trial_derangement(len(unique), session_name=session_name, seed=seed)
    remapped = np.empty_like(z)
    for dest_index, source_index in enumerate(order):
        dest_trial = unique[dest_index]
        source_trial = unique[source_index]
        remapped[ids == dest_trial] = z[ids == source_trial]
    return remapped, r


def source_normalizer(raw_carriers: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([np.asarray(item, dtype=np.float64) for item in raw_carriers], axis=0)
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale[scale <= 1.0e-6] = 1.0
    return mean, scale


def build_controls(
    *,
    raw_syn3: np.ndarray,
    normalizer_mean: np.ndarray,
    normalizer_scale: np.ndarray,
    session_name: str,
    seed: int,
    n_units: int,
    target_fit_invoked: bool,
    raw_ls4: np.ndarray | None = None,
    raw_pca3: np.ndarray | None = None,
) -> dict[str, object]:
    syn = normalize_carriers(raw_syn3, normalizer_mean, normalizer_scale)
    zero = np.zeros((n_units, plan.CARRIER_DIM), dtype=np.float64)
    b4 = syn.copy()
    b4[:, :3] = 0.0
    rs = syn[deterministic_row_permutation(n_units, session_name=session_name, seed=seed)]
    payload = {
        "Zero4": zero,
        "Syn3": syn,
        "RS4": rs,
        "B4": b4,
        "zero4_target_fit_calls": 0 if not target_fit_invoked else int(target_fit_invoked),
    }
    if raw_ls4 is not None:
        payload["LS4"] = normalize_carriers(raw_ls4, normalizer_mean, normalizer_scale)
    if raw_pca3 is not None:
        payload["PCA3"] = normalize_carriers(raw_pca3, normalizer_mean, normalizer_scale)
    return payload


def trial_mean_pca_reliability(emg_means: np.ndarray, rates: np.ndarray) -> dict[str, object]:
    """Historical trial-mean signed-PCA split-half, disclosed beside Syn3."""
    means = np.asarray(emg_means, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    _require(means.shape[0] == r.shape[0] >= 2, "trial-mean reliability rows")
    basis = fit_source_pca(means)
    scores = project_basis(means, basis)
    trial_ids = np.arange(means.shape[0], dtype=np.int64)
    return trial_stratified_split_half(scores, r, trial_ids=trial_ids)
