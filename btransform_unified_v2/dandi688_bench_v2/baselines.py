"""CPU-only fitted baselines for the DANDI-688 SUA/PMUA benchmark.

These primitives deliberately operate on *observed* PMUA electrode columns.
``canonical_indices`` carries the hardware coordinate of every input column;
an absent electrode is never treated as an observed zero while fitting a
mean, covariance, or factor model.  Canonical zero filling happens only after
centering, for a downstream feature matrix that requires all 96 coordinates.

No function in this file consumes behavior labels except :func:`fit_ridge`.
The eventual runner decides which record indices are M33 neural/calibration
support and Q50 queries.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal
import re

import numpy as np

from external_baselines_v1.fair_v2.numerics import (
    CoralMap,
    FactorModel,
    RidgeReadout,
    causal_lags,
    fit_ridge_grid,
    smooth_raw,
)


CANONICAL_PMUA_CHANNELS = 96
_SCALE_FLOOR = 1e-6


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a nonempty [time, observed-channel] matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _indices(indices: np.ndarray, width: int, canonical_size: int) -> np.ndarray:
    result = np.asarray(indices, dtype=np.int64)
    if result.ndim != 1 or len(result) != width:
        raise ValueError("canonical_indices must have one entry per observed column")
    if np.any(result < 0) or np.any(result >= canonical_size) or len(np.unique(result)) != len(result):
        raise ValueError("canonical_indices must be unique valid canonical coordinates")
    return result


def _intersection(source_indices: np.ndarray, target_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common, source_columns, target_columns = np.intersect1d(
        source_indices, target_indices, assume_unique=True, return_indices=True
    )
    if not len(common):
        raise ValueError("source and target have no jointly observed canonical electrodes")
    return common, source_columns, target_columns


@dataclass(frozen=True)
class CanonicalDiagZ:
    """Per-session observed-channel z-score embedded in canonical PMUA space."""

    canonical_indices: np.ndarray
    observed_columns: np.ndarray
    mean_: np.ndarray
    scale_: np.ndarray
    canonical_size: int = CANONICAL_PMUA_CHANNELS

    @classmethod
    def fit(cls, support: np.ndarray, canonical_indices: np.ndarray, *, allowed_indices: np.ndarray | None = None,
            canonical_size: int = CANONICAL_PMUA_CHANNELS) -> "CanonicalDiagZ":
        values = _matrix(support, "support")
        indices = _indices(canonical_indices, values.shape[1], canonical_size)
        columns = np.arange(values.shape[1])
        if allowed_indices is not None:
            allowed = np.asarray(allowed_indices, dtype=np.int64)
            keep = np.flatnonzero(np.isin(indices, allowed))
            if not len(keep):
                raise ValueError("no observed electrode overlaps the fixed reference support")
            values, indices, columns = values[:, keep], indices[keep], columns[keep]
        scale = values.std(axis=0)
        scale[scale < _SCALE_FLOOR] = 1.0
        return cls(indices, columns, values.mean(axis=0), scale, canonical_size)

    def transform(self, observed: np.ndarray) -> np.ndarray:
        values = _matrix(observed, "observed")
        if values.shape[1] <= int(self.observed_columns.max()):
            raise ValueError("observed channel count differs from fitted diag-z map")
        # A zero in an unfilled canonical column means zero *after centering*,
        # i.e. mean imputation; it is not included in fit statistics.
        result = np.zeros((len(values), self.canonical_size), dtype=np.float64)
        result[:, self.canonical_indices] = (values[:, self.observed_columns] - self.mean_) / self.scale_
        return result

    @property
    def diagnostics(self) -> dict:
        return {"method": "canonical_diag_z", "observed_count": int(len(self.canonical_indices)),
                "canonical_size": int(self.canonical_size), "missing_fill": "zero_after_centering"}


@dataclass(frozen=True)
class CanonicalCoral:
    """Target-to-source CORAL map on jointly observed PMUA electrodes only."""

    common_indices: np.ndarray
    target_columns: np.ndarray
    target_mean_: np.ndarray
    target_scale_: np.ndarray
    coral: CoralMap
    canonical_size: int

    @classmethod
    def fit(cls, source_support: np.ndarray, source_indices: np.ndarray,
            target_support: np.ndarray, target_indices: np.ndarray, *, ridge: float = 1e-3,
            shrinkage: float = .1, canonical_size: int = CANONICAL_PMUA_CHANNELS) -> "CanonicalCoral":
        source = _matrix(source_support, "source_support")
        target = _matrix(target_support, "target_support")
        source_ids = _indices(source_indices, source.shape[1], canonical_size)
        target_ids = _indices(target_indices, target.shape[1], canonical_size)
        common, source_columns, target_columns = _intersection(source_ids, target_ids)
        source_common, target_common = source[:, source_columns], target[:, target_columns]
        source_mean, source_scale = source_common.mean(0), source_common.std(0)
        target_mean, target_scale = target_common.mean(0), target_common.std(0)
        source_scale[source_scale < _SCALE_FLOOR] = 1.0
        target_scale[target_scale < _SCALE_FLOOR] = 1.0
        source_z = (source_common - source_mean) / source_scale
        target_z = (target_common - target_mean) / target_scale
        return cls(common, target_columns, target_mean, target_scale,
                   CoralMap.fit(source_z, target_z, ridge=ridge, shrinkage=shrinkage), canonical_size)

    def transform(self, target_observed: np.ndarray) -> np.ndarray:
        target = _matrix(target_observed, "target_observed")
        if target.shape[1] <= int(self.target_columns.max()):
            raise ValueError("target channel count differs from fitted CORAL map")
        target_z = (target[:, self.target_columns] - self.target_mean_) / self.target_scale_
        mapped = self.coral.transform(target_z)
        result = np.zeros((len(target), self.canonical_size), dtype=np.float64)
        result[:, self.common_indices] = mapped
        return result

    @property
    def diagnostics(self) -> dict:
        return {"method": "canonical_coral", "common_observed_count": int(len(self.common_indices)),
                "canonical_size": int(self.canonical_size), "missing_fill": "zero_after_centering",
                **self.coral.diagnostics}


@dataclass(frozen=True)
class CanonicalAlignedFA:
    """FA fitted per observed array, aligned on common hardware electrodes."""

    target_fa: FactorModel
    target_columns: np.ndarray
    posterior_columns: np.ndarray
    target_mean_: np.ndarray
    target_scale_: np.ndarray
    rotation: np.ndarray
    stable_common_indices: np.ndarray
    posterior: Literal["all", "stable"]
    diagnostics: dict

    @classmethod
    def fit(cls, source_support: np.ndarray, source_indices: np.ndarray,
            target_support: np.ndarray, target_indices: np.ndarray, *, latent_dim: int,
            stable_fraction: float, posterior: Literal["all", "stable"], seed: int = 42,
            max_iter: int = 10_000, retry_max_iter: int = 50_000, tol: float = 1e-6, n_init: int = 3,
            noise_floor: float = 1e-6, loading_threshold: float = .01,
            canonical_size: int = CANONICAL_PMUA_CHANNELS) -> "CanonicalAlignedFA":
        if posterior not in {"all", "stable"}:
            raise ValueError("posterior must be 'all' or 'stable'")
        source, target = _matrix(source_support, "source_support"), _matrix(target_support, "target_support")
        source_ids = _indices(source_indices, source.shape[1], canonical_size)
        target_ids = _indices(target_indices, target.shape[1], canonical_size)
        common, source_columns, target_columns = _intersection(source_ids, target_ids)
        if not 0 < latent_dim < min(source.shape[1], target.shape[1]):
            raise ValueError("latent_dim must be smaller than each session's observed electrode count")
        source_mean, source_scale = source.mean(0), source.std(0)
        target_mean, target_scale = target.mean(0), target.std(0)
        source_scale[source_scale < _SCALE_FLOOR] = 1.0
        target_scale[target_scale < _SCALE_FLOOR] = 1.0
        source_z, target_z = (source - source_mean) / source_scale, (target - target_mean) / target_scale
        source_fa = _fit_factor(source_z, latent_dim, seed=seed, max_iter=max_iter,
                                retry_max_iter=retry_max_iter, tol=tol, n_init=n_init, noise_floor=noise_floor)
        target_fa = _fit_factor(target_z, latent_dim, seed=seed + 10_000, max_iter=max_iter,
                                retry_max_iter=retry_max_iter, tol=tol, n_init=n_init, noise_floor=noise_floor)
        source_loadings, target_loadings = source_fa.components_[source_columns], target_fa.components_[target_columns]
        eligible = np.flatnonzero((np.linalg.norm(source_loadings, axis=1) >= loading_threshold) &
                                  (np.linalg.norm(target_loadings, axis=1) >= loading_threshold))
        if len(eligible) < latent_dim:
            raise ValueError("too few common loading rows pass the threshold for latent alignment")
        retain = max(latent_dim, int(np.ceil(stable_fraction * len(common))))
        if retain > len(eligible):
            raise ValueError("stable fraction requires more eligible common rows than are available")
        rows = eligible.copy()
        while len(rows) > retain:
            rotation = _procrustes(target_loadings[rows], source_loadings[rows])
            residual = np.linalg.norm(target_loadings[rows] @ rotation - source_loadings[rows], axis=1)
            rows = np.delete(rows, int(np.argmax(residual)))
        cross_gram = target_loadings[rows].T @ source_loadings[rows]
        singular_values = np.linalg.svd(cross_gram, compute_uv=False)
        threshold = 1e-8 * max(float(singular_values.max()), 1e-12)
        if float(singular_values.min()) <= threshold:
            raise ValueError("common stable loading cross-Gram is numerically rank deficient")
        rotation = _procrustes(target_loadings[rows], source_loadings[rows])
        posterior_columns = np.arange(target.shape[1]) if posterior == "all" else target_columns[rows]
        diagnostics = {
            "method": f"canonical_aligned_fa_{posterior}", "latent_dim": int(latent_dim),
            "common_observed_count": int(len(common)), "eligible_count": int(len(eligible)),
            "stable_count": int(len(rows)), "stable_common_indices": common[rows].tolist(),
            "singular_values": singular_values.tolist(), "rank_threshold": threshold,
            "posterior": posterior, "loading_threshold": float(loading_threshold),
            "source_fa": source_fa.diagnostics, "target_fa": target_fa.diagnostics,
        }
        return cls(target_fa, target_columns, posterior_columns, target_mean, target_scale, rotation,
                   common[rows], posterior, diagnostics)

    def transform(self, target_observed: np.ndarray) -> np.ndarray:
        target = _matrix(target_observed, "target_observed")
        if target.shape[1] != self.target_fa.components_.shape[0]:
            raise ValueError("target channel count differs from fitted FA")
        beta = self.target_fa.posterior_matrix(self.posterior_columns)
        target_z = (target - self.target_mean_) / self.target_scale_
        return (target_z[:, self.posterior_columns] - self.target_fa.mean_[self.posterior_columns]) @ beta.T @ self.rotation


def _procrustes(target_loadings: np.ndarray, source_loadings: np.ndarray) -> np.ndarray:
    left, _singular, right = np.linalg.svd(target_loadings.T @ source_loadings, full_matrices=False)
    return left @ right


def _fit_factor(values: np.ndarray, latent_dim: int, *, seed: int, max_iter: int,
                retry_max_iter: int, tol: float, n_init: int, noise_floor: float) -> FactorModel:
    """Use fair_v2's bounded covariance-EM policy and fail closed on nonconvergence."""
    model = FactorModel.fit(values, latent_dim, seed=seed, max_iter=max_iter, tol=tol,
                            n_init=n_init, noise_floor=noise_floor)
    if model.diagnostics["converged"]:
        return model
    initial = model.diagnostics
    if retry_max_iter <= max_iter:
        raise RuntimeError(f"FA did not converge within {max_iter} iterations")
    model = FactorModel.fit(values, latent_dim, seed=seed, max_iter=retry_max_iter, tol=tol,
                            n_init=n_init, noise_floor=noise_floor)
    model.diagnostics["initial_attempt"] = initial
    if not model.diagnostics["converged"]:
        raise RuntimeError(f"FA did not converge within sealed retry budget {retry_max_iter}")
    return model


def fit_ridge(features: np.ndarray, labels: np.ndarray, alpha: float) -> RidgeReadout:
    """Closed-form, intercept-unpenalized ridge for WF-FSS M33-window labels."""
    return fit_ridge_grid_adaptive(features, labels, [alpha])[float(alpha)]


def fit_ridge_grid_adaptive(features: np.ndarray, labels: np.ndarray, alphas) -> dict[float, RidgeReadout]:
    """Intercept ridge grid with a dual solve when labelled rows are fewer than D.

    The dual branch is algebraically ``Xc.T @ (Xc @ Xc.T + alpha I)^-1 @ Yc``;
    it is essential for M33 WF-FSS H50 SUA, where about 825 labels fit a
    roughly 4,550-dimensional feature row.
    """
    x, y = _matrix(features, "features"), _matrix(labels, "labels")
    if len(x) != len(y) or not len(x): raise ValueError("feature/label rows must be equal and nonempty")
    values = [float(value) for value in alphas]
    if not values or any(value < 0 for value in values): raise ValueError("alphas must be nonnegative")
    if len(x) >= x.shape[1]: return fit_ridge_grid(x, y, values)
    mx, my = x.mean(0), y.mean(0); xc, yc = x - mx, y - my
    gram = xc @ xc.T
    out = {}
    identity = np.eye(len(x))
    for alpha in values:
        coefficient = xc.T @ np.linalg.solve(gram + alpha * identity, yc)
        out[alpha] = RidgeReadout(coefficient, my - mx @ coefficient, alpha)
    return out


def _ridge_stats(chunks: Any):
    """Exact intercept ridge from bounded feature chunks, never a giant pool."""
    n = 0
    sum_x = sum_y = gram = cross = None
    for features, labels in chunks:
        x, y = _matrix(features, "ridge features"), _matrix(labels, "ridge labels")
        if len(x) != len(y):
            raise ValueError("ridge feature/label chunk lengths differ")
        if sum_x is None:
            sum_x, sum_y = np.zeros(x.shape[1]), np.zeros(y.shape[1])
            gram, cross = np.zeros((x.shape[1], x.shape[1])), np.zeros((x.shape[1], y.shape[1]))
        if x.shape[1] != len(sum_x) or y.shape[1] != len(sum_y):
            raise ValueError("ridge chunk feature or output dimension drift")
        n += len(x); sum_x += x.sum(0); sum_y += y.sum(0); gram += x.T @ x; cross += x.T @ y
    if not n:
        raise ValueError("ridge requires at least one feature row")
    return n, sum_x, sum_y, gram, cross


def _ridge_from_stats(stats, alpha: float) -> RidgeReadout:
    n, sum_x, sum_y, gram, cross = stats
    mean_x, mean_y = sum_x / n, sum_y / n
    centered_gram = gram - n * np.outer(mean_x, mean_x)
    centered_cross = cross - n * np.outer(mean_x, mean_y)
    values, vectors = np.linalg.eigh((centered_gram + centered_gram.T) * .5)
    coefficient = vectors @ ((vectors.T @ centered_cross) / (values[:, None] + float(alpha)))
    return RidgeReadout(coefficient, mean_y - mean_x @ coefficient, float(alpha))


def fit_ridge_sufficient(chunks: Any, alpha: float) -> RidgeReadout:
    """Exact intercept ridge from bounded feature chunks, never a giant pool."""
    return _ridge_from_stats(_ridge_stats(chunks), alpha)


def _field(record: Any, name: str) -> Any:
    try:
        return getattr(record, name)
    except AttributeError as exc:
        raise TypeError(f"SessionData is missing required field {name!r}") from exc


def _source_guard(records: list[Any]) -> None:
    if not records:
        raise ValueError("at least one source SessionData record is required")
    forbidden = [str(_field(r, "session_id")) for r in records if str(_field(r, "split")).lower() not in {"train", "source", "source_train"}]
    if forbidden:
        raise ValueError(f"baseline source fit accepts train/source records only, got {forbidden}")
    pattern = re.compile(r"^(?:sub-C_ses-CO-)?(2015\d{4})$")
    non_2015 = [str(_field(r, "session_id")) for r in records if pattern.fullmatch(str(_field(r, "session_id"))) is None]
    if non_2015:
        raise ValueError(f"2015-only benchmark rejects non-2015 source sessions: {non_2015}")


def _config(config: dict | None) -> dict:
    out = {"alpha": 1e3, "history_bins": 10, "smooth": True, "bin_ms": 20., "tau_ms": 240.,
           "fa_dim": 10, "stable_fraction": .5, "shrinkage": .1, "coral_ridge": 1e-3,
           "fa_seed": 42, "fa_max_iter": 10_000, "fa_retry_max_iter": 50_000,
           "fa_tol": 1e-6, "fa_n_init": 3, "loading_threshold": .01}
    if config:
        out.update(config)
    return out


def _raw(record: Any, cfg: dict) -> np.ndarray:
    values = _matrix(_field(record, "neural"), "record.neural")
    return smooth_raw(values, bin_ms=cfg["bin_ms"], tau_ms=cfg["tau_ms"]) if cfg["smooth"] else values


def _support(record: Any, raw: np.ndarray) -> np.ndarray:
    indices = np.asarray(_field(record, "support_indices"), dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= len(raw)):
        raise ValueError("record.support_indices must select nonempty native neural rows")
    return raw[indices]


def _calibration_stream(record: Any, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Causal M33-only stream: outside-support rows are missing/zero, never data.

    We zero them before smoothing, so a retained support row cannot inherit a
    prehistory sample outside M33.  We zero them again after smoothing, so a
    later lag builder cannot turn missing rows into ``(0-mean)/std`` samples.
    """
    raw = _matrix(_field(record, "neural"), "record.neural")
    indices = np.asarray(_field(record, "support_indices"), dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= len(raw)):
        raise ValueError("record.support_indices must select nonempty native neural rows")
    mask = np.zeros(len(raw), dtype=bool); mask[indices] = True
    masked = np.zeros_like(raw); masked[mask] = raw[mask]
    filtered = smooth_raw(masked, bin_ms=cfg["bin_ms"], tau_ms=cfg["tau_ms"]) if cfg["smooth"] else masked
    filtered[~mask] = 0.
    return filtered, mask


def _pmua_indices(record: Any, width: int) -> np.ndarray:
    if str(_field(record, "representation")).upper() != "PMUA":
        raise ValueError("this source-free adaptation baseline is PMUA-only")
    return _indices(_field(record, "channel_indices"), width, CANONICAL_PMUA_CHANNELS)


def _features(values: np.ndarray, record: Any, history_bins: int) -> np.ndarray:
    endpoints = np.asarray(_field(record, "query_indices"), dtype=np.int64)
    return causal_lags(values, endpoints, history_bins=history_bins)


@dataclass
class FittedBaseline:
    """Pickle-safe source-supervised, target-unlabelled WF-family model."""

    method: str
    config: dict
    reference_session: str
    readout: RidgeReadout
    reference_support: np.ndarray
    reference_indices: np.ndarray
    source_diagnostics: dict
    ridge_stats: Any


@dataclass
class LocalWFFSS:
    """Target ridge on Full's same M33/time window, with dense velocity labels.

    This is a label-window matched calibration reference, not a label-content
    matched control: Full consumes direction labels while WF-FSS receives
    velocity at every allowed native row in that window.
    """

    config: dict
    mean_: np.ndarray
    scale_: np.ndarray
    readout: RidgeReadout
    representation: str
    fit_features: np.ndarray
    fit_labels: np.ndarray


def _target_adapter(method: str, reference_support: np.ndarray, reference_indices: np.ndarray,
                    record: Any, raw: np.ndarray, cfg: dict):
    calibration, _mask = _calibration_stream(record, cfg)
    support, indices = _support(record, calibration), _pmua_indices(record, raw.shape[1])
    if method in {"wf_zs_h0", "diag_z_wf"}:
        return CanonicalDiagZ.fit(support, indices, allowed_indices=reference_indices)
    if method == "coral_wf":
        return CanonicalCoral.fit(reference_support, reference_indices, support, indices,
                                  ridge=cfg["coral_ridge"], shrinkage=cfg["shrinkage"])
    if method in {"aligned_fa_wf", "aligned_fa_stable_wf"}:
        return CanonicalAlignedFA.fit(reference_support, reference_indices, support, indices,
                                     latent_dim=int(cfg["fa_dim"]), stable_fraction=float(cfg["stable_fraction"]),
                                     posterior="all" if method == "aligned_fa_wf" else "stable",
                                     seed=int(cfg["fa_seed"]), max_iter=int(cfg["fa_max_iter"]),
                                     retry_max_iter=int(cfg["fa_retry_max_iter"]), tol=float(cfg["fa_tol"]),
                                     n_init=int(cfg["fa_n_init"]), loading_threshold=float(cfg["loading_threshold"]))
    raise ValueError(f"unknown baseline method {method!r}")


def fit_baseline(method: str, source_records: list[Any], config: dict | None = None,
                 reference_session: str = "20150716") -> FittedBaseline:
    """Fit one of the five fair_v2 PMUA source-supervised/no-target-Y arms.

    ``wf_zs_h0`` is the one-bin basic reference.  ``aligned_fa_wf`` uses the
    target's all-observed-electrode posterior; ``aligned_fa_stable_wf`` uses
    only its stable shared-electrode posterior.  Hyperparameter selection is
    intentionally outside this function and must use source/dev only.
    """
    methods = {"wf_zs_h0", "diag_z_wf", "coral_wf", "aligned_fa_wf", "aligned_fa_stable_wf"}
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}")
    _source_guard(source_records)
    cfg = _config(config)
    by_id = {str(_field(record, "session_id")): record for record in source_records}
    if reference_session not in by_id:
        raise ValueError(f"reference session {reference_session!r} is absent from source records")
    reference = by_id[reference_session]
    reference_raw = _raw(reference, cfg)
    reference_calibration, _reference_mask = _calibration_stream(reference, cfg)
    reference_support = _support(reference, reference_calibration)
    reference_indices = _pmua_indices(reference, reference_raw.shape[1])
    history = 1 if method == "wf_zs_h0" else int(cfg["history_bins"])
    diagnostics = {}
    def chunks():
        for record in source_records:
            raw = _raw(record, cfg)
            adapter = _target_adapter(method, reference_support, reference_indices, record, raw, cfg)
            representation = adapter.transform(raw)
            endpoints = np.asarray(_field(record, "query_indices"), dtype=np.int64)
            velocity = _matrix(_field(record, "velocity"), "record.velocity")
            diagnostics[str(_field(record, "session_id"))] = getattr(adapter, "diagnostics", {})
            for start in range(0, len(endpoints), 4096):
                part = endpoints[start:start + 4096]
                yield causal_lags(representation, part, history_bins=history), velocity[part]
    stats = _ridge_stats(chunks())
    readout = _ridge_from_stats(stats, float(cfg["alpha"]))
    return FittedBaseline(method, cfg, reference_session, readout, reference_support, reference_indices, diagnostics, stats)


def refit_baseline_alpha(base: FittedBaseline, alpha: float) -> FittedBaseline:
    """Change only ridge alpha; reuse source alignments and sufficient statistics."""
    config = dict(base.config); config["alpha"] = float(alpha)
    return replace(base, config=config, readout=_ridge_from_stats(base.ridge_stats, float(alpha)))


def predict_baseline(model: FittedBaseline, record: Any) -> np.ndarray:
    """Predict Q50 rows without reading ``record.velocity`` or target labels."""
    raw = _raw(record, model.config)
    adapter = _target_adapter(model.method, model.reference_support, model.reference_indices, record, raw, model.config)
    history = 1 if model.method == "wf_zs_h0" else int(model.config["history_bins"])
    return np.asarray(model.readout.predict(_features(adapter.transform(raw), record, history)), dtype=np.float32)


def prepare_baseline_features(model: FittedBaseline, record: Any, *, return_diagnostics: bool = False):
    """Fit target's neural-only adapter once and return Q50 feature rows."""
    raw = _raw(record, model.config)
    adapter = _target_adapter(model.method, model.reference_support, model.reference_indices, record, raw, model.config)
    history = 1 if model.method == "wf_zs_h0" else int(model.config["history_bins"])
    features = _features(adapter.transform(raw), record, history)
    return (features, getattr(adapter, "diagnostics", {})) if return_diagnostics else features


def fit_wf_fss(record: Any, config: dict | None = None) -> LocalWFFSS:
    """Fit a target-local WF on exactly the generic allowed ``carrier_indices`` rows."""
    cfg = _config(config)
    raw = _raw(record, cfg)
    calibration, calibration_mask = _calibration_stream(record, cfg)
    indices = np.asarray(_field(record, "carrier_indices"), dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= len(raw)):
        raise ValueError("record.carrier_indices must select nonempty allowed labelled rows")
    mean, scale = _support(record, calibration).mean(0), _support(record, calibration).std(0)
    scale[scale < _SCALE_FLOOR] = 1.0
    z = (calibration - mean) / scale
    z[~calibration_mask] = 0.
    features = causal_lags(z, indices, history_bins=int(cfg["history_bins"]))
    velocity = _matrix(_field(record, "velocity"), "record.velocity")
    labels = velocity[indices]
    return LocalWFFSS(cfg, mean, scale, fit_ridge(features, labels, float(cfg["alpha"])),
                      str(_field(record, "representation")), features, labels)


def refit_wf_fss_alpha(base: LocalWFFSS, alpha: float) -> LocalWFFSS:
    config = dict(base.config); config["alpha"] = float(alpha)
    return replace(base, config=config, readout=fit_ridge_grid_adaptive(base.fit_features, base.fit_labels, [alpha])[float(alpha)])


def predict_wf_fss(model: LocalWFFSS, record: Any) -> np.ndarray:
    if str(_field(record, "representation")) != model.representation:
        raise ValueError("WF-FSS representation differs from its target-local fit")
    raw = _raw(record, model.config)
    features = _features((raw - model.mean_) / model.scale_, record, int(model.config["history_bins"]))
    return np.asarray(model.readout.predict(features), dtype=np.float32)


@dataclass(frozen=True)
class StaticAdapter:
    kind: Literal["diag_z", "coral"]
    target_indices: np.ndarray
    calibrated_columns: np.ndarray
    source_mean_: np.ndarray
    source_scale_: np.ndarray
    target_mean_: np.ndarray
    target_scale_: np.ndarray
    coral: CoralMap | None
    diagnostics: dict

    def transform(self, raw_compact: np.ndarray) -> np.ndarray:
        raw = _matrix(raw_compact, "raw_compact")
        if raw.shape[1] != len(self.target_indices):
            raise ValueError("static adapter channel geometry differs from fitted target")
        # Preserve the target compact PMUA geometry.  Coordinates absent from
        # the fixed hardware reference were observed, but cannot honestly be
        # reference-calibrated, so their raw values pass through unchanged.
        result = raw.copy()
        if self.kind == "diag_z":
            columns = self.calibrated_columns
            result[:, columns] = ((raw[:, columns] - self.target_mean_) / self.target_scale_
                                  * self.source_scale_ + self.source_mean_)
            return np.asarray(result, np.float32)
        assert self.coral is not None
        target_z = (raw[:, self.calibrated_columns] - self.target_mean_) / self.target_scale_
        result[:, self.calibrated_columns] = self.coral.transform(target_z) * self.source_scale_ + self.source_mean_
        return np.asarray(result, np.float32)


def fitstatic_adapter(source_records: list[Any], target_record: Any, *, kind: Literal["diag_z", "coral"],
                      shrinkage: float = .1, ridge: float = 1e-3,
                      reference_session: str = "20150716") -> StaticAdapter:
    """Fit a PMUA raw-set frozen-checkpoint input adapter without target Y.

    The output stays in the target record's compact observed-electrode order;
    no unobserved hardware coordinate is expanded into the RIFT input.  The
    fixed latest-source reference supplies both diag-z moments and CORAL's
    covariance.  Only target electrodes observed by that reference are
    transformed; other target observed columns pass through raw and the
    partial coverage is explicit in diagnostics.
    """
    if kind not in {"diag_z", "coral"}:
        raise ValueError("kind must be 'diag_z' or 'coral'")
    _source_guard(source_records)
    cfg = _config({"smooth": False})
    target_raw = _raw(target_record, cfg)
    target_support, target_ids = _support(target_record, target_raw), _pmua_indices(target_record, target_raw.shape[1])
    by_session = {str(_field(record, "session_id")): record for record in source_records}
    if reference_session not in by_session:
        raise ValueError(f"static reference session {reference_session!r} is absent from source records")
    reference_raw = _raw(by_session[reference_session], cfg)
    reference_support = _support(by_session[reference_session], reference_raw)
    reference_ids = _pmua_indices(by_session[reference_session], reference_raw.shape[1])
    _common, reference_columns, target_columns = _intersection(reference_ids, target_ids)
    source_reference = reference_support[:, reference_columns]
    target_reference = target_support[:, target_columns]
    source_mean, source_scale = source_reference.mean(0), source_reference.std(0)
    source_scale[source_scale < _SCALE_FLOOR] = 1.0
    target_mean, target_scale = target_reference.mean(0), target_reference.std(0)
    target_scale[target_scale < _SCALE_FLOOR] = 1.0
    source_z = (source_reference - source_mean) / source_scale
    target_z = (target_reference - target_mean) / target_scale
    coral = CoralMap.fit(source_z, target_z, ridge=ridge, shrinkage=shrinkage) if kind == "coral" else None
    return StaticAdapter(kind, target_ids, target_columns, source_mean, source_scale, target_mean, target_scale, coral,
                         {"kind": kind, "reference_session": reference_session,
                          "target_observed_count": int(len(target_ids)), "calibrated_observed_count": int(len(target_columns)),
                          "coverage_fraction": float(len(target_columns) / len(target_ids)),
                          "uncalibrated_observed_policy": "raw_passthrough", "target_labels_used": False,
                          "target_backpropagation": False, "shrinkage": float(shrinkage), "ridge": float(ridge)})


__all__ = [
    "CANONICAL_PMUA_CHANNELS", "CanonicalAlignedFA", "CanonicalCoral", "CanonicalDiagZ", "FittedBaseline",
    "LocalWFFSS", "StaticAdapter", "causal_lags", "fit_baseline", "fit_ridge", "fit_ridge_grid",
    "fit_wf_fss", "fitstatic_adapter", "fit_ridge_sufficient", "fit_ridge_grid_adaptive", "predict_baseline", "predict_wf_fss", "prepare_baseline_features", "refit_baseline_alpha", "refit_wf_fss_alpha", "smooth_raw",
]
