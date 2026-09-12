"""Numerical primitives for the fair_v2 paired baseline protocol.

All fitting and exported maps use float64.  Inputs are observations by channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

_EPS = np.finfo(np.float64).eps


def _array2(x: np.ndarray, name: str = "x") -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional (observations, channels), got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def smooth_raw(x: np.ndarray, bin_ms: float = 20, tau_ms: float = 240,
               extent: float = 1) -> np.ndarray:
    """Causally convolve a complete raw stream with FALCON's normalized kernel.

    The output is float64 and has the same shape as ``x``.  There is no trial
    reset: callers must pass the whole recording before selecting support bins.
    ``extent=1, tau_ms=240, bin_ms=20`` produces its official 12 taps.
    """
    a = _array2(x)
    if bin_ms <= 0 or tau_ms <= 0 or extent <= 0:
        raise ValueError("bin_ms, tau_ms, and extent must be positive")
    n_taps = max(1, int(round(extent * tau_ms / bin_ms)))
    t = np.arange(n_taps, dtype=np.float64) * float(bin_ms)
    kernel = np.exp(-t / float(tau_ms))
    kernel /= kernel.sum()
    out = np.empty_like(a, dtype=np.float64)
    for channel in range(a.shape[1]):
        out[:, channel] = np.convolve(a[:, channel], kernel, mode="full")[:a.shape[0]]
    return out


@dataclass
class Standardizer:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, support: np.ndarray) -> "Standardizer":
        a = _array2(support, "support")
        if len(a) == 0:
            raise ValueError("support must contain at least one row")
        mean = a.mean(axis=0)
        scale = a.std(axis=0)
        scale[scale == 0] = 1.0
        return cls(mean, scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        a = _array2(x)
        if a.shape[1] != self.mean_.size:
            raise ValueError("x channel count differs from fitted standardizer")
        return (a - self.mean_) / self.scale_



def _regularized_cov(x: np.ndarray, shrinkage: float, ridge: float) -> np.ndarray:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    n, c = x.shape
    if n < 2:
        raise ValueError("at least two support observations are required for covariance")
    centered = x - x.mean(axis=0)
    cov = centered.T @ centered / (n - 1)
    # Shrink toward an isotropic covariance; this preserves the average scale.
    isotropic = np.eye(c) * (np.trace(cov) / c)
    return (1.0 - shrinkage) * cov + shrinkage * isotropic + ridge * np.eye(c)


def _sym_psd_power(a: np.ndarray, exponent: float) -> np.ndarray:
    values, vectors = np.linalg.eigh((a + a.T) * 0.5)
    values = np.maximum(values, _EPS)
    return (vectors * values ** exponent) @ vectors.T


@dataclass
class CoralMap:
    source_mean_: np.ndarray
    target_mean_: np.ndarray
    matrix_: np.ndarray
    source_cov_: np.ndarray
    target_cov_: np.ndarray
    diagnostics: dict

    @classmethod
    def fit(cls, source_z_support: np.ndarray, target_z_support: np.ndarray,
            ridge: float = 1e-3, shrinkage: float = .1) -> "CoralMap":
        source = _array2(source_z_support, "source_z_support")
        target = _array2(target_z_support, "target_z_support")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source and target channel counts differ")
        cs = _regularized_cov(source, shrinkage, ridge)
        ct = _regularized_cov(target, shrinkage, ridge)
        # The map is applied to target query rows and maps target -> source.
        # With row-vector observations, A=Ct^-1/2 Cs^1/2 gives A' Ct A=Cs.
        matrix = _sym_psd_power(ct, -.5) @ _sym_psd_power(cs, .5)
        diagnostics = {
            "ridge": float(ridge), "shrinkage": float(shrinkage),
            "source_condition": float(np.linalg.cond(cs)),
            "target_condition": float(np.linalg.cond(ct)),
            "n_source": int(len(source)), "n_target": int(len(target)),
        }
        return cls(source.mean(0), target.mean(0), matrix, cs, ct, diagnostics)

    def transform(self, x: np.ndarray) -> np.ndarray:
        a = _array2(x)
        if a.shape[1] != self.matrix_.shape[0]:
            raise ValueError("x channel count differs from fitted CORAL map")
        return (a - self.target_mean_) @ self.matrix_ + self.source_mean_


@dataclass
class FactorModel:
    """Diagonal-noise FA fitted with covariance-based EM and KxK E steps."""
    components_: np.ndarray  # C, channels by latent dimensions
    mean_: np.ndarray
    noise_variance_: np.ndarray
    latent_dim: int
    diagnostics: dict

    @classmethod
    def fit(cls, x: np.ndarray, latent_dim: int, seed: int = 42,
            max_iter: int = 10_000, tol: float = 1e-6, n_init: int = 3,
            noise_floor: float = 1e-6) -> "FactorModel":
        a = _array2(x)
        n, c = a.shape
        if not 0 < latent_dim < c:
            raise ValueError("latent_dim must be positive and smaller than channel count")
        if n < 2 or max_iter < 1 or n_init < 1 or tol < 0 or noise_floor <= 0:
            raise ValueError("invalid FactorModel fitting parameters")
        mean = a.mean(0)
        xc = a - mean
        # The entire sample dependence is compressed once into CxC sufficient
        # statistics.  EM iterations below never form an NxC latent matrix.
        sample_cov = xc.T @ xc / n
        var = np.maximum(np.diag(sample_cov), noise_floor)
        rng = np.random.RandomState(seed)
        runs = []
        best = None
        for restart in range(n_init):
            # A covariance eigendecomposition is used only for initialization,
            # never in the repeated EM E-step.
            if restart == 0:
                eigval, eigvec = np.linalg.eigh((sample_cov + sample_cov.T) * .5)
                order = np.argsort(eigval)[::-1][:latent_dim]
                base_noise = max(float(np.median(eigval)), noise_floor)
                load = eigvec[:, order] * np.sqrt(np.maximum(eigval[order] - base_noise, noise_floor))
                psi = np.maximum(np.diag(sample_cov) - (load * load).sum(1), noise_floor)
            else:
                load = rng.normal(scale=np.sqrt(var.mean() / latent_dim), size=(c, latent_dim))
                psi = var.copy()
            previous_ll = -np.inf
            converged = False
            last_improvement = np.nan
            for iteration in range(1, max_iter + 1):
                try:
                    load, psi = _fa_em_step_sufficient(sample_cov, load, psi, noise_floor)
                except np.linalg.LinAlgError:
                    break
                ll = _fa_loglikelihood_sufficient(sample_cov, n, load, psi)
                last_improvement = (ll - previous_ll) / n
                if iteration > 1 and np.isfinite(last_improvement) and abs(last_improvement) <= tol:
                    converged = True
                    break
                previous_ll = ll
            final_ll = _fa_loglikelihood_sufficient(sample_cov, n, load, psi)
            record = {"restart": restart, "iterations": iteration, "converged": converged,
                      "log_likelihood": float(final_ll),
                      "per_sample_improvement": float(last_improvement),
                      "noise_min": float(psi.min()), "noise_max": float(psi.max())}
            runs.append(record)
            candidate = (final_ll, load.copy(), psi.copy(), record)
            if best is None or candidate[0] > best[0]:
                best = candidate
        best_ll, components, noise, best_record = best
        diagnostics = {"algorithm": "diagonal_psi_covariance_em_woodbury",
                       "best_log_likelihood": float(best_ll),
                       "best_restart": int(best_record["restart"]),
                       "converged": bool(best_record["converged"]),
                       "iterations": int(best_record["iterations"]),
                       "per_sample_improvement": float(best_record["per_sample_improvement"]),
                       "n_init": int(n_init), "max_iter": int(max_iter), "tol": float(tol),
                       "noise_floor": float(noise_floor), "runs": runs}
        return cls(components, mean, noise, latent_dim, diagnostics)

    @property
    def C_(self) -> np.ndarray:
        return self.components_

    @property
    def psi_(self) -> np.ndarray:
        return self.noise_variance_

    def posterior_matrix(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Return beta (latent by selected-channel) for E[z|x]."""
        if rows is None:
            c, psi = self.components_, self.noise_variance_
        else:
            rows = np.asarray(rows, dtype=int)
            c, psi = self.components_[rows], self.noise_variance_[rows]
        invpsi = 1.0 / psi
        m = np.eye(self.latent_dim) + c.T @ (invpsi[:, None] * c)
        return np.linalg.solve(m, (c * invpsi[:, None]).T)

    def transform(self, x: np.ndarray) -> np.ndarray:
        a = _array2(x)
        if a.shape[1] != self.components_.shape[0]:
            raise ValueError("x channel count differs from fitted FactorModel")
        return (a - self.mean_) @ self.posterior_matrix().T


def _fa_em_step_sufficient(sample_cov: np.ndarray, load: np.ndarray,
                           psi: np.ndarray, noise_floor: float) -> tuple[np.ndarray, np.ndarray]:
    """One diagonal-FA EM step from S=X'X/N, with no observation-axis arrays."""
    invpsi = 1.0 / psi
    m = np.eye(load.shape[1]) + load.T @ (invpsi[:, None] * load)
    minv = np.linalg.inv(m)
    beta = minv @ (load * invpsi[:, None]).T  # E[z x'] coefficient
    ezx = beta @ sample_cov                   # E[z x'] averaged over samples
    ezz = minv + beta @ sample_cov @ beta.T   # E[z z'] averaged over samples
    new_load = (sample_cov @ beta.T) @ np.linalg.inv(ezz)
    new_psi = np.maximum(np.diag(sample_cov) - np.diag(new_load @ ezx), noise_floor)
    return new_load, new_psi


def _fa_loglikelihood_sufficient(sample_cov: np.ndarray, n: int,
                                  load: np.ndarray, psi: np.ndarray) -> float:
    """Full-covariance Gaussian LL evaluated from CxC sample covariance."""
    invpsi = 1.0 / psi
    m = np.eye(load.shape[1]) + load.T @ (invpsi[:, None] * load)
    sign, logdet_m = np.linalg.slogdet(m)
    if sign <= 0:
        return -np.inf
    beta = np.linalg.solve(m, (load * invpsi[:, None]).T)
    # trace[(Psi + CC')^-1 S] by Woodbury, with no NxC intermediates.
    trace_precision_s = np.sum(np.diag(sample_cov) * invpsi) - np.trace(
        beta @ sample_cov @ (invpsi[:, None] * load)
    )
    per_sample = load.shape[0] * np.log(2*np.pi) + np.log(psi).sum() + logdet_m + trace_precision_s
    return float(-.5 * n * per_sample)


@dataclass
class AlignedFAMap:
    source_fa: FactorModel
    target_fa: FactorModel
    posterior: str
    stable_rows: np.ndarray
    rotation: np.ndarray
    matrix_: np.ndarray
    intercept_: np.ndarray
    diagnostics: dict

    @classmethod
    def fit(cls, source_fa: FactorModel, target_fa: FactorModel,
            stable_fraction: float = .5, posterior: str = "all",
            loading_threshold: float = .01) -> "AlignedFAMap":
        if posterior not in {"all", "stable"}:
            raise ValueError("posterior must be 'all' or 'stable'")
        if source_fa.components_.shape != target_fa.components_.shape:
            raise ValueError("source and target FA loading shapes differ")
        if not 0 < stable_fraction <= 1 or loading_threshold < 0:
            raise ValueError("stable_fraction must be in (0, 1] and threshold non-negative")
        c, k = source_fa.components_.shape
        eligible = np.flatnonzero((np.linalg.norm(source_fa.components_, axis=1) >= loading_threshold) &
                                  (np.linalg.norm(target_fa.components_, axis=1) >= loading_threshold))
        if len(eligible) < k:
            raise ValueError(f"only {len(eligible)} rows pass loading_threshold={loading_threshold}; need at least latent_dim={k}")
        retain = min(len(eligible), max(k, int(np.ceil(stable_fraction * c))))
        rows = eligible.copy()
        while len(rows) > retain:
            rot = _procrustes_rotation(target_fa.components_[rows], source_fa.components_[rows])
            residual = np.linalg.norm(target_fa.components_[rows] @ rot - source_fa.components_[rows], axis=1)
            rows = np.delete(rows, np.argmax(residual))
        rotation = _procrustes_rotation(target_fa.components_[rows], source_fa.components_[rows])
        posterior_rows = None if posterior == "all" else rows
        beta = target_fa.posterior_matrix(posterior_rows)
        observed_rows = np.arange(c) if posterior_rows is None else rows
        # row-vector input map x @ matrix + intercept
        matrix = beta.T @ rotation
        intercept = -target_fa.mean_[observed_rows] @ matrix
        diagnostics = {"posterior": posterior, "loading_threshold": float(loading_threshold),
                       "stable_fraction": float(stable_fraction), "eligible_count": int(len(eligible)),
                       "stable_count": int(len(rows)), "stable_rows": rows.tolist(),
                       "alignment_rmse": float(np.sqrt(np.mean((target_fa.components_[rows] @ rotation - source_fa.components_[rows])**2)))}
        return cls(source_fa, target_fa, posterior, rows, rotation, matrix, intercept, diagnostics)

    def transform(self, x: np.ndarray) -> np.ndarray:
        a = _array2(x)
        expected = self.matrix_.shape[0]
        if a.shape[1] == self.target_fa.components_.shape[0] and self.posterior == "stable":
            a = a[:, self.stable_rows]
        if a.shape[1] != expected:
            raise ValueError(f"x has {a.shape[1]} channels; expected {expected}")
        return a @ self.matrix_ + self.intercept_


def _procrustes_rotation(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(target.T @ source, full_matrices=False)
    return u @ vt


def causal_lags(features: np.ndarray, endpoints: np.ndarray, history_bins: int = 10) -> np.ndarray:
    """Newest-to-oldest causal feature rows with literal-zero left padding."""
    a = _array2(features, "features")
    ep = np.asarray(endpoints)
    if ep.ndim != 1 or not np.issubdtype(ep.dtype, np.integer):
        raise ValueError("endpoints must be a one-dimensional integer array")
    if history_bins < 1:
        raise ValueError("history_bins is the total bin count and must be at least one")
    if np.any(ep < 0) or np.any(ep >= len(a)):
        raise ValueError("endpoints must lie within the feature stream")
    out = np.zeros((len(ep), a.shape[1] * history_bins), dtype=np.float64)
    for j in range(history_bins):
        indices = ep - j
        valid = indices >= 0
        out[valid, j*a.shape[1]:(j+1)*a.shape[1]] = a[indices[valid]]
    return out


@dataclass
class RidgeReadout:
    coef_: np.ndarray
    intercept_: np.ndarray
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = _array2(x)
        if a.shape[1] != self.coef_.shape[0]:
            raise ValueError("x feature count differs from fitted ridge readout")
        return a @ self.coef_ + self.intercept_


def fit_ridge_grid(x: np.ndarray, y: np.ndarray, alphas: Iterable[float]) -> dict[float, RidgeReadout]:
    """Fit sklearn-Ridge-equivalent intercept models, sharing one Gram eigensystem."""
    a, target = _array2(x, "x"), _array2(y, "y")
    if len(a) != len(target) or len(a) == 0:
        raise ValueError("x and y must have equal nonzero observation counts")
    alpha_values = [float(alpha) for alpha in alphas]
    if not alpha_values or any(alpha < 0 for alpha in alpha_values):
        raise ValueError("alphas must be a nonempty iterable of non-negative values")
    mx, my = a.mean(0), target.mean(0)
    xc, yc = a - mx, target - my
    gram = xc.T @ xc
    cross = xc.T @ yc
    values, vectors = np.linalg.eigh((gram + gram.T) * .5)
    projected = vectors.T @ cross
    result = {}
    for alpha in alpha_values:
        coef = vectors @ (projected / (values[:, None] + alpha))
        result[alpha] = RidgeReadout(coef, my - mx @ coef, alpha)
    return result
