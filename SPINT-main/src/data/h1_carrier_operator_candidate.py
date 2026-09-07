"""CPU-only candidates for an analytic H1 carrier operator.

This module is intentionally a *standalone numerical candidate*.  It accepts
already-materialized rate and kinematic arrays and has no dataset, checkpoint,
or model dependency.  In particular, it is not imported by any active H1
producer.

The two public paths are algebraically equivalent for a fixed calibration
block:

``fit_o1``
    computes the sufficient statistics directly and removes the calibration-
    length-squared leverage computation.

``CanonicalFP64Accumulator.solve_carrier``
    exposes the same computation through an FP64 running state.  Its packed
    state contains ``n``, ``DtD``, ``Dty``, and ``yty`` only.  Feed rows in
    source order for the canonical accumulator result.  Combining independently
    accumulated states is valid in real arithmetic, but floating point merge
    grouping is deliberately not promised to be bitwise associative.

This candidate does not add any adaptive decay, sparsity penalty, or numerical
format conversion.  Those are separate design questions rather than an exact
operator refactor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-12


class CarrierOperatorError(ValueError):
    """Raised when a numerical carrier-operator contract is violated."""


def _as_finite_f64(value: np.ndarray, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise CarrierOperatorError(f"{name} must be {ndim}-D, got {array.ndim}-D")
    if not np.isfinite(array).all():
        raise CarrierOperatorError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class FrozenCarrierOperator:
    """Frozen matrices used after rates and labels have been materialized.

    ``pcs`` is the selected projection matrix with shape ``[Q, N]`` rather
    than the full PCA basis.  ``U`` maps each of ``Y`` label dimensions into a
    ``C``-dimensional carrier, normally ``C=4``.
    """

    mean: np.ndarray
    scale: np.ndarray
    pcs: np.ndarray
    ridge_lambda: float
    U: np.ndarray
    mu: np.ndarray
    tau2: float

    def __post_init__(self) -> None:
        mean = _as_finite_f64(self.mean, name="mean", ndim=1)
        scale = _as_finite_f64(self.scale, name="scale", ndim=1)
        pcs = _as_finite_f64(self.pcs, name="pcs", ndim=2)
        U = _as_finite_f64(self.U, name="U", ndim=2)
        mu = _as_finite_f64(self.mu, name="mu", ndim=1)
        if mean.size == 0 or pcs.shape[0] == 0:
            raise CarrierOperatorError("operator must contain at least one channel and one component")
        if scale.shape != mean.shape or np.any(scale <= 0.0):
            raise CarrierOperatorError("scale must be positive and match mean")
        if pcs.shape[1] != mean.size:
            raise CarrierOperatorError("pcs must have one column per rate channel")
        if U.shape[0] == 0 or U.shape[1] == 0 or U.shape[1] != mu.size:
            raise CarrierOperatorError("U/mu carrier dimensions are inconsistent")
        if not np.isfinite(self.ridge_lambda) or self.ridge_lambda < 0.0:
            raise CarrierOperatorError("ridge_lambda must be finite and nonnegative")
        if not np.isfinite(self.tau2) or self.tau2 <= EPS:
            raise CarrierOperatorError("tau2 must be finite and positive")
        # Store contiguous FP64 values so the candidate has a clear arithmetic
        # contract even when callers pass integer or non-contiguous arrays.
        object.__setattr__(self, "mean", np.ascontiguousarray(mean))
        object.__setattr__(self, "scale", np.ascontiguousarray(scale))
        object.__setattr__(self, "pcs", np.ascontiguousarray(pcs))
        object.__setattr__(self, "U", np.ascontiguousarray(U))
        object.__setattr__(self, "mu", np.ascontiguousarray(mu))

    @property
    def num_channels(self) -> int:
        return int(self.mean.size)

    @property
    def projection_dim(self) -> int:
        return int(self.pcs.shape[0])

    @property
    def label_dim(self) -> int:
        return int(self.U.shape[0])

    @property
    def carrier_dim(self) -> int:
        return int(self.U.shape[1])

    @property
    def design_dim(self) -> int:
        return self.projection_dim + 1


def projected_rates(rates: np.ndarray, operator: FrozenCarrierOperator) -> np.ndarray:
    """Return standardized, fixed-PCA rate features ``z`` with shape ``[B,Q]``."""

    values = _as_finite_f64(rates, name="rates", ndim=2)
    if values.shape[0] == 0 or values.shape[1] != operator.num_channels:
        raise CarrierOperatorError("rates must be nonempty [B, num_channels]")
    return ((values - operator.mean[None, :]) / operator.scale[None, :]) @ operator.pcs.T


def design_from_rates(rates: np.ndarray, operator: FrozenCarrierOperator) -> np.ndarray:
    """Build ``D = [1, z]`` without touching any dataset or model state."""

    z = projected_rates(rates, operator)
    return np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))


@dataclass(frozen=True)
class CarrierSufficientStatistics:
    """Full FP64 sufficient statistics used by the analytic readout."""

    n: int
    DtD: np.ndarray
    Dty: np.ndarray
    yty: np.ndarray

    def __post_init__(self) -> None:
        if int(self.n) <= 0:
            raise CarrierOperatorError("n must be a positive integer")
        dtd = _as_finite_f64(self.DtD, name="DtD", ndim=2)
        dty = _as_finite_f64(self.Dty, name="Dty", ndim=2)
        yty = _as_finite_f64(self.yty, name="yty", ndim=2)
        if dtd.shape[0] == 0 or dtd.shape[0] != dtd.shape[1]:
            raise CarrierOperatorError("DtD must be nonempty square")
        if dty.shape[0] != dtd.shape[0] or yty.shape != (dty.shape[1], dty.shape[1]):
            raise CarrierOperatorError("sufficient-statistic shapes are inconsistent")
        if not np.allclose(dtd, dtd.T, rtol=0.0, atol=1.0e-12):
            raise CarrierOperatorError("DtD must be symmetric")
        if not np.allclose(yty, yty.T, rtol=0.0, atol=1.0e-12):
            raise CarrierOperatorError("yty must be symmetric")
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "DtD", np.ascontiguousarray((dtd + dtd.T) / 2.0))
        object.__setattr__(self, "Dty", np.ascontiguousarray(dty))
        object.__setattr__(self, "yty", np.ascontiguousarray((yty + yty.T) / 2.0))

    @property
    def design_dim(self) -> int:
        return int(self.DtD.shape[0])

    @property
    def label_dim(self) -> int:
        return int(self.Dty.shape[1])


def statistics_from_design_labels(design: np.ndarray, labels: np.ndarray) -> CarrierSufficientStatistics:
    """Form ``n, DtD, Dty, yty`` directly, never a calibration-bin square matrix."""

    D = _as_finite_f64(design, name="design", ndim=2)
    y = _as_finite_f64(labels, name="labels", ndim=2)
    if D.shape[0] == 0 or y.shape[0] != D.shape[0] or D.shape[1] == 0 or y.shape[1] == 0:
        raise CarrierOperatorError("design and labels must be nonempty with matching rows")
    return CarrierSufficientStatistics(
        n=int(D.shape[0]),
        DtD=D.T @ D,
        Dty=D.T @ y,
        yty=y.T @ y,
    )


def statistics_from_rates_labels(
    rates: np.ndarray, labels: np.ndarray, operator: FrozenCarrierOperator
) -> CarrierSufficientStatistics:
    """O2 front end: make the fixed design, then form its sufficient statistics."""

    return statistics_from_design_labels(design_from_rates(rates, operator), labels)


def _regularizer(design_dim: int, ridge_lambda: float) -> np.ndarray:
    regularizer = np.eye(design_dim, dtype=np.float64) * float(ridge_lambda)
    regularizer[0, 0] = 0.0
    return regularizer


def _factor_and_solve(
    dtd: np.ndarray, dty: np.ndarray, operator: FrozenCarrierOperator
) -> tuple[np.ndarray, np.ndarray, float]:
    """Use one SPD factorization for beta, leverage, and covariance solves."""

    system = dtd + _regularizer(dtd.shape[0], operator.ridge_lambda)
    try:
        lower = np.linalg.cholesky(system)
        beta = _solve_from_cholesky(lower, dty)
        solve_system_dtd = _solve_from_cholesky(lower, dtd)
        # With symmetric S, solve(S, A.T).T equals A @ S^-1.  Reuse the same
        # Cholesky factor for the right-side solve and retain solve-only G.
        G = _solve_from_cholesky(lower, solve_system_dtd.T).T
    except np.linalg.LinAlgError as exc:
        raise CarrierOperatorError("ridge system is not positive definite") from exc
    return np.asarray(beta), np.asarray((G + G.T) / 2.0), float(np.trace(solve_system_dtd))


def _solve_from_cholesky(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``(L L.T)x=rhs`` from one already-computed lower Cholesky factor."""

    L = _as_finite_f64(lower, name="lower Cholesky factor", ndim=2)
    values = _as_finite_f64(rhs, name="right-hand side")
    if L.shape[0] != L.shape[1] or values.shape[0] != L.shape[0]:
        raise CarrierOperatorError("Cholesky factor/right-hand side shapes are inconsistent")
    was_vector = values.ndim == 1
    if values.ndim not in (1, 2):
        raise CarrierOperatorError("right-hand side must be a vector or matrix")
    work = values[:, None].copy() if was_vector else values.copy()
    # Explicit forward/back substitution avoids refactorizing either triangular
    # system.  Thus one Cholesky factor of S serves beta, S^-1 DtD, and the
    # right-side solve used to form G.
    for row in range(L.shape[0]):
        work[row] = (work[row] - L[row, :row] @ work[:row]) / L[row, row]
    for row in range(L.shape[0] - 1, -1, -1):
        work[row] = (work[row] - L[row + 1 :, row] @ work[row + 1 :]) / L[row, row]
    return work[:, 0] if was_vector else work


def _postprocess_carrier(
    beta: np.ndarray,
    G: np.ndarray,
    sigma2: np.ndarray,
    *,
    hat_trace: float,
    rss: np.ndarray,
    operator: FrozenCarrierOperator,
) -> dict[str, np.ndarray | float]:
    """Apply the unchanged fixed projection and EB shrinkage after the solve."""

    raw_rows = (operator.pcs.T @ beta[1:]) / operator.scale[:, None]
    raw_carrier = raw_rows @ operator.U
    projection = operator.pcs.T
    channel_factor = ((projection @ G[1:, 1:]) * projection).sum(axis=1) / np.square(operator.scale)
    # trace(U.T @ diag(sigma2) @ U) without materializing the diagonal matrix.
    projected_covariance_trace = float(np.sum(sigma2 * np.square(operator.U).sum(axis=1)))
    projected_variance = channel_factor * projected_covariance_trace / float(operator.carrier_dim)
    if np.any(~np.isfinite(projected_variance)) or np.any(projected_variance < 0.0):
        raise CarrierOperatorError("analytic EB projected variance is undefined")
    weight = operator.tau2 / (operator.tau2 + projected_variance)
    carrier = operator.mu[None, :] + weight[:, None] * (raw_carrier - operator.mu[None, :])
    if not np.isfinite(carrier).all():
        raise CarrierOperatorError("analytic carrier must be finite")
    return {
        "carrier": np.asarray(carrier, dtype=np.float64),
        "raw_carrier": np.asarray(raw_carrier, dtype=np.float64),
        "raw_rows": np.asarray(raw_rows, dtype=np.float64),
        "beta": np.asarray(beta, dtype=np.float64),
        "G": np.asarray(G, dtype=np.float64),
        "rss": np.asarray(rss, dtype=np.float64),
        "hat_trace": float(hat_trace),
        "sigma2": np.asarray(sigma2, dtype=np.float64),
        "projected_variance": np.asarray(projected_variance, dtype=np.float64),
        "weight": np.asarray(weight, dtype=np.float64),
    }


def solve_carrier(
    state: CarrierSufficientStatistics, operator: FrozenCarrierOperator
) -> dict[str, np.ndarray | float]:
    """O2 readout from sufficient statistics with a quadratic-form RSS.

    The leverage trace uses ``trace(solve(S, DtD))``.  The covariance factor
    is constructed as ``S^-1 DtD S^-1`` through two linear solves.  Hence this
    function has no calibration-bin-square intermediate and no explicit matrix
    inverse.
    """

    if state.design_dim != operator.design_dim or state.label_dim != operator.label_dim:
        raise CarrierOperatorError("state and frozen operator dimensions are inconsistent")
    beta, G, hat_trace = _factor_and_solve(state.DtD, state.Dty, operator)
    denominator = float(state.n - hat_trace)
    if not np.isfinite(denominator) or denominator <= EPS:
        raise CarrierOperatorError("ridge residual covariance degrees of freedom undefined")

    # diag(beta.T @ A) is the columnwise inner product of beta and A.
    cross = np.einsum("ij,ij->j", beta, state.Dty)
    quadratic = np.einsum("ij,ij->j", beta, state.DtD @ beta)
    rss = np.diag(state.yty) - 2.0 * cross + quadratic
    # Well-conditioned inputs should make this unnecessary.  It only removes
    # negative round-off at the scale of the operands, never a real negative
    # residual sum of squares.
    roundoff = np.finfo(np.float64).eps * np.maximum(1.0, np.diag(state.yty)) * 64.0
    if np.any(rss < -roundoff):
        raise CarrierOperatorError("quadratic residual computation became materially negative")
    rss = np.maximum(rss, 0.0)
    sigma2 = rss / denominator
    return _postprocess_carrier(beta, G, sigma2, hat_trace=hat_trace, rss=rss, operator=operator)


def fit_o1(rates: np.ndarray, labels: np.ndarray, operator: FrozenCarrierOperator) -> dict[str, np.ndarray | float]:
    """O1 minimal refactor from a materialized block.

    This intentionally retains the literal residual expression
    ``sum((labels - D @ beta)^2)``.  O2 is the separate sufficient-statistics
    path and is the only path that uses the quadratic-form RSS.
    """

    D = design_from_rates(rates, operator)
    y = _as_finite_f64(labels, name="labels", ndim=2)
    if y.shape != (D.shape[0], operator.label_dim):
        raise CarrierOperatorError("labels must match rate rows and operator label dimension")
    dtd = D.T @ D
    dty = D.T @ y
    beta, G, hat_trace = _factor_and_solve(dtd, dty, operator)
    denominator = float(D.shape[0] - hat_trace)
    if not np.isfinite(denominator) or denominator <= EPS:
        raise CarrierOperatorError("ridge residual covariance degrees of freedom undefined")
    rss = np.square(y - D @ beta).sum(axis=0)
    sigma2 = rss / denominator
    return _postprocess_carrier(beta, G, sigma2, hat_trace=hat_trace, rss=rss, operator=operator)


def _upper_indices(size: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(size)


def _pack_symmetric(matrix: np.ndarray) -> np.ndarray:
    values = _as_finite_f64(matrix, name="symmetric matrix", ndim=2)
    if values.shape[0] != values.shape[1]:
        raise CarrierOperatorError("packed matrix must be square")
    rows, cols = _upper_indices(values.shape[0])
    return np.asarray(values[rows, cols], dtype=np.float64)


def _unpack_symmetric(packed: np.ndarray, size: int) -> np.ndarray:
    expected = size * (size + 1) // 2
    values = _as_finite_f64(packed, name="packed symmetric matrix", ndim=1)
    if values.size != expected:
        raise CarrierOperatorError("packed symmetric matrix length is inconsistent")
    rows, cols = _upper_indices(size)
    result = np.zeros((size, size), dtype=np.float64)
    result[rows, cols] = values
    result[cols, rows] = values
    return result


@dataclass
class CanonicalFP64Accumulator:
    """Packed FP64 streaming state for the O2 operator.

    The state has exactly ``1 + P(P+1)/2 + P*Y + Y(Y+1)/2`` scalar values,
    where ``P=Q+1`` includes the intercept.  In the deployed H1 dimensions
    ``P=17, Y=7`` this is 301 floats.  ``update`` visits rows in supplied order;
    use a single chronological traversal (or ordered chunks) to obtain its
    canonical FP64 result.

    ``merged`` is intentionally explicit about its left/right grouping.  It is
    algebraically sound but cannot promise bitwise equality to a row-wise pass,
    because floating-point addition is not associative.
    """

    n: int
    _dtd_upper: np.ndarray
    Dty: np.ndarray
    _yty_upper: np.ndarray

    def __post_init__(self) -> None:
        if int(self.n) < 0:
            raise CarrierOperatorError("n must be nonnegative")
        dtd_upper = _as_finite_f64(self._dtd_upper, name="_dtd_upper", ndim=1)
        dty = _as_finite_f64(self.Dty, name="Dty", ndim=2)
        yty_upper = _as_finite_f64(self._yty_upper, name="_yty_upper", ndim=1)
        design_dim = dty.shape[0]
        label_dim = dty.shape[1]
        if design_dim == 0 or label_dim == 0:
            raise CarrierOperatorError("Dty must be nonempty")
        if dtd_upper.size != design_dim * (design_dim + 1) // 2:
            raise CarrierOperatorError("_dtd_upper length is inconsistent with Dty")
        if yty_upper.size != label_dim * (label_dim + 1) // 2:
            raise CarrierOperatorError("_yty_upper length is inconsistent with Dty")
        self.n = int(self.n)
        self._dtd_upper = np.ascontiguousarray(dtd_upper.copy())
        self.Dty = np.ascontiguousarray(dty.copy())
        self._yty_upper = np.ascontiguousarray(yty_upper.copy())

    @classmethod
    def zeros(cls, design_dim: int, label_dim: int) -> "CanonicalFP64Accumulator":
        if int(design_dim) <= 0 or int(label_dim) <= 0:
            raise CarrierOperatorError("design_dim and label_dim must be positive")
        p, y = int(design_dim), int(label_dim)
        return cls(
            n=0,
            _dtd_upper=np.zeros(p * (p + 1) // 2, dtype=np.float64),
            Dty=np.zeros((p, y), dtype=np.float64),
            _yty_upper=np.zeros(y * (y + 1) // 2, dtype=np.float64),
        )

    @classmethod
    def from_design_labels(cls, design: np.ndarray, labels: np.ndarray) -> "CanonicalFP64Accumulator":
        D = _as_finite_f64(design, name="design", ndim=2)
        y = _as_finite_f64(labels, name="labels", ndim=2)
        if D.shape[0] == 0 or y.shape[0] != D.shape[0] or D.shape[1] == 0 or y.shape[1] == 0:
            raise CarrierOperatorError("design and labels must be nonempty with matching rows")
        result = cls.zeros(D.shape[1], y.shape[1])
        result.update(D, y)
        return result

    @classmethod
    def from_rates_labels(
        cls, rates: np.ndarray, labels: np.ndarray, operator: FrozenCarrierOperator
    ) -> "CanonicalFP64Accumulator":
        return cls.from_design_labels(design_from_rates(rates, operator), labels)

    @property
    def design_dim(self) -> int:
        return int(self.Dty.shape[0])

    @property
    def label_dim(self) -> int:
        return int(self.Dty.shape[1])

    @property
    def DtD(self) -> np.ndarray:
        """Unpacked symmetric ``DtD`` view reconstructed from the packed state."""

        return _unpack_symmetric(self._dtd_upper, self.design_dim)

    @property
    def yty(self) -> np.ndarray:
        """Unpacked symmetric ``yty`` view reconstructed from the packed state."""

        return _unpack_symmetric(self._yty_upper, self.label_dim)

    @property
    def packed_float_count(self) -> int:
        return 1 + int(self._dtd_upper.size) + int(self.Dty.size) + int(self._yty_upper.size)

    def update(self, design: np.ndarray, labels: np.ndarray) -> "CanonicalFP64Accumulator":
        """Accumulate rows in exactly the supplied order, using FP64 rank-one sums."""

        D = _as_finite_f64(design, name="design", ndim=2)
        y = _as_finite_f64(labels, name="labels", ndim=2)
        if D.shape[0] == 0 or D.shape != (D.shape[0], self.design_dim) or y.shape != (D.shape[0], self.label_dim):
            raise CarrierOperatorError("update shapes do not match accumulator dimensions")
        d_rows, d_cols = _upper_indices(self.design_dim)
        y_rows, y_cols = _upper_indices(self.label_dim)
        for d, target in zip(D, y):
            self.n += 1
            self._dtd_upper += (d[:, None] * d[None, :])[d_rows, d_cols]
            self.Dty += d[:, None] * target[None, :]
            self._yty_upper += (target[:, None] * target[None, :])[y_rows, y_cols]
        return self

    def statistics(self) -> CarrierSufficientStatistics:
        """Materialize the small full matrices required by the final solve."""

        if self.n <= 0:
            raise CarrierOperatorError("cannot solve an empty accumulator")
        return CarrierSufficientStatistics(n=self.n, DtD=self.DtD, Dty=self.Dty, yty=self.yty)

    def solve_carrier(self, operator: FrozenCarrierOperator) -> dict[str, np.ndarray | float]:
        """Read out the analytic carrier from the current packed FP64 state."""

        return solve_carrier(self.statistics(), operator)

    def merged(self, right: "CanonicalFP64Accumulator") -> "CanonicalFP64Accumulator":
        """Return the left-then-right statistic merge.

        This operation preserves the real-number sufficient statistics.  The
        explicit grouping matters for FP64 rounding and is therefore recorded by
        the calling schedule rather than hidden behind a claim of bitwise
        associativity.
        """

        if not isinstance(right, CanonicalFP64Accumulator) or (
            right.design_dim != self.design_dim or right.label_dim != self.label_dim
        ):
            raise CarrierOperatorError("can only merge accumulator states of identical shape")
        return CanonicalFP64Accumulator(
            n=self.n + right.n,
            _dtd_upper=self._dtd_upper + right._dtd_upper,
            Dty=self.Dty + right.Dty,
            _yty_upper=self._yty_upper + right._yty_upper,
        )
