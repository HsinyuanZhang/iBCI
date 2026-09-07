#!/usr/bin/env python3
"""Pure-CPU selection-stage machinery for the M1 deployable Δ-identity map.

This module is intentionally data-loader and checkpoint agnostic.  It accepts only
already prepared held-in *selection* windows declared as [10,210), and has no API
for report, held-out, EvalAI, model training, or target-support teacher-oracle work.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

SUPPORT_START, SUPPORT_END = 0, 10
SELECTION_START, SELECTION_END = 10, 210
SEALED_REPORT_START = 210
RANK_GRID = (1, 2, 3)
LAMBDA_GRID = (1.0e-4, 1.0e-2, 1.0, 100.0)
ARM_NAMES = ("full", "rate_only", "label_shuffle", "rate_residualized_condition_only")
FeatureArm = Literal["full", "rate_only", "label_shuffle", "rate_residualized_condition_only"]


def _finite(name: str, value: np.ndarray, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (ndim is not None and array.ndim != ndim) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite rank-{ndim if ndim is not None else 'any'} array")
    return array


def _stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little")


def deterministic_support_label_shuffle(labels: np.ndarray, *, session_name: str, seed: int) -> np.ndarray:
    """Permute support positions deterministically; preserve multiset and force nonidentity permutation."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.shape != (SUPPORT_END,):
        raise ValueError("label shuffle accepts exactly chronological M=10 support labels")
    permutation = np.random.RandomState(_stable_seed("m1-dla-label-shuffle", session_name, seed)).permutation(SUPPORT_END)
    if np.array_equal(permutation, np.arange(SUPPORT_END)):
        permutation = np.roll(permutation, 1)
    return labels[permutation]


@dataclass(frozen=True)
class UnitSupportFeatures:
    """Raw support-only per-unit feature tables, before any source-train transform."""

    rate: np.ndarray          # [N,2] = log mean rate, log mean exposure
    conditioned: np.ndarray   # [N,8] = four condition rates, four missing masks

    def __post_init__(self) -> None:
        rate, conditioned = _finite("rate", self.rate, 2), _finite("conditioned", self.conditioned, 2)
        if rate.shape[1] != 2 or conditioned.shape != (rate.shape[0], 8):
            raise ValueError("feature tables require rate [N,2] and conditioned [N,8]")


def support_unit_features(
    spike_sums: np.ndarray,
    exposure: np.ndarray,
    obj_id: np.ndarray,
    *,
    session_name: str,
    seed: int,
    shuffle_labels: bool = False,
) -> UnitSupportFeatures:
    """Construct the frozen `F(C)` ingredients from M10 support only.

    `spike_sums` is [10,N], `exposure` is [10], and no query object is accepted.
    Missing condition levels are represented by a 0 rate and 0 mask rather than
    being imputed from any other session.
    """
    sums = _finite("spike_sums", spike_sums, 2)
    lengths = _finite("exposure", exposure, 1)
    labels = np.asarray(obj_id, dtype=np.int64).reshape(-1)
    if sums.shape[0] != SUPPORT_END or lengths.shape != (SUPPORT_END,) or labels.shape != (SUPPORT_END,):
        raise ValueError("support features require [10,N] sums, [10] exposure, and [10] obj_id")
    if np.any(lengths <= 0):
        raise ValueError("support exposure must be positive")
    if not set(labels.tolist()) <= {1, 2, 3, 4}:
        raise ValueError("M1 support obj_id must be within {1,2,3,4}")
    if shuffle_labels:
        labels = deterministic_support_label_shuffle(labels, session_name=session_name, seed=seed)
    trial_rates = sums / lengths[:, None]
    rate = np.column_stack((np.log1p(np.maximum(trial_rates.mean(axis=0), 0.0)),
                            np.full(sums.shape[1], np.log(lengths.mean()))))
    values, masks = [], []
    for level in (1, 2, 3, 4):
        present = labels == level
        values.append(trial_rates[present].mean(axis=0) if np.any(present) else np.zeros(sums.shape[1]))
        masks.append(np.full(sums.shape[1], float(np.any(present))))
    return UnitSupportFeatures(rate=rate, conditioned=np.column_stack(values + masks))


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = _finite("features", x, 2)
        if x.shape[1] != self.mean.shape[0]:
            raise ValueError("feature width differs from train-only standardizer")
        return (x - self.mean) / self.scale


@dataclass(frozen=True)
class ConditionResidualizer:
    rate_mean: np.ndarray
    rate_scale: np.ndarray
    condition_mean: np.ndarray
    weights: np.ndarray

    def transform(self, raw: UnitSupportFeatures) -> np.ndarray:
        z_rate = (raw.rate - self.rate_mean) / self.rate_scale
        predicted = self.condition_mean + z_rate @ self.weights
        return raw.conditioned - predicted


@dataclass(frozen=True)
class FittedFeatureTransform:
    arm: FeatureArm
    standardizer: Standardizer
    residualizer: ConditionResidualizer | None = None

    def transform(self, raw: UnitSupportFeatures) -> np.ndarray:
        if self.arm == "rate_only":
            features = raw.rate
        elif self.arm in {"full", "label_shuffle"}:
            features = np.column_stack((raw.rate, raw.conditioned))
        elif self.arm == "rate_residualized_condition_only":
            if self.residualizer is None:
                raise RuntimeError("rate-residualized arm lacks its train-only residualizer")
            features = self.residualizer.transform(raw)
        else:
            raise ValueError(f"unknown arm: {self.arm}")
        return self.standardizer.transform(features)


def _fit_standardizer(x: np.ndarray) -> Standardizer:
    x = _finite("train features", x, 2)
    scale = x.std(axis=0)
    scale[scale <= 1.0e-12] = 1.0
    return Standardizer(mean=x.mean(axis=0), scale=scale)


def _fit_condition_residualizer(train: Sequence[UnitSupportFeatures], ridge: float = 1.0e-6) -> ConditionResidualizer:
    if not train or ridge < 0.0:
        raise ValueError("residualizer needs nonempty train sessions and nonnegative ridge")
    rate, condition = np.concatenate([item.rate for item in train]), np.concatenate([item.conditioned for item in train])
    mean, scale = rate.mean(axis=0), rate.std(axis=0)
    scale[scale <= 1.0e-12] = 1.0
    z = (rate - mean) / scale
    condition_mean = condition.mean(axis=0)
    weights = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ (condition - condition_mean))
    return ConditionResidualizer(mean, scale, condition_mean, weights)


def fit_feature_transform(arm: FeatureArm, train: Sequence[UnitSupportFeatures]) -> FittedFeatureTransform:
    if arm not in ARM_NAMES or not train:
        raise ValueError("arm must be declared and transform train set nonempty")
    if arm == "rate_only":
        features, residualizer = np.concatenate([item.rate for item in train]), None
    elif arm in {"full", "label_shuffle"}:
        features, residualizer = np.concatenate([np.column_stack((item.rate, item.conditioned)) for item in train]), None
    else:
        residualizer = _fit_condition_residualizer(train)
        features = np.concatenate([residualizer.transform(item) for item in train])
    return FittedFeatureTransform(arm=arm, standardizer=_fit_standardizer(features), residualizer=residualizer)


@dataclass(frozen=True)
class ReducedRankMap:
    arm: FeatureArm
    rank: int
    ridge_lambda: float
    transform: FittedFeatureTransform
    basis_b: np.ndarray       # [W,r]
    ridge_weights: np.ndarray # [D+1,r], last row is unpenalized intercept

    def predict_delta(self, raw: UnitSupportFeatures) -> np.ndarray:
        x = self.transform.transform(raw)
        if x.shape[1] + 1 != self.ridge_weights.shape[0]:
            raise RuntimeError("ridge map and feature transform have incompatible widths")
        coefficients = np.column_stack((x, np.ones(x.shape[0])) ) @ self.ridge_weights
        return coefficients @ self.basis_b.T


def fit_reduced_rank_map(
    arm: FeatureArm,
    train_features: Sequence[UnitSupportFeatures],
    train_delta: Sequence[np.ndarray],
    *,
    rank: int,
    ridge_lambda: float,
) -> ReducedRankMap:
    """Fit `B_r` from only train Δ rows, then ridge `F -> A=ΔB_r` with intercept."""
    if rank not in RANK_GRID or ridge_lambda not in LAMBDA_GRID or len(train_features) != len(train_delta) or not train_features:
        raise ValueError("fit violates frozen rank/lambda grid or aligned source-train inputs")
    transform = fit_feature_transform(arm, train_features)
    x = np.concatenate([transform.transform(item) for item in train_features])
    delta = np.concatenate([_finite("train Delta_star", item, 2) for item in train_delta])
    if delta.shape[0] != x.shape[0] or rank > min(delta.shape):
        raise ValueError("train Δ rows do not match F rows or requested rank")
    _, _, vt = np.linalg.svd(delta, full_matrices=False)
    basis = vt[:rank].T
    target_a = delta @ basis
    design = np.column_stack((x, np.ones(x.shape[0])))
    penalty = np.eye(design.shape[1]) * ridge_lambda
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ target_a)
    return ReducedRankMap(arm, rank, ridge_lambda, transform, basis, weights)


@dataclass(frozen=True)
class SourceSelectionSession:
    """One source session allowed in inner LOSO; its query is exactly [10,210)."""

    name: str
    e0: np.ndarray                 # [N,W]
    delta_star: np.ndarray         # [N,W], source-supervision only
    features: Mapping[str, UnitSupportFeatures]
    query_neural: np.ndarray
    query_behavior: np.ndarray
    query_window: tuple[int, int] = (SELECTION_START, SELECTION_END)

    def __post_init__(self) -> None:
        _validate_selection_session(self.name, self.e0, self.features, self.query_neural, self.query_behavior, self.query_window)
        if _finite("source Delta_star", self.delta_star, 2).shape != np.asarray(self.e0).shape:
            raise ValueError("source Delta_star must match E0 [N,W]")


@dataclass(frozen=True)
class OuterSelectionSession:
    """Outer-left-out selection gate deliberately has no `delta_star`/teacher target field."""

    name: str
    e0: np.ndarray
    features: Mapping[str, UnitSupportFeatures]
    query_neural: np.ndarray
    query_behavior: np.ndarray
    query_window: tuple[int, int] = (SELECTION_START, SELECTION_END)

    def __post_init__(self) -> None:
        _validate_selection_session(self.name, self.e0, self.features, self.query_neural, self.query_behavior, self.query_window)


def _validate_selection_session(name: str, e0: np.ndarray, features: Mapping[str, UnitSupportFeatures], neural: np.ndarray, behavior: np.ndarray, window: tuple[int, int]) -> None:
    if not name or window != (SELECTION_START, SELECTION_END):
        raise ValueError("selection stage accepts only a declared [10,210) query window")
    e0 = _finite("E0", e0, 2)
    if not set(ARM_NAMES) <= set(features) or any(item.rate.shape[0] != e0.shape[0] for item in features.values()):
        raise ValueError("every selection session needs all declared per-unit support feature arms")
    neural, behavior = _finite("selection query neural", neural), _finite("selection query behavior", behavior)
    if neural.shape[0] != behavior.shape[0]:
        raise ValueError("selection query neural and behavior must share time axis")


DecodeFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def variance_weighted_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = _finite("prediction", prediction), _finite("target", target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    target = target.reshape(target.shape[0], -1)
    prediction = prediction.reshape(prediction.shape[0], -1)
    variance = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    active = variance > 0.0
    if not np.any(active):
        raise ValueError("R2 target has no nonconstant output")
    residual = ((target - prediction) ** 2).sum(axis=0)
    return float(1.0 - residual[active].sum() / variance[active].sum())


def paired_contiguous_trial_block_bootstrap(
    f0_prediction: np.ndarray,
    dla_prediction: np.ndarray,
    target: np.ndarray,
    *,
    block_trials: int,
    replicates: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    """Paired uncertainty on trial-major predictions; every replicate recomputes whole-sample R².

    The first axis must be chronological trials. Resampling is by fixed contiguous
    blocks, then all selected trials/bins/outputs are concatenated before a single
    variance-weighted R² calculation. It deliberately has no per-trial-R² API.
    """
    f0, dla, target = _finite("F0 predictions", f0_prediction), _finite("DLA predictions", dla_prediction), _finite("query targets", target)
    if f0.shape != dla.shape or f0.shape != target.shape or f0.ndim < 2 or block_trials <= 0 or replicates < 2:
        raise ValueError("paired bootstrap needs matching trial-major tensors, positive blocks, and >=2 replicates")
    n_trials = f0.shape[0]
    if n_trials < block_trials:
        raise ValueError("block length exceeds selection trial count")
    blocks = [np.arange(start, min(start + block_trials, n_trials)) for start in range(0, n_trials, block_trials)]
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(replicates):
        chosen = np.concatenate([blocks[index] for index in rng.randint(0, len(blocks), size=len(blocks))])
        deltas.append(variance_weighted_r2(dla[chosen], target[chosen]) - variance_weighted_r2(f0[chosen], target[chosen]))
    values = np.asarray(deltas, dtype=np.float64)
    standard_error = float(values.std(ddof=1))
    return {"block_trials": block_trials, "replicates": replicates, "paired_delta_r2_mean": float(values.mean()),
            "paired_delta_r2_standard_error": standard_error, "paired_delta_r2_ci95_low": float(np.quantile(values, 0.025)),
            "paired_delta_r2_ci95_high": float(np.quantile(values, 0.975)), "two_sided_mde_r2": float(1.96 * standard_error),
            "bootstrap_deltas_r2": values.tolist()}


def _score(decoder: DecodeFn, session: SourceSelectionSession | OuterSelectionSession, identity: np.ndarray) -> float:
    prediction = decoder(session.query_neural, identity)
    return variance_weighted_r2(prediction, session.query_behavior)


@dataclass(frozen=True)
class CandidateResult:
    arm: FeatureArm
    rank: int
    ridge_lambda: float
    inner_validation_delta_r2: Mapping[str, float]

    @property
    def mean_delta_r2(self) -> float:
        return float(np.mean(list(self.inner_validation_delta_r2.values())))


def inner_loso_candidates(
    arm: FeatureArm,
    outer_train: Sequence[SourceSelectionSession],
    *,
    decoder: DecodeFn,
    outer_left_out_name: str,
) -> list[CandidateResult]:
    """Inner-LOSO selects only rank/lambda, never sees the outer-left-out object."""
    names = [item.name for item in outer_train]
    if len(outer_train) < 2 or len(names) != len(set(names)) or outer_left_out_name in names:
        raise ValueError("inner LOSO requires distinct outer-train sources and excludes outer-left-out")
    # Identity baseline is candidate-independent; decode once per inner validation
    # session/arm rather than 12 times in the rank/lambda grid.
    baselines = {item.name: _score(decoder, item, item.e0) for item in outer_train}
    candidates: list[CandidateResult] = []
    for rank in RANK_GRID:
        for ridge_lambda in LAMBDA_GRID:
            deltas: dict[str, float] = {}
            for validation in outer_train:
                train = [item for item in outer_train if item.name != validation.name]
                fitted = fit_reduced_rank_map(arm, [item.features[arm] for item in train], [item.delta_star for item in train], rank=rank, ridge_lambda=ridge_lambda)
                ehat = validation.e0 + fitted.predict_delta(validation.features[arm])
                deltas[validation.name] = _score(decoder, validation, ehat) - baselines[validation.name]
            candidates.append(CandidateResult(arm, rank, ridge_lambda, deltas))
    return candidates


def inner_loso_select(
    arm: FeatureArm,
    outer_train: Sequence[SourceSelectionSession],
    *,
    decoder: DecodeFn,
    outer_left_out_name: str,
) -> CandidateResult:
    """Lock rank/lambda by the inner validation behavior-R² criterion only."""
    candidates = inner_loso_candidates(arm, outer_train, decoder=decoder, outer_left_out_name=outer_left_out_name)
    return max(candidates, key=lambda item: (item.mean_delta_r2, -item.rank, -item.ridge_lambda))


@dataclass(frozen=True)
class LockedOuterGateResult:
    arm: FeatureArm
    rank: int
    ridge_lambda: float
    outer_selection_delta_r2: float


def run_locked_outer_selection_gate(
    arm: FeatureArm,
    outer_train: Sequence[SourceSelectionSession],
    outer_left_out: OuterSelectionSession,
    locked: CandidateResult,
    *,
    decoder: DecodeFn,
) -> LockedOuterGateResult:
    """One outer `[10,210)` evaluation with a prelocked candidate; no target-leftout argument exists."""
    if locked.arm != arm or outer_left_out.name in {item.name for item in outer_train}:
        raise ValueError("outer gate needs matching locked arm and a distinct outer-left-out session")
    fitted = fit_reduced_rank_map(arm, [item.features[arm] for item in outer_train], [item.delta_star for item in outer_train], rank=locked.rank, ridge_lambda=locked.ridge_lambda)
    ehat = outer_left_out.e0 + fitted.predict_delta(outer_left_out.features[arm])
    return LockedOuterGateResult(arm, locked.rank, locked.ridge_lambda, _score(decoder, outer_left_out, ehat) - _score(decoder, outer_left_out, outer_left_out.e0))


def feature_state_accounting(arm: FeatureArm, rank: int, *, n_units: int = 64, width: int = 100) -> dict[str, int]:
    """Static accounting for the later report; no checkpoint/model inference is performed."""
    if arm not in ARM_NAMES or rank not in RANK_GRID or n_units <= 0 or width <= 0:
        raise ValueError("invalid frozen arm/rank/state dimensions")
    feature_dim = {"full": 10, "label_shuffle": 10, "rate_only": 2, "rate_residualized_condition_only": 8}[arm]
    return {"support_feature_values": n_units * feature_dim, "residual_coefficients": n_units * rank,
            "shared_basis_values": width * rank, "ridge_weight_values": (feature_dim + 1) * rank,
            "feature_summary_mac_proxy": 10 * n_units * (2 + 4),
            "ridge_coefficient_map_mac": n_units * (feature_dim + 1) * rank,
            "low_rank_ABt_map_mac": n_units * rank * width}
