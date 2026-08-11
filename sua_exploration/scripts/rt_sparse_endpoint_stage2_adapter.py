#!/usr/bin/env python3
"""Isolated RT T4d endpoint carrier and post-freeze decoder-target boundary.

This module deliberately accepts no dense velocity input while constructing a
carrier.  A decoder target callback is permitted only after the carrier's
bytes/hash are frozen, which makes dense decoder supervision auditable without
letting it enter the T4d estimator.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


M24 = 24
BIN_SECONDS = 0.020
BLOCK_BINS = 5
LEAD_BINS = 2
MIN_DISPLACEMENT_CM = 0.50
EPOCHS = 35
CHECKPOINT_METRIC = "val_heldin/r2_mean"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _design(theta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(theta.size), np.cos(theta), np.sin(theta)])


@dataclass(frozen=True)
class EndpointCarrierResult:
    raw_feature: np.ndarray  # [units,4], exact zero in columns 2:4
    reach_rows: int
    design_rank: int
    design_condition: float
    access_log: tuple[str, ...]


def construct_t4d_carrier(*, reaches: Mapping[str, np.ndarray], neural_bins: np.ndarray) -> EndpointCarrierResult:
    """Fit `mean_rate = b+a*cos(theta)+c*sin(theta)` without dense velocity.

    `reaches` is an already endpoint-qualified M24 payload containing only
    `trial_index`, `theta_rad`, `left_bin`, and `right_bin`.  A row is emitted
    once per reach by averaging eligible five-bin neural rates; the two lead
    bins are containment-only.  The explicit accepted schema is a proof that
    dense decoder targets cannot be routed into this carrier function.
    """
    required = {"trial_index", "theta_rad", "left_bin", "right_bin"}
    _need(set(reaches) == required, f"T4d carrier schema must be exactly {sorted(required)}")
    neural = np.asarray(neural_bins, dtype=np.float64)
    _need(neural.ndim == 2 and np.isfinite(neural).all(), "neural bins must be finite [time,unit]")
    trial = np.asarray(reaches["trial_index"], dtype=np.int64).reshape(-1)
    theta = np.asarray(reaches["theta_rad"], dtype=np.float64).reshape(-1)
    lefts, rights = np.asarray(reaches["left_bin"], dtype=np.int64).reshape(-1), np.asarray(reaches["right_bin"], dtype=np.int64).reshape(-1)
    _need(trial.shape == theta.shape == lefts.shape == rights.shape, "reach arrays differ in length")
    select = trial < M24
    theta_rows: list[float] = []; response_rows: list[np.ndarray] = []
    for angle, left_bound, right_bound in zip(theta[select], lefts[select], rights[select]):
        if not np.isfinite(angle): continue
        rates = []
        for left in range(int(left_bound), int(right_bound) - BLOCK_BINS - LEAD_BINS + 1, BLOCK_BINS):
            right = left + BLOCK_BINS
            if left >= 0 and right + LEAD_BINS <= int(right_bound) and right <= neural.shape[0]:
                rates.append(neural[left:right].sum(axis=0) / (BLOCK_BINS * BIN_SECONDS))
        if rates:
            theta_rows.append(float(angle)); response_rows.append(np.mean(np.asarray(rates), axis=0))
    angles = np.asarray(theta_rows, dtype=np.float64)
    _need(angles.size >= 3, "T4d has fewer than three endpoint/neural reach rows")
    design = _design(angles); rank = int(np.linalg.matrix_rank(design)); _need(rank == 3, "T4d reach design is rank deficient")
    condition = float(np.linalg.cond(design)); _need(np.isfinite(condition), "T4d reach design condition is nonfinite")
    coef, *_ = np.linalg.lstsq(design, np.asarray(response_rows), rcond=None)
    raw = np.column_stack([coef[1], coef[2], np.zeros(neural.shape[1]), np.zeros(neural.shape[1])]).astype(np.float32)
    _need(np.array_equal(raw[:, 2:], np.zeros((neural.shape[1], 2), dtype=np.float32)), "T4d padded dimensions drifted from exact zero")
    return EndpointCarrierResult(raw, int(angles.size), rank, condition, ("trial_events", "cursor_position_endpoints", "spikes"))


@dataclass(frozen=True)
class T4dNormalizer:
    mean: np.ndarray
    std: np.ndarray
    fit_sessions: tuple[str, ...]


def fit_t4d_normalizer(features: Mapping[str, np.ndarray], *, inner_train_sessions: tuple[str, ...]) -> T4dNormalizer:
    """Fit the 4-D normalizer only on exactly the 13 inner-train sessions."""
    _need(len(inner_train_sessions) == 13 and set(features) == set(inner_train_sessions), "T4d normalizer requires exactly 13 inner-train sessions")
    stacked = np.concatenate([np.asarray(features[name], dtype=np.float32) for name in inner_train_sessions], axis=0)
    _need(stacked.ndim == 2 and stacked.shape[1] == 4 and np.array_equal(stacked[:, 2:], np.zeros_like(stacked[:, 2:])), "T4d normalizer input must retain exact zero pad")
    mean = np.array([stacked[:, 0].mean(), stacked[:, 1].mean(), 0.0, 0.0], dtype=np.float32)
    std = np.array([stacked[:, 0].std(), stacked[:, 1].std(), 1.0, 1.0], dtype=np.float32)
    _need(np.all(std[:2] > 0) and np.isfinite(std).all(), "T4d source normalizer has degenerate AC dimensions")
    return T4dNormalizer(mean, std, tuple(inner_train_sessions))


def apply_t4d_normalizer(feature: np.ndarray, normalizer: T4dNormalizer) -> np.ndarray:
    value = np.asarray(feature, dtype=np.float32); _need(value.ndim == 2 and value.shape[1] == 4, "invalid T4d feature")
    _need(np.array_equal(value[:, 2:], np.zeros_like(value[:, 2:])), "T4d source feature lost exact zero pad")
    output = ((value - normalizer.mean) / normalizer.std).astype(np.float32)
    output[:, 2:] = 0.0
    _need(np.array_equal(output[:, 2:], np.zeros_like(output[:, 2:])), "T4d normalized pad is not exact zero")
    return output


def outer_target_after_carrier_freeze(carrier: EndpointCarrierResult, decoder_target_loader: Callable[[], Any]) -> dict[str, Any]:
    """Allow decoder target acquisition only after immutable T4d carrier state."""
    before = _hash(carrier.raw_feature)
    target = decoder_target_loader()
    after = _hash(carrier.raw_feature)
    _need(before == after, "outer target read mutated frozen T4d carrier")
    return {"carrier_sha256_before_decoder_target": before, "carrier_sha256_after_decoder_target": after,
            "carrier_state_equal": True, "carrier_access_log": list(carrier.access_log),
            "decoder_target_loaded_after_carrier_freeze": True, "decoder_target": target}


def matched_training_contract(base: Mapping[str, Any]) -> dict[str, Any]:
    """Extract exact common fresh-system training fields for all three arms."""
    required = {"epochs", "optimizer", "learning_rate", "checkpoint_metric", "window_size", "query_start_trial", "session_window_budget", "seed"}
    _need(required.issubset(base), f"missing matched training fields: {sorted(required - set(base))}")
    _need(int(base["epochs"]) == EPOCHS and base["checkpoint_metric"] == CHECKPOINT_METRIC and int(base["seed"]) == 42, "frozen RT training spec drift")
    return {key: base[key] for key in sorted(required)}


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
