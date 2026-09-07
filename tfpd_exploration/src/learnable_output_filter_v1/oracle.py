"""§8 adaptive-filter oracles on the raw B0 activity-only streams.

Both oracles read TRUE labels and are leakage diagnostics, never deployable
policies.  Every returned row is labelled ``target_label_leakage=True`` and the
word ``oracle`` is part of every label.

* noncoherent switch ceiling (§8.1): at every row the candidate gains are
  compared from the SAME frozen parent history — the selected fixed filter's
  own state trajectory — so each row's choice is independent (a per-row
  ceiling, not a coherent trajectory);
* coherent greedy gain oracle (§8.2): at row ``t`` the predeclared candidate
  gains are compared by the instantaneous true error (horizon = the current
  row, predeclared), the winner is emitted, and ONLY THEN does the actual
  filter state advance with the chosen gain.

The candidate gain grid is the frozen plan ``GAIN_GRID`` (which contains every
selectable F2 alpha, so the oracle never starts below the fixed filter).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import ladder, plan


class OracleError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


GAIN_HISTOGRAM_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def _grid_errors(state: np.ndarray, row: np.ndarray, target: np.ndarray,
                 gains: np.ndarray) -> np.ndarray:
    """Squared error of every candidate gain at one row ([T, G])."""
    delta = row - state                                    # [T, 2]
    residual = state - target                              # [T, 2]
    # ||residual + g*delta||^2 per gain g (quadratic in g, evaluated directly)
    base = np.einsum("tc,tc->t", residual, residual)
    linear = np.einsum("tc,tc->t", residual, delta)
    quad = np.einsum("tc,tc->t", delta, delta)
    return base[:, None] + 2.0 * gains[None, :] * linear[:, None] + gains[None, :] ** 2 * quad[:, None]


def _argmin_gain(errors: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Deterministic argmin over the grid (ties resolve to the lowest gain)."""
    return gains[np.argmin(errors, axis=1)]


def noncoherent_switch_ceiling(
    stream: ladder.SessionStream, *, parent: ladder.FilterSpec,
    gains: Sequence[float] = plan.GAIN_GRID,
) -> dict[str, Any]:
    """§8.1 per-row ceiling with the fixed filter's state as frozen parent."""
    grid = np.asarray(sorted({float(item) for item in gains}), dtype=np.float64)
    _require(grid.size >= 2, "oracle gain grid too small")
    batch = ladder.BatchView.from_stream(stream)
    raw = batch.raw
    pad_target = np.zeros_like(raw)
    for index, block in enumerate(stream.blocks):
        rows = int(batch.lengths[index])
        pad_target[index, :rows] = np.asarray(block.target, dtype=np.float64)
    parent_state = ladder.apply_linear(batch, parent)
    out = np.zeros_like(raw)
    chosen = np.zeros((raw.shape[0], raw.shape[1]), dtype=np.float64)
    out[:, 0] = raw[:, 0]
    chosen[:, 0] = 1.0
    for position in range(1, raw.shape[1]):
        errors = _grid_errors(parent_state[:, position - 1], raw[:, position],
                              pad_target[:, position], grid)
        gain = _argmin_gain(errors, grid)
        out[:, position] = (1.0 - gain)[:, None] * parent_state[:, position - 1] + gain[:, None] * raw[:, position]
        chosen[:, position] = gain
    blocks = ladder.unbatch(stream, batch, out)
    return {
        "oracle": "noncoherent_switch_ceiling",
        "section": "8.1",
        "target_label_leakage": True,
        "deployable": False,
        "parent_history": parent.label(),
        "gain_grid": [float(item) for item in grid],
        "blocks": blocks,
        "gains": chosen,
        "lengths": batch.lengths,
    }


def coherent_greedy_gain_oracle(
    stream: ladder.SessionStream, *, gains: Sequence[float] = plan.GAIN_GRID,
) -> dict[str, Any]:
    """§8.2 coherent clairvoyant policy (horizon = current row, predeclared)."""
    grid = np.asarray(sorted({float(item) for item in gains}), dtype=np.float64)
    _require(grid.size >= 2, "oracle gain grid too small")
    batch = ladder.BatchView.from_stream(stream)
    raw = batch.raw
    pad_target = np.zeros_like(raw)
    for index, block in enumerate(stream.blocks):
        rows = int(batch.lengths[index])
        pad_target[index, :rows] = np.asarray(block.target, dtype=np.float64)
    lengths = batch.lengths
    out = np.zeros_like(raw)
    chosen = np.zeros((raw.shape[0], raw.shape[1]), dtype=np.float64)
    state = raw[:, 0].copy()
    out[:, 0] = raw[:, 0]
    chosen[:, 0] = 1.0
    for position in range(1, raw.shape[1]):
        errors = _grid_errors(state, raw[:, position], pad_target[:, position], grid)
        gain = _argmin_gain(errors, grid)
        active = position < lengths
        state = np.where(
            active[:, None],
            (1.0 - gain)[:, None] * state + gain[:, None] * raw[:, position],
            state,
        )
        out[:, position] = state
        chosen[:, position] = np.where(active, gain, 0.0)
    blocks = ladder.unbatch(stream, batch, out)
    return {
        "oracle": "coherent_greedy_gain_oracle",
        "section": "8.2",
        "target_label_leakage": True,
        "deployable": False,
        "selection_horizon": "current_row",
        "gain_grid": [float(item) for item in grid],
        "blocks": blocks,
        "gains": chosen,
        "lengths": lengths,
    }


def gain_histogram(gains: np.ndarray, lengths: np.ndarray) -> dict[str, Any]:
    """Fixed-bin histogram over the emitted gains (padded rows excluded)."""
    collected: list[float] = []
    for index in range(gains.shape[0]):
        rows = int(lengths[index])
        collected.extend(float(item) for item in gains[index, :rows])
    values = np.asarray(collected, dtype=np.float64)
    counts, edges = np.histogram(values, bins=np.asarray(GAIN_HISTOGRAM_BINS))
    return {
        "bins": [float(item) for item in edges],
        "counts": [int(item) for item in counts],
        "n": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "min": float(values.min()) if values.size else None,
    }


def score_oracle(
    oracle: dict[str, Any], stream: ladder.SessionStream,
) -> dict[str, Any]:
    from . import metrics

    return {
        "oracle": oracle["oracle"],
        "target_label_leakage": True,
        "matrix_r2": metrics.matrix_r2(oracle["blocks"], stream),
        "filtered_prediction_sha256": metrics.prediction_sha256(oracle["blocks"], stream),
        "gain_histogram": gain_histogram(oracle["gains"], oracle["lengths"]),
    }
