"""The completed-trial activity-pool recency laws (inference-side, label-free).

The frozen M2 student's B3S identity is

    identity = post_pool(concat(mean_over_pool(pre_pool(trial)), side))

so the completed-trial pool reaches the decoder ONLY through the mean of the
per-trial pre-pool features.  Every policy in this scan is therefore a
reweighting of one and the same per-trial feature stream, and the laws below
are implemented with the frozen encoder's OWN arithmetic:

* ``FrozenB3SUniformPool``  keeps one streaming ``push_trial`` state (the
  frozen accumulation order: support first, then completed trials in arrival
  order) and never evicts.  Its pooled mean is bitwise the mean the frozen
  ``forward_batch`` would produce over the same growing stack, because it IS
  that accumulation.
* ``FrozenB3SEmaPool``  holds ``E`` in the same pre-pool feature space,
  initialized at the M-budget support mean, and applies
  ``E <- alpha*E + beta*phi`` per completed trial in float32, with the
  completed-trial count carried separately from the mean law.
* UNIFORM_CAP30 (the sealed law) is NOT reimplemented here at all: the scan
  runs it through the sealed executors verbatim (see ``physical``).

No training, no gradients, no target access: the pools consume only the
completed trial's B3S activity view.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class MemoryLawError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MemoryLawError(message)


def trial_tensor(torch: Any, activity: np.ndarray) -> Any:
    """One B3S activity row [T, N] -> the frozen encoder's [1, T, N] input."""
    values = np.ascontiguousarray(np.asarray(activity, dtype=np.float32))
    _require(values.ndim == 2, "B3S activity row must be [T, N]")
    return torch.from_numpy(values).unsqueeze(0)


def float32_pair(alpha: float) -> tuple[np.float32, np.float32]:
    """The float32 (alpha, 1 - alpha) law pair used by every EMA update."""
    alpha32 = np.float32(float(alpha))
    return alpha32, np.float32(np.float32(1.0) - alpha32)


def ema_closed_form(
    support_features: np.ndarray, completed_features: Sequence[np.ndarray],
    alpha: float, *, dtype: np.dtype = np.float64,
) -> np.ndarray:
    """E_k = alpha^k * mean(support phi) + (1-alpha) * sum alpha^(k-i) phi_i.

    The reference closed form the recurrence must match (used by the no-data
    tests and by the receipts).  ``support_features`` is [M, ...] and each
    completed entry has the same trailing shape.
    """
    support = np.asarray(support_features, dtype=dtype)
    _require(support.ndim >= 1 and support.shape[0] >= 1, "support features needed")
    state = support.mean(axis=0)
    alpha_value = dtype.type(float(alpha)) if hasattr(dtype, "type") else float(alpha)
    for feature in completed_features:
        state = alpha_value * state + (1.0 - alpha_value) * np.asarray(feature, dtype=dtype)
    return state


class FrozenB3SUniformPool:
    """Uniform mean over support + every completed trial, no eviction ever."""

    family = "uniform_uncapped"

    def __init__(self, *, id_encoder: Any, support_activities: Sequence[np.ndarray],
                 channels: int, device: Any, dtype: Any, torch: Any) -> None:
        self.torch = torch
        self.encoder = id_encoder
        self.state = id_encoder.reset_stream(1, int(channels), device, dtype)
        for activity in support_activities:
            self.state = id_encoder.push_trial(
                self.state, trial_tensor(torch, activity),
            )
        self.support_count = int(self.state["trial_count"])
        self.completed_count = 0

    def identity(self, side_tensor: Any) -> Any:
        state = dict(self.state)
        state["side_features"] = side_tensor
        return self.encoder.finalize_identity(state)

    def commit(self, activity: np.ndarray) -> None:
        self.state = self.encoder.push_trial(
            self.state, trial_tensor(self.torch, activity),
        )
        self.completed_count += 1

    @property
    def pool_count(self) -> int:
        return int(self.state["trial_count"])

    def pooled_features_numpy(self) -> np.ndarray:
        value = self.state["sum_feat"] / self.state["trial_count"]
        return value.detach().numpy().astype(np.float32, copy=False)[0]

    def law_payload(self) -> dict[str, Any]:
        return {
            "policy": "UNIFORM_UNCAPPED",
            "family": "uniform",
            "eviction": "none_ever",
            "support_count": int(self.support_count),
            "completed_count": int(self.completed_count),
            "pool_count": int(self.pool_count),
        }


class FrozenB3SEmaPool:
    """Exponential recency on the pool mean in the frozen pre-pool space."""

    family = "ema"

    def __init__(self, *, id_encoder: Any, support_activities: Sequence[np.ndarray],
                 channels: int, alpha: float, device: Any, dtype: Any, torch: Any) -> None:
        _require(0.0 < float(alpha) < 1.0, "EMA alpha must lie in (0, 1)")
        self.torch = torch
        self.encoder = id_encoder
        self.channels = int(channels)
        self.device = device
        self.dtype = dtype
        self.alpha32, self.beta32 = float32_pair(alpha)
        state = id_encoder.reset_stream(1, int(channels), device, dtype)
        for activity in support_activities:
            state = id_encoder.push_trial(state, trial_tensor(torch, activity))
        support_count = int(state["trial_count"])
        _require(support_count >= 1, "EMA initialization needs the support mean")
        self.support_count = support_count
        # E0 = the M-budget support mean (the frozen mean arithmetic itself).
        self.E = state["sum_feat"] / state["trial_count"]
        self.completed_count = 0

    def _phi(self, activity: np.ndarray) -> Any:
        state = self.encoder.reset_stream(1, self.channels, self.device, self.dtype)
        state = self.encoder.push_trial(state, trial_tensor(self.torch, activity))
        return state["sum_feat"]

    def identity(self, side_tensor: Any) -> Any:
        # finalize divides the single weighted-mean row by one (exact), then
        # applies the frozen concat + post_pool.
        state = {"sum_feat": self.E, "trial_count": 1,
                 "side_features": side_tensor}
        return self.encoder.finalize_identity(state)

    def commit(self, activity: np.ndarray) -> None:
        phi = self._phi(activity)
        self.E = self.alpha32 * self.E + self.beta32 * phi
        self.completed_count += 1

    @property
    def pool_count(self) -> int:
        """The completed-trial count, reported separately from the mean law."""
        return int(self.completed_count)

    def pooled_features_numpy(self) -> np.ndarray:
        return self.E.detach().numpy().astype(np.float32, copy=False)[0]

    def law_payload(self) -> dict[str, Any]:
        alpha = float(self.alpha32)
        return {
            "policy": "EMA",
            "family": "ema",
            "alpha": alpha,
            "beta": float(self.beta32),
            "effective_pool": float(1.0 / (1.0 - alpha)),
            "initial_support_mean_weight_after_k_trials": float(
                pow(alpha, self.completed_count)),
            "support_count": int(self.support_count),
            "completed_count": int(self.completed_count),
            "count_role": "reported separately from the mean law",
        }
