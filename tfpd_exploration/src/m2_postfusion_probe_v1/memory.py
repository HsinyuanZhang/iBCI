"""The post-fusion identity pools (inference-side, label-free).

The frozen M2 student's B3S identity is

    identity = post_pool(concat(mean_over_pool(pre_pool(trial)), side))

so the sealed pooled readout takes the mean of the per-trial pre-pool
features BEFORE the fusion MLP.  This module holds the placement variants:
the mean is taken AFTER the MLP, i.e. over the per-trial identities

    identity_i = post_pool(concat(pre_pool(trial_i), side))

produced by the frozen encoder's OWN single-trial finalize arithmetic
(``push_trial`` then ``finalize_identity`` with ``trial_count = 1``; the
division by one is exact, so ``identity_i`` IS
``post_pool(concat(phi(trial_i), T4))`` bitwise).

* ``per_trial_identity``    one B3S activity row -> the frozen per-trial
  identity tensor [1, N, W] (the law every pool consumes);
* ``PostFusionUniformIdentityPool``  a running float32 mean over member
  identities in arrival order (support first, then completed trials), with
  optional FIFO eviction of the oldest COMPLETED identity at capacity (the
  sealed CAP30 membership mirrored in identity space) or no eviction ever
  (the ``m2_memory_law_scan_v1`` UNIFORM_UNCAPPED accumulator law in
  identity space);
* ``PostFusionEmaIdentityPool``  ``E <- alpha*E + beta*identity_i`` in
  identity space, ``E0`` = the uniform mean of the support identities (the
  EMA_A090 precedent law, float32 (alpha, 1-alpha) pair).

No training, no gradients, no target access: the pools consume only the
completed trial's B3S activity view through the frozen encoder.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional, Sequence

import numpy as np


class PostFusionMemoryError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostFusionMemoryError(message)


def trial_tensor(torch: Any, activity: np.ndarray) -> Any:
    """One B3S activity row [T, N] -> the frozen encoder's [1, T, N] input."""
    values = np.ascontiguousarray(np.asarray(activity, dtype=np.float32))
    _require(values.ndim == 2, "B3S activity row must be [T, N]")
    return torch.from_numpy(values).unsqueeze(0)


def float32_pair(alpha: float) -> tuple[np.float32, np.float32]:
    """The float32 (alpha, 1 - alpha) law pair used by every EMA update."""
    alpha32 = np.float32(float(alpha))
    return alpha32, np.float32(np.float32(1.0) - alpha32)


def per_trial_identity(
    id_encoder: Any, torch: Any, activity: np.ndarray, side_tensor: Any,
    *, channels: int, device: Any, dtype: Any,
) -> Any:
    """identity_i = the frozen encoder's single-trial finalize arithmetic.

    ``push_trial`` accumulates one trial into a fresh stream state and
    ``finalize_identity`` divides the single-row sum by one (exact), concats
    the side features and applies the frozen post-pool MLP, so the result is
    bitwise ``post_pool(concat(pre_pool(trial_i), side))``.
    """
    state = id_encoder.reset_stream(1, int(channels), device, dtype)
    state = id_encoder.push_trial(state, trial_tensor(torch, activity))
    state["side_features"] = side_tensor
    return id_encoder.finalize_identity(state)


def sequential_mean(identities: Sequence[Any]) -> Any:
    """The from-scratch authority: sequential float32 sum in arrival order / n."""
    _require(len(identities) >= 1, "sequential mean needs at least one identity")
    total = identities[0]
    for identity in identities[1:]:
        total = total + identity
    return total / len(identities)


class PostFusionUniformIdentityPool:
    """Running uniform mean over member identities, optional FIFO capacity."""

    family = "uniform"

    def __init__(
        self, *, support_identities: Sequence[Any], capacity: Optional[int] = None,
    ) -> None:
        _require(len(support_identities) >= 1, "the pool needs the support identities")
        if capacity is not None:
            _require(int(capacity) > len(support_identities),
                     "capacity must exceed the support count (support is never evicted)")
        self.capacity = None if capacity is None else int(capacity)
        self.support_count = int(len(support_identities))
        self._members: deque[Any] = deque()
        total = None
        for identity in support_identities:
            total = identity if total is None else total + identity
            self._members.append(identity)
        self._sum = total
        self.completed_count = 0
        self.evictions = 0

    # -- the deployed readout ------------------------------------------------

    def deployed(self) -> Any:
        """The running mean sum / count (the frozen pooling's mean law)."""
        return self._sum / len(self._members)

    # -- the completed-trial stream ------------------------------------------

    def commit(self, identity: Any) -> None:
        self._members.append(identity)
        self._sum = self._sum + identity
        if self.capacity is not None and len(self._members) > self.capacity:
            # evict the OLDEST COMPLETED identity; support is never evicted
            _require(len(self._members) > self.support_count,
                     "capacity law would evict a support identity")
            evicted = self._members[self.support_count]
            del self._members[self.support_count]
            self._sum = self._sum - evicted
            self.evictions += 1
        self.completed_count += 1

    # -- audits and receipts -------------------------------------------------

    @property
    def pool_count(self) -> int:
        return int(len(self._members))

    def member_identities(self) -> list[Any]:
        return list(self._members)

    def from_scratch_mean(self) -> Any:
        """The audit authority over the CURRENT members (arrival order)."""
        return sequential_mean(self.member_identities())

    def law_payload(self) -> dict[str, Any]:
        return {
            "policy": "UNIFORM_CAP30" if self.capacity is not None else "UNIFORM_UNCAPPED",
            "family": "uniform",
            "domain": "post_fusion_identity",
            "capacity": self.capacity,
            "eviction": (
                "oldest completed identity evicted at capacity 30" if self.capacity is not None
                else "none, ever"
            ),
            "support_count": int(self.support_count),
            "completed_count": int(self.completed_count),
            "pool_count": int(self.pool_count),
            "evictions": int(self.evictions),
        }


class PostFusionEmaIdentityPool:
    """Exponential recency on the mean in the post-fusion identity space."""

    family = "ema"

    def __init__(
        self, *, support_identities: Sequence[Any], alpha: float,
    ) -> None:
        _require(0.0 < float(alpha) < 1.0, "EMA alpha must lie in (0, 1)")
        _require(len(support_identities) >= 1, "the EMA pool needs the support identities")
        self.alpha32, self.beta32 = float32_pair(alpha)
        self.alpha = float(self.alpha32)
        self.support_count = int(len(support_identities))
        # E0 = the uniform mean of the support identities (the same sequential
        # float32 mean law the uniform pool deploys before any commit).
        self.E = sequential_mean(list(support_identities))
        self.completed_count = 0

    def deployed(self) -> Any:
        return self.E

    def commit(self, identity: Any) -> None:
        self.E = self.alpha32 * self.E + self.beta32 * identity
        self.completed_count += 1

    def law_payload(self) -> dict[str, Any]:
        return {
            "policy": "EMA",
            "family": "ema",
            "domain": "post_fusion_identity",
            "alpha": float(self.alpha32),
            "beta": float(self.beta32),
            "effective_pool": float(1.0 / (1.0 - float(self.alpha32))),
            "initial_support_mean_weight_after_k_trials": float(
                pow(float(self.alpha32), self.completed_count)),
            "support_count": int(self.support_count),
            "completed_count": int(self.completed_count),
            "count_role": "reported separately from the mean law",
        }
