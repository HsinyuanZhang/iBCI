"""C3 reliability feature, constant control, and matched model widening.

C3 changes only the B3S side-input width from four to five.  The first four
columns are the exact C2 posterior-mean carrier.  The fifth column is either
the source-normalized angular reliability (``real``) or the source mean,
which is exactly zero after normalization (``constant``).

The widened encoder is initialized from an already strict-loaded C2 model.
Every old parameter is copied exactly; only the newly introduced q column in
the first post-pool affine layer is initialized to exact zero.  This gives the
two C3 arms an identical initial state and makes the added degree of freedom
explicit rather than silently rebuilding the whole model from a different RNG
stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from budget_matched_posterior_cal_aug_v1.posterior import (
    PosteriorContractError,
    array_sha256,
)
from budget_matched_posterior_cal_aug_v1.training import (
    BudgetMatchedPosteriorDataset,
    SessionPosteriorFeatures,
    TRAINING_BUDGET_CYCLE,
)


class C3Arm(str, Enum):
    CONSTANT = "constant"
    REAL = "real"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PosteriorContractError(message)


@dataclass(frozen=True)
class ReliabilityNormalizer:
    mean: float
    std: float
    row_count: int
    rows_sha256: str
    per_budget_row_count: Mapping[int, int]
    per_budget_rows_sha256: Mapping[int, str]

    def __post_init__(self) -> None:
        _require(np.isfinite(self.mean), "q mean must be finite")
        _require(np.isfinite(self.std) and self.std > 0.0, "q std must be positive")
        _require(type(self.row_count) is int and self.row_count > 0, "q row count invalid")
        budgets = set(TRAINING_BUDGET_CYCLE)
        _require(set(self.per_budget_row_count) == budgets, "q normalizer budget counts drift")
        _require(set(self.per_budget_rows_sha256) == budgets, "q normalizer budget digests drift")
        _require(
            sum(int(value) for value in self.per_budget_row_count.values()) == self.row_count,
            "q normalizer row total drift",
        )

    def normalize(self, values: object) -> np.ndarray:
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        _require(array.ndim == 1 and np.isfinite(array).all(), "q rows must be finite [units]")
        return np.ascontiguousarray((array - self.mean) / self.std, dtype=np.float32)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "budget_matched_posterior_cal_aug_c3_q_normalizer_v1",
            "source_only": True,
            "equal_budget_weight": True,
            "budget_order": [4, 10, 30],
            "mean_float64": self.mean,
            "std_float64": self.std,
            "row_count": self.row_count,
            "rows_sha256": self.rows_sha256,
            "per_budget_row_count": {
                str(key): int(self.per_budget_row_count[key]) for key in (4, 10, 30)
            },
            "per_budget_rows_sha256": {
                str(key): str(self.per_budget_rows_sha256[key]) for key in (4, 10, 30)
            },
            "constant_raw_value": self.mean,
            "constant_normalized_value": 0.0,
        }


def fit_source_reliability_normalizer(
    session_features: Mapping[str, SessionPosteriorFeatures],
    *,
    source_roster: Sequence[str],
) -> ReliabilityNormalizer:
    """Fit one source-only q z-score with equal M4/M10/M30 row weights."""

    roster = tuple(str(value) for value in source_roster)
    _require(len(roster) > 0 and len(roster) == len(set(roster)), "source roster invalid")
    _require(set(roster) == set(session_features), "q source roster/features mismatch")
    chunks: list[np.ndarray] = []
    counts: dict[int, int] = {}
    digests: dict[int, str] = {}
    for budget in (4, 10, 30):
        rows = np.ascontiguousarray(np.concatenate([
            np.asarray(session_features[name].angular_reliability_by_budget[budget], dtype=np.float64)
            for name in roster
        ]), dtype=np.float64)
        _require(rows.ndim == 1 and np.isfinite(rows).all(), f"M{budget} q rows invalid")
        chunks.append(rows)
        counts[budget] = int(rows.size)
        digests[budget] = array_sha256(rows)
    _require(len(set(counts.values())) == 1, "q normalizer budgets are not equally weighted")
    all_rows = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float64)
    std = float(all_rows.std(ddof=0))
    _require(np.isfinite(std) and std > 1.0e-12, "q source std is degenerate")
    return ReliabilityNormalizer(
        mean=float(all_rows.mean()),
        std=std,
        row_count=int(all_rows.size),
        rows_sha256=array_sha256(all_rows),
        per_budget_row_count=counts,
        per_budget_rows_sha256=digests,
    )


class BudgetMatchedReliabilityDataset:
    """C2 dataset view plus one source-normalized q column."""

    def __init__(
        self,
        base_dataset,
        session_features: Mapping[str, SessionPosteriorFeatures],
        reliability_normalizer: ReliabilityNormalizer,
        *,
        arm: C3Arm | str,
    ) -> None:
        self.c2_view = BudgetMatchedPosteriorDataset(base_dataset, session_features)
        self.session_features = dict(session_features)
        self.reliability_normalizer = reliability_normalizer
        try:
            self.arm = C3Arm(arm)
        except ValueError as error:
            raise PosteriorContractError(f"unknown C3 arm: {arm!r}") from error

    def __len__(self) -> int:
        return len(self.c2_view)

    def __getattr__(self, name: str):
        if name in {"c2_view", "session_features", "reliability_normalizer", "arm"}:
            raise AttributeError(name)
        return getattr(self.c2_view, name)

    def __getitem__(self, tagged_index):
        rewritten = self.c2_view[tagged_index]
        session = rewritten[3]
        budget = int(tagged_index[1])
        side4 = rewritten[4]
        _require(session in self.session_features, "C3 session is not authorized")
        q_raw = self.session_features[session].angular_reliability_by_budget[budget]
        q = self.reliability_normalizer.normalize(q_raw)
        if self.arm is C3Arm.CONSTANT:
            q = np.zeros_like(q, dtype=np.float32)

        import torch

        q_tensor = torch.from_numpy(np.ascontiguousarray(q[:, None], dtype=np.float32))
        q_tensor = q_tensor.to(dtype=side4.dtype, device=side4.device)
        side5 = torch.cat((side4, q_tensor), dim=-1)
        _require(tuple(side5.shape) == (q.shape[0], 5), "C3 side tensor shape drift")
        output = list(rewritten)
        output[4] = side5
        return tuple(output) if isinstance(rewritten, tuple) else output


def widen_cell_d_for_reliability(model, build_encoder):
    """Replace only a side_dim=4 B3S encoder with its side_dim=5 counterpart."""

    import torch

    old = model.id_encoder
    _require(getattr(old, "variant", None) == "B3S", "C3 requires B3S encoder")
    _require(getattr(old, "side_dim", None) == 4, "C3 requires a four-column C2 encoder")
    new = build_encoder(
        "B3S",
        window_size=50,
        trial_length=100,
        id_hidden_dim=128,
        hidden_dim=64,
        side_dim=5,
    )
    old_state = old.state_dict()
    new_state = new.state_dict()
    _require(set(old_state) == set(new_state), "C3 encoder state-key drift")
    widened_key = "post_pool.0.weight"
    with torch.no_grad():
        for key in new_state:
            source = old_state[key]
            target = new_state[key]
            if key == widened_key:
                _require(
                    target.ndim == 2
                    and source.ndim == 2
                    and target.shape[0] == source.shape[0]
                    and target.shape[1] == source.shape[1] + 1,
                    "C3 q-column widening shape drift",
                )
                target[:, : source.shape[1]].copy_(source)
                target[:, source.shape[1] :].zero_()
            else:
                _require(tuple(target.shape) == tuple(source.shape), f"C3 unexpected shape drift: {key}")
                target.copy_(source)
    new.load_state_dict(new_state, strict=True)
    model.id_encoder = new
    _require(
        bool(torch.count_nonzero(model.id_encoder.state_dict()[widened_key][:, -1]).item()) is False,
        "C3 q initialization is not exact zero",
    )
    model._budget_matched_posterior_c3 = {
        "side_dim": 5,
        "q_column": 4,
        "q_initialization": "exact_zero_extension_of_strict_loaded_C2",
    }
    return model
