"""Source-only SPINT Lightning module for M2 post-33 Phase-B v3.

The historical Falcon module aggregates all seven M2 held-in sessions and
turns the deliberately absent outer session into ``-inf``.  This versioned
module makes the six-source validation and single-outer test contracts
explicit and fail-closed.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torchmetrics.regression import R2Score

from src.models.falcon_module import FalconLitModule


PROTOCOL_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
FOLDS = {
    0: "ses-2020-10-19-Run1",
    1: "ses-2020-10-19-Run2",
    2: "ses-2020-10-20-Run1",
    3: "ses-2020-10-20-Run2",
    4: "ses-2020-10-27-Run1",
    5: "ses-2020-10-27-Run2",
    6: "ses-2020-10-28-Run1",
}


def fold_sessions(loso_fold: int) -> tuple[tuple[str, ...], str]:
    if isinstance(loso_fold, bool) or loso_fold not in FOLDS:
        raise ValueError("loso_fold must be an integer in [0, 6]")
    return tuple(session for fold, session in FOLDS.items() if fold != loso_fold), FOLDS[loso_fold]


def aggregate_exact_source_metrics(
    *,
    expected_sources: Sequence[str],
    outer_session: str,
    values: Mapping[str, float],
    totals: Mapping[str, int],
    outer_total: int,
) -> float:
    """Return an unweighted six-session mean or fail on any scope defect."""
    expected = tuple(expected_sources)
    if len(expected) != 6 or len(set(expected)) != 6 or outer_session in expected:
        raise ValueError("expected_sources must be exactly six unique non-outer sessions")
    if set(values) != set(expected) or set(totals) != set(expected):
        raise ValueError("validation must contain exactly the six source sessions")
    if isinstance(outer_total, bool) or outer_total != 0:
        raise ValueError("outer validation total must be exactly zero")
    converted = []
    for session in expected:
        total = totals[session]
        value = values[session]
        if isinstance(total, bool) or not isinstance(total, int) or total <= 2:
            raise ValueError(f"source session {session} has insufficient total {total!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"source session {session} has non-finite R2 {value!r}")
        converted.append(float(value))
    result = sum(converted) / len(converted)
    if not math.isfinite(result):
        raise ValueError("source equal-session mean is non-finite")
    return result


def validate_exact_outer_test(
    *, outer_session: str, values: Mapping[str, float], totals: Mapping[str, int]
) -> float:
    """Accept only the one real outer metric; no empty source placeholders."""
    if set(values) != {outer_session} or set(totals) != {outer_session}:
        raise ValueError("test must contain exactly the unique outer session")
    total = totals[outer_session]
    value = values[outer_session]
    if isinstance(total, bool) or not isinstance(total, int) or total <= 2:
        raise ValueError("outer test total must be an integer > 2")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("outer test R2 must be finite")
    return float(value)


class M2Post33SourceOnlyFalconLitModule(FalconLitModule):
    """Exact-six source validation and exact-one outer test module."""

    def __init__(self, *, loso_fold: int, **kwargs: Any) -> None:
        if kwargs.get("task") != "m2":
            raise ValueError(f"{PROTOCOL_ID} requires task='m2'")
        super().__init__(**kwargs)
        # Keep the checkpoint loadable by the historical FalconLitModule used
        # by the frozen streaming wrapper; the fold is runtime protocol state,
        # not a constructor argument of that historical class.
        self.hparams.pop("loso_fold", None)
        sources, outer = fold_sessions(loso_fold)
        self.protocol_loso_fold = loso_fold
        self.source_session_names = sources
        self.outer_session_name = outer

        # Replace the seven-session historical metric tables with exact scope.
        self.train_r2 = nn.ModuleDict(
            {name: R2Score(multioutput="variance_weighted") for name in sources}
        )
        self.val_source_r2 = nn.ModuleDict(
            {name: R2Score(multioutput="variance_weighted") for name in sources}
        )
        self.val_outer_audit_r2 = R2Score(multioutput="variance_weighted")
        self.test_outer_r2 = nn.ModuleDict(
            {outer: R2Score(multioutput="variance_weighted")}
        )
        self.source_selector_records: list[dict[str, Any]] = []

    @staticmethod
    def _single_session(session_names: Sequence[str]) -> str:
        unique = set(session_names)
        if len(unique) != 1:
            raise ValueError("all batch samples must belong to one session")
        return next(iter(unique))

    @staticmethod
    def _metric_total(metric: R2Score) -> int:
        return int(metric.total.detach().cpu().item())

    def on_train_start(self) -> None:
        self.val_heldin_loss.reset()
        self.val_heldin_r2_mean_best.reset()
        for metric in self.val_source_r2.values():
            metric.reset()
        self.val_outer_audit_r2.reset()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if dataloader_idx != 0:
            raise ValueError("v3 validation exposes exactly one source-only dataloader")
        loss, prediction, target, session_names = self.model_step(batch)
        session = self._single_session(session_names)
        if session not in self.val_source_r2 or session == self.outer_session_name:
            raise ValueError(f"non-source session entered validation: {session}")
        self.val_heldin_loss(loss)
        self.val_source_r2[session].update(
            prediction.flatten(start_dim=0, end_dim=1),
            target.flatten(start_dim=0, end_dim=1),
        )
        self.log(
            "val_source/loss",
            self.val_heldin_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def on_validation_epoch_end(self) -> None:
        values: dict[str, float] = {}
        totals: dict[str, int] = {}
        for session in self.source_session_names:
            metric = self.val_source_r2[session]
            total = self._metric_total(metric)
            totals[session] = total
            if total <= 2:
                raise ValueError(f"missing/insufficient source validation session {session}")
            value = float(metric.compute().detach().cpu().item())
            values[session] = value
            self.log(f"val_source_{session}/r2", value, sync_dist=True, add_dataloader_idx=False)
        outer_total = self._metric_total(self.val_outer_audit_r2)
        mean = aggregate_exact_source_metrics(
            expected_sources=self.source_session_names,
            outer_session=self.outer_session_name,
            values=values,
            totals=totals,
            outer_total=outer_total,
        )
        self.log("val_source/r2_equal_session_mean", mean, sync_dist=True, prog_bar=True)
        self.val_heldin_r2_mean_best(torch.tensor(mean, device=self.device))
        self.log(
            "val_source/r2_equal_session_mean_best",
            self.val_heldin_r2_mean_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )
        sanity = bool(getattr(getattr(self, "_trainer", None), "sanity_checking", False))
        if not sanity:
            self.source_selector_records.append(
                {
                    "epoch": int(self.current_epoch),
                    "metric_name": "val_source/r2_equal_session_mean",
                    "metric_value": mean,
                    "metric_scope": "exact_six_outer_train_source_sessions_only",
                    "source_sessions": list(self.source_session_names),
                    "source_totals": totals,
                    "outer_session": self.outer_session_name,
                    "outer_total": outer_total,
                }
            )
        for metric in self.val_source_r2.values():
            metric.reset()
        self.val_outer_audit_r2.reset()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if dataloader_idx != 0:
            raise ValueError("v3 test exposes exactly one outer-only dataloader")
        loss, prediction, target, session_names = self.model_step(batch)
        session = self._single_session(session_names)
        if session != self.outer_session_name:
            raise ValueError(f"source/extra session entered outer test: {session}")
        self.test_heldin_loss(loss)
        self.test_outer_r2[session].update(
            prediction.flatten(start_dim=0, end_dim=1),
            target.flatten(start_dim=0, end_dim=1),
        )

    def on_test_epoch_end(self) -> None:
        metric = self.test_outer_r2[self.outer_session_name]
        total = self._metric_total(metric)
        if total <= 2:
            raise ValueError("outer test session is missing or insufficient")
        value = float(metric.compute().detach().cpu().item())
        result = validate_exact_outer_test(
            outer_session=self.outer_session_name,
            values={self.outer_session_name: value},
            totals={self.outer_session_name: total},
        )
        self.log("test_outer/r2", result, sync_dist=True, prog_bar=True)
        metric.reset()
