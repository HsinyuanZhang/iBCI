"""Isolated H1 teacher wrapper with empty-session-safe metric aggregation.

``FalconLitModule`` keeps metric slots for every public held-in/held-out
session.  A clean nested H1 fit intentionally feeds only one inner-validation
session, so the generic epoch-end aggregation can stack empty held-out slots
and produce ``-inf``/integer-dtype failures.  This subclass keeps the exact
teacher state/forward interface (students can still restore it through
``FalconLitModule.load_from_checkpoint``) while aggregating only metrics that
received samples.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from src.models.falcon_module import FalconLitModule


def nonempty_metric_values(metrics: Mapping[str, Any], *, minimum_total: int = 3) -> list[tuple[str, torch.Tensor]]:
    """Return computed values only for metric states with enough samples."""

    values: list[tuple[str, torch.Tensor]] = []
    for name, metric in metrics.items():
        total = getattr(metric, "total", None)
        if total is None:
            continue
        try:
            enough = bool(total.detach().cpu().item() >= int(minimum_total))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            enough = bool(total >= int(minimum_total))
        if enough:
            value = metric.compute()
            if not torch.isfinite(value):
                raise RuntimeError(f"H1 metric {name!r} is non-finite despite having samples")
            values.append((str(name), value))
    return values


class H1TeacherLitModule(FalconLitModule):
    """Falcon-compatible H1 teacher with finite-session-only epoch metrics."""

    def on_train_epoch_end(self) -> None:
        values = nonempty_metric_values(self.train_r2)
        if not values:
            raise RuntimeError("H1 teacher train epoch had no non-empty session metrics")
        for session_name, value in values:
            self.log(f"train_{session_name}/r2", value, sync_dist=True, add_dataloader_idx=False)
            self.train_r2[session_name].reset()
        mean = torch.stack([value for _name, value in values]).mean()
        std = torch.stack([value for _name, value in values]).std(unbiased=False)
        self.log("train/r2_mean", mean, sync_dist=True, prog_bar=True)
        self.log("train/r2_std", std, sync_dist=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        # Only the one inner-validation held-in session is populated.  Empty
        # held-in and every held-out metric are deliberately skipped.
        heldin = nonempty_metric_values(self.val_heldin_r2)
        if len(heldin) != 1:
            raise RuntimeError(
                "H1 teacher expects exactly one non-empty inner-validation held-in metric, "
                f"got {[name for name, _value in heldin]}"
            )
        session_name, value = heldin[0]
        self.log(f"val_heldin_{session_name}/r2", value, sync_dist=True, add_dataloader_idx=False)
        self.val_heldin_r2[session_name].reset()
        mean = value
        self.log("val_heldin/r2_mean", mean, sync_dist=True, prog_bar=True)
        self.val_heldin_r2_mean_best(mean)
        self.log("val_heldin/r2_mean_best", self.val_heldin_r2_mean_best.compute(), sync_dist=True, prog_bar=True)
        for metric in self.val_heldin_r2.values():
            metric.reset()
        # No held-out loader is exposed by H1 clean nested LOSO.  Do not touch
        # or aggregate its empty metric slots.

