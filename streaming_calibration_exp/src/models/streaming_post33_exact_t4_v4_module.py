"""Exact-six source selection and exact-one outer test for Phase-C T4."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torchmetrics.regression import R2Score

from src.models.streaming_calibration_module import StreamingCalibrationLitModule
from src.models.components.streaming_cached_deployment_v4 import T4CachedDeploymentAdapterV4
from src.utils.post33_paired_teacher_phase_c_v4 import resolve_phase_c_paired_spint_teacher


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


def exact_six_equal_session_mean(
    *,
    expected_sources: Sequence[str],
    outer_session: str,
    values: Mapping[str, float],
    totals: Mapping[str, int],
    outer_total: int,
) -> float:
    sources = tuple(expected_sources)
    if len(sources) != 6 or len(set(sources)) != 6 or outer_session in sources:
        raise ValueError("expected exactly six unique non-outer source sessions")
    if set(values) != set(sources) or set(totals) != set(sources):
        raise ValueError("T4 validation must contain exactly the ordered six sources")
    if isinstance(outer_total, bool) or outer_total != 0:
        raise ValueError("T4 outer validation total must be exactly zero")
    ordered_values = []
    for session in sources:
        total = totals[session]
        value = values[session]
        if isinstance(total, bool) or not isinstance(total, int) or total <= 2:
            raise ValueError(f"T4 source {session} total must be integer >2")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"T4 source {session} R2 must be finite")
        ordered_values.append(float(value))
    result = sum(ordered_values) / 6
    if not math.isfinite(result):
        raise ValueError("T4 source equal-session mean is non-finite")
    return result


def exact_one_outer_metric(
    *, outer_session: str, values: Mapping[str, float], totals: Mapping[str, int]
) -> float:
    if set(values) != {outer_session} or set(totals) != {outer_session}:
        raise ValueError("T4 test must contain exactly the unique outer session")
    total = totals[outer_session]
    value = values[outer_session]
    if isinstance(total, bool) or not isinstance(total, int) or total <= 2:
        raise ValueError("T4 outer total must be integer >2")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("T4 outer R2 must be finite")
    return float(value)


class M2Post33ExactPairedT4LitModuleV4(StreamingCalibrationLitModule):
    """Fail closed on any source/test cardinality drift."""

    def __init__(
        self,
        *,
        paired_spint_completion_receipt: str,
        phase_c_t4_owner_path: str,
        phase_c_t4_owner_token: str,
        loso_fold: int,
        seed: int,
        **kwargs: Any,
    ) -> None:
        if not paired_spint_completion_receipt or paired_spint_completion_receipt == "???":
            raise ValueError("Phase-C paired SPINT completion receipt is mandatory")
        if "teacher_ckpt_path" in kwargs:
            raise ValueError("manual teacher_ckpt_path is forbidden")
        if not phase_c_t4_owner_path or phase_c_t4_owner_path == "???":
            raise ValueError("Phase-C T4 ownership path is mandatory")
        if not phase_c_t4_owner_token or phase_c_t4_owner_token == "???":
            raise ValueError("Phase-C T4 ownership token is mandatory")
        teacher, binding = resolve_phase_c_paired_spint_teacher(
            paired_spint_completion_receipt,
            loso_fold=loso_fold,
            seed=seed,
            phase_c_t4_owner_path=phase_c_t4_owner_path,
            phase_c_t4_owner_token=phase_c_t4_owner_token,
        )
        # The historical base class does a delayed ``load_from_checkpoint``
        # in ``setup``.  Give it a retained-FD snapshot rather than the
        # mutable sealed SPINT pathname it just validated.
        from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
            create_pinned_file_snapshot,
            release_pinned_file_snapshot,
        )

        teacher_snapshot = create_pinned_file_snapshot(
            teacher,
            expected_metadata=binding["selected_checkpoint"],
            label="paired SPINT teacher",
        )
        try:
            super().__init__(
                teacher_ckpt_path=str(teacher_snapshot.trainer_checkpoint_path), **kwargs
            )
        except BaseException:
            release_pinned_file_snapshot(teacher_snapshot)
            raise
        self.paired_spint_completion_receipt = paired_spint_completion_receipt
        self.phase_c_t4_owner_path = phase_c_t4_owner_path
        self.phase_c_t4_owner_token = phase_c_t4_owner_token
        self._phase_c_teacher_binding = binding
        self._phase_c_teacher_snapshot = teacher_snapshot
        self._phase_c_teacher_snapshot_released = False
        self.protocol_loso_fold = loso_fold
        self.protocol_seed = seed
        sources, outer = fold_sessions(loso_fold)
        self.source_session_names = sources
        self.outer_session_name = outer
        self.val_source_r2 = nn.ModuleDict(
            {name: R2Score(multioutput="variance_weighted") for name in sources}
        )
        self.val_outer_audit_r2 = R2Score(multioutput="variance_weighted")
        self.test_outer_r2 = nn.ModuleDict(
            {outer: R2Score(multioutput="variance_weighted")}
        )
        self.source_selector_records: list[dict[str, Any]] = []
        self.outer_test_runtime_metric_evidence: dict[str, Any] | None = None
        self.outer_test_score_for_payload: float | None = None
        self.deployment_cache_evidence: dict[str, Any] | None = None
        self._deployment_profiler = None
        self._phase_c_selected_checkpoint_snapshot = None

    def _revalidate_phase_c_teacher_binding(self) -> None:
        from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
            validate_pinned_file_origin,
            validate_pinned_file_snapshot,
        )

        if self._phase_c_teacher_snapshot_released:
            raise RuntimeError("Phase-C paired SPINT teacher snapshot was released too early")
        teacher, _ = resolve_phase_c_paired_spint_teacher(
            self.paired_spint_completion_receipt,
            loso_fold=self.protocol_loso_fold,
            seed=self.protocol_seed,
            phase_c_t4_owner_path=self.phase_c_t4_owner_path,
            phase_c_t4_owner_token=self.phase_c_t4_owner_token,
            expected_binding=self._phase_c_teacher_binding,
        )
        if teacher != self._phase_c_teacher_snapshot.canonical_selected_checkpoint:
            raise ValueError("Phase-C paired SPINT teacher path changed before checkpoint load")
        if self._teacher_ckpt_path != str(
            self._phase_c_teacher_snapshot.trainer_checkpoint_path
        ):
            raise ValueError("Phase-C paired SPINT teacher restore path is not pinned")
        validate_pinned_file_snapshot(
            self._phase_c_teacher_snapshot, label="paired SPINT teacher"
        )
        validate_pinned_file_origin(
            self._phase_c_teacher_snapshot, label="paired SPINT teacher"
        )

    def _release_phase_c_teacher_snapshot(self) -> None:
        if self._phase_c_teacher_snapshot_released:
            return
        from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
            release_pinned_file_snapshot,
        )

        release_pinned_file_snapshot(self._phase_c_teacher_snapshot)
        self._phase_c_teacher_snapshot_released = True

    def setup(self, stage: str) -> None:
        # ``StreamingCalibrationLitModule.setup`` later dereferences the
        # teacher checkpoint.  Validate the frozen same-root binding on both
        # sides of that call so a receipt/checkpoint swap cannot silently load
        # a different teacher through a delayed generic trainer lifecycle.
        self._revalidate_phase_c_teacher_binding()
        super().setup(stage)
        self._revalidate_phase_c_teacher_binding()

    def enable_deployment_profiler(self, profiler: Any) -> None:
        self._deployment_profiler = profiler

    def on_fit_end(self) -> None:
        # Source selection/completion follows the fit lifecycle.  Verify the
        # canonical paired source at that terminal boundary, but retain the
        # pinned FD: legacy ``train.py`` intentionally calls ``setup('fit')``
        # after ``Trainer.fit`` to obtain its encoder-cost profile.
        self._revalidate_phase_c_teacher_binding()

    def finalize_phase_c_teacher_snapshot(self) -> None:
        """Release only after the wrapper's post-fit legacy profile has run."""
        try:
            self._revalidate_phase_c_teacher_binding()
        finally:
            self._release_phase_c_teacher_snapshot()

    def bind_phase_c_selected_checkpoint_snapshot(self, snapshot: Any) -> None:
        """Bind the evaluator's private selected-checkpoint restore artifact."""
        from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
            SelectedCheckpointSnapshot,
            validate_selected_checkpoint_snapshot,
        )

        if self._phase_c_selected_checkpoint_snapshot is not None:
            raise RuntimeError("Phase-C selected checkpoint snapshot was already bound")
        if not isinstance(snapshot, SelectedCheckpointSnapshot):
            raise TypeError("Phase-C selected checkpoint snapshot binding is invalid")
        validate_selected_checkpoint_snapshot(snapshot)
        self._phase_c_selected_checkpoint_snapshot = snapshot

    def _revalidate_phase_c_selected_checkpoint_snapshot(self) -> None:
        from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
            validate_selected_checkpoint_snapshot,
        )

        snapshot = self._phase_c_selected_checkpoint_snapshot
        if snapshot is None:
            raise RuntimeError("Phase-C selected checkpoint snapshot is not bound")
        validate_selected_checkpoint_snapshot(snapshot)

    def on_test_start(self) -> None:
        # Lightning restores the selected *student* checkpoint for
        # ``Trainer.test(..., ckpt_path=...)`` after its setup lifecycle.
        # Re-check the sealed same-root teacher receipt/checkpoint binding
        # before claiming any outer calibration/query state, so an altered
        # paired teacher is rejected before the evaluator can expose it to the
        # cached deployment path.  The lifecycle callback then proves the
        # restored student decoder remains bit-exact.
        self._revalidate_phase_c_selected_checkpoint_snapshot()
        self._revalidate_phase_c_teacher_binding()
        self._query_decode_invocations = 0
        self._query_batch_sizes: list[int] = []
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "claim_deployment_calibration"):
            raise RuntimeError("T4 v4 requires one-shot datamodule calibration")
        calibration = datamodule.claim_deployment_calibration()
        support_cpu = calibration.get("support")
        side_cpu = calibration.get("side_features")
        digest = calibration.get("support_and_side_sha256")
        if (
            not isinstance(support_cpu, torch.Tensor)
            or support_cpu.shape != (1, 33, 100, 96)
            or not isinstance(side_cpu, torch.Tensor)
            or side_cpu.shape != (1, 96, 4)
        ):
            raise ValueError("T4 one-shot calibration state has wrong shape")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("T4 one-shot calibration digest is invalid")
        assert self.student is not None
        adapter = T4CachedDeploymentAdapterV4(self.student)
        if self._deployment_profiler is None:
            support = support_cpu.to(self.device)
            side = side_cpu.to(self.device)
            self._cached_outer_identity = adapter.compute_identity(support, side)
        else:
            with self._deployment_profiler.measure("support_calibration"):
                support = support_cpu.to(self.device)
                side = side_cpu.to(self.device)
                self._cached_outer_identity = adapter.compute_identity(support, side)
        self._support_and_side_sha256 = digest
        self._support_identity_computations = 1

    @staticmethod
    def _single_session(session_names: Sequence[str]) -> str:
        unique = set(session_names)
        if len(unique) != 1:
            raise ValueError("all T4 batch samples must belong to exactly one session")
        return next(iter(unique))

    @staticmethod
    def _metric_total(metric: R2Score) -> int:
        return int(metric.total.detach().cpu().item())

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if dataloader_idx != 0:
            raise ValueError("T4 v4 exposes exactly one source-only validation loader")
        out = self.model_step(batch)
        session = self._single_session(out["session_name"])
        if session not in self.val_source_r2 or session == self.outer_session_name:
            raise ValueError(f"non-source session entered T4 validation: {session}")
        self.val_heldin_loss(out["loss"])
        self.val_source_r2[session].update(
            out["behavior_pred"].flatten(start_dim=0, end_dim=1),
            out["behavior_target"].flatten(start_dim=0, end_dim=1),
        )
        self.val_identity_mse(out["identity_mse"])
        self.val_prediction_distill_mse(out["prediction_distill_mse"])
        self.log("val_source/loss", self.val_heldin_loss, on_epoch=True, add_dataloader_idx=False)

    def on_validation_epoch_end(self) -> None:
        # The Phase-C source loader emits complete batches session by session.
        # Lightning sanity checking consumes only a bounded prefix, so it
        # cannot satisfy this module's exact-six selection contract.  Keep the
        # probe for import/forward-path diagnostics, but clear all state it
        # touched rather than treating it as a scientific validation epoch.
        if bool(getattr(getattr(self, "_trainer", None), "sanity_checking", False)):
            self.val_heldin_loss.reset()
            self.val_identity_mse.reset()
            self.val_prediction_distill_mse.reset()
            for metric in self.val_source_r2.values():
                metric.reset()
            self.val_outer_audit_r2.reset()
            return
        values: dict[str, float] = {}
        totals: dict[str, int] = {}
        for session in self.source_session_names:
            metric = self.val_source_r2[session]
            total = self._metric_total(metric)
            totals[session] = total
            if total <= 2:
                raise ValueError(f"missing/empty T4 source validation session {session}")
            value = float(metric.compute().detach().cpu().item())
            values[session] = value
            self.log(f"val_source_{session}/r2", value, add_dataloader_idx=False)
        outer_total = self._metric_total(self.val_outer_audit_r2)
        mean = exact_six_equal_session_mean(
            expected_sources=self.source_session_names,
            outer_session=self.outer_session_name,
            values=values,
            totals=totals,
            outer_total=outer_total,
        )
        self.log("val_source/r2_equal_session_mean", mean, prog_bar=True)
        self.val_heldin_r2_mean_best(torch.tensor(mean, device=self.device))
        self.log(
            "val_source/r2_equal_session_mean_best",
            self.val_heldin_r2_mean_best.compute(),
            prog_bar=True,
        )
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
            raise ValueError("T4 v4 exposes exactly one outer-only test loader")
        if len(batch) != 3:
            raise ValueError("T4 streaming query batch must exclude support and descriptors")
        neural, behavior_target, session_names = batch
        session = self._single_session(session_names)
        if session != self.outer_session_name:
            raise ValueError(f"non-outer session entered T4 test: {session}")
        assert self.student is not None
        adapter = T4CachedDeploymentAdapterV4(self.student)
        if self._deployment_profiler is None:
            prediction = adapter.decode_with_identity(neural, self._cached_outer_identity)
        else:
            with self._deployment_profiler.measure("streaming_inference"):
                prediction = adapter.decode_with_identity(neural, self._cached_outer_identity)
        self._query_decode_invocations += 1
        self._query_batch_sizes.append(int(neural.shape[0]))
        prediction, behavior_target = self._slice_last_timestep(prediction, behavior_target)
        self.test_outer_r2[session].update(
            prediction.flatten(start_dim=0, end_dim=1),
            behavior_target.flatten(start_dim=0, end_dim=1),
        )

    def on_test_epoch_end(self) -> None:
        metric = self.test_outer_r2[self.outer_session_name]
        total = self._metric_total(metric)
        if total <= 2:
            raise ValueError("T4 outer test is missing/empty")
        value = float(metric.compute().detach().cpu().item())
        exact_one_outer_metric(
            outer_session=self.outer_session_name,
            values={self.outer_session_name: value},
            totals={self.outer_session_name: total},
        )
        # The value itself remains scorer/commitment-only.  This runtime fact is
        # safe to bind into the result receipt without disclosing the endpoint.
        self.outer_test_runtime_metric_evidence = {
            "outer_session": self.outer_session_name,
            "metric_total": total,
            "metric_finite": True,
            "metric_value_disclosed": False,
            "non_outer_sessions_observed": 0,
        }
        self.outer_test_score_for_payload = value
        if self._support_identity_computations != 1 or self._query_decode_invocations <= 0:
            raise RuntimeError("T4 cached deployment invocation contract failed")
        if sum(self._query_batch_sizes) != total:
            raise RuntimeError("T4 metric/query window count mismatch")
        self.deployment_cache_evidence = {
            "support_identity_computations": self._support_identity_computations,
            "query_decode_invocations": self._query_decode_invocations,
            "support_and_side_sha256": self._support_and_side_sha256,
            "cached_identity_shape": list(self._cached_outer_identity.shape),
            "cached_identity_numel": int(self._cached_outer_identity.numel()),
            "cached_identity_dtype": str(self._cached_outer_identity.dtype),
            "cached_identity_bytes": int(
                self._cached_outer_identity.numel()
                * self._cached_outer_identity.element_size()
            ),
            "descriptor_state_bytes_after_finalize": 0,
            "raw_support_required_for_online_decode": False,
            "dedicated_calibration_tensors_released_after_finalize": True,
            "offline_evaluator_retains_outer_dataset": True,
            "support_or_descriptor_in_streaming_query_batch": False,
            "query_batch_sizes": list(self._query_batch_sizes),
            "query_window_count": sum(self._query_batch_sizes),
        }
        metric.reset()

    def cached_online_forward(self, neural: torch.Tensor) -> torch.Tensor:
        if self._cached_outer_identity is None or self.student is None:
            raise RuntimeError("T4 cached identity is unavailable")
        return T4CachedDeploymentAdapterV4(self.student).decode_with_identity(
            neural, self._cached_outer_identity
        )
