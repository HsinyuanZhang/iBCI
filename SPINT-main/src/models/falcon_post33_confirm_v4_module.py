"""Phase-C SPINT metric module with non-logging exact-one outer evidence."""
from __future__ import annotations

from typing import Any

import torch

from src.models.components.spint_cached_deployment_v4 import (
    SpintCachedDeploymentAdapterV4,
)

from src.models.falcon_post33_confirm_v3_module import (
    M2Post33SourceOnlyFalconLitModule,
    validate_exact_outer_test,
)


class M2Post33ExactOuterFalconLitModuleV4(M2Post33SourceOnlyFalconLitModule):
    """Keep Phase-B source selection and retain one outer score for the evaluator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.outer_test_score_for_payload: float | None = None
        self.outer_test_runtime_metric_evidence: dict[str, object] | None = None
        self.deployment_cache_evidence: dict[str, object] | None = None
        self._deployment_profiler = None
        self._phase_c_selected_checkpoint_snapshot = None

    def enable_deployment_profiler(self, profiler: Any) -> None:
        self._deployment_profiler = profiler

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

    def on_validation_epoch_end(self) -> None:
        # The Phase-C source loader emits complete batches session by session,
        # while Lightning sanity checking runs only a bounded prefix (pinned
        # to two batches in the Phase-C plan).  That prefix cannot define an
        # exact-six selection metric.  Keep the forward-path probe, but clear
        # its partial state before the inherited strict all-six hook runs.
        if bool(getattr(getattr(self, "_trainer", None), "sanity_checking", False)):
            self.val_heldin_loss.reset()
            for metric in self.val_source_r2.values():
                metric.reset()
            self.val_outer_audit_r2.reset()
            return
        super().on_validation_epoch_end()

    def on_test_start(self) -> None:
        # Lightning has restored the evaluator snapshot at this point.  Verify
        # its identity and bytes before any datamodule calibration/query access.
        self._revalidate_phase_c_selected_checkpoint_snapshot()
        self._query_decode_invocations = 0
        self._query_batch_sizes: list[int] = []
        self._support_identity_computations = 0
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "claim_deployment_calibration"):
            raise RuntimeError("SPINT v4 requires one-shot datamodule calibration")
        calibration = datamodule.claim_deployment_calibration()
        support_cpu = calibration.get("support")
        digest = calibration.get("support_sha256")
        if not isinstance(support_cpu, torch.Tensor) or support_cpu.shape != (1, 33, 100, 96):
            raise ValueError("SPINT one-shot support has wrong shape")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("SPINT one-shot support digest is invalid")
        adapter = SpintCachedDeploymentAdapterV4(self.net)
        if self._deployment_profiler is None:
            support = support_cpu.to(self.device)
            self._cached_identity = adapter.compute_identity(support)
        else:
            with self._deployment_profiler.measure("support_calibration"):
                support = support_cpu.to(self.device)
                self._cached_identity = adapter.compute_identity(support)
        self._support_sha256 = digest
        self._support_identity_computations = 1

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if dataloader_idx != 0:
            raise ValueError("SPINT v4 exposes exactly one outer-only test loader")
        if len(batch) != 3:
            raise ValueError("SPINT streaming query batch must exclude calibration support")
        neural, behavior_target, session_names = batch
        session = self._single_session(session_names)
        if session != self.outer_session_name:
            raise ValueError(f"non-outer session entered SPINT test: {session}")
        adapter = SpintCachedDeploymentAdapterV4(self.net)
        if self._deployment_profiler is None:
            prediction = adapter.decode_with_identity(neural, self._cached_identity)
        else:
            with self._deployment_profiler.measure("streaming_inference"):
                prediction = adapter.decode_with_identity(neural, self._cached_identity)
        self._query_decode_invocations += 1
        self._query_batch_sizes.append(int(neural.shape[0]))
        if self.hparams.decode_last_timestep_only:
            prediction = prediction[:, -1:, :]
            behavior_target = behavior_target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            prediction = prediction / self.hparams.behavior_scaling_factor
        self.test_outer_r2[session].update(
            prediction.flatten(start_dim=0, end_dim=1),
            behavior_target.flatten(start_dim=0, end_dim=1),
        )

    def on_test_epoch_end(self) -> None:
        metric = self.test_outer_r2[self.outer_session_name]
        total = self._metric_total(metric)
        if total <= 2:
            raise ValueError("outer test session is missing or insufficient")
        value = float(metric.compute().detach().cpu().item())
        value = validate_exact_outer_test(
            outer_session=self.outer_session_name,
            values={self.outer_session_name: value},
            totals={self.outer_session_name: total},
        )
        self.outer_test_score_for_payload = value
        self.outer_test_runtime_metric_evidence = {
            "outer_session": self.outer_session_name,
            "metric_total": total,
            "metric_finite": True,
            "metric_value_disclosed": False,
            "non_outer_sessions_observed": 0,
        }
        if self._support_identity_computations != 1 or self._query_decode_invocations <= 0:
            raise RuntimeError("SPINT cached deployment invocation contract failed")
        if sum(self._query_batch_sizes) != total:
            raise RuntimeError("SPINT metric/query window count mismatch")
        self.deployment_cache_evidence = {
            "support_identity_computations": self._support_identity_computations,
            "query_decode_invocations": self._query_decode_invocations,
            "support_sha256": self._support_sha256,
            "cached_identity_shape": list(self._cached_identity.shape),
            "cached_identity_numel": int(self._cached_identity.numel()),
            "cached_identity_dtype": str(self._cached_identity.dtype),
            "cached_identity_bytes": int(
                self._cached_identity.numel() * self._cached_identity.element_size()
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
        if self._cached_identity is None:
            raise RuntimeError("SPINT cached identity is unavailable")
        return SpintCachedDeploymentAdapterV4(self.net).decode_with_identity(
            neural, self._cached_identity
        )
