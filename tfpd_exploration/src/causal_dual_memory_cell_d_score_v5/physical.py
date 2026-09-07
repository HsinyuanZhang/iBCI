"""Deferred V5 physical composition over the reviewed V1 scorer seam.

Only the executor loader, dual-memory constructor, commit operation, and
independent transition receipt codec differ from V1.  Parser, held-data,
sealed-SWA, model-forward, metric, resource, and lifecycle machinery remain
inherited; no global/module substitution is used.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import score as v1score

from . import plan, score


class PhysicalV5ScoreError(score.V5ScoreError):
    """Fail closed for the V5 loader/transition-only physical seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalV5ScoreError(message)


class V5ReviewedCDMScoreRuntime(v1physical.ReviewedCDMScoreRuntime):
    """V1 runtime with exact V5 executor and independent target transitions."""

    _V5_RUNTIME_MODULES = {
        "src.causal_dual_memory_cell_d_v1.source_execute_v5": (
            "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v5.py"
        ),
        "src.causal_dual_memory_cell_d_v1.source_execute_physical_v5": (
            "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v5.py"
        ),
    }

    def prepare(self, *, identity: plan.ScoreIdentity) -> None:
        """Bind the two V5-only runtime modules to the durable closure first.

        ``importlib.import_module`` is deliberately ordinary package loading,
        not a private module replacement.  The extra source-file identity and
        byte check below prevents an already-imported ambient module from
        silently satisfying the V5 executor seam with bytes other than the
        exact closure that the durable identity authenticated.
        """
        payload = identity.payload()
        closure = payload.get("closure")
        rows = closure.get("paths") if isinstance(closure, Mapping) else None
        if not isinstance(rows, list):
            raise PhysicalV5ScoreError("V5 runtime identity closure rows are absent")
        by_path: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
                raise PhysicalV5ScoreError("V5 runtime identity closure row drift")
            path, digest = row["path"], row["sha256"]
            if not isinstance(path, str) or not isinstance(digest, str):
                raise PhysicalV5ScoreError("V5 runtime identity closure value drift")
            by_path[path] = digest
        if set(self._V5_RUNTIME_MODULES.values()) - set(by_path):
            raise PhysicalV5ScoreError("V5 runtime modules are absent from implementation closure")
        self._v5_runtime_sha256_by_path = by_path
        try:
            super().prepare(identity=identity)
        except BaseException:
            self._v5_runtime_sha256_by_path = {}
            raise

    def _closure_bound_runtime_module(self, module: Any, *, module_name: str, relative: str) -> Any:
        """Require ordinary import resolution to name the exact reviewed leaf."""
        if self._V5_RUNTIME_MODULES.get(module_name) != relative:
            raise PhysicalV5ScoreError("V5 runtime module route mapping drift")
        module_path = getattr(module, "__file__", None)
        expected = (self.root / relative).absolute()
        if (
            getattr(module, "__name__", None) != module_name
            or not isinstance(module_path, str)
            or Path(module_path).absolute() != expected
        ):
            raise PhysicalV5ScoreError("V5 runtime module import identity/path drift")
        expected_sha = getattr(self, "_v5_runtime_sha256_by_path", {}).get(relative)
        if not isinstance(expected_sha, str):
            raise PhysicalV5ScoreError("V5 runtime module closure binding is absent")
        try:
            actual_sha = plan._read_regular_no_follow(expected)
        except plan.V5PlanError as error:
            raise PhysicalV5ScoreError(str(error)) from error
        if actual_sha != expected_sha:
            raise PhysicalV5ScoreError("V5 runtime module bytes differ from durable closure")
        return module

    def _source_execution_runtime_modules(self) -> tuple[Any, Any]:
        source_execute = self._closure_bound_runtime_module(
            importlib.import_module("src.causal_dual_memory_cell_d_v1.source_execute_v5"),
            module_name="src.causal_dual_memory_cell_d_v1.source_execute_v5",
            relative=self._V5_RUNTIME_MODULES["src.causal_dual_memory_cell_d_v1.source_execute_v5"],
        )
        source_physical = self._closure_bound_runtime_module(
            importlib.import_module("src.causal_dual_memory_cell_d_v1.source_execute_physical_v5"),
            module_name="src.causal_dual_memory_cell_d_v1.source_execute_physical_v5",
            relative=self._V5_RUNTIME_MODULES["src.causal_dual_memory_cell_d_v1.source_execute_physical_v5"],
        )
        if (
            getattr(source_execute, "WORKORDER_SHA256", None) != "503fa0276b3bb32fd31b9da51dca961d1214dfef0a0d7c40cd5e8bec3e87cb9b"
            or getattr(source_execute, "SOURCE_GATE_ROOT_RELATIVE", None)
            != "tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v5"
            or not hasattr(source_physical, "V5OneShotFinalizedRowExecutor")
        ):
            raise PhysicalV5ScoreError("V5 source-execution loader/runtime module identity drift")
        return source_execute, source_physical

    def _build_source_executor(self, source_physical: Any) -> Any:
        executor = source_physical.V5OneShotFinalizedRowExecutor(root=self.root)
        if type(executor) is not source_physical.V5OneShotFinalizedRowExecutor:
            raise PhysicalV5ScoreError("V5 source-execution executor type drift")
        return executor

    @staticmethod
    def _make_dual_memory(*, core: Any, activity: Any, carrier: Any) -> Any:
        memory = core.IndependentActivityCausalDualMemory(activity=activity, carrier=carrier)
        if type(memory) is not core.IndependentActivityCausalDualMemory:
            raise PhysicalV5ScoreError("V5 target route requires exact independent activity memory")
        return memory

    @staticmethod
    def _normalized_active_t4(state: v1physical._RuntimeState, raw_t4: Any) -> Any:
        executor_type = getattr(state.source_physical, "V5OneShotFinalizedRowExecutor", None)
        if type(state.executor) is not executor_type:
            raise PhysicalV5ScoreError("V5 normalized-side executor identity drift")
        return executor_type._normalized_active_t4(state.executor_state, raw_t4)

    @staticmethod
    def _variable_prefix_forward(
        state: v1physical._RuntimeState, *, neural_windows: Any, activity_stack: Any, normalized_t4: Any,
        held_mask: Any, validity: Any, expected_prefix_length: int, expected_prefix_activity_sha256: str,
    ) -> tuple[Any, Mapping[str, object]]:
        executor_type = getattr(state.source_physical, "V5OneShotFinalizedRowExecutor", None)
        if type(state.executor) is not executor_type:
            raise PhysicalV5ScoreError("V5 group-forward executor identity drift")
        return executor_type._torch_variable_prefix_forward(
            state.executor_state,
            neural_windows=neural_windows, activity_stack=activity_stack, normalized_t4=normalized_t4,
            held_mask=held_mask, validity=validity, expected_prefix_length=expected_prefix_length,
            expected_prefix_activity_sha256=expected_prefix_activity_sha256,
        )

    def _commit_completed_transition(
        self, *, memory: Any, pending: Any, trial_id: str,
    ) -> v1physical.CompletedTrialTransition:
        state = self._require_state()
        core = state.modules["core"]
        if type(memory) is not core.IndependentActivityCausalDualMemory:
            raise PhysicalV5ScoreError("V5 independent transition memory type drift")
        outcome = memory.commit_independent(pending)
        try:
            core.validate_independent_activity_outcome_payload(outcome.payload())
        except Exception as error:
            raise PhysicalV5ScoreError("V5 independent transition payload validation failed") from error
        return v1physical.CompletedTrialTransition(
            trial_id=trial_id,
            activity_transition_committed=outcome.activity_transition_committed,
            activity_fifo_changed=outcome.activity_fifo_changed,
            carrier_transition_committed=outcome.carrier_transition_committed,
            activity_rejection_reason=self._reason_value(outcome.activity_rejection_reason),
            carrier_rejection_reason=self._reason_value(outcome.carrier_rejection_reason),
            state_before_sha256=outcome.state_before_sha256,
            state_after_sha256=outcome.state_after_sha256,
            activity_before_sha256=outcome.activity_before_sha256,
            activity_after_sha256=outcome.activity_after_sha256,
            carrier_before_sha256=outcome.carrier_before_sha256,
            carrier_after_sha256=outcome.carrier_after_sha256,
        )

    def _make_session_score(
        self,
        *,
        session: v1physical.PreparedEvaluationSession,
        budget: int,
        system: str,
        n_windows: int,
        r2: float,
        prediction_sha256: str,
        model_state_before_sha256: str,
        model_state_after_sha256: str,
        initial: Mapping[str, object],
        full_system_forward_count: int,
        group_forward_count: int,
        transitions: Sequence[v1physical.CompletedTrialTransition],
        target_last_bin_sha256: str,
        valid_mask_sha256: str,
        sealed_model_load_proof_sha256: str,
    ) -> score.IndependentSessionScore:
        carrier_committed = sum(item.carrier_transition_committed for item in transitions)
        carrier_rejections: dict[str, int] = {}
        activity_rejections: dict[str, int] = {}
        for item in transitions:
            if item.carrier_rejection_reason is not None:
                carrier_rejections[item.carrier_rejection_reason] = carrier_rejections.get(item.carrier_rejection_reason, 0) + 1
            if item.activity_rejection_reason is not None:
                activity_rejections[item.activity_rejection_reason] = activity_rejections.get(item.activity_rejection_reason, 0) + 1
        base = v1score.SessionScore(
            session=session.session, n_windows=n_windows, r2=r2, prediction_sha256=prediction_sha256,
            input_record_sha256=session.input_record_sha256,
            model_state_before_sha256=model_state_before_sha256,
            model_state_after_sha256=model_state_after_sha256,
            initial_carrier_sha256=str(initial["initial_carrier_sha256"]),
            group_assignment_sha256=str(initial["group_assignment_sha256"]),
            group_valid_mask_sha256=str(initial["group_valid_mask_sha256"]),
            initial_activity_sha256=str(initial["initial_activity_sha256"]),
            support_trial_ids_sha256=v1physical._object_sha(list(session.support_trial_ids[budget])),
            raw_m30_t4_axis_proof_sha256=v1physical._object_sha(session.raw_m30_axis_proof.payload()),
            sealed_normalizer_sha256=v1physical.plan.SEALED_OLS_NORMALIZER_SHA256,
            sealed_model_load_proof_sha256=sealed_model_load_proof_sha256,
            target_last_bin_sha256=target_last_bin_sha256, valid_mask_sha256=valid_mask_sha256,
            valid_last_bin_count=n_windows, activity_fifo_capacity=plan.FIFO_CAPACITY[budget],
            accepted_updates=carrier_committed * plan.GROUP_COUNT if system == plan.SYSTEM_CDMD else 0,
            rejected_updates={} if system == plan.SYSTEM_SEALED else {
                key: value * plan.GROUP_COUNT for key, value in carrier_rejections.items()
            },
            group_forward_count=group_forward_count if system == plan.SYSTEM_CDMD else 0,
            full_system_forward_count=full_system_forward_count, dropout_calls=0, target_label_state_uses=0,
        )
        return score.IndependentSessionScore(
            base=base,
            transition_records=tuple(item.payload() for item in transitions) if system == plan.SYSTEM_CDMD else (),
            activity_transition_committed_count=sum(item.activity_transition_committed for item in transitions),
            activity_fifo_changed_count=sum(item.activity_fifo_changed for item in transitions),
            carrier_transition_committed_count=carrier_committed,
            activity_rejection_counts=activity_rejections,
            carrier_rejection_counts=carrier_rejections,
        )

    def _make_cell_evidence(
        self,
        *, surface: str, budget: int, system: str, input_authority_sha256: str,
        sessions: Sequence[Any], resources: Mapping[str, object],
    ) -> score.CellEvidence:
        if not all(isinstance(item, score.IndependentSessionScore) for item in sessions):
            raise PhysicalV5ScoreError("V5 session evidence codec type drift")
        return score.CellEvidence(
            surface=surface, budget=budget, system=system,
            input_authority_sha256=input_authority_sha256,
            model_swa_sha256=plan.v1plan.SEALED_CELL_D_SWA_SHA256,
            sessions=tuple(sessions), resources=resources,
        )


def build_reviewed_physical_backend(
    *, root: Path, selected_device_profile: Mapping[str, object],
) -> v1physical.PhysicalCDMDMatchedScoreBackend:
    """Build the sole V5 physical backend without target/model/CUDA action."""
    profile = plan.v1plan.validate_compatible_device_profile(selected_device_profile)
    return v1physical.PhysicalCDMDMatchedScoreBackend(
        root=Path(root), selected_device_profile=profile,
        runtime_factory=lambda item_root, item_profile: V5ReviewedCDMScoreRuntime(
            root=item_root, selected_device_profile=item_profile,
        ),
    )


def execute_reviewed_physical_score(
    root: Path, *, identity: plan.ScoreIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Reviewed V5-only physical launch adapter; public CLI cannot call it."""
    score.validate_selected_launch_environment(identity, environ)
    profile = plan.v1plan.validate_compatible_device_profile(_identity_profile(identity))
    backend = build_reviewed_physical_backend(root=Path(root), selected_device_profile=profile)
    preflight, authorization, pre_sha, auth_sha = score.load_durable_authority(Path(root), identity=identity)
    backend_preflight = backend.preflight(root=Path(root), identity=identity)
    if any(backend_preflight.get(key) is not False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )):
        raise PhysicalV5ScoreError("V5 reviewed backend preflight is not target/model/CUDA-free")
    artifact = score.reserve_score_artifact(Path(root), identity=identity, capability=capability, environ=environ)
    return score.run_authorized_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization,
    )


def _identity_profile(identity: plan.ScoreIdentity) -> Mapping[str, object]:
    payload = identity.payload()
    selected = payload.get("selected_device_profile")
    if not isinstance(selected, Mapping):
        raise PhysicalV5ScoreError("V5 identity selected-device profile drift")
    return selected
