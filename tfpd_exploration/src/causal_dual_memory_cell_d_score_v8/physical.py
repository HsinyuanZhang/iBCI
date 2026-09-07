"""Deferred V8 physical adapter for the authenticated V1 helper seam.

V5 deliberately keeps its finalized-row executor in a wrapper module.  The
wrapper owns capture/consume semantics but intentionally does not re-export
V1's private variable-prefix helpers.  V8 authenticates that wrapper's exact
``v1_physical`` dependency and routes only the three inherited helper calls to
it; parser, model, evaluator, target chronology, and V5 executor selection
remain inherited unchanged.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v7 import physical as v7physical

from . import plan, score


class PhysicalV8ScoreError(score.V8ScoreError):
    """Fail closed for the sole V8 helper-module composition seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalV8ScoreError(message)


class V8ReviewedCDMScoreRuntime(v7physical.V7ReviewedCDMScoreRuntime):
    """V7 runtime retaining V5 executor while authenticating V1 helpers."""

    _V5_PHYSICAL_MODULE_NAME = "src.causal_dual_memory_cell_d_v1.source_execute_physical_v5"
    _V1_HELPER_MODULE_NAME = "src.causal_dual_memory_cell_d_v1.source_execute_physical"
    _V1_HELPER_RELATIVE = "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py"

    def _source_physical_helper_module(self, source_physical: Any) -> Any:
        """Return the one closure-bound V1 helper behind the V5 wrapper.

        This intentionally validates the wrapper *lacks* its own helper
        surfaces.  There is no attribute probing fallback: a wrapper that
        re-exports a look-alike helper, a different object, or mismatched
        bytes fails before the sealed checkpoint, CUDA, or held assets open.
        """
        _require(
            getattr(source_physical, "__name__", None) == self._V5_PHYSICAL_MODULE_NAME
            and "_variable_prefix_array_digest" not in vars(source_physical)
            and "ConcreteCellDFourGroupExecutor" not in vars(source_physical)
            and "_normalized_active_t4" not in vars(source_physical)
            and "_torch_variable_prefix_forward" not in vars(source_physical),
            "V8 requires exact V5 executor wrapper with no re-exported V1 helper surfaces",
        )
        helper = vars(source_physical).get("v1_physical")
        try:
            canonical = importlib.import_module(self._V1_HELPER_MODULE_NAME)
        except ImportError as error:
            raise PhysicalV8ScoreError("V8 V1 physical-helper dependency import failed") from error
        _require(helper is canonical, "V8 V5-wrapper V1 physical-helper object drift")
        expected_path = (self.root / self._V1_HELPER_RELATIVE).absolute()
        module_path = getattr(helper, "__file__", None)
        _require(
            getattr(helper, "__name__", None) == self._V1_HELPER_MODULE_NAME
            and isinstance(module_path, str)
            and Path(module_path).absolute() == expected_path,
            "V8 V1 physical-helper module/path drift",
        )
        expected_sha = getattr(self, "_v5_runtime_sha256_by_path", {}).get(self._V1_HELPER_RELATIVE)
        _require(isinstance(expected_sha, str), "V8 V1 physical-helper closure digest is absent")
        try:
            actual_sha = plan._read_regular_no_follow(expected_path)
        except plan.V8PlanError as error:
            raise PhysicalV8ScoreError(str(error)) from error
        _require(actual_sha == expected_sha, "V8 V1 physical-helper bytes differ from durable closure")
        digest = vars(helper).get("_variable_prefix_array_digest")
        executor_type = vars(helper).get("ConcreteCellDFourGroupExecutor")
        _require(
            callable(digest)
            and isinstance(executor_type, type)
            and executor_type.__name__ == "ConcreteCellDFourGroupExecutor"
            and executor_type.__module__ == self._V1_HELPER_MODULE_NAME
            and callable(vars(executor_type).get("_normalized_active_t4"))
            and callable(vars(executor_type).get("_torch_variable_prefix_forward")),
            "V8 V1 physical-helper callable/type surface drift",
        )
        return helper

    @staticmethod
    def _normalized_active_t4(state: v1physical._RuntimeState, raw_t4: Any) -> Any:
        """Dispatch normalized T4 through the authenticated V1 helper only."""
        wrapper_type = vars(state.source_physical).get("V5OneShotFinalizedRowExecutor")
        helper_type = vars(state.source_physical_helpers).get("ConcreteCellDFourGroupExecutor")
        _require(
            isinstance(wrapper_type, type)
            and type(state.executor) is wrapper_type
            and isinstance(helper_type, type),
            "V8 executor/helper role separation drift for normalized T4",
        )
        return helper_type._normalized_active_t4(state.executor_state, raw_t4)

    @staticmethod
    def _variable_prefix_forward(
        state: v1physical._RuntimeState, *, neural_windows: Any, activity_stack: Any, normalized_t4: Any,
        held_mask: Any, validity: Any, expected_prefix_length: int, expected_prefix_activity_sha256: str,
    ) -> tuple[Any, Mapping[str, object]]:
        """Dispatch group forward through V1 helper while retaining V5 executor."""
        wrapper_type = vars(state.source_physical).get("V5OneShotFinalizedRowExecutor")
        helper_type = vars(state.source_physical_helpers).get("ConcreteCellDFourGroupExecutor")
        _require(
            isinstance(wrapper_type, type)
            and type(state.executor) is wrapper_type
            and isinstance(helper_type, type),
            "V8 executor/helper role separation drift for variable-prefix forward",
        )
        return helper_type._torch_variable_prefix_forward(
            state.executor_state,
            neural_windows=neural_windows, activity_stack=activity_stack, normalized_t4=normalized_t4,
            held_mask=held_mask, validity=validity, expected_prefix_length=expected_prefix_length,
            expected_prefix_activity_sha256=expected_prefix_activity_sha256,
        )


class PhysicalV8ScoreBackend(v7physical.PhysicalV7ScoreBackend):
    """Inherited V5/V7 evaluator with final held V7-lineage revalidation."""

    def revalidate(self, runtime: Any, *, root: Path, identity: plan.ScoreIdentity) -> None:
        super().revalidate(runtime, root=Path(root), identity=identity)
        score.validate_v7_physical_helper_predecessor(Path(root))


def build_reviewed_physical_backend(
    *, root: Path, selected_device_profile: Mapping[str, object],
) -> PhysicalV8ScoreBackend:
    """Build the exact V5 executor route with only V8's helper selector."""
    profile = plan.v7plan.v6plan.v5plan.v1plan.validate_compatible_device_profile(selected_device_profile)
    return PhysicalV8ScoreBackend(
        root=Path(root), selected_device_profile=profile,
        runtime_factory=lambda item_root, item_profile: V8ReviewedCDMScoreRuntime(
            root=item_root, selected_device_profile=item_profile,
        ),
    )


def _identity_profile(identity: plan.ScoreIdentity) -> Mapping[str, object]:
    payload = identity.payload()
    selected = payload.get("selected_device_profile")
    if not isinstance(selected, Mapping):
        raise PhysicalV8ScoreError("V8 identity selected-device profile drift")
    return selected


def _assert_v8_typed_capability(capability: object, identity: plan.ScoreIdentity) -> v1score.ExecutionCapability:
    return v1score.require_execution_capability(capability, identity)


def execute_reviewed_physical_score(
    root: Path, *, identity: plan.ScoreIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Run one V8 score only behind the durable opaque capability boundary."""
    _assert_v8_typed_capability(capability, identity)
    score.validate_selected_launch_environment(identity, environ)
    profile = plan.v7plan.v6plan.v5plan.v1plan.validate_compatible_device_profile(_identity_profile(identity))
    backend = build_reviewed_physical_backend(root=Path(root), selected_device_profile=profile)
    preflight, authorization, pre_sha, auth_sha = score.load_durable_authority(Path(root), identity=identity)
    backend_preflight = backend.preflight(root=Path(root), identity=identity)
    if not isinstance(backend_preflight, Mapping) or any(backend_preflight.get(key) is not False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )):
        raise PhysicalV8ScoreError("V8 reviewed backend preflight is not target/model/CUDA-free")
    artifact = score.reserve_score_artifact(Path(root), identity=identity, capability=capability, environ=environ)
    return score.run_authorized_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization,
    )
