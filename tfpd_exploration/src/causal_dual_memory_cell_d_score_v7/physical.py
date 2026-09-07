"""Deferred V7 physical adapter for the authenticated V1 RuntimeFlags seam.

The V5 physical route deliberately imports its source-execution successor as a
wrapper module.  That wrapper has an explicitly named ``v1`` dependency, but
does not re-export V1's ``RuntimeFlags`` class.  V7 changes precisely that
factory lookup after both wrapper and dependency bytes have been bound into
the durable implementation closure.  It inherits every parser, sealed-SWA,
model, input, target chronology, evaluator, and independent-activity method
from the reviewed V5 backend.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import physical as v5physical

from . import plan, score


class PhysicalV7ScoreError(score.V7ScoreError):
    """Fail closed for the sole V7 flags-factory composition seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalV7ScoreError(message)


class V7ReviewedCDMScoreRuntime(v5physical.V5ReviewedCDMScoreRuntime):
    """V5 runtime with one exact V1 ``RuntimeFlags`` factory override."""

    _V1_FLAGS_MODULE_NAME = "src.causal_dual_memory_cell_d_v1.source_execute"
    _V1_FLAGS_RELATIVE = "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute.py"

    def _build_runtime_flags(self, source_execute: Any) -> Any:
        """Construct only the authenticated V1 flags object for V5 wrapping.

        ``source_execute`` has already passed V5's ordinary-import,
        root-relative path, and durable-closure checks.  V7 additionally
        proves that its direct ``v1`` dependency resolves to the exact V1
        source leaf whose digest appears in the same closure.  There is no
        probing fallback, no generic ``AttributeError`` recovery, and no
        mutation of the wrapper/module cache.
        """
        wrapper_name = "src.causal_dual_memory_cell_d_v1.source_execute_v5"
        _require(
            getattr(source_execute, "__name__", None) == wrapper_name
            and "RuntimeFlags" not in vars(source_execute),
            "V7 requires the exact V5 wrapper with no top-level RuntimeFlags",
        )
        dependency = getattr(source_execute, "v1", None)
        # The V5 wrapper's ordinary ``from . import source_execute as v1``
        # binding must be the same ordinary imported module object, not a
        # look-alike object carrying copied metadata/class names.
        try:
            canonical_dependency = importlib.import_module(self._V1_FLAGS_MODULE_NAME)
        except ImportError as error:
            raise PhysicalV7ScoreError("V7 V1 RuntimeFlags dependency import failed") from error
        _require(dependency is canonical_dependency, "V7 V5-wrapper V1 RuntimeFlags dependency object drift")
        expected_path = (self.root / self._V1_FLAGS_RELATIVE).absolute()
        module_path = getattr(dependency, "__file__", None)
        _require(
            getattr(dependency, "__name__", None) == self._V1_FLAGS_MODULE_NAME
            and isinstance(module_path, str)
            and Path(module_path).absolute() == expected_path,
            "V7 V5-wrapper V1 RuntimeFlags dependency module/path drift",
        )
        expected_sha = getattr(self, "_v5_runtime_sha256_by_path", {}).get(self._V1_FLAGS_RELATIVE)
        _require(isinstance(expected_sha, str), "V7 V1 RuntimeFlags closure digest is absent")
        try:
            actual_sha = plan._read_regular_no_follow(expected_path)
        except plan.V7PlanError as error:
            raise PhysicalV7ScoreError(str(error)) from error
        _require(actual_sha == expected_sha, "V7 V1 RuntimeFlags dependency bytes differ from durable closure")
        flags_type = getattr(dependency, "RuntimeFlags", None)
        _require(
            isinstance(flags_type, type)
            and flags_type.__name__ == "RuntimeFlags"
            and flags_type.__module__ == self._V1_FLAGS_MODULE_NAME,
            "V7 V1 RuntimeFlags type/module drift",
        )
        flags = flags_type(stage="score_prepare")
        _require(
            type(flags) is flags_type and getattr(flags, "stage", None) == "score_prepare",
            "V7 RuntimeFlags construction/type drift",
        )
        return flags


class PhysicalV7ScoreBackend(v1physical.PhysicalCDMDMatchedScoreBackend):
    """Inherited evaluator with one final held V6-lineage recheck.

    The original V1 backend remains responsible for input rederivation and
    all forward/metric mechanics.  V7 merely proves its immutable V6
    predecessor still names the same held receipt graph immediately before
    atomic score/terminal publication.
    """

    def revalidate(self, runtime: Any, *, root: Path, identity: plan.ScoreIdentity) -> None:
        super().revalidate(runtime, root=Path(root), identity=identity)
        score.validate_v6_runtime_flags_predecessor(Path(root))


def build_reviewed_physical_backend(
    *, root: Path, selected_device_profile: Mapping[str, object],
) -> PhysicalV7ScoreBackend:
    """Construct the inherited V5 backend with only the V7 runtime factory.

    The returned object has not opened an evaluation asset or a checkpoint and
    has not imported/initialized CUDA.  V1's physical backend invokes the
    typed V7 flag hook only after the shared lifecycle published its attempt.
    """
    profile = plan.v6plan.v5plan.v1plan.validate_compatible_device_profile(selected_device_profile)
    return PhysicalV7ScoreBackend(
        root=Path(root), selected_device_profile=profile,
        runtime_factory=lambda item_root, item_profile: V7ReviewedCDMScoreRuntime(
            root=item_root, selected_device_profile=item_profile,
        ),
    )


def _identity_profile(identity: plan.ScoreIdentity) -> Mapping[str, object]:
    payload = identity.payload()
    selected = payload.get("selected_device_profile")
    if not isinstance(selected, Mapping):
        raise PhysicalV7ScoreError("V7 identity selected-device profile drift")
    return selected


def _assert_v7_typed_capability(capability: object, identity: plan.ScoreIdentity) -> v1score.ExecutionCapability:
    """Keep the opaque capability check explicit and testable."""
    return v1score.require_execution_capability(capability, identity)


def execute_reviewed_physical_score(
    root: Path, *, identity: plan.ScoreIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Run one V7 score only behind the durable capability boundary."""
    _assert_v7_typed_capability(capability, identity)
    score.validate_selected_launch_environment(identity, environ)
    profile = plan.v6plan.v5plan.v1plan.validate_compatible_device_profile(_identity_profile(identity))
    backend = build_reviewed_physical_backend(root=Path(root), selected_device_profile=profile)
    preflight, authorization, pre_sha, auth_sha = score.load_durable_authority(Path(root), identity=identity)
    backend_preflight = backend.preflight(root=Path(root), identity=identity)
    if not isinstance(backend_preflight, Mapping) or any(backend_preflight.get(key) is not False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )):
        raise PhysicalV7ScoreError("V7 reviewed backend preflight is not target/model/CUDA-free")
    artifact = score.reserve_score_artifact(Path(root), identity=identity, capability=capability, environ=environ)
    return score.run_authorized_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization,
    )
