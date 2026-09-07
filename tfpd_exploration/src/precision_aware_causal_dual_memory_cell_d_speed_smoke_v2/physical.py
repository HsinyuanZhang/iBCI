"""Deferred V2 diagnostic composition over the exact V1 physical runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.causal_dual_memory_cell_d_score_v1 import score as shared_score
from src.precision_aware_causal_dual_memory_cell_d_speed_smoke_v1 import physical as v1_physical

from . import plan, score


class PhysicalSpeedSmokeV2Error(score.SpeedSmokeV2Error):
    """Fail closed for the sole V2 diagnostic adapter seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalSpeedSmokeV2Error(message)


class PhysicalPrecisionSpeedSmokeV2Backend:
    """Reuse V1's parser/runtime; retain the comparison before V2 validation.

    The V1 delegate still owns environment gating, strict sealed loading,
    physical input parsing, eager B128 execution, O1/O2 execution, causal
    state transitions, OOM fallback, resource observation, and final held-FD
    revalidation.  V2 only retains its returned numeric comparison so a later
    policy failure cannot erase it from the immutable receipt graph.
    """

    def __init__(self, *, root: Path, identity: score.SpeedSmokeV2Identity) -> None:
        self._root = Path(root).absolute()
        self._identity = identity
        self._v1_identity = identity.v1_identity()
        self._delegate = v1_physical.build_reviewed_physical_backend(
            root=self._root, identity=self._v1_identity,
        )
        self._runtime: Any | None = None
        self._last_diagnostic_comparison: Mapping[str, object] | None = None

    def _gate(self, *, root: Path, identity: score.SpeedSmokeV2Identity) -> None:
        _require(identity is self._identity, "speed-smoke V2 physical identity object drift")
        _require(Path(root).absolute() == self._root, "speed-smoke V2 physical root drift")
        score._require_live_predecessors(self._root, identity)
        plan.validate_selected_launch_environment()

    def preflight(self, *, root: Path, identity: score.SpeedSmokeV2Identity) -> Mapping[str, object]:
        self._gate(root=Path(root), identity=identity)
        return self._delegate.preflight(root=self._root, identity=self._v1_identity)

    def prepare(self, *, root: Path, identity: score.SpeedSmokeV2Identity) -> Any:
        self._gate(root=Path(root), identity=identity)
        if self._runtime is not None:
            raise PhysicalSpeedSmokeV2Error("speed-smoke V2 runtime prepared more than once")
        runtime = self._delegate.prepare(root=self._root, identity=self._v1_identity)
        _require(isinstance(runtime, v1_physical._ComparativeSpeedPrecisionV2Runtime),
                 "speed-smoke V2 must reuse exact V1 comparative runtime")
        self._runtime = runtime
        return runtime

    def materialize_inputs(
        self, runtime: Any, *, identity: score.SpeedSmokeV2Identity, evaluation_authority: Any,
    ) -> Any:
        self._gate(root=self._root, identity=identity)
        _require(runtime is self._runtime, "speed-smoke V2 materialize runtime identity drift")
        return self._delegate.materialize_inputs(
            runtime, identity=self._v1_identity, evaluation_authority=evaluation_authority,
        )

    def score_budget(
        self, runtime: Any, *, budget: int, input_authority_sha256: str, identity: score.SpeedSmokeV2Identity,
    ) -> tuple[Mapping[str, object], ...]:
        self._gate(root=self._root, identity=identity)
        _require(runtime is self._runtime, "speed-smoke V2 budget runtime identity drift")
        expected = [item for item in plan.SMOKE_ROWS if item.budget == budget]
        _require(len(expected) == 1 and budget == 4, "speed-smoke V2 budget is outside exact M4 matrix")
        witness = expected[0]
        state = runtime._require_state()
        session = state.sessions.get((witness.surface, witness.session))
        _require(session is not None, "speed-smoke V2 selected materialized session is absent")
        try:
            baseline, optimized, raw_comparison, speed_evidence = runtime.score_eager_baseline_and_optimized(
                session=session, budget=budget,
            )
            # Capture immediately: policy validation below may reject, but
            # subsequent failure publication must disclose this live evidence.
            self._last_diagnostic_comparison = score.diagnostic_numeric_comparison(raw_comparison)
            resources = runtime._resources()
            return (score.build_diagnostic_cell(
                identity=identity, input_authority_sha256=input_authority_sha256,
                baseline_session_row=baseline, optimized_session_row=optimized,
                raw_comparison=raw_comparison, speed_evidence=speed_evidence,
                resources=resources, witness=witness,
            ),)
        except PhysicalSpeedSmokeV2Error:
            raise
        except Exception as error:
            raise PhysicalSpeedSmokeV2Error("speed-smoke V2 eager/optimized comparison failed") from error

    def revalidate(self, runtime: Any, *, root: Path, identity: score.SpeedSmokeV2Identity) -> None:
        self._gate(root=Path(root), identity=identity)
        _require(runtime is self._runtime, "speed-smoke V2 final runtime identity drift")
        self._delegate.revalidate(runtime, root=self._root, identity=self._v1_identity)
        self._gate(root=Path(root), identity=identity)

    def failure_progress(self, runtime: Any | None) -> Mapping[str, object]:
        base = self._delegate.failure_progress(runtime)
        if not isinstance(base, Mapping):
            raise PhysicalSpeedSmokeV2Error("speed-smoke V2 inherited failure progress drift")
        return {**dict(base), "diagnostic_numeric_comparison": self._last_diagnostic_comparison}

    def close(self, runtime: Any | None) -> None:
        try:
            self._delegate.close(runtime)
        finally:
            self._runtime = None


def build_reviewed_physical_backend(
    *, root: Path, identity: score.SpeedSmokeV2Identity,
) -> PhysicalPrecisionSpeedSmokeV2Backend:
    if not isinstance(identity, score.SpeedSmokeV2Identity):
        raise PhysicalSpeedSmokeV2Error("speed-smoke V2 physical identity type drift")
    plan.validate_selected_launch_environment(
        {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", **plan.CANONICAL_SOURCE_ROOTS},
    )
    return PhysicalPrecisionSpeedSmokeV2Backend(root=Path(root), identity=identity)


def execute_reviewed_physical_speed_smoke_v2(
    root: Path, *, identity: score.SpeedSmokeV2Identity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Future root-only execution adapter; the public CLI never calls this."""
    score._validate_mutating_launch_environment(environ)
    score._require_live_predecessors(Path(root), identity)
    backend = build_reviewed_physical_backend(root=Path(root), identity=identity)
    preflight, authorization, pre_sha, auth_sha = score.load_durable_authority(Path(root), identity=identity)
    checked = backend.preflight(root=Path(root), identity=identity)
    _require(isinstance(checked, Mapping) and all(checked.get(key) is False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )), "speed-smoke V2 backend preflight is not target/model/CUDA-free")
    artifact = score.reserve_score_artifact(Path(root), identity=identity, capability=capability, environ=environ)
    return score.run_authorized_speed_smoke_v2_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization, environ=environ,
    )
