"""Deferred V6 physical adapter over the accepted V5 scorer backend.

V6 changes only output-root sequencing.  It constructs the exact reviewed V5
parser/model/independent-activity backend and delegates all science and target
handling to it; the V6 lifecycle supplies the held-reserved-root validator
before it permits backend preparation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import physical as v5physical

from . import plan, score


class PhysicalV6ScoreError(score.V6ScoreError):
    """Fail closed for the narrow V6 launch adapter."""


def _identity_profile(identity: plan.ScoreIdentity) -> Mapping[str, object]:
    payload = identity.payload()
    selected = payload.get("selected_device_profile")
    if not isinstance(selected, Mapping):
        raise PhysicalV6ScoreError("V6 identity selected-device profile drift")
    return selected


def build_reviewed_physical_backend(
    *, root: Path, selected_device_profile: Mapping[str, object],
) -> object:
    """Return the exact V5 physical backend without opening target/model/CUDA.

    This is deliberately a single typed delegation, not a fork of V5's
    parser, model loader, same-input materializer, or evaluator.
    """
    profile = plan.v5plan.v1plan.validate_compatible_device_profile(selected_device_profile)
    return v5physical.build_reviewed_physical_backend(root=Path(root), selected_device_profile=profile)


def execute_reviewed_physical_score(
    root: Path, *, identity: plan.ScoreIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Run one reviewed V6 score only with an opaque in-process capability.

    The mandatory order is durable authorization -> fresh reserve -> actual
    held-root validation -> immutable attempt -> V5 backend prepare.  Public
    CLI flags cannot manufacture ``capability`` and never call this function.
    """
    _assert_v6_typed_capability(capability, identity)
    score.validate_selected_launch_environment(identity, environ)
    profile = plan.v5plan.v1plan.validate_compatible_device_profile(_identity_profile(identity))
    backend = build_reviewed_physical_backend(root=Path(root), selected_device_profile=profile)
    preflight, authorization, pre_sha, auth_sha = score.load_durable_authority(Path(root), identity=identity)
    backend_preflight = backend.preflight(root=Path(root), identity=identity)
    if not isinstance(backend_preflight, Mapping) or any(backend_preflight.get(key) is not False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )):
        raise PhysicalV6ScoreError("V6 reviewed backend preflight is not target/model/CUDA-free")
    artifact = score.reserve_score_artifact(Path(root), identity=identity, capability=capability, environ=environ)
    return score.run_authorized_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization,
    )


def _assert_v6_typed_capability(capability: object, identity: plan.ScoreIdentity) -> v1score.ExecutionCapability:
    """Small testable typed seam; no public flag or mapping can substitute."""
    return v1score.require_execution_capability(capability, identity)
