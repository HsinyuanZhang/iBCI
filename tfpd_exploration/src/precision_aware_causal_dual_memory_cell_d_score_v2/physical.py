"""Deferred V2 physical composition with an exact source-root environment seam."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.causal_dual_memory_cell_d_score_v1 import score as sharedscore
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import physical as predecessor_physical

from . import plan, score


class PhysicalPrecisionMatchedScoreV2Error(score.PrecisionMatchedScoreV2Error):
    """Fail closed for a V2 environment or delegating-backend seam drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalPrecisionMatchedScoreV2Error(message)


def _assert_actual_environment_matches(
    identity: plan.ScoreIdentity, environ: Mapping[str, str] | None,
) -> None:
    """Reject a test/caller mapping that differs from the process to be used.

    The inherited evaluator deliberately reads `os.environ` when it opens a
    held source root.  A separate caller mapping could therefore validate a
    harmless string and still let the inherited parser consume another root.
    Production execution consequently exact-compares the four relevant
    process variables before constructing the backend; no environment is
    mutated by this route.
    """
    score.validate_selected_launch_environment(identity, environ)
    if environ is not None:
        keys = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", *plan.CANONICAL_SOURCE_ROOTS)
        if any(os.environ.get(key) != environ.get(key) for key in keys):
            raise PhysicalPrecisionMatchedScoreV2Error("V2 supplied/actual launch environment drift")
    score.validate_selected_launch_environment(identity, os.environ)


class PhysicalPrecisionMatchedScoreV2Backend:
    """A route-local guard around the exact V1 V8/V5/Precision runtime.

    It does not override parser, model, input, forward, metric, transition,
    or revalidation semantics.  It calls the V1 backend with the V2 identity,
    which is a typed subclass of V1 identity and carries the same immutable
    V8 science witness.  The only added behavior is checking the exact two
    source-root strings before every physical boundary and at final
    revalidation.
    """

    def __init__(self, *, root: Path, identity: plan.ScoreIdentity) -> None:
        selected = identity.payload().get("selected_device_profile")
        if not isinstance(selected, Mapping):
            raise PhysicalPrecisionMatchedScoreV2Error("V2 selected-device profile drift")
        self._identity = identity
        self._root = Path(root).absolute()
        self._delegate = predecessor_physical.build_reviewed_physical_backend(
            root=self._root, selected_device_profile=selected,
        )

    def _gate(self, *, root: Path, identity: plan.ScoreIdentity) -> None:
        _require(identity is self._identity, "V2 physical identity object drift")
        _require(Path(root).absolute() == self._root, "V2 physical root drift")
        score.validate_selected_launch_environment(identity, os.environ)
        score.validate_v1_failed_predecessor(self._root)
        if score.validate_v1_failed_predecessor(self._root).payload() != identity.payload()["v1_failed_predecessor_binding"]:
            raise PhysicalPrecisionMatchedScoreV2Error("V2 physical failed-V1 predecessor binding drift")

    def preflight(self, *, root: Path, identity: plan.ScoreIdentity) -> Mapping[str, object]:
        self._gate(root=root, identity=identity)
        return self._delegate.preflight(root=Path(root), identity=identity)

    def prepare(self, *, root: Path, identity: plan.ScoreIdentity) -> Any:
        self._gate(root=root, identity=identity)
        return self._delegate.prepare(root=Path(root), identity=identity)

    def materialize_inputs(self, runtime: Any, *, identity: plan.ScoreIdentity,
                           evaluation_authority: Any) -> sharedscore.InputAuthority:
        self._gate(root=self._root, identity=identity)
        return self._delegate.materialize_inputs(runtime, identity=identity, evaluation_authority=evaluation_authority)

    def score_budget(self, runtime: Any, *, budget: int, input_authority_sha256: str,
                     identity: plan.ScoreIdentity) -> Any:
        self._gate(root=self._root, identity=identity)
        return self._delegate.score_budget(
            runtime, budget=budget, input_authority_sha256=input_authority_sha256, identity=identity,
        )

    def revalidate(self, runtime: Any, *, root: Path, identity: plan.ScoreIdentity) -> None:
        self._gate(root=root, identity=identity)
        self._delegate.revalidate(runtime, root=Path(root), identity=identity)
        self._gate(root=root, identity=identity)

    def failure_progress(self, runtime: Any | None) -> Mapping[str, object]:
        return self._delegate.failure_progress(runtime)

    def close(self, runtime: Any | None) -> None:
        self._delegate.close(runtime)


def build_reviewed_physical_backend(
    *, root: Path, identity: plan.ScoreIdentity,
) -> PhysicalPrecisionMatchedScoreV2Backend:
    """Construct only composition objects; no environment/data/CUDA action."""
    if not isinstance(identity, plan.ScoreIdentity):
        raise PhysicalPrecisionMatchedScoreV2Error("V2 physical identity type drift")
    return PhysicalPrecisionMatchedScoreV2Backend(root=Path(root), identity=identity)


def execute_reviewed_physical_score(
    root: Path, *, identity: plan.ScoreIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Future reviewed launch adapter; public CLI remains dry-only.

    The environment check is intentionally first.  Thus a missing, swapped,
    relative, or alias source-root string cannot reserve a V2 authority/score
    root, publish an attempt, load a checkpoint, initialize CUDA, or open an
    evaluation asset.
    """
    _assert_actual_environment_matches(identity, environ)
    score.validate_v1_failed_predecessor(Path(root))
    selected = identity.payload().get("selected_device_profile")
    _require(isinstance(selected, Mapping), "V2 selected-device profile drift")
    backend = build_reviewed_physical_backend(root=Path(root), identity=identity)
    preflight, authorization, pre_sha, auth_sha = score.load_durable_authority(Path(root), identity=identity)
    checked = backend.preflight(root=Path(root), identity=identity)
    _require(isinstance(checked, Mapping) and all(checked.get(key) is False for key in (
        "target_paths_resolved", "target_opened", "checkpoint_opened", "cuda_initialized",
    )), "V2 backend preflight is not target/model/CUDA-free")
    artifact = score.reserve_score_artifact(Path(root), identity=identity, capability=capability, environ=environ)
    return score.run_authorized_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization, environ=environ,
    )
