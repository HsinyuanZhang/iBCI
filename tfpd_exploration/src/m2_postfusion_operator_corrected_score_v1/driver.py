"""Attempt-first CPU production entry point for the operator correction."""
from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import binding, plan


class DriverError(RuntimeError):
    pass


_TOKEN = object()
_CAPS: set[int] = set()


def closure(repo_root: Path) -> dict[str, object]:
    root = Path(repo_root).absolute()
    files: dict[str, str] = {}
    for relative in plan.CLOSURE_RELATIVES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise DriverError(f"operator-corrected closure drift: {relative}")
        files[relative] = plan.sha256_file(path)
    return {"files": files, "sha256": plan.sha256_bytes(plan.canonical_json(files))}


def _cpu() -> None:
    observed = {key: os.environ.get(key) for key in plan.CPU_ENV}
    if observed != plan.CPU_ENV:
        raise DriverError(f"CPU environment drift: {observed}")


def _target(root: Path) -> tuple[Path, tuple[int, int]]:
    target = Path(root).absolute() / plan.RESULT_ROOT_RELATIVE
    parent = target.parent
    info = os.lstat(parent)
    if os.path.lexists(target) or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise DriverError("operator-corrected canonical root unavailable or not fresh")
    return target, (int(info.st_dev), int(info.st_ino))


@dataclass
class _Capability:
    root: Path
    result_root: Path
    parent_identity: tuple[int, int]
    closure_sha256: str
    _token: object = field(repr=False, compare=False)
    consumed: bool = False


def issue_live_capability(repo_root: Path, *, token: object) -> _Capability:
    """Root-only opaque mint; it reads no held graph before attempt."""
    if token is not _TOKEN:
        raise DriverError("operator-corrected opaque token mismatch")
    root = Path(repo_root).absolute()
    plan.validate_static(root)
    _cpu()
    target, parent = _target(root)
    cap = _Capability(root=root, result_root=target, parent_identity=parent,
                      closure_sha256=str(closure(root)["sha256"]), _token=_TOKEN)
    _CAPS.add(id(cap))
    return cap


def _consume(cap: _Capability) -> None:
    if not isinstance(cap, _Capability) or cap._token is not _TOKEN or id(cap) not in _CAPS or cap.consumed:
        raise DriverError("operator-corrected capability forged or reused")
    _cpu()
    if str(closure(cap.root)["sha256"]) != cap.closure_sha256:
        raise DriverError("operator-corrected closure drift")
    target, parent = _target(cap.root)
    if target != cap.result_root or parent != cap.parent_identity:
        raise DriverError("operator-corrected root identity drift")
    cap.consumed = True


def _revalidate(cap: _Capability) -> None:
    _cpu()
    if str(closure(cap.root)["sha256"]) != cap.closure_sha256:
        raise DriverError("operator-corrected final closure drift")
    info = os.lstat(cap.result_root); parent = os.lstat(cap.result_root.parent)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or (int(parent.st_dev), int(parent.st_ino)) != cap.parent_identity:
        raise DriverError("operator-corrected reserved root drift")


def _require_streaming_namespace_clean(root: Path) -> None:
    """Fail before source setup if `src` would resolve to TFPD, not streaming.

    Falcon's resolved Hydra target is deliberately `src.data...`.  Qualified
    TFPD imports are safe, but placing `tfpd_exploration/` itself ahead of the
    repo root makes that target resolve to the wrong top-level package.  The
    successful V2 run used repo-root import context; this successor makes that
    operational prerequisite explicit rather than failing after its attempt.
    """
    if "src" in sys.modules:
        raise DriverError("top-level src already imported; streaming namespace is ambiguous")
    tfpd_root = (Path(root).absolute() / "tfpd_exploration").resolve()
    for entry in sys.path:
        if entry and Path(entry).absolute().resolve() == tfpd_root:
            raise DriverError("PYTHONPATH exposes tfpd_exploration as top-level src; require repo-root-only import context")


def _public_input(materialized: Mapping[str, Any], prepared: Mapping[str, Any], v2: Mapping[str, Any]) -> dict[str, object]:
    records = materialized.get("records")
    if not isinstance(records, Mapping) or len(records) != 13:
        raise DriverError("13 input authority records absent")
    public = {str(key): {str(field): value for field, value in row.items() if field != "_runtime"}
              for key, row in records.items()}
    expected = v2["input_authority"]["records"]
    if public != expected:
        raise DriverError("operator correction input authority differs from locked V2 input")
    return {"schema": f"{plan.SCHEMA}_input_authority_v1", "record_count": 13, "records": public,
            "v2_input_authority_sha256": v2["body_sha256"]["input_authority.json"],
            "frozen_model_evidence": prepared.get("evidence"), "cuda_initialized": False}


def execute_production(capability: _Capability) -> tuple[str | None, str | None]:
    """Execute the one fixed, frozen operator correction after `attempt`.

    No caller passes a model, records, metric, or callback.  All live material
    is built exclusively after immutable attempt publication.
    """
    _consume(capability)
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import binding as v1_binding
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import laws, lifecycle, physical as v1_physical
    from . import physical
    progress: dict[str, object] = {"stage": "reserved", "rows": 0, "cuda_initialized": False,
                                   "v2_descriptor_attempted": False, "v2_descriptor_opened": False,
                                   "screen_descriptor_attempted": False, "screen_descriptor_opened": False,
                                   "checkpoint_strict_load_attempted": False, "checkpoint_strict_loaded": False,
                                   "target_materialization_attempted": False, "target_materialized": False,
                                   "published_prefix": []}
    attempt = {"schema": f"{plan.SCHEMA}_attempt_v1", "status": "ATTEMPT_RESERVED",
               "closure_sha256": capability.closure_sha256, "cpu_environment": dict(plan.CPU_ENV),
               "cuda_initialized": False, "v2_predecessor": {"root_relative": plan.V2_ROOT_RELATIVE,
               "body_sha256": dict(plan.V2_BODIES), "closure_sha256": plan.V2_CLOSURE_SHA256}}

    def launch() -> Mapping[str, object]:
        _revalidate(capability)
        return {"schema": f"{plan.SCHEMA}_launch_v1", "cuda_visible_devices": "", "cuda_initialized": False,
                "result_root": plan.RESULT_ROOT_RELATIVE, "cpu_decode_batch_size": plan.CPU_DECODE_BATCH_SIZE}

    def bodies(artifact: Any) -> Mapping[str, str]:
        _require_streaming_namespace_clean(capability.root)
        progress["v2_descriptor_attempted"] = True
        v2 = binding.validate_v2_success_graph(capability.root)
        progress["v2_descriptor_opened"] = True
        progress["screen_descriptor_attempted"] = True
        literals = v1_binding.require_live_literals()
        screen = v1_binding.validate_screen_graph(capability.root / str(literals["screen_root_relative"]), literals,
                                                  capture_checkpoint_bytes=True)
        progress["screen_descriptor_opened"] = True
        pooled = v1_binding.validate_pooled_comparator_score(capability.root / v1_physical.plan.POOLED_COMPARATOR_ROOT_RELATIVE)
        progress["checkpoint_strict_load_attempted"] = True
        prepared = v1_physical.prepare_three_frozen_arms_from_held_screen(repo_root=capability.root,
                                                                            checkpoint_bytes=screen["checkpoint_bytes"])
        progress["checkpoint_strict_loaded"] = True
        progress["target_materialization_attempted"] = True
        materialized = v1_physical.materialize_13_inputs_and_pooled_comparators(prepared=prepared,
                                                                                   pooled_comparator=pooled["comparators"])
        progress["target_materialized"] = True
        input_sha = artifact.publish_json("input_authority.json", _public_input(materialized, prepared, v2))
        progress["published_prefix"] = ["attempt.json", "launch.json", "input_authority.json"]
        rows = physical.score_operator_corrected_78_rows(prepared=prepared, materialized=materialized,
                                                          v2_control=v2["pfmean_control"])
        laws.validate_rows(rows)
        derived = laws.recompute(rows, pooled["comparators"])
        score_sha = artifact.publish_json("score.json", {"schema": f"{plan.SCHEMA}_score_v1", "rows": rows,
                                           **derived, "v2_pfmean_control_exact": True,
                                           "legacy_literal_residual_rows": {"computed_only_to_preserve_pfmean_control": True,
                                               "published": False, "used_for_summary_or_gate": False},
                                           "operator_correction": {"PF-MEAN": "inherited_v2_literal_singleton_pool",
                                           "PF-R1": "trained_adapter_whole_ordered_activity_stack",
                                           "PF-R50": "trained_adapter_whole_ordered_activity_stack"},
                                           "runtime": {"cuda_initialized": False, "parameter_updates": 0,
                                                       "target_updates": 0, "total_rows": 78}})
        progress.update({"stage": "score_complete", "rows": 78,
                         "published_prefix": ["attempt.json", "launch.json", "input_authority.json", "score.json"]})
        return {"input_authority.json": input_sha, "score.json": score_sha}

    def terminal(published: Mapping[str, str]) -> Mapping[str, object]:
        _revalidate(capability)
        final_v2 = binding.validate_v2_success_graph(capability.root)
        return {"v2_predecessor": {"root_relative": final_v2["root_relative"], "body_sha256": final_v2["body_sha256"],
                "closure_sha256": final_v2["closure_sha256"]}, "input_authority_sha256": published["input_authority.json"],
                "score_sha256": published["score.json"], "row_count": 78, "cuda_initialized": False,
                "parameter_updates": 0, "target_updates": 0, "pfmean_v2_control_exact": True,
                "limitations": ["fixed_operator_correction_not_retraining", "historical_pooled_unmatched_training_context",
                                "CPU_only_no_CUDA"]}

    return lifecycle.execute(capability.root, attempt=attempt, launch=launch, bodies=bodies, terminal=terminal,
                             progress=lambda: dict(progress), root_relative=plan.RESULT_ROOT_RELATIVE, schema=plan.SCHEMA,
                             include_failure_diagnostic=True, failure_revalidate=lambda: (_revalidate(capability), binding.validate_v2_success_graph(capability.root)))


def _execute_synthetic_lifecycle_for_test(capability: _Capability, *, fail_after: str | None = None) -> tuple[str | None, str | None]:
    """Private test seam for this driver's real attempt/terminal/failure order.

    It never accepts arbitrary production callbacks and is intentionally not
    imported by the public CLI.  Physical source/target work remains exclusive
    to :func:`execute_production`.
    """
    _consume(capability)
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import lifecycle
    progress: dict[str, object] = {"stage": "synthetic", "rows": 0, "cuda_initialized": False,
                                   "published_prefix": []}
    attempt = {"schema": f"{plan.SCHEMA}_attempt_v1", "status": "ATTEMPT_RESERVED",
               "closure_sha256": capability.closure_sha256, "cuda_initialized": False}
    def launch() -> Mapping[str, object]:
        _revalidate(capability)
        if fail_after == "attempt":
            raise DriverError("synthetic failure after attempt")
        return {"schema": f"{plan.SCHEMA}_launch_v1", "cuda_visible_devices": "", "cuda_initialized": False}
    def bodies(artifact: Any) -> Mapping[str, str]:
        first = artifact.publish_json("input_authority.json", {"records": {}, "cuda_initialized": False})
        progress["published_prefix"] = ["attempt.json", "launch.json", "input_authority.json"]
        if fail_after == "input":
            raise DriverError("synthetic failure after input")
        second = artifact.publish_json("score.json", {"rows": [], "cuda_initialized": False})
        progress["published_prefix"] = ["attempt.json", "launch.json", "input_authority.json", "score.json"]
        if fail_after == "score":
            raise DriverError("synthetic failure after score")
        return {"input_authority.json": first, "score.json": second}
    def terminal(published: Mapping[str, str]) -> Mapping[str, object]:
        _revalidate(capability)
        return {"input_authority_sha256": published["input_authority.json"], "score_sha256": published["score.json"],
                "row_count": 0, "cuda_initialized": False}
    return lifecycle.execute(capability.root, attempt=attempt, launch=launch, bodies=bodies, terminal=terminal,
                             progress=lambda: dict(progress), root_relative=plan.RESULT_ROOT_RELATIVE, schema=plan.SCHEMA,
                             include_failure_diagnostic=True)
