"""Attempt-first, CPU-only production coordinator for the checkpoint scorer.

The public CLI deliberately has no route to these functions.  A root-owned
in-process issuer creates one opaque capability; the executor owns all
predecessor reads, physical work, and receipt construction itself.
"""
from __future__ import annotations

import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from . import binding, plan


class DriverError(RuntimeError):
    pass


_ISSUER_TOKEN = object()
_ISSUED_CAPABILITY_IDS: set[int] = set()


@dataclass(frozen=True)
class ExecutionProfile:
    """Lifecycle-only variance; the physical scoring operator is shared."""
    schema: str
    result_root_relative: str
    closure_fn: Callable[[Path], Mapping[str, object]]
    predecessor_validator: Callable[[Path], Mapping[str, object]] | None = None
    include_failure_diagnostic: bool = False


V1_PROFILE = ExecutionProfile(schema=plan.SCHEMA, result_root_relative=plan.RESULT_ROOT_RELATIVE,
                              closure_fn=lambda root: closure(root))


@dataclass
class _Capability:
    root: Path
    result_root: Path
    result_parent_identity: tuple[int, int]
    result_name: str
    closure_sha256: str
    screen_literal_sha256: Mapping[str, str]
    pooled_score_sha256: str
    profile: ExecutionProfile
    _issuer: object = field(repr=False, compare=False)
    lineage_witness: Mapping[str, object] | None = None
    consumed: bool = False


def closure(repo_root: Path) -> dict[str, object]:
    files: dict[str, str] = {}
    for relative in plan.CLOSURE_RELATIVES:
        path = Path(repo_root).absolute() / relative
        if not path.is_file() or path.is_symlink():
            raise DriverError(f"closure drift: {relative}")
        files[relative] = plan.sha256_file(path)
    return {"files": files, "sha256": plan.sha256_bytes(plan.canonical_json(files))}


def _require_cpu_environment() -> None:
    observed = {name: os.environ.get(name) for name in plan.CPU_ENV}
    if observed != plan.CPU_ENV:
        raise DriverError(f"CPU scorer environment drift: {observed}")


def _canonical_result_root(root: Path, profile: ExecutionProfile = V1_PROFILE) -> Path:
    candidate = Path(root).absolute() / profile.result_root_relative
    if candidate.parent != Path(root).absolute() / "tfpd_exploration/results":
        raise DriverError("score result root is not canonical")
    return candidate


def _fresh_parent(root: Path, profile: ExecutionProfile = V1_PROFILE) -> tuple[Path, tuple[int, int]]:
    target = _canonical_result_root(root, profile)
    parent = target.parent
    info = os.lstat(parent)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or target.exists():
        raise DriverError("score root must be fresh under a regular canonical parent")
    return target, (int(info.st_dev), int(info.st_ino))


def issue_live_capability(repo_root: Path, *, token: object, profile: ExecutionProfile = V1_PROFILE,
                          lineage_witness: Mapping[str, object] | None = None) -> _Capability:
    """Root-only issuance: it deliberately does not open predecessors yet."""
    if token is not _ISSUER_TOKEN:
        raise DriverError("opaque issuer token mismatch")
    _require_cpu_environment()
    root = Path(repo_root).absolute()
    target, parent_identity = _fresh_parent(root, profile)
    literals = binding.require_live_literals()
    sha = literals.get("sha256")
    if not isinstance(sha, Mapping) or set(sha) != {
        "attempt.json", "launch.json", "source_authority.json", "screen.json", "terminal.json",
        "source_best_pf_mean.pt", "source_best_pf_r1.pt", "source_best_pf_r50.pt",
    }:
        raise DriverError("screen literal topology is incomplete")
    cap = _Capability(root=root, result_root=target, result_parent_identity=parent_identity,
                      result_name=target.name, closure_sha256=str(profile.closure_fn(root)["sha256"]),
                      screen_literal_sha256={str(k): str(v) for k, v in sha.items()},
                      pooled_score_sha256=plan.POOLED_COMPARATOR_SHA256, profile=profile,
                      lineage_witness=dict(lineage_witness) if lineage_witness is not None else None,
                      _issuer=_ISSUER_TOKEN)
    _ISSUED_CAPABILITY_IDS.add(id(cap))
    return cap


def _mint_synthetic(repo_root: Path) -> _Capability:
    """Private test-only issuer; never exposed by the public CLI."""
    return issue_live_capability(Path(repo_root), token=_ISSUER_TOKEN)


def _consume(capability: _Capability) -> None:
    if (not isinstance(capability, _Capability) or capability._issuer is not _ISSUER_TOKEN
            or id(capability) not in _ISSUED_CAPABILITY_IDS or capability.consumed):
        raise DriverError("capability invalid, forged, or reused")
    _require_cpu_environment()
    if str(capability.profile.closure_fn(capability.root)["sha256"]) != capability.closure_sha256:
        raise DriverError("closure drift")
    target, parent_identity = _fresh_parent(capability.root, capability.profile)
    if (target != capability.result_root or target.name != capability.result_name
            or parent_identity != capability.result_parent_identity):
        raise DriverError("canonical result-root/parent identity drift")
    capability.consumed = True


def _revalidate_after_reservation(capability: _Capability) -> None:
    _require_cpu_environment()
    if str(capability.profile.closure_fn(capability.root)["sha256"]) != capability.closure_sha256:
        raise DriverError("final closure drift")
    info = os.lstat(capability.result_root)
    parent = os.lstat(capability.result_root.parent)
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != capability.result_parent_identity):
        raise DriverError("reserved score root/parent drift")


def _public_input_authority(materialized: Mapping[str, Any], pooled: Mapping[str, Mapping[str, object]],
                            prepared: Mapping[str, Any]) -> dict[str, object]:
    records = materialized.get("records")
    if not isinstance(records, Mapping) or len(records) != 13:
        raise DriverError("materialized record authority is incomplete")
    public_records = {
        str(key): {str(field): value for field, value in record.items() if field != "_runtime"}
        for key, record in records.items()
    }
    if any("_runtime" in record for record in public_records.values()):
        raise DriverError("runtime object escaped input authority")
    return {"schema": f"{plan.SCHEMA}_input_authority_v1", "record_count": 13,
            "records": public_records, "append_heldout_evidence": materialized.get("append_heldout_evidence"),
            "pooled_comparator": {str(key): dict(value) for key, value in pooled.items()},
            "frozen_model_evidence": prepared.get("evidence"), "cuda_initialized": False}


def _canonical_record_keys(materialized: Mapping[str, Any]) -> list[str]:
    records = materialized.get("records")
    if not isinstance(records, Mapping):
        raise DriverError("materialized records absent")
    ordered: list[str] = []
    for surface in plan.SURFACES:
        values = sorted((value for value in records.values() if value.get("surface") == surface),
                        key=lambda value: str(value["session"]))
        if len(values) != plan.ROSTER_SIZES[surface]:
            raise DriverError("materialized canonical roster drift")
        ordered.extend(str(value["key"]) for value in values)
    return ordered


def execute_production(capability: _Capability) -> tuple[str | None, str | None]:
    """Run the one route-owned production graph after immutable attempt.

    This function is intentionally never exercised against target data by the
    no-data test suite.  Its only variable work is behind reviewed physical
    primitives; no caller can supply a scorer/materializer callback.
    """
    _consume(capability)
    # Keep admission stdlib-only.  The physical/metric modules are imported
    # inside ``bodies`` only after the lifecycle has published attempt.json.
    from . import lifecycle
    progress: dict[str, object] = {"stage": "reserved", "rows": 0, "cuda_initialized": False,
                                   "target_or_checkpoint_opened": False, "published_prefix": [],
                                   "screen_descriptor_attempted": False, "screen_descriptor_opened": False,
                                   "pooled_descriptor_attempted": False, "pooled_descriptor_opened": False,
                                   "checkpoint_strict_load_attempted": False, "checkpoint_strict_loaded": False,
                                   "target_materialization_attempted": False, "target_materialized": False}
    if capability.lineage_witness is not None:
        progress["lineage_witness"] = dict(capability.lineage_witness)
    attempt = {"schema": f"{capability.profile.schema}_attempt_v1", "status": "ATTEMPT_RESERVED",
               "closure_sha256": capability.closure_sha256, "cpu_environment": dict(plan.CPU_ENV),
               "screen_literal_sha256": dict(capability.screen_literal_sha256),
               "pooled_score_sha256": capability.pooled_score_sha256,
               "target_or_checkpoint_opened": False, "cuda_initialized": False}
    if capability.lineage_witness is not None:
        attempt["lineage_witness"] = dict(capability.lineage_witness)

    def launch() -> Mapping[str, object]:
        _revalidate_after_reservation(capability)
        progress["stage"] = "launch_published"
        return {"schema": f"{capability.profile.schema}_launch_v1", "attempt_closure_sha256": capability.closure_sha256,
                "cuda_visible_devices": "", "cuda_initialized": False,
                "result_root": capability.profile.result_root_relative,
                "cpu_decode_batch_size": plan.CPU_DECODE_BATCH_SIZE}

    def bodies(artifact: Any) -> Mapping[str, str]:
        # All held predecessor/checkpoint reads occur strictly after attempt.
        from . import laws, physical
        if capability.profile.predecessor_validator is not None:
            capability.profile.predecessor_validator(capability.root)
        progress["screen_descriptor_attempted"] = True
        screen = binding.validate_screen_graph(
            capability.root / str(binding.require_live_literals()["screen_root_relative"]),
            binding.require_live_literals(), capture_checkpoint_bytes=True)
        progress["screen_descriptor_opened"] = True
        progress["pooled_descriptor_attempted"] = True
        pooled_witness = binding.validate_pooled_comparator_score(
            capability.root / plan.POOLED_COMPARATOR_ROOT_RELATIVE)
        progress["pooled_descriptor_opened"] = True
        if str(pooled_witness["body_sha256"]) != capability.pooled_score_sha256:
            raise DriverError("held POOLED predecessor literal drift")
        progress["target_or_checkpoint_opened"] = True
        progress["stage"] = "held_predecessors_validated"
        progress["checkpoint_strict_load_attempted"] = True
        prepared = physical.prepare_three_frozen_arms_from_held_screen(
            repo_root=capability.root, checkpoint_bytes=screen["checkpoint_bytes"])
        progress["checkpoint_strict_loaded"] = True
        progress["target_materialization_attempted"] = True
        materialized = physical.materialize_13_inputs_and_pooled_comparators(
            prepared=prepared, pooled_comparator=pooled_witness["comparators"])
        progress["target_materialized"] = True
        input_payload = _public_input_authority(materialized, pooled_witness["comparators"], prepared)
        input_payload["schema"] = f"{capability.profile.schema}_input_authority_v1"
        input_sha = artifact.publish_json("input_authority.json", input_payload)
        progress["published_prefix"] = ["attempt.json", "launch.json", "input_authority.json"]
        record_keys = _canonical_record_keys(materialized)
        smoke_key = record_keys[0]
        smoke_start = time.monotonic()
        smoke_rows = physical.score_records_from_materialized(prepared=prepared, materialized=materialized,
                                                               record_keys=[smoke_key])
        smoke_elapsed = float(time.monotonic() - smoke_start)
        projected = smoke_elapsed * len(record_keys)
        progress.update({"stage": "smoke_complete", "rows": len(smoke_rows), "smoke_record": smoke_key,
                         "smoke_wall_seconds": smoke_elapsed, "projected_full_wall_seconds": projected})
        if projected > 7200.0:
            raise DriverError("one-session smoke projects full CPU score beyond 7200 seconds")
        remaining_rows = physical.score_records_from_materialized(prepared=prepared, materialized=materialized,
                                                                   record_keys=record_keys[1:])
        rows = smoke_rows + remaining_rows
        laws.validate_rows(rows)
        derived = laws.recompute(rows, pooled_witness["comparators"])
        progress.update({"stage": "score_complete", "rows": len(rows)})
        score_payload = {"schema": f"{capability.profile.schema}_score_v1", "rows": rows, **derived,
                         "smoke": {"record_key": smoke_key, "row_count": len(smoke_rows),
                                   "wall_seconds": smoke_elapsed, "projected_full_wall_seconds": projected,
                                   "hard_cap_seconds": 7200.0, "passed": True},
                         "runtime": {"cuda_initialized": False, "cpu_decode_batch_size": plan.CPU_DECODE_BATCH_SIZE,
                                     "total_rows": len(rows)}}
        score_sha = artifact.publish_json("score.json", score_payload)
        progress["published_prefix"] = ["attempt.json", "launch.json", "input_authority.json", "score.json"]
        return {"input_authority.json": input_sha, "score.json": score_sha}

    def terminal(published: Mapping[str, str]) -> Mapping[str, object]:
        _revalidate_after_reservation(capability)
        if capability.profile.predecessor_validator is not None:
            capability.profile.predecessor_validator(capability.root)
        # Descriptor-reload all immutable predecessors again at final time.
        screen = binding.validate_screen_graph(
            capability.root / str(binding.require_live_literals()["screen_root_relative"]),
            binding.require_live_literals())
        pooled = binding.validate_pooled_comparator_score(capability.root / plan.POOLED_COMPARATOR_ROOT_RELATIVE)
        if str(pooled["body_sha256"]) != capability.pooled_score_sha256:
            raise DriverError("final POOLED predecessor drift")
        payload = {"input_authority_sha256": published["input_authority.json"], "score_sha256": published["score.json"],
                "row_count": plan.EXPECTED_ROWS, "target_updates": 0, "parameter_updates": 0,
                "cuda_initialized": False, "screen_predecessor_sha256": dict(screen["digests"]),
                "pooled_predecessor_score_sha256": pooled["body_sha256"],
                "current_closure_sha256": capability.closure_sha256,
                "matched_prefusion_control_trained": False, "official_submission_surface_evaluated": False,
                "exploratory_candidate_selection_only": True,
                "limitations": ["historical_pooled_comparator_is_unmatched_training_context",
                                "CPU_only_no_CUDA", "nomination_requires_future_matched_prefusion_confirmation"]}
        if capability.lineage_witness is not None:
            payload["lineage_witness"] = dict(capability.lineage_witness)
        return payload

    return lifecycle.execute(capability.root, attempt=attempt, launch=launch, bodies=bodies,
                             terminal=terminal, progress=lambda: dict(progress),
                             root_relative=capability.profile.result_root_relative, schema=capability.profile.schema,
                             include_failure_diagnostic=capability.profile.include_failure_diagnostic,
                             failure_revalidate=(lambda: capability.profile.predecessor_validator(capability.root))
                             if capability.profile.predecessor_validator is not None else None)


def _execute_synthetic_for_test(*, capability: _Capability, fail_after: str | None = None) -> tuple[str | None, str | None]:
    """Private lifecycle-only test seam; it is not a production callback API."""
    _consume(capability)
    from . import lifecycle
    progress: dict[str, object] = {"stage": "synthetic", "rows": 0, "cuda_initialized": False}
    attempt = {"schema": f"{plan.SCHEMA}_attempt_v1", "status": "ATTEMPT_RESERVED",
               "closure_sha256": capability.closure_sha256, "cuda_initialized": False}
    def launch() -> Mapping[str, object]:
        _revalidate_after_reservation(capability)
        if fail_after == "attempt":
            raise DriverError("synthetic failure after attempt")
        return {"schema": f"{plan.SCHEMA}_launch_v1", "cuda_visible_devices": "", "cuda_initialized": False}
    def bodies(artifact: Any) -> Mapping[str, str]:
        first = artifact.publish_json("input_authority.json", {"records": [], "cuda_initialized": False})
        progress["stage"] = "input_published"
        if fail_after == "input_authority":
            raise DriverError("synthetic failure after input authority")
        second = artifact.publish_json("score.json", {"rows": [], "cuda_initialized": False})
        progress["stage"] = "score_published"
        if fail_after == "score":
            raise DriverError("synthetic failure after score")
        return {"input_authority.json": first, "score.json": second}
    def terminal(published: Mapping[str, str]) -> Mapping[str, object]:
        _revalidate_after_reservation(capability)
        return {"input_authority_sha256": published["input_authority.json"],
                "score_sha256": published["score.json"], "row_count": 0, "cuda_initialized": False}
    return lifecycle.execute(capability.root, attempt=attempt, launch=launch, bodies=bodies,
                             terminal=terminal, progress=lambda: dict(progress))
