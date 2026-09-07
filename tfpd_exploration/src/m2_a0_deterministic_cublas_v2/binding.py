"""Descriptor-first V2 admission and immutable science-profile comparison.

This file stays Torch/CUDA-free at import and never enumerates any device.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import plan
from tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1 import receipts


class BindingError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def source_closure(repo_root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative in plan.BOUND_PATTERNS:
        path = repo_root / relative
        _require(path.is_file() and not path.is_symlink(), f"V2 closure leaf missing/nonregular: {relative}")
        body = path.read_bytes()
        files[relative] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    _require(set(files) == set(plan.BOUND_PATTERNS), "V2 closure topology drift")
    return {"files": files, "closure_sha256": _canonical_sha(files),
            "workorder_sha256": files[plan.WORKORDER_RELATIVE]["sha256"],
            "historical_cpre_binding_sha256": files[plan.HISTORICAL_CPRE_BINDING_RELATIVE]["sha256"]}


def assert_bound_workorder(closure: Mapping[str, Any]) -> None:
    _require(closure.get("workorder_sha256") == plan.WORKORDER_SHA256
             and closure.get("historical_cpre_binding_sha256") == plan.HISTORICAL_CPRE_BINDING_SHA256,
             "V2 workorder/lineage authority byte drift")


def validate_deterministic_environment(environ: Mapping[str, str]) -> dict[str, str]:
    observed = {name: str(environ.get(name, "")) for name in plan.DETERMINISTIC_ENVIRONMENT}
    _require(observed == plan.DETERMINISTIC_ENVIRONMENT,
             "deterministic cuBLAS environment missing or drifted before root reservation")
    return observed


def _canonical_root(repo_root: Path, relative: str) -> Path:
    path = repo_root / relative
    _require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
             "result relative path escapes repository")
    return path


def _fresh_root_witness(repo_root: Path, relative: str, supplied_root: Path) -> dict[str, Any]:
    root = _canonical_root(repo_root, relative); parent = root.parent
    _require(supplied_root.absolute() == root.absolute() and parent.is_dir() and not parent.is_symlink()
             and not root.exists(), "V2 requires a fresh exact canonical root")
    info = os.stat(parent, follow_symlinks=False)
    return {"root_relative": relative, "root_name": root.name,
            "parent_device": int(info.st_dev), "parent_inode": int(info.st_ino)}


def verify_fresh_root_witness(repo_root: Path, relative: str, supplied_root: Path,
                              witness: Mapping[str, Any]) -> None:
    _require(set(witness) == {"root_relative", "root_name", "parent_device", "parent_inode"}
             and witness.get("root_relative") == relative, "V2 root witness schema/identity drift")
    current = _fresh_root_witness(repo_root, relative, supplied_root)
    _require(current == dict(witness), "V2 canonical root parent/freshness drift")


def _v1_failure_semantics(attempt: Mapping[str, Any], launch: Mapping[str, Any], failure: Mapping[str, Any]) -> None:
    _require(attempt.get("schema") == "m2_a0_attempt_v1"
             and launch.get("schema") == "m2_a0_launch_v1"
             and failure.get("schema") == "m2_a0_failure_v1"
             and failure.get("status") == "FAIL_CLOSED"
             and failure.get("stage") == "replay"
             and launch.get("attempt_sha256") == plan.V1_EXTERNAL_ATTEMPT_SHA256
             and failure.get("attempt_sha256") == plan.V1_EXTERNAL_ATTEMPT_SHA256,
             "V1 external failure receipt schema/link drift")
    _require(attempt.get("source_closure", {}).get("closure_sha256") == plan.V1_HISTORICAL_CLOSURE_SHA256,
             "V1 external historical closure drift")
    progress = failure.get("progress")
    _require(isinstance(progress, Mapping) and progress.get("parameter_updates") == 0
             and progress.get("target_gradients") == 0,
             "V1 external failure update/gradient law drift")
    error = str(failure.get("error", ""))
    _require("CUBLAS_WORKSPACE_CONFIG" in error and "deterministic" in error.lower(),
             "V1 external failure is not the deterministic-cuBLAS workspace failure")


def validate_v1_external_failure(repo_root: Path) -> dict[str, str]:
    """Held-FD/O_NOFOLLOW exact V1 failure graph validator; no model/data/CUDA."""
    root = _canonical_root(repo_root, plan.V1_EXTERNAL_ROOT_RELATIVE)
    descriptors = receipts.verify_topology(root, bodies=("attempt.json", "launch.json", "failure.json"))
    attempt, launch, failure = (descriptors[name] for name in ("attempt.json", "launch.json", "failure.json"))
    _require(attempt.sha256 == plan.V1_EXTERNAL_ATTEMPT_SHA256
             and launch.sha256 == plan.V1_EXTERNAL_LAUNCH_SHA256
             and failure.sha256 == plan.V1_EXTERNAL_FAILURE_SHA256,
             "V1 external failure exact body SHA drift")
    _v1_failure_semantics(attempt.payload, launch.payload, failure.payload)
    return {"root_relative": plan.V1_EXTERNAL_ROOT_RELATIVE, "attempt_sha256": attempt.sha256,
            "launch_sha256": launch.sha256, "failure_sha256": failure.sha256,
            "historical_closure_sha256": plan.V1_HISTORICAL_CLOSURE_SHA256}


@dataclass(frozen=True)
class ScienceProfile:
    carrier_budget: int = 10
    chronological_selected_indices: tuple[int, ...] = tuple(range(10))
    activity_seed_trials: int = 30
    activity_row_bins: int = 100
    chunk_length: int = 100
    chunk_phases: tuple[int, ...] = (0, 50)
    decoder_window_bins: int = 50
    arms: tuple[str, ...] = ("static", "true_trial", "chunk_phase0", "chunk_phase50")
    frozen_decode_batch_size: int = 32
    parameter_updates: int = 0
    target_gradients: int = 0


def v1_science_profile() -> ScienceProfile:
    """Typed exact mirror of reviewed V1 science; no mutable plan patching."""
    from tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1 import plan as v1_plan
    return ScienceProfile(
        carrier_budget=v1_plan.CARRIER_BUDGET,
        chronological_selected_indices=tuple(v1_plan.M10_CHRONOLOGICAL_SELECTED_INDICES),
        activity_seed_trials=v1_plan.CALIBRATION_TRIALS,
        activity_row_bins=v1_plan.B3S_BINS, chunk_length=v1_plan.CHUNK_LENGTH,
        chunk_phases=(v1_plan.PRIMARY_PHASE, v1_plan.SENSITIVITY_PHASE),
        decoder_window_bins=v1_plan.WINDOW_BINS,
        frozen_decode_batch_size=v1_plan.A0_FROZEN_DECODE_BATCH_SIZE,
    )


def assert_science_profile_parity(candidate: ScienceProfile) -> None:
    _require(candidate == v1_science_profile(), "V2 science profile differs from V1")


class _Capability:
    __slots__ = ("_token", "_binding", "_consumed")
    def __init__(self, token: object, binding: Mapping[str, Any]) -> None:
        self._token, self._binding, self._consumed = token, dict(binding), False


_ISSUER_TOKEN = object()


def _mint_test_capability(binding: Mapping[str, Any]) -> _Capability:
    return _Capability(_ISSUER_TOKEN, binding)


def issue_live_capability(*, repo_root: Path, root: Path, surface: str, roster: tuple[str, ...],
                          environ: Mapping[str, str]) -> _Capability:
    """Root-only V2 issuer; all predecessor/env checks precede root reservation.

    It is deliberately not reachable from the public CLI.  The imports below
    remain descriptor-only and do not construct a model or initialize CUDA.
    """
    _require(surface in ("external_post30_local", "within_post30") and bool(roster)
             and tuple(sorted(roster)) == tuple(roster), "unknown V2 shard surface/roster")
    environment = validate_deterministic_environment(environ)
    closure = source_closure(repo_root); assert_bound_workorder(closure)
    v1_failure = validate_v1_external_failure(repo_root)
    from tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1 import physical as v1_physical
    cpre = v1_physical._future_cpre_completion(repo_root=repo_root, closure=None)
    anchor, anchor_sha = v1_physical._future_static_anchor(repo_root)
    query = v1_physical._derive_a0_query_window_authority(
        metadata=cpre["metadata"], anchor_payload=anchor, surface=surface, roster=roster)
    scheduler = v1_physical.scheduler_profile(surface)
    scheduler_attestation = v1_physical.attest_a0_runtime_scheduler(surface=surface)
    witness = _fresh_root_witness(repo_root, plan.V2_ROOTS[surface], root)
    return _Capability(_ISSUER_TOKEN, {
        "mode": "live", "root_relative": plan.V2_ROOTS[surface], "root_witness": witness,
        "closure_sha256": closure["closure_sha256"], "deterministic_environment": environment,
        "v1_external_failure_predecessor": v1_failure,
        "historical_cpre_v2_witness": {
            "terminal_sha256": cpre["terminal_sha256"],
            "metadata_inventory_sha256": cpre["metadata_inventory_sha256"],
            "historical_cpre_closure_sha256": cpre["historical_cpre_closure_sha256"],
        },
        "surface": surface, "roster": list(roster),
        "cpre_terminal_sha256": cpre["terminal_sha256"],
        "cpre_metadata_inventory_sha256": cpre["metadata_inventory_sha256"],
        "static_anchor_payload": anchor, "static_anchor_descriptor_sha256": anchor_sha,
        "static_anchor_payload_sha256": _canonical_sha(anchor),
        "query_window_authority": query, "query_window_authority_sha256": _canonical_sha(query),
        "device_profile": v1_physical.static_device_profile(), "scheduler_profile": scheduler,
        "scheduler_attestation": scheduler_attestation,
        "frozen_decode_batch_size": 32, "resolved_window_size": 50,
        "science_profile": v1_science_profile().__dict__,
    })


def revalidate_held_binding(*, held: Mapping[str, Any], repo_root: Path, root: Path,
                            surface: str, roster: tuple[str, ...], closure: Mapping[str, Any],
                            environ: Mapping[str, str], allow_reserved_root: bool = False) -> None:
    """Current authority/environment final check for profile bridge execution."""
    _require(held.get("root_relative") == plan.V2_ROOTS[surface]
             and held.get("closure_sha256") == closure.get("closure_sha256")
             and held.get("surface") == surface and tuple(held.get("roster", ())) == roster,
             "V2 held root/current closure/surface/roster drift")
    _require(held.get("deterministic_environment") == validate_deterministic_environment(environ),
             "V2 held deterministic environment drift")
    assert_science_profile_parity(ScienceProfile(**dict(held.get("science_profile", {}))))
    if held.get("mode") == "live":
        if allow_reserved_root:
            witness = held.get("root_witness")
            _require(isinstance(witness, Mapping) and root.absolute() == _canonical_root(
                repo_root, plan.V2_ROOTS[surface]).absolute() and root.is_dir() and not root.is_symlink(),
                     "V2 reserved root identity drift")
            parent_info = os.stat(root.parent, follow_symlinks=False)
            _require(int(parent_info.st_dev) == int(witness.get("parent_device", -1))
                     and int(parent_info.st_ino) == int(witness.get("parent_inode", -1)),
                     "V2 reserved root parent drift")
        else:
            verify_fresh_root_witness(repo_root, plan.V2_ROOTS[surface], root, held["root_witness"])
        _require(validate_v1_external_failure(repo_root) == held.get("v1_external_failure_predecessor"),
                 "V2 final V1 external failure predecessor drift")
        from tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1 import physical as v1_physical
        cpre = v1_physical._future_cpre_completion(repo_root=repo_root, closure=None)
        current_anchor, anchor_sha = v1_physical._future_static_anchor(repo_root)
        _require(cpre["terminal_sha256"] == held.get("cpre_terminal_sha256")
                 and cpre["metadata_inventory_sha256"] == held.get("cpre_metadata_inventory_sha256")
                 and anchor_sha == held.get("static_anchor_descriptor_sha256")
                 and _canonical_sha(current_anchor) == held.get("static_anchor_payload_sha256")
                 and _canonical_sha(v1_physical._derive_a0_query_window_authority(
                     metadata=cpre["metadata"], anchor_payload=current_anchor, surface=surface, roster=roster))
                 == held.get("query_window_authority_sha256"),
                 "V2 final C-Pre/anchor/query authority drift")


def consume(capability: object, *, expected_root: str, closure: Mapping[str, Any],
            repo_root: Path | None = None, root: Path | None = None,
            environ: Mapping[str, str] | None = None) -> Mapping[str, Any]:
    _require(isinstance(capability, _Capability) and capability._token is _ISSUER_TOKEN
             and not capability._consumed, "opaque one-shot V2 capability required")
    binding = capability._binding
    _require(binding.get("root_relative") == expected_root and binding.get("closure_sha256") == closure.get("closure_sha256"),
             "V2 capability root/current-closure drift")
    _require(binding.get("deterministic_environment") == validate_deterministic_environment(
        os.environ if environ is None else environ), "V2 deterministic environment changed after issuance")
    witness = binding.get("root_witness")
    if witness is not None:
        _require(repo_root is not None and root is not None and isinstance(witness, Mapping),
                 "V2 live capability root evidence missing")
        verify_fresh_root_witness(repo_root, expected_root, root, witness)
    capability._consumed = True
    return binding
