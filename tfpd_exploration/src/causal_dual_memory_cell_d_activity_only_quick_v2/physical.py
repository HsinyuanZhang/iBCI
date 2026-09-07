"""Environment-bound lifecycle over the V1 activity-only science/runtime."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import time
from typing import Any, Mapping

from src.causal_dual_memory_cell_d_activity_only_quick_v1 import physical as v1quick
from src.causal_dual_memory_cell_d_score_v1 import plan as score_plan
from src.causal_dual_memory_cell_d_score_v1 import score as score_contract
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import score as predecessor_score

from . import plan


class ActivityOnlyQuickV2Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActivityOnlyQuickV2Error(message)


def _read_regular_at(directory_fd: int, name: str) -> tuple[bytes, os.stat_result]:
    _require(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW is required")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode), f"predecessor leaf is not regular: {name}")
        _require(stat.S_IMODE(info.st_mode) == 0o444, f"predecessor leaf mode drift: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), info
    finally:
        os.close(descriptor)


def validate_failed_v1(root: Path) -> dict[str, object]:
    base = Path(root).absolute()
    path = base / plan.V1_FAILED_ROOT_RELATIVE
    _require(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW is required")
    before = os.lstat(path)
    _require(stat.S_ISDIR(before.st_mode), "V1 failed predecessor is not a directory")
    _require(stat.S_IMODE(before.st_mode) == 0o555, "V1 failed predecessor directory mode drift")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(path, flags)
    try:
        directory_info = os.fstat(directory_fd)
        expected_names = {
            name
            for body in plan.V1_FAILED_BODY_SHA256S
            for name in (body, body + ".sha256")
        }
        _require(set(os.listdir(directory_fd)) == expected_names, "V1 failed predecessor topology drift")
        bodies: dict[str, Mapping[str, object]] = {}
        for name, expected_sha in plan.V1_FAILED_BODY_SHA256S.items():
            raw, _ = _read_regular_at(directory_fd, name)
            digest = hashlib.sha256(raw).hexdigest()
            _require(digest == expected_sha, f"V1 failed predecessor body drift: {name}")
            sidecar, _ = _read_regular_at(directory_fd, name + ".sha256")
            _require(sidecar == f"{digest}  {name}\n".encode("ascii"), f"V1 sidecar drift: {name}")
            payload = json.loads(raw)
            _require(isinstance(payload, dict), f"V1 predecessor payload drift: {name}")
            bodies[name] = payload
        attempt = bodies["attempt.json"]
        failure = bodies["failure.json"]
        _require(attempt.get("cell") == "CAUSAL_DUAL_MEMORY_CELL_D_ACTIVITY_ONLY_QUICK_V1", "V1 cell drift")
        _require(attempt.get("status") == "ATTEMPT_RESERVED", "V1 attempt status drift")
        _require(failure.get("status") == "FAILED" and failure.get("terminal") is False, "V1 failure status drift")
        _require(failure.get("attempt_sha256") == plan.V1_FAILED_BODY_SHA256S["attempt.json"], "V1 failure lineage drift")
        _require(failure.get("stage") == "materialize_inputs", "V1 failure stage drift")
        _require(failure.get("input_authority_sha256_or_null") is None, "V1 unexpectedly opened input authority")
        after = os.lstat(path)
        _require(
            (before.st_dev, before.st_ino) == (directory_info.st_dev, directory_info.st_ino)
            == (after.st_dev, after.st_ino),
            "V1 failed predecessor directory replaced during validation",
        )
        return {
            "schema": "causal_dual_memory_cell_d_activity_only_quick_v1_failed_binding_v1",
            "root_relative": plan.V1_FAILED_ROOT_RELATIVE,
            "directory_identity": [directory_info.st_dev, directory_info.st_ino],
            "body_sha256s": dict(plan.V1_FAILED_BODY_SHA256S),
            "failure_stage": "materialize_inputs",
            "input_authority_opened": False,
            "checkpoint_opened": False,
            "cuda_initialized": False,
            "forward_count": 0,
        }
    finally:
        os.close(directory_fd)


def validate_environment(*, gpu_index: int) -> dict[str, str]:
    _require(gpu_index in (0, 1), "activity-only V2 GPU index must be 0 or 1")
    expected = {
        **plan.EXACT_DATA_ROOT_ENV,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
    }
    for name, value in expected.items():
        _require(os.environ.get(name) == value, f"activity-only V2 environment drift: {name}")
    return expected


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).encode("utf-8") + b"\n"


def _publish(path: Path, payload: Mapping[str, object]) -> str:
    body = _json_bytes(payload)
    digest = hashlib.sha256(body).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, body)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    sidecar = path.with_name(path.name + ".sha256")
    descriptor = os.open(sidecar, flags, 0o444)
    try:
        os.write(descriptor, f"{digest}  {path.name}\n".encode("ascii"))
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def execute(root: Path, *, gpu_index: int = 1) -> Mapping[str, object]:
    base = Path(root).absolute()
    output = base / plan.RESULT_ROOT_RELATIVE
    environment = validate_environment(gpu_index=gpu_index)
    failed_v1 = validate_failed_v1(base)
    _require(not output.exists(), "activity-only V2 result root already exists")
    profile = score_plan.COMPATIBLE_DEVICE_PROFILES[f"gpu{gpu_index}"]
    binding = predecessor_score.validate_completed_v8_predecessor(base)
    identity = predecessor_score.build_reviewed_identity(base, selected_device_profile=profile)
    owned = plan.owned_sha256s(base)
    output.mkdir(mode=0o755, parents=False, exist_ok=False)
    attempt = {
        "schema": "causal_dual_memory_cell_d_activity_only_quick_attempt_v2",
        "status": "ATTEMPT_RESERVED",
        "cell": plan.CELL,
        "identity_sha256": identity.sha256,
        "v1_failed_predecessor": failed_v1,
        "v8_predecessor": binding.payload(),
        "owned_sha256s": owned,
        "environment": environment,
        "selected_device_profile": dict(profile),
        "budgets": [10, 4],
        "surfaces": ["within", "external"],
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = _publish(output / "attempt.json", attempt)
    runtime: v1quick.ActivityOnlyRuntime | None = None
    input_sha: str | None = None
    stage = "prepare"
    started = time.monotonic()
    try:
        runtime = v1quick.ActivityOnlyRuntime(root=base, selected_device_profile=profile)
        runtime.prepare(identity=identity)
        stage = "materialize_inputs"
        fixed = score_contract.derive_fixed_evaluation_authority(base)
        authority = runtime.materialize_inputs(identity=identity, authority=fixed)
        input_payload = authority.payload(identity=identity)
        predecessor_score.validate_v8_input_equivalence(input_payload, identity)
        input_sha = hashlib.sha256(plan.canonical_json_bytes(input_payload)).hexdigest()
        stage = "score"
        cells: list[Mapping[str, object]] = []
        for budget in (10, 4):
            cells.extend(runtime.score_activity_budget(budget=budget, input_authority_sha256=input_sha))
        summary = v1quick.summarize(cells=cells, binding=binding, input_sha256=input_sha)
        result = {
            "schema": "causal_dual_memory_cell_d_activity_only_quick_result_v2",
            "status": "TERMINAL",
            "scope": "non_governing_matched_engineering_screen",
            "attempt_sha256": attempt_sha,
            "input_authority_sha256": input_sha,
            "v1_failed_predecessor": failed_v1,
            "cells": cells,
            "summary": summary,
            "wall_seconds": float(time.monotonic() - started),
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "model_or_checkpoint_updated": False,
            "target_optimizer_backward_update": 0,
            "carrier_proposal_attempt_count": 0,
        }
        result_sha = _publish(output / "result.json", result)
        stage = "terminal_revalidation"
        _require(validate_environment(gpu_index=gpu_index) == environment, "activity-only V2 final environment drift")
        _require(validate_failed_v1(base) == failed_v1, "activity-only V2 final predecessor drift")
        terminal = {
            "schema": "causal_dual_memory_cell_d_activity_only_quick_terminal_v2",
            "status": "TERMINAL",
            "attempt_sha256": attempt_sha,
            "input_authority_sha256": input_sha,
            "result_sha256": result_sha,
            "verdict": summary["verdict"],
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = _publish(output / "terminal.json", terminal)
        os.chmod(output, 0o555)
        return {"attempt_sha256": attempt_sha, "result_sha256": result_sha, "terminal_sha256": terminal_sha, **summary}
    except BaseException as error:
        failure = {
            "schema": "causal_dual_memory_cell_d_activity_only_quick_failure_v2",
            "status": "FAILED",
            "attempt_sha256": attempt_sha,
            "input_authority_sha256_or_null": input_sha,
            "stage": stage,
            "error_class": type(error).__name__,
            "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            "terminal": False,
        }
        _publish(output / "failure.json", failure)
        os.chmod(output, 0o555)
        raise
    finally:
        if runtime is not None:
            runtime.close()

