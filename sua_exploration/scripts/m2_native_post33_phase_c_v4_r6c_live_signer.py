#!/usr/bin/env python3
"""Single-purpose, live-only r6c signing controller.

This production coordinator repairs the r6b lifecycle defect without making a
new evaluator, fitting path, or statistical gate.  It owns one Ed25519 private
key only as a Python object in its own address space.  The key is generated
once, its public half is written to the r6c trust anchor, and the process then
    fails closed if it dies. It never serializes a private key or accepts one
    through argv, environment, files, or JSON.

Lifecycle (all write-once):

1. `--bootstrap` creates the fresh public anchor and a PID/starttime-bound
   ready receipt, then waits for a r6c plan created by the CPU-only builder.
2. It signs exactly the three predeclared Stage-A capability files.
3. It waits score-blind for exact Stage-A completion, invokes the existing
   delayed Stage-A opener with stdout/stderr discarded, and signs the raw
   decision bytes for *both* decision branches.
4. A separate no-output gate process returns only an exit status.  Only a
   validated `continue_without_positive_claim` causes exact Stage-B GPU0/GPU1
   and full-opening capabilities to be materialized.  Each binds the decision
   and detached decision-signature metadata.

The controller's own source and the no-output gate are registered Phase-C
runtime roots, so the r6c program receipt seals both source hashes.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import socket
import sys
import time
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[2]
R6C_RECEIPTS = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6c"
R6C_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6c"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_live_signer_20260805"
PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6c_root_ed25519_public.pem"
READY = RUNTIME / "live_signer_ready.json"
PLAN = R6C_RECEIPTS / "signer_requests/r6c_live_signer_plan.json"
STAGE_A_ACK = R6C_RECEIPTS / "signer_acks/stage_a_capabilities_signed.json"
STAGE_B_ACK = R6C_RECEIPTS / "signer_acks/conditional_stage_b_capabilities_signed.json"
STOP_ACK = R6C_RECEIPTS / "signer_acks/stage_a_futility_stop_no_stage_b_capabilities.json"
FAILURE = RUNTIME / "controller_failure.json"
GATE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_continue_gate.py"
OPEN_STAGE_A = ROOT / "sua_exploration/scripts/open_m2_native_post33_phase_c_v4_stage_a.py"
POLL_SECONDS = 0.5

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import m2_native_post33_authorization_v4 as authorization  # noqa: E402
from sua_exploration.mc_maze import m2_native_post33_program_v4 as program_module  # noqa: E402
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    sha256_file,
    stage_a_paths,
    validate_shard_manifest,
    validate_stage_a_cell_directory_set,
    write_bytes_exclusive,
    write_json_exclusive,
)
EXIT_CONTINUE = 0
EXIT_VALIDATED_STOP = 10
EXIT_INVALID = 1


def _proc_starttime(pid: int) -> int:
    """Read Linux proc field 22 without confusing spaces inside comm()."""
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    tail = stat.rsplit(")", 1)[1].strip().split()
    # `tail[0]` is stat field 3; starttime is field 22 -> offset 19.
    if len(tail) <= 19:
        raise RuntimeError("/proc stat lacks starttime field")
    return int(tail[19])


def _proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest()


def _canonical_json(path: Path) -> dict[str, Any]:
    return json.loads(require_canonical_regular_file(path).read_text(encoding="utf-8"))


def _time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise PermissionError("plan time field must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PermissionError("plan time field must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _reload_live_authorization() -> None:
    """Reload the post-bootstrap anchor and expanded r6c source closure.

    The coordinator intentionally starts before the public-anchor literals are
    patched into production source.  Reload only after an atomically published
    plan is visible; a stale r6b module is a hard failure, never a fallback.
    """
    global authorization, program_module
    importlib.invalidate_caches()
    authorization = importlib.reload(authorization)
    program_module = importlib.reload(program_module)


def _metadata(value: Any, expected: Path, label: str) -> Path:
    canonical = require_canonical_regular_file(expected)
    if value != file_metadata(canonical):
        raise PermissionError(f"r6c live-signer {label} metadata mismatch")
    return canonical


def _source_is_sealed(program: Mapping[str, Any], source: Path) -> None:
    relative = source.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix()
    expected = {
        "relative_path": relative,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    rows = program.get("source_map")
    if not isinstance(rows, list) or expected not in rows:
        raise PermissionError(f"r6c program receipt does not seal {relative}")


def _assert_live_binding(ready: Mapping[str, Any]) -> None:
    """Fail closed on PID reuse, source swap, or command-line substitution."""
    if ready.get("schema") != "m2_post33_phase_c_v4_r6c_live_signer_ready_v1":
        raise PermissionError("live signer ready schema mismatch")
    pid = ready.get("pid")
    starttime = ready.get("proc_starttime_ticks")
    command = ready.get("proc_cmdline_sha256")
    if (
        pid != os.getpid()
        or not isinstance(starttime, int)
        or not isinstance(command, str)
        or ready.get("host_id") != socket.gethostname()
    ):
        raise PermissionError("live signer PID binding mismatch")
    try:
        observed_starttime = _proc_starttime(pid)
        observed_command = _proc_cmdline_sha256(pid)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise PermissionError("live signer process is no longer observable") from exc
    if observed_starttime != starttime or observed_command != command:
        raise PermissionError("live signer PID/starttime/command binding drift")
    if ready.get("controller_source") != file_metadata(Path(__file__).resolve()):
        raise PermissionError("live signer source hash drift")
    if ready.get("public_key") != file_metadata(PUBLIC_KEY):
        raise PermissionError("live signer public key hash drift")


def _assert_current_authorization_window(body: Mapping[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    issued, expires = _parse_time(body.get("issued_at")), _parse_time(body.get("expires_at"))
    if not (issued <= now <= expires):
        raise PermissionError("authorization is not live at signing time")


def _write_event(name: str, payload: Mapping[str, Any]) -> Path:
    return write_json_exclusive(RUNTIME / "events" / name, dict(payload))


def _child_environment() -> dict[str, str]:
    """Fixed, key-free environment for a separately spawned helper."""
    return {
        "PYTHONPATH": str(ROOT),
        "PYTHONNOUSERSITE": "1",
    }


def _run_score_blind_child(argv: list[str]) -> int:
    """Use `os.posix_spawn` for an external no-key helper.

    The controller supplies no private-key bytes through argv, environment,
    files, stdin, or IPC. The protocol makes no stronger claim about every OS
    implementation detail than this explicit non-supply boundary.
    """
    if not argv or Path(argv[0]).resolve(strict=True) != Path(sys.executable).resolve(strict=True):
        raise PermissionError("score-blind helper must use this exact Python executable")
    devnull = os.open(os.devnull, os.O_RDWR)
    try:
        actions = [
            (os.POSIX_SPAWN_DUP2, devnull, 0),
            (os.POSIX_SPAWN_DUP2, devnull, 1),
            (os.POSIX_SPAWN_DUP2, devnull, 2),
        ]
        pid = os.posix_spawn(argv[0], argv, _child_environment(), file_actions=actions)
    finally:
        os.close(devnull)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    raise RuntimeError("score-blind helper ended without an exit status")


def _expected_paths(plan: Mapping[str, Any]) -> dict[str, Path]:
    required = {
        "schema", "receipt_root", "cell_root", "ready", "public_key", "program", "portable",
        "cost_supplement", "controller_source", "continue_gate_source", "issued_at", "expires_at",
        "stage_a", "conditional_targets",
    }
    if set(plan) != required or plan.get("schema") != "m2_post33_phase_c_v4_r6c_live_signer_plan_v1":
        raise PermissionError("r6c live-signer plan exact schema mismatch")
    if plan.get("receipt_root") != str(R6C_RECEIPTS.resolve()) or plan.get("cell_root") != str(R6C_CELL_ROOT.resolve()):
        raise PermissionError("r6c plan root substitution")
    _metadata(plan.get("ready"), READY, "ready receipt")
    _metadata(plan.get("public_key"), PUBLIC_KEY, "public anchor")
    _metadata(plan.get("controller_source"), Path(__file__).resolve(), "controller source")
    _metadata(plan.get("continue_gate_source"), GATE, "continue-gate source")
    program = _metadata(plan.get("program"), R6C_RECEIPTS / "program/phase_c_program_r6c.json", "program")
    portable = _metadata(plan.get("portable"), R6C_RECEIPTS / "manifest/portable_r6c.json", "portable")
    cost = _metadata(
        plan.get("cost_supplement"),
        ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3/cost/cost_supplement_r3.json",
        "cost supplement",
    )
    program_payload = program_module.validate_phase_c_program_receipt(program)
    _source_is_sealed(program_payload, Path(__file__).resolve())
    _source_is_sealed(program_payload, GATE)
    if authorization.PUBLIC_KEY != PUBLIC_KEY or authorization.PUBLIC_KEY_SHA256 != sha256_file(PUBLIC_KEY):
        raise PermissionError("production verifier is not pinned to live r6c anchor")
    issued, expires = _parse_time(plan["issued_at"]), _parse_time(plan["expires_at"])
    if expires <= issued or (expires - issued).total_seconds() > authorization.MAX_VALIDITY_SECONDS:
        raise PermissionError("r6c plan validity is invalid")
    return {"program": program, "portable": portable, "cost": cost}


def _assert_stage_a_auth(path: Path, *, plan: Mapping[str, Any], shard: Path, paths: Mapping[str, Path]) -> None:
    envelope = _canonical_json(path)
    body = envelope.get("authorization") if isinstance(envelope, Mapping) else None
    if not isinstance(body, Mapping):
        raise PermissionError("Stage-A authorization envelope malformed")
    expected = {
        "stage_a_execution_gpu0": ("stage_a", "cell_execution", [0, 2, 4, 6]),
        "stage_a_execution_gpu1": ("stage_a", "cell_execution", [1, 3, 5]),
        "stage_a_opening": ("stage_a_opening", "opening", list(range(7))),
    }
    matched = next((name for name in expected if path.name == f"{name}_r6c.json"), None)
    if matched is None:
        raise PermissionError("unexpected Stage-A authorization filename")
    stage, scope, folds = expected[matched]
    if (
        body.get("stage") != stage
        or body.get("capability_scope") != scope
        or body.get("fold_allowlist") != folds
        or body.get("seed_allowlist") != [42]
        or body.get("absolute_cell_root") != str(R6C_CELL_ROOT.resolve())
        or body.get("phase_c_program_receipt") != file_metadata(paths["program"])
        or body.get("portable_manifest") != file_metadata(paths["portable"])
        or body.get("shard_manifest") != file_metadata(shard)
        or body.get("public_key") != file_metadata(PUBLIC_KEY)
        or body.get("issued_at") != plan["issued_at"]
        or body.get("expires_at") != plan["expires_at"]
        or "stage_a_decision" in body
        or "stage_a_decision_signature" in body
    ):
        raise PermissionError("Stage-A authorization scope/binding mismatch")


def _sign_stage_a(plan: Mapping[str, Any], private_key: Ed25519PrivateKey, ready: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    stage_a = plan.get("stage_a")
    if not isinstance(stage_a, Mapping) or set(stage_a) != {"gpu0", "gpu1", "opening"}:
        raise PermissionError("Stage-A plan scope mismatch")
    expected = {
        "gpu0": ("stage_a_execution_gpu0_r6c.json", "shard_stage_a_gpu0_r6c.json"),
        "gpu1": ("stage_a_execution_gpu1_r6c.json", "shard_stage_a_gpu1_r6c.json"),
        "opening": ("stage_a_opening_r6c.json", "shard_stage_a_opening_r6c.json"),
    }
    signed: dict[str, Any] = {}
    for name, (auth_name, shard_name) in expected.items():
        row = stage_a.get(name)
        if not isinstance(row, Mapping) or set(row) != {"authorization", "shard_manifest"}:
            raise PermissionError("Stage-A signing request exact key mismatch")
        auth_path = R6C_RECEIPTS / "auth" / auth_name
        shard_path = R6C_RECEIPTS / "manifest" / shard_name
        _metadata(row["authorization"], auth_path, f"Stage-A {name} authorization")
        _metadata(row["shard_manifest"], shard_path, f"Stage-A {name} shard")
        validate_shard_manifest(shard_path, portable_manifest_path=paths["portable"], cell_root=R6C_CELL_ROOT)
        _assert_stage_a_auth(auth_path, plan=plan, shard=shard_path, paths=paths)
        body = _canonical_json(auth_path)["authorization"]
        _assert_current_authorization_window(body)
        sig = auth_path.with_suffix(".sig")
        if sig.exists():
            raise FileExistsError("Stage-A detached signature already exists")
        write_bytes_exclusive(sig, base64.b64encode(private_key.sign(auth_path.read_bytes())))
        authorization.verify_signed_authorization(
            auth_path, sig,
            phase_c_program_receipt_path=paths["program"], portable_manifest_path=paths["portable"],
            shard_manifest_path=shard_path, cost_supplement_path=paths["cost"], cell_root=R6C_CELL_ROOT,
            now=datetime.now(timezone.utc),
        )
        signed[name] = {"authorization": file_metadata(auth_path), "signature": file_metadata(sig)}
    _assert_live_binding(ready)
    write_json_exclusive(STAGE_A_ACK, {
        "schema": "m2_post33_phase_c_v4_r6c_stage_a_capabilities_signed_v1",
        "ready": file_metadata(READY),
        "program": file_metadata(paths["program"]),
        "stage_a": signed,
        "private_key_serialized_or_disk_persisted": False,
        "private_key_not_serialized": True,
        "private_key_not_supplied_via_argv_environment_file_or_ipc": True,
        "private_key_retained_live_in_memory": True,
        "score_data_accessed_before_stage_a_capabilities_signed": False,
        "helper_spawn_api": "os.posix_spawn",
    })


def _open_and_sign_decision(plan: Mapping[str, Any], private_key: Ed25519PrivateKey, ready: Mapping[str, Any], paths: Mapping[str, Path]) -> Path:
    # This validator establishes the exact-14, score-sealed topology before the
    # existing opener is allowed to read any endpoint payload.  It returns no
    # endpoint values to this controller.
    validate_stage_a_cell_directory_set(R6C_CELL_ROOT)
    _write_event("exact14_score_sealed_stage_a_observed.json", {
        "schema": "m2_post33_phase_c_v4_r6c_live_signer_event_v1",
        "event": "exact14_score_sealed_stage_a_observed",
        "stage_a_completed": file_metadata(stage_a_paths(R6C_CELL_ROOT)["completed"]),
        "controller_read_score_values": False,
    })
    stage_a = plan["stage_a"]
    opening = stage_a["opening"]
    auth = Path(str(opening["authorization"]["canonical_path"])).resolve(strict=True)
    shard = Path(str(opening["shard_manifest"]["canonical_path"])).resolve(strict=True)
    command = [
        sys.executable, str(OPEN_STAGE_A), "--root", str(R6C_CELL_ROOT),
        "--program-receipt", str(paths["program"]), "--portable-manifest", str(paths["portable"]),
        "--shard-manifest", str(shard), "--cost-supplement", str(paths["cost"]),
        "--authorization", str(auth), "--authorization-signature", str(auth.with_suffix(".sig")),
    ]
    _assert_live_binding(ready)
    if _run_score_blind_child(command) != 0:
        raise RuntimeError("existing Stage-A delayed opener failed without exposing output")
    decision = require_canonical_regular_file(stage_a_paths(R6C_CELL_ROOT)["decision"], within=R6C_CELL_ROOT)
    _assert_live_binding(ready)
    signature = stage_a_paths(R6C_CELL_ROOT)["decision_signature"]
    if signature.exists():
        raise FileExistsError("Stage-A decision signature already exists")
    # Crucially this signs opaque decision bytes before any validation/class
    # branch. The controller does not deserialize or log the decision payload.
    write_bytes_exclusive(signature, base64.b64encode(private_key.sign(decision.read_bytes())))
    _write_event("stage_a_decision_signed_unconditionally.json", {
        "schema": "m2_post33_phase_c_v4_r6c_live_signer_event_v1",
        "event": "stage_a_decision_signed_unconditionally",
        "decision": file_metadata(decision),
        "decision_signature": file_metadata(signature),
        "controller_read_score_values": False,
        "score_opening_was_external_no_key_posix_spawn_helper": True,
        "helper_spawn_api": "os.posix_spawn",
    })
    return decision


def _decision_gate() -> str:
    """Classify an already signed decision using strict score-blind exit codes."""
    code = _run_score_blind_child(
        [sys.executable, str(GATE), "--root", str(R6C_CELL_ROOT), "--classify-signed"]
    )
    if code == EXIT_CONTINUE:
        return "continue"
    if code == EXIT_VALIDATED_STOP:
        return "validated_stop"
    # Invalid policy, signature, helper error, or signal is never a result.
    raise PermissionError(f"Stage-A signed decision gate failed closed (exit={code})")


def _issue_one_conditional(
    *,
    base_auth: Path,
    output: Path,
    shard: Path,
    stage: str,
    scope: str,
    expected_folds: list[int],
    expected_seeds: list[int],
    authorization_id: str,
    plan: Mapping[str, Any],
    private_key: Ed25519PrivateKey,
    paths: Mapping[str, Path],
    decision: Path,
) -> dict[str, Any]:
    if output.exists() or output.with_suffix(".sig").exists():
        raise FileExistsError("conditional authorization already materialized")
    validate_shard_manifest(shard, portable_manifest_path=paths["portable"], cell_root=R6C_CELL_ROOT)
    shard_payload = _canonical_json(shard)
    if shard_payload.get("fold_allowlist") != expected_folds or shard_payload.get("seed_allowlist") != expected_seeds:
        raise PermissionError("conditional shard scope mismatch")
    envelope = _canonical_json(base_auth)
    body = dict(envelope["authorization"])
    body.update({
        "authorization_id": authorization_id,
        "single_use_nonce": secrets.token_hex(32),
        "issued_at": plan["issued_at"],
        "expires_at": plan["expires_at"],
        "stage": stage,
        "capability_scope": scope,
        "host_id": shard_payload["host_id"],
        "gpu_id": shard_payload["gpu_id"],
        "fold_allowlist": expected_folds,
        "seed_allowlist": expected_seeds,
        "shard_manifest": file_metadata(shard),
        "stage_a_decision": file_metadata(decision),
        "stage_a_decision_signature": file_metadata(stage_a_paths(R6C_CELL_ROOT)["decision_signature"]),
    })
    _assert_current_authorization_window(body)
    write_json_exclusive(output, {
        "schema": "m2_post33_phase_c_signed_authorization_envelope_v4", "authorization": body,
    })
    signature = output.with_suffix(".sig")
    write_bytes_exclusive(signature, base64.b64encode(private_key.sign(output.read_bytes())))
    authorization.verify_signed_authorization(
        output, signature, phase_c_program_receipt_path=paths["program"],
        portable_manifest_path=paths["portable"], shard_manifest_path=shard,
        cost_supplement_path=paths["cost"], cell_root=R6C_CELL_ROOT,
        now=datetime.now(timezone.utc),
    )
    return {"authorization": file_metadata(output), "signature": file_metadata(signature), "shard": file_metadata(shard)}


def _issue_conditional_capabilities(plan: Mapping[str, Any], private_key: Ed25519PrivateKey, ready: Mapping[str, Any], paths: Mapping[str, Path], decision: Path) -> None:
    targets = plan.get("conditional_targets")
    if not isinstance(targets, Mapping) or set(targets) != {"gpu0", "gpu1", "full_opening"}:
        raise PermissionError("conditional target exact set mismatch")
    expected = {
        "gpu0": ("stage_a_execution_gpu0_r6c.json", "stage_b_execution_gpu0_r6c.json", "shard_stage_b_gpu0_r6c.json", "stage_b", "cell_execution", [0, 2, 4, 6], [43, 44]),
        "gpu1": ("stage_a_execution_gpu1_r6c.json", "stage_b_execution_gpu1_r6c.json", "shard_stage_b_gpu1_r6c.json", "stage_b", "cell_execution", [1, 3, 5], [43, 44]),
        "full_opening": ("stage_a_opening_r6c.json", "full_opening_r6c.json", "shard_full_opening_r6c.json", "full_opening", "opening", list(range(7)), [42, 43, 44]),
    }
    result: dict[str, Any] = {}
    for name, values in expected.items():
        base_name, output_name, shard_name, stage, scope, folds, seeds = values
        row = targets[name]
        if not isinstance(row, Mapping) or set(row) != {"authorization_path", "shard_manifest"}:
            raise PermissionError("conditional target schema mismatch")
        output = R6C_RECEIPTS / "auth" / output_name
        shard = R6C_RECEIPTS / "manifest" / shard_name
        if row.get("authorization_path") != str(output.resolve()):
            raise PermissionError("conditional authorization path substitution")
        _metadata(row.get("shard_manifest"), shard, f"conditional {name} shard")
        result[name] = _issue_one_conditional(
            base_auth=R6C_RECEIPTS / "auth" / base_name, output=output, shard=shard,
            stage=stage, scope=scope, expected_folds=folds, expected_seeds=seeds,
            authorization_id=f"m2-post33-phase-c-v4-{stage}_{name}_r6c-20260805",
            plan=plan, private_key=private_key, paths=paths, decision=decision,
        )
    _assert_live_binding(ready)
    write_json_exclusive(STAGE_B_ACK, {
        "schema": "m2_post33_phase_c_v4_r6c_conditional_capabilities_signed_v1",
        "ready": file_metadata(READY),
        "decision": file_metadata(decision),
        "decision_signature": file_metadata(stage_a_paths(R6C_CELL_ROOT)["decision_signature"]),
        "conditional_capabilities": result,
        "stage_b_issued_only_after_continue": True,
        "private_key_serialized_or_disk_persisted": False,
        "private_key_not_serialized": True,
        "private_key_not_supplied_via_argv_environment_file_or_ipc": True,
        "private_key_retained_live_in_memory": True,
        "controller_read_score_values": False,
        "helper_spawn_api": "os.posix_spawn",
    })


def _wait_for(path: Path, *, label: str) -> None:
    while True:
        if not path.exists():
            time.sleep(POLL_SECONDS)
            continue
        try:
            first = file_metadata(require_canonical_regular_file(path))
            time.sleep(POLL_SECONDS)
            second = file_metadata(require_canonical_regular_file(path))
            if first == second:
                return
        except (FileNotFoundError, OSError, ValueError):
            pass
        time.sleep(POLL_SECONDS)


def _read_stable_plan() -> dict[str, Any]:
    """Tolerate a writer's visible-but-incomplete O_EXCL file without racing."""
    while True:
        if not PLAN.exists():
            time.sleep(POLL_SECONDS)
            continue
        try:
            first = file_metadata(require_canonical_regular_file(PLAN, within=R6C_RECEIPTS))
            candidate = _canonical_json(PLAN)
            time.sleep(POLL_SECONDS)
            second = file_metadata(require_canonical_regular_file(PLAN, within=R6C_RECEIPTS))
            if first == second:
                return candidate
        except (json.JSONDecodeError, FileNotFoundError, OSError, ValueError):
            pass
        time.sleep(POLL_SECONDS)


def _failure(exc: BaseException) -> None:
    if not FAILURE.exists():
        write_json_exclusive(FAILURE, {
            "schema": "m2_post33_phase_c_v4_r6c_live_signer_failure_v1",
            "failure_class": type(exc).__name__,
            "score_or_decision_content_logged": False,
            "private_key_serialized_or_disk_persisted": False,
            "private_key_not_serialized": True,
            "private_key_not_supplied_via_argv_environment_file_or_ipc": True,
            "private_key_retained_live_in_memory": True,
            "fail_closed": True,
        })


def bootstrap() -> None:
    if any(path.exists() for path in (PUBLIC_KEY, RUNTIME, R6C_RECEIPTS, R6C_CELL_ROOT)):
        raise FileExistsError("r6c live signer requires wholly absent fresh anchor/runtime/roots")
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    write_bytes_exclusive(PUBLIC_KEY, public_bytes)
    ready = {
        "schema": "m2_post33_phase_c_v4_r6c_live_signer_ready_v1",
        "status": "LIVE_PRIVATE_KEY_IN_MEMORY_ONLY",
        "host_id": socket.gethostname(),
        "pid": os.getpid(),
        "proc_starttime_ticks": _proc_starttime(os.getpid()),
        "proc_cmdline_sha256": _proc_cmdline_sha256(os.getpid()),
        "controller_source": file_metadata(Path(__file__).resolve()),
        "public_key": file_metadata(PUBLIC_KEY),
        "intended_receipt_root": str(R6C_RECEIPTS.resolve()),
        "intended_cell_root": str(R6C_CELL_ROOT.resolve()),
        "private_key_serialized_or_disk_persisted": False,
        "private_key_not_serialized": True,
        "private_key_not_supplied_via_argv_environment_file_or_ipc": True,
        "private_key_retained_live_in_memory": True,
        "private_key_retained_live_in_memory_until_terminal_exit": True,
        "private_key_in_argv": False,
        "private_key_in_environment": False,
        "stage_b_authorizations_materialized": False,
        "full_opening_authorization_materialized": False,
    }
    write_json_exclusive(READY, ready)
    try:
        _assert_live_binding(ready)
        plan = _read_stable_plan()
        _reload_live_authorization()
        paths = _expected_paths(plan)
        _assert_live_binding(ready)
        _sign_stage_a(plan, private_key, ready, paths)
        _wait_for(stage_a_paths(R6C_CELL_ROOT)["completed"], label="exact-14 Stage-A completion")
        decision = _open_and_sign_decision(plan, private_key, ready, paths)
        gate_outcome = _decision_gate()
        if gate_outcome == "continue":
            _assert_live_binding(ready)
            _issue_conditional_capabilities(plan, private_key, ready, paths, decision)
        elif gate_outcome == "validated_stop":
            # The decision has already been signed.  This branch merely records
            # that no Stage-B/full-opening capability was ever materialized.
            write_json_exclusive(STOP_ACK, {
                "schema": "m2_post33_phase_c_v4_r6c_stage_a_futility_stop_v1",
                "ready": file_metadata(READY),
                "decision": file_metadata(decision),
                "decision_signature": file_metadata(stage_a_paths(R6C_CELL_ROOT)["decision_signature"]),
                "stage_b_authorizations_materialized": False,
                "full_opening_authorization_materialized": False,
                "controller_read_score_values": False,
                "score_opening_was_external_no_key_posix_spawn_helper": True,
            })
        else:
            raise AssertionError("unreachable decision-gate outcome")
    except BaseException as exc:
        _failure(exc)
        raise
    finally:
        # Drop the only Python reference on every normal/exceptional exit.
        private_key = None  # type: ignore[assignment]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    if not args.bootstrap:
        raise SystemExit("only --bootstrap is supported; the signer never imports a private key")
    bootstrap()


if __name__ == "__main__":
    main()
