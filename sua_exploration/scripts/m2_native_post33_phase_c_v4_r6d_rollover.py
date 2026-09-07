#!/usr/bin/env python3
"""CPU-only, preflight-first r6d fresh-root builder.

This program has no private key and cannot sign or launch a GPU.  Unlike r6c,
every external/history/absence check and a disposable scratch-manifest
validation complete before the first r6d receipt is written.  Static artifacts
are then written in dependency order; the final trigger is a one-shot O_EXCL
link operation after the source-sealed launch, core, and proof already exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
BUILDER = Path(__file__).resolve()
R3 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3"
R6C = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6c"
R6C_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6c"
R6C_RETIREMENT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_retirement_20260805/r6c_pre_gpu_retirement.json"
R6C_RETIREMENT_SHA256 = "22ef77ce1a697a9901c85235b2233c820a5d7015d884fb9b25e463272c89448a"
R6D = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6d"
R6D_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6d"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6d_live_signer_20260805"
READY = RUNTIME / "live_signer_ready.json"
PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6d_root_ed25519_public.pem"
CONTROLLER = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_live_signer.py"
GATE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_continue_gate.py"
SUPERVISOR = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_supervisor.py"
ASSERT_LIVE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_assert_live.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
EOF_RECEIPT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_upstream_canonicalization_20260805/upstream_eof_canonicalization.json"
DEEP_SOURCE_AUDIT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_deep_source_20260805/deep_source_audit.json"
COST = R3 / "cost/cost_supplement_r3.json"
DATA = ROOT / "SPINT-main/data/000953"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "sua_exploration/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sua_exploration.mc_maze import m2_native_post33_authorization_v4 as authorization  # noqa: E402
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    sha256_file,
    validate_portable_transfer_manifest,
    validate_shard_manifest,
    write_bytes_exclusive,
    write_json_exclusive,
)
from sua_exploration.mc_maze.m2_native_post33_program_v4 import (  # noqa: E402
    build_phase_c_program_receipt,
    validate_phase_c_program_receipt,
)
import m2_native_post33_phase_c_v4_r6d_assert_live as watchdog  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    return json.loads(require_canonical_regular_file(path).read_text(encoding="utf-8"))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _predicted_metadata(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    data = _json_bytes(payload)
    return {
        "canonical_path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _proc_starttime(pid: int) -> int:
    tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
    if len(tail) <= 19:
        raise PermissionError("live signer /proc starttime unavailable")
    return int(tail[19])


def _proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest()


def _assert_runtime_exact_ready() -> dict[str, Any]:
    if not RUNTIME.is_dir() or RUNTIME.is_symlink():
        raise PermissionError("r6d runtime must be a real directory")
    entries = sorted(item.name for item in RUNTIME.iterdir())
    if entries != [READY.name]:
        raise PermissionError(f"r6d runtime must contain only READY before trigger: {entries}")
    ready = _json(READY)
    if (
        ready.get("schema") != "m2_post33_phase_c_v4_r6d_live_signer_ready_v1"
        or ready.get("status") != "LIVE_PRIVATE_KEY_IN_MEMORY_ONLY"
        or ready.get("host_id") != socket.gethostname()
        or ready.get("intended_receipt_root") != str(R6D.resolve())
        or ready.get("intended_cell_root") != str(R6D_ROOT.resolve())
        or ready.get("controller_source") != file_metadata(CONTROLLER)
        or ready.get("public_key") != file_metadata(PUBLIC_KEY)
        or ready.get("private_key_not_serialized") is not True
        or ready.get("private_key_retained_live_in_memory") is not True
        or ready.get("private_key_not_supplied_via_argv_environment_file_or_ipc") is not True
    ):
        raise PermissionError("r6d ready receipt binding mismatch")
    pid = ready.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        raise PermissionError("r6d live signer PID invalid")
    try:
        os.kill(pid, 0)
        if _proc_starttime(pid) != ready.get("proc_starttime_ticks"):
            raise PermissionError("r6d ready signer starttime mismatch")
        if _proc_cmdline_sha256(pid) != ready.get("proc_cmdline_sha256"):
            raise PermissionError("r6d ready signer command mismatch")
    except ProcessLookupError as exc:
        raise PermissionError("r6d live signer is dead") from exc
    return ready


def _assert_checker_live() -> None:
    result = subprocess.run(
        list(watchdog.check_once_argv_template()), cwd=ROOT, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode != 0:
        raise PermissionError("r6d independent checker rejected live signer")


def _assert_r6c_retirement() -> dict[str, Any]:
    if sha256_file(R6C_RETIREMENT) != R6C_RETIREMENT_SHA256:
        raise PermissionError("r6c retirement receipt hash mismatch")
    payload = _json(R6C_RETIREMENT)
    if (
        payload.get("schema") != "m2_post33_phase_c_v4_r6c_pre_gpu_retirement_v1"
        or payload.get("disposition") != "RETIRED_BEFORE_GPU_OR_DATA_ACCESS"
        or payload.get("gpu_or_data_worker_started") is not False
        or payload.get("endpoint_or_score_opened") is not False
        or payload.get("r6c_mutation_or_reuse_forbidden") is not True
        or payload.get("signer_process_alive_after_retirement") is not False
        or payload.get("fresh_r6d_root_anchor_and_source_closure_required") is not True
    ):
        raise PermissionError("r6c retirement disposition binding mismatch")
    retired_pid = payload.get("signer_pid")
    if not isinstance(retired_pid, int) or retired_pid <= 1:
        raise PermissionError("r6c retirement signer PID is invalid")
    if Path(f"/proc/{retired_pid}").exists():
        raise PermissionError("retired r6c signer PID remains observable")
    if R6C_ROOT.exists() or R6C_ROOT.is_symlink():
        raise PermissionError("retired r6c cell root was recreated")
    retired_execution_artifacts = [
        R6C / "launch" / f"stage_a_gpu{gpu}_{kind}"
        for gpu in (0, 1)
        for kind in (
            "execution_reserved.json", "execution_started.json", "execution_completed.json",
            "execution_failed.json", "matrix.stdout.log", "matrix.stderr.log",
        )
    ]
    if any(path.exists() or path.is_symlink() for path in retired_execution_artifacts):
        raise PermissionError("retired r6c execution artifact was recreated")
    return payload


def _source_rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("source_map")
    if not isinstance(rows, list):
        raise ValueError("program source map missing")
    result = {str(row.get("relative_path")): row for row in rows if isinstance(row, Mapping)}
    if len(result) != len(rows):
        raise ValueError("program source map malformed")
    return result


def _program_delta(r6c: Mapping[str, Any], r6d: Mapping[str, Any]) -> dict[str, Any]:
    for field in (
        "schema", "protocol_id", "phase_id", "absolute_workspace_root", "score_data_accessed",
        "formal_data_accessed", "gpu_used", "phase_a_b_eof_canonicalization", "deep_source_audit_receipt",
    ):
        if r6c.get(field) != r6d.get(field):
            raise ValueError(f"r6d changed program semantic field {field}")
    before, after = _source_rows(r6c), _source_rows(r6d)
    added, removed = sorted(set(after) - set(before)), sorted(set(before) - set(after))
    changed = sorted(name for name in set(before) & set(after) if before[name] != after[name])
    r6c_roots = {
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_live_signer.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_continue_gate.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_supervisor.py",
    }
    r6d_roots = {path.replace("r6c", "r6d") for path in r6c_roots}
    expected_changed = {
        "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py",
        "sua_exploration/mc_maze/m2_native_post33_program_v4.py",
    }
    if set(added) != r6d_roots or set(removed) != r6c_roots or set(changed) != expected_changed:
        raise ValueError(f"unexpected r6d source delta: added={added}, removed={removed}, changed={changed}")
    return {"added_runtime_roots": added, "retired_runtime_roots": removed, "changed_governance_sources": changed}


def _portable(template: Mapping[str, Any], *, program: Mapping[str, Any], cell_root: Path) -> dict[str, Any]:
    payload = dict(template)
    payload["absolute_cell_root"] = str(cell_root.resolve())
    rows = [dict(row) for row in template["hash_closures"]]
    for row in rows:
        if row.get("role") == "phase_c_program_receipt":
            row.clear()
            row.update({"role": "phase_c_program_receipt", **program})
    payload["hash_closures"] = rows
    return payload


def _shard(template: Mapping[str, Any], *, portable_sha256: str, cell_root: Path, seeds: list[int]) -> dict[str, Any]:
    payload = dict(template)
    payload["absolute_cell_root"] = str(cell_root.resolve())
    payload["portable_transfer_manifest_sha256"] = portable_sha256
    payload["seed_allowlist"] = seeds
    return payload


def _stage_a_auth(
    template: Mapping[str, Any], *, program: Mapping[str, Any], portable: Mapping[str, Any],
    shard: Mapping[str, Any], issued: str, expires: str, nonce: str, cell_root: Path,
) -> dict[str, Any]:
    body = dict(template["authorization"])
    original = str(body["authorization_id"])
    body["authorization_id"] = original.replace("r6c", "r6d")
    if body["authorization_id"] == original:
        raise ValueError("r6c template authorization id lacks r6c marker")
    body.update({
        "single_use_nonce": nonce, "issued_at": issued, "expires_at": expires,
        "absolute_cell_root": str(cell_root.resolve()), "phase_c_program_receipt": program,
        "portable_manifest": portable, "shard_manifest": shard, "public_key": file_metadata(PUBLIC_KEY),
    })
    return {"schema": "m2_post33_phase_c_signed_authorization_envelope_v4", "authorization": body}


def _atomic_publish_json(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(f"r6d trigger already exists: {path}")
    data = _json_bytes(payload)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    write_bytes_exclusive(temporary, data)
    try:
        os.link(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=False)
    return path.resolve(strict=True)


@dataclass(frozen=True)
class R6DStaticPlan:
    program: Mapping[str, Any]
    portable: Mapping[str, Any]
    shards: Mapping[str, Mapping[str, Any]]
    authorizations: Mapping[str, Mapping[str, Any]]
    launch: Mapping[str, Any]
    core: Mapping[str, Any]
    proof: Mapping[str, Any]
    trigger: Mapping[str, Any]


def _scratch_validate(
    *, program: Mapping[str, Any], portable_template: Mapping[str, Any], shard_templates: Mapping[str, Mapping[str, Any]]
) -> None:
    """Validate transformed manifest shapes outside r6d's fresh target root."""
    with tempfile.TemporaryDirectory(prefix="spint_r6d_preflight_") as temporary:
        scratch = Path(temporary)
        scratch_cell = scratch / "cells"
        program_path = scratch / "program.json"
        write_json_exclusive(program_path, program)
        validate_phase_c_program_receipt(program_path)
        portable = _portable(portable_template, program=file_metadata(program_path), cell_root=scratch_cell)
        portable_path = scratch / "portable.json"
        write_json_exclusive(portable_path, portable)
        validate_portable_transfer_manifest(
            portable_path, workspace_root=ROOT, data_root=DATA, cell_root=scratch_cell
        )
        for name, template in shard_templates.items():
            seeds = [42] if name in {"gpu0", "gpu1", "opening"} else ([43, 44] if name.startswith("stage_b") else [42, 43, 44])
            payload = _shard(template, portable_sha256=sha256_file(portable_path), cell_root=scratch_cell, seeds=seeds)
            path = scratch / f"{name}.json"
            write_json_exclusive(path, payload)
            validate_shard_manifest(path, portable_manifest_path=portable_path, cell_root=scratch_cell)


def preflight_r6d() -> R6DStaticPlan:
    """Perform every external/history/absence check before any r6d root write."""
    if R6D.exists() or R6D.is_symlink() or R6D_ROOT.exists() or R6D_ROOT.is_symlink():
        raise FileExistsError("r6d receipt/cell roots must be absent before preflight")
    for required in (CONTROLLER, GATE, SUPERVISOR, ASSERT_LIVE, COST, DATA, EOF_RECEIPT, DEEP_SOURCE_AUDIT):
        if required == DATA:
            required.resolve(strict=True)
        else:
            require_canonical_regular_file(required)
    _assert_r6c_retirement()
    ready = _assert_runtime_exact_ready()
    _assert_checker_live()
    if authorization.PUBLIC_KEY != PUBLIC_KEY or authorization.PUBLIC_KEY_SHA256 != sha256_file(PUBLIC_KEY):
        raise PermissionError("r6d production verifier is not pinned to live r6d public anchor")
    r6c_program = _json(R6C / "program/phase_c_program_r6c.json")
    portable_template = _json(R6C / "manifest/portable_r6c.json")
    shard_templates = {
        "gpu0": _json(R6C / "manifest/shard_stage_a_gpu0_r6c.json"),
        "gpu1": _json(R6C / "manifest/shard_stage_a_gpu1_r6c.json"),
        "opening": _json(R6C / "manifest/shard_stage_a_opening_r6c.json"),
    }
    all_templates = {
        **shard_templates,
        "stage_b_gpu0": shard_templates["gpu0"], "stage_b_gpu1": shard_templates["gpu1"],
        "full_opening": shard_templates["opening"],
    }
    auth_templates = {
        "gpu0": _json(R6C / "auth/stage_a_execution_gpu0_r6c.json"),
        "gpu1": _json(R6C / "auth/stage_a_execution_gpu1_r6c.json"),
        "opening": _json(R6C / "auth/stage_a_opening_r6c.json"),
    }
    program = build_phase_c_program_receipt(
        eof_canonicalization_receipt_path=EOF_RECEIPT, deep_source_audit_receipt_path=DEEP_SOURCE_AUDIT,
    )
    _program_delta(r6c_program, program)
    _scratch_validate(program=program, portable_template=portable_template, shard_templates=all_templates)
    issued_dt = datetime.now(timezone.utc).replace(microsecond=0)
    expires_dt = issued_dt + timedelta(hours=29)
    if (expires_dt - issued_dt).total_seconds() > authorization.MAX_VALIDITY_SECONDS:
        raise PermissionError("r6d validity exceeds production maximum")
    issued, expires = _time(issued_dt), _time(expires_dt)
    paths = {
        "program": R6D / "program/phase_c_program_r6d.json",
        "portable": R6D / "manifest/portable_r6d.json",
        "launch": R6D / "launch/stage_a_commands_r6d.json",
        "core": R6D / "signer_requests/r6d_live_signer_plan_core.json",
        "proof": R6D / "r6d_static_pretrigger_evidence.json",
        "trigger": R6D / "signer_requests/r6d_live_signer_trigger.json",
    }
    program_meta = _predicted_metadata(paths["program"], program)
    portable = _portable(portable_template, program=program_meta, cell_root=R6D_ROOT)
    portable_meta = _predicted_metadata(paths["portable"], portable)
    shards = {
        "gpu0": _shard(shard_templates["gpu0"], portable_sha256=portable_meta["sha256"], cell_root=R6D_ROOT, seeds=[42]),
        "gpu1": _shard(shard_templates["gpu1"], portable_sha256=portable_meta["sha256"], cell_root=R6D_ROOT, seeds=[42]),
        "opening": _shard(shard_templates["opening"], portable_sha256=portable_meta["sha256"], cell_root=R6D_ROOT, seeds=[42]),
        "stage_b_gpu0": _shard(shard_templates["gpu0"], portable_sha256=portable_meta["sha256"], cell_root=R6D_ROOT, seeds=[43, 44]),
        "stage_b_gpu1": _shard(shard_templates["gpu1"], portable_sha256=portable_meta["sha256"], cell_root=R6D_ROOT, seeds=[43, 44]),
        "full_opening": _shard(shard_templates["opening"], portable_sha256=portable_meta["sha256"], cell_root=R6D_ROOT, seeds=[42, 43, 44]),
    }
    shard_paths = {name: R6D / "manifest" / f"shard_{'stage_a_' if name in {'gpu0','gpu1','opening'} else ''}{name}_r6d.json" for name in shards}
    shard_paths["gpu0"] = R6D / "manifest/shard_stage_a_gpu0_r6d.json"
    shard_paths["gpu1"] = R6D / "manifest/shard_stage_a_gpu1_r6d.json"
    shard_paths["opening"] = R6D / "manifest/shard_stage_a_opening_r6d.json"
    shard_paths["stage_b_gpu0"] = R6D / "manifest/shard_stage_b_gpu0_r6d.json"
    shard_paths["stage_b_gpu1"] = R6D / "manifest/shard_stage_b_gpu1_r6d.json"
    shard_paths["full_opening"] = R6D / "manifest/shard_full_opening_r6d.json"
    shard_meta = {name: _predicted_metadata(shard_paths[name], payload) for name, payload in shards.items()}
    nonces = [secrets.token_hex(32) for _ in range(3)]
    if len(set(nonces)) != 3:
        raise RuntimeError("r6d Stage-A nonce collision")
    auth_paths = {
        "gpu0": R6D / "auth/stage_a_execution_gpu0_r6d.json",
        "gpu1": R6D / "auth/stage_a_execution_gpu1_r6d.json",
        "opening": R6D / "auth/stage_a_opening_r6d.json",
    }
    authorizations = {
        name: _stage_a_auth(
            auth_templates[name], program=program_meta, portable=portable_meta, shard=shard_meta[name],
            issued=issued, expires=expires, nonce=nonce, cell_root=R6D_ROOT,
        )
        for name, nonce in zip(("gpu0", "gpu1", "opening"), nonces, strict=True)
    }
    auth_meta = {name: _predicted_metadata(auth_paths[name], payload) for name, payload in authorizations.items()}
    contract = {
        "controller": file_metadata(CONTROLLER), "continue_gate": file_metadata(GATE),
        "supervisor": file_metadata(SUPERVISOR), "checker": file_metadata(ASSERT_LIVE),
        "ready": file_metadata(READY), "program": program_meta,
        "check_once_argv_template": list(watchdog.check_once_argv_template()),
        "watch_argv_template": list(watchdog.watch_argv_template()),
        "matrix_must_start_new_session": True, "matrix_process_group_must_equal_matrix_pid": True,
        "matrix_pr_set_pdeathsig": "SIGTERM", "matrix_pdeathsig_parent_recheck": True,
        "watcher_must_not_use_pdeathsig": True, "one_independent_watcher_per_gpu_shard": True,
        "execution_start_receipt_required": True,
    }
    launch = {
        "schema": "m2_post33_phase_c_v4_r6d_stage_a_launch_commands_v1", "status": "PREPARED_NOT_EXECUTED",
        "source_sealed_supervision_contract": contract,
        "commands": {
            name: {
                "gpu_id": index,
                "supervisor_argv": [PYTHON, "-u", str(SUPERVISOR), "--gpu-id", str(index)],
                "authorization": auth_meta[name],
                "authorization_signature_path": str(auth_paths[name].with_suffix(".sig").resolve()),
                "shard_manifest": shard_meta[name], "program": program_meta, "portable": portable_meta,
                "cost_supplement": file_metadata(COST), "fold_allowlist": shards[name]["fold_allowlist"],
                "cuda_visible_devices": shards[name]["gpu_id"],
            }
            for index, name in enumerate(("gpu0", "gpu1"))
        },
        "cell_root_must_be_absent_until_matrix_execution": str(R6D_ROOT.resolve()),
        "gpu_used": False, "score_data_accessed": False,
    }
    launch_meta = _predicted_metadata(paths["launch"], launch)
    core = {
        "schema": "m2_post33_phase_c_v4_r6d_signer_plan_core_v1",
        "receipt_root": str(R6D.resolve()), "cell_root": str(R6D_ROOT.resolve()),
        "ready": file_metadata(READY), "public_key": file_metadata(PUBLIC_KEY), "program": program_meta,
        "portable": portable_meta, "cost_supplement": file_metadata(COST),
        "controller_source": file_metadata(CONTROLLER), "continue_gate_source": file_metadata(GATE),
        "supervisor_source": file_metadata(SUPERVISOR), "assert_live_source": file_metadata(ASSERT_LIVE),
        "stage_a_launch": launch_meta, "issued_at": issued, "expires_at": expires,
        "stage_a": {name: {"authorization": auth_meta[name], "shard_manifest": shard_meta[name]} for name in auth_paths},
        "conditional_targets": {
            name: {"authorization_path": str((R6D / "auth" / filename).resolve()), "shard_manifest": shard_meta[shard_name]}
            for name, filename, shard_name in (
                ("gpu0", "stage_b_execution_gpu0_r6d.json", "stage_b_gpu0"),
                ("gpu1", "stage_b_execution_gpu1_r6d.json", "stage_b_gpu1"),
                ("full_opening", "full_opening_r6d.json", "full_opening"),
            )
        },
    }
    core_meta = _predicted_metadata(paths["core"], core)
    stage_a_absence = {
        name: {"signature_path": str(auth_paths[name].with_suffix(".sig").resolve()), "present_at_pretrigger": False}
        for name in auth_paths
    }
    conditional_absence = {
        name: {
            "authorization_path": str((R6D / "auth" / filename).resolve()), "authorization_present_at_pretrigger": False,
            "signature_path": str((R6D / "auth" / filename).with_suffix(".sig").resolve()),
            "signature_present_at_pretrigger": False,
        }
        for name, filename in (
            ("gpu0", "stage_b_execution_gpu0_r6d.json"),
            ("gpu1", "stage_b_execution_gpu1_r6d.json"),
            ("full_opening", "full_opening_r6d.json"),
        )
    }
    proof = {
        "schema": "m2_post33_phase_c_v4_r6d_static_pretrigger_evidence_v1",
        "r6c_retirement": file_metadata(R6C_RETIREMENT), "r6d_fresh_receipt_root": str(R6D.resolve()),
        "r6d_fresh_cell_root": str(R6D_ROOT.resolve()), "plan_core": core_meta, "launch": launch_meta,
        "program": program_meta, "portable": portable_meta,
        "runtime_sources": {
            "controller": file_metadata(CONTROLLER), "continue_gate": file_metadata(GATE),
            "supervisor": file_metadata(SUPERVISOR), "checker": file_metadata(ASSERT_LIVE),
        },
        "builder_source": file_metadata(BUILDER),
        "stage_a_unsigned_authorizations": auth_meta,
        "stage_a_nonce_inventory": {"count": 3, "pairwise_distinct": True,
                                    "all_lowercase_256_bit_hex": True, "nonces": nonces},
        "stage_a_signature_absence": stage_a_absence,
        "conditional_authorization_absence": conditional_absence,
        "gpu_used": False, "score_data_accessed": False, "formal_data_accessed": False,
    }
    proof_meta = _predicted_metadata(paths["proof"], proof)
    trigger = {
        "schema": "m2_post33_phase_c_v4_r6d_live_signer_trigger_v1",
        "receipt_root": str(R6D.resolve()), "cell_root": str(R6D_ROOT.resolve()),
        "ready": core["ready"], "public_key": core["public_key"], "program": program_meta,
        "portable": portable_meta, "cost_supplement": core["cost_supplement"],
        "controller_source": core["controller_source"], "continue_gate_source": core["continue_gate_source"],
        "supervisor_source": core["supervisor_source"], "assert_live_source": core["assert_live_source"],
        "stage_a_launch": launch_meta, "plan_core": core_meta, "pretrigger_proof": proof_meta,
    }
    # All absence checks are repeated immediately before target writes.
    if any(path.exists() or path.with_suffix(".sig").exists() for path in (R6D / "auth" / item for item in (
        "stage_b_execution_gpu0_r6d.json", "stage_b_execution_gpu1_r6d.json", "full_opening_r6d.json"
    ))):
        raise PermissionError("r6d conditional capability path exists before first target write")
    return R6DStaticPlan(program, portable, shards, authorizations, launch, core, proof, trigger)


def build_r6d_rollover() -> dict[str, Any]:
    """Write a fully preflighted static r6d tree; final trigger is last only."""
    plan = preflight_r6d()
    # Nothing above this line writes an r6d receipt/cell artifact.
    program_path = R6D / "program/phase_c_program_r6d.json"
    portable_path = R6D / "manifest/portable_r6d.json"
    shard_paths = {
        "gpu0": R6D / "manifest/shard_stage_a_gpu0_r6d.json",
        "gpu1": R6D / "manifest/shard_stage_a_gpu1_r6d.json",
        "opening": R6D / "manifest/shard_stage_a_opening_r6d.json",
        "stage_b_gpu0": R6D / "manifest/shard_stage_b_gpu0_r6d.json",
        "stage_b_gpu1": R6D / "manifest/shard_stage_b_gpu1_r6d.json",
        "full_opening": R6D / "manifest/shard_full_opening_r6d.json",
    }
    auth_paths = {
        "gpu0": R6D / "auth/stage_a_execution_gpu0_r6d.json",
        "gpu1": R6D / "auth/stage_a_execution_gpu1_r6d.json",
        "opening": R6D / "auth/stage_a_opening_r6d.json",
    }
    write_json_exclusive(program_path, plan.program)
    write_json_exclusive(portable_path, plan.portable)
    for name, path in shard_paths.items():
        write_json_exclusive(path, plan.shards[name])
    for name, path in auth_paths.items():
        write_json_exclusive(path, plan.authorizations[name])
        if path.with_suffix(".sig").exists():
            raise PermissionError("r6d Stage-A signature preexists before trigger")
    launch_path = R6D / "launch/stage_a_commands_r6d.json"
    core_path = R6D / "signer_requests/r6d_live_signer_plan_core.json"
    proof_path = R6D / "r6d_static_pretrigger_evidence.json"
    write_json_exclusive(launch_path, plan.launch)
    write_json_exclusive(core_path, plan.core)
    write_json_exclusive(proof_path, plan.proof)
    # Exact expected metadata must still agree just before the final trigger.
    for path, payload in ((program_path, plan.program), (portable_path, plan.portable), (launch_path, plan.launch), (core_path, plan.core), (proof_path, plan.proof)):
        if file_metadata(path) != _predicted_metadata(path, payload):
            raise PermissionError(f"r6d pretrigger artifact byte drift: {path.name}")
    trigger_path = _atomic_publish_json(R6D / "signer_requests/r6d_live_signer_trigger.json", plan.trigger)
    return {
        "receipt_root": str(R6D), "cell_root": str(R6D_ROOT), "program": file_metadata(program_path),
        "launch": file_metadata(launch_path), "plan_core": file_metadata(core_path),
        "proof": file_metadata(proof_path), "trigger": file_metadata(trigger_path),
        "ready": file_metadata(READY),
    }


def main() -> None:
    print(json.dumps(build_r6d_rollover(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
