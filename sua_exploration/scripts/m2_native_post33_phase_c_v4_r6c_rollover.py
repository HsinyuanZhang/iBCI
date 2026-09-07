#!/usr/bin/env python3
"""CPU-only r6c fresh-root builder for the live-signer Phase-C recovery.

This builder never receives or serializes a private key.  It requires the
prestarted r6c live signer to have created its public anchor and PID-bound ready
receipt.  It constructs only source/data receipts, portable/shard manifests,
and three *unsigned* Stage-A authorization bodies.  An atomically published
plan lets that already-live signer produce the three detached signatures.

Stage-B/full-opening manifests may be prepared here, but their authorization
bodies and signatures are deliberately absent until the live signer has signed
and externally classified the Stage-A decision as `continue`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3"
R6B = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6b"
R6B_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6b"
R3_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r3"
R6B_ABORT_INCIDENT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6b_abort_incident_20260805/r6b_controlled_stop_incident.json"
ABORTED_R6_PUBLIC = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6_root_ed25519_public.pem"
R6C_RUNTIME_INCIDENT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_synthetic_watchdog_incident_20260805/incident_before_quarantine.json"
R6C_RUNTIME_QUARANTINE = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_synthetic_watchdog_incident_20260805/post_quarantine_verification.json"
R6C_RUNTIME_INCIDENT_SHA256 = "48482e5ae6cfaf9b6fdae5ff8d246e13d21960d707c9f41b57f53da2889e819b"
R6C_RUNTIME_QUARANTINE_SHA256 = "a54561fe82d221f813536dabfdae84e036fc252a8150d5963d29423304a81efd"
R6C = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6c"
R6C_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6c"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_live_signer_20260805"
READY = RUNTIME / "live_signer_ready.json"
PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6c_root_ed25519_public.pem"
CONTROLLER = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_live_signer.py"
GATE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_continue_gate.py"
ASSERT_LIVE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py"
SUPERVISOR = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_supervisor.py"
EOF_RECEIPT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_upstream_canonicalization_20260805/upstream_eof_canonicalization.json"
DEEP_SOURCE_AUDIT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_deep_source_20260805/deep_source_audit.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _json(path: Path) -> dict[str, Any]:
    return json.loads(require_canonical_regular_file(path).read_text(encoding="utf-8"))


def _time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _proc_starttime(pid: int) -> int:
    tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
    if len(tail) <= 19:
        raise PermissionError("live signer proc stat has no starttime")
    return int(tail[19])


def _proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest()


def assert_live_ready() -> dict[str, Any]:
    """Independent builder-side host/PID/starttime/cmdline/source/anchor check."""
    ready = _json(READY)
    if (
        ready.get("schema") != "m2_post33_phase_c_v4_r6c_live_signer_ready_v1"
        or ready.get("status") != "LIVE_PRIVATE_KEY_IN_MEMORY_ONLY"
        or ready.get("host_id") != socket.gethostname()
        or ready.get("intended_receipt_root") != str(R6C.resolve())
        or ready.get("intended_cell_root") != str(R6C_ROOT.resolve())
        or ready.get("controller_source") != file_metadata(CONTROLLER)
        or ready.get("public_key") != file_metadata(PUBLIC_KEY)
        or ready.get("private_key_not_serialized") is not True
        or ready.get("private_key_not_supplied_via_argv_environment_file_or_ipc") is not True
        or ready.get("private_key_retained_live_in_memory") is not True
    ):
        raise PermissionError("r6c live signer ready receipt binding mismatch")
    pid = ready.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        raise PermissionError("r6c live signer PID invalid")
    try:
        os.kill(pid, 0)
        if _proc_starttime(pid) != ready.get("proc_starttime_ticks"):
            raise PermissionError("r6c live signer starttime mismatch")
        if _proc_cmdline_sha256(pid) != ready.get("proc_cmdline_sha256"):
            raise PermissionError("r6c live signer command mismatch")
    except ProcessLookupError as exc:
        raise PermissionError("r6c live signer is dead") from exc
    return ready


def _independent_assert_live_check() -> None:
    """Run the separately source-sealed checker before triggering the signer."""
    command = [
        "/home/xinyuan/miniconda3/envs/spint/bin/python", "-u",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py", "--check-once",
    ]
    result = subprocess.run(
        command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode != 0:
        raise PermissionError("independent r6c assert-live preflight failed")


def _assert_quarantine_receipts_and_clean_runtime() -> dict[str, Any]:
    """Bind the real synthetic-fixture incident before any r6c-root write."""
    if sha256_file(R6C_RUNTIME_INCIDENT) != R6C_RUNTIME_INCIDENT_SHA256:
        raise PermissionError("r6c synthetic incident receipt hash mismatch")
    if sha256_file(R6C_RUNTIME_QUARANTINE) != R6C_RUNTIME_QUARANTINE_SHA256:
        raise PermissionError("r6c post-quarantine receipt hash mismatch")
    observed_entries = sorted(str(path.relative_to(RUNTIME)) for path in RUNTIME.rglob("*"))
    expected_entries = [str(READY.relative_to(RUNTIME))]
    if observed_entries != expected_entries:
        raise PermissionError(
            f"r6c live runtime must contain exactly READY; observed={observed_entries}"
        )
    return {
        "incident_before_quarantine": file_metadata(R6C_RUNTIME_INCIDENT),
        "post_quarantine_verification": file_metadata(R6C_RUNTIME_QUARANTINE),
        "live_runtime_exact_files": [file_metadata(READY)],
    }


def _source_rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("source_map")
    if not isinstance(rows, list):
        raise ValueError("program source map is missing")
    result = {str(row.get("relative_path")): row for row in rows if isinstance(row, Mapping)}
    if len(result) != len(rows):
        raise ValueError("program source map malformed")
    return result


def _program_delta(r6b: Mapping[str, Any], r6c: Mapping[str, Any]) -> dict[str, Any]:
    for field in (
        "schema", "protocol_id", "phase_id", "absolute_workspace_root", "score_data_accessed",
        "formal_data_accessed", "gpu_used", "phase_a_b_eof_canonicalization", "deep_source_audit_receipt",
    ):
        if r6b.get(field) != r6c.get(field):
            raise ValueError(f"r6c changed program semantic field {field}")
    before, after = _source_rows(r6b), _source_rows(r6c)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    expected_added = {
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_live_signer.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_continue_gate.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_supervisor.py",
    }
    expected_changed = {
        "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py",
        "sua_exploration/mc_maze/m2_native_post33_program_v4.py",
    }
    if set(added) != expected_added or removed or set(changed) != expected_changed:
        raise ValueError(f"unexpected r6c source closure delta added={added} removed={removed} changed={changed}")
    return {"added_runtime_roots": added, "changed_governance_sources": changed, "removed": removed}


def _portable(template: Mapping[str, Any], program: Path) -> dict[str, Any]:
    payload = dict(template)
    payload["absolute_cell_root"] = str(R6C_ROOT.resolve())
    rows = [dict(row) for row in template["hash_closures"]]
    for row in rows:
        if row.get("role") == "phase_c_program_receipt":
            row.clear()
            row.update({"role": "phase_c_program_receipt", **file_metadata(program)})
    payload["hash_closures"] = rows
    return payload


def _shard(template: Mapping[str, Any], portable: Path, *, seeds: list[int]) -> dict[str, Any]:
    payload = dict(template)
    payload["absolute_cell_root"] = str(R6C_ROOT.resolve())
    payload["portable_transfer_manifest_sha256"] = sha256_file(portable)
    payload["seed_allowlist"] = seeds
    return payload


def _stage_a_auth(template: Mapping[str, Any], *, program: Path, portable: Path, shard: Path, issued: datetime, expires: datetime) -> dict[str, Any]:
    body = dict(template["authorization"])
    original = str(body["authorization_id"])
    body["authorization_id"] = original.replace("r6b", "r6c")
    if body["authorization_id"] == original:
        raise ValueError("r6b authorization id lacks r6b marker")
    body.update({
        "single_use_nonce": secrets.token_hex(32),
        "issued_at": _time(issued),
        "expires_at": _time(expires),
        "absolute_cell_root": str(R6C_ROOT.resolve()),
        "phase_c_program_receipt": file_metadata(program),
        "portable_manifest": file_metadata(portable),
        "shard_manifest": file_metadata(shard),
        "public_key": file_metadata(PUBLIC_KEY),
    })
    return {"schema": "m2_post33_phase_c_signed_authorization_envelope_v4", "authorization": body}


def _atomic_publish_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Publish complete JSON atomically without an overwrite capability."""
    if path.exists():
        raise FileExistsError(f"plan already exists: {path}")
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    write_bytes_exclusive(temporary, data)
    try:
        os.link(temporary, path)
    except BaseException:
        # The temporary is private to this failed publication and cannot be a
        # protocol artifact.  Preserve the final absent/unchanged on failure.
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=False)
    return path.resolve(strict=True)


def _launch_argv(*, shard: Path, auth: Path, program: Path, portable: Path, supplement: Path) -> list[str]:
    return [
        "/home/xinyuan/miniconda3/envs/spint/bin/python", "-u",
        str(ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py"),
        "--execute", "--cell-root", str(R6C_ROOT.resolve()), "--workspace-root", str(ROOT.resolve()),
        "--data-root", str((ROOT / "SPINT-main/data/000953").resolve(strict=True)),
        "--shard-manifest", str(shard.resolve(strict=True)), "--portable-manifest", str(portable.resolve(strict=True)),
        "--program-receipt", str(program.resolve(strict=True)), "--cost-supplement", str(supplement.resolve(strict=True)),
        "--authorization", str(auth.resolve(strict=True)), "--authorization-signature", str(auth.with_suffix(".sig").resolve()),
    ]


def build_r6c_rollover() -> dict[str, Any]:
    """Build all CPU-only r6c artifacts through plan publication, not GPU launch."""
    if R6C.exists() or R6C_ROOT.exists():
        raise FileExistsError("r6c receipt/cell root must be absent before build")
    if not ASSERT_LIVE.is_file() or not SUPERVISOR.is_file():
        raise FileNotFoundError("independent r6c checker/supervisor must be published before build")
    quarantine_evidence = _assert_quarantine_receipts_and_clean_runtime()
    ready = assert_live_ready()
    _independent_assert_live_check()
    if authorization.PUBLIC_KEY != PUBLIC_KEY or authorization.PUBLIC_KEY_SHA256 != sha256_file(PUBLIC_KEY):
        raise PermissionError("authorization source is not fixed to r6c public anchor")
    issued = datetime.now(timezone.utc).replace(microsecond=0)
    expires = issued + timedelta(hours=29)
    if expires - issued > timedelta(seconds=authorization.MAX_VALIDITY_SECONDS):
        raise PermissionError("r6c capability validity exceeds production maximum")
    r6b_program = _json(R6B / "program/phase_c_program_r6b.json")
    r6b_portable = _json(R6B / "manifest/portable_r6b.json")
    templates = {
        "gpu0": _json(R6B / "manifest/shard_stage_a_gpu0_r6b.json"),
        "gpu1": _json(R6B / "manifest/shard_stage_a_gpu1_r6b.json"),
        "opening": _json(R6B / "manifest/shard_stage_a_opening_r6b.json"),
    }
    auth_templates = {
        "gpu0": _json(R6B / "auth/stage_a_execution_gpu0_r6b.json"),
        "gpu1": _json(R6B / "auth/stage_a_execution_gpu1_r6b.json"),
        "opening": _json(R6B / "auth/stage_a_opening_r6b.json"),
    }
    program_payload = build_phase_c_program_receipt(
        eof_canonicalization_receipt_path=EOF_RECEIPT, deep_source_audit_receipt_path=DEEP_SOURCE_AUDIT,
    )
    delta = _program_delta(r6b_program, program_payload)
    program = R6C / "program/phase_c_program_r6c.json"
    portable = R6C / "manifest/portable_r6c.json"
    shards = {
        "gpu0": R6C / "manifest/shard_stage_a_gpu0_r6c.json",
        "gpu1": R6C / "manifest/shard_stage_a_gpu1_r6c.json",
        "opening": R6C / "manifest/shard_stage_a_opening_r6c.json",
        "stage_b_gpu0": R6C / "manifest/shard_stage_b_gpu0_r6c.json",
        "stage_b_gpu1": R6C / "manifest/shard_stage_b_gpu1_r6c.json",
        "full_opening": R6C / "manifest/shard_full_opening_r6c.json",
    }
    auths = {
        "gpu0": R6C / "auth/stage_a_execution_gpu0_r6c.json",
        "gpu1": R6C / "auth/stage_a_execution_gpu1_r6c.json",
        "opening": R6C / "auth/stage_a_opening_r6c.json",
    }
    write_json_exclusive(program, program_payload)
    validate_phase_c_program_receipt(program)
    r6c_portable = _portable(r6b_portable, program)
    write_json_exclusive(portable, r6c_portable)
    validate_portable_transfer_manifest(
        portable, workspace_root=ROOT, data_root=ROOT / "SPINT-main/data/000953", cell_root=R6C_ROOT,
    )
    shard_payloads = {
        "gpu0": _shard(templates["gpu0"], portable, seeds=[42]),
        "gpu1": _shard(templates["gpu1"], portable, seeds=[42]),
        "opening": _shard(templates["opening"], portable, seeds=[42]),
        "stage_b_gpu0": _shard(templates["gpu0"], portable, seeds=[43, 44]),
        "stage_b_gpu1": _shard(templates["gpu1"], portable, seeds=[43, 44]),
        "full_opening": _shard(templates["opening"], portable, seeds=[42, 43, 44]),
    }
    for name, path in shards.items():
        write_json_exclusive(path, shard_payloads[name])
        validate_shard_manifest(path, portable_manifest_path=portable, cell_root=R6C_ROOT)
    stage_a_payloads = {
        name: _stage_a_auth(auth_templates[name], program=program, portable=portable, shard=shards[name], issued=issued, expires=expires)
        for name in auths
    }
    nonces = [row["authorization"]["single_use_nonce"] for row in stage_a_payloads.values()]
    if len(nonces) != 3 or len(set(nonces)) != 3:
        raise RuntimeError("r6c Stage-A requires three unique nonces")
    for name, path in auths.items():
        write_json_exclusive(path, stage_a_payloads[name])
        if path.with_suffix(".sig").exists():
            raise RuntimeError("Stage-A must be unsigned before live signer consumes plan")
    launch = {
        "schema": "m2_post33_phase_c_v4_r6c_stage_a_launch_commands_v1",
        "status": "PREPARED_NOT_EXECUTED",
            "source_sealed_supervision_contract": {
            "controller": file_metadata(CONTROLLER),
            "continue_gate": file_metadata(GATE),
            "supervisor": file_metadata(SUPERVISOR),
            "checker": file_metadata(ASSERT_LIVE),
            "ready": file_metadata(READY),
            "check_once_argv_template": [
                "/home/xinyuan/miniconda3/envs/spint/bin/python", "-u",
                "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py", "--check-once",
            ],
            "watch_argv_template": [
                "/home/xinyuan/miniconda3/envs/spint/bin/python", "-u",
                "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py", "--watch",
                "--pid", "<MATRIX_PID>", "--pid-starttime", "<MATRIX_STARTTIME_TICKS>",
                "--matrix-parent-pid", "<MATRIX_PARENT_PID>",
                "--matrix-parent-starttime", "<MATRIX_PARENT_STARTTIME_TICKS>",
                "--matrix-pgid", "<MATRIX_PROCESS_GROUP_ID>",
            ],
            "matrix_must_start_new_session": True,
            "matrix_process_group_must_equal_matrix_pid": True,
            "one_independent_watcher_per_gpu_shard": True,
            "execution_start_receipt_required": True,
        },
        "commands": {
            name: {
                "matrix_argv": _launch_argv(shard=shards[name], auth=auths[name], program=program, portable=portable, supplement=R3 / "cost/cost_supplement_r3.json"),
                "fold_allowlist": shard_payloads[name]["fold_allowlist"],
                "cuda_visible_devices": shard_payloads[name]["gpu_id"],
                "authorization": file_metadata(auths[name]),
                "authorization_signature_path": str(auths[name].with_suffix(".sig").resolve()),
            }
            for name in ("gpu0", "gpu1")
        },
        "cell_root_must_be_absent_until_matrix_execution": str(R6C_ROOT.resolve()),
        "gpu_used": False, "score_data_accessed": False,
    }
    launch_path = R6C / "launch/stage_a_commands_r6c.json"
    write_json_exclusive(launch_path, launch)
    conditional_absence = {
        name: {
            "authorization_json": str(path.resolve()),
            "authorization_json_present": path.exists(),
            "authorization_signature": str(path.with_suffix(".sig").resolve()),
            "authorization_signature_present": path.with_suffix(".sig").exists(),
        }
        for name, path in {
            "gpu0": R6C / "auth/stage_b_execution_gpu0_r6c.json",
            "gpu1": R6C / "auth/stage_b_execution_gpu1_r6c.json",
            "full_opening": R6C / "auth/full_opening_r6c.json",
        }.items()
    }
    if any(value["authorization_json_present"] or value["authorization_signature_present"] for value in conditional_absence.values()):
        raise RuntimeError("Stage-B/full authorization must be absent before live signer plan publication")
    proof = {
        "schema": "m2_post33_phase_c_v4_r6c_static_preplan_evidence_v1",
        "historical_boundaries": {
            "r6b_abort_incident": file_metadata(R6B_ABORT_INCIDENT),
            "r6b_receipt_root": str(R6B.resolve()), "r6b_cell_root": str(R6B_ROOT.resolve()),
            "r3_failed_status_receipts": [
                file_metadata(R3_ROOT / "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1/PHASE_C_V4/cells/arm-spint/fold-0/seed-42/control/status.failed.json"),
                file_metadata(R3_ROOT / "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1/PHASE_C_V4/cells/arm-spint/fold-1/seed-42/control/status.failed.json"),
            ],
            "burned_prewrite_r6_public_anchor": file_metadata(ABORTED_R6_PUBLIC),
            "r6c_runtime_synthetic_fixture_quarantine": quarantine_evidence,
        },
        "r6c_fresh_receipt_root": str(R6C.resolve()), "r6c_fresh_cell_root": str(R6C_ROOT.resolve()),
        "r6c_cell_root_exists_at_preplan": False, "live_signer": file_metadata(READY),
        "public_anchor": file_metadata(PUBLIC_KEY), "source_delta_from_r6b": delta,
        "program": file_metadata(program), "portable": file_metadata(portable),
        "stage_a_unsigned_authorizations": {name: file_metadata(path) for name, path in auths.items()},
        "stage_a_signatures_present_at_plan": False,
        "stage_a_nonce_inventory": {
            "count": len(nonces), "pairwise_distinct": len(set(nonces)) == len(nonces),
            "all_lowercase_256_bit_hex": all(len(nonce) == 64 and all(c in "0123456789abcdef" for c in nonce) for nonce in nonces),
            "authorization_files": {name: file_metadata(path) for name, path in auths.items()},
        },
        "stage_b_and_full_authorizations_absent_at_preplan": conditional_absence,
        "launch": file_metadata(launch_path),
        "gpu_used": False, "score_data_accessed": False, "formal_data_accessed": False,
    }
    proof_path = R6C / "r6c_static_preplan_evidence.json"
    write_json_exclusive(proof_path, proof)
    # The plan is the signer trigger.  It is atomically published only after
    # every static source/launch/proof artifact above is complete and sealed.
    plan = {
        "schema": "m2_post33_phase_c_v4_r6c_live_signer_plan_v1",
        "receipt_root": str(R6C.resolve()), "cell_root": str(R6C_ROOT.resolve()),
        "ready": file_metadata(READY), "public_key": file_metadata(PUBLIC_KEY),
        "program": file_metadata(program), "portable": file_metadata(portable),
        "cost_supplement": file_metadata(R3 / "cost/cost_supplement_r3.json"),
        "controller_source": file_metadata(CONTROLLER), "continue_gate_source": file_metadata(GATE),
        "issued_at": _time(issued), "expires_at": _time(expires),
        "stage_a": {
            name: {"authorization": file_metadata(auths[name]), "shard_manifest": file_metadata(shards[name])}
            for name in auths
        },
        "conditional_targets": {
            "gpu0": {"authorization_path": str((R6C / "auth/stage_b_execution_gpu0_r6c.json").resolve()), "shard_manifest": file_metadata(shards["stage_b_gpu0"])},
            "gpu1": {"authorization_path": str((R6C / "auth/stage_b_execution_gpu1_r6c.json").resolve()), "shard_manifest": file_metadata(shards["stage_b_gpu1"])},
            "full_opening": {"authorization_path": str((R6C / "auth/full_opening_r6c.json").resolve()), "shard_manifest": file_metadata(shards["full_opening"])},
        },
    }
    plan_path = _atomic_publish_json(R6C / "signer_requests/r6c_live_signer_plan.json", plan)
    return {
        "receipt_root": str(R6C), "cell_root": str(R6C_ROOT), "program": file_metadata(program),
        "plan": file_metadata(plan_path), "launch": file_metadata(launch_path), "proof": file_metadata(proof_path),
        "ready": file_metadata(READY),
    }


def main() -> None:
    print(json.dumps(build_r6c_rollover(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
