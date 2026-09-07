#!/usr/bin/env python3
"""Single-purpose r6d live-only capability signer.

The r6d controller owns a newly generated Ed25519 private key only in its
live address space.  Before it signs Stage A, it verifies a three-part sealed
preflight chain:

``trigger -> plan core -> pretrigger proof + launch contract``.

The core is the exact capability-semantic preimage; the proof binds its bytes
and the launch/source metadata without a circular self-hash; the final trigger
is the last O_EXCL event and contains no new capability semantics.  The
controller validates all of them, including the program source map for itself,
the gate, checker, and supervisor.  It never reads an endpoint value: opening
and decision classification remain score-blind child processes.
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
R6D_RECEIPTS = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6d"
R6D_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6d"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6d_live_signer_20260805"
PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6d_root_ed25519_public.pem"
READY = RUNTIME / "live_signer_ready.json"
TRIGGER = R6D_RECEIPTS / "signer_requests/r6d_live_signer_trigger.json"
CORE = R6D_RECEIPTS / "signer_requests/r6d_live_signer_plan_core.json"
PROOF = R6D_RECEIPTS / "r6d_static_pretrigger_evidence.json"
LAUNCH = R6D_RECEIPTS / "launch/stage_a_commands_r6d.json"
STAGE_A_ACK = R6D_RECEIPTS / "signer_acks/stage_a_capabilities_signed.json"
STAGE_B_ACK = R6D_RECEIPTS / "signer_acks/conditional_stage_b_capabilities_signed.json"
STOP_ACK = R6D_RECEIPTS / "signer_acks/stage_a_futility_stop_no_stage_b_capabilities.json"
FAILURE = RUNTIME / "controller_failure.json"
GATE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_continue_gate.py"
SUPERVISOR = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_supervisor.py"
ASSERT_LIVE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_assert_live.py"
ROLLOVER = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_rollover.py"
OPEN_STAGE_A = ROOT / "sua_exploration/scripts/open_m2_native_post33_phase_c_v4_stage_a.py"
R6C_RETIREMENT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_retirement_20260805/r6c_pre_gpu_retirement.json"
COST = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3/cost/cost_supplement_r3.json"
POLL_SECONDS = 0.5
EXIT_CONTINUE = 0
EXIT_VALIDATED_STOP = 10

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


def _time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise PermissionError("r6d time field must be ISO-8601 text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PermissionError("r6d time field lacks timezone")
    return parsed.astimezone(timezone.utc)


def _proc_starttime(pid: int) -> int:
    tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
    if len(tail) <= 19:
        raise RuntimeError("/proc stat lacks starttime")
    return int(tail[19])


def _proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(require_canonical_regular_file(path).read_text(encoding="utf-8"))


def _metadata(value: Any, expected: Path, label: str) -> Path:
    canonical = require_canonical_regular_file(expected)
    if value != file_metadata(canonical):
        raise PermissionError(f"r6d {label} metadata mismatch")
    return canonical


def _source_is_sealed(program: Mapping[str, Any], source: Path) -> None:
    relative = source.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix()
    expected = {
        "relative_path": relative,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    if not isinstance(program.get("source_map"), list) or expected not in program["source_map"]:
        raise PermissionError(f"r6d program source map omits {relative}")


def _child_environment() -> dict[str, str]:
    return {"PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1"}


def _run_score_blind_child(argv: list[str]) -> int:
    if not argv or Path(argv[0]).resolve(strict=True) != Path(sys.executable).resolve(strict=True):
        raise PermissionError("score-blind helper Python substitution")
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
    raise RuntimeError("score-blind child ended without status")


def _reload_live_modules() -> None:
    """Load the post-bootstrap r6d public-anchor literals only after trigger."""
    global authorization, program_module
    importlib.invalidate_caches()
    authorization = importlib.reload(authorization)
    program_module = importlib.reload(program_module)


def _assert_live_binding(ready: Mapping[str, Any]) -> None:
    if (
        ready.get("schema") != "m2_post33_phase_c_v4_r6d_live_signer_ready_v1"
        or ready.get("pid") != os.getpid()
        or ready.get("host_id") != socket.gethostname()
        or ready.get("controller_source") != file_metadata(Path(__file__).resolve())
        or ready.get("public_key") != file_metadata(PUBLIC_KEY)
    ):
        raise PermissionError("r6d live signer ready binding mismatch")
    if (
        ready.get("private_key_not_serialized") is not True
        or ready.get("private_key_retained_live_in_memory") is not True
        or ready.get("private_key_not_supplied_via_argv_environment_file_or_ipc") is not True
    ):
        raise PermissionError("r6d ready receipt lacks key-boundary assertions")
    if (
        _proc_starttime(os.getpid()) != ready.get("proc_starttime_ticks")
        or _proc_cmdline_sha256(os.getpid()) != ready.get("proc_cmdline_sha256")
    ):
        raise PermissionError("r6d live signer process identity drift")


def _assert_auth_window(body: Mapping[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    issued, expires = _parse_time(body.get("issued_at")), _parse_time(body.get("expires_at"))
    if not (issued <= now <= expires):
        raise PermissionError("r6d authorization window is not live")


def _expected_stage_a_paths() -> dict[str, tuple[Path, Path]]:
    return {
        "gpu0": (
            R6D_RECEIPTS / "auth/stage_a_execution_gpu0_r6d.json",
            R6D_RECEIPTS / "manifest/shard_stage_a_gpu0_r6d.json",
        ),
        "gpu1": (
            R6D_RECEIPTS / "auth/stage_a_execution_gpu1_r6d.json",
            R6D_RECEIPTS / "manifest/shard_stage_a_gpu1_r6d.json",
        ),
        "opening": (
            R6D_RECEIPTS / "auth/stage_a_opening_r6d.json",
            R6D_RECEIPTS / "manifest/shard_stage_a_opening_r6d.json",
        ),
    }


def _conditional_paths() -> dict[str, tuple[Path, Path]]:
    return {
        "gpu0": (
            R6D_RECEIPTS / "auth/stage_b_execution_gpu0_r6d.json",
            R6D_RECEIPTS / "manifest/shard_stage_b_gpu0_r6d.json",
        ),
        "gpu1": (
            R6D_RECEIPTS / "auth/stage_b_execution_gpu1_r6d.json",
            R6D_RECEIPTS / "manifest/shard_stage_b_gpu1_r6d.json",
        ),
        "full_opening": (
            R6D_RECEIPTS / "auth/full_opening_r6d.json",
            R6D_RECEIPTS / "manifest/shard_full_opening_r6d.json",
        ),
    }


def _validate_core(core: Mapping[str, Any], trigger: Mapping[str, Any]) -> dict[str, Path]:
    required = {
        "schema", "receipt_root", "cell_root", "ready", "public_key", "program", "portable",
        "cost_supplement", "controller_source", "continue_gate_source", "supervisor_source",
        "assert_live_source", "stage_a_launch", "issued_at", "expires_at", "stage_a",
        "conditional_targets",
    }
    if set(core) != required or core.get("schema") != "m2_post33_phase_c_v4_r6d_signer_plan_core_v1":
        raise PermissionError("r6d core exact schema mismatch")
    if core.get("receipt_root") != str(R6D_RECEIPTS.resolve()) or core.get("cell_root") != str(R6D_CELL_ROOT.resolve()):
        raise PermissionError("r6d core root substitution")
    fixed = {
        "ready": READY, "public_key": PUBLIC_KEY,
        "program": R6D_RECEIPTS / "program/phase_c_program_r6d.json",
        "portable": R6D_RECEIPTS / "manifest/portable_r6d.json", "cost_supplement": COST,
        "controller_source": Path(__file__).resolve(), "continue_gate_source": GATE,
        "supervisor_source": SUPERVISOR, "assert_live_source": ASSERT_LIVE, "stage_a_launch": LAUNCH,
    }
    paths: dict[str, Path] = {}
    for field, path in fixed.items():
        paths[field] = _metadata(core.get(field), path, f"core {field}")
        if trigger.get(field) != core.get(field):
            raise PermissionError(f"r6d trigger/core {field} mismatch")
    issued, expires = _parse_time(core["issued_at"]), _parse_time(core["expires_at"])
    if expires <= issued or (expires - issued).total_seconds() > authorization.MAX_VALIDITY_SECONDS:
        raise PermissionError("r6d core validity window invalid")
    return paths


def _assert_proof_absence(proof: Mapping[str, Any]) -> None:
    stage_a = proof.get("stage_a_signature_absence")
    conditional = proof.get("conditional_authorization_absence")
    if not isinstance(stage_a, Mapping) or set(stage_a) != {"gpu0", "gpu1", "opening"}:
        raise PermissionError("r6d proof Stage-A absence scope mismatch")
    if not isinstance(conditional, Mapping) or set(conditional) != {"gpu0", "gpu1", "full_opening"}:
        raise PermissionError("r6d proof conditional absence scope mismatch")
    for name, (auth, _) in _expected_stage_a_paths().items():
        item = stage_a[name]
        if not isinstance(item, Mapping) or item.get("signature_path") != str(auth.with_suffix(".sig").resolve()):
            raise PermissionError("r6d proof Stage-A signature path mismatch")
        if item.get("present_at_pretrigger") is not False or auth.with_suffix(".sig").exists():
            raise PermissionError("r6d Stage-A signature is present before signing")
    for name, (auth, _) in _conditional_paths().items():
        item = conditional[name]
        if not isinstance(item, Mapping) or item.get("authorization_path") != str(auth.resolve()):
            raise PermissionError("r6d proof conditional path mismatch")
        if (
            item.get("authorization_present_at_pretrigger") is not False
            or item.get("signature_path") != str(auth.with_suffix(".sig").resolve())
            or item.get("signature_present_at_pretrigger") is not False
            or auth.exists() or auth.with_suffix(".sig").exists()
        ):
            raise PermissionError("r6d conditional capability materialized before decision")
    if R6D_CELL_ROOT.exists() or R6D_CELL_ROOT.is_symlink():
        raise PermissionError("r6d cell root/claim surface exists before Stage-A signing")


def _validate_proof(proof: Mapping[str, Any], *, trigger: Mapping[str, Any], core: Mapping[str, Any]) -> None:
    required = {
        "schema", "r6c_retirement", "r6d_fresh_receipt_root", "r6d_fresh_cell_root",
        "plan_core", "launch", "program", "portable", "runtime_sources", "builder_source",
        "stage_a_unsigned_authorizations",
        "stage_a_nonce_inventory", "stage_a_signature_absence", "conditional_authorization_absence",
        "gpu_used", "score_data_accessed", "formal_data_accessed",
    }
    if set(proof) != required or proof.get("schema") != "m2_post33_phase_c_v4_r6d_static_pretrigger_evidence_v1":
        raise PermissionError("r6d proof exact schema mismatch")
    if (
        proof.get("r6d_fresh_receipt_root") != str(R6D_RECEIPTS.resolve())
        or proof.get("r6d_fresh_cell_root") != str(R6D_CELL_ROOT.resolve())
        or proof.get("r6c_retirement") != file_metadata(R6C_RETIREMENT)
        or proof.get("plan_core") != file_metadata(CORE)
        or proof.get("launch") != file_metadata(LAUNCH)
        or proof.get("program") != core["program"]
        or proof.get("portable") != core["portable"]
        or proof.get("gpu_used") is not False
        or proof.get("score_data_accessed") is not False
        or proof.get("formal_data_accessed") is not False
    ):
        raise PermissionError("r6d proof core/retirement binding mismatch")
    sources = proof.get("runtime_sources")
    expected_sources = {
        "controller": core["controller_source"], "continue_gate": core["continue_gate_source"],
        "supervisor": core["supervisor_source"], "checker": core["assert_live_source"],
    }
    if sources != expected_sources:
        raise PermissionError("r6d proof runtime-source binding mismatch")
    # The builder is provenance-bound by the proof but intentionally is not a
    # runtime source-map root: it is not allowed on the execution path.
    if proof.get("builder_source") != file_metadata(ROLLOVER):
        raise PermissionError("r6d proof rollover-builder provenance mismatch")
    stage_a = proof.get("stage_a_unsigned_authorizations")
    if not isinstance(stage_a, Mapping) or set(stage_a) != {"gpu0", "gpu1", "opening"}:
        raise PermissionError("r6d proof Stage-A authorization scope mismatch")
    for name, (auth, _) in _expected_stage_a_paths().items():
        if stage_a[name] != file_metadata(auth):
            raise PermissionError("r6d proof Stage-A authorization metadata mismatch")
    inventory = proof.get("stage_a_nonce_inventory")
    if not isinstance(inventory, Mapping):
        raise PermissionError("r6d nonce inventory missing")
    values = inventory.get("nonces")
    if (
        not isinstance(values, list) or len(values) != 3 or len(set(values)) != 3
        or any(not isinstance(nonce, str) or len(nonce) != 64 or any(c not in "0123456789abcdef" for c in nonce) for nonce in values)
        or inventory.get("count") != 3 or inventory.get("pairwise_distinct") is not True
        or inventory.get("all_lowercase_256_bit_hex") is not True
    ):
        raise PermissionError("r6d Stage-A nonce inventory invalid")
    observed = [_json(auth)["authorization"].get("single_use_nonce") for auth, _ in _expected_stage_a_paths().values()]
    if observed != values:
        raise PermissionError("r6d proof nonce inventory does not bind authorization bodies")
    _assert_proof_absence(proof)


def _validate_trigger(trigger: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, Any]]:
    required = {
        "schema", "receipt_root", "cell_root", "ready", "public_key", "program", "portable",
        "cost_supplement", "controller_source", "continue_gate_source", "supervisor_source",
        "assert_live_source", "stage_a_launch", "plan_core", "pretrigger_proof",
    }
    if set(trigger) != required or trigger.get("schema") != "m2_post33_phase_c_v4_r6d_live_signer_trigger_v1":
        raise PermissionError("r6d trigger exact schema mismatch")
    if trigger.get("receipt_root") != str(R6D_RECEIPTS.resolve()) or trigger.get("cell_root") != str(R6D_CELL_ROOT.resolve()):
        raise PermissionError("r6d trigger root substitution")
    core = _metadata(trigger.get("plan_core"), CORE, "trigger plan core")
    proof = _metadata(trigger.get("pretrigger_proof"), PROOF, "trigger proof")
    core_payload = _json(core)
    paths = _validate_core(core_payload, trigger)
    program_payload = program_module.validate_phase_c_program_receipt(paths["program"])
    for source in (Path(__file__).resolve(), GATE, SUPERVISOR, ASSERT_LIVE):
        _source_is_sealed(program_payload, source)
    if authorization.PUBLIC_KEY != PUBLIC_KEY or authorization.PUBLIC_KEY_SHA256 != sha256_file(PUBLIC_KEY):
        raise PermissionError("r6d authorization verifier not pinned to live anchor")
    _validate_proof(_json(proof), trigger=trigger, core=core_payload)
    return {"program": paths["program"], "portable": paths["portable"], "cost": paths["cost_supplement"]}, core_payload


def _assert_stage_a_auth(auth: Path, *, core: Mapping[str, Any], shard: Path, paths: Mapping[str, Path]) -> None:
    body = _json(auth).get("authorization")
    expected = {
        "stage_a_execution_gpu0_r6d.json": ("stage_a", "cell_execution", [0, 2, 4, 6]),
        "stage_a_execution_gpu1_r6d.json": ("stage_a", "cell_execution", [1, 3, 5]),
        "stage_a_opening_r6d.json": ("stage_a_opening", "opening", list(range(7))),
    }
    if not isinstance(body, Mapping) or auth.name not in expected:
        raise PermissionError("r6d Stage-A authorization malformed/name mismatch")
    stage, scope, folds = expected[auth.name]
    if (
        body.get("stage") != stage or body.get("capability_scope") != scope
        or body.get("fold_allowlist") != folds or body.get("seed_allowlist") != [42]
        or body.get("absolute_cell_root") != str(R6D_CELL_ROOT.resolve())
        or body.get("phase_c_program_receipt") != file_metadata(paths["program"])
        or body.get("portable_manifest") != file_metadata(paths["portable"])
        or body.get("shard_manifest") != file_metadata(shard)
        or body.get("public_key") != file_metadata(PUBLIC_KEY)
        or body.get("issued_at") != core["issued_at"] or body.get("expires_at") != core["expires_at"]
        or "stage_a_decision" in body or "stage_a_decision_signature" in body
    ):
        raise PermissionError("r6d Stage-A authorization scope/binding mismatch")


def _sign_stage_a(core: Mapping[str, Any], private_key: Ed25519PrivateKey, ready: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    stage_a = core.get("stage_a")
    if not isinstance(stage_a, Mapping) or set(stage_a) != {"gpu0", "gpu1", "opening"}:
        raise PermissionError("r6d Stage-A core scope mismatch")
    signed: dict[str, Any] = {}
    for name, (auth, shard) in _expected_stage_a_paths().items():
        row = stage_a[name]
        if not isinstance(row, Mapping) or set(row) != {"authorization", "shard_manifest"}:
            raise PermissionError("r6d Stage-A core row mismatch")
        _metadata(row["authorization"], auth, f"Stage-A {name} authorization")
        _metadata(row["shard_manifest"], shard, f"Stage-A {name} shard")
        validate_shard_manifest(shard, portable_manifest_path=paths["portable"], cell_root=R6D_CELL_ROOT)
        _assert_stage_a_auth(auth, core=core, shard=shard, paths=paths)
        body = _json(auth)["authorization"]
        _assert_auth_window(body)
        signature = auth.with_suffix(".sig")
        if signature.exists():
            raise FileExistsError("r6d Stage-A signature preexists")
        write_bytes_exclusive(signature, base64.b64encode(private_key.sign(auth.read_bytes())))
        authorization.verify_signed_authorization(
            auth, signature, phase_c_program_receipt_path=paths["program"],
            portable_manifest_path=paths["portable"], shard_manifest_path=shard,
            cost_supplement_path=paths["cost"], cell_root=R6D_CELL_ROOT, now=datetime.now(timezone.utc),
        )
        signed[name] = {"authorization": file_metadata(auth), "signature": file_metadata(signature)}
    _assert_live_binding(ready)
    write_json_exclusive(STAGE_A_ACK, {
        "schema": "m2_post33_phase_c_v4_r6d_stage_a_capabilities_signed_v1",
        "ready": file_metadata(READY), "program": file_metadata(paths["program"]),
        "trigger": file_metadata(TRIGGER), "plan_core": file_metadata(CORE),
        "pretrigger_proof": file_metadata(PROOF), "launch": file_metadata(LAUNCH),
        "runtime_sources": {
            "controller": file_metadata(Path(__file__).resolve()),
            "continue_gate": file_metadata(GATE),
            "supervisor": file_metadata(SUPERVISOR),
            "checker": file_metadata(ASSERT_LIVE),
        },
        "stage_a": signed, "private_key_serialized_or_disk_persisted": False,
        "private_key_not_serialized": True,
        "private_key_not_supplied_via_argv_environment_file_or_ipc": True,
        "private_key_retained_live_in_memory": True,
        "score_data_accessed_before_stage_a_capabilities_signed": False,
        "helper_spawn_api": "os.posix_spawn",
    })


def _revalidate_immediately_before_stage_a_sign(
    ready: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Re-read the whole sealed trigger chain at the final signing boundary.

    A proof-only absence check cannot detect replacement of the trigger, core,
    proof, or metadata-only launch row after an earlier successful validation.
    This helper makes the second, complete validation explicit and testable.
    """
    fresh_trigger = _read_stable_trigger()
    paths, core = _validate_trigger(fresh_trigger)
    _assert_live_binding(ready)
    # Retain a final direct re-read as a narrow belt-and-suspenders check after
    # the complete chain validation and before the first signature side effect.
    _assert_proof_absence(_json(PROOF))
    return paths, core


def _wait_for(path: Path) -> None:
    while True:
        try:
            first = file_metadata(require_canonical_regular_file(path))
            time.sleep(POLL_SECONDS)
            if first == file_metadata(require_canonical_regular_file(path)):
                return
        except (FileNotFoundError, OSError, ValueError):
            pass
        time.sleep(POLL_SECONDS)


def _read_stable_trigger() -> dict[str, Any]:
    while True:
        try:
            first = file_metadata(require_canonical_regular_file(TRIGGER, within=R6D_RECEIPTS))
            payload = _json(TRIGGER)
            time.sleep(POLL_SECONDS)
            if first == file_metadata(require_canonical_regular_file(TRIGGER, within=R6D_RECEIPTS)):
                return payload
        except (json.JSONDecodeError, FileNotFoundError, OSError, ValueError):
            pass
        time.sleep(POLL_SECONDS)


def _open_and_sign_decision(core: Mapping[str, Any], private_key: Ed25519PrivateKey, ready: Mapping[str, Any], paths: Mapping[str, Path]) -> Path:
    validate_stage_a_cell_directory_set(R6D_CELL_ROOT)
    stage_a = core["stage_a"]
    opening = stage_a["opening"]
    auth = Path(str(opening["authorization"]["canonical_path"])).resolve(strict=True)
    shard = Path(str(opening["shard_manifest"]["canonical_path"])).resolve(strict=True)
    command = [
        sys.executable, str(OPEN_STAGE_A), "--root", str(R6D_CELL_ROOT),
        "--program-receipt", str(paths["program"]), "--portable-manifest", str(paths["portable"]),
        "--shard-manifest", str(shard), "--cost-supplement", str(paths["cost"]),
        "--authorization", str(auth), "--authorization-signature", str(auth.with_suffix(".sig")),
    ]
    _assert_live_binding(ready)
    if _run_score_blind_child(command) != 0:
        raise RuntimeError("r6d Stage-A opener failed without exposing output")
    decision = require_canonical_regular_file(stage_a_paths(R6D_CELL_ROOT)["decision"], within=R6D_CELL_ROOT)
    signature = stage_a_paths(R6D_CELL_ROOT)["decision_signature"]
    if signature.exists():
        raise FileExistsError("r6d Stage-A decision signature preexists")
    _assert_live_binding(ready)
    write_bytes_exclusive(signature, base64.b64encode(private_key.sign(decision.read_bytes())))
    return decision


def _decision_gate() -> str:
    code = _run_score_blind_child([sys.executable, str(GATE), "--root", str(R6D_CELL_ROOT), "--classify-signed"])
    if code == EXIT_CONTINUE:
        return "continue"
    if code == EXIT_VALIDATED_STOP:
        return "validated_stop"
    raise PermissionError(f"r6d Stage-A decision gate failed closed: {code}")


def _existing_nonces() -> set[str]:
    values: set[str] = set()
    for auth, _ in _expected_stage_a_paths().values():
        values.add(str(_json(auth)["authorization"].get("single_use_nonce")))
    for auth, _ in _conditional_paths().values():
        if auth.exists():
            values.add(str(_json(auth)["authorization"].get("single_use_nonce")))
    return values


def _issue_conditional_one(
    *, base_auth: Path, output: Path, shard: Path, stage: str, scope: str,
    folds: list[int], seeds: list[int], authorization_id: str, core: Mapping[str, Any],
    private_key: Ed25519PrivateKey, paths: Mapping[str, Path], decision: Path,
) -> dict[str, Any]:
    if output.exists() or output.with_suffix(".sig").exists():
        raise FileExistsError("r6d conditional capability already exists")
    validate_shard_manifest(shard, portable_manifest_path=paths["portable"], cell_root=R6D_CELL_ROOT)
    shard_payload = _json(shard)
    if shard_payload.get("fold_allowlist") != folds or shard_payload.get("seed_allowlist") != seeds:
        raise PermissionError("r6d conditional shard scope mismatch")
    body = dict(_json(base_auth)["authorization"])
    nonce = secrets.token_hex(32)
    while nonce in _existing_nonces():
        nonce = secrets.token_hex(32)
    body.update({
        "authorization_id": authorization_id, "single_use_nonce": nonce,
        "issued_at": core["issued_at"], "expires_at": core["expires_at"],
        "stage": stage, "capability_scope": scope, "host_id": shard_payload["host_id"],
        "gpu_id": shard_payload["gpu_id"], "fold_allowlist": folds, "seed_allowlist": seeds,
        "shard_manifest": file_metadata(shard), "stage_a_decision": file_metadata(decision),
        "stage_a_decision_signature": file_metadata(stage_a_paths(R6D_CELL_ROOT)["decision_signature"]),
    })
    _assert_auth_window(body)
    write_json_exclusive(output, {"schema": "m2_post33_phase_c_signed_authorization_envelope_v4", "authorization": body})
    signature = output.with_suffix(".sig")
    write_bytes_exclusive(signature, base64.b64encode(private_key.sign(output.read_bytes())))
    authorization.verify_signed_authorization(
        output, signature, phase_c_program_receipt_path=paths["program"], portable_manifest_path=paths["portable"],
        shard_manifest_path=shard, cost_supplement_path=paths["cost"], cell_root=R6D_CELL_ROOT,
        now=datetime.now(timezone.utc),
    )
    return {"authorization": file_metadata(output), "signature": file_metadata(signature), "shard": file_metadata(shard)}


def _issue_conditional_capabilities(core: Mapping[str, Any], private_key: Ed25519PrivateKey, ready: Mapping[str, Any], paths: Mapping[str, Path], decision: Path) -> None:
    targets = core.get("conditional_targets")
    if not isinstance(targets, Mapping) or set(targets) != {"gpu0", "gpu1", "full_opening"}:
        raise PermissionError("r6d conditional target scope mismatch")
    specification = {
        "gpu0": ("stage_a_execution_gpu0_r6d.json", "stage_b", "cell_execution", [0, 2, 4, 6], [43, 44]),
        "gpu1": ("stage_a_execution_gpu1_r6d.json", "stage_b", "cell_execution", [1, 3, 5], [43, 44]),
        "full_opening": ("stage_a_opening_r6d.json", "full_opening", "opening", list(range(7)), [42, 43, 44]),
    }
    signed: dict[str, Any] = {}
    for name, (base_name, stage, scope, folds, seeds) in specification.items():
        output, shard = _conditional_paths()[name]
        row = targets[name]
        if not isinstance(row, Mapping) or set(row) != {"authorization_path", "shard_manifest"}:
            raise PermissionError("r6d conditional core row mismatch")
        if row.get("authorization_path") != str(output.resolve()):
            raise PermissionError("r6d conditional output path substitution")
        _metadata(row.get("shard_manifest"), shard, f"conditional {name} shard")
        signed[name] = _issue_conditional_one(
            base_auth=R6D_RECEIPTS / "auth" / base_name, output=output, shard=shard,
            stage=stage, scope=scope, folds=folds, seeds=seeds,
            authorization_id=f"m2-post33-phase-c-v4-{stage}_{name}_r6d-20260805",
            core=core, private_key=private_key, paths=paths, decision=decision,
        )
    if len(_existing_nonces()) != 6:
        raise PermissionError("r6d cross-stage nonce inventory is not pairwise unique")
    _assert_live_binding(ready)
    write_json_exclusive(STAGE_B_ACK, {
        "schema": "m2_post33_phase_c_v4_r6d_conditional_capabilities_signed_v1",
        "ready": file_metadata(READY), "decision": file_metadata(decision),
        "decision_signature": file_metadata(stage_a_paths(R6D_CELL_ROOT)["decision_signature"]),
        "conditional_capabilities": signed, "stage_b_issued_only_after_continue": True,
        "private_key_serialized_or_disk_persisted": False, "private_key_not_serialized": True,
        "private_key_not_supplied_via_argv_environment_file_or_ipc": True,
        "private_key_retained_live_in_memory": True, "controller_read_score_values": False,
        "helper_spawn_api": "os.posix_spawn",
    })


def _failure(exc: BaseException) -> None:
    if not FAILURE.exists():
        write_json_exclusive(FAILURE, {
            "schema": "m2_post33_phase_c_v4_r6d_live_signer_failure_v1",
            "failure_class": type(exc).__name__, "failure_message": str(exc),
            "score_or_decision_content_logged": False,
            "private_key_serialized_or_disk_persisted": False,
        })


def bootstrap() -> None:
    if any(path.exists() or path.is_symlink() for path in (R6D_RECEIPTS, R6D_CELL_ROOT, RUNTIME, PUBLIC_KEY)):
        raise FileExistsError("r6d bootstrap requires fresh receipt/cell/runtime/anchor roots")
    private_key: Ed25519PrivateKey | None = Ed25519PrivateKey.generate()
    try:
        public = private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        write_bytes_exclusive(PUBLIC_KEY, public)
        ready = {
            "schema": "m2_post33_phase_c_v4_r6d_live_signer_ready_v1",
            "status": "LIVE_PRIVATE_KEY_IN_MEMORY_ONLY", "host_id": socket.gethostname(),
            "pid": os.getpid(), "proc_starttime_ticks": _proc_starttime(os.getpid()),
            "proc_cmdline_sha256": _proc_cmdline_sha256(os.getpid()),
            "intended_receipt_root": str(R6D_RECEIPTS.resolve()),
            "intended_cell_root": str(R6D_CELL_ROOT.resolve()),
            "controller_source": file_metadata(Path(__file__).resolve()), "public_key": file_metadata(PUBLIC_KEY),
            "private_key_serialized_or_disk_persisted": False, "private_key_not_serialized": True,
            "private_key_not_supplied_via_argv_environment_file_or_ipc": True,
            "private_key_retained_live_in_memory": True,
        }
        write_json_exclusive(READY, ready)
        trigger = _read_stable_trigger()
        _reload_live_modules()
        _assert_live_binding(ready)
        paths, core = _validate_trigger(trigger)
        # Validate once when the trigger is received, then read the entire chain
        # again immediately before signing to close replacement races.
        paths, core = _revalidate_immediately_before_stage_a_sign(ready)
        _sign_stage_a(core, private_key, ready, paths)
        _wait_for(stage_a_paths(R6D_CELL_ROOT)["completed"])
        decision = _open_and_sign_decision(core, private_key, ready, paths)
        result = _decision_gate()
        if result == "continue":
            _issue_conditional_capabilities(core, private_key, ready, paths, decision)
        elif result == "validated_stop":
            write_json_exclusive(STOP_ACK, {
                "schema": "m2_post33_phase_c_v4_r6d_stage_a_futility_stop_v1",
                "ready": file_metadata(READY), "decision": file_metadata(decision),
                "decision_signature": file_metadata(stage_a_paths(R6D_CELL_ROOT)["decision_signature"]),
                "stage_b_authorizations_materialized": False,
                "full_opening_authorization_materialized": False,
                "controller_read_score_values": False,
                "score_opening_was_external_no_key_posix_spawn_helper": True,
            })
        else:
            raise AssertionError("unreachable decision outcome")
    except BaseException as exc:
        _failure(exc)
        raise
    finally:
        private_key = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args(argv)
    if not args.bootstrap:
        raise SystemExit("only --bootstrap is supported")
    bootstrap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
