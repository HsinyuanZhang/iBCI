"""Authorization-first support for V5R2's explicit V5-source closure.

At import time this module uses only the standard library and cryptography.
The V5R2 wrapper, V5 bridge, V3 runtime, Torch, NumPy, PyNWB, model and data
owners remain forbidden until an exact V5R2 authorization is verified and its
fresh nonce is atomically claimed.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_VALIDITY_SECONDS = 15 * 60
AUTHORIZATION_SCHEMA = "dandi_000688_subc_parity_explicit_v5_source_closure_execution_authorization_v5r2"
AUTHORIZATION_ENVELOPE_SCHEMA = "dandi_000688_subc_parity_explicit_v5_source_closure_execution_authorization_envelope_v5r2"
AUTHORIZATION_STATUS = "AUTHORIZED_FOR_ONE_CONSUMED_SUBC_EXPLICIT_V5_SOURCE_CLOSURE_PARITY_EXECUTION"
PERMITTED_ACTION = "concrete_parity_after_future_authorization_v5r2"
CPU_POLICY = {
    "device": "cpu",
    "cuda_visible_devices": "",
    "torch_num_threads": 1,
    "torch_num_interop_threads": 1,
    "deterministic_algorithms": True,
    "tf32": False,
    "autocast": False,
}
CPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class ParityV5R2AuthorizationError(PermissionError):
    """Authorization failed before any data/model runtime may be imported."""


class ParityV5R2ExecutionError(RuntimeError):
    """A post-authorization parity or immutable-output operation failed."""


@dataclass
class PreAuthorizationAudit:
    authorization_files_read: int = 0
    signature_files_read: int = 0
    public_key_files_read: int = 0
    source_hash_checks: int = 0
    signature_checks: int = 0
    nonce_claim_writes: int = 0
    v5r2_helper_imports: int = 0
    v5_v3_torch_owner_imports: int = 0
    checkpoint_files_opened: int = 0
    nwb_files_opened: int = 0
    npz_files_opened: int = 0
    model_forward_calls: int = 0

    def zero_data_access(self) -> bool:
        return (
            self.v5r2_helper_imports == 0
            and self.v5_v3_torch_owner_imports == 0
            and self.checkpoint_files_opened == 0
            and self.nwb_files_opened == 0
            and self.npz_files_opened == 0
            and self.model_forward_calls == 0
        )


@dataclass(frozen=True)
class AuthorizationGrant:
    authorization: dict[str, Any]
    authorization_sha256: str
    signature_sha256: str
    claim_path: Path
    audit: PreAuthorizationAudit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _regular(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ParityV5R2AuthorizationError(f"{label} missing") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ParityV5R2AuthorizationError(f"{label} non-regular or symlinked")
    return resolved


def _file_pin(path: Path) -> dict[str, Any]:
    checked = _regular(path, "pinned file")
    return {
        "path": str(checked),
        "sha256": sha256_file(checked),
        "bytes": checked.stat().st_size,
    }


def runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "host": socket.gethostname(),
        "python": {
            "path": str(executable),
            "sha256": sha256_file(executable),
            "bytes": executable.stat().st_size,
            "implementation": getattr(sys.implementation, "name", ""),
            "version": sys.version,
        },
    }


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ParityV5R2AuthorizationError(f"{label} must be timezone-aware ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ParityV5R2AuthorizationError(f"{label} malformed") from exc
    if parsed.tzinfo is None:
        raise ParityV5R2AuthorizationError(f"{label} lacks timezone")
    return parsed.astimezone(timezone.utc)


def _assert_preimport_boundary() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ParityV5R2AuthorizationError("CUDA_VISIBLE_DEVICES must be empty before V5R2 authorization")
    forbidden = (
        "sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5r2",
        "sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5",
        "sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v3",
        "torch",
    )
    loaded = set(sys.modules)
    if any(name == item or name.startswith(item + ".") for item in forbidden for name in loaded):
        raise ParityV5R2AuthorizationError("V5R2/V5/V3 helper or Torch imported before authorization")


def _within(path: Path, parent: Path, label: str) -> Path:
    candidate, root = path.resolve(), parent.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ParityV5R2AuthorizationError(f"{label} escapes V5R2 dedicated namespace") from exc
    return candidate


def _verify_sources(policy: Mapping[str, Any], audit: PreAuthorizationAudit) -> None:
    pins = policy.get("source_pins")
    if not isinstance(pins, Mapping) or not pins:
        raise ParityV5R2AuthorizationError("V5R2 source pins missing")
    for raw_path, expected in pins.items():
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ParityV5R2AuthorizationError("V5R2 source-pin schema invalid")
        path = _regular(Path(raw_path), "source pin")
        if sha256_file(path) != expected:
            raise ParityV5R2AuthorizationError(f"source drift: {path}")
        audit.source_hash_checks += 1


def _verify_key(policy: Mapping[str, Any], audit: PreAuthorizationAudit) -> Ed25519PublicKey:
    pin = policy.get("public_key")
    if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256", "bytes"}:
        raise ParityV5R2AuthorizationError("public-key policy malformed")
    path = _regular(Path(str(pin["path"])), "dedicated V5R2 root public key")
    audit.public_key_files_read += 1
    if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
        raise ParityV5R2AuthorizationError("dedicated root public-key pin drift")
    try:
        key = serialization.load_pem_public_key(path.read_bytes())
    except ValueError as exc:
        raise ParityV5R2AuthorizationError("public-key PEM malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ParityV5R2AuthorizationError("dedicated root key is not Ed25519")
    return key


def _load_signed(
    authorization_path: Path,
    signature_path: Path,
    key: Ed25519PublicKey,
    audit: PreAuthorizationAudit,
) -> tuple[dict[str, Any], bytes]:
    auth_file = _regular(authorization_path, "authorization")
    sig_file = _regular(signature_path, "signature")
    raw = auth_file.read_bytes()
    audit.authorization_files_read += 1
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParityV5R2AuthorizationError("authorization canonical JSON malformed") from exc
    if not isinstance(envelope, dict) or _canonical_bytes(envelope) != raw:
        raise ParityV5R2AuthorizationError("authorization must be exact canonical JSON")
    if set(envelope) != {"schema", "authorization"} or envelope.get("schema") != AUTHORIZATION_ENVELOPE_SCHEMA:
        raise ParityV5R2AuthorizationError("authorization V5R2 envelope mismatch")
    body = envelope.get("authorization")
    if not isinstance(body, dict):
        raise ParityV5R2AuthorizationError("authorization body missing")
    try:
        signature = base64.b64decode(sig_file.read_bytes(), validate=True)
    except (OSError, ValueError) as exc:
        raise ParityV5R2AuthorizationError("detached signature malformed") from exc
    audit.signature_files_read += 1
    if len(signature) != 64:
        raise ParityV5R2AuthorizationError("detached signature length mismatch")
    try:
        key.verify(signature, raw)
    except InvalidSignature as exc:
        raise ParityV5R2AuthorizationError("invalid detached V5R2 signature") from exc
    audit.signature_checks += 1
    return body, raw


def _validate_body(
    body: Mapping[str, Any], *, policy: Mapping[str, Any], output_root: Path, now: datetime
) -> None:
    required = {
        "schema", "kind", "status", "authorization_id", "single_use_nonce", "issued_at",
        "expires_at", "permitted_action", "output_root", "execution_policy_sha256", "bindings",
        "external_subm_scoring_permitted", "normalizer_fitting_permitted", "optimizer_or_backward_permitted",
    }
    if set(body) != required:
        raise ParityV5R2AuthorizationError("authorization exact V5R2 key set mismatch")
    if (
        body.get("schema") != AUTHORIZATION_SCHEMA
        or body.get("kind") != "dandi_000688_subc_cpu_explicit_v5_source_closure_parity_execution"
        or body.get("status") != AUTHORIZATION_STATUS
        or body.get("permitted_action") != PERMITTED_ACTION
    ):
        raise ParityV5R2AuthorizationError("authorization V5R2 schema/status/action mismatch")
    if not isinstance(body.get("authorization_id"), str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", body["authorization_id"]) is None:
        raise ParityV5R2AuthorizationError("authorization id malformed")
    if not isinstance(body.get("single_use_nonce"), str) or re.fullmatch(r"[0-9a-f]{64}", body["single_use_nonce"]) is None:
        raise ParityV5R2AuthorizationError("authorization nonce malformed")
    issued = _parse_time(body.get("issued_at"), "issued_at")
    expires = _parse_time(body.get("expires_at"), "expires_at")
    if not (issued <= now <= expires):
        raise ParityV5R2AuthorizationError("authorization expired or not-yet-valid")
    if (expires - issued).total_seconds() <= 0 or (expires - issued).total_seconds() > MAX_VALIDITY_SECONDS:
        raise ParityV5R2AuthorizationError("authorization duration exceeds V5R2 15-minute maximum")
    if (
        body.get("external_subm_scoring_permitted") is not False
        or body.get("normalizer_fitting_permitted") is not False
        or body.get("optimizer_or_backward_permitted") is not False
    ):
        raise ParityV5R2AuthorizationError("V5R2 forbidden capability flag enabled")
    if body.get("bindings") != policy.get("authorization_bindings"):
        raise ParityV5R2AuthorizationError("authorization V5R2 binding drift")
    if body.get("execution_policy_sha256") != policy.get("execution_policy_sha256"):
        raise ParityV5R2AuthorizationError("authorization V5R2 policy SHA drift")
    if body.get("output_root") != str(output_root.resolve()):
        raise ParityV5R2AuthorizationError("authorization output-root mismatch")
    output_parent = Path(str(policy["output_parent"])).resolve()
    _within(output_root, output_parent, "output root")
    if output_root.resolve() == output_parent or output_root.exists():
        raise ParityV5R2AuthorizationError("output root must be a new unique V5R2 child")
    if runtime_identity() != policy.get("runtime_identity"):
        raise ParityV5R2AuthorizationError("SPINT Python/host drift")
    if policy.get("cpu_policy") != CPU_POLICY:
        raise ParityV5R2AuthorizationError("V5R2 CPU policy malformed")


def _claim(
    *, policy: Mapping[str, Any], body: Mapping[str, Any], authorization_path: Path,
    signature_path: Path, audit: PreAuthorizationAudit,
) -> Path:
    output_parent = Path(str(policy["output_parent"])).resolve()
    claim_root = Path(str(policy["claim_root"])).resolve()
    _within(claim_root, output_parent.parent, "claim root")
    claim_root.mkdir(parents=True, exist_ok=True)
    destination = claim_root / f"{body['single_use_nonce']}.json"
    payload = {
        "schema": "dandi_000688_subc_explicit_v5_source_closure_nonce_claim_v5r2",
        "authorization_id": body["authorization_id"],
        "single_use_nonce": body["single_use_nonce"],
        "authorization": _file_pin(authorization_path),
        "signature": _file_pin(signature_path),
        "output_root": body["output_root"],
        "permitted_action": PERMITTED_ACTION,
    }
    raw = _canonical_bytes(payload)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ParityV5R2AuthorizationError("V5R2 nonce already atomically claimed") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(destination, 0o444)
    if stat.S_IMODE(destination.stat().st_mode) != 0o444:
        raise ParityV5R2AuthorizationError("V5R2 nonce-claim mode failure")
    audit.nonce_claim_writes += 1
    return destination


def _preauthorize_and_claim_with_policy(
    *, authorization_path: Path, signature_path: Path, output_root: Path,
    policy: Mapping[str, Any], now: datetime | None = None,
    audit: PreAuthorizationAudit | None = None,
) -> AuthorizationGrant:
    """Test seam; production reaches this only via fixed V5R2 prelaunch."""
    active = audit if audit is not None else PreAuthorizationAudit()
    _assert_preimport_boundary()
    key = _verify_key(policy, active)
    body, raw = _load_signed(authorization_path, signature_path, key, active)
    _verify_sources(policy, active)
    _validate_body(
        body, policy=policy, output_root=output_root,
        now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc),
    )
    claim = _claim(
        policy=policy, body=body, authorization_path=authorization_path,
        signature_path=signature_path, audit=active,
    )
    return AuthorizationGrant(
        authorization=dict(body), authorization_sha256=hashlib.sha256(raw).hexdigest(),
        signature_sha256=sha256_file(_regular(signature_path, "signature")),
        claim_path=claim, audit=active,
    )


def _set_cpu_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ParityV5R2AuthorizationError("CUDA_VISIBLE_DEVICES drifted after authorization")
    for key, value in CPU_ENVIRONMENT.items():
        os.environ[key] = value


def _write_immutable(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise ParityV5R2ExecutionError(f"immutable mode failed: {path.name}")
    return hashlib.sha256(raw).hexdigest()


def _execute_after_authorization_with_policy(
    *, authorization_path: Path, signature_path: Path, output_root: Path, policy: Mapping[str, Any]
) -> dict[str, Any]:
    grant = _preauthorize_and_claim_with_policy(
        authorization_path=authorization_path, signature_path=signature_path,
        output_root=output_root, policy=policy,
    )
    preauthorization_audit = asdict(grant.audit)
    _set_cpu_environment()
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ParityV5R2ExecutionError("V5R2 output collision after nonce claim") from exc
    from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5r2 import (
        concrete_parity_after_future_authorization_v5r2,
    )

    result = concrete_parity_after_future_authorization_v5r2(REPO_ROOT)
    input_path = output_root / "input_trace.json"
    environment_path = output_root / "environment.json"
    receipt_path = output_root / "parity_execution_receipt.json"
    seal_path = output_root / "seal.json"
    input_sha = _write_immutable(input_path, result["input_trace"])
    environment = {
        "schema": "dandi_000688_subc_explicit_v5_source_closure_environment_v5r2",
        "runtime_identity": runtime_identity(), "cpu_policy": policy["cpu_policy"],
        "thread_environment": CPU_ENVIRONMENT, "source_pins": policy["source_pins"],
        "fixture_pins": policy["authorization_bindings"]["fixture_pins"],
        "v5_sources": policy["authorization_bindings"]["v5_sources"],
    }
    environment_sha = _write_immutable(environment_path, environment)
    receipt = {
        "schema": "dandi_000688_subc_explicit_v5_source_closure_execution_receipt_v5r2",
        "status": result["status"],
        "authorization": {
            "authorization_sha256": grant.authorization_sha256,
            "signature_sha256": grant.signature_sha256,
            "authorization_id": grant.authorization["authorization_id"],
            "single_use_nonce": grant.authorization["single_use_nonce"],
            "nonce_claim": _file_pin(grant.claim_path),
        },
        "input_trace": {"path": input_path.name, "sha256": input_sha},
        "environment": {"path": environment_path.name, "sha256": environment_sha},
        "reference": result["reference"], "adapter": result["adapter"],
        "observer": result["observer"], "preauthorization_audit": preauthorization_audit,
        "runtime_imports": {
            "v5r2_helper_imported_after_authorization": True,
            "pinned_v5_v3_torch_imports_after_authorization": True,
        },
        "data_asset_access": {
            "instrumentation": "not_instrumented; open counts are not inferred",
            "logical_pinned_assets_used": ["checkpoint", "teacher", "consumed_subc_nwb", "behavior_normalizer", "t4_normalizer"],
        },
        "external_subm_scoring_performed": False,
    }
    receipt_sha = _write_immutable(receipt_path, receipt)
    seal = {
        "schema": "dandi_000688_subc_explicit_v5_source_closure_execution_seal_v5r2",
        "status": receipt["status"],
        "artifacts": [
            {"path": input_path.name, "sha256": input_sha, "bytes": input_path.stat().st_size, "mode": "0444"},
            {"path": environment_path.name, "sha256": environment_sha, "bytes": environment_path.stat().st_size, "mode": "0444"},
            {"path": receipt_path.name, "sha256": receipt_sha, "bytes": receipt_path.stat().st_size, "mode": "0444"},
        ],
        "external_subm_scoring_performed": False,
    }
    seal_sha = _write_immutable(seal_path, seal)
    return {
        "status": receipt["status"], "output_root": str(output_root),
        "input_trace_sha256": input_sha, "environment_sha256": environment_sha,
        "receipt_sha256": receipt_sha, "seal_sha256": seal_sha,
        "external_subm_scoring_performed": False,
    }


def execute_from_fixed_prelaunch(
    *, authorization_path: Path, signature_path: Path, output_root: Path,
    prelaunch_dir: Path, root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Production entrypoint: only a stored 0444 V5R2 prelaunch supplies policy."""
    if root.resolve() != REPO_ROOT.resolve():
        raise ParityV5R2AuthorizationError("V5R2 production runner requires fixed repository root")
    from sua_exploration.scripts.write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v5r2 import (
        canonical_bytes, load_stored_prelaunch,
    )

    stored = load_stored_prelaunch(prelaunch_dir, root)
    policy = stored["execution_policy"]
    without_sha = {key: value for key, value in policy.items() if key != "execution_policy_sha256"}
    actual_sha = hashlib.sha256(canonical_bytes(without_sha)).hexdigest()
    if (
        policy.get("execution_policy_sha256") != stored.get("execution_policy_sha256")
        or stored.get("execution_policy_sha256") != actual_sha
    ):
        raise ParityV5R2AuthorizationError("stored V5R2 policy hash mismatch")
    return _execute_after_authorization_with_policy(
        authorization_path=authorization_path, signature_path=signature_path,
        output_root=output_root, policy=policy,
    )
