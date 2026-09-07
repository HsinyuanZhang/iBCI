"""Authorization-first CPU execution support for scorer-adapter parity v4.

This module imports only standard-library/cryptography code. It deliberately
does not import parity v3, Torch, NumPy, PyNWB, checkpoint loaders, normalizer
loaders, or data owners at module import time. The production entrypoint
validates and atomically claims a signed capability before it dynamically
imports the v3 execution body.
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
AUTHORIZATION_SCHEMA = "dandi_000688_subc_parity_execution_authorization_v4"
AUTHORIZATION_ENVELOPE_SCHEMA = "dandi_000688_subc_parity_execution_authorization_envelope_v4"
PERMITTED_ACTION = "concrete_parity_after_future_authorization"
AUTHORIZATION_STATUS = "AUTHORIZED_FOR_ONE_CONSUMED_SUBC_PARITY_EXECUTION"
CPU_THREAD_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class ParityV4AuthorizationError(PermissionError):
    """The signed capability is absent, malformed, expired, replayed, or drifted."""


class ParityV4ExecutionError(RuntimeError):
    """A post-authorization CPU parity execution or seal write failed."""


@dataclass
class PreAuthorizationAudit:
    authorization_files_read: int = 0
    signature_files_read: int = 0
    public_key_files_read: int = 0
    prelaunch_hash_checks: int = 0
    source_hash_checks: int = 0
    signature_checks: int = 0
    nonce_claim_writes: int = 0
    v3_helper_imports: int = 0
    torch_or_owner_imports: int = 0
    checkpoint_files_opened: int = 0
    nwb_files_opened: int = 0
    npz_files_opened: int = 0
    model_forward_calls: int = 0

    def zero_data_access(self) -> bool:
        return (
            self.v3_helper_imports == 0
            and self.torch_or_owner_imports == 0
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
    nonce_claim_path: Path
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


def _require_regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ParityV4AuthorizationError(f"{label} missing, non-regular, or symlinked")
    return resolved


def _file_pin(path: Path) -> dict[str, Any]:
    checked = _require_regular_file(path, "pinned file")
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
        raise ParityV4AuthorizationError(f"{label} must be ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ParityV4AuthorizationError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ParityV4AuthorizationError(f"{label} must include timezone")
    return parsed.astimezone(timezone.utc)


def _assert_preimport_runtime() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ParityV4AuthorizationError("CUDA_VISIBLE_DEVICES must be empty before authorization/runtime import")
    loaded = set(sys.modules)
    if "sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v3" in loaded:
        raise ParityV4AuthorizationError("v3 helper was imported before authorization")
    if any(name == "torch" or name.startswith("torch.") for name in loaded):
        raise ParityV4AuthorizationError("Torch was imported before authorization")


def _require_within(path: Path, parent: Path, label: str) -> Path:
    candidate = path.resolve()
    root = parent.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ParityV4AuthorizationError(f"{label} escapes dedicated v4 namespace") from exc
    return candidate


def _verify_policy_sources(policy: Mapping[str, Any], audit: PreAuthorizationAudit) -> None:
    sources = policy.get("source_pins")
    if not isinstance(sources, Mapping) or not sources:
        raise ParityV4AuthorizationError("execution policy source pins missing")
    for raw_path, expected in sources.items():
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ParityV4AuthorizationError("execution policy source-pin schema invalid")
        path = _require_regular_file(Path(raw_path), "source pin")
        if sha256_file(path) != expected:
            raise ParityV4AuthorizationError(f"source drift: {path}")
        audit.source_hash_checks += 1


def _verify_public_key(policy: Mapping[str, Any], audit: PreAuthorizationAudit) -> Ed25519PublicKey:
    pin = policy.get("public_key")
    if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256", "bytes"}:
        raise ParityV4AuthorizationError("execution policy public-key pin malformed")
    key_path = _require_regular_file(Path(str(pin["path"])), "root public key")
    audit.public_key_files_read += 1
    if key_path.stat().st_size != pin["bytes"] or sha256_file(key_path) != pin["sha256"]:
        raise ParityV4AuthorizationError("root public-key pin drift")
    try:
        key = serialization.load_pem_public_key(key_path.read_bytes())
    except ValueError as exc:
        raise ParityV4AuthorizationError("root public-key PEM invalid") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ParityV4AuthorizationError("root trust anchor is not Ed25519")
    return key


def _decode_authorization(
    authorization_path: Path,
    signature_path: Path,
    key: Ed25519PublicKey,
    audit: PreAuthorizationAudit,
) -> tuple[dict[str, Any], bytes]:
    authorization_file = _require_regular_file(authorization_path, "authorization")
    signature_file = _require_regular_file(signature_path, "authorization signature")
    raw = authorization_file.read_bytes()
    audit.authorization_files_read += 1
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParityV4AuthorizationError("authorization is not UTF-8 canonical JSON") from exc
    if not isinstance(envelope, dict) or _canonical_bytes(envelope) != raw:
        raise ParityV4AuthorizationError("authorization must be exact canonical JSON bytes")
    if set(envelope) != {"schema", "authorization"} or envelope.get("schema") != AUTHORIZATION_ENVELOPE_SCHEMA:
        raise ParityV4AuthorizationError("authorization envelope schema mismatch")
    authorization = envelope.get("authorization")
    if not isinstance(authorization, dict):
        raise ParityV4AuthorizationError("authorization body is not an object")
    try:
        signature = base64.b64decode(signature_file.read_bytes(), validate=True)
    except (ValueError, OSError) as exc:
        raise ParityV4AuthorizationError("detached signature is not strict base64") from exc
    audit.signature_files_read += 1
    if len(signature) != 64:
        raise ParityV4AuthorizationError("detached Ed25519 signature must be exactly 64 bytes")
    try:
        key.verify(signature, raw)
    except InvalidSignature as exc:
        raise ParityV4AuthorizationError("invalid detached Ed25519 authorization signature") from exc
    audit.signature_checks += 1
    return authorization, raw


def _validate_authorization_body(
    authorization: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    output_root: Path,
    now: datetime,
    audit: PreAuthorizationAudit,
) -> None:
    expected_keys = {
        "schema",
        "kind",
        "status",
        "authorization_id",
        "single_use_nonce",
        "issued_at",
        "expires_at",
        "permitted_action",
        "output_root",
        "execution_policy_sha256",
        "bindings",
        "external_subm_scoring_permitted",
        "normalizer_fitting_permitted",
        "optimizer_or_backward_permitted",
    }
    if set(authorization) != expected_keys:
        raise ParityV4AuthorizationError("authorization exact key set mismatch")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("kind") != "dandi_000688_subc_cpu_parity_execution"
        or authorization.get("status") != AUTHORIZATION_STATUS
        or authorization.get("permitted_action") != PERMITTED_ACTION
    ):
        raise ParityV4AuthorizationError("authorization schema/status/action mismatch")
    identifier = authorization.get("authorization_id")
    nonce = authorization.get("single_use_nonce")
    if not isinstance(identifier, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", identifier) is None:
        raise ParityV4AuthorizationError("authorization identifier malformed")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise ParityV4AuthorizationError("authorization nonce must be 256-bit lowercase hex")
    issued = _parse_time(authorization.get("issued_at"), "issued_at")
    expires = _parse_time(authorization.get("expires_at"), "expires_at")
    if not (issued <= now <= expires):
        raise ParityV4AuthorizationError("authorization expired or not yet valid")
    if (expires - issued).total_seconds() <= 0 or (expires - issued).total_seconds() > MAX_VALIDITY_SECONDS:
        raise ParityV4AuthorizationError("authorization validity must be positive and at most 15 minutes")
    if authorization.get("external_subm_scoring_permitted") is not False:
        raise ParityV4AuthorizationError("external sub-M scoring is prohibited")
    if authorization.get("normalizer_fitting_permitted") is not False:
        raise ParityV4AuthorizationError("normalizer fitting is prohibited")
    if authorization.get("optimizer_or_backward_permitted") is not False:
        raise ParityV4AuthorizationError("optimizer/backward is prohibited")
    if authorization.get("bindings") != policy.get("authorization_bindings"):
        raise ParityV4AuthorizationError("authorization prelaunch/source/fixture/runtime binding drift")
    if authorization.get("execution_policy_sha256") != policy.get("execution_policy_sha256"):
        raise ParityV4AuthorizationError("authorization fixed execution-policy SHA-256 drift")
    if authorization.get("output_root") != str(output_root.resolve()):
        raise ParityV4AuthorizationError("authorization output-root binding drift")
    output_parent = Path(str(policy["output_parent"])).resolve()
    _require_within(output_root, output_parent, "authorization output root")
    if output_root.resolve() == output_parent:
        raise ParityV4AuthorizationError("output root must be a unique child directory")
    if output_root.exists():
        raise ParityV4AuthorizationError("authorized output root already exists; overwrite/retry forbidden")
    if runtime_identity() != policy.get("runtime_identity"):
        raise ParityV4AuthorizationError("SPINT Python or host identity drift")
    cpu_policy = policy.get("cpu_policy")
    if cpu_policy != {
        "device": "cpu",
        "cuda_visible_devices": "",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "deterministic_algorithms": True,
        "tf32": False,
        "autocast": False,
    }:
        raise ParityV4AuthorizationError("execution policy CPU boundary malformed")
    audit.prelaunch_hash_checks += 1


def _write_claim(
    *,
    policy: Mapping[str, Any],
    authorization: Mapping[str, Any],
    authorization_path: Path,
    signature_path: Path,
    audit: PreAuthorizationAudit,
) -> Path:
    claim_root = Path(str(policy["claim_root"])).resolve()
    output_parent = Path(str(policy["output_parent"])).resolve()
    _require_within(claim_root, output_parent.parent, "nonce claim root")
    nonce = str(authorization["single_use_nonce"])
    claim = claim_root / f"{nonce}.json"
    payload = {
        "schema": "dandi_000688_subc_cpu_parity_nonce_claim_v4",
        "authorization_id": authorization["authorization_id"],
        "single_use_nonce": nonce,
        "authorization": _file_pin(authorization_path),
        "signature": _file_pin(signature_path),
        "output_root": authorization["output_root"],
        "permitted_action": PERMITTED_ACTION,
    }
    claim_root.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(payload)
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ParityV4AuthorizationError("authorization nonce already atomically claimed") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(claim, 0o444)
    if stat.S_IMODE(claim.stat().st_mode) != 0o444:
        raise ParityV4AuthorizationError("nonce claim immutable-mode failure")
    audit.nonce_claim_writes += 1
    return claim


def _preauthorize_and_claim_with_policy(
    *,
    authorization_path: Path,
    signature_path: Path,
    output_root: Path,
    policy: Mapping[str, Any],
    now: datetime | None = None,
    audit: PreAuthorizationAudit | None = None,
) -> AuthorizationGrant:
    """Verify then atomically claim without importing a data/model runtime."""
    active_audit = audit if audit is not None else PreAuthorizationAudit()
    _assert_preimport_runtime()
    key = _verify_public_key(policy, active_audit)
    authorization, raw = _decode_authorization(
        authorization_path, signature_path, key, active_audit
    )
    _verify_policy_sources(policy, active_audit)
    _validate_authorization_body(
        authorization,
        policy=policy,
        output_root=output_root,
        now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc),
        audit=active_audit,
    )
    claim = _write_claim(
        policy=policy,
        authorization=authorization,
        authorization_path=authorization_path,
        signature_path=signature_path,
        audit=active_audit,
    )
    return AuthorizationGrant(
        authorization=dict(authorization),
        authorization_sha256=hashlib.sha256(raw).hexdigest(),
        signature_sha256=sha256_file(_require_regular_file(signature_path, "authorization signature")),
        nonce_claim_path=claim,
        audit=active_audit,
    )


def _set_cpu_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ParityV4AuthorizationError("CUDA_VISIBLE_DEVICES drifted after authorization")
    for key, value in CPU_THREAD_ENVIRONMENT.items():
        os.environ[key] = value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise ParityV4ExecutionError(f"immutable mode failed: {path.name}")
    return hashlib.sha256(raw).hexdigest()


def _execute_after_authorization_with_policy(
    *,
    authorization_path: Path,
    signature_path: Path,
    output_root: Path,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """The only v4 path allowed to import and call v3 concrete parity once."""
    grant = _preauthorize_and_claim_with_policy(
        authorization_path=authorization_path,
        signature_path=signature_path,
        output_root=output_root,
        policy=policy,
    )
    # This immutable snapshot describes the whole pre-import phase.  It is the
    # only audit record used to establish that rejected capabilities performed
    # no data access.  Runtime loader opens are deliberately not guessed here.
    preauthorization_audit = asdict(grant.audit)
    _set_cpu_environment()
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ParityV4ExecutionError("output-root collision after nonce claim") from exc
    # This import is intentionally after signature, full binding verification,
    # freshness checks, and atomic nonce consumption.
    from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v3 import (
        concrete_parity_after_future_authorization,
    )

    result = concrete_parity_after_future_authorization(REPO_ROOT)
    input_path = output_root / "input_trace.json"
    environment_path = output_root / "environment.json"
    receipt_path = output_root / "parity_execution_receipt.json"
    seal_path = output_root / "seal.json"
    input_sha = _write_immutable_json(input_path, result["input_trace"])
    environment = {
        "schema": "dandi_000688_subc_cpu_parity_environment_v4",
        "runtime_identity": runtime_identity(),
        "cpu_policy": policy["cpu_policy"],
        "thread_environment": dict(CPU_THREAD_ENVIRONMENT),
        "source_pins": dict(policy["source_pins"]),
        "fixture_pins": policy["authorization_bindings"]["fixture_pins"],
    }
    environment_sha = _write_immutable_json(environment_path, environment)
    receipt = {
        "schema": "dandi_000688_subc_cpu_parity_execution_receipt_v4",
        "status": "PARITY_CONFIRMED_CONSUMED_SUBC_DEV_SESSION_PENDING_SEPARATE_ROOT_REVIEW",
        "authorization": {
            "authorization_sha256": grant.authorization_sha256,
            "signature_sha256": grant.signature_sha256,
            "authorization_id": grant.authorization["authorization_id"],
            "single_use_nonce": grant.authorization["single_use_nonce"],
            "nonce_claim": _file_pin(grant.nonce_claim_path),
        },
        "input_trace": {"path": input_path.name, "sha256": input_sha},
        "environment": {"path": environment_path.name, "sha256": environment_sha},
        "reference": result["reference"],
        "adapter": result["adapter"],
        "observer": result["observer"],
        "preauthorization_audit": preauthorization_audit,
        "runtime_imports": {
            "v3_helper_imported_after_authorization": True,
            "torch_and_owner_imports_occurred_after_authorization": True,
        },
        "data_asset_access": {
            "instrumentation": "not_instrumented; no checkpoint/NWB/NPZ open counts are inferred or fabricated",
            "logical_pinned_assets_used": ["checkpoint", "teacher", "consumed_subc_nwb", "behavior_normalizer", "t4_normalizer"],
        },
        "external_subm_scoring_performed": False,
    }
    receipt_sha = _write_immutable_json(receipt_path, receipt)
    seal = {
        "schema": "dandi_000688_subc_cpu_parity_execution_seal_v4",
        "status": receipt["status"],
        "artifacts": [
            {"path": input_path.name, "sha256": input_sha, "bytes": input_path.stat().st_size, "mode": "0444"},
            {"path": environment_path.name, "sha256": environment_sha, "bytes": environment_path.stat().st_size, "mode": "0444"},
            {"path": receipt_path.name, "sha256": receipt_sha, "bytes": receipt_path.stat().st_size, "mode": "0444"},
        ],
        "external_subm_scoring_performed": False,
    }
    seal_sha = _write_immutable_json(seal_path, seal)
    return {
        "status": receipt["status"],
        "output_root": str(output_root),
        "input_trace_sha256": input_sha,
        "environment_sha256": environment_sha,
        "receipt_sha256": receipt_sha,
        "seal_sha256": seal_sha,
        "external_subm_scoring_performed": False,
    }


def execute_from_fixed_prelaunch(
    *,
    authorization_path: Path,
    signature_path: Path,
    output_root: Path,
    prelaunch_dir: Path,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Production-only entrypoint: policy originates in fixed 0444 prelaunch.

    The writer performs the stored-bundle mode/seal/source verification and
    returns a content-addressed policy.  No caller-supplied policy or public
    key can reach the production runner.
    """
    if root.resolve() != REPO_ROOT.resolve():
        raise ParityV4AuthorizationError("production v4 runner requires the fixed repository root")
    from sua_exploration.scripts.write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v4 import (
        canonical_bytes,
        load_stored_prelaunch,
    )

    stored = load_stored_prelaunch(prelaunch_dir, root)
    policy = stored["execution_policy"]
    policy_without_sha = {
        key: value for key, value in policy.items() if key != "execution_policy_sha256"
    }
    actual_policy_sha256 = hashlib.sha256(canonical_bytes(policy_without_sha)).hexdigest()
    if (
        policy.get("execution_policy_sha256") != stored.get("execution_policy_sha256")
        or stored.get("execution_policy_sha256") != actual_policy_sha256
    ):
        raise ParityV4AuthorizationError("stored fixed execution-policy SHA-256 mismatch")
    return _execute_after_authorization_with_policy(
        authorization_path=authorization_path,
        signature_path=signature_path,
        output_root=output_root,
        policy=policy,
    )
