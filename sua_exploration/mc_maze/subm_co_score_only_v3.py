"""Authorization-first V3 adapter for the frozen external sub-M score matrix.

Importing this module uses only the standard library and cryptography.  The
score-only V2 implementation, parity-proven V5 bridge, C1 owners, NumPy,
Torch, external NWB assets, checkpoints, and normalizers remain unreachable
until an exact V3 detached authorization has been validated and a fresh nonce
has been atomically claimed.
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
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_VALIDITY_SECONDS = 15 * 60
AUTHORIZATION_SCHEMA = "dandi_000688_subm_co_score_only_execution_authorization_v3"
AUTHORIZATION_ENVELOPE_SCHEMA = "dandi_000688_subm_co_score_only_execution_authorization_envelope_v3"
AUTHORIZATION_STATUS = "AUTHORIZED_FOR_ONE_FROZEN_SUBM_SCORE_ONLY_CPU_EXECUTION_V3"
PERMITTED_ACTION = "score_frozen_subm_matrix_via_v5_bridge_v3"
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


class ScoreV3AuthorizationError(PermissionError):
    """A V3 capability failed before any external score runtime is reachable."""


class ScoreV3ExecutionError(RuntimeError):
    """A post-authorization V3 score operation or immutable write failed."""


@dataclass
class PreAuthorizationAudit:
    authorization_files_read: int = 0
    signature_files_read: int = 0
    public_key_files_read: int = 0
    source_hash_checks: int = 0
    signature_checks: int = 0
    nonce_claim_writes: int = 0
    v5_bridge_runtime_imports: int = 0
    score_runtime_imports: int = 0
    external_nwb_files_opened: int = 0
    checkpoint_files_opened: int = 0
    normalizer_files_opened: int = 0
    model_forward_calls: int = 0
    r2_computations: int = 0

    def zero_data_or_runtime_access(self) -> bool:
        return (
            self.v5_bridge_runtime_imports == 0
            and self.score_runtime_imports == 0
            and self.external_nwb_files_opened == 0
            and self.checkpoint_files_opened == 0
            and self.normalizer_files_opened == 0
            and self.model_forward_calls == 0
            and self.r2_computations == 0
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
        raise ScoreV3AuthorizationError(f"{label} missing") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ScoreV3AuthorizationError(f"{label} non-regular or symlinked")
    return resolved


def _file_pin(path: Path) -> dict[str, Any]:
    checked = _regular(path, "pinned file")
    return {"path": str(checked), "sha256": sha256_file(checked), "bytes": checked.stat().st_size}


def runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "host": socket.gethostname(),
        "python": {
            "path": str(executable), "sha256": sha256_file(executable), "bytes": executable.stat().st_size,
            "implementation": getattr(sys.implementation, "name", ""), "version": sys.version,
        },
    }


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ScoreV3AuthorizationError(f"{label} must be timezone-aware ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScoreV3AuthorizationError(f"{label} malformed") from exc
    if parsed.tzinfo is None:
        raise ScoreV3AuthorizationError(f"{label} lacks timezone")
    return parsed.astimezone(timezone.utc)


def _assert_preimport_boundary() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ScoreV3AuthorizationError("CUDA_VISIBLE_DEVICES must be empty before V3 authorization")
    forbidden = (
        "mc_maze.subm_co_score_only_v2",
        "mc_maze.subm_co_score_only",
        "sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5",
        "sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v3",
        "torch",
        "numpy",
    )
    loaded = set(sys.modules)
    if any(name == item or name.startswith(item + ".") for item in forbidden for name in loaded):
        raise ScoreV3AuthorizationError("score/V5/Torch/NumPy runtime imported before V3 authorization")


def _within(path: Path, parent: Path, label: str) -> Path:
    candidate, root = path.resolve(), parent.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ScoreV3AuthorizationError(f"{label} escapes V3 dedicated namespace") from exc
    return candidate


def _verify_sources(policy: Mapping[str, Any], audit: PreAuthorizationAudit) -> None:
    pins = policy.get("source_pins")
    if not isinstance(pins, Mapping) or not pins:
        raise ScoreV3AuthorizationError("V3 source pins missing")
    for raw_path, expected in pins.items():
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ScoreV3AuthorizationError("V3 source-pin schema invalid")
        path = _regular(Path(raw_path), "source pin")
        if sha256_file(path) != expected:
            raise ScoreV3AuthorizationError(f"source drift: {path}")
        audit.source_hash_checks += 1


def _verify_key(policy: Mapping[str, Any], audit: PreAuthorizationAudit) -> Ed25519PublicKey:
    pin = policy.get("public_key")
    if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256", "bytes"}:
        raise ScoreV3AuthorizationError("public-key policy malformed")
    path = _regular(Path(str(pin["path"])), "dedicated V3 root public key")
    audit.public_key_files_read += 1
    if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
        raise ScoreV3AuthorizationError("dedicated V3 public-key pin drift")
    try:
        key = serialization.load_pem_public_key(path.read_bytes())
    except ValueError as exc:
        raise ScoreV3AuthorizationError("public-key PEM malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ScoreV3AuthorizationError("dedicated root key is not Ed25519")
    return key


def _load_signed(
    authorization_path: Path, signature_path: Path, key: Ed25519PublicKey, audit: PreAuthorizationAudit
) -> tuple[dict[str, Any], bytes]:
    auth_file = _regular(authorization_path, "authorization")
    sig_file = _regular(signature_path, "signature")
    raw = auth_file.read_bytes()
    audit.authorization_files_read += 1
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreV3AuthorizationError("authorization canonical JSON malformed") from exc
    if not isinstance(envelope, dict) or _canonical_bytes(envelope) != raw:
        raise ScoreV3AuthorizationError("authorization must be exact canonical JSON")
    if set(envelope) != {"schema", "authorization"} or envelope.get("schema") != AUTHORIZATION_ENVELOPE_SCHEMA:
        raise ScoreV3AuthorizationError("authorization V3 envelope mismatch")
    body = envelope.get("authorization")
    if not isinstance(body, dict):
        raise ScoreV3AuthorizationError("authorization body missing")
    try:
        signature = base64.b64decode(sig_file.read_bytes(), validate=True)
    except (OSError, ValueError) as exc:
        raise ScoreV3AuthorizationError("detached signature malformed") from exc
    audit.signature_files_read += 1
    if len(signature) != 64:
        raise ScoreV3AuthorizationError("detached signature length mismatch")
    try:
        key.verify(signature, raw)
    except InvalidSignature as exc:
        raise ScoreV3AuthorizationError("invalid detached V3 signature") from exc
    audit.signature_checks += 1
    return body, raw


def _validate_body(
    body: Mapping[str, Any], *, policy: Mapping[str, Any], output_root: Path,
    external_nwb_root: Path, now: datetime,
) -> None:
    required = {
        "schema", "kind", "status", "authorization_id", "single_use_nonce", "issued_at", "expires_at",
        "permitted_action", "output_root", "external_nwb_root", "execution_policy_sha256", "bindings",
        "external_subm_scoring_permitted", "cpu_forward_r2_only", "normalizer_fitting_permitted",
        "optimizer_or_backward_permitted", "target_updates_permitted",
    }
    if set(body) != required:
        raise ScoreV3AuthorizationError("authorization exact V3 key set mismatch")
    if (
        body.get("schema") != AUTHORIZATION_SCHEMA
        or body.get("kind") != "dandi_000688_subm_co_external_score_only_authorization_v3"
        or body.get("status") != AUTHORIZATION_STATUS
        or body.get("permitted_action") != PERMITTED_ACTION
    ):
        raise ScoreV3AuthorizationError("authorization V3 schema/status/action mismatch")
    if not isinstance(body.get("authorization_id"), str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", body["authorization_id"]) is None:
        raise ScoreV3AuthorizationError("authorization id malformed")
    if not isinstance(body.get("single_use_nonce"), str) or re.fullmatch(r"[0-9a-f]{64}", body["single_use_nonce"]) is None:
        raise ScoreV3AuthorizationError("authorization nonce malformed")
    issued, expires = _parse_time(body.get("issued_at"), "issued_at"), _parse_time(body.get("expires_at"), "expires_at")
    if not (issued <= now <= expires):
        raise ScoreV3AuthorizationError("authorization expired or not-yet-valid")
    if (expires - issued).total_seconds() <= 0 or (expires - issued).total_seconds() > MAX_VALIDITY_SECONDS:
        raise ScoreV3AuthorizationError("authorization duration exceeds V3 15-minute maximum")
    if (
        body.get("external_subm_scoring_permitted") is not True
        or body.get("cpu_forward_r2_only") is not True
        or body.get("normalizer_fitting_permitted") is not False
        or body.get("optimizer_or_backward_permitted") is not False
        or body.get("target_updates_permitted") is not False
    ):
        raise ScoreV3AuthorizationError("V3 capability flags drift from CPU-forward/R2-only policy")
    if body.get("bindings") != policy.get("authorization_bindings"):
        raise ScoreV3AuthorizationError("authorization V3 binding drift")
    if body.get("execution_policy_sha256") != policy.get("execution_policy_sha256"):
        raise ScoreV3AuthorizationError("authorization V3 policy SHA drift")
    if body.get("output_root") != str(output_root.resolve()):
        raise ScoreV3AuthorizationError("authorization output-root mismatch")
    if body.get("external_nwb_root") != str(external_nwb_root.resolve()):
        raise ScoreV3AuthorizationError("authorization external-NWB-root mismatch")
    output_parent = Path(str(policy["output_parent"])).resolve()
    _within(output_root, output_parent, "output root")
    if output_root.resolve() == output_parent or output_root.exists():
        raise ScoreV3AuthorizationError("output root must be a new unique V3 child")
    if runtime_identity() != policy.get("runtime_identity"):
        raise ScoreV3AuthorizationError("SPINT Python/host drift")
    if policy.get("cpu_policy") != CPU_POLICY:
        raise ScoreV3AuthorizationError("V3 CPU policy malformed")


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
        "schema": "dandi_000688_subm_co_score_only_nonce_claim_v3",
        "authorization_id": body["authorization_id"], "single_use_nonce": body["single_use_nonce"],
        "authorization": _file_pin(authorization_path), "signature": _file_pin(signature_path),
        "output_root": body["output_root"], "external_nwb_root": body["external_nwb_root"],
        "permitted_action": PERMITTED_ACTION,
    }
    raw = _canonical_bytes(payload)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ScoreV3AuthorizationError("V3 nonce already atomically claimed") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(destination, 0o444)
    if stat.S_IMODE(destination.stat().st_mode) != 0o444:
        raise ScoreV3AuthorizationError("V3 nonce-claim mode failure")
    audit.nonce_claim_writes += 1
    return destination


def _preauthorize_and_claim_with_policy(
    *, authorization_path: Path, signature_path: Path, output_root: Path, external_nwb_root: Path,
    policy: Mapping[str, Any], now: datetime | None = None,
    audit: PreAuthorizationAudit | None = None,
) -> AuthorizationGrant:
    """Test seam; production policy is reconstructed from the stored V3 bundle."""
    active = audit if audit is not None else PreAuthorizationAudit()
    _assert_preimport_boundary()
    key = _verify_key(policy, active)
    body, raw = _load_signed(authorization_path, signature_path, key, active)
    _verify_sources(policy, active)
    _validate_body(
        body, policy=policy, output_root=output_root, external_nwb_root=external_nwb_root,
        now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc),
    )
    claim = _claim(policy=policy, body=body, authorization_path=authorization_path, signature_path=signature_path, audit=active)
    return AuthorizationGrant(
        authorization=dict(body), authorization_sha256=hashlib.sha256(raw).hexdigest(),
        signature_sha256=sha256_file(_regular(signature_path, "signature")), claim_path=claim, audit=active,
    )


def _set_cpu_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ScoreV3AuthorizationError("CUDA_VISIBLE_DEVICES drifted after authorization")
    for key, value in CPU_ENVIRONMENT.items():
        os.environ[key] = value


def prepare_external_session_with_v5_bridge_v3(
    *, nwb_path: Path, view: str, feature_group: str, permutation_seed: int | None,
    behavior_mean: Any, behavior_std: Any, side_mean: Any, side_std: Any,
) -> tuple[Any, dict[str, Any]]:
    """Build one future external fixture through the parity-proven V5 bridge.

    This function is post-authorization only. It deliberately uses the exact
    V5 owner chronology bridge before the pre-existing C1 calibration builder;
    it does not call the older score-only adapter loader.
    """
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity_v3
    from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5 import (
        bridge_owner_chronology_for_c1_builder,
    )

    if view not in {"sua", "pseudo_mua"} or feature_group not in {"t4", "ts4"}:
        raise ScoreV3ExecutionError("future V3 view/feature-group contract drift")
    owners = parity_v3._runtime_owners()
    record = owners["load_dandi688_session"](
        nwb_path, bin_size_ms=parity_v3.BIN_SIZE_MS, window_size=parity_v3.HISTORY_BINS,
        calibration_n_trials=parity_v3.SUPPORT_TRIALS, max_trial_length=parity_v3.TRIAL_LENGTH_BINS,
        pad_value=parity_v3.PAD_VALUE, interpolate_trials=True, behavior_mean=behavior_mean,
        behavior_std=behavior_std, trial_result_filter="R", exclude_calibration_trials_from_windows=True,
        cache_dir=None, signal_view=view,
    )
    raw_trials = owners["list_datamodule_rewarded_trials"](
        nwb_path, bin_size_ms=parity_v3.BIN_SIZE_MS, window_size=parity_v3.HISTORY_BINS,
        trial_result_filter="R",
    )
    raw_evidence, builder_trials, bridge_trace = bridge_owner_chronology_for_c1_builder(raw_trials)
    indices = owners["selection"].select_calibration_trial_indices(
        builder_trials, parity_v3.IDENTITY_TRIALS, parity_v3.SUPPORT_TRIALS, "first"
    )
    rebuilt_calibration = owners["c1"].build_calib_trials_for_indices(
        {"neural": record.neural, "trials": builder_trials, "n_units": int(record.neural.shape[1])},
        indices, parity_v3.IDENTITY_TRIALS,
    )
    features, _metadata = owners["load_unit_side_features"](
        nwb_path, feature_group=feature_group, pool_size=parity_v3.SUPPORT_TRIALS,
        mean=side_mean, std=side_std, cache_dir=None, permutation_seed=permutation_seed,
        bin_size_ms=parity_v3.BIN_SIZE_MS, window_size=parity_v3.HISTORY_BINS,
        trial_result_filter="R", signal_view=view,
    )
    dataset = owners["MCMazeSessionDataset"](
        neural_data=record.neural, behavior_data=record.behavior, valid_starts=record.valid_starts,
        calib_trials=rebuilt_calibration, window_size=parity_v3.HISTORY_BINS,
        session_name=record.name, side_features=features, electrode_ids=None,
    )
    bridge_trace.update({
        "raw_owner_trial_count": len(raw_evidence), "c1_builder_trial_count": len(builder_trials),
        "selection_indices": list(indices), "support_trials": parity_v3.SUPPORT_TRIALS,
        "identity_trials": parity_v3.IDENTITY_TRIALS,
        "query_boundary": "owner-loader valid_starts after first 50 rewarded trials",
        "future_v3_view": view, "future_v3_feature_group": feature_group,
    })
    return dataset, bridge_trace


def score_frozen_subm_matrix_via_v5_bridge_v3(
    *, root: Path, external_nwb_root: Path, output_root: Path, grant: AuthorizationGrant,
) -> dict[str, Any]:
    """Reserved post-authorization execution hook for the frozen 180-cell matrix.

    The static V3 delivery intentionally does not execute this hook. Any later
    implementation must call ``prepare_external_session_with_v5_bridge_v3``
    for every cell's owner fixture and can perform only CPU frozen forwards/R2.
    """
    del root, external_nwb_root, output_root, grant
    raise ScoreV3ExecutionError(
        "V3 static package is prelaunch-only; a separately reviewed external-score executor is required"
    )


def execute_from_fixed_prelaunch(
    *, authorization_path: Path, signature_path: Path, output_root: Path, external_nwb_root: Path,
    prelaunch_dir: Path, root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Production entrypoint: policy comes only from stored immutable V3 evidence."""
    if root.resolve() != REPO_ROOT.resolve():
        raise ScoreV3AuthorizationError("V3 production runner requires fixed repository root")
    from sua_exploration.scripts.write_dandi688_subm_co_score_only_prelaunch_v3 import (
        canonical_bytes, load_stored_prelaunch,
    )

    stored = load_stored_prelaunch(prelaunch_dir, root)
    policy = stored["execution_policy"]
    without_sha = {key: value for key, value in policy.items() if key != "execution_policy_sha256"}
    actual_sha = hashlib.sha256(canonical_bytes(without_sha)).hexdigest()
    if policy.get("execution_policy_sha256") != stored.get("execution_policy_sha256") or actual_sha != stored.get("execution_policy_sha256"):
        raise ScoreV3AuthorizationError("stored V3 policy hash mismatch")
    grant = _preauthorize_and_claim_with_policy(
        authorization_path=authorization_path, signature_path=signature_path, output_root=output_root,
        external_nwb_root=external_nwb_root, policy=policy,
    )
    _set_cpu_environment()
    return score_frozen_subm_matrix_via_v5_bridge_v3(
        root=root, external_nwb_root=external_nwb_root, output_root=output_root, grant=grant,
    )
