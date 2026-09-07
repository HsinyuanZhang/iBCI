"""Control plane for the blocked three-arm external sub-M score-only V4.

V4 is append-only preparation, not an external scoring capability.  Its
current checkpoint table deliberately has three empty shared-zero4 terminal
hash slots, so the fixed prelaunch must remain blocked.  This module contains
the reviewable authorization-first and atomic-ledger primitives required by a
later append-only completion, but does not import Torch/NumPy/data owners, open
an NWB/checkpoint, run a model, compute R2, or create a signature.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import sys
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS = ("shared_t4", "shared_zero4", "shared_ts4")
VIEWS = ("sua", "pseudo_mua")
SEEDS = (42, 43, 44)
EXPECTED_SESSION_COUNT = 15
EXPECTED_CELL_COUNT = EXPECTED_SESSION_COUNT * len(VIEWS) * len(SEEDS) * len(ARMS)
EXPECTED_QUERY_WINDOWS_PER_VIEW = 708_795
EXPECTED_TOTAL_MODEL_WINDOWS = EXPECTED_QUERY_WINDOWS_PER_VIEW * len(VIEWS) * len(SEEDS) * len(ARMS)
MAX_AUTHORIZATION_SECONDS = 15 * 60

AUTHORIZATION_SCHEMA = "dandi_000688_subm_three_arm_cpu_score_authorization_v4"
AUTHORIZATION_ENVELOPE_SCHEMA = "dandi_000688_subm_three_arm_cpu_score_authorization_envelope_v4"
AUTHORIZATION_STATUS = "AUTHORIZED_FOR_ONE_THREE_ARM_CPU_SCORE_V4"
PERMITTED_ACTION = "score_frozen_subm_three_arm_matrix_v4"

CPU_POLICY = {
    "device": "cpu",
    "cuda_visible_devices": "",
    "torch_num_threads": 1,
    "torch_num_interop_threads": 1,
    "deterministic_algorithms": True,
    "tf32": False,
    "autocast": False,
    "normalizer_fitting": False,
    "optimizer_or_backward": False,
    "target_updates": False,
}

COMPARISON_GATES = {
    "t4_minus_zero4": {
        "claim": "descriptor-present versus direct standardized neutral coordinate",
        "delta": "R2(shared_t4)-R2(shared_zero4)",
        "views_evaluated_separately": list(VIEWS),
        "grand_paired_mean_minimum_r2": 0.03,
        "all_three_seed_means_strictly_positive": True,
        "session_cross_seed_positive_fraction_minimum": 0.75,
        "hierarchical_session_seed_bootstrap_lower_95_strictly_positive": True,
        "shared_t4_absolute_grand_and_seed_means_strictly_positive": True,
    },
    "t4_minus_ts4": {
        "claim": "correct descriptor attachment/content versus complete-row permutation",
        "delta": "R2(shared_t4)-R2(shared_ts4)",
        "views_evaluated_separately": list(VIEWS),
        "grand_paired_mean_minimum_r2": 0.03,
        "all_three_seed_means_strictly_positive": True,
        "session_cross_seed_positive_fraction_minimum": 0.75,
        "hierarchical_session_seed_bootstrap_lower_95_strictly_positive": True,
        "shared_t4_absolute_grand_and_seed_means_strictly_positive": True,
    },
}

BOOTSTRAP_POLICY = {
    "replicates": 100_000,
    "rng": "numpy.random.Generator(numpy.random.PCG64(68820260805))",
    "session_resample_count": EXPECTED_SESSION_COUNT,
    "seed_resample_count_within_each_session": len(SEEDS),
    "quantiles": [0.025, 0.975],
    "quantile_method": "linear",
    "cross_comparison_pooling": "FORBIDDEN",
    "cross_view_rescue": "FORBIDDEN",
}


def _checkpoint_path(arm: str, seed: int) -> str:
    if arm in {"shared_t4", "shared_ts4"}:
        return (
            "sua_exploration/checkpoints/"
            f"t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_{arm}_s{seed}/"
            "epoch_ckpts/epoch_011.ckpt"
        )
    return (
        "sua_exploration/checkpoints/"
        f"t4_paired_view_c1_shared_zero4_source_prelaunch_v4_20260805_{arm}_s{seed}/"
        "epoch_ckpts/epoch_011.ckpt"
    )


_KNOWN_TERMINAL_HASHES = {
    ("shared_t4", 42): "ab9df840a07d7aeb6cc417bb684f1f5e0265d50f98168400ac915647cdfd7b9f",
    ("shared_t4", 43): "05c05b3ab82a2fba43c55aca523248982a954faf5f0363a0235a29d64e57ab22",
    ("shared_t4", 44): "a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6",
    ("shared_ts4", 42): "a21da5a72a991bd2665af50572a4132998ac79d7f801879048553efcdc8281b2",
    ("shared_ts4", 43): "c8dd22dfadb2bc11555fc21abe464316886d221e2dbcd71bf20a6bffe9cb158e",
    ("shared_ts4", 44): "a2d877ac81a4e553e5221c54e465db26eba8592888b8cb5339e9dfc4acd66ced",
}


def checkpoint_slots_v4() -> list[dict[str, Any]]:
    """Return nine metadata slots without opening any checkpoint file."""

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            digest = _KNOWN_TERMINAL_HASHES.get((arm, seed))
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "path": _checkpoint_path(arm, seed),
                    "epoch": 11,
                    "sha256": digest,
                    "bytes": None,
                    "status": "PINNED_FROM_EXISTING_SEALED_METADATA" if digest else "MISSING_TERMINAL_PIN",
                }
            )
    return rows


class ThreeArmV4Error(RuntimeError):
    pass


class ThreeArmV4BlockedError(ThreeArmV4Error):
    pass


class ThreeArmV4AuthorizationError(PermissionError):
    pass


class ThreeArmV4LedgerError(ThreeArmV4Error):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ThreeArmV4Error(message)


def auth_require(condition: bool, message: str) -> None:
    if not condition:
        raise ThreeArmV4AuthorizationError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _regular_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ThreeArmV4AuthorizationError(f"{label} missing") from exc
    auth_require(resolved.is_file() and not resolved.is_symlink(), f"{label} unsafe")
    return resolved


def missing_checkpoint_slots(slots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in slots if not _is_sha256(row.get("sha256"))]


def validate_checkpoint_slots(
    slots: Sequence[Mapping[str, Any]], *, require_complete: bool
) -> list[dict[str, Any]]:
    require(isinstance(slots, Sequence) and len(slots) == 9, "checkpoint slot count must be nine")
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed: set[tuple[str, int]] = set()
    normalized: list[dict[str, Any]] = []
    for row in slots:
        require(isinstance(row, Mapping), "checkpoint slot must be an object")
        arm, seed = row.get("arm"), row.get("seed")
        require(arm in ARMS and seed in SEEDS, "checkpoint slot arm/seed drift")
        key = (str(arm), int(seed))
        require(key not in observed, "duplicate checkpoint slot")
        observed.add(key)
        require(row.get("path") == _checkpoint_path(*key), "checkpoint slot path drift")
        require(row.get("epoch") == 11, "checkpoint epoch drift")
        digest = row.get("sha256")
        if key in _KNOWN_TERMINAL_HASHES:
            require(digest == _KNOWN_TERMINAL_HASHES[key], "known checkpoint hash drift")
        if digest is not None:
            require(_is_sha256(digest), "malformed checkpoint SHA-256")
        normalized.append(dict(row))
    require(observed == expected, "checkpoint arm/seed map incomplete")
    missing = missing_checkpoint_slots(normalized)
    if require_complete and missing:
        raise ThreeArmV4BlockedError("BLOCKED_MISSING_ZERO4_TERMINALS")
    return normalized


def validate_frozen_cohort(cohort: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(cohort) == EXPECTED_SESSION_COUNT, "frozen cohort must contain N=15")
    required = {"asset_id", "session_id", "frozen_path", "nwb_sha256", "nwb_bytes"}
    assets: set[str] = set()
    sessions: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in cohort:
        require(isinstance(row, Mapping) and set(row) == required, "frozen cohort row schema drift")
        asset_id, session_id = row["asset_id"], row["session_id"]
        require(isinstance(asset_id, str) and asset_id not in assets, "cohort asset ID drift/duplicate")
        require(isinstance(session_id, str) and session_id not in sessions, "cohort session ID drift/duplicate")
        require(str(row["frozen_path"]).startswith("sub-M/"), "cohort frozen path scope drift")
        require(_is_sha256(row["nwb_sha256"]), "cohort NWB metadata hash malformed")
        require(isinstance(row["nwb_bytes"], int) and row["nwb_bytes"] > 0, "cohort NWB size malformed")
        assets.add(asset_id)
        sessions.add(session_id)
        normalized.append(dict(row))
    return normalized


def build_three_arm_contract(
    *, cohort: Sequence[Mapping[str, Any]], query_counts: Mapping[str, int],
    checkpoint_slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frozen = validate_frozen_cohort(cohort)
    slots = validate_checkpoint_slots(checkpoint_slots, require_complete=False)
    asset_ids = {row["asset_id"] for row in frozen}
    require(set(query_counts) == asset_ids, "query-count/cohort asset map drift")
    require(
        all(isinstance(value, int) and value > 0 for value in query_counts.values()),
        "query count malformed",
    )
    require(sum(query_counts.values()) == EXPECTED_QUERY_WINDOWS_PER_VIEW, "N15 query total drift")
    contract = {
        "schema": "dandi_000688_subm_three_arm_score_contract_v4",
        "scope": "external_subM_CO_held_out_score_only",
        "N": EXPECTED_SESSION_COUNT,
        "cohort": frozen,
        "cohort_sha256": canonical_sha256(frozen),
        "arms": list(ARMS),
        "views": list(VIEWS),
        "seeds": list(SEEDS),
        "cell_count": EXPECTED_CELL_COUNT,
        "query_window_count_by_asset_id": dict(query_counts),
        "query_windows_per_view": EXPECTED_QUERY_WINDOWS_PER_VIEW,
        "total_model_windows": EXPECTED_TOTAL_MODEL_WINDOWS,
        "checkpoint_slots": slots,
        "checkpoint_slots_sha256": canonical_sha256(slots),
        "comparison_gates": COMPARISON_GATES,
        "bootstrap_policy": BOOTSTRAP_POLICY,
        "claim_separation": {
            "t4_minus_zero4": "descriptor-present versus direct neutral coordinate",
            "t4_minus_ts4": "attachment/content permutation mechanism",
            "cross_claim_substitution": "FORBIDDEN",
            "overall_three_arm_claim_requires_both_comparisons_all_views": True,
        },
        "score_protocol": {
            "support_trials": 50,
            "activity_identity_trials": 30,
            "query": "all valid 50-bin windows strictly after rewarded trial 50",
            "target": "last behavior bin of each query window",
            "behavior_scaling_factor_applied_once": 5.0,
            "r2": "variance_weighted",
            "session_drop_after_scoring": "FORBIDDEN",
        },
        "runtime": CPU_POLICY,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_contract_exact(contract: Mapping[str, Any], expected_sha256: str) -> None:
    require(contract.get("contract_sha256") == expected_sha256, "contract declared hash drift")
    body = dict(contract)
    body.pop("contract_sha256", None)
    require(canonical_sha256(body) == expected_sha256, "contract content hash drift")
    require(contract.get("N") == 15 and contract.get("cell_count") == 270, "contract size drift")
    require(contract.get("arms") == list(ARMS), "contract arm order drift")
    require(contract.get("views") == list(VIEWS), "contract view order drift")
    require(contract.get("seeds") == list(SEEDS), "contract seed order drift")
    validate_frozen_cohort(contract.get("cohort", []))
    validate_checkpoint_slots(contract.get("checkpoint_slots", []), require_complete=False)
    require(contract.get("comparison_gates") == COMPARISON_GATES, "comparison gates drift")


@dataclass(frozen=True, order=True)
class CellKey:
    session_id: str
    asset_id: str
    view: str
    seed: int
    arm: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stable_id(self) -> str:
        return f"{self.session_id}__{self.asset_id}__{self.view}__s{self.seed}__{self.arm}"


def expected_cell_keys(contract: Mapping[str, Any]) -> tuple[CellKey, ...]:
    validate_contract_exact(contract, str(contract.get("contract_sha256")))
    keys = tuple(
        CellKey(
            session_id=str(row["session_id"]), asset_id=str(row["asset_id"]),
            view=view, seed=seed, arm=arm,
        )
        for row in contract["cohort"]
        for view in VIEWS
        for seed in SEEDS
        for arm in ARMS
    )
    require(len(keys) == EXPECTED_CELL_COUNT and len(set(keys)) == EXPECTED_CELL_COUNT, "cell map drift")
    return keys


def validate_cell_key(key: CellKey, contract: Mapping[str, Any]) -> None:
    require(key in set(expected_cell_keys(contract)), "wrong arm/seed/view/cohort cell key")


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


@dataclass
class PreAuthorizationAudit:
    authorization_files_read: int = 0
    signature_files_read: int = 0
    public_key_files_read: int = 0
    signature_checks: int = 0
    nonce_claim_writes: int = 0
    nwb_files_opened: int = 0
    checkpoint_files_opened: int = 0
    normalizer_files_opened: int = 0
    torch_imports: int = 0
    model_forward_calls: int = 0
    r2_computations: int = 0

    def data_runtime_zero(self) -> bool:
        return all(
            value == 0
            for value in (
                self.nwb_files_opened,
                self.checkpoint_files_opened,
                self.normalizer_files_opened,
                self.torch_imports,
                self.model_forward_calls,
                self.r2_computations,
            )
        )


@dataclass(frozen=True)
class AuthorizationGrant:
    authorization_sha256: str
    nonce: str
    output_root: Path
    external_nwb_root: Path
    contract_sha256: str
    preimport_authorization_complete: bool


def _parse_time(value: Any, label: str) -> datetime:
    auth_require(isinstance(value, str), f"{label} must be ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ThreeArmV4AuthorizationError(f"{label} malformed") from exc
    auth_require(parsed.tzinfo is not None, f"{label} lacks timezone")
    return parsed.astimezone(timezone.utc)


def _path_within(path: Path, parent: Path, label: str) -> Path:
    candidate, root = path.resolve(), parent.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ThreeArmV4AuthorizationError(f"{label} escapes fixed parent") from exc
    return candidate


def _authorization_bindings(policy: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "contract_sha256",
        "cohort_sha256",
        "checkpoint_slots_sha256",
        "parity_bundle_sha256",
        "source_snapshot_sha256",
        "runtime_identity",
        "cpu_policy",
    )
    return {key: policy[key] for key in keys}


def validate_authorization_payload(
    authorization: Mapping[str, Any], *, policy: Mapping[str, Any],
    output_root: Path, external_nwb_root: Path, now: datetime,
) -> str:
    expected_keys = {
        "schema", "status", "permitted_action", "issued_at", "expires_at", "nonce",
        "output_root", "external_nwb_root", "cell_count", "bindings",
    }
    auth_require(set(authorization) == expected_keys, "authorization key set drift")
    auth_require(authorization.get("schema") == AUTHORIZATION_SCHEMA, "authorization schema drift")
    auth_require(authorization.get("status") == AUTHORIZATION_STATUS, "authorization status drift")
    auth_require(authorization.get("permitted_action") == PERMITTED_ACTION, "authorization action drift")
    issued = _parse_time(authorization.get("issued_at"), "issued_at")
    expires = _parse_time(authorization.get("expires_at"), "expires_at")
    auth_require(issued <= now <= expires, "authorization expired or not yet valid")
    auth_require(
        0 < (expires - issued).total_seconds() <= MAX_AUTHORIZATION_SECONDS,
        "authorization validity exceeds 15 minutes",
    )
    nonce = authorization.get("nonce")
    auth_require(_is_sha256(nonce), "nonce must be fresh 256-bit lowercase hex")
    expected_output = _path_within(output_root, Path(str(policy["output_parent"])), "output root")
    expected_external = _path_within(
        external_nwb_root, Path(str(policy["external_root_parent"])), "external NWB root"
    )
    auth_require(str(expected_output) == authorization.get("output_root"), "output-root binding drift")
    auth_require(
        str(expected_external) == authorization.get("external_nwb_root"),
        "external-root binding drift",
    )
    auth_require(authorization.get("cell_count") == EXPECTED_CELL_COUNT, "authorization cell count drift")
    auth_require(authorization.get("bindings") == _authorization_bindings(policy), "authorization bindings drift")
    auth_require(policy.get("runtime_identity") == runtime_identity(), "runtime identity drift")
    auth_require(policy.get("cpu_policy") == CPU_POLICY, "CPU policy drift")
    auth_require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CUDA_VISIBLE_DEVICES must be empty")
    validate_checkpoint_slots(policy.get("checkpoint_slots", []), require_complete=True)
    return str(nonce)


def _load_json_bytes(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    checked = _regular_file(path, label)
    raw = checked.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThreeArmV4AuthorizationError(f"{label} malformed") from exc
    auth_require(isinstance(value, dict), f"{label} must be an object")
    return value, raw


def _verify_signature_envelope(
    *, authorization_raw: bytes, envelope: Mapping[str, Any], policy: Mapping[str, Any],
    audit: PreAuthorizationAudit,
    test_verifier: Callable[[bytes, bytes], bool] | None = None,
) -> None:
    expected_keys = {"schema", "algorithm", "authorization_sha256", "signature_b64"}
    auth_require(set(envelope) == expected_keys, "signature envelope key set drift")
    auth_require(envelope.get("schema") == AUTHORIZATION_ENVELOPE_SCHEMA, "signature schema drift")
    auth_require(envelope.get("algorithm") == "Ed25519", "signature algorithm drift")
    auth_require(
        envelope.get("authorization_sha256") == hashlib.sha256(authorization_raw).hexdigest(),
        "signature authorization hash drift",
    )
    try:
        signature = base64.b64decode(str(envelope.get("signature_b64")), validate=True)
    except (ValueError, TypeError) as exc:
        raise ThreeArmV4AuthorizationError("signature base64 malformed") from exc
    auth_require(len(signature) == 64, "Ed25519 signature length drift")
    audit.signature_checks += 1
    if test_verifier is not None:
        auth_require(test_verifier(authorization_raw, signature) is True, "invalid detached signature")
        return

    pin = policy.get("public_key")
    auth_require(isinstance(pin, Mapping), "public key pin absent")
    path = _regular_file(Path(str(pin.get("path"))), "dedicated V4 public key")
    audit.public_key_files_read += 1
    auth_require(path.stat().st_size == pin.get("bytes"), "public key size drift")
    auth_require(sha256_file(path) == pin.get("sha256"), "public key hash drift")
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        key = serialization.load_pem_public_key(path.read_bytes())
    except ValueError as exc:
        raise ThreeArmV4AuthorizationError("public key PEM malformed") from exc
    auth_require(isinstance(key, Ed25519PublicKey), "public key is not Ed25519")
    try:
        key.verify(signature, authorization_raw)
    except InvalidSignature as exc:
        raise ThreeArmV4AuthorizationError("invalid detached signature") from exc


def claim_nonce_once(nonce: str, *, claim_root: Path, audit: PreAuthorizationAudit) -> Path:
    auth_require(_is_sha256(nonce), "claim nonce malformed")
    claim_root = claim_root.resolve()
    claim_root.mkdir(parents=True, exist_ok=True)
    claim = claim_root / f"{nonce}.claimed"
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise ThreeArmV4AuthorizationError("authorization nonce already claimed") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write((nonce + "\n").encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
    audit.nonce_claim_writes += 1
    return claim


def preauthorize_and_claim_with_policy(
    *, authorization_path: Path, signature_path: Path, output_root: Path,
    external_nwb_root: Path, policy: Mapping[str, Any], now: datetime,
    audit: PreAuthorizationAudit,
    test_verifier: Callable[[bytes, bytes], bool] | None = None,
) -> AuthorizationGrant:
    """Authorize and claim before any Torch/checkpoint/NWB/runtime import."""

    auth_require(audit.data_runtime_zero(), "data/runtime access occurred before authorization")
    validate_checkpoint_slots(policy.get("checkpoint_slots", []), require_complete=True)
    authorization, authorization_raw = _load_json_bytes(authorization_path, "authorization")
    audit.authorization_files_read += 1
    envelope, _envelope_raw = _load_json_bytes(signature_path, "signature envelope")
    audit.signature_files_read += 1
    _verify_signature_envelope(
        authorization_raw=authorization_raw,
        envelope=envelope,
        policy=policy,
        audit=audit,
        test_verifier=test_verifier,
    )
    nonce = validate_authorization_payload(
        authorization,
        policy=policy,
        output_root=output_root,
        external_nwb_root=external_nwb_root,
        now=now,
    )
    claim_nonce_once(nonce, claim_root=Path(str(policy["claim_root"])), audit=audit)
    auth_require(audit.data_runtime_zero(), "data/runtime access occurred during authorization")
    return AuthorizationGrant(
        authorization_sha256=hashlib.sha256(authorization_raw).hexdigest(),
        nonce=nonce,
        output_root=output_root.resolve(),
        external_nwb_root=external_nwb_root.resolve(),
        contract_sha256=str(policy["contract_sha256"]),
        preimport_authorization_complete=True,
    )


def attach_arm_descriptor_after_authorization(
    record: Mapping[str, Any], *, arm: str, grant: AuthorizationGrant,
    t4_loader: Callable[..., Any] | None = None, t4_loader_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Future post-authorization descriptor branch; no call is made by V4 prelaunch.

    The zero4 branch is deliberately separate from the T4/TS4 feature loader.
    It imports the parity-proven direct-zero helper only after a real grant and
    accepts no normalizer, direction, rate, or permutation argument.
    """

    auth_require(grant.preimport_authorization_complete is True, "descriptor import before authorization")
    auth_require(arm in ARMS, "descriptor arm drift")
    if arm == "shared_zero4":
        auth_require(t4_loader is None and not t4_loader_kwargs, "zero4 cannot receive T4 loader inputs")
        exploration_root = str(REPO_ROOT / "sua_exploration")
        if exploration_root not in sys.path:
            sys.path.insert(0, exploration_root)
        from mc_maze.paired_view_c1_shared_zero4 import (
            attach_standardized_zero4_to_evaluation_record,
        )

        return attach_standardized_zero4_to_evaluation_record(record)
    auth_require(t4_loader is not None, "T4/TS4 arm requires parity-pinned loader")
    kwargs = dict(t4_loader_kwargs or {})
    expected_group = "t4" if arm == "shared_t4" else "ts4"
    auth_require(kwargs.get("feature_group") == expected_group, "T4/TS4 feature-group drift")
    updated = dict(record)
    features, metadata = t4_loader(**kwargs)
    updated["side_features"] = features
    updated["side_feature_metadata"] = metadata
    return updated


def validate_direct_zero4_buffer(
    *, channel_count: int, dtype: str, shape: Sequence[int], raw_bytes: bytes
) -> None:
    """Pure-stdlib negative fence for a serialized direct-zero adapter result.

    Runtime code must additionally call the parity-proven NumPy/Torch helper
    after authorization.  This metadata fence is available before any such
    import and rejects nonzero bytes, negative zero, wrong dtype, and wrong
    shape.  IEEE-754 positive float32 zero is four zero bytes, so checking every
    byte is stronger than a numeric equality test.
    """

    auth_require(
        isinstance(channel_count, int) and not isinstance(channel_count, bool) and channel_count >= 0,
        "zero4 channel count malformed",
    )
    auth_require(dtype == "float32", "zero4 dtype drift")
    auth_require(list(shape) == [channel_count, 4], "zero4 shape drift")
    auth_require(isinstance(raw_bytes, bytes), "zero4 raw buffer must be bytes")
    auth_require(len(raw_bytes) == channel_count * 4 * 4, "zero4 raw byte length drift")
    auth_require(not any(raw_bytes), "nonzero or negative-zero zero4 buffer")


def _cell_path(output_root: Path, key: CellKey) -> Path:
    return output_root / "cells" / key.session_id / key.view / f"s{key.seed}" / f"{key.arm}.json"


def _write_immutable_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = canonical_bytes(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ThreeArmV4LedgerError(f"duplicate output artifact: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise ThreeArmV4LedgerError(f"immutable mode failed: {path}")
    return hashlib.sha256(encoded).hexdigest()


def publish_cell_result(
    output_root: Path, *, key: CellKey, contract: Mapping[str, Any], r2: float,
    prediction_target_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    validate_cell_key(key, contract)
    if not isinstance(r2, (int, float)) or isinstance(r2, bool) or not math.isfinite(float(r2)):
        raise ThreeArmV4LedgerError("cell R2 must be finite")
    artifact = dict(prediction_target_artifact)
    required_artifact = {"path", "sha256", "bytes", "mode"}
    if set(artifact) != required_artifact or not _is_sha256(artifact.get("sha256")):
        raise ThreeArmV4LedgerError("cell prediction/target artifact binding malformed")
    if artifact.get("mode") != "0444" or not isinstance(artifact.get("bytes"), int):
        raise ThreeArmV4LedgerError("cell prediction/target artifact mode/size malformed")
    payload = {
        "schema": "dandi_000688_subm_three_arm_cell_result_v4",
        "status": "COMPLETE",
        "cell": key.as_dict(),
        "r2": float(r2),
        "prediction_target_artifact": artifact,
        "contract_sha256": contract["contract_sha256"],
    }
    path = _cell_path(output_root.resolve(), key)
    digest = _write_immutable_json_exclusive(path, payload)
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size, "mode": "0444"}


def scan_resume_state(output_root: Path, *, contract: Mapping[str, Any]) -> dict[str, Any]:
    output_root = output_root.resolve()
    expected = {key: _cell_path(output_root, key) for key in expected_cell_keys(contract)}
    cells_root = output_root / "cells"
    observed_files: set[Path] = set()
    if cells_root.exists():
        for path in cells_root.rglob("*"):
            if path.is_symlink():
                raise ThreeArmV4LedgerError(f"symlink in cell ledger: {path}")
            if path.is_file():
                observed_files.add(path.resolve())
    expected_paths = {path.resolve() for path in expected.values()}
    unexpected = observed_files - expected_paths
    if unexpected:
        raise ThreeArmV4LedgerError("partial/unknown cell artifact present")
    complete: list[CellKey] = []
    for key, path in expected.items():
        if not path.exists():
            continue
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise ThreeArmV4LedgerError("mutable cell artifact")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ThreeArmV4LedgerError("malformed cell artifact") from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "dandi_000688_subm_three_arm_cell_result_v4"
            or payload.get("status") != "COMPLETE"
            or payload.get("cell") != key.as_dict()
            or payload.get("contract_sha256") != contract["contract_sha256"]
        ):
            raise ThreeArmV4LedgerError("cell artifact semantic drift")
        complete.append(key)
    missing = [key for key in expected if key not in set(complete)]
    return {
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "complete_cell_count": len(complete),
        "missing_cell_count": len(missing),
        "complete": [key.as_dict() for key in complete],
        "missing": [key.as_dict() for key in missing],
    }


def publish_full_aggregate(
    output_root: Path, *, contract: Mapping[str, Any], aggregate: Mapping[str, Any]
) -> dict[str, Any]:
    state = scan_resume_state(output_root, contract=contract)
    if state["missing_cell_count"] != 0 or state["complete_cell_count"] != EXPECTED_CELL_COUNT:
        raise ThreeArmV4LedgerError("full aggregate forbidden before all 270 cells complete")
    required = {"t4_minus_zero4", "t4_minus_ts4"}
    if set(aggregate) != required:
        raise ThreeArmV4LedgerError("aggregate comparison map drift")
    payload = {
        "schema": "dandi_000688_subm_three_arm_full_aggregate_v4",
        "status": "COMPLETE_270_CELLS",
        "contract_sha256": contract["contract_sha256"],
        "cell_count": EXPECTED_CELL_COUNT,
        "comparisons": dict(aggregate),
    }
    path = output_root.resolve() / "aggregate" / "full_aggregate.json"
    digest = _write_immutable_json_exclusive(path, payload)
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size, "mode": "0444"}


def refuse_blocked_execution(*, audit: PreAuthorizationAudit) -> None:
    """The current V4 runner calls this before reading any authorization path."""

    if not audit.data_runtime_zero():
        raise ThreeArmV4BlockedError("blocked package observed forbidden runtime access")
    raise ThreeArmV4BlockedError("BLOCKED_MISSING_ZERO4_TERMINALS")
