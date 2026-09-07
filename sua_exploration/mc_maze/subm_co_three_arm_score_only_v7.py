"""Fail-closed V7 control plane for the external sub-M three-arm endpoint.

V7 deliberately has two distinct states:

* the checked-in package is ``BLOCKED_MISSING_ZERO4_TERMINALS`` and performs
  no checkpoint, NWB, Torch, GPU, or external-score operation; and
* a later formal run is possible only after an *independently provisioned*
  :class:`TrustedRoots` object verifies a complete policy, nine independently
  Ed25519-signed terminal closures, the nine real terminal checkpoint files,
  and one short-lived run authorization.

The runtime policy never supplies a public key or a filesystem root.  Those
are properties of the pre-installed trusted root object.  In particular, a
signed policy is verified before a contract is constructed, and a contract
cannot be built from an unsigned mapping.

This module intentionally does not import Torch.  Its metric routine is a
small, tested numerical reproduction of the frozen TorchMetrics 1.5.1
``R2Score(multioutput='variance_weighted')`` computation on CPU float32
arrays, including its 1e-4 near-constant branch.  Tests compare that routine
against the frozen implementation when the SPINT environment is available.
"""
from __future__ import annotations

import ast
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, Mapping, Sequence
import zipfile

from sua_exploration.mc_maze import subm_co_three_arm_score_only_v6 as v6


ARMS = v6.ARMS
VIEWS = v6.VIEWS
SEEDS = v6.SEEDS
COMPARISONS = v6.COMPARISONS
N = v6.N
CELL_COUNT = v6.CELL_COUNT
QUERY_WINDOWS_PER_VIEW = v6.QUERY_WINDOWS_PER_VIEW
OUTPUT_DIM = v6.OUTPUT_DIM
BOOTSTRAP_REPLICATES = v6.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = v6.BOOTSTRAP_SEED
MAX_AUTH_SECONDS = 900
TORCHMETRICS_NEAR_CONSTANT_ATOL = 1.0e-4
NPY_HEADER_MAX_BYTES = 64 * 1024
NPZ_ABSOLUTE_MAX_BYTES = 128 * 1024 * 1024

# The only production trust-anchor entrypoint is this source-pinned immutable
# blocker.  It contains no active roots today, so V7 cannot accidentally turn
# into a formal runner merely because somebody supplies self-signed roots at
# runtime.  Activating a future root set requires a reviewed source/anchor
# revision rather than a command-line payload.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PINNED_TRUST_ANCHOR_PATH = _WORKSPACE_ROOT / "sua_exploration/configs/dandi_000688_subm_v7_pinned_trust_anchor.json"
PINNED_TRUST_ANCHOR_SHA256 = "29b4b8c677c2219d0f7e69df4fedac11532f39151b7e65208244d291c7d76196"
PINNED_TRUST_ANCHOR_BYTES = 200
PINNED_TRUST_ANCHOR_MODE = "0444"

TRUSTED_ROOTS_SCHEMA = "dandi_000688_subm_v7_trusted_roots"
POLICY_SCHEMA = "dandi_000688_subm_three_arm_complete_policy_v7"
POLICY_STATUS = "COMPLETE_POLICY_SIGNED_BEFORE_CONTRACT_V7"
POLICY_ENVELOPE_SCHEMA = "dandi_000688_subm_policy_ed25519_envelope_v7"
CLOSURE_SCHEMA = "dandi_000688_subm_terminal_checkpoint_closure_v7"
CLOSURE_STATUS = "INDEPENDENTLY_VERIFIED_SEALED_TERMINAL_V7"
CLOSURE_ENVELOPE_SCHEMA = "dandi_000688_subm_checkpoint_closure_ed25519_envelope_v7"
AUTH_SCHEMA = "dandi_000688_subm_three_arm_run_authorization_v7"
AUTH_STATUS = "AUTHORIZED_FOR_ONE_EXTERNAL_SUBM_SCORE_V7"
AUTH_ACTION = "score_publish_resume_aggregate_exact_270_v7"
AUTH_ENVELOPE_SCHEMA = "dandi_000688_subm_run_authorization_ed25519_envelope_v7"
CONTRACT_SCHEMA = "dandi_000688_subm_three_arm_score_contract_v7"
GRANT_SCHEMA = "dandi_000688_subm_three_arm_verified_grant_v7"
CELL_SCHEMA = "dandi_000688_subm_three_arm_cell_result_v7"
AGGREGATE_SCHEMA = "dandi_000688_subm_three_arm_aggregate_v7"

CPU_POLICY = {
    "device": "cpu", "cuda_visible_devices": "", "torch_num_threads": 1,
    "torch_num_interop_threads": 1, "deterministic_algorithms": True,
    "tf32": False, "autocast": False, "normalizer_fitting": False,
    "optimizer_or_backward": False, "target_updates": False,
}
SCORE_PROTOCOL = {
    "support_trials": 50,
    "activity_identity_trials": 30,
    "query": "all valid 50-bin windows strictly after rewarded trial 50",
    "target": "last behavior bin of each query window",
    "output_dim": OUTPUT_DIM,
    "behavior_scaling_factor_applied_once": 5.0,
    "r2": "frozen_torchmetrics_1_5_1_variance_weighted_cpu_float32",
    "near_constant_atol": TORCHMETRICS_NEAR_CONSTANT_ATOL,
    "caller_supplied_r2": "FORBIDDEN",
    "session_drop_after_scoring": "FORBIDDEN",
}
CLAIM_SEPARATION = dict(v6.CLAIM_SEPARATION)
COMPARISON_GATES = dict(v6.COMPARISON_GATES)
BOOTSTRAP_POLICY = dict(v6.BOOTSTRAP_POLICY)

FILE_PIN_KEYS = {"path", "sha256", "bytes", "mode"}
TRUSTED_ROOT_KEYS = {
    "schema", "root_id", "policy_root", "checkpoint_root", "run_auth_root",
    "output_parent", "external_nwb_root", "claim_root", "keys",
}
TRUSTED_KEY_KEYS = {"policy", "checkpoint", "run_auth"}
COHORT_KEYS = {"asset_id", "session_id", "frozen_path", "nwb_sha256", "nwb_bytes"}
POLICY_KEYS = {
    "schema", "status", "trusted_root_id", "frozen_predecessor_sha256",
    "cohort", "query_counts", "reviewed_checkpoint_slots",
    "reviewed_checkpoint_slots_sha256", "source_snapshot_sha256",
    "parity_bundle_sha256", "runtime_identity", "cpu_policy",
}
SLOT_KEYS = {"arm", "seed", "epoch", "path", "sha256", "bytes", "mode", "status", "closure"}
CLOSURE_REF_KEYS = {"kind", "payload", "signature"}
DETACHED_ENVELOPE_KEYS = {"schema", "algorithm", "payload_sha256", "signature_b64"}
AUTH_KEYS = {
    "schema", "status", "permitted_action", "trusted_root_id", "policy_sha256",
    "contract_sha256", "issued_at", "expires_at", "nonce", "run_id", "cell_count",
}
CELL_ID_KEYS = {"session_id", "asset_id", "view", "seed", "arm"}
ARTIFACT_PIN_KEYS = {"path", "sha256", "bytes", "mode", "verified_grant_sha256", "arrays"}
CELL_KEYS = {
    "schema", "status", "verified_grant_sha256", "cell", "r2",
    "query_window_count", "prediction_target_artifact", "contract_sha256",
}
AGGREGATE_KEYS = {
    "schema", "status", "verified_grant_sha256", "contract_sha256",
    "statistics_source", "verified_cell_count", "bootstrap_policy",
    "comparisons", "overall_three_arm_claim_pass",
}

SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
SAFE_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}")


class V7Error(RuntimeError):
    pass


class V7BlockedError(V7Error):
    pass


class V7AuthorizationError(PermissionError):
    pass


class V7LedgerError(V7Error):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise V7Error(message)


def auth_require(condition: bool, message: str) -> None:
    if not condition:
        raise V7AuthorizationError(message)


def ledger_require(condition: bool, message: str) -> None:
    if not condition:
        raise V7LedgerError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_relative_file(path: Any) -> bool:
    if not isinstance(path, str) or not path or "\\" in path:
        return False
    candidate = Path(path)
    return not candidate.is_absolute() and all(part not in {"", ".", ".."} for part in candidate.parts)


def _relative_parts(path: str) -> tuple[str, ...]:
    ledger_require(_is_relative_file(path), "relative path is unsafe")
    return tuple(Path(path).parts)


def _open_absolute_directory(path: Path) -> int:
    """Open every absolute component using O_NOFOLLOW (dirfd root anchor)."""
    absolute = _absolute(path)
    ledger_require(absolute.is_absolute(), "trusted root is not absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor); descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        ledger_require(stat.S_ISDIR(metadata.st_mode), "trusted root is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_parent_at(root_fd: int, relative: str, *, create: bool) -> tuple[int, str]:
    parts = _relative_parts(relative)
    current = os.dup(root_fd)
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=current)
                except FileExistsError:
                    pass
            next_descriptor = os.open(part, flags, dir_fd=current)
            os.close(current); current = next_descriptor
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


@dataclass(frozen=True)
class FDRead:
    raw: bytes
    sha256: str
    bytes: int
    mode: str


def fd_read_regular_at(
    root: Path, relative: str, *, expected_mode: str | None = None,
    expected_sha256: str | None = None, expected_bytes: int | None = None,
    max_bytes: int = 16 * 1024 * 1024,
) -> FDRead:
    """Read one regular file through a fixed directory fd, without symlinks."""
    root_fd = _open_absolute_directory(root)
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd, leaf = _open_parent_at(root_fd, relative, create=False)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(leaf, flags, dir_fd=parent_fd)
        before = os.fstat(descriptor)
        ledger_require(stat.S_ISREG(before.st_mode), "opened path is not a regular file")
        ledger_require(before.st_size <= max_bytes, "file exceeds fixed safety limit")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, min(1024 * 1024, max_bytes + 1))
            if not block:
                break
            chunks.append(block)
            ledger_require(sum(len(item) for item in chunks) <= max_bytes, "file exceeds fixed safety limit")
        after = os.fstat(descriptor)
        ledger_require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "file changed during fd read",
        )
        check = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        ledger_require(not stat.S_ISLNK(check.st_mode) and (check.st_dev, check.st_ino) == (after.st_dev, after.st_ino), "path identity changed after fd read")
        raw = b"".join(chunks)
        mode = f"0{stat.S_IMODE(after.st_mode):03o}"
        digest = hashlib.sha256(raw).hexdigest()
        if expected_mode is not None:
            ledger_require(mode == expected_mode, "file mode drift")
        if expected_sha256 is not None:
            ledger_require(digest == expected_sha256, "file SHA-256 drift")
        if expected_bytes is not None:
            ledger_require(len(raw) == expected_bytes, "file byte-size drift")
        return FDRead(raw, digest, len(raw), mode)
    except OSError as exc:
        raise V7LedgerError("safe dirfd read failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)
        os.close(root_fd)


def _read_canonical_json_at(
    root: Path, relative: str, *, pin: Mapping[str, Any] | None = None,
    expected_mode: str = "0444", max_bytes: int = 16 * 1024 * 1024,
) -> tuple[dict[str, Any], FDRead]:
    kwargs: dict[str, Any] = {"expected_mode": expected_mode, "max_bytes": max_bytes}
    if pin is not None:
        ledger_require(set(pin) == FILE_PIN_KEYS and _is_relative_file(pin["path"]), "file pin schema/path drift")
        kwargs.update(expected_mode=pin["mode"], expected_sha256=pin["sha256"], expected_bytes=pin["bytes"])
        relative = str(pin["path"])
    observed = fd_read_regular_at(root, relative, **kwargs)
    try:
        value = json.loads(observed.raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V7LedgerError("malformed canonical JSON") from exc
    ledger_require(isinstance(value, dict) and observed.raw == canonical_bytes(value), "JSON is not a canonical exact object")
    return value, observed


def _write_exclusive_at(root: Path, relative: str, raw: bytes) -> dict[str, Any]:
    """Immutable write using mkdirat/openat; the returned pin is root-relative."""
    root_fd = _open_absolute_directory(root)
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd, leaf = _open_parent_at(root_fd, relative, create=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(leaf, flags, 0o600, dir_fd=parent_fd)
        total = 0
        while total < len(raw):
            total += os.write(descriptor, raw[total:])
        os.fsync(descriptor); os.fchmod(descriptor, 0o444)
        metadata = os.fstat(descriptor)
        ledger_require(stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o444 and metadata.st_size == len(raw), "immutable openat write failed")
        os.fsync(parent_fd)
    except FileExistsError as exc:
        raise V7LedgerError(f"duplicate immutable output: {relative}") from exc
    except OSError as exc:
        raise V7LedgerError("safe dirfd write failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)
        os.close(root_fd)
    observed = fd_read_regular_at(root, relative, expected_mode="0444", expected_sha256=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw), max_bytes=max(len(raw), 1))
    return {"path": relative, "sha256": observed.sha256, "bytes": observed.bytes, "mode": "0444"}


def _read_pinned_key(root: Path, pin: Mapping[str, Any]) -> bytes:
    ledger_require(isinstance(pin, Mapping) and set(pin) == FILE_PIN_KEYS and _is_relative_file(pin.get("path")), "trusted key pin drift")
    return fd_read_regular_at(root, str(pin["path"]), expected_mode=pin["mode"], expected_sha256=pin["sha256"], expected_bytes=pin["bytes"], max_bytes=64 * 1024).raw


_ROOT_MINT = object()


class TrustedRoots:
    """Pre-installed trust anchor; it is deliberately not part of policy JSON."""
    __slots__ = ("_mint", "_payload", "_origin", "_anchor_sha256", "_sealed", "root_digest")

    def __init__(self, *, _mint: object | None = None, payload: Mapping[str, Any] | None = None, origin: str | None = None, anchor_sha256: str | None = None):
        if _mint is not _ROOT_MINT or payload is None:
            raise TypeError("TrustedRoots can only be installed out-of-band")
        object.__setattr__(self, "_mint", _mint)
        object.__setattr__(self, "_payload", dict(payload))
        object.__setattr__(self, "_origin", origin)
        object.__setattr__(self, "_anchor_sha256", anchor_sha256)
        object.__setattr__(self, "root_digest", canonical_sha256(payload))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("TrustedRoots is immutable")
        object.__setattr__(self, name, value)

    @property
    def payload(self) -> dict[str, Any]:
        return dict(self._payload)

    @property
    def root_id(self) -> str:
        return str(self._payload["root_id"])

    def path(self, name: str) -> Path:
        return Path(str(self._payload[name]))

    def key_pin(self, name: str) -> dict[str, Any]:
        return dict(self._payload["keys"][name])


def _validate_roots_payload(payload: Mapping[str, Any]) -> None:
    require(isinstance(payload, Mapping) and set(payload) == TRUSTED_ROOT_KEYS and payload.get("schema") == TRUSTED_ROOTS_SCHEMA, "trusted roots exact schema drift")
    require(isinstance(payload.get("root_id"), str) and SAFE_IDENTIFIER.fullmatch(str(payload["root_id"])) is not None, "trusted root id malformed")
    for name in ("policy_root", "checkpoint_root", "run_auth_root", "output_parent", "external_nwb_root", "claim_root"):
        value = payload.get(name)
        require(isinstance(value, str) and Path(value).is_absolute(), f"trusted root {name} must be absolute")
    keys = payload.get("keys")
    require(isinstance(keys, Mapping) and set(keys) == TRUSTED_KEY_KEYS, "trusted key map drift")
    key_roots = {"policy": "policy_root", "checkpoint": "checkpoint_root", "run_auth": "run_auth_root"}
    for name, root_name in key_roots.items():
        pin = keys[name]
        require(isinstance(pin, Mapping) and set(pin) == FILE_PIN_KEYS and _is_relative_file(pin.get("path")) and is_sha256(pin.get("sha256")) and positive_int(pin.get("bytes")) and pin.get("mode") == "0444", f"trusted {name} key pin drift")
        # Verify the anchor's own public key pin at installation, before any
        # policy is touched.  This is the fixed root chain's first live link.
        _read_pinned_key(Path(str(payload[root_name])), pin)


def _install_synthetic_trusted_roots_for_test(payload: Mapping[str, Any]) -> TrustedRoots:
    """Test-only factory.  Formal verification rejects its resulting object.

    It is intentionally private and only supports isolated synthetic unit
    tests of key pinning and closure verification.  Production callers must
    use :func:`install_trusted_roots`, which takes no user-controlled payload.
    """
    _validate_roots_payload(payload)
    return TrustedRoots(_mint=_ROOT_MINT, payload=payload, origin="synthetic_test_only", anchor_sha256=None)


def install_trusted_roots() -> TrustedRoots:
    """Load the sole source-pinned production trust-anchor state.

    The current pinned anchor is deliberately a blocked marker, so this raises
    ``BLOCKED_MISSING_ZERO4_TERMINALS`` rather than accepting caller roots.
    """
    observed = fd_read_regular_at(
        PINNED_TRUST_ANCHOR_PATH.parent, PINNED_TRUST_ANCHOR_PATH.name,
        expected_mode=PINNED_TRUST_ANCHOR_MODE,
        expected_sha256=PINNED_TRUST_ANCHOR_SHA256,
        expected_bytes=PINNED_TRUST_ANCHOR_BYTES,
        max_bytes=64 * 1024,
    )
    try:
        anchor = json.loads(observed.raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V7AuthorizationError("pinned trust anchor malformed") from exc
    auth_require(
        isinstance(anchor, dict) and observed.raw == canonical_bytes(anchor)
        and anchor == {
            "anchor_id": "subm-v7-no-active-formal-roots",
            "kind": "dandi_000688_subm_v7_pinned_trust_anchor",
            "schema": "dandi_000688_subm_v7_pinned_trust_anchor",
            "status": "BLOCKED_NO_ACTIVE_FORMAL_TRUST_ROOTS_V7",
        },
        "pinned trust anchor semantic drift",
    )
    raise V7BlockedError("BLOCKED_MISSING_ZERO4_TERMINALS")


def _assert_roots(roots: TrustedRoots) -> None:
    auth_require(
        type(roots) is TrustedRoots and getattr(roots, "_mint", None) is _ROOT_MINT
        and roots.root_digest == canonical_sha256(roots._payload)
        and roots._origin == "pinned_v7_anchor"
        and roots._anchor_sha256 == PINNED_TRUST_ANCHOR_SHA256,
        "trusted roots were not loaded from the V7 pinned trust anchor",
    )
    # Re-read the three pinned authority keys on every privileged transition.
    _read_pinned_key(roots.path("policy_root"), roots.key_pin("policy"))
    _read_pinned_key(roots.path("checkpoint_root"), roots.key_pin("checkpoint"))
    _read_pinned_key(roots.path("run_auth_root"), roots.key_pin("run_auth"))


def checkpoint_path(arm: str, seed: int) -> str:
    require(arm in ARMS and seed in SEEDS, "checkpoint slot identifier drift")
    return v6.checkpoint_path(arm, seed)


def closure_relative_path(arm: str, seed: int) -> str:
    require(arm in ARMS and seed in SEEDS, "closure slot identifier drift")
    return f"sua_exploration/results/dandi_000688_subm_v7_checkpoint_closures/{arm}_s{seed}/closure.json"


def closure_signature_relative_path(arm: str, seed: int) -> str:
    return closure_relative_path(arm, seed).replace("closure.json", "closure.ed25519.json")


def slot_binding(row: Mapping[str, Any]) -> str:
    return canonical_sha256({key: row[key] for key in ("arm", "seed", "epoch", "path", "sha256", "bytes", "mode")})


def _validate_cohort(cohort: Any) -> list[dict[str, Any]]:
    require(isinstance(cohort, list) and len(cohort) == N, "frozen cohort must contain exactly 15 sessions")
    assets: set[str] = set(); sessions: set[str] = set(); result = []
    for row in cohort:
        require(isinstance(row, Mapping) and set(row) == COHORT_KEYS, "cohort exact schema drift")
        asset, session = row.get("asset_id"), row.get("session_id")
        require(isinstance(asset, str) and SAFE_IDENTIFIER.fullmatch(asset) is not None and asset not in assets, "cohort asset id malformed/duplicate")
        require(isinstance(session, str) and SAFE_IDENTIFIER.fullmatch(session) is not None and session.startswith("sub-M_ses-") and session not in sessions, "cohort session id malformed/duplicate")
        require(isinstance(row.get("frozen_path"), str) and row["frozen_path"].startswith("sub-M/") and _is_relative_file(row["frozen_path"]), "cohort frozen path drift")
        require(is_sha256(row.get("nwb_sha256")) and positive_int(row.get("nwb_bytes")), "cohort data pin malformed")
        assets.add(asset); sessions.add(session); result.append(dict(row))
    return result


def _validate_complete_slots(slots: Any) -> list[dict[str, Any]]:
    require(isinstance(slots, list) and len(slots) == 9, "complete policy requires nine checkpoint slots")
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}; seen = set(); result = []
    for row in slots:
        require(isinstance(row, Mapping) and set(row) == SLOT_KEYS, "checkpoint slot exact schema drift")
        key = (row.get("arm"), row.get("seed"))
        require(key in expected and key not in seen, "checkpoint slot duplicate/drift")
        seen.add(key)
        require(row.get("epoch") == 11 and row.get("path") == checkpoint_path(*key), "checkpoint terminal path/epoch drift")
        require(is_sha256(row.get("sha256")) and positive_int(row.get("bytes")) and row.get("mode") == "0444", "checkpoint pin malformed")
        require(row.get("status") == "INDEPENDENT_V7_TERMINAL_CLOSURE_SIGNED", "checkpoint lacks independent V7 terminal closure")
        closure = row.get("closure")
        require(isinstance(closure, Mapping) and set(closure) == CLOSURE_REF_KEYS and closure.get("kind") == "independent_ed25519_checkpoint_closure_v7", "V7 closure reference schema drift")
        for label, expected_path in (("payload", closure_relative_path(*key)), ("signature", closure_signature_relative_path(*key))):
            pin = closure.get(label)
            require(isinstance(pin, Mapping) and set(pin) == FILE_PIN_KEYS and pin.get("path") == expected_path and is_sha256(pin.get("sha256")) and positive_int(pin.get("bytes")) and pin.get("mode") == "0444", f"V7 closure {label} pin drift")
        result.append(dict(row))
    require(seen == expected, "complete checkpoint slot map incomplete")
    return result


def _verify_detached_signature(*, raw: bytes, envelope_raw: bytes, public_pem: bytes, schema: str) -> None:
    try:
        envelope = json.loads(envelope_raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V7AuthorizationError("detached signature envelope malformed") from exc
    auth_require(isinstance(envelope, dict) and envelope_raw == canonical_bytes(envelope) and set(envelope) == DETACHED_ENVELOPE_KEYS, "detached signature envelope schema drift")
    auth_require(envelope["schema"] == schema and envelope["algorithm"] == "Ed25519" and envelope["payload_sha256"] == hashlib.sha256(raw).hexdigest(), "detached signature envelope binding drift")
    try:
        signature = base64.b64decode(envelope["signature_b64"], validate=True)
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        key = serialization.load_pem_public_key(public_pem)
        auth_require(isinstance(key, Ed25519PublicKey) and len(signature) == 64, "Ed25519 key/signature type drift")
        key.verify(signature, raw)
    except (ValueError, TypeError, InvalidSignature) as exc:
        raise V7AuthorizationError("invalid Ed25519 detached signature") from exc


def _verify_complete_closure(row: Mapping[str, Any], roots: TrustedRoots) -> dict[str, Any]:
    """Verify signature *and then* live SHA/bytes/mode/path of checkpoint."""
    key = (str(row["arm"]), int(row["seed"]))
    closure = row["closure"]
    payload_pin = closure["payload"]; signature_pin = closure["signature"]
    payload_read = fd_read_regular_at(roots.path("checkpoint_root"), str(payload_pin["path"]), expected_mode="0444", expected_sha256=str(payload_pin["sha256"]), expected_bytes=int(payload_pin["bytes"]), max_bytes=1024 * 1024)
    signature_read = fd_read_regular_at(roots.path("checkpoint_root"), str(signature_pin["path"]), expected_mode="0444", expected_sha256=str(signature_pin["sha256"]), expected_bytes=int(signature_pin["bytes"]), max_bytes=1024 * 1024)
    _verify_detached_signature(raw=payload_read.raw, envelope_raw=signature_read.raw, public_pem=_read_pinned_key(roots.path("checkpoint_root"), roots.key_pin("checkpoint")), schema=CLOSURE_ENVELOPE_SCHEMA)
    try:
        payload = json.loads(payload_read.raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V7AuthorizationError("checkpoint closure payload malformed") from exc
    auth_require(isinstance(payload, dict) and payload_read.raw == canonical_bytes(payload), "checkpoint closure is not canonical")
    expected_keys = {"schema", "status", "arm", "seed", "epoch", "checkpoint", "slot_binding_sha256"}
    auth_require(set(payload) == expected_keys and payload["schema"] == CLOSURE_SCHEMA and payload["status"] == CLOSURE_STATUS, "checkpoint closure schema/status drift")
    auth_require((payload["arm"], payload["seed"], payload["epoch"]) == (row["arm"], row["seed"], row["epoch"]), "checkpoint closure slot identity drift")
    checkpoint = {name: row[name] for name in ("path", "sha256", "bytes", "mode")}
    auth_require(payload["checkpoint"] == checkpoint and payload["slot_binding_sha256"] == slot_binding(row), "checkpoint closure pin/binding drift")
    # This is intentionally after signature verification and before any formal
    # scoring authority can be minted.
    fd_read_regular_at(roots.path("checkpoint_root"), str(row["path"]), expected_mode="0444", expected_sha256=str(row["sha256"]), expected_bytes=int(row["bytes"]), max_bytes=max(int(row["bytes"]), 1))
    return {"arm": key[0], "seed": key[1], "slot_binding_sha256": slot_binding(row), "closure_payload": dict(payload_pin), "closure_signature": dict(signature_pin)}


def _verify_all_nine_closures(slots: Sequence[Mapping[str, Any]], roots: TrustedRoots) -> tuple[str, list[dict[str, Any]]]:
    verified = [_verify_complete_closure(row, roots) for row in _validate_complete_slots(list(slots))]
    return canonical_sha256(verified), verified


def _policy_relative_path() -> str:
    return "complete_policy_v7.json"


def _policy_signature_relative_path() -> str:
    return "complete_policy_v7.ed25519.json"


_POLICY_MINT = object()


class VerifiedPolicy:
    __slots__ = ("_mint", "_roots", "_policy", "_policy_pin", "_signature_pin", "_closure_bundle_sha256", "_sealed", "policy_sha256")

    def __init__(self, *, _mint: object | None = None, roots: TrustedRoots | None = None, policy: Mapping[str, Any] | None = None, policy_pin: Mapping[str, Any] | None = None, signature_pin: Mapping[str, Any] | None = None, closure_bundle_sha256: str | None = None):
        if _mint is not _POLICY_MINT or roots is None or policy is None or policy_pin is None or signature_pin is None or closure_bundle_sha256 is None:
            raise TypeError("VerifiedPolicy can only be minted after policy/closure verification")
        object.__setattr__(self, "_mint", _mint); object.__setattr__(self, "_roots", roots)
        object.__setattr__(self, "_policy", dict(policy)); object.__setattr__(self, "_policy_pin", dict(policy_pin)); object.__setattr__(self, "_signature_pin", dict(signature_pin))
        object.__setattr__(self, "_closure_bundle_sha256", closure_bundle_sha256)
        object.__setattr__(self, "policy_sha256", str(policy_pin["sha256"])); object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("VerifiedPolicy is immutable")
        object.__setattr__(self, name, value)

    @property
    def roots(self) -> TrustedRoots:
        return self._roots

    @property
    def policy(self) -> dict[str, Any]:
        return dict(self._policy)

    @property
    def closure_bundle_sha256(self) -> str:
        return self._closure_bundle_sha256


def _validate_complete_policy_shape(policy: Mapping[str, Any], roots: TrustedRoots) -> list[dict[str, Any]]:
    auth_require(isinstance(policy, Mapping) and set(policy) == POLICY_KEYS and policy.get("schema") == POLICY_SCHEMA and policy.get("status") == POLICY_STATUS, "complete policy exact schema/status drift")
    auth_require(policy.get("trusted_root_id") == roots.root_id, "complete policy trusted-root binding drift")
    cohort = _validate_cohort(policy["cohort"])
    query = policy["query_counts"]
    auth_require(isinstance(query, Mapping) and set(query) == {row["asset_id"] for row in cohort} and all(positive_int(value) for value in query.values()) and sum(query.values()) == QUERY_WINDOWS_PER_VIEW, "complete policy query-map drift")
    slots = _validate_complete_slots(policy["reviewed_checkpoint_slots"])
    auth_require(canonical_sha256(slots) == policy["reviewed_checkpoint_slots_sha256"], "complete policy slot digest drift")
    auth_require(is_sha256(policy.get("frozen_predecessor_sha256")) and is_sha256(policy.get("source_snapshot_sha256")) and is_sha256(policy.get("parity_bundle_sha256")), "complete policy static digest drift")
    auth_require(policy.get("runtime_identity") == v6.runtime_identity() and policy.get("cpu_policy") == CPU_POLICY, "complete policy runtime/CPU binding drift")
    # A policy must not smuggle a runtime public key, root path, or grant key:
    # exact-key validation above is intentional and security-relevant.
    return slots


def verify_complete_policy(roots: TrustedRoots) -> VerifiedPolicy:
    """Verify signed policy, then every closure/checkpoint, then mint policy."""
    _assert_roots(roots)
    policy, policy_read = _read_canonical_json_at(roots.path("policy_root"), _policy_relative_path(), expected_mode="0444", max_bytes=16 * 1024 * 1024)
    signature_read = fd_read_regular_at(roots.path("policy_root"), _policy_signature_relative_path(), expected_mode="0444", max_bytes=1024 * 1024)
    _verify_detached_signature(raw=policy_read.raw, envelope_raw=signature_read.raw, public_pem=_read_pinned_key(roots.path("policy_root"), roots.key_pin("policy")), schema=POLICY_ENVELOPE_SCHEMA)
    slots = _validate_complete_policy_shape(policy, roots)
    closure_digest, _ = _verify_all_nine_closures(slots, roots)
    return VerifiedPolicy(
        _mint=_POLICY_MINT, roots=roots, policy=policy,
        policy_pin={"path": _policy_relative_path(), "sha256": policy_read.sha256, "bytes": policy_read.bytes, "mode": policy_read.mode},
        signature_pin={"path": _policy_signature_relative_path(), "sha256": signature_read.sha256, "bytes": signature_read.bytes, "mode": signature_read.mode},
        closure_bundle_sha256=closure_digest,
    )


def _assert_verified_policy(verified: VerifiedPolicy) -> VerifiedPolicy:
    auth_require(type(verified) is VerifiedPolicy and getattr(verified, "_mint", None) is _POLICY_MINT, "real verified policy required")
    # Re-verify the trusted chain.  This makes a forged Python object or an
    # altered terminal checkpoint fail before a ledger operation.
    fresh = verify_complete_policy(verified.roots)
    auth_require(
        fresh.policy_sha256 == verified.policy_sha256
        and fresh.closure_bundle_sha256 == verified.closure_bundle_sha256
        and fresh.policy == verified.policy,
        "verified policy proof drift",
    )
    return fresh


def build_expected_contract(verified: VerifiedPolicy) -> dict[str, Any]:
    """Construct a contract only from a live, previously signed policy."""
    verified = _assert_verified_policy(verified)
    policy = verified.policy
    cohort = [dict(row) for row in policy["cohort"]]
    query = {str(key): int(value) for key, value in policy["query_counts"].items()}
    slots = [dict(row) for row in policy["reviewed_checkpoint_slots"]]
    contract: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA, "scope": "external_subM_CO_held_out_score_only",
        "trusted_root_id": verified.roots.root_id,
        "signed_policy_sha256": verified.policy_sha256,
        "frozen_predecessor_sha256": policy["frozen_predecessor_sha256"],
        "N": N, "cohort": cohort, "cohort_sha256": canonical_sha256(cohort),
        "arms": list(ARMS), "views": list(VIEWS), "seeds": list(SEEDS), "cell_count": CELL_COUNT,
        "query_window_count_by_asset_id": query, "query_map_sha256": canonical_sha256(query),
        "query_windows_per_view": QUERY_WINDOWS_PER_VIEW,
        "total_model_windows": QUERY_WINDOWS_PER_VIEW * len(VIEWS) * len(SEEDS) * len(ARMS),
        "reviewed_checkpoint_slots": slots, "reviewed_checkpoint_slots_sha256": canonical_sha256(slots),
        "verified_closure_bundle_sha256": verified.closure_bundle_sha256,
        "score_protocol": SCORE_PROTOCOL, "runtime": CPU_POLICY,
        "claim_separation": CLAIM_SEPARATION, "comparison_gates": COMPARISON_GATES,
        "bootstrap_policy": BOOTSTRAP_POLICY,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def _parse_time(value: Any, label: str) -> datetime:
    auth_require(isinstance(value, str), f"{label} is not ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise V7AuthorizationError(f"{label} malformed") from exc
    auth_require(parsed.tzinfo is not None, f"{label} lacks timezone")
    return parsed.astimezone(timezone.utc)


def _authorization_relative_paths(run_id: str) -> tuple[str, str]:
    auth_require(SAFE_RUN_ID.fullmatch(run_id) is not None, "run id malformed")
    base = f"authorizations/{run_id}"
    return f"{base}/run_authorization.json", f"{base}/run_authorization.ed25519.json"


def _nonce_claim_relative_path(nonce: str) -> str:
    auth_require(is_sha256(nonce), "authorization nonce malformed")
    # This is the only accepted nonce destination: a caller cannot choose a
    # second spelling, parent, extension, or claim root.
    return f"claims/{nonce[:2]}/{nonce}.claim.json"


_GRANT_MINT = object()


class VerifiedGrant:
    __slots__ = ("_mint", "_verified_policy", "_contract", "_proof", "_sealed", "verified_grant_sha256")

    def __init__(self, *, _mint: object | None = None, verified_policy: VerifiedPolicy | None = None, contract: Mapping[str, Any] | None = None, proof: Mapping[str, Any] | None = None):
        if _mint is not _GRANT_MINT or verified_policy is None or contract is None or proof is None:
            raise TypeError("VerifiedGrant can only be minted by the run-authorization verifier")
        object.__setattr__(self, "_mint", _mint); object.__setattr__(self, "_verified_policy", verified_policy)
        object.__setattr__(self, "_contract", dict(contract)); object.__setattr__(self, "_proof", dict(proof))
        object.__setattr__(self, "verified_grant_sha256", canonical_sha256(proof)); object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("VerifiedGrant is immutable")
        object.__setattr__(self, name, value)

    @property
    def verified_policy(self) -> VerifiedPolicy:
        return self._verified_policy

    @property
    def contract(self) -> dict[str, Any]:
        return dict(self._contract)

    @property
    def proof(self) -> dict[str, Any]:
        return dict(self._proof)

    @property
    def output_root(self) -> Path:
        return self._verified_policy.roots.path("output_parent") / "runs" / str(self._proof["run_id"])


def _claim_nonce(roots: TrustedRoots, *, nonce: str, authorization_sha256: str, run_id: str) -> dict[str, Any]:
    relative = _nonce_claim_relative_path(nonce)
    raw = canonical_bytes({"schema": "atomic_canonical_nonce_claim_v7", "nonce": nonce, "authorization_sha256": authorization_sha256, "run_id": run_id})
    return _write_exclusive_at(roots.path("claim_root"), relative, raw)


def verify_run_authorization(verified: VerifiedPolicy, *, run_id: str, now: datetime) -> VerifiedGrant:
    """Verify a short-lived signed authorization in the fixed run-auth root."""
    auth_require(now.tzinfo is not None, "verification time must be timezone-aware")
    verified = _assert_verified_policy(verified)
    roots = verified.roots; contract = build_expected_contract(verified)
    authorization_relative, signature_relative = _authorization_relative_paths(run_id)
    authorization, authorization_read = _read_canonical_json_at(roots.path("run_auth_root"), authorization_relative, expected_mode="0444", max_bytes=1024 * 1024)
    signature_read = fd_read_regular_at(roots.path("run_auth_root"), signature_relative, expected_mode="0444", max_bytes=1024 * 1024)
    _verify_detached_signature(raw=authorization_read.raw, envelope_raw=signature_read.raw, public_pem=_read_pinned_key(roots.path("run_auth_root"), roots.key_pin("run_auth")), schema=AUTH_ENVELOPE_SCHEMA)
    auth_require(isinstance(authorization, Mapping) and set(authorization) == AUTH_KEYS, "run authorization exact schema drift")
    auth_require(authorization["schema"] == AUTH_SCHEMA and authorization["status"] == AUTH_STATUS and authorization["permitted_action"] == AUTH_ACTION and authorization["trusted_root_id"] == roots.root_id and authorization["run_id"] == run_id and authorization["cell_count"] == CELL_COUNT, "run authorization scope/status drift")
    issued, expires = _parse_time(authorization["issued_at"], "issued_at"), _parse_time(authorization["expires_at"], "expires_at")
    now_utc = now.astimezone(timezone.utc)
    auth_require(issued <= now_utc <= expires and 0 < (expires - issued).total_seconds() <= MAX_AUTH_SECONDS, "run authorization expired/not-yet-valid/too-long")
    auth_require(is_sha256(authorization["nonce"]) and authorization["policy_sha256"] == verified.policy_sha256 and authorization["contract_sha256"] == contract["contract_sha256"], "run authorization policy/contract/nonce binding drift")
    auth_pin = {"path": authorization_relative, "sha256": authorization_read.sha256, "bytes": authorization_read.bytes, "mode": authorization_read.mode}
    sig_pin = {"path": signature_relative, "sha256": signature_read.sha256, "bytes": signature_read.bytes, "mode": signature_read.mode}
    claim_pin = _claim_nonce(roots, nonce=authorization["nonce"], authorization_sha256=authorization_read.sha256, run_id=run_id)
    proof = {
        "schema": GRANT_SCHEMA, "trusted_root_digest": roots.root_digest,
        "signed_policy": {"path": _policy_relative_path(), "sha256": verified.policy_sha256},
        "authorization": auth_pin, "signature": sig_pin, "nonce_claim": claim_pin,
        "nonce": authorization["nonce"], "run_id": run_id,
        "contract_sha256": contract["contract_sha256"],
    }
    return VerifiedGrant(_mint=_GRANT_MINT, verified_policy=verified, contract=contract, proof=proof)


def _assert_verified_grant(grant: VerifiedGrant) -> tuple[VerifiedPolicy, dict[str, Any]]:
    auth_require(type(grant) is VerifiedGrant and getattr(grant, "_mint", None) is _GRANT_MINT, "real verifier-minted grant required")
    auth_require(grant.verified_grant_sha256 == canonical_sha256(grant._proof), "verified grant digest drift")
    verified = _assert_verified_policy(grant.verified_policy)
    contract = build_expected_contract(verified)
    proof = grant._proof
    auth_require(
        isinstance(proof, Mapping)
        and proof.get("schema") == GRANT_SCHEMA
        and proof.get("trusted_root_digest") == verified.roots.root_digest
        and proof.get("contract_sha256") == contract["contract_sha256"]
        and proof.get("signed_policy") == {"path": _policy_relative_path(), "sha256": verified.policy_sha256},
        "verified grant policy/contract proof drift",
    )
    run_id = proof.get("run_id")
    auth_require(isinstance(run_id, str) and SAFE_RUN_ID.fullmatch(run_id) is not None, "verified grant run id drift")
    expected_auth, expected_signature = _authorization_relative_paths(run_id)
    for label, expected_path in (("authorization", expected_auth), ("signature", expected_signature)):
        pin = proof.get(label)
        auth_require(isinstance(pin, Mapping) and set(pin) == FILE_PIN_KEYS and pin.get("path") == expected_path, f"verified grant {label} pin drift")
        fd_read_regular_at(verified.roots.path("run_auth_root"), expected_path, expected_mode="0444", expected_sha256=pin["sha256"], expected_bytes=pin["bytes"], max_bytes=1024 * 1024)
    nonce = proof.get("nonce")
    auth_require(is_sha256(nonce), "verified grant nonce drift")
    claim = proof.get("nonce_claim")
    expected_claim = _nonce_claim_relative_path(str(nonce))
    auth_require(isinstance(claim, Mapping) and set(claim) == FILE_PIN_KEYS and claim.get("path") == expected_claim, "verified grant canonical nonce-claim pin drift")
    claim_payload, _ = _read_canonical_json_at(verified.roots.path("claim_root"), expected_claim, pin=claim, max_bytes=1024 * 1024)
    auth_require(claim_payload == {"schema": "atomic_canonical_nonce_claim_v7", "nonce": nonce, "authorization_sha256": proof["authorization"]["sha256"], "run_id": run_id}, "verified grant nonce-claim contents drift")
    auth_require(grant.contract == contract, "verified grant stored contract drift")
    return verified, contract


@dataclass(frozen=True, order=True)
class CellKey:
    session_id: str
    asset_id: str
    view: str
    seed: int
    arm: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_cell_keys(verified: VerifiedPolicy) -> tuple[CellKey, ...]:
    contract = build_expected_contract(verified)
    keys = tuple(
        CellKey(row["session_id"], row["asset_id"], view, seed, arm)
        for row in contract["cohort"]
        for view in VIEWS
        for seed in SEEDS
        for arm in ARMS
    )
    auth_require(len(keys) == len(set(keys)) == CELL_COUNT, "exact 270 cell map drift")
    return keys


def _output_root(grant: VerifiedGrant) -> Path:
    verified, _ = _assert_verified_grant(grant)
    root = grant.output_root
    root_parent = verified.roots.path("output_parent")
    # The run root is derived from signed run_id, not supplied by a policy or
    # caller.  Ensure it is a child of the fixed output root before opening it.
    auth_require(_absolute(root).parent.parent == _absolute(root_parent), "derived output root escapes trusted parent")
    root_fd = _open_absolute_directory(root_parent)
    try:
        parent_fd, leaf = _open_parent_at(root_fd, f"runs/{grant.proof['run_id']}", create=True)
        try:
            metadata = os.fstat(parent_fd)
            ledger_require(stat.S_ISDIR(metadata.st_mode), "derived output root is not a directory")
            # ``parent_fd`` itself points to the leaf because _open_parent_at
            # returns the parent of the final component.  Create/open the final
            # run directory with openat without accepting a symlink.
            try:
                os.mkdir(leaf, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            run_fd = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            try:
                ledger_require(stat.S_ISDIR(os.fstat(run_fd).st_mode), "derived output root failed directory verification")
            finally:
                os.close(run_fd)
        finally:
            os.close(parent_fd)
    finally:
        os.close(root_fd)
    return root


def _artifact_relative(key: CellKey) -> str:
    return f"artifacts/{key.session_id}/{key.view}/seed_{key.seed}/{key.arm}.predictions_targets.npz"


def _cell_relative(key: CellKey) -> str:
    return f"cells/{key.session_id}/{key.view}/seed_{key.seed}/{key.arm}.json"


def _validate_arrays(prediction: Any, target: Any, expected_rows: int) -> tuple[Any, Any]:
    import numpy as np

    ledger_require(isinstance(prediction, np.ndarray) and isinstance(target, np.ndarray), "prediction/target must be numpy arrays")
    ledger_require(prediction.dtype == target.dtype == np.dtype("float32"), "prediction/target dtype must be exact float32")
    ledger_require(prediction.shape == target.shape == (expected_rows, OUTPUT_DIM), "prediction/target shape or row count drift")
    ledger_require(prediction.flags.c_contiguous and target.flags.c_contiguous, "prediction/target must be C-contiguous")
    ledger_require(bool(np.isfinite(prediction).all()) and bool(np.isfinite(target).all()), "prediction/target contains NaN/Inf")
    return prediction, target


def _npz_max_bytes(expected_rows: int) -> int:
    # Two dense float32 arrays require 8 * rows * output_dim raw bytes.  The
    # 16x envelope permits ordinary NPZ/container overhead but bounds both the
    # archive and a potential decompression bomb independently of caller data.
    return min(NPZ_ABSOLUTE_MAX_BYTES, max(1024 * 1024, expected_rows * OUTPUT_DIM * 16 + 1024 * 1024))


def _npz_bytes(prediction: Any, target: Any) -> bytes:
    import numpy as np

    buffer = io.BytesIO()
    np.savez_compressed(buffer, prediction=prediction, target=target)
    return buffer.getvalue()


def _parse_npy_header(member: bytes, *, expected_rows: int) -> tuple[int, Any]:
    """Parse the NPY header before any NumPy deserialization/allocation."""
    import numpy as np

    ledger_require(len(member) >= 10 and member[:6] == b"\x93NUMPY", "NPY magic/version missing")
    major, minor = member[6], member[7]
    ledger_require((major, minor) in {(1, 0), (2, 0)}, "NPY version forbidden")
    length_bytes = 2 if major == 1 else 4
    ledger_require(len(member) >= 8 + length_bytes, "NPY header truncated")
    header_length = int.from_bytes(member[8:8 + length_bytes], "little")
    header_start = 8 + length_bytes; header_end = header_start + header_length
    ledger_require(0 < header_length <= NPY_HEADER_MAX_BYTES and header_end <= len(member), "NPY header length unsafe")
    try:
        header = ast.literal_eval(member[header_start:header_end].decode("latin1"))
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise V7LedgerError("NPY header malformed") from exc
    ledger_require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"}, "NPY header schema drift")
    ledger_require(header["descr"] in {"<f4", "=f4"} and header["fortran_order"] is False and header["shape"] == (expected_rows, OUTPUT_DIM), "NPY dtype/layout/shape forbidden")
    data_bytes = expected_rows * OUTPUT_DIM * 4
    ledger_require(len(member) == header_end + data_bytes, "NPY payload length drift")
    return header_end, np.dtype("<f4")


def _safe_npz_from_raw(raw: bytes, *, expected_rows: int) -> tuple[Any, Any]:
    """Safely load only two exact float32 NPY members from a bounded NPZ."""
    import numpy as np

    ledger_require(len(raw) <= _npz_max_bytes(expected_rows), "NPZ archive exceeds fixed safety cap")
    member_cap = expected_rows * OUTPUT_DIM * 4 + NPY_HEADER_MAX_BYTES
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            ledger_require([info.filename for info in infos] == ["prediction.npy", "target.npy"], "NPZ members must be exactly prediction/target in canonical order")
            members: list[bytes] = []
            for info in infos:
                ledger_require(not info.is_dir() and not (info.flag_bits & 0x1), "encrypted/directory NPZ member forbidden")
                ledger_require(info.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}, "NPZ compression method forbidden")
                ledger_require(0 < info.file_size <= member_cap and 0 < info.compress_size <= len(raw), "NPZ member size unsafe")
                member = archive.read(info)
                ledger_require(len(member) == info.file_size, "NPZ member byte count drift")
                members.append(member)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise V7LedgerError("unsafe or malformed NPZ") from exc
    arrays = []
    for member in members:
        offset, dtype = _parse_npy_header(member, expected_rows=expected_rows)
        array = np.frombuffer(member, dtype=dtype, count=expected_rows * OUTPUT_DIM, offset=offset).reshape((expected_rows, OUTPUT_DIM))
        arrays.append(np.ascontiguousarray(array))
    return _validate_arrays(arrays[0], arrays[1], expected_rows)


def frozen_torchmetrics_variance_weighted_r2(prediction: Any, target: Any) -> float:
    """TorchMetrics 1.5.1 ``R2Score(...variance_weighted)`` semantics.

    The order, float32 reduction dtype, 1e-4 ``torch.isclose`` zero branches,
    and final variance weighting mirror ``_r2_score_update/_r2_score_compute``
    in the frozen 1.5.1 source.  A non-finite result is returned as such; the
    ledger rejects it rather than silently converting it into an R² claim.
    """
    import numpy as np

    rows = int(prediction.shape[0]) if hasattr(prediction, "shape") and len(prediction.shape) == 2 else -1
    prediction, target = _validate_arrays(prediction, target, rows)
    ledger_require(rows >= 2, "TorchMetrics R2 needs at least two samples")
    # TorchMetrics accepts float32 tensors here and keeps its metric states in
    # float32.  Retain that dtype at every arithmetic/reduction point.
    sum_obs = np.sum(target, axis=0, dtype=np.float32)
    sum_squared_obs = np.sum(target * target, axis=0, dtype=np.float32)
    residual = target - prediction
    rss = np.sum(residual * residual, axis=0, dtype=np.float32)
    count = np.float32(rows)
    mean_obs = sum_obs / count
    tss = sum_squared_obs - sum_obs * mean_obs
    cond_rss = np.logical_not(np.isclose(rss, np.zeros_like(rss), rtol=1.0e-5, atol=TORCHMETRICS_NEAR_CONSTANT_ATOL))
    cond_tss = np.logical_not(np.isclose(tss, np.zeros_like(tss), rtol=1.0e-5, atol=TORCHMETRICS_NEAR_CONSTANT_ATOL))
    raw_scores = np.ones_like(rss, dtype=np.float32)
    ordinary = np.logical_and(cond_rss, cond_tss)
    raw_scores[ordinary] = np.float32(1.0) - rss[ordinary] / tss[ordinary]
    raw_scores[np.logical_and(cond_rss, np.logical_not(cond_tss))] = np.float32(0.0)
    tss_sum = np.sum(tss, dtype=np.float32)
    value = np.sum(tss / tss_sum * raw_scores, dtype=np.float32)
    return float(value)


def _cell_rows(contract: Mapping[str, Any], key: CellKey) -> int:
    rows = contract["query_window_count_by_asset_id"].get(key.asset_id)
    ledger_require(positive_int(rows), "cell asset has no frozen query count")
    return int(rows)


def publish_prediction_target_artifact(grant: VerifiedGrant, *, key: CellKey, prediction: Any, target: Any) -> dict[str, Any]:
    verified, contract = _assert_verified_grant(grant)
    ledger_require(key in set(expected_cell_keys(verified)), "unknown cell key")
    rows = _cell_rows(contract, key)
    prediction, target = _validate_arrays(prediction, target, rows)
    raw = _npz_bytes(prediction, target)
    ledger_require(len(raw) <= _npz_max_bytes(rows), "generated NPZ exceeded fixed cap")
    root = _output_root(grant)
    pin = _write_exclusive_at(root, _artifact_relative(key), raw)
    pin.update({
        "verified_grant_sha256": grant.verified_grant_sha256,
        "arrays": {
            "prediction": {"dtype": "float32", "shape": [rows, OUTPUT_DIM]},
            "target": {"dtype": "float32", "shape": [rows, OUTPUT_DIM]},
        },
    })
    return pin


def _load_artifact(grant: VerifiedGrant, key: CellKey, pin: Mapping[str, Any]) -> tuple[Any, Any]:
    verified, contract = _assert_verified_grant(grant)
    root = _output_root(grant); rows = _cell_rows(contract, key)
    ledger_require(isinstance(pin, Mapping) and set(pin) == ARTIFACT_PIN_KEYS, "artifact pin exact schema drift")
    expected_path = _artifact_relative(key)
    expected_arrays = {
        "prediction": {"dtype": "float32", "shape": [rows, OUTPUT_DIM]},
        "target": {"dtype": "float32", "shape": [rows, OUTPUT_DIM]},
    }
    ledger_require(pin.get("path") == expected_path and pin.get("mode") == "0444" and pin.get("verified_grant_sha256") == grant.verified_grant_sha256 and pin.get("arrays") == expected_arrays and is_sha256(pin.get("sha256")) and positive_int(pin.get("bytes")), "artifact path/binding drift")
    observed = fd_read_regular_at(root, expected_path, expected_mode="0444", expected_sha256=pin["sha256"], expected_bytes=pin["bytes"], max_bytes=_npz_max_bytes(rows))
    return _safe_npz_from_raw(observed.raw, expected_rows=rows)


def publish_cell_result(grant: VerifiedGrant, *, key: CellKey, prediction_target_artifact: Mapping[str, Any]) -> dict[str, Any]:
    verified, contract = _assert_verified_grant(grant)
    ledger_require(key in set(expected_cell_keys(verified)), "unknown cell key")
    prediction, target = _load_artifact(grant, key, prediction_target_artifact)
    value = frozen_torchmetrics_variance_weighted_r2(prediction, target)
    ledger_require(math.isfinite(value), "frozen TorchMetrics R2 is nonfinite")
    payload = {
        "schema": CELL_SCHEMA, "status": "COMPLETE",
        "verified_grant_sha256": grant.verified_grant_sha256,
        "cell": key.as_dict(), "r2": value,
        "query_window_count": _cell_rows(contract, key),
        "prediction_target_artifact": dict(prediction_target_artifact),
        "contract_sha256": contract["contract_sha256"],
    }
    _write_exclusive_at(_output_root(grant), _cell_relative(key), canonical_bytes(payload))
    return payload


def _read_cell(grant: VerifiedGrant, key: CellKey) -> dict[str, Any]:
    verified, contract = _assert_verified_grant(grant)
    root = _output_root(grant)
    payload, _ = _read_canonical_json_at(root, _cell_relative(key), expected_mode="0444", max_bytes=4 * 1024 * 1024)
    ledger_require(set(payload) == CELL_KEYS and payload.get("schema") == CELL_SCHEMA and payload.get("status") == "COMPLETE", "cell exact schema/status drift")
    ledger_require(isinstance(payload.get("cell"), Mapping) and set(payload["cell"]) == CELL_ID_KEYS and payload["cell"] == key.as_dict(), "cell identity drift")
    ledger_require(payload.get("verified_grant_sha256") == grant.verified_grant_sha256 and payload.get("contract_sha256") == contract["contract_sha256"] and payload.get("query_window_count") == _cell_rows(contract, key), "cell grant/contract/query binding drift")
    prediction, target = _load_artifact(grant, key, payload["prediction_target_artifact"])
    expected_r2 = frozen_torchmetrics_variance_weighted_r2(prediction, target)
    value = payload.get("r2")
    ledger_require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and math.isfinite(expected_r2) and float(value) == expected_r2, "cell R2 is not exact frozen-metric recomputation")
    return payload


def _list_regular_files_at(root: Path) -> set[str]:
    """Recursive dirfd walk which rejects every symlink and special file."""
    root_fd = _open_absolute_directory(root)
    result: set[str] = set()

    def walk(directory_fd: int, prefix: tuple[str, ...]) -> None:
        try:
            names = os.listdir(directory_fd)
        except OSError as exc:
            raise V7LedgerError("cannot list secure output directory") from exc
        for name in names:
            ledger_require(name not in {"", ".", ".."}, "unsafe output directory entry")
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise V7LedgerError("cannot stat secure output entry") from exc
            ledger_require(not stat.S_ISLNK(info.st_mode), "symlink in output topology")
            relative = "/".join((*prefix, name))
            if stat.S_ISDIR(info.st_mode):
                child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
                try:
                    current = os.fstat(child_fd)
                    ledger_require((current.st_dev, current.st_ino) == (info.st_dev, info.st_ino), "output directory identity changed during walk")
                    walk(child_fd, (*prefix, name))
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode):
                result.add(relative)
            else:
                raise V7LedgerError("special file in output topology")

    try:
        walk(root_fd, ())
        return result
    finally:
        os.close(root_fd)


def _scan_base(grant: VerifiedGrant) -> tuple[VerifiedPolicy, dict[str, Any], dict[CellKey, dict[str, Any]]]:
    verified, contract = _assert_verified_grant(grant)
    root = _output_root(grant)
    keys = expected_cell_keys(verified)
    expected_cells = {_cell_relative(key) for key in keys}
    expected_artifacts = {_artifact_relative(key) for key in keys}
    aggregate_path = "aggregate/aggregate.json"
    observed_files = _list_regular_files_at(root)
    ledger_require(observed_files <= expected_cells | expected_artifacts | {aggregate_path}, "unknown output file")
    cells: dict[CellKey, dict[str, Any]] = {}
    for key in keys:
        if _cell_relative(key) in observed_files:
            cells[key] = _read_cell(grant, key)
    referenced = {str(payload["prediction_target_artifact"]["path"]) for payload in cells.values()}
    observed_artifacts = observed_files & expected_artifacts
    ledger_require(observed_artifacts == referenced, "partial or unreferenced artifact state")
    if aggregate_path in observed_files:
        ledger_require(len(cells) == CELL_COUNT, "aggregate exists before all 270 verified cells")
        stored, _ = _read_canonical_json_at(root, aggregate_path, expected_mode="0444", max_bytes=16 * 1024 * 1024)
        ledger_require(set(stored) == AGGREGATE_KEYS, "aggregate schema drift")
        ledger_require(stored == _reconstruct(cells, contract, grant), "aggregate is not exact 270-cell reconstruction")
    return verified, contract, cells


def _bootstrap(delta: Any, rng: Any) -> tuple[float, float]:
    import numpy as np

    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 10_000):
        count = min(10_000, BOOTSTRAP_REPLICATES - start)
        sessions = rng.integers(0, N, size=(count, N))
        seeds = rng.integers(0, len(SEEDS), size=(count, N, len(SEEDS)))
        values[start:start + count] = delta[sessions[:, :, None], seeds].mean(axis=(1, 2))
    quantiles = np.quantile(values, [0.025, 0.975], method="linear")
    return float(quantiles[0]), float(quantiles[1])


def _reconstruct(payloads: Mapping[CellKey, Mapping[str, Any]], contract: Mapping[str, Any], grant: VerifiedGrant) -> dict[str, Any]:
    import numpy as np

    sessions = [(row["session_id"], row["asset_id"]) for row in contract["cohort"]]
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    comparisons: dict[str, Any] = {}
    for name, comparator in COMPARISONS:
        by_view: dict[str, Any] = {}
        for view in VIEWS:
            t4 = np.asarray([[payloads[CellKey(session, asset, view, seed, "shared_t4")]["r2"] for seed in SEEDS] for session, asset in sessions], dtype=np.float64)
            other = np.asarray([[payloads[CellKey(session, asset, view, seed, comparator)]["r2"] for seed in SEEDS] for session, asset in sessions], dtype=np.float64)
            delta = t4 - other; seed_means = delta.mean(0); session_means = delta.mean(1); t4_seeds = t4.mean(0)
            lower, upper = _bootstrap(delta, rng); positive = int((session_means > 0).sum()); grand = float(delta.mean())
            gates = {
                "grand_paired_mean_at_least_0p03": grand >= 0.03,
                "all_three_seed_means_strictly_positive": bool((seed_means > 0).all()),
                "at_least_12_of_15_session_means_positive": positive >= 12,
                "hierarchical_bootstrap_lower_95_strictly_positive": lower > 0,
                "shared_t4_absolute_grand_and_seed_means_strictly_positive": bool(t4.mean() > 0 and (t4_seeds > 0).all()),
            }
            by_view[view] = {
                "grand_paired_mean_r2": grand,
                "seed_means_r2": {str(seed): float(seed_means[index]) for index, seed in enumerate(SEEDS)},
                "session_cross_seed_means_r2": {sessions[index][0]: float(value) for index, value in enumerate(session_means)},
                "positive_session_count": positive, "positive_session_required": 12,
                "hierarchical_bootstrap_95": {"lower": lower, "upper": upper},
                "shared_t4_absolute_grand_r2": float(t4.mean()),
                "shared_t4_absolute_seed_means_r2": {str(seed): float(t4_seeds[index]) for index, seed in enumerate(SEEDS)},
                "gates": gates, "view_pass": all(gates.values()),
            }
        comparisons[name] = {"views": by_view, "comparison_pass": all(by_view[view]["view_pass"] for view in VIEWS), "cross_view_rescue_used": False}
    return {
        "schema": AGGREGATE_SCHEMA, "status": "RECONSTRUCTED_FROM_270_VERIFIED_CELLS",
        "verified_grant_sha256": grant.verified_grant_sha256,
        "contract_sha256": contract["contract_sha256"],
        "statistics_source": "frozen_torchmetrics_semantics_recomputed_from_safe_npz_and_270_verified_cells",
        "verified_cell_count": CELL_COUNT, "bootstrap_policy": BOOTSTRAP_POLICY,
        "comparisons": comparisons,
        "overall_three_arm_claim_pass": all(comparisons[name]["comparison_pass"] for name, _ in COMPARISONS),
    }


def scan_resume_state(grant: VerifiedGrant) -> dict[str, Any]:
    verified, _, payloads = _scan_base(grant)
    keys = expected_cell_keys(verified); complete = set(payloads)
    return {
        "expected_cell_count": CELL_COUNT, "complete_cell_count": len(complete),
        "missing_cell_count": CELL_COUNT - len(complete),
        "complete": [key.as_dict() for key in keys if key in complete],
        "missing": [key.as_dict() for key in keys if key not in complete],
    }


def publish_full_aggregate(grant: VerifiedGrant) -> dict[str, Any]:
    _, contract, payloads = _scan_base(grant)
    ledger_require(len(payloads) == CELL_COUNT, "aggregate forbidden before all 270 verified cells")
    payload = _reconstruct(payloads, contract, grant)
    _write_exclusive_at(_output_root(grant), "aggregate/aggregate.json", canonical_bytes(payload))
    return payload


def blocked_checkpoint_slots() -> list[dict[str, Any]]:
    """Historical V6 slots are evidence only; none is a V7 closure grant."""
    return v6.blocked_checkpoint_slots()


def refuse_blocked_execution() -> None:
    raise V7BlockedError("BLOCKED_MISSING_ZERO4_TERMINALS")
