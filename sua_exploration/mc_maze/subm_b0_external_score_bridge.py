"""Fail-closed score-only bridge for the archived original-SPINT B0 sources.

This is an additive consumer of two already-existing lineages:

* the three B0 source runs from ``sua_spint_t4_mainline_fp32_v1``; and
* the sealed A2 M30 subject-M T4/Z4 receipts.

It never trains, never chooses an epoch from target outcomes, and has no GPU
path.  The default/preflight path is deliberately target-free.  The eventual
CPU scorer is reachable only through an immutable root authorization pair that
pins both this exact source audit and the six A2 external reference receipts.

The bridge does *not* call B0 a carrier ablation.  B0 is original SPINT
(``BatchReferenceEncoder`` with trainable copied ``fc_id_in/out``); A2 T4/Z4
are B3S systems.  Consequently T4−B0 and Z4−B0 are system contrasts.  The
separately sealed A2 T4−Z4 external-minus-within interaction remains the only
carrier-content interaction reported by this route.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
CHECKPOINT_ROOT = SUA_ROOT / "checkpoints"
A2_ROOT = SUA_ROOT / "results" / "a2_matched_subject_shift_v2"
RESULT_ROOT = SUA_ROOT / "results" / "subm_b0_external_score_bridge_v1"
SUBM_SCHEMA_LEDGER = SUA_ROOT / "results" / "dandi_000688_subm_co_schema_preflight_v2" / "receipt.json"
SUBM_SCOPE_MANIFEST = SUA_ROOT / "manifests" / "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json"

SEEDS = (42, 43, 44)
EPOCH_WINDOW = tuple(range(5, 13))
EXTERNAL_DOMAIN = "external_subject_M"
EXPECTED_EXTERNAL_SESSION_COUNT = 15
SCHEMA = "subm_b0_external_score_bridge_v1"
PREFLIGHT_SCHEMA = "subm_b0_external_score_bridge_preflight_v1"
SCORE_SCHEMA = "subm_b0_external_score_bridge_cpu_score_v1"
AGGREGATE_SCHEMA = "subm_b0_external_score_bridge_aggregate_v1"
ROOT_AUTHORIZATION_SCHEMA = "subm_b0_external_score_bridge_root_execution_authorization_v1"

STRICT_MANIFEST = SUA_ROOT / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
TEACHER = CHECKPOINT_ROOT / "teacher_mc_maze" / "best-epoch=083-val_heldin" / "r2_mean=0.9061.ckpt"
BEHAVIOR_NORMALIZER = SUA_ROOT / "cache" / "dandi688_subc_co_v1" / "behavior_stats" / "be50f588491c004f721e.npz"

# The prospective CPU scorer imports precisely these active surfaces.  Root's
# immutable preflight snapshots the map, and both authorization loading and
# every eventual score cell reject a changed path or byte before target access.
IMPLEMENTATION_BINDING_PATHS: dict[str, Path] = {
    "bridge_core": Path(__file__).resolve(),
    "read_only_preflight_cli": SUA_ROOT / "scripts" / "preflight_subm_b0_external_score_bridge.py",
    "score_cli": SUA_ROOT / "scripts" / "score_subm_b0_external_score_bridge.py",
    "aggregate_cli": SUA_ROOT / "scripts" / "aggregate_subm_b0_external_score_bridge.py",
    "root_preflight_publisher_cli": SUA_ROOT / "scripts" / "write_subm_b0_external_score_bridge_preflight.py",
    "root_authorization_publisher_cli": SUA_ROOT / "scripts" / "authorize_subm_b0_external_score_bridge.py",
    "a2_contract_core": SUA_ROOT / "mc_maze" / "a2_matched_subject_shift_v2_core.py",
    "multisession_datamodule": SUA_ROOT / "mc_maze" / "multisession_datamodule.py",
    "adaptation_evaluator": SUA_ROOT / "scripts" / "eval_adaptation_dandi688.py",
    "frozen_model_loader": SUA_ROOT / "scripts" / "select_gradient_free_protocol_dandi688.py",
    "streaming_calibration_module": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "streaming_calibration_module.py",
    "streaming_encoders": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_encoders.py",
    "streaming_spint": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_spint.py",
}

EXPECTED_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
EXPECTED_TEACHER_SHA256 = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"
EXPECTED_BEHAVIOR_NORMALIZER_FILE_SHA256 = "821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd"
EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256 = "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
A2_TERMINAL_SHA256 = "5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc"
A2_OFFICIAL_PREFLIGHT_SHA256 = "8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd"
SUBM_SCHEMA_LEDGER_SHA256 = "1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283"
SUBM_SCOPE_MANIFEST_SHA256 = "68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55"

A2_EXTERNAL_RECEIPT_SHA256: dict[tuple[str, int], str] = {
    ("t4", 42): "3ab4ca6993b110bea61e8a1dfdb9e8f9cc510b8daf2b0dccf15cb110e98f7548",
    ("t4", 43): "a646e126b17f1877ee40da8a593f8aaf086174fb2f358e5e912bee6b91cb33a6",
    ("t4", 44): "9f340c542fd84bc3b9cd3458492ba0806dbb3c593d5f7852b477fb94add38818",
    ("z4", 42): "9417e853b408ff6f7959be18a2b54ad8443012372ebcaf19d38b93c0df2cde7a",
    ("z4", 43): "98f93ebb89cceb5c9ea5e599513d037e9b6cc251f94e561b6794cd4c8cb85950",
    ("z4", 44): "c4d132f893cda3c1af65b06d592c681a411aa65324a801f0bb0c7ef6b7258d11",
}

# The A11 receipt is an immutable CPU-only replay of the six development
# sub-C validation sessions.  It supplies B0's within-domain reference only;
# it is not a target score and it is never substituted for an external cell.
A11_WITHIN_B0_PATH = SUA_ROOT / "results" / "a11_b0_convergence_full_access_v1" / (
    "a11_b0_convergence_full_access_v1_full_cpu_forward_2843108a665b53b4.json"
)
A11_WITHIN_B0_SHA256 = "955ebaf8b1ac229317118bb7c3bf3ebdc8c6c62c5f5be2ea1440d32d78724614"
A11_WITHIN_B0_MEAN_R2 = 0.23641659274774915

A2_WITHIN_RECEIPT_SHA256: dict[tuple[str, int], str] = {
    ("t4", 42): "588c4123b895878e03b6b4a18d829a8fc9bcc9f1771fc943d9cc06e72c106955",
    ("t4", 43): "0ad9117aeec4c7771ff93f68d727150cc3cfaebb800cd2c7654a2f941badc227",
    ("t4", 44): "9df1022e23dff0868b33c40e795c203e99b0c51629a354b1a4ae56c639ce307c",
    ("z4", 42): "0e10261a59d3c111b4bf4766c5511d62735fe73910b744a0e60eafce71cc83d5",
    ("z4", 43): "05fa7b154abeba402caa7e48546692a6ad95c6094686874ce47365218393c383",
    ("z4", 44): "d3d59f5a5f9f91b80dcd98bd3916e95f2d885c350cd9f89b482f0529e070af0d",
}


class B0BridgeError(RuntimeError):
    """A source/A2 binding or score-only boundary has drifted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise B0BridgeError(message)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


@dataclass(frozen=True)
class _VerifiedBytes:
    """Exactly the bytes read through one non-following file descriptor.

    The bridge must not calculate an input hash from one pathname read and
    then hand a second pathname read to Torch/NumPy.  ``raw`` is consequently
    the sole byte authority used by an in-process consumer, or the source for
    a private immutable snapshot used by a third-party pathname-only loader.
    """

    path: Path
    raw: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]
    mode: int


@dataclass(frozen=True)
class _VerifiedDigest:
    """Same-FD file identity and digest without retaining large file bytes."""

    path: Path
    sha256: str
    identity: tuple[int, int, int, int, int, int]
    mode: int
    byte_count: int


def _absolute_lexical(path: Path) -> Path:
    """Return an absolute path without resolving any symlink."""
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path(os.path.abspath(str(expanded)))


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mode,
            info.st_mtime_ns, info.st_ctime_ns)


def _assert_real_directory_chain(path: Path, label: str, *, create: bool = False) -> Path:
    """Reject symlinked parents, optionally creating only real directories."""
    directory = _absolute_lexical(path)
    require(directory.is_absolute(), f"{label} parent path must be absolute")
    current = Path(directory.anchor)
    for part in directory.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            require(create, f"{label} parent is absent: {current}")
            try:
                os.mkdir(current, 0o755)
            except FileExistsError:
                pass
            info = current.lstat()
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                f"{label} parent must be a real directory: {current}")
    return directory


def _path_identity_matches(path: Path, identity: tuple[int, int, int, int, int, int], label: str) -> None:
    """Fail if a pathname was replaced after its descriptor was opened."""
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise B0BridgeError(f"{label} pathname disappeared after open") from exc
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            f"{label} pathname is no longer a regular non-symlink")
    require(_file_identity(info) == identity, f"{label} pathname identity changed after open")


def _read_all_fd(fd: int) -> bytes:
    blocks: list[bytes] = []
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            return b"".join(blocks)
        blocks.append(block)


def _read_verified_bytes(
    path: Path,
    label: str,
    *,
    mode_0444: bool = False,
    expected_sha256: str | None = None,
) -> _VerifiedBytes:
    """Read, hash, and identity-check a regular file from one O_NOFOLLOW FD."""
    lexical = _absolute_lexical(path)
    _assert_real_directory_chain(lexical.parent, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW required by B0 bridge")
    flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lexical, flags)
    except OSError as exc:
        raise B0BridgeError(f"{label} cannot be opened without following symlinks") from exc
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        if mode_0444:
            require(stat.S_IMODE(before.st_mode) == 0o444, f"{label} must have mode 0444")
        raw = _read_all_fd(fd)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = _file_identity(before)
    require(_file_identity(after) == identity, f"{label} changed while being read")
    _path_identity_matches(lexical, identity, label)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        require(digest == expected_sha256, f"{label} SHA drift")
    return _VerifiedBytes(path=lexical, raw=raw, sha256=digest, identity=identity,
                          mode=stat.S_IMODE(before.st_mode))


def _hash_verified_file_same_fd(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> _VerifiedDigest:
    """Stream-hash a large regular file through one O_NOFOLLOW descriptor."""
    lexical = _absolute_lexical(path)
    _assert_real_directory_chain(lexical.parent, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW required by B0 bridge")
    try:
        fd = os.open(lexical, flags | os.O_NOFOLLOW)
    except OSError as exc:
        raise B0BridgeError(f"{label} cannot be opened without following symlinks") from exc
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = _file_identity(before)
    require(_file_identity(after) == identity and byte_count == before.st_size,
            f"{label} changed while being read")
    _path_identity_matches(lexical, identity, label)
    observed = digest.hexdigest()
    if expected_sha256 is not None:
        require(observed == expected_sha256, f"{label} SHA drift")
    return _VerifiedDigest(path=lexical, sha256=observed, identity=identity,
                           mode=stat.S_IMODE(before.st_mode), byte_count=byte_count)


def sha256_file(path: Path) -> str:
    """Compatibility helper with the bridge's no-reopen integrity semantics."""
    return _read_verified_bytes(path, f"file SHA {path}").sha256


def _regular(path: Path, label: str, *, mode_0444: bool = False) -> Path:
    """Compatibility path validator; source consumers should use verified bytes."""
    return _read_verified_bytes(path, label, mode_0444=mode_0444).path


def _strict_pair(path: Path, *, label: str, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    body = _read_verified_bytes(path, label, mode_0444=True, expected_sha256=expected_sha256)
    sidecar = _read_verified_bytes(body.path.with_name(f"{body.path.name}.sha256"),
                                   f"{label} sidecar", mode_0444=True)
    # The body must still name exactly the descriptor we hashed after the
    # sidecar was consumed; otherwise a rename between the two reads is a
    # failed pair rather than a stale success.
    _path_identity_matches(body.path, body.identity, label)
    require(sidecar.raw == f"{body.sha256}  {body.path.name}\n".encode("ascii"),
            f"{label} body/sidecar mismatch")
    try:
        payload = json.loads(body.raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise B0BridgeError(f"{label} malformed JSON") from exc
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload, body.sha256


def _strict_unpaired_json(path: Path, *, label: str, expected_sha256: str) -> tuple[dict[str, Any], str]:
    """Read a legacy immutable 0444 JSON body that intentionally has no sidecar."""
    body = _read_verified_bytes(path, label, mode_0444=True, expected_sha256=expected_sha256)
    require(not os.path.lexists(body.path.with_name(f"{body.path.name}.sha256")),
            f"{label} must remain a sidecarless immutable legacy body")
    try:
        payload = json.loads(body.raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise B0BridgeError(f"{label} malformed JSON") from exc
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload, body.sha256


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW required by B0 bridge")
    fd = os.open(directory, flags | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all_fd(fd: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        offset += os.write(fd, raw[offset:])


def _write_exclusive_regular(path: Path, raw: bytes, *, mode: int) -> tuple[int, int, int, int, int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        _write_all_fd(fd, raw)
        os.fsync(fd)
        os.fchmod(fd, mode)
        os.fsync(fd)
        info = os.fstat(fd)
    finally:
        os.close(fd)
    require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == mode,
            "new bridge output lost its requested regular-file mode")
    return _file_identity(info)


def _rollback_owned_output(path: Path, identity: tuple[int, int, int, int, int, int], label: str) -> None:
    """Delete only the exact file created by this transaction, never a swap."""
    try:
        _path_identity_matches(path, identity, label)
    except B0BridgeError as exc:
        raise B0BridgeError(f"{label} rollback unsafe; refusing to unlink a replacement") from exc
    os.unlink(path)


def _write_pair_once(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
    body = _absolute_lexical(path)
    _assert_real_directory_chain(body.parent, "bridge output", create=True)
    sidecar = body.with_name(f"{body.name}.sha256")
    require(not os.path.lexists(body) and not os.path.lexists(sidecar), "refusing to overwrite bridge output")
    raw = canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    created: list[tuple[Path, tuple[int, int, int, int, int, int], str]] = []
    try:
        created.append((body, _write_exclusive_regular(body, raw, mode=0o444), "bridge output body"))
        _fsync_directory(body.parent)
        created.append((sidecar, _write_exclusive_regular(
            sidecar, f"{digest}  {body.name}\n".encode("ascii"), mode=0o444), "bridge output sidecar"))
        _fsync_directory(body.parent)
        _strict_pair(body, label="new bridge output", expected_sha256=digest)
    except Exception as exc:
        rollback_errors: list[str] = []
        for created_path, identity, created_label in reversed(created):
            try:
                _rollback_owned_output(created_path, identity, created_label)
            except (B0BridgeError, OSError) as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        try:
            _fsync_directory(body.parent)
        except OSError as rollback_exc:
            rollback_errors.append(f"directory fsync after rollback: {rollback_exc}")
        if rollback_errors:
            raise B0BridgeError("bridge pair transaction failed and rollback was incomplete: " + "; ".join(rollback_errors)) from exc
        if isinstance(exc, B0BridgeError):
            raise
        raise B0BridgeError("bridge pair transaction failed; created body was rolled back") from exc
    return body, sidecar, digest


def b0_run_dir(seed: int) -> Path:
    require(seed in SEEDS, "unsupported B0 seed")
    return CHECKPOINT_ROOT / f"sua_spint_t4_mainline_fp32_v1_b0_dandi688_co_s{seed}"


def b0_score_path(seed: int, *, result_root: Path = RESULT_ROOT) -> Path:
    require(seed in SEEDS, "unsupported B0 seed")
    return Path(result_root) / f"external_subject_M_b0_s{seed}.json"


def implementation_bindings() -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, raw_path in IMPLEMENTATION_BINDING_PATHS.items():
        verified = _read_verified_bytes(raw_path, f"implementation binding {name}")
        bindings[name] = {"path": str(verified.path), "sha256": verified.sha256}
    return bindings


def verify_implementation_bindings(bindings: Any) -> dict[str, dict[str, str]]:
    require(isinstance(bindings, Mapping) and set(bindings) == set(IMPLEMENTATION_BINDING_PATHS),
            "B0 bridge implementation binding key set drift")
    normalized: dict[str, dict[str, str]] = {}
    for name in IMPLEMENTATION_BINDING_PATHS:
        row = bindings.get(name)
        require(isinstance(row, Mapping) and isinstance(row.get("path"), str) and isinstance(row.get("sha256"), str),
                f"B0 bridge implementation binding malformed: {name}")
        normalized[name] = {"path": str(row["path"]), "sha256": str(row["sha256"])}
    require(normalized == implementation_bindings(), "B0 bridge implementation source/path drift")
    return normalized


def _a2_external_path(arm: str, seed: int) -> Path:
    require(arm in {"t4", "z4"} and seed in SEEDS, "invalid A2 external cell")
    return A2_ROOT / f"external_subject_M_source_{arm}_s{seed}.json"


def _a2_within_path(arm: str, seed: int) -> Path:
    require(arm in {"t4", "z4"} and seed in SEEDS, "invalid A2 within cell")
    return A2_ROOT / f"within_subject_source_{arm}_s{seed}.json"


def _query_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "usable_rewarded_trial_count", "activity_support_usable_indices", "activity_support_original_trial_indices",
        "query_usable_trial_indices_start", "query_usable_trial_count", "post30_query_window_count",
        "dataset_query_window_count",
    )
    require(all(key in row for key in keys), "query receipt lacks M30 projection field")
    return {key: row[key] for key in keys}


def _validate_fixed_epoch_means(payload: Mapping[str, Any], *, roster: Sequence[str], label: str) -> None:
    per_epoch = payload.get("per_epoch")
    per_session = payload.get("per_session_mean_r2")
    require(isinstance(per_epoch, Mapping) and set(per_epoch) == {str(epoch) for epoch in EPOCH_WINDOW},
            f"{label} fixed epoch set drift")
    require(isinstance(per_session, Mapping) and tuple(per_session) == tuple(roster),
            f"{label} session mean roster/order drift")
    for epoch in EPOCH_WINDOW:
        row = per_epoch[str(epoch)]
        require(isinstance(row, Mapping) and isinstance(row.get("per_session_r2"), Mapping) and
                tuple(row["per_session_r2"]) == tuple(roster), f"{label} per-epoch roster drift")
        recomputed_epoch = sum(float(row["per_session_r2"][name]) for name in roster) / len(roster)
        require(float(row.get("mean_r2")) == recomputed_epoch, f"{label} per-epoch mean drift")
    for name in roster:
        recomputed_session = sum(float(per_epoch[str(epoch)]["per_session_r2"][name]) for epoch in EPOCH_WINDOW) / len(EPOCH_WINDOW)
        require(float(per_session[name]) == recomputed_session, f"{label} fixed epoch session mean drift")
    require(float(payload.get("mean_r2")) == sum(float(per_session[name]) for name in roster) / len(roster),
            f"{label} overall mean drift")


def _validate_a2_external_receipt(payload: Mapping[str, Any], *, arm: str, seed: int) -> tuple[str, ...]:
    require(payload.get("screen_id") == "a2_matched_subject_shift_v2", "A2 screen drift")
    require(payload.get("source_arm") == f"source_{arm}" and payload.get("arm") == arm, "A2 arm drift")
    require(payload.get("seed") == seed and payload.get("domain") == EXTERNAL_DOMAIN, "A2 cell identity drift")
    require(payload.get("variant") == "B3S", "A2 comparator must remain B3S")
    protocol = payload.get("protocol")
    require(isinstance(protocol, Mapping) and protocol.get("epoch_window") == list(EPOCH_WINDOW), "A2 epoch window drift")
    require(protocol.get("activity_calibration_n") == 30 and protocol.get("evaluation_start_trial_index") == 30,
            "A2 M30/query boundary drift")
    norm = payload.get("normalizer_authority")
    require(isinstance(norm, Mapping) and norm.get("behavior_normalizer_value_sha256") == EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256,
            "A2 behavior normalizer drift")
    require(norm.get("target_domain_normalizer_refit_performed") is False, "A2 target normalizer was refit")
    sessions = payload.get("domain_sessions")
    queries = payload.get("session_query_receipts")
    require(isinstance(sessions, list) and len(sessions) == EXPECTED_EXTERNAL_SESSION_COUNT and len(set(sessions)) == len(sessions),
            "A2 external roster drift")
    require(isinstance(queries, Mapping) and tuple(queries) == tuple(sessions), "A2 external query roster/order drift")
    for session in sessions:
        _query_projection(queries[session])
    return tuple(str(item) for item in sessions)


def load_sealed_a2_external_authority() -> dict[str, Any]:
    """Read no target data: verify the A2 external matrix and interaction authority."""
    terminal, terminal_sha = _strict_pair(A2_ROOT / "terminal_aggregate.json", label="A2 terminal aggregate",
                                          expected_sha256=A2_TERMINAL_SHA256)
    interaction = terminal.get("interaction")
    require(isinstance(interaction, Mapping), "A2 terminal interaction absent")
    require(interaction.get("definition") == (
        "mean_session_R2(t4 external_subject_M) - mean_session_R2(z4 external_subject_M) - "
        "[mean_session_R2(t4 within_subject) - mean_session_R2(z4 within_subject)]"
    ), "A2 interaction definition drift")
    cells: dict[str, dict[str, Any]] = {}
    roster: tuple[str, ...] | None = None
    for arm in ("t4", "z4"):
        for seed in SEEDS:
            payload, digest = _strict_pair(_a2_external_path(arm, seed), label=f"A2 {arm}/s{seed}",
                                            expected_sha256=A2_EXTERNAL_RECEIPT_SHA256[(arm, seed)])
            observed = _validate_a2_external_receipt(payload, arm=arm, seed=seed)
            if roster is None:
                roster = observed
            else:
                require(observed == roster, "A2 external roster differs across arms/seeds")
            cells[f"{arm}_s{seed}"] = {"payload": payload, "sha256": digest}
    require(roster is not None, "A2 external authority empty")
    for seed in SEEDS:
        t4, z4 = cells[f"t4_s{seed}"]["payload"], cells[f"z4_s{seed}"]["payload"]
        require(t4.get("normalizer_authority") == z4.get("normalizer_authority"),
                "sealed A2 T4/Z4 normalizer authority differs")
        for session in roster:
            require(_query_projection(t4["session_query_receipts"][session]) ==
                    _query_projection(z4["session_query_receipts"][session]),
                    "sealed A2 T4/Z4 query projection differs")
    return {
        "terminal_aggregate_path": str(_absolute_lexical(A2_ROOT / "terminal_aggregate.json")),
        "terminal_aggregate_sha256": terminal_sha,
        "interaction": dict(interaction),
        "external_session_order": list(roster),
        "external_cells": cells,
    }


def _build_subm_target_byte_authority(
    *,
    official_preflight: Mapping[str, Any],
    ledger: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_session_order: Sequence[str],
) -> dict[str, Any]:
    """Join the A2 audit, v2 ledger, manifest, and downloaded-byte pins.

    This operates only on the three immutable JSON authorities.  It neither
    resolves nor opens a subject-M NWB.  The eventual scorer separately
    verifies every current local NWB against the result before passing its
    pathname to the HDF5/NWB loader.
    """
    require(official_preflight.get("screen_id") == "a2_matched_subject_shift_v2" and
            official_preflight.get("official_preflight") is True and
            official_preflight.get("status") == "CPU_PREFLIGHT_PASSED_AWAITING_ROOT_GO" and
            official_preflight.get("receipt_kind") == "a2_matched_subject_shift_v2_official_preflight",
            "A2 official preflight identity/status drift")
    external_audit = official_preflight.get("external_subject_M_audit")
    require(isinstance(external_audit, Mapping) and external_audit.get("expected_count") == 15 and
            external_audit.get("admissible_count") == 15, "A2 official external audit count drift")
    a2_rows = external_audit.get("sessions")
    require(isinstance(a2_rows, list) and len(a2_rows) == EXPECTED_EXTERNAL_SESSION_COUNT,
            "A2 official external audit rows absent")
    observed_order = tuple(str(row.get("session")) for row in a2_rows if isinstance(row, Mapping))
    require(observed_order == tuple(expected_session_order) and len(set(observed_order)) == EXPECTED_EXTERNAL_SESSION_COUNT,
            "A2 official external session order drift")
    for row in a2_rows:
        require(isinstance(row, Mapping) and row.get("admissible") is True and
                row.get("eligibility_ledger_sha256") == SUBM_SCHEMA_LEDGER_SHA256 and
                isinstance(row.get("nwb_path"), str), "A2 official eligibility-ledger binding drift")

    require(ledger.get("schema_version") == 2 and
            ledger.get("receipt_kind") == "dandi_000688_subm_co_score_blind_schema_preflight_v2" and
            ledger.get("scope_id") == "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2" and
            ledger.get("status") == "COMPLETE_SCORE_BLIND_SCHEMA_PREFLIGHT" and
            ledger.get("eligible_session_count") == EXPECTED_EXTERNAL_SESSION_COUNT,
            "sub-M immutable schema ledger identity/status drift")
    manifest_binding = ledger.get("immutable_v2_manifest")
    require(isinstance(manifest_binding, Mapping) and
            manifest_binding.get("path") == "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json" and
            manifest_binding.get("sha256") == SUBM_SCOPE_MANIFEST_SHA256 and
            manifest_binding.get("selected_asset_count") == 22,
            "sub-M ledger/manifest binding drift")
    require(manifest.get("scope_id") == ledger.get("scope_id") and
            manifest.get("status") == "candidate_frozen_metadata_only_external_runner_blocked" and
            isinstance(manifest.get("selected_assets"), list) and len(manifest["selected_assets"]) == 22,
            "sub-M scope manifest identity/asset count drift")
    selected = manifest["selected_assets"]
    downloads = ledger.get("verified_downloads")
    dispositions = ledger.get("asset_disposition_ledger")
    eligible_ids = ledger.get("eligible_session_ids")
    require(isinstance(downloads, list) and len(downloads) == 22 and isinstance(dispositions, list) and
            len(dispositions) == 22 and isinstance(eligible_ids, list) and len(eligible_ids) == EXPECTED_EXTERNAL_SESSION_COUNT,
            "sub-M ledger rows/counts drift")
    selected_by_session = {str(row.get("session_id")): row for row in selected if isinstance(row, Mapping)}
    download_by_asset = {str(row.get("asset_id")): row for row in downloads if isinstance(row, Mapping)}
    ledger_by_session = {str(row.get("session_id")): row for row in dispositions if isinstance(row, Mapping)}
    require(len(selected_by_session) == len(download_by_asset) == len(ledger_by_session) == 22,
            "sub-M manifest/download/ledger uniqueness drift")
    a2_by_session = {str(row["session"]): row for row in a2_rows}
    sessions: dict[str, dict[str, Any]] = {}
    for session in expected_session_order:
        asset, download, disposition, a2_row = (selected_by_session.get(session), download_by_asset.get(
            str(selected_by_session.get(session, {}).get("asset_id"))), ledger_by_session.get(session), a2_by_session.get(session))
        require(isinstance(asset, Mapping) and isinstance(download, Mapping) and isinstance(disposition, Mapping) and
                isinstance(a2_row, Mapping), f"sub-M target authority missing session {session}")
        asset_id = asset.get("asset_id")
        frozen_path = asset.get("path")
        expected_sha = asset.get("sha256")
        expected_bytes = asset.get("size")
        require(isinstance(asset_id, str) and isinstance(frozen_path, str) and isinstance(expected_sha, str) and
                len(expected_sha) == 64 and isinstance(expected_bytes, int) and expected_bytes > 0,
                f"sub-M manifest target row malformed: {session}")
        require(download.get("asset_id") == asset_id and download.get("sha256") == expected_sha and
                download.get("bytes") == expected_bytes and download.get("size_and_sha256_verified_before_nwb_open") is True,
                f"sub-M verified-download pin drift: {session}")
        require(disposition.get("asset_id") == asset_id and disposition.get("frozen_path") == frozen_path and
                disposition.get("eligible") is True and disposition.get("disposition") == "ELIGIBLE" and
                disposition.get("score_blind") is True and asset_id in eligible_ids,
                f"sub-M score-blind eligibility drift: {session}")
        official_path = _absolute_lexical(Path(str(a2_row["nwb_path"])))
        require(official_path.name == Path(frozen_path).name and official_path.parts[-2:] == tuple(Path(frozen_path).parts),
                f"A2 official local target path does not match frozen manifest path: {session}")
        sessions[session] = {
            "asset_id": asset_id,
            "frozen_path": frozen_path,
            "expected_sha256": expected_sha,
            "expected_bytes": expected_bytes,
            "a2_official_nwb_path": str(official_path),
        }
    return {
        "a2_official_preflight_path": str(_absolute_lexical(A2_ROOT / "official_cpu_preflight.json")),
        "a2_official_preflight_sha256": A2_OFFICIAL_PREFLIGHT_SHA256,
        "schema_ledger_path": str(_absolute_lexical(SUBM_SCHEMA_LEDGER)),
        "schema_ledger_sha256": SUBM_SCHEMA_LEDGER_SHA256,
        "scope_manifest_path": str(_absolute_lexical(SUBM_SCOPE_MANIFEST)),
        "scope_manifest_sha256": SUBM_SCOPE_MANIFEST_SHA256,
        "external_session_order": list(expected_session_order),
        "sessions": sessions,
        "target_nwb_opened_while_building_authority": False,
    }


def load_subm_target_byte_authority(*, expected_session_order: Sequence[str] | None = None) -> dict[str, Any]:
    """Load the immutable subject-M byte authority without opening an NWB."""
    expected = tuple(expected_session_order) if expected_session_order is not None else tuple(
        load_sealed_a2_external_authority()["external_session_order"]
    )
    official, official_sha = _strict_pair(A2_ROOT / "official_cpu_preflight.json", label="A2 official preflight",
                                          expected_sha256=A2_OFFICIAL_PREFLIGHT_SHA256)
    ledger, ledger_sha = _strict_unpaired_json(SUBM_SCHEMA_LEDGER, label="sub-M schema ledger",
                                               expected_sha256=SUBM_SCHEMA_LEDGER_SHA256)
    manifest, manifest_sha = _strict_unpaired_json(SUBM_SCOPE_MANIFEST, label="sub-M scope manifest",
                                                    expected_sha256=SUBM_SCOPE_MANIFEST_SHA256)
    authority = _build_subm_target_byte_authority(official_preflight=official, ledger=ledger, manifest=manifest,
                                                  expected_session_order=expected)
    require(official_sha == authority["a2_official_preflight_sha256"] and
            ledger_sha == authority["schema_ledger_sha256"] and manifest_sha == authority["scope_manifest_sha256"],
            "sub-M authority SHA projection drift")
    return authority


def _verify_target_nwb_before_or_after_loader(
    path: Path, *, session: str, target_authority: Mapping[str, Any], phase: str,
) -> dict[str, Any]:
    """Verify current target bytes without parsing NWB content or following links."""
    require(phase in {"before_nwb_loader", "after_nwb_loader"}, "invalid target byte verification phase")
    sessions = target_authority.get("sessions")
    require(isinstance(sessions, Mapping) and session in sessions and isinstance(sessions[session], Mapping),
            f"target byte authority lacks session {session}")
    expected = sessions[session]
    lexical = _absolute_lexical(path)
    require(lexical == _absolute_lexical(Path(str(expected.get("a2_official_nwb_path", "")))),
            f"target path differs from A2 official preflight: {session}")
    frozen_path = Path(str(expected.get("frozen_path", "")))
    require(lexical.name == frozen_path.name and lexical.parts[-2:] == tuple(frozen_path.parts),
            f"target path differs from frozen manifest path: {session}")
    expected_sha = expected.get("expected_sha256")
    expected_bytes = expected.get("expected_bytes")
    require(isinstance(expected_sha, str) and isinstance(expected_bytes, int), "target byte authority row malformed")
    verified = _hash_verified_file_same_fd(lexical, f"target NWB {session} {phase}", expected_sha256=expected_sha)
    require(verified.byte_count == expected_bytes and verified.identity[2] == expected_bytes,
            f"target NWB byte-size drift: {session}")
    identity = {
        "st_dev": verified.identity[0], "st_ino": verified.identity[1], "st_size": verified.identity[2],
        "st_mode": verified.identity[3], "st_mtime_ns": verified.identity[4], "st_ctime_ns": verified.identity[5],
    }
    return {
        "phase": phase,
        "actual_sha256": verified.sha256,
        "actual_bytes": verified.byte_count,
        "identity": identity,
        "identity_sha256": hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
    }


def load_sealed_a2_within_authority() -> dict[str, Any]:
    """Load exact A2 development receipts needed for a B0 system-shift view."""
    cells: dict[str, dict[str, Any]] = {}
    roster: tuple[str, ...] | None = None
    for arm in ("t4", "z4"):
        for seed in SEEDS:
            payload, digest = _strict_pair(_a2_within_path(arm, seed), label=f"A2 within {arm}/s{seed}",
                                            expected_sha256=A2_WITHIN_RECEIPT_SHA256[(arm, seed)])
            require(payload.get("screen_id") == "a2_matched_subject_shift_v2" and
                    payload.get("source_arm") == f"source_{arm}" and payload.get("arm") == arm and
                    payload.get("seed") == seed and payload.get("domain") == "within_subject",
                    "A2 within cell identity drift")
            protocol = payload.get("protocol")
            normalizer = payload.get("normalizer_authority")
            sessions = payload.get("domain_sessions")
            require(isinstance(protocol, Mapping) and protocol.get("epoch_window") == list(EPOCH_WINDOW) and
                    protocol.get("activity_calibration_n") == 30 and protocol.get("evaluation_start_trial_index") == 30 and
                    protocol.get("signal_view") == "sua", "A2 within protocol drift")
            require(isinstance(normalizer, Mapping) and
                    normalizer.get("behavior_normalizer_value_sha256") == EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256 and
                    normalizer.get("target_domain_normalizer_refit_performed") is False,
                    "A2 within behavior normalizer drift")
            require(isinstance(sessions, list) and len(sessions) == 6 and len(set(sessions)) == 6,
                    "A2 within roster drift")
            observed = tuple(str(session) for session in sessions)
            if roster is None:
                roster = observed
            else:
                require(observed == roster, "A2 within roster differs across arms/seeds")
            _validate_fixed_epoch_means(payload, roster=observed, label=f"A2 within {arm}/s{seed}")
            cells[f"{arm}_s{seed}"] = {"payload": payload, "sha256": digest}
    require(roster is not None, "A2 within authority empty")
    return {"within_session_order": list(roster), "within_cells": cells}


def _load_manifest() -> dict[str, list[str]]:
    verified = _read_verified_bytes(STRICT_MANIFEST, "strict manifest",
                                    expected_sha256=EXPECTED_MANIFEST_SHA256)
    payload = json.loads(verified.raw.decode("utf-8"))
    splits = payload.get("session_splits") if isinstance(payload, Mapping) else None
    require(isinstance(splits, Mapping), "strict manifest malformed")
    result: dict[str, list[str]] = {}
    for key, count in (("train", 27), ("val", 6), ("test", 6)):
        rows = splits.get(key)
        require(isinstance(rows, list) and len(rows) == count and len(set(rows)) == count, "strict manifest split drift")
        result[key] = list(rows)
    require(len(set(result["train"] + result["val"] + result["test"])) == 39, "strict manifest overlap")
    return result


def _checkpoint_payload_audit(checkpoint: _VerifiedBytes) -> dict[str, Any]:
    """CPU/source-only topology check, deliberately local-imported Torch."""
    import torch

    payload = torch.load(io.BytesIO(checkpoint.raw), map_location="cpu", weights_only=True)
    require(isinstance(payload, Mapping), "B0 checkpoint payload malformed")
    hyper, state = payload.get("hyper_parameters"), payload.get("state_dict")
    require(isinstance(hyper, Mapping) and isinstance(state, Mapping), "B0 checkpoint fields missing")
    for key, expected in (("variant", "B0"), ("identity_mode", "calibrated"), ("loss_mode", "task_only"),
                          ("side_dim", 0), ("freeze_decoder", False), ("freeze_encoder_base", False)):
        require(hyper.get(key) == expected, f"B0 checkpoint hyperparameter drift: {key}")
    expected = {
        *(f"student.id_encoder.fc_id_in.{layer}.{part}" for layer in (0, 2, 4) for part in ("weight", "bias")),
        *(f"student.id_encoder.fc_id_out.{layer}.{part}" for layer in (0, 2, 4) for part in ("weight", "bias")),
    }
    actual = {str(key) for key in state if str(key).startswith("student.id_encoder.")}
    require(actual == expected, "B0 must be original fc_id_in/out copied encoder topology")
    decoder_count = sum(str(key).startswith("student.decoder.") for key in state)
    optimizers = payload.get("optimizer_states")
    require(decoder_count == 31 and isinstance(optimizers, list) and len(optimizers) == 1,
            "B0 decoder/optimizer topology drift")
    return {"identity_encoder_tensor_count": len(actual), "decoder_tensor_count": decoder_count,
            "topology": "BatchReferenceEncoder__trainable_copied_fc_id_in_out"}


def audit_b0_source_bundle(*, verify_checkpoint_payload: bool = False) -> dict[str, Any]:
    """Audit only archived source bytes; never resolve/open a target NWB."""
    manifest = _load_manifest()
    _read_verified_bytes(TEACHER, "teacher checkpoint", expected_sha256=EXPECTED_TEACHER_SHA256)
    source_rows: dict[str, Any] = {}
    for seed in SEEDS:
        run = b0_run_dir(seed)
        run_info = run.lstat()
        require(stat.S_ISDIR(run_info.st_mode) and not stat.S_ISLNK(run_info.st_mode),
                "B0 run directory missing or symlinked")
        metadata_verified = _read_verified_bytes(run / "run_metadata.json", f"B0 s{seed} metadata")
        metadata_path, metadata_raw = metadata_verified.path, metadata_verified.raw
        metadata = json.loads(metadata_raw.decode("utf-8"))
        require(isinstance(metadata, Mapping), "B0 metadata malformed")
        require(metadata.get("status") == "completed" and metadata.get("seed") == seed and metadata.get("variant") == "B0",
                "B0 metadata identity drift")
        require(metadata.get("teacher_sha256") == EXPECTED_TEACHER_SHA256 and
                metadata.get("train_val_manifest_sha256") == EXPECTED_MANIFEST_SHA256, "B0 teacher/manifest drift")
        require(metadata.get("session_splits") == manifest and metadata.get("held_out_test_evaluated") is False,
                "B0 source/test isolation drift")
        training = metadata.get("training")
        require(isinstance(training, Mapping), "B0 training metadata absent")
        for key, expected in (("max_epochs", 12), ("no_early_stopping", True), ("checkpoint_every_epoch", True),
                              ("calibration_n_trials", 30), ("window_size", 50), ("trial_length", 100),
                              ("loss_mode", "task_only"), ("identity_mode", "calibrated"),
                              ("freeze_decoder", False)):
            require(training.get(key) == expected, f"B0 schedule drift: {key}")
        side = metadata.get("side_features")
        require(isinstance(side, Mapping) and side.get("group") == "none" and side.get("side_dim") == 0,
                "B0 must not masquerade as a carrier arm")
        bundle: dict[str, str] = {}
        payload_audit: dict[str, Any] = {}
        source_checkpoint_mode: str | None = None
        for epoch in EPOCH_WINDOW:
            checkpoint = _read_verified_bytes(run / "epoch_ckpts" / f"epoch_{epoch - 1:03d}.ckpt",
                                              f"B0 s{seed} epoch {epoch}")
            bundle[str(epoch)] = checkpoint.sha256
            if source_checkpoint_mode is None:
                source_checkpoint_mode = format(checkpoint.mode, "04o")
            if verify_checkpoint_payload:
                payload_audit[str(epoch)] = _checkpoint_payload_audit(checkpoint)
        source_rows[str(seed)] = {
            "run_dir": str(_absolute_lexical(run)), "run_metadata_path": str(metadata_path),
            "run_metadata_sha256": metadata_verified.sha256,
            "source_checkpoint_sha256_bundle": bundle,
            "source_checkpoint_sha256_bundle_sha256": hashlib.sha256(canonical_json_bytes(bundle)).hexdigest(),
            "source_checkpoint_mode": source_checkpoint_mode,
            "checkpoint_payload_audits": payload_audit,
        }
    return {
        "source_runs": source_rows,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "teacher_sha256": EXPECTED_TEACHER_SHA256,
        "epoch_window": list(EPOCH_WINDOW),
        "source_topology": "original_SPINT_B0__BatchReferenceEncoder__trainable_copied_fc_id_in_out",
        "formal_subc_test_nwb_opened": False,
        "target_data_opened": False,
        "target_data_discovered": False,
    }


def load_sealed_a11_within_b0_authority(*, source_audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Verify A11's fixed B0 within-domain curve before reporting system shift.

    A11 remains a separate historical CPU-forward authority.  The comparison
    is allowed only after its B0 checkpoint SHA map, M30/query-after-30
    protocol, behavior-normalizer value SHA, six-session roster, and fixed
    logical 5--12 reduction all agree with this bridge's source lineage.
    """
    payload, digest = _strict_pair(A11_WITHIN_B0_PATH, label="A11 within B0 authority",
                                   expected_sha256=A11_WITHIN_B0_SHA256)
    require(payload.get("receipt_kind") == "scientific_full_curve" and
            payload.get("program_id") == "a11_b0_convergence_full_access_v1" and
            payload.get("status") == "completed" and payload.get("science_claim_allowed") is True,
            "A11 within B0 receipt identity/status drift")
    execution = payload.get("execution_contract")
    query_contract = payload.get("query_and_metric_contract")
    data_authority = payload.get("data_authority")
    source_configuration = payload.get("source_configuration_authority")
    cpu = payload.get("cpu_forward_result")
    convergence = payload.get("convergence_summary")
    require(isinstance(execution, Mapping) and execution.get("device") == "cpu" and
            execution.get("forward_only") is True and execution.get("backward_gradients") is False and
            execution.get("weight_updates") is False and execution.get("cuda_must_be_invisible") is True,
            "A11 within B0 execution boundary drift")
    require(isinstance(query_contract, Mapping) and
            query_contract.get("predeclared_epoch_window_human_epochs") == list(EPOCH_WINDOW) and
            query_contract.get("predeclared_epoch_window_checkpoint_names") ==
            [f"epoch_{epoch - 1:03d}.ckpt" for epoch in EPOCH_WINDOW],
            "A11 within B0 fixed epoch rule drift")
    contract = query_contract.get("query_contract") if isinstance(query_contract, Mapping) else None
    metric = contract.get("metric") if isinstance(contract, Mapping) else None
    require(isinstance(contract, Mapping) and contract.get("calibration_n_trials") == 30 and
            contract.get("excluded_prefix_pool_trials") == 30 and
            contract.get("query_trial_rule") == "usable_rewarded_trials[30:]" and
            contract.get("window_size_bins") == 50 and contract.get("max_trial_length_bins") == 100 and
            contract.get("signal_view") == "sua" and contract.get("trial_result_filter") == "R" and
            contract.get("query_is_disjoint_from_calibration") is True and
            isinstance(metric, Mapping) and metric == {
                "name": "torchmetrics.regression.R2Score", "multioutput": "variance_weighted",
                "prediction": "decoder_output_last_bin / 5.0", "target": "normalized_cursor_velocity_last_bin",
            }, "A11 within B0 query/metric semantics drift")
    normalizer = data_authority.get("normalizer") if isinstance(data_authority, Mapping) else None
    require(isinstance(normalizer, Mapping) and normalizer.get("sha256") == EXPECTED_BEHAVIOR_NORMALIZER_FILE_SHA256 and
            normalizer.get("array_contract_sha256") == EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256,
            "A11 within B0 normalizer authority drift")
    require(isinstance(cpu, Mapping) and cpu.get("normalizer_arrays_sha256") == EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256 and
            cpu.get("opened_nwb_path_count") == 6, "A11 within B0 CPU normalizer/open-count drift")
    sessions = data_authority.get("allowed_validation_sessions") if isinstance(data_authority, Mapping) else None
    query_windows = cpu.get("query_windows") if isinstance(cpu, Mapping) else None
    require(isinstance(sessions, list) and len(sessions) == 6 and len(set(sessions)) == 6 and
            isinstance(query_windows, Mapping) and tuple(query_windows) == tuple(sessions),
            "A11 within B0 six-session query roster drift")
    for session in sessions:
        row = query_windows[session]
        require(isinstance(row, Mapping) and row.get("name") == session and row.get("support_trial_count") == 30 and
                isinstance(row.get("query_trial_count"), int) and row["query_trial_count"] > 0 and
                isinstance(row.get("scored_window_count"), int) and row["scored_window_count"] > 0 and
                all(isinstance(row.get(key), str) and len(row[key]) == 64 for key in
                    ("support_trial_sha256", "query_trial_sha256", "scored_window_start_sha256")),
                "A11 within B0 query receipt drift")
    require(isinstance(source_configuration, Mapping) and source_configuration.get("variant") == "B0" and
            source_configuration.get("task") == "CO", "A11 within B0 source configuration drift")
    hparams = source_configuration.get("expected_hparams")
    require(isinstance(hparams, Mapping) and hparams.get("variant") == "B0" and
            hparams.get("identity_mode") == "calibrated" and hparams.get("side_dim") == 0 and
            hparams.get("loss_mode") == "task_only" and hparams.get("window_size") == 50 and
            hparams.get("trial_length") == 100, "A11 within B0 architecture/schedule drift")
    run_receipts = source_configuration.get("run_receipts")
    require(isinstance(run_receipts, Mapping) and set(run_receipts) == {str(seed) for seed in SEEDS},
            "A11 within B0 source run receipt set drift")
    audited = source_audit if source_audit is not None else audit_b0_source_bundle(verify_checkpoint_payload=False)
    source_rows = audited.get("source_runs") if isinstance(audited, Mapping) else None
    require(isinstance(source_rows, Mapping), "A11 within B0 bridge source audit missing")
    for seed in SEEDS:
        run = run_receipts[str(seed)]
        source_row = source_rows.get(str(seed))
        require(isinstance(run, Mapping) and isinstance(source_row, Mapping) and
                run.get("run_metadata_sha256") == source_row.get("run_metadata_sha256"),
                "A11 within B0 run metadata lineage drift")
        checkpoints = run.get("epoch_checkpoints")
        bundle = source_row.get("source_checkpoint_sha256_bundle")
        require(isinstance(checkpoints, Mapping) and isinstance(bundle, Mapping), "A11 within B0 checkpoint map absent")
        for epoch in EPOCH_WINDOW:
            row = checkpoints.get(f"epoch_{epoch - 1:03d}")
            require(isinstance(row, Mapping) and row.get("sha256") == bundle.get(str(epoch)),
                    "A11 within B0 checkpoint lineage drift")
    require(isinstance(convergence, Mapping) and convergence.get("epoch_5_to_12_average_supported") is True and
            isinstance(convergence.get("per_seed"), Mapping) and isinstance(cpu.get("per_seed"), Mapping),
            "A11 within B0 convergence receipt absent")
    per_seed_mean: dict[str, float] = {}
    for seed in SEEDS:
        summary = convergence["per_seed"].get(str(seed))
        cpu_seed = cpu["per_seed"].get(str(seed))
        per_epoch = cpu_seed.get("per_epoch") if isinstance(cpu_seed, Mapping) else None
        require(isinstance(summary, Mapping) and isinstance(per_epoch, Mapping) and
                set(per_epoch) == {str(epoch) for epoch in range(12)},
                "A11 within B0 full-curve receipt drift")
        values: list[float] = []
        for epoch in EPOCH_WINDOW:
            row = per_epoch[str(epoch - 1)]
            require(isinstance(row, Mapping) and isinstance(row.get("per_session_r2"), Mapping) and
                    tuple(row["per_session_r2"]) == tuple(sessions), "A11 within B0 per-epoch session map drift")
            epoch_mean = sum(float(row["per_session_r2"][session]) for session in sessions) / len(sessions)
            require(float(row.get("mean_r2")) == epoch_mean, "A11 within B0 per-epoch mean drift")
            values.append(float(row["mean_r2"]))
        # A11 fixes the aggregate as session means over logical 5..12, then
        # a six-session mean.  Averaging already-rounded per-epoch means is
        # algebraically equal but not bitwise identical in two seeds.
        session_means = [sum(float(per_epoch[str(epoch - 1)]["per_session_r2"][session]) for epoch in EPOCH_WINDOW)
                         / len(EPOCH_WINDOW) for session in sessions]
        mean = sum(session_means) / len(session_means)
        require(float(summary.get("epoch_5_to_12_mean_validation_r2")) == mean and
                summary.get("epoch_5_to_12_values") == values, "A11 within B0 per-seed epoch reduction drift")
        per_seed_mean[str(seed)] = mean
    overall = sum(per_seed_mean.values()) / len(per_seed_mean)
    require(float(convergence.get("three_seed_mean_of_epoch_5_to_12_scores")) == overall == A11_WITHIN_B0_MEAN_R2,
            "A11 within B0 three-seed mean drift")
    return {
        "path": str(_absolute_lexical(A11_WITHIN_B0_PATH)), "sha256": digest,
        "within_session_order": list(sessions), "per_seed_mean_r2": per_seed_mean, "mean_r2": overall,
        "query_contract_sha256": query_contract.get("query_contract_sha256"),
        "normalizer_value_sha256": EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256,
        "source_checkpoint_lineage_verified": True,
    }


def source_bundle_sha256(audit: Mapping[str, Any]) -> str:
    """Digest the execution-relevant source lineage, not optional CPU audit detail.

    ``verify_checkpoint_payload`` is an extra source-only topology check.  It
    cannot change the authorization identity of identical checkpoint bytes, so
    the root authorization remains valid whether its preflight printed those
    redundant per-epoch topology rows or the scorer rechecked them later.
    """
    normalized = dict(audit)
    rows = audit.get("source_runs")
    require(isinstance(rows, Mapping), "B0 source audit lacks source_runs")
    source_runs: dict[str, Any] = {}
    for seed, row in rows.items():
        require(isinstance(row, Mapping), "B0 source audit row malformed")
        source_runs[str(seed)] = {key: value for key, value in row.items() if key != "checkpoint_payload_audits"}
    normalized["source_runs"] = source_runs
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def build_preflight(*, output_root: Path = RESULT_ROOT, verify_checkpoint_payload: bool = False) -> dict[str, Any]:
    """CPU/source-only proof.  It creates no root and cannot discover target data."""
    output = _absolute_lexical(output_root)
    _assert_real_directory_chain(output.parent, "bridge output root")
    require(not os.path.lexists(output), "bridge output root must be fresh before preflight")
    a2 = load_sealed_a2_external_authority()
    target_byte_authority = load_subm_target_byte_authority(expected_session_order=a2["external_session_order"])
    audit = audit_b0_source_bundle(verify_checkpoint_payload=verify_checkpoint_payload)
    normalizer = _read_verified_bytes(BEHAVIOR_NORMALIZER, "source behavior normalizer",
                                      expected_sha256=EXPECTED_BEHAVIOR_NORMALIZER_FILE_SHA256)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "B0_EXTERNAL_SCORE_BRIDGE_PREFLIGHT_PASSED__NOT_AUTHORIZED_TO_OPEN_TARGET",
        "output_root": str(output),
        "a2_external_authority": {
            "terminal_aggregate_path": a2["terminal_aggregate_path"],
            "terminal_aggregate_sha256": a2["terminal_aggregate_sha256"],
            "external_cell_sha256": {key: row["sha256"] for key, row in a2["external_cells"].items()},
            "external_session_order": a2["external_session_order"],
        },
        "b0_source_audit": audit,
        "b0_source_audit_sha256": source_bundle_sha256(audit),
        "implementation_bindings": implementation_bindings(),
        "subm_target_byte_authority": target_byte_authority,
        "behavior_normalizer": {"path": str(normalizer.path), "file_sha256": normalizer.sha256,
                                  "value_sha256": EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256},
        "fixed_policy": {"seeds": list(SEEDS), "epoch_window": list(EPOCH_WINDOW),
                         "epoch_mapping": "logical epoch e maps to epoch_{e-1:03d}.ckpt",
                         "activity_calibration_trials": 30, "query_after_rewarded_trial": 30,
                         "normalizer_fit_scope": "strict_subc_source_train_27_only",
                         "target_normalizer_refit_forbidden": True, "target_backward_forbidden": True,
                         "target_decoder_update_forbidden": True, "target_epoch_selection_forbidden": True},
        "science_scope": {"t4_minus_b0_and_z4_minus_b0": "system contrasts, not carrier causal contrasts",
                          "carrier_interaction": "reuse only sealed A2 T4-minus-Z4 external-minus-within interaction",
                          "b0_consumes_target_direction_carrier": False},
        "target_data_opened": False, "target_data_discovered": False, "formal_subc_test_nwb_opened": False,
        "gpu_used": False, "cebra_imported": False, "score_emitted": False,
    }


def load_immutable_preflight(path: Path) -> tuple[dict[str, Any], str]:
    """Load root's immutable bridge preflight for a later score cell.

    A score cell must not rebuild the initial freshness proof after another
    fixed seed has written its own output.  Instead it consumes the exact
    root-reviewed immutable preflight body that the authorization binds.
    """
    payload, digest = _strict_pair(path, label="B0 bridge immutable preflight")
    require(payload.get("schema") == PREFLIGHT_SCHEMA and
            payload.get("status") == "B0_EXTERNAL_SCORE_BRIDGE_PREFLIGHT_PASSED__NOT_AUTHORIZED_TO_OPEN_TARGET",
            "B0 bridge preflight schema/status drift")
    require(isinstance(payload.get("b0_source_audit_sha256"), str), "B0 bridge preflight source audit missing")
    verify_implementation_bindings(payload.get("implementation_bindings"))
    target_byte_authority = payload.get("subm_target_byte_authority")
    require(isinstance(target_byte_authority, Mapping) and
            canonical_json_bytes(target_byte_authority) == canonical_json_bytes(load_subm_target_byte_authority()),
            "B0 bridge immutable preflight target byte authority drift")
    policy = payload.get("fixed_policy")
    require(isinstance(policy, Mapping) and policy.get("epoch_window") == list(EPOCH_WINDOW) and
            policy.get("activity_calibration_trials") == 30 and policy.get("query_after_rewarded_trial") == 30,
            "B0 bridge immutable preflight policy drift")
    return payload, digest


def _authorization_payload(path: Path, *, preflight: Mapping[str, Any], output_path: Path, seed: int) -> tuple[dict[str, Any], str]:
    payload, digest = _strict_pair(path, label="root B0 bridge authorization")
    expected_keys = {
        "schema", "root_authorized", "preflight_path", "preflight_sha256", "b0_source_audit_sha256",
        "output_root", "allowed_seeds", "cpu_only", "gpu_permitted", "target_updates_permitted",
        "target_epoch_selection_permitted", "scientific_scope",
    }
    require(set(payload) == expected_keys and payload.get("schema") == ROOT_AUTHORIZATION_SCHEMA and
            payload.get("root_authorized") is True, "B0 bridge root authorization schema/status drift")
    require(payload.get("preflight_sha256") == hashlib.sha256(canonical_json_bytes(preflight)).hexdigest(),
            "root B0 bridge authorization preflight binding drift")
    require(payload.get("b0_source_audit_sha256") == preflight.get("b0_source_audit_sha256"),
            "root B0 bridge authorization source bundle binding drift")
    require(payload.get("preflight_path") == preflight.get("immutable_preflight_path"),
            "root B0 bridge authorization preflight path binding drift")
    output_parent = _absolute_lexical(output_path).parent
    _assert_real_directory_chain(output_parent, "B0 score output")
    require(payload.get("output_root") == preflight.get("output_root") == str(output_parent),
            "root B0 bridge authorization output-root binding drift")
    require(payload.get("allowed_seeds") == list(SEEDS) and seed in payload["allowed_seeds"],
            "root B0 bridge authorization seed policy drift")
    require(payload.get("cpu_only") is True and payload.get("gpu_permitted") is False and
            payload.get("target_updates_permitted") is False and payload.get("target_epoch_selection_permitted") is False,
            "root B0 bridge authorization execution policy drift")
    return payload, digest


def _require_fresh_score_output_pair(path: Path) -> tuple[Path, Path]:
    """Reject a prior body *or* sidecar before any target pathname is requested."""
    body = _absolute_lexical(path)
    _assert_real_directory_chain(body.parent, "B0 score output")
    sidecar = body.with_name(f"{body.name}.sha256")
    require(not os.path.lexists(body) and not os.path.lexists(sidecar),
            "B0 score output body or sidecar already exists")
    return body, sidecar


@contextmanager
def _private_verified_snapshots(named_inputs: Mapping[str, _VerifiedBytes]):
    """Offer pathname-only third-party loaders only immutable copies of verified bytes.

    Torch's canonical archived B0 loader and Lightning's nested teacher loader
    accept paths and reopen them internally.  The original source files are
    mode 0664/0600, so a hash before that reopen would not establish what was
    consumed.  This context creates fresh O_EXCL, 0400 copies in a private
    directory directly from the same verified bytes, re-verifies them before
    and after the third-party loader returns, and removes only that private
    directory on exit.  It is intentionally not a persistent execution
    authority or an output receipt.
    """
    require(set(named_inputs) == {"b0_checkpoint", "teacher_checkpoint"},
            "B0 private snapshot requires exactly B0 and teacher inputs")
    with tempfile.TemporaryDirectory(prefix="subm_b0_verified_inputs_") as raw_directory:
        directory = Path(raw_directory)
        directory_info = directory.lstat()
        require(stat.S_ISDIR(directory_info.st_mode) and not stat.S_ISLNK(directory_info.st_mode) and
                stat.S_IMODE(directory_info.st_mode) == 0o700,
                "B0 private snapshot directory is not private and real")
        snapshots: dict[str, Path] = {}
        for name, verified in named_inputs.items():
            filename = "b0_checkpoint.ckpt" if name == "b0_checkpoint" else "teacher_checkpoint.ckpt"
            snapshot = directory / filename
            identity = _write_exclusive_regular(snapshot, verified.raw, mode=0o400)
            _fsync_directory(directory)
            snap_verified = _read_verified_bytes(snapshot, f"B0 private {name} snapshot",
                                                expected_sha256=verified.sha256)
            require(snap_verified.raw == verified.raw and snap_verified.identity == identity and snap_verified.mode == 0o400,
                    f"B0 private {name} snapshot byte/mode drift")
            snapshots[name] = snapshot
        try:
            yield snapshots
        finally:
            for name, verified in named_inputs.items():
                snap_verified = _read_verified_bytes(snapshots[name], f"B0 private {name} snapshot after loader",
                                                    expected_sha256=verified.sha256)
                require(snap_verified.raw == verified.raw and snap_verified.mode == 0o400,
                        f"B0 private {name} snapshot changed while third-party loader ran")


def _load_b0_model_from_verified_source_bytes(
    checkpoint: _VerifiedBytes,
    teacher: _VerifiedBytes,
    *,
    device: Any,
) -> tuple[Any, dict[str, str]]:
    """Use the exact archived loader, but only against verified private snapshots."""
    _ensure_archived_import_roots()
    from scripts.select_gradient_free_protocol_dandi688 import load_frozen_model

    require(checkpoint.sha256 and teacher.sha256, "B0 verified model inputs lack SHA")
    with _private_verified_snapshots({"b0_checkpoint": checkpoint, "teacher_checkpoint": teacher}) as snapshots:
        model = load_frozen_model(snapshots["b0_checkpoint"], snapshots["teacher_checkpoint"], "B0", device,
                                  identity_mode="calibrated")
    return model, {
        "checkpoint_sha256": checkpoint.sha256,
        "teacher_sha256": teacher.sha256,
        "checkpoint_consumption": "same_verified_bytes_via_private_0400_snapshot__canonical_loader",
        "teacher_consumption": "same_verified_bytes_via_private_0400_snapshot__StreamingCalibrationLitModule_setup",
    }


def _ensure_archived_import_roots() -> None:
    """Make historical top-level archived imports available without edits."""
    # The archived loader imports both ``scripts.*``/``mc_maze.*`` and sibling
    # script modules such as ``dandi688_gradient_free_protocol``.  A command
    # run as a script has these entries incidentally; an imported bridge does
    # not.  Add only these fixed source roots, never a data directory.
    for root in (SUA_ROOT, SUA_ROOT / "scripts"):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def _load_behavior_stats(normalizer: _VerifiedBytes) -> tuple[Any, Any]:
    import numpy as np
    require(normalizer.sha256 == EXPECTED_BEHAVIOR_NORMALIZER_FILE_SHA256,
            "B0 behavior normalizer file SHA drift")
    with np.load(io.BytesIO(normalizer.raw), allow_pickle=False) as data:
        mean, std = data["mean"].copy(), data["std"].copy()
    # Import the canonical A2 semantic digest only after the source-only pin.
    from mc_maze import a2_matched_subject_shift_v2_core as a2core
    require(a2core.normalizer_value_sha256(mean, std) == EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256,
            "B0 behavior normalizer semantic SHA drift")
    return mean, std


def _require_model_state_parity(first: Any, second: Any) -> None:
    first_state, second_state = first.state_dict(), second.state_dict()
    require(tuple(first_state) == tuple(second_state), "B0 canonical/snapshot state key drift")
    import torch
    for key in first_state:
        require(torch.equal(first_state[key].detach().cpu(), second_state[key].detach().cpu()),
                f"B0 canonical/snapshot state value drift: {key}")


def source_only_checkpoint_loader_smoke(
    *, seed: int = 42, logical_epoch: int = 5, verify_canonical_loader_state_parity: bool = True,
) -> dict[str, Any]:
    """Load one archived B0 checkpoint on CPU without discovering target data.

    This is an explicit source-only operational smoke for the same-byte
    snapshot route.  It does not score a session, instantiate a datamodule,
    or mint any bridge artifact.
    """
    require(seed in SEEDS and logical_epoch in EPOCH_WINDOW, "unsupported B0 source smoke cell")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "B0 source smoke requires CUDA_VISIBLE_DEVICES empty")
    import torch

    teacher = _read_verified_bytes(TEACHER, "B0 source-smoke teacher",
                                   expected_sha256=EXPECTED_TEACHER_SHA256)
    checkpoint_path = b0_run_dir(seed) / "epoch_ckpts" / f"epoch_{logical_epoch - 1:03d}.ckpt"
    checkpoint = _read_verified_bytes(checkpoint_path, f"B0 source-smoke s{seed} epoch {logical_epoch}")
    _checkpoint_payload_audit(checkpoint)
    model, consumption = _load_b0_model_from_verified_source_bytes(checkpoint, teacher, device=torch.device("cpu"))
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    require(parameter_count > 0 and model.training is False, "B0 source smoke model was not frozen/eval")
    state_parity: bool | None = None
    if verify_canonical_loader_state_parity:
        _ensure_archived_import_roots()
        from scripts.select_gradient_free_protocol_dandi688 import load_frozen_model
        canonical_model = load_frozen_model(checkpoint.path, teacher.path, "B0", torch.device("cpu"),
                                            identity_mode="calibrated")
        _require_model_state_parity(model, canonical_model)
        # The parity comparison is a source-only test of the exact original
        # loader.  The actual score path never consumes the mutable archive
        # names after verification; it uses the private snapshots above.
        require(_read_verified_bytes(checkpoint.path, "B0 source-smoke checkpoint after canonical parity",
                                     expected_sha256=checkpoint.sha256).raw == checkpoint.raw,
                "B0 checkpoint changed during canonical parity smoke")
        require(_read_verified_bytes(teacher.path, "B0 source-smoke teacher after canonical parity",
                                     expected_sha256=teacher.sha256).raw == teacher.raw,
                "B0 teacher changed during canonical parity smoke")
        state_parity = True
    return {
        "status": "B0_SOURCE_ONLY_CHECKPOINT_LOADER_SMOKE_PASSED",
        "seed": seed,
        "logical_epoch": logical_epoch,
        "checkpoint_sha256": checkpoint.sha256,
        "teacher_sha256": teacher.sha256,
        "model_parameter_count": parameter_count,
        "model_training": bool(model.training),
        "canonical_loader_state_parity": state_parity,
        "consumption_authority": consumption,
        "target_data_opened": False,
        "target_data_discovered": False,
        "formal_subc_test_nwb_opened": False,
        "gpu_used": False,
        "score_emitted": False,
    }


def execute_cpu_score(seed: int, *, output_path: Path, preflight: Mapping[str, Any], authorization_path: Path) -> dict[str, Any]:
    """Score all fixed B0 epochs on the frozen external M30 query roster.

    This is intentionally not invoked by preflight or tests.  It must receive
    a root-minted immutable authorization pair and runs CPU-only/no-grad.
    """
    require(seed in SEEDS, "unsupported B0 seed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "B0 bridge requires CUDA_VISIBLE_DEVICES empty")
    _require_fresh_score_output_pair(output_path)
    _authorization, authorization_sha = _authorization_payload(authorization_path, preflight=preflight,
                                                                output_path=output_path, seed=seed)
    active_implementation_bindings = verify_implementation_bindings(preflight.get("implementation_bindings"))
    a2 = load_sealed_a2_external_authority()
    target_byte_authority = load_subm_target_byte_authority(expected_session_order=a2["external_session_order"])
    preflight_target_authority = preflight.get("subm_target_byte_authority")
    require(isinstance(preflight_target_authority, Mapping) and
            canonical_json_bytes(target_byte_authority) == canonical_json_bytes(preflight_target_authority),
            "B0 target byte authority drift since immutable preflight")
    fresh_audit = audit_b0_source_bundle(verify_checkpoint_payload=True)
    require(source_bundle_sha256(fresh_audit) == preflight.get("b0_source_audit_sha256"), "B0 source bytes drift since preflight")

    _ensure_archived_import_roots()
    import torch
    from mc_maze import a2_matched_subject_shift_v2_core as a2core
    from mc_maze.multisession_datamodule import session_name_from_path
    from scripts.eval_adaptation_dandi688 import PAD_VALUE, TRIAL_LENGTH, WINDOW_SIZE, build_calib_trials_for_indices, eval_r2, load_session_with_trials, make_subset_dataset

    require(WINDOW_SIZE == 50 and TRIAL_LENGTH == 100, "shared evaluation constants drift")
    normalizer = _read_verified_bytes(BEHAVIOR_NORMALIZER, "B0 source behavior normalizer before target",
                                      expected_sha256=EXPECTED_BEHAVIOR_NORMALIZER_FILE_SHA256)
    behavior_mean, behavior_std = _load_behavior_stats(normalizer)
    teacher = _read_verified_bytes(TEACHER, "B0 source teacher before target", expected_sha256=EXPECTED_TEACHER_SHA256)
    paths = a2core.external_session_paths()
    roster = tuple(a2["external_session_order"])
    require(tuple(session_name_from_path(path) for path in paths) == roster, "frozen external roster path/order drift")
    records: list[tuple[str, Any, dict[str, Any]]] = []
    target_byte_receipts: dict[str, dict[str, Any]] = {}
    for path in paths:
        name = session_name_from_path(path)
        before_loader = _verify_target_nwb_before_or_after_loader(
            path, session=name, target_authority=target_byte_authority, phase="before_nwb_loader",
        )
        record = load_session_with_trials(path, 20, 50, 30, 100, PAD_VALUE, behavior_mean, behavior_std,
                                          trial_result_filter="R", cache_dir=None, signal_view="sua")
        after_loader = _verify_target_nwb_before_or_after_loader(
            path, session=name, target_authority=target_byte_authority, phase="after_nwb_loader",
        )
        require(before_loader["actual_sha256"] == after_loader["actual_sha256"] and
                before_loader["actual_bytes"] == after_loader["actual_bytes"] and
                before_loader["identity_sha256"] == after_loader["identity_sha256"],
                f"target NWB changed during loader access: {name}")
        target_byte_receipts[name] = {
            **target_byte_authority["sessions"][name],
            "before_nwb_loader": before_loader,
            "after_nwb_loader": after_loader,
            "same_bytes_and_identity_before_after_loader": True,
        }
        require(record["name"] == name, "B0 target record identity drift")
        semantics = a2core.trial30_semantics_from_trials(record["trials"], require_target_labels=False, session=name)
        ref_projection = _query_projection(a2["external_cells"][f"t4_s{seed}"]["payload"]["session_query_receipts"][name])
        require(_query_projection({**semantics, "dataset_query_window_count": semantics["post30_query_window_count"]}) == ref_projection,
                "B0 M30/query row projection differs from sealed A2 reference")
        record["calib_trials"] = build_calib_trials_for_indices(record, list(range(30)), 30)
        dataset = make_subset_dataset(record, record["trials"][30:], name)
        require(len(dataset) == semantics["post30_query_window_count"] > 0, "B0 query dataset count drift")
        records.append((name, dataset, {**semantics, "dataset_query_window_count": len(dataset)}))
    per_epoch: dict[str, Any] = {}
    device = torch.device("cpu")
    with torch.no_grad():
        for epoch in EPOCH_WINDOW:
            checkpoint_path = b0_run_dir(seed) / "epoch_ckpts" / f"epoch_{epoch - 1:03d}.ckpt"
            checkpoint = _read_verified_bytes(checkpoint_path, f"B0 score s{seed} epoch {epoch}",
                                              expected_sha256=fresh_audit["source_runs"][str(seed)]["source_checkpoint_sha256_bundle"][str(epoch)])
            model, consumption = _load_b0_model_from_verified_source_bytes(checkpoint, teacher, device=device)
            values = {name: float(eval_r2(model, dataset, device)) for name, dataset, _trace in records}
            per_epoch[str(epoch)] = {"checkpoint_path": str(checkpoint.path),
                                     "checkpoint_sha256": checkpoint.sha256,
                                     "input_consumption_authority": consumption,
                                     "per_session_r2": values, "mean_r2": sum(values.values()) / len(values)}
    mean_by_session = {name: sum(per_epoch[str(epoch)]["per_session_r2"][name] for epoch in EPOCH_WINDOW) / len(EPOCH_WINDOW)
                       for name in roster}
    post_audit = audit_b0_source_bundle(verify_checkpoint_payload=False)
    require(source_bundle_sha256(post_audit) == preflight.get("b0_source_audit_sha256"),
            "B0 source bytes drifted during CPU score")
    payload = {
        "schema": SCORE_SCHEMA, "status": "B0_EXTERNAL_CPU_SCORE_COMPLETE__FIXED_EPOCH_AVERAGE",
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "domain": EXTERNAL_DOMAIN,
        "variant": "B0", "source_topology": fresh_audit["source_topology"],
        "root_authorization_sha256": authorization_sha, "preflight_sha256": hashlib.sha256(canonical_json_bytes(preflight)).hexdigest(),
        "b0_source_audit_sha256": source_bundle_sha256(fresh_audit),
        "b0_source_audit_sha256_after_score": source_bundle_sha256(post_audit),
        "implementation_bindings": active_implementation_bindings,
        "source_checkpoint_sha256_bundle": fresh_audit["source_runs"][str(seed)]["source_checkpoint_sha256_bundle"],
        "target_nwb_byte_authority": {
            **{key: target_byte_authority[key] for key in (
                "a2_official_preflight_path", "a2_official_preflight_sha256", "schema_ledger_path",
                "schema_ledger_sha256", "scope_manifest_path", "scope_manifest_sha256", "external_session_order",
            )},
            "session_bytes": target_byte_receipts,
        },
        "domain_sessions": list(roster), "per_epoch": per_epoch, "per_session_mean_r2": mean_by_session,
        "mean_r2": sum(mean_by_session.values()) / len(mean_by_session),
        "session_query_receipts": {name: {**trace, "activity_calibration_trial_indices": list(range(30)),
             "target_session_carrier_fit_performed": False, "target_direction_labels_used_for_carrier": False,
             "target_velocity_labels_used_for_weight_updates": False, "backward_gradients": False,
             "decoder_weight_updates": False} for name, _dataset, trace in records},
        "normalizer_authority": {"behavior_normalizer_value_sha256": EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256,
             "target_domain_normalizer_refit_performed": False, "target_domain_normalizer_refit_forbidden": True,
             "side_normalizer": "not_applicable__B0_side_dim_0"},
        "target_data_opened": True, "formal_subc_test_nwb_opened": False, "gpu_used": False,
        "target_updates": False, "checkpoint_selection": "fixed unweighted mean of logical epochs 5..12; no target selection",
        "science_scope": "B0 has no target carrier; comparison to A2 B3S systems is not carrier-causal",
    }
    _write_pair_once(output_path, payload)
    return payload


def _validate_b0_score(payload: Mapping[str, Any], *, seed: int, a2: Mapping[str, Any]) -> None:
    require(payload.get("schema") == SCORE_SCHEMA and payload.get("seed") == seed and payload.get("domain") == EXTERNAL_DOMAIN,
            "B0 score identity/schema drift")
    require(payload.get("variant") == "B0" and payload.get("source_topology") ==
            "original_SPINT_B0__BatchReferenceEncoder__trainable_copied_fc_id_in_out", "B0 topology label drift")
    verify_implementation_bindings(payload.get("implementation_bindings"))
    roster = tuple(a2["external_session_order"])
    require(tuple(payload.get("domain_sessions") or ()) == roster, "B0 score roster drift")
    require(payload.get("target_updates") is False and payload.get("gpu_used") is False and
            payload.get("formal_subc_test_nwb_opened") is False, "B0 score boundary drift")
    require(payload.get("normalizer_authority", {}).get("behavior_normalizer_value_sha256") == EXPECTED_BEHAVIOR_NORMALIZER_VALUE_SHA256,
            "B0 score behavior normalizer drift")
    per_epoch = payload.get("per_epoch")
    require(isinstance(per_epoch, Mapping) and set(per_epoch) == {str(x) for x in EPOCH_WINDOW}, "B0 score epoch set drift")
    queries = payload.get("session_query_receipts")
    means = payload.get("per_session_mean_r2")
    require(isinstance(queries, Mapping) and isinstance(means, Mapping), "B0 score arrays missing")
    target_byte_authority = payload.get("target_nwb_byte_authority")
    expected_target_authority = load_subm_target_byte_authority(expected_session_order=roster)
    require(isinstance(target_byte_authority, Mapping) and
            target_byte_authority.get("a2_official_preflight_sha256") == A2_OFFICIAL_PREFLIGHT_SHA256 and
            target_byte_authority.get("schema_ledger_sha256") == SUBM_SCHEMA_LEDGER_SHA256 and
            target_byte_authority.get("scope_manifest_sha256") == SUBM_SCOPE_MANIFEST_SHA256 and
            tuple(target_byte_authority.get("external_session_order") or ()) == roster,
            "B0 score target byte authority header drift")
    byte_rows = target_byte_authority.get("session_bytes")
    require(isinstance(byte_rows, Mapping) and tuple(byte_rows) == roster, "B0 score target byte receipt roster drift")
    for name in roster:
        require(_query_projection(queries[name]) == _query_projection(a2["external_cells"][f"t4_s{seed}"]["payload"]["session_query_receipts"][name]),
                "B0 query projection differs from A2")
        recomputed = sum(float(per_epoch[str(epoch)]["per_session_r2"][name]) for epoch in EPOCH_WINDOW) / len(EPOCH_WINDOW)
        require(float(means[name]) == recomputed, "B0 session epoch averaging drift")
        byte_row = byte_rows[name]
        expected = expected_target_authority["sessions"][name]
        require(isinstance(byte_row, Mapping) and
                {key: byte_row.get(key) for key in expected} == expected and
                byte_row.get("same_bytes_and_identity_before_after_loader") is True,
                "B0 target byte receipt expected pin drift")
        before, after = byte_row.get("before_nwb_loader"), byte_row.get("after_nwb_loader")
        require(isinstance(before, Mapping) and isinstance(after, Mapping) and
                before.get("phase") == "before_nwb_loader" and after.get("phase") == "after_nwb_loader" and
                before.get("actual_sha256") == after.get("actual_sha256") == expected["expected_sha256"] and
                before.get("actual_bytes") == after.get("actual_bytes") == expected["expected_bytes"] and
                isinstance(before.get("identity"), Mapping) and before.get("identity") == after.get("identity") and
                isinstance(before.get("identity_sha256"), str) and before.get("identity_sha256") == after.get("identity_sha256") and
                len(before["identity_sha256"]) == 64,
                "B0 target byte receipt before/after loader drift")
    require(isinstance(payload.get("mean_r2"), (int, float)) and
            float(payload["mean_r2"]) == sum(float(means[name]) for name in roster) / len(roster),
            "B0 external overall mean drift")


def aggregate_payload(*, b0_score_pairs: Mapping[int, Path]) -> dict[str, Any]:
    """Aggregate fixed external contrasts and a separately disclosed system shift."""
    require(set(b0_score_pairs) == set(SEEDS), "B0 aggregate requires all three fixed seeds")
    active_implementation_bindings = implementation_bindings()
    verify_implementation_bindings(active_implementation_bindings)
    a2 = load_sealed_a2_external_authority()
    a2_within = load_sealed_a2_within_authority()
    b0_source_audit = audit_b0_source_bundle(verify_checkpoint_payload=False)
    a11_within_b0 = load_sealed_a11_within_b0_authority(source_audit=b0_source_audit)
    b0: dict[int, tuple[dict[str, Any], str]] = {}
    for seed in SEEDS:
        payload, digest = _strict_pair(b0_score_pairs[seed], label=f"B0 external score s{seed}")
        _validate_b0_score(payload, seed=seed, a2=a2)
        b0[seed] = (payload, digest)
    roster = tuple(a2["external_session_order"])
    contrasts: dict[str, Any] = {}
    b0_external_per_seed = {str(seed): float(b0[seed][0]["mean_r2"]) for seed in SEEDS}
    b0_external_mean = sum(b0_external_per_seed.values()) / len(b0_external_per_seed)
    for arm in ("t4", "z4"):
        per_seed: dict[str, float] = {}
        per_session_values: dict[str, list[float]] = {name: [] for name in roster}
        for seed in SEEDS:
            a2_scores = a2["external_cells"][f"{arm}_s{seed}"]["payload"]["per_session_mean_r2"]
            b0_scores = b0[seed][0]["per_session_mean_r2"]
            deltas = {name: float(a2_scores[name]) - float(b0_scores[name]) for name in roster}
            per_seed[str(seed)] = sum(deltas.values()) / len(deltas)
            for name in roster:
                per_session_values[name].append(deltas[name])
        contrasts[f"{arm}_minus_b0"] = {
            "definition": f"A2 external B3S {arm} minus original-SPINT B0, paired by seed and external session",
            "mean_paired_delta_r2": sum(per_seed.values()) / len(per_seed),
            "per_seed_mean_delta_r2": per_seed,
            "per_session_mean_delta_r2": {name: sum(values) / len(values) for name, values in per_session_values.items()},
            "interpretation_limit": "system contrast only: changes B3S-vs-B0 topology/training lineage and target carrier use; not a carrier causal effect",
        }
    system_shift_interactions: dict[str, Any] = {}
    for arm in ("t4", "z4"):
        per_seed: dict[str, float] = {}
        external_contrast_per_seed: dict[str, float] = {}
        within_contrast_per_seed: dict[str, float] = {}
        for seed in SEEDS:
            external_a2 = float(a2["external_cells"][f"{arm}_s{seed}"]["payload"]["mean_r2"])
            within_a2 = float(a2_within["within_cells"][f"{arm}_s{seed}"]["payload"]["mean_r2"])
            external_b0 = b0_external_per_seed[str(seed)]
            within_b0 = float(a11_within_b0["per_seed_mean_r2"][str(seed)])
            external_contrast_per_seed[str(seed)] = external_a2 - external_b0
            within_contrast_per_seed[str(seed)] = within_a2 - within_b0
            per_seed[str(seed)] = external_contrast_per_seed[str(seed)] - within_contrast_per_seed[str(seed)]
        system_shift_interactions[f"{arm}_minus_b0_external_minus_within"] = {
            "definition": f"(A2 {arm} minus B0)_external_subject_M minus (A2 {arm} minus B0)_within_subject",
            "per_seed_system_shift_interaction_r2": per_seed,
            "mean_system_shift_interaction_r2": sum(per_seed.values()) / len(per_seed),
            "external_system_contrast_per_seed_r2": external_contrast_per_seed,
            "within_system_contrast_per_seed_r2": within_contrast_per_seed,
            "interpretation_limit": "system-shift interaction only: B3S-vs-original-SPINT differences include topology, training lineage, and target carrier use; it is not a carrier causal interaction",
        }
    b0_shift_per_seed = {
        str(seed): b0_external_per_seed[str(seed)] - float(a11_within_b0["per_seed_mean_r2"][str(seed)])
        for seed in SEEDS
    }
    return {
        "schema": AGGREGATE_SCHEMA, "status": "B0_EXTERNAL_SYSTEM_CONTRASTS_COMPLETE__NO_POSTHOC_SELECTION",
        "a2_external_authority": {"terminal_aggregate_path": a2["terminal_aggregate_path"],
                                  "terminal_aggregate_sha256": a2["terminal_aggregate_sha256"],
                                  "external_cell_sha256": {key: row["sha256"] for key, row in a2["external_cells"].items()}},
        "b0_external_score_sha256": {str(seed): digest for seed, (_payload, digest) in b0.items()},
        "implementation_bindings": active_implementation_bindings,
        "a2_within_authority": {"within_cell_sha256": {key: row["sha256"] for key, row in a2_within["within_cells"].items()},
                                "within_session_order": a2_within["within_session_order"]},
        "a11_within_b0_authority": a11_within_b0,
        "fixed_epoch_window": list(EPOCH_WINDOW), "external_session_order": list(roster),
        "b0_external_summary": {"per_seed_mean_r2": b0_external_per_seed, "mean_r2": b0_external_mean},
        "b0_external_minus_within_subject_shift": {
            "definition": "B0 external_subject_M mean R2 minus A11 B0 within_subject mean R2, paired only by fixed seed",
            "per_seed_shift_r2": b0_shift_per_seed,
            "mean_shift_r2": sum(b0_shift_per_seed.values()) / len(b0_shift_per_seed),
            "interpretation_limit": "domain/system shift summary across different session rosters; not a session-paired or carrier-causal effect",
        },
        "contrasts": contrasts,
        "system_shift_interactions": system_shift_interactions,
        "sealed_cross_domain_carrier_interaction": {"definition": "A2 T4-minus-Z4 external-minus-within interaction; B0 is not included",
            "terminal_aggregate_sha256": a2["terminal_aggregate_sha256"], "interaction": a2["interaction"]},
        "science_scope": "B0 is the original-SPINT identity architecture under the matched streaming source protocol, not a published-SPINT system/checkpoint claim. T4/B0 and Z4/B0, including their external-minus-within system-shift interactions, are non-causal system comparisons and not carrier causal effects. The sealed A2 carrier interaction is reported separately and unchanged.",
        "target_data_opened_by_aggregator": False, "formal_subc_test_nwb_opened": False,
    }


__all__ = [
    "A11_WITHIN_B0_MEAN_R2", "A11_WITHIN_B0_SHA256", "A2_EXTERNAL_RECEIPT_SHA256", "A2_TERMINAL_SHA256", "B0BridgeError", "EPOCH_WINDOW", "RESULT_ROOT",
    "SEEDS", "aggregate_payload", "audit_b0_source_bundle", "b0_score_path", "build_preflight",
    "canonical_json_bytes", "execute_cpu_score", "implementation_bindings", "load_immutable_preflight", "load_sealed_a11_within_b0_authority", "load_sealed_a2_external_authority", "load_sealed_a2_within_authority", "sha256_file", "source_bundle_sha256", "source_only_checkpoint_loader_smoke", "verify_implementation_bindings",
    "_strict_pair", "_write_pair_once",
]
