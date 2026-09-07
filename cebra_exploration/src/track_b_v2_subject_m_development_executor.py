"""Fail-closed subject-M Track-B v2 live-executor/scorer *skeleton*.

This is intentionally an additive, no-target contract.  It does not import a
target/NWB/NPZ loader, CEBRA, Torch, sklearn, or a scorer; it does not construct
an encoder, fit a readout, use a GPU, write a receipt, or mint authority.  The
sole public operation renders a dry plan for a future root-authorised subject-M
executor.  A real implementation must be a separately reviewed successor.

The dry plan consumes the canonical development materializer (which in turn
rebuilds its canonical pointer/source authority), then hard-gates any future
target operation on the sole d8/it250 fixed-GPU engineering receipt.  RT is
not accepted here: its local target-byte ledger remains a separate NO-GO.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterator, Mapping

import track_b_v2_contract as base
import track_b_v2_development_target_materializer as materializer
import track_b_v2_fixed_gpu_engineering as fixed_gpu


SUBJECT_M_DEVELOPMENT_EXECUTOR_DRY_PLAN_SCHEMA = "track_b_v2_subject_m_development_executor_dry_plan_v1"
FIXED_GPU_COST_GATE_SCHEMA = "track_b_v2_subject_m_fixed_gpu_cost_gate_v1"
SUBJECT_M_EXECUTION_RECEIPT_TOPOLOGY_SCHEMA = "track_b_v2_subject_m_future_execution_receipt_topology_v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXED_GPU_COST_RUNNER = _REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_fixed_gpu_source_cost.py"
_CEBRA_EXECUTION_SEEDS = (42, 43, 44)
_DECODERS = ("linear_ridge", "knn_cosine_k3")
_READOUT_ROUTES = (
    "source_only_consumer_mechanism_alignment",
    "target_support_only_standard_cebra_accuracy",
    "source_plus_target_support_hybrid_sensitivity",
)
_MODEL_ARMS = ("cebra_joint_behavior", "cebra_frozen_source_adapt", "cebra_adapt_unaligned")


class TrackBV2SubjectMDevelopmentExecutorError(materializer.TrackBV2DevelopmentTargetMaterializerError):
    """Raised before this skeleton could reach a target or score operation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2SubjectMDevelopmentExecutorError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _absolute_lexical(path: Path) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path(os.path.abspath(str(expanded)))


def _assert_real_directory_chain(directory: Path, *, label: str) -> Path:
    directory = _absolute_lexical(directory)
    current = Path(directory.anchor)
    for part in directory.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            raise TrackBV2SubjectMDevelopmentExecutorError(
                f"{label} parent is absent: {current}"
            ) from exc
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                f"{label} parent must be a real directory: {current}")
    return directory


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mode, info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True)
class _VerifiedImmutable:
    path: Path
    raw: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]


@dataclass
class ParserBoundVerifiedTargetSnapshot:
    """A private target snapshot whose descriptor, not pathname, reaches a parser.

    This deliberately keeps the parser descriptor open.  A future NWB/HDF5
    adapter must consume :attr:`parser_fd_path` (``/proc/self/fd/<n>``), or an
    equivalent explicit descriptor API, while the context is live.  It must
    never replace that proof with a normal reopen of ``source_path`` or
    ``snapshot_path``.  This exists here as a generic byte boundary only; it
    performs no NWB parsing and is not called by the dry-plan CLI.
    """

    source_path: Path
    source_sha256: str
    source_byte_count: int
    source_identity: tuple[int, int, int, int, int, int]
    snapshot_path: Path
    snapshot_sha256: str
    snapshot_byte_count: int
    snapshot_identity: tuple[int, int, int, int, int, int]
    parser_fd: int

    @property
    def parser_fd_path(self) -> str:
        return f"/proc/self/fd/{self.parser_fd}"

    def as_contract_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "source_byte_count": self.source_byte_count,
            "source_identity_sha256": _sha_json(self.source_identity),
            "private_snapshot_path": str(self.snapshot_path),
            "private_snapshot_sha256": self.snapshot_sha256,
            "private_snapshot_byte_count": self.snapshot_byte_count,
            "private_snapshot_identity_sha256": _sha_json(self.snapshot_identity),
            "parser_consumption": "must_use_continuously_held_snapshot_fd_via_proc_self_fd_or_equivalent_descriptor_api",
            "parser_fd_path": self.parser_fd_path,
            "ordinary_source_or_snapshot_pathname_reopen_permitted": False,
            "pathname_pre_and_post_hash_alone_is_sufficient": False,
        }


def _read_0444_same_fd(path: Path, *, label: str) -> _VerifiedImmutable:
    """Read an immutable receipt through one O_NOFOLLOW fd and recheck its path."""
    lexical = _absolute_lexical(path)
    _assert_real_directory_chain(lexical.parent, label=label)
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW required by fixed-GPU cost gate")
    try:
        descriptor = os.open(lexical, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise TrackBV2SubjectMDevelopmentExecutorError(
            f"cannot open {label} without following symlinks"
        ) from exc
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        require(stat.S_IMODE(before.st_mode) == 0o444, f"{label} must be mode 0444")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        raw = b"".join(blocks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = _identity(before)
    require(_identity(after) == identity and len(raw) == before.st_size,
            f"{label} changed while read")
    try:
        named = lexical.lstat()
    except OSError as exc:
        raise TrackBV2SubjectMDevelopmentExecutorError(
            f"{label} pathname disappeared or was replaced after read"
        ) from exc
    require(stat.S_ISREG(named.st_mode) and not stat.S_ISLNK(named.st_mode) and
            _identity(named) == identity,
            f"{label} pathname identity changed after read")
    return _VerifiedImmutable(path=lexical, raw=raw, sha256=hashlib.sha256(raw).hexdigest(), identity=identity)


def _fsync_directory(directory: Path) -> None:
    """Persist a private snapshot directory entry without following aliases."""
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise TrackBV2SubjectMDevelopmentExecutorError(
            f"cannot fsync private snapshot parent: {directory}"
        ) from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_exact_open_fd(descriptor: int, *, label: str) -> tuple[bytes, os.stat_result]:
    """Read a regular fd once, proving it did not mutate while consumed."""
    before = os.fstat(descriptor)
    require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
    blocks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1 << 20)
        if not block:
            break
        blocks.append(block)
    after = os.fstat(descriptor)
    require(_identity(after) == _identity(before), f"{label} changed while read")
    raw = b"".join(blocks)
    require(len(raw) == before.st_size, f"{label} byte count changed while read")
    return raw, before


@contextmanager
def open_verified_target_asset_private_snapshot(
    *, source_path: Path, expected_sha256: str, expected_bytes: int, snapshot_path: Path,
) -> Iterator[ParserBoundVerifiedTargetSnapshot]:
    """Copy one verified O_NOFOLLOW source FD into an O_EXCL private snapshot.

    A future live loader can only receive ``parser_fd_path`` from this context.
    This prevents the classic ABA sequence in which a target pathname is
    checked, replaced during parsing, then restored before a post-check.  The
    source pathname is checked before snapshot publication, but parser
    consumption is bound to a descriptor for the verified snapshot inode.

    This helper never recognises NWB contents and intentionally has no role in
    the no-data CLI.  It is unit-tested solely with synthetic byte files.
    """
    require(_valid_sha(expected_sha256), "private target snapshot requires a lowercase expected SHA-256")
    require(isinstance(expected_bytes, int) and expected_bytes > 0,
            "private target snapshot requires a positive expected byte count")
    source = _absolute_lexical(source_path)
    snapshot = _absolute_lexical(snapshot_path)
    _assert_real_directory_chain(source.parent, label="verified target source")
    _assert_real_directory_chain(snapshot.parent, label="private target snapshot")
    require(source != snapshot, "private target snapshot must not overwrite its source")
    require(not os.path.lexists(snapshot), "private target snapshot path must be fresh")
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW for verified target snapshot")

    source_fd = -1
    snapshot_fd = -1
    parser_fd = -1
    snapshot_created = False
    completed = False
    owned_snapshot_identity: tuple[int, int, int, int, int, int] | None = None
    try:
        try:
            source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
        except OSError as exc:
            raise TrackBV2SubjectMDevelopmentExecutorError(
                "cannot open verified target source without following symlinks"
            ) from exc
        source_before = os.fstat(source_fd)
        require(stat.S_ISREG(source_before.st_mode), "verified target source must be a regular file")

        try:
            snapshot_fd = os.open(
                snapshot,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        except OSError as exc:
            raise TrackBV2SubjectMDevelopmentExecutorError(
                "cannot create a fresh private target snapshot with O_EXCL"
            ) from exc
        snapshot_created = True
        digest = hashlib.sha256()
        copied = 0
        while True:
            block = os.read(source_fd, 1 << 20)
            if not block:
                break
            digest.update(block)
            copied += len(block)
            view = memoryview(block)
            while view:
                written = os.write(snapshot_fd, view)
                require(written > 0, "private target snapshot short write")
                view = view[written:]
        source_after = os.fstat(source_fd)
        source_identity = _identity(source_before)
        require(_identity(source_after) == source_identity and copied == source_before.st_size,
                "verified target source changed while copied")
        observed_sha = digest.hexdigest()
        require(copied == expected_bytes and observed_sha == expected_sha256,
                "verified target source byte size or SHA drift")
        try:
            source_named = source.lstat()
        except OSError as exc:
            raise TrackBV2SubjectMDevelopmentExecutorError(
                "verified target source pathname disappeared or was replaced during copy"
            ) from exc
        require(stat.S_ISREG(source_named.st_mode) and not stat.S_ISLNK(source_named.st_mode) and
                _identity(source_named) == source_identity,
                "verified target source pathname identity changed during copy")
        os.fsync(snapshot_fd)
        snapshot_written = os.fstat(snapshot_fd)
        require(stat.S_ISREG(snapshot_written.st_mode) and snapshot_written.st_size == copied,
                "private target snapshot size drift")
        os.close(snapshot_fd)
        snapshot_fd = -1
        os.chmod(snapshot, 0o444)
        _fsync_directory(snapshot.parent)

        try:
            parser_fd = os.open(snapshot, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
        except OSError as exc:
            raise TrackBV2SubjectMDevelopmentExecutorError(
                "cannot reopen private target snapshot without following symlinks"
            ) from exc
        snapshot_before = os.fstat(parser_fd)
        require(stat.S_ISREG(snapshot_before.st_mode) and stat.S_IMODE(snapshot_before.st_mode) == 0o444,
                "private target snapshot must be immutable regular mode 0444")
        # Hash through the very descriptor whose /proc/self/fd endpoint is
        # handed to a future parser; do not establish trust via pathname.
        snapshot_raw, snapshot_info = _read_exact_open_fd(parser_fd, label="private target snapshot parser fd")
        snapshot_identity = _identity(snapshot_info)
        owned_snapshot_identity = snapshot_identity
        require(hashlib.sha256(snapshot_raw).hexdigest() == observed_sha and len(snapshot_raw) == copied,
                "private target snapshot bytes diverge from verified source fd")
        # Rewind so a parser sees the entire immutable snapshot from byte zero.
        os.lseek(parser_fd, 0, os.SEEK_SET)
        try:
            snapshot_named = snapshot.lstat()
        except OSError as exc:
            raise TrackBV2SubjectMDevelopmentExecutorError(
                "private target snapshot pathname disappeared or was replaced"
            ) from exc
        require(stat.S_ISREG(snapshot_named.st_mode) and not stat.S_ISLNK(snapshot_named.st_mode) and
                _identity(snapshot_named) == snapshot_identity,
                "private target snapshot pathname identity changed before parser handoff")
        yield ParserBoundVerifiedTargetSnapshot(
            source_path=source, source_sha256=observed_sha, source_byte_count=copied,
            source_identity=source_identity, snapshot_path=snapshot,
            snapshot_sha256=observed_sha, snapshot_byte_count=copied,
            snapshot_identity=snapshot_identity, parser_fd=parser_fd,
        )
        parser_after = os.fstat(parser_fd)
        require(_identity(parser_after) == snapshot_identity,
                "private target snapshot parser descriptor changed during parser use")
        try:
            snapshot_named_after = snapshot.lstat()
        except OSError as exc:
            raise TrackBV2SubjectMDevelopmentExecutorError(
                "private target snapshot pathname disappeared during parser use"
            ) from exc
        require(stat.S_ISREG(snapshot_named_after.st_mode) and not stat.S_ISLNK(snapshot_named_after.st_mode) and
                _identity(snapshot_named_after) == snapshot_identity,
                "private target snapshot pathname identity changed during parser use")
        completed = True
    finally:
        if parser_fd >= 0:
            os.close(parser_fd)
        if snapshot_fd >= 0:
            os.close(snapshot_fd)
        if source_fd >= 0:
            os.close(source_fd)
        # An exception before a complete successful parser handoff must not
        # leave a future parser an apparently authoritative file.  Do not ever
        # unlink a path after an attacker replaced it: remove only the inode
        # that this call created and verified.
        if snapshot_created and not completed and owned_snapshot_identity is not None and os.path.lexists(snapshot):
            try:
                named = snapshot.lstat()
                if stat.S_ISREG(named.st_mode) and not stat.S_ISLNK(named.st_mode) and \
                        _identity(named) == owned_snapshot_identity:
                    snapshot.unlink()
                    _fsync_directory(snapshot.parent)
            except OSError:
                pass


def _parse_json(verified: _VerifiedImmutable, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(verified.raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2SubjectMDevelopmentExecutorError(f"{label} is not a JSON object") from exc
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _validate_fixed_gpu_cost_payload(
    payload: Mapping[str, Any], *, canonical_body_raw: bytes,
) -> dict[str, Any]:
    """Rebuild every live engineering-cost binding before accepting the gate."""
    require(canonical_body_raw == base.canonical_json_bytes(payload),
            "canonical fixed-GPU cost receipt is not canonical JSON")
    require(payload.get("schema") == fixed_gpu.SCHEMA_COST and
            payload.get("status") == fixed_gpu.STATUS_COST and payload.get("official") is False,
            "canonical fixed-GPU cost receipt schema/status/role drift")
    require(payload.get("fixed_final_geometry") == fixed_gpu.FIXED_FINAL_GEOMETRY.as_dict() and
            payload.get("cost_smoke_geometry") == fixed_gpu.COST_SMOKE_GEOMETRY.as_dict(),
            "canonical fixed-GPU cost receipt geometry drift")
    require(payload.get("fixed_normalized_ridge_lambda") == fixed_gpu.FIXED_NORMALIZED_RIDGE_LAMBDA and
            payload.get("fixed_cosine_knn_k") == fixed_gpu.FIXED_KNN_K and
            payload.get("seed") == fixed_gpu.SEED,
            "canonical fixed-GPU cost receipt fixed decoder/seed drift")
    for key in (
        "scientific_metric_emitted", "winner_emitted", "selector_executed",
        "fixed_geometry_was_selected_from_source_data", "fixed_geometry_was_selected_from_target_data",
        "historical_selector_plan_executed", "historical_selector_plan_selected_geometry",
        "historical_selector_plan_authorizes_this_execution", "outer_target_discovered",
        "outer_target_path_resolved", "outer_target_opened", "formal_data_opened",
    ):
        require(payload.get(key) is False, f"canonical fixed-GPU cost receipt prohibited flag drift: {key}")
    require(payload.get("gpu_fit_call_count") == 1 and
            payload.get("launch_closure_exact_equal_to_final_live") is True,
            "canonical fixed-GPU cost receipt fit/closure evidence drift")

    live_closure = fixed_gpu.route.snapshot_file_closure(
        fixed_gpu.implementation_paths(_FIXED_GPU_COST_RUNNER)
    )
    require(payload.get("implementation_closure_at_launch") == live_closure,
            "canonical fixed-GPU cost receipt implementation closure is not live")
    preflight = payload.get("preflight")
    require(isinstance(preflight, Mapping) and
            preflight.get("schema") == fixed_gpu.SCHEMA_PREFLIGHT and
            preflight.get("status") == fixed_gpu.STATUS_PREFLIGHT and
            preflight.get("official") is False,
            "canonical fixed-GPU cost preflight identity drift")
    require(preflight.get("implementation_closure_at_preflight") == live_closure,
            "canonical fixed-GPU cost preflight closure is not live")
    require(preflight.get("vendored_cuda_static_audit") ==
            fixed_gpu.vendored_cuda_static_audit(live_closure),
            "canonical fixed-GPU cost vendored CUDA audit drift")
    require(preflight.get("physical_cuda_visible_devices") ==
            fixed_gpu.CANONICAL_PHYSICAL_GPU_INDEX and
            preflight.get("logical_sklearn_device") == "cuda:0" and
            preflight.get("source_data_opened") is False and
            preflight.get("outer_target_discovered") is False and
            preflight.get("outer_target_opened") is False and
            preflight.get("formal_data_opened") is False and
            preflight.get("model_fit_called") is False,
            "canonical fixed-GPU cost preflight scope/device drift")

    bindings = payload.get("source_authority_bindings")
    require(isinstance(bindings, Mapping) and set(bindings) == set(fixed_gpu.AUTHORITY_FILES),
            "canonical fixed-GPU cost source-authority key set drift")
    for label, (name, expected_sha, _schema) in fixed_gpu.AUTHORITY_FILES.items():
        binding = bindings.get(label)
        require(isinstance(binding, Mapping) and
                binding.get("path") == str((fixed_gpu.AUTHORITY_ROOT / name).absolute()) and
                binding.get("sha256") == expected_sha and binding.get("mode") == "0444" and
                binding.get("read_once_from_verified_fd") is True,
                f"canonical fixed-GPU cost source-authority binding drift: {label}")
    roles = payload.get("source_authority_roles")
    require(isinstance(roles, Mapping) and
            roles.get("selector_plan") == (
                "historical_source_bundle_lineage_only__not_executed__not_selected__"
                "not_authorizing_fixed_canonical_gpu_cost"
            ), "canonical fixed-GPU cost selector-plan role drift")

    boundary = payload.get("source_fold_boundary")
    require(isinstance(boundary, Mapping) and
            boundary.get("held_source_session_id") == fixed_gpu.HELD_SOURCE_ID and
            boundary.get("peer_session_count") == 26 and
            boundary.get("query_neural_in_fit") is False and
            boundary.get("query_auxiliary_in_fit") is False and
            type(boundary.get("query_rows")) is int and boundary["query_rows"] > 0,
            "canonical fixed-GPU cost source-fold boundary drift")
    validation = payload.get("gpu_fit_validation")
    fit_calls = validation.get("fit_calls") if isinstance(validation, Mapping) else None
    require(isinstance(fit_calls, list) and len(fit_calls) == 1 and
            fit_calls[0].get("label") == "gpu_joint_multisession_fit" and
            fit_calls[0].get("iterations") == 250 and
            fit_calls[0].get("resolved_estimator_device") == "cuda:0" and
            isinstance(fit_calls[0].get("session_model_parameter_devices"), list) and
            len(fit_calls[0]["session_model_parameter_devices"]) == 27 and
            set(fit_calls[0]["session_model_parameter_devices"]) == {"cuda:0"},
            "canonical fixed-GPU cost fitted-device evidence drift")
    require(validation.get("query_neural_in_fit") is False and
            validation.get("query_auxiliary_in_fit") is False,
            "canonical fixed-GPU cost query entered fit")

    cuda_identity = payload.get("cuda_identity")
    require(isinstance(cuda_identity, Mapping) and
            cuda_identity.get("cuda_visible_devices") == fixed_gpu.CANONICAL_PHYSICAL_GPU_INDEX and
            cuda_identity.get("logical_device") == "cuda:0" and
            cuda_identity.get("visible_device_count") == 1 and
            cuda_identity.get("torch_cuda_available") is True and
            isinstance(cuda_identity.get("device_uuid"), str) and
            cuda_identity["device_uuid"].startswith("GPU-"),
            "canonical fixed-GPU cost CUDA identity drift")
    backend = payload.get("cebra_backend_identity")
    require(isinstance(backend, Mapping) and backend.get("actual_cebra") is True and
            backend.get("cebra_version") == fixed_gpu.route.VENDORED_CEBRA_VERSION and
            backend.get("cebra_commit") == fixed_gpu.route.VENDORED_CEBRA_COMMIT and
            backend.get("requested_device") == "cuda:0" and
            backend.get("cpu_fallback_permitted") is False,
            "canonical fixed-GPU cost CEBRA backend identity drift")
    runtime = payload.get("runtime")
    require(isinstance(runtime, Mapping), "canonical fixed-GPU cost runtime evidence missing")
    for key in (
        "strict27_materialization_wall_clock_s", "gpu_fit_plus_transform_wall_clock_s",
        "total_wall_clock_s", "peak_rss_kib",
    ):
        value = runtime.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(float(value)) and float(value) > 0.0,
                f"canonical fixed-GPU cost runtime field invalid: {key}")
    require(isinstance(runtime.get("python_executable"), str) and runtime["python_executable"],
            "canonical fixed-GPU cost Python identity missing")
    return {
        "live_implementation_closure": live_closure,
        "live_implementation_closure_sha256": _sha_json(live_closure),
        "source_authority_binding_sha256": _sha_json(bindings),
        "device_uuid": cuda_identity["device_uuid"],
        "measured_total_wall_clock_s": float(runtime["total_wall_clock_s"]),
        "measured_gpu_fit_plus_transform_wall_clock_s": float(
            runtime["gpu_fit_plus_transform_wall_clock_s"]),
    }


def _cost_gate_unavailable(*, reason: str, body: Path, sidecar: Path) -> dict[str, Any]:
    return {
        "schema": FIXED_GPU_COST_GATE_SCHEMA,
        "status": "NO_GO__FIXED_D8IT250_GPU_COST_RECEIPT_NOT_FRESH_AND_VALID",
        "reason": reason,
        "canonical_body_path": str(body),
        "canonical_sidecar_path": str(sidecar),
        "body_lexists": os.path.lexists(body),
        "sidecar_lexists": os.path.lexists(sidecar),
        "target_execution_permitted": False,
        "target_data_opened": False,
        "gpu_used_by_this_gate": False,
    }


def inspect_fixed_d8it250_gpu_cost_receipt() -> dict[str, Any]:
    """Validate the sole canonical fixed-GPU cost receipt without GPU/data I/O.

    An absent, partial, stale, aliased, malformed, or scientifically non-cost
    receipt is an explicit NO-GO.  The returned binding is deliberately not an
    execution authority: the scaffold has no real executor implementation.
    """
    body = _absolute_lexical(fixed_gpu.CANONICAL_COST_OUTPUT)
    sidecar = body.with_name(f"{body.name}.sha256")
    if not os.path.lexists(body) or not os.path.lexists(sidecar):
        return _cost_gate_unavailable(reason="canonical cost body+sidecar pair is absent or partial", body=body, sidecar=sidecar)
    try:
        verified_body = _read_0444_same_fd(body, label="canonical fixed d8/it250 GPU cost receipt")
        verified_sidecar = _read_0444_same_fd(sidecar, label="canonical fixed d8/it250 GPU cost sidecar")
        require(verified_sidecar.raw == f"{verified_body.sha256}  {body.name}\n".encode("ascii"),
                "canonical fixed-GPU cost body/sidecar mismatch")
        # Recheck the body after sidecar consumption so a rename cannot split a
        # superficially valid pair across two different files.
        named = body.lstat()
        require(_identity(named) == verified_body.identity,
                "canonical fixed-GPU cost body changed after sidecar read")
        payload = _parse_json(verified_body, label="canonical fixed d8/it250 GPU cost receipt")
        validation = _validate_fixed_gpu_cost_payload(
            payload, canonical_body_raw=verified_body.raw,
        )
    except TrackBV2SubjectMDevelopmentExecutorError as exc:
        return _cost_gate_unavailable(reason=str(exc), body=body, sidecar=sidecar)
    return {
        "schema": FIXED_GPU_COST_GATE_SCHEMA,
        "status": "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION",
        "canonical_body_path": str(body),
        "canonical_body_sha256": verified_body.sha256,
        "canonical_sidecar_path": str(sidecar),
        "canonical_sidecar_sha256": verified_sidecar.sha256,
        "fixed_final_geometry": dict(fixed_gpu.FIXED_FINAL_GEOMETRY.as_dict()),
        "cost_smoke_geometry": dict(fixed_gpu.COST_SMOKE_GEOMETRY.as_dict()),
        "cost_smoke_seed": fixed_gpu.SEED,
        "live_implementation_closure_sha256": validation["live_implementation_closure_sha256"],
        "source_authority_binding_sha256": validation["source_authority_binding_sha256"],
        "device_uuid": validation["device_uuid"],
        "measured_total_wall_clock_s": validation["measured_total_wall_clock_s"],
        "measured_gpu_fit_plus_transform_wall_clock_s": validation[
            "measured_gpu_fit_plus_transform_wall_clock_s"],
        "fresh_body_and_sidecar": True,
        "target_execution_permitted": False,
        "target_data_opened": False,
        "gpu_used_by_this_gate": False,
    }


def _exact_offset10_contract() -> dict[str, Any]:
    """Use current successor authority semantics, never the old ambiguous label."""
    return {
        "model_architecture": "offset10-model",
        "offset_left": 5,
        "offset_right": 5,
        "half_open_offsets_relative_to_prediction_endpoint": [-5, 5],
        "exact_per_endpoint_raw_receptive_field": "range(endpoint-5, endpoint+5)",
        "receptive_field_width_raw_bins": 10,
        "previous_raw_bins": 5,
        "prediction_endpoint_bin_included": True,
        "strictly_future_raw_bins_after_endpoint": 4,
        "causal_temporal_exposure_matched": False,
        "bias_direction": "favors_CEBRA_accuracy",
        "online_or_latency_equivalent_language_permitted": False,
    }


def _future_subject_m_target_byte_lineage(*, materializer_plan: Mapping[str, Any]) -> dict[str, Any]:
    canonical = materializer_plan["canonical_development_authority"]
    gate = materializer_plan["target_asset_ledger_gate"]
    asset = gate["target_asset"]
    return {
        "a2_verified_asset_ledger": {
            "canonical_development_target_authority_sha256": canonical[
                "development_target_query_authority_sha256"],
            "target_session_id": asset["session_id"],
            "asset_id": asset["asset_id"],
            "local_path": asset["a2_official_local_nwb_path"],
            "expected_bytes": asset["expected_bytes"],
            "expected_sha256": asset["expected_sha256"],
            "pre_and_post_loader_same_fd_inode_sha_size_proof_required": True,
            "pathname_pre_and_post_hash_alone_is_sufficient": False,
            "required_parser_consumption_boundary": (
                "open_verified_target_asset_private_snapshot__parser_must_consume_continuously_held_"
                "snapshot_fd_via_proc_self_fd_or_equivalent_descriptor_api"
            ),
            "ordinary_target_pathname_reopen_for_parser_permitted": False,
        },
        "v9_t4_reference_lineage": {
            "reference_seed_set": list(_CEBRA_EXECUTION_SEEDS),
            "reference_seed_structure": "sealed_T4_three_seed_aggregate__no_one_to_one_stochastic_pairing_with_CEBRA",
            "pointer_bound_membership_required": True,
            "for_each_reference_seed": {
                "v9_commit_receipt_sha256": "REQUIRED__EXACT_ASSET_VIEW_SEED_COMMIT",
                "v9_runtime_receipt_sha256": "REQUIRED__EXACT_ASSET_VIEW_SEED_RUNTIME",
                "base_input_trace_sha256": "REQUIRED__ACTIVITY_IDENTITY_FIRST_30_REWARDED_TRIALS",
                "query_behavior_trace_sha256": "REQUIRED__STRICTLY_AFTER_REWARDED_TRIAL_50",
                "predictions_targets_npz_sha256": "REQUIRED__FLOAT32_Q_BY_2_TARGET_ARTIFACT",
                "ordered_query_identity_sha256": "REQUIRED__BYTE_IDENTICAL_T4_ENDPOINT_ORDER",
                "parent_two_output_r2_implementation_sha256": "REQUIRED__TORCHMETRICS_PARENT_SEMANTICS",
            },
            "all_cebra_decoders_must_score_the_same_ordered_target_float32_bytes": True,
            "unpaired_claim_required_if_any_target_byte_or_endpoint_authority_differs": True,
        },
        "sua_pmua_cross_view_parity": {
            "same_target_asset_id": asset["asset_id"],
            "same_target_record_required": True,
            "same_ordered_target_behavior_bytes_required": True,
            "same_ordered_prediction_endpoint_authority_required": True,
            "same_t4_query_identity_authority_required": True,
            "cross_view_behavior_order_authority_sha256": "REQUIRED__EQUAL_FOR_SUA_AND_PMUA",
            "pseudo_mua_target_pooling_replay": {
                "input_sua_feature_sha256": "REQUIRED__EXACT_TARGET_RECORD_INPUT",
                "ordered_unit_ids_raw_bytes_sha256": "REQUIRED",
                "ordered_unit_electrode_ids_raw_bytes_sha256": "REQUIRED",
                "unit_count": "REQUIRED",
                "unique_electrode_channel_count": "REQUIRED",
                "electrode_channel_order": "np_unique_ascending_electrode_id",
                "pooling_method": "electrode_ids_from_units_then_pool_spikes_by_electrode",
                "replayed_output_shape": "REQUIRED__BINS_BY_UNIQUE_ELECTRODES",
                "replayed_output_dtype": "float32",
                "replayed_output_pseudo_mua_feature_sha256": "REQUIRED",
                "replay_exact_equal_to_target_pmua_feature": True,
                "caller_supplied_pooling_dict_without_same_record_replay_permitted": False,
            },
        },
    }


def _receipt_topology(*, materializer_plan: Mapping[str, Any], cost_gate: Mapping[str, Any]) -> dict[str, Any]:
    """Predeclare immutable future receipt roles without creating any paths."""
    fixed = materializer_plan["canonical_development_authority"]["fixed_canonical_geometry_contract_sha256"]
    return {
        "schema": SUBJECT_M_EXECUTION_RECEIPT_TOPOLOGY_SCHEMA,
        "status": "FUTURE_ONLY__O_EXCL_0444_BODY_SIDECAR_PAIRS__NOT_CREATED",
        "all_bodies_and_sidecars": {
            "body_creation": "O_EXCL_ONLY",
            "sidecar_creation": "O_EXCL_ONLY_WITH_ROLLBACK_ON_COLLISION",
            "mode": "0444",
            "parent_symlink_forbidden": True,
            "body_and_sidecar_freshness_checked_before_target_open": True,
            "canonical_root_path": "REQUIRES_SEPARATE_ROOT_AUTHORIZATION__NOT_CREATED_BY_DRY_PLAN",
        },
        "target_parser_byte_consumption": {
            "source_open": "O_NOFOLLOW__single_fd__stream_hash_size_identity",
            "source_path_pre_post_hash_without_held_fd_is_sufficient": False,
            "private_snapshot": "fresh_O_EXCL__fsync_file_and_parent__0444__same_bytes_as_verified_source_fd",
            "parser_handoff": "continuously_held_snapshot_fd_via_proc_self_fd_or_equivalent_descriptor_api_only",
            "ordinary_source_or_snapshot_pathname_reopen_permitted": False,
            "post_parser_proof": "same_held_snapshot_fd_identity_and_snapshot_path_identity",
            "ABA_path_swap_detected_or_prevented": True,
        },
        "required_root_preflight": {
            "canonical_development_target_authority_sha256": materializer_plan[
                "canonical_development_authority"]["development_target_query_authority_sha256"],
            "fixed_geometry_contract_sha256": fixed,
            "fixed_d8it250_gpu_cost_receipt_body_sha256": cost_gate.get("canonical_body_sha256", "REQUIRED"),
            "fixed_d8it250_gpu_cost_receipt_sidecar_sha256": cost_gate.get("canonical_sidecar_sha256", "REQUIRED"),
        },
        "per_target_session_once": [
            "pre_loader_target_asset_byte_identity",
            "post_loader_target_asset_byte_identity",
            "continuous_M50_support_and_strict_post_M_query_index_authority",
            "exact_T4_target_byte_and_offset10_receptive_field_authority",
            "cross_view_behavior_order_authority",
        ],
        "per_cebra_seed": {
            "seed_set": list(_CEBRA_EXECUTION_SEEDS),
            "same_legal_target_serviceable_model_bundle_scores_all_readout_routes_and_decoders": True,
            "model_arms": list(_MODEL_ARMS),
            "encoder_bundle": {
                "geometry": "d8-it10000",
                "legal_multisession_target_serviceability_required": True,
                "never_transform_unfitted_unseen_target_session": True,
                "target_support_dense_velocity_enters_encoder_fit": True,
                "target_support_neural_enters_encoder_fit": True,
                "target_query_neural_or_labels_enter_fit": False,
                "model_and_embedding_provenance_required": True,
            },
            "readout_routes": {
                "source_only_consumer_mechanism_alignment": {
                    "readout_fit_scope": "source_fit_only",
                    "encoder_fit_scope": "legal_joint_multisession_source_plus_target_support",
                    "source_readout_embeddings": "source_session_embeddings_from_the_same_joint_fit_model_bundle",
                    "target_query_embedding": "target_session_query_embedding_from_the_same_joint_fit_model_bundle",
                    "target_support_dense_labels_in_readout_fit": False,
                    "unfitted_target_transform_or_source_only_encoder_permitted": False,
                },
                "target_support_only_standard_cebra_accuracy": {
                    "readout_fit_scope": "target_support_only",
                    "encoder_fit_scope": "legal_joint_multisession_source_plus_target_support",
                    "target_support_dense_labels_in_readout_fit": True,
                    "independent_readout_fit_required": True,
                },
                "source_plus_target_support_hybrid_sensitivity": {
                    "readout_fit_scope": "source_fit_plus_target_support",
                    "encoder_fit_scope": "legal_joint_multisession_source_plus_target_support",
                    "target_support_dense_labels_in_readout_fit": True,
                    "independent_readout_fit_required": True,
                    "may_replace_headline_or_mechanism_route": False,
                },
            },
            "decoders": list(_DECODERS),
            "one_score_receipt_per_model_arm_route_decoder": True,
            "posthoc_best_seed_route_decoder_or_arm_selection_permitted": False,
        },
        "aggregate": {
            "session_then_cebra_seed_aggregation_required": True,
            "subject_m_reference_three_seed_structure_disclosed": True,
            "paired_claim_requires_exact_target_byte_order_parity": True,
            "no_best_seed_or_best_route_substitution": True,
        },
    }


def build_subject_m_development_executor_dry_plan(
    *, dataset: str, view: str | None, outer_fold_id: str, target_session_id: object,
    proposed_target_path: object | None = None, proposed_target_discovery: object | None = None,
    execution_requested: bool = False, device: str = "cpu",
) -> dict[str, Any]:
    """Render the full future subject-M execution/scoring contract, no I/O on target.

    The only non-target file reads are canonical authority/materializer JSON
    pairs and the fixed GPU cost receipt pair.  This order makes a missing or
    invalid GPU cost receipt fail closed before any hypothetical target path
    exists in a future successor.
    """
    dataset, view = base.validate_scope(dataset, view)
    require(dataset == "subject_m", "subject-M executor skeleton rejects RT; RT target-byte ledger remains NO-GO")
    require(proposed_target_path is None, "subject-M executor dry plan accepts no target path")
    require(proposed_target_discovery is None, "subject-M executor dry plan accepts no target discovery callable")
    require(execution_requested is False, "subject-M executor skeleton has no execution mode")
    require(device == "cpu", "subject-M executor skeleton dry path is CPU/no-data only")
    try:
        materializer_plan = materializer.build_development_target_materializer_dry_plan(
            dataset=dataset, view=view, outer_fold_id=outer_fold_id,
            target_session_id=target_session_id,
        )
    except materializer.TrackBV2DevelopmentTargetMaterializerError as exc:
        raise TrackBV2SubjectMDevelopmentExecutorError(str(exc)) from exc
    require(materializer_plan.get("status") == "CANONICAL_DEVELOPMENT_AUTHORITY_AND_SUBM_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS",
            "subject-M executor requires canonical materializer plus verified A2 target asset ledger")
    cost_gate = inspect_fixed_d8it250_gpu_cost_receipt()
    cost_valid = cost_gate["status"] == "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION"
    payload = {
        "schema": SUBJECT_M_DEVELOPMENT_EXECUTOR_DRY_PLAN_SCHEMA,
        "status": (
            "NO_GO__FIXED_D8IT250_GPU_COST_RECEIPT_REQUIRED_BEFORE_ANY_TARGET_EXECUTOR"
            if not cost_valid else
            "GPU_COST_GATE_VALID__FUTURE_SUBJECT_M_EXECUTOR_AND_SCORER_STILL_NOT_IMPLEMENTED"
        ),
        "dataset": dataset,
        "view": view,
        "outer_fold_id": materializer_plan["outer_fold_id"],
        "target_session_id": materializer_plan["target_session_id"],
        "canonical_materializer_binding": {
            "canonical_development_target_authority_sha256": materializer_plan[
                "canonical_development_authority"]["development_target_query_authority_sha256"],
            "canonical_metric_pointer_body_sha256": materializer_plan[
                "canonical_development_authority"]["metric_pointer_body_sha256"],
            "fixed_geometry_contract_sha256": materializer_plan[
                "canonical_development_authority"]["fixed_canonical_geometry_contract_sha256"],
            "target_asset_ledger_is_a2_v2_verified": True,
            "caller_supplied_authority_or_asset_sha_permitted": False,
        },
        "fixed_gpu_cost_gate": cost_gate,
        "future_target_loader_and_encoder_contract": {
            "canonical_loader_semantics_must_be_revalidated_live": True,
            "target_asset_byte_pre_and_post_loader_proof_required": True,
            "parser_must_consume_private_snapshot_from_continuously_held_verified_fd": True,
            "ordinary_target_pathname_reopen_is_not_parser_consumption_proof": True,
            "continuous_target_support_prefix": "start_of_record_through_stop_of_rewarded_trial_50__all_intervening_rows",
            "strict_post_M_target_query": "rewarded_trials_strictly_after_50_only",
            "target_support_dense_velocity_enters_standard_joint_encoder_fit": True,
            "target_query_neural_or_dense_velocity_enters_any_fit": False,
            "fixed_encoder_geometry": {
                "output_dimension": 8,
                "iterations": 10_000,
                "source_or_target_geometry_selection_performed": False,
            },
            "cebra_import_or_fit_permitted_by_this_plan": False,
        },
        "exact_offset10_endpoint_and_receptive_field_contract": _exact_offset10_contract(),
        "exact_t4_target_byte_lineage": _future_subject_m_target_byte_lineage(
            materializer_plan=materializer_plan,
        ),
        "immutable_future_receipt_topology": _receipt_topology(
            materializer_plan=materializer_plan, cost_gate=cost_gate,
        ),
        "target_execution_permitted": False,
        "target_data_discovery_permitted": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_trained": False,
        "readout_fit_called": False,
        "score_emitted": False,
        "gpu_used": False,
        "official_execution_receipt_minted": False,
    }
    return payload


def refuse_subject_m_target_execution(**kwargs: Any) -> None:
    """A named tripwire for callers attempting to turn this dry skeleton live."""
    build_subject_m_development_executor_dry_plan(**kwargs)
    raise TrackBV2SubjectMDevelopmentExecutorError(
        "subject-M target executor/scorer is a dry skeleton; a separately root-authorised live implementation is required"
    )
