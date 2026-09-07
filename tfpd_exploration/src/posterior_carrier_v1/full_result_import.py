"""Descriptor-safe future import of a completed remote Posterior full result.

This module is deliberately inert at import time.  It imports only the Python
standard library, never opens SSH, a result, source data, a checkpoint tensor,
or CUDA, and never creates the canonical mirror.  A future root reviewer must
provide the opaque capability defined here before an injected transport can be
used.

The importer keeps two provenance layers separate:

* ``ImportedFullMirrorProvenance`` compatibility is retained exactly for the
  matched scorer's existing local-mirror loader; and
* a richer remote-import attestation binds the SSH endpoint, host-key pin, and
  both remote directory identities.  It is returned to the future authority
  publisher rather than stored as an extra leaf inside the mirror, because the
  scorer intentionally rejects any mirror leaf outside the completed-training
  body/sidecar topology.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable, Mapping, Protocol


CELL = "POSTERIOR_CARRIER_BUDGETMIX_D_SEED42"
FULL_TRAIN_PHASE = "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_FULL_TRAIN_V1"
IMPORT_PHASE = "POSTERIOR_CARRIER_COMPLETED_FULL_IMPORT_V1"
IMPORT_SCHEMA = "posterior_carrier_completed_full_remote_import_v1"
REMOTE_MANIFEST_SCHEMA = "posterior_carrier_remote_full_snapshot_manifest_v1"
REMOTE_ATTESTATION_SCHEMA = "posterior_carrier_completed_full_import_attestation_v1"
COPY_PROTOCOL = "held_descriptor_byte_preserving_copy_to_fresh_local_immutable_mirror"

REMOTE_HOST_ALIAS = "xinyuan-ROG"
REMOTE_USER = "xinyuan"
REMOTE_HOST_KEY_FINGERPRINT = "SHA256:LowyHTkZ7BXkHiL1tKajgvwtHhEXsieClZar2++cKwI"
REMOTE_STAGE_ROOT = "/home/xinyuan/Work_host/posterior_carrier_budgetmix_d_seed42_full_train_stage_v1"
REMOTE_RESULT_ROOT = (
    "/home/xinyuan/Work_host/posterior_carrier_budgetmix_d_seed42_full_train_stage_v1/"
    "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_full_train_v1"
)
REMOTE_RESULT_RELATIVE_TO_STAGE = "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_full_train_v1"
LOCAL_MIRROR_RELATIVE = (
    "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_full_train_import_mirror_v1"
)

SOURCE_EPOCHS = 48
CHECKPOINT_EPOCHS = (44, 45, 46, 47)
FRAME_MAGIC = b"POSTERIOR-CARRIER-FULL-IMPORT-FRAME-V1\n"
MAX_FRAME_METADATA_BYTES = 1 << 20
COPY_CHUNK_BYTES = 1 << 20

_REMOTE_TERMINAL_KEYS = frozenset({
    "schema", "cell", "phase", "spec", "identity", "final_identity", "attempt_sha256", "launch_sha256",
    "source_authority_sha256", "artifact_sha256s", "checkpoint_state_sha256s", "swa_state_sha256",
    "swa_proof", "full_launch_closure", "full_final_closure", "phase_b_v3_launch_closure",
    "phase_b_v3_final_closure", "boundaries", "status",
})


class FullResultImportError(RuntimeError):
    """Fail closed before a remote byte becomes a local canonical artifact."""


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise FullResultImportError(f"{label} must be an exact lowercase SHA-256")
    return value


def _safe_relative(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
        or value in {".", ".."}
    ):
        raise FullResultImportError(f"{label} must be a safe relative path")
    return value


def _directory_identity_from_stat(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        value = os.lstat(path)
    except OSError as error:
        raise FullResultImportError(f"cannot lstat directory: {path}") from error
    if not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode):
        raise FullResultImportError("directory must be a real non-symlink")
    return _directory_identity_from_stat(value)


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, COPY_CHUNK_BYTES)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _write_all(fd: int, body: bytes) -> None:
    view = memoryview(body)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise FullResultImportError("short immutable import metadata write")
        view = view[count:]


def full_training_artifact_names() -> tuple[str, ...]:
    """Exact successful full-training body topology; no glob is ever allowed."""
    return (
        "attempt.json",
        "launch.json",
        "source_authority.json",
        "throughput100.json",
        *(f"epoch-{epoch:02d}.json" for epoch in range(SOURCE_EPOCHS)),
        *(f"checkpoint-{epoch:02d}.pt" for epoch in CHECKPOINT_EPOCHS),
        "swa_final4.pt",
        "terminal.json",
    )


def remote_leaf_names() -> tuple[str, ...]:
    """Deterministic body/sidecar stream order, never caller-controlled."""
    return tuple(name for body in full_training_artifact_names() for name in (body, f"{body}.sha256"))


def _source_only_boundaries() -> dict[str, object]:
    return {
        "source_only": True,
        "target_opened": False,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "h1_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "scientific_score": False,
        "cache_read_or_write": False,
    }


@dataclass(frozen=True)
class DirectoryIdentity:
    """Observed descriptor identity, supplied only at a completed terminal."""

    dev: int
    ino: int

    def __post_init__(self) -> None:
        if type(self.dev) is not int or type(self.ino) is not int or self.dev < 0 or self.ino < 0:
            raise FullResultImportError("directory dev/inode must be nonnegative exact integers")

    def payload(self) -> list[int]:
        return [self.dev, self.ino]


@dataclass(frozen=True)
class RemoteEndpoint:
    """Fixed host/user/path authority; observed inodes remain dynamic evidence."""

    host_alias: str = REMOTE_HOST_ALIAS
    user: str = REMOTE_USER
    stage_root: str = REMOTE_STAGE_ROOT
    result_root: str = REMOTE_RESULT_ROOT
    result_relative_to_stage: str = REMOTE_RESULT_RELATIVE_TO_STAGE
    host_key_fingerprint: str = REMOTE_HOST_KEY_FINGERPRINT

    def __post_init__(self) -> None:
        if (
            self.host_alias != REMOTE_HOST_ALIAS
            or self.user != REMOTE_USER
            or self.stage_root != REMOTE_STAGE_ROOT
            or self.result_root != REMOTE_RESULT_ROOT
            or self.result_relative_to_stage != REMOTE_RESULT_RELATIVE_TO_STAGE
            or self.host_key_fingerprint != REMOTE_HOST_KEY_FINGERPRINT
        ):
            raise FullResultImportError("remote endpoint literal authority drift")
        if not Path(self.stage_root).is_absolute() or not Path(self.result_root).is_absolute():
            raise FullResultImportError("remote stage/result roots must be absolute")
        if Path(self.stage_root) / self.result_relative_to_stage != Path(self.result_root):
            raise FullResultImportError("remote result path is not the exact stage-relative route")
        _safe_relative(self.result_relative_to_stage, "remote result relative path")

    def payload(self) -> dict[str, str]:
        return {
            "host_alias": self.host_alias,
            "user": self.user,
            "stage_root": self.stage_root,
            "result_root": self.result_root,
            "result_relative_to_stage": self.result_relative_to_stage,
            "host_key_fingerprint": self.host_key_fingerprint,
        }


def _openssh_fingerprint_from_known_hosts_line(line: str) -> str:
    if not isinstance(line, str) or not line or "\n" in line or "\r" in line:
        raise FullResultImportError("known-hosts line must be exactly one nonempty line")
    fields = line.split()
    if len(fields) < 3:
        raise FullResultImportError("known-hosts line must contain host, algorithm, and public key")
    try:
        public = base64.b64decode(fields[2].encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise FullResultImportError("known-hosts public key base64 drift") from error
    if not public:
        raise FullResultImportError("known-hosts public key is empty")
    rendered = base64.b64encode(hashlib.sha256(public).digest()).decode("ascii").rstrip("=")
    return f"SHA256:{rendered}"


@dataclass(frozen=True)
class HostKeyPin:
    """Root-supplied host-key pin.

    The fingerprint is always an authority.  The actual known-hosts line is
    deliberately optional for synthetic transports: no CPU-only test needs a
    preimage of the real remote host-key fingerprint.  A physical SSH
    transport, however, calls :meth:`require_live_known_hosts_line`, so a
    capability without the real public-key line can never open a network
    connection.
    """

    known_hosts_line: str | None = None
    expected_fingerprint: str = REMOTE_HOST_KEY_FINGERPRINT

    def payload(self, *, endpoint: RemoteEndpoint) -> dict[str, object]:
        if self.expected_fingerprint != endpoint.host_key_fingerprint:
            raise FullResultImportError("host-key fingerprint authority drift")
        if self.known_hosts_line is None:
            line_sha256: str | None = None
        else:
            if _openssh_fingerprint_from_known_hosts_line(self.known_hosts_line) != self.expected_fingerprint:
                raise FullResultImportError("known-hosts line does not match frozen host-key fingerprint")
            host_field = self.known_hosts_line.split()[0]
            if host_field != endpoint.host_alias:
                raise FullResultImportError("known-hosts line host alias drift")
            line_sha256 = _digest(self.known_hosts_line.encode("utf-8"))
        return {
            "host_alias": endpoint.host_alias,
            "fingerprint": self.expected_fingerprint,
            "known_hosts_line_sha256": line_sha256,
            "live_known_hosts_line_required": True,
        }

    def require_live_known_hosts_line(self, *, endpoint: RemoteEndpoint) -> str:
        self.payload(endpoint=endpoint)
        if self.known_hosts_line is None:
            raise FullResultImportError("physical SSH import requires a root-supplied pinned known-hosts line")
        return self.known_hosts_line


@dataclass(frozen=True)
class RemoteTerminalAuthority:
    """Observed terminal identities supplied by the future root reviewer.

    The current dev/inode observations are intentionally *not* module literals.
    The reviewer must provide a freshly observed completed-terminal authority,
    and the remote held-FD program rechecks it immediately before streaming.
    """

    stage_identity: DirectoryIdentity
    result_identity: DirectoryIdentity
    terminal_sha256: str
    full_closure_sha256: str

    def __post_init__(self) -> None:
        if self.stage_identity == self.result_identity:
            raise FullResultImportError("remote stage and completed-result descriptors must be distinct")
        _sha(self.terminal_sha256, "remote terminal SHA")
        _sha(self.full_closure_sha256, "remote full closure SHA")

    def payload(self) -> dict[str, object]:
        return {
            "stage_descriptor_identity": self.stage_identity.payload(),
            "result_descriptor_identity": self.result_identity.payload(),
            "terminal_sha256": _sha(self.terminal_sha256, "remote terminal SHA"),
            "full_closure_sha256": _sha(self.full_closure_sha256, "remote full closure SHA"),
        }


class _RootImportCapabilitySeal:
    __slots__ = ()


_ROOT_IMPORT_CAPABILITY_SEAL = _RootImportCapabilitySeal()


@dataclass(frozen=True)
class RootReviewedImportCapability:
    """Opaque in-process authorization for one exact remote terminal import."""

    endpoint: RemoteEndpoint
    terminal_authority: RemoteTerminalAuthority
    host_key_pin: HostKeyPin
    destination_relative: str
    _seal: object = field(repr=False, compare=False)

    def validate(self) -> dict[str, object]:
        if self._seal is not _ROOT_IMPORT_CAPABILITY_SEAL:
            raise FullResultImportError("remote import requires an in-process root-reviewed capability")
        if self.destination_relative != LOCAL_MIRROR_RELATIVE:
            raise FullResultImportError("remote import destination authority drift")
        endpoint = self.endpoint.payload()
        host_key = self.host_key_pin.payload(endpoint=self.endpoint)
        terminal = self.terminal_authority.payload()
        return {
            "endpoint": endpoint,
            "host_key_pin": host_key,
            "terminal_authority": terminal,
            "destination_relative": self.destination_relative,
        }


def _issue_root_review_capability_for_completed_remote_full_result(
    *,
    endpoint: RemoteEndpoint,
    terminal_authority: RemoteTerminalAuthority,
    host_key_pin: HostKeyPin,
) -> RootReviewedImportCapability:
    """Internal-only root factory; no public CLI can construct this token."""
    capability = RootReviewedImportCapability(
        endpoint=endpoint,
        terminal_authority=terminal_authority,
        host_key_pin=host_key_pin,
        destination_relative=LOCAL_MIRROR_RELATIVE,
        _seal=_ROOT_IMPORT_CAPABILITY_SEAL,
    )
    capability.validate()
    return capability


@dataclass
class TransportHandle:
    """One binary remote stream and its exact lifecycle finalizer."""

    stream: BinaryIO
    finalize: Callable[[bool], None]
    _closed: bool = False

    def close(self, *, success: bool) -> None:
        if self._closed:
            return
        self._closed = True
        self.finalize(success)


class RemoteTransport(Protocol):
    """Injected physical transport; tests use a purely in-memory implementation."""

    def open_completed_terminal_stream(
        self,
        *,
        endpoint: RemoteEndpoint,
        terminal_authority: RemoteTerminalAuthority,
        host_key_pin: HostKeyPin,
    ) -> TransportHandle: ...


def _frame(kind: str, name: str | None, payload: bytes) -> bytes:
    metadata = _json({"kind": kind, "name": name, "bytes": len(payload)})
    if len(metadata) > MAX_FRAME_METADATA_BYTES:
        raise FullResultImportError("frame metadata exceeds fixed bound")
    return struct.pack(">I", len(metadata)) + metadata + payload


def build_framed_snapshot(manifest: Mapping[str, object], leaves: Mapping[str, bytes]) -> bytes:
    """Test-only deterministic encoder matching the remote stream protocol."""
    payload = bytearray(FRAME_MAGIC)
    payload += _frame("manifest", None, _json(dict(manifest)))
    for name in remote_leaf_names():
        if name not in leaves:
            raise FullResultImportError("test framed snapshot omitted an exact remote leaf")
        payload += _frame("leaf", name, leaves[name])
    return bytes(payload)


class FramedSnapshotReader:
    """Strict reader with fixed-order frames and no pathname interpretation."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._started = False

    def _read_exact(self, count: int) -> bytes:
        if type(count) is not int or count < 0:
            raise FullResultImportError("framed stream requested invalid byte count")
        chunks: list[bytes] = []
        remaining = count
        while remaining:
            block = self._stream.read(remaining)
            if not isinstance(block, bytes) or not block:
                raise FullResultImportError("framed remote stream truncated")
            if len(block) > remaining:
                raise FullResultImportError("framed remote stream over-read/injection drift")
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)

    def _frame_header(self) -> dict[str, object]:
        length = struct.unpack(">I", self._read_exact(4))[0]
        if length < 2 or length > MAX_FRAME_METADATA_BYTES:
            raise FullResultImportError("framed remote metadata length drift")
        try:
            metadata = json.loads(self._read_exact(length))
        except (TypeError, json.JSONDecodeError) as error:
            raise FullResultImportError("framed remote metadata JSON drift") from error
        if not isinstance(metadata, dict) or set(metadata) != {"kind", "name", "bytes"}:
            raise FullResultImportError("framed remote metadata schema drift")
        if metadata["kind"] not in {"manifest", "leaf"}:
            raise FullResultImportError("framed remote metadata kind drift")
        if metadata["name"] is not None and (not isinstance(metadata["name"], str) or "/" in metadata["name"]):
            raise FullResultImportError("framed remote leaf path traversal/drift")
        if type(metadata["bytes"]) is not int or metadata["bytes"] < 0:
            raise FullResultImportError("framed remote payload length drift")
        return metadata

    def read_manifest(self) -> dict[str, object]:
        if self._started:
            raise FullResultImportError("framed remote manifest may be read once")
        self._started = True
        if self._read_exact(len(FRAME_MAGIC)) != FRAME_MAGIC:
            raise FullResultImportError("framed remote stream magic drift")
        metadata = self._frame_header()
        if metadata != {"kind": "manifest", "name": None, "bytes": metadata["bytes"]}:
            raise FullResultImportError("framed remote manifest metadata drift")
        if metadata["bytes"] > MAX_FRAME_METADATA_BYTES:
            raise FullResultImportError("remote snapshot manifest exceeds fixed bound")
        try:
            value = json.loads(self._read_exact(int(metadata["bytes"])))
        except (TypeError, json.JSONDecodeError) as error:
            raise FullResultImportError("remote snapshot manifest JSON drift") from error
        if not isinstance(value, dict):
            raise FullResultImportError("remote snapshot manifest root drift")
        return value

    def copy_next_leaf(
        self,
        *,
        expected_name: str,
        expected_bytes: int,
        expected_sha256: str,
        write: Callable[[bytes], None],
    ) -> None:
        metadata = self._frame_header()
        if (
            metadata.get("kind") != "leaf"
            or metadata.get("name") != expected_name
            or metadata.get("bytes") != expected_bytes
        ):
            raise FullResultImportError("framed remote leaf order/name/length drift")
        digest = hashlib.sha256()
        remaining = expected_bytes
        while remaining:
            block = self._stream.read(min(COPY_CHUNK_BYTES, remaining))
            if not isinstance(block, bytes) or not block:
                raise FullResultImportError("framed remote leaf truncated")
            if len(block) > remaining:
                raise FullResultImportError("framed remote leaf over-read/injection drift")
            write(block)
            digest.update(block)
            remaining -= len(block)
        if digest.hexdigest() != _sha(expected_sha256, f"remote leaf SHA {expected_name}"):
            raise FullResultImportError("framed remote leaf body/SHA drift")

    def ensure_eof(self) -> None:
        if self._stream.read(1):
            raise FullResultImportError("framed remote stream contains injected trailing data")


def _manifest_leaf_map(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(remote_leaf_names()):
        raise FullResultImportError("remote manifest leaf cardinality drift")
    result: dict[str, dict[str, object]] = {}
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"name", "bytes", "sha256", "kind", "mode"}:
            raise FullResultImportError("remote manifest leaf schema drift")
        name = row.get("name")
        if not isinstance(name, str) or name in result:
            raise FullResultImportError("remote manifest duplicate/malformed leaf")
        result[name] = dict(row)
    if tuple(result) != remote_leaf_names():
        raise FullResultImportError("remote manifest leaf topology/order/extra drift")
    for name, row in result.items():
        if (
            row["kind"] != "regular"
            or row["mode"] != "0444"
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
        ):
            raise FullResultImportError("remote manifest leaf type/mode/length drift")
        _sha(row["sha256"], f"remote manifest leaf SHA {name}")
    return result


def _validate_closure_reference(value: object, *, expected_sha256: str, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("closure_sha256") != _sha(expected_sha256, f"expected {label} SHA"):
        raise FullResultImportError(f"{label} closure binding drift")


def _validate_pre_stream_terminal_graph(
    terminal: Mapping[str, object],
    *,
    body_sha: Mapping[str, str],
    terminal_authority: RemoteTerminalAuthority,
) -> None:
    """Validate the completed terminal graph before any remote leaf is copied.

    This is intentionally narrower than the scorer's later 48-epoch semantic
    validator, but it is not merely a status check: the complete terminal
    schema, artifact graph, final-four state topology, identity equality and
    both closure families must already be coherent before the destination is
    even reserved.
    """
    if set(terminal) != _REMOTE_TERMINAL_KEYS:
        raise FullResultImportError("remote snapshot terminal schema drift")
    if (
        terminal.get("schema") != "posterior_carrier_full_train_terminal_v1"
        or terminal.get("cell") != CELL
        or terminal.get("phase") != FULL_TRAIN_PHASE
        or not isinstance(terminal.get("spec"), Mapping)
        or not isinstance(terminal.get("identity"), Mapping)
        or terminal.get("final_identity") != terminal.get("identity")
        or terminal.get("attempt_sha256") != body_sha["attempt.json"]
        or terminal.get("launch_sha256") != body_sha["launch.json"]
        or terminal.get("source_authority_sha256") != body_sha["source_authority.json"]
        or terminal.get("artifact_sha256s")
        != {name: body_sha[name] for name in full_training_artifact_names() if name != "terminal.json"}
        or terminal.get("boundaries") != _source_only_boundaries()
        or terminal.get("status") != "FULL_TRAINING_COMPLETE__SOURCE_ONLY__AWAITING_SEPARATE_SCORER"
    ):
        raise FullResultImportError("remote snapshot terminal graph drift")
    checkpoints = terminal.get("checkpoint_state_sha256s")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {str(epoch) for epoch in CHECKPOINT_EPOCHS}:
        raise FullResultImportError("remote snapshot terminal checkpoint-state topology drift")
    for epoch in CHECKPOINT_EPOCHS:
        _sha(checkpoints[str(epoch)], f"remote terminal checkpoint {epoch} state SHA")
    _sha(terminal.get("swa_state_sha256"), "remote terminal SWA state SHA")
    if not isinstance(terminal.get("swa_proof"), Mapping):
        raise FullResultImportError("remote snapshot terminal SWA proof schema drift")
    _validate_closure_reference(
        terminal.get("full_launch_closure"), expected_sha256=terminal_authority.full_closure_sha256,
        label="remote terminal launch",
    )
    _validate_closure_reference(
        terminal.get("full_final_closure"), expected_sha256=terminal_authority.full_closure_sha256,
        label="remote terminal final",
    )
    if terminal.get("full_launch_closure") != terminal.get("full_final_closure"):
        raise FullResultImportError("remote snapshot terminal launch/final closure drift")
    phase_launch = terminal.get("phase_b_v3_launch_closure")
    phase_final = terminal.get("phase_b_v3_final_closure")
    if not isinstance(phase_launch, Mapping) or phase_launch != phase_final:
        raise FullResultImportError("remote snapshot terminal Phase-B-v3 closure drift")
    phase_sha = phase_launch.get("closure_sha256")
    _sha(phase_sha, "remote terminal Phase-B-v3 closure SHA")


def validate_remote_snapshot_manifest(
    value: Mapping[str, object],
    *,
    endpoint: RemoteEndpoint,
    terminal_authority: RemoteTerminalAuthority,
) -> dict[str, object]:
    """Accept the remote terminal graph *before* any artifact frame is copied."""
    expected = {
        "schema", "endpoint", "stage_descriptor_identity", "result_descriptor_identity", "topology",
        "artifact_sha256s", "leaves", "terminal", "terminal_sha256", "full_closure_sha256", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != REMOTE_MANIFEST_SCHEMA:
        raise FullResultImportError("remote snapshot manifest schema drift")
    if value.get("endpoint") != endpoint.payload():
        raise FullResultImportError("remote snapshot endpoint binding drift")
    if value.get("stage_descriptor_identity") != terminal_authority.stage_identity.payload():
        raise FullResultImportError("remote snapshot stage descriptor identity drift")
    if value.get("result_descriptor_identity") != terminal_authority.result_identity.payload():
        raise FullResultImportError("remote snapshot result descriptor identity drift")
    if value.get("topology") != list(full_training_artifact_names()):
        raise FullResultImportError("remote snapshot successful topology/failure/extra drift")
    leaves = _manifest_leaf_map(value.get("leaves"))
    artifacts = value.get("artifact_sha256s")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(full_training_artifact_names()):
        raise FullResultImportError("remote snapshot artifact SHA topology drift")
    body_sha = {name: _sha(artifacts[name], f"remote artifact SHA {name}") for name in full_training_artifact_names()}
    if any(leaves[name]["sha256"] != body_sha[name] for name in full_training_artifact_names()):
        raise FullResultImportError("remote snapshot body/leaf SHA map drift")
    terminal = value.get("terminal")
    if not isinstance(terminal, Mapping):
        raise FullResultImportError("remote snapshot terminal missing")
    if (
        value.get("terminal_sha256") != body_sha["terminal.json"]
        or terminal_authority.terminal_sha256 != body_sha["terminal.json"]
    ):
        # The actual terminal body has not been copied yet.  Its byte digest
        # is proved by the remote held-FD program and then again by the exact
        # framed leaf and local sidecar checks below; do not silently assume
        # that a receipt publisher used this module's JSON serialization.
        raise FullResultImportError("remote snapshot terminal SHA authority drift")
    _validate_pre_stream_terminal_graph(
        terminal, body_sha=body_sha, terminal_authority=terminal_authority,
    )
    if value.get("full_closure_sha256") != terminal_authority.full_closure_sha256:
        raise FullResultImportError("remote snapshot full closure authority drift")
    if terminal.get("boundaries") != _source_only_boundaries():
        raise FullResultImportError("remote snapshot terminal source-only boundary drift")
    if value.get("status") != "REMOTE_FULL_TERMINAL_SNAPSHOT_READY":
        raise FullResultImportError("remote snapshot terminal status drift")
    return {
        "artifact_sha256s": body_sha,
        "leaves": leaves,
        "terminal": dict(terminal),
        "terminal_sha256": body_sha["terminal.json"],
        "full_closure_sha256": terminal_authority.full_closure_sha256,
    }


def _repository_root(root: Path) -> tuple[Path, int, tuple[int, int]]:
    root = Path(root).absolute()
    identity = _directory_identity(root)
    try:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise FullResultImportError("cannot O_NOFOLLOW open local repository root") from error
    if _directory_identity_from_stat(os.fstat(descriptor)) != identity:
        os.close(descriptor)
        raise FullResultImportError("local repository root identity drift")
    return root, descriptor, identity


def _open_relative_directory(root_fd: int, relative: str) -> tuple[int, tuple[int, int]]:
    _safe_relative(relative, "local relative directory")
    current = root_fd
    opened: list[int] = []
    try:
        for piece in Path(relative).parts:
            child = os.open(piece, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current)
            opened.append(child)
            current = child
        info = os.fstat(current)
        if not stat.S_ISDIR(info.st_mode):
            raise FullResultImportError("local relative parent is not a directory")
        # The final descriptor is returned to the transaction.  Intermediate
        # directory descriptors are only traversal capabilities and must not
        # leak across a long import.
        for descriptor in opened[:-1]:
            os.close(descriptor)
        return current, _directory_identity_from_stat(info)
    except BaseException:
        for descriptor in reversed(opened):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


@dataclass
class LocalMirrorTransaction:
    """Fresh-root O_EXCL/fsync/0444 local publication with owned rollback."""

    destination: Path
    repository: Path
    parent: Path
    repository_descriptor: int
    repository_identity: tuple[int, int]
    parent_descriptor: int
    parent_identity: tuple[int, int]
    descriptor: int
    identity: tuple[int, int]
    created_leaf_identities: dict[str, tuple[int, int]] = field(default_factory=dict)
    _closed: bool = False

    @classmethod
    def assert_fresh(cls, root: Path) -> None:
        root_path, root_fd, root_identity = _repository_root(root)
        try:
            parent_relative = str(Path(LOCAL_MIRROR_RELATIVE).parent)
            parent_fd, parent_identity = _open_relative_directory(root_fd, parent_relative)
            try:
                if _directory_identity_from_stat(os.fstat(root_fd)) != root_identity:
                    raise FullResultImportError("local repository root identity drift before import")
                try:
                    os.stat(Path(LOCAL_MIRROR_RELATIVE).name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return
                raise FullResultImportError("canonical local import mirror already exists or collides")
            finally:
                os.close(parent_fd)
        finally:
            os.close(root_fd)

    @classmethod
    def reserve(cls, root: Path) -> "LocalMirrorTransaction":
        root_path, root_fd, root_identity = _repository_root(root)
        parent_fd = -1
        descriptor = -1
        created_identity: tuple[int, int] | None = None
        name: str | None = None
        try:
            parent_relative = str(Path(LOCAL_MIRROR_RELATIVE).parent)
            parent_fd, parent_identity = _open_relative_directory(root_fd, parent_relative)
            if _directory_identity_from_stat(os.fstat(root_fd)) != root_identity:
                raise FullResultImportError("local repository root identity drift before reservation")
            name = Path(LOCAL_MIRROR_RELATIVE).name
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
            except FileExistsError as error:
                raise FullResultImportError("canonical local import mirror already exists or collides") from error
            created = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(created.st_mode) or stat.S_ISLNK(created.st_mode):
                raise FullResultImportError("fresh local mirror creation type drift")
            created_identity = _directory_identity_from_stat(created)
            descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            identity = _directory_identity_from_stat(os.fstat(descriptor))
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode) or _directory_identity_from_stat(named) != identity:
                raise FullResultImportError("fresh local mirror named-root identity drift")
            return cls(
                destination=root_path / LOCAL_MIRROR_RELATIVE,
                repository=root_path,
                parent=root_path / Path(LOCAL_MIRROR_RELATIVE).parent,
                repository_descriptor=root_fd,
                repository_identity=root_identity,
                parent_descriptor=parent_fd,
                parent_identity=parent_identity,
                descriptor=descriptor,
                identity=identity,
            )
        except BaseException:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                descriptor = -1
            if created_identity is not None and name is not None and parent_fd >= 0:
                try:
                    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    if (
                        stat.S_ISDIR(current.st_mode)
                        and not stat.S_ISLNK(current.st_mode)
                        and _directory_identity_from_stat(current) == created_identity
                    ):
                        os.rmdir(name, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                except OSError:
                    # A swapped named root is not owned by this transaction;
                    # leave it untouched and fail closed with the original
                    # reservation error.
                    pass
            for item in (parent_fd, root_fd):
                if item >= 0:
                    try:
                        os.close(item)
                    except OSError:
                        pass
            raise
        # Keep the repository descriptor held for the entire transaction so
        # a pathname swap cannot redirect publication after reservation.

    def _assert_identity(self) -> None:
        if self._closed:
            raise FullResultImportError("local mirror transaction has closed")
        if _directory_identity_from_stat(os.fstat(self.repository_descriptor)) != self.repository_identity:
            raise FullResultImportError("local repository descriptor identity drift")
        if _directory_identity(self.repository) != self.repository_identity:
            raise FullResultImportError("local repository named-root identity drift")
        if _directory_identity_from_stat(os.fstat(self.parent_descriptor)) != self.parent_identity:
            raise FullResultImportError("local mirror parent descriptor identity drift")
        if _directory_identity(self.parent) != self.parent_identity:
            raise FullResultImportError("local mirror named-parent identity drift")
        if _directory_identity_from_stat(os.fstat(self.descriptor)) != self.identity:
            raise FullResultImportError("local mirror descriptor identity drift")
        named = os.stat(self.destination.name, dir_fd=self.parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _directory_identity_from_stat(named) != self.identity
        ):
            raise FullResultImportError("local mirror named-root identity drift")

    def publish_from_stream(self, *, name: str, expected_bytes: int, expected_sha256: str, reader: FramedSnapshotReader) -> None:
        if name not in remote_leaf_names() or name in self.created_leaf_identities:
            raise FullResultImportError("local mirror leaf lies outside exact topology or duplicates")
        self._assert_identity()
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=self.descriptor,
            )
        except OSError as error:
            raise FullResultImportError("cannot O_EXCL publish local imported mirror leaf") from error
        # Ownership begins as soon as this O_EXCL descriptor exists.  If a
        # stream truncates before the leaf is fsynced/fchmod'd, rollback still
        # has to remove this partial *owned* file rather than leave a fresh
        # canonical root stranded behind.
        opened_identity = _directory_identity_from_stat(os.fstat(descriptor))
        self.created_leaf_identities[name] = opened_identity
        try:
            def write(block: bytes) -> None:
                view = memoryview(block)
                while view:
                    count = os.write(descriptor, view)
                    if count <= 0:
                        raise OSError("short local mirror write")
                    view = view[count:]

            reader.copy_next_leaf(
                expected_name=name,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
                write=write,
            )
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            final = os.fstat(descriptor)
            if not stat.S_ISREG(final.st_mode) or stat.S_IMODE(final.st_mode) != 0o444:
                raise FullResultImportError("local imported mirror leaf mode drift")
        finally:
            os.close(descriptor)
        os.fsync(self.descriptor)
        self._assert_identity()

    def validate_complete(self, *, leaves: Mapping[str, Mapping[str, object]], body_sha256s: Mapping[str, str]) -> None:
        self._assert_identity()
        names = tuple(sorted(os.listdir(self.descriptor)))
        if set(names) != set(remote_leaf_names()):
            raise FullResultImportError("local imported mirror topology/partial publication drift")
        if set(self.created_leaf_identities) != set(remote_leaf_names()):
            raise FullResultImportError("local imported mirror missing owned leaf publication")
        for name in remote_leaf_names():
            expected = leaves[name]
            before = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o444
                or int(before.st_size) != expected["bytes"]
                or _directory_identity_from_stat(before) != self.created_leaf_identities[name]
            ):
                raise FullResultImportError("local imported mirror leaf type/mode/size drift")
            descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.descriptor)
            try:
                body = _read_all(descriptor)
            finally:
                os.close(descriptor)
            if _digest(body) != expected["sha256"]:
                raise FullResultImportError("local imported mirror leaf SHA drift")
        for body_name in full_training_artifact_names():
            sidecar_name = f"{body_name}.sha256"
            descriptor = os.open(sidecar_name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.descriptor)
            try:
                sidecar = _read_all(descriptor)
            finally:
                os.close(descriptor)
            expected_sidecar = f"{body_sha256s[body_name]}  {body_name}\n".encode("ascii")
            if sidecar != expected_sidecar:
                raise FullResultImportError("local imported mirror sidecar content drift")
        self._assert_identity()

    def rollback(self) -> None:
        """Remove only a fresh owned root with only leaves this transaction wrote."""
        if self._closed:
            return
        try:
            self._assert_identity()
            current = set(os.listdir(self.descriptor))
            if not current.issubset(self.created_leaf_identities):
                raise FullResultImportError("cannot rollback local mirror containing non-owned leaf")
            for name in sorted(current):
                leaf = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISREG(leaf.st_mode)
                    or stat.S_ISLNK(leaf.st_mode)
                    or _directory_identity_from_stat(leaf) != self.created_leaf_identities[name]
                ):
                    raise FullResultImportError("cannot rollback local mirror leaf whose ownership identity drifted")
                os.unlink(name, dir_fd=self.descriptor)
            os.fsync(self.descriptor)
            os.close(self.descriptor)
            self.descriptor = -1
            os.rmdir(self.destination.name, dir_fd=self.parent_descriptor)
            os.fsync(self.parent_descriptor)
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in (self.descriptor, self.parent_descriptor, self.repository_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        self.descriptor = -1
        self.parent_descriptor = -1
        self.repository_descriptor = -1


def _imported_full_mirror_provenance_payload(
    *,
    endpoint: RemoteEndpoint,
    terminal_authority: RemoteTerminalAuthority,
    body_sha256s: Mapping[str, str],
) -> dict[str, object]:
    names = full_training_artifact_names()
    if set(body_sha256s) != set(names):
        raise FullResultImportError("import provenance body map topology drift")
    artifact_map = {name: _sha(body_sha256s[name], f"import provenance artifact {name}") for name in names}
    body = {
        "schema": "posterior_carrier_completed_full_import_mirror_v1",
        "local_mirror_relative": LOCAL_MIRROR_RELATIVE,
        # Keep the compatibility field as the exact remote *absolute path*;
        # the separate attestation binds its host/user transport endpoint.
        "remote_result_root": endpoint.result_root,
        "remote_root_descriptor_identity": terminal_authority.result_identity.payload(),
        "remote_terminal_sha256": artifact_map["terminal.json"],
        "remote_artifact_sha256s": artifact_map,
        "imported_artifact_sha256s": artifact_map,
        "remote_full_closure_sha256": terminal_authority.full_closure_sha256,
        "copy_protocol": COPY_PROTOCOL,
        "source_stage_identity_rebuilt_locally": False,
    }
    return {**body, "provenance_sha256": _digest(_json(body))}


def validate_imported_full_mirror_provenance(value: Mapping[str, object]) -> dict[str, object]:
    """Exact compatibility validator for ``ImportedFullMirrorProvenance`` payloads."""
    expected = {
        "schema", "local_mirror_relative", "remote_result_root", "remote_root_descriptor_identity",
        "remote_terminal_sha256", "remote_artifact_sha256s", "imported_artifact_sha256s",
        "remote_full_closure_sha256", "copy_protocol", "source_stage_identity_rebuilt_locally", "provenance_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "posterior_carrier_completed_full_import_mirror_v1":
        raise FullResultImportError("imported full mirror provenance schema drift")
    if value.get("local_mirror_relative") != LOCAL_MIRROR_RELATIVE:
        raise FullResultImportError("imported full mirror local destination drift")
    endpoint = RemoteEndpoint()
    expected_remote_label = endpoint.result_root
    if value.get("remote_result_root") != expected_remote_label:
        raise FullResultImportError("imported full mirror remote-root provenance drift")
    identities = value.get("remote_root_descriptor_identity")
    if not isinstance(identities, list) or len(identities) != 2:
        raise FullResultImportError("imported full mirror remote identity schema drift")
    DirectoryIdentity(identities[0], identities[1])
    names = full_training_artifact_names()
    remote, imported = value.get("remote_artifact_sha256s"), value.get("imported_artifact_sha256s")
    if not isinstance(remote, Mapping) or not isinstance(imported, Mapping) or set(remote) != set(names) or set(imported) != set(names):
        raise FullResultImportError("imported full mirror artifact map topology drift")
    remote_map = {name: _sha(remote[name], f"remote imported artifact {name}") for name in names}
    imported_map = {name: _sha(imported[name], f"local imported artifact {name}") for name in names}
    if (
        remote_map != imported_map
        or value.get("remote_terminal_sha256") != remote_map["terminal.json"]
        or value.get("copy_protocol") != COPY_PROTOCOL
        or value.get("source_stage_identity_rebuilt_locally") is not False
    ):
        raise FullResultImportError("imported full mirror provenance graph drift")
    _sha(value.get("remote_full_closure_sha256"), "imported full mirror closure SHA")
    body = {key: value[key] for key in expected if key != "provenance_sha256"}
    if value.get("provenance_sha256") != _digest(_json(body)):
        raise FullResultImportError("imported full mirror provenance canonical digest drift")
    return dict(value)


def _remote_import_attestation_payload(
    *,
    endpoint: RemoteEndpoint,
    terminal_authority: RemoteTerminalAuthority,
    host_key_pin: HostKeyPin,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    checked_provenance = validate_imported_full_mirror_provenance(provenance)
    body = {
        "schema": REMOTE_ATTESTATION_SCHEMA,
        "endpoint": endpoint.payload(),
        "host_key_pin": host_key_pin.payload(endpoint=endpoint),
        "stage_descriptor_identity": terminal_authority.stage_identity.payload(),
        "result_descriptor_identity": terminal_authority.result_identity.payload(),
        "terminal_authority": terminal_authority.payload(),
        "imported_full_mirror_provenance": checked_provenance,
        "copy_protocol": COPY_PROTOCOL,
        "canonical_mirror_contains_only_completed_training_body_sidecar_pairs": True,
    }
    return {**body, "attestation_sha256": _digest(_json(body))}


def validate_remote_import_attestation(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema", "endpoint", "host_key_pin", "stage_descriptor_identity", "result_descriptor_identity",
        "terminal_authority", "imported_full_mirror_provenance", "copy_protocol",
        "canonical_mirror_contains_only_completed_training_body_sidecar_pairs", "attestation_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != REMOTE_ATTESTATION_SCHEMA:
        raise FullResultImportError("remote import attestation schema drift")
    endpoint = RemoteEndpoint()
    if value.get("endpoint") != endpoint.payload():
        raise FullResultImportError("remote import attestation endpoint drift")
    host_key = value.get("host_key_pin")
    expected_host_key_keys = {
        "host_alias", "fingerprint", "known_hosts_line_sha256", "live_known_hosts_line_required",
    }
    if (
        not isinstance(host_key, Mapping)
        or set(host_key) != expected_host_key_keys
        or host_key.get("host_alias") != endpoint.host_alias
        or host_key.get("fingerprint") != endpoint.host_key_fingerprint
        or host_key.get("live_known_hosts_line_required") is not True
        or (
            host_key.get("known_hosts_line_sha256") is not None
            and not isinstance(host_key.get("known_hosts_line_sha256"), str)
        )
    ):
        raise FullResultImportError("remote import attestation host-key pin drift")
    if isinstance(host_key.get("known_hosts_line_sha256"), str):
        _sha(host_key["known_hosts_line_sha256"], "remote import known-hosts line SHA")
    stage, result = value.get("stage_descriptor_identity"), value.get("result_descriptor_identity")
    if not isinstance(stage, list) or not isinstance(result, list) or len(stage) != 2 or len(result) != 2:
        raise FullResultImportError("remote import attestation directory identity drift")
    authority = value.get("terminal_authority")
    if not isinstance(authority, Mapping):
        raise FullResultImportError("remote import terminal authority schema drift")
    terminal_authority = RemoteTerminalAuthority(
        stage_identity=DirectoryIdentity(stage[0], stage[1]),
        result_identity=DirectoryIdentity(result[0], result[1]),
        terminal_sha256=authority.get("terminal_sha256"),
        full_closure_sha256=authority.get("full_closure_sha256"),
    )
    if authority != terminal_authority.payload():
        raise FullResultImportError("remote import terminal authority cross-binding drift")
    provenance = validate_imported_full_mirror_provenance(value.get("imported_full_mirror_provenance"))
    if (
        provenance["remote_root_descriptor_identity"] != result
        or provenance["remote_terminal_sha256"] != terminal_authority.terminal_sha256
        or provenance["remote_full_closure_sha256"] != terminal_authority.full_closure_sha256
        or value.get("copy_protocol") != COPY_PROTOCOL
        or value.get("canonical_mirror_contains_only_completed_training_body_sidecar_pairs") is not True
    ):
        raise FullResultImportError("remote import attestation provenance binding drift")
    body = {key: value[key] for key in expected if key != "attestation_sha256"}
    if value.get("attestation_sha256") != _digest(_json(body)):
        raise FullResultImportError("remote import attestation canonical digest drift")
    return dict(value)


@dataclass(frozen=True)
class ImportResult:
    """Future root-only import output; neither payload is published here."""

    destination: Path
    provenance: Mapping[str, object]
    attestation: Mapping[str, object]


def import_completed_remote_full_result(
    root: Path,
    *,
    capability: RootReviewedImportCapability | None,
    transport: RemoteTransport | None,
) -> ImportResult:
    """Copy a terminal remote result only under a root-reviewed capability.

    The remote manifest is semantically accepted before a fresh local mirror
    directory is reserved.  Any subsequent stream or local publication
    failure rolls back only leaves in the just-created owned destination.
    """
    if capability is None or transport is None:
        raise FullResultImportError("remote import is dry until an in-process root-reviewed capability and transport exist")
    capability.validate()
    LocalMirrorTransaction.assert_fresh(Path(root))
    handle = transport.open_completed_terminal_stream(
        endpoint=capability.endpoint,
        terminal_authority=capability.terminal_authority,
        host_key_pin=capability.host_key_pin,
    )
    transaction: LocalMirrorTransaction | None = None
    remote_success = False
    try:
        reader = FramedSnapshotReader(handle.stream)
        manifest = reader.read_manifest()
        checked = validate_remote_snapshot_manifest(
            manifest,
            endpoint=capability.endpoint,
            terminal_authority=capability.terminal_authority,
        )
        # Only an accepted remote terminal graph is allowed to reserve the
        # canonical local root.  No partial or failed remote result gets a
        # durable destination directory.
        transaction = LocalMirrorTransaction.reserve(Path(root))
        for name in remote_leaf_names():
            leaf = checked["leaves"][name]
            transaction.publish_from_stream(
                name=name,
                expected_bytes=int(leaf["bytes"]),
                expected_sha256=str(leaf["sha256"]),
                reader=reader,
            )
        reader.ensure_eof()
        transaction.validate_complete(leaves=checked["leaves"], body_sha256s=checked["artifact_sha256s"])
        provenance = _imported_full_mirror_provenance_payload(
            endpoint=capability.endpoint,
            terminal_authority=capability.terminal_authority,
            body_sha256s=checked["artifact_sha256s"],
        )
        attestation = _remote_import_attestation_payload(
            endpoint=capability.endpoint,
            terminal_authority=capability.terminal_authority,
            host_key_pin=capability.host_key_pin,
            provenance=provenance,
        )
        # A transport is successful only once the streamed bytes have also
        # passed the local immutable body/sidecar topology verification.  If
        # local validation fails, the remote finalizer is told failure and the
        # owned fresh root is rolled back below.
        handle.close(success=True)
        remote_success = True
        transaction.close()
        return ImportResult(destination=Path(root).absolute() / LOCAL_MIRROR_RELATIVE, provenance=provenance, attestation=attestation)
    except BaseException:
        if transaction is not None:
            try:
                transaction.rollback()
            except BaseException:
                # Preserve the original failure; an identity-safe rollback may
                # itself refuse to remove a directory altered by another actor.
                pass
        if not remote_success:
            try:
                handle.close(success=False)
            except BaseException:
                pass
        raise


def dry_plan() -> dict[str, object]:
    """Static no-SSH/no-write statement for the public import CLI."""
    endpoint = RemoteEndpoint()
    return {
        "schema": IMPORT_SCHEMA,
        "cell": CELL,
        "phase": IMPORT_PHASE,
        "status": "DRY_ONLY__NO_SSH_NO_REMOTE_RESULT_NO_CHECKPOINT_NO_DATA_NO_CUDA_NO_WRITE",
        "endpoint": endpoint.payload(),
        "destination_relative": LOCAL_MIRROR_RELATIVE,
        "topology": list(full_training_artifact_names()),
        "boundaries": {
            "root_capability_required": True,
            "remote_terminal_identity_must_be_freshly_supplied": True,
            "host_key_pinned": True,
            "batch_mode": True,
            "ssh_agent_mutation": False,
            "target_opened": False,
            "data_opened": False,
            "checkpoint_tensor_loaded": False,
            "cuda_initialized": False,
            "canonical_mirror_created": False,
        },
    }


def execute_authorized(
    *,
    capability: RootReviewedImportCapability | None = None,
    transport: RemoteTransport | None = None,
    root: Path | None = None,
) -> ImportResult:
    """Fail closed for public callers; tests/root must call the typed function."""
    if capability is None or transport is None or root is None:
        raise FullResultImportError("remote import is dry until a separate root-reviewed capability and transport are supplied")
    return import_completed_remote_full_result(Path(root), capability=capability, transport=transport)


def _remote_python_program(*, endpoint: RemoteEndpoint, terminal_authority: RemoteTerminalAuthority) -> str:
    """Build a fixed remote Python program without a shell-interpolated path.

    The SSH invocation below sends this source to ``python3 -``.  Every route
    value is a validated literal embedded in a JSON string, not a shell word;
    the program traverses the stage and result directories with held
    ``O_NOFOLLOW`` descriptors and sends the framed manifest first.
    """
    config = {
        "endpoint": endpoint.payload(),
        "stage_identity": terminal_authority.stage_identity.payload(),
        "result_identity": terminal_authority.result_identity.payload(),
        "terminal_sha256": terminal_authority.terminal_sha256,
        "full_closure_sha256": terminal_authority.full_closure_sha256,
        "topology": list(full_training_artifact_names()),
        "leaves": list(remote_leaf_names()),
        "magic": FRAME_MAGIC.decode("ascii"),
    }
    # This program deliberately validates only the terminal graph needed to
    # decide whether bytes may leave the remote host.  The local scorer's
    # closure-bound full semantic validator repeats the complete 48-epoch,
    # checkpoint, and SWA validation after exact byte import.
    return """import hashlib, json, os, stat, struct, sys
CONFIG = json.loads(%r)
FLAGS = os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_NOFOLLOW', 0)
LEAF_FLAGS = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
CHUNK = 1 << 20

def fail(message):
    raise RuntimeError(message)

def ident(fd):
    value = os.fstat(fd)
    return [int(value.st_dev), int(value.st_ino)]

def open_abs_dir(path):
    current = os.open('/', FLAGS)
    try:
        for piece in [item for item in path.split('/') if item]:
            child = os.open(piece, FLAGS, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise

def open_rel_dir(parent, relative):
    current = os.dup(parent)
    try:
        for piece in relative.split('/'):
            child = os.open(piece, FLAGS, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise

def leaf_meta(root_fd, name):
    before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444:
        fail('remote leaf type/mode drift')
    fd = os.open(name, LEAF_FLAGS, dir_fd=root_fd)
    try:
        opened = os.fstat(fd)
        if (int(opened.st_dev), int(opened.st_ino), int(opened.st_size)) != (int(before.st_dev), int(before.st_ino), int(before.st_size)):
            fail('remote leaf identity drift')
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, CHUNK)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(fd)
    after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    if (int(after.st_dev), int(after.st_ino), int(after.st_size)) != (int(before.st_dev), int(before.st_ino), int(before.st_size)):
        fail('remote leaf changed during digest')
    return {'name': name, 'bytes': int(before.st_size), 'sha256': digest.hexdigest(), 'kind': 'regular', 'mode': '0444', 'identity': [int(before.st_dev), int(before.st_ino)]}

def leaf_bytes(root_fd, metadata):
    name = metadata['name']
    before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or [int(before.st_dev), int(before.st_ino)] != metadata['identity']
            or int(before.st_size) != metadata['bytes']):
        fail('remote leaf identity changed before stream')
    fd = os.open(name, LEAF_FLAGS, dir_fd=root_fd)
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != 0o444
                or [int(opened.st_dev), int(opened.st_ino)] != metadata['identity']
                or int(opened.st_size) != metadata['bytes']):
            fail('remote leaf descriptor identity changed before stream')
        return fd
    except BaseException:
        os.close(fd)
        raise

def emit(kind, name, payload):
    meta = json.dumps({'kind': kind, 'name': name, 'bytes': len(payload)}, sort_keys=True, separators=(',', ':')).encode('ascii')
    sys.stdout.buffer.write(struct.pack('>I', len(meta)))
    sys.stdout.buffer.write(meta)
    sys.stdout.buffer.write(payload)

stage = open_abs_dir(CONFIG['endpoint']['stage_root'])
try:
    if ident(stage) != CONFIG['stage_identity']:
        fail('remote stage descriptor identity drift')
    result = open_rel_dir(stage, CONFIG['endpoint']['result_relative_to_stage'])
    try:
        if ident(result) != CONFIG['result_identity']:
            fail('remote result descriptor identity drift')
        expected_leaves = CONFIG['leaves']
        if set(os.listdir(result)) != set(expected_leaves):
            fail('remote topology/failure/extra leaf drift')
        metadata = [leaf_meta(result, name) for name in expected_leaves]
        by_name = {item['name']: item for item in metadata}
        bodies = {name: by_name[name]['sha256'] for name in CONFIG['topology']}
        for body_name in CONFIG['topology']:
            sidecar_name = body_name + '.sha256'
            fd = leaf_bytes(result, by_name[sidecar_name])
            try:
                sidecar = b''
                while True:
                    block = os.read(fd, CHUNK)
                    if not block:
                        break
                    sidecar += block
            finally:
                os.close(fd)
            if sidecar != (bodies[body_name] + '  ' + body_name + '\\n').encode('ascii'):
                fail('remote sidecar content drift')
        fd = leaf_bytes(result, by_name['terminal.json'])
        try:
            terminal_body = b''
            while True:
                block = os.read(fd, CHUNK)
                if not block:
                    break
                terminal_body += block
        finally:
            os.close(fd)
        terminal = json.loads(terminal_body)
        expected_terminal_keys = {
                'schema', 'cell', 'phase', 'spec', 'identity', 'final_identity', 'attempt_sha256', 'launch_sha256',
                'source_authority_sha256', 'artifact_sha256s', 'checkpoint_state_sha256s', 'swa_state_sha256',
                'swa_proof', 'full_launch_closure', 'full_final_closure', 'phase_b_v3_launch_closure',
                'phase_b_v3_final_closure', 'boundaries', 'status'}
        if (not isinstance(terminal, dict)
                or set(terminal) != expected_terminal_keys
                or hashlib.sha256(terminal_body).hexdigest() != CONFIG['terminal_sha256']
                or terminal.get('schema') != 'posterior_carrier_full_train_terminal_v1'
                or terminal.get('cell') != 'POSTERIOR_CARRIER_BUDGETMIX_D_SEED42'
                or terminal.get('phase') != 'POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_FULL_TRAIN_V1'
                or not isinstance(terminal.get('spec'), dict)
                or not isinstance(terminal.get('identity'), dict)
                or terminal.get('final_identity') != terminal.get('identity')
                or terminal.get('status') != 'FULL_TRAINING_COMPLETE__SOURCE_ONLY__AWAITING_SEPARATE_SCORER'
                or terminal.get('attempt_sha256') != bodies['attempt.json']
                or terminal.get('launch_sha256') != bodies['launch.json']
                or terminal.get('source_authority_sha256') != bodies['source_authority.json']
                or terminal.get('artifact_sha256s') != {name: bodies[name] for name in CONFIG['topology'] if name != 'terminal.json'}
                or terminal.get('full_launch_closure', {}).get('closure_sha256') != CONFIG['full_closure_sha256']
                or terminal.get('full_final_closure', {}).get('closure_sha256') != CONFIG['full_closure_sha256']
                or terminal.get('full_launch_closure') != terminal.get('full_final_closure')
                or terminal.get('phase_b_v3_launch_closure') != terminal.get('phase_b_v3_final_closure')
                or not isinstance(terminal.get('phase_b_v3_launch_closure'), dict)
                or not isinstance(terminal.get('phase_b_v3_launch_closure', {}).get('closure_sha256'), str)
                or len(terminal.get('phase_b_v3_launch_closure', {}).get('closure_sha256', '')) != 64
                or not isinstance(terminal.get('checkpoint_state_sha256s'), dict)
                or set(terminal.get('checkpoint_state_sha256s', {})) != {'44', '45', '46', '47'}
                or not all(isinstance(item, str) and len(item) == 64
                           for item in terminal.get('checkpoint_state_sha256s', {}).values())
                or not isinstance(terminal.get('swa_state_sha256'), str)
                or len(terminal.get('swa_state_sha256', '')) != 64
                or not isinstance(terminal.get('swa_proof'), dict)
                or terminal.get('boundaries') != {
                    'source_only': True, 'target_opened': False, 'within_opened': False,
                    'external_opened': False, 'formal_opened': False, 'h1_opened': False,
                    'target_optimizer_steps': 0, 'target_backward_calls': 0,
                    'target_update_calls': 0, 'scientific_score': False,
                    'cache_read_or_write': False}):
            fail('remote terminal graph/closure drift')
        manifest = {
            'schema': 'posterior_carrier_remote_full_snapshot_manifest_v1',
            'endpoint': CONFIG['endpoint'],
            'stage_descriptor_identity': CONFIG['stage_identity'],
            'result_descriptor_identity': CONFIG['result_identity'],
            'topology': CONFIG['topology'],
            'artifact_sha256s': bodies,
            'leaves': [{key: item[key] for key in ('name','bytes','sha256','kind','mode')} for item in metadata],
            'terminal': terminal,
            'terminal_sha256': bodies['terminal.json'],
            'full_closure_sha256': CONFIG['full_closure_sha256'],
            'status': 'REMOTE_FULL_TERMINAL_SNAPSHOT_READY',
        }
        sys.stdout.buffer.write(CONFIG['magic'].encode('ascii'))
        emit('manifest', None, json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('ascii'))
        for item in metadata:
            meta = json.dumps({'kind': 'leaf', 'name': item['name'], 'bytes': item['bytes']}, sort_keys=True, separators=(',', ':')).encode('ascii')
            sys.stdout.buffer.write(struct.pack('>I', len(meta)))
            sys.stdout.buffer.write(meta)
            fd = leaf_bytes(result, item)
            try:
                remaining = item['bytes']
                while remaining:
                    block = os.read(fd, min(CHUNK, remaining))
                    if not block:
                        fail('remote leaf truncated during stream')
                    sys.stdout.buffer.write(block)
                    remaining -= len(block)
            finally:
                os.close(fd)
        if ident(stage) != CONFIG['stage_identity'] or ident(result) != CONFIG['result_identity']:
            fail('remote held directory identity drift after stream')
        sys.stdout.buffer.flush()
    finally:
        os.close(result)
finally:
    os.close(stage)
""" % json.dumps(config, sort_keys=True)


class SshRemoteTransport:
    """Future real transport; construction is inert and public dry code never instantiates it."""

    def __init__(self, *, popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen) -> None:
        self._popen_factory = popen_factory

    @staticmethod
    def command(*, endpoint: RemoteEndpoint, known_hosts_path: str) -> tuple[str, ...]:
        """No shell interpolation, batch mode, host-key pin, and no agent mutation."""
        return (
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={known_hosts_path}",
            "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "UpdateHostKeys=no",
            "-o", "IdentityAgent=none",
            "-o", "IdentitiesOnly=yes",
            "-o", "AddKeysToAgent=no",
            "-o", "ForwardAgent=no",
            "-o", "ClearAllForwardings=yes",
            "-o", "ControlMaster=no",
            f"{endpoint.user}@{endpoint.host_alias}",
            "python3",
            "-",
        )

    def open_completed_terminal_stream(
        self,
        *,
        endpoint: RemoteEndpoint,
        terminal_authority: RemoteTerminalAuthority,
        host_key_pin: HostKeyPin,
    ) -> TransportHandle:
        known_hosts_line = host_key_pin.require_live_known_hosts_line(endpoint=endpoint)
        descriptor, known_hosts_path = tempfile.mkstemp(prefix="posterior-carrier-hostkey-", text=True)
        process: subprocess.Popen[bytes] | None = None
        try:
            _write_all(descriptor, (known_hosts_line + "\n").encode("utf-8"))
            os.fchmod(descriptor, 0o600)
            os.close(descriptor)
            descriptor = -1
            process = self._popen_factory(
                self.command(endpoint=endpoint, known_hosts_path=known_hosts_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
            )
            if process.stdin is None or process.stdout is None or process.stderr is None:
                raise FullResultImportError("SSH transport did not expose required stdio pipes")
            program = _remote_python_program(endpoint=endpoint, terminal_authority=terminal_authority).encode("utf-8")
            process.stdin.write(program)
            process.stdin.close()

            def finalize(success: bool) -> None:
                try:
                    if not success:
                        # On a local framed-stream or publication failure the
                        # remote program may still be blocked writing stdout.
                        # Close that pipe and terminate this one fresh SSH
                        # child instead of waiting indefinitely or leaving it
                        # alive.  The caller preserves its original failure.
                        try:
                            process.stdout.close()
                        except OSError:
                            pass
                        try:
                            process.terminate()
                        except OSError:
                            pass
                    stderr = process.stderr.read()
                    code = process.wait()
                    if success and code != 0:
                        raise FullResultImportError(
                            f"remote SSH snapshot program failed with exit={code}; stderr_sha256={_digest(stderr)}"
                        )
                finally:
                    try:
                        process.stdout.close()
                    except OSError:
                        pass
                    try:
                        process.stderr.close()
                    except OSError:
                        pass
                    try:
                        os.unlink(known_hosts_path)
                    except FileNotFoundError:
                        pass

            return TransportHandle(stream=process.stdout, finalize=finalize)
        except BaseException:
            if process is not None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                except OSError:
                    pass
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=5)
                except BaseException:
                    pass
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                os.unlink(known_hosts_path)
            except FileNotFoundError:
                pass
            raise
