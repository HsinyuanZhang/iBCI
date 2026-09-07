"""Descriptor-safe physical substrate for the Posterior Carrier scorer.

This module is deliberately inert at import time: it has no Torch, NumPy,
NWB, CUDA, cache, or result-root imports.  It contains the two parts that must
be auditable before a future reviewed score launch:

* immutable, held-FD readers for a copied completed remote full-training
  result and the sealed Cell-D artifacts; and
* a deferred runtime backend factory.  The factory is reachable only after
  the main scorer has published an attempt and validated a root authorization.

The imported full result is *not* re-identified from the local score stage.
Its remote descriptor identity remains a provenance fact in an immutable
import receipt; the local mirror is accepted only when every copied byte is
identical to that receipt's explicit artifact map.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class PhysicalScoreError(RuntimeError):
    """Fail closed before an unbound artifact, asset, or tensor is consumed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalScoreError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PhysicalScoreError(f"{label} must be an exact lowercase SHA-256")
    return value


def _safe_relative(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value or Path(value).is_absolute()
            or ".." in Path(value).parts or value in {".", ".."}):
        raise PhysicalScoreError(f"{label} must be a safe relative path")
    return value


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _identity(info: os.stat_result) -> tuple[int, int, int]:
    return (int(info.st_dev), int(info.st_ino), int(info.st_size))


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise PhysicalScoreError(f"cannot lstat immutable directory: {path}") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PhysicalScoreError("immutable directory must be a real non-symlink")
    return (int(info.st_dev), int(info.st_ino))


@dataclass(frozen=True)
class ImmutableFile:
    """Body loaded through one held immutable directory descriptor."""

    name: str
    body: bytes
    sha256: str
    identity: tuple[int, int, int]

    def json_object(self) -> dict[str, object]:
        try:
            value = json.loads(self.body)
        except (TypeError, json.JSONDecodeError) as error:
            raise PhysicalScoreError(f"immutable JSON decode drift: {self.name}") from error
        if not isinstance(value, dict):
            raise PhysicalScoreError(f"immutable JSON root must be object: {self.name}")
        return value


@dataclass
class ImmutableDirectory:
    """A held directory FD plus parent/name identity for a finite receipt tree."""

    directory: Path
    parent: Path
    identity: tuple[int, int]
    parent_identity: tuple[int, int]
    descriptor: int
    parent_descriptor: int
    _closed: bool = False

    @classmethod
    def open(cls, directory: Path) -> "ImmutableDirectory":
        path = Path(directory).absolute()
        parent = path.parent
        identity = _directory_identity(path)
        parent_identity = _directory_identity(parent)
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        try:
            parent_fd = os.open(parent, flags)
            fd = os.open(path.name, flags, dir_fd=parent_fd)
        except OSError as error:
            try:
                os.close(parent_fd)
            except (UnboundLocalError, OSError):
                pass
            raise PhysicalScoreError("cannot O_NOFOLLOW open immutable directory") from error
        instance = cls(path, parent, identity, parent_identity, fd, parent_fd)
        try:
            instance.reverify()
            return instance
        except BaseException:
            instance.close()
            raise

    def reverify(self) -> None:
        if self._closed:
            raise PhysicalScoreError("immutable directory has already been closed")
        held = os.fstat(self.descriptor)
        parent_held = os.fstat(self.parent_descriptor)
        if (
            not stat.S_ISDIR(held.st_mode)
            or not stat.S_ISDIR(parent_held.st_mode)
            or (int(held.st_dev), int(held.st_ino)) != self.identity
            or (int(parent_held.st_dev), int(parent_held.st_ino)) != self.parent_identity
            or _directory_identity(self.directory) != self.identity
            or _directory_identity(self.parent) != self.parent_identity
        ):
            raise PhysicalScoreError("immutable directory/parent identity drift")
        try:
            named = os.stat(self.directory.name, dir_fd=self.parent_descriptor, follow_symlinks=False)
        except OSError as error:
            raise PhysicalScoreError("immutable directory named identity disappeared") from error
        if (
            not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (int(named.st_dev), int(named.st_ino)) != self.identity
        ):
            raise PhysicalScoreError("immutable directory named identity drift")

    def names(self) -> tuple[str, ...]:
        self.reverify()
        try:
            names = tuple(sorted(os.listdir(self.descriptor)))
        except OSError as error:
            raise PhysicalScoreError("cannot enumerate immutable receipt topology") from error
        self.reverify()
        return names

    def _read_leaf(self, name: str, *, expected_mode: int = 0o444) -> ImmutableFile:
        if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
            raise PhysicalScoreError("immutable leaf name is malformed")
        self.reverify()
        try:
            before = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
            fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.descriptor)
        except OSError as error:
            raise PhysicalScoreError(f"immutable leaf missing/inaccessible: {name}") from error
        try:
            opened = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or stat.S_IMODE(before.st_mode) != expected_mode
                or not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != expected_mode
                or _identity(before) != _identity(opened)
            ):
                raise PhysicalScoreError(f"immutable leaf type/mode/identity drift: {name}")
            body = _read_all(fd)
        finally:
            os.close(fd)
        try:
            after = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except OSError as error:
            raise PhysicalScoreError(f"immutable leaf vanished during read: {name}") from error
        if _identity(after) != _identity(before) or len(body) != int(before.st_size):
            raise PhysicalScoreError(f"immutable leaf changed during read: {name}")
        self.reverify()
        return ImmutableFile(name=name, body=body, sha256=_digest(body), identity=_identity(before))

    def read_pair(self, name: str, *, expected_sha256: str | None = None) -> ImmutableFile:
        body = self._read_leaf(name)
        if expected_sha256 is not None and body.sha256 != _sha(expected_sha256, f"expected SHA {name}"):
            raise PhysicalScoreError(f"immutable artifact body SHA drift: {name}")
        sidecar = self._read_leaf(name + ".sha256")
        expected_sidecar = f"{body.sha256}  {name}\n".encode("ascii")
        if sidecar.body != expected_sidecar:
            raise PhysicalScoreError(f"immutable artifact sidecar drift: {name}")
        return body

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in (self.descriptor, self.parent_descriptor):
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "ImmutableDirectory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _repository_pair(root: Path, relative: str, *, expected_sha256: str) -> ImmutableFile:
    """Descriptor-read one nested repository file without trusting parent paths.

    Each path component is opened with ``O_NOFOLLOW`` from the repository root;
    this differs deliberately from a convenience ``Path.read_bytes`` call.
    """
    relative = _safe_relative(relative, "repository artifact relative")
    root_fd = -1
    held_dirs: list[int] = []
    try:
        base = Path(root).absolute()
        base_identity = _directory_identity(base)
        root_fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        if (int(os.fstat(root_fd).st_dev), int(os.fstat(root_fd).st_ino)) != base_identity:
            raise PhysicalScoreError("repository root identity drift before immutable artifact read")
        current_fd = root_fd
        pieces = Path(relative).parts
        for piece in pieces[:-1]:
            child = os.open(piece, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
            held_dirs.append(child)
            current_fd = child
        name = pieces[-1]

        def read_leaf(leaf: str) -> ImmutableFile:
            try:
                before = os.stat(leaf, dir_fd=current_fd, follow_symlinks=False)
                descriptor = os.open(leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
            except OSError as error:
                raise PhysicalScoreError(f"sealed artifact missing/inaccessible: {relative}") from error
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
                    or stat.S_IMODE(before.st_mode) != 0o444 or not stat.S_ISREG(opened.st_mode)
                    or stat.S_IMODE(opened.st_mode) != 0o444 or _identity(before) != _identity(opened)
                ):
                    raise PhysicalScoreError(f"sealed artifact type/mode drift: {relative}")
                body = _read_all(descriptor)
            finally:
                os.close(descriptor)
            after = os.stat(leaf, dir_fd=current_fd, follow_symlinks=False)
            if _identity(after) != _identity(before) or len(body) != int(before.st_size):
                raise PhysicalScoreError(f"sealed artifact changed during read: {relative}")
            return ImmutableFile(leaf, body, _digest(body), _identity(before))

        result = read_leaf(name)
        if result.sha256 != _sha(expected_sha256, "sealed artifact expected SHA"):
            raise PhysicalScoreError(f"sealed artifact SHA drift: {relative}")
        sidecar = read_leaf(name + ".sha256")
        if sidecar.body != f"{result.sha256}  {name}\n".encode("ascii"):
            raise PhysicalScoreError(f"sealed artifact sidecar drift: {relative}")
        if (int(os.fstat(root_fd).st_dev), int(os.fstat(root_fd).st_ino)) != base_identity:
            raise PhysicalScoreError("repository root identity drift after immutable artifact read")
        return result
    finally:
        for descriptor in reversed(held_dirs):
            os.close(descriptor)
        if root_fd >= 0:
            os.close(root_fd)


def _repository_regular(
    root: Path,
    relative: str,
    *,
    expected_sha256: str,
    expected_mode: int,
) -> ImmutableFile:
    """Descriptor-read one explicitly named non-sidecar authority file.

    The paired-view manifest is intentionally mode ``0600`` and has no SHA
    sidecar.  The fixed external ledger and scope have their own immutable
    receipts rather than a caller-supplied asset table.  This helper gives all
    three the same no-follow, parent-identity, exact-mode treatment without
    pretending that the manifest has a sidecar it does not own.
    """
    if expected_mode not in {0o444, 0o600}:
        raise PhysicalScoreError("fixed authority expected mode is unsupported")
    relative = _safe_relative(relative, "fixed authority relative")
    root_fd = -1
    held_dirs: list[tuple[int, tuple[int, int], int, str]] = []
    try:
        base = Path(root).absolute()
        base_identity = _directory_identity(base)
        root_fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        if (int(os.fstat(root_fd).st_dev), int(os.fstat(root_fd).st_ino)) != base_identity:
            raise PhysicalScoreError("repository root identity drift before fixed-authority read")
        current_fd = root_fd
        pieces = Path(relative).parts
        for piece in pieces[:-1]:
            try:
                before_child = os.stat(piece, dir_fd=current_fd, follow_symlinks=False)
                child = os.open(piece, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
            except OSError as error:
                raise PhysicalScoreError(f"fixed authority parent missing/aliased: {relative}") from error
            opened_child = os.fstat(child)
            child_identity = (int(before_child.st_dev), int(before_child.st_ino))
            if (
                not stat.S_ISDIR(before_child.st_mode) or stat.S_ISLNK(before_child.st_mode)
                or not stat.S_ISDIR(opened_child.st_mode)
                or (int(opened_child.st_dev), int(opened_child.st_ino)) != child_identity
            ):
                os.close(child)
                raise PhysicalScoreError(f"fixed authority parent identity drift: {relative}")
            held_dirs.append((child, child_identity, current_fd, piece))
            current_fd = child
        name = pieces[-1]
        try:
            before = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
        except OSError as error:
            raise PhysicalScoreError(f"fixed authority missing/inaccessible: {relative}") from error
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
                or stat.S_IMODE(before.st_mode) != expected_mode
                or not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != expected_mode
                or _identity(before) != _identity(opened)
            ):
                raise PhysicalScoreError(f"fixed authority type/mode/identity drift: {relative}")
            body = _read_all(descriptor)
        finally:
            os.close(descriptor)
        try:
            after = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
        except OSError as error:
            raise PhysicalScoreError(f"fixed authority vanished during read: {relative}") from error
        if _identity(after) != _identity(before) or len(body) != int(before.st_size):
            raise PhysicalScoreError(f"fixed authority changed during read: {relative}")
        if _digest(body) != _sha(expected_sha256, "fixed authority expected SHA"):
            raise PhysicalScoreError(f"fixed authority SHA drift: {relative}")
        for child, child_identity, parent_fd, piece in held_dirs:
            held_child = os.fstat(child)
            try:
                named_child = os.stat(piece, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as error:
                raise PhysicalScoreError(f"fixed authority parent vanished after read: {relative}") from error
            if (
                not stat.S_ISDIR(held_child.st_mode) or not stat.S_ISDIR(named_child.st_mode)
                or stat.S_ISLNK(named_child.st_mode)
                or (int(held_child.st_dev), int(held_child.st_ino)) != child_identity
                or (int(named_child.st_dev), int(named_child.st_ino)) != child_identity
            ):
                raise PhysicalScoreError(f"fixed authority parent named identity drift: {relative}")
        if (int(os.fstat(root_fd).st_dev), int(os.fstat(root_fd).st_ino)) != base_identity:
            raise PhysicalScoreError("repository root identity drift after fixed-authority read")
        return ImmutableFile(name=name, body=body, sha256=_digest(body), identity=_identity(before))
    finally:
        for descriptor, _identity_value, _parent_fd, _piece in reversed(held_dirs):
            os.close(descriptor)
        if root_fd >= 0:
            os.close(root_fd)


# Fixed metadata authorities for the only two non-formal public surfaces.
# These literals are intentionally local to the physical route.  A caller may
# supply rosters in the score identity, but it may not supply asset UUIDs,
# file paths, byte sizes, or hashes as an authority.
STRICT_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE = (
    "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v1/c1_train_val_33_manifest.json"
)
WITHIN_PAIRED_VIEW_MANIFEST_SHA256 = "bb3440b688b6d16dabbf91db3ce43e91711f1e9e389de4b241e80a827fbcab7d"
WITHIN_PAIRED_VIEW_MANIFEST_MODE = 0o600
EXTERNAL_LEDGER_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_schema_preflight_v2/receipt.json"
EXTERNAL_LEDGER_SHA256 = "1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283"
EXTERNAL_SCOPE_RELATIVE = "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json"
EXTERNAL_SCOPE_SHA256 = "68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55"
EXTERNAL_AUTHORITY_MODE = 0o444
FORMAL_TEST_SESSION_NAMES = (
    "sub-C_ses-CO-20151113", "sub-C_ses-CO-20151116", "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119", "sub-C_ses-CO-20151120", "sub-C_ses-CO-20151201",
)
SEALED_WITHIN_PAIRED_VIEW_ROWS = (
    ("sub-C_ses-CO-20151103", "sub-C/sub-C_ses-CO-20151103_behavior+ecephys.nwb", 62_145_872,
     "7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7"),
    ("sub-C_ses-CO-20151104", "sub-C/sub-C_ses-CO-20151104_behavior+ecephys.nwb", 125_164_148,
     "7a5c0319414937ab9db5628701d3fe34db4ea876c730f77c3f5d8e9b593a65f1"),
    ("sub-C_ses-CO-20151106", "sub-C/sub-C_ses-CO-20151106_behavior+ecephys.nwb", 119_065_400,
     "7020f6430a66f857da13254b6a367bff6da81175ed8ad92eb272fb7deb18aeb6"),
    ("sub-C_ses-CO-20151109", "sub-C/sub-C_ses-CO-20151109_behavior+ecephys.nwb", 88_439_880,
     "7c4b484387c73b707bee067289de694857db475256e7848c1b345c2b41657e4e"),
    ("sub-C_ses-CO-20151110", "sub-C/sub-C_ses-CO-20151110_behavior+ecephys.nwb", 73_550_120,
     "c5a9c48cdd187ea945a48d111fa640734f02e33aee78b188e9d90ca8df7df5f2"),
    ("sub-C_ses-CO-20151112", "sub-C/sub-C_ses-CO-20151112_behavior+ecephys.nwb", 45_326_504,
     "1162d61afa85bcd33bc022dbeef2f34a7a421cfbc3d5b1f06864ca81bc6d74f2"),
)


@dataclass(frozen=True)
class FixedInputAuthoritySpec:
    """Literal metadata authorities used to derive—not accept—score assets.

    ``spec`` is injectable only for focused no-data tests.  Production calls
    use ``DEFAULT_FIXED_INPUT_AUTHORITIES`` below, whose row values and
    hashes are closure-bound literals.  The public scorer never receives a
    caller-defined asset mapping.
    """

    within_manifest_relative: str
    within_manifest_sha256: str
    within_manifest_mode: int
    strict_manifest_sha256: str
    within_rows: tuple[tuple[str, str, int, str], ...]
    external_ledger_relative: str
    external_ledger_sha256: str
    external_scope_relative: str
    external_scope_sha256: str
    external_authority_mode: int
    formal_test_sessions: tuple[str, ...]
    external_count: int

    def __post_init__(self) -> None:
        _safe_relative(self.within_manifest_relative, "within manifest relative")
        _safe_relative(self.external_ledger_relative, "external ledger relative")
        _safe_relative(self.external_scope_relative, "external scope relative")
        for label, value in (
            ("within manifest SHA", self.within_manifest_sha256),
            ("strict manifest SHA", self.strict_manifest_sha256),
            ("external ledger SHA", self.external_ledger_sha256),
            ("external scope SHA", self.external_scope_sha256),
        ):
            _sha(value, label)
        if self.within_manifest_mode != 0o600 or self.external_authority_mode != 0o444:
            raise PhysicalScoreError("fixed input authority mode topology drift")
        if self.external_count < 1 or not self.within_rows:
            raise PhysicalScoreError("fixed input authority row cardinality drift")
        sessions = tuple(row[0] for row in self.within_rows)
        if len(sessions) != len(set(sessions)):
            raise PhysicalScoreError("fixed within authority contains duplicate sessions")
        for session, frozen_path, byte_count, digest in self.within_rows:
            if (
                not isinstance(session, str) or not session
                or not isinstance(frozen_path, str) or Path(frozen_path).name != f"{session}_behavior+ecephys.nwb"
                or type(byte_count) is not int or byte_count <= 0
            ):
                raise PhysicalScoreError("fixed within authority row topology drift")
            _sha(digest, "fixed within authority file SHA")


DEFAULT_FIXED_INPUT_AUTHORITIES = FixedInputAuthoritySpec(
    within_manifest_relative=WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE,
    within_manifest_sha256=WITHIN_PAIRED_VIEW_MANIFEST_SHA256,
    within_manifest_mode=WITHIN_PAIRED_VIEW_MANIFEST_MODE,
    strict_manifest_sha256=STRICT_MANIFEST_SHA256,
    within_rows=SEALED_WITHIN_PAIRED_VIEW_ROWS,
    external_ledger_relative=EXTERNAL_LEDGER_RELATIVE,
    external_ledger_sha256=EXTERNAL_LEDGER_SHA256,
    external_scope_relative=EXTERNAL_SCOPE_RELATIVE,
    external_scope_sha256=EXTERNAL_SCOPE_SHA256,
    external_authority_mode=EXTERNAL_AUTHORITY_MODE,
    formal_test_sessions=FORMAL_TEST_SESSION_NAMES,
    external_count=15,
)


def _json_mapping(body: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise PhysicalScoreError(f"{label} JSON decode drift") from error
    if not isinstance(value, dict):
        raise PhysicalScoreError(f"{label} JSON root must be an object")
    return value


def _unique_rows(value: object, *, key: str, label: str) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, list):
        raise PhysicalScoreError(f"{label} must be a list")
    result: dict[str, Mapping[str, object]] = {}
    for row in value:
        if not isinstance(row, Mapping) or not isinstance(row.get(key), str) or not row[key]:
            raise PhysicalScoreError(f"{label} row/key drift")
        rendered = str(row[key])
        if rendered in result:
            raise PhysicalScoreError(f"{label} duplicate {key}")
        result[rendered] = row
    return result


def _manifest_within_assets(
    manifest_file: ImmutableFile,
    *,
    spec: FixedInputAuthoritySpec,
    within_roster: Sequence[str],
) -> tuple[dict[str, object], ...]:
    """Validate the complete C1-val semantics and return the six sealed rows."""
    manifest = _json_mapping(manifest_file.body, label="within paired-view manifest")
    expected_sessions = tuple(row[0] for row in spec.within_rows)
    expected_inventory = [
        {"session": session, "filename": Path(frozen_path).name, "bytes": byte_count, "sha256": digest}
        for session, frozen_path, byte_count, digest in spec.within_rows
    ]
    splits = manifest.get("session_splits")
    inventory = manifest.get("file_inventory")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("task") != "CO"
        or manifest.get("file_count") != 33
        or manifest.get("split_counts") != [27, len(expected_sessions), len(spec.formal_test_sessions)]
        or manifest.get("max_units_exclusive") != 100
        or manifest.get("source_manifest_sha256") != spec.strict_manifest_sha256
        or manifest.get("formal_test_file_hashes") != []
        or manifest.get("formal_test_file_paths") != []
        or manifest.get("formal_test_paths_resolved") is not False
        or not isinstance(splits, Mapping) or set(splits) != {"train", "val", "test"}
        or not isinstance(inventory, Mapping) or set(inventory) != {"train", "val"}
        or not isinstance(splits.get("train"), list) or len(splits["train"]) != 27
        or tuple(splits.get("val", ())) != expected_sessions
        or tuple(splits.get("test", ())) != spec.formal_test_sessions
        or not isinstance(inventory.get("train"), list) or len(inventory["train"]) != 27
        or inventory.get("val") != expected_inventory
        or tuple(within_roster) != expected_sessions
    ):
        raise PhysicalScoreError("within paired-view manifest semantic/roster drift")
    authority = {
        "kind": "c1_paired_view_manifest_val6",
        "manifest_relative": spec.within_manifest_relative,
        "manifest_sha256": spec.within_manifest_sha256,
        "manifest_mode": format(spec.within_manifest_mode, "04o"),
        "manifest_descriptor_identity": list(manifest_file.identity),
        "selected_split": "val",
        "manifest_read_once": True,
    }
    return tuple(
        {
            "schema": "posterior_carrier_matched_score_input_asset_v1",
            "surface": "within",
            "asset_id": f"c1_paired_view:{session}",
            "session": session,
            "frozen_path": frozen_path,
            "bytes": byte_count,
            "sha256": digest,
            "authority": authority,
        }
        for session, frozen_path, byte_count, digest in spec.within_rows
    )


def _external_ledger_assets(
    ledger_file: ImmutableFile,
    scope_file: ImmutableFile,
    *,
    spec: FixedInputAuthoritySpec,
    external_roster: Sequence[str],
) -> tuple[dict[str, object], ...]:
    """Resolve UUID eligibility through disposition, verified bytes, and scope."""
    ledger = _json_mapping(ledger_file.body, label="external fixed ledger")
    scope = _json_mapping(scope_file.body, label="external fixed scope")
    eligible_ids = ledger.get("eligible_session_ids")
    if (
        not isinstance(eligible_ids, list)
        or len(eligible_ids) != spec.external_count
        or len(set(eligible_ids)) != spec.external_count
        or any(not isinstance(item, str) or not item for item in eligible_ids)
        or ledger.get("eligible_session_count") != spec.external_count
    ):
        raise PhysicalScoreError("external eligible UUID topology drift")
    ledger_rows = _unique_rows(ledger.get("asset_disposition_ledger"), key="asset_id", label="external disposition ledger")
    downloads = _unique_rows(ledger.get("verified_downloads"), key="asset_id", label="external verified downloads")
    selected = _unique_rows(scope.get("selected_assets"), key="asset_id", label="external scope selected assets")
    shared_authority = {
        "kind": "dandi688_fixed_v2_uuid_ledger_join",
        "ledger_relative": spec.external_ledger_relative,
        "ledger_sha256": spec.external_ledger_sha256,
        "ledger_mode": format(spec.external_authority_mode, "04o"),
        "ledger_descriptor_identity": list(ledger_file.identity),
        "scope_relative": spec.external_scope_relative,
        "scope_sha256": spec.external_scope_sha256,
        "scope_mode": format(spec.external_authority_mode, "04o"),
        "scope_descriptor_identity": list(scope_file.identity),
        "eligible_session_ids_semantics": "asset_uuid_then_exact_disposition_verified_download_scope_join",
    }
    rows: list[dict[str, object]] = []
    for asset_id in eligible_ids:
        disposition = ledger_rows.get(asset_id)
        download = downloads.get(asset_id)
        scoped = selected.get(asset_id)
        if disposition is None or download is None or scoped is None:
            raise PhysicalScoreError("external UUID/disposition/download/scope join drift")
        session, frozen_path = disposition.get("session_id"), disposition.get("frozen_path")
        byte_count, digest = download.get("bytes"), download.get("sha256")
        if (
            disposition.get("asset_id") != asset_id
            or disposition.get("eligible") is not True
            or disposition.get("disposition") != "ELIGIBLE"
            or not isinstance(session, str) or not session or session in spec.formal_test_sessions
            or not isinstance(frozen_path, str) or Path(frozen_path).name != f"{session}_behavior+ecephys.nwb"
            or download.get("asset_id") != asset_id
            or scoped.get("asset_id") != asset_id
            or scoped.get("session_id") != session
            or scoped.get("path") != frozen_path
            or scoped.get("size") != byte_count
            or scoped.get("sha256") != digest
            or download.get("size_and_sha256_verified_before_nwb_open") is not True
            or type(byte_count) is not int or byte_count <= 0
        ):
            raise PhysicalScoreError("external fixed-ledger row semantic drift")
        rows.append({
            "schema": "posterior_carrier_matched_score_input_asset_v1",
            "surface": "external",
            "asset_id": asset_id,
            "session": session,
            "frozen_path": frozen_path,
            "bytes": byte_count,
            "sha256": _sha(digest, "external verified asset SHA"),
            "authority": shared_authority,
        })
    ordered = tuple(sorted(rows, key=lambda item: str(item["session"])))
    if tuple(item["session"] for item in ordered) != tuple(external_roster):
        raise PhysicalScoreError("external fixed-ledger roster/order differs from score identity")
    return ordered


def derive_fixed_input_assets(
    root: Path,
    *,
    within_roster: Sequence[str],
    external_roster: Sequence[str],
    spec: FixedInputAuthoritySpec = DEFAULT_FIXED_INPUT_AUTHORITIES,
) -> dict[str, list[dict[str, object]]]:
    """Descriptor-derive the only legal within6/external15 asset rows.

    This is deliberately the sole production construction path.  The caller
    supplies the reviewed roster labels only; all asset UUIDs, paths, byte
    counts and SHA-256 values are recovered from fixed immutable metadata.
    It opens no NWB path and creates no artifact.
    """
    if len(tuple(within_roster)) != len(spec.within_rows) or len(tuple(external_roster)) != spec.external_count:
        raise PhysicalScoreError("fixed input authority roster cardinality drift")
    manifest = _repository_regular(
        root, spec.within_manifest_relative,
        expected_sha256=spec.within_manifest_sha256, expected_mode=spec.within_manifest_mode,
    )
    ledger = _repository_regular(
        root, spec.external_ledger_relative,
        expected_sha256=spec.external_ledger_sha256, expected_mode=spec.external_authority_mode,
    )
    scope = _repository_regular(
        root, spec.external_scope_relative,
        expected_sha256=spec.external_scope_sha256, expected_mode=spec.external_authority_mode,
    )
    return {
        "within": list(_manifest_within_assets(manifest, spec=spec, within_roster=within_roster)),
        "external": list(_external_ledger_assets(ledger, scope, spec=spec, external_roster=external_roster)),
    }


def _identity_list(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 3 or any(type(item) is not int or item < 0 for item in value):
        raise PhysicalScoreError(f"{label} descriptor identity drift")
    return list(value)


def validate_fixed_input_assets(
    value: object,
    *,
    within_roster: Sequence[str],
    external_roster: Sequence[str],
    spec: FixedInputAuthoritySpec = DEFAULT_FIXED_INPUT_AUTHORITIES,
) -> dict[str, list[dict[str, object]]]:
    """Validate persisted derived rows; live re-derivation remains mandatory.

    This function intentionally validates the durable shape and immutable
    literals but does not treat an arbitrary caller mapping as authority.
    ``build_target_free_preflight`` and final authorization both call
    ``derive_fixed_input_assets`` and exact-compare this output.
    """
    if not isinstance(value, Mapping) or set(value) != {"within", "external"}:
        raise PhysicalScoreError("derived input-assets surface topology drift")
    expected_within = tuple(row[0] for row in spec.within_rows)
    if tuple(within_roster) != expected_within or len(tuple(external_roster)) != spec.external_count:
        raise PhysicalScoreError("derived input-assets roster contract drift")
    result: dict[str, list[dict[str, object]]] = {}
    for surface, roster in (("within", tuple(within_roster)), ("external", tuple(external_roster))):
        rows = value.get(surface)
        if not isinstance(rows, list) or len(rows) != len(roster):
            raise PhysicalScoreError("derived input-assets row cardinality drift")
        checked: list[dict[str, object]] = []
        for row in rows:
            expected = {"schema", "surface", "asset_id", "session", "frozen_path", "bytes", "sha256", "authority"}
            if not isinstance(row, Mapping) or set(row) != expected:
                raise PhysicalScoreError("derived input-assets row schema drift")
            if (
                row.get("schema") != "posterior_carrier_matched_score_input_asset_v1"
                or row.get("surface") != surface
                or not isinstance(row.get("asset_id"), str) or not row["asset_id"]
                or not isinstance(row.get("session"), str) or not row["session"]
                or not isinstance(row.get("frozen_path"), str)
                or Path(row["frozen_path"]).is_absolute() or ".." in Path(row["frozen_path"]).parts
                or Path(row["frozen_path"]).name != f"{row['session']}_behavior+ecephys.nwb"
                or type(row.get("bytes")) is not int or row["bytes"] <= 0
                or not isinstance(row.get("authority"), Mapping)
            ):
                raise PhysicalScoreError("derived input-assets row value drift")
            _sha(row.get("sha256"), "derived input-assets row SHA")
            authority = dict(row["authority"])
            if surface == "within":
                expected_authority = {
                    "kind": "c1_paired_view_manifest_val6",
                    "manifest_relative": spec.within_manifest_relative,
                    "manifest_sha256": spec.within_manifest_sha256,
                    "manifest_mode": format(spec.within_manifest_mode, "04o"),
                    "manifest_descriptor_identity": _identity_list(
                        authority.get("manifest_descriptor_identity"), label="within manifest",
                    ),
                    "selected_split": "val",
                    "manifest_read_once": True,
                }
                if authority != expected_authority:
                    raise PhysicalScoreError("derived within paired-view authority drift")
            else:
                expected_authority = {
                    "kind": "dandi688_fixed_v2_uuid_ledger_join",
                    "ledger_relative": spec.external_ledger_relative,
                    "ledger_sha256": spec.external_ledger_sha256,
                    "ledger_mode": format(spec.external_authority_mode, "04o"),
                    "ledger_descriptor_identity": _identity_list(
                        authority.get("ledger_descriptor_identity"), label="external ledger",
                    ),
                    "scope_relative": spec.external_scope_relative,
                    "scope_sha256": spec.external_scope_sha256,
                    "scope_mode": format(spec.external_authority_mode, "04o"),
                    "scope_descriptor_identity": _identity_list(
                        authority.get("scope_descriptor_identity"), label="external scope",
                    ),
                    "eligible_session_ids_semantics": "asset_uuid_then_exact_disposition_verified_download_scope_join",
                }
                if authority != expected_authority:
                    raise PhysicalScoreError("derived external fixed-ledger authority drift")
            checked.append(dict(row))
        if tuple(row["session"] for row in checked) != roster:
            raise PhysicalScoreError("derived input-assets roster/order drift")
        result[surface] = checked
    # The within table is literal even after descriptor derivation.  A forged
    # manifest cannot exchange a canonical filename/hash while retaining only
    # the roster labels.
    literal_within = tuple(
        (row["session"], row["frozen_path"], row["bytes"], row["sha256"])
        for row in result["within"]
    )
    if literal_within != spec.within_rows:
        raise PhysicalScoreError("derived within paired-view literal row drift")
    return result


def full_training_artifact_names() -> tuple[str, ...]:
    """Exact successful full-training leaf graph, never a directory scan/glob."""
    return (
        "attempt.json",
        "launch.json",
        "source_authority.json",
        "throughput100.json",
        *(f"epoch-{epoch:02d}.json" for epoch in range(48)),
        *(f"checkpoint-{epoch:02d}.pt" for epoch in (44, 45, 46, 47)),
        "swa_final4.pt",
        "terminal.json",
    )


@dataclass(frozen=True)
class ImportedFullMirrorProvenance:
    """Byte-preserving copy attestation for a completed remote full result."""

    local_mirror_relative: str
    remote_result_root: str
    remote_root_descriptor_identity: tuple[int, int]
    remote_terminal_sha256: str
    remote_artifact_sha256s: Mapping[str, str]
    imported_artifact_sha256s: Mapping[str, str]
    remote_full_closure_sha256: str
    copy_protocol: str = "held_descriptor_byte_preserving_copy_to_fresh_local_immutable_mirror"
    source_stage_identity_rebuilt_locally: bool = False

    def payload(self) -> dict[str, object]:
        names = full_training_artifact_names()
        local = _safe_relative(self.local_mirror_relative, "local full mirror")
        if not isinstance(self.remote_result_root, str) or not self.remote_result_root:
            raise PhysicalScoreError("remote full-result root label is missing")
        if (
            not isinstance(self.remote_root_descriptor_identity, tuple)
            or len(self.remote_root_descriptor_identity) != 2
            or any(type(item) is not int or item < 0 for item in self.remote_root_descriptor_identity)
        ):
            raise PhysicalScoreError("remote full-result directory identity drift")
        if not isinstance(self.remote_artifact_sha256s, Mapping) or not isinstance(self.imported_artifact_sha256s, Mapping):
            raise PhysicalScoreError("remote/imported full artifact map must be a mapping")
        if set(self.remote_artifact_sha256s) != set(names) or set(self.imported_artifact_sha256s) != set(names):
            raise PhysicalScoreError("remote/imported full artifact topology drift")
        remote = {name: _sha(self.remote_artifact_sha256s[name], f"remote artifact {name}") for name in names}
        imported = {name: _sha(self.imported_artifact_sha256s[name], f"imported artifact {name}") for name in names}
        if remote != imported:
            raise PhysicalScoreError("imported full-result bytes differ from remote immutable artifact map")
        if remote["terminal.json"] != _sha(self.remote_terminal_sha256, "remote terminal SHA"):
            raise PhysicalScoreError("remote terminal SHA does not bind remote artifact map")
        if self.copy_protocol != "held_descriptor_byte_preserving_copy_to_fresh_local_immutable_mirror":
            raise PhysicalScoreError("full result import protocol drift")
        if self.source_stage_identity_rebuilt_locally is not False:
            raise PhysicalScoreError("remote stage identity may not be rebuilt from local score stage")
        body = {
            "schema": "posterior_carrier_completed_full_import_mirror_v1",
            "local_mirror_relative": local,
            "remote_result_root": self.remote_result_root,
            "remote_root_descriptor_identity": list(self.remote_root_descriptor_identity),
            "remote_terminal_sha256": remote["terminal.json"],
            "remote_artifact_sha256s": remote,
            "imported_artifact_sha256s": imported,
            "remote_full_closure_sha256": _sha(self.remote_full_closure_sha256, "remote full closure SHA"),
            "copy_protocol": self.copy_protocol,
            "source_stage_identity_rebuilt_locally": False,
        }
        return {**body, "provenance_sha256": _digest(_json(body))}


def provenance_from_payload(value: Mapping[str, object]) -> ImportedFullMirrorProvenance:
    expected = {
        "schema", "local_mirror_relative", "remote_result_root", "remote_root_descriptor_identity",
        "remote_terminal_sha256", "remote_artifact_sha256s", "imported_artifact_sha256s",
        "remote_full_closure_sha256", "copy_protocol", "source_stage_identity_rebuilt_locally", "provenance_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "posterior_carrier_completed_full_import_mirror_v1":
        raise PhysicalScoreError("full import provenance schema drift")
    identity = value.get("remote_root_descriptor_identity")
    if not isinstance(identity, list) or len(identity) != 2:
        raise PhysicalScoreError("full import remote directory identity schema drift")
    try:
        result = ImportedFullMirrorProvenance(
            local_mirror_relative=value["local_mirror_relative"],
            remote_result_root=value["remote_result_root"],
            remote_root_descriptor_identity=(identity[0], identity[1]),
            remote_terminal_sha256=value["remote_terminal_sha256"],
            remote_artifact_sha256s=value["remote_artifact_sha256s"],
            imported_artifact_sha256s=value["imported_artifact_sha256s"],
            remote_full_closure_sha256=value["remote_full_closure_sha256"],
            copy_protocol=value["copy_protocol"],
            source_stage_identity_rebuilt_locally=value["source_stage_identity_rebuilt_locally"],
        )
    except (KeyError, TypeError) as error:
        raise PhysicalScoreError("full import provenance nested schema drift") from error
    if result.payload() != dict(value):
        raise PhysicalScoreError("full import provenance canonical digest/binding drift")
    return result


@dataclass(frozen=True)
class CompletedFullMirror:
    """All durable full training bytes, held only during the score lifecycle."""

    provenance: ImportedFullMirrorProvenance
    artifact_sha256s: Mapping[str, str]
    json_payloads: Mapping[str, Mapping[str, object]]
    bodies: Mapping[str, bytes]

    @property
    def terminal(self) -> Mapping[str, object]:
        return self.json_payloads["terminal.json"]

    @property
    def source_authority(self) -> Mapping[str, object]:
        return self.json_payloads["source_authority.json"]


def _expect_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PhysicalScoreError(f"{label} must be a mapping")
    return value


_FULL_CELL = "POSTERIOR_CARRIER_BUDGETMIX_D_SEED42"
_FULL_PHASE = "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_FULL_TRAIN_V1"
_SOURCE_EPOCHS = 48
_SOURCE_STEPS_PER_EPOCH = 33_925
_SOURCE_TOTAL_STEPS = _SOURCE_EPOCHS * _SOURCE_STEPS_PER_EPOCH
_SOURCE_BATCH_SIZE = 32
_SOURCE_SESSION_COUNT = 27
_THROUGHPUT_STEPS = 100
_CHECKPOINT_EPOCHS = (44, 45, 46, 47)
_BUDGETS = (4, 10, 30)
_FULL_CRITICAL_GRADIENT_KEYS = (
    "b3s_pre_pool_activity",
    "b3s_post_pool_activity",
    "b3s_post_pool_t4",
    "decoder_fc_in",
    "cross_attention",
    "ffn",
    "query_rep",
    "output_fc",
)
_FULL_RESOURCE_KEYS = (
    "rss_bytes",
    "current_allocated_bytes",
    "current_reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)
_FULL_OPTIMIZER_LITERAL = {
    "class": "Adam",
    "lr_constructor": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "schedule": "arm_common.lr_at_step(48,33925)",
}
_FULL_EXECUTION_POLICY_LITERAL = {
    "amp": False,
    "tf32": False,
    "torch_compile": False,
    "batch_size": _SOURCE_BATCH_SIZE,
    "source_only": True,
    "target_optimizer_steps": 0,
    "target_backward_calls": 0,
    "target_update_calls": 0,
}
_FULL_DROPOUT_CONTRACT = {
    "dynamic_dropout": True,
    "low": 0.0,
    "high": 1.0,
    "semantics": "complete_fused_unit_token_placeholder_with_inverse_probability_gain",
    "extra_dropout_draws": 0,
}
_REMOTE_TORCH_AUTHORITY = {
    "torch_version": "2.13.0+cu130",
    "torch_cuda_version": "13.0",
    "cudnn_version": 92000,
    "visible_devices": 1,
    "logical_device": "cuda:0",
    "name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
    "capability": [12, 0],
    "total_memory_bytes": 12_346_195_968,
    "nvml_status": "UNAVAILABLE_DRIVER_LIBRARY_MISMATCH",
}


def _source_only_boundaries() -> dict[str, object]:
    """The literal full-route no-target/no-cache boundary contract."""
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


def _full_training_spec() -> dict[str, object]:
    """Literal public 48-epoch spec copied from the sealed full route.

    This intentionally lives in the inert scorer substrate rather than
    importing ``full_train`` (which imports Torch).  It validates the remote
    receipt semantics while preserving the rule that a remote stage identity
    must never be reconstructed from the local scoring stage.
    """
    return {
        "epochs": _SOURCE_EPOCHS,
        "batch_size": _SOURCE_BATCH_SIZE,
        "steps_per_epoch": _SOURCE_STEPS_PER_EPOCH,
        "total_steps": _SOURCE_TOTAL_STEPS,
        "checkpoint_epochs": list(_CHECKPOINT_EPOCHS),
        "throughput_steps": _THROUGHPUT_STEPS,
        "seed": 42,
        "public": True,
        "optimizer": dict(_FULL_OPTIMIZER_LITERAL),
        "schedule": {
            "authority": "arm_common.lr_at_step",
            "epochs": _SOURCE_EPOCHS,
            "steps_per_epoch": _SOURCE_STEPS_PER_EPOCH,
            "warmup_epochs": 2,
            "warmup_start_lr": 1e-5,
            "warmup_end_lr": 1e-4,
            "cosine_final_lr": 1e-6,
        },
        "posterior_budget_schedule": {
            "formula": "budgets[(epoch + session_index) % 3]",
            "budgets": list(_BUDGETS),
            "session_static_within_logical_epoch": True,
            "epochs_per_budget_per_session": 16,
        },
        "boundaries": _source_only_boundaries(),
    }


def _full_receipt_topology() -> list[str]:
    """The historical attempt topology includes a mutually-exclusive failure."""
    return [
        "attempt.json", "launch.json", "source_authority.json", "throughput100.json",
        "swa_final4.pt", "terminal.json", "failure.json",
        *(f"epoch-{epoch:02d}.json" for epoch in range(_SOURCE_EPOCHS)),
        *(f"checkpoint-{epoch:02d}.pt" for epoch in _CHECKPOINT_EPOCHS),
    ]


def _finite(value: object, label: str, *, nonnegative: bool = False, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PhysicalScoreError(f"{label} must be finite")
    result = float(value)
    if nonnegative and result < 0.0:
        raise PhysicalScoreError(f"{label} must be nonnegative")
    if positive and result <= 0.0:
        raise PhysicalScoreError(f"{label} must be positive")
    return result


def _validate_canonical_closure(
    value: object,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Validate an explicit non-glob closure without opening its live paths."""
    mapping = _expect_mapping(value, label)
    if set(mapping) != {"paths", "sha256_by_path", "closure_sha256"}:
        raise PhysicalScoreError(f"{label} schema drift")
    paths, hashes = mapping.get("paths"), mapping.get("sha256_by_path")
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) for path in paths)
        or len(paths) != len(set(paths))
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(paths)
    ):
        raise PhysicalScoreError(f"{label} topology drift")
    normalized_paths = [_safe_relative(path, f"{label} path") for path in paths]
    normalized_hashes = {path: _sha(hashes[path], f"{label} SHA {path}") for path in normalized_paths}
    body = {"paths": normalized_paths, "sha256_by_path": normalized_hashes}
    digest = _digest(_json(body))
    if mapping.get("closure_sha256") != digest:
        raise PhysicalScoreError(f"{label} canonical digest drift")
    if expected_sha256 is not None and digest != _sha(expected_sha256, f"expected {label} SHA"):
        raise PhysicalScoreError(f"{label} expected SHA drift")
    return {**body, "closure_sha256": digest}


def _validate_hand_off(value: object, label: str) -> dict[str, str]:
    mapping = _expect_mapping(value, label)
    if set(mapping) != {"path", "sha256"}:
        raise PhysicalScoreError(f"{label} schema drift")
    return {
        "path": _safe_relative(mapping.get("path"), f"{label} path"),
        "sha256": _sha(mapping.get("sha256"), f"{label} SHA"),
    }


def _validate_phase_b_v1_identity(value: object, label: str) -> tuple[str, ...]:
    """Parse only the immutable source fields needed to audit a full receipt.

    The remote stage's descriptor identity is not rebuilt here.  This checks
    its frozen serialized form, including the strict source roster used by the
    per-epoch M(e,j) schedule, and returns that roster for exact schedule
    reconstruction below.
    """
    mapping = _expect_mapping(value, label)
    expected = {
        "cell", "phase", "handoff", "phase_b_normalizer_amendment", "source_authority",
        "closure", "remote_device", "boundaries",
    }
    if set(mapping) != expected or mapping.get("cell") != _FULL_CELL:
        raise PhysicalScoreError(f"{label} schema/cell drift")
    _validate_hand_off(mapping.get("handoff"), f"{label} handoff")
    if not isinstance(mapping.get("phase"), str) or not mapping["phase"]:
        raise PhysicalScoreError(f"{label} phase drift")
    if not isinstance(mapping.get("phase_b_normalizer_amendment"), str) or not mapping["phase_b_normalizer_amendment"]:
        raise PhysicalScoreError(f"{label} normalizer-amendment drift")
    if mapping.get("remote_device") != _REMOTE_TORCH_AUTHORITY:
        raise PhysicalScoreError(f"{label} remote device authority drift")
    if mapping.get("boundaries") != {
        "source_only": True, "target_opened": False, "within_opened": False,
        "external_opened": False, "formal_opened": False, "h1_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0, "scientific_score": False,
    }:
        raise PhysicalScoreError(f"{label} source-only boundary drift")
    _validate_canonical_closure(mapping.get("closure"), label=f"{label} closure")
    source = _expect_mapping(mapping.get("source_authority"), f"{label} source authority")
    source_expected = {
        "schema", "roster", "roster_sha256", "strict_source_metadata_sha256", "manifest_sha256",
        "ordinary_raw_t4_semantic_sha256", "behavior_normalizer_semantic_sha256", "source_lineage_sha256",
        "source_data_root", "source_only", "target_opened", "within_opened", "external_opened",
        "formal_opened", "h1_opened",
    }
    if set(source) != source_expected or source.get("schema") != "posterior_carrier_target_free_source_identity_v1":
        raise PhysicalScoreError(f"{label} strict-source authority schema drift")
    roster = source.get("roster")
    if (
        not isinstance(roster, list)
        or len(roster) != _SOURCE_SESSION_COUNT
        or any(not isinstance(item, str) or not item for item in roster)
        or len(set(roster)) != len(roster)
        or source.get("roster_sha256") != _digest(_json(roster))
    ):
        raise PhysicalScoreError(f"{label} strict-source roster/digest drift")
    for key in (
        "strict_source_metadata_sha256", "manifest_sha256", "ordinary_raw_t4_semantic_sha256",
        "behavior_normalizer_semantic_sha256", "source_lineage_sha256",
    ):
        _sha(source.get(key), f"{label} source {key}")
    if not isinstance(source.get("source_data_root"), Mapping):
        raise PhysicalScoreError(f"{label} source-data-root schema drift")
    if (
        source.get("source_only") is not True
        or any(source.get(key) is not False for key in ("target_opened", "within_opened", "external_opened", "formal_opened", "h1_opened"))
    ):
        raise PhysicalScoreError(f"{label} source authority boundary drift")
    return tuple(roster)


def _validate_phase_b_v2_identity(value: object, label: str) -> tuple[str, ...]:
    mapping = _expect_mapping(value, label)
    expected = {
        "cell", "phase", "handoff", "v1_base_source_identity", "v1_failed_predecessor",
        "theta_recovery", "closure", "remote_device", "boundaries",
    }
    if set(mapping) != expected or mapping.get("cell") != _FULL_CELL:
        raise PhysicalScoreError(f"{label} schema/cell drift")
    _validate_hand_off(mapping.get("handoff"), f"{label} handoff")
    if not isinstance(mapping.get("phase"), str) or not mapping["phase"]:
        raise PhysicalScoreError(f"{label} phase drift")
    if mapping.get("remote_device") != _REMOTE_TORCH_AUTHORITY:
        raise PhysicalScoreError(f"{label} remote device authority drift")
    if mapping.get("boundaries") != {
        "source_only": True, "target_opened": False, "within_opened": False,
        "external_opened": False, "formal_opened": False, "h1_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0, "scientific_score": False, "v1_result_mutated": False,
    }:
        raise PhysicalScoreError(f"{label} source-only boundary drift")
    _validate_canonical_closure(mapping.get("closure"), label=f"{label} closure")
    if not isinstance(mapping.get("v1_failed_predecessor"), Mapping):
        raise PhysicalScoreError(f"{label} V1 predecessor schema drift")
    theta = _expect_mapping(mapping.get("theta_recovery"), f"{label} theta recovery")
    required_theta = {
        "schema", "fallback_topology", "fallback_topology_sha256", "snap_tolerance_rad",
        "same_prefix_only", "no_later_row_substitution",
    }
    if (
        set(theta) != required_theta
        or not isinstance(theta.get("schema"), str)
        or not theta["schema"]
        or not isinstance(theta.get("fallback_topology"), Mapping)
        or not isinstance(theta.get("fallback_topology_sha256"), str)
        or theta["fallback_topology_sha256"] != _digest(_json(dict(theta["fallback_topology"])))
        or _finite(theta.get("snap_tolerance_rad"), f"{label} theta snap tolerance", positive=True) <= 0.0
        or theta.get("same_prefix_only") is not True
        or theta.get("no_later_row_substitution") is not True
    ):
        raise PhysicalScoreError(f"{label} theta-recovery binding drift")
    return _validate_phase_b_v1_identity(mapping.get("v1_base_source_identity"), f"{label} base V1")


def _validate_tf32_enforcement(value: object, label: str) -> None:
    mapping = _expect_mapping(value, label)
    expected = {
        "schema", "observed_pre_state", "enforced_post_state", "enforced_before_model_construction",
        "enforced_before_optimizer_construction", "route_local_restore_on_close", "acceptance_uses_enforced_post_state",
    }
    fields = {"cuda_matmul_allow_tf32", "cudnn_allow_tf32", "amp_enabled"}
    if set(mapping) != expected or mapping.get("schema") != "posterior_carrier_tf32_enforcement_v3":
        raise PhysicalScoreError(f"{label} schema drift")
    pre, post = mapping.get("observed_pre_state"), mapping.get("enforced_post_state")
    if (
        not isinstance(pre, Mapping) or not isinstance(post, Mapping)
        or set(pre) != fields or set(post) != fields
        or any(type(pre[key]) is not bool or type(post[key]) is not bool for key in fields)
        or any(post[key] is not False for key in fields)
        or mapping.get("enforced_before_model_construction") is not True
        or mapping.get("enforced_before_optimizer_construction") is not True
        or mapping.get("route_local_restore_on_close") is not True
        or mapping.get("acceptance_uses_enforced_post_state") is not True
    ):
        raise PhysicalScoreError(f"{label} state/order drift")


def _validate_phase_b_v3_identity(value: object, label: str) -> tuple[tuple[str, ...], dict[str, object]]:
    mapping = _expect_mapping(value, label)
    expected = {
        "cell", "phase", "handoff", "v2_source_identity", "v1_failed_predecessor",
        "v2_failed_predecessor", "closure", "remote_device", "tf32_contract", "boundaries",
    }
    if set(mapping) != expected or mapping.get("cell") != _FULL_CELL:
        raise PhysicalScoreError(f"{label} schema/cell drift")
    _validate_hand_off(mapping.get("handoff"), f"{label} handoff")
    if not isinstance(mapping.get("phase"), str) or not mapping["phase"]:
        raise PhysicalScoreError(f"{label} phase drift")
    if mapping.get("remote_device") != _REMOTE_TORCH_AUTHORITY:
        raise PhysicalScoreError(f"{label} remote device authority drift")
    if mapping.get("boundaries") != {
        "source_only": True, "target_opened": False, "within_opened": False,
        "external_opened": False, "formal_opened": False, "h1_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0, "scientific_score": False,
        "v1_result_mutated": False, "v2_result_mutated": False,
    }:
        raise PhysicalScoreError(f"{label} source-only boundary drift")
    if not isinstance(mapping.get("v1_failed_predecessor"), Mapping) or not isinstance(mapping.get("v2_failed_predecessor"), Mapping):
        raise PhysicalScoreError(f"{label} failed-predecessor schema drift")
    if mapping.get("tf32_contract") != {
        "amp": False,
        "cuda_matmul_allow_tf32_enforced": False,
        "cudnn_allow_tf32_enforced": False,
        "enforcement_before_model_and_optimizer": True,
        "pre_state_disclosed_not_gating": True,
        "route_local_restore": True,
    }:
        raise PhysicalScoreError(f"{label} TF32 contract drift")
    closure = _validate_canonical_closure(mapping.get("closure"), label=f"{label} closure")
    roster = _validate_phase_b_v2_identity(mapping.get("v2_source_identity"), f"{label} base V2")
    return roster, closure


def _stage_bound_source_authority_from_v3_identity(value: object, label: str) -> Mapping[str, object]:
    """Return the one serialized V1 source authority nested in a V3 identity."""
    v3 = _expect_mapping(value, label)
    v2 = _expect_mapping(v3.get("v2_source_identity"), f"{label} V2 identity")
    v1 = _expect_mapping(v2.get("v1_base_source_identity"), f"{label} V1 identity")
    return _expect_mapping(v1.get("source_authority"), f"{label} strict source authority")


def _validate_full_identity(
    value: object,
    *,
    expected_full_closure_sha256: str,
) -> tuple[dict[str, object], tuple[str, ...], dict[str, object], dict[str, object]]:
    """Validate serialized remote identity rather than rebuilding it locally."""
    mapping = _expect_mapping(value, "completed full identity")
    expected = {
        "cell", "phase", "handoff", "completed_smoke_v3_identity", "fresh_full_stage_v3_identity",
        "smoke_to_full_source_identity_binding", "phase_b_v3_closure", "full_closure",
        "source_smoke_lineage", "remote_torch_authority", "boundaries",
    }
    if set(mapping) != expected or mapping.get("cell") != _FULL_CELL or mapping.get("phase") != _FULL_PHASE:
        raise PhysicalScoreError("completed full identity schema/cell/phase drift")
    _validate_hand_off(mapping.get("handoff"), "completed full identity handoff")
    if mapping.get("remote_torch_authority") != _REMOTE_TORCH_AUTHORITY or mapping.get("boundaries") != _source_only_boundaries():
        raise PhysicalScoreError("completed full identity device/boundary drift")
    completed_roster, completed_v3_closure = _validate_phase_b_v3_identity(
        mapping.get("completed_smoke_v3_identity"), "completed smoke V3 identity",
    )
    fresh_roster, fresh_v3_closure = _validate_phase_b_v3_identity(
        mapping.get("fresh_full_stage_v3_identity"), "fresh full-stage V3 identity",
    )
    if completed_roster != fresh_roster or completed_v3_closure != fresh_v3_closure:
        raise PhysicalScoreError("completed smoke/fresh full V3 stable binding drift")
    completed_source = _stage_bound_source_authority_from_v3_identity(
        mapping["completed_smoke_v3_identity"], "completed smoke V3 identity",
    )
    fresh_source = _stage_bound_source_authority_from_v3_identity(
        mapping["fresh_full_stage_v3_identity"], "fresh full-stage V3 identity",
    )
    phase_v3 = _validate_canonical_closure(mapping.get("phase_b_v3_closure"), label="completed full Phase-B-v3 closure")
    if phase_v3 != fresh_v3_closure:
        raise PhysicalScoreError("completed full Phase-B-v3 closure/identity drift")
    full_closure = _validate_canonical_closure(
        mapping.get("full_closure"), label="completed full closure", expected_sha256=expected_full_closure_sha256,
    )
    lineage = _expect_mapping(mapping.get("source_smoke_lineage"), "completed full source-smoke lineage")
    expected_lineage = {
        "root_relative", "attempt_sha256", "launch_sha256", "source_authority_sha256",
        "step100_sha256", "terminal_sha256", "step100_status", "terminal_status",
        "phase_b_v3_closure_sha256", "completed_smoke_identity_sha256",
    }
    if set(lineage) != expected_lineage:
        raise PhysicalScoreError("completed full source-smoke lineage schema drift")
    _safe_relative(lineage.get("root_relative"), "completed full source-smoke root")
    for key in (
        "attempt_sha256", "launch_sha256", "source_authority_sha256", "step100_sha256",
        "terminal_sha256", "completed_smoke_identity_sha256",
    ):
        _sha(lineage.get(key), f"completed full source-smoke {key}")
    if (
        lineage.get("step100_status") != "SOURCE_SMOKE_V3_100_STEPS_COMPLETE"
        or lineage.get("terminal_status") != "SOURCE_SMOKE_V3_COMPLETE__NON_AUTHORITATIVE"
        or lineage.get("phase_b_v3_closure_sha256") != phase_v3["closure_sha256"]
        or lineage.get("completed_smoke_identity_sha256") != _digest(_json(mapping["completed_smoke_v3_identity"]))
    ):
        raise PhysicalScoreError("completed full source-smoke lineage binding drift")
    binding = _expect_mapping(mapping.get("smoke_to_full_source_identity_binding"), "completed smoke/full identity binding")
    required_binding = {
        "schema", "stage_bound_field", "stage_bound_difference_reason", "stage_bound_metadata_must_differ",
        "completed_smoke_strict_source_metadata_sha256", "fresh_full_stage_strict_source_metadata_sha256",
        "completed_smoke_v3_identity_sha256", "fresh_full_stage_v3_identity_sha256",
        "stage_agnostic_v3_identity_sha256", "accepted_phase_b_v3_closure_sha256",
        "source_authority_asset_sha256s", "strict_roster_sha256", "source_data_root",
    }
    if (
        set(binding) != required_binding
        or binding.get("schema") != "posterior_carrier_completed_smoke_to_full_source_identity_binding_v1"
        or binding.get("stage_bound_metadata_must_differ") is not True
        or binding.get("completed_smoke_v3_identity_sha256") != _digest(_json(mapping["completed_smoke_v3_identity"]))
        or binding.get("fresh_full_stage_v3_identity_sha256") != _digest(_json(mapping["fresh_full_stage_v3_identity"]))
        or binding.get("accepted_phase_b_v3_closure_sha256") != phase_v3["closure_sha256"]
        or binding.get("completed_smoke_strict_source_metadata_sha256") != completed_source.get("strict_source_metadata_sha256")
        or binding.get("fresh_full_stage_strict_source_metadata_sha256") != fresh_source.get("strict_source_metadata_sha256")
        or binding.get("completed_smoke_strict_source_metadata_sha256") == binding.get("fresh_full_stage_strict_source_metadata_sha256")
        or binding.get("strict_roster_sha256") != _digest(_json(list(fresh_roster)))
        or binding.get("source_data_root") != fresh_source.get("source_data_root")
        or completed_source.get("source_data_root") != fresh_source.get("source_data_root")
    ):
        raise PhysicalScoreError("completed smoke/full identity binding drift")
    for key in (
        "completed_smoke_strict_source_metadata_sha256", "fresh_full_stage_strict_source_metadata_sha256",
        "stage_agnostic_v3_identity_sha256", "strict_roster_sha256",
    ):
        _sha(binding.get(key), f"completed smoke/full binding {key}")
    assets = binding.get("source_authority_asset_sha256s")
    if not isinstance(assets, Mapping) or not assets or any(
        not isinstance(path, str) or _sha(digest, f"completed smoke/full asset {path}") != digest
        for path, digest in assets.items()
    ):
        raise PhysicalScoreError("completed smoke/full authority-asset binding drift")
    if not isinstance(binding.get("source_data_root"), Mapping):
        raise PhysicalScoreError("completed smoke/full source-data-root binding drift")
    return dict(mapping), fresh_roster, full_closure, phase_v3


def _full_lr_at_step(step: int) -> float:
    if type(step) is not int or not 0 <= step < _SOURCE_TOTAL_STEPS:
        raise PhysicalScoreError("completed full LR step lies outside frozen budget")
    warmup = 2 * _SOURCE_STEPS_PER_EPOCH
    if step < warmup:
        return 1e-5 + (1e-4 - 1e-5) * (step / warmup)
    span = _SOURCE_TOTAL_STEPS - warmup
    progress = (step - warmup) / span
    return 1e-6 + 0.5 * (1e-4 - 1e-6) * (1.0 + math.cos(math.pi * progress))


def _full_budget_schedule() -> list[list[int]]:
    return [
        [_BUDGETS[(epoch + session_index) % len(_BUDGETS)] for session_index in range(_SOURCE_SESSION_COUNT)]
        for epoch in range(_SOURCE_EPOCHS)
    ]


def _expected_epoch_schedule(roster: Sequence[str], epoch: int) -> dict[str, object]:
    budget_rows = [_BUDGETS[(epoch + index) % len(_BUDGETS)] for index in range(len(roster))]
    return {
        "epoch": epoch,
        "roster": list(roster),
        "budget_by_session": {session: budget for session, budget in zip(roster, budget_rows, strict=True)},
        "epoch_budget_row_sha256": _digest(_json([budget_rows])),
        "full_budget_schedule_sha256": _digest(_json(_full_budget_schedule())),
        "session_static_within_epoch": True,
        "epochs_per_budget_per_session": 16,
    }


def _expected_source_schedule(roster: Sequence[str]) -> dict[str, object]:
    return {
        "roster": list(roster),
        "budgets": list(_BUDGETS),
        "formula": "budgets[(epoch + session_index) % 3]",
        "epochs": _SOURCE_EPOCHS,
        "session_count": _SOURCE_SESSION_COUNT,
        "epochs_per_budget_per_session": 16,
        "schedule_sha256": _digest(_json(_full_budget_schedule())),
    }


def _validate_resources(value: object, label: str) -> None:
    mapping = _expect_mapping(value, label)
    if set(mapping) != set(_FULL_RESOURCE_KEYS):
        raise PhysicalScoreError(f"{label} schema drift")
    for key in _FULL_RESOURCE_KEYS:
        if type(mapping[key]) is not int or mapping[key] < 0:
            raise PhysicalScoreError(f"{label} value drift")
    if (
        mapping["peak_allocated_bytes"] < mapping["current_allocated_bytes"]
        or mapping["peak_reserved_bytes"] < mapping["current_reserved_bytes"]
    ):
        raise PhysicalScoreError(f"{label} peak/current accounting drift")


def _validate_epoch_receipt(
    value: object,
    *,
    identity: Mapping[str, object],
    roster: Sequence[str],
    epoch: int,
) -> None:
    mapping = _expect_mapping(value, f"completed full epoch {epoch}")
    expected_keys = {
        "schema", "cell", "phase", "spec", "identity", "epoch", "cumulative_optimizer_steps",
        "schedule", "loss", "lr", "batch_count_by_budget", "epoch_boundary_proof", "posterior_cache",
        "dropout_contract", "resources", "progress", "boundaries", "elapsed_seconds",
        "throughput_steps_per_second",
    }
    global_step = (epoch + 1) * _SOURCE_STEPS_PER_EPOCH
    if (
        set(mapping) != expected_keys
        or mapping.get("schema") != "posterior_carrier_full_epoch_v1"
        or mapping.get("cell") != _FULL_CELL
        or mapping.get("phase") != _FULL_PHASE
        or mapping.get("spec") != _full_training_spec()
        or mapping.get("identity") != identity
        or mapping.get("epoch") != epoch
        or mapping.get("cumulative_optimizer_steps") != global_step
        or mapping.get("schedule") != _expected_epoch_schedule(roster, epoch)
        or mapping.get("dropout_contract") != _FULL_DROPOUT_CONTRACT
        or mapping.get("boundaries") != _source_only_boundaries()
    ):
        raise PhysicalScoreError(f"completed full epoch {epoch} frozen binding drift")
    loss = _expect_mapping(mapping.get("loss"), f"completed full epoch {epoch} loss")
    if set(loss) != {"mean", "min", "max"}:
        raise PhysicalScoreError(f"completed full epoch {epoch} loss schema drift")
    minimum, mean, maximum = (
        _finite(loss[key], f"completed full epoch {epoch} loss {key}", nonnegative=True)
        for key in ("min", "mean", "max")
    )
    if not minimum <= mean <= maximum:
        raise PhysicalScoreError(f"completed full epoch {epoch} loss ordering drift")
    lr = _expect_mapping(mapping.get("lr"), f"completed full epoch {epoch} LR")
    if set(lr) != {"first", "last", "expected_first", "expected_last"}:
        raise PhysicalScoreError(f"completed full epoch {epoch} LR schema drift")
    expected_first, expected_last = _full_lr_at_step(epoch * _SOURCE_STEPS_PER_EPOCH), _full_lr_at_step(global_step - 1)
    if (
        lr.get("first") != lr.get("expected_first")
        or lr.get("last") != lr.get("expected_last")
        or lr.get("first") != expected_first
        or lr.get("last") != expected_last
    ):
        raise PhysicalScoreError(f"completed full epoch {epoch} LR binding drift")
    counts = _expect_mapping(mapping.get("batch_count_by_budget"), f"completed full epoch {epoch} budget counts")
    if (
        set(counts) != {str(item) for item in _BUDGETS}
        or any(type(counts[str(item)]) is not int or counts[str(item)] < 0 for item in _BUDGETS)
        or sum(int(counts[str(item)]) for item in _BUDGETS) != _SOURCE_STEPS_PER_EPOCH
    ):
        raise PhysicalScoreError(f"completed full epoch {epoch} budget-count drift")
    proof = _expect_mapping(mapping.get("epoch_boundary_proof"), f"completed full epoch {epoch} boundary proof")
    if (
        set(proof) != {
            "critical_gradients", "finite_model", "finite_adam", "model_state_sha256", "optimizer_state_sha256",
        }
        or not isinstance(proof.get("critical_gradients"), Mapping)
        or set(proof["critical_gradients"]) != set(_FULL_CRITICAL_GRADIENT_KEYS)
        or not all(item is True for item in proof["critical_gradients"].values())
        or proof.get("finite_model") is not True
        or proof.get("finite_adam") is not True
    ):
        raise PhysicalScoreError(f"completed full epoch {epoch} critical-gradient/finite proof drift")
    _sha(proof.get("model_state_sha256"), f"completed full epoch {epoch} model state SHA")
    _sha(proof.get("optimizer_state_sha256"), f"completed full epoch {epoch} optimizer state SHA")
    expected_cache = {
        "posterior_fit_calls": _SOURCE_SESSION_COUNT * len(_BUDGETS),
        "posterior_inverse_calls": _SOURCE_SESSION_COUNT * len(_BUDGETS),
        "deterministic_mean_view_builds": _SOURCE_SESSION_COUNT * len(_BUDGETS),
        "epoch_sampled_view_builds": _SOURCE_SESSION_COUNT * (epoch + 1),
        "device_epoch_view_builds": _SOURCE_SESSION_COUNT * (epoch + 1),
        "normalized_view_builds": _SOURCE_SESSION_COUNT * len(_BUDGETS) + _SOURCE_SESSION_COUNT * (epoch + 1),
        "batch_loop_requests": global_step,
        "batch_loop_inverse_calls": 0,
        "source_sessions": _SOURCE_SESSION_COUNT,
        "scheduled_session_epochs": _SOURCE_SESSION_COUNT * (epoch + 1),
    }
    if mapping.get("posterior_cache") != expected_cache:
        raise PhysicalScoreError(f"completed full epoch {epoch} posterior-cache/inverse proof drift")
    _validate_resources(mapping.get("resources"), f"completed full epoch {epoch} resources")
    if mapping.get("progress") != {
        "epoch": epoch,
        "completed_epochs": epoch + 1,
        "epochs": _SOURCE_EPOCHS,
        "optimizer_steps_completed": global_step,
        "total_optimizer_steps": _SOURCE_TOTAL_STEPS,
        "source_opened": True,
        "remote_initialized": True,
    }:
        raise PhysicalScoreError(f"completed full epoch {epoch} progress/boundary drift")
    _finite(mapping.get("elapsed_seconds"), f"completed full epoch {epoch} elapsed seconds", positive=True)
    _finite(mapping.get("throughput_steps_per_second"), f"completed full epoch {epoch} throughput", positive=True)


def _validate_throughput_receipt(value: object, *, identity: Mapping[str, object]) -> None:
    mapping = _expect_mapping(value, "completed full throughput100")
    expected = {
        "schema", "cell", "phase", "spec", "identity", "steps", "epoch", "elapsed_seconds",
        "steps_per_second", "resources", "boundaries", "status",
    }
    if (
        set(mapping) != expected
        or mapping.get("schema") != "posterior_carrier_full_throughput_v1"
        or mapping.get("cell") != _FULL_CELL
        or mapping.get("phase") != _FULL_PHASE
        or mapping.get("spec") != _full_training_spec()
        or mapping.get("identity") != identity
        or mapping.get("steps") != _THROUGHPUT_STEPS
        or mapping.get("epoch") != 0
        or mapping.get("boundaries") != _source_only_boundaries()
        or mapping.get("status") != "ENGINEERING_THROUGHPUT_ONLY"
    ):
        raise PhysicalScoreError("completed full throughput100 frozen semantics drift")
    _finite(mapping.get("elapsed_seconds"), "completed full throughput100 elapsed seconds", positive=True)
    _finite(mapping.get("steps_per_second"), "completed full throughput100 rate", positive=True)
    _validate_resources(mapping.get("resources"), "completed full throughput100 resources")


def _validate_completed_full_graph(
    *,
    provenance: ImportedFullMirrorProvenance,
    bodies: Mapping[str, bytes],
    json_payloads: Mapping[str, Mapping[str, object]],
    expected_terminal_sha256: str,
    expected_swa_sha256: str,
    expected_swa_state_sha256: str,
    expected_source_authority_sha256: str,
    expected_checkpoint_sha256s: Mapping[str, str],
    expected_full_closure_sha256: str,
) -> None:
    names = full_training_artifact_names()
    expected = set(names)
    if set(bodies) != expected or set(json_payloads) != {
        name for name in names if name.endswith(".json")
    }:
        raise PhysicalScoreError("completed full mirror body/JSON graph topology drift")
    actual_sha = {name: _digest(bodies[name]) for name in names}
    if actual_sha != dict(provenance.imported_artifact_sha256s):
        raise PhysicalScoreError("completed full mirror body bytes differ from import provenance")
    if actual_sha["terminal.json"] != _sha(expected_terminal_sha256, "expected full terminal SHA"):
        raise PhysicalScoreError("completed full terminal SHA drift")
    if actual_sha["swa_final4.pt"] != _sha(expected_swa_sha256, "expected full SWA SHA"):
        raise PhysicalScoreError("completed full SWA SHA drift")
    if actual_sha["source_authority.json"] != _sha(expected_source_authority_sha256, "expected full source-authority SHA"):
        raise PhysicalScoreError("completed full source-authority SHA drift")
    checkpoint_names = {str(epoch): f"checkpoint-{epoch:02d}.pt" for epoch in (44, 45, 46, 47)}
    if set(expected_checkpoint_sha256s) != set(checkpoint_names):
        raise PhysicalScoreError("expected final-four checkpoint topology drift")
    for epoch, name in checkpoint_names.items():
        if actual_sha[name] != _sha(expected_checkpoint_sha256s[epoch], f"expected checkpoint {epoch} SHA"):
            raise PhysicalScoreError("completed full checkpoint body SHA drift")

    attempt, launch = json_payloads["attempt.json"], json_payloads["launch.json"]
    source, throughput, terminal = (
        json_payloads["source_authority.json"], json_payloads["throughput100.json"], json_payloads["terminal.json"],
    )
    # Do not recreate a ``FullRunIdentity`` from this local scoring stage.
    # Instead, validate the remote serialized identity and all of the public
    # full-route receipt semantics against literal frozen constraints.
    identity = _expect_mapping(attempt.get("identity"), "completed full attempt identity")
    validated_identity, roster, full_closure, phase_v3_closure = _validate_full_identity(
        identity, expected_full_closure_sha256=expected_full_closure_sha256,
    )
    full_spec = _full_training_spec()
    attempt_expected = {
        "schema", "cell", "phase", "spec", "identity", "topology", "source_opened",
        "remote_initialized", "optimizer_steps_completed", "boundaries", "status",
    }
    if (
        set(attempt) != attempt_expected
        or attempt.get("schema") != "posterior_carrier_full_train_attempt_v1"
        or attempt.get("cell") != _FULL_CELL
        or attempt.get("phase") != _FULL_PHASE
        or attempt.get("spec") != full_spec
        or attempt.get("identity") != validated_identity
        or attempt.get("topology") != _full_receipt_topology()
        or attempt.get("source_opened") is not False
        or attempt.get("remote_initialized") is not False
        or attempt.get("optimizer_steps_completed") != 0
        or attempt.get("boundaries") != _source_only_boundaries()
        or attempt.get("status") != "ATTEMPT_STARTED_SOURCE_ONLY"
    ):
        raise PhysicalScoreError("completed full attempt frozen schema/boundary drift")
    launch_expected = {
        "schema", "cell", "phase", "spec", "identity", "attempt_sha256", "full_launch_closure",
        "phase_b_v3_launch_closure", "remote_torch_authority", "optimizer", "execution_policy",
        "dropout_contract", "boundaries", "status",
    }
    if (
        set(launch) != launch_expected
        or launch.get("schema") != "posterior_carrier_full_train_launch_v1"
        or launch.get("cell") != _FULL_CELL
        or launch.get("phase") != _FULL_PHASE
        or launch.get("spec") != full_spec
        or launch.get("identity") != validated_identity
        or launch.get("attempt_sha256") != actual_sha["attempt.json"]
        or _validate_canonical_closure(
            launch.get("full_launch_closure"), label="completed full launch closure",
            expected_sha256=expected_full_closure_sha256,
        ) != full_closure
        or _validate_canonical_closure(
            launch.get("phase_b_v3_launch_closure"), label="completed full launch Phase-B-v3 closure",
        ) != phase_v3_closure
        or launch.get("remote_torch_authority") != _REMOTE_TORCH_AUTHORITY
        or launch.get("optimizer") != _FULL_OPTIMIZER_LITERAL
        or launch.get("execution_policy") != _FULL_EXECUTION_POLICY_LITERAL
        or launch.get("dropout_contract") != _FULL_DROPOUT_CONTRACT
        or launch.get("boundaries") != _source_only_boundaries()
        or launch.get("status") != "FULL_TRAINING_LAUNCHED"
    ):
        raise PhysicalScoreError("completed full launch frozen schema/boundary drift")
    source_expected = {
        "schema", "cell", "phase", "spec", "identity", "full_launch_sha256",
        "phase_b_v3_source_authority", "phase_b_v3_source_authority_sha256", "schedule",
        "remote_torch_authority", "boundaries", "status",
    }
    nested_source = _expect_mapping(source.get("phase_b_v3_source_authority"), "completed full nested Phase-B-v3 authority")
    nested_source_expected = {
        "schema", "cell", "v2_compatible_authority", "tf32_enforcement", "closure", "launch_sha256",
        "v2_failed_predecessor",
    }
    if (
        set(source) != source_expected
        or source.get("schema") != "posterior_carrier_full_source_authority_v1"
        or source.get("cell") != _FULL_CELL
        or source.get("phase") != _FULL_PHASE
        or source.get("spec") != full_spec
        or source.get("identity") != validated_identity
        or source.get("full_launch_sha256") != actual_sha["launch.json"]
        or source.get("phase_b_v3_source_authority_sha256") != _digest(_json(dict(nested_source)))
        or source.get("schedule") != _expected_source_schedule(roster)
        or source.get("remote_torch_authority") != _REMOTE_TORCH_AUTHORITY
        or source.get("boundaries") != _source_only_boundaries()
        or source.get("status") != "STRICT27_POSTERIOR_SOURCE_AUTHORITY_READY"
        or set(nested_source) != nested_source_expected
        or nested_source.get("schema") != "posterior_carrier_source_authority_v3"
        or nested_source.get("cell") != _FULL_CELL
        or nested_source.get("launch_sha256") != actual_sha["launch.json"]
        or _validate_canonical_closure(
            nested_source.get("closure"), label="completed full nested Phase-B-v3 closure",
        ) != phase_v3_closure
        or not isinstance(nested_source.get("v2_compatible_authority"), Mapping)
        or not isinstance(nested_source.get("v2_failed_predecessor"), Mapping)
    ):
        raise PhysicalScoreError("completed full source-authority frozen schema/binding drift")
    _validate_tf32_enforcement(nested_source.get("tf32_enforcement"), "completed full nested TF32 enforcement")
    _validate_throughput_receipt(throughput, identity=validated_identity)
    terminal_expected = {
        "schema", "cell", "phase", "spec", "identity", "final_identity", "attempt_sha256", "launch_sha256",
        "source_authority_sha256", "artifact_sha256s", "checkpoint_state_sha256s", "swa_state_sha256",
        "swa_proof", "full_launch_closure", "full_final_closure", "phase_b_v3_launch_closure",
        "phase_b_v3_final_closure", "boundaries", "status",
    }
    if (
        set(terminal) != terminal_expected
        or terminal.get("schema") != "posterior_carrier_full_train_terminal_v1"
        or terminal.get("cell") != _FULL_CELL
        or terminal.get("phase") != _FULL_PHASE
        or terminal.get("spec") != full_spec
        or terminal.get("identity") != validated_identity
        or terminal.get("final_identity") != validated_identity
        or terminal.get("attempt_sha256") != actual_sha["attempt.json"]
        or terminal.get("launch_sha256") != actual_sha["launch.json"]
        or terminal.get("source_authority_sha256") != actual_sha["source_authority.json"]
        or terminal.get("swa_state_sha256") != _sha(expected_swa_state_sha256, "expected full SWA state SHA")
        or _validate_canonical_closure(
            terminal.get("full_launch_closure"), label="completed full terminal launch closure",
            expected_sha256=expected_full_closure_sha256,
        ) != full_closure
        or _validate_canonical_closure(
            terminal.get("full_final_closure"), label="completed full terminal final closure",
            expected_sha256=expected_full_closure_sha256,
        ) != full_closure
        or _validate_canonical_closure(
            terminal.get("phase_b_v3_launch_closure"), label="completed full terminal launch Phase-B-v3 closure",
        ) != phase_v3_closure
        or _validate_canonical_closure(
            terminal.get("phase_b_v3_final_closure"), label="completed full terminal final Phase-B-v3 closure",
        ) != phase_v3_closure
        or terminal.get("boundaries") != _source_only_boundaries()
        or terminal.get("status") != "FULL_TRAINING_COMPLETE__SOURCE_ONLY__AWAITING_SEPARATE_SCORER"
    ):
        raise PhysicalScoreError("completed full terminal graph/closure/spec drift")
    terminal_artifacts = terminal.get("artifact_sha256s")
    expected_terminal_artifacts = {name: actual_sha[name] for name in names if name != "terminal.json"}
    if terminal_artifacts != expected_terminal_artifacts:
        raise PhysicalScoreError("completed full terminal artifact graph drift")
    checkpoints = terminal.get("checkpoint_state_sha256s")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(checkpoint_names):
        raise PhysicalScoreError("completed full terminal checkpoint-state topology drift")
    for value in checkpoints.values():
        _sha(value, "completed full checkpoint state SHA")
    proof = _expect_mapping(terminal.get("swa_proof"), "completed full SWA proof")
    expected_proof_keys = {
        "checkpoint_epochs", "checkpoint_state_sha256s", "fresh_strict_load", "eval_mode",
        "repeat_bitwise_equal", "state_unchanged", "eval_no_sampling", "prediction_shape",
        "prediction_sha256", "state_digest_before_eval", "state_digest_after_eval", "boundaries",
    }
    if (
        set(proof) != expected_proof_keys
        or proof.get("checkpoint_epochs") != list(_CHECKPOINT_EPOCHS)
        or proof.get("checkpoint_state_sha256s") != dict(checkpoints)
        or proof.get("fresh_strict_load") is not True
        or proof.get("eval_mode") is not True
        or proof.get("repeat_bitwise_equal") is not True
        or proof.get("state_unchanged") is not True
        or proof.get("eval_no_sampling") is not True
        or proof.get("prediction_shape") != [_SOURCE_BATCH_SIZE, 50, 2]
        or proof.get("boundaries") != _source_only_boundaries()
        or proof.get("state_digest_before_eval") != proof.get("state_digest_after_eval")
    ):
        raise PhysicalScoreError("completed full SWA proof drift")
    for key in ("prediction_sha256", "state_digest_before_eval", "state_digest_after_eval"):
        _sha(proof.get(key), f"completed full SWA proof {key}")
    for epoch in range(_SOURCE_EPOCHS):
        row = json_payloads[f"epoch-{epoch:02d}.json"]
        _validate_epoch_receipt(row, identity=validated_identity, roster=roster, epoch=epoch)


def load_completed_full_mirror(
    mirror_directory: Path,
    *,
    provenance: ImportedFullMirrorProvenance,
    expected_terminal_sha256: str,
    expected_swa_sha256: str,
    expected_swa_state_sha256: str,
    expected_source_authority_sha256: str,
    expected_checkpoint_sha256s: Mapping[str, str],
    expected_full_closure_sha256: str,
) -> CompletedFullMirror:
    """Read the full remote result copy through one held local mirror FD.

    This never calls a full-training identity builder.  That would make a
    local score stage look like the remote stage that produced the model.
    """
    with ImmutableDirectory.open(mirror_directory) as directory:
        expected_names = set(full_training_artifact_names())
        expected_leaves = expected_names | {name + ".sha256" for name in expected_names}
        names = set(directory.names())
        if names != expected_leaves or "failure.json" in names:
            raise PhysicalScoreError("completed full mirror immutable topology/failure drift")
        loaded = {
            name: directory.read_pair(name, expected_sha256=provenance.imported_artifact_sha256s[name])
            for name in full_training_artifact_names()
        }
        directory.reverify()
    json_payloads = {
        name: loaded[name].json_object() for name in full_training_artifact_names() if name.endswith(".json")
    }
    bodies = {name: loaded[name].body for name in full_training_artifact_names()}
    _validate_completed_full_graph(
        provenance=provenance, bodies=bodies, json_payloads=json_payloads,
        expected_terminal_sha256=expected_terminal_sha256, expected_swa_sha256=expected_swa_sha256,
        expected_swa_state_sha256=expected_swa_state_sha256,
        expected_source_authority_sha256=expected_source_authority_sha256,
        expected_checkpoint_sha256s=expected_checkpoint_sha256s,
        expected_full_closure_sha256=expected_full_closure_sha256,
    )
    return CompletedFullMirror(
        provenance=provenance,
        artifact_sha256s={name: loaded[name].sha256 for name in full_training_artifact_names()},
        json_payloads=json_payloads,
        bodies=bodies,
    )


SEALED_CELL_D_TERMINAL_RELATIVE = "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json"
SEALED_CELL_D_SWA_RELATIVE = "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt"
SEALED_CELL_D_BASELINE_RELATIVE = "tfpd_exploration/results/sparsification_score_v1/sparsification_score_receipt.json"
SEALED_CELL_D_TERMINAL_SHA256 = "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442"
SEALED_CELL_D_SWA_SHA256 = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
SEALED_CELL_D_BASELINE_SHA256 = "583b899bb9e6b132a50b23552d9734b0ccb9b12ecc5c43c27c96ba49bd71980f"


@dataclass(frozen=True)
class SealedCellDMaterial:
    """Sealed Cell-D bytes and parsed governing table source."""

    terminal: Mapping[str, object]
    terminal_body: bytes
    swa_body: bytes
    baseline: Mapping[str, object]
    terminal_sha256: str
    swa_sha256: str
    baseline_sha256: str


def load_sealed_cell_d_material(root: Path) -> SealedCellDMaterial:
    """Load only explicit sealed Cell-D files through descriptor-safe paths."""
    terminal = _repository_pair(root, SEALED_CELL_D_TERMINAL_RELATIVE, expected_sha256=SEALED_CELL_D_TERMINAL_SHA256)
    swa = _repository_pair(root, SEALED_CELL_D_SWA_RELATIVE, expected_sha256=SEALED_CELL_D_SWA_SHA256)
    baseline = _repository_pair(root, SEALED_CELL_D_BASELINE_RELATIVE, expected_sha256=SEALED_CELL_D_BASELINE_SHA256)
    terminal_json = terminal.json_object()
    baseline_json = baseline.json_object()
    terminal_swa = terminal_json.get("swa")
    if (
        terminal_json.get("schema") != "tfpd_pop_robust_cell_v1"
        or terminal_json.get("status") != "CELL_TERMINAL"
        or terminal_json.get("cell") != "D"
        or not isinstance(terminal_swa, Mapping)
        or terminal_swa.get("sha256") != SEALED_CELL_D_SWA_SHA256
        or terminal_swa.get("window_epochs") != [44, 45, 46, 47]
        or terminal_swa.get("strict_reload_finite_forward_smoke") is not True
    ):
        raise PhysicalScoreError("sealed Cell-D terminal/SWA semantic drift")
    results = baseline_json.get("results")
    governing = results.get("D_swa", {}).get("governing_last_bin") if isinstance(results, Mapping) else None
    if (
        baseline_json.get("schema") != "tfpd_sparsification_score_v1"
        or baseline_json.get("status") != "SPARSIFICATION_SCORED"
        or not isinstance(governing, Mapping)
        or set(governing) != {"within", "external"}
    ):
        raise PhysicalScoreError("sealed Cell-D governing last-bin table drift")
    return SealedCellDMaterial(
        terminal=terminal_json, terminal_body=terminal.body, swa_body=swa.body, baseline=baseline_json,
        terminal_sha256=terminal.sha256, swa_sha256=swa.sha256, baseline_sha256=baseline.sha256,
    )


def sealed_governing_rows(baseline: Mapping[str, object]) -> dict[str, tuple[dict[str, object], ...]]:
    """Extract exactly the sealed D-SWA last-bin table, never a legacy score."""
    results = baseline.get("results") if isinstance(baseline, Mapping) else None
    governing = results.get("D_swa", {}).get("governing_last_bin") if isinstance(results, Mapping) else None
    if not isinstance(governing, Mapping) or set(governing) != {"within", "external"}:
        raise PhysicalScoreError("sealed governing table topology drift")
    result: dict[str, tuple[dict[str, object], ...]] = {}
    expected_counts = {"within": 6, "external": 15}
    for surface, count in expected_counts.items():
        block = governing.get(surface)
        rows = block.get("per_session") if isinstance(block, Mapping) else None
        if not isinstance(rows, list) or len(rows) != count:
            raise PhysicalScoreError("sealed governing per-session cardinality drift")
        normalized: list[dict[str, object]] = []
        for row in rows:
            if (
                not isinstance(row, Mapping) or set(row) < {"session", "n_windows", "r2"}
                or not isinstance(row.get("session"), str) or not row["session"]
                or type(row.get("n_windows")) is not int or row["n_windows"] <= 0
                or isinstance(row.get("r2"), bool) or not isinstance(row.get("r2"), (int, float))
            ):
                raise PhysicalScoreError("sealed governing per-session row drift")
            normalized.append({"session": row["session"], "n_windows": row["n_windows"], "r2": float(row["r2"])})
        if tuple(item["session"] for item in normalized) != tuple(sorted(item["session"] for item in normalized)):
            raise PhysicalScoreError("sealed governing per-session order drift")
        mean = sum(float(item["r2"]) for item in normalized) / count
        if block.get("mean_r2") != mean:
            raise PhysicalScoreError("sealed governing equal-session mean drift")
        result[surface] = tuple(normalized)
    return result


def source_authority_posterior_payloads(source_authority: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Find the unique frozen prior and posterior normalizer without refitting.

    The full source-authority schema nests the reviewed Phase-B-v3/v2/v1
    evidence.  A recursive structural search is safer than hard-coding a
    private nesting path, but uniqueness is mandatory: an extra plausible
    normalizer/prior is itself a provenance failure.
    """
    priors: list[Mapping[str, object]] = []
    normalizers: list[Mapping[str, object]] = []

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            if value.get("schema") == "posterior_carrier_source_prior_v1":
                priors.append(value)
            if value.get("schema") == "posterior_carrier_source_normalizer_v1":
                normalizers.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(source_authority)
    # The reviewed authority intentionally repeats the same normalizer in a
    # bank cross-link.  Repetition of byte-identical evidence is harmless;
    # two distinct plausible priors/normalizers are not.
    unique_priors = {_json(item): item for item in priors}
    unique_normalizers = {_json(item): item for item in normalizers}
    if len(unique_priors) != 1 or len(unique_normalizers) != 1:
        raise PhysicalScoreError("full source authority must contain one unique frozen posterior prior and normalizer")
    return next(iter(unique_priors.values())), next(iter(unique_normalizers.values()))


def derive_posterior_normalizer_from_full_mirror(
    root: Path,
    *,
    provenance: ImportedFullMirrorProvenance,
    terminal_sha256: str,
    swa_sha256: str,
    swa_state_sha256: str,
    source_authority_sha256: str,
    checkpoint_sha256s: Mapping[str, str],
    full_closure_sha256: str,
) -> dict[str, object]:
    """Read the sole frozen posterior normalizer from the completed mirror.

    This is metadata-only: ``load_completed_full_mirror`` validates immutable
    body/sidecar/topology bindings but does not deserialize any checkpoint
    tensor.  A later physical prepare repeats the load before strict model
    placement.  Returning the source-authority payload rather than accepting
    a caller copy makes the normalizer a derivation, not a self-hashed claim.
    """
    full = load_completed_full_mirror(
        Path(root).absolute() / provenance.local_mirror_relative,
        provenance=provenance,
        expected_terminal_sha256=terminal_sha256,
        expected_swa_sha256=swa_sha256,
        expected_swa_state_sha256=swa_state_sha256,
        expected_source_authority_sha256=source_authority_sha256,
        expected_checkpoint_sha256s=checkpoint_sha256s,
        expected_full_closure_sha256=full_closure_sha256,
    )
    _prior, normalizer = source_authority_posterior_payloads(full.source_authority)
    copied = json.loads(_json(normalizer))
    if not isinstance(copied, dict):
        raise PhysicalScoreError("full posterior normalizer JSON-root drift")
    if copied.get("body_sha256") != _digest(_json({key: value for key, value in copied.items() if key != "body_sha256"})):
        raise PhysicalScoreError("full posterior normalizer body digest drift")
    return copied


def physical_dependency_manifest() -> tuple[str, ...]:
    """Explicit runtime inputs used only after reviewed execution capability."""
    return (
        "tfpd_exploration/src/__init__.py",
        "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
        "tfpd_exploration/src/posterior_carrier_v1/plan.py",
        "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        "tfpd_exploration/src/posterior_carrier_v1/full_train.py",
        "tfpd_exploration/src/posterior_carrier_v1/core.py",
        "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
        "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
        "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
        "tfpd_exploration/src/posterior_carrier_v1/phase_b_v2.py",
        "tfpd_exploration/src/posterior_carrier_v1/phase_b_v3.py",
        "tfpd_exploration/src/tfpd_lane/arm_common.py",
        "tfpd_exploration/src/tfpd_lane/pop_robust.py",
        "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
        "sua_exploration/mc_maze/__init__.py",
        "sua_exploration/mc_maze/multisession_datamodule.py",
        "sua_exploration/mc_maze/unit_side_features.py",
        "sua_exploration/mc_maze/datamodule.py",
        "streaming_calibration_exp/src/models/components/spint.py",
        "streaming_calibration_exp/src/models/components/streaming_encoders.py",
        "streaming_calibration_exp/src/models/components/streaming_spint.py",
        "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    )


# ---------------------------------------------------------------------------
# Deferred physical evaluator
# ---------------------------------------------------------------------------
#
# The remaining definitions deliberately keep all tensor/runtime imports inside
# methods reached only after a reviewed score attempt is durable.  The static
# contract imports this file via a file-spec loader, so importing the module
# itself must remain data-, Torch-, CUDA-, and result-root-free.


def _module_from_exact_path(name: str, path: Path) -> Any:
    """Load one closure-bound runtime module without ambient path discovery."""
    import importlib.util
    import sys

    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PhysicalScoreError(f"runtime module path is missing/aliased: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PhysicalScoreError(f"cannot load closure-bound runtime module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _tensor_digest(torch: Any, value: Any) -> str:
    """Canonical tensor binding used only after reviewed runtime entry."""
    if not torch.is_tensor(value):
        raise PhysicalScoreError("tensor digest received a non-tensor")
    item = value.detach().cpu().contiguous()
    if item.is_floating_point():
        item = item + 0  # stable signed-zero semantics.
    digest = hashlib.sha256()
    digest.update(str(item.dtype).encode("utf-8"))
    digest.update(str(tuple(int(part) for part in item.shape)).encode("utf-8"))
    if int(item.numel()):
        digest.update(item.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _array_digest(value: Any) -> str:
    """Hash a contiguous NumPy-like array after the parser has produced it."""
    return _digest(value.tobytes())


@dataclass
class HeldDataRoot:
    """One descriptor-held target data root and its named-parent identity."""

    root: Path
    parent: Path
    identity: tuple[int, int]
    parent_identity: tuple[int, int]
    descriptor: int
    parent_descriptor: int
    _closed: bool = False

    @classmethod
    def from_environment(cls, variable: str) -> "HeldDataRoot":
        value = os.environ.get(variable)
        if not isinstance(value, str) or not value:
            raise PhysicalScoreError(f"physical scorer requires {variable}")
        root = Path(value).absolute()
        parent = root.parent
        identity = _directory_identity(root)
        parent_identity = _directory_identity(parent)
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        parent_fd = -1
        try:
            parent_fd = os.open(parent, flags)
            descriptor = os.open(root.name, flags, dir_fd=parent_fd)
        except OSError as error:
            if parent_fd >= 0:
                os.close(parent_fd)
            raise PhysicalScoreError(f"cannot hold {variable} root through O_NOFOLLOW") from error
        result = cls(root, parent, identity, parent_identity, descriptor, parent_fd)
        try:
            result.reverify()
        except BaseException:
            result.close()
            raise
        return result

    def reverify(self) -> None:
        if self._closed:
            raise PhysicalScoreError("held data root is closed")
        root_info, parent_info = os.fstat(self.descriptor), os.fstat(self.parent_descriptor)
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or not stat.S_ISDIR(parent_info.st_mode)
            or (int(root_info.st_dev), int(root_info.st_ino)) != self.identity
            or (int(parent_info.st_dev), int(parent_info.st_ino)) != self.parent_identity
            or _directory_identity(self.root) != self.identity
            or _directory_identity(self.parent) != self.parent_identity
        ):
            raise PhysicalScoreError("held data-root descriptor/path identity drift")
        try:
            named = os.stat(self.root.name, dir_fd=self.parent_descriptor, follow_symlinks=False)
        except OSError as error:
            raise PhysicalScoreError("held data-root named entry disappeared") from error
        if (
            not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (int(named.st_dev), int(named.st_ino)) != self.identity
        ):
            raise PhysicalScoreError("held data-root named entry drift")

    def open_asset(
        self,
        *,
        relative: str,
        expected_bytes: int,
        expected_sha256: str,
        surface: str,
        session: str,
    ) -> "HeldInputAsset":
        relative = _safe_relative(relative, "evaluation asset relative")
        if type(expected_bytes) is not int or expected_bytes <= 0:
            raise PhysicalScoreError("evaluation asset expected byte count drift")
        expected_sha256 = _sha(expected_sha256, "evaluation asset expected SHA")
        self.reverify()
        current_fd = self.descriptor
        intermediates: list[int] = []
        try:
            for piece in Path(relative).parts[:-1]:
                child = os.open(
                    piece,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current_fd,
                )
                intermediates.append(child)
                current_fd = child
            name = Path(relative).name
            before = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
        except OSError as error:
            for descriptor_to_close in reversed(intermediates):
                os.close(descriptor_to_close)
            raise PhysicalScoreError("evaluation asset cannot be opened through held data root") from error
        for descriptor_to_close in reversed(intermediates):
            os.close(descriptor_to_close)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or _identity(before) != _identity(opened)
            or int(before.st_size) != expected_bytes
        ):
            os.close(descriptor)
            raise PhysicalScoreError("evaluation asset type/identity/size drift")
        # Hash the held descriptor once before parsing.  This has no cache
        # side effect and prevents the parser from consuming an unverified
        # pathname.  Reset to offset zero for the later private snapshot copy.
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            os.close(descriptor)
            raise PhysicalScoreError("evaluation asset body SHA drift")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return HeldInputAsset(
            root=self,
            relative=relative,
            surface=surface,
            session=session,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            descriptor=descriptor,
            identity=_identity(before),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in (self.descriptor, self.parent_descriptor):
            try:
                os.close(descriptor)
            except OSError:
                pass


@dataclass
class HeldInputAsset:
    """Verified NWB bytes held open while a private no-cache copy is parsed."""

    root: HeldDataRoot
    relative: str
    surface: str
    session: str
    expected_bytes: int
    expected_sha256: str
    descriptor: int
    identity: tuple[int, int, int]
    _closed: bool = False

    def reverify(self) -> None:
        if self._closed:
            raise PhysicalScoreError("held evaluation asset is closed")
        self.root.reverify()
        opened = os.fstat(self.descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _identity(opened) != self.identity
            or int(opened.st_size) != self.expected_bytes
        ):
            raise PhysicalScoreError("held evaluation asset descriptor drift")
        # Resolve every path component below the held root again without
        # following a symlink, then compare only the final inode/size.
        current_fd = self.root.descriptor
        intermediates: list[int] = []
        try:
            for piece in Path(self.relative).parts[:-1]:
                child = os.open(
                    piece,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current_fd,
                )
                intermediates.append(child)
                current_fd = child
            info = os.stat(Path(self.relative).name, dir_fd=current_fd, follow_symlinks=False)
        except OSError as error:
            raise PhysicalScoreError("held evaluation asset named path drift") from error
        finally:
            for descriptor in reversed(intermediates):
                os.close(descriptor)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or _identity(info) != self.identity:
            raise PhysicalScoreError("held evaluation asset named inode drift")

    def private_snapshot(self) -> "PrivateInputSnapshot":
        """Copy verified bytes to a parser-private temporary filename once."""
        import tempfile

        self.reverify()
        directory = tempfile.TemporaryDirectory(prefix="posterior_carrier_score_", dir="/tmp")
        path = Path(directory.name) / Path(self.relative).name
        try:
            output = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.lseek(self.descriptor, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                written = 0
                while True:
                    block = os.read(self.descriptor, 1 << 20)
                    if not block:
                        break
                    digest.update(block)
                    _write_bytes(output, block)
                    written += len(block)
                os.fsync(output)
            finally:
                os.close(output)
            os.chmod(path, 0o400)
            if written != self.expected_bytes or digest.hexdigest() != self.expected_sha256:
                raise PhysicalScoreError("private parser snapshot body digest/size drift")
            snapshot = PrivateInputSnapshot(directory=directory, path=path, expected_bytes=written,
                                            expected_sha256=self.expected_sha256)
            snapshot.reverify()
            return snapshot
        except BaseException:
            directory.cleanup()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self.descriptor)
        except OSError:
            pass


@dataclass
class PrivateInputSnapshot:
    """Ephemeral parser copy; never a cache and never an authority artifact."""

    directory: Any
    path: Path
    expected_bytes: int
    expected_sha256: str

    def reverify(self) -> None:
        info = os.lstat(self.path)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or int(info.st_size) != self.expected_bytes:
            raise PhysicalScoreError("private parser snapshot identity/size drift")
        descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o400:
                raise PhysicalScoreError("private parser snapshot mode drift")
            body = _read_all(descriptor)
        finally:
            os.close(descriptor)
        if _digest(body) != self.expected_sha256:
            raise PhysicalScoreError("private parser snapshot SHA drift")

    def close(self) -> None:
        self.directory.cleanup()


def _write_bytes(descriptor: int, body: bytes) -> None:
    offset = 0
    while offset < len(body):
        count = os.write(descriptor, body[offset:])
        if count <= 0:
            raise PhysicalScoreError("short private snapshot write")
        offset += count


def sealed_governing_table_sha256(baseline: Mapping[str, object]) -> str:
    """Bind the exact last-bin D-SWA table used for native M30 parity."""
    rows = sealed_governing_rows(baseline)
    payload = {
        surface: [dict(row) for row in rows[surface]]
        for surface in ("within", "external")
    }
    return _digest(_json(payload))


@dataclass
class _PhysicalSession:
    """One materialized input shared by every model/mode/budget cell."""

    surface: str
    session: str
    held: HeldInputAsset
    neural: Any
    behavior: Any
    calibration_m30: Any
    starts: Any
    point_side: Mapping[int, Any]
    posterior_view: Mapping[int, Any]
    posterior_raw: Mapping[int, Any]
    last_targets: Any
    last_valid_mask: Any
    input_record: Any
    raw_axis_proofs: Mapping[int, Mapping[str, object]]


def _valid_last_bin_authority(np: Any, *, behavior: Any, starts: Any) -> tuple[Any, Any, str, str, int]:
    """Extract the exact governing query targets before any model forward."""
    if behavior.ndim != 2 or behavior.shape[1] != 2 or starts.ndim != 1 or int(starts.size) <= 0:
        raise PhysicalScoreError("last-bin authority input shape drift")
    target = np.ascontiguousarray(behavior[starts + 49], dtype=np.float32)
    mask = np.ascontiguousarray(np.all(target != -1.0, axis=1), dtype=np.uint8)
    if target.shape != (int(starts.size), 2) or not bool(mask.all()) or int(mask.sum()) != int(starts.size):
        raise PhysicalScoreError("governing query contains invalid/padded target windows")
    return target, mask, _array_digest(target), _array_digest(mask), int(mask.sum())


def _raw_t4_axis_proof(
    np: Any,
    *,
    session: str,
    raw: Any,
    budget: int,
    record: Any,
    metadata: Any,
) -> dict[str, object]:
    """Prove a raw T4 row has the exact SUA neural-unit ordering."""
    neural = getattr(record, "neural", None)
    channel_ids = getattr(record, "channel_ids", None)
    source_units = getattr(record, "source_unit_count", None)
    if neural is None or not isinstance(budget, int) or budget not in {4, 10, 30}:
        raise PhysicalScoreError("raw T4 proof input budget/record drift")
    unit_count = int(neural.shape[1])
    channels = np.ascontiguousarray(channel_ids, dtype=np.int64) if channel_ids is not None else None
    if (
        raw.shape != (unit_count, 4)
        or source_units != unit_count
        or channels is None
        or channels.shape != (unit_count,)
        or not np.array_equal(channels, np.arange(unit_count, dtype=np.int64))
        or getattr(metadata, "feature_group", None) != "t4"
        or getattr(metadata, "pool_size", None) != budget
        or getattr(metadata, "signal_view", "sua") != "sua"
    ):
        raise PhysicalScoreError("raw T4/SUA neural unit-axis proof drift")
    unit_ids = tuple(f"{session}:channel:{int(item)}" for item in channels.tolist())
    if len(set(unit_ids)) != unit_count:
        raise PhysicalScoreError("raw T4 canonical unit-ID uniqueness drift")
    proof = {
        "schema": "posterior_carrier_matched_score_raw_t4_sua_axis_v1",
        "session": session,
        "pool_size": budget,
        "feature_group": "t4",
        "signal_view": "sua",
        "source_unit_count": unit_count,
        "neural_unit_count": unit_count,
        "channel_ids_are_exact_int64_arange": True,
        "ordered_unit_ids_sha256": _digest(_json(list(unit_ids))),
        "raw_t4_sha256": _array_digest(np.ascontiguousarray(raw, dtype=np.float32)),
        "prefix_selection_semantics": (
            "closure_bound_compute_unit_side_features_uncached_t4_uses_"
            "list_datamodule_rewarded_trials_then_exact_first_pool_size"
        ),
        "function_semantics": (
            "closure_bound_compute_unit_side_features_uncached_t4_rows_follow_nwb_sua_unit_order"
        ),
    }
    return {**proof, "body_sha256": _digest(_json(proof))}


def _validate_raw_t4_axis_proof(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema", "session", "pool_size", "feature_group", "signal_view", "source_unit_count",
        "neural_unit_count", "channel_ids_are_exact_int64_arange", "ordered_unit_ids_sha256",
        "raw_t4_sha256", "prefix_selection_semantics", "function_semantics", "body_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PhysicalScoreError("raw T4/SUA axis proof schema drift")
    body = {key: value[key] for key in expected - {"body_sha256"}}
    if (
        value.get("schema") != "posterior_carrier_matched_score_raw_t4_sua_axis_v1"
        or not isinstance(value.get("session"), str) or not value["session"]
        or value.get("pool_size") not in {4, 10, 30}
        or value.get("feature_group") != "t4" or value.get("signal_view") != "sua"
        or type(value.get("source_unit_count")) is not int or value["source_unit_count"] < 1
        or value.get("neural_unit_count") != value["source_unit_count"]
        or value.get("channel_ids_are_exact_int64_arange") is not True
        or value.get("prefix_selection_semantics") != (
            "closure_bound_compute_unit_side_features_uncached_t4_uses_"
            "list_datamodule_rewarded_trials_then_exact_first_pool_size"
        )
        or value.get("function_semantics") != (
            "closure_bound_compute_unit_side_features_uncached_t4_rows_follow_nwb_sua_unit_order"
        )
        or value.get("body_sha256") != _digest(_json(body))
    ):
        raise PhysicalScoreError("raw T4/SUA axis proof value/digest drift")
    _sha(value.get("ordered_unit_ids_sha256"), "raw T4 ordered unit IDs SHA")
    _sha(value.get("raw_t4_sha256"), "raw T4 bytes SHA")
    _sha(value.get("body_sha256"), "raw T4 proof body SHA")
    return dict(value)


def _posterior_prior_from_payload(core: Any, value: Mapping[str, object]) -> Any:
    required = {
        "schema", "mu0", "source_mean_b", "tau_ac2", "tau_b2", "directional_prior_mean_exact_zero",
        "directional_prior_isotropic", "variance_floor", "source_roster", "source_roster_sha256",
        "raw_m30_t4_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise PhysicalScoreError("frozen posterior prior schema drift")
    roster = value.get("source_roster")
    if not isinstance(roster, list):
        raise PhysicalScoreError("frozen posterior prior roster drift")
    try:
        prior = core.SourcePrior(
            source_mean_b=float(value["source_mean_b"]),
            tau_ac2=float(value["tau_ac2"]),
            tau_b2=float(value["tau_b2"]),
            source_roster=tuple(roster),
            raw_m30_t4_sha256=value["raw_m30_t4_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PhysicalScoreError("frozen posterior prior cannot be reconstructed") from error
    if prior.payload() != dict(value):
        raise PhysicalScoreError("frozen posterior prior exact payload drift")
    return prior


def _posterior_normalizer_from_payload(core: Any, torch: Any, value: Mapping[str, object]) -> Any:
    required = {
        "schema", "source_roster", "source_roster_sha256", "row_order", "row_count",
        "per_budget_row_counts", "per_budget_raw_rows_sha256", "raw_rows_sha256", "mean_float64",
        "std_float64", "ddof", "source_only", "contains_only_deterministic_posterior_means",
        "all_zero_raw_rows_retained", "body_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise PhysicalScoreError("frozen posterior normalizer schema drift")
    try:
        counts = {int(key): item for key, item in value["per_budget_row_counts"].items()}
        hashes = {int(key): item for key, item in value["per_budget_raw_rows_sha256"].items()}
        normalizer = core.PosteriorSourceT4Normalizer(
            mean=torch.tensor(value["mean_float64"], dtype=torch.float64, device="cpu"),
            std=torch.tensor(value["std_float64"], dtype=torch.float64, device="cpu"),
            source_roster=tuple(value["source_roster"]),
            row_order=value["row_order"],
            row_count=value["row_count"],
            per_budget_row_counts=counts,
            per_budget_raw_rows_sha256=hashes,
            raw_rows_sha256=value["raw_rows_sha256"],
            body_sha256=value["body_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PhysicalScoreError("frozen posterior normalizer cannot be reconstructed") from error
    if normalizer.payload() != dict(value):
        raise PhysicalScoreError("frozen posterior normalizer exact payload drift")
    return normalizer


def _posterior_runtime_view_from_authority(
    core: Any,
    torch: Any,
    authority: Any,
    *,
    device: Any,
) -> Any:
    """Make the only device-local view of a CPU-frozen posterior authority.

    ``PosteriorSourceT4Normalizer`` intentionally owns CPU/float64 source
    moments: that is the immutable fit authority and must never be moved or
    refit by the scorer.  ``posterior_mean_view`` is stricter than
    ``normalize_raw``: it requires its normalizer object itself to share the
    posterior's device and dtype.  A separate ``FrozenSourceT4Normalizer`` is
    therefore the execution view.  It preserves the authority's exact body
    SHA and copies only the already-validated four float64 values to the
    reviewed device.  No target statistic is created and the CPU authority
    remains available for payload revalidation.
    """
    try:
        view = core.FrozenSourceT4Normalizer(
            mean=authority.mean.detach().to(device=device, dtype=torch.float64).clone(),
            std=authority.std.detach().to(device=device, dtype=torch.float64).clone(),
            authority_sha256=authority.authority_sha256,
        )
    except Exception as error:
        raise PhysicalScoreError("cannot construct device-local posterior-normalizer view") from error
    if (
        view.authority_sha256 != authority.body_sha256
        or view.mean.device != device
        or view.std.device != device
        or view.mean.dtype != torch.float64
        or view.std.dtype != torch.float64
        or tuple(view.mean.shape) != (4,)
        or tuple(view.std.shape) != (4,)
    ):
        raise PhysicalScoreError("posterior-normalizer device view authority/device/dtype drift")
    return view


class PhysicalPosteriorMatchedBackend:
    """Deferred no-cache physical backend for the 16-cell matched matrix.

    Construction is deliberately inert.  The route owner supplies the static
    contract module, a root-reviewed preflight, and an already identity-bound
    score identity.  Only ``prepare``—which the core calls after writing an
    immutable attempt—imports Torch, attests the reviewed GPU, loads model
    bytes, or opens the imported full-result mirror.  Evaluation assets are
    still later: ``resolve_inputs`` is the first method that may resolve a
    within/external pathname.
    """

    _BATCH_SIZE = 32
    _DEVICE_FIELDS = (
        "cuda_visible_devices", "cuda_device_order", "logical_device", "uuid", "bdf", "name",
        "nvidia_smi_memory_total_mib", "torch_total_memory_bytes", "torch_version",
        "torch_cuda_version", "cudnn_version",
    )

    def __init__(self, *, root: Path, preflight: Mapping[str, object], contract: Any) -> None:
        if not isinstance(preflight, Mapping):
            raise ValueError("physical posterior score preflight must be a mapping")
        self._root = Path(root).absolute()
        self._preflight = json.loads(_json(preflight))
        self._contract = contract
        self._runtime: Mapping[str, Any] | None = None
        self._full: CompletedFullMirror | None = None
        self._sealed: SealedCellDMaterial | None = None
        self._models: dict[str, Any] = {}
        self._state_at_load: dict[str, str] = {}
        self._sessions: dict[str, dict[str, _PhysicalSession]] = {"within": {}, "external": {}}
        self._held_roots: list[HeldDataRoot] = []
        self._held_assets: list[HeldInputAsset] = []
        self._prior: Any | None = None
        # Keep the CPU ``PosteriorSourceT4Normalizer`` as immutable source
        # evidence.  The separate runtime view is the only object passed to
        # posterior_mean_view, whose device/dtype contract is intentionally
        # strict.
        self._posterior_normalizer_authority: Any | None = None
        self._posterior_normalizer_view: Any | None = None
        self._sealed_rows: dict[str, tuple[dict[str, object], ...]] | None = None
        self._closed = False

    @staticmethod
    def _all_gradients_none(model: Any) -> bool:
        return all(parameter.grad is None for parameter in model.parameters())

    @staticmethod
    def _lazy_topology(model: Any, torch: Any) -> tuple[int, tuple[str, ...]]:
        from torch.nn.parameter import UninitializedParameter

        initialized = 0
        lazy: list[str] = []
        for name, parameter in model.named_parameters():
            if isinstance(parameter, UninitializedParameter):
                lazy.append(name)
            else:
                initialized += int(parameter.numel())
        return initialized, tuple(sorted(lazy))

    def _require_base_cell_d_topology(self, model: Any, torch: Any, *, label: str) -> None:
        initialized, lazy = self._lazy_topology(model, torch)
        if initialized != 3_510_842 or lazy != (
            "decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight",
        ):
            raise PhysicalScoreError(f"{label} exact sealed Cell-D parameter/lazy topology drift")

    def _require_posterior_wrapper_topology(self, model: Any, torch: Any, *, label: str) -> None:
        initialized, lazy = self._lazy_topology(model, torch)
        if initialized != 3_510_842 or lazy != (
            "cell_d.decoder.fc_id_in.0.bias", "cell_d.decoder.fc_id_in.0.weight",
        ):
            raise PhysicalScoreError(f"{label} exact posterior-wrapper parameter/lazy topology drift")
        audit = model.preservation_audit()
        if (
            audit.base_live_parameter_count != 3_510_842
            or audit.wrapper_live_parameter_count != 3_510_842
            or audit.wrapper_new_parameter_count != 0
            or audit.parameter_object_ids_identical is not True
        ):
            raise PhysicalScoreError(f"{label} posterior wrapper changed Cell-D parameter topology")

    def _runtime_device_attestation(self, *, torch: Any, torchmetrics: Any) -> dict[str, object]:
        """Reproduce every reviewed device field exactly; no rounding bridge."""
        import subprocess
        import sys

        expected = self._preflight.get("device_contract")
        if not isinstance(expected, Mapping) or set(expected) != set(self._DEVICE_FIELDS):
            raise PhysicalScoreError("physical preflight device contract is unavailable")
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES") != expected["cuda_visible_devices"]
            or os.environ.get("CUDA_DEVICE_ORDER") != expected["cuda_device_order"]
            or sys.flags.no_user_site != 1
            or expected["cuda_visible_devices"] != "0"
            or expected["cuda_device_order"] != "PCI_BUS_ID"
            or expected["logical_device"] != "cuda:0"
        ):
            raise PhysicalScoreError("physical scorer environment/device-order authority drift")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise PhysicalScoreError("physical scorer requires exactly one visible CUDA device as cuda:0")
        try:
            output = subprocess.run(
                ["nvidia-smi", "--id=0", "--query-gpu=uuid,pci.bus_id,name,memory.total",
                 "--format=csv,noheader,nounits"],
                check=True, text=True, capture_output=True,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            raise PhysicalScoreError("physical scorer cannot attest nvidia-smi GPU0 identity") from error
        if len(output) != 1:
            raise PhysicalScoreError("physical scorer nvidia-smi GPU topology drift")
        fields = tuple(item.strip() for item in output[0].split(","))
        if len(fields) != 4:
            raise PhysicalScoreError("physical scorer nvidia-smi response schema drift")
        uuid, bdf, name, nominal_text = fields
        try:
            nominal = int(nominal_text)
        except ValueError as error:
            raise PhysicalScoreError("physical scorer nominal GPU memory is malformed") from error
        properties = torch.cuda.get_device_properties(0)
        actual = {
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
            "logical_device": "cuda:0",
            "uuid": uuid,
            "bdf": bdf.upper(),
            "name": name,
            "nvidia_smi_memory_total_mib": nominal,
            "torch_total_memory_bytes": int(properties.total_memory),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cudnn_version": int(torch.backends.cudnn.version()),
        }
        if actual != dict(expected):
            raise PhysicalScoreError("physical scorer GPU/runtime contract differs from reviewed preflight")
        if str(torchmetrics.__version__) != "1.5.1":
            raise PhysicalScoreError("physical scorer requires TorchMetrics 1.5.1")
        return actual

    def _load_runtime(self) -> Mapping[str, Any]:
        """Deferred runtime import after an immutable score attempt exists."""
        if self._runtime is not None:
            return self._runtime
        import importlib
        import sys

        for extra in (
            self._root / "tfpd_exploration",
            self._root / "sua_exploration",
            self._root / "streaming_calibration_exp",
        ):
            rendered = str(extra)
            if rendered not in sys.path:
                sys.path.insert(0, rendered)
        import numpy as np
        import torch
        import torchmetrics

        attestation = self._runtime_device_attestation(torch=torch, torchmetrics=torchmetrics)
        # Do not import the broad ``tfpd_lane`` package initializer: it pulls
        # unrelated exploratory mechanisms.  These three exact files are the
        # only shared runtime surfaces required here.
        arm_common = _module_from_exact_path(
            "_posterior_carrier_score_arm_common",
            self._root / "tfpd_exploration/src/tfpd_lane/arm_common.py",
        )
        pop_robust = _module_from_exact_path(
            "_posterior_carrier_score_pop_robust",
            self._root / "tfpd_exploration/src/tfpd_lane/pop_robust.py",
        )
        metric = _module_from_exact_path(
            "_posterior_carrier_score_matched_scorer",
            self._root / "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
        )
        if not callable(getattr(metric, "session_r2", None)):
            raise PhysicalScoreError("closure-bound last-bin session metric loader drift")
        core = importlib.import_module("src.posterior_carrier_v1.core")
        source_adapter_v2 = importlib.import_module("src.posterior_carrier_v1.source_adapter_v2")
        from mc_maze.multisession_datamodule import load_dandi688_session
        from mc_maze.unit_side_features import compute_unit_side_features_uncached

        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        self._runtime = {
            "np": np,
            "torch": torch,
            "torchmetrics": torchmetrics,
            "arm_common": arm_common,
            "pop_robust": pop_robust,
            "metric": metric,
            "core": core,
            "source_adapter_v2": source_adapter_v2,
            "load_dandi688_session": load_dandi688_session,
            "compute_unit_side_features_uncached": compute_unit_side_features_uncached,
            "device": torch.device("cuda:0"),
            "device_attestation": attestation,
        }
        return self._runtime

    def _safe_weights_only_load(self, body: bytes, *, label: str) -> Mapping[str, Any]:
        """Load state bytes under a local, nonpersistent safe allowlist."""
        import io

        runtime = self._load_runtime()
        torch = runtime["torch"]
        from torch.nn.parameter import UninitializedParameter
        from torch.torch_version import TorchVersion

        try:
            with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
                value = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        except Exception as error:
            raise PhysicalScoreError(f"{label} weights-only state load failed") from error
        if not isinstance(value, Mapping):
            raise PhysicalScoreError(f"{label} weights-only payload root drift")
        return value

    def _fresh_base_cell_d(self, state: Mapping[str, Any], *, label: str, place_on_device: bool) -> Any:
        runtime = self._load_runtime()
        torch, pop_robust = runtime["torch"], runtime["pop_robust"]
        model = pop_robust.build_population_robustness_model(seed=42, cell="D")
        self._require_base_cell_d_topology(model, torch, label=f"{label} fresh")
        if set(state) != set(model.state_dict()):
            raise PhysicalScoreError(f"{label} state-key topology drift")
        model.load_state_dict(state, strict=True)
        self._require_base_cell_d_topology(model, torch, label=f"{label} strict CPU")
        if place_on_device:
            model = model.to(runtime["device"])
            self._require_base_cell_d_topology(model, torch, label=f"{label} strict GPU")
        model.eval()
        if model.training or not self._all_gradients_none(model):
            raise PhysicalScoreError(f"{label} eval/gradient invariant drift")
        return model

    def _fresh_posterior_wrapper(self, state: Mapping[str, Any], *, label: str, place_on_device: bool) -> Any:
        runtime = self._load_runtime()
        core = runtime["core"]
        base = runtime["pop_robust"].build_population_robustness_model(seed=42, cell="D")
        wrapper = core.CellDPosteriorWrapper(base)
        self._require_posterior_wrapper_topology(wrapper, runtime["torch"], label=f"{label} fresh")
        if set(state) != set(wrapper.state_dict()):
            raise PhysicalScoreError(f"{label} state-key topology drift")
        wrapper.load_state_dict(state, strict=True)
        self._require_posterior_wrapper_topology(wrapper, runtime["torch"], label=f"{label} strict CPU")
        if place_on_device:
            wrapper = wrapper.to(runtime["device"])
            self._require_posterior_wrapper_topology(wrapper, runtime["torch"], label=f"{label} strict GPU")
        wrapper.eval()
        if wrapper.training or not self._all_gradients_none(wrapper):
            raise PhysicalScoreError(f"{label} eval/gradient invariant drift")
        return wrapper

    def _validate_checkpoint_state(
        self,
        *,
        body: bytes,
        epoch: int,
        terminal: Mapping[str, object],
        expected_state_sha256: str,
    ) -> Mapping[str, Any]:
        """Strictly load every final-four state before accepting full SWA."""
        payload = self._safe_weights_only_load(body, label=f"full checkpoint {epoch}")
        required = {
            "schema", "epoch", "global_step", "model_state", "optimizer_state",
            "model_state_sha256", "binding",
        }
        if set(payload) != required or payload.get("schema") != "posterior_carrier_full_checkpoint_v1":
            raise PhysicalScoreError("full checkpoint schema drift")
        if (
            payload.get("epoch") != epoch
            or payload.get("global_step") != (epoch + 1) * 33_925
            or not isinstance(payload.get("model_state"), Mapping)
            or payload.get("model_state_sha256") != expected_state_sha256
            or not isinstance(payload.get("binding"), Mapping)
        ):
            raise PhysicalScoreError("full checkpoint epoch/state binding drift")
        binding = payload["binding"]
        if (
            binding.get("cell") != "POSTERIOR_CARRIER_BUDGETMIX_D_SEED42"
            or binding.get("launch_sha256") != terminal.get("launch_sha256")
            or binding.get("source_authority_sha256") != terminal.get("source_authority_sha256")
            or binding.get("full_launch_closure") != terminal.get("full_launch_closure")
        ):
            raise PhysicalScoreError("full checkpoint launch/source/closure binding drift")
        fresh = self._fresh_posterior_wrapper(payload["model_state"], label=f"full checkpoint {epoch}", place_on_device=False)
        digest = self._load_runtime()["arm_common"].state_sha256(fresh)
        if digest != expected_state_sha256:
            raise PhysicalScoreError("full checkpoint stored state digest does not match strict-loaded tensors")
        return payload

    @staticmethod
    def _state_equal(torch: Any, left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        from torch.nn.parameter import UninitializedParameter

        if set(left) != set(right):
            return False
        for key in left:
            a, b = left[key], right[key]
            if isinstance(a, UninitializedParameter) or isinstance(b, UninitializedParameter):
                if type(a) is not type(b):
                    return False
                continue
            if not torch.is_tensor(a) or not torch.is_tensor(b) or not torch.equal(a, b):
                return False
        return True

    def _validate_swa_against_checkpoints(
        self,
        *,
        swa_body: bytes,
        checkpoints: Mapping[int, Mapping[str, Any]],
        terminal: Mapping[str, object],
        expected_swa_state_sha256: str,
    ) -> Mapping[str, Any]:
        """Rebuild FP64 final-four mean instead of trusting a proof boolean."""
        runtime = self._load_runtime()
        torch = runtime["torch"]
        payload = self._safe_weights_only_load(swa_body, label="full final-four SWA")
        required = {"schema", "state", "state_sha256", "proof", "binding"}
        if (
            set(payload) != required
            or payload.get("schema") != "posterior_carrier_full_swa_v1"
            or not isinstance(payload.get("state"), Mapping)
            or payload.get("state_sha256") != expected_swa_state_sha256
            or not isinstance(payload.get("proof"), Mapping)
            or not isinstance(payload.get("binding"), Mapping)
        ):
            raise PhysicalScoreError("full SWA schema/state binding drift")
        binding = payload["binding"]
        if (
            binding.get("cell") != "POSTERIOR_CARRIER_BUDGETMIX_D_SEED42"
            or binding.get("launch_sha256") != terminal.get("launch_sha256")
            or binding.get("source_authority_sha256") != terminal.get("source_authority_sha256")
            or binding.get("full_launch_closure") != terminal.get("full_launch_closure")
        ):
            raise PhysicalScoreError("full SWA exact launch/source/closure binding drift")
        proof = payload["proof"]
        expected_states = {str(epoch): checkpoints[epoch]["model_state_sha256"] for epoch in (44, 45, 46, 47)}
        required_proof = {
            "checkpoint_epochs", "checkpoint_state_sha256s", "fresh_strict_load", "eval_mode",
            "repeat_bitwise_equal", "state_unchanged", "eval_no_sampling", "prediction_shape",
            "prediction_sha256", "state_digest_before_eval", "state_digest_after_eval", "boundaries",
        }
        if (
            set(proof) != required_proof or proof.get("checkpoint_epochs") != [44, 45, 46, 47]
            or proof.get("checkpoint_state_sha256s") != expected_states
            or proof.get("fresh_strict_load") is not True or proof.get("eval_mode") is not True
            or proof.get("repeat_bitwise_equal") is not True or proof.get("state_unchanged") is not True
            or proof.get("eval_no_sampling") is not True or proof.get("prediction_shape") != [32, 50, 2]
            or proof.get("state_digest_before_eval") != proof.get("state_digest_after_eval")
        ):
            raise PhysicalScoreError("full SWA persisted proof binding drift")
        # Explicit arithmetic, including nonfloating/lazy contracts.  A
        # checkpoint load that merely says it was averaged is not evidence.
        checkpoint_states = [checkpoints[epoch]["model_state"] for epoch in (44, 45, 46, 47)]
        from torch.nn.parameter import UninitializedParameter
        expected: dict[str, Any] = {}
        keys = set(checkpoint_states[0])
        if any(set(state) != keys for state in checkpoint_states[1:]) or set(payload["state"]) != keys:
            raise PhysicalScoreError("full SWA/checkpoint state-key topology drift")
        for key in sorted(keys):
            values = [state[key] for state in checkpoint_states]
            if isinstance(values[0], UninitializedParameter):
                if any(type(value) is not type(values[0]) for value in values[1:]):
                    raise PhysicalScoreError("full SWA lazy checkpoint topology drift")
                expected[key] = values[-1]
                continue
            if any(not torch.is_tensor(value) for value in values):
                raise PhysicalScoreError("full SWA checkpoint state contains non-tensor")
            if any(tuple(value.shape) != tuple(values[0].shape) or value.dtype != values[0].dtype for value in values[1:]):
                raise PhysicalScoreError("full SWA checkpoint tensor shape/dtype drift")
            if values[0].is_floating_point():
                accumulator = values[0].double().clone()
                for value in values[1:]:
                    accumulator.add_(value.double())
                expected[key] = (accumulator / 4.0).to(values[0].dtype)
            else:
                if any(not torch.equal(value, values[0]) for value in values[1:]):
                    raise PhysicalScoreError("full SWA nonfloating checkpoint tensor drift")
                expected[key] = values[0].clone()
        if not self._state_equal(torch, payload["state"], expected):
            raise PhysicalScoreError("full SWA is not exact arithmetic final-four state mean")
        fresh = self._fresh_posterior_wrapper(payload["state"], label="full SWA", place_on_device=False)
        actual_state = runtime["arm_common"].state_sha256(fresh)
        if actual_state != expected_swa_state_sha256:
            raise PhysicalScoreError("full SWA strict-loaded state digest drift")
        return payload

    def _ensure_models(self) -> None:
        if self._models:
            if set(self._models) != {"sealed_cell_d", "posterior"}:
                raise PhysicalScoreError("physical matched model cache topology drift")
            return
        if self._full is None or self._sealed is None:
            raise PhysicalScoreError("physical models require completed full mirror and sealed Cell-D material")
        runtime = self._load_runtime()
        torch = runtime["torch"]
        full_terminal = self._full.terminal
        expected_checkpoint_states = full_terminal.get("checkpoint_state_sha256s")
        if not isinstance(expected_checkpoint_states, Mapping) or set(expected_checkpoint_states) != {"44", "45", "46", "47"}:
            raise PhysicalScoreError("full terminal final-four checkpoint state map drift")
        checkpoints: dict[int, Mapping[str, Any]] = {}
        for epoch in (44, 45, 46, 47):
            checkpoints[epoch] = self._validate_checkpoint_state(
                body=self._full.bodies[f"checkpoint-{epoch:02d}.pt"], epoch=epoch,
                terminal=full_terminal, expected_state_sha256=_sha(expected_checkpoint_states[str(epoch)], "terminal checkpoint state SHA"),
            )
        full_swa = self._validate_swa_against_checkpoints(
            swa_body=self._full.bodies["swa_final4.pt"], checkpoints=checkpoints,
            terminal=full_terminal,
            expected_swa_state_sha256=self._contract._sha(
                self._preflight["identity"]["full_training"]["swa_state_sha256"], "reviewed full SWA state SHA",
            ),
        )
        sealed_payload = self._safe_weights_only_load(self._sealed.swa_body, label="sealed Cell-D SWA")
        sealed_state = sealed_payload.get("state_dict", sealed_payload.get("state"))
        manifest = sealed_payload.get("swa_manifest")
        if (
            not isinstance(sealed_state, Mapping) or not isinstance(manifest, Mapping)
            or manifest.get("uninitialized_lazy_tensor_count") != 2
            or manifest.get("optimizer_state_included") is not False
            or manifest.get("fp64_arithmetic") is not True
            or not isinstance(manifest.get("components"), list) or len(manifest["components"]) != 4
        ):
            raise PhysicalScoreError("sealed Cell-D SWA payload/manifest drift")
        sealed = self._fresh_base_cell_d(sealed_state, label="sealed Cell-D SWA", place_on_device=True)
        posterior = self._fresh_posterior_wrapper(full_swa["state"], label="posterior full SWA", place_on_device=True)
        self._models = {"sealed_cell_d": sealed, "posterior": posterior}
        self._state_at_load = {
            name: runtime["arm_common"].state_sha256(model) for name, model in self._models.items()
        }
        if (
            self._state_at_load["posterior"] != full_swa["state_sha256"]
            or any(not isinstance(value, str) or len(value) != 64 for value in self._state_at_load.values())
            or any(model.training or not self._all_gradients_none(model) for model in self._models.values())
        ):
            raise PhysicalScoreError("strict physical model state/eval/gradient proof drift")

    def _prepare_frozen_carrier_authorities(self) -> None:
        if self._full is None:
            raise PhysicalScoreError("posterior carrier authority requires completed full mirror")
        runtime = self._load_runtime()
        core, torch = runtime["core"], runtime["torch"]
        prior_payload, normalizer_payload = source_authority_posterior_payloads(self._full.source_authority)
        preflight_normalizer = self._preflight["normalizers"]["posterior_distribution"]
        if (
            not isinstance(preflight_normalizer, Mapping)
            or preflight_normalizer.get("payload") != normalizer_payload
            or preflight_normalizer.get("body_sha256") != normalizer_payload.get("body_sha256")
        ):
            raise PhysicalScoreError("reviewed posterior normalizer does not equal full frozen source authority")
        self._prior = _posterior_prior_from_payload(core, prior_payload)
        authority = _posterior_normalizer_from_payload(core, torch, normalizer_payload)
        self._posterior_normalizer_authority = authority
        self._posterior_normalizer_view = _posterior_runtime_view_from_authority(
            core, torch, authority, device=runtime["device"],
        )

    def prepare(self, *, identity: Any, flags: Any) -> None:
        """Validate all immutable models before an evaluation asset is resolved."""
        if self._closed:
            raise PhysicalScoreError("closed physical backend cannot prepare")
        if self._preflight.get("identity") != self._contract.validate_score_identity(identity):
            raise PhysicalScoreError("physical backend/preflight score identity drift")
        provenance = provenance_from_payload(self._preflight.get("full_import_provenance"))
        full = load_completed_full_mirror(
            self._root / provenance.local_mirror_relative,
            provenance=provenance,
            expected_terminal_sha256=identity.full_training.terminal_sha256,
            expected_swa_sha256=identity.full_training.swa_sha256,
            expected_swa_state_sha256=identity.full_training.swa_state_sha256,
            expected_source_authority_sha256=identity.full_training.source_authority_sha256,
            expected_checkpoint_sha256s=identity.full_training.checkpoint_sha256s,
            expected_full_closure_sha256=identity.full_training.full_closure_sha256,
        )
        sealed = load_sealed_cell_d_material(self._root)
        if (
            sealed.terminal_sha256 != identity.sealed_cell_d.terminal_sha256
            or sealed.swa_sha256 != identity.sealed_cell_d.swa_sha256
            or sealed_governing_table_sha256(sealed.baseline) != identity.sealed_cell_d.m30_ols_point_last_bin_table_sha256
        ):
            raise PhysicalScoreError("sealed Cell-D terminal/SWA/governing-table identity drift")
        self._full, self._sealed = full, sealed
        self._sealed_rows = sealed_governing_rows(sealed.baseline)
        self._load_runtime()
        self._prepare_frozen_carrier_authorities()
        self._ensure_models()
        if flags.h1_opened or flags.formal_opened:
            raise PhysicalScoreError("physical backend crossed forbidden H1/formal boundary during pre-input prepare")

    def _ordinary_point_normalizer(self) -> tuple[Any, Any, str]:
        runtime = self._load_runtime()
        torch = runtime["torch"]
        value = self._preflight.get("normalizers", {}).get("ordinary_point")
        if not isinstance(value, Mapping):
            raise PhysicalScoreError("reviewed ordinary point normalizer is unavailable")
        mean, std = value.get("mean"), value.get("std")
        if (
            not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 4 or len(std) != 4
            or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in (*mean, *std))
            or value.get("system_comparison_role")
            != "sealed_cell_d_training_time_ols_normalizer_reused_for_m30_m10_m4_replay"
            or value.get("not_point_budget_mix_trained_control") is not True
            or value.get("sealed_cell_d_point_replay_authority_sha256")
            != self._preflight["identity"]["sealed_cell_d"]["sealed_point_replay_authority_sha256"]
        ):
            raise PhysicalScoreError("reviewed sealed Cell-D ordinary point-normalizer provenance/schema drift")
        mean_tensor = torch.tensor(mean, dtype=torch.float32, device=runtime["device"])
        std_tensor = torch.tensor(std, dtype=torch.float32, device=runtime["device"])
        if not bool(torch.isfinite(mean_tensor).all().item()) or not bool((torch.isfinite(std_tensor) & (std_tensor > 0)).all().item()):
            raise PhysicalScoreError("reviewed ordinary point normalizer finite/range drift")
        semantic = value.get("semantic_sha256")
        return mean_tensor, std_tensor, _sha(semantic, "ordinary point normalizer semantic SHA")

    def _behavior_normalizer(self) -> tuple[Any, Any, str]:
        runtime = self._load_runtime()
        np = runtime["np"]
        value = self._preflight.get("normalizers", {}).get("behavior")
        if not isinstance(value, Mapping):
            raise PhysicalScoreError("reviewed behavior normalizer is unavailable")
        mean, std = value.get("mean"), value.get("std")
        if (
            not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 2 or len(std) != 2
            or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in (*mean, *std))
        ):
            raise PhysicalScoreError("reviewed behavior normalizer numeric schema drift")
        mean_array = np.asarray(mean, dtype=np.float32)
        std_array = np.asarray(std, dtype=np.float32)
        if not bool(np.isfinite(mean_array).all()) or not bool((np.isfinite(std_array) & (std_array > 0)).all()):
            raise PhysicalScoreError("reviewed behavior normalizer finite/range drift")
        return mean_array, std_array, _sha(value.get("semantic_sha256"), "behavior normalizer semantic SHA")

    @staticmethod
    def _asset_rows(preflight: Mapping[str, object], *, surface: str) -> tuple[Mapping[str, object], ...]:
        assets = preflight.get("input_assets")
        if not isinstance(assets, Mapping) or not isinstance(assets.get(surface), list):
            raise PhysicalScoreError("reviewed input-asset table is unavailable")
        rows = tuple(assets[surface])
        if any(not isinstance(row, Mapping) for row in rows):
            raise PhysicalScoreError("reviewed input-asset row type drift")
        return rows

    @staticmethod
    def _raw_t4_by_budget(
        *,
        runtime: Mapping[str, Any],
        snapshot: PrivateInputSnapshot,
        session: str,
        record: Any,
    ) -> tuple[dict[int, Any], dict[int, dict[str, object]]]:
        np = runtime["np"]
        raw: dict[int, Any] = {}
        proofs: dict[int, dict[str, object]] = {}
        for budget in (30, 10, 4):
            values, metadata = runtime["compute_unit_side_features_uncached"](
                snapshot.path,
                feature_group="t4",
                pool_size=budget,
                bin_size_ms=20,
                window_size=50,
                trial_result_filter="R",
                signal_view="sua",
            )
            array = np.ascontiguousarray(values, dtype=np.float32)
            if not bool(np.isfinite(array).all()):
                raise PhysicalScoreError("recomputed raw OLS T4 contains nonfinite value")
            proof = _raw_t4_axis_proof(
                np, session=session, raw=array, budget=budget, record=record, metadata=metadata,
            )
            proofs[budget] = _validate_raw_t4_axis_proof(proof)
            raw[budget] = array
        return raw, proofs

    def _parse_one(self, *, asset: Mapping[str, object], flags: Any) -> _PhysicalSession:
        """Open one held source once and derive all M30/M10/M4 views from it."""
        runtime = self._load_runtime()
        np, torch = runtime["np"], runtime["torch"]
        surface = asset.get("surface")
        session = asset.get("session")
        if surface not in {"within", "external"} or not isinstance(session, str) or not session:
            raise PhysicalScoreError("physical input asset surface/session drift")
        root_variable = "SUBC_DATA_ROOT" if surface == "within" else "SUBM_DATA_ROOT"
        # One root capability per surface; this also prevents a later session
        # from silently changing the environment-selected root mid-pass.
        roots = [item for item in self._held_roots if item.root == Path(os.environ.get(root_variable, "")).absolute()]
        if roots:
            data_root = roots[0]
        else:
            data_root = HeldDataRoot.from_environment(root_variable)
            self._held_roots.append(data_root)
        held = data_root.open_asset(
            # Metadata retains ``sub-C/...`` or ``sub-M/...`` as a provenance
            # label.  The reviewed source-root capability, however, is the
            # subject directory itself, so the only legal live resolution is
            # ``SUB*_DATA_ROOT / basename(frozen_path)``.
            relative=Path(str(asset["frozen_path"])).name,
            expected_bytes=asset["bytes"], expected_sha256=asset["sha256"],
            surface=surface, session=session,
        )
        self._held_assets.append(held)
        if surface == "within":
            flags.within_opened = True
        else:
            flags.external_opened = True
        snapshot = held.private_snapshot()
        try:
            behavior_mean, behavior_std, _behavior_semantic = self._behavior_normalizer()
            record = runtime["load_dandi688_session"](
                snapshot.path,
                bin_size_ms=20,
                window_size=50,
                calibration_n_trials=30,
                max_trial_length=100,
                pad_value=-1.0,
                interpolate_trials=True,
                behavior_mean=behavior_mean,
                behavior_std=behavior_std,
                trial_result_filter="R",
                exclude_calibration_trials_from_windows=True,
                cache_dir=None,
                signal_view="sua",
            )
            raw_point, raw_proofs = self._raw_t4_by_budget(
                runtime=runtime, snapshot=snapshot, session=session, record=record,
            )
            # This exact V2 source-adapter primitive reads only the same first
            # thirty chronological rewarded label rows.  It cannot choose a
            # later target direction as a replacement and retains its fallback
            # proof for the one audited missing direction case.
            counts, exposure, theta, prefix_row_ids, theta_recovery = runtime["source_adapter_v2"]._source_counts_exposure_theta_v2(
                path=snapshot.path, session=session, expected_units=int(record.neural.shape[1]),
            )
            snapshot.reverify()
        finally:
            snapshot.close()
        held.reverify()
        neural = np.ascontiguousarray(getattr(record, "neural", None), dtype=np.float32)
        behavior = np.ascontiguousarray(getattr(record, "behavior", None), dtype=np.float32)
        calibration = np.ascontiguousarray(getattr(record, "calib_trials", None), dtype=np.float32)
        starts = np.ascontiguousarray(getattr(record, "valid_starts", None), dtype=np.int64)
        channels = getattr(record, "channel_ids", None)
        if (
            getattr(record, "name", None) != session or getattr(record, "signal_view", None) != "sua"
            or neural.ndim != 2 or neural.shape[1] < 2 or behavior.shape != (neural.shape[0], 2)
            or calibration.shape != (30, 100, neural.shape[1]) or starts.ndim != 1 or int(starts.size) <= 0
            or channels is None or len(channels) != neural.shape[1]
            or not bool(np.isfinite(neural).all()) or not bool(np.isfinite(behavior).all())
            or not bool(np.isfinite(calibration).all()) or not bool((starts[:-1] < starts[1:]).all())
            or int(starts.min()) < 0 or int(starts.max()) + 50 > neural.shape[0]
        ):
            raise PhysicalScoreError("physical no-cache session tensor/window/channel contract drift")
        channel_array = np.ascontiguousarray(channels, dtype=np.int64)
        if not np.array_equal(channel_array, np.arange(neural.shape[1], dtype=np.int64)):
            raise PhysicalScoreError("physical neural unit ordering is not exact SUA arange")
        counts = counts.to(dtype=torch.int64, device=runtime["device"])
        exposure = exposure.to(dtype=torch.float64, device=runtime["device"])
        theta = theta.to(dtype=torch.float64, device=runtime["device"])
        if (
            tuple(counts.shape) != (neural.shape[1], 30) or tuple(exposure.shape) != (30,)
            or tuple(theta.shape) != (30,) or len(prefix_row_ids) != 30
        ):
            raise PhysicalScoreError("physical direct-count/exposure/theta prefix shape drift")
        # Validate the exact V2 recovery evidence at the time it is consumed;
        # this catches a moved/forged fallback row before any posterior fit.
        runtime["source_adapter_v2"].validate_theta_recovery_evidence(
            theta_recovery, session=session, prefix_row_ids=prefix_row_ids,
            theta_sha256=runtime["core"].tensor_digest(theta),
        )
        # This is deliberately a *target-local* same-prefix validator.  Do
        # not call source_adapter_v2.theta_fallback_topology here: that
        # strict-27 aggregate gate is a source-training authority and would
        # make a target session's independently observed fallback topology a
        # covert source-cohort selection rule.
        theta_recovery = self._contract._validate_target_theta_recovery_evidence(
            theta_recovery, session=session,
        )
        ordinary_mean, ordinary_std, _ordinary_sha = self._ordinary_point_normalizer()
        if (
            self._prior is None
            or self._posterior_normalizer_authority is None
            or self._posterior_normalizer_view is None
        ):
            raise PhysicalScoreError("posterior carrier authorities were not prepared before target parse")
        point_side: dict[int, Any] = {}
        posterior_view: dict[int, Any] = {}
        posterior_raw: dict[int, Any] = {}
        core = runtime["core"]
        for budget in (30, 10, 4):
            raw_tensor = torch.as_tensor(raw_point[budget], dtype=torch.float32, device=runtime["device"])
            point = ((raw_tensor - ordinary_mean) / ordinary_std).detach().clone()
            if not bool(torch.isfinite(point).all().item()):
                raise PhysicalScoreError("ordinary OLS point carrier normalization produced nonfinite tensor")
            posterior = core.fit_conjugate_posterior(
                counts=counts[:, :budget], exposure=exposure[:budget], theta=theta[:budget], prior=self._prior,
            )
            view = core.posterior_mean_view(posterior, self._posterior_normalizer_view)
            if view.sampled or view.session_id is not None or view.epoch is not None:
                raise PhysicalScoreError("physical posterior evaluation path must use deterministic mean without sampling")
            point_side[budget] = point
            posterior_view[budget] = view
            posterior_raw[budget] = posterior
        last_targets, last_mask, target_sha, mask_sha, valid_count = _valid_last_bin_authority(
            np, behavior=behavior, starts=starts,
        )
        def prefix_digest(budget: int) -> str:
            return _digest(_json({
                "budget": budget,
                "counts": _tensor_digest(torch, counts[:, :budget]),
                "exposure": _tensor_digest(torch, exposure[:budget]),
                "theta": _tensor_digest(torch, theta[:budget]),
                "prefix_row_ids": list(prefix_row_ids[:budget]),
            }))
        posterior_prefix = {str(budget): prefix_digest(budget) for budget in (30, 10, 4)}
        point_prefix = {
            str(budget): _digest(_json({"budget": budget, "raw_t4": raw_proofs[budget]["raw_t4_sha256"],
                                         "axis_proof": raw_proofs[budget]["body_sha256"]}))
            for budget in (30, 10, 4)
        }
        matched_prefix_rows: dict[str, str] = {}
        for budget in (30, 10, 4):
            # Both consumers are predeclared to use the same rewarded-trial
            # prefix.  The posterior adapter exposes the selected original
            # row IDs; the closure-bound OLS implementation independently
            # commits to ``list_datamodule_rewarded_trials(... )[:M]`` in its
            # raw-axis proof.  Bind the common rows explicitly so a future
            # short-prefix delta cannot silently compare different prefixes.
            if raw_proofs[budget].get("prefix_selection_semantics") != (
                "closure_bound_compute_unit_side_features_uncached_t4_uses_"
                "list_datamodule_rewarded_trials_then_exact_first_pool_size"
            ):
                raise PhysicalScoreError("OLS point prefix-selection semantic drift")
            matched_prefix_rows[str(budget)] = _digest(_json(list(prefix_row_ids[:budget])))
        if matched_prefix_rows["30"] != theta_recovery["prefix_rows_sha256"]:
            raise PhysicalScoreError("posterior/OLS M30 selected-prefix row digest drift")
        input_record = self._contract.SessionInput(
            surface=surface, session=session, n_windows=int(starts.size),
            neural_sha256=_array_digest(neural), calibration_m30_sha256=_array_digest(calibration),
            last_bin_target_sha256=target_sha, last_bin_valid_mask_sha256=mask_sha,
            last_bin_valid_count=valid_count,
            posterior_prefix_input_sha256s=posterior_prefix,
            ols_point_prefix_input_sha256s=point_prefix,
            matched_prefix_row_ids_sha256s=matched_prefix_rows,
            normalized_posterior_carrier_sha256s={
                str(budget): _tensor_digest(torch, posterior_view[budget].normalized_t4) for budget in (30, 10, 4)
            },
            normalized_ols_point_carrier_sha256s={
                str(budget): _tensor_digest(torch, point_side[budget]) for budget in (30, 10, 4)
            },
            target_theta_recovery_evidence=theta_recovery,
        )
        return _PhysicalSession(
            surface=surface, session=session, held=held, neural=neural, behavior=behavior,
            calibration_m30=calibration, starts=starts, point_side=point_side,
            posterior_view=posterior_view, posterior_raw=posterior_raw,
            last_targets=last_targets, last_valid_mask=last_mask, input_record=input_record,
            raw_axis_proofs=raw_proofs,
        )

    def resolve_inputs(self, *, identity: Any, flags: Any) -> Any:
        if self._closed or not self._models or self._full is None:
            raise PhysicalScoreError("physical backend must prepare strict models before resolving target inputs")
        records: list[Any] = []
        for surface, roster in (("within", tuple(identity.within_roster)), ("external", tuple(identity.external_roster))):
            rows = self._asset_rows(self._preflight, surface=surface)
            if tuple(row.get("session") for row in rows) != roster:
                raise PhysicalScoreError("physical input-asset roster/order differs from reviewed preflight")
            for row in rows:
                parsed = self._parse_one(asset=row, flags=flags)
                if parsed.session in self._sessions[surface]:
                    raise PhysicalScoreError("physical matched input pass duplicated a session")
                self._sessions[surface][parsed.session] = parsed
                records.append(parsed.input_record)
        if (
            tuple(self._sessions["within"]) != tuple(identity.within_roster)
            or tuple(self._sessions["external"]) != tuple(identity.external_roster)
        ):
            raise PhysicalScoreError("physical parsed session order differs from score identity")
        posterior_sha = self._preflight["normalizers"]["posterior_distribution"]["body_sha256"]
        behavior_sha = self._preflight["normalizers"]["behavior"]["semantic_sha256"]
        return self._contract.InputAuthority(
            records=tuple(records), source_posterior_normalizer_sha256=posterior_sha,
            behavior_normalizer_sha256=behavior_sha, shared_input_pass=True, cache_read_or_write=False,
        )

    def _session_batch(self, session: _PhysicalSession, starts: Sequence[int]) -> tuple[Any, Any, Any]:
        runtime = self._load_runtime()
        np, torch = runtime["np"], runtime["torch"]
        if not starts:
            raise PhysicalScoreError("physical score batch cannot be empty")
        neural = torch.from_numpy(np.stack([session.neural[start:start + 50] for start in starts])).to(
            runtime["device"], dtype=torch.float32,
        )
        behavior = torch.from_numpy(np.stack([session.behavior[start:start + 50] for start in starts])).to(
            runtime["device"], dtype=torch.float32,
        )
        calibration = torch.from_numpy(session.calibration_m30).to(runtime["device"], dtype=torch.float32)
        calibration = calibration.unsqueeze(0).expand(neural.shape[0], -1, -1, -1)
        if (
            tuple(neural.shape) != (len(starts), 50, session.neural.shape[1])
            or tuple(behavior.shape) != (len(starts), 50, 2)
            or tuple(calibration.shape) != (len(starts), 30, 100, session.neural.shape[1])
        ):
            raise PhysicalScoreError("physical matched score batch shape drift")
        return neural, behavior, calibration

    def _posterior_control(self, *, session: _PhysicalSession, mode: str) -> Any:
        """Build deterministic posterior diagnostic modes without a refit."""
        runtime = self._load_runtime()
        torch, core = runtime["torch"], runtime["core"]
        aligned = session.posterior_view[30]
        if mode == "posterior_mean_precision":
            return aligned
        if mode == "posterior_zero_carrier_m30_diagnostic":
            # Zero means no content and no unit-specific posterior confidence:
            # uniform credibility makes the centered attention-bias exactly 0.
            return core.PosteriorCarrierView(
                raw_beta=torch.zeros_like(aligned.raw_beta),
                raw_t4=torch.zeros_like(aligned.raw_t4),
                normalized_t4=torch.zeros_like(aligned.normalized_t4),
                credibility=torch.ones_like(aligned.credibility),
                zero_spike_mask=torch.zeros_like(aligned.zero_spike_mask),
                sampled=False, session_id=None, epoch=None,
                posterior_sha256=_digest((aligned.posterior_sha256 + "|zero-control").encode("ascii")),
                normalizer_authority_sha256=aligned.normalizer_authority_sha256,
            )
        if mode == "posterior_cyclic_wrong_pair_m30_diagnostic":
            unit_count = aligned.unit_count
            if unit_count < 2:
                raise PhysicalScoreError("cyclic wrong-pair diagnostic requires at least two units")
            permutation = torch.roll(torch.arange(unit_count, device=aligned.raw_t4.device, dtype=torch.long), shifts=1)
            return aligned.joint_permute(permutation)
        raise PhysicalScoreError("posterior carrier control mode drift")

    def _carrier_for_cell(self, *, cell: Any, session: _PhysicalSession) -> tuple[str, Any]:
        mode = cell.mode
        if mode == "sealed_cell_d_ols_point":
            return "sealed_cell_d", session.point_side[cell.budget]
        if mode == "posterior_mean_precision":
            return "posterior", session.posterior_view[cell.budget]
        if mode in {"posterior_zero_carrier_m30_diagnostic", "posterior_cyclic_wrong_pair_m30_diagnostic"}:
            if cell.budget != 30:
                raise PhysicalScoreError("posterior diagnostic can only use frozen M30 budget")
            return "posterior", self._posterior_control(session=session, mode=mode)
        raise PhysicalScoreError("physical score cell mode drift")

    def _forward(
        self,
        *,
        cell: Any,
        session: _PhysicalSession,
        neural: Any,
        calibration: Any,
    ) -> Any:
        runtime = self._load_runtime()
        torch, pop_robust = runtime["torch"], runtime["pop_robust"]
        system, carrier = self._carrier_for_cell(cell=cell, session=session)
        model = self._models.get(system)
        if model is None or model.training or not self._all_gradients_none(model):
            raise PhysicalScoreError("physical score forward model/eval/gradient drift")
        base = model if system == "sealed_cell_d" else model.cell_d
        calls = {"post_pool": 0}

        def hook(_module: Any, _inputs: Any, _output: Any) -> None:
            calls["post_pool"] += 1

        handle = base.id_encoder.post_pool.register_forward_hook(hook)
        try:
            with pop_robust.dynamic_dropout_recorder() as recorder:
                with torch.no_grad():
                    if torch.is_grad_enabled():
                        raise PhysicalScoreError("physical score forward unexpectedly has autograd enabled")
                    if system == "sealed_cell_d":
                        side = carrier.unsqueeze(0).expand(neural.shape[0], -1, -1).detach().clone()
                        output, identity = model(neural, calib_trials=calibration, side_features=side)
                    else:
                        output, identity = model(neural, calib_trials_m30=calibration, carrier=carrier)
        finally:
            handle.remove()
        if (
            calls["post_pool"] <= 0 or not torch.is_tensor(identity) or not bool(torch.isfinite(identity).all().item())
            or recorder["uniform_calls"] != 0 or recorder["dropout_calls"] != []
            or not torch.is_tensor(output) or tuple(output.shape) != (neural.shape[0], 50, 2)
            or not bool(torch.isfinite(output).all().item())
        ):
            raise PhysicalScoreError("physical B3S/eval-no-dropout/output proof drift")
        return output

    def _repeat_probe(self, *, cell: Any, session: _PhysicalSession) -> bool:
        starts = tuple(int(item) for item in session.starts[:self._BATCH_SIZE])
        neural, _behavior, calibration = self._session_batch(session, starts)
        first = self._forward(cell=cell, session=session, neural=neural, calibration=calibration)
        second = self._forward(cell=cell, session=session, neural=neural, calibration=calibration)
        if not self._load_runtime()["torch"].equal(first, second):
            raise PhysicalScoreError("physical repeated fixed-batch forward is not bitwise equal")
        return True

    def _assert_native_m30_parity(self, *, cell: Any, sessions: Sequence[Any]) -> None:
        if cell.mode != "sealed_cell_d_ols_point" or cell.budget != 30:
            return
        if self._sealed_rows is None:
            raise PhysicalScoreError("sealed governing table unavailable for native M30 parity")
        expected = self._sealed_rows[cell.surface]
        actual = tuple(
            {"session": row.session, "n_windows": row.n_windows, "r2": float(row.r2)} for row in sessions
        )
        if tuple(expected) != actual:
            raise PhysicalScoreError("sealed Cell-D OLS M30 live per-session last-bin parity failure")

    def score_cell(self, *, cell: Any, input_payload: Mapping[str, object], flags: Any) -> Any:
        if self._closed or set(self._models) != {"sealed_cell_d", "posterior"}:
            raise PhysicalScoreError("physical scoring requires two strict distinct model systems")
        runtime = self._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        if cell.surface not in self._sessions or not self._sessions[cell.surface]:
            raise PhysicalScoreError("physical score requested before shared surface input materialization")
        system = "sealed_cell_d" if cell.mode == "sealed_cell_d_ols_point" else "posterior"
        model = self._models[system]
        state_before = runtime["arm_common"].state_sha256(model)
        if state_before != self._state_at_load.get(system):
            raise PhysicalScoreError("physical score model state drifted before cell")
        roster = tuple(self._contract._strict_unique_roster(
            self._preflight["identity"]["rosters"][cell.surface],
            count=6 if cell.surface == "within" else 15,
            surface=cell.surface,
        ))
        # The slightly explicit identity roster recovery avoids sorting a date
        # ordered external table.  It must be byte-identical to input authority.
        records = {row["session"]: row for row in input_payload["records"] if row["surface"] == cell.surface}
        if tuple(records) != roster:
            raise PhysicalScoreError("physical score shared-input roster/order drift")
        first_session = self._sessions[cell.surface][roster[0]]
        repeated = self._repeat_probe(cell=cell, session=first_session)
        rows: list[Any] = []
        for session_name in roster:
            session = self._sessions[cell.surface][session_name]
            predictions: list[Any] = []
            targets: list[Any] = []
            digest = hashlib.sha256()
            starts = tuple(int(item) for item in session.starts)
            for offset in range(0, len(starts), self._BATCH_SIZE):
                chunk = starts[offset:offset + self._BATCH_SIZE]
                neural, behavior, calibration = self._session_batch(session, chunk)
                output = self._forward(cell=cell, session=session, neural=neural, calibration=calibration)
                valid = (behavior[:, -1, :] != -1.0).all(dim=-1)
                if not bool(valid.all().item()):
                    raise PhysicalScoreError("physical governing last-bin batch includes an invalid query")
                last_prediction = output[:, -1, :].detach().cpu().contiguous()
                last_target = behavior[:, -1, :].detach().cpu().contiguous()
                digest.update(last_prediction.numpy().tobytes())
                predictions.append(last_prediction)
                targets.append(last_target)
            if not predictions:
                raise PhysicalScoreError("physical score session emitted no governing prediction")
            prediction = torch.cat(predictions)
            target = torch.cat(targets)
            target_array = np.ascontiguousarray(target.numpy(), dtype=np.float32)
            mask_array = np.ascontiguousarray(np.all(target_array != -1.0, axis=1), dtype=np.uint8)
            if (
                target_array.shape != session.last_targets.shape
                or not np.array_equal(target_array, session.last_targets)
                or not np.array_equal(mask_array, session.last_valid_mask)
                or _array_digest(target_array) != session.input_record.last_bin_target_sha256
                or _array_digest(mask_array) != session.input_record.last_bin_valid_mask_sha256
                or int(mask_array.sum()) != session.input_record.last_bin_valid_count
            ):
                raise PhysicalScoreError("physical governing target/mask/count binding drift")
            r2 = runtime["metric"].session_r2(prediction, target)
            rows.append(self._contract.SessionScore(
                session=session_name, n_windows=len(starts), r2=float(r2),
                prediction_sha256=digest.hexdigest(),
                input_record_sha256=self._contract._digest(self._contract._json(records[session_name])),
            ))
        state_after = runtime["arm_common"].state_sha256(model)
        if state_after != state_before or model.training or not self._all_gradients_none(model):
            raise PhysicalScoreError("physical score modified model state/gradient/eval mode")
        self._assert_native_m30_parity(cell=cell, sessions=rows)
        credibility = {
            "sealed_cell_d_ols_point": "sealed_cell_d_ols_point_no_posterior_bias",
            "posterior_mean_precision": "posterior_precision_logit_bias",
            "posterior_zero_carrier_m30_diagnostic": "posterior_zero_carrier_m30_diagnostic",
            "posterior_cyclic_wrong_pair_m30_diagnostic": "posterior_cyclic_wrong_pair_m30_diagnostic",
        }[cell.mode]
        return self._contract.ModeEvidence(
            cell=cell,
            model_system="sealed_cell_d_checkpoint" if system == "sealed_cell_d" else "posterior_carrier_full_swa",
            model_swa_sha256=(self._preflight["identity"]["sealed_cell_d"]["swa_sha256"]
                              if system == "sealed_cell_d" else self._preflight["identity"]["full_training"]["swa_sha256"]),
            sessions=tuple(rows), input_authority_sha256=self._contract._digest(self._contract._json(input_payload)),
            model_state_before_sha256=state_before, model_state_after_sha256=state_after,
            eval_mode=True, dropout_disabled=True, gradients_none=True, finite_outputs=True,
            repeated_fixed_batch_bitwise_equal=repeated, b3s_m30_recomputed=True,
            posterior_mean_eval=(system == "posterior"), credibility_mode=credibility,
        )

    def reverify_after_forwards(self, *, identity: Any, flags: Any) -> Any:
        if self._closed:
            raise PhysicalScoreError("closed physical backend cannot reverify")
        runtime = self._load_runtime()
        for held in self._held_assets:
            held.reverify()
        for root in self._held_roots:
            root.reverify()
        for name, model in self._models.items():
            if (
                runtime["arm_common"].state_sha256(model) != self._state_at_load.get(name)
                or model.training or not self._all_gradients_none(model)
            ):
                raise PhysicalScoreError("physical post-forward model state/eval/gradient revalidation drift")
        if (
            flags.h1_opened or flags.formal_opened or flags.target_optimizer_steps != 0
            or flags.target_backward_calls != 0 or flags.target_update_calls != 0
        ):
            raise PhysicalScoreError("physical scorer crossed target-update/formal/H1 boundary")
        # This is descriptor-safe source closure recomputation, not a cache or
        # target asset operation.  The core requires exact launch/final bytes.
        return self._contract.implementation_closure(self._root)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for asset in reversed(self._held_assets):
            asset.close()
        self._held_assets.clear()
        for root in reversed(self._held_roots):
            root.close()
        self._held_roots.clear()
        self._sessions = {"within": {}, "external": {}}
        self._models.clear()
