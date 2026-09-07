"""Route-owned, descriptor-safe native M1 source reader for CS-WG.

This module deliberately imports neither Torch nor the historical top-level
``src`` runtime at import time.  The native parser is loaded only from an
authorized physical reader call, after the source lifecycle has published its
durable attempt.  The CS-WG route itself is always imported through
``tfpd_exploration.src.cross_session_worst_group_v1``; the unmodified
``streaming_calibration_exp`` tree retains its historical top-level ``src``
package without a ``sys.modules`` replacement or a parser copy.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import stat
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from . import core, plan
from . import source_lifecycle as lifecycle
from . import source_physical as physical


class SourceReaderError(RuntimeError):
    """Fail closed for route-owned source reader, parser, or descriptor drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceReaderError(message)


def _json_bytes(value: object) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"CS-WG {label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: str) -> tuple[str, ...]:
    path = Path(value)
    _require(isinstance(value, str) and value and not path.is_absolute()
             and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG source reader path is unsafe")
    return tuple(path.parts)


@dataclass(frozen=True)
class NativeM1ReaderRecipe:
    """The immutable direct ``prepare_session_data``/``FalconDataset`` recipe."""

    task: str = "m1"
    window_size: int = plan.M1_WINDOW_SIZE
    calibration_n_trials: int = plan.M1_CALIBRATION_TRIALS
    random_calibration: bool = False
    smooth_calibration: bool = False
    max_trial_length: int = plan.M1_B3S_MAX_TRIAL_LENGTH
    standardize_covariates: bool = False
    use_intertrials: bool = True
    use_calib_intertrials: bool = False
    trial_feature_type: str = "raw"
    interpolate_trials: bool = True
    interpolate_trials_kind: str = "cubic"
    pad_value: float = -1.0
    side_feature_group: str = "none"
    query_start_trial: int = 0

    def __post_init__(self) -> None:
        _require(
            self.task == "m1"
            and self.window_size == plan.M1_WINDOW_SIZE
            and self.calibration_n_trials == plan.M1_CALIBRATION_TRIALS
            and self.random_calibration is False
            and self.smooth_calibration is False
            and self.max_trial_length == plan.M1_B3S_MAX_TRIAL_LENGTH
            and self.standardize_covariates is False
            and self.use_intertrials is True
            and self.use_calib_intertrials is False
            and self.trial_feature_type == "raw"
            and self.interpolate_trials is True
            and self.interpolate_trials_kind == "cubic"
            and self.pad_value == -1.0
            and self.side_feature_group == "none"
            and self.query_start_trial == 0,
            "CS-WG native M1 source reader recipe drift",
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_native_reader_recipe_v1",
            "task": self.task,
            "window_size": self.window_size,
            "calibration_n_trials": self.calibration_n_trials,
            "random_calibration": self.random_calibration,
            "smooth_calibration": self.smooth_calibration,
            "max_trial_length": self.max_trial_length,
            "standardize_covariates": self.standardize_covariates,
            "use_intertrials": self.use_intertrials,
            "use_calib_intertrials": self.use_calib_intertrials,
            "trial_feature_type": self.trial_feature_type,
            "interpolate_trials": self.interpolate_trials,
            "interpolate_trials_kind": self.interpolate_trials_kind,
            "pad_value": self.pad_value,
            "side_feature_group": self.side_feature_group,
            "query_start_trial": self.query_start_trial,
            "calibration_shape_per_row": list(plan.M1_CALIBRATION_SHAPE_PER_ROW),
            "direct_parser": "FalconDataModule.prepare_session_data",
            "direct_dataset": "FalconDataset",
            "ordinary_setup_called": False,
            "target_minival_heldout_formal_evalai_absent": True,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))

    def datamodule_kwargs(self, *, data_dir: Path) -> dict[str, object]:
        """Exact public constructor literals; ``setup`` is never called."""
        return {
            "task": self.task,
            "data_dir": str(data_dir),
            "heldin_session_names": [],
            "batch_size": plan.TOTAL_BATCH_SIZE,
            "window_size": self.window_size,
            "calibration_n_trials": self.calibration_n_trials,
            "random_calibration": self.random_calibration,
            "smooth_calibration": self.smooth_calibration,
            "max_trial_length": self.max_trial_length,
            "standardize_covariates": self.standardize_covariates,
            "use_intertrials": self.use_intertrials,
            "use_calib_intertrials": self.use_calib_intertrials,
            "trial_feature_type": self.trial_feature_type,
            "interpolate_trials": self.interpolate_trials,
            "interpolate_trials_kind": self.interpolate_trials_kind,
            "pad_value": self.pad_value,
            "num_workers": 0,
            "pin_memory": False,
            "validation_protocol": "loso",
            "loso_fold": 0,
            "include_heldout_in_fit": False,
            "include_heldout_in_test": False,
            "query_start_trial": self.query_start_trial,
            "heldin_query_start_trial": 0,
            "heldin_query_end_trial": None,
            "allow_empty_heldout_query": False,
            "sampler_seed": 42,
            "balance_session_batches": False,
            "reshuffle_train_sampler_each_epoch": False,
            "side_feature_group": self.side_feature_group,
            "side_feature_shuffle_seed": 0,
        }

    def dataset_kwargs(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "split": "train",
            "calibration_n_trials": self.calibration_n_trials,
            "random_calibration": self.random_calibration,
            "smooth_calibration": self.smooth_calibration,
            "max_trial_length": self.max_trial_length,
            "use_calib_intertrials": self.use_calib_intertrials,
            "trial_feature_type": self.trial_feature_type,
            "remove_still_times": False,
            "remove_calib_still_times": False,
            "use_calib_active_segments": False,
            "calib_n_active_segments": 1,
            "interpolate_trials": self.interpolate_trials,
            "interpolate_trials_kind": self.interpolate_trials_kind,
            "pad_value": self.pad_value,
            "side_feature_group": self.side_feature_group,
            "side_feature_shuffle_seed": 0,
            "query_start_trial": self.query_start_trial,
        }


NATIVE_M1_READER_RECIPE = NativeM1ReaderRecipe()


@dataclass(frozen=True)
class NativeM1Runtime:
    """Closure-bound native parser exports, loaded only at physical prepare."""

    falcon_datamodule_type: type
    falcon_dataset_type: type
    task: object
    external_versions: Mapping[str, str]
    parser_module_path: str

    def __post_init__(self) -> None:
        versions = dict(self.external_versions)
        _require(isinstance(self.falcon_datamodule_type, type)
                 and isinstance(self.falcon_dataset_type, type)
                 and isinstance(self.parser_module_path, str) and self.parser_module_path
                 and tuple(sorted(versions)) == physical._REQUIRED_EXTERNAL_VERSION_NAMES
                 and all(isinstance(item, str) and item for item in versions.values()),
                 "CS-WG native M1 runtime export/version topology drift")
        object.__setattr__(self, "external_versions", MappingProxyType(versions))


def _regular_no_follow_sha(root: Path, relative: str) -> str:
    path = Path(root).absolute() / Path(relative)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise SourceReaderError(f"CS-WG native parser closure leaf inaccessible: {relative}") from error
    _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
             f"CS-WG native parser closure leaf not regular: {relative}")
    with path.open("rb") as handle:
        return _sha(handle.read())


def _path_is_within(path: Path, base: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(base.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _distribution_version(module: object, *candidate_distributions: str) -> str:
    for candidate in candidate_distributions:
        try:
            value = importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
        if value:
            return str(value)
    value = getattr(module, "__version__", None)
    _require(isinstance(value, str) and value,
             f"CS-WG native parser cannot attest installed external package {candidate_distributions}")
    return value


def load_native_m1_runtime(code_root: Path) -> NativeM1Runtime:
    """Load the historical top-level ``src`` parser without replacing modules.

    The caller must have imported this route as ``tfpd_exploration.src...``.
    If a different top-level ``src`` is already resident, we fail rather than
    deleting/replacing it.  When absent, the streaming tree is inserted only
    long enough to perform its normal import and remains the owner of its
    historical ``src`` entries afterwards.
    """
    root = Path(code_root).absolute()
    closure = lifecycle.implementation_closure(root)
    closure_rows = {str(item["path"]): str(item["sha256"]) for item in closure["paths"]}
    required = (
        "streaming_calibration_exp/src/__init__.py",
        *plan.M1_SOURCE_DECODER_RUNTIME_IMPORT_RELATIVES,
    )
    for relative in required:
        _require(relative in closure_rows and _regular_no_follow_sha(root, relative) == closure_rows[relative],
                 f"CS-WG native parser closure binding drift: {relative}")
    streaming_root = root / "streaming_calibration_exp"
    historical_src_root = streaming_root / "src"
    existing = sys.modules.get("src")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        _require(isinstance(existing_file, str)
                 and _path_is_within(Path(existing_file), historical_src_root),
                 "CS-WG native parser refuses to replace an existing non-streaming top-level src")
        module = importlib.import_module("src.data.falcon_datamodule")
    else:
        insertion = str(streaming_root)
        sys.path.insert(0, insertion)
        try:
            module = importlib.import_module("src.data.falcon_datamodule")
        finally:
            try:
                sys.path.remove(insertion)
            except ValueError:
                pass
    module_file = getattr(module, "__file__", None)
    _require(isinstance(module_file, str) and _path_is_within(Path(module_file), historical_src_root),
             "CS-WG native Falcon parser module path/namespace drift")
    falcon_datamodule_type = getattr(module, "FalconDataModule", None)
    falcon_dataset_type = getattr(module, "FalconDataset", None)
    _require(isinstance(falcon_datamodule_type, type) and isinstance(falcon_dataset_type, type)
             and callable(getattr(falcon_datamodule_type, "prepare_session_data", None)),
             "CS-WG native Falcon parser exports drift")
    config_module = importlib.import_module("falcon_challenge.config")
    FalconConfig = getattr(config_module, "FalconConfig", None)
    FalconTask = getattr(config_module, "FalconTask", None)
    _require(callable(FalconConfig) and FalconTask is not None and hasattr(FalconTask, "m1"),
             "CS-WG native Falcon task export drift")
    task = FalconConfig(task=FalconTask.m1).task
    import lightning
    import scipy
    import torch

    versions = {
        "falcon_challenge": _distribution_version(config_module, "falcon-challenge", "falcon_challenge"),
        "lightning": _distribution_version(lightning, "lightning"),
        "numpy": _distribution_version(np, "numpy"),
        "scipy": _distribution_version(scipy, "scipy"),
        "torch": _distribution_version(torch, "torch"),
    }
    return NativeM1Runtime(
        falcon_datamodule_type=falcon_datamodule_type,
        falcon_dataset_type=falcon_dataset_type,
        task=task,
        external_versions=versions,
        parser_module_path=str(Path(module_file).resolve(strict=True)),
    )


@dataclass
class HeldSourceFile:
    """One held no-follow source body with immutable named identity evidence."""

    source_root: Path
    descriptor: physical.SourceFileDescriptor
    root_fd: int
    directory_fds: tuple[int, ...]
    file_fd: int
    directory_identities: tuple[tuple[str, int, int], ...]
    file_identity: tuple[int, int]
    before: Mapping[str, object]
    _closed: bool = field(default=False, repr=False)

    @staticmethod
    def _open_dir(path: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            return os.open(path, flags)
        except OSError as error:
            raise SourceReaderError(f"CS-WG source root cannot be opened no-follow: {path}") from error

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, int]:
        return int(info.st_dev), int(info.st_ino)

    @classmethod
    def open(cls, source_root: Path, descriptor: physical.SourceFileDescriptor) -> "HeldSourceFile":
        _require(isinstance(descriptor, physical.SourceFileDescriptor),
                 "CS-WG held source needs a typed descriptor")
        root = Path(source_root).absolute()
        try:
            root_info = os.lstat(root)
        except OSError as error:
            raise SourceReaderError("CS-WG source root is inaccessible") from error
        _require(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
                 "CS-WG source root is not a regular non-symlink directory")
        root_fd = cls._open_dir(root)
        directory_fds: list[int] = []
        file_fd: int | None = None
        identities: list[tuple[str, int, int]] = [(".", *cls._identity(root_info))]
        current_fd = root_fd
        parts = _safe_relative(descriptor.relative_path)
        try:
            # ``lstat`` before ``open`` is not sufficient: a directory can be
            # exchanged in the tiny interval between those operations.  Bind
            # the descriptor actually opened by this process to the named
            # identity that was just approved before descending into it.
            opened_root = os.fstat(root_fd)
            _require(cls._identity(opened_root) == cls._identity(root_info)
                     and stat.S_ISDIR(opened_root.st_mode)
                     and not stat.S_ISLNK(opened_root.st_mode),
                     "CS-WG source root changed between lstat/open")
            for position, name in enumerate(parts[:-1]):
                info = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                         "CS-WG source descriptor parent is not no-follow directory")
                child_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                   | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
                directory_fds.append(child_fd)
                opened_child = os.fstat(child_fd)
                _require(cls._identity(opened_child) == cls._identity(info)
                         and stat.S_ISDIR(opened_child.st_mode)
                         and not stat.S_ISLNK(opened_child.st_mode),
                         "CS-WG source descriptor parent changed between stat/open")
                current_fd = child_fd
                identities.append(("/".join(parts[: position + 1]), *cls._identity(info)))
            leaf = parts[-1]
            file_fd = os.open(leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
            info = os.fstat(file_fd)
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and int(info.st_size) == descriptor.byte_count,
                     "CS-WG source body type/hardlink/size drift")
            result = cls(root, descriptor, root_fd, tuple(directory_fds), file_fd, tuple(identities),
                         cls._identity(info), MappingProxyType({}))
            initial = result._held_identity()
            _require(initial["body_sha256"] == descriptor.sha256,
                     "CS-WG source body SHA drift before native parse")
            object.__setattr__(result, "before", MappingProxyType(initial))
            return result
        except OSError as error:
            if file_fd is not None:
                os.close(file_fd)
            for handle in reversed(directory_fds):
                os.close(handle)
            os.close(root_fd)
            raise SourceReaderError("CS-WG held source descriptor no-follow/open drift") from error
        except BaseException:
            if file_fd is not None:
                os.close(file_fd)
            for handle in reversed(directory_fds):
                os.close(handle)
            os.close(root_fd)
            raise

    @property
    def parser_path(self) -> Path:
        _require(not self._closed, "CS-WG held source descriptor is closed")
        return Path("/proc/self/fd") / str(self.file_fd)

    def _read_digest(self) -> str:
        os.lseek(self.file_fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        try:
            while True:
                block = os.read(self.file_fd, 1 << 20)
                if not block:
                    break
                digest.update(block)
        finally:
            os.lseek(self.file_fd, 0, os.SEEK_SET)
        return digest.hexdigest()

    def _held_identity(self) -> dict[str, object]:
        info = os.fstat(self.file_fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                 and self._identity(info) == self.file_identity
                 and int(info.st_size) == self.descriptor.byte_count,
                 "CS-WG held source descriptor identity drift")
        digest = self._read_digest()
        _require(digest == self.descriptor.sha256,
                 "CS-WG held source body SHA drift")
        return physical.held_source_identity_payload(
            self.descriptor,
            device=int(info.st_dev),
            inode=int(info.st_ino),
            byte_count=int(info.st_size),
            body_sha256=digest,
            mode=stat.S_IMODE(info.st_mode),
            hard_link_count=int(info.st_nlink),
        )

    def _reopen_named_identities(self) -> tuple[tuple[str, int, int], ...]:
        root_fd = self._open_dir(self.source_root)
        handles: list[int] = []
        current_fd = root_fd
        result: list[tuple[str, int, int]] = []
        try:
            root_info = os.fstat(root_fd)
            result.append((".", *self._identity(root_info)))
            parts = _safe_relative(self.descriptor.relative_path)
            for position, name in enumerate(parts[:-1]):
                info = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                         "CS-WG named source parent type/symlink drift")
                child = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
                handles.append(child)
                opened_child = os.fstat(child)
                _require(self._identity(opened_child) == self._identity(info)
                         and stat.S_ISDIR(opened_child.st_mode)
                         and not stat.S_ISLNK(opened_child.st_mode),
                         "CS-WG named source parent changed between stat/open")
                current_fd = child
                result.append(("/".join(parts[: position + 1]), *self._identity(info)))
            leaf_info = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
            _require(stat.S_ISREG(leaf_info.st_mode) and leaf_info.st_nlink == 1
                     and self._identity(leaf_info) == self.file_identity
                     and int(leaf_info.st_size) == self.descriptor.byte_count,
                     "CS-WG named source body identity/hardlink/size drift")
            return tuple(result)
        except OSError as error:
            raise SourceReaderError("CS-WG named source descriptor no-follow/reopen drift") from error
        finally:
            for handle in reversed(handles):
                os.close(handle)
            os.close(root_fd)

    def revalidate_after_parse(self) -> dict[str, object]:
        _require(not self._closed, "CS-WG held source descriptor is closed")
        named = self._reopen_named_identities()
        _require(named == self.directory_identities,
                 "CS-WG source root/parent named identity drift after parser")
        after = self._held_identity()
        _require(after == dict(self.before), "CS-WG source body identity/SHA drift after parser")
        return after

    def close(self) -> None:
        if not self._closed:
            os.close(self.file_fd)
            for handle in reversed(self.directory_fds):
                os.close(handle)
            os.close(self.root_fd)
            self._closed = True

    def __enter__(self) -> "HeldSourceFile":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


@dataclass
class DeferredHeldM1DescriptorLoader:
    """Resolve only manifest-selected source descriptor sizes after attempt.

    The sealed metadata manifest is the complete *pre-attempt* authority for
    session order, relative path, and body SHA.  It intentionally omits byte
    counts, so this loader performs only a held no-follow descriptor walk when
    ``StrictM1SourceProvider.prepare`` asks for the selected source sessions.
    It neither opens nor hashes a source body; :class:`HeldSourceFile` does
    the body-SHA verification immediately afterwards while holding the file.

    Construction is deliberately inert: ``source_root`` is retained as a
    lexical path and is not resolved, statted, or opened here.  This is the
    precise attempt-before-source seam used by the route-owned factories.
    """

    source_root: Path
    metadata_authority: Mapping[str, object]
    resolution_events: list[tuple[str, ...]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        authority = dict(self.metadata_authority) if isinstance(self.metadata_authority, Mapping) else {}
        expected = lifecycle.m1_metadata_manifest_binding_payload()
        expected_rows = [
            {"session_id": session, "relative_path": relative, "sha256": digest}
            for session, relative, digest in lifecycle.M1_METADATA_SOURCE_ROWS
        ]
        _require(isinstance(self.source_root, Path)
                 and all(authority.get(key) == value for key, value in expected.items())
                 and authority.get("source_rows") == expected_rows
                 and authority.get("metadata_only") is True,
                 "CS-WG deferred held descriptor loader metadata authority drift")
        object.__setattr__(self, "metadata_authority", MappingProxyType(authority))

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, int]:
        return int(info.st_dev), int(info.st_ino)

    @staticmethod
    def _open_directory_at(parent_fd: int, name: str, expected: os.stat_result) -> int:
        try:
            child_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                               | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        except OSError as error:
            raise SourceReaderError("CS-WG deferred source descriptor parent open drift") from error
        try:
            observed = os.fstat(child_fd)
            _require(DeferredHeldM1DescriptorLoader._identity(observed)
                     == DeferredHeldM1DescriptorLoader._identity(expected)
                     and stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
                     "CS-WG deferred source descriptor parent changed between stat/open")
            return child_fd
        except BaseException:
            os.close(child_fd)
            raise

    def _selected_descriptor(self, *, session: str, relative_path: str, digest: str) -> physical.SourceFileDescriptor:
        """Obtain an exact current byte count through a no-follow FD chain."""
        root = Path(self.source_root).absolute()
        try:
            root_info = os.lstat(root)
        except OSError as error:
            raise SourceReaderError("CS-WG deferred source root is inaccessible") from error
        _require(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
                 "CS-WG deferred source root is not a regular non-symlink directory")
        root_fd = HeldSourceFile._open_dir(root)
        opened_dirs: list[int] = []
        leaf_fd: int | None = None
        try:
            opened_root = os.fstat(root_fd)
            _require(self._identity(opened_root) == self._identity(root_info)
                     and stat.S_ISDIR(opened_root.st_mode) and not stat.S_ISLNK(opened_root.st_mode),
                     "CS-WG deferred source root changed between lstat/open")
            current_fd = root_fd
            parts = _safe_relative(relative_path)
            for name in parts[:-1]:
                before = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                _require(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
                         "CS-WG deferred source descriptor parent is not no-follow directory")
                child = self._open_directory_at(current_fd, name, before)
                opened_dirs.append(child)
                current_fd = child
            leaf_fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
            leaf = os.fstat(leaf_fd)
            _require(stat.S_ISREG(leaf.st_mode) and not stat.S_ISLNK(leaf.st_mode)
                     and leaf.st_nlink == 1 and int(leaf.st_size) > 0,
                     "CS-WG deferred source body type/hardlink/size drift")
            return physical.SourceFileDescriptor(
                session_id=session,
                relative_path=relative_path,
                sha256=digest,
                byte_count=int(leaf.st_size),
            )
        except OSError as error:
            raise SourceReaderError("CS-WG deferred source descriptor no-follow/stat drift") from error
        finally:
            if leaf_fd is not None:
                os.close(leaf_fd)
            for descriptor in reversed(opened_dirs):
                os.close(descriptor)
            os.close(root_fd)

    def __call__(self, sessions: tuple[str, ...]) -> tuple[physical.SourceFileDescriptor, ...]:
        _require(isinstance(sessions, tuple) and len(sessions) == 3
                 and tuple(dict.fromkeys(sessions)) == sessions
                 and all(session in plan.HELD_IN_SOURCE_SESSIONS for session in sessions),
                 "CS-WG deferred held descriptor selection topology drift")
        rows = {
            str(row["session_id"]): row
            for row in self.metadata_authority["source_rows"]  # type: ignore[index]
            if isinstance(row, Mapping)
        }
        _require(set(rows) == set(plan.HELD_IN_SOURCE_SESSIONS),
                 "CS-WG deferred held descriptor metadata row topology drift")
        result = tuple(
            self._selected_descriptor(
                session=session,
                relative_path=str(rows[session]["relative_path"]),
                digest=str(rows[session]["sha256"]),
            )
            for session in sessions
        )
        self.resolution_events.append(sessions)
        return result


def build_sealed_metadata_descriptor_provider(
    *, source_root: Path, metadata_authority: Mapping[str, object],
) -> physical.SealedMetadataBoundSourceDescriptorProvider:
    """Construct the route-owned deferred provider without touching source.

    This is intentionally the only live-factory construction path.  The
    caller supplies a lexical source-root capability and the descriptor-read
    sealed metadata authority; the concrete loader delays all source path
    resolution until ``prepare_source``.
    """
    loader = DeferredHeldM1DescriptorLoader(Path(source_root), metadata_authority)
    return physical.SealedMetadataBoundSourceDescriptorProvider(loader, metadata_authority)


RuntimeLoader = Callable[[Path], NativeM1Runtime]


def _array_digest_chain(*arrays: np.ndarray) -> str:
    return _sha(_json_bytes([core.array_digest(value) for value in arrays]))


@dataclass
class RouteOwnedM1SourceReader:
    """Direct all-four-fold M1 parser; never calls shared ``setup`` or globbing."""

    code_root: Path
    source_root: Path
    runtime_loader: RuntimeLoader = load_native_m1_runtime
    recipe: NativeM1ReaderRecipe = NATIVE_M1_READER_RECIPE
    read_events: list[str] = field(default_factory=list, init=False)
    source_opened: bool = field(default=False, init=False)
    _runtime: NativeM1Runtime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _require(isinstance(self.code_root, Path) and isinstance(self.source_root, Path)
                 and callable(self.runtime_loader) and isinstance(self.recipe, NativeM1ReaderRecipe),
                 "CS-WG route-owned native reader construction drift")

    def _runtime_once(self) -> NativeM1Runtime:
        if self._runtime is None:
            runtime = self.runtime_loader(Path(self.code_root))
            _require(isinstance(runtime, NativeM1Runtime), "CS-WG native reader runtime loader type drift")
            self._runtime = runtime
        return self._runtime

    def _native_dataset(self, runtime: NativeM1Runtime, held: HeldSourceFile) -> tuple[Any, Mapping[str, object]]:
        descriptor = held.descriptor
        # The direct primitive is deliberately constructed with the fixed public
        # constructor, then only ``prepare_session_data`` is invoked.  Calling
        # ``setup`` would discover validation/minival surfaces and is forbidden.
        datamodule = runtime.falcon_datamodule_type(**self.recipe.datamodule_kwargs(data_dir=self.source_root))
        _require(callable(getattr(datamodule, "prepare_session_data", None))
                 and callable(getattr(datamodule, "setup", None)),
                 "CS-WG native reader FalconDataModule interface drift")
        record = datamodule.prepare_session_data(
            held.parser_path,
            runtime.task,
            standardize_covariates=self.recipe.standardize_covariates,
            covariates_mean=None,
            covariates_std=None,
            use_intertrials=self.recipe.use_intertrials,
            include_trial_targets=False,
            include_trial_obj_ids=False,
        )
        _require(isinstance(record, Mapping)
                 and set(("neural", "covariates", "trial_change", "eval_mask")).issubset(record)
                 and "trial_target_angles" not in record and "trial_obj_ids" not in record,
                 "CS-WG native reader parser record/source-only boundary drift")
        sessions = OrderedDict(((descriptor.session_id, dict(record)),))
        dataset = runtime.falcon_dataset_type(
            sessions_dict=sessions,
            calib_sessions_dict=sessions,
            **self.recipe.dataset_kwargs(),
        )
        _require(hasattr(dataset, "window_indices") and hasattr(dataset, "__getitem__"),
                 "CS-WG native reader dataset/setup boundary drift")
        return dataset, record

    def read_source_session(self, descriptor: physical.SourceFileDescriptor) -> physical.SourceSessionMaterial:
        _require(isinstance(descriptor, physical.SourceFileDescriptor),
                 "CS-WG route-owned reader requires typed source descriptor")
        runtime = self._runtime_once()
        self.source_opened = True
        self.read_events.append(f"open:{descriptor.session_id}")
        with HeldSourceFile.open(self.source_root, descriptor) as held:
            dataset, _record = self._native_dataset(runtime, held)
            _require(len(dataset) > 0, "CS-WG native reader returned no valid source query windows")
            calibration: np.ndarray | None = None
            calibration_sha256: str | None = None
            rows: dict[int, physical.UnassignedSourceM1Row] = {}
            targets: list[np.ndarray] = []
            starts: list[int] = []
            endpoint_valid: list[bool] = []
            for sample_index in range(len(dataset)):
                item = dataset[sample_index]
                _require(isinstance(item, tuple) and len(item) == 4,
                         "CS-WG native FalconDataset output/source-only interface drift")
                x_raw, covariates_raw, calibration_raw, session_name = item
                _require(session_name == descriptor.session_id,
                         "CS-WG native FalconDataset returned a different session")
                x = np.asarray(x_raw)
                covariates = np.asarray(covariates_raw)
                current_calibration = np.asarray(calibration_raw)
                _require(x.dtype == np.dtype(np.float32)
                         and covariates.dtype == np.dtype(np.float32)
                         and current_calibration.dtype == np.dtype(np.float32)
                         and x.shape == (plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT)
                         and covariates.shape == (plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS)
                         and current_calibration.shape == plan.M1_CALIBRATION_SHAPE_PER_ROW
                         and np.isfinite(x).all() and np.isfinite(covariates).all()
                         and np.isfinite(current_calibration).all(),
                         "CS-WG native FalconDataset row shape/dtype drift")
                current_calibration = np.ascontiguousarray(current_calibration)
                if calibration is None:
                    # The native dataset returns the same deterministic M10
                    # tensor for every query row in a session.  Retain one
                    # immutable C-contiguous backing allocation and hand each
                    # typed row a read-only header/view of it.  The mixed B32
                    # route later performs its explicit per-row stack; this
                    # avoids a catastrophic O(query_rows * 2.5 MiB) reader
                    # allocation without introducing batch broadcasting.
                    calibration = np.array(current_calibration, dtype=np.float32, copy=True, order="C")
                    calibration.setflags(write=False)
                    calibration_sha256 = core.array_digest(calibration)
                _require(calibration is not None and np.array_equal(current_calibration, calibration),
                         "CS-WG native reader calibration changed within one source session")
                window_index = dataset.window_indices[sample_index]
                _require(isinstance(window_index, tuple) and len(window_index) == 2
                         and window_index[0] == descriptor.session_id and type(window_index[1]) is int,
                         "CS-WG native FalconDataset query identity drift")
                start = int(window_index[1])
                target = np.ascontiguousarray(covariates[-1])
                endpoint_mask = True
                if hasattr(dataset, "eval_mask"):
                    mask = np.asarray(dataset.eval_mask[descriptor.session_id], dtype=np.bool_)
                    endpoint = start + self.recipe.window_size - 1
                    _require(0 <= endpoint < mask.shape[0], "CS-WG native final-bin eval-mask index drift")
                    endpoint_mask = bool(mask[endpoint])
                _require(endpoint_mask, "CS-WG native FalconDataset emitted an invalid final-bin window")
                _require(calibration_sha256 is not None,
                         "CS-WG native reader session calibration digest is absent")
                row_calibration = calibration.view()
                row_calibration.setflags(write=False)
                _require(np.shares_memory(row_calibration, calibration)
                         and row_calibration.flags.c_contiguous and not row_calibration.flags.writeable,
                         "CS-WG native reader lost the immutable session calibration backing")
                rows[sample_index] = physical.UnassignedSourceM1Row(
                    session_id=descriptor.session_id,
                    sample_index=sample_index,
                    sample_id=f"{descriptor.session_id}:native_window_start:{start}",
                    calibration_session=descriptor.session_id,
                    calibration_sha256=calibration_sha256,
                    model_inputs={
                        "x": np.ascontiguousarray(x),
                        "calib_trialized_neural_features": row_calibration,
                    },
                    raw_final_target=target,
                )
                targets.append(target)
                starts.append(start)
                endpoint_valid.append(endpoint_mask)
            _require(calibration is not None and calibration_sha256 is not None,
                     "CS-WG native reader calibration is absent")
            target_array = np.ascontiguousarray(np.stack(targets, axis=0), dtype=np.float32)
            valid_array = np.ascontiguousarray(np.asarray(endpoint_valid, dtype=np.bool_))
            labels = core.SourceOnlyFinalBinLabels(descriptor.session_id, target_array, valid_array)
            after = held.revalidate_after_parse()
            start_array = np.ascontiguousarray(np.asarray(starts, dtype=np.int64))
            target_mask_digest = _array_digest_chain(target_array, valid_array)
            query_digest = _sha(_json_bytes({
                "window_starts": core.array_digest(start_array),
                "targets_and_eval_mask": target_mask_digest,
                "label_digest": labels.digest,
            }))
            evidence = physical.native_session_evidence(
                descriptor,
                calibration_session=descriptor.session_id,
                calibration_sha256=calibration_sha256,
                row_count=len(rows),
                ordered_query_identity_sha256=query_digest,
                ordered_window_start_sha256=core.array_digest(start_array),
                ordered_target_evalmask_sha256=target_mask_digest,
                held_before=held.before,
                held_after=after,
                reader_recipe_sha256=self.recipe.sha256,
                external_versions=runtime.external_versions,
            )
        self.read_events.append(f"parsed:{descriptor.session_id}")
        return physical.SourceSessionMaterial(
            descriptor=descriptor,
            labels=labels,
            rows_by_sample_index=rows,
            valid_source_windows=len(rows),
            calibration_session=descriptor.session_id,
            calibration_sha256=calibration_sha256,
            calibration_backing=calibration,
            native_evidence=evidence,
        )


def route_owned_reader_factory(
    *, code_root: Path, source_root: Path,
    runtime_loader: RuntimeLoader = load_native_m1_runtime,
) -> RouteOwnedM1SourceReader:
    """Deferred factory used by the reviewed physical backend only."""
    return RouteOwnedM1SourceReader(
        code_root=Path(code_root),
        source_root=Path(source_root),
        runtime_loader=runtime_loader,
    )
