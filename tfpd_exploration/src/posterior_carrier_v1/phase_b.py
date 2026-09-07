"""Phase-B source-only lifecycle for the posterior-carrier Cell-D system.

This module intentionally has no import-time data, cache, checkpoint, CUDA,
remote-host, or output-root action.  It contains the typed authority and
training-side cache contracts used by a later reviewed remote smoke.  The
public CLI imports only :mod:`posterior_carrier_v1.plan`; this module is loaded
only by an in-process, root-reviewed execution capability or CPU synthetic
tests.

The normalizer policy below is a Phase-B amendment to the handoff: posterior
means at M4/M10/M30 get their own strict-source float64 normalizer.  Reusing
ordinary point-T4 moments is deliberately impossible through this surface.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch

from . import core
from .plan import BUDGETS, CELL, HANDOFF_RELATIVE, HANDOFF_SHA256


PHASE_B = "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_PHASE_B_SOURCE_SMOKE_V1"
PHASE_B_NORMALIZER_AMENDMENT = (
    "posterior-specific strict-source float64 M4/M10/M30 normalizer; "
    "this supersedes ordinary point-T4 normalization for the headline system"
)
SOURCE_SMOKE_STEPS = 100
SOURCE_BATCH_SIZE = 32
SOURCE_EPOCHS = 48
SOURCE_STEPS_PER_EPOCH = 33_925
SOURCE_TOTAL_STEPS = SOURCE_EPOCHS * SOURCE_STEPS_PER_EPOCH
SOURCE_SESSION_COUNT = 27
SOURCE_SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_source_smoke_v1"

# A reviewed remote stage contains code plus a small immutable authority bundle;
# it must never contain, symlink to, or copy the 9.1-GB source NWBs.  The
# source files remain at this separately bound canonical external root.
CANONICAL_SOURCE_DATA_ROOT = "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C"
STRICT27_MANIFEST_RELATIVE = "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
SOURCE_AUTHORITY_ASSET_SPECS: tuple[tuple[str, str, int | None], ...] = (
    ("tfpd_exploration/results/admission_arms_v1/preflight_armA.json",
     "2632c6a6a4cfb8a4c0fb2b23e0cc8205ea323240b376903c60d5b27110f59e43", 0o444),
    ("tfpd_exploration/results/sparsification_theta_authority_v1/theta_authority_receipt.json",
     "d023dd632c4717443f1f55e924a09be1747fc58d5c30a8fb6fa38f4b7b117184", 0o444),
    ("tfpd_exploration/results/sparsification_theta_authority_v1/theta_authority.pt",
     "cef39dc8220aa253214963a32e5457dede1045e64b158e37fc267a6fb4146319", 0o444),
    ("sua_exploration/results/misleading_identity_swap_v2_source_authority_dev/strict27_m30_source_lineage_v3.json",
     "7375a37c8c5e59cb7e6786e0c67e930f918d59ff4dc769742391ad6dc97579fa", 0o444),
    # The strict manifest is versioned/closure-bound but historically mode
    # 0664 and has no sidecar, so live SHA revalidation—not a fictitious
    # immutable-file mode claim—is its authority contract.
    (STRICT27_MANIFEST_RELATIVE,
     "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9", None),
)
SOURCE_AUTHORITY_ASSET_PATHS = tuple(item[0] for item in SOURCE_AUTHORITY_ASSET_SPECS)

# These are the sealed strict-27 source-authority literals consumed by this
# route.  They intentionally live here rather than importing the historical
# TFSR package: importing that package executes its ``__init__`` and pulls a
# different model route into a fresh posterior-carrier stage.  The values are
# copied from the sealed authority chain, while all *bytes* are still read
# descriptor-safely from ``SOURCE_AUTHORITY_ASSET_SPECS`` below.
STRICT27_ADMISSION_SHA = "2632c6a6a4cfb8a4c0fb2b23e0cc8205ea323240b376903c60d5b27110f59e43"
STRICT27_THETA_RECEIPT_SHA = "d023dd632c4717443f1f55e924a09be1747fc58d5c30a8fb6fa38f4b7b117184"
STRICT27_THETA_ARTIFACT_SHA = "cef39dc8220aa253214963a32e5457dede1045e64b158e37fc267a6fb4146319"
STRICT27_SIDE_SEMANTIC_SHA = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
STRICT27_BEHAVIOR_SEMANTIC_SHA = "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
STRICT27_MANIFEST_SHA = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
STRICT27_SOURCE_LINEAGE_SHA = "7375a37c8c5e59cb7e6786e0c67e930f918d59ff4dc769742391ad6dc97579fa"
STRICT27_MATCHING_AUTHORITY_CONSUMED_SHA = "ccebdf41b3053703c35ad2323665aa6efcd7b91b39e9f5744aa3564745a82f71"
STRICT27_SOURCE_T4_MEAN = [0.04627712443470955, 0.4544036388397217, 1.3432163000106812, 10.150517463684082]
STRICT27_SOURCE_T4_STD = [1.126278281211853, 1.284820556640625, 1.2352101802825928, 9.115250587463379]

# The remote authority is intentionally Torch-only.  NVML currently cannot be
# trusted on the remote host, so no UUID/BDF/nvidia-smi value is invented.
REMOTE_TORCH_AUTHORITY = {
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

PHASE_B_CLOSURE_PATHS = (
    HANDOFF_RELATIVE,
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
    "tfpd_exploration/scripts/run_posterior_carrier_phase_b_source_smoke.py",
    "tfpd_exploration/tests/test_posterior_carrier_phase_b.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "sua_exploration/mc_maze/__init__.py",
) + SOURCE_AUTHORITY_ASSET_PATHS


class PhaseBError(RuntimeError):
    """Fail-closed error for a Phase-B contract or lifecycle violation."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseBError(message)


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PhaseBError(f"{name} must be an exact lowercase SHA-256")
    return value


def _immutable_roster(values: Sequence[str], *, expected_count: int | None = None) -> tuple[str, ...]:
    roster = tuple(values)
    if not roster or any(not isinstance(value, str) or not value for value in roster) or len(set(roster)) != len(roster):
        raise PhaseBError("source roster must be an ordered unique nonempty string tuple")
    if expected_count is not None and len(roster) != expected_count:
        raise PhaseBError(f"source roster must contain exactly {expected_count} sessions")
    return roster


def _safe_relative_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative:
        raise PhaseBError("stage asset path must be a safe nonempty relative path")
    candidate = Path(relative)
    if (candidate.is_absolute()
            or any(part in {"", ".", ".."} for part in candidate.parts)):
        raise PhaseBError("stage asset path must be a safe nonempty relative path")
    return candidate.parts


def descriptor_read_stage_file(
    stage_root: Path,
    relative: str,
    *,
    expected_sha256: str | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, int]]:
    """Read one stage-relative regular file through held no-follow FDs.

    This helper intentionally has no understanding of NWB/data roots.  It is
    used for code closure and the small immutable authority bundle only.  The
    source-data capability below owns the entirely separate data-root binding.
    """
    parts = _safe_relative_parts(relative)
    root = Path(stage_root).absolute()
    root_info = os.lstat(root)
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise PhaseBError("stage root must be a canonical non-symlink directory")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    held: list[int] = [root_fd]
    try:
        current_fd = root_fd
        for part in parts[:-1]:
            child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd)
            child_info = os.fstat(child_fd)
            if not stat.S_ISDIR(child_info.st_mode):
                os.close(child_fd)
                raise PhaseBError("stage asset parent is not a directory")
            held.append(child_fd)
            current_fd = child_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd)
        try:
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise PhaseBError("stage asset must be a regular non-symlink file")
            mode = stat.S_IMODE(info.st_mode)
            if expected_mode is not None and mode != expected_mode:
                raise PhaseBError("stage asset immutable mode drift")
            body = _read_all(file_fd)
        finally:
            os.close(file_fd)
        digest = sha256_bytes(body)
        if expected_sha256 is not None and digest != _sha(expected_sha256, "stage asset expected SHA"):
            raise PhaseBError("stage asset body SHA drift")
        root_after = os.fstat(root_fd)
        if (root_after.st_dev, root_after.st_ino) != (root_info.st_dev, root_info.st_ino):
            raise PhaseBError("stage root identity drift during descriptor read")
        return body, {"device": int(info.st_dev), "inode": int(info.st_ino), "mode": mode}
    finally:
        for descriptor in reversed(held):
            os.close(descriptor)


class _SourceDataRootSeal:
    pass


_SOURCE_DATA_ROOT_SEAL = _SourceDataRootSeal()


@dataclass(frozen=True)
class SourceDataRootCapability:
    """A typed, immutable binding to the external strict-27 NWB directory."""

    source_data_root: str
    directory_identity: tuple[int, int]
    _seal: object

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_carrier_strict27_source_data_root_v1",
            "source_data_root": self.source_data_root,
            "directory_device": self.directory_identity[0],
            "directory_inode": self.directory_identity[1],
            "nwb_copied_into_stage": False,
            "nwb_symlink_or_bind_mount_authorized": False,
        }

    def validate(self) -> Path:
        if self._seal is not _SOURCE_DATA_ROOT_SEAL:
            raise PhaseBError("source-data root capability is not root-issued")
        path = Path(self.source_data_root)
        info = os.lstat(path)
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or (info.st_dev, info.st_ino) != self.directory_identity):
            raise PhaseBError("external source-data root identity drift")
        return path

    def session_path(self, *, session: str, lineage_row: Mapping[str, object]) -> Path:
        root = self.validate()
        if not isinstance(session, str) or not session:
            raise PhaseBError("source-data session identifier drift")
        expected = root / f"{session}_behavior+ecephys.nwb"
        if (not isinstance(lineage_row, Mapping) or lineage_row.get("session") != session
                or lineage_row.get("path") != str(expected)):
            raise PhaseBError("sealed lineage path does not bind the external source-data root")
        if expected.parent != root or expected.is_symlink():
            raise PhaseBError("source NWB must be a direct non-symlink child of the external root")
        return expected


def validate_source_data_root_payload(value: Mapping[str, object]) -> dict[str, object]:
    expected_keys = {
        "schema", "source_data_root", "directory_device", "directory_inode",
        "nwb_copied_into_stage", "nwb_symlink_or_bind_mount_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise PhaseBError("source-data root payload schema drift")
    payload = dict(value)
    if (payload["schema"] != "posterior_carrier_strict27_source_data_root_v1"
            or payload["source_data_root"] != CANONICAL_SOURCE_DATA_ROOT
            or type(payload["directory_device"]) is not int or payload["directory_device"] < 0
            or type(payload["directory_inode"]) is not int or payload["directory_inode"] <= 0
            or payload["nwb_copied_into_stage"] is not False
            or payload["nwb_symlink_or_bind_mount_authorized"] is not False):
        raise PhaseBError("source-data root payload binding drift")
    return payload


def _issue_source_data_root_capability(
    *,
    source_data_root: Path,
    expected_canonical_root: Path | None = None,
) -> SourceDataRootCapability:
    """Root-reviewed factory; tests may provide an isolated expected root."""
    candidate = Path(source_data_root).absolute()
    expected = Path(CANONICAL_SOURCE_DATA_ROOT if expected_canonical_root is None else expected_canonical_root).absolute()
    if candidate != expected:
        raise PhaseBError("source-data root path differs from the frozen canonical external root")
    info = os.lstat(candidate)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PhaseBError("source-data root must be a canonical non-symlink directory")
    return SourceDataRootCapability(
        source_data_root=str(candidate), directory_identity=(int(info.st_dev), int(info.st_ino)),
        _seal=_SOURCE_DATA_ROOT_SEAL,
    )


def validate_stage_source_separation(
    *,
    stage_root: Path,
    source_data: SourceDataRootCapability,
) -> None:
    """Reject a stage that contains/aliases the external source-NWB tree."""
    stage = Path(stage_root).absolute()
    source = source_data.validate().absolute()
    info = os.lstat(stage)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PhaseBError("stage root must be a canonical non-symlink directory")
    try:
        source.relative_to(stage)
    except ValueError:
        return
    raise PhaseBError("fresh stage root must not contain the external source-NWB tree")


def phase_b_closure(root: Path) -> dict[str, object]:
    """Read-only explicit closure; no glob, data, output, or CUDA path."""
    base = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in PHASE_B_CLOSURE_PATHS:
        body, _identity = descriptor_read_stage_file(base, relative)
        hashes[relative] = sha256_bytes(body)
    if hashes[HANDOFF_RELATIVE] != HANDOFF_SHA256:
        raise PhaseBError("Phase-B handoff SHA drift")
    body = {"paths": list(PHASE_B_CLOSURE_PATHS), "sha256_by_path": hashes}
    return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def validate_phase_b_closure(value: Mapping[str, object]) -> dict[str, object]:
    expected_keys = {"paths", "sha256_by_path", "closure_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise PhaseBError("Phase-B closure schema drift")
    paths, hashes = value.get("paths"), value.get("sha256_by_path")
    if paths != list(PHASE_B_CLOSURE_PATHS) or not isinstance(hashes, Mapping) or set(hashes) != set(PHASE_B_CLOSURE_PATHS):
        raise PhaseBError("Phase-B closure paths drift")
    for relative in PHASE_B_CLOSURE_PATHS:
        _sha(hashes[relative], f"closure hash {relative}")
    if hashes[HANDOFF_RELATIVE] != HANDOFF_SHA256:
        raise PhaseBError("Phase-B closure handoff binding drift")
    body = {"paths": list(PHASE_B_CLOSURE_PATHS), "sha256_by_path": dict(hashes)}
    digest = sha256_bytes(canonical_json_bytes(body))
    if value.get("closure_sha256") != digest:
        raise PhaseBError("Phase-B closure digest drift")
    return {**body, "closure_sha256": digest}


@dataclass(frozen=True)
class SourceSmokeSpec:
    """Fixed engineering-only source smoke; this is not a full-training spec."""

    cell: str = CELL
    seed: int = 42
    batch_size: int = SOURCE_BATCH_SIZE
    optimizer_steps: int = SOURCE_SMOKE_STEPS
    source_epochs_authorized: int = 0
    train_epochs_held: int = SOURCE_EPOCHS
    train_steps_per_epoch_held: int = SOURCE_STEPS_PER_EPOCH
    no_target_or_scoring: bool = True

    def __post_init__(self) -> None:
        if (self.cell != CELL or self.seed != 42 or self.batch_size != SOURCE_BATCH_SIZE
                or self.optimizer_steps != SOURCE_SMOKE_STEPS or self.source_epochs_authorized != 0
                or self.train_epochs_held != SOURCE_EPOCHS
                or self.train_steps_per_epoch_held != SOURCE_STEPS_PER_EPOCH
                or self.no_target_or_scoring is not True):
            raise PhaseBError("source-smoke spec drift")

    def payload(self) -> dict[str, object]:
        return {
            "cell": self.cell,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "optimizer_steps": self.optimizer_steps,
            "source_epochs_authorized": self.source_epochs_authorized,
            "train_epochs_held": self.train_epochs_held,
            "train_steps_per_epoch_held": self.train_steps_per_epoch_held,
            "train_total_steps_held": SOURCE_TOTAL_STEPS,
            "no_target_or_scoring": self.no_target_or_scoring,
        }


SMOKE_SPEC = SourceSmokeSpec()


@dataclass(frozen=True)
class SmokeExecutionProgress:
    """Monotone physical progress carried through honest failure receipts."""

    source_opened: bool = False
    remote_initialized: bool = False
    optimizer_steps_completed: int = 0
    source_authority_sha256: str | None = None

    def __post_init__(self) -> None:
        if (type(self.source_opened) is not bool or type(self.remote_initialized) is not bool
                or type(self.optimizer_steps_completed) is not int
                or not 0 <= self.optimizer_steps_completed <= SOURCE_SMOKE_STEPS):
            raise PhaseBError("source-smoke progress schema drift")
        if self.source_authority_sha256 is not None:
            _sha(self.source_authority_sha256, "source-smoke progress source-authority SHA")

    def payload(self) -> dict[str, object]:
        return {
            "source_opened": self.source_opened,
            "remote_initialized": self.remote_initialized,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "source_authority_sha256": self.source_authority_sha256,
        }

    def merge(self, other: "SmokeExecutionProgress") -> "SmokeExecutionProgress":
        if not isinstance(other, SmokeExecutionProgress):
            raise PhaseBError("source-smoke progress merge type drift")
        source_authority = self.source_authority_sha256 or other.source_authority_sha256
        if (self.source_authority_sha256 is not None and other.source_authority_sha256 is not None
                and self.source_authority_sha256 != other.source_authority_sha256):
            raise PhaseBError("source-smoke progress source-authority SHA conflict")
        return SmokeExecutionProgress(
            source_opened=self.source_opened or other.source_opened,
            remote_initialized=self.remote_initialized or other.remote_initialized,
            optimizer_steps_completed=max(self.optimizer_steps_completed, other.optimizer_steps_completed),
            source_authority_sha256=source_authority,
        )


class SourceSmokeExecutionError(PhaseBError):
    """Typed backend failure with the exact physical state reached so far."""

    def __init__(self, *, stage: str, progress: SmokeExecutionProgress, cause: BaseException) -> None:
        if stage not in {"prepare", "source_authority", "steps", "terminal"}:
            raise PhaseBError("typed source-smoke error stage drift")
        if not isinstance(progress, SmokeExecutionProgress):
            raise PhaseBError("typed source-smoke error progress drift")
        self.stage = stage
        self.progress = progress
        self.cause = cause
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")


def _runtime_progress(runtime: Any | None) -> SmokeExecutionProgress:
    if runtime is None:
        return SmokeExecutionProgress()
    explicit = getattr(runtime, "progress", None)
    if isinstance(explicit, SmokeExecutionProgress):
        return explicit
    return SmokeExecutionProgress(
        source_opened=bool(getattr(runtime, "source_opened", isinstance(runtime, Mapping) and runtime.get("source_opened", False))),
        remote_initialized=bool(getattr(runtime, "remote_initialized", isinstance(runtime, Mapping) and runtime.get("remote_initialized", False))),
        optimizer_steps_completed=int(getattr(runtime, "optimizer_steps_completed", 0)),
    )


@dataclass(frozen=True)
class PosteriorEpochCacheObserver:
    """Immutable audit record for pre-loop posterior work and batch-loop absence."""

    posterior_fit_calls: int
    posterior_inverse_calls: int
    deterministic_mean_view_builds: int
    epoch_sampled_view_builds: int
    device_epoch_view_builds: int
    normalized_view_builds: int
    batch_loop_requests: int
    batch_loop_inverse_calls: int
    source_sessions: int
    scheduled_session_epochs: int

    def payload(self) -> dict[str, int]:
        return {
            "posterior_fit_calls": self.posterior_fit_calls,
            "posterior_inverse_calls": self.posterior_inverse_calls,
            "deterministic_mean_view_builds": self.deterministic_mean_view_builds,
            "epoch_sampled_view_builds": self.epoch_sampled_view_builds,
            "device_epoch_view_builds": self.device_epoch_view_builds,
            "normalized_view_builds": self.normalized_view_builds,
            "batch_loop_requests": self.batch_loop_requests,
            "batch_loop_inverse_calls": self.batch_loop_inverse_calls,
            "source_sessions": self.source_sessions,
            "scheduled_session_epochs": self.scheduled_session_epochs,
        }


class PosteriorEpochCarrierBank:
    """Pre-loop posterior bank and one-view-per-session/epoch cache.

    Fitting is deliberately keyed by ``(session, budget)`` because every
    deterministic posterior is reused for sixteen logical epochs.  Sampling is
    keyed by ``(session, epoch)`` and thus creates exactly 27 × 48 training
    views over a full run.  The optimizer batch loop can only retrieve an
    existing view; it never calls the inverse or a posterior fitting function.
    """

    def __init__(
        self,
        *,
        roster: Sequence[str],
        posterior_by_session_budget: Mapping[str, Mapping[int, core.PosteriorCarrier]],
        normalizer: core.PosteriorSourceT4Normalizer,
        seed: int = 42,
    ) -> None:
        self._roster = _immutable_roster(roster, expected_count=None)
        if normalizer.source_roster != self._roster:
            raise PhaseBError("posterior bank/normalizer source roster drift")
        if not isinstance(posterior_by_session_budget, Mapping) or set(posterior_by_session_budget) != set(self._roster):
            raise PhaseBError("posterior bank source session topology drift")
        copied: dict[str, dict[int, core.PosteriorCarrier]] = {}
        for session in self._roster:
            rows = posterior_by_session_budget[session]
            if not isinstance(rows, Mapping) or set(rows) != set(BUDGETS):
                raise PhaseBError("posterior bank requires exactly M4/M10/M30 per source session")
            copied[session] = {}
            for budget in BUDGETS:
                carrier = rows[budget]
                if not isinstance(carrier, core.PosteriorCarrier):
                    raise PhaseBError("posterior bank carrier type drift")
                copied[session][budget] = carrier
        self._posteriors = copied
        self._normalizer = normalizer
        self._sampler = core.SessionStaticPosteriorSampler(seed=seed, normalizer=normalizer)
        self._mean_views: dict[tuple[str, int], core.PosteriorCarrierView] = {}
        self._epoch_views: dict[tuple[str, int], core.PosteriorCarrierView] = {}
        # Device views are materialized at an explicit logical-epoch boundary,
        # never by an optimizer batch.  The full typed carrier is copied (not
        # just normalized T4) so raw-before-normalization and credibility
        # lineage remain inspectable on the model device.
        self._device_epoch_views: dict[tuple[str, int, str], core.PosteriorCarrierView] = {}
        self._batch_loop_requests = 0

    @property
    def roster(self) -> tuple[str, ...]:
        return self._roster

    @property
    def normalizer(self) -> core.PosteriorSourceT4Normalizer:
        return self._normalizer

    def budget_for(self, *, session: str, epoch: int) -> int:
        if session not in self._posteriors:
            raise PhaseBError("posterior bank refuses a non-source session")
        return core.budget_for_epoch(epoch, self._roster.index(session))

    def posterior_for(self, *, session: str, epoch: int) -> core.PosteriorCarrier:
        return self._posteriors[session][self.budget_for(session=session, epoch=epoch)]

    def deterministic_mean_view(self, *, session: str, budget: int) -> core.PosteriorCarrierView:
        if session not in self._posteriors or budget not in BUDGETS:
            raise PhaseBError("posterior bank mean-view key drift")
        key = (session, budget)
        cached = self._mean_views.get(key)
        if cached is None:
            cached = core.posterior_mean_view(self._posteriors[session][budget], self._normalizer)
            self._mean_views[key] = cached.clone()
        return cached.clone()

    def training_view(self, *, session: str, epoch: int) -> core.PosteriorCarrierView:
        if session not in self._posteriors or type(epoch) is not int or not 0 <= epoch < SOURCE_EPOCHS:
            raise PhaseBError("posterior bank source session/epoch drift")
        key = (session, epoch)
        cached = self._epoch_views.get(key)
        if cached is None:
            cached = self._sampler.carrier_for(
                self.posterior_for(session=session, epoch=epoch), session_id=session, epoch=epoch, training=True,
            )
            self._epoch_views[key] = cached.clone()
        return cached.clone()

    @staticmethod
    def _device_key(device: torch.device | str) -> str:
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            raise PhaseBError("posterior device cache requires an explicit CUDA logical index")
        return str(resolved)

    @staticmethod
    def _copy_view_to_device(
        view: core.PosteriorCarrierView,
        *,
        device: torch.device | str,
    ) -> core.PosteriorCarrierView:
        # All tensors are detached carrier constants.  ``copy=True`` makes the
        # cache boundary explicit even for a CPU synthetic test, while all
        # finite checks happen before the optimizer's timed core.
        destination = torch.device(device)
        return core.PosteriorCarrierView(
            raw_beta=view.raw_beta.to(device=destination, copy=True),
            raw_t4=view.raw_t4.to(device=destination, copy=True),
            normalized_t4=view.normalized_t4.to(device=destination, copy=True),
            credibility=view.credibility.to(device=destination, copy=True),
            zero_spike_mask=view.zero_spike_mask.to(device=destination, copy=True),
            sampled=view.sampled,
            session_id=view.session_id,
            epoch=view.epoch,
            posterior_sha256=view.posterior_sha256,
            normalizer_authority_sha256=view.normalizer_authority_sha256,
        )

    def materialize_epoch_for_device(self, *, epoch: int, device: torch.device | str) -> None:
        """Copy all prewarmed session carriers once at an epoch/device boundary."""
        if type(epoch) is not int or not 0 <= epoch < SOURCE_EPOCHS:
            raise PhaseBError("posterior device cache epoch drift")
        device_key = self._device_key(device)
        for session in self._roster:
            cpu_key = (session, epoch)
            if cpu_key not in self._epoch_views:
                raise PhaseBError("posterior device cache requires a prewarmed CPU carrier")
            key = (session, epoch, device_key)
            if key not in self._device_epoch_views:
                self._device_epoch_views[key] = self._copy_view_to_device(
                    self._epoch_views[cpu_key], device=device,
                )

    def optimizer_batch_view(
        self,
        *,
        session: str,
        epoch: int,
        device: torch.device | str | None = None,
    ) -> core.PosteriorCarrierView:
        """The sole batch-loop entry point; it cannot fit/invert a posterior."""
        key = (session, epoch)
        if key not in self._epoch_views:
            # First construction belongs exclusively to a logical
            # session/epoch boundary.  Reject *before* asking the sampler for a
            # view: a failed batch-loop request must not create a sample,
            # consume any route-local work, or mutate the cache it is auditing.
            # The physical backend prewarms every required epoch view before
            # constructing its one DataLoader iterator.
            raise PhaseBError("optimizer batch loop attempted to construct a posterior carrier view")
        if device is not None:
            device_key = self._device_key(device)
            device_view = self._device_epoch_views.get((session, epoch, device_key))
            if device_view is None:
                raise PhaseBError("optimizer batch loop attempted to materialize a device posterior carrier")
            view = device_view
        else:
            view = self._epoch_views[key]
        self._batch_loop_requests += 1
        # ``PosteriorCarrierView`` is frozen and CellDPosteriorWrapper only
        # uses out-of-place device/dtype conversions and expands.  Reusing this
        # exact object therefore preserves the prepared sample while avoiding a
        # fresh CPU clone of every raw/covariance/normalized carrier on each
        # B32 optimizer step.
        return view

    def prewarm_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or not 0 <= epoch < SOURCE_EPOCHS:
            raise PhaseBError("posterior bank prewarm epoch drift")
        for session in self._roster:
            self.training_view(session=session, epoch=epoch)

    def reference_uncached_training_view(self, *, session: str, epoch: int) -> core.PosteriorCarrierView:
        """Synthetic parity reference; it does not mutate the bank cache."""
        carrier = self.posterior_for(session=session, epoch=epoch)
        return core._sample_view(carrier, self._normalizer, session_id=session, epoch=epoch, seed=42)

    def observer(self, *, expected_full_schedule: bool = False) -> PosteriorEpochCacheObserver:
        fit_calls = len(self._roster) * len(BUDGETS)
        scheduled = len(self._roster) * SOURCE_EPOCHS if expected_full_schedule else len(self._epoch_views)
        if expected_full_schedule and len(self._epoch_views) != scheduled:
            raise PhaseBError("full-run posterior carrier cache is incomplete")
        return PosteriorEpochCacheObserver(
            posterior_fit_calls=fit_calls,
            posterior_inverse_calls=fit_calls,
            deterministic_mean_view_builds=len(self._mean_views),
            epoch_sampled_view_builds=len(self._epoch_views),
            device_epoch_view_builds=len(self._device_epoch_views),
            # Device entries are immutable copies of an already normalized
            # view, not a second normalizer fit or normalization operation.
            normalized_view_builds=len(self._epoch_views) + len(self._mean_views),
            batch_loop_requests=self._batch_loop_requests,
            batch_loop_inverse_calls=0,
            source_sessions=len(self._roster),
            scheduled_session_epochs=scheduled,
        )

    def source_posterior_digest_payload(self) -> dict[str, object]:
        # Keep the posterior digest *and* the immutable direct-prefix bindings
        # together.  This makes a receipt independently checkable without
        # serializing raw tensors or trusting an opaque carrier digest.
        posteriors: dict[str, dict[str, dict[str, object]]] = {}
        for session in self._roster:
            posteriors[session] = {}
            for budget in BUDGETS:
                carrier = self._posteriors[session][budget]
                posteriors[session][str(budget)] = {
                    "posterior_sha256": carrier.digest(),
                    "unit_count": int(carrier.mean.shape[0]),
                    "counts_sha256": carrier.counts_sha256,
                    "exposure_sha256": carrier.exposure_sha256,
                    "theta_sha256": carrier.theta_sha256,
                    "prior_raw_m30_t4_sha256": carrier.prior.raw_m30_t4_sha256,
                }
        return {
            "roster": list(self._roster),
            "roster_sha256": sha256_bytes(canonical_json_bytes(list(self._roster))),
            "posteriors": posteriors,
            "normalizer": self._normalizer.payload(),
            "schedule_sha256": core.budget_schedule_digest(
                core.build_budget_schedule(epochs=SOURCE_EPOCHS, session_count=len(self._roster))
            ),
        }


def build_posterior_epoch_bank(
    *,
    roster: Sequence[str],
    posterior_by_session_budget: Mapping[str, Mapping[int, core.PosteriorCarrier]],
    seed: int = 42,
) -> PosteriorEpochCarrierBank:
    """Fit the Phase-B source-only normalizer before any sampling or model loop."""
    strict_roster = _immutable_roster(roster)
    raw = {
        budget: {
            session: posterior_by_session_budget[session][budget].raw_t4
            for session in strict_roster
        }
        for budget in BUDGETS
    }
    normalizer = core.PosteriorSourceT4Normalizer.fit(
        source_roster=strict_roster,
        raw_mean_t4_by_budget=raw,
    )
    return PosteriorEpochCarrierBank(
        roster=strict_roster,
        posterior_by_session_budget=posterior_by_session_budget,
        normalizer=normalizer,
        seed=seed,
    )


def critical_gradient_proof(model: Any) -> dict[str, bool]:
    """Require eight live Cell-D paths, including the T4-only B3S block slice."""
    base = getattr(model, "cell_d", model)
    named = dict(base.named_parameters())
    required = {
        "b3s_pre_pool_activity": ("id_encoder.pre_pool.0.weight", None),
        "b3s_post_pool_activity": ("id_encoder.post_pool.0.weight", slice(0, 64)),
        "b3s_post_pool_t4": ("id_encoder.post_pool.0.weight", slice(64, 68)),
        "decoder_fc_in": ("decoder.fc_in.0.weight", None),
        "cross_attention": ("decoder.transformer.layers.0.cross_attn.in_proj_weight", None),
        "ffn": ("decoder.transformer.layers.0.ffn.0.weight", None),
        "query_rep": ("decoder.rep", None),
        "output_fc": ("decoder.fc_out.weight", None),
    }
    proof: dict[str, bool] = {}
    for label, (name, columns) in required.items():
        parameter = named.get(name)
        if parameter is None or parameter.grad is None:
            proof[label] = False
            continue
        gradient = parameter.grad if columns is None else parameter.grad[:, columns]
        proof[label] = bool(torch.isfinite(gradient).all().item() and gradient.abs().sum().item() > 0)
    if not all(proof.values()):
        raise PhaseBError("one or more required posterior-Cell-D gradients are absent/nonfinite/zero")
    return proof


def finite_model_and_adam(model: Any, optimizer: Any) -> tuple[bool, bool]:
    """Boundary-only full state scan; callers must not invoke this per batch."""
    from torch.nn.parameter import UninitializedParameter

    finite_parameters = True
    for value in model.parameters():
        if isinstance(value, UninitializedParameter):
            continue
        finite_parameters &= bool(torch.isfinite(value).all().item())
    finite_adam = True
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                finite_adam &= bool(torch.isfinite(value).all().item())
    return finite_parameters, finite_adam


def lazy_safe_state_sha256(model: Any) -> str:
    """State digest compatible with Cell-D's two inactive lazy parameters."""
    from torch.nn.parameter import UninitializedParameter

    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        if isinstance(value, UninitializedParameter):
            digest.update(b"|uninitialized-lazy|")
        else:
            digest.update(core.tensor_digest(value).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RemoteTorchAttestation:
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if dict(self.payload) != REMOTE_TORCH_AUTHORITY:
            raise PhaseBError("remote Torch-only device authority drift")


def attest_remote_torch_only(torch_module: Any, *, nvml_status: str) -> RemoteTorchAttestation:
    """Execution-only gate; does not query or infer an NVML identity."""
    if nvml_status != REMOTE_TORCH_AUTHORITY["nvml_status"]:
        raise PhaseBError("remote NVML mismatch status drift")
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise PhaseBError("remote Torch gate requires exactly one visible CUDA device")
    properties = torch_module.cuda.get_device_properties(0)
    payload = {
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(torch_module.version.cuda),
        "cudnn_version": int(torch_module.backends.cudnn.version()),
        "visible_devices": int(torch_module.cuda.device_count()),
        "logical_device": "cuda:0",
        "name": str(properties.name),
        "capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "nvml_status": nvml_status,
    }
    return RemoteTorchAttestation(payload=payload)


@dataclass(frozen=True)
class RunIdentity:
    source_authority: Mapping[str, object]
    closure: Mapping[str, object]
    remote_device: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        return {
            "cell": CELL,
            "phase": PHASE_B,
            "handoff": {"path": HANDOFF_RELATIVE, "sha256": HANDOFF_SHA256},
            "phase_b_normalizer_amendment": PHASE_B_NORMALIZER_AMENDMENT,
            "source_authority": dict(self.source_authority),
            "closure": validate_phase_b_closure(self.closure),
            "remote_device": dict(self.remote_device),
            "boundaries": {
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
            },
        }


def validate_run_identity(value: RunIdentity) -> None:
    if not isinstance(value, RunIdentity):
        raise PhaseBError("run identity type drift")
    data = value.payload()
    if data["handoff"] != {"path": HANDOFF_RELATIVE, "sha256": HANDOFF_SHA256}:
        raise PhaseBError("run identity handoff drift")
    if data["remote_device"] != REMOTE_TORCH_AUTHORITY:
        raise PhaseBError("run identity remote Torch authority drift")
    source = data["source_authority"]
    expected_source_keys = {
        "schema", "roster", "roster_sha256", "strict_source_metadata_sha256", "manifest_sha256",
        "ordinary_raw_t4_semantic_sha256", "behavior_normalizer_semantic_sha256", "source_lineage_sha256",
        "source_data_root",
        "source_only", "target_opened", "within_opened", "external_opened", "formal_opened", "h1_opened",
    }
    if not isinstance(source, Mapping) or set(source) != expected_source_keys:
        raise PhaseBError("run identity strict-source authority schema drift")
    roster = _immutable_roster(source.get("roster", ()), expected_count=SOURCE_SESSION_COUNT)
    if (source.get("schema") != "posterior_carrier_target_free_source_identity_v1"
            or source.get("roster_sha256") != sha256_bytes(canonical_json_bytes(list(roster)))
            or source.get("source_only") is not True
            or any(source.get(key) is not False for key in ("target_opened", "within_opened", "external_opened", "formal_opened", "h1_opened"))):
        raise PhaseBError("run identity strict-source boundary drift")
    for key in ("strict_source_metadata_sha256", "manifest_sha256", "ordinary_raw_t4_semantic_sha256",
                "behavior_normalizer_semantic_sha256", "source_lineage_sha256"):
        _sha(source.get(key), f"run identity {key}")
    validate_source_data_root_payload(source.get("source_data_root"))
    if data["boundaries"] != {
        "source_only": True, "target_opened": False, "within_opened": False,
        "external_opened": False, "formal_opened": False, "h1_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0, "scientific_score": False,
    }:
        raise PhaseBError("run identity boundary drift")


def _finite_number(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseBError(f"{name} must be a finite numeric scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise PhaseBError(f"{name} finite/range drift")
    return result


def _validate_source_metadata_crosslinks(
    value: Mapping[str, object],
    *,
    roster: Sequence[str],
    expected_source: Mapping[str, object],
) -> dict[str, object]:
    """Validate the route-local staged strict-27 authority schema.

    This is deliberately stronger than a body-SHA check: posterior evidence
    later names individual source files and unit axes, so that metadata must be
    reconstructed and cross-bound before publication.
    """
    expected_keys = {
        "admission_preflight", "theta_receipt", "theta_artifact", "roster",
        "normalized_t4_sha256", "raw_authority_sha256", "normalizer_authority_sha256",
        "side_semantic_sha256", "behavior_semantic_sha256", "manifest_sha256",
        "stage_manifest_relative", "source_data_root", "stage_authority_assets",
        "source_lineage",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise PhaseBError("source authority metadata schema drift")
    metadata = dict(value)
    if (metadata["roster"] != list(roster)
            or metadata["side_semantic_sha256"] != expected_source["ordinary_raw_t4_semantic_sha256"]
            or metadata["behavior_semantic_sha256"] != expected_source["behavior_normalizer_semantic_sha256"]
            or metadata["manifest_sha256"] != expected_source["manifest_sha256"]
            or metadata["source_data_root"] != expected_source["source_data_root"]
            or metadata["stage_manifest_relative"] != STRICT27_MANIFEST_RELATIVE):
        raise PhaseBError("source authority metadata identity binding drift")
    normalized = metadata["normalized_t4_sha256"]
    if not isinstance(normalized, Mapping) or set(normalized) != set(roster):
        raise PhaseBError("source authority normalized-T4 roster/schema drift")
    for item in normalized.values():
        _sha(item, "source normalized T4 SHA")
    for key in ("raw_authority_sha256", "normalizer_authority_sha256"):
        _sha(metadata[key], f"source authority metadata {key}")
    for key in ("admission_preflight", "theta_receipt", "theta_artifact"):
        entry = metadata[key]
        if (not isinstance(entry, Mapping) or set(entry) != {"path", "body_sha256"}
                or not isinstance(entry["path"], str)):
            raise PhaseBError("source authority sealed asset schema drift")
        _sha(entry["body_sha256"], f"source authority {key} body SHA")
    assets = metadata["stage_authority_assets"]
    if not isinstance(assets, Mapping) or set(assets) != set(SOURCE_AUTHORITY_ASSET_PATHS):
        raise PhaseBError("source authority staged-asset topology drift")
    for relative in SOURCE_AUTHORITY_ASSET_PATHS:
        entry = assets[relative]
        if (not isinstance(entry, Mapping) or set(entry) != {"sha256", "descriptor_identity"}):
            raise PhaseBError("source authority staged-asset schema drift")
        _sha(entry["sha256"], "source authority staged asset SHA")
        descriptor = entry["descriptor_identity"]
        if (not isinstance(descriptor, Mapping) or set(descriptor) != {"device", "inode", "mode"}
                or type(descriptor["device"]) is not int or descriptor["device"] < 0
                or type(descriptor["inode"]) is not int or descriptor["inode"] <= 0
                or type(descriptor["mode"]) is not int or not 0 <= descriptor["mode"] <= 0o777):
            raise PhaseBError("source authority staged-asset descriptor drift")
    lineage = metadata["source_lineage"]
    if (not isinstance(lineage, Mapping)
            or set(lineage) != {"path", "body_sha256", "matching_authority_consumed_bytes_sha256", "rows_by_session"}
            or lineage["body_sha256"] != expected_source["source_lineage_sha256"]
            or not isinstance(lineage["path"], str)):
        raise PhaseBError("source authority lineage schema/binding drift")
    _sha(lineage["matching_authority_consumed_bytes_sha256"], "source authority matching SHA")
    rows = lineage["rows_by_session"]
    expected_row_keys = {"feature_version", "path", "session", "sha256", "size_bytes", "unit_count"}
    if not isinstance(rows, Mapping) or set(rows) != set(roster):
        raise PhaseBError("source authority lineage roster drift")
    for session in roster:
        row = rows[session]
        if (not isinstance(row, Mapping) or set(row) != expected_row_keys
                or row["session"] != session or row["feature_version"] != 1
                or row["path"] != f"{CANONICAL_SOURCE_DATA_ROOT}/{session}_behavior+ecephys.nwb"
                or type(row["size_bytes"]) is not int or row["size_bytes"] <= 0
                or type(row["unit_count"]) is not int or row["unit_count"] <= 0):
            raise PhaseBError("source authority lineage row drift")
        _sha(row["sha256"], "source authority lineage file SHA")
    return metadata


def _validate_posterior_prior_payload(
    value: Mapping[str, object], *, roster: Sequence[str], source_raw_sha256: str,
) -> dict[str, object]:
    expected_keys = {
        "schema", "mu0", "source_mean_b", "tau_ac2", "tau_b2",
        "directional_prior_mean_exact_zero", "directional_prior_isotropic",
        "variance_floor", "source_roster", "source_roster_sha256", "raw_m30_t4_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise PhaseBError("posterior prior schema drift")
    prior = dict(value)
    if prior["schema"] != core.SOURCE_PRIOR_SCHEMA or prior["source_roster"] != list(roster):
        raise PhaseBError("posterior prior source roster/schema drift")
    if prior["source_roster_sha256"] != sha256_bytes(canonical_json_bytes(list(roster))):
        raise PhaseBError("posterior prior roster digest drift")
    mu0 = prior["mu0"]
    if not isinstance(mu0, list) or len(mu0) != 3:
        raise PhaseBError("posterior prior mu0 schema drift")
    mean_b = _finite_number(prior["source_mean_b"], name="posterior prior source_mean_b")
    if (_finite_number(mu0[0], name="posterior prior mu0[0]") != 0.0
            or _finite_number(mu0[1], name="posterior prior mu0[1]") != 0.0
            or _finite_number(mu0[2], name="posterior prior mu0[2]") != mean_b
            or prior["directional_prior_mean_exact_zero"] is not True
            or prior["directional_prior_isotropic"] is not True
            or _finite_number(prior["tau_ac2"], name="posterior prior tau_ac2", positive=True) < core.PRIOR_VARIANCE_FLOOR
            or _finite_number(prior["tau_b2"], name="posterior prior tau_b2", positive=True) < core.PRIOR_VARIANCE_FLOOR
            or _finite_number(prior["variance_floor"], name="posterior prior variance floor", positive=True) != core.PRIOR_VARIANCE_FLOOR):
        raise PhaseBError("posterior prior moment semantics drift")
    if prior["raw_m30_t4_sha256"] != _sha(source_raw_sha256, "source raw M30 T4 SHA"):
        raise PhaseBError("posterior prior/source raw T4 cross-binding drift")
    return prior


def _validate_posterior_normalizer_payload(value: Mapping[str, object], *, roster: Sequence[str]) -> dict[str, object]:
    body_keys = {
        "schema", "source_roster", "source_roster_sha256", "row_order", "row_count",
        "per_budget_row_counts", "per_budget_raw_rows_sha256", "raw_rows_sha256",
        "mean_float64", "std_float64", "ddof", "source_only",
        "contains_only_deterministic_posterior_means", "all_zero_raw_rows_retained",
    }
    expected_keys = body_keys | {"body_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise PhaseBError("posterior normalizer schema drift")
    normalizer = dict(value)
    body = {key: normalizer[key] for key in body_keys}
    if (normalizer["schema"] != core.POSTERIOR_SOURCE_NORMALIZER_SCHEMA
            or normalizer["source_roster"] != list(roster)
            or normalizer["source_roster_sha256"] != sha256_bytes(canonical_json_bytes(list(roster)))
            or normalizer["row_order"] != core.PosteriorSourceT4Normalizer.ROW_ORDER
            or normalizer["ddof"] != 0
            or normalizer["source_only"] is not True
            or normalizer["contains_only_deterministic_posterior_means"] is not True
            or normalizer["all_zero_raw_rows_retained"] is not True):
        raise PhaseBError("posterior normalizer semantic schema drift")
    counts, per_budget_sha = normalizer["per_budget_row_counts"], normalizer["per_budget_raw_rows_sha256"]
    budget_keys = {str(budget) for budget in BUDGETS}
    if not isinstance(counts, Mapping) or not isinstance(per_budget_sha, Mapping) or set(counts) != budget_keys or set(per_budget_sha) != budget_keys:
        raise PhaseBError("posterior normalizer budget topology drift")
    values: list[int] = []
    for budget in BUDGETS:
        count = counts[str(budget)]
        if type(count) is not int or count < 1:
            raise PhaseBError("posterior normalizer budget count drift")
        values.append(count)
        _sha(per_budget_sha[str(budget)], "posterior normalizer per-budget rows SHA")
    if len(set(values)) != 1 or type(normalizer["row_count"]) is not int or normalizer["row_count"] != sum(values):
        raise PhaseBError("posterior normalizer equal-budget/count drift")
    for field, positive in (("mean_float64", False), ("std_float64", True)):
        values4 = normalizer[field]
        if (not isinstance(values4, list) or len(values4) != 4
                or any(not math.isfinite(_finite_number(item, name=f"posterior normalizer {field}"))
                       or (positive and _finite_number(item, name=f"posterior normalizer {field}") <= 0.0)
                       for item in values4)):
            raise PhaseBError("posterior normalizer moment drift")
    _sha(normalizer["raw_rows_sha256"], "posterior normalizer raw rows SHA")
    if normalizer["body_sha256"] != sha256_bytes(canonical_json_bytes(body)):
        raise PhaseBError("posterior normalizer body SHA drift")
    return normalizer


def _validate_posterior_inputs_payload(
    value: Mapping[str, object], *, roster: Sequence[str], metadata: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != set(roster):
        raise PhaseBError("posterior inputs strict-27 topology drift")
    rows = metadata["source_lineage"]["rows_by_session"]
    expected_keys = {
        "session_id", "raw_m30_t4_sha256", "counts_m30_sha256", "exposure_m30_sha256", "theta_m30_sha256",
        "prefix_evidence_by_budget", "prefix_row_ids", "prefix_rows_sha256", "source_path_sha256",
        "unit_order_sha256", "raw_t4_row_order_proof", "unit_count", "direct_integer_counts", "exposure_semantics", "theta_semantics",
    }
    result: dict[str, dict[str, object]] = {}
    budget_keys = {str(budget) for budget in BUDGETS}
    for session in roster:
        item = value[session]
        if not isinstance(item, Mapping) or set(item) != expected_keys or item["session_id"] != session:
            raise PhaseBError("posterior input schema/session drift")
        for field in ("raw_m30_t4_sha256", "counts_m30_sha256", "exposure_m30_sha256", "theta_m30_sha256", "source_path_sha256", "unit_order_sha256"):
            _sha(item[field], f"posterior input {field}")
        row_ids = item["prefix_row_ids"]
        if (not isinstance(row_ids, list) or len(row_ids) != 30 or len(set(row_ids)) != 30
                or any(not isinstance(row_id, str) or not row_id for row_id in row_ids)
                or item["prefix_rows_sha256"] != sha256_bytes(canonical_json_bytes(row_ids))):
            raise PhaseBError("posterior input prefix-row authority drift")
        if (item["source_path_sha256"] != rows[session]["sha256"]
                or item["unit_count"] != rows[session]["unit_count"]
                or type(item["unit_count"]) is not int or item["unit_count"] <= 0
                or item["direct_integer_counts"] is not True
                or item["exposure_semantics"] != "trial_stop_time_minus_start_time_seconds"
                or item["theta_semantics"] != "chronological_labelled_rewarded_trial_target_direction_radians"):
            raise PhaseBError("posterior input source/file/count semantics drift")
        proof = item["raw_t4_row_order_proof"]
        expected_proof = {
            "feature_group", "feature_version", "pool_size", "signal_view", "source_unit_count",
            "channel_ids_are_exact_arange", "raw_row_count", "unit_order_sha256", "row_semantics",
        }
        if (not isinstance(proof, Mapping) or set(proof) != expected_proof
                or proof["feature_group"] != "t4" or proof["feature_version"] != 1 or proof["pool_size"] != 30
                or proof["signal_view"] != "sua" or proof["source_unit_count"] != item["unit_count"]
                or proof["raw_row_count"] != item["unit_count"]
                or proof["channel_ids_are_exact_arange"] is not True
                or proof["unit_order_sha256"] != item["unit_order_sha256"]
                or proof["row_semantics"] != "closure_bound_compute_unit_side_features_uncached_sua_rows_follow_nwb_units_order"):
            raise PhaseBError("posterior input raw-T4/unit-order proof drift")
        prefixes = item["prefix_evidence_by_budget"]
        if not isinstance(prefixes, Mapping) or set(prefixes) != budget_keys:
            raise PhaseBError("posterior input budget-prefix topology drift")
        for budget in BUDGETS:
            prefix = prefixes[str(budget)]
            if not isinstance(prefix, Mapping) or set(prefix) != {"counts_sha256", "exposure_sha256", "theta_sha256"}:
                raise PhaseBError("posterior input budget-prefix schema drift")
            for digest in prefix.values():
                _sha(digest, "posterior input budget-prefix SHA")
        result[session] = dict(item)
    return result


def _validate_posterior_bank_payload(
    value: Mapping[str, object], *, roster: Sequence[str], normalizer: Mapping[str, object],
    inputs: Mapping[str, Mapping[str, object]], prior: Mapping[str, object],
) -> dict[str, object]:
    expected_keys = {"roster", "roster_sha256", "posteriors", "normalizer", "schedule_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise PhaseBError("posterior bank schema drift")
    bank = dict(value)
    if (bank["roster"] != list(roster)
            or bank["roster_sha256"] != sha256_bytes(canonical_json_bytes(list(roster)))
            or bank["normalizer"] != normalizer
            or bank["schedule_sha256"] != core.budget_schedule_digest(
                core.build_budget_schedule(epochs=SOURCE_EPOCHS, session_count=len(roster))
            )):
        raise PhaseBError("posterior bank roster/normalizer/schedule drift")
    all_posteriors = bank["posteriors"]
    budget_keys = {str(budget) for budget in BUDGETS}
    expected_entry_keys = {
        "posterior_sha256", "unit_count", "counts_sha256", "exposure_sha256", "theta_sha256",
        "prior_raw_m30_t4_sha256",
    }
    if not isinstance(all_posteriors, Mapping) or set(all_posteriors) != set(roster):
        raise PhaseBError("posterior bank strict-27 map drift")
    for session in roster:
        session_entries = all_posteriors[session]
        if not isinstance(session_entries, Mapping) or set(session_entries) != budget_keys:
            raise PhaseBError("posterior bank budget map drift")
        item = inputs[session]
        prefixes = item["prefix_evidence_by_budget"]
        for budget in BUDGETS:
            entry = session_entries[str(budget)]
            if not isinstance(entry, Mapping) or set(entry) != expected_entry_keys:
                raise PhaseBError("posterior bank entry schema drift")
            for field in ("posterior_sha256", "counts_sha256", "exposure_sha256", "theta_sha256", "prior_raw_m30_t4_sha256"):
                _sha(entry[field], f"posterior bank {field}")
            if (type(entry["unit_count"]) is not int or entry["unit_count"] != item["unit_count"]
                    or entry["counts_sha256"] != prefixes[str(budget)]["counts_sha256"]
                    or entry["exposure_sha256"] != prefixes[str(budget)]["exposure_sha256"]
                    or entry["theta_sha256"] != prefixes[str(budget)]["theta_sha256"]
                    or entry["prior_raw_m30_t4_sha256"] != prior["raw_m30_t4_sha256"]):
                raise PhaseBError("posterior bank/direct-evidence cross-binding drift")
    return bank


def _validate_posterior_statistics(value: Mapping[str, object], *, normalizer: Mapping[str, object]) -> None:
    expected = {
        "credibility_min", "credibility_max", "credibility_mean",
        "zero_spike_unit_rows_across_m4_m10_m30", "zero_raw_t4_standardized_not_clamped",
        "zero_standardized_rows_sha256", "posterior_normalizer_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PhaseBError("posterior credibility statistics schema drift")
    minimum = _finite_number(value["credibility_min"], name="posterior credibility min", positive=True)
    maximum = _finite_number(value["credibility_max"], name="posterior credibility max", positive=True)
    mean = _finite_number(value["credibility_mean"], name="posterior credibility mean", positive=True)
    if minimum > mean or mean > maximum or maximum > 1.0:
        raise PhaseBError("posterior credibility range drift")
    if (type(value["zero_spike_unit_rows_across_m4_m10_m30"]) is not int
            or value["zero_spike_unit_rows_across_m4_m10_m30"] < 0
            or value["zero_raw_t4_standardized_not_clamped"] is not True
            or value["posterior_normalizer_sha256"] != normalizer["body_sha256"]):
        raise PhaseBError("posterior credibility semantics/cross-binding drift")
    _sha(value["zero_standardized_rows_sha256"], "posterior zero standardized rows SHA")


def _validate_optimizer_and_execution_policy(optimizer: Mapping[str, object], policy: Mapping[str, object]) -> None:
    expected_optimizer = {
        "class": "Adam", "lr_constructor": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
        "weight_decay": 0.0, "amsgrad": False, "schedule": "arm_common.lr_at_step(48,33925)",
    }
    expected_policy = {"amp": False, "tf32": False, "torch_compile": False, "batch_size": 32}
    if dict(optimizer) != expected_optimizer:
        raise PhaseBError("posterior optimizer literal drift")
    if dict(policy) != expected_policy:
        raise PhaseBError("posterior execution-policy literal drift")


def validate_source_authority_for_identity(
    value: Mapping[str, object],
    *,
    identity: RunIdentity,
    launch_sha256: str,
) -> dict[str, object]:
    """Validate the physical source-only authority before it becomes durable.

    The launch identity binds the immutable strict-27 metadata available before
    source files are opened.  The later physical authority may add posterior
    tensors and normalizer moments, but it must never substitute a roster,
    ordinary-T4/behaviour authority, boundary, or code closure.  This check is
    deliberately performed before the source-authority receipt is published.
    """
    validate_run_identity(identity)
    if not isinstance(value, Mapping):
        raise PhaseBError("source authority must be a mapping")
    payload = dict(value)
    required = {
        "schema", "cell", "source_only", "target_opened", "within_opened", "external_opened",
        "formal_opened", "h1_opened", "roster", "roster_sha256", "strict_source_metadata_sha256",
        "manifest_sha256", "ordinary_raw_t4_semantic_sha256", "behavior_normalizer_semantic_sha256",
        "source_lineage_sha256", "source_data_root", "source_authority_metadata", "posterior_prior", "posterior_normalizer",
        "posterior_inputs", "posterior_bank", "source_raw_m30_t4_sha256", "posterior_preparation_seconds",
        "normalizer_phase_b_amendment", "m30_b3s_activity_prefix", "cache_read_or_write",
        "remote_torch_authority", "posterior_credibility_statistics", "optimizer", "execution_policy",
        "launch_sha256", "closure",
    }
    if set(payload) != required:
        raise PhaseBError("source authority schema/topology drift")
    expected_source = dict(identity.source_authority)
    expected_roster = list(_immutable_roster(expected_source["roster"], expected_count=SOURCE_SESSION_COUNT))
    if (
        payload["schema"] != "posterior_carrier_source_authority_v1"
        or payload["cell"] != CELL
        or payload["roster"] != expected_roster
        or payload["roster_sha256"] != expected_source["roster_sha256"]
        or payload["normalizer_phase_b_amendment"] != PHASE_B_NORMALIZER_AMENDMENT
        or payload["m30_b3s_activity_prefix"] != "held; carrier label budgets only vary M4/M10/M30"
        or payload["cache_read_or_write"] is not False
        or payload["remote_torch_authority"] != REMOTE_TORCH_AUTHORITY
        or payload["launch_sha256"] != _sha(launch_sha256, "source authority launch SHA")
    ):
        raise PhaseBError("source authority immutable binding drift")
    if any(payload[key] is not expected for key, expected in {
        "source_only": True,
        "target_opened": False,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "h1_opened": False,
    }.items()):
        raise PhaseBError("source authority target-surface boundary drift")
    for key in (
        "strict_source_metadata_sha256",
        "manifest_sha256",
        "ordinary_raw_t4_semantic_sha256",
        "behavior_normalizer_semantic_sha256",
        "source_lineage_sha256",
    ):
        if payload[key] != expected_source[key]:
            raise PhaseBError(f"source authority {key} drift")
    if payload["source_data_root"] != expected_source["source_data_root"]:
        raise PhaseBError("source authority external source-data root drift")
    validate_source_data_root_payload(payload["source_data_root"])
    if not isinstance(payload["source_authority_metadata"], Mapping):
        raise PhaseBError("source authority metadata payload drift")
    if sha256_bytes(canonical_json_bytes(dict(payload["source_authority_metadata"]))) != payload["strict_source_metadata_sha256"]:
        raise PhaseBError("source authority metadata digest drift")
    metadata = _validate_source_metadata_crosslinks(
        payload["source_authority_metadata"], roster=expected_roster, expected_source=expected_source,
    )
    for field in ("posterior_prior", "posterior_normalizer", "posterior_inputs", "posterior_bank",
                  "posterior_credibility_statistics", "optimizer", "execution_policy"):
        if not isinstance(payload[field], Mapping):
            raise PhaseBError(f"source authority {field} schema drift")
    source_raw = _sha(payload["source_raw_m30_t4_sha256"], "source authority raw M30 T4 SHA")
    prior = _validate_posterior_prior_payload(payload["posterior_prior"], roster=expected_roster, source_raw_sha256=source_raw)
    normalizer = _validate_posterior_normalizer_payload(payload["posterior_normalizer"], roster=expected_roster)
    inputs = _validate_posterior_inputs_payload(payload["posterior_inputs"], roster=expected_roster, metadata=metadata)
    _validate_posterior_bank_payload(
        payload["posterior_bank"], roster=expected_roster, normalizer=normalizer, inputs=inputs, prior=prior,
    )
    _validate_posterior_statistics(payload["posterior_credibility_statistics"], normalizer=normalizer)
    _validate_optimizer_and_execution_policy(payload["optimizer"], payload["execution_policy"])
    if (not isinstance(payload["posterior_preparation_seconds"], (float, int))
            or not math.isfinite(float(payload["posterior_preparation_seconds"]))
            or float(payload["posterior_preparation_seconds"]) < 0):
        raise PhaseBError("source authority posterior preparation duration drift")
    closure = validate_phase_b_closure(payload["closure"])
    if closure != validate_phase_b_closure(identity.closure):
        raise PhaseBError("source authority closure drift")
    return payload


class _CapabilitySeal:
    pass


_CAPABILITY_SEAL = _CapabilitySeal()


@dataclass(frozen=True)
class RootReviewedCapability:
    route: str
    _seal: object
    closure_sha256: str

    def validate(self, *, closure: Mapping[str, object]) -> None:
        if self.route != CELL or self._seal is not _CAPABILITY_SEAL:
            raise PhaseBError("source-smoke execution requires an in-process root-reviewed capability")
        if self.closure_sha256 != validate_phase_b_closure(closure)["closure_sha256"]:
            raise PhaseBError("root-reviewed capability closure drift")


def _issue_root_review_capability_for_audited_route(*, closure: Mapping[str, object]) -> RootReviewedCapability:
    """Internal root-only capability factory; the CLI has no path to this object."""
    return RootReviewedCapability(route=CELL, _seal=_CAPABILITY_SEAL,
                                  closure_sha256=str(validate_phase_b_closure(closure)["closure_sha256"]))


def _directory_identity(path: Path) -> tuple[int, int]:
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PhaseBError("artifact directory must be a canonical non-symlink directory")
    return info.st_dev, info.st_ino


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _write_all(fd: int, body: bytes) -> None:
    view = memoryview(body)
    while view:
        wrote = os.write(fd, view)
        if wrote <= 0:
            raise OSError("short immutable artifact write")
        view = view[wrote:]


@dataclass(frozen=True)
class ArtifactRoot:
    """O_EXCL/fsync/0444 route-owned artifact capability used by smoke tests."""

    directory: Path
    topology: tuple[str, ...]
    identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]

    def _assert_identity(self) -> None:
        if _directory_identity(self.directory) != self.identity:
            raise PhaseBError("artifact root identity drift")
        pfd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(pfd)
            if (info.st_dev, info.st_ino) != self.parent_identity:
                raise PhaseBError("artifact parent identity drift")
            named = os.stat(self.directory.name, dir_fd=pfd, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != self.identity or not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode):
                raise PhaseBError("artifact named root identity drift")
        finally:
            os.close(pfd)

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or name not in self.topology or "/" in name:
            raise PhaseBError("artifact name lies outside frozen topology")

    def publish_json(self, name: str, value: Mapping[str, object]) -> str:
        self._check_name(name)
        body = canonical_json_bytes(value)
        digest = sha256_bytes(body)
        self._assert_identity()
        dfd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        made: list[tuple[str, int, int]] = []
        try:
            for leaf, content in ((name, body), (f"{name}.sha256", f"{digest}  {name}\n".encode("ascii"))):
                fd = os.open(leaf, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=dfd)
                try:
                    info = os.fstat(fd)
                    made.append((leaf, info.st_dev, info.st_ino))
                    _write_all(fd, content)
                    os.fchmod(fd, 0o444)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(dfd)
            self.reload_json(name, digest)
            self._assert_identity()
            return digest
        except BaseException:
            for leaf, device, inode in reversed(made):
                try:
                    info = os.stat(leaf, dir_fd=dfd, follow_symlinks=False)
                    if (info.st_dev, info.st_ino) == (device, inode):
                        os.unlink(leaf, dir_fd=dfd)
                except OSError:
                    pass
            raise
        finally:
            os.close(dfd)

    def reload_json(self, name: str, expected_sha256: str | None = None) -> Mapping[str, object]:
        self._check_name(name)
        self._assert_identity()
        dfd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            def read(leaf: str) -> bytes:
                fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                        raise PhaseBError("immutable artifact type/mode drift")
                    return _read_all(fd)
                finally:
                    os.close(fd)
            body = read(name)
            digest = sha256_bytes(body)
            if expected_sha256 is not None and digest != expected_sha256:
                raise PhaseBError("immutable artifact body SHA drift")
            if read(f"{name}.sha256") != f"{digest}  {name}\n".encode("ascii"):
                raise PhaseBError("immutable artifact sidecar drift")
            value = json.loads(body)
            if not isinstance(value, Mapping):
                raise PhaseBError("immutable artifact JSON root drift")
            return value
        finally:
            os.close(dfd)


SMOKE_TOPOLOGY = (
    "attempt.json", "launch.json", "source_authority.json", "step100.json", "terminal.json", "failure.json",
)


def reserve_source_smoke_root(root: Path, *, relative: str = SOURCE_SMOKE_ROOT_RELATIVE) -> ArtifactRoot:
    """Reserve a fresh root only after a reviewed caller has crossed the gate."""
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise PhaseBError("source-smoke root must be a safe relative path")
    target = Path(root).absolute() / relative
    if target.exists() or target.is_symlink():
        raise PhaseBError("posterior source-smoke root must be fresh")
    parent = target.parent
    parent_identity = _directory_identity(parent)
    pfd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.mkdir(target.name, 0o755, dir_fd=pfd)
        os.fsync(pfd)
        info = os.stat(target.name, dir_fd=pfd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise PhaseBError("source-smoke root construction drift")
    except FileExistsError as error:
        raise PhaseBError("posterior source-smoke root collision") from error
    finally:
        os.close(pfd)
    return ArtifactRoot(target, SMOKE_TOPOLOGY, (info.st_dev, info.st_ino), parent, parent_identity)


@dataclass(frozen=True)
class SmokeStepSummary:
    loss_first: float
    loss_last: float
    loss_min: float
    loss_max: float
    nonincreasing_transitions: int
    losses_count: int
    critical_gradients: Mapping[str, bool]
    finite_model: bool
    finite_adam: bool
    model_state_sha256: str
    optimizer_state_sha256: str
    posterior_prepare_seconds: float
    optimizer_core_seconds: float
    optimizer_steps_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    current_allocated_bytes: int
    current_reserved_bytes: int
    posterior_cache: Mapping[str, int]
    dropout: Mapping[str, object]
    remote_device: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        values = {
            "loss_first": self.loss_first, "loss_last": self.loss_last, "loss_min": self.loss_min,
            "loss_max": self.loss_max, "nonincreasing_transitions": self.nonincreasing_transitions,
            "losses_count": self.losses_count, "critical_gradients": dict(self.critical_gradients),
            "finite_model": self.finite_model, "finite_adam": self.finite_adam,
            "model_state_sha256": self.model_state_sha256, "optimizer_state_sha256": self.optimizer_state_sha256,
            "posterior_prepare_seconds": self.posterior_prepare_seconds,
            "optimizer_core_seconds": self.optimizer_core_seconds,
            "optimizer_steps_per_second": self.optimizer_steps_per_second,
            "peak_allocated_bytes": self.peak_allocated_bytes, "peak_reserved_bytes": self.peak_reserved_bytes,
            "current_allocated_bytes": self.current_allocated_bytes, "current_reserved_bytes": self.current_reserved_bytes,
            "posterior_cache": dict(self.posterior_cache), "dropout": dict(self.dropout),
            "remote_device": dict(self.remote_device),
        }
        if (self.losses_count != SOURCE_SMOKE_STEPS or not all(isinstance(values[key], (float, int)) and math.isfinite(float(values[key]))
                for key in ("loss_first", "loss_last", "loss_min", "loss_max", "posterior_prepare_seconds",
                            "optimizer_core_seconds", "optimizer_steps_per_second"))
                or self.loss_min < 0 or self.loss_max < self.loss_min or self.loss_first < 0 or self.loss_last < 0
                or type(self.nonincreasing_transitions) is not int or not 0 <= self.nonincreasing_transitions < SOURCE_SMOKE_STEPS
                or set(self.critical_gradients) != {"b3s_pre_pool_activity", "b3s_post_pool_activity", "b3s_post_pool_t4",
                                                     "decoder_fc_in", "cross_attention", "ffn", "query_rep", "output_fc"}
                or not all(self.critical_gradients.values()) or self.finite_model is not True or self.finite_adam is not True
                or self.peak_allocated_bytes < self.current_allocated_bytes or self.peak_reserved_bytes < self.current_reserved_bytes
                or self.posterior_prepare_seconds < 0 or self.optimizer_core_seconds <= 0 or self.optimizer_steps_per_second <= 0):
            raise PhaseBError("source-smoke summary contract drift")
        _sha(self.model_state_sha256, "source-smoke model SHA")
        _sha(self.optimizer_state_sha256, "source-smoke optimizer SHA")
        if dict(self.remote_device) != REMOTE_TORCH_AUTHORITY:
            raise PhaseBError("source-smoke remote device drift")
        expected_cache = {
            "posterior_fit_calls": SOURCE_SESSION_COUNT * len(BUDGETS),
            "posterior_inverse_calls": SOURCE_SESSION_COUNT * len(BUDGETS),
            # The physical authority computes deterministic M4/M10/M30
            # credibility statistics before the step loop, then one sampled
            # and one device-copied carrier for each M(e=0,j).
            "deterministic_mean_view_builds": SOURCE_SESSION_COUNT * len(BUDGETS),
            "epoch_sampled_view_builds": SOURCE_SESSION_COUNT,
            "device_epoch_view_builds": SOURCE_SESSION_COUNT,
            "normalized_view_builds": SOURCE_SESSION_COUNT * (len(BUDGETS) + 1),
            "batch_loop_requests": SOURCE_SMOKE_STEPS,
            "batch_loop_inverse_calls": 0,
            "source_sessions": SOURCE_SESSION_COUNT,
            "scheduled_session_epochs": SOURCE_SESSION_COUNT,
        }
        if dict(self.posterior_cache) != expected_cache:
            raise PhaseBError("source-smoke posterior cache accounting drift")
        expected_dropout = {
            "dynamic_dropout": True,
            "low": 0.0,
            "high": 1.0,
            "semantics": "complete_fused_unit_token_placeholder_with_inverse_probability_gain",
            "extra_dropout_draws": 0,
        }
        if dict(self.dropout) != expected_dropout:
            raise PhaseBError("source-smoke Cell-D dropout law drift")
        if any(type(values[key]) is not int or values[key] < 0 for key in (
            "peak_allocated_bytes", "peak_reserved_bytes", "current_allocated_bytes", "current_reserved_bytes",
        )):
            raise PhaseBError("source-smoke memory accounting schema drift")
        return values


class SourceSmokeBackend(Protocol):
    """No-data mock / deferred physical source-only execution seam."""

    def prepare(self, spec: SourceSmokeSpec, identity: RunIdentity) -> Any: ...
    def source_authority(self, runtime: Any, identity: RunIdentity) -> Mapping[str, object]: ...
    def run_steps(self, runtime: Any, spec: SourceSmokeSpec) -> SmokeStepSummary: ...
    def close(self, runtime: Any | None) -> None: ...


def _attempt_payload(identity: RunIdentity) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_source_smoke_attempt_v1",
        "cell": CELL,
        "phase": PHASE_B,
        "spec": SMOKE_SPEC.payload(),
        "identity": identity.payload(),
        "boundaries": identity.payload()["boundaries"],
        "status": "ATTEMPT_STARTED_SOURCE_ONLY",
    }


def _launch_payload(identity: RunIdentity, attempt_sha256: str) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_source_smoke_launch_v1",
        "cell": CELL,
        "spec": SMOKE_SPEC.payload(),
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "attempt SHA"),
        "launch_closure": validate_phase_b_closure(identity.closure),
        "status": "SOURCE_SMOKE_LAUNCHED",
    }


def _failure_payload(
    *,
    identity: RunIdentity,
    attempt_sha256: str,
    launch_sha256: str | None,
    stage: str,
    error: BaseException,
    progress: SmokeExecutionProgress,
) -> dict[str, object]:
    if stage not in {"prepare", "source_authority", "steps", "terminal"}:
        raise PhaseBError("failure stage drift")
    if not isinstance(progress, SmokeExecutionProgress):
        raise PhaseBError("failure progress type drift")
    if launch_sha256 is not None:
        _sha(launch_sha256, "failure launch SHA")
    return {
        "schema": "posterior_carrier_source_smoke_failure_v1",
        "cell": CELL,
        "phase": PHASE_B,
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "failure attempt SHA"),
        "launch_sha256": launch_sha256,
        "source_authority_sha256": progress.source_authority_sha256,
        "stage": stage,
        "error_class": type(error).__name__,
        "error_sha256": sha256_bytes(repr(error).encode("utf-8")),
        "source_opened": progress.source_opened,
        "remote_initialized": progress.remote_initialized,
        "optimizer_steps_completed": progress.optimizer_steps_completed,
        "boundaries": identity.payload()["boundaries"],
        "terminal_published": False,
        "status": "SOURCE_SMOKE_FAILED_HONESTLY",
    }


def validate_failure_payload(
    value: Mapping[str, object],
    *,
    identity: RunIdentity,
    attempt_sha256: str,
    launch_sha256: str | None,
    progress: SmokeExecutionProgress,
) -> dict[str, object]:
    """Exact failure validator: no silent default progress fields are legal."""
    validate_run_identity(identity)
    expected_keys = {
        "schema", "cell", "phase", "identity", "attempt_sha256", "launch_sha256", "source_authority_sha256",
        "stage", "error_class", "error_sha256", "source_opened", "remote_initialized",
        "optimizer_steps_completed", "boundaries", "terminal_published", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise PhaseBError("source-smoke failure schema drift")
    payload = dict(value)
    if (payload["schema"] != "posterior_carrier_source_smoke_failure_v1" or payload["cell"] != CELL
            or payload["phase"] != PHASE_B or payload["identity"] != identity.payload()
            or payload["attempt_sha256"] != _sha(attempt_sha256, "failure validator attempt SHA")
            or payload["launch_sha256"] != launch_sha256
            or payload["source_authority_sha256"] != progress.source_authority_sha256
            or payload["stage"] not in {"prepare", "source_authority", "steps", "terminal"}
            or not isinstance(payload["error_class"], str) or not payload["error_class"]
            or not isinstance(payload["error_sha256"], str) or len(payload["error_sha256"]) != 64
            or payload["source_opened"] is not progress.source_opened
            or payload["remote_initialized"] is not progress.remote_initialized
            or payload["optimizer_steps_completed"] != progress.optimizer_steps_completed
            or payload["boundaries"] != identity.payload()["boundaries"]
            or payload["terminal_published"] is not False
            or payload["status"] != "SOURCE_SMOKE_FAILED_HONESTLY"):
        raise PhaseBError("source-smoke failure binding drift")
    if launch_sha256 is not None:
        _sha(launch_sha256, "failure validator launch SHA")
    if progress.source_authority_sha256 is not None:
        _sha(progress.source_authority_sha256, "failure validator source-authority SHA")
    return payload


def run_source_smoke_lifecycle(
    *,
    backend: SourceSmokeBackend,
    artifact: ArtifactRoot,
    identity: RunIdentity,
    spec: SourceSmokeSpec = SMOKE_SPEC,
    stage_root: Path | None = None,
) -> Mapping[str, object]:
    """Run only the reviewed 100-step source-only lifecycle through an injected backend."""
    validate_run_identity(identity)
    if spec != SMOKE_SPEC:
        raise PhaseBError("public posterior source smoke spec drift")
    runtime: Any | None = None
    stage = "prepare"
    progress = SmokeExecutionProgress()
    launch_sha: str | None = None
    terminal_published = False
    if stage_root is not None:
        live_before_attempt = phase_b_closure(Path(stage_root).absolute())
        if live_before_attempt != validate_phase_b_closure(identity.closure):
            raise PhaseBError("identity/live Phase-B closure drift before attempt")
    attempt_sha = artifact.publish_json("attempt.json", _attempt_payload(identity))
    try:
        launch = _launch_payload(identity, attempt_sha)
        launch_sha = artifact.publish_json("launch.json", launch)
        stage = "prepare"
        runtime = backend.prepare(spec, identity)
        progress = progress.merge(_runtime_progress(runtime))
        stage = "source_authority"
        source_authority = dict(backend.source_authority(runtime, identity))
        source_authority["launch_sha256"] = launch_sha
        source_authority["closure"] = validate_phase_b_closure(identity.closure)
        source_authority = validate_source_authority_for_identity(
            source_authority,
            identity=identity,
            launch_sha256=launch_sha,
        )
        source_authority_sha = artifact.publish_json("source_authority.json", source_authority)
        progress = progress.merge(SmokeExecutionProgress(
            source_opened=progress.source_opened,
            remote_initialized=progress.remote_initialized,
            optimizer_steps_completed=progress.optimizer_steps_completed,
            source_authority_sha256=source_authority_sha,
        ))
        stage = "steps"
        summary = backend.run_steps(runtime, spec)
        step = {
            "schema": "posterior_carrier_source_smoke_step100_v1", "cell": CELL,
            "spec": spec.payload(), "identity": identity.payload(), "launch_sha256": launch_sha,
            "source_authority_sha256": source_authority_sha, "summary": summary.payload(),
            "boundaries": identity.payload()["boundaries"], "status": "SOURCE_SMOKE_100_STEPS_COMPLETE",
        }
        step_sha = artifact.publish_json("step100.json", step)
        progress = progress.merge(SmokeExecutionProgress(
            source_opened=progress.source_opened,
            remote_initialized=progress.remote_initialized,
            optimizer_steps_completed=SOURCE_SMOKE_STEPS,
            source_authority_sha256=source_authority_sha,
        ))
        stage = "terminal"
        final_closure = (
            phase_b_closure(Path(stage_root).absolute())
            if stage_root is not None else validate_phase_b_closure(identity.closure)
        )
        if final_closure != launch["launch_closure"]:
            raise PhaseBError("launch/final Phase-B closure drift")
        terminal = {
            "schema": "posterior_carrier_source_smoke_terminal_v1", "cell": CELL,
            "phase": PHASE_B, "spec": spec.payload(), "identity": identity.payload(),
            "attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
            "source_authority_sha256": source_authority_sha, "step100_sha256": step_sha,
            "launch_closure": launch["launch_closure"], "final_closure": final_closure,
            "boundaries": identity.payload()["boundaries"],
            "status": "SOURCE_SMOKE_COMPLETE__NON_AUTHORITATIVE",
        }
        terminal_sha = artifact.publish_json("terminal.json", terminal)
        terminal_published = True
        if artifact.reload_json("terminal.json", terminal_sha) != terminal:
            raise PhaseBError("source-smoke terminal reload drift")
        return terminal
    except BaseException as error:
        if isinstance(error, SourceSmokeExecutionError):
            progress = progress.merge(error.progress)
            stage = error.stage
            error_for_receipt: BaseException = error.cause
        else:
            progress = progress.merge(_runtime_progress(runtime))
            error_for_receipt = error
        # Once the immutable terminal pair exists, it is the durable outcome;
        # never append a contradictory failure receipt.  `publish_json` rolls
        # back its own pair on an internal post-write validation failure, so a
        # false terminal cannot be hidden behind this branch.
        if not terminal_published:
            try:
                failure = _failure_payload(
                    identity=identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha, stage=stage,
                    error=error_for_receipt, progress=progress,
                )
                failure = validate_failure_payload(
                    failure, identity=identity, attempt_sha256=attempt_sha,
                    launch_sha256=launch_sha, progress=progress,
                )
                if launch_sha is not None:
                    artifact.reload_json("launch.json", launch_sha)
                if progress.source_authority_sha256 is not None:
                    artifact.reload_json("source_authority.json", progress.source_authority_sha256)
                failure_sha = artifact.publish_json("failure.json", failure)
                if artifact.reload_json("failure.json", failure_sha) != failure:
                    raise PhaseBError("source-smoke failure reload drift")
            except BaseException:
                pass
        raise
    finally:
        backend.close(runtime)


def reviewed_remote_source_smoke(
    *,
    root: Path,
    capability: RootReviewedCapability,
    backend: SourceSmokeBackend,
    identity: RunIdentity,
    source_data: SourceDataRootCapability,
) -> Mapping[str, object]:
    """Execution-only route; unavailable to public CLI flags without capability."""
    stage_root = Path(root).absolute()
    if source_data.payload() != identity.source_authority.get("source_data_root"):
        raise PhaseBError("reviewed route/source-data capability drift")
    validate_stage_source_separation(stage_root=stage_root, source_data=source_data)
    live_closure = phase_b_closure(stage_root)
    if live_closure != validate_phase_b_closure(identity.closure):
        raise PhaseBError("identity/live Phase-B closure drift before output reservation")
    capability.validate(closure=live_closure)
    artifact = reserve_source_smoke_root(stage_root)
    return run_source_smoke_lifecycle(
        backend=backend, artifact=artifact, identity=identity, stage_root=stage_root,
    )


def remote_staging_plan(*, closure: Mapping[str, object], source_authority_sha256: str,
                        source_data: SourceDataRootCapability,
                        remote_root_name: str = "posterior_carrier_budgetmix_d_seed42_stage_v1") -> dict[str, object]:
    """Pure remote staging manifest; it never opens a socket or invokes ssh/tmux."""
    if not isinstance(remote_root_name, str) or not remote_root_name or "/" in remote_root_name or ".." in remote_root_name:
        raise PhaseBError("remote staging root name must be one safe directory component")
    stable_closure = validate_phase_b_closure(closure)
    for relative, expected_sha, _mode in SOURCE_AUTHORITY_ASSET_SPECS:
        if stable_closure["sha256_by_path"].get(relative) != expected_sha:
            raise PhaseBError("staging closure authority-asset drift")
    assets: list[dict[str, object]] = []
    for relative, expected_sha, expected_mode in SOURCE_AUTHORITY_ASSET_SPECS:
        assets.append({"relative_path": relative, "sha256": expected_sha, "mode": expected_mode})
        if expected_mode == 0o444:
            assets.append({
                "relative_path": f"{relative}.sha256",
                "contents": f"{expected_sha}  {Path(relative).name}\\n",
                "mode": 0o444,
            })
    # This is the exact non-glob staging file list.  The authority subset is
    # repeated above because it has sidecars/mode rules; all executable and
    # package-init dependencies are listed here so a fresh stage never falls
    # back to an historical workspace checkout through Python import paths.
    authority_mode_by_path = {
        relative: mode for relative, _expected_sha, mode in SOURCE_AUTHORITY_ASSET_SPECS
    }
    stage_files = [
        {
            "relative_path": relative,
            "sha256": stable_closure["sha256_by_path"][relative],
            "role": "immutable_authority" if relative in SOURCE_AUTHORITY_ASSET_PATHS else "closure_dependency",
            **({"mode": authority_mode_by_path[relative]}
               if authority_mode_by_path.get(relative) is not None else {}),
        }
        for relative in PHASE_B_CLOSURE_PATHS
    ]
    # Sidecars are not Python closure inputs, but the physical same-FD
    # authority parser requires them.  Make them first-class members of the
    # staging manifest rather than an undocumented transfer convention.
    for relative, expected_sha, expected_mode in SOURCE_AUTHORITY_ASSET_SPECS:
        if expected_mode == 0o444:
            stage_files.append({
                "relative_path": f"{relative}.sha256",
                "contents": f"{expected_sha}  {Path(relative).name}\n",
                "mode": 0o444,
                "role": "immutable_authority_sidecar",
            })
    return {
        "schema": "posterior_carrier_remote_staging_plan_v1",
        "cell": CELL,
        "phase": PHASE_B,
        "remote_host": "xinyuan@100.103.97.12",
        "remote_root_name": remote_root_name,
        "closure": stable_closure,
        "source_authority_sha256": _sha(source_authority_sha256, "remote staging source authority SHA"),
        "stage_files": stage_files,
        "immutable_authority_assets": assets,
        "external_source_data_root": validate_source_data_root_payload(source_data.payload()),
        "nwb_assets_in_stage": False,
        "remote_torch_authority": dict(REMOTE_TORCH_AUTHORITY),
        "forbidden": ["target", "within", "external", "formal", "h1", "teacher", "remote_historical_repo_mutation"],
        "launch": "NOT_AUTHORIZED_BY_PLAN; requires a fresh in-process root-reviewed capability",
    }
