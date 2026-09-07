"""Fail-closed, source-only preparation for ``CELL_D_EQUAL_SESSION_SEED42``.

This module intentionally imports only the Python standard library at module
load.  It is an additive training-method route: it preserves the sealed
Cell-D graph and changes only the source-session batch exposure schedule.  A
future root-reviewed execution backend may use the explicit epoch-bound
sampler defined here, but this module neither constructs a model nor opens a
dataset unless its source-only audit entrypoint is explicitly called.

The public schedule is deliberately not a tunable sampler:

* 27 strict-source sessions, B32, 33,925 batches/epoch, 48 epochs;
* every epoch gives each session 1,256 batches plus the frozen 13 extras;
* extras are exactly ``(13 * epoch + j) mod 27`` for ``j=0..12``;
* all schedule randomness is local, SHA-256-derived, and domain-separated.

The result is a reproducible source-only plan that can be reviewed before a
separate GPU smoke/full-run authorization exists.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import stat
import struct
import sys
import time
import traceback
from array import array
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence


CELL = "CELL_D_EQUAL_SESSION_SEED42"
SCHEMA = "cell_d_equal_session_v1"
HANDOFF_RELATIVE = "tfpd_exploration/docs/HANDOFF_DEPLOYMENT_MATCHED_SESSION_BALANCING_20260820.md"
HANDOFF_SHA256 = "d1a6692b50b48028c5d8e0934d72dd0498ab80a457d28d8528b7a116363128d6"
MANIFEST_RELATIVE = "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
SOURCE_DATA_RELATIVE = "sua_exploration/data/dandi_000688/sub-C"
SEALED_CELL_D_LAUNCH_RELATIVE = (
    "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/launch_receipt.json"
)
SEALED_CELL_D_TERMINAL_RELATIVE = (
    "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json"
)
SEALED_ARM_A_PREFLIGHT_RELATIVE = "tfpd_exploration/results/admission_arms_v1/preflight_armA.json"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cell_d_equal_session_seed42_v1"
# v1 is an immutable, honest pre-CUDA failure.  Its canonical successor must
# never reuse that directory, even though the scientific smoke contract is
# unchanged.
FAILED_SMOKE_V1_ROOT_RELATIVE = "tfpd_exploration/results/cell_d_equal_session_seed42_smoke_v1"
SMOKE_RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cell_d_equal_session_seed42_smoke_v2"
FAILED_SMOKE_V1_ATTEMPT_RELATIVE = f"{FAILED_SMOKE_V1_ROOT_RELATIVE}/attempt.json"
FAILED_SMOKE_V1_LAUNCH_RELATIVE = f"{FAILED_SMOKE_V1_ROOT_RELATIVE}/launch.json"
FAILED_SMOKE_V1_FAILURE_RELATIVE = f"{FAILED_SMOKE_V1_ROOT_RELATIVE}/failure.json"
FAILED_SMOKE_V1_ATTEMPT_SHA256 = "6767fa7a926e138cc2e3e672a801e850cbeb8b7ce6748bdf8297178d87ec10b9"
FAILED_SMOKE_V1_LAUNCH_SHA256 = "39c67e8929638b25af9daaaa998ca0718238d0975a970bd81c2fa59e667a2dde"
FAILED_SMOKE_V1_FAILURE_SHA256 = "1c3fc725d6da094421a6224bcafd6741c00ae1b4b2de1513cd2909c4af890bb9"
FAILED_SMOKE_V1_CLOSURE_SHA256 = "d147dad1e0534bfcdbc61cc371598dc59ef93a66eba135b4eb6041a6f49a9f54"
CANONICAL_INITIAL_STATE_RELATIVE = "tfpd_exploration/results/admission_arms_v1/canonical_initial_state.pt"
CANONICAL_INITIAL_STATE_SHA256 = "b0a340fe4d09eac1f2b498658d39a8a87e87f52304cad2539753ae9e040fcbd4"
CANONICAL_INITIAL_STATE_STATE_SHA256 = "65bacb85447df40ea5e03cffed03b1763d1964bfba50cbbe3d21d1614c07f2a3"
CANONICAL_INITIAL_STATE_TORCH_VERSION = "2.5.1.post303"
SEALED_BEHAVIOR_NORMALIZER_SHA256 = "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
SEALED_LAUNCH_RECEIPT_SHA256 = "a3d071fa5b7ad4f356f784bf48eb7d7c1f3e1c51b9e0885bad5f1d57f67381c5"
SEALED_TERMINAL_RECEIPT_SHA256 = "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442"
SEALED_ARM_A_PREFLIGHT_SHA256 = "2632c6a6a4cfb8a4c0fb2b23e0cc8205ea323240b376903c60d5b27110f59e43"
SEALED_T4_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
PREDECESSOR_STARTED_UTC = "2026-08-17T17:24:26Z"
PREDECESSOR_FINISHED_UTC = "2026-08-17T22:32:58Z"
PREDECESSOR_WALL_CLOCK_SECONDS = 18_512
SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS = 1_086_007
# The predecessor's 33,925 B32 batches consume this many source windows;
# despite legacy receipt wording, this is a *window* count, not a batch count.
SOURCE_ONLY_EXPECTED_PREDECESSOR_CONSUMED_WINDOWS = 1_085_600
SOURCE_ONLY_EXPECTED_REMAINDER_WINDOWS = 407
AUDITED_FULL_PLAN_SHA256 = "17c1ee6d1ddc62e50aa62c5e25ff800a1612631b4aa36f7508805ce6f83fc98b"
AUDITED_EPOCH0_SCHEDULE_SHA256 = "3ae2e3e07c96d14df20236cee45c5da628186371bd031c2df6fbd3230c0e2bef"
AUDITED_EPOCH47_SCHEDULE_SHA256 = "905fb8888ce63e73e617b8b9d8ab0c32fe3026e5e9663282784fd814d44f7b19"
CELL_D_INITIALIZED_TRAINABLE_PARAMETERS = 3_510_842
CELL_D_UNINITIALIZED_LAZY_KEYS = (
    "decoder.fc_id_in.0.bias",
    "decoder.fc_id_in.0.weight",
)

# This is deliberately a named, exact live-path contract rather than a broad
# "some encoder" / "some decoder" predicate.  The B3S mean and side entries
# are distinct slices of the same first post-pool affine tensor; both must
# receive finite, non-zero gradients at a source-smoke or full-epoch boundary.
CRITICAL_GRADIENT_PATHS: Mapping[str, Mapping[str, object]] = {
    "b3s_pre_pool_activity": {
        "parameter": "id_encoder.pre_pool.0.weight", "selector": "all",
    },
    "b3s_post_pool_mean": {
        "parameter": "id_encoder.post_pool.0.weight", "selector": "columns[0:64]",
    },
    "b3s_post_pool_t4_side": {
        "parameter": "id_encoder.post_pool.0.weight", "selector": "columns[64:68]",
    },
    "decoder_fc_in": {
        "parameter": "decoder.fc_in.0.weight", "selector": "all",
    },
    "decoder_cross_attention": {
        "parameter": "decoder.transformer.layers.0.cross_attn.in_proj_weight", "selector": "all",
    },
    "decoder_ffn": {
        "parameter": "decoder.transformer.layers.0.ffn.0.weight", "selector": "all",
    },
    "decoder_query_rep": {
        "parameter": "decoder.rep", "selector": "all",
    },
    "decoder_fc_out": {
        "parameter": "decoder.fc_out.weight", "selector": "all",
    },
}

# These are deliberately two separate physical authorities.  The nominal
# nvidia-smi capacity is not a rounded rendering of Torch's byte value.
FROZEN_GPU0: Mapping[str, object] = {
    "cuda_visible_devices": "0",
    "cuda_device_order": "PCI_BUS_ID",
    "logical_device": "cuda:0",
    "uuid": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
    "bdf": "00000000:01:00.0",
    "name": "NVIDIA GeForce RTX 3090",
    "nvidia_smi_memory_total_mib": 24_576,
    "torch_total_memory_bytes": 25_435_111_424,
}
# These version authorities were sealed by the immutable GPU0 throughput
# receipt.  A reviewed physical smoke must bind this exact runtime, not merely
# report arbitrary nonempty version strings after it has begun preparation.
FROZEN_RUNTIME_VERSIONS: Mapping[str, object] = {
    "torch_version": CANONICAL_INITIAL_STATE_TORCH_VERSION,
    "torch_cuda_version": "11.8",
    "cudnn_version": 90_300,
}
RUNTIME_ENVIRONMENT_SCHEMA = "cell_d_equal_session_runtime_environment_v1"
SYNTHETIC_RUNTIME_ENVIRONMENT_SCHEMA = "cell_d_equal_session_synthetic_runtime_environment_v1"

IMPLEMENTATION_CLOSURE = (
    HANDOFF_RELATIVE,
    "tfpd_exploration/src/cell_d_equal_session_v1.py",
    "tfpd_exploration/scripts/run_cell_d_equal_session_seed42.py",
    "tfpd_exploration/tests/test_cell_d_equal_session_v1.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/datamodule.py",
    MANIFEST_RELATIVE,
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
)


class ContractError(RuntimeError):
    """A fail-closed route/provenance/schedule violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative(relative: str) -> None:
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts, f"unsafe relative path: {relative!r}")


def _regular_bytes(root: Path, relative: str, *, mode: int) -> bytes:
    """Read one fixed repository file without accepting aliases or symlinks."""
    _safe_relative(relative)
    root = Path(root).absolute()
    candidate = root / relative
    try:
        before = os.lstat(candidate)
    except OSError as error:
        raise ContractError(f"missing required file: {relative}") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ContractError(f"required file is not a regular non-symlink: {relative}")
    if stat.S_IMODE(before.st_mode) != mode:
        raise ContractError(f"required file mode drift: {relative}")
    descriptor = None
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != mode
                or (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size)):
            raise ContractError(f"required file identity drift before read: {relative}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        body = b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    after = os.lstat(candidate)
    if (not stat.S_ISREG(after.st_mode) or stat.S_ISLNK(after.st_mode)
            or stat.S_IMODE(after.st_mode) != mode
            or (after.st_dev, after.st_ino, after.st_size) != (before.st_dev, before.st_ino, before.st_size)
            or len(body) != before.st_size):
        raise ContractError(f"required file changed during read: {relative}")
    return body


def implementation_closure(root: Path) -> dict[str, object]:
    """Return an explicit, non-glob closure for a future reviewed launch."""
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_CLOSURE:
        hashes[relative] = _sha256(_regular_bytes(root, relative, mode=0o664))
    if hashes.get(HANDOFF_RELATIVE) != HANDOFF_SHA256:
        raise ContractError("equal-session handoff SHA drift")
    body = _canonical_json_bytes({"sha256_by_path": hashes})
    return {
        "schema": "cell_d_equal_session_implementation_closure_v1",
        "paths": list(IMPLEMENTATION_CLOSURE),
        "sha256_by_path": hashes,
        "closure_sha256": _sha256(body),
    }


@dataclass(frozen=True)
class ScheduleSpec:
    """Frozen public batch arithmetic, with generic small fixtures allowed in tests."""

    cell: str
    seed: int
    roster_size: int
    batch_size: int
    base_batches_per_session: int
    extras_per_epoch: int
    epochs: int

    def __post_init__(self) -> None:
        _require(isinstance(self.cell, str) and self.cell, "schedule cell must be a nonempty string")
        _require(type(self.seed) is int and self.seed >= 0, "schedule seed must be a nonnegative integer")
        _require(type(self.roster_size) is int and self.roster_size >= 2, "schedule roster size drift")
        _require(type(self.batch_size) is int and self.batch_size > 0, "schedule batch size drift")
        _require(type(self.base_batches_per_session) is int and self.base_batches_per_session > 0,
                 "schedule base batch quota drift")
        _require(type(self.extras_per_epoch) is int and 0 < self.extras_per_epoch < self.roster_size,
                 "schedule extra quota drift")
        _require(type(self.epochs) is int and self.epochs > 0, "schedule epoch count drift")

    @property
    def steps_per_epoch(self) -> int:
        return self.roster_size * self.base_batches_per_session + self.extras_per_epoch

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    def payload(self) -> dict[str, object]:
        return {
            "cell": self.cell,
            "seed": self.seed,
            "roster_size": self.roster_size,
            "batch_size": self.batch_size,
            "base_batches_per_session": self.base_batches_per_session,
            "extras_per_epoch": self.extras_per_epoch,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
        }


PUBLIC_SPEC = ScheduleSpec(
    cell=CELL,
    seed=42,
    roster_size=27,
    batch_size=32,
    base_batches_per_session=1_256,
    extras_per_epoch=13,
    epochs=48,
)


def require_public_spec(spec: ScheduleSpec) -> None:
    if spec.payload() != PUBLIC_SPEC.payload():
        raise ContractError("candidate schedule spec differs from frozen public Cell-D equal-session spec")


def _seed_payload(spec: ScheduleSpec, *, domain: str, epoch: int, session: str | None = None,
                  cycle: int | None = None) -> bytes:
    _require(type(epoch) is int and 0 <= epoch < spec.epochs, "epoch outside frozen schedule")
    if domain == "global-order":
        _require(session is None and cycle is None, "global-order seed unexpectedly has session/cycle")
        rendered = f"{spec.cell}|global-order|{spec.seed}|{epoch}"
    elif domain == "session-cycle":
        _require(isinstance(session, str) and session, "session-cycle seed lacks session")
        _require(type(cycle) is int and cycle >= 0, "session-cycle seed lacks nonnegative cycle")
        rendered = f"{spec.cell}|session-cycle|{spec.seed}|{epoch}|{session}|{cycle}"
    else:
        raise ContractError(f"unsupported local-RNG domain: {domain!r}")
    return rendered.encode("ascii")


def local_rng_seed(spec: ScheduleSpec, *, domain: str, epoch: int, session: str | None = None,
                   cycle: int | None = None) -> int:
    """Interpret the full lowercase SHA-256 hexadecimal digest as unsigned BE."""
    digest = hashlib.sha256(_seed_payload(spec, domain=domain, epoch=epoch, session=session, cycle=cycle)).hexdigest()
    return int(digest, 16)


def extra_session_indices(spec: ScheduleSpec, epoch: int) -> tuple[int, ...]:
    """The handoff-frozen rotating extras: ``(13*epoch+j) mod 27`` publicly."""
    _require(type(epoch) is int and 0 <= epoch < spec.epochs, "epoch outside frozen schedule")
    values = tuple((spec.extras_per_epoch * epoch + offset) % spec.roster_size for offset in range(spec.extras_per_epoch))
    _require(len(set(values)) == spec.extras_per_epoch, "extra-session rotation has a duplicate")
    return values


def _python_rng_snapshot() -> str:
    return _sha256(repr(random.getstate()).encode("utf-8"))


def rng_snapshot(*, numpy_module: Any | None = None, torch_module: Any | None = None) -> dict[str, str]:
    """Digest caller-supplied global RNGs without importing NumPy/Torch itself."""
    snapshot = {"python": _python_rng_snapshot()}
    if numpy_module is not None:
        state = numpy_module.random.get_state()
        # numpy's MT state array is binary-stable; no schedule code may alter it.
        snapshot["numpy"] = _sha256(
            state[0].encode("ascii") + state[1].tobytes() + repr(state[2:]).encode("ascii")
        )
    if torch_module is not None:
        state = torch_module.get_rng_state()
        snapshot["torch_cpu"] = _sha256(state.detach().cpu().numpy().tobytes())
    return snapshot


@dataclass(frozen=True)
class SessionInventory:
    """One source session's exact global dataset indices and valid-window binding."""

    session: str
    dataset_indices: tuple[int, ...]
    valid_starts_sha256: str

    def __post_init__(self) -> None:
        _require(isinstance(self.session, str) and self.session, "source session name drift")
        _require(_is_sha256(self.valid_starts_sha256), "valid-start digest drift")
        _require(bool(self.dataset_indices), f"{self.session}: no eligible windows")
        _require(all(type(index) is int and index >= 0 for index in self.dataset_indices),
                 f"{self.session}: non-integer/negative dataset index")
        _require(len(set(self.dataset_indices)) == len(self.dataset_indices),
                 f"{self.session}: duplicate eligible dataset index")
        _require(
            self.dataset_indices
            == tuple(range(self.dataset_indices[0], self.dataset_indices[0] + len(self.dataset_indices))),
            f"{self.session}: eligible dataset indices are not one contiguous canonical range",
        )

    @property
    def eligible_windows(self) -> int:
        return len(self.dataset_indices)

    def full_batches(self, spec: ScheduleSpec) -> int:
        count = self.eligible_windows // spec.batch_size
        _require(count > 0, f"{self.session}: fewer than one full B{spec.batch_size} batch")
        return count

    def remainder_windows(self, spec: ScheduleSpec) -> int:
        return self.eligible_windows % spec.batch_size

    def payload(self, spec: ScheduleSpec) -> dict[str, object]:
        return {
            "session": self.session,
            "eligible_windows": self.eligible_windows,
            "full_batches": self.full_batches(spec),
            "remainder_windows": self.remainder_windows(spec),
            "dataset_index_min": min(self.dataset_indices),
            "dataset_index_max": max(self.dataset_indices),
            "valid_starts_sha256": self.valid_starts_sha256,
        }


@dataclass(frozen=True)
class SourceInventory:
    """Canonical roster ordered exactly as the strict manifest train field."""

    roster: tuple[str, ...]
    sessions: tuple[SessionInventory, ...]

    def __post_init__(self) -> None:
        _require(len(self.roster) == len(self.sessions) and len(self.roster) >= 2,
                 "source inventory cardinality drift")
        _require(len(set(self.roster)) == len(self.roster), "source inventory roster duplicates")
        _require(tuple(item.session for item in self.sessions) == self.roster,
                 "source inventory session order differs from canonical roster")
        # Dataset indices must be a complete, non-overlapping global index map.
        flattened = tuple(index for item in self.sessions for index in item.dataset_indices)
        _require(tuple(sorted(flattened)) == tuple(range(len(flattened))),
                 "source inventory dataset indices are not one canonical contiguous map")

    def by_name(self) -> dict[str, SessionInventory]:
        return {item.session: item for item in self.sessions}

    @property
    def eligible_windows(self) -> int:
        return sum(item.eligible_windows for item in self.sessions)

    def full_batches(self, spec: ScheduleSpec) -> int:
        return sum(item.full_batches(spec) for item in self.sessions)

    def remainder_windows(self, spec: ScheduleSpec) -> int:
        return sum(item.remainder_windows(spec) for item in self.sessions)

    def payload(self, spec: ScheduleSpec) -> dict[str, object]:
        return {
            "roster": list(self.roster),
            "sessions": [item.payload(spec) for item in self.sessions],
            "eligible_windows": self.eligible_windows,
            "full_batches": self.full_batches(spec),
            "remainder_windows": self.remainder_windows(spec),
        }


def inventory_from_window_counts(
    roster: Sequence[str],
    session_to_valid_starts: Mapping[str, Sequence[int]],
) -> SourceInventory:
    """Build the exact global Dataset-index layout without importing Torch."""
    canonical_roster = tuple(roster)
    _require(len(canonical_roster) == len(set(canonical_roster)), "input roster has duplicates")
    _require(set(session_to_valid_starts) == set(canonical_roster), "window-count source roster mismatch")
    offset = 0
    sessions: list[SessionInventory] = []
    for session in canonical_roster:
        starts = tuple(session_to_valid_starts[session])
        _require(all(type(value) is int and value >= 0 for value in starts),
                 f"{session}: valid-window starts must be nonnegative integers")
        # The Dataset indexes *window rows*, not unique wall-clock starts.  Keep
        # any source-order duplicate start exactly as the established datamodule
        # would: its global Dataset indices remain distinct and are what the
        # sampler must prove unique within a B32 batch.
        # The existing Dataset consumes starts in their source/datamodule order.
        starts_digest = _sha256(_canonical_json_bytes({"session": session, "valid_starts": list(starts)}))
        indices = tuple(range(offset, offset + len(starts)))
        sessions.append(SessionInventory(session=session, dataset_indices=indices,
                                         valid_starts_sha256=starts_digest))
        offset += len(starts)
    return SourceInventory(roster=canonical_roster, sessions=tuple(sessions))


@dataclass(frozen=True)
class ScheduledBatch:
    epoch: int
    position: int
    session: str
    cycle: int
    cycle_batch: int
    dataset_indices: tuple[int, ...]

    def payload(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "position": self.position,
            "session": self.session,
            "cycle": self.cycle,
            "cycle_batch": self.cycle_batch,
            "dataset_indices": list(self.dataset_indices),
        }


class EpochSchedule:
    """An explicit epoch-bound sampler; ``__iter__`` never advances an epoch."""

    def __init__(self, *, spec: ScheduleSpec, inventory: SourceInventory, epoch: int) -> None:
        _require(len(inventory.roster) == spec.roster_size, "schedule/inventory roster size mismatch")
        _require(type(epoch) is int and 0 <= epoch < spec.epochs, "epoch outside schedule plan")
        self.spec = spec
        self.inventory = inventory
        self.epoch = epoch
        self.extra_indices = extra_session_indices(spec, epoch)
        self.extra_sessions = tuple(inventory.roster[index] for index in self.extra_indices)
        self.per_session_batches = {
            session: spec.base_batches_per_session + int(index in self.extra_indices)
            for index, session in enumerate(inventory.roster)
        }
        positions = [
            session
            for session in inventory.roster
            for _ in range(self.per_session_batches[session])
        ]
        _require(len(positions) == spec.steps_per_epoch, "epoch schedule batch arithmetic drift")
        random.Random(local_rng_seed(spec, domain="global-order", epoch=epoch)).shuffle(positions)
        self._positions = tuple(positions)
        self._by_name = inventory.by_name()

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def positions(self) -> tuple[str, ...]:
        """Canonical session position sequence; safe to inspect, never mutable."""
        return self._positions

    def _cycle_indices(self, session: str, cycle: int) -> array:
        """Return one compact shuffled full-batch cycle for one source session.

        An ``array('I')`` is deliberately used instead of a Python list of B32
        lists.  The public plan therefore retains only compact per-cycle index
        buffers while an iterator is live, not 1,628,400 Python batch objects.
        Dataset indices are bounded by the source-window authority and fit in
        unsigned 32-bit storage; the explicit check below makes that invariant
        fail closed rather than truncating an unexpected future dataset.
        """
        item = self._by_name[session]
        full_batches = item.full_batches(self.spec)
        _require(item.dataset_indices[-1] <= 0xFFFFFFFF,
                 f"{session}: dataset index exceeds compact unsigned-32 schedule representation")
        shuffled = array("I", item.dataset_indices)
        random.Random(
            local_rng_seed(self.spec, domain="session-cycle", epoch=self.epoch, session=session, cycle=cycle)
        ).shuffle(shuffled)
        usable_count = full_batches * self.spec.batch_size
        del shuffled[usable_count:]
        _require(len(shuffled) == usable_count, f"{session}: deterministic cycle full-batch drift")
        _require(len(set(shuffled)) == usable_count,
                 f"{session}: repeated index before deterministic cycle exhaustion")
        return shuffled

    def iter_batches(self) -> Iterator[ScheduledBatch]:
        """Yield one exact immutable epoch schedule, with no hidden epoch state."""
        occurrences = {session: 0 for session in self.inventory.roster}
        cycle_cache: dict[tuple[str, int], array] = {}
        for position, session in enumerate(self._positions):
            ordinal = occurrences[session]
            occurrences[session] += 1
            full_batches = self._by_name[session].full_batches(self.spec)
            cycle, cycle_batch = divmod(ordinal, full_batches)
            key = (session, cycle)
            if key not in cycle_cache:
                cycle_cache[key] = self._cycle_indices(session, cycle)
            start = cycle_batch * self.spec.batch_size
            indices = tuple(cycle_cache[key][start:start + self.spec.batch_size])
            _require(len(indices) == self.spec.batch_size and len(set(indices)) == self.spec.batch_size,
                     f"{session}: compact deterministic cycle emitted a duplicate/short batch")
            yield ScheduledBatch(
                epoch=self.epoch,
                position=position,
                session=session,
                cycle=cycle,
                cycle_batch=cycle_batch,
                dataset_indices=indices,
            )
        _require(occurrences == self.per_session_batches, "epoch schedule occurrence accounting drift")


class ExplicitEpochBatchSampler:
    """DataLoader-compatible batch sampler that cannot infer/increment an epoch.

    It may be iterated more than once by a framework (sanity/reload/worker
    behavior), but every iteration reproduces the same explicitly bound epoch.
    The future training loop must claim the epoch through ``EpochLedger`` once
    before asking a DataLoader to consume this object.
    """

    def __init__(self, schedule: EpochSchedule) -> None:
        self.schedule = schedule

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.schedule.iter_batches():
            yield list(batch.dataset_indices)

    def __len__(self) -> int:
        return len(self.schedule)


def _update_epoch_hasher(hasher: Any, batch: ScheduledBatch) -> None:
    hasher.update(struct.pack(
        ">IIII", batch.epoch, batch.position, batch.cycle, batch.cycle_batch,
    ))
    encoded_session = batch.session.encode("utf-8")
    hasher.update(struct.pack(">I", len(encoded_session)))
    hasher.update(encoded_session)
    hasher.update(struct.pack(">I", len(batch.dataset_indices)))
    for index in batch.dataset_indices:
        hasher.update(struct.pack(">Q", index))


def epoch_schedule_evidence(schedule: EpochSchedule) -> dict[str, object]:
    """Stream one epoch's batch contents into a compact, exact evidence record."""
    hasher = hashlib.sha256()
    header = {
        "schema": "cell_d_equal_session_epoch_schedule_v1",
        "spec": schedule.spec.payload(),
        "epoch": schedule.epoch,
        "roster": list(schedule.inventory.roster),
        "extra_indices": list(schedule.extra_indices),
        "extra_sessions": list(schedule.extra_sessions),
        "per_session_batches": schedule.per_session_batches,
        "inventory_valid_start_digests": {
            item.session: item.valid_starts_sha256 for item in schedule.inventory.sessions
        },
    }
    hasher.update(_canonical_json_bytes(header))
    batch_count = 0
    seen_positions: set[int] = set()
    cycle_counts = {session: 0 for session in schedule.inventory.roster}
    valid_ranges = {
        item.session: (item.dataset_indices[0], item.dataset_indices[-1])
        for item in schedule.inventory.sessions
    }
    for batch in schedule.iter_batches():
        _require(batch.position not in seen_positions, "epoch schedule position duplicated")
        seen_positions.add(batch.position)
        _require(batch.session in valid_ranges, "epoch schedule emitted forbidden source session")
        _require(len(batch.dataset_indices) == schedule.spec.batch_size,
                 "epoch schedule emitted non-B32 batch")
        _require(len(set(batch.dataset_indices)) == schedule.spec.batch_size,
                 "epoch schedule emitted a duplicate inside batch")
        lower, upper = valid_ranges[batch.session]
        _require(all(lower <= index <= upper for index in batch.dataset_indices),
                 "epoch schedule emitted an out-of-session dataset index")
        _update_epoch_hasher(hasher, batch)
        batch_count += 1
        cycle_counts[batch.session] = max(cycle_counts[batch.session], batch.cycle + 1)
    _require(batch_count == schedule.spec.steps_per_epoch, "epoch schedule total batch count drift")
    _require(len(seen_positions) == schedule.spec.steps_per_epoch, "epoch schedule position coverage drift")
    _require(sum(schedule.per_session_batches.values()) == batch_count,
             "epoch schedule per-session exposure sum drift")
    return {
        **header,
        "batch_count": batch_count,
        "all_batches_single_session": True,
        "all_batches_exact_batch_size": True,
        "all_batches_unique_within_batch": True,
        "all_indices_authorized_for_session": True,
        "cycles_used_by_session": cycle_counts,
        "schedule_sha256": hasher.hexdigest(),
    }


def full_run_exposure(spec: ScheduleSpec, roster: Sequence[str]) -> dict[str, int]:
    _require(len(roster) == spec.roster_size and len(set(roster)) == spec.roster_size,
             "full-run exposure roster drift")
    counts = {session: 0 for session in roster}
    for epoch in range(spec.epochs):
        extras = set(extra_session_indices(spec, epoch))
        for index, session in enumerate(roster):
            counts[session] += spec.base_batches_per_session + int(index in extras)
    _require(max(counts.values()) - min(counts.values()) <= 1, "full-run equal exposure imbalance exceeds one")
    return counts


def build_plan_evidence(
    *,
    spec: ScheduleSpec,
    inventory: SourceInventory,
    snapshot_rng: Callable[[], Mapping[str, str]] | None = None,
) -> dict[str, object]:
    """Stream all epoch schedules without retaining 1,628,400 Python batches."""
    _require(len(inventory.roster) == spec.roster_size, "plan inventory roster size drift")
    snapshot = snapshot_rng or (lambda: {"python": _python_rng_snapshot()})
    before_rng = dict(snapshot())
    full_hasher = hashlib.sha256()
    full_hasher.update(_canonical_json_bytes({
        "schema": "cell_d_equal_session_full_plan_v1",
        "spec": spec.payload(),
        "roster": list(inventory.roster),
        "inventory": inventory.payload(spec),
    }))
    epochs: list[dict[str, object]] = []
    for epoch in range(spec.epochs):
        evidence = epoch_schedule_evidence(EpochSchedule(spec=spec, inventory=inventory, epoch=epoch))
        # The full plan is a domain-separated stream of complete per-epoch
        # records.  It stores 48 compact summaries, not all batch lists.
        compact = {
            "epoch": evidence["epoch"],
            "extra_indices": evidence["extra_indices"],
            "per_session_batches": evidence["per_session_batches"],
            "schedule_sha256": evidence["schedule_sha256"],
        }
        full_hasher.update(_canonical_json_bytes(compact))
        epochs.append(evidence)
    after_rng = dict(snapshot())
    if before_rng != after_rng:
        raise ContractError("schedule construction mutated a global Python/NumPy/Torch RNG state")
    full_counts = full_run_exposure(spec, inventory.roster)
    _require(sum(full_counts.values()) == spec.total_steps, "full-run total schedule count drift")
    _require(sum(int(item["batch_count"]) for item in epochs) == spec.total_steps,
             "full-run streamed epoch count drift")
    _require(len({str(item["schedule_sha256"]) for item in epochs}) == spec.epochs,
             "planned epoch schedules are not 48 distinct schedule records")
    return {
        "schema": "cell_d_equal_session_full_plan_v1",
        "spec": spec.payload(),
        "inventory": inventory.payload(spec),
        "epochs": epochs,
        "full_run_batches_by_session": full_counts,
        "full_run_max_minus_min_batches": max(full_counts.values()) - min(full_counts.values()),
        "full_plan_sha256": full_hasher.hexdigest(),
        "global_rng_unchanged": True,
        "plan_retains_batch_lists": False,
    }


class EpochLedger:
    """Enforce exactly one explicit, ordered consumption of each planned epoch."""

    def __init__(self, plan: Mapping[str, object]) -> None:
        if plan.get("schema") != "cell_d_equal_session_full_plan_v1":
            raise ContractError("epoch ledger requires a validated full plan")
        raw_epochs = plan.get("epochs")
        if not isinstance(raw_epochs, list) or not raw_epochs:
            raise ContractError("epoch ledger plan lacks epoch evidence")
        self._expected = {
            int(item["epoch"]): str(item["schedule_sha256"])
            for item in raw_epochs
            if isinstance(item, Mapping) and type(item.get("epoch")) is int and _is_sha256(item.get("schedule_sha256"))
        }
        if sorted(self._expected) != list(range(len(raw_epochs))):
            raise ContractError("epoch ledger plan epoch topology drift")
        self._claimed: list[int] = []

    def claim(self, schedule: EpochSchedule) -> str:
        epoch = schedule.epoch
        if epoch in self._claimed or epoch != len(self._claimed):
            raise ContractError("training attempted an implicit/repeated/out-of-order epoch schedule")
        observed = str(epoch_schedule_evidence(schedule)["schedule_sha256"])
        if observed != self._expected.get(epoch):
            raise ContractError("training epoch schedule differs from preflight plan")
        self._claimed.append(epoch)
        return observed

    def final_assert_exact(self) -> None:
        if self._claimed != list(range(len(self._expected))):
            raise ContractError("training did not consume exactly the 48 planned epoch schedules")


def original_vs_equal_exposure(spec: ScheduleSpec, inventory: SourceInventory,
                               plan: Mapping[str, object]) -> list[dict[str, object]]:
    """Compact per-session exposure comparison, retaining both floor-batch facts."""
    full_total = inventory.full_batches(spec)
    eligible_total = inventory.eligible_windows
    full_counts = plan.get("full_run_batches_by_session")
    if not isinstance(full_counts, Mapping):
        raise ContractError("plan lacks full-run exposure counts")
    rows: list[dict[str, object]] = []
    for item in inventory.sessions:
        source_batches = item.full_batches(spec)
        full_run_batches = full_counts.get(item.session)
        if type(full_run_batches) is not int:
            raise ContractError("plan full-run exposure session drift")
        equal_mean_per_epoch = full_run_batches / spec.epochs
        rows.append({
            "session": item.session,
            "eligible_windows": item.eligible_windows,
            "full_batches_predecessor": source_batches,
            "remainder_windows_discarded_per_epoch": item.remainder_windows(spec),
            "original_eligible_window_fraction": item.eligible_windows / eligible_total,
            "original_floor_batch_fraction": source_batches / full_total,
            "equal_batches_epoch_base": spec.base_batches_per_session,
            "equal_batches_full_run": full_run_batches,
            "equal_batches_mean_per_epoch": equal_mean_per_epoch,
            "equal_vs_window_proportional_batch_ratio": equal_mean_per_epoch / source_batches,
        })
    return rows


# ---------------------------------------------------------------------------
# Immutable Cell-D and strict-source authorities
# ---------------------------------------------------------------------------


def _load_immutable_json(
    root: Path,
    relative: str,
    *,
    expected_sha256: str,
    mode: int,
) -> dict[str, object]:
    """Descriptor-read one receipt/manifest and bind its conventional sidecar.

    This route reads immutable predecessor *metadata* only.  It never opens a
    predecessor checkpoint, a target session, or an evaluation surface.
    """
    body = _regular_bytes(root, relative, mode=mode)
    observed_sha256 = _sha256(body)
    _require(observed_sha256 == expected_sha256,
             f"immutable authority SHA drift: {relative}")
    sidecar_relative = f"{relative}.sha256"
    try:
        sidecar = _regular_bytes(root, sidecar_relative, mode=mode).decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ContractError(f"immutable authority sidecar is not ASCII: {sidecar_relative}") from error
    fields = sidecar.split()
    _require(len(fields) == 2 and fields[0] == observed_sha256
             and fields[1] == Path(relative).name,
             f"immutable authority sidecar drift: {sidecar_relative}")
    try:
        decoded = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise ContractError(f"immutable authority is not a JSON object: {relative}") from error
    _require(isinstance(decoded, dict), f"immutable authority root is not an object: {relative}")
    return decoded


def _mapping(value: object, label: str) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} is not a mapping")
    return value


def _exact(value: object, expected: object, label: str) -> None:
    _require(value == expected, f"{label} drift: observed={value!r}, expected={expected!r}")


def _utc_seconds(rendered: str) -> int:
    try:
        parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"invalid UTC timestamp: {rendered!r}") from error
    _require(parsed.tzinfo is not None, "timestamp lacks timezone")
    return int(parsed.astimezone(timezone.utc).timestamp())


def validate_sealed_cell_d_contract(
    launch: Mapping[str, object],
    terminal: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact non-sampler Cell-D authority for this successor.

    The successor is not allowed to inherit a loose verbal description of Cell
    D.  This validator binds the training graph/configuration, data visibility,
    optimizer schedule, fixed initial state, checkpoint/SWA topology, and the
    observed predecessor budget.  The future route may change only the sampler
    name and per-epoch source exposure proof.
    """
    for label, receipt, schema, status in (
        ("launch", launch, "tfpd_pop_robust_cell_v1_launch", "CELL_LAUNCHED"),
        ("terminal", terminal, "tfpd_pop_robust_cell_v1", "CELL_TERMINAL"),
    ):
        _exact(receipt.get("schema"), schema, f"sealed Cell-D {label} schema")
        _exact(receipt.get("cell"), "D", f"sealed Cell-D {label} cell")
        _exact(receipt.get("status"), status, f"sealed Cell-D {label} status")
        _exact(receipt.get("smoke"), False, f"sealed Cell-D {label} smoke flag")

        config = _mapping(receipt.get("head_and_dropout_config"), f"sealed {label} head/dropout")
        _exact(config.get("num_heads"), 2, f"sealed {label} head count")
        _exact(config.get("dynamic_dropout"), True, f"sealed {label} dynamic dropout")
        _exact(config.get("dynamic_dropout_low"), 0.0, f"sealed {label} dropout low")
        _exact(config.get("dynamic_dropout_high"), 1.0, f"sealed {label} dropout high")

        initial = _mapping(receipt.get("initial_state"), f"sealed {label} initial state")
        _exact(initial.get("artifact_sha256"), CANONICAL_INITIAL_STATE_SHA256,
               f"sealed {label} canonical-initial artifact")
        _exact(initial.get("loaded_state_sha256"), CANONICAL_INITIAL_STATE_STATE_SHA256,
               f"sealed {label} loaded initial state")
        _exact(initial.get("state_dict_sha256"), CANONICAL_INITIAL_STATE_STATE_SHA256,
               f"sealed {label} initial state dict")
        _exact(initial.get("strict_load"), True, f"sealed {label} strict initial load")
        proof = _mapping(initial.get("bitwise_equality_proof"), f"sealed {label} initial proof")
        proof_d = _mapping(proof.get("D"), f"sealed {label} Cell-D initial proof")
        _exact(proof_d.get("strict_load"), True, f"sealed {label} proof strict-load")
        _exact(proof_d.get("state_keys_equal_to_canonical"), True,
               f"sealed {label} proof state-key equality")
        _exact(proof_d.get("num_heads"), 2, f"sealed {label} proof head count")
        _exact(proof_d.get("dynamic_dropout"), True, f"sealed {label} proof dropout")
        _exact(proof_d.get("trainable_parameters"), 3_510_842,
               f"sealed {label} proof trainable-parameter count")

        budget = _mapping(receipt.get("budget"), f"sealed {label} budget")
        _exact(budget.get("epochs"), 48, f"sealed {label} epoch count")
        _exact(budget.get("steps_per_epoch"), 33_925, f"sealed {label} steps per epoch")
        _exact(budget.get("total_optimizer_steps"), 1_628_400,
               f"sealed {label} total optimizer steps")
        optimizer = _mapping(budget.get("optimizer"), f"sealed {label} optimizer")
        _exact(optimizer.get("cls"), "torch.optim.Adam", f"sealed {label} optimizer class")
        _exact(optimizer.get("lr"), 1e-4, f"sealed {label} optimizer learning rate")
        _exact(optimizer.get("betas"), [0.9, 0.999], f"sealed {label} optimizer betas")
        _exact(optimizer.get("eps"), 1e-8, f"sealed {label} optimizer epsilon")
        _exact(optimizer.get("weight_decay"), 0.0, f"sealed {label} optimizer weight decay")
        _exact(optimizer.get("amsgrad"), False, f"sealed {label} optimizer amsgrad")
        schedule = _mapping(budget.get("schedule"), f"sealed {label} learning-rate schedule")
        for key, expected in {
            "kind": "warmup_then_cosine",
            "n_epochs": 48,
            "steps_per_epoch": 33_925,
            "total_steps": 1_628_400,
            "warmup_epochs": 2,
            "warmup_steps": 67_850,
            "lr_warmup_start": 1e-5,
            "lr_warmup_end": 1e-4,
            "lr_final": 1e-6,
            "phase_local_steps": True,
        }.items():
            _exact(schedule.get(key), expected, f"sealed {label} schedule {key}")

        data_contract = _mapping(receipt.get("data_contract"), f"sealed {label} data contract")
        _exact(data_contract.get("roster_n"), 27, f"sealed {label} source roster size")
        _exact(data_contract.get("visible_side"), "canonical normalized T4",
               f"sealed {label} visible side")
        _exact(data_contract.get("behavior_normalizer_semantic_sha256"),
               SEALED_BEHAVIOR_NORMALIZER_SHA256, f"sealed {label} behavior normalizer")
        for forbidden_open in (
            "within_dev_sessions_opened",
            "external_sub_m_opened",
            "formal_or_organizer_held_data_opened",
        ):
            _exact(data_contract.get(forbidden_open), False, f"sealed {label} {forbidden_open}")

        disclosures = _mapping(receipt.get("disclosures"), f"sealed {label} disclosures")
        _exact(disclosures.get("teacher_checkpoint_logits_or_loss_used"), False,
               f"sealed {label} teacher use")
        _exact(disclosures.get("pretraining_used"), False, f"sealed {label} pretraining use")
        _exact(disclosures.get("extra_seed"), False, f"sealed {label} extra seed")
        _exact(disclosures.get("width_changed"), False, f"sealed {label} width change")
        _exact(disclosures.get("clipping"), "none", f"sealed {label} clipping")

    _exact(launch.get("started_utc"), PREDECESSOR_STARTED_UTC, "sealed Cell-D predecessor start")
    _exact(terminal.get("started_utc"), PREDECESSOR_STARTED_UTC, "sealed Cell-D terminal start")
    _exact(terminal.get("finished_utc"), PREDECESSOR_FINISHED_UTC, "sealed Cell-D terminal finish")
    _exact(_utc_seconds(PREDECESSOR_FINISHED_UTC) - _utc_seconds(PREDECESSOR_STARTED_UTC),
           PREDECESSOR_WALL_CLOCK_SECONDS, "sealed Cell-D predecessor duration")
    _exact(terminal.get("epochs_run"), 48, "sealed Cell-D terminal epochs run")
    _exact(terminal.get("invariant_failures"), [], "sealed Cell-D terminal invariant failures")

    diagnostics = terminal.get("diagnostics_per_epoch")
    _require(isinstance(diagnostics, list) and len(diagnostics) == 48,
             "sealed Cell-D diagnostics topology drift")
    checkpoint_epochs: list[int] = []
    for expected_epoch, diagnostic in enumerate(diagnostics):
        diagnostic_map = _mapping(diagnostic, f"sealed Cell-D diagnostic {expected_epoch}")
        _exact(diagnostic_map.get("epoch"), expected_epoch,
               f"sealed Cell-D diagnostic epoch {expected_epoch}")
        _exact(diagnostic_map.get("optimizer_steps"), 33_925,
               f"sealed Cell-D diagnostic optimizer steps {expected_epoch}")
        _exact(diagnostic_map.get("optimizer_steps_total"), (expected_epoch + 1) * 33_925,
               f"sealed Cell-D diagnostic total optimizer steps {expected_epoch}")
        _exact(diagnostic_map.get("train_example_windows"), SOURCE_ONLY_EXPECTED_PREDECESSOR_CONSUMED_WINDOWS,
               f"sealed Cell-D diagnostic source windows {expected_epoch}")
        _exact(diagnostic_map.get("parameters_finite"), True,
               f"sealed Cell-D diagnostic parameter finite {expected_epoch}")
        _exact(diagnostic_map.get("optimizer_state_finite"), True,
               f"sealed Cell-D diagnostic optimizer finite {expected_epoch}")
        _exact(diagnostic_map.get("t4_authority_unchanged"), True,
               f"sealed Cell-D diagnostic T4 authority {expected_epoch}")
        checkpoint = diagnostic_map.get("checkpoint_saved")
        if checkpoint is not None:
            _require(isinstance(checkpoint, str) and checkpoint.startswith("epoch"),
                     f"sealed Cell-D diagnostic checkpoint name {expected_epoch}")
            checkpoint_epochs.append(expected_epoch)
    _exact(checkpoint_epochs, [44, 45, 46, 47], "sealed Cell-D diagnostic checkpoints")

    checkpoints = terminal.get("checkpoints")
    _require(isinstance(checkpoints, list) and len(checkpoints) == 4,
             "sealed Cell-D checkpoint list topology drift")
    _exact([entry.get("epoch") if isinstance(entry, Mapping) else None for entry in checkpoints],
           [44, 45, 46, 47], "sealed Cell-D checkpoint epochs")
    _exact([entry.get("file") if isinstance(entry, Mapping) else None for entry in checkpoints],
           ["epoch044.ckpt", "epoch045.ckpt", "epoch046.ckpt", "epoch047.ckpt"],
           "sealed Cell-D checkpoint names")
    _require(all(isinstance(entry, Mapping) and _is_sha256(entry.get("sha256")) for entry in checkpoints),
             "sealed Cell-D checkpoint digest topology drift")

    swa = _mapping(terminal.get("swa"), "sealed Cell-D SWA")
    _exact(swa.get("window_epochs"), [44, 45, 46, 47], "sealed Cell-D SWA window")
    _exact(swa.get("strict_reload_finite_forward_smoke"), True,
           "sealed Cell-D SWA strict reload/finiteness")
    _require(_is_sha256(swa.get("sha256")), "sealed Cell-D SWA digest topology drift")
    swa_manifest = _mapping(swa.get("manifest"), "sealed Cell-D SWA manifest")
    _exact(swa_manifest.get("strict_reload_verified"), True, "sealed Cell-D SWA strict reload")
    _exact(swa_manifest.get("uninitialized_lazy_tensor_count"), 2,
           "sealed Cell-D SWA lazy topology")
    _exact(swa_manifest.get("optimizer_state_included"), False,
           "sealed Cell-D SWA optimizer state")
    components = swa_manifest.get("components")
    _require(isinstance(components, list) and len(components) == 4,
             "sealed Cell-D SWA components topology drift")
    _require(all(isinstance(component, Mapping) and _is_sha256(component.get("sha256"))
                 for component in components), "sealed Cell-D SWA component digest drift")

    return {
        "schema": "cell_d_equal_session_predecessor_authority_v1",
        "cell": "D",
        "model_graph": "sealed Cell-D, B3S + normalized T4, two attention heads",
        "trainable_parameters": 3_510_842,
        "initial_state_artifact_sha256": CANONICAL_INITIAL_STATE_SHA256,
        "initial_state_state_dict_sha256": CANONICAL_INITIAL_STATE_STATE_SHA256,
        "dynamic_dropout": {"enabled": True, "low": 0.0, "high": 1.0},
        "optimizer": {"class": "torch.optim.Adam", "lr": 1e-4, "weight_decay": 0.0},
        "budget": PUBLIC_SPEC.payload(),
        "swa_epochs": [44, 45, 46, 47],
        "source_windows_per_epoch_predecessor": SOURCE_ONLY_EXPECTED_PREDECESSOR_CONSUMED_WINDOWS,
        "predecessor_started_utc": PREDECESSOR_STARTED_UTC,
        "predecessor_finished_utc": PREDECESSOR_FINISHED_UTC,
        "predecessor_wall_clock_seconds": PREDECESSOR_WALL_CLOCK_SECONDS,
        "planning_estimate_hours": "5-6",
    }


def load_sealed_cell_d_contract(root: Path) -> dict[str, object]:
    """Read only immutable predecessor receipt bytes and validate the contract."""
    launch = _load_immutable_json(
        root,
        SEALED_CELL_D_LAUNCH_RELATIVE,
        expected_sha256=SEALED_LAUNCH_RECEIPT_SHA256,
        mode=0o444,
    )
    terminal = _load_immutable_json(
        root,
        SEALED_CELL_D_TERMINAL_RELATIVE,
        expected_sha256=SEALED_TERMINAL_RECEIPT_SHA256,
        mode=0o444,
    )
    payload = validate_sealed_cell_d_contract(launch, terminal)
    return {
        **payload,
        "launch_receipt_sha256": SEALED_LAUNCH_RECEIPT_SHA256,
        "terminal_receipt_sha256": SEALED_TERMINAL_RECEIPT_SHA256,
        "predecessor_metadata_only": True,
        "checkpoint_or_model_opened": False,
    }


def load_sealed_arm_a_source_authority(root: Path) -> dict[str, object]:
    """Read the immutable strict-27 T4/window authority, never a checkpoint."""
    receipt = _load_immutable_json(
        root,
        SEALED_ARM_A_PREFLIGHT_RELATIVE,
        expected_sha256=SEALED_ARM_A_PREFLIGHT_SHA256,
        mode=0o444,
    )
    _exact(receipt.get("schema"), "tfpd_admission_arm_v1_preflight", "sealed Arm-A preflight schema")
    _exact(receipt.get("status"), "ARM_PREFLIGHT_PASSED", "sealed Arm-A preflight status")
    _exact(receipt.get("arm"), "A", "sealed Arm-A preflight arm")
    _exact(receipt.get("smoke"), False, "sealed Arm-A preflight smoke")
    data = _mapping(receipt.get("data_contract"), "sealed Arm-A source data contract")
    _exact(data.get("manifest_sha256"), MANIFEST_SHA256, "sealed Arm-A manifest")
    _exact(data.get("n_train_sessions"), 27, "sealed Arm-A source roster size")
    _exact(data.get("n_train_windows"), SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS,
           "sealed Arm-A source window count")
    _exact(data.get("steps_per_epoch"), PUBLIC_SPEC.steps_per_epoch,
           "sealed Arm-A predecessor steps per epoch")
    normalizers = _mapping(data.get("normalizers"), "sealed Arm-A normalizers")
    _exact(normalizers.get("behavior_semantic_sha256"), SEALED_BEHAVIOR_NORMALIZER_SHA256,
           "sealed Arm-A behavior normalizer")
    _exact(normalizers.get("side_feature_semantic_sha256"), SEALED_T4_NORMALIZER_SHA256,
           "sealed Arm-A T4 normalizer")
    roster = data.get("roster")
    _require(isinstance(roster, list) and len(roster) == 27 and all(isinstance(item, str) for item in roster),
             "sealed Arm-A roster topology")
    fingerprints = receipt.get("t4_authority_sha256")
    _require(isinstance(fingerprints, Mapping) and set(fingerprints) == set(roster)
             and all(_is_sha256(value) for value in fingerprints.values()),
             "sealed Arm-A T4 fingerprint topology")
    return {
        "schema": "cell_d_equal_session_arm_a_source_authority_v1",
        "preflight_receipt_relative": SEALED_ARM_A_PREFLIGHT_RELATIVE,
        "preflight_receipt_sha256": SEALED_ARM_A_PREFLIGHT_SHA256,
        "roster": list(roster),
        "behavior_normalizer_semantic_sha256": SEALED_BEHAVIOR_NORMALIZER_SHA256,
        "t4_normalizer_semantic_sha256": SEALED_T4_NORMALIZER_SHA256,
        "t4_authority_sha256": dict(fingerprints),
        "metadata_only": True,
    }


def _failed_smoke_v1_spec_payload() -> dict[str, object]:
    """The immutable v1 smoke topology, retained only as failure lineage."""
    return {
        "kind": "source_smoke",
        "root_relative": FAILED_SMOKE_V1_ROOT_RELATIVE,
        "epochs": 1,
        "steps_per_epoch": 1,
        "total_steps": 1,
        "checkpoint_epochs": [],
        "throughput_steps": 1,
    }


def _failed_smoke_v1_lineage_payload() -> dict[str, object]:
    """Compact facts from the immutable, honest v1 pre-CUDA failure."""
    return {
        "schema": "cell_d_equal_session_failed_smoke_v1_lineage_v1",
        "failed_root_relative": FAILED_SMOKE_V1_ROOT_RELATIVE,
        "attempt": {
            "relative": FAILED_SMOKE_V1_ATTEMPT_RELATIVE,
            "sha256": FAILED_SMOKE_V1_ATTEMPT_SHA256,
        },
        "launch": {
            "relative": FAILED_SMOKE_V1_LAUNCH_RELATIVE,
            "sha256": FAILED_SMOKE_V1_LAUNCH_SHA256,
        },
        "failure": {
            "relative": FAILED_SMOKE_V1_FAILURE_RELATIVE,
            "sha256": FAILED_SMOKE_V1_FAILURE_SHA256,
            "stage": "backend_prepare",
            "kind": "UnpicklingError",
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "backward_calls": 0,
            "update_calls": 0,
            "target_resolved_or_opened": False,
            "formal_or_organizer_held_resolved_or_opened": False,
        },
        "run_spec": _failed_smoke_v1_spec_payload(),
        "identity_closure_sha256": FAILED_SMOKE_V1_CLOSURE_SHA256,
        "source_train_opened_before_canonical_validation": True,
        "immutable_metadata_only": True,
    }


def validate_failed_smoke_v1_lineage(
    attempt: Mapping[str, object],
    launch: Mapping[str, object],
    failure: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact v1 failure facts before any successor capability.

    The three bodies are digest- and sidecar-bound by the caller.  This
    semantic layer prevents a same-shaped receipt from being misdescribed as
    the known safe pre-CUDA checkpoint-format failure.
    """
    expected_spec = _failed_smoke_v1_spec_payload()
    _require(
        attempt.get("schema") == "cell_d_equal_session_attempt_v1"
        and attempt.get("cell") == CELL
        and attempt.get("status") == "ATTEMPT_RESERVED"
        and attempt.get("run_spec") == expected_spec
        and attempt.get("source_only") is True
        and attempt.get("target_or_formal_or_external_forbidden") is True,
        "immutable smoke-v1 attempt schema/status drift",
    )
    identity = attempt.get("identity")
    _require(isinstance(identity, Mapping), "immutable smoke-v1 attempt identity drift")
    closure = identity.get("closure")
    canonical = identity.get("canonical_initial_state")
    _require(
        isinstance(closure, Mapping)
        and closure.get("closure_sha256") == FAILED_SMOKE_V1_CLOSURE_SHA256
        and isinstance(canonical, Mapping)
        and canonical.get("artifact_sha256") == CANONICAL_INITIAL_STATE_SHA256
        and canonical.get("state_dict_sha256") == CANONICAL_INITIAL_STATE_STATE_SHA256
        and canonical.get("strict_load") is True,
        "immutable smoke-v1 attempt identity authority drift",
    )
    _require(
        launch.get("schema") == "cell_d_equal_session_launch_v1"
        and launch.get("cell") == CELL
        and launch.get("status") == "LAUNCHED"
        and launch.get("run_spec") == expected_spec
        and launch.get("attempt_sha256") == FAILED_SMOKE_V1_ATTEMPT_SHA256
        and launch.get("identity") == identity,
        "immutable smoke-v1 launch binding drift",
    )
    failure_detail = failure.get("failure")
    access = failure.get("access_disclosure")
    _require(
        failure.get("schema") == "cell_d_equal_session_failure_v1"
        and failure.get("cell") == CELL
        and failure.get("status") == "FAILED"
        and failure.get("run_spec") == expected_spec
        and failure.get("identity") == identity
        and failure.get("stage") == "backend_prepare"
        and failure.get("terminal_published") is False
        and isinstance(failure_detail, Mapping)
        and failure_detail.get("kind") == "UnpicklingError"
        and isinstance(failure_detail.get("detail"), str)
        and "torch.torch_version.TorchVersion" in failure_detail["detail"]
        and "weights_only=True" in failure_detail["detail"]
        and isinstance(access, Mapping)
        and access.get("source_train_opened") is True
        and access.get("cuda_initialized") is False
        and access.get("optimizer_steps_completed") == 0
        and access.get("backward_calls") == 0
        and access.get("update_calls") == 0
        and access.get("within_dev_resolved_or_opened") is False
        and access.get("external_resolved_or_opened") is False
        and access.get("formal_or_organizer_held_resolved_or_opened") is False
        and access.get("target_resolved_or_opened") is False
        and access.get("cache_read_or_write") is False,
        "immutable smoke-v1 failure provenance drift",
    )
    return _failed_smoke_v1_lineage_payload()


def load_failed_smoke_v1_lineage(root: Path) -> dict[str, object]:
    """Descriptor-read all immutable v1 failure receipts and sidecars."""
    attempt = _load_immutable_json(
        root,
        FAILED_SMOKE_V1_ATTEMPT_RELATIVE,
        expected_sha256=FAILED_SMOKE_V1_ATTEMPT_SHA256,
        mode=0o444,
    )
    launch = _load_immutable_json(
        root,
        FAILED_SMOKE_V1_LAUNCH_RELATIVE,
        expected_sha256=FAILED_SMOKE_V1_LAUNCH_SHA256,
        mode=0o444,
    )
    failure = _load_immutable_json(
        root,
        FAILED_SMOKE_V1_FAILURE_RELATIVE,
        expected_sha256=FAILED_SMOKE_V1_FAILURE_SHA256,
        mode=0o444,
    )
    return validate_failed_smoke_v1_lineage(attempt, launch, failure)


def load_strict_source_roster(root: Path) -> tuple[str, ...]:
    """Bind the strict manifest while leaving val/test names inert strings.

    The function intentionally constructs paths only for ``session_splits.train``
    later in the source-only audit.  It never resolves or opens val/test/formal
    data even though those literal names live in the static manifest.
    """
    body = _regular_bytes(root, MANIFEST_RELATIVE, mode=0o664)
    _require(_sha256(body) == MANIFEST_SHA256, "strict source manifest SHA drift")
    try:
        manifest = json.loads(body)
    except json.JSONDecodeError as error:
        raise ContractError("strict source manifest is invalid JSON") from error
    _require(isinstance(manifest, dict), "strict source manifest root is not an object")
    _exact(manifest.get("schema_version"), 1, "strict source manifest schema")
    _exact(manifest.get("purpose"), "Strict validation-only SUA manifest: only train and validation NWBs may be opened.",
           "strict source manifest purpose")
    _exact(manifest.get("task"), "CO", "strict source manifest task")
    _exact(manifest.get("max_units_exclusive"), 100, "strict source manifest unit cap")
    _exact(manifest.get("split_counts"), [27, 6, 6], "strict source manifest split counts")
    splits = _mapping(manifest.get("session_splits"), "strict source manifest splits")
    train = splits.get("train")
    _require(isinstance(train, list) and len(train) == 27 and all(isinstance(item, str) for item in train),
             "strict source manifest train roster drift")
    _require(len(set(train)) == len(train), "strict source manifest train roster duplicates")
    # Validate the static split topology but keep non-train names inert: do not
    # derive a pathname from them in this source-only route.
    for inert_split in ("val", "test"):
        inert = splits.get(inert_split)
        _require(isinstance(inert, list) and len(inert) == 6
                 and all(isinstance(item, str) for item in inert),
                 f"strict source manifest {inert_split} topology drift")
        _require(len(set(inert)) == len(inert), f"strict source manifest {inert_split} duplicates")
    _require(not (set(train) & set(splits["val"])) and not (set(train) & set(splits["test"])),
             "strict source manifest train/non-train overlap")
    _require(all(item.startswith("sub-C_ses-CO-") and "/" not in item and "\\" not in item
                 for item in train), "strict source manifest unsafe train session name")
    return tuple(train)


# ---------------------------------------------------------------------------
# Source-only read adapter and no-target preflight evidence
# ---------------------------------------------------------------------------


def _source_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    """Return a minimal non-symlink identity tuple for one permitted NWB file."""
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ContractError(f"permitted source session is missing: {path.name}") from error
    _require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
             f"permitted source session is not a regular non-symlink: {path.name}")
    _require(metadata.st_size > 0, f"permitted source session is empty: {path.name}")
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns,
            stat.S_IMODE(metadata.st_mode))


def _direct_source_file(root: Path, session: str, *, allowed_train_roster: Sequence[str]) -> Path:
    """Construct a pathname only for one train-roster source session."""
    _require(isinstance(session, str) and session.startswith("sub-C_ses-CO-")
             and "/" not in session and "\\" not in session,
             "attempted to construct a non-canonical source-session path")
    _require(session in tuple(allowed_train_roster),
             "source-only adapter refused a non-train manifest session")
    source_directory = Path(root).absolute() / SOURCE_DATA_RELATIVE
    try:
        directory_metadata = os.lstat(source_directory)
    except OSError as error:
        raise ContractError("canonical strict-27 source directory is absent") from error
    _require(stat.S_ISDIR(directory_metadata.st_mode) and not stat.S_ISLNK(directory_metadata.st_mode),
             "canonical strict-27 source directory is not a direct non-symlink directory")
    candidate = source_directory / f"{session}_behavior+ecephys.nwb"
    _require(candidate.parent == source_directory and candidate.name.startswith(session),
             "source-only adapter attempted non-direct source path construction")
    _source_file_identity(candidate)
    return candidate


def _default_source_trial_reader(
    root: Path,
    nwb_path: Path,
    *,
    bin_size_ms: int,
    window_size: int,
    trial_result_filter: str,
) -> Sequence[Mapping[str, object]]:
    """Call exactly the established source trial filter, with no cache argument.

    The import is deliberately deferred until the explicit source-only audit.
    At module import/default dry-plan time this route does not import Torch,
    NWB, the datamodule, a model, or CUDA.
    """
    source_package = str(Path(root).absolute() / "sua_exploration")
    inserted = source_package not in sys.path
    if inserted:
        sys.path.insert(0, source_package)
    try:
        from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
        return list_datamodule_rewarded_trials(
            nwb_path,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
        )
    finally:
        if inserted:
            try:
                sys.path.remove(source_package)
            except ValueError as error:  # An unrelated mutator is unsafe here.
                raise ContractError("source-only adapter sys.path identity drift") from error


def _exact_nonnegative_int(value: object, label: str) -> int:
    """Accept only exact integral trial-bin coordinates from the source reader."""
    if type(value) is int:
        result = value
    elif type(value) is float and value.is_integer():
        result = int(value)
    else:
        raise ContractError(f"{label} is not an exact integer bin coordinate: {value!r}")
    _require(result >= 0, f"{label} is negative")
    return result


def valid_starts_from_rewarded_trials(
    trials: Sequence[Mapping[str, object]],
    *,
    window_size: int = 50,
) -> tuple[int, ...]:
    """Mirror the predecessor's valid-window row construction exactly.

    This uses chronological trial records returned by the established source
    filter.  It does not load behavior labels, unit-side features, T4, a
    calibration pool, or any validation/external/formal session.
    """
    _require(type(window_size) is int and window_size == 50,
             "source-only audit window size must be the sealed 50 bins")
    starts: list[int] = []
    for trial_number, trial in enumerate(trials):
        trial_map = _mapping(trial, f"source-only trial {trial_number}")
        start = _exact_nonnegative_int(trial_map.get("start"), f"trial {trial_number} start")
        stop = _exact_nonnegative_int(trial_map.get("stop"), f"trial {trial_number} stop")
        _require(stop - start >= window_size,
                 f"source-only reader yielded an underlength rewarded trial {trial_number}")
        starts.extend(range(start, stop - window_size + 1))
    _require(bool(starts), "source-only reader yielded no valid source windows")
    return tuple(starts)


def _audit_rng_snapshot_for_default_reader() -> Callable[[], Mapping[str, str]]:
    """Bind Python/NumPy/Torch CPU RNG state after the explicit source import."""
    numpy_module = sys.modules.get("numpy")
    torch_module = sys.modules.get("torch")
    _require(numpy_module is not None and torch_module is not None,
             "established source reader did not import its expected CPU dependencies")
    return lambda: rng_snapshot(numpy_module=numpy_module, torch_module=torch_module)


def _imbalance_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    _require(bool(rows), "cannot summarize empty source exposure table")
    source_batches = [int(row["full_batches_predecessor"]) for row in rows]
    eligible_windows = [int(row["eligible_windows"]) for row in rows]
    ratios = [float(row["equal_vs_window_proportional_batch_ratio"]) for row in rows]
    return {
        "schema": "cell_d_equal_session_source_imbalance_v1",
        "predecessor_full_batches_min": min(source_batches),
        "predecessor_full_batches_max": max(source_batches),
        "predecessor_full_batch_max_minus_min": max(source_batches) - min(source_batches),
        "predecessor_eligible_windows_min": min(eligible_windows),
        "predecessor_eligible_windows_max": max(eligible_windows),
        "predecessor_eligible_window_max_minus_min": max(eligible_windows) - min(eligible_windows),
        "equal_vs_predecessor_batch_ratio_min": min(ratios),
        "equal_vs_predecessor_batch_ratio_max": max(ratios),
    }


def source_only_window_audit(
    root: Path,
    *,
    trial_reader: Callable[..., Sequence[Mapping[str, object]]] | None = None,
    schedule_rng_snapshot: Callable[[], Mapping[str, str]] | None = None,
) -> dict[str, object]:
    """Measure source-only strict-27 window/exposure facts without writing.

    This is the sole function in the additive route allowed to open source NWB
    files.  Its direct child paths are the 27 manifest ``train`` sessions only;
    no validation, target, external, formal, cache, checkpoint, model, T4, or
    CUDA path is constructed.  It returns in-memory evidence for reviewer
    inspection and intentionally never creates a result directory or receipt.
    """
    root = Path(root).absolute()
    require_public_spec(PUBLIC_SPEC)
    predecessor = load_sealed_cell_d_contract(root)
    roster = load_strict_source_roster(root)
    _exact(len(roster), PUBLIC_SPEC.roster_size, "source-only strict train roster size")

    default_reader = trial_reader is None
    session_to_starts: dict[str, tuple[int, ...]] = {}
    source_file_identities: dict[str, dict[str, int]] = {}
    source_paths_opened: list[str] = []
    for session in roster:
        source_path = _direct_source_file(root, session, allowed_train_roster=roster)
        before_identity = _source_file_identity(source_path)
        reader = _default_source_trial_reader if default_reader else trial_reader
        assert reader is not None  # Established immediately above; helps static readers.
        trials = reader(
            root,
            source_path,
            bin_size_ms=20,
            window_size=50,
            trial_result_filter="R",
        ) if default_reader else reader(
            source_path,
            bin_size_ms=20,
            window_size=50,
            trial_result_filter="R",
        )
        _require(isinstance(trials, Sequence), f"{session}: source trial reader returned non-sequence")
        after_identity = _source_file_identity(source_path)
        _require(before_identity == after_identity, f"{session}: source file identity changed during audit")
        session_to_starts[session] = valid_starts_from_rewarded_trials(trials, window_size=50)
        source_file_identities[session] = {
            "device": before_identity[0],
            "inode": before_identity[1],
            "bytes": before_identity[2],
            "mtime_ns": before_identity[3],
            "mode": before_identity[4],
        }
        source_paths_opened.append(str(source_path.relative_to(root)))

    inventory = inventory_from_window_counts(roster, session_to_starts)
    _exact(inventory.eligible_windows, SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS,
           "strict-27 source eligible-window authority")
    _exact(inventory.full_batches(PUBLIC_SPEC) * PUBLIC_SPEC.batch_size,
           SOURCE_ONLY_EXPECTED_PREDECESSOR_CONSUMED_WINDOWS,
           "strict-27 predecessor consumed-window authority")
    _exact(inventory.remainder_windows(PUBLIC_SPEC), SOURCE_ONLY_EXPECTED_REMAINDER_WINDOWS,
           "strict-27 per-session floor-batch remainder authority")
    _exact(inventory.eligible_windows - inventory.full_batches(PUBLIC_SPEC) * PUBLIC_SPEC.batch_size,
           SOURCE_ONLY_EXPECTED_REMAINDER_WINDOWS,
           "strict-27 lost-window arithmetic")

    snapshot = schedule_rng_snapshot
    if snapshot is None:
        snapshot = _audit_rng_snapshot_for_default_reader() if default_reader else lambda: rng_snapshot()
    plan = build_plan_evidence(spec=PUBLIC_SPEC, inventory=inventory, snapshot_rng=snapshot)
    rows = original_vs_equal_exposure(PUBLIC_SPEC, inventory, plan)
    _require(sum(int(row["full_batches_predecessor"]) for row in rows)
             == inventory.full_batches(PUBLIC_SPEC), "source-only original exposure row sum drift")
    _require(sum(int(row["equal_batches_full_run"]) for row in rows) == PUBLIC_SPEC.total_steps,
             "source-only equal exposure row sum drift")
    _require(max(int(row["equal_batches_full_run"]) for row in rows)
             - min(int(row["equal_batches_full_run"]) for row in rows) <= 1,
             "source-only equal full-run exposure imbalance drift")

    return {
        "schema": "cell_d_equal_session_source_only_audit_v1",
        "cell": CELL,
        "source_only": True,
        "write_performed": False,
        "predecessor_authority": predecessor,
        "manifest_sha256": MANIFEST_SHA256,
        "strict_train_roster": list(roster),
        "source_paths_opened": source_paths_opened,
        "source_file_identities": source_file_identities,
        "source_inventory": {
            **inventory.payload(PUBLIC_SPEC),
            "predecessor_consumed_windows": inventory.full_batches(PUBLIC_SPEC) * PUBLIC_SPEC.batch_size,
        },
        "schedule_plan": plan,
        "per_session_exposure": rows,
        "current_source_imbalance": _imbalance_summary(rows),
        "access_disclosure": {
            "source_train_opened": True,
            "within_dev_resolved_or_opened": False,
            "external_resolved_or_opened": False,
            "formal_or_organizer_held_resolved_or_opened": False,
            "target_resolved_or_opened": False,
            "checkpoint_or_model_opened": False,
            "t4_or_calibration_features_constructed": False,
            "cache_read_or_write": False,
            "cuda_initialized": False,
            "optimizer_or_backward_or_update": False,
        },
    }


# ---------------------------------------------------------------------------
# Reviewed execution lifecycle
# ---------------------------------------------------------------------------

# The code below is deliberately route-owned.  Importantly, it remains pure
# standard-library code until a reviewed capability reaches the physical
# backend.  This lets the public CLI, static checks, and mock lifecycle tests
# remain genuinely no-Torch/no-data/no-CUDA/no-write.


@dataclass(frozen=True)
class LifecycleSpec:
    """Immutable topology for either the one-step smoke or full Cell-D run."""

    kind: str
    root_relative: str
    epochs: int
    steps_per_epoch: int
    checkpoint_epochs: tuple[int, ...]
    throughput_steps: int

    def __post_init__(self) -> None:
        _require(self.kind in {"source_smoke", "full_train", "synthetic"}, "lifecycle kind drift")
        _safe_relative(self.root_relative)
        _require(type(self.epochs) is int and self.epochs > 0, "lifecycle epoch count drift")
        _require(type(self.steps_per_epoch) is int and self.steps_per_epoch > 0,
                 "lifecycle step count drift")
        _require(type(self.throughput_steps) is int and 0 < self.throughput_steps <= self.total_steps,
                 "lifecycle throughput point drift")
        _require(tuple(sorted(set(self.checkpoint_epochs))) == self.checkpoint_epochs,
                 "lifecycle checkpoint order/duplicates drift")
        _require(all(type(epoch) is int and 0 <= epoch < self.epochs for epoch in self.checkpoint_epochs),
                 "lifecycle checkpoint epoch drift")
        if self.kind == "source_smoke":
            _require((self.epochs, self.steps_per_epoch, self.checkpoint_epochs, self.throughput_steps)
                     == (1, 1, (), 1), "source smoke must be one exact epoch-0 B32 optimizer step")
        elif self.kind == "full_train":
            _require((self.epochs, self.steps_per_epoch, self.checkpoint_epochs, self.throughput_steps)
                     == (48, 33_925, (44, 45, 46, 47), 100),
                     "full lifecycle differs from sealed Cell-D budget/topology")

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def topology(self) -> tuple[str, ...]:
        common = ("attempt.json", "launch.json", "source_authority.json", "terminal.json", "failure.json")
        if self.kind == "source_smoke":
            return common + ("smoke.json",)
        return common + tuple(f"epoch{epoch:03d}.json" for epoch in range(self.epochs)) + tuple(
            f"epoch{epoch:03d}.pt" for epoch in self.checkpoint_epochs
        ) + (f"throughput{self.throughput_steps}.json", "swa_final4.pt")

    def payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "root_relative": self.root_relative,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "checkpoint_epochs": list(self.checkpoint_epochs),
            "throughput_steps": self.throughput_steps,
        }


FULL_TRAIN_SPEC = LifecycleSpec(
    kind="full_train",
    root_relative=RESULT_ROOT_RELATIVE,
    epochs=48,
    steps_per_epoch=33_925,
    checkpoint_epochs=(44, 45, 46, 47),
    throughput_steps=100,
)
SOURCE_SMOKE_SPEC = LifecycleSpec(
    kind="source_smoke",
    root_relative=SMOKE_RESULT_ROOT_RELATIVE,
    epochs=1,
    steps_per_epoch=1,
    checkpoint_epochs=(),
    throughput_steps=1,
)


def _require_public_lifecycle(spec: LifecycleSpec) -> None:
    _require(spec in (FULL_TRAIN_SPEC, SOURCE_SMOKE_SPEC),
             "only the frozen equal-session source-smoke/full lifecycle specs are authorized")
    if spec == FULL_TRAIN_SPEC:
        require_public_spec(PUBLIC_SPEC)
        _require(spec.total_steps == PUBLIC_SPEC.total_steps, "full lifecycle/public schedule total drift")


def _directory_identity(path: Path) -> tuple[int, int]:
    metadata = os.lstat(path)
    _require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
             "artifact root/parent is not a direct directory")
    return metadata.st_dev, metadata.st_ino


def _write_full(descriptor: int, body: bytes) -> None:
    view = memoryview(body)
    while view:
        wrote = os.write(descriptor, view)
        if wrote <= 0:
            raise ContractError("short immutable artifact write")
        view = view[wrote:]


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1 << 20)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


@dataclass(frozen=True)
class ArtifactRoot:
    """A named-directory capability for O_EXCL/fsync/0444 receipt pairs."""

    directory: Path
    topology: tuple[str, ...]
    identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]

    def _assert_named_identity(self) -> None:
        _require(_directory_identity(self.directory) == self.identity, "artifact root path identity drift")
        parent_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            parent_info = os.fstat(parent_fd)
            _require((parent_info.st_dev, parent_info.st_ino) == self.parent_identity,
                     "artifact root parent identity drift")
            named = os.stat(self.directory.name, dir_fd=parent_fd, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode) and not stat.S_ISLNK(named.st_mode)
                     and (named.st_dev, named.st_ino) == self.identity,
                     "artifact named-root identity drift")
        finally:
            os.close(parent_fd)

    def _check_name(self, name: str) -> None:
        _require(isinstance(name, str) and name in self.topology and "/" not in name
                 and name not in {"", ".", ".."}, "artifact name lies outside fixed topology")

    def has_name(self, name: str) -> bool:
        self._check_name(name)
        self._assert_named_identity()
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            try:
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                return False
            return True
        finally:
            os.close(directory_fd)

    def publish_bytes(self, name: str, body: bytes) -> str:
        """Atomically publish one immutable body+sidecar pair, or roll it back."""
        self._check_name(name)
        _require(isinstance(body, bytes), "artifact body must be bytes")
        self._assert_named_identity()
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        made: list[tuple[str, int, int]] = []
        try:
            opened = os.fstat(directory_fd)
            _require((opened.st_dev, opened.st_ino) == self.identity,
                     "artifact root changed between named check/open")
            digest = _sha256(body)
            members = ((name, body), (name + ".sha256", f"{digest}  {name}\n".encode("ascii")))
            for leaf, contents in members:
                descriptor = os.open(
                    leaf,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    info = os.fstat(descriptor)
                    _require(stat.S_ISREG(info.st_mode), "artifact O_EXCL did not create a regular file")
                    made.append((leaf, info.st_dev, info.st_ino))
                    _write_full(descriptor, contents)
                    os.fchmod(descriptor, 0o444)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            os.fsync(directory_fd)
            for leaf, expected in members:
                descriptor = os.open(leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
                try:
                    info = os.fstat(descriptor)
                    observed = _read_all(descriptor)
                    _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444
                             and observed == expected, "artifact same-FD reload/mode proof failed")
                finally:
                    os.close(descriptor)
            self._assert_named_identity()
            parent_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            return digest
        except BaseException:
            for leaf, device, inode in reversed(made):
                try:
                    now = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
                    if (now.st_dev, now.st_ino) == (device, inode):
                        os.unlink(leaf, dir_fd=directory_fd)
                except OSError:
                    pass
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            raise
        finally:
            os.close(directory_fd)

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        return self.publish_bytes(name, _canonical_json_bytes(dict(payload)))

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        self._check_name(name)
        self._assert_named_identity()
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            def read(leaf: str) -> bytes:
                descriptor = os.open(leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
                try:
                    info = os.fstat(descriptor)
                    _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444,
                             "artifact pair mode/type drift")
                    return _read_all(descriptor)
                finally:
                    os.close(descriptor)
            body = read(name)
            digest = _sha256(body)
            _require(expected_sha256 is None or digest == expected_sha256, "artifact pair body SHA drift")
            _require(read(name + ".sha256") == f"{digest}  {name}\n".encode("ascii"),
                     "artifact pair sidecar drift")
            self._assert_named_identity()
            return body
        finally:
            os.close(directory_fd)

    def reload_json(self, name: str, expected_sha256: str | None = None) -> Mapping[str, object]:
        try:
            value = json.loads(self.reload_pair(name, expected_sha256))
        except (TypeError, json.JSONDecodeError) as error:
            raise ContractError("artifact JSON decode drift") from error
        _require(isinstance(value, Mapping), "artifact JSON root is not a mapping")
        return value


def reserve_artifact_root(root: Path, relative: str, topology: tuple[str, ...]) -> ArtifactRoot:
    """Reserve a fresh independent result root before source/CUDA imports."""
    _safe_relative(relative)
    _require(isinstance(topology, tuple) and topology and len(set(topology)) == len(topology),
             "artifact topology drift")
    _require(all(isinstance(name, str) and name and "/" not in name for name in topology),
             "artifact topology leaf drift")
    root = Path(root).absolute()
    target = root / relative
    _require(not target.exists() and not target.is_symlink(), "fresh equal-session result root required")
    parent = target.parent
    parent_identity = _directory_identity(parent)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(parent_fd)
        _require((opened.st_dev, opened.st_ino) == parent_identity, "artifact parent changed before reservation")
        os.mkdir(target.name, 0o755, dir_fd=parent_fd)
        os.fsync(parent_fd)
        metadata = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        _require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
                 "reserved artifact root is not a direct directory")
        identity = (metadata.st_dev, metadata.st_ino)
    except FileExistsError as error:
        raise ContractError("equal-session canonical output collision") from error
    finally:
        os.close(parent_fd)
    return ArtifactRoot(target, topology, identity, parent, parent_identity)


@dataclass(frozen=True)
class RunIdentity:
    """All launch-bound, non-result authorities for a reviewed lifecycle."""

    predecessor: Mapping[str, object]
    closure: Mapping[str, object]
    source_schedule: Mapping[str, object]
    canonical_initial_state: Mapping[str, object]
    failed_smoke_v1_lineage: Mapping[str, object]
    gpu: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        return {
            "predecessor": dict(self.predecessor),
            "closure": dict(self.closure),
            "source_schedule": dict(self.source_schedule),
            "canonical_initial_state": dict(self.canonical_initial_state),
            "failed_smoke_v1_lineage": dict(self.failed_smoke_v1_lineage),
            "gpu": dict(self.gpu),
        }


def _build_run_identity(root: Path) -> RunIdentity:
    predecessor = load_sealed_cell_d_contract(root)
    arm_a_source = load_sealed_arm_a_source_authority(root)
    # Bind the immutable v1 failure before capability validation or any fresh
    # successor-root reservation.  This is receipt metadata only: no source
    # NWB, model, CUDA, or output mutation occurs here.
    failed_smoke_v1_lineage = load_failed_smoke_v1_lineage(root)
    _exact(arm_a_source["roster"], list(load_strict_source_roster(root)),
           "sealed Arm-A/strict-manifest train roster order")
    closure = implementation_closure(root)
    _require(closure.get("schema") == "cell_d_equal_session_implementation_closure_v1",
             "equal-session closure schema drift")
    source_schedule = {
        "strict_manifest_sha256": MANIFEST_SHA256,
        "plan_sha256": AUDITED_FULL_PLAN_SHA256,
        "epoch0_schedule_sha256": AUDITED_EPOCH0_SCHEDULE_SHA256,
        "epoch47_schedule_sha256": AUDITED_EPOCH47_SCHEDULE_SHA256,
        "source_only_expected_eligible_windows": SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS,
        "source_only_expected_predecessor_consumed_windows": SOURCE_ONLY_EXPECTED_PREDECESSOR_CONSUMED_WINDOWS,
        "source_only_expected_remainder_windows": SOURCE_ONLY_EXPECTED_REMAINDER_WINDOWS,
        "spec": PUBLIC_SPEC.payload(),
        "sealed_arm_a_source_authority": arm_a_source,
    }
    return RunIdentity(
        predecessor=predecessor,
        closure=closure,
        source_schedule=source_schedule,
        canonical_initial_state={
            "relative": CANONICAL_INITIAL_STATE_RELATIVE,
            "artifact_sha256": CANONICAL_INITIAL_STATE_SHA256,
            "state_dict_sha256": CANONICAL_INITIAL_STATE_STATE_SHA256,
            "mode": "0444",
            "strict_load": True,
            "initialized_trainable_parameters": 3_510_842,
            "uninitialized_lazy_keys": ["decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight"],
        },
        failed_smoke_v1_lineage=failed_smoke_v1_lineage,
        gpu=dict(FROZEN_GPU0),
    )


def _validate_run_identity(identity: RunIdentity, spec: LifecycleSpec) -> None:
    _require(isinstance(identity, RunIdentity), "run identity type drift")
    predecessor = identity.predecessor
    _require(predecessor.get("cell") == "D" and predecessor.get("trainable_parameters") == 3_510_842,
             "sealed Cell-D predecessor graph/count drift")
    closure = identity.closure
    _require(_is_sha256(closure.get("closure_sha256"))
             and closure.get("paths") == list(IMPLEMENTATION_CLOSURE),
             "implementation closure identity drift")
    source = identity.source_schedule
    _exact(source.get("strict_manifest_sha256"), MANIFEST_SHA256, "run identity manifest")
    _exact(source.get("plan_sha256"), AUDITED_FULL_PLAN_SHA256, "run identity full schedule")
    _exact(source.get("epoch0_schedule_sha256"), AUDITED_EPOCH0_SCHEDULE_SHA256,
           "run identity epoch-0 schedule")
    _exact(source.get("epoch47_schedule_sha256"), AUDITED_EPOCH47_SCHEDULE_SHA256,
           "run identity epoch-47 schedule")
    _exact(source.get("source_only_expected_eligible_windows"), SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS,
           "run identity strict-27 eligible-window authority")
    _exact(source.get("source_only_expected_predecessor_consumed_windows"),
           SOURCE_ONLY_EXPECTED_PREDECESSOR_CONSUMED_WINDOWS,
           "run identity predecessor consumed-window authority")
    _exact(source.get("source_only_expected_remainder_windows"), SOURCE_ONLY_EXPECTED_REMAINDER_WINDOWS,
           "run identity strict-27 remainder authority")
    _exact(source.get("spec"), PUBLIC_SPEC.payload(), "run identity public schedule spec")
    arm_a_source = source.get("sealed_arm_a_source_authority")
    _require(isinstance(arm_a_source, Mapping)
             and arm_a_source.get("preflight_receipt_sha256") == SEALED_ARM_A_PREFLIGHT_SHA256
             and arm_a_source.get("behavior_normalizer_semantic_sha256") == SEALED_BEHAVIOR_NORMALIZER_SHA256
             and arm_a_source.get("t4_normalizer_semantic_sha256") == SEALED_T4_NORMALIZER_SHA256
             and isinstance(arm_a_source.get("roster"), list)
             and len(arm_a_source["roster"]) == 27
             and len(set(arm_a_source["roster"])) == 27
             and isinstance(arm_a_source.get("t4_authority_sha256"), Mapping)
             and set(arm_a_source["t4_authority_sha256"]) == set(arm_a_source["roster"])
             and all(_is_sha256(item) for item in arm_a_source["t4_authority_sha256"].values()),
             "run identity sealed Arm-A source authority drift")
    initial = identity.canonical_initial_state
    _exact(initial.get("artifact_sha256"), CANONICAL_INITIAL_STATE_SHA256, "run identity canonical artifact")
    _exact(initial.get("state_dict_sha256"), CANONICAL_INITIAL_STATE_STATE_SHA256,
           "run identity canonical state")
    _exact(initial.get("initialized_trainable_parameters"), 3_510_842,
           "run identity initialized parameter count")
    _exact(initial.get("uninitialized_lazy_keys"), ["decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight"],
           "run identity lazy topology")
    _exact(
        dict(identity.failed_smoke_v1_lineage),
        _failed_smoke_v1_lineage_payload(),
        "run identity immutable smoke-v1 failure lineage",
    )
    _exact(dict(identity.gpu), dict(FROZEN_GPU0), "run identity fixed GPU0 contract")
    if spec.kind == "full_train":
        _require_public_lifecycle(spec)


_CAPABILITY_SEAL = object()


@dataclass(frozen=True)
class RootReviewedExecutionCapability:
    """In-process root-review capability; public CLI cannot manufacture one."""

    authorization_sha256: str
    closure_sha256: str
    route: str
    kind: str
    _seal: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        _require(_is_sha256(self.authorization_sha256), "root-review authorization digest drift")
        _require(_is_sha256(self.closure_sha256), "root-review closure digest drift")
        _require(self.route == CELL and self.kind in {"source_smoke", "full_train"},
                 "root-review route/kind drift")
        _require(self._seal is _CAPABILITY_SEAL, "root-review capability was not issued in-process")


def _issue_root_review_capability_for_audited_route(
    root: Path,
    *,
    authorization_sha256: str,
    spec: LifecycleSpec,
) -> RootReviewedExecutionCapability:
    """Private root-only issuance point; it writes nothing and opens no data."""
    _require_public_lifecycle(spec)
    identity = _build_run_identity(Path(root).absolute())
    _validate_run_identity(identity, spec)
    return RootReviewedExecutionCapability(
        authorization_sha256=authorization_sha256,
        closure_sha256=str(identity.closure["closure_sha256"]),
        route=CELL,
        kind=spec.kind,
        _seal=_CAPABILITY_SEAL,
    )


def _validate_capability(capability: RootReviewedExecutionCapability | None, identity: RunIdentity,
                         spec: LifecycleSpec) -> None:
    _require(isinstance(capability, RootReviewedExecutionCapability),
             "future equal-session execution lacks an in-process root capability")
    _require(capability.route == CELL and capability.kind == spec.kind,
             "root-review capability route/kind mismatch")
    _exact(capability.closure_sha256, identity.closure.get("closure_sha256"),
           "root-review capability/current closure")


def assert_fresh_candidate_roots(root: Path, *, required_roots: Sequence[str] | None = None) -> dict[str, object]:
    """Prove designated future roots are absent/non-symlink; never create them.

    The default checks both roots for preflight.  An authorized full run checks
    only its own root, so a completed non-authorizing smoke receipt cannot
    silently block the separate full-run lineage.
    """
    root = Path(root).absolute()
    relatives = tuple(required_roots) if required_roots is not None else (
        RESULT_ROOT_RELATIVE, SMOKE_RESULT_ROOT_RELATIVE,
    )
    _require(bool(relatives) and all(relative in {RESULT_ROOT_RELATIVE, SMOKE_RESULT_ROOT_RELATIVE}
                                    for relative in relatives)
             and len(set(relatives)) == len(relatives), "fresh-root request topology drift")
    checked: dict[str, str] = {}
    for relative in relatives:
        _safe_relative(relative)
        candidate = root / relative
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            checked[relative] = "absent"
            continue
        except OSError as error:
            raise ContractError(f"cannot inspect prospective result root: {relative}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError(f"prospective result root is an alias/symlink: {relative}")
        raise ContractError(f"prospective result root already exists: {relative}")
    return {"schema": "cell_d_equal_session_fresh_roots_v1", "roots": checked, "created": False}


def validate_future_gpu0_environment(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Validate literal future visibility without importing Torch or CUDA."""
    environment = os.environ if environ is None else environ
    _exact(environment.get("CUDA_VISIBLE_DEVICES"), FROZEN_GPU0["cuda_visible_devices"],
           "future equal-session CUDA visibility")
    _exact(environment.get("CUDA_DEVICE_ORDER"), FROZEN_GPU0["cuda_device_order"],
           "future equal-session CUDA device order")
    _require(sys.flags.no_user_site == 1, "future equal-session route requires no-user-site interpreter")
    return {
        "CUDA_VISIBLE_DEVICES": FROZEN_GPU0["cuda_visible_devices"],
        "CUDA_DEVICE_ORDER": FROZEN_GPU0["cuda_device_order"],
        "logical_device": FROZEN_GPU0["logical_device"],
        "no_user_site": True,
        "gpu_authorities": dict(FROZEN_GPU0),
    }


def dry_plan() -> dict[str, object]:
    """Public default: no data/no model/no CUDA/no write/no capability issuance."""
    require_public_spec(PUBLIC_SPEC)
    return {
        "schema": "cell_d_equal_session_dry_plan_v2",
        "cell": CELL,
        "status": "DRY_NO_DATA_NO_WRITE",
        "changed_factor": "window-proportional source-session exposure -> deterministic equal-session B32 schedule",
        "frozen_budget": PUBLIC_SPEC.payload(),
        "source_smoke": SOURCE_SMOKE_SPEC.payload(),
        "full_train": FULL_TRAIN_SPEC.payload(),
        "future_execution_contract": {
            "full_flags": ["--execute", "--i-have-root-reviewed-equal-session-authorization"],
            "smoke_flags": ["--source-smoke", "--i-have-root-reviewed-equal-session-smoke-authorization"],
            "requires_in_process_root_capability": True,
            "requires_fresh_independent_roots": [RESULT_ROOT_RELATIVE, SMOKE_RESULT_ROOT_RELATIVE],
            "immutable_failed_smoke_v1_lineage": _failed_smoke_v1_lineage_payload(),
            "physical_gpu0": dict(FROZEN_GPU0),
            "physical_runtime_versions": dict(FROZEN_RUNTIME_VERSIONS),
            "data_access_before_root_review": False,
            "cuda_before_root_review": False,
            "result_publication_before_root_review": False,
            "epoch_binding": "EpochLedger claim before one explicit DataLoader iterator per epoch",
            "worker_seed_policy": "no sampler/worker-init RNG draw beyond predecessor iterator-level worker seeding",
        },
        "not_a_launch_authorization": True,
    }


@dataclass
class LifecycleFlags:
    source_train_opened: bool = False
    within_dev_resolved_or_opened: bool = False
    external_resolved_or_opened: bool = False
    formal_or_organizer_held_resolved_or_opened: bool = False
    target_resolved_or_opened: bool = False
    cache_read_or_write: bool = False
    cuda_initialized: bool = False
    optimizer_steps_completed: int = 0
    backward_calls: int = 0
    update_calls: int = 0
    terminal_published: bool = False
    stage: str = "identity"
    data_loader_constructor_rng_unchanged: bool = True
    data_loader_iterators_created: int = 0
    fixed_diagnostic_forwards: int = 0

    def disclosure(self) -> dict[str, object]:
        return {
            "source_train_opened": self.source_train_opened,
            "within_dev_resolved_or_opened": self.within_dev_resolved_or_opened,
            "external_resolved_or_opened": self.external_resolved_or_opened,
            "formal_or_organizer_held_resolved_or_opened": self.formal_or_organizer_held_resolved_or_opened,
            "target_resolved_or_opened": self.target_resolved_or_opened,
            "cache_read_or_write": self.cache_read_or_write,
            "cuda_initialized": self.cuda_initialized,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "backward_calls": self.backward_calls,
            "update_calls": self.update_calls,
            "data_loader_constructor_rng_unchanged": self.data_loader_constructor_rng_unchanged,
            "data_loader_iterators_created": self.data_loader_iterators_created,
            "fixed_diagnostic_forwards": self.fixed_diagnostic_forwards,
        }


@dataclass(frozen=True)
class StepOutcome:
    loss: float
    lr: float
    dropout_p: float
    kept: int
    dropped: int
    all_zero_examples: int
    population_examples: int
    max_gain: float
    # These expensive full-state proofs are required for source-smoke and for
    # the final optimizer step of each full epoch only.  Ordinary steps must
    # carry ``None`` rather than fabricated success evidence.
    critical_gradients: Mapping[str, bool] | None
    finite_model: bool | None
    finite_optimizer: bool | None
    model_state_sha256: str | None
    optimizer_state_sha256: str | None
    batch_evidence: Mapping[str, object]


@dataclass(frozen=True)
class CheckpointPayload:
    body: bytes
    model_state_sha256: str


@dataclass(frozen=True)
class SWAPayload:
    body: bytes
    state_sha256: str
    proof: Mapping[str, object]


class TrainingBackend(Protocol):
    """Injected backend.  Physical source/CUDA work stays behind this line."""

    def prepare(self, spec: LifecycleSpec, identity: RunIdentity, flags: LifecycleFlags) -> Any: ...
    def source_authority(self, runtime: Any, spec: LifecycleSpec, identity: RunIdentity,
                         flags: LifecycleFlags) -> Mapping[str, object]: ...
    def begin_epoch(self, runtime: Any, epoch: int, flags: LifecycleFlags) -> Mapping[str, object]: ...
    def train_step(self, runtime: Any, *, global_step: int, expected_lr: float, require_full_proof: bool,
                   flags: LifecycleFlags) -> StepOutcome: ...
    def end_epoch(self, runtime: Any, epoch: int, outcomes: Sequence[StepOutcome],
                  flags: LifecycleFlags) -> Mapping[str, object]: ...
    def synchronize_for_measurement(self, runtime: Any) -> None: ...
    def resources(self, runtime: Any) -> Mapping[str, object]: ...
    def make_checkpoint(self, runtime: Any, epoch: int, global_step: int,
                        binding: Mapping[str, object]) -> CheckpointPayload: ...
    def validate_checkpoint(self, body: bytes, *, epoch: int, global_step: int,
                            spec: LifecycleSpec, binding: Mapping[str, object]) -> Mapping[str, object]: ...
    def build_swa(self, runtime: Any, checkpoints: Mapping[int, bytes], spec: LifecycleSpec,
                  binding: Mapping[str, object]) -> SWAPayload: ...
    def validate_swa(self, body: bytes, *, spec: LifecycleSpec,
                     binding: Mapping[str, object]) -> Mapping[str, object]: ...
    def close(self, runtime: Any | None) -> None: ...


def _finite_float(value: object, *, nonnegative: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    rendered = float(value)
    return rendered == rendered and abs(rendered) != float("inf") and (not nonnegative or rendered >= 0.0)


def _critical_gradient_contract_payload() -> dict[str, object]:
    """Receipt-stable map of the live Cell-D paths proved at a boundary."""
    return {
        "schema": "cell_d_equal_session_critical_gradients_v1",
        "required_paths": {key: dict(value) for key, value in CRITICAL_GRADIENT_PATHS.items()},
        "excluded_uninitialized_lazy_keys": list(CELL_D_UNINITIALIZED_LAZY_KEYS),
    }


def _validate_critical_gradient_proof(value: object) -> None:
    _require(
        isinstance(value, Mapping)
        and set(value) == set(CRITICAL_GRADIENT_PATHS)
        and all(item is True for item in value.values()),
        "critical gradient proof drift",
    )


def _validate_step_outcome(outcome: StepOutcome, expected_lr: float, *, require_full_proof: bool) -> None:
    _require(isinstance(outcome, StepOutcome), "backend did not return StepOutcome")
    _require(_finite_float(outcome.loss, nonnegative=True) and outcome.lr == expected_lr,
             "step loss/LR drift")
    _require(_finite_float(outcome.dropout_p, nonnegative=True) and 0.0 <= outcome.dropout_p <= 1.0,
             "step dropout probability drift")
    _require(all(type(item) is int and item >= 0 for item in (
        outcome.kept, outcome.dropped, outcome.all_zero_examples, outcome.population_examples,
    )), "step dropout counts drift")
    _require(outcome.population_examples > 0 and outcome.all_zero_examples <= outcome.population_examples
             and outcome.kept + outcome.dropped > 0, "step dropout denominator drift")
    _require(_finite_float(outcome.max_gain, nonnegative=True), "step max-gain drift")
    if require_full_proof:
        _validate_critical_gradient_proof(outcome.critical_gradients)
        _require(outcome.finite_model is True and outcome.finite_optimizer is True,
                 "step finite model/optimizer proof drift")
        _require(_is_sha256(outcome.model_state_sha256) and _is_sha256(outcome.optimizer_state_sha256),
                 "step state digest drift")
    else:
        _require(all(value is None for value in (
            outcome.critical_gradients, outcome.finite_model, outcome.finite_optimizer,
            outcome.model_state_sha256, outcome.optimizer_state_sha256,
        )), "ordinary step fabricated a full-state proof")
    _require(isinstance(outcome.batch_evidence, Mapping), "step batch evidence drift")


def _quantile(values: Sequence[float], fraction: float) -> float:
    _require(bool(values) and 0.0 <= fraction <= 1.0, "invalid quantile input")
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    low, high = int(location), int(location + 0.999999999999)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def _resource_payload(value: Mapping[str, object]) -> dict[str, object]:
    required = {
        "cuda_memory_allocated",
        "cuda_memory_reserved",
        "cuda_peak_memory_allocated",
        "cuda_peak_memory_reserved",
        "peak_stats_reset_before_training",
        "rss_bytes",
        "throughput_windows_per_s",
    }
    _require(isinstance(value, Mapping) and set(value) == required, "resource disclosure key drift")
    result: dict[str, object] = {}
    for key in required - {"peak_stats_reset_before_training"}:
        item = value[key]
        _require(type(item) is int and item >= 0, "resource disclosure value drift")
        result[key] = item
    _require(value["peak_stats_reset_before_training"] is True,
             "CUDA peak-memory reset proof drift")
    _require(
        int(result["cuda_peak_memory_allocated"]) >= int(result["cuda_memory_allocated"])
        and int(result["cuda_peak_memory_reserved"]) >= int(result["cuda_memory_reserved"]),
        "CUDA peak memory is below current allocation/reservation",
    )
    result["peak_stats_reset_before_training"] = True
    return result


def _epoch_payload(spec: LifecycleSpec, epoch: int, global_step: int, outcomes: Sequence[StepOutcome],
                   schedule: Mapping[str, object], end_detail: Mapping[str, object],
                   resources: Mapping[str, object]) -> dict[str, object]:
    _require(len(outcomes) == spec.steps_per_epoch and global_step == (epoch + 1) * spec.steps_per_epoch,
             "epoch exact step accounting drift")
    _require(isinstance(schedule, Mapping) and schedule.get("epoch") == epoch
             and _is_sha256(schedule.get("schedule_sha256")), "epoch schedule evidence drift")
    loss = [float(item.loss) for item in outcomes]
    probability = [float(item.dropout_p) for item in outcomes]
    kept, dropped = sum(item.kept for item in outcomes), sum(item.dropped for item in outcomes)
    population, all_zero = sum(item.population_examples for item in outcomes), sum(item.all_zero_examples for item in outcomes)
    _require(kept + dropped > 0 and population > 0, "epoch dropout aggregate denominator drift")
    last = outcomes[-1]
    _require(last.critical_gradients is not None and last.finite_model is True and last.finite_optimizer is True
             and _is_sha256(last.model_state_sha256) and _is_sha256(last.optimizer_state_sha256),
             "epoch terminal full-state proof absent")
    return {
        "schema": "cell_d_equal_session_epoch_v1",
        "cell": CELL,
        "epoch": epoch,
        "steps": spec.steps_per_epoch,
        "cumulative_optimizer_steps": global_step,
        "schedule": dict(schedule),
        "loss": {"mean": sum(loss) / len(loss), "min": min(loss), "max": max(loss)},
        "lr": {"first": outcomes[0].lr, "last": outcomes[-1].lr},
        "dropout": {
            "p_min": min(probability), "p_max": max(probability), "p_mean": sum(probability) / len(probability),
            "p_q25": _quantile(probability, 0.25), "p_q50": _quantile(probability, 0.50),
            "p_q75": _quantile(probability, 0.75), "kept": kept, "dropped": dropped,
            "kept_fraction": kept / (kept + dropped), "dropped_fraction": dropped / (kept + dropped),
            "all_zero_examples": all_zero, "population_examples": population,
            "all_zero_fraction": all_zero / population, "max_gain": max(item.max_gain for item in outcomes),
        },
        "critical_gradients": dict(last.critical_gradients),
        "critical_gradient_contract": _critical_gradient_contract_payload(),
        "finite": {"model": last.finite_model, "optimizer": last.finite_optimizer},
        "state": {"model_sha256": last.model_state_sha256, "optimizer_sha256": last.optimizer_state_sha256},
        "last_batch": dict(last.batch_evidence),
        "end_detail": dict(end_detail),
        "resources": _resource_payload(resources),
    }


def _validate_epoch_payload(value: Mapping[str, object], spec: LifecycleSpec, epoch: int) -> None:
    _require(value.get("schema") == "cell_d_equal_session_epoch_v1" and value.get("cell") == CELL
             and value.get("epoch") == epoch and value.get("steps") == spec.steps_per_epoch
             and value.get("cumulative_optimizer_steps") == (epoch + 1) * spec.steps_per_epoch,
             "epoch receipt topology drift")
    schedule = value.get("schedule")
    _require(isinstance(schedule, Mapping) and schedule.get("epoch") == epoch
             and _is_sha256(schedule.get("schedule_sha256")), "epoch receipt schedule drift")
    if spec == FULL_TRAIN_SPEC:
        _exact(schedule.get("full_plan_sha256"), AUDITED_FULL_PLAN_SHA256,
               "epoch receipt equal-session plan binding")
    finite = value.get("finite")
    state = value.get("state")
    _validate_critical_gradient_proof(value.get("critical_gradients"))
    _exact(value.get("critical_gradient_contract"), _critical_gradient_contract_payload(),
           "epoch critical-gradient contract")
    _require(isinstance(finite, Mapping) and finite == {"model": True, "optimizer": True}
             and isinstance(state, Mapping) and _is_sha256(state.get("model_sha256"))
             and _is_sha256(state.get("optimizer_sha256")), "epoch receipt final proof drift")
    _resource_payload(_mapping(value.get("resources"), "epoch resources"))


def _attempt_payload(spec: LifecycleSpec, identity: RunIdentity) -> dict[str, object]:
    return {
        "schema": "cell_d_equal_session_attempt_v1", "cell": CELL, "status": "ATTEMPT_RESERVED",
        "run_spec": spec.payload(), "identity": identity.payload(), "source_only": True,
        "target_or_formal_or_external_forbidden": True,
    }


def _launch_payload(spec: LifecycleSpec, identity: RunIdentity, attempt_sha256: str) -> dict[str, object]:
    return {
        "schema": "cell_d_equal_session_launch_v1", "cell": CELL, "status": "LAUNCHED",
        "run_spec": spec.payload(), "identity": identity.payload(), "attempt_sha256": attempt_sha256,
        "changed_factor": "deterministic equal-session source B32 schedule only",
        "held": {
            "sealed_cell_d_graph": True, "b3s_normalized_t4": True,
            "whole_token_dynamic_dropout_uniform_0_1": True, "dense_all_bin_loss": True,
            "adam": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False},
            "warmup_cosine": True, "final_four_swa": spec.kind != "source_smoke",
        },
        "data_loading_control_flow": {
            "predecessor_shared_builder": "_initialize_splits then clear val before setup(fit)",
            "predecessor_preclear_metadata_read": "shared _initialize_splits currently calls nwb_unit_count over train+val",
            "successor_train_only_repair": "bind sealed train paths; set val/test empty; sentinel-skip empty val load_split; mark splits initialized; setup(fit); restore val=None",
            "nontrain_path_or_nwb_metadata_open": False,
            "bytewise_identical_shared_builder_control_flow": False,
            "science_factor_changed": False,
        },
    }


def _validate_source_authority_payload(
    value: Mapping[str, object],
    spec: LifecycleSpec,
    identity: RunIdentity,
    *,
    allow_synthetic_runtime: bool = False,
) -> None:
    """Validate a source authority receipt.

    The public/default path accepts only the physical, throughput-sealed
    runtime environment.  Focused lifecycle tests use a deliberately
    different fixture schema and must opt into it at this call boundary; a
    synthetic payload can therefore never pass a production validator merely
    because its GPU-shaped fields happen to look plausible.
    """
    _require(value.get("schema") == "cell_d_equal_session_source_authority_v1"
             and value.get("cell") == CELL and value.get("run_spec") == spec.payload(),
             "source authority receipt schema/run-spec drift")
    access = value.get("access_disclosure")
    _require(isinstance(access, Mapping) and access.get("source_train_opened") is True
             and access.get("within_dev_resolved_or_opened") is False
             and access.get("external_resolved_or_opened") is False
             and access.get("formal_or_organizer_held_resolved_or_opened") is False
             and access.get("target_resolved_or_opened") is False
             and access.get("cache_read_or_write") is False,
             "source authority access boundary drift")
    if allow_synthetic_runtime:
        _validate_synthetic_runtime_environment_payload(value.get("runtime_environment"))
    else:
        _validate_runtime_environment_payload(value.get("runtime_environment"))
    loading = value.get("data_loading_control_flow")
    if spec in (FULL_TRAIN_SPEC, SOURCE_SMOKE_SPEC):
        _require(
            isinstance(loading, Mapping)
            and loading.get("nontrain_path_or_nwb_metadata_open") is False
            and loading.get("bytewise_identical_shared_builder_control_flow") is False
            and loading.get("science_factor_changed") is False,
            "source authority train-only access-order disclosure drift",
        )
    if spec in (FULL_TRAIN_SPEC, SOURCE_SMOKE_SPEC):
        sealed = identity.source_schedule["sealed_arm_a_source_authority"]
        roster = sealed["roster"]
        _exact(value.get("strict_train_roster"), roster, "source authority strict roster")
        expected_paths = [
            f"{SOURCE_DATA_RELATIVE}/{session}_behavior+ecephys.nwb" for session in roster
        ]
        _exact(value.get("train_source_relative_paths"), expected_paths,
               "source authority direct strict-train paths")
        inventory = value.get("source_inventory")
        plan = value.get("schedule_plan")
        normalizers = value.get("normalizers")
        exposure = value.get("per_session_exposure")
        t4 = value.get("t4_authority_sha256")
        sessions = inventory.get("sessions") if isinstance(inventory, Mapping) else None
        plan_inventory = dict(inventory) if isinstance(inventory, Mapping) else None
        if isinstance(plan_inventory, dict):
            plan_inventory.pop("predecessor_consumed_windows", None)
        inventory_topology = False
        if isinstance(sessions, list) and len(sessions) == len(roster):
            expected_index = 0
            inventory_topology = True
            for session, item in zip(roster, sessions, strict=True):
                if not isinstance(item, Mapping):
                    inventory_topology = False
                    break
                eligible = item.get("eligible_windows")
                full_batches = item.get("full_batches")
                remainder = item.get("remainder_windows")
                if not (
                    item.get("session") == session
                    and type(eligible) is int and eligible > 0
                    and type(full_batches) is int and full_batches > 0
                    and type(remainder) is int and 0 <= remainder < PUBLIC_SPEC.batch_size
                    and eligible == full_batches * PUBLIC_SPEC.batch_size + remainder
                    and item.get("dataset_index_min") == expected_index
                    and item.get("dataset_index_max") == expected_index + eligible - 1
                    and _is_sha256(item.get("valid_starts_sha256"))
                ):
                    inventory_topology = False
                    break
                expected_index += eligible
            inventory_topology = inventory_topology and expected_index == SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS
        epochs = plan.get("epochs") if isinstance(plan, Mapping) else None
        epoch_topology = (
            isinstance(epochs, list)
            and len(epochs) == 48
            and all(
                isinstance(item, Mapping) and item.get("epoch") == index
                and _is_sha256(item.get("schedule_sha256")) and "batches" not in item
                for index, item in enumerate(epochs)
            )
        )
        _require(
            isinstance(inventory, Mapping)
            and inventory.get("roster") == roster
            and inventory.get("eligible_windows") == SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS
            and inventory.get("full_batches") == PUBLIC_SPEC.steps_per_epoch
            and inventory.get("predecessor_consumed_windows") == SOURCE_ONLY_EXPECTED_PREDECESSOR_CONSUMED_WINDOWS
            and inventory.get("remainder_windows") == SOURCE_ONLY_EXPECTED_REMAINDER_WINDOWS
            and inventory_topology
            and sum(item["full_batches"] for item in sessions) == PUBLIC_SPEC.steps_per_epoch
            and sum(item["remainder_windows"] for item in sessions) == SOURCE_ONLY_EXPECTED_REMAINDER_WINDOWS
            and isinstance(plan, Mapping)
            and plan.get("schema") == "cell_d_equal_session_full_plan_v1"
            and plan.get("spec") == PUBLIC_SPEC.payload()
            and plan.get("inventory") == plan_inventory
            and plan.get("full_plan_sha256") == AUDITED_FULL_PLAN_SHA256
            and plan.get("plan_retains_batch_lists") is False
            and plan.get("global_rng_unchanged") is True
            and epoch_topology
            and isinstance(epochs[0], Mapping)
            and isinstance(epochs[47], Mapping)
            and epochs[0].get("schedule_sha256") == AUDITED_EPOCH0_SCHEDULE_SHA256
            and epochs[47].get("schedule_sha256") == AUDITED_EPOCH47_SCHEDULE_SHA256
            and plan.get("full_run_batches_by_session") == full_run_exposure(PUBLIC_SPEC, roster)
            and plan.get("full_run_max_minus_min_batches") == 1
            and isinstance(exposure, list)
            and len(exposure) == 27
            and [item.get("session") if isinstance(item, Mapping) else None for item in exposure] == roster
            and all(
                isinstance(row, Mapping)
                and row.get("eligible_windows") == session["eligible_windows"]
                and row.get("full_batches_predecessor") == session["full_batches"]
                and row.get("remainder_windows_discarded_per_epoch") == session["remainder_windows"]
                and row.get("equal_batches_full_run") == plan["full_run_batches_by_session"].get(row.get("session"))
                for row, session in zip(exposure, sessions, strict=True)
            )
            and isinstance(normalizers, Mapping)
            and normalizers.get("behavior_semantic_sha256") == SEALED_BEHAVIOR_NORMALIZER_SHA256
            and normalizers.get("t4_semantic_sha256") == SEALED_T4_NORMALIZER_SHA256
            and t4 == sealed["t4_authority_sha256"],
            "source authority inventory/plan/normalizer/T4 drift",
        )



def _validate_attempt_payload(value: Mapping[str, object], spec: LifecycleSpec, identity: RunIdentity) -> None:
    _require(value.get("schema") == "cell_d_equal_session_attempt_v1" and value.get("cell") == CELL
             and value.get("status") == "ATTEMPT_RESERVED" and value.get("run_spec") == spec.payload()
             and value.get("identity") == identity.payload() and value.get("source_only") is True
             and value.get("target_or_formal_or_external_forbidden") is True,
             "attempt receipt drift")


def _validate_launch_payload(value: Mapping[str, object], spec: LifecycleSpec, identity: RunIdentity,
                             attempt_sha256: str) -> None:
    _require(value.get("schema") == "cell_d_equal_session_launch_v1" and value.get("cell") == CELL
             and value.get("status") == "LAUNCHED" and value.get("run_spec") == spec.payload()
             and value.get("identity") == identity.payload() and value.get("attempt_sha256") == attempt_sha256,
             "launch receipt drift")
    held = value.get("held")
    _require(isinstance(held, Mapping) and held.get("sealed_cell_d_graph") is True
             and held.get("b3s_normalized_t4") is True
             and held.get("whole_token_dynamic_dropout_uniform_0_1") is True
             and held.get("dense_all_bin_loss") is True
             and held.get("final_four_swa") is (spec.kind != "source_smoke"),
             "launch held-contract drift")
    loading = value.get("data_loading_control_flow")
    _require(isinstance(loading, Mapping) and loading.get("nontrain_path_or_nwb_metadata_open") is False
             and loading.get("bytewise_identical_shared_builder_control_flow") is False
             and loading.get("science_factor_changed") is False,
             "launch train-only access-order disclosure drift")


def _failure_payload(spec: LifecycleSpec, flags: LifecycleFlags, identity: RunIdentity | None,
                     error: BaseException) -> dict[str, object]:
    return {
        "schema": "cell_d_equal_session_failure_v1", "cell": CELL, "status": "FAILED",
        "run_spec": spec.payload(), "identity": None if identity is None else identity.payload(),
        "stage": flags.stage, "failure": {"kind": type(error).__name__, "detail": str(error)},
        "access_disclosure": flags.disclosure(), "terminal_published": False,
    }


def _terminal_payload(spec: LifecycleSpec, identity: RunIdentity, *, attempt_sha256: str,
                      launch_sha256: str, artifact_sha256s: Mapping[str, str],
                      flags: LifecycleFlags, final_identity: RunIdentity,
                      terminal_detail: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "cell_d_equal_session_terminal_v1", "cell": CELL,
        "status": "SOURCE_SMOKE_COMPLETE__NON_AUTHORITATIVE" if spec.kind == "source_smoke" else "CELL_TRAINING_TERMINAL__UNSCORED",
        "run_spec": spec.payload(), "attempt_sha256": attempt_sha256, "launch_sha256": launch_sha256,
        "artifact_sha256s": dict(artifact_sha256s), "identity_launch": identity.payload(),
        "identity_final": final_identity.payload(),
        "launch_final_closure_equal": identity.closure.get("closure_sha256") == final_identity.closure.get("closure_sha256"),
        "access_disclosure": flags.disclosure(), "terminal_detail": dict(terminal_detail),
        "data_loading_control_flow": {
            "predecessor_preclear_metadata_read": "train+val nwb_unit_count in shared _initialize_splits",
            "successor_train_only_prebinding": True,
            "nontrain_path_or_nwb_metadata_open": False,
            "bytewise_identical_shared_builder_control_flow": False,
            "science_factor_changed": False,
        },
        "target_score_or_evaluation_performed": False,
    }


def _validate_terminal_payload(
    value: Mapping[str, object],
    spec: LifecycleSpec,
    identity: RunIdentity,
    *,
    expected_artifact_sha256s: Mapping[str, str] | None = None,
) -> None:
    expected_status = "SOURCE_SMOKE_COMPLETE__NON_AUTHORITATIVE" if spec.kind == "source_smoke" else "CELL_TRAINING_TERMINAL__UNSCORED"
    _require(value.get("schema") == "cell_d_equal_session_terminal_v1" and value.get("cell") == CELL
             and value.get("status") == expected_status and value.get("run_spec") == spec.payload()
             and value.get("identity_launch") == identity.payload() and value.get("identity_final") == identity.payload()
             and value.get("launch_final_closure_equal") is True
             and value.get("target_score_or_evaluation_performed") is False,
             "terminal receipt identity/status drift")
    if expected_artifact_sha256s is not None:
        _exact(value.get("artifact_sha256s"), dict(expected_artifact_sha256s),
               "terminal artifact SHA binding")
    access = value.get("access_disclosure")
    _require(isinstance(access, Mapping) and access.get("source_train_opened") is True
             and access.get("within_dev_resolved_or_opened") is False
             and access.get("external_resolved_or_opened") is False
             and access.get("formal_or_organizer_held_resolved_or_opened") is False
             and access.get("target_resolved_or_opened") is False
             and access.get("cache_read_or_write") is False,
             "terminal access boundary drift")
    loading = value.get("data_loading_control_flow")
    _require(isinstance(loading, Mapping) and loading.get("successor_train_only_prebinding") is True
             and loading.get("nontrain_path_or_nwb_metadata_open") is False
             and loading.get("bytewise_identical_shared_builder_control_flow") is False
             and loading.get("science_factor_changed") is False,
        "terminal train-only access-order disclosure drift")
    if spec == SOURCE_SMOKE_SPEC:
        _require(
            access.get("optimizer_steps_completed") == 1
            and access.get("backward_calls") == 1
            and access.get("update_calls") == 1
            and access.get("fixed_diagnostic_forwards") == 0,
            "source smoke terminal update/non-authorization accounting drift",
        )


def _checkpoint_binding(spec: LifecycleSpec, identity: RunIdentity, launch_sha256: str) -> dict[str, object]:
    return {
        "cell": CELL, "run_spec": spec.payload(), "launch_sha256": launch_sha256,
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "canonical_initial_state_sha256": CANONICAL_INITIAL_STATE_STATE_SHA256,
        "schedule_plan_sha256": AUDITED_FULL_PLAN_SHA256,
    }


def _validate_preterminal(
    artifact: ArtifactRoot,
    spec: LifecycleSpec,
    identity: RunIdentity,
    hashes: Mapping[str, str],
    backend: TrainingBackend,
    *,
    allow_synthetic_runtime: bool,
) -> None:
    _validate_attempt_payload(artifact.reload_json("attempt.json", hashes["attempt.json"]), spec, identity)
    _validate_launch_payload(artifact.reload_json("launch.json", hashes["launch.json"]), spec, identity,
                             hashes["attempt.json"])
    _validate_source_authority_payload(
        artifact.reload_json("source_authority.json", hashes["source_authority.json"]), spec, identity,
        allow_synthetic_runtime=allow_synthetic_runtime,
    )
    if spec.kind == "source_smoke":
        smoke = artifact.reload_json("smoke.json", hashes["smoke.json"])
        _validate_source_smoke_payload(smoke, spec)
        return
    binding = _checkpoint_binding(spec, identity, hashes["launch.json"])
    for epoch in range(spec.epochs):
        _validate_epoch_payload(artifact.reload_json(f"epoch{epoch:03d}.json", hashes[f"epoch{epoch:03d}.json"]), spec, epoch)
    for epoch in spec.checkpoint_epochs:
        backend.validate_checkpoint(
            artifact.reload_pair(f"epoch{epoch:03d}.pt", hashes[f"epoch{epoch:03d}.pt"]),
            epoch=epoch, global_step=(epoch + 1) * spec.steps_per_epoch, spec=spec, binding=binding,
        )
    backend.validate_swa(artifact.reload_pair("swa_final4.pt", hashes["swa_final4.pt"]),
                         spec=spec, binding=binding)
    throughput_name = f"throughput{spec.throughput_steps}.json"
    throughput = artifact.reload_json(throughput_name, hashes[throughput_name])
    _require(throughput.get("schema") == "cell_d_equal_session_throughput_v1"
             and throughput.get("steps") == spec.throughput_steps
             and _is_sha256(throughput.get("schedule_sha256")), "throughput receipt drift")


def _validate_source_smoke_payload(value: Mapping[str, object], spec: LifecycleSpec) -> None:
    _require(spec == SOURCE_SMOKE_SPEC, "source smoke validator received a non-smoke spec")
    _require(value.get("schema") == "cell_d_equal_session_source_smoke_v1"
             and value.get("cell") == CELL and value.get("run_spec") == spec.payload()
             and value.get("epoch") == 0 and value.get("non_authorizing") is True,
             "source smoke receipt topology drift")
    schedule = value.get("schedule")
    outcome = value.get("outcome")
    end_detail = value.get("end_detail")
    _require(
        isinstance(schedule, Mapping)
        and schedule.get("epoch") == 0
        and schedule.get("schedule_sha256") == AUDITED_EPOCH0_SCHEDULE_SHA256
        and schedule.get("full_plan_sha256") == AUDITED_FULL_PLAN_SHA256
        and schedule.get("consumed_batch_count") == 1
        and schedule.get("full_epoch_batch_count") == PUBLIC_SPEC.steps_per_epoch
        and schedule.get("source_smoke_prefix_only") is True
        and isinstance(outcome, Mapping)
        and isinstance(outcome.get("critical_gradients"), Mapping)
        and outcome.get("finite_model") is True and outcome.get("finite_optimizer") is True
        and _is_sha256(outcome.get("model_state_sha256"))
        and _is_sha256(outcome.get("optimizer_state_sha256"))
        and isinstance(end_detail, Mapping)
        and end_detail.get("one_iterator") is True
        and end_detail.get("sampler_iter_calls") == 1
        and end_detail.get("scheduled_batches_consumed") == 1
        and end_detail.get("observed_batches_consumed") == 1
        and _resource_payload(_mapping(value.get("resources"), "source smoke resources"))["throughput_windows_per_s"] >= 0,
        "source smoke receipt contract drift",
    )
    _validate_critical_gradient_proof(outcome.get("critical_gradients"))
    _exact(outcome.get("critical_gradient_contract"), _critical_gradient_contract_payload(),
           "source smoke critical-gradient contract")


def run_lifecycle(*, spec: LifecycleSpec, identity_factory: Callable[[], RunIdentity],
                  backend: TrainingBackend, artifact: ArtifactRoot) -> Mapping[str, object]:
    """Only receipt-producing path; mocks and physical backend share it exactly."""
    _require_public_lifecycle(spec) if spec in (FULL_TRAIN_SPEC, SOURCE_SMOKE_SPEC) else None
    flags = LifecycleFlags()
    runtime: Any | None = None
    identity: RunIdentity | None = None
    try:
        # Only this exact no-data test double may carry the visibly separate
        # fixture runtime schema.  Physical reviewed execution and arbitrary
        # injected backends have no opt-in and are held to the literal
        # throughput-sealed runtime tuple.
        allow_synthetic_runtime = type(backend) is DeterministicMockBackend
        flags.stage = "identity"
        identity = identity_factory()
        _validate_run_identity(identity, spec)
        flags.stage = "attempt"
        attempt = _attempt_payload(spec, identity)
        _validate_attempt_payload(attempt, spec, identity)
        hashes: dict[str, str] = {"attempt.json": artifact.publish_json("attempt.json", attempt)}
        _validate_attempt_payload(artifact.reload_json("attempt.json", hashes["attempt.json"]), spec, identity)

        flags.stage = "launch"
        launch = _launch_payload(spec, identity, hashes["attempt.json"])
        _validate_launch_payload(launch, spec, identity, hashes["attempt.json"])
        hashes["launch.json"] = artifact.publish_json("launch.json", launch)
        _validate_launch_payload(artifact.reload_json("launch.json", hashes["launch.json"]), spec, identity,
                                 hashes["attempt.json"])

        flags.stage = "backend_prepare"
        runtime = backend.prepare(spec, identity, flags)
        _require(not flags.within_dev_resolved_or_opened and not flags.external_resolved_or_opened
                 and not flags.formal_or_organizer_held_resolved_or_opened and not flags.target_resolved_or_opened
                 and not flags.cache_read_or_write, "forbidden data/cache surface touched by training backend")
        flags.stage = "source_authority"
        source_authority = backend.source_authority(runtime, spec, identity, flags)
        _require(isinstance(source_authority, Mapping), "backend source authority payload type drift")
        _validate_source_authority_payload(
            source_authority, spec, identity, allow_synthetic_runtime=allow_synthetic_runtime,
        )
        hashes["source_authority.json"] = artifact.publish_json("source_authority.json", source_authority)
        _validate_source_authority_payload(
            artifact.reload_json("source_authority.json", hashes["source_authority.json"]), spec, identity,
            allow_synthetic_runtime=allow_synthetic_runtime,
        )
        global_step = 0
        checkpoint_bodies: dict[int, bytes] = {}
        epoch_hashes: list[str] = []
        terminal_detail: dict[str, object] = {}
        for epoch in range(spec.epochs):
            flags.stage = "epoch_begin"
            schedule = backend.begin_epoch(runtime, epoch, flags)
            _require(isinstance(schedule, Mapping) and schedule.get("epoch") == epoch,
                     "backend did not bind explicit epoch schedule")
            outcomes: list[StepOutcome] = []
            started = time.perf_counter()
            for step_in_epoch in range(spec.steps_per_epoch):
                expected_lr = None
                # The physical backend obtains the reviewed arm_common formula.
                # Mock backends use this deterministic scalar through their runtime.
                expected_lr = float(getattr(backend, "lr_for_step", lambda step: 1e-4)(global_step))
                flags.stage = "optimizer_step"
                require_full_proof = spec.kind == "source_smoke" or step_in_epoch == spec.steps_per_epoch - 1
                outcome = backend.train_step(
                    runtime, global_step=global_step, expected_lr=expected_lr,
                    require_full_proof=require_full_proof, flags=flags,
                )
                _validate_step_outcome(outcome, expected_lr, require_full_proof=require_full_proof)
                outcomes.append(outcome)
                global_step += 1
                flags.optimizer_steps_completed = global_step
                if spec.kind != "source_smoke" and global_step == spec.throughput_steps:
                    # GPU work is asynchronous.  Synchronize before taking
                    # the wall-clock sample so the engineering throughput is
                    # a measured completed-work rate, not launch latency.
                    backend.synchronize_for_measurement(runtime)
                    elapsed = max(time.perf_counter() - started, 1e-12)
                    raw = dict(backend.resources(runtime))
                    raw["throughput_windows_per_s"] = int(round(spec.throughput_steps * PUBLIC_SPEC.batch_size / elapsed))
                    throughput = {
                        "schema": "cell_d_equal_session_throughput_v1", "cell": CELL,
                        "steps": spec.throughput_steps, "resources": _resource_payload(raw),
                        "schedule_epoch": epoch, "schedule_sha256": schedule.get("schedule_sha256"),
                    }
                    throughput_name = f"throughput{spec.throughput_steps}.json"
                    hashes[throughput_name] = artifact.publish_json(throughput_name, throughput)
                    artifact.reload_json(throughput_name, hashes[throughput_name])
            flags.stage = "epoch_end"
            end_detail = backend.end_epoch(runtime, epoch, outcomes, flags)
            if spec.kind == "source_smoke":
                _require(epoch == 0 and global_step == 1, "source smoke step accounting drift")
                backend.synchronize_for_measurement(runtime)
                elapsed = max(time.perf_counter() - started, 1e-12)
                smoke_resources = dict(backend.resources(runtime))
                smoke_resources["throughput_windows_per_s"] = int(
                    round(PUBLIC_SPEC.batch_size / elapsed)
                )
                smoke = {
                    "schema": "cell_d_equal_session_source_smoke_v1", "cell": CELL,
                    "run_spec": spec.payload(), "epoch": 0, "schedule": dict(schedule),
                    "outcome": {
                        "loss": outcomes[0].loss, "lr": outcomes[0].lr, "dropout_p": outcomes[0].dropout_p,
                        "critical_gradients": dict(outcomes[0].critical_gradients or {}),
                        "critical_gradient_contract": _critical_gradient_contract_payload(),
                        "finite_model": outcomes[0].finite_model, "finite_optimizer": outcomes[0].finite_optimizer,
                        "model_state_sha256": outcomes[0].model_state_sha256,
                        "optimizer_state_sha256": outcomes[0].optimizer_state_sha256,
                        "batch": dict(outcomes[0].batch_evidence),
                    },
                    "resources": _resource_payload(smoke_resources), "end_detail": dict(end_detail),
                    "non_authorizing": True,
                }
                _validate_source_smoke_payload(smoke, spec)
                hashes["smoke.json"] = artifact.publish_json("smoke.json", smoke)
                _validate_source_smoke_payload(
                    artifact.reload_json("smoke.json", hashes["smoke.json"]), spec,
                )
                terminal_detail = {"smoke": smoke}
            else:
                receipt = _epoch_payload(spec, epoch, global_step, outcomes, schedule, end_detail,
                                         backend.resources(runtime))
                _validate_epoch_payload(receipt, spec, epoch)
                name = f"epoch{epoch:03d}.json"
                hashes[name] = artifact.publish_json(name, receipt)
                epoch_hashes.append(hashes[name])
                _validate_epoch_payload(artifact.reload_json(name, hashes[name]), spec, epoch)
                if epoch in spec.checkpoint_epochs:
                    flags.stage = "checkpoint"
                    binding = _checkpoint_binding(spec, identity, hashes["launch.json"])
                    checkpoint = backend.make_checkpoint(runtime, epoch, global_step, binding)
                    _require(isinstance(checkpoint, CheckpointPayload) and isinstance(checkpoint.body, bytes)
                             and _is_sha256(checkpoint.model_state_sha256), "checkpoint backend payload drift")
                    backend.validate_checkpoint(checkpoint.body, epoch=epoch, global_step=global_step,
                                                spec=spec, binding=binding)
                    name = f"epoch{epoch:03d}.pt"
                    hashes[name] = artifact.publish_bytes(name, checkpoint.body)
                    body = artifact.reload_pair(name, hashes[name])
                    backend.validate_checkpoint(body, epoch=epoch, global_step=global_step, spec=spec, binding=binding)
                    checkpoint_bodies[epoch] = body
        _require(global_step == spec.total_steps, "lifecycle total optimizer-step drift")
        if spec.kind != "source_smoke":
            throughput_name = f"throughput{spec.throughput_steps}.json"
            _require(len(checkpoint_bodies) == len(spec.checkpoint_epochs) and throughput_name in hashes,
                     "full lifecycle checkpoint/throughput topology drift")
            flags.stage = "swa"
            binding = _checkpoint_binding(spec, identity, hashes["launch.json"])
            swa = backend.build_swa(runtime, checkpoint_bodies, spec, binding)
            _require(isinstance(swa, SWAPayload) and isinstance(swa.body, bytes) and _is_sha256(swa.state_sha256)
                     and isinstance(swa.proof, Mapping), "SWA backend payload drift")
            backend.validate_swa(swa.body, spec=spec, binding=binding)
            hashes["swa_final4.pt"] = artifact.publish_bytes("swa_final4.pt", swa.body)
            backend.validate_swa(artifact.reload_pair("swa_final4.pt", hashes["swa_final4.pt"]),
                                 spec=spec, binding=binding)
            terminal_detail = {"epoch_receipt_sha256s": epoch_hashes, "swa_state_sha256": swa.state_sha256,
                               "swa_proof": dict(swa.proof)}

        flags.stage = "terminal"
        final_identity = identity_factory()
        _validate_run_identity(final_identity, spec)
        _require(final_identity.payload() == identity.payload(), "launch/final authority or closure drift")
        _validate_preterminal(
            artifact, spec, identity, hashes, backend,
            allow_synthetic_runtime=allow_synthetic_runtime,
        )
        terminal = _terminal_payload(
            spec, identity, attempt_sha256=hashes["attempt.json"], launch_sha256=hashes["launch.json"],
            artifact_sha256s=hashes, flags=flags, final_identity=final_identity, terminal_detail=terminal_detail,
        )
        _validate_terminal_payload(terminal, spec, identity, expected_artifact_sha256s=hashes)
        hashes["terminal.json"] = artifact.publish_json("terminal.json", terminal)
        _validate_terminal_payload(
            artifact.reload_json("terminal.json", hashes["terminal.json"]),
            spec,
            identity,
            expected_artifact_sha256s={key: value for key, value in hashes.items() if key != "terminal.json"},
        )
        flags.terminal_published = True
        return terminal
    except BaseException as error:
        if identity is not None and not flags.terminal_published and not artifact.has_name("failure.json"):
            try:
                artifact.publish_json("failure.json", _failure_payload(spec, flags, identity, error))
            except BaseException:
                # Preserve the original failure; an incomplete failure receipt is
                # never misreported as success because terminal was not published.
                pass
        raise
    finally:
        try:
            backend.close(runtime)
        except BaseException:
            pass


# ---------------------------------------------------------------------------
# Physical backend (deferred imports only)
# ---------------------------------------------------------------------------


class _OneIteratorEpochBatchSampler:
    """DataLoader adapter that makes one planned epoch iterable exactly once."""

    def __init__(self, schedule: EpochSchedule, *, maximum_batches: int | None = None) -> None:
        self.schedule = schedule
        _require(maximum_batches is None or (type(maximum_batches) is int and maximum_batches > 0),
                 "epoch sampler maximum-batch limit drift")
        self.maximum_batches = maximum_batches
        self.iter_calls = 0
        self.emitted_batches = 0
        # A worker-backed DataLoader may prefetch several batches before the
        # trainer receives the first one.  ``last_batch`` would therefore bind
        # the received tensor to a later scheduled batch.  This FIFO contains
        # only the bounded in-flight prefetch frontier, not an epoch's batch
        # list, and is consumed exactly once in DataLoader delivery order.
        self._pending: deque[ScheduledBatch] = deque()
        self.observed_batches = 0

    def __len__(self) -> int:
        return len(self.schedule) if self.maximum_batches is None else min(len(self.schedule), self.maximum_batches)

    def __iter__(self) -> Iterator[list[int]]:
        self.iter_calls += 1
        _require(self.iter_calls == 1, "an epoch schedule received a second DataLoader iterator")
        for batch in self.schedule.iter_batches():
            if self.maximum_batches is not None and self.emitted_batches >= self.maximum_batches:
                break
            self._pending.append(batch)
            self.emitted_batches += 1
            yield list(batch.dataset_indices)

    def consume_observed_batch(self) -> ScheduledBatch:
        _require(bool(self._pending), "DataLoader yielded a batch without scheduled evidence")
        self.observed_batches += 1
        return self._pending.popleft()

    def assert_complete(self) -> None:
        _require(self.iter_calls == 1 and self.emitted_batches == len(self)
                 and self.observed_batches == len(self) and not self._pending,
                 "epoch did not consume exact one-iterator scheduled batch count")


def _prepend_runtime_packages(root: Path) -> None:
    """Make the two reviewed source trees importable only inside physical work."""
    for relative in ("sua_exploration", "streaming_calibration_exp"):
        path = str(Path(root).absolute() / relative)
        if path not in sys.path:
            sys.path.insert(0, path)


def _torch_rng_snapshot(torch: Any, numpy_module: Any) -> dict[str, str]:
    return rng_snapshot(numpy_module=numpy_module, torch_module=torch)


def _read_immutable_binary(root: Path, relative: str, *, expected_sha256: str, mode: int = 0o444) -> bytes:
    """Read a sealed binary plus ordinary SHA sidecar without path aliases."""
    body = _regular_bytes(Path(root), relative, mode=mode)
    digest = _sha256(body)
    _exact(digest, expected_sha256, "sealed binary authority SHA")
    sidecar = _regular_bytes(Path(root), relative + ".sha256", mode=mode)
    _require(sidecar == f"{digest}  {Path(relative).name}\n".encode("ascii"),
             "sealed binary authority sidecar drift")
    return body


def _validate_canonical_initial_state_payload(
    payload: object,
    *,
    torch_version_type: Any,
) -> Mapping[str, Any]:
    """Validate the sealed CPU payload after its narrow weights-only decode."""
    _require(
        isinstance(payload, Mapping)
        and set(payload) == {
            "kind", "seed", "builder", "state_dict", "state_sha256",
            "w_side_exact_positive_zero", "created_utc", "torch",
        }
        and payload.get("kind") == "tfpd_admission_canonical_initial_state_v1"
        and payload.get("seed") == 42
        and payload.get("builder") == "src/tfpd/spintshape_module.py:build_spintshape_model"
        and payload.get("state_sha256") == CANONICAL_INITIAL_STATE_STATE_SHA256
        and payload.get("w_side_exact_positive_zero") is True
        and payload.get("created_utc") == "2026-08-16T12:36:27Z"
        and type(payload.get("torch")) is torch_version_type
        and str(payload.get("torch")) == CANONICAL_INITIAL_STATE_TORCH_VERSION
        and isinstance(payload.get("state_dict"), Mapping),
        "canonical initial artifact schema/runtime/state authority drift",
    )
    return payload["state_dict"]


def _load_canonical_initial_state(root: Path, torch: Any) -> Mapping[str, Any]:
    """Strict local weights-only load of the immutable Cell-D canonical state.

    The sealed artifact serializes exactly two trusted non-tensor classes:
    ``UninitializedParameter`` for the known dead lazy keys and
    ``TorchVersion`` for the sealed runtime field.  The allowlist is scoped to
    this decode only; it is never installed process-wide and ``weights_only``
    is never disabled.
    """
    from torch.nn.parameter import UninitializedParameter
    from torch.torch_version import TorchVersion

    body = _read_immutable_binary(
        root,
        CANONICAL_INITIAL_STATE_RELATIVE,
        expected_sha256=CANONICAL_INITIAL_STATE_SHA256,
    )
    with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
        payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    return _validate_canonical_initial_state_payload(payload, torch_version_type=TorchVersion)


def _lazy_safe_parameter_count(model: Any, torch: Any) -> tuple[int, list[str]]:
    from torch.nn.parameter import UninitializedParameter

    initialized = 0
    lazy: list[str] = []
    for key, value in model.named_parameters():
        if isinstance(value, UninitializedParameter):
            lazy.append(key)
            continue
        initialized += int(value.numel())
    return initialized, sorted(lazy)


@contextlib.contextmanager
def _preserve_cpu_rng_for_state_validation(torch: Any) -> Iterator[None]:
    """Fresh CPU strict-loads must not perturb the training/dropout RNG stream."""
    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _strict_recompute_cell_d_state_digest(
    *,
    root: Path,
    torch: Any,
    state_dict: Mapping[str, Any],
    claimed_sha256: object,
    label: str,
) -> dict[str, object]:
    """Strict-load an artifact state into a fresh CPU Cell-D graph and hash it.

    A self-declared ``state_dict_sha256`` is never accepted merely because it
    has SHA-shaped syntax.  This helper also rejects a payload that attempts to
    initialize, drop, or relocate either intentionally uninitialized lazy
    ``fc_id_in`` parameter.
    """
    from torch.nn.parameter import UninitializedParameter

    _require(isinstance(state_dict, Mapping), f"{label} state dictionary type drift")
    _require(_is_sha256(claimed_sha256), f"{label} claimed state SHA syntax drift")
    serialized_lazy = sorted(
        key for key, value in state_dict.items() if isinstance(value, UninitializedParameter)
    )
    _exact(serialized_lazy, list(CELL_D_UNINITIALIZED_LAZY_KEYS),
           f"{label} serialized lazy topology")
    _require(
        all(
            isinstance(value, UninitializedParameter) or torch.is_tensor(value)
            for value in state_dict.values()
        ),
        f"{label} serialized state contains a non-tensor entry",
    )
    # ``build_population_robustness_model`` intentionally seeds Torch.  State
    # validation is a receipt check, not a schedule/RNG event, so restore all
    # host RNGs around the fresh construction and strict load.
    with _preserve_cpu_rng_for_state_validation(torch):
        _prepend_runtime_packages(root)
        arm_common = _load_runtime_module(
            "_equal_session_state_validation_arm_common",
            Path(root).absolute() / "tfpd_exploration/src/tfpd_lane/arm_common.py",
        )
        pop_robust = _load_runtime_module(
            "_equal_session_state_validation_pop_robust",
            Path(root).absolute() / "tfpd_exploration/src/tfpd_lane/pop_robust.py",
        )
        fresh = pop_robust.build_population_robustness_model(seed=PUBLIC_SPEC.seed, cell="D")
        _require(
            all(
                isinstance(parameter, UninitializedParameter) or parameter.device.type == "cpu"
                for parameter in fresh.parameters()
            ),
            f"{label} fresh strict-validation graph is not CPU-resident",
        )
        before_initialized, before_lazy = _lazy_safe_parameter_count(fresh, torch)
        _exact(before_initialized, CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
               f"{label} fresh Cell-D initialized count")
        _exact(before_lazy, list(CELL_D_UNINITIALIZED_LAZY_KEYS),
               f"{label} fresh Cell-D lazy topology")
        _require(set(state_dict) == set(fresh.state_dict()), f"{label} state key topology drift")
        fresh.load_state_dict(state_dict, strict=True)
        _require(
            all(
                isinstance(parameter, UninitializedParameter) or parameter.device.type == "cpu"
                for parameter in fresh.parameters()
            ),
            f"{label} strict-loaded validation graph left CPU",
        )
        initialized, lazy = _lazy_safe_parameter_count(fresh, torch)
        _exact(initialized, CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
               f"{label} strict-loaded initialized count")
        _exact(lazy, list(CELL_D_UNINITIALIZED_LAZY_KEYS),
               f"{label} strict-loaded lazy topology")
        actual_sha256 = arm_common.state_sha256(fresh)
    _exact(actual_sha256, claimed_sha256, f"{label} strict-loaded state SHA")
    return {
        "state_dict_sha256": actual_sha256,
        "initialized_trainable_parameters": CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
        "uninitialized_lazy_keys": list(CELL_D_UNINITIALIZED_LAZY_KEYS),
    }


def _finite_model_and_optimizer(model: Any, optimizer: Any, torch: Any) -> tuple[bool, bool]:
    from torch.nn.parameter import UninitializedParameter

    model_ok = True
    for parameter in model.parameters():
        if isinstance(parameter, UninitializedParameter):
            continue
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            model_ok = False
            break

    optimizer_ok = True

    def walk(value: Any) -> None:
        nonlocal optimizer_ok
        if torch.is_tensor(value):
            if not bool(torch.isfinite(value.detach()).all().item()):
                optimizer_ok = False
        elif isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(optimizer.state_dict())
    return model_ok, optimizer_ok


def _critical_gradient_proof(model: Any, torch: Any) -> dict[str, bool]:
    """Prove gradients through each named live Cell-D route boundary.

    This intentionally excludes only the two known dead lazy ``fc_id_in``
    parameters, which are disclosed in the paired contract payload.  It does
    not accept a gradient on an arbitrary encoder/decoder parameter as a
    substitute for a required path.
    """
    from torch.nn.parameter import UninitializedParameter

    named = dict(model.named_parameters())
    hidden = int(model.id_encoder.hidden_dim)
    side_dim = int(model.id_encoder.side_dim)
    _require(hidden == 64 and side_dim == 4, "Cell-D B3S gradient-block dimensions drift")
    _require(
        sorted(name for name, parameter in named.items() if isinstance(parameter, UninitializedParameter))
        == list(CELL_D_UNINITIALIZED_LAZY_KEYS),
        "Cell-D lazy topology drift before gradient proof",
    )

    def selected_nonzero(name: str, *, columns: slice | None = None) -> bool:
        parameter = named.get(name)
        _require(parameter is not None and not isinstance(parameter, UninitializedParameter),
                 f"critical gradient parameter missing/lazy: {name}")
        gradient = parameter.grad
        if gradient is None:
            return False
        _require(tuple(gradient.shape) == tuple(parameter.shape),
                 f"critical gradient shape drift: {name}")
        if columns is not None:
            _require(gradient.ndim == 2 and gradient.shape[1] == hidden + side_dim,
                     f"critical B3S post-pool shape drift: {name}")
            gradient = gradient[:, columns]
        return bool(torch.isfinite(gradient).all().item() and (gradient != 0).any().item())

    return {
        "b3s_pre_pool_activity": selected_nonzero("id_encoder.pre_pool.0.weight"),
        "b3s_post_pool_mean": selected_nonzero("id_encoder.post_pool.0.weight", columns=slice(0, hidden)),
        "b3s_post_pool_t4_side": selected_nonzero(
            "id_encoder.post_pool.0.weight", columns=slice(hidden, hidden + side_dim),
        ),
        "decoder_fc_in": selected_nonzero("decoder.fc_in.0.weight"),
        "decoder_cross_attention": selected_nonzero(
            "decoder.transformer.layers.0.cross_attn.in_proj_weight",
        ),
        "decoder_ffn": selected_nonzero("decoder.transformer.layers.0.ffn.0.weight"),
        "decoder_query_rep": selected_nonzero("decoder.rep"),
        "decoder_fc_out": selected_nonzero("decoder.fc_out.weight"),
    }


def _rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return 0


def _require_exact_gpu0(torch: Any) -> dict[str, object]:
    """Check physical GPU0 only; GPU1 is never enumerated or touched."""
    import subprocess

    validate_future_gpu0_environment()
    _require(torch.cuda.is_available(), "GPU0 CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "CUDA_VISIBLE_DEVICES=0 did not expose exactly one logical device")
    # The command explicitly queries physical index zero.  No broad device
    # listing or GPU1 probe is performed.
    completed = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=uuid,pci.bus_id,name,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    _require(len(rows) == 1, "GPU0 nvidia-smi query topology drift")
    columns = [column.strip() for column in rows[0].split(",")]
    _require(len(columns) == 4, "GPU0 nvidia-smi field topology drift")
    uuid, bdf, name, memory_mib = columns
    _exact(uuid, FROZEN_GPU0["uuid"], "GPU0 UUID")
    _exact(bdf.upper(), str(FROZEN_GPU0["bdf"]).upper(), "GPU0 BDF")
    _exact(name, FROZEN_GPU0["name"], "GPU0 name")
    _exact(int(memory_mib), FROZEN_GPU0["nvidia_smi_memory_total_mib"], "GPU0 nominal MiB")
    properties = torch.cuda.get_device_properties(0)
    _exact(properties.name, FROZEN_GPU0["name"], "GPU0 Torch name")
    _exact(int(properties.total_memory), FROZEN_GPU0["torch_total_memory_bytes"], "GPU0 Torch byte capacity")
    torch.cuda.set_device(0)
    payload = {
        "schema": RUNTIME_ENVIRONMENT_SCHEMA,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_authorities": dict(FROZEN_GPU0),
        "physical_nvidia_smi": {
            "uuid": uuid,
            "bdf": bdf,
            "name": name,
            "nvidia_smi_memory_total_mib": int(memory_mib),
        },
        "torch_properties": {
            "name": str(properties.name),
            "torch_total_memory_bytes": int(properties.total_memory),
        },
    }
    _validate_runtime_environment_payload(payload)
    return payload


def _validate_runtime_environment_payload(value: object) -> None:
    """Validate the literal physical runtime sealed by throughput authority.

    This deliberately does not accept a merely nonempty Torch/CUDA/cuDNN
    triple.  The reviewed GPU0 smoke is meaningful only if it runs under the
    exact runtime that produced the immutable engineering authority.
    """
    _require(isinstance(value, Mapping), "runtime environment payload type drift")
    _require(
        set(value) == {
            "schema",
            "torch_version",
            "torch_cuda_version",
            "cudnn_version",
            "gpu_authorities",
            "physical_nvidia_smi",
            "torch_properties",
        }
        and value.get("schema") == RUNTIME_ENVIRONMENT_SCHEMA
        and value.get("torch_version") == FROZEN_RUNTIME_VERSIONS["torch_version"]
        and value.get("torch_cuda_version") == FROZEN_RUNTIME_VERSIONS["torch_cuda_version"]
        and value.get("cudnn_version") == FROZEN_RUNTIME_VERSIONS["cudnn_version"],
        "runtime Torch/CUDA/cuDNN version drift",
    )
    _exact(value.get("gpu_authorities"), dict(FROZEN_GPU0), "runtime frozen GPU authority")
    nvidia = value.get("physical_nvidia_smi")
    properties = value.get("torch_properties")
    _require(
        isinstance(nvidia, Mapping)
        and nvidia == {
            "uuid": FROZEN_GPU0["uuid"],
            "bdf": FROZEN_GPU0["bdf"],
            "name": FROZEN_GPU0["name"],
            "nvidia_smi_memory_total_mib": FROZEN_GPU0["nvidia_smi_memory_total_mib"],
        }
        and isinstance(properties, Mapping)
        and properties == {
            "name": FROZEN_GPU0["name"],
            "torch_total_memory_bytes": FROZEN_GPU0["torch_total_memory_bytes"],
        },
        "runtime GPU physical/Torch identity drift",
    )


def _frozen_runtime_environment_fixture() -> dict[str, object]:
    """Build the literal production-schema fixture for no-CUDA validator tests.

    This is not a mock execution environment and is never emitted by the
    mock lifecycle backend.  It exists solely so focused tests can exercise
    the production validator without probing a device.
    """
    payload: dict[str, object] = {
        "schema": RUNTIME_ENVIRONMENT_SCHEMA,
        **dict(FROZEN_RUNTIME_VERSIONS),
        "gpu_authorities": dict(FROZEN_GPU0),
        "physical_nvidia_smi": {
            "uuid": FROZEN_GPU0["uuid"],
            "bdf": FROZEN_GPU0["bdf"],
            "name": FROZEN_GPU0["name"],
            "nvidia_smi_memory_total_mib": FROZEN_GPU0["nvidia_smi_memory_total_mib"],
        },
        "torch_properties": {
            "name": FROZEN_GPU0["name"],
            "torch_total_memory_bytes": FROZEN_GPU0["torch_total_memory_bytes"],
        },
    }
    _validate_runtime_environment_payload(payload)
    return payload


def _validate_synthetic_runtime_environment_payload(value: object) -> None:
    """Validate a no-device mock fixture; it is intentionally non-production."""
    _require(
        isinstance(value, Mapping)
        and value == {
            "schema": SYNTHETIC_RUNTIME_ENVIRONMENT_SCHEMA,
            "fixture_only": True,
            "fixture_runtime": {
                "torch_version": "synthetic-torch",
                "torch_cuda_version": "synthetic-cuda",
                "cudnn_version": 1,
            },
            "frozen_gpu_authorities": dict(FROZEN_GPU0),
        },
        "synthetic runtime fixture schema/authority drift",
    )


def _synthetic_runtime_environment_payload() -> dict[str, object]:
    """Return the distinct no-CUDA fixture accepted only by mock validation."""
    payload: dict[str, object] = {
        "schema": SYNTHETIC_RUNTIME_ENVIRONMENT_SCHEMA,
        "fixture_only": True,
        "fixture_runtime": {
            "torch_version": "synthetic-torch",
            "torch_cuda_version": "synthetic-cuda",
            "cudnn_version": 1,
        },
        "frozen_gpu_authorities": dict(FROZEN_GPU0),
    }
    _validate_synthetic_runtime_environment_payload(payload)
    return payload


def _source_inventory_from_dataset(dataset: Any, roster: Sequence[str]) -> SourceInventory:
    """Rebuild and bind the exact train dataset row map used by the sampler."""
    _require(tuple(dataset.sessions) == tuple(roster), "live train dataset session insertion order drift")
    grouped: dict[str, list[int]] = {session: [] for session in roster}
    observed_indices: dict[str, list[int]] = {session: [] for session in roster}
    for index, pair in enumerate(dataset.window_indices):
        _require(isinstance(pair, tuple) and len(pair) == 2, "live train dataset window row shape drift")
        session, start = pair
        _require(session in grouped and type(start) is int, "live train dataset has forbidden session/window")
        grouped[session].append(start)
        observed_indices[session].append(index)
    inventory = inventory_from_window_counts(tuple(roster), {key: tuple(value) for key, value in grouped.items()})
    for session_inventory in inventory.sessions:
        _require(tuple(observed_indices[session_inventory.session]) == session_inventory.dataset_indices,
                 "live train dataset global index mapping differs from audited inventory")
    return inventory


def prebind_strict_train_only_manifest(
    datamodule: Any,
    root: Path,
    roster: Sequence[str],
    *,
    direct_source_file: Callable[[Path, str], Path] | None = None,
) -> tuple[Path, ...]:
    """Bind only train paths before shared ``setup('fit')`` can inspect splits.

    This function is intentionally injection-friendly: focused tests can make
    every non-train resolver or ``_initialize_splits`` raise, proving the
    successor never asks the shared datamodule to construct a val/test path.
    """
    roster_tuple = tuple(roster)
    _require(len(roster_tuple) == 27 and len(set(roster_tuple)) == 27,
             "strict train-only prebind roster topology drift")
    resolver = direct_source_file
    if resolver is None:
        resolver = lambda candidate_root, session: _direct_source_file(
            candidate_root, session, allowed_train_roster=roster_tuple,
        )
    train_files = tuple(resolver(Path(root).absolute(), session) for session in roster_tuple)
    _require(len(train_files) == 27 and len(set(train_files)) == 27,
             "strict train-only prebind source path topology drift")
    # Do not call datamodule._initialize_splits(): its current shared body
    # calls nwb_unit_count over train+val before a caller can clear val.
    datamodule.session_files = {"train": list(train_files), "val": [], "test": []}
    datamodule.session_splits = {"train": list(roster_tuple), "val": [], "test": []}
    datamodule.session_unit_counts = {}
    datamodule._splits_initialized = True
    _require(getattr(datamodule, "train_dataset", None) is None
             and getattr(datamodule, "val_dataset", None) is None
             and getattr(datamodule, "test_dataset", None) is None,
             "strict train-only prebind requires fresh shared datasets")
    # The shared ``setup('fit')`` unconditionally constructs an empty
    # ``Dandi688MultiSessionDataset({})`` when ``val_dataset is None``, even
    # though val files are empty.  A unique non-data sentinel skips that
    # branch, preventing even an empty val ``load_split`` call; restore None
    # only after proving the shared setup preserved the exact sentinel.
    val_sentinel = object()
    datamodule.val_dataset = val_sentinel
    try:
        datamodule.setup("fit")
        _require(getattr(datamodule, "val_dataset", None) is val_sentinel,
                 "strict train-only prebind allowed shared setup to construct a val dataset")
    finally:
        if getattr(datamodule, "val_dataset", None) is val_sentinel:
            datamodule.val_dataset = None
    _require(getattr(datamodule, "session_files")["val"] == []
             and getattr(datamodule, "session_files")["test"] == []
             and getattr(datamodule, "session_splits")["val"] == []
             and getattr(datamodule, "session_splits")["test"] == []
             and getattr(datamodule, "val_dataset", None) is None
             and getattr(datamodule, "test_dataset", None) is None,
             "strict train-only prebind allowed a non-train split")
    return train_files


def _construct_strict_train_only_datamodule(
    root: Path,
    *,
    num_workers: int,
    datamodule_factory: Callable[..., Any],
    a2_module: Any,
    roster: Sequence[str],
    direct_source_file: Callable[[Path, str], Path] | None = None,
) -> tuple[Any, Any, tuple[str, ...], tuple[Path, ...]]:
    """Construct the shared fit datamodule through route-owned train binding.

    This deliberately small injection seam is not a science knob: production
    supplies the existing ``Dandi688MultiSessionDataModule`` and A2 module,
    while focused tests make every non-train resolver, ``nwb_unit_count``, and
    the empty-val ``load_split`` branch fail.  The shared setup still calls its
    prebound no-op initializer, exactly as the real control flow does.  This
    lets the route prove its access-order repair without opening a real NWB in
    a unit test.
    """
    roster_tuple = tuple(roster)
    _require(len(roster_tuple) == 27 and len(set(roster_tuple)) == 27,
             "strict train-only constructor roster drift")
    dm = datamodule_factory(
        data_dir=str(Path(root).absolute() / SOURCE_DATA_RELATIVE),
        task="CO", split_counts=(27, 6, 6), batch_size=PUBLIC_SPEC.batch_size,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=num_workers, pin_memory=True, random_calibration=False, seed=PUBLIC_SPEC.seed,
        max_units_exclusive=100, cache_dir=None, signal_view="sua", side_feature_group="t4",
        side_feature_pool_size=30, train_val_manifest_path=str(Path(root).absolute() / MANIFEST_RELATIVE),
    )
    # This is the strict train-only equivalent of initialize-manifest -> clear
    # val -> setup(fit), but avoids the shared helper's pre-clear validation
    # unit-count scan described above.  No non-train pathname is constructed.
    train_files = prebind_strict_train_only_manifest(
        dm, root, roster_tuple, direct_source_file=direct_source_file,
    )
    _require(dm.train_dataset is not None and dm.val_dataset is None and dm.test_dataset is None,
             "strict train-only datamodule touched a non-train dataset")
    _require(dm.cache_dir is None, "equal-session route may not read or write a data cache")
    _require(tuple(dm.session_splits["train"]) == roster_tuple and dm.session_files["val"] == []
             and dm.session_files["test"] == [], "strict train-only datamodule roster drift")
    _exact(a2_module.EXPECTED_MANIFEST_SHA256, MANIFEST_SHA256, "A2/shared manifest authority")
    return dm, a2_module, roster_tuple, train_files


def _build_strict_train_only_datamodule(
    root: Path, *, num_workers: int
) -> tuple[Any, Any, tuple[str, ...], tuple[Path, ...]]:
    """Build the reviewed shared datamodule with a source-only manifest bind.

    The shared helper's ``_initialize_splits`` currently computes unit counts
    for validation files before a caller can clear them.  This route therefore
    binds only the already sealed train roster into the same datamodule before
    ``setup('fit')``.  That preserves the strict training construction while
    avoiding a hidden within-session metadata open.
    """
    _prepend_runtime_packages(root)
    import mc_maze.a2_matched_subject_shift_v2_core as a2
    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

    return _construct_strict_train_only_datamodule(
        root,
        num_workers=num_workers,
        datamodule_factory=Dandi688MultiSessionDataModule,
        a2_module=a2,
        roster=load_strict_source_roster(root),
    )


class PhysicalEqualSessionBackend:
    """Reviewed, deferred Cell-D training backend; never selected by public CLI."""

    def __init__(self, root: Path, *, num_workers: int = 4) -> None:
        self.root = Path(root).absolute()
        self.num_workers = num_workers
        self.lr_for_step: Callable[[int], float] = lambda _step: 1e-4

    def prepare(self, spec: LifecycleSpec, identity: RunIdentity, flags: LifecycleFlags) -> dict[str, Any]:
        _require_public_lifecycle(spec)
        _validate_run_identity(identity, spec)
        _require(self.num_workers == 4, "physical equal-session worker count differs from predecessor")
        _prepend_runtime_packages(self.root)
        import lightning.pytorch as pl
        import numpy as np
        import torch
        from torch.utils.data import default_collate
        from torch.nn.parameter import UninitializedParameter

        # Validate the immutable CPU canonical artifact before constructing a
        # train datamodule.  A checkpoint-format/schema failure must fail
        # before resolving or opening any of the 27 source NWBs; this load is
        # weights-only, local-safe-globals, CPU-only, and does not initialize
        # CUDA or consume a source surface.
        canonical_state = _load_canonical_initial_state(self.root, torch)

        # The shared ``setup('fit')`` below may begin opening train NWBs before
        # it can return.  Mark the source surface only immediately before
        # entering it, so its failure envelope is conservative and the
        # canonical-artifact precheck above remains visibly pre-source.
        flags.source_train_opened = True
        flags.cache_read_or_write = False
        dm, a2, roster, train_files = _build_strict_train_only_datamodule(
            self.root, num_workers=self.num_workers,
        )
        dataset = dm.train_dataset
        assert dataset is not None
        source_authority = identity.source_schedule["sealed_arm_a_source_authority"]
        _exact(list(roster), source_authority["roster"], "live/Arm-A strict train roster order")
        behavior_semantic = a2.normalizer_value_sha256(*dm._behavior_stats)
        _exact(behavior_semantic, SEALED_BEHAVIOR_NORMALIZER_SHA256,
               "live train-only behavior normalizer")
        from mc_maze.unit_side_features import side_feature_stats_sha256
        _require(dm._side_feature_stats is not None, "live T4 side normalizer absent")
        t4_normalizer = side_feature_stats_sha256(*dm._side_feature_stats)
        _exact(t4_normalizer, SEALED_T4_NORMALIZER_SHA256, "live train-only T4 normalizer")
        inventory = _source_inventory_from_dataset(dataset, roster)
        _exact(inventory.eligible_windows, SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS,
               "live strict-27 eligible-window count")
        _exact(inventory.full_batches(PUBLIC_SPEC), PUBLIC_SPEC.steps_per_epoch,
               "live strict-27 predecessor full-B32 batch count")
        _exact(inventory.full_batches(PUBLIC_SPEC) * PUBLIC_SPEC.batch_size,
               SOURCE_ONLY_EXPECTED_PREDECESSOR_CONSUMED_WINDOWS,
               "live strict-27 predecessor consumed-window count")
        _exact(inventory.remainder_windows(PUBLIC_SPEC), SOURCE_ONLY_EXPECTED_REMAINDER_WINDOWS,
               "live strict-27 remainder count")
        plan = build_plan_evidence(
            spec=PUBLIC_SPEC, inventory=inventory,
            snapshot_rng=lambda: _torch_rng_snapshot(torch, np),
        )
        _exact(plan.get("full_plan_sha256"), AUDITED_FULL_PLAN_SHA256, "live equal-session full plan SHA")
        _exact(plan["epochs"][0].get("schedule_sha256"), AUDITED_EPOCH0_SCHEDULE_SHA256,
               "live equal-session epoch-0 SHA")
        _exact(plan["epochs"][47].get("schedule_sha256"), AUDITED_EPOCH47_SCHEDULE_SHA256,
               "live equal-session epoch-47 SHA")
        _require(plan.get("global_rng_unchanged") is True, "live schedule construction consumed global RNG")

        # This is the predecessor's seed/model ordering.  The schedule had
        # already been built with local RNG only, and the shared model builder
        # resets torch.manual_seed(seed) exactly as sealed Cell D did.
        pl.seed_everything(PUBLIC_SPEC.seed, workers=True)
        arm_common = _load_runtime_module("_equal_session_arm_common", self.root / "tfpd_exploration/src/tfpd_lane/arm_common.py")
        pop_robust = _load_runtime_module("_equal_session_pop_robust", self.root / "tfpd_exploration/src/tfpd_lane/pop_robust.py")
        t4_fingerprints = arm_common.t4_authority_fingerprint(dataset.sessions)
        _exact(t4_fingerprints, source_authority["t4_authority_sha256"],
               "live normalized T4 per-session fingerprint authority")
        model = pop_robust.build_population_robustness_model(seed=PUBLIC_SPEC.seed, cell="D")
        _exact(getattr(model, "_pop_robust_cell", None), "D", "Cell-D graph selector")
        _exact(
            getattr(model, "_pop_robust_config", None),
            {"num_heads": 2, "dynamic_dropout": True,
             "note": "2 heads + dynamic dropout; head count identical to arm A"},
            "Cell-D population-robust graph metadata",
        )
        _require(
            getattr(model.decoder, "window_size", None) == 50
            and getattr(model.decoder, "num_covariates", None) == 2
            and getattr(model.decoder, "dynamic_dropout", None) is True
            and getattr(model.decoder, "dynamic_dropout_low", None) == 0.0
            and getattr(model.decoder, "dynamic_dropout_high", None) == 1.0
            and model.decoder.transformer.layers[0].cross_attn.num_heads == 2
            and getattr(model.id_encoder, "side_dim", None) == 4
            and getattr(model.id_encoder, "hidden_dim", None) == 64,
            "Cell-D B3S/normalized-T4/dropout graph drift",
        )
        model.load_state_dict(canonical_state, strict=True)
        _exact(arm_common.state_sha256(model), CANONICAL_INITIAL_STATE_STATE_SHA256,
               "strict-loaded canonical Cell-D state SHA")
        initialized, lazy = _lazy_safe_parameter_count(model, torch)
        _exact(initialized, 3_510_842, "Cell-D initialized trainable parameter count")
        _exact(lazy, ["decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight"], "Cell-D lazy topology")
        proof = pop_robust.initial_state_equality_proof(canonical_state, seed=PUBLIC_SPEC.seed)
        proof_d = proof.get("D") if isinstance(proof, Mapping) else None
        _require(isinstance(proof_d, Mapping) and proof_d.get("strict_load") is True
                 and proof_d.get("state_keys_equal_to_canonical") is True
                 and proof_d.get("trainable_parameters") == 3_510_842,
                 "Cell-D canonical initial-state equality proof drift")

        flags.cuda_initialized = True
        runtime_environment = _require_exact_gpu0(torch)
        device = torch.device(str(FROZEN_GPU0["logical_device"]))
        model.to(device)
        # The fixed diagnostic path mirrors the predecessor's one direct
        # source batch (not a DataLoader iterator) and is explicitly disclosed.
        initial_schedule = EpochSchedule(spec=PUBLIC_SPEC, inventory=inventory, epoch=0)
        first_batch = next(initial_schedule.iter_batches())
        before_fixed_input = _torch_rng_snapshot(torch, np)
        fixed = default_collate([dataset[index] for index in first_batch.dataset_indices])
        fixed = tuple(value.to(device) if torch.is_tensor(value) else value for value in fixed[:5])
        after_fixed_input = _torch_rng_snapshot(torch, np)
        _require(before_fixed_input == after_fixed_input,
                 "fixed predecessor-like diagnostic input consumed global/model/dropout RNG")
        # Preserve the predecessor's ordering: model/device and its fixed
        # source-only diagnostic batch precede exact Adam construction.
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.0, amsgrad=False,
        )
        # The feasibility window begins only after the exact model, canonical
        # state, fixed diagnostic input, and Adam state have been constructed.
        # Thus source-smoke/full receipts distinguish current usage from the
        # peak incurred by the reviewed training step(s), rather than calling
        # a current counter a peak.
        torch.cuda.reset_peak_memory_stats(0)
        self.lr_for_step = lambda step: float(arm_common.lr_at_step(step, 48, 33_925))
        return {
            "torch": torch, "numpy": np, "pl": pl, "arm_common": arm_common, "pop_robust": pop_robust,
            "model": model, "optimizer": optimizer, "dm": dm, "dataset": dataset, "roster": roster,
            "train_files": train_files,
            "inventory": inventory, "plan": plan, "ledger": EpochLedger(plan), "device": device,
            "runtime_environment": runtime_environment,
            "peak_stats_reset_before_training": True,
            "fixed": fixed, "fixed_batch_schedule": first_batch.payload(),
            "fixed_diagnostic_input_rng_unchanged": True,
            "behavior_normalizer_semantic_sha256": behavior_semantic,
            "t4_normalizer_semantic_sha256": t4_normalizer,
            "t4_authority_fingerprint": t4_fingerprints,
            "epoch_iterator": None, "epoch_sampler": None, "epoch_schedule": None,
            "lifecycle_spec": spec,
        }

    def source_authority(self, runtime: Mapping[str, Any], spec: LifecycleSpec,
                         identity: RunIdentity, flags: LifecycleFlags) -> Mapping[str, object]:
        """Publish the consumed train-only source authority before any update.

        The payload intentionally records the successor's safer access order
        rather than claiming bytewise equivalence with the predecessor's
        shared train+validation metadata initialization.  All paths are made
        repository-relative after exact strict-roster binding; this keeps the
        receipt portable while still proving that no val/test pathname entered
        the live construction.
        """
        _require(spec == runtime["lifecycle_spec"], "source authority/runtime lifecycle drift")
        roster = tuple(runtime["roster"])
        train_files = tuple(runtime["train_files"])
        _exact(list(roster), identity.source_schedule["sealed_arm_a_source_authority"]["roster"],
               "source authority live roster")
        _require(len(train_files) == len(roster), "source authority train file count drift")
        source_paths: list[str] = []
        for session, path in zip(roster, train_files, strict=True):
            candidate = Path(path).absolute()
            try:
                relative = candidate.relative_to(self.root)
            except ValueError as error:
                raise ContractError("live strict-train path lies outside repository root") from error
            rendered = relative.as_posix()
            _exact(rendered, f"{SOURCE_DATA_RELATIVE}/{session}_behavior+ecephys.nwb",
                   "source authority canonical direct train path")
            source_paths.append(rendered)
        inventory = runtime["inventory"]
        plan = runtime["plan"]
        _require(isinstance(inventory, SourceInventory) and isinstance(plan, Mapping),
                 "source authority runtime inventory/plan type drift")
        payload: dict[str, object] = {
            "schema": "cell_d_equal_session_source_authority_v1",
            "cell": CELL,
            "run_spec": spec.payload(),
            "strict_train_roster": list(roster),
            "train_source_relative_paths": source_paths,
            "source_inventory": {
                **inventory.payload(PUBLIC_SPEC),
                "predecessor_consumed_windows": inventory.full_batches(PUBLIC_SPEC) * PUBLIC_SPEC.batch_size,
            },
            "schedule_plan": dict(plan),
            "per_session_exposure": original_vs_equal_exposure(PUBLIC_SPEC, inventory, plan),
            "normalizers": {
                "behavior_semantic_sha256": runtime["behavior_normalizer_semantic_sha256"],
                "t4_semantic_sha256": runtime["t4_normalizer_semantic_sha256"],
            },
            "runtime_environment": dict(runtime["runtime_environment"]),
            "t4_authority_sha256": dict(runtime["t4_authority_fingerprint"]),
            "data_loading_control_flow": {
                "predecessor_shared_builder": "_initialize_splits then clear val before setup(fit)",
                "predecessor_preclear_metadata_read": "shared _initialize_splits currently calls nwb_unit_count over train+val",
                "successor_train_only_repair": "bind sealed train paths; set val/test empty; sentinel-skip empty val load_split; mark splits initialized; setup(fit); restore val=None",
                "nontrain_path_or_nwb_metadata_open": False,
                "bytewise_identical_shared_builder_control_flow": False,
                "science_factor_changed": False,
            },
            "access_disclosure": flags.disclosure(),
        }
        _validate_source_authority_payload(payload, spec, identity)
        return payload

    def begin_epoch(self, runtime: Mapping[str, Any], epoch: int, flags: LifecycleFlags) -> Mapping[str, object]:
        torch = runtime["torch"]
        from torch.utils.data import DataLoader

        schedule = EpochSchedule(spec=PUBLIC_SPEC, inventory=runtime["inventory"], epoch=epoch)
        schedule_sha = runtime["ledger"].claim(schedule)
        sampler = _OneIteratorEpochBatchSampler(
            schedule,
            maximum_batches=1 if runtime["lifecycle_spec"].kind == "source_smoke" else None,
        )
        before = _torch_rng_snapshot(torch, runtime["numpy"])
        loader = DataLoader(runtime["dataset"], batch_sampler=sampler, num_workers=self.num_workers, pin_memory=True)
        after_constructor = _torch_rng_snapshot(torch, runtime["numpy"])
        _require(before == after_constructor, "DataLoader construction consumed global Python/NumPy/Torch RNG")
        flags.data_loader_constructor_rng_unchanged &= before == after_constructor
        # Exactly one iterator per planned epoch; its normal worker-seed draw is
        # intentionally the same iterator-level behavior as the predecessor.
        iterator = iter(loader)
        flags.data_loader_iterators_created += 1
        runtime["epoch_iterator"] = iterator
        runtime["epoch_sampler"] = sampler
        runtime["epoch_schedule"] = schedule
        evidence = dict(epoch_schedule_evidence(schedule))
        _exact(evidence.get("schedule_sha256"), schedule_sha, "claimed/current epoch schedule SHA")
        evidence["full_plan_sha256"] = runtime["plan"]["full_plan_sha256"]
        evidence["consumed_batch_count"] = len(sampler)
        evidence["full_epoch_batch_count"] = len(schedule)
        evidence["source_smoke_prefix_only"] = runtime["lifecycle_spec"].kind == "source_smoke"
        return evidence

    def train_step(self, runtime: Mapping[str, Any], *, global_step: int, expected_lr: float,
                   require_full_proof: bool, flags: LifecycleFlags) -> StepOutcome:
        torch = runtime["torch"]
        iterator = runtime.get("epoch_iterator")
        sampler = runtime.get("epoch_sampler")
        _require(iterator is not None and isinstance(sampler, _OneIteratorEpochBatchSampler),
                 "train step lacks an explicitly-bound one-iterator epoch loader")
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise ContractError("DataLoader exhausted before frozen epoch step count") from error
        scheduled = sampler.consume_observed_batch()
        neural, behavior, calib, sessions, side = batch[:5]
        observed_sessions = tuple(sessions)
        _require(len(observed_sessions) == PUBLIC_SPEC.batch_size
                 and set(observed_sessions) == {scheduled.session}, "loaded batch/session schedule mismatch")
        neural, behavior, calib, side = (
            neural.to(runtime["device"]), behavior.to(runtime["device"]),
            calib.to(runtime["device"]), side.to(runtime["device"]),
        )
        optimizer = runtime["optimizer"]
        optimizer.param_groups[0]["lr"] = expected_lr
        optimizer.zero_grad(set_to_none=True)
        with runtime["pop_robust"].dynamic_dropout_recorder() as dropout_record:
            prediction, _identity = runtime["model"](neural, calib_trials=calib, side_features=side)
        valid = (behavior != -1.0).all(dim=-1)
        diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
        _require(bool(torch.isfinite(loss).item()), "nonfinite equal-session dense loss")
        loss.backward()
        flags.backward_calls += 1
        critical = _critical_gradient_proof(runtime["model"], torch) if require_full_proof else None
        optimizer.step()
        flags.update_calls += 1
        calls = dropout_record["dropout_calls"]
        sampled = dropout_record["sampled_p"]
        _require(len(sampled) == 1 and len(calls) == 1, "dynamic U(0,1) dropout call topology drift")
        call = calls[0]
        shape = call.get("shape")
        _require(isinstance(shape, list) and len(shape) == 2 and shape[0] == PUBLIC_SPEC.batch_size,
                 "dynamic dropout unit-mask shape drift")
        total_mask = int(shape[0]) * int(shape[1])
        kept = int(round(float(call["retained_unit_fraction"]) * total_mask))
        dropped = total_mask - kept
        p = float(sampled[0])
        gain = 0.0 if p == 1.0 else 1.0 / (1.0 - p)
        if require_full_proof:
            finite_model, finite_optimizer = _finite_model_and_optimizer(runtime["model"], optimizer, torch)
            _require(finite_model and finite_optimizer and critical is not None, "post-step full proof failed")
            model_sha = runtime["arm_common"].state_sha256(runtime["model"])
            optimizer_sha = runtime["arm_common"].optimizer_sha256(optimizer)
        else:
            finite_model = finite_optimizer = model_sha = optimizer_sha = None
        return StepOutcome(
            loss=float(loss.detach().item()), lr=expected_lr, dropout_p=p, kept=kept, dropped=dropped,
            all_zero_examples=int(call["all_zero_population_samples"]), population_examples=PUBLIC_SPEC.batch_size,
            max_gain=float(gain), critical_gradients=critical, finite_model=finite_model,
            finite_optimizer=finite_optimizer, model_state_sha256=model_sha,
            optimizer_state_sha256=optimizer_sha,
            batch_evidence={"session": scheduled.session, "dataset_indices": list(scheduled.dataset_indices),
                            "cycle": scheduled.cycle, "epoch": scheduled.epoch,
                            "global_step": global_step},
        )

    def end_epoch(self, runtime: Mapping[str, Any], epoch: int, outcomes: Sequence[StepOutcome],
                  flags: LifecycleFlags) -> Mapping[str, object]:
        iterator = runtime.get("epoch_iterator")
        sampler = runtime.get("epoch_sampler")
        _require(iterator is not None and isinstance(sampler, _OneIteratorEpochBatchSampler),
                 "epoch finalizer lacks iterator/sampler")
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ContractError("DataLoader emitted more batches than frozen epoch schedule")
        sampler.assert_complete()
        runtime["epoch_iterator"] = None
        if epoch == 47:
            runtime["ledger"].final_assert_exact()
        current_t4 = runtime["arm_common"].t4_authority_fingerprint(runtime["dataset"].sessions)
        _exact(current_t4, runtime["t4_authority_fingerprint"], "T4 authority changed during train epoch")
        if runtime["lifecycle_spec"].kind == "source_smoke":
            # A smoke is contractually exactly one optimizer step, not an
            # abbreviated epoch plus a second diagnostic forward.
            diagnostic = None
            fixed_forwards = 0
        else:
            # Same one forward-only fixed dropout diagnostic per epoch as the
            # predecessor, based on the first audited equal-session epoch-0 batch.
            neural, _behavior, calib, _sessions, side = runtime["fixed"]
            diagnostic = runtime["pop_robust"].fixed_batch_dropout_diagnostic(
                runtime["model"], neural[:8], calib[:8], side[:8],
            )
            flags.fixed_diagnostic_forwards += 1
            fixed_forwards = 1
        return {
            "one_iterator": True, "sampler_iter_calls": sampler.iter_calls,
            "scheduled_batches_consumed": sampler.emitted_batches,
            "observed_batches_consumed": sampler.observed_batches,
            "fixed_diagnostic_forwards_this_epoch": fixed_forwards,
            "fixed_batch_schedule": runtime["fixed_batch_schedule"],
            "fixed_diagnostic_input_rng_unchanged": runtime["fixed_diagnostic_input_rng_unchanged"],
            "fixed_batch_dropout_diagnostic": diagnostic,
            "behavior_normalizer_semantic_sha256": runtime["behavior_normalizer_semantic_sha256"],
            "t4_normalizer_semantic_sha256": runtime["t4_normalizer_semantic_sha256"],
            "t4_authority_unchanged": True,
        }

    def synchronize_for_measurement(self, runtime: Mapping[str, Any]) -> None:
        runtime["torch"].cuda.synchronize(0)

    def resources(self, runtime: Mapping[str, Any]) -> Mapping[str, object]:
        torch = runtime["torch"]
        # End-of-epoch and smoke resource evidence is also synchronized: peak
        # allocator counters then describe completed work, not a queued kernel.
        torch.cuda.synchronize(0)
        return {
            "cuda_memory_allocated": int(torch.cuda.memory_allocated(0)),
            "cuda_memory_reserved": int(torch.cuda.memory_reserved(0)),
            "cuda_peak_memory_allocated": int(torch.cuda.max_memory_allocated(0)),
            "cuda_peak_memory_reserved": int(torch.cuda.max_memory_reserved(0)),
            "peak_stats_reset_before_training": bool(runtime["peak_stats_reset_before_training"]),
            "rss_bytes": _rss_bytes(),
            "throughput_windows_per_s": 0,
        }

    def make_checkpoint(self, runtime: Mapping[str, Any], epoch: int, global_step: int,
                        binding: Mapping[str, object]) -> CheckpointPayload:
        torch = runtime["torch"]
        state_sha = runtime["arm_common"].state_sha256(runtime["model"])
        buffer = io.BytesIO()
        torch.save({"schema": "cell_d_equal_session_checkpoint_v1", "cell": CELL,
                    "epoch": epoch, "global_step": global_step,
                    "state_dict": runtime["model"].state_dict(), "state_dict_sha256": state_sha,
                    "binding": dict(binding)}, buffer)
        return CheckpointPayload(body=buffer.getvalue(), model_state_sha256=state_sha)

    def validate_checkpoint(self, body: bytes, *, epoch: int, global_step: int,
                            spec: LifecycleSpec, binding: Mapping[str, object]) -> Mapping[str, object]:
        torch = _runtime_torch()
        from torch.nn.parameter import UninitializedParameter

        with torch.serialization.safe_globals([UninitializedParameter]):
            value = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        _require(isinstance(value, Mapping) and value.get("schema") == "cell_d_equal_session_checkpoint_v1"
                 and value.get("cell") == CELL and value.get("epoch") == epoch
                 and value.get("global_step") == global_step and _is_sha256(value.get("state_dict_sha256"))
                 and value.get("binding") == dict(binding) and isinstance(value.get("state_dict"), Mapping),
                 "checkpoint schema/binding drift")
        _strict_recompute_cell_d_state_digest(
            root=self.root,
            torch=torch,
            state_dict=value["state_dict"],
            claimed_sha256=value["state_dict_sha256"],
            label=f"checkpoint epoch {epoch}",
        )
        return value

    def build_swa(self, runtime: Mapping[str, Any], checkpoints: Mapping[int, bytes], spec: LifecycleSpec,
                  binding: Mapping[str, object]) -> SWAPayload:
        torch = runtime["torch"]
        from torch.nn.parameter import UninitializedParameter

        _require(set(checkpoints) == set(spec.checkpoint_epochs), "SWA checkpoint set drift")
        states: list[Mapping[str, Any]] = []
        component_sha: dict[str, str] = {}
        for epoch in spec.checkpoint_epochs:
            value = self.validate_checkpoint(
                checkpoints[epoch], epoch=epoch, global_step=(epoch + 1) * spec.steps_per_epoch,
                spec=spec, binding=binding,
            )
            states.append(value["state_dict"])
            component_sha[str(epoch)] = str(value["state_dict_sha256"])
        keys = list(states[0])
        _require(all(list(state) == keys for state in states[1:]), "SWA state key topology drift")
        swa_state: dict[str, Any] = {}
        for key in keys:
            values = [state[key] for state in states]
            first = values[0]
            if isinstance(first, UninitializedParameter):
                _require(all(isinstance(item, UninitializedParameter) for item in values),
                         "SWA lazy-state topology drift")
                swa_state[key] = first
            elif torch.is_floating_point(first):
                total = torch.zeros_like(first, dtype=torch.float64, device="cpu")
                for item in values:
                    _require(torch.is_floating_point(item) and tuple(item.shape) == tuple(first.shape),
                             "SWA floating tensor topology drift")
                    total.add_(item.detach().cpu().to(torch.float64))
                swa_state[key] = (total / len(values)).to(dtype=first.dtype)
            else:
                _require(all(torch.equal(first.detach().cpu(), item.detach().cpu()) for item in values[1:]),
                         "SWA non-floating state differs across components")
                swa_state[key] = first.detach().cpu().clone()
        fresh = runtime["pop_robust"].build_population_robustness_model(seed=PUBLIC_SPEC.seed, cell="D")
        fresh.load_state_dict(swa_state, strict=True)
        fresh.to(runtime["device"]).eval()
        before = runtime["arm_common"].state_sha256(fresh)
        neural, _behavior, calib, _sessions, side = runtime["fixed"]
        with runtime["pop_robust"].dynamic_dropout_recorder() as recorder:
            with torch.no_grad():
                first_prediction, _ = fresh(neural[:4], calib_trials=calib[:4], side_features=side[:4])
                second_prediction, _ = fresh(neural[:4], calib_trials=calib[:4], side_features=side[:4])
        after = runtime["arm_common"].state_sha256(fresh)
        _require(tuple(first_prediction.shape) == (4, 50, 2) and bool(torch.isfinite(first_prediction).all().item())
                 and torch.equal(first_prediction, second_prediction) and before == after
                 and recorder["uniform_calls"] == 0 and recorder["dropout_calls"] == [],
                 "SWA fresh strict-reload/eval/no-mask proof drift")
        state_sha = runtime["arm_common"].state_sha256(fresh)
        buffer = io.BytesIO()
        proof = {
            "window_epochs": list(spec.checkpoint_epochs), "component_state_sha256": component_sha,
            "fp64_arithmetic": True, "fresh_strict_load": True, "eval_mode": True,
            "eval_no_mask": True, "repeat_bitwise_equal": True, "state_unchanged": True,
            "prediction_shape": [4, 50, 2], "prediction_sha256": _sha256(first_prediction.detach().cpu().numpy().tobytes()),
            "state_sha256_before_eval": before, "state_sha256_after_eval": after,
            "uninitialized_lazy_keys": ["decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight"],
        }
        torch.save({"schema": "cell_d_equal_session_swa_v1", "cell": CELL,
                    "state_dict": fresh.state_dict(), "state_dict_sha256": state_sha,
                    "binding": dict(binding), "proof": proof}, buffer)
        return SWAPayload(body=buffer.getvalue(), state_sha256=state_sha, proof=proof)

    def validate_swa(self, body: bytes, *, spec: LifecycleSpec,
                     binding: Mapping[str, object]) -> Mapping[str, object]:
        torch = _runtime_torch()
        from torch.nn.parameter import UninitializedParameter

        with torch.serialization.safe_globals([UninitializedParameter]):
            value = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        proof = value.get("proof") if isinstance(value, Mapping) else None
        _require(isinstance(value, Mapping) and value.get("schema") == "cell_d_equal_session_swa_v1"
                 and value.get("cell") == CELL and _is_sha256(value.get("state_dict_sha256"))
                 and value.get("binding") == dict(binding) and isinstance(value.get("state_dict"), Mapping)
                 and isinstance(proof, Mapping)
                 and set(proof) == {
                     "window_epochs", "component_state_sha256", "fp64_arithmetic", "fresh_strict_load",
                     "eval_mode", "eval_no_mask", "repeat_bitwise_equal", "state_unchanged",
                     "prediction_shape", "prediction_sha256", "state_sha256_before_eval",
                     "state_sha256_after_eval", "uninitialized_lazy_keys",
                 }
                 and proof.get("window_epochs") == list(spec.checkpoint_epochs)
                 and isinstance(proof.get("component_state_sha256"), Mapping)
                 and set(proof["component_state_sha256"]) == {str(epoch) for epoch in spec.checkpoint_epochs}
                 and all(_is_sha256(item) for item in proof["component_state_sha256"].values())
                 and proof.get("fp64_arithmetic") is True and proof.get("fresh_strict_load") is True
                 and proof.get("eval_mode") is True and proof.get("eval_no_mask") is True
                 and proof.get("repeat_bitwise_equal") is True
                 and proof.get("state_unchanged") is True
                 and proof.get("prediction_shape") == [4, 50, 2]
                 and _is_sha256(proof.get("prediction_sha256"))
                 and _is_sha256(proof.get("state_sha256_before_eval"))
                 and _is_sha256(proof.get("state_sha256_after_eval"))
                 and proof.get("uninitialized_lazy_keys") == list(CELL_D_UNINITIALIZED_LAZY_KEYS),
                 "SWA schema/binding/proof drift")
        recomputed = _strict_recompute_cell_d_state_digest(
            root=self.root,
            torch=torch,
            state_dict=value["state_dict"],
            claimed_sha256=value["state_dict_sha256"],
            label="SWA",
        )
        _exact(proof.get("state_sha256_before_eval"), recomputed["state_dict_sha256"],
               "SWA proof before-eval state SHA")
        _exact(proof.get("state_sha256_after_eval"), recomputed["state_dict_sha256"],
               "SWA proof after-eval state SHA")
        return value

    def close(self, runtime: Any | None) -> None:
        if not isinstance(runtime, Mapping):
            return
        iterator = runtime.get("epoch_iterator")
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()


def _load_runtime_module(name: str, path: Path) -> Any:
    """Deferred local import with no module-level runtime dependency."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"runtime module loader failed: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_torch() -> Any:
    """Resolve already-installed Torch only inside checkpoint/SWA physical paths."""
    import importlib

    return importlib.import_module("torch")


# ---------------------------------------------------------------------------
# Dependency-injected no-data backend for lifecycle tests
# ---------------------------------------------------------------------------


class DeterministicMockBackend:
    """Synthetic backend used only by focused CPU/no-data lifecycle tests."""

    def __init__(self, *, fail_stage: str | None = None) -> None:
        self.fail_stage = fail_stage
        self.lr_for_step: Callable[[int], float] = lambda step: 1e-4 / (step + 1)
        self.closed = False
        self.prepared = False
        self.current_epoch: int | None = None
        self.steps_by_epoch: dict[int, int] = {}
        self.iterator_count = 0
        self.measurement_sync_count = 0

    def _maybe_fail(self, stage: str) -> None:
        if self.fail_stage == stage:
            raise ContractError(f"synthetic requested failure at {stage}")

    def prepare(self, spec: LifecycleSpec, identity: RunIdentity, flags: LifecycleFlags) -> dict[str, object]:
        self._maybe_fail("prepare")
        self.prepared = True
        flags.source_train_opened = True
        return {"spec": spec.payload()}

    def source_authority(self, runtime: Any, spec: LifecycleSpec, identity: RunIdentity,
                         flags: LifecycleFlags) -> Mapping[str, object]:
        self._maybe_fail("source_authority")
        if spec in (FULL_TRAIN_SPEC, SOURCE_SMOKE_SPEC):
            sealed = identity.source_schedule["sealed_arm_a_source_authority"]
            roster = list(sealed["roster"])
            # Coherent synthetic counts only: these never stand in for the
            # sealed real per-session census, but they exercise every public
            # authority validator without opening an NWB.
            mock_full_batches = [1_256] * 26 + [1_269]
            mock_remainders = [31] * 13 + [4] + [0] * 13
            offset = 0
            mock_sessions: list[dict[str, object]] = []
            for session, full_batches, remainder in zip(
                roster, mock_full_batches, mock_remainders, strict=True,
            ):
                eligible = full_batches * PUBLIC_SPEC.batch_size + remainder
                mock_sessions.append({
                    "session": session,
                    "eligible_windows": eligible,
                    "full_batches": full_batches,
                    "remainder_windows": remainder,
                    "dataset_index_min": offset,
                    "dataset_index_max": offset + eligible - 1,
                    "valid_starts_sha256": "c" * 64,
                })
                offset += eligible
            inventory = {
                "roster": roster,
                "sessions": mock_sessions,
                "eligible_windows": SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS,
                "full_batches": PUBLIC_SPEC.steps_per_epoch,
                "predecessor_consumed_windows": SOURCE_ONLY_EXPECTED_PREDECESSOR_CONSUMED_WINDOWS,
                "remainder_windows": SOURCE_ONLY_EXPECTED_REMAINDER_WINDOWS,
            }
            plan_inventory = dict(inventory)
            plan_inventory.pop("predecessor_consumed_windows")
            # Public mock routes are not executed by tests, but retaining a
            # structurally complete authority makes their validator coverage
            # independent of the physical backend.
            plan = {
                "schema": "cell_d_equal_session_full_plan_v1",
                "spec": PUBLIC_SPEC.payload(),
                "inventory": plan_inventory,
                "epochs": [
                    {
                        "epoch": epoch,
                        "schedule_sha256": (
                            AUDITED_EPOCH0_SCHEDULE_SHA256 if epoch == 0 else
                            AUDITED_EPOCH47_SCHEDULE_SHA256 if epoch == 47 else
                            _sha256(f"synthetic-public-epoch:{epoch}".encode())
                        ),
                    }
                    for epoch in range(48)
                ],
                "full_run_batches_by_session": full_run_exposure(PUBLIC_SPEC, roster),
                "full_run_max_minus_min_batches": 1,
                "full_plan_sha256": AUDITED_FULL_PLAN_SHA256,
                "global_rng_unchanged": True,
                "plan_retains_batch_lists": False,
            }
            exposure = [
                {
                    "session": item["session"],
                    "eligible_windows": item["eligible_windows"],
                    "full_batches_predecessor": item["full_batches"],
                    "remainder_windows_discarded_per_epoch": item["remainder_windows"],
                    "equal_batches_full_run": plan["full_run_batches_by_session"][str(item["session"])],
                }
                for item in mock_sessions
            ]
            return {
                "schema": "cell_d_equal_session_source_authority_v1",
                "cell": CELL,
                "run_spec": spec.payload(),
                "strict_train_roster": roster,
                "train_source_relative_paths": [
                    f"{SOURCE_DATA_RELATIVE}/{session}_behavior+ecephys.nwb" for session in roster
                ],
                "source_inventory": inventory,
                "schedule_plan": plan,
                "per_session_exposure": exposure,
                "normalizers": {
                    "behavior_semantic_sha256": SEALED_BEHAVIOR_NORMALIZER_SHA256,
                    "t4_semantic_sha256": SEALED_T4_NORMALIZER_SHA256,
                },
                "runtime_environment": _synthetic_runtime_environment_payload(),
                "t4_authority_sha256": dict(sealed["t4_authority_sha256"]),
                "data_loading_control_flow": {
                    "nontrain_path_or_nwb_metadata_open": False,
                    "bytewise_identical_shared_builder_control_flow": False,
                    "science_factor_changed": False,
                },
                "access_disclosure": flags.disclosure(),
            }
        return {
            "schema": "cell_d_equal_session_source_authority_v1",
            "cell": CELL,
            "run_spec": spec.payload(),
            "runtime_environment": _synthetic_runtime_environment_payload(),
            "access_disclosure": flags.disclosure(),
        }

    def begin_epoch(self, runtime: Any, epoch: int, flags: LifecycleFlags) -> Mapping[str, object]:
        self._maybe_fail("begin_epoch")
        _require(self.prepared and self.current_epoch is None, "mock epoch topology drift")
        self.current_epoch = epoch
        self.steps_by_epoch[epoch] = 0
        self.iterator_count += 1
        if runtime["spec"]["kind"] == "source_smoke":
            return {
                "epoch": 0, "schedule_sha256": AUDITED_EPOCH0_SCHEDULE_SHA256,
                "full_plan_sha256": AUDITED_FULL_PLAN_SHA256,
                "batch_count": PUBLIC_SPEC.steps_per_epoch,
                "consumed_batch_count": 1, "full_epoch_batch_count": PUBLIC_SPEC.steps_per_epoch,
                "source_smoke_prefix_only": True, "one_iterator": True,
            }
        return {"epoch": epoch, "schedule_sha256": _sha256(f"mock-schedule:{epoch}".encode()),
                "batch_count": int(runtime["spec"]["steps_per_epoch"]), "one_iterator": True}

    def train_step(self, runtime: Any, *, global_step: int, expected_lr: float,
                   require_full_proof: bool, flags: LifecycleFlags) -> StepOutcome:
        self._maybe_fail("step")
        _require(self.current_epoch is not None, "mock train step outside epoch")
        self.steps_by_epoch[self.current_epoch] += 1
        flags.backward_calls += 1
        flags.update_calls += 1
        proof = {key: True for key in CRITICAL_GRADIENT_PATHS} if require_full_proof else None
        digest = _sha256(f"mock-state:{self.current_epoch}:{global_step}".encode()) if require_full_proof else None
        return StepOutcome(
            loss=1.0 / (global_step + 1), lr=expected_lr, dropout_p=0.25,
            kept=24, dropped=8, all_zero_examples=0, population_examples=4, max_gain=4.0 / 3.0,
            critical_gradients=proof, finite_model=True if require_full_proof else None,
            finite_optimizer=True if require_full_proof else None, model_state_sha256=digest,
            optimizer_state_sha256=(None if digest is None else _sha256((digest + ":optim").encode())),
            batch_evidence={"session": f"synthetic-{self.current_epoch}", "dataset_indices": [0, 1, 2, 3],
                            "cycle": 0, "epoch": self.current_epoch, "global_step": global_step},
        )

    def end_epoch(self, runtime: Any, epoch: int, outcomes: Sequence[StepOutcome],
                  flags: LifecycleFlags) -> Mapping[str, object]:
        self._maybe_fail("end_epoch")
        _require(self.current_epoch == epoch and self.steps_by_epoch[epoch] == len(outcomes),
                 "mock end-epoch accounting drift")
        self.current_epoch = None
        fixed_forwards = 0 if runtime["spec"]["kind"] == "source_smoke" else 1
        flags.fixed_diagnostic_forwards += fixed_forwards
        return {"one_iterator": True, "sampler_iter_calls": 1,
                "scheduled_batches_consumed": len(outcomes), "observed_batches_consumed": len(outcomes),
                "fixed_diagnostic_forwards_this_epoch": fixed_forwards,
                "fixed_batch_schedule": {"synthetic": True}}

    def synchronize_for_measurement(self, runtime: Any) -> None:
        self.measurement_sync_count += 1

    def resources(self, runtime: Any) -> Mapping[str, object]:
        return {
            "cuda_memory_allocated": 0,
            "cuda_memory_reserved": 0,
            "cuda_peak_memory_allocated": 0,
            "cuda_peak_memory_reserved": 0,
            "peak_stats_reset_before_training": True,
            "rss_bytes": 1,
            "throughput_windows_per_s": 0,
        }

    def make_checkpoint(self, runtime: Any, epoch: int, global_step: int,
                        binding: Mapping[str, object]) -> CheckpointPayload:
        self._maybe_fail("checkpoint")
        state = _sha256(f"mock-checkpoint:{epoch}:{global_step}".encode())
        payload = {"schema": "cell_d_equal_session_mock_checkpoint_v1", "cell": CELL,
                   "epoch": epoch, "global_step": global_step, "state_dict_sha256": state,
                   "binding": dict(binding)}
        return CheckpointPayload(_canonical_json_bytes(payload), state)

    def validate_checkpoint(self, body: bytes, *, epoch: int, global_step: int,
                            spec: LifecycleSpec, binding: Mapping[str, object]) -> Mapping[str, object]:
        value = json.loads(body)
        _require(isinstance(value, Mapping) and value.get("schema") == "cell_d_equal_session_mock_checkpoint_v1"
                 and value.get("cell") == CELL and value.get("epoch") == epoch
                 and value.get("global_step") == global_step and _is_sha256(value.get("state_dict_sha256"))
                 and value.get("binding") == dict(binding), "mock checkpoint validation drift")
        return value

    def build_swa(self, runtime: Any, checkpoints: Mapping[int, bytes], spec: LifecycleSpec,
                  binding: Mapping[str, object]) -> SWAPayload:
        self._maybe_fail("swa")
        _require(set(checkpoints) == set(spec.checkpoint_epochs), "mock SWA checkpoint topology drift")
        state = _sha256(b"mock-swa:" + b"|".join(checkpoints[epoch] for epoch in spec.checkpoint_epochs))
        proof = {
            "window_epochs": list(spec.checkpoint_epochs), "fp64_arithmetic": True,
            "fresh_strict_load": True, "eval_mode": True, "eval_no_mask": True,
            "repeat_bitwise_equal": True, "state_unchanged": True,
        }
        return SWAPayload(_canonical_json_bytes({"schema": "cell_d_equal_session_mock_swa_v1", "cell": CELL,
                                                  "state_dict_sha256": state, "binding": dict(binding),
                                                  "proof": proof}), state, proof)

    def validate_swa(self, body: bytes, *, spec: LifecycleSpec,
                     binding: Mapping[str, object]) -> Mapping[str, object]:
        value = json.loads(body)
        proof = value.get("proof") if isinstance(value, Mapping) else None
        _require(isinstance(value, Mapping) and value.get("schema") == "cell_d_equal_session_mock_swa_v1"
                 and value.get("cell") == CELL and _is_sha256(value.get("state_dict_sha256"))
                 and value.get("binding") == dict(binding) and isinstance(proof, Mapping)
                 and proof.get("window_epochs") == list(spec.checkpoint_epochs)
                 and proof.get("fresh_strict_load") is True and proof.get("eval_no_mask") is True,
                 "mock SWA validation drift")
        return value

    def close(self, runtime: Any | None) -> None:
        self.closed = True


def synthetic_identity() -> RunIdentity:
    """A no-file identity allowed only for lifecycle unit tests."""
    return RunIdentity(
        predecessor={"cell": "D", "trainable_parameters": 3_510_842},
        closure={"schema": "cell_d_equal_session_implementation_closure_v1",
                 "paths": list(IMPLEMENTATION_CLOSURE), "closure_sha256": "a" * 64},
        source_schedule={
            "strict_manifest_sha256": MANIFEST_SHA256, "plan_sha256": AUDITED_FULL_PLAN_SHA256,
            "epoch0_schedule_sha256": AUDITED_EPOCH0_SCHEDULE_SHA256,
            "epoch47_schedule_sha256": AUDITED_EPOCH47_SCHEDULE_SHA256,
            "source_only_expected_eligible_windows": SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS,
            "source_only_expected_predecessor_consumed_windows": SOURCE_ONLY_EXPECTED_PREDECESSOR_CONSUMED_WINDOWS,
            "source_only_expected_remainder_windows": SOURCE_ONLY_EXPECTED_REMAINDER_WINDOWS,
            "spec": PUBLIC_SPEC.payload(),
            "sealed_arm_a_source_authority": {
                "preflight_receipt_sha256": SEALED_ARM_A_PREFLIGHT_SHA256,
                "roster": [f"synthetic-source-{index:02d}" for index in range(27)],
                "behavior_normalizer_semantic_sha256": SEALED_BEHAVIOR_NORMALIZER_SHA256,
                "t4_normalizer_semantic_sha256": SEALED_T4_NORMALIZER_SHA256,
                "t4_authority_sha256": {f"synthetic-source-{index:02d}": "b" * 64 for index in range(27)},
            },
        },
        canonical_initial_state={
            "artifact_sha256": CANONICAL_INITIAL_STATE_SHA256,
            "state_dict_sha256": CANONICAL_INITIAL_STATE_STATE_SHA256,
            "initialized_trainable_parameters": 3_510_842,
            "uninitialized_lazy_keys": ["decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight"],
        },
        failed_smoke_v1_lineage=_failed_smoke_v1_lineage_payload(),
        gpu=dict(FROZEN_GPU0),
    )


def execute_reviewed_training(
    *,
    root: Path,
    execute_flag: bool,
    acknowledgement_flag: bool,
    capability: RootReviewedExecutionCapability | None,
    source_smoke: bool = False,
    backend_factory: Callable[[Path], TrainingBackend] | None = None,
) -> Mapping[str, object]:
    """Capability-gated physical/mocked entrypoint; public CLI supplies none.

    Fresh root reservation happens only after flags, the in-process capability,
    immutable source closure, and literal GPU0 environment have all passed.
    Thus an ordinary CLI invocation cannot create a directory, resolve a data
    surface, import Torch, or initialize CUDA.
    """
    spec = SOURCE_SMOKE_SPEC if source_smoke else FULL_TRAIN_SPEC
    _require(execute_flag and acknowledgement_flag, "equal-session execution requires both exact flags")
    _require_public_lifecycle(spec)
    root = Path(root).absolute()
    identity = _build_run_identity(root)
    _validate_run_identity(identity, spec)
    _validate_capability(capability, identity, spec)
    validate_future_gpu0_environment()
    assert_fresh_candidate_roots(root, required_roots=(spec.root_relative,))
    artifact = reserve_artifact_root(root, spec.root_relative, spec.topology)
    backend = backend_factory(root) if backend_factory is not None else PhysicalEqualSessionBackend(root)
    return run_lifecycle(spec=spec, identity_factory=lambda: _build_run_identity(root), backend=backend, artifact=artifact)
