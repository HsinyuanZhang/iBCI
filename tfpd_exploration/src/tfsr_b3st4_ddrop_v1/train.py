"""Fail-closed, source-only Phase-D training lifecycle for TF-SR.

The public route is deliberately small and fixed: one 48-epoch seed-42
source training run.  The lifecycle itself is dependency-injected so that it
can be exercised end-to-end with a two-epoch CPU-free mock without exposing a
test-only CLI knob, source file, target file, or CUDA device.

This module has no top-level torch/data import.  Those imports live strictly
inside :class:`TorchTrainingBackend.prepare`, which is reachable only through
the two-flag authorized public route.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import source_smoke
from .contract import _canonical_regular_bytes, compute_live_closure


CELL = "TFSR_B3ST4_DDROP_SEED42"
# v1 is immutable failed-at-boundary evidence.  v2 is a fresh successor, not
# a resume or a renamed copy of that attempt.
TRAIN_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_train_v2"
PREDECESSOR_TRAIN_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_train_v1"
PREDECESSOR_FAILURE_RELATIVE = f"{PREDECESSOR_TRAIN_ROOT_RELATIVE}/failure.json"
PREDECESSOR_FAILURE_SHA256 = "997be82b3fdaf3774fba9c1428fd9526ba7dd56e6cbf3a2734fd917ef4c5f0a4"
PREDECESSOR_BODY_SHA256 = {
    "attempt.json": "47b57b023840aabad44b1d8aacf95c67e925b7226351e6cdfed90396cde2ff71",
    "launch.json": "5e8359da3744c11be2c782f023135f5e46e2a89a4cb0b20c5187b742364d85d8",
    "throughput100.json": "85e9ad521b410e7b585aae659064c52445a685672d1ace76a3b491c769544362",
    "failure.json": PREDECESSOR_FAILURE_SHA256,
}
PREDECESSOR_EXPECTED_LEAVES = frozenset(
    leaf for name in PREDECESSOR_BODY_SHA256 for leaf in (name, f"{name}.sha256")
)
SUCCESSOR_WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_TFSR_PHASE_D_SUCCESSOR_V2_20260819.md"
SUCCESSOR_WORKORDER_SHA256 = "5286a648e9ddf889fc2cd98475fa6c428a3a52f52264711c528f6cfef01d6b20"
PHASE_C_RECEIPT_RELATIVE = source_smoke.CANONICAL_RECEIPT_RELATIVE_PATH
PHASE_C_RECEIPT_SHA = "022ca7a253e208c86c846b593bc684372ac9ab21197db7a974277416df90dfbc"
PHASE_D_CLOSURE = (
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/train.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/contract.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/__init__.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/model.py",
    "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed42_train.py",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_train.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/source_smoke.py",
    "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_source_smoke.py",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_source_smoke.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd/bilinear_readin.py",
    SUCCESSOR_WORKORDER_RELATIVE,
)

OPTIMIZER = {
    "cls": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
    "weight_decay": 0.0, "amsgrad": False, "gradient_clipping": None,
}
_CRITICAL_GRADIENT_GROUPS = tuple(source_smoke._CRITICAL_PARAMETER_PREFIXES)
_RESOURCE_KEYS = ("rss_bytes", "peak_allocated_bytes", "peak_reserved_bytes")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite_number(value: object, *, positive: bool = False, nonnegative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and (not positive or number > 0.0) and (not nonnegative or number >= 0.0)


def _strict_bool_map(value: object, keys: Sequence[str], label: str) -> Mapping[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(keys) or any(item is not True for item in value.values()):
        raise RuntimeError(f"{label} drift")
    return value


def _safe_mapping_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Detach a JSON-shaped metadata map from callers before receipt use."""
    return json.loads(json.dumps(dict(value), sort_keys=True))


@dataclass(frozen=True)
class RunSpec:
    """Immutable run budget injected into the lifecycle, never parsed from CLI."""

    epochs: int
    batch_size: int
    steps_per_epoch: int
    checkpoint_epochs: tuple[int, ...]
    throughput_probe_steps: int
    seed: int = 42
    capture_diagnostics: bool = False

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (self.epochs, self.batch_size, self.steps_per_epoch, self.throughput_probe_steps)):
            raise ValueError("RunSpec integer budgets must be positive exact ints")
        if self.seed != 42 or self.capture_diagnostics is not False:
            raise ValueError("Phase-D fixes seed=42 and capture_diagnostics=False")
        if not isinstance(self.checkpoint_epochs, tuple) or not self.checkpoint_epochs:
            raise ValueError("RunSpec must name nonempty checkpoint epochs")
        if tuple(sorted(self.checkpoint_epochs)) != self.checkpoint_epochs or len(set(self.checkpoint_epochs)) != len(self.checkpoint_epochs):
            raise ValueError("checkpoint epochs must be unique sorted tuple")
        if any(isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < self.epochs
               for epoch in self.checkpoint_epochs):
            raise ValueError("checkpoint epoch lies outside run")
        if self.throughput_probe_steps > self.total_optimizer_steps:
            raise ValueError("throughput probe lies outside run")

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def topology(self) -> tuple[str, ...]:
        return (
            "attempt.json", "launch.json", f"throughput{self.throughput_probe_steps}.json", "swa.pt",
            "terminal.json", "failure.json",
            *(f"epoch-{epoch:02d}.json" for epoch in range(self.epochs)),
            *(f"checkpoint-{epoch:02d}.pt" for epoch in self.checkpoint_epochs),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs, "batch_size": self.batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "total_optimizer_steps": self.total_optimizer_steps,
            "checkpoint_epochs": list(self.checkpoint_epochs),
            "throughput_probe_steps": self.throughput_probe_steps,
            "seed": self.seed, "capture_diagnostics": self.capture_diagnostics,
        }


PUBLIC_SPEC = RunSpec(
    epochs=48, batch_size=32, steps_per_epoch=33_925,
    checkpoint_epochs=(44, 45, 46, 47), throughput_probe_steps=100,
)
# Compatibility aliases make the frozen public accounting easy to audit.
EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH = PUBLIC_SPEC.epochs, PUBLIC_SPEC.batch_size, PUBLIC_SPEC.steps_per_epoch
TOTAL_STEPS, WARMUP_STEPS = PUBLIC_SPEC.total_optimizer_steps, 2 * PUBLIC_SPEC.steps_per_epoch
TOPOLOGY = PUBLIC_SPEC.topology


def phase_d_closure(root: Path) -> dict[str, object]:
    return compute_live_closure(root, PHASE_D_CLOSURE)


def _canonical_directory(path: Path, *, label: str) -> None:
    """Require one non-symlink directory at its declared canonical name."""
    absolute = path.absolute()
    try:
        if path.resolve(strict=True) != absolute:
            raise RuntimeError(f"{label} alias or symlink is forbidden")
        info = os.lstat(absolute)
    except OSError as error:
        raise RuntimeError(f"{label} is absent or unreadable") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"{label} is not a canonical directory")


def _validate_predecessor_failure_payload(value: object) -> None:
    expected = {
        "schema", "cell", "stage", "source_opened", "gpu_initialized",
        "optimizer_steps_completed", "target_or_formal_opened",
        "terminal_published", "traceback_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError("v1 failure receipt schema drift")
    if (value.get("schema") != "tfsr_b3st4_ddrop_train_failure_v2" or value.get("cell") != CELL
            or value.get("stage") != "optimizer_step" or value.get("source_opened") is not True
            or value.get("gpu_initialized") is not True or value.get("optimizer_steps_completed") != 33_924
            or value.get("target_or_formal_opened") is not False
            or value.get("terminal_published") is not False
            or not _is_sha(value.get("traceback_sha256"))):
        raise RuntimeError("v1 failure receipt semantic drift")


def _verify_predecessor_failure_at(root: Path, relative: str, expected_body_sha256: Mapping[str, str]) -> dict[str, Any]:
    """Descriptor-check immutable failed-v1 evidence before a v2 identity exists.

    The injectable arguments are used only by no-data adversarial tests.  The
    public caller uses the fixed canonical v1 path and exact SHA map above.
    """
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError("v1 predecessor relative path drift")
    predecessor = root.absolute() / relative
    _canonical_directory(predecessor, label="v1 predecessor root")
    before_directory = os.lstat(predecessor)
    directory_identity = (before_directory.st_dev, before_directory.st_ino)
    expected_leaves = frozenset(leaf for name in expected_body_sha256 for leaf in (name, f"{name}.sha256"))
    try:
        observed_leaves = frozenset(item.name for item in predecessor.iterdir())
    except OSError as error:
        raise RuntimeError("v1 predecessor directory cannot be enumerated") from error
    if observed_leaves != expected_leaves:
        raise RuntimeError("v1 predecessor topology drift")
    bodies: dict[str, bytes] = {}
    for name, expected_sha in expected_body_sha256.items():
        if not _is_sha(expected_sha):
            raise RuntimeError("v1 predecessor expected SHA is malformed")
        body = _canonical_regular_bytes(predecessor / name, expected_mode=0o444)
        if _sha(body) != expected_sha:
            raise RuntimeError("v1 predecessor body SHA drift")
        sidecar = _canonical_regular_bytes(predecessor / f"{name}.sha256", expected_mode=0o444)
        if sidecar != f"{expected_sha}  {name}\n".encode("ascii"):
            raise RuntimeError("v1 predecessor sidecar drift")
        bodies[name] = body
    try:
        failure = json.loads(bodies["failure.json"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("v1 predecessor failure is not valid JSON") from error
    _validate_predecessor_failure_payload(failure)
    forbidden = ("terminal.json", "swa.pt", *(f"epoch-{index:02d}.json" for index in range(48)),
                 *(f"checkpoint-{index:02d}.pt" for index in range(48)))
    if any((predecessor / name).exists() or (predecessor / name).is_symlink() for name in forbidden):
        raise RuntimeError("v1 predecessor has an illegal accepted-artifact topology")
    after_directory = os.lstat(predecessor)
    if ((after_directory.st_dev, after_directory.st_ino) != directory_identity
            or not stat.S_ISDIR(after_directory.st_mode) or stat.S_ISLNK(after_directory.st_mode)
            or frozenset(item.name for item in predecessor.iterdir()) != expected_leaves):
        raise RuntimeError("v1 predecessor directory changed during verification")
    return {
        "predecessor_root_relative": relative,
        "failure_relative": f"{relative}/failure.json",
        "failure_sha256": expected_body_sha256["failure.json"],
        "v1_accepted_checkpoints": 0,
        "v1_resume_forbidden": True,
    }


def verify_predecessor_failure(root: Path) -> dict[str, Any]:
    """Bind the one accepted failed predecessor and this successor work order."""
    lineage = _verify_predecessor_failure_at(root, PREDECESSOR_TRAIN_ROOT_RELATIVE, PREDECESSOR_BODY_SHA256)
    workorder = _canonical_regular_bytes(root.absolute() / SUCCESSOR_WORKORDER_RELATIVE)
    if _sha(workorder) != SUCCESSOR_WORKORDER_SHA256:
        raise RuntimeError("Phase-D successor work-order SHA drift")
    return {
        **lineage,
        "successor_workorder_relative": SUCCESSOR_WORKORDER_RELATIVE,
        "successor_workorder_sha256": SUCCESSOR_WORKORDER_SHA256,
    }


def _validate_predecessor_lineage(value: object, *, exact: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    expected = {
        "predecessor_root_relative", "failure_relative", "failure_sha256",
        "v1_accepted_checkpoints", "v1_resume_forbidden",
        "successor_workorder_relative", "successor_workorder_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError("successor predecessor-lineage schema drift")
    if (not isinstance(value.get("predecessor_root_relative"), str)
            or not isinstance(value.get("failure_relative"), str)
            or not _is_sha(value.get("failure_sha256"))
            or value.get("v1_accepted_checkpoints") != 0
            or value.get("v1_resume_forbidden") is not True
            or not isinstance(value.get("successor_workorder_relative"), str)
            or not _is_sha(value.get("successor_workorder_sha256"))):
        raise RuntimeError("successor predecessor-lineage semantic drift")
    if exact is not None and _safe_mapping_copy(value) != _safe_mapping_copy(exact):
        raise RuntimeError("successor predecessor-lineage exact binding drift")
    return value


def _public_predecessor_lineage() -> dict[str, Any]:
    """Static form used after production_identity already descriptor-verified it."""
    return {
        "predecessor_root_relative": PREDECESSOR_TRAIN_ROOT_RELATIVE,
        "failure_relative": PREDECESSOR_FAILURE_RELATIVE,
        "failure_sha256": PREDECESSOR_FAILURE_SHA256,
        "v1_accepted_checkpoints": 0,
        "v1_resume_forbidden": True,
        "successor_workorder_relative": SUCCESSOR_WORKORDER_RELATIVE,
        "successor_workorder_sha256": SUCCESSOR_WORKORDER_SHA256,
    }


def verify_phase_c_acceptance(root: Path) -> dict[str, object]:
    """Same-FD bind the independently accepted immutable Phase-C receipt."""
    path = root / PHASE_C_RECEIPT_RELATIVE
    body = _canonical_regular_bytes(path, expected_mode=0o444)
    if _sha(body) != PHASE_C_RECEIPT_SHA:
        raise RuntimeError("accepted Phase-C receipt SHA drift")
    sidecar = _canonical_regular_bytes(Path(str(path) + ".sha256"), expected_mode=0o444)
    if sidecar != f"{PHASE_C_RECEIPT_SHA}  {path.name}\n".encode():
        raise RuntimeError("accepted Phase-C sidecar drift")
    value = json.loads(body)
    source_smoke.validate_smoke_receipt(value)
    if value.get("final_closure") != value.get("launch_closure"):
        raise RuntimeError("accepted Phase-C launch/final closure drift")
    current = source_smoke.verify_stage0_and_phase_c_closures(root)
    accepted = value["final_closure"]
    if accepted.get("stage0") != current["stage0"] or accepted.get("phase_c") != current["phase_c"]:
        raise RuntimeError("current Stage-0/Phase-C closure differs from accepted smoke receipt")
    return {
        "path": PHASE_C_RECEIPT_RELATIVE, "body_sha256": PHASE_C_RECEIPT_SHA,
        "phase_c_closure_sha256": value["final_closure"]["phase_c"]["closure_sha256"],
        "stage0_closure_sha256": value["final_closure"]["stage0"]["closure_sha256"],
    }


def lr_for_step(step: int) -> float:
    """Bitwise live-arm parity for every public optimizer step."""
    if type(step) is not int or not 0 <= step < PUBLIC_SPEC.total_optimizer_steps:
        raise ValueError("step must be an integer in the exact 48-epoch schedule")
    # Deferred to preserve static/no-torch import behavior.
    from tfpd_lane.arm_common import lr_at_step
    return lr_at_step(step, PUBLIC_SPEC.epochs, PUBLIC_SPEC.steps_per_epoch)


def lr_for_spec(spec: RunSpec, step: int) -> float:
    """Use arm-common exactly for public; identical formula for injected mocks."""
    if type(step) is not int or not 0 <= step < spec.total_optimizer_steps:
        raise ValueError("step lies outside RunSpec")
    if spec == PUBLIC_SPEC:
        return lr_for_step(step)
    warm = min(2 * spec.steps_per_epoch, spec.total_optimizer_steps)
    if step < warm:
        return 1e-5 + (1e-4 - 1e-5) * (step / warm)
    span = spec.total_optimizer_steps - warm
    if span <= 0:
        return 1e-4
    progress = (step - warm) / span
    return 1e-6 + 0.5 * (1e-4 - 1e-6) * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class RunIdentity:
    """Receipt identity injected once at launch and recomputed at terminal."""

    phase_c_acceptance: Mapping[str, Any]
    source_authorities: Mapping[str, Any]
    closures: Mapping[str, Any]
    device: Mapping[str, Any]
    predecessor: Mapping[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "phase_c_acceptance": _safe_mapping_copy(self.phase_c_acceptance),
            "source_authorities": _safe_mapping_copy(self.source_authorities),
            "closures": _safe_mapping_copy(self.closures),
            "device": _safe_mapping_copy(self.device),
            "predecessor": _safe_mapping_copy(self.predecessor),
        }


def production_identity(root: Path) -> RunIdentity:
    """Descriptor-only identity binding; it opens no NWB, target, or CUDA object."""
    return RunIdentity(
        phase_c_acceptance=verify_phase_c_acceptance(root),
        source_authorities=source_smoke.verify_canonical_source_authorities(root),
        closures={**source_smoke.verify_stage0_and_phase_c_closures(root), "phase_d": phase_d_closure(root)},
        device=dict(source_smoke.FROZEN_DEVICE),
        predecessor=verify_predecessor_failure(root),
    )


def training_plan(root: Path) -> dict[str, object]:
    identity = production_identity(root)
    return {
        "cell": CELL, "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH", "authorization": "none",
        "execution_flags_required_together": ["--execute", "--i-have-48epoch-authorization"],
        **identity.payload(),
        "training": {
            **PUBLIC_SPEC.payload(), "optimizer": dict(OPTIMIZER),
            "schedule": {"authority": "tfpd_lane.arm_common.lr_at_step", "warmup_epochs": 2,
                         "warmup_start_lr": 1e-5, "warmup_end_lr": 1e-4, "cosine_final_lr": 1e-6},
            "validation_or_target": "forbidden", "swa": "arithmetic_checkpoint_state_mean_final_4",
        },
    }


def output_gate(root: Path) -> Path:
    """No output collision/alias may survive to source or CUDA imports."""
    if TRAIN_ROOT_RELATIVE == PREDECESSOR_TRAIN_ROOT_RELATIVE:
        raise RuntimeError("v1 predecessor cannot be a canonical v2 output alias")
    target = root / TRAIN_ROOT_RELATIVE
    predecessor = root / PREDECESSOR_TRAIN_ROOT_RELATIVE
    if target.absolute() == predecessor.absolute():
        raise RuntimeError("v1 predecessor cannot be a canonical v2 output alias")
    if target.exists() or target.is_symlink():
        raise RuntimeError("fresh canonical Phase-D output root required")
    ancestor = target.parent
    if ancestor.is_symlink() or not ancestor.is_dir():
        raise RuntimeError("canonical Phase-D output ancestor is invalid")
    return target


def _directory_identity(path: Path) -> tuple[int, int]:
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError("artifact root is not a canonical directory")
    return info.st_dev, info.st_ino


def _write_full(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        wrote = os.write(fd, view)
        if wrote <= 0:
            raise OSError("short artifact write")
        view = view[wrote:]


def _read_fd_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


@dataclass(frozen=True)
class ArtifactRoot:
    """Named-root capability for immutable body+sidecar publication.

    The capability stores the directory inode and verifies that the same name
    in the parent directory still resolves to that inode before and after each
    operation.  A caller cannot redirect publication with a replacement path.
    """

    directory: Path
    topology: tuple[str, ...]
    identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]

    def _assert_named_identity(self) -> None:
        if _directory_identity(self.directory) != self.identity:
            raise RuntimeError("artifact root path identity drift")
        pfd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            pinfo = os.fstat(pfd)
            if (pinfo.st_dev, pinfo.st_ino) != self.parent_identity:
                raise RuntimeError("artifact parent identity drift")
            named = os.stat(self.directory.name, dir_fd=pfd, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != self.identity or not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode):
                raise RuntimeError("artifact named-root identity drift")
        finally:
            os.close(pfd)

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or name not in self.topology or "/" in name or name in {"", ".", ".."}:
            raise RuntimeError("artifact name lies outside fixed topology")

    def has_name(self, name: str) -> bool:
        self._check_name(name)
        self._assert_named_identity()
        dfd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            try:
                os.stat(name, dir_fd=dfd, follow_symlinks=False)
            except FileNotFoundError:
                return False
            return True
        finally:
            os.close(dfd)

    def publish_bytes(self, name: str, body: bytes) -> str:
        self._check_name(name)
        if not isinstance(body, bytes):
            raise TypeError("artifact body must be bytes")
        self._assert_named_identity()
        dfd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        made: list[tuple[str, int, int]] = []
        try:
            opened = os.fstat(dfd)
            if (opened.st_dev, opened.st_ino) != self.identity:
                raise RuntimeError("artifact root changed between named check/open")
            digest = _sha(body)
            payloads = ((name, body), (name + ".sha256", f"{digest}  {name}\n".encode()))
            for leaf, payload in payloads:
                fd = os.open(leaf, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=dfd)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode):
                        raise RuntimeError("artifact O_EXCL did not create a regular file")
                    made.append((leaf, info.st_dev, info.st_ino))
                    _write_full(fd, payload)
                    os.fchmod(fd, 0o444)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(dfd)
            # Exact same-FD reload is a publication proof, not an optimistic
            # path-based read after close.
            for leaf, expected in payloads:
                fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
                try:
                    info = os.fstat(fd)
                    actual = _read_fd_all(fd)
                    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or actual != expected):
                        raise RuntimeError("artifact post-write body/sidecar drift")
                finally:
                    os.close(fd)
            self._assert_named_identity()
            pfd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(pfd)
            finally:
                os.close(pfd)
            return digest
        except BaseException:
            for leaf, device, inode in reversed(made):
                try:
                    current = os.stat(leaf, dir_fd=dfd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == (device, inode):
                        os.unlink(leaf, dir_fd=dfd)
                except OSError:
                    pass
            try:
                os.fsync(dfd)
            except OSError:
                pass
            raise
        finally:
            os.close(dfd)

    def reload_pair(self, name: str, expected_sha: str | None = None) -> bytes:
        self._check_name(name)
        self._assert_named_identity()
        dfd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            def read(leaf: str) -> bytes:
                fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                        raise RuntimeError("artifact mode/type drift")
                    return _read_fd_all(fd)
                finally:
                    os.close(fd)
            body = read(name)
            digest = _sha(body)
            if expected_sha is not None and digest != expected_sha:
                raise RuntimeError("artifact SHA drift")
            if read(name + ".sha256") != f"{digest}  {name}\n".encode():
                raise RuntimeError("artifact sidecar drift")
            self._assert_named_identity()
            return body
        finally:
            os.close(dfd)

    def publish_json(self, name: str, payload: Mapping[str, Any]) -> str:
        return self.publish_bytes(name, _json(payload))

    def reload_json(self, name: str, expected_sha: str | None = None) -> Mapping[str, Any]:
        try:
            value = json.loads(self.reload_pair(name, expected_sha))
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("artifact JSON decode drift") from error
        if not isinstance(value, Mapping):
            raise RuntimeError("artifact JSON root must be an object")
        return value


def reserve_artifact_root(root: Path, relative: str, topology: tuple[str, ...]) -> ArtifactRoot:
    """Reserve an exact named output directory before source/CUDA imports."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("artifact relative path must be safe")
    if not isinstance(topology, tuple) or not topology or len(set(topology)) != len(topology):
        raise ValueError("artifact topology must be a nonempty duplicate-free tuple")
    if any(not isinstance(name, str) or not name or "/" in name for name in topology):
        raise ValueError("artifact topology name drift")
    target = root / relative
    if target.exists() or target.is_symlink():
        raise RuntimeError("fresh canonical Phase-D output root required")
    parent = target.parent
    parent_identity = _directory_identity(parent)
    pfd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened_parent = os.fstat(pfd)
        if (opened_parent.st_dev, opened_parent.st_ino) != parent_identity:
            raise RuntimeError("artifact parent changed between lstat/open")
        os.mkdir(target.name, 0o755, dir_fd=pfd)
        os.fsync(pfd)
        info = os.stat(target.name, dir_fd=pfd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError("reserved output root is invalid")
        identity = (info.st_dev, info.st_ino)
    except FileExistsError as error:
        raise RuntimeError("canonical Phase-D output collision") from error
    finally:
        os.close(pfd)
    return ArtifactRoot(target, topology, identity, parent, parent_identity)


def reserve_output_root(root: Path) -> ArtifactRoot:
    """Public fixed-topology reservation compatibility wrapper."""
    output_gate(root)
    return reserve_artifact_root(root, TRAIN_ROOT_RELATIVE, PUBLIC_SPEC.topology)


def _validate_identity(identity: RunIdentity, *, public: bool) -> None:
    if not isinstance(identity, RunIdentity):
        raise RuntimeError("run identity type drift")
    data = identity.payload()
    if set(data) != {"phase_c_acceptance", "source_authorities", "closures", "device", "predecessor"}:
        raise RuntimeError("run identity key drift")
    if not all(isinstance(data[key], Mapping) for key in data):
        raise RuntimeError("run identity mapping drift")
    if public:
        if data["phase_c_acceptance"].get("body_sha256") != PHASE_C_RECEIPT_SHA:
            raise RuntimeError("public Phase-C acceptance drift")
        if data["device"] != source_smoke.FROZEN_DEVICE:
            raise RuntimeError("public frozen device drift")
        if data["source_authorities"].get("manifest_sha256") != source_smoke.MANIFEST_SHA:
            raise RuntimeError("public source authority drift")
        _validate_predecessor_lineage(data["predecessor"], exact=_public_predecessor_lineage())
        closures = data["closures"]
        if set(closures) != {"stage0", "phase_c", "phase_d"}:
            raise RuntimeError("public closure map drift")


@dataclass
class LifecycleFlags:
    source_opened: bool = False
    gpu_initialized: bool = False
    optimizer_steps_completed: int = 0
    target_or_formal_opened: bool = False
    terminal_published: bool = False
    stage: str = "identity"
    predecessor: Mapping[str, Any] | None = None
    identity_closure: Mapping[str, Any] | None = None
    launch_closure: Mapping[str, Any] | None = None

    def boundary_payload(self) -> dict[str, bool]:
        return {
            "source_only": True,
            "target_or_formal_opened": self.target_or_formal_opened,
            "scientific_result": False,
            "score": False,
            "capture_diagnostics": False,
        }

    def predecessor_payload(self) -> dict[str, Any]:
        if self.predecessor is None:
            raise RuntimeError("lifecycle predecessor lineage is not bound")
        _validate_predecessor_lineage(self.predecessor)
        return _safe_mapping_copy(self.predecessor)

    def launch_closure_payload(self) -> dict[str, Any]:
        if self.launch_closure is None:
            raise RuntimeError("lifecycle launch closure is not bound")
        if not isinstance(self.launch_closure, Mapping):
            raise RuntimeError("lifecycle launch closure is malformed")
        return _safe_mapping_copy(self.launch_closure)

    def identity_closure_payload(self) -> dict[str, Any]:
        if self.identity_closure is None or not isinstance(self.identity_closure, Mapping):
            raise RuntimeError("lifecycle identity closure is not bound")
        return _safe_mapping_copy(self.identity_closure)


@dataclass(frozen=True)
class StepOutcome:
    loss: float
    lr_observed: float
    dropout_p: float
    kept: int
    dropped: int
    all_zero_examples: int
    population_examples: int
    max_gain: float
    # Full-model/Adam observability is intentionally an epoch-boundary proof.
    # Leaving these fields absent on ordinary steps prevents an inexpensive
    # placeholder from looking like a verified full-state observation.
    epoch_boundary_proof: bool
    critical_gradients: Mapping[str, bool] | None
    finite_model: bool | None
    finite_optimizer: bool | None
    model_state_digest: str | None
    optimizer_state_digest: str | None


@dataclass(frozen=True)
class CheckpointPayload:
    body: bytes
    model_state_digest: str


@dataclass(frozen=True)
class SWAPayload:
    body: bytes
    state_digest: str
    evaluation_proof: Mapping[str, Any]


class TrainingBackend(Protocol):
    """Injected backend: source/CUDA work stays behind this interface."""

    def prepare(self, spec: RunSpec, identity: RunIdentity, flags: LifecycleFlags) -> Any: ...
    def begin_epoch(self, runtime: Any, epoch: int) -> None: ...
    def train_step(self, runtime: Any, lr: float, global_step: int, *, require_epoch_proof: bool) -> StepOutcome: ...
    def after_optimizer_step(self, runtime: Any, global_step: int, flags: LifecycleFlags) -> None: ...
    def resources(self, runtime: Any) -> Mapping[str, int]: ...
    def make_checkpoint(self, runtime: Any, epoch: int, global_step: int, binding: Mapping[str, Any]) -> CheckpointPayload: ...
    def validate_checkpoint(self, body: bytes, epoch: int, global_step: int, spec: RunSpec,
                            *, expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]: ...
    def build_swa(self, runtime: Any, checkpoints: Mapping[int, bytes], spec: RunSpec,
                  *, expected_binding: Mapping[str, Any]) -> SWAPayload: ...
    def validate_swa(self, body: bytes, spec: RunSpec,
                     *, expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]: ...
    def close(self, runtime: Any | None) -> None: ...


def requires_epoch_proof(spec: RunSpec, global_step: int) -> bool:
    """Return whether the *next* zero-based step closes an epoch.

    This predicate is intentionally public to tests: it proves the public
    throughput probe (steps 1..100) cannot accidentally request full-state
    hashes because its first epoch ends only at step 33,925.
    """
    if type(global_step) is not int or not 0 <= global_step < spec.total_optimizer_steps:
        raise ValueError("global_step lies outside RunSpec")
    return (global_step + 1) % spec.steps_per_epoch == 0


def _validate_step_outcome(value: StepOutcome, expected_lr: float, *, require_epoch_proof: bool) -> None:
    if not isinstance(value, StepOutcome):
        raise RuntimeError("backend did not return StepOutcome")
    if not _finite_number(value.loss, nonnegative=True) or value.lr_observed != expected_lr:
        raise RuntimeError("step loss/LR drift")
    if not _finite_number(value.dropout_p, nonnegative=True) or not 0.0 <= value.dropout_p <= 1.0:
        raise RuntimeError("step dropout p drift")
    if any(type(item) is not int or item < 0 for item in (value.kept, value.dropped, value.all_zero_examples, value.population_examples)):
        raise RuntimeError("step mask count drift")
    if value.population_examples <= 0 or value.all_zero_examples > value.population_examples or value.kept + value.dropped <= 0:
        raise RuntimeError("step mask population drift")
    if not _finite_number(value.max_gain, nonnegative=True):
        raise RuntimeError("step mask gain drift")
    if value.epoch_boundary_proof is not require_epoch_proof:
        raise RuntimeError("step epoch-boundary proof request drift")
    if require_epoch_proof:
        _strict_bool_map(value.critical_gradients, _CRITICAL_GRADIENT_GROUPS, "critical gradient")
        if value.finite_model is not True or value.finite_optimizer is not True:
            raise RuntimeError("finite model/optimizer proof drift")
        if not _is_sha(value.model_state_digest) or not _is_sha(value.optimizer_state_digest):
            raise RuntimeError("step deterministic digest drift")
    elif any(item is not None for item in (value.critical_gradients, value.finite_model, value.finite_optimizer,
                                           value.model_state_digest, value.optimizer_state_digest)):
        raise RuntimeError("ordinary step fabricated an epoch-boundary proof")


def _quantile(values: list[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise RuntimeError("empty/invalid quantile")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = int(math.floor(index)), int(math.ceil(index))
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _resource_payload(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_RESOURCE_KEYS):
        raise RuntimeError("resource map key drift")
    result: dict[str, int] = {}
    for key in _RESOURCE_KEYS:
        item = value[key]
        if type(item) is not int or item < 0:
            raise RuntimeError("resource map value drift")
        result[key] = item
    return result


def _epoch_payload(spec: RunSpec, epoch: int, global_step: int, outcomes: Sequence[StepOutcome], elapsed: float,
                   resources: Mapping[str, Any], flags: LifecycleFlags) -> dict[str, Any]:
    if len(outcomes) != spec.steps_per_epoch or not _finite_number(elapsed, positive=True):
        raise RuntimeError("epoch outcome/time accounting drift")
    p = [item.dropout_p for item in outcomes]
    loss = [item.loss for item in outcomes]
    kept, dropped = sum(item.kept for item in outcomes), sum(item.dropped for item in outcomes)
    population = sum(item.population_examples for item in outcomes)
    all_zero = sum(item.all_zero_examples for item in outcomes)
    if kept + dropped <= 0 or population <= 0:
        raise RuntimeError("epoch aggregate mask denominator drift")
    boundary = outcomes[-1]
    if boundary.epoch_boundary_proof is not True or any(item is None for item in (
            boundary.critical_gradients, boundary.finite_model, boundary.finite_optimizer,
            boundary.model_state_digest, boundary.optimizer_state_digest)):
        raise RuntimeError("epoch lacks a final full-state proof")
    if any(item.epoch_boundary_proof for item in outcomes[:-1]):
        raise RuntimeError("epoch has a nonfinal full-state proof")
    return {
        "schema": "tfsr_b3st4_ddrop_epoch_v2", "cell": CELL, "epoch": epoch,
        "steps": spec.steps_per_epoch, "cumulative_steps": global_step,
        "loss": {"mean": sum(loss) / len(loss), "min": min(loss), "max": max(loss)},
        "lr": {"first": outcomes[0].lr_observed, "last": outcomes[-1].lr_observed,
               "expected_first": lr_for_spec(spec, global_step - spec.steps_per_epoch),
               "expected_last": lr_for_spec(spec, global_step - 1)},
        "dropout": {
            "p_min": min(p), "p_max": max(p), "p_mean": sum(p) / len(p),
            "p_q25": _quantile(p, 0.25), "p_q50": _quantile(p, 0.50), "p_q75": _quantile(p, 0.75),
            "kept": kept, "dropped": dropped, "kept_fraction": kept / (kept + dropped),
            "dropped_fraction": dropped / (kept + dropped), "all_zero_examples": all_zero,
            "population_examples": population, "all_zero_fraction": all_zero / population,
            "max_gain": max(item.max_gain for item in outcomes),
        },
        "critical_gradients": dict(boundary.critical_gradients),
        "finite": {"model": boundary.finite_model, "optimizer": boundary.finite_optimizer},
        "state": {"model_state_digest": boundary.model_state_digest,
                  "optimizer_state_digest": boundary.optimizer_state_digest},
        "predecessor": flags.predecessor_payload(),
        "launch_closure": flags.launch_closure_payload(),
        "resources": _resource_payload(resources),
        "boundaries": flags.boundary_payload(),
        "progress": {"epoch": epoch, "completed_epochs": epoch + 1, "epochs": spec.epochs,
                     "global_step": global_step, "total_optimizer_steps": spec.total_optimizer_steps},
        "elapsed_seconds": elapsed, "throughput_steps_per_second": spec.steps_per_epoch / elapsed,
    }


def validate_epoch_receipt(value: Mapping[str, Any], epoch: int, cumulative: int,
                           spec: RunSpec = PUBLIC_SPEC, identity: RunIdentity | None = None) -> None:
    expected = {"schema", "cell", "epoch", "steps", "cumulative_steps", "loss", "lr", "dropout",
                "critical_gradients", "finite", "state", "predecessor", "launch_closure", "resources", "boundaries", "progress",
                "elapsed_seconds", "throughput_steps_per_second"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_b3st4_ddrop_epoch_v2":
        raise RuntimeError("epoch receipt schema drift")
    if value.get("cell") != CELL or value.get("epoch") != epoch or value.get("steps") != spec.steps_per_epoch or value.get("cumulative_steps") != cumulative:
        raise RuntimeError("epoch receipt accounting drift")
    loss = value.get("loss")
    if not isinstance(loss, Mapping) or set(loss) != {"mean", "min", "max"} or not all(_finite_number(loss[key], nonnegative=True) for key in loss):
        raise RuntimeError("epoch loss payload drift")
    if not loss["min"] <= loss["mean"] <= loss["max"]:
        raise RuntimeError("epoch loss ordering drift")
    lr = value.get("lr")
    if not isinstance(lr, Mapping) or set(lr) != {"first", "last", "expected_first", "expected_last"}:
        raise RuntimeError("epoch LR payload drift")
    first_step, last_step = cumulative - spec.steps_per_epoch, cumulative - 1
    if lr["first"] != lr["expected_first"] or lr["last"] != lr["expected_last"] or lr["first"] != lr_for_spec(spec, first_step) or lr["last"] != lr_for_spec(spec, last_step):
        raise RuntimeError("epoch expected LR drift")
    dropout = value.get("dropout")
    expected_dropout = {"p_min", "p_max", "p_mean", "p_q25", "p_q50", "p_q75", "kept", "dropped",
                        "kept_fraction", "dropped_fraction", "all_zero_examples", "population_examples",
                        "all_zero_fraction", "max_gain"}
    if not isinstance(dropout, Mapping) or set(dropout) != expected_dropout:
        raise RuntimeError("epoch dropout schema drift")
    if not all(_finite_number(dropout[key], nonnegative=True) for key in ("p_min", "p_max", "p_mean", "p_q25", "p_q50", "p_q75", "kept_fraction", "dropped_fraction", "all_zero_fraction", "max_gain")):
        raise RuntimeError("epoch dropout numeric drift")
    if any(type(dropout[key]) is not int or dropout[key] < 0 for key in ("kept", "dropped", "all_zero_examples", "population_examples")):
        raise RuntimeError("epoch dropout count drift")
    if not 0 <= dropout["p_min"] <= dropout["p_q25"] <= dropout["p_q50"] <= dropout["p_q75"] <= dropout["p_max"] <= 1:
        raise RuntimeError("epoch p quantile drift")
    if (dropout["kept"] + dropout["dropped"] <= 0 or dropout["population_examples"] != spec.steps_per_epoch * spec.batch_size
            or dropout["all_zero_examples"] > dropout["population_examples"]
            or not math.isclose(dropout["kept_fraction"] + dropout["dropped_fraction"], 1.0, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(dropout["all_zero_fraction"], dropout["all_zero_examples"] / dropout["population_examples"], rel_tol=0.0, abs_tol=1e-12)):
        raise RuntimeError("epoch kept/drop fraction drift")
    _strict_bool_map(value.get("critical_gradients"), _CRITICAL_GRADIENT_GROUPS, "epoch critical gradient")
    if value.get("finite") != {"model": True, "optimizer": True}:
        raise RuntimeError("epoch finite proof drift")
    state = value.get("state")
    if not isinstance(state, Mapping) or set(state) != {"model_state_digest", "optimizer_state_digest"} or not all(_is_sha(state[key]) for key in state):
        raise RuntimeError("epoch state digest drift")
    _validate_predecessor_lineage(value.get("predecessor"), exact=(identity.predecessor if identity is not None else None))
    if not isinstance(value.get("launch_closure"), Mapping):
        raise RuntimeError("epoch launch closure drift")
    if identity is not None and value.get("launch_closure") != _safe_mapping_copy(identity.closures):
        raise RuntimeError("epoch exact launch closure drift")
    _resource_payload(value.get("resources"))
    if value.get("boundaries") != {"source_only": True, "target_or_formal_opened": False, "scientific_result": False, "score": False, "capture_diagnostics": False}:
        raise RuntimeError("epoch source-only boundary drift")
    expected_progress = {"epoch": epoch, "completed_epochs": epoch + 1, "epochs": spec.epochs,
                         "global_step": cumulative, "total_optimizer_steps": spec.total_optimizer_steps}
    if value.get("progress") != expected_progress or not _finite_number(value.get("elapsed_seconds"), positive=True) or not _finite_number(value.get("throughput_steps_per_second"), positive=True):
        raise RuntimeError("epoch progress/resource timing drift")


def _throughput_payload(spec: RunSpec, elapsed: float, resources: Mapping[str, Any], flags: LifecycleFlags) -> dict[str, Any]:
    if not _finite_number(elapsed, positive=True):
        raise RuntimeError("throughput elapsed drift")
    rate = spec.throughput_probe_steps / elapsed
    return {
        "schema": "tfsr_b3st4_ddrop_throughput_v2", "cell": CELL,
        "threshold_steps": spec.throughput_probe_steps, "elapsed_seconds": elapsed,
        "steps_per_second": rate,
        "eta_seconds_estimate_only": (spec.total_optimizer_steps - spec.throughput_probe_steps) / rate,
        "predecessor": flags.predecessor_payload(), "launch_closure": flags.launch_closure_payload(),
        "resources": _resource_payload(resources), "boundaries": flags.boundary_payload(),
    }


def validate_throughput_receipt(value: Mapping[str, Any], spec: RunSpec = PUBLIC_SPEC,
                                identity: RunIdentity | None = None) -> None:
    expected = {"schema", "cell", "threshold_steps", "elapsed_seconds", "steps_per_second", "eta_seconds_estimate_only", "predecessor", "launch_closure", "resources", "boundaries"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_b3st4_ddrop_throughput_v2" or value.get("cell") != CELL or value.get("threshold_steps") != spec.throughput_probe_steps:
        raise RuntimeError("throughput receipt schema drift")
    if not all(_finite_number(value[key], nonnegative=True) for key in ("elapsed_seconds", "steps_per_second", "eta_seconds_estimate_only")) or value["elapsed_seconds"] <= 0 or value["steps_per_second"] <= 0:
        raise RuntimeError("throughput receipt numeric drift")
    _resource_payload(value.get("resources"))
    _validate_predecessor_lineage(value.get("predecessor"), exact=(identity.predecessor if identity is not None else None))
    if not isinstance(value.get("launch_closure"), Mapping):
        raise RuntimeError("throughput launch closure drift")
    if identity is not None and value.get("launch_closure") != _safe_mapping_copy(identity.closures):
        raise RuntimeError("throughput exact launch closure drift")
    if value.get("boundaries") != {"source_only": True, "target_or_formal_opened": False, "scientific_result": False, "score": False, "capture_diagnostics": False}:
        raise RuntimeError("throughput source-only boundary drift")


def _attempt_payload(spec: RunSpec, identity: RunIdentity) -> dict[str, Any]:
    return {
        "schema": "tfsr_b3st4_ddrop_train_attempt_v2", "cell": CELL, "run_spec": spec.payload(),
        "phase_c_acceptance": _safe_mapping_copy(identity.phase_c_acceptance),
        "closures": _safe_mapping_copy(identity.closures), "predecessor": _safe_mapping_copy(identity.predecessor),
        "topology": list(spec.topology),
        "source_opened": False, "gpu_initialized": False, "target_or_formal_opened": False,
    }


def validate_attempt_receipt(value: Mapping[str, Any], spec: RunSpec = PUBLIC_SPEC, identity: RunIdentity | None = None) -> None:
    expected = {"schema", "cell", "run_spec", "phase_c_acceptance", "closures", "predecessor", "topology", "source_opened", "gpu_initialized", "target_or_formal_opened"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_b3st4_ddrop_train_attempt_v2" or value.get("cell") != CELL:
        raise RuntimeError("attempt receipt schema drift")
    if value.get("run_spec") != spec.payload() or value.get("topology") != list(spec.topology) or value.get("source_opened") is not False or value.get("gpu_initialized") is not False or value.get("target_or_formal_opened") is not False:
        raise RuntimeError("attempt receipt state drift")
    _validate_predecessor_lineage(value.get("predecessor"), exact=(identity.predecessor if identity is not None else None))
    if identity is not None and (value.get("phase_c_acceptance") != _safe_mapping_copy(identity.phase_c_acceptance) or value.get("closures") != _safe_mapping_copy(identity.closures)):
        raise RuntimeError("attempt identity drift")
    if spec == PUBLIC_SPEC and value.get("phase_c_acceptance", {}).get("body_sha256") != PHASE_C_RECEIPT_SHA:
        raise RuntimeError("attempt public Phase-C drift")


def _launch_payload(spec: RunSpec, identity: RunIdentity) -> dict[str, Any]:
    sample_steps = sorted({0, min(2 * spec.steps_per_epoch - 1, spec.total_optimizer_steps - 1),
                           min(2 * spec.steps_per_epoch, spec.total_optimizer_steps - 1), spec.total_optimizer_steps - 1})
    return {
        "schema": "tfsr_b3st4_ddrop_train_launch_v2", "cell": CELL, "run_spec": spec.payload(),
        "phase_c_acceptance": _safe_mapping_copy(identity.phase_c_acceptance),
        "source_authorities": _safe_mapping_copy(identity.source_authorities),
        "closures": _safe_mapping_copy(identity.closures), "device": _safe_mapping_copy(identity.device),
        "predecessor": _safe_mapping_copy(identity.predecessor),
        "optimizer": dict(OPTIMIZER),
        "schedule_formula": "tfpd_lane.arm_common.lr_at_step (public); mathematically identical injected-spec schedule (mock only)",
        "sampled_lr": {str(step): lr_for_spec(spec, step) for step in sample_steps},
        "boundaries": {"source_only": True, "target_or_formal_opened": False, "scientific_result": False, "score": False, "capture_diagnostics": False},
    }


def validate_launch_receipt(value: Mapping[str, Any], spec: RunSpec = PUBLIC_SPEC, identity: RunIdentity | None = None) -> None:
    expected = {"schema", "cell", "run_spec", "phase_c_acceptance", "source_authorities", "closures", "device", "predecessor", "optimizer", "schedule_formula", "sampled_lr", "boundaries"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_b3st4_ddrop_train_launch_v2" or value.get("cell") != CELL:
        raise RuntimeError("launch receipt schema drift")
    if value.get("run_spec") != spec.payload() or value.get("optimizer") != OPTIMIZER or not isinstance(value.get("schedule_formula"), str) or not value["schedule_formula"]:
        raise RuntimeError("launch receipt run/optimizer drift")
    if value.get("boundaries") != {"source_only": True, "target_or_formal_opened": False, "scientific_result": False, "score": False, "capture_diagnostics": False}:
        raise RuntimeError("launch source-only boundary drift")
    if identity is not None:
        payload = identity.payload()
        if any(value.get(key) != payload[key] for key in ("phase_c_acceptance", "source_authorities", "closures", "device", "predecessor")):
            raise RuntimeError("launch identity drift")
    _validate_predecessor_lineage(value.get("predecessor"), exact=(identity.predecessor if identity is not None else None))
    if spec == PUBLIC_SPEC:
        if value.get("phase_c_acceptance", {}).get("body_sha256") != PHASE_C_RECEIPT_SHA or value.get("device") != source_smoke.FROZEN_DEVICE or value.get("source_authorities", {}).get("manifest_sha256") != source_smoke.MANIFEST_SHA:
            raise RuntimeError("launch public authority/device drift")
    if not isinstance(value.get("sampled_lr"), Mapping) or not value["sampled_lr"] or any(not _finite_number(item, positive=True) for item in value["sampled_lr"].values()):
        raise RuntimeError("launch sampled LR drift")


def validate_failure_receipt(value: Mapping[str, Any], identity: RunIdentity | None = None) -> None:
    expected = {"schema", "cell", "stage", "source_opened", "gpu_initialized", "optimizer_steps_completed", "target_or_formal_opened", "terminal_published", "predecessor", "closures", "traceback_sha256"}
    allowed_stages = {"identity", "attempt", "backend_prepare", "epoch", "optimizer_step", "checkpoint", "swa", "terminal"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_b3st4_ddrop_train_failure_v2" or value.get("cell") != CELL:
        raise RuntimeError("failure receipt schema drift")
    # A failure receipt must be honest even if a buggy backend attempted a
    # forbidden boundary.  It is never a source-only scientific receipt; the
    # terminal/epoch/throughput validators remain strict ``False``.
    if value.get("stage") not in allowed_stages or type(value.get("source_opened")) is not bool or type(value.get("gpu_initialized")) is not bool or type(value.get("target_or_formal_opened")) is not bool or value.get("terminal_published") is not False:
        raise RuntimeError("failure receipt boundary drift")
    if type(value.get("optimizer_steps_completed")) is not int or value["optimizer_steps_completed"] < 0 or not _is_sha(value.get("traceback_sha256")):
        raise RuntimeError("failure receipt accounting drift")
    _validate_predecessor_lineage(value.get("predecessor"), exact=(identity.predecessor if identity is not None else None))
    if not isinstance(value.get("closures"), Mapping):
        raise RuntimeError("failure receipt closure drift")
    if identity is not None and value.get("closures") != _safe_mapping_copy(identity.closures):
        raise RuntimeError("failure receipt exact closure drift")


def _validate_swa_evaluation_proof(proof: object, spec: RunSpec) -> Mapping[str, Any]:
    expected = {"checkpoint_epochs", "checkpoint_model_state_digests", "fresh_strict_load", "eval_mode",
                "capture_diagnostics", "repeat_bitwise_equal", "state_unchanged", "eval_no_mask",
                "prediction_shape", "prediction_sha256", "state_digest_before_eval", "state_digest_after_eval",
                "boundaries"}
    if not isinstance(proof, Mapping) or set(proof) != expected or proof.get("checkpoint_epochs") != list(spec.checkpoint_epochs):
        raise RuntimeError("SWA evaluation proof schema drift")
    if (proof.get("fresh_strict_load") is not True or proof.get("eval_mode") is not True
            or proof.get("capture_diagnostics") is not False or proof.get("repeat_bitwise_equal") is not True
            or proof.get("state_unchanged") is not True or proof.get("eval_no_mask") is not True
            or not _is_sha(proof.get("prediction_sha256")) or not _is_sha(proof.get("state_digest_before_eval"))
            or not _is_sha(proof.get("state_digest_after_eval"))
            or proof["state_digest_before_eval"] != proof["state_digest_after_eval"]):
        raise RuntimeError("SWA evaluation proof semantic drift")
    if proof.get("boundaries") != {"source_only": True, "target_or_formal_opened": False, "scientific_result": False, "score": False}:
        raise RuntimeError("SWA source-only boundary drift")
    shape = proof.get("prediction_shape")
    if not isinstance(shape, list) or len(shape) != 3 or type(shape[0]) is not int or shape[0] <= 0 or shape[1:] != [50, 2]:
        raise RuntimeError("SWA prediction shape drift")
    expected_epochs = {str(epoch) for epoch in spec.checkpoint_epochs}
    digest_map = proof.get("checkpoint_model_state_digests")
    if not isinstance(digest_map, Mapping) or set(digest_map) != expected_epochs or not all(_is_sha(item) for item in digest_map.values()):
        raise RuntimeError("SWA checkpoint digest proof drift")
    return proof


def _terminal_payload(spec: RunSpec, identity: RunIdentity, final_identity: RunIdentity, *, global_step: int,
                      attempt_sha: str, launch_sha: str, throughput_sha: str, epoch_sha: list[str],
                      checkpoint_sha: Mapping[str, str], swa_sha: str, swa_state_digest: str,
                      swa_proof: Mapping[str, Any], artifact_validation: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": "tfsr_b3st4_ddrop_train_terminal_v2", "status": "TRAINING_COMPLETE", "cell": CELL,
        "run_spec": spec.payload(), "phase_c_acceptance": _safe_mapping_copy(identity.phase_c_acceptance),
        "source_authorities": _safe_mapping_copy(identity.source_authorities),
        "launch_closure": _safe_mapping_copy(identity.closures), "final_closure": _safe_mapping_copy(final_identity.closures),
        "device": _safe_mapping_copy(identity.device), "predecessor": _safe_mapping_copy(identity.predecessor),
        "epochs": spec.epochs, "steps_per_epoch": spec.steps_per_epoch,
        "total_optimizer_steps": global_step, "attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
        "throughput_sha256": throughput_sha, "epoch_receipt_sha256": epoch_sha,
        "checkpoint_sha256": dict(checkpoint_sha), "swa_sha256": swa_sha,
        "swa_state_digest": swa_state_digest, "swa_evaluation_proof": _safe_mapping_copy(swa_proof),
        "artifact_validation": dict(artifact_validation),
        "boundaries": {"source_only": True, "target_or_formal_opened": False, "scientific_result": False, "score": False, "capture_diagnostics": False},
    }


def validate_terminal_receipt(value: Mapping[str, Any], spec: RunSpec = PUBLIC_SPEC, identity: RunIdentity | None = None) -> None:
    expected = {"schema", "status", "cell", "run_spec", "phase_c_acceptance", "source_authorities", "launch_closure", "final_closure", "device", "predecessor", "epochs", "steps_per_epoch", "total_optimizer_steps", "attempt_sha256", "launch_sha256", "throughput_sha256", "epoch_receipt_sha256", "checkpoint_sha256", "swa_sha256", "swa_state_digest", "swa_evaluation_proof", "artifact_validation", "boundaries"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_b3st4_ddrop_train_terminal_v2" or value.get("status") != "TRAINING_COMPLETE" or value.get("cell") != CELL:
        raise RuntimeError("terminal receipt schema drift")
    if value.get("run_spec") != spec.payload() or value.get("epochs") != spec.epochs or value.get("steps_per_epoch") != spec.steps_per_epoch or value.get("total_optimizer_steps") != spec.total_optimizer_steps:
        raise RuntimeError("terminal accounting drift")
    if value.get("launch_closure") != value.get("final_closure"):
        raise RuntimeError("terminal launch/final closure drift")
    if value.get("boundaries") != {"source_only": True, "target_or_formal_opened": False, "scientific_result": False, "score": False, "capture_diagnostics": False}:
        raise RuntimeError("terminal source-only boundary drift")
    if identity is not None:
        payload = identity.payload()
        if any(value.get(key) != payload[key] for key in ("phase_c_acceptance", "source_authorities", "device", "predecessor")) or value.get("launch_closure") != payload["closures"]:
            raise RuntimeError("terminal identity drift")
    _validate_predecessor_lineage(value.get("predecessor"), exact=(identity.predecessor if identity is not None else None))
    direct_hashes = ("attempt_sha256", "launch_sha256", "throughput_sha256", "swa_sha256", "swa_state_digest")
    if not all(_is_sha(value.get(key)) for key in direct_hashes):
        raise RuntimeError("terminal direct SHA drift")
    epochs = value.get("epoch_receipt_sha256")
    if not isinstance(epochs, list) or len(epochs) != spec.epochs or not all(_is_sha(item) for item in epochs):
        raise RuntimeError("terminal epoch SHA list drift")
    checkpoints = value.get("checkpoint_sha256")
    expected_checkpoint_keys = {str(epoch) for epoch in spec.checkpoint_epochs}
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != expected_checkpoint_keys or not all(_is_sha(item) for item in checkpoints.values()):
        raise RuntimeError("terminal checkpoint SHA map drift")
    try:
        _validate_swa_evaluation_proof(value.get("swa_evaluation_proof"), spec)
    except RuntimeError as error:
        raise RuntimeError("terminal SWA evaluation proof drift") from error
    artifacts = value.get("artifact_validation")
    expected_artifacts = {"attempt.json", "launch.json", f"throughput{spec.throughput_probe_steps}.json", "swa.pt", *(f"epoch-{epoch:02d}.json" for epoch in range(spec.epochs)), *(f"checkpoint-{epoch:02d}.pt" for epoch in spec.checkpoint_epochs)}
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts or not all(_is_sha(item) for item in artifacts.values()):
        raise RuntimeError("terminal artifact validation map drift")


def _failure_payload(flags: LifecycleFlags) -> dict[str, Any]:
    return {
        "schema": "tfsr_b3st4_ddrop_train_failure_v2", "cell": CELL, "stage": flags.stage,
        "source_opened": flags.source_opened, "gpu_initialized": flags.gpu_initialized,
        "optimizer_steps_completed": flags.optimizer_steps_completed,
        "target_or_formal_opened": flags.target_or_formal_opened,
        "terminal_published": flags.terminal_published,
        "predecessor": flags.predecessor_payload(),
        "closures": flags.identity_closure_payload(),
        "traceback_sha256": _sha(traceback.format_exc().encode()),
    }


def _publish_failure(artifact: ArtifactRoot, flags: LifecycleFlags, identity: RunIdentity | None = None) -> None:
    if flags.terminal_published or artifact.has_name("terminal.json"):
        raise RuntimeError("failure publication is forbidden after terminal")
    if artifact.has_name("failure.json"):
        return
    receipt = _failure_payload(flags)
    validate_failure_receipt(receipt, identity)
    digest = artifact.publish_json("failure.json", receipt)
    validate_failure_receipt(artifact.reload_json("failure.json", digest), identity)


def _validate_all_preterminal_artifacts(artifact: ArtifactRoot, spec: RunSpec, identity: RunIdentity,
                                        backend: TrainingBackend, hashes: Mapping[str, str]) -> dict[str, str]:
    """Reload every body+sidecar and its schema before terminal publication."""
    expected_names = {"attempt.json", "launch.json", f"throughput{spec.throughput_probe_steps}.json", "swa.pt",
                      *(f"epoch-{epoch:02d}.json" for epoch in range(spec.epochs)),
                      *(f"checkpoint-{epoch:02d}.pt" for epoch in spec.checkpoint_epochs)}
    if set(hashes) != expected_names:
        raise RuntimeError("preterminal artifact hash set drift")
    attempt = artifact.reload_json("attempt.json", hashes["attempt.json"])
    validate_attempt_receipt(attempt, spec, identity)
    launch = artifact.reload_json("launch.json", hashes["launch.json"])
    validate_launch_receipt(launch, spec, identity)
    throughput_name = f"throughput{spec.throughput_probe_steps}.json"
    throughput = artifact.reload_json(throughput_name, hashes[throughput_name])
    validate_throughput_receipt(throughput, spec, identity)
    for epoch in range(spec.epochs):
        name = f"epoch-{epoch:02d}.json"
        value = artifact.reload_json(name, hashes[name])
        validate_epoch_receipt(value, epoch, (epoch + 1) * spec.steps_per_epoch, spec, identity)
    expected_binding = {
        "cell": CELL, "run_spec": spec.payload(), "launch_sha256": hashes["launch.json"],
        "launch_closure": _safe_mapping_copy(identity.closures),
        "predecessor": _safe_mapping_copy(identity.predecessor),
    }
    for epoch in spec.checkpoint_epochs:
        name = f"checkpoint-{epoch:02d}.pt"
        backend.validate_checkpoint(artifact.reload_pair(name, hashes[name]), epoch, (epoch + 1) * spec.steps_per_epoch,
                                    spec, expected_binding=expected_binding)
    backend.validate_swa(artifact.reload_pair("swa.pt", hashes["swa.pt"]), spec, expected_binding=expected_binding)
    if artifact.has_name("failure.json") or artifact.has_name("terminal.json"):
        raise RuntimeError("terminal finalizer sees an illegal prior terminal/failure artifact")
    return dict(hashes)


def run_lifecycle(*, spec: RunSpec, backend: TrainingBackend, artifact: ArtifactRoot,
                  identity_factory: Callable[[], RunIdentity]) -> Mapping[str, Any]:
    """Run the injected lifecycle and leave either terminal or failure evidence.

    This is intentionally the only path which emits lifecycle artifacts.  The
    public caller injects the frozen 48ep spec and physical backend; tests
    inject a tiny immutable spec plus a no-data mock backend.
    """
    flags = LifecycleFlags()
    runtime: Any | None = None
    identity: RunIdentity | None = None
    try:
        flags.stage = "identity"
        identity = identity_factory()
        _validate_identity(identity, public=(spec == PUBLIC_SPEC))
        flags.predecessor = _safe_mapping_copy(identity.predecessor)
        flags.identity_closure = _safe_mapping_copy(identity.closures)
        flags.stage = "attempt"
        attempt = _attempt_payload(spec, identity)
        validate_attempt_receipt(attempt, spec, identity)
        hashes: dict[str, str] = {"attempt.json": artifact.publish_json("attempt.json", attempt)}
        validate_attempt_receipt(artifact.reload_json("attempt.json", hashes["attempt.json"]), spec, identity)

        flags.stage = "backend_prepare"
        runtime = backend.prepare(spec, identity, flags)
        if flags.target_or_formal_opened:
            raise RuntimeError("target/formal access is forbidden in Phase-D")
        flags.stage = "epoch"
        launch = _launch_payload(spec, identity)
        validate_launch_receipt(launch, spec, identity)
        hashes["launch.json"] = artifact.publish_json("launch.json", launch)
        validate_launch_receipt(artifact.reload_json("launch.json", hashes["launch.json"]), spec, identity)
        flags.launch_closure = _safe_mapping_copy(identity.closures)
        checkpoint_binding = {
            "cell": CELL, "run_spec": spec.payload(), "launch_sha256": hashes["launch.json"],
            "launch_closure": _safe_mapping_copy(identity.closures),
            "predecessor": _safe_mapping_copy(identity.predecessor),
        }

        global_step = 0
        checkpoint_bodies: dict[int, bytes] = {}
        epoch_sha: list[str] = []
        throughput_written = False
        for epoch in range(spec.epochs):
            backend.begin_epoch(runtime, epoch)
            start = time.perf_counter()
            outcomes: list[StepOutcome] = []
            for _ in range(spec.steps_per_epoch):
                expected_lr = lr_for_spec(spec, global_step)
                boundary_proof = requires_epoch_proof(spec, global_step)
                flags.stage = "optimizer_step"
                outcome = backend.train_step(runtime, expected_lr, global_step, require_epoch_proof=boundary_proof)
                _validate_step_outcome(outcome, expected_lr, require_epoch_proof=boundary_proof)
                outcomes.append(outcome)
                global_step += 1
                flags.optimizer_steps_completed = global_step
                backend.after_optimizer_step(runtime, global_step, flags)
                if global_step == spec.throughput_probe_steps:
                    elapsed = max(time.perf_counter() - start, 1e-12)
                    payload = _throughput_payload(spec, elapsed, backend.resources(runtime), flags)
                    validate_throughput_receipt(payload, spec, identity)
                    name = f"throughput{spec.throughput_probe_steps}.json"
                    hashes[name] = artifact.publish_json(name, payload)
                    validate_throughput_receipt(artifact.reload_json(name, hashes[name]), spec, identity)
                    throughput_written = True
            flags.stage = "epoch"
            elapsed = max(time.perf_counter() - start, 1e-12)
            receipt = _epoch_payload(spec, epoch, global_step, outcomes, elapsed, backend.resources(runtime), flags)
            validate_epoch_receipt(receipt, epoch, global_step, spec, identity)
            name = f"epoch-{epoch:02d}.json"
            digest = artifact.publish_json(name, receipt)
            hashes[name] = digest
            epoch_sha.append(digest)
            validate_epoch_receipt(artifact.reload_json(name, digest), epoch, global_step, spec, identity)
            if epoch in spec.checkpoint_epochs:
                flags.stage = "checkpoint"
                checkpoint = backend.make_checkpoint(runtime, epoch, global_step, checkpoint_binding)
                if not isinstance(checkpoint, CheckpointPayload) or not isinstance(checkpoint.body, bytes) or not _is_sha(checkpoint.model_state_digest):
                    raise RuntimeError("checkpoint backend payload drift")
                backend.validate_checkpoint(checkpoint.body, epoch, global_step, spec, expected_binding=checkpoint_binding)
                name = f"checkpoint-{epoch:02d}.pt"
                digest = artifact.publish_bytes(name, checkpoint.body)
                hashes[name] = digest
                canonical = artifact.reload_pair(name, digest)
                backend.validate_checkpoint(canonical, epoch, global_step, spec, expected_binding=checkpoint_binding)
                checkpoint_bodies[epoch] = canonical
        if global_step != spec.total_optimizer_steps or not throughput_written or len(checkpoint_bodies) != len(spec.checkpoint_epochs):
            raise RuntimeError("terminal training accounting drift")

        flags.stage = "swa"
        swa = backend.build_swa(runtime, checkpoint_bodies, spec, expected_binding=checkpoint_binding)
        if not isinstance(swa, SWAPayload) or not isinstance(swa.body, bytes) or not _is_sha(swa.state_digest) or not isinstance(swa.evaluation_proof, Mapping):
            raise RuntimeError("SWA backend payload drift")
        backend.validate_swa(swa.body, spec, expected_binding=checkpoint_binding)
        hashes["swa.pt"] = artifact.publish_bytes("swa.pt", swa.body)
        backend.validate_swa(artifact.reload_pair("swa.pt", hashes["swa.pt"]), spec,
                             expected_binding=checkpoint_binding)

        flags.stage = "terminal"
        final_identity = identity_factory()
        _validate_identity(final_identity, public=(spec == PUBLIC_SPEC))
        if identity.payload() != final_identity.payload():
            raise RuntimeError("launch/final identity drift")
        validation = _validate_all_preterminal_artifacts(artifact, spec, identity, backend, hashes)
        checkpoint_sha = {str(epoch): hashes[f"checkpoint-{epoch:02d}.pt"] for epoch in spec.checkpoint_epochs}
        terminal = _terminal_payload(spec, identity, final_identity, global_step=global_step,
                                     attempt_sha=hashes["attempt.json"], launch_sha=hashes["launch.json"],
                                     throughput_sha=hashes[f"throughput{spec.throughput_probe_steps}.json"],
                                     epoch_sha=epoch_sha, checkpoint_sha=checkpoint_sha,
                                     swa_sha=hashes["swa.pt"], swa_state_digest=swa.state_digest,
                                     swa_proof=swa.evaluation_proof, artifact_validation=validation)
        validate_terminal_receipt(terminal, spec, identity)
        terminal_sha = artifact.publish_json("terminal.json", terminal)
        validate_terminal_receipt(artifact.reload_json("terminal.json", terminal_sha), spec, identity)
        flags.terminal_published = True
        return terminal
    except BaseException:
        try:
            _publish_failure(artifact, flags, identity)
        finally:
            try:
                backend.close(runtime)
            except BaseException:
                pass
        raise
    finally:
        if flags.terminal_published:
            backend.close(runtime)


def _tensor_digest(state: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode())
        if torch.is_tensor(value):
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            # Adam stores ``step`` as a 0-D tensor.  Reinterpretation through
            # uint8 rejects a scalar, while flattening first preserves the
            # original shape metadata and byte sequence for every non-scalar.
            material = value.detach().cpu().contiguous().reshape(-1)
            digest.update(material.view(torch.uint8).numpy().tobytes())
        else:
            raise RuntimeError("state mapping contains non-tensor entry")
    return digest.hexdigest()


def _jsonable_optimizer(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return {"tensor": _tensor_digest({"value": value}, torch)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable_optimizer(item, torch) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable_optimizer(item, torch) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise RuntimeError("unsupported optimizer state value")


def _optimizer_digest(optimizer: Any, torch: Any) -> str:
    return _sha(_json(_jsonable_optimizer(optimizer.state_dict(), torch)))


def _finite_parameters_and_optimizer(model: Any, optimizer: Any, torch: Any) -> None:
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise RuntimeError("nonfinite model parameter")
    def walk(value: Any) -> None:
        if torch.is_tensor(value) and not torch.isfinite(value).all().item():
            raise RuntimeError("nonfinite optimizer state")
        if isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
    walk(optimizer.state_dict())


def _epoch_boundary_proof(model: Any, optimizer: Any, torch: Any,
                          gradient_prover: Callable[[Any, Any], Mapping[str, bool]]) -> tuple[Mapping[str, bool], bool, bool, str, str]:
    """One physical full-state proof, intentionally called only at epoch end."""
    gradients = gradient_prover(model, torch)
    _strict_bool_map(gradients, _CRITICAL_GRADIENT_GROUPS, "critical gradient")
    _finite_parameters_and_optimizer(model, optimizer, torch)
    return gradients, True, True, _tensor_digest(model.state_dict(), torch), _optimizer_digest(optimizer, torch)


def _batch_iter(adapter: Any, torch: Any):
    for indices in adapter._sampler:
        yield torch.utils.data.default_collate([adapter._dataset[index] for index in indices])


def _capability_for_batch(batch: Any, adapter: Any, authorities: Mapping[str, Any], torch: Any):
    session = batch[3][0]
    info = source_smoke.validate_source_batch(batch, adapter.records[session], set(authorities["roster"]))
    if (info["normalized_side_authority_sha256"] != authorities["normalized_t4_sha256"][session]
            or adapter.raw_t4_sha256[session] != adapter.theta_authority["authority"][session]["raw_t4_sha256"]):
        raise RuntimeError("per-batch source/T4 authority drift")
    capability = source_smoke.capability_from_verified_side(
        info, batch[4].to("cuda:0"), raw_authority_sha256=source_smoke.THETA_ARTIFACT_SHA,
        normalizer_authority_sha256=source_smoke.ADMISSION_SHA,
        roster_digest=source_smoke._digest_json(tuple(authorities["roster"])),
        lineage=(source_smoke.ADMISSION_PREFLIGHT, source_smoke.THETA_RECEIPT, source_smoke.THETA_ARTIFACT,
                 session, info["ordered_unit_digest"], info["normalized_side_authority_sha256"],
                 adapter.raw_t4_sha256[session]),
    )
    return capability, info


class TorchTrainingBackend:
    """The one physical source-only backend; instantiated only after authorization."""

    def __init__(self, root: Path):
        self.root = root

    def prepare(self, spec: RunSpec, identity: RunIdentity, flags: LifecycleFlags) -> dict[str, Any]:
        if spec != PUBLIC_SPEC:
            raise RuntimeError("physical backend accepts only the frozen public RunSpec")
        import random
        import numpy as np
        import torch
        from .model import TFSRDecoder
        authorities = source_smoke.verify_canonical_source_authorities(self.root)
        if _safe_mapping_copy(authorities) != _safe_mapping_copy(identity.source_authorities):
            raise RuntimeError("launch source authorities drift before adapter open")
        source_smoke.require_single_visible_cuda(torch)
        flags.gpu_initialized = True
        flags.source_opened = True
        adapter = source_smoke.build_train_only_adapter(self.root, authorities)
        random.seed(spec.seed); np.random.seed(spec.seed); torch.manual_seed(spec.seed); torch.cuda.manual_seed_all(spec.seed)
        model = TFSRDecoder(capture_diagnostics=False).to("cuda:0")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0, amsgrad=False)
        return {"torch": torch, "adapter": adapter, "authorities": authorities, "model": model,
                "optimizer": optimizer, "iterator": None, "fixed_eval": None}

    def begin_epoch(self, runtime: Mapping[str, Any], epoch: int) -> None:
        torch = runtime["torch"]
        torch.cuda.reset_peak_memory_stats(0)
        runtime["iterator"] = iter(_batch_iter(runtime["adapter"], torch))

    def train_step(self, runtime: Mapping[str, Any], lr: float, global_step: int, *,
                   require_epoch_proof: bool) -> StepOutcome:
        torch, adapter, model, optimizer = runtime["torch"], runtime["adapter"], runtime["model"], runtime["optimizer"]
        batch = next(runtime["iterator"])
        cap, _ = _capability_for_batch(batch, adapter, runtime["authorities"], torch)
        x, behavior, calib = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
        if runtime["fixed_eval"] is None:
            runtime["fixed_eval"] = (x.detach().clone(), calib.detach().clone(), cap)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, calib, cap)
        valid = (behavior != -1.0).all(dim=-1)
        loss = model.dense_valid_bin_mse(prediction, behavior, valid)
        if not torch.isfinite(loss).item():
            raise RuntimeError("nonfinite source-only training loss")
        loss.backward()
        optimizer.step()
        # Full gradients, full parameter/Adam finite scans, and CPU state
        # digests are deliberately paid once at the final optimizer step of
        # each epoch.  The epoch receipt consumes exactly that boundary proof;
        # ordinary steps must not fabricate a cheap stand-in.
        if require_epoch_proof:
            (gradients, finite_model, finite_optimizer, model_state_digest,
             optimizer_state_digest) = _epoch_boundary_proof(
                 model, optimizer, torch, source_smoke.require_critical_gradients,
             )
        else:
            gradients = None
            finite_model = None
            finite_optimizer = None
            model_state_digest = None
            optimizer_state_digest = None
        gain, survivor, p = model.last_unit_gain_mask, model.last_unit_survivor_mask, model.last_dropout_p
        if gain is None or survivor is None or p is None:
            raise RuntimeError("training dropout accounting missing")
        kept = int(survivor.sum().item())
        return StepOutcome(
            loss=float(loss.item()), lr_observed=lr, dropout_p=float(p.item()), kept=kept,
            dropped=survivor.numel() - kept, all_zero_examples=int((survivor.sum(dim=1) == 0).sum().item()),
            population_examples=int(survivor.shape[0]), max_gain=float(gain.max().item()),
            epoch_boundary_proof=require_epoch_proof, critical_gradients=gradients,
            finite_model=finite_model, finite_optimizer=finite_optimizer,
            model_state_digest=model_state_digest, optimizer_state_digest=optimizer_state_digest,
        )

    def after_optimizer_step(self, runtime: Any, global_step: int, flags: LifecycleFlags) -> None:
        return None

    def resources(self, runtime: Mapping[str, Any]) -> Mapping[str, int]:
        import resource
        torch = runtime["torch"]
        return {"rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}

    @staticmethod
    def _checkpoint_bytes(epoch: int, global_step: int, model: Any, optimizer: Any,
                          binding: Mapping[str, Any], torch: Any) -> CheckpointPayload:
        import io
        state = model.state_dict()
        digest = _tensor_digest(state, torch)
        value = {"schema": "tfsr_b3st4_ddrop_checkpoint_v2", "epoch": epoch, "global_step": global_step,
                 "model_state": state, "optimizer_state": optimizer.state_dict(), "model_state_digest": digest,
                 "binding": _safe_mapping_copy(binding)}
        stream = io.BytesIO()
        torch.save(value, stream)
        return CheckpointPayload(stream.getvalue(), digest)

    def make_checkpoint(self, runtime: Mapping[str, Any], epoch: int, global_step: int,
                        binding: Mapping[str, Any]) -> CheckpointPayload:
        return self._checkpoint_bytes(epoch, global_step, runtime["model"], runtime["optimizer"], binding, runtime["torch"])

    def validate_checkpoint(self, body: bytes, epoch: int, global_step: int, spec: RunSpec, *,
                            expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        import io
        import torch
        value = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        expected = {"schema", "epoch", "global_step", "model_state", "optimizer_state", "model_state_digest", "binding"}
        if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_b3st4_ddrop_checkpoint_v2" or value.get("epoch") != epoch or value.get("global_step") != global_step:
            raise RuntimeError("checkpoint schema/accounting drift")
        if not isinstance(value.get("model_state"), Mapping) or _tensor_digest(value["model_state"], torch) != value.get("model_state_digest") or not _is_sha(value.get("model_state_digest")):
            raise RuntimeError("checkpoint model-state digest drift")
        binding = value.get("binding")
        expected_keys = {"cell", "run_spec", "launch_sha256", "launch_closure", "predecessor"}
        if (not isinstance(binding, Mapping) or set(binding) != expected_keys or binding.get("cell") != CELL
                or binding.get("run_spec") != spec.payload() or not _is_sha(binding.get("launch_sha256"))
                or not isinstance(binding.get("launch_closure"), Mapping)):
            raise RuntimeError("checkpoint binding drift")
        _validate_predecessor_lineage(binding.get("predecessor"))
        if expected_binding is not None and _safe_mapping_copy(binding) != _safe_mapping_copy(expected_binding):
            raise RuntimeError("checkpoint exact launch binding drift")
        return value

    def build_swa(self, runtime: Mapping[str, Any], checkpoints: Mapping[int, bytes], spec: RunSpec, *,
                  expected_binding: Mapping[str, Any]) -> SWAPayload:
        import io
        torch = runtime["torch"]
        from .model import TFSRDecoder
        if set(checkpoints) != set(spec.checkpoint_epochs):
            raise RuntimeError("SWA checkpoint epoch set drift")
        states: list[Mapping[str, Any]] = []
        checkpoint_digests: dict[str, str] = {}
        for epoch in spec.checkpoint_epochs:
            item = self.validate_checkpoint(checkpoints[epoch], epoch, (epoch + 1) * spec.steps_per_epoch, spec,
                                            expected_binding=expected_binding)
            states.append(item["model_state"])
            checkpoint_digests[str(epoch)] = item["model_state_digest"]
        keys = tuple(states[0])
        if any(tuple(state) != keys for state in states[1:]):
            raise RuntimeError("SWA checkpoint state key drift")
        swa_state: dict[str, Any] = {}
        for key in keys:
            values = [state[key] for state in states]
            if not all(torch.is_tensor(value) and value.shape == values[0].shape and value.dtype == values[0].dtype for value in values):
                raise RuntimeError("SWA checkpoint tensor type/shape drift")
            if values[0].is_floating_point():
                total = torch.zeros_like(values[0])
                for value in values:
                    total.add_(value)
                swa_state[key] = total.div(len(values))
            elif not all(torch.equal(values[0], value) for value in values[1:]):
                raise RuntimeError("SWA nonfloating checkpoint buffer drift")
            else:
                swa_state[key] = values[0].detach().clone()
        state_digest = _tensor_digest(swa_state, torch)
        fresh = TFSRDecoder(capture_diagnostics=False).to("cuda:0")
        fresh.load_state_dict(swa_state, strict=True)
        if fresh.capture_diagnostics is not False:
            raise RuntimeError("SWA fresh-model capture diagnostics drift")
        fixed = runtime.get("fixed_eval")
        if fixed is None:
            raise RuntimeError("SWA has no fixed verified source capability")
        x, calib, capability = fixed
        fresh.eval()
        state_before = _tensor_digest(fresh.state_dict(), torch)
        with torch.no_grad():
            first = fresh(x, calib, capability)
            first_p = fresh.last_dropout_p
            first_gain, first_survivor = fresh.last_unit_gain_mask, fresh.last_unit_survivor_mask
            second = fresh(x, calib, capability)
            second_p = fresh.last_dropout_p
            second_gain, second_survivor = fresh.last_unit_gain_mask, fresh.last_unit_survivor_mask
        state_after = _tensor_digest(fresh.state_dict(), torch)
        if (not torch.is_tensor(first) or first.shape != (x.shape[0], 50, 2) or not torch.isfinite(first).all().item()
                or not torch.equal(first, second) or first_p is not None or second_p is not None
                or first_gain is None or first_survivor is None or second_gain is None or second_survivor is None
                or not torch.equal(first_gain, torch.ones_like(first_gain)) or not bool(first_survivor.all().item())
                or not torch.equal(second_gain, torch.ones_like(second_gain)) or not bool(second_survivor.all().item())
                or state_before != state_after):
            raise RuntimeError("SWA fixed-source eval/no-mask/state proof drift")
        proof = {
            "checkpoint_epochs": list(spec.checkpoint_epochs), "checkpoint_model_state_digests": checkpoint_digests,
            "fresh_strict_load": True, "eval_mode": True, "capture_diagnostics": False,
            "repeat_bitwise_equal": True, "state_unchanged": True, "eval_no_mask": True,
            "prediction_shape": list(first.shape), "prediction_sha256": _sha(first.detach().cpu().contiguous().numpy().tobytes()),
            "state_digest_before_eval": state_before, "state_digest_after_eval": state_after,
            "boundaries": {"source_only": True, "target_or_formal_opened": False, "scientific_result": False, "score": False},
        }
        value = {"schema": "tfsr_b3st4_ddrop_swa_v2", "state": swa_state, "state_digest": state_digest,
                 "evaluation_proof": proof, "binding": _safe_mapping_copy(expected_binding)}
        stream = io.BytesIO()
        torch.save(value, stream)
        return SWAPayload(stream.getvalue(), state_digest, proof)

    def validate_swa(self, body: bytes, spec: RunSpec,
                     *, expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        import io
        import torch
        value = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        if not isinstance(value, Mapping) or set(value) != {"schema", "state", "state_digest", "evaluation_proof", "binding"} or value.get("schema") != "tfsr_b3st4_ddrop_swa_v2":
            raise RuntimeError("SWA schema drift")
        if not isinstance(value.get("state"), Mapping) or _tensor_digest(value["state"], torch) != value.get("state_digest") or not _is_sha(value.get("state_digest")):
            raise RuntimeError("SWA state digest drift")
        _validate_swa_evaluation_proof(value.get("evaluation_proof"), spec)
        binding = value.get("binding")
        expected_keys = {"cell", "run_spec", "launch_sha256", "launch_closure", "predecessor"}
        if (not isinstance(binding, Mapping) or set(binding) != expected_keys or binding.get("cell") != CELL
                or binding.get("run_spec") != spec.payload() or not _is_sha(binding.get("launch_sha256"))
                or not isinstance(binding.get("launch_closure"), Mapping)):
            raise RuntimeError("SWA binding drift")
        _validate_predecessor_lineage(binding.get("predecessor"))
        if expected_binding is not None and _safe_mapping_copy(binding) != _safe_mapping_copy(expected_binding):
            raise RuntimeError("SWA exact launch binding drift")
        return value

    def close(self, runtime: Any | None) -> None:
        return None


class DeterministicMockBackend:
    """No-data/no-CUDA backend used only by tests to prove the lifecycle.

    It serializes JSON under ``.pt`` names deliberately: all artifact integrity
    checks remain real, while a unit test never needs a tensor runtime.
    """

    def __init__(self, *, failure: str | None = None):
        if failure not in {None, "before_gpu", "during_adapter", "after_step"}:
            raise ValueError("unknown deterministic mock failure")
        self.failure = failure
        self.closed = False
        self.proof_requests: list[bool] = []
        self.expensive_proof_count = 0

    @staticmethod
    def _digest(label: str) -> str:
        return _sha(label.encode())

    def prepare(self, spec: RunSpec, identity: RunIdentity, flags: LifecycleFlags) -> dict[str, Any]:
        if self.failure == "before_gpu":
            raise RuntimeError("synthetic failure before GPU initialization")
        flags.gpu_initialized = True
        if self.failure == "during_adapter":
            flags.source_opened = True
            raise RuntimeError("synthetic failure during source adapter")
        flags.source_opened = True
        return {"epoch": -1, "step": 0}

    def begin_epoch(self, runtime: dict[str, Any], epoch: int) -> None:
        runtime["epoch"] = epoch

    def train_step(self, runtime: dict[str, Any], lr: float, global_step: int, *,
                   require_epoch_proof: bool) -> StepOutcome:
        runtime["step"] += 1
        self.proof_requests.append(require_epoch_proof)
        even = (global_step % 2) == 0
        if require_epoch_proof:
            self.expensive_proof_count += 1
            gradients: Mapping[str, bool] | None = {key: True for key in _CRITICAL_GRADIENT_GROUPS}
            finite_model: bool | None = True
            finite_optimizer: bool | None = True
            model_state_digest: str | None = self._digest(f"model:{global_step}")
            optimizer_state_digest: str | None = self._digest(f"optimizer:{global_step}")
        else:
            gradients = None
            finite_model = None
            finite_optimizer = None
            model_state_digest = None
            optimizer_state_digest = None
        return StepOutcome(
            loss=1.0 + global_step / 10.0, lr_observed=lr, dropout_p=0.25 if even else 0.75,
            kept=6 if even else 4, dropped=2 if even else 4, all_zero_examples=0,
            population_examples=2, max_gain=1.0 / (1.0 - (0.25 if even else 0.75)),
            epoch_boundary_proof=require_epoch_proof, critical_gradients=gradients,
            finite_model=finite_model, finite_optimizer=finite_optimizer,
            model_state_digest=model_state_digest, optimizer_state_digest=optimizer_state_digest,
        )

    def after_optimizer_step(self, runtime: Any, global_step: int, flags: LifecycleFlags) -> None:
        if self.failure == "after_step" and global_step == 1:
            raise RuntimeError("synthetic failure after optimizer step")

    def resources(self, runtime: Any) -> Mapping[str, int]:
        return {"rss_bytes": 1, "peak_allocated_bytes": 2, "peak_reserved_bytes": 3}

    def make_checkpoint(self, runtime: Any, epoch: int, global_step: int, binding: Mapping[str, Any]) -> CheckpointPayload:
        state_digest = self._digest(f"state:{epoch}:{global_step}")
        value = {"schema": "tfsr_b3st4_ddrop_mock_checkpoint_v1", "epoch": epoch,
                 "global_step": global_step, "state_digest": state_digest, "binding": _safe_mapping_copy(binding)}
        return CheckpointPayload(_json(value), state_digest)

    def validate_checkpoint(self, body: bytes, epoch: int, global_step: int, spec: RunSpec, *,
                            expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        value = json.loads(body)
        if not isinstance(value, Mapping) or set(value) != {"schema", "epoch", "global_step", "state_digest", "binding"} or value.get("schema") != "tfsr_b3st4_ddrop_mock_checkpoint_v1" or value.get("epoch") != epoch or value.get("global_step") != global_step or not _is_sha(value.get("state_digest")):
            raise RuntimeError("mock checkpoint validation drift")
        binding = value.get("binding")
        if (not isinstance(binding, Mapping) or set(binding) != {"cell", "run_spec", "launch_sha256", "launch_closure", "predecessor"}
                or binding.get("cell") != CELL or binding.get("run_spec") != spec.payload()
                or not _is_sha(binding.get("launch_sha256")) or not isinstance(binding.get("launch_closure"), Mapping)):
            raise RuntimeError("mock checkpoint binding drift")
        _validate_predecessor_lineage(binding.get("predecessor"))
        if expected_binding is not None and _safe_mapping_copy(binding) != _safe_mapping_copy(expected_binding):
            raise RuntimeError("mock checkpoint exact launch binding drift")
        return value

    def build_swa(self, runtime: Any, checkpoints: Mapping[int, bytes], spec: RunSpec, *,
                  expected_binding: Mapping[str, Any]) -> SWAPayload:
        if set(checkpoints) != set(spec.checkpoint_epochs):
            raise RuntimeError("mock SWA checkpoint set drift")
        digests = {str(epoch): self.validate_checkpoint(body, epoch, (epoch + 1) * spec.steps_per_epoch, spec,
                                                         expected_binding=expected_binding)["state_digest"]
                   for epoch, body in checkpoints.items()}
        state_digest = self._digest("mock-swa:" + ",".join(digests.values()))
        proof = {"checkpoint_epochs": list(spec.checkpoint_epochs), "checkpoint_model_state_digests": digests,
                 "fresh_strict_load": True, "eval_mode": True, "capture_diagnostics": False,
                 "repeat_bitwise_equal": True, "state_unchanged": True, "eval_no_mask": True,
                 "prediction_shape": [2, 50, 2], "prediction_sha256": self._digest("mock-prediction"),
                 "state_digest_before_eval": state_digest, "state_digest_after_eval": state_digest,
                 "boundaries": {"source_only": True, "target_or_formal_opened": False, "scientific_result": False, "score": False}}
        value = {"schema": "tfsr_b3st4_ddrop_mock_swa_v1", "state_digest": state_digest,
                 "evaluation_proof": proof, "binding": _safe_mapping_copy(expected_binding)}
        return SWAPayload(_json(value), state_digest, proof)

    def validate_swa(self, body: bytes, spec: RunSpec,
                     *, expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        value = json.loads(body)
        if not isinstance(value, Mapping) or set(value) != {"schema", "state_digest", "evaluation_proof", "binding"} or value.get("schema") != "tfsr_b3st4_ddrop_mock_swa_v1" or not _is_sha(value.get("state_digest")):
            raise RuntimeError("mock SWA schema drift")
        proof = value.get("evaluation_proof")
        if not isinstance(proof, Mapping) or proof.get("checkpoint_epochs") != list(spec.checkpoint_epochs) or proof.get("fresh_strict_load") is not True or proof.get("eval_no_mask") is not True:
            raise RuntimeError("mock SWA proof drift")
        binding = value.get("binding")
        expected_keys = {"cell", "run_spec", "launch_sha256", "launch_closure", "predecessor"}
        if (not isinstance(binding, Mapping) or set(binding) != expected_keys or binding.get("cell") != CELL
                or binding.get("run_spec") != spec.payload() or not _is_sha(binding.get("launch_sha256"))
                or not isinstance(binding.get("launch_closure"), Mapping)):
            raise RuntimeError("mock SWA binding drift")
        _validate_predecessor_lineage(binding.get("predecessor"))
        if expected_binding is not None and _safe_mapping_copy(binding) != _safe_mapping_copy(expected_binding):
            raise RuntimeError("mock SWA exact launch binding drift")
        return value

    def close(self, runtime: Any | None) -> None:
        self.closed = True


def mock_identity() -> RunIdentity:
    """Explicit synthetic identity for isolated lifecycle tests only."""
    return RunIdentity(
        phase_c_acceptance={"body_sha256": "a" * 64, "path": "synthetic_phase_c.json"},
        source_authorities={"manifest_sha256": "b" * 64, "roster": ["synthetic-source"]},
        closures={"stage0": {"closure_sha256": "c" * 64}, "phase_c": {"closure_sha256": "d" * 64}, "phase_d": {"closure_sha256": "e" * 64}},
        device={"internal_device": "synthetic:0"},
        predecessor={
            "predecessor_root_relative": "synthetic/v1",
            "failure_relative": "synthetic/v1/failure.json",
            "failure_sha256": "f" * 64,
            "v1_accepted_checkpoints": 0,
            "v1_resume_forbidden": True,
            "successor_workorder_relative": "synthetic/workorder.md",
            "successor_workorder_sha256": "0" * 64,
        },
    )


def execute_training(root: Path) -> None:
    """Authorized-only public lifecycle.  There are no CLI budget overrides."""
    output_gate(root)
    # Verify all immutable inputs before reserving a successor root.  A
    # predecessor/closure failure therefore cannot leave a partial v2 alias.
    production_identity(root)
    artifact = reserve_output_root(root)
    run_lifecycle(spec=PUBLIC_SPEC, backend=TorchTrainingBackend(root), artifact=artifact,
                  identity_factory=lambda: production_identity(root))
