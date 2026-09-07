"""Fail-closed Phase-E matched-score scaffold for TF-SR seed 42.

This module deliberately has *only* standard-library imports at module load.
The public command is therefore an inspection-only dry plan: it cannot import
Torch, discover an NWB path, construct a model, touch CUDA, or create a score
root.  Real tensor/data work is behind an injected backend and can be reached
only after a future root-reviewed authority pair has been accepted.

The implementation is intentionally additive.  It does not change the Phase-D
trainer or either decoder.  Its job is to make a future matched Cell-D versus
TF-SR evaluation auditable, with Cell-D replay as a parity blocker rather than
as a loose reference number.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


CELL = "TFSR_B3ST4_DDROP_SEED42"
PHASE = "TFSR_PHASE_E_MATCHED_SCORE_V1"
AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_score_authority_v1"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_matched_score_v1"
SCORE_TOPOLOGY = (
    "attempt.json",
    "input_authority.json",
    "score.json",
    "terminal.json",
    "failure.json",
)

# The Phase-E closure is intentionally explicit.  This is not a recursive
# source glob: each file below is a physical runtime dependency of the only
# live no-cache route.  Phase-D/model integrity is additionally checked by
# the accepted terminal, but that does not make its direct runtime imports
# invisible to Phase-E provenance.
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_TFSR_PHASE_E_MATCHED_SCORE_20260819.md"
WORKORDER_SHA256 = "c4131d7bbdc3a979f45a6873ef872180a3805ef5a51826af93e7a7c563fa9958"
# ``validate_phase_d_training_terminal`` validates every epoch receipt before
# any Phase-E evaluation path may be resolved.  That validation calls
# ``train.lr_for_step``, whose ``from tfpd_lane.arm_common import lr_at_step``
# first executes the top-level ``tfpd_lane`` package initializer.  Its eager
# imports are therefore live Phase-E runtime dependencies too -- not merely
# Phase-D provenance.  ``mech_diag`` and ``pregate`` in turn import the three
# ``src.tfpd`` leaves already listed immediately below.  Keep this complete
# package-import chain explicit; a glob would not protect a future launch.
TFPD_LANE_LR_IMPORT_CLOSURE = (
    "tfpd_exploration/src/tfpd_lane/__init__.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/mech_diag.py",
    "tfpd_exploration/src/tfpd_lane/pregate.py",
    "tfpd_exploration/src/tfpd_lane/receipt.py",
)
PHASE_E_CLOSURE = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/score.py",
    "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed42_score.py",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_score.py",
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/__init__.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/model.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/train.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/source_smoke.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/contract.py",
    *TFPD_LANE_LR_IMPORT_CLOSURE,
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd/__init__.py",
    "tfpd_exploration/src/tfpd/bilinear_readin.py",
    "tfpd_exploration/src/tfpd/population_vector.py",
    "tfpd_exploration/src/tfpd/synth.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
)

SOURCE_SMOKE_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_v1/source_smoke_receipt.json"
STRICT_MANIFEST_RELATIVE = "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
# The strict manifest freezes the source/within session labels, but it is not
# the immutable byte authority for the six live within-development NWBs.  That
# authority is the paired-view C1 manifest below.  It is intentionally mode
# 0600 and has no sidecar, so it must be descriptor-read by the dedicated
# helper rather than being silently accepted through the generic 0444/0664
# AuthoritySpec path.
WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE = (
    "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v1/"
    "c1_train_val_33_manifest.json"
)
A2_RECEIPT_RELATIVE = "tfpd_exploration/results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json"
CELL_D_TABLE_RELATIVE = "tfpd_exploration/results/sparsification_step0_v1/step0_receipt.json"
CELL_D_TERMINAL_RELATIVE = "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json"
CELL_D_SWA_RELATIVE = "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt"
EXTERNAL_LEDGER_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_schema_preflight_v2/receipt.json"
EXTERNAL_SCOPE_RELATIVE = "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json"
# Phase-D v1 is an immutable failed predecessor.  Phase-E may only consume the
# separately reviewed v2 successor terminal/SWA chain.
PHASE_D_TRAIN_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_train_v2"

STRICT_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
WITHIN_PAIRED_VIEW_MANIFEST_SHA256 = "bb3440b688b6d16dabbf91db3ce43e91711f1e9e389de4b241e80a827fbcab7d"
WITHIN_PAIRED_VIEW_MANIFEST_MODE = 0o600

# Keep this literal independent of a caller-supplied mapping.  These are the
# exact six `file_inventory.val` rows from the paired-view manifest, retained
# here so durable preflight validation remains meaningful even after the
# descriptor used to read the 0600 manifest has been closed.
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

WITHIN_PAIRED_VIEW_CLOSURE_BINDING = {
    "relative_path": WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE,
    "body_sha256": WITHIN_PAIRED_VIEW_MANIFEST_SHA256,
    "mode": format(WITHIN_PAIRED_VIEW_MANIFEST_MODE, "04o"),
    "sidecar_required": False,
    "descriptor_read_required": True,
}

# The Cell-D terminal's strict initial-state proof is the authority for the
# initialized, trainable deployable parameter count.  The decoder also
# intentionally retains two dead lazy parameters; they are not treated as
# zero-sized parameters and must be disclosed as a separate topology fact.
CELL_D_INITIALIZED_TRAINABLE_PARAMETERS = 3_510_842
CELL_D_UNINITIALIZED_LAZY_PARAMETER_KEYS = (
    "decoder.fc_id_in.0.weight",
    "decoder.fc_id_in.0.bias",
)
CELL_D_UNINITIALIZED_LAZY_PARAMETER_ROLE = "dead_decoder_fc_id_in_identity_path"
CELL_D_PARAMETER_ACCOUNTING_SCHEMA = "tfsr_phase_e_cell_d_lazy_safe_parameter_accounting_v1"

AUTHORITY_TOPOLOGY = ("official_preflight.json", "root_authorization.json")

T4_NORMALIZER_SEMANTIC_SHA = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
BEHAVIOR_NORMALIZER_SEMANTIC_SHA = "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
SOURCE_NORMALIZER_BODY_SHA = "2632c6a6a4cfb8a4c0fb2b23e0cc8205ea323240b376903c60d5b27110f59e43"
STAGE0_CLOSURE_SHA = "0e7154000d670e06edeee19e6c57dfe5a5ae7606e993f62650b83b906d858ae4"
PHASE_C_CLOSURE_SHA = "2be132f3f3e4f8a232e40c02da62f9e34bc9d43966f6279ce3f4e9f05be0b1bc"

FORMAL_TEST_SESSION_NAMES = (
    "sub-C_ses-CO-20151113", "sub-C_ses-CO-20151116", "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119", "sub-C_ses-CO-20151120", "sub-C_ses-CO-20151201",
)


class FailClosedError(RuntimeError):
    """A boundary/integrity violation; never reinterpret it as a score."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).encode("utf-8") + b"\n"


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _require_sha(value: object, label: str) -> str:
    if not _is_sha(value):
        raise FailClosedError(f"{label} must be an exact lowercase SHA-256")
    return str(value)


def _safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise FailClosedError("path must be canonical safe relative")
    return value


def _regular_identity(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size)


def _read_fd_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        item = os.read(descriptor, 1024 * 1024)
        if not item:
            return b"".join(chunks)
        chunks.append(item)


def _canonical_regular_bytes(root: Path, relative: str, *, expected_mode: int) -> tuple[bytes, tuple[int, int, int]]:
    """Read one exact non-symlink regular file through a stable descriptor.

    This intentionally rejects aliases, final-path symlinks, replacement during
    read, and mode drift.  The caller receives the descriptor identity that a
    live adapter must bind before passing a file to an NWB parser.
    """
    relative = _safe_relative(relative)
    root = root.absolute()
    try:
        root_info = os.lstat(root)
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            raise FailClosedError("repository root is not canonical directory")
        path = root / relative
        absolute = path.absolute()
        if path.resolve(strict=True) != absolute:
            raise FailClosedError(f"authority alias/symlink forbidden: {relative}")
        before = os.lstat(absolute)
    except OSError as error:
        raise FailClosedError(f"cannot lstat canonical authority: {relative}") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise FailClosedError(f"authority is not a regular non-symlink: {relative}")
    if stat.S_IMODE(before.st_mode) != expected_mode:
        raise FailClosedError(f"authority mode drift: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise FailClosedError(f"cannot O_NOFOLLOW open authority: {relative}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or stat.S_ISLNK(opened.st_mode):
            raise FailClosedError(f"opened authority is not a regular file: {relative}")
        if stat.S_IMODE(opened.st_mode) != expected_mode or _regular_identity(before) != _regular_identity(opened):
            raise FailClosedError(f"authority identity/mode changed before read: {relative}")
        body = _read_fd_all(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(absolute)
    except OSError as error:
        raise FailClosedError(f"cannot post-lstat authority: {relative}") from error
    if (not stat.S_ISREG(after.st_mode) or stat.S_ISLNK(after.st_mode)
            or stat.S_IMODE(after.st_mode) != expected_mode
            or _regular_identity(before) != _regular_identity(after)
            or len(body) != before.st_size):
        raise FailClosedError(f"authority changed during read: {relative}")
    return body, _regular_identity(before)


@dataclass(frozen=True)
class AuthoritySpec:
    name: str
    relative: str
    sha256: str
    mode: int
    sidecar_required: bool

    def __post_init__(self) -> None:
        _safe_relative(self.relative)
        _require_sha(self.sha256, self.name)
        if self.mode not in {0o444, 0o664}:
            raise ValueError("authority must freeze its exact known mode")


FIXED_AUTHORITIES = (
    AuthoritySpec("source_smoke", SOURCE_SMOKE_RELATIVE,
                  "022ca7a253e208c86c846b593bc684372ac9ab21197db7a974277416df90dfbc", 0o444, True),
    AuthoritySpec("strict_manifest", STRICT_MANIFEST_RELATIVE, STRICT_MANIFEST_SHA256, 0o664, False),
    AuthoritySpec("a2_matched_reference", A2_RECEIPT_RELATIVE,
                  "0ae0f74c6b5d606599d1108378836c8cbb7d1e5df05a453aa83f682255734d4e", 0o444, True),
    AuthoritySpec("cell_d_table", CELL_D_TABLE_RELATIVE,
                  "92b5cef8c3fd1a023b1c560c95cfc13af739cb948351a4231d7be41bea59fdca", 0o444, True),
    AuthoritySpec("cell_d_terminal", CELL_D_TERMINAL_RELATIVE,
                  "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442", 0o444, True),
    AuthoritySpec("cell_d_swa", CELL_D_SWA_RELATIVE,
                  "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd", 0o444, True),
    AuthoritySpec("external_asset_ledger", EXTERNAL_LEDGER_RELATIVE,
                  "1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283", 0o444, False),
    AuthoritySpec("external_scope", EXTERNAL_SCOPE_RELATIVE,
                  "68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55", 0o444, False),
)


@dataclass(frozen=True)
class AuthorityMaterial:
    spec: AuthoritySpec
    body: bytes
    value: Mapping[str, Any] | None
    identity: tuple[int, int, int]

    def binding(self) -> dict[str, object]:
        return {
            "relative_path": self.spec.relative,
            "body_sha256": self.spec.sha256,
            "mode": format(self.spec.mode, "04o"),
            "descriptor_identity": list(self.identity),
        }


def verify_authority(root: Path, spec: AuthoritySpec) -> AuthorityMaterial:
    body, identity = _canonical_regular_bytes(root, spec.relative, expected_mode=spec.mode)
    if _sha(body) != spec.sha256:
        raise FailClosedError(f"{spec.name} SHA-256 drift")
    if spec.sidecar_required:
        sidecar_relative = spec.relative + ".sha256"
        sidecar, _ = _canonical_regular_bytes(root, sidecar_relative, expected_mode=0o444)
        expected = f"{spec.sha256}  {Path(spec.relative).name}\n".encode("ascii")
        if sidecar != expected:
            raise FailClosedError(f"{spec.name} malformed sidecar")
    value: Mapping[str, Any] | None = None
    if spec.relative.endswith(".json"):
        try:
            decoded = json.loads(body)
        except (TypeError, json.JSONDecodeError) as error:
            raise FailClosedError(f"{spec.name} is malformed JSON") from error
        if not isinstance(decoded, Mapping):
            raise FailClosedError(f"{spec.name} JSON root must be object")
        value = decoded
    return AuthorityMaterial(spec=spec, body=body, value=value, identity=identity)


def verify_fixed_authorities(root: Path) -> dict[str, AuthorityMaterial]:
    """Load only fixed, sealed metadata authorities; never a neural/data asset."""
    checked = {item.name: verify_authority(root, item) for item in FIXED_AUTHORITIES}
    if set(checked) != {item.name for item in FIXED_AUTHORITIES}:
        raise FailClosedError("fixed authority set drift")
    return checked


def phase_e_closure(root: Path) -> dict[str, object]:
    sha256_by_path: dict[str, str] = {}
    for relative in PHASE_E_CLOSURE:
        body, _ = _canonical_regular_bytes(root, relative, expected_mode=0o664)
        sha256_by_path[relative] = _sha(body)
    if sha256_by_path.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
        raise FailClosedError("Phase-E work-order SHA drift")
    immutable_input_bindings = {
        "within_paired_view_manifest": dict(WITHIN_PAIRED_VIEW_CLOSURE_BINDING),
    }
    encoded = json.dumps(
        {"files": sha256_by_path, "immutable_input_bindings": immutable_input_bindings},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "paths": list(PHASE_E_CLOSURE),
        "sha256_by_path": sha256_by_path,
        "immutable_input_bindings": immutable_input_bindings,
        "closure_sha256": _sha(encoded),
    }


def dry_plan() -> dict[str, object]:
    """Pure static declaration: deliberately no repository read or Torch import."""
    return {
        "cell": CELL,
        "phase": PHASE,
        "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_SCORE",
        "authorization": "none",
        "execution_flags_required_together": ["--execute", "--i-have-phase-e-root-authorization"],
        "score_root": SCORE_ROOT_RELATIVE,
        "authority_root": AUTHORITY_ROOT_RELATIVE,
        "topology": list(SCORE_TOPOLOGY),
        "metric": {
            "governing": "torchmetrics.R2Score(multioutput='variance_weighted')",
            "query": "last_valid_bin_of_50",
            "aggregation": "unweighted_equal_session_mean",
            "batch_size": 128,
            "autograd": "forbidden",
        },
        "fixed": {
            "M": 30, "T": 50, "bin_ms": 20, "padding": -1.0,
            "within_paired_view_manifest": dict(WITHIN_PAIRED_VIEW_CLOSURE_BINDING),
            "t4_normalizer_semantic_sha256": T4_NORMALIZER_SEMANTIC_SHA,
            "behavior_normalizer_semantic_sha256": BEHAVIOR_NORMALIZER_SEMANTIC_SHA,
            "source_normalizer_body_sha256": SOURCE_NORMALIZER_BODY_SHA,
            "accepted_stage0_closure_sha256": STAGE0_CLOSURE_SHA,
            "accepted_phase_c_closure_sha256": PHASE_C_CLOSURE_SHA,
        },
        "forbidden": [
            "target_optimizer", "backward", "target_update", "checkpoint_selection", "session_selection",
            "formal_session_resolution", "A2_checkpoint_live_pass", "unit_table", "session_table",
        ],
    }


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise FailClosedError(f"cannot stat artifact directory: {path}") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise FailClosedError("artifact directory is not canonical")
    return (info.st_dev, info.st_ino)


def _write_full(descriptor: int, body: bytes) -> None:
    view = memoryview(body)
    while view:
        wrote = os.write(descriptor, view)
        if wrote <= 0:
            raise OSError("short artifact write")
        view = view[wrote:]


@dataclass(frozen=True)
class ArtifactRoot:
    """A named-directory capability for immutable Phase-E publications.

    It carries both the root and parent inode.  Each operation proves that the
    same directory is still named beneath the same parent before it creates or
    reloads a body/sidecar pair.  This prevents a post-reservation rename from
    redirecting a receipt into a replacement directory.
    """

    directory: Path
    topology: tuple[str, ...]
    identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]

    def _assert_named_identity(self) -> None:
        if _directory_identity(self.directory) != self.identity:
            raise FailClosedError("artifact root path identity drift")
        try:
            pfd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as error:
            raise FailClosedError("cannot open artifact parent") from error
        try:
            info = os.fstat(pfd)
            if (info.st_dev, info.st_ino) != self.parent_identity:
                raise FailClosedError("artifact parent identity drift")
            named = os.stat(self.directory.name, dir_fd=pfd, follow_symlinks=False)
            if (not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode)
                    or (named.st_dev, named.st_ino) != self.identity):
                raise FailClosedError("artifact named-root identity drift")
        finally:
            os.close(pfd)

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or name not in self.topology or "/" in name or name in {"", ".", ".."}:
            raise FailClosedError("artifact name outside exact topology")

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

    def _leaf_exists(self, dfd: int, name: str) -> bool:
        try:
            os.stat(name, dir_fd=dfd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def publish_bytes(self, name: str, body: bytes) -> str:
        """Atomically publish one body/sidecar pair or roll back owned leaves."""
        self._check_name(name)
        if not isinstance(body, bytes):
            raise TypeError("artifact body must be bytes")
        self._assert_named_identity()
        dfd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        created: list[tuple[str, int, int]] = []
        try:
            opened = os.fstat(dfd)
            if (opened.st_dev, opened.st_ino) != self.identity:
                raise FailClosedError("artifact root changed between checks")
            # A lone body or lone sidecar is an attempted output collision, not
            # something a future run may silently complete.
            if self._leaf_exists(dfd, name) or self._leaf_exists(dfd, name + ".sha256"):
                raise FailClosedError("artifact body/sidecar collision")
            digest = _sha(body)
            payloads = ((name, body), (name + ".sha256", f"{digest}  {name}\n".encode("ascii")))
            for leaf, payload in payloads:
                fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dfd)
                try:
                    made = os.fstat(fd)
                    if not stat.S_ISREG(made.st_mode):
                        raise FailClosedError("artifact O_EXCL did not make regular file")
                    created.append((leaf, made.st_dev, made.st_ino))
                    _write_full(fd, payload)
                    os.fchmod(fd, 0o444)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(dfd)
            # A descriptor-level reload is a publication proof, not merely an
            # optimistic read through a pathname after closing the writer.
            for leaf, expected in payloads:
                fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
                try:
                    info = os.fstat(fd)
                    actual = _read_fd_all(fd)
                    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444
                            or actual != expected):
                        raise FailClosedError("artifact post-write reload drift")
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
            for leaf, device, inode in reversed(created):
                try:
                    now = os.stat(leaf, dir_fd=dfd, follow_symlinks=False)
                    if (now.st_dev, now.st_ino) == (device, inode):
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

    def publish_group(
        self,
        bodies: Mapping[str, bytes],
        *,
        post_publish: Callable[[Mapping[str, bytes], Mapping[str, str]], None] | None = None,
    ) -> dict[str, str]:
        """Publish an all-or-nothing group of immutable receipt pairs.

        The score and terminal form one scientific publication unit: a failure
        while making either pair must not leave a valid-looking ``score.json``
        without its terminal.  The group is written through one opened
        directory FD, every potential collision is rejected *before* the first
        leaf is created, and only inodes made by this transaction are removed
        on any error.
        """
        if not isinstance(bodies, Mapping) or not bodies:
            raise TypeError("artifact group must be a nonempty mapping")
        names = tuple(sorted(bodies))
        if len(set(names)) != len(names):
            raise FailClosedError("duplicate artifact group name")
        for name in names:
            self._check_name(name)
            if not isinstance(bodies[name], bytes):
                raise TypeError("artifact group bodies must be bytes")
        self._assert_named_identity()
        dfd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        created: list[tuple[str, int, int]] = []
        try:
            opened = os.fstat(dfd)
            if (opened.st_dev, opened.st_ino) != self.identity:
                raise FailClosedError("artifact root changed between group checks")
            for name in names:
                if self._leaf_exists(dfd, name) or self._leaf_exists(dfd, name + ".sha256"):
                    raise FailClosedError("artifact group body/sidecar collision")
            digests = {name: _sha(bodies[name]) for name in names}
            payloads: list[tuple[str, bytes]] = []
            for name in names:
                payloads.append((name, bodies[name]))
                payloads.append((name + ".sha256", f"{digests[name]}  {name}\n".encode("ascii")))
            for leaf, payload in payloads:
                fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dfd)
                try:
                    made = os.fstat(fd)
                    if not stat.S_ISREG(made.st_mode):
                        raise FailClosedError("artifact group O_EXCL did not make regular file")
                    created.append((leaf, made.st_dev, made.st_ino))
                    _write_full(fd, payload)
                    os.fchmod(fd, 0o444)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(dfd)
            # Validate every pair through the same held directory capability.
            for name in names:
                self.reload_pair(name, digests[name])
            # The caller may need to validate cross-pair semantics (for
            # example, terminal.score_sha256 must bind the exact score body).
            # Run that check *inside* this transaction, before its inodes can
            # become a partially valid scientific result.  Any exception uses
            # the owned-inode rollback below.
            if post_publish is not None:
                post_publish(dict(bodies), dict(digests))
            self._assert_named_identity()
            pfd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(pfd)
            finally:
                os.close(pfd)
            return digests
        except BaseException:
            for leaf, device, inode in reversed(created):
                try:
                    now = os.stat(leaf, dir_fd=dfd, follow_symlinks=False)
                    if (now.st_dev, now.st_ino) == (device, inode):
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
                try:
                    fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
                except OSError as error:
                    raise FailClosedError(f"missing/corrupt artifact leaf: {leaf}") from error
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                        raise FailClosedError("artifact type/mode drift")
                    return _read_fd_all(fd)
                finally:
                    os.close(fd)
            body = read(name)
            digest = _sha(body)
            if expected_sha is not None and digest != expected_sha:
                raise FailClosedError("artifact SHA drift")
            if read(name + ".sha256") != f"{digest}  {name}\n".encode("ascii"):
                raise FailClosedError("artifact sidecar drift")
            self._assert_named_identity()
            return body
        finally:
            os.close(dfd)

    def publish_json(self, name: str, payload: Mapping[str, Any]) -> str:
        return self.publish_bytes(name, _json_bytes(payload))

    def reload_json(self, name: str, expected_sha: str | None = None) -> Mapping[str, Any]:
        try:
            value = json.loads(self.reload_pair(name, expected_sha))
        except (TypeError, json.JSONDecodeError) as error:
            raise FailClosedError("artifact JSON decode drift") from error
        if not isinstance(value, Mapping):
            raise FailClosedError("artifact JSON root must be object")
        return value


def reserve_artifact_root(parent: Path, name: str, topology: tuple[str, ...] = SCORE_TOPOLOGY) -> ArtifactRoot:
    """Reserve a fresh exact root before any future data path resolution."""
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ValueError("artifact root name must be one path component")
    if (not isinstance(topology, tuple) or not topology or len(topology) != len(set(topology))
            or any(not isinstance(item, str) or not item or "/" in item for item in topology)):
        raise ValueError("artifact topology must be exact nonempty leaves")
    parent = parent.absolute()
    parent_identity = _directory_identity(parent)
    pfd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(pfd)
        if (opened.st_dev, opened.st_ino) != parent_identity:
            raise FailClosedError("artifact parent identity drift before reservation")
        try:
            os.stat(name, dir_fd=pfd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FailClosedError("fresh canonical score root required")
        os.mkdir(name, 0o755, dir_fd=pfd)
        os.fsync(pfd)
        created = os.stat(name, dir_fd=pfd, follow_symlinks=False)
        if not stat.S_ISDIR(created.st_mode) or stat.S_ISLNK(created.st_mode):
            raise FailClosedError("reserved score root is invalid")
        identity = (created.st_dev, created.st_ino)
    finally:
        os.close(pfd)
    return ArtifactRoot(parent / name, topology, identity, parent, parent_identity)


def canonical_score_parent(root: Path) -> tuple[Path, str]:
    relative = _safe_relative(SCORE_ROOT_RELATIVE)
    path = root.absolute() / relative
    return path.parent, path.name


@dataclass(frozen=True)
class ScoreSpec:
    """The immutable matched-estimand contract, injectable only for mocks."""

    within_count: int = 6
    external_count: int = 15
    window_bins: int = 50
    calibration_trials: int = 30
    batch_size: int = 128
    bootstrap_draws: int = 10_000
    bootstrap_seed: int = 42

    def __post_init__(self) -> None:
        if (self.within_count, self.external_count, self.window_bins, self.calibration_trials,
                self.batch_size, self.bootstrap_draws, self.bootstrap_seed) != (6, 15, 50, 30, 128, 10_000, 42):
            raise ValueError("public Phase-E ScoreSpec is fully frozen")

    def payload(self) -> dict[str, int]:
        return {
            "within_count": self.within_count, "external_count": self.external_count,
            "window_bins": self.window_bins, "calibration_trials": self.calibration_trials,
            "batch_size": self.batch_size, "bootstrap_draws": self.bootstrap_draws,
            "bootstrap_seed": self.bootstrap_seed,
        }


PUBLIC_SPEC = ScoreSpec()


@dataclass
class ScoreFlags:
    stage: str = "non_data_authority"
    source_resolved: bool = False
    within_resolved: bool = False
    external_resolved: bool = False
    formal_resolved: bool = False
    source_opened: bool = False
    within_opened: bool = False
    external_opened: bool = False
    formal_opened: bool = False
    forward_calls: dict[str, dict[str, dict[str, int]]] = field(default_factory=lambda: {
        "cell_d": {
            "within": {"aligned": 0},
            "external": {"aligned": 0},
        },
        "tfsr": {
            "within": {"aligned": 0, "zero": 0, "wrong_pair": 0},
            "external": {"aligned": 0, "zero": 0, "wrong_pair": 0},
        },
    })
    backward_calls: int = 0
    optimizer_calls: int = 0
    terminal_published: bool = False

    def record_forward(self, system: str, surface: str, mode: str) -> None:
        try:
            current = self.forward_calls[system][surface][mode]
        except (KeyError, TypeError) as error:
            raise FailClosedError("unrecognized system/surface/mode forward accounting") from error
        if type(current) is not int or current < 0:
            raise FailClosedError("forward accounting counter drift")
        self.forward_calls[system][surface][mode] = current + 1

    @staticmethod
    def validate_forward_calls(value: object) -> dict[str, dict[str, dict[str, int]]]:
        expected = {
            "cell_d": {"within": {"aligned"}, "external": {"aligned"}},
            "tfsr": {
                "within": {"aligned", "zero", "wrong_pair"},
                "external": {"aligned", "zero", "wrong_pair"},
            },
        }
        if not isinstance(value, Mapping) or set(value) != set(expected):
            raise FailClosedError("forward accounting system matrix drift")
        checked: dict[str, dict[str, dict[str, int]]] = {}
        for system, surfaces in expected.items():
            block = value.get(system)
            if not isinstance(block, Mapping) or set(block) != set(surfaces):
                raise FailClosedError("forward accounting surface matrix drift")
            checked[system] = {}
            for surface, modes in surfaces.items():
                row = block.get(surface)
                if not isinstance(row, Mapping) or set(row) != modes:
                    raise FailClosedError("forward accounting mode matrix drift")
                if any(type(count) is not int or count < 0 for count in row.values()):
                    raise FailClosedError("forward accounting count drift")
                checked[system][surface] = {mode: int(row[mode]) for mode in sorted(modes)}
        return checked

    def payload(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "resolved": {
                "source": self.source_resolved, "within": self.within_resolved,
                "external": self.external_resolved, "formal": self.formal_resolved,
            },
            "opened": {
                "source": self.source_opened, "within": self.within_opened,
                "external": self.external_opened, "formal": self.formal_opened,
            },
            "forward_calls": self.validate_forward_calls(self.forward_calls),
            "backward_calls": self.backward_calls,
            "optimizer_calls": self.optimizer_calls,
            "terminal_published": self.terminal_published,
        }


@dataclass(frozen=True)
class ScoreIdentity:
    fixed_authorities: Mapping[str, Mapping[str, object]]
    training_terminal_sha256: str
    training_swa_sha256: str
    training_swa_state_digest: str
    launch_closure: Mapping[str, object]
    phase_e_authorization: Mapping[str, object]

    def __post_init__(self) -> None:
        for label, value in (
            ("training terminal", self.training_terminal_sha256),
            ("training SWA", self.training_swa_sha256),
            ("training SWA state", self.training_swa_state_digest),
        ):
            _require_sha(value, label)
        if not isinstance(self.fixed_authorities, Mapping) or not self.fixed_authorities:
            raise ValueError("score identity must bind fixed authorities")
        if not isinstance(self.launch_closure, Mapping) or not isinstance(self.phase_e_authorization, Mapping):
            raise ValueError("score identity closure/authorization must be mappings")

    def payload(self) -> dict[str, object]:
        return {
            "fixed_authorities": json.loads(json.dumps(self.fixed_authorities, sort_keys=True)),
            "training_terminal_sha256": self.training_terminal_sha256,
            "training_swa_sha256": self.training_swa_sha256,
            "training_swa_state_digest": self.training_swa_state_digest,
            "launch_closure": json.loads(json.dumps(self.launch_closure, sort_keys=True)),
            "phase_e_authorization": json.loads(json.dumps(self.phase_e_authorization, sort_keys=True)),
        }


def _require_string_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise FailClosedError(f"{label} must be nonempty ordered string list")
    if len(set(value)) != len(value):
        raise FailClosedError(f"{label} contains duplicate sessions")
    return tuple(value)


def extract_within_roster(strict_manifest: Mapping[str, Any], *, formal_names: Sequence[str] = FORMAL_TEST_SESSION_NAMES) -> tuple[str, ...]:
    """Take exactly `session_splits.val`, never a convenience path helper."""
    if not isinstance(strict_manifest, Mapping):
        raise FailClosedError("strict manifest is not mapping")
    splits = strict_manifest.get("session_splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "val", "test"}:
        raise FailClosedError("strict manifest split schema drift")
    val = _require_string_sequence(splits.get("val"), "strict manifest val")
    if len(val) != PUBLIC_SPEC.within_count:
        raise FailClosedError("within roster must be exact strict-manifest val[6]")
    forbidden = set(formal_names)
    if any(name in forbidden for name in val):
        raise FailClosedError("formal name illegally appears in within roster")
    return val


@dataclass(frozen=True)
class DataRootCapability:
    """A held named-directory identity for one target-data root.

    File SHA verification alone prevents the wrong bytes from reaching a
    parser, but it does not prove that the root authorized after ``attempt``
    remained the root that supplied them.  This compact capability binds the
    root inode and its parent name; every target file open/reverify repeats the
    check through a parent directory FD.
    """

    directory: Path
    identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]

    @classmethod
    def from_directory(cls, path: Path, *, label: str) -> "DataRootCapability":
        directory = _canonical_data_root(path, label=label)
        parent = directory.parent
        return cls(
            directory=directory,
            identity=_directory_identity(directory),
            parent=parent,
            parent_identity=_directory_identity(parent),
        )

    def assert_named_identity(self) -> None:
        if _directory_identity(self.directory) != self.identity:
            raise FailClosedError("target data root path identity drift")
        try:
            pfd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as error:
            raise FailClosedError("cannot open target data-root parent") from error
        try:
            parent_info = os.fstat(pfd)
            if (parent_info.st_dev, parent_info.st_ino) != self.parent_identity:
                raise FailClosedError("target data-root parent identity drift")
            named = os.stat(self.directory.name, dir_fd=pfd, follow_symlinks=False)
            if (not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode)
                    or (named.st_dev, named.st_ino) != self.identity):
                raise FailClosedError("target data-root named identity drift")
        finally:
            os.close(pfd)

    def payload(self) -> dict[str, object]:
        return {
            "directory_basename": self.directory.name,
            "directory_identity": [self.identity[0], self.identity[1]],
            "parent_identity": [self.parent_identity[0], self.parent_identity[1]],
        }


@dataclass(frozen=True)
class ExternalAssetBinding:
    """The only legal external asset address, produced from frozen ledger joins."""

    asset_id: str
    session: str
    local_path: Path
    expected_bytes: int
    expected_sha256: str
    frozen_path: str
    data_root: DataRootCapability | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.asset_id, str) or not self.asset_id:
            raise ValueError("asset ID must be nonempty")
        if not isinstance(self.session, str) or not self.session:
            raise ValueError("external session must be nonempty")
        if type(self.expected_bytes) is not int or self.expected_bytes <= 0:
            raise ValueError("asset byte count must be positive exact int")
        _require_sha(self.expected_sha256, "asset SHA")
        _safe_relative(self.frozen_path)
        if self.local_path.name != Path(self.frozen_path).name:
            raise ValueError("external local path must be canonical basename of frozen path")
        if self.data_root is not None and self.local_path.parent.absolute() != self.data_root.directory:
            raise ValueError("external local path must be rooted in its held data-root capability")

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "asset_id": self.asset_id, "session": self.session,
            "canonical_local_basename": self.local_path.name,
            "bytes": self.expected_bytes, "sha256": self.expected_sha256,
            "frozen_path": self.frozen_path,
        }
        if self.data_root is not None:
            payload["data_root"] = self.data_root.payload()
        return payload


@dataclass(frozen=True)
class EvaluationAssetBinding:
    """One target asset permitted to reach the no-cache parser.

    External rows are derived exclusively from the frozen v2 UUID ledger.
    Within rows arrive only as preflight-bound immutable metadata.  Both share
    the same held-FD/no-follow parser boundary afterwards.
    """

    surface: str
    asset_id: str
    session: str
    local_path: Path
    expected_bytes: int
    expected_sha256: str
    frozen_path: str
    data_root: DataRootCapability | None = None

    def __post_init__(self) -> None:
        if self.surface not in {"within", "external"}:
            raise ValueError("evaluation asset surface drift")
        if not isinstance(self.asset_id, str) or not self.asset_id:
            raise ValueError("evaluation asset ID drift")
        if not isinstance(self.session, str) or not self.session:
            raise ValueError("evaluation asset session drift")
        if type(self.expected_bytes) is not int or self.expected_bytes <= 0:
            raise ValueError("evaluation asset bytes drift")
        _require_sha(self.expected_sha256, "evaluation asset SHA")
        _safe_relative(self.frozen_path)
        if self.local_path.name != Path(self.frozen_path).name:
            raise ValueError("evaluation asset basename drift")
        if self.data_root is not None and self.local_path.parent.absolute() != self.data_root.directory:
            raise ValueError("evaluation asset path must be rooted in its held data-root capability")

    def payload(self, *, held_descriptor_identity: tuple[int, int, int] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "surface": self.surface,
            "asset_id": self.asset_id,
            "session": self.session,
            "canonical_local_basename": self.local_path.name,
            "bytes": self.expected_bytes,
            "sha256": self.expected_sha256,
            "frozen_path": self.frozen_path,
        }
        if self.data_root is not None:
            payload["data_root"] = self.data_root.payload()
        if held_descriptor_identity is not None:
            if (not isinstance(held_descriptor_identity, tuple) or len(held_descriptor_identity) != 3
                    or any(type(item) is not int or item < 0 for item in held_descriptor_identity)):
                raise ValueError("held descriptor identity drift")
            payload["held_descriptor_identity"] = list(held_descriptor_identity)
        return payload


def _as_evaluation_asset(binding: ExternalAssetBinding) -> EvaluationAssetBinding:
    return EvaluationAssetBinding(
        surface="external", asset_id=binding.asset_id, session=binding.session,
        local_path=binding.local_path, expected_bytes=binding.expected_bytes,
        expected_sha256=binding.expected_sha256, frozen_path=binding.frozen_path,
        data_root=binding.data_root,
    )


def _unique_by_key(rows: object, *, key: str, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise FailClosedError(f"{label} must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get(key), str) or not row[key]:
            raise FailClosedError(f"{label} row/key drift")
        item = str(row[key])
        if item in result:
            raise FailClosedError(f"{label} has duplicate {key}")
        result[item] = row
    return result


def _eligible_external_rows(
    ledger_receipt: Mapping[str, Any], *, formal_names: Sequence[str] = FORMAL_TEST_SESSION_NAMES,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve v2 ``eligible_session_ids`` as *asset UUIDs*, never names.

    The unfortunately named field is a UUID list in the fixed v2 receipt.  A
    permissive ``session_id`` lookup silently creates a different roster, so
    this join is kept in one small helper and used by both metadata-only roster
    extraction and the eventual pathname binding.
    """
    if not isinstance(ledger_receipt, Mapping):
        raise FailClosedError("external ledger receipt is not mapping")
    ledger = _unique_by_key(ledger_receipt.get("asset_disposition_ledger"), key="asset_id", label="asset ledger")
    eligible_asset_ids = _require_string_sequence(ledger_receipt.get("eligible_session_ids"), "eligible asset UUIDs")
    if (len(eligible_asset_ids) != PUBLIC_SPEC.external_count
            or ledger_receipt.get("eligible_session_count") != PUBLIC_SPEC.external_count):
        raise FailClosedError("external roster must have exact 15 eligible assets")
    rows: list[Mapping[str, Any]] = []
    forbidden = set(formal_names)
    sessions: set[str] = set()
    for asset_id in eligible_asset_ids:
        row = ledger.get(asset_id)
        if row is None:
            raise FailClosedError("eligible asset UUID missing from disposition ledger")
        session = row.get("session_id")
        if (row.get("asset_id") != asset_id or row.get("eligible") is not True
                or row.get("disposition") != "ELIGIBLE" or not isinstance(session, str) or not session):
            raise FailClosedError("eligible asset/ledger semantics drift")
        if session in forbidden:
            raise FailClosedError("formal session appeared in external ledger scope")
        if session in sessions:
            raise FailClosedError("eligible assets map to duplicate canonical session")
        sessions.add(session)
        rows.append(row)
    if len(rows) != PUBLIC_SPEC.external_count or len(sessions) != PUBLIC_SPEC.external_count:
        raise FailClosedError("external eligible asset cardinality drift")
    return tuple(sorted(rows, key=lambda row: str(row["session_id"])))


def join_external_assets(
    ledger_receipt: Mapping[str, Any],
    scope_manifest: Mapping[str, Any],
    subm_data_root: Path,
    *,
    formal_names: Sequence[str] = FORMAL_TEST_SESSION_NAMES,
) -> tuple[ExternalAssetBinding, ...]:
    """Join the *fixed v2 ledger* by asset ID, without an A2 scorer/path helper.

    The verified-download cache pathname is deliberately never returned.  A
    future evaluator must instead use ``SUBM_DATA_ROOT / basename(frozen_path)``
    and hold that exact file descriptor across parsing.  Thus a mutable cache,
    a rerun of a path-discovering scorer, or a conveniently named formal file
    cannot enter the live score path.
    """
    if not isinstance(ledger_receipt, Mapping) or not isinstance(scope_manifest, Mapping):
        raise FailClosedError("external authorities are not mappings")
    data_root = DataRootCapability.from_directory(subm_data_root, label="external")
    ledger = _unique_by_key(ledger_receipt.get("asset_disposition_ledger"), key="asset_id", label="asset ledger")
    downloads = _unique_by_key(ledger_receipt.get("verified_downloads"), key="asset_id", label="verified downloads")
    selected = _unique_by_key(scope_manifest.get("selected_assets"), key="asset_id", label="scope selected assets")
    eligible_rows = _eligible_external_rows(ledger_receipt, formal_names=formal_names)
    bindings: list[ExternalAssetBinding] = []
    # Sort by canonical session name once and preserve it through every later
    # paired statistic.  No performance or target signal participates.
    for row in eligible_rows:
        session = str(row["session_id"])
        asset_id = row.get("asset_id")
        if row.get("eligible") is not True or row.get("disposition") != "ELIGIBLE" or not isinstance(asset_id, str):
            raise FailClosedError("eligible ledger semantics drift")
        verified = downloads.get(asset_id)
        scoped = selected.get(asset_id)
        if verified is None or scoped is None:
            raise FailClosedError("eligible asset missing verified/scope join")
        frozen_path = row.get("frozen_path")
        if (not isinstance(frozen_path, str) or scoped.get("path") != frozen_path
                or scoped.get("session_id") != session):
            raise FailClosedError("ledger/scope frozen path/session mismatch")
        expected_bytes = verified.get("bytes")
        expected_sha = verified.get("sha256")
        if (type(expected_bytes) is not int or expected_bytes <= 0 or not _is_sha(expected_sha)
                or verified.get("asset_id") != asset_id or scoped.get("sha256") != expected_sha
                or scoped.get("size") != expected_bytes
                or verified.get("size_and_sha256_verified_before_nwb_open") is not True):
            raise FailClosedError("external verified-download integrity drift")
        # The use of basename is an explicit security/authority rule from the
        # frozen work order, not a convenience interpretation of `verified.path`.
        local = data_root.directory / Path(frozen_path).name
        bindings.append(ExternalAssetBinding(
            asset_id=asset_id, session=session, local_path=local,
            expected_bytes=expected_bytes, expected_sha256=str(expected_sha), frozen_path=frozen_path,
            data_root=data_root,
        ))
    if len(bindings) != PUBLIC_SPEC.external_count or len({item.asset_id for item in bindings}) != len(bindings):
        raise FailClosedError("external binding cardinality/identity drift")
    return tuple(bindings)


def extract_external_sessions(ledger_receipt: Mapping[str, Any], *, formal_names: Sequence[str] = FORMAL_TEST_SESSION_NAMES) -> tuple[str, ...]:
    """Read canonical session names only after the v2 UUID-to-row join."""
    return tuple(str(row["session_id"]) for row in _eligible_external_rows(ledger_receipt, formal_names=formal_names))


def _canonical_data_root(path: Path, *, label: str) -> Path:
    """Accept one real, non-symlink target-data directory only after attempt.

    The public CLI never calls this helper.  It deliberately does not discover
    candidate files: callers have already bound every legal basename and SHA
    in a sealed authority, so this only makes that directory capability
    explicit before a pathname is formed.
    """
    candidate = path.absolute()
    try:
        info = os.lstat(candidate)
    except OSError as error:
        raise FailClosedError(f"{label} data root is absent") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise FailClosedError(f"{label} data root is not a canonical directory")
    return candidate


def join_within_assets(
    preflight: Mapping[str, object],
    subc_data_root: Path,
) -> tuple[EvaluationAssetBinding, ...]:
    """Materialize only the six root-authorized within bindings.

    The source of truth is the signed target-free preflight—not a directory
    scan, an A2 helper, or an inferred session name.  The local target is
    always the canonical root joined with the frozen basename.
    """
    if not isinstance(preflight, Mapping):
        raise FailClosedError("within binding requires a validated preflight")
    roster = _require_string_sequence(preflight.get("within_roster"), "preflight within roster")
    if len(roster) != PUBLIC_SPEC.within_count:
        raise FailClosedError("within binding roster cardinality drift")
    rows = _validate_preflight_within_assets(preflight.get("within_assets"), tuple(sorted(roster)))
    data_root = DataRootCapability.from_directory(subc_data_root, label="within")
    bindings: list[EvaluationAssetBinding] = []
    for row in rows:
        session = str(row["session"])
        frozen_path = str(row["frozen_path"])
        bindings.append(EvaluationAssetBinding(
            surface="within", asset_id=str(row["asset_id"]), session=session,
            local_path=data_root.directory / Path(frozen_path).name,
            expected_bytes=int(row["bytes"]), expected_sha256=str(row["sha256"]),
            frozen_path=frozen_path, data_root=data_root,
        ))
    if tuple(item.session for item in bindings) != tuple(sorted(roster)):
        raise FailClosedError("within binding roster/order drift")
    return tuple(bindings)


@dataclass
class HeldAsset:
    """An O_NOFOLLOW file descriptor that remains valid until parser teardown."""

    binding: EvaluationAssetBinding
    descriptor: int
    identity: tuple[int, int, int]

    def reverify(self) -> None:
        if self.binding.data_root is None:
            raise FailClosedError("held target asset lacks data-root capability")
        self.binding.data_root.assert_named_identity()
        info = os.fstat(self.descriptor)
        if (not stat.S_ISREG(info.st_mode) or _regular_identity(info) != self.identity
                or info.st_size != self.binding.expected_bytes):
            raise FailClosedError("held external descriptor identity/size drift")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        observed = _sha(_read_fd_all(self.descriptor))
        if observed != self.binding.expected_sha256:
            raise FailClosedError("held external descriptor SHA drift")
        try:
            named = os.lstat(self.binding.local_path)
        except OSError as error:
            raise FailClosedError("external canonical path disappeared") from error
        if (not stat.S_ISREG(named.st_mode) or stat.S_ISLNK(named.st_mode)
                or _regular_identity(named) != self.identity):
            raise FailClosedError("external canonical pathname identity drift")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "HeldAsset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def hold_verified_asset(binding: EvaluationAssetBinding) -> HeldAsset:
    """Open and hash one evaluation file before *any* parser sees its bytes."""
    if binding.data_root is None:
        raise FailClosedError("target asset open requires held data-root capability")
    binding.data_root.assert_named_identity()
    path = binding.local_path
    try:
        before = os.lstat(path)
    except OSError as error:
        raise FailClosedError("cannot lstat external canonical asset") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise FailClosedError("external canonical asset is not regular non-symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FailClosedError("cannot O_NOFOLLOW open external asset") from error
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or _regular_identity(before) != _regular_identity(opened)
                or opened.st_size != binding.expected_bytes):
            raise FailClosedError("external asset identity/size changed before parse")
        observed = _sha(_read_fd_all(descriptor))
        if observed != binding.expected_sha256:
            raise FailClosedError("external asset SHA mismatch before parse")
        held = HeldAsset(binding=binding, descriptor=descriptor, identity=_regular_identity(opened))
        held.reverify()
        return held
    except BaseException:
        os.close(descriptor)
        raise


def hold_verified_external_asset(binding: ExternalAssetBinding) -> HeldAsset:
    """Compatibility wrapper for the fixed-v2 external binding API."""
    return hold_verified_asset(_as_evaluation_asset(binding))


@dataclass
class HeldSnapshot:
    """Private no-cache parser view copied from a held verified descriptor."""

    held: HeldAsset
    temporary_directory: tempfile.TemporaryDirectory[str]
    path: Path

    def reverify(self) -> None:
        self.held.reverify()
        body, identity = _canonical_regular_bytes(
            self.path.parent, self.path.name, expected_mode=0o400,
        )
        if identity[2] != self.held.binding.expected_bytes or _sha(body) != self.held.binding.expected_sha256:
            raise FailClosedError("private verified parser snapshot drift")

    def close(self) -> None:
        try:
            self.held.close()
        finally:
            self.temporary_directory.cleanup()

    def __enter__(self) -> "HeldSnapshot":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def private_verified_snapshot(held: HeldAsset) -> HeldSnapshot:
    """Materialize a parser-only snapshot while retaining the verified FD.

    Many NWB APIs reopen a pathname internally.  Passing the canonical target
    pathname after its initial hash would reopen a mutable name.  This helper
    copies from the still-held descriptor to an inaccessible temporary regular
    file whose basename remains the canonical session filename, verifies both
    byte streams, and keeps the original descriptor alive until the caller's
    full score lifecycle has finished.
    """
    held.reverify()
    directory = tempfile.TemporaryDirectory(prefix="tfsr-phase-e-")
    target = Path(directory.name) / held.binding.local_path.name
    fd = -1
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.lseek(held.descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(held.descriptor, 1024 * 1024)
            if not block:
                break
            _write_full(fd, block)
        os.fchmod(fd, 0o400)
        os.fsync(fd)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        directory.cleanup()
        raise
    else:
        os.close(fd)
    snapshot = HeldSnapshot(held=held, temporary_directory=directory, path=target)
    try:
        snapshot.reverify()
        return snapshot
    except BaseException:
        snapshot.close()
        raise


def _torch_runtime() -> tuple[Any, type[Any]]:
    """Deferred tensor import: unreachable from public dry plan/CLI default."""
    import torch
    from .model import NormalizedT4Batch
    return torch, NormalizedT4Batch


def _tensor_digest(tensor: Any) -> str:
    # This runs only through an explicit injected backend/test after tensor
    # import.  Exact dtype/shape are separately bound in `tensor_binding`.
    return _sha(tensor.detach().cpu().contiguous().numpy().tobytes())


def tensor_binding(tensor: Any) -> dict[str, object]:
    torch, _ = _torch_runtime()
    if not torch.is_tensor(tensor):
        raise FailClosedError("expected tensor for compact authority binding")
    return {
        "dtype": str(tensor.dtype), "shape": list(tensor.shape),
        "bytes_sha256": _tensor_digest(tensor),
    }


@dataclass(frozen=True)
class T4ControlBundle:
    aligned: Any
    zero: Any
    wrong_pair: Any
    permutation: tuple[int, ...]
    permutation_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "aligned": tensor_binding(self.aligned.tensor),
            "zero": tensor_binding(self.zero.tensor),
            "wrong_pair": tensor_binding(self.wrong_pair.tensor),
            "permutation_sha256": self.permutation_sha256,
            "permutation_is_derangement": True,
            "ordered_unit_digest": self.aligned.ordered_unit_digest,
            "unit_count": self.aligned.num_units,
        }


def make_t4_controls(aligned: Any) -> T4ControlBundle:
    """Make typed post-normalization aligned/zero/cyclic-wrong capabilities.

    A bare Tensor is deliberately rejected.  The zero and wrong-pair controls
    retain the receiving unit-axis identity and explicitly change only the
    model-visible normalized T4 values.
    """
    torch, capability_type = _torch_runtime()
    if not isinstance(aligned, capability_type):
        raise FailClosedError("T4 controls require NormalizedT4Batch capability")
    aligned.validate()
    if aligned.diagnostic_mode != "aligned":
        raise FailClosedError("control source must be aligned normalized T4")
    if aligned.num_units < 2:
        raise FailClosedError("wrong-pair control requires at least two units")
    n = aligned.num_units
    # `[1,2,...,0]` is a fixed cyclic derangement for every N >= 2.
    permutation_tensor = torch.remainder(torch.arange(n, device=aligned.tensor.device, dtype=torch.long) + 1, n)
    if bool(torch.any(permutation_tensor == torch.arange(n, device=aligned.tensor.device, dtype=torch.long)).item()):
        raise FailClosedError("wrong-pair permutation is not derangement")
    values = aligned.tensor.index_select(1, permutation_tensor).detach().clone()
    zero_values = torch.zeros_like(aligned.tensor)
    common = {
        "raw_authority_sha256": aligned.raw_authority_sha256,
        "normalizer_authority_sha256": aligned.normalizer_authority_sha256,
        "roster_digest": aligned.roster_digest,
        "ordered_unit_digest": aligned.ordered_unit_digest,
        "ordered_unit_ids": aligned.ordered_unit_ids,
        "lineage": aligned.lineage,
    }
    zero = capability_type(tensor=zero_values, diagnostic_mode="zero", **common)
    wrong = capability_type(tensor=values, diagnostic_mode="wrong_pair", **common)
    if not torch.equal(zero.tensor, torch.zeros_like(aligned.tensor)):
        raise FailClosedError("zero control is not exact zeros_like")
    # A sorted row-value multiset comparison would be needlessly ambiguous for
    # repeated values; index-select establishes exact value multiset lineage.
    if not torch.equal(wrong.tensor, aligned.tensor.index_select(1, permutation_tensor)):
        raise FailClosedError("wrong-pair did not preserve aligned T4 multiset")
    permutation = tuple(int(item) for item in permutation_tensor.detach().cpu().tolist())
    encoded = json.dumps(permutation, separators=(",", ":")).encode("ascii")
    return T4ControlBundle(aligned=aligned, zero=zero, wrong_pair=wrong,
                           permutation=permutation, permutation_sha256=_sha(encoded))


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FailClosedError("expected finite scalar")
    return float(value)


def _ordered_scores(values: Mapping[str, float], expected_sessions: Sequence[str]) -> tuple[float, ...]:
    expected = tuple(sorted(expected_sessions))
    if tuple(sorted(values)) != expected:
        raise FailClosedError("paired score session roster/order drift")
    return tuple(_finite_float(values[name]) for name in expected)


def paired_statistics(deltas: Sequence[float], *, seed: int = 42, draws: int = 10_000) -> dict[str, object]:
    """House paired-session summary, deferred NumPy import, no model/data path."""
    if seed != 42 or draws != 10_000:
        raise FailClosedError("paired bootstrap seed/draw count is frozen")
    if not deltas:
        raise FailClosedError("paired statistics needs nonempty deltas")
    # NumPy’s Generator/percentile exactly matches the already established
    # matched-scorer convention.  It is imported only after an authorized
    # backend has completed a scoring surface, never on public dry startup.
    import numpy as np
    data = np.asarray([_finite_float(item) for item in deltas], dtype=np.float64)
    generator = np.random.default_rng(seed)
    n = len(data)
    boot = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        boot[index] = data[generator.integers(0, n, size=n)].mean()
    return {
        "mean": float(data.mean()), "median": float(np.median(data)),
        "n_positive": int((data > 0).sum()), "n_total": int(n),
        "min": float(data.min()), "max": float(data.max()),
        "all_deltas": [float(item) for item in data],
        "bootstrap_95_interval": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "exact_sign_pattern": "".join("+" if item > 0 else ("-" if item < 0 else "0") for item in data),
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
    }


def decide_verdict(*, external: Mapping[str, object], within: Mapping[str, object]) -> str:
    """Apply the one predeclared decision in its required STOP/CLEAR/HOLD order."""
    for label, stats, count in (("external", external, 15), ("within", within, 6)):
        if not isinstance(stats, Mapping) or stats.get("n_total") != count:
            raise FailClosedError(f"{label} paired statistic schema/count drift")
        for key in ("mean", "median"):
            _finite_float(stats.get(key))
        if type(stats.get("n_positive")) is not int or not 0 <= int(stats["n_positive"]) <= count:
            raise FailClosedError(f"{label} positive count drift")
    external_mean = float(external["mean"])
    within_mean = float(within["mean"])
    if external_mean < 0.0 or within_mean < -0.03:
        return "STOP"
    if (external_mean >= 0.03 and within_mean >= -0.03
            and float(external["median"]) > 0.0 and int(external["n_positive"]) >= 9):
        return "CLEAR_GO"
    return "HOLD"


def variance_weighted_r2(predictions: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> float:
    """Dense [n,2] variance-weighted R² reference for no-data synthetic tests.

    For nonconstant outputs this is algebraically equivalent to the
    ``torchmetrics.R2Score(multioutput='variance_weighted')`` estimator used
    by the live backend: sum per-output squared residuals divided by sum
    per-output total variance.  The live backend must still call torchmetrics;
    this routine makes the expected reduction independently inspectable.
    """
    if len(predictions) != len(targets) or not predictions:
        raise FailClosedError("R2 needs equal nonempty prediction/target rows")
    width: int | None = None
    pred_rows: list[tuple[float, ...]] = []
    target_rows: list[tuple[float, ...]] = []
    for prediction, target in zip(predictions, targets):
        if not isinstance(prediction, Sequence) or not isinstance(target, Sequence) or len(prediction) != len(target):
            raise FailClosedError("R2 row shape drift")
        if width is None:
            width = len(prediction)
        if width is None or width < 1 or len(prediction) != width:
            raise FailClosedError("R2 output width drift")
        pred_rows.append(tuple(_finite_float(item) for item in prediction))
        target_rows.append(tuple(_finite_float(item) for item in target))
    assert width is not None
    means = [sum(row[channel] for row in target_rows) / len(target_rows) for channel in range(width)]
    residual = sum((pred_rows[row][channel] - target_rows[row][channel]) ** 2
                   for row in range(len(target_rows)) for channel in range(width))
    total = sum((target_rows[row][channel] - means[channel]) ** 2
                for row in range(len(target_rows)) for channel in range(width))
    if total <= 0.0:
        raise FailClosedError("R2 undefined with all constant targets")
    return 1.0 - residual / total


def last_bin_r2(
    predictions: Sequence[Sequence[Sequence[float]]],
    targets: Sequence[Sequence[Sequence[float]]],
    valid_mask: Sequence[Sequence[bool]],
) -> float:
    """Extract one final valid query bin per window, matching deployed scoring."""
    if len(predictions) != len(targets) or len(predictions) != len(valid_mask) or not predictions:
        raise FailClosedError("last-bin R2 window cardinality drift")
    collected_predictions: list[Sequence[float]] = []
    collected_targets: list[Sequence[float]] = []
    for prediction, target, mask in zip(predictions, targets, valid_mask):
        if len(prediction) != len(target) or len(prediction) != len(mask) or len(mask) != PUBLIC_SPEC.window_bins:
            raise FailClosedError("last-bin R2 50-bin window shape drift")
        valid = [index for index, item in enumerate(mask) if item is True]
        if not valid:
            raise FailClosedError("last-bin R2 empty valid window")
        index = valid[-1]
        collected_predictions.append(prediction[index])
        collected_targets.append(target[index])
    return variance_weighted_r2(collected_predictions, collected_targets)


def full_window_r2(
    predictions: Sequence[Sequence[Sequence[float]]],
    targets: Sequence[Sequence[Sequence[float]]],
    valid_mask: Sequence[Sequence[bool]],
) -> float:
    """Diagnostic all-valid-bin R²; it cannot alter a governing decision."""
    if len(predictions) != len(targets) or len(predictions) != len(valid_mask) or not predictions:
        raise FailClosedError("full-window R2 window cardinality drift")
    collected_predictions: list[Sequence[float]] = []
    collected_targets: list[Sequence[float]] = []
    for prediction, target, mask in zip(predictions, targets, valid_mask):
        if len(prediction) != len(target) or len(prediction) != len(mask) or len(mask) != PUBLIC_SPEC.window_bins:
            raise FailClosedError("full-window R2 50-bin window shape drift")
        for index, is_valid in enumerate(mask):
            if is_valid is True:
                collected_predictions.append(prediction[index])
                collected_targets.append(target[index])
    return variance_weighted_r2(collected_predictions, collected_targets)


@dataclass(frozen=True)
class SessionScore:
    session: str
    n_windows: int
    governing_r2: float
    full_window_r2: float
    output_sha256: str
    input_authority_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.session, str) or not self.session or type(self.n_windows) is not int or self.n_windows <= 0:
            raise ValueError("session score session/window count drift")
        _finite_float(self.governing_r2)
        _finite_float(self.full_window_r2)
        _require_sha(self.output_sha256, "session output")
        _require_sha(self.input_authority_sha256, "session input authority")

    def payload(self) -> dict[str, object]:
        return {
            "session": self.session, "n_windows": self.n_windows,
            "governing_r2": self.governing_r2, "full_window_r2": self.full_window_r2,
            "output_sha256": self.output_sha256,
            "input_authority_sha256": self.input_authority_sha256,
        }


@dataclass(frozen=True)
class ModeEvidence:
    """One system/surface/mode forward result supplied by an injected backend."""

    system: str
    surface: str
    mode: str
    sessions: tuple[SessionScore, ...]
    state_before_sha256: str
    state_after_sha256: str
    eval_mode: bool
    gradients_none: bool
    finite_output: bool
    output_shape: tuple[int, int, int]
    repeat_bitwise_equal: bool | None = None
    eval_no_mask: bool | None = None
    capture_diagnostics: bool | None = None
    b3s_recomputed: bool | None = None
    t4_control: Mapping[str, object] | None = None
    latency_ms: float | None = None
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.system not in {"cell_d", "tfsr"} or self.surface not in {"within", "external"}:
            raise ValueError("mode evidence system/surface drift")
        allowed = {"aligned"} if self.system == "cell_d" else {"aligned", "zero", "wrong_pair"}
        if self.mode not in allowed:
            raise ValueError("mode evidence invalid diagnostic mode")
        required_count = PUBLIC_SPEC.within_count if self.surface == "within" else PUBLIC_SPEC.external_count
        if len(self.sessions) != required_count or len({item.session for item in self.sessions}) != required_count:
            raise ValueError("mode evidence session roster cardinality drift")
        if tuple(item.session for item in self.sessions) != tuple(sorted(item.session for item in self.sessions)):
            raise ValueError("mode evidence sessions must be sorted")
        _require_sha(self.state_before_sha256, "state before")
        _require_sha(self.state_after_sha256, "state after")
        if self.eval_mode is not True or self.gradients_none is not True or self.finite_output is not True:
            raise ValueError("mode evidence eval/gradient/finite invariant drift")
        if self.state_before_sha256 != self.state_after_sha256:
            raise ValueError("mode evidence model state changed during score")
        if (not isinstance(self.output_shape, tuple) or len(self.output_shape) != 3 or self.output_shape[0] <= 0
                or self.output_shape[1:] != (50, 2)):
            raise ValueError("mode evidence output shape must be [B,50,2]")
        if self.system == "tfsr":
            if self.capture_diagnostics is not False or self.eval_no_mask is not True:
                raise ValueError("TF-SR eval/no-mask/state invariant drift")
            if self.mode == "aligned" and self.repeat_bitwise_equal is not True:
                raise ValueError("TF-SR aligned repeat proof missing")
            if self.b3s_recomputed is not True or not isinstance(self.t4_control, Mapping):
                raise ValueError("TF-SR control/B3S proof missing")
        if self.latency_ms is not None and _finite_float(self.latency_ms) < 0.0:
            raise ValueError("latency drift")
        if self.peak_memory_bytes is not None and (type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0):
            raise ValueError("peak memory drift")

    @property
    def mean_governing(self) -> float:
        return sum(item.governing_r2 for item in self.sessions) / len(self.sessions)

    @property
    def mean_full_window(self) -> float:
        return sum(item.full_window_r2 for item in self.sessions) / len(self.sessions)

    def payload(self) -> dict[str, object]:
        return {
            "system": self.system, "surface": self.surface, "mode": self.mode,
            "sessions": [item.payload() for item in self.sessions],
            "equal_session_mean_governing_r2": self.mean_governing,
            "equal_session_mean_full_window_r2": self.mean_full_window,
            "state_before_sha256": self.state_before_sha256,
            "state_after_sha256": self.state_after_sha256,
            "eval_mode": self.eval_mode, "gradients_none": self.gradients_none,
            "finite_output": self.finite_output, "output_shape": list(self.output_shape),
            "repeat_bitwise_equal": self.repeat_bitwise_equal, "eval_no_mask": self.eval_no_mask,
            "capture_diagnostics": self.capture_diagnostics, "b3s_recomputed": self.b3s_recomputed,
            "t4_control": dict(self.t4_control) if self.t4_control is not None else None,
            "latency_ms": self.latency_ms, "peak_memory_bytes": self.peak_memory_bytes,
        }


def mode_evidence_from_payload(value: Mapping[str, Any]) -> ModeEvidence:
    """Strictly reconstruct one persisted mode cell before receipt reuse."""
    expected = {
        "system", "surface", "mode", "sessions", "equal_session_mean_governing_r2",
        "equal_session_mean_full_window_r2", "state_before_sha256", "state_after_sha256",
        "eval_mode", "gradients_none", "finite_output", "output_shape", "repeat_bitwise_equal",
        "eval_no_mask", "capture_diagnostics", "b3s_recomputed", "t4_control", "latency_ms",
        "peak_memory_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value.get("sessions"), list):
        raise FailClosedError("persisted mode evidence schema drift")
    try:
        sessions = tuple(SessionScore(
            session=item["session"], n_windows=item["n_windows"], governing_r2=item["governing_r2"],
            full_window_r2=item["full_window_r2"], output_sha256=item["output_sha256"],
            input_authority_sha256=item["input_authority_sha256"],
        ) for item in value["sessions"] if isinstance(item, Mapping))
        shape = value["output_shape"]
        if not isinstance(shape, list):
            raise ValueError("output shape must be list")
        evidence = ModeEvidence(
            system=value["system"], surface=value["surface"], mode=value["mode"], sessions=sessions,
            state_before_sha256=value["state_before_sha256"], state_after_sha256=value["state_after_sha256"],
            eval_mode=value["eval_mode"], gradients_none=value["gradients_none"], finite_output=value["finite_output"],
            output_shape=tuple(shape), repeat_bitwise_equal=value["repeat_bitwise_equal"],
            eval_no_mask=value["eval_no_mask"], capture_diagnostics=value["capture_diagnostics"],
            b3s_recomputed=value["b3s_recomputed"], t4_control=value["t4_control"],
            latency_ms=value["latency_ms"], peak_memory_bytes=value["peak_memory_bytes"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FailClosedError("persisted mode evidence nested value drift") from error
    if len(sessions) != len(value["sessions"]) or evidence.payload() != dict(value):
        raise FailClosedError("persisted mode evidence roundtrip drift")
    return evidence


def _sealed_d_surface(table: Mapping[str, Any], surface: str) -> tuple[tuple[str, int, float], ...]:
    results = table.get("results")
    d = results.get("D_swa") if isinstance(results, Mapping) else None
    block = d.get(surface) if isinstance(d, Mapping) else None
    if not isinstance(block, Mapping) or not isinstance(block.get("per_session"), list):
        raise FailClosedError("sealed Cell-D table surface schema drift")
    rows: list[tuple[str, int, float]] = []
    for row in block["per_session"]:
        if (not isinstance(row, Mapping) or not isinstance(row.get("session"), str)
                or type(row.get("n_windows")) is not int or row["n_windows"] <= 0):
            raise FailClosedError("sealed Cell-D per-session schema drift")
        rows.append((row["session"], row["n_windows"], _finite_float(row.get("r2"))))
    expected_count = PUBLIC_SPEC.within_count if surface == "within" else PUBLIC_SPEC.external_count
    if len(rows) != expected_count or len({row[0] for row in rows}) != expected_count:
        raise FailClosedError("sealed Cell-D surface cardinality drift")
    if tuple(row[0] for row in rows) != tuple(sorted(row[0] for row in rows)):
        raise FailClosedError("sealed Cell-D table ordering drift")
    expected_mean = _finite_float(block.get("mean_r2"))
    if sum(row[2] for row in rows) / len(rows) != expected_mean:
        raise FailClosedError("sealed Cell-D equal-session mean drift")
    return tuple(rows)


def assert_cell_d_parity(actual: ModeEvidence, sealed_table: Mapping[str, Any]) -> None:
    """Block TF-SR interpretation unless the live Cell-D replay is exact."""
    if actual.system != "cell_d" or actual.mode != "aligned":
        raise FailClosedError("Cell-D parity requires aligned Cell-D evidence")
    sealed = _sealed_d_surface(sealed_table, actual.surface)
    live = tuple((item.session, item.n_windows, item.governing_r2) for item in actual.sessions)
    if live != sealed:
        raise FailClosedError("Cell-D exact per-session/window/R2 parity failure")
    sealed_block = sealed_table["results"]["D_swa"][actual.surface]
    if actual.mean_governing != sealed_block["mean_r2"]:
        raise FailClosedError("Cell-D exact equal-session mean parity failure")


def extract_a2_pooled_tables(a2_receipt: Mapping[str, Any], *, within_sessions: Sequence[str], external_sessions: Sequence[str]) -> dict[str, dict[str, float]]:
    """Read sealed A2 pooled tables only; do not reopen any A2 checkpoint."""
    pooled = a2_receipt.get("pooled_per_session")
    if not isinstance(pooled, Mapping):
        raise FailClosedError("A2 receipt pooled table missing")
    result: dict[str, dict[str, float]] = {}
    for surface, expected_sessions in (("within", within_sessions), ("external", external_sessions)):
        key = f"A2_t4_pooled_{surface}"
        values = pooled.get(key)
        if not isinstance(values, Mapping) or set(values) != set(expected_sessions):
            raise FailClosedError("A2 pooled session table roster drift")
        table = {name: _finite_float(values[name]) for name in sorted(expected_sessions)}
        # Force the exact estimator convention now.  The resulting mean is
        # stored in the result even though it never participates in a gate.
        result[surface] = table
    return result


def contextual_a2_contrast(tfsr: ModeEvidence, a2_scores: Mapping[str, float]) -> dict[str, object]:
    """Non-gating TF-SR seed42 minus A2 three-seed pooled paired statistic."""
    tfsr_values = {item.session: item.governing_r2 for item in tfsr.sessions}
    sessions = tuple(sorted(tfsr_values))
    deltas = [tfsr_values[session] - a2_scores[session] for session in sessions]
    payload = paired_statistics(deltas)
    payload["label"] = "seed42-versus-A2-three-seed-pooled"
    payload["non_gating"] = True
    payload["sessions_sorted"] = list(sessions)
    payload["a2_equal_session_mean"] = paired_statistics(tuple(a2_scores[session] for session in sessions))["mean"]
    return payload


@dataclass(frozen=True)
class TrainingEvidence:
    """Read-only result of exact Phase-D terminal/SWA validation."""

    terminal_sha256: str
    swa_sha256: str
    swa_state_digest: str
    closure: Mapping[str, object]
    checkpoint_sha256: Mapping[str, str]
    training_peak_memory_bytes: int = 0

    def __post_init__(self) -> None:
        _require_sha(self.terminal_sha256, "training terminal SHA")
        _require_sha(self.swa_sha256, "training SWA SHA")
        _require_sha(self.swa_state_digest, "training SWA state digest")
        if (not isinstance(self.closure, Mapping) or not isinstance(self.checkpoint_sha256, Mapping)
                or type(self.training_peak_memory_bytes) is not int or self.training_peak_memory_bytes < 0):
            raise ValueError("training evidence closure/checkpoints drift")
        if set(self.checkpoint_sha256) != {"44", "45", "46", "47"} or not all(_is_sha(item) for item in self.checkpoint_sha256.values()):
            raise ValueError("training evidence final-four checkpoint binding drift")

    def payload(self) -> dict[str, object]:
        return {
            "terminal_sha256": self.terminal_sha256, "swa_sha256": self.swa_sha256,
            "swa_state_digest": self.swa_state_digest,
            "closure": json.loads(json.dumps(self.closure, sort_keys=True)),
            "checkpoint_sha256": dict(self.checkpoint_sha256),
            "training_peak_memory_bytes": self.training_peak_memory_bytes,
        }


class TrainingTerminalValidator(Protocol):
    def __call__(self, root: Path) -> TrainingEvidence: ...


FROZEN_SCORE_DEVICE = {
    "cuda_visible_devices": "1",
    "internal_device": "cuda:0",
    "torch_version": "2.5.1.post303",
    "uuid": "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
    "bdf": "00000000:03:00.0",
    "name": "NVIDIA GeForce RTX 3090",
    # These are distinct physical authorities.  The nominal nvidia-smi
    # value must never be derived from, compared with, or rounded from the
    # Torch allocator-visible byte count.
    "nvidia_smi_memory_total_mib": 24576,
    "torch_total_memory_bytes": 25438126080,
}
SOURCE_T4_MEAN = (0.04627712443470955, 0.4544036388397217, 1.3432163000106812, 10.150517463684082)
SOURCE_T4_STD = (1.126278281211853, 1.284820556640625, 1.2352101802825928, 9.115250587463379)
# These are the exact Python-float renderings of the sealed strict-27 source
# behavior cache's float32 statistics.  The cache itself is intentionally not
# a Phase-E runtime input: its mutable pathname is neither opened nor trusted
# during scoring.  Instead, these reviewed numerics are closure-bound here and
# are required in the target-free preflight alongside the sealed semantic SHA.
SOURCE_BEHAVIOR_MEAN = (-0.001148765324614942, 0.002653369214385748)
SOURCE_BEHAVIOR_STD = (8.63547420501709, 8.086690902709961)


class RootPublicationCapability:
    """Opaque root-only token for authority-pair publication interfaces."""

    __slots__ = ("_seal",)

    def __init__(self, seal: object) -> None:
        if seal is not _ROOT_PUBLICATION_SEAL:
            raise TypeError("root publication capability may only be issued internally")
        self._seal = seal


class ExecutionCapability:
    """Opaque proof that an exact preflight/root pair was reloaded."""

    __slots__ = ("authorization", "_seal")

    def __init__(self, authorization: "PhaseEAuthorization", seal: object) -> None:
        if seal is not _EXECUTION_SEAL:
            raise TypeError("execution capability may only be issued after authorization verification")
        self.authorization = authorization
        self._seal = seal


_ROOT_PUBLICATION_SEAL = object()
_EXECUTION_SEAL = object()


def issue_root_publication_capability() -> RootPublicationCapability:
    """Return the explicit root-mint capability; callers are role-audited.

    This function has no data/GPU side effect.  Keeping authority publication
    behind a distinct capability prevents the score core from silently minting
    its own permission, while still letting root use the same implementation
    against a reviewed canonical authority root.
    """
    return RootPublicationCapability(_ROOT_PUBLICATION_SEAL)


def _issue_execution_capability(authorization: "PhaseEAuthorization") -> ExecutionCapability:
    return ExecutionCapability(authorization, _EXECUTION_SEAL)


def _require_execution_capability(capability: object, identity: ScoreIdentity) -> ExecutionCapability:
    if not isinstance(capability, ExecutionCapability) or capability._seal is not _EXECUTION_SEAL:
        raise FailClosedError("score core requires verified explicit execution capability")
    if capability.authorization.payload() != identity.phase_e_authorization:
        raise FailClosedError("execution capability/score identity mismatch")
    return capability


def canonical_authority_parent(root: Path) -> tuple[Path, str]:
    relative = _safe_relative(AUTHORITY_ROOT_RELATIVE)
    path = root.absolute() / relative
    return path.parent, path.name


def reserve_authority_artifact(root: Path, capability: RootPublicationCapability) -> ArtifactRoot:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_PUBLICATION_SEAL:
        raise FailClosedError("only root publication capability may reserve authority root")
    parent, name = canonical_authority_parent(root)
    return reserve_artifact_root(parent, name, AUTHORITY_TOPOLOGY)


def _finite_number_list(value: object, *, length: int, positive: bool = False, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise FailClosedError(f"{label} length drift")
    result: list[float] = []
    for item in value:
        number = _finite_float(item)
        if positive and number <= 0.0:
            raise FailClosedError(f"{label} requires positive values")
        result.append(number)
    return result


def _validate_device_contract(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(FROZEN_SCORE_DEVICE):
        raise FailClosedError("Phase-E device contract schema drift")
    if dict(value) != FROZEN_SCORE_DEVICE:
        raise FailClosedError("Phase-E device contract differs from sealed Cell-D parity device")
    return dict(value)


def _attest_physical_score_runtime(
    *,
    torch: Any,
    torchmetrics: Any,
    nvidia_smi_query: Callable[[], Sequence[str]],
) -> None:
    """Require the two literal GPU-memory authorities independently.

    This intentionally runs before importing either decoder, the NWB adapter,
    or a model builder in :meth:`PhysicalMatchedScoreBackend._load_runtime`.
    ``torch.cuda.get_device_properties(...).total_memory`` is an exact byte
    authority; no MiB floor, rounding, or conversion is permitted.  The
    separately queried nvidia-smi nominal field is exact in MiB.
    """
    if str(torch.__version__) != FROZEN_SCORE_DEVICE["torch_version"]:
        raise FailClosedError("physical Torch version differs from sealed Cell-D parity runtime")
    if str(torchmetrics.__version__) != "1.5.1":
        raise FailClosedError("physical TorchMetrics version must be exactly 1.5.1")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != FROZEN_SCORE_DEVICE["cuda_visible_devices"]
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise FailClosedError("physical score device must expose only CUDA_VISIBLE_DEVICES=1 as cuda:0")
    if torch.cuda.current_device() != 0:
        raise FailClosedError("physical score logical CUDA device is not cuda:0")
    properties = torch.cuda.get_device_properties(0)
    torch_total_memory = getattr(properties, "total_memory", None)
    if (properties.name != FROZEN_SCORE_DEVICE["name"]
            or type(torch_total_memory) is not int
            or torch_total_memory != FROZEN_SCORE_DEVICE["torch_total_memory_bytes"]):
        raise FailClosedError("physical score Torch device name/byte-memory drift")
    try:
        response = tuple(nvidia_smi_query())
    except Exception as error:
        raise FailClosedError("cannot attest fixed physical score GPU") from error
    if len(response) != 1 or not isinstance(response[0], str):
        raise FailClosedError("physical score GPU query cardinality drift")
    try:
        uuid, bdf, name, memory_mib_text = (part.strip() for part in response[0].split(",", 3))
        nvidia_smi_memory_mib = int(memory_mib_text)
    except (TypeError, ValueError) as error:
        raise FailClosedError("physical score GPU query schema drift") from error
    if (uuid != FROZEN_SCORE_DEVICE["uuid"] or bdf != FROZEN_SCORE_DEVICE["bdf"]
            or name != FROZEN_SCORE_DEVICE["name"]
            or nvidia_smi_memory_mib != FROZEN_SCORE_DEVICE["nvidia_smi_memory_total_mib"]):
        raise FailClosedError("physical score GPU UUID/BDF/name/nominal-memory drift")


def _validate_phase_e_closure_payload(value: object) -> dict[str, object]:
    """Validate the explicit (never globbed) Phase-E closure payload.

    A syntactically valid 64-hex closure field is not enough: the digest must
    commit to exactly the declared ordered runtime path map.  The later live
    recheck still recomputes this map from descriptor-safe files.
    """
    required = {"paths", "sha256_by_path", "immutable_input_bindings", "closure_sha256"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise FailClosedError("Phase-E closure schema drift")
    paths = value.get("paths")
    hashes = value.get("sha256_by_path")
    if paths != list(PHASE_E_CLOSURE) or not isinstance(hashes, Mapping) or set(hashes) != set(PHASE_E_CLOSURE):
        raise FailClosedError("Phase-E closure path map drift")
    if any(not _is_sha(hashes.get(path)) for path in PHASE_E_CLOSURE):
        raise FailClosedError("Phase-E closure file SHA drift")
    immutable_inputs = value.get("immutable_input_bindings")
    expected_inputs = {"within_paired_view_manifest": WITHIN_PAIRED_VIEW_CLOSURE_BINDING}
    if immutable_inputs != expected_inputs:
        raise FailClosedError("Phase-E closure immutable input binding drift")
    encoded = json.dumps(
        {"files": dict(hashes), "immutable_input_bindings": expected_inputs},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if value.get("closure_sha256") != _sha(encoded):
        raise FailClosedError("Phase-E closure aggregate SHA drift")
    return {
        "paths": list(PHASE_E_CLOSURE),
        "sha256_by_path": {path: str(hashes[path]) for path in PHASE_E_CLOSURE},
        "immutable_input_bindings": {"within_paired_view_manifest": dict(WITHIN_PAIRED_VIEW_CLOSURE_BINDING)},
        "closure_sha256": str(value["closure_sha256"]),
    }


def sealed_within_assets() -> tuple[dict[str, object], ...]:
    """Return the only legal within-6 asset rows.

    This table is deliberately literal and independent of every caller.  The
    target-free builder separately descriptor-reads the paired-view manifest
    and proves that it still yields these same rows before it can create a
    durable preflight.
    """
    return tuple(
        {
            "asset_id": f"c1_paired_view:{session}",
            "session": session,
            "frozen_path": frozen_path,
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
        for session, frozen_path, expected_bytes, expected_sha256 in SEALED_WITHIN_PAIRED_VIEW_ROWS
    )


@dataclass(frozen=True)
class WithinPairedViewManifestBinding:
    """Descriptor-read identity for the mode-0600/no-sidecar C1 manifest."""

    relative_path: str
    body_sha256: str
    mode: str
    descriptor_identity: tuple[int, int, int]
    selected_split: str
    read_once: bool

    def __post_init__(self) -> None:
        if (
            self.relative_path != WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE
            or self.body_sha256 != WITHIN_PAIRED_VIEW_MANIFEST_SHA256
            or self.mode != format(WITHIN_PAIRED_VIEW_MANIFEST_MODE, "04o")
            or self.selected_split != "val"
            or self.read_once is not True
            or not isinstance(self.descriptor_identity, tuple)
            or len(self.descriptor_identity) != 3
            or any(type(item) is not int or item < 0 for item in self.descriptor_identity)
        ):
            raise ValueError("within paired-view manifest binding drift")

    def payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "body_sha256": self.body_sha256,
            "mode": self.mode,
            "descriptor_identity": list(self.descriptor_identity),
            "selected_split": self.selected_split,
            "read_once": self.read_once,
        }


def _within_manifest_binding_from_payload(value: object) -> WithinPairedViewManifestBinding:
    expected = {"relative_path", "body_sha256", "mode", "descriptor_identity", "selected_split", "read_once"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FailClosedError("within paired-view manifest binding schema drift")
    try:
        binding = WithinPairedViewManifestBinding(
            relative_path=value["relative_path"],
            body_sha256=value["body_sha256"],
            mode=value["mode"],
            descriptor_identity=tuple(value["descriptor_identity"]),
            selected_split=value["selected_split"],
            read_once=value["read_once"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FailClosedError("within paired-view manifest binding nested drift") from error
    if binding.payload() != dict(value):
        raise FailClosedError("within paired-view manifest binding roundtrip drift")
    return binding


def _expected_paired_view_val_rows() -> list[dict[str, object]]:
    return [
        {
            "session": session,
            "filename": Path(frozen_path).name,
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
        for session, frozen_path, expected_bytes, expected_sha256 in SEALED_WITHIN_PAIRED_VIEW_ROWS
    ]


def canonical_within_assets_from_paired_view_manifest(
    root: Path,
    *,
    within_roster: Sequence[str],
) -> tuple[tuple[dict[str, object], ...], WithinPairedViewManifestBinding]:
    """Descriptor-read the sole allowed within-6 asset authority.

    This intentionally reads only the sealed metadata manifest, never a
    within NWB.  Its fixed SHA/mode, selected `val` split, and exact lstat/open
    identity become durable preflight facts; later lifecycle validation also
    replays this reader before it can resolve a data pathname.
    """
    expected_sessions = tuple(row[0] for row in SEALED_WITHIN_PAIRED_VIEW_ROWS)
    if tuple(within_roster) != expected_sessions:
        raise FailClosedError("within paired-view manifest/strict roster drift")
    body, identity = _canonical_regular_bytes(
        root,
        WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE,
        expected_mode=WITHIN_PAIRED_VIEW_MANIFEST_MODE,
    )
    if _sha(body) != WITHIN_PAIRED_VIEW_MANIFEST_SHA256:
        raise FailClosedError("within paired-view manifest SHA drift")
    try:
        manifest = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("within paired-view manifest JSON drift") from error
    if not isinstance(manifest, Mapping):
        raise FailClosedError("within paired-view manifest root drift")
    splits = manifest.get("session_splits")
    inventory = manifest.get("file_inventory")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("task") != "CO"
        or manifest.get("file_count") != 33
        or manifest.get("split_counts") != [27, 6, 6]
        or manifest.get("max_units_exclusive") != 100
        or manifest.get("source_manifest_sha256") != STRICT_MANIFEST_SHA256
        or manifest.get("formal_test_file_hashes") != []
        or manifest.get("formal_test_file_paths") != []
        or manifest.get("formal_test_paths_resolved") is not False
        or not isinstance(splits, Mapping)
        or set(splits) != {"train", "val", "test"}
        or not isinstance(inventory, Mapping)
        or set(inventory) != {"train", "val"}
        or not isinstance(splits.get("train"), list)
        or len(splits["train"]) != 27
        or tuple(splits.get("val", ())) != expected_sessions
        or tuple(splits.get("test", ())) != FORMAL_TEST_SESSION_NAMES
        or not isinstance(inventory.get("train"), list)
        or len(inventory["train"]) != 27
        or inventory.get("val") != _expected_paired_view_val_rows()
    ):
        raise FailClosedError("within paired-view manifest semantic row drift")
    return sealed_within_assets(), WithinPairedViewManifestBinding(
        relative_path=WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE,
        body_sha256=WITHIN_PAIRED_VIEW_MANIFEST_SHA256,
        mode=format(WITHIN_PAIRED_VIEW_MANIFEST_MODE, "04o"),
        descriptor_identity=identity,
        selected_split="val",
        read_once=True,
    )


def _validate_preflight_within_assets(value: object, within_roster: Sequence[str]) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or len(value) != PUBLIC_SPEC.within_count:
        raise FailClosedError("target-free preflight within asset cardinality drift")
    expected_keys = {"asset_id", "session", "frozen_path", "bytes", "sha256"}
    rows: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise FailClosedError("target-free preflight within asset schema drift")
        session = row.get("session")
        path = row.get("frozen_path")
        if (not isinstance(session, str) or not isinstance(path, str)
                or session in FORMAL_TEST_SESSION_NAMES or type(row.get("bytes")) is not int
                or int(row["bytes"]) <= 0 or not _is_sha(row.get("sha256"))
                or not isinstance(row.get("asset_id"), str) or not row["asset_id"]):
            raise FailClosedError("target-free preflight within asset value drift")
        _safe_relative(path)
        if Path(path).name != f"{session}_behavior+ecephys.nwb":
            raise FailClosedError("target-free preflight within asset basename/session drift")
        rows.append({key: row[key] for key in sorted(expected_keys)})
    rows = sorted(rows, key=lambda row: str(row["session"]))
    expected = sealed_within_assets()
    expected_sessions = tuple(str(row["session"]) for row in expected)
    if tuple(within_roster) != expected_sessions or tuple(rows) != expected:
        raise FailClosedError("target-free preflight within paired-view asset authority drift")
    return tuple(rows)


def _validate_preflight_normalizers(value: object) -> dict[str, object]:
    expected = {
        "source_normalizer_body_sha256", "t4_semantic_sha256", "behavior_semantic_sha256",
        "t4_mean", "t4_std", "behavior_mean", "behavior_std",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FailClosedError("target-free preflight normalizer schema drift")
    if (value.get("source_normalizer_body_sha256") != SOURCE_NORMALIZER_BODY_SHA
            or value.get("t4_semantic_sha256") != T4_NORMALIZER_SEMANTIC_SHA
            or value.get("behavior_semantic_sha256") != BEHAVIOR_NORMALIZER_SEMANTIC_SHA):
        raise FailClosedError("target-free preflight normalizer authority drift")
    t4_mean = _finite_number_list(value.get("t4_mean"), length=4, label="T4 mean")
    t4_std = _finite_number_list(value.get("t4_std"), length=4, positive=True, label="T4 std")
    if tuple(t4_mean) != SOURCE_T4_MEAN or tuple(t4_std) != SOURCE_T4_STD:
        raise FailClosedError("target-free preflight T4 source normalizer numeric drift")
    behavior_mean = _finite_number_list(value.get("behavior_mean"), length=2, label="behavior mean")
    behavior_std = _finite_number_list(value.get("behavior_std"), length=2, positive=True, label="behavior std")
    if tuple(behavior_mean) != SOURCE_BEHAVIOR_MEAN or tuple(behavior_std) != SOURCE_BEHAVIOR_STD:
        raise FailClosedError("target-free preflight behavior source normalizer numeric drift")
    return {
        "source_normalizer_body_sha256": SOURCE_NORMALIZER_BODY_SHA,
        "t4_semantic_sha256": T4_NORMALIZER_SEMANTIC_SHA,
        "behavior_semantic_sha256": BEHAVIOR_NORMALIZER_SEMANTIC_SHA,
        "t4_mean": t4_mean, "t4_std": t4_std,
        "behavior_mean": behavior_mean, "behavior_std": behavior_std,
    }


def build_target_free_preflight(
    *,
    root: Path,
    training: TrainingEvidence,
    fixed_authorities: Mapping[str, Mapping[str, object]],
    closure: Mapping[str, object],
    within_roster: Sequence[str],
    normalizers: Mapping[str, object],
    device_contract: Mapping[str, object] = FROZEN_SCORE_DEVICE,
) -> dict[str, object]:
    """Build (but never publish) the exact target-free Phase-E preflight.

    Within assets are intentionally absent from this public signature.  The
    only accepted six rows are descriptor-derived from the frozen C1 paired
    view, not supplied by the caller that happens to request a preflight.
    """
    if not isinstance(fixed_authorities, Mapping) or set(fixed_authorities) != {item.name for item in FIXED_AUTHORITIES}:
        raise FailClosedError("target-free preflight fixed authority set drift")
    for key, binding in fixed_authorities.items():
        if not isinstance(binding, Mapping) or binding.get("body_sha256") != next(item.sha256 for item in FIXED_AUTHORITIES if item.name == key):
            raise FailClosedError("target-free preflight fixed authority binding drift")
    checked_closure = _validate_phase_e_closure_payload(closure)
    roster = tuple(sorted(within_roster))
    if len(roster) != PUBLIC_SPEC.within_count or len(set(roster)) != len(roster):
        raise FailClosedError("target-free preflight within roster drift")
    assets, paired_view_manifest = canonical_within_assets_from_paired_view_manifest(
        root,
        within_roster=roster,
    )
    assets = _validate_preflight_within_assets(list(assets), roster)
    checked_normalizers = _validate_preflight_normalizers(normalizers)
    return {
        "schema": "tfsr_phase_e_target_free_preflight_v2",
        "status": "PREFLIGHT_ACCEPTED",
        "cell": CELL,
        "phase": PHASE,
        "score_spec": PUBLIC_SPEC.payload(),
        "training": training.payload(),
        "fixed_authorities": json.loads(json.dumps(fixed_authorities, sort_keys=True)),
        "phase_e_closure": checked_closure,
        "device_contract": _validate_device_contract(device_contract),
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": SCORE_ROOT_RELATIVE,
        "within_roster": list(roster),
        "within_assets": [dict(item) for item in assets],
        "within_paired_view_manifest": paired_view_manifest.payload(),
        "normalizers": checked_normalizers,
        "target_free": True,
        "formal_sessions_inert": list(FORMAL_TEST_SESSION_NAMES),
    }


def validate_target_free_preflight(
    value: Mapping[str, Any], *, training: TrainingEvidence, fixed_authorities: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    expected = {
        "schema", "status", "cell", "phase", "score_spec", "training", "fixed_authorities",
        "phase_e_closure", "device_contract", "authority_root_relative", "score_root_relative",
        "within_roster", "within_assets", "within_paired_view_manifest", "normalizers", "target_free", "formal_sessions_inert",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_phase_e_target_free_preflight_v2"
            or value.get("status") != "PREFLIGHT_ACCEPTED" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("score_spec") != PUBLIC_SPEC.payload()
            or value.get("training") != training.payload()
            or value.get("fixed_authorities") != json.loads(json.dumps(fixed_authorities, sort_keys=True))
            or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE
            or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
            or value.get("target_free") is not True
            or value.get("formal_sessions_inert") != list(FORMAL_TEST_SESSION_NAMES)):
        raise FailClosedError("target-free preflight schema/binding drift")
    roster = _require_string_sequence(value.get("within_roster"), "target-free preflight within roster")
    if len(roster) != PUBLIC_SPEC.within_count:
        raise FailClosedError("target-free preflight within roster count drift")
    _validate_preflight_within_assets(value.get("within_assets"), roster)
    _within_manifest_binding_from_payload(value.get("within_paired_view_manifest"))
    _validate_preflight_normalizers(value.get("normalizers"))
    _validate_phase_e_closure_payload(value.get("phase_e_closure"))
    _validate_device_contract(value.get("device_contract"))
    return dict(value)


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    """Build the separate root decision, binding the exact preflight bytes."""
    _require_sha(official_preflight_sha256, "official preflight SHA")
    closure = preflight.get("phase_e_closure") if isinstance(preflight, Mapping) else None
    device = preflight.get("device_contract") if isinstance(preflight, Mapping) else None
    if not isinstance(closure, Mapping):
        raise FailClosedError("root authorization requires validated preflight closure")
    return {
        "schema": "tfsr_phase_e_root_authorization_v2",
        "status": "ROOT_AUTHORIZED",
        "cell": CELL,
        "phase": PHASE,
        "official_preflight_sha256": official_preflight_sha256,
        "score_root_relative": SCORE_ROOT_RELATIVE,
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "phase_e_closure": json.loads(json.dumps(closure, sort_keys=True)),
        "device_contract": _validate_device_contract(device),
        "target_free_preflight_required": True,
    }


def validate_root_authorization(
    value: Mapping[str, Any],
    *,
    official_preflight_sha256: str,
    preflight: Mapping[str, Any],
) -> dict[str, object]:
    """Validate the root-only decision independently of its filesystem pair.

    Keeping this schema check separate from ``verify_phase_e_authorization``
    lets the root minting workflow validate the exact object *before* it is
    made immutable, while the live route repeats the same check after a
    descriptor-safe reload.  It is deliberately incapable of issuing an
    execution capability by itself.
    """
    _require_sha(official_preflight_sha256, "official preflight SHA")
    expected = {
        "schema", "status", "cell", "phase", "official_preflight_sha256", "score_root_relative",
        "authority_root_relative", "phase_e_closure", "device_contract", "target_free_preflight_required",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_phase_e_root_authorization_v2"
            or value.get("status") != "ROOT_AUTHORIZED" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("official_preflight_sha256") != official_preflight_sha256
            or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
            or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE
            or value.get("target_free_preflight_required") is not True
            or value.get("phase_e_closure") != preflight.get("phase_e_closure")
            or value.get("device_contract") != preflight.get("device_contract")):
        raise FailClosedError("Phase-E root authorization binding drift")
    _validate_device_contract(value.get("device_contract"))
    return dict(value)


def publish_target_free_preflight(
    artifact: ArtifactRoot,
    capability: RootPublicationCapability,
    payload: Mapping[str, object],
    *,
    training: TrainingEvidence,
    fixed_authorities: Mapping[str, Mapping[str, object]],
) -> str:
    """Validate the reviewed preflight before immutable pair publication.

    A root capability only authorizes publication; it does not make arbitrary
    JSON scientifically valid.  Validate the exact candidate against the
    reviewed Phase-D terminal and fixed authority bindings before the first
    body/sidecar inode can exist.
    """
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_PUBLICATION_SEAL:
        raise FailClosedError("only root may publish target-free preflight")
    artifact._check_name("official_preflight.json")
    if artifact.has_name("root_authorization.json"):
        raise FailClosedError("cannot publish preflight after root authorization")
    checked = validate_target_free_preflight(
        payload, training=training, fixed_authorities=fixed_authorities,
    )
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization(
    artifact: ArtifactRoot,
    capability: RootPublicationCapability,
    payload: Mapping[str, object],
    *,
    training: TrainingEvidence,
    fixed_authorities: Mapping[str, Mapping[str, object]],
) -> str:
    """Validate both durable preflight and root decision before publication."""
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_PUBLICATION_SEAL:
        raise FailClosedError("only root may publish root authorization")
    if not artifact.has_name("official_preflight.json"):
        raise FailClosedError("root authorization requires durable preflight")
    # ``reload_pair`` is descriptor-safe and validates the immutable sidecar.
    # Derive the expected binding from those exact durable bytes rather than a
    # caller-provided SHA or an in-memory preflight object.
    preflight_body = artifact.reload_pair("official_preflight.json")
    actual_preflight_sha256 = _sha(preflight_body)
    try:
        preflight = json.loads(preflight_body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("durable target-free preflight JSON malformed") from error
    if not isinstance(preflight, Mapping):
        raise FailClosedError("durable target-free preflight JSON root drift")
    checked_preflight = validate_target_free_preflight(
        preflight, training=training, fixed_authorities=fixed_authorities,
    )
    checked_authorization = validate_root_authorization(
        payload,
        official_preflight_sha256=actual_preflight_sha256,
        preflight=checked_preflight,
    )
    return artifact.publish_json("root_authorization.json", checked_authorization)


def _attach_readonly_phase_d_artifact(root: Path, train_module: Any) -> Any:
    """Construct Phase-D's own inode-capability without reserving/writing it."""
    directory = root.absolute() / PHASE_D_TRAIN_ROOT_RELATIVE
    if not directory.exists() or directory.is_symlink():
        raise FailClosedError("canonical Phase-D training root absent")
    parent = directory.parent
    try:
        identity = train_module._directory_identity(directory)
        parent_identity = train_module._directory_identity(parent)
    except Exception as error:  # convert private implementation failure to boundary failure
        raise FailClosedError("canonical Phase-D artifact root invalid") from error
    return train_module.ArtifactRoot(directory, train_module.PUBLIC_SPEC.topology, identity, parent, parent_identity)


def validate_phase_d_training_terminal(root: Path) -> TrainingEvidence:
    """Strictly reload the complete canonical Phase-D terminal/SWA chain.

    This function is reachable only through the explicit execution route.  Its
    imports are therefore intentionally lazy: a static Phase-E plan cannot
    import Torch or construct a model.  It neither opens an evaluation asset
    nor writes a result.
    """
    from . import train
    from .model import TFSRDecoder

    artifact = _attach_readonly_phase_d_artifact(root, train)
    identity = train.production_identity(root)
    terminal_body = artifact.reload_pair("terminal.json")
    terminal_sha = _sha(terminal_body)
    try:
        terminal = json.loads(terminal_body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("Phase-D terminal JSON drift") from error
    if not isinstance(terminal, Mapping):
        raise FailClosedError("Phase-D terminal root drift")
    train.validate_terminal_receipt(terminal, train.PUBLIC_SPEC, identity)
    if terminal.get("launch_closure") != terminal.get("final_closure"):
        raise FailClosedError("Phase-D launch/final closure drift")
    # Reload and validate every receipt before touching the final-four states.
    attempt = artifact.reload_json("attempt.json", terminal["attempt_sha256"])
    train.validate_attempt_receipt(attempt, train.PUBLIC_SPEC, identity)
    launch = artifact.reload_json("launch.json", terminal["launch_sha256"])
    train.validate_launch_receipt(launch, train.PUBLIC_SPEC, identity)
    throughput_name = f"throughput{train.PUBLIC_SPEC.throughput_probe_steps}.json"
    throughput = artifact.reload_json(throughput_name, terminal["throughput_sha256"])
    train.validate_throughput_receipt(throughput, train.PUBLIC_SPEC, identity)
    epoch_shas = terminal.get("epoch_receipt_sha256")
    if not isinstance(epoch_shas, list) or len(epoch_shas) != train.PUBLIC_SPEC.epochs:
        raise FailClosedError("Phase-D terminal epoch SHA list drift")
    training_peak_memory_bytes = 0
    for epoch, digest in enumerate(epoch_shas):
        value = artifact.reload_json(f"epoch-{epoch:02d}.json", digest)
        train.validate_epoch_receipt(
            value,
            epoch,
            (epoch + 1) * train.PUBLIC_SPEC.steps_per_epoch,
            train.PUBLIC_SPEC,
            identity,
        )
        resources = value.get("resources")
        if isinstance(resources, Mapping) and type(resources.get("peak_allocated_bytes")) is int:
            training_peak_memory_bytes = max(training_peak_memory_bytes, int(resources["peak_allocated_bytes"]))
    # This is the exact checkpoint binding hardened in Phase-D: the SWA source
    # checkpoints may not merely contain plausible hashes; all must bind this
    # terminal's launch receipt and immutable closure map exactly.
    expected_binding = {
        "cell": train.CELL,
        "run_spec": train.PUBLIC_SPEC.payload(),
        "launch_sha256": terminal["launch_sha256"],
        "launch_closure": terminal["launch_closure"],
        "predecessor": terminal["predecessor"],
    }
    backend = train.TorchTrainingBackend(root)
    checkpoint_shas = terminal.get("checkpoint_sha256")
    if not isinstance(checkpoint_shas, Mapping) or set(checkpoint_shas) != {"44", "45", "46", "47"}:
        raise FailClosedError("Phase-D checkpoint map drift")
    for epoch in train.PUBLIC_SPEC.checkpoint_epochs:
        body = artifact.reload_pair(f"checkpoint-{epoch:02d}.pt", checkpoint_shas[str(epoch)])
        backend.validate_checkpoint(body, epoch, (epoch + 1) * train.PUBLIC_SPEC.steps_per_epoch,
                                    train.PUBLIC_SPEC, expected_binding=expected_binding)
    swa_body = artifact.reload_pair("swa.pt", terminal["swa_sha256"])
    swa = backend.validate_swa(swa_body, train.PUBLIC_SPEC, expected_binding=expected_binding)
    if swa.get("state_digest") != terminal.get("swa_state_digest"):
        raise FailClosedError("Phase-D SWA state digest differs from terminal")
    # Reperform the meaningful portion of strict architecture binding rather
    # than trusting a historical receipt alone.  No forward/data is needed.
    fresh = TFSRDecoder(capture_diagnostics=False)
    fresh.load_state_dict(swa["state"], strict=True)
    if fresh.capture_diagnostics is not False:
        raise FailClosedError("Phase-D SWA strict fresh model diagnostics drift")
    return TrainingEvidence(
        terminal_sha256=terminal_sha, swa_sha256=str(terminal["swa_sha256"]),
        swa_state_digest=str(terminal["swa_state_digest"]), closure=terminal["launch_closure"],
        checkpoint_sha256={str(key): str(value) for key, value in checkpoint_shas.items()},
        training_peak_memory_bytes=training_peak_memory_bytes,
    )


@dataclass(frozen=True)
class PhaseEAuthorization:
    preflight_sha256: str
    root_authorization_sha256: str
    preflight: Mapping[str, Any]
    root_authorization: Mapping[str, Any]
    expected_closure: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_sha(self.preflight_sha256, "Phase-E preflight SHA")
        _require_sha(self.root_authorization_sha256, "Phase-E root authorization SHA")
        if not all(isinstance(item, Mapping) for item in (self.preflight, self.root_authorization, self.expected_closure)):
            raise ValueError("Phase-E authorization maps required")

    def payload(self) -> dict[str, object]:
        return {
            "preflight_sha256": self.preflight_sha256,
            "root_authorization_sha256": self.root_authorization_sha256,
            "expected_closure": json.loads(json.dumps(self.expected_closure, sort_keys=True)),
        }


def _read_authority_pair(directory: Path, name: str) -> tuple[Mapping[str, Any], str]:
    """Read an immutable authority pair under a canonical non-symlink directory."""
    if not isinstance(name, str) or not name.endswith(".json") or "/" in name:
        raise ValueError("authority pair requires one .json leaf")
    parent = directory.absolute()
    try:
        info = os.lstat(parent)
    except OSError as error:
        raise FailClosedError("Phase-E authority root absent") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise FailClosedError("Phase-E authority root is not canonical directory")
    # Reuse the same descriptor-safe reader by treating the authority directory
    # as the root and the leaf as a strict relative path.
    body, _ = _canonical_regular_bytes(parent, name, expected_mode=0o444)
    digest = _sha(body)
    sidecar, _ = _canonical_regular_bytes(parent, name + ".sha256", expected_mode=0o444)
    if sidecar != f"{digest}  {name}\n".encode("ascii"):
        raise FailClosedError("Phase-E authority sidecar drift")
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("Phase-E authority JSON malformed") from error
    if not isinstance(value, Mapping):
        raise FailClosedError("Phase-E authority JSON root drift")
    return value, digest


def _closure_matches_authorization(root: Path, expected: Mapping[str, Any]) -> Mapping[str, object]:
    observed = phase_e_closure(root)
    checked = _validate_phase_e_closure_payload(expected)
    if observed != checked:
        raise FailClosedError("Phase-E launch/final implementation closure drift")
    return observed


def verify_phase_e_authorization(
    root: Path,
    training: TrainingEvidence,
    fixed_authorities: Mapping[str, AuthorityMaterial],
) -> PhaseEAuthorization:
    """Verify target-free preflight/root authorization before score-root reserve."""
    directory = root.absolute() / AUTHORITY_ROOT_RELATIVE
    preflight, preflight_sha = _read_authority_pair(directory, "official_preflight.json")
    authorization, authorization_sha = _read_authority_pair(directory, "root_authorization.json")
    expected_fixed = {name: material.binding() for name, material in fixed_authorities.items()}
    checked_preflight = validate_target_free_preflight(
        preflight,
        training=training,
        fixed_authorities=expected_fixed,
    )
    # Re-read the no-sidecar paired-view manifest before a live lifecycle can
    # reserve a score root.  This repeats the exact descriptor identity and
    # val[6] row proof embedded in the durable preflight, closing the gap in
    # which a syntactically plausible root-signed mapping could otherwise be
    # replayed after its source authority had changed.
    within_roster = extract_within_roster(fixed_authorities["strict_manifest"].value or {})
    assets, manifest_binding = canonical_within_assets_from_paired_view_manifest(
        root,
        within_roster=within_roster,
    )
    if (
        checked_preflight.get("within_assets") != [dict(item) for item in assets]
        or checked_preflight.get("within_paired_view_manifest") != manifest_binding.payload()
    ):
        raise FailClosedError("Phase-E paired-view within authority changed after preflight")
    validate_root_authorization(
        authorization,
        official_preflight_sha256=preflight_sha,
        preflight=checked_preflight,
    )
    expected_closure = authorization.get("phase_e_closure")
    if not isinstance(expected_closure, Mapping):
        raise FailClosedError("Phase-E root authorization closure missing")
    _validate_device_contract(authorization.get("device_contract"))
    _closure_matches_authorization(root, expected_closure)
    return PhaseEAuthorization(
        preflight_sha256=preflight_sha, root_authorization_sha256=authorization_sha,
        preflight=preflight, root_authorization=authorization, expected_closure=expected_closure,
    )


def _identity_list(value: object, *, length: int, label: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or len(value) != length
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError(f"{label} identity schema drift")
    return tuple(int(item) for item in value)


def _validate_input_asset_payload(value: Mapping[str, object]) -> None:
    """Validate compact target binding facts before they enter a receipt."""
    base = {
        "surface", "asset_id", "session", "canonical_local_basename", "bytes", "sha256", "frozen_path",
    }
    allowed = base | {"data_root", "held_descriptor_identity"}
    if set(value) - allowed or not base.issubset(value):
        raise ValueError("session input asset schema drift")
    if (value.get("surface") not in {"within", "external"}
            or not isinstance(value.get("asset_id"), str) or not value["asset_id"]
            or not isinstance(value.get("session"), str) or not value["session"]
            or not isinstance(value.get("canonical_local_basename"), str) or not value["canonical_local_basename"]
            or type(value.get("bytes")) is not int or int(value["bytes"]) <= 0
            or not _is_sha(value.get("sha256")) or not isinstance(value.get("frozen_path"), str)):
        raise ValueError("session input asset value drift")
    _safe_relative(str(value["frozen_path"]))
    if Path(str(value["frozen_path"])).name != value["canonical_local_basename"]:
        raise ValueError("session input asset canonical basename drift")
    root = value.get("data_root")
    if root is not None:
        if (not isinstance(root, Mapping) or set(root) != {"directory_basename", "directory_identity", "parent_identity"}
                or not isinstance(root.get("directory_basename"), str) or not root["directory_basename"]):
            raise ValueError("session input data-root payload drift")
        _identity_list(root.get("directory_identity"), length=2, label="data-root directory")
        _identity_list(root.get("parent_identity"), length=2, label="data-root parent")
    held = value.get("held_descriptor_identity")
    if held is not None:
        _identity_list(held, length=3, label="held descriptor")


@dataclass(frozen=True)
class SessionInputAuthority:
    """Compact per-session facts published before any model forward."""

    session: str
    surface: str
    asset: Mapping[str, object]
    ordered_unit_digest: str
    unit_count: int
    raw_t4: Mapping[str, object]
    normalized_t4: Mapping[str, object]
    calibration_sha256: str
    query_start_sha256: str
    neural_sha256: str
    behavior_sha256: str
    valid_mask_sha256: str
    target_last_bin_sha256: str
    last_bin_valid_mask_sha256: str
    last_bin_valid_count: int
    n_windows: int
    raw_t4_sua_axis_proof: Mapping[str, object]
    raw_t4_sua_axis_proof_sha256: str

    def __post_init__(self) -> None:
        if self.surface not in {"within", "external"} or not isinstance(self.session, str) or not self.session:
            raise ValueError("session input surface/name drift")
        _require_sha(self.ordered_unit_digest, "ordered unit digest")
        if type(self.unit_count) is not int or self.unit_count < 2:
            raise ValueError("session input requires >=2 units for wrong-pair control")
        if (
            type(self.n_windows) is not int
            or self.n_windows <= 0
            or type(self.last_bin_valid_count) is not int
            or self.last_bin_valid_count != self.n_windows
        ):
            raise ValueError("session input governing-query window/count drift")
        if not isinstance(self.asset, Mapping) or not isinstance(self.raw_t4, Mapping) or not isinstance(self.normalized_t4, Mapping):
            raise ValueError("session input asset/T4 bindings must be mappings")
        _validate_input_asset_payload(self.asset)
        for binding in (self.raw_t4, self.normalized_t4):
            if set(binding) != {"dtype", "shape", "bytes_sha256"} or not isinstance(binding["dtype"], str):
                raise ValueError("tensor compact binding schema drift")
            if not isinstance(binding["shape"], list) or not _is_sha(binding["bytes_sha256"]):
                raise ValueError("tensor compact binding value drift")
        if self.raw_t4["shape"] != [self.unit_count, 4] or self.normalized_t4["shape"] != [self.unit_count, 4]:
            raise ValueError("session input T4 tensor/unit axis shape drift")
        for item in (self.calibration_sha256, self.query_start_sha256, self.neural_sha256,
                     self.behavior_sha256, self.valid_mask_sha256, self.target_last_bin_sha256,
                     self.last_bin_valid_mask_sha256, self.raw_t4_sua_axis_proof_sha256):
            _require_sha(item, "session input bytes")
        raw_proof = dict(self.raw_t4_sua_axis_proof) if isinstance(self.raw_t4_sua_axis_proof, Mapping) else None
        expected_raw_proof = {
            "schema", "session", "signal_view", "channel_ids_dtype", "channel_ids_sha256",
            "source_unit_count", "raw_t4_shape", "feature_group", "pool_size", "mapping",
            "closure_bound_functions",
        }
        if (
            raw_proof is None
            or set(raw_proof) != expected_raw_proof
            or raw_proof.get("schema") != "tfsr_phase_e_raw_t4_sua_axis_proof_v1"
            or raw_proof.get("session") != self.session
            or raw_proof.get("signal_view") != "sua"
            or raw_proof.get("channel_ids_dtype") != "int64"
            or raw_proof.get("source_unit_count") != self.unit_count
            or raw_proof.get("raw_t4_shape") != [self.unit_count, 4]
            or raw_proof.get("feature_group") != "t4"
            or raw_proof.get("pool_size") != 30
            or raw_proof.get("mapping")
            != "raw_t4_row_k_equals_sua_neural_column_k_for_contiguous_channel_ids"
            or raw_proof.get("closure_bound_functions")
            != [
                "mc_maze.multisession_datamodule.load_dandi688_session",
                "mc_maze.unit_side_features.compute_unit_side_features_uncached",
            ]
            or not _is_sha(raw_proof.get("channel_ids_sha256"))
            or _sha(_json_bytes(raw_proof)) != self.raw_t4_sua_axis_proof_sha256
        ):
            raise ValueError("session input raw M30 T4/SUA axis proof drift")

    def payload(self) -> dict[str, object]:
        return {
            "session": self.session, "surface": self.surface, "asset": dict(self.asset),
            "ordered_unit_digest": self.ordered_unit_digest, "unit_count": self.unit_count,
            "raw_t4": dict(self.raw_t4), "normalized_t4": dict(self.normalized_t4),
            "calibration_sha256": self.calibration_sha256, "query_start_sha256": self.query_start_sha256,
            "neural_sha256": self.neural_sha256, "behavior_sha256": self.behavior_sha256,
            "valid_mask_sha256": self.valid_mask_sha256, "target_last_bin_sha256": self.target_last_bin_sha256,
            "last_bin_valid_mask_sha256": self.last_bin_valid_mask_sha256,
            "last_bin_valid_count": self.last_bin_valid_count,
            "n_windows": self.n_windows,
            "raw_t4_sua_axis_proof": dict(self.raw_t4_sua_axis_proof),
            "raw_t4_sua_axis_proof_sha256": self.raw_t4_sua_axis_proof_sha256,
        }


@dataclass(frozen=True)
class InputAuthorityEvidence:
    records: tuple[SessionInputAuthority, ...]
    source_normalizer_body_sha256: str
    t4_normalizer_semantic_sha256: str
    behavior_normalizer_semantic_sha256: str
    strict_manifest_sha256: str
    raw_to_normalized_exact: bool
    no_cache_readonly_adapter: bool

    def __post_init__(self) -> None:
        if len(self.records) != PUBLIC_SPEC.within_count + PUBLIC_SPEC.external_count:
            raise ValueError("input authority must bind exact 6+15 sessions")
        if len({(item.surface, item.session) for item in self.records}) != len(self.records):
            raise ValueError("input authority session duplicate")
        if tuple((item.surface, item.session) for item in self.records) != tuple(sorted((item.surface, item.session) for item in self.records)):
            raise ValueError("input authority records must be sorted")
        within = [item.session for item in self.records if item.surface == "within"]
        external = [item.session for item in self.records if item.surface == "external"]
        if len(within) != 6 or len(external) != 15:
            raise ValueError("input authority surface roster drift")
        if self.raw_to_normalized_exact is not True or self.no_cache_readonly_adapter is not True:
            raise ValueError("input authority exact normalization/no-cache proof drift")
        for item in (self.source_normalizer_body_sha256, self.t4_normalizer_semantic_sha256,
                     self.behavior_normalizer_semantic_sha256, self.strict_manifest_sha256):
            _require_sha(item, "input authority normalizer/manifest")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "tfsr_phase_e_input_authority_v1", "cell": CELL,
            "records": [item.payload() for item in self.records],
            "source_normalizer_body_sha256": self.source_normalizer_body_sha256,
            "t4_normalizer_semantic_sha256": self.t4_normalizer_semantic_sha256,
            "behavior_normalizer_semantic_sha256": self.behavior_normalizer_semantic_sha256,
            "strict_manifest_sha256": self.strict_manifest_sha256,
            "raw_to_normalized_exact": self.raw_to_normalized_exact,
            "no_cache_readonly_adapter": self.no_cache_readonly_adapter,
        }


def validate_input_authority_evidence(
    evidence: InputAuthorityEvidence,
    *,
    within_roster: Sequence[str],
    within_assets: Sequence[EvaluationAssetBinding],
    external_roster: Sequence[EvaluationAssetBinding],
) -> None:
    """Validate all target-free binding facts before writing input authority."""
    if not isinstance(evidence, InputAuthorityEvidence):
        raise FailClosedError("backend did not return InputAuthorityEvidence")
    if evidence.source_normalizer_body_sha256 != SOURCE_NORMALIZER_BODY_SHA:
        raise FailClosedError("source normalizer body authority drift")
    if evidence.t4_normalizer_semantic_sha256 != T4_NORMALIZER_SEMANTIC_SHA:
        raise FailClosedError("T4 normalizer semantic authority drift")
    if evidence.behavior_normalizer_semantic_sha256 != BEHAVIOR_NORMALIZER_SEMANTIC_SHA:
        raise FailClosedError("behavior normalizer semantic authority drift")
    strict_manifest = next(item for item in FIXED_AUTHORITIES if item.name == "strict_manifest")
    if evidence.strict_manifest_sha256 != strict_manifest.sha256:
        raise FailClosedError("strict manifest authority drift")
    by_surface = {
        surface: tuple(item.session for item in evidence.records if item.surface == surface)
        for surface in ("within", "external")
    }
    if by_surface["within"] != tuple(sorted(within_roster)):
        raise FailClosedError("input authority within roster differs from strict val[6]")
    expected_within = tuple(item.session for item in within_assets)
    if expected_within != tuple(sorted(within_roster)):
        raise FailClosedError("input authority within asset roster differs from strict val[6]")
    expected_external = tuple(item.session for item in external_roster)
    if by_surface["external"] != expected_external:
        raise FailClosedError("input authority external roster differs from fixed ledger")
    expected_assets = {
        **{("within", item.session): item for item in within_assets},
        **{("external", item.session): item for item in external_roster},
    }
    for record in evidence.records:
        if record.surface in {"within", "external"}:
            expected = expected_assets.get((record.surface, record.session))
            if expected is None:
                raise FailClosedError("input authority record has no expected target asset")
            # A real no-cache opening additionally binds the descriptor that
            # remained live through parsing.  Mock-only bindings deliberately
            # have no DataRootCapability and therefore cannot fabricate that
            # physical fact.
            held_identity = record.asset.get("held_descriptor_identity")
            if expected.data_root is None:
                expected_payload = expected.payload()
            else:
                try:
                    identity = _identity_list(held_identity, length=3, label="held descriptor")
                except ValueError as error:
                    raise FailClosedError("input authority lacks held descriptor identity") from error
                expected_payload = expected.payload(held_descriptor_identity=(identity[0], identity[1], identity[2]))
            if record.asset != expected_payload:
                raise FailClosedError("input authority target asset binding drift")
        if record.session in FORMAL_TEST_SESSION_NAMES:
            raise FailClosedError("input authority attempted formal session")


class ScoreBackend(Protocol):
    """Injected live adapter; production code cannot access target data otherwise."""

    def resolve_inputs(
        self,
        *,
        spec: ScoreSpec,
        within_roster: tuple[str, ...],
        within_assets: tuple[EvaluationAssetBinding, ...],
        external_roster: tuple[EvaluationAssetBinding, ...],
        flags: ScoreFlags,
    ) -> InputAuthorityEvidence: ...

    def score_cell_d(
        self, *, surface: str, input_authority_sha256: str, flags: ScoreFlags,
    ) -> ModeEvidence: ...

    def score_tfsr(
        self, *, surface: str, mode: str, input_authority_sha256: str, flags: ScoreFlags,
    ) -> ModeEvidence: ...

    def reverify_after_forwards(self, *, flags: ScoreFlags) -> None: ...

    def resource_disclosure(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


class NoLiveBackend:
    """Public script fallback: every live route remains fail-closed for review."""

    def resolve_inputs(self, **_: Any) -> InputAuthorityEvidence:
        raise FailClosedError("no reviewed Phase-E live backend injected")

    def score_cell_d(self, **_: Any) -> ModeEvidence:
        raise AssertionError("unreachable without input authority")

    def score_tfsr(self, **_: Any) -> ModeEvidence:
        raise AssertionError("unreachable without input authority")

    def reverify_after_forwards(self, **_: Any) -> None:
        raise AssertionError("unreachable without input authority")

    def resource_disclosure(self) -> Mapping[str, object]:
        raise AssertionError("unreachable without input authority")

    def close(self) -> None:
        return None


def _raw_t4_sua_axis_proof_payload(
    np: Any,
    *,
    session: str,
    neural_unit_count: int,
    channel_ids: object,
    source_unit_count: object,
    raw_t4: Any,
    metadata: object,
) -> dict[str, object]:
    """Prove M30 T4 row k is attached to SUA neural column k.

    The closure binds the two concrete functions named below.  Their shared
    sorted-unit semantics are only accepted after this runtime check confirms
    the parser returned the contiguous SUA unit axis and the independent T4
    producer returned one four-vector per same source-unit index.
    """
    supplied_channel_ids = np.asarray(channel_ids)
    if supplied_channel_ids.dtype != np.dtype(np.int64):
        raise FailClosedError("raw M30 T4/SUA channel-order proof drift")
    channel_array = np.ascontiguousarray(supplied_channel_ids)
    expected_channels = np.arange(neural_unit_count, dtype=np.int64)
    raw_array = np.ascontiguousarray(raw_t4, dtype=np.float32)
    if (
        not isinstance(session, str)
        or not session
        or type(neural_unit_count) is not int
        or neural_unit_count <= 0
        or type(source_unit_count) is not int
        or source_unit_count != neural_unit_count
        or channel_array.shape != (neural_unit_count,)
        or not np.array_equal(channel_array, expected_channels)
        or raw_array.shape != (neural_unit_count, 4)
        or getattr(metadata, "feature_group", None) != "t4"
        or getattr(metadata, "pool_size", None) != 30
    ):
        raise FailClosedError("raw M30 T4/SUA channel-order proof drift")
    return {
        "schema": "tfsr_phase_e_raw_t4_sua_axis_proof_v1",
        "session": session,
        "signal_view": "sua",
        "channel_ids_dtype": "int64",
        "channel_ids_sha256": _sha(channel_array.tobytes()),
        "source_unit_count": source_unit_count,
        "raw_t4_shape": [neural_unit_count, 4],
        "feature_group": "t4",
        "pool_size": 30,
        "mapping": "raw_t4_row_k_equals_sua_neural_column_k_for_contiguous_channel_ids",
        "closure_bound_functions": [
            "mc_maze.multisession_datamodule.load_dandi688_session",
            "mc_maze.unit_side_features.compute_unit_side_features_uncached",
        ],
    }


def _last_bin_authority_from_behavior(
    np: Any,
    *,
    behavior: Any,
    starts: Any,
) -> tuple[Any, Any, str, str, int]:
    """Return the durable query-level target/mask/count authority.

    A full behavior mask is useful diagnostic provenance but cannot prove that
    the exact final-bin target fed to the governing metric stayed unchanged.
    This helper binds those 2-D query targets and the uint8 query mask
    separately, and accepts only windows whose governed final bin is valid.
    """
    behavior_array = np.ascontiguousarray(behavior, dtype=np.float32)
    starts_array = np.ascontiguousarray(starts, dtype=np.int64)
    if (
        behavior_array.ndim != 2
        or behavior_array.shape[1] != 2
        or starts_array.ndim != 1
        or starts_array.size <= 0
        or int(starts_array.min()) < 0
        or int(starts_array.max()) + PUBLIC_SPEC.window_bins > behavior_array.shape[0]
    ):
        raise FailClosedError("last-bin target authority shape/window drift")
    targets = np.ascontiguousarray(
        behavior_array[starts_array + (PUBLIC_SPEC.window_bins - 1)],
        dtype=np.float32,
    )
    valid_mask = np.ascontiguousarray(np.all(targets != -1.0, axis=1), dtype=np.uint8)
    valid_count = int(valid_mask.sum())
    if targets.shape != (starts_array.size, 2) or valid_count != int(starts_array.size):
        raise FailClosedError("last-bin target authority contains invalid governed windows")
    return (
        targets,
        valid_mask,
        _sha(targets.tobytes()),
        _sha(valid_mask.tobytes()),
        valid_count,
    )


def _validate_last_bin_authority_arrays(
    np: Any,
    *,
    target: Any,
    valid_mask: Any,
    authority: SessionInputAuthority,
) -> None:
    """Recheck the exact governed query arrays before every metric call."""
    target_array = np.ascontiguousarray(target, dtype=np.float32)
    mask_array = np.ascontiguousarray(valid_mask, dtype=np.uint8)
    if (
        target_array.shape != (authority.n_windows, 2)
        or mask_array.shape != (authority.n_windows,)
        or _sha(target_array.tobytes()) != authority.target_last_bin_sha256
        or _sha(mask_array.tobytes()) != authority.last_bin_valid_mask_sha256
        or int(mask_array.sum()) != authority.last_bin_valid_count
        or authority.last_bin_valid_count != authority.n_windows
    ):
        raise FailClosedError("physical governing last-bin target/mask authority drift")


@dataclass
class _PhysicalSession:
    """One parsed target session whose original asset FD remains held.

    Arrays are retained only after their parser consumed a private verified
    snapshot.  The original descriptor remains open until final revalidation,
    so a later pathname replacement cannot retroactively make the input
    authority look valid.
    """

    binding: EvaluationAssetBinding
    record: Any
    held: HeldAsset
    raw_t4: Any
    aligned_t4: Any
    last_bin_targets: Any
    last_bin_valid_mask: Any
    input_record: SessionInputAuthority


_UNINITIALIZED_LAZY_STATE_SENTINEL = b"|uninitialized-lazy|"


def _physical_initialized_tensor_digest(value: Any, torch: Any) -> str:
    """Canonical initialized-tensor leaf digest used by Phase-E state hashing.

    The representation matches the laboratory's state-hash discipline: CPU
    contiguous values, IEEE signed-zero normalized for floating tensors, then
    dtype, original shape, and raw byte content.  This helper is deliberately
    never called on a lazy parameter.
    """
    material = value.detach().cpu().contiguous().reshape(-1)
    if material.is_floating_point():
        material = material + 0
    digest = hashlib.sha256()
    digest.update(str(material.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    if material.numel():
        digest.update(material.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _physical_model_state_digest(model: Any, torch: Any) -> str:
    """Lazy-safe canonical digest for the exact live Cell-D/TF-SR state.

    The byte algorithm is explicit and stable: traverse sorted state-dict
    keys; append each UTF-8 key; append the exact
    ``|uninitialized-lazy|`` sentinel for an ``UninitializedParameter``; or,
    for every initialized tensor, append the ASCII SHA-256 of its normalized
    dtype/shape/bytes leaf.  Thus dead lazy parameters are bound by both key
    and sentinel without being detached, materialized, dropped, or initialized.
    """
    from torch.nn.parameter import UninitializedParameter

    state = model.state_dict()
    if not isinstance(state, Mapping):
        raise FailClosedError("model state is not a mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        if not isinstance(name, str) or not name:
            raise FailClosedError("model state contains an invalid key")
        value = state[name]
        digest.update(name.encode("utf-8"))
        if isinstance(value, UninitializedParameter):
            digest.update(_UNINITIALIZED_LAZY_STATE_SENTINEL)
            continue
        if not torch.is_tensor(value):
            raise FailClosedError("model state contains a non-tensor value")
        digest.update(_physical_initialized_tensor_digest(value, torch).encode("ascii"))
    return digest.hexdigest()


def _sealed_cell_d_parameter_authority(value: object) -> dict[str, int]:
    """Extract only the sealed terminal facts needed for lazy-safe counting."""
    if not isinstance(value, Mapping) or value.get("cell") != "D":
        raise FailClosedError("Cell-D terminal parameter authority cell drift")
    initial = value.get("initial_state")
    if not isinstance(initial, Mapping):
        raise FailClosedError("Cell-D terminal initial-state authority missing")
    proof_map = initial.get("bitwise_equality_proof")
    if not isinstance(proof_map, Mapping) or not isinstance(proof_map.get("D"), Mapping):
        raise FailClosedError("Cell-D terminal D parameter proof missing")
    proof = proof_map["D"]
    sealed_parameters = proof.get("trainable_parameters")
    # The fixed terminal's complete receipt has already passed its own
    # authority validation.  Resource accounting binds only the two literal
    # facts which the Phase-E work order names as this helper's authority;
    # inventing ancillary terminal fields here would make the live scorer
    # depend on an undocumented receipt shape.
    if (type(sealed_parameters) is not int
            or sealed_parameters != CELL_D_INITIALIZED_TRAINABLE_PARAMETERS):
        raise FailClosedError("Cell-D terminal initialized-parameter authority drift")
    swa = value.get("swa")
    if not isinstance(swa, Mapping) or not isinstance(swa.get("manifest"), Mapping):
        raise FailClosedError("Cell-D terminal SWA manifest missing")
    sealed_lazy_count = swa["manifest"].get("uninitialized_lazy_tensor_count")
    if type(sealed_lazy_count) is not int or sealed_lazy_count != len(CELL_D_UNINITIALIZED_LAZY_PARAMETER_KEYS):
        raise FailClosedError("Cell-D terminal lazy-parameter authority drift")
    return {
        "initialized_trainable_parameters": sealed_parameters,
        "uninitialized_lazy_parameter_count": sealed_lazy_count,
    }


def _cell_d_lazy_parameter_keys(model: Any) -> tuple[str, ...]:
    """Observe Cell-D's dead lazy topology without materializing it.

    This helper deliberately reads only parameter names and ``isinstance`` for
    lazy parameters.  In particular it must not call ``numel``, ``detach``,
    ``shape``, or a tensor conversion on an ``UninitializedParameter``.
    """
    from torch.nn.parameter import UninitializedParameter

    names: list[str] = []
    seen: set[str] = set()
    for name, parameter in model.named_parameters():
        if not isinstance(name, str) or not name or name in seen:
            raise FailClosedError("Cell-D named-parameter key topology drift")
        seen.add(name)
        if isinstance(parameter, UninitializedParameter):
            names.append(name)
    return tuple(sorted(names))


def _require_cell_d_lazy_parameter_topology(model: Any) -> None:
    """Fail closed if construction, placement, or strict load changes Cell-D."""
    if _cell_d_lazy_parameter_keys(model) != tuple(sorted(CELL_D_UNINITIALIZED_LAZY_PARAMETER_KEYS)):
        raise FailClosedError("Cell-D lazy parameter topology drift")


def _cell_d_lazy_safe_parameter_accounting(
    model: Any,
    torch: Any,
    *,
    sealed_terminal: object,
) -> dict[str, object]:
    """Count Cell-D deployable parameters without touching its dead lazy path.

    The terminal supplies the exact initialized/trainable count.  This live
    check independently observes the same count while recording, rather than
    silently zero-counting, the exact two uninitialized decoder-ID entries.
    No operation other than ``isinstance`` is performed on a lazy parameter.
    """
    from torch.nn.parameter import UninitializedParameter

    authority = _sealed_cell_d_parameter_authority(sealed_terminal)
    observed_lazy: list[str] = []
    initialized_trainable = 0
    seen: set[str] = set()
    for name, parameter in model.named_parameters():
        if not isinstance(name, str) or not name or name in seen:
            raise FailClosedError("Cell-D named-parameter key topology drift")
        seen.add(name)
        if isinstance(parameter, UninitializedParameter):
            observed_lazy.append(name)
            continue
        if not torch.is_tensor(parameter):
            raise FailClosedError("Cell-D initialized parameter is not a tensor")
        if parameter.requires_grad:
            initialized_trainable += int(parameter.numel())
    if tuple(sorted(observed_lazy)) != tuple(sorted(CELL_D_UNINITIALIZED_LAZY_PARAMETER_KEYS)):
        raise FailClosedError("Cell-D lazy parameter topology drift")
    if initialized_trainable != authority["initialized_trainable_parameters"]:
        raise FailClosedError("Cell-D initialized trainable parameter count drift")
    return {
        "schema": CELL_D_PARAMETER_ACCOUNTING_SCHEMA,
        "count_semantics": "initialized_trainable_parameters_only",
        "initialized_trainable_parameters": initialized_trainable,
        "sealed_initialized_trainable_parameters": authority["initialized_trainable_parameters"],
        "uninitialized_lazy_parameter_count": len(observed_lazy),
        "sealed_uninitialized_lazy_parameter_count": authority["uninitialized_lazy_parameter_count"],
        "uninitialized_lazy_parameter_keys": list(CELL_D_UNINITIALIZED_LAZY_PARAMETER_KEYS),
        "uninitialized_lazy_parameter_role": CELL_D_UNINITIALIZED_LAZY_PARAMETER_ROLE,
    }


def _initialized_trainable_parameter_count(model: Any, torch: Any, *, label: str) -> int:
    """Count an ordinary model, failing rather than coercing an unexpected lazy parameter."""
    from torch.nn.parameter import UninitializedParameter

    count = 0
    for name, parameter in model.named_parameters():
        if not isinstance(name, str) or not name:
            raise FailClosedError(f"{label} named-parameter key drift")
        if isinstance(parameter, UninitializedParameter):
            raise FailClosedError(f"{label} unexpectedly contains an uninitialized lazy parameter")
        if not torch.is_tensor(parameter):
            raise FailClosedError(f"{label} parameter is not a tensor")
        if parameter.requires_grad:
            count += int(parameter.numel())
    return count


def _build_physical_resource_disclosure(
    *,
    cell_d: Any,
    tfsr: Any,
    torch: Any,
    torchmetrics_version: str,
    training_peak_memory_bytes: int,
    score_peak_memory_bytes: int,
    aligned_latency_ms: float,
    sealed_cell_d_terminal: object,
) -> dict[str, object]:
    """Build the exact final resource receipt shared by the physical scorer/tests."""
    if not isinstance(torchmetrics_version, str):
        raise FailClosedError("TorchMetrics resource-version type drift")
    tfsr_accounting = tfsr.accounting(representative_n=128)
    if not isinstance(tfsr_accounting, Mapping):
        raise FailClosedError("TF-SR accounting payload drift")
    try:
        analytic_macs = int(tfsr_accounting["analytic_mac_estimate_n"])
        persistent_bytes = int(tfsr_accounting["persistent_state_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise FailClosedError("TF-SR accounting fields drift") from error
    cell_d_accounting = _cell_d_lazy_safe_parameter_accounting(
        cell_d, torch, sealed_terminal=sealed_cell_d_terminal,
    )
    payload = {
        "cell_d_parameters": cell_d_accounting["initialized_trainable_parameters"],
        "cell_d_parameter_accounting": cell_d_accounting,
        "tfsr_parameters": _initialized_trainable_parameter_count(tfsr, torch, label="TF-SR"),
        "tfsr_analytic_macs": analytic_macs,
        "persistent_state": {
            "window_local": True,
            "bytes": persistent_bytes,
            "description": "zeroed GRU state per 50-bin query window",
        },
        "training_peak_memory_bytes": training_peak_memory_bytes,
        "score_peak_memory_bytes": score_peak_memory_bytes,
        "aligned_latency_ms": aligned_latency_ms,
        "runtime": {**FROZEN_SCORE_DEVICE, "torchmetrics_version": torchmetrics_version},
    }
    return _validate_resource_disclosure(payload)


def _canonical_checkpoint_state(payload: object, *, label: str) -> Mapping[str, Any]:
    """Extract a strict model state from a sealed checkpoint payload."""
    if not isinstance(payload, Mapping):
        raise FailClosedError(f"{label} checkpoint payload is not a mapping")
    state = payload.get("state_dict", payload.get("state"))
    if not isinstance(state, Mapping) or not state:
        raise FailClosedError(f"{label} checkpoint state missing")
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise FailClosedError(f"{label} checkpoint state key drift")
        stripped = key[6:] if key.startswith("model.") else key
        if stripped in normalized:
            raise FailClosedError(f"{label} checkpoint state prefix collision")
        normalized[stripped] = value
    return normalized


def _load_cell_d_swa_weights_only_state(torch: Any, body: bytes) -> Mapping[str, Any]:
    """Load the SHA-verified in-memory Cell-D SWA with one temporary safe global.

    The sealed payload legitimately serializes ``UninitializedParameter``.
    This leaves ``weights_only=True`` in force and scopes exactly that one
    class to the local ``safe_globals`` context; no global allowlist mutation
    or pathname load is permitted.
    """
    import io
    from torch.nn.parameter import UninitializedParameter

    if not isinstance(body, bytes):
        raise FailClosedError("Cell-D SWA body must be immutable bytes")
    serialization = getattr(torch, "serialization", None)
    safe_globals = getattr(serialization, "safe_globals", None)
    if not callable(safe_globals):
        raise FailClosedError("Torch safe_globals context is unavailable")
    try:
        with safe_globals([UninitializedParameter]):
            payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    except Exception as error:
        raise FailClosedError("cannot read sealed Cell-D SWA payload") from error
    return _canonical_checkpoint_state(payload, label="Cell-D")


class PhysicalMatchedScoreBackend:
    """The production no-cache, read-only Phase-E evaluator.

    Construction is intentionally inert: it imports neither Torch nor data
    modules and opens no path.  ``resolve_inputs`` is reached only after the
    core has durably published/reloaded an attempt and all target bindings were
    formed from the root-signed preflight plus fixed external v2 ledger.
    """

    def __init__(
        self,
        *,
        root: Path,
        fixed_authorities: Mapping[str, AuthorityMaterial],
        training: TrainingEvidence,
        authorization: PhaseEAuthorization,
    ) -> None:
        expected = {item.name for item in FIXED_AUTHORITIES}
        if set(fixed_authorities) != expected:
            raise ValueError("physical backend fixed-authority set drift")
        self._root = root.absolute()
        self._fixed = dict(fixed_authorities)
        self._training = training
        self._authorization = authorization
        self._runtime: Mapping[str, Any] | None = None
        self._sessions: dict[str, dict[str, _PhysicalSession]] = {"within": {}, "external": {}}
        self._held: list[HeldAsset] = []
        self._cell_d: Any | None = None
        self._tfsr: Any | None = None
        self._state_at_load: dict[str, str] = {}
        self._aligned_latency_ms: float | None = None
        self._closed = False

    def _load_runtime(self) -> Mapping[str, Any]:
        """Import the exact physical scorer only after a durable attempt."""
        if self._runtime is not None:
            return self._runtime
        import importlib.util
        import subprocess
        import sys
        import time as runtime_time

        # The public static loader does not put these packages on sys.path.
        # Add fixed repository roots only here; this is not file discovery.
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

        def _query_nvidia_smi() -> Sequence[str]:
            return subprocess.check_output(
                [
                    "nvidia-smi", "-i", "1", "--query-gpu=uuid,pci.bus_id,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip().splitlines()

        # Attest both memory authorities before importing a decoder, parser,
        # or model builder.  A device mismatch therefore cannot reach an
        # evaluation input or forward path.
        _attest_physical_score_runtime(
            torch=torch,
            torchmetrics=torchmetrics,
            nvidia_smi_query=_query_nvidia_smi,
        )

        from torchmetrics import R2Score
        from . import train
        from .model import NormalizedT4Batch, T4Normalizer, TFSRDecoder, ordered_unit_digest
        from mc_maze.multisession_datamodule import load_dandi688_session
        from mc_maze.unit_side_features import compute_unit_side_features_uncached
        # Do not import ``src.tfpd_lane`` as a package: its package initializer
        # intentionally imports exploratory mechanisms irrelevant to Cell D.
        # Loading this explicit, closure-bound builder file keeps the live
        # runtime dependency graph narrow and auditable.
        pop_robust_path = self._root / "tfpd_exploration/src/tfpd_lane/pop_robust.py"
        pop_spec = importlib.util.spec_from_file_location("_tfsr_phase_e_pop_robust", pop_robust_path)
        if pop_spec is None or pop_spec.loader is None:
            raise FailClosedError("cannot construct explicit Cell-D builder loader")
        pop_module = importlib.util.module_from_spec(pop_spec)
        sys.modules[pop_spec.name] = pop_module
        pop_spec.loader.exec_module(pop_module)
        build_population_robustness_model = pop_module.build_population_robustness_model

        # The resource receipt must describe this evaluation lifecycle rather
        # than an arbitrary earlier allocation in the same CUDA process.
        torch.cuda.reset_peak_memory_stats(0)
        self._runtime = {
            "np": np,
            "torch": torch,
            "R2Score": R2Score,
            "time": runtime_time,
            "train": train,
            "NormalizedT4Batch": NormalizedT4Batch,
            "T4Normalizer": T4Normalizer,
            "TFSRDecoder": TFSRDecoder,
            "ordered_unit_digest": ordered_unit_digest,
            "load_dandi688_session": load_dandi688_session,
            "compute_unit_side_features_uncached": compute_unit_side_features_uncached,
            "build_population_robustness_model": build_population_robustness_model,
            "device": torch.device("cuda:0"),
        }
        return self._runtime

    def _normalizers(self) -> tuple[Any, Any, Any, Any]:
        runtime = self._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        value = self._authorization.preflight.get("normalizers")
        checked = _validate_preflight_normalizers(value)
        behavior_mean = np.asarray(checked["behavior_mean"], dtype=np.float32)
        behavior_std = np.asarray(checked["behavior_std"], dtype=np.float32)
        mean = torch.tensor(checked["t4_mean"], dtype=torch.float32, device=runtime["device"])
        std = torch.tensor(checked["t4_std"], dtype=torch.float32, device=runtime["device"])
        return behavior_mean, behavior_std, mean, std

    def _parse_one(
        self,
        *,
        binding: EvaluationAssetBinding,
        behavior_mean: Any,
        behavior_std: Any,
        t4_mean: Any,
        t4_std: Any,
        flags: ScoreFlags,
    ) -> _PhysicalSession:
        runtime = self._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        held = hold_verified_asset(binding)
        self._held.append(held)
        if binding.surface == "within":
            flags.within_opened = True
        elif binding.surface == "external":
            flags.external_opened = True
        else:  # EvaluationAssetBinding already rejects this; retain a hard stop.
            raise FailClosedError("physical target surface drift")
        snapshot = private_verified_snapshot(held)
        try:
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
            raw_t4, metadata = runtime["compute_unit_side_features_uncached"](
                snapshot.path,
                feature_group="t4",
                pool_size=30,
                bin_size_ms=20,
                window_size=50,
                trial_result_filter="R",
                signal_view="sua",
            )
            snapshot.reverify()
        finally:
            # Keep `held` open for post-forward revalidation, but release only
            # the private parser copy as soon as both parsers have consumed it.
            snapshot.temporary_directory.cleanup()
        held.reverify()
        if (getattr(record, "name", None) != binding.session
                or getattr(record, "signal_view", None) != "sua"):
            raise FailClosedError("no-cache parser session/signal-view drift")
        neural = np.asarray(record.neural, dtype=np.float32)
        behavior = np.asarray(record.behavior, dtype=np.float32)
        calib = np.asarray(record.calib_trials, dtype=np.float32)
        starts = np.asarray(record.valid_starts, dtype=np.int64)
        channel_ids = getattr(record, "channel_ids", None)
        if (neural.ndim != 2 or behavior.shape != (neural.shape[0], 2)
                or calib.shape != (30, 100, neural.shape[1])
                or starts.ndim != 1 or starts.size == 0
                or not np.isfinite(neural).all() or not np.isfinite(behavior).all()
                or not np.isfinite(calib).all() or channel_ids is None
                or len(channel_ids) != neural.shape[1] or neural.shape[1] < 2):
            raise FailClosedError("no-cache parsed session tensor/unit shape drift")
        if (not np.all(starts[:-1] < starts[1:])
                or int(starts.min()) < 0 or int(starts.max()) + 50 > neural.shape[0]):
            raise FailClosedError("no-cache parsed query-start contract drift")
        raw_array = np.ascontiguousarray(raw_t4, dtype=np.float32)
        if raw_array.shape != (neural.shape[1], 4) or not np.isfinite(raw_array).all():
            raise FailClosedError("recomputed raw M30 T4 shape/finite drift")
        raw_t4_sua_axis_proof = _raw_t4_sua_axis_proof_payload(
            np,
            session=binding.session,
            neural_unit_count=int(neural.shape[1]),
            channel_ids=channel_ids,
            source_unit_count=getattr(record, "source_unit_count", None),
            raw_t4=raw_array,
            metadata=metadata,
        )
        raw_t4_sua_axis_proof_sha256 = _sha(_json_bytes(raw_t4_sua_axis_proof))
        ordered_ids = tuple(f"{binding.session}:channel:{int(item)}" for item in channel_ids)
        if len(set(ordered_ids)) != len(ordered_ids):
            raise FailClosedError("canonical ordered unit IDs are not unique")
        raw_tensor = torch.as_tensor(raw_array, dtype=torch.float32, device=runtime["device"]).unsqueeze(0)
        raw_sha = _sha(raw_array.tobytes())
        roster_digest = _sha(_json_bytes(self._fixed["source_smoke"].value["authorities"]["roster"]))
        normalizer = runtime["T4Normalizer"](
            t4_mean, t4_std, raw_sha, SOURCE_NORMALIZER_BODY_SHA,
        ).to(runtime["device"])
        aligned = normalizer(
            raw_tensor,
            roster_digest=roster_digest,
            ordered_unit_ids=ordered_ids,
            lineage=(
                SOURCE_SMOKE_RELATIVE,
                binding.surface,
                binding.session,
                binding.expected_sha256,
                raw_sha,
                SOURCE_NORMALIZER_BODY_SHA,
            ),
        )
        direct = ((raw_tensor - t4_mean) / t4_std).detach().clone()
        if not torch.equal(aligned.tensor, direct):
            raise FailClosedError("raw M30 T4 to source-normalized side tensor mismatch")
        target_last, last_bin_valid_mask, target_last_sha, last_bin_mask_sha, last_bin_count = _last_bin_authority_from_behavior(
            np,
            behavior=behavior,
            starts=starts,
        )
        input_record = SessionInputAuthority(
            session=binding.session,
            surface=binding.surface,
            # The record binds both the preflight/ledger address and the
            # descriptor identity that stayed live while the parser consumed
            # its verified private snapshot.
            asset=binding.payload(held_descriptor_identity=held.identity),
            ordered_unit_digest=runtime["ordered_unit_digest"](ordered_ids),
            unit_count=int(neural.shape[1]),
            raw_t4=tensor_binding(raw_tensor[0]),
            normalized_t4=tensor_binding(aligned.tensor[0]),
            calibration_sha256=_sha(np.ascontiguousarray(calib).tobytes()),
            query_start_sha256=_sha(np.ascontiguousarray(starts).tobytes()),
            neural_sha256=_sha(np.ascontiguousarray(neural).tobytes()),
            behavior_sha256=_sha(np.ascontiguousarray(behavior).tobytes()),
            valid_mask_sha256=_sha(np.ascontiguousarray((behavior != -1.0).all(axis=-1)).tobytes()),
            target_last_bin_sha256=target_last_sha,
            last_bin_valid_mask_sha256=last_bin_mask_sha,
            last_bin_valid_count=last_bin_count,
            n_windows=int(starts.size),
            raw_t4_sua_axis_proof=raw_t4_sua_axis_proof,
            raw_t4_sua_axis_proof_sha256=raw_t4_sua_axis_proof_sha256,
        )
        return _PhysicalSession(
            binding=binding,
            record=record,
            held=held,
            raw_t4=raw_tensor,
            aligned_t4=aligned,
            last_bin_targets=target_last,
            last_bin_valid_mask=last_bin_valid_mask,
            input_record=input_record,
        )

    def resolve_inputs(
        self,
        *,
        spec: ScoreSpec,
        within_roster: tuple[str, ...],
        within_assets: tuple[EvaluationAssetBinding, ...],
        external_roster: tuple[EvaluationAssetBinding, ...],
        flags: ScoreFlags,
    ) -> InputAuthorityEvidence:
        if self._closed or spec != PUBLIC_SPEC:
            raise FailClosedError("physical backend public ScoreSpec/lifecycle drift")
        if (tuple(item.session for item in within_assets) != tuple(sorted(within_roster))
                or len(external_roster) != PUBLIC_SPEC.external_count):
            raise FailClosedError("physical backend target roster binding drift")
        behavior_mean, behavior_std, t4_mean, t4_std = self._normalizers()
        all_records: list[SessionInputAuthority] = []
        for binding in (*within_assets, *external_roster):
            parsed = self._parse_one(
                binding=binding,
                behavior_mean=behavior_mean,
                behavior_std=behavior_std,
                t4_mean=t4_mean,
                t4_std=t4_std,
                flags=flags,
            )
            if parsed.binding.session in self._sessions[parsed.binding.surface]:
                raise FailClosedError("physical backend duplicate session binding")
            self._sessions[parsed.binding.surface][parsed.binding.session] = parsed
            all_records.append(parsed.input_record)
        if (tuple(sorted(self._sessions["within"])) != tuple(sorted(within_roster))
                or len(self._sessions["external"]) != PUBLIC_SPEC.external_count):
            raise FailClosedError("physical backend parsed roster drift")
        return InputAuthorityEvidence(
            records=tuple(sorted(all_records, key=lambda item: (item.surface, item.session))),
            source_normalizer_body_sha256=SOURCE_NORMALIZER_BODY_SHA,
            t4_normalizer_semantic_sha256=T4_NORMALIZER_SEMANTIC_SHA,
            behavior_normalizer_semantic_sha256=BEHAVIOR_NORMALIZER_SEMANTIC_SHA,
            strict_manifest_sha256=next(item.sha256 for item in FIXED_AUTHORITIES if item.name == "strict_manifest"),
            raw_to_normalized_exact=True,
            no_cache_readonly_adapter=True,
        )

    def _ensure_models(self) -> None:
        if self._cell_d is not None and self._tfsr is not None:
            return
        runtime = self._load_runtime()
        torch = runtime["torch"]
        device = runtime["device"]
        # Revalidate the exact Phase-D SWA binding at point of use.  This is a
        # read-only model-artifact operation, never an evaluation-data access.
        train = runtime["train"]
        artifact = _attach_readonly_phase_d_artifact(self._root, train)
        terminal = artifact.reload_json("terminal.json", self._training.terminal_sha256)
        identity = train.production_identity(self._root)
        train.validate_terminal_receipt(terminal, train.PUBLIC_SPEC, identity)
        expected_binding = {
            "cell": train.CELL,
            "run_spec": train.PUBLIC_SPEC.payload(),
            "launch_sha256": terminal["launch_sha256"],
            "launch_closure": terminal["launch_closure"],
            "predecessor": terminal["predecessor"],
        }
        body = artifact.reload_pair("swa.pt", self._training.swa_sha256)
        train_backend = train.TorchTrainingBackend(self._root)
        tfsr_swa = train_backend.validate_swa(body, train.PUBLIC_SPEC, expected_binding=expected_binding)
        if tfsr_swa.get("state_digest") != self._training.swa_state_digest:
            raise FailClosedError("physical TF-SR SWA state digest drift")
        tfsr = runtime["TFSRDecoder"](capture_diagnostics=False).to(device)
        tfsr.load_state_dict(tfsr_swa["state"], strict=True)
        tfsr.eval()
        if tfsr.capture_diagnostics is not False:
            raise FailClosedError("physical TF-SR capture_diagnostics drift")

        cell_d_state = _load_cell_d_swa_weights_only_state(torch, self._fixed["cell_d_swa"].body)
        cell_d = runtime["build_population_robustness_model"](seed=42, cell="D")
        _require_cell_d_lazy_parameter_topology(cell_d)
        # ``Module.to`` is permitted only because the dead lazy topology is
        # checked before and after placement; it must not materialize/drop the
        # unused decoder-ID parameters merely to reach the attested GPU.
        cell_d = cell_d.to(device)
        _require_cell_d_lazy_parameter_topology(cell_d)
        cell_d.load_state_dict(cell_d_state, strict=True)
        _require_cell_d_lazy_parameter_topology(cell_d)
        cell_d.eval()
        self._cell_d, self._tfsr = cell_d, tfsr
        self._state_at_load = {
            "cell_d": _physical_model_state_digest(cell_d, torch),
            "tfsr": _physical_model_state_digest(tfsr, torch),
        }
        if not self._all_gradients_none(cell_d, torch) or not self._all_gradients_none(tfsr, torch):
            raise FailClosedError("fresh strict score model has preexisting gradients")

    @staticmethod
    def _all_gradients_none(model: Any, torch: Any) -> bool:
        return all(parameter.grad is None for parameter in model.parameters())

    def _session_batch(self, session: _PhysicalSession, starts: Sequence[int]) -> tuple[Any, Any, Any]:
        runtime = self._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        record = session.record
        neural = torch.from_numpy(np.stack([record.neural[start:start + 50] for start in starts])).float().to(runtime["device"])
        behavior = torch.from_numpy(np.stack([record.behavior[start:start + 50] for start in starts])).float().to(runtime["device"])
        calib = torch.from_numpy(np.ascontiguousarray(record.calib_trials)).float().unsqueeze(0).to(runtime["device"])
        calib = calib.expand(neural.shape[0], -1, -1, -1)
        if (neural.shape != (len(starts), 50, session.aligned_t4.num_units)
                or behavior.shape != (len(starts), 50, 2)
                or calib.shape != (len(starts), 30, 100, session.aligned_t4.num_units)):
            raise FailClosedError("physical score batch shape drift")
        return neural, behavior, calib

    def _t4_for_batch(self, session: _PhysicalSession, batch: int, mode: str) -> tuple[Any, Mapping[str, object]]:
        runtime = self._load_runtime()
        capability_type = runtime["NormalizedT4Batch"]
        aligned_one = session.aligned_t4
        aligned = capability_type(
            tensor=aligned_one.tensor.expand(batch, -1, -1).detach().clone(),
            raw_authority_sha256=aligned_one.raw_authority_sha256,
            normalizer_authority_sha256=aligned_one.normalizer_authority_sha256,
            roster_digest=aligned_one.roster_digest,
            ordered_unit_digest=aligned_one.ordered_unit_digest,
            ordered_unit_ids=aligned_one.ordered_unit_ids,
            lineage=aligned_one.lineage,
            diagnostic_mode="aligned",
        )
        controls = make_t4_controls(aligned)
        chosen = {"aligned": controls.aligned, "zero": controls.zero, "wrong_pair": controls.wrong_pair}.get(mode)
        if chosen is None:
            raise FailClosedError("physical T4 diagnostic mode drift")
        proof: dict[str, object] = {
            "mode": mode,
            "post_normalization": True,
            "b3s_recomputed": True,
            "ordered_unit_digest": chosen.ordered_unit_digest,
            "unit_count": chosen.num_units,
        }
        if mode == "zero":
            proof["exact_zeros_like"] = True
        if mode == "wrong_pair":
            proof.update({
                "permutation_is_derangement": True,
                "permutation_sha256": controls.permutation_sha256,
            })
        return chosen, proof

    def _r2(self, predictions: Any, targets: Any) -> float:
        runtime = self._load_runtime()
        torch, R2Score = runtime["torch"], runtime["R2Score"]
        if predictions.ndim != 2 or targets.shape != predictions.shape or predictions.shape[0] == 0:
            raise FailClosedError("TorchMetrics score input shape/cardinality drift")
        metric = R2Score(num_outputs=2, multioutput="variance_weighted")
        with torch.no_grad():
            metric.update(predictions.detach().cpu(), targets.detach().cpu())
            value = metric.compute()
        if not torch.isfinite(value).item():
            raise FailClosedError("TorchMetrics variance-weighted R2 is nonfinite")
        return float(value.item())

    def _assert_tfsr_eval_no_mask(self, model: Any, torch: Any) -> None:
        gain, survivor = model.last_unit_gain_mask, model.last_unit_survivor_mask
        if (model.training or model.capture_diagnostics is not False or model.last_dropout_p is not None
                or gain is None or survivor is None or not torch.equal(gain, torch.ones_like(gain))
                or not bool(survivor.all().item())):
            raise FailClosedError("TF-SR eval/no-mask invariant drift")

    def _forward_tfsr(self, model: Any, neural: Any, calib: Any, capability: Any, *,
                      surface: str, mode: str, flags: ScoreFlags, measure_latency: bool = False) -> Any:
        runtime = self._load_runtime()
        torch = runtime["torch"]
        if model.training or not self._all_gradients_none(model, torch):
            raise FailClosedError("TF-SR scoring requires eval model with all gradients None")
        if measure_latency:
            torch.cuda.synchronize(0)
            start = runtime["time"].perf_counter()
        with torch.no_grad():
            if torch.is_grad_enabled():
                raise FailClosedError("TF-SR scoring autograd boundary drift")
            output = model(neural, calib, capability)
        flags.record_forward("tfsr", surface, mode)
        if measure_latency:
            torch.cuda.synchronize(0)
            self._aligned_latency_ms = (runtime["time"].perf_counter() - start) * 1000.0
        if (not torch.is_tensor(output) or output.shape != (neural.shape[0], 50, 2)
                or not torch.isfinite(output).all().item()):
            raise FailClosedError("TF-SR forward output shape/finite drift")
        self._assert_tfsr_eval_no_mask(model, torch)
        return output

    def _forward_cell_d(self, model: Any, neural: Any, calib: Any, side: Any, *,
                        surface: str, flags: ScoreFlags) -> Any:
        runtime = self._load_runtime()
        torch = runtime["torch"]
        if model.training or not self._all_gradients_none(model, torch):
            raise FailClosedError("Cell-D scoring requires eval model with all gradients None")
        with torch.no_grad():
            if torch.is_grad_enabled():
                raise FailClosedError("Cell-D scoring autograd boundary drift")
            output = model(neural, calib_trials=calib, side_features=side)
        flags.record_forward("cell_d", surface, "aligned")
        raw = output[0] if isinstance(output, tuple) else output
        if (not torch.is_tensor(raw) or raw.shape != (neural.shape[0], 50, 2)
                or not torch.isfinite(raw).all().item()):
            raise FailClosedError("Cell-D forward output shape/finite drift")
        return raw

    def _repeat_tfsr_probe(self, *, session: _PhysicalSession, surface: str, mode: str,
                           model: Any, flags: ScoreFlags) -> bool:
        starts = [int(item) for item in session.record.valid_starts[:PUBLIC_SPEC.batch_size]]
        if not starts:
            raise FailClosedError("TF-SR repeat probe has no query windows")
        neural, _behavior, calib = self._session_batch(session, starts)
        capability, _proof = self._t4_for_batch(session, neural.shape[0], mode)
        first = self._forward_tfsr(model, neural, calib, capability, surface=surface, mode=mode, flags=flags)
        second = self._forward_tfsr(model, neural, calib, capability, surface=surface, mode=mode, flags=flags)
        runtime = self._load_runtime()
        if not runtime["torch"].equal(first, second):
            raise FailClosedError("TF-SR repeated fixed-batch forward is not bitwise equal")
        return True

    def _score_system_surface(
        self,
        *,
        system: str,
        surface: str,
        mode: str,
        model: Any,
        input_authority_sha256: str,
        flags: ScoreFlags,
    ) -> ModeEvidence:
        runtime = self._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        if surface not in self._sessions or not self._sessions[surface]:
            raise FailClosedError("physical scorer has no parsed surface sessions")
        state_before = _physical_model_state_digest(model, torch)
        if state_before != self._state_at_load[system]:
            raise FailClosedError("model state drifted before score surface")
        repeat = False
        first_session = self._sessions[surface][sorted(self._sessions[surface])[0]]
        if system == "tfsr" and mode == "aligned":
            repeat = self._repeat_tfsr_probe(
                session=first_session, surface=surface, mode=mode, model=model, flags=flags,
            )
        session_scores: list[SessionScore] = []
        control_rows: dict[str, Mapping[str, object]] = {}
        output_shape: tuple[int, int, int] | None = None
        for session_name in sorted(self._sessions[surface]):
            session = self._sessions[surface][session_name]
            starts = [int(item) for item in session.record.valid_starts]
            all_last_predictions: list[Any] = []
            all_last_targets: list[Any] = []
            all_full_predictions: list[Any] = []
            all_full_targets: list[Any] = []
            output_digest = hashlib.sha256()
            one_control: Mapping[str, object] | None = None
            for offset in range(0, len(starts), PUBLIC_SPEC.batch_size):
                chunk = starts[offset:offset + PUBLIC_SPEC.batch_size]
                neural, behavior, calib = self._session_batch(session, chunk)
                if system == "cell_d":
                    side = session.aligned_t4.tensor.expand(neural.shape[0], -1, -1).detach().clone()
                    output = self._forward_cell_d(model, neural, calib, side, surface=surface, flags=flags)
                else:
                    capability, control = self._t4_for_batch(session, neural.shape[0], mode)
                    output = self._forward_tfsr(
                        model, neural, calib, capability, surface=surface, mode=mode, flags=flags,
                        measure_latency=(mode == "aligned" and self._aligned_latency_ms is None),
                    )
                    one_control = control
                if output_shape is None:
                    output_shape = tuple(int(item) for item in output.shape)
                output_digest.update(output.detach().cpu().contiguous().numpy().tobytes())
                valid = (behavior != -1.0).all(dim=-1)
                valid_last = valid[:, -1]
                # A governed query is one last bin for *every* authorized
                # 50-bin window.  Filtering invalid entries would silently
                # change the estimator, so require all last-bin targets and
                # their query mask to remain valid before appending them.
                if not bool(valid_last.all().item()) or not bool(valid.any().item()):
                    raise FailClosedError("score batch lacks valid last/full target positions")
                all_last_predictions.append(output[:, -1, :].detach().cpu())
                all_last_targets.append(behavior[:, -1, :].detach().cpu())
                all_full_predictions.append(output[valid].detach().cpu())
                all_full_targets.append(behavior[valid].detach().cpu())
            if one_control is not None:
                control_rows[session_name] = one_control
            last_predictions = torch.cat(all_last_predictions)
            last_targets = torch.cat(all_last_targets)
            target_array = np.ascontiguousarray(last_targets.numpy(), dtype=np.float32)
            mask_array = np.ascontiguousarray(np.all(target_array != -1.0, axis=1), dtype=np.uint8)
            if (
                target_array.shape != session.last_bin_targets.shape
                or not np.array_equal(target_array, session.last_bin_targets)
                or not np.array_equal(mask_array, session.last_bin_valid_mask)
            ):
                raise FailClosedError("physical governing last-bin target/mask session drift")
            # This validation is intentionally in the shared Cell-D/TF-SR
            # scoring core and directly precedes the governing metric.
            _validate_last_bin_authority_arrays(
                np,
                target=target_array,
                valid_mask=mask_array,
                authority=session.input_record,
            )
            session_scores.append(SessionScore(
                session=session_name,
                n_windows=len(starts),
                governing_r2=self._r2(last_predictions, last_targets),
                full_window_r2=self._r2(torch.cat(all_full_predictions), torch.cat(all_full_targets)),
                output_sha256=output_digest.hexdigest(),
                input_authority_sha256=input_authority_sha256,
            ))
        if output_shape is None:
            raise FailClosedError("score surface emitted no model output")
        state_after = _physical_model_state_digest(model, torch)
        if state_before != state_after or not self._all_gradients_none(model, torch):
            raise FailClosedError("model state/gradient drifted during score surface")
        control_payload: Mapping[str, object] | None = None
        if system == "tfsr":
            control_payload = {
                "mode": mode,
                "post_normalization": True,
                "b3s_recomputed": True,
                "per_session_control_sha256": _sha(_json_bytes({key: dict(value) for key, value in sorted(control_rows.items())})),
            }
            if mode == "zero":
                control_payload = {**control_payload, "exact_zeros_like": True}
            if mode == "wrong_pair":
                control_payload = {
                    **control_payload,
                    "permutation_is_derangement": True,
                    "permutation_sha256": _sha(_json_bytes({
                        key: value.get("permutation_sha256") for key, value in sorted(control_rows.items())
                    })),
                }
        return ModeEvidence(
            system=system,
            surface=surface,
            mode=mode,
            sessions=tuple(session_scores),
            state_before_sha256=state_before,
            state_after_sha256=state_after,
            eval_mode=True,
            gradients_none=True,
            finite_output=True,
            output_shape=output_shape,
            repeat_bitwise_equal=(repeat if system == "tfsr" and mode == "aligned" else None),
            eval_no_mask=(True if system == "tfsr" else None),
            capture_diagnostics=(False if system == "tfsr" else None),
            b3s_recomputed=(True if system == "tfsr" else None),
            t4_control=control_payload,
            latency_ms=(self._aligned_latency_ms if system == "tfsr" and mode == "aligned" else None),
            peak_memory_bytes=(int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_available() else None),
        )

    def score_cell_d(self, *, surface: str, input_authority_sha256: str, flags: ScoreFlags) -> ModeEvidence:
        self._ensure_models()
        if self._cell_d is None:
            raise FailClosedError("Cell-D model construction drift")
        return self._score_system_surface(
            system="cell_d", surface=surface, mode="aligned", model=self._cell_d,
            input_authority_sha256=input_authority_sha256, flags=flags,
        )

    def score_tfsr(self, *, surface: str, mode: str, input_authority_sha256: str, flags: ScoreFlags) -> ModeEvidence:
        self._ensure_models()
        if self._tfsr is None:
            raise FailClosedError("TF-SR model construction drift")
        return self._score_system_surface(
            system="tfsr", surface=surface, mode=mode, model=self._tfsr,
            input_authority_sha256=input_authority_sha256, flags=flags,
        )

    def reverify_after_forwards(self, *, flags: ScoreFlags) -> None:
        runtime = self._load_runtime()
        torch = runtime["torch"]
        for held in self._held:
            held.reverify()
        for name, model in (("cell_d", self._cell_d), ("tfsr", self._tfsr)):
            if model is None or _physical_model_state_digest(model, torch) != self._state_at_load.get(name):
                raise FailClosedError("physical model state changed after score forwards")
            if not self._all_gradients_none(model, torch):
                raise FailClosedError("physical model accumulated a gradient during scoring")
        if flags.backward_calls != 0 or flags.optimizer_calls != 0:
            raise FailClosedError("physical scorer recorded an update operation")

    def resource_disclosure(self) -> Mapping[str, object]:
        runtime = self._load_runtime()
        torch = runtime["torch"]
        import torchmetrics
        if self._cell_d is None or self._tfsr is None or self._aligned_latency_ms is None:
            raise FailClosedError("resource disclosure requested before all live scores")
        return _build_physical_resource_disclosure(
            cell_d=self._cell_d,
            tfsr=self._tfsr,
            torch=torch,
            torchmetrics_version=str(torchmetrics.__version__),
            training_peak_memory_bytes=self._training.training_peak_memory_bytes,
            score_peak_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
            aligned_latency_ms=float(self._aligned_latency_ms),
            sealed_cell_d_terminal=self._fixed["cell_d_terminal"].value,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for held in reversed(self._held):
            held.close()
        self._held.clear()


def _validate_mode_against_roster(
    evidence: ModeEvidence,
    *,
    system: str,
    surface: str,
    mode: str,
    expected_sessions: Sequence[str],
    input_authority_sha256: str,
) -> None:
    if evidence.system != system or evidence.surface != surface or evidence.mode != mode:
        raise FailClosedError("backend returned wrong system/surface/mode evidence")
    if tuple(item.session for item in evidence.sessions) != tuple(sorted(expected_sessions)):
        raise FailClosedError("model evidence roster/order drift")
    if any(item.input_authority_sha256 != input_authority_sha256 for item in evidence.sessions):
        raise FailClosedError("model evidence does not bind published input authority")
    if evidence.state_before_sha256 != evidence.state_after_sha256:
        raise FailClosedError("model state changed during no-grad scoring")
    if system == "tfsr":
        control = evidence.t4_control
        if not isinstance(control, Mapping) or control.get("mode") != mode:
            raise FailClosedError("TF-SR typed T4 control evidence mode drift")
        if control.get("post_normalization") is not True or control.get("b3s_recomputed") is not True:
            raise FailClosedError("TF-SR control/B3S recomputation evidence drift")
        if mode == "wrong_pair" and (control.get("permutation_is_derangement") is not True or not _is_sha(control.get("permutation_sha256"))):
            raise FailClosedError("TF-SR wrong-pair permutation proof drift")
        if mode == "zero" and control.get("exact_zeros_like") is not True:
            raise FailClosedError("TF-SR zero control proof drift")


def _surface_map(
    items: Sequence[ModeEvidence], *, system: str, modes: Sequence[str],
) -> dict[str, dict[str, ModeEvidence]]:
    result: dict[str, dict[str, ModeEvidence]] = {"within": {}, "external": {}}
    for evidence in items:
        if evidence.system != system or evidence.surface not in result or evidence.mode not in modes:
            raise FailClosedError("unexpected evidence cell")
        if evidence.mode in result[evidence.surface]:
            raise FailClosedError("duplicate evidence cell")
        result[evidence.surface][evidence.mode] = evidence
    if any(set(result[surface]) != set(modes) for surface in result):
        raise FailClosedError("incomplete evidence matrix")
    return result


def _control_contrast(aligned: ModeEvidence, control: ModeEvidence) -> dict[str, object]:
    aligned_values = {item.session: item.governing_r2 for item in aligned.sessions}
    control_values = {item.session: item.governing_r2 for item in control.sessions}
    sessions = tuple(sorted(aligned_values))
    if tuple(sorted(control_values)) != sessions:
        raise FailClosedError("diagnostic control roster drift")
    statistics = paired_statistics([aligned_values[name] - control_values[name] for name in sessions])
    statistics["non_rescuing"] = True
    statistics["sessions_sorted"] = list(sessions)
    return statistics


def _validate_cell_d_parameter_accounting(value: object) -> dict[str, object]:
    """Validate the separate lazy-safe Cell-D resource disclosure.

    ``cell_d_parameters`` remains the initialized deployable count for compact
    comparison tables.  This nested object prevents the two deliberately dead
    lazy parameters from being silently treated as ordinary zero-sized tensors.
    """
    required = {
        "schema", "count_semantics", "initialized_trainable_parameters",
        "sealed_initialized_trainable_parameters", "uninitialized_lazy_parameter_count",
        "sealed_uninitialized_lazy_parameter_count", "uninitialized_lazy_parameter_keys",
        "uninitialized_lazy_parameter_role",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise FailClosedError("Cell-D parameter-accounting schema drift")
    if value.get("schema") != CELL_D_PARAMETER_ACCOUNTING_SCHEMA:
        raise FailClosedError("Cell-D parameter-accounting schema/version drift")
    if value.get("count_semantics") != "initialized_trainable_parameters_only":
        raise FailClosedError("Cell-D parameter-accounting count semantics drift")
    for key in (
        "initialized_trainable_parameters", "sealed_initialized_trainable_parameters",
        "uninitialized_lazy_parameter_count", "sealed_uninitialized_lazy_parameter_count",
    ):
        if type(value.get(key)) is not int:
            raise FailClosedError("Cell-D parameter-accounting integer drift")
    if (value["initialized_trainable_parameters"] != CELL_D_INITIALIZED_TRAINABLE_PARAMETERS
            or value["sealed_initialized_trainable_parameters"] != CELL_D_INITIALIZED_TRAINABLE_PARAMETERS):
        raise FailClosedError("Cell-D parameter-accounting initialized count drift")
    expected_lazy_count = len(CELL_D_UNINITIALIZED_LAZY_PARAMETER_KEYS)
    if (value["uninitialized_lazy_parameter_count"] != expected_lazy_count
            or value["sealed_uninitialized_lazy_parameter_count"] != expected_lazy_count):
        raise FailClosedError("Cell-D parameter-accounting lazy count drift")
    if value.get("uninitialized_lazy_parameter_keys") != list(CELL_D_UNINITIALIZED_LAZY_PARAMETER_KEYS):
        raise FailClosedError("Cell-D parameter-accounting lazy topology drift")
    if value.get("uninitialized_lazy_parameter_role") != CELL_D_UNINITIALIZED_LAZY_PARAMETER_ROLE:
        raise FailClosedError("Cell-D parameter-accounting lazy role drift")
    return dict(value)


def _validate_resource_disclosure(value: Mapping[str, object]) -> dict[str, object]:
    required = {
        "cell_d_parameters", "cell_d_parameter_accounting", "tfsr_parameters", "tfsr_analytic_macs",
        "persistent_state", "training_peak_memory_bytes", "score_peak_memory_bytes", "aligned_latency_ms",
        "runtime",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise FailClosedError("resource disclosure schema drift")
    cell_d_accounting = _validate_cell_d_parameter_accounting(value.get("cell_d_parameter_accounting"))
    if value.get("cell_d_parameters") != cell_d_accounting["initialized_trainable_parameters"]:
        raise FailClosedError("Cell-D compact/resource parameter count disagreement")
    for key in ("cell_d_parameters", "tfsr_parameters", "tfsr_analytic_macs", "training_peak_memory_bytes", "score_peak_memory_bytes"):
        if type(value.get(key)) is not int or int(value[key]) < 0:
            raise FailClosedError("resource disclosure integer drift")
    if not isinstance(value.get("persistent_state"), Mapping):
        raise FailClosedError("persistent-state disclosure drift")
    if _finite_float(value.get("aligned_latency_ms")) < 0.0:
        raise FailClosedError("latency disclosure drift")
    expected_runtime = {**FROZEN_SCORE_DEVICE, "torchmetrics_version": "1.5.1"}
    if value.get("runtime") != expected_runtime:
        raise FailClosedError("score runtime/device disclosure drift")
    return dict(value)


def build_score_payload(
    *,
    identity: ScoreIdentity,
    input_authority_sha256: str,
    cell_d: Sequence[ModeEvidence],
    tfsr: Sequence[ModeEvidence],
    a2_pooled: Mapping[str, Mapping[str, float]],
    resources: Mapping[str, object],
    flags: ScoreFlags,
) -> dict[str, object]:
    """Assemble the sole scientific receipt after parity and all controls pass."""
    _require_sha(input_authority_sha256, "published input authority")
    d_map = _surface_map(cell_d, system="cell_d", modes=("aligned",))
    t_map = _surface_map(tfsr, system="tfsr", modes=("aligned", "zero", "wrong_pair"))
    contrasts: dict[str, object] = {}
    a2_contrasts: dict[str, object] = {}
    diagnostics: dict[str, object] = {}
    for surface in ("within", "external"):
        d_aligned = d_map[surface]["aligned"]
        tf_aligned = t_map[surface]["aligned"]
        d_values = {item.session: item.governing_r2 for item in d_aligned.sessions}
        tf_values = {item.session: item.governing_r2 for item in tf_aligned.sessions}
        sessions = tuple(sorted(d_values))
        if tuple(sorted(tf_values)) != sessions:
            raise FailClosedError("TF-SR/Cell-D paired roster mismatch")
        paired = paired_statistics([tf_values[name] - d_values[name] for name in sessions])
        paired["sessions_sorted"] = list(sessions)
        contrasts[surface] = paired
        if surface not in a2_pooled:
            raise FailClosedError("sealed A2 pooled surface missing")
        a2_contrasts[surface] = contextual_a2_contrast(tf_aligned, a2_pooled[surface])
        diagnostics[surface] = {
            "aligned_minus_zero": _control_contrast(tf_aligned, t_map[surface]["zero"]),
            "aligned_minus_wrong_pair": _control_contrast(tf_aligned, t_map[surface]["wrong_pair"]),
            "non_rescuing": True,
        }
    if flags.backward_calls != 0 or flags.optimizer_calls != 0:
        raise FailClosedError("scoring lifecycle performed backward/optimizer operation")
    forward_calls = ScoreFlags.validate_forward_calls(flags.forward_calls)
    if any(
        count <= 0
        for surfaces in forward_calls.values()
        for modes in surfaces.values()
        for count in modes.values()
    ):
        raise FailClosedError("successful score has an unobserved system/surface/mode forward")
    if (not flags.within_resolved or not flags.external_resolved
            or not flags.within_opened or not flags.external_opened
            or flags.formal_resolved or flags.formal_opened):
        raise FailClosedError("successful score lacks honest evaluation/formal access evidence")
    payload = {
        "schema": "tfsr_phase_e_matched_score_v1", "cell": CELL,
        "status": "MATCHED_SCORE_COMPLETE", "score_spec": PUBLIC_SPEC.payload(),
        "identity": identity.payload(), "input_authority_sha256": input_authority_sha256,
        "cell_d": {surface: d_map[surface]["aligned"].payload() for surface in ("within", "external")},
        "tfsr": {
            surface: {mode: t_map[surface][mode].payload() for mode in ("aligned", "zero", "wrong_pair")}
            for surface in ("within", "external")
        },
        "tfsr_minus_cell_d": contrasts,
        "a2_contextual": {
            "label": "seed42-versus-A2-three-seed-pooled", "non_gating": True,
            "pooled_per_session": {surface: dict(a2_pooled[surface]) for surface in ("within", "external")},
            "tfsr_minus_a2": a2_contrasts,
        },
        "controls": diagnostics,
        "resources": _validate_resource_disclosure(resources),
        "zero_target_optimizer_backward_update_evidence": {
            # Within/external targets are deliberately opened for a forward
            # evaluation.  This field claims only what Phase-E can honestly
            # prohibit: no update operation and no formal-test resolution.
            "target_optimizer_steps": 0,
            "target_backward_calls": flags.backward_calls,
            "target_update_calls": flags.optimizer_calls,
            "within_readonly_opened": flags.within_opened,
            "external_readonly_opened": flags.external_opened,
            "formal_resolved": flags.formal_resolved,
            "formal_opened": flags.formal_opened,
        },
        "boundaries": flags.payload(),
    }
    validate_score_payload(payload, identity=identity, input_authority_sha256=input_authority_sha256)
    return payload


def validate_score_payload(
    value: Mapping[str, Any],
    *,
    identity: ScoreIdentity | None = None,
    input_authority_sha256: str | None = None,
) -> None:
    """Reconstruct and recompute every persisted non-model score relation."""
    required = {
        "schema", "cell", "status", "score_spec", "identity", "input_authority_sha256", "cell_d", "tfsr",
        "tfsr_minus_cell_d", "a2_contextual", "controls", "resources",
        "zero_target_optimizer_backward_update_evidence", "boundaries",
    }
    if (not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "tfsr_phase_e_matched_score_v1"
            or value.get("cell") != CELL or value.get("status") != "MATCHED_SCORE_COMPLETE"
            or value.get("score_spec") != PUBLIC_SPEC.payload() or not _is_sha(value.get("input_authority_sha256"))):
        raise FailClosedError("score receipt schema/header drift")
    if identity is not None and value.get("identity") != identity.payload():
        raise FailClosedError("score receipt identity drift")
    if input_authority_sha256 is not None and value.get("input_authority_sha256") != input_authority_sha256:
        raise FailClosedError("score receipt input-authority binding drift")

    cell_block = value.get("cell_d")
    tfsr_block = value.get("tfsr")
    if not isinstance(cell_block, Mapping) or set(cell_block) != {"within", "external"}:
        raise FailClosedError("score Cell-D evidence matrix drift")
    if not isinstance(tfsr_block, Mapping) or set(tfsr_block) != {"within", "external"}:
        raise FailClosedError("score TF-SR evidence matrix drift")
    d_map: dict[str, ModeEvidence] = {}
    t_map: dict[str, dict[str, ModeEvidence]] = {}
    for surface, expected_count in (("within", PUBLIC_SPEC.within_count), ("external", PUBLIC_SPEC.external_count)):
        d_value = cell_block.get(surface)
        tf_value = tfsr_block.get(surface)
        if not isinstance(d_value, Mapping) or not isinstance(tf_value, Mapping) or set(tf_value) != {"aligned", "zero", "wrong_pair"}:
            raise FailClosedError("score persisted evidence cell schema drift")
        d = mode_evidence_from_payload(d_value)
        t = {mode: mode_evidence_from_payload(tf_value[mode]) for mode in ("aligned", "zero", "wrong_pair")}
        expected_sessions = tuple(item.session for item in d.sessions)
        _validate_mode_against_roster(
            d, system="cell_d", surface=surface, mode="aligned", expected_sessions=expected_sessions,
            input_authority_sha256=str(value["input_authority_sha256"]),
        )
        if len(expected_sessions) != expected_count:
            raise FailClosedError("score persisted Cell-D roster cardinality drift")
        for mode, evidence in t.items():
            _validate_mode_against_roster(
                evidence, system="tfsr", surface=surface, mode=mode, expected_sessions=expected_sessions,
                input_authority_sha256=str(value["input_authority_sha256"]),
            )
        d_map[surface], t_map[surface] = d, t

    paired = value.get("tfsr_minus_cell_d")
    if not isinstance(paired, Mapping) or set(paired) != {"within", "external"}:
        raise FailClosedError("score paired contrast surfaces drift")
    for surface in ("within", "external"):
        d_values = {item.session: item.governing_r2 for item in d_map[surface].sessions}
        tf_values = {item.session: item.governing_r2 for item in t_map[surface]["aligned"].sessions}
        sessions = tuple(sorted(d_values))
        expected_paired = paired_statistics([tf_values[name] - d_values[name] for name in sessions])
        expected_paired["sessions_sorted"] = list(sessions)
        if paired.get(surface) != expected_paired:
            raise FailClosedError("score paired contrast statistic drift")

    a2 = value.get("a2_contextual")
    if (not isinstance(a2, Mapping) or a2.get("label") != "seed42-versus-A2-three-seed-pooled"
            or a2.get("non_gating") is not True or not isinstance(a2.get("pooled_per_session"), Mapping)
            or not isinstance(a2.get("tfsr_minus_a2"), Mapping)):
        raise FailClosedError("score contextual A2 disclosure drift")
    pooled = a2["pooled_per_session"]
    contrasts = a2["tfsr_minus_a2"]
    if set(pooled) != {"within", "external"} or set(contrasts) != {"within", "external"}:
        raise FailClosedError("score contextual A2 surface drift")
    for surface in ("within", "external"):
        sessions = tuple(item.session for item in d_map[surface].sessions)
        source_values = pooled.get(surface)
        if not isinstance(source_values, Mapping) or set(source_values) != set(sessions):
            raise FailClosedError("score contextual A2 roster drift")
        a2_scores = {session: _finite_float(source_values[session]) for session in sorted(sessions)}
        if contrasts.get(surface) != contextual_a2_contrast(t_map[surface]["aligned"], a2_scores):
            raise FailClosedError("score contextual A2 statistic drift")

    controls = value.get("controls")
    if not isinstance(controls, Mapping) or set(controls) != {"within", "external"}:
        raise FailClosedError("score control contrast surface drift")
    for surface in ("within", "external"):
        expected_controls = {
            "aligned_minus_zero": _control_contrast(t_map[surface]["aligned"], t_map[surface]["zero"]),
            "aligned_minus_wrong_pair": _control_contrast(t_map[surface]["aligned"], t_map[surface]["wrong_pair"]),
            "non_rescuing": True,
        }
        if controls.get(surface) != expected_controls:
            raise FailClosedError("score diagnostic control contrast drift")

    zero_evidence = value.get("zero_target_optimizer_backward_update_evidence")
    if zero_evidence != {
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "within_readonly_opened": True,
        "external_readonly_opened": True,
        "formal_resolved": False,
        "formal_opened": False,
    }:
        raise FailClosedError("score target-update boundary drift")
    boundaries = value.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise FailClosedError("score boundary receipt missing")
    opened = boundaries.get("opened")
    resolved = boundaries.get("resolved")
    if (not isinstance(opened, Mapping) or not isinstance(resolved, Mapping)
            or opened.get("within") is not True or opened.get("external") is not True
            or opened.get("formal") is not False or resolved.get("within") is not True
            or resolved.get("external") is not True or resolved.get("formal") is not False
            or boundaries.get("backward_calls") != 0 or boundaries.get("optimizer_calls") != 0):
        raise FailClosedError("score formal-data boundary drift")
    forward_calls = ScoreFlags.validate_forward_calls(boundaries.get("forward_calls"))
    if any(
        count <= 0
        for surfaces in forward_calls.values()
        for modes in surfaces.values()
        for count in modes.values()
    ):
        raise FailClosedError("score forward accounting is incomplete")
    _validate_resource_disclosure(value.get("resources"))


def _attempt_payload(identity: ScoreIdentity) -> dict[str, object]:
    return {
        "schema": "tfsr_phase_e_score_attempt_v1", "cell": CELL, "phase": PHASE,
        "score_spec": PUBLIC_SPEC.payload(), "identity": identity.payload(),
        "topology": list(SCORE_TOPOLOGY),
        "resolved": {"source": False, "within": False, "external": False, "formal": False},
        "opened": {"source": False, "within": False, "external": False, "formal": False},
        "backward_calls": 0, "optimizer_calls": 0,
    }


def validate_attempt_payload(value: Mapping[str, Any], identity: ScoreIdentity) -> None:
    expected = {
        "schema", "cell", "phase", "score_spec", "identity", "topology", "resolved", "opened",
        "backward_calls", "optimizer_calls",
    }
    if (not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_phase_e_score_attempt_v1"
            or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("score_spec") != PUBLIC_SPEC.payload() or value.get("identity") != identity.payload()
            or value.get("topology") != list(SCORE_TOPOLOGY)
            or value.get("resolved") != {"source": False, "within": False, "external": False, "formal": False}
            or value.get("opened") != {"source": False, "within": False, "external": False, "formal": False}
            or value.get("backward_calls") != 0 or value.get("optimizer_calls") != 0):
        raise FailClosedError("attempt receipt schema/boundary drift")


def _input_authority_payload(evidence: InputAuthorityEvidence, identity: ScoreIdentity) -> dict[str, object]:
    return {
        **evidence.payload(),
        "identity_training_terminal_sha256": identity.training_terminal_sha256,
        "identity_training_swa_sha256": identity.training_swa_sha256,
        "formal_sessions_inert": list(FORMAL_TEST_SESSION_NAMES),
    }


def validate_input_authority_payload(value: Mapping[str, Any], identity: ScoreIdentity) -> None:
    expected = {
        "schema", "cell", "records", "source_normalizer_body_sha256", "t4_normalizer_semantic_sha256",
        "behavior_normalizer_semantic_sha256", "strict_manifest_sha256", "raw_to_normalized_exact",
        "no_cache_readonly_adapter", "identity_training_terminal_sha256", "identity_training_swa_sha256",
        "formal_sessions_inert",
    }
    if (not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_phase_e_input_authority_v1"
            or value.get("cell") != CELL or value.get("identity_training_terminal_sha256") != identity.training_terminal_sha256
            or value.get("identity_training_swa_sha256") != identity.training_swa_sha256
            or value.get("formal_sessions_inert") != list(FORMAL_TEST_SESSION_NAMES)):
        raise FailClosedError("input-authority receipt schema/identity drift")
    # Reconstructing the typed object avoids a weak acceptance of a JSON shape
    # that happens to have the top-level field names but loses a session digest.
    records_value = value.get("records")
    if not isinstance(records_value, list):
        raise FailClosedError("input-authority records missing")
    try:
        records = tuple(SessionInputAuthority(
            session=item["session"], surface=item["surface"], asset=item["asset"],
            ordered_unit_digest=item["ordered_unit_digest"], unit_count=item["unit_count"],
            raw_t4=item["raw_t4"], normalized_t4=item["normalized_t4"],
            calibration_sha256=item["calibration_sha256"], query_start_sha256=item["query_start_sha256"],
            neural_sha256=item["neural_sha256"], behavior_sha256=item["behavior_sha256"],
            valid_mask_sha256=item["valid_mask_sha256"], target_last_bin_sha256=item["target_last_bin_sha256"],
            last_bin_valid_mask_sha256=item["last_bin_valid_mask_sha256"],
            last_bin_valid_count=item["last_bin_valid_count"], n_windows=item["n_windows"],
            raw_t4_sua_axis_proof=item["raw_t4_sua_axis_proof"],
            raw_t4_sua_axis_proof_sha256=item["raw_t4_sua_axis_proof_sha256"],
        ) for item in records_value if isinstance(item, Mapping))
        evidence = InputAuthorityEvidence(
            records=records, source_normalizer_body_sha256=value["source_normalizer_body_sha256"],
            t4_normalizer_semantic_sha256=value["t4_normalizer_semantic_sha256"],
            behavior_normalizer_semantic_sha256=value["behavior_normalizer_semantic_sha256"],
            strict_manifest_sha256=value["strict_manifest_sha256"],
            raw_to_normalized_exact=value["raw_to_normalized_exact"],
            no_cache_readonly_adapter=value["no_cache_readonly_adapter"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FailClosedError("input-authority nested evidence drift") from error
    expected_payload = evidence.payload()
    if len(records) != len(records_value) or any(value.get(key) != expected_payload[key] for key in expected_payload):
        raise FailClosedError("input-authority nested payload roundtrip drift")


def _failure_payload(flags: ScoreFlags) -> dict[str, object]:
    return {
        "schema": "tfsr_phase_e_score_failure_v1", "cell": CELL, "phase": PHASE,
        "stage": flags.stage, **flags.payload(),
        "traceback_sha256": _sha(traceback.format_exc().encode("utf-8")),
    }


def validate_failure_payload(value: Mapping[str, Any]) -> None:
    expected = {
        "schema", "cell", "phase", "stage", "resolved", "opened", "forward_calls",
        "backward_calls", "optimizer_calls", "terminal_published", "traceback_sha256",
    }
    if (not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_phase_e_score_failure_v1"
            or value.get("cell") != CELL or value.get("phase") != PHASE or not isinstance(value.get("stage"), str)
            or value.get("terminal_published") is not False or not _is_sha(value.get("traceback_sha256"))):
        raise FailClosedError("failure receipt header drift")
    for group in ("resolved", "opened", "forward_calls"):
        if not isinstance(value.get(group), Mapping):
            raise FailClosedError("failure receipt group missing")
    if value["resolved"].get("formal") is not False or value["opened"].get("formal") is not False:
        raise FailClosedError("failure receipt formal-boundary drift")
    if type(value.get("backward_calls")) is not int or type(value.get("optimizer_calls")) is not int:
        raise FailClosedError("failure receipt operation count drift")
    ScoreFlags.validate_forward_calls(value["forward_calls"])


def _publish_failure(artifact: ArtifactRoot, flags: ScoreFlags) -> None:
    if flags.terminal_published or artifact.has_name("terminal.json"):
        raise FailClosedError("failure publication forbidden after terminal")
    if artifact.has_name("failure.json"):
        return
    payload = _failure_payload(flags)
    validate_failure_payload(payload)
    digest = artifact.publish_json("failure.json", payload)
    validate_failure_payload(artifact.reload_json("failure.json", digest))


def _terminal_payload(
    *,
    identity: ScoreIdentity,
    attempt_sha256: str,
    input_authority_sha256: str,
    score_sha256: str,
    score_payload: Mapping[str, Any],
    final_closure: Mapping[str, object],
) -> dict[str, object]:
    external = score_payload["tfsr_minus_cell_d"]["external"]
    within = score_payload["tfsr_minus_cell_d"]["within"]
    verdict = decide_verdict(external=external, within=within)
    return {
        "schema": "tfsr_phase_e_score_terminal_v1", "status": "SCORE_COMPLETE", "cell": CELL,
        "phase": PHASE, "score_spec": PUBLIC_SPEC.payload(), "identity": identity.payload(),
        "attempt_sha256": attempt_sha256, "input_authority_sha256": input_authority_sha256,
        "score_sha256": score_sha256, "launch_closure": identity.launch_closure,
        "final_closure": dict(final_closure), "verdict": verdict,
        "verdict_rule": "STOP_then_CLEAR_GO_else_HOLD; controls_and_A2_non_gating",
        "boundaries": {
            "target_optimizer_steps": 0, "backward_calls": 0, "formal_resolved": False,
            "formal_opened": False, "terminal_after_revalidation": True,
        },
    }


def validate_terminal_payload(
    value: Mapping[str, Any],
    identity: ScoreIdentity,
    score_payload: Mapping[str, Any],
    *,
    expected_score_sha: str | None = None,
) -> None:
    expected = {
        "schema", "status", "cell", "phase", "score_spec", "identity", "attempt_sha256",
        "input_authority_sha256", "score_sha256", "launch_closure", "final_closure", "verdict",
        "verdict_rule", "boundaries",
    }
    if (not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_phase_e_score_terminal_v1"
            or value.get("status") != "SCORE_COMPLETE" or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("score_spec") != PUBLIC_SPEC.payload() or value.get("identity") != identity.payload()
            or value.get("launch_closure") != identity.launch_closure
            or value.get("final_closure") != identity.launch_closure
            or value.get("verdict_rule") != "STOP_then_CLEAR_GO_else_HOLD; controls_and_A2_non_gating"):
        raise FailClosedError("terminal receipt schema/closure drift")
    if not all(_is_sha(value.get(key)) for key in ("attempt_sha256", "input_authority_sha256", "score_sha256")):
        raise FailClosedError("terminal artifact SHA drift")
    if expected_score_sha is not None and value.get("score_sha256") != expected_score_sha:
        raise FailClosedError("terminal does not bind exact reloaded score SHA")
    expected_verdict = decide_verdict(
        external=score_payload["tfsr_minus_cell_d"]["external"],
        within=score_payload["tfsr_minus_cell_d"]["within"],
    )
    if value.get("verdict") != expected_verdict:
        raise FailClosedError("terminal verdict does not match predeclared gate")
    if value.get("boundaries") != {
        "target_optimizer_steps": 0, "backward_calls": 0, "formal_resolved": False,
        "formal_opened": False, "terminal_after_revalidation": True,
    }:
        raise FailClosedError("terminal boundary drift")


def run_score_lifecycle(
    *,
    artifact: ArtifactRoot,
    identity: ScoreIdentity,
    execution_capability: ExecutionCapability,
    within_roster: tuple[str, ...],
    external_sessions: tuple[str, ...],
    within_roster_factory: Callable[[], tuple[EvaluationAssetBinding, ...]],
    external_roster_factory: Callable[[], tuple[ExternalAssetBinding, ...]],
    sealed_cell_d_table: Mapping[str, Any],
    a2_pooled: Mapping[str, Mapping[str, float]],
    backend: ScoreBackend,
    final_reverify: Callable[[], Mapping[str, object]],
) -> Mapping[str, Any]:
    """Injected no-partial-result scoring lifecycle.

    The caller must have performed all fixed/training/authorization validation
    and must have reserved a fresh root.  This function starts at durable
    ``attempt.json`` and consequently every later exception is represented by
    an immutable honest failure pair rather than a partial scientific result.
    """
    # A syntactically plausible ScoreIdentity is not authority to score.  The
    # sole production caller obtains this opaque capability by reloading the
    # root-signed preflight/authorization pair; injected tests must make that
    # fact explicit too.
    _require_execution_capability(execution_capability, identity)
    flags = ScoreFlags()
    attempt_written = False
    try:
        flags.stage = "attempt"
        attempt = _attempt_payload(identity)
        validate_attempt_payload(attempt, identity)
        hashes: dict[str, str] = {"attempt.json": artifact.publish_json("attempt.json", attempt)}
        validate_attempt_payload(artifact.reload_json("attempt.json", hashes["attempt.json"]), identity)
        attempt_written = True

        flags.stage = "resolve_inputs"
        # Do not even form local source/within/external path bindings until the
        # immutable attempt has been durably published and reloaded.
        within_assets = within_roster_factory()
        if tuple(item.session for item in within_assets) != tuple(sorted(within_roster)):
            raise FailClosedError("post-attempt within preflight binding roster drift")
        flags.within_resolved = True
        raw_external_roster = external_roster_factory()
        if tuple(item.session for item in raw_external_roster) != external_sessions:
            raise FailClosedError("post-attempt external ledger binding roster drift")
        external_roster = tuple(_as_evaluation_asset(item) for item in raw_external_roster)
        flags.external_resolved = True
        evidence = backend.resolve_inputs(spec=PUBLIC_SPEC, within_roster=within_roster,
                                          within_assets=within_assets,
                                          external_roster=external_roster, flags=flags)
        if flags.formal_resolved or flags.formal_opened or flags.backward_calls or flags.optimizer_calls:
            raise FailClosedError("input adapter crossed forbidden formal/update boundary")
        validate_input_authority_evidence(
            evidence,
            within_roster=within_roster,
            within_assets=within_assets,
            external_roster=external_roster,
        )
        input_payload = _input_authority_payload(evidence, identity)
        validate_input_authority_payload(input_payload, identity)
        hashes["input_authority.json"] = artifact.publish_json("input_authority.json", input_payload)
        validate_input_authority_payload(artifact.reload_json("input_authority.json", hashes["input_authority.json"]), identity)

        cell_d_evidence: list[ModeEvidence] = []
        flags.stage = "cell_d_within"
        for surface, roster in (("within", within_roster), ("external", tuple(item.session for item in external_roster))):
            flags.stage = f"cell_d_{surface}"
            item = backend.score_cell_d(surface=surface, input_authority_sha256=hashes["input_authority.json"], flags=flags)
            _validate_mode_against_roster(item, system="cell_d", surface=surface, mode="aligned",
                                          expected_sessions=roster, input_authority_sha256=hashes["input_authority.json"])
            assert_cell_d_parity(item, sealed_cell_d_table)
            cell_d_evidence.append(item)

        tfsr_evidence: list[ModeEvidence] = []
        for surface, roster in (("within", within_roster), ("external", tuple(item.session for item in external_roster))):
            for mode in ("aligned", "zero", "wrong_pair"):
                flags.stage = f"tfsr_{surface}_{mode}"
                item = backend.score_tfsr(surface=surface, mode=mode,
                                          input_authority_sha256=hashes["input_authority.json"], flags=flags)
                _validate_mode_against_roster(item, system="tfsr", surface=surface, mode=mode,
                                              expected_sessions=roster, input_authority_sha256=hashes["input_authority.json"])
                tfsr_evidence.append(item)
        if flags.formal_resolved or flags.formal_opened or flags.backward_calls or flags.optimizer_calls:
            raise FailClosedError("forward path crossed forbidden formal/update boundary")
        flags.stage = "post_forward_reverify"
        backend.reverify_after_forwards(flags=flags)
        if flags.formal_resolved or flags.formal_opened or flags.backward_calls or flags.optimizer_calls:
            raise FailClosedError("post-forward verifier reports forbidden boundary")

        flags.stage = "score"
        score = build_score_payload(identity=identity, input_authority_sha256=hashes["input_authority.json"],
                                    cell_d=cell_d_evidence, tfsr=tfsr_evidence, a2_pooled=a2_pooled,
                                    resources=backend.resource_disclosure(), flags=flags)
        score_body = _json_bytes(score)
        score_sha = _sha(score_body)
        flags.stage = "terminal_revalidation"
        final_closure = final_reverify()
        if final_closure != identity.launch_closure:
            raise FailClosedError("launch/final closure differs before terminal publication")
        # Revalidate the pre-score immutable pairs before constructing the
        # all-or-nothing score+terminal publication group.  There is
        # deliberately no standalone score publication point.
        validate_attempt_payload(artifact.reload_json("attempt.json", hashes["attempt.json"]), identity)
        validate_input_authority_payload(artifact.reload_json("input_authority.json", hashes["input_authority.json"]), identity)
        validate_score_payload(score, identity=identity, input_authority_sha256=hashes["input_authority.json"])
        terminal = _terminal_payload(identity=identity, attempt_sha256=hashes["attempt.json"],
                                     input_authority_sha256=hashes["input_authority.json"], score_sha256=score_sha,
                                     score_payload=score, final_closure=final_closure)
        validate_terminal_payload(terminal, identity, score)
        terminal_body = _json_bytes(terminal)

        def validate_group(bodies: Mapping[str, bytes], digests: Mapping[str, str]) -> None:
            # This callback runs inside ArtifactRoot.publish_group's rollback
            # envelope.  A malformed body, sidecar, schema, or cross-binding
            # therefore cannot leave score.json without a valid terminal.
            if (bodies.get("score.json") != score_body or bodies.get("terminal.json") != terminal_body
                    or digests.get("score.json") != score_sha):
                raise FailClosedError("score/terminal publication group binding drift")
            try:
                group_score = json.loads(artifact.reload_pair("score.json", score_sha))
                group_terminal = json.loads(artifact.reload_pair("terminal.json", _sha(terminal_body)))
            except (TypeError, json.JSONDecodeError) as error:
                raise FailClosedError("score/terminal publication JSON reload drift") from error
            if not isinstance(group_score, Mapping) or not isinstance(group_terminal, Mapping):
                raise FailClosedError("score/terminal publication JSON root drift")
            validate_score_payload(group_score, identity=identity, input_authority_sha256=hashes["input_authority.json"])
            validate_terminal_payload(group_terminal, identity, group_score, expected_score_sha=score_sha)

        group_hashes = artifact.publish_group(
            {"score.json": score_body, "terminal.json": terminal_body},
            post_publish=validate_group,
        )
        hashes.update(group_hashes)
        flags.terminal_published = True
        final_score = artifact.reload_json("score.json", hashes["score.json"])
        validate_score_payload(final_score, identity=identity, input_authority_sha256=hashes["input_authority.json"])
        reloaded_terminal = artifact.reload_json("terminal.json", hashes["terminal.json"])
        validate_terminal_payload(reloaded_terminal, identity, final_score, expected_score_sha=hashes["score.json"])
        if artifact.has_name("failure.json"):
            raise FailClosedError("terminal cannot coexist with failure receipt")
        return reloaded_terminal
    except BaseException:
        if attempt_written:
            try:
                _publish_failure(artifact, flags)
            except BaseException:
                # The original exception is more useful to the caller.  The
                # artifact publisher itself performs owned rollback; a failed
                # failure receipt can never masquerade as a terminal.
                pass
        raise
    finally:
        backend.close()


def execute_authorized(
    root: Path,
    *,
    training_validator: TrainingTerminalValidator = validate_phase_d_training_terminal,
) -> Mapping[str, Any]:
    """The two-flag route; its default is the reviewed physical backend.

    This function is deliberately unreachable from the CLI unless both flags
    are present.  It still cannot cross the data boundary without a valid
    terminal, a reloaded root authorization pair, a fresh score root, and the
    opaque execution capability created below.
    """
    # Mandatory ordering: these reads are sealed metadata/training artifacts;
    # none discovers an evaluation NWB path or imports a target dataset.
    fixed = verify_fixed_authorities(root)
    training = training_validator(root)
    authorization = verify_phase_e_authorization(root, training, fixed)
    within_roster = extract_within_roster(fixed["strict_manifest"].value or {})
    external_sessions = extract_external_sessions(fixed["external_asset_ledger"].value or {})
    a2_pooled = extract_a2_pooled_tables(fixed["a2_matched_reference"].value or {},
                                         within_sessions=within_roster,
                                         external_sessions=external_sessions)
    parent, name = canonical_score_parent(root)
    artifact = reserve_artifact_root(parent, name)
    identity = ScoreIdentity(
        fixed_authorities={name: material.binding() for name, material in fixed.items()},
        training_terminal_sha256=training.terminal_sha256, training_swa_sha256=training.swa_sha256,
        training_swa_state_digest=training.swa_state_digest, launch_closure=authorization.expected_closure,
        phase_e_authorization=authorization.payload(),
    )
    execution_capability = _issue_execution_capability(authorization)

    def final_reverify() -> Mapping[str, object]:
        checked = verify_fixed_authorities(root)
        if {name: material.binding() for name, material in checked.items()} != identity.fixed_authorities:
            raise FailClosedError("fixed authority identity changed after live forwards")
        final_training = training_validator(root)
        if final_training.payload() != training.payload():
            raise FailClosedError("Phase-D terminal/SWA changed after live forwards")
        final_authorization = verify_phase_e_authorization(root, final_training, checked)
        if final_authorization.payload() != authorization.payload():
            raise FailClosedError("Phase-E authority changed after live forwards")
        return _closure_matches_authorization(root, final_authorization.expected_closure)

    def external_roster_factory() -> tuple[ExternalAssetBinding, ...]:
        # Root audit instruction: use only the fixed v2 ledger join and the
        # canonical SUBM_DATA_ROOT/basename route.  This factory is invoked
        # strictly after durable `attempt.json`, never by a dry plan.
        raw_root = os.environ.get("SUBM_DATA_ROOT")
        if not isinstance(raw_root, str) or not raw_root:
            raise FailClosedError("SUBM_DATA_ROOT is required after durable attempt")
        return join_external_assets(fixed["external_asset_ledger"].value or {},
                                    fixed["external_scope"].value or {}, Path(raw_root))

    def within_roster_factory() -> tuple[EvaluationAssetBinding, ...]:
        # As with sub-M, the preflight's frozen basename/bytes/SHA are the
        # only legal way to address a within-development NWB.  A separate
        # environment root prevents the scorer from rediscovering or scanning
        # sub-C filenames.
        raw_root = os.environ.get("SUBC_DATA_ROOT")
        if not isinstance(raw_root, str) or not raw_root:
            raise FailClosedError("SUBC_DATA_ROOT is required after durable attempt")
        return join_within_assets(authorization.preflight, Path(raw_root))

    return run_score_lifecycle(
        artifact=artifact, identity=identity, execution_capability=execution_capability,
        within_roster=within_roster, external_sessions=external_sessions,
        within_roster_factory=within_roster_factory,
        external_roster_factory=external_roster_factory,
        sealed_cell_d_table=fixed["cell_d_table"].value or {}, a2_pooled=a2_pooled,
        # Deliberately not injectable on the public two-flag route.  Synthetic
        # tests exercise ``run_score_lifecycle`` directly; a production caller
        # cannot substitute a path-discovering, cached, or non-parity backend.
        backend=PhysicalMatchedScoreBackend(
            root=root, fixed_authorities=fixed, training=training, authorization=authorization,
        ),
        final_reverify=final_reverify,
    )
