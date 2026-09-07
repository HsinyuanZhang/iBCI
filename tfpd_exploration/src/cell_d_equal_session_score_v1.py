"""Fail-closed matched scorer for ``CELL_D_EQUAL_SESSION_SEED42``.

The equal-session successor is compared with the sealed Cell-D seed-42 SWA on
the *same* future within/external inputs.  This file deliberately has only
standard-library imports at module import time: a zero-argument CLI can audit
the immutable plan without importing Torch, resolving an NWB pathname,
creating a result directory, or touching CUDA.

This candidate contains both the score contract and a physical, no-cache
adapter.  The adapter is inert until a separate root review supplies an
in-process execution capability: importing this file, calling ``dry_plan``,
or invoking the public CLI never imports Torch, resolves an NWB pathname,
creates an output root, or initialises CUDA.  That makes an unreviewed command
fail before target resolution rather than silently falling back to a mutable
legacy scorer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


CELL = "CELL_D_EQUAL_SESSION_SEED42"
PHASE = "CELL_D_EQUAL_SESSION_MATCHED_SCORE_V1"
SCHEMA = "cell_d_equal_session_matched_score_v1"

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/cell_d_equal_session_seed42_score_authority_v1"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/cell_d_equal_session_seed42_score_v1"
AUTHORITY_TOPOLOGY = ("official_preflight.json", "root_authorization.json")
SCORE_TOPOLOGY = (
    "attempt.json",
    "input_authority.json",
    "score.json",
    "terminal.json",
    "failure.json",
)

# These paths make a future scoring route explicit without granting it any
# discovery privilege.  ``IMPLEMENTATION_CLOSURE`` is intentionally a list,
# never a glob: any direct runtime helper must be named before root review.
STRICT_MANIFEST_RELATIVE = "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
# The strict manifest fixes only the session labels.  The paired-view manifest
# below is the separate immutable byte/file authority for the six within
# assets.  It is mode 0600 and intentionally has no sidecar, so it cannot be
# treated as a generic ``AuthoritySpec``; the dedicated descriptor reader
# below records its exact path, mode, SHA and read-once identity in preflight.
WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE = (
    "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v1/"
    "c1_train_val_33_manifest.json"
)
BASELINE_RECEIPT_RELATIVE = "tfpd_exploration/results/sparsification_score_v1/sparsification_score_receipt.json"
SEALED_CELL_D_TERMINAL_RELATIVE = "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json"
SEALED_CELL_D_SWA_RELATIVE = "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt"
SUCCESSOR_ROOT_RELATIVE = "tfpd_exploration/results/cell_d_equal_session_seed42_v1"
SUCCESSOR_TERMINAL_RELATIVE = f"{SUCCESSOR_ROOT_RELATIVE}/terminal.json"
SUCCESSOR_SWA_RELATIVE = f"{SUCCESSOR_ROOT_RELATIVE}/swa_final4.pt"
SUCCESSOR_SOURCE_AUTHORITY_RELATIVE = f"{SUCCESSOR_ROOT_RELATIVE}/source_authority.json"
EXTERNAL_LEDGER_RELATIVE = "sua_exploration/results/dandi_000688_subm_co_schema_preflight_v2/receipt.json"
EXTERNAL_SCOPE_RELATIVE = "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json"

STRICT_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
WITHIN_PAIRED_VIEW_MANIFEST_SHA256 = "bb3440b688b6d16dabbf91db3ce43e91711f1e9e389de4b241e80a827fbcab7d"
WITHIN_PAIRED_VIEW_MANIFEST_MODE = 0o600
BASELINE_RECEIPT_SHA256 = "583b899bb9e6b132a50b23552d9734b0ccb9b12ecc5c43c27c96ba49bd71980f"
SEALED_CELL_D_TERMINAL_SHA256 = "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442"
SEALED_CELL_D_SWA_SHA256 = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
SUCCESSOR_TERMINAL_SHA256 = "cb0bc529e2db368d968ca758f101d1a31e8c83b8de5f93a76f97f01fa67d80ca"
SUCCESSOR_SWA_SHA256 = "512af5a75c712e675b666c676619297c64b4b21f476d63c694d0a83b70a332ff"
SUCCESSOR_SWA_STATE_SHA256 = "e7f8959d8e51dafd6a8a729f6947cdd5e7d65d775d38f3df0667c082fd60672d"
SUCCESSOR_SOURCE_AUTHORITY_SHA256 = "ca6714629f9226ca5e279b28d355773c06810364985c5728aabf42adecf5ab5d"
EXTERNAL_LEDGER_SHA256 = "1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283"
EXTERNAL_SCOPE_SHA256 = "68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55"

SOURCE_T4_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
SOURCE_BEHAVIOR_NORMALIZER_SHA256 = "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
SOURCE_NORMALIZER_BODY_SHA256 = "2632c6a6a4cfb8a4c0fb2b23e0cc8205ea323240b376903c60d5b27110f59e43"
# Closure-bound float32 source statistics.  They are never inferred/refit on
# within/external data; the exact same arrays are supplied to parser and both
# Cell-D graphs in the future physical scorer.
SOURCE_T4_MEAN = (0.04627712443470955, 0.4544036388397217, 1.3432163000106812, 10.150517463684082)
SOURCE_T4_STD = (1.126278281211853, 1.284820556640625, 1.2352101802825928, 9.115250587463379)
SOURCE_BEHAVIOR_MEAN = (-0.001148765324614942, 0.002653369214385748)
SOURCE_BEHAVIOR_STD = (8.63547420501709, 8.086690902709961)
TORCHMETRICS_VERSION = "1.5.1"

# The metric contract names the shared implementation explicitly.  A future
# physical adapter must import and call this exact function family, rather
# than a window-flattened or sample-weighted replacement.
METRIC_CONTRACT = {
    "estimator": "tfpd_lane.matched_scorer.session_r2",
    "torchmetrics": "torchmetrics.regression.R2Score(multioutput='variance_weighted')",
    "query": "last_bin_only_of_each_valid_50_bin_window",
    "equal_weight_per_session": True,
    "window_bins": 50,
    "mode": "aligned_native_only",
}

FROZEN_GPU0 = {
    "cuda_visible_devices": "0",
    "cuda_device_order": "PCI_BUS_ID",
    "logical_device": "cuda:0",
    "uuid": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
    "bdf": "00000000:01:00.0",
    "name": "NVIDIA GeForce RTX 3090",
    "nvidia_smi_memory_total_mib": 24_576,
    "torch_total_memory_bytes": 25_435_111_424,
    "torch_version": "2.5.1.post303",
    "torch_cuda_version": "11.8",
    "cudnn_version": 90_300,
}

FORMAL_TEST_SESSION_NAMES = (
    "sub-C_ses-CO-20151113", "sub-C_ses-CO-20151116", "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119", "sub-C_ses-CO-20151120", "sub-C_ses-CO-20151201",
)

# Do not infer within files from a caller supplied mapping.  The six rows are
# copied from the paired-view manifest whose exact body SHA is bound above;
# the canonical builder verifies both the manifest semantics and this full
# row table before a durable authority can be made.  ``frozen_path`` retains
# the subject directory as an authority label, while the descriptor-safe live
# adapter resolves only its basename under an independently held SUBC root.
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

IMPLEMENTATION_CLOSURE = (
    "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
    "tfpd_exploration/scripts/run_cell_d_equal_session_seed42_score.py",
    "tfpd_exploration/tests/test_cell_d_equal_session_score_v1.py",
    # The successor SWA validator is invoked through this reviewed route.  Its
    # own explicitly bound handoff/CLI/test are therefore also live closure
    # inputs for this scorer rather than hidden indirect dependencies.
    "tfpd_exploration/docs/HANDOFF_DEPLOYMENT_MATCHED_SESSION_BALANCING_20260820.md",
    "tfpd_exploration/src/cell_d_equal_session_v1.py",
    "tfpd_exploration/scripts/run_cell_d_equal_session_seed42.py",
    "tfpd_exploration/tests/test_cell_d_equal_session_v1.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/datamodule.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
)


class FailClosedError(RuntimeError):
    """A provenance, schema, or access-order violation."""


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


def _finite(value: object, label: str = "finite value") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FailClosedError(f"{label} must be finite")
    return float(value)


def _safe_relative(value: str) -> str:
    if (not isinstance(value, str) or not value or Path(value).is_absolute()
            or ".." in Path(value).parts):
        raise FailClosedError("path must be a safe relative path")
    return value


def _regular_identity(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size)


def _read_fd_all(descriptor: int) -> bytes:
    parts: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(parts)
        parts.append(block)


def _canonical_regular_bytes(root: Path, relative: str, *, expected_mode: int) -> tuple[bytes, tuple[int, int, int]]:
    """Descriptor-read a non-symlink immutable file and prove pathname stability."""
    relative = _safe_relative(relative)
    root = root.absolute()
    try:
        root_info = os.lstat(root)
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            raise FailClosedError("repository root is not a canonical directory")
        candidate = root / relative
        if candidate.resolve(strict=True) != candidate.absolute():
            raise FailClosedError(f"authority alias forbidden: {relative}")
        before = os.lstat(candidate)
    except OSError as error:
        raise FailClosedError(f"cannot lstat authority: {relative}") from error
    if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
            or stat.S_IMODE(before.st_mode) != expected_mode):
        raise FailClosedError(f"authority type/mode drift: {relative}")
    descriptor = -1
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or stat.S_ISLNK(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != expected_mode
                or _regular_identity(before) != _regular_identity(opened)):
            raise FailClosedError(f"authority identity drift before read: {relative}")
        body = _read_fd_all(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        after = os.lstat(candidate)
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
    sidecar_required: bool = True
    json_required: bool = False

    def __post_init__(self) -> None:
        _safe_relative(self.relative)
        _require_sha(self.sha256, self.name)
        if self.mode not in {0o444, 0o664}:
            raise ValueError("authority mode must be exact")


@dataclass(frozen=True)
class AuthorityMaterial:
    spec: AuthoritySpec
    body: bytes
    identity: tuple[int, int, int]
    value: Mapping[str, Any] | None

    def binding(self) -> dict[str, object]:
        return {
            "relative_path": self.spec.relative,
            "body_sha256": self.spec.sha256,
            "mode": format(self.spec.mode, "04o"),
            "descriptor_identity": list(self.identity),
        }


FIXED_AUTHORITIES = (
    AuthoritySpec("strict_manifest", STRICT_MANIFEST_RELATIVE, STRICT_MANIFEST_SHA256, 0o664, False, True),
    AuthoritySpec("last_bin_baseline", BASELINE_RECEIPT_RELATIVE, BASELINE_RECEIPT_SHA256, 0o444, True, True),
    AuthoritySpec("sealed_cell_d_terminal", SEALED_CELL_D_TERMINAL_RELATIVE,
                  SEALED_CELL_D_TERMINAL_SHA256, 0o444, True, True),
    AuthoritySpec("sealed_cell_d_swa", SEALED_CELL_D_SWA_RELATIVE, SEALED_CELL_D_SWA_SHA256, 0o444, True, False),
    AuthoritySpec("successor_terminal", SUCCESSOR_TERMINAL_RELATIVE, SUCCESSOR_TERMINAL_SHA256, 0o444, True, True),
    AuthoritySpec("successor_swa", SUCCESSOR_SWA_RELATIVE, SUCCESSOR_SWA_SHA256, 0o444, True, False),
    AuthoritySpec("successor_source_authority", SUCCESSOR_SOURCE_AUTHORITY_RELATIVE,
                  SUCCESSOR_SOURCE_AUTHORITY_SHA256, 0o444, True, True),
    AuthoritySpec("external_asset_ledger", EXTERNAL_LEDGER_RELATIVE, EXTERNAL_LEDGER_SHA256, 0o444, False, True),
    AuthoritySpec("external_scope", EXTERNAL_SCOPE_RELATIVE, EXTERNAL_SCOPE_SHA256, 0o444, False, True),
)


def verify_authority(root: Path, spec: AuthoritySpec) -> AuthorityMaterial:
    body, identity = _canonical_regular_bytes(root, spec.relative, expected_mode=spec.mode)
    if _sha(body) != spec.sha256:
        raise FailClosedError(f"{spec.name} SHA-256 drift")
    if spec.sidecar_required:
        sidecar, _ = _canonical_regular_bytes(root, f"{spec.relative}.sha256", expected_mode=spec.mode)
        expected = f"{spec.sha256}  {Path(spec.relative).name}\n".encode("ascii")
        if sidecar != expected:
            raise FailClosedError(f"{spec.name} sidecar drift")
    value: Mapping[str, Any] | None = None
    if spec.json_required:
        try:
            decoded = json.loads(body)
        except (TypeError, json.JSONDecodeError) as error:
            raise FailClosedError(f"{spec.name} JSON decode drift") from error
        if not isinstance(decoded, Mapping):
            raise FailClosedError(f"{spec.name} JSON root must be object")
        value = decoded
    return AuthorityMaterial(spec=spec, body=body, identity=identity, value=value)


def verify_fixed_authorities(root: Path) -> dict[str, AuthorityMaterial]:
    return {spec.name: verify_authority(root, spec) for spec in FIXED_AUTHORITIES}


def _closure_file_bytes(root: Path, relative: str) -> bytes:
    # Runtime source is mutable until the launch closure is sealed, but it must
    # still be an exact regular source file and cannot be a symlink.
    return _canonical_regular_bytes(root, relative, expected_mode=0o664)[0]


def implementation_closure(root: Path) -> dict[str, object]:
    hashes = {relative: _sha(_closure_file_bytes(root, relative)) for relative in IMPLEMENTATION_CLOSURE}
    # This mode-0600 no-sidecar manifest is not source code, so it cannot use
    # the generic 0664 source reader.  Its literal immutable binding is part
    # of the closure and the target-free builder additionally descriptor-reads
    # it exactly once before embedding a read identity in preflight.
    body = _json_bytes({
        "sha256_by_path": hashes,
        "immutable_input_bindings": {"within_paired_view_manifest": WITHIN_PAIRED_VIEW_CLOSURE_BINDING},
    })
    return {
        "schema": "cell_d_equal_session_score_implementation_closure_v1",
        "paths": list(IMPLEMENTATION_CLOSURE),
        "sha256_by_path": hashes,
        "immutable_input_bindings": {"within_paired_view_manifest": dict(WITHIN_PAIRED_VIEW_CLOSURE_BINDING)},
        "closure_sha256": _sha(body),
    }


@dataclass(frozen=True)
class ScoreSpec:
    within_sessions: int = 6
    external_sessions: int = 15
    window_bins: int = 50
    batch_size: int = 128

    def __post_init__(self) -> None:
        if (self.within_sessions, self.external_sessions, self.window_bins, self.batch_size) != (6, 15, 50, 128):
            raise ValueError("public score specification is frozen")

    def payload(self) -> dict[str, int]:
        return {
            "within_sessions": self.within_sessions,
            "external_sessions": self.external_sessions,
            "window_bins": self.window_bins,
            "batch_size": self.batch_size,
        }


PUBLIC_SPEC = ScoreSpec()


@dataclass(frozen=True)
class BaselineRow:
    session: str
    n_windows: int
    r2: float

    def __post_init__(self) -> None:
        if not isinstance(self.session, str) or not self.session or type(self.n_windows) is not int or self.n_windows <= 0:
            raise ValueError("baseline session/window schema drift")
        _finite(self.r2, "baseline R2")

    def payload(self) -> dict[str, object]:
        return {"session": self.session, "n_windows": self.n_windows, "r2": self.r2}


@dataclass(frozen=True)
class BaselineAuthority:
    receipt_sha256: str
    within: tuple[BaselineRow, ...]
    external: tuple[BaselineRow, ...]

    def __post_init__(self) -> None:
        _require_sha(self.receipt_sha256, "baseline receipt")
        for surface, rows, expected_count in (
            ("within", self.within, PUBLIC_SPEC.within_sessions),
            ("external", self.external, PUBLIC_SPEC.external_sessions),
        ):
            if len(rows) != expected_count or len({row.session for row in rows}) != expected_count:
                raise ValueError(f"baseline {surface} cardinality drift")
            if tuple(row.session for row in rows) != tuple(sorted(row.session for row in rows)):
                raise ValueError(f"baseline {surface} order must be canonical")

    def rows(self, surface: str) -> tuple[BaselineRow, ...]:
        if surface == "within":
            return self.within
        if surface == "external":
            return self.external
        raise FailClosedError("unknown baseline surface")

    def mean(self, surface: str) -> float:
        rows = self.rows(surface)
        return sum(row.r2 for row in rows) / len(rows)

    def payload(self) -> dict[str, object]:
        return {
            "receipt_sha256": self.receipt_sha256,
            "within": [row.payload() for row in self.within],
            "external": [row.payload() for row in self.external],
            "within_mean": self.mean("within"),
            "external_mean": self.mean("external"),
        }


def _baseline_rows(block: object, *, surface: str, expected_count: int) -> tuple[BaselineRow, ...]:
    if not isinstance(block, Mapping) or not isinstance(block.get("per_session"), list):
        raise FailClosedError(f"baseline governing {surface} schema drift")
    rows: list[BaselineRow] = []
    for item in block["per_session"]:
        if not isinstance(item, Mapping):
            raise FailClosedError(f"baseline {surface} row type drift")
        try:
            row = BaselineRow(session=item["session"], n_windows=item["n_windows"], r2=item["r2"])
        except (KeyError, TypeError, ValueError) as error:
            raise FailClosedError(f"baseline {surface} row drift") from error
        rows.append(row)
    ordered = tuple(rows)
    if len(ordered) != expected_count or tuple(row.session for row in ordered) != tuple(sorted(row.session for row in ordered)):
        raise FailClosedError(f"baseline {surface} roster/order drift")
    mean = sum(row.r2 for row in ordered) / len(ordered)
    if mean != _finite(block.get("mean_r2"), f"baseline {surface} mean"):
        raise FailClosedError(f"baseline {surface} equal-session mean drift")
    return ordered


def load_last_bin_baseline(value: Mapping[str, Any], *, receipt_sha256: str = BASELINE_RECEIPT_SHA256) -> BaselineAuthority:
    """Parse exactly ``results.D_swa.governing_last_bin``—never a legacy table."""
    if not isinstance(value, Mapping):
        raise FailClosedError("baseline receipt root drift")
    if value.get("schema") != "tfpd_sparsification_score_v1" or value.get("status") != "SPARSIFICATION_SCORED":
        raise FailClosedError("baseline receipt is not the sealed sparsification authority")
    results = value.get("results")
    d_swa = results.get("D_swa") if isinstance(results, Mapping) else None
    governing = d_swa.get("governing_last_bin") if isinstance(d_swa, Mapping) else None
    if not isinstance(governing, Mapping) or set(governing) != {"within", "external"}:
        raise FailClosedError("baseline requires D_swa.governing_last_bin only")
    authority = BaselineAuthority(
        receipt_sha256=receipt_sha256,
        within=_baseline_rows(governing["within"], surface="within", expected_count=PUBLIC_SPEC.within_sessions),
        external=_baseline_rows(governing["external"], surface="external", expected_count=PUBLIC_SPEC.external_sessions),
    )
    if authority.mean("within") != 0.5696851710478464 or authority.mean("external") != 0.4179362749059995:
        raise FailClosedError("sealed Cell-D governing means drift")
    return authority


def extract_within_roster(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    splits = manifest.get("session_splits") if isinstance(manifest, Mapping) else None
    if not isinstance(splits, Mapping) or set(splits) != {"train", "val", "test"}:
        raise FailClosedError("strict manifest split schema drift")
    val = splits.get("val")
    if (not isinstance(val, list) or len(val) != PUBLIC_SPEC.within_sessions
            or any(not isinstance(item, str) or not item for item in val)
            or len(set(val)) != len(val) or tuple(val) != tuple(sorted(val))):
        raise FailClosedError("strict manifest val[6] roster/order drift")
    if any(item in FORMAL_TEST_SESSION_NAMES for item in val):
        raise FailClosedError("formal session appears in strict within roster")
    return tuple(val)


def _unique_rows(rows: object, *, key: str, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise FailClosedError(f"{label} must be list")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get(key), str) or not row[key]:
            raise FailClosedError(f"{label} row/key drift")
        rendered = str(row[key])
        if rendered in result:
            raise FailClosedError(f"{label} duplicate {key}")
        result[rendered] = row
    return result


def extract_external_roster(ledger: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve UUID ``eligible_session_ids`` to canonical sub-M sessions."""
    if not isinstance(ledger, Mapping):
        raise FailClosedError("external ledger root drift")
    by_asset = _unique_rows(ledger.get("asset_disposition_ledger"), key="asset_id", label="asset ledger")
    eligible = ledger.get("eligible_session_ids")
    if (not isinstance(eligible, list) or len(eligible) != PUBLIC_SPEC.external_sessions
            or ledger.get("eligible_session_count") != PUBLIC_SPEC.external_sessions
            or any(not isinstance(item, str) or not item for item in eligible)
            or len(set(eligible)) != len(eligible)):
        raise FailClosedError("external eligible UUID topology drift")
    sessions: list[str] = []
    for asset_id in eligible:
        row = by_asset.get(asset_id)
        if row is None or row.get("asset_id") != asset_id or row.get("eligible") is not True or row.get("disposition") != "ELIGIBLE":
            raise FailClosedError("external eligible UUID ledger semantics drift")
        session = row.get("session_id")
        if not isinstance(session, str) or not session or session in FORMAL_TEST_SESSION_NAMES:
            raise FailClosedError("external canonical session/final boundary drift")
        sessions.append(session)
    if len(set(sessions)) != PUBLIC_SPEC.external_sessions:
        raise FailClosedError("external UUID mapping produced duplicate sessions")
    return tuple(sorted(sessions))


def validate_baseline_rosters(
    baseline: BaselineAuthority, *, within_roster: Sequence[str], external_roster: Sequence[str],
) -> None:
    if tuple(within_roster) != tuple(row.session for row in baseline.within):
        raise FailClosedError("strict within roster differs from sealed last-bin baseline")
    if tuple(external_roster) != tuple(row.session for row in baseline.external):
        raise FailClosedError("fixed-ledger external roster differs from sealed last-bin baseline")


def validate_sealed_lineage(fixed: Mapping[str, AuthorityMaterial]) -> dict[str, object]:
    """Validate complete metadata lineage before any evaluation pathname exists."""
    if set(fixed) != {spec.name for spec in FIXED_AUTHORITIES}:
        raise FailClosedError("fixed authority set drift")
    baseline = load_last_bin_baseline(fixed["last_bin_baseline"].value or {}, receipt_sha256=BASELINE_RECEIPT_SHA256)
    manifest = fixed["strict_manifest"].value or {}
    within = extract_within_roster(manifest)
    external = extract_external_roster(fixed["external_asset_ledger"].value or {})
    validate_baseline_rosters(baseline, within_roster=within, external_roster=external)

    d_terminal = fixed["sealed_cell_d_terminal"].value or {}
    d_swa = d_terminal.get("swa") if isinstance(d_terminal, Mapping) else None
    if (d_terminal.get("schema") != "tfpd_pop_robust_cell_v1" or d_terminal.get("status") != "CELL_TERMINAL"
            or d_terminal.get("cell") != "D" or not isinstance(d_swa, Mapping)
            or d_swa.get("sha256") != SEALED_CELL_D_SWA_SHA256
            or d_swa.get("window_epochs") != [44, 45, 46, 47]
            or d_swa.get("strict_reload_finite_forward_smoke") is not True):
        raise FailClosedError("sealed Cell-D terminal/SWA metadata drift")
    d_manifest = d_swa.get("manifest")
    if (not isinstance(d_manifest, Mapping) or d_manifest.get("strict_reload_verified") is not True
            or d_manifest.get("uninitialized_lazy_tensor_count") != 2
            or d_manifest.get("optimizer_state_included") is not False):
        raise FailClosedError("sealed Cell-D SWA lazy/strict metadata drift")

    terminal = fixed["successor_terminal"].value or {}
    detail = terminal.get("terminal_detail") if isinstance(terminal, Mapping) else None
    artifacts = terminal.get("artifact_sha256s") if isinstance(terminal, Mapping) else None
    if (terminal.get("schema") != "cell_d_equal_session_terminal_v1"
            or terminal.get("status") != "CELL_TRAINING_TERMINAL__UNSCORED"
            or terminal.get("cell") != CELL
            or terminal.get("target_score_or_evaluation_performed") is not False
            or terminal.get("launch_final_closure_equal") is not True
            or not isinstance(detail, Mapping) or not isinstance(artifacts, Mapping)
            or artifacts.get("swa_final4.pt") != SUCCESSOR_SWA_SHA256
            or detail.get("swa_state_sha256") != SUCCESSOR_SWA_STATE_SHA256):
        raise FailClosedError("equal-session successor terminal/SWA lineage drift")
    proof = detail.get("swa_proof")
    if (not isinstance(proof, Mapping) or proof.get("window_epochs") != [44, 45, 46, 47]
            or proof.get("fresh_strict_load") is not True or proof.get("eval_mode") is not True
            or proof.get("eval_no_mask") is not True or proof.get("repeat_bitwise_equal") is not True
            or proof.get("state_unchanged") is not True
            or proof.get("state_sha256_before_eval") != SUCCESSOR_SWA_STATE_SHA256
            or proof.get("state_sha256_after_eval") != SUCCESSOR_SWA_STATE_SHA256
            or proof.get("uninitialized_lazy_keys") != ["decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight"]):
        raise FailClosedError("equal-session successor SWA proof drift")
    run_spec = terminal.get("run_spec")
    if (not isinstance(run_spec, Mapping) or run_spec.get("epochs") != 48
            or run_spec.get("steps_per_epoch") != 33_925 or run_spec.get("total_steps") != 1_628_400
            or run_spec.get("checkpoint_epochs") != [44, 45, 46, 47]):
        raise FailClosedError("equal-session successor full-run topology drift")
    source = fixed["successor_source_authority"].value or {}
    normalizers = source.get("normalizers") if isinstance(source, Mapping) else None
    if (source.get("schema") != "cell_d_equal_session_source_authority_v1" or not isinstance(normalizers, Mapping)
            or normalizers.get("t4_semantic_sha256") != SOURCE_T4_NORMALIZER_SHA256
            or normalizers.get("behavior_semantic_sha256") != SOURCE_BEHAVIOR_NORMALIZER_SHA256):
        raise FailClosedError("equal-session successor source-normalizer lineage drift")
    return {
        "baseline": baseline,
        "within_roster": within,
        "external_roster": external,
        "successor_terminal_sha256": SUCCESSOR_TERMINAL_SHA256,
        "successor_swa_sha256": SUCCESSOR_SWA_SHA256,
        "successor_swa_state_sha256": SUCCESSOR_SWA_STATE_SHA256,
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
    remainder = memoryview(body)
    while remainder:
        count = os.write(descriptor, remainder)
        if count <= 0:
            raise OSError("short artifact write")
        remainder = remainder[count:]


@dataclass(frozen=True)
class ArtifactRoot:
    """A held named-directory capability for receipt publication."""

    directory: Path
    topology: tuple[str, ...]
    identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]

    def _assert_named_identity(self) -> None:
        if _directory_identity(self.directory) != self.identity:
            raise FailClosedError("artifact root path identity drift")
        descriptor = -1
        try:
            descriptor = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != self.parent_identity:
                raise FailClosedError("artifact parent identity drift")
            named = os.stat(self.directory.name, dir_fd=descriptor, follow_symlinks=False)
            if (not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode)
                    or (named.st_dev, named.st_ino) != self.identity):
                raise FailClosedError("artifact root named identity drift")
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or name not in self.topology or "/" in name or name in {"", ".", ".."}:
            raise FailClosedError("artifact name outside exact topology")

    @staticmethod
    def _exists(descriptor: int, leaf: str) -> bool:
        try:
            os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def has_name(self, name: str) -> bool:
        self._check_name(name)
        self._assert_named_identity()
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            return self._exists(descriptor, name)
        finally:
            os.close(descriptor)

    def _read_leaf(self, descriptor: int, leaf: str) -> bytes:
        try:
            fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        except OSError as error:
            raise FailClosedError(f"missing/corrupt artifact leaf: {leaf}") from error
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                raise FailClosedError("artifact type/mode drift")
            return _read_fd_all(fd)
        finally:
            os.close(fd)

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        self._check_name(name)
        self._assert_named_identity()
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            body = self._read_leaf(descriptor, name)
            digest = _sha(body)
            if expected_sha256 is not None and digest != expected_sha256:
                raise FailClosedError("artifact body SHA drift")
            if self._read_leaf(descriptor, name + ".sha256") != f"{digest}  {name}\n".encode("ascii"):
                raise FailClosedError("artifact sidecar drift")
        finally:
            os.close(descriptor)
        self._assert_named_identity()
        return body

    def reload_json(self, name: str, expected_sha256: str | None = None) -> Mapping[str, Any]:
        try:
            value = json.loads(self.reload_pair(name, expected_sha256))
        except (TypeError, json.JSONDecodeError) as error:
            raise FailClosedError("artifact JSON decode drift") from error
        if not isinstance(value, Mapping):
            raise FailClosedError("artifact JSON root must be object")
        return value

    def publish_json(self, name: str, payload: Mapping[str, Any]) -> str:
        return self.publish_group({name: _json_bytes(payload)})[name]

    def publish_group(
        self,
        bodies: Mapping[str, bytes],
        *,
        post_publish: Callable[[Mapping[str, bytes], Mapping[str, str]], None] | None = None,
    ) -> dict[str, str]:
        """O_EXCL/fsync a receipt group; rollback all owned leaves on any error."""
        if not isinstance(bodies, Mapping) or not bodies:
            raise TypeError("artifact group must be nonempty mapping")
        names = tuple(sorted(bodies))
        if len(set(names)) != len(names):
            raise FailClosedError("duplicate artifact group name")
        for name in names:
            self._check_name(name)
            if not isinstance(bodies[name], bytes):
                raise TypeError("artifact body must be bytes")
        self._assert_named_identity()
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        created: list[tuple[str, int, int]] = []
        try:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != self.identity:
                raise FailClosedError("artifact root changed before publication")
            for name in names:
                if self._exists(descriptor, name) or self._exists(descriptor, name + ".sha256"):
                    raise FailClosedError("artifact output collision")
            digests = {name: _sha(bodies[name]) for name in names}
            leaves: list[tuple[str, bytes]] = []
            for name in names:
                leaves.extend(((name, bodies[name]), (name + ".sha256", f"{digests[name]}  {name}\n".encode("ascii"))))
            for leaf, body in leaves:
                fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
                try:
                    made = os.fstat(fd)
                    if not stat.S_ISREG(made.st_mode):
                        raise FailClosedError("artifact O_EXCL did not make regular leaf")
                    created.append((leaf, made.st_dev, made.st_ino))
                    _write_full(fd, body)
                    os.fchmod(fd, 0o444)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(descriptor)
            for name in names:
                if self._read_leaf(descriptor, name) != bodies[name]:
                    raise FailClosedError("artifact post-write body drift")
                if self._read_leaf(descriptor, name + ".sha256") != f"{digests[name]}  {name}\n".encode("ascii"):
                    raise FailClosedError("artifact post-write sidecar drift")
            if post_publish is not None:
                post_publish(dict(bodies), dict(digests))
            self._assert_named_identity()
            return digests
        except BaseException:
            for leaf, device, inode in reversed(created):
                try:
                    current = os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == (device, inode):
                        os.unlink(leaf, dir_fd=descriptor)
                except OSError:
                    pass
            try:
                os.fsync(descriptor)
            except OSError:
                pass
            raise
        finally:
            os.close(descriptor)


def reserve_artifact_root(parent: Path, name: str, *, topology: tuple[str, ...]) -> ArtifactRoot:
    if (not isinstance(name, str) or not name or "/" in name or name in {".", ".."}
            or not topology or len(set(topology)) != len(topology)):
        raise ValueError("artifact root/topology schema drift")
    parent = parent.absolute()
    parent_identity = _directory_identity(parent)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != parent_identity:
            raise FailClosedError("artifact parent identity drift before reserve")
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FailClosedError("fresh canonical artifact root required")
        os.mkdir(name, 0o755, dir_fd=descriptor)
        os.fsync(descriptor)
        created = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(created.st_mode) or stat.S_ISLNK(created.st_mode):
            raise FailClosedError("reserved artifact root is not canonical")
        identity = (created.st_dev, created.st_ino)
    finally:
        os.close(descriptor)
    return ArtifactRoot(parent / name, topology, identity, parent, parent_identity)


def _canonical_parent_and_name(root: Path, relative: str) -> tuple[Path, str]:
    relative = _safe_relative(relative)
    target = root.absolute() / relative
    return target.parent, target.name


def canonical_authority_parent(root: Path) -> tuple[Path, str]:
    return _canonical_parent_and_name(root, AUTHORITY_ROOT_RELATIVE)


def canonical_score_parent(root: Path) -> tuple[Path, str]:
    return _canonical_parent_and_name(root, SCORE_ROOT_RELATIVE)


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        copied = json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise FailClosedError("payload is not canonical-JSON serializable") from error
    if not isinstance(copied, dict):
        raise FailClosedError("payload JSON copy root drift")
    return copied


@dataclass(frozen=True)
class ScoreIdentity:
    """All immutable inputs that a score and its terminal must bind."""

    fixed_authorities: Mapping[str, Mapping[str, object]]
    closure: Mapping[str, object]
    baseline_receipt_sha256: str
    sealed_cell_d_terminal_sha256: str
    sealed_cell_d_swa_sha256: str
    successor_terminal_sha256: str
    successor_swa_sha256: str
    successor_swa_state_sha256: str
    preflight_sha256: str
    root_authorization_sha256: str

    def __post_init__(self) -> None:
        if set(self.fixed_authorities) != {spec.name for spec in FIXED_AUTHORITIES}:
            raise ValueError("score identity fixed authority set drift")
        for name, binding in self.fixed_authorities.items():
            if not isinstance(binding, Mapping) or binding.get("body_sha256") != next(
                spec.sha256 for spec in FIXED_AUTHORITIES if spec.name == name
            ):
                raise ValueError("score identity fixed authority binding drift")
        for label, value in (
            ("baseline receipt", self.baseline_receipt_sha256),
            ("sealed Cell-D terminal", self.sealed_cell_d_terminal_sha256),
            ("sealed Cell-D SWA", self.sealed_cell_d_swa_sha256),
            ("successor terminal", self.successor_terminal_sha256),
            ("successor SWA", self.successor_swa_sha256),
            ("successor state", self.successor_swa_state_sha256),
            ("preflight", self.preflight_sha256),
            ("root authorization", self.root_authorization_sha256),
        ):
            _require_sha(value, label)
        if not isinstance(self.closure, Mapping):
            raise ValueError("score identity closure must be mapping")

    def payload(self) -> dict[str, object]:
        return {
            "fixed_authorities": _json_copy(self.fixed_authorities),
            "closure": _json_copy(self.closure),
            "baseline_receipt_sha256": self.baseline_receipt_sha256,
            "sealed_cell_d_terminal_sha256": self.sealed_cell_d_terminal_sha256,
            "sealed_cell_d_swa_sha256": self.sealed_cell_d_swa_sha256,
            "successor_terminal_sha256": self.successor_terminal_sha256,
            "successor_swa_sha256": self.successor_swa_sha256,
            "successor_swa_state_sha256": self.successor_swa_state_sha256,
            "preflight_sha256": self.preflight_sha256,
            "root_authorization_sha256": self.root_authorization_sha256,
        }


def _validate_closure(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FailClosedError("implementation closure missing")
    expected = {"schema", "paths", "sha256_by_path", "immutable_input_bindings", "closure_sha256"}
    if set(value) != expected or value.get("schema") != "cell_d_equal_session_score_implementation_closure_v1":
        raise FailClosedError("implementation closure schema drift")
    paths = value.get("paths")
    hashes = value.get("sha256_by_path")
    if (paths != list(IMPLEMENTATION_CLOSURE) or not isinstance(hashes, Mapping)
            or set(hashes) != set(IMPLEMENTATION_CLOSURE)
            or any(not _is_sha(item) for item in hashes.values())):
        raise FailClosedError("implementation closure path/hash drift")
    immutable_inputs = value.get("immutable_input_bindings")
    expected_inputs = {"within_paired_view_manifest": WITHIN_PAIRED_VIEW_CLOSURE_BINDING}
    if immutable_inputs != expected_inputs:
        raise FailClosedError("implementation closure immutable input binding drift")
    expected_digest = _sha(_json_bytes({
        "sha256_by_path": dict(hashes),
        "immutable_input_bindings": expected_inputs,
    }))
    if value.get("closure_sha256") != expected_digest:
        raise FailClosedError("implementation closure digest drift")
    return _json_copy(value)


@dataclass(frozen=True)
class PreflightAsset:
    """Metadata-only target asset permission; no pathname is resolved here."""

    surface: str
    asset_id: str
    session: str
    frozen_path: str
    expected_bytes: int
    expected_sha256: str

    def __post_init__(self) -> None:
        if self.surface not in {"within", "external"}:
            raise ValueError("preflight asset surface drift")
        if not all(isinstance(value, str) and value for value in (self.asset_id, self.session, self.frozen_path)):
            raise ValueError("preflight asset string drift")
        _safe_relative(self.frozen_path)
        if type(self.expected_bytes) is not int or self.expected_bytes <= 0:
            raise ValueError("preflight asset byte count drift")
        _require_sha(self.expected_sha256, "preflight asset SHA")

    def payload(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "asset_id": self.asset_id,
            "session": self.session,
            "frozen_path": self.frozen_path,
            "bytes": self.expected_bytes,
            "sha256": self.expected_sha256,
        }


def _preflight_assets_from_payload(value: object, *, surface: str, roster: Sequence[str]) -> tuple[PreflightAsset, ...]:
    if not isinstance(value, list):
        raise FailClosedError(f"preflight {surface} assets must be list")
    assets: list[PreflightAsset] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise FailClosedError(f"preflight {surface} asset row type drift")
        try:
            asset = PreflightAsset(
                surface=item["surface"], asset_id=item["asset_id"], session=item["session"],
                frozen_path=item["frozen_path"], expected_bytes=item["bytes"], expected_sha256=item["sha256"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FailClosedError(f"preflight {surface} asset row drift") from error
        assets.append(asset)
    if (len(assets) != len(roster) or tuple(asset.session for asset in assets) != tuple(roster)
            or any(asset.surface != surface for asset in assets)
            or len({asset.asset_id for asset in assets}) != len(assets)):
        raise FailClosedError(f"preflight {surface} asset roster/order drift")
    return tuple(assets)


def sealed_within_assets() -> tuple[PreflightAsset, ...]:
    """Return the six within rows frozen by the paired-view C1 manifest.

    This table is deliberately not caller configurable.  The canonical
    target-free builder descriptor-reads the manifest and proves it still
    yields these six exact rows; validators compare every persisted row again
    so a forged authority cannot substitute a different within file mapping.
    """
    assets: list[PreflightAsset] = []
    for session, frozen_path, expected_bytes, expected_sha256 in SEALED_WITHIN_PAIRED_VIEW_ROWS:
        assets.append(PreflightAsset(
            surface="within", asset_id=f"c1_paired_view:{session}", session=session,
            frozen_path=frozen_path, expected_bytes=expected_bytes, expected_sha256=expected_sha256,
        ))
    return tuple(assets)


@dataclass(frozen=True)
class WithinPairedViewManifestBinding:
    """One descriptor-read snapshot of the non-sidecar paired-view manifest."""

    relative_path: str
    body_sha256: str
    mode: str
    descriptor_identity: tuple[int, int, int]
    selected_split: str
    read_once: bool

    def __post_init__(self) -> None:
        if (self.relative_path != WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE
                or self.body_sha256 != WITHIN_PAIRED_VIEW_MANIFEST_SHA256
                or self.mode != format(WITHIN_PAIRED_VIEW_MANIFEST_MODE, "04o")
                or self.selected_split != "val" or self.read_once is not True
                or not isinstance(self.descriptor_identity, tuple) or len(self.descriptor_identity) != 3
                or any(type(item) is not int or item < 0 for item in self.descriptor_identity)):
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
            relative_path=value["relative_path"], body_sha256=value["body_sha256"], mode=value["mode"],
            descriptor_identity=tuple(value["descriptor_identity"]), selected_split=value["selected_split"],
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
    root: Path, *, within_roster: Sequence[str],
) -> tuple[tuple[PreflightAsset, ...], WithinPairedViewManifestBinding]:
    """Descriptor-read the only accepted within-6 asset authority.

    The mode-0600 manifest has no immutable sidecar.  This function therefore
    makes its exact SHA, lstat/open identity, selected split and canonical six
    row semantics durable in the target-free preflight.  It never forms an NWB
    path or opens a data asset.
    """
    body, identity = _canonical_regular_bytes(
        root, WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE, expected_mode=WITHIN_PAIRED_VIEW_MANIFEST_MODE,
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
    expected_sessions = tuple(row[0] for row in SEALED_WITHIN_PAIRED_VIEW_ROWS)
    if (manifest.get("schema_version") != 1 or manifest.get("task") != "CO"
            or manifest.get("file_count") != 33 or manifest.get("split_counts") != [27, 6, 6]
            or manifest.get("max_units_exclusive") != 100
            or manifest.get("source_manifest_sha256") != STRICT_MANIFEST_SHA256
            or manifest.get("formal_test_file_hashes") != [] or manifest.get("formal_test_file_paths") != []
            or manifest.get("formal_test_paths_resolved") is not False
            or not isinstance(splits, Mapping) or set(splits) != {"train", "val", "test"}
            or not isinstance(inventory, Mapping) or set(inventory) != {"train", "val"}
            or not isinstance(splits.get("train"), list) or len(splits["train"]) != 27
            or tuple(splits.get("val", ())) != expected_sessions
            or tuple(splits.get("test", ())) != FORMAL_TEST_SESSION_NAMES
            or not isinstance(inventory.get("train"), list) or len(inventory["train"]) != 27
            or inventory.get("val") != _expected_paired_view_val_rows()):
        raise FailClosedError("within paired-view manifest semantic row drift")
    assets = sealed_within_assets()
    if tuple(within_roster) != expected_sessions or tuple(asset.session for asset in assets) != expected_sessions:
        raise FailClosedError("within paired-view manifest/strict roster drift")
    return assets, WithinPairedViewManifestBinding(
        relative_path=WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE,
        body_sha256=WITHIN_PAIRED_VIEW_MANIFEST_SHA256,
        mode=format(WITHIN_PAIRED_VIEW_MANIFEST_MODE, "04o"),
        descriptor_identity=identity,
        selected_split="val",
        read_once=True,
    )


def external_assets_from_ledger(ledger: Mapping[str, Any], scope: Mapping[str, Any]) -> tuple[PreflightAsset, ...]:
    """Join fixed v2 UUID ledger rows to verified bytes and scope rows.

    This is intentionally metadata-only.  The later physical adapter may form
    ``SUBM_DATA_ROOT / basename(frozen_path)`` only after durable attempt and
    an already-reviewed preflight bind every row below.
    """
    if not isinstance(ledger, Mapping) or not isinstance(scope, Mapping):
        raise FailClosedError("external fixed authorities must be mappings")
    ledger_rows = _unique_rows(ledger.get("asset_disposition_ledger"), key="asset_id", label="asset ledger")
    downloads = _unique_rows(ledger.get("verified_downloads"), key="asset_id", label="verified downloads")
    selected = _unique_rows(scope.get("selected_assets"), key="asset_id", label="scope assets")
    eligible_ids = ledger.get("eligible_session_ids")
    if not isinstance(eligible_ids, list) or len(eligible_ids) != PUBLIC_SPEC.external_sessions:
        raise FailClosedError("external eligible UUID cardinality drift")
    assets: list[PreflightAsset] = []
    for asset_id in eligible_ids:
        row = ledger_rows.get(asset_id) if isinstance(asset_id, str) else None
        verified = downloads.get(asset_id) if isinstance(asset_id, str) else None
        scoped = selected.get(asset_id) if isinstance(asset_id, str) else None
        if row is None or verified is None or scoped is None:
            raise FailClosedError("external UUID/ledger/download/scope join drift")
        session, frozen_path = row.get("session_id"), row.get("frozen_path")
        expected_bytes, expected_sha = verified.get("bytes"), verified.get("sha256")
        if (row.get("eligible") is not True or row.get("disposition") != "ELIGIBLE"
                or not isinstance(session, str) or not isinstance(frozen_path, str)
                or row.get("asset_id") != asset_id or scoped.get("session_id") != session
                or scoped.get("path") != frozen_path or verified.get("asset_id") != asset_id
                or scoped.get("size") != expected_bytes or scoped.get("sha256") != expected_sha
                or verified.get("size_and_sha256_verified_before_nwb_open") is not True):
            raise FailClosedError("external ledger integrity semantics drift")
        try:
            assets.append(PreflightAsset(
                surface="external", asset_id=asset_id, session=session, frozen_path=frozen_path,
                expected_bytes=expected_bytes, expected_sha256=expected_sha,
            ))
        except (TypeError, ValueError) as error:
            raise FailClosedError("external ledger asset binding drift") from error
    ordered = tuple(sorted(assets, key=lambda item: item.session))
    if len({item.session for item in ordered}) != PUBLIC_SPEC.external_sessions:
        raise FailClosedError("external ledger duplicate session drift")
    return ordered


def _normalizer_payload(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "source_authority_sha256", "source_normalizer_body_sha256",
        "t4_semantic_sha256", "behavior_semantic_sha256", "models_share_exact_normalizers",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("source_authority_sha256") != SUCCESSOR_SOURCE_AUTHORITY_SHA256
            or value.get("source_normalizer_body_sha256") != SOURCE_NORMALIZER_BODY_SHA256
            or value.get("t4_semantic_sha256") != SOURCE_T4_NORMALIZER_SHA256
            or value.get("behavior_semantic_sha256") != SOURCE_BEHAVIOR_NORMALIZER_SHA256
            or value.get("models_share_exact_normalizers") is not True):
        raise FailClosedError("source normalizer binding drift")
    return _json_copy(value)


def public_normalizer_payload() -> dict[str, object]:
    return {
        "source_authority_sha256": SUCCESSOR_SOURCE_AUTHORITY_SHA256,
        "source_normalizer_body_sha256": SOURCE_NORMALIZER_BODY_SHA256,
        "t4_semantic_sha256": SOURCE_T4_NORMALIZER_SHA256,
        "behavior_semantic_sha256": SOURCE_BEHAVIOR_NORMALIZER_SHA256,
        "models_share_exact_normalizers": True,
    }


def _fixed_bindings(fixed: Mapping[str, AuthorityMaterial]) -> dict[str, dict[str, object]]:
    if set(fixed) != {spec.name for spec in FIXED_AUTHORITIES}:
        raise FailClosedError("fixed authority set drift")
    return {name: material.binding() for name, material in fixed.items()}


def build_target_free_preflight(
    *,
    root: Path,
    fixed_authorities: Mapping[str, Mapping[str, object]],
    closure: Mapping[str, object],
    baseline: BaselineAuthority,
    within_roster: Sequence[str],
    external_roster: Sequence[str],
    external_assets: Sequence[PreflightAsset],
    normalizers: Mapping[str, object],
) -> dict[str, object]:
    """Build, but never publish, a target-free reviewed score preflight.

    The within assets are intentionally *not* an argument: the only permitted
    mapping is descriptor-read from the frozen C1 paired-view manifest above.
    External rows are separately derived from the fixed v2 ledger before this
    builder is called.
    """
    if set(fixed_authorities) != {spec.name for spec in FIXED_AUTHORITIES}:
        raise FailClosedError("preflight fixed authority set drift")
    for name, binding in fixed_authorities.items():
        expected = next(spec.sha256 for spec in FIXED_AUTHORITIES if spec.name == name)
        if not isinstance(binding, Mapping) or binding.get("body_sha256") != expected:
            raise FailClosedError("preflight fixed authority binding drift")
    checked_closure = _validate_closure(closure)
    within = tuple(within_roster)
    external = tuple(external_roster)
    validate_baseline_rosters(baseline, within_roster=within, external_roster=external)
    within_assets, within_manifest = canonical_within_assets_from_paired_view_manifest(
        Path(root), within_roster=within,
    )
    if tuple(asset.session for asset in external_assets) != external:
        raise FailClosedError("preflight external asset roster drift")
    if any(asset.surface != "within" for asset in within_assets) or any(asset.surface != "external" for asset in external_assets):
        raise FailClosedError("preflight asset surface drift")
    return {
        "schema": "cell_d_equal_session_score_target_free_preflight_v1",
        "status": "PREFLIGHT_ACCEPTED",
        "cell": CELL,
        "phase": PHASE,
        "score_spec": PUBLIC_SPEC.payload(),
        "metric_contract": dict(METRIC_CONTRACT),
        "fixed_authorities": _json_copy(fixed_authorities),
        "closure": checked_closure,
        "baseline": baseline.payload(),
        "within_roster": list(within),
        "external_roster": list(external),
        "within_paired_view_manifest": within_manifest.payload(),
        "within_assets": [asset.payload() for asset in within_assets],
        "external_assets": [asset.payload() for asset in external_assets],
        "normalizers": _normalizer_payload(normalizers),
        "predecessor_successor": {
            "sealed_cell_d_terminal_sha256": SEALED_CELL_D_TERMINAL_SHA256,
            "sealed_cell_d_swa_sha256": SEALED_CELL_D_SWA_SHA256,
            "successor_terminal_sha256": SUCCESSOR_TERMINAL_SHA256,
            "successor_swa_sha256": SUCCESSOR_SWA_SHA256,
            "successor_swa_state_sha256": SUCCESSOR_SWA_STATE_SHA256,
            "successor_graph": "exact_cell_d_seed42_graph_with_equal_session_source_schedule_only",
            "swa_epochs": [44, 45, 46, 47],
        },
        "device_contract": dict(FROZEN_GPU0),
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": SCORE_ROOT_RELATIVE,
        "target_free": True,
        "boundaries": {
            "target_optimizer_steps": 0,
            "backward_calls": 0,
            "update_calls": 0,
            "normalizer_refit": False,
            "formal_or_held_out_forbidden": True,
            "controls_forbidden": ["zero", "wrong_pair", "destroyed", "full_window_governing"],
        },
    }


def _baseline_from_payload(value: object) -> BaselineAuthority:
    if not isinstance(value, Mapping):
        raise FailClosedError("preflight baseline payload drift")
    try:
        within = tuple(BaselineRow(item["session"], item["n_windows"], item["r2"])
                       for item in value["within"] if isinstance(item, Mapping))
        external = tuple(BaselineRow(item["session"], item["n_windows"], item["r2"])
                         for item in value["external"] if isinstance(item, Mapping))
        rebuilt = BaselineAuthority(value["receipt_sha256"], within, external)
    except (KeyError, TypeError, ValueError) as error:
        raise FailClosedError("preflight baseline nested value drift") from error
    if rebuilt.payload() != dict(value):
        raise FailClosedError("preflight baseline roundtrip drift")
    return rebuilt


def validate_target_free_preflight(
    value: Mapping[str, Any], *, fixed_authorities: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    expected = {
        "schema", "status", "cell", "phase", "score_spec", "metric_contract", "fixed_authorities", "closure",
        "baseline", "within_roster", "external_roster", "within_paired_view_manifest", "within_assets", "external_assets", "normalizers",
        "predecessor_successor", "device_contract", "authority_root_relative", "score_root_relative", "target_free",
        "boundaries",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "cell_d_equal_session_score_target_free_preflight_v1"
            or value.get("status") != "PREFLIGHT_ACCEPTED" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("score_spec") != PUBLIC_SPEC.payload()
            or value.get("metric_contract") != METRIC_CONTRACT
            or value.get("fixed_authorities") != _json_copy(fixed_authorities)
            or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE
            or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
            or value.get("target_free") is not True):
        raise FailClosedError("target-free preflight schema/binding drift")
    closure = _validate_closure(value.get("closure"))
    baseline = _baseline_from_payload(value.get("baseline"))
    if baseline.receipt_sha256 != BASELINE_RECEIPT_SHA256:
        raise FailClosedError("target-free preflight baseline receipt drift")
    within_raw, external_raw = value.get("within_roster"), value.get("external_roster")
    if (not isinstance(within_raw, list) or not isinstance(external_raw, list)
            or tuple(within_raw) != tuple(row.session for row in baseline.within)
            or tuple(external_raw) != tuple(row.session for row in baseline.external)):
        raise FailClosedError("target-free preflight roster/baseline drift")
    within_assets = _preflight_assets_from_payload(value.get("within_assets"), surface="within", roster=tuple(within_raw))
    _within_manifest_binding_from_payload(value.get("within_paired_view_manifest"))
    if tuple(asset.payload() for asset in within_assets) != tuple(asset.payload() for asset in sealed_within_assets()):
        raise FailClosedError("target-free preflight within paired-view asset authority drift")
    _preflight_assets_from_payload(value.get("external_assets"), surface="external", roster=tuple(external_raw))
    _normalizer_payload(value.get("normalizers"))
    predecessor = value.get("predecessor_successor")
    exact_predecessor = {
        "sealed_cell_d_terminal_sha256": SEALED_CELL_D_TERMINAL_SHA256,
        "sealed_cell_d_swa_sha256": SEALED_CELL_D_SWA_SHA256,
        "successor_terminal_sha256": SUCCESSOR_TERMINAL_SHA256,
        "successor_swa_sha256": SUCCESSOR_SWA_SHA256,
        "successor_swa_state_sha256": SUCCESSOR_SWA_STATE_SHA256,
        "successor_graph": "exact_cell_d_seed42_graph_with_equal_session_source_schedule_only",
        "swa_epochs": [44, 45, 46, 47],
    }
    if predecessor != exact_predecessor or value.get("device_contract") != FROZEN_GPU0:
        raise FailClosedError("target-free preflight lineage/device drift")
    if value.get("boundaries") != {
        "target_optimizer_steps": 0,
        "backward_calls": 0,
        "update_calls": 0,
        "normalizer_refit": False,
        "formal_or_held_out_forbidden": True,
        "controls_forbidden": ["zero", "wrong_pair", "destroyed", "full_window_governing"],
    }:
        raise FailClosedError("target-free preflight no-update/control boundary drift")
    return _json_copy(value)


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    _require_sha(official_preflight_sha256, "official preflight SHA")
    if not isinstance(preflight, Mapping):
        raise FailClosedError("root authorization needs preflight mapping")
    return {
        "schema": "cell_d_equal_session_score_root_authorization_v1",
        "status": "ROOT_AUTHORIZED",
        "cell": CELL,
        "phase": PHASE,
        "official_preflight_sha256": official_preflight_sha256,
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": SCORE_ROOT_RELATIVE,
        "closure": _json_copy(preflight.get("closure")) if isinstance(preflight.get("closure"), Mapping) else None,
        "device_contract": _json_copy(preflight.get("device_contract")) if isinstance(preflight.get("device_contract"), Mapping) else None,
        "baseline_receipt_sha256": BASELINE_RECEIPT_SHA256,
        "target_free_preflight_required": True,
        "explicit_execution_capability_required": True,
    }


def validate_root_authorization(
    value: Mapping[str, Any], *, official_preflight_sha256: str, preflight: Mapping[str, Any],
) -> dict[str, object]:
    _require_sha(official_preflight_sha256, "official preflight SHA")
    expected = {
        "schema", "status", "cell", "phase", "official_preflight_sha256", "authority_root_relative",
        "score_root_relative", "closure", "device_contract", "baseline_receipt_sha256",
        "target_free_preflight_required", "explicit_execution_capability_required",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "cell_d_equal_session_score_root_authorization_v1"
            or value.get("status") != "ROOT_AUTHORIZED" or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("official_preflight_sha256") != official_preflight_sha256
            or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE
            or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
            or value.get("closure") != preflight.get("closure")
            or value.get("device_contract") != FROZEN_GPU0
            or value.get("baseline_receipt_sha256") != BASELINE_RECEIPT_SHA256
            or value.get("target_free_preflight_required") is not True
            or value.get("explicit_execution_capability_required") is not True):
        raise FailClosedError("root authorization schema/binding drift")
    return _json_copy(value)


_ROOT_PUBLICATION_SEAL = object()
_EXECUTION_SEAL = object()


@dataclass(frozen=True)
class RootPublicationCapability:
    _seal: object


@dataclass(frozen=True)
class ExecutionCapability:
    identity_digest: str
    _seal: object


def _issue_root_publication_capability() -> RootPublicationCapability:
    """Private in-process issuer used only by the reviewed root workflow/tests."""
    return RootPublicationCapability(_ROOT_PUBLICATION_SEAL)


def _issue_execution_capability(identity: ScoreIdentity) -> ExecutionCapability:
    return ExecutionCapability(_sha(_json_bytes(identity.payload())), _EXECUTION_SEAL)


def _require_execution_capability(value: object, identity: ScoreIdentity) -> None:
    if (not isinstance(value, ExecutionCapability) or value._seal is not _EXECUTION_SEAL
            or value.identity_digest != _sha(_json_bytes(identity.payload()))):
        raise FailClosedError("root-reviewed in-process execution capability required before target resolution")


def reserve_authority_artifact(root: Path, capability: RootPublicationCapability) -> ArtifactRoot:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_PUBLICATION_SEAL:
        raise FailClosedError("only root may reserve score authority artifact")
    parent, name = canonical_authority_parent(root)
    return reserve_artifact_root(parent, name, topology=AUTHORITY_TOPOLOGY)


def publish_target_free_preflight(
    artifact: ArtifactRoot,
    capability: RootPublicationCapability,
    payload: Mapping[str, object],
    *,
    fixed_authorities: Mapping[str, Mapping[str, object]],
) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_PUBLICATION_SEAL:
        raise FailClosedError("only root may publish target-free preflight")
    if artifact.topology != AUTHORITY_TOPOLOGY or artifact.has_name("root_authorization.json"):
        raise FailClosedError("authority publication topology/order drift")
    checked = validate_target_free_preflight(payload, fixed_authorities=fixed_authorities)
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization(
    artifact: ArtifactRoot,
    capability: RootPublicationCapability,
    payload: Mapping[str, object],
    *,
    fixed_authorities: Mapping[str, Mapping[str, object]],
) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_PUBLICATION_SEAL:
        raise FailClosedError("only root may publish root authorization")
    if artifact.topology != AUTHORITY_TOPOLOGY or not artifact.has_name("official_preflight.json"):
        raise FailClosedError("root authorization requires durable preflight")
    preflight_body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(preflight_body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("durable preflight JSON drift") from error
    if not isinstance(preflight, Mapping):
        raise FailClosedError("durable preflight root drift")
    checked = validate_target_free_preflight(preflight, fixed_authorities=fixed_authorities)
    authorization = validate_root_authorization(
        payload, official_preflight_sha256=_sha(preflight_body), preflight=checked,
    )
    return artifact.publish_json("root_authorization.json", authorization)


def _read_authority_pair(directory: Path, name: str) -> tuple[Mapping[str, Any], str]:
    if not isinstance(name, str) or not name.endswith(".json") or "/" in name:
        raise ValueError("authority leaf schema drift")
    body, _ = _canonical_regular_bytes(directory, name, expected_mode=0o444)
    digest = _sha(body)
    sidecar, _ = _canonical_regular_bytes(directory, name + ".sha256", expected_mode=0o444)
    if sidecar != f"{digest}  {name}\n".encode("ascii"):
        raise FailClosedError("authority sidecar drift")
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("authority JSON decode drift") from error
    if not isinstance(value, Mapping):
        raise FailClosedError("authority JSON root drift")
    return value, digest


def verify_score_authorization(
    root: Path, *, fixed: Mapping[str, AuthorityMaterial], lineage: Mapping[str, object],
) -> ScoreIdentity:
    """Verify reviewed authority before a score root can be reserved."""
    directory = root.absolute() / AUTHORITY_ROOT_RELATIVE
    preflight, preflight_sha = _read_authority_pair(directory, "official_preflight.json")
    authorization, authorization_sha = _read_authority_pair(directory, "root_authorization.json")
    bindings = _fixed_bindings(fixed)
    checked = validate_target_free_preflight(preflight, fixed_authorities=bindings)
    validate_root_authorization(authorization, official_preflight_sha256=preflight_sha, preflight=checked)
    closure = implementation_closure(root)
    if closure != checked.get("closure"):
        raise FailClosedError("launch implementation closure differs from reviewed preflight")
    baseline = lineage.get("baseline")
    if not isinstance(baseline, BaselineAuthority) or checked.get("baseline") != baseline.payload():
        raise FailClosedError("reviewed baseline authority drift")
    return ScoreIdentity(
        fixed_authorities=bindings,
        closure=closure,
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        sealed_cell_d_terminal_sha256=SEALED_CELL_D_TERMINAL_SHA256,
        sealed_cell_d_swa_sha256=SEALED_CELL_D_SWA_SHA256,
        successor_terminal_sha256=SUCCESSOR_TERMINAL_SHA256,
        successor_swa_sha256=SUCCESSOR_SWA_SHA256,
        successor_swa_state_sha256=SUCCESSOR_SWA_STATE_SHA256,
        preflight_sha256=preflight_sha,
        root_authorization_sha256=authorization_sha,
    )


@dataclass
class ScoreFlags:
    """Honest access/operation accounting for the scoring-only lifecycle."""

    stage: str = "non_data_authority"
    within_resolved: bool = False
    external_resolved: bool = False
    within_opened: bool = False
    external_opened: bool = False
    formal_resolved: bool = False
    formal_opened: bool = False
    normalizer_refit: bool = False
    backward_calls: int = 0
    optimizer_calls: int = 0
    update_calls: int = 0
    terminal_published: bool = False
    forward_calls: dict[str, dict[str, dict[str, int]]] = field(default_factory=lambda: {
        "cell_d": {
            "within": {"aligned_native": 0},
            "external": {"aligned_native": 0},
        },
        "equal_session": {
            "within": {"aligned_native": 0},
            "external": {"aligned_native": 0},
        },
    })

    @staticmethod
    def validate_forward_calls(value: object) -> dict[str, dict[str, dict[str, int]]]:
        expected = {
            "cell_d": {"within": {"aligned_native"}, "external": {"aligned_native"}},
            "equal_session": {"within": {"aligned_native"}, "external": {"aligned_native"}},
        }
        if not isinstance(value, Mapping) or set(value) != set(expected):
            raise FailClosedError("forward system matrix drift")
        checked: dict[str, dict[str, dict[str, int]]] = {}
        for system, surfaces in expected.items():
            block = value.get(system)
            if not isinstance(block, Mapping) or set(block) != set(surfaces):
                raise FailClosedError("forward surface matrix drift")
            checked[system] = {}
            for surface, modes in surfaces.items():
                row = block.get(surface)
                if not isinstance(row, Mapping) or set(row) != modes:
                    raise FailClosedError("forward mode matrix drift")
                if any(type(count) is not int or count < 0 for count in row.values()):
                    raise FailClosedError("forward count drift")
                checked[system][surface] = {"aligned_native": int(row["aligned_native"])}
        return checked

    def record_forward(self, system: str, surface: str) -> None:
        try:
            count = self.forward_calls[system][surface]["aligned_native"]
        except (KeyError, TypeError) as error:
            raise FailClosedError("unknown score forward cell") from error
        if type(count) is not int or count < 0:
            raise FailClosedError("forward counter drift")
        self.forward_calls[system][surface]["aligned_native"] = count + 1

    def payload(self) -> dict[str, object]:
        if type(self.backward_calls) is not int or type(self.optimizer_calls) is not int or type(self.update_calls) is not int:
            raise FailClosedError("operation counter type drift")
        return {
            "stage": self.stage,
            "resolved": {"within": self.within_resolved, "external": self.external_resolved, "formal": self.formal_resolved},
            "opened": {"within": self.within_opened, "external": self.external_opened, "formal": self.formal_opened},
            "normalizer_refit": self.normalizer_refit,
            "forward_calls": self.validate_forward_calls(self.forward_calls),
            "backward_calls": self.backward_calls,
            "optimizer_calls": self.optimizer_calls,
            "update_calls": self.update_calls,
            "terminal_published": self.terminal_published,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ScoreFlags":
        expected = {
            "stage", "resolved", "opened", "normalizer_refit", "forward_calls", "backward_calls",
            "optimizer_calls", "update_calls", "terminal_published",
        }
        if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value.get("stage"), str):
            raise FailClosedError("score flag payload schema drift")
        resolved, opened = value.get("resolved"), value.get("opened")
        if (not isinstance(resolved, Mapping) or not isinstance(opened, Mapping)
                or set(resolved) != {"within", "external", "formal"}
                or set(opened) != {"within", "external", "formal"}
                or any(type(item) is not bool for item in (*resolved.values(), *opened.values()))
                or type(value.get("normalizer_refit")) is not bool
                or type(value.get("terminal_published")) is not bool
                or any(type(value.get(key)) is not int or int(value[key]) < 0
                       for key in ("backward_calls", "optimizer_calls", "update_calls"))):
            raise FailClosedError("score flag payload nested drift")
        flags = cls(
            stage=str(value["stage"]),
            within_resolved=bool(resolved["within"]), external_resolved=bool(resolved["external"]),
            formal_resolved=bool(resolved["formal"]), within_opened=bool(opened["within"]),
            external_opened=bool(opened["external"]), formal_opened=bool(opened["formal"]),
            normalizer_refit=bool(value["normalizer_refit"]), backward_calls=int(value["backward_calls"]),
            optimizer_calls=int(value["optimizer_calls"]), update_calls=int(value["update_calls"]),
            terminal_published=bool(value["terminal_published"]),
            forward_calls=cls.validate_forward_calls(value["forward_calls"]),
        )
        if flags.payload() != dict(value):
            raise FailClosedError("score flag roundtrip drift")
        return flags


def _require_no_updates(flags: ScoreFlags) -> None:
    if (flags.formal_resolved or flags.formal_opened or flags.normalizer_refit
            or flags.backward_calls != 0 or flags.optimizer_calls != 0 or flags.update_calls != 0):
        raise FailClosedError("scoring crossed forbidden formal/refit/update boundary")


@dataclass(frozen=True)
class SessionInputAuthority:
    """One source-normalized, target-free input binding shared by both models."""

    surface: str
    session: str
    n_windows: int
    asset: Mapping[str, object]
    ordered_unit_digest: str
    neural_sha256: str
    behavior_sha256: str
    query_start_sha256: str
    last_bin_target_sha256: str
    last_bin_valid_mask_sha256: str
    last_bin_valid_count: int
    normalized_t4_sha256: str
    raw_t4_sha256: str
    raw_t4_channel_alignment: Mapping[str, object]
    raw_t4_channel_alignment_sha256: str
    calibration_sha256: str
    held_descriptor_identity: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.surface not in {"within", "external"} or not isinstance(self.session, str) or not self.session:
            raise ValueError("input authority surface/session drift")
        if (type(self.n_windows) is not int or self.n_windows <= 0 or not isinstance(self.asset, Mapping)
                or type(self.last_bin_valid_count) is not int
                or self.last_bin_valid_count != self.n_windows):
            raise ValueError("input authority window/asset drift")
        for label, value in (
            ("ordered units", self.ordered_unit_digest), ("neural", self.neural_sha256),
            ("behavior", self.behavior_sha256), ("query starts", self.query_start_sha256),
            ("last-bin target", self.last_bin_target_sha256),
            ("last-bin valid mask", self.last_bin_valid_mask_sha256),
            ("normalized T4", self.normalized_t4_sha256), ("raw T4", self.raw_t4_sha256),
            ("raw T4/channel alignment", self.raw_t4_channel_alignment_sha256),
            ("calibration", self.calibration_sha256),
        ):
            _require_sha(value, label)
        raw_proof = _json_copy(self.raw_t4_channel_alignment)
        expected_keys = {
            "schema", "session", "signal_view", "channel_ids_dtype", "channel_ids_sha256",
            "source_unit_count", "raw_t4_shape", "feature_group", "pool_size", "mapping",
            "closure_bound_functions",
        }
        if (set(raw_proof) != expected_keys
                or raw_proof.get("schema") != "cell_d_equal_session_raw_t4_sua_axis_proof_v1"
                or raw_proof.get("session") != self.session or raw_proof.get("signal_view") != "sua"
                or raw_proof.get("channel_ids_dtype") != "int64"
                or raw_proof.get("feature_group") != "t4" or raw_proof.get("pool_size") != 30
                or raw_proof.get("mapping") != "raw_t4_row_k_equals_sua_neural_column_k_for_contiguous_channel_ids"
                or raw_proof.get("closure_bound_functions") != [
                    "mc_maze.multisession_datamodule.load_dandi688_session",
                    "mc_maze.unit_side_features.compute_unit_side_features_uncached",
                ]
                or type(raw_proof.get("source_unit_count")) is not int
                or raw_proof.get("source_unit_count") <= 0
                or raw_proof.get("raw_t4_shape") != [raw_proof.get("source_unit_count"), 4]
                or not _is_sha(raw_proof.get("channel_ids_sha256"))
                or _sha(_json_bytes(raw_proof)) != self.raw_t4_channel_alignment_sha256):
            raise ValueError("input authority raw T4/SUA axis proof drift")
        if (not isinstance(self.held_descriptor_identity, tuple) or len(self.held_descriptor_identity) != 3
                or any(type(item) is not int or item < 0 for item in self.held_descriptor_identity)):
            raise ValueError("input authority held descriptor identity drift")

    def payload(self) -> dict[str, object]:
        return {
            "surface": self.surface, "session": self.session, "n_windows": self.n_windows,
            "asset": _json_copy(self.asset), "ordered_unit_digest": self.ordered_unit_digest,
            "neural_sha256": self.neural_sha256, "behavior_sha256": self.behavior_sha256,
            "query_start_sha256": self.query_start_sha256,
            "last_bin_target_sha256": self.last_bin_target_sha256,
            "last_bin_valid_mask_sha256": self.last_bin_valid_mask_sha256,
            "last_bin_valid_count": self.last_bin_valid_count,
            "normalized_t4_sha256": self.normalized_t4_sha256,
            "raw_t4_sha256": self.raw_t4_sha256,
            "raw_t4_channel_alignment": _json_copy(self.raw_t4_channel_alignment),
            "raw_t4_channel_alignment_sha256": self.raw_t4_channel_alignment_sha256,
            "calibration_sha256": self.calibration_sha256,
            "held_descriptor_identity": list(self.held_descriptor_identity),
        }


@dataclass(frozen=True)
class InputAuthorityEvidence:
    records: tuple[SessionInputAuthority, ...]
    source_normalizer_body_sha256: str
    t4_normalizer_semantic_sha256: str
    behavior_normalizer_semantic_sha256: str
    no_cache_readonly_adapter: bool
    shared_input_pass: bool

    def __post_init__(self) -> None:
        if len(self.records) != PUBLIC_SPEC.within_sessions + PUBLIC_SPEC.external_sessions:
            raise ValueError("input authority record count drift")
        if tuple((item.surface, item.session) for item in self.records) != tuple(sorted((item.surface, item.session) for item in self.records)):
            raise ValueError("input authority record order drift")
        if len({(item.surface, item.session) for item in self.records}) != len(self.records):
            raise ValueError("input authority record duplication")
        for label, value, expected in (
            ("source normalizer", self.source_normalizer_body_sha256, SOURCE_NORMALIZER_BODY_SHA256),
            ("T4 normalizer", self.t4_normalizer_semantic_sha256, SOURCE_T4_NORMALIZER_SHA256),
            ("behavior normalizer", self.behavior_normalizer_semantic_sha256, SOURCE_BEHAVIOR_NORMALIZER_SHA256),
        ):
            _require_sha(value, label)
            if value != expected:
                raise ValueError(f"{label} authority drift")
        if self.no_cache_readonly_adapter is not True or self.shared_input_pass is not True:
            raise ValueError("input authority no-cache/shared-input proof drift")

    def payload(self) -> dict[str, object]:
        return {
            "records": [item.payload() for item in self.records],
            "source_normalizer_body_sha256": self.source_normalizer_body_sha256,
            "t4_normalizer_semantic_sha256": self.t4_normalizer_semantic_sha256,
            "behavior_normalizer_semantic_sha256": self.behavior_normalizer_semantic_sha256,
            "no_cache_readonly_adapter": self.no_cache_readonly_adapter,
            "shared_input_pass": self.shared_input_pass,
        }


def validate_input_authority_evidence(
    evidence: InputAuthorityEvidence,
    *,
    within_roster: Sequence[str],
    external_roster: Sequence[str],
    within_assets: Sequence[PreflightAsset],
    external_assets: Sequence[PreflightAsset],
) -> None:
    if tuple(asset.session for asset in within_assets) != tuple(within_roster):
        raise FailClosedError("input authority within asset roster drift")
    if tuple(asset.session for asset in external_assets) != tuple(external_roster):
        raise FailClosedError("input authority external asset roster drift")
    by_key = {(record.surface, record.session): record for record in evidence.records}
    if set(by_key) != ({("within", item) for item in within_roster} | {("external", item) for item in external_roster}):
        raise FailClosedError("input authority session roster drift")
    for surface, assets in (("within", within_assets), ("external", external_assets)):
        for asset in assets:
            record = by_key[(surface, asset.session)]
            expected_asset = asset.payload()
            if record.asset != expected_asset:
                raise FailClosedError("input authority asset binding drift")


def input_authority_from_payload(value: Mapping[str, Any]) -> InputAuthorityEvidence:
    expected = {
        "records", "source_normalizer_body_sha256", "t4_normalizer_semantic_sha256",
        "behavior_normalizer_semantic_sha256", "no_cache_readonly_adapter", "shared_input_pass",
    }
    if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value.get("records"), list):
        raise FailClosedError("input authority payload schema drift")
    records: list[SessionInputAuthority] = []
    for item in value["records"]:
        if not isinstance(item, Mapping):
            raise FailClosedError("input authority record type drift")
        try:
            records.append(SessionInputAuthority(
                surface=item["surface"], session=item["session"], n_windows=item["n_windows"], asset=item["asset"],
                ordered_unit_digest=item["ordered_unit_digest"], neural_sha256=item["neural_sha256"],
                behavior_sha256=item["behavior_sha256"], query_start_sha256=item["query_start_sha256"],
                last_bin_target_sha256=item["last_bin_target_sha256"],
                last_bin_valid_mask_sha256=item["last_bin_valid_mask_sha256"],
                last_bin_valid_count=item["last_bin_valid_count"],
                normalized_t4_sha256=item["normalized_t4_sha256"], raw_t4_sha256=item["raw_t4_sha256"],
                raw_t4_channel_alignment=item["raw_t4_channel_alignment"],
                raw_t4_channel_alignment_sha256=item["raw_t4_channel_alignment_sha256"],
                calibration_sha256=item["calibration_sha256"],
                held_descriptor_identity=tuple(item["held_descriptor_identity"]),
            ))
        except (KeyError, TypeError, ValueError) as error:
            raise FailClosedError("input authority record nested drift") from error
    try:
        rebuilt = InputAuthorityEvidence(
            records=tuple(records), source_normalizer_body_sha256=value["source_normalizer_body_sha256"],
            t4_normalizer_semantic_sha256=value["t4_normalizer_semantic_sha256"],
            behavior_normalizer_semantic_sha256=value["behavior_normalizer_semantic_sha256"],
            no_cache_readonly_adapter=value["no_cache_readonly_adapter"], shared_input_pass=value["shared_input_pass"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FailClosedError("input authority payload nested drift") from error
    if rebuilt.payload() != dict(value):
        raise FailClosedError("input authority payload roundtrip drift")
    return rebuilt


@dataclass(frozen=True)
class SessionScore:
    session: str
    n_windows: int
    governing_last_bin_r2: float
    prediction_sha256: str
    input_authority_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.session, str) or not self.session or type(self.n_windows) is not int or self.n_windows <= 0:
            raise ValueError("session score session/window drift")
        _finite(self.governing_last_bin_r2, "governing last-bin R2")
        _require_sha(self.prediction_sha256, "prediction digest")
        _require_sha(self.input_authority_sha256, "input authority digest")

    def payload(self) -> dict[str, object]:
        return {
            "session": self.session, "n_windows": self.n_windows,
            "governing_last_bin_r2": self.governing_last_bin_r2,
            "prediction_sha256": self.prediction_sha256,
            "input_authority_sha256": self.input_authority_sha256,
        }


@dataclass(frozen=True)
class ModeEvidence:
    """One aligned/native, last-bin-only model score over a surface."""

    system: str
    surface: str
    mode: str
    sessions: tuple[SessionScore, ...]
    state_before_sha256: str
    state_after_sha256: str
    eval_mode: bool
    dropout_disabled: bool
    gradients_none: bool
    finite_output: bool
    output_shape: tuple[int, int, int]
    repeat_fixed_batch_bitwise_equal: bool
    b3s_recomputed: bool

    def __post_init__(self) -> None:
        if self.system not in {"cell_d", "equal_session"} or self.surface not in {"within", "external"}:
            raise ValueError("mode evidence system/surface drift")
        if self.mode != "aligned_native":
            raise ValueError("only aligned/native score mode is authorized")
        count = PUBLIC_SPEC.within_sessions if self.surface == "within" else PUBLIC_SPEC.external_sessions
        if (len(self.sessions) != count or len({item.session for item in self.sessions}) != count
                or tuple(item.session for item in self.sessions) != tuple(sorted(item.session for item in self.sessions))):
            raise ValueError("mode evidence session roster/order drift")
        for label, value in (("state before", self.state_before_sha256), ("state after", self.state_after_sha256)):
            _require_sha(value, label)
        if self.state_before_sha256 != self.state_after_sha256:
            raise ValueError("model state changed during scoring")
        if not all(item is True for item in (
            self.eval_mode, self.dropout_disabled, self.gradients_none, self.finite_output,
            self.repeat_fixed_batch_bitwise_equal, self.b3s_recomputed,
        )):
            raise ValueError("score eval/dropout/gradient/state invariant drift")
        if (not isinstance(self.output_shape, tuple) or len(self.output_shape) != 3
                or self.output_shape[0] <= 0 or self.output_shape[1:] != (PUBLIC_SPEC.window_bins, 2)):
            raise ValueError("score output shape must be [B,50,2]")

    @property
    def equal_session_mean(self) -> float:
        return sum(item.governing_last_bin_r2 for item in self.sessions) / len(self.sessions)

    def payload(self) -> dict[str, object]:
        return {
            "system": self.system, "surface": self.surface, "mode": self.mode,
            "sessions": [item.payload() for item in self.sessions],
            "equal_session_mean_governing_last_bin_r2": self.equal_session_mean,
            "state_before_sha256": self.state_before_sha256, "state_after_sha256": self.state_after_sha256,
            "eval_mode": self.eval_mode, "dropout_disabled": self.dropout_disabled,
            "gradients_none": self.gradients_none, "finite_output": self.finite_output,
            "output_shape": list(self.output_shape),
            "repeat_fixed_batch_bitwise_equal": self.repeat_fixed_batch_bitwise_equal,
            "b3s_recomputed": self.b3s_recomputed,
        }


def mode_evidence_from_payload(value: Mapping[str, Any]) -> ModeEvidence:
    expected = {
        "system", "surface", "mode", "sessions", "equal_session_mean_governing_last_bin_r2",
        "state_before_sha256", "state_after_sha256", "eval_mode", "dropout_disabled", "gradients_none",
        "finite_output", "output_shape", "repeat_fixed_batch_bitwise_equal", "b3s_recomputed",
    }
    if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value.get("sessions"), list):
        raise FailClosedError("mode evidence payload schema drift")
    sessions: list[SessionScore] = []
    for item in value["sessions"]:
        if not isinstance(item, Mapping):
            raise FailClosedError("mode evidence session type drift")
        try:
            sessions.append(SessionScore(
                session=item["session"], n_windows=item["n_windows"],
                governing_last_bin_r2=item["governing_last_bin_r2"], prediction_sha256=item["prediction_sha256"],
                input_authority_sha256=item["input_authority_sha256"],
            ))
        except (KeyError, TypeError, ValueError) as error:
            raise FailClosedError("mode evidence session nested drift") from error
    shape = value.get("output_shape")
    if not isinstance(shape, list):
        raise FailClosedError("mode evidence output shape drift")
    try:
        evidence = ModeEvidence(
            system=value["system"], surface=value["surface"], mode=value["mode"], sessions=tuple(sessions),
            state_before_sha256=value["state_before_sha256"], state_after_sha256=value["state_after_sha256"],
            eval_mode=value["eval_mode"], dropout_disabled=value["dropout_disabled"],
            gradients_none=value["gradients_none"], finite_output=value["finite_output"], output_shape=tuple(shape),
            repeat_fixed_batch_bitwise_equal=value["repeat_fixed_batch_bitwise_equal"], b3s_recomputed=value["b3s_recomputed"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FailClosedError("mode evidence nested drift") from error
    if evidence.payload() != dict(value):
        raise FailClosedError("mode evidence roundtrip drift")
    return evidence


def _validate_mode_against_roster(
    evidence: ModeEvidence,
    *,
    system: str,
    surface: str,
    roster: Sequence[str],
    input_authority_sha256: str,
) -> None:
    if (evidence.system != system or evidence.surface != surface or evidence.mode != "aligned_native"
            or tuple(item.session for item in evidence.sessions) != tuple(roster)
            or any(item.input_authority_sha256 != input_authority_sha256 for item in evidence.sessions)):
        raise FailClosedError("score mode roster/input authority drift")


def assert_cell_d_baseline_parity(evidence: ModeEvidence, baseline: BaselineAuthority) -> None:
    if evidence.system != "cell_d" or evidence.mode != "aligned_native":
        raise FailClosedError("Cell-D baseline parity needs aligned/native Cell-D evidence")
    expected = baseline.rows(evidence.surface)
    observed = tuple((row.session, row.n_windows, row.governing_last_bin_r2) for row in evidence.sessions)
    sealed = tuple((row.session, row.n_windows, row.r2) for row in expected)
    if observed != sealed or evidence.equal_session_mean != baseline.mean(evidence.surface):
        raise FailClosedError("Cell-D governing last-bin full-table parity failure")


def _paired_statistics(deltas: Sequence[float]) -> dict[str, object]:
    if not deltas:
        raise FailClosedError("paired delta vector cannot be empty")
    values = tuple(_finite(item, "paired delta") for item in deltas)
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return {
        "mean": sum(values) / len(values),
        "median": median,
        "n_positive": sum(item > 0.0 for item in values),
        "n_total": len(values),
        "min": min(values),
        "max": max(values),
        "all_deltas": list(values),
        "exact_sign_pattern": "".join("+" if item > 0 else ("-" if item < 0 else "0") for item in values),
    }


def decide_verdict(*, external: Mapping[str, object], within: Mapping[str, object]) -> str:
    for surface, stats, count in (("external", external, PUBLIC_SPEC.external_sessions), ("within", within, PUBLIC_SPEC.within_sessions)):
        if (not isinstance(stats, Mapping) or stats.get("n_total") != count
                or type(stats.get("n_positive")) is not int or not 0 <= int(stats["n_positive"]) <= count):
            raise FailClosedError(f"{surface} gate statistics schema drift")
        _finite(stats.get("mean"), f"{surface} mean")
        _finite(stats.get("median"), f"{surface} median")
    external_mean, within_mean = float(external["mean"]), float(within["mean"])
    if external_mean < 0.0 or within_mean < -0.03:
        return "STOP"
    if (external_mean >= 0.03 and within_mean >= -0.03
            and float(external["median"]) > 0.0 and int(external["n_positive"]) >= 9):
        return "CLEAR_GO"
    return "HOLD"


def _contrast(successor: ModeEvidence, baseline: ModeEvidence) -> dict[str, object]:
    if successor.surface != baseline.surface:
        raise FailClosedError("paired contrast surface mismatch")
    if tuple(item.session for item in successor.sessions) != tuple(item.session for item in baseline.sessions):
        raise FailClosedError("paired contrast session order mismatch")
    rows: list[dict[str, object]] = []
    deltas: list[float] = []
    for equal, cell_d in zip(successor.sessions, baseline.sessions, strict=True):
        if equal.n_windows != cell_d.n_windows:
            raise FailClosedError("paired contrast window count mismatch")
        delta = equal.governing_last_bin_r2 - cell_d.governing_last_bin_r2
        deltas.append(delta)
        rows.append({
            "session": equal.session,
            "n_windows": equal.n_windows,
            "equal_session_r2": equal.governing_last_bin_r2,
            "cell_d_r2": cell_d.governing_last_bin_r2,
            "delta": delta,
        })
    statistics = _paired_statistics(deltas)
    return {"per_session": rows, **statistics}


def _resource_disclosure(value: Mapping[str, object]) -> dict[str, object]:
    """Validate score-only resource/boundary facts without hiding diagnostics."""
    required = {
        "device_contract", "forward_only", "no_grad", "optimizer_steps", "backward_calls", "update_calls",
        "normalizer_refit", "formal_or_held_out_opened", "same_inputs_for_both_models", "details",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise FailClosedError("resource disclosure schema drift")
    if (value.get("device_contract") != FROZEN_GPU0 or value.get("forward_only") is not True
            or value.get("no_grad") is not True or value.get("optimizer_steps") != 0
            or value.get("backward_calls") != 0 or value.get("update_calls") != 0
            or value.get("normalizer_refit") is not False or value.get("formal_or_held_out_opened") is not False
            or value.get("same_inputs_for_both_models") is not True or not isinstance(value.get("details"), Mapping)):
        raise FailClosedError("resource disclosure score-only boundary drift")
    return _json_copy(value)


def build_score_payload(
    *,
    identity: ScoreIdentity,
    baseline: BaselineAuthority,
    input_authority_sha256: str,
    cell_d: Sequence[ModeEvidence],
    equal_session: Sequence[ModeEvidence],
    resources: Mapping[str, object],
    flags: ScoreFlags,
) -> dict[str, object]:
    """Build the only admissible aligned/native matched score payload."""
    _require_sha(input_authority_sha256, "input authority SHA")
    _require_no_updates(flags)
    if flags.terminal_published:
        raise FailClosedError("cannot build score after terminal publication")
    if len(cell_d) != 2 or len(equal_session) != 2:
        raise FailClosedError("matched score requires exactly two surfaces per model")
    d_map, successor_map = ({item.surface: item for item in cell_d}, {item.surface: item for item in equal_session})
    if set(d_map) != {"within", "external"} or set(successor_map) != {"within", "external"}:
        raise FailClosedError("matched score surface topology drift")
    for surface in ("within", "external"):
        roster = tuple(row.session for row in baseline.rows(surface))
        _validate_mode_against_roster(
            d_map[surface], system="cell_d", surface=surface, roster=roster,
            input_authority_sha256=input_authority_sha256,
        )
        _validate_mode_against_roster(
            successor_map[surface], system="equal_session", surface=surface, roster=roster,
            input_authority_sha256=input_authority_sha256,
        )
        assert_cell_d_baseline_parity(d_map[surface], baseline)
    deltas = {
        surface: _contrast(successor_map[surface], d_map[surface])
        for surface in ("within", "external")
    }
    verdict = decide_verdict(external=deltas["external"], within=deltas["within"])
    return {
        "schema": "cell_d_equal_session_matched_score_v1",
        "status": "MATCHED_SCORE_COMPLETE",
        "cell": CELL,
        "phase": PHASE,
        "score_spec": PUBLIC_SPEC.payload(),
        "metric_contract": dict(METRIC_CONTRACT),
        "identity": identity.payload(),
        "input_authority_sha256": input_authority_sha256,
        "baseline_authority": baseline.payload(),
        "cell_d": {surface: d_map[surface].payload() for surface in ("within", "external")},
        "equal_session": {surface: successor_map[surface].payload() for surface in ("within", "external")},
        "equal_session_minus_cell_d": deltas,
        "verdict": verdict,
        "verdict_rule": (
            "STOP_if_external_mean_lt_0_or_within_mean_lt_-0.03;"
            "CLEAR_GO_if_external_mean_ge_0.03_and_within_mean_ge_-0.03_and_external_median_gt_0_and_external_positive_ge_9;"
            "otherwise_HOLD"
        ),
        "resources": _resource_disclosure(resources),
        "access_disclosure": flags.payload(),
        "controls": {"zero": False, "wrong_pair": False, "destroyed": False, "full_window_governing": False},
    }


def validate_score_payload(
    value: Mapping[str, Any],
    *,
    identity: ScoreIdentity,
    baseline: BaselineAuthority,
    input_authority_sha256: str,
) -> dict[str, object]:
    expected = {
        "schema", "status", "cell", "phase", "score_spec", "metric_contract", "identity", "input_authority_sha256",
        "baseline_authority", "cell_d", "equal_session", "equal_session_minus_cell_d", "verdict", "verdict_rule",
        "resources", "access_disclosure", "controls",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "cell_d_equal_session_matched_score_v1"
            or value.get("status") != "MATCHED_SCORE_COMPLETE" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("score_spec") != PUBLIC_SPEC.payload()
            or value.get("metric_contract") != METRIC_CONTRACT or value.get("identity") != identity.payload()
            or value.get("input_authority_sha256") != input_authority_sha256
            or value.get("baseline_authority") != baseline.payload()
            or value.get("controls") != {"zero": False, "wrong_pair": False, "destroyed": False, "full_window_governing": False}):
        raise FailClosedError("score payload header/authority/control drift")
    d_raw, successor_raw = value.get("cell_d"), value.get("equal_session")
    if (not isinstance(d_raw, Mapping) or not isinstance(successor_raw, Mapping)
            or set(d_raw) != {"within", "external"} or set(successor_raw) != {"within", "external"}):
        raise FailClosedError("score payload model/surface topology drift")
    try:
        cell_d = tuple(mode_evidence_from_payload(d_raw[surface]) for surface in ("within", "external"))
        successor = tuple(mode_evidence_from_payload(successor_raw[surface]) for surface in ("within", "external"))
    except (KeyError, TypeError) as error:
        raise FailClosedError("score payload model evidence drift") from error
    flags = ScoreFlags.from_payload(value.get("access_disclosure"))
    rebuilt = build_score_payload(
        identity=identity, baseline=baseline, input_authority_sha256=input_authority_sha256,
        cell_d=cell_d, equal_session=successor, resources=value.get("resources"), flags=flags,
    )
    if rebuilt != dict(value):
        raise FailClosedError("score payload roundtrip/statistic/verdict drift")
    return rebuilt


def _attempt_payload(identity: ScoreIdentity) -> dict[str, object]:
    return {
        "schema": "cell_d_equal_session_score_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": CELL,
        "phase": PHASE,
        "identity": identity.payload(),
        "target_resolution_after_attempt_only": True,
        "formal_or_held_out_forbidden": True,
        "mode": "aligned_native_only",
    }


def validate_attempt_payload(value: Mapping[str, Any], identity: ScoreIdentity) -> None:
    expected = {
        "schema", "status", "cell", "phase", "identity", "target_resolution_after_attempt_only",
        "formal_or_held_out_forbidden", "mode",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "cell_d_equal_session_score_attempt_v1"
            or value.get("status") != "ATTEMPT_RESERVED" or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("identity") != identity.payload() or value.get("target_resolution_after_attempt_only") is not True
            or value.get("formal_or_held_out_forbidden") is not True or value.get("mode") != "aligned_native_only"):
        raise FailClosedError("attempt receipt drift")


def _input_authority_payload(evidence: InputAuthorityEvidence, identity: ScoreIdentity) -> dict[str, object]:
    return {
        "schema": "cell_d_equal_session_score_input_authority_v1",
        "cell": CELL,
        "phase": PHASE,
        "identity": identity.payload(),
        "input_evidence": evidence.payload(),
        "same_input_pass_for_both_models": True,
    }


def validate_input_authority_payload(value: Mapping[str, Any], identity: ScoreIdentity) -> InputAuthorityEvidence:
    expected = {"schema", "cell", "phase", "identity", "input_evidence", "same_input_pass_for_both_models"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "cell_d_equal_session_score_input_authority_v1"
            or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("identity") != identity.payload() or value.get("same_input_pass_for_both_models") is not True
            or not isinstance(value.get("input_evidence"), Mapping)):
        raise FailClosedError("input authority receipt drift")
    return input_authority_from_payload(value["input_evidence"])


def _failure_payload(flags: ScoreFlags) -> dict[str, object]:
    return {
        "schema": "cell_d_equal_session_score_failure_v1",
        "status": "FAILED",
        "cell": CELL,
        "phase": PHASE,
        "stage": flags.stage,
        "access_disclosure": flags.payload(),
        "traceback_sha256": _sha(traceback.format_exc().encode("utf-8")),
    }


def validate_failure_payload(value: Mapping[str, Any]) -> None:
    expected = {"schema", "status", "cell", "phase", "stage", "access_disclosure", "traceback_sha256"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "cell_d_equal_session_score_failure_v1" or value.get("status") != "FAILED"
            or value.get("cell") != CELL or value.get("phase") != PHASE or not isinstance(value.get("stage"), str)
            or not _is_sha(value.get("traceback_sha256"))):
        raise FailClosedError("failure receipt header drift")
    flags = ScoreFlags.from_payload(value.get("access_disclosure"))
    if flags.terminal_published:
        raise FailClosedError("failure receipt cannot claim terminal publication")
    # A failure receipt must describe an observed boundary violation honestly;
    # it must not erase a non-zero attempted update merely because that fact
    # would be unacceptable in a successful scientific terminal.


def _terminal_payload(
    *,
    identity: ScoreIdentity,
    attempt_sha256: str,
    input_authority_sha256: str,
    score_sha256: str,
    score_payload: Mapping[str, Any],
    final_closure: Mapping[str, object],
) -> dict[str, object]:
    external = score_payload["equal_session_minus_cell_d"]["external"]
    within = score_payload["equal_session_minus_cell_d"]["within"]
    return {
        "schema": "cell_d_equal_session_score_terminal_v1",
        "status": "SCORE_COMPLETE",
        "cell": CELL,
        "phase": PHASE,
        "identity": identity.payload(),
        "attempt_sha256": attempt_sha256,
        "input_authority_sha256": input_authority_sha256,
        "score_sha256": score_sha256,
        "launch_closure": _json_copy(identity.closure),
        "final_closure": _json_copy(final_closure),
        "verdict": decide_verdict(external=external, within=within),
        "verdict_rule": score_payload["verdict_rule"],
        "boundaries": {
            "target_optimizer_steps": 0,
            "backward_calls": 0,
            "update_calls": 0,
            "normalizer_refit": False,
            "formal_or_held_out_opened": False,
            "score_and_terminal_transactional": True,
        },
    }


def validate_terminal_payload(
    value: Mapping[str, Any],
    *,
    identity: ScoreIdentity,
    score_payload: Mapping[str, Any],
    expected_score_sha256: str | None = None,
) -> None:
    expected = {
        "schema", "status", "cell", "phase", "identity", "attempt_sha256", "input_authority_sha256",
        "score_sha256", "launch_closure", "final_closure", "verdict", "verdict_rule", "boundaries",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "cell_d_equal_session_score_terminal_v1" or value.get("status") != "SCORE_COMPLETE"
            or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("identity") != identity.payload()
            or value.get("launch_closure") != identity.closure or value.get("final_closure") != identity.closure
            or value.get("verdict_rule") != score_payload.get("verdict_rule")):
        raise FailClosedError("terminal receipt schema/identity/closure drift")
    for label in ("attempt_sha256", "input_authority_sha256", "score_sha256"):
        _require_sha(value.get(label), label)
    if expected_score_sha256 is not None and value.get("score_sha256") != expected_score_sha256:
        raise FailClosedError("terminal does not bind exact score body")
    expected_verdict = decide_verdict(
        external=score_payload["equal_session_minus_cell_d"]["external"],
        within=score_payload["equal_session_minus_cell_d"]["within"],
    )
    if value.get("verdict") != expected_verdict:
        raise FailClosedError("terminal verdict drift")
    if value.get("boundaries") != {
        "target_optimizer_steps": 0,
        "backward_calls": 0,
        "update_calls": 0,
        "normalizer_refit": False,
        "formal_or_held_out_opened": False,
        "score_and_terminal_transactional": True,
    }:
        raise FailClosedError("terminal no-update boundary drift")


class ScoreBackend(Protocol):
    """Injected physical adapter boundary; no target work occurs before attempt."""

    def resolve_inputs(
        self,
        *,
        within_roster: tuple[str, ...],
        external_roster: tuple[str, ...],
        within_assets: tuple[PreflightAsset, ...],
        external_assets: tuple[PreflightAsset, ...],
        flags: ScoreFlags,
    ) -> InputAuthorityEvidence: ...

    def score(
        self,
        *,
        system: str,
        surface: str,
        input_authority_sha256: str,
        flags: ScoreFlags,
    ) -> ModeEvidence: ...

    def reverify_after_forwards(self, *, flags: ScoreFlags) -> None: ...

    def resource_disclosure(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _ScoreDataRoot:
    """A held named-directory capability for one reviewed evaluation surface.

    The physical scorer never lets a parser reopen a mutable pathname directly
    from an environment variable.  It first pins the directory identity, then
    binds every preflight asset to ``root / basename(frozen_path)``.
    """

    directory: Path
    identity: tuple[int, int]
    label: str

    @classmethod
    def from_environment(cls, *, variable: str, label: str) -> "_ScoreDataRoot":
        rendered = os.environ.get(variable)
        if not isinstance(rendered, str) or not rendered:
            raise FailClosedError(f"{label} data-root environment authority is absent")
        candidate = Path(rendered).absolute()
        try:
            info = os.lstat(candidate)
        except OSError as error:
            raise FailClosedError(f"{label} data-root authority is not statable") from error
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise FailClosedError(f"{label} data-root must be a canonical non-symlink directory")
        return cls(directory=candidate, identity=(info.st_dev, info.st_ino), label=label)

    def assert_named_identity(self) -> None:
        try:
            info = os.lstat(self.directory)
        except OSError as error:
            raise FailClosedError(f"{self.label} data-root disappeared") from error
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or (info.st_dev, info.st_ino) != self.identity):
            raise FailClosedError(f"{self.label} data-root identity drift")


@dataclass(frozen=True)
class _ScoreAssetBinding:
    """One reviewed path binding formed only after the durable attempt."""

    asset: PreflightAsset
    local_path: Path
    data_root: _ScoreDataRoot

    @classmethod
    def from_preflight(cls, asset: PreflightAsset, data_root: _ScoreDataRoot) -> "_ScoreAssetBinding":
        data_root.assert_named_identity()
        basename = Path(asset.frozen_path).name
        if basename in {"", ".", ".."} or basename != Path(asset.frozen_path).parts[-1]:
            raise FailClosedError("frozen target asset basename drift")
        return cls(asset=asset, local_path=data_root.directory / basename, data_root=data_root)

    def payload(self) -> dict[str, object]:
        return self.asset.payload()


@dataclass
class _HeldScoreAsset:
    """A read-only O_NOFOLLOW descriptor kept alive through all forwards."""

    binding: _ScoreAssetBinding
    descriptor: int
    identity: tuple[int, int, int]

    def reverify(self) -> None:
        self.binding.data_root.assert_named_identity()
        if self.descriptor < 0:
            raise FailClosedError("held evaluation descriptor is already closed")
        info = os.fstat(self.descriptor)
        if (not stat.S_ISREG(info.st_mode) or _regular_identity(info) != self.identity
                or info.st_size != self.binding.asset.expected_bytes):
            raise FailClosedError("held evaluation descriptor identity/size drift")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        if _sha(_read_fd_all(self.descriptor)) != self.binding.asset.expected_sha256:
            raise FailClosedError("held evaluation descriptor SHA drift")
        try:
            named = os.lstat(self.binding.local_path)
        except OSError as error:
            raise FailClosedError("bound evaluation pathname disappeared") from error
        if (not stat.S_ISREG(named.st_mode) or stat.S_ISLNK(named.st_mode)
                or _regular_identity(named) != self.identity):
            raise FailClosedError("bound evaluation pathname identity drift")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _hold_verified_score_asset(binding: _ScoreAssetBinding) -> _HeldScoreAsset:
    """Hash a preflight-bound asset before any parser is allowed to see it."""
    binding.data_root.assert_named_identity()
    try:
        before = os.lstat(binding.local_path)
    except OSError as error:
        raise FailClosedError("cannot stat reviewed evaluation asset") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise FailClosedError("reviewed evaluation asset is not a regular non-symlink")
    descriptor = -1
    try:
        descriptor = os.open(binding.local_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or _regular_identity(opened) != _regular_identity(before)
                or opened.st_size != binding.asset.expected_bytes):
            raise FailClosedError("evaluation asset changed before descriptor verification")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if _sha(_read_fd_all(descriptor)) != binding.asset.expected_sha256:
            raise FailClosedError("evaluation asset SHA mismatch before parser")
        held = _HeldScoreAsset(binding=binding, descriptor=descriptor, identity=_regular_identity(opened))
        held.reverify()
        return held
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


@dataclass
class _PrivateScoreSnapshot:
    """A parser-only, verified copy of an asset held through the score."""

    held: _HeldScoreAsset
    temporary_directory: Any
    path: Path

    def reverify(self) -> None:
        self.held.reverify()
        body, identity = _canonical_regular_bytes(
            self.path.parent, self.path.name, expected_mode=0o400,
        )
        if (identity[2] != self.held.binding.asset.expected_bytes
                or _sha(body) != self.held.binding.asset.expected_sha256):
            raise FailClosedError("private parser snapshot drift")

    def close(self) -> None:
        self.temporary_directory.cleanup()


def _private_verified_score_snapshot(held: _HeldScoreAsset) -> _PrivateScoreSnapshot:
    """Copy a held descriptor to a no-cache private parser pathname.

    NWB readers reopen pathnames internally.  The copy prevents a reader from
    observing a later replacement of the reviewed canonical file while the
    original descriptor remains available for post-forward revalidation.
    """
    import tempfile

    held.reverify()
    temporary_directory = tempfile.TemporaryDirectory(prefix="cell-d-equal-score-")
    path = Path(temporary_directory.name) / held.binding.local_path.name
    descriptor = -1
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600,
        )
        os.lseek(held.descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(held.descriptor, 1024 * 1024)
            if not block:
                break
            _write_full(descriptor, block)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_directory.cleanup()
        raise
    else:
        os.close(descriptor)
    snapshot = _PrivateScoreSnapshot(held=held, temporary_directory=temporary_directory, path=path)
    try:
        snapshot.reverify()
        return snapshot
    except BaseException:
        snapshot.close()
        raise


def _validate_source_normalizer_numerics(np: Any) -> tuple[Any, Any, Any, Any]:
    """Return only the closure-bound float32 source normalizers.

    This scorer must never open a cache or refit a target normalizer.  The
    literal float32 arrays below are part of its implementation closure and
    intentionally match the sealed source-authority semantic hashes.
    """
    t4_mean = np.asarray(SOURCE_T4_MEAN, dtype=np.float32)
    t4_std = np.asarray(SOURCE_T4_STD, dtype=np.float32)
    behavior_mean = np.asarray(SOURCE_BEHAVIOR_MEAN, dtype=np.float32)
    behavior_std = np.asarray(SOURCE_BEHAVIOR_STD, dtype=np.float32)
    if (t4_mean.shape != (4,) or t4_std.shape != (4,) or behavior_mean.shape != (2,)
            or behavior_std.shape != (2,) or not np.isfinite(t4_mean).all()
            or not np.isfinite(t4_std).all() or not np.isfinite(behavior_mean).all()
            or not np.isfinite(behavior_std).all() or not bool((t4_std > 0).all())
            or not bool((behavior_std > 0).all())):
        raise FailClosedError("closure-bound source normalizer numeric drift")
    # Explicit byte equality prevents a semantic-hash-shaped string from
    # silently standing in for altered float32 normalizer values.
    if (t4_mean.tobytes() != np.asarray(SOURCE_T4_MEAN, dtype=np.float32).tobytes()
            or t4_std.tobytes() != np.asarray(SOURCE_T4_STD, dtype=np.float32).tobytes()
            or behavior_mean.tobytes() != np.asarray(SOURCE_BEHAVIOR_MEAN, dtype=np.float32).tobytes()
            or behavior_std.tobytes() != np.asarray(SOURCE_BEHAVIOR_STD, dtype=np.float32).tobytes()):
        raise FailClosedError("source normalizer float32 representation drift")
    return t4_mean, t4_std, behavior_mean, behavior_std


def _validate_physical_gpu0_attestation(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the two independent GPU memory authorities exactly."""
    expected = {
        "cuda_visible_devices", "cuda_device_order", "logical_device", "uuid", "bdf", "name",
        "nvidia_smi_memory_total_mib", "torch_total_memory_bytes", "torch_version",
        "torch_cuda_version", "cudnn_version", "torchmetrics_version", "no_user_site",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FailClosedError("physical GPU0 attestation schema drift")
    required = {**FROZEN_GPU0, "torchmetrics_version": TORCHMETRICS_VERSION, "no_user_site": True}
    if dict(value) != required:
        raise FailClosedError("physical GPU0 attestation authority drift")
    return _json_copy(value)


def _physical_ordered_unit_digest(ordered_unit_ids: tuple[str, ...]) -> str:
    """The exact compact ordered-unit binding used for the side-feature axis."""
    if (not ordered_unit_ids or any(not isinstance(item, str) or not item for item in ordered_unit_ids)
            or len(set(ordered_unit_ids)) != len(ordered_unit_ids)):
        raise FailClosedError("physical ordered-unit identifiers drift")
    return _sha(json.dumps(ordered_unit_ids, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))


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
    """Bind the raw M30 carrier's rows to the parser's exact SUA unit axis.

    ``load_dandi688_session(..., signal_view='sua')`` constructs neural column
    ``k`` by iterating the NWB units table in source order and returns
    ``channel_ids = arange(source_unit_count)``.  The closure-bound
    ``compute_unit_side_features_uncached(..., feature_group='t4',
    signal_view='sua')`` constructs row ``k`` by that same units-table index.
    We require the returned metadata and the live contiguous channel proof,
    then publish a compact semantic binding rather than assuming axis order.
    """
    ids = np.ascontiguousarray(np.asarray(channel_ids, dtype=np.int64))
    expected = np.arange(neural_unit_count, dtype=np.int64)
    raw = np.ascontiguousarray(raw_t4, dtype=np.float32)
    if (not isinstance(session, str) or not session
            or type(neural_unit_count) is not int or neural_unit_count <= 0
            or type(source_unit_count) is not int or source_unit_count != neural_unit_count
            or ids.shape != (neural_unit_count,) or not np.array_equal(ids, expected)
            or raw.shape != (neural_unit_count, 4)
            or getattr(metadata, "feature_group", None) != "t4"
            or getattr(metadata, "pool_size", None) != 30):
        raise FailClosedError("raw M30 T4/SUA channel-order proof drift")
    return {
        "schema": "cell_d_equal_session_raw_t4_sua_axis_proof_v1",
        "session": session,
        "signal_view": "sua",
        "channel_ids_dtype": "int64",
        "channel_ids_sha256": _sha(ids.tobytes()),
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


def _prove_t4_row_order_against_sua_channels(
    np: Any,
    *,
    session: str,
    neural_unit_count: int,
    channel_ids: object,
    source_unit_count: object,
    raw_t4: Any,
    metadata: object,
) -> str:
    """Return the digest of the explicit raw-T4/SUA-axis proof payload."""
    return _sha(_json_bytes(_raw_t4_sua_axis_proof_payload(
        np, session=session, neural_unit_count=neural_unit_count, channel_ids=channel_ids,
        source_unit_count=source_unit_count, raw_t4=raw_t4, metadata=metadata,
    )))


def _last_bin_authority_from_behavior(np: Any, *, behavior: Any, starts: Any) -> tuple[Any, Any, str, str, int]:
    """Canonical target/mask/count proof for every governed last-bin query."""
    behavior_array = np.ascontiguousarray(behavior, dtype=np.float32)
    starts_array = np.ascontiguousarray(starts, dtype=np.int64)
    if (behavior_array.ndim != 2 or behavior_array.shape[1] != 2 or starts_array.ndim != 1
            or starts_array.size <= 0 or int(starts_array.min()) < 0
            or int(starts_array.max()) + PUBLIC_SPEC.window_bins > behavior_array.shape[0]):
        raise FailClosedError("last-bin target authority shape/window drift")
    target = np.ascontiguousarray(behavior_array[starts_array + (PUBLIC_SPEC.window_bins - 1)], dtype=np.float32)
    # Store bytes as uint8, not platform-dependent bool representation.
    valid_mask = np.ascontiguousarray(np.all(target != -1.0, axis=1), dtype=np.uint8)
    valid_count = int(valid_mask.sum())
    if target.shape != (starts_array.size, 2) or valid_count != int(starts_array.size):
        raise FailClosedError("last-bin target authority contains invalid governed windows")
    return target, valid_mask, _sha(target.tobytes()), _sha(valid_mask.tobytes()), valid_count


def _validate_last_bin_authority_arrays(
    np: Any, *, target: Any, valid_mask: Any, authority: SessionInputAuthority,
) -> None:
    """Require scored last-bin targets to equal the durable input evidence."""
    target_array = np.ascontiguousarray(target, dtype=np.float32)
    mask_array = np.ascontiguousarray(valid_mask, dtype=np.uint8)
    if (target_array.shape != (authority.n_windows, 2) or mask_array.shape != (authority.n_windows,)
            or _sha(target_array.tobytes()) != authority.last_bin_target_sha256
            or _sha(mask_array.tobytes()) != authority.last_bin_valid_mask_sha256
            or int(mask_array.sum()) != authority.last_bin_valid_count
            or authority.last_bin_valid_count != authority.n_windows):
        raise FailClosedError("physical governing last-bin target/mask authority drift")


@dataclass
class _PhysicalScoreSession:
    binding: _ScoreAssetBinding
    held: _HeldScoreAsset
    neural: Any
    behavior: Any
    calibration: Any
    starts: Any
    raw_t4: Any
    normalized_t4: Any
    last_bin_targets: Any
    last_bin_valid_mask: Any
    ordered_unit_ids: tuple[str, ...]
    authority: SessionInputAuthority


class PhysicalMatchedScoreBackend:
    """Reviewed future physical evaluator; construction itself is inert.

    This class intentionally owns no fallback to a mutable legacy scorer.  It
    is reached only by the root-only in-process capability after a durable
    attempt has been written.  Its inputs are descriptor-held, private parser
    snapshots; all model paths are exact immutable SWA bytes supplied by the
    validated fixed-authority map.
    """

    def __init__(self, *, root: Path, fixed_authorities: Mapping[str, AuthorityMaterial]) -> None:
        if set(fixed_authorities) != {spec.name for spec in FIXED_AUTHORITIES}:
            raise ValueError("physical score backend fixed-authority set drift")
        self._root = Path(root).absolute()
        self._fixed = dict(fixed_authorities)
        self._runtime: Mapping[str, Any] | None = None
        self._models: dict[str, Any] = {}
        self._state_at_load: dict[str, str] = {}
        self._sessions: dict[str, dict[str, _PhysicalScoreSession]] = {"within": {}, "external": {}}
        self._held: list[_HeldScoreAsset] = []
        self._closed = False
        self._attestation: Mapping[str, object] | None = None

    @staticmethod
    def _load_module(name: str, path: Path) -> Any:
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise FailClosedError(f"cannot build explicit runtime loader for {path.name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    def _load_runtime(self) -> Mapping[str, Any]:
        """Deferred Torch/data import and exact GPU0 attestation.

        This method is never reached by the public CLI, preflight builder, or
        static test path.  It checks the physical GPU contract before a model
        builder, parser, or target pathname is accessed.
        """
        if self._runtime is not None:
            return self._runtime
        import subprocess
        import sys

        if (os.environ.get("CUDA_VISIBLE_DEVICES") != FROZEN_GPU0["cuda_visible_devices"]
                or os.environ.get("CUDA_DEVICE_ORDER") != FROZEN_GPU0["cuda_device_order"]
                or sys.flags.no_user_site != 1):
            raise FailClosedError("physical scorer environment authority drift")
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
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise FailClosedError("physical scorer requires exactly one visible GPU0")
        try:
            output = subprocess.run(
                ["nvidia-smi", "--id=0", "--query-gpu=uuid,pci.bus_id,name,memory.total",
                 "--format=csv,noheader,nounits"],
                check=True, text=True, capture_output=True,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            raise FailClosedError("physical GPU0 nvidia-smi attestation failed") from error
        if len(output) != 1:
            raise FailClosedError("physical GPU0 nvidia-smi topology drift")
        columns = [item.strip() for item in output[0].split(",")]
        if len(columns) != 4:
            raise FailClosedError("physical GPU0 nvidia-smi field topology drift")
        uuid, bdf, name, nominal_mib = columns
        try:
            nominal_memory = int(nominal_mib)
        except ValueError as error:
            raise FailClosedError("physical GPU0 nominal memory is not integer") from error
        properties = torch.cuda.get_device_properties(0)
        attestation = {
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
            "logical_device": "cuda:0",
            "uuid": uuid,
            "bdf": bdf.upper(),
            "name": name,
            "nvidia_smi_memory_total_mib": nominal_memory,
            "torch_total_memory_bytes": int(properties.total_memory),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version(),
            "torchmetrics_version": str(torchmetrics.__version__),
            "no_user_site": True,
        }
        _validate_physical_gpu0_attestation(attestation)
        if properties.name != FROZEN_GPU0["name"]:
            raise FailClosedError("physical GPU0 Torch device-name drift")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        # Importing these closure-bound parser symbols is deferred until the
        # device contract is already attested.  Import itself opens no data;
        # actual parser invocation remains after the durable attempt and held
        # descriptor verification in ``_parse_one``.
        from mc_maze.multisession_datamodule import load_dandi688_session
        from mc_maze.unit_side_features import compute_unit_side_features_uncached
        pop_robust = self._load_module(
            "_cell_d_equal_score_pop_robust",
            self._root / "tfpd_exploration/src/tfpd_lane/pop_robust.py",
        )
        arm_common = self._load_module(
            "_cell_d_equal_score_arm_common",
            self._root / "tfpd_exploration/src/tfpd_lane/arm_common.py",
        )
        metric = self._load_module(
            "_cell_d_equal_score_matched_scorer",
            self._root / "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
        )
        equal_route = self._load_module(
            "_cell_d_equal_score_successor_route",
            self._root / "tfpd_exploration/src/cell_d_equal_session_v1.py",
        )
        if not callable(getattr(metric, "session_r2", None)):
            raise FailClosedError("closure-bound last-bin metric loader drift")
        self._attestation = _validate_physical_gpu0_attestation(attestation)
        self._runtime = {
            "np": np, "torch": torch, "pop_robust": pop_robust, "arm_common": arm_common,
            "session_r2": metric.session_r2, "equal_route": equal_route,
            "load_dandi688_session": load_dandi688_session,
            "compute_unit_side_features_uncached": compute_unit_side_features_uncached,
            "device": torch.device("cuda:0"),
        }
        return self._runtime

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

    def _require_cell_d_topology(self, model: Any, torch: Any, *, label: str) -> None:
        initialized, lazy = self._lazy_topology(model, torch)
        if initialized != 3_510_842 or lazy != (
            "decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight",
        ):
            raise FailClosedError(f"{label} exact Cell-D initialized/lazy topology drift")

    @staticmethod
    def _all_gradients_none(model: Any) -> bool:
        return all(parameter.grad is None for parameter in model.parameters())

    def _load_cell_d_swa_state(self, body: bytes) -> Mapping[str, Any]:
        """Load a sealed state only with a local weights-only allowlist."""
        import io

        runtime = self._load_runtime()
        torch = runtime["torch"]
        from torch.nn.parameter import UninitializedParameter
        from torch.torch_version import TorchVersion

        try:
            with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
                payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        except Exception as error:
            raise FailClosedError("sealed Cell-D SWA weights-only load failed") from error
        if not isinstance(payload, Mapping):
            raise FailClosedError("sealed Cell-D SWA payload root drift")
        state = payload.get("state_dict", payload.get("state"))
        manifest = payload.get("swa_manifest")
        # ``matched_scorer.build_swa_final_four`` writes the tensor payload
        # before appending its in-memory strict-reload diagnostic to the
        # terminal manifest.  The on-disk body therefore binds its immutable
        # arithmetic/component topology here; the sealed terminal already
        # independently binds the strict-reload proof in ``validate_sealed_lineage``.
        if (not isinstance(state, Mapping) or not state or not isinstance(manifest, Mapping)
                or manifest.get("uninitialized_lazy_tensor_count") != 2
                or manifest.get("optimizer_state_included") is not False
                or manifest.get("fp64_arithmetic") is not True
                or not isinstance(manifest.get("components"), list)
                or len(manifest["components"]) != 4):
            raise FailClosedError("sealed Cell-D SWA state/manifest drift")
        return state

    def _strict_fresh_cell_d(self, state: Mapping[str, Any], *, label: str) -> Any:
        """Strict-load a fresh exact graph, preserve lazy topology, then place it."""
        import random

        runtime = self._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        python_state, numpy_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(0)
        try:
            model = runtime["pop_robust"].build_population_robustness_model(seed=42, cell="D")
            self._require_cell_d_topology(model, torch, label=f"{label} fresh")
            if set(state) != set(model.state_dict()):
                raise FailClosedError(f"{label} state-key topology drift")
            model.load_state_dict(state, strict=True)
            self._require_cell_d_topology(model, torch, label=f"{label} strict-loaded CPU")
            model = model.to(runtime["device"])
            self._require_cell_d_topology(model, torch, label=f"{label} strict-loaded GPU")
            model.eval()
            if model.training or not self._all_gradients_none(model):
                raise FailClosedError(f"{label} eval/gradient invariant drift")
            return model
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(cuda_state, 0)

    def _ensure_models(self) -> None:
        if self._models:
            if set(self._models) != {"cell_d", "equal_session"}:
                raise FailClosedError("physical score model cache topology drift")
            return
        runtime = self._load_runtime()
        torch = runtime["torch"]
        # Cell-D's exact bytes were descriptor-verified by fixed authorities
        # before this backend was constructed.  Strict loading is still needed
        # because a body SHA alone cannot prove a usable graph topology.
        baseline_state = self._load_cell_d_swa_state(self._fixed["sealed_cell_d_swa"].body)
        baseline_model = self._strict_fresh_cell_d(baseline_state, label="sealed Cell-D SWA")

        # Reuse the successor route's strict SWA validator, which binds the
        # terminal launch, full 48x33925 schedule, final-four checkpoint graph,
        # lazy keys, and state SHA.  It receives immutable bytes only—never a
        # mutable model pathname or target input.
        route = runtime["equal_route"]
        terminal = self._fixed["successor_terminal"].value
        if not isinstance(terminal, Mapping) or not isinstance(terminal.get("launch_sha256"), str):
            raise FailClosedError("successor terminal launch binding unavailable")
        identity = route._build_run_identity(self._root)
        binding = route._checkpoint_binding(route.FULL_TRAIN_SPEC, identity, terminal["launch_sha256"])
        validator = route.PhysicalEqualSessionBackend(self._root)
        try:
            successor_payload = validator.validate_swa(
                self._fixed["successor_swa"].body, spec=route.FULL_TRAIN_SPEC, binding=binding,
            )
        finally:
            validator.close(None)
        successor_state = successor_payload.get("state_dict") if isinstance(successor_payload, Mapping) else None
        if (not isinstance(successor_state, Mapping)
                or successor_payload.get("state_dict_sha256") != SUCCESSOR_SWA_STATE_SHA256):
            raise FailClosedError("successor strict SWA state authority drift")
        successor_model = self._strict_fresh_cell_d(successor_state, label="equal-session SWA")

        self._models = {"cell_d": baseline_model, "equal_session": successor_model}
        self._state_at_load = {
            name: runtime["arm_common"].state_sha256(model) for name, model in self._models.items()
        }
        if any(not _is_sha(value) for value in self._state_at_load.values()):
            raise FailClosedError("strictly-loaded physical model state digest drift")

    def _parse_one(self, *, binding: _ScoreAssetBinding, flags: ScoreFlags) -> _PhysicalScoreSession:
        runtime = self._load_runtime()
        np, torch = runtime["np"], runtime["torch"]
        t4_mean, t4_std, behavior_mean, behavior_std = _validate_source_normalizer_numerics(np)
        held = _hold_verified_score_asset(binding)
        self._held.append(held)
        if binding.asset.surface == "within":
            flags.within_opened = True
        elif binding.asset.surface == "external":
            flags.external_opened = True
        else:
            raise FailClosedError("physical scorer received forbidden surface")
        snapshot = _private_verified_score_snapshot(held)
        try:
            record = runtime["load_dandi688_session"](
                snapshot.path,
                bin_size_ms=20, window_size=50, calibration_n_trials=30, max_trial_length=100,
                pad_value=-1.0, interpolate_trials=True, behavior_mean=behavior_mean,
                behavior_std=behavior_std, trial_result_filter="R",
                exclude_calibration_trials_from_windows=True, cache_dir=None, signal_view="sua",
            )
            raw_t4, metadata = runtime["compute_unit_side_features_uncached"](
                snapshot.path, feature_group="t4", pool_size=30, bin_size_ms=20,
                window_size=50, trial_result_filter="R", signal_view="sua",
            )
            snapshot.reverify()
        finally:
            snapshot.close()
        held.reverify()
        if getattr(record, "name", None) != binding.asset.session or getattr(record, "signal_view", None) != "sua":
            raise FailClosedError("no-cache parser session/signal-view drift")
        neural = np.ascontiguousarray(record.neural, dtype=np.float32)
        behavior = np.ascontiguousarray(record.behavior, dtype=np.float32)
        calibration = np.ascontiguousarray(record.calib_trials, dtype=np.float32)
        starts = np.ascontiguousarray(record.valid_starts, dtype=np.int64)
        channel_ids = getattr(record, "channel_ids", None)
        if (neural.ndim != 2 or behavior.shape != (neural.shape[0], 2)
                or calibration.shape != (30, 100, neural.shape[1]) or starts.ndim != 1 or starts.size <= 0
                or channel_ids is None or len(channel_ids) != neural.shape[1] or neural.shape[1] < 2
                or not np.isfinite(neural).all() or not np.isfinite(behavior).all()
                or not np.isfinite(calibration).all() or not bool((starts[:-1] < starts[1:]).all())
                or int(starts.min()) < 0 or int(starts.max()) + PUBLIC_SPEC.window_bins > neural.shape[0]):
            raise FailClosedError("no-cache parsed input shape/window/unit contract drift")
        raw = np.ascontiguousarray(raw_t4, dtype=np.float32)
        if raw.shape != (neural.shape[1], 4) or not np.isfinite(raw).all():
            raise FailClosedError("raw M30 T4 shape/finite drift")
        raw_t4_channel_alignment = _raw_t4_sua_axis_proof_payload(
            np,
            session=binding.asset.session,
            neural_unit_count=int(neural.shape[1]),
            channel_ids=channel_ids,
            source_unit_count=getattr(record, "source_unit_count", None),
            raw_t4=raw,
            metadata=metadata,
        )
        raw_t4_channel_alignment_sha256 = _sha(_json_bytes(raw_t4_channel_alignment))
        ordered_ids = tuple(f"{binding.asset.session}:channel:{int(item)}" for item in channel_ids)
        if len(set(ordered_ids)) != len(ordered_ids):
            raise FailClosedError("physical ordered unit IDs are not unique")
        raw_tensor = torch.as_tensor(raw, dtype=torch.float32, device=runtime["device"]).unsqueeze(0)
        raw_sha = _sha(raw.tobytes())
        mean_tensor = torch.as_tensor(t4_mean, dtype=torch.float32, device=runtime["device"])
        std_tensor = torch.as_tensor(t4_std, dtype=torch.float32, device=runtime["device"])
        # This is the exact frozen source-normalizer arithmetic used by the
        # capability wrapper in the reviewed routes: float32 ``(raw-mean)/std``
        # after the raw M30 carrier is derived.  It is intentionally written
        # locally so this scorer does not import an unrelated TFSR architecture
        # or accidentally acquire a learned normalizer path.
        aligned = ((raw_tensor - mean_tensor) / std_tensor).detach().clone()
        direct = ((raw_tensor - mean_tensor) / std_tensor).detach().clone()
        if not torch.equal(aligned, direct) or not bool(torch.isfinite(aligned).all().item()):
            raise FailClosedError("raw M30 T4 source-normalization mismatch")
        last_targets, last_mask, last_target_sha, last_mask_sha, last_count = _last_bin_authority_from_behavior(
            np, behavior=behavior, starts=starts,
        )
        authority = SessionInputAuthority(
            surface=binding.asset.surface, session=binding.asset.session, n_windows=int(starts.size),
            asset=binding.payload(), ordered_unit_digest=_physical_ordered_unit_digest(ordered_ids),
            neural_sha256=_sha(neural.tobytes()), behavior_sha256=_sha(behavior.tobytes()),
            query_start_sha256=_sha(starts.tobytes()), normalized_t4_sha256=_sha(
                aligned.detach().cpu().contiguous().numpy().tobytes()), raw_t4_sha256=raw_sha,
            last_bin_target_sha256=last_target_sha, last_bin_valid_mask_sha256=last_mask_sha,
            last_bin_valid_count=last_count, raw_t4_channel_alignment=raw_t4_channel_alignment,
            raw_t4_channel_alignment_sha256=raw_t4_channel_alignment_sha256,
            calibration_sha256=_sha(calibration.tobytes()), held_descriptor_identity=held.identity,
        )
        return _PhysicalScoreSession(
            binding=binding, held=held, neural=neural, behavior=behavior, calibration=calibration,
            starts=starts, raw_t4=raw_tensor, normalized_t4=aligned, ordered_unit_ids=ordered_ids,
            last_bin_targets=last_targets, last_bin_valid_mask=last_mask,
            authority=authority,
        )

    def resolve_inputs(
        self,
        *,
        within_roster: tuple[str, ...],
        external_roster: tuple[str, ...],
        within_assets: tuple[PreflightAsset, ...],
        external_assets: tuple[PreflightAsset, ...],
        flags: ScoreFlags,
    ) -> InputAuthorityEvidence:
        if self._closed:
            raise FailClosedError("closed physical score backend cannot resolve inputs")
        # Strict immutable model checks deliberately precede *all* target-root
        # access.  A malformed SWA therefore fails before a single NWB path is
        # formed or a source target descriptor is opened.
        self._ensure_models()
        if (tuple(asset.session for asset in within_assets) != within_roster
                or tuple(asset.session for asset in external_assets) != external_roster):
            raise FailClosedError("physical preflight asset roster/order drift")
        within_root = _ScoreDataRoot.from_environment(variable="SUBC_DATA_ROOT", label="within")
        external_root = _ScoreDataRoot.from_environment(variable="SUBM_DATA_ROOT", label="external")
        bindings = tuple(_ScoreAssetBinding.from_preflight(asset, within_root) for asset in within_assets) + tuple(
            _ScoreAssetBinding.from_preflight(asset, external_root) for asset in external_assets
        )
        records: list[SessionInputAuthority] = []
        for binding in bindings:
            parsed = self._parse_one(binding=binding, flags=flags)
            surface = binding.asset.surface
            if binding.asset.session in self._sessions[surface]:
                raise FailClosedError("physical duplicate parsed session")
            self._sessions[surface][binding.asset.session] = parsed
            records.append(parsed.authority)
        if (tuple(sorted(self._sessions["within"])) != within_roster
                or tuple(sorted(self._sessions["external"])) != external_roster):
            raise FailClosedError("physical parsed session roster drift")
        return InputAuthorityEvidence(
            records=tuple(sorted(records, key=lambda item: (item.surface, item.session))),
            source_normalizer_body_sha256=SOURCE_NORMALIZER_BODY_SHA256,
            t4_normalizer_semantic_sha256=SOURCE_T4_NORMALIZER_SHA256,
            behavior_normalizer_semantic_sha256=SOURCE_BEHAVIOR_NORMALIZER_SHA256,
            no_cache_readonly_adapter=True, shared_input_pass=True,
        )

    def _session_batch(self, session: _PhysicalScoreSession, starts: Sequence[int]) -> tuple[Any, Any, Any, Any]:
        runtime = self._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        if not starts:
            raise FailClosedError("physical score batch cannot be empty")
        neural = torch.from_numpy(np.stack([session.neural[start:start + 50] for start in starts])).to(runtime["device"])
        behavior = torch.from_numpy(np.stack([session.behavior[start:start + 50] for start in starts])).to(runtime["device"])
        calibration = torch.from_numpy(session.calibration).unsqueeze(0).to(runtime["device"])
        calibration = calibration.expand(neural.shape[0], -1, -1, -1)
        side = session.normalized_t4.expand(neural.shape[0], -1, -1).detach().clone()
        if (tuple(neural.shape) != (len(starts), 50, session.neural.shape[1])
                or tuple(behavior.shape) != (len(starts), 50, 2)
                or tuple(calibration.shape) != (len(starts), 30, 100, session.neural.shape[1])
                or tuple(side.shape) != (len(starts), session.neural.shape[1], 4)):
            raise FailClosedError("physical matched batch shape drift")
        return neural, behavior, calibration, side

    def _forward(self, *, model: Any, system: str, surface: str, neural: Any, calibration: Any,
                 side: Any, flags: ScoreFlags) -> Any:
        runtime = self._load_runtime()
        torch, pop_robust = runtime["torch"], runtime["pop_robust"]
        if model.training or not self._all_gradients_none(model):
            raise FailClosedError("physical matched scoring requires eval/no-gradient model")
        calls = {"post_pool": 0}

        def _b3s_hook(_module: Any, _inputs: Any, _output: Any) -> None:
            calls["post_pool"] += 1

        handle = model.id_encoder.post_pool.register_forward_hook(_b3s_hook)
        try:
            with pop_robust.dynamic_dropout_recorder() as recorder:
                with torch.no_grad():
                    if torch.is_grad_enabled():
                        raise FailClosedError("physical matched scorer autograd boundary drift")
                    output, identity = model(neural, calib_trials=calibration, side_features=side)
        finally:
            handle.remove()
        flags.record_forward(system, surface)
        if (calls["post_pool"] <= 0 or not torch.is_tensor(identity) or not torch.isfinite(identity).all().item()
                or recorder["uniform_calls"] != 0 or recorder["dropout_calls"] != []):
            raise FailClosedError("B3S/eval-no-dropout proof drift")
        if (not torch.is_tensor(output) or tuple(output.shape) != (neural.shape[0], 50, 2)
                or not torch.isfinite(output).all().item()):
            raise FailClosedError("physical matched forward output shape/finite drift")
        return output

    def _repeat_probe(self, *, model: Any, system: str, surface: str, session: _PhysicalScoreSession,
                      flags: ScoreFlags) -> bool:
        starts = tuple(int(item) for item in session.starts[:PUBLIC_SPEC.batch_size])
        neural, _behavior, calibration, side = self._session_batch(session, starts)
        first = self._forward(model=model, system=system, surface=surface, neural=neural,
                              calibration=calibration, side=side, flags=flags)
        second = self._forward(model=model, system=system, surface=surface, neural=neural,
                               calibration=calibration, side=side, flags=flags)
        if not self._load_runtime()["torch"].equal(first, second):
            raise FailClosedError("physical repeated fixed-batch forward is not bitwise equal")
        return True

    def score(
        self, *, system: str, surface: str, input_authority_sha256: str, flags: ScoreFlags,
    ) -> ModeEvidence:
        if system not in {"cell_d", "equal_session"} or surface not in {"within", "external"}:
            raise FailClosedError("physical system/surface score request drift")
        runtime = self._load_runtime()
        torch = runtime["torch"]
        model = self._models.get(system)
        sessions = self._sessions.get(surface)
        if model is None or not sessions:
            raise FailClosedError("physical scorer lacks strict model/shared session inputs")
        state_before = runtime["arm_common"].state_sha256(model)
        if state_before != self._state_at_load.get(system):
            raise FailClosedError("physical model state drifted before score")
        first_session = sessions[sorted(sessions)[0]]
        repeated = self._repeat_probe(model=model, system=system, surface=surface, session=first_session, flags=flags)
        rows: list[SessionScore] = []
        output_shape: tuple[int, int, int] | None = None
        for session_name in sorted(sessions):
            session = sessions[session_name]
            all_prediction: list[Any] = []
            all_target: list[Any] = []
            output_digest = hashlib.sha256()
            starts = tuple(int(item) for item in session.starts)
            for offset in range(0, len(starts), PUBLIC_SPEC.batch_size):
                chunk = starts[offset:offset + PUBLIC_SPEC.batch_size]
                neural, behavior, calibration, side = self._session_batch(session, chunk)
                output = self._forward(model=model, system=system, surface=surface, neural=neural,
                                       calibration=calibration, side=side, flags=flags)
                output_shape = tuple(int(item) for item in output.shape)
                output_digest.update(output.detach().cpu().contiguous().numpy().tobytes())
                valid_last = (behavior[:, -1, :] != -1.0).all(dim=-1)
                if not bool(valid_last.all().item()):
                    raise FailClosedError("governing last-bin score has invalid authorized query window")
                all_prediction.append(output[:, -1, :].detach().cpu())
                all_target.append(behavior[:, -1, :].detach().cpu())
            if not all_prediction:
                raise FailClosedError("physical session emitted no governing predictions")
            prediction_tensor = torch.cat(all_prediction)
            target_tensor = torch.cat(all_target)
            np = runtime["np"]
            target_array = np.ascontiguousarray(target_tensor.numpy(), dtype=np.float32)
            mask_array = np.ascontiguousarray(np.all(target_array != -1.0, axis=1), dtype=np.uint8)
            authority = session.authority
            if (target_array.shape != session.last_bin_targets.shape
                    or not np.array_equal(target_array, session.last_bin_targets)
                    or not np.array_equal(mask_array, session.last_bin_valid_mask)):
                raise FailClosedError("physical governing last-bin target/mask session drift")
            _validate_last_bin_authority_arrays(
                np, target=target_array, valid_mask=mask_array, authority=authority,
            )
            r2 = runtime["session_r2"](prediction_tensor, target_tensor)
            rows.append(SessionScore(
                session=session_name, n_windows=len(starts), governing_last_bin_r2=float(r2),
                prediction_sha256=output_digest.hexdigest(), input_authority_sha256=input_authority_sha256,
            ))
        if output_shape is None:
            raise FailClosedError("physical score surface emitted no tensor shape")
        state_after = runtime["arm_common"].state_sha256(model)
        if (state_after != state_before or not self._all_gradients_none(model) or model.training):
            raise FailClosedError("physical score changed model state/gradient/eval mode")
        return ModeEvidence(
            system=system, surface=surface, mode="aligned_native", sessions=tuple(rows),
            state_before_sha256=state_before, state_after_sha256=state_after, eval_mode=True,
            dropout_disabled=True, gradients_none=True, finite_output=True, output_shape=output_shape,
            repeat_fixed_batch_bitwise_equal=repeated, b3s_recomputed=True,
        )

    def reverify_after_forwards(self, *, flags: ScoreFlags) -> None:
        runtime = self._load_runtime()
        for held in self._held:
            held.reverify()
        for system, model in self._models.items():
            if (runtime["arm_common"].state_sha256(model) != self._state_at_load.get(system)
                    or not self._all_gradients_none(model) or model.training):
                raise FailClosedError("physical post-forward model proof drift")
        if (flags.backward_calls != 0 or flags.optimizer_calls != 0 or flags.update_calls != 0
                or flags.normalizer_refit or flags.formal_resolved or flags.formal_opened):
            raise FailClosedError("physical score crossed no-update/formal boundary")

    def resource_disclosure(self) -> Mapping[str, object]:
        runtime = self._load_runtime()
        torch = runtime["torch"]
        if (set(self._models) != {"cell_d", "equal_session"} or self._attestation is None
                or any(not sessions for sessions in self._sessions.values())):
            raise FailClosedError("physical resource disclosure requested before full matched score")
        return {
            "device_contract": dict(FROZEN_GPU0), "forward_only": True, "no_grad": True,
            "optimizer_steps": 0, "backward_calls": 0, "update_calls": 0,
            "normalizer_refit": False, "formal_or_held_out_opened": False,
            "same_inputs_for_both_models": True,
            "details": {
                "physical_gpu0_attestation": _json_copy(self._attestation),
                "torchmetrics_version": TORCHMETRICS_VERSION,
                "held_asset_count": len(self._held), "no_cache_adapter": True,
                "last_bin_only": True,
                "cuda_memory_allocated": int(torch.cuda.memory_allocated(0)),
                "cuda_memory_reserved": int(torch.cuda.memory_reserved(0)),
                "cuda_peak_memory_allocated": int(torch.cuda.max_memory_allocated(0)),
                "cuda_peak_memory_reserved": int(torch.cuda.max_memory_reserved(0)),
                "model_state_sha256": dict(self._state_at_load),
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for held in reversed(self._held):
            held.close()
        self._held.clear()
        self._sessions = {"within": {}, "external": {}}
        self._models.clear()


class NoLiveBackend:
    """Test-only fail-closed adapter for injected lifecycle failure tests."""

    def _deny(self) -> None:
        raise FailClosedError("physical matched scorer is unavailable without a reviewed injected adapter")

    def resolve_inputs(self, **_: Any) -> InputAuthorityEvidence:
        self._deny()

    def score(self, **_: Any) -> ModeEvidence:
        self._deny()

    def reverify_after_forwards(self, **_: Any) -> None:
        self._deny()

    def resource_disclosure(self) -> Mapping[str, object]:
        self._deny()

    def close(self) -> None:
        return None


def _publish_failure(artifact: ArtifactRoot, flags: ScoreFlags) -> None:
    if flags.terminal_published or artifact.has_name("terminal.json"):
        raise FailClosedError("failure receipt forbidden after terminal")
    if artifact.has_name("failure.json"):
        return
    payload = _failure_payload(flags)
    validate_failure_payload(payload)
    digest = artifact.publish_json("failure.json", payload)
    validate_failure_payload(artifact.reload_json("failure.json", digest))


def run_score_lifecycle(
    *,
    artifact: ArtifactRoot,
    identity: ScoreIdentity,
    execution_capability: ExecutionCapability,
    baseline: BaselineAuthority,
    within_roster: tuple[str, ...],
    external_roster: tuple[str, ...],
    within_assets: tuple[PreflightAsset, ...],
    external_assets: tuple[PreflightAsset, ...],
    backend: ScoreBackend,
    final_reverify: Callable[[], Mapping[str, object]],
) -> Mapping[str, Any]:
    """Run a no-partial-scientific-result matched score lifecycle.

    The durable attempt is intentionally published before the backend can
    resolve a target filename, open a data descriptor, or build a live model.
    Cell-D baseline replay precedes every successor score and is an exact
    per-session, last-bin parity blocker.
    """
    _require_execution_capability(execution_capability, identity)
    if artifact.topology != SCORE_TOPOLOGY:
        raise FailClosedError("score artifact topology drift")
    validate_baseline_rosters(baseline, within_roster=within_roster, external_roster=external_roster)
    if tuple(asset.session for asset in within_assets) != within_roster or tuple(asset.session for asset in external_assets) != external_roster:
        raise FailClosedError("lifecycle preflight asset roster drift")
    if tuple(asset.payload() for asset in within_assets) != tuple(asset.payload() for asset in sealed_within_assets()):
        raise FailClosedError("lifecycle within asset authority drift")
    flags = ScoreFlags()
    attempt_written = False
    try:
        flags.stage = "attempt"
        attempt = _attempt_payload(identity)
        validate_attempt_payload(attempt, identity)
        hashes: dict[str, str] = {"attempt.json": artifact.publish_json("attempt.json", attempt)}
        validate_attempt_payload(artifact.reload_json("attempt.json", hashes["attempt.json"]), identity)
        attempt_written = True

        # All target/evaluation input resolution is deliberately after the
        # durable attempt pair.  No path factory is called earlier by this
        # lifecycle; adapters receive only already-reviewed metadata bindings.
        flags.stage = "resolve_identical_inputs_after_attempt"
        flags.within_resolved = True
        flags.external_resolved = True
        evidence = backend.resolve_inputs(
            within_roster=within_roster, external_roster=external_roster,
            within_assets=within_assets, external_assets=external_assets, flags=flags,
        )
        _require_no_updates(flags)
        validate_input_authority_evidence(
            evidence, within_roster=within_roster, external_roster=external_roster,
            within_assets=within_assets, external_assets=external_assets,
        )
        input_payload = _input_authority_payload(evidence, identity)
        input_evidence = validate_input_authority_payload(input_payload, identity)
        if input_evidence != evidence:
            raise FailClosedError("input authority evidence roundtrip mismatch")
        hashes["input_authority.json"] = artifact.publish_json("input_authority.json", input_payload)
        persisted_input = validate_input_authority_payload(
            artifact.reload_json("input_authority.json", hashes["input_authority.json"]), identity,
        )
        if persisted_input != evidence:
            raise FailClosedError("durable input authority reload drift")

        cell_d: list[ModeEvidence] = []
        for surface, roster in (("within", within_roster), ("external", external_roster)):
            flags.stage = f"cell_d_{surface}_baseline_parity"
            result = backend.score(
                system="cell_d", surface=surface, input_authority_sha256=hashes["input_authority.json"], flags=flags,
            )
            _validate_mode_against_roster(
                result, system="cell_d", surface=surface, roster=roster,
                input_authority_sha256=hashes["input_authority.json"],
            )
            assert_cell_d_baseline_parity(result, baseline)
            cell_d.append(result)

        successor: list[ModeEvidence] = []
        for surface, roster in (("within", within_roster), ("external", external_roster)):
            flags.stage = f"equal_session_{surface}_aligned_native"
            result = backend.score(
                system="equal_session", surface=surface,
                input_authority_sha256=hashes["input_authority.json"], flags=flags,
            )
            _validate_mode_against_roster(
                result, system="equal_session", surface=surface, roster=roster,
                input_authority_sha256=hashes["input_authority.json"],
            )
            successor.append(result)
        _require_no_updates(flags)

        flags.stage = "post_forward_reverify"
        backend.reverify_after_forwards(flags=flags)
        _require_no_updates(flags)

        flags.stage = "score"
        score = build_score_payload(
            identity=identity, baseline=baseline, input_authority_sha256=hashes["input_authority.json"],
            cell_d=cell_d, equal_session=successor, resources=backend.resource_disclosure(), flags=flags,
        )
        score_body = _json_bytes(score)
        score_sha = _sha(score_body)
        flags.stage = "terminal_revalidation"
        final_closure = final_reverify()
        if _validate_closure(final_closure) != identity.closure:
            raise FailClosedError("launch/final closure differs before terminal")
        validate_attempt_payload(artifact.reload_json("attempt.json", hashes["attempt.json"]), identity)
        durable_input = validate_input_authority_payload(
            artifact.reload_json("input_authority.json", hashes["input_authority.json"]), identity,
        )
        if durable_input != evidence:
            raise FailClosedError("input authority changed before terminal")
        validate_score_payload(
            score, identity=identity, baseline=baseline, input_authority_sha256=hashes["input_authority.json"],
        )
        terminal = _terminal_payload(
            identity=identity, attempt_sha256=hashes["attempt.json"],
            input_authority_sha256=hashes["input_authority.json"], score_sha256=score_sha,
            score_payload=score, final_closure=final_closure,
        )
        validate_terminal_payload(terminal, identity=identity, score_payload=score, expected_score_sha256=score_sha)
        terminal_body = _json_bytes(terminal)

        def validate_group(bodies: Mapping[str, bytes], digests: Mapping[str, str]) -> None:
            if (bodies.get("score.json") != score_body or bodies.get("terminal.json") != terminal_body
                    or digests.get("score.json") != score_sha):
                raise FailClosedError("score/terminal group binding drift")
            try:
                grouped_score = json.loads(artifact.reload_pair("score.json", score_sha))
                grouped_terminal = json.loads(artifact.reload_pair("terminal.json", _sha(terminal_body)))
            except (TypeError, json.JSONDecodeError) as error:
                raise FailClosedError("score/terminal group JSON reload drift") from error
            if not isinstance(grouped_score, Mapping) or not isinstance(grouped_terminal, Mapping):
                raise FailClosedError("score/terminal group root drift")
            validate_score_payload(
                grouped_score, identity=identity, baseline=baseline,
                input_authority_sha256=hashes["input_authority.json"],
            )
            validate_terminal_payload(
                grouped_terminal, identity=identity, score_payload=grouped_score,
                expected_score_sha256=score_sha,
            )

        group_hashes = artifact.publish_group(
            {"score.json": score_body, "terminal.json": terminal_body}, post_publish=validate_group,
        )
        hashes.update(group_hashes)
        flags.terminal_published = True
        final_score = artifact.reload_json("score.json", hashes["score.json"])
        validate_score_payload(
            final_score, identity=identity, baseline=baseline, input_authority_sha256=hashes["input_authority.json"],
        )
        final_terminal = artifact.reload_json("terminal.json", hashes["terminal.json"])
        validate_terminal_payload(
            final_terminal, identity=identity, score_payload=final_score, expected_score_sha256=hashes["score.json"],
        )
        if artifact.has_name("failure.json"):
            raise FailClosedError("terminal cannot coexist with failure receipt")
        return final_terminal
    except BaseException:
        if attempt_written:
            try:
                _publish_failure(artifact, flags)
            except BaseException:
                # Preserve the original failed condition.  The group publisher
                # already rolls back any owned partial score leaves.
                pass
        raise
    finally:
        backend.close()


def dry_plan() -> dict[str, object]:
    """Static, no-Torch/no-data/no-write public plan."""
    return {
        "cell": CELL,
        "phase": PHASE,
        "status": "DRY_NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE",
        "score_spec": PUBLIC_SPEC.payload(),
        "metric_contract": dict(METRIC_CONTRACT),
        "baseline": {
            "relative": BASELINE_RECEIPT_RELATIVE,
            "sha256": BASELINE_RECEIPT_SHA256,
            "sealed_within_mean": 0.5696851710478464,
            "sealed_external_mean": 0.4179362749059995,
            "legacy_full_window_receipt_forbidden": "tfpd_exploration/results/pop_robust_v1/matched_score_receipt.json",
        },
        "within_asset_authority": {
            **WITHIN_PAIRED_VIEW_CLOSURE_BINDING,
            "selected_split": "val",
            "strict_rows": [asset.payload() for asset in sealed_within_assets()],
            "no_caller_supplied_within_asset_mapping": True,
        },
        "lineage": {
            "sealed_cell_d_terminal_sha256": SEALED_CELL_D_TERMINAL_SHA256,
            "sealed_cell_d_swa_sha256": SEALED_CELL_D_SWA_SHA256,
            "successor_terminal_sha256": SUCCESSOR_TERMINAL_SHA256,
            "successor_swa_sha256": SUCCESSOR_SWA_SHA256,
            "successor_swa_state_sha256": SUCCESSOR_SWA_STATE_SHA256,
        },
        "canonical_roots": {
            "authority": AUTHORITY_ROOT_RELATIVE,
            "score": SCORE_ROOT_RELATIVE,
            "required_fresh_before_reviewed_execution": True,
        },
        "physical_gpu0_contract": dict(FROZEN_GPU0),
        "verdict": {
            "STOP": "external_mean_delta < 0 OR within_mean_delta < -0.03",
            "CLEAR_GO": "external_mean_delta >= 0.03 AND within_mean_delta >= -0.03 AND external_median_delta > 0 AND external_positive_sessions >= 9/15",
            "otherwise": "HOLD",
        },
        "forbidden": [
            "target_optimizer", "backward", "update", "normalizer_refit", "formal_or_held_out",
            "zero_controls", "wrong_pair_controls", "destroyed_controls", "legacy_full_window_authority",
        ],
        "execution": "requires paired public flags and a root-reviewed in-process capability; public CLI has none",
    }


def execute_authorized(
    root: Path,
    *,
    capability: ExecutionCapability | None = None,
    backend: ScoreBackend | None = None,
) -> Mapping[str, Any]:
    """Root-only injection point; never callable from the public CLI alone."""
    # A public two-flag invocation has no opaque in-process capability.  Stop
    # before even reading authority roots, which is stronger than merely
    # stopping before a target/NWB path can be resolved.
    if capability is None:
        raise FailClosedError("root-reviewed in-process execution capability required before target resolution")
    # Metadata lineage and signed authority are checked before a fresh score
    # root is reserved.  Neither operation resolves a target/NWB pathname.
    fixed = verify_fixed_authorities(root)
    lineage = validate_sealed_lineage(fixed)
    identity = verify_score_authorization(root, fixed=fixed, lineage=lineage)
    _require_execution_capability(capability, identity)
    # The public CLI cannot supply ``capability`` and therefore stops above.
    # Once a root-reviewed in-process capability has bound the immutable
    # authority pair, the only default is this closure-bound, no-cache physical
    # backend—never a legacy mutable-path scorer.  Tests may still inject a
    # narrow synthetic backend at this protocol seam.
    if backend is None:
        backend = PhysicalMatchedScoreBackend(root=root, fixed_authorities=fixed)
    preflight_directory = root.absolute() / AUTHORITY_ROOT_RELATIVE
    preflight, _ = _read_authority_pair(preflight_directory, "official_preflight.json")
    bindings = _fixed_bindings(fixed)
    checked_preflight = validate_target_free_preflight(preflight, fixed_authorities=bindings)
    baseline = lineage["baseline"]
    if not isinstance(baseline, BaselineAuthority):
        raise FailClosedError("sealed baseline lineage object drift")
    within_roster = tuple(checked_preflight["within_roster"])
    external_roster = tuple(checked_preflight["external_roster"])
    within_assets = _preflight_assets_from_payload(
        checked_preflight["within_assets"], surface="within", roster=within_roster,
    )
    external_assets = _preflight_assets_from_payload(
        checked_preflight["external_assets"], surface="external", roster=external_roster,
    )
    parent, name = canonical_score_parent(root)
    artifact = reserve_artifact_root(parent, name, topology=SCORE_TOPOLOGY)

    def final_reverify() -> Mapping[str, object]:
        final_fixed = verify_fixed_authorities(root)
        final_lineage = validate_sealed_lineage(final_fixed)
        final_identity = verify_score_authorization(root, fixed=final_fixed, lineage=final_lineage)
        if final_identity.payload() != identity.payload():
            raise FailClosedError("fixed lineage/authority changed after score forwards")
        return implementation_closure(root)

    return run_score_lifecycle(
        artifact=artifact, identity=identity, execution_capability=capability, baseline=baseline,
        within_roster=within_roster, external_roster=external_roster, within_assets=within_assets,
        external_assets=external_assets, backend=backend, final_reverify=final_reverify,
    )
