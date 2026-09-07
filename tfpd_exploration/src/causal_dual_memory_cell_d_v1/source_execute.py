"""Standard-library-only CDM-D strict-source execution contract.

This module owns the future source-smoke and source-constructibility lifecycle,
but deliberately contains no Torch, NWB, model, datamodule, or CUDA import.
The public CLI can therefore remain an inert static entry point.  The physical
backend lives in :mod:`source_execute_physical` and is reachable only through
an opaque in-process root-reviewed capability.

The implementation keeps two boundaries separate:

* immutable fixed authorities and implementation closure are descriptor-read
  before a result root is reserved; and
* strict-27 source resolution, checkpoint loading, CUDA, and model forwards
  occur only after an immutable attempt is durable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


CELL = "CAUSAL_DUAL_MEMORY_CELL_D_V1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V1_20260825.md"
WORKORDER_SHA256 = "644ad075ee1f0d7bcd9d97e80ed5b765190bc9b4de278baaef309d0017fe9981"

STAGE0_WORKORDER_SHA256 = "5d4a22bf8d1700b4230f2f9970c9ff98b2e6a31d0a9bc1bd828e6d844ff1c6fc"
STAGE0_CLOSURE_SHA256 = "3ab6d3de931e4630cb9c80b07e25e3b38af4c444f3c937d3d28560b596c29590"
SOURCE_AUDIT_WORKORDER_SHA256 = "8489fcbf1c83d5c174fb10e440ecd95387b46a8dca4466a3ac355d0024bb0260"
SOURCE_AUDIT_CLOSURE_SHA256 = "cfc192c45f674ccd4d56f19e3d5a58dc9146cf2fb02b02d4f561e7e0bb2db4dc"

STRICT_MANIFEST_RELATIVE = "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
STRICT_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"

SEALED_CELL_D_TERMINAL_RELATIVE = (
    "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json"
)
SEALED_CELL_D_TERMINAL_SHA256 = "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442"
SEALED_CELL_D_SWA_RELATIVE = "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt"
SEALED_CELL_D_SWA_SHA256 = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
THETA_RECEIPT_RELATIVE = "tfpd_exploration/results/sparsification_theta_authority_v1/theta_authority_receipt.json"
THETA_RECEIPT_SHA256 = "d023dd632c4717443f1f55e924a09be1747fc58d5c30a8fb6fa38f4b7b117184"
THETA_ARTIFACT_RELATIVE = "tfpd_exploration/results/sparsification_theta_authority_v1/theta_authority.pt"
THETA_ARTIFACT_SHA256 = "cef39dc8220aa253214963a32e5457dede1045e64b158e37fc267a6fb4146319"

SEALED_T4_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
SEALED_T4_MEAN_FLOAT32 = (
    0.04627712443470955,
    0.4544036388397217,
    1.3432163000106812,
    10.150517463684082,
)
SEALED_T4_STD_FLOAT32 = (
    1.126278281211853,
    1.284820556640625,
    1.2352101802825928,
    9.115250587463379,
)
SEALED_BEHAVIOR_NORMALIZER_SHA256 = "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
SEALED_BEHAVIOR_MEAN = (-0.001148765324614942, 0.002653369214385748)
SEALED_BEHAVIOR_STD = (8.63547420501709, 8.086690902709961)

SOURCE_SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v1"
SOURCE_GATE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v1"
SOURCE_SMOKE_SESSION = "sub-C_ses-CO-20131003"
STRICT_SOURCE_COUNT = 27
GROUP_COUNT = 4
BREADTH_MIN_PASSING_SESSIONS = 14
FAIL_FAST_BUDGET_ORDER = (30, 10, 4)

# The sealed historical fixed-ridge support fit and the online CDM-D update
# intentionally use different, explicitly disclosed rate domains.  The former
# is exact-duration spike counting; the latter is the Stage-0 native 20-ms
# scalar-rate capability.  Neither may silently substitute for the other.
INITIAL_SUPPORT_RATE_DOMAIN = "exact_duration_raw_spike_counts_divided_by_exposure_seconds"
ONLINE_UPDATE_RATE_DOMAIN = "mean(native_20ms_binned_counts)/0.020"

# These two source-session exceptions are part of the sealed theta authority:
# the physical channels are retained, but their tuning direction is undefined
# and their complementary assignment is exactly ``-1``.  Keeping the known
# topology literal here prevents a future parser from silently discarding the
# rows or treating an unexpected source session as equivalent.
KNOWN_THETA_INVALID_UNIT_COUNTS: Mapping[str, int] = {
    "sub-C_ses-CO-20131101": 1,
    "sub-C_ses-CO-20131220": 2,
}

# The strict SWA loader lives behind the reviewed execution-only executor
# seam.  These constants deliberately bind its receipt proof to the same
# sealed Cell-D graph rather than accepting a merely SHA-shaped state claim.
SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS = 3_510_842
SEALED_CELL_D_UNINITIALIZED_LAZY_KEYS = (
    "decoder.fc_id_in.0.bias",
    "decoder.fc_id_in.0.weight",
)
SEALED_SWA_LOAD_PROOF_SCHEMA = "causal_dual_memory_cell_d_source_execution_swa_load_proof_v1"


class SourceExecutionError(RuntimeError):
    """Fail closed for malformed source-execution authority or lifecycle state."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExecutionError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value),
             f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative(relative: str) -> Path:
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."},
             "source-execution path is not a safe relative pathname")
    return path


@dataclass(frozen=True)
class FixedAsset:
    """One fixed local asset bound before any source path is resolved."""

    label: str
    relative: str
    sha256: str
    mode: int
    requires_sidecar: bool

    def __post_init__(self) -> None:
        _require(self.label and self.label.replace("_", "").isalnum(), "fixed-asset label drift")
        _safe_relative(self.relative)
        _sha(self.sha256, f"{self.label} SHA")
        _require(self.mode in (0o444, 0o664), "fixed-asset mode must be exact reviewed literal")

    def payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "relative": self.relative,
            "sha256": self.sha256,
            "mode": self.mode,
            "requires_sidecar": self.requires_sidecar,
        }


FIXED_ASSETS: tuple[FixedAsset, ...] = (
    FixedAsset("strict_manifest", STRICT_MANIFEST_RELATIVE, STRICT_MANIFEST_SHA256, 0o664, False),
    FixedAsset("sealed_cell_d_terminal", SEALED_CELL_D_TERMINAL_RELATIVE, SEALED_CELL_D_TERMINAL_SHA256, 0o444, True),
    FixedAsset("sealed_cell_d_swa", SEALED_CELL_D_SWA_RELATIVE, SEALED_CELL_D_SWA_SHA256, 0o444, True),
    FixedAsset("theta_receipt", THETA_RECEIPT_RELATIVE, THETA_RECEIPT_SHA256, 0o444, True),
    FixedAsset("theta_artifact", THETA_ARTIFACT_RELATIVE, THETA_ARTIFACT_SHA256, 0o444, True),
)


@dataclass(frozen=True)
class BoundAsset:
    """Bytes read through a held parent descriptor with exact identity evidence."""

    asset: FixedAsset
    body: bytes = field(repr=False, compare=False)
    parent_device: int
    parent_inode: int

    def payload(self) -> dict[str, object]:
        return {
            **self.asset.payload(),
            "bytes": len(self.body),
            "parent_device": self.parent_device,
            "parent_inode": self.parent_inode,
        }


def _read_fd_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        part = os.read(fd, 1 << 20)
        if not part:
            return b"".join(chunks)
        chunks.append(part)


def _open_held_directory(root: Path, relative_parent: Path) -> tuple[int, tuple[int, int]]:
    target = Path(root).absolute() / relative_parent
    try:
        named = os.lstat(target)
        _require(stat.S_ISDIR(named.st_mode) and not stat.S_ISLNK(named.st_mode),
                 "fixed-asset parent is not a regular non-symlink directory")
        fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise SourceExecutionError("fixed-asset parent cannot be opened descriptor-safely") from error
    held = os.fstat(fd)
    identity = (int(named.st_dev), int(named.st_ino))
    try:
        _require(stat.S_ISDIR(held.st_mode) and (int(held.st_dev), int(held.st_ino)) == identity,
                 "fixed-asset parent identity changed before held read")
    except BaseException:
        os.close(fd)
        raise
    return fd, identity


def _read_held_leaf(fd: int, leaf: str, *, mode: int) -> bytes:
    _require(Path(leaf).name == leaf and leaf not in {"", ".", ".."}, "held leaf name drift")
    try:
        leaf_fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    except OSError as error:
        raise SourceExecutionError("fixed asset leaf cannot be opened descriptor-safely") from error
    try:
        info = os.fstat(leaf_fd)
        _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == mode,
                 "fixed asset leaf type/mode drift")
        return _read_fd_all(leaf_fd)
    finally:
        os.close(leaf_fd)


def descriptor_read_fixed_asset(root: Path, asset: FixedAsset) -> BoundAsset:
    """Read one exact fixed asset and optional basename-only sidecar.

    The body and sidecar use one held parent directory descriptor, and a final
    named/held identity comparison rejects parent replacement during the read.
    """
    _require(isinstance(asset, FixedAsset), "fixed asset must be typed")
    relative = _safe_relative(asset.relative)
    fd, identity = _open_held_directory(Path(root), relative.parent)
    try:
        body = _read_held_leaf(fd, relative.name, mode=asset.mode)
        _require(sha256_bytes(body) == asset.sha256, "fixed asset body SHA drift")
        if asset.requires_sidecar:
            expected = f"{asset.sha256}  {relative.name}\n".encode("ascii")
            _require(_read_held_leaf(fd, f"{relative.name}.sha256", mode=asset.mode) == expected,
                     "fixed asset canonical sidecar drift")
        named_after = os.lstat(Path(root).absolute() / relative.parent)
        held_after = os.fstat(fd)
        _require(stat.S_ISDIR(named_after.st_mode) and not stat.S_ISLNK(named_after.st_mode)
                 and (int(named_after.st_dev), int(named_after.st_ino)) == identity
                 and (int(held_after.st_dev), int(held_after.st_ino)) == identity,
                 "fixed asset parent changed during held read")
        return BoundAsset(asset, body, *identity)
    finally:
        os.close(fd)


def descriptor_read_fixed_assets(root: Path) -> dict[str, BoundAsset]:
    bound = {asset.label: descriptor_read_fixed_asset(root, asset) for asset in FIXED_ASSETS}
    _require(tuple(bound) == tuple(asset.label for asset in FIXED_ASSETS), "fixed-asset topology drift")
    return bound


def parse_strict_train_roster(manifest_body: bytes) -> tuple[str, ...]:
    """Validate manifest topology without ever deriving non-train paths."""
    try:
        value = json.loads(manifest_body)
    except json.JSONDecodeError as error:
        raise SourceExecutionError("strict source manifest is not JSON") from error
    _require(isinstance(value, Mapping), "strict source manifest root drift")
    _require(value.get("schema_version") == 1 and value.get("task") == "CO"
             and value.get("split_counts") == [27, 6, 6], "strict source manifest schema drift")
    splits = value.get("session_splits")
    _require(isinstance(splits, Mapping), "strict source manifest split mapping drift")
    train = splits.get("train")
    _require(isinstance(train, list) and len(train) == STRICT_SOURCE_COUNT
             and all(isinstance(item, str) and item.startswith("sub-C_ses-CO-")
                     and "/" not in item and "\\" not in item for item in train)
             and len(set(train)) == len(train), "strict source train roster drift")
    # Validate static val/test topology but deliberately leave those strings
    # inert: this function never constructs a pathname from them.
    for name in ("val", "test"):
        inert = splits.get(name)
        _require(isinstance(inert, list) and len(inert) == 6 and all(isinstance(item, str) for item in inert)
                 and len(set(inert)) == len(inert), f"strict source {name} split topology drift")
        _require(not (set(train) & set(inert)), f"strict source train/{name} overlap")
    return tuple(train)


def roster_sha256(roster: Sequence[str]) -> str:
    values = tuple(roster)
    _require(len(values) == STRICT_SOURCE_COUNT and len(set(values)) == len(values), "strict source roster digest input drift")
    return sha256_bytes(_json_bytes(list(values)))


@dataclass(frozen=True)
class SourceNormalizers:
    """Literal normalizer authority; no caller-provided moments are accepted."""

    t4_semantic_sha256: str = SEALED_T4_NORMALIZER_SHA256
    t4_mean_float32: tuple[float, ...] = SEALED_T4_MEAN_FLOAT32
    t4_std_float32: tuple[float, ...] = SEALED_T4_STD_FLOAT32
    behavior_semantic_sha256: str = SEALED_BEHAVIOR_NORMALIZER_SHA256
    behavior_mean: tuple[float, ...] = SEALED_BEHAVIOR_MEAN
    behavior_std: tuple[float, ...] = SEALED_BEHAVIOR_STD

    def __post_init__(self) -> None:
        _require(self.t4_semantic_sha256 == SEALED_T4_NORMALIZER_SHA256
                 and self.t4_mean_float32 == SEALED_T4_MEAN_FLOAT32
                 and self.t4_std_float32 == SEALED_T4_STD_FLOAT32,
                 "sealed OLS T4 normalizer numeric/semantic drift")
        _require(self.behavior_semantic_sha256 == SEALED_BEHAVIOR_NORMALIZER_SHA256
                 and self.behavior_mean == SEALED_BEHAVIOR_MEAN
                 and self.behavior_std == SEALED_BEHAVIOR_STD,
                 "sealed behavior normalizer numeric/semantic drift")
        _require(all(isinstance(item, float) for item in self.t4_mean_float32 + self.t4_std_float32
                     + self.behavior_mean + self.behavior_std)
                 and all(item > 0.0 for item in self.t4_std_float32 + self.behavior_std),
                 "normalizer literal dtype/range drift")

    def payload(self) -> dict[str, object]:
        return {
            "t4_semantic_sha256": self.t4_semantic_sha256,
            "t4_mean_float32": list(self.t4_mean_float32),
            "t4_std_float32": list(self.t4_std_float32),
            "behavior_semantic_sha256": self.behavior_semantic_sha256,
            "behavior_mean": list(self.behavior_mean),
            "behavior_std": list(self.behavior_std),
            "raw_before_t4_normalization": True,
            "behavior_destandardized_before_displacement": True,
            "normalizer_refit": False,
        }


SEALED_NORMALIZERS = SourceNormalizers()


COMPATIBLE_DEVICE_PROFILES: Mapping[str, Mapping[str, object]] = {
    "gpu0": {
        "cuda_visible_devices": "0", "cuda_device_order": "PCI_BUS_ID", "logical_device": "cuda:0",
        "uuid": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9", "bdf": "00000000:01:00.0",
        "name": "NVIDIA GeForce RTX 3090", "nvidia_smi_memory_total_mib": 24_576,
        "torch_total_memory_bytes": 25_435_111_424, "torch_version": "2.5.1.post303",
        "torch_cuda_version": "11.8", "cudnn_version": 90_300,
    },
    "gpu1": {
        "cuda_visible_devices": "1", "cuda_device_order": "PCI_BUS_ID", "logical_device": "cuda:0",
        "uuid": "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86", "bdf": "00000000:03:00.0",
        "name": "NVIDIA GeForce RTX 3090", "nvidia_smi_memory_total_mib": 24_576,
        "torch_total_memory_bytes": 25_438_126_080, "torch_version": "2.5.1.post303",
        "torch_cuda_version": "11.8", "cudnn_version": 90_300,
    },
}


def validate_compatible_device_profile(profile: Mapping[str, object]) -> dict[str, object]:
    observed = dict(profile) if isinstance(profile, Mapping) else None
    _require(observed is not None, "selected device profile must be a mapping")
    for expected in COMPATIBLE_DEVICE_PROFILES.values():
        if observed == dict(expected):
            return dict(expected)
    raise SourceExecutionError("selected device profile is not an exact reviewed RTX-3090 authority")


def validate_selected_device_environment(profile: Mapping[str, object], environ: Mapping[str, str] | None = None) -> dict[str, object]:
    selected = validate_compatible_device_profile(profile)
    values = os.environ if environ is None else environ
    _require(values.get("CUDA_VISIBLE_DEVICES") == selected["cuda_visible_devices"],
             "selected device CUDA_VISIBLE_DEVICES drift")
    _require(values.get("CUDA_DEVICE_ORDER") == selected["cuda_device_order"],
             "selected device CUDA_DEVICE_ORDER drift")
    return {
        "CUDA_VISIBLE_DEVICES": selected["cuda_visible_devices"],
        "CUDA_DEVICE_ORDER": selected["cuda_device_order"],
        "logical_device": selected["logical_device"],
        "profile": selected,
    }


def validate_runtime_attestation(profile: Mapping[str, object], observed: Mapping[str, object]) -> dict[str, object]:
    """Validate the physical execution proof after CUDA is intentionally initialized."""
    selected = validate_compatible_device_profile(profile)
    _require(isinstance(observed, Mapping), "runtime device attestation must be a mapping")
    exact = {
        **selected,
        "visible_devices": 1,
        "attested": True,
        "torch_cuda_matmul_allow_tf32": False,
        "torch_cudnn_allow_tf32": False,
    }
    _require(dict(observed) == exact, "physical device/runtime/TF32 authority drift")
    return dict(exact)


def validate_sealed_cell_d_swa_load_proof(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the reviewed executor's strict sealed-Cell-D load evidence.

    ``source_execute`` intentionally remains Torch-free, so it does not decode
    the SWA itself.  Instead it requires the physical executor to supply the
    result of a fresh strict state load/re-hash.  This narrow receipt surface
    is stronger than a boolean-only acknowledgement: it binds the immutable
    terminal/SWA bytes, the exact live/lazy parameter topology, the recomputed
    state digest, and eval/no-grad forward invariants.
    """
    _require(isinstance(value, Mapping), "sealed Cell-D SWA proof must be a mapping")
    expected_keys = {
        "schema",
        "sealed_terminal_sha256",
        "sealed_swa_sha256",
        "fresh_strict_load",
        "recomputed_state_dict_sha256",
        "initialized_trainable_parameters",
        "uninitialized_lazy_keys",
        "model_eval",
        "no_grad",
        "finite_forward",
        "repeated_fixed_forward_bitwise_equal",
        "model_state_unchanged",
        "dynamic_dropout_calls",
    }
    _require(set(value) == expected_keys, "sealed Cell-D SWA proof schema topology drift")
    _require(
        value.get("schema") == SEALED_SWA_LOAD_PROOF_SCHEMA
        and value.get("sealed_terminal_sha256") == SEALED_CELL_D_TERMINAL_SHA256
        and value.get("sealed_swa_sha256") == SEALED_CELL_D_SWA_SHA256
        and value.get("fresh_strict_load") is True
        and _sha(value.get("recomputed_state_dict_sha256"), "sealed Cell-D SWA recomputed state SHA")
        == value.get("recomputed_state_dict_sha256")
        and value.get("initialized_trainable_parameters") == SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS
        and value.get("uninitialized_lazy_keys") == list(SEALED_CELL_D_UNINITIALIZED_LAZY_KEYS)
        and value.get("model_eval") is True
        and value.get("no_grad") is True
        and value.get("finite_forward") is True
        and value.get("repeated_fixed_forward_bitwise_equal") is True
        and value.get("model_state_unchanged") is True
        and type(value.get("dynamic_dropout_calls")) is int
        and value["dynamic_dropout_calls"] == 0,
        "sealed Cell-D strict SWA load/state/lazy/eval proof drift",
    )
    return dict(value)


@dataclass(frozen=True)
class SourceExecutionSpec:
    kind: str
    root_relative: str
    smoke_session: str | None
    smoke_budget: int | None
    smoke_audit_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(self.kind in {"source_smoke", "source_gate"}, "source execution spec kind drift")
        _safe_relative(self.root_relative)
        if self.kind == "source_smoke":
            _require(self.smoke_session == "sub-C_ses-CO-20131003" and self.smoke_budget == 30
                     and self.smoke_audit_positions == (30, 31), "fixed source smoke identity drift")
        else:
            _require(self.smoke_session is None and self.smoke_budget is None and not self.smoke_audit_positions,
                     "full source gate may not inherit smoke-only fields")

    def payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "root_relative": self.root_relative,
            "source_only": True,
            "target_optimizer_steps": 0,
            "within_external_formal_target_forbidden": True,
            "fail_fast_budget_order": list(FAIL_FAST_BUDGET_ORDER) if self.kind == "source_gate" else [],
            "breadth_min_passing_sessions": BREADTH_MIN_PASSING_SESSIONS if self.kind == "source_gate" else None,
            "smoke": None if self.kind != "source_smoke" else {
                "session": self.smoke_session,
                "budget": self.smoke_budget,
                "support_positions": list(range(30)),
                "audit_positions": list(self.smoke_audit_positions),
                "group_count": GROUP_COUNT,
                "max_endpoints_per_forward_chunk": 128,
            },
        }


SOURCE_SMOKE_SPEC = SourceExecutionSpec(
    "source_smoke", SOURCE_SMOKE_ROOT_RELATIVE, SOURCE_SMOKE_SESSION, 30, (30, 31),
)
SOURCE_GATE_SPEC = SourceExecutionSpec("source_gate", SOURCE_GATE_ROOT_RELATIVE, None, None, ())


# The accepted dependency closures are intentionally spelled out rather than
# reconstructed from a mutable module-level path list.
_STAGE0_ACCEPTED_ROWS: tuple[tuple[str, str], ...] = (
    ("tfpd_exploration/docs/WORKORDER_CDM_D_STAGE0_20260825.md", STAGE0_WORKORDER_SHA256),
    ("tfpd_exploration/src/causal_dual_memory_cell_d_v1/__init__.py", "4b5226cdbde741789d0e26acb40e1742818a567f5e71e38aa8b46d99ad91e336"),
    ("tfpd_exploration/src/causal_dual_memory_cell_d_v1/plan.py", "08f384b5151af4d883516c67639c2afded2b6a8da9a4e9dd1c012b022923079c"),
    ("tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py", "968ff2a4f91f3f73161daa97ad65e833f4713f45450631f4574144639e44fffc"),
    ("tfpd_exploration/scripts/run_causal_dual_memory_cell_d_stage0.py", "393cfaca1a4facd49e02b41b31dc67d3ec66914755c9a7c85bf543ce0a3de72e"),
    ("tfpd_exploration/tests/test_causal_dual_memory_cell_d_stage0.py", "decf7622c724c9c4a8dbdf1b6cabc5acaefc2336fedbfb9cf5421e035f55b276"),
    ("sua_exploration/mc_maze/d_optimal_calibration_design.py", "1281a4ed7a78d5ce7f3e531f29a0d9de4dc9af48e900941b3edb8fe9bd3b6fb2"),
    ("tfpd_exploration/src/calibration_budget_comparators_v1.py", "98af15205c7a0bcba26cb063eff9f0562305e7facb5feb77cb3a4d1e0355cffa"),
)
_SOURCE_AUDIT_ACCEPTED_ROWS: tuple[tuple[str, str], ...] = (
    *_STAGE0_ACCEPTED_ROWS,
    ("tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_AUDIT_20260825.md", SOURCE_AUDIT_WORKORDER_SHA256),
    ("sua_exploration/mc_maze/pseudo_label_carrier_gate.py", "35619557d5722dae0c42298d0d4830550055c682b5e5548679b67e87f30adb91"),
    ("tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_adapter.py", "0d732be38a255e9c7a9cd83de1190749fe3b78c6dc94679fe3bc411884d66af3"),
    ("tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_audit.py", "d7bb00393440ac3e5f343f7bb70716b96ed518741b654b4b85439571898a2d76"),
    ("tfpd_exploration/src/causal_dual_memory_cell_d_v1/physical.py", "ae78ae86e81536127d3759a5052a0757a4e20025332b487c94e182bee9218c0a"),
    ("tfpd_exploration/src/causal_dual_memory_cell_d_v1/lifecycle.py", "fc0ae8c7306a8cdbfb9d0d2c1012789a5d0f72b224d7c58e9ed7e649acc72e58"),
    ("tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_audit.py", "4b7974de4968d67873ff7d6da7f4336817daba2f7485b489d0d1799a1ccb735a"),
    ("tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_audit.py", "90c6150a87b6c7d80a150823d787239f61ad235d70364ebcc4642e167397b13c"),
)

_EXECUTION_ADDITIONAL_PATHS: tuple[str, ...] = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/cell_d_equal_session_v1.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/__init__.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/plan.py",
    # The deferred physical route imports the audited same-prefix v2
    # recovery and source-prior helpers directly.  These remain explicit
    # closure leaves rather than ambient package dependencies.
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    # These two files are descriptor-executed under private module names, so
    # importing them does not execute the unrelated tfpd_lane package init.
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution.py",
)


def _regular_sha256(root: Path, relative: str) -> str:
    path = Path(root).absolute() / _safe_relative(relative)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise SourceExecutionError(f"closure path is inaccessible: {relative}") from error
    _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
             f"closure path is not a regular non-symlink: {relative}")
    with open(path, "rb") as handle:
        return sha256_bytes(handle.read())


def _accepted_rows(root: Path, rows: Sequence[tuple[str, str]], expected_closure: str, label: str) -> list[dict[str, str]]:
    observed: list[dict[str, str]] = []
    for relative, expected in rows:
        actual = _regular_sha256(root, relative)
        _require(actual == expected, f"accepted {label} dependency byte drift: {relative}")
        observed.append({"path": relative, "sha256": actual})
    _require(sha256_bytes(_json_bytes(observed)) == expected_closure,
             f"accepted {label} closure reconstruction drift")
    return observed


def execution_closure_payload(root: Path) -> dict[str, object]:
    """Reconstruct exact predecessor and current execution closure without globs."""
    stage0_rows = _accepted_rows(root, _STAGE0_ACCEPTED_ROWS, STAGE0_CLOSURE_SHA256, "Stage-0")
    audit_rows = _accepted_rows(root, _SOURCE_AUDIT_ACCEPTED_ROWS, SOURCE_AUDIT_CLOSURE_SHA256, "Source-Audit")
    extra_paths = _EXECUTION_ADDITIONAL_PATHS
    all_paths: list[str] = []
    for relative in [*(item["path"] for item in audit_rows), *extra_paths]:
        if relative not in all_paths:
            all_paths.append(relative)
    rows = [{"path": relative, "sha256": _regular_sha256(root, relative)} for relative in all_paths]
    _require(next((item["sha256"] for item in rows if item["path"] == WORKORDER_RELATIVE), None) == WORKORDER_SHA256,
             "Source Execution workorder SHA drift")
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_closure_v1",
        "stage0_closure_sha256": STAGE0_CLOSURE_SHA256,
        "source_audit_closure_sha256": SOURCE_AUDIT_CLOSURE_SHA256,
        "paths": rows,
        "closure_sha256": sha256_bytes(_json_bytes(rows)),
    }


@dataclass(frozen=True)
class SourceExecutionIdentity:
    spec: SourceExecutionSpec
    closure: Mapping[str, object]
    strict_train_roster: tuple[str, ...]
    fixed_assets: Mapping[str, Mapping[str, object]]
    normalizers: SourceNormalizers
    selected_device: Mapping[str, object]

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceExecutionSpec), "source execution identity spec drift")
        roster = tuple(self.strict_train_roster)
        _require(len(roster) == STRICT_SOURCE_COUNT and len(set(roster)) == len(roster),
                 "source execution identity strict roster drift")
        _require(isinstance(self.closure, Mapping) and _sha(self.closure.get("closure_sha256"), "identity closure")
                 == self.closure.get("closure_sha256"), "source execution identity closure drift")
        _require(isinstance(self.normalizers, SourceNormalizers), "source execution normalizer type drift")
        _require(validate_compatible_device_profile(self.selected_device) == dict(self.selected_device),
                 "source execution selected device drift")
        expected_labels = tuple(asset.label for asset in FIXED_ASSETS)
        _require(tuple(self.fixed_assets) == expected_labels, "source execution fixed-asset topology drift")
        for asset in FIXED_ASSETS:
            row = self.fixed_assets[asset.label]
            _require(isinstance(row, Mapping) and dict(row) == asset.payload(),
                     f"source execution fixed-asset identity drift: {asset.label}")
        object.__setattr__(self, "strict_train_roster", roster)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_identity_v1",
            "cell": CELL,
            "run_spec": self.spec.payload(),
            "closure": dict(self.closure),
            "strict_train_roster": list(self.strict_train_roster),
            "strict_train_roster_sha256": roster_sha256(self.strict_train_roster),
            "fixed_assets": {label: dict(value) for label, value in self.fixed_assets.items()},
            "normalizers": self.normalizers.payload(),
            "selected_device": dict(self.selected_device),
            "source_only": True,
            "within_external_formal_target_forbidden": True,
            "target_optimizer_backward_update": 0,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


def build_identity(
    root: Path,
    *,
    spec: SourceExecutionSpec,
    selected_device: Mapping[str, object],
    fixed_assets: Mapping[str, BoundAsset] | None = None,
) -> SourceExecutionIdentity:
    """Execution-only descriptor preflight; it never resolves source NWBs."""
    _require(isinstance(spec, SourceExecutionSpec), "identity requires a typed source-execution spec")
    assets = dict(descriptor_read_fixed_assets(root) if fixed_assets is None else fixed_assets)
    _require(tuple(assets) == tuple(asset.label for asset in FIXED_ASSETS), "fixed-asset preflight topology drift")
    manifest = assets["strict_manifest"]
    _require(isinstance(manifest, BoundAsset) and manifest.asset == FIXED_ASSETS[0], "strict manifest binding drift")
    roster = parse_strict_train_roster(manifest.body)
    return SourceExecutionIdentity(
        spec=spec,
        closure=execution_closure_payload(root),
        strict_train_roster=roster,
        fixed_assets={label: bound.asset.payload() for label, bound in assets.items()},
        normalizers=SEALED_NORMALIZERS,
        selected_device=validate_compatible_device_profile(selected_device),
    )


def validate_identity_current(root: Path, identity: SourceExecutionIdentity) -> None:
    _require(isinstance(identity, SourceExecutionIdentity), "execution identity must be typed")
    _require(identity.closure == execution_closure_payload(root), "source execution implementation closure drift")
    _require(identity.normalizers == SEALED_NORMALIZERS, "source execution normalizer authority drift")
    _require(validate_compatible_device_profile(identity.selected_device) == dict(identity.selected_device),
             "source execution current device profile drift")


class _RootReviewSeal:
    pass


_ROOT_REVIEW_SEAL = _RootReviewSeal()


@dataclass(frozen=True)
class StrictSourceDataRootCapability:
    """Opaque root-reviewed authority for the direct strict-27 source tree.

    The capability deliberately carries only a canonical *directory* and the
    manifest roster digest.  It does not lstat, resolve, enumerate, or open a
    source pathname when it is minted.  The physical provider performs those
    descriptor checks only after the durable attempt.  This lets the same
    implementation bind the smoke's one permitted source file without
    constructing the other 26 paths.
    """

    canonical_root: str
    strict_train_roster_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        root = Path(self.canonical_root)
        _require(root.is_absolute() and str(root) not in {"/", ""}
                 and ".." not in root.parts,
                 "strict source-data root must be a direct absolute directory literal")
        _sha(self.strict_train_roster_sha256, "strict source-data roster SHA")
        _require(self._seal is _ROOT_REVIEW_SEAL,
                 "strict source-data capability is not root-reviewed")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_strict_source_data_root_v1",
            "canonical_root": self.canonical_root,
            "strict_train_roster_sha256": self.strict_train_roster_sha256,
            "direct_child_filename_rule": "{session_id}_behavior+ecephys.nwb",
            "no_symlink_bind_copy_or_cache": True,
        }

    def checked_relative_name(self, session_id: str, *, roster: Sequence[str]) -> str:
        """Derive one direct child only after the caller selected a train row."""
        values = tuple(roster)
        _require(
            roster_sha256(values) == self.strict_train_roster_sha256
            and session_id in values,
            "source-data capability refuses a non-strict-train session",
        )
        _require(isinstance(session_id, str) and "/" not in session_id and "\\" not in session_id,
                 "source-data session name is unsafe")
        return f"{session_id}_behavior+ecephys.nwb"


@dataclass(frozen=True)
class SourceExecutionCapability:
    """Opaque in-process capability; CLI flags cannot manufacture this object."""

    identity_sha256: str
    _seal: object = field(repr=False, compare=False)
    source_data_root: StrictSourceDataRootCapability | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _sha(self.identity_sha256, "execution capability identity SHA")
        _require(self._seal is _ROOT_REVIEW_SEAL, "source execution capability is not root-reviewed")
        _require(self.source_data_root is None or isinstance(self.source_data_root, StrictSourceDataRootCapability),
                 "source execution capability source-data authority type drift")


def _issue_root_reviewed_source_data_capability(
    *,
    canonical_root: Path,
    strict_train_roster: Sequence[str],
    seal: object,
) -> StrictSourceDataRootCapability:
    """Root-only minting hook; does not touch the source directory."""
    _require(seal is _ROOT_REVIEW_SEAL, "only the root reviewer may issue source-data authority")
    return StrictSourceDataRootCapability(
        canonical_root=str(Path(canonical_root)),
        strict_train_roster_sha256=roster_sha256(tuple(strict_train_roster)),
        _seal=seal,
    )


def _issue_root_reviewed_capability(
    identity: SourceExecutionIdentity,
    *,
    seal: object,
    source_data_root: StrictSourceDataRootCapability | None = None,
) -> SourceExecutionCapability:
    _require(seal is _ROOT_REVIEW_SEAL, "only the in-process root reviewer may issue execution capability")
    if source_data_root is not None:
        _require(source_data_root.strict_train_roster_sha256 == roster_sha256(identity.strict_train_roster),
                 "root-reviewed source-data capability roster drift")
    return SourceExecutionCapability(identity.sha256, seal, source_data_root)


def require_execution_capability(capability: object, identity: SourceExecutionIdentity) -> SourceExecutionCapability:
    _require(isinstance(capability, SourceExecutionCapability) and capability._seal is _ROOT_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "source execution requires an exact in-process root-reviewed capability")
    return capability


@dataclass
class RuntimeFlags:
    stage: str = "preflight"
    source_resolved: bool = False
    source_opened: bool = False
    checkpoint_opened: bool = False
    cuda_initialized: bool = False
    model_forward_calls: int = 0
    backward_calls: int = 0
    optimizer_steps: int = 0
    parameter_updates: int = 0
    normalizer_refit: bool = False
    within_opened: bool = False
    external_opened: bool = False
    formal_opened: bool = False
    target_opened: bool = False
    oom_retry_attempted: bool = False

    def payload(self) -> dict[str, object]:
        _require(self.backward_calls == self.optimizer_steps == self.parameter_updates == 0,
                 "CDM-D source execution cannot train or update parameters")
        _require(not self.normalizer_refit and not self.within_opened and not self.external_opened
                 and not self.formal_opened and not self.target_opened and not self.oom_retry_attempted,
                 "source execution crossed a forbidden side-effect boundary")
        return {
            "stage": self.stage,
            "source_resolved": self.source_resolved,
            "source_opened": self.source_opened,
            "checkpoint_opened": self.checkpoint_opened,
            "cuda_initialized": self.cuda_initialized,
            "model_forward_calls": self.model_forward_calls,
            "backward_calls": self.backward_calls,
            "optimizer_steps": self.optimizer_steps,
            "parameter_updates": self.parameter_updates,
            "normalizer_refit": self.normalizer_refit,
            "within_opened": self.within_opened,
            "external_opened": self.external_opened,
            "formal_opened": self.formal_opened,
            "target_opened": self.target_opened,
            "oom_retry_attempted": self.oom_retry_attempted,
        }


class SourceExecutionBackend(Protocol):
    """Deferred physical surface; implementations may touch source only after attempt."""

    def preflight(self, *, root: Path, identity: SourceExecutionIdentity, flags: RuntimeFlags) -> Mapping[str, object]: ...
    def prepare(self, *, root: Path, identity: SourceExecutionIdentity, flags: RuntimeFlags) -> Any: ...
    def source_authority(self, runtime: Any, *, identity: SourceExecutionIdentity, flags: RuntimeFlags) -> Mapping[str, object]: ...
    def run_smoke(self, runtime: Any, *, identity: SourceExecutionIdentity, flags: RuntimeFlags) -> Mapping[str, object]: ...
    def run_budget(self, runtime: Any, *, budget: int, identity: SourceExecutionIdentity,
                   flags: RuntimeFlags) -> Sequence[Mapping[str, object]]: ...
    def resources(self, runtime: Any, *, flags: RuntimeFlags) -> Mapping[str, object]: ...
    def revalidate(self, *, root: Path, identity: SourceExecutionIdentity, flags: RuntimeFlags) -> None: ...
    def close(self, runtime: Any | None) -> None: ...


@dataclass
class ArtifactRoot:
    """One fresh result directory held through pair publication and reload."""

    directory: Path
    fd: int
    identity: tuple[int, int]
    allowed_body_names: frozenset[str]
    published: set[str] = field(default_factory=set)

    def _revalidate(self) -> None:
        named = os.lstat(self.directory)
        held = os.fstat(self.fd)
        _require(stat.S_ISDIR(named.st_mode) and not stat.S_ISLNK(named.st_mode)
                 and (int(named.st_dev), int(named.st_ino)) == self.identity
                 and (int(held.st_dev), int(held.st_ino)) == self.identity,
                 "source execution result root identity drift")

    def _write_leaf(self, name: str, body: bytes) -> None:
        _require(Path(name).name == name and name not in {"", ".", ".."}, "artifact leaf name drift")
        try:
            leaf_fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        except OSError as error:
            raise SourceExecutionError("artifact leaf collision or unsafe publication target") from error
        try:
            total = 0
            while total < len(body):
                total += os.write(leaf_fd, body[total:])
            os.fchmod(leaf_fd, 0o444)
            os.fsync(leaf_fd)
        finally:
            os.close(leaf_fd)

    def _unlink_owned(self, name: str) -> None:
        try:
            os.unlink(name, dir_fd=self.fd)
        except FileNotFoundError:
            return

    def publish_json_pair(self, name: str, payload: Mapping[str, object]) -> str:
        self._revalidate()
        _require(name in self.allowed_body_names and name not in self.published,
                 "artifact body topology/collision drift")
        _require(isinstance(payload, Mapping), "artifact payload must be a mapping")
        body = _json_bytes(dict(payload))
        digest = sha256_bytes(body)
        sidecar = f"{digest}  {name}\n".encode("ascii")
        body_done = sidecar_done = False
        try:
            self._write_leaf(name, body)
            body_done = True
            self._write_leaf(f"{name}.sha256", sidecar)
            sidecar_done = True
            os.fsync(self.fd)
            observed = self.read_json_pair(name, expected_sha256=digest)
            _require(observed == dict(payload), "artifact pair reload payload drift")
            self.published.add(name)
            return digest
        except BaseException:
            # A pair is never left half-published by this route.  These leaves
            # were freshly O_EXCL-created under the held root descriptor.
            if sidecar_done:
                self._unlink_owned(f"{name}.sha256")
            if body_done:
                self._unlink_owned(name)
            os.fsync(self.fd)
            raise

    def read_json_pair(self, name: str, *, expected_sha256: str) -> dict[str, object]:
        self._revalidate()
        expected = _sha(expected_sha256, "artifact pair expected SHA")
        body = _read_held_leaf(self.fd, name, mode=0o444)
        _require(sha256_bytes(body) == expected, "artifact pair body SHA drift")
        _require(_read_held_leaf(self.fd, f"{name}.sha256", mode=0o444)
                 == f"{expected}  {name}\n".encode("ascii"), "artifact pair sidecar drift")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            raise SourceExecutionError("artifact pair body is not JSON") from error
        _require(isinstance(payload, Mapping), "artifact pair JSON root drift")
        return dict(payload)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _allowed_body_names(spec: SourceExecutionSpec, roster: Sequence[str]) -> frozenset[str]:
    base = {"attempt.json", "launch.json", "source_authority.json", "terminal.json", "failure.json"}
    if spec.kind == "source_smoke":
        base.add("smoke.json")
    else:
        for budget in FAIL_FAST_BUDGET_ORDER:
            base.add(f"budget_m{budget}_aggregate.json")
            for session in roster:
                base.add(f"budget_m{budget}__{session}.json")
    return frozenset(base)


def reserve_artifact_root(root: Path, *, spec: SourceExecutionSpec, roster: Sequence[str]) -> ArtifactRoot:
    """Reserve a fresh named root only after source-free preflight passes."""
    relative = _safe_relative(spec.root_relative)
    parent = Path(root).absolute() / relative.parent
    try:
        info = os.lstat(parent)
        _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                 "artifact parent is not a direct non-symlink directory")
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise SourceExecutionError("artifact parent cannot be opened") from error
    try:
        try:
            os.mkdir(relative.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise SourceExecutionError("prospective source-execution root already exists") from error
        created = Path(parent) / relative.name
        named = os.lstat(created)
        _require(stat.S_ISDIR(named.st_mode) and not stat.S_ISLNK(named.st_mode), "created artifact root drift")
        fd = os.open(relative.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        held = os.fstat(fd)
        identity = (int(named.st_dev), int(named.st_ino))
        _require((int(held.st_dev), int(held.st_ino)) == identity, "created artifact root descriptor drift")
        os.fsync(parent_fd)
        return ArtifactRoot(created, fd, identity, _allowed_body_names(spec, roster))
    finally:
        os.close(parent_fd)


def _attempt_payload(identity: SourceExecutionIdentity) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_attempt_v1",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "source_only": True,
        "source_resolved_or_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
        "within_external_formal_target_forbidden": True,
    }


def _launch_payload(identity: SourceExecutionIdentity, attempt_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_launch_v1",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "launch attempt SHA"),
        "preflight": dict(preflight),
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }


def _validate_source_authority(value: Mapping[str, object], identity: SourceExecutionIdentity, flags: RuntimeFlags) -> None:
    _require(value.get("schema") == "causal_dual_memory_cell_d_source_execution_authority_v1"
             and value.get("cell") == CELL and value.get("identity_sha256") == identity.sha256
             and value.get("strict_train_roster") == list(identity.strict_train_roster)
             and value.get("strict_train_roster_sha256") == roster_sha256(identity.strict_train_roster)
             and value.get("normalizers") == identity.normalizers.payload()
             and value.get("source_only") is True,
             "source authority schema/identity/normalizer drift")
    access = value.get("access")
    _require(isinstance(access, Mapping) and access.get("source_opened") is True
             and access.get("within_opened") is False and access.get("external_opened") is False
             and access.get("formal_opened") is False and access.get("target_opened") is False
             and access.get("optimizer_steps") == 0 and access.get("backward_calls") == 0
             and access.get("parameter_updates") == 0 and access.get("normalizer_refit") is False,
             "source authority access boundary drift")
    validate_runtime_attestation(identity.selected_device, value.get("runtime_environment"))
    _require(value.get("fixed_assets") == {asset.label: asset.payload() for asset in FIXED_ASSETS},
             "source authority fixed-asset binding drift")
    validate_sealed_cell_d_swa_load_proof(value.get("sealed_swa_load_proof"))
    sessions = value.get("sessions")
    # Smoke must bind the full immutable strict-27 manifest without opening
    # the other 26 source files.  The full source gate is the first surface
    # permitted to materialize every source session.  Keeping these two sets
    # separate makes the smoke authority honest rather than fabricating 26
    # unopened physical rows.
    expected_physical_sessions = (
        (SOURCE_SMOKE_SESSION,)
        if identity.spec.kind == "source_smoke"
        else tuple(identity.strict_train_roster)
    )
    _require(
        isinstance(sessions, list)
        and value.get("physical_session_count") == len(expected_physical_sessions)
        and tuple(
            row.get("session_id") if isinstance(row, Mapping) else None
            for row in sessions
        ) == expected_physical_sessions,
        "source authority physical session topology drift",
    )
    _require(
        value.get("strict_manifest_bound_without_nonphysical_resolution") is True,
        "source authority strict-manifest versus physical-session boundary drift",
    )
    for session, row in zip(expected_physical_sessions, sessions, strict=True):
        _require(isinstance(row, Mapping)
                 and row.get("session_id") == session
                 and type(row.get("total_unit_count")) is int
                 and type(row.get("valid_unit_count")) is int
                 and type(row.get("invalid_unit_count")) is int
                 and row["total_unit_count"] == row["valid_unit_count"] + row["invalid_unit_count"]
                 and row.get("invalid_assignment_is_minus_one") is True
                 and row.get("theta_authority_binds_only_validity_not_budget_groups") is True
                 and _sha(row.get("theta_valid_mask_sha256"), "source authority theta valid-mask SHA")
                 == row.get("theta_valid_mask_sha256"),
                 "source authority invalid-theta topology drift")
        expected_invalid = KNOWN_THETA_INVALID_UNIT_COUNTS.get(session, 0)
        _require(row["invalid_unit_count"] == expected_invalid,
                 "source authority known undefined-theta topology drift")
    _require(flags.source_opened and not flags.target_opened and flags.optimizer_steps == 0,
             "runtime flags/source authority drift")


def _validate_smoke(value: Mapping[str, object], identity: SourceExecutionIdentity) -> None:
    smoke = identity.spec.payload()["smoke"]
    _require(identity.spec.kind == "source_smoke" and isinstance(smoke, Mapping)
             and value.get("schema") == "causal_dual_memory_cell_d_source_execution_smoke_v1"
             and value.get("session") == smoke["session"] and value.get("budget") == smoke["budget"]
             and value.get("support_positions") == smoke["support_positions"]
             and value.get("audit_positions") == smoke["audit_positions"]
             and value.get("group_count") == GROUP_COUNT and value.get("b8_threshold_applied") is False
             and value.get("source_only") is True,
             "fixed smoke payload drift")
    _require(
        value.get("budget_initial_carrier_recipe") == "fixed_ridge_by_trial"
        and value.get("budget_initial_carrier_support_rows") == 30
        and value.get("raw_m30_t4_used_as_initializer") is False
        and value.get("initial_support_rate_domain") == INITIAL_SUPPORT_RATE_DOMAIN
        and value.get("online_update_rate_domain") == ONLINE_UPDATE_RATE_DOMAIN
        and isinstance(value.get("budget_initial_carrier_parity"), Mapping)
        and value["budget_initial_carrier_parity"].get("mode") == "fixed_ridge_by_trial",
        "fixed smoke support-only carrier/rate-domain evidence drift",
    )
    for key in (
        "initial_support_rates_sha256",
        "initial_support_exposure_seconds_sha256",
        "budget_initial_carrier_sha256",
        "budget_groups_sha256",
        "budget_group_assignment_sha256",
        "budget_group_valid_mask_sha256",
    ):
        _require(_sha(value.get(key), f"fixed smoke {key}") == value.get(key),
                 "fixed smoke initial-carrier/group digest drift")


def _validate_session_evidence(value: Mapping[str, object], *, budget: int, expected_session: str) -> None:
    _require(value.get("schema") == "causal_dual_memory_cell_d_source_execution_session_b8_v1"
             and value.get("budget") == budget and value.get("session") == expected_session
             and value.get("source_only") is True and value.get("target_optimizer_backward_update") == 0,
             "source session B8 evidence schema/identity drift")
    status = value.get("status")
    _require(status in {"COMPLETE_FIXED_POOL", "STOP_FIXED_POOL_REJECTION", "STOP_MISSING_REQUIRED_CHRONOLOGY", "STOP_NO_DEFINED_VALID_COSINES"},
             "source session B8 status drift")
    _require(type(value.get("pass")) is bool, "source session B8 pass must be exact bool")
    topology = value.get("unit_topology")
    _require(isinstance(topology, Mapping)
             and type(topology.get("total_unit_count")) is int
             and type(topology.get("valid_unit_count")) is int
             and type(topology.get("invalid_unit_count")) is int
             and topology["total_unit_count"] == topology["valid_unit_count"] + topology["invalid_unit_count"]
             and _sha(topology.get("valid_mask_sha256"), "source session valid-mask SHA") == topology.get("valid_mask_sha256"),
             "source session B8 invalid-theta topology drift")
    source_topology = value.get("source_authority_unit_topology")
    _require(isinstance(source_topology, Mapping)
             and source_topology.get("session_id") == expected_session
             and type(source_topology.get("total_unit_count")) is int
             and type(source_topology.get("valid_unit_count")) is int
             and type(source_topology.get("invalid_unit_count")) is int
             and source_topology["total_unit_count"]
             == source_topology["valid_unit_count"] + source_topology["invalid_unit_count"]
             and source_topology.get("invalid_assignment_is_minus_one") is True
             and _sha(source_topology.get("valid_mask_sha256"), "source session source valid-mask SHA")
             == source_topology.get("valid_mask_sha256")
             and _sha(source_topology.get("theta_valid_mask_sha256"), "source session theta valid-mask SHA")
             == source_topology.get("theta_valid_mask_sha256"),
             "source session source-authority topology drift")
    expected_invalid = KNOWN_THETA_INVALID_UNIT_COUNTS.get(expected_session, 0)
    _require(
        source_topology["invalid_unit_count"] == expected_invalid
        and topology["invalid_unit_count"] == expected_invalid
        and topology["total_unit_count"] == source_topology["total_unit_count"]
        and topology["valid_unit_count"] == source_topology["valid_unit_count"]
        and topology["valid_mask_sha256"] == source_topology["valid_mask_sha256"],
             "source session known undefined-theta topology drift")
    if status != "COMPLETE_FIXED_POOL":
        _require(value["pass"] is False, "non-complete fixed pool cannot pass B8")
    if status != "STOP_MISSING_REQUIRED_CHRONOLOGY":
        _require(
            value.get("budget_initial_carrier_recipe") == "fixed_ridge_by_trial"
            and value.get("budget_initial_carrier_support_rows") == budget
            and value.get("raw_m30_t4_used_as_initializer") is False
            and value.get("initial_support_rate_domain") == INITIAL_SUPPORT_RATE_DOMAIN
            and value.get("online_update_rate_domain") == ONLINE_UPDATE_RATE_DOMAIN
            and isinstance(value.get("budget_initial_carrier_parity"), Mapping)
            and value["budget_initial_carrier_parity"].get("mode") == "fixed_ridge_by_trial",
            "source session support-only fixed-ridge receipt drift",
        )
        for key in (
            "initial_support_rates_sha256",
            "initial_support_exposure_seconds_sha256",
            "budget_initial_carrier_sha256",
            "budget_groups_sha256",
            "budget_group_assignment_sha256",
            "budget_group_valid_mask_sha256",
        ):
            _require(_sha(value.get(key), f"source session {key}") == value.get(key),
                     "source session initial-carrier/group digest drift")


def _aggregate_budget(rows: Sequence[Mapping[str, object]], *, budget: int, roster: Sequence[str]) -> dict[str, object]:
    _require(len(rows) == STRICT_SOURCE_COUNT and tuple(row.get("session") for row in rows) == tuple(roster),
             "budget evidence must retain all 27 strict source sessions in manifest order")
    for session, row in zip(roster, rows, strict=True):
        _validate_session_evidence(row, budget=budget, expected_session=session)
    pass_count = sum(bool(row["pass"]) for row in rows)
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_budget_aggregate_v1",
        "budget": budget,
        "strict_source_session_count": STRICT_SOURCE_COUNT,
        "passing_session_count": pass_count,
        "breadth_min_passing_sessions": BREADTH_MIN_PASSING_SESSIONS,
        "breadth_pass": pass_count >= BREADTH_MIN_PASSING_SESSIONS,
        "session_body_sha256s": [sha256_bytes(_json_bytes(dict(row))) for row in rows],
        "source_only": True,
    }


def _validate_resources(value: Mapping[str, object]) -> None:
    required = (
        "endpoint_chunks_per_s", "trials_per_s", "wall_seconds",
        "endpoint_chunks_completed", "completed_trials", "rss_bytes",
        "current_cuda_allocated_bytes", "current_cuda_reserved_bytes",
        "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "selected_device",
    )
    _require(isinstance(value, Mapping) and all(key in value for key in required), "source execution resource schema drift")
    for key in required[:-1]:
        observed = value[key]
        _require(isinstance(observed, (int, float)) and not isinstance(observed, bool)
                 and math.isfinite(float(observed)) and observed >= 0,
                 f"source execution resource {key} drift")
    # Both terminal types are reached only after real held-group forwards.  A
    # zero rate or duration here would therefore be a decorative resource
    # receipt rather than a measurement.  The physical executor synchronizes
    # before sampling these values, so this is a genuine launch gate rather
    # than a timing estimate.
    for key in ("endpoint_chunks_per_s", "trials_per_s", "wall_seconds"):
        _require(value[key] > 0, f"source execution measured {key} must be positive after forwards")
    _require(type(value["endpoint_chunks_completed"]) is int and value["endpoint_chunks_completed"] > 0
             and type(value["completed_trials"]) is int and value["completed_trials"] > 0,
             "source execution measured counter topology drift")
    expected_endpoint_rate = value["endpoint_chunks_completed"] / value["wall_seconds"]
    expected_trial_rate = value["completed_trials"] / value["wall_seconds"]
    _require(
        math.isclose(value["endpoint_chunks_per_s"], expected_endpoint_rate, rel_tol=1.0e-12, abs_tol=1.0e-12)
        and math.isclose(value["trials_per_s"], expected_trial_rate, rel_tol=1.0e-12, abs_tol=1.0e-12),
        "source execution measured rate/counter binding drift",
    )
    _require(value["peak_cuda_allocated_bytes"] >= value["current_cuda_allocated_bytes"]
             and value["peak_cuda_reserved_bytes"] >= value["current_cuda_reserved_bytes"],
             "source execution peak/current CUDA resource drift")
    validate_compatible_device_profile(value["selected_device"])


def _terminal_payload(
    identity: SourceExecutionIdentity,
    *,
    attempt_sha256: str,
    launch_sha256: str,
    source_authority_sha256: str,
    evidence_sha256s: Mapping[str, str],
    status: str,
    resources: Mapping[str, object],
) -> dict[str, object]:
    _require(status in {"PASS_SOURCE_CONSTRUCTIBLE", "STOP_SOURCE_B8_CONSTRUCTIBILITY", "SMOKE_COMPLETED"},
             "terminal source-execution status drift")
    _validate_resources(resources)
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_terminal_v1",
        "cell": CELL,
        "status": status,
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "terminal attempt SHA"),
        "launch_sha256": _sha(launch_sha256, "terminal launch SHA"),
        "source_authority_sha256": _sha(source_authority_sha256, "terminal source authority SHA"),
        "evidence_sha256s": dict(evidence_sha256s),
        "resources": dict(resources),
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }


def _failure_payload(identity: SourceExecutionIdentity, *, attempt_sha256: str, launch_sha256: str | None,
                     source_authority_sha256: str | None, flags: RuntimeFlags, error: BaseException) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_failure_v1",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "failure attempt SHA"),
        "launch_sha256": None if launch_sha256 is None else _sha(launch_sha256, "failure launch SHA"),
        "source_authority_sha256": None if source_authority_sha256 is None else _sha(source_authority_sha256, "failure source-authority SHA"),
        "stage": flags.stage,
        "error_class": type(error).__name__,
        "error_sha256": sha256_bytes(repr(error).encode("utf-8")),
        "flags": flags.payload(),
        "terminal_published": False,
        "source_only": True,
    }


def execute_authorized(
    root: Path,
    *,
    identity: SourceExecutionIdentity,
    capability: object,
    backend: SourceExecutionBackend,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run the reviewed future route; unreachable from public CLI flags.

    This function is intentionally testable with an injected backend.  A
    physical backend must itself descriptor-revalidate fixed assets before and
    after execution, and must update ``RuntimeFlags`` honestly as source,
    checkpoint, and CUDA stages become real.
    """
    require_execution_capability(capability, identity)
    validate_identity_current(root, identity)
    validate_selected_device_environment(identity.selected_device, environ)
    flags = RuntimeFlags(stage="preflight")
    preflight = backend.preflight(root=Path(root), identity=identity, flags=flags)
    _require(isinstance(preflight, Mapping) and preflight.get("source_resolved_or_opened") is False
             and preflight.get("checkpoint_opened") is False and preflight.get("cuda_initialized") is False,
             "source-free preflight boundary drift")
    artifact = reserve_artifact_root(Path(root), spec=identity.spec, roster=identity.strict_train_roster)
    runtime: Any | None = None
    attempt_sha: str | None = None
    launch_sha: str | None = None
    authority_sha: str | None = None
    try:
        flags.stage = "attempt"
        attempt_sha = artifact.publish_json_pair("attempt.json", _attempt_payload(identity))
        flags.stage = "launch"
        launch_sha = artifact.publish_json_pair("launch.json", _launch_payload(identity, attempt_sha, preflight))
        flags.stage = "prepare"
        runtime = backend.prepare(root=Path(root), identity=identity, flags=flags)
        flags.stage = "source_authority"
        authority = dict(backend.source_authority(runtime, identity=identity, flags=flags))
        _validate_source_authority(authority, identity, flags)
        authority_sha = artifact.publish_json_pair("source_authority.json", authority)
        evidence_sha: dict[str, str] = {}
        if identity.spec.kind == "source_smoke":
            flags.stage = "smoke"
            smoke = dict(backend.run_smoke(runtime, identity=identity, flags=flags))
            _validate_smoke(smoke, identity)
            evidence_sha["smoke.json"] = artifact.publish_json_pair("smoke.json", smoke)
            final_status = "SMOKE_COMPLETED"
        else:
            final_status = "PASS_SOURCE_CONSTRUCTIBLE"
            for budget in FAIL_FAST_BUDGET_ORDER:
                flags.stage = f"budget_m{budget}"
                rows = tuple(dict(item) for item in backend.run_budget(
                    runtime, budget=budget, identity=identity, flags=flags,
                ))
                aggregate = _aggregate_budget(rows, budget=budget, roster=identity.strict_train_roster)
                for session, row in zip(identity.strict_train_roster, rows, strict=True):
                    name = f"budget_m{budget}__{session}.json"
                    evidence_sha[name] = artifact.publish_json_pair(name, row)
                aggregate_name = f"budget_m{budget}_aggregate.json"
                evidence_sha[aggregate_name] = artifact.publish_json_pair(aggregate_name, aggregate)
                if aggregate["breadth_pass"] is False:
                    final_status = "STOP_SOURCE_B8_CONSTRUCTIBILITY"
                    break
        flags.stage = "final_revalidation"
        validate_identity_current(root, identity)
        backend.revalidate(root=Path(root), identity=identity, flags=flags)
        resources = dict(backend.resources(runtime, flags=flags))
        flags.stage = "terminal"
        terminal = _terminal_payload(
            identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
            source_authority_sha256=authority_sha, evidence_sha256s=evidence_sha,
            status=final_status, resources=resources,
        )
        terminal_sha = artifact.publish_json_pair("terminal.json", terminal)
        return {"attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
                "source_authority_sha256": authority_sha, "terminal_sha256": terminal_sha,
                "status": final_status, "flags": flags.payload()}
    except BaseException as error:
        if attempt_sha is not None and "terminal.json" not in artifact.published:
            flags.stage = flags.stage if flags.stage else "unknown"
            try:
                artifact.publish_json_pair(
                    "failure.json",
                    _failure_payload(identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
                                     source_authority_sha256=authority_sha, flags=flags, error=error),
                )
            except BaseException:
                # Preserve the original error if failure publication itself is
                # impossible; a fresh root may then contain only the durable attempt.
                pass
        raise
    finally:
        try:
            backend.close(runtime)
        finally:
            artifact.close()


def execute_reviewed_physical(
    root: Path,
    *,
    identity: SourceExecutionIdentity,
    capability: object,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """The sole reviewed concrete-launch entry point.

    It is intentionally separate from the injectable lifecycle core used by
    synthetic tests.  A caller cannot supply an arbitrary parser/executor:
    the opaque root capability must carry the direct strict-source authority,
    and the route-local physical factory builds the only accepted provider and
    sealed Cell-D executor.  Importing the factory remains side-effect free;
    source paths, checkpoint tensors, and CUDA are still reached only after
    :func:`execute_authorized` has published the immutable attempt.
    """
    approved = require_execution_capability(capability, identity)
    _require(approved.source_data_root is not None,
             "reviewed physical source execution requires a root-bound source-data capability")
    _require(
        approved.source_data_root.strict_train_roster_sha256 == roster_sha256(identity.strict_train_roster),
        "reviewed physical source-data capability/identity roster drift",
    )
    from .source_execute_physical import build_reviewed_physical_backend

    backend = build_reviewed_physical_backend(
        root=Path(root),
        source_data=approved.source_data_root,
        selected_device=identity.selected_device,
    )
    return execute_authorized(
        Path(root), identity=identity, capability=approved, backend=backend, environ=environ,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Public inert plan: no project, data, Torch, CUDA, or write import is needed."""
    result: dict[str, object] = {
        "cell": CELL,
        "phase": "source_execution_v1_scaffold",
        "workorder_sha256": WORKORDER_SHA256,
        "accepted_stage0_closure_sha256": STAGE0_CLOSURE_SHA256,
        "accepted_source_audit_closure_sha256": SOURCE_AUDIT_CLOSURE_SHA256,
        "source_smoke_root_relative": SOURCE_SMOKE_ROOT_RELATIVE,
        "source_gate_root_relative": SOURCE_GATE_ROOT_RELATIVE,
        "execution_authorized": False,
        "opens_source": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_root": False,
        "scores": False,
        "launches": False,
    }
    if root is not None:
        result["closure"] = execution_closure_payload(root)
    return result
