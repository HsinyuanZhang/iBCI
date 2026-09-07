"""Dry, additive contract for a small Posterior Carrier engineering screen.

This module intentionally imports only the Python standard library.  In
particular, import and the public zero-argument CLI do not import Torch,
resolve a NWB path, read a checkpoint, create a result directory, contact the
remote host, or initialize CUDA.  The future physical implementation is
separately capability-gated and may only run after an independent review.

The route is deliberately *not* a reduced governing scorer.  It fixes three
within and three external sessions before scores, evaluates four cells per
surface, and emits a ``NON_GOVERNING_QUICK_SCREEN`` receipt with no decision
rule.  Its only legitimate purpose is a fast engineering signal that informs
whether the completed full matched scorer deserves the next remote window.
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


CELL = "POSTERIOR_CARRIER_BUDGETMIX_D_SEED42"
PHASE = "POSTERIOR_CARRIER_QUICK_SCREEN_V3"
SCHEMA = "posterior_carrier_quick_screen_v3"
CLASSIFICATION = "NON_GOVERNING_QUICK_SCREEN"

WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_POSTERIOR_CARRIER_QUICK_SCREEN_20260822.md"
WORKORDER_SHA256 = "4839140dfc773654faf5420e25c003c55ecad7cb4d74830c37b7f73e2bf338de"

PACKAGE_RELATIVE = "tfpd_exploration/src/posterior_carrier_quick_screen_v1"
SCRIPT_RELATIVE = "tfpd_exploration/scripts/run_posterior_carrier_quick_screen.py"
TEST_RELATIVE = "tfpd_exploration/tests/test_posterior_carrier_quick_screen_v1.py"

WITHIN = "within"
EXTERNAL = "external"
SURFACES = (WITHIN, EXTERNAL)
WITHIN_INDICES = (0, 2, 5)
EXTERNAL_INDICES = (0, 7, 14)
FULL_WITHIN_COUNT = 6
FULL_EXTERNAL_COUNT = 15

SEALED_POINT_MODE = "sealed_cell_d_ols_point"
POSTERIOR_MODE = "posterior_full_swa_aligned"
BUDGETS = (30, 4)

BASE_SCORE_AUTHORITY_ROOT = "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_score_authority_v1"
BASE_OFFICIAL_PREFLIGHT_RELATIVE = f"{BASE_SCORE_AUTHORITY_ROOT}/official_preflight.json"
BASE_ROOT_AUTHORIZATION_RELATIVE = f"{BASE_SCORE_AUTHORITY_ROOT}/root_authorization.json"
BASE_OFFICIAL_PREFLIGHT_SHA256 = "919d3db56f71a76597e1ffdf8e157e156e6ec37e701604f9d53801cdcafa255d"
BASE_ROOT_AUTHORIZATION_SHA256 = "f5c54a98e82cb669d9af1077564f6ffe4580d7502e34d76e28db3bfb63f4dc76"

FULL_MIRROR_ROOT = "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_full_train_import_mirror_v1"
SEALED_CELL_D_ROOT = "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout"
SEALED_CELL_D_TERMINAL_RELATIVE = f"{SEALED_CELL_D_ROOT}/terminal_receipt.json"
SEALED_CELL_D_SWA_RELATIVE = f"{SEALED_CELL_D_ROOT}/swa_final4.pt"
SEALED_CELL_D_BASELINE_RELATIVE = "tfpd_exploration/results/sparsification_score_v1/sparsification_score_receipt.json"
SEALED_CELL_D_TERMINAL_SHA256 = "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442"
SEALED_CELL_D_SWA_SHA256 = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
SEALED_CELL_D_BASELINE_SHA256 = "583b899bb9e6b132a50b23552d9734b0ccb9b12ecc5c43c27c96ba49bd71980f"

POSTERIOR_FULL_TERMINAL_SHA256 = "8cbbade4a77d9f0434712ccf292de9eac8d79fd6aa53624ab3f13fa7c15c3f41"
POSTERIOR_FULL_SWA_SHA256 = "def27d8edffc0c6ed17dee80292dd1c1b628ba1d454dc214278caee814cf40fa"
POSTERIOR_FULL_SWA_STATE_SHA256 = "cd3df34ce009636837207ef4ad3b7b213e70b4b7a6e4e73ec273cc8512f206e5"
POSTERIOR_FULL_SOURCE_AUTHORITY_SHA256 = "26d85c5f92f353752786f75dff6c959e3475e8a9eb0920095691a14ff8c95241"
POSTERIOR_FULL_CLOSURE_SHA256 = "238acf8b6b68e5b25009baf3081a679c07b9b7aad75d87bbdf851cbd48b75a43"
POSTERIOR_FULL_IMPORT_PROVENANCE_SHA256 = "bc7ad2515e8fb3f91ae148f3198d629fafbfed58b5edbe8f5a1db7d2b3352375"

REMOTE_STAGE_NAME = "posterior_carrier_quick_screen_stage_v3"
REMOTE_STAGE_ROOT = f"/home/xinyuan/Work_host/{REMOTE_STAGE_NAME}"
REMOTE_SCORE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_quick_screen_v3"
REMOTE_EVALUATION_ROOT_RELATIVE = "evaluation_assets"

# V1/V2 are immutable failed-predecessor evidence, never V3 staging or output
# locations.  The physical V3 route opens both exact remote receipt directories
# through held O_NOFOLLOW descriptors before V3 capability acceptance/root
# reservation.
V1_REMOTE_STAGE_ROOT = "/home/xinyuan/Work_host/posterior_carrier_quick_screen_stage_v1"
V1_REMOTE_SCORE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_quick_screen_v1"
V1_REMOTE_RESULT_DIRECTORY = f"{V1_REMOTE_STAGE_ROOT}/{V1_REMOTE_SCORE_ROOT_RELATIVE}"
V1_ATTEMPT_SHA256 = "7620c853d2b4ff5c2aae359859956bc94bca9c48740294103c453674bd0a8e23"
V1_FAILURE_SHA256 = "275111598568502cabc7c68350644b2599108c1930cdb622057c27dd7a5cf965"
V1_FAILURE_ERROR_SHA256 = "8ef26ffbfedefaedf122e82ff78b30b946b2c77356f9c1b3ba046629886c87a2"

V2_REMOTE_STAGE_ROOT = "/home/xinyuan/Work_host/posterior_carrier_quick_screen_stage_v2"
V2_REMOTE_SCORE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_quick_screen_v2"
V2_REMOTE_RESULT_DIRECTORY = f"{V2_REMOTE_STAGE_ROOT}/{V2_REMOTE_SCORE_ROOT_RELATIVE}"
V2_ATTEMPT_SHA256 = "11c586cc6bf837cf21c4972acd797018ebb40cf48b2be12c0619490f45147383"
V2_FAILURE_SHA256 = "d06a6f9afc5ba0015010a9cedf1e96ab872944c6851334a6b67308c851a348be"
V2_FAILURE_ERROR_SHA256 = "85b311481adb2e5455238d651192d4ab5c9f2cc55baf3d7a374fa4f3306e9456"

# The approved engineering host has a Torch-visible 5070 Ti but an NVML
# driver/library mismatch.  The route must therefore attest the exact Torch
# runtime facts and record the NVML fields as honestly unavailable; it must
# never manufacture a UUID/BDF/nominal-memory claim by falling back to an
# unrelated local authority or calling ``nvidia-smi``.
REMOTE_ENGINEERING_DEVICE_SCHEMA = "posterior_carrier_quick_screen_remote_torch_only_device_v3"
REMOTE_ENGINEERING_DEVICE_AUTHORITY = {
    "schema": REMOTE_ENGINEERING_DEVICE_SCHEMA,
    "cuda_visible_devices": "0",
    "cuda_device_order": "PCI_BUS_ID",
    "visible_device_count": 1,
    "logical_device": "cuda:0",
    "uuid": None,
    "bdf": None,
    "name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
    "nvidia_smi_memory_total_mib": None,
    "nvml_status": "UNAVAILABLE_DRIVER_LIBRARY_MISMATCH",
    "nvidia_smi_called": False,
    "torch_total_memory_bytes": 12_346_195_968,
    "torch_version": "2.13.0+cu130",
    "torch_cuda_version": "13.0",
    "cudnn_version": 92_000,
    "torchmetrics_version": "1.9.0",
    "compute_capability": [12, 0],
    "engineering_only": True,
    "current_local_authority": False,
    "route_role": "remote_5070ti_non_governing_quick_screen",
    "variance_weighted_r2_parity_gate": "PASS__SYNTHETIC_MANUAL_REFERENCE",
}

METRIC = {
    "estimator": "tfpd_lane.matched_scorer.session_r2",
    "torchmetrics": "torchmetrics.regression.R2Score(multioutput='variance_weighted')",
    "engineering_implementation": "posterior_carrier_quick_screen_v3.manual_variance_weighted_two_coordinate_r2",
    "engineering_parity": "manual_vs_torchmetrics_1_9_pre_input_and_cpu_1_5_helper_fixture_checked",
    "query": "last_bin_only_of_each_valid_50_bin_window",
    "window_bins": 50,
    "behavior_coordinates": 2,
    "equal_weight_per_session": True,
}

# V3 is authorized only when the stored full-SWA state is reproduced by the
# *producer's* frozen FP32 accumulation rule.  This is separate from, and does
# not change, the sealed Cell-D FP64 SWA validator in the inherited scorer.
POSTERIOR_FINAL4_COMPATIBILITY = {
    "schema": "posterior_carrier_quick_screen_final4_compatibility_v3",
    "posterior_swa_state_sha256": POSTERIOR_FULL_SWA_STATE_SHA256,
    "checkpoint_epochs": [44, 45, 46, 47],
    "final4_exact_recompute": "PASS__AUTHORITATIVE_FULL_TRAIN_FP32_ACCUMULATION",
    "authoritative_accumulation": "torch.zeros_like(first); ordered_add_(44,45,46,47); div(4)",
    "stored_state_bitwise_equal_to_authoritative_fp32": True,
    "fresh_strict_load": True,
    "strict_loaded_state_sha256": POSTERIOR_FULL_SWA_STATE_SHA256,
    "frozen_base_fp64_recompute": "INCOMPATIBLE__AUTHORITATIVE_FP32_CONSTRUCTION",
}

BOUNDARIES = {
    "classification": CLASSIFICATION,
    "target_optimizer_steps": 0,
    "target_backward_calls": 0,
    "target_update_calls": 0,
    "normalizer_refit": False,
    "target_sampling": False,
    "checkpoint_selection": False,
    "h1_opened": False,
    "formal_opened": False,
    "zero_control_scored": False,
    "wrong_pair_control_scored": False,
    "m10_scored": False,
    "full_window_headline": False,
}

# This is a closure for *this* route, including the exact frozen physical
# substrate it reuses only behind an in-process capability.  It is explicit:
# no package scan or glob can add a hidden run-time file.
IMPLEMENTATION_CLOSURE = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/quick_screen.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/physical.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/remote_stage.py",
    SCRIPT_RELATIVE,
    TEST_RELATIVE,
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
    "tfpd_exploration/src/posterior_carrier_v1/full_result_import.py",
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
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "tfpd_exploration/docs/WORKORDER_POSTERIOR_CARRIER_MATCHED_SCORE_20260822.md",
    "tfpd_exploration/scripts/run_posterior_carrier_matched_score.py",
    "tfpd_exploration/scripts/run_posterior_carrier_full_result_import.py",
    "tfpd_exploration/tests/test_posterior_carrier_matched_score.py",
    "tfpd_exploration/tests/test_posterior_carrier_full_result_import.py",
    "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v1/c1_train_val_33_manifest.json",
    "sua_exploration/results/dandi_000688_subm_co_schema_preflight_v2/receipt.json",
    "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json",
)


class QuickScreenError(RuntimeError):
    """Fail closed before a quick-screen score can become durable."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise QuickScreenError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha(value: object, label: str = "SHA-256") -> str:
    if not isinstance(value, str) or len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise QuickScreenError(f"{label} must be an exact lowercase SHA-256")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise QuickScreenError(f"{label} must be finite")
    return float(value)


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise QuickScreenError(f"{label} must be one safe nonempty path component")
    return value


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise QuickScreenError(f"{label} must be a safe nonempty relative path")
    return value


def _copy_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise QuickScreenError(f"{label} must be a mapping")
    try:
        copied = json.loads(_json(value))
    except (TypeError, ValueError) as error:
        raise QuickScreenError(f"{label} must be JSON serializable") from error
    if not isinstance(copied, dict):
        raise QuickScreenError(f"{label} must decode to an object")
    return copied


def remote_engineering_device_payload() -> dict[str, object]:
    """Return the sole honest remote engineering device receipt schema."""
    return _copy_mapping(REMOTE_ENGINEERING_DEVICE_AUTHORITY, "remote engineering device authority")


def validate_remote_engineering_device_payload(value: object) -> dict[str, object]:
    """Reject a local/NVML fallback or any drift in the Torch-only authority."""
    payload = _copy_mapping(value, "quick engineering device attestation")
    if payload != REMOTE_ENGINEERING_DEVICE_AUTHORITY:
        raise QuickScreenError("quick engineering Torch-only device authority drift")
    return payload


def posterior_final4_compatibility_payload() -> dict[str, object]:
    """Return the one accepted V3 Posterior final-four reconstruction fact."""
    return _copy_mapping(POSTERIOR_FINAL4_COMPATIBILITY, "posterior final-four compatibility")


def validate_posterior_final4_compatibility_payload(value: object) -> dict[str, object]:
    """Reject a bypass, FP64 substitute, state-digest drift, or schema drift."""
    payload = _copy_mapping(value, "posterior final-four compatibility")
    if payload != POSTERIOR_FINAL4_COMPATIBILITY:
        raise QuickScreenError("posterior final-four compatibility authority drift")
    return payload


def upstream_formal_preflight_device_payload(value: object) -> dict[str, object]:
    """Bind—not reinterpret—the upstream formal device authority in a score."""
    device = _copy_mapping(value, "upstream formal preflight device contract")
    expected_device_keys = {
        "cuda_visible_devices", "cuda_device_order", "logical_device", "uuid", "bdf", "name",
        "nvidia_smi_memory_total_mib", "torch_total_memory_bytes", "torch_version", "torch_cuda_version", "cudnn_version",
    }
    if set(device) != expected_device_keys:
        raise QuickScreenError("upstream formal preflight device schema drift")
    if (
        device.get("cuda_visible_devices") != "0" or device.get("cuda_device_order") != "PCI_BUS_ID"
        or device.get("logical_device") != "cuda:0" or not isinstance(device.get("uuid"), str)
        or not device["uuid"].startswith("GPU-") or not isinstance(device.get("bdf"), str)
        or len(device["bdf"].split(":")) != 3 or not isinstance(device.get("name"), str) or not device["name"]
        or type(device.get("nvidia_smi_memory_total_mib")) is not int or device["nvidia_smi_memory_total_mib"] <= 0
        or type(device.get("torch_total_memory_bytes")) is not int or device["torch_total_memory_bytes"] <= 0
        or not isinstance(device.get("torch_version"), str) or not device["torch_version"]
        or not isinstance(device.get("torch_cuda_version"), str) or not device["torch_cuda_version"]
        or type(device.get("cudnn_version")) is not int or device["cudnn_version"] <= 0
    ):
        raise QuickScreenError("upstream formal preflight device literal/type drift")
    return {
        "official_preflight_sha256": BASE_OFFICIAL_PREFLIGHT_SHA256,
        "role": "upstream_formal_preflight_device_authority_not_current_engineering_runtime",
        "device_contract": device,
    }


def validate_upstream_formal_preflight_device_payload(value: object) -> dict[str, object]:
    payload = _copy_mapping(value, "quick upstream formal preflight device")
    expected = {"official_preflight_sha256", "role", "device_contract"}
    if set(payload) != expected or payload.get("official_preflight_sha256") != BASE_OFFICIAL_PREFLIGHT_SHA256:
        raise QuickScreenError("quick upstream formal preflight device binding drift")
    if payload.get("role") != "upstream_formal_preflight_device_authority_not_current_engineering_runtime":
        raise QuickScreenError("quick upstream formal preflight device role drift")
    rebuilt = upstream_formal_preflight_device_payload(payload.get("device_contract"))
    if rebuilt != payload:
        raise QuickScreenError("quick upstream formal preflight device exact reconstruction drift")
    return rebuilt


@dataclass(frozen=True)
class ImplementationClosure:
    """Exact non-glob source/metadata closure for the quick-screen stage."""

    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        if not isinstance(self.sha256_by_path, Mapping) or set(self.sha256_by_path) != set(IMPLEMENTATION_CLOSURE):
            raise QuickScreenError("quick-screen implementation closure topology drift")
        hashes = {path: _sha(self.sha256_by_path[path], f"closure SHA {path}") for path in IMPLEMENTATION_CLOSURE}
        if hashes[WORKORDER_RELATIVE] != WORKORDER_SHA256:
            raise QuickScreenError("quick-screen workorder SHA drift")
        body = {"paths": list(IMPLEMENTATION_CLOSURE), "sha256_by_path": hashes}
        return {**body, "closure_sha256": _digest(_json(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    """Read only named regular leaves and reject alias/swap drift."""
    base = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_CLOSURE:
        path = base / relative
        try:
            before = os.lstat(path)
        except OSError as error:
            raise QuickScreenError(f"closure path missing: {relative}") from error
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise QuickScreenError(f"closure path must be a regular non-symlink: {relative}")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_ISLNK(opened.st_mode)
                or (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size)
            ):
                raise QuickScreenError(f"closure leaf descriptor identity drift: {relative}")
            chunks: list[bytes] = []
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                chunks.append(block)
            hashes[relative] = _digest(b"".join(chunks))
        finally:
            os.close(fd)
        after = os.lstat(path)
        if (after.st_dev, after.st_ino, after.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise QuickScreenError(f"closure leaf changed during read: {relative}")
    return ImplementationClosure(sha256_by_path=hashes)


@dataclass(frozen=True)
class SelectedAsset:
    """One target evaluation byte stream selected before any score exists."""

    surface: str
    full_roster_index: int
    asset_id: str
    session: str
    frozen_path: str
    byte_count: int
    sha256: str

    def payload(self) -> dict[str, object]:
        if self.surface not in SURFACES:
            raise QuickScreenError("selected asset surface drift")
        indices = WITHIN_INDICES if self.surface == WITHIN else EXTERNAL_INDICES
        if self.full_roster_index not in indices:
            raise QuickScreenError("selected asset roster index drift")
        if not isinstance(self.asset_id, str) or not self.asset_id:
            raise QuickScreenError("selected asset id drift")
        if not isinstance(self.session, str) or not self.session:
            raise QuickScreenError("selected asset session drift")
        path = _safe_relative(self.frozen_path, "selected asset frozen path")
        if Path(path).name != f"{self.session}_behavior+ecephys.nwb":
            raise QuickScreenError("selected asset filename/session drift")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise QuickScreenError("selected asset byte count drift")
        return {
            "surface": self.surface,
            "full_roster_index": self.full_roster_index,
            "asset_id": self.asset_id,
            "session": self.session,
            "frozen_path": path,
            "bytes": self.byte_count,
            "sha256": _sha(self.sha256, "selected asset SHA"),
        }


# These literal records make selection tamper-evident even if a forged metadata
# authority preserves a session label.  They are independently re-derived from
# the C1 manifest or v2 UUID ledger before staging/preflight publication.
_SELECTED_LITERALS: dict[str, tuple[tuple[int, str, str, str, int, str], ...]] = {
    WITHIN: (
        (0, "c1_paired_view:sub-C_ses-CO-20151103", "sub-C_ses-CO-20151103",
         "sub-C/sub-C_ses-CO-20151103_behavior+ecephys.nwb", 62_145_872,
         "7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7"),
        (2, "c1_paired_view:sub-C_ses-CO-20151106", "sub-C_ses-CO-20151106",
         "sub-C/sub-C_ses-CO-20151106_behavior+ecephys.nwb", 119_065_400,
         "7020f6430a66f857da13254b6a367bff6da81175ed8ad92eb272fb7deb18aeb6"),
        (5, "c1_paired_view:sub-C_ses-CO-20151112", "sub-C_ses-CO-20151112",
         "sub-C/sub-C_ses-CO-20151112_behavior+ecephys.nwb", 45_326_504,
         "1162d61afa85bcd33bc022dbeef2f34a7a421cfbc3d5b1f06864ca81bc6d74f2"),
    ),
    EXTERNAL: (
        (0, "a72cae17-6e18-4c36-bdaa-f0f31d557888", "sub-M_ses-CO-20140307",
         "sub-M/sub-M_ses-CO-20140307_behavior+ecephys.nwb", 74_037_256,
         "2f109d6daed0ad2c3dba12742d62b7be1f385117c1b0b018205093a873c29927"),
        (7, "43a8aa34-557e-44b9-a533-8456b7833f9b", "sub-M_ses-CO-20150611",
         "sub-M/sub-M_ses-CO-20150611_behavior+ecephys.nwb", 44_541_220,
         "23df89f58e66f12a20ea92f00522826bb3d63e99f166b6c1549ca8fe5c780f88"),
        (14, "07513cb0-727d-4ad1-8499-a28e244419f2", "sub-M_ses-CO-20150626",
         "sub-M/sub-M_ses-CO-20150626_behavior+ecephys.nwb", 47_764_096,
         "122542a1b56e75c786c71055b5a0a924d72a97890d38609090c28bb11bbbb6d9"),
    ),
}


def selected_assets_from_payload(value: object) -> dict[str, tuple[SelectedAsset, ...]]:
    """Validate the complete six-row subset, its order, and its literals."""
    if not isinstance(value, Mapping) or set(value) != set(SURFACES):
        raise QuickScreenError("selected-asset surface topology drift")
    selected: dict[str, tuple[SelectedAsset, ...]] = {}
    for surface in SURFACES:
        rows = value.get(surface)
        if not isinstance(rows, (list, tuple)) or len(rows) != 3:
            raise QuickScreenError("selected-asset row count drift")
        rebuilt: list[SelectedAsset] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "surface", "full_roster_index", "asset_id", "session", "frozen_path", "bytes", "sha256",
            }:
                raise QuickScreenError("selected-asset schema drift")
            asset = SelectedAsset(
                surface=row["surface"], full_roster_index=row["full_roster_index"], asset_id=row["asset_id"],
                session=row["session"], frozen_path=row["frozen_path"], byte_count=row["bytes"], sha256=row["sha256"],
            )
            asset.payload()
            rebuilt.append(asset)
        literal = _SELECTED_LITERALS[surface]
        actual = tuple(
            (item.full_roster_index, item.asset_id, item.session, item.frozen_path, item.byte_count, item.sha256)
            for item in rebuilt
        )
        if actual != literal:
            raise QuickScreenError("selected-asset literal/order drift")
        selected[surface] = tuple(rebuilt)
    return selected


def selected_asset_payload() -> dict[str, list[dict[str, object]]]:
    """The frozen six rows, useful for a dry plan and synthetic fixtures."""
    return {
        surface: [
            SelectedAsset(
                surface=surface, full_roster_index=index, asset_id=asset_id, session=session,
                frozen_path=frozen_path, byte_count=byte_count, sha256=digest,
            ).payload()
            for index, asset_id, session, frozen_path, byte_count, digest in _SELECTED_LITERALS[surface]
        ]
        for surface in SURFACES
    }


def select_from_full_authority_rows(value: object) -> dict[str, list[dict[str, object]]]:
    """Select the fixed indices only after exact full C1/v2 rows are present.

    The production adapter invokes the existing descriptor-derived full-authority
    construction and then this function.  This function deliberately refuses a
    shortened or caller-selected table: it protects against a convenient 3-row
    surrogate being treated as C1 val-6 or external sub-M-15 authority.
    """
    if not isinstance(value, Mapping) or set(value) != set(SURFACES):
        raise QuickScreenError("full authority rows surface topology drift")
    counts = {WITHIN: FULL_WITHIN_COUNT, EXTERNAL: FULL_EXTERNAL_COUNT}
    result: dict[str, list[dict[str, object]]] = {}
    for surface in SURFACES:
        rows = value.get(surface)
        if not isinstance(rows, list) or len(rows) != counts[surface]:
            raise QuickScreenError("full authority rows cardinality drift")
        chosen = []
        for index in (WITHIN_INDICES if surface == WITHIN else EXTERNAL_INDICES):
            row = rows[index]
            if not isinstance(row, Mapping):
                raise QuickScreenError("full authority selected row type drift")
            chosen.append({
                "surface": surface,
                "full_roster_index": index,
                "asset_id": row.get("asset_id"),
                "session": row.get("session"),
                "frozen_path": row.get("frozen_path"),
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256"),
            })
        result[surface] = chosen
    # A durable quick preflight must bind the literal selection rather than a
    # row map merely self-consistent with a possibly altered manifest/ledger.
    selected_assets_from_payload(result)
    return result


def selected_transfer_bytes() -> int:
    return sum(item[4] for surface in SURFACES for item in _SELECTED_LITERALS[surface])


@dataclass(frozen=True)
class BasePredecessorEvidence:
    """Exact immutable inputs reused by the engineering-only screen."""

    official_preflight_sha256: str = BASE_OFFICIAL_PREFLIGHT_SHA256
    root_authorization_sha256: str = BASE_ROOT_AUTHORIZATION_SHA256
    full_terminal_sha256: str = POSTERIOR_FULL_TERMINAL_SHA256
    full_swa_sha256: str = POSTERIOR_FULL_SWA_SHA256
    full_swa_state_sha256: str = POSTERIOR_FULL_SWA_STATE_SHA256
    full_source_authority_sha256: str = POSTERIOR_FULL_SOURCE_AUTHORITY_SHA256
    full_closure_sha256: str = POSTERIOR_FULL_CLOSURE_SHA256
    full_import_provenance_sha256: str = POSTERIOR_FULL_IMPORT_PROVENANCE_SHA256
    sealed_terminal_sha256: str = SEALED_CELL_D_TERMINAL_SHA256
    sealed_swa_sha256: str = SEALED_CELL_D_SWA_SHA256
    sealed_baseline_sha256: str = SEALED_CELL_D_BASELINE_SHA256

    def payload(self) -> dict[str, object]:
        result = {
            "base_authority": {
                "official_preflight_relative": BASE_OFFICIAL_PREFLIGHT_RELATIVE,
                "official_preflight_sha256": self.official_preflight_sha256,
                "root_authorization_relative": BASE_ROOT_AUTHORIZATION_RELATIVE,
                "root_authorization_sha256": self.root_authorization_sha256,
            },
            "posterior_full": {
                "import_mirror_relative": FULL_MIRROR_ROOT,
                "terminal_sha256": self.full_terminal_sha256,
                "swa_sha256": self.full_swa_sha256,
                "swa_state_sha256": self.full_swa_state_sha256,
                "source_authority_sha256": self.full_source_authority_sha256,
                "full_closure_sha256": self.full_closure_sha256,
                "import_provenance_sha256": self.full_import_provenance_sha256,
            },
            "sealed_cell_d": {
                "terminal_relative": SEALED_CELL_D_TERMINAL_RELATIVE,
                "terminal_sha256": self.sealed_terminal_sha256,
                "swa_relative": SEALED_CELL_D_SWA_RELATIVE,
                "swa_sha256": self.sealed_swa_sha256,
                "baseline_relative": SEALED_CELL_D_BASELINE_RELATIVE,
                "baseline_sha256": self.sealed_baseline_sha256,
            },
        }
        for section in result.values():
            if not isinstance(section, Mapping):
                raise AssertionError("literal predecessor payload topology")
            for key, item in section.items():
                if key.endswith("_sha256"):
                    _sha(item, f"predecessor {key}")
        if result != BasePredecessorEvidence()._literal_payload_without_validation():
            raise QuickScreenError("quick-screen predecessor literal drift")
        return result

    def _literal_payload_without_validation(self) -> dict[str, object]:
        return {
            "base_authority": {
                "official_preflight_relative": BASE_OFFICIAL_PREFLIGHT_RELATIVE,
                "official_preflight_sha256": BASE_OFFICIAL_PREFLIGHT_SHA256,
                "root_authorization_relative": BASE_ROOT_AUTHORIZATION_RELATIVE,
                "root_authorization_sha256": BASE_ROOT_AUTHORIZATION_SHA256,
            },
            "posterior_full": {
                "import_mirror_relative": FULL_MIRROR_ROOT,
                "terminal_sha256": POSTERIOR_FULL_TERMINAL_SHA256,
                "swa_sha256": POSTERIOR_FULL_SWA_SHA256,
                "swa_state_sha256": POSTERIOR_FULL_SWA_STATE_SHA256,
                "source_authority_sha256": POSTERIOR_FULL_SOURCE_AUTHORITY_SHA256,
                "full_closure_sha256": POSTERIOR_FULL_CLOSURE_SHA256,
                "import_provenance_sha256": POSTERIOR_FULL_IMPORT_PROVENANCE_SHA256,
            },
            "sealed_cell_d": {
                "terminal_relative": SEALED_CELL_D_TERMINAL_RELATIVE,
                "terminal_sha256": SEALED_CELL_D_TERMINAL_SHA256,
                "swa_relative": SEALED_CELL_D_SWA_RELATIVE,
                "swa_sha256": SEALED_CELL_D_SWA_SHA256,
                "baseline_relative": SEALED_CELL_D_BASELINE_RELATIVE,
                "baseline_sha256": SEALED_CELL_D_BASELINE_SHA256,
            },
        }


@dataclass(frozen=True)
class V1FailedPredecessorEvidence:
    """Literal remote V1 failure that remains one V3 predecessor.

    This evidence is intentionally separate from the completed full-training
    and sealed Cell-D inputs.  It binds the prior quick route as a failed
    predecessor, never as a writable V3 result location.  The physical V3
    executor descriptor-validates these exact bytes before it can reserve a
    V3 artifact root.
    """

    attempt_sha256: str = V1_ATTEMPT_SHA256
    failure_sha256: str = V1_FAILURE_SHA256
    failure_error_sha256: str = V1_FAILURE_ERROR_SHA256

    def payload(self) -> dict[str, object]:
        attempt_sha = _sha(self.attempt_sha256, "V1 quick attempt SHA")
        failure_sha = _sha(self.failure_sha256, "V1 quick failure SHA")
        error_sha = _sha(self.failure_error_sha256, "V1 quick failure error SHA")
        result = {
            "schema": "posterior_carrier_quick_screen_v1_failed_predecessor_v1",
            "remote_stage_root": V1_REMOTE_STAGE_ROOT,
            "remote_result_relative": V1_REMOTE_SCORE_ROOT_RELATIVE,
            "remote_result_directory": V1_REMOTE_RESULT_DIRECTORY,
            "attempt": {
                "name": "attempt.json",
                "sha256": attempt_sha,
                "sidecar": f"{attempt_sha}  attempt.json\n",
                "mode": "0444",
                "schema": "posterior_carrier_quick_screen_attempt_v1",
                "status": "ATTEMPT_RESERVED_BEFORE_EVALUATION_PATH_RESOLUTION",
            },
            "failure": {
                "name": "failure.json",
                "sha256": failure_sha,
                "sidecar": f"{failure_sha}  failure.json\n",
                "mode": "0444",
                "schema": "posterior_carrier_quick_screen_failure_v1",
                "stage": "prepare",
                "error_class": "PhysicalQuickScreenError",
                "error_sha256": error_sha,
                "input_authority_sha256": None,
                "terminal_published": False,
            },
            "required_absent": ["input_authority.json", "score.json", "terminal.json"],
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
        }
        if result != V1FailedPredecessorEvidence()._literal_payload_without_validation():
            raise QuickScreenError("quick V1 failure predecessor literal drift")
        return result

    @staticmethod
    def _literal_payload_without_validation() -> dict[str, object]:
        return {
            "schema": "posterior_carrier_quick_screen_v1_failed_predecessor_v1",
            "remote_stage_root": V1_REMOTE_STAGE_ROOT,
            "remote_result_relative": V1_REMOTE_SCORE_ROOT_RELATIVE,
            "remote_result_directory": V1_REMOTE_RESULT_DIRECTORY,
            "attempt": {
                "name": "attempt.json",
                "sha256": V1_ATTEMPT_SHA256,
                "sidecar": f"{V1_ATTEMPT_SHA256}  attempt.json\n",
                "mode": "0444",
                "schema": "posterior_carrier_quick_screen_attempt_v1",
                "status": "ATTEMPT_RESERVED_BEFORE_EVALUATION_PATH_RESOLUTION",
            },
            "failure": {
                "name": "failure.json",
                "sha256": V1_FAILURE_SHA256,
                "sidecar": f"{V1_FAILURE_SHA256}  failure.json\n",
                "mode": "0444",
                "schema": "posterior_carrier_quick_screen_failure_v1",
                "stage": "prepare",
                "error_class": "PhysicalQuickScreenError",
                "error_sha256": V1_FAILURE_ERROR_SHA256,
                "input_authority_sha256": None,
                "terminal_published": False,
            },
            "required_absent": ["input_authority.json", "score.json", "terminal.json"],
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
        }


def validate_v1_failed_predecessor_evidence(value: object) -> dict[str, object]:
    """Accept only the root-provided immutable V1 failure literal."""
    if isinstance(value, V1FailedPredecessorEvidence):
        payload = value.payload()
    else:
        payload = _copy_mapping(value, "V1 quick failure predecessor evidence")
    expected = V1FailedPredecessorEvidence()._literal_payload_without_validation()
    if payload != expected:
        raise QuickScreenError("quick V1 failure predecessor evidence drift")
    return expected


@dataclass(frozen=True)
class V2FailedPredecessorEvidence:
    """Literal remote V2 failure that licenses a fresh V3 successor only.

    V2 is preserved as immutable pre-input failure evidence.  It cannot be
    reused as a V3 source, stage, or result root; its four exact leaves must be
    descriptor-validated before a V3 capability is accepted or a V3 root is
    reserved.
    """

    attempt_sha256: str = V2_ATTEMPT_SHA256
    failure_sha256: str = V2_FAILURE_SHA256
    failure_error_sha256: str = V2_FAILURE_ERROR_SHA256

    def payload(self) -> dict[str, object]:
        attempt_sha = _sha(self.attempt_sha256, "V2 quick attempt SHA")
        failure_sha = _sha(self.failure_sha256, "V2 quick failure SHA")
        error_sha = _sha(self.failure_error_sha256, "V2 quick failure error SHA")
        result = {
            "schema": "posterior_carrier_quick_screen_v2_failed_predecessor_v1",
            "remote_stage_root": V2_REMOTE_STAGE_ROOT,
            "remote_result_relative": V2_REMOTE_SCORE_ROOT_RELATIVE,
            "remote_result_directory": V2_REMOTE_RESULT_DIRECTORY,
            "attempt": {
                "name": "attempt.json",
                "sha256": attempt_sha,
                "sidecar": f"{attempt_sha}  attempt.json\n",
                "mode": "0444",
                "schema": "posterior_carrier_quick_screen_attempt_v2",
                "status": "ATTEMPT_RESERVED_BEFORE_EVALUATION_PATH_RESOLUTION",
            },
            "failure": {
                "name": "failure.json",
                "sha256": failure_sha,
                "sidecar": f"{failure_sha}  failure.json\n",
                "mode": "0444",
                "schema": "posterior_carrier_quick_screen_failure_v2",
                "stage": "prepare",
                "error_class": "PhysicalQuickScreenError",
                "error_sha256": error_sha,
                "input_authority_sha256": None,
                "terminal_published": False,
            },
            "required_absent": ["input_authority.json", "score.json", "terminal.json"],
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
        }
        if result != V2FailedPredecessorEvidence()._literal_payload_without_validation():
            raise QuickScreenError("quick V2 failure predecessor literal drift")
        return result

    @staticmethod
    def _literal_payload_without_validation() -> dict[str, object]:
        return {
            "schema": "posterior_carrier_quick_screen_v2_failed_predecessor_v1",
            "remote_stage_root": V2_REMOTE_STAGE_ROOT,
            "remote_result_relative": V2_REMOTE_SCORE_ROOT_RELATIVE,
            "remote_result_directory": V2_REMOTE_RESULT_DIRECTORY,
            "attempt": {
                "name": "attempt.json",
                "sha256": V2_ATTEMPT_SHA256,
                "sidecar": f"{V2_ATTEMPT_SHA256}  attempt.json\n",
                "mode": "0444",
                "schema": "posterior_carrier_quick_screen_attempt_v2",
                "status": "ATTEMPT_RESERVED_BEFORE_EVALUATION_PATH_RESOLUTION",
            },
            "failure": {
                "name": "failure.json",
                "sha256": V2_FAILURE_SHA256,
                "sidecar": f"{V2_FAILURE_SHA256}  failure.json\n",
                "mode": "0444",
                "schema": "posterior_carrier_quick_screen_failure_v2",
                "stage": "prepare",
                "error_class": "PhysicalQuickScreenError",
                "error_sha256": V2_FAILURE_ERROR_SHA256,
                "input_authority_sha256": None,
                "terminal_published": False,
            },
            "required_absent": ["input_authority.json", "score.json", "terminal.json"],
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
        }


def validate_v2_failed_predecessor_evidence(value: object) -> dict[str, object]:
    """Accept only the root-provided immutable V2 failure literal."""
    if isinstance(value, V2FailedPredecessorEvidence):
        payload = value.payload()
    else:
        payload = _copy_mapping(value, "V2 quick failure predecessor evidence")
    expected = V2FailedPredecessorEvidence()._literal_payload_without_validation()
    if payload != expected:
        raise QuickScreenError("quick V2 failure predecessor evidence drift")
    return expected


@dataclass(frozen=True)
class QuickScreenIdentity:
    """Frozen screen scope, code closure, selection, and predecessor family."""

    closure: ImplementationClosure
    predecessors: BasePredecessorEvidence = field(default_factory=BasePredecessorEvidence)
    v1_failed_predecessor: V1FailedPredecessorEvidence = field(default_factory=V1FailedPredecessorEvidence)
    v2_failed_predecessor: V2FailedPredecessorEvidence = field(default_factory=V2FailedPredecessorEvidence)
    selected_assets: Mapping[str, Sequence[Mapping[str, object]]] = field(default_factory=selected_asset_payload)

    def payload(self) -> dict[str, object]:
        selected = selected_assets_from_payload(self.selected_assets)
        selected_payload = {surface: [item.payload() for item in selected[surface]] for surface in SURFACES}
        closure = self.closure.payload()
        return {
            "cell": CELL,
            "phase": PHASE,
            "classification": CLASSIFICATION,
            "workorder": {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256},
            "implementation_closure": closure,
            "predecessors": self.predecessors.payload(),
            "v1_failed_predecessor": validate_v1_failed_predecessor_evidence(self.v1_failed_predecessor),
            "v2_failed_predecessor": validate_v2_failed_predecessor_evidence(self.v2_failed_predecessor),
            "selected_assets": selected_payload,
            "selected_rosters": {surface: [asset.session for asset in selected[surface]] for surface in SURFACES},
            "matrix": [cell.payload() for cell in quick_screen_matrix()],
            "metric": dict(METRIC),
            "boundaries": dict(BOUNDARIES),
            "remote_stage": {
                "stage_name": REMOTE_STAGE_NAME,
                "stage_root": REMOTE_STAGE_ROOT,
                "score_root_relative": REMOTE_SCORE_ROOT_RELATIVE,
                "evaluation_root_relative": REMOTE_EVALUATION_ROOT_RELATIVE,
                "engineering_device_authority": remote_engineering_device_payload(),
            },
        }


def validate_identity(value: QuickScreenIdentity) -> dict[str, object]:
    if not isinstance(value, QuickScreenIdentity):
        raise QuickScreenError("quick-screen identity must be typed")
    payload = value.payload()
    if (
        payload["classification"] != CLASSIFICATION
        or payload["matrix"] != [cell.payload() for cell in quick_screen_matrix()]
        or payload["boundaries"] != BOUNDARIES
        or payload["metric"] != METRIC
        or payload.get("v1_failed_predecessor") != validate_v1_failed_predecessor_evidence(value.v1_failed_predecessor)
        or payload.get("v2_failed_predecessor") != validate_v2_failed_predecessor_evidence(value.v2_failed_predecessor)
        or not isinstance(payload.get("remote_stage"), Mapping)
        or validate_remote_engineering_device_payload(payload["remote_stage"].get("engineering_device_authority"))
        != remote_engineering_device_payload()
    ):
        raise QuickScreenError("quick-screen identity scientific boundary drift")
    return payload


@dataclass(frozen=True)
class ScoreCell:
    surface: str
    mode: str
    budget: int

    def payload(self) -> dict[str, object]:
        if self.surface not in SURFACES or self.mode not in (SEALED_POINT_MODE, POSTERIOR_MODE) or self.budget not in BUDGETS:
            raise QuickScreenError("quick score-cell topology drift")
        return {"surface": self.surface, "mode": self.mode, "budget": self.budget}


def quick_screen_matrix() -> tuple[ScoreCell, ...]:
    """Exact two-surface × four-cell order, with no mechanism diagnostics."""
    return tuple(
        cell
        for surface in SURFACES
        for cell in (
            ScoreCell(surface, SEALED_POINT_MODE, 30),
            ScoreCell(surface, POSTERIOR_MODE, 30),
            ScoreCell(surface, SEALED_POINT_MODE, 4),
            ScoreCell(surface, POSTERIOR_MODE, 4),
        )
    )


@dataclass(frozen=True)
class SessionInput:
    """One materialized no-cache input shared across both system forwards."""

    surface: str
    session: str
    n_windows: int
    neural_sha256: str
    calibration_m30_sha256: str
    last_bin_target_sha256: str
    last_bin_valid_mask_sha256: str
    last_bin_valid_count: int
    prefix_row_ids_sha256s: Mapping[str, str]
    point_carrier_sha256s: Mapping[str, str]
    posterior_carrier_sha256s: Mapping[str, str]
    materialization_sha256: str

    def payload(self) -> dict[str, object]:
        if self.surface not in SURFACES or not isinstance(self.session, str) or not self.session:
            raise QuickScreenError("quick session-input surface/session drift")
        if type(self.n_windows) is not int or self.n_windows <= 0 or self.last_bin_valid_count != self.n_windows:
            raise QuickScreenError("quick session-input governing window/count drift")
        for name, value in (
            ("neural", self.neural_sha256), ("calibration", self.calibration_m30_sha256),
            ("last-bin target", self.last_bin_target_sha256), ("last-bin mask", self.last_bin_valid_mask_sha256),
            ("materialization", self.materialization_sha256),
        ):
            _sha(value, f"quick session-input {name} SHA")
        for label, mapping in (
            ("prefix rows", self.prefix_row_ids_sha256s),
            ("point carrier", self.point_carrier_sha256s),
            ("posterior carrier", self.posterior_carrier_sha256s),
        ):
            if not isinstance(mapping, Mapping) or set(mapping) != {"30", "4"}:
                raise QuickScreenError(f"quick session-input {label} M30/M4 topology drift")
            for item in mapping.values():
                _sha(item, f"quick session-input {label} SHA")
        return {
            "surface": self.surface,
            "session": self.session,
            "n_windows": self.n_windows,
            "neural_sha256": self.neural_sha256,
            "calibration_m30_sha256": self.calibration_m30_sha256,
            "last_bin_target_sha256": self.last_bin_target_sha256,
            "last_bin_valid_mask_sha256": self.last_bin_valid_mask_sha256,
            "last_bin_valid_count": self.last_bin_valid_count,
            "prefix_row_ids_sha256s": {key: self.prefix_row_ids_sha256s[key] for key in ("30", "4")},
            "point_carrier_sha256s": {key: self.point_carrier_sha256s[key] for key in ("30", "4")},
            "posterior_carrier_sha256s": {key: self.posterior_carrier_sha256s[key] for key in ("30", "4")},
            "materialization_sha256": self.materialization_sha256,
        }


def _session_input_from_payload(value: Mapping[str, object]) -> SessionInput:
    expected = {
        "surface", "session", "n_windows", "neural_sha256", "calibration_m30_sha256",
        "last_bin_target_sha256", "last_bin_valid_mask_sha256", "last_bin_valid_count",
        "prefix_row_ids_sha256s", "point_carrier_sha256s", "posterior_carrier_sha256s", "materialization_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise QuickScreenError("quick session-input payload schema drift")
    return SessionInput(
        surface=value["surface"], session=value["session"], n_windows=value["n_windows"],
        neural_sha256=value["neural_sha256"], calibration_m30_sha256=value["calibration_m30_sha256"],
        last_bin_target_sha256=value["last_bin_target_sha256"], last_bin_valid_mask_sha256=value["last_bin_valid_mask_sha256"],
        last_bin_valid_count=value["last_bin_valid_count"], prefix_row_ids_sha256s=value["prefix_row_ids_sha256s"],
        point_carrier_sha256s=value["point_carrier_sha256s"], posterior_carrier_sha256s=value["posterior_carrier_sha256s"],
        materialization_sha256=value["materialization_sha256"],
    )


@dataclass(frozen=True)
class InputAuthority:
    records: tuple[SessionInput, ...]
    shared_input_pass: bool = True
    normalizer_refit: bool = False
    target_sampling: bool = False

    def payload(self, *, identity: QuickScreenIdentity) -> dict[str, object]:
        identity_payload = validate_identity(identity)
        expected = tuple(
            (surface, session)
            for surface in SURFACES
            for session in identity_payload["selected_rosters"][surface]
        )
        records = tuple(item.payload() for item in self.records)
        if tuple((item["surface"], item["session"]) for item in records) != expected:
            raise QuickScreenError("quick input-authority selected roster/order drift")
        if self.shared_input_pass is not True or self.normalizer_refit is not False or self.target_sampling is not False:
            raise QuickScreenError("quick input-authority materialization/refit boundary drift")
        return {
            "schema": "posterior_carrier_quick_screen_input_authority_v3",
            "identity": identity_payload,
            "records": list(records),
            "shared_input_pass": True,
            "normalizer_refit": False,
            "target_sampling": False,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "formal_opened": False,
            "h1_opened": False,
        }


def validate_input_authority_payload(value: Mapping[str, object], *, identity: QuickScreenIdentity) -> dict[str, object]:
    expected = {
        "schema", "identity", "records", "shared_input_pass", "normalizer_refit", "target_sampling",
        "target_optimizer_steps", "target_backward_calls", "target_update_calls", "formal_opened", "h1_opened",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise QuickScreenError("quick input-authority schema drift")
    if value.get("identity") != validate_identity(identity) or not isinstance(value.get("records"), list):
        raise QuickScreenError("quick input-authority identity/records drift")
    rebuilt = InputAuthority(
        records=tuple(_session_input_from_payload(item) for item in value["records"]),
        shared_input_pass=value["shared_input_pass"], normalizer_refit=value["normalizer_refit"],
        target_sampling=value["target_sampling"],
    ).payload(identity=identity)
    if rebuilt != dict(value):
        raise QuickScreenError("quick input-authority exact reconstruction drift")
    return rebuilt


@dataclass(frozen=True)
class SessionScore:
    session: str
    n_windows: int
    r2: float
    prediction_sha256: str
    input_record_sha256: str

    def payload(self) -> dict[str, object]:
        if not isinstance(self.session, str) or not self.session or type(self.n_windows) is not int or self.n_windows <= 0:
            raise QuickScreenError("quick session-score session/window drift")
        return {
            "session": self.session,
            "n_windows": self.n_windows,
            "r2": _finite(self.r2, "quick session R2"),
            "prediction_sha256": _sha(self.prediction_sha256, "quick prediction SHA"),
            "input_record_sha256": _sha(self.input_record_sha256, "quick input-record SHA"),
        }


@dataclass(frozen=True)
class CellEvidence:
    cell: ScoreCell
    model_system: str
    model_swa_sha256: str
    sessions: tuple[SessionScore, ...]
    input_authority_sha256: str
    model_state_before_sha256: str
    model_state_after_sha256: str
    eval_mode: bool
    dropout_disabled: bool
    gradients_none: bool
    finite_outputs: bool
    repeated_fixed_batch_bitwise_equal: bool
    b3s_m30_recomputed: bool
    no_target_sampling: bool

    def payload(self, *, identity: QuickScreenIdentity, input_payload: Mapping[str, object]) -> dict[str, object]:
        cell = self.cell.payload()
        expected_system = "sealed_cell_d_checkpoint" if self.cell.mode == SEALED_POINT_MODE else "posterior_carrier_full_swa"
        expected_swa = (
            identity.predecessors.sealed_swa_sha256 if self.cell.mode == SEALED_POINT_MODE
            else identity.predecessors.full_swa_sha256
        )
        selected = validate_identity(identity)["selected_rosters"][self.cell.surface]
        rows = tuple(item.payload() for item in self.sessions)
        if tuple(item["session"] for item in rows) != tuple(selected):
            raise QuickScreenError("quick cell evidence selected session order drift")
        records = {
            item["session"]: item
            for item in input_payload["records"]
            if item["surface"] == self.cell.surface
        }
        if tuple(records) != tuple(selected):
            raise QuickScreenError("quick cell evidence input record roster drift")
        if any(
            item["n_windows"] != records[item["session"]]["n_windows"]
            or item["input_record_sha256"] != _digest(_json(records[item["session"]]))
            for item in rows
        ):
            raise QuickScreenError("quick cell evidence same-materialized-input drift")
        if (
            self.model_system != expected_system or self.model_swa_sha256 != expected_swa
            or self.input_authority_sha256 != _digest(_json(input_payload))
            or self.model_state_before_sha256 != self.model_state_after_sha256
            or self.eval_mode is not True or self.dropout_disabled is not True or self.gradients_none is not True
            or self.finite_outputs is not True or self.repeated_fixed_batch_bitwise_equal is not True
            or self.b3s_m30_recomputed is not True or self.no_target_sampling is not True
        ):
            raise QuickScreenError("quick cell evidence model/eval/no-update boundary drift")
        return {
            "cell": cell,
            "model_system": expected_system,
            "model_swa_sha256": _sha(expected_swa, "quick model SWA SHA"),
            "sessions": list(rows),
            "input_authority_sha256": _sha(self.input_authority_sha256, "quick input authority SHA"),
            "model_state_before_sha256": _sha(self.model_state_before_sha256, "quick model-state-before SHA"),
            "model_state_after_sha256": _sha(self.model_state_after_sha256, "quick model-state-after SHA"),
            "eval_mode": True,
            "dropout_disabled": True,
            "gradients_none": True,
            "finite_outputs": True,
            "repeated_fixed_batch_bitwise_equal": True,
            "b3s_m30_recomputed": True,
            "no_target_sampling": True,
        }


def _cell_from_payload(value: Mapping[str, object]) -> ScoreCell:
    if not isinstance(value, Mapping) or set(value) != {"surface", "mode", "budget"}:
        raise QuickScreenError("quick score-cell payload schema drift")
    cell = ScoreCell(surface=value["surface"], mode=value["mode"], budget=value["budget"])
    cell.payload()
    return cell


def _session_score_from_payload(value: Mapping[str, object]) -> SessionScore:
    if not isinstance(value, Mapping) or set(value) != {"session", "n_windows", "r2", "prediction_sha256", "input_record_sha256"}:
        raise QuickScreenError("quick session-score payload schema drift")
    return SessionScore(
        session=value["session"], n_windows=value["n_windows"], r2=value["r2"],
        prediction_sha256=value["prediction_sha256"], input_record_sha256=value["input_record_sha256"],
    )


def validate_cell_evidence_payload(
    value: Mapping[str, object], *, identity: QuickScreenIdentity, input_payload: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "cell", "model_system", "model_swa_sha256", "sessions", "input_authority_sha256",
        "model_state_before_sha256", "model_state_after_sha256", "eval_mode", "dropout_disabled",
        "gradients_none", "finite_outputs", "repeated_fixed_batch_bitwise_equal", "b3s_m30_recomputed",
        "no_target_sampling",
    }
    if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value.get("sessions"), list):
        raise QuickScreenError("quick cell-evidence schema drift")
    rebuilt = CellEvidence(
        cell=_cell_from_payload(value["cell"]), model_system=value["model_system"], model_swa_sha256=value["model_swa_sha256"],
        sessions=tuple(_session_score_from_payload(item) for item in value["sessions"]),
        input_authority_sha256=value["input_authority_sha256"], model_state_before_sha256=value["model_state_before_sha256"],
        model_state_after_sha256=value["model_state_after_sha256"], eval_mode=value["eval_mode"],
        dropout_disabled=value["dropout_disabled"], gradients_none=value["gradients_none"], finite_outputs=value["finite_outputs"],
        repeated_fixed_batch_bitwise_equal=value["repeated_fixed_batch_bitwise_equal"],
        b3s_m30_recomputed=value["b3s_m30_recomputed"], no_target_sampling=value["no_target_sampling"],
    ).payload(identity=identity, input_payload=input_payload)
    if rebuilt != dict(value):
        raise QuickScreenError("quick cell-evidence exact reconstruction drift")
    return rebuilt


@dataclass
class RuntimeFlags:
    """Observed runtime facts; no post-hoc inference is allowed."""

    within_opened: bool = False
    external_opened: bool = False
    remote_initialized: bool = False
    target_optimizer_steps: int = 0
    target_backward_calls: int = 0
    target_update_calls: int = 0
    normalizer_refit: bool = False
    target_sampling: bool = False
    h1_opened: bool = False
    formal_opened: bool = False
    forward_cells: list[tuple[str, str, int]] = field(default_factory=list)

    def record_forward(self, cell: ScoreCell) -> None:
        self.forward_cells.append((cell.surface, cell.mode, cell.budget))

    def payload(self) -> dict[str, object]:
        return {
            "within_opened": self.within_opened,
            "external_opened": self.external_opened,
            "remote_initialized": self.remote_initialized,
            "target_optimizer_steps": self.target_optimizer_steps,
            "target_backward_calls": self.target_backward_calls,
            "target_update_calls": self.target_update_calls,
            "normalizer_refit": self.normalizer_refit,
            "target_sampling": self.target_sampling,
            "h1_opened": self.h1_opened,
            "formal_opened": self.formal_opened,
            "forward_cells": [list(item) for item in self.forward_cells],
        }


def _validate_runtime_flags(flags: RuntimeFlags, *, require_all_cells: bool) -> None:
    expected = [(item.surface, item.mode, item.budget) for item in quick_screen_matrix()]
    if (
        flags.target_optimizer_steps != 0 or flags.target_backward_calls != 0 or flags.target_update_calls != 0
        or flags.normalizer_refit is not False or flags.target_sampling is not False
        or flags.h1_opened is not False or flags.formal_opened is not False
    ):
        raise QuickScreenError("quick-screen crossed target/refit/H1/formal boundary")
    if require_all_cells:
        if (
            flags.within_opened is not True or flags.external_opened is not True
            or flags.remote_initialized is not True or flags.forward_cells != expected
        ):
            raise QuickScreenError("quick-screen input/forward matrix boundary drift")


def _median(values: Sequence[float], *, label: str) -> float:
    if not values:
        raise QuickScreenError(f"{label} requires values")
    ordered = sorted(_finite(item, label) for item in values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _surface_summary(rows: Sequence[Mapping[str, object]], *, label: str) -> dict[str, object]:
    if not rows:
        raise QuickScreenError(f"{label} requires rows")
    values = [_finite(row.get("r2"), f"{label} R2") for row in rows]
    return {"equal_session_mean": sum(values) / len(values), "equal_session_median": _median(values, label=label)}


def _paired_summary(posterior: Sequence[Mapping[str, object]], point: Sequence[Mapping[str, object]], *, label: str) -> dict[str, object]:
    names = tuple(row.get("session") for row in posterior)
    if names != tuple(row.get("session") for row in point) or any(not isinstance(name, str) or not name for name in names):
        raise QuickScreenError(f"{label} paired session topology drift")
    deltas = [_finite(left.get("r2"), "posterior R2") - _finite(right.get("r2"), "point R2") for left, right in zip(posterior, point)]
    return {
        "label": label,
        "sessions": list(names),
        "deltas": deltas,
        "mean": sum(deltas) / len(deltas),
        "median": _median(deltas, label=f"{label} deltas"),
        "positive_count": sum(item > 0.0 for item in deltas),
        "n_sessions": len(deltas),
    }


def build_score_payload(
    *, identity: QuickScreenIdentity, input_payload: Mapping[str, object], evidence: Sequence[Mapping[str, object],],
    flags: RuntimeFlags, device_attestation: Mapping[str, object],
    upstream_formal_preflight_device: Mapping[str, object], posterior_final4_compatibility: Mapping[str, object],
) -> dict[str, object]:
    """Build a descriptive-only result after every exact cell is validated."""
    checked_input = validate_input_authority_payload(input_payload, identity=identity)
    if not isinstance(evidence, Sequence) or len(evidence) != len(quick_screen_matrix()):
        raise QuickScreenError("quick screen evidence cardinality drift")
    checked = [validate_cell_evidence_payload(item, identity=identity, input_payload=checked_input) for item in evidence]
    if tuple(item["cell"] for item in checked) != tuple(cell.payload() for cell in quick_screen_matrix()):
        raise QuickScreenError("quick screen evidence order/matrix drift")
    _validate_runtime_flags(flags, require_all_cells=True)
    device = validate_remote_engineering_device_payload(device_attestation)
    upstream_device = validate_upstream_formal_preflight_device_payload(upstream_formal_preflight_device)
    final4 = validate_posterior_final4_compatibility_payload(posterior_final4_compatibility)
    summaries: dict[str, dict[str, object]] = {}
    paired: dict[str, dict[str, object]] = {}
    for surface in SURFACES:
        surface_rows = [item for item in checked if item["cell"]["surface"] == surface]
        if len(surface_rows) != 4:
            raise QuickScreenError("quick screen surface evidence count drift")
        by_key = {(row["cell"]["mode"], row["cell"]["budget"]): row for row in surface_rows}
        if set(by_key) != {(SEALED_POINT_MODE, 30), (POSTERIOR_MODE, 30), (SEALED_POINT_MODE, 4), (POSTERIOR_MODE, 4)}:
            raise QuickScreenError("quick screen surface cell topology drift")
        summaries[surface] = {}
        paired[surface] = {}
        for budget in BUDGETS:
            point = by_key[(SEALED_POINT_MODE, budget)]["sessions"]
            posterior = by_key[(POSTERIOR_MODE, budget)]["sessions"]
            summaries[surface][f"sealed_point_m{budget}"] = _surface_summary(point, label=f"{surface} sealed M{budget}")
            summaries[surface][f"posterior_m{budget}"] = _surface_summary(posterior, label=f"{surface} posterior M{budget}")
            paired[surface][f"posterior_minus_point_m{budget}"] = _paired_summary(
                posterior, point, label=f"{surface} posterior-minus-point M{budget}",
            )
    return {
        "schema": "posterior_carrier_quick_screen_score_v3",
        "status": "NON_GOVERNING_QUICK_SCREEN_COMPLETE__NO_GOVERNING_VERDICT",
        "classification": CLASSIFICATION,
        "cell": CELL,
        "phase": PHASE,
        "identity": validate_identity(identity),
        "input_authority_sha256": _digest(_json(checked_input)),
        "metric": dict(METRIC),
        "matrix": checked,
        "summaries": summaries,
        "paired_posterior_minus_sealed_point": paired,
        "device_attestation": device,
        "upstream_formal_preflight_device": upstream_device,
        "posterior_final4_compatibility": final4,
        "flags": flags.payload(),
        "no_governing_decision": True,
        "excluded": {
            "m10": True, "zero": True, "wrong_pair": True, "h1": True, "formal": True,
            "full_window_headline": True, "target_update": True, "normalizer_refit": True,
            "target_sampling": True, "checkpoint_selection": True,
        },
    }


def validate_score_payload(value: Mapping[str, object], *, identity: QuickScreenIdentity,
                           input_payload: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema", "status", "classification", "cell", "phase", "identity", "input_authority_sha256", "metric",
        "matrix", "summaries", "paired_posterior_minus_sealed_point", "device_attestation",
        "upstream_formal_preflight_device", "posterior_final4_compatibility", "flags",
        "no_governing_decision", "excluded",
    }
    if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value.get("matrix"), list):
        raise QuickScreenError("quick score payload schema drift")
    if (
        value.get("schema") != "posterior_carrier_quick_screen_score_v3"
        or value.get("status") != "NON_GOVERNING_QUICK_SCREEN_COMPLETE__NO_GOVERNING_VERDICT"
        or value.get("classification") != CLASSIFICATION or value.get("cell") != CELL or value.get("phase") != PHASE
        or value.get("identity") != validate_identity(identity)
        or value.get("input_authority_sha256") != _digest(_json(validate_input_authority_payload(input_payload, identity=identity)))
        or value.get("metric") != METRIC or value.get("no_governing_decision") is not True
    ):
        raise QuickScreenError("quick score payload identity/metric/non-governing drift")
    flags = value.get("flags")
    if not isinstance(flags, Mapping):
        raise QuickScreenError("quick score flags schema drift")
    rebuilt_flags = RuntimeFlags(
        within_opened=flags.get("within_opened"), external_opened=flags.get("external_opened"),
        remote_initialized=flags.get("remote_initialized"), target_optimizer_steps=flags.get("target_optimizer_steps"),
        target_backward_calls=flags.get("target_backward_calls"), target_update_calls=flags.get("target_update_calls"),
        normalizer_refit=flags.get("normalizer_refit"), target_sampling=flags.get("target_sampling"),
        h1_opened=flags.get("h1_opened"), formal_opened=flags.get("formal_opened"),
        forward_cells=[tuple(item) for item in flags.get("forward_cells", [])],
    )
    if rebuilt_flags.payload() != dict(flags):
        raise QuickScreenError("quick score flags exact reconstruction drift")
    rebuilt = build_score_payload(
        identity=identity, input_payload=input_payload, evidence=value["matrix"], flags=rebuilt_flags,
        device_attestation=value["device_attestation"],
        upstream_formal_preflight_device=value["upstream_formal_preflight_device"],
        posterior_final4_compatibility=value["posterior_final4_compatibility"],
    )
    if rebuilt != dict(value):
        raise QuickScreenError("quick score payload exact reconstruction drift")
    return rebuilt


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise QuickScreenError(f"cannot lstat artifact directory: {path}") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise QuickScreenError("artifact directory must be a real non-symlink")
    return int(info.st_dev), int(info.st_ino)


@dataclass
class ArtifactRoot:
    """Finite immutable receipt root with O_EXCL/fsync/0444 leaves."""

    directory: Path
    topology: tuple[str, ...]
    identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]

    def _reverify(self) -> None:
        if _directory_identity(self.directory) != self.identity or _directory_identity(self.parent) != self.parent_identity:
            raise QuickScreenError("artifact root/parent identity drift")

    def has_name(self, name: str) -> bool:
        _safe_component(name, "artifact name")
        self._reverify()
        return (self.directory / name).exists()

    def _write_pair(self, name: str, body: bytes) -> str:
        _safe_component(name, "artifact name")
        if name not in self.topology:
            raise QuickScreenError("artifact leaf outside reviewed topology")
        self._reverify()
        digest = _digest(body)
        paths = ((name, body), (f"{name}.sha256", f"{digest}  {name}\n".encode("ascii")))
        made: list[Path] = []
        try:
            for leaf, content in paths:
                path = self.directory / leaf
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
                try:
                    view = memoryview(content)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise QuickScreenError("short immutable receipt write")
                        view = view[written:]
                    os.fsync(fd)
                    os.fchmod(fd, 0o444)
                finally:
                    os.close(fd)
                made.append(path)
            parent_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            for path in reversed(made):
                try:
                    info = os.lstat(path)
                    if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                        os.unlink(path)
                except OSError:
                    pass
            raise
        self._reverify()
        return digest

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        return self._write_pair(name, _json(payload))

    def reload_json(self, name: str, expected_sha256: str) -> dict[str, object]:
        _safe_component(name, "artifact name")
        expected = _sha(expected_sha256, "expected artifact SHA")
        self._reverify()
        path = self.directory / name
        sidecar = self.directory / f"{name}.sha256"
        for item in (path, sidecar):
            info = os.lstat(item)
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                raise QuickScreenError("immutable artifact leaf type/mode drift")
        body = path.read_bytes()
        if _digest(body) != expected or sidecar.read_bytes() != f"{expected}  {name}\n".encode("ascii"):
            raise QuickScreenError("immutable artifact body/sidecar drift")
        try:
            payload = json.loads(body)
        except (TypeError, json.JSONDecodeError) as error:
            raise QuickScreenError("immutable artifact JSON decode drift") from error
        if not isinstance(payload, dict):
            raise QuickScreenError("immutable artifact JSON root drift")
        self._reverify()
        return payload

    def publish_group(self, bodies: Mapping[str, bytes], *, post_publish: Callable[[Mapping[str, str]], None]) -> dict[str, str]:
        if set(bodies) != {"score.json", "terminal.json"}:
            raise QuickScreenError("quick success group must be exactly score plus terminal")
        if self.has_name("failure.json"):
            raise QuickScreenError("quick successful terminal cannot coexist with failure")
        made: list[str] = []
        try:
            hashes: dict[str, str] = {}
            for name in ("score.json", "terminal.json"):
                hashes[name] = self._write_pair(name, bodies[name])
                made.append(name)
            post_publish(hashes)
            return hashes
        except BaseException:
            for name in reversed(made):
                for suffix in (".sha256", ""):
                    path = self.directory / f"{name}{suffix}"
                    try:
                        info = os.lstat(path)
                        if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                            os.unlink(path)
                    except OSError:
                        pass
            raise


RESULT_TOPOLOGY = ("attempt.json", "input_authority.json", "score.json", "terminal.json", "failure.json")


def reserve_result_artifact(root: Path) -> ArtifactRoot:
    """Reserve a fresh result root only from an already reviewed private route."""
    base = Path(root).absolute()
    parent = base / "tfpd_exploration" / "results"
    parent_identity = _directory_identity(parent)
    name = Path(REMOTE_SCORE_ROOT_RELATIVE).name
    _safe_component(name, "quick result root name")
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if _directory_identity(parent) != parent_identity:
            raise QuickScreenError("quick result parent identity drift before reservation")
        try:
            os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise QuickScreenError("quick result root must be fresh")
        os.mkdir(name, 0o755, dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)
    result = parent / name
    return ArtifactRoot(result, RESULT_TOPOLOGY, _directory_identity(result), parent, parent_identity)


_EXECUTION_SEAL = object()


@dataclass(frozen=True)
class QuickScreenExecutionCapability:
    identity_sha256: str
    v1_failed_predecessor_sha256: str
    v2_failed_predecessor_sha256: str
    _seal: object = field(repr=False, compare=False)


def _issue_root_review_capability(identity: QuickScreenIdentity) -> QuickScreenExecutionCapability:
    """Private root-only factory; public flags cannot manufacture this object."""
    checked = validate_identity(identity)
    predecessor = validate_v1_failed_predecessor_evidence(checked["v1_failed_predecessor"])
    predecessor_v2 = validate_v2_failed_predecessor_evidence(checked["v2_failed_predecessor"])
    return QuickScreenExecutionCapability(
        _digest(_json(checked)),
        _digest(_json(predecessor)),
        _digest(_json(predecessor_v2)),
        _EXECUTION_SEAL,
    )


def _require_capability(value: object, *, identity: QuickScreenIdentity) -> QuickScreenExecutionCapability:
    if (
        not isinstance(value, QuickScreenExecutionCapability)
        or value._seal is not _EXECUTION_SEAL
        or value.identity_sha256 != _digest(_json(validate_identity(identity)))
        or value.v1_failed_predecessor_sha256 != _digest(_json(
            validate_v1_failed_predecessor_evidence(validate_identity(identity)["v1_failed_predecessor"]),
        ))
        or value.v2_failed_predecessor_sha256 != _digest(_json(
            validate_v2_failed_predecessor_evidence(validate_identity(identity)["v2_failed_predecessor"]),
        ))
    ):
        raise QuickScreenError("root-reviewed in-process quick-screen capability required before evaluation access")
    return value


def _attempt_payload(identity: QuickScreenIdentity) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_quick_screen_attempt_v3",
        "status": "ATTEMPT_RESERVED_BEFORE_EVALUATION_PATH_RESOLUTION",
        "classification": CLASSIFICATION,
        "cell": CELL,
        "phase": PHASE,
        "identity": validate_identity(identity),
        "boundaries": dict(BOUNDARIES),
        "evaluation_assets_resolved": False,
        "remote_initialized": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
    }


def validate_attempt_payload(value: Mapping[str, object], *, identity: QuickScreenIdentity) -> dict[str, object]:
    expected = {
        "schema", "status", "classification", "cell", "phase", "identity", "boundaries",
        "evaluation_assets_resolved", "remote_initialized", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise QuickScreenError("quick attempt schema drift")
    if (
        value.get("schema") != "posterior_carrier_quick_screen_attempt_v3"
        or value.get("status") != "ATTEMPT_RESERVED_BEFORE_EVALUATION_PATH_RESOLUTION"
        or value.get("classification") != CLASSIFICATION or value.get("cell") != CELL or value.get("phase") != PHASE
        or value.get("identity") != validate_identity(identity) or value.get("boundaries") != BOUNDARIES
        or value.get("evaluation_assets_resolved") is not False or value.get("remote_initialized") is not False
        or value.get("target_optimizer_steps") != 0 or value.get("target_backward_calls") != 0
        or value.get("target_update_calls") != 0
    ):
        raise QuickScreenError("quick attempt immutable boundary drift")
    return dict(value)


def _terminal_payload(*, identity: QuickScreenIdentity, attempt_sha256: str, input_authority_sha256: str,
                      score_sha256: str, final_closure: Mapping[str, object]) -> dict[str, object]:
    closure = identity.closure.payload()
    if dict(final_closure) != closure:
        raise QuickScreenError("quick launch/final closure drift")
    return {
        "schema": "posterior_carrier_quick_screen_terminal_v3",
        "status": "NON_GOVERNING_QUICK_SCREEN_COMPLETE__NO_GOVERNING_VERDICT",
        "classification": CLASSIFICATION,
        "cell": CELL,
        "phase": PHASE,
        "identity": validate_identity(identity),
        "attempt_sha256": _sha(attempt_sha256, "quick terminal attempt SHA"),
        "input_authority_sha256": _sha(input_authority_sha256, "quick terminal input-authority SHA"),
        "score_sha256": _sha(score_sha256, "quick terminal score SHA"),
        "launch_closure": closure,
        "final_closure": closure,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "normalizer_refit": False,
        "target_sampling": False,
        "formal_opened": False,
        "h1_opened": False,
        "score_terminal_transactional_group": True,
        "no_governing_decision": True,
    }


def validate_terminal_payload(value: Mapping[str, object], *, identity: QuickScreenIdentity,
                              score_sha256: str | None = None) -> dict[str, object]:
    expected = {
        "schema", "status", "classification", "cell", "phase", "identity", "attempt_sha256",
        "input_authority_sha256", "score_sha256", "launch_closure", "final_closure", "target_optimizer_steps",
        "target_backward_calls", "target_update_calls", "normalizer_refit", "target_sampling", "formal_opened",
        "h1_opened", "score_terminal_transactional_group", "no_governing_decision",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise QuickScreenError("quick terminal schema drift")
    closure = identity.closure.payload()
    if (
        value.get("schema") != "posterior_carrier_quick_screen_terminal_v3"
        or value.get("status") != "NON_GOVERNING_QUICK_SCREEN_COMPLETE__NO_GOVERNING_VERDICT"
        or value.get("classification") != CLASSIFICATION or value.get("cell") != CELL or value.get("phase") != PHASE
        or value.get("identity") != validate_identity(identity) or value.get("launch_closure") != closure
        or value.get("final_closure") != closure or value.get("target_optimizer_steps") != 0
        or value.get("target_backward_calls") != 0 or value.get("target_update_calls") != 0
        or value.get("normalizer_refit") is not False or value.get("target_sampling") is not False
        or value.get("formal_opened") is not False or value.get("h1_opened") is not False
        or value.get("score_terminal_transactional_group") is not True or value.get("no_governing_decision") is not True
    ):
        raise QuickScreenError("quick terminal immutable binding drift")
    for key in ("attempt_sha256", "input_authority_sha256", "score_sha256"):
        _sha(value.get(key), f"quick terminal {key}")
    if score_sha256 is not None and value["score_sha256"] != _sha(score_sha256, "expected quick score SHA"):
        raise QuickScreenError("quick terminal exact score binding drift")
    return dict(value)


def _failure_payload(*, identity: QuickScreenIdentity, stage: str, flags: RuntimeFlags,
                     attempt_sha256: str | None, input_authority_sha256: str | None,
                     error: BaseException) -> dict[str, object]:
    if not isinstance(stage, str) or not stage:
        raise QuickScreenError("quick failure stage is missing")
    return {
        "schema": "posterior_carrier_quick_screen_failure_v3",
        "classification": CLASSIFICATION,
        "cell": CELL,
        "phase": PHASE,
        "identity": validate_identity(identity),
        "stage": stage,
        "attempt_sha256": attempt_sha256,
        "input_authority_sha256": input_authority_sha256,
        "flags": flags.payload(),
        "terminal_published": False,
        "error_class": type(error).__name__,
        "error_sha256": _digest(repr(error).encode("utf-8")),
        "traceback_sha256": _digest(traceback.format_exc().encode("utf-8")),
    }


def validate_failure_payload(value: Mapping[str, object], *, identity: QuickScreenIdentity) -> dict[str, object]:
    expected = {
        "schema", "classification", "cell", "phase", "identity", "stage", "attempt_sha256",
        "input_authority_sha256", "flags", "terminal_published", "error_class", "error_sha256", "traceback_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise QuickScreenError("quick failure schema drift")
    if (
        value.get("schema") != "posterior_carrier_quick_screen_failure_v3"
        or value.get("classification") != CLASSIFICATION or value.get("cell") != CELL or value.get("phase") != PHASE
        or value.get("identity") != validate_identity(identity) or not isinstance(value.get("stage"), str) or not value["stage"]
        or value.get("terminal_published") is not False or not isinstance(value.get("flags"), Mapping)
        or not isinstance(value.get("error_class"), str) or not value["error_class"]
    ):
        raise QuickScreenError("quick failure immutable boundary drift")
    if value["attempt_sha256"] is not None:
        _sha(value["attempt_sha256"], "quick failure attempt SHA")
    if value["input_authority_sha256"] is not None:
        _sha(value["input_authority_sha256"], "quick failure input authority SHA")
    _sha(value["error_sha256"], "quick failure error SHA")
    _sha(value["traceback_sha256"], "quick failure traceback SHA")
    flags = value["flags"]
    if (
        flags.get("target_optimizer_steps") != 0 or flags.get("target_backward_calls") != 0
        or flags.get("target_update_calls") != 0 or flags.get("normalizer_refit") is not False
        or flags.get("target_sampling") is not False or flags.get("h1_opened") is not False
        or flags.get("formal_opened") is not False
    ):
        raise QuickScreenError("quick failure crossed forbidden boundary")
    return dict(value)


class QuickScreenBackend(Protocol):
    def prepare(self, *, identity: QuickScreenIdentity, flags: RuntimeFlags) -> None: ...

    def resolve_inputs(self, *, identity: QuickScreenIdentity, flags: RuntimeFlags) -> InputAuthority: ...

    def score_cell(self, *, cell: ScoreCell, input_payload: Mapping[str, object], flags: RuntimeFlags) -> CellEvidence: ...

    def reverify_after_forwards(self, *, identity: QuickScreenIdentity, flags: RuntimeFlags) -> ImplementationClosure: ...

    def device_attestation(self) -> Mapping[str, object]: ...

    def upstream_formal_preflight_device(self) -> Mapping[str, object]: ...

    def posterior_final4_compatibility(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


def _publish_failure(artifact: ArtifactRoot, *, identity: QuickScreenIdentity, stage: str, flags: RuntimeFlags,
                     attempt_sha256: str | None, input_authority_sha256: str | None, error: BaseException) -> None:
    if artifact.has_name("terminal.json"):
        raise QuickScreenError("quick failure cannot coexist with terminal")
    if artifact.has_name("failure.json"):
        return
    payload = _failure_payload(
        identity=identity, stage=stage, flags=flags, attempt_sha256=attempt_sha256,
        input_authority_sha256=input_authority_sha256, error=error,
    )
    validate_failure_payload(payload, identity=identity)
    digest = artifact.publish_json("failure.json", payload)
    validate_failure_payload(artifact.reload_json("failure.json", digest), identity=identity)


def run_quick_screen_lifecycle(
    *, artifact: ArtifactRoot, identity: QuickScreenIdentity, execution_capability: object,
    backend: QuickScreenBackend, final_authorization_reverify: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Run a mockable reviewed future route with no partial scientific result."""
    _require_capability(execution_capability, identity=identity)
    flags = RuntimeFlags()
    attempt_sha256: str | None = None
    input_authority_sha256: str | None = None
    terminal_published = False
    stage = "attempt"
    try:
        attempt = _attempt_payload(identity)
        validate_attempt_payload(attempt, identity=identity)
        attempt_sha256 = artifact.publish_json("attempt.json", attempt)
        validate_attempt_payload(artifact.reload_json("attempt.json", attempt_sha256), identity=identity)

        stage = "prepare"
        backend.prepare(identity=identity, flags=flags)
        _validate_runtime_flags(flags, require_all_cells=False)
        stage = "input_authority"
        authority = backend.resolve_inputs(identity=identity, flags=flags)
        input_payload = authority.payload(identity=identity)
        validate_input_authority_payload(input_payload, identity=identity)
        input_authority_sha256 = artifact.publish_json("input_authority.json", input_payload)
        validate_input_authority_payload(artifact.reload_json("input_authority.json", input_authority_sha256), identity=identity)

        stage = "forwards"
        evidence: list[dict[str, object]] = []
        for cell in quick_screen_matrix():
            item = backend.score_cell(cell=cell, input_payload=input_payload, flags=flags)
            if item.cell != cell:
                raise QuickScreenError("quick backend returned wrong score cell")
            payload = item.payload(identity=identity, input_payload=input_payload)
            validate_cell_evidence_payload(payload, identity=identity, input_payload=input_payload)
            flags.record_forward(cell)
            evidence.append(payload)
        _validate_runtime_flags(flags, require_all_cells=True)

        stage = "final_reverify"
        final_closure = backend.reverify_after_forwards(identity=identity, flags=flags).payload()
        if final_closure != identity.closure.payload():
            raise QuickScreenError("quick post-forward closure drift")
        if final_authorization_reverify is not None:
            final_authorization_reverify()
        _validate_runtime_flags(flags, require_all_cells=True)
        score_payload = build_score_payload(
            identity=identity, input_payload=input_payload, evidence=evidence, flags=flags,
            device_attestation=backend.device_attestation(),
            upstream_formal_preflight_device=backend.upstream_formal_preflight_device(),
            posterior_final4_compatibility=backend.posterior_final4_compatibility(),
        )
        validate_score_payload(score_payload, identity=identity, input_payload=input_payload)
        score_body = _json(score_payload)
        score_sha256 = _digest(score_body)
        terminal_payload = _terminal_payload(
            identity=identity, attempt_sha256=attempt_sha256, input_authority_sha256=input_authority_sha256,
            score_sha256=score_sha256, final_closure=final_closure,
        )
        validate_terminal_payload(terminal_payload, identity=identity, score_sha256=score_sha256)
        terminal_body = _json(terminal_payload)

        def validate_group(hashes: Mapping[str, str]) -> None:
            if hashes.get("score.json") != score_sha256:
                raise QuickScreenError("quick group score digest drift")
            score = artifact.reload_json("score.json", score_sha256)
            terminal = artifact.reload_json("terminal.json", hashes["terminal.json"])
            validate_score_payload(score, identity=identity, input_payload=input_payload)
            validate_terminal_payload(terminal, identity=identity, score_sha256=score_sha256)

        artifact.publish_group({"score.json": score_body, "terminal.json": terminal_body}, post_publish=validate_group)
        terminal_published = True
        if artifact.has_name("failure.json"):
            raise QuickScreenError("quick success terminal coexists with failure")
        return artifact.reload_json("terminal.json", _digest(terminal_body))
    except BaseException as error:
        if attempt_sha256 is not None and not terminal_published:
            try:
                _publish_failure(
                    artifact, identity=identity, stage=stage, flags=flags, attempt_sha256=attempt_sha256,
                    input_authority_sha256=input_authority_sha256, error=error,
                )
            except BaseException:
                pass
        raise
    finally:
        backend.close()


def dry_plan() -> dict[str, object]:
    """Pure static plan used by the public zero-argument CLI."""
    selected = selected_asset_payload()
    return {
        "cell": CELL,
        "phase": PHASE,
        "classification": CLASSIFICATION,
        "status": "DRY_ONLY__NO_NETWORK_NO_TORCH_NO_NWB_NO_CHECKPOINT_NO_CUDA_NO_WRITE",
        "workorder": {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256},
        "selected_indices": {WITHIN: list(WITHIN_INDICES), EXTERNAL: list(EXTERNAL_INDICES)},
        "selected_assets": selected,
        "selected_nwb_transfer_bytes": selected_transfer_bytes(),
        "matrix": [item.payload() for item in quick_screen_matrix()],
        "metric": dict(METRIC),
        "boundaries": dict(BOUNDARIES),
        "predecessors": BasePredecessorEvidence().payload(),
        "v1_failed_predecessor": V1FailedPredecessorEvidence().payload(),
        "v2_failed_predecessor": V2FailedPredecessorEvidence().payload(),
        "posterior_final4_compatibility": posterior_final4_compatibility_payload(),
        "remote": {
            "stage_root": REMOTE_STAGE_ROOT,
            "score_root_relative": REMOTE_SCORE_ROOT_RELATIVE,
            "engineering_only": True,
            "requires_in_process_root_capability": True,
            "engineering_device_authority": remote_engineering_device_payload(),
        },
    }


def execute_authorized(
    *, root: Path | None = None, identity: QuickScreenIdentity | None = None, execute: bool = False,
    root_reviewed: bool = False, execution_capability: object | None = None,
) -> None:
    """Public fail-closed boundary; physical launch is deliberately private."""
    if execution_capability is None or root is None or identity is None:
        raise QuickScreenError("quick screen remains dry until an in-process root-reviewed capability exists")
    if execute is not True or root_reviewed is not True:
        raise QuickScreenError("quick screen requires both explicit reviewed flags")
    _require_capability(execution_capability, identity=identity)
    raise QuickScreenError("physical quick-screen launcher is available only through the reviewed remote module")
