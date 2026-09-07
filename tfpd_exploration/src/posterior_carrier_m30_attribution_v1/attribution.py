"""Dry contract and receipt lifecycle for M30 Posterior Carrier attribution.

This module intentionally imports only the Python standard library.  The
physical model/parser implementation lives in :mod:`physical` and is never
loaded by the public zero-argument CLI.  Its sole purpose is to distinguish
five *predeclared* M30 consumer/carrier combinations after the completed V3
quick screen was strongly negative.  It produces descriptive engineering
evidence, not a formal score or a scientific gate.
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
from typing import Any, Mapping, Protocol, Sequence


class AttributionError(RuntimeError):
    """Raised when the non-governing attribution contract would drift."""


CELL = "POSTERIOR_CARRIER_M30_ATTRIBUTION_V1"
PHASE = "POSTERIOR_CARRIER_M30_ATTRIBUTION_QUICK_SCREEN_V1"
CLASSIFICATION = "NON_GOVERNING_ATTRIBUTION_QUICK_SCREEN"
SCHEMA = "posterior_carrier_m30_attribution_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_POSTERIOR_CARRIER_M30_ATTRIBUTION_20260822.md"

V3_STAGE_ROOT = "/home/xinyuan/Work_host/posterior_carrier_quick_screen_stage_v3"
V3_RESULT_RELATIVE = "tfpd_exploration/results/posterior_carrier_quick_screen_v3"
V3_ATTEMPT_SHA256 = "fd85e83bbc9180c9d7ca27720de3063c76be20cf6889797570234cfd82f413d2"
V3_INPUT_AUTHORITY_SHA256 = "bad14dadec2c4e525c4cffbaf689b75ce1cebfa9bccf27bc022256574176386e"
V3_SCORE_SHA256 = "07bb30a018ad33a63d89ed3cd4fcc0293d7150b59262f0db84ae9ae5ad6183b5"
V3_TERMINAL_SHA256 = "51103a5703ff6933362f11913afd379c3b580c59abac62630194bf2a44626949"
V3_CLOSURE_SHA256 = "3b56ec450374e0b67d13fbcb8f68d00463b1bba837a3cd57990f3fb2fa4756cd"

# Exact canonical JSON digests of the four original V3 M30 cell bodies.  They
# were independently descriptor-read from the immutable V3 score receipt and
# recomputed with this route's canonical ``_json`` domain.  A SHA-shaped value
# is not enough: A/E may only be replayed if their copied evidence recreates
# precisely these original bodies.
V3_REUSED_M30_CELL_SHA256S = {
    "within:sealed_cell_d_ols_point": "1bad74aae7e4e871ed4de0eaf293c1caa4c4fc396adbcefcbafa31130ae5621d",
    "within:posterior_swa_posterior_mean_posterior_normalizer_nonuniform_credibility": "dafdd37de50d23b964bd2b75a2db8a061808334d80955dab93ae32727ff91cbc",
    "external:sealed_cell_d_ols_point": "f619a29d2aec9ac44531263a0e0b1926dc931884e18b0a509018e501d180431f",
    "external:posterior_swa_posterior_mean_posterior_normalizer_nonuniform_credibility": "275732f0cc3b384d8d14bc5e11aac77bde214da17753ca113d55928fc0c9f49c",
}

# The current route is a non-governing remote engineering route and therefore
# retains V3's honest Torch-only (not NVML-shaped) device authority verbatim.
# This is deliberately a literal contract: no caller mapping, rounded memory,
# or substituted local GPU may enter a score receipt.
V3_TORCH_ONLY_DEVICE_AUTHORITY = {
    "schema": "posterior_carrier_quick_screen_remote_torch_only_device_v3",
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


def _remote_stage_payload() -> dict[str, object]:
    """The attribution route may run only on V3's exact engineering device.

    Keeping this as a single literal builder prevents the identity, durable
    score body, and physical attestation from accidentally describing three
    slightly different 5070Ti environments.
    """
    return {
        "stage_root": REMOTE_STAGE_ROOT,
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "requires_in_process_root_capability": True,
        "device_contract": dict(V3_TORCH_ONLY_DEVICE_AUTHORITY),
    }

REMOTE_STAGE_ROOT = "/home/xinyuan/Work_host/posterior_carrier_m30_attribution_stage_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_m30_attribution_v1"

WITHIN = "within"
EXTERNAL = "external"
SURFACES = (WITHIN, EXTERNAL)
BUDGET = 30
WITHIN_ROSTER = (
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151112",
)
EXTERNAL_ROSTER = (
    "sub-M_ses-CO-20140307",
    "sub-M_ses-CO-20150611",
    "sub-M_ses-CO-20150626",
)
SELECTED_ROSTERS = {WITHIN: WITHIN_ROSTER, EXTERNAL: EXTERNAL_ROSTER}

SEALED_OLS_POINT = "sealed_cell_d_ols_point"
SEALED_POSTERIOR_MEAN = "sealed_cell_d_posterior_mean_sealed_normalizer_no_credibility"
POSTERIOR_OLS_UNIFORM = "posterior_swa_ols_point_posterior_normalizer_uniform_credibility"
POSTERIOR_MEAN_UNIFORM = "posterior_swa_posterior_mean_posterior_normalizer_uniform_credibility"
POSTERIOR_MEAN_PRECISION = "posterior_swa_posterior_mean_posterior_normalizer_nonuniform_credibility"
SYSTEMS = (
    SEALED_OLS_POINT,
    SEALED_POSTERIOR_MEAN,
    POSTERIOR_OLS_UNIFORM,
    POSTERIOR_MEAN_UNIFORM,
    POSTERIOR_MEAN_PRECISION,
)
V3_REUSED_SYSTEMS = frozenset({SEALED_OLS_POINT, POSTERIOR_MEAN_PRECISION})
NEW_FORWARD_SYSTEMS = frozenset(set(SYSTEMS) - set(V3_REUSED_SYSTEMS))

METRIC = {
    "estimator": "tfpd_lane.matched_scorer.session_r2",
    "engineering_implementation": "posterior_carrier_quick_screen_v3.manual_variance_weighted_two_coordinate_r2",
    "query": "last_bin_only_of_each_valid_50_bin_window",
    "equal_weight_per_session": True,
    "behavior_coordinates": 2,
    "window_bins": 50,
}
BOUNDARIES = {
    "classification": CLASSIFICATION,
    "m30_only": True,
    "m10_scored": False,
    "zero_control_scored": False,
    "wrong_pair_control_scored": False,
    "h1_opened": False,
    "formal_opened": False,
    "full_window_headline": False,
    "checkpoint_selection": False,
    "normalizer_refit": False,
    "target_sampling": False,
    "target_optimizer_steps": 0,
    "target_backward_calls": 0,
    "target_update_calls": 0,
}

# Explicitly list every code/authority dependency that can be imported by the
# deferred composition.  There is no glob expansion at planning or execution.
V3_CLOSURE_PATHS = (
    "tfpd_exploration/docs/WORKORDER_POSTERIOR_CARRIER_QUICK_SCREEN_20260822.md",
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/quick_screen.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/physical.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/remote_stage.py",
    "tfpd_exploration/scripts/run_posterior_carrier_quick_screen.py",
    "tfpd_exploration/tests/test_posterior_carrier_quick_screen_v1.py",
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
NEW_CLOSURE_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/posterior_carrier_m30_attribution_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_m30_attribution_v1/attribution.py",
    "tfpd_exploration/src/posterior_carrier_m30_attribution_v1/physical.py",
    "tfpd_exploration/src/posterior_carrier_m30_attribution_v1/remote_stage.py",
    "tfpd_exploration/scripts/run_posterior_carrier_m30_attribution.py",
    "tfpd_exploration/tests/test_posterior_carrier_m30_attribution_v1.py",
)
IMPLEMENTATION_CLOSURE = tuple(dict.fromkeys((*V3_CLOSURE_PATHS, *NEW_CLOSURE_PATHS)))


def _require(value: bool, message: str) -> None:
    if not value:
        raise AttributionError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AttributionError(f"{label} must be a SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as error:
        raise AttributionError(f"{label} must be SHA-256 hex") from error
    return value


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise AttributionError(f"{label} must be finite")
    return float(value)


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise AttributionError(f"{label} must be a nonempty relative path")
    candidate = Path(value)
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise AttributionError(f"{label} contains a forbidden path component")
    return value


def _copy_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AttributionError(f"{label} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        if not isinstance(self.sha256_by_path, Mapping) or set(self.sha256_by_path) != set(IMPLEMENTATION_CLOSURE):
            raise AttributionError("attribution implementation closure topology drift")
        hashes = {path: _sha(self.sha256_by_path[path], f"closure SHA {path}") for path in IMPLEMENTATION_CLOSURE}
        body = {"paths": list(IMPLEMENTATION_CLOSURE), "sha256_by_path": hashes}
        return {**body, "closure_sha256": _digest(_json(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    """Descriptor-hash only explicit code and immutable metadata leaves."""
    base = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_CLOSURE:
        path = base / relative
        try:
            before = os.lstat(path)
        except OSError as error:
            raise AttributionError(f"cannot lstat attribution closure leaf: {relative}") from error
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise AttributionError(f"attribution closure leaf must be regular non-symlink: {relative}")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
                raise AttributionError("attribution closure descriptor identity drift")
            chunks: list[bytes] = []
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                chunks.append(block)
            body = b"".join(chunks)
        finally:
            os.close(fd)
        after = os.lstat(path)
        if (after.st_dev, after.st_ino, after.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise AttributionError("attribution closure leaf changed during read")
        hashes[relative] = _digest(body)
    return ImplementationClosure(hashes)


@dataclass(frozen=True)
class V3Evidence:
    stage_root: str = V3_STAGE_ROOT
    result_relative: str = V3_RESULT_RELATIVE
    closure_sha256: str = V3_CLOSURE_SHA256
    attempt_sha256: str = V3_ATTEMPT_SHA256
    input_authority_sha256: str = V3_INPUT_AUTHORITY_SHA256
    score_sha256: str = V3_SCORE_SHA256
    terminal_sha256: str = V3_TERMINAL_SHA256

    def payload(self) -> dict[str, object]:
        if self.stage_root != V3_STAGE_ROOT or self.result_relative != V3_RESULT_RELATIVE:
            raise AttributionError("V3 predecessor root literal drift")
        values = {
            "stage_root": self.stage_root,
            "result_relative": self.result_relative,
            "closure_sha256": self.closure_sha256,
            "attempt_sha256": self.attempt_sha256,
            "input_authority_sha256": self.input_authority_sha256,
            "score_sha256": self.score_sha256,
            "terminal_sha256": self.terminal_sha256,
            "reused_m30_cell_sha256s": dict(V3_REUSED_M30_CELL_SHA256S),
            "expected_result_topology": [
                "attempt.json", "attempt.json.sha256", "input_authority.json", "input_authority.json.sha256",
                "score.json", "score.json.sha256", "terminal.json", "terminal.json.sha256",
            ],
            "failure_must_be_absent": True,
        }
        for key in ("closure_sha256", "attempt_sha256", "input_authority_sha256", "score_sha256", "terminal_sha256"):
            _sha(values[key], f"V3 {key}")
        return values


def validate_v3_evidence_payload(value: Mapping[str, object]) -> dict[str, object]:
    expected = V3Evidence().payload()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise AttributionError("V3 immutable predecessor evidence drift")
    return expected


def validate_v3_torch_only_device_payload(value: object) -> dict[str, object]:
    """Require V3's full Torch-only engineering runtime authority exactly."""
    if not isinstance(value, Mapping) or dict(value) != V3_TORCH_ONLY_DEVICE_AUTHORITY:
        raise AttributionError("attribution V3 Torch-only device authority drift")
    return dict(V3_TORCH_ONLY_DEVICE_AUTHORITY)


def validate_v3_validation_payload(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the small post-replay proof emitted by the physical backend.

    The static contract freezes the independently recomputed original V3
    cell-body digests.  The physical backend must derive the completed V3 rows
    descriptor-safely and exact-match those literals before it may expose any
    A/E replay evidence.
    """
    expected_keys = {
        "schema", "v3_predecessor", "v3_identity_closure_sha256", "v3_input_replayed_exactly",
        "v3_score_terminal_chain_valid", "reused_v3_m30_cell_sha256s",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise AttributionError("V3 replay-validation schema drift")
    if (
        value.get("schema") != "posterior_carrier_m30_attribution_v3_replay_validation_v1"
        or validate_v3_evidence_payload(value.get("v3_predecessor")) != V3Evidence().payload()
        or value.get("v3_identity_closure_sha256") != V3_CLOSURE_SHA256
        or value.get("v3_input_replayed_exactly") is not True
        or value.get("v3_score_terminal_chain_valid") is not True
    ):
        raise AttributionError("V3 replay-validation immutable binding drift")
    rows = value.get("reused_v3_m30_cell_sha256s")
    expected_rows = {f"{surface}:{system}" for surface in SURFACES for system in V3_REUSED_SYSTEMS}
    if not isinstance(rows, Mapping) or set(rows) != expected_rows:
        raise AttributionError("V3 replay-validation reusable-cell topology drift")
    if dict(rows) != V3_REUSED_M30_CELL_SHA256S:
        raise AttributionError("V3 replay-validation reusable-cell literal digest drift")
    checked = {key: _sha(rows[key], f"V3 reusable M30 cell {key}") for key in sorted(expected_rows)}
    return {
        "schema": "posterior_carrier_m30_attribution_v3_replay_validation_v1",
        "v3_predecessor": V3Evidence().payload(),
        "v3_identity_closure_sha256": V3_CLOSURE_SHA256,
        "v3_input_replayed_exactly": True,
        "v3_score_terminal_chain_valid": True,
        "reused_v3_m30_cell_sha256s": dict(V3_REUSED_M30_CELL_SHA256S),
    }


@dataclass(frozen=True)
class AttributionCell:
    surface: str
    system: str
    budget: int = BUDGET

    def payload(self) -> dict[str, object]:
        if self.surface not in SURFACES or self.system not in SYSTEMS or self.budget != BUDGET:
            raise AttributionError("attribution score-cell topology drift")
        config = _SYSTEM_CONFIG[self.system]
        return {"surface": self.surface, "system": self.system, "budget": BUDGET, **config}


_SYSTEM_CONFIG: dict[str, dict[str, object]] = {
    SEALED_OLS_POINT: {
        "consumer": "sealed_cell_d", "carrier_estimator": "ols_point", "normalizer": "sealed_cell_d_ordinary_point",
        "credibility": "none__sealed_decoder_has_no_posterior_attention_bias", "evidence_origin": "V3_REUSED_VALIDATED",
    },
    SEALED_POSTERIOR_MEAN: {
        "consumer": "sealed_cell_d", "carrier_estimator": "posterior_mean", "normalizer": "sealed_cell_d_ordinary_point",
        "credibility": "none__sealed_decoder_has_no_posterior_attention_bias", "evidence_origin": "NEW_FORWARD",
    },
    POSTERIOR_OLS_UNIFORM: {
        "consumer": "posterior_full_swa", "carrier_estimator": "ols_point", "normalizer": "posterior_distribution",
        "credibility": "uniform__exact_centered_logit_cancellation", "evidence_origin": "NEW_FORWARD",
    },
    POSTERIOR_MEAN_UNIFORM: {
        "consumer": "posterior_full_swa", "carrier_estimator": "posterior_mean", "normalizer": "posterior_distribution",
        "credibility": "uniform__exact_centered_logit_cancellation", "evidence_origin": "NEW_FORWARD",
    },
    POSTERIOR_MEAN_PRECISION: {
        "consumer": "posterior_full_swa", "carrier_estimator": "posterior_mean", "normalizer": "posterior_distribution",
        "credibility": "nonuniform__posterior_precision_logit_bias", "evidence_origin": "V3_REUSED_VALIDATED",
    },
}


def attribution_matrix() -> tuple[AttributionCell, ...]:
    return tuple(AttributionCell(surface, system) for surface in SURFACES for system in SYSTEMS)


@dataclass(frozen=True)
class AttributionIdentity:
    closure: ImplementationClosure
    v3: V3Evidence = field(default_factory=V3Evidence)

    def payload(self) -> dict[str, object]:
        closure = self.closure.payload()
        return {
            "schema": SCHEMA,
            "cell": CELL,
            "phase": PHASE,
            "classification": CLASSIFICATION,
            "implementation_closure": closure,
            "v3_predecessor": self.v3.payload(),
            "selected_rosters": {surface: list(SELECTED_ROSTERS[surface]) for surface in SURFACES},
            "metric": dict(METRIC),
            "matrix": [cell.payload() for cell in attribution_matrix()],
            "boundaries": dict(BOUNDARIES),
            "remote_stage": _remote_stage_payload(),
        }


def validate_identity(value: AttributionIdentity | Mapping[str, object]) -> dict[str, object]:
    payload = value.payload() if isinstance(value, AttributionIdentity) else _copy_mapping(value, "attribution identity")
    expected_keys = {
        "schema", "cell", "phase", "classification", "implementation_closure", "v3_predecessor",
        "selected_rosters", "metric", "matrix", "boundaries", "remote_stage",
    }
    if set(payload) != expected_keys:
        raise AttributionError("attribution identity schema drift")
    if (
        payload.get("schema") != SCHEMA or payload.get("cell") != CELL or payload.get("phase") != PHASE
        or payload.get("classification") != CLASSIFICATION or payload.get("selected_rosters")
        != {surface: list(SELECTED_ROSTERS[surface]) for surface in SURFACES}
        or payload.get("metric") != METRIC or payload.get("boundaries") != BOUNDARIES
        or payload.get("matrix") != [cell.payload() for cell in attribution_matrix()]
        or payload.get("remote_stage") != _remote_stage_payload()
    ):
        raise AttributionError("attribution identity scientific boundary drift")
    closure = payload.get("implementation_closure")
    if not isinstance(closure, Mapping) or set(closure) != {"paths", "sha256_by_path", "closure_sha256"}:
        raise AttributionError("attribution identity closure schema drift")
    if tuple(closure["paths"]) != IMPLEMENTATION_CLOSURE or set(closure["sha256_by_path"]) != set(IMPLEMENTATION_CLOSURE):
        raise AttributionError("attribution identity closure topology drift")
    hashes = {path: _sha(closure["sha256_by_path"][path], f"attribution closure {path}") for path in IMPLEMENTATION_CLOSURE}
    rebuilt = {"paths": list(IMPLEMENTATION_CLOSURE), "sha256_by_path": hashes}
    if closure["closure_sha256"] != _digest(_json(rebuilt)):
        raise AttributionError("attribution identity closure digest drift")
    validate_v3_evidence_payload(payload["v3_predecessor"])
    return {**payload, "implementation_closure": {**rebuilt, "closure_sha256": closure["closure_sha256"]}}


def _record_from_payload(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "surface", "session", "n_windows", "neural_sha256", "calibration_m30_sha256", "last_bin_target_sha256",
        "last_bin_valid_mask_sha256", "last_bin_valid_count", "prefix_row_ids_sha256s", "point_carrier_sha256s",
        "posterior_carrier_sha256s", "materialization_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise AttributionError("V3 input record schema drift")
    surface, session = value.get("surface"), value.get("session")
    if surface not in SURFACES or session not in SELECTED_ROSTERS[surface]:
        raise AttributionError("V3 input record selected roster drift")
    if type(value.get("n_windows")) is not int or value["n_windows"] <= 0 or value.get("last_bin_valid_count") != value["n_windows"]:
        raise AttributionError("V3 input record governing count drift")
    for key in ("neural_sha256", "calibration_m30_sha256", "last_bin_target_sha256", "last_bin_valid_mask_sha256", "materialization_sha256"):
        _sha(value.get(key), f"V3 input record {key}")
    for key in ("prefix_row_ids_sha256s", "point_carrier_sha256s", "posterior_carrier_sha256s"):
        mapping = value.get(key)
        if not isinstance(mapping, Mapping) or set(mapping) != {"30", "4"}:
            raise AttributionError(f"V3 input record {key} budget topology drift")
        for digest in mapping.values():
            _sha(digest, f"V3 input record {key}")
    return dict(value)


@dataclass(frozen=True)
class InputReplay:
    """A descriptor-validated replay of V3's one shared materialization pass."""

    records: tuple[Mapping[str, object], ...]
    v3_input_authority_sha256: str = V3_INPUT_AUTHORITY_SHA256
    v3_score_sha256: str = V3_SCORE_SHA256
    v3_terminal_sha256: str = V3_TERMINAL_SHA256

    def payload(self, *, identity: AttributionIdentity) -> dict[str, object]:
        checked_identity = validate_identity(identity)
        rows = [_record_from_payload(item) for item in self.records]
        expected = tuple((surface, session) for surface in SURFACES for session in SELECTED_ROSTERS[surface])
        if tuple((row["surface"], row["session"]) for row in rows) != expected:
            raise AttributionError("attribution V3 input replay roster/order drift")
        if (
            _sha(self.v3_input_authority_sha256, "V3 input authority") != V3_INPUT_AUTHORITY_SHA256
            or _sha(self.v3_score_sha256, "V3 score") != V3_SCORE_SHA256
            or _sha(self.v3_terminal_sha256, "V3 terminal") != V3_TERMINAL_SHA256
        ):
            raise AttributionError("attribution V3 input replay predecessor binding drift")
        return {
            "schema": "posterior_carrier_m30_attribution_input_replay_v1",
            "identity": checked_identity,
            "v3_input_authority_sha256": self.v3_input_authority_sha256,
            "v3_score_sha256": self.v3_score_sha256,
            "v3_terminal_sha256": self.v3_terminal_sha256,
            "records": rows,
            "shared_input_pass": True,
            "v3_input_records_exactly_replayed": True,
            "normalizer_refit": False,
            "target_sampling": False,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "formal_opened": False,
            "h1_opened": False,
        }


def validate_input_replay_payload(value: Mapping[str, object], *, identity: AttributionIdentity) -> dict[str, object]:
    expected = {
        "schema", "identity", "v3_input_authority_sha256", "v3_score_sha256", "v3_terminal_sha256", "records",
        "shared_input_pass", "v3_input_records_exactly_replayed", "normalizer_refit", "target_sampling",
        "target_optimizer_steps", "target_backward_calls", "target_update_calls", "formal_opened", "h1_opened",
    }
    if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value.get("records"), list):
        raise AttributionError("attribution input-replay schema drift")
    rebuilt = InputReplay(
        records=tuple(value["records"]), v3_input_authority_sha256=value["v3_input_authority_sha256"],
        v3_score_sha256=value["v3_score_sha256"], v3_terminal_sha256=value["v3_terminal_sha256"],
    ).payload(identity=identity)
    if rebuilt != dict(value):
        raise AttributionError("attribution input-replay exact reconstruction drift")
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
            raise AttributionError("attribution session-score schema drift")
        return {
            "session": self.session, "n_windows": self.n_windows, "r2": _finite(self.r2, "attribution R2"),
            "prediction_sha256": _sha(self.prediction_sha256, "attribution prediction SHA"),
            "input_record_sha256": _sha(self.input_record_sha256, "attribution input-record SHA"),
        }


def _session_score_from_payload(value: Mapping[str, object]) -> SessionScore:
    if not isinstance(value, Mapping) or set(value) != {"session", "n_windows", "r2", "prediction_sha256", "input_record_sha256"}:
        raise AttributionError("attribution session-score payload drift")
    return SessionScore(
        session=value["session"], n_windows=value["n_windows"], r2=value["r2"],
        prediction_sha256=value["prediction_sha256"], input_record_sha256=value["input_record_sha256"],
    )


def _model_system(system: str) -> str:
    return "sealed_cell_d_checkpoint" if system in {SEALED_OLS_POINT, SEALED_POSTERIOR_MEAN} else "posterior_carrier_full_swa"


def _evidence_origin(system: str) -> str:
    return "V3_REUSED_VALIDATED" if system in V3_REUSED_SYSTEMS else "NEW_FORWARD"


def _v3_reused_key(cell: AttributionCell) -> str:
    if cell.system not in V3_REUSED_SYSTEMS:
        raise AttributionError("only A/E have a V3 reusable-cell key")
    return f"{cell.surface}:{cell.system}"


def _reconstruct_original_v3_reused_cell(value: Mapping[str, object]) -> dict[str, object]:
    """Recover the exact *V3* cell JSON body from copied A/E facts.

    The successor's CellEvidence has additional carrier-digest fields and a
    new input-replay body SHA.  Neither existed in V3.  This conversion is
    intentionally one-way: only the V3 score-cell mode, original V3 input
    authority digest, copied session rows, and model/state facts enter the
    original canonical JSON domain.  It makes a local A/E row acceptable only
    when it recreates the independently frozen V3 row byte-for-byte.
    """
    if not isinstance(value, Mapping):
        raise AttributionError("V3 reusable evidence must be a mapping")
    cell = _cell_from_payload(value.get("cell"))
    if cell.system not in V3_REUSED_SYSTEMS:
        raise AttributionError("new attribution forward has no V3 original cell")
    sessions = value.get("sessions")
    if not isinstance(sessions, list):
        raise AttributionError("V3 reusable evidence sessions must be a list")
    mode = "sealed_cell_d_ols_point" if cell.system == SEALED_OLS_POINT else "posterior_full_swa_aligned"
    expected = {
        "cell": {"surface": cell.surface, "mode": mode, "budget": BUDGET},
        "model_system": _model_system(cell.system),
        "model_swa_sha256": value.get("model_swa_sha256"),
        "sessions": [dict(row) if isinstance(row, Mapping) else row for row in sessions],
        # V3's original CellEvidence binds its completed input authority,
        # while this route separately binds a replay body digest.
        "input_authority_sha256": V3_INPUT_AUTHORITY_SHA256,
        "model_state_before_sha256": value.get("model_state_before_sha256"),
        "model_state_after_sha256": value.get("model_state_after_sha256"),
        "eval_mode": True,
        "dropout_disabled": True,
        "gradients_none": True,
        "finite_outputs": True,
        "repeated_fixed_batch_bitwise_equal": True,
        "b3s_m30_recomputed": True,
        "no_target_sampling": True,
    }
    # Exact key topology is part of the digest domain.  Validate every SHA
    # before hashing so a malformed value cannot masquerade as source proof.
    _sha(expected["model_swa_sha256"], "reconstructed V3 model SWA SHA")
    _sha(expected["input_authority_sha256"], "reconstructed V3 input authority SHA")
    _sha(expected["model_state_before_sha256"], "reconstructed V3 state-before SHA")
    _sha(expected["model_state_after_sha256"], "reconstructed V3 state-after SHA")
    for index, row in enumerate(expected["sessions"]):
        if not isinstance(row, Mapping):
            raise AttributionError(f"reconstructed V3 session {index} is not a mapping")
        # Reuse the public session parser so the exact original V3 row has no
        # hidden, caller-defined fields.
        expected["sessions"][index] = _session_score_from_payload(row).payload()
    return expected


def _validate_v3_reused_cell_reconstruction(value: Mapping[str, object]) -> str:
    """Require exact V3 original-body replay for one A/E attribution row."""
    cell = _cell_from_payload(value.get("cell"))
    key = _v3_reused_key(cell)
    supplied = _sha(value.get("v3_reused_cell_sha256"), "V3 reused cell SHA")
    expected = V3_REUSED_M30_CELL_SHA256S[key]
    if supplied != expected:
        raise AttributionError("V3-reused attribution cell literal SHA drift")
    reconstructed = _reconstruct_original_v3_reused_cell(value)
    if _digest(_json(reconstructed)) != expected:
        raise AttributionError("V3-reused attribution cell original-body replay drift")
    return expected


@dataclass(frozen=True)
class CellEvidence:
    cell: AttributionCell
    model_swa_sha256: str
    sessions: tuple[SessionScore, ...]
    input_replay_sha256: str
    model_state_before_sha256: str
    model_state_after_sha256: str
    carrier_raw_sha256s: Mapping[str, str]
    carrier_normalized_sha256s: Mapping[str, str]
    credibility_sha256s: Mapping[str, str]
    raw_before_normalization_verified: bool
    eval_mode: bool
    dropout_disabled: bool
    gradients_none: bool
    finite_outputs: bool
    repeated_fixed_batch_bitwise_equal: bool
    b3s_m30_recomputed: bool
    no_target_sampling: bool
    evidence_origin: str
    v3_reused_cell_sha256: str | None = None

    def payload(self, *, identity: AttributionIdentity, input_payload: Mapping[str, object]) -> dict[str, object]:
        cell = self.cell.payload()
        config = _SYSTEM_CONFIG[self.cell.system]
        expected_sessions = SELECTED_ROSTERS[self.cell.surface]
        rows = [item.payload() for item in self.sessions]
        if tuple(item["session"] for item in rows) != expected_sessions:
            raise AttributionError("attribution cell session roster/order drift")
        records = {item["session"]: item for item in input_payload["records"] if item["surface"] == self.cell.surface}
        if tuple(records) != expected_sessions:
            raise AttributionError("attribution cell input replay roster drift")
        if any(
            item["n_windows"] != records[item["session"]]["n_windows"]
            or item["input_record_sha256"] != _digest(_json(records[item["session"]]))
            for item in rows
        ):
            raise AttributionError("attribution cell same-input proof drift")
        expected_swa = (
            "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
            if self.cell.system in {SEALED_OLS_POINT, SEALED_POSTERIOR_MEAN}
            else "def27d8edffc0c6ed17dee80292dd1c1b628ba1d454dc214278caee814cf40fa"
        )
        if _sha(self.model_swa_sha256, "attribution model SWA") != expected_swa:
            raise AttributionError("attribution cell model/SWA binding drift")
        expected_origin = _evidence_origin(self.cell.system)
        if self.evidence_origin != expected_origin:
            raise AttributionError("attribution evidence origin drift")
        if expected_origin == "V3_REUSED_VALIDATED":
            if self.v3_reused_cell_sha256 is None:
                raise AttributionError("V3-reused attribution cell lacks V3 cell SHA")
        elif self.v3_reused_cell_sha256 is not None:
            raise AttributionError("new attribution forward must not pretend to be V3 reused")
        for label, mapping in (
            ("raw carrier", self.carrier_raw_sha256s),
            ("normalized carrier", self.carrier_normalized_sha256s),
            ("credibility", self.credibility_sha256s),
        ):
            if not isinstance(mapping, Mapping) or tuple(mapping) != expected_sessions:
                raise AttributionError(f"attribution {label} session topology drift")
            for digest in mapping.values():
                _sha(digest, f"attribution {label} SHA")
        if (
            self.input_replay_sha256 != _digest(_json(input_payload))
            or self.model_state_before_sha256 != self.model_state_after_sha256
            or self.raw_before_normalization_verified is not True
            or self.eval_mode is not True or self.dropout_disabled is not True or self.gradients_none is not True
            or self.finite_outputs is not True or self.repeated_fixed_batch_bitwise_equal is not True
            or self.b3s_m30_recomputed is not True or self.no_target_sampling is not True
        ):
            raise AttributionError("attribution cell physical invariants drift")
        result = {
            "cell": cell,
            "model_system": _model_system(self.cell.system),
            "model_swa_sha256": expected_swa,
            "sessions": rows,
            "input_replay_sha256": self.input_replay_sha256,
            "model_state_before_sha256": _sha(self.model_state_before_sha256, "attribution state before"),
            "model_state_after_sha256": _sha(self.model_state_after_sha256, "attribution state after"),
            "carrier_raw_sha256s": dict(self.carrier_raw_sha256s),
            "carrier_normalized_sha256s": dict(self.carrier_normalized_sha256s),
            "credibility_sha256s": dict(self.credibility_sha256s),
            "raw_before_normalization_verified": True,
            "eval_mode": True, "dropout_disabled": True, "gradients_none": True, "finite_outputs": True,
            "repeated_fixed_batch_bitwise_equal": True, "b3s_m30_recomputed": True, "no_target_sampling": True,
            "evidence_origin": expected_origin, "v3_reused_cell_sha256": self.v3_reused_cell_sha256,
        }
        if expected_origin == "V3_REUSED_VALIDATED":
            _validate_v3_reused_cell_reconstruction(result)
        return result


def _cell_from_payload(value: Mapping[str, object]) -> AttributionCell:
    expected = {"surface", "system", "budget", "consumer", "carrier_estimator", "normalizer", "credibility", "evidence_origin"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise AttributionError("attribution cell payload schema drift")
    cell = AttributionCell(surface=value["surface"], system=value["system"], budget=value["budget"])
    if cell.payload() != dict(value):
        raise AttributionError("attribution cell payload exact reconstruction drift")
    return cell


def validate_cell_evidence_payload(value: Mapping[str, object], *, identity: AttributionIdentity,
                                   input_payload: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "cell", "model_system", "model_swa_sha256", "sessions", "input_replay_sha256", "model_state_before_sha256",
        "model_state_after_sha256", "carrier_raw_sha256s", "carrier_normalized_sha256s", "credibility_sha256s",
        "raw_before_normalization_verified", "eval_mode", "dropout_disabled", "gradients_none", "finite_outputs",
        "repeated_fixed_batch_bitwise_equal", "b3s_m30_recomputed", "no_target_sampling", "evidence_origin",
        "v3_reused_cell_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value.get("sessions"), list):
        raise AttributionError("attribution evidence schema drift")
    rebuilt = CellEvidence(
        cell=_cell_from_payload(value["cell"]), model_swa_sha256=value["model_swa_sha256"],
        sessions=tuple(_session_score_from_payload(item) for item in value["sessions"]),
        input_replay_sha256=value["input_replay_sha256"], model_state_before_sha256=value["model_state_before_sha256"],
        model_state_after_sha256=value["model_state_after_sha256"], carrier_raw_sha256s=value["carrier_raw_sha256s"],
        carrier_normalized_sha256s=value["carrier_normalized_sha256s"], credibility_sha256s=value["credibility_sha256s"],
        raw_before_normalization_verified=value["raw_before_normalization_verified"], eval_mode=value["eval_mode"],
        dropout_disabled=value["dropout_disabled"], gradients_none=value["gradients_none"], finite_outputs=value["finite_outputs"],
        repeated_fixed_batch_bitwise_equal=value["repeated_fixed_batch_bitwise_equal"],
        b3s_m30_recomputed=value["b3s_m30_recomputed"], no_target_sampling=value["no_target_sampling"],
        evidence_origin=value["evidence_origin"], v3_reused_cell_sha256=value["v3_reused_cell_sha256"],
    ).payload(identity=identity, input_payload=input_payload)
    if rebuilt != dict(value):
        raise AttributionError("attribution evidence exact reconstruction drift")
    return rebuilt


@dataclass
class RuntimeFlags:
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
    new_forward_cells: list[tuple[str, str, int]] = field(default_factory=list)
    v3_reused_cells: list[tuple[str, str, int]] = field(default_factory=list)

    def record_cell(self, cell: AttributionCell) -> None:
        target = self.v3_reused_cells if cell.system in V3_REUSED_SYSTEMS else self.new_forward_cells
        target.append((cell.surface, cell.system, cell.budget))

    def payload(self) -> dict[str, object]:
        return {
            "within_opened": self.within_opened, "external_opened": self.external_opened,
            "remote_initialized": self.remote_initialized, "target_optimizer_steps": self.target_optimizer_steps,
            "target_backward_calls": self.target_backward_calls, "target_update_calls": self.target_update_calls,
            "normalizer_refit": self.normalizer_refit, "target_sampling": self.target_sampling,
            "h1_opened": self.h1_opened, "formal_opened": self.formal_opened,
            "new_forward_cells": [list(item) for item in self.new_forward_cells],
            "v3_reused_cells": [list(item) for item in self.v3_reused_cells],
        }


def _validate_runtime_flags(flags: RuntimeFlags, *, complete: bool) -> None:
    if (
        flags.target_optimizer_steps != 0 or flags.target_backward_calls != 0 or flags.target_update_calls != 0
        or flags.normalizer_refit is not False or flags.target_sampling is not False
        or flags.h1_opened is not False or flags.formal_opened is not False
    ):
        raise AttributionError("attribution route crossed target/refit/formal boundary")
    if complete:
        expected_new = [(cell.surface, cell.system, cell.budget) for cell in attribution_matrix() if cell.system in NEW_FORWARD_SYSTEMS]
        expected_reused = [(cell.surface, cell.system, cell.budget) for cell in attribution_matrix() if cell.system in V3_REUSED_SYSTEMS]
        if (
            flags.within_opened is not True or flags.external_opened is not True or flags.remote_initialized is not True
            or flags.new_forward_cells != expected_new or flags.v3_reused_cells != expected_reused
        ):
            raise AttributionError("attribution route matrix/runtime topology drift")


def _median(values: Sequence[float], label: str) -> float:
    if not values:
        raise AttributionError(f"{label} has no values")
    ordered = sorted(_finite(item, label) for item in values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _surface_summary(rows: Sequence[Mapping[str, object]], label: str) -> dict[str, object]:
    values = [_finite(item.get("r2"), f"{label} R2") for item in rows]
    return {"equal_session_mean": sum(values) / len(values), "equal_session_median": _median(values, label)}


def _paired(left: Sequence[Mapping[str, object]], right: Sequence[Mapping[str, object]], label: str) -> dict[str, object]:
    sessions = [row.get("session") for row in left]
    if sessions != [row.get("session") for row in right] or any(not isinstance(name, str) for name in sessions):
        raise AttributionError(f"{label} paired session topology drift")
    deltas = [_finite(a.get("r2"), label) - _finite(b.get("r2"), label) for a, b in zip(left, right)]
    return {
        "label": label, "sessions": sessions, "deltas": deltas, "mean": sum(deltas) / len(deltas),
        "median": _median(deltas, label), "positive_count": sum(delta > 0.0 for delta in deltas),
        "n_sessions": len(deltas),
    }


def _direction(value: Mapping[str, object]) -> str:
    deltas = value["deltas"]
    if all(float(item) > 0.0 for item in deltas):
        return "CONSISTENT_POSITIVE"
    if all(float(item) < 0.0 for item in deltas):
        return "CONSISTENT_NEGATIVE"
    return "MIXED_OR_ZERO"


def _attribution_decision(*, sealed_estimator: Mapping[str, object], posterior_estimator: Mapping[str, object],
                          bias: Mapping[str, object], consumer_system: Mapping[str, object]) -> dict[str, object]:
    sealed_direction = _direction(sealed_estimator)
    posterior_direction = _direction(posterior_estimator)
    return {
        "classification": "DESCRIPTIVE_NON_GOVERNING_NON_INFERENTIAL",
        "bias_rescue": {
            "contrast": "posterior_mean_nonuniform_minus_uniform", "direction": _direction(bias),
            "decision": "PRESENT" if _direction(bias) == "CONSISTENT_POSITIVE" else "NOT_DEMONSTRATED",
        },
        "estimator_harm": {
            "sealed_consumer_direction": sealed_direction, "posterior_consumer_direction": posterior_direction,
            "decision": "CONSISTENT_HARM" if sealed_direction == posterior_direction == "CONSISTENT_NEGATIVE" else "NOT_CONSISTENTLY_DEMONSTRATED",
        },
        "training_weight_harm": {
            "contrast": "posterior_consumer_plus_frozen_posterior_normalizer_minus_sealed_consumer_plus_frozen_ordinary_normalizer",
            "direction": _direction(consumer_system),
            "decision": "HARM_DIRECTION" if _direction(consumer_system) == "CONSISTENT_NEGATIVE" else "NOT_CONSISTENTLY_DEMONSTRATED",
            "not_parameter_only": True,
        },
    }


def build_score_payload(*, identity: AttributionIdentity, input_payload: Mapping[str, object],
                        evidence: Sequence[Mapping[str, object]], flags: RuntimeFlags,
                        device_attestation: Mapping[str, object], v3_validation: Mapping[str, object]) -> dict[str, object]:
    checked_identity = validate_identity(identity)
    checked_input = validate_input_replay_payload(input_payload, identity=identity)
    if not isinstance(evidence, Sequence) or len(evidence) != len(attribution_matrix()):
        raise AttributionError("attribution evidence cardinality drift")
    checked = [validate_cell_evidence_payload(item, identity=identity, input_payload=checked_input) for item in evidence]
    if [row["cell"] for row in checked] != [cell.payload() for cell in attribution_matrix()]:
        raise AttributionError("attribution evidence matrix/order drift")
    _validate_runtime_flags(flags, complete=True)
    checked_device = validate_v3_torch_only_device_payload(device_attestation)
    checked_v3_validation = validate_v3_validation_payload(v3_validation)
    summaries: dict[str, object] = {}
    contrasts: dict[str, object] = {}
    decisions: dict[str, object] = {}
    for surface in SURFACES:
        rows = {row["cell"]["system"]: row for row in checked if row["cell"]["surface"] == surface}
        if set(rows) != set(SYSTEMS):
            raise AttributionError("attribution surface system topology drift")
        summaries[surface] = {system: _surface_summary(rows[system]["sessions"], f"{surface} {system}") for system in SYSTEMS}
        sealed_estimator = _paired(rows[SEALED_POSTERIOR_MEAN]["sessions"], rows[SEALED_OLS_POINT]["sessions"], f"{surface} B-minus-A")
        posterior_estimator = _paired(rows[POSTERIOR_MEAN_UNIFORM]["sessions"], rows[POSTERIOR_OLS_UNIFORM]["sessions"], f"{surface} D-minus-C")
        bias = _paired(rows[POSTERIOR_MEAN_PRECISION]["sessions"], rows[POSTERIOR_MEAN_UNIFORM]["sessions"], f"{surface} E-minus-D")
        consumer_system = _paired(rows[POSTERIOR_OLS_UNIFORM]["sessions"], rows[SEALED_OLS_POINT]["sessions"], f"{surface} C-minus-A")
        contrasts[surface] = {
            "posterior_mean_minus_ols_under_sealed_consumer": sealed_estimator,
            "posterior_mean_minus_ols_under_posterior_consumer": posterior_estimator,
            "nonuniform_precision_bias_minus_uniform": bias,
            "posterior_consumer_plus_normalizer_minus_sealed_consumer_plus_normalizer": consumer_system,
        }
        decisions[surface] = _attribution_decision(
            sealed_estimator=sealed_estimator, posterior_estimator=posterior_estimator,
            bias=bias, consumer_system=consumer_system,
        )
    return {
        "schema": "posterior_carrier_m30_attribution_score_v1",
        "status": "NON_GOVERNING_M30_ATTRIBUTION_COMPLETE__NO_FORMAL_VERDICT",
        "classification": CLASSIFICATION, "cell": CELL, "phase": PHASE, "identity": checked_identity,
        "input_replay_sha256": _digest(_json(checked_input)), "metric": dict(METRIC), "matrix": checked,
        "summaries": summaries, "paired_attribution": contrasts, "attribution_decision": decisions,
        "device_attestation": checked_device, "v3_validation": checked_v3_validation, "flags": flags.payload(),
        "no_governing_decision": True,
        "excluded": {"m10": True, "zero": True, "wrong_pair": True, "formal": True, "h1": True,
                     "target_update": True, "target_sampling": True, "normalizer_refit": True,
                     "checkpoint_selection": True, "full_window_headline": True},
    }


def validate_score_payload(value: Mapping[str, object], *, identity: AttributionIdentity,
                           input_payload: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema", "status", "classification", "cell", "phase", "identity", "input_replay_sha256", "metric", "matrix",
        "summaries", "paired_attribution", "attribution_decision", "device_attestation", "v3_validation", "flags",
        "no_governing_decision", "excluded",
    }
    if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value.get("matrix"), list):
        raise AttributionError("attribution score schema drift")
    if (
        value.get("schema") != "posterior_carrier_m30_attribution_score_v1"
        or value.get("status") != "NON_GOVERNING_M30_ATTRIBUTION_COMPLETE__NO_FORMAL_VERDICT"
        or value.get("classification") != CLASSIFICATION or value.get("cell") != CELL or value.get("phase") != PHASE
        or value.get("identity") != validate_identity(identity) or value.get("metric") != METRIC
        or value.get("input_replay_sha256") != _digest(_json(validate_input_replay_payload(input_payload, identity=identity)))
        or value.get("no_governing_decision") is not True
    ):
        raise AttributionError("attribution score identity/metric boundary drift")
    flags_value = value.get("flags")
    if not isinstance(flags_value, Mapping):
        raise AttributionError("attribution score flags missing")
    flags = RuntimeFlags(
        within_opened=flags_value.get("within_opened"), external_opened=flags_value.get("external_opened"),
        remote_initialized=flags_value.get("remote_initialized"), target_optimizer_steps=flags_value.get("target_optimizer_steps"),
        target_backward_calls=flags_value.get("target_backward_calls"), target_update_calls=flags_value.get("target_update_calls"),
        normalizer_refit=flags_value.get("normalizer_refit"), target_sampling=flags_value.get("target_sampling"),
        h1_opened=flags_value.get("h1_opened"), formal_opened=flags_value.get("formal_opened"),
        new_forward_cells=[tuple(item) for item in flags_value.get("new_forward_cells", [])],
        v3_reused_cells=[tuple(item) for item in flags_value.get("v3_reused_cells", [])],
    )
    if flags.payload() != dict(flags_value):
        raise AttributionError("attribution score flags reconstruction drift")
    rebuilt = build_score_payload(
        identity=identity, input_payload=input_payload, evidence=value["matrix"], flags=flags,
        device_attestation=value["device_attestation"], v3_validation=value["v3_validation"],
    )
    if rebuilt != dict(value):
        raise AttributionError("attribution score exact reconstruction drift")
    return rebuilt


def _attempt_payload(identity: AttributionIdentity) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_m30_attribution_attempt_v1",
        "status": "ATTEMPT_RESERVED_BEFORE_EVALUATION_PATH_RESOLUTION", "classification": CLASSIFICATION,
        "cell": CELL, "phase": PHASE, "identity": validate_identity(identity), "boundaries": dict(BOUNDARIES),
        "v3_score_sha256": V3_SCORE_SHA256, "v3_terminal_sha256": V3_TERMINAL_SHA256,
        "evaluation_assets_resolved": False, "remote_initialized": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
    }


def validate_attempt_payload(value: Mapping[str, object], *, identity: AttributionIdentity) -> dict[str, object]:
    expected = {
        "schema", "status", "classification", "cell", "phase", "identity", "boundaries", "v3_score_sha256",
        "v3_terminal_sha256", "evaluation_assets_resolved", "remote_initialized", "target_optimizer_steps",
        "target_backward_calls", "target_update_calls",
    }
    if not isinstance(value, Mapping) or set(value) != expected or value != _attempt_payload(identity):
        raise AttributionError("attribution attempt payload drift")
    return dict(value)


def _terminal_payload(*, identity: AttributionIdentity, attempt_sha256: str, input_replay_sha256: str,
                      score_sha256: str, final_closure: Mapping[str, object]) -> dict[str, object]:
    closure = identity.closure.payload()
    if dict(final_closure) != closure:
        raise AttributionError("attribution launch/final closure drift")
    return {
        "schema": "posterior_carrier_m30_attribution_terminal_v1",
        "status": "NON_GOVERNING_M30_ATTRIBUTION_COMPLETE__NO_FORMAL_VERDICT", "classification": CLASSIFICATION,
        "cell": CELL, "phase": PHASE, "identity": validate_identity(identity),
        "attempt_sha256": _sha(attempt_sha256, "attribution attempt SHA"),
        "input_replay_sha256": _sha(input_replay_sha256, "attribution input replay SHA"),
        "score_sha256": _sha(score_sha256, "attribution score SHA"),
        "v3_score_sha256": V3_SCORE_SHA256, "v3_terminal_sha256": V3_TERMINAL_SHA256,
        "launch_closure": closure, "final_closure": closure,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        "normalizer_refit": False, "target_sampling": False, "formal_opened": False, "h1_opened": False,
        "score_terminal_transactional_group": True, "no_governing_decision": True,
    }


def validate_terminal_payload(value: Mapping[str, object], *, identity: AttributionIdentity,
                              score_sha256: str | None = None) -> dict[str, object]:
    expected = {
        "schema", "status", "classification", "cell", "phase", "identity", "attempt_sha256", "input_replay_sha256",
        "score_sha256", "v3_score_sha256", "v3_terminal_sha256", "launch_closure", "final_closure",
        "target_optimizer_steps", "target_backward_calls", "target_update_calls", "normalizer_refit", "target_sampling",
        "formal_opened", "h1_opened", "score_terminal_transactional_group", "no_governing_decision",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise AttributionError("attribution terminal schema drift")
    closure = identity.closure.payload()
    if (
        value.get("schema") != "posterior_carrier_m30_attribution_terminal_v1"
        or value.get("status") != "NON_GOVERNING_M30_ATTRIBUTION_COMPLETE__NO_FORMAL_VERDICT"
        or value.get("classification") != CLASSIFICATION or value.get("cell") != CELL or value.get("phase") != PHASE
        or value.get("identity") != validate_identity(identity) or value.get("v3_score_sha256") != V3_SCORE_SHA256
        or value.get("v3_terminal_sha256") != V3_TERMINAL_SHA256 or value.get("launch_closure") != closure
        or value.get("final_closure") != closure or value.get("target_optimizer_steps") != 0
        or value.get("target_backward_calls") != 0 or value.get("target_update_calls") != 0
        or value.get("normalizer_refit") is not False or value.get("target_sampling") is not False
        or value.get("formal_opened") is not False or value.get("h1_opened") is not False
        or value.get("score_terminal_transactional_group") is not True or value.get("no_governing_decision") is not True
    ):
        raise AttributionError("attribution terminal immutable boundary drift")
    for key in ("attempt_sha256", "input_replay_sha256", "score_sha256"):
        _sha(value.get(key), f"attribution terminal {key}")
    if score_sha256 is not None and value["score_sha256"] != _sha(score_sha256, "expected attribution score SHA"):
        raise AttributionError("attribution terminal score binding drift")
    return dict(value)


def _failure_payload(*, identity: AttributionIdentity, stage: str, flags: RuntimeFlags,
                     attempt_sha256: str | None, input_replay_sha256: str | None, error: BaseException) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_m30_attribution_failure_v1", "classification": CLASSIFICATION,
        "cell": CELL, "phase": PHASE, "identity": validate_identity(identity), "stage": stage,
        "attempt_sha256": attempt_sha256, "input_replay_sha256": input_replay_sha256,
        "v3_score_sha256": V3_SCORE_SHA256, "v3_terminal_sha256": V3_TERMINAL_SHA256,
        "flags": flags.payload(), "terminal_published": False, "error_class": type(error).__name__,
        "error_sha256": _digest(repr(error).encode("utf-8")), "traceback_sha256": _digest(traceback.format_exc().encode("utf-8")),
    }


class _ArtifactRoot:
    """Small O_EXCL/fsync immutable publisher for a fresh route-owned root."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root).absolute()
        self.parent = self.path.parent
        self._parent_fd: int | None = None
        self._dir_fd: int | None = None

    def reserve(self) -> None:
        if self.path.name != Path(RESULT_ROOT_RELATIVE).name:
            raise AttributionError("attribution publisher may reserve only its canonical result leaf")
        self.parent.mkdir(parents=True, exist_ok=True)
        self._parent_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.mkdir(self.path.name, 0o700, dir_fd=self._parent_fd)
        except FileExistsError as error:
            raise AttributionError("attribution canonical output root already exists") from error
        self._dir_fd = os.open(self.path.name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self._parent_fd)
        os.fsync(self._parent_fd)

    def _remove_pair(self, name: str) -> None:
        if self._dir_fd is None:
            return
        for leaf in (name + ".sha256", name):
            try:
                info = os.stat(leaf, dir_fd=self._dir_fd, follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    os.unlink(leaf, dir_fd=self._dir_fd)
            except FileNotFoundError:
                continue

    def _write_pair(self, name: str, body: bytes) -> str:
        if self._dir_fd is None or "/" in name or not name.endswith(".json"):
            raise AttributionError("attribution publisher is not reserved for one JSON leaf")
        digest = _digest(body); sidecar = f"{digest}  {name}\n".encode("ascii")
        made = False
        try:
            for leaf, data in ((name, body), (name + ".sha256", sidecar)):
                made = True
                fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444, dir_fd=self._dir_fd)
                try:
                    offset = 0
                    while offset < len(data):
                        count = os.write(fd, data[offset:])
                        if count <= 0:
                            raise AttributionError("short immutable receipt write")
                        offset += count
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.chmod(leaf, 0o444, dir_fd=self._dir_fd)
        except BaseException:
            if made:
                self._remove_pair(name)
            raise
        os.fsync(self._dir_fd)
        return digest

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        return self._write_pair(name, _json(dict(payload)))

    def reload_json(self, name: str, expected_sha256: str) -> dict[str, object]:
        if self._dir_fd is None or "/" in name:
            raise AttributionError("attribution publisher is not reserved for immutable reload")
        expected = _sha(expected_sha256, "attribution immutable reload SHA")
        body_fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self._dir_fd)
        side_fd = os.open(name + ".sha256", os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self._dir_fd)
        try:
            body_info, side_info = os.fstat(body_fd), os.fstat(side_fd)
            if (
                not stat.S_ISREG(body_info.st_mode) or not stat.S_ISREG(side_info.st_mode)
                or stat.S_IMODE(body_info.st_mode) != 0o444 or stat.S_IMODE(side_info.st_mode) != 0o444
            ):
                raise AttributionError("attribution immutable reload type/mode drift")
            body = b""; side = b""
            while True:
                block = os.read(body_fd, 1 << 20)
                if not block:
                    break
                body += block
            while True:
                block = os.read(side_fd, 1 << 20)
                if not block:
                    break
                side += block
        finally:
            os.close(body_fd); os.close(side_fd)
        if _digest(body) != expected or side != f"{expected}  {name}\n".encode("ascii"):
            raise AttributionError("attribution immutable reload body/sidecar drift")
        try:
            value = json.loads(body)
        except (TypeError, json.JSONDecodeError) as error:
            raise AttributionError("attribution immutable reload JSON drift") from error
        if not isinstance(value, dict):
            raise AttributionError("attribution immutable reload root must be object")
        return value

    def publish_success_group(self, *, score: Mapping[str, object], terminal: Mapping[str, object]) -> dict[str, str]:
        """Publish score+terminal as one rollback-on-any-error immutable group."""
        made: list[str] = []
        try:
            score_sha = self._write_pair("score.json", _json(dict(score))); made.append("score.json")
            terminal_sha = self._write_pair("terminal.json", _json(dict(terminal))); made.append("terminal.json")
            if self.reload_json("score.json", score_sha) != dict(score):
                raise AttributionError("attribution score group reload drift")
            if self.reload_json("terminal.json", terminal_sha) != dict(terminal):
                raise AttributionError("attribution terminal group reload drift")
            return {"score.json": score_sha, "terminal.json": terminal_sha}
        except BaseException:
            for name in reversed(made):
                self._remove_pair(name)
            raise

    def close(self) -> None:
        for descriptor in (self._dir_fd, self._parent_fd):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        self._dir_fd = self._parent_fd = None


_CAPABILITY_SEAL = object()


@dataclass(frozen=True)
class AttributionExecutionCapability:
    identity_sha256: str
    v3_score_sha256: str
    v3_terminal_sha256: str
    _seal: object = field(repr=False, compare=False, default=_CAPABILITY_SEAL)


def _issue_root_review_capability(identity: AttributionIdentity) -> AttributionExecutionCapability:
    """Private helper; the public CLI cannot construct this opaque object."""
    return AttributionExecutionCapability(
        identity_sha256=_digest(_json(validate_identity(identity))), v3_score_sha256=V3_SCORE_SHA256,
        v3_terminal_sha256=V3_TERMINAL_SHA256,
    )


def _require_capability(value: object, *, identity: AttributionIdentity) -> AttributionExecutionCapability:
    if (
        not isinstance(value, AttributionExecutionCapability) or value._seal is not _CAPABILITY_SEAL
        or value.identity_sha256 != _digest(_json(validate_identity(identity)))
        or value.v3_score_sha256 != V3_SCORE_SHA256 or value.v3_terminal_sha256 != V3_TERMINAL_SHA256
    ):
        raise AttributionError("root-reviewed in-process attribution capability required before evaluation access")
    return value


class AttributionBackend(Protocol):
    def prepare(self, *, identity: AttributionIdentity, flags: RuntimeFlags) -> None: ...
    def resolve_inputs(self, *, identity: AttributionIdentity, flags: RuntimeFlags) -> InputReplay: ...
    def score_cell(self, *, cell: AttributionCell, input_payload: Mapping[str, object], flags: RuntimeFlags) -> CellEvidence: ...
    def reverify_after_forwards(self, *, identity: AttributionIdentity, flags: RuntimeFlags) -> ImplementationClosure: ...
    def device_attestation(self) -> Mapping[str, object]: ...
    def v3_validation(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


class NoLiveBackend:
    def prepare(self, *, identity: AttributionIdentity, flags: RuntimeFlags) -> None:
        raise AttributionError("attribution physical backend is unavailable in the dry route")
    def resolve_inputs(self, *, identity: AttributionIdentity, flags: RuntimeFlags) -> InputReplay:
        raise AssertionError("NoLiveBackend.prepare must fail before input resolution")
    def score_cell(self, *, cell: AttributionCell, input_payload: Mapping[str, object], flags: RuntimeFlags) -> CellEvidence:
        raise AssertionError("NoLiveBackend cannot score")
    def reverify_after_forwards(self, *, identity: AttributionIdentity, flags: RuntimeFlags) -> ImplementationClosure:
        raise AssertionError("NoLiveBackend cannot reverify")
    def device_attestation(self) -> Mapping[str, object]:
        raise AssertionError("NoLiveBackend cannot attest device")
    def v3_validation(self) -> Mapping[str, object]:
        raise AssertionError("NoLiveBackend cannot validate V3")
    def close(self) -> None:
        return None


def run_attribution_lifecycle(*, root: Path, identity: AttributionIdentity, backend: AttributionBackend,
                              execution_capability: object) -> dict[str, object]:
    """Future reviewed execution lifecycle; public CLI cannot reach it."""
    _require_capability(execution_capability, identity=identity)
    checked_identity = validate_identity(identity)
    result_root = Path(root).absolute() / RESULT_ROOT_RELATIVE
    artifact = _ArtifactRoot(result_root)
    flags = RuntimeFlags(); attempt_sha: str | None = None; input_sha: str | None = None
    try:
        artifact.reserve()
        attempt = _attempt_payload(identity)
        attempt_sha = artifact.publish_json("attempt.json", attempt)
        backend.prepare(identity=identity, flags=flags)
        input_payload = backend.resolve_inputs(identity=identity, flags=flags).payload(identity=identity)
        input_payload = validate_input_replay_payload(input_payload, identity=identity)
        input_sha = artifact.publish_json("input_replay.json", input_payload)
        evidence: list[dict[str, object]] = []
        for cell in attribution_matrix():
            item = backend.score_cell(cell=cell, input_payload=input_payload, flags=flags).payload(
                identity=identity, input_payload=input_payload,
            )
            evidence.append(item); flags.record_cell(cell)
        final_closure = backend.reverify_after_forwards(identity=identity, flags=flags).payload()
        score = build_score_payload(
            identity=identity, input_payload=input_payload, evidence=evidence, flags=flags,
            device_attestation=backend.device_attestation(), v3_validation=backend.v3_validation(),
        )
        score_sha = _digest(_json(score))
        terminal = _terminal_payload(
            identity=identity, attempt_sha256=attempt_sha, input_replay_sha256=input_sha,
            score_sha256=score_sha, final_closure=final_closure,
        )
        # Validate all logical bindings before the irreversible success group.
        # There must be no validation branch after group publication that
        # could manufacture a contradictory failure beside score+terminal.
        validate_terminal_payload(terminal, identity=identity, score_sha256=score_sha)
        group = artifact.publish_success_group(score=score, terminal=terminal)
        return {
            "attempt_sha256": attempt_sha,
            "input_replay_sha256": input_sha,
            "score_sha256": group["score.json"],
            "terminal_sha256": group["terminal.json"],
        }
    except BaseException as error:
        if attempt_sha is not None and artifact._dir_fd is not None:
            try:
                artifact.publish_json("failure.json", _failure_payload(
                    identity=identity, stage="runtime", flags=flags, attempt_sha256=attempt_sha,
                    input_replay_sha256=input_sha, error=error,
                ))
            except BaseException:
                pass
        raise
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


def dry_plan() -> dict[str, object]:
    """Pure contract output used by the public zero-argument CLI."""
    return {
        "status": "DRY_ONLY__NO_NETWORK_NO_TORCH_NO_NWB_NO_CHECKPOINT_NO_CUDA_NO_WRITE",
        "classification": CLASSIFICATION, "cell": CELL, "phase": PHASE,
        "v3_predecessor": V3Evidence().payload(),
        "matrix": [cell.payload() for cell in attribution_matrix()], "metric": dict(METRIC),
        "boundaries": dict(BOUNDARIES),
        "stage_root": REMOTE_STAGE_ROOT, "result_root_relative": RESULT_ROOT_RELATIVE,
        "requires_in_process_root_capability": True,
    }


def execute_authorized(*, execution_capability: object | None = None) -> None:
    """Public flags cannot cross the capability boundary."""
    del execution_capability
    raise AttributionError("attribution route remains dry until a reviewed in-process capability and physical backend exist")
