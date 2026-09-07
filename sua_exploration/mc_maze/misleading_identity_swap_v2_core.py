"""Fail-closed scientific core for the additive misleading-identity swap-v2 scaffold.

This module is deliberately data-agnostic.  It validates the frozen development
contract, builds deterministic synthetic/source matching authorities from caller-
supplied descriptors, and aggregates already-produced scores.  It never discovers
or opens NWB files, checkpoints, target data, formal data, or CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mc_maze.misleading_identity_swap import (
    MatchedSwap,
    build_partial_matched_involution,
    session_epoch_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
CONFIG_PATH = SUA_ROOT / "configs" / "misleading_identity_swap_v2.json"
CONTRACT_PATH = SUA_ROOT / "docs" / "MISLEADING_IDENTITY_SWAP_V2_SUCCESSOR_V3_CONTRACT_20260815.md"
RESULT_ROOT = SUA_ROOT / "results" / "misleading_identity_swap_v2_stage_p_successor_v3"
A2_PARENT_RESULT_ROOT = SUA_ROOT / "results" / "a2_matched_subject_shift_v2"
OFFICIAL_PREFLIGHT_PATH = (
    SUA_ROOT / "results" / "misleading_identity_swap_v2_source_authority_dev" /
    "official_cpu_preflight_v3.json"
)
CELL_OUTPUT_ROOT = SUA_ROOT / "checkpoints" / "misleading_identity_swap_v2_stage_p_successor_v3_seed42"
SOURCE_AUTHORITY_PATH = (
    SUA_ROOT / "results" / "misleading_identity_swap_v2_source_authority_dev" /
    "strict27_m30_matching_authority_v3.json"
)
SOURCE_LINEAGE_PATH = (
    SUA_ROOT / "results" / "misleading_identity_swap_v2_source_authority_dev" /
    "strict27_m30_source_lineage_v3.json"
)
INITIAL_STATE_PATH = (
    SUA_ROOT / "results" / "misleading_identity_swap_v2_source_authority_dev" /
    "shared_initial_state_v2.pt"
)

SCREEN_ID = "misleading_identity_swap_v2_stage_p"
STAGE_SEED = 42
FRACTION = 0.5
BASE_SEED = 42
TRAINING_EPOCHS_ZERO_BASED = tuple(range(12))
SCORE_EPOCHS_ONE_BASED = tuple(range(5, 13))
CELLS = ("clean_z4", "clean_t4", "swap_z4", "swap_t4")
DOMAINS = ("within_subject", "external_subject_M")
EVAL_INPUT_MODES = ("clean", "swapped_diagnostic")
DESCRIPTOR_COLUMNS = ("m_cos_phi", "m_sin_phi", "m", "b")
AUTHORITY_KIND = "misleading_identity_swap_v2_matching_authority"
AUTHORITY_STATUS = "DEVELOPMENT_AUTHORITY_NOT_OFFICIAL"
TARGET_AUTHORITY_KIND = "misleading_identity_swap_v2_target_diagnostic_authority"
TARGET_AUTHORITY_STATUS = "DEVELOPMENT_TARGET_DIAGNOSTIC_AUTHORITY_NOT_OFFICIAL"
PREFLIGHT_KIND = "misleading_identity_swap_v2_cpu_preflight"
PREFLIGHT_STATUS = "DEVELOPMENT_PREFLIGHT_PASS__NON_AUTHORIZING__NO_DATA_NO_GPU"
OFFICIAL_PREFLIGHT_KIND = "misleading_identity_swap_v2_official_cpu_preflight"
OFFICIAL_PREFLIGHT_STATUS = "OFFICIAL_CPU_PREFLIGHT_PASS__BOUNDED_STAGE_P_SUCCESSOR_V3_ONLY"
INVALID_V2_ATTEMPT_PATH = (
    SUA_ROOT / "checkpoints" / "misleading_identity_swap_v2_stage_p_seed42" /
    "clean_z4" / "launch_receipt.json"
)
INVALID_V2_ATTEMPT_SHA256 = "1dfb05e1155fbfc7d0f39f38081e2ef19e5f5fd6612d76b7e5f0924ce0d06f73"


class SwapV2ContractError(ValueError):
    """A frozen swap-v2 invariant is absent or has drifted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SwapV2ContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_bindings(bindings: Any, *, label: str) -> None:
    require(isinstance(bindings, Mapping) and bool(bindings), f"{label} bindings missing")
    for name, row in bindings.items():
        require(isinstance(name, str) and isinstance(row, Mapping), f"{label} binding malformed")
        path = Path(str(row.get("path", ""))).resolve()
        expected = row.get("sha256")
        require(path.is_file() and isinstance(expected, str) and len(expected) == 64,
                f"{label} binding missing: {name}")
        require(sha256_file(path) == expected, f"{label} implementation drift: {name}")


def _array_binding_sha256(value: np.ndarray, *, semantic_dtype: str) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json_bytes(
        {"semantic_dtype": semantic_dtype, "shape": list(array.shape)}
    )
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def descriptor_sha256(value: torch.Tensor) -> str:
    array = np.asarray(value.detach().cpu(), dtype="<f8", order="C")
    return _array_binding_sha256(array, semantic_dtype="float64_le")


def permutation_sha256(value: Sequence[int]) -> str:
    array = np.asarray(list(value), dtype="<i8")
    return _array_binding_sha256(array, semantic_dtype="int64_le")


def selected_mask_sha256(value: Sequence[bool]) -> str:
    array = np.asarray(list(value), dtype=np.uint8)
    return _array_binding_sha256(array, semantic_dtype="bool_as_uint8")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SwapV2ContractError(f"cannot read JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    payload = load_json_object(path)
    require(payload.get("schema_version") == 2, "config schema drift")
    require(payload.get("screen_id") == SCREEN_ID, "config screen id drift")
    require(payload.get("status") == "DEVELOPMENT_LIVE_INTEGRATION_SUCCESSOR_V3_NON_AUTHORIZING", "config status drift")
    require(payload.get("contract") == "sua_exploration/docs/MISLEADING_IDENTITY_SWAP_V2_SUCCESSOR_V3_CONTRACT_20260815.md",
            "contract path drift")
    parent = payload.get("parent_protocol")
    require(isinstance(parent, Mapping), "parent protocol binding missing")
    require(parent.get("screen_id") == "a2_matched_subject_shift_v2", "parent protocol screen drift")
    require(parent.get("config") == "sua_exploration/configs/a2_matched_subject_shift_v2.json",
            "parent protocol config drift")
    require(parent.get("use") == "read_only_protocol_and_numerical_reference_only",
            "A2 parent use expanded beyond read-only reference")
    require(parent.get("sealed_outputs_must_not_be_modified") is True, "A2 sealed protection missing")
    stage = payload.get("stage")
    require(isinstance(stage, Mapping), "stage config missing")
    require(stage.get("name") == "stage_p" and stage.get("seed") == STAGE_SEED, "Stage-P seed drift")
    require(tuple(stage.get("fresh_training_cells") or ()) == CELLS, "fresh 2x2 cell topology drift")
    require(tuple(stage.get("scoring_domains") or ()) == DOMAINS, "scoring-domain drift")
    require(tuple(stage.get("evaluation_input_modes") or ()) == EVAL_INPUT_MODES, "evaluation-input mode drift")
    intervention = payload.get("intervention")
    require(isinstance(intervention, Mapping), "intervention config missing")
    require(intervention.get("fraction") == FRACTION, "swap fraction drift")
    require(intervention.get("base_seed") == BASE_SEED, "swap base seed drift")
    require(intervention.get("mapping_kind") == "session_epoch_partial_involution", "mapping kind drift")
    require(intervention.get("epoch_namespace") == "lightning_current_epoch_zero_based", "epoch namespace drift")
    require(intervention.get("application_point") == "B3S.mean_feat_after_pool_before_side_concat", "swap point drift")
    for key in ("query_permuted", "visible_carrier_permuted", "electrode_ids_permuted"):
        require(intervention.get(key) is False, f"forbidden permutation enabled: {key}")
    descriptor = payload.get("matching_descriptor")
    require(isinstance(descriptor, Mapping), "matching descriptor config missing")
    require(tuple(descriptor.get("columns") or ()) == DESCRIPTOR_COLUMNS, "descriptor column drift")
    require(descriptor.get("rate_coordinate") == "b" and descriptor.get("extra_rate_column") is False,
            "descriptor rate semantics drift")
    require(descriptor.get("dtype") == "float64", "descriptor dtype drift")
    require(descriptor.get("normalization") == "ordinary_t4_source_train_normalizer",
            "descriptor normalizer authority drift")
    require(descriptor.get("support") == "chronological_first_30_rewarded_trials",
            "descriptor support chronology drift")
    require(descriptor.get("shared_authority_required") is True, "shared authority requirement missing")
    require(descriptor.get("model_visible_in_z4") is False, "hidden descriptor became model-visible in Z4")
    frozen = payload.get("frozen_training")
    require(isinstance(frozen, Mapping), "frozen training config missing")
    expected_training = {
        "variant": "B3S",
        "loss_mode": "task_only",
        "activity_calibration_trials": 30,
        "side_feature_pool_trials": 30,
        "evaluation_start_trial_index": 30,
        "epochs": 12,
        "epoch_window": list(SCORE_EPOCHS_ONE_BASED),
        "learning_rate": 1e-4,
        "batch_size": 32,
        "chronological_calibration": True,
        "target_updates": 0,
        "formal_subc_test_opened": False,
    }
    for key, expected in expected_training.items():
        require(frozen.get(key) == expected, f"frozen training drift: {key}")
    gates = payload.get("gates")
    require(isinstance(gates, Mapping), "gate config missing")
    require(gates.get("external_clean_input_t4_lift_min") == 0.03, "external gate drift")
    require(gates.get("within_clean_input_t4_delta_min") == -0.03, "within gate drift")
    require(gates.get("carrier_interaction") == "report_non_rescuing", "interaction became rescuing")
    require(gates.get("external_t4_robustness_contrast") == "diagnostic_non_rescuing",
            "robustness diagnostic became a rescue gate")
    require(gates.get("stop_on_any_primary_failure") is True, "fail-fast primary gate missing")
    execution = payload.get("execution")
    require(isinstance(execution, Mapping), "execution config missing")
    require(execution.get("official_preflight_minted") is True, "canonical official preflight requirement missing")
    require(execution.get("official_preflight") ==
            "sua_exploration/results/misleading_identity_swap_v2_source_authority_dev/official_cpu_preflight_v3.json",
            "official preflight path drift")
    require(execution.get("cell_output_root") ==
            "sua_exploration/checkpoints/misleading_identity_swap_v2_stage_p_successor_v3_seed42",
            "canonical cell output root drift")
    require(execution.get("default_commands_are_dry_run") is True, "default execution became active")
    require(execution.get("gpu_launch_implemented") is True, "dedicated GPU launcher missing")
    require(execution.get("target_scorer_implemented") is True, "dedicated target scorer missing")
    require(execution.get("source_authority_builder") ==
            "sua_exploration/scripts/misleading_identity_swap_v2_build_source_authority.py",
            "source authority builder path drift")
    require(execution.get("cell_trainer") ==
            "sua_exploration/scripts/misleading_identity_swap_v2_train_cell.py",
            "cell trainer path drift")
    require(execution.get("queue_bridge") ==
            "sua_exploration/scripts/misleading_identity_swap_v2_queue_bridge.sh",
            "queue bridge path drift")
    require(execution.get("fresh_result_root") == "sua_exploration/results/misleading_identity_swap_v2_stage_p_successor_v3",
            "fresh result-root drift")
    require(execution.get("predecessor_invalid_attempt") == {
        "path": "sua_exploration/checkpoints/misleading_identity_swap_v2_stage_p_seed42/clean_z4/launch_receipt.json",
        "launch_receipt_sha256": INVALID_V2_ATTEMPT_SHA256,
        "outcome": "LIGHTNING_SANITY_VALIDATION_ABORT__NO_EPOCH_CHECKPOINT_OR_TERMINAL",
        "old_b117_v2_official_preflight_reusable": False,
        "old_cell_root_reusable": False,
        "successor_source_only_sanity_regression_required": True,
    }, "successor invalid-attempt binding drift")
    return payload


def _normalized_descriptor(value: torch.Tensor, *, session: str) -> torch.Tensor:
    require(isinstance(session, str) and bool(session), "authority session name is required")
    require(isinstance(value, torch.Tensor), f"{session}: descriptor must be a tensor")
    require(value.ndim == 2 and value.shape[1] == len(DESCRIPTOR_COLUMNS),
            f"{session}: descriptor must have shape [N,{len(DESCRIPTOR_COLUMNS)}]")
    require(value.shape[0] >= 4, f"{session}: descriptor needs at least four units")
    normalized = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    require(torch.isfinite(normalized).all().item(), f"{session}: descriptor contains non-finite values")
    return normalized


def build_matching_authority(
    descriptors: Mapping[str, torch.Tensor],
    *,
    epochs: Sequence[int] = TRAINING_EPOCHS_ZERO_BASED,
    fraction: float = FRACTION,
    base_seed: int = BASE_SEED,
    source_only: bool = True,
) -> dict[str, Any]:
    """Build deterministic authority bytes from already-authorized descriptors.

    This function does no data access.  A future data-side preparer must supply
    descriptors derived from the frozen first-M30 support and bind that lineage.
    """
    require(float(fraction) == FRACTION, "authority fraction must remain 0.5")
    require(int(base_seed) == BASE_SEED, "authority base seed must remain 42")
    epoch_values = tuple(int(value) for value in epochs)
    require(epoch_values == TRAINING_EPOCHS_ZERO_BASED, "authority must cover zero-based epochs 0..11")
    require(bool(descriptors), "authority requires at least one session")
    require(tuple(sorted(descriptors)) == tuple(descriptors), "authority sessions must be unique and sorted")
    sessions: list[dict[str, Any]] = []
    for session, supplied in descriptors.items():
        descriptor = _normalized_descriptor(supplied, session=session)
        mappings: dict[str, dict[str, Any]] = {}
        for epoch in epoch_values:
            seed = session_epoch_seed(session, epoch, base_seed=base_seed)
            swap = build_partial_matched_involution(descriptor, fraction=fraction, seed=seed)
            permutation = [int(value) for value in swap.permutation.tolist()]
            selected = [bool(value) for value in swap.selected.tolist()]
            mappings[str(epoch)] = {
                "lightning_current_epoch": epoch,
                "epoch_number_one_based": epoch + 1,
                "seed": seed,
                "permutation": permutation,
                "permutation_sha256": permutation_sha256(permutation),
                "selected_mask": selected,
                "selected_mask_sha256": selected_mask_sha256(selected),
                "selected_count": swap.selected_count,
                "actual_fraction": swap.selected_count / descriptor.shape[0],
                "mean_pair_distance": swap.mean_pair_distance,
                "max_pair_distance": swap.max_pair_distance,
            }
        sessions.append(
            {
                "session": session,
                "descriptor_columns": list(DESCRIPTOR_COLUMNS),
                "descriptor_dtype": "float64",
                "descriptor_shape": list(descriptor.shape),
                "descriptor_values": descriptor.tolist(),
                "descriptor_sha256": descriptor_sha256(descriptor),
                "mappings": mappings,
            }
        )
    return {
        "schema_version": 1,
        "receipt_kind": AUTHORITY_KIND,
        "status": AUTHORITY_STATUS,
        "screen_id": SCREEN_ID,
        "source_only": bool(source_only),
        "fraction": FRACTION,
        "base_seed": BASE_SEED,
        "epoch_namespace": "lightning_current_epoch_zero_based",
        "epochs": list(TRAINING_EPOCHS_ZERO_BASED),
        "descriptor_schema": {
            "columns": list(DESCRIPTOR_COLUMNS),
            "dtype": "float64",
            "rate_coordinate": "b",
            "extra_rate_column": False,
            "model_visible_in_z4": False,
        },
        "sessions": sessions,
        "t4_z4_shared_bytes_required": True,
        "target_nwb_opened": False,
        "formal_subc_test_nwb_opened": False,
        "gpu_used": False,
    }


def validate_matching_authority(payload: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(payload, Mapping), "matching authority must be an object")
    require(payload.get("schema_version") == 1, "authority schema drift")
    require(payload.get("receipt_kind") == AUTHORITY_KIND, "authority kind drift")
    require(payload.get("status") == AUTHORITY_STATUS, "authority status drift")
    require(payload.get("screen_id") == SCREEN_ID, "authority screen drift")
    require(payload.get("source_only") is True, "authority must be source-only")
    require(payload.get("fraction") == FRACTION and payload.get("base_seed") == BASE_SEED,
            "authority fraction/base seed drift")
    require(payload.get("epochs") == list(TRAINING_EPOCHS_ZERO_BASED), "authority epoch coverage drift")
    require(payload.get("t4_z4_shared_bytes_required") is True, "authority sibling-sharing gate missing")
    require(payload.get("target_nwb_opened") is False, "development authority opened target data")
    require(payload.get("formal_subc_test_nwb_opened") is False, "authority opened formal data")
    require(payload.get("gpu_used") is False, "authority unexpectedly used GPU")
    schema = payload.get("descriptor_schema")
    require(isinstance(schema, Mapping), "authority descriptor schema missing")
    require(tuple(schema.get("columns") or ()) == DESCRIPTOR_COLUMNS, "authority descriptor columns drift")
    require(schema.get("dtype") == "float64" and schema.get("rate_coordinate") == "b",
            "authority descriptor semantics drift")
    require(schema.get("extra_rate_column") is False and schema.get("model_visible_in_z4") is False,
            "authority descriptor visibility drift")
    rows = payload.get("sessions")
    require(isinstance(rows, list) and rows, "authority session rows missing")
    descriptors: dict[str, torch.Tensor] = {}
    names: list[str] = []
    for row in rows:
        require(isinstance(row, Mapping), "authority session row malformed")
        session = row.get("session")
        require(isinstance(session, str) and bool(session), "authority session name malformed")
        names.append(session)
        require(row.get("descriptor_columns") == list(DESCRIPTOR_COLUMNS), f"{session}: descriptor columns drift")
        require(row.get("descriptor_dtype") == "float64", f"{session}: descriptor dtype drift")
        descriptor = torch.as_tensor(row.get("descriptor_values"), dtype=torch.float64)
        descriptor = _normalized_descriptor(descriptor, session=session)
        require(row.get("descriptor_shape") == list(descriptor.shape), f"{session}: descriptor shape drift")
        require(row.get("descriptor_sha256") == descriptor_sha256(descriptor), f"{session}: descriptor SHA drift")
        descriptors[session] = descriptor
    require(names == sorted(names) and len(set(names)) == len(names), "authority sessions must be sorted and unique")
    expected = build_matching_authority(
        descriptors,
        epochs=TRAINING_EPOCHS_ZERO_BASED,
        fraction=FRACTION,
        base_seed=BASE_SEED,
        source_only=payload.get("source_only") is True,
    )
    require(dict(payload) == expected, "authority mapping or metadata drift")
    return expected


def build_target_diagnostic_authority(
    descriptors: Mapping[str, torch.Tensor],
    *,
    domain: str,
) -> dict[str, Any]:
    """Build the one-per-domain M30 diagnostic mapping used only for scoring.

    Unlike the frozen source authority this receipt truthfully records that
    development target support was opened.  It remains label-free with respect
    to weight updates: support direction is used only to construct the hidden
    matching descriptor and no query velocity can update the model.
    """
    require(domain in DOMAINS, "target diagnostic domain drift")
    payload = build_matching_authority(descriptors)
    payload.update({
        "receipt_kind": TARGET_AUTHORITY_KIND,
        "status": TARGET_AUTHORITY_STATUS,
        "source_only": False,
        "domain": domain,
        "target_nwb_opened": True,
        "target_support_direction_used_for_hidden_descriptor": True,
        "target_query_velocity_used_for_weight_updates": False,
        "target_optimizer_or_backward_steps": 0,
    })
    return payload


def validate_target_diagnostic_authority(payload: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(payload, Mapping), "target diagnostic authority must be an object")
    domain = payload.get("domain")
    require(domain in DOMAINS, "target diagnostic authority domain drift")
    rows = payload.get("sessions")
    require(isinstance(rows, list) and rows, "target diagnostic authority sessions missing")
    descriptors: dict[str, torch.Tensor] = {}
    for row in rows:
        require(isinstance(row, Mapping), "target diagnostic authority row malformed")
        session = row.get("session")
        require(isinstance(session, str) and session, "target diagnostic session malformed")
        descriptors[session] = torch.as_tensor(row.get("descriptor_values"), dtype=torch.float64)
    expected = build_target_diagnostic_authority(descriptors, domain=str(domain))
    require(dict(payload) == expected, "target diagnostic authority mapping or metadata drift")
    return expected


@dataclass(frozen=True)
class VerifiedMatchingAuthority:
    payload: dict[str, Any]
    sha256: str
    canonical_sha256: str
    immutable_file_path: str | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        consumed_bytes_sha256: str | None = None,
        immutable_file_path: Path | None = None,
    ) -> "VerifiedMatchingAuthority":
        validated = validate_matching_authority(payload)
        canonical = canonical_json_sha256(validated)
        if immutable_file_path is None:
            require(consumed_bytes_sha256 is None,
                    "authority byte SHA requires a verified immutable authority path")
            binding = canonical
            resolved_path = None
        else:
            requested_path = Path(immutable_file_path).expanduser()
            require(not requested_path.is_symlink(), "immutable authority path may not be a symlink")
            file_payload, file_sha = load_verified_immutable_json(requested_path)
            require(dict(file_payload) == validated, "authority payload/file bytes disagree")
            require(consumed_bytes_sha256 is None or consumed_bytes_sha256 == file_sha,
                    "authority consumed-byte SHA disagrees with immutable file")
            binding = file_sha
            resolved_path = str(requested_path.resolve())
        require(isinstance(binding, str) and len(binding) == 64, "authority byte SHA malformed")
        return cls(
            validated,
            binding,
            canonical,
            resolved_path,
        )

    def swap_for(self, session: str, lightning_current_epoch: int) -> MatchedSwap:
        require(lightning_current_epoch in TRAINING_EPOCHS_ZERO_BASED, "runtime epoch outside authority")
        rows = [row for row in self.payload["sessions"] if row["session"] == session]
        require(len(rows) == 1, f"authority has no unique session {session!r}")
        mapping = rows[0]["mappings"][str(lightning_current_epoch)]
        permutation = torch.tensor(mapping["permutation"], dtype=torch.long)
        selected = torch.tensor(mapping["selected_mask"], dtype=torch.bool)
        require(mapping["permutation_sha256"] == permutation_sha256(mapping["permutation"]),
                "runtime permutation SHA drift")
        require(mapping["selected_mask_sha256"] == selected_mask_sha256(mapping["selected_mask"]),
                "runtime selected-mask SHA drift")
        return MatchedSwap(
            permutation=permutation,
            selected=selected,
            selected_count=int(mapping["selected_count"]),
            mean_pair_distance=float(mapping["mean_pair_distance"]),
            max_pair_distance=float(mapping["max_pair_distance"]),
            seed=int(mapping["seed"]),
        )

    @classmethod
    def from_target_diagnostic_file(cls, path: Path) -> "VerifiedMatchingAuthority":
        payload, file_sha = load_verified_immutable_json(path)
        validated = validate_target_diagnostic_authority(payload)
        requested = Path(path).expanduser()
        require(not requested.is_symlink(), "immutable authority path may not be a symlink")
        return cls(
            validated,
            file_sha,
            canonical_json_sha256(validated),
            str(requested.resolve()),
        )


def verify_sibling_authority_bindings(bindings: Mapping[str, str], *, expected_sha256: str) -> None:
    require(set(bindings) == {"t4", "z4"}, "authority consumer set must be exactly T4/Z4")
    require(bindings["t4"] == expected_sha256 and bindings["z4"] == expected_sha256,
            "T4/Z4 did not consume byte-identical matching authority")


def aggregate_stage_p(scores: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate the frozen 4-cell x 2-domain x 2-input-mode scalar matrix."""
    require(set(scores) == set(CELLS), "score cell topology drift")
    values: dict[str, dict[str, dict[str, float]]] = {}
    for cell in CELLS:
        domains = scores[cell]
        require(isinstance(domains, Mapping) and set(domains) == set(DOMAINS), f"{cell}: domain topology drift")
        values[cell] = {}
        for domain in DOMAINS:
            modes = domains[domain]
            require(isinstance(modes, Mapping) and set(modes) == set(EVAL_INPUT_MODES),
                    f"{cell}/{domain}: input-mode topology drift")
            values[cell][domain] = {}
            for mode in EVAL_INPUT_MODES:
                value = float(modes[mode])
                require(np.isfinite(value), f"{cell}/{domain}/{mode}: non-finite score")
                values[cell][domain][mode] = value
    ext_t4 = values["swap_t4"]["external_subject_M"]["clean"] - values["clean_t4"]["external_subject_M"]["clean"]
    within_t4 = values["swap_t4"]["within_subject"]["clean"] - values["clean_t4"]["within_subject"]["clean"]
    interactions = {}
    for domain in DOMAINS:
        t4 = values["swap_t4"][domain]["clean"] - values["clean_t4"][domain]["clean"]
        z4 = values["swap_z4"][domain]["clean"] - values["clean_z4"][domain]["clean"]
        interactions[domain] = t4 - z4
    robustness = (
        values["swap_t4"]["external_subject_M"]["swapped_diagnostic"]
        - values["swap_t4"]["external_subject_M"]["clean"]
        - values["clean_t4"]["external_subject_M"]["swapped_diagnostic"]
        + values["clean_t4"]["external_subject_M"]["clean"]
    )
    gates = {
        "external_clean_input_t4_lift": ext_t4,
        "external_clean_input_t4_lift_pass": ext_t4 >= 0.03,
        "within_clean_input_t4_delta": within_t4,
        "within_clean_input_t4_delta_pass": within_t4 >= -0.03,
    }
    passed = all(value for key, value in gates.items() if key.endswith("_pass"))
    return {
        "schema_version": 1,
        "receipt_kind": "misleading_identity_swap_v2_stage_p_aggregate",
        "screen_id": SCREEN_ID,
        "stage_seed": STAGE_SEED,
        "scores": values,
        "primary_gates": gates,
        "carrier_interactions_non_rescuing": interactions,
        "external_t4_robustness_contrast_diagnostic_non_rescuing": robustness,
        "verdict": "STAGE_P_PASS_PRIMARY_GATES__SUCCESSOR_REQUIRED" if passed else "STOP_STAGE_P_PRIMARY_GATE_FAILURE",
        "robustness_can_rescue": False,
        "interaction_can_rescue": False,
        "formal_subc_test_nwb_opened": False,
        "target_optimizer_or_backward_steps": 0,
    }


def _readonly_regular(path: Path) -> bool:
    try:
        info = Path(path).lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444


def sidecar_path(path: Path) -> Path:
    return Path(path).with_name(f"{Path(path).name}.sha256")


def assert_immutable_pair_fresh(path: Path, *, label: str) -> None:
    body = Path(path).expanduser()
    sidecar = sidecar_path(body)
    require(not body.is_symlink() and not sidecar.is_symlink(), f"{label} path may not be a symlink")
    if os.path.lexists(body) or os.path.lexists(sidecar):
        raise FileExistsError(f"{label} immutable output pair is not fresh: {body}")


def require_canonical_official_preflight_path(path: Path) -> Path:
    requested = Path(path).expanduser()
    require(not requested.is_symlink(), "official preflight path may not be a symlink")
    resolved = requested.resolve()
    require(resolved == OFFICIAL_PREFLIGHT_PATH.resolve(),
            "caller-supplied official preflight path is not canonical")
    return resolved


def canonical_cell_output_paths() -> dict[str, str]:
    root = CELL_OUTPUT_ROOT.resolve()
    return {cell: str((root / cell).resolve()) for cell in CELLS}


def a2_parent_score_path(domain: str, carrier: str) -> Path:
    require(domain in DOMAINS, "A2 parent domain drift")
    require(carrier in {"t4", "z4"}, "A2 parent carrier drift")
    return A2_PARENT_RESULT_ROOT / f"{domain}_source_{carrier}_s42.json"


def load_a2_parent_domain_bindings(domain: str) -> dict[str, Any]:
    """Load both sealed matched-A2 s42 receipts and require scientific equality."""
    rows: dict[str, Any] = {}
    for carrier in ("t4", "z4"):
        path = a2_parent_score_path(domain, carrier)
        payload, digest = load_verified_immutable_json(path)
        require(payload.get("screen_id") == "a2_matched_subject_shift_v2", "A2 parent screen drift")
        require(payload.get("source_arm") == f"source_{carrier}", "A2 parent source arm drift")
        require(payload.get("seed") == 42 and payload.get("domain") == domain,
                "A2 parent seed/domain drift")
        rows[carrier] = {"path": str(path.resolve()), "sha256": digest, "payload": payload}
    left, right = rows["t4"]["payload"], rows["z4"]["payload"]
    for key in ("query_policy", "normalizer_authority", "domain_sessions", "session_query_receipts"):
        require(left.get(key) == right.get(key), f"matched A2 T4/Z4 parent {key} drift")
    return {
        "domain": domain,
        "receipts": {
            carrier: {"path": rows[carrier]["path"], "sha256": rows[carrier]["sha256"]}
            for carrier in ("t4", "z4")
        },
        "query_policy": left["query_policy"],
        "normalizer_authority": left["normalizer_authority"],
        "domain_sessions": left["domain_sessions"],
        "session_query_receipts": left["session_query_receipts"],
    }


def write_immutable_json_pair(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
    """Publish an all-or-none immutable body/sidecar pair without overwriting."""
    body = Path(path).expanduser().resolve()
    sidecar = sidecar_path(body)
    require(not Path(path).is_symlink(), "immutable output path may not be a symlink")
    if os.path.lexists(body) or os.path.lexists(sidecar):
        raise FileExistsError(f"immutable output pair is not fresh: {body}")
    body.parent.mkdir(parents=True, exist_ok=True)
    raw = pretty_json_bytes(dict(payload))
    digest = hashlib.sha256(raw).hexdigest()
    side_raw = f"{digest}  {body.name}\n".encode("ascii")
    temp_paths: list[Path] = []
    published_body = False
    try:
        for suffix, content in ((".body.tmp", raw), (".side.tmp", side_raw)):
            fd, temp_name = tempfile.mkstemp(prefix=f".{body.name}.", suffix=suffix, dir=body.parent)
            temp = Path(temp_name)
            temp_paths.append(temp)
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp, 0o444)
        os.link(temp_paths[0], body)
        published_body = True
        os.link(temp_paths[1], sidecar)
        directory_fd = os.open(body.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if published_body and not os.path.lexists(sidecar):
            body.unlink(missing_ok=True)
        raise
    finally:
        for temp in temp_paths:
            temp.unlink(missing_ok=True)
    require(_readonly_regular(body) and _readonly_regular(sidecar), "immutable pair mode drift")
    return body, sidecar, digest


def load_verified_immutable_json(path: Path) -> tuple[dict[str, Any], str]:
    requested_path = Path(path).expanduser()
    require(not requested_path.is_symlink(), "immutable input path may not be a symlink")
    body = requested_path.resolve()
    sidecar = sidecar_path(body)
    require(_readonly_regular(body) and _readonly_regular(sidecar), "immutable pair must be regular mode 0444")
    body_raw = body.read_bytes()
    side_raw = sidecar.read_bytes()
    digest = hashlib.sha256(body_raw).hexdigest()
    require(side_raw == f"{digest}  {body.name}\n".encode("ascii"), "immutable sidecar/body mismatch")
    try:
        payload = json.loads(body_raw)
    except json.JSONDecodeError as exc:
        raise SwapV2ContractError(f"invalid immutable JSON: {body}") from exc
    require(isinstance(payload, dict), "immutable JSON root must be an object")
    return payload, digest


def load_verified_authority(path: Path) -> VerifiedMatchingAuthority:
    payload, file_sha = load_verified_immutable_json(path)
    validated = validate_matching_authority(payload)
    require(len(file_sha) == 64, "authority file SHA malformed")
    return VerifiedMatchingAuthority.from_payload(
        validated,
        consumed_bytes_sha256=file_sha,
        immutable_file_path=path,
    )


def load_verified_runtime_authority(
    path: Path, *, expected_kind: str
) -> VerifiedMatchingAuthority:
    if expected_kind == "source":
        return load_verified_authority(path)
    if expected_kind == "target_diagnostic":
        return VerifiedMatchingAuthority.from_target_diagnostic_file(path)
    raise SwapV2ContractError(f"unsupported runtime authority kind: {expected_kind!r}")
