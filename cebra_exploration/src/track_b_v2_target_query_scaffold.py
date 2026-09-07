"""No-data target/query lineage scaffold for the H1-excluded Track-B v2 route.

Nothing in this module opens NWB/NPZ data, imports CEBRA, creates a model,
scores a prediction, uses CUDA, or writes a receipt.  It instead freezes the
objects a later, separately authorised target integration must present before
it can make a paired accuracy claim.  Keeping this interface array-free is
intentional: a target session ID is an opaque, grammar-checked identifier, not
authority to discover that session.

The route is deliberately limited to subject-M SUA/pMUA and RT.  H1 and M2
are rejected by ``base.validate_scope`` before any proposed target object is
inspected or coerced.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence
import re

import track_b_v2_contract as base
import track_b_v2_live_contract as live
import track_b_v2_metric_pointer_authority as metric_pointer
import track_b_v2_source_adapter as source_adapter


TARGET_QUERY_SCAFFOLD_SCHEMA = "track_b_v2_target_query_no_data_scaffold_v1"
POINTER_PROPOSAL_BINDING_SCHEMA = "track_b_v2_root_audited_pointer_proposal_binding_v1"
TARGET_SPLIT_SCHEMA = "track_b_v2_target_support_query_trial_window_contract_v1"
QUERY_RECEPTIVE_FIELD_PROOF_SCHEMA = "track_b_v2_query_receptive_field_and_target_byte_proof_v1"
READOUT_AUTHORITY_PLAN_SCHEMA = "track_b_v2_frozen_readout_authority_plan_v1"
TARGET_ADAPTER_PREFLIGHT_SCHEMA = "track_b_v2_source_only_target_adapter_preflight_v1"

_SUBJECT_M_TARGET = re.compile(r"sub-M_ses-CO-\d{8}")
_RT_TARGET = re.compile(r"ses-RT-\d{8}")
_SOURCE_AUTHORITY_KEYS = (
    "source_roster",
    "source_coverage",
    "source_neural_input_authority",
    "source_behavior_auxiliary_scaler_authority",
    "source_readout_embedding_identity_authority",
    "source_only_dual_geometry_selection_plan",
)


class TrackBV2TargetQueryScaffoldError(live.TrackBV2LiveContractError):
    """Raised before an unmaterialized target plan can become usable."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2TargetQueryScaffoldError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _opaque_target_session_id(dataset: str, value: object) -> str:
    """Validate a target identifier without treating it as a path or opening it."""
    require(isinstance(value, str) and value == value.strip(), "target session ID must be a trimmed string")
    pattern = _SUBJECT_M_TARGET if dataset == "subject_m" else _RT_TARGET
    require(pattern.fullmatch(value) is not None, "target session ID violates canonical target grammar")
    return value


def _canonical_indices(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    indices = tuple(values)
    require(indices, f"{label} must not be empty")
    require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in indices),
            f"{label} must contain non-negative integer indices")
    require(len(set(indices)) == len(indices), f"{label} contains duplicate indices")
    require(tuple(sorted(indices)) == indices, f"{label} must be in strict chronological order")
    return indices


def _require_authority_sha_map(value: Mapping[str, str]) -> dict[str, str]:
    require(isinstance(value, Mapping) and set(value) == set(_SOURCE_AUTHORITY_KEYS),
            "source authority SHA map must contain the exact six source-only authority members")
    result = dict(value)
    require(all(_valid_sha(item) for item in result.values()), "source authority SHA map contains invalid SHA")
    return result


def _half_open_intervals(indices: tuple[int, ...]) -> list[list[int]]:
    """Losslessly compact a sorted raw-index expansion into half-open runs."""
    intervals: list[list[int]] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            intervals.append([start, previous + 1])
            start = value
        previous = value
    intervals.append([start, previous + 1])
    return intervals


def _compact_sorted_index_authority(indices: tuple[int, ...], *, role: str) -> dict[str, Any]:
    """Emit a compact authority, never a potentially huge raw expansion."""
    expansion = {"role": role, "indices": list(indices)}
    return {
        "index_role": role,
        "sorted_half_open_intervals": _half_open_intervals(indices),
        "expansion_sha256": _sha_json(expansion),
        "index_count": len(indices),
        "first_index": indices[0],
        "last_index": indices[-1],
    }


def _compact_ordered_index_authority(indices: tuple[int, ...], *, role: str) -> dict[str, Any]:
    """Bind ordered prediction endpoints without serializing all endpoints."""
    ordered = {"role": role, "ordered_indices": list(indices)}
    return {
        "index_role": role,
        "ordered_index_sha256": _sha_json(ordered),
        "index_count": len(indices),
        "first_index": indices[0],
        "last_index": indices[-1],
    }


def bind_root_audited_pointer_proposal(
    *,
    dataset: str,
    view: str | None,
    proposal_payload: Mapping[str, Any],
    root_audit_attestation_sha256: str,
) -> dict[str, Any]:
    """Bind an exact pointer *proposal*, never a minted metric authority.

    The caller supplies the complete root-audited proposal body plus an
    independent audit-attestation SHA.  This makes its bytes identifiable at
    the future target gate while preserving the crucial state distinction:
    until root mints an immutable pointer body+sidecar, the baseline remains
    unavailable as a metric authority.
    """
    dataset, view = base.validate_scope(dataset, view)
    require(isinstance(proposal_payload, Mapping), "pointer proposal payload must be a mapping")
    proposal = dict(proposal_payload)
    require(proposal.get("schema") == "track_b_v2_metric_pointer_mint_proposal_v1",
            "pointer proposal schema drift")
    require(proposal.get("status") == "ROOT_AUDITED_IMMUTABLE_POINTER_REQUIRED__NOT_MINTED__NOT_AUTHORITY",
            "only a not-minted root-audited pointer proposal may enter this scaffold")
    require((proposal.get("dataset"), proposal.get("view")) == (dataset, view),
            "pointer proposal scope drift")
    require(proposal.get("rounded_literals_accepted") is False,
            "pointer proposal may not accept rounded metric literals")
    require(proposal.get("legacy_bodies_modified") is False,
            "pointer proposal may not modify legacy sealed bodies")
    require(_valid_sha(root_audit_attestation_sha256), "root audit attestation SHA invalid")
    bodies = proposal.get("bodies")
    require(isinstance(bodies, Mapping) and bodies, "pointer proposal bodies missing")
    required = (
        ("summary_cross_check_body", "per_session_seed_body")
        if dataset == "subject_m" else
        ("aggregate_metric_query_identity_body", "per_fold_t4d_body", "stage2_delta_companion")
    )
    require(set(bodies) == set(required), "pointer proposal body set drift")
    for name, body in bodies.items():
        require(isinstance(body, Mapping) and _valid_sha(body.get("sha256")),
                f"pointer proposal body SHA invalid: {name}")
        require(isinstance(body.get("path"), str) and body["path"], f"pointer proposal body path invalid: {name}")
        pointer = body.get("metric_json_pointer", body.get("required_json_pointer"))
        if pointer is not None:
            require(isinstance(pointer, str) and pointer.startswith("/"),
                    f"pointer proposal metric path invalid: {name}")
            require(body.get("metric_pointer_verified_against_same_fd_bytes") is True,
                    f"pointer proposal metric path was not verified from same-fd bytes: {name}")
    return {
        "schema": POINTER_PROPOSAL_BINDING_SCHEMA,
        "dataset": dataset,
        "view": view,
        "root_audit_attestation_sha256": root_audit_attestation_sha256,
        "proposal_payload_sha256": _sha_json(proposal),
        "proposal_status": proposal["status"],
        "immutable_pointer_minted": False,
        "metric_authority_available": False,
        "future_gate": "ROOT_MUST_MINT_NEW_IMMUTABLE_POINTER_BODY_AND_SIDECAR_BEFORE_METRIC_AUTHORITY",
        "proposal_body_sha256s": {name: body["sha256"] for name, body in bodies.items()},
        "legacy_body_bytes_reopened": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "gpu_used": False,
    }


@dataclass(frozen=True)
class TargetSupportQueryTrialWindowContract:
    """Future support/query trial-window rules, materialized with no target bytes."""

    dataset: str
    view: str | None
    outer_fold_id: str
    target_session_id: str

    def __post_init__(self) -> None:
        dataset, view = base.validate_scope(self.dataset, self.view)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "view", view)
        require(isinstance(self.outer_fold_id, str) and self.outer_fold_id == self.outer_fold_id.strip() and self.outer_fold_id,
                "outer fold ID must be a trimmed nonempty string")
        object.__setattr__(self, "target_session_id", _opaque_target_session_id(dataset, self.target_session_id))

    def as_dict(self) -> dict[str, Any]:
        if self.dataset == "subject_m":
            support = {
                "trial_ordinal_namespace": "rewarded_trial_ordinal",
                "required_support_trial_ordinals": list(range(50)),
                "required_query_trial_ordinals": "strictly_greater_than_49_rewarded_trials",
                "support_window_semantics": "first_50_rewarded_trials__all_eligible_dense_bins",
                "query_window_semantics": "rewarded_trials_after_M50__canonical_evaluation_rows_only",
                "standard_cebra_support_sequence": {
                    "sequence_semantics": "one_continuous_chronological_prefix",
                    "start": "CANONICAL_TARGET_RECORD_START_RAW_BIN",
                    "stop": "STOP_OF_REWARDED_TRIAL_50",
                    "all_intervening_raw_rows_included": True,
                    "boundary_matched": True,
                    "trial_bin_or_neural_exposure_matched": False,
                    "bias_direction": "favors_CEBRA_accuracy",
                    "rewarded_segments_concatenated": False,
                    "rewarded_segment_only_sensitivity": {
                        "status": "PREDECLARED_NOT_IMPLEMENTED__REVIEWED_VALID_INDEX_SEGMENT_IMPLEMENTATION_REQUIRED",
                        "cross_trial_boundary_positives_permitted": False,
                    },
                },
                "t4_neural_support_trial_count": 30,
                "t4_label_event_count": 50,
                "t4_label_row_count": 50,
                "t4_label_scalar_count": 50,
                "t4_label_semantics": live.SUBJECT_M_T4_LABEL_SEMANTICS,
                "cebra_neural_support_trial_count": 50,
                "cebra_dense_label_support_trial_count": 50,
                "neural_exposure_matched": False,
                "label_information_matched": False,
                "bias_direction": "favors_CEBRA_accuracy",
            }
        else:
            support = {
                "trial_ordinal_namespace": "chronological_RT_trial_ordinal",
                "required_support_trial_ordinals": list(range(24)),
                "required_query_trial_ordinals": "strictly_greater_than_23__sealed_RT_outer_q24_eligible_windows_only",
                "support_window_semantics": "chronological_first_24_trials__M24",
                "query_window_semantics": "sealed_RT_outer_q24_eligible_full_causal_windows_only",
                "standard_cebra_support_sequence": {
                    "sequence_semantics": "one_continuous_chronological_prefix",
                    "start": "CANONICAL_TARGET_RECORD_START_RAW_BIN",
                    "stop": "STOP_OF_CHRONOLOGICAL_TRIAL_24",
                    "all_intervening_raw_rows_included": True,
                    "boundary_matched": True,
                    "trial_bin_or_neural_exposure_matched": False,
                    "bias_direction": "favors_CEBRA_accuracy",
                    "rewarded_segments_concatenated": False,
                    "rewarded_segment_only_sensitivity": {
                        "status": "PREDECLARED_NOT_IMPLEMENTED__REVIEWED_VALID_INDEX_SEGMENT_IMPLEMENTATION_REQUIRED",
                        "cross_trial_boundary_positives_permitted": False,
                    },
                },
                "t4_neural_support_trial_count": 24,
                "t4_label_event_count": 24,
                "t4_label_row_count": "MUST_BIND_ACTUAL_ELIGIBLE_ENDPOINT_REACH_ROWS",
                "t4_label_scalar_count": "MUST_BIND_ACTUAL_ENDPOINT_DISPLACEMENT_COORDINATE_COUNT",
                "t4_label_semantics": live.RT_T4D_LABEL_SEMANTICS,
                "cebra_neural_support_trial_count": 24,
                "cebra_dense_label_support_trial_count": 24,
                "neural_exposure_matched": True,
                "label_information_matched": False,
                "bias_direction": "favors_CEBRA_accuracy",
            }
        return {
            "schema": TARGET_SPLIT_SCHEMA,
            "dataset": self.dataset,
            "view": self.view,
            "outer_fold_id": self.outer_fold_id,
            "target_session_id": self.target_session_id,
            "support_query_contract": support,
            "future_live_lineage_must_bind": {
                "target_support_raw_coverage_index_sha256": "REQUIRED",
                "standard_cebra_support_prefix_start_raw_bin": "REQUIRED",
                "standard_cebra_support_prefix_stop_raw_bin": "REQUIRED__STOP_OF_M50_OR_M24_TRIAL",
                "standard_cebra_support_prefix_all_intervening_raw_row_count": "REQUIRED",
                "standard_cebra_support_prefix_raw_coverage_expansion_sha256": "REQUIRED",
                "target_query_raw_coverage_index_sha256": "REQUIRED",
                "ordered_valid_window_start_indices_sha256": "REQUIRED_FOR_SUBJECT_M_ENDPOINT_PARITY",
                "ordered_prediction_target_raw_bin_indices_sha256": "REQUIRED__EXACTLY_VALID_START_PLUS_49_FOR_SUBJECT_M",
                "ordered_t4_reference_prediction_target_raw_bin_indices_sha256": "REQUIRED",
                "ordered_cebra_scored_prediction_target_raw_bin_indices_sha256": "REQUIRED",
                "t4_target_float32_raw_bytes_sha256": "REQUIRED",
                "cebra_target_float32_raw_bytes_sha256": "REQUIRED",
                "t4_runtime_receipt_sha256": "REQUIRED",
                "v9_commit_receipt_sha256": "REQUIRED_FOR_SUBJECT_M__EXACT_ASSET_VIEW_SEED_LINEAGE",
                "v9_runtime_base_input_trace_sha256": "REQUIRED_FOR_SUBJECT_M__ACTIVITY_IDENTITY_TRIALS_30",
                "v9_runtime_query_behavior_trace_sha256": "REQUIRED_FOR_SUBJECT_M__QUERY_STRICTLY_AFTER_REWARDED_TRIAL_50",
                "t4_predictions_targets_npz_sha256": "REQUIRED_FOR_SUBJECT_M__OR_EXPLICIT_UNPAIRED_CLAIM",
                "target_feature_sha256": "REQUIRED__PMUA_POOLING_PROVENANCE_ALSO_REQUIRED_FOR_PMUA",
                "metric_implementation_authority_sha256": "REQUIRED__MATCH_PARENT_TWO_OUTPUT_R2_REDUCTION",
            },
            "target_data_materialized": False,
            "target_data_opened": False,
            "target_query_opened": False,
        }


@dataclass(frozen=True)
class QueryReceptiveFieldAndTargetByteProof:
    """Pure verifier for a future paired target-query claim.

    It accepts only explicit synthetic/previously materialized indices and
    SHA labels.  It never receives a file path or an array.  Every full neural
    receptive field of every scored CEBRA output must remain inside the target
    query raw coverage, which makes support/query disjointness meaningful even
    for overlapping RT causal windows.
    """

    dataset: str
    view: str | None
    target_session_id: str
    support_raw_indices: tuple[int, ...]
    query_raw_indices: tuple[int, ...]
    valid_window_start_indices: tuple[int, ...]
    t4_reference_ordered_prediction_target_raw_bin_indices: tuple[int, ...]
    cebra_scored_ordered_prediction_target_raw_bin_indices: tuple[int, ...]
    full_receptive_field_raw_indices: tuple[tuple[int, ...], ...]
    vendored_model_alignment_authority_sha256: str
    t4_target_float32_raw_bytes_sha256: str
    cebra_target_float32_raw_bytes_sha256: str
    t4_reference_query_identity_sha256: str
    metric_implementation_authority_sha256: str

    def __post_init__(self) -> None:
        dataset, view = base.validate_scope(self.dataset, self.view)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "view", view)
        object.__setattr__(self, "target_session_id", _opaque_target_session_id(dataset, self.target_session_id))
        support = _canonical_indices(self.support_raw_indices, label="support raw coverage")
        query = _canonical_indices(self.query_raw_indices, label="query raw coverage")
        starts = _canonical_indices(self.valid_window_start_indices, label="ordered valid window starts")
        reference_endpoints = _canonical_indices(
            self.t4_reference_ordered_prediction_target_raw_bin_indices,
            label="T4 reference ordered prediction target raw bins",
        )
        cebra_endpoints = _canonical_indices(
            self.cebra_scored_ordered_prediction_target_raw_bin_indices,
            label="CEBRA ordered prediction target raw bins",
        )
        require(set(support).isdisjoint(query), "support and query raw coverage overlap")
        require(len(starts) == len(reference_endpoints), "each valid window start requires exactly one T4 prediction endpoint")
        require(all(endpoint == start + 49 for start, endpoint in zip(starts, reference_endpoints)),
                "T4 prediction targets must be causal valid-window endpoints: valid_start + 49")
        require(reference_endpoints == cebra_endpoints,
                "CEBRA prediction target raw bins must exactly equal T4 endpoint ordering")
        require(len(self.full_receptive_field_raw_indices) == len(cebra_endpoints),
                "every CEBRA scored row requires exactly one receptive-field declaration")
        query_set = set(query)
        require(set(reference_endpoints).issubset(query_set),
                "every prediction target raw-bin endpoint must lie inside target query coverage")
        canonical_receptive_fields: list[tuple[int, ...]] = []
        for index, receptive in enumerate(self.full_receptive_field_raw_indices):
            canonical = _canonical_indices(receptive, label=f"receptive field for scored row {index}")
            require(set(canonical).issubset(query_set),
                    "a scored CEBRA receptive field escapes target query raw coverage")
            require(set(canonical).isdisjoint(support),
                    "a scored CEBRA receptive field exposes target support")
            canonical_receptive_fields.append(canonical)
        for name, value in (
            ("T4 target bytes", self.t4_target_float32_raw_bytes_sha256),
            ("CEBRA target bytes", self.cebra_target_float32_raw_bytes_sha256),
            ("T4 query identity", self.t4_reference_query_identity_sha256),
            ("metric implementation authority", self.metric_implementation_authority_sha256),
            ("vendored model alignment authority", self.vendored_model_alignment_authority_sha256),
        ):
            require(_valid_sha(value), f"{name} SHA invalid")
        require(self.t4_target_float32_raw_bytes_sha256 == self.cebra_target_float32_raw_bytes_sha256,
                "CEBRA target bytes differ from ordered T4 target bytes; paired claim forbidden")
        object.__setattr__(self, "support_raw_indices", support)
        object.__setattr__(self, "query_raw_indices", query)
        object.__setattr__(self, "valid_window_start_indices", starts)
        object.__setattr__(self, "t4_reference_ordered_prediction_target_raw_bin_indices", reference_endpoints)
        object.__setattr__(self, "cebra_scored_ordered_prediction_target_raw_bin_indices", cebra_endpoints)
        object.__setattr__(self, "full_receptive_field_raw_indices", tuple(canonical_receptive_fields))

    def as_dict(self) -> dict[str, Any]:
        relative_offsets = tuple(
            tuple(raw_index - endpoint for raw_index in receptive)
            for endpoint, receptive in zip(
                self.cebra_scored_ordered_prediction_target_raw_bin_indices,
                self.full_receptive_field_raw_indices,
            )
        )
        future_counts = tuple(sum(offset > 0 for offset in offsets) for offsets in relative_offsets)
        max_future = max((max(offsets) for offsets in relative_offsets), default=0)
        payload = {
            "schema": QUERY_RECEPTIVE_FIELD_PROOF_SCHEMA,
            "dataset": self.dataset,
            "view": self.view,
            "target_session_id": self.target_session_id,
            "support_raw_coverage": _compact_sorted_index_authority(
                self.support_raw_indices, role="target_support_full_raw_coverage",
            ),
            "query_raw_coverage": _compact_sorted_index_authority(
                self.query_raw_indices, role="target_query_full_raw_coverage",
            ),
            "ordered_valid_window_start_indices": _compact_ordered_index_authority(
                self.valid_window_start_indices, role="valid_window_starts",
            ),
            "ordered_prediction_target_raw_bin_indices": _compact_ordered_index_authority(
                self.t4_reference_ordered_prediction_target_raw_bin_indices,
                role="T4_and_CEBRA_prediction_target_raw_bin_endpoints__valid_start_plus_49",
            ),
            "vendored_model_receptive_field_alignment": {
                "alignment_authority_sha256": self.vendored_model_alignment_authority_sha256,
                "prediction_endpoint_semantics": "causal_window_endpoint__valid_start_plus_49",
                "assumed_symmetric_window_permitted": False,
                "per_scored_endpoint_actual_alignment_required": True,
                "future_raw_bins_relative_to_prediction_target": {
                    "per_endpoint_future_bin_count_sha256": _sha_json({
                        "role": "future_raw_bins_per_prediction_target_endpoint",
                        "counts": list(future_counts),
                    }),
                    "prediction_endpoint_count_with_future_raw_bins": sum(count > 0 for count in future_counts),
                    "maximum_future_raw_bins": max_future,
                    "causal_temporal_exposure_matched": False,
                    "bias_direction": "favors_CEBRA_accuracy",
                    "online_or_latency_equivalent_language_permitted": False,
                },
                "ordered_full_receptive_field_raw_indices_sha256": _sha_json({
                    "role": "full_receptive_field_raw_indices_by_prediction_endpoint",
                    "endpoints": list(self.cebra_scored_ordered_prediction_target_raw_bin_indices),
                    "receptive_fields": [list(item) for item in self.full_receptive_field_raw_indices],
                }),
                "checked_prediction_endpoint_count": len(self.full_receptive_field_raw_indices),
                "receptive_field_width_min": min(len(item) for item in self.full_receptive_field_raw_indices),
                "receptive_field_width_max": max(len(item) for item in self.full_receptive_field_raw_indices),
                "receptive_field_containment_violation_count": 0,
            },
            "t4_target_float32_raw_bytes_sha256": self.t4_target_float32_raw_bytes_sha256,
            "cebra_target_float32_raw_bytes_sha256": self.cebra_target_float32_raw_bytes_sha256,
            "t4_reference_query_identity_sha256": self.t4_reference_query_identity_sha256,
            "metric_implementation_authority_sha256": self.metric_implementation_authority_sha256,
            "paired_accuracy_claim_permitted_by_this_boundary_proof": True,
            "target_arrays_opened_by_this_verifier": False,
        }
        return payload | {"proof_sha256": _sha_json(payload)}


def frozen_readout_authority_plan(dataset: str, view: str | None = None) -> dict[str, Any]:
    """Freeze all readout roles; a successor has no free `may_fit_on` field."""
    dataset, view = base.validate_scope(dataset, view)
    common = {
        "dataset": dataset,
        "view": view,
        "mandatory_decoders": list(base.SUPPORTED_DECODERS),
        "target_query_neural_or_labels_may_fit_readout": False,
        "target_support_dense_labels_enter_cebra_encoder_fit": True,
        "runtime_outcome_selection_permitted": False,
        "model_arm_roles": live.MODEL_ARM_ROLES,
        "temporal_exposure_policy": {
            "standard_joint_cebra_actual_model_offset_authority_required": True,
            "default_offset10_model_offset": {"past_raw_bins": 5, "future_raw_bins": 5},
            "causal_temporal_exposure_matched": False,
            "bias_direction": "favors_CEBRA_accuracy",
            "online_or_latency_equivalent_language_permitted": False,
            "causal_alignment_sensitivity": {
                "status": "PREDECLARED_NOT_IMPLEMENTED__ARCHITECTURE_AND_ALIGNMENT_MUST_FREEZE_BEFORE_SCORES",
                "posthoc_embedding_shift_permitted": False,
                "may_replace_standard_joint_behavior_headline": False,
            },
        },
    }
    routes = {
        "source_only_consumer_mechanism_alignment": {
            "readout_fit_scope": "source_fit_only",
            "target_support_dense_labels_in_readout_fit": False,
            "required_authorities": [
                "source_readout_embedding_identity_authority_sha256",
                "source_only_linear_geometry_selection_sha256",
                "source_only_knn_geometry_selection_sha256",
            ],
            "scientific_role": "mandatory_mechanism_alignment_not_accuracy_headline",
        },
        "target_support_only_standard_cebra_accuracy": {
            "readout_fit_scope": "target_support_only",
            "target_support_dense_labels_in_readout_fit": True,
            "required_authorities": [
                "target_support_only_readout_embedding_scaler_sha256",
                "target_support_only_linear_readout_authority_sha256",
                "target_support_only_knn_readout_authority_sha256",
                "source_only_linear_geometry_selection_sha256",
                "source_only_knn_geometry_selection_sha256",
            ],
            "scientific_role": "frozen_accuracy_table_headline_with_extra_target_estimator_declared",
        },
        "source_plus_target_support_hybrid_sensitivity": {
            "readout_fit_scope": "source_fit_plus_target_support",
            "target_support_dense_labels_in_readout_fit": True,
            "required_authorities": [
                "hybrid_readout_embedding_scaler_sha256",
                "hybrid_linear_readout_authority_sha256",
                "hybrid_knn_readout_authority_sha256",
                "source_only_linear_geometry_selection_sha256",
                "source_only_knn_geometry_selection_sha256",
            ],
            "scientific_role": "mandatory_sensitivity_not_upper_bound_not_headline",
        },
    }
    return {
        "schema": READOUT_AUTHORITY_PLAN_SCHEMA,
        **common,
        "readout_routes": routes,
        "all_routes_mandatory_once_score_authorised": True,
        "target_stage_authorities_materialized": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "score_emitted": False,
    }


def build_no_data_target_query_scaffold(
    *,
    dataset: str,
    view: str | None,
    outer_fold_id: str,
    target_session_id: object,
    source_authority_sha256s: Mapping[str, str],
    pointer_proposal_payload: Mapping[str, Any],
    root_audit_attestation_sha256: str,
    proposed_target_path: object | None = None,
    proposed_target_discovery: object | None = None,
    execution_requested: bool = False,
) -> dict[str, Any]:
    """Create only an in-memory no-data target plan; execution always fails closed."""
    # Scope must be resolved before inspecting even a hostile target object.
    dataset, view = base.validate_scope(dataset, view)
    require(proposed_target_path is None, "target/query scaffold accepts no target data path")
    require(proposed_target_discovery is None, "target/query scaffold accepts no target discovery callable")
    require(execution_requested is False, "target/query scaffold has no execution mode")
    target_id = _opaque_target_session_id(dataset, target_session_id)
    source_authorities = _require_authority_sha_map(source_authority_sha256s)
    pointer_binding = bind_root_audited_pointer_proposal(
        dataset=dataset,
        view=view,
        proposal_payload=pointer_proposal_payload,
        root_audit_attestation_sha256=root_audit_attestation_sha256,
    )
    split = TargetSupportQueryTrialWindowContract(
        dataset=dataset,
        view=view,
        outer_fold_id=outer_fold_id,
        target_session_id=target_id,
    ).as_dict()
    return {
        "schema": TARGET_QUERY_SCAFFOLD_SCHEMA,
        "status": "NO_DATA_CPU_ONLY_SCAFFOLD__NOT_LIVE_AUTHORITY__NOT_EXECUTABLE",
        "dataset": dataset,
        "view": view,
        "outer_fold_id": outer_fold_id,
        "target_session_id": target_id,
        "source_authority_sha256s": source_authorities,
        "pointer_proposal_binding": pointer_binding,
        "official_metric_pointer_pair_required_before_target_discovery": True,
        "target_support_query_trial_window_contract": split,
        "readout_authority_plan": frozen_readout_authority_plan(dataset, view),
        "future_paired_claim_requires": [
            "root_minted_immutable_metric_pointer_body_and_sidecar",
            "per_fold_immutable_source_authorities_and_source_roster_binding",
            "target_support_and_target_query_raw_coverage_indices",
            "ordered_target_byte_and_receptive_field_proof",
            "subject_M_exact_V9_commit_and_runtime_trace_lineage__query_after_rewarded_trial_50",
            "exact_parent_two_output_R2_implementation_and_reduction_binding",
            "all_model_arms_readout_routes_and_decoders_without_outcome_selection",
        ],
        "target_data_discovery_permitted": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_discovery_permitted": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_trained": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_receipt_minted": False,
    }


def build_source_only_target_adapter_preflight(
    *,
    dataset: str,
    view: str | None,
    outer_fold_id: str,
    target_session_id: object,
    source_authority_sha256s: Mapping[str, str],
    official_metric_pointer_pair: live.ExplicitSealedReceiptPair | None,
    rt_15fold_source_authority_plan: Mapping[str, Any] | None = None,
    proposed_target_path: object | None = None,
    proposed_target_discovery: object | None = None,
    execution_requested: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Validate source code + an official pointer *before* any target discovery.

    This is intentionally a source-only CPU preflight, not a target adapter
    execution.  Most importantly, it validates the official immutable pointer
    pair before it even coerces the opaque target session ID.  Thus absence of a
    root-minted pointer fails closed with zero target discovery or target-path
    handling.
    """
    dataset, view = base.validate_scope(dataset, view)
    require(proposed_target_path is None, "target adapter preflight accepts no target data path")
    require(proposed_target_discovery is None, "target adapter preflight accepts no target discovery callable")
    require(execution_requested is False, "target adapter preflight has no execution mode")
    require(device == "cpu", "target adapter preflight is CPU-only")
    require(official_metric_pointer_pair is not None,
            "official root-audited immutable metric pointer pair is required before target discovery")
    source_authorities = _require_authority_sha_map(source_authority_sha256s)
    pointer_validation = metric_pointer.validate_root_audited_metric_pointer_pair(
        dataset=dataset, view=view, pointer_pair=official_metric_pointer_pair,
    )
    # Only after all non-target authority gates passed may an opaque session ID
    # be checked.  This still does not resolve a path or open a target record.
    target_id = _opaque_target_session_id(dataset, target_session_id)
    rt_source_plan_binding: dict[str, Any] | None = None
    if dataset == "rt":
        require(rt_15fold_source_authority_plan is not None,
                "RT target adapter preflight requires the full 15-fold source-authority plan")
        rt_plan = source_adapter.validate_rt_15fold_source_authority_plan(rt_15fold_source_authority_plan)
        matches = [item for item in rt_plan["outer_folds"] if item["outer_fold_id"] == outer_fold_id]
        require(len(matches) == 1, "RT outer fold missing or ambiguous in full source-authority plan")
        fold = matches[0]
        require(fold["opaque_held_out_target_session_id"] == target_id,
                "RT target ID does not equal the full source-authority plan held-out session")
        rt_source_plan_binding = {
            "rt_15fold_source_authority_plan_sha256": rt_plan["rt_15fold_source_authority_plan_sha256"],
            "outer_fold_source_session_roster_sha256": fold["source_session_roster_sha256"],
            "outer_fold_source_session_count": fold["source_session_count"],
            "outer_fold_held_out_target_data_opened": fold["held_out_target_data_opened"],
            "per_fold_development_source_authority_bundles_required": True,
        }
    else:
        require(rt_15fold_source_authority_plan is None,
                "subject-M target adapter preflight may not accept an RT source-authority plan")
    split = TargetSupportQueryTrialWindowContract(
        dataset=dataset, view=view, outer_fold_id=outer_fold_id, target_session_id=target_id,
    ).as_dict()
    loader_semantics = live.build_sealed_loader_semantics_contract(dataset, view)
    return {
        "schema": TARGET_ADAPTER_PREFLIGHT_SCHEMA,
        "status": "OFFICIAL_POINTER_VALIDATED__SOURCE_ONLY_CPU_PREFLIGHT__TARGET_DISCOVERY_STILL_FORBIDDEN",
        "dataset": dataset,
        "view": view,
        "outer_fold_id": outer_fold_id,
        "target_session_id": target_id,
        "source_authority_sha256s": source_authorities,
        "official_metric_pointer_validation": pointer_validation,
        "rt_15fold_source_authority_plan_binding": rt_source_plan_binding,
        "loader_semantics": loader_semantics,
        "target_support_query_trial_window_contract": split,
        "continuous_prefix_policy_preserved": True,
        "causal_endpoint_policy": "valid_window_start_plus_49",
        "standard_offset10_temporal_exposure": {
            "causal_temporal_exposure_matched": False,
            "bias_direction": "favors_CEBRA_accuracy",
            "online_or_latency_equivalent_language_permitted": False,
        },
        "accuracy_headline_readout": "target_support_only_standard_cebra_accuracy",
        "posthoc_geometry_seed_or_readout_selection_permitted": False,
        "target_data_discovery_permitted": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_trained": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_execution_receipt_minted": False,
    }
