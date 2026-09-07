"""Fail-closed Track-B v2 Subject-M one-cell live-executor successor.

This additive module is deliberately a *reviewable execution contract*, not a
live executor.  A cell is exactly one ``(view, canonical external fold,
subject-M session, CEBRA seed)``.  It creates no output, opens no target/NWB or
NPZ, imports no CEBRA/Torch/sklearn, fits no encoder or readout, emits no
metric, uses no GPU, and mints no authority.

The module internally rebuilds the canonical Subject-M materializer and the
fixed d8/it250 GPU-cost gate through the existing development-executor
scaffold.  Thus it has no caller-supplied target path, asset SHA, normalizer,
checkpoint, output root, or source-authority escape hatch.  A later root
authorised implementation may use the declared receipt topology only after an
independent review of this successor and the canonical engineering-cost pair.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_subject_m_development_executor as development_executor


ONE_CELL_PREFLIGHT_SCHEMA = "track_b_v2_subject_m_one_cell_preflight_v1"
ONE_CELL_FUTURE_EXECUTION_CHAIN_SCHEMA = "track_b_v2_subject_m_one_cell_future_execution_chain_v1"
ONE_CELL_RECEIPT_TOPOLOGY_SCHEMA = "track_b_v2_subject_m_one_cell_future_receipt_topology_v1"
ONE_CELL_SCORER_INTERFACE_SCHEMA = "track_b_v2_subject_m_one_cell_scorer_interface_v1"
ONE_CELL_AGGREGATE_INTERFACE_SCHEMA = "track_b_v2_subject_m_view_future_aggregate_interface_v1"
ONE_CELL_SYNTHETIC_SCORE_LAYOUT_SCHEMA = "track_b_v2_subject_m_synthetic_one_cell_score_layout_v1"

_TARGET = re.compile(r"sub-M_ses-CO-(\d{8})")
_SEEDS = (42, 43, 44)
_READOUT_ROUTES = (
    "source_only_consumer_mechanism_alignment",
    "target_support_only_standard_cebra_accuracy",
    "source_plus_target_support_hybrid_sensitivity",
)
_DECODERS = ("linear_ridge", "knn_cosine_k3")
_MODEL_ARMS = ("cebra_joint_behavior", "cebra_frozen_source_adapt", "cebra_adapt_unaligned")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_subject_m_one_cell_successor.py"


class TrackBV2SubjectMOneCellSuccessorError(development_executor.TrackBV2SubjectMDevelopmentExecutorError):
    """Raised before a one-cell plan could resolve or consume target data."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2SubjectMOneCellSuccessorError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _source_binding(path: Path, *, label: str) -> dict[str, Any]:
    """Bind reviewed implementation bytes; this never resolves data paths."""
    raw = Path(path).read_bytes()
    return {"label": label, "path": str(Path(path).absolute()), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


@dataclass(frozen=True)
class SubjectMOneCellIdentity:
    """The only legal identity grammar for a future Subject-M executor cell."""

    view: str
    outer_fold_id: str
    target_session_id: object
    cebra_seed: int

    def __post_init__(self) -> None:
        dataset, view = base.validate_scope("subject_m", self.view)
        require(dataset == "subject_m" and view in ("sua", "pseudo_mua"),
                "one-cell successor only permits subject-M SUA/pMUA")
        require(isinstance(self.target_session_id, str) and self.target_session_id == self.target_session_id.strip(),
                "one-cell target session must be a trimmed opaque string")
        target_match = _TARGET.fullmatch(self.target_session_id)
        require(target_match is not None, "one-cell target session violates canonical subject-M grammar")
        expected_fold = f"subject_m_{view}_external_target_{target_match.group(1)}"
        require(self.outer_fold_id == expected_fold,
                "one-cell outer fold must be the unique canonical view/date target fold")
        require(type(self.cebra_seed) is int and self.cebra_seed in _SEEDS,
                f"one-cell CEBRA seed must be one of {_SEEDS}")
        object.__setattr__(self, "view", str(view))
        object.__setattr__(self, "target_session_id", str(self.target_session_id))

    @property
    def cell_id(self) -> str:
        return f"subject_m__{self.view}__{self.outer_fold_id}__seed{self.cebra_seed}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": "subject_m", "view": self.view, "outer_fold_id": self.outer_fold_id,
            "target_session_id": self.target_session_id, "cebra_seed": self.cebra_seed,
            "canonical_cell_id": self.cell_id,
        }


def _require_development_executor_contract(
    plan: Mapping[str, Any], *, identity: SubjectMOneCellIdentity,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Accept only the internally rebuilt materializer/cost gate contract."""
    require(plan.get("dataset") == "subject_m" and plan.get("view") == identity.view and
            plan.get("outer_fold_id") == identity.outer_fold_id and
            plan.get("target_session_id") == identity.target_session_id,
            "development executor scope/cell identity drift")
    binding = plan.get("canonical_materializer_binding")
    gate = plan.get("fixed_gpu_cost_gate")
    topology = plan.get("immutable_future_receipt_topology")
    require(isinstance(binding, Mapping) and isinstance(gate, Mapping) and isinstance(topology, Mapping),
            "development executor lacks canonical materializer/cost/topology binding")
    require(_valid_sha(binding.get("canonical_development_target_authority_sha256")) and
            _valid_sha(binding.get("canonical_metric_pointer_body_sha256")) and
            _valid_sha(binding.get("fixed_geometry_contract_sha256")) and
            binding.get("target_asset_ledger_is_a2_v2_verified") is True and
            binding.get("caller_supplied_authority_or_asset_sha_permitted") is False,
            "one-cell successor requires an internally rebuilt canonical materializer authority")
    fixed = plan.get("future_target_loader_and_encoder_contract")
    require(isinstance(fixed, Mapping) and fixed.get("fixed_encoder_geometry") == {
        "output_dimension": 8, "iterations": 10_000, "source_or_target_geometry_selection_performed": False,
    }, "one-cell successor fixed d8-it10000 geometry drift")
    offset = plan.get("exact_offset10_endpoint_and_receptive_field_contract")
    require(isinstance(offset, Mapping) and offset.get("exact_per_endpoint_raw_receptive_field") ==
            "range(endpoint-5, endpoint+5)" and offset.get("receptive_field_width_raw_bins") == 10 and
            offset.get("strictly_future_raw_bins_after_endpoint") == 4 and
            offset.get("causal_temporal_exposure_matched") is False and
            offset.get("bias_direction") == "favors_CEBRA_accuracy",
            "one-cell successor offset10 noncausal endpoint contract drift")
    per_seed = topology.get("per_cebra_seed")
    require(isinstance(per_seed, Mapping) and tuple(per_seed.get("seed_set", ())) == _SEEDS and
            tuple(per_seed.get("decoders", ())) == _DECODERS and
            tuple(per_seed.get("model_arms", ())) == _MODEL_ARMS,
            "one-cell successor execution roster drift")
    encoder = per_seed.get("encoder_bundle")
    routes = per_seed.get("readout_routes")
    require(isinstance(encoder, Mapping) and isinstance(routes, Mapping) and set(routes) == set(_READOUT_ROUTES),
            "one-cell successor encoder/readout topology drift")
    require(encoder.get("geometry") == "d8-it10000" and
            encoder.get("legal_multisession_target_serviceability_required") is True and
            encoder.get("never_transform_unfitted_unseen_target_session") is True and
            encoder.get("target_support_neural_enters_encoder_fit") is True and
            encoder.get("target_support_dense_velocity_enters_encoder_fit") is True and
            encoder.get("target_query_neural_or_labels_enter_fit") is False,
            "one-cell successor legal joint target encoder contract drift")
    source_only = routes["source_only_consumer_mechanism_alignment"]
    standard = routes["target_support_only_standard_cebra_accuracy"]
    hybrid = routes["source_plus_target_support_hybrid_sensitivity"]
    require(source_only.get("readout_fit_scope") == "source_fit_only" and
            source_only.get("encoder_fit_scope") == "legal_joint_multisession_source_plus_target_support" and
            source_only.get("target_support_dense_labels_in_readout_fit") is False and
            source_only.get("unfitted_target_transform_or_source_only_encoder_permitted") is False,
            "source-only consumer must constrain only the readout, never target encoder serviceability")
    require(standard.get("readout_fit_scope") == "target_support_only" and
            standard.get("target_support_dense_labels_in_readout_fit") is True and
            hybrid.get("readout_fit_scope") == "source_fit_plus_target_support" and
            hybrid.get("target_support_dense_labels_in_readout_fit") is True and
            hybrid.get("may_replace_headline_or_mechanism_route") is False,
            "one-cell standard/hybrid readout role drift")
    return binding, gate, topology


def _future_execution_chain(
    *, identity: SubjectMOneCellIdentity, development_plan: Mapping[str, Any],
    materializer_binding: Mapping[str, Any], cost_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Declare the only legal future transition sequence without executing it."""
    lineage = development_plan.get("exact_t4_target_byte_lineage")
    require(isinstance(lineage, Mapping), "development executor target-byte lineage missing")
    snapshot = development_plan.get("immutable_future_receipt_topology", {}).get("target_parser_byte_consumption")
    require(isinstance(snapshot, Mapping), "development executor snapshot boundary missing")
    return {
        "schema": ONE_CELL_FUTURE_EXECUTION_CHAIN_SCHEMA,
        "status": "FUTURE_ONLY__ROOT_REVIEW_AND_EXPLICIT_EXECUTOR_AUTHORIZATION_REQUIRED",
        "cell": identity.as_dict(),
        "pre_target_gates_in_exact_order": [
            "fresh_canonical_one_cell_output_body_and_sidecar_pair__O_EXCL__0444__before_target_resolution",
            "rebuild_canonical_subject_m_materializer_authority_before_asset_id_or_path_use",
            "revalidate_canonical_fixed_d8it250_GPU_cost_body_sidecar_and_live_closure",
            "bind_exact_V9_T4_commit_runtime_targets_and_metric_semantics_for_this_asset_view_seed",
        ],
        "target_byte_and_parser_boundary": {
            "only_expected_path_sha_size_from_internal_A2_materializer_ledger": True,
            "caller_target_path_or_asset_sha_permitted": False,
            "snapshot_factory": {
                "module": str(Path(development_executor.__file__).absolute()),
                "sha256": hashlib.sha256(Path(development_executor.__file__).read_bytes()).hexdigest(),
                "callable": "open_verified_target_asset_private_snapshot",
            },
            "source_open": snapshot.get("source_open"),
            "private_snapshot": snapshot.get("private_snapshot"),
            "parser_handoff": snapshot.get("parser_handoff"),
            "ordinary_source_or_snapshot_pathname_reopen_permitted": False,
            "pathname_pre_and_post_hash_alone_is_sufficient": False,
        },
        "target_support_and_query": {
            "support": "one_continuous_chronological_prefix_through_stop_of_rewarded_trial_50",
            "query": "rewarded_trials_strictly_after_50_only",
            "prediction_target_timestamp": "valid_window_start_plus_49",
            "receptive_field": "range(endpoint-5, endpoint+5)",
            "strictly_future_raw_bins": 4,
            "causal_temporal_exposure_matched": False,
            "bias_direction": "favors_CEBRA_accuracy",
            "all_CEBRA_outputs_must_equal_exact_T4_ordered_float32_target_bytes": True,
        },
        "joint_encoder": {
            "fixed_geometry": "d8-it10000",
            "source_sessions": 27,
            "target_support_session": identity.target_session_id,
            "target_support_neural_and_dense_velocity_in_encoder_fit": True,
            "target_query_neural_or_labels_in_encoder_fit": False,
            "unseen_target_transform_without_legal_joint_fit_permitted": False,
            "same_fitted_encoder_checkpoint_and_embeddings_for_all_routes_and_decoders_within_model_arm": True,
            "readout_or_decoder_specific_encoder_retraining_permitted": False,
        },
        "score_routes": {
            "model_arms": list(_MODEL_ARMS),
            "routes": list(_READOUT_ROUTES),
            "decoders": list(_DECODERS),
            "headline": "cebra_joint_behavior__target_support_only_standard_cebra_accuracy",
            "source_only_consumer": "readout_source_fit_only__encoder_remains_legal_joint_target_serviceable",
            "hybrid": "mandatory_sensitivity__not_headline__not_upper_bound",
            "posthoc_best_seed_route_decoder_or_arm_selection_permitted": False,
        },
        "reference_lineage": {
            "canonical_materializer_sha256": materializer_binding["canonical_development_target_authority_sha256"],
            "canonical_metric_pointer_sha256": materializer_binding["canonical_metric_pointer_body_sha256"],
            "cebra_seed": identity.cebra_seed,
            "sealed_T4_reference_seed_set": list(_SEEDS),
            "one_to_one_CEBRA_to_T4_stochastic_pairing_claim_permitted": False,
            "runtime_receipt_target_bytes_and_endpoint_order_required": True,
            "lineage_contract": lineage,
        },
        "cost_gate": {
            "canonical_body_sha256": cost_gate.get("canonical_body_sha256", "REQUIRED"),
            "canonical_sidecar_sha256": cost_gate.get("canonical_sidecar_sha256", "REQUIRED"),
            "fixed_final_geometry": {"output_dimension": 8, "iterations": 10_000},
            "cost_smoke_geometry": {"output_dimension": 8, "iterations": 250},
            "cost_gate_is_execution_authority": False,
        },
        "target_data_opened": False,
        "cebra_imported": False,
        "cebra_trained": False,
        "readout_fit_called": False,
        "score_emitted": False,
        "gpu_used": False,
        "official_execution_receipt_minted": False,
    }


def _future_receipt_topology(*, identity: SubjectMOneCellIdentity) -> dict[str, Any]:
    """Predeclare immutable output roles without accepting an output root or writing."""
    per_arm = {
        arm: {
            "one_encoder_bundle_receipt": True,
            "encoder_may_not_be_refit_for_route_or_decoder": True,
            "score_receipts": [
                {"readout_route": route, "decoder": decoder,
                 "exactly_one_terminal_receipt": True, "metric_name": "two_output_R2__FUTURE_ONLY"}
                for route in _READOUT_ROUTES for decoder in _DECODERS
            ],
        }
        for arm in _MODEL_ARMS
    }
    return {
        "schema": ONE_CELL_RECEIPT_TOPOLOGY_SCHEMA,
        "status": "FUTURE_ONLY__NO_OUTPUT_PATH_ACCEPTED_OR_CREATED",
        "cell": identity.as_dict(),
        "future_terminal_preflight_receipt_body_and_sidecar_sha256": "REQUIRED__BIND_FULL_WRITTEN_PREFLIGHT_PAIR",
        "writer_invariants": {
            "canonical_output_root_from_root_authorized_successor_only": True,
            "caller_output_path_permitted": False,
            "body_creation": "O_EXCL_ONLY",
            "sidecar_creation": "O_EXCL_ONLY_WITH_OWNED_BODY_ROLLBACK_ON_COLLISION",
            "body_and_sidecar_mode": "0444",
            "parent_symlink_forbidden": True,
            "body_and_sidecar_fresh_before_target_resolution": True,
        },
        "per_cell_once": [
            "immutable_one_cell_preflight_pair",
            "verified_private_target_snapshot_and_held_parser_fd_consumption_receipt",
            "continuous_M50_support_strict_post_M_query_and_offset10_endpoint_receipt",
            "exact_V9_T4_target_byte_order_metric_semantics_receipt",
            "cross_view_behavior_order_and_pmua_pooling_replay_receipt",
        ],
        "per_model_arm": per_arm,
        "aggregate_admission": {
            "only_immutable_terminal_one_cell_pairs": True,
            "same_cell_id_preflight_sha256_and_target_byte_authority_required": True,
            "session_then_CEBRA_seed_aggregation": True,
            "no_best_seed_route_decoder_or_model_arm_substitution": True,
        },
        "target_data_opened": False,
        "score_emitted": False,
        "official_execution_receipt_minted": False,
    }


def build_subject_m_one_cell_preflight(
    *, view: str, outer_fold_id: str, target_session_id: object, cebra_seed: int,
    execution_requested: bool = False, device: str = "cpu",
) -> dict[str, Any]:
    """Build a canonical one-cell preflight with no target array or model I/O.

    There are deliberately no caller authority, target-path, target-array,
    checkpoint, output-root, score, or CEBRA parameters.  The only source of
    target identity is the existing materializer rebuilt inside the development
    executor scaffold.
    """
    identity = SubjectMOneCellIdentity(
        view=view, outer_fold_id=outer_fold_id, target_session_id=target_session_id, cebra_seed=cebra_seed,
    )
    require(execution_requested is False, "one-cell successor has no execute mode")
    require(device == "cpu", "one-cell successor preflight is no-data/CPU only")
    try:
        development_plan = development_executor.build_subject_m_development_executor_dry_plan(
            dataset="subject_m", view=identity.view, outer_fold_id=identity.outer_fold_id,
            target_session_id=identity.target_session_id,
        )
    except development_executor.TrackBV2SubjectMDevelopmentExecutorError as exc:
        raise TrackBV2SubjectMOneCellSuccessorError(str(exc)) from exc
    materializer_binding, cost_gate, _topology = _require_development_executor_contract(
        development_plan, identity=identity,
    )
    cost_valid = cost_gate.get("status") == (
        "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION"
    )
    core = _source_binding(Path(__file__), label="subject_m_one_cell_successor_core")
    runner = _source_binding(_RUNNER, label="subject_m_one_cell_successor_runner") if _RUNNER.exists() else {
        "label": "subject_m_one_cell_successor_runner", "path": str(_RUNNER), "sha256": "NOT_YET_CREATED", "bytes": 0,
    }
    base_payload = {
        "schema": ONE_CELL_PREFLIGHT_SCHEMA,
        "status": (
            "NO_GO__CANONICAL_FIXED_D8IT250_GPU_COST_RECEIPT_REQUIRED"
            if not cost_valid else
            "COST_GATE_VALID__ONE_CELL_EXECUTOR_CONTRACT_READY_FOR_ROOT_REVIEW_ONLY"
        ),
        "cell": identity.as_dict(),
        "canonical_materializer_binding": dict(materializer_binding),
        "fixed_d8it250_gpu_cost_gate": dict(cost_gate),
        "fixed_final_geometry": {
            "output_dimension": 8, "iterations": 10_000,
            "linear_ridge_normalized_lambda": 0.01, "cosine_knn_k": 3,
            "source_or_target_geometry_selection_performed": False,
        },
        "implementation_closure": {"one_cell_core": core, "one_cell_runner": runner},
        "root_authorized_live_execution_permitted": False,
        "target_data_discovery_permitted": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_trained": False,
        "readout_fit_called": False,
        "score_emitted": False,
        "gpu_used": False,
        "official_execution_receipt_minted": False,
    }
    draft = base_payload | {
        "future_execution_chain": _future_execution_chain(
            identity=identity, development_plan=development_plan,
            materializer_binding=materializer_binding, cost_gate=cost_gate,
        ),
        "future_receipt_topology": _future_receipt_topology(identity=identity),
    }
    return draft | {"preflight_payload_sha256": _sha_json(draft)}


def build_subject_m_one_cell_scorer_interface(
    *, view: str, outer_fold_id: str, target_session_id: object, cebra_seed: int,
) -> dict[str, Any]:
    """Render future scorer inputs, internally rebuilding the canonical preflight."""
    preflight = build_subject_m_one_cell_preflight(
        view=view, outer_fold_id=outer_fold_id, target_session_id=target_session_id, cebra_seed=cebra_seed,
    )
    return {
        "schema": ONE_CELL_SCORER_INTERFACE_SCHEMA,
        "status": "FUTURE_ONLY__SCORER_NOT_IMPLEMENTED__NO_TARGET_OR_METRIC_IO",
        "cell": preflight["cell"],
        "preflight_payload_sha256": preflight["preflight_payload_sha256"],
        "requires_preflight_cost_gate_status": preflight["fixed_d8it250_gpu_cost_gate"]["status"],
        "exact_input_requirements": {
            "same_model_arm_encoder_bundle_services_each_route_decoder": True,
            "routes": list(_READOUT_ROUTES), "decoders": list(_DECODERS), "model_arms": list(_MODEL_ARMS),
            "ordered_T4_and_CEBRA_target_float32_bytes_must_match": True,
            "exact_endpoint_order_and_parent_two_output_R2_semantics_required": True,
            "target_query_bins_are_not_independent_aggregate_samples": True,
            "aggregate_session_then_seed": True,
        },
        "target_data_opened": False,
        "cebra_imported": False,
        "readout_fit_called": False,
        "score_emitted": False,
        "official_execution_receipt_minted": False,
    }


def build_subject_m_view_future_aggregate_interface(*, view: str) -> dict[str, Any]:
    """Declare aggregate admission without accepting receipts or metrics now."""
    _dataset, checked_view = base.validate_scope("subject_m", view)
    return {
        "schema": ONE_CELL_AGGREGATE_INTERFACE_SCHEMA,
        "status": "FUTURE_ONLY__NO_RECEIPT_OR_METRIC_INPUT_ACCEPTED",
        "dataset": "subject_m", "view": checked_view,
        "expected_external_target_cell_count": 15,
        "cebra_seed_set": list(_SEEDS),
        "required_one_cell_receipt_count_per_model_arm_route_decoder": 45,
        "model_arms": list(_MODEL_ARMS), "routes": list(_READOUT_ROUTES), "decoders": list(_DECODERS),
        "admission_requirements": {
            "all_cells_bind_current_canonical_materializer_and_fixed_cost_pairs": True,
            "all_cells_use_one_predeclared_seed_without_best_seed_selection": True,
            "aggregate_session_then_seed_not_bins": True,
            "paired_claim_requires_exact_T4_target_byte_endpoint_metric_parity": True,
            "sua_pmua_cross_view_behavior_and_endpoint_authorities_must_match": True,
        },
        "metrics_or_receipts_consumed": False,
        "target_data_opened": False,
        "score_emitted": False,
        "official_execution_receipt_minted": False,
    }


def validate_synthetic_one_cell_score_layout(
    *, preflight: Mapping[str, Any], model_arm: str, score_rows: Sequence[Mapping[str, Any]],
    synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """Validate a no-metric synthetic scorer layout, never a real score receipt.

    The function exists solely to make the six shared-encoder route/decoder
    obligations adversarially testable before any target scorer exists.  It
    rejects R²/accuracy values so synthetic fixtures cannot become paper data.
    """
    require(synthetic_fixture is True, "score layout validation is synthetic-fixture only")
    require(isinstance(preflight, Mapping) and preflight.get("schema") == ONE_CELL_PREFLIGHT_SCHEMA,
            "synthetic score layout requires one-cell preflight")
    declared = preflight.get("preflight_payload_sha256")
    bare_preflight = dict(preflight)
    bare_preflight.pop("preflight_payload_sha256", None)
    require(_valid_sha(declared) and declared == _sha_json(bare_preflight),
            "synthetic score layout preflight body SHA drift")
    for flag in ("target_data_opened", "cebra_imported", "cebra_trained", "readout_fit_called",
                 "score_emitted", "gpu_used", "official_execution_receipt_minted"):
        require(preflight.get(flag) is False, f"synthetic score layout preflight prohibited flag drift: {flag}")
    require(model_arm in _MODEL_ARMS, "synthetic score layout model arm undeclared")
    require(isinstance(score_rows, Sequence) and not isinstance(score_rows, (str, bytes)),
            "synthetic score layout rows invalid")
    expected = {(route, decoder) for route in _READOUT_ROUTES for decoder in _DECODERS}
    observed: set[tuple[str, str]] = set()
    encoder_ids: set[str] = set()
    target_shas: set[str] = set()
    for row in score_rows:
        require(isinstance(row, Mapping), "synthetic score layout row malformed")
        require(set(row) == {
            "readout_route", "decoder", "encoder_bundle_identity_sha256",
            "ordered_target_float32_raw_bytes_sha256", "synthetic_only",
        }, "synthetic score layout row field set drift")
        pair = (row.get("readout_route"), row.get("decoder"))
        require(pair in expected and pair not in observed, "synthetic score layout route/decoder missing or duplicate")
        encoder_sha = row.get("encoder_bundle_identity_sha256")
        target_sha = row.get("ordered_target_float32_raw_bytes_sha256")
        require(_valid_sha(encoder_sha) and _valid_sha(target_sha) and row.get("synthetic_only") is True,
                "synthetic score layout provenance drift")
        observed.add(pair)
        encoder_ids.add(str(encoder_sha))
        target_shas.add(str(target_sha))
    require(observed == expected, "synthetic score layout must contain all three routes by two decoders")
    require(len(encoder_ids) == 1, "synthetic score layout illegally refit encoder by route or decoder")
    require(len(target_shas) == 1, "synthetic score layout target byte authority drift")
    return {
        "schema": ONE_CELL_SYNTHETIC_SCORE_LAYOUT_SCHEMA,
        "status": "SYNTHETIC_LAYOUT_VALIDATED__NO_METRIC__NO_TARGET_EXECUTION",
        "cell": preflight["cell"], "model_arm": model_arm,
        "route_decoder_count": len(observed),
        "shared_encoder_bundle_identity_sha256": next(iter(encoder_ids)),
        "ordered_target_float32_raw_bytes_sha256": next(iter(target_shas)),
        "target_data_opened": False, "score_emitted": False,
    }


def refuse_subject_m_one_cell_execution(**kwargs: Any) -> None:
    """Explicit tripwire: the successor emits plans only in this turn."""
    build_subject_m_one_cell_preflight(**kwargs)
    raise TrackBV2SubjectMOneCellSuccessorError(
        "subject-M one-cell successor is a no-target review contract; root-authorised live execution is not implemented"
    )
