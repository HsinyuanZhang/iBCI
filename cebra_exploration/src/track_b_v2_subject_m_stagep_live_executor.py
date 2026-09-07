"""Additive, fail-closed paired Stage-P live-executor plan.

This module deliberately does *not* perform a Subject-M target operation.  It
does not import NumPy, Torch, CEBRA, pynwb, or a target parser.  Instead it
freezes the only admissible future execution topology for the predeclared
development pilot: SUA first and its already-predeclared paired pMUA cell
second.  The module is intentionally separate from the Stage-P runtime core:
the latter remains the sole owner of admission validation and of the future
numerical primitives.

There is no public caller choice of date, seed, target path, output root,
device, geometry, decoder, or readout.  An ``--execute`` caller can only
rebuild the exact current Stage-P admission and then reaches an unconditional
no-target tripwire.  Real producers are represented below as explicit,
unimplemented interfaces so no synthetic layout object can accidentally
become a live parser, fit, score, or receipt publisher.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_subject_m_stagep_runtime as runtime


LIVE_EXECUTOR_SCHEMA = "track_b_v2_subject_m_stagep_paired_live_executor_plan_v1"
LIVE_EXECUTOR_STATUS = "NO_GO__ADDITIVE_PLAN_ONLY__REAL_PRODUCERS_AND_INDEPENDENT_REVIEW_REQUIRED"
LIVE_ADMISSION_STATUS = "STAGEP_LIVE_ADMISSION_VALID__SEPARATE_ROOT_REVIEWED_EXECUTION_LAUNCH_REQUIRED"
PAIRED_VIEW_ORDER = ("sua", "pseudo_mua")
PHYSICAL_GPU_INDEX = 1
LOGICAL_CUDA_DEVICE = "cuda:0"
PRIVATE_SNAPSHOT_RETENTION = (
    "private_snapshot_is_0600_during_same_fd_parse_then_0444_only_if_retained_for_reproducibility; "
    "never_publish_as_result; unlink_after_target_receipt_and_all_derived_array_hashes_are_independently_verified; "
    "failed_or_aborted_cells_must_cleanup_before_any_completion_receipt"
)
PAIRED_STATE_SEQUENCE = (
    "PAIR_CAPABILITY_BOUND",
    "SUA_START_PUBLISHED",
    "SUA_STRICT27_SOURCE_MATERIALIZED",
    "SUA_TARGET_M50_QUERY_MATERIALIZED",
    "SUA_PRIMARY_JOINT_ENCODER_FIT",
    "SUA_ARTIFACTS_PERSISTED_AND_RELOADED",
    "SUA_SIX_SCORES_PUBLISHED",
    "SUA_TERMINAL_PUBLISHED",
    "PMUA_START_PUBLISHED",
    "PMUA_STRICT27_SOURCE_MATERIALIZED",
    "PMUA_TARGET_M50_QUERY_AND_POOLING_REPLAY_MATERIALIZED",
    "PMUA_PRIMARY_JOINT_ENCODER_FIT",
    "PMUA_ARTIFACTS_PERSISTED_AND_RELOADED",
    "PMUA_SIX_SCORES_PUBLISHED",
    "PMUA_TERMINAL_PUBLISHED",
    "PAIRED_COMPLETION_PUBLISHED",
)


class TrackBV2SubjectMStagePLiveExecutorError(runtime.TrackBV2SubjectMStagePRuntimeError):
    """Raised before a target path, ML import, GPU action, or receipt write."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2SubjectMStagePLiveExecutorError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))


@dataclass(frozen=True)
class StagePLiveExecutionCapability:
    """Opaque admission result made only by the exact predecessor builder.

    It contains no authority supplied by a future executor caller.  The
    `admission_sha256` is over the same mapping returned by
    :func:`runtime.build_stagep_live_admission`, including the two official
    preflights, root authorization, cost pair and runtime-control pair.
    """

    view: str
    cell: Mapping[str, Any]
    admission: Mapping[str, Any]
    admission_sha256: str


@dataclass(frozen=True)
class ProducerInterface:
    """One future implementation obligation; never an executable callback."""

    role: str
    required_before_target_open: bool
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    current_status: str
    forbidden_substitutes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "required_before_target_open": self.required_before_target_open,
            "consumes": list(self.consumes),
            "produces": list(self.produces),
            "current_status": self.current_status,
            "forbidden_substitutes": list(self.forbidden_substitutes),
        }


# Exactly eight producer gaps.  Keeping these as structured data makes it
# impossible for the runner to mistake a synthetic schema validator for a
# real-data implementation.
PRODUCER_INTERFACES: tuple[ProducerInterface, ...] = (
    ProducerInterface(
        "canonical_cell_root_start_and_completion_publisher",
        True,
        ("capability_bound_pair", "literal_cell_topology", "canonical_gpu_contract"),
        ("fresh_cell_root", "immutable_start_receipt", "immutable_completion_receipt"),
        "UNIMPLEMENTED__NO_WRITES_IN_THIS_SCAFFOLD",
        ("caller_output_path", "alias_or_symlink", "existing_body_or_sidecar"),
    ),
    ProducerInterface(
        "strict27_source_materializer_and_sealed_hash_verifier",
        True,
        ("sealed_view_specific_strict27_source_authority", "canonical_source_adapter"),
        ("ordered_27_source_arrays", "per_session_source_hash_proofs", "source_pMUA_pooling_replay"),
        "UNIMPLEMENTED__NO_SOURCE_NWB_OPEN_IN_THIS_SCAFFOLD",
        ("other_subM_sessions_as_source", "unverified_cached_array", "source_roster_reorder"),
    ),
    ProducerInterface(
        "held_target_private_snapshot_parser_and_materializer",
        True,
        ("A2_target_asset_ledger", "canonical_development_authority", "held_O_NOFOLLOW_fd"),
        ("continuous_M50_prefix", "strict_post_M50_query", "T4_target_byte_and_endpoint_proof", "pMUA_replay_proof"),
        "UNIMPLEMENTED__NO_TARGET_PATH_OR_NWB_OPEN_IN_THIS_SCAFFOLD",
        ("pathname_reopen", "pre_post_hash_only", "target_query_in_fit", "formal_subC_discovery"),
    ),
    ProducerInterface(
        "isolated_gpu_child_identity_and_environment_producer",
        True,
        ("cost_pair_device_identity", "canonical_CUDA_VISIBLE_DEVICES"),
        ("child_PID", "GPU_UUID_and_logical_cuda0_identity", "pre_import_environment_proof"),
        "UNIMPLEMENTED__NO_GPU_OR_TORCH_IMPORT_IN_THIS_SCAFFOLD",
        ("caller_gpu_override", "post_import_CVD_change", "logical_device_without_physical_UUID"),
    ),
    ProducerInterface(
        "independent_primary_joint_28_session_encoder_fit",
        False,
        ("27_verified_source_blocks", "held_M50_support", "fixed_d8_it10000", "canonical_gpu_child"),
        ("one_view_specific_joint_encoder", "view_specific_checkpoint_state", "source_support_query_embeddings"),
        "UNIMPLEMENTED__NO_CEBRA_IMPORT_OR_FIT_IN_THIS_SCAFFOLD",
        ("source_only_transform_of_unseen_target", "cross_view_encoder_reuse", "control_arm_checkpoint"),
    ),
    ProducerInterface(
        "checkpoint_embedding_private_persistence_and_reload_verifier",
        False,
        ("view_specific_joint_encoder", "embedding_arrays", "literal_artifact_paths"),
        ("O_EXCL_0444_checkpoint_pair", "O_EXCL_0444_embedding_pair", "same_byte_reload_proof"),
        "UNIMPLEMENTED__NO_ARTIFACT_WRITER_IN_THIS_SCAFFOLD",
        ("mutable_checkpoint_path", "model_state_values_only_digest", "shared_SUA_pMUA_artifact"),
    ),
    ProducerInterface(
        "six_route_decoder_scorer_and_torchmetrics_receipt_publisher",
        False,
        ("one_fixed_joint_encoder", "separately_cropped_blocks", "ordered_query_target_bytes"),
        ("six_immutable_score_pairs", "route_decoder_readout_state", "TorchMetrics151_CPU_float32_metric"),
        "UNIMPLEMENTED__NO_SCORE_OR_TARGET_METRIC_IN_THIS_SCAFFOLD",
        ("readout_specific_encoder_refit", "padded_embeddings", "pooled_multi_session_R2"),
    ),
    ProducerInterface(
        "capability_bound_live_validation_terminal_and_paired_completion_publisher",
        False,
        ("official_preflight", "target_encoder_artifact_score_pairs", "both_view_terminals"),
        ("one_view_terminal", "SUA_to_pMUA_cross_view_parity", "paired_pilot_completion"),
        "UNIMPLEMENTED__CURRENT_CORE_VALIDATORS_ARE_SYNTHETIC_ONLY",
        ("synthetic_validator_as_live_authority", "pMUA_before_SUA_terminal", "population_inference"),
    ),
)


def _cell_topology(cell: runtime.StagePCell) -> dict[str, Any]:
    """Literal additive topology for one view; no directory is created."""
    base_topology = runtime.stagep_output_topology(cell)
    root = _absolute(base_topology["cell_root"])
    scores = {role: _absolute(path) for role, path in base_topology["scores"].items()}
    return {
        "cell": cell.as_dict(),
        "cell_root": str(root),
        "official_preflight": str(_absolute(base_topology["official_preflight"])),
        "start": str(root / "start.json"),
        "target_materialization": str(_absolute(base_topology["target_materialization"])),
        "joint_encoder": str(_absolute(base_topology["joint_encoder"])),
        "joint_encoder_checkpoint": str(_absolute(base_topology["joint_encoder_checkpoint"])),
        "joint_embedding_bundle": str(_absolute(base_topology["joint_embedding_bundle"])),
        "scores": {role: str(path) for role, path in scores.items()},
        "terminal": str(_absolute(base_topology["terminal"])),
        "completion": str(root / "completion.json"),
        "private_snapshot": {
            "path": str(root / "private_snapshot" / "held_target_asset.snapshot"),
            "retention_policy": PRIVATE_SNAPSHOT_RETENTION,
            "may_be_published_or_used_as_score_input": False,
        },
        "aggregate": str(_absolute(base_topology["aggregate"])),
        "all_paths_are_literal_and_caller_overrides_are_forbidden": True,
        "cell_root_admission_policy": (
            "absent_before_initial_official_preflight_publication_or_afterward_a_real_non_symlink_directory_"
            "containing_only_the_verified_official_preflight_pair_before_start"
        ),
        "publication_rule": "O_EXCL_regular_0444_body_and_sha256_sidecar__raw_artifacts_same_pair_rule",
    }


def build_unadmitted_paired_execution_plan() -> dict[str, Any]:
    """Render the no-data topology without reading any admission or asset pair."""
    cells = [runtime.StagePCell.from_view(view) for view in PAIRED_VIEW_ORDER]
    return {
        "schema": LIVE_EXECUTOR_SCHEMA,
        "status": LIVE_EXECUTOR_STATUS,
        "pair_order": list(PAIRED_VIEW_ORDER),
        "predeclared_cells": [cell.as_dict() for cell in cells],
        "cell_topology_by_view": {cell.view: _cell_topology(cell) for cell in cells},
        "model_arm": runtime.PRIMARY_ARM,
        "fixed_geometry": {"output_dimension": 8, "iterations": 10000},
        "readout_routes": list(runtime.ROUTES),
        "decoders": list(runtime.DECODERS),
        "per_view_execution_contract": {
            "one_28_session_joint_fit": True,
            "source_session_count": 27,
            "held_target_support_session_count": 1,
            "same_encoder_serves_all_three_readout_routes_and_two_decoders": True,
            "cross_view_encoder_checkpoint_embedding_reuse_permitted": False,
            "target_support_neural_and_dense_auxiliary_enter_encoder_fit": True,
            "target_query_enters_encoder_or_readout_fit": False,
            "exact_offset10": {"left": 5, "right": 5, "raw_receptive_field": "[endpoint-5,endpoint+5)"},
            "future_bins_relative_to_endpoint": [1, 2, 3, 4],
            "causal_temporal_exposure_matched": False,
            "bias_direction": "favors_CEBRA_accuracy",
        },
        "cross_view_contract": {
            "same_target_behavior_order_endpoint_and_T4_target_bytes_required": True,
            "SUA_then_pMUA_terminal_order_required": True,
            "pMUA_requires_same_record_live_pooling_replay": True,
            "pMUA_fit_is_independent_from_SUA_fit": True,
        },
        "state_machine": {
            "ordered_states": list(PAIRED_STATE_SEQUENCE),
            "initial_state": PAIRED_STATE_SEQUENCE[0],
            "pMUA_producer_may_begin_only_after": "SUA_TERMINAL_PUBLISHED",
            "paired_completion_requires": ["SUA_TERMINAL_PUBLISHED", "PMUA_TERMINAL_PUBLISHED"],
            "transition_execution_enabled_in_this_scaffold": False,
        },
        "gpu_contract": {
            "CUDA_VISIBLE_DEVICES_literal_before_torch_or_cebra_import": str(PHYSICAL_GPU_INDEX),
            "expected_logical_device_after_masking": LOGICAL_CUDA_DEVICE,
            "physical_gpu_index": PHYSICAL_GPU_INDEX,
            "physical_gpu_uuid_must_equal_cost_receipt": "required_from_exact_live_admission",
            "caller_gpu_override_permitted": False,
            "one_isolated_child_per_view": True,
        },
        "producer_interfaces": [interface.as_dict() for interface in PRODUCER_INTERFACES],
        "all_real_target_gpu_cebra_score_operations_disabled": True,
        "synthetic_layout_or_validator_is_not_live_authority": True,
    }


def validate_paired_execution_plan(*, plan: Mapping[str, Any]) -> None:
    """Reject a mutated plan before it can be used for freshness or admission.

    This is deliberately structural rather than permissive: no caller may
    turn target-query fitting on, alter the SUA→pMUA order, redirect a literal
    output, change geometry, or share an encoder merely by editing a rendered
    JSON plan between preflight and a future launch.
    """
    expected = build_unadmitted_paired_execution_plan()
    require(plan.get("schema") == LIVE_EXECUTOR_SCHEMA,
            "paired plan schema drift")
    require(plan.get("pair_order") == list(PAIRED_VIEW_ORDER),
            "paired plan must retain canonical SUA then pMUA order")
    require(plan.get("predeclared_cells") == expected["predeclared_cells"],
            "paired plan cell/date/seed roster drift")
    require(plan.get("fixed_geometry") == expected["fixed_geometry"] and
            plan.get("readout_routes") == expected["readout_routes"] and
            plan.get("decoders") == expected["decoders"] and
            plan.get("model_arm") == runtime.PRIMARY_ARM,
            "paired plan fixed model/geometry/readout/decoder contract drift")
    require(plan.get("cell_topology_by_view") == expected["cell_topology_by_view"],
            "paired plan caller output alias/path drift")
    per_view = plan.get("per_view_execution_contract")
    require(isinstance(per_view, Mapping) and
            per_view.get("target_support_neural_and_dense_auxiliary_enter_encoder_fit") is True and
            per_view.get("target_query_enters_encoder_or_readout_fit") is False and
            per_view.get("cross_view_encoder_checkpoint_embedding_reuse_permitted") is False and
            per_view.get("one_28_session_joint_fit") is True and
            per_view.get("source_session_count") == 27 and
            per_view.get("held_target_support_session_count") == 1 and
            per_view.get("exact_offset10") == expected["per_view_execution_contract"]["exact_offset10"],
            "paired plan target-query fit, 28-session, or offset contract drift")
    require(plan.get("cross_view_contract") == expected["cross_view_contract"] and
            plan.get("state_machine") == expected["state_machine"] and
            plan.get("gpu_contract") == expected["gpu_contract"] and
            plan.get("producer_interfaces") == expected["producer_interfaces"],
            "paired plan cross-view/GPU/producer interface drift")


def next_paired_execution_state(*, completed_state: str) -> str:
    """Return the sole successor state; this computes no operation itself."""
    require(completed_state in PAIRED_STATE_SEQUENCE,
            "unknown paired Stage-P state")
    index = PAIRED_STATE_SEQUENCE.index(completed_state)
    require(index + 1 < len(PAIRED_STATE_SEQUENCE),
            "paired Stage-P completion has no successor")
    return PAIRED_STATE_SEQUENCE[index + 1]


def _validate_admission_mapping(*, view: str, admission: Mapping[str, Any]) -> None:
    """Check the exact, immutable predecessor result without accepting substitutes."""
    expected_cell = runtime.StagePCell.from_view(view).as_dict()
    require(admission.get("status") == LIVE_ADMISSION_STATUS,
            f"{view} capability is not the exact current Stage-P live admission")
    require(admission.get("cell") == expected_cell, f"{view} admission cell drift")
    require(admission.get("target_path_resolution_permitted") is False,
            "predecessor admission must remain a no-target authorization boundary")
    require(_valid_sha(admission.get("official_stagep_preflight_body_sha256")) and
            _valid_sha(admission.get("official_stagep_preflight_sidecar_sha256")),
            "official Stage-P preflight pair is missing from live admission")
    root_pair = admission.get("root_authorization_pair")
    control_pair = admission.get("fixed_runtime_control_pair")
    require(isinstance(root_pair, Mapping) and isinstance(control_pair, Mapping) and
            _valid_sha(root_pair.get("body_sha256")) and _valid_sha(root_pair.get("sidecar_sha256")) and
            _valid_sha(control_pair.get("body_sha256")) and _valid_sha(control_pair.get("sidecar_sha256")),
            "root authorization/runtime-control immutable pairs are missing from live admission")
    gate = admission.get("fixed_d8it250_gpu_cost_gate")
    require(isinstance(gate, Mapping) and
            gate.get("status") == "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION" and
            _valid_sha(gate.get("canonical_body_sha256")),
            "canonical d8/it250 GPU cost pair is missing from live admission")
    closure = admission.get("implementation_closure_sha256")
    require(_valid_sha(closure), "Stage-P implementation closure is missing from live admission")


def _obtain_capability(*, view: str,
                       admission_builder: Callable[..., Mapping[str, Any]]) -> StagePLiveExecutionCapability:
    """Create a capability only by calling the pinned predecessor interface."""
    # Deliberately use a keyword-only exact call: no target path, output root,
    # seed, GPU, geometry, checkpoint, or parser can enter this boundary.
    raw = admission_builder(view=view)
    require(isinstance(raw, Mapping), "Stage-P live admission builder returned no mapping")
    admission = dict(raw)
    _validate_admission_mapping(view=view, admission=admission)
    return StagePLiveExecutionCapability(
        view=view,
        cell=dict(admission["cell"]),
        admission=admission,
        admission_sha256=_sha_json(admission),
    )


def _validate_capability_pair(capabilities: Sequence[StagePLiveExecutionCapability]) -> None:
    require(tuple(capability.view for capability in capabilities) == PAIRED_VIEW_ORDER,
            "live capabilities must be built in canonical SUA then pMUA order")
    require(len({capability.admission_sha256 for capability in capabilities}) == 2,
            "per-view admissions must remain distinct view-specific immutable predecessor outputs")
    root_shas = {str(capability.admission["root_authorization_pair"]["body_sha256"]) for capability in capabilities}
    control_shas = {str(capability.admission["fixed_runtime_control_pair"]["body_sha256"]) for capability in capabilities}
    cost_shas = {str(capability.admission["fixed_d8it250_gpu_cost_gate"]["canonical_body_sha256"]) for capability in capabilities}
    closures = {str(capability.admission["implementation_closure_sha256"]) for capability in capabilities}
    require(len(root_shas) == len(control_shas) == len(cost_shas) == len(closures) == 1,
            "SUA/pMUA must bind the same root authorization, control, cost, and implementation closure")


def build_capability_bound_paired_execution_plan(
    *, admission_builder: Callable[..., Mapping[str, Any]] = runtime.build_stagep_live_admission,
) -> dict[str, Any]:
    """Build, but never execute, the planned chain after exact live admission.

    Injectable builders exist solely for no-data tests.  The public CLI always
    uses the exact `runtime.build_stagep_live_admission` default.
    """
    capabilities = tuple(_obtain_capability(view=view, admission_builder=admission_builder)
                         for view in PAIRED_VIEW_ORDER)
    _validate_capability_pair(capabilities)
    plan = build_unadmitted_paired_execution_plan()
    plan["status"] = "CAPABILITY_BOUND_PLAN_VALID__REAL_PRODUCERS_STILL_UNIMPLEMENTED"
    plan["capability_bindings"] = {
        capability.view: {
            "cell": dict(capability.cell),
            "admission_sha256": capability.admission_sha256,
            "official_stagep_preflight_body_sha256": capability.admission["official_stagep_preflight_body_sha256"],
            "official_stagep_preflight_sidecar_sha256": capability.admission["official_stagep_preflight_sidecar_sha256"],
            "root_authorization_body_sha256": capability.admission["root_authorization_pair"]["body_sha256"],
            "fixed_runtime_control_body_sha256": capability.admission["fixed_runtime_control_pair"]["body_sha256"],
            "fixed_d8it250_gpu_cost_body_sha256": capability.admission["fixed_d8it250_gpu_cost_gate"]["canonical_body_sha256"],
            "implementation_closure_sha256": capability.admission["implementation_closure_sha256"],
        }
        for capability in capabilities
    }
    plan["capability_token_origin"] = "exact_current_runtime.build_stagep_live_admission__not_caller_supplied"
    plan["capability_pair_sha256"] = _sha_json(plan["capability_bindings"])
    return plan


def _fresh_output_paths(topology: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return every prospective result pair, excluding required pre-existing preflight."""
    score_paths = topology.get("scores")
    require(isinstance(score_paths, Mapping) and set(score_paths) ==
            {f"{route}__{decoder}" for route in runtime.ROUTES for decoder in runtime.DECODERS},
            "cell topology lacks the exact six score paths")
    ordered = [
        topology["start"], topology["target_materialization"], topology["joint_encoder"],
        topology["joint_encoder_checkpoint"], topology["joint_embedding_bundle"],
        *(score_paths[f"{route}__{decoder}"] for route in runtime.ROUTES for decoder in runtime.DECODERS),
        topology["terminal"], topology["completion"],
    ]
    paths = tuple(_absolute(path) for path in ordered)
    # 1 start + target + encoder receipt + checkpoint + embedding + 6 scores
    # + terminal + completion.  The pre-existing official preflight is an
    # admission input and intentionally does not appear here.
    require(len(paths) == 13 and len(set(paths)) == 13,
            "Stage-P prospective output topology must contain exactly 13 distinct non-preflight paths per view")
    return paths


def require_fresh_paired_execution_outputs(*, plan: Mapping[str, Any]) -> None:
    """Reject a collision before any possible target path or ML import.

    This performs only lexical filesystem checks.  It does not create a root,
    follow a path, open an asset, import a numerical library, or load a model.
    A later producer must additionally establish safe real parent directories
    before O_EXCL publication.
    """
    validate_paired_execution_plan(plan=plan)
    topologies = plan.get("cell_topology_by_view")
    require(isinstance(topologies, Mapping) and tuple(topologies) == PAIRED_VIEW_ORDER,
            "paired plan lacks canonical ordered cell topology")
    seen: set[Path] = set()
    for view in PAIRED_VIEW_ORDER:
        topology = topologies[view]
        require(isinstance(topology, Mapping), "cell topology must be a mapping")
        _require_clean_cell_root_before_start(topology=topology)
        for path in _fresh_output_paths(topology):
            require(path not in seen, "two planned Stage-P output roles collide")
            seen.add(path)
            require(not os.path.lexists(path) and not os.path.lexists(f"{path}.sha256"),
                    f"Stage-P planned output is not fresh before target admission: {path}")


def _require_clean_cell_root_before_start(*, topology: Mapping[str, Any]) -> None:
    """Allow only the immutable official preflight input in a reserved root.

    A root can legitimately exist because the official preflight is an input to
    the later live admission.  It cannot contain a prior start, target,
    checkpoint, embedding, score, terminal, or stray alias before this planned
    execution.  The subsequent real producer must validate that preflight pair
    with the exact runtime admission; this early check merely prevents a stale
    root from being silently reused.
    """
    root = _absolute(str(topology["cell_root"]))
    if not os.path.lexists(root):
        return
    info = root.lstat()
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            "Stage-P cell root must be an absent or real non-symlink directory")
    expected = {Path(str(topology["official_preflight"])).name,
                f"{Path(str(topology['official_preflight'])).name}.sha256"}
    observed = {entry.name for entry in os.scandir(root)}
    require(observed <= expected,
            "Stage-P cell root is not fresh/reserved: it contains non-preflight output or alias")


def refuse_paired_stagep_live_execution(
    *, admission_builder: Callable[..., Mapping[str, Any]] = runtime.build_stagep_live_admission,
) -> None:
    """Public final tripwire: validate only non-target prerequisites then stop.

    Freshness is deliberately checked first.  If required immutable authority
    pairs are absent, `_obtain_capability` fails before any target path,
    private snapshot, Torch/CEBRA import, GPU device query, parser, fit,
    scorer, artifact publication, or target readout is reachable.
    """
    static_plan = build_unadmitted_paired_execution_plan()
    require_fresh_paired_execution_outputs(plan=static_plan)
    build_capability_bound_paired_execution_plan(admission_builder=admission_builder)
    raise TrackBV2SubjectMStagePLiveExecutorError(
        "paired Stage-P execution deliberately remains disabled: eight real producer interfaces and independent launch review are required before target access"
    )


def executable_producer_interface_gaps() -> list[dict[str, Any]]:
    """Return the exact downstream implementation obligations in canonical order."""
    return [interface.as_dict() for interface in PRODUCER_INTERFACES]
