"""Fail-closed Track-B v2 RT one-cell live-execution successor.

This additive module defines the canonical one-cell configuration, no-target
preflight, future execution ordering, immutable receipt schemas, scorer
validation, and 45-cell aggregate validation for the RT development route.
It deliberately contains no NWB/NPZ parser, CEBRA/Torch/sklearn import, model
fit, GPU operation, or scientific scorer.  The only future target opener it
authorizes is the canonical authority-owned context manager
``track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority``;
callers cannot provide a target session, path, ledger, byte count, or SHA.

No function in this module calls that opener.  A separately root-reviewed
runtime successor must preserve the exact state machine declared here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_development_target_authority as development_authority
import track_b_v2_development_target_materializer as materializer
import track_b_v2_fixed_gpu_engineering as fixed_gpu
import track_b_v2_live_contract as sealed
import track_b_v2_rt_development_executor as rt_scaffold
import track_b_v2_rt_local_asset_authority as rt_asset_authority
import track_b_v2_source_adapter as source
import track_b_v2_subject_m_development_executor as subject_executor


CONFIG_SCHEMA = "track_b_v2_rt_one_cell_canonical_config_v1"
PREFLIGHT_SCHEMA = "track_b_v2_rt_one_cell_no_target_preflight_v1"
TARGET_RECEIPT_SCHEMA = "track_b_v2_rt_one_cell_target_materialization_receipt_v1"
ENCODER_RECEIPT_SCHEMA = "track_b_v2_rt_one_cell_joint_encoder_receipt_v1"
SCORE_RECEIPT_SCHEMA = "track_b_v2_rt_one_cell_score_receipt_v1"
TERMINAL_RECEIPT_SCHEMA = "track_b_v2_rt_one_cell_terminal_receipt_v1"
AGGREGATE_SCHEMA = "track_b_v2_rt_45cell_terminal_aggregate_v1"
STATUS_COST_NO_GO = "NO_GO__CANONICAL_FIXED_GPU_COST_RECEIPT_REQUIRED__NO_TARGET"
STATUS_CONTROL_NO_GO = "NO_GO__ROOT_FROZEN_FIXED_GEOMETRY_RUNTIME_CONTROL_REQUIRED__NO_TARGET"
STATUS_REVIEW_ONLY = "NO_TARGET_PREFLIGHT_PASS__ROOT_REVIEW_REQUIRED_BEFORE_ANY_EXECUTION"

SEEDS = (42, 43, 44)
PILOT_OUTER_FOLD_INDEX = 0
PILOT_SEED = 42
ROUTES = rt_scaffold.READOUT_ROUTES
DECODERS = rt_scaffold.DECODERS
LOCAL_ASSET_AUTHORITY_SHA256 = rt_scaffold.LOCAL_ASSET_AUTHORITY_SHA256

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_rt_one_cell_live_executor.py"
RESULT_ROOT = REPO_ROOT / "cebra_exploration/results/track_b_v2_rt_one_cell_live_v1"
ROOT_FROZEN_PROTOCOL = REPO_ROOT / "cebra_exploration/docs/TRACK_B_V2_H1_EXCLUDED_PROTOCOL.md"
ROOT_FROZEN_PROTOCOL_SHA256 = "14e39b7bff31fc31fe257e5a4998758b947dcac1d68af0cec26a1192585eb51d"
RUNTIME_CONTROL_BODY = (
    REPO_ROOT / "cebra_exploration/results"
    / "track_b_v2_rt_fixed_geometry_synthetic_runtime_control_v1/receipt.json"
)
RUNTIME_CONTROL_SCHEMA = "track_b_v2_rt_fixed_geometry_synthetic_runtime_control_v1"
RUNTIME_CONTROL_STATUS = "ROOT_FROZEN_FIXED_GEOMETRY_SYNTHETIC_RUNTIME_CONTROL_PASS__NO_TARGET"
SYNTHETIC_V2_TERMINAL_SHA256 = "99269afd2770ac333b5a529768920a1adbd4be560dcd942839697edab2c14de8"
SYNTHETIC_V2_CLOSURE_SHA256 = "025a34913925973cab6faf255ae44c5f008bb4646da5237f4bb4a0571316affe"
SYNTHETIC_V2_PERMUTATION_SHA256 = "b101d5fb8d0d7b4703a0df87377253c055f653e970e799de52b733f7250a9444"
SYNTHETIC_V2_RAW_BOUND_EVIDENCE_SHA256 = "62c87b61a1c24619e7fa4a0da1bb801378e1a33b7c404ad16af43ae229e36c2d"
SYNTHETIC_V2_SETTLED_FILES = {
    "v2_core": "56226348b110988f227311703cbcba8495a2e703da1058ad77d98fa009997fd7",
    "v2_cli": "4efc13b8836138e8599ac0958e27f098183e03f90e7ec93c3cd1c984ec038d88",
    "v2_focused_tests": "2b5800b9eb82e6060776b81ae3277f974961a98454a4c15d68cdccae39c9f0fe",
}
POSITIVE_CONTROL_THRESHOLD_R2 = 0.70
DERANGED_HARD_NULL_THRESHOLD_R2 = 0.60
POSITIVE_THRESHOLD_ORIGIN = "PREDECLARED_BEFORE_SYNTHETIC_V2_TERMINAL"
HARD_NULL_THRESHOLD_ORIGIN = (
    "ROOT_FROZEN_AFTER_IMMUTABLE_SYNTHETIC_V2_TERMINAL_BEFORE_REAL_TARGET"
)

RT_T4_QUERY_BODY = (
    REPO_ROOT / "sua_exploration/comparators/receipts/rt_classical_comparators"
    / "rt_classical_comparators_receipt.json"
)
RT_T4_QUERY_BODY_SHA256 = "c51cb0ff7dadd3c40ca7861dea92d80f3457c709801ea8ec7ba3e91b9e52b042"
RT_T4_PER_FOLD_BODY = (
    REPO_ROOT / "sua_exploration/results/rt_terminal_stage2_20260811_canonical"
    / "rt_sparse_t4d_b2_forward_reeval_v2_20260811"
    / "RT_T4D_VS_B2_D1024_FORWARD_ONLY_15FOLD_FINAL_v1.json"
)
RT_T4_PER_FOLD_BODY_SHA256 = "c36ec0e31ed913ed4e8077f9a4d9d634d53529ce037ad06af1f48d279b16820e"
RT_T4_STAGE2_BODY = (
    REPO_ROOT / "sua_exploration/results/rt_terminal_stage2_20260811_canonical"
    / "matrix_v1/STAGE2_MATRIX_AGGREGATE_v1.json"
)
RT_T4_STAGE2_BODY_SHA256 = "bb2806953e979180c408fb55744534be6fa470d4144f210cc50917a9b1006b7d"


class TrackBV2RTOneCellError(rt_scaffold.TrackBV2RTDevelopmentExecutorError):
    """Raised before target access or when a future receipt drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2RTOneCellError(message)


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _body_sha(payload: Mapping[str, Any], *, declared_key: str) -> str:
    declared = payload.get(declared_key)
    require(_valid_sha(declared), f"{declared_key} missing or malformed")
    bare = dict(payload)
    bare.pop(declared_key, None)
    require(declared == _sha_json(bare), f"{declared_key} does not bind payload")
    return str(declared)


def _read_named_json(path: Path, expected_sha: str, *, label: str) -> dict[str, Any]:
    """Read one named non-target lineage body through the existing one-FD reader."""
    raw = source._read_regular_file(path, label=label)
    require(hashlib.sha256(raw).hexdigest() == expected_sha, f"{label} SHA drift")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2RTOneCellError(f"{label} is not JSON") from exc
    require(isinstance(payload, dict), f"{label} root is not an object")
    return payload


def _canonical_fold_rows() -> tuple[dict[str, Any], ...]:
    plan = source.build_rt_15fold_source_authority_plan()
    source.validate_rt_15fold_source_authority_plan(plan)
    rows = plan.get("outer_folds")
    require(isinstance(rows, list) and len(rows) == 15, "canonical RT fold topology is not 15 rows")
    ordered = tuple(dict(row) for row in rows)
    require(tuple(row.get("outer_fold_index") for row in ordered) == tuple(range(15)),
            "canonical RT fold indices are not ordered 0..14")
    return ordered


@dataclass(frozen=True)
class CellKey:
    outer_fold_index: int
    outer_fold_id: str
    target_session_id: str
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_fold_index": self.outer_fold_index,
            "outer_fold_id": self.outer_fold_id,
            "target_session_id": self.target_session_id,
            "seed": self.seed,
        }


def canonical_cell_key(*, outer_fold_index: int, seed: int) -> CellKey:
    require(type(outer_fold_index) is int and 0 <= outer_fold_index < 15,
            "outer fold index must be one canonical integer in 0..14")
    require(type(seed) is int and seed in SEEDS, "seed must be one of 42,43,44")
    row = _canonical_fold_rows()[outer_fold_index]
    fold_id = row.get("outer_fold_id")
    target = row.get("opaque_held_out_target_session_id")
    require(fold_id == f"rt_outer_fold_{outer_fold_index:02d}" and isinstance(target, str),
            "canonical RT fold identity drift")
    return CellKey(outer_fold_index, str(fold_id), target, seed)


def require_exact_development_pilot(*, outer_fold_index: int, seed: int) -> CellKey:
    """Freeze the sole executable Stage-P cell before any authority/target work."""
    require((outer_fold_index, seed) == (PILOT_OUTER_FOLD_INDEX, PILOT_SEED),
            "actual RT development pilot is frozen to rt_outer_fold_00 seed42; substitution forbidden")
    return canonical_cell_key(outer_fold_index=outer_fold_index, seed=seed)


def canonical_cell_dir(key: CellKey) -> Path:
    return RESULT_ROOT / "cells" / key.outer_fold_id / f"seed_{key.seed}"


def canonical_output_topology(key: CellKey) -> dict[str, Any]:
    root = canonical_cell_dir(key)
    score_paths = {
        f"{route}__{decoder}": str(root / "scores" / f"{route}__{decoder}.json")
        for route in ROUTES for decoder in DECODERS
    }
    return {
        "cell_root": str(root),
        "canonical_config": str(root / "canonical_config.json"),
        "official_preflight": str(root / "official_preflight.json"),
        "target_materialization_receipt": str(root / "target_materialization_receipt.json"),
        "joint_encoder_receipt": str(root / "joint_encoder_receipt.json"),
        "score_receipts": score_paths,
        "terminal_receipt": str(root / "terminal_receipt.json"),
        "aggregate_receipt": str(RESULT_ROOT / "aggregate" / "terminal_45cell_aggregate.json"),
        "caller_output_override_permitted": False,
        "every_body_sidecar_suffix": ".sha256",
        "publication": "O_EXCL_regular_0444_body_and_sidecar",
    }


def _implementation_closure() -> dict[str, dict[str, Any]]:
    paths = {
        "rt_one_cell_core": Path(__file__).absolute(),
        "rt_one_cell_cli": CLI,
        "rt_development_scaffold": Path(rt_scaffold.__file__).absolute(),
        "canonical_materializer": Path(materializer.__file__).absolute(),
        "development_target_authority": Path(development_authority.__file__).absolute(),
        "rt_local_asset_authority_and_same_fd_opener": Path(rt_asset_authority.__file__).absolute(),
        "source_adapter": Path(source.__file__).absolute(),
        "fixed_gpu_engineering": Path(fixed_gpu.__file__).absolute(),
        "fixed_gpu_receipt_validator": Path(subject_executor.__file__).absolute(),
        "torchmetrics151_reference_scorer": (
            REPO_ROOT / "sua_exploration/mc_maze/native_m2_m24_ridge_w50.py"
        ),
        "root_frozen_H1_excluded_protocol": ROOT_FROZEN_PROTOCOL,
    }
    closure: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        raw = source._read_regular_file(path, label=f"RT one-cell implementation {label}")
        digest = hashlib.sha256(raw).hexdigest()
        if label == "root_frozen_H1_excluded_protocol":
            require(digest == ROOT_FROZEN_PROTOCOL_SHA256, "root-frozen RT execution protocol SHA drift")
        closure[label] = {"path": str(path), "sha256": digest, "bytes": len(raw)}
    return closure


def _fixed_contract() -> dict[str, Any]:
    return {
        "model_arm": "cebra_joint_behavior",
        "output_dimension": 8,
        "iterations": 10_000,
        "seed_set": list(SEEDS),
        "selected_cell_seed_is_stochastic_sensitivity_not_bitwise_determinism": True,
        "source_stream_count": 14,
        "held_support_trials": 24,
        "encoder_fit_streams": "exact_14_full_source_streams_plus_one_continuous_held_M24_support_stream",
        "encoder_fit_count_per_cell": 1,
        "linear_ridge_normalized_lambda": 0.01,
        "cosine_knn_k": 3,
        "readout_routes": list(ROUTES),
        "decoders": list(DECODERS),
        "same_encoder_state_for_all_six_scores": True,
        "query_enters_encoder_or_readout_fit": False,
        "posthoc_geometry_seed_route_or_decoder_selection_permitted": False,
        "offset10": {
            "half_open_offsets_relative_to_prediction_target": [-5, 5],
            "exact_raw_indices": "range(endpoint-5, endpoint+5)",
            "previous_raw_bins": 5,
            "prediction_target_bin_included": True,
            "strictly_future_raw_bins": 4,
            "receptive_field_must_be_wholly_in_query": True,
            "support_query_boundary_crossing_permitted": False,
            "causal": False,
            "bias_direction": "favors_CEBRA_accuracy",
            "online_or_latency_equivalent_claim_permitted": False,
        },
        "valid_transform_alignment": {
            "vendored_CEBRA_transform_returns_padded_full_length": True,
            "crop_half_open_offsets": [5, 5],
            "source_blocks_cropped_separately_before_any_concatenation": True,
            "source_block_count": 14,
            "held_M24_support_cropped_as_its_own_block": True,
            "query_cropped_as_its_own_wholly_query_block": True,
            "concatenate_then_crop_permitted": False,
            "padded_edge_rows_in_any_readout_fit_or_score": 0,
            "ordered_endpoint_and_full_RF_hash_required_per_block": True,
            "embedding_and_auxiliary_label_row_parity_required_per_block": True,
        },
        "metric": {
            "implementation": "torchmetrics.regression.R2Score",
            "torchmetrics_version": "1.5.1",
            "multioutput": "variance_weighted",
            "dtype": "float32",
            "device": "cpu",
            "update_scope": "one_complete_ordered_fold_session_query_then_compute_once",
            "one_R2_per_fold_session_seed_route_decoder": True,
            "pooled_query_rows_across_folds_permitted": False,
            "custom_numpy_float64_metric_may_claim_exact_parity": False,
            "prediction_and_target_float32_byte_SHA_required": True,
        },
    }


def inspect_root_frozen_runtime_control_pair(*, cost_gate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the future root-frozen synthetic control/hard-null admission pair."""
    body = RUNTIME_CONTROL_BODY
    sidecar = body.with_name(f"{body.name}.sha256")
    if not os.path.lexists(body) or not os.path.lexists(sidecar):
        return {
            "schema": RUNTIME_CONTROL_SCHEMA,
            "status": STATUS_CONTROL_NO_GO,
            "reason": "canonical runtime-control body+sidecar pair is absent or partial",
            "canonical_body_path": str(body),
            "canonical_sidecar_path": str(sidecar),
            "body_lexists": os.path.lexists(body),
            "sidecar_lexists": os.path.lexists(sidecar),
            "control_scale_and_threshold": "ROOT_PENDING_UNTIL_FIXED_GPU_COST_REVIEW",
            "target_access_permitted": False,
        }
    pair = sealed.ExplicitSealedReceiptPair(
        role="rt_fixed_geometry_synthetic_runtime_control", body_path=body, sidecar_path=sidecar,
    )
    try:
        payload, body_sha = sealed._strict_readonly_pair(pair)
    except sealed.TrackBV2LiveContractError as exc:
        raise TrackBV2RTOneCellError(str(exc)) from exc
    require(payload.get("schema") == RUNTIME_CONTROL_SCHEMA and
            payload.get("status") == RUNTIME_CONTROL_STATUS,
            "runtime-control schema/status drift")
    require(payload.get("fixed_geometry") == {"output_dimension": 8, "iterations": 10_000} and
            payload.get("vendored_cebra_version") == "0.6.1",
            "runtime-control geometry/CEBRA drift")
    require(payload.get("minted_after_fixed_gpu_cost_review") is True and
            payload.get("fixed_gpu_cost_receipt_body_sha256") == cost_gate.get("canonical_body_sha256"),
            "runtime-control/cost review lineage drift")
    controls = payload.get("controls")
    require(isinstance(controls, Mapping) and
            controls.get("positive_control_pass") is True and
            controls.get("ordinary_unaligned_distribution_reported") is True and
            controls.get("ordinary_unaligned_role") ==
            "diagnostic_distribution_only__not_a_hard_null_gate" and
            controls.get("deranged_support_hard_null_pass") is True and
            controls.get("deranged_permutation_fixed_seed_independent_nonidentity") is True and
            controls.get("thresholds_frozen_before_target") is True,
            "runtime-control positive/negative/hard-null contract drift")
    evidence = payload.get("raw_bound_control_evidence")
    require(isinstance(evidence, Mapping) and
            evidence.get("schema") == "track_b_v2_post_synthetic_raw_bound_control_evidence_v1" and
            evidence.get("synthetic_v2_terminal_body_sha256") == SYNTHETIC_V2_TERMINAL_SHA256 and
            evidence.get("synthetic_v2_terminal_closure_sha256") == SYNTHETIC_V2_CLOSURE_SHA256 and
            evidence.get("settled_synthetic_v2_file_sha256") == SYNTHETIC_V2_SETTLED_FILES and
            evidence.get("positive_control_threshold_r2") == POSITIVE_CONTROL_THRESHOLD_R2 and
            evidence.get("positive_threshold_origin") == POSITIVE_THRESHOLD_ORIGIN and
            evidence.get("deranged_hard_null_threshold_r2") == DERANGED_HARD_NULL_THRESHOLD_R2 and
            evidence.get("hard_null_threshold_origin") == HARD_NULL_THRESHOLD_ORIGIN and
            evidence.get("hard_null_threshold_may_be_relaxed_or_backfilled") is False and
            evidence.get("comparison") ==
            "strict_greater_positive_threshold__strict_less_hard_null_threshold" and
            evidence.get("permutation_authority_sha256") == SYNTHETIC_V2_PERMUTATION_SHA256 and
            evidence.get("arm_run_count") == 32 and evidence.get("cebra_fit_call_count") == 56 and
            evidence.get("decoder_measurement_count") == 192,
            "RT runtime-control terminal evidence lineage/count/threshold drift")
    evidence_without_sha = dict(evidence)
    declared_evidence_sha = evidence_without_sha.pop("raw_bound_evidence_sha256", None)
    require(declared_evidence_sha == SYNTHETIC_V2_RAW_BOUND_EVIDENCE_SHA256 and
            _sha_json(evidence_without_sha) == SYNTHETIC_V2_RAW_BOUND_EVIDENCE_SHA256,
            "RT runtime-control raw-bound evidence exact SHA drift")
    rows = evidence.get("target_support_only_raw_measurements")
    require(isinstance(rows, list) and len(rows) == 64,
            "RT runtime-control target-support raw evidence coverage drift")
    expected = {(seed, arm, decoder) for seed in range(8) for arm in (
        "cebra_joint_behavior", "cebra_frozen_source_adapt", "cebra_adapt_unaligned",
        "cebra_joint_behavior__target_support_auxiliary_rows_deranged")
                for decoder in ("linear_ridge", "knn_cosine_k3")}
    require({(row.get("seed"), row.get("arm"), row.get("decoder")) for row in rows} == expected and
            all(row.get("readout_route") == "target_support_only_standard_cebra_accuracy" and
                row.get("query_neural_or_auxiliary_in_fit") is False and
                isinstance(row.get("target_query_r2"), float) and math.isfinite(row["target_query_r2"])
                for row in rows),
            "RT runtime-control raw evidence grid/query-fit drift")
    for arm in ("cebra_joint_behavior", "cebra_frozen_source_adapt"):
        for decoder in ("linear_ridge", "knn_cosine_k3"):
            values = [row["target_query_r2"] for row in rows
                      if row["arm"] == arm and row["decoder"] == decoder]
            require(len(values) == 8 and all(value > POSITIVE_CONTROL_THRESHOLD_R2 for value in values),
                    f"RT positive control failed strict predeclared threshold: {arm}/{decoder}")
    deranged = "cebra_joint_behavior__target_support_auxiliary_rows_deranged"
    for decoder in ("linear_ridge", "knn_cosine_k3"):
        values = [row["target_query_r2"] for row in rows
                  if row["arm"] == deranged and row["decoder"] == decoder]
        require(len(values) == 8 and all(value < DERANGED_HARD_NULL_THRESHOLD_R2 for value in values),
                f"RT deranged hard null failed root-frozen post-smoke threshold: {decoder}")
    computed_positive_min = min(row["target_query_r2"] for row in rows
                                if row["arm"] in ("cebra_joint_behavior", "cebra_frozen_source_adapt"))
    computed_unaligned_max = max(row["target_query_r2"] for row in rows
                                 if row["arm"] == "cebra_adapt_unaligned")
    computed_deranged_max = max(row["target_query_r2"] for row in rows if row["arm"] == deranged)
    require(evidence.get("ordinary_unaligned_distribution_reported") is True and
            evidence.get("ordinary_unaligned_role") ==
            "diagnostic_distribution_only__not_a_hard_null_gate" and
            evidence.get("no_query_neural_or_auxiliary_in_any_fit") is True and
            evidence.get("target_data_discovered") is False and
            evidence.get("target_data_opened") is False and evidence.get("formal_data_opened") is False and
            evidence.get("NWB_or_NPZ_opened") is False,
            "RT unaligned role or no-query/no-target evidence drift")
    scale_threshold = payload.get("control_scale_and_threshold")
    require(isinstance(scale_threshold, Mapping) and
            scale_threshold.get("frozen_after_cost_review_before_target") is True and
            set(scale_threshold) == {
                "frozen_after_cost_review_before_target", "synthetic_signal_scale",
                "positive_control_threshold_r2", "deranged_hard_null_threshold_r2",
                "positive_threshold_origin", "hard_null_threshold_origin",
                "positive_min", "ordinary_negative_max", "deranged_hard_null_max",
            } and
            all(type(scale_threshold.get(key)) is float and math.isfinite(scale_threshold[key])
                for key in ("synthetic_signal_scale", "positive_min", "ordinary_negative_max",
                            "deranged_hard_null_max")) and
            scale_threshold.get("synthetic_signal_scale", 0.0) > 0.0,
            "runtime-control scale/threshold is not an exact post-cost root freeze")
    require(scale_threshold.get("positive_control_threshold_r2") == POSITIVE_CONTROL_THRESHOLD_R2 and
            scale_threshold.get("deranged_hard_null_threshold_r2") == DERANGED_HARD_NULL_THRESHOLD_R2 and
            scale_threshold.get("positive_threshold_origin") == POSITIVE_THRESHOLD_ORIGIN and
            scale_threshold.get("hard_null_threshold_origin") == HARD_NULL_THRESHOLD_ORIGIN and
            scale_threshold.get("positive_min") == computed_positive_min and
            scale_threshold.get("ordinary_negative_max") == computed_unaligned_max and
            scale_threshold.get("deranged_hard_null_max") == computed_deranged_max,
            "runtime-control scale/threshold summary does not equal raw-bound evidence")
    require(payload.get("target_data_opened") is False and payload.get("formal_data_opened") is False and
            payload.get("score_emitted") is False,
            "runtime-control receipt claims target/formal/score access")
    require(payload.get("decision_threshold_r2") == POSITIVE_CONTROL_THRESHOLD_R2 and
            payload.get("positive_threshold_origin") == POSITIVE_THRESHOLD_ORIGIN and
            payload.get("deranged_hard_null_threshold_r2") == DERANGED_HARD_NULL_THRESHOLD_R2 and
            payload.get("hard_null_threshold_origin") == HARD_NULL_THRESHOLD_ORIGIN and
            payload.get("ordinary_unaligned_is_threshold_or_pass_gate") is False,
            "RT predeclared threshold or unaligned non-gating role drift")
    import track_b_v2_post_synthetic_runtime_control_authority as control_authority
    live_publisher_closure = control_authority.implementation_closure()
    require(payload.get("publisher_implementation_closure_at_launch") ==
            payload.get("publisher_implementation_closure_at_final") == live_publisher_closure and
            payload.get("publisher_launch_final_live_closure_equal") is True,
            "RT runtime-control publisher launch/final/live closure drift")
    return {
        "schema": RUNTIME_CONTROL_SCHEMA,
        "status": RUNTIME_CONTROL_STATUS,
        "canonical_body_path": str(body),
        "canonical_body_sha256": body_sha,
        "canonical_sidecar_path": str(sidecar),
        "control_scale_and_threshold": payload.get("control_scale_and_threshold"),
        "target_access_permitted": False,
    }


def _sealed_t4_lineage(*, key: CellKey, expected_asset_sha256: str) -> dict[str, Any]:
    """Bind exact sealed T4/query JSON bodies without opening an RT asset."""
    require(_valid_sha(expected_asset_sha256), "internal asset SHA binding malformed")
    proposal = source.canonical_metric_pointer_mint_proposal("rt", None)
    bodies = proposal.get("bodies")
    require(isinstance(bodies, Mapping), "RT metric proposal bodies missing")
    exact = {
        "aggregate_metric_query_identity_body": (RT_T4_QUERY_BODY, RT_T4_QUERY_BODY_SHA256),
        "per_fold_t4d_body": (RT_T4_PER_FOLD_BODY, RT_T4_PER_FOLD_BODY_SHA256),
        "stage2_delta_companion": (RT_T4_STAGE2_BODY, RT_T4_STAGE2_BODY_SHA256),
    }
    for role, (path, sha) in exact.items():
        item = bodies.get(role)
        require(isinstance(item, Mapping) and item.get("path") == str(path) and item.get("sha256") == sha,
                f"sealed RT lineage proposal drift: {role}")

    aggregate = _read_named_json(RT_T4_QUERY_BODY, RT_T4_QUERY_BODY_SHA256,
                                 label="sealed RT T4 query identity body")
    require(aggregate.get("schema") == "rt_classical_comparators_v1" and
            aggregate.get("query_identity_binding", {}).get("all_folds_bound_to_sealed_stage2") is True,
            "sealed RT T4 query body schema/status drift")
    per_fold = aggregate.get("per_fold")
    require(isinstance(per_fold, list) and len(per_fold) == 15, "sealed RT T4 per-fold query rows drift")
    selected = [row for row in per_fold if isinstance(row, Mapping) and
                row.get("fold") == key.outer_fold_index and row.get("session_name") == key.target_session_id]
    require(len(selected) == 1 and selected[0].get("query_identity_bound") is True,
            "sealed RT T4 fold/session query lineage missing or ambiguous")
    row = dict(selected[0])
    query = row.get("sealed_query_identity")
    require(isinstance(query, Mapping) and set(query) == {
        "ordered_window_start_sha256", "ordered_target_covariate_evalmask_sha256",
        "ordered_query_identity_sha256",
    } and all(_valid_sha(value) for value in query.values()), "sealed RT query digest set drift")
    query_windows = row.get("endpoint_pv_diagnostic", {}).get("query_windows")
    require(type(query_windows) is int and query_windows > 0, "sealed RT query window count drift")
    bound_query = aggregate.get("query_identity_binding", {}).get("per_fold_hashes", {}).get(
        key.target_session_id
    )
    require(bound_query == query, "sealed RT per-fold row/query-identity binding mismatch")
    input_nwb_sha = aggregate.get("input_bindings", {}).get("nwb_sha256", {}).get(key.target_session_id)
    require(input_nwb_sha == expected_asset_sha256, "sealed T4 target NWB SHA/local asset authority mismatch")
    implementation = aggregate.get("input_bindings", {}).get("implementation_sha256")
    require(isinstance(implementation, Mapping) and all(_valid_sha(v) for v in implementation.values()),
            "sealed RT metric implementation lineage malformed")

    t4_rows_body = _read_named_json(RT_T4_PER_FOLD_BODY, RT_T4_PER_FOLD_BODY_SHA256,
                                    label="sealed RT T4 per-fold body")
    t4_rows = t4_rows_body.get("rows")
    matches = [item for item in t4_rows if isinstance(item, Mapping) and
               item.get("fold") == key.outer_fold_index and item.get("session") == key.target_session_id]
    require(len(matches) == 1 and math.isfinite(float(matches[0].get("t4d_r2"))),
            "sealed RT per-fold T4 lineage missing or malformed")
    stage2 = _read_named_json(RT_T4_STAGE2_BODY, RT_T4_STAGE2_BODY_SHA256,
                             label="sealed RT Stage2 companion")
    require(stage2.get("schema") == "rt_sparse_endpoint_stage2_matrix_aggregate_v1" and
            stage2.get("cells") == 45, "sealed RT Stage2 diagnostic companion drift")
    return {
        "metric_pointer_body_sha256": "d2e53165cba76aea5f28b893592b7bcf2951b8a4f13bafe8d5a14061b30072b6",
        "aggregate_query_identity_body": {"path": str(RT_T4_QUERY_BODY), "sha256": RT_T4_QUERY_BODY_SHA256},
        "per_fold_t4_body": {"path": str(RT_T4_PER_FOLD_BODY), "sha256": RT_T4_PER_FOLD_BODY_SHA256},
        "stage2_diagnostic_body": {"path": str(RT_T4_STAGE2_BODY), "sha256": RT_T4_STAGE2_BODY_SHA256,
                                     "metric_authority": False},
        "outer_fold_index": key.outer_fold_index,
        "outer_fold_id": key.outer_fold_id,
        "target_session_id": key.target_session_id,
        "target_nwb_sha256": input_nwb_sha,
        "query_window_count": query_windows,
        "sealed_query_identity": dict(query),
        "historical_metric_implementation_sha256": dict(implementation),
        "t4d_reference_r2_lineage_only_not_a_CEBRA_score": float(matches[0]["t4d_r2"]),
        "target_data_opened": False,
    }


def build_canonical_config(*, outer_fold_index: int, seed: int) -> dict[str, Any]:
    """Build one canonical config; fold/session/asset identity is internally derived."""
    key = canonical_cell_key(outer_fold_index=outer_fold_index, seed=seed)
    authority = development_authority.build_development_target_query_authority(
        dataset="rt", view=None, outer_fold_id=key.outer_fold_id,
        target_session_id=key.target_session_id,
    )
    require(authority.get("development_target_query_authority_sha256") ==
            _sha_json({k: v for k, v in authority.items()
                       if k != "development_target_query_authority_sha256"}),
            "development target authority self SHA drift")
    source_binding = authority.get("source_authority_binding")
    require(isinstance(source_binding, Mapping) and source_binding.get("source_session_count") == 14,
            "one-cell source binding is not exact 14")
    source_ids = source_binding.get("source_session_ids")
    require(isinstance(source_ids, list) and len(source_ids) == len(set(source_ids)) == 14 and
            key.target_session_id not in source_ids, "one-cell source roster/held exclusion drift")

    materialized_plan = materializer.build_development_target_materializer_dry_plan(
        dataset="rt", view=None, outer_fold_id=key.outer_fold_id,
        target_session_id=key.target_session_id,
    )
    require(materialized_plan.get("canonical_development_authority", {}).get(
        "development_target_query_authority_sha256") ==
        authority["development_target_query_authority_sha256"],
        "canonical materializer/development authority SHA mismatch")
    ledger = materialized_plan.get("target_asset_ledger_gate")
    require(isinstance(ledger, Mapping) and ledger.get("authority", {}).get("body_sha256") ==
            LOCAL_ASSET_AUTHORITY_SHA256, "sealed RT local asset authority drift")
    asset = ledger.get("target_asset")
    require(isinstance(asset, Mapping) and asset.get("session_id") == key.target_session_id and
            asset.get("held_target_outer_fold_id") == key.outer_fold_id,
            "internally selected RT asset/fold drift")
    asset_sha = asset.get("expected_sha256")
    require(_valid_sha(asset_sha), "internally selected RT asset SHA malformed")
    t4 = _sealed_t4_lineage(key=key, expected_asset_sha256=str(asset_sha))
    closure = _implementation_closure()
    payload = {
        "schema": CONFIG_SCHEMA,
        "status": "CANONICAL_ONE_CELL_CONFIG__NOT_EXECUTED__NOT_AUTHORITY_UNTIL_IMMUTABLY_PUBLISHED",
        "cell_key": key.as_dict(),
        "dataset": "rt",
        "view": None,
        "formal": False,
        "root_frozen_protocol": {
            "path": str(ROOT_FROZEN_PROTOCOL),
            "sha256": ROOT_FROZEN_PROTOCOL_SHA256,
            "section": "Root-frozen first live cells and score semantics (2026-08-15)",
        },
        "development_pilot": key.outer_fold_index == PILOT_OUTER_FOLD_INDEX and key.seed == PILOT_SEED,
        "execution_scope": (
            "SOLE_STAGE_P_DEVELOPMENT_PILOT__RT_OUTER_FOLD_00_SEED42"
            if key.outer_fold_index == PILOT_OUTER_FOLD_INDEX and key.seed == PILOT_SEED
            else "FUTURE_LATTICE_CONFIG_ONLY__NOT_EXECUTION_AUTHORITY"
        ),
        "fold_or_seed_substitution_permitted_for_actual_pilot": False,
        "future_45cell_lattice_config_does_not_authorize_queue_or_execution": True,
        "fixed_scientific_contract": _fixed_contract(),
        "source_authority": {
            "source_session_ids": list(source_ids),
            "source_session_count": 14,
            "selected_outer_fold_execution_receipt": source_binding[
                "selected_outer_fold_execution_receipt"],
            "source_authority_receipts": source_binding["source_authority_receipts"],
            "rt_full_source_manifest": source_binding["rt_full_source_manifest"],
            "rt_full_source_cost_aggregate": source_binding["rt_full_source_cost_aggregate"],
        },
        "development_target_query_authority_sha256": authority[
            "development_target_query_authority_sha256"],
        "materializer_plan_sha256": _sha_json(materialized_plan),
        "rt_local_asset_authority": {
            "canonical_body_path": str(rt_asset_authority.CANONICAL_OUTPUT),
            "body_sha256": LOCAL_ASSET_AUTHORITY_SHA256,
            "asset_count": 15,
            "canonical_internal_opener": (
                "track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority"
            ),
            "caller_ledger_path_sha_or_byte_count_permitted": False,
        },
        "sealed_rt_t4_target_query_lineage": t4,
        "output_topology": canonical_output_topology(key),
        "future_execution_chain": [
            "dual_CLI_authorization_before_torch_authority_or_data",
            "load_exact_canonical_config_and_official_preflight_pairs_same_fd",
            "validate_live_implementation_closure_and_fixed_GPU_cost_pair",
            "validate_root_frozen_synthetic_runtime_control_and_hard_null_pair",
            "assert_all_target_encoder_score_terminal_output_pairs_fresh",
            "internally_call_rt_local_asset_authority.open_canonical_verified_asset_after_authority",
            "copy_verified_held_FD_to_O_EXCL_private_snapshot_and_parse_from_held_snapshot_FD",
            "materialize_continuous_M24_support_and_strict_post_M_query",
            "prove_exact_T4_query_digests_target_bytes_endpoints_and_offset10_RFs",
            "fit_one_joint_exact14_source_plus_held_M24_encoder",
            "seal_joint_encoder_receipt_before_readouts",
            "fit_and_score_all_three_routes_times_linear_and_knn_on_same_encoder",
            "seal_six_score_receipts_then_one_terminal_cell_receipt",
        ],
        "implementation_closure": closure,
        "implementation_closure_sha256": _sha_json(closure),
        "target_data_opened": False,
        "target_query_opened": False,
        "cebra_imported": False,
        "cebra_fit_called": False,
        "gpu_used": False,
        "score_emitted": False,
        "receipt_minted": False,
    }
    return payload | {"canonical_config_sha256": _sha_json(payload)}


def build_no_target_preflight(*, outer_fold_index: int, seed: int) -> dict[str, Any]:
    """Validate all named non-target authorities and the cost gate; write nothing."""
    require_exact_development_pilot(outer_fold_index=outer_fold_index, seed=seed)
    config = build_canonical_config(outer_fold_index=outer_fold_index, seed=seed)
    gate = subject_executor.inspect_fixed_d8it250_gpu_cost_receipt()
    cost_valid = rt_scaffold._cost_gate_valid(gate)
    control_gate = (
        inspect_root_frozen_runtime_control_pair(cost_gate=gate)
        if cost_valid else {
            "schema": RUNTIME_CONTROL_SCHEMA,
            "status": "NOT_INSPECTED_BEFORE_FIXED_GPU_COST_GATE",
            "canonical_body_path": str(RUNTIME_CONTROL_BODY),
            "canonical_sidecar_path": str(RUNTIME_CONTROL_BODY.with_name(
                f"{RUNTIME_CONTROL_BODY.name}.sha256")),
            "control_scale_and_threshold": "ROOT_PENDING_UNTIL_FIXED_GPU_COST_REVIEW",
            "target_access_permitted": False,
        }
    )
    control_valid = control_gate.get("status") == RUNTIME_CONTROL_STATUS
    if not cost_valid:
        status = STATUS_COST_NO_GO
    elif not control_valid:
        status = STATUS_CONTROL_NO_GO
    else:
        status = STATUS_REVIEW_ONLY
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "status": status,
        "cell_key": config["cell_key"],
        "canonical_config": config,
        "canonical_config_sha256": config["canonical_config_sha256"],
        "fixed_gpu_cost_gate": gate,
        "root_frozen_runtime_control_gate": control_gate,
        "cost_gate_is_required_but_not_sufficient": True,
        "runtime_control_hard_null_gate_required_before_target": True,
        "root_review_and_immutable_canonical_config_preflight_pairs_required": True,
        "target_access_permitted": False,
        "execute_permitted": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_fit_called": False,
        "gpu_used": False,
        "score_emitted": False,
        "receipt_minted": False,
    }
    return payload | {"preflight_sha256": _sha_json(payload)}


def _pair_fresh(path: Path, *, label: str) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    require(not os.path.lexists(path) and not os.path.lexists(sidecar),
            f"{label} body/sidecar must both be fresh")


def assert_future_result_outputs_fresh(config: Mapping[str, Any]) -> None:
    topology = config.get("output_topology")
    require(isinstance(topology, Mapping), "config output topology missing")
    for key in ("target_materialization_receipt", "joint_encoder_receipt", "terminal_receipt"):
        _pair_fresh(Path(str(topology.get(key))), label=key)
    scores = topology.get("score_receipts")
    require(isinstance(scores, Mapping) and len(scores) == 6, "score output topology drift")
    for role, raw in scores.items():
        _pair_fresh(Path(str(raw)), label=f"score {role}")


def _canonical_config_preflight_pairs(key: CellKey) -> tuple[sealed.ExplicitSealedReceiptPair, sealed.ExplicitSealedReceiptPair]:
    topology = canonical_output_topology(key)
    config_path = Path(topology["canonical_config"])
    preflight_path = Path(topology["official_preflight"])
    return (
        sealed.ExplicitSealedReceiptPair(
            role="rt_one_cell_canonical_config", body_path=config_path,
            sidecar_path=config_path.with_name(f"{config_path.name}.sha256"),
        ),
        sealed.ExplicitSealedReceiptPair(
            role="rt_one_cell_official_preflight", body_path=preflight_path,
            sidecar_path=preflight_path.with_name(f"{preflight_path.name}.sha256"),
        ),
    )


def load_canonical_config_and_official_preflight(
    *, outer_fold_index: int, seed: int,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Load exact canonical immutable pairs and compare with all live authorities."""
    key = require_exact_development_pilot(outer_fold_index=outer_fold_index, seed=seed)
    config_pair, preflight_pair = _canonical_config_preflight_pairs(key)
    try:
        config, config_body_sha = sealed._strict_readonly_pair(config_pair)
        preflight, preflight_body_sha = sealed._strict_readonly_pair(preflight_pair)
    except sealed.TrackBV2LiveContractError as exc:
        raise TrackBV2RTOneCellError(str(exc)) from exc
    live_config = build_canonical_config(outer_fold_index=outer_fold_index, seed=seed)
    require(config == live_config, "published canonical config differs from current live authority/closure")
    _body_sha(config, declared_key="canonical_config_sha256")
    live_preflight = build_no_target_preflight(outer_fold_index=outer_fold_index, seed=seed)
    require(live_preflight.get("status") == STATUS_REVIEW_ONLY,
            "current fixed GPU cost gate cannot authorize an official preflight")
    require(preflight == live_preflight, "official preflight differs from current live no-target preflight")
    _body_sha(preflight, declared_key="preflight_sha256")
    return config, preflight, config_body_sha, preflight_body_sha


def publish_canonical_config_and_preflight_for_root_review(
    *, outer_fold_index: int, seed: int, root_reviewed: bool,
) -> dict[str, Any]:
    """Publish the two canonical pairs transactionally after an explicit root review.

    Neither CLI calls this function.  Current missing cost evidence makes it
    fail before directories or files are created.
    """
    require(root_reviewed is True, "canonical config/preflight publication requires root review")
    require_exact_development_pilot(outer_fold_index=outer_fold_index, seed=seed)
    preflight = build_no_target_preflight(outer_fold_index=outer_fold_index, seed=seed)
    require(preflight.get("status") == STATUS_REVIEW_ONLY,
            "cannot publish canonical config/preflight until fixed GPU cost gate is valid")
    config = preflight["canonical_config"]
    key = canonical_cell_key(outer_fold_index=outer_fold_index, seed=seed)
    config_pair, preflight_pair = _canonical_config_preflight_pairs(key)
    require(not os.path.lexists(config_pair.body_path.parent),
            "canonical cell root must be fresh before config/preflight publication")
    config_pair.body_path.parent.mkdir(parents=True, exist_ok=False)
    config_written = False
    try:
        config_result = _write_immutable_pair_for_root_review(
            config_pair.body_path, config, root_reviewed=True,
        )
        config_written = True
        preflight_result = _write_immutable_pair_for_root_review(
            preflight_pair.body_path, preflight, root_reviewed=True,
        )
        return {"canonical_config": config_result, "official_preflight": preflight_result}
    except BaseException:
        if config_written:
            config_pair.sidecar_path.unlink(missing_ok=True)
            config_pair.body_path.unlink(missing_ok=True)
        preflight_pair.sidecar_path.unlink(missing_ok=True)
        preflight_pair.body_path.unlink(missing_ok=True)
        try:
            config_pair.body_path.parent.rmdir()
        except OSError:
            pass
        raise


def _canonical_held_asset_context_for_separately_reviewed_runtime(key: CellKey) -> Any:
    """The sole future held-asset gateway; accepts no path, ledger, size, or SHA."""
    return rt_asset_authority.open_canonical_verified_asset_after_authority(key.target_session_id)


def execution_is_not_authorized_this_successor(*, outer_fold_index: int, seed: int) -> None:
    """No-target tripwire used by the CLI even when both execution flags appear."""
    require_exact_development_pilot(outer_fold_index=outer_fold_index, seed=seed)
    preflight = build_no_target_preflight(outer_fold_index=outer_fold_index, seed=seed)
    require(preflight["status"] == STATUS_REVIEW_ONLY,
            "fixed GPU cost and root-frozen runtime-control gates are not jointly valid; execution stops before target")
    config, _official, _config_sha, _preflight_sha = load_canonical_config_and_official_preflight(
        outer_fold_index=outer_fold_index, seed=seed,
    )
    assert_future_result_outputs_fresh(config)
    raise TrackBV2RTOneCellError(
        "this successor is contract/preflight only; root must separately review and seal a live runtime before target"
    )


def _require_cell(payload: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    require(payload.get("cell_key") == config.get("cell_key"), "receipt cell key/config drift")
    require(payload.get("canonical_config_sha256") == config.get("canonical_config_sha256"),
            "receipt canonical config SHA drift")


def validate_target_materialization_receipt(payload: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    require(payload.get("schema") == TARGET_RECEIPT_SCHEMA and
            payload.get("status") == "RT_M24_SUPPORT_AND_STRICT_POST_M_QUERY_MATERIALIZED__NO_FIT",
            "target materialization receipt schema/status drift")
    _require_cell(payload, config)
    require(payload.get("formal") is False, "target materialization may not be formal")
    require(payload.get("rt_local_asset_authority_body_sha256") == LOCAL_ASSET_AUTHORITY_SHA256,
            "target receipt local asset authority drift")
    require(payload.get("canonical_internal_opener") ==
            "track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority",
            "target receipt did not use canonical internal same-FD opener")
    require(payload.get("caller_target_path_ledger_sha_or_bytes_used") is False,
            "target receipt accepted caller target identity")
    snapshot = payload.get("held_fd_private_snapshot")
    require(isinstance(snapshot, Mapping) and
            snapshot.get("source_opened_once_O_NOFOLLOW") is True and
            snapshot.get("source_size_sha_verified_on_same_fd") is True and
            snapshot.get("snapshot_created_O_EXCL_0600_then_fsync_0444") is True and
            snapshot.get("parser_consumed_held_snapshot_fd_not_path_reopen") is True and
            snapshot.get("source_and_snapshot_inode_revalidated_after_parse") is True,
            "target receipt held-FD/private-snapshot proof drift")
    support = payload.get("support")
    query = payload.get("query")
    require(isinstance(support, Mapping) and isinstance(query, Mapping), "target support/query receipt missing")
    require(support.get("rewarded_trial_ordinals") == list(range(24)) and
            support.get("continuous_prefix") is True and
            support.get("all_intervening_rows") is True and
            support.get("rewarded_segments_concatenated") is False,
            "target M24 support contract drift")
    lineage = config.get("sealed_rt_t4_target_query_lineage", {})
    require(query.get("query_window_count") == lineage.get("query_window_count") and
            query.get("sealed_query_identity") == lineage.get("sealed_query_identity"),
            "target query does not exactly bind sealed T4 identity")
    require(query.get("strict_post_M24") is True and query.get("query_enters_any_fit") is False,
            "target query split/fit contract drift")
    require(query.get("prediction_endpoints_equal_valid_starts_plus_49") is True and
            query.get("every_offset10_RF_wholly_in_query") is True and
            query.get("support_query_boundary_crossings") == 0 and
            query.get("offset10_strictly_future_bins") == 4,
            "target endpoint/offset10 proof drift")
    t4_sha = query.get("ordered_t4_target_float32_bytes_sha256")
    cebra_sha = query.get("ordered_cebra_target_float32_bytes_sha256")
    require(_valid_sha(t4_sha) and t4_sha == cebra_sha,
            "ordered T4 and CEBRA target float32 bytes are not exact-equal")
    for key in ("support_neural_float32_sha256", "support_dense_velocity_float32_sha256",
                "query_neural_float32_sha256", "ordered_endpoint_int64_sha256",
                "full_offset10_RF_int64_sha256"):
        require(_valid_sha(payload.get(key)), f"target materialization {key} missing")
    require(payload.get("target_backprop_or_update") is False and
            payload.get("score_emitted") is False, "target materialization performed update/score")
    return _body_sha(payload, declared_key="target_materialization_receipt_sha256")


def _validate_valid_transform_alignment(
    alignment: Mapping[str, Any], config: Mapping[str, Any],
) -> str:
    require(alignment.get("schema") == "track_b_v2_rt_offset10_valid_transform_alignment_v1",
            "readout alignment schema drift")
    require(alignment.get("offset_crop") == {"left": 5, "right": 5} and
            alignment.get("each_block_cropped_separately") is True and
            alignment.get("concatenate_then_crop_used") is False and
            alignment.get("padded_edge_rows_used") == 0,
            "readout alignment crop policy drift")
    sources = alignment.get("source_blocks")
    source_ids = config.get("source_authority", {}).get("source_session_ids")
    require(isinstance(sources, list) and isinstance(source_ids, list) and len(sources) == 14 and
            [row.get("session_id") for row in sources if isinstance(row, Mapping)] == source_ids,
            "readout alignment exact14 source block order drift")
    blocks = list(sources)
    support = alignment.get("held_M24_support_block")
    query = alignment.get("strict_post_M24_query_block")
    require(isinstance(support, Mapping) and isinstance(query, Mapping),
            "readout alignment held support/query block missing")
    blocks.append(support)
    blocks.append(query)
    for index, block in enumerate(blocks):
        require(isinstance(block, Mapping), "readout alignment block malformed")
        input_rows = block.get("padded_transform_rows")
        valid_rows = block.get("valid_rows_after_separate_5_to_minus5_crop")
        require(type(input_rows) is int and type(valid_rows) is int and input_rows >= 11 and
                valid_rows == input_rows - 10 and
                block.get("embedding_row_count") == valid_rows and
                block.get("auxiliary_label_row_count") == valid_rows,
                f"readout alignment block {index} row parity/crop drift")
        for key in (
            "ordered_valid_endpoint_int64_sha256", "full_valid_offset10_RF_int64_sha256",
            "valid_embedding_float32_sha256", "aligned_auxiliary_float32_sha256",
        ):
            require(_valid_sha(block.get(key)), f"readout alignment block {index} {key} missing")
    require(support.get("support_trial_count") == 24 and support.get("block_role") == "held_M24_support",
            "readout alignment held support role/trial drift")
    require(query.get("block_role") == "strict_post_M24_query" and
            query.get("valid_rows_after_separate_5_to_minus5_crop") ==
            config.get("sealed_rt_t4_target_query_lineage", {}).get("query_window_count") and
            query.get("enters_any_fit") is False and query.get("RF_wholly_inside_query") is True,
            "readout alignment strict query crop/fit drift")
    declared = alignment.get("alignment_authority_sha256")
    require(_valid_sha(declared), "readout alignment authority SHA missing")
    bare = dict(alignment)
    bare.pop("alignment_authority_sha256", None)
    require(declared == _sha_json(bare), "readout alignment authority SHA drift")
    return str(declared)


def validate_encoder_receipt(
    payload: Mapping[str, Any], config: Mapping[str, Any], *, target_receipt_sha256: str,
) -> str:
    require(payload.get("schema") == ENCODER_RECEIPT_SCHEMA and
            payload.get("status") == "ONE_JOINT_RT_ENCODER_TERMINAL__READY_FOR_SIX_READOUTS",
            "encoder receipt schema/status drift")
    _require_cell(payload, config)
    require(payload.get("target_materialization_receipt_sha256") == target_receipt_sha256,
            "encoder/target receipt SHA drift")
    contract = config.get("fixed_scientific_contract", {})
    require(payload.get("geometry") == {"output_dimension": 8, "iterations": 10_000} and
            payload.get("seed") == config.get("cell_key", {}).get("seed"),
            "encoder geometry/seed drift")
    require(payload.get("source_session_ids") == config.get("source_authority", {}).get("source_session_ids") and
            payload.get("source_stream_count") == contract.get("source_stream_count") == 14,
            "encoder exact14 source roster drift")
    require(payload.get("held_support_trials") == 24 and
            payload.get("fit_stream_count") == 15 and
            payload.get("fit_count") == 1 and
            payload.get("held_query_enters_fit") is False,
            "joint encoder fit scope/count drift")
    require(_valid_sha(payload.get("encoder_state_sha256")) and
            _valid_sha(payload.get("fit_index_authority_sha256")),
            "encoder state/fit index SHA missing")
    alignment = payload.get("valid_transform_alignment")
    require(isinstance(alignment, Mapping), "encoder valid-transform alignment missing")
    alignment_sha = _validate_valid_transform_alignment(alignment, config)
    require(payload.get("valid_transform_alignment_sha256") == alignment_sha,
            "encoder valid-transform alignment binding drift")
    require(payload.get("target_backprop_or_update_after_fit") is False and
            payload.get("score_emitted") is False, "encoder receipt claims target update/score")
    return _body_sha(payload, declared_key="encoder_receipt_sha256")


def validate_score_receipt(
    payload: Mapping[str, Any], config: Mapping[str, Any], *, target_receipt_sha256: str,
    encoder_receipt_sha256: str, encoder_state_sha256: str,
    valid_transform_alignment_sha256: str,
) -> str:
    require(payload.get("schema") == SCORE_RECEIPT_SCHEMA and
            payload.get("status") == "ONE_RT_ROUTE_DECODER_SCORE_TERMINAL",
            "score receipt schema/status drift")
    _require_cell(payload, config)
    route = payload.get("route")
    decoder = payload.get("decoder")
    require(route in ROUTES and decoder in DECODERS, "score route/decoder is not canonical")
    require(payload.get("target_materialization_receipt_sha256") == target_receipt_sha256 and
            payload.get("encoder_receipt_sha256") == encoder_receipt_sha256 and
            payload.get("encoder_state_sha256") == encoder_state_sha256,
            "score target/encoder lineage drift")
    require(payload.get("valid_transform_alignment_sha256") == valid_transform_alignment_sha256 and
            _valid_sha(valid_transform_alignment_sha256),
            "score valid-transform alignment SHA drift")
    require(payload.get("sealed_query_identity") ==
            config.get("sealed_rt_t4_target_query_lineage", {}).get("sealed_query_identity"),
            "score sealed T4 query identity drift")
    require(payload.get("ordered_t4_target_float32_bytes_sha256") ==
            payload.get("ordered_cebra_target_float32_bytes_sha256") and
            _valid_sha(payload.get("ordered_t4_target_float32_bytes_sha256")),
            "score target bytes are not exact T4/CEBRA equal")
    score = payload.get("r2_variance_weighted")
    require(type(score) is float and math.isfinite(score), "score is not one finite float")
    metric = payload.get("metric_runtime")
    require(isinstance(metric, Mapping) and
            metric.get("implementation") == "torchmetrics.regression.R2Score" and
            metric.get("torchmetrics_version") == "1.5.1" and
            metric.get("multioutput") == "variance_weighted" and
            metric.get("dtype") == "float32" and metric.get("device") == "cpu" and
            metric.get("update_scope") == "one_complete_ordered_fold_session_query_then_compute_once" and
            metric.get("update_call_count") == 1 and metric.get("compute_call_count") == 1 and
            metric.get("pooled_query_rows_across_folds") is False and
            metric.get("custom_numpy_float64_clone_used") is False,
            "score TorchMetrics 1.5.1 CPU float32 runtime contract drift")
    q_count = config.get("sealed_rt_t4_target_query_lineage", {}).get("query_window_count")
    require(metric.get("prediction_shape") == [q_count, 2] and
            metric.get("target_shape") == [q_count, 2] and
            metric.get("prediction_float32_bytes_sha256") ==
            payload.get("ordered_prediction_float32_bytes_sha256") and
            metric.get("target_float32_bytes_sha256") ==
            payload.get("ordered_t4_target_float32_bytes_sha256") and
            _valid_sha(payload.get("ordered_prediction_float32_bytes_sha256")),
            "score metric prediction/target shape or float32 byte binding drift")
    require(payload.get("query_enters_fit") is False and
            payload.get("target_backprop_or_encoder_update") is False and
            payload.get("formal") is False, "score fit/update/formal contract drift")
    readout_scope = payload.get("readout_fit_scope")
    expected_scope = {
        "source_only_consumer_mechanism_alignment": "source_fit_only",
        "target_support_only_standard_cebra_accuracy": "held_target_M24_support_only",
        "source_plus_target_support_hybrid_sensitivity": "source_fit_plus_held_target_M24_support",
    }[str(route)]
    require(readout_scope == expected_scope, "score readout fit scope drift")
    return _body_sha(payload, declared_key="score_receipt_sha256")


def validate_terminal_receipt(
    payload: Mapping[str, Any], config: Mapping[str, Any], *, target_payload: Mapping[str, Any],
    encoder_payload: Mapping[str, Any], score_payloads: Sequence[Mapping[str, Any]],
) -> str:
    require(payload.get("schema") == TERMINAL_RECEIPT_SCHEMA and
            payload.get("status") == "ONE_RT_CELL_TERMINAL__SIX_MANDATORY_SCORES_COMPLETE",
            "terminal receipt schema/status drift")
    _require_cell(payload, config)
    target_sha = validate_target_materialization_receipt(target_payload, config)
    encoder_sha = validate_encoder_receipt(
        encoder_payload, config, target_receipt_sha256=target_sha,
    )
    encoder_state = str(encoder_payload["encoder_state_sha256"])
    alignment_sha = str(encoder_payload["valid_transform_alignment_sha256"])
    require(len(score_payloads) == 6, "terminal cell requires exact six score receipts")
    actual: dict[str, str] = {}
    for score in score_payloads:
        score_sha = validate_score_receipt(
            score, config, target_receipt_sha256=target_sha,
            encoder_receipt_sha256=encoder_sha, encoder_state_sha256=encoder_state,
            valid_transform_alignment_sha256=alignment_sha,
        )
        role = f"{score['route']}__{score['decoder']}"
        require(role not in actual, "duplicate terminal route/decoder score")
        actual[role] = score_sha
    expected_roles = {f"{route}__{decoder}" for route in ROUTES for decoder in DECODERS}
    require(set(actual) == expected_roles, "terminal score route/decoder grid incomplete")
    require(payload.get("target_materialization_receipt_sha256") == target_sha and
            payload.get("encoder_receipt_sha256") == encoder_sha and
            payload.get("encoder_state_sha256") == encoder_state and
            payload.get("valid_transform_alignment_sha256") == alignment_sha and
            payload.get("score_receipt_sha256_by_role") == actual,
            "terminal receipt lineage map drift")
    require(payload.get("target_backprop_or_update") is False and
            payload.get("formal") is False, "terminal target-update/formal drift")
    require(payload.get("metric_aggregation_scope") ==
            "one_R2_per_fold_session_seed_route_decoder__aggregate_session_then_seed" and
            payload.get("pooled_query_rows_across_folds") is False,
            "terminal metric aggregation scope drift")
    return _body_sha(payload, declared_key="terminal_receipt_sha256")


def build_terminal_aggregate(
    *, terminal_records: Sequence[Mapping[str, Any]], config_by_cell: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Pure aggregate interface; accepts validated terminal summaries, no data."""
    require(len(terminal_records) == 45 and len(config_by_cell) == 45,
            "aggregate requires exact 15 folds x 3 seeds")
    expected = {(fold, seed) for fold in range(15) for seed in SEEDS}
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    for record in terminal_records:
        require(record.get("schema") == TERMINAL_RECEIPT_SCHEMA and
                record.get("status") == "ONE_RT_CELL_TERMINAL__SIX_MANDATORY_SCORES_COMPLETE",
                "aggregate terminal schema/status drift")
        key = record.get("cell_key")
        require(isinstance(key, Mapping), "aggregate terminal cell key missing")
        identity = (key.get("outer_fold_index"), key.get("seed"))
        require(identity in expected and identity not in seen, "aggregate cell missing/duplicate/noncanonical")
        seen.add(identity)  # type: ignore[arg-type]
        cell_id = f"rt_outer_fold_{identity[0]:02d}__seed_{identity[1]}"
        config = config_by_cell.get(cell_id)
        require(isinstance(config, Mapping) and record.get("canonical_config_sha256") ==
                config.get("canonical_config_sha256"), "aggregate terminal/config binding drift")
        require(_body_sha(config, declared_key="canonical_config_sha256") ==
                config.get("canonical_config_sha256"), "aggregate config self SHA drift")
        require(_body_sha(record, declared_key="terminal_receipt_sha256") ==
                record.get("terminal_receipt_sha256") and
                record.get("formal") is False and record.get("target_backprop_or_update") is False,
                "aggregate terminal SHA/formal/update drift")
        require(record.get("metric_aggregation_scope") ==
                "one_R2_per_fold_session_seed_route_decoder__aggregate_session_then_seed" and
                record.get("pooled_query_rows_across_folds") is False,
                "aggregate terminal metric pooling drift")
        scores = record.get("scores_r2_by_role")
        require(isinstance(scores, Mapping) and set(scores) ==
                {f"{r}__{d}" for r in ROUTES for d in DECODERS} and
                all(type(value) is float and math.isfinite(value) for value in scores.values()),
                "aggregate terminal score map drift")
        rows.append({"cell_id": cell_id, "cell_key": dict(key),
                     "terminal_receipt_sha256": record["terminal_receipt_sha256"],
                     "scores_r2_by_role": dict(scores)})
    require(seen == expected, "aggregate canonical 45-cell grid incomplete")
    rows.sort(key=lambda item: (item["cell_key"]["outer_fold_index"], item["cell_key"]["seed"]))
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "RT_DEVELOPMENT_45CELL_AGGREGATE__NOT_FORMAL",
        "cell_count": 45,
        "fold_count": 15,
        "seeds": list(SEEDS),
        "routes": list(ROUTES),
        "decoders": list(DECODERS),
        "rows": rows,
        "aggregation_policy": "session_then_seed__no_best_seed_route_or_decoder_selection",
        "one_R2_computed_per_fold_session_seed_route_decoder": True,
        "pooled_query_rows_across_folds": False,
        "interface_only_does_not_authorize_45cell_queue_or_execution": True,
        "formal": False,
        "target_backprop_or_update": False,
    }
    return payload | {"aggregate_receipt_sha256": _sha_json(payload)}


def _write_all(fd: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(fd, raw[offset:])
        if written <= 0:
            raise TrackBV2RTOneCellError("short immutable receipt write")
        offset += written


def _write_immutable_pair_for_root_review(
    path: Path, payload: Mapping[str, Any], *, root_reviewed: bool,
) -> dict[str, str]:
    """Transactional O_EXCL/0444 writer; not called by either CLI path."""
    require(root_reviewed is True, "immutable publication requires explicit root review")
    path = Path(path).absolute()
    sidecar = path.with_name(f"{path.name}.sha256")
    require(path.is_absolute(), "immutable output path must be absolute")
    require(path.parent.exists() and path.parent.is_dir() and not path.parent.is_symlink(),
            "immutable output parent must pre-exist as a real directory")
    require(path.parent.resolve() == path.parent,
            "immutable output parent may not resolve through a symlinked ancestor")
    _pair_fresh(path, label="immutable output")
    raw = base.canonical_json_bytes(dict(payload))
    digest = hashlib.sha256(raw).hexdigest()
    side_raw = f"{digest}  {path.name}\n".encode("ascii")
    body_fd: int | None = None
    side_fd: int | None = None
    body_owned = False
    side_owned = False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        body_fd = os.open(path, flags, 0o600)
        body_owned = True
        _write_all(body_fd, raw)
        os.fsync(body_fd)
        os.fchmod(body_fd, 0o444)
        os.close(body_fd)
        body_fd = None
        side_fd = os.open(sidecar, flags, 0o600)
        side_owned = True
        _write_all(side_fd, side_raw)
        os.fsync(side_fd)
        os.fchmod(side_fd, 0o444)
        os.close(side_fd)
        side_fd = None
        dirfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
        require(stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(sidecar.stat().st_mode) == 0o444,
                "immutable output pair mode drift")
        return {"body_path": str(path), "body_sha256": digest, "sidecar_path": str(sidecar)}
    except BaseException:
        if body_fd is not None:
            os.close(body_fd)
        if side_fd is not None:
            os.close(side_fd)
        if side_owned:
            sidecar.unlink(missing_ok=True)
        if body_owned:
            path.unlink(missing_ok=True)
        raise
