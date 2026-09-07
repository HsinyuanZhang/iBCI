"""Canonical, no-target development authority for Track-B v2 target folds.

This is intentionally a *receipt graph verifier*, not a data adapter.  It
opens only already-sealed Track-B v2 metric-pointer and source-authority JSON
pairs.  It never accepts an NWB/NPZ path, discovers a target session, imports
CEBRA, builds an embedding, fits a readout, or writes a receipt.

The pre-existing target-query scaffold correctly describes the support/query
and temporal contract, but its six source SHA inputs are deliberately generic
for synthetic tests.  A future live target adapter must instead first consume
the canonical authority rendered here: a real root metric pointer plus the
real strict-27 (subject-M) or materialized 15-fold (RT) source receipts.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import track_b_v2_contract as base
import track_b_v2_actual_cpu_successor_plans as fixed_successor
import track_b_v2_fixed_gpu_engineering as fixed_engineering
import track_b_v2_live_contract as live
import track_b_v2_metric_pointer_authority as metric_pointer
import track_b_v2_source_adapter as source_adapter
import track_b_v2_target_query_scaffold as target_scaffold


DEVELOPMENT_TARGET_AUTHORITY_SCHEMA = "track_b_v2_development_target_query_authority_v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS = _REPO_ROOT / "cebra_exploration" / "results"
_SUBJECT_M_SOURCE_ROOTS: dict[str, Path] = {
    "sua": _RESULTS / "track_b_v2_source_authority_20260814_strict27_sua_continuous_v2_dev",
    "pseudo_mua": _RESULTS / "track_b_v2_source_authority_20260814_strict27_pmua_continuous_v2_dev",
}
_RT_SOURCE_ROOT = _RESULTS / "track_b_v2_rt_15fold_source_authority_20260814_dev"
_RT_MANIFEST = _RT_SOURCE_ROOT / "rt_15fold_source_authority_manifest.json"
_RT_AGGREGATE = _RT_SOURCE_ROOT / "rt_15fold_source_authority_aggregate.json"
_RT_MANIFEST_SHA256 = "775212fbd800129eb32ca03a68e53089a0ac97746be748491a840f8a694d0e02"
_RT_AGGREGATE_SHA256 = "66dd4a42567de87d0113f1f3c312208308e150f1643e1f5ee21776033f508b35"

_MEMBER_FILENAMES = {
    "source_roster": "source_roster.json",
    "source_coverage": "source_coverage.json",
    "source_neural_input_authority": "source_neural_input_authority.json",
    "source_behavior_auxiliary_scaler_authority": "source_behavior_auxiliary_scaler_authority.json",
    "source_readout_embedding_identity_authority": "source_readout_embedding_identity_authority.json",
    "source_only_dual_geometry_selection_plan": "source_only_dual_geometry_selection_plan.json",
}
_MEMBER_SCHEMAS = {
    "source_roster": source_adapter.SOURCE_ROSTER_SCHEMA,
    "source_coverage": source_adapter.SOURCE_COVERAGE_SCHEMA,
    "source_neural_input_authority": source_adapter.SOURCE_NEURAL_AUTHORITY_SCHEMA,
    "source_behavior_auxiliary_scaler_authority": source_adapter.SOURCE_BEHAVIOR_AUTHORITY_SCHEMA,
    "source_readout_embedding_identity_authority": source_adapter.SOURCE_EMBEDDING_AUTHORITY_SCHEMA,
    "source_only_dual_geometry_selection_plan": source_adapter.SOURCE_SELECTOR_PLAN_SCHEMA,
}
_MEMBER_STATUSES = {
    "source_roster": "DEVELOPMENT_SOURCE_ONLY__NOT_CEBRA_EXECUTION__NOT_CITABLE",
    "source_coverage": "DEVELOPMENT_SOURCE_ONLY__FULL_SOURCE_RAW_COVERAGE__NOT_CEBRA_EXECUTION",
    "source_neural_input_authority": "DEVELOPMENT_SOURCE_ONLY__DIMENSION_AGNOSTIC_IDENTITY__NOT_CEBRA_EXECUTION",
    "source_behavior_auxiliary_scaler_authority": "DEVELOPMENT_SOURCE_ONLY__DENSE_BEHAVIOR_SCALER__NOT_CEBRA_EXECUTION",
    "source_readout_embedding_identity_authority": "DEVELOPMENT_SOURCE_ONLY__EXPLICIT_IDENTITY__NOT_CEBRA_EXECUTION",
    "source_only_dual_geometry_selection_plan": "SOURCE_ONLY_GEOMETRY_SELECTION_REQUIRED__NOT_EXECUTED__NO_CEBRA",
}


class TrackBV2DevelopmentTargetAuthorityError(live.TrackBV2LiveContractError):
    """Raised before a target session identifier can be used as live authority."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2DevelopmentTargetAuthorityError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(item in "0123456789abcdef" for item in value)


def _canonical_real_path(path: Path, *, label: str) -> Path:
    """Require the one expected non-symlink path before descriptor verification."""
    path = Path(path)
    require(path.is_absolute(), f"{label} path must be absolute")
    require(path.resolve() == path, f"{label} path may not traverse an alias or symlink")
    require(path.parent.is_dir() and not path.parent.is_symlink(), f"{label} parent is not a real directory")
    return path


def _pair_at(path: Path, *, role: str) -> live.ExplicitSealedReceiptPair:
    body = _canonical_real_path(path, label=role)
    return live.ExplicitSealedReceiptPair(
        role=role,
        body_path=body,
        sidecar_path=body.with_name(f"{body.name}.sha256"),
    )


def _strict_payload(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    return live._strict_readonly_pair(_pair_at(path, role=role))


def _current_fixed_canonical_contract() -> dict[str, Any]:
    """Bind root's no-selection geometry decision and its current code closure.

    The historical source bundle still contains a 27x12 selector *plan* from
    before the CPU-economics stop.  Its immutable SHA is valuable lineage, but
    it cannot authorize a geometry.  The successor/no-data engineering modules
    are the independent current authorities for the exact fixed constants.
    """
    successor_geometry = fixed_successor.FIXED_GEOMETRY.as_dict()
    engineering_geometry = fixed_engineering.FIXED_FINAL_GEOMETRY.as_dict()
    require(successor_geometry == engineering_geometry, "fixed successor/engineering geometry drift")
    require(successor_geometry == {"output_dimension": 8, "source_iterations": 10_000,
                                   "encoder_geometry_key": "d8-it10000"},
            "fixed canonical CEBRA geometry is no longer d8-it10000")
    require(fixed_successor.FIXED_NORMALIZED_LAMBDA == fixed_engineering.FIXED_NORMALIZED_RIDGE_LAMBDA == 0.01,
            "fixed successor/engineering ridge lambda drift")
    require(fixed_engineering.FIXED_KNN_K == 3, "fixed canonical cosine kNN k drift")
    modules = {
        "fixed_canonical_successor": Path(fixed_successor.__file__),
        "fixed_canonical_engineering": Path(fixed_engineering.__file__),
        "actual_cpu_route_geometry": Path(fixed_engineering.route.__file__),
    }
    closure: dict[str, dict[str, Any]] = {}
    for name, raw_path in modules.items():
        path = _canonical_real_path(raw_path, label=f"fixed canonical contract source {name}")
        # These are live code files rather than immutable receipts.  Reuse the
        # source adapter's one-FD no-follow reader so this emitted closure binds
        # the bytes that supplied the constants, not a racy pathname reread.
        raw = source_adapter._read_regular_file(path, label=f"fixed canonical contract source {name}")
        closure[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return {
        "schema": "track_b_v2_fixed_canonical_geometry_binding_v1",
        "status": "ROOT_FIXED_CANONICAL_GEOMETRY__NO_SOURCE_OR_TARGET_SELECTION",
        "embedding_geometry": successor_geometry,
        "linear_ridge_normalized_lambda": 0.01,
        "cosine_knn_k": 3,
        "source_geometry_selection_performed": False,
        "target_geometry_selection_performed": False,
        "source_selector_fit_count": 0,
        "target_selector_fit_count": 0,
        "linear_and_knn_readouts_reported_separately": True,
        "linear_and_knn_share_fixed_embedding_geometry": True,
        "posthoc_geometry_selection_permitted": False,
        "root_decision": {
            "legacy_27_fold_x_12_geometry_selector": "NO_GO_MEASURED_CPU_ECONOMICS",
            "grouped_12_fit_support_only_selector": "NOT_ADOPTED_CHANGED_EXPOSURE_UNKNOWN_DEPLOYMENT_BIAS",
            "fixed_geometry_is_not_selected_from_source_or_target_scores": True,
        },
        "implementation_closure": closure,
        "implementation_closure_sha256": _sha_json(closure),
        "target_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
    }


def _require_no_execution(payload: Mapping[str, Any], *, label: str) -> None:
    for key in (
        "target_data_opened", "target_query_opened", "formal_data_opened",
        "cebra_imported", "cebra_solver_called", "gpu_used", "score_emitted",
    ):
        if key in payload:
            require(payload[key] is False, f"{label}: prohibited execution flag {key} is not false")


def _payload_receipt_sha(payload: Mapping[str, Any], *, label: str) -> str:
    declared = payload.get("receipt_payload_sha256")
    require(_valid_sha(declared), f"{label}: receipt payload SHA missing or malformed")
    bare = dict(payload)
    bare.pop("receipt_payload_sha256", None)
    require(declared == _sha_json(bare), f"{label}: receipt payload SHA does not bind its own body")
    return str(declared)


def _require_scope(payload: Mapping[str, Any], *, dataset: str, view: str | None, label: str) -> None:
    require((payload.get("dataset"), payload.get("view")) == (dataset, view),
            f"{label}: dataset/view scope drift")
    _require_no_execution(payload, label=label)


def _validate_source_bundle(
    *, root: Path, dataset: str, view: str | None, expected_source_ids: tuple[str, ...],
    expected_rt_fold: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read six named pairs and prove their internal source-only SHA graph."""
    root = _canonical_real_path(root, label="source authority root")
    require(root.is_dir(), "source authority root does not exist")
    payloads: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, str]] = {}
    for name, filename in _MEMBER_FILENAMES.items():
        path = root / filename
        payload, body_sha = _strict_payload(path, role=f"development_source_authority_{name}")
        require(payload.get("schema") == _MEMBER_SCHEMAS[name], f"{name}: source authority schema drift")
        require(payload.get("status") == _MEMBER_STATUSES[name], f"{name}: source authority status drift")
        _require_scope(payload, dataset=dataset, view=view, label=name)
        payloads[name] = payload
        receipts[name] = {
            "body_path": str(path),
            "body_sha256": body_sha,
            "sidecar_path": str(path.with_name(f"{path.name}.sha256")),
            "receipt_payload_sha256": _payload_receipt_sha(payload, label=name),
        }

    roster = payloads["source_roster"]
    require(tuple(roster.get("source_session_ids", ())) == expected_source_ids,
            "source roster is not the exact canonical source session order")
    require(roster.get("target_session_id") is None, "source roster may not name an outer target")
    expected_loader_semantics_sha = _sha_json(live.build_sealed_loader_semantics_contract(dataset, view))
    require(roster.get("loader_semantics_sha256") == expected_loader_semantics_sha,
            "source roster no longer binds the current sealed canonical loader semantics")

    payload_sha = {name: item["receipt_payload_sha256"] for name, item in receipts.items()}
    coverage = payloads["source_coverage"]
    neural = payloads["source_neural_input_authority"]
    behavior = payloads["source_behavior_auxiliary_scaler_authority"]
    embedding = payloads["source_readout_embedding_identity_authority"]
    selector = payloads["source_only_dual_geometry_selection_plan"]
    require(coverage.get("source_roster_receipt_payload_sha256") == payload_sha["source_roster"],
            "source coverage/roster graph drift")
    require(neural.get("source_roster_receipt_payload_sha256") == payload_sha["source_roster"],
            "source neural/roster graph drift")
    require(neural.get("source_coverage_receipt_payload_sha256") == payload_sha["source_coverage"],
            "source neural/coverage graph drift")
    require(behavior.get("source_roster_receipt_payload_sha256") == payload_sha["source_roster"],
            "source behavior/roster graph drift")
    require(behavior.get("source_coverage_receipt_payload_sha256") == payload_sha["source_coverage"],
            "source behavior/coverage graph drift")
    require(behavior.get("source_neural_input_authority_payload_sha256") == payload_sha["source_neural_input_authority"],
            "source behavior/neural graph drift")
    require(embedding.get("source_roster_receipt_payload_sha256") == payload_sha["source_roster"],
            "source embedding/roster graph drift")
    require(embedding.get("source_neural_input_authority_payload_sha256") == payload_sha["source_neural_input_authority"],
            "source embedding/neural graph drift")
    require(embedding.get("source_behavior_auxiliary_scaler_authority_payload_sha256") == payload_sha["source_behavior_auxiliary_scaler_authority"],
            "source embedding/behavior graph drift")
    for key, name in (
        ("source_roster_receipt_payload_sha256", "source_roster"),
        ("source_coverage_receipt_payload_sha256", "source_coverage"),
        ("source_neural_input_authority_payload_sha256", "source_neural_input_authority"),
        ("source_behavior_auxiliary_scaler_authority_payload_sha256", "source_behavior_auxiliary_scaler_authority"),
        ("source_readout_embedding_identity_authority_payload_sha256", "source_readout_embedding_identity_authority"),
    ):
        require(selector.get(key) == payload_sha[name], f"source selector/{name} graph drift")
    require(neural.get("method") == "dimension_agnostic_identity_float32_binned_counts",
            "headline neural preprocessor is no longer dimension-agnostic identity")
    require(neural.get("parameter_count") == 0, "headline neural preprocessor gained parameters")
    require(neural.get("target_support_neural_standardization_permitted_in_headline") is False,
            "headline target neural standardization became permitted")
    require(coverage.get("auxiliary") == "continuous_velocity_dense_bin_level",
            "source auxiliary no longer dense continuous velocity")
    require(selector.get("cebra_imported") is False and selector.get("cebra_solver_called") is False,
            "historical source selector plan was unexpectedly executed")
    require(selector.get("target_data_opened") is False and selector.get("score_emitted") is False,
            "source selector may not contain target access or a score")

    if dataset == "subject_m":
        require(roster.get("outer_fold_lineage") is None, "subject-M shared strict27 roster may not claim a target fold")
        require(behavior.get("behavior_auxiliary_method") == "canonical_loader_final_auxiliary__no_second_stage_refit",
                "subject-M behavior authority no longer binds the one canonical scaler stage")
        require(behavior.get("second_behavior_refit_on_canonical_final_auxiliary_permitted") is False,
                "subject-M behavior authority permits a hidden second refit")
        if view == "pseudo_mua":
            coverage_rows = coverage.get("source_sessions")
            require(isinstance(coverage_rows, list) and len(coverage_rows) == len(expected_source_ids),
                    "pMUA source coverage row count drift")
            for row in coverage_rows:
                require(isinstance(row, Mapping), "pMUA source coverage row malformed")
                pooling = row.get("actual_pooling_provenance")
                neural_feature = row.get("neural_feature")
                require(isinstance(pooling, Mapping) and isinstance(neural_feature, Mapping),
                        "pMUA source row lacks pooling or neural feature authority")
                require(pooling.get("method") == "electrode_ids_from_units_then_pool_spikes_by_electrode"
                        and pooling.get("replay_equal") is True,
                        "pMUA source pooling replay provenance drift")
                require(pooling.get("output_pseudo_mua_feature_sha256") == neural_feature.get("sha256"),
                        "pMUA pooled feature SHA differs from source neural feature SHA")
    else:
        require(expected_rt_fold is not None, "RT source bundle requires its sealed outer-fold topology")
        lineage = roster.get("outer_fold_lineage")
        require(isinstance(lineage, Mapping), "RT source roster lacks outer-fold lineage")
        require(lineage.get("outer_fold_id") == expected_rt_fold["outer_fold_id"], "RT source fold ID drift")
        require(lineage.get("outer_fold_index") == expected_rt_fold["outer_fold_index"], "RT source fold index drift")
        require(lineage.get("opaque_held_out_target_session_id") == expected_rt_fold["opaque_held_out_target_session_id"],
                "RT source roster held-target lineage drift")
        require(lineage.get("held_out_target_data_opened") is False,
                "RT source bundle claims held target open")
        require(expected_rt_fold["opaque_held_out_target_session_id"] not in expected_source_ids,
                "RT held target appears in its source roster")

    return {
        "canonical_source_authority_root": str(root),
        "source_authority_receipts": receipts,
        "source_authority_receipt_payload_sha256s": payload_sha,
        "source_session_ids": list(expected_source_ids),
        "source_session_count": len(expected_source_ids),
        "source_authority_roles": {
            "source_roster": "sealed_development_source_lineage",
            "source_coverage": "sealed_development_source_lineage",
            "source_neural_input_authority": "headline_dimension_agnostic_identity_preprocessor",
            "source_behavior_auxiliary_scaler_authority": "source_fitted_dense_behavior_scaler",
            "source_readout_embedding_identity_authority": "source_only_mechanism_readout_lineage",
            "source_only_dual_geometry_selection_plan": "historical_unexecuted_lineage_not_authorizing_fixed_geometry",
        },
        "historical_selector_plan": {
            "body_sha256": receipts["source_only_dual_geometry_selection_plan"]["body_sha256"],
            "historical_status": selector["status"],
            "historical_selector_plan_executed": False,
            "historical_selector_plan_selected_geometry": False,
            "historical_selector_plan_authorizes_fixed_execution": False,
            "may_override_root_fixed_geometry": False,
        },
        "source_data_opened_by_authority_verifier": False,
        "target_data_opened_by_authority_verifier": False,
    }


def _canonical_pointer_validation(dataset: str, view: str | None) -> dict[str, Any]:
    body = metric_pointer.canonical_metric_pointer_body_path(dataset, view)
    pair = _pair_at(body, role="canonical_reference_body_pointer")
    return metric_pointer.validate_root_audited_metric_pointer_pair(
        dataset=dataset, view=view, pointer_pair=pair,
    )


def _subject_m_source_authority(*, view: str) -> dict[str, Any]:
    expected = source_adapter._strict27_train_ids()
    return _validate_source_bundle(
        root=_SUBJECT_M_SOURCE_ROOTS[view], dataset="subject_m", view=view,
        expected_source_ids=expected,
    )


def _subject_m_target_lineage_membership(*, view: str, target_session_id: str) -> dict[str, Any]:
    """Require one of the 15 pointer-bound V9 T4 external target sessions.

    The metric pointer validates the aggregate and companion body SHA, but it
    intentionally does not expose each target identifier.  This no-data gate
    reads that already pointer-bound *lineage* body through the metric
    authority module's same-FD sealed reader and derives only membership,
    seed, asset and query-count identity—not an outcome value.
    """
    spec = metric_pointer._SCOPE_SPECS[("subject_m", view)]["lineage"]["per_session_seed_body"]
    decoded, _raw = metric_pointer._read_legacy_spec(spec, label="subject-M target lineage membership")
    rows = decoded.get("cells")
    audited = metric_pointer._validate_subject_m_lineage_rows(rows, view=view)
    matching = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("arm") == "shared_t4" and row.get("view") == view
    ]
    records = sorted(
        ({
            "session_id": str(row["session_id"]),
            "seed": int(row["seed"]),
            "asset_id": str(row["asset_id"]),
            "query_window_count": int(row["query_window_count"]),
        } for row in matching),
        key=lambda item: (item["session_id"], item["seed"]),
    )
    selected = [item for item in records if item["session_id"] == target_session_id]
    require(len(selected) == 3, "subject-M target ID is absent from the pointer-bound 15x3 V9 lineage")
    require([item["seed"] for item in selected] == [42, 43, 44],
            "subject-M target lineage must bind exactly V9 seeds {42,43,44}")
    require(len({item["asset_id"] for item in selected}) == 1,
            "subject-M target lineage asset ID differs across sealed V9 seeds")
    require(len({item["query_window_count"] for item in selected}) == 1,
            "subject-M target lineage query window count differs across sealed V9 seeds")
    compact = [{key: value for key, value in item.items() if key != "session_id"} for item in selected]
    return {
        "lineage_body_sha256": spec["sha256"],
        "lineage_body_metric_authority": False,
        "target_session_id": target_session_id,
        "target_session_seed_set": [42, 43, 44],
        "target_asset_id": selected[0]["asset_id"],
        "target_query_window_count": selected[0]["query_window_count"],
        "target_seed_asset_query_identity_sha256": _sha_json({
            "role": "subject_m_v9_target_seed_asset_query_identity", "records": compact,
        }),
        "complete_15_session_x_3_seed_lineage_digest": audited["ordered_session_seed_r2_query_window_asset_sha256"],
        "target_data_opened": False,
    }


def _rt_source_authority(*, outer_fold_id: str, expected_pointer_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the completed root 15-fold manifest, selected receipt, and six pairs."""
    manifest, manifest_sha = _strict_payload(_RT_MANIFEST, role="rt_15fold_source_authority_manifest")
    aggregate, aggregate_sha = _strict_payload(_RT_AGGREGATE, role="rt_15fold_source_authority_aggregate")
    require(manifest_sha == _RT_MANIFEST_SHA256, "RT full source manifest SHA drift")
    require(aggregate_sha == _RT_AGGREGATE_SHA256, "RT full source aggregate SHA drift")
    require(manifest.get("schema") == "track_b_v2_rt_15fold_source_authority_manifest_v1", "RT source manifest schema drift")
    require(manifest.get("status") == "RT_SOURCE_ONLY_MANIFEST__NO_TARGET_NO_CEBRA_NO_SCORE", "RT source manifest status drift")
    require(manifest.get("outer_fold_count") == 15 and manifest.get("dataset") == "rt" and manifest.get("view") is None,
            "RT source manifest scope/topology drift")
    require(manifest.get("rt_metric_pointer_body_sha256") == expected_pointer_sha256,
            "RT source manifest does not bind the current sealed RT metric pointer")
    _require_no_execution(manifest, label="RT source manifest")
    require(aggregate.get("schema") == "track_b_v2_rt_15fold_source_authority_aggregate_v1", "RT source aggregate schema drift")
    require(aggregate.get("manifest_body_sha256") == manifest_sha, "RT source aggregate/manifest SHA drift")
    _require_no_execution(aggregate, label="RT source aggregate")
    plan = source_adapter.build_rt_15fold_source_authority_plan()
    source_adapter.validate_rt_15fold_source_authority_plan(plan)
    require(manifest.get("rt_15fold_source_authority_plan_sha256") == plan["rt_15fold_source_authority_plan_sha256"],
            "RT materialized source manifest/immutable topology plan drift")
    candidates = [item for item in plan["outer_folds"] if item["outer_fold_id"] == outer_fold_id]
    require(len(candidates) == 1, "RT outer fold is missing or ambiguous in canonical 15-fold topology")
    expected_fold = dict(candidates[0])
    manifest_folds = manifest.get("fold_execution_receipts")
    require(isinstance(manifest_folds, list) and len(manifest_folds) == 15, "RT full manifest fold receipt list drift")
    fold_entries = [item for item in manifest_folds if isinstance(item, Mapping) and item.get("outer_fold_id") == outer_fold_id]
    require(len(fold_entries) == 1, "RT full manifest selected fold receipt missing or ambiguous")
    entry = dict(fold_entries[0])
    expected_receipt = _RT_SOURCE_ROOT / "folds" / outer_fold_id / "rt_source_only_fold_execution_receipt.json"
    require(entry.get("body_path") == str(expected_receipt), "RT fold receipt path is not canonical")
    fold_payload, fold_sha = _strict_payload(expected_receipt, role="rt_source_only_fold_execution")
    require(entry.get("body_sha256") == fold_sha, "RT full manifest/fold receipt SHA drift")
    require(fold_payload.get("outer_fold_id") == outer_fold_id and fold_payload.get("outer_fold_index") == expected_fold["outer_fold_index"],
            "RT fold receipt identity drift")
    require(fold_payload.get("opaque_held_out_target_session_id") == expected_fold["opaque_held_out_target_session_id"],
            "RT fold receipt held target drift")
    require(tuple(fold_payload.get("source_session_ids", ())) == tuple(expected_fold["source_session_ids"]),
            "RT fold receipt source roster drift")
    _require_no_execution(fold_payload, label="RT fold receipt")
    fold_pointer = fold_payload.get("rt_metric_pointer")
    require(isinstance(fold_pointer, Mapping) and fold_pointer.get("body_sha256") == expected_pointer_sha256,
            "RT fold receipt does not bind the current sealed RT metric pointer")
    bundle = _validate_source_bundle(
        root=expected_receipt.parent, dataset="rt", view=None,
        expected_source_ids=tuple(expected_fold["source_session_ids"]), expected_rt_fold=expected_fold,
    )
    claimed = fold_payload.get("source_authority_receipt_sha256_by_member")
    require(isinstance(claimed, Mapping), "RT fold receipt source member SHA map missing")
    for name, receipt in bundle["source_authority_receipts"].items():
        require(claimed.get(name) == receipt["body_sha256"], f"RT fold receipt/{name} body SHA drift")
    return ({
        "rt_full_source_manifest": {
            "body_path": str(_RT_MANIFEST), "body_sha256": manifest_sha,
            "sidecar_path": str(_RT_MANIFEST.with_name(f"{_RT_MANIFEST.name}.sha256")),
        },
        "rt_full_source_cost_aggregate": {
            "body_path": str(_RT_AGGREGATE), "body_sha256": aggregate_sha,
            "sidecar_path": str(_RT_AGGREGATE.with_name(f"{_RT_AGGREGATE.name}.sha256")),
        },
        "rt_15fold_source_authority_plan_sha256": plan["rt_15fold_source_authority_plan_sha256"],
        "selected_outer_fold_execution_receipt": {
            "body_path": str(expected_receipt), "body_sha256": fold_sha,
            "sidecar_path": str(expected_receipt.with_name(f"{expected_receipt.name}.sha256")),
        },
        "selected_outer_fold_id": outer_fold_id,
        "opaque_held_out_target_session_id": expected_fold["opaque_held_out_target_session_id"],
        **bundle,
    }, expected_fold)


def _target_query_requirements(
    *, dataset: str, view: str | None, support_query_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze the required future target objects without accepting a data path."""
    return {
        "support_query_contract": dict(support_query_contract),
        "target_data_materialized": False,
        "target_data_discovery_permitted_by_this_authority": False,
        "target_support_must_bind": {
            "continuous_chronological_prefix_required": True,
            "all_intervening_rows_required": True,
            "rewarded_segment_concatenation_permitted": False,
            "support_raw_coverage_expansion_sha256": "REQUIRED_AT_SEPARATE_LIVE_TARGET_GATE",
            "target_feature_sha256": "REQUIRED_AT_SEPARATE_LIVE_TARGET_GATE",
            "dense_velocity_float32_bytes_sha256": "REQUIRED_AT_SEPARATE_LIVE_TARGET_GATE",
            "information_matched": False,
            "bias_direction": "favors_CEBRA_accuracy",
        },
        "target_query_must_bind": {
            "query_is_post_M_only": True,
            "ordered_valid_window_start_indices_sha256": "REQUIRED_AT_SEPARATE_LIVE_TARGET_GATE",
            "ordered_prediction_target_raw_bin_indices": "EXACTLY_VALID_WINDOW_START_PLUS_49",
            "ordered_t4_and_cebra_target_bytes_must_match": True,
            "t4_query_identity_and_parent_metric_implementation_required": True,
            "full_actual_offset10_receptive_field_per_endpoint_required": True,
            "receptive_field_must_lie_within_query_and_disjoint_from_support": True,
            # CEBRA's Offset(5, 5) expands an index with
            # arange(-left, right), hence [-5, ..., +4].  The right extent
            # includes the prediction-target bin itself; only +1..+4 are
            # strictly future relative to that target.
            "default_offset10_receptive_field_width_raw_bins": 10,
            "default_offset10_previous_raw_bins": 5,
            "default_offset10_right_extent_including_target_raw_bins": 5,
            "default_offset10_strictly_future_raw_bins": 4,
            "causal_temporal_exposure_matched": False,
            "bias_direction": "favors_CEBRA_accuracy",
            "online_or_latency_equivalent_language_permitted": False,
        },
        "target_data_opened": False,
        "target_query_opened": False,
    }


def build_development_target_query_authority(
    *, dataset: str, view: str | None, outer_fold_id: str, target_session_id: object,
    proposed_target_path: object | None = None, proposed_target_discovery: object | None = None,
    execution_requested: bool = False, device: str = "cpu",
) -> dict[str, Any]:
    """Render the one canonical, no-target authority plan for an outer fold.

    Pointer and source receipts are deliberately validated before the opaque
    target ID is coerced, ensuring a stale/missing authority pair fails before
    a later caller can exploit target-like input handling.
    """
    dataset, view = base.validate_scope(dataset, view)
    require(proposed_target_path is None, "development target authority accepts no target data path")
    require(proposed_target_discovery is None, "development target authority accepts no target discovery callable")
    require(execution_requested is False, "development target authority has no execution mode")
    require(device == "cpu", "development target authority is CPU/no-data only")
    pointer = _canonical_pointer_validation(dataset, view)
    fixed_contract = _current_fixed_canonical_contract()
    if dataset == "subject_m":
        source_binding = _subject_m_source_authority(view=str(view))
        expected_target_id: str | None = None
    else:
        source_binding, fold = _rt_source_authority(
            outer_fold_id=outer_fold_id, expected_pointer_sha256=str(pointer["pointer_body_sha256"]),
        )
        expected_target_id = str(fold["opaque_held_out_target_session_id"])
    target_id = target_scaffold._opaque_target_session_id(dataset, target_session_id)
    if expected_target_id is not None:
        require(target_id == expected_target_id, "RT target ID does not equal the sealed selected fold held-out ID")
        target_lineage_membership: dict[str, Any] = {
            "lineage_body_sha256": source_binding["selected_outer_fold_execution_receipt"]["body_sha256"],
            "lineage_body_metric_authority": False,
            "target_session_id": target_id,
            "membership_semantics": "sealed_RT_15fold_topology_selected_outer_fold_held_out_target",
            "target_data_opened": False,
        }
    else:
        target_lineage_membership = _subject_m_target_lineage_membership(
            view=str(view), target_session_id=target_id,
        )
        date = target_id.removeprefix("sub-M_ses-CO-")
        expected_outer_fold_id = f"subject_m_{view}_external_target_{date}"
        require(outer_fold_id == expected_outer_fold_id,
                "subject-M outer fold ID must canonically bind its view and target session date")
    split = target_scaffold.TargetSupportQueryTrialWindowContract(
        dataset=dataset, view=view, outer_fold_id=outer_fold_id, target_session_id=target_id,
    ).as_dict()
    readouts = target_scaffold.frozen_readout_authority_plan(dataset, view)
    # The root-attested legacy scaffold records Offset.right as
    # ``future_raw_bins``.  Preserve that sealed source byte-for-byte, but do
    # not propagate the ambiguous label into this successor authority:
    # Offset.right includes the current target.  Future execution consumes
    # this corrected successor rendering.
    readouts = dict(readouts)
    temporal = dict(readouts["temporal_exposure_policy"])
    temporal["default_offset10_model_offset"] = {
        "left_previous_raw_bins": 5,
        "right_extent_including_target_raw_bins": 5,
        "strictly_future_raw_bins_after_target": 4,
        "receptive_field_width_raw_bins": 10,
        "half_open_offsets_relative_to_target": [-5, 5],
    }
    readouts["temporal_exposure_policy"] = temporal
    payload = {
        "schema": DEVELOPMENT_TARGET_AUTHORITY_SCHEMA,
        "status": "SEALED_POINTER_AND_SOURCE_AUTHORITY_VALIDATED__NO_TARGET_DATA__NOT_EXECUTABLE",
        "dataset": dataset,
        "view": view,
        "outer_fold_id": outer_fold_id,
        "target_session_id": target_id,
        "metric_pointer_validation": pointer,
        "source_authority_binding": source_binding,
        "target_reference_lineage_membership": target_lineage_membership,
        "target_support_query_trial_window_contract": split,
        "target_query_authority_requirements": _target_query_requirements(
            dataset=dataset, view=view, support_query_contract=split["support_query_contract"],
        ),
        "fixed_canonical_geometry_contract": fixed_contract,
        "readout_authority_plan": readouts,
        "standard_model_arm": "cebra_joint_behavior",
        "accuracy_table_headline_readout": "target_support_only_standard_cebra_accuracy",
        "mandatory_readout_routes": list(readouts["readout_routes"]),
        "linear_and_knn_readouts_reported_separately_with_shared_fixed_geometry": True,
        "source_geometry_selection_performed": False,
        "target_geometry_selection_performed": False,
        "posthoc_model_geometry_seed_or_readout_selection_permitted": False,
        "future_live_gate_must_consume_this_authority": True,
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
    return payload | {"development_target_query_authority_sha256": _sha_json(payload)}
