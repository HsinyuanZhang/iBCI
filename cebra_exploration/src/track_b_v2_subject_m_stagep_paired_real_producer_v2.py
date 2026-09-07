"""Execution-only v2 successor for the failed Subject-M paired producer.

The v1 result tree is immutable failure evidence and is never reused.  This
successor changes one scientific implementation detail only: normalizer array
SHAs are checked in the exact authority domain used by
``track_b_v2_source_adapter._array_sha256`` (dtype, shape, bytes), rather than
against a raw-byte-only digest.  The float32 values and every downstream
target, fit, query, and metric rule remain unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_subject_m_stagep_paired_real_producer as v1

_V1_IMPLEMENTATION_CLOSURE = v1.implementation_closure


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_subject_m_stagep_paired_real_producer_v2.py"
TEST = REPO_ROOT / "cebra_exploration/tests/test_track_b_v2_subject_m_stagep_paired_real_producer_v2.py"
NOTE = REPO_ROOT / "cebra_exploration/docs/TRACK_B_V2_SUBJECT_M_STAGEP_PAIRED_REAL_PRODUCER_V2.md"
RESULT_ROOT = REPO_ROOT / "cebra_exploration/results/track_b_v2_subject_m_stagep_paired_real_producer_v2"
ADDENDUM_PATH = RESULT_ROOT / "execution_addendum_v2.json"
V1_RESULT_ROOT = v1.RESULT_ROOT
V1_FAILURE_COMPLETION = (
    V1_RESULT_ROOT / "cells/sua/sub-M_ses-CO-20140307/seed_42/completion.json"
)
V1_FAILURE_COMPLETION_BODY_SHA256 = "ea51d191d0715ce12f37c0f0afb5a09e9ade17c7fb186bb0ef26ef1eb9fa1b59"
V1_FAILURE_PAYLOAD_SHA256 = "f0abed105c83a38202a595f6fb1c7ab82995611960ff6b1bd278db90b247805b"
V1_STDERR_SHA256 = "5304d86fe584b526cbd9703737a1e61f2fac03be5ee6a960c61923472654affc"

SCHEMA_REVIEW_PLAN = "track_b_v2_subject_m_stagep_paired_real_producer_review_v2"
SCHEMA_EXECUTION_ADDENDUM = "track_b_v2_subject_m_stagep_paired_real_producer_addendum_v2"
SCHEMA_TARGET_ACCESS_ATTEMPT = "track_b_v2_subject_m_stagep_target_access_attempt_v2"
SCHEMA_TARGET_LINEAGE = "track_b_v2_subject_m_stagep_prefit_target_lineage_v2"
STATUS_REVIEW = "NO_GO__V2_SUCCESSOR_REVIEW_BOUNDARY__EXECUTE_TRIPWIRE_ARMED"

PAIR_ORDER = v1.PAIR_ORDER
ROUTES = v1.ROUTES
DECODERS = v1.DECODERS
MODEL_CONTRACT = v1.MODEL_CONTRACT
TARGET_SESSION_ID = v1.TARGET_SESSION_ID
SEED = v1.SEED
V9_PREFLIGHT_SHA256 = v1.V9_PREFLIGHT_SHA256
V9_TARGET_SHA256 = v1.V9_TARGET_SHA256
V9_VALID_STARTS_SHA256 = v1.V9_VALID_STARTS_SHA256
V9_QUERY_COUNT = v1.V9_QUERY_COUNT
SCHEMA_START = v1.SCHEMA_START
SCHEMA_SOURCE = v1.SCHEMA_SOURCE
SCHEMA_TARGET = v1.SCHEMA_TARGET
SCHEMA_ENCODER = v1.SCHEMA_ENCODER
SCHEMA_SCORE = v1.SCHEMA_SCORE
SCHEMA_COMPLETION = v1.SCHEMA_COMPLETION
SCHEMA_TERMINAL = v1.SCHEMA_TERMINAL
SCHEMA_PAIRED_COMPLETION = v1.SCHEMA_PAIRED_COMPLETION

ViewExecutionCapability = v1.ViewExecutionCapability
SourceProducerOutput = v1.SourceProducerOutput
TargetProducerOutput = v1.TargetProducerOutput
JointEncoderOutput = v1.JointEncoderOutput


class TrackBV2SubjectMRealProducerV2Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2SubjectMRealProducerV2Error(message)


def _canonical_bytes(value: Any) -> bytes:
    return base.canonical_json_bytes(value)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical_bytes(value))


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _authority_array_sha256(value: Any) -> str:
    """Exact local copy of source_adapter._array_sha256's authority domain."""
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _topology(view: str) -> dict[str, Any]:
    require(view in PAIR_ORDER, "v2 producer topology view invalid")
    root = RESULT_ROOT / "cells" / view / TARGET_SESSION_ID / "seed_42"
    scores = {f"{route}__{decoder}": str(root / "scores" / f"{route}__{decoder}.json")
              for route in ROUTES for decoder in DECODERS}
    return {
        "cell_root": str(root), "start": str(root / "start.json"),
        "source": str(root / "source_materialization.json"),
        "target_access_attempt": str(root / "target_access_attempt.json"),
        "target_lineage": str(root / "prefit_target_lineage.json"),
        "target": str(root / "target_materialization.json"),
        "private_snapshot": str(root / "private_snapshot/held_target.nwb"),
        "encoder": str(root / "joint_encoder.json"),
        "checkpoint": str(root / "joint_encoder_state.pt"),
        "embeddings": str(root / "joint_embeddings.npz"), "scores": scores,
        "child_stdout": str(root / "child_stdout.raw"),
        "child_stderr": str(root / "child_stderr.raw"),
        "completion": str(root / "completion.json"), "terminal": str(root / "terminal.json"),
        "caller_path_override_permitted": False, "successor_version": "v2",
    }


def validate_immutable_v1_failure() -> dict[str, Any]:
    body = v1._read_same_fd(V1_FAILURE_COMPLETION, label="immutable v1 failure completion",
                            required_mode=0o444)
    side = v1._read_same_fd(Path(f"{V1_FAILURE_COMPLETION}.sha256"),
                            label="immutable v1 failure completion sidecar", required_mode=0o444)
    require(body.sha256 == V1_FAILURE_COMPLETION_BODY_SHA256 and
            side.raw == f"{body.sha256}  {V1_FAILURE_COMPLETION.name}\n".encode("ascii"),
            "immutable v1 failure completion pair drift")
    payload = v1._json_from_verified(body, label="immutable v1 failure completion")
    cell_root = V1_FAILURE_COMPLETION.parent
    forbidden = (
        cell_root / "target_materialization.json", cell_root / "joint_encoder.json",
        cell_root / "terminal.json", cell_root / "private_snapshot/held_target.nwb",
        V1_RESULT_ROOT / "cells/pseudo_mua",
    )
    require(payload.get("failure_completion_payload_sha256") == V1_FAILURE_PAYLOAD_SHA256 and
            payload.get("actual_exit_code") == 1 and
            payload.get("stderr_sha256") == V1_STDERR_SHA256 and
            payload.get("status") == "TERMINAL_CHILD_FAILURE__CELL_NONREUSABLE" and
            all(not os.path.lexists(path) for path in forbidden),
            "v1 terminal-failure provenance/status drift")
    return {
        "result_root": str(V1_RESULT_ROOT), "completion_path": str(V1_FAILURE_COMPLETION),
        "completion_body_sha256": body.sha256, "completion_sidecar_sha256": side.sha256,
        "failure_payload_sha256": V1_FAILURE_PAYLOAD_SHA256,
        "stderr_sha256_without_persisted_bytes": V1_STDERR_SHA256,
        "status": "IMMUTABLE_V1_FAILURE__SUPERSEDED_BY_V2__REEXECUTION_FORBIDDEN",
        "v1_output_reuse_permitted": False,
    }


def implementation_closure() -> dict[str, Any]:
    settled = _V1_IMPLEMENTATION_CLOSURE()
    files = {
        "successor_v2_core": v1._source_binding(Path(__file__), label="v2 producer core"),
        "successor_v2_cli": v1._source_binding(CLI, label="v2 producer CLI"),
        "successor_v2_tests": v1._source_binding(TEST, label="v2 producer tests"),
        "successor_v2_note": v1._source_binding(NOTE, label="v2 producer note"),
        "settled_v1_recursive_scientific_runtime": settled,
        "source_adapter_authority_hash_implementation": v1._source_binding(
            Path(v1.source_adapter.__file__), label="source adapter authority array hash"),
    }
    return {"files": files, "closure_sha256": _sha_json(files)}


@contextmanager
def _v2_runtime_context() -> Iterator[None]:
    """Temporarily route settled numerical helpers to v2 paths/closure."""
    saved = (v1.RESULT_ROOT, v1.ADDENDUM_PATH, v1._topology, v1.implementation_closure)
    v1.RESULT_ROOT = RESULT_ROOT
    v1.ADDENDUM_PATH = ADDENDUM_PATH
    v1._topology = _topology
    v1.implementation_closure = implementation_closure
    try:
        yield
    finally:
        v1.RESULT_ROOT, v1.ADDENDUM_PATH, v1._topology, v1.implementation_closure = saved


def build_no_target_review_plan() -> dict[str, Any]:
    v1_failure = validate_immutable_v1_failure()
    sources = {view: v1.load_source_authority_bundle(view)[1] for view in PAIR_ORDER}
    query = v1.load_v9_query_authority()
    payload = {
        "schema": SCHEMA_REVIEW_PLAN, "status": STATUS_REVIEW,
        "canonical_result_root": str(RESULT_ROOT), "pair_order": list(PAIR_ORDER),
        "cells": [v1.sealed_runtime.StagePCell.from_view(view).as_dict() for view in PAIR_ORDER],
        "fixed_primary_model": dict(MODEL_CONTRACT), "source_authorities": sources,
        "v9_post50_authority": query,
        "output_topology": {view: _topology(view) for view in PAIR_ORDER},
        "v1_failure_provenance": v1_failure,
        "normalizer_hash_domain_repair": {
            "v1_bug": "raw_bytes_hash_compared_to_dtype_shape_bytes_authority_hash",
            "v2_rule": "exact_source_adapter_array_sha256_dtype_shape_contiguous_bytes",
            "normalizer_values_changed": False, "scientific_read_rule_changed": False,
        },
        "pre_fit_target_evidence": [SCHEMA_TARGET_ACCESS_ATTEMPT, SCHEMA_TARGET_LINEAGE],
        "child_stream_evidence": "stdout_and_stderr_each_published_as_immutable_raw_pair",
        "implementation_closure": implementation_closure(),
        "execution_addendum": {"canonical_path": str(ADDENDUM_PATH),
                                "present_now": os.path.lexists(ADDENDUM_PATH)},
        "execute_enabled": False, "target_opened": False, "formal_data_opened": False,
        "cebra_imported": False, "gpu_used": False, "receipt_minted": False,
    }
    return payload | {"review_plan_sha256": _sha_json(payload)}


def bind_execution_capabilities(*, admissions: Mapping[str, Mapping[str, Any]],
                                reviewed_plan: Mapping[str, Any]) -> dict[str, ViewExecutionCapability]:
    expected = build_no_target_review_plan()
    require(dict(reviewed_plan) == expected and tuple(admissions) == PAIR_ORDER,
            "v2 reviewed plan/admission order drift")
    capabilities: dict[str, ViewExecutionCapability] = {}
    common: set[tuple[str, str, str]] = set()
    for view in PAIR_ORDER:
        admission = admissions[view]
        live = v1.sealed_runtime.build_stagep_live_admission(view=view)
        require(dict(admission) == live and admission.get("target_path_resolution_permitted") is False,
                f"{view} sealed admission drift")
        root = admission.get("root_authorization_pair", {}).get("body_sha256")
        control = admission.get("fixed_runtime_control_pair", {}).get("body_sha256")
        cost = admission.get("fixed_d8it250_gpu_cost_gate", {}).get("canonical_body_sha256")
        require(all(_valid_sha(value) for value in (root, control, cost)),
                f"{view} admission authority SHA missing")
        common.add((str(root), str(control), str(cost)))
        capabilities[view] = ViewExecutionCapability(
            view=view, cell=v1.sealed_runtime.StagePCell.from_view(view).as_dict(),
            admission_sha256=_sha_json(admission),
            official_preflight_body_sha256=str(admission["official_stagep_preflight_body_sha256"]),
            implementation_closure_sha256=str(admission["implementation_closure_sha256"]),
            source_authority_set_sha256=reviewed_plan["source_authorities"][view][
                "source_authority_set_sha256"], v9_preflight_sha256=V9_PREFLIGHT_SHA256)
    require(len(common) == 1, "paired v2 admissions do not share root/control/cost")
    return capabilities


def materialize_and_verify_strict27_source(capability: ViewExecutionCapability) -> SourceProducerOutput:
    return v1.materialize_and_verify_strict27_source(capability)


def materialize_target_from_private_snapshot(
        capability: ViewExecutionCapability, source: SourceProducerOutput) -> TargetProducerOutput:
    """Settled v1 target logic with only the normalizer hash-domain bug repaired."""
    import numpy as np

    require(source.capability == capability and source.payload.get("schema") == SCHEMA_SOURCE and
            source.payload.get("source_authority_exact_rebuild_match") is True,
            "v2 target requires exact strict27 source")
    _, source_authority = v1.load_source_authority_bundle(capability.view)
    scaler = source_authority["source_behavior_normalizer"]
    mean = np.ascontiguousarray(scaler.get("mean_float32"), dtype=np.float32)
    std = np.ascontiguousarray(scaler.get("std_float32"), dtype=np.float32)
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all() and
            np.isfinite(std).all() and np.all(std > 0) and
            _authority_array_sha256(mean) == scaler.get("mean_array_sha256") and
            _authority_array_sha256(std) == scaler.get("std_array_sha256"),
            "v2 source-only normalizer value/authority-domain drift")
    plan = v1.target_materializer.build_development_target_materializer_dry_plan(
        dataset="subject_m", view=capability.view,
        outer_fold_id=str(capability.cell["outer_fold_id"]), target_session_id=TARGET_SESSION_ID)
    require(plan.get("status") ==
            "CANONICAL_DEVELOPMENT_AUTHORITY_AND_SUBM_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS" and
            plan.get("target_session_id") == TARGET_SESSION_ID and plan.get("view") == capability.view,
            "v2 canonical target materializer plan drift")
    gate = plan.get("target_asset_ledger_gate")
    asset = gate.get("target_asset") if isinstance(gate, Mapping) else None
    require(isinstance(asset, Mapping) and asset.get("session_id") == TARGET_SESSION_ID and
            _valid_sha(asset.get("expected_sha256")) and type(asset.get("expected_bytes")) is int and
            asset["expected_bytes"] > 0 and isinstance(asset.get("a2_official_local_nwb_path"), str),
            "v2 canonical A2 target asset drift")
    snapshot_path = Path(_topology(capability.view)["private_snapshot"])
    require(not os.path.lexists(snapshot_path), "v2 private target snapshot must be fresh")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    loader, multi = v1._canonical_subject_m_modules()
    with v1.development_executor.open_verified_target_asset_private_snapshot(
            source_path=Path(asset["a2_official_local_nwb_path"]),
            expected_sha256=str(asset["expected_sha256"]), expected_bytes=int(asset["expected_bytes"]),
            snapshot_path=snapshot_path) as snapshot:
        record = loader.load_session_with_trials(
            Path(snapshot.parser_fd_path), 20, 50, 50, 100, -1.0, mean, std,
            cache_dir=None, signal_view=capability.view)
        neural, behavior, trials = v1._target_record_arrays(record, view=capability.view)
        replay: dict[str, Any] | None = None
        if capability.view == "pseudo_mua":
            sua_record = loader.load_session_with_trials(
                Path(snapshot.parser_fd_path), 20, 50, 50, 100, -1.0, mean, std,
                cache_dir=None, signal_view="sua")
            sua_neural, sua_behavior, sua_trials = v1._target_record_arrays(sua_record, view="sua")
            with loader.NWBHDF5IO(snapshot.parser_fd_path, "r") as io:
                electrode_ids = multi.electrode_ids_from_units(io.read().units.to_dataframe())
            pooled, channel_ids = multi.pool_spikes_by_electrode(sua_neural, electrode_ids)
            pooled = np.ascontiguousarray(pooled, dtype=np.float32)
            require(np.array_equal(pooled, neural) and np.array_equal(sua_behavior, behavior) and
                    sua_trials == trials, "v2 pMUA held-target pooling replay drift")
            replay = {
                "replay_exact_equal": True, "same_behavior_and_query_authority_as_sua": True,
                "input_sua_float32_sha256": v1._raw_array_sha(sua_neural),
                "electrode_ids_int64_sha256": v1._raw_array_sha(np.asarray(electrode_ids, dtype=np.int64)),
                "ordered_pooled_channel_ids_int64_sha256": v1._raw_array_sha(
                    np.asarray(channel_ids, dtype=np.int64)),
                "output_pmua_float32_sha256": v1._raw_array_sha(pooled),
                "source_unit_count": int(sua_neural.shape[1]),
                "pooled_channel_count": int(pooled.shape[1]),
            }
        valid_starts = np.ascontiguousarray(multi._compute_valid_starts(trials[50:], 50), dtype=np.int64)
        require(valid_starts.shape == (V9_QUERY_COUNT,) and
                v1._raw_array_sha(valid_starts) == V9_VALID_STARTS_SHA256,
                "v2 target valid-start authority drift")
        support_stop = int(trials[49]["stop"]); suffix_start = int(trials[50]["start"])
        require(0 < support_stop <= suffix_start < neural.shape[0],
                "v2 target support/suffix chronological boundary drift")
        target_values = np.ascontiguousarray(behavior[valid_starts + 49], dtype=np.float32)
        require(target_values.shape == (V9_QUERY_COUNT, 2) and
                v1._raw_array_sha(target_values) == V9_TARGET_SHA256,
                "v2 target behavior differs from V9/T4 bytes")
        snapshot_contract = snapshot.as_contract_dict()
        snapshot_contract.update({"parser_consumed_continuously_held_fd": True,
                                  "pathname_reopen_permitted": False,
                                  "parser_module": "eval_adaptation_dandi688.load_session_with_trials"})
    support_neural = np.ascontiguousarray(neural[:support_stop], dtype=np.float32)
    support_behavior = np.ascontiguousarray(behavior[:support_stop], dtype=np.float32)
    suffix_neural = np.ascontiguousarray(neural[suffix_start:], dtype=np.float32)
    suffix_behavior = np.ascontiguousarray(behavior[suffix_start:], dtype=np.float32)
    support_receipt = {
        "semantics": "CONTINUOUS_RAW_PREFIX_THROUGH_REWARDED_TRIAL50_STOP_EXCLUSIVE",
        "continuous_raw_prefix_start_inclusive": 0,
        "continuous_raw_prefix_stop_exclusive": support_stop, "through_rewarded_trial": 50,
        "all_intervening_raw_rows_retained": True,
        "support_neural_shape": list(support_neural.shape),
        "support_behavior_shape": list(support_behavior.shape),
        "support_neural_float32_sha256": v1._raw_array_sha(support_neural),
        "support_behavior_float32_sha256": v1._raw_array_sha(support_behavior),
        "source_only_normalizer": dict(scaler), "target_rows_in_normalizer_fit": 0,
        "normalizer_authority_hash_domain": "dtype_shape_contiguous_bytes",
    }
    asset_receipt = dict(asset) | {"ledger_gate_sha256": _sha_json(gate),
                                   "private_snapshot_only": True,
                                   "caller_path_or_SHA_permitted": False}
    return TargetProducerOutput(
        capability=capability, neural_support=support_neural, behavior_support=support_behavior,
        neural_suffix=suffix_neural, behavior_suffix=suffix_behavior, valid_starts=valid_starts,
        payload_inputs={"asset": asset_receipt, "support_receipt": support_receipt,
                        "private_snapshot": snapshot_contract, "pmua_replay": replay,
                        "support_stop_exclusive": support_stop, "suffix_start_raw": suffix_start})


def build_target_access_attempt_payload(
        *, capability: ViewExecutionCapability, source_receipt_body_sha256: str,
        source_payload_sha256: str, launcher_preflight_body_sha256: str,
        target_authority: Mapping[str, Any], caller_pid: int) -> dict[str, Any]:
    require(all(_valid_sha(value) for value in (source_receipt_body_sha256, source_payload_sha256,
                                                launcher_preflight_body_sha256)),
            "target-attempt predecessor SHA missing")
    payload = {
        "schema": SCHEMA_TARGET_ACCESS_ATTEMPT,
        "status": "TARGET_ACCESS_ATTEMPT_PUBLISHED_BEFORE_POSSIBLE_OPEN",
        "cell": dict(capability.cell), "caller_pid": int(caller_pid), "actual_started": True,
        "source_receipt_body_sha256": source_receipt_body_sha256,
        "source_payload_sha256": source_payload_sha256,
        "launcher_preflight_body_sha256": launcher_preflight_body_sha256,
        "target_authority": dict(target_authority),
        "target_opened_before_publication": False,
        "next_operation_may_open_target": True, "formal_data_opened": False,
    }
    return payload | {"target_access_attempt_payload_sha256": _sha_json(payload)}


def build_prefit_target_lineage_payload(
        *, target: TargetProducerOutput, source_receipt_body_sha256: str,
        target_access_attempt_body_sha256: str) -> dict[str, Any]:
    import numpy as np

    values = np.ascontiguousarray(
        target.behavior_suffix[target.valid_starts + 49 - int(target.payload_inputs["suffix_start_raw"])],
        dtype=np.float32)
    endpoints = np.ascontiguousarray(target.valid_starts + 49, dtype=np.int64)
    receptive_fields = np.ascontiguousarray(
        endpoints[:, None] + np.arange(-5, 5, dtype=np.int64)[None, :], dtype=np.int64)
    support_stop = int(target.payload_inputs["support_stop_exclusive"])
    suffix_start = int(target.payload_inputs["suffix_start_raw"])
    suffix_stop = suffix_start + int(target.neural_suffix.shape[0])
    require(v1._raw_array_sha(target.valid_starts) == V9_VALID_STARTS_SHA256 and
            v1._raw_array_sha(values) == V9_TARGET_SHA256 and
            np.all(receptive_fields[:, 0] >= suffix_start) and
            np.all(receptive_fields[:, 0] >= support_stop) and
            np.all(receptive_fields[:, -1] < suffix_stop),
            "prefit target lineage V9 endpoints/target bytes drift")
    inputs = target.payload_inputs
    payload = {
        "schema": SCHEMA_TARGET_LINEAGE,
        "status": "HELD_FD_TARGET_MATERIALIZED_AND_V9_VALIDATED__BEFORE_CEBRA_FIT",
        "cell": dict(target.capability.cell),
        "source_receipt_body_sha256": source_receipt_body_sha256,
        "target_access_attempt_body_sha256": target_access_attempt_body_sha256,
        "asset": dict(inputs["asset"]), "support": dict(inputs["support_receipt"]),
        "private_snapshot": dict(inputs["private_snapshot"]),
        "valid_starts_int64_sha256": v1._raw_array_sha(target.valid_starts),
        "ordered_prediction_endpoint_int64_sha256": v1._raw_array_sha(endpoints),
        "ordered_offset10_RF_int64_sha256": v1._raw_array_sha(receptive_fields),
        "ordered_target_behavior_float32_sha256": v1._raw_array_sha(values),
        "query_row_count": int(target.valid_starts.shape[0]),
        "support_stop_exclusive": support_stop, "suffix_start_raw": suffix_start,
        "suffix_stop_raw_exclusive": suffix_stop, "offset": [5, 5],
        "every_RF_wholly_inside_held_suffix": True,
        "every_RF_support_disjoint": True, "target_query_enters_fit": False,
        "cebra_fit_started": False, "gpu_fit_started": False, "formal_data_opened": False,
    }
    return payload | {"target_lineage_payload_sha256": _sha_json(payload)}


def build_execution_addendum_candidate(*, admissions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    plan = build_no_target_review_plan()
    capabilities = bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)
    payload = {
        "schema": SCHEMA_EXECUTION_ADDENDUM,
        "status": "ROOT_REVIEWED_V2_EXECUTION_SUCCESSOR__NO_SCIENTIFIC_READ_CHANGE",
        "canonical_path": str(ADDENDUM_PATH), "canonical_result_root": str(RESULT_ROOT),
        "pair_order": list(PAIR_ORDER),
        "cells": {view: dict(capabilities[view].cell) for view in PAIR_ORDER},
        "admission_sha256_by_view": {view: capabilities[view].admission_sha256 for view in PAIR_ORDER},
        "producer_implementation_closure": implementation_closure(),
        "fixed_model_contract": dict(MODEL_CONTRACT), "v9_preflight_sha256": V9_PREFLIGHT_SHA256,
        "v1_failure_provenance": plan["v1_failure_provenance"],
        "v1_reexecution_or_output_reuse_permitted": False,
        "normalizer_values_changed": False, "scientific_read_rule_unchanged": True,
        "stdout_stderr_raw_pairs_required": True,
        "target_access_attempt_before_target_helper_required": True,
        "prefit_target_lineage_required": True,
        "authorizes_target_or_GPU_by_itself": False,
        "target_opened_while_building": False, "formal_data_opened": False,
    }
    return payload | {"execution_addendum_payload_sha256": _sha_json(payload)}


def publish_execution_addendum(*, admissions: Mapping[str, Mapping[str, Any]],
                               i_have_independent_root_review: bool = False) -> dict[str, Any]:
    require(i_have_independent_root_review is True, "v2 addendum mint requires root review")
    require(not os.path.lexists(ADDENDUM_PATH) and
            not os.path.lexists(Path(f"{ADDENDUM_PATH}.sha256")), "v2 addendum pair not fresh")
    ADDENDUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_execution_addendum_candidate(admissions=admissions)
    binding = v1.publish_json_pair(ADDENDUM_PATH, payload)
    try:
        require(build_execution_addendum_candidate(admissions=admissions) == payload,
                "v2 addendum launch/final closure drift")
        loaded = load_execution_addendum(admissions=admissions)
        require(loaded["payload"] == payload, "v2 addendum reload drift")
        return binding | {"payload_sha256": payload["execution_addendum_payload_sha256"]}
    except BaseException:
        v1._rollback_owned_raw_pair(binding)
        raise


def load_execution_addendum(*, admissions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    body = v1._read_same_fd(ADDENDUM_PATH, label="v2 execution addendum", required_mode=0o444)
    side = v1._read_same_fd(Path(f"{ADDENDUM_PATH}.sha256"),
                            label="v2 execution addendum sidecar", required_mode=0o444)
    payload = v1._json_from_verified(body, label="v2 execution addendum")
    require(side.raw == f"{body.sha256}  {ADDENDUM_PATH.name}\n".encode("ascii") and
            payload == build_execution_addendum_candidate(admissions=admissions),
            "v2 execution addendum pair/current closure drift")
    return {"payload": payload, "body_sha256": body.sha256,
            "sidecar_sha256": side.sha256, "read_once_from_verified_fd": True}


# Stable numerical/persistence surface delegated to the settled v1 code.  Only
# the two path/closure-sensitive helpers are wrapped in the v2 context.
load_source_authority_bundle = v1.load_source_authority_bundle
load_v9_query_authority = v1.load_v9_query_authority
build_contiguous_fit_block = v1.build_contiguous_fit_block
build_sparse_v9_query_block = v1.build_sparse_v9_query_block
fit_one_joint_encoder = v1.fit_one_joint_encoder
score_all_six_readouts = v1.score_all_six_readouts
finalize_target_payload = v1.finalize_target_payload
build_target_payload = v1.build_target_payload
build_encoder_payload = v1.build_encoder_payload
build_score_payload = v1.build_score_payload
build_completion_payload = v1.build_completion_payload
build_paired_completion_payload = v1.build_paired_completion_payload
publish_immutable_raw_pair = v1.publish_immutable_raw_pair
publish_json_pair = v1.publish_json_pair
open_verified_raw_pair = v1.open_verified_raw_pair
_read_same_fd = v1._read_same_fd
_json_from_verified = v1._json_from_verified
_source_binding = v1._source_binding
_rollback_owned_raw_pair = v1._rollback_owned_raw_pair
_raw_array_sha = v1._raw_array_sha


def build_start_payload(**kwargs: Any) -> dict[str, Any]:
    with _v2_runtime_context():
        return v1.build_start_payload(**kwargs)


def persist_sklearn_checkpoint_and_embeddings(**kwargs: Any) -> dict[str, Any]:
    with _v2_runtime_context():
        return v1.persist_sklearn_checkpoint_and_embeddings(**kwargs)


def build_terminal_payload(**kwargs: Any) -> dict[str, Any]:
    with _v2_runtime_context():
        return v1.build_terminal_payload(**kwargs)


def verify_isolated_cuda_identity() -> dict[str, Any]:
    return v1.verify_isolated_cuda_identity()
