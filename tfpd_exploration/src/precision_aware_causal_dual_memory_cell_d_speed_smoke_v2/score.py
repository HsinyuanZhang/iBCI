"""Descriptor-safe V2 diagnostic codec composed over the immutable V1 route.

V2 owns no parser, evaluator loop, model, transition law, or generic score
lifecycle. It holds V1's terminal failure graph, delegates physical execution
to the exact V1 runtime, and changes only the receipt-level engineering
numeric policy so the comparison cannot disappear again.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.causal_dual_memory_cell_d_score_v1 import score as shared_score
from src.precision_aware_causal_dual_memory_cell_d_speed_smoke_v1 import score as v1_score

from . import plan


AUTHORITY_TOPOLOGY = shared_score.AUTHORITY_TOPOLOGY
SCORE_TOPOLOGY = shared_score.SCORE_TOPOLOGY


class SpeedSmokeV2Error(RuntimeError):
    """Fail closed for V2 predecessor, diagnostic, or lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpeedSmokeV2Error(message)


def _json(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _digest(value: object) -> str:
    return plan.sha256_bytes(_json(value))


def _sha(value: object, label: str) -> str:
    try:
        return plan.require_sha256(value, label)
    except plan.SpeedSmokeV2PlanError as error:
        raise SpeedSmokeV2Error(str(error)) from error


def _directory_identity(path: Path, label: str) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise SpeedSmokeV2Error(f"speed-smoke V2 {label} directory is inaccessible") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SpeedSmokeV2Error(f"speed-smoke V2 {label} directory is noncanonical")
    return int(info.st_dev), int(info.st_ino)


def _no_follow_flag() -> int:
    value = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(value, int) or value == 0:
        raise SpeedSmokeV2Error("speed-smoke V2 durable reader requires O_NOFOLLOW")
    return value


def _validate_v1_failure_semantics(graph: Mapping[str, Mapping[str, object]]) -> tuple[dict[str, object], str]:
    """Validate the supplied V1 failure as a real post-comparison predecessor."""
    attempt = graph["attempt.json"]
    input_authority = graph["input_authority.json"]
    failure = graph["failure.json"]
    identity = attempt.get("identity")
    if not isinstance(identity, Mapping):
        raise SpeedSmokeV2Error("speed-smoke V2 V1 failure attempt identity is absent")
    identity_payload = dict(identity)
    identity_sha = _digest(identity_payload)
    if (
        failure.get("identity") != identity_payload
        or failure.get("attempt_sha256") != plan.V1_FAILURE_SHAS["attempt.json"]
        or failure.get("input_authority_sha256") != plan.V1_FAILURE_SHAS["input_authority.json"]
        or failure.get("schema") != "precision_aware_cdmd_speed_smoke_failure_v1"
        or failure.get("status") != "FAILED"
        or failure.get("stage") != "budget_m4"
        or failure.get("error_class") != "SpeedSmokeError"
        or failure.get("error_sha256") != plan.V1_FAILURE_ERROR_SHA256
        or failure.get("terminal_published") is not False
        or failure.get("checkpoint_opened") is not True
        or failure.get("cuda_initialized") is not True
        or failure.get("full_system_forward_count") != 1016
        or failure.get("group_forward_count") != 3493
        or any(failure.get(key) != 0 for key in (
            "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        ))
        or input_authority.get("identity_sha256") != identity_sha
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 V1 failure semantic/progress drift")
    return identity_payload, identity_sha


@dataclass(frozen=True)
class V1FailureBinding:
    """Held directory identity plus exact immutable V1 failure semantics."""

    directory_identity: tuple[int, int]
    historical_v1_identity: Mapping[str, object]
    historical_v1_identity_sha256: str

    def payload(self) -> dict[str, object]:
        _require(
            all(type(value) is int and value > 0 for value in self.directory_identity),
            "speed-smoke V2 V1 failure directory identity drift",
        )
        identity = dict(self.historical_v1_identity)
        _require(_digest(identity) == self.historical_v1_identity_sha256, "speed-smoke V2 V1 identity digest drift")
        body = {
            "schema": "precision_aware_cdmd_speed_smoke_v2_v1_failure_binding_v1",
            "contract": plan.V1_FAILURE.payload(),
            "score_directory_identity": list(self.directory_identity),
            "historical_v1_identity": identity,
            "historical_v1_identity_sha256": _sha(self.historical_v1_identity_sha256, "V1 failure identity"),
        }
        return {**body, "binding_sha256": _digest(body)}


def validate_v1_failure_binding(value: object) -> dict[str, object]:
    required = {
        "schema", "contract", "score_directory_identity", "historical_v1_identity",
        "historical_v1_identity_sha256", "binding_sha256",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_v2_v1_failure_binding_v1"
        or value.get("contract") != plan.V1_FAILURE.payload()
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 V1 failure binding schema drift")
    raw_directory = value.get("score_directory_identity")
    raw_identity = value.get("historical_v1_identity")
    raw_identity_sha = value.get("historical_v1_identity_sha256")
    if (
        not isinstance(raw_directory, list) or len(raw_directory) != 2
        or any(type(item) is not int or item <= 0 for item in raw_directory)
        or not isinstance(raw_identity, Mapping)
        or _sha(raw_identity_sha, "V1 failure identity") != raw_identity_sha
        or _digest(dict(raw_identity)) != raw_identity_sha
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 V1 failure binding topology drift")
    expected = V1FailureBinding(
        directory_identity=(int(raw_directory[0]), int(raw_directory[1])),
        historical_v1_identity=dict(raw_identity), historical_v1_identity_sha256=str(raw_identity_sha),
    ).payload()
    if dict(value) != expected:
        raise SpeedSmokeV2Error("speed-smoke V2 V1 failure binding canonical drift")
    return expected


def validate_v1_failed_speed_smoke(root: Path) -> V1FailureBinding:
    """Held-FD/no-follow read of exactly V1's attempt/input/failure pairs."""
    directory = Path(root).absolute() / plan.V1_SCORE_ROOT_RELATIVE
    try:
        directory_identity, graph = v1_score._read_exact_directory(
            directory, label="speed-smoke V2 immutable V1 failure",
            expected_sha256s=plan.V1_FAILURE_SHAS,
        )
    except v1_score.SpeedSmokeError as error:
        raise SpeedSmokeV2Error("speed-smoke V2 immutable V1 failure descriptor read failed") from error
    identity, identity_sha = _validate_v1_failure_semantics(graph)
    return V1FailureBinding(
        directory_identity=directory_identity, historical_v1_identity=identity,
        historical_v1_identity_sha256=identity_sha,
    )


def _typed_v1_identity(value: object) -> v1_score.SpeedSmokeIdentity:
    """Rehydrate V1 only from its canonical typed identity payload."""
    if not isinstance(value, Mapping):
        raise SpeedSmokeV2Error("speed-smoke V2 V1 runtime identity type drift")
    required = {
        "schema", "cell", "phase", "closure", "completed_precision_v2_binding",
        "completed_precision_v2_binding_sha256", "historical_v2_identity_sha256",
        "historical_v2_gpu1_evidence", "selected_device_profile", "runtime_v1_science_identity",
        "canonical_source_environment", "smoke_execution_order", "authority_root_relative",
        "score_root_relative", "m30_rerun", "sealed_comparator_rerun",
        "non_governing_engineering_smoke", "target_updates_forbidden",
    }
    if set(value) != required:
        raise SpeedSmokeV2Error("speed-smoke V2 V1 runtime identity schema drift")
    try:
        identity = v1_score.SpeedSmokeIdentity(
            closure=value["closure"], completed_v2_binding=value["completed_precision_v2_binding"],
            runtime_v1_identity=value["runtime_v1_science_identity"],
        )
    except (KeyError, TypeError, v1_score.SpeedSmokeError) as error:
        raise SpeedSmokeV2Error("speed-smoke V2 V1 runtime identity reconstruction failed") from error
    if identity.payload() != dict(value):
        raise SpeedSmokeV2Error("speed-smoke V2 V1 runtime identity canonical drift")
    return identity


@dataclass(frozen=True)
class SpeedSmokeV2Identity:
    """Current V2 closure plus exact V1 failure and V1 runtime evidence."""

    closure: Mapping[str, object]
    v1_failure_binding: Mapping[str, object]
    v1_runtime_identity: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        closure = plan.validate_implementation_closure(self.closure)
        failure = validate_v1_failure_binding(self.v1_failure_binding)
        runtime = _typed_v1_identity(self.v1_runtime_identity).payload()
        if runtime != failure["historical_v1_identity"]:
            raise SpeedSmokeV2Error("speed-smoke V2 V1 failed/runtime identity substitution drift")
        return {
            "schema": "precision_aware_cdmd_speed_smoke_v2_identity_v1",
            "cell": plan.CELL,
            "phase": plan.PHASE,
            "closure": closure,
            "v1_failed_predecessor_binding": failure,
            "v1_failed_predecessor_binding_sha256": failure["binding_sha256"],
            "v1_runtime_identity": runtime,
            "v1_runtime_identity_sha256": _digest(runtime),
            "selected_device_profile": dict(plan.GPU0_PROFILE),
            "canonical_source_environment": plan.canonical_source_environment_payload(),
            "smoke_execution_order": [list(item) for item in plan.SMOKE_EXECUTION_ORDER],
            "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
            "score_root_relative": plan.SCORE_ROOT_RELATIVE,
            "v1_result_immutable_not_retryable": True,
            "target_updates_forbidden": True,
            "non_governing_engineering_smoke": True,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.payload())

    def v1_identity(self) -> v1_score.SpeedSmokeIdentity:
        return _typed_v1_identity(self.v1_runtime_identity)


def _identity_payload(identity: SpeedSmokeV2Identity) -> dict[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity):
        raise SpeedSmokeV2Error("speed-smoke V2 requires its exact typed identity")
    return identity.payload()


def build_reviewed_identity(root: Path) -> SpeedSmokeV2Identity:
    """Build from current V2 closure and immutable V1/V2 predecessor bytes only."""
    v1_identity = v1_score.build_reviewed_identity(Path(root))
    failure = validate_v1_failed_speed_smoke(Path(root))
    if failure.payload()["historical_v1_identity"] != v1_identity.payload():
        raise SpeedSmokeV2Error("speed-smoke V2 V1 historical/current identity drift")
    return SpeedSmokeV2Identity(
        closure=plan.implementation_closure(Path(root)).payload(),
        v1_failure_binding=failure.payload(), v1_runtime_identity=v1_identity.payload(),
    )


def _require_live_predecessors(root: Path, identity: SpeedSmokeV2Identity) -> tuple[V1FailureBinding, v1_score.CompletedV2Binding]:
    failure = validate_v1_failed_speed_smoke(Path(root))
    if failure.payload() != _identity_payload(identity)["v1_failed_predecessor_binding"]:
        raise SpeedSmokeV2Error("speed-smoke V2 held V1 failure predecessor drift")
    completed = v1_score.validate_completed_precision_v2(Path(root))
    if completed.payload() != identity.v1_identity().payload()["completed_precision_v2_binding"]:
        raise SpeedSmokeV2Error("speed-smoke V2 held accepted Precision predecessor drift")
    return failure, completed


def _assert_current_closure(root: Path, identity: SpeedSmokeV2Identity) -> None:
    if plan.implementation_closure(Path(root)).payload() != _identity_payload(identity)["closure"]:
        raise SpeedSmokeV2Error("speed-smoke V2 current implementation closure drift")


def _validate_mutating_launch_environment(environ: Mapping[str, str] | None) -> dict[str, object]:
    checked = plan.validate_selected_launch_environment(environ)
    if environ is not None:
        names = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", *plan.CANONICAL_SOURCE_ROOTS)
        if any(os.environ.get(name) != environ.get(name) for name in names):
            raise SpeedSmokeV2Error("speed-smoke V2 supplied/actual launch environment drift")
    plan.validate_selected_launch_environment(os.environ)
    return checked


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SpeedSmokeV2Error(f"speed-smoke V2 {label} must be finite float")
    return float(value)


def diagnostic_numeric_comparison(value: object) -> dict[str, object]:
    """Validate and classify a raw V1 physical comparison without hiding it."""
    required = {
        "schema", "baseline_physical_eval_batch_size", "optimized_physical_eval_batch_size",
        "baseline_prediction_sha256", "optimized_prediction_sha256", "max_abs_prediction_error",
        "prediction_max_abs_tolerance", "baseline_governing_r2", "optimized_governing_r2", "r2_abs_error",
        "r2_absolute_tolerance", "baseline_transition_records_sha256", "optimized_transition_records_sha256",
        "transition_sequence_exact", "baseline_wall_seconds", "optimized_wall_seconds", "speedup_ratio",
        "baseline_full_system_forward_count", "baseline_group_forward_count",
        "optimized_full_system_forward_count", "optimized_group_forward_count",
        "smoke_min_speedup_ratio_exclusive", "recommended_speedup_ratio",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "precision_aware_cdmd_speed_smoke_numerical_comparison_v1":
        raise SpeedSmokeV2Error("speed-smoke V2 raw numerical-comparison schema drift")
    raw = dict(value)
    max_abs = _finite_float(raw.get("max_abs_prediction_error"), "maximum prediction error")
    r2_error = _finite_float(raw.get("r2_abs_error"), "R2 error")
    baseline_r2 = _finite_float(raw.get("baseline_governing_r2"), "baseline R2")
    optimized_r2 = _finite_float(raw.get("optimized_governing_r2"), "optimized R2")
    baseline_wall = _finite_float(raw.get("baseline_wall_seconds"), "baseline wall")
    optimized_wall = _finite_float(raw.get("optimized_wall_seconds"), "optimized wall")
    speedup = _finite_float(raw.get("speedup_ratio"), "speedup")
    if (
        raw.get("baseline_physical_eval_batch_size") != plan.BASELINE_PHYSICAL_EVAL_BATCH_SIZE
        or raw.get("optimized_physical_eval_batch_size") not in plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES
        or raw.get("prediction_max_abs_tolerance") != plan.V1_MAX_ABS_PREDICTION_ERROR
        or raw.get("r2_absolute_tolerance") != plan.V1_R2_ABS_ERROR
        or raw.get("smoke_min_speedup_ratio_exclusive") != plan.V1_SPEEDUP_EXCLUSIVE
        or raw.get("recommended_speedup_ratio") != plan.RECOMMENDED_SPEEDUP_RATIO
        or raw.get("transition_sequence_exact") is not True
        or _sha(raw.get("baseline_prediction_sha256"), "baseline prediction") != raw.get("baseline_prediction_sha256")
        or _sha(raw.get("optimized_prediction_sha256"), "optimized prediction") != raw.get("optimized_prediction_sha256")
        or _sha(raw.get("baseline_transition_records_sha256"), "baseline transition") != raw.get("baseline_transition_records_sha256")
        or _sha(raw.get("optimized_transition_records_sha256"), "optimized transition") != raw.get("optimized_transition_records_sha256")
        or max_abs < 0.0 or r2_error < 0.0 or baseline_wall <= 0.0 or optimized_wall <= 0.0
        or r2_error != abs(baseline_r2 - optimized_r2)
        or speedup != baseline_wall / optimized_wall
        or any(type(raw.get(name)) is not int or raw[name] < 0 for name in (
            "baseline_full_system_forward_count", "baseline_group_forward_count",
            "optimized_full_system_forward_count", "optimized_group_forward_count",
        ))
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 raw numerical-comparison literal drift")
    v1 = {
        "max_abs_prediction_error_passed": max_abs <= plan.V1_MAX_ABS_PREDICTION_ERROR,
        "r2_absolute_error_passed": r2_error <= plan.V1_R2_ABS_ERROR,
        "speedup_strictly_gt_one_passed": speedup > plan.V1_SPEEDUP_EXCLUSIVE,
    }
    v2 = {
        "max_abs_prediction_error_passed": max_abs <= plan.V2_MAX_ABS_PREDICTION_ERROR,
        "r2_absolute_error_passed": r2_error <= plan.V2_R2_ABS_ERROR,
        "speedup_required": False,
        "accepted": max_abs <= plan.V2_MAX_ABS_PREDICTION_ERROR and r2_error <= plan.V2_R2_ABS_ERROR,
    }
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_diagnostic_numeric_comparison_v1",
        "raw_comparison": raw,
        "v1_thresholds": {
            "max_abs_prediction_error": plan.V1_MAX_ABS_PREDICTION_ERROR,
            "r2_absolute_error": plan.V1_R2_ABS_ERROR,
            "speedup_strictly_gt": plan.V1_SPEEDUP_EXCLUSIVE,
        },
        "v1_predicates": v1,
        "v2_thresholds": {
            "max_abs_prediction_error": plan.V2_MAX_ABS_PREDICTION_ERROR,
            "r2_absolute_error": plan.V2_R2_ABS_ERROR,
            "speedup_is_descriptive": True,
        },
        "v2_predicates": v2,
        "recommended_speedup_ratio": plan.RECOMMENDED_SPEEDUP_RATIO,
        "recommendation_reached": speedup >= plan.RECOMMENDED_SPEEDUP_RATIO,
    }


def validate_diagnostic_numeric_comparison(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or value.get("schema") != "precision_aware_cdmd_speed_smoke_v2_diagnostic_numeric_comparison_v1":
        raise SpeedSmokeV2Error("speed-smoke V2 diagnostic numerical-comparison schema drift")
    raw = value.get("raw_comparison")
    rebuilt = diagnostic_numeric_comparison(raw)
    if dict(value) != rebuilt:
        raise SpeedSmokeV2Error("speed-smoke V2 diagnostic numerical-comparison canonical drift")
    return rebuilt


def _record_for_witness(records: Sequence[Mapping[str, object]], witness: object) -> dict[str, object]:
    surface, session = getattr(witness, "surface", None), getattr(witness, "session", None)
    matches = [
        dict(item) for item in records
        if isinstance(item, Mapping) and item.get("surface") == surface and item.get("session") == session
    ]
    if len(matches) != 1:
        raise SpeedSmokeV2Error("speed-smoke V2 selected input-record topology drift")
    return matches[0]


def input_payload(authority: Any, identity: SpeedSmokeV2Identity) -> dict[str, object]:
    """Wrap V1's full exact input reconstruction under V2 provenance."""
    inherited = v1_score.input_payload(authority, identity.v1_identity())
    records = inherited.get("records") if isinstance(inherited, Mapping) else None
    if not isinstance(records, list):
        raise SpeedSmokeV2Error("speed-smoke V2 inherited input records are absent")
    selected = {
        f"m{item.budget}:{item.surface}:{item.session}": _digest(_record_for_witness(records, item))
        for item in plan.SMOKE_ROWS
    }
    expected = {
        f"m{item.budget}:{item.surface}:{item.session}": item.input_record_sha256
        for item in plan.SMOKE_ROWS
    }
    if selected != expected:
        raise SpeedSmokeV2Error("speed-smoke V2 selected input-record/V1 witness drift")
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_input_authority_v1",
        "identity_sha256": identity.sha256,
        "v1_input_payload": dict(inherited),
        "v1_input_payload_sha256": _digest(inherited),
        "records": records,
        "selected_record_sha256s": selected,
        "same_materialized_input_as_v1_and_accepted_precision_v2": True,
        "target_labels_metric_only": True,
        "cache_read_or_write": False,
    }


def validate_input_payload(
    value: object, *, identity: SpeedSmokeV2Identity, evaluation_authority: shared_score.FixedEvaluationAuthority,
) -> dict[str, object]:
    required = {
        "schema", "identity_sha256", "v1_input_payload", "v1_input_payload_sha256", "records",
        "selected_record_sha256s", "same_materialized_input_as_v1_and_accepted_precision_v2",
        "target_labels_metric_only", "cache_read_or_write",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_v2_input_authority_v1"
        or value.get("identity_sha256") != identity.sha256
        or not isinstance(value.get("v1_input_payload"), Mapping)
        or value.get("v1_input_payload_sha256") != _digest(value["v1_input_payload"])
        or value.get("same_materialized_input_as_v1_and_accepted_precision_v2") is not True
        or value.get("target_labels_metric_only") is not True
        or value.get("cache_read_or_write") is not False
        or not isinstance(value.get("records"), list) or not isinstance(value.get("selected_record_sha256s"), Mapping)
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 input authority schema/boundary drift")
    try:
        v1_checked = v1_score.validate_input_payload(
            value["v1_input_payload"], identity=identity.v1_identity(), evaluation_authority=evaluation_authority,
        )
    except v1_score.SpeedSmokeError as error:
        raise SpeedSmokeV2Error("speed-smoke V2 inherited input authority drift") from error
    if value.get("records") != v1_checked["records"]:
        raise SpeedSmokeV2Error("speed-smoke V2 inherited input records substitution drift")
    expected = {
        f"m{item.budget}:{item.surface}:{item.session}": item.input_record_sha256
        for item in plan.SMOKE_ROWS
    }
    if value.get("selected_record_sha256s") != expected:
        raise SpeedSmokeV2Error("speed-smoke V2 selected input record digest drift")
    return dict(value)


def _baseline_row(row: Mapping[str, object], witness: object) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        return v1_score._validate_baseline_row(row, witness)
    except v1_score.SpeedSmokeError as error:
        raise SpeedSmokeV2Error("speed-smoke V2 exact eager baseline anchor drift") from error


def _optimized_row(
    row: Mapping[str, object], *, witness: object, baseline_transitions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    checked = dict(row)
    r2 = _finite_float(checked.get("governing_r2"), "optimized R2")
    if (
        checked.get("session") != getattr(witness, "session")
        or checked.get("budget") != getattr(witness, "budget")
        or checked.get("input_record_sha256") != getattr(witness, "input_record_sha256")
        or checked.get("n_windows") != getattr(witness, "n_windows")
        or checked.get("model_state_before_sha256") != getattr(witness, "model_state_sha256")
        or checked.get("model_state_after_sha256") != getattr(witness, "model_state_sha256")
        or checked.get("sealed_model_load_proof_sha256") != getattr(witness, "sealed_model_load_proof_sha256")
        or _sha(checked.get("prediction_sha256"), "optimized prediction") != checked.get("prediction_sha256")
        or not math.isfinite(r2)
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 optimized row static boundary drift")
    transitions = checked.get("transition_records")
    if not isinstance(transitions, list) or any(not isinstance(item, Mapping) for item in transitions):
        raise SpeedSmokeV2Error("speed-smoke V2 optimized transition row type drift")
    result = [dict(item) for item in transitions]
    if result != [dict(item) for item in baseline_transitions]:
        raise SpeedSmokeV2Error("speed-smoke V2 optimized transition chronology drift")
    return checked, result


def _validate_speed_evidence(value: object) -> dict[str, object]:
    try:
        return v1_score._validate_speed_evidence(value)
    except v1_score.SpeedSmokeError as error:
        raise SpeedSmokeV2Error("speed-smoke V2 O1/O2 evidence drift") from error


def _validate_resources(value: object, identity: SpeedSmokeV2Identity) -> dict[str, object]:
    try:
        return v1_score._validate_resources(value, identity=identity.v1_identity())
    except v1_score.SpeedSmokeError as error:
        raise SpeedSmokeV2Error("speed-smoke V2 resource evidence drift") from error


def build_diagnostic_cell(
    *, identity: SpeedSmokeV2Identity, input_authority_sha256: str,
    baseline_session_row: Mapping[str, object], optimized_session_row: Mapping[str, object],
    raw_comparison: Mapping[str, object], speed_evidence: Mapping[str, object], resources: Mapping[str, object],
    witness: object,
) -> dict[str, object]:
    """Persist the comparison first, then apply only V2's predeclared limits."""
    baseline, baseline_transitions = _baseline_row(baseline_session_row, witness)
    optimized, optimized_transitions = _optimized_row(
        optimized_session_row, witness=witness, baseline_transitions=baseline_transitions,
    )
    diagnostic = diagnostic_numeric_comparison(raw_comparison)
    raw = diagnostic["raw_comparison"]
    if (
        raw.get("baseline_prediction_sha256") != baseline.get("prediction_sha256")
        or raw.get("optimized_prediction_sha256") != optimized.get("prediction_sha256")
        or raw.get("baseline_governing_r2") != baseline.get("governing_r2")
        or raw.get("optimized_governing_r2") != optimized.get("governing_r2")
        or raw.get("baseline_transition_records_sha256") != _digest([dict(item) for item in baseline_transitions])
        or raw.get("optimized_transition_records_sha256") != _digest([dict(item) for item in optimized_transitions])
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 comparison/row binding drift")
    if diagnostic["v2_predicates"]["accepted"] is not True:
        raise SpeedSmokeV2Error("speed-smoke V2 numerical-equivalence limits exceeded")
    evidence = _validate_speed_evidence(speed_evidence)
    checked_resources = _validate_resources(resources, identity)
    engine = evidence["stage0_o1_o2_engine_evidence"]
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_cell_v1",
        "budget": getattr(witness, "budget"),
        "surface": getattr(witness, "surface"),
        "session": getattr(witness, "session"),
        "historical_precision_v2_row_sha256": getattr(witness, "canonical_row_sha256"),
        "input_authority_sha256": _sha(input_authority_sha256, "cell input authority"),
        "historical_input_record_sha256": getattr(witness, "input_record_sha256"),
        "runtime_input_record_sha256": optimized["input_record_sha256"],
        "baseline_prediction_sha256": baseline["prediction_sha256"],
        "optimized_prediction_sha256": optimized["prediction_sha256"],
        "baseline_governing_r2": baseline["governing_r2"],
        "optimized_governing_r2": optimized["governing_r2"],
        "baseline_row_sha256": _digest(baseline),
        "optimized_row_sha256": _digest(optimized),
        "transition_sequence_exact": True,
        "transition_records_sha256": _digest([dict(item) for item in baseline_transitions]),
        "carrier_transition_committed_count": getattr(witness, "carrier_commits"),
        "transition_count": getattr(witness, "transition_count"),
        "speed_evidence": evidence,
        "forward_counts": {
            "optimized_actual_model_forward_count": engine["actual_model_forward_count"],
            "optimized_identity_encoder_forward_count": engine["identity_encoder_forward_count"],
            "optimized_actual_model_forward_count_by_path": engine["actual_model_forward_count_by_path"],
            "optimized_logical_chunk_count_by_path": engine["logical_chunk_count_by_path"],
            "baseline_full_system_forward_count": baseline["full_system_forward_count"],
            "baseline_group_forward_count": baseline["group_forward_count"],
            "optimized_full_system_forward_count": optimized["full_system_forward_count"],
            "optimized_group_forward_count": optimized["group_forward_count"],
        },
        "resources": checked_resources,
        "diagnostic_numeric_comparison": diagnostic,
        "v2_numerical_equivalence_accepted": True,
        "speedup_is_descriptive_not_terminal_gate": True,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "m10_m30_rerun": False,
        "sealed_comparator_rerun": False,
    }


def validate_diagnostic_cell(
    value: object, *, identity: SpeedSmokeV2Identity, input_payload_value: Mapping[str, object],
    input_authority_sha256: str, witness: object,
) -> dict[str, object]:
    required = {
        "schema", "budget", "surface", "session", "historical_precision_v2_row_sha256", "input_authority_sha256",
        "historical_input_record_sha256", "runtime_input_record_sha256", "baseline_prediction_sha256",
        "optimized_prediction_sha256", "baseline_governing_r2", "optimized_governing_r2", "baseline_row_sha256",
        "optimized_row_sha256", "transition_sequence_exact", "transition_records_sha256",
        "carrier_transition_committed_count", "transition_count", "speed_evidence", "forward_counts", "resources",
        "diagnostic_numeric_comparison", "v2_numerical_equivalence_accepted",
        "speedup_is_descriptive_not_terminal_gate", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls", "m10_m30_rerun", "sealed_comparator_rerun",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_v2_cell_v1"
        or value.get("budget") != getattr(witness, "budget")
        or value.get("surface") != getattr(witness, "surface")
        or value.get("session") != getattr(witness, "session")
        or value.get("historical_precision_v2_row_sha256") != getattr(witness, "canonical_row_sha256")
        or value.get("input_authority_sha256") != _sha(input_authority_sha256, "cell expected input authority")
        or value.get("historical_input_record_sha256") != getattr(witness, "input_record_sha256")
        or value.get("runtime_input_record_sha256") != getattr(witness, "input_record_sha256")
        or value.get("baseline_prediction_sha256") != getattr(witness, "prediction_sha256")
        or value.get("baseline_governing_r2") != getattr(witness, "governing_r2")
        or value.get("baseline_row_sha256") != getattr(witness, "canonical_row_sha256")
        or _sha(value.get("optimized_prediction_sha256"), "cell optimized prediction") != value.get("optimized_prediction_sha256")
        or _sha(value.get("optimized_row_sha256"), "cell optimized row") != value.get("optimized_row_sha256")
        or value.get("transition_sequence_exact") is not True
        or value.get("carrier_transition_committed_count") != getattr(witness, "carrier_commits")
        or value.get("transition_count") != getattr(witness, "transition_count")
        or value.get("v2_numerical_equivalence_accepted") is not True
        or value.get("speedup_is_descriptive_not_terminal_gate") is not True
        or any(value.get(key) != 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls"))
        or value.get("m10_m30_rerun") is not False or value.get("sealed_comparator_rerun") is not False
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 diagnostic cell boundary drift")
    diagnostic = validate_diagnostic_numeric_comparison(value.get("diagnostic_numeric_comparison"))
    raw = diagnostic["raw_comparison"]
    if (
        raw.get("baseline_prediction_sha256") != value.get("baseline_prediction_sha256")
        or raw.get("optimized_prediction_sha256") != value.get("optimized_prediction_sha256")
        or raw.get("baseline_governing_r2") != value.get("baseline_governing_r2")
        or raw.get("optimized_governing_r2") != value.get("optimized_governing_r2")
        or raw.get("baseline_transition_records_sha256") != value.get("transition_records_sha256")
        or raw.get("optimized_transition_records_sha256") != value.get("transition_records_sha256")
        or diagnostic["v2_predicates"]["accepted"] is not True
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 cell diagnostic binding drift")
    evidence = _validate_speed_evidence(value.get("speed_evidence"))
    engine = evidence["stage0_o1_o2_engine_evidence"]
    expected_forward = {
        "optimized_actual_model_forward_count": engine["actual_model_forward_count"],
        "optimized_identity_encoder_forward_count": engine["identity_encoder_forward_count"],
        "optimized_actual_model_forward_count_by_path": engine["actual_model_forward_count_by_path"],
        "optimized_logical_chunk_count_by_path": engine["logical_chunk_count_by_path"],
        "baseline_full_system_forward_count": raw["baseline_full_system_forward_count"],
        "baseline_group_forward_count": raw["baseline_group_forward_count"],
        "optimized_full_system_forward_count": raw["optimized_full_system_forward_count"],
        "optimized_group_forward_count": raw["optimized_group_forward_count"],
    }
    if value.get("forward_counts") != expected_forward:
        raise SpeedSmokeV2Error("speed-smoke V2 forward-count receipt drift")
    _validate_resources(value.get("resources"), identity)
    records = input_payload_value.get("records")
    if not isinstance(records, list) or _digest(_record_for_witness(records, witness)) != getattr(witness, "input_record_sha256"):
        raise SpeedSmokeV2Error("speed-smoke V2 runtime input record drift")
    return dict(value)


def build_target_free_preflight(
    *, root: Path, identity: SpeedSmokeV2Identity, predecessor: V1FailureBinding,
    fixed_authority: shared_score.FixedEvaluationAuthority | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    environment = plan.validate_selected_launch_environment(environ)
    if predecessor.payload() != _identity_payload(identity)["v1_failed_predecessor_binding"]:
        raise SpeedSmokeV2Error("speed-smoke V2 preflight V1 failure/identity drift")
    observed, _completed = _require_live_predecessors(Path(root), identity)
    if observed.payload() != predecessor.payload():
        raise SpeedSmokeV2Error("speed-smoke V2 preflight held V1 failure drift")
    fixed = shared_score.derive_fixed_evaluation_authority(Path(root)) if fixed_authority is None else fixed_authority
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_target_free_preflight_v1",
        "status": "PREFLIGHT_ACCEPTED",
        "identity": _identity_payload(identity),
        "source_gate": predecessor.payload(),
        "v1_failed_predecessor": predecessor.payload(),
        "evaluation_authority": fixed.payload(),
        "metric": dict(v1_score.base_plan_metric()),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "canonical_launch_environment": environment,
        "target_free": True,
        "target_paths_resolved": False,
        "model_or_checkpoint_opened": False,
        "cuda_initialized": False,
        "boundaries": {
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "normalizer_refit": False, "m10_m30_rerun": False, "sealed_comparator_rerun": False,
        },
        "receipt_codec": "speed_smoke_v2__baseline_exact_diagnostic_numeric_comparison",
    }


def validate_target_free_preflight(value: object, identity: SpeedSmokeV2Identity) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "source_gate", "v1_failed_predecessor", "evaluation_authority", "metric",
        "authority_root_relative", "score_root_relative", "canonical_launch_environment", "target_free",
        "target_paths_resolved", "model_or_checkpoint_opened", "cuda_initialized", "boundaries", "receipt_codec",
    }
    expected_environment = plan.validate_selected_launch_environment(
        {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", **plan.CANONICAL_SOURCE_ROOTS},
    )
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_v2_target_free_preflight_v1"
        or value.get("status") != "PREFLIGHT_ACCEPTED"
        or value.get("identity") != _identity_payload(identity)
        or value.get("source_gate") != _identity_payload(identity)["v1_failed_predecessor_binding"]
        or value.get("v1_failed_predecessor") != _identity_payload(identity)["v1_failed_predecessor_binding"]
        or value.get("metric") != dict(v1_score.base_plan_metric())
        or value.get("authority_root_relative") != plan.AUTHORITY_ROOT_RELATIVE
        or value.get("score_root_relative") != plan.SCORE_ROOT_RELATIVE
        or value.get("canonical_launch_environment") != expected_environment
        or value.get("target_free") is not True or value.get("target_paths_resolved") is not False
        or value.get("model_or_checkpoint_opened") is not False or value.get("cuda_initialized") is not False
        or value.get("boundaries") != {
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "normalizer_refit": False, "m10_m30_rerun": False, "sealed_comparator_rerun": False,
        }
        or value.get("receipt_codec") != "speed_smoke_v2__baseline_exact_diagnostic_numeric_comparison"
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 preflight schema/identity/boundary drift")
    try:
        shared_score._fixed_authority_from_payload(value.get("evaluation_authority"))
    except shared_score.ScoreError as error:
        raise SpeedSmokeV2Error("speed-smoke V2 preflight fixed authority drift") from error
    return dict(value)


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    checked = dict(preflight)
    source_gate = checked.get("source_gate")
    if not isinstance(source_gate, Mapping):
        raise SpeedSmokeV2Error("speed-smoke V2 authorization predecessor absent")
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_root_authorization_v1",
        "status": "ROOT_AUTHORIZED",
        "official_preflight_sha256": _sha(official_preflight_sha256, "preflight SHA"),
        "identity_sha256": _digest(checked["identity"]),
        "v1_failed_predecessor_binding_sha256": source_gate.get("binding_sha256"),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "target_free_preflight_required": True,
        "explicit_execution_capability_required": True,
        "gpu0_environment_required_before_reserve_publish_capability_attempt_and_final": True,
    }


def validate_root_authorization(
    value: object, *, official_preflight_sha256: str, preflight: Mapping[str, object], identity: SpeedSmokeV2Identity,
) -> dict[str, object]:
    checked = validate_target_free_preflight(preflight, identity)
    expected = build_root_authorization(official_preflight_sha256=official_preflight_sha256, preflight=checked)
    if dict(value) != expected or expected["identity_sha256"] != identity.sha256:
        raise SpeedSmokeV2Error("speed-smoke V2 root authorization canonical identity drift")
    return expected


def reserve_authority_artifact(
    root: Path, capability: object, *, identity: SpeedSmokeV2Identity, environ: Mapping[str, str] | None = None,
) -> shared_score.ArtifactRoot:
    _validate_mutating_launch_environment(environ)
    shared_score._require_root_publication_capability(capability)
    _require_live_predecessors(Path(root), identity)
    _assert_current_closure(Path(root), identity)
    plan.assert_fresh_prospective_root(Path(root), plan.AUTHORITY_ROOT_RELATIVE)
    parent = Path(root).absolute() / Path(plan.AUTHORITY_ROOT_RELATIVE).parent
    return shared_score._equal_session_module(Path(root)).reserve_artifact_root(
        parent, Path(plan.AUTHORITY_ROOT_RELATIVE).name, topology=AUTHORITY_TOPOLOGY,
    )


def publish_target_free_preflight(
    root: Path, artifact: shared_score.ArtifactRoot, capability: object, payload: Mapping[str, object], *,
    identity: SpeedSmokeV2Identity, environ: Mapping[str, str] | None = None,
) -> str:
    _validate_mutating_launch_environment(environ)
    shared_score._require_root_publication_capability(capability)
    _require_live_predecessors(Path(root), identity)
    checked = validate_target_free_preflight(payload, identity)
    observed = shared_score.derive_fixed_evaluation_authority(Path(root)).payload()
    if checked["evaluation_authority"] != observed:
        raise SpeedSmokeV2Error("speed-smoke V2 fixed authority drift before preflight publication")
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization(
    root: Path, artifact: shared_score.ArtifactRoot, capability: object, payload: Mapping[str, object], *,
    identity: SpeedSmokeV2Identity, environ: Mapping[str, str] | None = None,
) -> str:
    _validate_mutating_launch_environment(environ)
    shared_score._require_root_publication_capability(capability)
    _require_live_predecessors(Path(root), identity)
    body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise SpeedSmokeV2Error("speed-smoke V2 durable preflight malformed") from error
    checked = validate_target_free_preflight(preflight, identity)
    observed = shared_score.derive_fixed_evaluation_authority(Path(root)).payload()
    if checked["evaluation_authority"] != observed:
        raise SpeedSmokeV2Error("speed-smoke V2 fixed authority drift before authorization publication")
    authorization = validate_root_authorization(
        payload, official_preflight_sha256=hashlib.sha256(body).hexdigest(), preflight=checked, identity=identity,
    )
    return artifact.publish_json("root_authorization.json", authorization)


def _read_durable_authority_pair(
    root: Path, *, identity: SpeedSmokeV2Identity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    directory = Path(root).absolute() / plan.AUTHORITY_ROOT_RELATIVE
    named = _directory_identity(directory, "durable authority")
    descriptor = -1
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | _no_follow_flag())
        held = os.fstat(descriptor)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != named:
            raise SpeedSmokeV2Error("speed-smoke V2 durable authority identity drift before read")
        expected = set(AUTHORITY_TOPOLOGY) | {f"{name}.sha256" for name in AUTHORITY_TOPOLOGY}
        if set(os.listdir(descriptor)) != expected:
            raise SpeedSmokeV2Error("speed-smoke V2 durable authority topology drift")
        pre_body, preflight, pre_sha = shared_score._read_unbound_0444_pair(descriptor, "official_preflight.json")
        auth_body, authorization, auth_sha = shared_score._read_unbound_0444_pair(descriptor, "root_authorization.json")
        if _directory_identity(directory, "durable authority") != named:
            raise SpeedSmokeV2Error("speed-smoke V2 durable authority identity drift during read")
        validate_target_free_preflight(preflight, identity)
        validate_root_authorization(authorization, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity)
        if hashlib.sha256(pre_body).hexdigest() != pre_sha or hashlib.sha256(auth_body).hexdigest() != auth_sha:
            raise SpeedSmokeV2Error("speed-smoke V2 durable authority body digest drift")
        return dict(preflight), dict(authorization), pre_sha, auth_sha
    except (OSError, shared_score.ScoreError) as error:
        raise SpeedSmokeV2Error("speed-smoke V2 durable authority descriptor read failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_durable_authority(
    root: Path, *, identity: SpeedSmokeV2Identity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    return _read_durable_authority_pair(Path(root), identity=identity)


def issue_durable_execution_capability(
    root: Path, *, identity: SpeedSmokeV2Identity, root_capability: object,
    environ: Mapping[str, str] | None = None,
) -> shared_score.ExecutionCapability:
    _validate_mutating_launch_environment(environ)
    shared_score._require_root_publication_capability(root_capability)
    _require_live_predecessors(Path(root), identity)
    _assert_current_closure(Path(root), identity)
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(Path(root), identity=identity)
    observed = shared_score.derive_fixed_evaluation_authority(Path(root)).payload()
    if preflight["evaluation_authority"] != observed:
        raise SpeedSmokeV2Error("speed-smoke V2 issuer fixed authority drift")
    plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)
    return shared_score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=root_capability,
    )


def reserve_score_artifact(
    root: Path, *, identity: SpeedSmokeV2Identity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> shared_score.ArtifactRoot:
    shared_score.require_execution_capability(capability, identity)
    _validate_mutating_launch_environment(environ)
    _require_live_predecessors(Path(root), identity)
    _assert_current_closure(Path(root), identity)
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(Path(root), identity=identity)
    approved = shared_score.require_execution_capability(capability, identity)
    if approved.official_preflight_sha256 != pre_sha or approved.root_authorization_sha256 != auth_sha:
        raise SpeedSmokeV2Error("speed-smoke V2 capability/durable authority drift")
    observed = shared_score.derive_fixed_evaluation_authority(Path(root)).payload()
    if preflight["evaluation_authority"] != observed:
        raise SpeedSmokeV2Error("speed-smoke V2 score-reservation fixed authority drift")
    plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)
    parent = Path(root).absolute() / Path(plan.SCORE_ROOT_RELATIVE).parent
    return shared_score._equal_session_module(Path(root)).reserve_artifact_root(
        parent, Path(plan.SCORE_ROOT_RELATIVE).name, topology=SCORE_TOPOLOGY,
    )


def _summary(
    cells: Sequence[Mapping[str, object]], *, budget: int, identity: SpeedSmokeV2Identity,
    input_payload_value: Mapping[str, object], input_authority_sha256: str,
) -> dict[str, object]:
    expected = [item for item in plan.SMOKE_ROWS if item.budget == budget]
    if len(expected) != 1 or len(cells) != 1:
        raise SpeedSmokeV2Error("speed-smoke V2 exact one-cell summary topology drift")
    cell = validate_diagnostic_cell(
        cells[0], identity=identity, input_payload_value=input_payload_value,
        input_authority_sha256=input_authority_sha256, witness=expected[0],
    )
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_budget_summary_v1",
        "budget": budget,
        "cells": [cell],
        "baseline_historical_precision_v2_parity": True,
        "v2_numerical_equivalence_accepted": True,
        "speedup_descriptive_only": True,
        "non_governing_engineering_smoke": True,
    }


def _budget_gate(
    summary: Mapping[str, object], *, budget: int, identity: SpeedSmokeV2Identity,
    input_payload_value: Mapping[str, object], input_authority_sha256: str,
) -> dict[str, object]:
    rebuilt = _summary(
        summary.get("cells", ()) if isinstance(summary, Mapping) else (), budget=budget, identity=identity,
        input_payload_value=input_payload_value, input_authority_sha256=input_authority_sha256,
    )
    if dict(summary) != rebuilt:
        raise SpeedSmokeV2Error("speed-smoke V2 summary reconstruction drift")
    return {
        "budget": budget,
        "baseline_exact_prediction_sha256_parity": True,
        "transition_sequence_exact": True,
        "v2_numerical_equivalence_accepted": True,
        "speedup_descriptive_only": True,
        "completed": True,
        "formal_gate": False,
    }


def _score_payload(
    identity: SpeedSmokeV2Identity, input_sha: str, summaries: Mapping[str, object],
    gates: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    expected_budgets = {str(item.budget) for item in plan.SMOKE_ROWS}
    if set(summaries) != expected_budgets or set(gates) != {item.budget for item in plan.SMOKE_ROWS}:
        raise SpeedSmokeV2Error("speed-smoke V2 score budget topology drift")
    payload = _identity_payload(identity)
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_score_v1",
        "identity": payload,
        "input_authority_sha256": _sha(input_sha, "score input authority"),
        "budget_summaries": {str(item.budget): dict(summaries[str(item.budget)]) for item in plan.SMOKE_ROWS},
        "budget_gates": {str(item.budget): dict(gates[item.budget]) for item in plan.SMOKE_ROWS},
        "cell_execution_order": [
            {"budget": budget, "surface": surface, "session": session}
            for budget, surface, session in plan.SMOKE_EXECUTION_ORDER
        ],
        "v1_failed_predecessor_binding_sha256": payload["v1_failed_predecessor_binding_sha256"],
        "m10_m30_rerun": False,
        "sealed_comparator_rerun": False,
        "target_optimizer_backward_update": 0,
        "formal_verdict_emitted": False,
    }


def validate_score_payload(
    value: object, *, identity: SpeedSmokeV2Identity, input_payload_value: Mapping[str, object],
    input_authority_sha256: str, evaluation_authority: shared_score.FixedEvaluationAuthority,
) -> dict[str, object]:
    required = {
        "schema", "identity", "input_authority_sha256", "budget_summaries", "budget_gates", "cell_execution_order",
        "v1_failed_predecessor_binding_sha256", "m10_m30_rerun", "sealed_comparator_rerun",
        "target_optimizer_backward_update", "formal_verdict_emitted",
    }
    expected_input = _sha(input_authority_sha256, "expected score input authority")
    if _digest(input_payload_value) != expected_input:
        raise SpeedSmokeV2Error("speed-smoke V2 score input payload/SHA drift")
    validate_input_payload(input_payload_value, identity=identity, evaluation_authority=evaluation_authority)
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_v2_score_v1"
        or value.get("identity") != _identity_payload(identity)
        or value.get("input_authority_sha256") != expected_input
        or value.get("cell_execution_order") != [
            {"budget": budget, "surface": surface, "session": session}
            for budget, surface, session in plan.SMOKE_EXECUTION_ORDER
        ]
        or value.get("v1_failed_predecessor_binding_sha256") != _identity_payload(identity)["v1_failed_predecessor_binding_sha256"]
        or value.get("m10_m30_rerun") is not False or value.get("sealed_comparator_rerun") is not False
        or value.get("target_optimizer_backward_update") != 0 or value.get("formal_verdict_emitted") is not False
        or not isinstance(value.get("budget_summaries"), Mapping) or not isinstance(value.get("budget_gates"), Mapping)
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 score schema/boundary drift")
    summaries = value["budget_summaries"]
    gates = value["budget_gates"]
    if set(summaries) != {str(item.budget) for item in plan.SMOKE_ROWS} or set(gates) != {str(item.budget) for item in plan.SMOKE_ROWS}:
        raise SpeedSmokeV2Error("speed-smoke V2 score summary/gate topology drift")
    for witness in plan.SMOKE_ROWS:
        raw_summary = summaries[str(witness.budget)]
        if not isinstance(raw_summary, Mapping):
            raise SpeedSmokeV2Error("speed-smoke V2 summary type drift")
        rebuilt = _summary(
            raw_summary.get("cells", ()), budget=witness.budget, identity=identity,
            input_payload_value=input_payload_value, input_authority_sha256=expected_input,
        )
        if dict(raw_summary) != rebuilt:
            raise SpeedSmokeV2Error("speed-smoke V2 score summary canonical drift")
        expected_gate = _budget_gate(
            rebuilt, budget=witness.budget, identity=identity,
            input_payload_value=input_payload_value, input_authority_sha256=expected_input,
        )
        if gates.get(str(witness.budget)) != expected_gate:
            raise SpeedSmokeV2Error("speed-smoke V2 score gate canonical drift")
    return dict(value)


def terminal_verdict(gates: Mapping[int, Mapping[str, object]]) -> str:
    if (
        set(gates) != {item.budget for item in plan.SMOKE_ROWS}
        or any(item.get("completed") is not True for item in gates.values())
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 terminal exact one-cell gate drift")
    return "SPEED_SMOKE_DIAGNOSTIC_NUMERICAL_EQUIVALENCE_COMPLETE_NON_GOVERNING"


def _terminal_payload(
    identity: SpeedSmokeV2Identity, attempt_sha: str, input_sha: str, score_sha: str,
    gates: Mapping[int, Mapping[str, object]], verdict: str,
) -> dict[str, object]:
    if verdict != "SPEED_SMOKE_DIAGNOSTIC_NUMERICAL_EQUIVALENCE_COMPLETE_NON_GOVERNING":
        raise SpeedSmokeV2Error("speed-smoke V2 terminal verdict drift")
    closure_sha = _identity_payload(identity)["closure"]["closure_sha256"]
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_terminal_v1",
        "status": "TERMINAL",
        "verdict": verdict,
        "identity": _identity_payload(identity),
        "attempt_sha256": _sha(attempt_sha, "terminal attempt"),
        "input_authority_sha256": _sha(input_sha, "terminal input"),
        "score_sha256": _sha(score_sha, "terminal score"),
        "budget_gates": {str(item.budget): dict(gates[item.budget]) for item in plan.SMOKE_ROWS},
        "launch_closure_sha256": closure_sha,
        "final_closure_sha256": closure_sha,
        "v1_failed_predecessor_binding_sha256": _identity_payload(identity)["v1_failed_predecessor_binding_sha256"],
        "target_optimizer_backward_update": 0,
        "informational_speedup_only": True,
    }


def validate_terminal_payload(
    value: object, *, identity: SpeedSmokeV2Identity, attempt_sha256: str, input_authority_sha256: str,
    score_payload_value: Mapping[str, object], score_sha256: str,
) -> dict[str, object]:
    gates = score_payload_value.get("budget_gates") if isinstance(score_payload_value, Mapping) else None
    if not isinstance(gates, Mapping):
        raise SpeedSmokeV2Error("speed-smoke V2 terminal score gates absent")
    typed = {
        int(key): dict(item)
        for key, item in gates.items()
        if isinstance(key, str) and key.isdigit() and isinstance(item, Mapping)
    }
    expected = _terminal_payload(
        identity, attempt_sha256, input_authority_sha256, score_sha256, typed, terminal_verdict(typed),
    )
    if dict(value) != expected:
        raise SpeedSmokeV2Error("speed-smoke V2 terminal canonical drift")
    return expected


def _attempt_payload(identity: SpeedSmokeV2Identity, pre_sha: str, auth_sha: str) -> dict[str, object]:
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "identity": _identity_payload(identity),
        "preflight_sha256": _sha(pre_sha, "attempt preflight"),
        "authorization_sha256": _sha(auth_sha, "attempt authorization"),
        "target_paths_resolved_or_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
        "attempt_precedes_backend_prepare": True,
    }


def _failure_progress(value: object | None) -> dict[str, object]:
    base = {
        "within_assets_opened": False, "external_assets_opened": False, "checkpoint_opened": False,
        "cuda_initialized": False, "full_system_forward_count": 0, "group_forward_count": 0,
        "diagnostic_numeric_comparison": None,
    }
    if value is None:
        return base
    if not isinstance(value, Mapping) or set(value) != set(base):
        raise SpeedSmokeV2Error("speed-smoke V2 failure progress schema drift")
    result = dict(value)
    if any(type(result[key]) is not bool for key in (
        "within_assets_opened", "external_assets_opened", "checkpoint_opened", "cuda_initialized",
    )) or any(type(result[key]) is not int or result[key] < 0 for key in (
        "full_system_forward_count", "group_forward_count",
    )):
        raise SpeedSmokeV2Error("speed-smoke V2 failure progress scalar drift")
    comparison = result["diagnostic_numeric_comparison"]
    if comparison is not None:
        result["diagnostic_numeric_comparison"] = validate_diagnostic_numeric_comparison(comparison)
    return result


def _failure_payload(
    identity: SpeedSmokeV2Identity, attempt_sha: str, input_sha: str | None, stage: str,
    error: BaseException, runtime_progress: object | None,
) -> dict[str, object]:
    progress = _failure_progress(runtime_progress)
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_failure_v1",
        "status": "FAILED",
        "identity": _identity_payload(identity),
        "attempt_sha256": _sha(attempt_sha, "failure attempt"),
        "input_authority_sha256": None if input_sha is None else _sha(input_sha, "failure input"),
        "stage": stage,
        "error_class": type(error).__name__,
        "error_sha256": hashlib.sha256(repr(error).encode("utf-8")).hexdigest(),
        "target_paths_resolved_or_opened": bool(progress["within_assets_opened"] or progress["external_assets_opened"]),
        **progress,
        "terminal_published": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "v1_failed_predecessor_binding_sha256": _identity_payload(identity)["v1_failed_predecessor_binding_sha256"],
    }


def validate_failure_payload(
    value: object, *, identity: SpeedSmokeV2Identity, attempt_sha256: str, input_authority_sha256: str | None,
) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "attempt_sha256", "input_authority_sha256", "stage", "error_class",
        "error_sha256", "target_paths_resolved_or_opened", "within_assets_opened", "external_assets_opened",
        "checkpoint_opened", "cuda_initialized", "full_system_forward_count", "group_forward_count",
        "diagnostic_numeric_comparison", "terminal_published", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls", "v1_failed_predecessor_binding_sha256",
    }
    stages = {"attempt", "prepare", "materialize_inputs", "budget_m4", "final_revalidate", "publish_terminal"}
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_speed_smoke_v2_failure_v1"
        or value.get("status") != "FAILED" or value.get("identity") != _identity_payload(identity)
        or value.get("attempt_sha256") != attempt_sha256 or value.get("input_authority_sha256") != input_authority_sha256
        or value.get("stage") not in stages or not isinstance(value.get("error_class"), str) or not value.get("error_class")
        or _sha(value.get("error_sha256"), "failure error") != value.get("error_sha256")
        or value.get("terminal_published") is not False
        or any(value.get(key) != 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls"))
        or value.get("v1_failed_predecessor_binding_sha256") != _identity_payload(identity)["v1_failed_predecessor_binding_sha256"]
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 failure schema/boundary drift")
    progress = _failure_progress({key: value.get(key) for key in (
        "within_assets_opened", "external_assets_opened", "checkpoint_opened", "cuda_initialized",
        "full_system_forward_count", "group_forward_count", "diagnostic_numeric_comparison",
    )})
    if value.get("target_paths_resolved_or_opened") is not bool(progress["within_assets_opened"] or progress["external_assets_opened"]):
        raise SpeedSmokeV2Error("speed-smoke V2 failure target/progress drift")
    return dict(value)


def _validate_preflight_hook(value: Mapping[str, object], identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle identity type drift")
    return validate_target_free_preflight(value, identity)


def _validate_authorization_hook(
    value: Mapping[str, object], pre_sha: str, preflight: Mapping[str, object], identity: Any,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle authorization identity drift")
    return validate_root_authorization(value, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity)


def _closure_hook(root: Path) -> Mapping[str, object]:
    return plan.implementation_closure(Path(root)).payload()


def _source_gate_hook(root: Path) -> V1FailureBinding:
    return validate_v1_failed_speed_smoke(Path(root))


def _fresh_score_hook(root: Path) -> None:
    plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)


def _fixed_from_preflight(preflight: Mapping[str, object]) -> shared_score.FixedEvaluationAuthority:
    try:
        return shared_score._fixed_authority_from_payload(preflight.get("evaluation_authority"))
    except shared_score.ScoreError as error:
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle fixed authority drift") from error


def _input_payload_hook(authority: Any, identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle input identity drift")
    return input_payload(authority, identity)


def _validate_input_hook(value: Mapping[str, object], identity: Any, fixed: Any) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity) or not isinstance(fixed, shared_score.FixedEvaluationAuthority):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle input boundary drift")
    return validate_input_payload(value, identity=identity, evaluation_authority=fixed)


def _summarize_hook(cells: Sequence[Mapping[str, object]], budget: int, input_payload_value: Mapping[str, object]) -> Mapping[str, object]:
    # The generic lifecycle deliberately does not inject identity/input SHA at
    # this hook. Full score validation reconstructs those exact links later.
    expected = [item for item in plan.SMOKE_ROWS if item.budget == budget]
    if len(expected) != 1 or len(cells) != 1 or not isinstance(cells[0], Mapping):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle summary topology drift")
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_budget_summary_v1",
        "budget": budget,
        "cells": [dict(cells[0])],
        "baseline_historical_precision_v2_parity": True,
        "v2_numerical_equivalence_accepted": True,
        "speedup_descriptive_only": True,
        "non_governing_engineering_smoke": True,
    }


def _budget_gate_hook(summary: Mapping[str, object], budget: int) -> Mapping[str, object]:
    if (
        not isinstance(summary, Mapping) or summary.get("budget") != budget
        or summary.get("baseline_historical_precision_v2_parity") is not True
        or summary.get("v2_numerical_equivalence_accepted") is not True
        or summary.get("speedup_descriptive_only") is not True
        or summary.get("non_governing_engineering_smoke") is not True
        or not isinstance(summary.get("cells"), list) or len(summary["cells"]) != 1
    ):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle budget summary drift")
    return {
        "budget": budget,
        "baseline_exact_prediction_sha256_parity": True,
        "transition_sequence_exact": True,
        "v2_numerical_equivalence_accepted": True,
        "speedup_descriptive_only": True,
        "completed": True,
        "formal_gate": False,
    }


def _build_score_hook(
    identity: Any, input_sha: str, summaries: Mapping[str, object], gates: Mapping[int, Mapping[str, object]],
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle score identity drift")
    return _score_payload(identity, input_sha, summaries, gates)


def _validate_score_hook(
    value: Mapping[str, object], identity: Any, input_payload_value: Mapping[str, object], input_sha: str, fixed: Any,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity) or not isinstance(fixed, shared_score.FixedEvaluationAuthority):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle score boundary drift")
    return validate_score_payload(
        value, identity=identity, input_payload_value=input_payload_value,
        input_authority_sha256=input_sha, evaluation_authority=fixed,
    )


def _terminal_verdict_hook(gates: Mapping[int, Mapping[str, object]]) -> str:
    return terminal_verdict(gates)


def _make_terminal_hook(
    identity: Any, attempt_sha: str, input_sha: str, score_sha: str, gates: Mapping[int, Mapping[str, object]], verdict: str,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle terminal identity drift")
    return _terminal_payload(identity, attempt_sha, input_sha, score_sha, gates, verdict)


def _validate_terminal_hook(
    value: Mapping[str, object], identity: Any, attempt_sha: str, input_sha: str,
    score_payload_value: Mapping[str, object], score_sha: str,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle terminal identity drift")
    return validate_terminal_payload(
        value, identity=identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha,
        score_payload_value=score_payload_value, score_sha256=score_sha,
    )


def _make_failure_hook(
    identity: Any, attempt_sha: str, input_sha: str | None, stage: str, error: BaseException, progress: object | None,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle failure identity drift")
    return _failure_payload(identity, attempt_sha, input_sha, stage, error, progress)


def _validate_failure_hook(
    value: Mapping[str, object], identity: Any, attempt_sha: str, input_sha: str | None,
) -> Mapping[str, object]:
    if not isinstance(identity, SpeedSmokeV2Identity):
        raise SpeedSmokeV2Error("speed-smoke V2 lifecycle failure identity drift")
    return validate_failure_payload(value, identity=identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha)


def _validate_reserved_score_artifact(root: Path, artifact: shared_score.ArtifactRoot, identity: Any) -> None:
    if not isinstance(identity, SpeedSmokeV2Identity):
        raise SpeedSmokeV2Error("speed-smoke V2 reserved artifact identity drift")
    try:
        shared_score.validate_reserved_artifact_root(
            Path(root), artifact, root_relative=plan.SCORE_ROOT_RELATIVE, topology=SCORE_TOPOLOGY,
        )
    except shared_score.ScoreError as error:
        raise SpeedSmokeV2Error(str(error)) from error


SPEED_SMOKE_V2_LIFECYCLE_HOOKS = shared_score.ProfiledLifecycleHooks(
    route="precision_aware_causal_dual_memory_cell_d_speed_smoke_v2",
    score_root_relative=plan.SCORE_ROOT_RELATIVE,
    budgets=(4,),
    require_capability=shared_score.require_execution_capability,
    validate_preflight=_validate_preflight_hook,
    validate_authorization=_validate_authorization_hook,
    implementation_closure=_closure_hook,
    validate_source_gate=_source_gate_hook,
    assert_fresh_score_root=_fresh_score_hook,
    make_attempt=_attempt_payload,
    fixed_authority_from_preflight=_fixed_from_preflight,
    input_payload=_input_payload_hook,
    validate_input_payload=_validate_input_hook,
    summarize_budget=_summarize_hook,
    budget_gate=_budget_gate_hook,
    continue_after_budget=lambda _budget, _gates: True,
    build_score=_build_score_hook,
    validate_score=_validate_score_hook,
    terminal_verdict=_terminal_verdict_hook,
    make_terminal=_make_terminal_hook,
    validate_terminal=_validate_terminal_hook,
    make_failure=_make_failure_hook,
    validate_failure=_validate_failure_hook,
    validate_reserved_score_artifact=_validate_reserved_score_artifact,
)


def run_authorized_speed_smoke_v2_lifecycle(
    root: Path, *, identity: SpeedSmokeV2Identity, capability: object, backend: shared_score.ScoreBackend,
    artifact: shared_score.ArtifactRoot, official_preflight_sha256: str, root_authorization_sha256: str,
    preflight: Mapping[str, object], authorization: Mapping[str, object], environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    _validate_mutating_launch_environment(environ)
    _require_live_predecessors(Path(root), identity)
    _assert_current_closure(Path(root), identity)
    return shared_score.run_profiled_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=official_preflight_sha256, root_authorization_sha256=root_authorization_sha256,
        preflight=preflight, authorization=authorization, hooks=SPEED_SMOKE_V2_LIFECYCLE_HOOKS,
    )
