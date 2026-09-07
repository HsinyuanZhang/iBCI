"""Immutable producer/anchor binding and the four-session paired gap lifecycle.

Standard-library-only.  The Torch/parser work lives behind ``physical`` and
runs only after this lifecycle has published a durable attempt.  Historical
producer and anchor receipt graphs are descriptor-read as immutable evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1 import score as shared
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

from . import plan


class M1GapScoreError(RuntimeError):
    """Fail closed for producer, anchor, lifecycle, or paired-gap drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M1GapScoreError(message)


def _json_bytes(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    try:
        return plan.require_sha(value, label)
    except plan.M1GapPlanError as error:
        raise M1GapScoreError(str(error)) from error


def _safe_relative(value: object) -> str:
    try:
        return plan.safe_relative(value)
    except plan.M1GapPlanError as error:
        raise M1GapScoreError(str(error)) from error


def session_arm(session_id: object) -> str:
    _require(isinstance(session_id, str) and session_id in plan.SCORE_ORDER,
             "M1 gap session roster drift")
    return "heldout_fold" if session_id in plan.HELDOUT_FOLD_SESSIONS else "heldin_training"


# ---------------------------------------------------------------------------
# Frozen evidence loaders (producer graph + anchor receipt graph)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorEvidence:
    """Held descriptor proof of the accepted frozen held-in receipt."""

    root_identity: tuple[int, int]
    named_chain_identities: tuple[tuple[str, int, int], ...]
    input_authority: Mapping[str, object] = field(repr=False, compare=False)
    score: Mapping[str, object] = field(repr=False, compare=False)
    anchor: plan.AnchorLiterals = plan.DEFAULT_ANCHOR

    def __post_init__(self) -> None:
        literals = self.anchor.payload()
        inputs = dict(self.input_authority)
        scored = dict(self.score)
        _require(isinstance(self.anchor, plan.AnchorLiterals)
                 and len(self.root_identity) == 2 and self.named_chain_identities
                 and _sha(_json_bytes(inputs)) == literals["input_authority_sha256"]
                 and _sha(_json_bytes(scored)) == literals["score_sha256"],
                 "M1 gap anchor receipt SHA drift")
        _require(scored.get("input_authority") == inputs
                 and scored.get("target_session") == plan.ANCHOR_SESSION
                 and float(scored["governing_r2"]) == literals["governing_r2"]
                 and scored.get("prediction_sha256") == literals["prediction_sha256"]
                 and scored.get("n_windows") == literals["n_windows"]
                 and scored.get("full_system_forward_count") == literals["forward_batches"],
                 "M1 gap anchor score value drift")
        object.__setattr__(self, "input_authority", MappingProxyType(inputs))
        object.__setattr__(self, "score", MappingProxyType(scored))

    def same_input_fragment(self) -> dict[str, object]:
        """The session-fact fields any reproduction must match verbatim."""
        inputs = dict(self.input_authority)
        return {
            "n_windows": inputs.get("n_windows"),
            "model_input_shape": inputs.get("model_input_shape"),
            "calibration_shape_per_row": inputs.get("calibration_shape_per_row"),
            "ordered_window_start_sha256": inputs.get("ordered_window_start_sha256"),
            "target_sha256": inputs.get("target_sha256"),
            "calibration_sha256": inputs.get("calibration_sha256"),
            "target_descriptor": inputs.get("target_descriptor"),
            "target_reader_native_evidence": inputs.get("target_reader_native_evidence"),
        }

    def payload(self) -> dict[str, object]:
        anchor = self.anchor.payload()
        return {
            "schema": "m1_heldin_heldout_gap_anchor_evidence_v1",
            "anchor_literals": anchor,
            "root_identity": list(self.root_identity),
            "named_chain_identities": [list(item) for item in self.named_chain_identities],
            "input_authority_sha256": anchor["input_authority_sha256"],
            "score_sha256": anchor["score_sha256"],
            "terminal_sha256": anchor["terminal_sha256"],
            "governing_r2": anchor["governing_r2"],
            "prediction_sha256": anchor["prediction_sha256"],
            "same_input_fields_validated": sorted(self.same_input_fragment()),
        }


def _open_held_directory(root: Path, relative: str) -> tuple[int, list[int], tuple[tuple[str, int, int], ...]]:
    try:
        return shared._open_held_result_directory(Path(root), relative)
    except shared.HeldInScoreError as error:
        raise M1GapScoreError(f"M1 gap held directory drift: {relative}") from error


def load_anchor_evidence(
    root: Path, *, anchor: plan.AnchorLiterals = plan.DEFAULT_ANCHOR,
) -> AnchorEvidence:
    """Hold and validate the exact frozen anchor receipt graph."""
    literals = anchor.payload()
    _base_fd, opened, identities = _open_held_directory(Path(root), plan.ANCHOR_ROOT_RELATIVE)
    result_fd = opened[-1]
    try:
        expected = tuple(sorted(shared._success_names(terminal=True)))
        _require(tuple(sorted(os.listdir(result_fd))) == expected,
                 "M1 gap anchor exact leaf topology drift")
        inputs, input_sha = shared._read_json_pair(
            result_fd, "input_authority.json", expected_sha256=literals["input_authority_sha256"],
        )
        scored, score_sha = shared._read_json_pair(
            result_fd, "score.json", expected_sha256=literals["score_sha256"],
        )
        terminal, _terminal_sha = shared._read_json_pair(
            result_fd, "terminal.json", expected_sha256=literals["terminal_sha256"],
        )
        _require(terminal.get("status") == plan.ANCHOR_TERMINAL_STATUS
                 and terminal.get("input_authority_sha256") == input_sha
                 and terminal.get("score_sha256") == score_sha
                 and terminal.get("governing_r2") == literals["governing_r2"],
                 "M1 gap anchor terminal linkage drift")
        _require(_named_chain_identities(Path(root), plan.ANCHOR_ROOT_RELATIVE) == identities,
                 "M1 gap anchor graph changed during read")
        info = os.fstat(result_fd)
        return AnchorEvidence(
            shared._directory_identity(info), identities, inputs, scored, anchor,
        )
    except shared.HeldInScoreError as error:
        raise M1GapScoreError("M1 gap anchor immutable receipt drift") from error
    finally:
        shared._close_held_chain(opened)


def _named_chain_identities(root: Path, relative: str) -> tuple[tuple[str, int, int], ...]:
    _root_fd, opened, identities = _open_held_directory(Path(root), relative)
    try:
        return identities
    finally:
        shared._close_held_chain(opened)


def load_producer_graph(
    root: Path, *, expectation: plan.cswg_plan.CompletedFullExpectation
    = plan.cswg_plan.DEFAULT_COMPLETED_FULL_EXPECTATION,
) -> shared.CompletedFullGraph:
    """Descriptor-load the frozen 56-leaf CS-WG producer graph."""
    try:
        return shared.load_completed_full_graph(Path(root), expectation=expectation)
    except shared.HeldInScoreError as error:
        raise M1GapScoreError("M1 gap frozen producer graph drift") from error


# ---------------------------------------------------------------------------
# Per-session input authority and score payload codecs
# ---------------------------------------------------------------------------

_SESSION_INPUT_PROTECTED = {
    "schema", "identity_sha256", "producer_graph_sha256", "anchor_evidence_sha256",
    "session_id", "arm", "budget", "selected_checkpoint_role", "metric", "last_bin_only",
    "target_metric_only", "target_labels_used_only_for_metric", "target_optimizer_backward_update",
}

_ANCHOR_COMPARED_FIELDS = (
    "n_windows", "model_input_shape", "calibration_shape_per_row",
    "ordered_window_start_sha256", "target_sha256", "calibration_sha256",
    "target_descriptor", "target_reader_native_evidence",
)


def _validate_session_prepared(
    session_id: str, prepared: Mapping[str, object], *, forbid_protected: bool = False,
) -> dict[str, object]:
    _require(isinstance(prepared, Mapping)
             and (not forbid_protected or not (set(prepared) & _SESSION_INPUT_PROTECTED)),
             f"M1 gap session {session_id} prepared fragment overwrote protected fields")
    item = dict(prepared)
    descriptor = item.get("target_descriptor")
    native = item.get("target_reader_native_evidence")
    _require(isinstance(descriptor, Mapping) and descriptor.get("session_id") == session_id
             and descriptor.get("role") == "m1_heldin_source_nwb"
             and descriptor.get("source_only") is True
             and isinstance(native, Mapping)
             and native.get("session_id") == session_id
             and native.get("all_row_calibration_sha256_match_session") is True
             and native.get("no_cross_session_calibration_substitution") is True
             and native.get("calibration_session") == session_id
             and type(item.get("n_windows")) is int and item["n_windows"] > 0
             and item.get("model_input_shape") == [plan.MODEL_SHAPE["window"], plan.MODEL_SHAPE["units"]]
             and item.get("calibration_shape_per_row") == plan.MODEL_SHAPE["calibration_shape_per_row"]
             and all(_require_sha(item.get(name), f"M1 gap session {session_id} {name}")
                     for name in ("input_record_sha256", "ordered_window_start_sha256",
                                  "target_sha256", "calibration_sha256")),
             f"M1 gap session {session_id} prepared input semantics drift")
    return item


def _session_input_authority_payload(
    identity: plan.GapIdentity, graph: shared.CompletedFullGraph, anchor: AnchorEvidence,
    session_id: str, prepared: Mapping[str, object],
) -> dict[str, object]:
    _require(isinstance(identity, plan.GapIdentity) and isinstance(graph, shared.CompletedFullGraph)
             and isinstance(anchor, AnchorEvidence) and session_id in plan.SCORE_ORDER,
             "M1 gap session input authority construction drift")
    item = _validate_session_prepared(session_id, prepared, forbid_protected=True)
    if session_id == plan.ANCHOR_SESSION:
        expected = anchor.same_input_fragment()
        for name in _ANCHOR_COMPARED_FIELDS:
            _require(item.get(name) == expected.get(name),
                     f"M1 gap anchor session reproduction drift: {name}")
    result = {
        "schema": "m1_heldin_heldout_gap_session_input_authority_v1",
        "identity_sha256": identity.sha256,
        "producer_graph_sha256": graph.sha256,
        "anchor_evidence_sha256": _sha(_json_bytes(anchor.payload())),
        "session_id": session_id,
        "arm": session_arm(session_id),
        "budget": plan.BUDGET_LABEL,
        "selected_checkpoint_role": "best_source_train_loss",
        "metric": plan.METRIC_LABEL,
        "last_bin_only": True,
        "target_metric_only": True,
        "target_labels_used_only_for_metric": True,
        "target_optimizer_backward_update": 0,
        "anchor_session_reproduction": session_id == plan.ANCHOR_SESSION,
    }
    result.update(item)
    return _validate_session_input_authority(result, identity, graph, anchor, session_id)


def _validate_session_input_authority(
    value: object, identity: plan.GapIdentity, graph: shared.CompletedFullGraph,
    anchor: AnchorEvidence, session_id: str,
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "M1 gap session input authority must be a mapping")
    item = dict(value)
    _require(item.get("schema") == "m1_heldin_heldout_gap_session_input_authority_v1"
             and item.get("identity_sha256") == identity.sha256
             and item.get("producer_graph_sha256") == graph.sha256
             and item.get("anchor_evidence_sha256") == _sha(_json_bytes(anchor.payload()))
             and item.get("session_id") == session_id and session_id in plan.SCORE_ORDER
             and item.get("arm") == session_arm(session_id)
             and item.get("budget") == plan.BUDGET_LABEL
             and item.get("selected_checkpoint_role") == "best_source_train_loss"
             and item.get("metric") == plan.METRIC_LABEL and item.get("last_bin_only") is True
             and item.get("target_metric_only") is True
             and item.get("target_labels_used_only_for_metric") is True
             and item.get("target_optimizer_backward_update") == 0
             and item.get("anchor_session_reproduction") is (session_id == plan.ANCHOR_SESSION),
             f"M1 gap session {session_id} input authority fixed fields drift")
    item = _validate_session_prepared(session_id, item)
    if session_id == plan.ANCHOR_SESSION:
        expected = anchor.same_input_fragment()
        for name in _ANCHOR_COMPARED_FIELDS:
            _require(item.get(name) == expected.get(name),
                     f"M1 gap anchor session reproduction drift: {name}")
    return item


def _session_score_payload(
    identity: plan.GapIdentity, graph: shared.CompletedFullGraph, anchor: AnchorEvidence,
    session_id: str, input_authority: Mapping[str, object], *, forward_count: int,
    prediction_sha256: str, target_sha256: str, governing_r2: float,
    model_state_before_sha256: str, model_state_after_sha256: str,
) -> dict[str, object]:
    inputs = _validate_session_input_authority(
        input_authority, identity, graph, anchor, session_id,
    )
    _require(model_state_before_sha256 == model_state_after_sha256
             == plan.ANCHOR_MODEL_STATE_SHA256,
             f"M1 gap session {session_id} frozen model-state drift")
    body = {
        "schema": "m1_heldin_heldout_gap_session_score_v1",
        "identity_sha256": identity.sha256,
        "producer_graph_sha256": graph.sha256,
        "input_authority_sha256": _sha(_json_bytes(inputs)),
        "session_id": session_id,
        "arm": session_arm(session_id),
        "budget": plan.BUDGET_LABEL,
        "selected_checkpoint_role": "best_source_train_loss",
        "metric": plan.METRIC_LABEL,
        "last_bin_only": True,
        "eval_mode": True,
        "no_grad": True,
        "dynamic_dropout_disabled": True,
        "target_metric_only": True,
        "target_labels_used_only_for_metric": True,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "full_system_forward_count": forward_count,
        "n_windows": inputs["n_windows"],
        "governing_r2": governing_r2,
        "prediction_sha256": _require_sha(prediction_sha256, f"M1 gap session {session_id} prediction"),
        "target_sha256": _require_sha(target_sha256, f"M1 gap session {session_id} target"),
        "model_state_before_sha256": _require_sha(model_state_before_sha256, "M1 gap state before"),
        "model_state_after_sha256": _require_sha(model_state_after_sha256, "M1 gap state after"),
        "anchor_session_reproduction": session_id == plan.ANCHOR_SESSION,
    }
    if session_id == plan.ANCHOR_SESSION:
        literals = anchor.anchor.payload()
        _require(float(governing_r2) == literals["governing_r2"]
                 and prediction_sha256 == literals["prediction_sha256"]
                 and target_sha256 == literals["target_sha256"]
                 and forward_count == literals["forward_batches"]
                 and inputs["n_windows"] == literals["n_windows"]
                 and inputs["calibration_sha256"] == literals["calibration_sha256"],
                 "M1 gap anchor session exact reproduction drift")
        body["anchor_reproduced_exactly"] = True
    return body


def _validate_session_score_payload(
    value: object, identity: plan.GapIdentity, graph: shared.CompletedFullGraph,
    anchor: AnchorEvidence, session_id: str,
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "M1 gap session score must be a mapping")
    item = dict(value)
    inputs = _validate_session_input_authority(
        item.get("input_authority"), identity, graph, anchor, session_id,
    )
    numeric = item.get("governing_r2")
    required_sha = ("prediction_sha256", "target_sha256",
                    "model_state_before_sha256", "model_state_after_sha256")
    _require(item.get("schema") == "m1_heldin_heldout_gap_session_score_v1"
             and item.get("identity_sha256") == identity.sha256
             and item.get("producer_graph_sha256") == graph.sha256
             and item.get("session_id") == session_id
             and item.get("arm") == session_arm(session_id)
             and item.get("budget") == plan.BUDGET_LABEL
             and item.get("selected_checkpoint_role") == "best_source_train_loss"
             and item.get("metric") == plan.METRIC_LABEL and item.get("last_bin_only") is True
             and item.get("eval_mode") is True and item.get("no_grad") is True
             and item.get("dynamic_dropout_disabled") is True and item.get("target_metric_only") is True
             and item.get("target_labels_used_only_for_metric") is True
             and item.get("target_optimizer_steps") == 0 and item.get("target_backward_calls") == 0
             and item.get("target_update_calls") == 0
             and type(item.get("full_system_forward_count")) is int and item["full_system_forward_count"] > 0
             and item.get("n_windows") == inputs.get("n_windows")
             and type(numeric) in {int, float} and math.isfinite(float(numeric))
             and all(_require_sha(item.get(name), f"M1 gap score {name}") for name in required_sha)
             and item.get("model_state_before_sha256") == item.get("model_state_after_sha256")
             == plan.ANCHOR_MODEL_STATE_SHA256
             and item.get("target_sha256") == inputs.get("target_sha256"),
             f"M1 gap session {session_id} score metric/state/update drift")
    if session_id == plan.ANCHOR_SESSION:
        literals = anchor.anchor.payload()
        _require(item.get("anchor_reproduced_exactly") is True
                 and float(item["governing_r2"]) == literals["governing_r2"]
                 and item["prediction_sha256"] == literals["prediction_sha256"]
                 and item["target_sha256"] == literals["target_sha256"]
                 and item["full_system_forward_count"] == literals["forward_batches"],
                 "M1 gap anchor session exact reproduction drift")
    return item


# ---------------------------------------------------------------------------
# Pre-registered paired statistics and verdict
# ---------------------------------------------------------------------------


def _finite_float(value: object) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(float(value)), "M1 gap expected finite scalar")
    return float(value)


def arm_summary(values: Mapping[str, float], *, sessions: tuple[str, ...]) -> dict[str, object]:
    expected = tuple(sessions)
    _require(tuple(sorted(values)) == expected, "M1 gap arm session roster drift")
    data = [_finite_float(values[name]) for name in expected]
    mean = sum(data) / len(data)
    variance = sum((item - mean) ** 2 for item in data) / len(data)
    return {
        "sessions": list(expected),
        "n": len(data),
        "equal_session_mean": mean,
        "equal_session_sd": math.sqrt(variance),
        "sd_ddof": 0,
        "per_session": {name: value for name, value in zip(expected, data)},
    }


def paired_bootstrap(
    heldin_values: Mapping[str, float], heldout_values: Mapping[str, float], *,
    seed: int = plan.BOOTSTRAP_SEED, draws: int = plan.BOOTSTRAP_DRAWS,
) -> dict[str, object]:
    """Program-convention session bootstrap of the paired gap statistic."""
    _require(seed == plan.BOOTSTRAP_SEED and draws == plan.BOOTSTRAP_DRAWS,
             "M1 gap bootstrap seed/draw count is frozen")
    import numpy as np

    in_sessions = tuple(plan.HELDIN_TRAINING_SESSIONS)
    out_sessions = tuple(plan.HELDOUT_FOLD_SESSIONS)
    in_data = np.asarray([_finite_float(heldin_values[name]) for name in in_sessions], dtype=np.float64)
    out_data = np.asarray([_finite_float(heldout_values[name]) for name in out_sessions], dtype=np.float64)
    generator = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        resampled_in = in_data[generator.integers(0, len(in_data), size=len(in_data))]
        resampled_out = out_data[generator.integers(0, len(out_data), size=len(out_data))]
        boot[index] = resampled_out.mean() - resampled_in.mean()
    return {
        "seed": seed,
        "draws": draws,
        "generator": "numpy.random.default_rng(42)",
        "resampling": plan.VERDICT_RULE["bootstrap"]["resampling"],
        "heldout_arm_degenerate": len(out_sessions) == 1,
        "gap_point": float(out_data.mean() - in_data.mean()),
        "gap_bootstrap_mean": float(boot.mean()),
        "gap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def apply_preregistered_verdict(*, gap: float, ci_low: float, ci_high: float) -> dict[str, object]:
    """Apply the four ordered branches exactly as pre-registered."""
    gap = _finite_float(gap)
    ci_low = _finite_float(ci_low)
    ci_high = _finite_float(ci_high)
    _require(ci_low <= ci_high, "M1 gap CI ordering drift")
    tolerance = plan.GAP_TOLERANCE
    branches = plan.VERDICT_RULE["branches_in_order"]
    if abs(gap) <= tolerance and ci_low >= -tolerance and ci_high <= tolerance:
        selected = branches[0]
    elif gap < -tolerance or ci_high < -tolerance:
        selected = branches[1]
    elif gap > tolerance or ci_low > tolerance:
        selected = branches[2]
    else:
        selected = branches[3]
    return {
        "rule_sha256": _sha(_json_bytes(plan.VERDICT_RULE)),
        "branch_index": selected["index"],
        "condition": selected["condition"],
        "verdict": selected["verdict"],
        "direction": selected["direction"],
        "gap": gap,
        "ci95": [ci_low, ci_high],
        "tolerance": tolerance,
        "decided_before_any_scoring": True,
    }


def build_paired_table(
    identity: plan.GapIdentity, graph: shared.CompletedFullGraph, anchor: AnchorEvidence,
    session_scores: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    _require(tuple(sorted(session_scores)) == tuple(sorted(plan.SCORE_ORDER)),
             "M1 gap paired table session roster drift")
    rows = []
    values: dict[str, dict[str, float]] = {"heldin_training": {}, "heldout_fold": {}}
    for session_id in plan.SCORE_ORDER:
        scored = _validate_session_score_payload(
            session_scores[session_id], identity, graph, anchor, session_id,
        )
        arm = session_arm(session_id)
        values[arm][session_id] = float(scored["governing_r2"])
        rows.append({
            "session_id": session_id,
            "arm": arm,
            "budget": plan.BUDGET_LABEL,
            "checkpoint_relationship": (
                "fold0_outer_target_held_out_from_training"
                if arm == "heldout_fold" else "producer_training_session"
            ),
            "also_fold_target_of": plan.FOLD_TARGET_OF_SESSION[session_id],
            "n_windows": scored["n_windows"],
            "governing_r2": scored["governing_r2"],
            "prediction_sha256": scored["prediction_sha256"],
            "target_sha256": scored["target_sha256"],
            "calibration_sha256": _validate_session_input_authority(
                scored["input_authority"], identity, graph, anchor, session_id,
            )["calibration_sha256"],
            "forward_batches": scored["full_system_forward_count"],
            "anchor_session_reproduction": scored["anchor_session_reproduction"],
        })
    heldin = arm_summary(values["heldin_training"], sessions=plan.HELDIN_TRAINING_SESSIONS)
    heldout = arm_summary(values["heldout_fold"], sessions=plan.HELDOUT_FOLD_SESSIONS)
    bootstrap = paired_bootstrap(values["heldin_training"], values["heldout_fold"])
    gap = heldout["equal_session_mean"] - heldin["equal_session_mean"]
    verdict = apply_preregistered_verdict(
        gap=gap, ci_low=bootstrap["gap_ci95"][0], ci_high=bootstrap["gap_ci95"][1],
    )
    _require(float(bootstrap["gap_point"]) == float(gap), "M1 gap point-estimate drift")
    return {
        "schema": "m1_heldin_heldout_gap_paired_table_v1",
        "cell": plan.CELL,
        "phase": plan.PHASE,
        "identity_sha256": identity.sha256,
        "producer_graph_sha256": graph.sha256,
        "budget": plan.BUDGET_LABEL,
        "metric": plan.METRIC_LABEL,
        "anchor": {
            "root_relative": plan.ANCHOR_ROOT_RELATIVE,
            "session_id": plan.ANCHOR_SESSION,
            "input_authority_sha256": plan.ANCHOR_INPUT_AUTHORITY_SHA256,
            "score_sha256": plan.ANCHOR_SCORE_SHA256,
            "governing_r2": plan.ANCHOR_GOVERNING_R2,
            "prediction_sha256": plan.ANCHOR_PREDICTION_SHA256,
            "reproduced_exactly": True,
            "receipt_naming_note": (
                "the frozen receipt names this session held-in in the held-in-calib split "
                "sense; operationally it is this producer's fold-defined held-out session"
            ),
        },
        "sessions": rows,
        "heldin_training_arm": heldin,
        "heldout_fold_arm": heldout,
        "gap": {
            "definition": plan.VERDICT_RULE["gap_definition"],
            "value": float(gap),
        },
        "bootstrap": bootstrap,
        "verdict": verdict,
        "official_heldout_surface_blocked": dict(plan.OFFICIAL_HELDOUT_BLOCKED),
        "supporting_diagnostic": {
            "relative_path": plan.SUPPORTING_DIAGNOSTIC_RELATIVE,
            "sha256": plan.SUPPORTING_DIAGNOSTIC_SHA256,
            "static_support_r2": plan.SUPPORTING_DIAGNOSTIC_STATIC_R2,
            "causal_growing_cap30_r2": plan.SUPPORTING_DIAGNOSTIC_GROWING_R2,
            "read_only_quote": True,
            "part_of_gap_or_verdict": False,
        },
        "formal_benchmark_verdict": False,
        "target_optimizer_backward_update": 0,
    }


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapProgress:
    sessions_resolved_or_opened: tuple[str, ...] = ()
    checkpoint_opened: bool = False
    model_constructed: bool = False
    cuda_initialized: bool = False
    full_system_forward_batches: int = 0
    input_authorities_published: int = 0
    session_scores_published: int = 0
    target_optimizer_steps: int = 0
    target_backward_calls: int = 0
    target_update_calls: int = 0

    def __post_init__(self) -> None:
        _require(all(isinstance(item, str) for item in self.sessions_resolved_or_opened)
                 and all(type(value) is bool for value in (
                     self.checkpoint_opened, self.model_constructed, self.cuda_initialized,
                 ))
                 and all(type(value) is int and value >= 0 for value in (
                     self.full_system_forward_batches, self.input_authorities_published,
                     self.session_scores_published, self.target_optimizer_steps,
                     self.target_backward_calls, self.target_update_calls,
                 )), "M1 gap lifecycle progress drift")

    def payload(self) -> dict[str, object]:
        return {
            "sessions_resolved_or_opened": list(self.sessions_resolved_or_opened),
            "checkpoint_opened": self.checkpoint_opened,
            "model_constructed": self.model_constructed,
            "cuda_initialized": self.cuda_initialized,
            "full_system_forward_batches": self.full_system_forward_batches,
            "input_authorities_published": self.input_authorities_published,
            "session_scores_published": self.session_scores_published,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "target_metric_only": True,
            "target_labels_used_only_for_metric": True,
            "target_training_forbidden": True,
        }


class DeferredGapBackend(Protocol):
    def launch_payload(self, identity: plan.GapIdentity, graph: shared.CompletedFullGraph,
                       anchor: AnchorEvidence) -> Mapping[str, object]: ...
    def prepare_session(self, identity: plan.GapIdentity, graph: shared.CompletedFullGraph,
                        session_id: str) -> Mapping[str, object]: ...
    def score_session(self, identity: plan.GapIdentity, graph: shared.CompletedFullGraph,
                      session_id: str, input_authority: Mapping[str, object]) -> Mapping[str, object]: ...
    def progress(self) -> GapProgress: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class GapLifecycleResult:
    root_identity: tuple[int, int]
    attempt_sha256: str
    launch_sha256: str | None
    session_input_sha256: dict[str, str]
    session_score_sha256: dict[str, str]
    paired_table_sha256: str | None
    terminal_sha256: str | None
    failure_sha256: str | None


def _attempt_payload(
    identity: plan.GapIdentity, graph: shared.CompletedFullGraph, anchor: AnchorEvidence,
) -> dict[str, object]:
    return {
        "schema": "m1_heldin_heldout_gap_attempt_v1",
        "cell": plan.CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "producer_completed_full_graph": graph.payload(),
        "anchor_evidence": anchor.payload(),
        "preregistered_verdict_rule": plan.VERDICT_RULE,
        **GapProgress().payload(),
    }


def _launch_payload(
    identity: plan.GapIdentity, graph: shared.CompletedFullGraph, anchor: AnchorEvidence,
    attempt_sha256: str, backend: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "m1_heldin_heldout_gap_launch_v1",
        "cell": plan.CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "producer_completed_full_graph_sha256": graph.sha256,
        "anchor_evidence_sha256": _sha(_json_bytes(anchor.payload())),
        "attempt_sha256": _require_sha(attempt_sha256, "M1 gap launch attempt"),
        "backend": dict(backend),
        "score_order": list(plan.SCORE_ORDER),
        **GapProgress().payload(),
    }


def _terminal_payload(
    identity: plan.GapIdentity, graph: shared.CompletedFullGraph, anchor: AnchorEvidence, *,
    attempt_sha256: str, launch_sha256: str, session_input_sha256: Mapping[str, str],
    session_score_sha256: Mapping[str, str], paired_table_sha256: str,
    paired_table: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "m1_heldin_heldout_gap_terminal_v1",
        "cell": plan.CELL,
        "status": "COMPLETE_DESCRIPTIVE_HELDIN_HELDOUT_GAP",
        "identity": identity.payload(),
        "producer_completed_full_graph_sha256": graph.sha256,
        "anchor_evidence_sha256": _sha(_json_bytes(anchor.payload())),
        "attempt_sha256": _require_sha(attempt_sha256, "M1 gap terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "M1 gap terminal launch"),
        "session_input_authority_sha256": dict(session_input_sha256),
        "session_score_sha256": dict(session_score_sha256),
        "paired_table_sha256": _require_sha(paired_table_sha256, "M1 gap terminal paired table"),
        "heldin_training_equal_session_mean": paired_table["heldin_training_arm"]["equal_session_mean"],
        "heldin_training_equal_session_sd": paired_table["heldin_training_arm"]["equal_session_sd"],
        "heldout_fold_equal_session_mean": paired_table["heldout_fold_arm"]["equal_session_mean"],
        "gap": paired_table["gap"]["value"],
        "gap_ci95": paired_table["bootstrap"]["gap_ci95"],
        "verdict": paired_table["verdict"]["verdict"],
        "verdict_direction": paired_table["verdict"]["direction"],
        "anchor_reproduced_exactly": True,
        "selected_checkpoint_role": "best_source_train_loss",
        "formal_benchmark_verdict": False,
        "target_optimizer_backward_update": 0,
        "target_metric_only": True,
    }


def _failure_payload(
    identity: plan.GapIdentity, graph: shared.CompletedFullGraph, anchor: AnchorEvidence, *,
    attempt_sha256: str, launch_sha256: str | None,
    session_input_sha256: Mapping[str, str], session_score_sha256: Mapping[str, str],
    paired_table_sha256: str | None, progress: GapProgress, error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "m1_heldin_heldout_gap_failure_v1",
        "cell": plan.CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "producer_completed_full_graph_sha256": graph.sha256,
        "anchor_evidence_sha256": _sha(_json_bytes(anchor.payload())),
        "attempt_sha256": _require_sha(attempt_sha256, "M1 gap failure attempt"),
        "launch_sha256": None if launch_sha256 is None else _require_sha(
            launch_sha256, "M1 gap failure launch"),
        "session_input_authority_sha256": dict(session_input_sha256),
        "session_score_sha256": dict(session_score_sha256),
        "paired_table_sha256": None if paired_table_sha256 is None else _require_sha(
            paired_table_sha256, "M1 gap failure paired table"),
        "progress": progress.payload(),
        "error_class": type(error).__name__,
        "error_sha256": _sha(repr(error).encode("utf-8")),
        "terminal_published": False,
        "target_metric_only": True,
        "target_optimizer_backward_update": 0,
    }


def _success_names(*, terminal: bool, paired_table: bool = True) -> tuple[str, ...]:
    bodies = ["attempt.json", "launch.json"]
    for session_id in plan.SCORE_ORDER:
        bodies.append(f"input_authority_{session_id}.json")
    for session_id in plan.SCORE_ORDER:
        bodies.append(f"score_{session_id}.json")
    if paired_table:
        bodies.append("paired_table.json")
    if terminal:
        bodies.append("terminal.json")
    return tuple(item for body in bodies for item in (body, f"{body}.sha256"))


def _revalidate_published_graph(
    artifact: v1.ImmutableArtifactRoot, identity: plan.GapIdentity, graph: shared.CompletedFullGraph,
    anchor: AnchorEvidence, *, attempt_sha256: str, launch_sha256: str,
    session_input_sha256: Mapping[str, str], session_score_sha256: Mapping[str, str],
) -> dict[str, Mapping[str, object]]:
    artifact.validate_live(expected_names=_success_names(terminal=False, paired_table=False))
    attempt = artifact.read_json_pair("attempt.json", expected_sha256=attempt_sha256)
    launch = artifact.read_json_pair("launch.json", expected_sha256=launch_sha256)
    _require(attempt == _attempt_payload(identity, graph, anchor)
             and launch.get("attempt_sha256") == attempt_sha256
             and launch.get("producer_completed_full_graph_sha256") == graph.sha256,
             "M1 gap published attempt/launch graph drift")
    inputs: dict[str, Mapping[str, object]] = {}
    scores: dict[str, Mapping[str, object]] = {}
    for session_id in plan.SCORE_ORDER:
        inputs[session_id] = artifact.read_json_pair(
            f"input_authority_{session_id}.json", expected_sha256=session_input_sha256[session_id],
        )
        _validate_session_input_authority(inputs[session_id], identity, graph, anchor, session_id)
        scores[session_id] = artifact.read_json_pair(
            f"score_{session_id}.json", expected_sha256=session_score_sha256[session_id],
        )
        _require(scores[session_id].get("input_authority") == inputs[session_id],
                 f"M1 gap published input snapshot drift: {session_id}")
        _validate_session_score_payload(scores[session_id], identity, graph, anchor, session_id)
    return scores


class _GapReviewSeal:
    pass


_GAP_REVIEW_SEAL = _GapReviewSeal()


@dataclass(frozen=True)
class GapCapability:
    identity_sha256: str
    producer_graph_sha256: str
    anchor_evidence_sha256: str
    target_source_root: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha(self.identity_sha256, "M1 gap capability identity")
        _require_sha(self.producer_graph_sha256, "M1 gap capability producer graph")
        _require_sha(self.anchor_evidence_sha256, "M1 gap capability anchor evidence")
        _require(isinstance(self.target_source_root, str) and self.target_source_root
                 and Path(self.target_source_root).is_absolute()
                 and str(Path(self.target_source_root)) == self.target_source_root
                 and self._seal is _GAP_REVIEW_SEAL,
                 "M1 gap needs an in-process root-reviewed capability")


def validate_identity_current(root: Path, identity: plan.GapIdentity) -> None:
    _require(isinstance(identity, plan.GapIdentity) and identity.spec == plan.GapSpec()
             and identity.anchor == plan.DEFAULT_ANCHOR,
             "M1 gap identity/spec/anchor provenance drift")
    try:
        plan.validate_current_closure(Path(root), identity.closure)
    except plan.M1GapPlanError as error:
        raise M1GapScoreError("M1 gap current closure drift") from error
    try:
        metadata = v1.load_m1_metadata_manifest_authority(Path(root))
    except v1.SourceLifecycleError as error:
        raise M1GapScoreError("M1 gap sealed target metadata authority drift") from error
    _require(metadata.get("body_sha256") == v1.M1_METADATA_MANIFEST_SHA256,
             "M1 gap target metadata body binding drift")


def assert_prospective_score_root_fresh(root: Path, spec: plan.GapSpec) -> None:
    candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise M1GapScoreError("M1 gap prospective root cannot be safely inspected") from error
    raise M1GapScoreError("M1 gap canonical score root already exists")


def issue_root_reviewed_gap_capability(
    root: Path, *, identity: plan.GapIdentity, source_root: Path, review_seal: object,
) -> GapCapability:
    _require(review_seal is _GAP_REVIEW_SEAL, "only the root reviewer may issue M1 gap capability")
    validate_identity_current(Path(root), identity)
    graph = load_producer_graph(Path(root))
    anchor = load_anchor_evidence(Path(root))
    assert_prospective_score_root_fresh(Path(root), identity.spec)
    _require(Path(source_root).is_absolute(),
             "M1 gap root reviewer must supply an absolute target source root")
    return GapCapability(
        identity.sha256, graph.sha256, _sha(_json_bytes(anchor.payload())),
        str(Path(source_root)), _GAP_REVIEW_SEAL,
    )


def _require_capability(capability: object, identity: plan.GapIdentity) -> GapCapability:
    _require(isinstance(capability, GapCapability) and capability._seal is _GAP_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "M1 gap exact root-reviewed capability is required")
    return capability


def execute_reviewed_gap_score(
    root: Path, *, identity: plan.GapIdentity, capability: object, backend: object,
    graph_loader: Callable[[Path], shared.CompletedFullGraph] = load_producer_graph,
    anchor_loader: Callable[[Path], AnchorEvidence] = load_anchor_evidence,
) -> GapLifecycleResult:
    """One reviewed four-session paired gap run through an immutable root."""
    _require(isinstance(identity, plan.GapIdentity), "M1 gap identity type drift")
    cap = _require_capability(capability, identity)
    validate_identity_current(Path(root), identity)
    graph = graph_loader(Path(root))
    anchor = anchor_loader(Path(root))
    _require(_sha(_json_bytes(anchor.payload())) == cap.anchor_evidence_sha256
             and graph.sha256 == cap.producer_graph_sha256,
             "M1 gap capability/producer/anchor drift before reserve")
    backend_source_root = Path(getattr(backend, "source_root", ""))
    _require(backend_source_root.is_absolute() and str(backend_source_root) == cap.target_source_root,
             "M1 gap capability/backend target source-root drift before reserve")
    assert_prospective_score_root_fresh(Path(root), identity.spec)
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), identity.spec)  # type: ignore[arg-type]
    session_input_sha256: dict[str, str] = {}
    session_score_sha256: dict[str, str] = {}
    launch_sha256: str | None = None
    paired_table_sha256: str | None = None
    terminal_sha256: str | None = None
    try:
        attempt_sha256 = artifact.publish_json("attempt.json", _attempt_payload(identity, graph, anchor))
        launch_method = getattr(backend, "launch_payload", None)
        prepare_method = getattr(backend, "prepare_session", None)
        score_method = getattr(backend, "score_session", None)
        progress_method = getattr(backend, "progress", None)
        _require(all(callable(method) for method in (launch_method, prepare_method, score_method)),
                 "M1 gap backend lifecycle seam drift")
        launch_backend = launch_method(identity, graph, anchor)
        _require(isinstance(launch_backend, Mapping), "M1 gap launch backend payload drift")
        launch_sha256 = artifact.publish_json(
            "launch.json", _launch_payload(identity, graph, anchor, attempt_sha256, launch_backend),
        )
        session_scores: dict[str, Mapping[str, object]] = {}
        for session_id in plan.SCORE_ORDER:
            prepared = prepare_method(identity, graph, session_id)
            _require(isinstance(prepared, Mapping),
                     f"M1 gap session {session_id} prepared payload type drift")
            inputs = _session_input_authority_payload(identity, graph, anchor, session_id, prepared)
            session_input_sha256[session_id] = artifact.publish_json(
                f"input_authority_{session_id}.json", inputs,
            )
            raw_score = score_method(identity, graph, session_id, inputs)
            _require(isinstance(raw_score, Mapping),
                     f"M1 gap session {session_id} physical payload type drift")
            body = dict(raw_score)
            body["input_authority"] = inputs
            _validate_session_score_payload(body, identity, graph, anchor, session_id)
            session_score_sha256[session_id] = artifact.publish_json(
                f"score_{session_id}.json", body,
            )
            session_scores[session_id] = body
        artifact.validate_live(expected_names=_success_names(terminal=False, paired_table=False))
        checked_scores = _revalidate_published_graph(
            artifact, identity, graph, anchor, attempt_sha256=attempt_sha256,
            launch_sha256=launch_sha256, session_input_sha256=session_input_sha256,
            session_score_sha256=session_score_sha256,
        )
        paired_table = build_paired_table(identity, graph, anchor, checked_scores)
        paired_table_sha256 = artifact.publish_json("paired_table.json", paired_table)
        validate_identity_current(Path(root), identity)
        graph_now = graph_loader(Path(root))
        anchor_now = anchor_loader(Path(root))
        _require(graph_now.sha256 == graph.sha256 == cap.producer_graph_sha256
                 and _sha(_json_bytes(anchor_now.payload())) == cap.anchor_evidence_sha256,
                 "M1 gap immutable producer/anchor graph drifted during evaluation")
        terminal = _terminal_payload(
            identity, graph, anchor, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
            session_input_sha256=session_input_sha256, session_score_sha256=session_score_sha256,
            paired_table_sha256=paired_table_sha256, paired_table=paired_table,
        )
        terminal_sha256 = artifact.publish_json("terminal.json", terminal)
        artifact.validate_live(expected_names=_success_names(terminal=True))
        _require(artifact.read_json_pair("terminal.json", expected_sha256=terminal_sha256) == terminal,
                 "M1 gap terminal descriptor reload drift")
        return GapLifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, session_input_sha256,
            session_score_sha256, paired_table_sha256, terminal_sha256, None,
        )
    except BaseException as error:
        if terminal_sha256 is not None:
            raise
        try:
            progress = progress_method() if callable(progress_method) else None
            _require(isinstance(progress, GapProgress), "M1 gap backend progress type drift")
        except BaseException:
            progress = GapProgress()
        failure_payload = _failure_payload(
            identity, graph, anchor, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
            session_input_sha256=session_input_sha256, session_score_sha256=session_score_sha256,
            paired_table_sha256=paired_table_sha256, progress=progress, error=error,
        )
        failure_sha256 = artifact.publish_json("failure.json", failure_payload)
        return GapLifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, session_input_sha256,
            session_score_sha256, paired_table_sha256, None, failure_sha256,
        )
    finally:
        try:
            close_method = getattr(backend, "close", None)
            _require(callable(close_method), "M1 gap backend close seam drift")
            close_method()
        finally:
            artifact.close()


def dry_plan(root: Path | None = None) -> dict[str, object]:
    return plan.dry_plan(root)


__all__ = (
    "M1GapScoreError", "AnchorEvidence", "GapProgress", "GapCapability", "GapLifecycleResult",
    "DeferredGapBackend", "session_arm", "load_anchor_evidence", "load_producer_graph",
    "validate_identity_current", "assert_prospective_score_root_fresh",
    "issue_root_reviewed_gap_capability", "execute_reviewed_gap_score",
    "arm_summary", "paired_bootstrap", "apply_preregistered_verdict", "build_paired_table",
    "dry_plan",
)
