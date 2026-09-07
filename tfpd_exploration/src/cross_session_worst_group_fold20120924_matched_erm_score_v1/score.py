"""Terminal-pinned MATCHED_ERM held-in score profile.

The shared V1 scorer owns the only immutable score lifecycle loop.  This
module supplies a typed future-producer binding and a distinct receipt codec;
it cannot construct an identity or capability without exact post-terminal
literals supplied by a reviewer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1 import score as shared
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

from . import plan


class MatchedERMHeldInScoreError(RuntimeError):
    """Fail closed for ERM producer/profile/lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedERMHeldInScoreError(message)


def _json_bytes(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    try:
        return plan.require_sha(value, label)
    except plan.MatchedERMHeldInScorePlanError as error:
        raise MatchedERMHeldInScoreError(str(error)) from error


def _safe_relative(value: object) -> str:
    try:
        return plan.safe_relative(value)
    except plan.MatchedERMHeldInScorePlanError as error:
        raise MatchedERMHeldInScoreError(str(error)) from error


@dataclass(frozen=True)
class CSWGComparatorEvidence:
    """Held descriptor proof of the accepted same-surface CS-WG receipt."""

    anchor: plan.SameInputAnchor
    root_identity: tuple[int, int]
    named_chain_identities: tuple[tuple[str, int, int], ...]
    input_authority: Mapping[str, object] = field(repr=False, compare=False)
    score: Mapping[str, object] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        inputs = dict(self.input_authority)
        scored = dict(self.score)
        _require(isinstance(self.anchor, plan.SameInputAnchor)
                 and len(self.root_identity) == 2 and all(type(value) is int and value >= 0 for value in self.root_identity)
                 and self.named_chain_identities
                 and _sha(_json_bytes(inputs)) == self.anchor.comparator_input_authority_sha256
                 and _sha(_json_bytes(scored)) == self.anchor.comparator_score_sha256,
                 "CS-WG matched-ERM held-in score comparator receipt SHA drift")
        _validate_same_input_fragment(inputs, self.anchor)
        _require(scored.get("schema") == "cross_session_worst_group_m1_fold20120924_heldin_score_payload_v1"
                 and scored.get("input_authority") == inputs
                 and scored.get("n_windows") == self.anchor.n_windows
                 and scored.get("target_sha256") == self.anchor.target_sha256
                 and type(scored.get("governing_r2")) in {int, float}
                 and math.isfinite(float(scored["governing_r2"])),
                 "CS-WG matched-ERM held-in score comparator score/input drift")
        object.__setattr__(self, "input_authority", inputs)
        object.__setattr__(self, "score", scored)

    def payload(self) -> dict[str, object]:
        body = {
            "schema": "cross_session_worst_group_m1_cswg_same_input_comparator_evidence_v1",
            "anchor": self.anchor.payload(),
            "root_identity": list(self.root_identity),
            "named_chain_identities": [list(item) for item in self.named_chain_identities],
            "input_authority_sha256": self.anchor.comparator_input_authority_sha256,
            "score_sha256": self.anchor.comparator_score_sha256,
            "n_windows": self.anchor.n_windows,
            "same_input_fields_validated": [
                "target_descriptor.sha256", "target_descriptor.byte_count", "calibration_sha256",
                "calibration_shape_per_row", "ordered_window_start_sha256",
                "target_reader_native_evidence.ordered_query_identity_sha256",
                "target_reader_native_evidence.ordered_target_evalmask_sha256",
                "target_reader_native_evidence.reader_recipe_sha256", "target_sha256",
            ],
        }
        return {**body, "binding_sha256": _sha(_json_bytes(body))}

    @property
    def sha256(self) -> str:
        return str(self.payload()["binding_sha256"])


def load_cswg_same_input_comparator_evidence(
    root: Path, *, anchor: plan.SameInputAnchor = plan.DEFAULT_SAME_INPUT_ANCHOR,
) -> CSWGComparatorEvidence:
    """Hold and validate only the accepted CS-WG input/score receipt graph."""
    _require(isinstance(anchor, plan.SameInputAnchor),
             "CS-WG matched-ERM held-in score comparator anchor type drift")
    try:
        _base_fd, opened, identities = shared._open_held_result_directory(
            Path(root), anchor.comparator_root_relative,
        )
    except shared.HeldInScoreError as error:
        raise MatchedERMHeldInScoreError("CS-WG matched-ERM held-in score comparator directory drift") from error
    result_fd = opened[-1]
    try:
        expected_names = shared._success_names(terminal=True)
        _require(tuple(sorted(os.listdir(result_fd))) == tuple(sorted(expected_names)),
                 "CS-WG matched-ERM held-in score comparator exact leaf topology drift")
        inputs, input_sha = shared._read_json_pair(
            result_fd, "input_authority.json", expected_sha256=anchor.comparator_input_authority_sha256,
        )
        scored, score_sha = shared._read_json_pair(
            result_fd, "score.json", expected_sha256=anchor.comparator_score_sha256,
        )
        terminal, _terminal_sha = shared._read_json_pair(result_fd, "terminal.json")
        _require(input_sha == anchor.comparator_input_authority_sha256
                 and score_sha == anchor.comparator_score_sha256
                 and terminal.get("status") == "COMPLETE_DESCRIPTIVE_HELDIN_R2"
                 and terminal.get("input_authority_sha256") == input_sha
                 and terminal.get("score_sha256") == score_sha,
                 "CS-WG matched-ERM held-in score comparator terminal linkage drift")
        _require(tuple(sorted(os.listdir(result_fd))) == tuple(sorted(expected_names))
                 and shared._named_chain_identities(Path(root), anchor.comparator_root_relative) == identities,
                 "CS-WG matched-ERM held-in score comparator graph changed during read")
        info = os.fstat(result_fd)
        return CSWGComparatorEvidence(
            anchor, shared._directory_identity(info), identities, inputs, scored,
        )
    except shared.HeldInScoreError as error:
        raise MatchedERMHeldInScoreError("CS-WG matched-ERM held-in score comparator receipt drift") from error
    finally:
        shared._close_held_chain(opened)


@dataclass(frozen=True)
class MatchedERMCompletedFullGraph:
    """Typed wrapper around the shared descriptor-held full graph."""

    binding: plan.MatchedERMFullBinding
    shared_graph: shared.CompletedFullGraph = field(repr=False, compare=False)
    comparator_evidence: CSWGComparatorEvidence | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        expected = self.binding.completed_full_expectation()
        graph = self.shared_graph
        _require(isinstance(self.binding, plan.MatchedERMFullBinding)
                 and isinstance(graph, shared.CompletedFullGraph)
                 and (self.comparator_evidence is None
                      or isinstance(self.comparator_evidence, CSWGComparatorEvidence))
                 and graph.expectation == expected
                 and graph.body_sha256.get("attempt.json") == self.binding.attempt_sha256
                 and graph.body_sha256.get("launch.json") == self.binding.launch_sha256
                 and graph.body_sha256.get("source_authority.json") == self.binding.source_authority_sha256
                 and graph.body_sha256.get("training.json") == self.binding.training_sha256
                 and graph.body_sha256.get("checkpoint_manifest.json") == self.binding.checkpoint_manifest_sha256
                 and graph.body_sha256.get("checkpoint_best_source_train_loss.pt") == self.binding.best_checkpoint_sha256
                 and graph.body_sha256.get("checkpoint_last.pt") == self.binding.last_checkpoint_sha256
                 and graph.body_sha256.get("terminal.json") == self.binding.terminal_sha256,
                 "CS-WG matched-ERM held-in score full graph/binding drift")

    def payload(self) -> dict[str, object]:
        body = {
            "schema": "cross_session_worst_group_m1_matched_erm_completed_full_graph_v1",
            "producer_binding": self.binding.payload(),
            "shared_completed_full_graph": self.shared_graph.payload(),
            "selected_checkpoint_role": "best_source_train_loss",
            "source_only": True,
            "swa_enabled": False,
            "failure_absent": True,
        }
        if self.comparator_evidence is not None:
            body["cswg_comparator_evidence"] = self.comparator_evidence.payload()
        return {**body, "binding_sha256": _sha(_json_bytes(body))}

    @property
    def sha256(self) -> str:
        return str(self.payload()["binding_sha256"])


def load_completed_matched_erm_full_graph(
    root: Path, *, binding: plan.MatchedERMFullBinding,
) -> MatchedERMCompletedFullGraph:
    """Descriptor-validate the future exact 56-leaf producer graph.

    Calling this before a completed binding exists is impossible by type and
    construction.  It delegates no-follow leaf/sidecar/topology handling to
    the reviewed shared loader and then pins all extra ERM facts.
    """
    _require(isinstance(binding, plan.MatchedERMFullBinding),
             "CS-WG matched-ERM held-in score producer binding type drift")
    try:
        graph = shared.load_completed_full_graph(
            Path(root), expectation=binding.completed_full_expectation(),
        )
    except shared.HeldInScoreError as error:
        raise MatchedERMHeldInScoreError("CS-WG matched-ERM held-in score producer graph drift") from error
    return MatchedERMCompletedFullGraph(binding, graph)


def load_matched_erm_score_graph(
    root: Path, *, binding: plan.MatchedERMFullBinding,
    anchor: plan.SameInputAnchor = plan.DEFAULT_SAME_INPUT_ANCHOR,
) -> MatchedERMCompletedFullGraph:
    """Bind both the future ERM producer and accepted CS-WG input receipt."""
    full = load_completed_matched_erm_full_graph(Path(root), binding=binding)
    comparator = load_cswg_same_input_comparator_evidence(Path(root), anchor=anchor)
    return MatchedERMCompletedFullGraph(binding, full.shared_graph, comparator)


def read_selected_checkpoint_bytes(root: Path, graph: MatchedERMCompletedFullGraph) -> bytes:
    _require(isinstance(graph, MatchedERMCompletedFullGraph),
             "CS-WG matched-ERM held-in score selected graph type drift")
    try:
        return shared.read_selected_checkpoint_bytes(Path(root), graph.shared_graph)
    except shared.HeldInScoreError as error:
        raise MatchedERMHeldInScoreError("CS-WG matched-ERM held-in score selected checkpoint graph drift") from error


def _validate_same_input_fragment(value: object, anchor: plan.SameInputAnchor) -> dict[str, object]:
    _require(isinstance(value, Mapping) and isinstance(anchor, plan.SameInputAnchor),
             "CS-WG matched-ERM held-in score same-input fragment type drift")
    item = dict(value)
    descriptor = item.get("target_descriptor")
    native = item.get("target_reader_native_evidence")
    _require(isinstance(descriptor, Mapping) and isinstance(native, Mapping)
             and item.get("n_windows") == anchor.n_windows
             and item.get("model_input_shape") == [plan.MODEL_SHAPE["window"], plan.MODEL_SHAPE["units"]]
             and item.get("calibration_shape_per_row") == plan.MODEL_SHAPE["calibration_shape_per_row"]
             and descriptor.get("sha256") == anchor.target_descriptor_sha256
             and descriptor.get("byte_count") == anchor.target_descriptor_byte_count
             and item.get("calibration_sha256") == anchor.calibration_sha256
             and item.get("ordered_window_start_sha256") == anchor.ordered_window_start_sha256
             and item.get("target_sha256") == anchor.target_sha256
             and native.get("ordered_query_identity_sha256") == anchor.ordered_query_identity_sha256
             and native.get("ordered_target_evalmask_sha256") == anchor.ordered_target_evalmask_sha256
             and native.get("reader_recipe_sha256") == anchor.reader_recipe_sha256,
             "CS-WG matched-ERM held-in score same-input evidence drift")
    return item


def _input_authority_payload(
    identity: plan.MatchedERMScoreIdentity, graph: MatchedERMCompletedFullGraph,
    prepared: Mapping[str, object],
) -> dict[str, object]:
    protected = {
        "schema", "identity_sha256", "completed_matched_erm_full_graph_sha256", "target_session",
        "selected_checkpoint_role", "metric", "last_bin_only", "target_metric_only",
        "target_optimizer_backward_update", "target_labels_used_only_for_metric", "same_input_anchor",
        "cswg_comparator_evidence_sha256",
    }
    _require(isinstance(identity, plan.MatchedERMScoreIdentity)
             and isinstance(graph, MatchedERMCompletedFullGraph)
             and isinstance(graph.comparator_evidence, CSWGComparatorEvidence)
             and graph.comparator_evidence.anchor == identity.same_input_anchor
             and isinstance(prepared, Mapping) and not (set(prepared) & protected),
             "CS-WG matched-ERM held-in score protected input authority drift")
    _validate_same_input_fragment(prepared, identity.same_input_anchor)
    result = {
        "schema": "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_input_authority_v1",
        "identity_sha256": identity.sha256,
        "completed_matched_erm_full_graph_sha256": graph.sha256,
        "target_session": plan.TARGET_SESSION,
        "selected_checkpoint_role": "best_source_train_loss",
        "metric": plan.METRIC_LABEL,
        "last_bin_only": True,
        "target_metric_only": True,
        "target_labels_used_only_for_metric": True,
        "target_optimizer_backward_update": 0,
        "same_input_anchor": identity.same_input_anchor.payload(),
        "cswg_comparator_evidence_sha256": graph.comparator_evidence.sha256,
    }
    result.update(dict(prepared))
    return _validate_input_authority(result, identity, graph)


def _validate_input_authority(
    value: object, identity: plan.MatchedERMScoreIdentity, graph: MatchedERMCompletedFullGraph,
) -> dict[str, object]:
    _require(isinstance(value, Mapping) and isinstance(identity, plan.MatchedERMScoreIdentity)
             and isinstance(graph, MatchedERMCompletedFullGraph),
             "CS-WG matched-ERM held-in score input authority type drift")
    item = dict(value)
    _require(item.get("schema") == "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_input_authority_v1"
             and item.get("identity_sha256") == identity.sha256
             and item.get("completed_matched_erm_full_graph_sha256") == graph.sha256
             and item.get("target_session") == plan.TARGET_SESSION
             and item.get("selected_checkpoint_role") == "best_source_train_loss"
             and item.get("metric") == plan.METRIC_LABEL and item.get("last_bin_only") is True
             and item.get("target_metric_only") is True and item.get("target_labels_used_only_for_metric") is True
             and item.get("target_optimizer_backward_update") == 0
             and item.get("same_input_anchor") == identity.same_input_anchor.payload()
             and isinstance(graph.comparator_evidence, CSWGComparatorEvidence)
             and item.get("cswg_comparator_evidence_sha256") == graph.comparator_evidence.sha256,
             "CS-WG matched-ERM held-in score input authority fixed fields drift")
    _validate_same_input_fragment(item, identity.same_input_anchor)
    return item


def _score_payload(
    identity: plan.MatchedERMScoreIdentity, graph: MatchedERMCompletedFullGraph,
    input_authority: Mapping[str, object], *, forward_count: int, n_windows: int,
    prediction_sha256: str, target_sha256: str, governing_r2: float,
    model_state_before_sha256: str, model_state_after_sha256: str,
) -> dict[str, object]:
    inputs = _validate_input_authority(input_authority, identity, graph)
    return {
        "schema": "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_payload_v1",
        "identity_sha256": identity.sha256,
        "completed_matched_erm_full_graph_sha256": graph.sha256,
        "input_authority_sha256": _sha(_json_bytes(inputs)),
        "target_session": plan.TARGET_SESSION,
        "producer_system": "MATCHED_ERM",
        "producer_objective_lambda": 0.0,
        "producer_objective_tau": 0.01,
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
        "n_windows": n_windows,
        "governing_r2": governing_r2,
        "prediction_sha256": _require_sha(prediction_sha256, "score prediction"),
        "target_sha256": _require_sha(target_sha256, "score target"),
        "model_state_before_sha256": _require_sha(model_state_before_sha256, "score state before"),
        "model_state_after_sha256": _require_sha(model_state_after_sha256, "score state after"),
        "same_input_anchor": identity.same_input_anchor.payload(),
        "comparator_cswg_input_authority_sha256": identity.same_input_anchor.comparator_input_authority_sha256,
        "comparator_cswg_score_sha256": identity.same_input_anchor.comparator_score_sha256,
        "cswg_comparator_evidence_sha256": graph.comparator_evidence.sha256
        if isinstance(graph.comparator_evidence, CSWGComparatorEvidence) else None,
    }


def _validate_score_payload(
    value: object, identity: plan.MatchedERMScoreIdentity, graph: MatchedERMCompletedFullGraph,
    input_authority_sha256: str,
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG matched-ERM held-in score payload must be a mapping")
    item = dict(value)
    input_authority = _validate_input_authority(item.get("input_authority"), identity, graph)
    numeric = item.get("governing_r2")
    required_sha = ("prediction_sha256", "target_sha256", "model_state_before_sha256", "model_state_after_sha256")
    _require(item.get("schema") == "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_payload_v1"
             and item.get("identity_sha256") == identity.sha256
             and item.get("completed_matched_erm_full_graph_sha256") == graph.sha256
             and item.get("input_authority_sha256") == input_authority_sha256
             and item.get("target_session") == plan.TARGET_SESSION
             and item.get("producer_system") == "MATCHED_ERM"
             and item.get("producer_objective_lambda") == 0.0 and item.get("producer_objective_tau") == 0.01
             and item.get("selected_checkpoint_role") == "best_source_train_loss"
             and item.get("metric") == plan.METRIC_LABEL and item.get("last_bin_only") is True
             and item.get("eval_mode") is True and item.get("no_grad") is True
             and item.get("dynamic_dropout_disabled") is True and item.get("target_metric_only") is True
             and item.get("target_labels_used_only_for_metric") is True
             and item.get("target_optimizer_steps") == 0 and item.get("target_backward_calls") == 0
             and item.get("target_update_calls") == 0
             and type(item.get("full_system_forward_count")) is int and item["full_system_forward_count"] > 0
             and item.get("n_windows") == identity.same_input_anchor.n_windows == input_authority.get("n_windows")
             and type(numeric) in {int, float} and math.isfinite(float(numeric))
             and all(_require_sha(item.get(name), f"score {name}") for name in required_sha)
             and item.get("model_state_before_sha256") == item.get("model_state_after_sha256")
             and item.get("target_sha256") == input_authority.get("target_sha256")
             and item.get("same_input_anchor") == identity.same_input_anchor.payload()
             and item.get("comparator_cswg_input_authority_sha256")
                 == identity.same_input_anchor.comparator_input_authority_sha256
             and item.get("comparator_cswg_score_sha256") == identity.same_input_anchor.comparator_score_sha256,
             "CS-WG matched-ERM held-in score metric/state/update drift")
    _require(isinstance(graph.comparator_evidence, CSWGComparatorEvidence)
             and item.get("cswg_comparator_evidence_sha256") == graph.comparator_evidence.sha256,
             "CS-WG matched-ERM held-in score metric/state/update drift")
    return item


@dataclass(frozen=True)
class MatchedERMPhysicalCodec:
    """Receipt codec injected into the reviewed V1 physical evaluator."""

    def validate_identity_graph(self, identity: object, graph: object) -> None:
        _require(isinstance(identity, plan.MatchedERMScoreIdentity)
                 and isinstance(graph, MatchedERMCompletedFullGraph)
                 and graph.binding == identity.producer_binding,
                 "CS-WG matched-ERM held-in score physical identity/graph drift")
        _require(isinstance(graph.comparator_evidence, CSWGComparatorEvidence)
                 and graph.comparator_evidence.anchor == identity.same_input_anchor,
                 "CS-WG matched-ERM held-in score physical identity/graph drift")

    def shared_checkpoint_graph(self, graph: object) -> shared.CompletedFullGraph:
        _require(isinstance(graph, MatchedERMCompletedFullGraph),
                 "CS-WG matched-ERM held-in score selected graph wrapper drift")
        return graph.shared_graph

    def launch_payload(
        self, identity: object, graph: object, *, device: str,
    ) -> Mapping[str, object]:
        self.validate_identity_graph(identity, graph)
        return {
            "schema": "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_physical_launch_v1",
            "parser": "FalconDataModule.prepare_session_data",
            "dataset": "FalconDataset",
            "model": "SpintModel",
            "device": device,
            "producer_system": "MATCHED_ERM",
            "target_opened": False,
            "checkpoint_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "target_metric_only": True,
        }

    def input_authority_payload(
        self, identity: object, graph: object, prepared: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.validate_identity_graph(identity, graph)
        return _input_authority_payload(identity, graph, prepared)

    def score_payload(
        self, identity: object, graph: object, input_authority: Mapping[str, object], **values: object,
    ) -> Mapping[str, object]:
        self.validate_identity_graph(identity, graph)
        required = {
            "forward_count", "n_windows", "prediction_sha256", "target_sha256", "governing_r2",
            "model_state_before_sha256", "model_state_after_sha256",
        }
        _require(set(values) == required, "CS-WG matched-ERM held-in score physical values topology drift")
        return _score_payload(identity, graph, input_authority, **values)  # type: ignore[arg-type]


def validate_identity_current(root: Path, identity: plan.MatchedERMScoreIdentity) -> None:
    _require(isinstance(identity, plan.MatchedERMScoreIdentity)
             and identity.spec == plan.MatchedERMScoreSpec(),
             "CS-WG matched-ERM held-in score identity/spec drift")
    try:
        plan.validate_current_closure(Path(root), identity.closure)
    except plan.MatchedERMHeldInScorePlanError as error:
        raise MatchedERMHeldInScoreError("CS-WG matched-ERM held-in score current closure drift") from error
    try:
        metadata = v1.load_m1_metadata_manifest_authority(Path(root))
    except v1.SourceLifecycleError as error:
        raise MatchedERMHeldInScoreError("CS-WG matched-ERM held-in score sealed target metadata authority drift") from error
    _require(metadata.get("body_sha256") == v1.M1_METADATA_MANIFEST_SHA256,
             "CS-WG matched-ERM held-in score target metadata body binding drift")


def assert_prospective_score_root_fresh(root: Path, spec: plan.MatchedERMScoreSpec) -> None:
    candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise MatchedERMHeldInScoreError("CS-WG matched-ERM held-in score prospective root inspection failed") from error
    raise MatchedERMHeldInScoreError("CS-WG matched-ERM held-in score canonical root already exists")


class _MatchedERMScoreReviewSeal:
    pass


_MATCHED_ERM_SCORE_REVIEW_SEAL = _MatchedERMScoreReviewSeal()


@dataclass(frozen=True)
class MatchedERMScoreCapability:
    identity_sha256: str
    completed_full_graph_sha256: str
    target_source_root: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha(self.identity_sha256, "matched-ERM score capability identity")
        _require_sha(self.completed_full_graph_sha256, "matched-ERM score capability graph")
        _require(isinstance(self.target_source_root, str) and Path(self.target_source_root).is_absolute()
                 and str(Path(self.target_source_root)) == self.target_source_root
                 and self._seal is _MATCHED_ERM_SCORE_REVIEW_SEAL,
                 "CS-WG matched-ERM held-in score capability drift")


def issue_root_reviewed_matched_erm_score_capability(
    root: Path, *, identity: plan.MatchedERMScoreIdentity, source_root: Path, review_seal: object,
) -> MatchedERMScoreCapability:
    _require(review_seal is _MATCHED_ERM_SCORE_REVIEW_SEAL,
             "only the root reviewer may issue CS-WG matched-ERM score capability")
    validate_identity_current(Path(root), identity)
    graph = load_matched_erm_score_graph(
        Path(root), binding=identity.producer_binding, anchor=identity.same_input_anchor,
    )
    assert_prospective_score_root_fresh(Path(root), identity.spec)
    _require(Path(source_root).is_absolute(),
             "CS-WG matched-ERM held-in score source root must be lexical absolute")
    return MatchedERMScoreCapability(identity.sha256, graph.sha256, str(Path(source_root)), _MATCHED_ERM_SCORE_REVIEW_SEAL)


def _require_capability(capability: object, identity: plan.MatchedERMScoreIdentity) -> MatchedERMScoreCapability:
    _require(isinstance(capability, MatchedERMScoreCapability)
             and capability._seal is _MATCHED_ERM_SCORE_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "CS-WG matched-ERM held-in score needs exact root-reviewed capability")
    return capability


def _attempt_payload(identity: plan.MatchedERMScoreIdentity, graph: MatchedERMCompletedFullGraph) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_attempt_v1",
        "cell": plan.CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "completed_matched_erm_full_graph": graph.payload(),
        **shared.ScoreProgress().payload(),
    }


def _launch_payload(
    identity: plan.MatchedERMScoreIdentity, graph: MatchedERMCompletedFullGraph,
    attempt_sha256: str, backend: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_launch_v1",
        "cell": plan.CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "completed_matched_erm_full_graph_sha256": graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "launch attempt"),
        "backend": dict(backend),
        **shared.ScoreProgress().payload(),
    }


def _terminal_payload(
    identity: plan.MatchedERMScoreIdentity, graph: MatchedERMCompletedFullGraph, attempt_sha256: str,
    launch_sha256: str, input_authority_sha256: str, score_sha256: str, score_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_terminal_v1",
        "cell": plan.CELL,
        "status": "COMPLETE_DESCRIPTIVE_MATCHED_ERM_HELDIN_R2",
        "identity": identity.payload(),
        "completed_matched_erm_full_graph_sha256": graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "terminal launch"),
        "input_authority_sha256": _require_sha(input_authority_sha256, "terminal input authority"),
        "score_sha256": _require_sha(score_sha256, "terminal score"),
        "governing_r2": score_payload["governing_r2"],
        "n_windows": score_payload["n_windows"],
        "prediction_sha256": score_payload["prediction_sha256"],
        "target_sha256": score_payload["target_sha256"],
        "selected_checkpoint_role": "best_source_train_loss",
        "producer_system": "MATCHED_ERM",
        "same_input_anchor": identity.same_input_anchor.payload(),
        "comparator_cswg_input_authority_sha256": identity.same_input_anchor.comparator_input_authority_sha256,
        "comparator_cswg_score_sha256": identity.same_input_anchor.comparator_score_sha256,
        "cswg_comparator_evidence_sha256": graph.comparator_evidence.sha256
        if isinstance(graph.comparator_evidence, CSWGComparatorEvidence) else None,
        "formal_benchmark_verdict": False,
        "target_optimizer_backward_update": 0,
        "target_metric_only": True,
    }


def _failure_payload(
    identity: plan.MatchedERMScoreIdentity, graph: MatchedERMCompletedFullGraph, attempt_sha256: str,
    launch_sha256: str | None, input_authority_sha256: str | None, score_sha256: str | None,
    progress: shared.ScoreProgress, error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_failure_v1",
        "cell": plan.CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "completed_matched_erm_full_graph_sha256": graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "failure attempt"),
        "launch_sha256": None if launch_sha256 is None else _require_sha(launch_sha256, "failure launch"),
        "input_authority_sha256": None if input_authority_sha256 is None else _require_sha(
            input_authority_sha256, "failure input authority",
        ),
        "score_sha256": None if score_sha256 is None else _require_sha(score_sha256, "failure score"),
        "progress": progress.payload(),
        "error_class": type(error).__name__,
        "error_sha256": _sha(repr(error).encode("utf-8")),
        "comparator_cswg_input_authority_sha256": identity.same_input_anchor.comparator_input_authority_sha256,
        "comparator_cswg_score_sha256": identity.same_input_anchor.comparator_score_sha256,
        "cswg_comparator_evidence_sha256": graph.comparator_evidence.sha256
        if isinstance(graph.comparator_evidence, CSWGComparatorEvidence) else None,
        "terminal_published": False,
        "target_metric_only": True,
        "target_optimizer_backward_update": 0,
    }


def _success_names(terminal: bool) -> tuple[str, ...]:
    bodies = ["attempt.json", "launch.json", "input_authority.json", "score.json"]
    if terminal:
        bodies.append("terminal.json")
    return tuple(item for body in bodies for item in (body, f"{body}.sha256"))


def _revalidate_published_score_graph(
    artifact: v1.ImmutableArtifactRoot, identity: plan.MatchedERMScoreIdentity,
    graph: MatchedERMCompletedFullGraph, *, attempt_sha256: str, launch_sha256: str,
    input_authority_sha256: str, score_sha256: str,
) -> dict[str, object]:
    artifact.validate_live(expected_names=_success_names(False))
    attempt = artifact.read_json_pair("attempt.json", expected_sha256=attempt_sha256)
    launch = artifact.read_json_pair("launch.json", expected_sha256=launch_sha256)
    inputs = artifact.read_json_pair("input_authority.json", expected_sha256=input_authority_sha256)
    scored = artifact.read_json_pair("score.json", expected_sha256=score_sha256)
    _require(attempt == _attempt_payload(identity, graph)
             and launch.get("attempt_sha256") == attempt_sha256
             and launch.get("completed_matched_erm_full_graph_sha256") == graph.sha256,
             "CS-WG matched-ERM held-in score published attempt/launch drift")
    _validate_input_authority(inputs, identity, graph)
    _require(scored.get("input_authority") == inputs,
             "CS-WG matched-ERM held-in score published input snapshot drift")
    return _validate_score_payload(scored, identity, graph, input_authority_sha256)


def _pre_reserve(
    root: Path, identity: object, capability: object, backend: object,
) -> MatchedERMCompletedFullGraph:
    _require(isinstance(identity, plan.MatchedERMScoreIdentity),
             "CS-WG matched-ERM held-in score identity type drift")
    cap = _require_capability(capability, identity)
    validate_identity_current(Path(root), identity)
    graph = load_matched_erm_score_graph(
        Path(root), binding=identity.producer_binding, anchor=identity.same_input_anchor,
    )
    _require(graph.sha256 == cap.completed_full_graph_sha256,
             "CS-WG matched-ERM held-in score capability/producer graph drift before reserve")
    source_root = Path(getattr(backend, "source_root", ""))
    _require(source_root.is_absolute() and str(source_root) == cap.target_source_root,
             "CS-WG matched-ERM held-in score capability/backend target source-root drift before reserve")
    assert_prospective_score_root_fresh(Path(root), identity.spec)
    return graph


def _revalidate_before_terminal(
    root: Path, identity: object, capability: object, graph: object, artifact: v1.ImmutableArtifactRoot,
    attempt_sha256: str, launch_sha256: str, input_authority_sha256: str, score_sha256: str,
) -> Mapping[str, object]:
    _require(isinstance(identity, plan.MatchedERMScoreIdentity)
             and isinstance(graph, MatchedERMCompletedFullGraph),
             "CS-WG matched-ERM held-in score final identity/graph type drift")
    cap = _require_capability(capability, identity)
    validate_identity_current(Path(root), identity)
    graph_now = load_matched_erm_score_graph(
        Path(root), binding=identity.producer_binding, anchor=identity.same_input_anchor,
    )
    _require(graph_now.sha256 == graph.sha256 == cap.completed_full_graph_sha256,
             "CS-WG matched-ERM held-in score immutable producer graph drifted during evaluation")
    return _revalidate_published_score_graph(
        artifact, identity, graph, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
        input_authority_sha256=input_authority_sha256, score_sha256=score_sha256,
    )


def execute_reviewed_matched_erm_heldin_score(
    root: Path, *, identity: plan.MatchedERMScoreIdentity, capability: object, backend: object,
) -> shared.ScoreLifecycleResult:
    """Root-only execution; public CLI deliberately cannot invoke this path."""
    hooks = shared.ProfiledHeldInScoreLifecycleHooks(
        label="matched_erm_fold20120924_v1",
        pre_reserve=_pre_reserve,
        spec_for_identity=lambda observed: observed.spec if isinstance(observed, plan.MatchedERMScoreIdentity) else None,
        attempt_payload=lambda observed, graph: _attempt_payload(observed, graph),
        launch_payload=lambda observed, graph, attempt_sha, backend_payload: _launch_payload(
            observed, graph, attempt_sha, backend_payload,
        ),
        input_authority_payload=lambda observed, graph, prepared: _input_authority_payload(observed, graph, prepared),
        validate_score_payload=lambda value, observed, graph, input_sha: _validate_score_payload(
            value, observed, graph, input_sha,
        ),
        revalidate_before_terminal=_revalidate_before_terminal,
        terminal_payload=lambda observed, graph, attempt_sha, launch_sha, input_sha, score_sha, checked: _terminal_payload(
            observed, graph, attempt_sha, launch_sha, input_sha, score_sha, checked,
        ),
        failure_payload=lambda observed, graph, attempt_sha, launch_sha, input_sha, score_sha, progress, error: _failure_payload(
            observed, graph, attempt_sha, launch_sha, input_sha, score_sha, progress, error,
        ),
        success_names=_success_names,
    )
    return shared.execute_profiled_heldin_score(
        Path(root), identity=identity, capability=capability, backend=backend, hooks=hooks,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    return plan.dry_plan(root)


__all__ = (
    "MatchedERMHeldInScoreError", "CSWGComparatorEvidence", "MatchedERMCompletedFullGraph", "MatchedERMPhysicalCodec",
    "MatchedERMScoreCapability", "load_cswg_same_input_comparator_evidence",
    "load_completed_matched_erm_full_graph", "load_matched_erm_score_graph", "read_selected_checkpoint_bytes",
    "validate_identity_current", "assert_prospective_score_root_fresh",
    "issue_root_reviewed_matched_erm_score_capability", "execute_reviewed_matched_erm_heldin_score", "dry_plan",
)
