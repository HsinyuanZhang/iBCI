"""Receipt/lifecycle composition for Precision-Aware CDM-D V2 matched score.

The only lifecycle engine used here is V1's ``run_profiled_score_lifecycle``.
This module owns the new typed receipt codec and held-FD V8 predecessor loader;
it deliberately does not copy a parser, forward loop, metric, or lifecycle.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import score as v5score
from src.causal_dual_memory_cell_d_score_v8 import plan as v8plan
from src.causal_dual_memory_cell_d_score_v8 import score as v8score

from . import plan


AUTHORITY_TOPOLOGY = v1score.AUTHORITY_TOPOLOGY
SCORE_TOPOLOGY = v1score.SCORE_TOPOLOGY
_PRECISION_REASON = "precision_credible_region"
_V5_REASONS = frozenset((*v5score._UPDATE_REJECTION_REASONS, _PRECISION_REASON))


class PrecisionMatchedScoreError(v1score.ScoreError):
    """Fail closed for V8 lineage, Precision-V2 receipt, or lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionMatchedScoreError(message)


def _json(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _digest(value: object) -> str:
    return plan.sha256_bytes(_json(value))


def _sha(value: object, label: str) -> str:
    try:
        return plan.require_sha(value, label)
    except plan.PrecisionMatchedScorePlanError as error:
        raise PrecisionMatchedScoreError(str(error)) from error


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PrecisionMatchedScoreError(f"{label} must be finite")
    return float(value)


def _identity_payload(identity: plan.ScoreIdentity) -> dict[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise PrecisionMatchedScoreError("Precision matched score requires its exact typed identity")
    return identity.payload()


@dataclass(frozen=True)
class V8PredecessorBinding:
    """A held exact V8 graph plus semantic V5 source-gate proof."""

    directory_identity: tuple[int, int]
    v8_identity: Mapping[str, object]
    input_payload: Mapping[str, object]
    score_payload: Mapping[str, object]
    terminal_payload: Mapping[str, object]
    source_gate: v1score.SourceGateBinding
    sealed_cells: Mapping[str, Mapping[str, object]]

    def payload(self) -> dict[str, object]:
        expected = {f"m{budget}:{surface}" for budget in (30, 10, 4) for surface in plan.SURFACES}
        _require(set(self.sealed_cells) == expected, "V8 binding sealed-cell topology drift")
        witness = {
            "schema": "precision_aware_cdmd_v8_binding_witness_v1",
            "contract": plan.V8_PREDECESSOR.payload(),
            "directory_identity": [int(self.directory_identity[0]), int(self.directory_identity[1])],
            "v8_identity_sha256": _digest(self.v8_identity),
            "v8_source_gate_binding": self.source_gate.payload(),
            "input_records_sha256": _digest(self.input_payload["records"]),
            "sealed_cell_sha256s": {key: _digest(self.sealed_cells[key]) for key in sorted(expected)},
        }
        result = {**witness, "binding_sha256": _digest(witness)}
        try:
            return plan.validate_v8_binding_witness(result)
        except plan.PrecisionMatchedScorePlanError as error:
            raise PrecisionMatchedScoreError(str(error)) from error

    def sealed_cell(self, *, budget: int, surface: str) -> dict[str, object]:
        key = f"m{budget}:{surface}"
        if key not in self.sealed_cells:
            raise PrecisionMatchedScoreError("V8 reused sealed cell is absent")
        return dict(self.sealed_cells[key])


def _v8_identity_from_attempt(attempt: Mapping[str, object]) -> v8plan.ScoreIdentity:
    identity_payload = attempt.get("identity")
    if not isinstance(identity_payload, Mapping):
        raise PrecisionMatchedScoreError("V8 attempt identity absent")
    closure = identity_payload.get("closure")
    selected = identity_payload.get("selected_device_profile")
    if not isinstance(closure, Mapping) or not isinstance(selected, Mapping):
        raise PrecisionMatchedScoreError("V8 attempt closure/device identity absent")
    try:
        identity = v8plan.ScoreIdentity(closure=closure, selected_device_profile=selected)
    except (TypeError, v8plan.V8PlanError) as error:
        raise PrecisionMatchedScoreError("V8 historical identity reconstruction failed") from error
    if identity.payload() != identity_payload:
        raise PrecisionMatchedScoreError("V8 historical identity canonical drift")
    return identity


def _read_v8_held_graph(root: Path) -> V8PredecessorBinding:
    """Descriptor-read the exact V8 four-pair terminal graph with no rebuild."""
    base = Path(root).absolute()
    directory = base / plan.V8_ROOT_RELATIVE
    try:
        named = v1score._directory_identity(directory)
    except v1score.ScoreError as error:
        raise PrecisionMatchedScoreError(str(error)) from error
    descriptor = -1
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        held = os.fstat(descriptor)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != named:
            raise PrecisionMatchedScoreError("V8 predecessor directory identity drift before read")
        expected = set(plan.V8_EXPECTED_TOPOLOGY) | {f"{name}.sha256" for name in plan.V8_EXPECTED_TOPOLOGY}
        if set(os.listdir(descriptor)) != expected:
            raise PrecisionMatchedScoreError("V8 predecessor exact four-pair topology drift")
        bodies: dict[str, dict[str, object]] = {}
        for name in plan.V8_EXPECTED_TOPOLOGY:
            _body, payload, digest = v1score._read_unbound_0444_pair(descriptor, name)
            if digest != plan.V8_EXPECTED_SHAS[name]:
                raise PrecisionMatchedScoreError(f"V8 predecessor body SHA drift: {name}")
            bodies[name] = dict(payload)
        if v1score._directory_identity(directory) != named:
            raise PrecisionMatchedScoreError("V8 predecessor directory identity drift during read")
    except PrecisionMatchedScoreError:
        raise
    except (OSError, v1score.ScoreError) as error:
        raise PrecisionMatchedScoreError("V8 predecessor descriptor read failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    attempt = bodies["attempt.json"]
    input_payload = bodies["input_authority.json"]
    score_payload = bodies["score.json"]
    terminal = bodies["terminal.json"]
    identity = _v8_identity_from_attempt(attempt)
    try:
        rebuilt_attempt = v8score._attempt_payload(
            identity, attempt["preflight_sha256"], attempt["authorization_sha256"],
        )
    except (KeyError, v8score.V8ScoreError) as error:
        raise PrecisionMatchedScoreError("V8 predecessor attempt semantic drift") from error
    if attempt != rebuilt_attempt:
        raise PrecisionMatchedScoreError("V8 predecessor attempt canonical drift")
    try:
        source_gate = v5score.validate_completed_source_gate(base)
        fixed = v1score.derive_fixed_evaluation_authority(base)
        v1score.validate_input_authority_payload(input_payload, identity=identity, evaluation_authority=fixed)
        v8score.validate_score_payload(
            score_payload, identity, input_payload, plan.V8_EXPECTED_SHAS["input_authority.json"], fixed,
        )
        v8score.validate_terminal_payload(
            terminal, identity, plan.V8_EXPECTED_SHAS["attempt.json"],
            plan.V8_EXPECTED_SHAS["input_authority.json"], score_payload,
            plan.V8_EXPECTED_SHAS["score.json"],
        )
    except (v1score.ScoreError, v5score.V5ScoreError, v8score.V8ScoreError) as error:
        raise PrecisionMatchedScoreError("V8 predecessor semantic graph drift") from error
    if (
        terminal.get("verdict") != plan.V8_TERMINAL_VERDICT
        or identity.payload()["closure"].get("closure_sha256") != plan.V8_HISTORICAL_CLOSURE_SHA256
        or identity.payload()["source_gate"].get("terminal_sha256") != source_gate.payload()["body_sha256s"].get("terminal.json")
    ):
        raise PrecisionMatchedScoreError("V8 predecessor verdict/closure/source-gate drift")
    sealed: dict[str, Mapping[str, object]] = {}
    summaries = score_payload.get("budget_summaries")
    if not isinstance(summaries, Mapping):
        raise PrecisionMatchedScoreError("V8 predecessor budget summaries absent")
    for budget in (30, 10, 4):
        summary = summaries.get(str(budget))
        cells = summary.get("cells") if isinstance(summary, Mapping) else None
        if not isinstance(cells, list):
            raise PrecisionMatchedScoreError(f"V8 predecessor M{budget} cells absent")
        for surface in plan.SURFACES:
            matching = [
                item for item in cells
                if isinstance(item, Mapping) and item.get("surface") == surface
                and item.get("system") == v8plan.SYSTEM_SEALED
            ]
            if len(matching) != 1:
                raise PrecisionMatchedScoreError("V8 predecessor sealed comparator row topology drift")
            sealed[f"m{budget}:{surface}"] = dict(matching[0])
    return V8PredecessorBinding(
        directory_identity=named, v8_identity=identity.payload(), input_payload=input_payload,
        score_payload=score_payload, terminal_payload=terminal, source_gate=source_gate, sealed_cells=sealed,
    )


def validate_completed_v8_predecessor(root: Path) -> V8PredecessorBinding:
    """Public held-FD V8/V5 predecessor validator for issuer and final checks."""
    return _read_v8_held_graph(Path(root))


def build_reviewed_identity(
    root: Path, *, selected_device_profile: Mapping[str, object] | None = None,
) -> plan.ScoreIdentity:
    """Construct an identity only from a freshly held V8 predecessor witness."""
    binding = validate_completed_v8_predecessor(Path(root))
    closure = plan.implementation_closure(Path(root)).payload()
    return plan.ScoreIdentity(closure=closure, v8_binding=binding.payload(), selected_device_profile=selected_device_profile)


def validate_v8_input_equivalence(input_payload: Mapping[str, object], identity: plan.ScoreIdentity) -> dict[str, object]:
    """Require successor records to be byte-identical to V8's input substrate."""
    payload = _identity_payload(identity)
    witness = plan.validate_v8_binding_witness(payload["v8_predecessor_binding"])
    records = input_payload.get("records")
    if not isinstance(records, list) or _digest(records) != witness["input_records_sha256"]:
        raise PrecisionMatchedScoreError("successor input records differ from immutable V8 authority")
    return {
        "v8_input_authority_sha256": plan.V8_EXPECTED_SHAS["input_authority.json"],
        "v8_input_records_sha256": witness["input_records_sha256"],
        "same_input_records_as_v8": True,
    }


def input_payload(authority: v1score.InputAuthority, identity: plan.ScoreIdentity) -> dict[str, object]:
    if not isinstance(authority, v1score.InputAuthority):
        raise PrecisionMatchedScoreError("physical route must return inherited typed input authority")
    base = authority.payload(identity=identity)
    evidence = validate_v8_input_equivalence(base, identity)
    return {
        **base,
        "schema": "precision_aware_cdmd_matched_score_input_authority_v1",
        "v8_input_equivalence": evidence,
    }


def validate_input_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, fixed: v1score.FixedEvaluationAuthority,
) -> dict[str, object]:
    required = {
        "schema", "identity_sha256", "fixed_evaluation_authority_sha256", "records",
        "same_materialized_input_for_both_systems", "target_labels_metric_only", "cache_read_or_write",
        "v8_input_equivalence",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "precision_aware_cdmd_matched_score_input_authority_v1":
        raise PrecisionMatchedScoreError("Precision input authority schema drift")
    base = dict(value)
    evidence = base.pop("v8_input_equivalence")
    base["schema"] = "causal_dual_memory_cell_d_score_input_authority_v1"
    try:
        checked = v1score.validate_input_authority_payload(base, identity=identity, evaluation_authority=fixed)
    except v1score.ScoreError as error:
        raise PrecisionMatchedScoreError(str(error)) from error
    expected = {
        **checked,
        "schema": "precision_aware_cdmd_matched_score_input_authority_v1",
        "v8_input_equivalence": validate_v8_input_equivalence(checked, identity),
    }
    if dict(value) != expected:
        raise PrecisionMatchedScoreError("Precision input authority canonical V8 equivalence drift")
    return expected


def _validate_posterior_payload(value: object, *, budget: int) -> dict[str, object]:
    required = {
        "schema", "budget", "fit_mode", "normalized_lambda", "covariance_estimand",
        "sampling_sandwich_covariance_used", "support_rates", "support_direction_indices", "valid_mask",
        "fixed_ridge_t4", "posterior_covariance_ac", "residual_variance", "design", "normal_matrix",
        "normal_inverse", "residual_degrees_of_freedom", "posterior_variance_floor",
        "groups_sha256_or_null", "pseudo_labels_used", "decoder_token_used",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_v2_support_conditional_posterior_v1"
        or value.get("budget") != budget or value.get("fit_mode") != "fixed_ridge_by_trial"
        or value.get("normalized_lambda") != 0.1
        or value.get("covariance_estimand") != "conditional_gaussian_ridge_posterior_sigma2_A_inverse"
        or value.get("sampling_sandwich_covariance_used") is not False
        or value.get("pseudo_labels_used") is not False or value.get("decoder_token_used") is not False
    ):
        raise PrecisionMatchedScoreError("Precision conditional posterior schema/law drift")
    for key in (
        "support_rates", "support_direction_indices", "valid_mask", "fixed_ridge_t4", "posterior_covariance_ac",
        "residual_variance", "design", "normal_matrix", "normal_inverse",
    ):
        row = value.get(key)
        if not isinstance(row, Mapping) or set(row) != {"dtype", "shape", "sha256"}:
            raise PrecisionMatchedScoreError("Precision posterior array digest schema drift")
        _sha(row.get("sha256"), f"Precision posterior {key}")
    _finite(value.get("residual_degrees_of_freedom"), "Precision posterior degrees of freedom")
    _finite(value.get("posterior_variance_floor"), "Precision posterior variance floor")
    group = value.get("groups_sha256_or_null")
    if group is not None:
        _sha(group, "Precision posterior groups")
    return dict(value)


def _validate_decision_payload(value: object, *, initial_carrier_sha256: str) -> dict[str, object]:
    required = {
        "schema", "accepted", "rejection_reason_or_null", "familywise_alpha", "threshold_law", "threshold",
        "valid_unit_count", "invalid_unit_count", "max_mahalanobis_squared", "min_mahalanobis_squared",
        "reference", "reference_initial_sha256", "proposed_sha256", "delta_ac", "mahalanobis_squared",
        "decoder_token_used",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_v2_familywise_credible_decision_v1"
        or type(value.get("accepted")) is not bool or value.get("familywise_alpha") != 0.05
        or value.get("threshold_law") != "-2*log(0.05/N_valid)"
        or value.get("reference") != "frozen_support_only_initial_fixed_ridge"
        or value.get("decoder_token_used") is not False
    ):
        raise PrecisionMatchedScoreError("Precision decision schema/law drift")
    valid, invalid = value.get("valid_unit_count"), value.get("invalid_unit_count")
    if type(valid) is not int or type(invalid) is not int or valid < 4 or invalid < 0:
        raise PrecisionMatchedScoreError("Precision decision valid-unit topology drift")
    threshold = _finite(value.get("threshold"), "Precision decision threshold")
    expected_threshold = -2.0 * math.log(0.05 / float(valid))
    if abs(threshold - expected_threshold) > 1.0e-12:
        raise PrecisionMatchedScoreError("Precision decision Bonferroni threshold drift")
    maximum = _finite(value.get("max_mahalanobis_squared"), "Precision decision maximum")
    minimum = _finite(value.get("min_mahalanobis_squared"), "Precision decision minimum")
    if minimum < -1.0e-12 or maximum < minimum:
        raise PrecisionMatchedScoreError("Precision decision statistic ordering drift")
    if bool(value["accepted"]) is not (maximum <= threshold):
        raise PrecisionMatchedScoreError("Precision decision accepted/statistic drift")
    expected_reason = None if value["accepted"] else _PRECISION_REASON
    if value.get("rejection_reason_or_null") != expected_reason:
        raise PrecisionMatchedScoreError("Precision decision rejection reason drift")
    for key in ("reference_initial_sha256", "proposed_sha256"):
        _sha(value.get(key), f"Precision decision {key}")
    for key in ("delta_ac", "mahalanobis_squared"):
        row = value.get(key)
        if not isinstance(row, Mapping) or set(row) != {"dtype", "shape", "sha256"}:
            raise PrecisionMatchedScoreError("Precision decision array digest drift")
        _sha(row.get("sha256"), f"Precision decision {key}")
    # The V2 decision uses float64 reference digest, while the inherited base
    # uses the exact float32 carrier digest.  Both are persisted, and neither
    # is silently substituted by a current active carrier digest.
    _sha(initial_carrier_sha256, "Precision base initial carrier")
    return dict(value)


def _validate_transition_rows(rows: Sequence[Mapping[str, object]], *, query_trial_ids: Sequence[str]) -> list[dict[str, object]]:
    expected_fields = {
        "trial_id", "activity_transition_committed", "activity_fifo_changed", "carrier_transition_committed",
        "activity_rejection_reason_or_null", "carrier_rejection_reason_or_null", "state_before_sha256",
        "state_after_sha256", "activity_before_sha256", "activity_after_sha256", "carrier_before_sha256",
        "carrier_after_sha256",
    }
    if len(rows) != len(query_trial_ids):
        raise PrecisionMatchedScoreError("Precision transition/query cardinality drift")
    result: list[dict[str, object]] = []
    for expected_id, raw in zip(query_trial_ids, rows, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields or raw.get("trial_id") != expected_id:
            raise PrecisionMatchedScoreError("Precision transition trial/order schema drift")
        row = dict(raw)
        for key in ("activity_transition_committed", "activity_fifo_changed", "carrier_transition_committed"):
            if type(row[key]) is not bool:
                raise PrecisionMatchedScoreError("Precision transition boolean drift")
        for key in (
            "state_before_sha256", "state_after_sha256", "activity_before_sha256", "activity_after_sha256",
            "carrier_before_sha256", "carrier_after_sha256",
        ):
            _sha(row[key], f"Precision transition {key}")
        for key in ("activity_rejection_reason_or_null", "carrier_rejection_reason_or_null"):
            reason = row[key]
            if reason is not None and (not isinstance(reason, str) or reason not in _V5_REASONS):
                raise PrecisionMatchedScoreError("Precision transition rejection-reason drift")
        if row["activity_transition_committed"] is False:
            if (
                row["activity_fifo_changed"] or row["carrier_transition_committed"]
                or row["activity_rejection_reason_or_null"] is None or row["carrier_rejection_reason_or_null"] is not None
                or row["state_before_sha256"] != row["state_after_sha256"]
                or row["activity_before_sha256"] != row["activity_after_sha256"]
                or row["carrier_before_sha256"] != row["carrier_after_sha256"]
            ):
                raise PrecisionMatchedScoreError("Precision rejected activity transition state drift")
        else:
            if row["activity_rejection_reason_or_null"] is not None or row["state_before_sha256"] == row["state_after_sha256"]:
                raise PrecisionMatchedScoreError("Precision committed activity transition state drift")
            if row["activity_fifo_changed"] is not (row["activity_before_sha256"] != row["activity_after_sha256"]):
                raise PrecisionMatchedScoreError("Precision activity FIFO digest drift")
            if row["carrier_transition_committed"]:
                if row["carrier_rejection_reason_or_null"] is not None or row["carrier_before_sha256"] == row["carrier_after_sha256"]:
                    raise PrecisionMatchedScoreError("Precision carrier commit digest drift")
            elif row["carrier_rejection_reason_or_null"] is None or row["carrier_before_sha256"] != row["carrier_after_sha256"]:
                raise PrecisionMatchedScoreError("Precision carrier rejection digest drift")
        result.append(row)
    if any(result[index]["state_after_sha256"] != result[index + 1]["state_before_sha256"] for index in range(len(result) - 1)):
        raise PrecisionMatchedScoreError("Precision transition state digest chain drift")
    return result


@dataclass(frozen=True)
class PrecisionSessionEvidence:
    """One new M4/M10 causal V2 session row with non-collapsed decisions."""

    base: v1score.SessionScore
    transition_records: tuple[Mapping[str, object], ...]
    conditional_posterior: Mapping[str, object]
    conditional_posterior_sha256: str
    frozen_support_reference_float64_sha256: str
    decisions: tuple[Any | None, ...]

    def payload(self, *, budget: int, query_trial_ids: Sequence[str]) -> dict[str, object]:
        if budget not in plan.BUDGETS:
            raise PrecisionMatchedScoreError("Precision session budget drift")
        base = self.base.payload(budget=budget, system=v1plan.SYSTEM_CDMD)
        rows = _validate_transition_rows(self.transition_records, query_trial_ids=query_trial_ids)
        posterior = _validate_posterior_payload(self.conditional_posterior, budget=budget)
        if self.conditional_posterior_sha256 != _digest(posterior):
            raise PrecisionMatchedScoreError("Precision session posterior digest drift")
        _sha(self.conditional_posterior_sha256, "Precision session posterior")
        frozen_reference = _sha(
            self.frozen_support_reference_float64_sha256,
            "Precision session frozen-support float64 reference",
        )
        if len(self.decisions) != len(rows):
            raise PrecisionMatchedScoreError("Precision session decision cardinality drift")
        decisions: list[dict[str, object] | None] = []
        accepted = 0
        rejected = 0
        for transition, decision in zip(rows, self.decisions, strict=True):
            if decision is None:
                # A carrier commit can only follow an inherited accepted
                # proposal.  The V2 mediator must then have evaluated that
                # proposal against its frozen support reference; omitting the
                # decision would make a live carrier mutation unauditable.
                if transition["carrier_transition_committed"] is True:
                    raise PrecisionMatchedScoreError("Precision carrier commit omitted decision evidence")
                if transition["carrier_rejection_reason_or_null"] == _PRECISION_REASON:
                    raise PrecisionMatchedScoreError("Precision-rejected transition omitted decision evidence")
                decisions.append(None)
                continue
            payload = decision.payload() if hasattr(decision, "payload") and callable(decision.payload) else decision
            checked = _validate_decision_payload(payload, initial_carrier_sha256=base["initial_carrier_sha256"])
            if checked["reference_initial_sha256"] != frozen_reference:
                raise PrecisionMatchedScoreError("Precision decision frozen-support reference digest drift")
            if checked["accepted"]:
                if transition["carrier_transition_committed"] is not True:
                    raise PrecisionMatchedScoreError("accepted Precision decision did not commit carrier")
                accepted += 1
            else:
                if (
                    transition["carrier_transition_committed"] is not False
                    or transition["carrier_rejection_reason_or_null"] != _PRECISION_REASON
                ):
                    raise PrecisionMatchedScoreError("rejected Precision decision did not preserve carrier")
                rejected += 1
            decisions.append(checked)
        carrier_commits = sum(row["carrier_transition_committed"] for row in rows)
        precision_reasons = sum(row["carrier_rejection_reason_or_null"] == _PRECISION_REASON for row in rows)
        if rejected != precision_reasons or carrier_commits != accepted:
            raise PrecisionMatchedScoreError("Precision session decision/commit count drift")
        base.pop("accepted_updates")
        base.pop("rejected_updates")
        return {
            **base,
            "schema": "precision_aware_cdmd_matched_score_session_v1",
            "system": plan.SYSTEM_PRECISION_V2,
            "budget": budget,
            "independent_activity_contract": "IndependentActivityCausalDualMemory__observe_completed_trial__commit_independent",
            "transition_records": rows,
            "conditional_posterior": posterior,
            "conditional_posterior_sha256": self.conditional_posterior_sha256,
            "frozen_support_reference_float64_sha256": frozen_reference,
            "conditional_posterior_decisions": decisions,
            "carrier_transition_committed_count": carrier_commits,
            "precision_decision_count": accepted + rejected,
            "precision_accept_count": accepted,
            "precision_rejection_count": rejected,
            "precision_reference": "frozen_support_only_initial_fixed_ridge_not_current_active",
            "target_label_state_uses": 0,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
        }


def _precision_session_from_payload(value: object, *, budget: int, query_trial_ids: Sequence[str]) -> PrecisionSessionEvidence:
    if not isinstance(value, Mapping):
        raise PrecisionMatchedScoreError("Precision session receipt type drift")
    required = {
        "session", "n_windows", "governing_r2", "prediction_sha256", "input_record_sha256",
        "model_state_before_sha256", "model_state_after_sha256", "initial_carrier_sha256", "group_assignment_sha256",
        "group_valid_mask_sha256", "initial_activity_sha256", "support_trial_ids_sha256", "raw_m30_t4_axis_proof_sha256",
        "sealed_normalizer_sha256", "sealed_model_load_proof_sha256", "target_last_bin_sha256", "valid_mask_sha256",
        "valid_last_bin_count", "activity_fifo_capacity", "group_forward_count", "full_system_forward_count",
        "dropout_calls", "target_label_state_uses", "nonfinite_prediction_count", "target_optimizer_steps",
        "target_backward_calls", "target_update_calls", "schema", "system", "budget",
        "independent_activity_contract", "transition_records", "conditional_posterior",
        "conditional_posterior_sha256", "frozen_support_reference_float64_sha256",
        "conditional_posterior_decisions", "carrier_transition_committed_count",
        "precision_decision_count", "precision_accept_count", "precision_rejection_count", "precision_reference", "metric",
    }
    if (
        set(value) != required or value.get("schema") != "precision_aware_cdmd_matched_score_session_v1"
        or value.get("system") != plan.SYSTEM_PRECISION_V2 or value.get("budget") != budget
        or value.get("independent_activity_contract") != "IndependentActivityCausalDualMemory__observe_completed_trial__commit_independent"
        or value.get("precision_reference") != "frozen_support_only_initial_fixed_ridge_not_current_active"
        or not isinstance(value.get("transition_records"), list) or not isinstance(value.get("conditional_posterior_decisions"), list)
    ):
        raise PrecisionMatchedScoreError("Precision session receipt schema drift")
    rows = _validate_transition_rows(value["transition_records"], query_trial_ids=query_trial_ids)
    rejected: dict[str, int] = {}
    for row in rows:
        reason = row["carrier_rejection_reason_or_null"]
        if isinstance(reason, str):
            rejected[reason] = rejected.get(reason, 0) + 1
    try:
        base = v1score.SessionScore(
            session=value["session"], n_windows=value["n_windows"], r2=value["governing_r2"],
            prediction_sha256=value["prediction_sha256"], input_record_sha256=value["input_record_sha256"],
            model_state_before_sha256=value["model_state_before_sha256"], model_state_after_sha256=value["model_state_after_sha256"],
            initial_carrier_sha256=value["initial_carrier_sha256"], group_assignment_sha256=value["group_assignment_sha256"],
            group_valid_mask_sha256=value["group_valid_mask_sha256"], initial_activity_sha256=value["initial_activity_sha256"],
            support_trial_ids_sha256=value["support_trial_ids_sha256"], raw_m30_t4_axis_proof_sha256=value["raw_m30_t4_axis_proof_sha256"],
            sealed_normalizer_sha256=value["sealed_normalizer_sha256"], sealed_model_load_proof_sha256=value["sealed_model_load_proof_sha256"],
            target_last_bin_sha256=value["target_last_bin_sha256"], valid_mask_sha256=value["valid_mask_sha256"],
            valid_last_bin_count=value["valid_last_bin_count"], activity_fifo_capacity=value["activity_fifo_capacity"],
            accepted_updates=sum(row["carrier_transition_committed"] for row in rows) * v1plan.GROUP_COUNT,
            rejected_updates={key: count * v1plan.GROUP_COUNT for key, count in rejected.items()},
            group_forward_count=value["group_forward_count"], full_system_forward_count=value["full_system_forward_count"],
            dropout_calls=value["dropout_calls"], target_label_state_uses=value["target_label_state_uses"],
            nonfinite_prediction_count=value["nonfinite_prediction_count"], target_optimizer_steps=value["target_optimizer_steps"],
            target_backward_calls=value["target_backward_calls"], target_update_calls=value["target_update_calls"],
        )
        decisions = tuple(value["conditional_posterior_decisions"])
        result = PrecisionSessionEvidence(
            base=base, transition_records=tuple(rows), conditional_posterior=value["conditional_posterior"],
            conditional_posterior_sha256=value["conditional_posterior_sha256"],
            frozen_support_reference_float64_sha256=value["frozen_support_reference_float64_sha256"],
            decisions=decisions,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PrecisionMatchedScoreError("Precision session reconstruction drift") from error
    if result.payload(budget=budget, query_trial_ids=query_trial_ids) != dict(value):
        raise PrecisionMatchedScoreError("Precision session canonical payload drift")
    return result


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values = [_finite(row.get("governing_r2"), "system governing R2") for row in rows]
    _require(bool(values), "system summary needs sessions")
    ordered = sorted(values)
    middle = len(ordered) // 2
    return {
        "mean": sum(values) / len(values),
        "median": ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0,
        "n_sessions": len(values),
    }


def _paired(reference: Sequence[Mapping[str, object]], candidate: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not reference or tuple(row.get("session") for row in reference) != tuple(row.get("session") for row in candidate):
        raise PrecisionMatchedScoreError("Precision paired roster/order drift")
    deltas = [
        _finite(current.get("governing_r2"), "Precision candidate R2") - _finite(base.get("governing_r2"), "V8 sealed R2")
        for base, current in zip(reference, candidate, strict=True)
    ]
    rng = random.Random(plan.PAIRED_BOOTSTRAP_SEED)
    draws = [sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(plan.PAIRED_BOOTSTRAP_DRAWS)]
    ordered_draws = sorted(draws)
    sorted_deltas = sorted(deltas)
    middle = len(sorted_deltas) // 2
    return {
        "n_sessions": len(deltas), "mean_delta": sum(deltas) / len(deltas),
        "median_delta": sorted_deltas[middle] if len(sorted_deltas) % 2 else (sorted_deltas[middle - 1] + sorted_deltas[middle]) / 2.0,
        "n_positive": sum(value > 0.0 for value in deltas), "deltas": deltas,
        "bootstrap": {
            "seed": plan.PAIRED_BOOTSTRAP_SEED, "draws": plan.PAIRED_BOOTSTRAP_DRAWS,
            "ci95": [
                ordered_draws[int(math.floor(0.025 * (len(ordered_draws) - 1)))],
                ordered_draws[int(math.ceil(0.975 * (len(ordered_draws) - 1)))],
            ],
        },
    }


@dataclass(frozen=True)
class PrecisionCellEvidence:
    surface: str
    budget: int
    input_authority_sha256: str
    model_swa_sha256: str
    sessions: tuple[PrecisionSessionEvidence, ...]
    resources: Mapping[str, object]
    v8_reused_sealed_cell: Mapping[str, object]
    v8_reused_sealed_cell_sha256: str
    v8_m30_reference_cell: Mapping[str, object]
    v8_m30_reference_cell_sha256: str
    v8_predecessor_binding_sha256: str

    def payload(self, *, input_payload: Mapping[str, object]) -> dict[str, object]:
        if self.surface not in plan.SURFACES or self.budget not in plan.BUDGETS:
            raise PrecisionMatchedScoreError("Precision cell matrix drift")
        _sha(self.input_authority_sha256, "Precision cell input authority")
        if self.model_swa_sha256 != v1plan.SEALED_CELL_D_SWA_SHA256:
            raise PrecisionMatchedScoreError("Precision cell sealed SWA drift")
        records = input_payload.get("records") if isinstance(input_payload, Mapping) else None
        if not isinstance(records, list):
            raise PrecisionMatchedScoreError("Precision cell input records absent")
        expected_records = [row for row in records if isinstance(row, Mapping) and row.get("surface") == self.surface]
        if len(expected_records) != v1plan.expected_session_count(self.surface) or len(self.sessions) != len(expected_records):
            raise PrecisionMatchedScoreError("Precision cell session cardinality drift")
        rows = [
            item.payload(budget=self.budget, query_trial_ids=record["query_trial_ids_by_budget"][str(self.budget)])
            for item, record in zip(self.sessions, expected_records, strict=True)
        ]
        if tuple(row["session"] for row in rows) != tuple(record["session"] for record in expected_records):
            raise PrecisionMatchedScoreError("Precision cell session order drift")
        for row, record in zip(rows, expected_records, strict=True):
            key = str(self.budget)
            if (
                row["input_record_sha256"] != _digest(record)
                or row["support_trial_ids_sha256"] != record["support_trial_ids_sha256_by_budget"][key]
                or row["target_last_bin_sha256"] != record["target_last_bin_sha256_by_budget"][key]
                or row["valid_mask_sha256"] != record["valid_last_bin_mask_sha256_by_budget"][key]
                or row["valid_last_bin_count"] != record["valid_last_bin_count_by_budget"][key]
            ):
                raise PrecisionMatchedScoreError("Precision cell input/target/mask binding drift")
        reused = dict(self.v8_reused_sealed_cell)
        m30 = dict(self.v8_m30_reference_cell)
        if (
            _digest(reused) != _sha(self.v8_reused_sealed_cell_sha256, "V8 reused sealed cell")
            or _digest(m30) != _sha(self.v8_m30_reference_cell_sha256, "V8 M30 reference cell")
            or reused.get("surface") != self.surface or reused.get("budget") != self.budget
            or reused.get("system") != v8plan.SYSTEM_SEALED
            or m30.get("surface") != self.surface or m30.get("budget") != 30
            or m30.get("system") != v8plan.SYSTEM_SEALED
        ):
            raise PrecisionMatchedScoreError("V8 reused sealed/M30 cell identity drift")
        reused_sessions = reused.get("sessions")
        if not isinstance(reused_sessions, list) or len(reused_sessions) != len(rows):
            raise PrecisionMatchedScoreError("V8 reused sealed session topology drift")
        for fresh, historical in zip(rows, reused_sessions, strict=True):
            if not isinstance(historical, Mapping) or any(
                fresh.get(key) != historical.get(key) for key in (
                    "session", "input_record_sha256", "support_trial_ids_sha256", "target_last_bin_sha256",
                    "valid_mask_sha256", "valid_last_bin_count", "initial_carrier_sha256",
                    "group_assignment_sha256", "group_valid_mask_sha256", "initial_activity_sha256",
                    "sealed_normalizer_sha256", "sealed_model_load_proof_sha256",
                )
            ):
                raise PrecisionMatchedScoreError("Precision/V8 same-input initial-state bridge drift")
        return {
            "schema": "precision_aware_cdmd_matched_score_cell_v1",
            "surface": self.surface, "budget": self.budget, "system": plan.SYSTEM_PRECISION_V2,
            "input_authority_sha256": self.input_authority_sha256, "model_swa_sha256": self.model_swa_sha256,
            "sessions": rows, "resources": dict(self.resources),
            "v8_reused_sealed_cell": reused, "v8_reused_sealed_cell_sha256": self.v8_reused_sealed_cell_sha256,
            "v8_m30_reference_cell": m30, "v8_m30_reference_cell_sha256": self.v8_m30_reference_cell_sha256,
            "v8_predecessor_binding_sha256": _sha(self.v8_predecessor_binding_sha256, "V8 binding"),
            "eval_mode": True, "no_grad": True, "dropout_disabled": True,
            "same_input_as_v8": True, "target_optimizer_backward_update": 0,
        }


def _cell_from_payload(
    value: object, *, input_payload: Mapping[str, object], expected_input_sha256: str, identity: plan.ScoreIdentity,
) -> PrecisionCellEvidence:
    required = {
        "schema", "surface", "budget", "system", "input_authority_sha256", "model_swa_sha256", "sessions",
        "resources", "v8_reused_sealed_cell", "v8_reused_sealed_cell_sha256", "v8_m30_reference_cell",
        "v8_m30_reference_cell_sha256", "v8_predecessor_binding_sha256", "eval_mode", "no_grad",
        "dropout_disabled", "same_input_as_v8", "target_optimizer_backward_update",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_matched_score_cell_v1"
        or value.get("system") != plan.SYSTEM_PRECISION_V2 or value.get("surface") not in plan.SURFACES
        or value.get("budget") not in plan.BUDGETS or value.get("eval_mode") is not True
        or value.get("no_grad") is not True or value.get("dropout_disabled") is not True
        or value.get("same_input_as_v8") is not True or value.get("target_optimizer_backward_update") != 0
        or not isinstance(value.get("sessions"), list) or not isinstance(value.get("resources"), Mapping)
    ):
        raise PrecisionMatchedScoreError("Precision cell receipt schema/boundary drift")
    records = input_payload.get("records") if isinstance(input_payload, Mapping) else None
    if not isinstance(records, list):
        raise PrecisionMatchedScoreError("Precision cell input record source drift")
    records = [row for row in records if isinstance(row, Mapping) and row.get("surface") == value["surface"]]
    rows = tuple(_precision_session_from_payload(
        row, budget=value["budget"], query_trial_ids=record["query_trial_ids_by_budget"][str(value["budget"])]
    ) for row, record in zip(value["sessions"], records, strict=True))
    result = PrecisionCellEvidence(
        surface=value["surface"], budget=value["budget"], input_authority_sha256=value["input_authority_sha256"],
        model_swa_sha256=value["model_swa_sha256"], sessions=rows, resources=value["resources"],
        v8_reused_sealed_cell=value["v8_reused_sealed_cell"],
        v8_reused_sealed_cell_sha256=value["v8_reused_sealed_cell_sha256"],
        v8_m30_reference_cell=value["v8_m30_reference_cell"],
        v8_m30_reference_cell_sha256=value["v8_m30_reference_cell_sha256"],
        v8_predecessor_binding_sha256=value["v8_predecessor_binding_sha256"],
    )
    if result.input_authority_sha256 != _sha(expected_input_sha256, "Precision expected input authority"):
        raise PrecisionMatchedScoreError("Precision cell durable input-authority SHA drift")
    try:
        v1score._validate_resource_evidence(value["resources"], identity=identity)
    except v1score.ScoreError as error:
        raise PrecisionMatchedScoreError(str(error)) from error
    witness = _identity_payload(identity)["v8_predecessor_binding"]
    if result.v8_predecessor_binding_sha256 != witness["binding_sha256"]:
        raise PrecisionMatchedScoreError("Precision cell V8 witness binding drift")
    expected_cell_sha = witness["sealed_cell_sha256s"].get(f"m{result.budget}:{result.surface}")
    expected_m30_sha = witness["sealed_cell_sha256s"].get(f"m30:{result.surface}")
    if result.v8_reused_sealed_cell_sha256 != expected_cell_sha or result.v8_m30_reference_cell_sha256 != expected_m30_sha:
        raise PrecisionMatchedScoreError("Precision cell V8 reused-row digest drift")
    if result.payload(input_payload=input_payload) != dict(value):
        raise PrecisionMatchedScoreError("Precision cell canonical payload drift")
    return result


def _acceptance(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    decisions = [item for row in rows for item in row["conditional_posterior_decisions"] if item is not None]
    accepted = sum(item["accepted"] is True for item in decisions)
    rejected = sum(item["accepted"] is False for item in decisions)
    return {
        "conditional_posterior_decision_count": len(decisions), "accept_count": accepted,
        "precision_rejection_count": rejected,
        "acceptance_rate_or_null": None if not decisions else accepted / len(decisions),
        "familywise_rule": "Bonferroni_chi2_df2__-2*log(0.05/N_valid)",
        "reference": "frozen_support_only_initial_fixed_ridge_not_current_active",
    }


def summarize_budget(cells: Sequence[PrecisionCellEvidence], budget: int, input_payload: Mapping[str, object]) -> dict[str, object]:
    if budget not in plan.BUDGETS or tuple((item.surface, item.budget) for item in cells) != tuple((surface, budget) for surface in plan.SURFACES):
        raise PrecisionMatchedScoreError("Precision budget cell order/topology drift")
    payloads = [item.payload(input_payload=input_payload) for item in cells]
    surfaces: dict[str, object] = {}
    m30: dict[str, object] = {}
    for cell in payloads:
        surface = cell["surface"]
        precision_rows = cell["sessions"]
        sealed = cell["v8_reused_sealed_cell"]
        sealed_rows = sealed["sessions"]
        surfaces[surface] = {
            "v8_reused_sealed_cell_sha256": cell["v8_reused_sealed_cell_sha256"],
            "v8_reused_sealed": _summary(sealed_rows),
            "precision_v2": _summary(precision_rows),
            "paired_precision_v2_minus_v8_sealed": _paired(sealed_rows, precision_rows),
            "precision_transition_acceptance": _acceptance(precision_rows),
        }
        reference = cell["v8_m30_reference_cell"]
        if surface in m30 and m30[surface] != reference:
            raise PrecisionMatchedScoreError("Precision M30 V8 reference changed across budgets")
        m30[surface] = reference
    return {
        "schema": "precision_aware_cdmd_matched_score_budget_summary_v1", "budget": budget,
        "precision_cells": payloads, "surfaces": surfaces,
        "m30_v8_sealed_deployment_reference": m30,
        "m30_rerun": False, "same_input_authority_for_all_new_cells": len({item.input_authority_sha256 for item in cells}) == 1,
        "target_optimizer_backward_update": 0,
    }


def budget_gate(summary: Mapping[str, object], budget: int) -> dict[str, object]:
    """Integrity-only screen completion; no post-hoc performance verdict."""
    if summary.get("budget") != budget or not isinstance(summary.get("surfaces"), Mapping):
        raise PrecisionMatchedScoreError("Precision screen budget summary drift")
    for surface in plan.SURFACES:
        row = summary["surfaces"].get(surface)
        if not isinstance(row, Mapping) or not isinstance(row.get("paired_precision_v2_minus_v8_sealed"), Mapping):
            raise PrecisionMatchedScoreError("Precision screen paired surface summary drift")
    return {"budget": budget, "screen_complete": True, "formal_gate": False, "posthoc_selection": False}


def terminal_verdict(gates: Mapping[int, Mapping[str, object]]) -> str:
    if set(gates) != set(plan.BUDGETS) or any(gates[key].get("screen_complete") is not True for key in plan.BUDGETS):
        raise PrecisionMatchedScoreError("Precision terminal requires complete M10/M4 screen")
    return "SCREEN_COMPLETE_NO_FORMAL_VERDICT"


def _score_payload(
    identity: plan.ScoreIdentity, input_sha: str, summaries: Mapping[str, object], gates: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    payload = _identity_payload(identity)
    return {
        "schema": "precision_aware_cdmd_matched_score_v1", "identity": payload,
        "input_authority_sha256": _sha(input_sha, "Precision score input authority"),
        "budget_summaries": {str(key): dict(summaries[str(key)]) for key in plan.BUDGETS},
        "budget_gates": {str(key): dict(gates[key]) for key in plan.BUDGETS},
        "budget_execution_order": list(plan.BUDGETS),
        "cell_execution_order": [
            {"budget": budget, "surface": surface, "system": plan.SYSTEM_PRECISION_V2}
            for budget in plan.BUDGETS for surface in plan.SURFACES
        ],
        "m30_execution": "v8_immutable_sealed_deployment_reference_only__not_rerun",
        "v8_predecessor_binding": payload["v8_predecessor_binding"],
        "target_optimizer_backward_update": 0,
        "formal_verdict_emitted": False,
    }


def validate_score_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, input_payload_value: Mapping[str, object],
    input_authority_sha256: str, evaluation_authority: v1score.FixedEvaluationAuthority,
) -> dict[str, object]:
    required = {
        "schema", "identity", "input_authority_sha256", "budget_summaries", "budget_gates",
        "budget_execution_order", "cell_execution_order", "m30_execution", "v8_predecessor_binding",
        "target_optimizer_backward_update", "formal_verdict_emitted",
    }
    input_sha = _sha(input_authority_sha256, "Precision score input authority")
    if _digest(input_payload_value) != input_sha:
        raise PrecisionMatchedScoreError("Precision score input body/SHA drift")
    validate_input_payload(input_payload_value, identity, evaluation_authority)
    expected_order = [
        {"budget": budget, "surface": surface, "system": plan.SYSTEM_PRECISION_V2}
        for budget in plan.BUDGETS for surface in plan.SURFACES
    ]
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_matched_score_v1"
        or value.get("identity") != _identity_payload(identity) or value.get("input_authority_sha256") != input_sha
        or value.get("budget_execution_order") != list(plan.BUDGETS) or value.get("cell_execution_order") != expected_order
        or value.get("m30_execution") != "v8_immutable_sealed_deployment_reference_only__not_rerun"
        or value.get("v8_predecessor_binding") != _identity_payload(identity)["v8_predecessor_binding"]
        or value.get("target_optimizer_backward_update") != 0 or value.get("formal_verdict_emitted") is not False
    ):
        raise PrecisionMatchedScoreError("Precision score schema/identity/matrix drift")
    summaries, gates = value.get("budget_summaries"), value.get("budget_gates")
    if not isinstance(summaries, Mapping) or not isinstance(gates, Mapping) or set(summaries) != {"10", "4"} or set(gates) != {"10", "4"}:
        raise PrecisionMatchedScoreError("Precision score complete M10/M4 topology drift")
    for budget in plan.BUDGETS:
        raw = summaries[str(budget)]
        if not isinstance(raw, Mapping) or raw.get("budget") != budget or not isinstance(raw.get("precision_cells"), list):
            raise PrecisionMatchedScoreError("Precision score budget summary schema drift")
        cells = tuple(_cell_from_payload(
            cell, input_payload=input_payload_value, expected_input_sha256=input_sha, identity=identity,
        ) for cell in raw["precision_cells"])
        rebuilt = summarize_budget(cells, budget, input_payload_value)
        if dict(raw) != rebuilt or gates[str(budget)] != budget_gate(rebuilt, budget):
            raise PrecisionMatchedScoreError("Precision score budget reconstruction drift")
    return dict(value)


def _terminal_payload(
    identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str, score_sha: str,
    gates: Mapping[int, Mapping[str, object]], verdict: str,
) -> dict[str, object]:
    if verdict != "SCREEN_COMPLETE_NO_FORMAL_VERDICT":
        raise PrecisionMatchedScoreError("Precision terminal verdict drift")
    closure = _identity_payload(identity)["closure"]["closure_sha256"]
    return {
        "schema": "precision_aware_cdmd_matched_score_terminal_v1", "status": "TERMINAL", "verdict": verdict,
        "identity": _identity_payload(identity), "attempt_sha256": _sha(attempt_sha, "Precision terminal attempt"),
        "input_authority_sha256": _sha(input_sha, "Precision terminal input"), "score_sha256": _sha(score_sha, "Precision terminal score"),
        "budget_gates": {str(key): dict(gates[key]) for key in plan.BUDGETS},
        "launch_closure_sha256": closure, "final_closure_sha256": closure,
        "v8_predecessor_binding": _identity_payload(identity)["v8_predecessor_binding"],
        "target_optimizer_backward_update": 0, "complete_new_m10_m4_matrix": True,
        "m30_reference_only": True,
    }


def validate_terminal_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str,
    score_payload: Mapping[str, object], score_sha: str,
) -> dict[str, object]:
    gates = score_payload.get("budget_gates")
    if not isinstance(gates, Mapping):
        raise PrecisionMatchedScoreError("Precision terminal score-gate source drift")
    typed = {int(key): item for key, item in gates.items() if isinstance(key, str) and key.isdigit() and isinstance(item, Mapping)}
    expected = _terminal_payload(identity, attempt_sha, input_sha, score_sha, typed, terminal_verdict(typed))
    if dict(value) != expected:
        raise PrecisionMatchedScoreError("Precision terminal canonical graph drift")
    return expected


def _attempt_payload(identity: plan.ScoreIdentity, pre_sha: str, auth_sha: str) -> dict[str, object]:
    return {
        "schema": "precision_aware_cdmd_matched_score_attempt_v1", "status": "ATTEMPT_RESERVED",
        "identity": _identity_payload(identity), "preflight_sha256": _sha(pre_sha, "Precision attempt preflight"),
        "authorization_sha256": _sha(auth_sha, "Precision attempt authorization"),
        "target_paths_resolved_or_opened": False, "checkpoint_opened": False, "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
    }


def _failure_payload(
    identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str | None, stage: str,
    error: BaseException, runtime_progress: object | None,
) -> dict[str, object]:
    try:
        progress = v1score._failure_progress(runtime_progress)
    except v1score.ScoreError as error_value:
        raise PrecisionMatchedScoreError(str(error_value)) from error_value
    return {
        "schema": "precision_aware_cdmd_matched_score_failure_v1", "status": "FAILED",
        "identity": _identity_payload(identity), "attempt_sha256": _sha(attempt_sha, "Precision failure attempt"),
        "input_authority_sha256": None if input_sha is None else _sha(input_sha, "Precision failure input"),
        "stage": stage, "error_class": type(error).__name__, "error_sha256": hashlib.sha256(repr(error).encode()).hexdigest(),
        "target_paths_resolved_or_opened": bool(progress["within_assets_opened"] or progress["external_assets_opened"]),
        **progress, "terminal_published": False, "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0, "v8_predecessor_binding": _identity_payload(identity)["v8_predecessor_binding"],
    }


def validate_failure_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str | None,
) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "attempt_sha256", "input_authority_sha256", "stage", "error_class",
        "error_sha256", "target_paths_resolved_or_opened", "within_assets_opened", "external_assets_opened",
        "checkpoint_opened", "cuda_initialized", "full_system_forward_count", "group_forward_count",
        "terminal_published", "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        "v8_predecessor_binding",
    }
    valid_stages = {"attempt", "prepare", "materialize_inputs", "final_revalidate", "publish_terminal"} | {
        f"budget_m{budget}" for budget in plan.BUDGETS
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_matched_score_failure_v1" or value.get("status") != "FAILED"
        or value.get("identity") != _identity_payload(identity) or value.get("attempt_sha256") != attempt_sha
        or value.get("input_authority_sha256") != input_sha or value.get("stage") not in valid_stages
        or value.get("terminal_published") is not False
        or any(value.get(key) != 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls"))
        or value.get("v8_predecessor_binding") != _identity_payload(identity)["v8_predecessor_binding"]
    ):
        raise PrecisionMatchedScoreError("Precision failure schema/progress drift")
    try:
        v1score._failure_progress({key: value[key] for key in (
            "within_assets_opened", "external_assets_opened", "checkpoint_opened", "cuda_initialized",
            "full_system_forward_count", "group_forward_count",
        )})
    except v1score.ScoreError as error:
        raise PrecisionMatchedScoreError(str(error)) from error
    return dict(value)


def build_target_free_preflight(
    *, root: Path, identity: plan.ScoreIdentity, predecessor: V8PredecessorBinding,
    fixed_authority: v1score.FixedEvaluationAuthority | None = None,
) -> dict[str, object]:
    payload = _identity_payload(identity)
    if predecessor.payload() != payload["v8_predecessor_binding"]:
        raise PrecisionMatchedScoreError("Precision preflight V8 binding/identity drift")
    fixed = v1score.derive_fixed_evaluation_authority(Path(root)) if fixed_authority is None else fixed_authority
    return {
        "schema": "precision_aware_cdmd_matched_score_target_free_preflight_v1", "status": "PREFLIGHT_ACCEPTED",
        # ``run_profiled_score_lifecycle`` has one non-optional predecessor
        # hook named ``source_gate``.  For this successor its value is the
        # held V8 graph (which semantically contains the accepted V5 source
        # gate), rather than a caller-selected source-gate mapping.  Retain
        # the explicit V8 name as a duplicate cross-binding so neither the
        # shared lifecycle nor route-local provenance can be silently
        # detached from the exact same descriptor-derived witness.
        "identity": payload, "source_gate": predecessor.payload(), "v8_predecessor": predecessor.payload(),
        "evaluation_authority": fixed.payload(),
        "metric": dict(v1plan.METRIC_CONTRACT), "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE, "target_free": True, "target_paths_resolved": False,
        "model_or_checkpoint_opened": False, "cuda_initialized": False,
        "boundaries": {**v1plan.EXECUTION_BOUNDARIES, "precision_v2_target_state": False},
    }


def validate_target_free_preflight(value: Mapping[str, object], identity: plan.ScoreIdentity) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "source_gate", "v8_predecessor", "evaluation_authority", "metric",
        "authority_root_relative", "score_root_relative", "target_free", "target_paths_resolved",
        "model_or_checkpoint_opened", "cuda_initialized", "boundaries",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_matched_score_target_free_preflight_v1"
        or value.get("status") != "PREFLIGHT_ACCEPTED" or value.get("identity") != _identity_payload(identity)
        or value.get("metric") != v1plan.METRIC_CONTRACT or value.get("authority_root_relative") != plan.AUTHORITY_ROOT_RELATIVE
        or value.get("score_root_relative") != plan.SCORE_ROOT_RELATIVE or value.get("target_free") is not True
        or value.get("target_paths_resolved") is not False or value.get("model_or_checkpoint_opened") is not False
        or value.get("cuda_initialized") is not False
    ):
        raise PrecisionMatchedScoreError("Precision preflight schema/boundary drift")
    source_gate = plan.validate_v8_binding_witness(value.get("source_gate"))
    witness = plan.validate_v8_binding_witness(value.get("v8_predecessor"))
    if (
        source_gate != witness
        or witness != _identity_payload(identity)["v8_predecessor_binding"]
    ):
        raise PrecisionMatchedScoreError("Precision preflight V8 witness drift")
    fixed = v1score._fixed_authority_from_payload(value.get("evaluation_authority"))
    if not fixed.payload():  # typed constructor has already fail-closed
        raise PrecisionMatchedScoreError("Precision preflight fixed authority drift")
    return dict(value)


def validate_preflight_against_fixed_authorities(root: Path, value: Mapping[str, object], *, identity: plan.ScoreIdentity) -> dict[str, object]:
    checked = validate_target_free_preflight(value, identity)
    try:
        fixed = v1score.derive_fixed_evaluation_authority(Path(root)).payload()
    except v1score.ScoreError as error:
        raise PrecisionMatchedScoreError(str(error)) from error
    if checked["evaluation_authority"] != fixed:
        raise PrecisionMatchedScoreError("Precision durable preflight fixed authority drift")
    observed = validate_completed_v8_predecessor(Path(root)).payload()
    if observed != checked["v8_predecessor"] or observed != checked["source_gate"]:
        raise PrecisionMatchedScoreError("Precision durable preflight V8 predecessor drift")
    return checked


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    checked = preflight
    if not isinstance(checked.get("v8_predecessor"), Mapping):
        raise PrecisionMatchedScoreError("Precision authorization V8 predecessor absent")
    return {
        "schema": "precision_aware_cdmd_matched_score_root_authorization_v1", "status": "ROOT_AUTHORIZED",
        "official_preflight_sha256": _sha(official_preflight_sha256, "Precision preflight SHA"),
        "identity_sha256": _digest(checked["identity"]),
        "v8_predecessor_binding_sha256": checked["v8_predecessor"].get("binding_sha256"),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE, "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "target_free_preflight_required": True, "explicit_execution_capability_required": True,
    }


def validate_root_authorization(
    value: Mapping[str, object], *, official_preflight_sha256: str, preflight: Mapping[str, object], identity: plan.ScoreIdentity,
) -> dict[str, object]:
    expected = build_root_authorization(official_preflight_sha256=official_preflight_sha256, preflight=preflight)
    if dict(value) != expected or expected["identity_sha256"] != identity.sha256:
        raise PrecisionMatchedScoreError("Precision root authorization canonical identity drift")
    return expected


def _read_durable_authority_pair(
    root: Path, *, identity: plan.ScoreIdentity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    directory = Path(root).absolute() / plan.AUTHORITY_ROOT_RELATIVE
    try:
        named = v1score._directory_identity(directory)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except (OSError, v1score.ScoreError) as error:
        raise PrecisionMatchedScoreError("Precision durable authority root cannot be held") from error
    try:
        held = os.fstat(descriptor)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != named:
            raise PrecisionMatchedScoreError("Precision durable authority identity drift before read")
        expected = set(AUTHORITY_TOPOLOGY) | {f"{name}.sha256" for name in AUTHORITY_TOPOLOGY}
        if set(os.listdir(descriptor)) != expected:
            raise PrecisionMatchedScoreError("Precision durable authority pair topology drift")
        _body, preflight, pre_sha = v1score._read_unbound_0444_pair(descriptor, "official_preflight.json")
        _body, authorization, auth_sha = v1score._read_unbound_0444_pair(descriptor, "root_authorization.json")
        if v1score._directory_identity(directory) != named:
            raise PrecisionMatchedScoreError("Precision durable authority identity drift during read")
        checked = validate_target_free_preflight(preflight, identity)
        validate_root_authorization(authorization, official_preflight_sha256=pre_sha, preflight=checked, identity=identity)
        return checked, dict(authorization), pre_sha, auth_sha
    except (v1score.ScoreError, OSError) as error:
        raise PrecisionMatchedScoreError("Precision durable authority descriptor read failed") from error
    finally:
        os.close(descriptor)


def load_durable_authority(root: Path, *, identity: plan.ScoreIdentity) -> tuple[dict[str, object], dict[str, object], str, str]:
    return _read_durable_authority_pair(Path(root), identity=identity)


def validate_selected_launch_environment(identity: plan.ScoreIdentity, environ: Mapping[str, str] | None = None) -> dict[str, object]:
    values = os.environ if environ is None else environ
    selected = _identity_payload(identity)["selected_device_profile"]
    if not isinstance(selected, Mapping) or (
        values.get("CUDA_VISIBLE_DEVICES") != selected.get("cuda_visible_devices")
        or values.get("CUDA_DEVICE_ORDER") != selected.get("cuda_device_order")
    ):
        raise PrecisionMatchedScoreError("Precision selected launch environment CVD/PCI order drift")
    return dict(selected)


def reserve_authority_artifact(root: Path, capability: object, *, identity: plan.ScoreIdentity) -> v1score.ArtifactRoot:
    v1score._require_root_publication_capability(capability)
    base = Path(root)
    if plan.implementation_closure(base).payload() != _identity_payload(identity)["closure"]:
        raise PrecisionMatchedScoreError("Precision authority reservation closure drift")
    validate_completed_v8_predecessor(base)
    plan.assert_fresh_prospective_root(base, plan.AUTHORITY_ROOT_RELATIVE)
    parent = base.absolute() / Path(plan.AUTHORITY_ROOT_RELATIVE).parent
    return v1score._equal_session_module(base).reserve_artifact_root(parent, Path(plan.AUTHORITY_ROOT_RELATIVE).name, topology=AUTHORITY_TOPOLOGY)


def publish_target_free_preflight(
    root: Path, artifact: v1score.ArtifactRoot, capability: object, payload: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> str:
    v1score._require_root_publication_capability(capability)
    checked = validate_preflight_against_fixed_authorities(Path(root), payload, identity=identity)
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization(
    root: Path, artifact: v1score.ArtifactRoot, capability: object, payload: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> str:
    v1score._require_root_publication_capability(capability)
    body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise PrecisionMatchedScoreError("Precision durable preflight malformed") from error
    if not isinstance(preflight, Mapping):
        raise PrecisionMatchedScoreError("Precision durable preflight root drift")
    checked = validate_preflight_against_fixed_authorities(Path(root), preflight, identity=identity)
    authorization = validate_root_authorization(
        payload, official_preflight_sha256=hashlib.sha256(body).hexdigest(), preflight=checked, identity=identity,
    )
    return artifact.publish_json("root_authorization.json", authorization)


def issue_durable_execution_capability(
    root: Path, *, identity: plan.ScoreIdentity, root_capability: object, environ: Mapping[str, str] | None = None,
) -> v1score.ExecutionCapability:
    v1score._require_root_publication_capability(root_capability)
    validate_selected_launch_environment(identity, environ)
    base = Path(root)
    if plan.implementation_closure(base).payload() != _identity_payload(identity)["closure"]:
        raise PrecisionMatchedScoreError("Precision execution issuer closure drift")
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(base, identity=identity)
    validate_preflight_against_fixed_authorities(base, preflight, identity=identity)
    plan.assert_fresh_prospective_root(base, plan.SCORE_ROOT_RELATIVE)
    return v1score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=root_capability,
    )


def reserve_score_artifact(
    root: Path, *, identity: plan.ScoreIdentity, capability: object, environ: Mapping[str, str] | None = None,
) -> v1score.ArtifactRoot:
    v1score.require_execution_capability(capability, identity)
    validate_selected_launch_environment(identity, environ)
    base = Path(root)
    if plan.implementation_closure(base).payload() != _identity_payload(identity)["closure"]:
        raise PrecisionMatchedScoreError("Precision score reservation closure drift")
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(base, identity=identity)
    approved = v1score.require_execution_capability(capability, identity)
    if approved.official_preflight_sha256 != pre_sha or approved.root_authorization_sha256 != auth_sha:
        raise PrecisionMatchedScoreError("Precision score capability/durable authority drift")
    validate_preflight_against_fixed_authorities(base, preflight, identity=identity)
    plan.assert_fresh_prospective_root(base, plan.SCORE_ROOT_RELATIVE)
    parent = base.absolute() / Path(plan.SCORE_ROOT_RELATIVE).parent
    return v1score._equal_session_module(base).reserve_artifact_root(parent, Path(plan.SCORE_ROOT_RELATIVE).name, topology=SCORE_TOPOLOGY)


def _validate_reserved_score_artifact(root: Path, artifact: v1score.ArtifactRoot, identity: Any) -> None:
    if not isinstance(identity, plan.ScoreIdentity):
        raise PrecisionMatchedScoreError("Precision reserved artifact identity type drift")
    try:
        v1score.validate_reserved_artifact_root(
            Path(root), artifact, root_relative=plan.SCORE_ROOT_RELATIVE, topology=SCORE_TOPOLOGY,
        )
    except v1score.ScoreError as error:
        raise PrecisionMatchedScoreError(str(error)) from error


def _validate_preflight_hook(value: Mapping[str, object], identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise PrecisionMatchedScoreError("Precision lifecycle identity type drift")
    return validate_target_free_preflight(value, identity)


def _validate_authorization_hook(value: Mapping[str, object], pre_sha: str, preflight: Mapping[str, object], identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise PrecisionMatchedScoreError("Precision lifecycle identity type drift")
    return validate_root_authorization(value, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity)


def _closure_hook(root: Path) -> Mapping[str, object]:
    return plan.implementation_closure(root).payload()


def _source_gate_hook(root: Path) -> V8PredecessorBinding:
    return validate_completed_v8_predecessor(root)


def _fresh_score_hook(root: Path) -> None:
    plan.assert_fresh_prospective_root(root, plan.SCORE_ROOT_RELATIVE)


def _fixed_from_preflight(preflight: Mapping[str, object]) -> v1score.FixedEvaluationAuthority:
    try:
        return v1score._fixed_authority_from_payload(preflight["evaluation_authority"])
    except (KeyError, v1score.ScoreError) as error:
        raise PrecisionMatchedScoreError("Precision fixed authority reconstruction drift") from error


def _input_payload_hook(authority: Any, identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity) or not isinstance(authority, v1score.InputAuthority):
        raise PrecisionMatchedScoreError("Precision physical input type drift")
    return input_payload(authority, identity)


def _validate_input_hook(value: Mapping[str, object], identity: Any, fixed: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity) or not isinstance(fixed, v1score.FixedEvaluationAuthority):
        raise PrecisionMatchedScoreError("Precision input hook typed boundary drift")
    return validate_input_payload(value, identity, fixed)


def _continue_after_budget(_budget: int, _gates: Mapping[int, Mapping[str, object]]) -> bool:
    return True


PRECISION_LIFECYCLE_HOOKS = v1score.ProfiledLifecycleHooks(
    route="precision_aware_causal_dual_memory_cell_d_matched_score_v1",
    score_root_relative=plan.SCORE_ROOT_RELATIVE, budgets=plan.BUDGETS,
    require_capability=v1score.require_execution_capability, validate_preflight=_validate_preflight_hook,
    validate_authorization=_validate_authorization_hook, implementation_closure=_closure_hook,
    validate_source_gate=_source_gate_hook, assert_fresh_score_root=_fresh_score_hook,
    make_attempt=_attempt_payload, fixed_authority_from_preflight=_fixed_from_preflight,
    input_payload=_input_payload_hook, validate_input_payload=_validate_input_hook,
    summarize_budget=summarize_budget, budget_gate=budget_gate, continue_after_budget=_continue_after_budget,
    build_score=_score_payload, validate_score=validate_score_payload, terminal_verdict=terminal_verdict,
    make_terminal=_terminal_payload, validate_terminal=validate_terminal_payload,
    make_failure=_failure_payload, validate_failure=validate_failure_payload,
    validate_reserved_score_artifact=_validate_reserved_score_artifact,
)


def run_authorized_score_lifecycle(
    root: Path, *, identity: plan.ScoreIdentity, capability: object, backend: v1score.ScoreBackend,
    artifact: v1score.ArtifactRoot, official_preflight_sha256: str, root_authorization_sha256: str,
    preflight: Mapping[str, object], authorization: Mapping[str, object],
) -> Mapping[str, object]:
    """Use the one shared immutable V1 lifecycle engine without a copy."""
    return v1score.run_profiled_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=official_preflight_sha256, root_authorization_sha256=root_authorization_sha256,
        preflight=preflight, authorization=authorization, hooks=PRECISION_LIFECYCLE_HOOKS,
    )
