"""No-data lifecycle and receipt contract for the V5-bound CDM-D score.

The actual target parser/model path remains in :mod:`physical` and is reached
only through an opaque capability after a durable attempt.  This module uses
the shared V1 profile lifecycle engine; it does not copy a scorer lifecycle.
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

from src.causal_dual_memory_cell_d_score_v1 import score as v1score

from . import plan


AUTHORITY_TOPOLOGY = v1score.AUTHORITY_TOPOLOGY
SCORE_TOPOLOGY = v1score.SCORE_TOPOLOGY

# Literal mirror of the closure-bound core.UpdateRejectionReason domain.  The
# score receipt must not permit a caller to invent a new opaque reason while
# recomputing its count map.  Keeping the wire values local preserves the
# static/no-Torch nature of this lifecycle module; the physical seam separately
# validates the actual core outcome before it is encoded here.
_UPDATE_REJECTION_REASONS = frozenset({
    "activity_shape", "activity_nonfinite", "trial_capability", "trial_binding",
    "carrier_counts", "velocity_validity", "velocity_shape", "velocity_nonfinite",
    "movement_mask", "movement_too_short", "low_displacement", "low_mean_speed",
    "canonical_direction_too_far", "complementary_disagreement", "insufficient_evidence",
    "insufficient_design_rank", "ill_conditioned_design", "nonfinite_carrier",
    "departure_freeze", "stale_pending_update",
})

# The accepted V5 producer carries this contract exactly once in its durable
# ``v5_binding``.  Attempt/launch/terminal bodies deliberately do *not* copy
# it at top level; source authority and every aggregate/session evidence body
# carry an equal top-level copy for their own schema.  Keep this wire contract
# literal here so the scorer can verify the nested producer authority without
# importing the source execution route or accepting a merely nonempty mapping.
_V3_INDEPENDENT_ACTIVITY_CONTRACT = {
    "schema": "causal_dual_memory_independent_activity_transition_contract_v3",
    "valid_b3s_activity_commits_independently_of_carrier": True,
    "carrier_rejection_preserves_exact_carrier": True,
    "completed_query_count_advances_on_valid_b3s": True,
    "m4_activity_fifo_capacity": 26,
    "m10_activity_fifo_capacity": 20,
    "m30_activity_fifo_capacity": 0,
    "m30_activity_stack_digest_unchanged": True,
    "source_audit_m30_positions_30_59_offline_no_deployment_commit": True,
    "state_visible_only_to_next_trial": True,
    "target_optimizer_backward_update": 0,
}


class V5ScoreError(v1score.ScoreError):
    """Fail closed for V5 lineage, independent-state, or receipt drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V5ScoreError(message)


def _json(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _digest(value: object) -> str:
    return plan.sha256_bytes(_json(value))


def _sha(value: object, label: str) -> str:
    try:
        return plan.require_sha(value, label)
    except plan.V5PlanError as error:
        raise V5ScoreError(str(error)) from error


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise V5ScoreError(f"{label} must be finite")
    return float(value)


def _identity_payload(identity: plan.ScoreIdentity) -> dict[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise V5ScoreError("V5 score requires its exact typed identity")
    return identity.payload()


def _canonical_independent_activity_contract(binding: Mapping[str, object]) -> dict[str, object]:
    """Extract the sole producer-authoritative V3 transition contract.

    The V5 source executor publishes it under ``v5_binding`` in attempt,
    launch, and terminal.  Requiring a fabricated top-level attempt copy made
    the scorer reject the immutable accepted V5 graph before it could even
    reach fixed-target authority validation.
    """
    nested = binding.get("v3_independent_activity_contract")
    if not isinstance(nested, Mapping) or dict(nested) != _V3_INDEPENDENT_ACTIVITY_CONTRACT:
        raise V5ScoreError("V5 source-gate nested independent-activity contract drift")
    return dict(nested)


def _validate_v5_source_gate_semantics(
    attempt: Mapping[str, object], launch: Mapping[str, object], source: Mapping[str, object],
    terminal: Mapping[str, object], evidence: Mapping[str, Mapping[str, object]],
    contract: Any,
) -> tuple[str, ...]:
    """Validate the accepted V5 88-body source graph after held-byte reload."""
    if contract is not plan.SOURCE_GATE_CONTRACT:
        raise V5ScoreError("V5 source-gate contract identity drift")
    fixed = plan.SOURCE_GATE_EXPECTED_SHAS
    if (
        attempt.get("schema") != "causal_dual_memory_cell_d_source_execution_attempt_v5"
        or attempt.get("status") != "ATTEMPT_RESERVED"
        or attempt.get("source_only") is not True
        or attempt.get("within_external_formal_target_forbidden") is not True
        or attempt.get("target_optimizer_backward_update") != 0
        or attempt.get("source_resolved_or_opened") is not False
        or attempt.get("checkpoint_opened") is not False
        or attempt.get("cuda_initialized") is not False
    ):
        raise V5ScoreError("V5 source-gate attempt schema/boundary drift")
    identity = attempt.get("identity")
    if not isinstance(identity, Mapping):
        raise V5ScoreError("V5 source-gate attempt identity absent")
    closure = identity.get("closure")
    if not isinstance(closure, Mapping) or closure.get("closure_sha256") != plan.SOURCE_GATE_CLOSURE_SHA256:
        raise V5ScoreError("V5 source-gate attempt closure drift")
    binding = attempt.get("v5_binding")
    if not isinstance(binding, Mapping) or binding.get("schema") != "causal_dual_memory_cell_d_source_execution_v5_binding":
        raise V5ScoreError("V5 source-gate attempt V5-binding drift")
    independent = _canonical_independent_activity_contract(binding)
    if (
        launch.get("schema") != "causal_dual_memory_cell_d_source_execution_launch_v5"
        or launch.get("status") != "LAUNCHED"
        or launch.get("attempt_sha256") != fixed["attempt.json"]
        or launch.get("launch_closure_sha256") != plan.SOURCE_GATE_CLOSURE_SHA256
        or launch.get("source_only") is not True
        or launch.get("target_optimizer_backward_update") != 0
        or launch.get("identity") != identity
        or launch.get("v5_binding") != binding
    ):
        raise V5ScoreError("V5 source-gate launch graph/boundary drift")
    roster = source.get("strict_train_roster")
    if (
        not isinstance(roster, list)
        or len(roster) != plan.SOURCE_GATE_STRICT_SOURCE_SESSION_TOTAL
        or len(set(roster)) != plan.SOURCE_GATE_STRICT_SOURCE_SESSION_TOTAL
        or any(not isinstance(item, str) or not item for item in roster)
        or source.get("source_only") is not True
        or source.get("v5_binding") != binding
        or source.get("independent_activity_contract") != independent
    ):
        raise V5ScoreError("V5 source-gate source-authority roster/contract drift")
    access = source.get("access")
    if not isinstance(access, Mapping) or any(access.get(key) is not False for key in (
        "within_opened", "external_opened", "formal_opened", "target_opened",
    )) or any(access.get(key) != 0 for key in ("optimizer_steps", "backward_calls", "parameter_updates")):
        raise V5ScoreError("V5 source-gate source-authority target/update boundary drift")
    if (
        terminal.get("schema") != "causal_dual_memory_cell_d_source_execution_terminal_v5"
        or terminal.get("status") != plan.SOURCE_GATE_STATUS
        or terminal.get("attempt_sha256") != fixed["attempt.json"]
        or terminal.get("launch_sha256") != fixed["launch.json"]
        or terminal.get("source_authority_sha256") != fixed["source_authority.json"]
        or terminal.get("launch_closure_sha256") != plan.SOURCE_GATE_CLOSURE_SHA256
        or terminal.get("final_closure_sha256") != plan.SOURCE_GATE_CLOSURE_SHA256
        or terminal.get("source_only") is not True
        or terminal.get("target_optimizer_backward_update") != 0
        or terminal.get("v5_binding") != binding
        or terminal.get("identity") != identity
    ):
        raise V5ScoreError("V5 source-gate terminal graph/boundary drift")
    evidence_map = terminal.get("evidence_sha256s")
    expected_names = {
        f"budget_m{budget}_aggregate.json" for budget in plan.BUDGETS
    } | {
        f"budget_m{budget}__{session}.json" for budget in plan.BUDGETS for session in roster
    }
    if not isinstance(evidence_map, Mapping) or set(evidence_map) != expected_names or set(evidence) != expected_names:
        raise V5ScoreError("V5 source-gate terminal evidence topology drift")
    for name in sorted(expected_names):
        if _digest(evidence[name]) != _sha(evidence_map[name], f"V5 source-gate evidence {name} SHA"):
            raise V5ScoreError(f"V5 source-gate evidence body/terminal digest drift: {name}")
        row = evidence[name]
        if row.get("v5_binding") != binding or row.get("independent_activity_contract") != independent:
            raise V5ScoreError(f"V5 source-gate independent activity evidence drift: {name}")
    # Bind the retained pass/total topology independently of the immutable
    # breadth threshold.  In particular, M30's accepted 27/27 aggregate has
    # producer field ``breadth_min_passing_sessions == 14`` rather than 27.
    # The row-level pass count cross-check prevents a forged aggregate from
    # claiming the literal pass count while its retained session evidence says
    # something different.
    m4_rows = [evidence[f"budget_m4__{session}.json"] for session in roster]
    if sum(row.get("pass") is True for row in m4_rows) != plan.SOURCE_GATE_EXPECTED_PASSING_SESSION_COUNTS[4]:
        raise V5ScoreError("V5 source-gate sole M4 source nonpass topology drift")
    for budget in plan.BUDGETS:
        aggregate = evidence[f"budget_m{budget}_aggregate.json"]
        rows = [evidence[f"budget_m{budget}__{session}.json"] for session in roster]
        expected_row_shas = [_digest(row) for row in rows]
        expected_passing = plan.SOURCE_GATE_EXPECTED_PASSING_SESSION_COUNTS[budget]
        observed_passing = sum(row.get("pass") is True for row in rows)
        if (
            aggregate.get("schema") != "causal_dual_memory_cell_d_source_execution_budget_aggregate_v1"
            or aggregate.get("budget") != budget
            or aggregate.get("strict_source_session_count") != plan.SOURCE_GATE_STRICT_SOURCE_SESSION_TOTAL
            or aggregate.get("session_body_sha256s") != expected_row_shas
            or aggregate.get("passing_session_count") != expected_passing
            or observed_passing != expected_passing
            or aggregate.get("breadth_min_passing_sessions") != plan.SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS
            or aggregate.get("breadth_pass") is not (
                expected_passing >= plan.SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS
            )
            or aggregate.get("source_only") is not True
        ):
            raise V5ScoreError(f"V5 source-gate M{budget} aggregate/breadth drift")
    return tuple(roster)


def validate_completed_source_gate(root: Path) -> v1score.SourceGateBinding:
    return v1score.load_completed_source_gate_contract(
        Path(root), contract=plan.SOURCE_GATE_CONTRACT, semantic_validator=_validate_v5_source_gate_semantics,
    )


def _binding_from_payload(value: object) -> v1score.SourceGateBinding:
    required = {
        "schema", "root_relative", "directory_identity", "body_sha256s", "terminal_status",
        "implementation_closure_sha256", "strict_source_roster", "exact_json_body_count",
        "exact_leaf_count", "source_gate_is_target_score", "binding_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise V5ScoreError("V5 source-gate binding schema drift")
    try:
        directory_identity = value["directory_identity"]
        binding = v1score.SourceGateBinding(
            directory_device=directory_identity[0], directory_inode=directory_identity[1],
            body_sha256s=value["body_sha256s"], terminal_status=value["terminal_status"],
            source_gate_closure_sha256=value["implementation_closure_sha256"],
            strict_source_roster=tuple(value["strict_source_roster"]), contract=plan.SOURCE_GATE_CONTRACT,
        )
    except (KeyError, TypeError, IndexError) as error:
        raise V5ScoreError("V5 source-gate binding reconstruction drift") from error
    if binding.payload() != dict(value):
        raise V5ScoreError("V5 source-gate binding canonical drift")
    return binding


def build_target_free_preflight(
    *, root: Path, identity: plan.ScoreIdentity, source_gate: v1score.SourceGateBinding,
    fixed_authority: v1score.FixedEvaluationAuthority | None = None,
) -> dict[str, object]:
    identity_payload = _identity_payload(identity)
    if not isinstance(source_gate, v1score.SourceGateBinding) or source_gate.contract is not plan.SOURCE_GATE_CONTRACT:
        raise V5ScoreError("V5 preflight requires exact typed V5 source-gate binding")
    fixed = v1score.derive_fixed_evaluation_authority(Path(root)) if fixed_authority is None else fixed_authority
    fixed_payload = fixed.payload()
    source_payload = source_gate.payload()
    if source_payload["body_sha256s"].get("terminal.json") != identity_payload["source_gate"]["terminal_sha256"]:
        raise V5ScoreError("V5 preflight source-gate/identity terminal drift")
    return {
        "schema": "causal_dual_memory_cell_d_score_target_free_preflight_v5",
        "status": "PREFLIGHT_ACCEPTED",
        "identity": identity_payload,
        "source_gate": source_payload,
        "evaluation_authority": fixed_payload,
        "metric": dict(plan.METRIC_CONTRACT),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "target_free": True,
        "target_paths_resolved": False,
        "model_or_checkpoint_opened": False,
        "cuda_initialized": False,
        "boundaries": dict(plan.EXECUTION_BOUNDARIES),
    }


def validate_target_free_preflight(value: Mapping[str, object], identity: plan.ScoreIdentity) -> dict[str, object]:
    expected = {
        "schema", "status", "identity", "source_gate", "evaluation_authority", "metric",
        "authority_root_relative", "score_root_relative", "target_free", "target_paths_resolved",
        "model_or_checkpoint_opened", "cuda_initialized", "boundaries",
    }
    identity_payload = _identity_payload(identity)
    if (
        not isinstance(value, Mapping) or set(value) != expected
        or value.get("schema") != "causal_dual_memory_cell_d_score_target_free_preflight_v5"
        or value.get("status") != "PREFLIGHT_ACCEPTED"
        or value.get("identity") != identity_payload
        or value.get("metric") != plan.METRIC_CONTRACT
        or value.get("authority_root_relative") != plan.AUTHORITY_ROOT_RELATIVE
        or value.get("score_root_relative") != plan.SCORE_ROOT_RELATIVE
        or value.get("target_free") is not True or value.get("target_paths_resolved") is not False
        or value.get("model_or_checkpoint_opened") is not False or value.get("cuda_initialized") is not False
        or value.get("boundaries") != plan.EXECUTION_BOUNDARIES
    ):
        raise V5ScoreError("V5 target-free preflight schema/identity/boundary drift")
    _binding_from_payload(value.get("source_gate"))
    fixed = v1score._fixed_authority_from_payload(value.get("evaluation_authority"))
    if tuple(item.session for item in fixed.within) == tuple(item.session for item in fixed.external):
        raise V5ScoreError("V5 fixed within/external roster surface conflation")
    return dict(value)


def validate_preflight_against_fixed_authorities(
    root: Path, value: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> dict[str, object]:
    checked = validate_target_free_preflight(value, identity)
    observed = v1score.derive_fixed_evaluation_authority(Path(root)).payload()
    if checked["evaluation_authority"] != observed:
        raise V5ScoreError("V5 durable preflight fixed evaluation authority drift")
    return checked


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    digest = _sha(official_preflight_sha256, "V5 official preflight SHA")
    source = preflight.get("source_gate")
    return {
        "schema": "causal_dual_memory_cell_d_score_root_authorization_v5",
        "status": "ROOT_AUTHORIZED",
        "official_preflight_sha256": digest,
        "identity_sha256": _digest(preflight["identity"]),
        "source_gate_binding_sha256": source.get("binding_sha256") if isinstance(source, Mapping) else None,
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "target_free_preflight_required": True,
        "explicit_execution_capability_required": True,
    }


def validate_root_authorization(
    value: Mapping[str, object], *, official_preflight_sha256: str,
    preflight: Mapping[str, object], identity: plan.ScoreIdentity,
) -> dict[str, object]:
    required = {
        "schema", "status", "official_preflight_sha256", "identity_sha256", "source_gate_binding_sha256",
        "authority_root_relative", "score_root_relative", "target_free_preflight_required",
        "explicit_execution_capability_required",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise V5ScoreError("V5 root authorization schema drift")
    validate_target_free_preflight(preflight, identity)
    expected = build_root_authorization(official_preflight_sha256=official_preflight_sha256, preflight=preflight)
    if dict(value) != expected or _digest(_identity_payload(identity)) != expected["identity_sha256"]:
        raise V5ScoreError("V5 root authorization identity/preflight drift")
    return expected


def reserve_authority_artifact(
    root: Path, capability: object, *, identity: plan.ScoreIdentity,
) -> v1score.ArtifactRoot:
    """Reserve V5 authority output only after the accepted V5 gate is held.

    A durable preflight cannot be allowed to create even its fresh authority
    directory from a caller-claimed predecessor.  The later publication path
    rechecks this graph again, but the first reservation itself is also an
    output-side boundary and therefore needs the exact held V5 terminal graph.
    """
    v1score._require_root_publication_capability(capability)
    root = Path(root)
    identity_payload = _identity_payload(identity)
    if plan.implementation_closure(root).payload() != identity_payload["closure"]:
        raise V5ScoreError("V5 authority reservation implementation closure drift")
    plan.assert_fresh_prospective_root(root, plan.AUTHORITY_ROOT_RELATIVE)
    observed = validate_completed_source_gate(root)
    if observed.payload()["body_sha256s"].get("terminal.json") != identity_payload["source_gate"]["terminal_sha256"]:
        raise V5ScoreError("V5 authority reservation completed source-gate terminal drift")
    parent = root.absolute() / Path(plan.AUTHORITY_ROOT_RELATIVE).parent
    return v1score._equal_session_module(root).reserve_artifact_root(
        parent, Path(plan.AUTHORITY_ROOT_RELATIVE).name, topology=AUTHORITY_TOPOLOGY,
    )


def publish_target_free_preflight(
    root: Path, artifact: v1score.ArtifactRoot, capability: object,
    payload: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> str:
    v1score._require_root_publication_capability(capability)
    checked = validate_preflight_against_fixed_authorities(Path(root), payload, identity=identity)
    observed = validate_completed_source_gate(Path(root))
    if checked["source_gate"] != observed.payload():
        raise V5ScoreError("V5 target-free preflight completed source-gate binding drift")
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization(
    root: Path, artifact: v1score.ArtifactRoot, capability: object,
    payload: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> str:
    v1score._require_root_publication_capability(capability)
    body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise V5ScoreError("V5 durable official preflight malformed") from error
    if not isinstance(preflight, Mapping):
        raise V5ScoreError("V5 durable official preflight root drift")
    preflight = validate_preflight_against_fixed_authorities(Path(root), preflight, identity=identity)
    observed = validate_completed_source_gate(Path(root))
    if preflight["source_gate"] != observed.payload():
        raise V5ScoreError("V5 root authorization completed source-gate binding drift")
    checked = validate_root_authorization(
        payload, official_preflight_sha256=hashlib.sha256(body).hexdigest(), preflight=preflight, identity=identity,
    )
    return artifact.publish_json("root_authorization.json", checked)


def _read_durable_authority_pair(root: Path, *, identity: plan.ScoreIdentity) -> tuple[dict[str, object], dict[str, object], str, str]:
    directory = Path(root).absolute() / plan.AUTHORITY_ROOT_RELATIVE
    named = v1score._directory_identity(directory)
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise V5ScoreError("V5 durable authority root cannot be held") from error
    try:
        held = os.fstat(fd)
        if (int(held.st_dev), int(held.st_ino)) != named or not stat.S_ISDIR(held.st_mode):
            raise V5ScoreError("V5 durable authority root identity drift before read")
        names = set(os.listdir(fd))
        expected = set(AUTHORITY_TOPOLOGY) | {f"{name}.sha256" for name in AUTHORITY_TOPOLOGY}
        if names != expected:
            raise V5ScoreError("V5 durable authority pair topology drift")
        _pre_body, preflight, pre_sha = v1score._read_unbound_0444_pair(fd, "official_preflight.json")
        _auth_body, authorization, auth_sha = v1score._read_unbound_0444_pair(fd, "root_authorization.json")
        if v1score._directory_identity(directory) != named:
            raise V5ScoreError("V5 durable authority root identity drift during read")
        validate_target_free_preflight(preflight, identity)
        validate_root_authorization(
            authorization, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity,
        )
        return dict(preflight), dict(authorization), pre_sha, auth_sha
    finally:
        os.close(fd)


def load_durable_authority(
    root: Path, *, identity: plan.ScoreIdentity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    return _read_durable_authority_pair(Path(root), identity=identity)


def validate_selected_launch_environment(identity: plan.ScoreIdentity, environ: Mapping[str, str] | None = None) -> dict[str, object]:
    values = os.environ if environ is None else environ
    selected = dict(_identity_payload(identity)["selected_device_profile"])
    if (
        values.get("CUDA_VISIBLE_DEVICES") != selected["cuda_visible_devices"]
        or values.get("CUDA_DEVICE_ORDER") != selected["cuda_device_order"]
    ):
        raise V5ScoreError("V5 selected score launch environment CVD/PCI-order drift")
    return selected


def issue_durable_execution_capability(
    root: Path, *, identity: plan.ScoreIdentity, root_capability: object,
    environ: Mapping[str, str] | None = None,
) -> v1score.ExecutionCapability:
    v1score._require_root_publication_capability(root_capability)
    validate_selected_launch_environment(identity, environ)
    current = plan.implementation_closure(Path(root)).payload()
    if current != _identity_payload(identity)["closure"]:
        raise V5ScoreError("V5 execution issuer implementation closure drift")
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(Path(root), identity=identity)
    validate_preflight_against_fixed_authorities(Path(root), preflight, identity=identity)
    if validate_completed_source_gate(Path(root)).payload() != preflight["source_gate"]:
        raise V5ScoreError("V5 execution issuer completed source-gate binding drift")
    plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)
    return v1score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=root_capability,
    )


def reserve_score_artifact(
    root: Path, *, identity: plan.ScoreIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> v1score.ArtifactRoot:
    v1score.require_execution_capability(capability, identity)
    validate_selected_launch_environment(identity, environ)
    if plan.implementation_closure(Path(root)).payload() != _identity_payload(identity)["closure"]:
        raise V5ScoreError("V5 score root reservation implementation closure drift")
    preflight, _auth, pre_sha, auth_sha = load_durable_authority(Path(root), identity=identity)
    approved = v1score.require_execution_capability(capability, identity)
    if approved.official_preflight_sha256 != pre_sha or approved.root_authorization_sha256 != auth_sha:
        raise V5ScoreError("V5 score root reservation durable capability SHA drift")
    validate_preflight_against_fixed_authorities(Path(root), preflight, identity=identity)
    if validate_completed_source_gate(Path(root)).payload() != preflight["source_gate"]:
        raise V5ScoreError("V5 score root reservation completed source-gate drift")
    plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)
    parent = Path(root).absolute() / Path(plan.SCORE_ROOT_RELATIVE).parent
    return v1score._equal_session_module(root).reserve_artifact_root(
        parent, Path(plan.SCORE_ROOT_RELATIVE).name, topology=SCORE_TOPOLOGY,
    )


@dataclass(frozen=True)
class IndependentSessionScore:
    """V5 per-session evidence with non-collapsed memory transition facts."""

    base: v1score.SessionScore
    transition_records: tuple[Mapping[str, object], ...]
    activity_transition_committed_count: int
    activity_fifo_changed_count: int
    carrier_transition_committed_count: int
    activity_rejection_counts: Mapping[str, int]
    carrier_rejection_counts: Mapping[str, int]

    def payload(self, *, budget: int, system: str, query_trial_ids: Sequence[str]) -> dict[str, object]:
        if budget not in plan.BUDGETS or system not in plan.SYSTEMS:
            raise V5ScoreError("V5 session matrix drift")
        base_payload = self.base.payload(budget=budget, system=system)
        # The V5 receipt deliberately removes the ambiguous coupled fields.
        base_payload.pop("accepted_updates")
        base_payload.pop("rejected_updates")
        if system == plan.SYSTEM_SEALED:
            if (
                self.transition_records or self.activity_transition_committed_count != 0
                or self.activity_fifo_changed_count != 0 or self.carrier_transition_committed_count != 0
                or self.activity_rejection_counts or self.carrier_rejection_counts
            ):
                raise V5ScoreError("V5 sealed comparator may not carry transition state")
        expected_ids = tuple(query_trial_ids)
        if system == plan.SYSTEM_CDMD and len(self.transition_records) != len(expected_ids):
            raise V5ScoreError("V5 independent transition/query cardinality drift")
        rows: list[dict[str, object]] = []
        ordered_rows = (
            zip(expected_ids, self.transition_records, strict=True)
            if system == plan.SYSTEM_CDMD else ()
        )
        for expected_id, raw in ordered_rows:
            if not isinstance(raw, Mapping):
                raise V5ScoreError("V5 transition row schema drift")
            row = dict(raw)
            required = {
                "trial_id", "activity_transition_committed", "activity_fifo_changed",
                "carrier_transition_committed", "activity_rejection_reason_or_null",
                "carrier_rejection_reason_or_null", "state_before_sha256", "state_after_sha256",
                "activity_before_sha256", "activity_after_sha256", "carrier_before_sha256", "carrier_after_sha256",
            }
            if set(row) != required or row.get("trial_id") != expected_id:
                raise V5ScoreError("V5 transition trial-ID/order schema drift")
            for key in ("activity_transition_committed", "activity_fifo_changed", "carrier_transition_committed"):
                if type(row.get(key)) is not bool:
                    raise V5ScoreError("V5 transition boolean drift")
            for key in (
                "state_before_sha256", "state_after_sha256", "activity_before_sha256", "activity_after_sha256",
                "carrier_before_sha256", "carrier_after_sha256",
            ):
                _sha(row.get(key), f"V5 transition {key}")
            for key in ("activity_rejection_reason_or_null", "carrier_rejection_reason_or_null"):
                value = row.get(key)
                if value is not None and (not isinstance(value, str) or value not in _UPDATE_REJECTION_REASONS):
                    raise V5ScoreError("V5 transition rejection reason drift")
            if row["activity_transition_committed"] is False:
                if (
                    row["activity_fifo_changed"] is not False or row["carrier_transition_committed"] is not False
                    or row["activity_rejection_reason_or_null"] is None
                    or row["carrier_rejection_reason_or_null"] is not None
                    or row["state_before_sha256"] != row["state_after_sha256"]
                    or row["activity_before_sha256"] != row["activity_after_sha256"]
                    or row["carrier_before_sha256"] != row["carrier_after_sha256"]
                ):
                    raise V5ScoreError("V5 invalid activity transition must preserve both memories")
            else:
                if row["activity_rejection_reason_or_null"] is not None or row["state_before_sha256"] == row["state_after_sha256"]:
                    raise V5ScoreError("V5 valid activity transition state-chain drift")
                if row["activity_fifo_changed"] is not (row["activity_before_sha256"] != row["activity_after_sha256"]):
                    raise V5ScoreError("V5 activity FIFO/digest relation drift")
                if row["carrier_transition_committed"]:
                    if row["carrier_rejection_reason_or_null"] is not None or row["carrier_before_sha256"] == row["carrier_after_sha256"]:
                        raise V5ScoreError("V5 carrier commit/reason/digest drift")
                elif row["carrier_rejection_reason_or_null"] is None or row["carrier_before_sha256"] != row["carrier_after_sha256"]:
                    raise V5ScoreError("V5 carrier rejection/reason/digest drift")
            rows.append(row)
        if any(rows[index]["state_after_sha256"] != rows[index + 1]["state_before_sha256"] for index in range(len(rows) - 1)):
            raise V5ScoreError("V5 independent transition digest chain drift")
        if budget == 30 and any(row["activity_before_sha256"] != row["activity_after_sha256"] for row in rows):
            raise V5ScoreError("V5 M30 capacity-zero activity invariance drift")
        for mapping in (self.activity_rejection_counts, self.carrier_rejection_counts):
            if not isinstance(mapping, Mapping) or any(
                not isinstance(key, str) or key not in _UPDATE_REJECTION_REASONS
                or type(value) is not int or value < 0
                for key, value in mapping.items()
            ):
                raise V5ScoreError("V5 transition rejection-count map drift")
        activity_count = sum(bool(row["activity_transition_committed"]) for row in rows)
        fifo_count = sum(bool(row["activity_fifo_changed"]) for row in rows)
        carrier_count = sum(bool(row["carrier_transition_committed"]) for row in rows)
        if (
            self.activity_transition_committed_count != activity_count
            or self.activity_fifo_changed_count != fifo_count
            or self.carrier_transition_committed_count != carrier_count
        ):
            raise V5ScoreError("V5 independent transition count reconstruction drift")
        activity_rejections: dict[str, int] = {}
        carrier_rejections: dict[str, int] = {}
        for row in rows:
            activity_reason = row["activity_rejection_reason_or_null"]
            carrier_reason = row["carrier_rejection_reason_or_null"]
            if isinstance(activity_reason, str):
                activity_rejections[activity_reason] = activity_rejections.get(activity_reason, 0) + 1
            if isinstance(carrier_reason, str):
                carrier_rejections[carrier_reason] = carrier_rejections.get(carrier_reason, 0) + 1
        if dict(sorted(self.activity_rejection_counts.items())) != activity_rejections or dict(sorted(self.carrier_rejection_counts.items())) != carrier_rejections:
            raise V5ScoreError("V5 independent rejection-count reconstruction drift")
        return {
            **base_payload,
            "schema": "causal_dual_memory_cell_d_score_session_independent_v5",
            "independent_activity_contract": "IndependentActivityCausalDualMemory__observe_completed_trial__commit_independent",
            "transition_records": rows,
            "activity_transition_committed_count": self.activity_transition_committed_count,
            "activity_fifo_changed_count": self.activity_fifo_changed_count,
            "carrier_transition_committed_count": self.carrier_transition_committed_count,
            "activity_rejection_counts": dict(sorted(self.activity_rejection_counts.items())),
            "carrier_rejection_counts": dict(sorted(self.carrier_rejection_counts.items())),
            "metric": "last_bin_variance_weighted_two_output_r2",
            "system": system,
            "budget": budget,
        }


@dataclass(frozen=True)
class CellEvidence:
    surface: str
    budget: int
    system: str
    input_authority_sha256: str
    model_swa_sha256: str
    sessions: tuple[IndependentSessionScore, ...]
    resources: Mapping[str, object]

    def payload(self, *, input_payload: Mapping[str, object]) -> dict[str, object]:
        if self.surface not in plan.SURFACES or self.budget not in plan.BUDGETS or self.system not in plan.SYSTEMS:
            raise V5ScoreError("V5 cell evidence matrix drift")
        _sha(self.input_authority_sha256, "V5 cell input authority SHA")
        if self.model_swa_sha256 != plan.v1plan.SEALED_CELL_D_SWA_SHA256:
            raise V5ScoreError("V5 cell sealed SWA provenance drift")
        records = input_payload.get("records") if isinstance(input_payload, Mapping) else None
        if not isinstance(records, list):
            raise V5ScoreError("V5 cell input authority records unavailable")
        expected_inputs = {
            item["session"]: item for item in records
            if isinstance(item, Mapping) and item.get("surface") == self.surface
        }
        if len(self.sessions) != plan.expected_session_count(self.surface) or tuple(row.base.session for row in self.sessions) != tuple(expected_inputs):
            raise V5ScoreError("V5 cell session/input order drift")
        rows = [
            session.payload(
                budget=self.budget, system=self.system,
                query_trial_ids=expected_inputs[session.base.session]["query_trial_ids_by_budget"][str(self.budget)],
            )
            for session in self.sessions
        ]
        for row in rows:
            record = expected_inputs[row["session"]]
            key = str(self.budget)
            if (
                row["input_record_sha256"] != _digest(record)
                or row["support_trial_ids_sha256"] != record["support_trial_ids_sha256_by_budget"][key]
                or row["target_last_bin_sha256"] != record["target_last_bin_sha256_by_budget"][key]
                or row["valid_mask_sha256"] != record["valid_last_bin_mask_sha256_by_budget"][key]
                or row["valid_last_bin_count"] != record["valid_last_bin_count_by_budget"][key]
                or row["n_windows"] != record["valid_last_bin_count_by_budget"][key]
            ):
                raise V5ScoreError("V5 cell input/target/mask/count binding drift")
        return {
            "schema": "causal_dual_memory_cell_d_score_cell_evidence_v5",
            "surface": self.surface,
            "budget": self.budget,
            "system": self.system,
            "input_authority_sha256": self.input_authority_sha256,
            "model_swa_sha256": self.model_swa_sha256,
            "sessions": rows,
            "resources": dict(self.resources),
            "eval_mode": True,
            "no_grad": True,
            "dropout_disabled": True,
            "same_sealed_model_state_for_both_systems": True,
        }


def _session_from_payload(value: object, *, query_trial_ids: Sequence[str]) -> IndependentSessionScore:
    if not isinstance(value, Mapping):
        raise V5ScoreError("V5 session payload type drift")
    required = {
        "session", "n_windows", "governing_r2", "prediction_sha256", "input_record_sha256",
        "model_state_before_sha256", "model_state_after_sha256", "initial_carrier_sha256",
        "group_assignment_sha256", "group_valid_mask_sha256", "initial_activity_sha256",
        "support_trial_ids_sha256", "raw_m30_t4_axis_proof_sha256", "sealed_normalizer_sha256",
        "sealed_model_load_proof_sha256", "target_last_bin_sha256", "valid_mask_sha256", "valid_last_bin_count",
        "activity_fifo_capacity", "group_forward_count", "full_system_forward_count", "dropout_calls",
        "target_label_state_uses", "nonfinite_prediction_count", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls", "schema", "independent_activity_contract", "transition_records",
        "activity_transition_committed_count", "activity_fifo_changed_count", "carrier_transition_committed_count",
        "activity_rejection_counts", "carrier_rejection_counts", "metric", "system", "budget",
    }
    if set(value) != required or value.get("schema") != "causal_dual_memory_cell_d_score_session_independent_v5":
        raise V5ScoreError("V5 session payload schema drift")
    try:
        base = v1score.SessionScore(
            session=value["session"], n_windows=value["n_windows"], r2=value["governing_r2"],
            prediction_sha256=value["prediction_sha256"], input_record_sha256=value["input_record_sha256"],
            model_state_before_sha256=value["model_state_before_sha256"],
            model_state_after_sha256=value["model_state_after_sha256"],
            initial_carrier_sha256=value["initial_carrier_sha256"],
            group_assignment_sha256=value["group_assignment_sha256"], group_valid_mask_sha256=value["group_valid_mask_sha256"],
            initial_activity_sha256=value["initial_activity_sha256"], support_trial_ids_sha256=value["support_trial_ids_sha256"],
            raw_m30_t4_axis_proof_sha256=value["raw_m30_t4_axis_proof_sha256"],
            sealed_normalizer_sha256=value["sealed_normalizer_sha256"],
            sealed_model_load_proof_sha256=value["sealed_model_load_proof_sha256"],
            target_last_bin_sha256=value["target_last_bin_sha256"], valid_mask_sha256=value["valid_mask_sha256"],
            valid_last_bin_count=value["valid_last_bin_count"], activity_fifo_capacity=value["activity_fifo_capacity"],
            accepted_updates=int(value["carrier_transition_committed_count"]) * plan.GROUP_COUNT,
            rejected_updates={}, group_forward_count=value["group_forward_count"],
            full_system_forward_count=value["full_system_forward_count"], dropout_calls=value["dropout_calls"],
            target_label_state_uses=value["target_label_state_uses"], nonfinite_prediction_count=value["nonfinite_prediction_count"],
            target_optimizer_steps=value["target_optimizer_steps"], target_backward_calls=value["target_backward_calls"],
            target_update_calls=value["target_update_calls"],
        )
        result = IndependentSessionScore(
            base=base, transition_records=tuple(value["transition_records"]),
            activity_transition_committed_count=value["activity_transition_committed_count"],
            activity_fifo_changed_count=value["activity_fifo_changed_count"],
            carrier_transition_committed_count=value["carrier_transition_committed_count"],
            activity_rejection_counts=value["activity_rejection_counts"], carrier_rejection_counts=value["carrier_rejection_counts"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise V5ScoreError("V5 session payload reconstruction drift") from error
    budget, system = value["budget"], value["system"]
    if result.payload(budget=budget, system=system, query_trial_ids=query_trial_ids) != dict(value):
        raise V5ScoreError("V5 session payload canonical transition drift")
    return result


def _validate_resources(value: Mapping[str, object], *, identity: plan.ScoreIdentity) -> dict[str, object]:
    try:
        return v1score._validate_resource_evidence(value, identity=identity)
    except v1score.ScoreError as error:
        raise V5ScoreError(str(error)) from error


def _cell_from_payload(
    value: object, *, input_payload: Mapping[str, object], expected_input_sha256: str, identity: plan.ScoreIdentity,
) -> CellEvidence:
    required = {
        "schema", "surface", "budget", "system", "input_authority_sha256", "model_swa_sha256", "sessions",
        "resources", "eval_mode", "no_grad", "dropout_disabled", "same_sealed_model_state_for_both_systems",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "causal_dual_memory_cell_d_score_cell_evidence_v5"
        or value.get("eval_mode") is not True or value.get("no_grad") is not True
        or value.get("dropout_disabled") is not True or value.get("same_sealed_model_state_for_both_systems") is not True
        or not isinstance(value.get("sessions"), list) or not isinstance(value.get("resources"), Mapping)
    ):
        raise V5ScoreError("V5 cell evidence schema/boundary drift")
    records = input_payload.get("records") if isinstance(input_payload, Mapping) else None
    if not isinstance(records, list):
        raise V5ScoreError("V5 cell input records absent")
    rows_for_surface = [item for item in records if isinstance(item, Mapping) and item.get("surface") == value.get("surface")]
    if len(rows_for_surface) != len(value["sessions"]):
        raise V5ScoreError("V5 cell session cardinality drift")
    sessions = tuple(
        _session_from_payload(row, query_trial_ids=record["query_trial_ids_by_budget"][str(value["budget"])])
        for row, record in zip(value["sessions"], rows_for_surface, strict=True)
    )
    result = CellEvidence(
        surface=value["surface"], budget=value["budget"], system=value["system"],
        input_authority_sha256=value["input_authority_sha256"], model_swa_sha256=value["model_swa_sha256"],
        sessions=sessions, resources=value["resources"],
    )
    _validate_resources(value["resources"], identity=identity)
    if result.input_authority_sha256 != _sha(expected_input_sha256, "V5 expected input authority SHA"):
        raise V5ScoreError("V5 cell durable input authority SHA drift")
    if result.payload(input_payload=input_payload) != dict(value):
        raise V5ScoreError("V5 cell evidence canonical/input binding drift")
    return result


def _system_summary(rows: Sequence[IndependentSessionScore]) -> dict[str, object]:
    if not rows:
        raise V5ScoreError("V5 system summary needs sessions")
    values = sorted(float(item.base.r2) for item in rows)
    midpoint = len(values) // 2
    median = values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2.0
    return {"mean": sum(values) / len(values), "median": median, "n_sessions": len(values)}


def _paired_summary(sealed: Sequence[IndependentSessionScore], cdmd: Sequence[IndependentSessionScore]) -> dict[str, object]:
    if not sealed or tuple(row.base.session for row in sealed) != tuple(row.base.session for row in cdmd):
        raise V5ScoreError("V5 paired session roster/order drift")
    deltas = [float(candidate.base.r2 - baseline.base.r2) for baseline, candidate in zip(sealed, cdmd, strict=True)]
    rng = random.Random(plan.PAIRED_BOOTSTRAP_SEED)
    draws = [sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(plan.PAIRED_BOOTSTRAP_DRAWS)]
    ordered = sorted(draws)
    sorted_delta = sorted(deltas)
    midpoint = len(sorted_delta) // 2
    median = sorted_delta[midpoint] if len(sorted_delta) % 2 else (sorted_delta[midpoint - 1] + sorted_delta[midpoint]) / 2.0
    return {
        "n_sessions": len(deltas), "mean_delta": sum(deltas) / len(deltas), "median_delta": median,
        "n_positive": sum(value > 0.0 for value in deltas), "deltas": deltas,
        "bootstrap": {
            "seed": plan.PAIRED_BOOTSTRAP_SEED, "draws": plan.PAIRED_BOOTSTRAP_DRAWS,
            "ci95": [
                ordered[int(math.floor(0.025 * (len(ordered) - 1)))],
                ordered[int(math.ceil(0.975 * (len(ordered) - 1)))],
            ],
        },
    }


def summarize_budget(cells: Sequence[CellEvidence], budget: int, input_payload: Mapping[str, object]) -> dict[str, object]:
    expected = [(surface, system) for surface in plan.SURFACES for system in plan.SYSTEMS]
    if [(cell.surface, cell.system) for cell in cells] != expected or any(cell.budget != budget for cell in cells):
        raise V5ScoreError("V5 budget summary cell order/topology drift")
    payloads = [cell.payload(input_payload=input_payload) for cell in cells]
    surfaces: dict[str, object] = {}
    for surface in plan.SURFACES:
        sealed = next(cell.sessions for cell in cells if cell.surface == surface and cell.system == plan.SYSTEM_SEALED)
        cdmd = next(cell.sessions for cell in cells if cell.surface == surface and cell.system == plan.SYSTEM_CDMD)
        for baseline, candidate in zip(sealed, cdmd, strict=True):
            if (
                baseline.base.session != candidate.base.session
                or baseline.base.input_record_sha256 != candidate.base.input_record_sha256
                or baseline.base.initial_carrier_sha256 != candidate.base.initial_carrier_sha256
                or baseline.base.group_assignment_sha256 != candidate.base.group_assignment_sha256
                or baseline.base.group_valid_mask_sha256 != candidate.base.group_valid_mask_sha256
                or baseline.base.initial_activity_sha256 != candidate.base.initial_activity_sha256
                or baseline.base.support_trial_ids_sha256 != candidate.base.support_trial_ids_sha256
                or baseline.base.raw_m30_t4_axis_proof_sha256 != candidate.base.raw_m30_t4_axis_proof_sha256
                or baseline.base.sealed_normalizer_sha256 != candidate.base.sealed_normalizer_sha256
                or baseline.base.sealed_model_load_proof_sha256 != candidate.base.sealed_model_load_proof_sha256
                or baseline.base.target_last_bin_sha256 != candidate.base.target_last_bin_sha256
                or baseline.base.valid_mask_sha256 != candidate.base.valid_mask_sha256
                or baseline.base.valid_last_bin_count != candidate.base.valid_last_bin_count
                or baseline.base.n_windows != candidate.base.n_windows
                or baseline.base.model_state_before_sha256 != candidate.base.model_state_before_sha256
                or baseline.base.model_state_after_sha256 != candidate.base.model_state_after_sha256
            ):
                raise V5ScoreError("V5 sealed/CDM-D same-input/initial-state paired binding drift")
        surfaces[surface] = {
            "sealed_cell_d": _system_summary(sealed), "cdm_d": _system_summary(cdmd),
            "paired_cdm_d_minus_sealed": _paired_summary(sealed, cdmd),
        }
    return {
        "schema": "causal_dual_memory_cell_d_score_budget_summary_v5", "budget": budget,
        "cells": payloads, "surfaces": surfaces,
        "same_input_authority_for_all_cells": len({cell.input_authority_sha256 for cell in cells}) == 1,
        "independent_activity_transition_required": True,
    }


def budget_gate(summary: Mapping[str, object], budget: int) -> dict[str, object]:
    if summary.get("budget") != budget or not isinstance(summary.get("surfaces"), Mapping):
        raise V5ScoreError("V5 budget-gate summary drift")
    external = summary["surfaces"].get(plan.v1plan.EXTERNAL)
    if not isinstance(external, Mapping) or not isinstance(external.get("paired_cdm_d_minus_sealed"), Mapping):
        raise V5ScoreError("V5 external paired summary drift")
    paired = external["paired_cdm_d_minus_sealed"]
    mean = _finite(paired.get("mean_delta"), "V5 external paired mean")
    positives = paired.get("n_positive")
    if type(positives) is not int:
        raise V5ScoreError("V5 external paired positives drift")
    if budget == 30:
        passed, rule = mean >= plan.M30_EXTERNAL_MIN, {"mean_delta_min": plan.M30_EXTERNAL_MIN}
    elif budget == 10:
        passed, rule = (
            mean >= plan.M10_EXTERNAL_MIN and positives >= plan.M10_EXTERNAL_POSITIVE_MIN,
            {"mean_delta_min": plan.M10_EXTERNAL_MIN, "positive_sessions_min": plan.M10_EXTERNAL_POSITIVE_MIN},
        )
    elif budget == 4:
        passed, rule = (
            mean >= plan.M4_EXTERNAL_MIN and positives >= plan.M4_EXTERNAL_POSITIVE_MIN,
            {"mean_delta_min": plan.M4_EXTERNAL_MIN, "positive_sessions_min": plan.M4_EXTERNAL_POSITIVE_MIN},
        )
    else:
        raise V5ScoreError("V5 unknown budget gate")
    return {
        "budget": budget, "external_gate_pass": passed, "rule": rule,
        "observed_mean_delta": mean, "observed_positive_sessions": positives,
    }


def terminal_verdict(gates: Mapping[int, Mapping[str, object]]) -> str:
    if set(gates) != set(plan.BUDGETS):
        raise V5ScoreError("V5 terminal requires complete fixed 12-cell matrix")
    m10, m4 = gates[10], gates[4]
    if m10.get("external_gate_pass") is True or m4.get("external_gate_pass") is True:
        return "ADVANCE_SHORT_BUDGET"
    m10_mean = _finite(m10.get("observed_mean_delta"), "V5 M10 mean")
    m4_mean = _finite(m4.get("observed_mean_delta"), "V5 M4 mean")
    if m10_mean > 0.0 or m4_mean > 0.0:
        return "HOLD_WEAK_SIGNAL"
    return "STOP_NO_PERFORMANCE_SIGNAL"


def _score_payload(
    identity: plan.ScoreIdentity, input_sha: str, summaries: Mapping[str, object], gates: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_matched_score_v5",
        "identity": _identity_payload(identity),
        "input_authority_sha256": _sha(input_sha, "V5 score input authority SHA"),
        "budget_summaries": {str(key): dict(summaries[str(key)]) for key in plan.BUDGETS},
        "budget_gates": {str(key): dict(gates[key]) for key in plan.BUDGETS},
        # JSON object key order is deliberately canonicalized/sorted at
        # publication, so preserve the reviewed non-adaptive execution order
        # as explicit receipt data rather than relying on mapping iteration.
        "budget_execution_order": list(plan.BUDGETS),
        "cell_execution_order": [
            {"budget": budget, "surface": surface, "system": system}
            for budget in plan.BUDGETS for surface in plan.SURFACES for system in plan.SYSTEMS
        ],
        "matrix_policy": "always_complete_m30_then_m10_then_m4_unless_integrity_or_execution_failure",
        "target_optimizer_backward_update": 0,
        "source_gate_terminal_sha256": plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
    }


def validate_score_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, input_payload: Mapping[str, object],
    input_authority_sha256: str, evaluation_authority: v1score.FixedEvaluationAuthority,
) -> dict[str, object]:
    required = {
        "schema", "identity", "input_authority_sha256", "budget_summaries", "budget_gates",
        "budget_execution_order", "cell_execution_order", "matrix_policy",
        "target_optimizer_backward_update", "source_gate_terminal_sha256",
    }
    input_sha = _sha(input_authority_sha256, "V5 score input authority SHA")
    if _digest(input_payload) != input_sha:
        raise V5ScoreError("V5 score input body/SHA drift")
    try:
        v1score.validate_input_authority_payload(input_payload, identity=identity, evaluation_authority=evaluation_authority)
    except v1score.ScoreError as error:
        raise V5ScoreError(str(error)) from error
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "causal_dual_memory_cell_d_matched_score_v5"
        or value.get("identity") != _identity_payload(identity)
        or value.get("input_authority_sha256") != input_sha
        or value.get("budget_execution_order") != list(plan.BUDGETS)
        or value.get("cell_execution_order") != [
            {"budget": budget, "surface": surface, "system": system}
            for budget in plan.BUDGETS for surface in plan.SURFACES for system in plan.SYSTEMS
        ]
        or value.get("matrix_policy") != "always_complete_m30_then_m10_then_m4_unless_integrity_or_execution_failure"
        or value.get("target_optimizer_backward_update") != 0
        or value.get("source_gate_terminal_sha256") != plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"]
    ):
        raise V5ScoreError("V5 score receipt schema/identity/boundary drift")
    summaries, gates = value.get("budget_summaries"), value.get("budget_gates")
    if not isinstance(summaries, Mapping) or not isinstance(gates, Mapping) or set(summaries) != {"30", "10", "4"} or set(gates) != {"30", "10", "4"}:
        raise V5ScoreError("V5 score complete budget topology drift")
    for budget in plan.BUDGETS:
        raw = summaries[str(budget)]
        if not isinstance(raw, Mapping) or raw.get("budget") != budget or not isinstance(raw.get("cells"), list):
            raise V5ScoreError(f"V5 score M{budget} summary schema drift")
        cells = tuple(_cell_from_payload(
            item, input_payload=input_payload, expected_input_sha256=input_sha, identity=identity,
        ) for item in raw["cells"])
        rebuilt = summarize_budget(cells, budget, input_payload)
        if dict(raw) != rebuilt or gates[str(budget)] != budget_gate(rebuilt, budget):
            raise V5ScoreError(f"V5 score M{budget} summary/gate reconstruction drift")
    return dict(value)


def _terminal_payload(
    identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str, score_sha: str,
    gates: Mapping[int, Mapping[str, object]], verdict: str,
) -> dict[str, object]:
    if verdict not in {"ADVANCE_SHORT_BUDGET", "HOLD_WEAK_SIGNAL", "STOP_NO_PERFORMANCE_SIGNAL"}:
        raise V5ScoreError("V5 terminal verdict drift")
    closure = _identity_payload(identity)["closure"]["closure_sha256"]
    return {
        "schema": "causal_dual_memory_cell_d_score_terminal_v5",
        "status": "TERMINAL",
        "verdict": verdict,
        "identity": _identity_payload(identity),
        "attempt_sha256": _sha(attempt_sha, "V5 terminal attempt SHA"),
        "input_authority_sha256": _sha(input_sha, "V5 terminal input SHA"),
        "score_sha256": _sha(score_sha, "V5 terminal score SHA"),
        "budget_gates": {str(key): dict(gates[key]) for key in plan.BUDGETS},
        "launch_closure_sha256": closure,
        "final_closure_sha256": closure,
        "source_gate_terminal_sha256": plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
        "target_optimizer_backward_update": 0,
        "complete_matrix": True,
    }


def validate_terminal_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str,
    score_payload: Mapping[str, object], score_sha: str,
) -> dict[str, object]:
    required = {
        "schema", "status", "verdict", "identity", "attempt_sha256", "input_authority_sha256", "score_sha256",
        "budget_gates", "launch_closure_sha256", "final_closure_sha256", "source_gate_terminal_sha256",
        "target_optimizer_backward_update", "complete_matrix",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise V5ScoreError("V5 terminal schema drift")
    gates = score_payload.get("budget_gates")
    if not isinstance(gates, Mapping):
        raise V5ScoreError("V5 terminal score-gate source drift")
    typed_gates = {int(key): item for key, item in gates.items() if isinstance(key, str) and key.isdigit() and isinstance(item, Mapping)}
    expected = _terminal_payload(
        identity, attempt_sha, input_sha, score_sha, typed_gates, terminal_verdict(typed_gates),
    )
    if dict(value) != expected:
        raise V5ScoreError("V5 terminal graph/verdict/closure drift")
    return expected


def _attempt_payload(identity: plan.ScoreIdentity, pre_sha: str, auth_sha: str) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_score_attempt_v5", "status": "ATTEMPT_RESERVED",
        "identity": _identity_payload(identity), "preflight_sha256": _sha(pre_sha, "V5 attempt preflight SHA"),
        "authorization_sha256": _sha(auth_sha, "V5 attempt authorization SHA"),
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
        raise V5ScoreError(str(error_value)) from error_value
    return {
        "schema": "causal_dual_memory_cell_d_score_failure_v5", "status": "FAILED",
        "identity": _identity_payload(identity), "attempt_sha256": _sha(attempt_sha, "V5 failure attempt SHA"),
        "input_authority_sha256": None if input_sha is None else _sha(input_sha, "V5 failure input SHA"),
        "stage": stage, "error_class": type(error).__name__, "error_sha256": hashlib.sha256(repr(error).encode()).hexdigest(),
        "target_paths_resolved_or_opened": bool(progress["within_assets_opened"] or progress["external_assets_opened"]),
        **progress,
        "terminal_published": False, "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0, "source_gate_terminal_sha256": plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"],
    }


def validate_failure_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str | None,
) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "attempt_sha256", "input_authority_sha256", "stage", "error_class",
        "error_sha256", "target_paths_resolved_or_opened", "within_assets_opened", "external_assets_opened",
        "checkpoint_opened", "cuda_initialized", "full_system_forward_count", "group_forward_count",
        "terminal_published", "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        "source_gate_terminal_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise V5ScoreError("V5 failure schema drift")
    valid_stages = {"attempt", "prepare", "materialize_inputs", "final_revalidate", "publish_terminal"} | {
        f"budget_m{budget}" for budget in plan.BUDGETS
    }
    if (
        value.get("schema") != "causal_dual_memory_cell_d_score_failure_v5" or value.get("status") != "FAILED"
        or value.get("identity") != _identity_payload(identity)
        or value.get("attempt_sha256") != _sha(attempt_sha, "V5 failure expected attempt SHA")
        or value.get("input_authority_sha256") != input_sha or value.get("stage") not in valid_stages
        or not isinstance(value.get("error_class"), str) or not value["error_class"]
        or value.get("terminal_published") is not False
        or any(value.get(key) != 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls"))
        or value.get("source_gate_terminal_sha256") != plan.SOURCE_GATE_EXPECTED_SHAS["terminal.json"]
    ):
        raise V5ScoreError("V5 failure provenance/boundary drift")
    v1score._failure_progress({key: value.get(key) for key in (
        "within_assets_opened", "external_assets_opened", "checkpoint_opened", "cuda_initialized",
        "full_system_forward_count", "group_forward_count",
    )})
    return dict(value)


def _validate_preflight_hook(value: Mapping[str, object], identity: Any) -> Mapping[str, object]:
    return validate_target_free_preflight(value, identity)


def _validate_authorization_hook(value: Mapping[str, object], pre_sha: str, preflight: Mapping[str, object], identity: Any) -> Mapping[str, object]:
    return validate_root_authorization(value, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity)


def _closure_hook(root: Path) -> Mapping[str, object]:
    return plan.implementation_closure(root).payload()


def _fresh_score_hook(root: Path) -> None:
    plan.assert_fresh_prospective_root(root, plan.SCORE_ROOT_RELATIVE)


def _input_payload_hook(authority: Any, identity: Any) -> Mapping[str, object]:
    if not isinstance(authority, v1score.InputAuthority):
        raise V5ScoreError("V5 physical route must return inherited typed input authority")
    return authority.payload(identity=identity)


def _validate_input_hook(value: Mapping[str, object], identity: Any, fixed: Any) -> Mapping[str, object]:
    if not isinstance(fixed, v1score.FixedEvaluationAuthority):
        raise V5ScoreError("V5 fixed evaluation authority type drift")
    return v1score.validate_input_authority_payload(value, identity=identity, evaluation_authority=fixed)


def _fixed_from_preflight(preflight: Mapping[str, object]) -> Any:
    return v1score._fixed_authority_from_payload(preflight["evaluation_authority"])


def _continue_after_budget(_budget: int, _gates: Mapping[int, Mapping[str, object]]) -> bool:
    return True


V5_LIFECYCLE_HOOKS = v1score.ProfiledLifecycleHooks(
    route="causal_dual_memory_cell_d_score_v5",
    score_root_relative=plan.SCORE_ROOT_RELATIVE,
    budgets=plan.BUDGETS,
    require_capability=v1score.require_execution_capability,
    validate_preflight=_validate_preflight_hook,
    validate_authorization=_validate_authorization_hook,
    implementation_closure=_closure_hook,
    validate_source_gate=validate_completed_source_gate,
    assert_fresh_score_root=_fresh_score_hook,
    make_attempt=_attempt_payload,
    fixed_authority_from_preflight=_fixed_from_preflight,
    input_payload=_input_payload_hook,
    validate_input_payload=_validate_input_hook,
    summarize_budget=summarize_budget,
    budget_gate=budget_gate,
    continue_after_budget=_continue_after_budget,
    build_score=_score_payload,
    validate_score=validate_score_payload,
    terminal_verdict=terminal_verdict,
    make_terminal=_terminal_payload,
    validate_terminal=validate_terminal_payload,
    make_failure=_failure_payload,
    validate_failure=validate_failure_payload,
)


def run_authorized_score_lifecycle(
    root: Path, *, identity: plan.ScoreIdentity, capability: object, backend: v1score.ScoreBackend,
    artifact: v1score.ArtifactRoot, official_preflight_sha256: str, root_authorization_sha256: str,
    preflight: Mapping[str, object], authorization: Mapping[str, object],
) -> Mapping[str, object]:
    """Run the shared V1 lifecycle engine with the frozen V5 hooks."""
    return v1score.run_profiled_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=official_preflight_sha256,
        root_authorization_sha256=root_authorization_sha256,
        preflight=preflight, authorization=authorization, hooks=V5_LIFECYCLE_HOOKS,
    )
