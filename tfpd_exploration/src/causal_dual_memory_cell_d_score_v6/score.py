"""V6 receipt/lineage composition over the reviewed V5 science route.

This module deliberately owns no evaluator and no copied lifecycle.  It binds
the V5 reserved-root incident, uses the shared V1 profiled lifecycle, and
reuses V5's typed independent-activity session/cell codecs and budget logic.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import plan as v5plan
from src.causal_dual_memory_cell_d_score_v5 import score as v5score

from . import plan


AUTHORITY_TOPOLOGY = v1score.AUTHORITY_TOPOLOGY
SCORE_TOPOLOGY = v1score.SCORE_TOPOLOGY


class V6ScoreError(v1score.ScoreError):
    """Fail closed for V6 route, historical-root, or receipt drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V6ScoreError(message)


def _json(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _digest(value: object) -> str:
    return plan.sha256_bytes(_json(value))


def _sha(value: object, label: str) -> str:
    try:
        return plan.require_sha(value, label)
    except plan.V6PlanError as error:
        raise V6ScoreError(str(error)) from error


def _identity_payload(identity: plan.ScoreIdentity) -> dict[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise V6ScoreError("V6 score requires its exact typed identity")
    return identity.payload()


def _v5_history_payload() -> dict[str, object]:
    return plan.V5_RESERVED_ROOT_HISTORY.payload()


def _read_v5_history(
    root: Path,
    *,
    contract: plan.HistoricalV5ReservationContract,
    require_literal_contract: bool,
) -> dict[str, object]:
    """Descriptor-check the immutable V5 authority and empty-root incident.

    The private generic form permits no-data temporary-directory tests.  Every
    production entry point below calls it only with the literal singleton,
    never a caller-supplied mapping or environment value.
    """
    if require_literal_contract and contract is not plan.V5_RESERVED_ROOT_HISTORY:
        raise V6ScoreError("V6 historical V5 reservation contract identity drift")
    payload = contract.payload()
    base = Path(root).absolute()
    authority_directory = base / contract.authority_root_relative
    score_directory = base / contract.score_root_relative
    try:
        authority_identity = v1score._directory_identity(authority_directory)
    except v1score.ScoreError as error:
        raise V6ScoreError(str(error)) from error
    authority_fd = -1
    try:
        authority_fd = os.open(authority_directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        held = os.fstat(authority_fd)
        if (
            not stat.S_ISDIR(held.st_mode)
            or (int(held.st_dev), int(held.st_ino)) != authority_identity
        ):
            raise V6ScoreError("V6 historical V5 authority directory identity drift before read")
        expected_names = set(AUTHORITY_TOPOLOGY) | {f"{name}.sha256" for name in AUTHORITY_TOPOLOGY}
        if set(os.listdir(authority_fd)) != expected_names:
            raise V6ScoreError("V6 historical V5 authority pair topology drift")
        try:
            _pre_body, preflight, pre_sha = v1score._read_unbound_0444_pair(authority_fd, "official_preflight.json")
            _auth_body, authorization, auth_sha = v1score._read_unbound_0444_pair(authority_fd, "root_authorization.json")
        except v1score.ScoreError as error:
            raise V6ScoreError(str(error)) from error
        if pre_sha != contract.preflight_sha256 or auth_sha != contract.authorization_sha256:
            raise V6ScoreError("V6 historical V5 authority body SHA drift")
        identity = preflight.get("identity") if isinstance(preflight, Mapping) else None
        closure = identity.get("closure") if isinstance(identity, Mapping) else None
        if (
            not isinstance(identity, Mapping)
            or _digest(identity) != contract.identity_sha256
            or preflight.get("schema") != "causal_dual_memory_cell_d_score_target_free_preflight_v5"
            or preflight.get("status") != "PREFLIGHT_ACCEPTED"
            or preflight.get("authority_root_relative") != contract.authority_root_relative
            or preflight.get("score_root_relative") != contract.score_root_relative
            or not isinstance(closure, Mapping)
            or closure.get("closure_sha256") != contract.implementation_closure_sha256
            or authorization.get("schema") != "causal_dual_memory_cell_d_score_root_authorization_v5"
            or authorization.get("status") != "ROOT_AUTHORIZED"
            or authorization.get("official_preflight_sha256") != pre_sha
            or authorization.get("identity_sha256") != contract.identity_sha256
            or authorization.get("authority_root_relative") != contract.authority_root_relative
            or authorization.get("score_root_relative") != contract.score_root_relative
        ):
            raise V6ScoreError("V6 historical V5 authority semantic lineage drift")
        if v1score._directory_identity(authority_directory) != authority_identity:
            raise V6ScoreError("V6 historical V5 authority directory identity drift during read")
    except V6ScoreError:
        raise
    except OSError as error:
        raise V6ScoreError("V6 historical V5 authority descriptor read failed") from error
    finally:
        if authority_fd >= 0:
            os.close(authority_fd)

    try:
        score_info = os.lstat(score_directory)
    except OSError as error:
        raise V6ScoreError("V6 historical V5 score root cannot be inspected") from error
    expected_score_identity = (contract.score_directory_device, contract.score_directory_inode)
    if (
        not stat.S_ISDIR(score_info.st_mode)
        or stat.S_ISLNK(score_info.st_mode)
        or stat.S_IMODE(score_info.st_mode) != contract.score_directory_mode
        or (int(score_info.st_dev), int(score_info.st_ino)) != expected_score_identity
    ):
        raise V6ScoreError("V6 historical V5 empty score-root identity/mode drift")
    score_fd = -1
    try:
        score_fd = os.open(score_directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        held_score = os.fstat(score_fd)
        if (
            not stat.S_ISDIR(held_score.st_mode)
            or (int(held_score.st_dev), int(held_score.st_ino)) != expected_score_identity
            or os.listdir(score_fd)
        ):
            raise V6ScoreError("V6 historical V5 score-root must retain exact empty topology")
    except V6ScoreError:
        raise
    except OSError as error:
        raise V6ScoreError("V6 historical V5 score-root descriptor read failed") from error
    finally:
        if score_fd >= 0:
            os.close(score_fd)
    try:
        after = os.lstat(score_directory)
    except OSError as error:
        raise V6ScoreError("V6 historical V5 score-root identity recheck failed") from error
    if (int(after.st_dev), int(after.st_ino)) != expected_score_identity:
        raise V6ScoreError("V6 historical V5 score-root identity drift during read")
    # The durable V6 identity binds the immutable historical facts, rather
    # than a transient descriptor-read result.  The descriptor identities
    # above are nevertheless checked before this canonical literal is
    # returned, so a replacement pair/root cannot be hidden by recomputing a
    # self-consistent payload.
    return payload


def validate_v5_reserved_root_history(root: Path) -> dict[str, object]:
    """Public production validator for the one literal historical V5 graph."""
    observed = _read_v5_history(
        Path(root), contract=plan.V5_RESERVED_ROOT_HISTORY, require_literal_contract=True,
    )
    # Keep the public binding exactly equal to the literal identity field.
    # ``_read_v5_history`` has already held/rechecked the live descriptors.
    if observed != _v5_history_payload():
        raise V6ScoreError("V6 historical V5 reservation canonical binding drift")
    return observed


def build_target_free_preflight(
    *,
    root: Path,
    identity: plan.ScoreIdentity,
    source_gate: v1score.SourceGateBinding,
    fixed_authority: v1score.FixedEvaluationAuthority | None = None,
) -> dict[str, object]:
    """Build V6 preflight by adapting the closure-bound V5 science schema."""
    _identity_payload(identity)
    if not isinstance(source_gate, v1score.SourceGateBinding) or source_gate.contract is not v5plan.SOURCE_GATE_CONTRACT:
        raise V6ScoreError("V6 preflight requires exact typed V5 source-gate binding")
    try:
        base = v5score.build_target_free_preflight(
            root=Path(root), identity=identity, source_gate=source_gate, fixed_authority=fixed_authority,
        )
    except v5score.V5ScoreError as error:
        raise V6ScoreError(str(error)) from error
    result = dict(base)
    result["schema"] = "causal_dual_memory_cell_d_score_target_free_preflight_v6"
    result["authority_root_relative"] = plan.AUTHORITY_ROOT_RELATIVE
    result["score_root_relative"] = plan.SCORE_ROOT_RELATIVE
    result["v5_reserved_root_history"] = _v5_history_payload()
    return result


def _source_binding_from_payload(value: object) -> v1score.SourceGateBinding:
    try:
        return v5score._binding_from_payload(value)
    except v5score.V5ScoreError as error:
        raise V6ScoreError(str(error)) from error


def validate_target_free_preflight(value: Mapping[str, object], identity: plan.ScoreIdentity) -> dict[str, object]:
    expected = {
        "schema", "status", "identity", "source_gate", "evaluation_authority", "metric",
        "authority_root_relative", "score_root_relative", "target_free", "target_paths_resolved",
        "model_or_checkpoint_opened", "cuda_initialized", "boundaries", "v5_reserved_root_history",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or value.get("schema") != "causal_dual_memory_cell_d_score_target_free_preflight_v6"
        or value.get("authority_root_relative") != plan.AUTHORITY_ROOT_RELATIVE
        or value.get("score_root_relative") != plan.SCORE_ROOT_RELATIVE
        or value.get("v5_reserved_root_history") != _v5_history_payload()
    ):
        raise V6ScoreError("V6 target-free preflight schema/root/history drift")
    source = _source_binding_from_payload(value.get("source_gate"))
    try:
        fixed = v1score._fixed_authority_from_payload(value.get("evaluation_authority"))
    except v1score.ScoreError as error:
        raise V6ScoreError(str(error)) from error
    expected_payload = build_target_free_preflight(
        root=Path("/v6-preflight-schema-only"), identity=identity, source_gate=source, fixed_authority=fixed,
    )
    if dict(value) != expected_payload:
        raise V6ScoreError("V6 target-free preflight canonical identity/boundary drift")
    return expected_payload


def validate_preflight_against_fixed_authorities(
    root: Path, value: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> dict[str, object]:
    checked = validate_target_free_preflight(value, identity)
    try:
        observed = v1score.derive_fixed_evaluation_authority(Path(root)).payload()
    except v1score.ScoreError as error:
        raise V6ScoreError(str(error)) from error
    if checked["evaluation_authority"] != observed:
        raise V6ScoreError("V6 durable preflight fixed evaluation authority drift")
    return checked


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    digest = _sha(official_preflight_sha256, "V6 official preflight SHA")
    if not isinstance(preflight.get("source_gate"), Mapping):
        raise V6ScoreError("V6 root authorization source-gate binding absent")
    return {
        "schema": "causal_dual_memory_cell_d_score_root_authorization_v6",
        "status": "ROOT_AUTHORIZED",
        "official_preflight_sha256": digest,
        "identity_sha256": _digest(preflight["identity"]),
        "source_gate_binding_sha256": preflight["source_gate"].get("binding_sha256"),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "target_free_preflight_required": True,
        "explicit_execution_capability_required": True,
        "v5_reserved_root_history": _v5_history_payload(),
    }


def validate_root_authorization(
    value: Mapping[str, object], *, official_preflight_sha256: str,
    preflight: Mapping[str, object], identity: plan.ScoreIdentity,
) -> dict[str, object]:
    required = {
        "schema", "status", "official_preflight_sha256", "identity_sha256", "source_gate_binding_sha256",
        "authority_root_relative", "score_root_relative", "target_free_preflight_required",
        "explicit_execution_capability_required", "v5_reserved_root_history",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise V6ScoreError("V6 root authorization schema drift")
    validate_target_free_preflight(preflight, identity)
    expected = build_root_authorization(official_preflight_sha256=official_preflight_sha256, preflight=preflight)
    if dict(value) != expected or _digest(_identity_payload(identity)) != expected["identity_sha256"]:
        raise V6ScoreError("V6 root authorization identity/preflight/history drift")
    return expected


def reserve_authority_artifact(
    root: Path, capability: object, *, identity: plan.ScoreIdentity,
) -> v1score.ArtifactRoot:
    v1score._require_root_publication_capability(capability)
    root = Path(root)
    if plan.implementation_closure(root).payload() != _identity_payload(identity)["closure"]:
        raise V6ScoreError("V6 authority reservation implementation closure drift")
    validate_v5_reserved_root_history(root)
    plan.assert_fresh_prospective_root(root, plan.AUTHORITY_ROOT_RELATIVE)
    observed = v5score.validate_completed_source_gate(root)
    if observed.payload()["body_sha256s"].get("terminal.json") != _identity_payload(identity)["source_gate"]["terminal_sha256"]:
        raise V6ScoreError("V6 authority reservation completed source-gate terminal drift")
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
    if checked["source_gate"] != v5score.validate_completed_source_gate(Path(root)).payload():
        raise V6ScoreError("V6 target-free preflight completed source-gate binding drift")
    if checked["v5_reserved_root_history"] != validate_v5_reserved_root_history(Path(root)):
        raise V6ScoreError("V6 target-free preflight historical V5 reservation drift")
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
        raise V6ScoreError("V6 durable official preflight malformed") from error
    if not isinstance(preflight, Mapping):
        raise V6ScoreError("V6 durable official preflight root drift")
    checked_preflight = validate_preflight_against_fixed_authorities(Path(root), preflight, identity=identity)
    if checked_preflight["source_gate"] != v5score.validate_completed_source_gate(Path(root)).payload():
        raise V6ScoreError("V6 root authorization completed source-gate binding drift")
    if checked_preflight["v5_reserved_root_history"] != validate_v5_reserved_root_history(Path(root)):
        raise V6ScoreError("V6 root authorization historical V5 reservation drift")
    checked = validate_root_authorization(
        payload, official_preflight_sha256=hashlib.sha256(body).hexdigest(),
        preflight=checked_preflight, identity=identity,
    )
    return artifact.publish_json("root_authorization.json", checked)


def _read_durable_authority_pair(
    root: Path, *, identity: plan.ScoreIdentity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    directory = Path(root).absolute() / plan.AUTHORITY_ROOT_RELATIVE
    try:
        named = v1score._directory_identity(directory)
    except v1score.ScoreError as error:
        raise V6ScoreError(str(error)) from error
    fd = -1
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        held = os.fstat(fd)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != named:
            raise V6ScoreError("V6 durable authority root identity drift before read")
        expected = set(AUTHORITY_TOPOLOGY) | {f"{name}.sha256" for name in AUTHORITY_TOPOLOGY}
        if set(os.listdir(fd)) != expected:
            raise V6ScoreError("V6 durable authority pair topology drift")
        try:
            _pre_body, preflight, pre_sha = v1score._read_unbound_0444_pair(fd, "official_preflight.json")
            _auth_body, authorization, auth_sha = v1score._read_unbound_0444_pair(fd, "root_authorization.json")
        except v1score.ScoreError as error:
            raise V6ScoreError(str(error)) from error
        if v1score._directory_identity(directory) != named:
            raise V6ScoreError("V6 durable authority root identity drift during read")
        validate_target_free_preflight(preflight, identity)
        validate_root_authorization(
            authorization, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity,
        )
        return dict(preflight), dict(authorization), pre_sha, auth_sha
    except V6ScoreError:
        raise
    except OSError as error:
        raise V6ScoreError("V6 durable authority descriptor read failed") from error
    finally:
        if fd >= 0:
            os.close(fd)


def load_durable_authority(
    root: Path, *, identity: plan.ScoreIdentity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    return _read_durable_authority_pair(Path(root), identity=identity)


def validate_selected_launch_environment(
    identity: plan.ScoreIdentity, environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    try:
        return v5score.validate_selected_launch_environment(identity, environ)
    except v5score.V5ScoreError as error:
        raise V6ScoreError(str(error)) from error


def issue_durable_execution_capability(
    root: Path, *, identity: plan.ScoreIdentity, root_capability: object,
    environ: Mapping[str, str] | None = None,
) -> v1score.ExecutionCapability:
    v1score._require_root_publication_capability(root_capability)
    validate_selected_launch_environment(identity, environ)
    if plan.implementation_closure(Path(root)).payload() != _identity_payload(identity)["closure"]:
        raise V6ScoreError("V6 execution issuer implementation closure drift")
    validate_v5_reserved_root_history(Path(root))
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(Path(root), identity=identity)
    validate_preflight_against_fixed_authorities(Path(root), preflight, identity=identity)
    if v5score.validate_completed_source_gate(Path(root)).payload() != preflight["source_gate"]:
        raise V6ScoreError("V6 execution issuer completed source-gate binding drift")
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
        raise V6ScoreError("V6 score-root reservation implementation closure drift")
    validate_v5_reserved_root_history(Path(root))
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(Path(root), identity=identity)
    approved = v1score.require_execution_capability(capability, identity)
    if approved.official_preflight_sha256 != pre_sha or approved.root_authorization_sha256 != auth_sha:
        raise V6ScoreError("V6 score-root reservation durable capability SHA drift")
    validate_preflight_against_fixed_authorities(Path(root), preflight, identity=identity)
    if v5score.validate_completed_source_gate(Path(root)).payload() != preflight["source_gate"]:
        raise V6ScoreError("V6 score-root reservation completed source-gate drift")
    plan.assert_fresh_prospective_root(Path(root), plan.SCORE_ROOT_RELATIVE)
    parent = Path(root).absolute() / Path(plan.SCORE_ROOT_RELATIVE).parent
    return v1score._equal_session_module(root).reserve_artifact_root(
        parent, Path(plan.SCORE_ROOT_RELATIVE).name, topology=SCORE_TOPOLOGY,
    )


def _validate_reserved_score_artifact(root: Path, artifact: v1score.ArtifactRoot, identity: Any) -> None:
    if not isinstance(identity, plan.ScoreIdentity):
        raise V6ScoreError("V6 reserved artifact identity type drift")
    try:
        v1score.validate_reserved_artifact_root(
            Path(root), artifact, root_relative=plan.SCORE_ROOT_RELATIVE, topology=SCORE_TOPOLOGY,
        )
    except v1score.ScoreError as error:
        raise V6ScoreError(str(error)) from error


def _v5_score_body(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    history = body.pop("v5_reserved_root_history", None)
    if history != _v5_history_payload():
        raise V6ScoreError("V6 score historical V5 reservation binding drift")
    body["schema"] = "causal_dual_memory_cell_d_matched_score_v5"
    return body


def _score_payload(
    identity: plan.ScoreIdentity, input_sha: str,
    summaries: Mapping[str, object], gates: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    try:
        base = v5score._score_payload(identity, input_sha, summaries, gates)
    except v5score.V5ScoreError as error:
        raise V6ScoreError(str(error)) from error
    result = dict(base)
    result["schema"] = "causal_dual_memory_cell_d_matched_score_v6"
    result["v5_reserved_root_history"] = _v5_history_payload()
    return result


def validate_score_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, input_payload: Mapping[str, object],
    input_authority_sha256: str, evaluation_authority: v1score.FixedEvaluationAuthority,
) -> dict[str, object]:
    required = {
        "schema", "identity", "input_authority_sha256", "budget_summaries", "budget_gates",
        "budget_execution_order", "cell_execution_order", "matrix_policy",
        "target_optimizer_backward_update", "source_gate_terminal_sha256", "v5_reserved_root_history",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "causal_dual_memory_cell_d_matched_score_v6":
        raise V6ScoreError("V6 score receipt schema drift")
    base = _v5_score_body(value)
    try:
        checked = v5score.validate_score_payload(
            base, identity, input_payload, input_authority_sha256, evaluation_authority,
        )
    except (v5score.V5ScoreError, v1score.ScoreError) as error:
        raise V6ScoreError(str(error)) from error
    expected = dict(checked)
    expected["schema"] = "causal_dual_memory_cell_d_matched_score_v6"
    expected["v5_reserved_root_history"] = _v5_history_payload()
    if dict(value) != expected:
        raise V6ScoreError("V6 score receipt canonical/history drift")
    return expected


def _v5_terminal_body(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    history = body.pop("v5_reserved_root_history", None)
    if history != _v5_history_payload():
        raise V6ScoreError("V6 terminal historical V5 reservation binding drift")
    body["schema"] = "causal_dual_memory_cell_d_score_terminal_v5"
    return body


def _terminal_payload(
    identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str, score_sha: str,
    gates: Mapping[int, Mapping[str, object]], verdict: str,
) -> dict[str, object]:
    try:
        base = v5score._terminal_payload(identity, attempt_sha, input_sha, score_sha, gates, verdict)
    except v5score.V5ScoreError as error:
        raise V6ScoreError(str(error)) from error
    result = dict(base)
    result["schema"] = "causal_dual_memory_cell_d_score_terminal_v6"
    result["v5_reserved_root_history"] = _v5_history_payload()
    return result


def validate_terminal_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str,
    score_payload: Mapping[str, object], score_sha: str,
) -> dict[str, object]:
    required = {
        "schema", "status", "verdict", "identity", "attempt_sha256", "input_authority_sha256", "score_sha256",
        "budget_gates", "launch_closure_sha256", "final_closure_sha256", "source_gate_terminal_sha256",
        "target_optimizer_backward_update", "complete_matrix", "v5_reserved_root_history",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "causal_dual_memory_cell_d_score_terminal_v6":
        raise V6ScoreError("V6 terminal schema drift")
    base_terminal = _v5_terminal_body(value)
    base_score = _v5_score_body(score_payload)
    try:
        checked = v5score.validate_terminal_payload(
            base_terminal, identity, attempt_sha, input_sha, base_score, score_sha,
        )
    except v5score.V5ScoreError as error:
        raise V6ScoreError(str(error)) from error
    expected = dict(checked)
    expected["schema"] = "causal_dual_memory_cell_d_score_terminal_v6"
    expected["v5_reserved_root_history"] = _v5_history_payload()
    if dict(value) != expected:
        raise V6ScoreError("V6 terminal canonical/history drift")
    return expected


def _attempt_payload(identity: plan.ScoreIdentity, pre_sha: str, auth_sha: str) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_score_attempt_v6",
        "status": "ATTEMPT_RESERVED",
        "identity": _identity_payload(identity),
        "preflight_sha256": _sha(pre_sha, "V6 attempt preflight SHA"),
        "authorization_sha256": _sha(auth_sha, "V6 attempt authorization SHA"),
        "target_paths_resolved_or_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
        "v5_reserved_root_history": _v5_history_payload(),
    }


def _failure_payload(
    identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str | None, stage: str,
    error: BaseException, runtime_progress: object | None,
) -> dict[str, object]:
    try:
        base = v5score._failure_payload(identity, attempt_sha, input_sha, stage, error, runtime_progress)
    except (v5score.V5ScoreError, v1score.ScoreError) as error_value:
        raise V6ScoreError(str(error_value)) from error_value
    result = dict(base)
    result["schema"] = "causal_dual_memory_cell_d_score_failure_v6"
    result["v5_reserved_root_history"] = _v5_history_payload()
    return result


def validate_failure_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str | None,
) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "attempt_sha256", "input_authority_sha256", "stage", "error_class",
        "error_sha256", "target_paths_resolved_or_opened", "within_assets_opened", "external_assets_opened",
        "checkpoint_opened", "cuda_initialized", "full_system_forward_count", "group_forward_count",
        "terminal_published", "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        "source_gate_terminal_sha256", "v5_reserved_root_history",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "causal_dual_memory_cell_d_score_failure_v6":
        raise V6ScoreError("V6 failure schema drift")
    base = dict(value)
    history = base.pop("v5_reserved_root_history", None)
    if history != _v5_history_payload():
        raise V6ScoreError("V6 failure historical V5 reservation binding drift")
    base["schema"] = "causal_dual_memory_cell_d_score_failure_v5"
    try:
        checked = v5score.validate_failure_payload(base, identity, attempt_sha, input_sha)
    except v5score.V5ScoreError as error:
        raise V6ScoreError(str(error)) from error
    expected = dict(checked)
    expected["schema"] = "causal_dual_memory_cell_d_score_failure_v6"
    expected["v5_reserved_root_history"] = _v5_history_payload()
    if dict(value) != expected:
        raise V6ScoreError("V6 failure canonical/history drift")
    return expected


def _validate_preflight_hook(value: Mapping[str, object], identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise V6ScoreError("V6 lifecycle identity type drift")
    return validate_target_free_preflight(value, identity)


def _validate_authorization_hook(
    value: Mapping[str, object], pre_sha: str, preflight: Mapping[str, object], identity: Any,
) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise V6ScoreError("V6 lifecycle identity type drift")
    return validate_root_authorization(value, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity)


def _closure_hook(root: Path) -> Mapping[str, object]:
    return plan.implementation_closure(root).payload()


def _fresh_score_hook(root: Path) -> None:
    # V6 supplies the reserved-artifact hook below, so this historical fallback
    # is not called from the V6 literal lifecycle hooks.
    plan.assert_fresh_prospective_root(root, plan.SCORE_ROOT_RELATIVE)


def _input_payload_hook(authority: Any, identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity) or not isinstance(authority, v1score.InputAuthority):
        raise V6ScoreError("V6 physical route must return inherited typed input authority")
    return authority.payload(identity=identity)


def _validate_input_hook(value: Mapping[str, object], identity: Any, fixed: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity) or not isinstance(fixed, v1score.FixedEvaluationAuthority):
        raise V6ScoreError("V6 fixed evaluation authority type drift")
    try:
        return v1score.validate_input_authority_payload(value, identity=identity, evaluation_authority=fixed)
    except v1score.ScoreError as error:
        raise V6ScoreError(str(error)) from error


def _fixed_from_preflight(preflight: Mapping[str, object]) -> Any:
    try:
        return v1score._fixed_authority_from_payload(preflight["evaluation_authority"])
    except v1score.ScoreError as error:
        raise V6ScoreError(str(error)) from error


def _continue_after_budget(_budget: int, _gates: Mapping[int, Mapping[str, object]]) -> bool:
    return True


V6_LIFECYCLE_HOOKS = v1score.ProfiledLifecycleHooks(
    route="causal_dual_memory_cell_d_score_v6",
    score_root_relative=plan.SCORE_ROOT_RELATIVE,
    budgets=plan.BUDGETS,
    require_capability=v1score.require_execution_capability,
    validate_preflight=_validate_preflight_hook,
    validate_authorization=_validate_authorization_hook,
    implementation_closure=_closure_hook,
    validate_source_gate=v5score.validate_completed_source_gate,
    assert_fresh_score_root=_fresh_score_hook,
    make_attempt=_attempt_payload,
    fixed_authority_from_preflight=_fixed_from_preflight,
    input_payload=_input_payload_hook,
    validate_input_payload=_validate_input_hook,
    summarize_budget=v5score.summarize_budget,
    budget_gate=v5score.budget_gate,
    continue_after_budget=_continue_after_budget,
    build_score=_score_payload,
    validate_score=validate_score_payload,
    terminal_verdict=v5score.terminal_verdict,
    make_terminal=_terminal_payload,
    validate_terminal=validate_terminal_payload,
    make_failure=_failure_payload,
    validate_failure=validate_failure_payload,
    validate_reserved_score_artifact=_validate_reserved_score_artifact,
)


def run_authorized_score_lifecycle(
    root: Path, *, identity: plan.ScoreIdentity, capability: object, backend: v1score.ScoreBackend,
    artifact: v1score.ArtifactRoot, official_preflight_sha256: str, root_authorization_sha256: str,
    preflight: Mapping[str, object], authorization: Mapping[str, object],
) -> Mapping[str, object]:
    """Run the sole shared lifecycle with the V6 reserved-root hook."""
    return v1score.run_profiled_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=official_preflight_sha256,
        root_authorization_sha256=root_authorization_sha256,
        preflight=preflight, authorization=authorization, hooks=V6_LIFECYCLE_HOOKS,
    )
