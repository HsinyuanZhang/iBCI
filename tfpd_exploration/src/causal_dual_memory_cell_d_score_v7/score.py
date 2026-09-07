"""V7 receipt and lifecycle composition for the RuntimeFlags successor.

V7 deliberately delegates all score science, cell codecs, gates, and the
shared immutable lifecycle to V5/V6/V1.  This module owns only fresh receipt
schemas and descriptor-held proof that the exact V6 authority plus its honest
post-attempt RuntimeFlags failure preceded any V7 authority or score root.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import score as v5score
from src.causal_dual_memory_cell_d_score_v6 import plan as v6plan
from src.causal_dual_memory_cell_d_score_v6 import score as v6score

from . import plan


AUTHORITY_TOPOLOGY = v1score.AUTHORITY_TOPOLOGY
SCORE_TOPOLOGY = v1score.SCORE_TOPOLOGY


class V7ScoreError(v1score.ScoreError):
    """Fail closed for V7 lineage, durable authority, or receipt drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V7ScoreError(message)


def _json(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _digest(value: object) -> str:
    return plan.sha256_bytes(_json(value))


def _sha(value: object, label: str) -> str:
    try:
        return plan.require_sha(value, label)
    except plan.V7PlanError as error:
        raise V7ScoreError(str(error)) from error


def _identity_payload(identity: plan.ScoreIdentity) -> dict[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise V7ScoreError("V7 score requires its exact typed identity")
    return identity.payload()


def _v6_failure_payload() -> dict[str, object]:
    return plan.V6_RUNTIME_FLAGS_FAILURE.payload()


def _read_exact_pair(fd: int, name: str) -> tuple[bytes, dict[str, object], str]:
    try:
        return v1score._read_unbound_0444_pair(fd, name)
    except v1score.ScoreError as error:
        raise V7ScoreError(str(error)) from error


def _v6_identity_from_preflight(preflight: Mapping[str, object]) -> v6plan.ScoreIdentity:
    identity_payload = preflight.get("identity")
    closure = identity_payload.get("closure") if isinstance(identity_payload, Mapping) else None
    if not isinstance(closure, Mapping):
        raise V7ScoreError("V7 V6 predecessor identity closure is absent")
    try:
        identity = v6plan.ScoreIdentity(closure=closure)
    except (TypeError, v6plan.V6PlanError) as error:
        raise V7ScoreError("V7 V6 predecessor identity reconstruction failed") from error
    if identity.payload() != identity_payload:
        raise V7ScoreError("V7 V6 predecessor identity canonical drift")
    return identity


def _read_v6_runtime_flags_failure(
    root: Path,
    *,
    contract: plan.V6RuntimeFlagsFailureContract,
    require_literal_contract: bool,
) -> dict[str, object]:
    """Hold and validate V6 authority plus its exact four-leaf failure graph.

    The generic contract form is test-only.  All public V7 issuer/reservation
    entry points below pass the one literal :data:`V6_RUNTIME_FLAGS_FAILURE`
    object, so neither an environment variable nor a caller mapping can point
    a V7 score at an alternative predecessor.
    """
    if require_literal_contract and contract is not plan.V6_RUNTIME_FLAGS_FAILURE:
        raise V7ScoreError("V7 V6 RuntimeFlags predecessor contract identity drift")
    payload = contract.payload()
    base = Path(root).absolute()
    authority_directory = base / contract.authority_root_relative
    failed_directory = base / contract.score_root_relative

    try:
        authority_identity = v1score._directory_identity(authority_directory)
    except v1score.ScoreError as error:
        raise V7ScoreError(str(error)) from error
    authority_fd = -1
    try:
        authority_fd = os.open(authority_directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        held = os.fstat(authority_fd)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != authority_identity:
            raise V7ScoreError("V7 V6 authority directory identity drift before read")
        expected = set(AUTHORITY_TOPOLOGY) | {f"{name}.sha256" for name in AUTHORITY_TOPOLOGY}
        if set(os.listdir(authority_fd)) != expected:
            raise V7ScoreError("V7 V6 authority pair topology drift")
        _pre_body, preflight, pre_sha = _read_exact_pair(authority_fd, "official_preflight.json")
        _auth_body, authorization, auth_sha = _read_exact_pair(authority_fd, "root_authorization.json")
        if pre_sha != contract.preflight_sha256 or auth_sha != contract.authorization_sha256:
            raise V7ScoreError("V7 V6 authority body SHA drift")
        identity = _v6_identity_from_preflight(preflight)
        try:
            checked_preflight = v6score.validate_target_free_preflight(preflight, identity)
            checked_auth = v6score.validate_root_authorization(
                authorization, official_preflight_sha256=pre_sha, preflight=checked_preflight, identity=identity,
            )
        except v6score.V6ScoreError as error:
            raise V7ScoreError(str(error)) from error
        identity_payload = identity.payload()
        if (
            _digest(identity_payload) != contract.identity_sha256
            or checked_preflight.get("authority_root_relative") != contract.authority_root_relative
            or checked_preflight.get("score_root_relative") != contract.score_root_relative
            or identity_payload.get("closure", {}).get("closure_sha256") != contract.implementation_closure_sha256
            or checked_auth.get("official_preflight_sha256") != pre_sha
            or checked_auth.get("identity_sha256") != contract.identity_sha256
        ):
            raise V7ScoreError("V7 V6 authority semantic lineage drift")
        if v1score._directory_identity(authority_directory) != authority_identity:
            raise V7ScoreError("V7 V6 authority directory identity drift during read")
    except V7ScoreError:
        raise
    except OSError as error:
        raise V7ScoreError("V7 V6 authority descriptor read failed") from error
    finally:
        if authority_fd >= 0:
            os.close(authority_fd)

    try:
        failed_named = os.lstat(failed_directory)
    except OSError as error:
        raise V7ScoreError("V7 V6 failed score root cannot be inspected") from error
    expected_identity = (contract.score_directory_device, contract.score_directory_inode)
    if (
        not stat.S_ISDIR(failed_named.st_mode)
        or stat.S_ISLNK(failed_named.st_mode)
        or stat.S_IMODE(failed_named.st_mode) != contract.score_directory_mode
        or (int(failed_named.st_dev), int(failed_named.st_ino)) != expected_identity
    ):
        raise V7ScoreError("V7 V6 failed score-root identity/mode drift")
    failed_fd = -1
    try:
        failed_fd = os.open(failed_directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        held = os.fstat(failed_fd)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != expected_identity:
            raise V7ScoreError("V7 V6 failed score-root identity drift before read")
        expected_names = {"attempt.json", "failure.json", "attempt.json.sha256", "failure.json.sha256"}
        if set(os.listdir(failed_fd)) != expected_names:
            raise V7ScoreError("V7 V6 failed score-root exact four-leaf topology drift")
        _attempt_body, attempt, attempt_sha = _read_exact_pair(failed_fd, "attempt.json")
        _failure_body, failure, failure_sha = _read_exact_pair(failed_fd, "failure.json")
        if attempt_sha != contract.attempt_sha256 or failure_sha != contract.failure_sha256:
            raise V7ScoreError("V7 V6 failed score body SHA drift")
        expected_attempt = v6score._attempt_payload(identity, contract.preflight_sha256, contract.authorization_sha256)
        if attempt != expected_attempt:
            raise V7ScoreError("V7 V6 failed score attempt semantic drift")
        try:
            checked_failure = v6score.validate_failure_payload(failure, identity, attempt_sha, None)
        except v6score.V6ScoreError as error:
            raise V7ScoreError(str(error)) from error
        facts = payload["failure"]
        if (
            checked_failure.get("stage") != facts["stage"]
            or checked_failure.get("error_class") != facts["error_class"]
            or checked_failure.get("error_sha256") != facts["error_sha256"]
            or checked_failure.get("checkpoint_opened") is not facts["sealed_checkpoint_opened"]
            or bool(checked_failure.get("within_assets_opened"))
            or bool(checked_failure.get("external_assets_opened"))
            or checked_failure.get("cuda_initialized") is not facts["cuda_initialized"]
            or checked_failure.get("full_system_forward_count") != facts["full_system_forward_count"]
            or checked_failure.get("group_forward_count") != facts["group_forward_count"]
            or checked_failure.get("input_authority_sha256") is not None
            or checked_failure.get("terminal_published") is not facts["terminal_published"]
            or any(checked_failure.get(key) != 0 for key in (
                "target_optimizer_steps", "target_backward_calls", "target_update_calls",
            ))
        ):
            raise V7ScoreError("V7 V6 failed score access/boundary facts drift")
        after = os.lstat(failed_directory)
        if (int(after.st_dev), int(after.st_ino)) != expected_identity:
            raise V7ScoreError("V7 V6 failed score-root identity drift during read")
    except V7ScoreError:
        raise
    except OSError as error:
        raise V7ScoreError("V7 V6 failed score descriptor read failed") from error
    finally:
        if failed_fd >= 0:
            os.close(failed_fd)
    return payload


def validate_v6_runtime_flags_predecessor(root: Path) -> dict[str, object]:
    """Public production validator for the literal V6 failed execution graph."""
    observed = _read_v6_runtime_flags_failure(
        Path(root), contract=plan.V6_RUNTIME_FLAGS_FAILURE, require_literal_contract=True,
    )
    if observed != _v6_failure_payload():
        raise V7ScoreError("V7 V6 RuntimeFlags predecessor canonical binding drift")
    # V6 itself was a successor.  Retain its two immutable transitive proofs;
    # neither one is caller-configurable and both are descriptor-held by V6.
    try:
        v6score.validate_v5_reserved_root_history(Path(root))
        v5score.validate_completed_source_gate(Path(root))
    except (v6score.V6ScoreError, v5score.V5ScoreError) as error:
        raise V7ScoreError(str(error)) from error
    return observed


def build_target_free_preflight(
    *, root: Path, identity: plan.ScoreIdentity, source_gate: v1score.SourceGateBinding,
    fixed_authority: v1score.FixedEvaluationAuthority | None = None,
) -> dict[str, object]:
    """Adapt the V6 target-free payload without changing science authority."""
    _identity_payload(identity)
    try:
        base = v6score.build_target_free_preflight(
            root=Path(root), identity=identity, source_gate=source_gate, fixed_authority=fixed_authority,
        )
    except v6score.V6ScoreError as error:
        raise V7ScoreError(str(error)) from error
    result = dict(base)
    result["schema"] = "causal_dual_memory_cell_d_score_target_free_preflight_v7"
    result["authority_root_relative"] = plan.AUTHORITY_ROOT_RELATIVE
    result["score_root_relative"] = plan.SCORE_ROOT_RELATIVE
    result["v6_runtime_flags_failure"] = _v6_failure_payload()
    return result


def _v6_preflight_body(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    predecessor = body.pop("v6_runtime_flags_failure", None)
    if predecessor != _v6_failure_payload():
        raise V7ScoreError("V7 preflight V6 RuntimeFlags predecessor binding drift")
    body["schema"] = "causal_dual_memory_cell_d_score_target_free_preflight_v6"
    body["authority_root_relative"] = v6plan.AUTHORITY_ROOT_RELATIVE
    body["score_root_relative"] = v6plan.SCORE_ROOT_RELATIVE
    return body


def validate_target_free_preflight(value: Mapping[str, object], identity: plan.ScoreIdentity) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "source_gate", "evaluation_authority", "metric",
        "authority_root_relative", "score_root_relative", "target_free", "target_paths_resolved",
        "model_or_checkpoint_opened", "cuda_initialized", "boundaries", "v5_reserved_root_history",
        "v6_runtime_flags_failure",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "causal_dual_memory_cell_d_score_target_free_preflight_v7"
        or value.get("authority_root_relative") != plan.AUTHORITY_ROOT_RELATIVE
        or value.get("score_root_relative") != plan.SCORE_ROOT_RELATIVE
    ):
        raise V7ScoreError("V7 target-free preflight schema/root drift")
    try:
        checked_v6 = v6score.validate_target_free_preflight(_v6_preflight_body(value), identity)
    except v6score.V6ScoreError as error:
        raise V7ScoreError(str(error)) from error
    expected = dict(checked_v6)
    expected["schema"] = "causal_dual_memory_cell_d_score_target_free_preflight_v7"
    expected["authority_root_relative"] = plan.AUTHORITY_ROOT_RELATIVE
    expected["score_root_relative"] = plan.SCORE_ROOT_RELATIVE
    expected["v6_runtime_flags_failure"] = _v6_failure_payload()
    if dict(value) != expected:
        raise V7ScoreError("V7 target-free preflight canonical predecessor drift")
    return expected


def validate_preflight_against_fixed_authorities(
    root: Path, value: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> dict[str, object]:
    checked = validate_target_free_preflight(value, identity)
    try:
        observed = v1score.derive_fixed_evaluation_authority(Path(root)).payload()
    except v1score.ScoreError as error:
        raise V7ScoreError(str(error)) from error
    if checked["evaluation_authority"] != observed:
        raise V7ScoreError("V7 durable preflight fixed evaluation authority drift")
    return checked


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    digest = _sha(official_preflight_sha256, "V7 official preflight SHA")
    if not isinstance(preflight.get("source_gate"), Mapping):
        raise V7ScoreError("V7 root authorization source-gate binding absent")
    return {
        "schema": "causal_dual_memory_cell_d_score_root_authorization_v7",
        "status": "ROOT_AUTHORIZED",
        "official_preflight_sha256": digest,
        "identity_sha256": _digest(preflight["identity"]),
        "source_gate_binding_sha256": preflight["source_gate"].get("binding_sha256"),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "target_free_preflight_required": True,
        "explicit_execution_capability_required": True,
        "v5_reserved_root_history": v6plan.V5_RESERVED_ROOT_HISTORY.payload(),
        "v6_runtime_flags_failure": _v6_failure_payload(),
    }


def _v6_authorization_body(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    predecessor = body.pop("v6_runtime_flags_failure", None)
    if predecessor != _v6_failure_payload():
        raise V7ScoreError("V7 authorization V6 RuntimeFlags predecessor binding drift")
    body["schema"] = "causal_dual_memory_cell_d_score_root_authorization_v6"
    body["authority_root_relative"] = v6plan.AUTHORITY_ROOT_RELATIVE
    body["score_root_relative"] = v6plan.SCORE_ROOT_RELATIVE
    return body


def validate_root_authorization(
    value: Mapping[str, object], *, official_preflight_sha256: str,
    preflight: Mapping[str, object], identity: plan.ScoreIdentity,
) -> dict[str, object]:
    required = {
        "schema", "status", "official_preflight_sha256", "identity_sha256", "source_gate_binding_sha256",
        "authority_root_relative", "score_root_relative", "target_free_preflight_required",
        "explicit_execution_capability_required", "v5_reserved_root_history", "v6_runtime_flags_failure",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise V7ScoreError("V7 root authorization schema drift")
    checked_preflight = validate_target_free_preflight(preflight, identity)
    expected = build_root_authorization(
        official_preflight_sha256=official_preflight_sha256, preflight=checked_preflight,
    )
    if dict(value) != expected or _digest(_identity_payload(identity)) != expected["identity_sha256"]:
        raise V7ScoreError("V7 root authorization canonical identity/predecessor drift")
    return expected


def reserve_authority_artifact(root: Path, capability: object, *, identity: plan.ScoreIdentity) -> v1score.ArtifactRoot:
    v1score._require_root_publication_capability(capability)
    base = Path(root)
    if plan.implementation_closure(base).payload() != _identity_payload(identity)["closure"]:
        raise V7ScoreError("V7 authority reservation implementation closure drift")
    validate_v6_runtime_flags_predecessor(base)
    plan.assert_fresh_prospective_root(base, plan.AUTHORITY_ROOT_RELATIVE)
    observed = v5score.validate_completed_source_gate(base)
    if observed.payload()["body_sha256s"].get("terminal.json") != _identity_payload(identity)["source_gate"]["terminal_sha256"]:
        raise V7ScoreError("V7 authority reservation completed source-gate terminal drift")
    parent = base.absolute() / Path(plan.AUTHORITY_ROOT_RELATIVE).parent
    return v1score._equal_session_module(base).reserve_artifact_root(
        parent, Path(plan.AUTHORITY_ROOT_RELATIVE).name, topology=AUTHORITY_TOPOLOGY,
    )


def publish_target_free_preflight(
    root: Path, artifact: v1score.ArtifactRoot, capability: object,
    payload: Mapping[str, object], *, identity: plan.ScoreIdentity,
) -> str:
    v1score._require_root_publication_capability(capability)
    checked = validate_preflight_against_fixed_authorities(Path(root), payload, identity=identity)
    if checked["source_gate"] != v5score.validate_completed_source_gate(Path(root)).payload():
        raise V7ScoreError("V7 durable preflight completed source-gate binding drift")
    if checked["v6_runtime_flags_failure"] != validate_v6_runtime_flags_predecessor(Path(root)):
        raise V7ScoreError("V7 durable preflight V6 RuntimeFlags predecessor drift")
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
        raise V7ScoreError("V7 durable official preflight malformed") from error
    if not isinstance(preflight, Mapping):
        raise V7ScoreError("V7 durable official preflight root drift")
    checked = validate_preflight_against_fixed_authorities(Path(root), preflight, identity=identity)
    if checked["source_gate"] != v5score.validate_completed_source_gate(Path(root)).payload():
        raise V7ScoreError("V7 root authorization completed source-gate binding drift")
    if checked["v6_runtime_flags_failure"] != validate_v6_runtime_flags_predecessor(Path(root)):
        raise V7ScoreError("V7 root authorization V6 RuntimeFlags predecessor drift")
    authorization = validate_root_authorization(
        payload, official_preflight_sha256=hashlib.sha256(body).hexdigest(), preflight=checked, identity=identity,
    )
    return artifact.publish_json("root_authorization.json", authorization)


def _read_durable_authority_pair(
    root: Path, *, identity: plan.ScoreIdentity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    directory = Path(root).absolute() / plan.AUTHORITY_ROOT_RELATIVE
    try:
        named = v1score._directory_identity(directory)
    except v1score.ScoreError as error:
        raise V7ScoreError(str(error)) from error
    fd = -1
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        held = os.fstat(fd)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != named:
            raise V7ScoreError("V7 durable authority root identity drift before read")
        expected = set(AUTHORITY_TOPOLOGY) | {f"{name}.sha256" for name in AUTHORITY_TOPOLOGY}
        if set(os.listdir(fd)) != expected:
            raise V7ScoreError("V7 durable authority pair topology drift")
        _pre_body, preflight, pre_sha = _read_exact_pair(fd, "official_preflight.json")
        _auth_body, authorization, auth_sha = _read_exact_pair(fd, "root_authorization.json")
        if v1score._directory_identity(directory) != named:
            raise V7ScoreError("V7 durable authority root identity drift during read")
        checked = validate_target_free_preflight(preflight, identity)
        validate_root_authorization(
            authorization, official_preflight_sha256=pre_sha, preflight=checked, identity=identity,
        )
        return dict(checked), dict(authorization), pre_sha, auth_sha
    except V7ScoreError:
        raise
    except OSError as error:
        raise V7ScoreError("V7 durable authority descriptor read failed") from error
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
        return v6score.validate_selected_launch_environment(identity, environ)
    except v6score.V6ScoreError as error:
        raise V7ScoreError(str(error)) from error


def issue_durable_execution_capability(
    root: Path, *, identity: plan.ScoreIdentity, root_capability: object,
    environ: Mapping[str, str] | None = None,
) -> v1score.ExecutionCapability:
    v1score._require_root_publication_capability(root_capability)
    validate_selected_launch_environment(identity, environ)
    base = Path(root)
    if plan.implementation_closure(base).payload() != _identity_payload(identity)["closure"]:
        raise V7ScoreError("V7 execution issuer implementation closure drift")
    validate_v6_runtime_flags_predecessor(base)
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(base, identity=identity)
    validate_preflight_against_fixed_authorities(base, preflight, identity=identity)
    if v5score.validate_completed_source_gate(base).payload() != preflight["source_gate"]:
        raise V7ScoreError("V7 execution issuer completed source-gate binding drift")
    plan.assert_fresh_prospective_root(base, plan.SCORE_ROOT_RELATIVE)
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
    base = Path(root)
    if plan.implementation_closure(base).payload() != _identity_payload(identity)["closure"]:
        raise V7ScoreError("V7 score-root reservation implementation closure drift")
    validate_v6_runtime_flags_predecessor(base)
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(base, identity=identity)
    approved = v1score.require_execution_capability(capability, identity)
    if approved.official_preflight_sha256 != pre_sha or approved.root_authorization_sha256 != auth_sha:
        raise V7ScoreError("V7 score-root reservation durable capability SHA drift")
    validate_preflight_against_fixed_authorities(base, preflight, identity=identity)
    if v5score.validate_completed_source_gate(base).payload() != preflight["source_gate"]:
        raise V7ScoreError("V7 score-root reservation completed source-gate drift")
    plan.assert_fresh_prospective_root(base, plan.SCORE_ROOT_RELATIVE)
    parent = base.absolute() / Path(plan.SCORE_ROOT_RELATIVE).parent
    return v1score._equal_session_module(base).reserve_artifact_root(
        parent, Path(plan.SCORE_ROOT_RELATIVE).name, topology=SCORE_TOPOLOGY,
    )


def _validate_reserved_score_artifact(root: Path, artifact: v1score.ArtifactRoot, identity: Any) -> None:
    if not isinstance(identity, plan.ScoreIdentity):
        raise V7ScoreError("V7 reserved artifact identity type drift")
    try:
        v1score.validate_reserved_artifact_root(
            Path(root), artifact, root_relative=plan.SCORE_ROOT_RELATIVE, topology=SCORE_TOPOLOGY,
        )
    except v1score.ScoreError as error:
        raise V7ScoreError(str(error)) from error


def _v6_score_body(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    predecessor = body.pop("v6_runtime_flags_failure", None)
    if predecessor != _v6_failure_payload():
        raise V7ScoreError("V7 score V6 RuntimeFlags predecessor binding drift")
    body["schema"] = "causal_dual_memory_cell_d_matched_score_v6"
    return body


def _score_payload(
    identity: plan.ScoreIdentity, input_sha: str,
    summaries: Mapping[str, object], gates: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    try:
        base = v6score._score_payload(identity, input_sha, summaries, gates)
    except v6score.V6ScoreError as error:
        raise V7ScoreError(str(error)) from error
    result = dict(base)
    result["schema"] = "causal_dual_memory_cell_d_matched_score_v7"
    result["v6_runtime_flags_failure"] = _v6_failure_payload()
    return result


def validate_score_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, input_payload: Mapping[str, object],
    input_authority_sha256: str, evaluation_authority: v1score.FixedEvaluationAuthority,
) -> dict[str, object]:
    required = {
        "schema", "identity", "input_authority_sha256", "budget_summaries", "budget_gates",
        "budget_execution_order", "cell_execution_order", "matrix_policy",
        "target_optimizer_backward_update", "source_gate_terminal_sha256", "v5_reserved_root_history",
        "v6_runtime_flags_failure",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "causal_dual_memory_cell_d_matched_score_v7":
        raise V7ScoreError("V7 score receipt schema drift")
    try:
        checked_v6 = v6score.validate_score_payload(
            _v6_score_body(value), identity, input_payload, input_authority_sha256, evaluation_authority,
        )
    except (v6score.V6ScoreError, v1score.ScoreError) as error:
        raise V7ScoreError(str(error)) from error
    expected = dict(checked_v6)
    expected["schema"] = "causal_dual_memory_cell_d_matched_score_v7"
    expected["v6_runtime_flags_failure"] = _v6_failure_payload()
    if dict(value) != expected:
        raise V7ScoreError("V7 score receipt canonical predecessor drift")
    return expected


def _v6_terminal_body(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    predecessor = body.pop("v6_runtime_flags_failure", None)
    if predecessor != _v6_failure_payload():
        raise V7ScoreError("V7 terminal V6 RuntimeFlags predecessor binding drift")
    body["schema"] = "causal_dual_memory_cell_d_score_terminal_v6"
    return body


def _terminal_payload(
    identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str, score_sha: str,
    gates: Mapping[int, Mapping[str, object]], verdict: str,
) -> dict[str, object]:
    try:
        base = v6score._terminal_payload(identity, attempt_sha, input_sha, score_sha, gates, verdict)
    except v6score.V6ScoreError as error:
        raise V7ScoreError(str(error)) from error
    result = dict(base)
    result["schema"] = "causal_dual_memory_cell_d_score_terminal_v7"
    result["v6_runtime_flags_failure"] = _v6_failure_payload()
    return result


def validate_terminal_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str,
    score_payload: Mapping[str, object], score_sha: str,
) -> dict[str, object]:
    required = {
        "schema", "status", "verdict", "identity", "attempt_sha256", "input_authority_sha256", "score_sha256",
        "budget_gates", "launch_closure_sha256", "final_closure_sha256", "source_gate_terminal_sha256",
        "target_optimizer_backward_update", "complete_matrix", "v5_reserved_root_history", "v6_runtime_flags_failure",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "causal_dual_memory_cell_d_score_terminal_v7":
        raise V7ScoreError("V7 terminal schema drift")
    try:
        checked_v6 = v6score.validate_terminal_payload(
            _v6_terminal_body(value), identity, attempt_sha, input_sha, _v6_score_body(score_payload), score_sha,
        )
    except v6score.V6ScoreError as error:
        raise V7ScoreError(str(error)) from error
    expected = dict(checked_v6)
    expected["schema"] = "causal_dual_memory_cell_d_score_terminal_v7"
    expected["v6_runtime_flags_failure"] = _v6_failure_payload()
    if dict(value) != expected:
        raise V7ScoreError("V7 terminal canonical predecessor drift")
    return expected


def _attempt_payload(identity: plan.ScoreIdentity, pre_sha: str, auth_sha: str) -> dict[str, object]:
    try:
        base = v6score._attempt_payload(identity, pre_sha, auth_sha)
    except v6score.V6ScoreError as error:
        raise V7ScoreError(str(error)) from error
    result = dict(base)
    result["schema"] = "causal_dual_memory_cell_d_score_attempt_v7"
    result["v6_runtime_flags_failure"] = _v6_failure_payload()
    return result


def _v6_failure_body(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    predecessor = body.pop("v6_runtime_flags_failure", None)
    if predecessor != _v6_failure_payload():
        raise V7ScoreError("V7 failure V6 RuntimeFlags predecessor binding drift")
    body["schema"] = "causal_dual_memory_cell_d_score_failure_v6"
    return body


def _failure_payload(
    identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str | None, stage: str,
    error: BaseException, runtime_progress: object | None,
) -> dict[str, object]:
    try:
        base = v6score._failure_payload(identity, attempt_sha, input_sha, stage, error, runtime_progress)
    except (v6score.V6ScoreError, v1score.ScoreError) as error_value:
        raise V7ScoreError(str(error_value)) from error_value
    result = dict(base)
    result["schema"] = "causal_dual_memory_cell_d_score_failure_v7"
    result["v6_runtime_flags_failure"] = _v6_failure_payload()
    return result


def validate_failure_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str | None,
) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "attempt_sha256", "input_authority_sha256", "stage", "error_class",
        "error_sha256", "target_paths_resolved_or_opened", "within_assets_opened", "external_assets_opened",
        "checkpoint_opened", "cuda_initialized", "full_system_forward_count", "group_forward_count",
        "terminal_published", "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        "source_gate_terminal_sha256", "v5_reserved_root_history", "v6_runtime_flags_failure",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "causal_dual_memory_cell_d_score_failure_v7":
        raise V7ScoreError("V7 failure schema drift")
    try:
        checked_v6 = v6score.validate_failure_payload(_v6_failure_body(value), identity, attempt_sha, input_sha)
    except v6score.V6ScoreError as error:
        raise V7ScoreError(str(error)) from error
    expected = dict(checked_v6)
    expected["schema"] = "causal_dual_memory_cell_d_score_failure_v7"
    expected["v6_runtime_flags_failure"] = _v6_failure_payload()
    if dict(value) != expected:
        raise V7ScoreError("V7 failure canonical predecessor drift")
    return expected


def _validate_preflight_hook(value: Mapping[str, object], identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise V7ScoreError("V7 lifecycle identity type drift")
    return validate_target_free_preflight(value, identity)


def _validate_authorization_hook(
    value: Mapping[str, object], pre_sha: str, preflight: Mapping[str, object], identity: Any,
) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise V7ScoreError("V7 lifecycle identity type drift")
    return validate_root_authorization(value, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity)


def _closure_hook(root: Path) -> Mapping[str, object]:
    return plan.implementation_closure(root).payload()


def _fresh_score_hook(root: Path) -> None:
    plan.assert_fresh_prospective_root(root, plan.SCORE_ROOT_RELATIVE)


def _input_payload_hook(authority: Any, identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity) or not isinstance(authority, v1score.InputAuthority):
        raise V7ScoreError("V7 physical route must return inherited typed input authority")
    return authority.payload(identity=identity)


def _validate_input_hook(value: Mapping[str, object], identity: Any, fixed: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity) or not isinstance(fixed, v1score.FixedEvaluationAuthority):
        raise V7ScoreError("V7 fixed evaluation authority type drift")
    try:
        return v1score.validate_input_authority_payload(value, identity=identity, evaluation_authority=fixed)
    except v1score.ScoreError as error:
        raise V7ScoreError(str(error)) from error


def _fixed_from_preflight(preflight: Mapping[str, object]) -> Any:
    try:
        return v1score._fixed_authority_from_payload(preflight["evaluation_authority"])
    except v1score.ScoreError as error:
        raise V7ScoreError(str(error)) from error


def _continue_after_budget(_budget: int, _gates: Mapping[int, Mapping[str, object]]) -> bool:
    """The V7 matrix is always complete unless execution/integrity fails."""
    return True


V7_LIFECYCLE_HOOKS = v1score.ProfiledLifecycleHooks(
    route="causal_dual_memory_cell_d_score_v7",
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
    """Use the single shared V1 lifecycle with V7 typed hook codecs."""
    return v1score.run_profiled_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=official_preflight_sha256,
        root_authorization_sha256=root_authorization_sha256,
        preflight=preflight, authorization=authorization, hooks=V7_LIFECYCLE_HOOKS,
    )
