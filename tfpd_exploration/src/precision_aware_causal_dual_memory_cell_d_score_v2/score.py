"""V2 authority, predecessor, and lifecycle composition.

The scientific score codec and physical evaluator are inherited directly from
the frozen V1 route.  V2 adds only an exact failed-V1 predecessor reader and
the lexical `SUBC_DATA_ROOT`/`SUBM_DATA_ROOT` launch gate.  It deliberately
uses the shared profiled lifecycle rather than copying publication ordering.
"""
from __future__ import annotations

import hashlib
import json
import os
import errno
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.causal_dual_memory_cell_d_score_v1 import plan as baseplan
from src.causal_dual_memory_cell_d_score_v1 import score as sharedscore
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import plan as predecessor_plan
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import score as predecessor_score

from . import plan


AUTHORITY_TOPOLOGY = sharedscore.AUTHORITY_TOPOLOGY
SCORE_TOPOLOGY = sharedscore.SCORE_TOPOLOGY


class PrecisionMatchedScoreV2Error(predecessor_score.PrecisionMatchedScoreError):
    """Fail closed for V1-failure lineage or V2 environment/provenance drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionMatchedScoreV2Error(message)


def _json(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _digest(value: object) -> str:
    return plan.sha256_bytes(_json(value))


def _identity_payload(identity: plan.ScoreIdentity) -> dict[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise PrecisionMatchedScoreV2Error("Precision score V2 requires its exact typed identity")
    return identity.payload()


def _sha(value: object, label: str) -> str:
    try:
        return plan.require_sha(value, label)
    except plan.PrecisionMatchedScoreV2PlanError as error:
        raise PrecisionMatchedScoreV2Error(str(error)) from error


def _directory_identity(path: Path, label: str) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise PrecisionMatchedScoreV2Error(f"V1 {label} directory is inaccessible") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PrecisionMatchedScoreV2Error(f"V1 {label} directory is noncanonical")
    return int(info.st_dev), int(info.st_ino)


def _no_follow_flag() -> int:
    value = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(value, int) or value == 0:
        raise PrecisionMatchedScoreV2Error("V2 immutable receipt reader requires O_NOFOLLOW")
    return value


def _read_held_0444_nlink1_leaf(fd: int, name: str, *, label: str) -> bytes:
    """Read exactly the leaf whose FD was verified, never an lstat snapshot."""
    if not isinstance(name, str) or Path(name).name != name:
        raise PrecisionMatchedScoreV2Error("V1 held leaf name drift")
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _no_follow_flag(),
            dir_fd=fd,
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o444 or int(info.st_nlink) != 1
        ):
            raise PrecisionMatchedScoreV2Error(f"V1 {label} leaf mode/link drift: {name}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    except PrecisionMatchedScoreV2Error:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise PrecisionMatchedScoreV2Error(f"V1 {label} leaf mode/link drift: {name}") from error
        raise PrecisionMatchedScoreV2Error(f"V1 {label} held leaf is inaccessible: {name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_0444_nlink1_pair(fd: int, name: str, expected_sha256: str, *, label: str) -> dict[str, object]:
    """Read one immutable basename pair through the same held directory FD."""
    _sha(expected_sha256, f"V1 {label} {name}")
    body = _read_held_0444_nlink1_leaf(fd, name, label=label)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = _read_held_0444_nlink1_leaf(fd, f"{name}.sha256", label=label)
    if sidecar != f"{digest}  {name}\n".encode("ascii"):
        raise PrecisionMatchedScoreV2Error(f"V1 {label} canonical sidecar drift: {name}")
    if digest != expected_sha256:
        raise PrecisionMatchedScoreV2Error(f"V1 {label} body SHA drift: {name}")
    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise PrecisionMatchedScoreV2Error(f"V1 {label} JSON body is malformed: {name}") from error
    if not isinstance(payload, Mapping):
        raise PrecisionMatchedScoreV2Error(f"V1 {label} JSON root drift: {name}")
    return dict(payload)


def _read_exact_directory(
    directory: Path, *, label: str, expected_sha256s: Mapping[str, str],
) -> tuple[tuple[int, int], dict[str, dict[str, object]]]:
    named = _directory_identity(directory, label)
    descriptor = -1
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | _no_follow_flag())
        held = os.fstat(descriptor)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != named:
            raise PrecisionMatchedScoreV2Error(f"V1 {label} directory identity drift before read")
        expected_names = set(expected_sha256s) | {f"{name}.sha256" for name in expected_sha256s}
        if set(os.listdir(descriptor)) != expected_names:
            raise PrecisionMatchedScoreV2Error(f"V1 {label} exact pair topology drift")
        bodies = {
            name: _read_0444_nlink1_pair(descriptor, name, expected_sha256s[name], label=label)
            for name in sorted(expected_sha256s)
        }
        if _directory_identity(directory, label) != named:
            raise PrecisionMatchedScoreV2Error(f"V1 {label} directory identity drift during read")
        return named, bodies
    except PrecisionMatchedScoreV2Error:
        raise
    except OSError as error:
        raise PrecisionMatchedScoreV2Error(f"V1 {label} descriptor read failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _historical_v1_identity(attempt: Mapping[str, object]) -> predecessor_plan.ScoreIdentity:
    raw = attempt.get("identity")
    if not isinstance(raw, Mapping):
        raise PrecisionMatchedScoreV2Error("V1 failed attempt identity is absent")
    try:
        identity = predecessor_plan.ScoreIdentity(
            closure=raw["closure"], v8_binding=raw["v8_predecessor_binding"],
            selected_device_profile=raw["selected_device_profile"],
        )
    except (KeyError, TypeError, predecessor_plan.PrecisionMatchedScorePlanError) as error:
        raise PrecisionMatchedScoreV2Error("V1 failed attempt identity reconstruction failed") from error
    if identity.payload() != dict(raw):
        raise PrecisionMatchedScoreV2Error("V1 failed attempt historical identity canonical drift")
    return identity


@dataclass(frozen=True)
class V1FailedPredecessorBinding:
    """Held exact V1 authority plus failure graph and its reconstructed identity."""

    authority_directory_identity: tuple[int, int]
    score_directory_identity: tuple[int, int]
    historical_identity: predecessor_plan.ScoreIdentity

    def payload(self) -> dict[str, object]:
        historical = self.historical_identity.payload()
        body = {
            "schema": "precision_aware_cdmd_matched_score_v1_failed_binding_v1",
            "contract": plan.V1_FAILED_PREDECESSOR.payload(),
            "authority_directory_identity": [int(self.authority_directory_identity[0]), int(self.authority_directory_identity[1])],
            "score_directory_identity": [int(self.score_directory_identity[0]), int(self.score_directory_identity[1])],
            "historical_v1_identity_sha256": _digest(historical),
        }
        result = {**body, "binding_sha256": _digest(body)}
        try:
            return plan.validate_v1_failed_predecessor_binding(result)
        except plan.PrecisionMatchedScoreV2PlanError as error:
            raise PrecisionMatchedScoreV2Error(str(error)) from error


def _validate_v1_failed_semantics(
    authority: Mapping[str, Mapping[str, object]], failed: Mapping[str, Mapping[str, object]],
) -> predecessor_plan.ScoreIdentity:
    attempt = failed["attempt.json"]
    failure = failed["failure.json"]
    identity = _historical_v1_identity(attempt)
    preflight = authority["official_preflight.json"]
    authorization = authority["root_authorization.json"]
    try:
        checked_preflight = predecessor_score.validate_target_free_preflight(preflight, identity)
        predecessor_score.validate_root_authorization(
            authorization,
            official_preflight_sha256=plan.V1_AUTHORITY_SHAS["official_preflight.json"],
            preflight=checked_preflight,
            identity=identity,
        )
        expected_attempt = predecessor_score._attempt_payload(
            identity,
            plan.V1_AUTHORITY_SHAS["official_preflight.json"],
            plan.V1_AUTHORITY_SHAS["root_authorization.json"],
        )
        predecessor_score.validate_failure_payload(
            failure, identity, plan.V1_FAILED_SCORE_SHAS["attempt.json"], None,
        )
    except predecessor_score.PrecisionMatchedScoreError as error:
        raise PrecisionMatchedScoreV2Error("V1 failed predecessor semantic graph drift") from error
    if dict(attempt) != expected_attempt:
        raise PrecisionMatchedScoreV2Error("V1 failed predecessor attempt canonical drift")
    expected_flags = {
        "stage": "materialize_inputs",
        "error_class": "PhysicalCDMDScoreError",
        "error_sha256": plan.V1_FAILURE_ERROR_SHA256,
        "within_assets_opened": False,
        "external_assets_opened": False,
        "target_paths_resolved_or_opened": False,
        "checkpoint_opened": True,
        "cuda_initialized": True,
        "full_system_forward_count": 0,
        "group_forward_count": 0,
        "terminal_published": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "input_authority_sha256": None,
    }
    if any(failure.get(key) != expected for key, expected in expected_flags.items()):
        raise PrecisionMatchedScoreV2Error("V1 failed predecessor failure-stage/progress drift")
    return identity


def validate_v1_failed_predecessor(root: Path) -> V1FailedPredecessorBinding:
    """Descriptor-bind V1 authority and failed score before any V2 action."""
    base = Path(root).absolute()
    authority_identity, authority = _read_exact_directory(
        base / plan.V1_AUTHORITY_ROOT_RELATIVE,
        label="authority", expected_sha256s=plan.V1_AUTHORITY_SHAS,
    )
    score_identity, failed = _read_exact_directory(
        base / plan.V1_SCORE_ROOT_RELATIVE,
        label="failed score", expected_sha256s=plan.V1_FAILED_SCORE_SHAS,
    )
    identity = _validate_v1_failed_semantics(authority, failed)
    return V1FailedPredecessorBinding(
        authority_directory_identity=authority_identity,
        score_directory_identity=score_identity,
        historical_identity=identity,
    )


def _require_live_v1_predecessor(root: Path, identity: plan.ScoreIdentity) -> V1FailedPredecessorBinding:
    observed = validate_v1_failed_predecessor(Path(root))
    if observed.payload() != _identity_payload(identity)["v1_failed_predecessor_binding"]:
        raise PrecisionMatchedScoreV2Error("V2 held V1 failed predecessor/identity binding drift")
    return observed


def build_reviewed_identity(
    root: Path, *, selected_device_profile: Mapping[str, object] | None = None,
) -> plan.ScoreIdentity:
    """Build V2 only from held historical V1 evidence and current V2 closure."""
    previous = validate_v1_failed_predecessor(Path(root))
    historical = previous.historical_identity
    return plan.ScoreIdentity(
        closure=historical.closure,
        v8_binding=historical.v8_binding,
        selected_device_profile=(historical.selected_device_profile if selected_device_profile is None else selected_device_profile),
        v8_predecessor=historical.v8_predecessor,
        successor_closure=plan.implementation_closure(Path(root)).payload(),
        v1_failed_predecessor_binding=previous.payload(),
    )


def validate_selected_launch_environment(
    identity: plan.ScoreIdentity, environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Validate device variables and lexical canonical evaluation roots only.

    No `Path.resolve`, `lstat`, open, hash, or target/source access occurs
    here.  Exact literal equality means an alias or symlink spelling fails
    before any authority/result write.
    """
    values = os.environ if environ is None else environ
    try:
        selected = predecessor_score.validate_selected_launch_environment(identity, values)
    except predecessor_score.PrecisionMatchedScoreError as error:
        raise PrecisionMatchedScoreV2Error(str(error)) from error
    roots = plan.CANONICAL_SOURCE_ROOTS
    if any(values.get(name) != expected for name, expected in roots.items()):
        raise PrecisionMatchedScoreV2Error("V2 canonical SUBC/SUBM launch environment drift")
    return {
        "selected_device_profile": dict(selected),
        "canonical_source_environment": plan.canonical_source_environment_payload(),
    }


def _validate_mutating_launch_environment(
    identity: plan.ScoreIdentity, environ: Mapping[str, str] | None,
) -> dict[str, object]:
    """Require the actual process, not merely an issuer-supplied mapping.

    The optional mapping is retained for deterministic no-data validation, but
    every API that can reserve, publish, mint, or enter an attempt proves the
    four process variables match it first.  This prevents an authority issuer
    from validating a caller's claimed roots while the inherited physical
    route would later read a different `os.environ` value.
    """
    checked = validate_selected_launch_environment(identity, environ)
    if environ is not None:
        keys = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", *plan.CANONICAL_SOURCE_ROOTS)
        if any(os.environ.get(key) != environ.get(key) for key in keys):
            raise PrecisionMatchedScoreV2Error("V2 supplied/actual launch environment drift")
    validate_selected_launch_environment(identity, os.environ)
    return checked


def build_target_free_preflight(
    *, root: Path, identity: plan.ScoreIdentity, predecessor: V1FailedPredecessorBinding,
    fixed_authority: sharedscore.FixedEvaluationAuthority | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    environment = validate_selected_launch_environment(identity, environ)
    if predecessor.payload() != _identity_payload(identity)["v1_failed_predecessor_binding"]:
        raise PrecisionMatchedScoreV2Error("V2 preflight V1 failed predecessor/identity drift")
    observed = _require_live_v1_predecessor(Path(root), identity)
    if observed.payload() != predecessor.payload():
        raise PrecisionMatchedScoreV2Error("V2 preflight held V1 failed predecessor drift")
    fixed = sharedscore.derive_fixed_evaluation_authority(Path(root)) if fixed_authority is None else fixed_authority
    return {
        "schema": "precision_aware_cdmd_matched_score_target_free_preflight_v2",
        "status": "PREFLIGHT_ACCEPTED",
        "identity": _identity_payload(identity),
        "source_gate": predecessor.payload(),
        "v8_predecessor": _identity_payload(identity)["v8_predecessor_binding"],
        "v1_failed_predecessor": predecessor.payload(),
        "evaluation_authority": fixed.payload(),
        "metric": dict(baseplan.METRIC_CONTRACT),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "canonical_launch_environment": environment,
        "target_free": True,
        "target_paths_resolved": False,
        "model_or_checkpoint_opened": False,
        "cuda_initialized": False,
        "boundaries": {**baseplan.EXECUTION_BOUNDARIES, "precision_v2_target_state": False},
        "receipt_codec": "precision_v1_compatible_science_codec__v2_identity_and_environment_bound",
    }


def validate_target_free_preflight(value: Mapping[str, object], identity: plan.ScoreIdentity) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "source_gate", "v8_predecessor", "v1_failed_predecessor",
        "evaluation_authority", "metric", "authority_root_relative", "score_root_relative",
        "canonical_launch_environment", "target_free", "target_paths_resolved", "model_or_checkpoint_opened",
        "cuda_initialized", "boundaries", "receipt_codec",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "precision_aware_cdmd_matched_score_target_free_preflight_v2"
        or value.get("status") != "PREFLIGHT_ACCEPTED" or value.get("identity") != _identity_payload(identity)
        or value.get("metric") != baseplan.METRIC_CONTRACT
        or value.get("authority_root_relative") != plan.AUTHORITY_ROOT_RELATIVE
        or value.get("score_root_relative") != plan.SCORE_ROOT_RELATIVE
        or value.get("canonical_launch_environment") != {
            "selected_device_profile": _identity_payload(identity)["selected_device_profile"],
            "canonical_source_environment": plan.canonical_source_environment_payload(),
        }
        or value.get("target_free") is not True or value.get("target_paths_resolved") is not False
        or value.get("model_or_checkpoint_opened") is not False or value.get("cuda_initialized") is not False
        or value.get("receipt_codec") != "precision_v1_compatible_science_codec__v2_identity_and_environment_bound"
    ):
        raise PrecisionMatchedScoreV2Error("V2 preflight schema/environment/boundary drift")
    binding = plan.validate_v1_failed_predecessor_binding(value.get("v1_failed_predecessor"))
    if binding != _identity_payload(identity)["v1_failed_predecessor_binding"] or value.get("source_gate") != binding:
        raise PrecisionMatchedScoreV2Error("V2 preflight V1 failed predecessor binding drift")
    if value.get("v8_predecessor") != _identity_payload(identity)["v8_predecessor_binding"]:
        raise PrecisionMatchedScoreV2Error("V2 preflight V8 binding drift")
    try:
        fixed = sharedscore._fixed_authority_from_payload(value.get("evaluation_authority"))
    except sharedscore.ScoreError as error:
        raise PrecisionMatchedScoreV2Error(str(error)) from error
    if not fixed.payload():
        raise PrecisionMatchedScoreV2Error("V2 preflight fixed authority drift")
    expected_boundaries = {**baseplan.EXECUTION_BOUNDARIES, "precision_v2_target_state": False}
    if value.get("boundaries") != expected_boundaries:
        raise PrecisionMatchedScoreV2Error("V2 preflight boundaries drift")
    return dict(value)


def validate_preflight_against_fixed_authorities(
    root: Path, value: Mapping[str, object], *, identity: plan.ScoreIdentity,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    validate_selected_launch_environment(identity, environ)
    checked = validate_target_free_preflight(value, identity)
    _require_live_v1_predecessor(Path(root), identity)
    try:
        fixed = sharedscore.derive_fixed_evaluation_authority(Path(root)).payload()
    except sharedscore.ScoreError as error:
        raise PrecisionMatchedScoreV2Error(str(error)) from error
    if checked["evaluation_authority"] != fixed:
        raise PrecisionMatchedScoreV2Error("V2 durable preflight fixed authority drift")
    return checked


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    checked = preflight
    binding = checked.get("v1_failed_predecessor") if isinstance(checked, Mapping) else None
    if not isinstance(binding, Mapping):
        raise PrecisionMatchedScoreV2Error("V2 authorization V1 failed predecessor absent")
    return {
        "schema": "precision_aware_cdmd_matched_score_root_authorization_v2",
        "status": "ROOT_AUTHORIZED",
        "official_preflight_sha256": _sha(official_preflight_sha256, "V2 preflight SHA"),
        "identity_sha256": _digest(checked["identity"]),
        "v1_failed_predecessor_binding_sha256": binding.get("binding_sha256"),
        "canonical_source_environment": plan.canonical_source_environment_payload(),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "target_free_preflight_required": True,
        "explicit_execution_capability_required": True,
    }


def validate_root_authorization(
    value: Mapping[str, object], *, official_preflight_sha256: str,
    preflight: Mapping[str, object], identity: plan.ScoreIdentity,
) -> dict[str, object]:
    checked = validate_target_free_preflight(preflight, identity)
    expected = build_root_authorization(official_preflight_sha256=official_preflight_sha256, preflight=checked)
    if dict(value) != expected or expected["identity_sha256"] != identity.sha256:
        raise PrecisionMatchedScoreV2Error("V2 root authorization canonical identity/environment drift")
    return expected


def reserve_authority_artifact(
    root: Path, capability: object, *, identity: plan.ScoreIdentity, environ: Mapping[str, str] | None = None,
) -> sharedscore.ArtifactRoot:
    _validate_mutating_launch_environment(identity, environ)
    sharedscore._require_root_publication_capability(capability)
    base = Path(root)
    _require_live_v1_predecessor(base, identity)
    if plan.implementation_closure(base).payload() != _identity_payload(identity)["closure"]:
        raise PrecisionMatchedScoreV2Error("V2 authority reservation closure drift")
    plan.assert_fresh_prospective_root(base, plan.AUTHORITY_ROOT_RELATIVE)
    parent = base.absolute() / Path(plan.AUTHORITY_ROOT_RELATIVE).parent
    return sharedscore._equal_session_module(base).reserve_artifact_root(
        parent, Path(plan.AUTHORITY_ROOT_RELATIVE).name, topology=AUTHORITY_TOPOLOGY,
    )


def publish_target_free_preflight(
    root: Path, artifact: sharedscore.ArtifactRoot, capability: object, payload: Mapping[str, object], *,
    identity: plan.ScoreIdentity, environ: Mapping[str, str] | None = None,
) -> str:
    _validate_mutating_launch_environment(identity, environ)
    sharedscore._require_root_publication_capability(capability)
    _require_live_v1_predecessor(Path(root), identity)
    checked = validate_preflight_against_fixed_authorities(Path(root), payload, identity=identity, environ=environ)
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization(
    root: Path, artifact: sharedscore.ArtifactRoot, capability: object, payload: Mapping[str, object], *,
    identity: plan.ScoreIdentity, environ: Mapping[str, str] | None = None,
) -> str:
    _validate_mutating_launch_environment(identity, environ)
    sharedscore._require_root_publication_capability(capability)
    _require_live_v1_predecessor(Path(root), identity)
    body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise PrecisionMatchedScoreV2Error("V2 durable preflight malformed") from error
    if not isinstance(preflight, Mapping):
        raise PrecisionMatchedScoreV2Error("V2 durable preflight root drift")
    checked = validate_preflight_against_fixed_authorities(Path(root), preflight, identity=identity, environ=environ)
    authorization = validate_root_authorization(
        payload, official_preflight_sha256=hashlib.sha256(body).hexdigest(), preflight=checked, identity=identity,
    )
    return artifact.publish_json("root_authorization.json", authorization)


def _read_durable_authority_pair(
    root: Path, *, identity: plan.ScoreIdentity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    directory = Path(root).absolute() / plan.AUTHORITY_ROOT_RELATIVE
    named = _directory_identity(directory, "V2 durable authority")
    descriptor = -1
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | _no_follow_flag())
        held = os.fstat(descriptor)
        if not stat.S_ISDIR(held.st_mode) or (int(held.st_dev), int(held.st_ino)) != named:
            raise PrecisionMatchedScoreV2Error("V2 durable authority identity drift before read")
        expected = set(AUTHORITY_TOPOLOGY) | {f"{name}.sha256" for name in AUTHORITY_TOPOLOGY}
        if set(os.listdir(descriptor)) != expected:
            raise PrecisionMatchedScoreV2Error("V2 durable authority exact topology drift")
        _pre_body, preflight, pre_sha = sharedscore._read_unbound_0444_pair(descriptor, "official_preflight.json")
        _auth_body, authorization, auth_sha = sharedscore._read_unbound_0444_pair(descriptor, "root_authorization.json")
        if _directory_identity(directory, "V2 durable authority") != named:
            raise PrecisionMatchedScoreV2Error("V2 durable authority identity drift during read")
        checked = validate_target_free_preflight(preflight, identity)
        validate_root_authorization(
            authorization, official_preflight_sha256=pre_sha, preflight=checked, identity=identity,
        )
        return checked, dict(authorization), pre_sha, auth_sha
    except (OSError, sharedscore.ScoreError) as error:
        raise PrecisionMatchedScoreV2Error("V2 durable authority descriptor read failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_durable_authority(
    root: Path, *, identity: plan.ScoreIdentity,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    return _read_durable_authority_pair(Path(root), identity=identity)


def issue_durable_execution_capability(
    root: Path, *, identity: plan.ScoreIdentity, root_capability: object,
    environ: Mapping[str, str] | None = None,
) -> sharedscore.ExecutionCapability:
    _validate_mutating_launch_environment(identity, environ)
    sharedscore._require_root_publication_capability(root_capability)
    base = Path(root)
    _require_live_v1_predecessor(base, identity)
    if plan.implementation_closure(base).payload() != _identity_payload(identity)["closure"]:
        raise PrecisionMatchedScoreV2Error("V2 execution issuer closure drift")
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(base, identity=identity)
    validate_preflight_against_fixed_authorities(base, preflight, identity=identity, environ=environ)
    plan.assert_fresh_prospective_root(base, plan.SCORE_ROOT_RELATIVE)
    return sharedscore.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=root_capability,
    )


def reserve_score_artifact(
    root: Path, *, identity: plan.ScoreIdentity, capability: object,
    environ: Mapping[str, str] | None = None,
) -> sharedscore.ArtifactRoot:
    sharedscore.require_execution_capability(capability, identity)
    _validate_mutating_launch_environment(identity, environ)
    base = Path(root)
    _require_live_v1_predecessor(base, identity)
    if plan.implementation_closure(base).payload() != _identity_payload(identity)["closure"]:
        raise PrecisionMatchedScoreV2Error("V2 score reservation closure drift")
    preflight, _authorization, pre_sha, auth_sha = load_durable_authority(base, identity=identity)
    approved = sharedscore.require_execution_capability(capability, identity)
    if approved.official_preflight_sha256 != pre_sha or approved.root_authorization_sha256 != auth_sha:
        raise PrecisionMatchedScoreV2Error("V2 score capability/durable authority drift")
    validate_preflight_against_fixed_authorities(base, preflight, identity=identity, environ=environ)
    plan.assert_fresh_prospective_root(base, plan.SCORE_ROOT_RELATIVE)
    parent = base.absolute() / Path(plan.SCORE_ROOT_RELATIVE).parent
    return sharedscore._equal_session_module(base).reserve_artifact_root(
        parent, Path(plan.SCORE_ROOT_RELATIVE).name, topology=SCORE_TOPOLOGY,
    )


def _validate_reserved_score_artifact(root: Path, artifact: sharedscore.ArtifactRoot, identity: Any) -> None:
    if not isinstance(identity, plan.ScoreIdentity):
        raise PrecisionMatchedScoreV2Error("V2 reserved artifact identity type drift")
    try:
        sharedscore.validate_reserved_artifact_root(
            Path(root), artifact, root_relative=plan.SCORE_ROOT_RELATIVE, topology=SCORE_TOPOLOGY,
        )
    except sharedscore.ScoreError as error:
        raise PrecisionMatchedScoreV2Error(str(error)) from error


def _attempt_payload(identity: plan.ScoreIdentity, pre_sha: str, auth_sha: str) -> dict[str, object]:
    base = predecessor_score._attempt_payload(identity, pre_sha, auth_sha)
    return {
        **base,
        "schema": "precision_aware_cdmd_matched_score_attempt_v2",
        "v1_failed_predecessor_binding": _identity_payload(identity)["v1_failed_predecessor_binding"],
        "canonical_source_environment": plan.canonical_source_environment_payload(),
    }


def _validate_attempt_payload(value: Mapping[str, object], identity: plan.ScoreIdentity, pre_sha: str, auth_sha: str) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "preflight_sha256", "authorization_sha256",
        "target_paths_resolved_or_opened", "checkpoint_opened", "cuda_initialized",
        "target_optimizer_backward_update", "v1_failed_predecessor_binding", "canonical_source_environment",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "precision_aware_cdmd_matched_score_attempt_v2":
        raise PrecisionMatchedScoreV2Error("V2 attempt schema drift")
    body = dict(value)
    predecessor = body.pop("v1_failed_predecessor_binding", None)
    environment = body.pop("canonical_source_environment", None)
    body["schema"] = "precision_aware_cdmd_matched_score_attempt_v1"
    expected_base = predecessor_score._attempt_payload(identity, pre_sha, auth_sha)
    if body != expected_base or predecessor != _identity_payload(identity)["v1_failed_predecessor_binding"] \
            or environment != plan.canonical_source_environment_payload():
        raise PrecisionMatchedScoreV2Error("V2 attempt canonical predecessor/environment drift")
    return dict(value)


def _failure_payload(
    identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str | None, stage: str,
    error: BaseException, runtime_progress: object | None,
) -> dict[str, object]:
    base = predecessor_score._failure_payload(identity, attempt_sha, input_sha, stage, error, runtime_progress)
    return {
        **base,
        "schema": "precision_aware_cdmd_matched_score_failure_v2",
        "v1_failed_predecessor_binding": _identity_payload(identity)["v1_failed_predecessor_binding"],
        "canonical_source_environment": plan.canonical_source_environment_payload(),
    }


def _validate_failure_payload(
    value: Mapping[str, object], identity: plan.ScoreIdentity, attempt_sha: str, input_sha: str | None,
) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "attempt_sha256", "input_authority_sha256", "stage", "error_class",
        "error_sha256", "target_paths_resolved_or_opened", "within_assets_opened", "external_assets_opened",
        "checkpoint_opened", "cuda_initialized", "full_system_forward_count", "group_forward_count",
        "terminal_published", "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        "v8_predecessor_binding", "v1_failed_predecessor_binding", "canonical_source_environment",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "precision_aware_cdmd_matched_score_failure_v2":
        raise PrecisionMatchedScoreV2Error("V2 failure schema drift")
    body = dict(value)
    predecessor = body.pop("v1_failed_predecessor_binding", None)
    environment = body.pop("canonical_source_environment", None)
    body["schema"] = "precision_aware_cdmd_matched_score_failure_v1"
    try:
        predecessor_score.validate_failure_payload(body, identity, attempt_sha, input_sha)
    except predecessor_score.PrecisionMatchedScoreError as error:
        raise PrecisionMatchedScoreV2Error(str(error)) from error
    if predecessor != _identity_payload(identity)["v1_failed_predecessor_binding"] \
            or environment != plan.canonical_source_environment_payload():
        raise PrecisionMatchedScoreV2Error("V2 failure predecessor/environment binding drift")
    return dict(value)


def _validate_preflight_hook(value: Mapping[str, object], identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise PrecisionMatchedScoreV2Error("V2 lifecycle identity type drift")
    return validate_target_free_preflight(value, identity)


def _validate_authorization_hook(value: Mapping[str, object], pre_sha: str, preflight: Mapping[str, object], identity: Any) -> Mapping[str, object]:
    if not isinstance(identity, plan.ScoreIdentity):
        raise PrecisionMatchedScoreV2Error("V2 lifecycle identity type drift")
    return validate_root_authorization(value, official_preflight_sha256=pre_sha, preflight=preflight, identity=identity)


def _closure_hook(root: Path) -> Mapping[str, object]:
    return plan.implementation_closure(root).payload()


def _source_gate_hook(root: Path) -> V1FailedPredecessorBinding:
    return validate_v1_failed_predecessor(Path(root))


def _fresh_score_hook(root: Path) -> None:
    plan.assert_fresh_prospective_root(root, plan.SCORE_ROOT_RELATIVE)


def _fixed_from_preflight(preflight: Mapping[str, object]) -> sharedscore.FixedEvaluationAuthority:
    try:
        return sharedscore._fixed_authority_from_payload(preflight["evaluation_authority"])
    except (KeyError, sharedscore.ScoreError) as error:
        raise PrecisionMatchedScoreV2Error("V2 fixed authority reconstruction drift") from error


V2_LIFECYCLE_HOOKS = sharedscore.ProfiledLifecycleHooks(
    route="precision_aware_causal_dual_memory_cell_d_matched_score_v2",
    score_root_relative=plan.SCORE_ROOT_RELATIVE,
    budgets=predecessor_plan.BUDGETS,
    require_capability=sharedscore.require_execution_capability,
    validate_preflight=_validate_preflight_hook,
    validate_authorization=_validate_authorization_hook,
    implementation_closure=_closure_hook,
    validate_source_gate=_source_gate_hook,
    assert_fresh_score_root=_fresh_score_hook,
    make_attempt=_attempt_payload,
    fixed_authority_from_preflight=_fixed_from_preflight,
    input_payload=predecessor_score._input_payload_hook,
    validate_input_payload=predecessor_score._validate_input_hook,
    summarize_budget=predecessor_score.summarize_budget,
    budget_gate=predecessor_score.budget_gate,
    continue_after_budget=lambda _budget, _gates: True,
    build_score=predecessor_score._score_payload,
    validate_score=predecessor_score.validate_score_payload,
    terminal_verdict=predecessor_score.terminal_verdict,
    make_terminal=predecessor_score._terminal_payload,
    validate_terminal=predecessor_score.validate_terminal_payload,
    make_failure=_failure_payload,
    validate_failure=_validate_failure_payload,
    validate_reserved_score_artifact=_validate_reserved_score_artifact,
)


def run_authorized_score_lifecycle(
    root: Path, *, identity: plan.ScoreIdentity, capability: object, backend: sharedscore.ScoreBackend,
    artifact: sharedscore.ArtifactRoot, official_preflight_sha256: str, root_authorization_sha256: str,
    preflight: Mapping[str, object], authorization: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Execute only after the environment and failed-V1 graph are revalidated."""
    _validate_mutating_launch_environment(identity, environ)
    _require_live_v1_predecessor(Path(root), identity)
    return sharedscore.run_profiled_score_lifecycle(
        Path(root), identity=identity, capability=capability, backend=backend, artifact=artifact,
        official_preflight_sha256=official_preflight_sha256, root_authorization_sha256=root_authorization_sha256,
        preflight=preflight, authorization=authorization, hooks=V2_LIFECYCLE_HOOKS,
    )
