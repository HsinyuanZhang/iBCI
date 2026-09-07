"""Guarded H1 all-source recovery-candidate Docker/EvalAI preflight.

The default command is read-only and is safe to run before a submission
decision.  It verifies the immutable recovery-package final audit and payload,
the local image's embedded ``/data/decoder.pkl`` bytes, challenge 2319 / phase
4599 / team 41975, and the current quota.  Docker push and EvalAI registration
are available only behind ``--execute`` and a complete image-id confirmation.

This helper is deliberately separate from ``h1_evalai_submit.py``: that module
continues to describe the released-code baseline.  No source training,
payload export, Docker build, private-label read, or formal query occurs here.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _datetime
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.h1_carrierid_all_source_submission_receipt import (
    EXPECTED_BATCH_SIZE,
    EXPECTED_CHALLENGE_ID,
    EXPECTED_MODEL_FILE_ALIASES,
    EXPECTED_PHASE_ID,
    EXPECTED_PHASE_SLUG,
    EXPECTED_TASK,
    EXPECTED_TEAM_ID,
    METHOD_DESCRIPTION,
    METHOD_NAME,
    SUBMISSION_ATTRIBUTES,
    validate_recovery_package,
)


PREFLIGHT_SCHEMA = "h1_carrierid_all_source_evalai_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_ALL_SOURCE_EVALAI_READ_ONLY_PREFLIGHT"
STATE_SCHEMA = "h1_carrierid_all_source_evalai_push_state_v1"
PUSH_STATE_SCHEMA = STATE_SCHEMA
_SHA_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ValueError(f"{label} must be a 64-hex SHA-256")
    return value.removeprefix("sha256:").lower()


def _env_map(image: Any) -> dict[str, str]:
    values: dict[str, str] = {}
    attrs = getattr(image, "attrs", {}) or {}
    for item in attrs.get("Config", {}).get("Env", ()) or ():
        key, separator, value = str(item).partition("=")
        if separator:
            values[key] = value
    # A few Docker fakes expose ``env`` directly; accepting it keeps this
    # checker straightforward to unit-test without weakening real image checks.
    for key, value in (getattr(image, "env", {}) or {}).items():
        values.setdefault(str(key), str(value))
    return values


def _embedded_decoder_sha(client: Any, image_id: str) -> str:
    """Hash the exact decoder bytes embedded in ``/data/decoder.pkl``."""

    output = client.containers.run(
        image_id,
        command=["sha256sum", "/data/decoder.pkl"],
        entrypoint="",
        remove=True,
        network_disabled=True,
    )
    if isinstance(output, bytes):
        text = output.decode("utf-8", "strict")
    else:
        text = str(output)
    token = text.strip().split()[0] if text.strip() else ""
    return _sha(token, "embedded decoder SHA")


def _runtime_init_smoke(client: Any, image_id: str) -> dict[str, Any]:
    """Import/init the exact all-source decoder in a network-disabled container.

    This does not open evaluation data or labels.  It proves that the dedicated
    entrypoint's Python import closure can load the embedded decoder payload and
    that the model is in frozen eval mode before a future push.
    """

    code = (
        "import decode; assert callable(getattr(decode, 'main', None)); "
        "from falcon_challenge.config import FalconConfig, FalconTask; "
        "from third_party.falcon_challenge.h1_carrierid_all_source_decoder import H1CarrierIdAllSourceDecoder; "
        "d=H1CarrierIdAllSourceDecoder(FalconConfig(task=FalconTask.h1), '/data/decoder.pkl', batch_size=1); "
        "assert getattr(d.clf, 'training', False) is False; "
        "assert len(d.calib_carriers) == 27 and len(d.calib_trial_features) == 27; "
        "assert set(d.calib_carriers) == set(d.calib_trial_features); "
        "print('H1_ALL_SOURCE_RUNTIME_SMOKE_PASS')"
    )
    output = client.containers.run(
        image_id,
        command=["python", "-c", code],
        entrypoint="",
        remove=True,
        network_disabled=True,
    )
    text = output.decode("utf-8", "strict") if isinstance(output, bytes) else str(output)
    if "H1_ALL_SOURCE_RUNTIME_SMOKE_PASS" not in text:
        raise RuntimeError(f"all-source decoder runtime import/init smoke failed: {text!r}")
    return {
        "status": "PASS_H1_ALL_SOURCE_RUNTIME_IMPORT_INIT",
        "decoder_path": "/data/decoder.pkl",
        "network_disabled": True,
        "query_recordings_opened": 0,
        "formal_test_labels_opened": 0,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
    }


def _get(path: str, request_fn: Callable[..., Any] | None = None) -> dict[str, Any]:
    if request_fn is None:
        try:
            from evalai.utils.requests import make_request
        except ImportError as exc:  # pragma: no cover - only in bare environments
            raise RuntimeError("EvalAI client is unavailable; pass mocked phase/challenge/quota data") from exc
        result = make_request(path, "GET")
    else:
        try:
            result = request_fn(path, "GET")
        except TypeError:
            # Keep the offline injection ergonomic for a one-argument lookup
            # function while retaining the real EvalAI ``(path, method)`` shape.
            result = request_fn(path)
    if not isinstance(result, dict):
        raise RuntimeError(f"EvalAI GET did not return an object: {path}")
    return result


def _quota(phase: Mapping[str, Any], submissions: Mapping[str, Any], *, now: _datetime.datetime | None = None) -> dict[str, int]:
    rows = submissions.get("results", ())
    if not isinstance(rows, list):
        raise RuntimeError("EvalAI submission listing has no results list")
    current = now or _datetime.datetime.now(_datetime.timezone.utc)
    parsed: list[_datetime.datetime] = []
    for row in rows:
        if not isinstance(row, Mapping) or "submitted_at" not in row:
            raise RuntimeError("EvalAI submission listing contains malformed timestamp")
        try:
            timestamp = str(row["submitted_at"])
            normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
            parsed.append(_datetime.datetime.fromisoformat(normalized))
        except ValueError as exc:
            raise RuntimeError("EvalAI submission listing contains malformed timestamp") from exc
    result = {
        "today": sum(value.date() == current.date() for value in parsed),
        "month": sum((value.year, value.month) == (current.year, current.month) for value in parsed),
        "total": int(submissions.get("count", len(rows))),
        "active": sum(row.get("status") in {"submitted", "queued", "running"} for row in rows),
        "max_per_day": int(phase["max_submissions_per_day"]),
        "max_per_month": int(phase["max_submissions_per_month"]),
        "max_total": int(phase["max_submissions"]),
        "max_concurrent": int(phase["max_concurrent_submissions_allowed"]),
    }
    if result["today"] >= result["max_per_day"]:
        raise RuntimeError("EvalAI daily submission quota is exhausted")
    if result["month"] >= result["max_per_month"]:
        raise RuntimeError("EvalAI monthly submission quota is exhausted")
    if result["total"] >= result["max_total"]:
        raise RuntimeError("EvalAI total submission quota is exhausted")
    if result["active"] >= result["max_concurrent"]:
        raise RuntimeError("EvalAI concurrent submission quota is exhausted")
    return result


def _phase_identity(phase: Mapping[str, Any]) -> None:
    phase_id = phase.get("id", phase.get("phase_id"))
    challenge_id = phase.get("challenge", phase.get("challenge_id"))
    try:
        phase_matches = int(phase_id) == EXPECTED_PHASE_ID
        challenge_matches = int(challenge_id) == EXPECTED_CHALLENGE_ID
    except (TypeError, ValueError):
        phase_matches = challenge_matches = False
    if not phase_matches or not challenge_matches:
        raise RuntimeError("EvalAI challenge/phase identity changed")
    if phase.get("is_active") is not True or phase.get("is_submission_paused") is not False:
        raise RuntimeError("EvalAI H1 target phase is not accepting submissions")


def _image_size(image: Any) -> int:
    attrs = getattr(image, "attrs", {}) or {}
    raw = attrs.get("Size", attrs.get("VirtualSize", 0))
    try:
        size = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("local Docker image size is malformed") from exc
    if size <= 0:
        raise RuntimeError("local Docker image has no positive size")
    return size


def preflight(
    *,
    recovery_package_audit: str | Path | None = None,
    recovery_audit_path: str | Path | None = None,
    image_tag: str,
    payload_path: str | Path | None = None,
    expected_audit_sha256: str | None = None,
    expected_payload_sha256: str | None = None,
    embedded_decoder_sha256: str | None = None,
    client: Any | None = None,
    phase: Mapping[str, Any] | None = None,
    challenge: Mapping[str, Any] | None = None,
    submissions: Mapping[str, Any] | None = None,
    request_fn: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], Any, Any]:
    """Run a read-only Docker/EvalAI preflight and return report/client/image.

    ``phase``, ``challenge``, ``submissions``, and ``request_fn`` are injectable
    for offline tests.  If omitted, the future operator invocation performs
    read-only EvalAI GET requests; no POST is made in this function.
    """

    audit_value = recovery_package_audit or recovery_audit_path
    if audit_value is None:
        raise ValueError("recovery package final audit path is required")
    recovery = validate_recovery_package(
        audit_value,
        expected_audit_sha256=expected_audit_sha256,
        payload_path=payload_path,
        expected_payload_sha256=expected_payload_sha256,
    )
    if client is None:
        try:
            import docker
        except ImportError as exc:  # pragma: no cover - only in bare environments
            raise RuntimeError("Docker SDK is unavailable; pass a mocked client") from exc
        client = docker.from_env()
    if hasattr(client, "ping") and not client.ping():
        raise RuntimeError("Docker daemon ping failed")
    image = client.images.get(image_tag)
    image_id_raw = getattr(image, "id", None)
    image_id = _sha(image_id_raw, "local Docker image ID")
    env = _env_map(image)
    model_file = env.get("MODEL_FILE")
    if model_file not in EXPECTED_MODEL_FILE_ALIASES:
        raise RuntimeError(f"H1 all-source Docker MODEL_FILE is not a recovery payload: {model_file!r}")
    expected_env = {
        "TASK": EXPECTED_TASK,
        "BATCH_SIZE": str(EXPECTED_BATCH_SIZE),
        "PHASE": "test",
        "MODEL_FILE": model_file,
        "RECOVERY_AUDIT_SHA256": recovery["sha256"],
        "RECOVERY_PAYLOAD_SHA256": recovery["payload"]["sha256"],
        "RECOVERY_STABILITY_SHA256": recovery["stability_audit"]["sha256"],
    }
    for key, expected in expected_env.items():
        if env.get(key) != expected:
            raise RuntimeError(f"H1 all-source Docker runtime environment mismatch at {key}: {env.get(key)!r}")
    embedded_sha = _embedded_decoder_sha(client, getattr(image, "id"))
    if embedded_decoder_sha256 is not None and embedded_sha != _sha(embedded_decoder_sha256, "embedded_decoder_sha256"):
        raise RuntimeError("Docker embedded decoder SHA differs from supplied SHA")
    if embedded_sha != recovery["payload"]["sha256"]:
        raise RuntimeError("Docker /data/decoder.pkl does not match the recovery payload SHA")
    # The dedicated Dockerfile requires this build argument.  An image built
    # with an omitted or literal ``UNBOUND`` value is intentionally not
    # submission-ready: immutable embedded bytes must be carried into the
    # preflight/receipt explicitly.
    if env.get("DECODER_SHA256") != embedded_sha:
        raise RuntimeError("Docker DECODER_SHA256 does not match embedded decoder bytes")
    runtime_smoke = _runtime_init_smoke(client, getattr(image, "id"))

    # EvalAI phase/challenge/quota are read-only; injected mappings avoid any
    # network requirement for the unit-test contract.
    phase_fetched = phase is None
    if phase is None:
        try:
            from evalai.utils.urls import URLS
            phase = _get(URLS.phase_details_using_slug.value.format(EXPECTED_PHASE_SLUG), request_fn)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("EvalAI URLs are unavailable; pass phase/challenge/submissions") from exc
    _phase_identity(phase)
    challenge_fetched = challenge is None
    if challenge is None:
        try:
            from evalai.utils.urls import URLS
            challenge = _get(URLS.challenge_details.value.format(EXPECTED_CHALLENGE_ID), request_fn)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("EvalAI URLs are unavailable; pass challenge mapping") from exc
    if challenge.get("id", challenge.get("challenge_id")) not in (EXPECTED_CHALLENGE_ID, str(EXPECTED_CHALLENGE_ID)):
        raise RuntimeError("EvalAI challenge identity changed")
    image_size = _image_size(image)
    limit = challenge.get("max_docker_image_size")
    if limit is None:
        raise RuntimeError("EvalAI challenge lacks max_docker_image_size")
    if image_size > int(limit):
        raise RuntimeError("H1 all-source image exceeds challenge Docker-size limit")
    submissions_fetched = submissions is None
    if submissions is None:
        try:
            from evalai.utils.urls import URLS
            submissions = _get(URLS.my_submissions.value.format(EXPECTED_CHALLENGE_ID, EXPECTED_PHASE_ID), request_fn)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("EvalAI URLs are unavailable; pass submissions mapping") from exc
    quota = _quota(phase, submissions)
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "mode": "read_only_preflight",
        "method_name": METHOD_NAME,
        "method_description": METHOD_DESCRIPTION,
        "submission_attributes": SUBMISSION_ATTRIBUTES,
        "submission_metadata": SUBMISSION_ATTRIBUTES,
        "recovery_package_final_audit": {"path": recovery["path"], "sha256": recovery["sha256"], "schema": recovery["schema"]},
        "recovery_payload": recovery["payload"],
        "payload_contract": recovery["payload_contract"],
        "stability_audit": recovery["stability_audit"],
        "root_recovery_audit": recovery["root_recovery_audit"],
        "code_provenance": recovery["code_provenance"],
        "recovery_submission": recovery["recovery_submission"],
        "provenance_gap": recovery["provenance_gap"],
        "scope": recovery["scope"],
        "terminal_checkpoint": recovery["terminal_checkpoint"],
        "image_tag": image_tag,
        "image_id": f"sha256:{image_id}",
        "image_size": image_size,
        "embedded_decoder_sha256": embedded_sha,
        "runtime_env": expected_env | {"DECODER_SHA256": env.get("DECODER_SHA256", embedded_sha)},
        "runtime_smoke": runtime_smoke,
        "challenge_id": EXPECTED_CHALLENGE_ID,
        "phase_id": EXPECTED_PHASE_ID,
        "phase_slug": EXPECTED_PHASE_SLUG,
        "team_id": EXPECTED_TEAM_ID,
        "challenge": {"challenge_id": EXPECTED_CHALLENGE_ID, "phase_id": EXPECTED_PHASE_ID, "phase_slug": EXPECTED_PHASE_SLUG, "team_id": EXPECTED_TEAM_ID, "visibility": "private"},
        "quota": quota,
        "private": True,
        "external_action": {
            "evalai_get_performed_by_this_tool": phase_fetched or challenge_fetched or submissions_fetched,
            "evalai_push_performed_by_this_tool": False,
            "docker_build_performed_by_this_tool": False,
            "private_labels_read_by_this_tool": False,
        },
    }
    return report, client, image


# Explicit descriptive alias for callers that never enable push/registration.
read_only_preflight = preflight


def _write_state(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write a restartable mode-0600 push state without replacing an existing file."""

    original = Path(path)
    output = original.resolve()
    if original.is_symlink() or output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite all-source push state {original}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to reuse temporary all-source push state {temporary}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # link() is atomic and fails rather than replacing a state created by
        # another operator between the existence check and publication.
        os.link(temporary, output)
        os.unlink(temporary)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _registry_local_binding(local_image_id: str, registry_digest: str, config_digest: str) -> str:
    local = _sha(local_image_id, "local image ID")
    manifest = _sha(registry_digest, "registry manifest digest")
    config = _sha(config_digest, "registry config digest")
    if local == manifest:
        return "registry_manifest_digest_equals_local_image_id"
    if local == config:
        return "registry_config_digest_equals_local_image_id"
    raise RuntimeError("pushed ECR manifest/config digests do not bind the local all-source image")


def _registry_config_digest(ecr: Any, repository_name: str, tag: str) -> tuple[str, str]:
    description = ecr.describe_images(repositoryName=repository_name, imageIds=[{"imageTag": tag}])["imageDetails"][0]
    registry_digest = str(description["imageDigest"])
    response = ecr.batch_get_image(
        repositoryName=repository_name,
        imageIds=[{"imageTag": tag}],
        acceptedMediaTypes=["application/vnd.docker.distribution.manifest.v2+json", "application/vnd.oci.image.manifest.v1+json"],
    )
    images = response.get("images", ())
    if len(images) != 1:
        raise RuntimeError(f"ECR did not return one pushed H1 all-source manifest: {response.get('failures')}")
    manifest = json.loads(images[0]["imageManifest"])
    config_digest = manifest.get("config", {}).get("digest")
    return registry_digest, _sha(config_digest, "registry config digest")


def push_image(*, report: Mapping[str, Any], client: Any, state_path: str | Path) -> dict[str, Any]:
    """Push one preflighted image and persist content-addressed state.

    This is only called from ``--execute``.  It is intentionally not reachable
    from a default read-only invocation or from any source-training path.
    """

    try:
        import boto3
        from evalai.utils.config import AWS_REGION, ENVIRONMENT
        from evalai.utils.urls import URLS
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("EvalAI/AWS clients are unavailable for execute mode") from exc
    if ENVIRONMENT != "PRODUCTION":
        raise RuntimeError("H1 all-source submission helper permits production EvalAI only")
    credentials_result = _get(URLS.get_aws_credentials.value.format(EXPECTED_PHASE_ID))
    success = credentials_result.get("success", {})
    federated = success.get("federated_user", {})
    credentials = federated.get("Credentials", {})
    repository_uri = success.get("docker_repository_uri")
    account_id = str(federated.get("FederatedUser", {}).get("FederatedUserId", "")).split(":", 1)[0]
    if not repository_uri or not account_id:
        raise RuntimeError("EvalAI did not return an ECR repository identity")
    ecr = boto3.client("ecr", region_name=AWS_REGION, aws_access_key_id=credentials.get("AccessKeyId"), aws_secret_access_key=credentials.get("SecretAccessKey"), aws_session_token=credentials.get("SessionToken"))
    authorization = ecr.get_authorization_token(registryIds=[account_id])["authorizationData"][0]
    username, password = base64.b64decode(authorization["authorizationToken"]).decode("utf-8").split(":", 1)
    client.login(username=username, password=password, registry=authorization["proxyEndpoint"])
    tag = str(uuid.uuid4())
    submitted_uri = f"{repository_uri}:{tag}"
    image = client.images.get(report["image_tag"])
    image.tag(submitted_uri)
    event_tail: list[dict[str, Any]] = []
    for event in client.images.push(repository_uri, tag, stream=True, decode=True):
        if event.get("errorDetail") or event.get("error"):
            raise RuntimeError(f"H1 all-source Docker push failed: {event}")
        event_tail = (event_tail + [event])[-8:]
    registry_digest, config_digest = _registry_config_digest(ecr, repository_uri.split("/", 1)[1], tag)
    binding = _registry_local_binding(str(report["image_id"]), registry_digest, config_digest)
    state = {
        "schema": STATE_SCHEMA,
        "recovery_package_final_audit": report["recovery_package_final_audit"],
        "recovery_payload": report["recovery_payload"],
        "code_provenance": report["code_provenance"],
        "recovery_submission": report["recovery_submission"],
        "provenance_gap": report["provenance_gap"],
        "image_tag": report["image_tag"],
        "local_image_id": report["image_id"],
        "embedded_decoder_sha256": report["embedded_decoder_sha256"],
        "challenge_id": EXPECTED_CHALLENGE_ID,
        "phase_id": EXPECTED_PHASE_ID,
        "team_id": EXPECTED_TEAM_ID,
        "quota": report["quota"],
        "repository_uri": repository_uri,
        "uuid_tag": tag,
        "submitted_image_uri": submitted_uri,
        "registry_manifest_digest": registry_digest,
        "registry_config_digest": config_digest,
        "registry_local_binding": binding,
        "push_event_tail": event_tail,
        "pushed": True,
        "registered": False,
    }
    _write_state(state_path, state)
    return state


def register_submission(state: Mapping[str, Any], state_path: str | Path) -> dict[str, Any]:
    """Register a pushed image, carrying the recovery/provenance metadata."""

    if state.get("schema") != STATE_SCHEMA or state.get("pushed") is not True or state.get("registered") is True:
        raise RuntimeError("H1 all-source push state is absent, already registered, or has the wrong schema")
    try:
        from evalai.utils.requests import make_request
        from evalai.utils.urls import URLS
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("EvalAI client is unavailable for execute mode") from exc
    payload = {"submitted_image_uri": state["submitted_image_uri"]}
    metadata = {
        "is_public": json.dumps(False),
        "method_name": METHOD_NAME,
        "method_description": METHOD_DESCRIPTION,
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
        "recovery_package_final_audit_sha256": state["recovery_package_final_audit"]["sha256"],
        "recovery_payload_sha256": state["recovery_payload"]["sha256"],
        "code_provenance": json.dumps(state.get("code_provenance", {}), sort_keys=True),
        "recovery_submission_policy": json.dumps(state.get("recovery_submission", {}), sort_keys=True),
        "embedded_decoder_sha256": state["embedded_decoder_sha256"],
        "provenance_gap": json.dumps(state["provenance_gap"], sort_keys=True),
    }
    with tempfile.TemporaryDirectory(prefix="spint_h1_all_source_evalai_") as directory:
        submission_file = Path(directory) / "submission.json"
        submission_file.write_text(json.dumps(payload), encoding="utf-8")
        response = make_request(URLS.make_submission.value.format(EXPECTED_CHALLENGE_ID, EXPECTED_PHASE_ID), "POST", files=str(submission_file), data=metadata)
    if int(response.get("challenge_phase", -1)) != EXPECTED_PHASE_ID or int(response.get("participant_team", -1)) != EXPECTED_TEAM_ID:
        raise RuntimeError(f"EvalAI registered all-source submission under unexpected phase/team: {response}")
    if response.get("is_public") is not False or response.get("status") not in {"submitted", "queued"}:
        raise RuntimeError(f"EvalAI returned an invalid all-source registration response: {response}")
    result = dict(state)
    result.update({"registered": True, "submission_id": int(response["id"]), "registration_response": response})
    # State updates after registration are append-only: write a sibling receipt
    # if the original state exists rather than silently replacing it.
    registered_path = Path(state_path).with_name(Path(state_path).name + ".registered.json")
    _write_state(registered_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Guarded H1 all-source recovery-candidate EvalAI submission")
    parser.add_argument("--recovery-package-audit", "--recovery-audit", dest="recovery_package_audit", type=Path, required=True)
    parser.add_argument("--payload-path", type=Path)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256")
    parser.add_argument("--expected-payload-sha256")
    parser.add_argument("--embedded-decoder-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    args = parser.parse_args(argv)
    report, client, _image = preflight(
        recovery_package_audit=args.recovery_package_audit,
        payload_path=args.payload_path,
        expected_audit_sha256=args.expected_audit_sha256,
        expected_payload_sha256=args.expected_payload_sha256,
        embedded_decoder_sha256=args.embedded_decoder_sha256,
        image_tag=args.image_tag,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if args.confirm_image_id != report["image_id"]:
        raise RuntimeError("execute mode requires the complete immutable local image ID")
    if args.state.exists():
        raise RuntimeError("existing all-source push state requires explicit operator review")
    state = push_image(report=report, client=client, state_path=args.state)
    completed = register_submission(state, args.state)
    print(json.dumps({"schema": STATE_SCHEMA, "submission_id": completed["submission_id"], "image_id": completed["local_image_id"]}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
