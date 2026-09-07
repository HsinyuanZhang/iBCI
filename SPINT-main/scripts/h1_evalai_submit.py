"""Guarded H1 Docker push and EvalAI registration.

The legacy EvalAI 1.3.18 ``push`` command is not reliable with Docker 29:
modern docker-py can finish an ECR upload without emitting the old ``aux.Tag``
event that triggers the final EvalAI registration request.  This helper keeps
the same service-side sequence, but makes the two steps explicit and records a
restartable per-variant state file.

Default invocation is read-only.  External mutation requires ``--execute`` and
the complete local Docker image ID printed by the read-only preflight.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import boto3
import docker
from evalai.utils.config import AWS_REGION, ENVIRONMENT
from evalai.utils.requests import make_request
from evalai.utils.urls import URLS


CHALLENGE_ID = 2319
PHASE_ID = 4599
PHASE_SLUG = "few-shot-test-2319"
TEAM_ID = 41975
PACKAGE_SCHEMA = "spint_h1_terminal_package_v1"
PACKAGE_STATUS = "PASS_H1_TERMINAL_PACKAGE_ALLOWLISTED"
PACKAGE_PROTOCOL = "spint_h1_baseline_reproduction_v1"
STATE_SCHEMA = "spint_h1_evalai_push_state_v1"

CANDIDATES = {
    "released_code_lr_5e-5": {
        "role": "implementation_primary",
        "package_basename": "spint_h1_released_code_lr_5e-5.pkl",
        "method_name": "Original SPINT H1 released-code LR 5e-5 epoch49",
        "method_description": (
            "Original SPINT H1 trained for the fixed 50-epoch terminal schedule using the "
            "released-code learning rate 5e-5. The decoder uses two public neural calibration "
            "trials per session, no calibration behavior labels, and no target-session optimizer "
            "or backpropagation."
        ),
    },
    "paper_lr_1e-5": {
        "role": "appendix_paper_reference",
        "package_basename": "spint_h1_paper_lr_1e-5.pkl",
        "method_name": "Original SPINT H1 paper LR 1e-5 epoch49",
        "method_description": (
            "Original SPINT H1 trained for the fixed 50-epoch terminal schedule using the "
            "paper Appendix learning rate 1e-5. The decoder uses two public neural calibration "
            "trials per session, no calibration behavior labels, and no target-session optimizer "
            "or backpropagation."
        ),
    },
}

SUBMISSION_ATTRIBUTES = [
    {
        "name": "IsHeldOutZeroShot",
        "type": "boolean",
        "description": "Uses no calibration data from held-out days?",
        "required": True,
        "value": False,
    },
    {
        "name": "IsTestTimeAdaptive",
        "type": "boolean",
        "description": "Performs test-time parameter recalibration?",
        "required": True,
        "value": False,
    },
    {
        "name": "IsPretrained",
        "type": "boolean",
        "description": "Uses training data outside FALCON?",
        "required": True,
        "value": False,
    },
]

_SHA_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {candidate}")
    return value


def validate_package_receipt(path: str | Path, variant: str) -> dict[str, Any]:
    if variant not in CANDIDATES:
        raise ValueError(f"unknown H1 candidate {variant!r}")
    receipt_path = Path(path).resolve()
    receipt = _load_json(receipt_path)
    candidate = CANDIDATES[variant]
    required = {
        "schema": PACKAGE_SCHEMA,
        "status": PACKAGE_STATUS,
        "protocol": PACKAGE_PROTOCOL,
        "protocol_variant": variant,
        "comparison_role": candidate["role"],
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"terminal package receipt mismatch at {key}: {receipt.get(key)!r}")
    package = Path(str(receipt.get("package_path", ""))).resolve()
    if package.name != candidate["package_basename"] or not package.is_file():
        raise ValueError("terminal package path/basename does not match the selected H1 role")
    package_sha = receipt.get("package_sha256")
    if not isinstance(package_sha, str) or not _SHA_RE.fullmatch(package_sha):
        raise ValueError("terminal package receipt lacks a valid package SHA-256")
    if sha256_file(package) != package_sha.removeprefix("sha256:"):
        raise ValueError("terminal package bytes no longer match their receipt")
    guards = receipt.get("selection_guards")
    if not isinstance(guards, Mapping) or any(value is not False for value in guards.values()):
        raise ValueError("terminal package selection/private-data guards are incomplete or true")
    manifest = receipt.get("input_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("terminal package input manifest is missing")
    if (
        manifest.get("heldin_calibration_recordings") != 13
        or manifest.get("public_heldout_calibration_recordings") != 14
        or manifest.get("minival_included") is not False
        or manifest.get("private_test_opened") is not False
    ):
        raise ValueError("terminal package calibration manifest is not the canonical H1 13+14 set")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "package_path": str(package),
        "package_sha256": package_sha.removeprefix("sha256:"),
        "checkpoint_sha256": str(receipt["checkpoint_sha256"]),
        "variant": variant,
        "role": candidate["role"],
    }


def _env_map(image: Any) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in image.attrs.get("Config", {}).get("Env", ()) or ():
        key, separator, value = str(item).partition("=")
        if separator:
            values[key] = value
    return values


def _embedded_package_sha(client: Any, image_id: str) -> str:
    output = client.containers.run(
        image_id,
        command=["sha256sum", "/data/decoder.pkl"],
        entrypoint="",
        remove=True,
        network_disabled=True,
    )
    token = output.decode("utf-8", "strict").strip().split()[0]
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise RuntimeError(f"cannot parse embedded decoder SHA-256: {output!r}")
    return token


def _get(path: str) -> dict[str, Any]:
    result = make_request(path, "GET")
    if not isinstance(result, dict):
        raise RuntimeError(f"EvalAI GET did not return an object: {path}")
    return result


def _quota(phase: Mapping[str, Any], submissions: Mapping[str, Any]) -> dict[str, int]:
    rows = submissions.get("results", ())
    if not isinstance(rows, list):
        raise RuntimeError("EvalAI submission listing has no results list")
    now = datetime.now(timezone.utc)
    dates = [datetime.fromisoformat(str(row["submitted_at"]).replace("Z", "+00:00")) for row in rows]
    result = {
        "today": sum(value.date() == now.date() for value in dates),
        "month": sum((value.year, value.month) == (now.year, now.month) for value in dates),
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


def preflight(*, package_receipt: str | Path, variant: str, image_tag: str) -> tuple[dict[str, Any], Any, Any]:
    package = validate_package_receipt(package_receipt, variant)
    client = docker.from_env()
    if not client.ping():
        raise RuntimeError("Docker daemon ping failed")
    image = client.images.get(image_tag)
    if not isinstance(image.id, str) or not _SHA_RE.fullmatch(image.id):
        raise RuntimeError("local Docker image has no content-addressed image ID")
    env = _env_map(image)
    expected_env = {"TASK": "h1", "BATCH_SIZE": "8", "PHASE": "test"}
    for key, expected in expected_env.items():
        if env.get(key) != expected:
            raise RuntimeError(f"H1 Docker runtime environment mismatch at {key}: {env.get(key)!r}")
    embedded_sha = _embedded_package_sha(client, image.id)
    if embedded_sha != package["package_sha256"]:
        raise RuntimeError("Docker /data/decoder.pkl does not match the terminal package receipt")

    phase = _get(URLS.phase_details_using_slug.value.format(PHASE_SLUG))
    if phase.get("id") != PHASE_ID or phase.get("challenge") != CHALLENGE_ID:
        raise RuntimeError("EvalAI challenge/phase identity changed")
    if not phase.get("is_active") or phase.get("is_submission_paused"):
        raise RuntimeError("EvalAI H1 target phase is not accepting submissions")
    challenge = _get(URLS.challenge_details.value.format(CHALLENGE_ID))
    image_size = int(image.attrs.get("Size", image.attrs.get("VirtualSize", 0)))
    if image_size <= 0 or image_size > int(challenge["max_docker_image_size"]):
        raise RuntimeError("H1 image size is invalid or exceeds the challenge limit")
    submissions = _get(URLS.my_submissions.value.format(CHALLENGE_ID, PHASE_ID))
    quota = _quota(phase, submissions)
    report = {
        "mode": "read_only_preflight",
        "variant": variant,
        "role": package["role"],
        "package": package,
        "image_tag": image_tag,
        "image_id": image.id,
        "image_size": image_size,
        "embedded_package_sha256": embedded_sha,
        "runtime_env": expected_env,
        "challenge_id": CHALLENGE_ID,
        "phase_id": PHASE_ID,
        "phase_slug": PHASE_SLUG,
        "team_id": TEAM_ID,
        "private": True,
        "quota": quota,
        "submission_attributes": SUBMISSION_ATTRIBUTES,
        "docker_sdk_version": docker.__version__,
    }
    return report, client, image


def _write_state(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(pending, 0o600)
    pending.replace(path)


def _registry_config_digest(ecr: Any, repository_name: str, tag: str) -> tuple[str, str]:
    description = ecr.describe_images(
        repositoryName=repository_name,
        imageIds=[{"imageTag": tag}],
    )["imageDetails"][0]
    registry_digest = str(description["imageDigest"])
    response = ecr.batch_get_image(
        repositoryName=repository_name,
        imageIds=[{"imageTag": tag}],
        acceptedMediaTypes=[
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
        ],
    )
    images = response.get("images", ())
    if len(images) != 1:
        raise RuntimeError(f"ECR did not return one pushed H1 manifest: {response.get('failures')}")
    manifest = json.loads(images[0]["imageManifest"])
    config_digest = manifest.get("config", {}).get("digest")
    if not isinstance(config_digest, str) or not _SHA_RE.fullmatch(config_digest):
        raise RuntimeError("pushed ECR manifest has no image-config digest")
    return registry_digest, config_digest


def _registry_local_binding(local_image_id: str, registry_digest: str, config_digest: str) -> str:
    """Return which content-addressed ECR object binds the local image.

    Docker 29's legacy builder/push path can expose the local image ID as the
    registry *manifest* digest, while BuildKit/schema-2 paths commonly expose
    it as the config digest.  Both objects are SHA-256 content addresses and
    ECR returns both from the exact pushed tag.  Accept either exact binding,
    record which one applied, and reject a tag that matches neither.
    """

    if local_image_id == registry_digest:
        return "registry_manifest_digest_equals_local_image_id"
    if local_image_id == config_digest:
        return "registry_config_digest_equals_local_image_id"
    raise RuntimeError(
        "pushed ECR manifest/config digests do not bind the local image: "
        f"local={local_image_id}, manifest={registry_digest}, config={config_digest}"
    )


def push_image(*, report: Mapping[str, Any], client: Any, image: Any, state_path: Path) -> dict[str, Any]:
    success = _get(URLS.get_aws_credentials.value.format(PHASE_ID))["success"]
    federated = success["federated_user"]
    repository_uri = success["docker_repository_uri"]
    credentials = federated["Credentials"]
    account_id = federated["FederatedUser"]["FederatedUserId"].split(":", 1)[0]
    if ENVIRONMENT != "PRODUCTION":
        raise RuntimeError("H1 submission helper permits production EvalAI only")
    ecr = boto3.client(
        "ecr",
        region_name=AWS_REGION,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )
    authorization = ecr.get_authorization_token(registryIds=[account_id])
    username, password = base64.b64decode(
        authorization["authorizationData"][0]["authorizationToken"]
    ).decode("utf-8").split(":", 1)
    registry = authorization["authorizationData"][0]["proxyEndpoint"]
    client.login(username=username, password=password, registry=registry)

    tag = str(uuid.uuid4())
    submitted_uri = f"{repository_uri}:{tag}"
    image.tag(submitted_uri)
    event_tail: list[dict[str, Any]] = []
    for event in client.images.push(repository_uri, tag, stream=True, decode=True):
        if event.get("errorDetail") or event.get("error"):
            raise RuntimeError(f"H1 Docker push failed: {event}")
        event_tail.append(event)
        event_tail = event_tail[-8:]
    repository_name = repository_uri.split("/", 1)[1]
    registry_digest, config_digest = _registry_config_digest(ecr, repository_name, tag)
    registry_local_binding = _registry_local_binding(
        str(report["image_id"]), registry_digest, config_digest
    )
    state = {
        "schema": STATE_SCHEMA,
        "variant": report["variant"],
        "role": report["role"],
        "package": report["package"],
        "image_tag": report["image_tag"],
        "local_image_id": report["image_id"],
        "repository_uri": repository_uri,
        "uuid_tag": tag,
        "submitted_image_uri": submitted_uri,
        "registry_manifest_digest": registry_digest,
        "registry_config_digest": config_digest,
        "registry_local_binding": registry_local_binding,
        "push_event_tail": event_tail,
        "pushed": True,
        "registered": False,
    }
    _write_state(state_path, state)
    return state


def register_submission(state: dict[str, Any], state_path: Path) -> dict[str, Any]:
    if state.get("registered"):
        raise RuntimeError(f"H1 submission {state.get('submission_id')} is already registered")
    if state.get("schema") != STATE_SCHEMA or not state.get("pushed"):
        raise RuntimeError("H1 push state is absent or has the wrong schema")
    candidate = CANDIDATES[str(state["variant"])]
    payload = {"submitted_image_uri": state["submitted_image_uri"]}
    metadata = {
        "is_public": json.dumps(False),
        "method_name": candidate["method_name"],
        "method_description": candidate["method_description"],
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
    }
    with tempfile.TemporaryDirectory(prefix="spint_h1_evalai_") as directory:
        submission_file = Path(directory) / "submission.json"
        submission_file.write_text(json.dumps(payload), encoding="utf-8")
        response = make_request(
            URLS.make_submission.value.format(CHALLENGE_ID, PHASE_ID),
            "POST",
            files=str(submission_file),
            data=metadata,
        )
    if response.get("challenge_phase") != PHASE_ID or response.get("participant_team") != TEAM_ID:
        raise RuntimeError(f"EvalAI registered H1 submission under unexpected phase/team: {response}")
    if response.get("is_public") is not False or response.get("status") not in {"submitted", "queued"}:
        raise RuntimeError(f"EvalAI returned an invalid H1 registration response: {response}")
    state.update(
        {
            "registered": True,
            "submission_id": int(response["id"]),
            "registration_response": response,
        }
    )
    _write_state(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded original-SPINT H1 EvalAI Docker submission")
    parser.add_argument("--variant", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    args = parser.parse_args()

    report, client, image = preflight(
        package_receipt=args.package_receipt,
        variant=args.variant,
        image_tag=args.image_tag,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.confirm_image_id != report["image_id"]:
        raise RuntimeError("execute mode requires the complete immutable local image ID")
    state_path = args.state.resolve()
    if state_path.exists():
        state = _load_json(state_path)
        if state.get("variant") != args.variant or state.get("local_image_id") != report["image_id"]:
            raise RuntimeError("existing H1 push state belongs to another candidate")
        if state.get("registered"):
            raise RuntimeError(f"H1 submission {state.get('submission_id')} is already registered")
    else:
        state = push_image(report=report, client=client, image=image, state_path=state_path)
    completed = register_submission(state, state_path)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "submission_id": completed["submission_id"],
                "local_image_id": completed["local_image_id"],
                "registry_manifest_digest": completed["registry_manifest_digest"],
                "submitted_image_uri": completed["submitted_image_uri"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
