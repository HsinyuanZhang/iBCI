"""Guarded EvalAI push/register for an H1 temporal Transformer snapshot.

Uses registrar.py for quota. POST only when CONTAINER_PASS and active < 3
and today < 6. /usr/bin/python3 + user EvalAI site-packages.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/xinyuan/.local/lib/python3.10/site-packages")
for _k in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_k, "")
    if "eval.ai" not in _val:
        os.environ[_k] = f"{_val},eval.ai".strip(",")

from tfpd_exploration.src.two_mainlines_long_v1.registrar import (  # noqa: E402
    CHALLENGE_ID,
    PHASE_ID,
    list_submissions,
    quota,
)

TEAM_ID = 41975
C2_CALIBRATION_SHA256 = "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215"

SUBMISSION_ATTRIBUTES = [
    {
        "name": "IsHeldOutZeroShot",
        "type": "boolean",
        "description": "Is held out prediction Zero Shot?",
        "required": True,
        "value": False,
    },
    {
        "name": "IsTestTimeAdaptive",
        "type": "boolean",
        "description": "Test Time Adaptive?",
        "required": True,
        "value": False,
    },
    {
        "name": "IsPretrained",
        "type": "boolean",
        "description": "Pretrained outside FALCON?",
        "required": True,
        "value": False,
    },
]


def _runtime() -> tuple[Any, Any, Any, str, str]:
    import boto3
    import docker
    from evalai.utils.config import AWS_REGION, ENVIRONMENT
    from evalai.utils.requests import make_request
    from evalai.utils.urls import URLS

    return docker, boto3, (URLS, make_request), AWS_REGION, ENVIRONMENT


def load_candidate(manifest: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    required = (
        "arm",
        "image_tag",
        "image_id",
        "payload_sha256",
        "method_label",
        "method_name",
        "method_description",
        "budget_disclosure",
        "state_path",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"candidate manifest missing {missing}")
    return payload


def preflight(candidate: dict[str, Any]) -> tuple[Any, Any, dict[str, Any], tuple[Any, Any, Any, str, str]]:
    runtime = _runtime()
    docker_module, _, evalai_runtime, _, _ = runtime
    urls, make_request = evalai_runtime
    client = docker_module.from_env()
    image = client.images.get(candidate["image_tag"])
    if image.id != candidate["image_id"]:
        raise RuntimeError(f"image ID guard failed: {image.id} != {candidate['image_id']}")
    labels = image.attrs.get("Config", {}).get("Labels", {}) or {}
    expected = {
        "ai.eval.payload.sha256": candidate["payload_sha256"],
        "ai.eval.calibration.sha256": C2_CALIBRATION_SHA256,
        "ai.eval.old_c2_decoder": "false",
        "ai.eval.method": candidate["method_label"],
        "ai.eval.epoch": "5",
        "ai.eval.view": "EMA",
    }
    for key, value in expected.items():
        if labels.get(key) != value:
            raise RuntimeError(f"image label {key} drift: {labels.get(key)!r}")
    limits = quota()
    if not limits["can_register"]:
        raise RuntimeError(f"quota refuses register: {limits}")
    if int(limits["active"]) >= 3:
        raise RuntimeError(f"concurrent slots full: active={limits['active']}")
    if int(limits["today"]) >= 6:
        raise RuntimeError(f"daily quota full: today={limits['today']}")
    challenge = make_request(urls.challenge_details.value.format(CHALLENGE_ID), "GET")
    image_size = int(image.attrs.get("Size", image.attrs.get("VirtualSize", 0)))
    if image_size > int(challenge["max_docker_image_size"]):
        raise RuntimeError("candidate image exceeds EvalAI size limit")
    return client, image, limits, runtime


def _save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def _push_image(candidate: dict[str, Any], client: Any, image: Any, runtime: tuple[Any, Any, Any, str, str]) -> dict[str, object]:
    _, boto3_module, evalai_runtime, aws_region, environment = runtime
    urls, make_request = evalai_runtime
    response = make_request(urls.get_aws_credentials.value.format(PHASE_ID), "GET")["success"]
    if not isinstance(response, dict):
        raise RuntimeError("EvalAI credential response schema drift")
    federated = response["federated_user"]
    repository_uri = response["docker_repository_uri"]
    credentials = federated["Credentials"]
    account_id = federated["FederatedUser"]["FederatedUserId"].split(":")[0]
    if environment != "PRODUCTION":
        raise RuntimeError("only the production EvalAI host is permitted")
    ecr = boto3_module.client(
        "ecr",
        region_name=aws_region,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )
    auth = ecr.get_authorization_token(registryIds=[account_id])
    username, password = base64.b64decode(auth["authorizationData"][0]["authorizationToken"]).decode().split(":", 1)
    registry = auth["authorizationData"][0]["proxyEndpoint"]
    client.login(username=username, password=password, registry=registry)
    tag = str(uuid.uuid4())
    submitted_image_uri = f"{repository_uri}:{tag}"
    image.tag(submitted_image_uri)
    for event in client.images.push(repository_uri, tag, stream=True, decode=True):
        if event.get("errorDetail") or event.get("error"):
            raise RuntimeError(f"Docker push failed: {event}")
    repository_name = repository_uri.split("/", 1)[1]
    manifest = ecr.describe_images(repositoryName=repository_name, imageIds=[{"imageTag": tag}])["imageDetails"][0][
        "imageDigest"
    ]
    if manifest != candidate["image_id"]:
        raise RuntimeError(f"uploaded digest mismatch: {manifest}")
    state = {
        "schema_version": "h1_temporal_trf_evalai_push_state",
        "arm": candidate["arm"],
        "image_tag": candidate["image_tag"],
        "image_id": candidate["image_id"],
        "payload_sha256": candidate["payload_sha256"],
        "repository_uri": repository_uri,
        "uuid_tag": tag,
        "submitted_image_uri": submitted_image_uri,
        "uploaded_manifest_digest": manifest,
        "pushed": True,
        "registered": False,
    }
    _save_state(Path(candidate["state_path"]), state)
    return state


def _already_registered(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in rows:
        if str(item.get("method_name") or "") == candidate["method_name"]:
            return item
    return None


def _register(candidate: dict[str, Any], state: dict[str, object], runtime: tuple[Any, Any, Any, str, str]) -> dict[str, object]:
    if state.get("registered"):
        raise RuntimeError(f"submission {state.get('submission_id')} is already registered")
    _, _, evalai_runtime, _, _ = runtime
    urls, make_request = evalai_runtime
    rows = list_submissions()
    existing = _already_registered(candidate, rows)
    if existing is not None:
        state.update({"registered": True, "submission_id": existing["id"], "recovered_from_list": True})
        _save_state(Path(candidate["state_path"]), state)
        return state
    limits = quota()
    if int(limits["active"]) >= 3 or int(limits["today"]) >= 6 or not limits["can_register"]:
        raise RuntimeError(f"refusing POST: quota={limits}")
    metadata = {
        "is_public": json.dumps(False),
        "method_name": candidate["method_name"],
        "method_description": candidate["method_description"],
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="h1_trf_evalai_") as tmp:
            submission_file = Path(tmp) / "submission.json"
            submission_file.write_text(
                json.dumps({"submitted_image_uri": state["submitted_image_uri"]}),
                encoding="utf-8",
            )
            response = make_request(
                urls.make_submission.value.format(CHALLENGE_ID, PHASE_ID),
                "POST",
                files=str(submission_file),
                data=metadata,
            )
    except Exception as exc:
        rows = list_submissions()
        recovered = _already_registered(candidate, rows)
        if recovered is not None:
            state.update(
                {
                    "registered": True,
                    "submission_id": recovered["id"],
                    "recovered_after_timeout": True,
                    "post_error": str(exc),
                }
            )
            _save_state(Path(candidate["state_path"]), state)
            return state
        raise
    state.update({"registered": True, "submission_id": response["id"], "server_response": response})
    _save_state(Path(candidate["state_path"]), state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    parser.add_argument("--confirm-payload-sha256", default="")
    args = parser.parse_args()
    candidate = load_candidate(Path(args.manifest))
    limits = quota()
    report = {
        "mode": "execute" if args.execute else "read_only_preflight",
        "arm": candidate["arm"],
        "quota": limits,
        "state_path": candidate["state_path"],
        "private": True,
        "tta": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if int(limits["active"]) >= 3 or int(limits["today"]) >= 6 or not limits["can_register"]:
        raise RuntimeError(f"refusing execute: quota={limits}")
    if args.confirm_image_id != candidate["image_id"]:
        raise RuntimeError("execute requires the complete immutable image ID")
    if args.confirm_payload_sha256 != candidate["payload_sha256"]:
        raise RuntimeError("execute requires the complete immutable payload SHA-256")
    client, image, limits, runtime = preflight(candidate)
    path = Path(candidate["state_path"])
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("registered"):
            raise RuntimeError(f"submission {state.get('submission_id')} already registered")
        if state.get("pushed") is not True:
            state = _push_image(candidate, client, image, runtime)
    else:
        state = _push_image(candidate, client, image, runtime)
    completed = _register(candidate, state, runtime)
    print(
        json.dumps(
            {
                "arm": completed["arm"],
                "submission_id": completed["submission_id"],
                "image_id": completed["image_id"],
                "payload_sha256": completed["payload_sha256"],
                "uploaded_manifest_digest": completed.get("uploaded_manifest_digest"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
