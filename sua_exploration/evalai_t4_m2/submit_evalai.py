"""Guarded EvalAI submission helper for the frozen T4 M2 image.

The installed EvalAI 1.3.18 client does not register an image after a
successful Docker 7 push because modern docker-py no longer emits the legacy
``aux.Tag`` event.  This helper preserves the official API sequence, verifies
the ECR manifest digest, and registers exactly once.  Its default mode is a
read-only preflight; external writes require both ``--execute`` and the full
immutable image ID.
"""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import docker

from evalai.utils.config import AWS_REGION, ENVIRONMENT
from evalai.utils.requests import make_request
from evalai.utils.urls import URLS


IMAGE_TAG = "spint-t4-m2:e8-seed42-epoch002-dcc449a-r3"
IMAGE_ID = "sha256:57b1fb2418ad2fd8f2d4e62f8fddb6d4f75b9c8a072730bf7afb6af53517d260"
IMAGE_PAYLOAD_SHA256 = (
    "dcc449a15bc478f3380c95add964fc344522a25bc9938c0a5563bf5a75ae0c96"
)
CHALLENGE_ID = 2319
PHASE_ID = 4599
PHASE_SLUG = "few-shot-test-2319"
TEAM_ID = 41975
STATE_PATH = Path(__file__).resolve().parent / "artifacts" / "evalai_push_state.json"

METHOD_NAME = "T4 cached identity M2 M33 seed42"
METHOD_DESCRIPTION = (
    "Frozen SPINT decoder with per-session T4 identity calibrated offline from "
    "the chronological first 33 public calibration trials and their trial target "
    "directions. No hidden query labels, optimizer, or backpropagation are used "
    "by the EvalAI runtime."
)

SUBMISSION_ATTRIBUTES = [
    {
        "name": "IsHeldOutZeroShot",
        "type": "boolean",
        "description": (
            "Is held out prediction Zero Shot? (Does not calibrate with any data "
            "from held out days)"
        ),
        "required": True,
        "value": False,
    },
    {
        "name": "IsTestTimeAdaptive",
        "type": "boolean",
        "description": "Test Time Adaptive? (Runs recalibration at test time)",
        "required": True,
        "value": False,
    },
    {
        "name": "IsPretrained",
        "type": "boolean",
        "description": "Pretrained? (Uses data outside FALCON datasets)",
        "required": True,
        "value": False,
    },
]


def _get(path: str) -> dict:
    return make_request(path, "GET")


def _local_image() -> tuple[docker.DockerClient, object]:
    client = docker.from_env()
    image = client.images.get(IMAGE_TAG)
    if image.id != IMAGE_ID:
        raise RuntimeError(f"image-ID guard failed: {image.id} != {IMAGE_ID}")
    labels = image.attrs.get("Config", {}).get("Labels", {}) or {}
    if labels.get("ai.eval.payload.sha256") != IMAGE_PAYLOAD_SHA256:
        raise RuntimeError("payload hash label does not match the frozen candidate")
    return client, image


def _preflight() -> tuple[dict, dict, dict, object, docker.DockerClient]:
    client, image = _local_image()
    phase = _get(URLS.phase_details_using_slug.value.format(PHASE_SLUG))
    if phase.get("id") != PHASE_ID or phase.get("challenge") != CHALLENGE_ID:
        raise RuntimeError("EvalAI phase identity changed")
    if not phase.get("is_active") or phase.get("is_submission_paused"):
        raise RuntimeError("EvalAI target phase is not currently accepting submissions")

    challenge = _get(URLS.challenge_details.value.format(CHALLENGE_ID))
    image_size = int(image.attrs.get("Size", image.attrs.get("VirtualSize", 0)))
    max_size = int(challenge["max_docker_image_size"])
    if image_size > max_size:
        raise RuntimeError(f"image is too large: {image_size} > {max_size}")

    submission_path = URLS.my_submissions.value.format(CHALLENGE_ID, PHASE_ID)
    submissions = _get(submission_path)
    results = submissions.get("results", [])
    now = datetime.now(timezone.utc)
    today = 0
    this_month = 0
    active = 0
    for item in results:
        stamp = datetime.fromisoformat(item["submitted_at"].replace("Z", "+00:00"))
        today += stamp.date() == now.date()
        this_month += (stamp.year, stamp.month) == (now.year, now.month)
        active += item.get("status") in {"submitted", "queued", "running"}
    total = int(submissions.get("count", len(results)))
    limits = {
        "today": today,
        "month": this_month,
        "total": total,
        "active": active,
        "max_per_day": int(phase["max_submissions_per_day"]),
        "max_per_month": int(phase["max_submissions_per_month"]),
        "max_total": int(phase["max_submissions"]),
        "max_concurrent": int(phase["max_concurrent_submissions_allowed"]),
    }
    if today >= limits["max_per_day"]:
        raise RuntimeError("daily submission quota is exhausted")
    if this_month >= limits["max_per_month"]:
        raise RuntimeError("monthly submission quota is exhausted")
    if total >= limits["max_total"]:
        raise RuntimeError("total submission quota is exhausted")
    if active >= limits["max_concurrent"]:
        raise RuntimeError("concurrent submission quota is exhausted")
    return phase, challenge, limits, image, client


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pending = STATE_PATH.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    pending.replace(STATE_PATH)


def _push_image(client: docker.DockerClient, image: object) -> dict:
    credentials_path = URLS.get_aws_credentials.value.format(PHASE_ID)
    response = _get(credentials_path)
    success = response["success"]
    federated = success["federated_user"]
    repository_uri = success["docker_repository_uri"]
    credentials = federated["Credentials"]
    account_id = federated["FederatedUser"]["FederatedUserId"].split(":")[0]
    ecr = boto3.client(
        "ecr",
        region_name=AWS_REGION,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )
    if ENVIRONMENT != "PRODUCTION":
        raise RuntimeError("this frozen helper only permits the production EvalAI host")
    auth = ecr.get_authorization_token(registryIds=[account_id])
    username, password = base64.b64decode(
        auth["authorizationData"][0]["authorizationToken"]
    ).decode().split(":", 1)
    registry = auth["authorizationData"][0]["proxyEndpoint"]
    client.login(username=username, password=password, registry=registry)

    tag = str(uuid.uuid4())
    submitted_image_uri = f"{repository_uri}:{tag}"
    image.tag(submitted_image_uri)
    for event in client.images.push(repository_uri, tag, stream=True, decode=True):
        if event.get("errorDetail") or event.get("error"):
            raise RuntimeError(f"Docker push failed: {event}")
        status = event.get("status", "")
        if status and (status.startswith("digest:") or status.startswith("latest:")):
            print(status, flush=True)

    repository_name = repository_uri.split("/", 1)[1]
    manifest = ecr.describe_images(
        repositoryName=repository_name, imageIds=[{"imageTag": tag}]
    )["imageDetails"][0]["imageDigest"]
    if manifest != IMAGE_ID:
        raise RuntimeError(f"uploaded digest mismatch: {manifest} != {IMAGE_ID}")
    state = {
        "schema_version": "e8_t4_m2_evalai_push_state_v1",
        "image_tag": IMAGE_TAG,
        "image_id": IMAGE_ID,
        "repository_uri": repository_uri,
        "uuid_tag": tag,
        "submitted_image_uri": submitted_image_uri,
        "uploaded_manifest_digest": manifest,
        "pushed": True,
        "registered": False,
    }
    _save_state(state)
    return state


def _register(state: dict) -> dict:
    if state.get("registered"):
        raise RuntimeError(
            f"state already contains submission {state.get('submission_id')}; refusing duplicate"
        )
    if state.get("image_id") != IMAGE_ID or not state.get("pushed"):
        raise RuntimeError("push state is absent or does not match the frozen image")
    payload = {"submitted_image_uri": state["submitted_image_uri"]}
    metadata = {
        "is_public": json.dumps(False),
        "method_name": METHOD_NAME,
        "method_description": METHOD_DESCRIPTION,
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
    }
    with tempfile.TemporaryDirectory(prefix="e8_t4_evalai_") as tmp:
        submission_file = Path(tmp) / "submission.json"
        submission_file.write_text(json.dumps(payload))
        endpoint = URLS.make_submission.value.format(CHALLENGE_ID, PHASE_ID)
        response = make_request(
            endpoint, "POST", files=str(submission_file), data=metadata
        )
    state.update(
        {
            "registered": True,
            "submission_id": response["id"],
            "server_response": response,
        }
    )
    _save_state(state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="push and register the formal private submission",
    )
    parser.add_argument(
        "--confirm-image-id",
        default="",
        help="required full immutable image ID in execute mode",
    )
    args = parser.parse_args()

    phase, challenge, limits, image, client = _preflight()
    report = {
        "mode": "execute" if args.execute else "read_only_preflight",
        "image_tag": IMAGE_TAG,
        "image_id": image.id,
        "image_size": image.attrs.get("Size"),
        "max_image_size": challenge["max_docker_image_size"],
        "phase_id": phase["id"],
        "phase_slug": phase["slug"],
        "phase_active": phase["is_active"],
        "team_id": TEAM_ID,
        "private": True,
        "quota": limits,
        "submission_attributes": {
            item["name"]: item["value"] for item in SUBMISSION_ATTRIBUTES
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute:
        return
    if args.confirm_image_id != IMAGE_ID:
        raise RuntimeError("execute mode requires the complete frozen --confirm-image-id")

    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        if state.get("registered"):
            raise RuntimeError(
                f"submission {state.get('submission_id')} already registered; refusing duplicate"
            )
        print("Resuming registration from the recorded successful image push.")
    else:
        state = _push_image(client, image)
    completed = _register(state)
    print(
        json.dumps(
            {
                "submission_id": completed["submission_id"],
                "image_id": completed["image_id"],
                "uploaded_manifest_digest": completed["uploaded_manifest_digest"],
                "uuid_tag": completed["uuid_tag"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
