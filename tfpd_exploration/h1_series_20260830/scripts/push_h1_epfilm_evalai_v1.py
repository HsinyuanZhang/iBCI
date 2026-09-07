#!/usr/bin/env python3
"""Guarded EvalAI push helper for the H1 EP-FILM cached-identity submission.

Adapted from the established sua_exploration submit helper: quota-checked,
label-guarded, one-shot (uuid ECR tag + digest verification + registration),
with a mutable push-state receipt under the h1_epfilm_evalai_v1 artifacts.
Run with an interpreter that has docker, boto3 and evalai installed.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import docker
from evalai.utils.config import AWS_REGION, ENVIRONMENT
from evalai.utils.requests import make_request
from evalai.utils.urls import URLS


CHALLENGE_ID = 2319
PHASE_ID = 4599
PHASE_SLUG = "few-shot-test-2319"
TEAM_ID = 41975
PAYLOAD_SHA256 = "df71cb9329a87b5073242044b2933391ced7bf866d1f481e994f71dbd97ba2d7"
CHECKPOINT_SHA256 = "0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06"
FILM_STATE_SHA256 = "b602c2e090f455cb6259fc76fb9a225bfdb9aa6bf40adcd5ccd10b9a0da13b37"
IMAGE_TAG_DEFAULT = f"h1-epfilm-c1:evalai-v1-{PAYLOAD_SHA256[:8]}"
STATE_PATH = (
    Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/h1_series_20260830/"
         "artifacts/h1_epfilm_evalai_v1/evalai_push_state.json")
)
METHOD_NAME = "H1 EP-FILM cached identity (all-source, sensitivity)"
METHOD_DESCRIPTION = (
    "Frozen all-source C1/M3 carrier-aware SPINT decoder for FALCON H1 with an "
    "early-pooling calibration-profile FiLM (648 parameters, H=32, profile4 = "
    "low/high speed state contrast, carrier4 context) trained on all held-in "
    "public calibration dates. Per-session identities and MAT7 readout maps are "
    "computed offline from each session's three public calibration trials only; "
    "the official runtime uses cached identities, performs no query-label read, "
    "no online calibration, no optimizer step, and no backpropagation."
)

SUBMISSION_ATTRIBUTES = [
    {"name": "IsHeldOutZeroShot", "type": "boolean", "description": "Is held out prediction Zero Shot?",
     "required": True, "value": False},
    {"name": "IsTestTimeAdaptive", "type": "boolean", "description": "Test Time Adaptive?",
     "required": True, "value": False},
    {"name": "IsPretrained", "type": "boolean", "description": "Pretrained outside FALCON?",
     "required": True, "value": False},
]


def get(path: str) -> dict:
    return make_request(path, "GET")


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pending = STATE_PATH.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    pending.replace(STATE_PATH)


def load_candidate(image_tag: str) -> dict:
    if not isinstance(image_tag, str) or not image_tag or any(c.isspace() for c in image_tag):
        raise ValueError("image tag is missing or invalid")
    return {
        "image_tag": image_tag,
        "payload_sha256": PAYLOAD_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "film_state_sha256": FILM_STATE_SHA256,
        "method_name": METHOD_NAME,
        "method_description": METHOD_DESCRIPTION,
    }


def preflight(candidate: dict):
    client = docker.from_env()
    image = client.images.get(candidate["image_tag"])
    labels = image.attrs.get("Config", {}).get("Labels", {}) or {}
    if labels.get("ibci.h1.package.sha256") != candidate["payload_sha256"]:
        raise RuntimeError("payload label guard failed")
    if labels.get("ibci.h1.checkpoint.sha256") != candidate["checkpoint_sha256"]:
        raise RuntimeError("checkpoint label guard failed")
    if labels.get("ibci.h1.film_state.sha256") != candidate["film_state_sha256"]:
        raise RuntimeError("film state label guard failed")
    if labels.get("ibci.h1.calibration.trials") != "3":
        raise RuntimeError("calibration trial label guard failed")
    phase = get(URLS.phase_details_using_slug.value.format(PHASE_SLUG))
    if phase.get("id") != PHASE_ID or phase.get("challenge") != CHALLENGE_ID:
        raise RuntimeError("EvalAI phase identity changed")
    if not phase.get("is_active") or phase.get("is_submission_paused"):
        raise RuntimeError("EvalAI phase is not accepting submissions")
    challenge = get(URLS.challenge_details.value.format(CHALLENGE_ID))
    image_size = int(image.attrs.get("Size", image.attrs.get("VirtualSize", 0)))
    if image_size > int(challenge["max_docker_image_size"]):
        raise RuntimeError("Image exceeds challenge size limit")
    submissions = get(URLS.my_submissions.value.format(CHALLENGE_ID, PHASE_ID))
    results = submissions.get("results", [])
    now = datetime.now(timezone.utc)

    def _when(row):
        return datetime.fromisoformat(row["submitted_at"].replace("Z", "+00:00"))

    limits = {
        "today": sum(_when(row).date() == now.date() for row in results),
        "month": sum((_when(row).year, _when(row).month) == (now.year, now.month) for row in results),
        "total": int(submissions.get("count", len(results))),
        "active": sum(row.get("status") in {"submitted", "queued", "running"} for row in results),
        "max_per_day": int(phase["max_submissions_per_day"]),
        "max_per_month": int(phase["max_submissions_per_month"]),
        "max_total": int(phase["max_submissions"]),
        "max_concurrent": int(phase["max_concurrent_submissions_allowed"]),
    }
    if limits["today"] >= limits["max_per_day"]:
        raise RuntimeError("Daily submission quota exhausted")
    if limits["month"] >= limits["max_per_month"]:
        raise RuntimeError("Monthly submission quota exhausted")
    if limits["total"] >= limits["max_total"]:
        raise RuntimeError("Total submission quota exhausted")
    if limits["active"] >= limits["max_concurrent"]:
        raise RuntimeError("Concurrent submission quota exhausted")
    return candidate, client, image, phase, challenge, limits


def push_image(candidate: dict, client, image) -> dict:
    response = get(URLS.get_aws_credentials.value.format(PHASE_ID))["success"]
    federated = response["federated_user"]
    repository_uri = response["docker_repository_uri"]
    credentials = federated["Credentials"]
    account_id = federated["FederatedUser"]["FederatedUserId"].split(":")[0]
    if ENVIRONMENT != "PRODUCTION":
        raise RuntimeError("Only production EvalAI is permitted")
    ecr = boto3.client(
        "ecr",
        region_name=AWS_REGION,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )
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
    repository_name = repository_uri.split("/", 1)[1]
    manifest = ecr.describe_images(
        repositoryName=repository_name, imageIds=[{"imageTag": tag}]
    )["imageDetails"][0]["imageDigest"]
    if manifest != image.id:
        raise RuntimeError("Uploaded digest mismatch")
    state = {
        "schema_version": "h1_epfilm_evalai_v1_push_state",
        "arm": "epfilm",
        "image_tag": candidate["image_tag"],
        "image_id": image.id,
        "payload_sha256": candidate["payload_sha256"],
        "repository_uri": repository_uri,
        "uuid_tag": tag,
        "submitted_image_uri": submitted_image_uri,
        "uploaded_manifest_digest": manifest,
        "pushed": True,
        "registered": False,
    }
    save_state(state)
    return state


def register(candidate: dict, state: dict) -> dict:
    if state.get("registered"):
        raise RuntimeError(f"Submission {state.get('submission_id')} already registered")
    payload = {"submitted_image_uri": state["submitted_image_uri"]}
    metadata = {
        "is_public": json.dumps(False),
        "method_name": candidate["method_name"],
        "method_description": candidate["method_description"],
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
    }
    with tempfile.TemporaryDirectory(prefix="h1_epfilm_evalai_") as tmp:
        submission_file = Path(tmp) / "submission.json"
        submission_file.write_text(json.dumps(payload))
        response = make_request(
            URLS.make_submission.value.format(CHALLENGE_ID, PHASE_ID),
            "POST",
            files=str(submission_file),
            data=metadata,
        )
    state.update({"registered": True, "submission_id": response["id"], "server_response": response})
    save_state(state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-tag", default=IMAGE_TAG_DEFAULT)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    args = parser.parse_args()
    if args.plan and args.execute:
        raise RuntimeError("--plan and --execute are mutually exclusive")
    candidate = load_candidate(args.image_tag)
    if args.plan:
        print(json.dumps({
            "mode": "plan_only",
            "challenge_id": CHALLENGE_ID,
            "phase_id": PHASE_ID,
            "phase_slug": PHASE_SLUG,
            "team_id": TEAM_ID,
            "image_tag": candidate["image_tag"],
            "payload_sha256": candidate["payload_sha256"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "film_state_sha256": candidate["film_state_sha256"],
            "method_name": candidate["method_name"],
            "submission_attributes": SUBMISSION_ATTRIBUTES,
            "state_path": str(STATE_PATH),
        }, indent=2, sort_keys=True))
        return
    loaded, client, image, phase, challenge, limits = preflight(candidate)
    report = {
        "mode": "execute" if args.execute else "read_only_preflight",
        "image_tag": loaded["image_tag"],
        "image_id": image.id,
        "image_size": image.attrs.get("Size"),
        "max_image_size": challenge["max_docker_image_size"],
        "phase_id": phase["id"],
        "phase_slug": phase["slug"],
        "phase_active": phase["is_active"],
        "team_id": TEAM_ID,
        "private": True,
        "method_name": loaded["method_name"],
        "quota": limits,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.confirm_image_id != image.id:
        raise RuntimeError("Execute mode requires the complete immutable image ID")
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else push_image(loaded, client, image)
    if state.get("registered"):
        raise RuntimeError(f"Submission {state.get('submission_id')} already registered")
    completed = register(loaded, state)
    print(json.dumps({
        "arm": "epfilm",
        "submission_id": completed["submission_id"],
        "image_id": completed["image_id"],
        "uploaded_manifest_digest": completed["uploaded_manifest_digest"],
        "uuid_tag": completed["uuid_tag"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
