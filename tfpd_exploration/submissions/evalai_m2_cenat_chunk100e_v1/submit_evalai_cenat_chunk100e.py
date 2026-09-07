"""Guarded EvalAI submission helper for the Ce-NAT chunk100e continual TTA M2 image.

Default mode is a read-only authenticated preflight (still needs the account
token configured for the ``evalai`` CLI).  A formal private submission needs
``--execute`` plus the complete immutable image ID and payload SHA-256 of the
frozen candidate.  A durable state file prevents an accidental repeat of a
successful push/registration.

The payload SHA-256 is read from the immutable export receipt on disk; the
image tag is derived from it (first eight hex characters).  The image ID only
exists after ``docker build`` and MUST be supplied via
``--pin-image-id <full id>`` once, before any execute run; this helper refuses
to run with an unpinned image.

This helper was NOT executed while building the package: the operator owns
the single network step (the coordinator push), gated on explicit user
confirmation.
"""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHALLENGE_ID = 2319
PHASE_ID = 4599
PHASE_SLUG = "few-shot-test-2319"
TEAM_ID = 41975
CHECKPOINT_SHA256 = "c5672a2bfc94f55729d64e5eeb2f6917162c345713e7102dec4ad371cd147ea2"

ARM = "cenat_chunk100e_tta"

# MUST stay byte-identical to the Dockerfile LABEL ai.eval.method of this
# package (the preflight label guard fails closed on any drift).
METHOD_LABEL = (
    "Ce-NAT chunk100e continual TTA (matched joint-trained checkpoint; seed=first-30 calibration block; label-free energy-gated 100-bin chunk commits, running-median rule, capacity 30; identity recomputed per commit; decode-before-commit)"
)

METHOD_NAME = "M2 Ce-NAT chunk100e continual TTA (exploratory, sensitivity tier)"

METHOD_DESCRIPTION = (
    "Jointly fine-tuned SPINT M2 decoder (warm-started from the sealed "
    "Selected-T4 checkpoint; source training pools built by the same "
    "boundary-free chunk law as deployment). Runtime: seed activity pool = "
    "the session's public first-30 calibration block; during the eval stream "
    "every 100-bin window whose mean rate passes a label-free running-median "
    "gate is committed to the pool (capacity 30) and the cached identity is "
    "recomputed. Decode-before-commit; no labels, no trial boundaries, no "
    "backpropagation online. T4 few-shot calibration uses the 4 public "
    "calibration labels per session. Local held-out result: +0.0785 R2 over "
    "the static champion (6/6 sessions)."
)

BUDGET_DISCLOSURE = (
    "4 target labels (D-opt within first-30 finite-direction candidates) "
    "consumed by the ridge T4 fit; activity pool = the first 30 calibration "
    "trials (label-free neural activity only); continual updates are fully "
    "label-free (neural rate statistics only); offline fine-tuning used only "
    "the 7 public held-in calibration sessions"
)

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
        "value": True,
    },
    {
        "name": "IsPretrained",
        "type": "boolean",
        "description": "Pretrained outside FALCON?",
        "required": True,
        "value": False,
    },
]

HERE = Path(__file__).resolve().parent
RECEIPT_PATH = HERE / "artifacts" / "t4_m2_seed42_cenat_chunk100e_tta.receipt.json"


def _candidate() -> dict[str, Any]:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    payload_sha256 = receipt["chain"]["payload_sha256"]
    return {
        "arm": ARM,
        "image_tag": f"spint-t4-m2:cenat-chunk100e-s42-{payload_sha256[:8]}",
        "image_id": "",
        "payload_sha256": payload_sha256,
        "method_label": METHOD_LABEL,
        "method_name": METHOD_NAME,
        "method_description": METHOD_DESCRIPTION,
        "budget_disclosure": BUDGET_DISCLOSURE,
    }


def state_path() -> Path:
    return HERE / "artifacts" / "evalai_push_state_cenat_chunk100e_v1.json"


def plan_report(candidate: dict[str, Any]) -> dict[str, object]:
    return {
        "mode": "plan_only",
        "arm": candidate["arm"],
        "challenge_id": CHALLENGE_ID,
        "phase_id": PHASE_ID,
        "phase_slug": PHASE_SLUG,
        "team_id": TEAM_ID,
        "private": True,
        "image_tag": candidate["image_tag"],
        "image_id": candidate["image_id"] or "<unpinned: pass --pin-image-id after docker build>",
        "payload_sha256": candidate["payload_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "label_budget": 4,
        "activity_budget": 30,
        "budget_disclosure": candidate["budget_disclosure"],
        "method_name": candidate["method_name"],
        "method_description": candidate["method_description"],
        "submission_attributes": SUBMISSION_ATTRIBUTES,
        "required_runtime": {
            "task": "m2",
            "phase": "test",
            "batch_size": 7,
            "cached_identity_only": True,
            "online_calibration": False,
            "query_labels_used": False,
            "backpropagation": False,
        },
        "state_path": str(state_path()),
        "next_gate": "authenticated read-only preflight",
    }


def _runtime() -> tuple[Any, Any, Any, str, str]:
    """Import external clients only after local plan mode has returned."""
    import boto3
    import docker
    from evalai.utils.config import AWS_REGION, ENVIRONMENT
    from evalai.utils.requests import make_request
    from evalai.utils.urls import URLS

    return docker, boto3, (URLS, make_request), AWS_REGION, ENVIRONMENT


def _get(make_request: Any, path: str) -> dict[str, object]:
    value = make_request(path, "GET")
    if not isinstance(value, dict):
        raise RuntimeError(f"EvalAI GET returned non-object for {path}")
    return value


def _validate_local_image(docker_module: Any, candidate: dict[str, Any]) -> tuple[Any, Any]:
    client = docker_module.from_env()
    image = client.images.get(candidate["image_tag"])
    if candidate["image_id"] and image.id != candidate["image_id"]:
        raise RuntimeError(f"image ID guard failed: {image.id} != {candidate['image_id']}")
    labels = image.attrs.get("Config", {}).get("Labels", {}) or {}
    expected = {
        "ai.eval.payload.sha256": candidate["payload_sha256"],
        "ai.eval.checkpoint.sha256": CHECKPOINT_SHA256,
        "ai.eval.label_budget": "4",
        "ai.eval.activity_budget": "30",
        "ai.eval.method": candidate["method_label"],
    }
    for key, value in expected.items():
        if labels.get(key) != value:
            raise RuntimeError(f"image label {key} drift: {labels.get(key)!r}")

    return client, image


def preflight(candidate: dict[str, Any], image_id: str) -> tuple[dict[str, object], Any, Any, dict[str, object], dict[str, object], tuple[Any, Any, Any, str, str]]:
    runtime = _runtime()
    docker_module, _, evalai_runtime, _, _ = runtime
    urls, make_request = evalai_runtime
    candidate = dict(candidate)
    candidate["image_id"] = image_id
    client, image = _validate_local_image(docker_module, candidate)

    phase = _get(make_request, urls.phase_details_using_slug.value.format(PHASE_SLUG))
    if phase.get("id") != PHASE_ID or phase.get("challenge") != CHALLENGE_ID:
        raise RuntimeError("EvalAI phase identity changed")
    if not phase.get("is_active") or phase.get("is_submission_paused"):
        raise RuntimeError("EvalAI phase is not accepting submissions")
    challenge = _get(make_request, urls.challenge_details.value.format(CHALLENGE_ID))
    image_size = int(image.attrs.get("Size", image.attrs.get("VirtualSize", 0)))
    if image_size > int(challenge["max_docker_image_size"]):
        raise RuntimeError("candidate image exceeds EvalAI size limit")

    submissions = _get(
        make_request, urls.my_submissions.value.format(CHALLENGE_ID, PHASE_ID)
    )
    results = submissions.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("EvalAI submission list schema drift")
    now = datetime.now(timezone.utc)
    today = 0
    month = 0
    active = 0
    for item in results:
        stamp = datetime.fromisoformat(str(item["submitted_at"]).replace("Z", "+00:00"))
        today += int(stamp.date() == now.date())
        month += int((stamp.year, stamp.month) == (now.year, now.month))
        active += int(item.get("status") in {"submitted", "queued", "running"})
    limits = {
        "today": today,
        "month": month,
        "total": int(submissions.get("count", len(results))),
        "active": active,
        "max_per_day": int(phase["max_submissions_per_day"]),
        "max_per_month": int(phase["max_submissions_per_month"]),
        "max_total": int(phase["max_submissions"]),
        "max_concurrent": int(phase["max_concurrent_submissions_allowed"]),
    }
    for used, maximum, label in (
        (limits["today"], limits["max_per_day"], "daily"),
        (limits["month"], limits["max_per_month"], "monthly"),
        (limits["total"], limits["max_total"], "total"),
        (limits["active"], limits["max_concurrent"], "concurrent"),
    ):
        if used >= maximum:
            raise RuntimeError(f"{label} submission quota is exhausted")
    return candidate, client, image, phase, limits, runtime


def _save_state(state: dict[str, object]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def _push_image(client: Any, image: Any, candidate: dict[str, Any], runtime: tuple[Any, Any, Any, str, str]) -> dict[str, object]:
    _, boto3_module, evalai_runtime, aws_region, environment = runtime
    urls, make_request = evalai_runtime
    response = _get(make_request, urls.get_aws_credentials.value.format(PHASE_ID))["success"]
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
    if manifest != candidate["image_id"]:
        raise RuntimeError(f"uploaded digest mismatch: {manifest}")
    state = {
        "schema_version": "m2_cenat_chunk100e_evalai_push_state_v1",
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
    _save_state(state)
    return state


def _register(state: dict[str, object], candidate: dict[str, Any], runtime: tuple[Any, Any, Any, str, str]) -> dict[str, object]:
    if state.get("registered"):
        raise RuntimeError(f"submission {state.get('submission_id')} is already registered")
    if (
        state.get("arm") != candidate["arm"]
        or state.get("image_id") != candidate["image_id"]
        or state.get("payload_sha256") != candidate["payload_sha256"]
        or state.get("pushed") is not True
    ):
        raise RuntimeError("push state does not match the frozen candidate")
    _, _, evalai_runtime, _, _ = runtime
    urls, make_request = evalai_runtime
    metadata = {
        "is_public": json.dumps(False),
        "method_name": candidate["method_name"],
        "method_description": candidate["method_description"],
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
    }
    with tempfile.TemporaryDirectory(prefix="m2_cenat_chunk100e_evalai_") as tmp:
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
    state.update(
        {"registered": True, "submission_id": response["id"], "server_response": response}
    )
    _save_state(state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pin-image-id", default="", help="full image ID after docker build")
    parser.add_argument("--confirm-image-id", default="")
    parser.add_argument("--confirm-payload-sha256", default="")
    args = parser.parse_args()
    candidate = _candidate()
    if args.plan and args.execute:
        raise RuntimeError("--plan and --execute are mutually exclusive")
    if args.plan:
        print(json.dumps(plan_report(candidate), indent=2, sort_keys=True))
        return

    candidate, client, image, phase, limits, runtime = preflight(candidate, args.pin_image_id)
    report = {
        "mode": "execute" if args.execute else "read_only_preflight",
        "arm": candidate["arm"],
        "image_tag": candidate["image_tag"],
        "image_id": image.id,
        "image_size": image.attrs.get("Size"),
        "phase_id": phase["id"],
        "phase_slug": phase["slug"],
        "phase_active": phase["is_active"],
        "team_id": TEAM_ID,
        "private": True,
        "quota": limits,
        "budget_disclosure": candidate["budget_disclosure"],
        "state_path": str(state_path()),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.confirm_image_id != candidate["image_id"]:
        raise RuntimeError("execute mode requires the complete immutable image ID")
    if args.confirm_payload_sha256 != candidate["payload_sha256"]:
        raise RuntimeError("execute mode requires the complete immutable payload SHA-256")

    path = state_path()
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("registered"):
            raise RuntimeError(f"submission {state.get('submission_id')} already registered")
    else:
        state = _push_image(client, image, candidate, runtime)
    completed = _register(state, candidate, runtime)
    print(
        json.dumps(
            {
                "arm": completed["arm"],
                "submission_id": completed["submission_id"],
                "image_id": completed["image_id"],
                "payload_sha256": completed["payload_sha256"],
                "uploaded_manifest_digest": completed["uploaded_manifest_digest"],
                "uuid_tag": completed["uuid_tag"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
