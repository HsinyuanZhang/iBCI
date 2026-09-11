#!/usr/bin/env python3
"""Guarded EvalAI push/register for the packed M1 RIFT ACTIVITY_ONLY R100 cached image.

Use the SPINT conda Python with the user EvalAI site-packages on sys.path.
Does nothing unless --execute is passed with the immutable image/payload IDs.
POST only when the server's phase-specific daily/month/total counters and the
phase's current concurrent limit permit it.
On POST timeout, GET the team/phase list before any retry.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

sys.path.insert(0, "/home/xinyuan/.local/lib/python3.10/site-packages")
for _k in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_k, "")
    if "eval.ai" not in _val:
        os.environ[_k] = f"{_val},eval.ai".strip(",")

CHALLENGE_ID = 2319
PHASE_ID = 4599
PHASE_SLUG = "few-shot-test-2319"

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


def _get(make_request: Any, path: str) -> dict[str, object]:
    value = make_request(path, "GET")
    if not isinstance(value, dict):
        raise RuntimeError(f"EvalAI GET returned non-object for {path}")
    return value


def _remaining(make_request: Any) -> Mapping[str, Any]:
    """Read the authoritative server-side quota record immediately before writes."""
    return _get(make_request, f"/api/jobs/{CHALLENGE_ID}/remaining_submissions/")


def _remaining_positive(value: Mapping[str, Any]) -> dict[str, int]:
    """Require the exact server quota record for phase 4599.

    The endpoint returns a challenge envelope whose ``phases`` member is a
    list.  Server daily/month/total counters are authoritative; do not infer
    them from an all-status submission list.
    """
    phases = value.get("phases")
    if not isinstance(phases, list):
        raise RuntimeError("remaining_submissions response lacks phases list")
    matched = [row for row in phases if isinstance(row, Mapping) and row.get("id") == PHASE_ID]
    if len(matched) != 1:
        raise RuntimeError(f"remaining_submissions response must contain exactly one phase {PHASE_ID}")
    phase = matched[0]
    if phase.get("slug") != PHASE_SLUG:
        raise RuntimeError("remaining_submissions phase slug drift")
    limits = phase.get("limits")
    if not isinstance(limits, Mapping):
        raise RuntimeError("remaining_submissions phase lacks limits object")
    names = (
        "remaining_submissions_today_count",
        "remaining_submissions_this_month_count",
        "remaining_submissions_count",
    )
    parsed: dict[str, int] = {}
    for name in names:
        raw = limits.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise RuntimeError(f"authoritative phase {PHASE_ID} quota is absent, non-integer, or exhausted: {name}={raw!r}")
        parsed[name] = raw
    return parsed


def _verify_host_receipt(candidate: Mapping[str, Any]) -> None:
    receipt_path = Path(str(candidate["host_receipt_path"])); payload_path = Path(str(candidate["payload_path"]))
    if not receipt_path.is_file() or not payload_path.is_file():
        raise RuntimeError("manifest host receipt or payload path is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "HOST_PACK_VERIFY_PASS":
        raise RuntimeError("host verification receipt is not passing")
    actual = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    if actual != str(candidate["payload_sha256"]):
        raise RuntimeError("payload file SHA-256 does not match immutable manifest")
    if receipt.get("payload_sha256") != actual:
        raise RuntimeError("host receipt payload SHA-256 does not bind actual payload")
    if receipt.get("route") != {
        "live_b_ema_zero_side_vs_static_checked": True,
        "sealed_t_literal_zero": True,
        "static_e0_sha_checked_by_runtime": True,
    }:
        raise RuntimeError("host verification information-route receipt drift")
    if receipt.get("raw_input_contract") != {
        "query_pad_bins_stripped": 99,
        "validity": "derived from raw end geometry, never neural amplitude",
    }:
        raise RuntimeError("host verification raw-input receipt drift")
    if receipt.get("B8_inactive_row_checked") is not True or receipt.get("B8_true_zero_valid_checked") is not True or receipt.get("full_stream_startup_truezero_checked") is not True:
        raise RuntimeError("host verification streaming/inactive checks are incomplete")
    for name in ("max_live_b_ema_vs_static", "max_packed_vs_full", "B8_cached_vs_full_max_abs", "querypad_t0_full_stream_max_abs", "full_context_stream_truezero_max_abs"):
        value = receipt.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 2e-5:
            raise RuntimeError(f"host verification tolerance is invalid for {name}: {value!r}")
    tags = receipt.get("per_ho_tag")
    if not isinstance(tags, Mapping) or set(tags) != {"20121004", "20121017", "20121024"}:
        raise RuntimeError("host verification held-out tag roster drift")
    for tag, row in tags.items():
        if not isinstance(row, Mapping) or row.get("steps") != 128 or row.get("full_context_endpoint_checked") != 99 or row.get("startup_endpoints_checked") != [0, 1] or row.get("true_zero_valid_endpoint") != 1:
            raise RuntimeError(f"host verification endpoint check drift for {tag}")
        for name in ("live_b_ema_vs_static_max_abs", "packed_vs_full_max_abs"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 2e-5:
                raise RuntimeError(f"host verification per-tag tolerance invalid for {tag}/{name}")


def list_submissions(make_request: Any, urls: Any) -> list[dict[str, Any]]:
    page = _get(make_request, urls.my_submissions.value.format(CHALLENGE_ID, PHASE_ID))
    rows = list(page.get("results") or [])
    nxt = page.get("next")
    while nxt:
        parsed = urlparse(str(nxt))
        page = _get(make_request, parsed.path + ("?" + parsed.query if parsed.query else ""))
        rows.extend(page.get("results") or [])
        nxt = page.get("next")
    return rows


def _quota(phase: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use list data only for the phase's current concurrent occupancy."""
    active = 0
    active_ids = []
    for item in rows:
        if item.get("status") in {"submitted", "submitting", "resuming", "queued", "running"}:
            active += 1
            active_ids.append({"id": item.get("id"), "status": item.get("status"), "method": item.get("method_name")})
    return {"active": active, "active_ids": active_ids,
            "max_concurrent": int(phase["max_concurrent_submissions_allowed"])}


def _require_quota(limits: Mapping[str, Any]) -> None:
    if int(limits["active"]) >= int(limits["max_concurrent"]):
        raise RuntimeError(f"concurrent slots full: active={limits['active']} limit={limits['max_concurrent']}")


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
        "host_receipt_path",
        "payload_path",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"candidate manifest missing {missing}")
    return payload


def preflight(candidate: dict[str, Any]) -> tuple[Any, Any, dict[str, Any], dict[str, Any], tuple[Any, Any, Any, str, str]]:
    _verify_host_receipt(candidate)
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
        "ai.eval.method": candidate["method_label"],
        "ai.eval.task": "m1",
        "ai.eval.architecture": "RIFT",
        "ai.eval.arm": "B_ACTIVITY_ONLY",
        "ai.eval.identity": "ema_b3s_zero_side_static_e0_concat",
        "ai.eval.dataloader_workers": "0",
    }
    for key, value in expected.items():
        if labels.get(key) != value:
            raise RuntimeError(f"image label {key} drift: {labels.get(key)!r}")
    phase = _get(make_request, urls.phase_details_using_slug.value.format(PHASE_SLUG))
    if phase.get("id") != PHASE_ID or phase.get("challenge") != CHALLENGE_ID:
        raise RuntimeError("EvalAI phase identity changed")
    if not phase.get("is_active") or phase.get("is_submission_paused"):
        raise RuntimeError("EvalAI phase is not accepting submissions")
    challenge = _get(make_request, urls.challenge_details.value.format(CHALLENGE_ID))
    image_size = int(image.attrs.get("Size", image.attrs.get("VirtualSize", 0)))
    if image_size > int(challenge["max_docker_image_size"]):
        raise RuntimeError("candidate image exceeds EvalAI size limit")
    limits = _quota(phase, list_submissions(make_request, urls))
    _require_quota(limits)
    remaining = _remaining(make_request)
    authoritative = _remaining_positive(remaining)
    limits["authoritative_remaining"] = dict(remaining)
    limits["authoritative_phase_limits"] = authoritative
    return client, image, phase, limits, runtime


def _save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def _push_image(candidate: dict[str, Any], client: Any, image: Any, runtime: tuple[Any, Any, Any, str, str]) -> dict[str, object]:
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
    detail = ecr.describe_images(repositoryName=repository_name, imageIds=[{"imageTag": tag}])["imageDetails"][0]
    manifest = str(detail["imageDigest"])
    remote = ecr.batch_get_image(repositoryName=repository_name, imageIds=[{"imageTag": tag}])
    images = list(remote.get("images") or [])
    if len(images) != 1 or images[0].get("imageId", {}).get("imageDigest") != manifest:
        raise RuntimeError("ECR manifest lookup does not bind the pushed tag to the described digest")
    try:
        manifest_body = json.loads(str(images[0]["imageManifest"]))
        config_digest = str(manifest_body["config"]["digest"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("ECR returned an unsupported image manifest") from exc
    local_image_id = str(image.id)
    if local_image_id == manifest:
        digest_binding = "local_manifest_digest"
    elif local_image_id == config_digest:
        digest_binding = "local_config_digest"
    else:
        raise RuntimeError(f"uploaded manifest config does not bind local Docker image ID: local={local_image_id}, config={config_digest}, manifest={manifest}")
    state = {
        "schema_version": "m1_rift_activity_only_evalai_push_state",
        "arm": candidate["arm"],
        "image_tag": candidate["image_tag"],
        "image_id": candidate["image_id"],
        "payload_sha256": candidate["payload_sha256"],
        "repository_uri": repository_uri,
        "uuid_tag": tag,
        "submitted_image_uri": submitted_image_uri,
        "uploaded_manifest_digest": manifest,
        "uploaded_config_digest": config_digest,
        "digest_binding": digest_binding,
        "pushed": True,
        "registered": False,
    }
    _save_state(Path(candidate["state_path"]), state)
    return state


def _item_image_uri(item: Mapping[str, Any]) -> str | None:
    for key in ("submitted_image_uri", "image_uri", "docker_image", "submitted_file"):
        value = item.get(key)
        if isinstance(value, str) and value: return value
    return None


def _resolve_submission_uri(make_request: Any, urls: Any, item: Mapping[str, Any]) -> str:
    """Resolve an omitted list-row URI before deciding whether POST is safe."""
    submission_id = item.get("id")
    if isinstance(submission_id, bool) or not isinstance(submission_id, int):
        raise RuntimeError("same-method list row lacks an integer submission id")
    endpoint = getattr(urls, "get_submission", None)
    if endpoint is None or not isinstance(getattr(endpoint, "value", None), str):
        raise RuntimeError("cannot resolve same-method submission: EvalAI get_submission URL is unavailable")
    detail = _get(make_request, endpoint.value.format(submission_id))
    uri = _item_image_uri(detail)
    if uri:
        return uri
    input_file = detail.get("input_file")
    if not isinstance(input_file, str) or not input_file:
        raise RuntimeError("same-method submission has neither image URI nor input_file")
    payload = _get(make_request, input_file)
    uri = _item_image_uri(payload)
    if not uri:
        raise RuntimeError("same-method submission input_file does not declare submitted_image_uri")
    return uri


def _already_registered(candidate: Mapping[str, Any], rows: list[dict[str, Any]], state: Mapping[str, object],
                        make_request: Any, urls: Any) -> dict[str, Any] | None:
    """Recovery is bound to the pushed immutable URI; method text is insufficient."""
    uri = state.get("submitted_image_uri")
    if not isinstance(uri, str) or not uri:
        raise RuntimeError("cannot compare existing submissions without the pushed immutable image URI")
    for item in rows:
        if str(item.get("method_name") or "") != str(candidate["method_name"]):
            continue
        item_uri = _item_image_uri(item) or _resolve_submission_uri(make_request, urls, item)
        if item_uri == uri:
            return item
        raise RuntimeError("same method name exists with a different image URI; refuse duplicate POST")
    return None

def _register(candidate: dict[str, Any], state: dict[str, object], runtime: tuple[Any, Any, Any, str, str]) -> dict[str, object]:
    if state.get("registered"):
        raise RuntimeError(f"submission {state.get('submission_id')} is already registered")
    _, _, evalai_runtime, _, _ = runtime
    urls, make_request = evalai_runtime
    rows = list_submissions(make_request, urls)
    existing = _already_registered(candidate, rows, state, make_request, urls)
    if existing is not None:
        state.update({"registered": True, "submission_id": existing["id"], "recovered_from_list": True})
        _save_state(Path(candidate["state_path"]), state)
        return state
    phase = _get(make_request, urls.phase_details_using_slug.value.format(PHASE_SLUG))
    if phase.get("id") != PHASE_ID or phase.get("challenge") != CHALLENGE_ID:
        raise RuntimeError("EvalAI phase identity changed before registration")
    if not phase.get("is_active") or phase.get("is_submission_paused"):
        raise RuntimeError("EvalAI phase is not accepting submissions")
    _require_quota(_quota(phase, rows))
    _remaining_positive(_remaining(make_request))
    metadata = {
        "is_public": json.dumps(False),
        "method_name": candidate["method_name"],
        "method_description": candidate["method_description"],
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="m1_activity_only_evalai_") as tmp:
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
        rows = list_submissions(make_request, urls)
        recovered = _already_registered(candidate, rows, state, make_request, urls)
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
    if candidate.get("register") is True and not args.execute:
        raise RuntimeError("manifest already marked registered; refuse silent re-preflight as execute")
    client, image, phase, limits, runtime = preflight(candidate)
    report = {
        "mode": "execute" if args.execute else "read_only_preflight",
        "arm": candidate["arm"],
        "image_id": image.id,
        "payload_sha256": candidate["payload_sha256"],
        "phase": phase["id"],
        "quota": limits,
        "state_path": candidate["state_path"],
        "private": True,
        "tta": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.confirm_image_id != candidate["image_id"]:
        raise RuntimeError("execute requires the complete immutable image ID")
    if args.confirm_payload_sha256 != candidate["payload_sha256"]:
        raise RuntimeError("execute requires the complete immutable payload SHA-256")
    path = Path(candidate["state_path"])
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("registered"):
            raise RuntimeError(f"submission {state.get('submission_id')} already registered")
        for key in ("arm", "image_tag", "image_id", "payload_sha256"):
            if state.get(key) != candidate.get(key):
                raise RuntimeError(f"resume immutable binding mismatch for {key}")
        if state.get("pushed") is True and not state.get("submitted_image_uri"):
            raise RuntimeError("pushed resume state lacks immutable submitted image URI")
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
