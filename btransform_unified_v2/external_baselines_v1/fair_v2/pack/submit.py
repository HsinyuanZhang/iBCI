#!/usr/bin/env python3
"""Guarded later-submit helper for fair_v2 packages.

Does nothing unless --execute is passed with the immutable image/payload IDs.
Default token/team is HKU-ECE / 41975. A same-group team may be selected with
--token-file plus --team-id/--team-name; the default ~/.evalai/token.json is
not overwritten.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, "/home/xinyuan/.local/lib/python3.10/site-packages")
for _k in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_k, "")
    if "eval.ai" not in _val:
        os.environ[_k] = f"{_val},eval.ai".strip(",")

CHALLENGE_ID = 2319
PHASE_ID = 4599
PHASE_SLUG = "few-shot-test-2319"
DEFAULT_TEAM = "HKU-ECE"
DEFAULT_TEAM_ID = 41975
ALLOWED_TEAMS = {
    41975: "HKU-ECE",
    41817: "falcon_m",
    42279: "sustechhku",
}

SUBMISSION_ATTRIBUTES = [
    {"name": "IsHeldOutZeroShot", "type": "boolean", "description": "Is held out prediction Zero Shot?", "required": True, "value": False},
    {"name": "IsTestTimeAdaptive", "type": "boolean", "description": "Test Time Adaptive?", "required": True, "value": False},
    {"name": "IsPretrained", "type": "boolean", "description": "Pretrained outside FALCON?", "required": True, "value": False},
]


def _runtime():
    import boto3
    import docker
    from evalai.utils.config import AWS_REGION, ENVIRONMENT
    from evalai.utils.requests import make_request
    from evalai.utils.urls import URLS

    return docker, boto3, (URLS, make_request), AWS_REGION, ENVIRONMENT


def _get(make_request, path: str) -> dict:
    value = make_request(path, "GET")
    if not isinstance(value, dict):
        raise RuntimeError(f"EvalAI GET returned non-object for {path}")
    return value


def bind_token_file(token_file: Path) -> Path:
    path = token_file.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"EvalAI token file is missing: {path}")
    import evalai.utils.auth as auth
    import evalai.utils.config as cfg

    cfg.AUTH_TOKEN_PATH = str(path)
    auth.AUTH_TOKEN_PATH = str(path)
    return path


def assert_team(make_request, urls, team_id: int, team_name: str) -> dict:
    if team_id not in ALLOWED_TEAMS or ALLOWED_TEAMS[team_id] != team_name:
        raise RuntimeError(f"refusing unknown team {team_name} / {team_id}")
    me = _get(make_request, urls.user_details.value if hasattr(urls, "user_details") else "/api/auth/user/")
    teams = _get(make_request, urls.participant_teams.value)
    rows = teams.get("results") or teams if isinstance(teams, dict) else teams
    if not isinstance(rows, list):
        rows = list((teams or {}).get("results") or [])
    matched = [row for row in rows if int(row.get("id", -1)) == team_id]
    if not matched:
        raise RuntimeError(f"active EvalAI token is not {team_name} / {team_id}")
    team = matched[0]
    if team_name not in str(team.get("team_name") or team.get("team") or ""):
        raise RuntimeError(f"team {team_id} name is not {team_name}")
    return {"team_id": team_id, "team_name": team.get("team_name"), "user": me.get("username")}


def list_submissions(make_request, urls) -> list[dict[str, Any]]:
    page = _get(make_request, urls.my_submissions.value.format(CHALLENGE_ID, PHASE_ID))
    rows = list(page.get("results") or [])
    nxt = page.get("next")
    while nxt:
        parsed = urlparse(str(nxt))
        page = _get(make_request, parsed.path + ("?" + parsed.query if parsed.query else ""))
        rows.extend(page.get("results") or [])
        nxt = page.get("next")
    return rows


def load_candidate(manifest: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    required = ("image_tag", "image_id", "payload_sha256", "method_name", "method_description", "task")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"candidate manifest missing {missing}")
    if not payload.get("submission_eligible"):
        raise RuntimeError("candidate is not submission_eligible")
    return payload


def preflight(candidate: dict[str, Any], team_id: int, team_name: str):
    runtime = _runtime()
    docker_module, _, evalai_runtime, _, _ = runtime
    urls, make_request = evalai_runtime
    team = assert_team(make_request, urls, team_id, team_name)
    client = docker_module.from_env()
    image = client.images.get(candidate["image_tag"])
    if image.id != candidate["image_id"]:
        raise RuntimeError(f"image ID guard failed: {image.id} != {candidate['image_id']}")
    labels = image.attrs.get("Config", {}).get("Labels", {}) or {}
    if labels.get("ai.eval.payload.sha256") != candidate["payload_sha256"]:
        raise RuntimeError("image payload SHA label drift")
    if labels.get("ai.eval.task") != candidate["task"]:
        raise RuntimeError("image task label drift")
    phase = _get(make_request, urls.phase_details_using_slug.value.format(PHASE_SLUG))
    if phase.get("id") != PHASE_ID or phase.get("challenge") != CHALLENGE_ID:
        raise RuntimeError("EvalAI phase identity changed")
    rows = list_submissions(make_request, urls)
    now = datetime.now(timezone.utc)
    today = sum(1 for item in rows if datetime.fromisoformat(str(item["submitted_at"]).replace("Z", "+00:00")).date() == now.date())
    active_ids = [
        {"id": item.get("id"), "status": item.get("status"), "method": item.get("method_name")}
        for item in rows
        if item.get("status") in {"submitted", "queued", "running"}
    ]
    limits = {
        "today": today,
        "active": len(active_ids),
        "active_ids": active_ids,
        "max_per_day": int(phase.get("max_submissions_per_day") or 6),
        "max_concurrent": int(phase.get("max_concurrent_submissions_allowed") or 3),
        "team": team,
    }
    if today >= limits["max_per_day"]:
        raise RuntimeError("daily submission quota is exhausted")
    if limits["active"] >= 3:
        raise RuntimeError(f"concurrent slots full: active={limits['active']}")
    return client, image, phase, limits, runtime


def _save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def _push_image(candidate, client, image, runtime):
    _, boto3_module, evalai_runtime, aws_region, environment = runtime
    urls, make_request = evalai_runtime
    response = _get(make_request, urls.get_aws_credentials.value.format(PHASE_ID))["success"]
    federated = response["federated_user"]
    repository_uri = response["docker_repository_uri"]
    credentials = federated["Credentials"]
    account_id = federated["FederatedUser"]["FederatedUserId"].split(":")[0]
    if environment != "PRODUCTION":
        raise RuntimeError("only the production EvalAI host is permitted")
    team_id = int(candidate["submit_team_id"])
    if f"team-{team_id}" not in repository_uri:
        raise RuntimeError(f"ECR repository is not team {team_id}: {repository_uri}")
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
    manifest = ecr.describe_images(repositoryName=repository_name, imageIds=[{"imageTag": tag}])["imageDetails"][0]["imageDigest"]
    if manifest != candidate["image_id"]:
        raise RuntimeError(f"uploaded digest mismatch: {manifest}")
    state = {
        "schema_version": "fair_v2_evalai_push_state",
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
    _save_state(Path(candidate.get("state_path") or Path(candidate.get("_manifest_dir", ".")) / "evalai_push_state.json"), state)
    return state


def _register(candidate, state, runtime):
    _, _, evalai_runtime, _, _ = runtime
    urls, make_request = evalai_runtime
    rows = list_submissions(make_request, urls)
    active = sum(1 for item in rows if item.get("status") in {"submitted", "queued", "running"})
    if active >= 3:
        raise RuntimeError(f"refusing POST: active={active} >= 3")
    metadata = {
        "is_public": json.dumps(False),
        "method_name": candidate["method_name"],
        "method_description": candidate["method_description"],
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
    }
    with tempfile.TemporaryDirectory(prefix="fair_v2_evalai_") as tmp:
        submission_file = Path(tmp) / "submission.json"
        submission_file.write_text(json.dumps({"submitted_image_uri": state["submitted_image_uri"]}), encoding="utf-8")
        response = make_request(urls.make_submission.value.format(CHALLENGE_ID, PHASE_ID), "POST", files=str(submission_file), data=metadata)
    state.update({"registered": True, "submission_id": response["id"], "server_response": response})
    _save_state(Path(candidate["state_path"]), state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    parser.add_argument("--confirm-payload-sha256", default="")
    parser.add_argument("--token-file", default=str(Path.home() / ".evalai" / "token.json"))
    parser.add_argument("--team-id", type=int, default=DEFAULT_TEAM_ID)
    parser.add_argument("--team-name", default=DEFAULT_TEAM)
    args = parser.parse_args()
    bind_token_file(Path(args.token_file))
    candidate = load_candidate(Path(args.manifest))
    if args.team_id != DEFAULT_TEAM_ID:
        suffix = ALLOWED_TEAMS.get(args.team_id, str(args.team_id))
        candidate["state_path"] = str(Path(args.manifest).parent / f"evalai_push_state_{suffix}.json")
    else:
        candidate["state_path"] = candidate.get("state_path") or str(Path(args.manifest).parent / "evalai_push_state.json")
    candidate["submit_team_id"] = args.team_id
    candidate["submit_team_name"] = args.team_name
    client, image, phase, limits, runtime = preflight(candidate, args.team_id, args.team_name)
    report = {
        "mode": "execute" if args.execute else "read_only_preflight",
        "image_id": image.id,
        "payload_sha256": candidate["payload_sha256"],
        "phase": phase["id"],
        "quota": limits,
        "private": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.confirm_image_id != candidate["image_id"]:
        raise RuntimeError("execute requires the complete immutable image ID")
    if args.confirm_payload_sha256 != candidate["payload_sha256"]:
        raise RuntimeError("execute requires the complete immutable payload SHA-256")
    path = Path(candidate["state_path"])
    state = json.loads(path.read_text()) if path.exists() else None
    if state and state.get("registered"):
        raise RuntimeError(f"submission {state.get('submission_id')} already registered")
    if not state or state.get("pushed") is not True:
        state = _push_image(candidate, client, image, runtime)
    completed = _register(candidate, state, runtime)
    print(json.dumps({"submission_id": completed["submission_id"], "image_id": completed["image_id"]}, indent=2))


if __name__ == "__main__":
    main()
