"""Guarded EvalAI submission helper for frozen M1 candidates.

The legacy Original/T4/D4 candidates retain immutable in-code guards.  The
new all-source q=3 AFC4 Full/B4 entries are deliberately unready until an
independently reviewed candidate manifest binds their image, payload, teacher,
and student-checkpoint hashes.  ``--plan`` is a local dry-run and never opens
Docker, calls EvalAI, or mutates a push/submission receipt.
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
CANDIDATE_MANIFEST_SCHEMA = "evalai_m1_afc4_submission_candidates_v1"
CANDIDATE_MANIFEST_DEFAULT = (
    Path(__file__).resolve().parent / "artifacts" / "evalai_m1_afc4_candidates.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARM_ALIASES = {
    # Human-facing names used in the final experiment receipts.  The shorter
    # ``full``/``b4`` names remain the canonical manifest keys and Docker label
    # arms, so aliases cannot create a second mutable candidate definition.
    "m1_afc4_full_allsource": "full",
    "m1_afc4_b4_allsource": "b4",
}

CANDIDATES = {
    "original": {
        "image_tag": "spint-original-m1:e9-epoch019-052e9ea",
        "image_id": "sha256:f5af9eb29b7f86616d898261070193b1b0777db75567848c62d7f888ce3d76cd",
        "payload_sha256": "052e9eab7be2af8bacd5298e348d414cce60dad767d6a0880b58dd39b27279f6",
        "method_name": "Original SPINT M1 epoch19",
        "method_description": (
            "Frozen original SPINT M1 epoch-19 decoder using the chronological first 10 "
            "public calibration trials per session. No hidden query labels or runtime "
            "backpropagation are used."
        ),
    },
    "t4": {
        "image_tag": "spint-t4-m1:e9-f1s42-b5cc6d2-1f2e4f9",
        "image_id": "sha256:7897eb6adbb8451d9ea0f2fe5f07850baedff685d89ead3e7e23ee34d786cf13",
        "payload_sha256": "1f2e4f96ac160abcd56dc5a79190bb3063099d6efe1039826817745556093957",
        "method_name": "T4 cached identity M1 M10 fold1 seed42",
        "method_description": (
            "Frozen SPINT M1 decoder with per-session T4 identities computed offline from "
            "the chronological first 10 public calibration trials and their target directions. "
            "No hidden query labels, optimizer, or runtime backpropagation are used."
        ),
    },
    "d4": {
        "image_tag": "spint-d4-m1:e9-f1s42-28530a7-a45471b",
        "image_id": "sha256:16aba3807945ca309c3a1041c9fdba7724e8321fc327962ce0fd2beba46e04a8",
        "payload_sha256": "a45471b29a4b847f510a0a21695b2fccbe9615a8dbec285d3dffe3c4be2f6910",
        "method_name": "D4 cached identity M1 M10 fold1 seed42",
        "method_description": (
            "Frozen SPINT M1 decoder with per-session D4 categorical-profile identities "
            "computed offline from the chronological first 10 public calibration trials and "
            "their obj_id labels. No hidden query labels, optimizer, or runtime backpropagation "
            "are used."
        ),
    },
    # The exact image/payload/checkpoint IDs are intentionally supplied through
    # ``--manifest`` after the final GPU artifacts are reviewed.  Keeping the
    # arm names and method metadata here lets ``--plan`` audit the complete
    # challenge/phase/team/metadata chain before those artifacts exist, while
    # ``preflight`` and ``--execute`` fail closed until every hash is bound.
    "full": {
        "arm": "full",
        "image_tag": "spint-m1-afc4-full-all-source:dev12",
        "image_id": None,
        "payload_sha256": None,
        "checkpoint_sha256": None,
        "teacher_sha256": None,
        "method_name": "M1 q3 EMG-AFC4 Full all-source M10",
        "method_description": (
            "Frozen M1 q=3 EMG-AFC4 Full decoder trained on all four held-in source "
            "sessions from a shared fixed-20-epoch train-loss teacher and a fixed 12-epoch "
            "task-only student run. Per-session identities "
            "are computed offline from chronological M10 public calibration support; the "
            "official runtime uses cached identities only and performs no query-label read, "
            "online calibration, optimizer step, or backpropagation."
        ),
    },
    "b4": {
        "arm": "b4",
        "image_tag": "spint-m1-afc4-b4-all-source:dev12",
        "image_id": None,
        "payload_sha256": None,
        "checkpoint_sha256": None,
        "teacher_sha256": None,
        "method_name": "M1 q3 EMG-AFC4 B4 all-source M10",
        "method_description": (
            "Frozen M1 q=3 EMG-AFC4 B4 decoder trained on all four held-in source "
            "sessions from a shared fixed-20-epoch train-loss teacher and a fixed 12-epoch "
            "task-only student run, with the normalized "
            "q=3 coordinates masked and the baseline coordinate retained. Per-session "
            "identities are computed offline from chronological M10 public calibration "
            "support; the official runtime uses cached identities only and performs no "
            "query-label read, online calibration, optimizer step, or backpropagation."
        ),
    },
}

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


def canonical_arm(arm: str) -> str:
    return ARM_ALIASES.get(str(arm), str(arm))


def get(path: str) -> dict:
    return make_request(path, "GET")


def _candidate_sha(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"candidate {name} must be a lowercase 64-hex SHA-256 string")
    return value


def _validate_bound_candidate(arm: str, candidate: dict) -> dict:
    """Validate a manifest-bound Full/B4 candidate before Docker/API access."""
    if candidate.get("arm", arm) != arm:
        raise ValueError(f"candidate arm field mismatch: expected {arm!r}")
    image_tag = candidate.get("image_tag")
    if not isinstance(image_tag, str) or not image_tag or any(char.isspace() for char in image_tag):
        raise ValueError(f"candidate {arm} image_tag is missing or invalid")
    image_id = candidate.get("image_id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ValueError(f"candidate {arm} image_id must start with sha256:")
    _candidate_sha(image_id.split(":", 1)[1], name=f"{arm}.image_id")
    for key in ("payload_sha256", "checkpoint_sha256", "teacher_sha256"):
        _candidate_sha(candidate.get(key), name=f"{arm}.{key}")
    for key in ("method_name", "method_description"):
        if not isinstance(candidate.get(key), str) or not candidate[key].strip():
            raise ValueError(f"candidate {arm} {key} is missing")
    return candidate


def load_candidate_manifest(path: str | Path) -> dict[str, dict]:
    """Load and validate the post-checkpoint Full/B4 candidate binding.

    This file is intentionally separate from source code because image IDs and
    payload/checkpoint hashes do not exist until the final GPU/export/image
    steps finish.  It is never written by this helper.
    """
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Full/B4 candidate manifest is missing: {manifest_path}; "
            "run --plan now, then create it only after reviewing final receipts"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid candidate manifest JSON: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA:
        raise ValueError(f"candidate manifest schema mismatch: expected {CANDIDATE_MANIFEST_SCHEMA}")
    for key, expected in (
        ("challenge_id", CHALLENGE_ID),
        ("phase_id", PHASE_ID),
        ("team_id", TEAM_ID),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"candidate manifest {key} changed: expected {expected}, got {payload.get(key)}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != {"full", "b4"}:
        raise ValueError("candidate manifest must contain exactly full and b4 entries")
    bound: dict[str, dict] = {}
    for arm in ("full", "b4"):
        entry = candidates[arm]
        if not isinstance(entry, dict):
            raise ValueError(f"candidate manifest {arm} entry must be an object")
        merged = dict(CANDIDATES[arm])
        merged.update(entry)
        _validate_bound_candidate(arm, merged)
        bound[arm] = merged
    if bound["full"]["image_tag"] == bound["b4"]["image_tag"]:
        raise ValueError("Full and B4 must use independent image tags")
    if bound["full"]["image_id"] == bound["b4"]["image_id"]:
        raise ValueError("Full and B4 must use independent immutable image IDs")
    if bound["full"]["payload_sha256"] == bound["b4"]["payload_sha256"]:
        raise ValueError("Full and B4 payload hashes unexpectedly match")
    return bound


def _candidate_for_plan(arm: str, manifest_path: str | Path | None = None) -> tuple[dict, str | None, bool]:
    arm = canonical_arm(arm)
    if arm not in CANDIDATES:
        raise ValueError(f"unknown candidate arm {arm!r}; choices: {sorted(CANDIDATES)}")
    if arm in {"full", "b4"} and manifest_path is not None:
        bound = load_candidate_manifest(manifest_path)
        return dict(bound[arm]), str(Path(manifest_path).resolve()), True
    candidate = dict(CANDIDATES[arm])
    ready = arm not in {"full", "b4"}
    return candidate, None, ready


def preflight(arm: str, *, manifest_path: str | Path | None = None):
    arm = canonical_arm(arm)
    candidate, resolved_manifest, ready = _candidate_for_plan(arm, manifest_path)
    if not ready:
        raise RuntimeError(
            f"{arm} is not bound to final image/payload/checkpoint/teacher hashes; "
            f"supply --manifest {CANDIDATE_MANIFEST_DEFAULT} after final export"
        )
    client = docker.from_env()
    image = client.images.get(candidate["image_tag"])
    if image.id != candidate["image_id"]:
        raise RuntimeError(f"{arm} image ID guard failed")
    labels = image.attrs.get("Config", {}).get("Labels", {}) or {}
    if labels.get("ai.eval.payload.sha256") != candidate["payload_sha256"]:
        raise RuntimeError(f"{arm} payload label guard failed")
    if labels.get("ai.eval.task") != "m1":
        raise RuntimeError(f"{arm} image is not M1")
    if arm in {"full", "b4"} and arm not in str(labels.get("ai.eval.method", "")).lower():
        raise RuntimeError(f"{arm} image method label does not bind the requested arm")
    if candidate.get("checkpoint_sha256") is not None and labels.get("ai.eval.checkpoint.sha256") != candidate["checkpoint_sha256"]:
        raise RuntimeError(f"{arm} checkpoint label guard failed")
    if candidate.get("teacher_sha256") is not None and labels.get("ai.eval.teacher.sha256") != candidate["teacher_sha256"]:
        raise RuntimeError(f"{arm} teacher label guard failed")

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
    limits = {
        "today": sum(datetime.fromisoformat(x["submitted_at"].replace("Z", "+00:00")).date() == now.date() for x in results),
        "month": sum((lambda d: (d.year, d.month) == (now.year, now.month))(datetime.fromisoformat(x["submitted_at"].replace("Z", "+00:00"))) for x in results),
        "total": int(submissions.get("count", len(results))),
        "active": sum(x.get("status") in {"submitted", "queued", "running"} for x in results),
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


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def push_image(arm: str, candidate: dict, client, image, state_path: Path) -> dict:
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
    if manifest != candidate["image_id"]:
        raise RuntimeError(f"Uploaded digest mismatch for {arm}")
    state = {
        "schema_version": "evalai_m1_threeway_push_state_v1",
        "arm": arm,
        "image_tag": candidate["image_tag"],
        "image_id": candidate["image_id"],
        "repository_uri": repository_uri,
        "uuid_tag": tag,
        "submitted_image_uri": submitted_image_uri,
        "uploaded_manifest_digest": manifest,
        "pushed": True,
        "registered": False,
    }
    save_state(state_path, state)
    return state


def register(candidate: dict, state_path: Path, state: dict) -> dict:
    if state.get("registered"):
        raise RuntimeError(f"Submission {state.get('submission_id')} already registered")
    payload = {"submitted_image_uri": state["submitted_image_uri"]}
    metadata = {
        "is_public": json.dumps(False),
        "method_name": candidate["method_name"],
        "method_description": candidate["method_description"],
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
    }
    with tempfile.TemporaryDirectory(prefix="evalai_m1_threeway_") as tmp:
        submission_file = Path(tmp) / "submission.json"
        submission_file.write_text(json.dumps(payload))
        response = make_request(
            URLS.make_submission.value.format(CHALLENGE_ID, PHASE_ID),
            "POST",
            files=str(submission_file),
            data=metadata,
        )
    state.update({"registered": True, "submission_id": response["id"], "server_response": response})
    save_state(state_path, state)
    return state


def plan_report(arm: str, manifest_path: str | Path | None = None) -> dict:
    """Return a no-side-effect candidate/submission plan for CPU dry-runs."""
    requested_arm = str(arm)
    arm = canonical_arm(arm)
    candidate, resolved_manifest, ready = _candidate_for_plan(arm, manifest_path)
    return {
        "mode": "plan_only",
        "ready_for_docker_preflight": ready,
        "candidate_manifest": resolved_manifest,
        "arm": arm,
        "requested_arm": requested_arm,
        "challenge_id": CHALLENGE_ID,
        "phase_id": PHASE_ID,
        "phase_slug": PHASE_SLUG,
        "team_id": TEAM_ID,
        "image_tag": candidate.get("image_tag"),
        "image_id": candidate.get("image_id"),
        "payload_sha256": candidate.get("payload_sha256"),
        "checkpoint_sha256": candidate.get("checkpoint_sha256"),
        "teacher_sha256": candidate.get("teacher_sha256"),
        "method_name": candidate.get("method_name"),
        "method_description": candidate.get("method_description"),
        "submission_attributes": SUBMISSION_ATTRIBUTES,
        "submission_endpoint": URLS.make_submission.value.format(CHALLENGE_ID, PHASE_ID),
        "required_runtime": {
            "task": "m1",
            "phase": "test",
            "batch_size": 4,
            "online_calibration": False,
            "query_labels_used": False,
            "backpropagation": False,
        },
        "next_gate": (
            "create and review the manifest, then run ordinary read-only preflight"
            if not ready
            else "run ordinary read-only preflight; --execute remains separately gated"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(set(CANDIDATES) | set(ARM_ALIASES)), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "reviewed Full/B4 candidate manifest; required for ordinary Full/B4 "
            f"preflight/execute (expected path: {CANDIDATE_MANIFEST_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print a local candidate/submission plan without Docker or EvalAI/API access",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    args = parser.parse_args()
    if args.plan and args.execute:
        raise RuntimeError("--plan and --execute are mutually exclusive")
    if args.plan:
        print(json.dumps(plan_report(args.arm, args.manifest), indent=2, sort_keys=True), flush=True)
        return
    requested_arm = args.arm
    canonical = canonical_arm(args.arm)
    candidate, client, image, phase, challenge, limits = preflight(
        canonical, manifest_path=args.manifest
    )
    report = {
        "mode": "execute" if args.execute else "read_only_preflight",
        "arm": canonical,
        "requested_arm": requested_arm,
        "image_tag": candidate["image_tag"],
        "image_id": image.id,
        "image_size": image.attrs.get("Size"),
        "max_image_size": challenge["max_docker_image_size"],
        "phase_id": phase["id"],
        "phase_slug": phase["slug"],
        "phase_active": phase["is_active"],
        "team_id": TEAM_ID,
        "private": True,
        "candidate_manifest": None if args.manifest is None else str(args.manifest.resolve()),
        "method_name": candidate["method_name"],
        "checkpoint_sha256": candidate.get("checkpoint_sha256"),
        "teacher_sha256": candidate.get("teacher_sha256"),
        "quota": limits,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.confirm_image_id != candidate["image_id"]:
        raise RuntimeError("Execute mode requires the complete immutable image ID")
    state_path = Path(__file__).resolve().parent / "artifacts" / f"evalai_push_state_{canonical}.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("registered"):
            raise RuntimeError(f"Submission {state.get('submission_id')} already registered")
    else:
        state = push_image(canonical, candidate, client, image, state_path)
    completed = register(candidate, state_path, state)
    print(json.dumps({
        "arm": canonical,
        "submission_id": completed["submission_id"],
        "image_id": completed["image_id"],
        "uploaded_manifest_digest": completed["uploaded_manifest_digest"],
        "uuid_tag": completed["uuid_tag"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
