"""Guarded remote push for all-source B3 / B3S-rSyn3 M1 images.

Default is a local plan. Authenticated preflight talks to the same challenge
as the successful M2 cached-identity submission 581644. A private push needs
``--execute`` plus the complete image ID and payload SHA-256.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import plan


class RemotePushError(RuntimeError):
    """Fail closed for the remote cached-identity push."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RemotePushError(message)


CELL_DIR = Path(__file__).resolve().parent
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

ARM_META = {
    "b3": {
        "method_label": "M1 all-source B3 cached identity chronological M10",
        "method_name": "M1 all-source B3 cached identity M10 seed42",
        "method_description": (
            "Frozen M1 B3 student trained on all four held-in source sessions "
            "from the official original SPINT epoch-19 teacher. Per-session "
            "identities are computed offline from chronological first-10 public "
            "calibration trials. The runtime serves cached identities only and "
            "performs no query-label read, online calibration, optimizer step, "
            "or backpropagation."
        ),
    },
    "b3s_rsyn3": {
        "method_label": "M1 all-source B3S EMG-rSyn3 cached identity chronological M10",
        "method_name": "M1 all-source B3S EMG-rSyn3 cached identity M10 seed42",
        "method_description": (
            "Frozen M1 B3S student with an EMG-rSyn3 carrier (ReLU, rank-3 NNMF, "
            "unit ridge) trained on all four held-in source sessions from the "
            "official original SPINT epoch-19 teacher. Per-session identities "
            "are computed offline from chronological first-10 public calibration "
            "trials and the frozen all-source rSyn3 dictionary. The runtime "
            "serves cached identities only and performs no query-label read, "
            "online calibration, optimizer step, NMF refit, or backpropagation."
        ),
    },
    "b3s_rsyn3_freeze_top4": {
        "method_label": "M1 freeze rSyn3 encoder M10 carrier D-opt tgt_loc k4",
        "method_name": "M1 freeze rSyn3 M10 neural D-opt-k4 carrier seed42",
        "method_description": (
            "Wave-2 freeze-decoder student: official original SPINT epoch-19 "
            "decoder is frozen; only the B3S encoder is trained with EMG-rSyn3. "
            "Cached identities use chronological M10 neural calibration. The "
            "rSyn3 unit ridge uses greedy D-optimal tgt_loc selection of 4 of "
            "those 10 calib trials. NMF stays source-frozen. No query-label "
            "read, online calibration, optimizer step, or backpropagation."
        ),
    },
    "b3s_rsyn3_acyc_top4": {
        "method_label": "M1 acyc rSyn3 encoder M10 carrier D-opt tgt_loc k4",
        "method_name": "M1 acyc rSyn3 M10 neural D-opt-k4 carrier seed42",
        "method_description": (
            "Wave-2 joint-decoder student trained with a (10,5,2) activity "
            "prefix cycle; evaluation identities stay chronological M10 neural. "
            "The rSyn3 unit ridge uses greedy D-optimal tgt_loc selection of 4 "
            "of those 10 calib trials. NMF stays source-frozen. No query-label "
            "read, online calibration, optimizer step, or backpropagation."
        ),
    },
}


def repo_root() -> Path:
    return CELL_DIR.parents[2]


def export_dir(root: Path, arm: str) -> Path:
    return Path(root) / plan.package_root_relative(arm)


def push_dir(root: Path, arm: str) -> Path:
    return Path(root) / plan.remote_push_root_relative(arm)


def state_path(root: Path, arm: str) -> Path:
    return push_dir(root, arm) / "push_state.json"


def image_receipt_path(root: Path, arm: str) -> Path:
    return push_dir(root, arm) / "image_receipt.json"


def image_tag_for(arm: str, payload_sha256: str) -> str:
    prefixes = {
        "b3": "spint-b3-m1",
        "b3s_rsyn3": "spint-b3s-rsyn3-m1",
        "b3s_rsyn3_freeze_top4": "spint-b3s-rsyn3-freeze-top4-m1",
        "b3s_rsyn3_acyc_top4": "spint-b3s-rsyn3-acyc-top4-m1",
    }
    _require(arm in prefixes, f"no image tag prefix for {arm}")
    return f"{prefixes[arm]}:allsource-s42-{payload_sha256[:8]}"


def load_export_receipt(root: Path, arm: str) -> dict[str, Any]:
    _require(arm in plan.REMOTE_ARMS, f"remote push is not defined for {arm}")
    terminal = export_dir(root, arm) / "terminal.json"
    receipt = export_dir(root, arm) / "export_receipt.json"
    payload = export_dir(root, arm) / "decoder.pt"
    _require(terminal.is_file() and receipt.is_file() and payload.is_file(), f"export missing for {arm}")
    body = json.loads(terminal.read_text(encoding="utf-8"))
    _require(body.get("status") == "COMPLETE" and body.get("arm") == arm, f"export {arm} is not COMPLETE")
    exported = json.loads(receipt.read_text(encoding="utf-8"))
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    _require(exported.get("payload_sha256") == digest, "export payload sha drift")
    _require(exported.get("arm") == arm, "export receipt arm drift")
    _require(int(exported.get("session_count", 0)) == 7, "export session coverage drift")
    _require(exported.get("max_direct_vs_cached_identity_abs") == 0.0, "cached identity not exact")
    return exported


def candidate_for(arm: str, root: Path | None = None) -> dict[str, Any]:
    root = Path(root) if root is not None else repo_root()
    meta = ARM_META[arm]
    exported = load_export_receipt(root, arm)
    payload_sha = str(exported["payload_sha256"])
    image_tag = image_tag_for(arm, payload_sha)
    image_id = None
    receipt_path = image_receipt_path(root, arm)
    if receipt_path.is_file():
        built = json.loads(receipt_path.read_text(encoding="utf-8"))
        _require(built.get("arm") == arm, "image receipt arm drift")
        _require(built.get("payload_sha256") == payload_sha, "image receipt payload drift")
        image_tag = str(built["image_tag"])
        image_id = str(built["image_id"])
    return {
        "arm": arm,
        "image_tag": image_tag,
        "image_id": image_id,
        "payload_sha256": payload_sha,
        "checkpoint_sha256": exported["checkpoint_sha256"],
        "teacher_sha256": plan.TEACHER_SHA256,
        "method_label": meta["method_label"],
        "method_name": meta["method_name"],
        "method_description": meta["method_description"],
        "payload_path": str(export_dir(root, arm) / "decoder.pt"),
    }


def plan_report(arm: str, root: Path | None = None) -> dict[str, object]:
    _require(arm in plan.REMOTE_ARMS, f"remote push is not defined for {arm}")
    candidate = candidate_for(arm, root)
    return {
        "mode": "plan_only",
        "arm": arm,
        "challenge_id": plan.REMOTE_CHALLENGE_ID,
        "phase_id": plan.REMOTE_PHASE_ID,
        "phase_slug": plan.REMOTE_PHASE_SLUG,
        "team_id": plan.REMOTE_TEAM_ID,
        "private": True,
        "image_tag": candidate["image_tag"],
        "image_id": candidate["image_id"],
        "payload_sha256": candidate["payload_sha256"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "teacher_sha256": candidate["teacher_sha256"],
        "method_name": candidate["method_name"],
        "method_description": candidate["method_description"],
        "submission_attributes": SUBMISSION_ATTRIBUTES,
        "required_runtime": {
            "task": "m1",
            "phase": "test",
            "batch_size": plan.REMOTE_BATCH_SIZE,
            "cached_identity_only": True,
            "online_calibration": False,
            "query_labels_used": False,
            "backpropagation": False,
        },
        "state_path": str(state_path(root or repo_root(), arm)),
        "predecessor_submission": 581644,
        "next_gate": "docker build, then authenticated read-only preflight",
        "formal_benchmark_verdict": False,
    }


def build_image(root: Path, arm: str) -> dict[str, Any]:
    _require(arm in plan.REMOTE_ARMS, f"remote push is not defined for {arm}")
    candidate = candidate_for(root=root, arm=arm)
    payload = Path(candidate["payload_path"])
    image_tag = candidate["image_tag"]
    dest = push_dir(root, arm)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"m1_b3_remote_{arm}_") as tmp:
        stage = Path(tmp)
        (stage / "artifacts").mkdir()
        shutil.copyfile(payload, stage / "artifacts" / "decoder.pkl")
        shutil.copyfile(CELL_DIR / "runtime.py", stage / "runtime.py")
        shutil.copyfile(CELL_DIR / "decode_remote.py", stage / "decode.py")
        shutil.copyfile(CELL_DIR / "Dockerfile", stage / "Dockerfile")
        command = [
            "docker", "build",
            "--build-arg", f"PAYLOAD_SHA256={candidate['payload_sha256']}",
            "--build-arg", f"CHECKPOINT_SHA256={candidate['checkpoint_sha256']}",
            "--build-arg", f"ARM={arm}",
            "--build-arg", f"METHOD_LABEL={candidate['method_label']}",
            "--build-arg", f"TEACHER_SHA256={plan.TEACHER_SHA256}",
            "--build-arg", f"BASE_IMAGE_ID={plan.REMOTE_BASE_IMAGE_ID}",
            "-t", image_tag,
            "-f", str(stage / "Dockerfile"),
            str(stage),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RemotePushError(completed.stderr[-4000:] or completed.stdout[-4000:])
    import docker

    client = docker.from_env()
    image = client.images.get(image_tag)
    labels = image.attrs.get("Config", {}).get("Labels", {}) or {}
    _require(labels.get("ai.eval.payload.sha256") == candidate["payload_sha256"], "payload label drift")
    _require(labels.get("ai.eval.checkpoint.sha256") == candidate["checkpoint_sha256"], "checkpoint label drift")
    _require(labels.get("ai.eval.task") == "m1", "image is not M1")
    _require(labels.get("ai.eval.method") == candidate["method_label"], "method label drift")
    receipt = {
        "schema": "m1_b3_allsource_image_receipt_v1",
        "arm": arm,
        "image_tag": image_tag,
        "image_id": image.id,
        "payload_sha256": candidate["payload_sha256"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "teacher_sha256": plan.TEACHER_SHA256,
        "method_label": candidate["method_label"],
        "base_image": plan.REMOTE_BASE_IMAGE,
        "base_image_id": plan.REMOTE_BASE_IMAGE_ID,
        "size_bytes": int(image.attrs.get("Size", 0)),
        "formal_benchmark_verdict": False,
    }
    pending = image_receipt_path(root, arm).with_suffix(".json.tmp")
    pending.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(image_receipt_path(root, arm))
    return receipt


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
        raise RemotePushError(f"GET returned non-object for {path}")
    return value


def _validate_local_image(docker_module: Any, candidate: Mapping[str, Any]) -> tuple[Any, Any]:
    _require(isinstance(candidate.get("image_id"), str) and str(candidate["image_id"]).startswith("sha256:"),
             "build the image before preflight")
    client = docker_module.from_env()
    image = client.images.get(candidate["image_tag"])
    if image.id != candidate["image_id"]:
        raise RemotePushError(f"image ID guard failed: {image.id} != {candidate['image_id']}")
    labels = image.attrs.get("Config", {}).get("Labels", {}) or {}
    expected = {
        "ai.eval.payload.sha256": candidate["payload_sha256"],
        "ai.eval.checkpoint.sha256": candidate["checkpoint_sha256"],
        "ai.eval.teacher.sha256": candidate["teacher_sha256"],
        "ai.eval.method": candidate["method_label"],
        "ai.eval.task": "m1",
    }
    for key, value in expected.items():
        if labels.get(key) != value:
            raise RemotePushError(f"image label {key} drift: {labels.get(key)!r}")
    return client, image


def preflight(arm: str, root: Path | None = None) -> tuple[dict[str, Any], Any, Any, dict[str, Any], dict[str, int], tuple[Any, Any, Any, str, str]]:
    root = Path(root) if root is not None else repo_root()
    candidate = candidate_for(arm, root)
    runtime = _runtime()
    docker_module, _, evalai_runtime, _, _ = runtime
    urls, make_request = evalai_runtime
    client, image = _validate_local_image(docker_module, candidate)
    phase = _get(make_request, urls.phase_details_using_slug.value.format(plan.REMOTE_PHASE_SLUG))
    if phase.get("id") != plan.REMOTE_PHASE_ID or phase.get("challenge") != plan.REMOTE_CHALLENGE_ID:
        raise RemotePushError("phase identity changed")
    if not phase.get("is_active") or phase.get("is_submission_paused"):
        raise RemotePushError("phase is not accepting submissions")
    challenge = _get(make_request, urls.challenge_details.value.format(plan.REMOTE_CHALLENGE_ID))
    image_size = int(image.attrs.get("Size", image.attrs.get("VirtualSize", 0)))
    if image_size > int(challenge["max_docker_image_size"]):
        raise RemotePushError("image exceeds size limit")
    submissions = _get(make_request, urls.my_submissions.value.format(plan.REMOTE_CHALLENGE_ID, plan.REMOTE_PHASE_ID))
    results = submissions.get("results", [])
    if not isinstance(results, list):
        raise RemotePushError("submission list schema drift")
    now = datetime.now(timezone.utc)
    today = month = active = 0
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
            raise RemotePushError(f"{label} submission quota is exhausted")
    return candidate, client, image, phase, limits, runtime


def _save_state(root: Path, arm: str, state: dict[str, object]) -> None:
    path = state_path(root, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def _push_image(root: Path, candidate: Mapping[str, Any], client: Any, image: Any, runtime: tuple[Any, Any, Any, str, str]) -> dict[str, object]:
    _, boto3_module, evalai_runtime, aws_region, environment = runtime
    urls, make_request = evalai_runtime
    response = _get(make_request, urls.get_aws_credentials.value.format(plan.REMOTE_PHASE_ID))["success"]
    if not isinstance(response, dict):
        raise RemotePushError("credential response schema drift")
    federated = response["federated_user"]
    repository_uri = response["docker_repository_uri"]
    credentials = federated["Credentials"]
    account_id = federated["FederatedUser"]["FederatedUserId"].split(":")[0]
    if environment != "PRODUCTION":
        raise RemotePushError("only the production host is permitted")
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
            raise RemotePushError(f"Docker push failed: {event}")
    repository_name = repository_uri.split("/", 1)[1]
    manifest = ecr.describe_images(
        repositoryName=repository_name, imageIds=[{"imageTag": tag}]
    )["imageDetails"][0]["imageDigest"]
    if manifest != candidate["image_id"]:
        raise RemotePushError(f"uploaded digest mismatch: {manifest}")
    state = {
        "schema_version": "m1_b3_allsource_push_state_v1",
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
    _save_state(root, candidate["arm"], state)
    return state


def _register(root: Path, candidate: Mapping[str, Any], state: dict[str, object], runtime: tuple[Any, Any, Any, str, str]) -> dict[str, object]:
    if state.get("registered"):
        raise RemotePushError(f"submission {state.get('submission_id')} is already registered")
    if (
        state.get("arm") != candidate["arm"]
        or state.get("image_id") != candidate["image_id"]
        or state.get("payload_sha256") != candidate["payload_sha256"]
        or state.get("pushed") is not True
    ):
        raise RemotePushError("push state does not match the candidate")
    _, _, evalai_runtime, _, _ = runtime
    urls, make_request = evalai_runtime
    metadata = {
        "is_public": json.dumps(False),
        "method_name": candidate["method_name"],
        "method_description": candidate["method_description"],
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
    }
    with tempfile.TemporaryDirectory(prefix="m1_b3_remote_") as tmp:
        submission_file = Path(tmp) / "submission.json"
        submission_file.write_text(
            json.dumps({"submitted_image_uri": state["submitted_image_uri"]}),
            encoding="utf-8",
        )
        response = make_request(
            urls.make_submission.value.format(plan.REMOTE_CHALLENGE_ID, plan.REMOTE_PHASE_ID),
            "POST",
            files=str(submission_file),
            data=metadata,
        )
    state.update({"registered": True, "submission_id": response["id"], "server_response": response})
    _save_state(root, candidate["arm"], state)
    return state


def execute_push(
    root: Path,
    arm: str,
    *,
    confirm_image_id: str,
    confirm_payload_sha256: str,
) -> dict[str, object]:
    candidate, client, image, phase, limits, runtime = preflight(arm, root)
    del phase, limits
    if confirm_image_id != candidate["image_id"]:
        raise RemotePushError("execute mode requires the complete immutable image ID")
    if confirm_payload_sha256 != candidate["payload_sha256"]:
        raise RemotePushError("execute mode requires the complete immutable payload SHA-256")
    path = state_path(root, arm)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("registered"):
            raise RemotePushError(f"submission {state.get('submission_id')} already registered")
    else:
        state = _push_image(root, candidate, client, image, runtime)
    return _register(root, candidate, state, runtime)
