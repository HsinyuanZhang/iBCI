#!/usr/bin/env python3
"""Guarded EvalAI push/register for the five fixed final-ablation controls.

Use the SPINT conda Python with the user EvalAI site-packages on sys.path.
Default preflight performs only local Docker reads and live EvalAI GETs.
It does nothing mutable unless --execute is passed with immutable image/payload IDs.
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


PROFILES = {
    "m1_none": {"task":"m1", "arm":"NONE", "build_status":"BUILT_NOT_HOST_VERIFIED", "build_schema":"m1_rift_none_r100_static_build_receipt_v1", "host_status":"HOST_VERIFIED", "host_schema":"m1_rift_none_r100_host_verify_v1", "labels":{"ai.eval.architecture":"RIFT","ai.eval.backend":"cached","ai.eval.identity":"literal_zero_e0_t_none_concat","ai.eval.arm":"NONE","ai.eval.dataloader_workers":"0"}},
    "m2_activity_only": {"task":"m2", "arm":"ACTIVITY_ONLY", "build_status":"BUILT_NOT_HOST_VERIFIED", "build_schema":None, "host_status":"HOST_REAL_PUBLIC_PARITY_PASS", "host_schema":None, "labels":{"ai.eval.architecture":"RIFT-R50-concat","ai.eval.backend":"cached-pytorch-cpu","ai.eval.identity":"static-ablation","ai.eval.arm":"ACTIVITY_ONLY","ai.eval.dataloader_workers":"0"}},
    "m2_none": {"task":"m2", "arm":"NONE", "build_status":"BUILT_NOT_HOST_VERIFIED", "build_schema":None, "host_status":"HOST_REAL_PUBLIC_PARITY_PASS", "host_schema":None, "labels":{"ai.eval.architecture":"RIFT-R50-concat","ai.eval.backend":"cached-pytorch-cpu","ai.eval.identity":"static-ablation","ai.eval.arm":"NONE","ai.eval.dataloader_workers":"0"}},
    "h1_activity_only": {"task":"h1", "arm":"ACTIVITY_ONLY", "build_status":"BUILT_NOT_HOST_VERIFIED", "build_schema":None, "host_status":"HOST_PACK_VERIFY_PASS", "host_schema":None, "labels":{"ai.eval.architecture":"RIFT","ai.eval.backend":"cached","ai.eval.identity":"ACTIVITY_ONLY","ai.eval.ablation.arm":"ACTIVITY_ONLY"}},
    "h1_none": {"task":"h1", "arm":"NONE", "build_status":"BUILT_NOT_HOST_VERIFIED", "build_schema":None, "host_status":"HOST_PACK_VERIFY_PASS", "host_schema":None, "labels":{"ai.eval.architecture":"RIFT","ai.eval.backend":"cached","ai.eval.identity":"NONE","ai.eval.ablation.arm":"NONE"}},
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
    profile = str(candidate["profile"]); spec = PROFILES[profile]
    payload = Path(str(candidate["payload_path"])); host_path = Path(str(candidate["host_receipt_path"])); build_path = Path(str(candidate["build_receipt_path"])); container_path = Path(str(candidate["container_receipt_path"]))
    for path, field in ((payload,"payload_sha256"),(host_path,"host_receipt_sha256"),(build_path,"build_receipt_sha256"),(container_path,"container_receipt_sha256")):
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != str(candidate[field]): raise RuntimeError(f"immutable {field} binding drift")
    build=json.loads(build_path.read_text()); host=json.loads(host_path.read_text()); container=json.loads(container_path.read_text())
    if build.get("status") != spec["build_status"] or (spec["build_schema"] and build.get("schema") != spec["build_schema"]): raise RuntimeError("build receipt contract drift")
    if build.get("payload_sha256") != candidate["payload_sha256"]: raise RuntimeError("build/payload binding drift")
    if host.get("status") != spec["host_status"] or (spec["host_schema"] and host.get("schema") != spec["host_schema"]): raise RuntimeError("host receipt contract drift")
    if host.get("payload_sha256") != candidate["payload_sha256"]: raise RuntimeError("host/payload binding drift")
    if profile != "m1_none" and (build.get("arm") != spec["arm"] or host.get("arm") != spec["arm"]): raise RuntimeError("receipt arm/profile drift")
    required={"schema":"root_final_ablation_container_verify_v1","status":"PASSED","image_id":candidate["image_id"],"payload_sha256":candidate["payload_sha256"],"bytes_exact":True,"smoke_exit_code":0}
    if any(container.get(k)!=v for k,v in required.items()): raise RuntimeError("container verification receipt drift")
    if "smoke" in str(payload).lower() or "smoke" in str(host_path).lower(): raise RuntimeError("runtime smoke is not formal readiness")
    def gate(value: Any, field: str) -> None:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not __import__("math").isfinite(value) or not 0.0 <= float(value) <= 1e-5: raise RuntimeError(f"host tolerance drift: {field}")
    if profile == "m1_none":
        if build.get("information_route",{}).get("e0") != "literal_zero_64x100" or host.get("route") != {"e0_literal_zero":True,"direct_t_literal_zero":True,"encoder_never_called":True,"exact_legal_tags":True}: raise RuntimeError("M1 NONE route drift")
        tags={"20120924","20120926","20120927","20120928","20121004","20121017","20121024"}; rows=host.get("per_tag")
        if not isinstance(rows,Mapping) or set(rows)!=tags: raise RuntimeError("M1 per-tag roster drift")
        for tag,row in rows.items():
            if not isinstance(row,Mapping) or row.get("steps")!=128: raise RuntimeError(f"M1 tag step drift {tag}")
            gate(row.get("packed_cached_vs_full_r100_max_abs"),f"M1 {tag}")
        for key in ("native_b8_inactive_row_state_unchanged","native_b8_packed_resume_after_inactive_checked","native_b8_all_zero_valid_checked"):
            if host.get(key) is not True: raise RuntimeError(f"M1 missing check {key}")
        for key in ("max_packed_cached_vs_full_r100","native_b8_cached_vs_full_r100_max_abs","native_b8_packed_vs_cached_max_abs","native_b8_packed_vs_full_r100_max_abs","mixed_seven_cached_vs_full_r100_max_abs","query_pad_99_full_stream_max_abs"): gate(host.get(key),key)
    elif profile.startswith("h1_"):
        rows=host.get("B")
        if not isinstance(rows,Mapping) or set(rows)!={"B1","B8","MIXED_2HI_2HO"} or host.get("true_zero_valid_checked") is not True: raise RuntimeError("H1 host roster/zero check drift")
        for name,row in rows.items():
            if not isinstance(row,Mapping) or row.get("full_endpoints") != [0,1,79,299,319,359] or row.get("reset_fresh_checked") is not True: raise RuntimeError(f"H1 {name} check drift")
            if name in ("B8","MIXED_2HI_2HO") and row.get("inactive_resume_checked") is not True: raise RuntimeError(f"H1 {name} inactive drift")
            if name=="B1" and row.get("inactive_resume_checked") is not False: raise RuntimeError("H1 B1 inactive drift")
            gate(row.get("max_full_abs"),f"H1 {name}")
    else:
        rows=host.get("parity")
        if not isinstance(rows,Mapping) or set(rows)!={"B1","B7_native","mixed_partial"} or host.get("partial_rows_raw4_kv_history_resume_checked") is not True: raise RuntimeError("M2 host parity drift")
        for name,row in rows.items():
            if not isinstance(row,Mapping) or row.get("steps")!=80: raise RuntimeError(f"M2 {name} step drift")
            gate(row.get("max_abs"),f"M2 {name}")
        gate(host.get("inactive_resume_max_abs"),"M2 inactive")
        probe=host.get("all_zero_valid_probe")
        if not isinstance(probe,list) or len(probe)!=1 or not isinstance(probe[0],list) or len(probe[0])!=2 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and __import__("math").isfinite(v) for v in probe[0]): raise RuntimeError("M2 zero probe drift")

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


def load_candidate(manifest: Path, profile: str) -> dict[str, Any]:
    payload=json.loads(manifest.read_text(encoding="utf-8"))
    if profile not in PROFILES or payload.get("profile") != profile: raise RuntimeError("manifest profile is not fixed/allowed")
    required=("profile","image_tag","image_id","payload_sha256","method_label","method_name","method_description","budget_disclosure","state_path","host_receipt_path","host_receipt_sha256","payload_path","build_receipt_path","build_receipt_sha256","container_receipt_path","container_receipt_sha256")
    missing=[k for k in required if not isinstance(payload.get(k),str) or not payload[k]]
    if missing: raise RuntimeError(f"candidate manifest missing {missing}")
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
    spec=PROFILES[candidate["profile"]]
    expected = {**spec["labels"], "ai.eval.payload.sha256": candidate["payload_sha256"], "ai.eval.method": candidate["method_label"], "ai.eval.task": spec["task"]}
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
        "schema_version": "root_final_ablation_fixed_control_evalai_push_state_v1",
        "profile": candidate["profile"],
        "arm": PROFILES[candidate["profile"]]["arm"],
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
        "method_description": candidate["method_description"] + "\n" + candidate["budget_disclosure"],
        "submission_metadata": json.dumps(SUBMISSION_ATTRIBUTES),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="final_ablation_evalai_") as tmp:
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
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    parser.add_argument("--confirm-payload-sha256", default="")
    args = parser.parse_args()
    candidate = load_candidate(Path(args.manifest), args.profile)
    if candidate.get("register") is True and not args.execute:
        raise RuntimeError("manifest already marked registered; refuse silent re-preflight as execute")
    client, image, phase, limits, runtime = preflight(candidate)
    report = {
        "mode": "execute" if args.execute else "read_only_preflight",
        "profile": candidate["profile"],
        "arm": PROFILES[candidate["profile"]]["arm"],
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
        for key in ("profile", "image_tag", "image_id", "payload_sha256"):
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
                "profile": completed["profile"],
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
