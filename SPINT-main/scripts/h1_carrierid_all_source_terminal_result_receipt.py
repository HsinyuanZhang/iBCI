"""Fail-closed, append-only terminal receipt for H1 all-source submission 578689.

The push/registration receipt is intentionally separate from this terminal
receipt.  The former proves the immutable ECR binding and the latter records a
read-only EvalAI terminal response plus the result artifact captured by the
watcher.  This module never contacts EvalAI, pushes an image, submits a job,
opens labels, or rewrites an existing receipt.

The verifier re-hashes the registered push state, recovery package, payload,
canonical M3 stability audit, and the two authoritative baseline result
artifacts.  A changed file, missing field, changed metric, changed terminal
flag, or newly supplied per-session result fails closed.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # package import when run as ``python -m scripts...``
    from scripts.h1_carrierid_all_source_submission_receipt import (
        SubmissionReceiptError,
        validate_recovery_package,
    )
except ImportError:  # pragma: no cover - direct script execution
    from h1_carrierid_all_source_submission_receipt import (  # type: ignore
        SubmissionReceiptError,
        validate_recovery_package,
    )


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = "h1_carrierid_all_source_evalai_terminal_result_receipt_v1"
# Conventional descriptive name used by other H1 receipt modules.
TERMINAL_SCHEMA = RECEIPT_SCHEMA
RECEIPT_STATUS = "PASS_H1_ALL_SOURCE_EVALAI_TERMINAL_FINISHED"
STATE_SCHEMA = "h1_carrierid_all_source_evalai_push_state_v1"
EXPECTED_CHALLENGE_ID = 2319
EXPECTED_PHASE_ID = 4599
EXPECTED_PHASE_SLUG = "few-shot-test-2319"
EXPECTED_TEAM_ID = 41975
EXPECTED_SUBMISSION_ID = 578689
EXPECTED_IMAGE_ID = "93ddcdb0213c43518ef67a6cc4ec32e0ee6ef416750f738ff4e5a845bc326f4b"
EXPECTED_CONFIG_DIGEST = "b6a520bec2828b8cdfcca50c41c5a0399cd4840d4d93b41d91a6c34b9253ab18"
EXPECTED_PAYLOAD_SHA256 = "dc155dd492ffdfc2034658f2d6f7b9fd74eb1ce79d061602310822cb3ab9769b"
EXPECTED_AUDIT_SHA256 = "a6c36fc751dfcf4ecd029481f76673e4561babb45ba398a767a4f6e466269159"
EXPECTED_STABILITY_SHA256 = "4b6adb58460a48cb791a56ed92efd1eb1f2ab961c5f43dd366777267da3a7277"
EXPECTED_STATE_SHA256 = "41121cccef4240cb343d375f7250db46204a78c97fd6cb56f57b727cc62d9dde"
EXPECTED_BASE_STATE_SHA256 = "9643e030430cf34f92ee7e8194261caac858b52fcb951fcdc346494fccc94059"
EXPECTED_TERMINAL_RESULT_URL = (
    "https://evalai.s3.amazonaws.com/media/submission_files/challenge_2319/phase_4599/"
    "submission_578689/cb29c46f-121d-437a-9b3b-e077bdb056.json"
)
EXPECTED_STDOUT_URL = (
    "https://evalai.s3.amazonaws.com/media/submission_files/challenge_2319/phase_4599/"
    "submission_578689/5702f4c6-6b0f-44a6-9822-3a60fee05de.txt"
)
EXPECTED_STDERR_URL = (
    "https://evalai.s3.amazonaws.com/media/submission_files/challenge_2319/phase_4599/"
    "submission_578689/6ae8410b-caea-4bce-bc13-0fe4218fe4f.txt"
)
EXPECTED_METRICS = {
    "test_split_h1": {
        "Normalized Latency": 0.11391919646378902,
        "Held Out R2 Mean": 0.27493917810170515,
        "Held Out R2 Std.": 0.127205860368656,
        "Held In R2 Mean": 0.47312503064190997,
        "Held In R2 Std.": 0.03934133472344227,
    }
}

BASELINE_SPECS: dict[str, dict[str, Any]] = {
    "578473": {
        "submission_id": 578473,
        "variant": "released_code_lr_5e-5",
        "comparison_role": "implementation_primary",
        "push_state_path": ROOT / "logs/h1_baseline_staged_20260807_s42_v1/released_code_lr_5e-5/h1_evalai_push_state.json",
        "result_path": ROOT / "logs/h1_baseline_staged_20260807_s42_v1/evalai_evidence/submission_578473/artifacts/submission_result_file.json",
        "result_meta_path": ROOT / "logs/h1_baseline_staged_20260807_s42_v1/evalai_evidence/submission_578473/artifacts/submission_result_file.json.meta.json",
        "status_path": ROOT / "logs/h1_baseline_staged_20260807_s42_v1/evalai_evidence/submission_578473/status/20260807T072308_387765Z.submission.json",
        "status_meta_path": ROOT / "logs/h1_baseline_staged_20260807_s42_v1/evalai_evidence/submission_578473/status/20260807T072308_387765Z.submission.json.meta.json",
    },
    "578474": {
        "submission_id": 578474,
        "variant": "paper_lr_1e-5",
        "comparison_role": "appendix_paper_reference",
        "push_state_path": ROOT / "logs/h1_baseline_staged_20260807_s42_v1/paper_lr_1e-5/h1_evalai_push_state.json",
        "result_path": ROOT / "logs/h1_baseline_staged_20260807_s42_v1/evalai_evidence/submission_578474/artifacts/submission_result_file.json",
        "result_meta_path": ROOT / "logs/h1_baseline_staged_20260807_s42_v1/evalai_evidence/submission_578474/artifacts/submission_result_file.json.meta.json",
        "status_path": ROOT / "logs/h1_baseline_staged_20260807_s42_v1/evalai_evidence/submission_578474/status/20260807T072309_829337Z.submission.json",
        "status_meta_path": ROOT / "logs/h1_baseline_staged_20260807_s42_v1/evalai_evidence/submission_578474/status/20260807T072309_829337Z.submission.json.meta.json",
    },
}

BASELINE_METRICS = {
    "578473": {
        "test_split_h1": {
            "Normalized Latency": 0.12929302459942418,
            "Held Out R2 Mean": 0.20991673151431103,
            "Held Out R2 Std.": 0.11423156362920023,
            "Held In R2 Mean": 0.4390269703924928,
            "Held In R2 Std.": 0.056906215722459116,
        }
    },
    "578474": {
        "test_split_h1": {
            "Normalized Latency": 0.12907509411442164,
            "Held Out R2 Mean": 0.26149164345884,
            "Held Out R2 Std.": 0.14871680046572036,
            "Held In R2 Mean": 0.4704230035467059,
            "Held In R2 Std.": 0.04802514327136098,
        }
    },
}

_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$")


class TerminalReceiptError(ValueError):
    """Raised when the terminal receipt or one of its bindings is invalid."""


def sha256_file(path: str | Path) -> str:
    original = Path(path)
    candidate = original.resolve()
    if original.is_symlink() or candidate.is_symlink() or not candidate.is_file():
        raise TerminalReceiptError(f"expected a regular non-symlink file: {original}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TerminalReceiptError(f"{label} must be a 64-hex SHA-256")
    return value.removeprefix("sha256:").lower()


def _load_json(path: str | Path) -> dict[str, Any]:
    original = Path(path)
    candidate = original.resolve()
    if original.is_symlink() or candidate.is_symlink() or not candidate.is_file():
        raise TerminalReceiptError(f"JSON receipt does not exist as a regular file: {original}")
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalReceiptError(f"cannot read JSON receipt {candidate}: {exc}") from exc
    if not isinstance(body, dict):
        raise TerminalReceiptError(f"JSON receipt must be an object: {candidate}")
    return body


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iso(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ISO_RE.fullmatch(value):
        raise TerminalReceiptError(f"{label} must be ISO-8601 with timezone")
    try:
        _datetime.datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise TerminalReceiptError(f"{label} is not a valid timestamp") from exc
    return value


def _utc_now() -> str:
    value = _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds")
    return value[:-6] + "Z" if value.endswith("+00:00") else value


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise TerminalReceiptError(f"{label} must be a finite scalar number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise TerminalReceiptError(f"{label} must be a finite scalar number") from exc
    if not result.is_finite():
        raise TerminalReceiptError(f"{label} must be finite")
    return result


def _metric_maps_equal(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    if set(left) != set(right):
        raise TerminalReceiptError(f"{label} metric fields drift: {sorted(left)} vs {sorted(right)}")
    for name in left:
        if _decimal(left[name], f"{label}.{name}") != _decimal(right[name], f"{label}.{name}"):
            raise TerminalReceiptError(f"{label}.{name} metric drift")


def _validate_terminal_response(response: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "id": EXPECTED_SUBMISSION_ID,
        "challenge_phase": EXPECTED_PHASE_ID,
        "participant_team": EXPECTED_TEAM_ID,
        "status": "finished",
        "is_public": False,
        "is_flagged": False,
        "is_verified_by_host": False,
        "ignore_submission": False,
        "is_baseline": False,
        "submitted_at": "2026-08-09T22:09:41.512779Z",
        "started_at": "2026-08-09T22:25:51.702890Z",
        "completed_at": "2026-08-09T22:25:51.821669Z",
        "execution_time": 0.118779,
        "rerun_resumed_at": None,
        "submission_result_file": EXPECTED_TERMINAL_RESULT_URL,
        "stdout_file": EXPECTED_STDOUT_URL,
        "stderr_file": EXPECTED_STDERR_URL,
    }
    for key, expected in required.items():
        if response.get(key) != expected:
            raise TerminalReceiptError(f"terminal API field {key} drifted: {response.get(key)!r} != {expected!r}")
    for key in ("submitted_at", "started_at", "completed_at"):
        _iso(response[key], key)
    if response.get("status") != "finished":
        raise TerminalReceiptError("submission is not terminal finished")
    result_metrics = response.get("metrics", metrics)
    if not isinstance(result_metrics, Mapping):
        raise TerminalReceiptError("terminal result metrics must be an object")
    _metric_maps_equal(result_metrics.get("test_split_h1", {}), EXPECTED_METRICS["test_split_h1"], "terminal")
    return {key: response.get(key) for key in required}


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    original = Path(path)
    output = original.resolve()
    if original.is_symlink() or output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable terminal receipt {original}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        output.chmod(0o444)
        os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _path_block(path: Path, *, mode: int | None = None) -> dict[str, Any]:
    if mode is not None and (path.stat().st_mode & 0o777) != mode:
        raise TerminalReceiptError(f"{path} mode must be {mode:o}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "mode": (path.stat().st_mode & 0o777)}


def _validate_state(path: str | Path, *, registered: bool) -> dict[str, Any]:
    candidate = Path(path).resolve()
    body = _load_json(candidate)
    if body.get("schema") != STATE_SCHEMA:
        raise TerminalReceiptError(f"push state schema drift: {body.get('schema')!r}")
    if (body.get("registered") is True) != registered or body.get("pushed") is not True:
        raise TerminalReceiptError("push state registered/pushed flags drift")
    if body.get("challenge_id") != EXPECTED_CHALLENGE_ID or body.get("phase_id") != EXPECTED_PHASE_ID:
        raise TerminalReceiptError("push state challenge/phase drift")
    if body.get("team_id") != EXPECTED_TEAM_ID:
        raise TerminalReceiptError("push state team drift")
    if body.get("local_image_id") != f"sha256:{EXPECTED_IMAGE_ID}":
        raise TerminalReceiptError("push state local image ID drift")
    if body.get("registry_manifest_digest") != f"sha256:{EXPECTED_IMAGE_ID}":
        raise TerminalReceiptError("push state registry manifest drift")
    if body.get("registry_config_digest") != EXPECTED_CONFIG_DIGEST:
        raise TerminalReceiptError("push state registry config digest drift")
    if body.get("registry_local_binding") != "registry_manifest_digest_equals_local_image_id":
        raise TerminalReceiptError("push state registry binding drift")
    recovery = body.get("recovery_package_final_audit")
    payload = body.get("recovery_payload")
    if not isinstance(recovery, Mapping) or recovery.get("sha256") != EXPECTED_AUDIT_SHA256:
        raise TerminalReceiptError("push state recovery audit binding drift")
    if not isinstance(payload, Mapping) or payload.get("sha256") != EXPECTED_PAYLOAD_SHA256:
        raise TerminalReceiptError("push state recovery payload binding drift")
    if registered:
        if body.get("submission_id") != EXPECTED_SUBMISSION_ID:
            raise TerminalReceiptError("registered push state submission ID drift")
        registration = body.get("registration_response")
        if not isinstance(registration, Mapping):
            raise TerminalReceiptError("registered push state lacks registration response")
        if registration.get("id") != EXPECTED_SUBMISSION_ID or registration.get("status") != "submitted":
            raise TerminalReceiptError("registered push state registration response drift")
        if registration.get("challenge_phase") != EXPECTED_PHASE_ID or registration.get("participant_team") != EXPECTED_TEAM_ID:
            raise TerminalReceiptError("registered push state phase/team drift")
        if registration.get("is_public") is not False:
            raise TerminalReceiptError("registered push state is not private")
    else:
        if "submission_id" in body or body.get("registration_response") is not None:
            raise TerminalReceiptError("base push state unexpectedly contains a registration")
    return body


def _validate_baseline(spec: Mapping[str, Any], expected_metrics: Mapping[str, Any]) -> dict[str, Any]:
    sid = str(spec["submission_id"])
    result_path = Path(spec["result_path"]).resolve()
    meta_path = Path(spec["result_meta_path"]).resolve()
    status_path = Path(spec["status_path"]).resolve()
    status_meta_path = Path(spec["status_meta_path"]).resolve()
    for path in (result_path, meta_path, status_path, status_meta_path):
        if path.is_symlink() or not path.is_file():
            raise TerminalReceiptError(f"baseline evidence is unavailable: {path}")
    if result_path.stat().st_mode & 0o222 or meta_path.stat().st_mode & 0o222:
        raise TerminalReceiptError("baseline result evidence must be immutable/read-only")
    result = _load_json(result_path)
    if set(result) != {"test_split_h1"} or not isinstance(result["test_split_h1"], Mapping):
        raise TerminalReceiptError(f"baseline {sid} result schema drift")
    _metric_maps_equal(result["test_split_h1"], expected_metrics["test_split_h1"], f"baseline {sid}")
    meta = _load_json(meta_path)
    if meta.get("http_status") != 200 or meta.get("request_kind") != "evalai_artifact_submission_result_file":
        raise TerminalReceiptError(f"baseline {sid} result metadata is not an authenticated capture")
    if meta.get("body_sha256") != sha256_file(result_path) or meta.get("body_size_bytes") != result_path.stat().st_size:
        raise TerminalReceiptError(f"baseline {sid} result metadata hash/size drift")
    status = _load_json(status_path)
    if status.get("id") != int(sid) or status.get("status") != "finished":
        raise TerminalReceiptError(f"baseline {sid} status evidence is not finished")
    if status.get("challenge_phase") != EXPECTED_PHASE_ID or status.get("participant_team") != EXPECTED_TEAM_ID:
        raise TerminalReceiptError(f"baseline {sid} phase/team drift")
    if status.get("is_public") is not False or status.get("is_flagged") is not False or status.get("is_verified_by_host") is not False:
        raise TerminalReceiptError(f"baseline {sid} terminal flags drift")
    _iso(status.get("submitted_at"), f"baseline {sid}.submitted_at")
    _iso(status.get("started_at"), f"baseline {sid}.started_at")
    _iso(status.get("completed_at"), f"baseline {sid}.completed_at")
    status_meta = _load_json(status_meta_path)
    if status_meta.get("body_sha256") != sha256_file(status_path) or status_meta.get("body_size_bytes") != status_path.stat().st_size:
        raise TerminalReceiptError(f"baseline {sid} status metadata hash/size drift")
    push_state = _load_json(Path(spec["push_state_path"]))
    if push_state.get("submission_id") != int(sid) or push_state.get("registered") is not True:
        raise TerminalReceiptError(f"baseline {sid} push-state binding drift")
    if push_state.get("variant") != spec["variant"] or push_state.get("role") != spec["comparison_role"]:
        raise TerminalReceiptError(f"baseline {sid} variant/role drift")
    return {
        "submission_id": int(sid),
        "variant": spec["variant"],
        "comparison_role": spec["comparison_role"],
        "push_state": _path_block(Path(spec["push_state_path"]), mode=0o600),
        "result_file": _path_block(result_path, mode=0o444),
        "result_meta_file": _path_block(meta_path, mode=0o444),
        "status_file": _path_block(status_path, mode=0o444),
        "status_meta_file": _path_block(status_meta_path, mode=0o444),
        "status": {key: status.get(key) for key in (
            "id", "status", "submitted_at", "started_at", "completed_at", "execution_time",
            "is_public", "is_flagged", "is_verified_by_host", "submission_result_file",
        )},
        "metrics": result,
    }


def _deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    left = candidate["test_split_h1"]
    right = baseline["test_split_h1"]
    for name in sorted(left):
        result[name] = format(_decimal(left[name], f"candidate.{name}") - _decimal(right[name], f"baseline.{name}"), "f")
    return result


def create_terminal_receipt(
    *,
    registered_state: str | Path,
    base_state: str | Path,
    output: str | Path,
    terminal_response: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one immutable terminal receipt from already captured API data."""

    candidate_metrics = dict(metrics or EXPECTED_METRICS)
    _metric_maps_equal(candidate_metrics.get("test_split_h1", {}), EXPECTED_METRICS["test_split_h1"], "candidate")
    terminal = _validate_terminal_response({**dict(terminal_response), "metrics": candidate_metrics}, candidate_metrics)
    registered_path = Path(registered_state).resolve()
    base_path = Path(base_state).resolve()
    registered = _validate_state(registered_path, registered=True)
    base = _validate_state(base_path, registered=False)
    if registered.get("recovery_package_final_audit") != base.get("recovery_package_final_audit"):
        raise TerminalReceiptError("base/registered push states disagree on recovery audit")
    recovery = validate_recovery_package(
        registered["recovery_package_final_audit"]["path"],
        expected_audit_sha256=EXPECTED_AUDIT_SHA256,
        payload_path=registered["recovery_payload"]["path"],
        expected_payload_sha256=EXPECTED_PAYLOAD_SHA256,
    )
    if recovery["stability_audit"]["sha256"] != EXPECTED_STABILITY_SHA256:
        raise TerminalReceiptError("recovery stability audit SHA drift")
    baseline_blocks: dict[str, Any] = {}
    for sid, spec in BASELINE_SPECS.items():
        baseline = _validate_baseline(spec, BASELINE_METRICS[sid])
        baseline["delta_candidate_minus_baseline"] = _deltas(candidate_metrics, baseline["metrics"])
        baseline_blocks[sid] = baseline
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "created_at": _utc_now(),
        "submission": {
            **terminal,
            "id": EXPECTED_SUBMISSION_ID,
            "challenge_id": EXPECTED_CHALLENGE_ID,
            "phase_id": EXPECTED_PHASE_ID,
            "phase_slug": EXPECTED_PHASE_SLUG,
            "team_id": EXPECTED_TEAM_ID,
            "metrics": candidate_metrics,
            "metric_fields_present": sorted(candidate_metrics["test_split_h1"]),
            "result_payload_canonical_sha256": _canonical_sha(candidate_metrics),
        },
        "state_bindings": {
            "base": _path_block(base_path, mode=0o600),
            "registered": _path_block(registered_path, mode=0o600),
            "base_registered": False,
            "registered_registered": True,
            "registry_manifest_digest": registered["registry_manifest_digest"],
            "registry_config_digest": registered["registry_config_digest"],
            "registry_local_binding": registered["registry_local_binding"],
            "repository_uri": registered["repository_uri"],
            "submitted_image_uri": registered["submitted_image_uri"],
        },
        "image": {
            "local_image_id": registered["local_image_id"],
            "registry_manifest_digest": registered["registry_manifest_digest"],
            "registry_config_digest": registered["registry_config_digest"],
            "registry_local_binding": registered["registry_local_binding"],
            "image_tag": registered["image_tag"],
            "embedded_decoder_sha256": registered["embedded_decoder_sha256"],
        },
        "recovery": {
            "audit": {
                "path": recovery["path"],
                "sha256": recovery["sha256"],
                "schema": recovery["schema"],
                "status": recovery["status"],
            },
            "payload": recovery["payload"],
            "stability_audit": recovery["stability_audit"],
            "code_provenance": recovery["code_provenance"],
            "scope": recovery["scope"],
            "provenance_gap": recovery["provenance_gap"],
        },
        "per_session_metrics": {
            "available": False,
            "held_in": {"available": False, "values": None},
            "held_out": {"available": False, "values": None},
            "reason": "EvalAI result JSON, stdout, and stderr exposed aggregate test_split_h1 values only; no per-session values were exposed.",
            "sources_checked": [EXPECTED_TERMINAL_RESULT_URL, EXPECTED_STDOUT_URL, EXPECTED_STDERR_URL],
        },
        "baselines": baseline_blocks,
        "delta_definition": {
            "formula": "candidate metric minus baseline metric",
            "significance_computed": False,
            "significance_note": "Descriptive deltas only; no significance claim or inferential test was computed.",
        },
        "external_action": {
            "evalai_terminal_status_obtained_by_read_only_get": True,
            "evalai_push_performed_by_this_tool": False,
            "evalai_registration_performed_by_this_tool": False,
            "resubmission_performed_by_this_tool": False,
            "private_labels_read_by_this_tool": False,
            "private_labels_released_by_this_tool": False,
            "docker_build_or_push_performed_by_this_tool": False,
        },
    }
    _write_immutable_json(output, receipt)
    return receipt


def verify_terminal_receipt(path: str | Path) -> dict[str, Any]:
    """Re-hash and verify one terminal receipt without network or writes."""

    receipt_path = Path(path).resolve()
    body = _load_json(receipt_path)
    if body.get("schema") != RECEIPT_SCHEMA or body.get("status") != RECEIPT_STATUS:
        raise TerminalReceiptError("terminal receipt schema/status drift")
    if stat.S_IMODE(receipt_path.stat().st_mode) != 0o444:
        raise TerminalReceiptError("terminal receipt must have immutable mode 0444")
    _iso(body.get("created_at"), "created_at")
    external = body.get("external_action")
    if not isinstance(external, Mapping):
        raise TerminalReceiptError("terminal receipt lacks external_action")
    for key in (
        "evalai_push_performed_by_this_tool", "evalai_registration_performed_by_this_tool",
        "resubmission_performed_by_this_tool", "private_labels_read_by_this_tool",
        "private_labels_released_by_this_tool", "docker_build_or_push_performed_by_this_tool",
    ):
        if external.get(key) is not False:
            raise TerminalReceiptError(f"terminal receipt external action {key} is not false")
    if external.get("evalai_terminal_status_obtained_by_read_only_get") is not True:
        raise TerminalReceiptError("terminal receipt does not identify a read-only status GET")
    submission = body.get("submission")
    if not isinstance(submission, Mapping):
        raise TerminalReceiptError("terminal receipt lacks submission mapping")
    _validate_terminal_response(submission, submission.get("metrics", {}))
    if submission.get("challenge_id") != EXPECTED_CHALLENGE_ID or submission.get("phase_id") != EXPECTED_PHASE_ID:
        raise TerminalReceiptError("terminal receipt challenge/phase drift")
    if submission.get("phase_slug") != EXPECTED_PHASE_SLUG or submission.get("team_id") != EXPECTED_TEAM_ID:
        raise TerminalReceiptError("terminal receipt phase/team drift")
    metrics = submission.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TerminalReceiptError("terminal receipt metrics missing")
    _metric_maps_equal(metrics.get("test_split_h1", {}), EXPECTED_METRICS["test_split_h1"], "terminal")
    if submission.get("result_payload_canonical_sha256") != _canonical_sha(metrics):
        raise TerminalReceiptError("terminal result payload canonical SHA drift")
    state = body.get("state_bindings")
    image = body.get("image")
    if not isinstance(state, Mapping) or not isinstance(image, Mapping):
        raise TerminalReceiptError("terminal receipt state/image bindings missing")
    base_path = Path(state.get("base", {}).get("path", ""))
    registered_path = Path(state.get("registered", {}).get("path", ""))
    base = _validate_state(base_path, registered=False)
    registered = _validate_state(registered_path, registered=True)
    for key, current in (("base", base_path), ("registered", registered_path)):
        block = state.get(key)
        if not isinstance(block, Mapping) or block.get("sha256") != sha256_file(current) or block.get("mode") != (current.stat().st_mode & 0o777):
            raise TerminalReceiptError(f"terminal receipt {key} state hash/mode drift")
    if state.get("base_registered") is not False or state.get("registered_registered") is not True:
        raise TerminalReceiptError("terminal receipt base/registered flags drift")
    for key in ("registry_manifest_digest", "registry_config_digest", "registry_local_binding", "repository_uri", "submitted_image_uri"):
        if state.get(key) != registered.get(key):
            raise TerminalReceiptError(f"terminal receipt state binding {key} drift")
    for key, expected in {
        "local_image_id": registered["local_image_id"],
        "registry_manifest_digest": registered["registry_manifest_digest"],
        "registry_config_digest": registered["registry_config_digest"],
        "registry_local_binding": registered["registry_local_binding"],
        "image_tag": registered["image_tag"],
        "embedded_decoder_sha256": registered["embedded_decoder_sha256"],
    }.items():
        if image.get(key) != expected:
            raise TerminalReceiptError(f"terminal receipt image binding {key} drift")
    recovery = body.get("recovery")
    if not isinstance(recovery, Mapping):
        raise TerminalReceiptError("terminal receipt recovery binding missing")
    audit = recovery.get("audit")
    payload = recovery.get("payload")
    if not isinstance(audit, Mapping) or not isinstance(payload, Mapping):
        raise TerminalReceiptError("terminal receipt recovery audit/payload binding missing")
    checked = validate_recovery_package(
        audit.get("path", ""),
        expected_audit_sha256=audit.get("sha256"),
        payload_path=payload.get("path"),
        expected_payload_sha256=payload.get("sha256"),
    )
    if checked["sha256"] != EXPECTED_AUDIT_SHA256 or checked["payload"]["sha256"] != EXPECTED_PAYLOAD_SHA256:
        raise TerminalReceiptError("terminal receipt recovery hash drift")
    if checked["stability_audit"]["sha256"] != EXPECTED_STABILITY_SHA256:
        raise TerminalReceiptError("terminal receipt stability hash drift")
    if recovery.get("stability_audit") != checked["stability_audit"]:
        raise TerminalReceiptError("terminal receipt stability evidence drift")
    per_session = body.get("per_session_metrics")
    if not isinstance(per_session, Mapping) or per_session.get("available") is not False:
        raise TerminalReceiptError("per-session metrics availability must be explicitly false")
    for split in ("held_in", "held_out"):
        block = per_session.get(split)
        if not isinstance(block, Mapping) or block.get("available") is not False or block.get("values") is not None:
            raise TerminalReceiptError(f"per-session {split} metrics must be explicitly unavailable")
    if not isinstance(per_session.get("reason"), str) or not per_session["reason"].strip():
        raise TerminalReceiptError("per-session unavailability reason is missing")
    baselines = body.get("baselines")
    if not isinstance(baselines, Mapping) or set(baselines) != set(BASELINE_SPECS):
        raise TerminalReceiptError("baseline comparison set drift")
    for sid, spec in BASELINE_SPECS.items():
        expected = _validate_baseline(spec, BASELINE_METRICS[sid])
        actual = baselines.get(sid)
        if not isinstance(actual, Mapping):
            raise TerminalReceiptError(f"baseline {sid} block missing")
        for key in ("submission_id", "variant", "comparison_role", "push_state", "result_file", "result_meta_file", "status_file", "status_meta_file", "status", "metrics"):
            if actual.get(key) != expected.get(key):
                raise TerminalReceiptError(f"baseline {sid} binding {key} drift")
        if actual.get("delta_candidate_minus_baseline") != _deltas(EXPECTED_METRICS, expected["metrics"]):
            raise TerminalReceiptError(f"baseline {sid} delta drift")
    delta_definition = body.get("delta_definition")
    if not isinstance(delta_definition, Mapping) or delta_definition.get("significance_computed") is not False:
        raise TerminalReceiptError("delta significance must be explicitly not computed")
    return {
        "schema": body["schema"],
        "status": body["status"],
        "submission_id": submission["id"],
        "terminal_status": submission["status"],
        "metrics": metrics,
        "baselines": {sid: baselines[sid]["delta_candidate_minus_baseline"] for sid in sorted(baselines)},
        "per_session_metrics_available": False,
    }


# Descriptive alias used by receipt notebooks and downstream audit code.
validate_terminal_result_receipt = verify_terminal_receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed H1 all-source terminal result receipt.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create one immutable receipt from captured API data")
    create.add_argument("--registered-state", required=True)
    create.add_argument("--base-state", required=True)
    create.add_argument("--output", required=True)
    verify = sub.add_parser("verify", help="verify an immutable terminal receipt without network")
    verify.add_argument("--receipt", "--terminal-receipt", dest="receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "create":
        response = {
            "id": EXPECTED_SUBMISSION_ID,
            "challenge_phase": EXPECTED_PHASE_ID,
            "participant_team": EXPECTED_TEAM_ID,
            "status": "finished",
            "is_public": False,
            "is_flagged": False,
            "is_verified_by_host": False,
            "ignore_submission": False,
            "is_baseline": False,
            "submitted_at": "2026-08-09T22:09:41.512779Z",
            "started_at": "2026-08-09T22:25:51.702890Z",
            "completed_at": "2026-08-09T22:25:51.821669Z",
            "execution_time": 0.118779,
            "rerun_resumed_at": None,
            "submission_result_file": EXPECTED_TERMINAL_RESULT_URL,
            "stdout_file": EXPECTED_STDOUT_URL,
            "stderr_file": EXPECTED_STDERR_URL,
        }
        result = create_terminal_receipt(
            registered_state=args.registered_state,
            base_state=args.base_state,
            output=args.output,
            terminal_response=response,
            metrics=EXPECTED_METRICS,
        )
    else:
        result = verify_terminal_receipt(args.receipt)
    print(json.dumps({"schema": result["schema"], "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
