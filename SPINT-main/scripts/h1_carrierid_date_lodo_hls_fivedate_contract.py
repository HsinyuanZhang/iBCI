#!/usr/bin/env python3
"""Pure-JSON contracts for the deferred H1 five-date H-LS control.

The current H1 H-S/H-C five-date producer is still running.  This helper is
therefore deliberately *not* a data or training module: it reads at most
immutable JSON receipts, never dereferences a path recorded in a receipt, and
does not import Torch, Lightning, Hydra, an NWB reader, or a target loader.

H-LS means a strong temporal velocity-label misalignment control.  It is not
a label-deletion control: while neural rates and support/query windows stay
fixed, the velocity rows used to fit the four-dimensional carrier are
deterministically time-rotated before refitting.  The source-trained H-LS
consumer and its deployment carrier must both use that declared intervention.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import uuid
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DATES = ("19250108", "19250113", "19250115", "19250119", "19250120")

UPSTREAM_AGGREGATE_SCHEMA = "h1_carrierid_date_lodo_five_date_heldout_aggregate_v1"
UPSTREAM_AGGREGATE_STATUS = (
    "PASS_H1_CARRIERID_DATE_LODO_FIVE_DATE_SOURCE_DATE_SCREEN_COMPLETE_NO_ROUTE_SELECTED"
)
UPSTREAM_AGGREGATE_DEFAULT = (
    ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase2/"
    "H1_CARRIERID_DATE_LODO_FIVE_DATE_HELDOUT_AGGREGATE_ROUTE_PREREQUISITE_v1.json"
)

NULL_AUDIT_SCHEMA = "h1_carrierid_label_rotation_null_strength_source_audit_v1"
NULL_AUDIT_STATUS = (
    "PASS_H1_CARRIERID_LABEL_ROTATION_SOURCE_NULL_STRENGTH_AUDIT__TEMPORAL_MISALIGNMENT_ONLY"
)
NULL_AUDIT_SHA256 = "6404572da206e7664dc236b9d15005cc05b129b64b1541a841ca75a18f248409"
NULL_AUDIT_DEFAULT = (
    ROOT / "pilot_artifacts/h1_carrierid_label_rotation_null_strength/"
    "H1_CARRIERID_LABEL_ROTATION_SOURCE_NULL_STRENGTH_AUDIT_v1.json"
)

WAITING_SCHEMA = "h1_carrierid_date_lodo_hls_fivedate_waiting_plan_v1"
WAITING_STATUS = "WAITING_FOR_HS_HC_FIVE_DATE_IMMUTABLE_AGGREGATE"
SOURCE_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_hls_source_preflight_v1"
SOURCE_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_SOURCE_ONLY_NOT_LAUNCHED"
SOURCE_TERMINAL_SCHEMA = "h1_carrierid_date_lodo_hls_source_e49_terminal_check_v1"
SOURCE_TERMINAL_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_SOURCE_E49_NO_TARGET"
EVALUATOR_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_hls_target_evaluator_preflight_v1"
EVALUATOR_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_TARGET_EVALUATOR_READY_NOT_RUN"
LAUNCH_SCHEMA = "h1_carrierid_date_lodo_hls_fivedate_launch_receipt_v1"
LAUNCH_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_FIVEDATE_PREPARED_NOT_LAUNCHED"


class HlsFiveDateContractError(ValueError):
    """A H-LS five-date provenance or no-target invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HlsFiveDateContractError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_mode_0444(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_file() and not candidate.is_symlink() and stat.S_IMODE(candidate.stat().st_mode) == 0o444


def read_immutable_json(path: str | Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(immutable_mode_0444(candidate), f"immutable mode-0444 JSON required: {candidate}")
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HlsFiveDateContractError(f"invalid immutable JSON: {candidate}") from error
    _need(isinstance(body, dict), f"immutable JSON is not an object: {candidate}")
    _need(body.get("schema") == schema and body.get("status") == status,
          f"receipt schema/status drift: {candidate}")
    return candidate, body, sha256_file(candidate)


def validate_null_strength_audit(path: str | Path = NULL_AUDIT_DEFAULT) -> tuple[Path, dict[str, Any], str]:
    candidate, body, digest = read_immutable_json(path, schema=NULL_AUDIT_SCHEMA, status=NULL_AUDIT_STATUS)
    _need(digest == NULL_AUDIT_SHA256, "H-LS must bind the sealed source-only label-rotation strength audit SHA")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping), "H-LS null-strength audit has no scope")
    _need(scope.get("outer_date_recordings_opened") == 0 and scope.get("outer_date_bytes_read") == 0,
          "H-LS null-strength audit opened an outer-date recording")
    _need(scope.get("cuda_constructed_or_launched") is False and scope.get("checkpoint_created_or_loaded") is False,
          "H-LS null-strength audit is not CPU/no-checkpoint evidence")
    return candidate, body, digest


def upstream_aggregate_state(path: str | Path = UPSTREAM_AGGREGATE_DEFAULT) -> dict[str, Any]:
    """Inspect only the known aggregate JSON path; missing means *waiting*.

    This deliberately does not scan any output directory, enumerate a target
    recording, or infer a date from a partial result.
    """

    candidate = Path(path).resolve()
    if not candidate.exists():
        return {"ready": False, "reason": "MISSING", "path": str(candidate), "sha256": None}
    try:
        aggregate_path, body, digest = read_immutable_json(
            candidate, schema=UPSTREAM_AGGREGATE_SCHEMA, status=UPSTREAM_AGGREGATE_STATUS,
        )
    except HlsFiveDateContractError as error:
        return {"ready": False, "reason": f"INVALID:{error}", "path": str(candidate), "sha256": None}
    route = body.get("route_prerequisite")
    complete = (
        tuple(body.get("required_outer_dates", ())) == DATES
        and body.get("all_five_date_receipts_present_and_validated") is True
        and isinstance(route, Mapping)
        and route.get("status") == "source/date screen complete"
        and route.get("automatic_route_selection") == "FORBIDDEN"
    )
    return {
        "ready": bool(complete), "reason": "READY" if complete else "INCOMPLETE_OR_NONCANONICAL",
        "path": str(aggregate_path), "sha256": digest,
    }


def write_immutable_json(path: str | Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    """Atomically create, never replace, a mode-0444 receipt."""

    output = Path(path).resolve()
    _need(not os.path.lexists(str(output)), f"refusing to overwrite receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(immutable_mode_0444(output), f"receipt publish lost mode 0444: {output}")
    return output, hashlib.sha256(encoded).hexdigest()


def require_complete_upstream(path: str | Path) -> tuple[Path, str]:
    state = upstream_aggregate_state(path)
    _need(state["ready"] is True, f"H-LS five-date route waits for H-S/H-C aggregate: {state['reason']}")
    return Path(str(state["path"])), str(state["sha256"])

