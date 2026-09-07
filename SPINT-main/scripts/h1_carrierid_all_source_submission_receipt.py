"""Offline receipts for the recovered all-source H1 CarrierID candidate.

This module is intentionally independent from :mod:`h1_evalai_submission_receipt`.
The latter describes the released-code H1 baseline, whereas this contract binds
the *recovery package* produced after the all-source source run.  No Docker
daemon, EvalAI client, benchmark labels, or payload deserialisation is required
here.  A preparation receipt records immutable bytes and the metadata that a
future operator must carry to EvalAI; a final receipt records an externally
returned submission id/status/metric map.

The all-source run was recovered from a legacy, unbound launch.  The missing
nonce-bound execution receipt is therefore represented as three explicit false
flags in every receipt.  A false flag means "not verified", not that a nonce
was reconstructed.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


RECOVERY_PACKAGE_SCHEMA_V4 = "h1_carrierid_all_public_source_recovery_package_audit_v4"
RECOVERY_PACKAGE_SCHEMA = "h1_carrierid_all_public_source_recovery_package_audit_v5"
RECOVERY_PACKAGE_SCHEMAS = {RECOVERY_PACKAGE_SCHEMA}
RECOVERY_PACKAGE_PASS_STATUSES = {
    "PASS_RECOVERY_PACKAGE_WITH_PROVENANCE_GAP_NO_SUBMISSION",
}
RECOVERY_PACKAGE_STATUS = "PASS_RECOVERY_PACKAGE_WITH_PROVENANCE_GAP_NO_SUBMISSION"
RECOVERY_SMOKE_SCHEMA_V4 = "h1_carrierid_all_public_source_recovery_package_smoke_v4"
RECOVERY_SMOKE_SCHEMA = "h1_carrierid_all_public_source_recovery_package_smoke_v5"
RECOVERY_SMOKE_STATUS = "PASS_RECOVERY_CPU_SMOKE_NO_QUERY_OPENED"
PAYLOAD_SCHEMA_V4 = "h1_carrierid_all_public_source_falcon_payload_v1"
PAYLOAD_SCHEMA = "h1_carrierid_all_public_source_falcon_payload_v5"
STABILITY_SCHEMA = "h1_carrierid_all_public_source_recovery_package_m3_stability_audit_v5"
STABILITY_STATUS = "PASS_RECOVERY_M3_FIRST4_PUBLIC_CALIBRATION_STABILITY_NO_QUERY_OPENED"
PREPARE_SCHEMA = "h1_carrierid_all_source_submission_prepare_v1"
FINAL_SCHEMA = "h1_carrierid_all_source_submission_receipt_v1"
EXPECTED_TASK = "h1"
EXPECTED_BATCH_SIZE = 8
EXPECTED_MODEL_FILE = "h1_carrierid_all_source_payload.pkl"
EXPECTED_MODEL_FILE_ALIASES = {
    EXPECTED_MODEL_FILE,
    "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD_v3.pkl",
    "h1_carrierid_all_source_recovery_package_payload_v3.pkl",
    "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD_v4.pkl",
    "h1_carrierid_all_source_recovery_package_payload_v4.pkl",
    "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD_v5.pkl",
    "h1_carrierid_all_source_recovery_package_payload_v5.pkl",
}
EXPECTED_CHALLENGE_ID = 2319
EXPECTED_PHASE_ID = 4599
EXPECTED_PHASE_SLUG = "few-shot-test-2319"
EXPECTED_TEAM_ID = 41975
EXPECTED_EPOCH = 49
EXPECTED_SOURCE_RECORDINGS = 13
PACKAGE_PREFLIGHT_SCHEMA = "h1_carrierid_all_public_source_package_preflight_v1"
PACKAGE_PREFLIGHT_STATUS = "PASS_PACKAGE_PREPARED_NOT_EXPORTED_NOT_SUBMITTED"

PROVENANCE_GAP_FLAGS = (
    "execution_receipt_present",
    "nonce_bound_launch_verified",
    "nonce_reconstructed",
)

METHOD_NAME = "H1 all-source CarrierID recovery candidate (epoch49)"
METHOD_DESCRIPTION = (
    "Source-trained on all 13 public held-in recordings at the fixed epoch49 terminal "
    "checkpoint, with zero target-session neural optimizer/backward steps. Each deployment "
    "record is fitted analytically from its actual calibration TrialNum values: held-in uses "
    "M=4 and held-out uses M=3, with identity tensors [M,1024,176] and no padding or "
    "duplication. The frozen model is therefore test-time adaptive through the explicit "
    "deployment carrier fit. The source run was recovered after an unbound launch: the "
    "missing nonce-bound execution receipt remains an explicit unbound-launch provenance gap "
    "and was not reconstructed."
)

SUBMISSION_ATTRIBUTES = [
    {
        "name": "IsHeldOutZeroShot",
        "type": "boolean",
        "description": "Uses no calibration data from held-out days?",
        "required": True,
        "value": False,
    },
    {
        "name": "IsTestTimeAdaptive",
        "type": "boolean",
        "description": "Fits an analytic per-session carrier from calibration labels at test time?",
        "required": True,
        "value": True,
    },
    {
        "name": "IsPretrained",
        "type": "boolean",
        "description": "Uses training data outside FALCON?",
        "required": True,
        "value": False,
    },
]

_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_IMAGE_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$")
ROOT = Path(__file__).resolve().parents[1]
STABILITY_AUDIT_PATH = (
    ROOT / "pilot_artifacts/h1_carrierid_all_source_official_v1/recovery_package_v1/"
    "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_M3_STABILITY_AUDIT_v5_r4.json"
)
STABILITY_AUDIT_SHA256 = "4b6adb58460a48cb791a56ed92efd1eb1f2ab961c5f43dd366777267da3a7277"


class SubmissionReceiptError(ValueError):
    """A recovery package or submission receipt failed a hard contract."""


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without following directories or symlinks."""

    original = Path(path)
    candidate = original.resolve()
    if original.is_symlink() or not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"expected a regular file: {original}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SubmissionReceiptError(f"{label} must be a 64-hex SHA-256 (optional sha256: prefix)")
    return value.removeprefix("sha256:").lower()


def _load_json(path: str | Path) -> dict[str, Any]:
    original = Path(path)
    candidate = original.resolve()
    if original.is_symlink() or not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"JSON receipt does not exist: {original}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionReceiptError(f"cannot read JSON receipt {candidate}: {exc}") from exc
    if not isinstance(value, dict):
        raise SubmissionReceiptError(f"JSON receipt must be an object: {candidate}")
    return value


def _resolve_path(anchor: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SubmissionReceiptError(f"{label} must be a non-empty path")
    path = Path(value)
    candidate = anchor.parent / path if not path.is_absolute() else path
    if candidate.is_symlink():
        raise SubmissionReceiptError(f"{label} must not be a symlink")
    return candidate.resolve()


def _optional_path(mapping: Mapping[str, Any], anchor: Path, names: Sequence[str], label: str) -> Path | None:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return _resolve_path(anchor, mapping[name], label)
    return None


def _optional_sha(mapping: Mapping[str, Any], names: Sequence[str], label: str) -> str | None:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return _require_sha(mapping[name], label)
    return None


def _validate_provenance_gap(body: Mapping[str, Any]) -> dict[str, Any]:
    gap = body.get("provenance_gap")
    if not isinstance(gap, Mapping):
        raise SubmissionReceiptError("recovery package final audit lacks provenance_gap mapping")
    missing = [key for key in PROVENANCE_GAP_FLAGS if key not in gap]
    if missing:
        raise SubmissionReceiptError(f"recovery provenance gap is missing explicit flags: {missing}")
    for key in PROVENANCE_GAP_FLAGS:
        if type(gap.get(key)) is not bool or gap.get(key) is not False:
            raise SubmissionReceiptError(f"recovery provenance gap {key} must be explicit false")
    # Preserve optional explanatory fields, but never silently manufacture an
    # execution receipt or claim that the launch was bound.
    result = {str(key): value for key, value in gap.items()}
    result.update({key: False for key in PROVENANCE_GAP_FLAGS})
    return result


def _validate_scope(body: Mapping[str, Any]) -> dict[str, Any]:
    scope = body.get("scope")
    if not isinstance(scope, Mapping):
        raise SubmissionReceiptError("recovery package final audit lacks scope mapping")
    checks: dict[str, Any] = {}
    if scope.get("source_only") is not True:
        raise SubmissionReceiptError("recovery package scope must be source_only=true")
    checks["source_only"] = True
    expected_zero = {
        "query_recordings_opened": 0,
        "formal_test_labels_opened": 0,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
    }
    for key, expected in expected_zero.items():
        if type(scope.get(key)) is not int or scope.get(key) != expected:
            raise SubmissionReceiptError(f"recovery package scope {key} must be {expected}")
        checks[key] = expected
    if type(scope.get("cuda_used")) is not bool or scope.get("cuda_used") is not False:
        raise SubmissionReceiptError("recovery package scope cuda_used must be false")
    checks["cuda_used"] = False
    if scope.get("evalai_accessed_or_submitted") is not False:
        raise SubmissionReceiptError("recovery package scope evalai_accessed_or_submitted must be false")
    checks["evalai_accessed_or_submitted"] = False
    if scope.get("calibration_files_indexed") is not None and (
        type(scope.get("calibration_files_indexed")) is not int or scope.get("calibration_files_indexed") != 27
    ):
        raise SubmissionReceiptError("recovery package must index exactly 27 public calibration files")
    if scope.get("calibration_files_indexed") == 27:
        checks["calibration_files_indexed"] = 27
    return checks


def _validate_allowlist(body: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact 13 held-in + 14 held-out calibration roster counts."""

    allowlist = body.get("calibration_allowlist")
    if not isinstance(allowlist, Mapping):
        raise SubmissionReceiptError("recovery package final audit lacks calibration_allowlist mapping")
    counts = allowlist.get("counts")
    if (
        not isinstance(counts, Mapping)
        or any(type(counts.get(key)) is not int for key in ("heldin", "heldout", "total"))
        or dict(counts) != {"heldin": 13, "heldout": 14, "total": 27}
    ):
        raise SubmissionReceiptError("recovery calibration allowlist counts must be exactly 13+14=27")
    files = allowlist.get("files")
    heldin = allowlist.get("heldin")
    heldout = allowlist.get("heldout")
    if not isinstance(files, list) or len(files) != 27:
        raise SubmissionReceiptError("recovery calibration allowlist must contain exactly 27 files")
    if not isinstance(heldin, list) or len(heldin) != 13:
        raise SubmissionReceiptError("recovery calibration allowlist must contain exactly 13 held-in files")
    if not isinstance(heldout, list) or len(heldout) != 14:
        raise SubmissionReceiptError("recovery calibration allowlist must contain exactly 14 held-out files")
    if files != heldin + heldout:
        raise SubmissionReceiptError("recovery calibration allowlist files do not preserve 13+14 order")
    if len(set(files)) != 27 or any(not isinstance(value, str) or not value.strip() for value in files):
        raise SubmissionReceiptError("recovery calibration allowlist contains empty or duplicate paths")
    dataset_tags = allowlist.get("dataset_tags")
    if not isinstance(dataset_tags, list) or len(dataset_tags) != 27 or len(set(dataset_tags)) != 27:
        raise SubmissionReceiptError("recovery calibration dataset_tags must contain exactly 27 unique entries")
    return {
        "counts": {"heldin": 13, "heldout": 14, "total": 27},
        "files": list(files),
        "heldin": list(heldin),
        "heldout": list(heldout),
        "dataset_tags": list(dataset_tags),
    }


def _validate_payload_contract(payload: Mapping[str, Any], allowlist: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the v5 per-record support budget and TrialNum binding.

    The recovery package validator has already loaded each calibration NWB and
    compared these values with its explicit ``TrialNum`` column.  The
    submission receipt deliberately does not reopen those bytes; it requires
    the resulting immutable per-record evidence to be present and rejects any
    missing, duplicated, padded, or shape-inconsistent support row.
    """

    receipts = payload.get("calibration_receipts")
    if not isinstance(receipts, list) or len(receipts) != 27:
        raise SubmissionReceiptError("recovery payload must contain exactly 27 calibration receipts")
    expected_paths = list(allowlist["files"])
    expected_heldin = {Path(value).resolve() for value in allowlist["heldin"]}
    expected_tags = list(allowlist["dataset_tags"])
    seen_tags: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(receipts):
        if not isinstance(row, Mapping):
            raise SubmissionReceiptError(f"recovery payload calibration receipt {index} is malformed")
        path = str(row.get("path", ""))
        if Path(path).resolve() != Path(expected_paths[index]).resolve():
            raise SubmissionReceiptError(f"recovery payload calibration receipt {index} path drift")
        tag = row.get("dataset_tag")
        if not isinstance(tag, str) or tag != expected_tags[index] or tag in seen_tags:
            raise SubmissionReceiptError(f"recovery payload calibration receipt {index} dataset tag drift")
        seen_tags.add(tag)
        expected_m = 4 if Path(path).resolve() in expected_heldin else 3
        support_m = row.get("support_m")
        if type(support_m) is not int or support_m != expected_m:
            raise SubmissionReceiptError(
                f"recovery payload {tag} support_m must be {expected_m} ({'held-in' if expected_m == 4 else 'held-out'})"
            )
        values = row.get("support_trial_numbers")
        if not isinstance(values, list) or len(values) != expected_m:
            raise SubmissionReceiptError(f"recovery payload {tag} support TrialNum list length drift")
        try:
            numeric_values = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise SubmissionReceiptError(f"recovery payload {tag} support TrialNum values are malformed") from exc
        if len(set(numeric_values)) != expected_m:
            raise SubmissionReceiptError(f"recovery payload {tag} support TrialNum values contain padding/duplicates")
        input_sha = row.get("input_sha256")
        if not isinstance(input_sha, str) or not _SHA256_RE.fullmatch(input_sha):
            raise SubmissionReceiptError(f"recovery payload {tag} input SHA is malformed")
        shape = row.get("identity_shape", [support_m, 1024, 176])
        if shape != [support_m, 1024, 176]:
            raise SubmissionReceiptError(f"recovery payload {tag} identity shape drift: {shape!r}")
        if row.get("carrier_shape", [176, 4]) != [176, 4]:
            raise SubmissionReceiptError(f"recovery payload {tag} carrier shape drift")
        if row.get("padding_or_duplication") not in (None, False):
            raise SubmissionReceiptError(f"recovery payload {tag} records padding/duplication")
        normalized.append({
            "path": path, "dataset_tag": tag, "support_m": support_m,
            "support_trial_numbers": numeric_values, "identity_shape": [support_m, 1024, 176],
            "carrier_shape": [176, 4], "input_sha256": input_sha.lower(),
        })
    if seen_tags != set(expected_tags):
        raise SubmissionReceiptError("recovery payload calibration dataset roster drift")
    identity_shapes = payload.get("identity_shapes")
    if isinstance(identity_shapes, Mapping):
        for row in normalized:
            if identity_shapes.get(row["dataset_tag"]) != row["identity_shape"]:
                raise SubmissionReceiptError(f"recovery payload {row['dataset_tag']} identity_shapes drift")
    if payload.get("shape", {}).get("carrier") not in (None, [176, 4]):
        raise SubmissionReceiptError("recovery payload carrier shape summary drift")
    return {
        "schema": payload.get("schema", PAYLOAD_SCHEMA),
        "dataset_count": 27,
        "calibration_receipts": normalized,
        "shape": {"carrier": [176, 4], "identity": "per_record_m_3_or_4"},
    }


def _validate_stability_receipt() -> dict[str, Any]:
    """Bind the immutable canonical v5_r4 M3-vs-M4 stability receipt."""

    if not STABILITY_AUDIT_PATH.is_file() or STABILITY_AUDIT_PATH.is_symlink():
        raise SubmissionReceiptError(f"canonical stability receipt is missing: {STABILITY_AUDIT_PATH}")
    observed_sha = sha256_file(STABILITY_AUDIT_PATH)
    if observed_sha != STABILITY_AUDIT_SHA256:
        raise SubmissionReceiptError("canonical stability receipt SHA mismatch")
    body = _load_json(STABILITY_AUDIT_PATH)
    if body.get("schema") != STABILITY_SCHEMA or body.get("status") != STABILITY_STATUS:
        raise SubmissionReceiptError("canonical stability receipt schema/status drift")
    if body.get("submission_authorized") is not False or body.get("mixed_m_retraining_indicated") is not False:
        raise SubmissionReceiptError("canonical stability receipt is not submission-denying")
    counts = body.get("counts")
    if counts != {"m_ge_4_files": 13, "heldin": 13, "heldout_m3_only": 14}:
        raise SubmissionReceiptError("canonical stability receipt counts drift")
    allowlist = body.get("calibration_allowlist")
    if not isinstance(allowlist, Mapping) or allowlist.get("counts") != {"heldin": 13, "heldout": 14, "total": 27}:
        raise SubmissionReceiptError("canonical stability receipt allowlist counts drift")
    heldin = allowlist.get("heldin")
    heldout = allowlist.get("heldout")
    files = allowlist.get("files")
    if not isinstance(heldin, list) or len(heldin) != 13 or not isinstance(heldout, list) or len(heldout) != 14:
        raise SubmissionReceiptError("canonical stability receipt allowlist partition drift")
    if files != heldin + heldout or len(set(files or [])) != 27:
        raise SubmissionReceiptError("canonical stability receipt allowlist order/duplicates drift")
    rows = body.get("rows")
    if not isinstance(rows, list) or len(rows) != 13:
        raise SubmissionReceiptError("canonical stability receipt must contain 13 held-in rows")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or Path(str(row.get("path", ""))).resolve() != Path(heldin[index]).resolve():
            raise SubmissionReceiptError(f"canonical stability row {index} path drift")
        for key, expected_m in (("m3", 3), ("m4", 4)):
            evidence = row.get(key)
            if not isinstance(evidence, Mapping) or evidence.get("support_m") != expected_m:
                raise SubmissionReceiptError(f"canonical stability row {index} {key} support_m drift")
            if evidence.get("identity_shape") != [expected_m, 1024, 176]:
                raise SubmissionReceiptError(f"canonical stability row {index} {key} identity shape drift")
            values = evidence.get("support_trial_numbers")
            if not isinstance(values, list) or len(values) != expected_m or len(set(values)) != expected_m:
                raise SubmissionReceiptError(f"canonical stability row {index} {key} TrialNum binding drift")
        if row.get("padding_or_duplication") is not False:
            raise SubmissionReceiptError(f"canonical stability row {index} records padding/duplication")
    scope = body.get("scope")
    if not isinstance(scope, Mapping):
        raise SubmissionReceiptError("canonical stability receipt lacks scope")
    for key, expected in {
        "source_public_calibration_only": True, "query_recordings_opened": 0,
        "formal_test_labels_opened": 0, "target_optimizer_steps": 0,
        "target_backward_steps": 0, "cuda_used": False,
        "evalai_accessed_or_submitted": False,
    }.items():
        if type(scope.get(key)) is not type(expected) or scope.get(key) != expected:
            raise SubmissionReceiptError(f"canonical stability scope {key} drift")
    code_paths = {
        "recovery_package": ROOT / "scripts/h1_carrierid_all_source_official_recovery_package.py",
        "official_package": ROOT / "scripts/h1_carrierid_all_source_official_package.py",
        "runtime": ROOT / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py",
        "carrier_model": ROOT / "src/models/components/h1_carrierid_spint.py",
        "carrier_data": ROOT / "src/data/h1_carrierid_all_source_official.py",
        "carrier_estimator": ROOT / "src/data/h1_m4_eb_pilot.py",
        "deployment_data": ROOT / "src/data/h1_carrierid_all_source_deployment.py",
    }
    code = body.get("code_sha256")
    if not isinstance(code, Mapping) or set(code) != set(code_paths):
        raise SubmissionReceiptError("canonical stability code-SHA closure drift")
    observed_code: dict[str, str] = {}
    for key, path in code_paths.items():
        declared = _require_sha(code.get(key), f"stability code_sha256.{key}")
        actual = sha256_file(path)
        if declared != actual:
            raise SubmissionReceiptError(f"canonical stability code SHA mismatch for {key}")
        observed_code[key] = actual
    return {
        "path": str(STABILITY_AUDIT_PATH.resolve()), "sha256": observed_sha,
        "schema": body["schema"], "status": body["status"],
        "counts": dict(counts), "scope": dict(scope), "code_sha256": observed_code,
    }


def _validate_smoke(
    body: Mapping[str, Any], *, payload_sha256: str, allowlist: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Require the CPU runtime smoke that covers all 13+14 calibration files."""

    smoke = body.get("smoke")
    if not isinstance(smoke, Mapping):
        raise SubmissionReceiptError("recovery package final audit must contain a runtime smoke block")
    if smoke.get("schema") != RECOVERY_SMOKE_SCHEMA:
        raise SubmissionReceiptError("recovery smoke schema drift")
    if smoke.get("status") != RECOVERY_SMOKE_STATUS:
        raise SubmissionReceiptError("recovery smoke is not a passing CPU smoke")
    counts = smoke.get("counts")
    if (
        not isinstance(counts, Mapping)
        or any(type(counts.get(key)) is not int for key in ("heldin", "heldout", "total"))
        or dict(counts) != {"heldin": 13, "heldout": 14, "total": 27}
    ):
        raise SubmissionReceiptError("recovery smoke counts must be exactly 13+14=27")
    rows = smoke.get("rows")
    if not isinstance(rows, list) or len(rows) != 27:
        raise SubmissionReceiptError("recovery smoke must contain exactly 27 rows")
    split_counts = {"heldin": 0, "heldout": 0}
    expected_paths: list[tuple[str, str]] = []
    if allowlist and allowlist.get("files"):
        expected_paths = [("heldin", value) for value in allowlist["heldin"]] + [
            ("heldout", value) for value in allowlist["heldout"]
        ]
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SubmissionReceiptError(f"recovery smoke row {index} is malformed")
        split = row.get("split")
        if split not in split_counts:
            raise SubmissionReceiptError(f"recovery smoke row {index} has an invalid split")
        split_counts[str(split)] += 1
        if expected_paths:
            expected_split, expected_path = expected_paths[index]
            if split != expected_split or Path(str(row.get("path", ""))).resolve() != Path(expected_path).resolve():
                raise SubmissionReceiptError(f"recovery smoke row {index} path/split drift")
        if row.get("output_shape") != [1, 7] or row.get("finite") is not True:
            raise SubmissionReceiptError(f"recovery smoke row {index} output invariant drift")
    if split_counts != {"heldin": 13, "heldout": 14}:
        raise SubmissionReceiptError("recovery smoke rows do not preserve 13 held-in + 14 held-out files")
    smoke_scope = smoke.get("scope")
    if not isinstance(smoke_scope, Mapping):
        raise SubmissionReceiptError("recovery smoke lacks scope mapping")
    for key, expected in {
        "query_recordings_opened": 0,
        "formal_test_labels_opened": 0,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "cuda_used": False,
        "evalai_accessed_or_submitted": False,
    }.items():
        expected_type = bool if isinstance(expected, bool) else int
        if type(smoke_scope.get(key)) is not expected_type or smoke_scope.get(key) != expected:
            raise SubmissionReceiptError(f"recovery smoke scope {key} must be {expected!r}")
    if smoke.get("submission_authorized") is not False:
        raise SubmissionReceiptError("recovery smoke authorizes submission")
    smoke_payload = smoke.get("payload")
    if not isinstance(smoke_payload, Mapping) or smoke_payload.get("sha256") != payload_sha256:
        raise SubmissionReceiptError("recovery smoke/payload SHA mismatch")
    return {
        "schema": smoke["schema"],
        "status": smoke["status"],
        "counts": dict(counts),
        "rows": [dict(row) for row in rows],
        "scope": dict(smoke_scope),
        "submission_authorized": False,
        "payload": dict(smoke_payload),
    }


def _validate_terminal(body: Mapping[str, Any]) -> dict[str, Any]:
    terminal = body.get("terminal_checkpoint")
    if not isinstance(terminal, Mapping):
        # A small number of early final-audit fixtures carried terminal data
        # only in the nested recovery audit.  The provenance/zero-target gates
        # above remain mandatory; do not reject such fixtures solely for this
        # informational block.
        return {}
    epoch = terminal.get("epoch", terminal.get("checkpoint_epoch_zero_based"))
    if epoch is not None and epoch != EXPECTED_EPOCH:
        raise SubmissionReceiptError("recovery terminal checkpoint must be fixed epoch 49")
    completed = terminal.get("epochs_completed")
    if completed is not None and completed != EXPECTED_EPOCH + 1:
        raise SubmissionReceiptError("recovery terminal checkpoint must report 50 completed epochs")
    return dict(terminal)


def _validate_code_provenance(body: Mapping[str, Any], *, audit_path: Path) -> dict[str, Any]:
    """Re-hash the v5 wrapper/official code and its sealed package preflight."""

    final_code = body.get("code_sha256")
    if not isinstance(final_code, Mapping):
        raise SubmissionReceiptError("recovery package final audit lacks code_sha256 mapping")
    final_paths = {
        "recovery_package": ROOT / "scripts/h1_carrierid_all_source_official_recovery_package.py",
        "official_package": ROOT / "scripts/h1_carrierid_all_source_official_package.py",
        "runtime": ROOT / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py",
        "carrier_model": ROOT / "src/models/components/h1_carrierid_spint.py",
        "carrier_data": ROOT / "src/data/h1_carrierid_all_source_official.py",
        "carrier_estimator": ROOT / "src/data/h1_m4_eb_pilot.py",
        "deployment_data": ROOT / "src/data/h1_carrierid_all_source_deployment.py",
    }
    if set(final_code) != set(final_paths):
        raise SubmissionReceiptError("recovery final-audit code-SHA closure drift")
    final_observed: dict[str, str] = {}
    for key, source_path in final_paths.items():
        declared = _require_sha(final_code.get(key), f"code_sha256.{key}")
        actual = sha256_file(source_path)
        if declared != actual:
            raise SubmissionReceiptError(f"recovery code SHA mismatch for {key}")
        final_observed[key] = actual

    preflight = body.get("package_preflight")
    if not isinstance(preflight, Mapping):
        raise SubmissionReceiptError("recovery package final audit lacks package_preflight binding")
    preflight_path = _optional_path(preflight, audit_path, ("path", "preflight_path"), "package preflight path")
    preflight_sha = _optional_sha(preflight, ("sha256", "preflight_sha256"), "package preflight SHA")
    if preflight_path is None or preflight_sha is None:
        raise SubmissionReceiptError("package preflight path/SHA must be supplied together")
    observed_preflight_sha = sha256_file(preflight_path)
    if observed_preflight_sha != preflight_sha:
        raise SubmissionReceiptError("package preflight SHA mismatch")
    preflight_body = _load_json(preflight_path)
    if preflight_body.get("schema") != PACKAGE_PREFLIGHT_SCHEMA:
        raise SubmissionReceiptError("package preflight schema drift")
    if preflight_body.get("status") != PACKAGE_PREFLIGHT_STATUS:
        raise SubmissionReceiptError("package preflight status drift")
    preflight_scope = preflight_body.get("scope")
    if not isinstance(preflight_scope, Mapping):
        raise SubmissionReceiptError("package preflight lacks scope mapping")
    for key, expected in {
        "calibration_recording_bytes_opened": 0,
        "query_recordings_opened": 0,
        "formal_test_labels_opened": 0,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "payload_exported": False,
        "docker_built_or_pushed": False,
        "evalai_accessed_or_submitted": False,
    }.items():
        expected_type = bool if isinstance(expected, bool) else int
        if type(preflight_scope.get(key)) is not expected_type or preflight_scope.get(key) != expected:
            raise SubmissionReceiptError(f"package preflight scope {key} must be {expected!r}")
    preflight_code = preflight_body.get("code_sha256")
    if not isinstance(preflight_code, Mapping):
        raise SubmissionReceiptError("package preflight lacks code_sha256 mapping")
    preflight_paths = {
        "package": ROOT / "scripts/h1_carrierid_all_source_official_package.py",
        "runtime": ROOT / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py",
        "carrier_model": ROOT / "src/models/components/h1_carrierid_spint.py",
        "carrier_data": ROOT / "src/data/h1_carrierid_all_source_official.py",
        "carrier_estimator": ROOT / "src/data/h1_m4_eb_pilot.py",
        "deployment_data": ROOT / "src/data/h1_carrierid_all_source_deployment.py",
    }
    if set(preflight_code) != set(preflight_paths):
        raise SubmissionReceiptError("package preflight code-SHA closure drift")
    preflight_observed: dict[str, str] = {}
    for key, source_path in preflight_paths.items():
        declared = _require_sha(preflight_code.get(key), f"package_preflight.code_sha256.{key}")
        actual = sha256_file(source_path)
        if declared != actual:
            raise SubmissionReceiptError(f"package preflight code SHA mismatch for {key}")
        preflight_observed[key] = actual
    return {
        "path": str(preflight_path),
        "sha256": observed_preflight_sha,
        "schema": preflight_body["schema"],
        "code_sha256": {"final_audit": final_observed, "preflight": preflight_observed},
    }


def validate_recovery_package(
    audit_path: str | Path,
    *,
    expected_audit_sha256: str | None = None,
    payload_path: str | Path | None = None,
    expected_payload_path: str | Path | None = None,
    expected_payload_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and hash one immutable recovery-package final audit.

    The audit must carry a payload path/SHA and explicit unbound-launch flags.
    ``payload_path``/expected hashes are optional overrides for callers that
    receive the path separately; when supplied they must agree with the audit.
    No pickle loading occurs, so this gate remains safe before a Docker build.
    """

    if payload_path is not None and expected_payload_path is not None and Path(payload_path).resolve() != Path(expected_payload_path).resolve():
        raise SubmissionReceiptError("payload_path and expected_payload_path disagree")
    if payload_path is None:
        payload_path = expected_payload_path
    audit_original = Path(audit_path)
    if audit_original.is_symlink():
        raise SubmissionReceiptError("recovery package final audit must not be a symlink")
    path = audit_original.resolve()
    audit_sha = sha256_file(path)
    declared_audit = _require_sha(expected_audit_sha256, "expected_audit_sha256") if expected_audit_sha256 else None
    if declared_audit is not None and declared_audit != audit_sha:
        raise SubmissionReceiptError("recovery package final audit SHA mismatch")
    body = _load_json(path)
    if body.get("schema") not in RECOVERY_PACKAGE_SCHEMAS:
        raise SubmissionReceiptError(f"unsupported recovery package audit schema: {body.get('schema')!r}")
    if body.get("status") not in RECOVERY_PACKAGE_PASS_STATUSES:
        raise SubmissionReceiptError(f"recovery package audit is not a passing final audit: {body.get('status')!r}")
    gap = _validate_provenance_gap(body)
    scope = _validate_scope(body)
    allowlist = _validate_allowlist(body)
    terminal = _validate_terminal(body)
    code_provenance = _validate_code_provenance(body, audit_path=path)
    submission = body.get("submission")
    if not isinstance(submission, Mapping) or submission.get("authorized") is not False:
        raise SubmissionReceiptError("recovery package audit must explicitly deny submission authorization")

    payload_block = body.get("payload")
    if not isinstance(payload_block, Mapping):
        raise SubmissionReceiptError("recovery package final audit has no exported payload evidence")
    nested_payload_path = _optional_path(payload_block, path, ("path", "output", "payload_path"), "payload path")
    declared_payload_sha = _optional_sha(payload_block, ("sha256", "output_sha256", "payload_sha256"), "payload SHA")
    if nested_payload_path is None:
        if payload_path is not None:
            override_path = Path(payload_path)
            if override_path.is_symlink():
                raise SubmissionReceiptError("payload path override must not be a symlink")
            nested_payload_path = override_path.resolve()
        else:
            nested_payload_path = None
    if nested_payload_path is None:
        raise SubmissionReceiptError("recovery package payload path is missing")
    if payload_path is not None:
        override_path = Path(payload_path)
        if override_path.is_symlink() or nested_payload_path != override_path.resolve():
            raise SubmissionReceiptError("payload path override does not match recovery package audit")
    observed_payload_sha = sha256_file(nested_payload_path)
    if declared_payload_sha is None:
        raise SubmissionReceiptError("recovery package payload SHA is missing")
    if declared_payload_sha != observed_payload_sha:
        raise SubmissionReceiptError("recovery package payload SHA mismatch")
    supplied_payload_sha = (
        _require_sha(expected_payload_sha256, "expected_payload_sha256") if expected_payload_sha256 else None
    )
    if supplied_payload_sha is not None and supplied_payload_sha != observed_payload_sha:
        raise SubmissionReceiptError("payload SHA override does not match recovery package payload")
    payload_schema = payload_block.get("schema")
    if payload_schema != PAYLOAD_SCHEMA:
        raise SubmissionReceiptError(f"recovery payload schema drift: {payload_schema!r}")
    payload_contract = _validate_payload_contract(payload_block, allowlist)
    smoke = _validate_smoke(body, payload_sha256=observed_payload_sha, allowlist=allowlist)
    stability = _validate_stability_receipt()

    # Bind the root recovery audit when present.  Its absence is tolerated for
    # old package fixtures, but a present path/SHA pair is always re-hashed.
    root_block = body.get("recovery_audit")
    root_evidence: dict[str, Any] = {}
    if isinstance(root_block, Mapping):
        root_path = _optional_path(root_block, path, ("path", "audit_path"), "root recovery audit path")
        root_sha = _optional_sha(root_block, ("sha256", "audit_sha256"), "root recovery audit SHA")
        if root_path is not None or root_sha is not None:
            if root_path is None or root_sha is None:
                raise SubmissionReceiptError("root recovery audit path/SHA must be supplied together")
            observed_root_sha = sha256_file(root_path)
            if observed_root_sha != root_sha:
                raise SubmissionReceiptError("root recovery audit SHA mismatch")
            root_evidence = {"path": str(root_path), "sha256": observed_root_sha}

    return {
        "path": str(path),
        "sha256": audit_sha,
        "schema": str(body["schema"]),
        "status": str(body["status"]),
        "payload": {"path": str(nested_payload_path), "sha256": observed_payload_sha, "schema": payload_schema or PAYLOAD_SCHEMA},
        "payload_contract": payload_contract,
        "stability_audit": stability,
        "root_recovery_audit": root_evidence,
        "provenance_gap": gap,
        "scope": scope,
        "calibration_allowlist": allowlist,
        "smoke": smoke,
        "terminal_checkpoint": terminal,
        "code_provenance": code_provenance,
        "recovery_submission": dict(submission),
        "body": body,
    }


def _validate_image(image_id: Any, image_tag: Any, embedded_decoder_sha256: Any, image_digest: Any = None) -> dict[str, Any]:
    image_norm = _require_sha(image_id, "image_id")
    embedded_norm = _require_sha(embedded_decoder_sha256, "embedded_decoder_sha256")
    if not isinstance(image_tag, str) or not _IMAGE_TAG_RE.fullmatch(image_tag) or "h1" not in image_tag.lower():
        raise SubmissionReceiptError("image_tag must be a non-empty H1 Docker tag")
    result: dict[str, Any] = {"id": f"sha256:{image_norm}", "tag": image_tag, "embedded_decoder_sha256": embedded_norm}
    if image_digest is not None:
        result["digest"] = f"sha256:{_require_sha(image_digest, 'image_digest')}"
    return result


def _validate_runtime(task: str, batch_size: int, model_file: str) -> dict[str, Any]:
    if task != EXPECTED_TASK:
        raise SubmissionReceiptError(f"EvalAI runtime TASK must be {EXPECTED_TASK!r}, got {task!r}")
    if int(batch_size) != EXPECTED_BATCH_SIZE:
        raise SubmissionReceiptError(f"EvalAI H1 BATCH_SIZE must be {EXPECTED_BATCH_SIZE}, got {batch_size!r}")
    if model_file not in EXPECTED_MODEL_FILE_ALIASES:
        raise SubmissionReceiptError(
            "EvalAI MODEL_FILE must identify the all-source recovery payload, "
            f"got {model_file!r}"
        )
    return {"TASK": EXPECTED_TASK, "BATCH_SIZE": EXPECTED_BATCH_SIZE, "PHASE": "test", "MODEL_FILE": model_file}


def _validate_challenge(challenge_id: int, phase_id: int, team_id: int, phase_slug: str) -> dict[str, Any]:
    if int(challenge_id) != EXPECTED_CHALLENGE_ID:
        raise SubmissionReceiptError(f"challenge_id must be {EXPECTED_CHALLENGE_ID}")
    if int(phase_id) != EXPECTED_PHASE_ID:
        raise SubmissionReceiptError(f"phase_id must be {EXPECTED_PHASE_ID}")
    if int(team_id) != EXPECTED_TEAM_ID:
        raise SubmissionReceiptError(f"team_id must be {EXPECTED_TEAM_ID}")
    if phase_slug != EXPECTED_PHASE_SLUG:
        raise SubmissionReceiptError(f"phase_slug must be {EXPECTED_PHASE_SLUG!r}")
    return {
        "challenge_id": EXPECTED_CHALLENGE_ID,
        "phase_id": EXPECTED_PHASE_ID,
        "phase_slug": EXPECTED_PHASE_SLUG,
        "team_id": EXPECTED_TEAM_ID,
        "visibility": "private",
    }


def _validate_quota(quota: Mapping[str, Any] | None) -> dict[str, Any]:
    if quota is None:
        return {"source": "not_queried_offline", "bound": False}
    if not isinstance(quota, Mapping):
        raise SubmissionReceiptError("quota must be a JSON object")
    result = {str(key): value for key, value in quota.items()}
    # Numeric quota counters are validated when present; unknown API fields
    # are preserved for forward-compatible auditing.
    for key in ("today", "month", "total", "active", "max_per_day", "max_per_month", "max_total", "max_concurrent"):
        if key in result and (type(result[key]) is not int or result[key] < 0):
            raise SubmissionReceiptError(f"quota.{key} must be a non-negative integer")
    result.setdefault("bound", True)
    return result


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    original = Path(path)
    output = original.resolve()
    if original.is_symlink() or output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable receipt {original}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Publish by hard-linking the already-fsynced temporary inode.  The
        # link call fails if a competing writer wins the race and can
        # therefore never overwrite an existing receipt.
        os.link(temporary, output)
        output.chmod(0o444)
        os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _utc_now() -> str:
    value = _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds")
    return value[:-6] + "Z" if value.endswith("+00:00") else value


def prepare_receipt(
    *,
    recovery_package_audit: str | Path | None = None,
    recovery_audit_path: str | Path | None = None,
    output: str | Path,
    image_id: str,
    embedded_decoder_sha256: str | None = None,
    decoder_sha256: str | None = None,
    image_tag: str,
    image_digest: str | None = None,
    expected_audit_sha256: str | None = None,
    expected_payload_sha256: str | None = None,
    payload_path: str | Path | None = None,
    task: str = EXPECTED_TASK,
    batch_size: int = EXPECTED_BATCH_SIZE,
    model_file: str = EXPECTED_MODEL_FILE,
    challenge_id: int = EXPECTED_CHALLENGE_ID,
    phase_id: int = EXPECTED_PHASE_ID,
    team_id: int = EXPECTED_TEAM_ID,
    phase_slug: str = EXPECTED_PHASE_SLUG,
    quota: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate recovery bytes and write one immutable offline preparation receipt."""

    audit_value = recovery_package_audit or recovery_audit_path
    if audit_value is None:
        raise SubmissionReceiptError("recovery package final audit path is required")
    embedded_value = embedded_decoder_sha256 or decoder_sha256
    if embedded_value is None:
        raise SubmissionReceiptError("Docker embedded decoder SHA-256 is required")
    recovery = validate_recovery_package(
        audit_value,
        expected_audit_sha256=expected_audit_sha256,
        payload_path=payload_path,
        expected_payload_sha256=expected_payload_sha256,
    )
    if _require_sha(embedded_value, "embedded_decoder_sha256") != recovery["payload"]["sha256"]:
        raise SubmissionReceiptError("Docker embedded decoder SHA does not match recovery payload SHA")
    image = _validate_image(image_id, image_tag, embedded_value, image_digest)
    runtime = _validate_runtime(task, batch_size, model_file)
    challenge = _validate_challenge(challenge_id, phase_id, team_id, phase_slug)
    quota_value = _validate_quota(quota)
    prepared = {
        "schema": PREPARE_SCHEMA,
        "status": "PASS_H1_ALL_SOURCE_SUBMISSION_PREPARED_OFFLINE",
        "created_at": _utc_now(),
        "method_name": METHOD_NAME,
        "method_description": METHOD_DESCRIPTION,
        "submission_attributes": SUBMISSION_ATTRIBUTES,
        "submission_metadata": SUBMISSION_ATTRIBUTES,
        "recovery_package_final_audit": {"path": recovery["path"], "sha256": recovery["sha256"], "schema": recovery["schema"]},
        "recovery_payload": recovery["payload"],
        "payload_contract": recovery["payload_contract"],
        "stability_audit": recovery["stability_audit"],
        "root_recovery_audit": recovery["root_recovery_audit"],
        "code_provenance": recovery["code_provenance"],
        "recovery_submission": recovery["recovery_submission"],
        "provenance_gap": recovery["provenance_gap"],
        "scope": recovery["scope"],
        "calibration_allowlist": recovery["calibration_allowlist"],
        "smoke": recovery["smoke"],
        "terminal_checkpoint": recovery["terminal_checkpoint"],
        "image": image,
        "runtime": runtime,
        "challenge": challenge,
        "quota": quota_value,
        "external_action": {
            "evalai_push_performed_by_this_tool": False,
            "docker_build_performed_by_this_tool": False,
            "private_labels_read_by_this_tool": False,
            "evalai_accessed_or_submitted_by_this_tool": False,
        },
        "submission": None,
    }
    _write_immutable_json(output, prepared)
    return prepared


def _parse_timestamp(value: str) -> str:
    if not isinstance(value, str) or not _ISO_RE.fullmatch(value):
        raise SubmissionReceiptError("timestamp must be ISO-8601 with timezone")
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        _datetime.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SubmissionReceiptError(f"invalid timestamp {value!r}") from exc
    return value


def _reload_preparation(path: str | Path) -> dict[str, Any]:
    preparation_path = Path(path).resolve()
    preparation = _load_json(preparation_path)
    if preparation.get("schema") != PREPARE_SCHEMA:
        raise SubmissionReceiptError("finalize requires an all-source offline preparation receipt")
    if preparation.get("status") != "PASS_H1_ALL_SOURCE_SUBMISSION_PREPARED_OFFLINE":
        raise SubmissionReceiptError("preparation receipt is not a passing offline receipt")
    if preparation.get("submission") is not None:
        raise SubmissionReceiptError("preparation receipt already contains a submission")
    gap = preparation.get("provenance_gap")
    _validate_provenance_gap({"provenance_gap": gap})
    external = preparation.get("external_action")
    if not isinstance(external, Mapping) or external.get("evalai_push_performed_by_this_tool") is not False:
        raise SubmissionReceiptError("preparation receipt claims this tool performed EvalAI push")
    audit = preparation.get("recovery_package_final_audit")
    payload = preparation.get("recovery_payload")
    if not isinstance(audit, Mapping) or not isinstance(payload, Mapping):
        raise SubmissionReceiptError("preparation receipt lacks recovery audit/payload bindings")
    audit_raw = Path(str(audit.get("path", "")))
    payload_raw = Path(str(payload.get("path", "")))
    if audit_raw.is_symlink() or payload_raw.is_symlink():
        raise SubmissionReceiptError("preparation recovery bindings must not use symlinks")
    audit_path = audit_raw.resolve()
    payload_path = payload_raw.resolve()
    checked = validate_recovery_package(
        audit_path,
        expected_audit_sha256=audit.get("sha256"),
        payload_path=payload_path,
        expected_payload_sha256=payload.get("sha256"),
    )
    if checked["sha256"] != audit.get("sha256") or checked["payload"]["sha256"] != payload.get("sha256"):
        raise SubmissionReceiptError("preparation recovery binding changed after prepare")
    if preparation.get("code_provenance") != checked.get("code_provenance"):
        raise SubmissionReceiptError("preparation code provenance binding changed")
    embedded = preparation.get("image", {}).get("embedded_decoder_sha256") if isinstance(preparation.get("image"), Mapping) else None
    if _require_sha(embedded, "embedded_decoder_sha256") != checked["payload"]["sha256"]:
        raise SubmissionReceiptError("preparation embedded decoder binding changed")
    return preparation


def finalize_receipt(
    *,
    preparation_receipt: str | Path,
    output: str | Path,
    submission_id: int | str,
    timestamp: str,
    status: str,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an externally supplied EvalAI result; never contacts EvalAI."""

    preparation = _reload_preparation(preparation_receipt)
    if isinstance(submission_id, bool) or not str(submission_id).strip() or not str(submission_id).isdigit():
        raise SubmissionReceiptError("submission_id must contain decimal digits only")
    if not isinstance(status, str) or not status.strip():
        raise SubmissionReceiptError("submission status must be non-empty")
    metric_map = dict(metrics or {})
    status_norm = status.strip().lower()
    if status_norm in {"finished", "complete", "completed", "success", "succeeded"} and not metric_map:
        raise SubmissionReceiptError("a successful terminal submission must record at least one metric field")
    for name, value in metric_map.items():
        if not isinstance(name, str) or not name.strip() or isinstance(value, (dict, list, tuple)):
            raise SubmissionReceiptError("metric names must be non-empty and metric values scalar")
    final = {
        "schema": FINAL_SCHEMA,
        "status": "RECORDED_EXTERNAL_H1_ALL_SOURCE_EVALAI_RESULT",
        "created_at": _utc_now(),
        "preparation_receipt_path": str(Path(preparation_receipt).resolve()),
        "preparation_receipt_sha256": sha256_file(preparation_receipt),
        "method_name": preparation["method_name"],
        "method_description": preparation["method_description"],
        "submission_attributes": preparation["submission_attributes"],
        "submission_metadata": preparation["submission_attributes"],
        "recovery_package_final_audit": preparation["recovery_package_final_audit"],
        "recovery_payload": preparation["recovery_payload"],
        "payload_contract": preparation.get("payload_contract", {}),
        "stability_audit": preparation.get("stability_audit", {}),
        "root_recovery_audit": preparation.get("root_recovery_audit", {}),
        "code_provenance": preparation["code_provenance"],
        "recovery_submission": preparation["recovery_submission"],
        "provenance_gap": preparation["provenance_gap"],
        "scope": preparation["scope"],
        "calibration_allowlist": preparation["calibration_allowlist"],
        "smoke": preparation["smoke"],
        "image": preparation["image"],
        "runtime": preparation["runtime"],
        "challenge": preparation["challenge"],
        "quota": preparation["quota"],
        "external_action": {
            **dict(preparation["external_action"]),
            "evalai_push_performed_by_this_tool": False,
            "submission_recorded_from_external_api_response": True,
        },
        "submission": {
            "id": int(str(submission_id)),
            "timestamp": _parse_timestamp(timestamp),
            "status": status,
            "metrics": metric_map,
            "metric_fields_present": sorted(metric_map),
        },
    }
    _write_immutable_json(output, final)
    return final


# Descriptive aliases used by operator notebooks and future integration code.
# Keep ``prepare_receipt``/``finalize_receipt`` as the short CLI-compatible API.
validate_recovery_package_audit = validate_recovery_package
prepare_submission_receipt = prepare_receipt
finalize_submission_receipt = finalize_receipt


def _parse_metric_items(items: Sequence[str], inline_json: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if inline_json is not None:
        try:
            parsed = json.loads(inline_json)
        except json.JSONDecodeError as exc:
            raise SubmissionReceiptError(f"--metrics-json must be an inline JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SubmissionReceiptError("--metrics-json must be an inline JSON object")
        result.update(parsed)
    for item in items:
        if "=" not in item:
            raise SubmissionReceiptError(f"metric must use NAME=VALUE syntax: {item!r}")
        name, raw = item.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise SubmissionReceiptError(f"duplicate or empty metric name: {name!r}")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if isinstance(value, (dict, list)):
            raise SubmissionReceiptError("metric values must be scalar JSON values")
        result[name] = value
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline H1 all-source recovery submission receipts.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="validate recovery audit/payload and write immutable receipt")
    prepare.add_argument("--recovery-package-audit", "--recovery-audit", dest="recovery_package_audit", required=True)
    prepare.add_argument("--payload-path")
    prepare.add_argument("--output", "--receipt", dest="output", required=True)
    prepare.add_argument("--image-id", required=True)
    prepare.add_argument("--embedded-decoder-sha256", "--decoder-sha256", dest="embedded_decoder_sha256", required=True)
    prepare.add_argument("--image-tag", required=True)
    prepare.add_argument("--image-digest")
    prepare.add_argument("--audit-sha256", dest="expected_audit_sha256")
    prepare.add_argument("--payload-sha256", dest="expected_payload_sha256")
    prepare.add_argument("--quota-json")
    prepare.add_argument("--task", default=EXPECTED_TASK)
    prepare.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    prepare.add_argument("--model-file", default=EXPECTED_MODEL_FILE)
    prepare.add_argument("--challenge-id", type=int, default=EXPECTED_CHALLENGE_ID)
    prepare.add_argument("--phase-id", type=int, default=EXPECTED_PHASE_ID)
    prepare.add_argument("--team-id", type=int, default=EXPECTED_TEAM_ID)
    prepare.add_argument("--phase-slug", default=EXPECTED_PHASE_SLUG)
    finalize = sub.add_parser("finalize", help="record externally supplied EvalAI result")
    finalize.add_argument("--preparation-receipt", required=True)
    finalize.add_argument("--output", "--receipt", dest="output", required=True)
    finalize.add_argument("--submission-id", required=True)
    finalize.add_argument("--timestamp", required=True)
    finalize.add_argument("--status", required=True)
    finalize.add_argument("--metric", action="append", default=[])
    finalize.add_argument("--metrics-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        quota: Mapping[str, Any] | None = None
        if args.quota_json is not None:
            try:
                parsed = json.loads(args.quota_json)
            except json.JSONDecodeError as exc:
                raise SubmissionReceiptError(f"--quota-json must be inline JSON: {exc}") from exc
            if not isinstance(parsed, Mapping):
                raise SubmissionReceiptError("--quota-json must be a JSON object")
            quota = parsed
        result = prepare_receipt(
            recovery_package_audit=args.recovery_package_audit,
            payload_path=args.payload_path,
            output=args.output,
            image_id=args.image_id,
            embedded_decoder_sha256=args.embedded_decoder_sha256,
            image_tag=args.image_tag,
            image_digest=args.image_digest,
            expected_audit_sha256=args.expected_audit_sha256,
            expected_payload_sha256=args.expected_payload_sha256,
            task=args.task,
            batch_size=args.batch_size,
            model_file=args.model_file,
            challenge_id=args.challenge_id,
            phase_id=args.phase_id,
            team_id=args.team_id,
            phase_slug=args.phase_slug,
            quota=quota,
        )
    else:
        result = finalize_receipt(
            preparation_receipt=args.preparation_receipt,
            output=args.output,
            submission_id=args.submission_id,
            timestamp=args.timestamp,
            status=args.status,
            metrics=_parse_metric_items(args.metric, args.metrics_json),
        )
    print(json.dumps({"schema": result["schema"], "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
