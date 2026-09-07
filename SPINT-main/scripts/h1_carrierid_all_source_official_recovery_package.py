#!/usr/bin/env python3
"""Fail-closed package finalization for the unbound H1 all-source run.

The source training run has an immutable recovery receipt, but it does not
have the nonce-bound execution receipt required by the official post-training
watcher.  This module provides a separate recovery namespace for package
preflight, optional payload validation, CPU-only runtime smoke, and a final
audit.  It never repairs the missing nonce and never writes the official
watcher's ``PREFLIGHT/PAYLOAD/SMOKE/AUDIT`` paths.

The normal command used during recovery is ``preflight``.  It only indexes the
exact 13 held-in and 14 held-out calibration paths through the existing
package preflight helper; it does not open calibration NWB bytes and does not
write a payload.  Payload export is an explicitly gated operation for a later
human-authorized step, and submission is never authorized by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Contract modules import torch transitively.  Refuse a CLI invocation before
# those imports unless the process was started with CUDA explicitly hidden.
if __name__ == "__main__" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit(
        "recovery package requires CUDA_VISIBLE_DEVICES='' from process start; refusing before runtime imports"
    )

from falcon_challenge.config import FalconConfig, FalconTask
from src.data.h1_carrierid_all_source_official import (
    assert_deployment_calibration_path,
    fit_frozen_carrier,
    interpolate_trial_identity,
    load_all_source_assets,
)
from src.data.h1_carrierid_all_source_deployment import load_deployment_calibration_record_v5
from src.data.h1_m4_eb_pilot import fit_deployment_carrier
from src.data.h1_m4_eb_pilot import PilotDataError
from src.h1_m4_cce_contract import sha256_file, write_immutable_json
from scripts import h1_carrierid_all_source_official_package as official_package
from scripts import h1_carrierid_all_source_official_recovery_audit as recovery_audit
from third_party.falcon_challenge.h1_carrierid_all_source_decoder import (
    H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5,
    H1CarrierIdPayloadError,
    validate_carrier_payload,
)


ART = ROOT / "pilot_artifacts/h1_carrierid_all_source_official_v1"
RECOVERY_ROOT = ART / "recovery_package_v1"
RECOVERY_AUDIT = ART / "H1_CARRIERID_ALL_SOURCE_RECOVERY_AUDIT_v1.json"
DEFAULT_PREFLIGHT = RECOVERY_ROOT / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PREFLIGHT_v5.json"
DEFAULT_PREFLIGHT_AUDIT = RECOVERY_ROOT / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PREFLIGHT_AUDIT_v5.json"
DEFAULT_PAYLOAD = RECOVERY_ROOT / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD_v5.pkl"
DEFAULT_SMOKE = RECOVERY_ROOT / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_SMOKE_v5.json"
DEFAULT_AUDIT = RECOVERY_ROOT / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_AUDIT_v5.json"
DEFAULT_STABILITY_AUDIT = RECOVERY_ROOT / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_M3_STABILITY_AUDIT_v5.json"

RECOVERY_SCHEMA = "h1_carrierid_all_public_source_recovery_package_audit_v5"
RECOVERY_PREFLIGHT_SCHEMA = "h1_carrierid_all_public_source_recovery_package_preflight_v5"
RECOVERY_SMOKE_SCHEMA = "h1_carrierid_all_public_source_recovery_package_smoke_v5"
RECOVERY_STABILITY_SCHEMA = "h1_carrierid_all_public_source_recovery_package_m3_stability_audit_v5"
H1_ALL_SOURCE_PAYLOAD_SCHEMA = H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5
# Explicit alias retained for contract fixtures; all v5 paths call the
# versioned deployment loader directly.
load_deployment_calibration_record = load_deployment_calibration_record_v5
RECOVERY_PREFLIGHT_STATUS = "PASS_RECOVERY_PACKAGE_PREFLIGHT_ONLY_WITH_PROVENANCE_GAP"
RECOVERY_STATUS = "PASS_RECOVERY_PACKAGE_WITH_PROVENANCE_GAP_NO_SUBMISSION"
RECOVERY_PAYLOAD_STATUS = "PASS_RECOVERY_PAYLOAD_VALIDATED_WITH_PROVENANCE_GAP"

# These are deliberately copied as a closed roster rather than discovered by
# globbing.  The package may never silently absorb a new file from data/.
HELDIN_SESSIONS: tuple[str, ...] = (
    "ses-19250101T111740", "ses-19250101T112404",
    "ses-19250108T110520", "ses-19250108T111022", "ses-19250108T111455",
    "ses-19250113T120811", "ses-19250113T121303",
    "ses-19250115T110633", "ses-19250115T111328",
    "ses-19250119T113543", "ses-19250119T114045",
    "ses-19250120T115044", "ses-19250120T115537",
)
HELDOUT_SESSIONS: tuple[str, ...] = (
    "ses-19250126T113454", "ses-19250126T114029",
    "ses-19250127T120333", "ses-19250127T120826",
    "ses-19250129T112555", "ses-19250129T113059",
    "ses-19250202T113958", "ses-19250202T114452",
    "ses-19250203T113515", "ses-19250203T114018",
    "ses-19250206T112219", "ses-19250206T112712",
    "ses-19250209T111826", "ses-19250209T112327",
)


class RecoveryPackageError(ValueError):
    """A recovery package boundary or immutable provenance gate failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryPackageError(message)


def _require_cuda_hidden() -> None:
    _need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "recovery package requires CUDA_VISIBLE_DEVICES='' before importing/using runtime modules",
    )


def _immutable(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444


def _sha256_file(path: str | Path) -> str:
    return sha256_file(Path(path))


def _recovery_package_code_paths() -> dict[str, Path]:
    """Return the complete v5 recovery/package code-hash closure.

    The final recovery audit is consumed independently from the package
    preflight and stability receipt, so it must bind the deployment-only
    loader as well as the historical estimator and runtime modules.  Keep the
    path list in one helper so the emitted receipt and its validator cannot
    silently diverge.
    """

    return {
        "recovery_package": Path(__file__).resolve(),
        "official_package": Path(official_package.__file__).resolve(),
        "runtime": (ROOT / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py").resolve(),
        "carrier_model": (ROOT / "src/models/components/h1_carrierid_spint.py").resolve(),
        "carrier_data": (ROOT / "src/data/h1_carrierid_all_source_official.py").resolve(),
        "carrier_estimator": (ROOT / "src/data/h1_m4_eb_pilot.py").resolve(),
        "deployment_data": (ROOT / "src/data/h1_carrierid_all_source_deployment.py").resolve(),
    }


def _current_recovery_package_code_sha256() -> dict[str, str]:
    """Hash every source path bound by the final v5 recovery audit."""

    paths = _recovery_package_code_paths()
    for key, path in paths.items():
        _need(path.is_file() and not path.is_symlink(), f"recovery final-audit code path is missing: {key}")
    return {key: _sha256_file(path) for key, path in paths.items()}


def validate_recovery_package_audit_code(body: Mapping[str, Any]) -> dict[str, str]:
    """Validate final-audit code hashes, including deployment-only loader code.

    This is intentionally separate from the immutable v1 recovery audit
    validator: the v5 wrapper is append-only and its code closure includes the
    new variable-budget deployment module.
    """

    code = body.get("code_sha256")
    _need(isinstance(code, Mapping), "recovery final audit code-SHA binding is malformed")
    expected = _current_recovery_package_code_sha256()
    _need(set(code) == set(expected), "recovery final-audit code-SHA fields drift")
    for key, digest in expected.items():
        recorded = code.get(key)
        _need(isinstance(recorded, str) and len(recorded) == 64,
              f"recovery final-audit code SHA is malformed: {key}")
        _need(recorded == digest, f"recovery final-audit code SHA drift: {key}")
    return dict(expected)


def _json_object(path: Path, *, immutable: bool = False) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(), f"receipt is not a regular file: {path}")
    if immutable:
        _need(_immutable(path), f"receipt is not immutable mode 0444: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryPackageError(f"receipt JSON is unreadable: {path}") from error
    _need(isinstance(value, dict), f"receipt is not a JSON object: {path}")
    return value


def _canonical_audit_sha256(body: Mapping[str, Any]) -> str:
    # recovery_audit.recover hashes the body before adding output metadata.
    canonical = dict(body)
    canonical.pop("audit_sha256", None)
    canonical.pop("output", None)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _expected_calibration_paths(data_root: Path | None = None) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    root = (data_root or (ROOT / "data/000954")).resolve()

    def make(role: str, sessions: Sequence[str]) -> tuple[Path, ...]:
        directory = (root / f"sub-HumanPitt-{role}-calib").resolve()
        _need(directory.is_dir() and not directory.is_symlink(), f"calibration directory missing: {directory}")
        paths: list[Path] = []
        for session in sessions:
            path = directory / f"sub-HumanPitt-{role}-calib_{session}.nwb"
            _need(path.is_file() and not path.is_symlink(), f"calibration file missing: {path}")
            paths.append(path.resolve())
        return tuple(paths)

    heldin = make("held-in", HELDIN_SESSIONS)
    heldout = make("held-out", HELDOUT_SESSIONS)
    _need(len(heldin) == 13 and len(heldout) == 14, "recovery calibration roster must be 13+14")
    _need(len(set(heldin + heldout)) == 27, "recovery calibration roster contains duplicates")
    return heldin, heldout


def _default_calibration_files() -> tuple[Path, ...]:
    heldin, heldout = _expected_calibration_paths()
    return heldin + heldout


def validate_exact_calibration_allowlist(
    paths: Sequence[str | Path], *, data_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the exact ordered 13+14 public calibration roster.

    This calls the existing package path guard but does not load an NWB.  A
    caller that supplies a subset (or a query/minival path) fails closed.
    """

    heldin, heldout = _expected_calibration_paths(data_root)
    expected = heldin + heldout
    observed = tuple(Path(path).resolve() for path in paths)
    _need(len(observed) == 27, f"recovery package requires exactly 27 calibration files, got {len(observed)}")
    _need(observed == expected, "recovery calibration order/allowlist drift")
    _need(len(set(observed)) == 27, "recovery calibration allowlist contains duplicates")
    config = FalconConfig(task=FalconTask.h1)
    tags: list[str] = []
    for path in observed:
        _need(assert_deployment_calibration_path(path) == path, f"deployment path guard drift: {path}")
        tags.append(config.hash_dataset(path.stem))
    _need(len(set(tags)) == 27, "recovery calibration dataset tags collide")
    return {
        "files": [str(path) for path in observed],
        "heldin": [str(path) for path in heldin],
        "heldout": [str(path) for path in heldout],
        "dataset_tags": tags,
        "counts": {"heldin": 13, "heldout": 14, "total": 27},
    }


_PACKAGE_PREFLIGHT_SCOPE_ZERO: dict[str, Any] = {
    "calibration_recording_bytes_opened": 0,
    "query_recordings_opened": 0,
    "formal_test_labels_opened": 0,
    "target_optimizer_steps": 0,
    "target_backward_steps": 0,
    "payload_exported": False,
    "docker_built_or_pushed": False,
    "evalai_accessed_or_submitted": False,
}


def _validate_package_preflight_body(
    body: Mapping[str, Any], *, preflight_path: Path | None,
    terminal: Mapping[str, Any], assets: Mapping[str, Any], allowlist: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every immutable package-preflight binding before downstream use.

    The official helper emits a source-only receipt.  Recovery consumers must
    independently bind that receipt to the immutable terminal checkpoint and
    asset manifest, then enforce the exact ordered 13+14 index and every
    no-side-effect scope field.  Keeping this gate shared by preflight,
    export, and audit prevents a schema-only or index-only receipt from being
    accepted at a later boundary.
    """

    if preflight_path is not None:
        _need(_immutable(preflight_path), f"recovery preflight is not immutable mode 0444: {preflight_path}")
    _need(body.get("schema") == official_package.PACKAGE_PREFLIGHT_SCHEMA,
          "recovery package preflight schema drift")
    _need(body.get("status") == official_package.PACKAGE_PREFLIGHT_STATUS,
          "recovery package preflight status drift")

    checkpoint = body.get("checkpoint")
    _need(isinstance(checkpoint, Mapping), "recovery package preflight checkpoint binding is malformed")
    terminal_path = Path(str(terminal.get("path", ""))).resolve()
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve()
    _need(checkpoint_path == terminal_path, "recovery package preflight checkpoint path drift")
    _need(checkpoint_path.is_file() and not checkpoint_path.is_symlink(),
          "recovery package preflight checkpoint is missing")
    checkpoint_sha = checkpoint.get("sha256")
    _need(isinstance(checkpoint_sha, str) and len(checkpoint_sha) == 64,
          "recovery package preflight checkpoint SHA is malformed")
    _need(checkpoint_sha == terminal.get("sha256"),
          "recovery package preflight checkpoint SHA is not recovery-bound")
    _need(_sha256_file(checkpoint_path) == checkpoint_sha,
          "recovery package preflight checkpoint SHA drift")

    asset_manifest = body.get("asset_manifest")
    _need(isinstance(asset_manifest, Mapping), "recovery package preflight asset binding is malformed")
    asset_path = Path(str(assets.get("path", ""))).resolve()
    preflight_asset_path = Path(str(asset_manifest.get("path", ""))).resolve()
    _need(preflight_asset_path == asset_path, "recovery package preflight asset path drift")
    _need(preflight_asset_path.is_file() and not preflight_asset_path.is_symlink(),
          "recovery package preflight asset manifest is missing")
    asset_sha = asset_manifest.get("sha256")
    _need(isinstance(asset_sha, str) and len(asset_sha) == 64,
          "recovery package preflight asset SHA is malformed")
    _need(asset_sha == assets.get("sha256"),
          "recovery package preflight asset SHA is not recovery-bound")
    _need(_sha256_file(preflight_asset_path) == asset_sha,
          "recovery package preflight asset SHA drift")

    indexed = body.get("calibration_files_indexed_only")
    _need(isinstance(indexed, list) and len(indexed) == 27,
          "recovery package preflight must contain exactly 27 indexed calibrations")
    expected_paths = tuple(str(path) for path in allowlist["files"])
    expected_tags = tuple(str(tag) for tag in allowlist["dataset_tags"])
    observed_paths: list[str] = []
    observed_tags: list[str] = []
    for row, (expected_path, expected_tag) in zip(indexed, zip(expected_paths, expected_tags)):
        _need(isinstance(row, Mapping), "recovery package preflight calibration index row is malformed")
        _need(set(row) == {"path", "dataset_tag"},
              "recovery package preflight calibration index row schema drift")
        observed_path = str(Path(str(row.get("path", ""))).resolve())
        observed_tag = str(row.get("dataset_tag"))
        _need(observed_path == str(Path(expected_path).resolve()),
              "recovery package preflight calibration index order/path drift")
        _need(observed_tag == expected_tag,
              "recovery package preflight calibration index dataset-tag drift")
        observed_paths.append(observed_path)
        observed_tags.append(observed_tag)
    _need(tuple(observed_paths) == tuple(str(Path(path).resolve()) for path in expected_paths),
          "recovery package preflight calibration index is not exact ordered 27")
    _need(tuple(observed_tags) == expected_tags,
          "recovery package preflight calibration dataset-tag roster drift")

    _need(body.get("runtime_class") ==
          "third_party.falcon_challenge.h1_carrierid_all_source_decoder.H1CarrierIdAllSourceDecoder",
          "recovery package preflight runtime class drift")
    _need(body.get("payload_schema") == H1_ALL_SOURCE_PAYLOAD_SCHEMA,
          "recovery package preflight payload schema drift")
    code_sha = body.get("code_sha256")
    _need(isinstance(code_sha, Mapping), "recovery package preflight code-SHA binding is malformed")
    code_paths = {
        "package": Path(official_package.__file__).resolve(),
        "runtime": (ROOT / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py").resolve(),
        "carrier_model": (ROOT / "src/models/components/h1_carrierid_spint.py").resolve(),
        "carrier_data": (ROOT / "src/data/h1_carrierid_all_source_official.py").resolve(),
        "carrier_estimator": (ROOT / "src/data/h1_m4_eb_pilot.py").resolve(),
        "deployment_data": (ROOT / "src/data/h1_carrierid_all_source_deployment.py").resolve(),
    }
    _need(set(code_sha) == set(code_paths), "recovery package preflight code-SHA fields drift")
    for key, source_path in code_paths.items():
        _need(source_path.is_file() and not source_path.is_symlink(),
              f"recovery package preflight code path is missing: {key}")
        recorded_sha = code_sha.get(key)
        _need(isinstance(recorded_sha, str) and len(recorded_sha) == 64,
              f"recovery package preflight code SHA is malformed: {key}")
        _need(recorded_sha == _sha256_file(source_path),
              f"recovery package preflight code SHA drift: {key}")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping), "recovery package preflight scope is malformed")
    _need(set(scope) == set(_PACKAGE_PREFLIGHT_SCOPE_ZERO),
          "recovery package preflight scope fields drift")
    for key, expected in _PACKAGE_PREFLIGHT_SCOPE_ZERO.items():
        _need(scope.get(key) == expected, f"recovery package preflight scope {key} drift")
    return {
        "checkpoint": dict(checkpoint),
        "asset_manifest": dict(asset_manifest),
        "calibration_files_indexed_only": [dict(row) for row in indexed],
        "scope": dict(scope),
    }


def _validate_recovery_scope(body: Mapping[str, Any]) -> None:
    gap = body.get("provenance_gap")
    _need(isinstance(gap, Mapping), "recovery receipt lacks provenance_gap")
    for key, expected in {
        "execution_receipt_present": False,
        "nonce_bound_launch_verified": False,
        "nonce_reconstructed": False,
    }.items():
        _need(gap.get(key) is expected, f"recovery provenance gap {key} is not explicit false")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping), "recovery receipt lacks scope")
    for key, expected in {
        "target_recordings_opened": 0,
        "formal_test_labels_opened": 0,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "evalai_submission_authorized": False,
        "target_evaluation_authorized_by_this_receipt": False,
    }.items():
        _need(scope.get(key) == expected, f"recovery scope {key} drift")


def _validate_recovery_terminal_bindings(body: Mapping[str, Any], asset_path: Path) -> None:
    assets = body.get("asset_evidence")
    _need(isinstance(assets, Mapping), "recovery receipt lacks asset evidence")
    _need(Path(str(assets.get("path", ""))).resolve() == asset_path.resolve(), "recovery asset path drift")
    _need(_sha256_file(asset_path) == assets.get("sha256"), "recovery asset manifest SHA drift")
    asset_sha = str(assets.get("sha256"))

    prepared = body.get("prepared_launch")
    _need(isinstance(prepared, Mapping), "recovery receipt lacks prepared-launch evidence")
    prepared_path = Path(str(prepared.get("path", ""))).resolve()
    _need(prepared_path.is_file() and _immutable(prepared_path), "prepared launch receipt is not immutable")
    _need(_sha256_file(prepared_path) == prepared.get("sha256"), "prepared launch receipt SHA drift")
    candidate = prepared.get("candidate")
    _need(isinstance(candidate, Mapping), "prepared launch candidate is malformed")
    _need(candidate.get("asset_manifest_sha256") == asset_sha, "prepared launch does not bind asset SHA")

    terminal = body.get("terminal_checkpoint")
    _need(isinstance(terminal, Mapping), "recovery receipt lacks terminal checkpoint")
    checkpoint = Path(str(terminal.get("path", ""))).resolve()
    _need(checkpoint.is_file() and not checkpoint.is_symlink(), "terminal checkpoint is missing")
    _need(_sha256_file(checkpoint) == terminal.get("sha256"), "terminal checkpoint SHA drift")
    config = terminal.get("config")
    _need(isinstance(config, Mapping), "terminal checkpoint lacks resolved config binding")
    config_path = Path(str(config.get("path", ""))).resolve()
    _need(config_path.is_file() and not config_path.is_symlink(), "resolved config is missing")
    _need(_sha256_file(config_path) == config.get("sha256"), "resolved config SHA drift")
    _need(terminal.get("epoch") == 49 and terminal.get("epochs_completed") == 50,
          "terminal checkpoint epoch contract drift")
    _need(terminal.get("global_step") == recovery_audit.EXPECTED_GLOBAL_STEP,
          "terminal checkpoint global-step contract drift")
    _need(terminal.get("optimizer_states") == 1 and terminal.get("lr_schedulers") == 0,
          "terminal optimizer/scheduler contract drift")
    _need(terminal.get("state_finite") is True and int(terminal.get("state_dict_tensors", 0)) > 0,
          "terminal state invariant is missing")
    metadata = terminal.get("metadata")
    _need(isinstance(metadata, Mapping), "terminal checkpoint metadata is missing")
    for key, expected in {
        "asset_manifest_sha256": asset_sha,
        "config_sha256": config.get("sha256"),
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "formal_test_labels_opened": 0,
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
    }.items():
        _need(metadata.get(key) == expected, f"terminal metadata {key} drift")


def validate_recovery_audit(
    path: str | Path = RECOVERY_AUDIT, *, expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate immutable v1 recovery provenance without reconstructing a nonce."""

    _require_cuda_hidden()
    candidate = Path(path).resolve()
    body = _json_object(candidate, immutable=True)
    _need(body.get("schema") == recovery_audit.SCHEMA, "recovery schema drift")
    _need(body.get("status") == recovery_audit.PASS_STATUS, "recovery status drift")
    _validate_recovery_scope(body)
    recorded_audit_sha = body.get("audit_sha256")
    _need(isinstance(recorded_audit_sha, str) and len(recorded_audit_sha) == 64,
          "recovery audit SHA is malformed")
    _need(recorded_audit_sha == _canonical_audit_sha256(body), "recovery internal audit SHA drift")
    actual_file_sha = _sha256_file(candidate)
    if expected_file_sha256 is None and candidate == RECOVERY_AUDIT.resolve():
        expected_file_sha256 = "50c26bc58be52fd663dbeba6239c7415c1093927c82ccebb4f354409adee5fd3"
    if expected_file_sha256 is not None:
        _need(actual_file_sha == expected_file_sha256, "recovery v1 file SHA drift")
    output = body.get("output")
    _need(isinstance(output, Mapping), "recovery output binding is missing")
    _need(Path(str(output.get("path", ""))).resolve() == candidate, "recovery output path drift")
    # v1 was published before the writer's post-publication return value was
    # attached to the in-memory body, so its immutable output object contains
    # only ``path``.  Verify the file SHA above (and the known v1 SHA when this
    # is the canonical receipt) without pretending that a missing field exists.
    if output.get("sha256") is not None:
        _need(output.get("sha256") == actual_file_sha, "recovery output SHA drift")
    _need(body.get("provenance_gap", {}).get("execution_receipt_present") is False,
          "recovery execution receipt is unexpectedly present")
    marker = body.get("provenance_gap", {})
    _need(marker.get("legacy_start_marker_has_nonce") is False, "legacy START nonce was inferred")

    assets = body.get("asset_evidence")
    _need(isinstance(assets, Mapping), "recovery asset evidence is malformed")
    asset_path = Path(str(assets.get("path", ""))).resolve()
    _validate_recovery_terminal_bindings(body, asset_path)

    source_files = assets.get("source_files")
    _need(isinstance(source_files, list) and len(source_files) == 13, "recovery source roster must contain 13 files")
    for row in source_files:
        _need(isinstance(row, Mapping), "recovery source row is malformed")
        source = Path(str(row.get("path", ""))).resolve()
        _need(source.is_file() and not source.is_symlink(), f"recovery source file is missing: {source}")
        _need(_sha256_file(source) == row.get("sha256"), f"recovery source SHA drift: {source}")

    code = body.get("code_evidence")
    _need(isinstance(code, Mapping) and code.get("all_match") is True, "recovery code evidence is not all-match")
    current = code.get("current")
    _need(isinstance(current, Mapping), "recovery current code evidence is malformed")
    for key, source_path in recovery_audit.CODE_PATHS.items():
        _need(source_path.is_file() and not source_path.is_symlink(), f"recovery code path missing: {source_path}")
        _need(current.get(key) == _sha256_file(source_path), f"recovery code SHA drift: {key}")
    return {
        "path": str(candidate),
        "sha256": actual_file_sha,
        "audit_sha256": recorded_audit_sha,
        "body": body,
    }


def _assert_recovery_output(path: Path, *, kind: str) -> Path:
    output = path.resolve()
    _need(output.parent == RECOVERY_ROOT.resolve(), f"recovery {kind} output must use the new append-only namespace")
    _need("RECOVERY_PACKAGE" in output.name and output.name.endswith((".json", ".pkl")),
          f"recovery {kind} output name is not append-only: {output.name}")
    old_names = {
        "H1_CARRIERID_ALL_SOURCE_PACKAGE_PREFLIGHT_v1.json",
        "h1_carrierid_all_source_official_payload_v1.pkl",
        "H1_CARRIERID_ALL_SOURCE_RUNTIME_SMOKE_v1.json",
        "H1_CARRIERID_ALL_SOURCE_TERMINAL_AUDIT_v1.json",
    }
    _need(output.name not in old_names, f"recovery {kind} output collides with official watcher path")
    return output


def _publish_staged(staged: Path, destination: Path) -> None:
    """Publish one already-validated staged artifact without replacement."""

    _need(staged.is_file() and not staged.is_symlink(), f"staged recovery artifact is missing: {staged}")
    _need(not destination.exists() and not destination.is_symlink(),
          f"refusing to overwrite recovery artifact: {destination}")
    os.chmod(staged, 0o444)
    try:
        os.link(staged, destination)
    except FileExistsError as error:
        raise RecoveryPackageError(f"refusing to overwrite recovery artifact: {destination}") from error
    _need(_immutable(destination), f"published recovery artifact is not immutable mode 0444: {destination}")


def prepare_recovery_package_preflight(
    *, checkpoint: str | Path | None = None, asset_manifest: str | Path | None = None,
    calibration_files: Sequence[str | Path] | None = None,
    output: str | Path = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    """Run the existing package preflight in a recovery-only output path."""

    _require_cuda_hidden()
    recovery = validate_recovery_audit()
    root_body = recovery["body"]
    terminal = root_body["terminal_checkpoint"]
    assets = root_body["asset_evidence"]
    checkpoint_path = Path(checkpoint or terminal["path"]).resolve()
    asset_path = Path(asset_manifest or assets["path"]).resolve()
    _need(checkpoint_path == Path(str(terminal["path"])).resolve(), "recovery checkpoint override is not bound")
    _need(asset_path == Path(str(assets["path"])).resolve(), "recovery asset override is not bound")
    allowlist = validate_exact_calibration_allowlist(
        _default_calibration_files() if calibration_files is None else calibration_files,
    )
    destination = _assert_recovery_output(Path(output), kind="preflight")
    _need(not destination.exists() and not destination.is_symlink(),
          f"refusing to overwrite recovery preflight: {destination}")
    with tempfile.TemporaryDirectory(prefix=".recovery_preflight_", dir=RECOVERY_ROOT) as staging:
        staged = Path(staging) / destination.name
        result = official_package.prepare_package_preflight(
            checkpoint=checkpoint_path, asset_manifest=asset_path,
            calibration_files=tuple(Path(path) for path in allowlist["files"]), output=staged,
        )
        _validate_package_preflight_body(
            result, preflight_path=staged, terminal=terminal, assets=assets, allowlist=allowlist,
        )
        _publish_staged(staged, destination)
    # Re-read the final hard-linked copy so all returned paths/hash values bind
    # the immutable artifact rather than the temporary staging pathname.
    result = _json_object(destination, immutable=True)
    return {
        **result,
        "recovery_mode": {
            "schema": RECOVERY_PREFLIGHT_SCHEMA,
            "provenance_gap": {
                "execution_receipt_present": False,
                "nonce_bound_launch_verified": False,
                "nonce_reconstructed": False,
            },
            "recovery_audit": {"path": recovery["path"], "sha256": recovery["sha256"]},
            "calibration_allowlist": allowlist,
            "submission_authorized": False,
        },
        "output": str(destination),
        "output_sha256": sha256_file(destination),
    }


def export_recovery_payload(
    *, preflight: str | Path, calibration_files: Sequence[str | Path] | None = None,
    output: str | Path = DEFAULT_PAYLOAD, allow_export: bool = False,
) -> dict[str, Any]:
    """Optionally invoke the existing exporter, only under explicit recovery gating.

    The default is deliberately disabled.  This function is not called by the
    dry-run command and does not grant submission authority.
    """

    _require_cuda_hidden()
    _need(allow_export is True, "recovery payload export is disabled without explicit allow_export=True")
    recovery = validate_recovery_audit()
    preflight_path = _assert_recovery_output(Path(preflight), kind="preflight")
    _need(preflight_path.is_file() and _immutable(preflight_path), "recovery preflight is not immutable")
    preflight_body = _json_object(preflight_path, immutable=True)
    terminal = recovery["body"]["terminal_checkpoint"]
    assets = recovery["body"]["asset_evidence"]
    allowlist = validate_exact_calibration_allowlist(
        _default_calibration_files() if calibration_files is None else calibration_files,
    )
    _validate_package_preflight_body(
        preflight_body, preflight_path=preflight_path, terminal=terminal, assets=assets, allowlist=allowlist,
    )
    destination = _assert_recovery_output(Path(output), kind="payload")
    _need(not destination.exists() and not destination.is_symlink(),
          f"refusing to overwrite recovery payload: {destination}")
    with tempfile.TemporaryDirectory(prefix=".recovery_payload_", dir=RECOVERY_ROOT) as staging:
        staged = Path(staging) / destination.name
        result = official_package.export_payload(
            checkpoint=Path(str(terminal["path"])), asset_manifest=Path(str(assets["path"])),
            calibration_files=tuple(Path(path) for path in allowlist["files"]), output=staged,
        )
        _need(staged.is_file() and not staged.is_symlink(), "recovery exporter did not publish staged payload")
        os.chmod(staged, 0o444)
        validated = validate_recovery_payload(staged, calibration_files=allowlist["files"])
        _publish_staged(staged, destination)
    validated["path"] = str(destination)
    validated["sha256"] = sha256_file(destination)
    return {
        "status": RECOVERY_PAYLOAD_STATUS,
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "datasets": result.get("datasets", []),
        "payload_validation": validated,
        "recovery_audit": {"path": recovery["path"], "sha256": recovery["sha256"]},
        "submission_authorized": False,
    }


def validate_recovery_payload(
    payload: str | Path, *, calibration_files: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Validate payload schema, all 27 shapes/finite arrays, hashes, and model state."""

    _require_cuda_hidden()
    path = Path(payload).resolve()
    _need(_immutable(path), f"recovery payload is not immutable mode 0444: {path}")
    try:
        from third_party.falcon_challenge.spint_decoder import CPU_Unpickler
        with path.open("rb") as handle:
            body = CPU_Unpickler(handle).load()
    except Exception as error:
        raise RecoveryPackageError(f"recovery payload is unreadable: {path}") from error
    _need(isinstance(body, dict), "recovery payload is not a mapping")
    expected_keys = {
        "decoder", "task", "window_size", "behavior_scaling_factor", "interpolate_trials",
        "interpolate_trials_kind", "calib_trial_features", "calib_carriers", "smooth_calibration",
        "carrier_payload_schema", "carrier_asset_manifest_sha256", "carrier_transform_sha256",
        "carrier_normalizer_sha256", "source_checkpoint_metadata", "calibration_receipts",
        "deployment_contract",
    }
    _need(set(body) == expected_keys, "recovery payload top-level schema drift")
    try:
        carriers = validate_carrier_payload(body, expected_task=FalconTask.h1)
    except (H1CarrierIdPayloadError, TypeError, ValueError) as error:
        raise RecoveryPackageError(str(error)) from error

    allowlist = validate_exact_calibration_allowlist(
        _default_calibration_files() if calibration_files is None else calibration_files,
    )
    config = FalconConfig(task=FalconTask.h1)
    expected_tags = tuple(allowlist["dataset_tags"])
    _need(tuple(sorted(carriers)) == tuple(sorted(expected_tags)), "recovery payload dataset roster drift")
    features = body["calib_trial_features"]
    receipts = body["calibration_receipts"]
    _need(isinstance(receipts, list) and len(receipts) == 27, "recovery payload calibration receipts drift")
    by_tag = {str(row.get("dataset_tag")): row for row in receipts if isinstance(row, Mapping)}
    _need(set(by_tag) == set(expected_tags), "recovery payload receipt tags drift")
    for expected_path, tag in zip(allowlist["files"], expected_tags):
        _need(Path(str(by_tag[tag].get("path", ""))).resolve() == Path(expected_path).resolve(),
              f"{tag}: calibration receipt path drift")
        carrier = np.asarray(body["calib_carriers"][tag], dtype=np.float32)
        identity = np.asarray(features[tag], dtype=np.float32)
        support_m = by_tag[tag].get("support_m")
        _need(support_m in (3, 4), f"{tag}: recovery payload support_m must be 3 or 4")
        expected_m = 4 if str(expected_path) in set(allowlist["heldin"]) else 3
        _need(support_m == expected_m, f"{tag}: recovery payload support_m does not match held-in/held-out contract")
        _need(carrier.shape == (176, 4) and identity.shape == (support_m, 1024, 176),
              f"{tag}: recovery payload shape drift")
        support_values = by_tag[tag].get("support_trial_numbers")
        _need(isinstance(support_values, list) and len(support_values) == support_m,
              f"{tag}: recovery payload support trial list drift")
        try:
            calibration_record = load_deployment_calibration_record_v5(Path(expected_path))
            expected_support_values = tuple(float(value) for value in calibration_record.trial_values[:support_m])
            observed_support_values = tuple(float(value) for value in support_values)
        except (OSError, TypeError, ValueError, PilotDataError) as error:
            raise RecoveryPackageError(f"{tag}: calibration support TrialNum values are unreadable") from error
        _need(len(set(observed_support_values)) == support_m,
              f"{tag}: recovery payload support trial list contains padding/duplicates")
        _need(observed_support_values == expected_support_values,
              f"{tag}: recovery payload support TrialNum values drift")
        _need(bool(np.isfinite(carrier).all()) and bool(np.isfinite(identity).all()),
              f"{tag}: recovery payload nonfinite values")
        carrier_sha = hashlib.sha256(np.ascontiguousarray(carrier).tobytes()).hexdigest()
        identity_sha = hashlib.sha256(np.ascontiguousarray(identity).tobytes()).hexdigest()
        _need(by_tag[tag].get("carrier_sha256") == carrier_sha, f"{tag}: carrier SHA drift")
        _need(by_tag[tag].get("identity_sha256") == identity_sha, f"{tag}: identity SHA drift")
        _need(isinstance(by_tag[tag].get("input_sha256"), str)
              and len(by_tag[tag]["input_sha256"]) == 64,
              f"{tag}: calibration input SHA is malformed")
        actual_input_sha = _sha256_file(Path(expected_path))
        _need(by_tag[tag]["input_sha256"] == actual_input_sha,
              f"{tag}: calibration input SHA drift")

    recovery = validate_recovery_audit()
    assets = recovery["body"]["asset_evidence"]
    prepared_candidate = recovery["body"]["prepared_launch"]["candidate"]
    _need(body.get("carrier_asset_manifest_sha256") == assets.get("sha256"), "payload asset SHA drift")
    _need(body.get("carrier_transform_sha256") == prepared_candidate.get("transform_sha256"),
          "payload transform SHA drift")
    _need(body.get("carrier_normalizer_sha256") == prepared_candidate.get("normalizer_sha256"),
          "payload normalizer SHA drift")
    metadata = body.get("source_checkpoint_metadata")
    terminal_metadata = recovery["body"]["terminal_checkpoint"]["metadata"]
    _need(isinstance(metadata, Mapping), "payload checkpoint metadata is missing")
    for key in ("schema", "fresh_seed", "checkpoint_epoch_zero_based", "epochs_completed",
                "asset_manifest_sha256", "config_sha256", "target_optimizer_steps",
                "target_backward_steps", "formal_test_labels_opened"):
        _need(metadata.get(key) == terminal_metadata.get(key), f"payload checkpoint metadata {key} drift")
    contract = body.get("deployment_contract")
    _need(isinstance(contract, Mapping), "payload deployment contract is malformed")
    for key, expected in {
        "target_optimizer_steps": 0, "target_backward_steps": 0,
        "formal_test_labels_packaged": 0, "query_labels_read": 0,
        "model_weights_frozen_at_runtime": True,
    }.items():
        _need(contract.get(key) == expected, f"payload deployment contract {key} drift")

    decoder = body.get("decoder")
    _need(getattr(decoder, "training", None) is False, "payload decoder is not in eval mode")
    state = decoder.state_dict() if hasattr(decoder, "state_dict") else {}
    _need(isinstance(state, Mapping) and bool(state), "payload decoder state is missing")
    for name, value in state.items():
        _need(hasattr(value, "detach") and bool(value.detach().isfinite().all()),
              f"payload decoder state is nonfinite: {name}")
        _need(getattr(value, "device", None) is None or value.device.type == "cpu",
              f"payload decoder state is not CPU: {name}")
    return {
        "schema": H1_ALL_SOURCE_PAYLOAD_SCHEMA,
        "path": str(path),
        "sha256": sha256_file(path),
        "dataset_count": 27,
        "dataset_tags": list(expected_tags),
        "calibration_receipts": [
            {
                "path": str(row.get("path")),
                "dataset_tag": str(row.get("dataset_tag")),
                "input_sha256": str(row.get("input_sha256")),
                "identity_sha256": str(row.get("identity_sha256")),
                "carrier_sha256": str(row.get("carrier_sha256")),
                "support_m": int(row.get("support_m")),
                "support_trial_numbers": list(row.get("support_trial_numbers", [])),
                "identity_shape": [int(row.get("support_m")), 1024, 176],
                "carrier_shape": [176, 4],
            }
            for row in (by_tag[tag] for tag in expected_tags)
        ],
        "shape": {"carrier": [176, 4], "identity": "per_record_m_3_or_4"},
        "identity_shapes": {
            str(tag): [int(by_tag[tag].get("support_m")), 1024, 176] for tag in expected_tags
        },
        "state_dict_tensors": len(state),
        "finite": True,
        "submission_authorized": False,
    }


def validate_recovery_smoke(
    smoke: str | Path, *, payload_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an immutable 13+14 CPU reset/predict smoke receipt."""

    _require_cuda_hidden()
    path = Path(smoke).resolve()
    body = _json_object(path, immutable=True)
    _need(body.get("schema") == RECOVERY_SMOKE_SCHEMA,
          "recovery smoke schema drift")
    _need(body.get("status") == "PASS_RECOVERY_CPU_SMOKE_NO_QUERY_OPENED", "recovery smoke status drift")
    counts = body.get("counts")
    _need(counts == {"heldin": 13, "heldout": 14, "total": 27}, "recovery smoke counts drift")
    rows = body.get("rows")
    _need(isinstance(rows, list) and len(rows) == 27, "recovery smoke must contain 27 rows")
    allowlist = validate_exact_calibration_allowlist(_default_calibration_files())
    expected_rows = [("heldin", path) for path in allowlist["heldin"]] + [("heldout", path) for path in allowlist["heldout"]]
    observed_rows: list[dict[str, Any]] = []
    for row, (split, expected_path) in zip(rows, expected_rows):
        _need(isinstance(row, Mapping), "recovery smoke row is malformed")
        _need(row.get("split") == split and Path(str(row.get("path", ""))).resolve() == Path(expected_path).resolve(),
              "recovery smoke row path/split drift")
        _need(row.get("output_shape") == [1, 7] and row.get("finite") is True,
              f"recovery smoke output invariant drift: {expected_path}")
        observed_rows.append(dict(row))
    scope = body.get("scope")
    _need(isinstance(scope, Mapping), "recovery smoke scope is malformed")
    for key, expected in {
        "query_recordings_opened": 0, "formal_test_labels_opened": 0,
        "target_optimizer_steps": 0, "target_backward_steps": 0,
        "cuda_used": False, "evalai_accessed_or_submitted": False,
    }.items():
        _need(scope.get(key) == expected, f"recovery smoke scope {key} drift")
    _need(body.get("submission_authorized") is False, "recovery smoke authorizes submission")
    payload = body.get("payload")
    _need(isinstance(payload, Mapping), "recovery smoke lacks payload evidence")
    if payload_sha256 is not None:
        _need(payload.get("sha256") == payload_sha256, "recovery smoke/payload SHA mismatch")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": body["schema"],
        "status": body["status"],
        "counts": dict(counts),
        "rows": observed_rows,
        "payload": dict(payload),
        "scope": dict(scope),
        "submission_authorized": False,
    }


def run_cpu_smoke(
    payload: str | Path, *, calibration_files: Sequence[str | Path] | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Reset/predict once per calibration path without reading any query NWB."""

    _require_cuda_hidden()
    _need(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "recovery smoke must hide CUDA explicitly")
    import torch

    _need(torch.cuda.is_available() is False, "recovery smoke unexpectedly sees CUDA")
    validated = validate_recovery_payload(payload, calibration_files=calibration_files)
    allowlist = validate_exact_calibration_allowlist(
        _default_calibration_files() if calibration_files is None else calibration_files,
    )
    from third_party.falcon_challenge.h1_carrierid_all_source_decoder import H1CarrierIdAllSourceDecoder

    decoder = H1CarrierIdAllSourceDecoder(FalconConfig(task=FalconTask.h1), str(Path(payload).resolve()), batch_size=1)
    rows: list[dict[str, Any]] = []
    for split, paths in (("heldin", allowlist["heldin"]), ("heldout", allowlist["heldout"])):
        for path in paths:
            decoder.reset([Path(path)])
            prediction = np.asarray(decoder.predict(np.zeros((1, 176), dtype=np.float32)))
            _need(prediction.shape == (1, 7) and bool(np.isfinite(prediction).all()),
                  f"{split} runtime output is not finite [1,7]: {path}")
            _need(getattr(decoder, "device", torch.device("cpu")).type == "cpu",
                  f"{split} runtime selected non-CPU device: {path}")
            rows.append({"split": split, "path": str(path), "output_shape": list(prediction.shape), "finite": True})
    result = {
        "schema": RECOVERY_SMOKE_SCHEMA,
        "status": "PASS_RECOVERY_CPU_SMOKE_NO_QUERY_OPENED",
        "payload": validated,
        "rows": rows,
        "counts": {"heldin": 13, "heldout": 14, "total": 27},
        "scope": {
            "query_recordings_opened": 0, "formal_test_labels_opened": 0,
            "target_optimizer_steps": 0, "target_backward_steps": 0,
            "cuda_used": False, "evalai_accessed_or_submitted": False,
        },
        "submission_authorized": False,
    }
    if output is not None:
        destination = _assert_recovery_output(Path(output), kind="smoke")
        write_immutable_json(destination, result)
        result["output"] = str(destination)
        result["output_sha256"] = sha256_file(destination)
    return result


def _pair_distance_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Return deterministic cosine/RMS/L2/norm-ratio diagnostics for two arrays."""

    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    _need(first.shape == second.shape and bool(np.isfinite(first).all()) and bool(np.isfinite(second).all()),
          "stability audit comparison arrays are malformed or nonfinite")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    _need(first_norm > 0.0 and second_norm > 0.0, "stability audit comparison norm is zero")
    return {
        "cosine": float(np.dot(first, second) / (first_norm * second_norm)),
        "rms": float(np.sqrt(np.mean(np.square(first - second)))),
        "distance": float(np.linalg.norm(first - second)),
        "norm_ratio_m3_over_m4": float(first_norm / second_norm),
        "norm_m3": first_norm,
        "norm_m4": second_norm,
    }


def _deployment_design_diagnostics(record: Any, plan: Any, values: Sequence[float]) -> dict[str, Any]:
    rates = np.concatenate([record.blocks_for(value).rates for value in values], axis=0)
    z = ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T
    design = np.column_stack((np.ones(z.shape[0]), z))
    ridge_system = design.T @ design + np.eye(design.shape[1]) * plan.ridge_lambda
    ridge_system[0, 0] = design[:, 0] @ design[:, 0]
    return {
        "rows": int(design.shape[0]),
        "columns": int(design.shape[1]),
        "rank": int(np.linalg.matrix_rank(design)),
        "condition": float(np.linalg.cond(design)),
        "ridge_system_condition": float(np.linalg.cond(ridge_system)),
    }


def run_m3_stability_audit(
    *, checkpoint: str | Path | None = None, asset_manifest: str | Path | None = None,
    calibration_files: Sequence[str | Path] | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Compare M=3 against first-four M=4 on public calibration only.

    The audit deliberately opens only the exact calibration allowlist and
    never constructs a query/formal-test loader.  It loads the frozen source
    checkpoint on CPU, compares raw/EB carriers and the model's identity
    projection, and records design rank/conditioning for every M>=4 file.
    """

    _require_cuda_hidden()
    import torch

    _need(torch.cuda.is_available() is False, "M3 stability audit unexpectedly sees CUDA")
    recovery = validate_recovery_audit()
    terminal = recovery["body"]["terminal_checkpoint"]
    assets_evidence = recovery["body"]["asset_evidence"]
    checkpoint_path = Path(checkpoint or terminal["path"]).resolve()
    asset_path = Path(asset_manifest or assets_evidence["path"]).resolve()
    _need(checkpoint_path == Path(str(terminal["path"])).resolve(), "stability audit checkpoint override is not bound")
    _need(asset_path == Path(str(assets_evidence["path"])).resolve(), "stability audit asset override is not bound")
    allowlist = validate_exact_calibration_allowlist(
        _default_calibration_files() if calibration_files is None else calibration_files,
    )
    _need(tuple(allowlist["heldin"]) == tuple(allowlist["files"][:13]),
          "stability audit held-in calibration partition drift")
    assets = load_all_source_assets(asset_path)
    _need(_sha256_file(asset_path) == assets_evidence["sha256"], "stability audit asset SHA drift")
    checkpoint_body = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _need(isinstance(checkpoint_body, Mapping), "stability audit checkpoint is not a mapping")
    official_package.validate_checkpoint_metadata(
        checkpoint_body, asset_manifest_sha256=assets.manifest_sha256,
    )
    model = official_package._load_checkpoint_model(checkpoint_path)
    model.eval().to("cpu")
    _need(getattr(model, "training", None) is False, "stability audit model is not in eval mode")
    net = getattr(model, "net", None)
    _need(hasattr(net, "carrierid_identity_projection"), "stability audit model lacks CarrierID identity projection")

    rows: list[dict[str, Any]] = []
    for path in allowlist["heldin"]:
        record = load_deployment_calibration_record_v5(path)
        _need(len(record.trial_values) >= 4, f"stability audit requires M>=4 source calibration: {path}")
        m3_values = tuple(float(value) for value in record.trial_values[:3])
        m4_values = tuple(float(value) for value in record.trial_values[:4])
        fit3 = fit_deployment_carrier(record, assets.plan, m3_values)
        fit4 = fit_frozen_carrier(record, assets.plan, m4_values)
        identity3 = np.stack([interpolate_trial_identity(record, value) for value in m3_values], axis=0)
        identity4 = np.stack([interpolate_trial_identity(record, value) for value in m4_values], axis=0)
        carrier3 = assets.normalizer.normalize(fit3["carrier"]).astype(np.float32)
        carrier4 = assets.normalizer.normalize(fit4["carrier"]).astype(np.float32)
        with torch.inference_mode():
            projection3 = net.carrierid_identity_projection(
                torch.from_numpy(identity3).unsqueeze(0), torch.from_numpy(carrier3).unsqueeze(0),
            ).detach().cpu().numpy()
            projection4 = net.carrierid_identity_projection(
                torch.from_numpy(identity4).unsqueeze(0), torch.from_numpy(carrier4).unsqueeze(0),
            ).detach().cpu().numpy()
        rows.append({
            "path": str(path),
            "input_sha256": _sha256_file(path),
            "dataset_tag": FalconConfig(task=FalconTask.h1).hash_dataset(Path(path).stem),
            "trial_count_available": len(record.trial_values),
            "m3": {"support_m": 3, "support_trial_numbers": list(m3_values), "identity_shape": list(identity3.shape)},
            "m4": {"support_m": 4, "support_trial_numbers": list(m4_values), "identity_shape": list(identity4.shape)},
            "raw_carrier": _pair_distance_metrics(fit3["raw_carrier"], fit4["raw_carrier"]),
            "eb_carrier": _pair_distance_metrics(fit3["carrier"], fit4["carrier"]),
            "model_identity_projection": {
                **_pair_distance_metrics(projection3, projection4),
                "shape_m3": list(projection3.shape),
                "shape_m4": list(projection4.shape),
            },
            "design_m3": _deployment_design_diagnostics(record, assets.plan, m3_values),
            "design_m4": _deployment_design_diagnostics(record, assets.plan, m4_values),
            "padding_or_duplication": False,
        })
    result: dict[str, Any] = {
        "schema": RECOVERY_STABILITY_SCHEMA,
        "status": "PASS_RECOVERY_M3_FIRST4_PUBLIC_CALIBRATION_STABILITY_NO_QUERY_OPENED",
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256_file(checkpoint_path)},
        "asset_manifest": {
            "path": str(asset_path), "sha256": _sha256_file(asset_path),
            "arrays_path": str(assets.arrays_path), "arrays_sha256": assets.arrays_sha256,
            "transform_sha256": assets.plan.transform_sha256,
            "normalizer_sha256": assets.normalizer.normalizer_sha256,
        },
        "calibration_allowlist": allowlist,
        "code_sha256": {
            "recovery_package": _sha256_file(Path(__file__).resolve()),
            "official_package": _sha256_file(Path(official_package.__file__).resolve()),
            "runtime": _sha256_file(ROOT / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py"),
            "carrier_model": _sha256_file(ROOT / "src/models/components/h1_carrierid_spint.py"),
            "carrier_data": _sha256_file(ROOT / "src/data/h1_carrierid_all_source_official.py"),
            "carrier_estimator": _sha256_file(ROOT / "src/data/h1_m4_eb_pilot.py"),
            "deployment_data": _sha256_file(ROOT / "src/data/h1_carrierid_all_source_deployment.py"),
        },
        "rows": rows,
        "counts": {"m_ge_4_files": len(rows), "heldin": len(rows), "heldout_m3_only": 14},
        "scope": {
            "source_public_calibration_only": True,
            "query_recordings_opened": 0,
            "formal_test_labels_opened": 0,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "cuda_used": False,
            "evalai_accessed_or_submitted": False,
        },
        "submission_authorized": False,
        "mixed_m_retraining_indicated": False,
        "interpretation": (
            "The frozen M=4-trained forward path accepts M=3 by mean pooling; this audit is a deployment-stability "
            "diagnostic, not a query-label performance claim."
        ),
    }
    if output is not None:
        destination = _assert_recovery_output(Path(output), kind="stability-audit")
        write_immutable_json(destination, result)
        result["output"] = str(destination)
        result["output_sha256"] = sha256_file(destination)
    return result


def write_recovery_package_audit(
    *, preflight: str | Path, payload: str | Path | None = None,
    smoke: Mapping[str, Any] | None = None, smoke_path: str | Path | None = None,
    output: str | Path = DEFAULT_AUDIT,
) -> dict[str, Any]:
    """Publish an append-only recovery audit; never authorize submission."""

    _require_cuda_hidden()
    recovery = validate_recovery_audit()
    preflight_path = _assert_recovery_output(Path(preflight), kind="preflight")
    preflight_body = _json_object(preflight_path, immutable=True)
    allowlist = validate_exact_calibration_allowlist(_default_calibration_files())
    terminal = recovery["body"]["terminal_checkpoint"]
    assets = recovery["body"]["asset_evidence"]
    _validate_package_preflight_body(
        preflight_body, preflight_path=preflight_path, terminal=terminal, assets=assets, allowlist=allowlist,
    )
    payload_evidence: dict[str, Any] | None = None
    if payload is not None:
        payload_evidence = validate_recovery_payload(payload)
    smoke_evidence: dict[str, Any] | None = None
    if smoke_path is not None:
        _need(smoke is None, "recovery audit accepts either smoke mapping or smoke path, not both")
        smoke_evidence = validate_recovery_smoke(
            smoke_path, payload_sha256=payload_evidence["sha256"] if payload_evidence is not None else None,
        )
    elif smoke is not None:
        _need(smoke.get("submission_authorized") is False, "recovery smoke cannot authorize submission")
        _need(smoke.get("scope", {}).get("query_recordings_opened") == 0, "recovery smoke opened query recordings")
        _need(isinstance(smoke.get("rows"), list) and len(smoke["rows"]) == 27,
              "recovery smoke mapping must contain 27 rows")
        smoke_evidence = dict(smoke)
    if payload_evidence is not None or smoke_evidence is not None:
        _need(payload_evidence is not None and smoke_evidence is not None,
              "full recovery audit requires both validated payload and smoke evidence")
        _need(smoke_evidence.get("payload", {}).get("sha256") == payload_evidence["sha256"],
              "recovery audit smoke does not bind payload SHA")
        status = RECOVERY_STATUS
    else:
        status = RECOVERY_PREFLIGHT_STATUS
    body: dict[str, Any] = {
        "schema": RECOVERY_SCHEMA,
        "status": status,
        "provenance_gap": {
            "execution_receipt_present": False,
            "nonce_bound_launch_verified": False,
            "nonce_reconstructed": False,
            "source": recovery["path"],
            "source_sha256": recovery["sha256"],
            "interpretation": "source-only package evidence remains valid only with an explicit unbound-launch provenance gap",
        },
        "recovery_audit": {"path": recovery["path"], "sha256": recovery["sha256"], "audit_sha256": recovery["audit_sha256"]},
        "package_preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path), "schema": preflight_body["schema"]},
        "calibration_allowlist": allowlist,
        "terminal_checkpoint": {
            "path": str(terminal["path"]), "sha256": terminal["sha256"],
            "config_path": str(terminal["config"]["path"]), "config_sha256": terminal["config"]["sha256"],
            "epoch": terminal["epoch"], "epochs_completed": terminal["epochs_completed"],
            "global_step": terminal["global_step"], "optimizer_states": terminal["optimizer_states"],
            "lr_schedulers": terminal["lr_schedulers"], "state_dict_tensors": terminal["state_dict_tensors"],
            "state_finite": terminal["state_finite"],
        },
        "payload": payload_evidence,
        "smoke": smoke_evidence,
        "scope": {
            "source_only": True, "calibration_files_indexed": 27,
            "query_recordings_opened": 0, "formal_test_labels_opened": 0,
            "target_optimizer_steps": 0, "target_backward_steps": 0,
            "cuda_used": False,
            "evalai_accessed_or_submitted": False,
        },
        "submission": {
            "authorized": False,
            "requires_independent_human_review": True,
            "nonce_bound_execution_receipt_required_for_official_watcher": True,
        },
        "code_sha256": _current_recovery_package_code_sha256(),
    }
    # Validate the complete closure before publishing the append-only audit;
    # in particular, the deployment-only loader must be hash-bound here just
    # as it is in preflight and stability receipts.
    validate_recovery_package_audit_code(body)
    destination = _assert_recovery_output(Path(output), kind="audit")
    _need(not destination.exists(), f"refusing to overwrite recovery package audit: {destination}")
    write_immutable_json(destination, body)
    return {**body, "output": str(destination), "output_sha256": sha256_file(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT)
    preflight.add_argument("--audit-output", type=Path, default=DEFAULT_PREFLIGHT_AUDIT)
    export = subparsers.add_parser("export")
    export.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    export.add_argument("--output", type=Path, default=DEFAULT_PAYLOAD)
    export.add_argument("--allow-export", action="store_true",
                        help="explicitly enable recovery payload export; never enables submission")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    smoke.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    stability = subparsers.add_parser("stability-audit")
    stability.add_argument("--checkpoint", type=Path)
    stability.add_argument("--asset-manifest", type=Path)
    stability.add_argument("--output", type=Path, default=DEFAULT_STABILITY_AUDIT)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    audit.add_argument("--payload", type=Path)
    audit.add_argument("--smoke", type=Path)
    audit.add_argument("--output", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    if args.command == "preflight":
        prepare_recovery_package_preflight(output=args.output)
        result = write_recovery_package_audit(preflight=args.output, output=args.audit_output)
    elif args.command == "export":
        result = export_recovery_payload(preflight=args.preflight, output=args.output, allow_export=args.allow_export)
    elif args.command == "smoke":
        result = run_cpu_smoke(payload=args.payload, output=args.output)
    elif args.command == "stability-audit":
        result = run_m3_stability_audit(
            checkpoint=args.checkpoint, asset_manifest=args.asset_manifest, output=args.output,
        )
    else:
        result = write_recovery_package_audit(
            preflight=args.preflight, payload=args.payload, smoke_path=args.smoke, output=args.output,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
