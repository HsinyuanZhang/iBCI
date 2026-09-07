#!/usr/bin/env python3
"""Isolated preflight/export path for the H1 all-source CarrierID candidate.

Only explicit ``*calib*.nwb`` files may be packaged.  Their permitted
calibration targets are used to fit the frozen analytic carrier; no minival,
query, formal-test label, EvalAI API, or target-session backward path exists.
The original SPINT exporter remains isolated; this deployment-only v5 payload
adds an explicit three-or-four-trial support budget without padding.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_challenge.config import FalconConfig, FalconTask
from src.data.h1_carrierid_all_source_official import (
    assert_deployment_calibration_path,
    load_all_source_assets,
)
from src.data.h1_carrierid_all_source_deployment import load_deployment_calibration_record_v5
from src.data.h1_m4_eb_pilot import fit_deployment_carrier, interpolate_trial_identity

# Compatibility alias for focused contract tests and downstream callers; v5
# exporter paths use the explicitly versioned deployment loader above.
load_deployment_calibration_record = load_deployment_calibration_record_v5
from src.h1_m4_cce_contract import sha256_file, write_immutable_json
from src.models.components.h1_carrierid_spint import H1CarrierIdSpint
from src.models.h1_carrierid_all_source_official_module import (
    ALL_SOURCE_CHECKPOINT_SCHEMA,
    H1CarrierIdAllSourceLitModule,
)
from third_party.falcon_challenge.h1_carrierid_all_source_decoder import H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5


PACKAGE_PREFLIGHT_SCHEMA = "h1_carrierid_all_public_source_package_preflight_v1"
PACKAGE_PREFLIGHT_STATUS = "PASS_PACKAGE_PREPARED_NOT_EXPORTED_NOT_SUBMITTED"


class H1CarrierIdPackageError(ValueError):
    """The official-candidate checkpoint, calibration allowlist, or payload failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise H1CarrierIdPackageError(message)


def validate_checkpoint_metadata(checkpoint: Mapping[str, Any], *, asset_manifest_sha256: str) -> Mapping[str, Any]:
    row = checkpoint.get("h1_carrierid_all_source_official")
    _need(isinstance(row, Mapping), "checkpoint lacks all-source CarrierID provenance")
    _need(row.get("schema") == ALL_SOURCE_CHECKPOINT_SCHEMA, "all-source checkpoint schema mismatch")
    _need(row.get("fresh_seed") == 42 and row.get("checkpoint_epoch_zero_based") == 49 and
          row.get("epochs_completed") == 50, "checkpoint seed/terminal epoch drift")
    _need(row.get("selected_by") == "fixed_terminal_epoch_no_validation_no_formal_selection",
          "checkpoint was selected through another endpoint")
    _need(row.get("checkpoint_warm_start") is False, "checkpoint records a forbidden warm-start")
    _need(row.get("target_optimizer_steps") == 0 and row.get("target_backward_steps") == 0,
          "checkpoint records target-session training")
    _need(row.get("formal_test_labels_opened") == 0, "checkpoint records formal label access")
    _need(row.get("asset_manifest_sha256") == asset_manifest_sha256, "checkpoint uses another carrier asset")
    return row


def validate_calibration_allowlist(paths: Sequence[str | Path], task_config: FalconConfig) -> tuple[Path, ...]:
    _need(bool(paths), "package calibration allowlist may not be empty")
    resolved = tuple(assert_deployment_calibration_path(path) for path in paths)
    _need(len(resolved) == len(set(resolved)), "package calibration allowlist contains duplicates")
    tags = tuple(task_config.hash_dataset(path.stem) for path in resolved)
    _need(len(tags) == len(set(tags)), "package calibration files collide after FALCON dataset hashing")
    return resolved


def prepare_package_preflight(
    *, checkpoint: Path, asset_manifest: Path, calibration_files: Sequence[Path], output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite package preflight: {output}")
    assets = load_all_source_assets(asset_manifest)
    checkpoint_path = checkpoint.resolve()
    _need(checkpoint_path.is_file(), f"checkpoint missing: {checkpoint_path}")
    body = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _need(isinstance(body, Mapping), "checkpoint is not a mapping")
    metadata = validate_checkpoint_metadata(body, asset_manifest_sha256=assets.manifest_sha256)
    config = FalconConfig(task=FalconTask.h1)
    allowlist = validate_calibration_allowlist(calibration_files, config)
    receipt = {
        "schema": PACKAGE_PREFLIGHT_SCHEMA,
        "status": PACKAGE_PREFLIGHT_STATUS,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path), "metadata": dict(metadata)},
        "asset_manifest": {"path": str(assets.manifest_path), "sha256": assets.manifest_sha256},
        "calibration_files_indexed_only": [
            {"path": str(path), "dataset_tag": config.hash_dataset(path.stem)} for path in allowlist
        ],
        "runtime_class": "third_party.falcon_challenge.h1_carrierid_all_source_decoder.H1CarrierIdAllSourceDecoder",
        "payload_schema": H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5,
        "scope": {
            "calibration_recording_bytes_opened": 0,
            "query_recordings_opened": 0,
            "formal_test_labels_opened": 0,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "payload_exported": False,
            "docker_built_or_pushed": False,
            "evalai_accessed_or_submitted": False,
        },
        "code_sha256": {
            "package": sha256_file(Path(__file__).resolve()),
            "runtime": sha256_file(ROOT / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py"),
            "carrier_model": sha256_file(ROOT / "src/models/components/h1_carrierid_spint.py"),
            "carrier_data": sha256_file(ROOT / "src/data/h1_carrierid_all_source_official.py"),
            "carrier_estimator": sha256_file(ROOT / "src/data/h1_m4_eb_pilot.py"),
            "deployment_data": sha256_file(ROOT / "src/data/h1_carrierid_all_source_deployment.py"),
        },
    }
    write_immutable_json(output, receipt)
    return {**receipt, "output": str(output.resolve()), "output_sha256": sha256_file(output)}


def _write_pickle_once(path: Path, value: Any) -> str:
    output = path.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite CarrierID payload: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(output)


def _load_checkpoint_model(checkpoint_path: Path) -> H1CarrierIdAllSourceLitModule:
    """Load the trusted local module without forwarding torch-only kwargs.

    Lightning's ``load_from_checkpoint`` forwards unknown keyword arguments
    into the module constructor.  ``weights_only`` is a ``torch.load`` option,
    not a constructor option, so passing it here fails on current Lightning.
    The checkpoint metadata load above remains explicitly CPU/mapping-safe.
    """

    return H1CarrierIdAllSourceLitModule.load_from_checkpoint(checkpoint_path, map_location="cpu")


def export_payload(
    *, checkpoint: Path, asset_manifest: Path, calibration_files: Sequence[Path], output: Path,
) -> dict[str, Any]:
    """Package the fixed source-trained model plus per-calibration identities/carriers."""

    assets = load_all_source_assets(asset_manifest)
    checkpoint_path = checkpoint.resolve()
    checkpoint_body = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = validate_checkpoint_metadata(checkpoint_body, asset_manifest_sha256=assets.manifest_sha256)
    config = FalconConfig(task=FalconTask.h1)
    allowlist = validate_calibration_allowlist(calibration_files, config)
    model = _load_checkpoint_model(checkpoint_path)
    _need(isinstance(model.net, H1CarrierIdSpint), "checkpoint consumer is not H1CarrierIdSpint")
    model.eval().to("cpu")
    identities: dict[str, np.ndarray] = {}
    carriers: dict[str, np.ndarray] = {}
    calibration_receipts: list[dict[str, Any]] = []
    for path in allowlist:
        record = load_deployment_calibration_record_v5(path)
        values = tuple(record.trial_values[:4])
        identity = np.stack([interpolate_trial_identity(record, value) for value in values], axis=0).astype(np.float32)
        raw_carrier = np.asarray(fit_deployment_carrier(record, assets.plan, values)["carrier"], dtype=np.float64)
        carrier = assets.normalizer.normalize(raw_carrier).astype(np.float32)
        tag = config.hash_dataset(path.stem)
        _need(identity.ndim == 3 and identity.shape[0] in (3, 4)
              and identity.shape[1:] == (1024, 176) and carrier.shape == (176, 4),
              f"{path}: packaged identity/carrier shape drift")
        identities[tag], carriers[tag] = identity, carrier
        calibration_receipts.append({
            "path": str(path), "dataset_tag": tag, "input_sha256": record.input_sha256,
            "support_m": len(values),
            "support_trial_numbers": list(values),
            "identity_sha256": hashlib.sha256(np.ascontiguousarray(identity).tobytes()).hexdigest(),
            "carrier_sha256": hashlib.sha256(np.ascontiguousarray(carrier).tobytes()).hexdigest(),
        })
    payload = {
        "decoder": model,
        "task": config.task,
        "window_size": 700,
        "behavior_scaling_factor": float(model.hparams.behavior_scaling_factor),
        "interpolate_trials": True,
        "interpolate_trials_kind": "cubic",
        "calib_trial_features": identities,
        "calib_carriers": carriers,
        "smooth_calibration": False,
        "carrier_payload_schema": H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5,
        "carrier_asset_manifest_sha256": assets.manifest_sha256,
        "carrier_transform_sha256": assets.plan.transform_sha256,
        "carrier_normalizer_sha256": assets.normalizer.normalizer_sha256,
        "source_checkpoint_metadata": dict(metadata),
        "calibration_receipts": calibration_receipts,
        "deployment_contract": {
            "carrier_fit": "forward_closed_form_from_each_explicit_calibration_recording",
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "formal_test_labels_packaged": 0,
            "query_labels_read": 0,
            "model_weights_frozen_at_runtime": True,
        },
    }
    payload_sha = _write_pickle_once(output, payload)
    return {"output": str(output.resolve()), "output_sha256": payload_sha, "datasets": calibration_receipts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "export"):
        child = subparsers.add_parser(name)
        child.add_argument("--checkpoint", type=Path, required=True)
        child.add_argument("--asset-manifest", type=Path, required=True)
        child.add_argument("--calibration-file", type=Path, action="append", required=True)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = prepare_package_preflight(
            checkpoint=args.checkpoint, asset_manifest=args.asset_manifest,
            calibration_files=args.calibration_file, output=args.output,
        )
    else:
        result = export_payload(
            checkpoint=args.checkpoint, asset_manifest=args.asset_manifest,
            calibration_files=args.calibration_file, output=args.output,
        )
    print(result)


if __name__ == "__main__":
    main()
