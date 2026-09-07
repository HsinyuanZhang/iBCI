"""Fail-closed terminal package gate for the original-SPINT H1 reproduction.

This program is intentionally separate from the held-in terminal evaluator.
The evaluator never opens formal held-out calibration inputs; packaging for the
FALCON private endpoint must instead use the *public* neural calibration
recordings for every held-out H1 session.  This gate binds that legitimate
calibration step to one fixed epoch-50 checkpoint and an explicit, audited
allowlist.  It never opens private test files or submits anything to EvalAI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from falcon_challenge.config import FalconConfig, FalconTask

from scripts.h1_baseline_eval import (
    EXPECTED_LRS,
    EXPECTED_VARIANTS,
    _validate_config,
    _validate_terminal_checkpoint,
    sha256_file,
)
from src.data.h1_baseline_datamodule import (
    H1_BASELINE_HELDIN_SESSIONS,
    H1_BASELINE_HELDOUT_SESSIONS,
)
from third_party.falcon_challenge.spint_decoder import main as export_spint_decoder


PACKAGE_SCHEMA = "spint_h1_terminal_package_v1"
PACKAGE_PROTOCOL = "spint_h1_baseline_reproduction_v1"
EXPECTED_TOP_LEVEL_KEYS = {
    "decoder",
    "task",
    "window_size",
    "behavior_scaling_factor",
    "interpolate_trials",
    "interpolate_trials_kind",
    "calib_trial_features",
    "smooth_calibration",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    relative = (
        "scripts/h1_terminal_package.py",
        "scripts/h1_baseline_eval.py",
        "src/data/h1_baseline_datamodule.py",
        "src/models/falcon_module.py",
        "src/models/components/spint.py",
        "third_party/falcon_challenge/spint_decoder.py",
        "third_party/falcon_challenge/spint_sample.py",
        "third_party/falcon_challenge/spint_sample.Dockerfile",
    )
    return {name: sha256_file(root / name) for name in relative}


def _session_from_h1_path(path: Path) -> str:
    marker = "_ses-"
    if marker not in path.stem:
        raise ValueError(f"cannot parse H1 session from calibration filename {path}")
    return path.stem[path.stem.index(marker) + 1 :]


def _allowlisted_calibration_files(data_dir: str | Path) -> list[dict[str, Any]]:
    """Return the exact 13 held-in + 14 public held-out H1 calibration files.

    Every path is resolved below its expected canonical directory.  The
    function deliberately neither enumerates minival nor performs a recursive
    glob at the data-root level.
    """

    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"H1 package data directory does not exist: {root}")
    groups = (
        ("heldin_calib", "sub-HumanPitt-held-in-calib", H1_BASELINE_HELDIN_SESSIONS),
        ("public_heldout_calib", "sub-HumanPitt-held-out-calib", H1_BASELINE_HELDOUT_SESSIONS),
    )
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for role, dirname, expected_sessions in groups:
        directory = (root / dirname).resolve()
        try:
            directory.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"H1 package calibration directory escapes data root: {directory}") from exc
        if not directory.is_dir():
            raise FileNotFoundError(f"H1 package missing required calibration directory: {directory}")
        observed: dict[str, Path] = {}
        for candidate in sorted(directory.glob("*.nwb")):
            canonical = candidate.resolve()
            try:
                canonical.relative_to(directory)
            except ValueError as exc:
                raise ValueError(f"H1 package rejects calibration symlink escape: {candidate}") from exc
            if "minival" in canonical.name.lower():
                raise ValueError(f"H1 package rejects minival input: {canonical}")
            session = _session_from_h1_path(canonical)
            if session in observed:
                raise ValueError(f"duplicate H1 calibration session {session} in {directory}")
            observed[session] = canonical
        if set(observed) != set(expected_sessions) or len(observed) != len(expected_sessions):
            raise ValueError(
                f"H1 package {role} allowlist mismatch: observed={sorted(observed)} "
                f"expected={sorted(expected_sessions)}"
            )
        for session in expected_sessions:
            path = observed[session]
            if path in seen_paths:
                raise ValueError(f"H1 package calibration path reused across roles: {path}")
            seen_paths.add(path)
            rows.append(
                {
                    "role": role,
                    "session": session,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if len(rows) != 27:
        raise AssertionError(f"H1 package allowlist must contain exactly 27 files, got {len(rows)}")
    return rows


def _validate_run_binding(run_dir: Path, config: Path, checkpoint: Path, variant: str) -> None:
    expected_config = (run_dir / ".hydra" / "config.yaml").resolve()
    if config != expected_config:
        raise ValueError("H1 terminal package config must be the resolved .hydra/config.yaml under --run-dir")
    expected_checkpoint = (run_dir / "checkpoints" / "fixed_epoch50" / "epoch_049.ckpt").resolve()
    if checkpoint != expected_checkpoint:
        raise ValueError("H1 terminal package accepts only run-dir/checkpoints/fixed_epoch50/epoch_049.ckpt")
    if run_dir.name != variant:
        raise ValueError(f"H1 terminal package run directory must be named for its frozen variant {variant!r}")
    if not run_dir.parent.name.startswith("h1_baseline_staged_"):
        raise ValueError("H1 terminal package run directory is not under a staged H1 baseline root")


def _feature_shapes_and_payload_audit(payload: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("H1 package payload must be a mapping")
    top_level_keys = set(payload)
    if top_level_keys != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError(
            "H1 package payload top-level keys drifted: "
            f"observed={sorted(top_level_keys)} expected={sorted(EXPECTED_TOP_LEVEL_KEYS)}"
        )
    if payload["task"] != FalconTask.h1:
        raise ValueError(f"H1 package payload task is not FalconTask.h1: {payload['task']!r}")
    features = payload["calib_trial_features"]
    if not isinstance(features, dict):
        raise ValueError("H1 package must serialize calibration features as a dataset-tag mapping")
    task_config = FalconConfig(task=FalconTask.h1)
    expected_tags: dict[str, str] = {}
    for row in rows:
        tag = task_config.hash_dataset(Path(row["path"]).stem)
        if tag in expected_tags:
            raise ValueError(f"H1 package dataset hash collision for {tag}: {expected_tags[tag]} / {row['path']}")
        expected_tags[tag] = row["path"]
    if set(features) != set(expected_tags):
        raise ValueError(
            "H1 package calibration feature tags do not exactly match the allowlist: "
            f"observed={sorted(features)} expected={sorted(expected_tags)}"
        )

    feature_shapes: dict[str, dict[str, Any]] = {}
    for tag in sorted(features):
        value = features[tag]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"H1 package feature {tag} is not a numpy neural-feature array")
        if value.ndim != 3 or tuple(value.shape) != (2, 1024, 176):
            raise ValueError(f"H1 package feature {tag} has invalid shape {tuple(value.shape)}; expected [2,1024,176]")
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"H1 package feature {tag} has non-numeric dtype {value.dtype}")
        feature_shapes[tag] = {
            "source_path": expected_tags[tag],
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    # The only serialized calibration arrays are the explicit neural features.
    # A decoder object necessarily has learned tensor parameters and the scalar
    # behavior scaling hyperparameter; neither is target behavior data.  Check
    # all non-model payload branches recursively and prohibit target-bearing
    # names there, while top-level exact-key validation prevents an alternate
    # container from being introduced unnoticed.
    forbidden = ("covariate", "target", "label")

    def inspect(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                lowered = str(key).lower()
                if any(token in lowered for token in forbidden) or (
                    "behavior" in lowered and lowered != "behavior_scaling_factor"
                ):
                    raise ValueError(f"H1 package contains forbidden target/covariate branch {child_path}")
                inspect(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                inspect(child, f"{path}[{index}]")
        elif isinstance(value, (np.ndarray, torch.Tensor)) and not path.startswith("calib_trial_features"):
            raise ValueError(f"H1 package contains an unexpected non-model array at {path}")

    non_model = {key: value for key, value in payload.items() if key != "decoder"}
    inspect(non_model, "")
    decoder = payload["decoder"]
    if isinstance(decoder, dict):
        inspect(decoder, "decoder")
    elif hasattr(decoder, "calib_trial_features"):
        raise ValueError("H1 package decoder object must not retain calibration or behavior data")

    return {
        "top_level_keys": sorted(top_level_keys),
        "feature_count": len(feature_shapes),
        "feature_shapes": feature_shapes,
        "payload_top_level_exact": True,
        "recursive_non_model_target_or_covariate_branches_absent": True,
        "decoder_retains_no_calibration_features": not hasattr(decoder, "calib_trial_features"),
        "neural_calibration_features_only": True,
    }


def package_terminal_h1(
    *,
    data_dir: str | Path,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    expected_variant: str,
    package_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Create one immutable, allowlisted H1 decoder package and receipt."""

    if expected_variant not in EXPECTED_VARIANTS:
        raise ValueError(f"unknown predeclared H1 variant {expected_variant!r}")
    package = Path(package_path)
    receipt_output = Path(receipt_path)
    if package.exists():
        raise FileExistsError(f"refusing to overwrite H1 terminal package {package}")
    if receipt_output.exists():
        raise FileExistsError(f"refusing to overwrite immutable H1 package receipt {receipt_output}")
    run = Path(run_dir).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    config = Path(config_path).resolve()
    _validate_run_binding(run, config, checkpoint, expected_variant)
    config_tree = _validate_config(config, expected_variant)
    checkpoint_meta = _validate_terminal_checkpoint(checkpoint, expected_variant)
    if tuple(config_tree["data"].get("heldin_session_names", ())) != H1_BASELINE_HELDIN_SESSIONS:
        raise ValueError("H1 terminal package resolved config does not contain the canonical 13 held-in sessions")

    input_rows = _allowlisted_calibration_files(data_dir)
    package.parent.mkdir(parents=True, exist_ok=True)
    export_spint_decoder(
        task="h1",
        checkpoint_dir=str(checkpoint),
        calibration_dir=str(Path(data_dir).resolve()),
        save_path=str(package),
        max_trial_length=int(config_tree["data"]["max_trial_length"]),
        window_size=int(config_tree["data"]["window_size"]),
        use_calib_intertrials=bool(config_tree["data"].get("use_calib_intertrials", False)),
        calib_start_trial_idx=0,
        calib_n_trials=int(config_tree["data"]["calibration_n_trials"]),
        trial_feature_type=str(config_tree["data"].get("trial_feature_type", "raw")),
        behavior_scaling_factor=float(config_tree["model"]["behavior_scaling_factor"]),
        interpolate_trials=bool(config_tree["data"].get("interpolate_trials", False)),
        interpolate_trials_kind=str(config_tree["data"].get("interpolate_trials_kind", "linear")),
        smooth_calibration=bool(config_tree["data"].get("smooth_calibration", False)),
        calibration_files=[row["path"] for row in input_rows],
    )
    if not package.is_file():
        raise RuntimeError("H1 terminal exporter did not create the requested package")
    with package.open("rb") as handle:
        payload = pickle.load(handle)
    payload_audit = _feature_shapes_and_payload_audit(payload, input_rows)
    input_manifest = {
        "schema": "spint_h1_terminal_public_calibration_allowlist_v1",
        "heldin_calibration_recordings": len(H1_BASELINE_HELDIN_SESSIONS),
        "public_heldout_calibration_recordings": len(H1_BASELINE_HELDOUT_SESSIONS),
        "minival_included": False,
        "private_test_opened": False,
        "files": input_rows,
    }
    receipt = {
        "schema": PACKAGE_SCHEMA,
        "status": "PASS_H1_TERMINAL_PACKAGE_ALLOWLISTED",
        "protocol": PACKAGE_PROTOCOL,
        "protocol_variant": expected_variant,
        "comparison_role": EXPECTED_VARIANTS[expected_variant],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch_zero_based": checkpoint_meta["checkpoint_epoch_zero_based"],
        "epochs_completed": checkpoint_meta["epochs_completed"],
        "global_step": checkpoint_meta["global_step"],
        "checkpoint_optimizer_lr": checkpoint_meta["optimizer_lr"],
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "package_path": str(package.resolve()),
        "package_sha256": sha256_file(package),
        "input_manifest": input_manifest,
        "input_manifest_sha256": _canonical_sha256(input_manifest),
        "payload_audit": payload_audit,
        "source_sha256": _source_hashes(),
        "selection_guards": {
            "validation_epoch_selection": False,
            "minival_used_for_epoch_selection": False,
            "minival_used_for_variant_selection": False,
            "minival_used_for_package_selection": False,
            "private_test_used_for_epoch_selection": False,
            "private_test_used_for_variant_selection": False,
            "private_test_used_for_package_selection": False,
            "private_test_opened": False,
            "evalai_submission_performed": False,
        },
        "calibration_boundary": {
            "public_heldout_calibration_is_allowed_for_neural_only_identity_features": True,
            "dense_behavior_or_covariate_targets_serialized": False,
            "formal_private_test_labels_opened": False,
        },
    }
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_output.chmod(0o444)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an allowlisted terminal H1 SPINT package; never submits EvalAI.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(EXPECTED_VARIANTS))
    parser.add_argument("--package", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    receipt = package_terminal_h1(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        expected_variant=args.variant,
        package_path=args.package,
        receipt_path=args.receipt,
    )
    print(json.dumps({"status": receipt["status"], "package_sha256": receipt["package_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
