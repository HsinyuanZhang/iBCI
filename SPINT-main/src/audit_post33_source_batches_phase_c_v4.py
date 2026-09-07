#!/usr/bin/env python3
"""Count exact SPINT source-only batches without opening the outer session."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from falcon_challenge.config import FalconConfig, FalconTask

from src.data.falcon_datamodule import FalconDataset, SessionBatchSampler
from src.data.falcon_post33_confirm_v1_datamodule import (
    FOLDS,
)
from src.data.falcon_post33_confirm_v4_datamodule import M2Post33ConfirmSPINTDataModuleV4

def _row(data_root: Path, fold: int) -> dict[str, object]:
    dm = M2Post33ConfirmSPINTDataModuleV4(
        deployment_constants_path=str(data_root / "source-audit-does-not-read-deployment-constants.json"),
        seed=42,
        task="m2", data_dir=str(data_root), validation_protocol="loso",
        loso_fold=fold, calibration_n_trials=33, heldin_query_start_trial=33,
        random_calibration=False, include_heldout_in_fit=False,
        include_heldout_in_test=False, query_start_trial=0,
        heldin_query_end_trial=None, batch_size=32, window_size=50,
        smooth_calibration=False, max_trial_length=100,
        standardize_covariates=False, use_intertrials=True,
        use_calib_intertrials=False, trial_feature_type="raw",
        remove_still_times=False, remove_calib_still_times=False,
        use_calib_active_segments=False, calib_n_active_segments=1,
        interpolate_trials=True, interpolate_trials_kind="cubic",
        pad_value=-1.0, num_workers=0, pin_memory=False,
    )
    dm.setup("fit")
    expected_access = {
        "stage": "fit", "source_files_opened": 12,
        "outer_calibration_files_opened": 0, "formal_files_opened": 0,
        "outer_directional_label_accesses": 0, "outer_query_batch_calls": 0,
        "scorer_calls": 0,
    }
    if dm.phase_c_stage_access_evidence != expected_access:
        raise ValueError("SPINT v4 source audit stage isolation failed")
    source_calib_files = dm._role_files(
        "sub-MonkeyN-held-in-calib", "held-in-calib", dm.source_session_names
    )
    source_minival_files = dm._role_files(
        "sub-MonkeyN-held-in-minival", "held-in-minival", dm.source_session_names
    )
    dm._approved_nwb_paths = set(source_calib_files + source_minival_files)
    source_calib: OrderedDict[str, dict] = OrderedDict()
    source_minival: OrderedDict[str, dict] = OrderedDict()
    covariates_mean = covariates_std = None
    for index, path in enumerate(source_calib_files):
        session = dm._session_from_path(path)
        data = dm.prepare_session_data(
            path, FalconTask.m2, standardize_covariates=False,
            covariates_mean=None if index == 0 else covariates_mean,
            covariates_std=None if index == 0 else covariates_std,
            use_intertrials=True,
        )
        if index == 0:
            covariates_mean, covariates_std = data["covariates_mean"], data["covariates_std"]
        source_calib[session] = data
    for path in source_minival_files:
        session = dm._session_from_path(path)
        source_minival[session] = dm.prepare_session_data(
            path, FalconTask.m2, standardize_covariates=False,
            covariates_mean=covariates_mean, covariates_std=covariates_std,
            use_intertrials=True,
        )
    common = dict(
        window_size=50, calibration_n_trials=33, smooth_calibration=False,
        max_trial_length=100, use_calib_intertrials=False,
        trial_feature_type="raw", remove_still_times=False,
        remove_calib_still_times=False, use_calib_active_segments=False,
        calib_n_active_segments=1, interpolate_trials=True,
        interpolate_trials_kind="cubic", pad_value=-1.0,
    )
    train = FalconDataset(
        sessions_dict=source_calib, calib_sessions_dict=source_calib,
        split="train", random_calibration=False, **common,
    )
    val = FalconDataset(
        sessions_dict=source_minival, calib_sessions_dict=source_calib,
        split="val_heldin", random_calibration=False, **common,
    )
    train_sampler = SessionBatchSampler(train, 32, shuffle=True)
    val_sampler = SessionBatchSampler(val, 32, shuffle=False)
    if (
        len(dm.train_dataset) != len(train) or len(dm.val_heldin_dataset) != len(val)
        or len(dm.train_batch_sampler) != len(train_sampler)
        or len(dm.val_heldin_batch_sampler) != len(val_sampler)
    ):
        raise ValueError("SPINT v4 source-only path differs from audit reconstruction")
    expected_sources = list(session for index, session in FOLDS.items() if index != fold)
    if list(source_calib) != expected_sources or list(source_minival) != expected_sources:
        raise ValueError("source-only audit session order mismatch")
    input_files = []
    for role, paths in (("source_calib", source_calib_files), ("source_minival", source_minival_files)):
        for path in paths:
            input_files.append({
                "role": role, "session": dm._session_from_path(path),
                "canonical_path": str(path), "size_bytes": path.stat().st_size,
            })
    train_windows_by_session = {
        session: len(train_sampler.session_to_indices[session]) for session in expected_sources
    }
    val_windows_by_session = {
        session: len(val_sampler.session_to_indices[session]) for session in expected_sources
    }
    train_channels_by_session = {
        session: int(source_calib[session]["neural"].shape[1]) for session in expected_sources
    }
    val_channels_by_session = {
        session: int(source_minival[session]["neural"].shape[1]) for session in expected_sources
    }
    if set(train_channels_by_session.values()) != {96} or set(val_channels_by_session.values()) != {96}:
        raise ValueError("native M2 source role contains non-96-channel data")
    return {
        "fold": fold,
        "outer_session": FOLDS[fold],
        "source_sessions": expected_sources,
        "batch_size": 32,
        "train_windows": len(train),
        "train_full_batches_per_epoch": len(train_sampler),
        "source_validation_windows": len(val),
        "source_validation_full_batches_per_epoch": len(val_sampler),
        "train_windows_by_session": train_windows_by_session,
        "source_validation_windows_by_session": val_windows_by_session,
        "train_batches_by_session": {
            session: windows // 32 for session, windows in train_windows_by_session.items()
        },
        "source_validation_batches_by_session": {
            session: windows // 32 for session, windows in val_windows_by_session.items()
        },
        "train_channels_by_session": train_channels_by_session,
        "source_validation_channels_by_session": val_channels_by_session,
        "stage_access_evidence": expected_access,
        "source_input_files": input_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve(strict=True)
    if data_root.name != "000953":
        raise ValueError("exact native-M2 000953 data root required")
    payload = {
        "schema": "m2_post33_phase_c_source_batch_arm_audit_v4",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "arm": "spint",
        "fold_outer_role_included": False,
        "scorer_imported": False,
        "formal_data_accessed": False,
        "folds": {str(fold): _row(data_root, fold) for fold in FOLDS},
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
