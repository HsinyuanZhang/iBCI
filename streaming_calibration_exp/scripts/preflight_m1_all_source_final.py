#!/usr/bin/env python3
"""CPU-only preflight for the clean all-source M1 teacher/Full/B4 path.

This command loads only the four held-in calibration NWBs.  It verifies the
train-only batch contract, fixed M10 trialization, source-frozen AFC4 width and
the shared Full/B4 basis.  It never creates a model, starts a trainer/GPU, or
opens minival, held-out, formal, EvalAI, query, or hidden evaluation files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "streaming_calibration_exp"
sys.path.insert(0, str(PACKAGE_ROOT))

from src.data.falcon_m1_all_source_emg_afc4_datamodule import (  # noqa: E402
    M1AllSourceEMGAFC4DataModule,
)
from src.data.falcon_m1_all_source_datamodule import M1_ALL_SOURCE_SESSIONS  # noqa: E402


DATA = ROOT / "SPINT-main" / "data" / "000941"
DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m1_all_source_final_preflight_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(data_dir: Path) -> dict[str, object]:
    module = M1AllSourceEMGAFC4DataModule(
        task="m1",
        data_dir=data_dir,
        source_session_names=list(M1_ALL_SOURCE_SESSIONS),
        heldin_session_names=list(M1_ALL_SOURCE_SESSIONS),
        batch_size=32,
        window_size=100,
        calibration_n_trials=10,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        remove_still_times=False,
        remove_calib_still_times=False,
        use_calib_active_segments=False,
        calib_n_active_segments=1,
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        validation_protocol="all_source",
        loso_fold=None,
        rotation_id=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=0,
        heldin_query_end_trial=None,
        allow_empty_heldout_query=False,
        num_workers=0,
        pin_memory=False,
        sampler_seed=42,
        balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False,
        side_feature_group="none",
        side_feature_shuffle_seed=42,
        afc4_arm="full",
    )
    module.setup("fit")
    batch = next(iter(module.train_dataloader()))
    batch_sessions = tuple(sorted(set(batch[3])))
    if not set(batch_sessions).issubset(set(M1_ALL_SOURCE_SESSIONS)):
        raise RuntimeError(f"non-source session escaped all-source guard: {batch_sessions}")
    if module.val_dataloader() != [] or module.val_heldin_dataset is not None or module.val_heldout_dataset is not None:
        raise RuntimeError("all-source module constructed a validation dataset")
    try:
        module.test_dataloader()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("all-source module did not fail closed for test_dataloader")

    plan = module.afc4_plan
    full = {name: plan.normalized(name, arm="full") for name in M1_ALL_SOURCE_SESSIONS}
    b4 = {name: plan.normalized(name, arm="b4") for name in M1_ALL_SOURCE_SESSIONS}
    for name in M1_ALL_SOURCE_SESSIONS:
        if full[name].shape != (64, 4) or b4[name].shape != (64, 4):
            raise RuntimeError(f"{name}: AFC4 width/units mismatch")
        if not np.array_equal(b4[name][:, :3], np.zeros((64, 3), dtype=np.float32)):
            raise RuntimeError(f"{name}: B4 did not zero normalized q=3 coordinates")
        if not np.array_equal(full[name][:, 3], b4[name][:, 3]):
            raise RuntimeError(f"{name}: B4 baseline coordinate differs from Full")

    source_files = dict(module.source_paths)
    if tuple(source_files) != M1_ALL_SOURCE_SESSIONS:
        raise RuntimeError(f"could not resolve all source file receipts: {source_files}")
    manifest = module.get_split_manifest()
    return {
        "schema": "m1_all_source_final_cpu_preflight_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_launched": False,
            "formal": False,
            "evalai": False,
            "minival_opened": False,
            "heldout_opened": False,
            "query_values_read": False,
        },
        "source_sessions": list(M1_ALL_SOURCE_SESSIONS),
        "source_files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in source_files.items()
        },
        "batch": {
            "sessions": list(batch_sessions),
            "neural_shape": list(batch[0].shape),
            "calibration_shape": list(batch[2].shape),
            "side_shape": list(batch[4].shape),
        },
        "contract": {
            "calibration_n_trials": 10,
            "query_start_trial": 0,
            "teacher_epochs": 20,
            "student_epochs": 12,
            "arms": ["full", "b4"],
            "shared_source_basis": True,
            "loss_mode": "task_only",
            "lambda_y": 0.0,
            "lambda_E": 0.0,
            "target_backpropagation": False,
        },
        "split_manifest": manifest,
        "afc4_receipt": {
            "full": module.afc4_plan.receipt(arm="full"),
            "b4": module.afc4_plan.receipt(arm="b4"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite preflight output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        payload = build(args.data_dir.resolve())
        output = args.out / "preflight.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.out / "preflight.sha256").write_text(f"{sha256(output)}  preflight.json\n", encoding="utf-8")
        print(output)
    except Exception:
        for child in args.out.iterdir():
            child.unlink()
        args.out.rmdir()
        raise


if __name__ == "__main__":
    main()
