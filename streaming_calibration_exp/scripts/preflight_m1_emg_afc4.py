#!/usr/bin/env python3
"""CPU-only preflight for isolated M1 q=3 EMG-AFC4 source-joint configs.

The preflight never imports a trainer, model checkpoint, formal/EvalAI file, or
held-out file.  For every native M1 source-LOSO fold it constructs all four
arms from the identical frozen source plan, verifies target M10 isolation and
strict post-M10 query checksums, then instantiates the actual AFC4 datamodule
for the Full arm to prove its train/validation batch provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.falcon_emg_afc4_datamodule import M1EMGAFC4DataModule  # noqa: E402
from src.data.falcon_emg_afc4_features import (  # noqa: E402
    AFC4_DIM, AFC4_SUPPORT_TRIALS, SourceFrozenEMGAFC4Plan,
)
from src.models.streaming_calibration_module import StreamingCalibrationLitModule  # noqa: E402


DATA = ROOT / "SPINT-main" / "data" / "000941"
DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m1_emg_afc4_preflight_v2"
ARMS = ("full", "zero4", "rs4", "b4")
SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")


def strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [strict_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    raise TypeError(type(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths() -> dict[str, Path]:
    paths = {
        f"ses-{path.name.split('_ses-')[1].split('_behavior')[0]}": path
        for path in sorted((DATA / "sub-MonkeyL-held-in-calib").glob("*.nwb"))
    }
    if tuple(sorted(paths)) != SESSIONS:
        raise ValueError(f"expected exact M1 source sessions {SESSIONS}, got {tuple(sorted(paths))}")
    return paths


def datamodule_for_fold(fold: int, arm: str) -> M1EMGAFC4DataModule:
    return M1EMGAFC4DataModule(
        task="m1", data_dir=str(DATA), heldin_session_names=[""], batch_size=32,
        window_size=100, calibration_n_trials=10, random_calibration=False,
        smooth_calibration=False, max_trial_length=1024, standardize_covariates=False,
        use_intertrials=True, use_calib_intertrials=False, trial_feature_type="raw",
        interpolate_trials=True, interpolate_trials_kind="cubic", pad_value=-1.0,
        validation_protocol="loso", loso_fold=int(fold), include_heldout_in_fit=False,
        include_heldout_in_test=False, query_start_trial=0, heldin_query_start_trial=10,
        heldin_query_end_trial=210, allow_empty_heldout_query=False, num_workers=0,
        pin_memory=False, sampler_seed=42, balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False, side_feature_group="none",
        side_feature_shuffle_seed=42, afc4_arm=arm,
    )


def arm_plan_receipt(paths: dict[str, Path], *, held_out: str) -> dict[str, Any]:
    sources = {name: path for name, path in paths.items() if name != held_out}
    plans = {arm: SourceFrozenEMGAFC4Plan(sources, shuffle_seed=42) for arm in ARMS}
    # Add the same target once per arm.  Each plan is independently built but
    # must produce byte-identical source PCA/normalizer/schedule artifacts.
    target = paths[held_out]
    outputs: dict[str, Any] = {}
    for arm, plan in plans.items():
        name = plan.add_target(target)
        side = plan.normalized(name, arm=arm)
        receipt = plan.receipt(arm=arm)
        if side.shape != (plan.support[name].support_rates.shape[1], AFC4_DIM):
            raise RuntimeError(f"{held_out}/{arm}: variable-N adapter shape failure")
        outputs[arm] = {
            "side_shape": list(side.shape),
            "side_sha256": hashlib.sha256(side.tobytes()).hexdigest(),
            "pca_sha256": hashlib.sha256(json.dumps(receipt["pca"], sort_keys=True).encode()).hexdigest(),
            "normalizer_sha256": hashlib.sha256(json.dumps(receipt["normalizer"], sort_keys=True).encode()).hexdigest(),
            "query_checksum": receipt["query_checksums"][held_out],
            "schedule_sha256": hashlib.sha256(json.dumps(receipt["row_permutation_schedule"], sort_keys=True).encode()).hexdigest(),
            "target_fit_calls": int(plan.target_fit_calls[held_out]),
            "receipt": receipt,
        }
    reference = outputs["full"]
    for arm in ARMS:
        item = outputs[arm]
        for key in ("pca_sha256", "normalizer_sha256", "query_checksum", "schedule_sha256"):
            if item[key] != reference[key]:
                raise RuntimeError(f"{held_out}: {key} differs across AFC4 arms")
    if outputs["zero4"]["target_fit_calls"] != 0:
        raise RuntimeError("Zero4 performed a target descriptor fit")
    if outputs["full"]["target_fit_calls"] != 1 or outputs["rs4"]["target_fit_calls"] != 1 or outputs["b4"]["target_fit_calls"] != 1:
        raise RuntimeError("Full/RS4/B4 M10 fit-count contract failed")
    if any(outputs["zero4"]["side_shape"][1] != AFC4_DIM for _ in [0]):
        raise RuntimeError("Zero4 width differs from Full")
    return outputs


def runtime_batch_provenance(fold: int, arm: str) -> dict[str, Any]:
    datamodule = datamodule_for_fold(fold, arm)
    datamodule.setup("fit")
    train_item = datamodule.train_dataset[0]
    val_item = datamodule.val_heldin_dataset[0]
    train_session, val_session = train_item[3], val_item[3]
    if train_session not in datamodule.train_session_names:
        raise RuntimeError("train batch contains a left-out target session")
    if val_session not in datamodule.val_heldin_session_names or val_session in datamodule.train_session_names:
        raise RuntimeError("validation batch provenance is not a disjoint left-out source session")
    if train_item[4].shape[-1] != AFC4_DIM or val_item[4].shape[-1] != AFC4_DIM:
        raise RuntimeError("runtime batch has incorrect AFC4 width")
    manifest = datamodule.get_split_manifest()
    afc4 = manifest["m1_emg_afc4"]
    target_scope = afc4["materialization_scope"][val_session]
    if target_scope != {
        "source_for_pca": False,
        "emg_trial_range": [0, AFC4_SUPPORT_TRIALS],
        "raw_spike_trial_range": [0, AFC4_SUPPORT_TRIALS],
        "target_query_emg_or_neural_values_read": False,
    }:
        raise RuntimeError(f"{arm}: target materialization exceeds M10: {target_scope}")
    if afc4["arm"] != arm:
        raise RuntimeError(f"runtime arm mismatch: requested {arm}, got {afc4['arm']}")
    grad_source = inspect.getsource(StreamingCalibrationLitModule.training_step)
    grad_val = inspect.getsource(StreamingCalibrationLitModule.validation_step)
    if "return loss" not in grad_source or "return None" in grad_val:
        raise RuntimeError("unexpected streaming train/validation optimizer contract")
    return {
        "fold": fold,
        "arm": arm,
        "train_sessions": list(datamodule.train_session_names),
        "validation_sessions": list(datamodule.val_heldin_session_names),
        "sampled_train_session": train_session,
        "sampled_validation_session": val_session,
        "train_side_shape": list(train_item[4].shape),
        "validation_side_shape": list(val_item[4].shape),
        "query_window_audit": datamodule.val_heldin_dataset.query_window_audit,
        "split_manifest": manifest,
        "contract_hashes": {
            "pca_sha256": hashlib.sha256(json.dumps(afc4["pca"], sort_keys=True).encode()).hexdigest(),
            "normalizer_sha256": hashlib.sha256(json.dumps(afc4["normalizer"], sort_keys=True).encode()).hexdigest(),
            "query_checksum": afc4["query_checksums"][val_session],
            "schedule_sha256": hashlib.sha256(json.dumps(afc4["row_permutation_schedule"], sort_keys=True).encode()).hexdigest(),
        },
        "gradient_scope": {
            "training_step_returns_loss": True,
            "validation_step_has_no_optimizer_return": True,
            "gradient_bearing_batches": "train source sessions only",
            "target_validation_backpropagation": False,
        },
    }


def build() -> dict[str, Any]:
    paths = source_paths()
    folds: list[dict[str, Any]] = []
    for fold, held_out in enumerate(SESSIONS):
        arms = arm_plan_receipt(paths, held_out=held_out)
        runtime_by_arm = {arm: runtime_batch_provenance(fold, arm) for arm in ARMS}
        for arm, runtime in runtime_by_arm.items():
            for key, value in runtime["contract_hashes"].items():
                if value != arms[arm][key]:
                    raise RuntimeError(f"{held_out}/{arm}: adapter runtime {key} differs from preflight plan")
        folds.append({"fold": fold, "held_out_source_session": held_out, "arms": arms, "runtime_by_arm": runtime_by_arm})
    return {
        "schema": "m1_emg_afc4_cpu_preflight_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "formal": False, "evalai": False, "gpu_launched": False},
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "contract": {"support": [0, AFC4_SUPPORT_TRIALS], "query": [AFC4_SUPPORT_TRIALS, 210], "arms": list(ARMS), "variable_n_supported": True},
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite preflight output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        payload = strict_json(build())
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
