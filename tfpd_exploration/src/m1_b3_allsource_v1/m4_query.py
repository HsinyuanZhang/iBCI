"""Local M4-calib / remaining-query diagnostic for all-source B3 students.

Chronological first-4 trials build identity. Query windows start at trial 4.
Default surface is the three later-day 10-trial public calib files (6 query
trials). Source-session remaining-query is in-train and off by default.
This is not EvalAI hidden held-out.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import plan
from . import receipts


class M4QueryError(RuntimeError):
    """Fail closed for the local M4 remaining-query diagnostic."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M4QueryError(message)


SUPPORT_TRIALS = int(plan.M4_QUERY_SUPPORT)
QUERY_START = int(plan.M4_QUERY_START)
SCHEMA = "m1_b3_allsource_m4_query_v1"
EVAL_BATCH_SIZE = 32


def last_bin_r2(predictions: Any, targets: Any) -> float:
    """Lightning val metric: last-bin variance-weighted R²."""
    import torch
    from torchmetrics.regression import R2Score

    prediction = torch.as_tensor(predictions, dtype=torch.float32)
    target = torch.as_tensor(targets, dtype=torch.float32)
    _require(
        prediction.ndim == target.ndim == 2
        and tuple(prediction.shape) == tuple(target.shape)
        and prediction.shape[0] > 0
        and prediction.shape[1] >= 1
        and bool(torch.isfinite(prediction).all())
        and bool(torch.isfinite(target).all()),
        f"M4 query R2 tensor shape/finite drift: pred={tuple(prediction.shape)} target={tuple(target.shape)}",
    )
    metric = R2Score(multioutput="variance_weighted")
    value = metric(prediction, target)
    _require(bool(torch.isfinite(value)), "M4 query R2 is nonfinite")
    return float(value.detach().cpu())


def session_role(session_name: str) -> str:
    if session_name in plan.SOURCE_SESSION_NAMES:
        return "source_in_train"
    later = tuple(f"ses-{item}" for item in plan.LATER_CALIB_SESSIONS)
    _require(session_name in later, f"unexpected M4 query session {session_name}")
    return "later_day_public_calib"


def selected_sessions(mapping: Mapping[str, Path], *, include_source_in_train: bool) -> OrderedDict[str, Path]:
    later = tuple(f"ses-{item}" for item in plan.LATER_CALIB_SESSIONS)
    names = list(later)
    if include_source_in_train:
        names = list(plan.SOURCE_SESSION_NAMES) + names
    selected: OrderedDict[str, Path] = OrderedDict()
    for name in names:
        _require(name in mapping, f"missing public calib {name}")
        selected[name] = Path(mapping[name])
    return selected


def score_arm(
    repo_root: Path,
    arm: str,
    *,
    include_source_in_train: bool = plan.M4_QUERY_INCLUDE_SOURCE_IN_TRAIN_DEFAULT,
) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _require(arm in plan.M4_QUERY_ARMS, f"M4 query is not defined for {arm}")
    from . import package as package_module

    package_module._ensure_streaming_paths(repo_root)
    spec = plan.ARM_SPECS[arm]
    hydra_run = package_module._load_train_hydra(repo_root, arm)
    ckpt = Path(hydra_run["student_checkpoint"])
    _require(ckpt.is_file(), f"student checkpoint missing: {ckpt}")
    _require(
        package_module.sha256_file(ckpt) == hydra_run["student_checkpoint_sha256"],
        "student checkpoint drift",
    )
    teacher = Path(repo_root) / plan.TEACHER_CHECKPOINT_RELATIVE
    _require(package_module.sha256_file(teacher) == plan.TEACHER_SHA256, "teacher checkpoint drift")
    hydra_dir = Path(hydra_run["hydra_output_dir"])
    resolved = hydra_dir / ".hydra" / "config.yaml"
    _require(resolved.is_file(), f"resolved hydra config missing: {resolved}")
    data_dir = Path(repo_root) / plan.DATA_DIR_RELATIVE
    mapping = package_module.calibration_file_map(data_dir)
    sessions = selected_sessions(mapping, include_source_in_train=bool(include_source_in_train))

    import torch
    from falcon_challenge.config import FalconConfig, FalconTask
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from src.data.falcon_datamodule import FalconDataset

    config = OmegaConf.load(resolved)
    _require(str(config.data.task).lower() == "m1", "resolved config is not native M1")
    _require(int(config.data.calibration_n_trials) == plan.CALIBRATION_N_TRIALS, "train M10 drift")
    _require(str(config.model.variant) == spec["variant"], "variant drift")
    _require(str(config.data.side_feature_group).lower() == spec["side_feature_group"], "side group drift")
    config.model.teacher_ckpt_path = str(teacher.resolve())
    config.data.data_dir = str(data_dir.resolve())
    config.data.num_workers = 0
    config.data.pin_memory = False

    model = instantiate(config.model)
    model.setup("fit")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    _require(isinstance(state, dict) and isinstance(state.get("state_dict"), dict), "not a Lightning ckpt")
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    student = model.student
    _require(student is not None and student.decoder_mode == "coupled", "student is not coupled SPINT")

    datamodule = instantiate(config.data)
    group = spec["side_feature_group"]
    neural_group = package_module.neural_loader_side_group(group)
    rsyn3_fitted = None
    if group == "rsyn3":
        rsyn3_fitted = package_module._load_rsyn3_bank(hydra_dir)
        _require(
            list(rsyn3_fitted.get("source_sessions", [])) == list(plan.SOURCE_SESSION_NAMES),
            "rSyn3 bank was not fit on the four source sessions",
        )
        _require(rsyn3_fitted.get("later_day_in_fit") is False, "later-day entered the NMF fit")

    task = FalconConfig(task=FalconTask.m1).task
    session_rows: list[dict[str, Any]] = []
    later_preds: list[np.ndarray] = []
    later_targets: list[np.ndarray] = []
    with torch.inference_mode():
        covariates_mean = covariates_std = None
        for index, (name, path) in enumerate(sessions.items()):
            record = datamodule.prepare_session_data(
                path,
                task,
                standardize_covariates=bool(config.data.standardize_covariates),
                covariates_mean=covariates_mean,
                covariates_std=covariates_std,
                use_intertrials=bool(config.data.use_intertrials),
                include_trial_targets=False,
            )
            if index == 0:
                covariates_mean, covariates_std = record["covariates_mean"], record["covariates_std"]
            dataset = FalconDataset(
                sessions_dict=OrderedDict([(name, record)]),
                calib_sessions_dict=OrderedDict([(name, record)]),
                window_size=int(config.data.window_size),
                split=None,
                calibration_n_trials=SUPPORT_TRIALS,
                random_calibration=False,
                smooth_calibration=bool(config.data.smooth_calibration),
                max_trial_length=int(config.data.max_trial_length),
                use_calib_intertrials=bool(config.data.use_calib_intertrials),
                trial_feature_type=str(config.data.trial_feature_type),
                remove_still_times=bool(config.data.remove_still_times),
                remove_calib_still_times=bool(config.data.remove_calib_still_times),
                use_calib_active_segments=bool(config.data.use_calib_active_segments),
                calib_n_active_segments=int(config.data.calib_n_active_segments),
                interpolate_trials=bool(config.data.interpolate_trials),
                interpolate_trials_kind=str(config.data.interpolate_trials_kind),
                pad_value=float(config.data.pad_value),
                side_feature_group=neural_group,
                query_start_trial=QUERY_START,
            )
            audit = dataset.query_window_audit[name]
            n_trials = int(audit["total_trials"])
            _require(int(audit["query_start_trial"]) == QUERY_START, "query start drift")
            _require(int(dataset.calib_n_trials[name]) == SUPPORT_TRIALS, "M4 identity budget drift")
            _require(n_trials > QUERY_START, f"{name} has no remaining query trials")
            _require(len(dataset) == int(audit["eligible_windows"]), "window/audit mismatch")
            calibration_np = np.asarray(
                dataset.calib_trialized_neural_features[name][:SUPPORT_TRIALS],
                dtype=np.float32,
            )
            _require(
                calibration_np.shape == (SUPPORT_TRIALS, plan.TRIAL_BINS, plan.CHANNELS),
                f"unexpected M4 calibration shape for {name}: {calibration_np.shape}",
            )
            side = None
            side_np = None
            if group == "rsyn3":
                from . import rsyn3_bank as bank_module

                _require(rsyn3_fitted is not None, "rSyn3 bank was not loaded")
                side_np = np.ascontiguousarray(
                    bank_module.encode_public_session(
                        path, rsyn3_fitted, support_trials=SUPPORT_TRIALS,
                    ),
                    dtype=np.float32,
                )
                _require(
                    side_np.shape == (plan.CHANNELS, 4),
                    f"unexpected M4 rSyn3 side shape for {name}: {side_np.shape}",
                )
                side = torch.from_numpy(side_np).unsqueeze(0)
            calibration = torch.from_numpy(calibration_np).unsqueeze(0)
            identity = student.compute_identity(calibration, side_features=side)
            _require(
                tuple(identity.shape) == (1, plan.CHANNELS, plan.IDENTITY_DIM),
                f"identity shape {tuple(identity.shape)}",
            )
            loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
            preds: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            for batch in loader:
                neural = batch[0].float() if hasattr(batch[0], "float") else torch.from_numpy(
                    np.asarray(batch[0], dtype=np.float32)
                )
                target = batch[1].float() if hasattr(batch[1], "float") else torch.from_numpy(
                    np.asarray(batch[1], dtype=np.float32)
                )
                identity_batch = identity.expand(neural.shape[0], -1, -1)
                pred, _ = student(neural, identity=identity_batch)
                preds.append(pred[:, -1, :].cpu().numpy())
                targets.append(target[:, -1, :].cpu().numpy())
            _require(preds, f"{name} has no remaining-query windows")
            pred_all = np.concatenate(preds, axis=0)
            target_all = np.concatenate(targets, axis=0)
            _require(pred_all.shape[0] == len(dataset), f"{name} scored window count drift")
            _require(pred_all.shape[1] == target_all.shape[1] == 16, f"{name} M1 EMG output dim {pred_all.shape}")
            score = last_bin_r2(pred_all, target_all)
            role = session_role(name)
            if role == "later_day_public_calib":
                _require(n_trials == 10, f"{name} later-day file is not 10 trials")
                _require(int(audit["query_trials"]) == plan.M4_QUERY_LATER_DAY_QUERY_TRIALS, "later-day query trial count")
                later_preds.append(pred_all)
                later_targets.append(target_all)
            row = {
                "session": name,
                "role": role,
                "n_trials": n_trials,
                "support_trials": SUPPORT_TRIALS,
                "query_start_trial": QUERY_START,
                "query_trials": int(audit["query_trials"]),
                "n_windows": int(len(dataset)),
                "n_outputs": int(pred_all.shape[1]),
                "r2_variance_weighted_last_bin": score,
                "in_train_windows": role == "source_in_train",
                "formal_benchmark_verdict": False,
            }
            if side_np is not None:
                row["rsyn3_encode_support_trials"] = SUPPORT_TRIALS
                row["rsyn3_nmf_refit"] = False
            session_rows.append(row)

    _require(any(row["role"] == "later_day_public_calib" for row in session_rows), "no later-day rows")
    later_rows = [row for row in session_rows if row["role"] == "later_day_public_calib"]
    later_mean = float(np.mean([row["r2_variance_weighted_last_bin"] for row in later_rows]))
    later_pooled = last_bin_r2(np.concatenate(later_preds, axis=0), np.concatenate(later_targets, axis=0))
    return {
        "schema": SCHEMA,
        "arm": arm,
        "variant": spec["variant"],
        "side_feature_group": group,
        "support_trials": SUPPORT_TRIALS,
        "query_start_trial": QUERY_START,
        "include_source_in_train": bool(include_source_in_train),
        "student_checkpoint": str(ckpt),
        "student_checkpoint_sha256": hydra_run["student_checkpoint_sha256"],
        "teacher_sha256": plan.TEACHER_SHA256,
        "sessions": session_rows,
        "later_day": {
            "n_sessions": len(later_rows),
            "query_trials_per_session": plan.M4_QUERY_LATER_DAY_QUERY_TRIALS,
            "n_outputs": 16,
            "mean_session_r2": later_mean,
            "pooled_r2": later_pooled,
            "n_windows": int(sum(row["n_windows"] for row in later_rows)),
        },
        "note": plan.M4_QUERY_NOT_OFFICIAL,
        "formal_benchmark_verdict": False,
        "official_original_heldout_r2": plan.OFFICIAL_ORIGINAL_HELDOUT_R2,
        "local_20120924_is_not_official_heldout": True,
    }


def execute(
    repo_root: Path,
    arm: str,
    *,
    include_source_in_train: bool = plan.M4_QUERY_INCLUDE_SOURCE_IN_TRAIN_DEFAULT,
) -> tuple[dict[str, str], str | None, str | None]:
    repo_root = Path(repo_root).absolute()
    _require(os.environ.get("PYTHONNOUSERSITE") == "1", "execute requires PYTHONNOUSERSITE=1")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES", None) in {None, ""}, "M4 query is CPU-only")
    _require(arm in plan.M4_QUERY_ARMS, f"M4 query is not defined for {arm}")
    receipts.refuse_sealed_roots(repo_root / plan.RESULT_ROOT_RELATIVE)
    parent = repo_root / plan.M4_QUERY_ROOT_RELATIVE
    parent.mkdir(parents=True, exist_ok=True)
    progress_state: dict[str, object] = {
        "mode": "m4_query",
        "arm": arm,
        "include_source_in_train": bool(include_source_in_train),
    }

    def launch_builder() -> dict[str, object]:
        return {
            "schema": "m1_b3_allsource_m4_query_launch_v1",
            "arm": arm,
            "status": "LAUNCHED",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "include_source_in_train": bool(include_source_in_train),
            "support_trials": SUPPORT_TRIALS,
            "query_start_trial": QUERY_START,
            "formal_benchmark_verdict": False,
        }

    def body_publisher(artifact: Any) -> dict[str, str]:
        payload = score_arm(
            repo_root, arm, include_source_in_train=bool(include_source_in_train),
        )
        progress_state["later_day_mean_session_r2"] = payload["later_day"]["mean_session_r2"]
        progress_state["later_day_pooled_r2"] = payload["later_day"]["pooled_r2"]
        return {"m4_query.json": artifact.publish_json("m4_query.json", payload)}

    def terminal_builder(shas: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema": "m1_b3_allsource_m4_query_terminal_v1",
            "status": "COMPLETE",
            "mode": "m4_query",
            "arm": arm,
            "include_source_in_train": bool(include_source_in_train),
            "support_trials": SUPPORT_TRIALS,
            "query_start_trial": QUERY_START,
            "formal_benchmark_verdict": False,
            "local_20120924_is_not_official_heldout": True,
            "official_original_heldout_r2": plan.OFFICIAL_ORIGINAL_HELDOUT_R2,
            "later_day_mean_session_r2": progress_state.get("later_day_mean_session_r2"),
            "later_day_pooled_r2": progress_state.get("later_day_pooled_r2"),
            "bodies": {key: value for key, value in shas.items() if not str(key).startswith("_")},
        }

    stage = Path(plan.m4_query_root_relative(arm)).relative_to(plan.RESULT_ROOT_RELATIVE)
    return receipts.run_stage0(
        repo_root / plan.RESULT_ROOT_RELATIVE,
        relative=str(stage),
        attempt_payload={
            "schema": "m1_b3_allsource_m4_query_attempt_v1",
            "cell": plan.CELL,
            "phase": plan.PHASE,
            "mode": "m4_query",
            "arm": arm,
            "status": "ATTEMPT_RESERVED",
            "include_source_in_train": bool(include_source_in_train),
            "data_or_model_accessed": False,
            "formal_benchmark_verdict": False,
        },
        launch_builder=launch_builder,
        body_publisher=body_publisher,
        terminal_builder=terminal_builder,
        progress=lambda: dict(progress_state),
    )
