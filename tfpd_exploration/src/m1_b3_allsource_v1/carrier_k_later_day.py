"""Later-day remaining-query carrier-k probe.

CPU only. Ranking R² is later-day public-calib last-6 after chronological
first-4 identity neural. Carrier ridge uses k trials from those 4 support
trials only. Encoder identity is M4 because later-day files have 10 trials.
Not EvalAI hidden held-out.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import carrier_k
from . import m4_query
from . import plan
from . import receipts


class CarrierKLaterDayError(RuntimeError):
    """Fail closed for the later-day carrier-k probe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierKLaterDayError(message)


SCHEMA = "m1_b3_allsource_carrier_k_later_day_v1"
SUPPORT = int(plan.CARRIER_K_LATER_DAY_SUPPORT)
QUERY_START = int(plan.CARRIER_K_LATER_DAY_QUERY_START)
EVAL_BATCH_SIZE = int(m4_query.EVAL_BATCH_SIZE)
BASELINE_NAME = "chronological_k4"


def score_arm(repo_root: Path, arm: str) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _require(arm in plan.CARRIER_K_LATER_DAY_ARMS, f"later-day carrier-k is not defined for {arm}")
    from . import package as package_module
    from . import rsyn3_bank as bank_module

    package_module._ensure_streaming_paths(repo_root)
    spec = plan.ARM_SPECS[arm]
    _require(spec["side_feature_group"] == "rsyn3", f"{arm} is not an rSyn3 student")
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
    sessions = m4_query.selected_sessions(mapping, include_source_in_train=False)

    import torch
    from falcon_challenge.config import FalconConfig, FalconTask
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from src.data.falcon_datamodule import FalconDataset
    from src.data.falcon_t4_features import calibration_target_angles

    config = OmegaConf.load(resolved)
    _require(str(config.data.task).lower() == "m1", "resolved config is not native M1")
    _require(int(config.data.calibration_n_trials) == plan.CALIBRATION_N_TRIALS, "train M10 drift")
    _require(str(config.model.variant) == spec["variant"], "variant drift")
    _require(str(config.data.side_feature_group).lower() == "rsyn3", "side group drift")
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
    rsyn3_fitted = package_module._load_rsyn3_bank(hydra_dir)
    _require(
        list(rsyn3_fitted.get("source_sessions", [])) == list(plan.SOURCE_SESSION_NAMES),
        "rSyn3 bank was not fit on the four source sessions",
    )
    _require(rsyn3_fitted.get("later_day_in_fit") is False, "later-day entered the NMF fit")
    basis = bank_module._basis_from_bank(rsyn3_fitted)
    grid = carrier_k.later_day_variant_grid()
    task = FalconConfig(task=FalconTask.m1).task

    per_variant: dict[str, dict[str, Any]] = {
        row["name"]: {**row, "sessions": [], "preds": [], "targets": []}
        for row in grid
    }

    with torch.inference_mode():
        covariates_mean = covariates_std = None
        for index, (name, path) in enumerate(sessions.items()):
            role = m4_query.session_role(name)
            _require(role == "later_day_public_calib", f"{name} is not later-day")
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
                calibration_n_trials=SUPPORT,
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
                side_feature_group="none",
                query_start_trial=QUERY_START,
            )
            audit = dataset.query_window_audit[name]
            n_trials = int(audit["total_trials"])
            _require(n_trials == 10, f"{name} later-day file is not 10 trials")
            _require(int(audit["query_start_trial"]) == QUERY_START, "query start drift")
            _require(int(dataset.calib_n_trials[name]) == SUPPORT, "M4 identity budget drift")
            _require(int(audit["query_trials"]) == plan.M4_QUERY_LATER_DAY_QUERY_TRIALS, "later-day query trial count")
            _require(len(dataset) == int(audit["eligible_windows"]) > 0, "window/audit mismatch")
            calibration_np = np.asarray(
                dataset.calib_trialized_neural_features[name][:SUPPORT],
                dtype=np.float32,
            )
            _require(
                calibration_np.shape == (SUPPORT, plan.TRIAL_BINS, plan.CHANNELS),
                f"unexpected M4 calibration shape for {name}: {calibration_np.shape}",
            )
            calibration = torch.from_numpy(calibration_np).unsqueeze(0)
            angles = calibration_target_angles(path, "m1")
            _require(int(angles.shape[0]) >= SUPPORT, f"{name} has fewer than {SUPPORT} tgt_loc rows")
            support = bank_module.load_public_calib_support(path, support_trials=SUPPORT)
            synergy = carrier_k.trial_mean_synergy(support, basis, pool_trials=SUPPORT)
            loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
            batches: list[tuple[Any, Any]] = []
            for batch in loader:
                neural = batch[0].float() if hasattr(batch[0], "float") else torch.from_numpy(
                    np.asarray(batch[0], dtype=np.float32)
                )
                target = batch[1].float() if hasattr(batch[1], "float") else torch.from_numpy(
                    np.asarray(batch[1], dtype=np.float32)
                )
                batches.append((neural, target))
            _require(batches, f"{name} has no remaining-query windows")

            for variant in grid:
                selected = carrier_k.select_indices(
                    str(variant["method"]),
                    int(variant["k"]),
                    angles=angles,
                    synergy_means=synergy,
                    pool_trials=SUPPORT,
                )
                _require(selected.shape == (int(variant["k"]),), f"{name} selection size")
                _require(int(selected.max()) < SUPPORT, f"{name} selector used a query trial")
                side_np = np.ascontiguousarray(
                    bank_module.encode_record_selected(
                        support, rsyn3_fitted, selected, pool_trials=SUPPORT,
                    ),
                    dtype=np.float32,
                )
                _require(side_np.shape == (plan.CHANNELS, 4), f"unexpected rSyn3 side shape for {name}")
                side = torch.from_numpy(side_np).unsqueeze(0)
                identity = student.compute_identity(calibration, side_features=side)
                _require(
                    tuple(identity.shape) == (1, plan.CHANNELS, plan.IDENTITY_DIM),
                    f"identity shape {tuple(identity.shape)}",
                )
                preds: list[np.ndarray] = []
                targets: list[np.ndarray] = []
                for neural, target in batches:
                    identity_batch = identity.expand(neural.shape[0], -1, -1)
                    pred, _ = student(neural, identity=identity_batch)
                    preds.append(pred[:, -1, :].cpu().numpy())
                    targets.append(target[:, -1, :].cpu().numpy())
                pred_all = np.concatenate(preds, axis=0)
                target_all = np.concatenate(targets, axis=0)
                _require(pred_all.shape[0] == len(dataset), f"{name} scored window count drift")
                _require(pred_all.shape[1] == target_all.shape[1] == 16, f"{name} EMG dim")
                score = m4_query.last_bin_r2(pred_all, target_all)
                row = {
                    "session": name,
                    "role": role,
                    "n_trials": n_trials,
                    "encoder_neural_trials": SUPPORT,
                    "carrier_k": int(variant["k"]),
                    "method": variant["method"],
                    "selected_trial_ids": [int(item) for item in selected.tolist()],
                    "query_start_trial": QUERY_START,
                    "query_trials": int(audit["query_trials"]),
                    "n_windows": int(len(dataset)),
                    "n_outputs": 16,
                    "r2_variance_weighted_last_bin": score,
                    "carrier_sha256": package_module.sha256_array(side_np),
                    "identity_sha256": package_module.sha256_array(identity.detach().cpu().numpy()),
                    "rsyn3_nmf_refit": False,
                    "in_train_windows": False,
                    "formal_benchmark_verdict": False,
                }
                per_variant[variant["name"]]["sessions"].append(row)
                per_variant[variant["name"]]["preds"].append(pred_all)
                per_variant[variant["name"]]["targets"].append(target_all)
                print(
                    f"{name} {variant['name']} r2={score:.6f} selected={row['selected_trial_ids']}",
                    flush=True,
                )

    summaries: list[dict[str, Any]] = []
    for variant in grid:
        payload = per_variant[variant["name"]]
        later_rows = payload["sessions"]
        _require(len(later_rows) == 3, "need three later-day sessions")
        _require(all(row["role"] == "later_day_public_calib" for row in later_rows), "source leaked")
        pred_all = np.concatenate(payload["preds"], axis=0)
        target_all = np.concatenate(payload["targets"], axis=0)
        pooled = m4_query.last_bin_r2(pred_all, target_all)
        mean_session = float(np.mean([row["r2_variance_weighted_last_bin"] for row in later_rows]))
        summaries.append(
            {
                "name": variant["name"],
                "method": variant["method"],
                "k": int(variant["k"]),
                "is_baseline": bool(variant["is_baseline"]),
                "later_day": {
                    "n_sessions": len(later_rows),
                    "query_trials_per_session": plan.M4_QUERY_LATER_DAY_QUERY_TRIALS,
                    "n_windows": int(sum(row["n_windows"] for row in later_rows)),
                    "mean_session_r2": mean_session,
                    "pooled_r2": pooled,
                },
                "sessions": later_rows,
            }
        )

    baseline = next(row for row in summaries if row["name"] == BASELINE_NAME)
    baseline_pooled = float(baseline["later_day"]["pooled_r2"])
    for row in summaries:
        row["later_day"]["delta_vs_chrono_k4_pooled"] = (
            float(row["later_day"]["pooled_r2"]) - baseline_pooled
        )
        row["formal_benchmark_verdict"] = False

    ranking = sorted(
        summaries,
        key=lambda row: float(row["later_day"]["pooled_r2"]),
        reverse=True,
    )
    return {
        "schema": SCHEMA,
        "arm": arm,
        "student_checkpoint": str(ckpt),
        "student_checkpoint_sha256": hydra_run["student_checkpoint_sha256"],
        "teacher_sha256": plan.TEACHER_SHA256,
        "support_trials": SUPPORT,
        "query_start_trial": QUERY_START,
        "encoder_neural_trials": SUPPORT,
        "selector_pool_is_support_only": True,
        "n_variants": len(summaries),
        "baseline": BASELINE_NAME,
        "ranking": [
            {
                "name": row["name"],
                "method": row["method"],
                "k": row["k"],
                "is_baseline": row["is_baseline"],
                "mean_session_r2": row["later_day"]["mean_session_r2"],
                "pooled_r2": row["later_day"]["pooled_r2"],
                "delta_vs_chrono_k4_pooled": row["later_day"]["delta_vs_chrono_k4_pooled"],
            }
            for row in ranking
        ],
        "variants": summaries,
        "note": plan.CARRIER_K_LATER_DAY_NOTE,
        "formal_benchmark_verdict": False,
        "official_original_heldout_r2": plan.OFFICIAL_ORIGINAL_HELDOUT_R2,
        "local_20120924_is_not_official_heldout": True,
        "local_source_post_m10_is_in_train": True,
    }


def execute(repo_root: Path, arm: str) -> tuple[dict[str, str], str | None, str | None]:
    repo_root = Path(repo_root).absolute()
    _require(os.environ.get("PYTHONNOUSERSITE") == "1", "execute requires PYTHONNOUSERSITE=1")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES", None) in {None, ""}, "later-day carrier-k is CPU-only")
    _require(arm in plan.CARRIER_K_LATER_DAY_ARMS, f"later-day carrier-k is not defined for {arm}")
    receipts.refuse_sealed_roots(repo_root / plan.RESULT_ROOT_RELATIVE)
    parent = repo_root / plan.CARRIER_K_LATER_DAY_ROOT_RELATIVE
    parent.mkdir(parents=True, exist_ok=True)
    progress_state: dict[str, object] = {
        "mode": "carrier_k_later_day",
        "arm": arm,
        "n_variants": plan.CARRIER_K_LATER_DAY_N_VARIANTS,
    }

    def launch_builder() -> dict[str, object]:
        return {
            "schema": "m1_b3_allsource_carrier_k_later_day_launch_v1",
            "arm": arm,
            "status": "LAUNCHED",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "support_trials": SUPPORT,
            "query_start_trial": QUERY_START,
            "encoder_neural_trials": SUPPORT,
            "selector_pool_is_support_only": True,
            "n_variants": plan.CARRIER_K_LATER_DAY_N_VARIANTS,
            "formal_benchmark_verdict": False,
        }

    def body_publisher(artifact: Any) -> dict[str, str]:
        payload = score_arm(repo_root, arm)
        progress_state["best_name"] = payload["ranking"][0]["name"]
        progress_state["best_pooled_r2"] = payload["ranking"][0]["pooled_r2"]
        progress_state["baseline_pooled_r2"] = next(
            row["pooled_r2"] for row in payload["ranking"] if row["name"] == BASELINE_NAME
        )
        return {"carrier_k_later_day.json": artifact.publish_json("carrier_k_later_day.json", payload)}

    def terminal_builder(shas: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema": "m1_b3_allsource_carrier_k_later_day_terminal_v1",
            "status": "COMPLETE",
            "mode": "carrier_k_later_day",
            "arm": arm,
            "support_trials": SUPPORT,
            "query_start_trial": QUERY_START,
            "formal_benchmark_verdict": False,
            "local_20120924_is_not_official_heldout": True,
            "official_original_heldout_r2": plan.OFFICIAL_ORIGINAL_HELDOUT_R2,
            "best_name": progress_state.get("best_name"),
            "best_pooled_r2": progress_state.get("best_pooled_r2"),
            "baseline_pooled_r2": progress_state.get("baseline_pooled_r2"),
            "bodies": {key: value for key, value in shas.items() if not str(key).startswith("_")},
        }

    stage = Path(plan.carrier_k_later_day_root_relative(arm)).relative_to(plan.RESULT_ROOT_RELATIVE)
    return receipts.run_stage0(
        repo_root / plan.RESULT_ROOT_RELATIVE,
        relative=str(stage),
        attempt_payload={
            "schema": "m1_b3_allsource_carrier_k_later_day_attempt_v1",
            "cell": plan.CELL,
            "phase": plan.PHASE,
            "mode": "carrier_k_later_day",
            "arm": arm,
            "status": "ATTEMPT_RESERVED",
            "data_or_model_accessed": False,
            "formal_benchmark_verdict": False,
        },
        launch_builder=launch_builder,
        body_publisher=body_publisher,
        terminal_builder=terminal_builder,
        progress=lambda: dict(progress_state),
    )
