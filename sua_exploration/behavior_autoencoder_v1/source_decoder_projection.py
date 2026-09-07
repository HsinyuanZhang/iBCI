"""Three-fold projection of sealed source-only full M1 SPINT decoders."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sua_exploration.h1_m1_priority_v1.core import DirectRidgeSession, array_sha256, regression_metrics

from .calibration_aware import CalibrationAwareSpec, fit_calibration_aware_manifold
from .core import need
from .full_projection import ROOT, STREAM_ROOT, _sha_file
from .m1_screen import _stable_seed


MODEL_CONFIG = ROOT / "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold1/runs/2026-08-06-16-20-17-515482_rid-m1_afc4_source_decoder_fold1_dev20_f1_s42/.hydra/config.yaml"
DATA_CONFIG = ROOT / "sua_exploration/m1_compact_replication/results/m1_e23_remote_pull_20260810_035000/configs/b3s_zero4_resolved_config.yaml"
CHECKPOINTS = {
    0: ROOT / "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold0/runs/2026-08-06-16-15-55-070150_rid-m1_afc4_source_decoder_fold0_dev20_resume_e1r1_fNone_s42/checkpoints/best_ckpt/epoch_018.ckpt",
    1: ROOT / "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold1/runs/2026-08-06-16-20-17-515482_rid-m1_afc4_source_decoder_fold1_dev20_f1_s42/checkpoints/best_ckpt/epoch_019.ckpt",
    2: ROOT / "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold2_remote/runs/remote_fold2_source_epoch019/checkpoints/best_ckpt/epoch_019.ckpt",
}
TARGETS = {0: "ses-20120924", 1: "ses-20120926", 2: "ses-20120927"}


def _prioritize_streaming_src() -> None:
    for value in (STREAM_ROOT, ROOT):
        while str(value) in sys.path: sys.path.remove(str(value))
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(STREAM_ROOT))
    loaded = sys.modules.get("src")
    need(loaded is None or str(getattr(loaded, "__file__", "")).startswith(str(STREAM_ROOT)), "wrong src package loaded")


def _forward_fold(fold: int, sessions: Mapping[str, DirectRidgeSession], device: torch.device) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _prioritize_streaming_src(); import hydra
    from omegaconf import OmegaConf
    target_name = TARGETS[fold]; sources = tuple(name for name in sorted(sessions) if name != target_name)
    model_config = OmegaConf.load(MODEL_CONFIG).model
    module = hydra.utils.instantiate(model_config); payload = torch.load(CHECKPOINTS[fold], map_location="cpu", weights_only=False); module.load_state_dict(payload["state_dict"], strict=True); module.to(device).eval()
    for parameter in module.parameters(): parameter.requires_grad_(False)
    full_config = OmegaConf.load(DATA_CONFIG); data_config = full_config.data
    data_config.loso_fold = fold; data_config.source_session_names = list(sources); data_config.heldin_session_names = list(sources); data_config.afc4_arm = "none"; data_config.data_dir = str((ROOT / "SPINT-main/data/000941").resolve())
    datamodule = hydra.utils.instantiate(data_config); datamodule.setup("test"); need(datamodule.outer_left_out == target_name, "source decoder target drift")
    predictions = []; targets = []
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            need(len(batch) == 4, "source decoder query arity drift")
            neural, target, calibration, names = batch; need(set(str(value) for value in names) == {target_name}, "source decoder session drift")
            prediction = module(neural.to(device), calib_trialized_neural_features=calibration.to(device))
            if bool(module.hparams.decode_last_timestep_only): prediction = prediction[:, -1:, :]; target = target[:, -1:, :]
            if bool(module.hparams.predict_scaled_behavior): prediction = prediction / module.hparams.behavior_scaling_factor
            predictions.append(prediction.flatten(0, 1).cpu().numpy()); targets.append(target.flatten(0, 1).numpy())
    prediction = np.concatenate(predictions).astype(np.float64); target = np.concatenate(targets).astype(np.float64)
    return target, prediction, {"fold": fold, "target_session": target_name, "source_sessions": list(sources), "checkpoint_path": str(CHECKPOINTS[fold].resolve()), "checkpoint_sha256": _sha_file(CHECKPOINTS[fold]), "checkpoint_epoch": int(payload["epoch"]), "checkpoint_global_step": int(payload["global_step"]), "model_config_path": str(MODEL_CONFIG.resolve()), "model_config_sha256": _sha_file(MODEL_CONFIG), "query_samples": int(target.shape[0]), "target_sha256": array_sha256(target), "prediction_sha256": array_sha256(prediction)}


def evaluate_source_decoder_projection(
    sessions: Mapping[str, DirectRidgeSession], seed_results: Sequence[Path], *, device: str,
) -> dict[str, Any]:
    target_device = torch.device(device); results = []
    for path in seed_results:
        body = json.loads(path.read_text(encoding="utf-8")); need(body.get("schema") == "m1_m10_source_frozen_calibration_aware_autoencoder_nested_loso_v1", "seed result drift"); results.append((path, body))
    folds = {}; baseline_scores = []; projected_scores = []
    for fold in (0, 1, 2):
        target_name = TARGETS[fold]; target, prediction, authority = _forward_fold(fold, sessions, target_device); sources = {name: sessions[name] for name in authority["source_sessions"]}
        baseline = regression_metrics(target, prediction); projections = []; models = []
        for path, body in results:
            selected = body["folds"][target_name]["selected"]; spec = CalibrationAwareSpec(**selected["spec"]); seed_offset = body.get("seed_offset")
            seed = _stable_seed("ca-final", target_name, asdict(spec)) if seed_offset is None else _stable_seed("ca-final", int(seed_offset), target_name, asdict(spec))
            manifold = fit_calibration_aware_manifold(sources, spec, device=target_device, seed=seed); need(manifold.fit_evidence["state_sha256"] == body["folds"][target_name]["final_fit"]["state_sha256"], "source projector state drift")
            value = manifold.decode(manifold.encode(prediction)); projections.append(value); models.append({"seed_result_path": str(path.resolve()), "seed_result_sha256": _sha_file(path), "state_sha256": manifold.fit_evidence["state_sha256"], "projected_prediction_sha256": array_sha256(value)})
        projected_prediction = np.mean(np.stack(projections), axis=0, dtype=np.float64); projected = regression_metrics(target, projected_prediction); baseline_value = float(baseline["pooled_variance_weighted_r2"]); projected_value = float(projected["pooled_variance_weighted_r2"])
        baseline_scores.append(baseline_value); projected_scores.append(projected_value)
        per_output_delta = np.asarray(projected["r2_per_output"]) - np.asarray(baseline["r2_per_output"])
        folds[str(fold)] = {"authority": authority, "baseline_metrics": baseline, "projected_metrics": projected, "projection_models": models, "delta": projected_value - baseline_value, "per_output_delta": per_output_delta.tolist(), "positive_outputs": int((per_output_delta > 0.0).sum()), "projected_prediction_sha256": array_sha256(projected_prediction)}
    baseline_array = np.asarray(baseline_scores); projected_array = np.asarray(projected_scores); delta = projected_array - baseline_array
    return {"schema": "m1_source_only_full_spint_three_fold_calibration_aware_projection_v1", "status": "COMPLETE_THREE_FOLD_FULL_DECODER_PROJECTION", "scope": {"folds": [0, 1, 2], "development_only": True, "formal_opened": False, "target_backward_steps": 0, "target_optimizer_steps": 0, "target_used_for_projection_selection": False}, "folds": folds, "baseline_equal_fold_mean_r2": float(baseline_array.mean()), "projected_equal_fold_mean_r2": float(projected_array.mean()), "paired_delta": {"mean": float(delta.mean()), "median": float(np.median(delta)), "positive_folds": int((delta > 0.0).sum()), "total_folds": 3, "per_fold": delta.tolist()}}

