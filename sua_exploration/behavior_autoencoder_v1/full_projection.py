"""Fold-0 full-neural-decoder post-hoc manifold projection diagnostic."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sua_exploration.h1_m1_priority_v1.core import DirectRidgeSession, array_sha256, regression_metrics

from .calibration_aware import CalibrationAwareSpec, fit_calibration_aware_manifold
from .core import need
from .m1_screen import _stable_seed


ROOT = Path(__file__).resolve().parents[2]
STREAM_ROOT = ROOT / "streaming_calibration_exp"
PULLED_ROOT = ROOT / "sua_exploration/m1_compact_replication/results/m1_e23_remote_pull_20260810_035000"
CONFIG = PULLED_ROOT / "configs/b3s_zero4_resolved_config.yaml"
CHECKPOINT = PULLED_ROOT / "checkpoints/b3s_zero4_epoch023.ckpt"
EXPECTED_BASELINE_R2 = 0.6275088717756858


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def _full_fold0_predictions(device: torch.device) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # This workspace contains several unrelated top-level ``src`` packages.
    # The full M1 checkpoint is owned by streaming_calibration_exp.
    for value in (STREAM_ROOT, ROOT):
        while str(value) in sys.path: sys.path.remove(str(value))
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(STREAM_ROOT))
    loaded_src = sys.modules.get("src")
    need(loaded_src is None or str(getattr(loaded_src, "__file__", "")).startswith(str(STREAM_ROOT)), "wrong top-level src package already imported")
    import hydra
    from omegaconf import OmegaConf

    need(CONFIG.is_file() and CHECKPOINT.is_file(), "fold-0 full decoder authority missing")
    config = OmegaConf.load(CONFIG); config.paths.output_dir = str(CONFIG.parent.resolve()); config.paths.work_dir = str(ROOT.resolve()); OmegaConf.resolve(config)
    need(str(config.data._target_).endswith("M1VersionBSourceLOSODataModule") and int(config.data.loso_fold) == 0, "fold-0 data config drift")
    need(str(config.model.variant) == "B3S" and config.model.freeze_decoder is False, "fold-0 model config drift")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    need(payload.get("epoch") == 23 and payload.get("global_step") == 118824, "fold-0 checkpoint topology drift")
    module = hydra.utils.instantiate(config.model); module.setup("test"); module.load_state_dict(payload["state_dict"], strict=True); module.to(device).eval()
    for parameter in module.parameters(): parameter.requires_grad_(False)
    datamodule = hydra.utils.instantiate(config.data); datamodule.setup("test")
    need(datamodule.outer_left_out == "ses-20120924", "fold-0 target drift")
    predictions = []; targets = []
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            neural, target, calibration, names, side = batch
            need(set(str(value) for value in names) == {"ses-20120924"}, "fold-0 query session drift")
            neural = neural.to(device); target = target.to(device); calibration = calibration.to(device); side = side.to(device)
            identity = module.student.compute_identity(calibration, side_features=side)
            prediction = module.student.decode_with_identity(neural, identity)
            prediction, sliced_target = module._slice_last_timestep(prediction, target)
            predictions.append(prediction.flatten(0, 1).cpu().numpy()); targets.append(sliced_target.flatten(0, 1).cpu().numpy())
    prediction = np.concatenate(predictions).astype(np.float64); target = np.concatenate(targets).astype(np.float64)
    baseline = regression_metrics(target, prediction)
    baseline_value = float(baseline["pooled_variance_weighted_r2"]); baseline_error = abs(baseline_value - EXPECTED_BASELINE_R2)
    need(baseline_error <= 1.0e-6, f"fold-0 baseline parity drift: observed={baseline_value}, expected={EXPECTED_BASELINE_R2}, error={baseline_error}")
    evidence = {"config_path": str(CONFIG.resolve()), "config_sha256": _sha_file(CONFIG), "checkpoint_path": str(CHECKPOINT.resolve()), "checkpoint_sha256": _sha_file(CHECKPOINT), "query_samples": int(target.shape[0]), "target_sha256": array_sha256(target), "prediction_sha256": array_sha256(prediction), "baseline_metrics": baseline, "historical_cpu_r2": EXPECTED_BASELINE_R2, "gpu_vs_cpu_r2_abs_error": baseline_error, "parity_tolerance": 1.0e-6}
    return target, prediction, evidence


def evaluate_fold0_projection(
    sessions: Mapping[str, DirectRidgeSession],
    seed_results: Sequence[Path],
    *,
    device: str,
) -> dict[str, Any]:
    target_device = torch.device(device); target, prediction, authority = _full_fold0_predictions(target_device)
    target_name = "ses-20120924"; source_names = tuple(name for name in sorted(sessions) if name != target_name); sources = {name: sessions[name] for name in source_names}
    projected = []; models = []
    for path in seed_results:
        body = json.loads(path.read_text(encoding="utf-8")); need(body.get("schema") == "m1_m10_source_frozen_calibration_aware_autoencoder_nested_loso_v1", "seed result drift")
        fold = body["folds"][target_name]; selected = fold["selected"]; spec = CalibrationAwareSpec(**selected["spec"]); seed_offset = body.get("seed_offset")
        seed = _stable_seed("ca-final", target_name, asdict(spec)) if seed_offset is None else _stable_seed("ca-final", int(seed_offset), target_name, asdict(spec))
        manifold = fit_calibration_aware_manifold(sources, spec, device=target_device, seed=seed)
        need(manifold.fit_evidence["state_sha256"] == fold["final_fit"]["state_sha256"], "projection manifold state drift")
        value = manifold.decode(manifold.encode(prediction)); projected.append(value)
        models.append({"result_path": str(path.resolve()), "result_sha256": _sha_file(path), "seed_offset": seed_offset, "state_sha256": manifold.fit_evidence["state_sha256"], "projected_prediction_sha256": array_sha256(value)})
    ensemble_projection = np.mean(np.stack(projected), axis=0, dtype=np.float64); metrics = regression_metrics(target, ensemble_projection); baseline = authority["baseline_metrics"]
    delta_per_output = np.asarray(metrics["r2_per_output"]) - np.asarray(baseline["r2_per_output"])
    return {
        "schema": "m1_full_neural_decoder_fold0_calibration_aware_projection_v1", "status": "COMPLETE_NON_GOVERNING_FULL_DECODER_PROJECTION",
        "scope": {"fold": 0, "target_session": target_name, "source_sessions": list(source_names), "development_only": True, "formal_opened": False, "target_backward_steps": 0, "target_optimizer_steps": 0, "target_used_for_projection_selection": False},
        "full_decoder_authority": authority, "projection_models": models, "projected_metrics": metrics,
        "delta_vs_unprojected_full_decoder": float(metrics["pooled_variance_weighted_r2"] - baseline["pooled_variance_weighted_r2"]),
        "per_output_delta": delta_per_output.tolist(), "positive_outputs": int((delta_per_output > 0.0).sum()), "output_count": int(delta_per_output.size),
        "ensemble_projection_sha256": array_sha256(ensemble_projection),
    }
