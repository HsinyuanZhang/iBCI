"""Rebuild and ensemble independently source-selected calibration-aware AEs."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sua_exploration.h1_m1_priority_v1.core import DirectRidgeSession, array_sha256, regression_metrics

from .calibration_aware import CalibrationAwareSpec, fit_calibration_aware_manifold
from .core import need
from .m1_screen import _stable_seed, _x_normalizer, score_one_session


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def ensemble_calibration_aware(
    sessions: Mapping[str, DirectRidgeSession],
    result_paths: Sequence[Path],
    *,
    device: str,
) -> dict[str, Any]:
    need(len(result_paths) >= 2 and len(set(result_paths)) == len(result_paths), "ensemble needs distinct seed results")
    results = []
    for path in result_paths:
        need(path.is_file() and not path.is_symlink(), f"missing seed result: {path}")
        body = json.loads(path.read_text(encoding="utf-8"))
        need(body.get("schema") == "m1_m10_source_frozen_calibration_aware_autoencoder_nested_loso_v1", "seed schema drift")
        need(body.get("target_used_for_selection") is False, "seed used target selection")
        results.append((path, body))
    names = tuple(sorted(sessions)); reference = results[0][1]
    for _, body in results:
        need(tuple(body["sessions"]) == names and body["output_names"] == reference["output_names"], "seed surface drift")
        need(body["directridge_equal_session"] == reference["directridge_equal_session"], "seed baseline drift")
    target_device = torch.device(device); folds = {}; truth_all = []; prediction_all = []
    for target_name in names:
        source_names = tuple(name for name in names if name != target_name); sources = {name: sessions[name] for name in source_names}
        x_mean, x_scale = _x_normalizer(sources); seed_predictions = []; seed_rows = []; truth_reference = None
        for path, body in results:
            fold = body["folds"][target_name]; selected = fold["selected"]; spec = CalibrationAwareSpec(**selected["spec"])
            seed_offset = body.get("seed_offset")
            seed = (
                _stable_seed("ca-final", target_name, asdict(spec))
                if seed_offset is None
                else _stable_seed("ca-final", int(seed_offset), target_name, asdict(spec))
            )
            manifold = fit_calibration_aware_manifold(sources, spec, device=target_device, seed=seed)
            need(manifold.fit_evidence["state_sha256"] == fold["final_fit"]["state_sha256"], f"{target_name}: rebuilt state SHA drift")
            metrics, truth, prediction = score_one_session(
                sessions[target_name], manifold, x_mean=x_mean, x_scale=x_scale,
                ridge_lambda=float(selected["ridge_lambda_per_sample"]),
            )
            need(metrics["prediction_sha256"] == fold["target_metrics"]["prediction_sha256"], f"{target_name}: rebuilt prediction drift")
            if truth_reference is None: truth_reference = truth
            else: need(np.array_equal(truth_reference, truth), f"{target_name}: seed target drift")
            seed_predictions.append(prediction)
            seed_rows.append({
                "result_path": str(path.resolve()), "result_sha256": _sha_file(path), "seed_offset": seed_offset,
                "selected": selected, "state_sha256": manifold.fit_evidence["state_sha256"],
                "prediction_sha256": array_sha256(prediction), "single_seed_r2": float(metrics["pooled_variance_weighted_r2"]),
            })
        need(truth_reference is not None, "missing ensemble truth")
        ensemble_prediction = np.mean(np.stack(seed_predictions, axis=0), axis=0, dtype=np.float64)
        metrics = regression_metrics(truth_reference, ensemble_prediction)
        baseline = results[0][1]["folds"][target_name]["baseline_target_metrics"]
        folds[target_name] = {
            "source_sessions": list(source_names), "seed_models": seed_rows,
            "ensemble_rule": "unweighted arithmetic mean of three source-selected predictions; no target selection",
            "target_metrics": metrics, "baseline_target_metrics": baseline,
            "delta_vs_directridge": float(metrics["pooled_variance_weighted_r2"] - baseline["pooled_variance_weighted_r2"]),
            "truth_sha256": array_sha256(truth_reference), "ensemble_prediction_sha256": array_sha256(ensemble_prediction),
        }
        truth_all.append(truth_reference); prediction_all.append(ensemble_prediction)
    scores = np.asarray([folds[name]["target_metrics"]["pooled_variance_weighted_r2"] for name in names]); baseline_scores = np.asarray([folds[name]["baseline_target_metrics"]["pooled_variance_weighted_r2"] for name in names]); delta = scores - baseline_scores
    return {
        "schema": "m1_m10_calibration_aware_autoencoder_three_seed_ensemble_v1", "status": "COMPLETE_THREE_SEED_SOURCE_SELECTED_ENSEMBLE",
        "protocol": {"target_used_for_seed_or_model_selection": False, "target_backward_steps": 0, "target_optimizer_steps": 0, "state_and_prediction_rebuild_exact": True},
        "seed_result_paths": [{"path": str(path.resolve()), "sha256": _sha_file(path)} for path, _ in results], "sessions": list(names), "output_names": reference["output_names"], "folds": folds,
        "equal_session": {"mean_r2": float(scores.mean()), "median_r2": float(np.median(scores)), "per_session_r2": dict(zip(names, scores.tolist()))}, "directridge_equal_session": reference["directridge_equal_session"],
        "paired_delta_vs_directridge": {"mean": float(delta.mean()), "median": float(np.median(delta)), "positive_sessions": int((delta > 0).sum()), "total_sessions": len(names), "per_session": dict(zip(names, delta.tolist()))},
        "pooled": regression_metrics(np.concatenate(truth_all), np.concatenate(prediction_all)), "target_used_for_selection": False,
    }

