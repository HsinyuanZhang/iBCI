"""E6: PCA-8 projection control on the three source-only full SPINT decoder folds.

The published three-fold test projected each fold's raw full-decoder query
predictions through the frozen calibration-aware manifold and lost -0.021957
equal-fold R2.  E6 repeats that exact machinery with a SOURCE-FROZEN PCA-8
basis instead: for each fold, PCA-8 is fitted on that fold's three source
sessions' behaviour with the very same fit_source_manifold path used by the
"Source-frozen PCA" control in the comparison matrix (deterministic, no seed,
no target data), the raw fold predictions are pushed through
decode(encode(.)), and the fold is rescored with the same equal-fold governing
convention.

Interpretation fork (pre-registered in the receipt): if PCA-8 loses MUCH MORE
than 0.022, the manifold's geometry -- not generic 8-dimensional variance
capture -- is where the decoder's signal lives (a specificity control for the
on-manifold claim); if PCA-8 loses about the same, the effect is a generic
dimensionality effect and must be reported as such.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import torch

from sua_exploration.behavior_autoencoder_v1.core import AutoencoderSpec
from sua_exploration.behavior_autoencoder_v1.m1_screen import fit_source_manifold
from sua_exploration.behavior_autoencoder_v1.source_decoder_projection import _forward_fold
from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    array_sha256,
    regression_metrics,
)

from .frozen import THREEFOLD_RECEIPT_NAME, THREEFOLD_RECEIPT_SHA256, VA1_ROOT, _sha_file

PCA_LATENT_DIM = 8
DECISION_MARGIN = 0.022  # the published calibration-aware three-fold loss magnitude


def experiment_e6(
    sessions: Mapping[str, DirectRidgeSession], *, device: str
) -> dict[str, Any]:
    receipt_path = VA1_ROOT / THREEFOLD_RECEIPT_NAME
    assert _sha_file(receipt_path) == THREEFOLD_RECEIPT_SHA256, "threefold receipt SHA drift"
    published = json.loads(receipt_path.read_text(encoding="utf-8"))
    device_object = torch.device(device)
    folds: dict[str, Any] = {}
    baseline_scores = []
    pca_scores = []
    for fold in (0, 1, 2):
        target, prediction, authority = _forward_fold(fold, sessions, device_object)
        baseline = regression_metrics(target, prediction)
        published_baseline = float(published["folds"][str(fold)]["baseline_metrics"]["pooled_variance_weighted_r2"])
        assert abs(float(baseline["pooled_variance_weighted_r2"]) - published_baseline) < 1.0e-12, (
            f"fold {fold}: raw full-decoder baseline drift vs published threefold receipt"
        )
        sources = {name: sessions[name] for name in authority["source_sessions"]}
        manifold = fit_source_manifold(
            sources, AutoencoderSpec("pca", PCA_LATENT_DIM), device=device_object, seed=0
        )
        projected_prediction = manifold.decode(manifold.encode(prediction))
        projected = regression_metrics(target, projected_prediction)
        baseline_value = float(baseline["pooled_variance_weighted_r2"])
        projected_value = float(projected["pooled_variance_weighted_r2"])
        per_output_delta = np.asarray(projected["r2_per_output"]) - np.asarray(baseline["r2_per_output"])
        baseline_scores.append(baseline_value)
        pca_scores.append(projected_value)
        folds[str(fold)] = {
            "authority": authority,
            "baseline_metrics": baseline,
            "pca8_projected_metrics": projected,
            "pca8_fit": manifold.fit_evidence,
            "delta_pca8_vs_raw": projected_value - baseline_value,
            "published_calibration_aware_delta": float(published["folds"][str(fold)]["delta"]),
            "per_output_delta": per_output_delta.tolist(),
            "positive_outputs": int((per_output_delta > 0.0).sum()),
            "projected_prediction_sha256": array_sha256(projected_prediction),
        }
        if device_object.type == "cuda":
            torch.cuda.empty_cache()
    baseline_array = np.asarray(baseline_scores)
    pca_array = np.asarray(pca_scores)
    delta = pca_array - baseline_array
    published_delta = float(published["paired_delta"]["mean"])
    pca_loss_magnitude = float(-delta.mean())
    published_loss_magnitude = -published_delta
    if pca_loss_magnitude > 2.0 * DECISION_MARGIN:
        reading = (
            "PCA-8 loses much more than the calibration-aware manifold: the decoder's on-manifold signal "
            "is specific to the manifold's geometry, not generic 8-dimensional variance capture"
        )
    elif pca_loss_magnitude < 0.5 * DECISION_MARGIN:
        reading = (
            "PCA-8 loses far less than the calibration-aware manifold: the projection loss is largely a "
            "generic dimensionality effect and must be reported as such"
        )
    else:
        reading = (
            "PCA-8 loses about the same as the calibration-aware manifold: the projection loss is "
            "consistent with a generic 8-dimensional bottleneck effect"
        )
    return {
        "schema": "m1_behavior_manifold_v2_e6_pca8_projection_control_v1",
        "status": "COMPLETE_E6_PCA8_PROJECTION_CONTROL",
        "basis": {
            "kind": "source-frozen PCA",
            "latent_dim": PCA_LATENT_DIM,
            "fit_path": "behavior_autoencoder_v1.m1_screen.fit_source_manifold (the Source-frozen PCA control path)",
            "fit_data": "the fold's three source sessions' full eval-valid behaviour; deterministic; no seed effect; no target data",
        },
        "folds": folds,
        "equal_fold": {
            "raw_full_decoder_mean_r2": float(baseline_array.mean()),
            "pca8_projected_mean_r2": float(pca_array.mean()),
            "paired_delta_mean": float(delta.mean()),
            "paired_delta_per_fold": delta.tolist(),
            "positive_folds": int((delta > 0.0).sum()),
            "total_folds": 3,
        },
        "side_by_side_with_calibration_aware": {
            "calibration_aware_paired_delta_mean": published_delta,
            "pca8_paired_delta_mean": float(delta.mean()),
            "difference_pca8_minus_calibration_aware": float(delta.mean() - published_delta),
            "pre_registered_fork": (
                "PCA-8 loss >> 0.022 -> manifold geometry is the specific ingredient; ~ equal -> generic "
                "dimensionality effect"
            ),
            "reading": reading,
        },
        "governing_convention": "per-fold variance-weighted 16-output R2 on the fold's strict query; equal-fold mean",
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
            "decoder_checkpoints_loaded_readonly": True,
            "projection_selected_on_target": False,
        },
    }
