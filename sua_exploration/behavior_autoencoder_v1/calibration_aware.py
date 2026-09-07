"""Calibration-aware source-frozen behavior autoencoder.

The differentiable training objective mirrors deployment.  For every source
session it encodes the M10 labels, applies a fixed closed-form ridge operator
from M10 neural activity to latent behavior, predicts source query latents,
and decodes them to behavior.  At target deployment the trained manifold is
frozen and exactly the same closed-form M10 ridge is used with no target BP.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sua_exploration.h1_m1_priority_v1.core import DirectRidgeSession, array_sha256, canonical_sha256, direct_ridge_loso, regression_metrics

from .core import AutoencoderSpec, BehaviorAutoencoder, need, state_sha256, weighted_reconstruction_loss
from .m1_screen import FittedManifold, _moments, _session_behavior, _stable_seed, _x_normalizer, score_one_session


@dataclass(frozen=True, order=True)
class CalibrationAwareSpec:
    latent_dim: int
    hidden_dim: int
    reconstruction_weight: float
    raw_loss_fraction: float = 1.0
    activation: str = "gelu"
    pretrain_epochs: int = 20
    fine_steps: int = 300

    def validate(self, output_dim: int) -> None:
        AutoencoderSpec("mlp", self.latent_dim, self.hidden_dim, self.raw_loss_fraction, self.activation).validate(output_dim)
        need(self.reconstruction_weight >= 0.0 and np.isfinite(self.reconstruction_weight), "invalid reconstruction weight")
        need(self.pretrain_epochs >= 0 and self.fine_steps >= 1, "invalid calibration-aware schedule")

    @property
    def name(self) -> str:
        weight = str(self.reconstruction_weight).replace(".", "p")
        return f"cae_q{self.latent_dim}_h{self.hidden_dim}_rw{weight}"


@dataclass(frozen=True)
class _SessionTensor:
    name: str
    support_x_augmented: torch.Tensor
    ridge_projection: torch.Tensor
    support_y: torch.Tensor
    query_x_augmented: torch.Tensor
    query_y: torch.Tensor


def _ridge_projection(x: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    need(x.ndim == 2 and x.shape[0] > x.shape[1] and ridge_lambda >= 0.0, "invalid ridge projection input")
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    augmented = torch.cat((x, ones), dim=1)
    penalty = torch.eye(augmented.shape[1], dtype=x.dtype, device=x.device) * (float(ridge_lambda) * float(x.shape[0]))
    penalty[-1, -1] = 0.0
    projection = torch.linalg.solve(augmented.T @ augmented + penalty, augmented.T)
    need(bool(torch.isfinite(projection).all()), "ridge projection became nonfinite")
    return projection


def fit_calibration_aware_manifold(
    sessions: Mapping[str, DirectRidgeSession],
    spec: CalibrationAwareSpec,
    *,
    device: torch.device,
    seed: int,
    meta_ridge_lambda: float = 0.3,
    reconstruction_batch_size: int = 4096,
    query_batch_size: int = 4096,
) -> FittedManifold:
    names = tuple(sorted(sessions)); need(len(names) >= 2, "calibration-aware fit needs at least two source sessions")
    output_dim = sessions[names[0]].target.shape[1]; spec.validate(output_dim)
    raw_by_session = {name: _session_behavior(sessions[name]) for name in names}
    raw_all = np.concatenate([raw_by_session[name] for name in names])
    behavior_mean, behavior_scale = _moments(raw_all)
    standardized_all = (raw_all - behavior_mean) / behavior_scale
    x_mean, x_scale = _x_normalizer(sessions)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = BehaviorAutoencoder(output_dim, spec.latent_dim, spec.hidden_dim, spec.activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    scale_tensor = torch.as_tensor(behavior_scale, dtype=torch.float32, device=device)
    reconstruction_tensor = torch.as_tensor(standardized_all, dtype=torch.float32, device=device)
    session_tensors: list[_SessionTensor] = []
    for name in names:
        support_x, support_y, _ = sessions[name].aligned(0, split="support")
        query_x, query_y, _ = sessions[name].aligned(0, split="query")
        support_x_tensor = torch.as_tensor((support_x - x_mean) / x_scale, dtype=torch.float32, device=device)
        query_x_tensor = torch.as_tensor((query_x - x_mean) / x_scale, dtype=torch.float32, device=device)
        support_augmented = torch.cat((support_x_tensor, torch.ones((support_x_tensor.shape[0], 1), device=device)), dim=1)
        query_augmented = torch.cat((query_x_tensor, torch.ones((query_x_tensor.shape[0], 1), device=device)), dim=1)
        session_tensors.append(_SessionTensor(
            name, support_augmented, _ridge_projection(support_x_tensor, meta_ridge_lambda),
            torch.as_tensor((support_y - behavior_mean) / behavior_scale, dtype=torch.float32, device=device),
            query_augmented, torch.as_tensor((query_y - behavior_mean) / behavior_scale, dtype=torch.float32, device=device),
        ))
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed + 1)
    started = time.perf_counter(); pretrain_history: list[float] = []
    # Short reconstruction pretraining makes the decoder nondegenerate before
    # the deployment-matched objective is switched on.
    for _ in range(spec.pretrain_epochs):
        model.train(); order = torch.randperm(reconstruction_tensor.shape[0], generator=generator)
        total = 0.0; seen = 0
        for start in range(0, order.numel(), reconstruction_batch_size):
            index = order[start : start + reconstruction_batch_size].to(device)
            target = reconstruction_tensor[index]
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_reconstruction_loss(model(target), target, source_scale=scale_tensor, raw_loss_fraction=spec.raw_loss_fraction)
            need(bool(torch.isfinite(loss)), "pretraining loss became nonfinite")
            loss.backward(); optimizer.step(); total += float(loss.detach()) * int(target.shape[0]); seen += int(target.shape[0])
        pretrain_history.append(total / float(seen))
    fine_history: list[dict[str, float | int]] = []
    best_loss = math.inf; best_step = -1; best_state: dict[str, torch.Tensor] | None = None
    for step in range(spec.fine_steps):
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction_losses: list[torch.Tensor] = []
        for row in session_tensors:
            support_latent = model.encode(row.support_y)
            beta = row.ridge_projection @ support_latent
            query_count = min(query_batch_size, row.query_y.shape[0])
            query_index = torch.randint(row.query_y.shape[0], (query_count,), generator=generator).to(device)
            predicted_latent = row.query_x_augmented[query_index] @ beta
            predicted_behavior = model.decode(predicted_latent)
            prediction_losses.append(weighted_reconstruction_loss(
                predicted_behavior, row.query_y[query_index], source_scale=scale_tensor,
                raw_loss_fraction=spec.raw_loss_fraction,
            ))
        reconstruction_index = torch.randint(reconstruction_tensor.shape[0], (min(reconstruction_batch_size, reconstruction_tensor.shape[0]),), generator=generator).to(device)
        reconstruction_target = reconstruction_tensor[reconstruction_index]
        reconstruction_loss = weighted_reconstruction_loss(
            model(reconstruction_target), reconstruction_target, source_scale=scale_tensor,
            raw_loss_fraction=spec.raw_loss_fraction,
        )
        prediction_loss = torch.stack(prediction_losses).mean()
        loss = prediction_loss + float(spec.reconstruction_weight) * reconstruction_loss
        need(bool(torch.isfinite(loss)), "calibration-aware loss became nonfinite")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value; best_step = step
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if step % 10 == 0 or step == spec.fine_steps - 1:
            fine_history.append({"step": step, "loss": value, "prediction_loss": float(prediction_loss.detach()), "reconstruction_loss": float(reconstruction_loss.detach())})
    need(best_state is not None and best_step >= 0, "calibration-aware fit produced no state")
    model.load_state_dict(best_state, strict=True); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    evidence = {
        "kind": "calibration_aware_mlp", "spec": asdict(spec), "source_sessions": list(names), "source_rows": int(raw_all.shape[0]),
        "source_raw_sha256": array_sha256(raw_all), "behavior_mean": behavior_mean.tolist(), "behavior_scale": behavior_scale.tolist(),
        "behavior_mean_sha256": array_sha256(behavior_mean), "behavior_scale_sha256": array_sha256(behavior_scale),
        "x_mean_sha256": array_sha256(x_mean), "x_scale_sha256": array_sha256(x_scale), "meta_ridge_lambda_per_sample": float(meta_ridge_lambda),
        "seed": int(seed), "parameter_count": int(sum(value.numel() for value in model.parameters())), "state_sha256": state_sha256(model),
        "best_step": int(best_step), "best_loss": float(best_loss), "pretrain_history_sha256": canonical_sha256(pretrain_history),
        "fine_history": fine_history, "fine_history_sha256": canonical_sha256(fine_history), "elapsed_seconds": float(time.perf_counter() - started),
        "closed_form_ridge_recomputed_differentiably": True, "target_session_in_fit": False,
    }
    ae_spec = AutoencoderSpec("mlp", spec.latent_dim, spec.hidden_dim, spec.raw_loss_fraction, spec.activation)
    return FittedManifold(ae_spec, behavior_mean, behavior_scale, None, model, device, evidence)


def nested_calibration_aware_loso(
    sessions: Mapping[str, DirectRidgeSession],
    specs: Sequence[CalibrationAwareSpec],
    *,
    device: str,
    deployment_lambda_grid: Sequence[float] = (0.1, 0.3),
    seed_offset: int = 0,
) -> dict[str, Any]:
    names = tuple(sorted(sessions)); need(len(names) == 4, "calibration-aware M1 requires four sessions")
    specs = tuple(specs); need(specs and len({spec.name for spec in specs}) == len(specs), "calibration-aware candidate drift")
    target_device = torch.device(device); need(target_device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    baseline = direct_ridge_loso(sessions, lag_grid=(0,), lambda_grid=deployment_lambda_grid)
    folds: dict[str, Any] = {}; truth_rows: list[np.ndarray] = []; prediction_rows: list[np.ndarray] = []
    started = time.perf_counter()
    for target_name in names:
        source_names = tuple(name for name in names if name != target_name); candidates: list[dict[str, Any]] = []
        for spec in specs:
            scores_by_lambda = {float(value): [] for value in deployment_lambda_grid}; fit_rows = []
            for validation_name in source_names:
                train_names = tuple(name for name in source_names if name != validation_name)
                train = {name: sessions[name] for name in train_names}
                manifold = fit_calibration_aware_manifold(
                    train, spec, device=target_device,
                    seed=_stable_seed("ca-inner", int(seed_offset), target_name, validation_name, asdict(spec)),
                )
                x_mean, x_scale = _x_normalizer(train); scores = {}
                for ridge_lambda in deployment_lambda_grid:
                    metrics, _, _ = score_one_session(sessions[validation_name], manifold, x_mean=x_mean, x_scale=x_scale, ridge_lambda=float(ridge_lambda))
                    value = float(metrics["pooled_variance_weighted_r2"]); scores_by_lambda[float(ridge_lambda)].append(value); scores[str(float(ridge_lambda))] = value
                fit_rows.append({"validation_session": validation_name, "train_sessions": list(train_names), "fit_evidence": manifold.fit_evidence, "validation_r2": scores})
                del manifold
                if target_device.type == "cuda": torch.cuda.empty_cache()
            fit_sha = canonical_sha256(fit_rows)
            for ridge_lambda in deployment_lambda_grid:
                source_scores = scores_by_lambda[float(ridge_lambda)]
                candidates.append({"spec": asdict(spec), "spec_name": spec.name, "ridge_lambda_per_sample": float(ridge_lambda), "source_validation_r2": source_scores, "equal_source_session_mean_r2": float(np.mean(source_scores)), "inner_fits_sha256": fit_sha})
        winner_index = max(range(len(candidates)), key=lambda index: (candidates[index]["equal_source_session_mean_r2"], -index))
        winner = candidates[winner_index]; winner_spec = CalibrationAwareSpec(**winner["spec"])
        sources = {name: sessions[name] for name in source_names}
        manifold = fit_calibration_aware_manifold(sources, winner_spec, device=target_device, seed=_stable_seed("ca-final", int(seed_offset), target_name, asdict(winner_spec)))
        x_mean, x_scale = _x_normalizer(sources)
        metrics, truth, prediction = score_one_session(sessions[target_name], manifold, x_mean=x_mean, x_scale=x_scale, ridge_lambda=float(winner["ridge_lambda_per_sample"]))
        baseline_metrics = baseline["folds"][target_name]["target_metrics"]
        folds[target_name] = {"source_sessions": list(source_names), "target_excluded": True, "selected": winner, "candidate_table": candidates, "candidate_table_sha256": canonical_sha256(candidates), "final_fit": manifold.fit_evidence, "target_metrics": metrics, "baseline_target_metrics": baseline_metrics, "delta_vs_directridge": float(metrics["pooled_variance_weighted_r2"] - baseline_metrics["pooled_variance_weighted_r2"]), "input_path": sessions[target_name].input_path, "input_sha256": sessions[target_name].input_sha256}
        truth_rows.append(truth); prediction_rows.append(prediction); del manifold
        if target_device.type == "cuda": torch.cuda.empty_cache()
    scores = np.asarray([folds[name]["target_metrics"]["pooled_variance_weighted_r2"] for name in names]); baseline_scores = np.asarray([folds[name]["baseline_target_metrics"]["pooled_variance_weighted_r2"] for name in names]); delta = scores - baseline_scores
    return {
        "schema": "m1_m10_source_frozen_calibration_aware_autoencoder_nested_loso_v1", "status": "COMPLETE_CALIBRATION_AWARE_AE_SCREEN",
        "protocol": {"outer_target_excluded": True, "inner_validation_session_excluded": True, "target_backward_steps": 0, "target_optimizer_steps": 0, "deployment": "frozen encoder/decoder plus one closed-form M10 ridge"},
        "candidate_specs": [asdict(spec) for spec in specs], "deployment_lambda_grid": [float(value) for value in deployment_lambda_grid], "seed_offset": int(seed_offset), "sessions": list(names), "output_names": list(sessions[names[0]].output_names), "folds": folds,
        "equal_session": {"mean_r2": float(scores.mean()), "median_r2": float(np.median(scores)), "per_session_r2": dict(zip(names, scores.tolist()))},
        "directridge_equal_session": baseline["equal_session"], "paired_delta_vs_directridge": {"mean": float(delta.mean()), "median": float(np.median(delta)), "positive_sessions": int((delta > 0).sum()), "total_sessions": len(names), "per_session": dict(zip(names, delta.tolist()))},
        "pooled": regression_metrics(np.concatenate(truth_rows), np.concatenate(prediction_rows)), "target_used_for_selection": False, "elapsed_seconds": float(time.perf_counter() - started),
    }
