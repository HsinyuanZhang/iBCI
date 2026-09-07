"""Nested M1 M10 screen for source-frozen output manifolds.

The outer target session is never used to train or select the manifold. For
each candidate, an inner leave-one-source-session-out loop fits the behavior
manifold on the remaining sources, fits one closed-form M10 neural-to-latent
ridge on the held source, and scores that source's post-M10 query.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    array_sha256,
    canonical_sha256,
    direct_ridge_loso,
    regression_metrics,
)

from .core import (
    AutoencoderSpec,
    BehaviorAutoencoder,
    PcaManifold,
    fit_affine_ridge,
    fit_pca_manifold,
    need,
    predict_affine_ridge,
    state_sha256,
    weighted_reconstruction_loss,
)


NORMALIZER_FLOOR = 1.0e-8
SCREEN_LAMBDA_GRID: tuple[float, ...] = (3.0e-2, 1.0e-1, 3.0e-1, 1.0, 3.0)


def _moments(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    need(array.ndim == 2 and array.shape[0] > array.shape[1] and np.isfinite(array).all(), "invalid normalizer values")
    mean = array.mean(axis=0, dtype=np.float64)
    scale = np.maximum(array.std(axis=0, dtype=np.float64), NORMALIZER_FLOOR)
    need(np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 0.0), "invalid normalizer moments")
    return mean, scale


def _session_behavior(session: DirectRidgeSession) -> np.ndarray:
    legal = np.asarray(session.eval_mask, dtype=bool) & (session.trial_id > 0)
    result = np.asarray(session.target[legal], dtype=np.float64)
    need(result.shape[0] > result.shape[1] and np.isfinite(result).all(), f"{session.name}: empty behavior source")
    return result


def _stable_seed(*parts: Any) -> int:
    body = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "little") % (2**31 - 1)


@dataclass
class FittedManifold:
    spec: AutoencoderSpec
    behavior_mean: np.ndarray
    behavior_scale: np.ndarray
    pca: PcaManifold | None
    mlp: BehaviorAutoencoder | None
    device: torch.device
    fit_evidence: dict[str, Any]

    def encode(self, raw_value: np.ndarray, *, batch_size: int = 32768) -> np.ndarray:
        standardized = (np.asarray(raw_value, dtype=np.float64) - self.behavior_mean) / self.behavior_scale
        if self.pca is not None:
            return self.pca.encode_numpy(standardized)
        need(self.mlp is not None, "missing MLP manifold")
        rows: list[np.ndarray] = []
        self.mlp.eval()
        with torch.inference_mode():
            for start in range(0, standardized.shape[0], batch_size):
                value = torch.as_tensor(standardized[start : start + batch_size], device=self.device, dtype=torch.float32)
                rows.append(self.mlp.encode(value).cpu().numpy().astype(np.float64))
        return np.concatenate(rows, axis=0)

    def decode(self, latent: np.ndarray, *, batch_size: int = 32768) -> np.ndarray:
        if self.pca is not None:
            standardized = self.pca.decode_numpy(latent)
        else:
            need(self.mlp is not None, "missing MLP manifold")
            rows: list[np.ndarray] = []
            self.mlp.eval()
            with torch.inference_mode():
                for start in range(0, latent.shape[0], batch_size):
                    value = torch.as_tensor(latent[start : start + batch_size], device=self.device, dtype=torch.float32)
                    rows.append(self.mlp.decode(value).cpu().numpy().astype(np.float64))
            standardized = np.concatenate(rows, axis=0)
        raw = standardized * self.behavior_scale + self.behavior_mean
        need(np.isfinite(raw).all(), "decoded behavior became nonfinite")
        return raw


def _split_train_validation(values_by_session: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    evidence: dict[str, Any] = {}
    for name in sorted(values_by_session):
        value = np.asarray(values_by_session[name], dtype=np.float64)
        offset = _stable_seed("validation", name) % 10
        index = np.arange(value.shape[0], dtype=np.int64)
        is_validation = index % 10 == offset
        need(is_validation.any() and (~is_validation).any(), f"{name}: invalid validation split")
        train.append(value[~is_validation]); validation.append(value[is_validation])
        evidence[name] = {
            "rows": int(value.shape[0]),
            "train_rows": int((~is_validation).sum()),
            "validation_rows": int(is_validation.sum()),
            "validation_indices_sha256": array_sha256(index[is_validation]),
        }
    return np.concatenate(train), np.concatenate(validation), evidence


def fit_source_manifold(
    sessions: Mapping[str, DirectRidgeSession],
    spec: AutoencoderSpec,
    *,
    device: torch.device,
    seed: int,
    max_epochs: int = 100,
    patience: int = 12,
    batch_size: int = 8192,
) -> FittedManifold:
    names = tuple(sorted(sessions))
    need(len(names) >= 2, "a source manifold needs at least two sessions")
    output_dims = {sessions[name].target.shape[1] for name in names}
    need(len(output_dims) == 1, "source output dimension drift")
    output_dim = next(iter(output_dims)); spec.validate(output_dim)
    raw_by_session = {name: _session_behavior(sessions[name]) for name in names}
    raw_all = np.concatenate([raw_by_session[name] for name in names])
    behavior_mean, behavior_scale = _moments(raw_all)
    standardized = {name: (value - behavior_mean) / behavior_scale for name, value in raw_by_session.items()}
    started = time.perf_counter()
    common = {
        "source_sessions": list(names),
        "source_rows": int(raw_all.shape[0]),
        "source_raw_sha256": array_sha256(raw_all),
        "behavior_mean": behavior_mean.tolist(),
        "behavior_scale": behavior_scale.tolist(),
        "behavior_mean_sha256": array_sha256(behavior_mean),
        "behavior_scale_sha256": array_sha256(behavior_scale),
        "seed": int(seed),
    }
    joined = np.concatenate([standardized[name] for name in names])
    if spec.kind == "pca":
        pca = fit_pca_manifold(joined, spec.latent_dim)
        reconstructed = pca.decode_numpy(pca.encode_numpy(joined))
        common.update({
            "kind": "pca",
            "components_sha256": array_sha256(pca.components),
            "standardized_reconstruction_mse": float(np.square(reconstructed - joined).mean()),
            "elapsed_seconds": float(time.perf_counter() - started),
        })
        return FittedManifold(spec, behavior_mean, behavior_scale, pca, None, device, common)

    train_np, validation_np, split_evidence = _split_train_validation(standardized)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = BehaviorAutoencoder(output_dim, spec.latent_dim, spec.hidden_dim, spec.activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    train_tensor = torch.as_tensor(train_np, dtype=torch.float32, device=device)
    validation_tensor = torch.as_tensor(validation_np, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(behavior_scale, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed + 1)
    best_loss = math.inf; best_epoch = -1; best_state: dict[str, torch.Tensor] | None = None
    wait = 0; history: list[dict[str, float | int]] = []
    for epoch in range(max_epochs):
        model.train()
        permutation = torch.randperm(train_tensor.shape[0], generator=generator)
        loss_sum = 0.0; rows_seen = 0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size].to(device=device)
            target = train_tensor[index]
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_reconstruction_loss(model(target), target, source_scale=scale_tensor, raw_loss_fraction=spec.raw_loss_fraction)
            need(bool(torch.isfinite(loss)), "MLP training loss became nonfinite")
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * int(target.shape[0]); rows_seen += int(target.shape[0])
        model.eval()
        with torch.inference_mode():
            validation_loss = float(weighted_reconstruction_loss(
                model(validation_tensor), validation_tensor,
                source_scale=scale_tensor, raw_loss_fraction=spec.raw_loss_fraction,
            ).cpu())
        train_loss = loss_sum / float(rows_seen)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1.0e-7:
            best_loss = validation_loss; best_epoch = epoch; wait = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    need(best_state is not None and best_epoch >= 0 and np.isfinite(best_loss), "MLP never produced a valid state")
    model.load_state_dict(best_state, strict=True); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    common.update({
        "kind": "mlp",
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": int(best_epoch),
        "epochs_executed": len(history),
        "best_validation_loss": float(best_loss),
        "final_train_loss": float(history[-1]["train_loss"]),
        "state_sha256": state_sha256(model),
        "split": split_evidence,
        "history_sha256": canonical_sha256(history),
        "elapsed_seconds": float(time.perf_counter() - started),
    })
    return FittedManifold(spec, behavior_mean, behavior_scale, None, model, device, common)


def _x_normalizer(sessions: Mapping[str, DirectRidgeSession]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([sessions[name].aligned(0, split="support")[0] for name in sorted(sessions)])
    return _moments(values)


def score_one_session(
    session: DirectRidgeSession,
    manifold: FittedManifold,
    *,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    ridge_lambda: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    support_x, support_y, support_indices = session.aligned(0, split="support")
    query_x, query_y, query_indices = session.aligned(0, split="query")
    support_z = manifold.encode(support_y)
    weight, intercept = fit_affine_ridge((support_x - x_mean) / x_scale, support_z, ridge_lambda=ridge_lambda)
    prediction_z = predict_affine_ridge((query_x - x_mean) / x_scale, weight, intercept)
    prediction_y = manifold.decode(prediction_z)
    metrics = regression_metrics(query_y, prediction_y)
    metrics.update({
        "support_bins": int(support_x.shape[0]), "query_bins": int(query_x.shape[0]),
        "support_indices_sha256": array_sha256(support_indices), "query_indices_sha256": array_sha256(query_indices),
        "latent_support_sha256": array_sha256(support_z), "ridge_weight_sha256": array_sha256(weight),
        "ridge_intercept_sha256": array_sha256(intercept),
    })
    return metrics, query_y, prediction_y


def nested_manifold_loso(
    sessions: Mapping[str, DirectRidgeSession],
    specs: Sequence[AutoencoderSpec],
    *,
    device: str,
    lambda_grid: Sequence[float] = SCREEN_LAMBDA_GRID,
    max_epochs: int = 100,
    patience: int = 12,
) -> dict[str, Any]:
    names = tuple(sorted(sessions))
    need(len(names) == 4, "M1 screen requires the exact four-session public surface")
    for session in sessions.values(): session.validate()
    specs = tuple(specs)
    need(specs and len({spec.name for spec in specs}) == len(specs), "candidate specs are empty or duplicated")
    device_object = torch.device(device)
    if device_object.type == "cuda": need(torch.cuda.is_available(), "requested CUDA is unavailable")
    baseline = direct_ridge_loso(sessions, lag_grid=(0,), lambda_grid=lambda_grid)
    folds: dict[str, Any] = {}; pooled_truth: list[np.ndarray] = []; pooled_prediction: list[np.ndarray] = []
    total_started = time.perf_counter()
    for target_name in names:
        source_names = tuple(name for name in names if name != target_name)
        candidate_rows: list[dict[str, Any]] = []
        for spec in specs:
            source_scores_by_lambda = {float(value): [] for value in lambda_grid}
            inner_fits: list[dict[str, Any]] = []
            for validation_name in source_names:
                train_names = tuple(name for name in source_names if name != validation_name)
                train_sessions = {name: sessions[name] for name in train_names}
                manifold = fit_source_manifold(
                    train_sessions, spec, device=device_object,
                    seed=_stable_seed("inner", target_name, validation_name, asdict(spec)),
                    max_epochs=max_epochs, patience=patience,
                )
                x_mean, x_scale = _x_normalizer(train_sessions)
                validation_scores: dict[str, float] = {}
                for ridge_lambda in lambda_grid:
                    metrics, _, _ = score_one_session(
                        sessions[validation_name], manifold, x_mean=x_mean, x_scale=x_scale,
                        ridge_lambda=float(ridge_lambda),
                    )
                    value = float(metrics["pooled_variance_weighted_r2"])
                    source_scores_by_lambda[float(ridge_lambda)].append(value)
                    validation_scores[str(float(ridge_lambda))] = value
                inner_fits.append({
                    "validation_session": validation_name, "manifold_train_sessions": list(train_names),
                    "validation_excluded_from_manifold_fit": validation_name not in train_names,
                    "fit_evidence": manifold.fit_evidence, "validation_r2_by_lambda": validation_scores,
                })
                del manifold
                if device_object.type == "cuda": torch.cuda.empty_cache()
            inner_sha = canonical_sha256(inner_fits)
            for ridge_lambda in lambda_grid:
                source_scores = source_scores_by_lambda[float(ridge_lambda)]
                candidate_rows.append({
                    "spec": asdict(spec), "spec_name": spec.name, "ridge_lambda_per_sample": float(ridge_lambda),
                    "source_validation_r2": source_scores,
                    "equal_source_session_mean_r2": float(np.mean(source_scores, dtype=np.float64)),
                    "inner_fits_sha256": inner_sha,
                })
        best_index = max(range(len(candidate_rows)), key=lambda index: (candidate_rows[index]["equal_source_session_mean_r2"], -index))
        best = candidate_rows[best_index]; selected_spec = AutoencoderSpec(**best["spec"])
        final_sources = {name: sessions[name] for name in source_names}
        final_manifold = fit_source_manifold(
            final_sources, selected_spec, device=device_object,
            seed=_stable_seed("outer-final", target_name, asdict(selected_spec)),
            max_epochs=max_epochs, patience=patience,
        )
        x_mean, x_scale = _x_normalizer(final_sources)
        metrics, truth, prediction = score_one_session(
            sessions[target_name], final_manifold, x_mean=x_mean, x_scale=x_scale,
            ridge_lambda=float(best["ridge_lambda_per_sample"]),
        )
        baseline_metrics = baseline["folds"][target_name]["target_metrics"]
        folds[target_name] = {
            "target_session": target_name, "source_sessions": list(source_names),
            "target_excluded_from_all_manifold_training_and_selection": target_name not in source_names,
            "selected": best, "candidate_count": len(candidate_rows),
            "candidate_table_sha256": canonical_sha256(candidate_rows), "candidate_table": candidate_rows,
            "final_manifold_fit": final_manifold.fit_evidence, "target_metrics": metrics,
            "baseline_target_metrics": baseline_metrics,
            "delta_vs_directridge": float(metrics["pooled_variance_weighted_r2"] - baseline_metrics["pooled_variance_weighted_r2"]),
            "target_boundary_bin": sessions[target_name].boundary, "input_path": sessions[target_name].input_path,
            "input_sha256": sessions[target_name].input_sha256,
        }
        pooled_truth.append(truth); pooled_prediction.append(prediction)
        del final_manifold
        if device_object.type == "cuda": torch.cuda.empty_cache()
    scores = np.asarray([folds[name]["target_metrics"]["pooled_variance_weighted_r2"] for name in names], dtype=np.float64)
    baseline_scores = np.asarray([folds[name]["baseline_target_metrics"]["pooled_variance_weighted_r2"] for name in names], dtype=np.float64)
    deltas = scores - baseline_scores
    return {
        "schema": "m1_m10_source_frozen_behavior_manifold_nested_loso_v1",
        "status": "COMPLETE_EXPLORATORY_SOURCE_FROZEN_MANIFOLD_SCREEN",
        "protocol": {
            "outer": "target session excluded from manifold fitting and all candidate selection",
            "inner": "each source validation session excluded from its candidate manifold fit",
            "manifold_labels": "eval-valid behavior from inner source sessions only",
            "target_adaptation": "M10 labels encode through frozen manifold; one affine closed-form neural-to-latent ridge; no gradient",
            "query": "strict post-M10 same-trial lag-0 bins", "metric": "per-session variance-weighted 16-output R2; equal session aggregation",
            "target_backward_steps": 0, "target_optimizer_steps": 0,
        },
        "device": str(device_object), "candidate_specs": [asdict(spec) for spec in specs],
        "lambda_grid_per_sample": [float(value) for value in lambda_grid], "sessions": list(names),
        "output_names": list(sessions[names[0]].output_names), "folds": folds,
        "equal_session": {"mean_r2": float(scores.mean()), "median_r2": float(np.median(scores)), "per_session_r2": dict(zip(names, scores.tolist()))},
        "directridge_equal_session": baseline["equal_session"],
        "paired_delta_vs_directridge": {"mean": float(deltas.mean()), "median": float(np.median(deltas)), "positive_sessions": int((deltas > 0.0).sum()), "total_sessions": int(deltas.size), "per_session": dict(zip(names, deltas.tolist()))},
        "pooled": regression_metrics(np.concatenate(pooled_truth), np.concatenate(pooled_prediction)),
        "baseline_receipt_sha256": canonical_sha256(baseline), "target_support_or_query_used_for_candidate_selection": False,
        "elapsed_seconds": float(time.perf_counter() - total_started),
    }
