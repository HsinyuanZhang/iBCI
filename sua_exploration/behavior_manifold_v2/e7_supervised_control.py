"""E7: supervised-but-deployment-mismatched middle control.

The published comparison matrix contrasts the calibration-aware MLP (supervised
AND deployment-matched: +0.071847) against unsupervised controls (PCA,
predictive PCA, reconstruction MLP).  Supervision and deployment-matching are
therefore bundled.  E7 isolates them with the missing middle cell: the same
3,224-parameter 16->64->8->64->16 encoder/decoder family, trained end to end on
source sessions with a supervised objective that does NOT put the M10
calibration ridge in the loop.  Concretely, the differentiable operator mapping
neural activity to latents inside the training objective is an ordinary
FULL-SESSION closed-form ridge (fitted on all eval-valid bins of the source
session, per-sample lambda 0.3) instead of the M10-support ridge; the query
behaviour loss and the source-only reconstruction regularizer, the data, the
sampled query batches, the optimizer, the schedule, the seed rule, and the
nested leave-one-session-out selection all match the calibration-aware run
exactly.  Deployment then goes through the exact frozen rule: frozen
encoder/decoder plus ONE closed-form M10 neural-to-latent ridge on the target
calibration, governing convention.

Interpretation fork (pre-registered in the receipt): if E7 also gains about
+0.06, the active ingredient is supervised nonlinear reparameterization and the
handoff's deployment-matching claim must be weakened; if E7 gains little, the
deployment-matched objective is confirmed as the active ingredient.
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

from sua_exploration.behavior_autoencoder_v1.calibration_aware import _ridge_projection
from sua_exploration.behavior_autoencoder_v1.core import (
    AutoencoderSpec,
    BehaviorAutoencoder,
    state_sha256,
    weighted_reconstruction_loss,
)
from sua_exploration.behavior_autoencoder_v1.m1_screen import (
    FittedManifold,
    _moments,
    _session_behavior,
    _stable_seed,
    _x_normalizer,
    score_one_session,
)
from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    array_sha256,
    direct_ridge_loso,
    regression_metrics,
)

from .protocol import full_session_arrays


@dataclass(frozen=True, order=True)
class SupervisedMismatchedSpec:
    latent_dim: int
    hidden_dim: int
    reconstruction_weight: float
    raw_loss_fraction: float = 1.0
    activation: str = "gelu"
    pretrain_epochs: int = 20
    fine_steps: int = 300

    def validate(self, output_dim: int) -> None:
        AutoencoderSpec("mlp", self.latent_dim, self.hidden_dim, self.raw_loss_fraction, self.activation).validate(output_dim)
        assert self.reconstruction_weight >= 0.0 and np.isfinite(self.reconstruction_weight), "invalid reconstruction weight"
        assert self.pretrain_epochs >= 0 and self.fine_steps >= 1, "invalid schedule"

    @property
    def name(self) -> str:
        weight = str(self.reconstruction_weight).replace(".", "p")
        return f"sup_mis_q{self.latent_dim}_h{self.hidden_dim}_rw{weight}"


@dataclass(frozen=True)
class _SessionTensor:
    name: str
    full_x_augmented: torch.Tensor
    ridge_projection: torch.Tensor
    full_y: torch.Tensor
    query_x_augmented: torch.Tensor
    query_y: torch.Tensor


def fit_supervised_mismatched_manifold(
    sessions: Mapping[str, DirectRidgeSession],
    spec: SupervisedMismatchedSpec,
    *,
    device: torch.device,
    seed: int,
    meta_ridge_lambda: float = 0.3,
    reconstruction_batch_size: int = 4096,
    query_batch_size: int = 4096,
) -> FittedManifold:
    """Calibration-aware training loop with the M10 ridge replaced by a full-session ridge."""

    names = tuple(sorted(sessions))
    assert len(names) >= 2, "supervised-mismatched fit needs at least two source sessions"
    output_dim = sessions[names[0]].target.shape[1]
    spec.validate(output_dim)
    raw_by_session = {name: _session_behavior(sessions[name]) for name in names}
    raw_all = np.concatenate([raw_by_session[name] for name in names])
    behavior_mean, behavior_scale = _moments(raw_all)
    standardized_all = (raw_all - behavior_mean) / behavior_scale
    x_mean, x_scale = _x_normalizer(sessions)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = BehaviorAutoencoder(output_dim, spec.latent_dim, spec.hidden_dim, spec.activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    scale_tensor = torch.as_tensor(behavior_scale, dtype=torch.float32, device=device)
    reconstruction_tensor = torch.as_tensor(standardized_all, dtype=torch.float32, device=device)
    session_tensors: list[_SessionTensor] = []
    for name in names:
        # The ONLY difference from the calibration-aware objective: the in-loop
        # neural-to-latent operator is fitted on the FULL session (support plus
        # query), i.e. it is supervised but mismatched to the M10 deployment.
        full_x, full_y, _ = full_session_arrays(sessions[name])
        full_x_tensor = torch.as_tensor((full_x - x_mean) / x_scale, dtype=torch.float32, device=device)
        full_augmented = torch.cat((full_x_tensor, torch.ones((full_x_tensor.shape[0], 1), device=device)), dim=1)
        query_x, query_y, _ = sessions[name].aligned(0, split="query")
        query_x_tensor = torch.as_tensor((query_x - x_mean) / x_scale, dtype=torch.float32, device=device)
        query_augmented = torch.cat((query_x_tensor, torch.ones((query_x_tensor.shape[0], 1), device=device)), dim=1)
        session_tensors.append(_SessionTensor(
            name,
            full_augmented,
            _ridge_projection(full_x_tensor, meta_ridge_lambda),
            torch.as_tensor((full_y - behavior_mean) / behavior_scale, dtype=torch.float32, device=device),
            query_augmented,
            torch.as_tensor((query_y - behavior_mean) / behavior_scale, dtype=torch.float32, device=device),
        ))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    started = time.perf_counter()
    pretrain_history: list[float] = []
    for _ in range(spec.pretrain_epochs):
        model.train()
        order = torch.randperm(reconstruction_tensor.shape[0], generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, order.numel(), reconstruction_batch_size):
            index = order[start : start + reconstruction_batch_size].to(device)
            target = reconstruction_tensor[index]
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_reconstruction_loss(
                model(target), target, source_scale=scale_tensor, raw_loss_fraction=spec.raw_loss_fraction
            )
            assert bool(torch.isfinite(loss)), "pretraining loss became nonfinite"
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(target.shape[0])
            seen += int(target.shape[0])
        pretrain_history.append(total / float(seen))
    fine_history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_step = -1
    best_state: dict[str, torch.Tensor] | None = None
    for step in range(spec.fine_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction_losses: list[torch.Tensor] = []
        for row in session_tensors:
            latent_full = model.encode(row.full_y)
            beta = row.ridge_projection @ latent_full
            query_count = min(query_batch_size, row.query_y.shape[0])
            query_index = torch.randint(row.query_y.shape[0], (query_count,), generator=generator).to(device)
            predicted_latent = row.query_x_augmented[query_index] @ beta
            predicted_behavior = model.decode(predicted_latent)
            prediction_losses.append(weighted_reconstruction_loss(
                predicted_behavior, row.query_y[query_index], source_scale=scale_tensor,
                raw_loss_fraction=spec.raw_loss_fraction,
            ))
        reconstruction_index = torch.randint(
            reconstruction_tensor.shape[0], (min(reconstruction_batch_size, reconstruction_tensor.shape[0]),),
            generator=generator,
        ).to(device)
        reconstruction_target = reconstruction_tensor[reconstruction_index]
        reconstruction_loss = weighted_reconstruction_loss(
            model(reconstruction_target), reconstruction_target, source_scale=scale_tensor,
            raw_loss_fraction=spec.raw_loss_fraction,
        )
        prediction_loss = torch.stack(prediction_losses).mean()
        loss = prediction_loss + float(spec.reconstruction_weight) * reconstruction_loss
        assert bool(torch.isfinite(loss)), "supervised-mismatched loss became nonfinite"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_step = step
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if step % 10 == 0 or step == spec.fine_steps - 1:
            fine_history.append({
                "step": step, "loss": value,
                "prediction_loss": float(prediction_loss.detach()),
                "reconstruction_loss": float(reconstruction_loss.detach()),
            })
    assert best_state is not None and best_step >= 0, "fit produced no state"
    model.load_state_dict(best_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    evidence = {
        "kind": "supervised_mismatched_mlp",
        "spec": asdict(spec),
        "source_sessions": list(names),
        "source_rows": int(raw_all.shape[0]),
        "source_raw_sha256": array_sha256(raw_all),
        "behavior_mean_sha256": array_sha256(behavior_mean),
        "behavior_scale_sha256": array_sha256(behavior_scale),
        "x_mean_sha256": array_sha256(x_mean),
        "x_scale_sha256": array_sha256(x_scale),
        "in_loop_ridge_support": "full_session",
        "in_loop_ridge_lambda_per_sample": float(meta_ridge_lambda),
        "deployment_matched_objective": False,
        "seed": int(seed),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "state_sha256": state_sha256(model),
        "best_step": int(best_step),
        "best_loss": float(best_loss),
        "fine_history_sha256": _json_sha(fine_history),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    ae_spec = AutoencoderSpec("mlp", spec.latent_dim, spec.hidden_dim, spec.raw_loss_fraction, spec.activation)
    return FittedManifold(ae_spec, behavior_mean, behavior_scale, None, model, device, evidence)


def _json_sha(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    ).hexdigest()


def nested_supervised_mismatched_loso(
    sessions: Mapping[str, DirectRidgeSession],
    specs: Sequence[SupervisedMismatchedSpec],
    *,
    device: str,
    deployment_lambda_grid: Sequence[float] = (0.1, 0.3),
    seed_offset: int = 0,
) -> dict[str, Any]:
    names = tuple(sorted(sessions))
    assert len(names) == 4, "E7 requires the exact four-session surface"
    specs = tuple(specs)
    assert specs and len({spec.name for spec in specs}) == len(specs), "candidate drift"
    device_object = torch.device(device)
    baseline = direct_ridge_loso(sessions, lag_grid=(0,), lambda_grid=deployment_lambda_grid)
    folds: dict[str, Any] = {}
    truth_rows: list[np.ndarray] = []
    prediction_rows: list[np.ndarray] = []
    started = time.perf_counter()
    for target_name in names:
        source_names = tuple(name for name in names if name != target_name)
        candidates: list[dict[str, Any]] = []
        inner_rows: list[dict[str, Any]] = []
        for spec in specs:
            scores_by_lambda = {float(value): [] for value in deployment_lambda_grid}
            for validation_name in source_names:
                train_names = tuple(name for name in source_names if name != validation_name)
                train = {name: sessions[name] for name in train_names}
                manifold = fit_supervised_mismatched_manifold(
                    train, spec, device=device_object,
                    seed=_stable_seed("e7m-inner", int(seed_offset), target_name, validation_name, asdict(spec)),
                )
                x_mean, x_scale = _x_normalizer(train)
                scores = {}
                for ridge_lambda in deployment_lambda_grid:
                    metrics, _, _ = score_one_session(
                        sessions[validation_name], manifold, x_mean=x_mean, x_scale=x_scale,
                        ridge_lambda=float(ridge_lambda),
                    )
                    value = float(metrics["pooled_variance_weighted_r2"])
                    scores_by_lambda[float(ridge_lambda)].append(value)
                    scores[str(float(ridge_lambda))] = value
                inner_rows.append({
                    "validation_session": validation_name,
                    "train_sessions": list(train_names),
                    "state_sha256": manifold.fit_evidence["state_sha256"],
                    "validation_r2": scores,
                })
                del manifold
                if device_object.type == "cuda":
                    torch.cuda.empty_cache()
            for ridge_lambda in deployment_lambda_grid:
                source_scores = scores_by_lambda[float(ridge_lambda)]
                candidates.append({
                    "spec": asdict(spec), "spec_name": spec.name,
                    "ridge_lambda_per_sample": float(ridge_lambda),
                    "source_validation_r2": source_scores,
                    "equal_source_session_mean_r2": float(np.mean(source_scores)),
                })
        winner_index = max(range(len(candidates)), key=lambda index: (candidates[index]["equal_source_session_mean_r2"], -index))
        winner = candidates[winner_index]
        winner_spec = SupervisedMismatchedSpec(**winner["spec"])
        sources = {name: sessions[name] for name in source_names}
        manifold = fit_supervised_mismatched_manifold(
            sources, winner_spec, device=device_object,
            seed=_stable_seed("e7m-final", int(seed_offset), target_name, asdict(winner_spec)),
        )
        x_mean, x_scale = _x_normalizer(sources)
        metrics, truth, prediction = score_one_session(
            sessions[target_name], manifold, x_mean=x_mean, x_scale=x_scale,
            ridge_lambda=float(winner["ridge_lambda_per_sample"]),
        )
        baseline_metrics = baseline["folds"][target_name]["target_metrics"]
        folds[target_name] = {
            "source_sessions": list(source_names),
            "target_excluded": True,
            "selected": winner,
            "inner_fits": inner_rows,
            "final_fit": manifold.fit_evidence,
            "target_metrics": metrics,
            "baseline_target_metrics": baseline_metrics,
            "delta_vs_directridge": float(metrics["pooled_variance_weighted_r2"] - baseline_metrics["pooled_variance_weighted_r2"]),
        }
        truth_rows.append(truth)
        prediction_rows.append(prediction)
        del manifold
        if device_object.type == "cuda":
            torch.cuda.empty_cache()
    scores = np.asarray([folds[name]["target_metrics"]["pooled_variance_weighted_r2"] for name in names])
    baseline_scores = np.asarray([folds[name]["baseline_target_metrics"]["pooled_variance_weighted_r2"] for name in names])
    delta = scores - baseline_scores
    return {
        "schema": "m1_m10_supervised_mismatched_manifold_nested_loso_v1",
        "status": "COMPLETE_SUPERVISED_MISMATCHED_CONTROL",
        "seed_offset": int(seed_offset),
        "protocol": {
            "outer_target_excluded": True,
            "inner_validation_session_excluded": True,
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "objective": "supervised query behaviour loss through a full-session (deployment-mismatched) closed-form latent ridge plus source-only reconstruction regularizer",
            "deployment": "frozen encoder/decoder plus one closed-form M10 neural-to-latent ridge",
        },
        "candidate_specs": [asdict(spec) for spec in specs],
        "deployment_lambda_grid": [float(value) for value in deployment_lambda_grid],
        "sessions": list(names),
        "folds": folds,
        "equal_session": {"mean_r2": float(scores.mean()), "per_session_r2": dict(zip(names, scores.tolist()))},
        "directridge_equal_session": baseline["equal_session"],
        "paired_delta_vs_directridge": {
            "mean": float(delta.mean()), "positive_sessions": int((delta > 0).sum()),
            "total_sessions": len(names), "per_session": dict(zip(names, delta.tolist())),
        },
        "pooled": regression_metrics(np.concatenate(truth_rows), np.concatenate(prediction_rows)),
        "target_used_for_selection": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def three_seed_supervised_ensemble(
    sessions: Mapping[str, DirectRidgeSession],
    specs: Sequence[SupervisedMismatchedSpec],
    *,
    device: str,
    seed_offsets: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Run the nested E7 control at three seed offsets and ensemble the deployments.

    Each seed's final manifold is refitted with the stored seed rule and verified
    against that seed's nested run by state SHA and prediction SHA before the
    mean is taken, mirroring the behavior_autoencoder_v1 ensemble discipline.
    """

    device_object = torch.device(device)
    seed_results = [
        nested_supervised_mismatched_loso(sessions, specs, device=device, seed_offset=int(offset))
        for offset in seed_offsets
    ]
    names = tuple(sorted(sessions))
    reference = seed_results[0]
    for body in seed_results[1:]:
        assert body["directridge_equal_session"] == reference["directridge_equal_session"], "seed baseline drift"
    folds: dict[str, Any] = {}
    truth_all = []
    prediction_all = []
    for name in names:
        seed_rows = []
        seed_predictions = []
        truth_reference = None
        for offset, body in zip(seed_offsets, seed_results):
            fold = body["folds"][name]
            sources = {source: sessions[source] for source in fold["source_sessions"]}
            manifold = fit_supervised_mismatched_manifold(
                sources, SupervisedMismatchedSpec(**fold["selected"]["spec"]),
                device=device_object,
                seed=_stable_seed("e7m-final", int(offset), name, fold["selected"]["spec"]),
            )
            assert manifold.fit_evidence["state_sha256"] == fold["final_fit"]["state_sha256"], (
                f"{name}/offset{offset}: E7 final refit state SHA drift"
            )
            x_mean, x_scale = _x_normalizer(sources)
            metrics, truth, prediction = score_one_session(
                sessions[name], manifold, x_mean=x_mean, x_scale=x_scale,
                ridge_lambda=float(fold["selected"]["ridge_lambda_per_sample"]),
            )
            assert metrics["prediction_sha256"] == fold["target_metrics"]["prediction_sha256"], (
                f"{name}/offset{offset}: E7 final refit prediction drift"
            )
            if truth_reference is None:
                truth_reference = truth
            else:
                assert np.array_equal(truth_reference, truth), "seed truth drift"
            seed_predictions.append(prediction)
            seed_rows.append({
                "seed_offset": int(offset),
                "selected": fold["selected"],
                "state_sha256": manifold.fit_evidence["state_sha256"],
                "single_seed_r2": float(metrics["pooled_variance_weighted_r2"]),
                "prediction_sha256": array_sha256(prediction),
            })
            del manifold
            if device_object.type == "cuda":
                torch.cuda.empty_cache()
        ensemble_prediction = np.mean(np.stack(seed_predictions, axis=0), axis=0, dtype=np.float64)
        metrics = regression_metrics(truth_reference, ensemble_prediction)
        baseline = reference["folds"][name]["baseline_target_metrics"]
        folds[name] = {
            "seed_models": seed_rows,
            "ensemble_rule": "unweighted arithmetic mean of three source-selected predictions; no target selection",
            "target_metrics": metrics,
            "baseline_target_metrics": baseline,
            "delta_vs_directridge": float(metrics["pooled_variance_weighted_r2"] - baseline["pooled_variance_weighted_r2"]),
        }
        truth_all.append(truth_reference)
        prediction_all.append(ensemble_prediction)
    scores = np.asarray([folds[name]["target_metrics"]["pooled_variance_weighted_r2"] for name in names])
    baseline_scores = np.asarray([folds[name]["baseline_target_metrics"]["pooled_variance_weighted_r2"] for name in names])
    delta = scores - baseline_scores
    per_seed_equal = {
        str(int(offset)): body["equal_session"]["mean_r2"] for offset, body in zip(seed_offsets, seed_results)
    }
    gain = float(delta.mean())
    if gain >= 0.75 * 0.071847:
        reading = (
            "E7 recovers most of the calibration-aware gain: supervised nonlinear reparameterization is the "
            "active ingredient and the deployment-matching claim must be weakened"
        )
    elif gain <= 0.25 * 0.071847:
        reading = (
            "E7 recovers little of the gain: the deployment-matched objective is confirmed as the active "
            "ingredient over and above supervision"
        )
    else:
        reading = "E7 recovers a partial share of the gain: both supervision and deployment-matching contribute"
    return {
        "schema": "m1_m10_supervised_mismatched_three_seed_ensemble_v1",
        "status": "COMPLETE_SUPERVISED_MISMATCHED_MIDDLE_CONTROL",
        "missing_control_framing": (
            "the published matrix bundles supervision with deployment-matching; E7 supplies the supervised "
            "but deployment-mismatched cell"
        ),
        "seed_offsets": [int(offset) for offset in seed_offsets],
        "per_seed_equal_session_mean": per_seed_equal,
        "per_seed_nested_results": {
            str(int(offset)): {
                "equal_session": body["equal_session"],
                "paired_delta_vs_directridge": body["paired_delta_vs_directridge"],
                "selected_per_session": {
                    name: {"spec_name": body["folds"][name]["selected"]["spec_name"],
                           "ridge_lambda_per_sample": body["folds"][name]["selected"]["ridge_lambda_per_sample"]}
                    for name in names
                },
            }
            for offset, body in zip(seed_offsets, seed_results)
        },
        "folds": folds,
        "equal_session": {"mean_r2": float(scores.mean()), "per_session_r2": dict(zip(names, scores.tolist()))},
        "directridge_equal_session": reference["directridge_equal_session"],
        "paired_delta_vs_directridge": {
            "mean": gain, "positive_sessions": int((delta > 0).sum()), "total_sessions": len(names),
            "per_session": dict(zip(names, delta.tolist())),
        },
        "calibration_aware_reference_gain": 0.071847,
        "pre_registered_reading": {
            "fork": "E7 ~ +0.06 -> supervision is the ingredient; E7 small -> deployment-matching confirmed",
            "observed_gain": gain,
            "reading": reading,
        },
        "pooled": regression_metrics(np.concatenate(truth_all), np.concatenate(prediction_all)),
        "target_used_for_selection": False,
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
            "deployment_rule_identical_to_calibration_aware": True,
        },
    }
