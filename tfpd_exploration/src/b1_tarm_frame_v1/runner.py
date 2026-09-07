"""Source-only paired pilot for the framewise B1 TARM model."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.b1_sfcj_v1.constants import N_CHANNELS, N_FREQ, N_SPEC_FRAMES, VALID_END, VALID_START
from tfpd_exploration.src.b1_sfcj_v1.metric import official_metric_from_trials
from tfpd_exploration.src.b1_sfcj_v1.util import sha256_array
from tfpd_exploration.src.b1_tarm_v1.dataset import FoldData, Sample, build_fold_data

from .model import build_arms, parameter_count
from .plan import (
    ARMS,
    EPOCHS,
    FOLD,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    RESIDUAL_LOSS_WEIGHT,
    RESULT_ROOT,
    SCHEMA,
    SEED,
    UNIT_DROPOUT_P,
    WEIGHT_DECAY,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _to_tensors(sample: Sample, law: str, device: torch.device) -> dict:
    history = sample.history if law == "GROWING" else sample.history[:3]
    return {
        "current": torch.as_tensor(sample.current, dtype=torch.float32, device=device),
        "history": torch.as_tensor(np.stack(history), dtype=torch.float32, device=device),
        "carrier": torch.as_tensor(sample.carrier, dtype=torch.float32, device=device),
        "template_stdlog": torch.as_tensor(sample.template_stdlog, dtype=torch.float32, device=device),
        "target_stdlog": torch.as_tensor(sample.target_stdlog, dtype=torch.float32, device=device),
        "target_raw": torch.as_tensor(sample.target_raw, dtype=torch.float32, device=device),
    }


def _loss(output: dict[str, torch.Tensor], tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    pred = output["raw_valid"]
    target = tensors["target_raw"][:, VALID_START:VALID_END].T
    pmin, pmax = pred.min(), pred.max()
    tmin, tmax = target.min(), target.max()
    if pmax <= pmin or tmax <= tmin:
        raise RuntimeError("zero range")
    official = torch.mean(((pred - pmin) / (pmax - pmin) - (target - tmin) / (tmax - tmin)) ** 2)
    target_delta = (
        tensors["target_stdlog"][:, VALID_START:VALID_END]
        - tensors["template_stdlog"][:, VALID_START:VALID_END]
    ).T
    residual = F.smooth_l1_loss(output["delta_stdlog"], target_delta)
    total = official + RESIDUAL_LOSS_WEIGHT * residual
    return total, {
        "total": float(total.detach().cpu()),
        "official_surrogate": float(official.detach().cpu()),
        "residual_smooth_l1": float(residual.detach().cpu()),
    }


def _unit_mask(rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    keep = rng.random(N_CHANNELS) >= UNIT_DROPOUT_P
    if not keep.any():
        keep[int(rng.integers(0, N_CHANNELS))] = True
    return torch.as_tensor(keep.astype(np.float32), device=device)


def _trial_raw(output: dict[str, torch.Tensor], template_raw: np.ndarray) -> np.ndarray:
    pred = np.asarray(template_raw, dtype=np.float64).copy()
    pred[:, VALID_START:VALID_END] = output["raw_valid"].detach().cpu().numpy().T.astype(np.float64)
    if pred.shape != (N_FREQ, N_SPEC_FRAMES) or not np.isfinite(pred).all():
        raise RuntimeError("invalid trial prediction")
    return pred


@torch.no_grad()
def evaluate(model, data: FoldData, law: str, device: torch.device) -> dict:
    model.eval()
    preds, targets, flags, keys = [], [], [], []
    ones = torch.ones(N_CHANNELS, dtype=torch.float32, device=device)
    template = data.template_raw_by_date[data.val_date]
    for sample in data.val_samples:
        tensors = _to_tensors(sample, law, device)
        output = model.forward_trial(
            current=tensors["current"],
            history=tensors["history"],
            carrier=tensors["carrier"],
            template_stdlog=tensors["template_stdlog"],
            unit_mask=ones,
        )
        preds.append(_trial_raw(output, template))
        targets.append(sample.target_raw.astype(np.float64))
        flags.append(bool(sample.in_range))
        keys.append(sample.key)
    full = official_metric_from_trials(preds, targets)
    idx = [i for i, flag in enumerate(flags) if flag]
    inner = official_metric_from_trials([preds[i] for i in idx], [targets[i] for i in idx])
    return {
        "law": law,
        "full_mse": float(full["MSE Mean"]),
        "full_std": float(full["MSE Std."]),
        "in_range_mse": float(inner["MSE Mean"]),
        "in_range_std": float(inner["MSE Std."]),
        "n_full": int(full["n_trials"]),
        "n_in_range": int(inner["n_trials"]),
        "per_trial_mse": [float(v) for v in full["per_trial_mse"]],
        "prediction_sha256": sha256_array(np.stack(preds)),
        "keys": keys,
    }


def _template_score(data: FoldData) -> dict:
    pred = [data.template_raw_by_date[data.val_date].astype(np.float64) for _ in data.val_samples]
    target = [s.target_raw.astype(np.float64) for s in data.val_samples]
    full = official_metric_from_trials(pred, target)
    idx = [i for i, s in enumerate(data.val_samples) if s.in_range]
    inner = official_metric_from_trials([pred[i] for i in idx], [target[i] for i in idx])
    return {"full_mse": float(full["MSE Mean"]), "in_range_mse": float(inner["MSE Mean"])}


def _state_evidence(model) -> dict:
    carrier = model.post_pool[0].weight[:, model.d_model :]
    return {
        "head_norm": float(model.head.weight.detach().norm().cpu()),
        "carrier_column_norm": float(carrier.detach().norm().cpu()),
        "alpha": None if model.alpha is None else float(model.alpha.detach().cpu()),
    }


def run(
    *,
    fold: int = FOLD,
    seed: int = SEED,
    epochs: int = EPOCHS,
    out_dir: Path | None = None,
    device_name: str = "cuda:0",
) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("frame TARM requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA GPU")
    device = torch.device(device_name)
    out_dir = Path(out_dir) if out_dir is not None else RESULT_ROOT / f"pilot_fold{fold}_seed{seed}"
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    started = time.time()
    attempt = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "fold": int(fold),
        "seed": int(seed),
        "epochs": int(epochs),
        "arms": [row[0] for row in ARMS],
        "device": str(device),
        "started_unix": started,
    }
    _write(out_dir / "attempt.json", attempt)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    data = build_fold_data(fold)
    models = build_arms(log_mean=data.basis.mean, log_std=data.basis.std, seed=seed)
    for model in models.values():
        model.to(device)
    opts = {name: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) for name, model in models.items()}
    rng = np.random.default_rng(seed)
    template = _template_score(data)
    history = []
    steps = 0
    try:
        for epoch in range(1, int(epochs) + 1):
            order = rng.permutation(len(data.train_samples))
            accum = {name: [] for name in models}
            epoch_start = time.time()
            for index in order:
                sample = data.train_samples[int(index)]
                tensors = _to_tensors(sample, "GROWING", device)
                mask = _unit_mask(rng, device)
                for name, model in models.items():
                    model.train()
                    opt = opts[name]
                    opt.zero_grad(set_to_none=True)
                    output = model.forward_trial(
                        current=tensors["current"],
                        history=tensors["history"],
                        carrier=tensors["carrier"],
                        template_stdlog=tensors["template_stdlog"],
                        unit_mask=mask,
                    )
                    loss, evidence = _loss(output, tensors)
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    if not torch.isfinite(grad):
                        raise RuntimeError(f"non-finite gradient {name}")
                    opt.step()
                    accum[name].append(evidence)
                steps += 1
            validation = {}
            for name, model in models.items():
                validation[name] = {
                    "GROWING": evaluate(model, data, "GROWING", device),
                    "FIXED3": evaluate(model, data, "FIXED3", device),
                    "state": _state_evidence(model),
                }
            record = {
                "epoch": epoch,
                "steps": steps,
                "epoch_wall_seconds": time.time() - epoch_start,
                "train": {
                    name: {
                        key: float(np.mean([row[key] for row in values]))
                        for key in ("total", "official_surrogate", "residual_smooth_l1")
                    }
                    for name, values in accum.items()
                },
                "validation": validation,
            }
            history.append(record)
            _write(out_dir / "progress.json", {**attempt, "template": template, "history": history})
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "steps": steps,
                        "validation": {name: value["GROWING"]["in_range_mse"] for name, value in validation.items()},
                        "state": {name: value["state"] for name, value in validation.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        checkpoints = {}
        for name, model in models.items():
            path = out_dir / f"{name}.final.pt"
            torch.save(
                {
                    "schema": SCHEMA,
                    "arm": name,
                    "fold": fold,
                    "seed": seed,
                    "epoch": epochs,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "basis_mean": data.basis.mean,
                    "basis_std": data.basis.std,
                    "profiles": data.profiles.standardized_by_date,
                },
                path,
            )
            checkpoints[name] = {"path": str(path), "bytes": int(path.stat().st_size)}
        terminal = {
            **attempt,
            "status": "COMPLETED",
            "wall_seconds": time.time() - started,
            "n_train_trials": len(data.train_samples),
            "n_train_frames_per_epoch": len(data.train_samples) * 700,
            "n_val_trials": len(data.val_samples),
            "template": template,
            "profile": {
                "lag_ms": data.profiles.lag_ms,
                "gamma": data.profiles.gamma,
                "gamma_grid": list(data.profiles.gamma_grid),
            },
            "parameters": {name: parameter_count(model) for name, model in models.items()},
            "history": history,
            "checkpoints": checkpoints,
        }
        _write(out_dir / "terminal.json", terminal)
        return terminal
    except BaseException as exc:
        _write(
            out_dir / "failure.json",
            {
                **attempt,
                "status": "FAILED",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "steps": steps,
                "history": history,
                "wall_seconds": time.time() - started,
            },
        )
        raise

