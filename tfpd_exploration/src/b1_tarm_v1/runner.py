"""Paired four-arm B1 TARM pilot runner."""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.b1_sfcj_v1.constants import N_CHANNELS, VALID_END, VALID_START
from tfpd_exploration.src.b1_sfcj_v1.metric import official_metric_from_trials
from tfpd_exploration.src.b1_sfcj_v1.util import sha256_array

from .dataset import FoldData, Sample, build_fold_data, collate
from .model import build_four_arms, parameter_count
from .plan import (
    ARMS,
    BATCH_SIZE,
    CONTROL_LAW,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    PILOT_EPOCHS,
    PILOT_FOLD,
    PILOT_SEED,
    PRIMARY_LAW,
    RESIDUAL_LOSS_WEIGHT,
    RESULT_ROOT,
    SCHEMA,
    UNIT_DROPOUT_P,
    WEIGHT_DECAY,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _tensor(batch: dict, key: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(batch[key]).to(device=device, non_blocking=False)


def _torch_batch(batch: dict, device: torch.device) -> dict:
    return {
        "current": _tensor(batch, "current", device),
        "history": _tensor(batch, "history", device),
        "history_mask": _tensor(batch, "history_mask", device),
        "carrier": _tensor(batch, "carrier", device),
        "template_stdlog": _tensor(batch, "template_stdlog", device),
        "target_stdlog": _tensor(batch, "target_stdlog", device),
        "target_raw": _tensor(batch, "target_raw", device),
    }


def _loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    pred = output["raw"][:, :, VALID_START:VALID_END]
    target = batch["target_raw"][:, :, VALID_START:VALID_END]
    pmin = pred.amin(dim=(1, 2), keepdim=True)
    pmax = pred.amax(dim=(1, 2), keepdim=True)
    tmin = target.amin(dim=(1, 2), keepdim=True)
    tmax = target.amax(dim=(1, 2), keepdim=True)
    if torch.any(pmax <= pmin) or torch.any(tmax <= tmin):
        raise RuntimeError("zero-range B1 prediction/target")
    official = torch.mean(((pred - pmin) / (pmax - pmin) - (target - tmin) / (tmax - tmin)) ** 2)
    target_delta = batch["target_stdlog"] - batch["template_stdlog"]
    residual = F.smooth_l1_loss(
        output["delta_stdlog"][:, :, VALID_START:VALID_END],
        target_delta[:, :, VALID_START:VALID_END],
    )
    total = official + RESIDUAL_LOSS_WEIGHT * residual
    return total, {
        "official_surrogate": float(official.detach().cpu()),
        "residual_smooth_l1": float(residual.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def _unit_mask(batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    keep = rng.random((batch_size, N_CHANNELS)) >= UNIT_DROPOUT_P
    for row in range(batch_size):
        if not keep[row].any():
            keep[row, int(rng.integers(0, N_CHANNELS))] = True
    return torch.as_tensor(keep.astype(np.float32), device=device)


def _batches(samples: tuple[Sample, ...], order: np.ndarray, batch_size: int):
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield [samples[int(i)] for i in idx]


@torch.no_grad()
def evaluate(
    *,
    model,
    samples: tuple[Sample, ...],
    law: str,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> dict:
    model.eval()
    preds = []
    targets = []
    flags = []
    keys = []
    ones_cache = {}
    for start in range(0, len(samples), batch_size):
        chunk = list(samples[start : start + batch_size])
        raw = collate(chunk, law=law)
        tb = _torch_batch(raw, device)
        n = len(chunk)
        if n not in ones_cache:
            ones_cache[n] = torch.ones((n, N_CHANNELS), dtype=torch.float32, device=device)
        output = model(**{k: tb[k] for k in ("current", "history", "history_mask", "carrier", "template_stdlog")}, unit_mask=ones_cache[n])
        pred = output["raw"].detach().cpu().numpy()
        preds.extend([p.astype(np.float64) for p in pred])
        targets.extend([s.target_raw.astype(np.float64) for s in chunk])
        flags.extend([bool(s.in_range) for s in chunk])
        keys.extend([s.key for s in chunk])
    full = official_metric_from_trials(preds, targets)
    chosen = [i for i, flag in enumerate(flags) if flag]
    in_range = official_metric_from_trials([preds[i] for i in chosen], [targets[i] for i in chosen])
    return {
        "law": law,
        "full_mse": float(full["MSE Mean"]),
        "full_std": float(full["MSE Std."]),
        "in_range_mse": float(in_range["MSE Mean"]),
        "in_range_std": float(in_range["MSE Std."]),
        "n_full": int(full["n_trials"]),
        "n_in_range": int(in_range["n_trials"]),
        "per_trial_mse": [float(v) for v in full["per_trial_mse"]],
        "prediction_sha256": sha256_array(np.stack(preds, axis=0)),
        "keys": keys,
    }


def _template_score(fold_data: FoldData) -> dict:
    pred = [fold_data.template_raw_by_date[fold_data.val_date].astype(np.float64) for _ in fold_data.val_samples]
    tgt = [sample.target_raw.astype(np.float64) for sample in fold_data.val_samples]
    flags = [sample.in_range for sample in fold_data.val_samples]
    full = official_metric_from_trials(pred, tgt)
    idx = [i for i, flag in enumerate(flags) if flag]
    inner = official_metric_from_trials([pred[i] for i in idx], [tgt[i] for i in idx])
    return {
        "full_mse": float(full["MSE Mean"]),
        "in_range_mse": float(inner["MSE Mean"]),
        "per_trial_mse": [float(v) for v in full["per_trial_mse"]],
    }


def _profile_receipt(data: FoldData) -> dict:
    profile = data.profiles
    return {
        "lag_ms": profile.lag_ms,
        "gamma": profile.gamma,
        "gamma_grid": list(profile.gamma_grid),
        "coefficient_mean_sha256": sha256_array(profile.coefficient_mean),
        "coefficient_std_sha256": sha256_array(profile.coefficient_std),
        "source_prior_sha256": sha256_array(profile.source_prior),
        "standardized_profile_sha256": {
            date: sha256_array(value) for date, value in profile.standardized_by_date.items()
        },
    }


def run_pilot(
    *,
    fold: int = PILOT_FOLD,
    seed: int = PILOT_SEED,
    epochs: int = PILOT_EPOCHS,
    out_dir: Path | None = None,
    device_name: str = "cuda:0",
) -> dict:
    if device_name.startswith("cuda"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("B1 TARM pilot requires CUDA_VISIBLE_DEVICES=0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one visible CUDA device")
    out_dir = Path(out_dir) if out_dir is not None else RESULT_ROOT / f"pilot_fold{fold}_seed{seed}"
    if out_dir.exists():
        raise FileExistsError(f"fresh pilot root required: {out_dir}")
    out_dir.mkdir(parents=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(device_name)
    start_wall = time.time()

    attempt = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "fold": int(fold),
        "seed": int(seed),
        "epochs": int(epochs),
        "arms": [row[0] for row in ARMS],
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_unix": start_wall,
    }
    _write_json(out_dir / "attempt.json", attempt)

    data = build_fold_data(fold)
    arms = build_four_arms(log_mean=data.basis.mean, log_std=data.basis.std, seed=seed)
    for model in arms.values():
        model.to(device)
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        for name, model in arms.items()
    }
    rng = np.random.default_rng(seed)
    template_score = _template_score(data)
    history = []
    steps = 0

    try:
        for epoch in range(1, int(epochs) + 1):
            for model in arms.values():
                model.train()
            order = rng.permutation(len(data.train_samples))
            train_acc = {name: [] for name in arms}
            epoch_start = time.time()
            for samples in _batches(data.train_samples, order, BATCH_SIZE):
                raw = collate(samples, law=PRIMARY_LAW)
                tb = _torch_batch(raw, device)
                mask = _unit_mask(len(samples), rng, device)
                for opt in optimizers.values():
                    opt.zero_grad(set_to_none=True)
                losses = {}
                loss_records = {}
                for name, model in arms.items():
                    output = model(
                        **{k: tb[k] for k in ("current", "history", "history_mask", "carrier", "template_stdlog")},
                        unit_mask=mask,
                    )
                    losses[name], loss_records[name] = _loss(output, tb)
                torch.stack(list(losses.values())).sum().backward()
                for name, model in arms.items():
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    if not torch.isfinite(grad_norm):
                        raise RuntimeError(f"non-finite gradient in {name}")
                    optimizers[name].step()
                    train_acc[name].append(loss_records[name])
                steps += 1

            validation = {}
            for name, model in arms.items():
                validation[name] = {
                    PRIMARY_LAW: evaluate(model=model, samples=data.val_samples, law=PRIMARY_LAW, device=device),
                    CONTROL_LAW: evaluate(model=model, samples=data.val_samples, law=CONTROL_LAW, device=device),
                }
            epoch_record = {
                "epoch": epoch,
                "steps": steps,
                "epoch_wall_seconds": time.time() - epoch_start,
                "train": {
                    name: {
                        key: float(np.mean([row[key] for row in rows]))
                        for key in ("total", "official_surrogate", "residual_smooth_l1")
                    }
                    for name, rows in train_acc.items()
                },
                "validation": validation,
            }
            history.append(epoch_record)
            _write_json(
                out_dir / "progress.json",
                {**attempt, "status": "RUNNING", "template": template_score, "profile": _profile_receipt(data), "history": history},
            )
            print(json.dumps({"epoch": epoch, "steps": steps, "validation": {n: v[PRIMARY_LAW]["in_range_mse"] for n, v in validation.items()}}, sort_keys=True), flush=True)

        checkpoint_map = {}
        for name, model in arms.items():
            path = out_dir / f"{name}.final.pt"
            torch.save(
                {
                    "schema": SCHEMA,
                    "arm": name,
                    "fold": fold,
                    "seed": seed,
                    "epoch": epochs,
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "basis_mean": data.basis.mean,
                    "basis_std": data.basis.std,
                    "profile": data.profiles.standardized_by_date,
                },
                path,
            )
            checkpoint_map[name] = {"path": str(path), "bytes": int(path.stat().st_size)}
        terminal = {
            **attempt,
            "status": "COMPLETED",
            "wall_seconds": time.time() - start_wall,
            "n_train_samples": len(data.train_samples),
            "n_val_samples": len(data.val_samples),
            "parameter_count_per_arm": {name: parameter_count(model) for name, model in arms.items()},
            "template": template_score,
            "profile": _profile_receipt(data),
            "history": history,
            "checkpoints": checkpoint_map,
        }
        _write_json(out_dir / "terminal.json", terminal)
        return terminal
    except BaseException as exc:
        failure = {
            **attempt,
            "status": "FAILED",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "steps_completed": steps,
            "history": history,
            "wall_seconds": time.time() - start_wall,
        }
        _write_json(out_dir / "failure.json", failure)
        raise


def pilot_summary(terminal: dict) -> dict:
    template = float(terminal["template"]["in_range_mse"])
    rows = []
    for epoch in terminal["history"]:
        for arm, laws in epoch["validation"].items():
            mse = float(laws[PRIMARY_LAW]["in_range_mse"])
            rows.append(
                {
                    "epoch": int(epoch["epoch"]),
                    "arm": arm,
                    "mse": mse,
                    "gain_vs_tpl": template - mse,
                    "relative_gain_vs_tpl": (template - mse) / template,
                }
            )
    return {"template_in_range_mse": template, "rows": rows}
