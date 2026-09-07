"""Stage0 CLI plus formal 24-epoch train. Train requires M2_SMALL_TRAIN=1."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tfpd_exploration.src.m2_dual_track_v1 import contracts as old_contracts
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan
from tfpd_exploration.src.m2_dual_track_v1 import sampler as old_sampler
from tfpd_exploration.src.m2_dual_track_v1 import training as old_training

from . import config as cfg
from .decoder import SmallTransformerDecoder, count_decoder_parameters, init_state_sha256
from .ema import DecoderEMA
from .stage0 import run_stage0
from .training import (
    TrainState,
    apply_cell_lr,
    capture_rng,
    load_checkpoint,
    make_optimizer,
    restore_rng,
    save_checkpoint,
    unit_dropout_mask,
)


def train_is_authorized() -> bool:
    return os.environ.get(cfg.TRAIN_ENV_FLAG) == "1"


def formal_train_cli() -> str:
    return (
        f"PYTHONNOUSERSITE=1 {cfg.TRAIN_ENV_FLAG}=1 {cfg.PYTHON} "
        "tfpd_exploration/scripts/run_m2_b_small_stability_v1.py --train "
        f"--cell {cfg.CELL_S1} --seed {cfg.SEED}"
    )


def run_stage0_cli(*, root: Path | None = None) -> dict[str, Any]:
    return run_stage0(root=root)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _visible_device() -> tuple[torch.device, str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cfg.require(raw.strip() != "" and "," not in raw, "exactly one CUDA_VISIBLE_DEVICES")
    index = int(raw.strip())
    uuid = {0: old_plan.GPU0_UUID, 1: old_plan.GPU1_UUID}[index]
    return torch.device("cuda:0"), uuid


def _load_banks(surface: str, device: torch.device) -> dict[str, Any]:
    return {
        session: old_data.load_session_bank(surface, session, device=device)
        for session in old_plan.HELDIN_SESSIONS
    }


def _score_minival(model: nn.Module, banks: dict[str, Any], device: torch.device) -> dict[str, Any]:
    per_session: dict[str, float] = {}
    losses: list[float] = []
    model.eval()
    for session, bank in banks.items():
        targets: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for batch in old_data.iter_session_batches(
            bank,
            batch_size=old_plan.EFFECTIVE_BATCH,
            device=device,
            target_space=old_plan.SCORING_TARGET_SPACE,
        ):
            with torch.inference_mode():
                raw = model.forward_last(batch.X, batch.bank, batch.unit_mask)
            native = raw.detach().cpu().numpy() / old_plan.BEHAVIOR_SCALE
            preds.append(np.ascontiguousarray(native, dtype=np.float32))
            targets.append(batch.last_target.detach().cpu().numpy())
        target = np.concatenate(targets, axis=0)
        pred = np.concatenate(preds, axis=0)
        per_session[session] = old_contracts.variance_weighted_r2(target, pred)
        losses.append(float(np.mean(np.square(target - pred))))
    summary = old_contracts.summarize_sessions(per_session)
    return {
        "per_session_r2": summary["per_session_r2"],
        "equal_session_mean": summary["equal_session_mean"],
        "native_mse": float(np.mean(losses)),
    }


def _select_epoch(scores: dict[int, float]) -> int:
    finite = {int(epoch): float(score) for epoch, score in scores.items() if np.isfinite(score)}
    cfg.require(bool(finite), "no finite source-minival scores")
    best = max(finite.values())
    tied = [epoch for epoch, score in finite.items() if abs(best - score) <= 1.0e-10]
    return min(tied)


def run_train(*, cell: str, seed: int = cfg.SEED) -> dict[str, Any]:
    if not train_is_authorized():
        raise RuntimeError(
            f"REFUSED: 24-epoch {cell} training requires {cfg.TRAIN_ENV_FLAG}=1."
        )
    cfg.require(
        cell in (cfg.CELL_S0, cfg.CELL_S1, cfg.CELL_N0, cfg.CELL_N1),
        f"unknown cell {cell}",
    )
    device, gpu_uuid = _visible_device()
    dest = cfg.run_root_for_cell(cell) / cell.replace("-", "_") / f"seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "summary.json").exists():
        raise RuntimeError(f"refusing to overwrite finished dest {dest}")

    manifest = old_sampler.load_manifest(cfg.OLD_MANIFEST_24_PATH)
    cfg.require(manifest["digest"] == cfg.MANIFEST_24_DIGEST, "24-epoch manifest digest")
    train_banks = _load_banks("source_train", device)
    minival_banks = _load_banks("source_minival", device)
    updates = old_training.count_updates(train_banks)
    cfg.require(updates == cfg.UPDATES_PER_EPOCH, f"updates/epoch {updates} != {cfg.UPDATES_PER_EPOCH}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallTransformerDecoder(seed=seed).to(device)
    init_sha = init_state_sha256(model)
    optimizer = make_optimizer(model.trainable_parameters().items())
    ema = DecoderEMA(model, decay=cfg.EMA_DECAY)
    state = TrainState(cell=cell, seed=seed, manifest_digest=manifest["digest"])
    metrics_path = dest / "metrics.jsonl"
    heartbeat = dest / "heartbeat.json"
    raw_scores: dict[int, float] = {}
    ema_scores: dict[int, float] = {}
    started = time.monotonic()
    deadline = started + 2.0 * 3600.0
    global_step = 0
    _write_json(
        dest / "run_meta.json",
        {
            "cell": cell,
            "seed": seed,
            "gpu_uuid": gpu_uuid,
            "init_sha256": init_sha,
            "decoder_params": count_decoder_parameters(model),
            "manifest_digest": manifest["digest"],
            "primary_candidate": (
                cfg.PRIMARY_CANDIDATE_LR if cell in (cfg.CELL_N0, cfg.CELL_N1) else cfg.PRIMARY_CANDIDATE
            ),
            "lr_bounds": list(cfg.cell_lr_bounds(cell)),
        },
    )

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        running = 0.0
        n_batches = 0
        epoch_t0 = time.monotonic()
        for batch_id, batch in enumerate(
            old_training.epoch_batches_shuffled(train_banks, manifest, epoch, device=device)
        ):
            if time.monotonic() >= deadline:
                raise RuntimeError("2 GPU-hour cell budget hit before finishing 24 epochs")
            global_step += 1
            apply_cell_lr(optimizer, cell, global_step)
            keep = unit_dropout_mask(
                batch.unit_mask, seed=seed, epoch=epoch, batch_id=batch_id
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model.forward_last(batch.X, batch.bank, batch.unit_mask, dropout_keep=keep)
                loss = nn.functional.mse_loss(pred.float(), batch.last_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), cfg.GRAD_CLIP)
            optimizer.step()
            ema.update_after_step(model)
            running += float(loss.detach().cpu())
            n_batches += 1
            if global_step % 20 == 0:
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "step",
                        "cell": cell,
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": float(loss.detach().cpu()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "ema_updates": ema.n_updates,
                        "unix": time.time(),
                    },
                )
                _write_json(
                    heartbeat,
                    {
                        "cell": cell,
                        "epoch": epoch,
                        "global_step": global_step,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "loss": float(loss.detach().cpu()),
                        "ema_updates": ema.n_updates,
                        "unix": time.time(),
                        "gpu_uuid": gpu_uuid,
                    },
                )
        raw = _score_minival(model, minival_banks, device)
        ema_minival = ema.score_with_ema(model, lambda current: _score_minival(current, minival_banks, device))
        raw_scores[epoch] = float(raw["equal_session_mean"])
        ema_scores[epoch] = float(ema_minival["equal_session_mean"])
        state.global_step = global_step
        state.epoch = epoch
        extras = {
            "train_mse": running / max(n_batches, 1),
            "minival_raw": raw,
            "minival_ema": ema_minival,
            "seconds": time.monotonic() - epoch_t0,
            "global_step": global_step,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "ema_updates": ema.n_updates,
        }
        ckpt = save_checkpoint(model, optimizer, ema, state)
        ckpt["extras"] = extras
        torch.save(ckpt, dest / f"epoch_{epoch:03d}.pt")
        _append_jsonl(metrics_path, {"event": "epoch", "epoch": epoch, "cell": cell, **extras, "unix": time.time()})
        _write_json(heartbeat, {"event": "epoch", "cell": cell, "epoch": epoch, **extras, "unix": time.time()})

    summary = {
        "cell": cell,
        "seed": seed,
        "gpu_uuid": gpu_uuid,
        "init_sha256": init_sha,
        "epochs_completed": list(range(1, cfg.EPOCHS + 1)),
        "source_pick_raw": _select_epoch(raw_scores),
        "source_pick_ema": _select_epoch(ema_scores),
        "endpoint24_raw": raw_scores[cfg.EPOCHS],
        "endpoint24_ema": ema_scores[cfg.EPOCHS],
        "epoch_minival_raw": raw_scores,
        "epoch_minival_ema": ema_scores,
        "primary_candidate": (
            cfg.PRIMARY_CANDIDATE_LR if cell in (cfg.CELL_N0, cfg.CELL_N1) else cfg.PRIMARY_CANDIDATE
        ),
        "ext4_scored": False,
        "elapsed_s": time.monotonic() - started,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(dest / "summary.json", summary)
    return summary
