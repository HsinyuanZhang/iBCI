"""Independent source/dev-only Flat decoder trainer for the CONCAT extension."""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
from learnable_recency_v1.wrap import apply_group_lrs

from .. import protocol
from ..common import (PairedSampler, aggregate_scores, atomic_json, digest, fit_source_stats,
                      fresh_directory, load_records, record_binding, require_full_source, sha256)
from ..data import padded_windows
from ..training import (_cpu_state, _evaluate, calibration_inputs, load_encoder, predict_network,
                        save_torch, use_ema)
from .contract import FLAT_RECIPE, FLAT_SCHEMA, flat_config_metadata, flat_training_source_hashes
from .model import make_model, validate_flat_model


def _encoder_binding(path: Path | None) -> dict | None:
    if path is None:
        return None
    receipt = json.loads(Path(path).with_suffix(".json").read_text())
    return {"path": str(Path(path).resolve()), "checkpoint_sha256": sha256(path),
            "receipt_sha256": sha256(Path(path).with_suffix(".json")),
            "representation": receipt.get("representation"), "schema": receipt.get("schema"),
            "source_stats_sha256": receipt.get("source_stats_sha256"),
            "global_step": receipt.get("global_step")}


def _flat_payload_ok(payload: dict, *, allow_smoke: bool) -> None:
    if payload.get("schema") != FLAT_SCHEMA + "_checkpoint" or payload.get("stage") != "train":
        raise ValueError("expected Flat decoder-training checkpoint")
    if payload.get("protocol") != protocol.protocol_dict() or payload.get("recipe") != FLAT_RECIPE:
        raise ValueError("Flat checkpoint protocol or recipe mismatch")
    if payload.get("flat_config") != flat_config_metadata():
        raise ValueError("Flat checkpoint recency configuration mismatch")
    proof = payload.get("zero_slope_proof", {})
    if proof.get("slopes_shape") != [4, 8] or proof.get("slopes_zero_count") != 32 or proof.get("learnable_recency_parameters") != []:
        raise ValueError("Flat checkpoint lacks exact zero-slope proof")
    if not allow_smoke and payload.get("status") != "FORMAL":
        raise ValueError("SMOKE Flat checkpoints cannot enter formal evaluation")


def load_trained_model(checkpoint: Path, *, device: str = "cpu", allow_smoke: bool = False):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _flat_payload_ok(payload, allow_smoke=allow_smoke)
    # A formal Full decoder always reloads with the inherited base encoder
    # frozen, exactly as it was trained.  Evaluation has no exception here.
    model = make_model(payload["arm"], seed=int(payload["seed"]),
                       freeze_encoder=payload["arm"] == "full").to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    validate_flat_model(model, require_frozen_full=payload["arm"] == "full")
    return model.eval(), payload["source_stats"]


def run_training(cache: Path, dest: Path, *, representation: str, arm: str = "full",
                 encoder_path: Path | None = None, seed: int = 42, device: str = "cpu",
                 smoke_updates: int | None = None, source_ids: tuple[str, ...] | None = None,
                 eval_records: list[Any] | None = None) -> dict:
    """Train only a Flat decoder; base encoder pretraining is intentionally absent."""
    if arm not in {"full", "activity", "raw_set"} or seed not in {42, 43, 44}:
        raise ValueError("Flat training requires a registered arm and seed 42, 43, or 44")
    smoke = smoke_updates is not None
    if smoke and not 1 <= int(smoke_updates) <= 8:
        raise ValueError("Flat smoke is limited to 1..8 updates")
    if not smoke and (source_ids is not None or eval_records is not None):
        raise ValueError("formal Flat training uses complete fixed source/development rosters")
    if (arm == "full") != (encoder_path is not None):
        raise ValueError("Flat Full requires exactly one base encoder; ACT/Raw-set forbid it")
    destination = fresh_directory(dest)
    records = load_records(cache, representation, "train", session_ids=source_ids)
    if not smoke:
        require_full_source(records)
    stats = fit_source_stats(records, smoke=smoke)
    atomic_json(destination / "source_stats.json", stats)
    target_records = (eval_records if smoke and eval_records is not None else
                      [replace(records[-1], query_indices=records[-1].query_indices[:8])] if smoke else
                      load_records(cache, representation, "dev"))
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable in selected Python environment")
    encoder = None
    if arm == "full":
        encoder = load_encoder(Path(encoder_path), representation=representation, stats=stats, allow_smoke=smoke)
    model = make_model(arm, seed=seed, encoder=encoder, freeze_encoder=arm == "full").to(torch_device)
    zero_proof = validate_flat_model(model, require_frozen_full=arm == "full")
    groups = model.optimizer_param_groups(lr=FLAT_RECIPE["lr_peak"], weight_decay=FLAT_RECIPE["weight_decay"])
    optimizer = torch.optim.AdamW(groups, lr=FLAT_RECIPE["lr_peak"], betas=tuple(FLAT_RECIPE["betas"]), eps=FLAT_RECIPE["eps"])
    ema = DecoderEMA(model, decay=FLAT_RECIPE["ema_decay"])
    updates = int(smoke_updates) if smoke else FLAT_RECIPE["updates_per_segment"]
    segments, batch = (1, 2) if smoke else (FLAT_RECIPE["segments"], FLAT_RECIPE["batch"])
    sampler = PairedSampler(records, seed, batch=batch, updates_per_segment=updates)
    binding = _encoder_binding(encoder_path)
    metadata = {"schema": FLAT_SCHEMA + "_training", "status": "SMOKE" if smoke else "FORMAL",
                "stage": "train", "arm": arm, "representation": representation, "seed": seed,
                "protocol": protocol.protocol_dict(), "recipe": FLAT_RECIPE, "flat_recipe": FLAT_RECIPE,
                "flat_config": flat_config_metadata(), "zero_slope_proof": zero_proof,
                "actual_budget": {"segments": segments, "updates_per_segment": updates, "batch": batch},
                "source_sessions": [r.session_id for r in records], "source_binding": record_binding(records),
                "source_stats_sha256": stats["sha256"], "code_hashes": flat_training_source_hashes(),
                "encoder_checkpoint_sha256": sha256(encoder_path) if encoder_path else None,
                "base_encoder_binding": binding, "final_sessions_opened": 0, "device": str(torch_device),
                "parameters_total": sum(p.numel() for p in model.parameters()),
                "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    atomic_json(destination / "protocol.json", metadata)
    calibration = {r.session_id: calibration_inputs(r, arm, stats, torch_device) for r in records}
    y_mean, y_std = (torch.tensor(stats["velocity_mean"], device=torch_device),
                     torch.tensor(stats["velocity_std"], device=torch_device))
    curve, step, best_value, best_path = [], 0, -float("inf"), None
    started = time.monotonic()
    for segment in range(segments):
        model.train(); losses = []
        for batch_index, (record, endpoints) in enumerate(sampler.segment()):
            x, timevalid, unitmask = padded_windows(record, endpoints)
            tensor_x = torch.from_numpy(x).to(torch_device)
            base_mask = torch.from_numpy(unitmask).to(torch_device).expand(len(endpoints), -1)
            generator = torch.Generator(device="cpu").manual_seed(unit_dropout_seed(seed, segment + 1, batch_index))
            keep = whole_unit_dropout(base_mask, p=FLAT_RECIPE["whole_unit_dropout"], generator=generator)
            target = (torch.from_numpy(record.velocity[endpoints]).to(torch_device) - y_mean) / y_std
            step += 1
            lr = warmup_cosine_lr(step, total_steps=segments * updates,
                                  warmup_steps=min(FLAT_RECIPE["warmup_updates"], max(1, segments * updates // 2)) if smoke else FLAT_RECIPE["warmup_updates"],
                                  peak=FLAT_RECIPE["lr_peak"], min_factor=FLAT_RECIPE["lr_min_factor"])
            apply_group_lrs(optimizer, lr); optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch_device.type == "cuda" else nullcontext()
            with amp:
                prediction = model(tensor_x, **calibration[record.session_id], unit_mask=base_mask,
                                   dropout_keep=keep, input_valid_mask=torch.from_numpy(timevalid).to(torch_device))
                loss = nn.functional.mse_loss(prediction.float(), target.float())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite Flat neural loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), FLAT_RECIPE["grad_clip"], error_if_nonfinite=True)
            optimizer.step(); ema.update_after_step(model); losses.append(float(loss.detach().cpu()))
        checkpoint = destination / f"segment_{segment + 1:02d}.pt"
        with use_ema(model, ema):
            proof = validate_flat_model(model, require_frozen_full=arm == "full")
            validation = _evaluate(model, target_records, stats, batch=batch)
            payload = {"schema": FLAT_SCHEMA + "_checkpoint", "status": metadata["status"], "stage": "train",
                       "arm": arm, "representation": representation, "seed": seed, "global_step": step,
                       "model_state": _cpu_state(model), "source_stats": stats, "protocol": protocol.protocol_dict(),
                       "recipe": FLAT_RECIPE, "flat_recipe": FLAT_RECIPE, "flat_config": flat_config_metadata(),
                       "zero_slope_proof": proof, "encoder_checkpoint_sha256": metadata["encoder_checkpoint_sha256"],
                       "base_encoder_binding": binding}
            save_torch(checkpoint, payload)
        row = {"segment": segment + 1, "global_step": step, "mean_loss": float(np.mean(losses)),
               "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint), "development": validation}
        curve.append(row)
        if validation["mean_r2"] > best_value:
            best_value, best_path = validation["mean_r2"], checkpoint
        atomic_json(destination / "progress.json", {"global_step": step, "segments": curve,
                                                     "elapsed_seconds": time.monotonic() - started})
    selection = {"schema": FLAT_SCHEMA + "_selection", "status": metadata["status"],
                 "rule": "earliest_max_equal_session_dev_r2", "checkpoint": str(best_path),
                 "checkpoint_sha256": sha256(best_path), "mean_dev_r2": best_value,
                 "dev_sessions": [r.session_id for r in target_records], "flat_config": flat_config_metadata(),
                 "zero_slope_proof": zero_proof, "base_encoder_binding": binding, "final_sessions_opened": 0}
    atomic_json(destination / "selection.json", selection)
    receipt = {**metadata, "global_step": step, "completed": True, "sampler": sampler.receipt(),
               "segments": curve, "elapsed_seconds": time.monotonic() - started}
    atomic_json(destination / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True); parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--representation", choices=("sua", "pmua"), required=True)
    parser.add_argument("--arm", choices=("full", "activity", "raw_set"), required=True)
    parser.add_argument("--encoder", type=Path); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--smoke-updates", type=int)
    args = parser.parse_args(); torch.set_num_threads(2)
    result = run_training(args.cache, args.dest, representation=args.representation, arm=args.arm,
                          encoder_path=args.encoder, seed=args.seed, device=args.device, smoke_updates=args.smoke_updates)
    print(json.dumps({k: result[k] for k in ("schema", "status", "completed", "final_sessions_opened")}, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["run_training", "load_trained_model"]
