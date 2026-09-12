"""Source-only encoder pretraining and M2-recipe DANDI learned-RIFT training.

Formal commands prepare artifacts; nothing starts merely by importing this
module.  Final data are available only through the separate sealed scorer.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager, nullcontext
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

from . import protocol
from .carrier import apply_carrier_normalizer, estimate_move_t4
from .common import (RECIPE, SCHEMA, PairedSampler, aggregate_scores, atomic_json, digest,
                     fit_source_stats, fresh_directory, load_records, record_binding,
                     require_full_source, score_predictions, sha256, source_hashes, verify_stats)
from .data import SessionData, padded_windows
from .model import B3SIdentityEncoder, DandiRiftDecoder


def save_torch(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_state(model: nn.Module) -> dict:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def calibration_inputs(record: SessionData, arm: str, stats: dict, device: torch.device,
                       *, n_pad: int = 100) -> dict[str, torch.Tensor]:
    if arm == "raw_set":
        return {}
    count = record.neural.shape[1]
    if count > n_pad:
        raise ValueError("real units exceed padding; truncation is forbidden")
    activity = np.zeros((protocol.ACTIVITY_TRIALS, 100, n_pad), np.float32)
    activity[:, :, :count] = record.activity
    result = {"activity": torch.from_numpy(activity).to(device)}
    if arm == "full":
        raw = estimate_move_t4(record.carrier_counts, record.carrier_angles)
        normalized = apply_carrier_normalizer(raw, stats["carrier"])
        carrier = np.zeros((n_pad, 4), np.float32)
        carrier[:count] = normalized
        result["carrier"] = torch.from_numpy(carrier).to(device)
    elif arm != "activity":
        raise ValueError("unknown network arm")
    return result


def load_encoder(path: Path, *, representation: str, stats: dict,
                 allow_smoke: bool = False) -> B3SIdentityEncoder:
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt.get("checkpoint_sha256") != sha256(path):
        raise ValueError("encoder checkpoint hash differs from its receipt")
    if receipt.get("schema") != SCHEMA + "_encoder" or receipt.get("representation") != representation:
        raise ValueError("encoder schema or representation mismatch")
    if receipt.get("source_stats_sha256") != stats["sha256"]:
        raise ValueError("encoder was pretrained with different source data/statistics")
    if not allow_smoke:
        if receipt.get("status") != "FORMAL" or receipt.get("source_sessions") != list(protocol.TRAIN_SESSIONS):
            raise ValueError("full requires a FORMAL representation-matched 18-source encoder")
        if receipt.get("global_step") != RECIPE["segments"] * RECIPE["updates_per_segment"]:
            raise ValueError("encoder pretraining budget is incomplete")
    if (receipt.get("protocol") != protocol.protocol_dict() or receipt.get("recipe") != RECIPE
            or receipt.get("final_sessions_opened") != 0):
        raise ValueError("encoder data provenance mismatch")
    if (receipt.get("encoder_line") != RECIPE["encoder_line"]
            or receipt.get("encoder_carrier_fusion") != RECIPE["encoder_carrier_fusion"]
            or receipt.get("encoder_film") is not RECIPE["encoder_film"]):
        raise ValueError("encoder concat-line contract mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("receipt_binding") != digest({k: v for k, v in receipt.items() if k != "checkpoint_sha256"}):
        raise ValueError("encoder payload/receipt binding mismatch")
    encoder = B3SIdentityEncoder(side_dim=4, seed=int(receipt["encoder_seed"]))
    encoder.load_state_dict(state["encoder_state"], strict=True)
    return encoder


def export_encoder(model: DandiRiftDecoder, dest: Path, metadata: dict) -> Path:
    if model.arm != "full" or not isinstance(model.encoder, B3SIdentityEncoder) or model.encoder.side_dim != 4:
        raise ValueError("only the full carrier-side encoder can be exported")
    path = dest / "encoder.pt"
    receipt = {"schema": SCHEMA + "_encoder", "protocol": protocol.protocol_dict(),
               "representation": metadata["representation"], "status": metadata["status"],
               "source_sessions": metadata["source_sessions"],
               "source_stats_sha256": metadata["source_stats_sha256"],
               "global_step": metadata["global_step"], "encoder_seed": metadata["seed"],
               "selection": "fixed_final_ema_source_only", "recipe": RECIPE,
               "encoder_line": RECIPE["encoder_line"],
               "encoder_carrier_fusion": RECIPE["encoder_carrier_fusion"],
               "encoder_film": RECIPE["encoder_film"],
               "code_hashes": metadata["code_hashes"], "final_sessions_opened": 0}
    save_torch(path, {"encoder_state": _cpu_state(model.encoder), "receipt_binding": digest(receipt)})
    receipt["checkpoint_sha256"] = sha256(path)
    atomic_json(path.with_suffix(".json"), receipt)
    return path


@contextmanager
def use_ema(model: nn.Module, ema: DecoderEMA):
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                if parameter.requires_grad}
    ema.apply_to(model)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in original:
                    parameter.copy_(original[name])


def predict_network(model: DandiRiftDecoder, record: SessionData, stats: dict, *, batch: int = 32,
                    adapter: Any = None) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    local_record = record
    if adapter is not None:
        if model.arm != "raw_set" or record.representation != "pmua":
            raise ValueError("frozen input adapters are restricted to raw-set PMUA")
        local_record = replace(record, neural=adapter.transform(record.neural))
    calibration = calibration_inputs(record, model.arm, stats, device, n_pad=model.units)
    e0 = None
    if model.arm != "raw_set":
        e0 = model.calibrate(calibration["activity"], calibration.get("carrier"))
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(record.query_indices), batch):
            endpoints = record.query_indices[start:start + batch]
            x, timevalid, unitmask = padded_windows(local_record, endpoints, model.units)
            x = torch.from_numpy(x).to(device)
            kwargs = {"unit_mask": torch.from_numpy(unitmask).to(device),
                      "input_valid_mask": torch.from_numpy(timevalid).to(device)}
            if model.arm == "raw_set":
                prediction = model(x, **kwargs)
            else:
                carrier = calibration.get("carrier", torch.zeros(model.units, 4, device=device))
                prediction = model.forward_identity(x, e0, carrier, **kwargs)
            predictions.append(prediction.float().cpu().numpy())
    normalized = np.concatenate(predictions)
    return np.asarray(normalized * np.asarray(stats["velocity_std"]) + np.asarray(stats["velocity_mean"]), np.float32)


def _evaluate(model: DandiRiftDecoder, records: list[SessionData], stats: dict, batch: int = 32) -> dict:
    return aggregate_scores([score_predictions(r, predict_network(model, r, stats, batch=batch)) for r in records])


def run_training(cache: Path, dest: Path, *, representation: str, arm: str = "full",
                 stage: str = "train", encoder_path: Path | None = None, seed: int = 42,
                 device: str = "cpu", smoke_updates: int | None = None,
                 source_ids: tuple[str, ...] | None = None,
                 eval_records: list[SessionData] | None = None) -> dict:
    """Train one explicit stage; smoke produces ineligible, separately named artifacts.

    Encoder pretraining jointly optimizes a fresh carrier-side encoder and a
    temporary RIFT decoder on source labels.  It exports the fixed-final EMA
    encoder.  Main full training constructs a freshly seeded RIFT decoder and
    freezes that encoder; ACT/raw_set are each initialized from scratch.
    """
    if stage not in {"pretrain", "train"} or (stage == "pretrain" and arm != "full"):
        raise ValueError("stage is train or full-encoder pretrain")
    smoke = smoke_updates is not None
    if smoke and not 1 <= int(smoke_updates) <= 8:
        raise ValueError("smoke is limited to 1..8 optimizer updates")
    if not smoke and (source_ids is not None or eval_records is not None):
        raise ValueError("formal training uses the complete fixed source/development roster")
    if seed not in {42, 43, 44}:
        raise ValueError("the preregistered neural seed roster is 42,43,44")
    if stage == "pretrain" and seed != 42:
        raise ValueError("paired decoder seeds share the representation's source encoder seed42")
    if stage == "pretrain" and encoder_path is not None:
        raise ValueError("2015 encoder pretraining must start from scratch")
    dest = fresh_directory(dest)
    records = load_records(cache, representation, "train", session_ids=source_ids)
    if not smoke:
        require_full_source(records)
    stats = fit_source_stats(records, smoke=smoke)
    atomic_json(dest / "source_stats.json", stats)
    target_records = []
    if stage == "train":
        target_records = eval_records if smoke and eval_records is not None else (
            [replace(records[-1], query_indices=records[-1].query_indices[:8])] if smoke
            else load_records(cache, representation, "dev"))
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable in the selected Python environment")
    encoder = None
    if stage == "train" and arm == "full":
        if encoder_path is None:
            raise ValueError("full decoder training requires its matching 2015-only encoder checkpoint")
        encoder = load_encoder(encoder_path, representation=representation, stats=stats, allow_smoke=smoke)
    model = DandiRiftDecoder(arm, seed=seed, encoder=encoder, freeze_encoder=stage != "pretrain").to(torch_device)
    groups = model.optimizer_param_groups(lr=RECIPE["lr_peak"], weight_decay=RECIPE["weight_decay"])
    optimizer = torch.optim.AdamW(groups, lr=RECIPE["lr_peak"], betas=tuple(RECIPE["betas"]), eps=RECIPE["eps"])
    ema = DecoderEMA(model, decay=RECIPE["ema_decay"])
    updates = int(smoke_updates) if smoke else RECIPE["updates_per_segment"]
    segments, batch = (1, 2) if smoke else (RECIPE["segments"], RECIPE["batch"])
    total_steps = segments * updates
    sampler = PairedSampler(records, seed, batch=batch, updates_per_segment=updates)
    metadata = {"schema": SCHEMA + "_training", "status": "SMOKE" if smoke else "FORMAL",
                "stage": stage, "arm": arm, "representation": representation, "seed": seed,
                "protocol": protocol.protocol_dict(), "recipe": RECIPE,
                "actual_budget": {"segments": segments, "updates_per_segment": updates, "batch": batch},
                "source_sessions": [r.session_id for r in records], "source_binding": record_binding(records),
                "source_stats_sha256": stats["sha256"], "code_hashes": source_hashes(),
                "encoder_checkpoint_sha256": sha256(encoder_path) if encoder_path is not None else None,
                "encoder_line": RECIPE["encoder_line"],
                "encoder_carrier_fusion": RECIPE["encoder_carrier_fusion"],
                "encoder_film": RECIPE["encoder_film"],
                "final_sessions_opened": 0, "device": str(torch_device),
                "parameters_total": sum(p.numel() for p in model.parameters()),
                "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    atomic_json(dest / "protocol.json", metadata)
    calibration = {r.session_id: calibration_inputs(r, arm, stats, torch_device) for r in records}
    y_mean = torch.tensor(stats["velocity_mean"], device=torch_device, dtype=torch.float32)
    y_std = torch.tensor(stats["velocity_std"], device=torch_device, dtype=torch.float32)
    curve, step, best_value, best_path = [], 0, -float("inf"), None
    checkpoint_roundtrip: dict[str, Any] | None = None
    started = time.monotonic()
    for segment in range(segments):
        model.train()
        losses = []
        for batch_index, (record, endpoints) in enumerate(sampler.segment()):
            x, timevalid, unitmask = padded_windows(record, endpoints)
            tensor_x = torch.from_numpy(x).to(torch_device)
            base_mask = torch.from_numpy(unitmask).to(torch_device).expand(len(endpoints), -1)
            keep_generator = torch.Generator(device="cpu").manual_seed(unit_dropout_seed(seed, segment + 1, batch_index))
            keep = whole_unit_dropout(base_mask, p=RECIPE["whole_unit_dropout"], generator=keep_generator)
            target = (torch.from_numpy(record.velocity[endpoints]).to(torch_device) - y_mean) / y_std
            # Current M2 increments before its schedule call: first update has
            # nonzero warmup LR and the final update reaches the cosine floor.
            step += 1
            lr = warmup_cosine_lr(step, total_steps=total_steps,
                                 warmup_steps=min(RECIPE["warmup_updates"], max(1, total_steps // 2)) if smoke else RECIPE["warmup_updates"],
                                 peak=RECIPE["lr_peak"], min_factor=RECIPE["lr_min_factor"])
            apply_group_lrs(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch_device.type == "cuda" else nullcontext()
            with amp:
                prediction = model(tensor_x, **calibration[record.session_id], unit_mask=base_mask,
                                   dropout_keep=keep, input_valid_mask=torch.from_numpy(timevalid).to(torch_device))
                loss = nn.functional.mse_loss(prediction.float(), target.float())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite neural loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), RECIPE["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            ema.update_after_step(model)
            losses.append(float(loss.detach().cpu()))
        checkpoint = dest / f"segment_{segment + 1:02d}.pt"
        with use_ema(model, ema):
            validation = _evaluate(model, target_records, stats, batch=batch) if target_records else None
            payload = {"schema": SCHEMA + "_checkpoint", "status": metadata["status"],
                       "arm": arm, "representation": representation, "seed": seed,
                       "stage": stage, "global_step": step, "model_state": _cpu_state(model),
                       "source_stats": stats, "protocol": protocol.protocol_dict(),
                       "recipe": RECIPE, "encoder_checkpoint_sha256": metadata["encoder_checkpoint_sha256"],
                       "encoder_line": RECIPE["encoder_line"],
                       "encoder_carrier_fusion": RECIPE["encoder_carrier_fusion"],
                       "encoder_film": RECIPE["encoder_film"]}
            save_torch(checkpoint, payload)
            if smoke and stage == "train":
                # A double reload only establishes deterministic deserialization.
                # This stronger check binds the in-memory EMA view that was
                # actually serialized to the checkpoint consumers will use.
                probe = replace(records[0], query_indices=records[0].query_indices[:8])
                in_memory = predict_network(model, probe, stats, batch=batch)
                reloaded, reloaded_stats = load_trained_model(checkpoint, device="cpu", allow_smoke=True)
                restored = predict_network(reloaded, probe, reloaded_stats, batch=batch)
                delta = float(np.max(np.abs(in_memory.astype(np.float64) - restored.astype(np.float64))))
                if not np.array_equal(in_memory, restored):
                    raise RuntimeError(f"SMOKE EMA checkpoint roundtrip drift: max_abs_delta={delta}")
                checkpoint_roundtrip = {"in_memory_ema_vs_reload": True, "max_abs_delta": delta,
                                        "probe_session": probe.session_id, "query_count": int(len(probe.query_indices))}
            if stage == "pretrain" and segment == segments - 1:
                export_encoder(model, dest, {**metadata, "global_step": step})
        row = {"segment": segment + 1, "global_step": step, "mean_loss": float(np.mean(losses)),
               "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint), "development": validation}
        curve.append(row)
        if validation is not None and validation["mean_r2"] > best_value:
            best_value, best_path = validation["mean_r2"], checkpoint
        atomic_json(dest / "progress.json", {"global_step": step, "segments": curve,
                                             "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"stage": stage, "arm": arm, "representation": representation,
                          "segment": segment + 1, "global_step": step,
                          "dev_r2": validation["mean_r2"] if validation else None}), flush=True)
    if best_path is not None:
        selection = {"schema": SCHEMA + "_selection", "status": metadata["status"],
                     "rule": "earliest_max_equal_session_dev_r2", "checkpoint": str(best_path),
                     "checkpoint_sha256": sha256(best_path), "mean_dev_r2": best_value,
                     "dev_sessions": [r.session_id for r in target_records],
                     "final_sessions_opened": 0}
        atomic_json(dest / "selection.json", selection)
    receipt = {**metadata, "global_step": step, "completed": True, "sampler": sampler.receipt(),
               "segments": curve, "elapsed_seconds": time.monotonic() - started}
    if smoke and stage == "train":
        if checkpoint_roundtrip is None:
            raise RuntimeError("SMOKE training did not produce an EMA checkpoint roundtrip")
        receipt["checkpoint_roundtrip"] = checkpoint_roundtrip
    atomic_json(dest / "receipt.json", receipt)
    return receipt


def load_trained_model(checkpoint: Path, *, device: str = "cpu", allow_smoke: bool = False) -> tuple[DandiRiftDecoder, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA + "_checkpoint" or payload.get("stage") != "train":
        raise ValueError("expected a decoder-training checkpoint")
    if payload.get("protocol") != protocol.protocol_dict() or payload.get("recipe") != RECIPE:
        raise ValueError("checkpoint protocol/recipe mismatch")
    if (payload.get("encoder_line") != RECIPE["encoder_line"]
            or payload.get("encoder_carrier_fusion") != RECIPE["encoder_carrier_fusion"]
            or payload.get("encoder_film") is not RECIPE["encoder_film"]):
        raise ValueError("checkpoint concat-line contract mismatch")
    if not allow_smoke and payload.get("status") != "FORMAL":
        raise ValueError("SMOKE checkpoints cannot enter formal evaluation")
    model = DandiRiftDecoder(payload["arm"], seed=int(payload["seed"])).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval(), payload["source_stats"]


def score_development(cache: Path, run: Path, dest: Path, *, static_controls: bool = False,
                      device: str = "cpu") -> dict:
    """Replay the selected checkpoint; optionally compare frozen PMUA input maps."""
    selection_path = Path(run) / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "FORMAL":
        raise ValueError("development replay requires a completed FORMAL selection")
    checkpoint = Path(selection["checkpoint"])
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed")
    model, stats = load_trained_model(checkpoint, device=device)
    source = load_records(cache, stats["representation"], "train")
    verify_stats(stats, source)
    development = load_records(cache, stats["representation"], "dev")
    dest = fresh_directory(dest)
    variants: dict[str, Any] = {"identity": _evaluate(model, development, stats)}
    if static_controls:
        if model.arm != "raw_set" or stats["representation"] != "pmua":
            raise ValueError("static calibration controls require the raw-set PMUA run")
        from .baselines import fitstatic_adapter
        for kind in ("diag_z", "coral"):
            candidates = [.0, .1, .5, 1.] if kind == "coral" else [.1]
            readings = []
            for shrinkage in candidates:
                rows, diagnostics = [], {}
                for record in development:
                    adapter = fitstatic_adapter(source, record, kind=kind, shrinkage=shrinkage,
                                                reference_session=protocol.TRAIN_SESSIONS[-1])
                    prediction = predict_network(model, record, stats, adapter=adapter)
                    rows.append(score_predictions(record, prediction))
                    diagnostics[record.session_id] = adapter.diagnostics
                readings.append({"shrinkage": shrinkage, "metrics": aggregate_scores(rows),
                                 "calibration": diagnostics})
            selected = max(range(len(readings)), key=lambda i: readings[i]["metrics"]["mean_r2"])
            variants[kind] = {"candidates": readings, "selected_index": selected,
                              "checkpoint_sha256": selection["checkpoint_sha256"]}
    receipt = {"schema": SCHEMA + "_development_replay", "status": "FORMAL",
               "selection_sha256": sha256(selection_path), "checkpoint_sha256": sha256(checkpoint),
               "protocol": protocol.protocol_dict(), "source_binding": record_binding(source),
               "dev_binding": record_binding(development), "variants": variants,
               "code_hashes": source_hashes(), "final_sessions_opened": 0}
    atomic_json(dest / "receipt.json", receipt)
    return receipt
