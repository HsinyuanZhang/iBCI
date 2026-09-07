#!/usr/bin/env python3
"""Formal M2 RIFT-R50-D4 recency trainer.

This runner intentionally keeps the frozen M2 source-seven/M33+B3S cache and
the old M2 batch manifest.  It replaces only the temporal decoder with the
RIFT local-attention stack.  Padding on the query timeline is represented by
``input_valid_mask`` and is never supplied as temporal evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT.parent / "btransform_unified_v1"
WORKSPACE = ROOT.parent
for _path in (ROOT / "src", V1 / "src", WORKSPACE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from btransform_unified_v1 import adapters, plan as v1_plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v1.schedule import warmup_cosine_lr
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan
from tfpd_exploration.src.m2_dual_track_v1 import sampler as old_sampler
from tfpd_exploration.src.m2_dual_track_v1 import training as old_training

CELL = "M2-RIFT-R50-D4-P16-RECENCY-V1"
SEED = 42
CONTEXT = 50
PROJ_DIM = 16
EPOCHS = 24
BATCH = 32
MANIFEST = WORKSPACE / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/sampler/shuffled_batch_manifest_24.json"
MANIFEST_DIGEST = "a95255fa339ea06f1e1cc3ef9f53f3d49d15799bf95e4579f672417fd878d79a"
QUERY_PAD_BINS = CONTEXT - 1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _heartbeat(dest: Path, payload: Mapping[str, Any], *, status: str) -> None:
    row = {"status": status, "pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat(), **payload}
    _atomic_json(dest / "heartbeat.json", row)


def _atomic_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state), temporary)
    temporary.replace(path)


def _decoder(device: torch.device):
    # Lazy import keeps --help and metadata inspection independent of CUDA/model edits.
    from btransform_unified_v2 import RiftDecoder
    model = RiftDecoder("m2", context_bins=CONTEXT, bias_mode="recency", seed=SEED, proj_dim=PROJ_DIM).to(device)
    model.temporal.set_attention_backend("local")
    windows = tuple(model.temporal_config.windows)
    if windows != (13, 12, 12, 12):
        raise RuntimeError(f"R50 D4 layer windows drifted: {windows}")
    return model


def valid_mask_from_padded_starts(window_ids: tuple[int, ...], *, query_pad_bins: int, device: torch.device) -> torch.Tensor:
    """Construct validity from the real padded-timeline coordinate, never X==0.

    A true zero spike bin remains valid.  The leading 49 query-padding bins
    are invalid when a window overlaps them.  Formal M33 windows normally all
    begin after this boundary, but deriving the mask here makes that fact an
    auditable property instead of silently turning padding into observations.
    """
    starts = torch.as_tensor(window_ids, dtype=torch.long, device=device).unsqueeze(1)
    offsets = torch.arange(CONTEXT, dtype=torch.long, device=device).unsqueeze(0)
    return starts + offsets >= int(query_pad_bins)


def _surface_padding(surface: str, banks: Mapping[str, Any]) -> dict[str, int]:
    """Read the cache's frozen timeline metadata; do not infer from spike values."""
    pads: dict[str, int] = {}
    for session, bank in banks.items():
        root = Path(str(bank.calibration_meta["cache_root"]))
        mapping = json.loads((root / surface / session / "mapping.json").read_text(encoding="utf-8"))
        if mapping.get("query_is_padded_timeline") is not True:
            raise RuntimeError(f"{surface}/{session} no longer declares padded query timeline")
        pad = int(mapping["query_pad_bins"])
        if pad != QUERY_PAD_BINS:
            raise RuntimeError(f"{surface}/{session} query padding drifted: {pad}")
        pads[session] = pad
    return pads


def _batch_valid(batch: Any, *, surface: str, padding: Mapping[str, int], device: torch.device) -> torch.Tensor:
    if surface == "source_train":
        # All seven gradient windows are post-M33 legal windows.  Record this
        # explicitly; do not manufacture a pseudo-padding mask from their X.
        return torch.ones((len(batch.window_ids), CONTEXT), dtype=torch.bool, device=device)
    return valid_mask_from_padded_starts(batch.window_ids, query_pad_bins=padding[batch.session_id], device=device)


def _cache_hashes(surface: str, dual: Mapping[str, Any], banks: Mapping[str, Any]) -> dict[str, Any]:
    return {
        session: {
            "eligible_starts_sha256": hashlib.sha256(np.asarray(value.eligible_starts, dtype=np.int64).tobytes()).hexdigest(),
            "e0_sha256": str(banks[session].calibration_meta.get("array_sha256")),
            "carrier_sha256": str(banks[session].calibration_meta.get("carrier_sha256")),
            "x_store_sha256": str(banks[session].calibration_meta.get("x_store_sha256")),
            "window_count": int(len(value.eligible_starts)),
            "surface": surface,
        }
        for session, value in dual.items()
    }


def _load_surface(surface: str, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    dual = old_data.load_surface_banks(surface, device=device)
    sessions = old_plan.EXT4_SESSIONS if surface == "ext4" else old_plan.HELDIN_SESSIONS
    banks = {session: adapters.build_m2_bank(surface, session, budget=33) for session in sessions}
    return dict(dual), banks


def _score(model: nn.Module, dual: Mapping[str, Any], banks: Mapping[str, Any], padding: Mapping[str, int], surface: str, device: torch.device, *, max_batches_per_session: int | None = None) -> dict[str, Any]:
    model.eval()
    per_session: dict[str, Any] = {}
    pooled_t: list[np.ndarray] = []; pooled_p: list[np.ndarray] = []
    for session, dual_bank in dual.items():
        targets: list[np.ndarray] = []; preds: list[np.ndarray] = []
        for batch_index, batch in enumerate(old_data.iter_session_batches(dual_bank, batch_size=BATCH, device=device, target_space=old_plan.SCORING_TARGET_SPACE)):
            if max_batches_per_session is not None and batch_index >= max_batches_per_session: break
            valid = _batch_valid(batch, surface=surface, padding=padding, device=device)
            with torch.inference_mode():
                raw = model(batch.X, banks[session], input_valid_mask=valid)
            pred = np.ascontiguousarray(raw.float().cpu().numpy() / old_plan.BEHAVIOR_SCALE, dtype=np.float32)
            target = np.ascontiguousarray(batch.last_target.float().cpu().numpy(), dtype=np.float32)
            preds.append(pred); targets.append(target)
        p = np.concatenate(preds); t = np.concatenate(targets)
        r2 = float(variance_weighted_r2(t, p))
        per_session[session] = {"r2": r2, "window_count": int(len(t)), "prediction_sha256": hashlib.sha256(p.tobytes()).hexdigest()}
        pooled_p.append(p); pooled_t.append(t)
    ordered = [per_session[s]["r2"] for s in sorted(per_session)]
    return {"per_session": per_session, "equal_session_mean": float(np.mean(ordered)), "partial": max_batches_per_session is not None,
            "pooled_r2": float(variance_weighted_r2(np.concatenate(pooled_t), np.concatenate(pooled_p))),
            "n_windows": int(sum(row["window_count"] for row in per_session.values()))}


def _with_ema(model: nn.Module, ema: DecoderEMA, fn):
    named = dict(model.named_parameters())
    backup = {name: value.detach().clone() for name, value in named.items()}
    was_training = model.training
    try:
        with torch.no_grad():
            for name, value in named.items(): value.copy_(ema.shadow[name].to(value.device, dtype=value.dtype))
        return fn()
    finally:
        with torch.no_grad():
            for name, value in named.items(): value.copy_(backup[name])
        model.train(was_training)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def _restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda" and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])


def _checkpoint(epoch: int, step: int, model: nn.Module, optimizer: torch.optim.Optimizer, ema: DecoderEMA, device: torch.device, manifest_digest: str, *, smoke: bool, cache_hashes: Mapping[str, Any], epochs: int) -> dict[str, Any]:
    return {"schema": "m2_rift_epoch_checkpoint_v1", "cell": CELL, "epoch": epoch, "global_step": step,
            "seed": SEED, "context_bins": CONTEXT, "bias_mode": "recency", "attention_backend": "local", "proj_dim": PROJ_DIM,
            "manifest_digest": manifest_digest, "smoke": smoke, "epochs": epochs, "source_hashes": _source_hashes(), "frozen_cache_hashes": cache_hashes, "raw_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(), "rng": _rng_state(device)}


def _source_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "src/btransform_unified_v2/model.py", ROOT / "src/btransform_unified_v2/temporal.py", ROOT / "src/btransform_unified_v2/config.py", MANIFEST]
    return {str(path): _sha(path) for path in paths}


def _run_meta(model: nn.Module, manifest: Mapping[str, Any], train_dual, train_banks, mini_dual, mini_banks, ext_dual, ext_banks, *, smoke: bool, epochs: int) -> dict[str, Any]:
    return {"schema": "m2_rift_train_v1", "status": "SMOKE" if smoke else "FORMAL", "cell": CELL, "seed": SEED,
            "task": "m2", "variant": "recency", "identity_interface": "proj_add", "proj_dim": PROJ_DIM,
            "context_bins": CONTEXT, "layer_windows": list(model.temporal_config.windows), "depth": 4, "width": 256,
            "attention_backend": "local", "batch": BATCH, "epochs": epochs, "source_train_only_for_gradients": True,
            "development_surfaces": ["source_minival", "ext4"], "official_test_used": False,
            "padding_contract": "valid_mask[start+offset >= 49]; values are never classified from spike magnitude",
            "manifest_path": str(MANIFEST), "manifest_digest": manifest["digest"], "manifest_digest_expected": MANIFEST_DIGEST,
            "updates_per_epoch": old_training.count_updates(train_dual), "behavior_scale": old_plan.BEHAVIOR_SCALE,
            "optimizer": {"name": "AdamW", "weight_decay": v1_plan.WEIGHT_DECAY, "betas": list(old_plan.ADAM_BETAS), "eps": old_plan.ADAM_EPS, "clip": v1_plan.GRAD_CLIP},
            "lr": {"peak": v1_plan.LR_PEAK, "min": v1_plan.LR_PEAK * v1_plan.LR_MIN_FACTOR, "warmup_updates": old_training.count_updates(train_dual)},
            "ema_decay": v1_plan.EMA_DECAY, "unit_dropout": v1_plan.UNIT_DROPOUT,
            "source_hashes": _source_hashes(), "frozen_cache_hashes": {"source_train": _cache_hashes("source_train", train_dual, train_banks), "source_minival": _cache_hashes("source_minival", mini_dual, mini_banks), "ext4": _cache_hashes("ext4", ext_dual, ext_banks)},
            "launch": {"argv": sys.argv, "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, "utc": datetime.now(timezone.utc).isoformat()}


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    smoke = args.max_updates_smoke is not None
    if not smoke and args.epochs != EPOCHS: raise ValueError("formal M2 RIFT requires exactly 24 epochs")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    manifest = old_sampler.load_manifest(MANIFEST)
    if manifest["digest"] != MANIFEST_DIGEST: raise RuntimeError("frozen 24-epoch source manifest digest mismatch")
    dest = args.dest.resolve()
    if args.resume is None and dest.exists() and any(dest.iterdir()): raise FileExistsError("new run destination must be empty")
    if args.resume is not None and not (dest / "run_meta.json").is_file(): raise RuntimeError("--resume requires existing matching --dest/run_meta.json")
    if args.resume is not None and args.resume.resolve().parent != dest: raise RuntimeError("resume checkpoint must belong directly to --dest")
    train_dual, train_banks = _load_surface("source_train", device)
    mini_dual, mini_banks = _load_surface("source_minival", device)
    ext_dual, ext_banks = _load_surface("ext4", device)
    padding = {surface: _surface_padding(surface, banks) for surface, banks in (("source_train", train_banks), ("source_minival", mini_banks), ("ext4", ext_banks))}
    updates = old_training.count_updates(train_dual)
    if updates != 3165 or int(manifest["batch_size"]) != BATCH: raise RuntimeError(f"M2 batch contract drift: updates={updates}")
    model = _decoder(device); ema = DecoderEMA(model, decay=v1_plan.EMA_DECAY)
    optimizer = old_training.build_optimizer(model.named_parameters(), lr=v1_plan.LR_PEAK, weight_decay=v1_plan.WEIGHT_DECAY)
    dest.mkdir(parents=True, exist_ok=True)
    current_meta = _run_meta(model, manifest, train_dual, train_banks, mini_dual, mini_banks, ext_dual, ext_banks, smoke=smoke, epochs=args.epochs)
    cache_hashes = current_meta["frozen_cache_hashes"]
    if args.resume is None:
        _atomic_json(dest / "run_meta.json", current_meta)
    else:
        existing = json.loads((dest / "run_meta.json").read_text())
        for key, value in (("status", "FORMAL"), ("cell", CELL), ("manifest_digest", MANIFEST_DIGEST), ("context_bins", CONTEXT), ("attention_backend", "local")):
            if existing.get(key) != value: raise RuntimeError(f"resume metadata mismatch for {key}")
        if existing.get("source_hashes") != current_meta["source_hashes"] or existing.get("frozen_cache_hashes") != cache_hashes:
            raise RuntimeError("resume source/cache contract mismatch")
    step = 0; start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        required = {"schema": "m2_rift_epoch_checkpoint_v1", "cell": CELL, "manifest_digest": MANIFEST_DIGEST, "seed": SEED, "context_bins": CONTEXT, "bias_mode": "recency", "attention_backend": "local", "proj_dim": PROJ_DIM, "smoke": False, "epochs": EPOCHS}
        if any(state.get(key) != value for key, value in required.items()): raise RuntimeError("resume checkpoint configuration mismatch")
        if state.get("source_hashes") != current_meta["source_hashes"] or state.get("frozen_cache_hashes") != cache_hashes: raise RuntimeError("resume checkpoint source/cache mismatch")
        if state.get("epoch", 0) >= args.epochs: raise RuntimeError("resume checkpoint already reaches requested epoch")
        model.load_state_dict(state["raw_state_dict"]); optimizer.load_state_dict(state["optimizer"]); ema.load_state_dict(state["ema"]); _restore_rng(state["rng"], device)
        step = int(state["global_step"]); start_epoch = int(state["epoch"]) + 1
    started = time.monotonic(); total = updates * EPOCHS
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); losses: list[float] = []
        for batch_id, batch in enumerate(old_sampler.iter_manifest_batches(train_dual, manifest, epoch, device=device, target_space=old_plan.TRAINING_TARGET_SPACE)):
            step += 1; lr = warmup_cosine_lr(step, total_steps=total, warmup_steps=updates, peak=v1_plan.LR_PEAK, min_factor=v1_plan.LR_MIN_FACTOR)
            for group in optimizer.param_groups: group["lr"] = lr
            keep_rng = torch.Generator(device="cpu"); keep_rng.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
            keep = whole_unit_dropout(batch.unit_mask, p=v1_plan.UNIT_DROPOUT, generator=keep_rng)
            valid = _batch_valid(batch, surface="source_train", padding=padding["source_train"], device=device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                pred = model(batch.X, train_banks[batch.session_id], dropout_keep=keep, input_valid_mask=valid)
                loss = nn.functional.mse_loss(pred.float(), batch.last_target)
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"nonfinite loss epoch={epoch} batch={batch_id}")
            loss.backward(); grad = float(nn.utils.clip_grad_norm_(model.parameters(), v1_plan.GRAD_CLIP, error_if_nonfinite=True)); optimizer.step(); ema.update_after_step(model)
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 100 == 0:
                elapsed = time.monotonic() - started
                heartbeat = {"event": "step", "epoch": epoch, "global_step": step, "loss": losses[-1], "lr": lr, "grad_norm": grad,
                             "elapsed_seconds": elapsed, "elapsed_seconds_per_update": elapsed / step,
                             "cuda_peak_alloc_mib": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None}
                _heartbeat(dest, heartbeat, status="TRAINING"); _append_jsonl(dest / "metrics.jsonl", heartbeat)
            if smoke and step >= args.max_updates_smoke: break
        score_limit = 1 if smoke else None
        raw_mini = _score(model, mini_dual, mini_banks, padding["source_minival"], "source_minival", device, max_batches_per_session=score_limit)
        ema_mini = _with_ema(model, ema, lambda: _score(model, mini_dual, mini_banks, padding["source_minival"], "source_minival", device, max_batches_per_session=score_limit))
        _atomic_checkpoint(dest / f"epoch_{epoch:03d}.pt", _checkpoint(epoch, step, model, optimizer, ema, device, manifest["digest"], smoke=smoke, cache_hashes=cache_hashes, epochs=args.epochs))
        elapsed = time.monotonic() - started
        row = {"event": "epoch", "epoch": epoch, "global_step": step, "train_mse": float(np.mean(losses)), "lr": lr, "ema_updates": ema.n_updates,
               "elapsed_seconds": elapsed, "elapsed_seconds_per_update": elapsed / step,
               "cuda_peak_alloc_mib": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
               "minival_raw": raw_mini, "minival_ema": ema_mini, "smoke": smoke}
        _append_jsonl(dest / "metrics.jsonl", row); _heartbeat(dest, row, status="SMOKE" if smoke else "TRAINING")
        if smoke: break
    result = {"status": "SMOKE_COMPLETED" if smoke else "TRAIN_COMPLETED", "epochs_completed": list(range(start_epoch, epoch + 1)), "global_step": step, "dest": str(dest)}
    if smoke:
        _atomic_json(dest / "smoke_receipt.json", {"schema": "m2_rift_smoke_receipt_v1", "cell": CELL, "status": "COMPLETED", "epoch": epoch, "global_step": step, "device": str(device), "finite_train_mse": bool(math.isfinite(float(np.mean(losses)))), "partial_minival": True, "cuda_initialized": bool(torch.cuda.is_initialized()), "checkpoint": str(dest / f"epoch_{epoch:03d}.pt"), "source_hashes": current_meta["source_hashes"], "frozen_cache_hashes": cache_hashes, "finished_utc": datetime.now(timezone.utc).isoformat()})
    else:
        _atomic_json(dest / "train_receipt.json", {"schema": "m2_rift_train_receipt_v1", "cell": CELL, "status": "COMPLETED", "epochs": EPOCHS, "global_step": step, "manifest_digest": manifest["digest"], "source_hashes": current_meta["source_hashes"], "frozen_cache_hashes": cache_hashes, "finished_utc": datetime.now(timezone.utc).isoformat()})
        _heartbeat(dest, {"event": "train_complete", **result}, status="TRAIN_COMPLETED")
    return result


def score_stage(args: argparse.Namespace) -> dict[str, Any]:
    dest = args.dest.resolve(); meta = json.loads((dest / "run_meta.json").read_text())
    if meta.get("status") != "FORMAL" or meta.get("cell") != CELL or meta.get("official_test_used") is not False or not (dest / "train_receipt.json").is_file(): raise RuntimeError("score requires completed formal matching M2 RIFT run")
    device = torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    ext_dual, ext_banks = _load_surface("ext4", device); ext_padding = _surface_padding("ext4", ext_banks)
    if _source_hashes() != meta.get("source_hashes") or _cache_hashes("ext4", ext_dual, ext_banks) != meta.get("frozen_cache_hashes", {}).get("ext4"):
        raise RuntimeError("score current code/ext4 cache contract mismatch")
    model = _decoder(device); progress_path = dest / "ext4_score_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {"schema": "m2_rift_ext4_progress_v1", "cell": CELL, "source_hashes": meta["source_hashes"], "frozen_cache_hashes": meta["frozen_cache_hashes"], "completed": {}}
    if progress.get("cell") != CELL or progress.get("source_hashes") != meta["source_hashes"] or progress.get("frozen_cache_hashes") != meta["frozen_cache_hashes"]: raise RuntimeError("ext4 progress contract mismatch")
    for epoch in range(1, EPOCHS + 1):
        checkpoint_path = dest / f"epoch_{epoch:03d}.pt"; checkpoint_sha = _sha(checkpoint_path)
        prior = progress["completed"].get(str(epoch))
        if prior is not None and prior.get("checkpoint_sha256") == checkpoint_sha: continue
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state.get("smoke") or state.get("source_hashes") != meta["source_hashes"] or state.get("frozen_cache_hashes") != meta["frozen_cache_hashes"]: raise RuntimeError("ext4 checkpoint contract mismatch")
        model.load_state_dict(state["raw_state_dict"])
        ema = DecoderEMA(model, decay=v1_plan.EMA_DECAY); ema.load_state_dict(state["ema"])
        report = _with_ema(model, ema, lambda: _score(model, ext_dual, ext_banks, ext_padding, "ext4", device))
        progress["completed"][str(epoch)] = {**report, "checkpoint_sha256": checkpoint_sha}; _atomic_json(progress_path, progress); _heartbeat(dest, {"event": "ext4_score", "epoch": epoch, "completed_epochs": len(progress["completed"])}, status="SCORING")
    values = {int(k): float(v["equal_session_mean"]) for k, v in progress["completed"].items()}
    if set(values) != set(range(1, EPOCHS + 1)): raise RuntimeError("incomplete ext4 score curve")
    best_epoch = 1
    for candidate in range(2, EPOCHS + 1):
        if values[candidate] > values[best_epoch] + old_plan.TIE_EPS: best_epoch = candidate
    scan = {"schema": "m2_rift_ext4_epoch_scan_v1", "cell": CELL, "surface": "ext4", "official_test_used": False, "ema_by_epoch": progress["completed"], "selection": {"rule": "earliest best equal_session_mean", "epoch": best_epoch, "equal_session_mean": values[best_epoch]}, "endpoint24": progress["completed"]["24"]}
    _atomic_json(dest / "ext4_epoch_scan.json", scan); _atomic_json(dest / "score_receipt.json", {"status": "COMPLETED", **scan}); _heartbeat(dest, {"event": "score_complete", "selected_epoch": best_epoch}, status="SCORE_COMPLETED")
    return {"status": "SCORE_COMPLETED", "selected_epoch": best_epoch, "endpoint24": values[24]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal M2 RIFT recency trainer and ext4 scorer")
    parser.add_argument("--dest", type=Path, required=True); parser.add_argument("--stage", choices=("train", "score", "all"), default="train")
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--epochs", type=int, default=EPOCHS); parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates-smoke", type=int); parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > EPOCHS or args.max_updates_smoke is not None and args.max_updates_smoke < 1: parser.error("invalid epochs/smoke update count")
    if args.stage in ("score", "all") and args.max_updates_smoke is not None: parser.error("smoke runs cannot score ext4")
    result = run_train(args) if args.stage in ("train", "all") else None
    if args.stage in ("score", "all"):
        if result is not None and result["status"] != "TRAIN_COMPLETED": raise RuntimeError("formal training did not complete")
        result = score_stage(args)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
