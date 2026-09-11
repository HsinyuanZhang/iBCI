#!/usr/bin/env python3
"""Isolated M2 R50/D4 concat-flat trainer; never changes the recency runner."""
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
WORKSPACE, V1 = ROOT.parent, ROOT.parent / "btransform_unified_v1"
for path in (ROOT, ROOT / "src", V1 / "src", V1 / "scripts", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from btransform_unified_v1 import adapters, plan as v1_plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2.streaming import RiftStreamDecoder
from scripts.rift_v1 import m2_concat_train as frozen
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan
from tfpd_exploration.src.m2_dual_track_v1 import sampler as old_sampler
from tfpd_exploration.src.m2_dual_track_v1 import training as old_training

CELL = "M2-RIFT-R50-D4-CONCAT-E50-FLAT-V1"
SCHEMA = "m2_rift_concat_flat_train_v1"
CHECKPOINT_SCHEMA = "m2_rift_concat_flat_epoch_checkpoint_v1"
SEED, CONTEXT, EPOCHS, BATCH = 42, 50, 24, 32
OUT_ROOT = ROOT / "results/recency_flat_ablation_v1"
DEFAULT_FORMAL = OUT_ROOT / "formal_m2_concat_flat_s42"
DEFAULT_SMOKE = OUT_ROOT / "root_smoke_m2_flat_s42"
REFERENCE = ROOT / "results/rift_v1/m2_r50_concat_s42_formal_v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atom(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def source_hashes() -> dict[str, str]:
    """Bind frozen helper code and every new implementation source."""
    paths = {Path(name) for name in frozen._source_hashes()}
    paths.update((Path(__file__).resolve(), Path(frozen.__file__).resolve()))
    return {str(path): sha(path) for path in sorted(paths)}


def reference_binding() -> dict[str, Any]:
    meta_path, receipt_path = REFERENCE / "run_meta.json", REFERENCE / "train_receipt.json"
    meta, receipt = json.loads(meta_path.read_text()), json.loads(receipt_path.read_text())
    needed = {"schema": "m2_rift_concat_train_v1", "status": "FORMAL", "cell": frozen.CELL,
              "seed": SEED, "context_bins": CONTEXT, "layer_windows": [13, 12, 12, 12],
              "attention_backend": "local", "identity_interface": "concat", "epochs": EPOCHS,
              "batch": BATCH, "updates_per_epoch": 3165}
    if any(meta.get(key) != value for key, value in needed.items()):
        raise RuntimeError("frozen recency reference metadata contract drift")
    if receipt.get("status") != "COMPLETED" or receipt.get("epochs") != EPOCHS or receipt.get("global_step") != EPOCHS * 3165:
        raise RuntimeError("frozen recency reference receipt contract drift")
    if meta.get("source_hashes") != frozen._source_hashes():
        raise RuntimeError("current frozen helper sources differ from recency formal source seal")
    recipe = {"optimizer": {"name": "AdamW", "weight_decay": v1_plan.WEIGHT_DECAY, "betas": list(old_plan.ADAM_BETAS), "eps": old_plan.ADAM_EPS, "clip": v1_plan.GRAD_CLIP},
              "lr": {"peak": v1_plan.LR_PEAK, "min": v1_plan.LR_PEAK * v1_plan.LR_MIN_FACTOR, "warmup_updates": 3165},
              "ema_decay": v1_plan.EMA_DECAY, "unit_dropout": v1_plan.UNIT_DROPOUT}
    if any(meta.get(key) != value for key, value in recipe.items()):
        raise RuntimeError("current optimizer/LR/EMA/dropout constants differ from recency formal recipe")
    return {"run": str(REFERENCE), "run_meta_sha256": sha(meta_path), "train_receipt_sha256": sha(receipt_path),
            "manifest_digest": meta["manifest_digest"], "frozen_cache_hashes": meta["frozen_cache_hashes"],
            "frozen_source_hashes": meta["source_hashes"],
            "recipe": {key: meta[key] for key in ("seed", "context_bins", "layer_windows", "depth", "width", "batch", "epochs", "updates_per_epoch", "optimizer", "lr", "ema_decay", "unit_dropout", "padding_contract")}}


def flat_decoder(device: torch.device) -> RiftConcatDecoder:
    model = RiftConcatDecoder("m2", context_bins=CONTEXT, bias_mode="flat", seed=SEED).to(device)
    model.temporal.set_attention_backend("local")
    if tuple(model.temporal_config.windows) != (13, 12, 12, 12):
        raise RuntimeError("flat R50 D4 windows drift")
    slopes = model.temporal.recency_slopes
    if slopes.dtype != torch.float32 or tuple(slopes.shape) != (8,) or torch.count_nonzero(slopes).item() != 0:
        raise RuntimeError("flat temporal slopes are not exact FP32 zeros")
    return model


def assert_paired_initialization(flat: RiftConcatDecoder, device: torch.device) -> dict[str, Any]:
    recency = RiftConcatDecoder("m2", context_bins=CONTEXT, bias_mode="recency", seed=SEED).to(device)
    recency.temporal.set_attention_backend("local")
    flat_parameters, recency_parameters = dict(flat.named_parameters()), dict(recency.named_parameters())
    if flat_parameters.keys() != recency_parameters.keys() or any(not torch.equal(flat_parameters[name], recency_parameters[name]) for name in flat_parameters):
        raise RuntimeError("flat/recency same-seed named parameters differ")
    flat_state, recency_state = flat.state_dict(), recency.state_dict()
    if flat_state.keys() != recency_state.keys():
        raise RuntimeError("flat/recency state keys differ")
    differing = [name for name in flat_state if not torch.equal(flat_state[name], recency_state[name])]
    if differing != ["temporal.recency_slopes"]:
        raise RuntimeError(f"unexpected flat/recency nonparameter delta: {differing}")
    return {"reference_factory": "RiftConcatDecoder(m2,R50,recency,seed=42)", "named_parameters_byte_equal": True,
            "sole_state_delta": "temporal.recency_slopes", "flat_slopes": flat.temporal.recency_slopes.detach().cpu().tolist(),
            "recency_slopes": recency.temporal.recency_slopes.detach().cpu().tolist()}


def cache_hashes(surface: str, dual: Mapping[str, Any], banks: Mapping[str, Any]) -> dict[str, Any]:
    return frozen._cache_hashes(surface, dual, banks)


def load_surface(surface: str, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    return frozen._load_surface(surface, device)


def padding(surface: str, banks: Mapping[str, Any]) -> dict[str, int]:
    return frozen._surface_padding(surface, banks)


def rng_state(device: torch.device) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda" and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def check_zero_slopes(model: RiftConcatDecoder, label: str) -> None:
    slopes = model.temporal.recency_slopes
    if slopes.dtype != torch.float32 or tuple(slopes.shape) != (8,) or torch.count_nonzero(slopes).item() != 0:
        raise RuntimeError(f"{label}: flat slopes drift")


def checkpoint(epoch: int, step: int, model: RiftConcatDecoder, optimizer: torch.optim.Optimizer, ema: DecoderEMA, *, smoke: bool, manifest_digest: str, caches: Mapping[str, Any], init: Mapping[str, Any]) -> dict[str, Any]:
    check_zero_slopes(model, "checkpoint")
    return {"schema": CHECKPOINT_SCHEMA, "cell": CELL, "epoch": epoch, "global_step": step, "seed": SEED,
            "context_bins": CONTEXT, "bias_mode": "flat", "attention_backend": "local", "identity_interface": "concat",
            "identity_e0_dim": 50, "proj_dim": None, "epochs": EPOCHS, "smoke": smoke, "manifest_digest": manifest_digest,
            "source_hashes": source_hashes(), "frozen_cache_hashes": caches, "reference_binding": reference_binding(),
            "initialization_pairing": dict(init), "flat_slopes": [0.0] * 8, "ema_updates": ema.n_updates, "raw_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(), "ema": ema.state_dict(), "rng": rng_state(next(model.parameters()).device)}


def atomic_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def metadata(manifest: Mapping[str, Any], caches: Mapping[str, Any], init: Mapping[str, Any], *, smoke: bool, device: torch.device) -> dict[str, Any]:
    reference = reference_binding()
    return {"schema": SCHEMA, "status": "SMOKE" if smoke else "FORMAL", "cell": CELL, "seed": SEED, "task": "m2",
            "variant": "flat", "bias_mode": "flat", "identity_interface": "concat", "identity_e0_dim": 50, "proj_dim": None,
            "context_bins": CONTEXT, "layer_windows": [13, 12, 12, 12], "depth": 4, "width": 256, "attention_backend": "local",
            "batch": BATCH, "epochs": EPOCHS, "updates_per_epoch": 3165, "source_train_only_for_gradients": True,
            "development_surfaces": ["source_minival", "ext4"], "official_test_used": False, "evalai_opened": False,
            "padding_contract": "valid_mask[start+offset >= 49]; values are never classified from spike magnitude",
            "manifest_path": str(frozen.MANIFEST), "manifest_digest": manifest["digest"], "manifest_digest_expected": frozen.MANIFEST_DIGEST,
            "optimizer": {"name": "AdamW", "weight_decay": v1_plan.WEIGHT_DECAY, "betas": list(old_plan.ADAM_BETAS), "eps": old_plan.ADAM_EPS, "clip": v1_plan.GRAD_CLIP},
            "lr": {"peak": v1_plan.LR_PEAK, "min": v1_plan.LR_PEAK * v1_plan.LR_MIN_FACTOR, "warmup_updates": 3165},
            "ema_decay": v1_plan.EMA_DECAY, "unit_dropout": v1_plan.UNIT_DROPOUT, "source_hashes": source_hashes(),
            "frozen_cache_hashes": caches, "paired_recency_reference": reference, "initialization_pairing": dict(init),
            "flat_slopes": [0.0] * 8, "runtime": {"device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu_threads": torch.get_num_threads(), "torch": torch.__version__, "numpy": np.__version__}, "utc": datetime.now(timezone.utc).isoformat()}


def validate_resume(state: Mapping[str, Any], meta: Mapping[str, Any], *, smoke: bool) -> None:
    required = {"schema": CHECKPOINT_SCHEMA, "cell": CELL, "seed": SEED, "context_bins": CONTEXT, "bias_mode": "flat",
                "attention_backend": "local", "identity_interface": "concat", "identity_e0_dim": 50, "proj_dim": None,
                "epochs": EPOCHS, "smoke": smoke, "flat_slopes": [0.0] * 8}
    if any(state.get(key) != value for key, value in required.items()):
        raise RuntimeError("flat resume checkpoint configuration drift")
    for key in ("source_hashes", "frozen_cache_hashes", "reference_binding", "initialization_pairing", "manifest_digest"):
        expected = meta.get("paired_recency_reference") if key == "reference_binding" else meta.get(key)
        if state.get(key) != expected:
            raise RuntimeError(f"flat resume checkpoint {key} drift")
    expected_updates = int(state.get("global_step", -1)) if smoke else int(state.get("epoch", -1)) * 3165
    if state.get("ema_updates") != expected_updates:
        raise RuntimeError("flat resume EMA update count drift")


def smoke_full_stream_parity(model: RiftConcatDecoder, batch: Any, bank: Any, valid: torch.Tensor) -> dict[str, Any]:
    """One real M2 batch row, checked before and after the smoke optimizer step."""
    was_training = model.training; model.eval()
    raw, mask = batch.X[:1], valid[:1]
    with torch.inference_mode():
        full = model(raw, bank, input_valid_mask=mask)
        stream = RiftStreamDecoder(model); streamed = None
        for offset in range(CONTEXT):
            streamed = stream.stream_step(raw[:, offset], bank, ["m2-flat-smoke"], valid_mask=mask[:, offset])
    if was_training: model.train()
    if streamed is None or not torch.allclose(full, streamed, rtol=2e-5, atol=2e-5):
        raise RuntimeError("flat M2 smoke full/stream parity failed")
    return {"passed": True, "atol": 2e-5, "max_abs": float((full - streamed).abs().max()), "bins": CONTEXT}


def run(args: argparse.Namespace) -> dict[str, Any]:
    smoke = args.max_updates_smoke is not None
    if not smoke and args.epochs != EPOCHS:
        raise ValueError("formal flat M2 requires exactly 24 epochs")
    device, dest = torch.device(args.device), args.dest.resolve()
    torch.set_num_threads(args.cpu_threads); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    manifest = old_sampler.load_manifest(frozen.MANIFEST)
    if manifest["digest"] != frozen.MANIFEST_DIGEST or int(manifest["batch_size"]) != BATCH:
        raise RuntimeError("frozen M2 sampler manifest drift")
    if args.resume is None and dest.exists() and any(dest.iterdir()):
        raise FileExistsError("fresh flat destination must be empty")
    if args.resume is not None and args.resume.resolve().parent != dest:
        raise RuntimeError("flat resume must belong directly to --dest")
    train_dual, train_banks = load_surface("source_train", device)
    mini_dual, mini_banks = load_surface("source_minival", device)
    ext_dual, ext_banks = load_surface("ext4", device)
    if old_training.count_updates(train_dual) != 3165:
        raise RuntimeError("source train update budget drift")
    caches = {"source_train": cache_hashes("source_train", train_dual, train_banks), "source_minival": cache_hashes("source_minival", mini_dual, mini_banks), "ext4": cache_hashes("ext4", ext_dual, ext_banks)}
    reference = reference_binding()
    if caches != reference["frozen_cache_hashes"] or manifest["digest"] != reference["manifest_digest"]:
        raise RuntimeError("flat input cache/manifest differs from paired recency reference")
    pads = {surface: padding(surface, banks) for surface, banks in (("source_train", train_banks), ("source_minival", mini_banks), ("ext4", ext_banks))}
    model = flat_decoder(device); init = assert_paired_initialization(model, device); check_zero_slopes(model, "post-init")
    ema = DecoderEMA(model, decay=v1_plan.EMA_DECAY)
    optimizer = old_training.build_optimizer(model.named_parameters(), lr=v1_plan.LR_PEAK, weight_decay=v1_plan.WEIGHT_DECAY)
    meta = metadata(manifest, caches, init, smoke=smoke, device=device)
    dest.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        atom(dest / "run_meta.json", meta)
    else:
        existing = json.loads((dest / "run_meta.json").read_text())
        existing.pop("utc", None); meta_without_utc = dict(meta); meta_without_utc.pop("utc", None)
        if existing != meta_without_utc:
            raise RuntimeError("flat resume metadata differs from current paired contract")
    step, start_epoch = 0, 1
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False); validate_resume(state, meta, smoke=smoke)
        model.load_state_dict(state["raw_state_dict"]); optimizer.load_state_dict(state["optimizer"]); ema.load_state_dict(state["ema"]); restore_rng(state["rng"], device)
        check_zero_slopes(model, "post-resume-load")
        step, start_epoch = int(state["global_step"]), int(state["epoch"]) + 1
        if start_epoch > EPOCHS:
            raise RuntimeError("resume checkpoint is already complete")
    started = time.monotonic(); total = EPOCHS * 3165
    smoke_preupdate_parity = smoke_postupdate_parity = None
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train(); losses: list[float] = []
        for batch_index, batch in enumerate(old_sampler.iter_manifest_batches(train_dual, manifest, epoch, device=device, target_space=old_plan.TRAINING_TARGET_SPACE)):
            step += 1; lr = warmup_cosine_lr(step, total_steps=total, warmup_steps=3165, peak=v1_plan.LR_PEAK, min_factor=v1_plan.LR_MIN_FACTOR)
            for group in optimizer.param_groups: group["lr"] = lr
            keep_rng = torch.Generator(device="cpu"); keep_rng.manual_seed(unit_dropout_seed(SEED, epoch, batch_index))
            keep = whole_unit_dropout(batch.unit_mask, p=v1_plan.UNIT_DROPOUT, generator=keep_rng)
            valid = frozen._batch_valid(batch, surface="source_train", padding=pads["source_train"], device=device)
            if smoke and smoke_preupdate_parity is None:
                smoke_preupdate_parity = smoke_full_stream_parity(model, batch, train_banks[batch.session_id], valid)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                pred = model(batch.X, train_banks[batch.session_id], dropout_keep=keep, input_valid_mask=valid)
                loss = nn.functional.mse_loss(pred.float(), batch.last_target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite flat loss at epoch={epoch}, batch={batch_index}")
            loss.backward(); grad = float(nn.utils.clip_grad_norm_(model.parameters(), v1_plan.GRAD_CLIP, error_if_nonfinite=True)); optimizer.step(); ema.update_after_step(model)
            if smoke and smoke_postupdate_parity is None:
                check_zero_slopes(model, "smoke-post-step")
                smoke_postupdate_parity = smoke_full_stream_parity(model, batch, train_banks[batch.session_id], valid)
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 100 == 0:
                atom(dest / "heartbeat.json", {"status": "TRAINING", "event": "step", "epoch": epoch, "global_step": step, "loss": losses[-1], "lr": lr, "grad_norm": grad, "elapsed_seconds": time.monotonic() - started, "flat_slopes_zero": True})
            if smoke and step >= args.max_updates_smoke:
                break
        score_limit = 1 if smoke else None
        raw = frozen._score(model, mini_dual, mini_banks, pads["source_minival"], "source_minival", device, max_batches_per_session=score_limit)
        ema_score = frozen._with_ema(model, ema, lambda: frozen._score(model, mini_dual, mini_banks, pads["source_minival"], "source_minival", device, max_batches_per_session=score_limit))
        check_zero_slopes(model, "post-ema-minival")
        atomic_checkpoint(dest / f"epoch_{epoch:03d}.pt", checkpoint(epoch, step, model, optimizer, ema, smoke=smoke, manifest_digest=manifest["digest"], caches=caches, init=init))
        elapsed = time.monotonic() - started
        epoch_row = {"status": "SMOKE" if smoke else "TRAINING", "event": "epoch", "epoch": epoch, "global_step": step, "epoch_update_count": len(losses), "train_mse": float(np.mean(losses)), "ema_updates": ema.n_updates, "elapsed_seconds": elapsed, "elapsed_seconds_per_update": elapsed / step, "minival_raw": raw, "minival_ema": ema_score, "flat_slopes_zero": True}
        atom(dest / "heartbeat.json", epoch_row); append_jsonl(dest / "metrics.jsonl", epoch_row)
        if not smoke and (len(losses) != 3165 or step != epoch * 3165 or ema.n_updates != step):
            raise RuntimeError("formal flat epoch update/EMA accounting drift")
        if smoke:
            receipt = {"schema": "m2_rift_concat_flat_smoke_receipt_v1", "status": "COMPLETED", "cell": CELL, "epoch": epoch, "global_step": step, "partial_minival": True, "finite_train_mse": bool(math.isfinite(float(np.mean(losses)))), "checkpoint": str(dest / f"epoch_{epoch:03d}.pt"), "runtime_seconds": elapsed, "runtime": meta["runtime"], "flat_slopes_zero": True, "full_stream_parity_preupdate": smoke_preupdate_parity, "full_stream_parity_postupdate": smoke_postupdate_parity, "source_hashes": meta["source_hashes"], "frozen_cache_hashes": caches, "reference_binding": reference, "finished_utc": datetime.now(timezone.utc).isoformat()}
            atom(dest / "smoke_receipt.json", receipt); return receipt
    if step != EPOCHS * 3165 or ema.n_updates != EPOCHS * 3165:
        raise RuntimeError("formal flat total update/EMA accounting drift")
    receipt = {"schema": "m2_rift_concat_flat_train_receipt_v1", "status": "COMPLETED", "cell": CELL, "epochs": EPOCHS, "global_step": step, "manifest_digest": manifest["digest"], "runtime_seconds": time.monotonic() - started, "runtime": meta["runtime"], "flat_slopes_zero": True, "source_hashes": meta["source_hashes"], "frozen_cache_hashes": caches, "reference_binding": reference, "finished_utc": datetime.now(timezone.utc).isoformat()}
    atom(dest / "train_receipt.json", receipt); return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_FORMAL); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4); parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=Path); parser.add_argument("--max-updates-smoke", type=int)
    args = parser.parse_args()
    if args.epochs != EPOCHS or args.cpu_threads < 1 or args.max_updates_smoke is not None and args.max_updates_smoke < 1:
        parser.error("formal flat run is fixed at 24 epochs; thread/smoke counts must be positive")
    if args.max_updates_smoke is not None and args.dest == DEFAULT_FORMAL:
        args.dest = DEFAULT_SMOKE
    if args.max_updates_smoke is not None and args.device == "cuda:0":
        args.device = "cpu"
    print(json.dumps(run(args), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
