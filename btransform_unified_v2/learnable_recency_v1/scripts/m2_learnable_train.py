#!/usr/bin/env python3
"""M2 R50 learnable-recency trainer. Mirrors m2_flat_train.py; does not edit it."""
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

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
WORKSPACE, V1 = ROOT.parent, ROOT.parent / "btransform_unified_v1"
FLAT_DIR = ROOT / "scripts/recency_flat_ablation_v1"
for path in (PKG / "src", FLAT_DIR, ROOT, ROOT / "src", V1 / "src", V1 / "scripts", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from btransform_unified_v1 import plan as v1_plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
from btransform_unified_v2.concat_model import RiftConcatDecoder
from learnable_recency_v1.config import add_learnable_flags, config_from_args, dataset_config
from learnable_recency_v1.wrap import (
    LearnableRiftConcatDecoder,
    LearnableRiftStreamDecoder,
    apply_group_lrs,
    assert_shared_byte_equal,
    install_temporal,
    new_parameter_names,
    recency_bias_snapshot,
    split_optimizer_parameters,
    trainable_new_parameter_count,
)
import m2_flat_train as flat
from scripts.rift_v1 import m2_concat_train as frozen
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan
from tfpd_exploration.src.m2_dual_track_v1 import sampler as old_sampler
from tfpd_exploration.src.m2_dual_track_v1 import training as old_training

SEED, CONTEXT, EPOCHS, BATCH = flat.SEED, flat.CONTEXT, flat.EPOCHS, flat.BATCH
RESULTS = PKG / "results"


def sha(path: Path) -> str:
    return flat.sha(path)


def atom(path: Path, value: Mapping[str, Any]) -> None:
    flat.atom(path, value)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    flat.append_jsonl(path, value)


def source_hashes() -> dict[str, str]:
    values = dict(flat.source_hashes())
    values[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    for path in sorted((PKG / "src/learnable_recency_v1").glob("*.py")):
        values[str(path)] = sha(path)
    return values


def learnable_decoder(device: torch.device, recency_cfg) -> LearnableRiftConcatDecoder:
    model = LearnableRiftConcatDecoder("m2", recency_cfg, context_bins=CONTEXT, seed=SEED).to(device)
    model.temporal.set_attention_backend("local")
    expected = tuple(recency_cfg.temporal_config.windows)
    if tuple(model.temporal_config.windows) != expected:
        raise RuntimeError(f"learnable R50 D{recency_cfg.layers} windows drift: {tuple(model.temporal_config.windows)} != {expected}")
    return model


def assert_paired_initialization(model: LearnableRiftConcatDecoder, device: torch.device, recency_cfg) -> dict[str, Any]:
    shared_cfg = dataset_config("m2", tier="fixed", layers=recency_cfg.layers, ladder="default")
    shared_ref = RiftConcatDecoder("m2", context_bins=CONTEXT, bias_mode="recency", seed=SEED).to(device)
    install_temporal(shared_ref, shared_cfg, SEED)
    shared_ref.temporal.set_attention_backend("local")
    shared = assert_shared_byte_equal(model, shared_ref)
    extra = new_parameter_names(model)
    ladder_cfg = dataset_config(
        "m2", tier="fixed", layers=recency_cfg.layers, half_life_seconds=recency_cfg.half_life_seconds
    )
    ladder_ref = RiftConcatDecoder("m2", context_bins=CONTEXT, bias_mode="recency", seed=SEED).to(device)
    install_temporal(ladder_ref, ladder_cfg, SEED)
    ladder_ref.temporal.set_attention_backend("local")
    z = torch.randn(2, CONTEXT, 256, device=device)
    mask = torch.ones(2, CONTEXT, dtype=torch.bool, device=device)
    mask[0, :3] = False
    with torch.inference_mode():
        left = model.temporal(z, mask)
        right = ladder_ref.temporal(z, mask)
    if not torch.allclose(left, right, atol=1e-6, rtol=1e-6):
        raise RuntimeError("learnable/same-ladder init forward drift")
    counts = {name: int(dict(model.named_parameters())[name].numel()) for name in extra}
    return {
        "reference_factory": f"RiftConcatDecoder(m2,R50,recency,seed=42,layers={recency_cfg.layers})",
        "named_parameters_byte_equal": True,
        "shared_parameter_names": shared,
        "new_parameter_names": extra,
        "new_parameter_counts": counts,
        "new_parameter_total": int(sum(counts.values())),
        "trainable_new_parameter_count": trainable_new_parameter_count(model),
        "init_forward_max_abs": float((left - right).abs().max().cpu()),
        "recency_slopes": model.temporal.recency_slopes.detach().cpu().tolist(),
        "stock_recency_slopes": shared_ref.temporal.recency_slopes.detach().cpu().tolist(),
        "recency_slopes_differ_from_stock": not torch.equal(
            model.temporal.recency_slopes.cpu(), shared_ref.temporal.recency_slopes.cpu()
        ),
        "tier": recency_cfg.tier,
        "ladder": recency_cfg.ladder_metadata(),
    }


def smoke_full_stream_parity(model: LearnableRiftConcatDecoder, batch: Any, bank: Any, valid: torch.Tensor) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    raw, mask = batch.X[:1], valid[:1]
    with torch.inference_mode():
        full = model(raw, bank, input_valid_mask=mask)
        stream = LearnableRiftStreamDecoder(model)
        streamed = None
        for offset in range(CONTEXT):
            streamed = stream.stream_step(raw[:, offset], bank, ["m2-learnable-smoke"], valid_mask=mask[:, offset])
    if was_training:
        model.train()
    if streamed is None or not torch.allclose(full, streamed, rtol=2e-5, atol=2e-5):
        raise RuntimeError("learnable M2 smoke full/stream parity failed")
    return {"passed": True, "atol": 2e-5, "max_abs": float((full - streamed).abs().max()), "bins": CONTEXT}


def bias_snapshot(model: LearnableRiftConcatDecoder, batch: Any, bank: Any, valid: torch.Tensor) -> dict[str, Any]:
    was = model.training
    model.eval()
    with torch.inference_mode():
        tokens = model.frontend_tokens(batch.X[: min(8, batch.X.shape[0])], bank)
        stats = recency_bias_snapshot(model.temporal, tokens, valid[: tokens.shape[0]])
    model.train(was)
    return stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    smoke = args.max_updates_smoke is not None
    recency_cfg = config_from_args(args, "m2")
    if not smoke and args.epochs != EPOCHS:
        raise ValueError("formal learnable M2 requires exactly 24 epochs")
    device, dest = torch.device(args.device), args.dest.resolve()
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    manifest = old_sampler.load_manifest(frozen.MANIFEST)
    if manifest["digest"] != frozen.MANIFEST_DIGEST or int(manifest["batch_size"]) != BATCH:
        raise RuntimeError("frozen M2 sampler manifest drift")
    if args.resume is None and dest.exists() and any(dest.iterdir()):
        raise FileExistsError("fresh learnable destination must be empty")
    if args.resume is not None and args.resume.resolve().parent != dest:
        raise RuntimeError("learnable resume must belong directly to --dest")
    train_dual, train_banks = flat.load_surface("source_train", device)
    mini_dual, mini_banks = flat.load_surface("source_minival", device)
    ext_dual, ext_banks = flat.load_surface("ext4", device)
    if old_training.count_updates(train_dual) != 3165:
        raise RuntimeError("source train update budget drift")
    caches = {
        "source_train": flat.cache_hashes("source_train", train_dual, train_banks),
        "source_minival": flat.cache_hashes("source_minival", mini_dual, mini_banks),
        "ext4": flat.cache_hashes("ext4", ext_dual, ext_banks),
    }
    reference = flat.reference_binding()
    if caches != reference["frozen_cache_hashes"] or manifest["digest"] != reference["manifest_digest"]:
        raise RuntimeError("learnable input cache/manifest differs from paired recency reference")
    pads = {
        surface: flat.padding(surface, banks)
        for surface, banks in (("source_train", train_banks), ("source_minival", mini_banks), ("ext4", ext_banks))
    }
    model = learnable_decoder(device, recency_cfg)
    init = assert_paired_initialization(model, device, recency_cfg)
    ema = DecoderEMA(model, decay=v1_plan.EMA_DECAY)
    groups = split_optimizer_parameters(
        model, peak_lr=v1_plan.LR_PEAK, weight_decay=v1_plan.WEIGHT_DECAY, lr_multiplier=recency_cfg.lr_multiplier
    )
    optimizer = torch.optim.AdamW(groups, lr=v1_plan.LR_PEAK, betas=old_plan.ADAM_BETAS, eps=old_plan.ADAM_EPS)
    meta = {
        "schema": "m2_rift_concat_learnable_train_v1",
        "status": "SMOKE" if smoke else "FORMAL",
        "cell": f"M2-RIFT-R50-D4-CONCAT-LEARNABLE-{recency_cfg.tier.upper()}-V1",
        "seed": SEED,
        "task": "m2",
        "tier": recency_cfg.tier,
        "learnable_config": recency_cfg.__dict__,
        "new_parameter_names": init["new_parameter_names"],
        "new_parameter_counts": init["new_parameter_counts"],
        "trainable_new_parameter_count": init["trainable_new_parameter_count"],
        "identity_interface": "concat",
        "context_bins": CONTEXT,
        "layer_windows": list(recency_cfg.temporal_config.windows),
        "depth": recency_cfg.layers,
        "ladder": recency_cfg.ladder_metadata(),
        "width": 256,
        "attention_backend": "local",
        "batch": BATCH,
        "epochs": EPOCHS,
        "updates_per_epoch": 3165,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": v1_plan.WEIGHT_DECAY,
            "new_param_weight_decay": 0.0,
            "new_param_lr_multiplier": recency_cfg.lr_multiplier,
            "betas": list(old_plan.ADAM_BETAS),
            "eps": old_plan.ADAM_EPS,
            "clip": v1_plan.GRAD_CLIP,
        },
        "lr": {"peak": v1_plan.LR_PEAK, "min": v1_plan.LR_PEAK * v1_plan.LR_MIN_FACTOR, "warmup_updates": 3165},
        "ema_decay": v1_plan.EMA_DECAY,
        "unit_dropout": v1_plan.UNIT_DROPOUT,
        "source_hashes": source_hashes(),
        "frozen_cache_hashes": caches,
        "paired_recency_reference": reference,
        "initialization_pairing": dict(init),
        "runtime": {
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        atom(dest / "run_meta.json", meta)
    step, start_epoch = 0, 1
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if str(state.get("tier")) != recency_cfg.tier:
            raise RuntimeError("learnable resume checkpoint tier drift")
        if int(state.get("seed", -1)) != SEED:
            raise RuntimeError("learnable resume checkpoint seed drift")
        model.load_state_dict(state["raw_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        flat.restore_rng(state["rng"], device)
        step, start_epoch = int(state["global_step"]), int(state["epoch"]) + 1
        if start_epoch > EPOCHS:
            raise RuntimeError("resume checkpoint is already complete")
    started = time.monotonic()
    total = EPOCHS * 3165
    smoke_preupdate_parity = smoke_postupdate_parity = None
    last_bias = None
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(
            old_sampler.iter_manifest_batches(
                train_dual, manifest, epoch, device=device, target_space=old_plan.TRAINING_TARGET_SPACE
            )
        ):
            step += 1
            lr = warmup_cosine_lr(step, total_steps=total, warmup_steps=3165, peak=v1_plan.LR_PEAK, min_factor=v1_plan.LR_MIN_FACTOR)
            apply_group_lrs(optimizer, lr)
            keep_rng = torch.Generator(device="cpu")
            keep_rng.manual_seed(unit_dropout_seed(SEED, epoch, batch_index))
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
                raise FloatingPointError(f"nonfinite learnable loss at epoch={epoch}, batch={batch_index}")
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(model.parameters(), v1_plan.GRAD_CLIP, error_if_nonfinite=True))
            optimizer.step()
            ema.update_after_step(model)
            if smoke and smoke_postupdate_parity is None:
                smoke_postupdate_parity = smoke_full_stream_parity(model, batch, train_banks[batch.session_id], valid)
            losses.append(float(loss.detach().cpu()))
            last_bias = bias_snapshot(model, batch, train_banks[batch.session_id], valid)
            if step == 1 or step % 100 == 0:
                atom(
                    dest / "heartbeat.json",
                    {
                        "status": "TRAINING",
                        "event": "step",
                        "epoch": epoch,
                        "global_step": step,
                        "loss": losses[-1],
                        "lr": lr,
                        "grad_norm": grad,
                        "elapsed_seconds": time.monotonic() - started,
                        "bias": last_bias,
                    },
                )
            if smoke and step >= args.max_updates_smoke:
                break
        score_limit = 1 if smoke else None
        raw = frozen._score(model, mini_dual, mini_banks, pads["source_minival"], "source_minival", device, max_batches_per_session=score_limit)
        ema_score = frozen._with_ema(
            model,
            ema,
            lambda: frozen._score(model, mini_dual, mini_banks, pads["source_minival"], "source_minival", device, max_batches_per_session=score_limit),
        )
        ckpt = dest / f"epoch_{epoch:03d}.pt"
        if ckpt.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {ckpt}")
        torch.save(
            {
                "schema": "m2_rift_concat_learnable_epoch_checkpoint_v1",
                "cell": meta["cell"],
                "epoch": epoch,
                "global_step": step,
                "seed": SEED,
                "tier": recency_cfg.tier,
                "new_parameter_names": init["new_parameter_names"],
                "smoke": smoke,
                "raw_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "rng": flat.rng_state(device),
                "initialization_pairing": dict(init),
            },
            ckpt,
        )
        elapsed = time.monotonic() - started
        epoch_row = {
            "status": "SMOKE" if smoke else "TRAINING",
            "event": "epoch",
            "epoch": epoch,
            "global_step": step,
            "epoch_update_count": len(losses),
            "train_mse": float(np.mean(losses)),
            "ema_updates": ema.n_updates,
            "elapsed_seconds": elapsed,
            "minival_raw": raw,
            "minival_ema": ema_score,
            "bias": last_bias,
        }
        atom(dest / "heartbeat.json", epoch_row)
        append_jsonl(dest / "metrics.jsonl", epoch_row)
        if smoke:
            receipt = {
                "schema": "m2_rift_concat_learnable_smoke_receipt_v1",
                "status": "COMPLETED",
                "cell": meta["cell"],
                "tier": recency_cfg.tier,
                "epoch": epoch,
                "global_step": step,
                "finite_train_mse": bool(math.isfinite(float(np.mean(losses)))),
                "finite_loss": bool(math.isfinite(float(np.mean(losses)))),
                "checkpoint": str(ckpt),
                "runtime_seconds": elapsed,
                "full_stream_parity_preupdate": smoke_preupdate_parity,
                "full_stream_parity_postupdate": smoke_postupdate_parity,
                "bias": last_bias,
                "new_parameter_names": init["new_parameter_names"],
                "new_parameter_counts": init["new_parameter_counts"],
                "trainable_new_parameter_count": init["trainable_new_parameter_count"],
                "initialization_pairing": init,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
            atom(dest / "smoke_receipt.json", receipt)
            return receipt
        if len(losses) != 3165 or step != epoch * 3165 or ema.n_updates != step:
            raise RuntimeError("formal learnable epoch update/EMA accounting drift")
    receipt = {
        "schema": "m2_rift_concat_learnable_train_receipt_v1",
        "status": "COMPLETED",
        "cell": meta["cell"],
        "tier": recency_cfg.tier,
        "epochs": EPOCHS,
        "global_step": step,
        "runtime_seconds": time.monotonic() - started,
        "source_hashes": meta["source_hashes"],
        "frozen_cache_hashes": caches,
        "reference_binding": reference,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atom(dest / "train_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_learnable_flags(parser)
    parser.add_argument("--config", default="m2")
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates-smoke", type=int)
    args = parser.parse_args()
    if args.config != "m2":
        parser.error("this runner is the M2 config")
    if args.max_updates_smoke is not None and args.max_updates_smoke < 1:
        parser.error("smoke counts must be positive")
    if args.dest is None:
        name = f"m2_{args.tier}_s42"
        if args.max_updates_smoke is not None:
            extra = "" if int(args.layers) == 4 else f"_layers{int(args.layers)}"
            args.dest = RESULTS / "smoke" / f"{name}{extra}_v2"
        else:
            args.dest = RESULTS / name
    if args.max_updates_smoke is not None and args.device == "cuda:0":
        args.device = "cpu"
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
