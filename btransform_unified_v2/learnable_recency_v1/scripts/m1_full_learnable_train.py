#!/usr/bin/env python3
"""M1 FULL muscle learnable-recency trainer. Mirrors m1_full_flat_train.py; does not edit it."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
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
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
WS = ROOT.parent
M1_FLAT = ROOT / "scripts/recency_flat_ablation_v1/m1_full_flat_train.py"
FLAT_DIR = M1_FLAT.parent
FROZEN_DIR = ROOT / "scripts/m1_muscle_r100_v1"
RESULTS = PKG / "results"

for candidate in (PKG / "src", FLAT_DIR, FROZEN_DIR, ROOT / "src", WS / "btransform_unified_v1/src", WS / "btransform_unified_v1/scripts", WS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

spec = importlib.util.spec_from_file_location("_m1_full_flat_learnable", M1_FLAT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot privately load m1_full_flat_train.py")
m1_flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1_flat)
frozen = m1_flat.frozen

from btransform_unified_v1 import plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v2.joint_m1_model import ARMS, JointM1ConcatDecoder
from learnable_recency_v1.config import add_learnable_flags, config_from_args, config_from_run_meta, dataset_config
from learnable_recency_v1.wrap import (
    LearnableJointM1ConcatDecoder,
    LearnableRiftStreamDecoder,
    apply_group_lrs,
    assert_shared_byte_equal,
    install_temporal,
    new_parameter_names,
    recency_bias_snapshot,
    split_optimizer_parameters,
    trainable_new_parameter_count,
)

SEED = 42
CONTEXT, EPOCHS, BATCH, LR = frozen.CONTEXT, frozen.EPOCHS, frozen.BATCH, frozen.LR
UPDATES_PER_EPOCH = frozen.UPDATES_PER_EPOCH
HO = frozen.HO
SOURCE = frozen.SOURCE_SESSIONS
FROZEN_SAMPLER_SEED = frozen.SEED


def _sha_file(path: Path) -> str:
    return frozen._sha_file(path)


def _atomic_json(path: Path, payload: Any) -> None:
    frozen._atomic_json(path, payload)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    return frozen._atomic_checkpoint(path, payload)


def shared_init_sha(model: nn.Module, names: list[str]) -> str:
    digest = hashlib.sha256()
    named = dict(model.named_parameters())
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(named[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def learnable_decoder(device: torch.device, arm: str, recency_cfg) -> LearnableJointM1ConcatDecoder:
    model = LearnableJointM1ConcatDecoder(arm, recency_cfg, seed=SEED).to(device)
    model.temporal.set_attention_backend("local")
    expected = tuple(recency_cfg.temporal_config.windows)
    if tuple(model.temporal_config.windows) != expected:
        raise RuntimeError(f"M1 R100 D{recency_cfg.layers} windows drifted: {tuple(model.temporal_config.windows)} != {expected}")
    return model


def assert_paired(model: LearnableJointM1ConcatDecoder, device: torch.device, paired: Mapping[str, Any], recency_cfg) -> dict[str, Any]:
    recency = JointM1ConcatDecoder(model.arm, seed=SEED).to(device)
    if recency_cfg.layers != 4:
        shared_cfg = dataset_config("m1", tier="fixed", layers=recency_cfg.layers, ladder="default")
        install_temporal(recency, shared_cfg, SEED)
    recency.temporal.set_attention_backend("local")
    shared = assert_shared_byte_equal(model, recency)
    extra = new_parameter_names(model)
    hashed = shared_init_sha(model, shared)
    if recency_cfg.layers == 4 and hashed != paired["initialization_named_parameters_sha256"]:
        raise RuntimeError("learnable shared named-parameter SHA differs from paired recency reference")
    ladder_cfg = dataset_config(
        "m1", tier="fixed", layers=recency_cfg.layers, half_life_seconds=recency_cfg.half_life_seconds
    )
    ladder_ref = JointM1ConcatDecoder(model.arm, seed=SEED).to(device)
    install_temporal(ladder_ref, ladder_cfg, SEED)
    ladder_ref.temporal.set_attention_backend("local")
    z = torch.randn(1, CONTEXT, 256, device=device)
    mask = torch.ones(1, CONTEXT, dtype=torch.bool, device=device)
    with torch.inference_mode():
        left = model.temporal(z, mask)
        right = ladder_ref.temporal(z, mask)
    if not torch.allclose(left, right, atol=1e-6, rtol=1e-6):
        raise RuntimeError("M1 learnable/same-ladder init forward drift")
    del recency, ladder_ref
    return {
        "shared_parameter_names": shared,
        "new_parameter_names": extra,
        "new_parameter_counts": {name: int(dict(model.named_parameters())[name].numel()) for name in extra},
        "trainable_new_parameter_count": trainable_new_parameter_count(model),
        "shared_initialization_sha256": hashed,
        "init_forward_max_abs": float((left - right).abs().max().cpu()),
        "ladder": recency_cfg.ladder_metadata(),
    }


def stream_parity(model: nn.Module, material: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    session = HO[0]
    bank = material[session]["bank"]
    x = torch.from_numpy(np.asarray(material[session]["X"][:1], dtype=np.float32)).to(device)
    if x.ndim != 3:
        x = torch.from_numpy(np.asarray(next(iter(material[session].values()))[:1])).to(device)
    # Frozen helper builds the same HO windows; reuse its first scored prefix when present.
    try:
        return _stream_parity_from_frozen_shape(model, material, device)
    except Exception:
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
        return _stream_one(model, x, bank, valid)


def _stream_parity_from_frozen_shape(model: nn.Module, material: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    report = frozen._assert_full_stream_parity(model, material, device)
    return report


def _stream_one(model: nn.Module, x: torch.Tensor, bank: Any, valid: torch.Tensor) -> dict[str, Any]:
    was = model.training
    model.eval()
    with torch.inference_mode():
        full = model(x, bank, input_valid_mask=valid)
        stream = LearnableRiftStreamDecoder(model)
        streamed = None
        for offset in range(x.shape[1]):
            streamed = stream.stream_step(x[:, offset], bank, ["m1-learnable-smoke"], valid_mask=valid[:, offset])
    model.train(was)
    if streamed is None or not torch.allclose(full, streamed, rtol=2e-5, atol=2e-5):
        raise RuntimeError("learnable M1 full/stream parity failed")
    return {"passed": True, "max_abs": float((full - streamed).abs().max()), "bins": int(x.shape[1])}


def bias_snapshot(model: nn.Module, x: torch.Tensor, bank: Any, valid: torch.Tensor) -> dict[str, Any]:
    was = model.training
    model.eval()
    with torch.inference_mode():
        tokens = model.frontend_tokens(x[:1], bank)
        stats = recency_bias_snapshot(model.temporal, tokens, valid[:1])
    model.train(was)
    return stats


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    recency_cfg = config_from_args(args, "m1")
    smoke = args.max_updates_smoke is not None
    if args.seed != 42 or (not smoke and args.epochs != EPOCHS):
        raise ValueError("learnable formal run requires seed42 and exactly 24 epochs")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    dest = args.dest.resolve()
    if args.resume is None and dest.exists() and any(dest.iterdir()):
        raise FileExistsError("new learnable destination must be empty")
    carriers, binding = m1_flat._carrier_binding(args.carrier_pack)
    paired = m1_flat._paired_reference(args.paired_reference, arm=args.arm, binding=binding)
    dataset, sampler = frozen.legacy.build_fullsession_face()
    legacy_banks, legacy_report = frozen.legacy.build_fullsession_banks(dataset)
    banks = frozen._replace_carriers(legacy_banks, carriers, SOURCE)
    contract = frozen._source_contract(
        dataset,
        sampler,
        banks,
        {
            "legacy_frozen_activity_bank_report": legacy_report,
            "carrier_replacement": binding["carrier_variant"],
            "actual_carrier_metadata": "source_contract.bank_hashes + carrier_binding",
        },
    )
    source_calib = frozen.m1_plan.calib_trials_from_dataset(dataset)
    contract["raw_m10_calib_sha256"] = {name: frozen._array_sha(value) for name, value in source_calib.items()}
    if contract != paired["source_contract"]:
        raise RuntimeError("learnable source contract differs from paired recency reference")
    model = learnable_decoder(device, args.arm, recency_cfg)
    model.install_session_memory(banks, source_calib)
    model.to(device)
    init = assert_paired(model, device, paired, recency_cfg)
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    groups = split_optimizer_parameters(model, peak_lr=LR, weight_decay=plan.WEIGHT_DECAY, lr_multiplier=recency_cfg.lr_multiplier)
    optimizer = torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.999), eps=1e-8)
    smoke_preflight = None
    material = None
    if smoke:
        material = frozen._ho_material(carriers)
        ho = frozen._ho_contract(material)
        if ho != paired["ho_contract"]:
            raise RuntimeError("learnable HO contract differs from paired recency reference")
        model.install_session_memory({s: material[s]["bank"] for s in HO}, {s: material[s]["calib10"] for s in HO})
        model.to(device)
        smoke_preflight = {
            "initialization_pairing": init,
            "streaming_parity": _stream_one(
                model,
                torch.ones(1, CONTEXT, model.units, device=device),
                material[HO[0]]["bank"],
                torch.ones(1, CONTEXT, dtype=torch.bool, device=device),
            ),
        }
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "m1_muscle_r100_full_learnable_train_v1",
        "status": "SMOKE" if smoke else "FORMAL",
        "cell": f"M1-MUSCLE-R100-D4-JOINT-LEARNABLE-{recency_cfg.tier.upper()}-V1",
        "task": "m1",
        "arm": args.arm,
        "tier": recency_cfg.tier,
        "learnable_config": recency_cfg.__dict__,
        "new_parameter_names": init["new_parameter_names"],
        "new_parameter_counts": init["new_parameter_counts"],
        "trainable_new_parameter_count": init["trainable_new_parameter_count"],
        "seed": args.seed,
        "sampler_seed": FROZEN_SAMPLER_SEED,
        "context_bins": CONTEXT,
        "layer_windows": list(recency_cfg.temporal_config.windows),
        "depth": recency_cfg.layers,
        "ladder": recency_cfg.ladder_metadata(),
        "batch": BATCH,
        "epochs": args.epochs,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "optimizer": {"name": "AdamW", "weight_decay": plan.WEIGHT_DECAY, "new_param_weight_decay": 0.0, "clip": 1.0},
        "lr": {"peak": LR, "min": LR * plan.LR_MIN_FACTOR, "warmup_updates": UPDATES_PER_EPOCH, "new_param_multiplier": recency_cfg.lr_multiplier},
        "ema_decay": plan.EMA_DECAY,
        "unit_dropout": 0.1,
        "initialization_pairing": init,
        "source_contract": dict(contract),
        "carrier_binding": dict(binding),
        "paired_reference": dict(paired),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.resume is None:
        _atomic_json(dest / "run_meta.json", meta)
        step, first_epoch = 0, 1
    else:
        raise RuntimeError("M1 learnable smoke/formal resume is not used in this drop")
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=frozen.legacy._collate, num_workers=0)
    started = time.monotonic()
    last_bias = None
    last_loss = None
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_id, (x, y, sessions) in enumerate(loader):
            if any(session != sessions[0] for session in sessions):
                raise RuntimeError("M1 source sampler produced mixed-session batch")
            step += 1
            lr = frozen.warmup_cosine_lr(
                step, total_steps=EPOCHS * UPDATES_PER_EPOCH, warmup_steps=UPDATES_PER_EPOCH, peak=LR, min_factor=plan.LR_MIN_FACTOR
            )
            apply_group_lrs(optimizer, lr)
            keep_rng = torch.Generator(device="cpu")
            keep_rng.manual_seed(unit_dropout_seed(42, epoch, batch_id))
            keep = whole_unit_dropout(banks[sessions[0]].unit_mask, p=0.1, generator=keep_rng)
            valid = torch.ones((len(x), CONTEXT), dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                prediction = model(x.float().to(device), banks[sessions[0]], dropout_keep=keep, input_valid_mask=valid)
                loss = nn.functional.mse_loss(prediction.float(), y.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite learnable loss epoch={epoch} batch={batch_id}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            ema.update_after_step(model)
            losses.append(float(loss.detach().cpu()))
            last_loss = losses[-1]
            last_bias = bias_snapshot(model, x.float().to(device), banks[sessions[0]], valid)
            if smoke and step >= args.max_updates_smoke:
                break
        postupdate = None
        if smoke and step >= args.max_updates_smoke and material is not None:
            postupdate = {
                "full_vs_stream": _stream_one(
                    model,
                    torch.ones(1, CONTEXT, model.units, device=device),
                    material[HO[0]]["bank"],
                    torch.ones(1, CONTEXT, dtype=torch.bool, device=device),
                )
            }
        checkpoint = {
            "schema": "m1_muscle_r100_full_learnable_epoch_checkpoint_v1",
            "epoch": epoch,
            "global_step": step,
            "smoke": smoke,
            "tier": recency_cfg.tier,
            "new_parameter_names": init["new_parameter_names"],
            "raw_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "rng": frozen._rng_state(device),
        }
        checkpoint_sha = _atomic_checkpoint(dest / f"epoch_{epoch:03d}.pt", checkpoint)
        row = {
            "status": "SMOKE" if smoke else "TRAINING",
            "event": "epoch",
            "epoch": epoch,
            "global_step": step,
            "train_mse": float(np.mean(losses)),
            "checkpoint_sha256": checkpoint_sha,
            "bias": last_bias,
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(dest / "heartbeat.json", row)
        frozen._append_jsonl(dest / "metrics.jsonl", row)
        if smoke:
            receipt = {
                "schema": meta["schema"],
                "status": "COMPLETED",
                "tier": recency_cfg.tier,
                "steps": step,
                "finite_loss": last_loss is not None and bool(np.isfinite(last_loss)),
                "preflight": smoke_preflight,
                "postupdate": postupdate,
                "bias": last_bias,
                "new_parameter_names": init["new_parameter_names"],
                "new_parameter_counts": init["new_parameter_counts"],
                "trainable_new_parameter_count": init["trainable_new_parameter_count"],
            }
            _atomic_json(dest / "smoke_receipt.json", receipt)
            return receipt
        if len(losses) != UPDATES_PER_EPOCH:
            raise RuntimeError(f"epoch {epoch} had {len(losses)} updates, expected {UPDATES_PER_EPOCH}")
    _atomic_json(
        dest / "train_receipt.json",
        {
            "schema": meta["schema"],
            "status": "COMPLETED",
            "tier": recency_cfg.tier,
            "epochs": EPOCHS,
            "steps": step,
            "paired_reference": meta["paired_reference"],
            "post_training_scoring_required": True,
        },
    )
    return {"status": "TRAIN_COMPLETED", "steps": step}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    dest = args.dest.resolve()
    meta = frozen._read_json(dest / "run_meta.json")
    recency_cfg = config_from_run_meta(meta, "m1")
    carriers, binding = m1_flat._carrier_binding(args.carrier_pack)
    paired = m1_flat._paired_reference(args.paired_reference, arm=args.arm, binding=binding)
    receipt = frozen._read_json(dest / "train_receipt.json")
    if receipt.get("status") != "COMPLETED":
        raise RuntimeError("score requires completed formal train receipt")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    material = frozen._ho_material(carriers)
    model = learnable_decoder(device, args.arm, recency_cfg)
    model.install_session_memory({s: material[s]["bank"] for s in HO}, {s: material[s]["calib10"] for s in HO})
    model.to(device)
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    completed = {}
    for epoch in range(1, EPOCHS + 1):
        path = dest / f"epoch_{epoch:03d}.pt"
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["raw_state_dict"], strict=True)
        ema.load_state_dict(state["ema"])
        report = frozen._with_ema_eval(model, ema, lambda: frozen._ho_score(model, material, device, capture_predictions=False))
        frozen._validate_scored_report(report)
        completed[str(epoch)] = report
    best = min(
        range(1, EPOCHS + 1),
        key=lambda epoch: (-completed[str(epoch)]["equal_session_mean_channel_variance_weighted_r2"], epoch),
    )
    result = {
        "schema": "m1_muscle_r100_full_learnable_ho_calib_epoch_scan_v1",
        "status": "COMPLETED",
        "tier": args.tier,
        "ema_by_epoch": completed,
        "selection": {
            "epoch": best,
            "metric": "channel_variance_weighted_r2",
            "rule": "earliest maximum equal-session mean EMA on visible HO3 calibration",
        },
        "paired_reference": dict(paired),
        "official_test_used": False,
    }
    _atomic_json(dest / "score_receipt.json", result)
    return {"status": "SCORE_COMPLETED", "best_epoch": best}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_learnable_flags(parser)
    parser.add_argument("--config", default="m1")
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--arm", choices=ARMS, default="D_JOINT")
    parser.add_argument("--stage", choices=("train", "score"), default="train")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--carrier-pack", type=Path, default=m1_flat.DEFAULT_PACK)
    parser.add_argument("--paired-reference", type=Path, default=m1_flat.DEFAULT_REFERENCE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates-smoke", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.config != "m1":
        parser.error("this runner is the M1 config")
    if args.dest is None:
        name = f"m1_{args.tier}_s42"
        if args.max_updates_smoke is not None:
            extra = "" if int(args.layers) == 4 else f"_layers{int(args.layers)}"
            args.dest = RESULTS / "smoke" / f"{name}{extra}_v2"
        else:
            args.dest = RESULTS / name
    if args.max_updates_smoke is not None and args.device == "cuda:0":
        args.device = "cpu"
    print(json.dumps(run_train(args) if args.stage == "train" else run_score(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
