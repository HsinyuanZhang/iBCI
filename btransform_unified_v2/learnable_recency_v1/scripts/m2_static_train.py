#!/usr/bin/env python3
"""Source-only, from-scratch static-identity M2 R50/P16 trainer.

This route intentionally reads only query spike/target windows and their
eligible-start metadata.  It never constructs a calibration bank: the model
owns a trainable ``static_identity[N,16]`` and has no per-session carrier.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
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
WS = ROOT.parent
V1 = WS / "btransform_unified_v1"
for p in (PKG / "src", ROOT, ROOT / "src", V1 / "src", V1 / "scripts", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
from btransform_unified_v1.r2 import variance_weighted_r2
from learnable_recency_v1.config import add_learnable_flags, config_from_args
from learnable_recency_v1.static_model import StaticLearnableRiftDecoder
from learnable_recency_v1.wrap import apply_group_lrs, split_optimizer_parameters
from scripts.rift_v1 import m2_train as frozen
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan

SEED, CONTEXT, EPOCHS, BATCH, PROJ_DIM = 42, 50, 24, 32, 16
UPDATES = 3165
CACHE = WS / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"
MANIFEST = (
    WS
    / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/sampler/shuffled_batch_manifest_24.json"
)
SCHEMA = "m2_static_train_v1"
CKPT_SCHEMA = "m2_static_epoch_checkpoint_v1"
RESULTS = PKG / "results"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atom(p: Path, v: Mapping[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(v, indent=2, sort_keys=True, default=str) + "\n")
    q.replace(p)


def append(p: Path, v: Mapping[str, Any]) -> None:
    with p.open("a", encoding="utf8") as f:
        f.write(json.dumps(v, sort_keys=True, default=str) + "\n")


def _array(path: Path, dtype=None):
    a = np.load(path, mmap_mode="r")
    if dtype is not None and a.dtype != dtype:
        raise RuntimeError(f"{path}: dtype drift {a.dtype}")
    return a


def load_static_surface(
    surface: str, *, cache_root: Path = CACHE
) -> dict[str, dict[str, Any]]:
    """Direct query-only reader.  Deliberately does not open E0/T/calibration."""
    root = cache_root / surface
    if not root.is_dir():
        raise FileNotFoundError(root)
    out = {}
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        x, t, s = (
            _array(d / "X_store.npy", np.float32),
            _array(d / "target_store.npy", np.float32),
            _array(d / "eligible_starts.npy", np.int64),
        )
        if (
            x.ndim != 2
            or x.shape[1] != 96
            or s.ndim != 1
            or len(s) == 0
            or t.shape != (len(s), 2)
            or np.any(np.diff(s) < 0)
        ):
            raise RuntimeError(f"{d}: static query shape drift")
        mapping = json.loads((d / "mapping.json").read_text())
        pad = int(mapping.get("query_pad_bins", 0))
        if pad != 49 or int(s[0]) < 0 or int(s[-1]) + CONTEXT > len(x):
            raise RuntimeError(f"{d}: static padded-start bounds drift")
        out[d.name] = {
            "X": x,
            "Y": t,
            "starts": s,
            "pad": pad,
            "dir": d,
            "hashes": {
                n: sha(d / n)
                for n in (
                    "X_store.npy",
                    "target_store.npy",
                    "eligible_starts.npy",
                    "mapping.json",
                )
            },
        }
    if not out:
        raise RuntimeError(f"{surface}: no sessions")
    return out


def windows(item: Mapping[str, Any], indices: np.ndarray, device: torch.device):
    starts = np.asarray(item["starts"])[indices]
    x = np.asarray(item["X"])
    raw = np.stack([x[int(s) : int(s) + CONTEXT] for s in starts]).astype(
        np.float32, copy=False
    )
    if raw.shape[1:] != (CONTEXT, 96):
        raise RuntimeError("window shape drift")
    valid = torch.as_tensor(
        starts[:, None] + np.arange(CONTEXT)[None, :] >= int(item["pad"]), device=device
    )
    return (
        torch.as_tensor(raw, device=device),
        torch.as_tensor(
            np.asarray(item["Y"])[indices] * old_plan.BEHAVIOR_SCALE, device=device
        ),
        valid,
    )


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        PKG / "src/learnable_recency_v1/static_model.py",
        PKG / "src/learnable_recency_v1/config.py",
        PKG / "src/learnable_recency_v1/temporal.py",
        PKG / "src/learnable_recency_v1/wrap.py",
        ROOT / "src/btransform_unified_v2/model.py",
        ROOT / "src/btransform_unified_v2/temporal.py",
        V1 / "src/btransform_unified_v1/ema.py",
        V1 / "src/btransform_unified_v1/model.py",
        V1 / "src/btransform_unified_v1/schedule.py",
        MANIFEST,
    ]
    return {str(p): sha(p) for p in paths}


def _rng(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def _restore(s, device):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch_cpu"].cpu())
    if device.type == "cuda" and s.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in s["torch_cuda"]])


def score_surface(model, surface, device, *, limit=None):
    if limit is not None and limit < 1:
        raise ValueError("score batch limit must be positive")
    model.eval()
    rows = {}
    ps = []
    ts = []
    for name, item in surface.items():
        pred = []
        target = []
        for i in range(0, len(item["starts"]), BATCH):
            if limit is not None and i // BATCH >= limit:
                break
            ids = np.arange(i, min(i + BATCH, len(item["starts"])))
            x, y, v = windows(item, ids, device)
            with torch.inference_mode():
                p = (
                    model(x, input_valid_mask=v).float().cpu().numpy()
                    / old_plan.BEHAVIOR_SCALE
                )
            # Native query Y is deliberately retained for metrics.  Dividing
            # training-space float32 targets could introduce avoidable rounding.
            target.append(np.asarray(item["Y"])[ids])
            pred.append(p)
        p = np.concatenate(pred)
        t = np.concatenate(target)
        if not np.isfinite(p).all() or not np.isfinite(t).all():
            raise FloatingPointError(f"{name}: non-finite prediction or target")
        r2 = float(variance_weighted_r2(t, p))
        if not math.isfinite(r2):
            raise FloatingPointError(f"{name}: non-finite R2")
        rows[name] = {
            "r2": r2,
            "window_count": len(t),
            "prediction_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
        }
        ps.append(p)
        ts.append(t)
    pooled_r2 = float(variance_weighted_r2(np.concatenate(ts), np.concatenate(ps)))
    equal_mean = float(np.mean([x["r2"] for x in rows.values()]))
    if not math.isfinite(pooled_r2) or not math.isfinite(equal_mean):
        raise FloatingPointError("non-finite aggregate R2")
    return {
        "per_session": rows,
        "equal_session_mean": equal_mean,
        "pooled_r2": pooled_r2,
        "n_windows": int(sum(x["window_count"] for x in rows.values())),
        "partial": limit is not None,
    }


def recipe(args):
    fixed = (
        args.seed == SEED
        and args.proj_dim == PROJ_DIM
        and args.tier == "learned_slope"
        and args.layers == 4
        and args.ladder == "default"
        and args.per_layer
        and not args.learn_flat_heads
        and not args.cable_nw
        and args.lr_multiplier == 1.0
        and args.half_lives is None
    )
    if not fixed:
        raise ValueError(
            "static recipe is seed42/P16/D4 learned_slope/default with per-layer slopes, default half-lives, no flat/CABLE heads, and LR multiplier 1"
        )
    if not args.max_updates_smoke and args.epochs != EPOCHS:
        raise ValueError("formal static run requires 24 epochs")


def run(args):
    recipe(args)
    smoke = args.max_updates_smoke is not None
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    dest = args.dest.resolve()
    if args.resume is None and dest.exists() and any(dest.iterdir()):
        raise FileExistsError("fresh destination must be empty")
    if args.resume is not None and args.resume.resolve().parent != dest:
        raise RuntimeError("resume checkpoint must be directly in --dest")
    manifest = frozen.old_sampler.load_manifest(MANIFEST)
    if (
        manifest.get("digest") != frozen.MANIFEST_DIGEST
        or int(manifest.get("batch_size", 0)) != BATCH
    ):
        raise RuntimeError("frozen source manifest drift")
    train = load_static_surface("source_train")
    mini = load_static_surface("source_minival")
    expected_sessions = set(manifest["sessions"])
    source_lengths = {name: len(item["starts"]) for name, item in train.items()}
    if (
        set(train) != expected_sessions
        or len(manifest["batches"]["1"]) != UPDATES
        or source_lengths != manifest["lengths"]
    ):
        raise RuntimeError("source roster or update budget drift")
    cfg = config_from_args(args, "m2")
    model = StaticLearnableRiftDecoder("m2", cfg, context_bins=CONTEXT, seed=SEED).to(
        device
    )
    model.temporal.set_attention_backend("local")
    if tuple(model.temporal_config.windows) != (13, 12, 12, 12) or tuple(
        model.static_identity.shape
    ) != (96, 16):
        raise RuntimeError("static architecture drift")
    ema = DecoderEMA(model, decay=frozen.v1_plan.EMA_DECAY)
    groups = split_optimizer_parameters(
        model,
        peak_lr=frozen.v1_plan.LR_PEAK,
        weight_decay=frozen.v1_plan.WEIGHT_DECAY,
        lr_multiplier=cfg.lr_multiplier,
    )
    opt = torch.optim.AdamW(
        groups,
        lr=frozen.v1_plan.LR_PEAK,
        betas=old_plan.ADAM_BETAS,
        eps=old_plan.ADAM_EPS,
    )
    step = 0
    start = 1
    before = model.static_identity.detach().clone()
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": SCHEMA,
        "status": "SMOKE" if smoke else "FORMAL",
        "cell": "M2-STATIC-R50-D4-P16-LEARNED-SLOPE-V1",
        "seed": SEED,
        "task": "m2",
        "identity_interface": "static_identity",
        "no_target_support_loading": True,
        "static_carrier": "zero internal",
        "fixedindex_assumption": "96 real unit columns; all-true mask; query padding is metadata mask (49)",
        "selection": {
            "rule": "EXT6 all24 earliest maximum equal_session_mean (same as FULL)",
            "source_only": True,
            "fixedfinal": False,
        },
        "learnable_config": cfg.__dict__,
        "context_bins": CONTEXT,
        "proj_dim": PROJ_DIM,
        "depth": 4,
        "layer_windows": list(model.temporal_config.windows),
        "batch": BATCH,
        "epochs": EPOCHS,
        "updates_per_epoch": UPDATES,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": frozen.v1_plan.WEIGHT_DECAY,
            "betas": list(old_plan.ADAM_BETAS),
            "eps": old_plan.ADAM_EPS,
            "clip": frozen.v1_plan.GRAD_CLIP,
        },
        "lr": {
            "peak": frozen.v1_plan.LR_PEAK,
            "min": frozen.v1_plan.LR_PEAK * frozen.v1_plan.LR_MIN_FACTOR,
            "warmup_updates": UPDATES,
        },
        "ema_decay": frozen.v1_plan.EMA_DECAY,
        "unit_dropout": frozen.v1_plan.UNIT_DROPOUT,
        "source_hashes": source_hashes(),
        "data_hashes": {
            s: {n: v["hashes"] for n, v in data.items()}
            for s, data in {"source_train": train, "source_minival": mini}.items()
        },
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.resume:
        existing = json.loads((dest / "run_meta.json").read_text())
        for key in (
            "schema",
            "seed",
            "context_bins",
            "proj_dim",
            "source_hashes",
            "data_hashes",
        ):
            if existing.get(key) != meta.get(key):
                raise RuntimeError(f"resume metadata mismatch: {key}")
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if (
            state.get("schema") != CKPT_SCHEMA
            or bool(state.get("smoke")) != smoke
            or int(state.get("seed", -1)) != SEED
        ):
            raise RuntimeError("resume checkpoint contract mismatch")
        step = int(state["global_step"])
        epoch_done = int(state["epoch"])
        if step != epoch_done * UPDATES or epoch_done < 1 or epoch_done >= EPOCHS:
            raise RuntimeError("resume checkpoint epoch/step boundary drift")
        model.load_state_dict(state["raw_state_dict"])
        opt.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        _restore(state["rng"], device)
        start = epoch_done + 1
    else:
        atom(dest / "run_meta.json", meta)
    started = time.monotonic()
    total = EPOCHS * UPDATES
    for epoch in range(start, EPOCHS + 1):
        model.train()
        losses = []
        for bi, row in enumerate(manifest["batches"][str(epoch)]):
            item = train[row["session"]]
            ids = np.asarray(row["indices"], dtype=np.int64)
            x, y, v = windows(item, ids, device)
            step += 1
            lr = warmup_cosine_lr(
                step,
                total_steps=total,
                warmup_steps=UPDATES,
                peak=frozen.v1_plan.LR_PEAK,
                min_factor=frozen.v1_plan.LR_MIN_FACTOR,
            )
            apply_group_lrs(opt, lr)
            g = torch.Generator(device="cpu")
            g.manual_seed(unit_dropout_seed(SEED, epoch, bi))
            keep = whole_unit_dropout(
                torch.ones(96, dtype=torch.bool),
                p=frozen.v1_plan.UNIT_DROPOUT,
                generator=g,
            )
            opt.zero_grad(set_to_none=True)
            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            ):
                loss = nn.functional.mse_loss(
                    model(x, dropout_keep=keep, input_valid_mask=v).float(), y
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite static loss")
            loss.backward()
            grad = float(
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    frozen.v1_plan.GRAD_CLIP,
                    error_if_nonfinite=True,
                )
            )
            opt.step()
            ema.update_after_step(model)
            losses.append(float(loss.detach().cpu()))
            if smoke and step >= args.max_updates_smoke:
                break
        raw = score_surface(model, mini, device, limit=1 if smoke else None)
        ema_score = frozen._with_ema(
            model,
            ema,
            lambda: score_surface(model, mini, device, limit=1 if smoke else None),
        )
        ckpt = dest / f"epoch_{epoch:03d}.pt"
        if ckpt.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {ckpt}")
        torch.save(
            {
                "schema": CKPT_SCHEMA,
                "epoch": epoch,
                "global_step": step,
                "seed": SEED,
                "smoke": smoke,
                "raw_state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "ema": ema.state_dict(),
                "rng": _rng(device),
                "static_identity_shape": [96, 16],
            },
            ckpt,
        )
        row = {
            "status": "SMOKE" if smoke else "TRAINING",
            "epoch": epoch,
            "global_step": step,
            "epoch_update_count": len(losses),
            "train_mse": float(np.mean(losses)),
            "ema_updates": ema.n_updates,
            "minival_raw": raw,
            "minival_ema": ema_score,
        }
        append(dest / "metrics.jsonl", row)
        atom(dest / "heartbeat.json", row)
        if smoke:
            restored = StaticLearnableRiftDecoder(
                "m2", cfg, context_bins=CONTEXT, seed=SEED
            ).to(device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            restored.load_state_dict(state["raw_state_dict"])
            x, y, v = windows(next(iter(train.values())), np.arange(1), device)
            p1 = model.eval()(x, input_valid_mask=v)
            p2 = restored.eval()(x, input_valid_mask=v)
            finite = bool(math.isfinite(row["train_mse"]))
            changed = bool(not torch.equal(before, model.static_identity.detach()))
            parity = bool(torch.equal(p1, p2))
            if not (finite and changed and parity):
                raise RuntimeError("static smoke receipt validation failed")
            receipt = {
                "schema": "m2_static_smoke_receipt_v1",
                "status": "COMPLETED",
                "formal_claim": False,
                "artifact_scope": "isolated smoke destination",
                "finite_loss": finite,
                "trainableidentity_updated": changed,
                "checkpoint_parity": parity,
                "checkpoint": str(ckpt),
                "global_step": step,
                "runtime_seconds": time.monotonic() - started,
            }
            atom(dest / "smoke_receipt.json", receipt)
            return receipt
        if len(losses) != UPDATES or step != epoch * UPDATES or ema.n_updates != step:
            raise RuntimeError("formal update/EMA accounting drift")
    receipt = {
        "schema": "m2_static_train_receipt_v1",
        "status": "COMPLETED",
        "selection": {
            "checkpoint": "epoch_024.pt",
            "view": "EMA",
            "rule": "pending EXT6 earliest-max pick; source_only",
        },
        "epochs": EPOCHS,
        "global_step": step,
        "runtime_seconds": time.monotonic() - started,
    }
    atom(dest / "train_receipt.json", receipt)
    return receipt


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_learnable_flags(p)
    p.set_defaults(ladder="default")
    p.add_argument("--config", default="m2")
    p.add_argument("--dest", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--proj-dim", type=int, default=PROJ_DIM)
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-updates-smoke", type=int)
    return p


def main():
    a = build_parser().parse_args()
    if a.config != "m2":
        raise ValueError("M2 only")
    if a.max_updates_smoke is not None and a.max_updates_smoke < 1:
        raise ValueError("positive smoke updates required")
    if a.dest is None:
        a.dest = RESULTS / (
            "smoke/m2_static_learned_slope_s42"
            if a.max_updates_smoke
            else "m2_static_learned_slope_s42"
        )
    if a.max_updates_smoke is not None and a.device == "cuda:0":
        a.device = "cpu"
    print(json.dumps(run(a), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
