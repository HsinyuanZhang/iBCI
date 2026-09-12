#!/usr/bin/env python3
"""Source-only STATIC M1 R100/D4/P16 training and HO3 scoring.

The route consumes raw M1 query windows only.  It has no E0, support set, or
session bank: ``StaticLearnableRiftDecoder`` learns its 64-by-16 identity in
the decoder.  Formal runs are exactly 24 fixed sampler epochs.  ``--stage
score`` reports the last-epoch EMA; ``--stage pick`` uses the same HO3
earliest-max equal-session mean rule as FULL.
"""

from __future__ import annotations

import argparse, contextlib, hashlib, json, math, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PKG, ROOT, WS = HERE.parent, HERE.parent.parent, HERE.parent.parent.parent
V1 = WS / "btransform_unified_v1"
for p in (PKG / "src", ROOT / "src", V1 / "src", WS, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v1.schedule import warmup_cosine_lr
from learnable_recency_v1.config import (
    add_learnable_flags,
    config_from_args,
    config_from_run_meta,
)
from learnable_recency_v1.static_model import (
    StaticLearnableRiftDecoder,
    StaticRiftStreamDecoder,
)
from m1_static_data import batch_tensors, load_heldout, load_source
from tfpd_exploration.src.m2_dual_track_v1.training import adamw_param_groups

SEED, CONTEXT, PROJ_DIM, EPOCHS, BATCH, UPDATES = 42, 100, 16, 24, 32, 6665
EMA_DECAY, WEIGHT_DECAY, UNIT_DROPOUT, LR, LR_MIN, CLIP = (
    0.9995,
    0.01,
    0.1,
    1e-4,
    1e-5,
    1.0,
)
RESULTS = PKG / "results"
SCHEMA, CKPT_SCHEMA, SCORE_SCHEMA, PICK_SCHEMA = (
    "m1_static_train_v1",
    "m1_static_epoch_checkpoint_v1",
    "m1_static_ho3_final_ema_score_v1",
    "m1_static_ho3_earliest_max_score_v1",
)
SELECTION_RULE = "HO3 all24 earliest maximum equal_session_mean (same as FULL)"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atom(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def append(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def atomic_checkpoint(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), tmp)
    tmp.replace(path)
    return sha(path)


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        PKG / "src/learnable_recency_v1/static_model.py",
        PKG / "src/learnable_recency_v1/temporal.py",
        PKG / "src/learnable_recency_v1/config.py",
        PKG / "src/learnable_recency_v1/wrap.py",
        ROOT / "src/btransform_unified_v2/model.py",
        ROOT / "src/btransform_unified_v2/temporal.py",
        ROOT / "src/btransform_unified_v2/config.py",
        V1 / "src/btransform_unified_v1/ema.py",
        V1 / "src/btransform_unified_v1/model.py",
        V1 / "src/btransform_unified_v1/schedule.py",
        V1 / "src/btransform_unified_v1/r2.py",
        HERE / "m1_static_data.py",
        WS / "tfpd_exploration/src/m2_dual_track_v1/training.py",
    ]
    return {str(p): sha(p) for p in paths}


def rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def recipe(args: argparse.Namespace) -> None:
    ok = (
        args.seed == SEED
        and args.proj_dim == PROJ_DIM
        and args.tier == "learned_slope"
        and args.layers == 4
        and args.ladder == "default"
        and args.per_layer
        and not args.learn_flat_heads
        and not args.cable_nw
        and args.lr_multiplier == 1
        and args.half_lives is None
    )
    if not ok:
        raise ValueError(
            "STATIC M1 locks seed42/R100/D4/P16 learned_slope default ladder, six learned heads plus two fixed flat heads, and all other learnable controls"
        )
    if args.max_updates_smoke is None and args.epochs != EPOCHS:
        raise ValueError("formal STATIC M1 requires exactly 24 epochs")


def decoder(cfg, device):
    m = StaticLearnableRiftDecoder("m1", cfg, context_bins=CONTEXT, seed=SEED).to(
        device
    )
    m.temporal.set_attention_backend("local")
    if tuple(m.static_identity.shape) != (64, 16) or tuple(
        m.temporal_config.windows
    ) != (25, 25, 25, 24):
        raise RuntimeError("M1 STATIC architecture drift")
    return m


def optimizer(model):
    named = list(model.named_parameters())
    recency = [(n, p) for n, p in named if n.startswith("temporal.slope_log")]
    shared = [(n, p) for n, p in named if n not in {x[0] for x in recency}]
    groups = adamw_param_groups(shared, weight_decay=WEIGHT_DECAY)
    groups.append({"params": [p for _, p in recency], "weight_decay": 0.0})
    for group in groups:
        group["lr"] = LR
    return torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.999), eps=1e-8)


def score_surface(model, data, device, limit=None):
    model.eval()
    rows = {}
    ps = []
    ts = []
    for session in data["sessions"]:
        item = data["items"][session]
        pred = []
        target = []
        for bi in range(0, len(item["starts"]), BATCH):
            if limit is not None and bi // BATCH >= limit:
                break
            ids = np.arange(bi, min(bi + BATCH, len(item["starts"])), dtype=np.int64)
            x, y, v = batch_tensors(item, ids, device)
            with torch.inference_mode():
                p = (
                    model(
                        x,
                        unit_mask=torch.as_tensor(item["unit_mask"], device=device),
                        input_valid_mask=v,
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
            pred.append(p)
            target.append(np.asarray(item["Y"])[ids])
        p = np.concatenate(pred)
        t = np.concatenate(target)
        r = float(variance_weighted_r2(t, p))
        if not math.isfinite(r) or not np.isfinite(p).all() or not np.isfinite(t).all():
            raise FloatingPointError("non-finite heldout R2")
        rows[session] = {
            "r2": r,
            "window_count": len(t),
            "prediction_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
        }
        ps.append(p)
        ts.append(t)
    equal = float(np.mean([r["r2"] for r in rows.values()]))
    pooled = float(variance_weighted_r2(np.concatenate(ts), np.concatenate(ps)))
    if not math.isfinite(equal) or not math.isfinite(pooled):
        raise FloatingPointError("non-finite aggregate heldout R2")
    return {
        "per_session": rows,
        "equal_session_mean": equal,
        "pooled_r2": pooled,
        "n_windows": int(sum(r["window_count"] for r in rows.values())),
        "partial": limit is not None,
    }


def _ema_load(model, state):
    model.load_state_dict(state["raw_state_dict"], strict=True)
    shadow = state["ema"]["shadow"]
    named = dict(model.named_parameters())
    if set(shadow) != set(named):
        raise RuntimeError("EMA/model parameter schema mismatch")
    with torch.no_grad():
        for n, p in named.items():
            p.copy_(shadow[n].to(p.device, p.dtype))


def run_train(args):
    recipe(args)
    smoke = args.max_updates_smoke is not None
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    dest = args.dest.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("refusing to overwrite an existing STATIC M1 destination")
    data = load_source()
    batches = data["batches"]
    if not smoke and (
        len(batches) != UPDATES
        or len(data["sessions"]) != 4
        or sum(len(data["items"][s]["starts"]) for s in data["sessions"]) != 213336
    ):
        raise RuntimeError("M1 source sampler/data contract drift")
    cfg = config_from_args(args, "m1")
    model = decoder(cfg, device)
    ema = DecoderEMA(model, decay=EMA_DECAY)
    opt = optimizer(model)
    before = model.static_identity.detach().clone()
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": SCHEMA,
        "status": "SMOKE" if smoke else "FORMAL",
        "task": "m1",
        "cell": "M1-STATIC-R100-D4-P16-LEARNED-SLOPE-V1",
        "seed": SEED,
        "identity_interface": "static_identity",
        "no_support_or_e0_or_banks": True,
        "selection": {
            "rule": "HO3 all24 earliest maximum equal_session_mean (same as FULL)",
            "fixedfinal": False,
        },
        "learnable_config": cfg.__dict__,
        "context_bins": CONTEXT,
        "proj_dim": PROJ_DIM,
        "layer_windows": list(model.temporal_config.windows),
        "batch": BATCH,
        "epochs": EPOCHS,
        "updates_per_epoch": UPDATES,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "recency_weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "clip": CLIP,
        },
        "lr": {"peak": LR, "min": LR_MIN, "warmup_updates": UPDATES},
        "ema_decay": EMA_DECAY,
        "unit_dropout": UNIT_DROPOUT,
        "source_hashes": source_hashes(),
        "data_contract": data["contract"],
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    atom(dest / "run_meta.json", meta)
    started = time.monotonic()
    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bi, (session, ids) in enumerate(batches):
            x, y, v = batch_tensors(
                data["items"][session], np.asarray(ids, dtype=np.int64), device
            )
            step += 1
            lr = warmup_cosine_lr(
                step,
                total_steps=EPOCHS * UPDATES,
                warmup_steps=UPDATES,
                peak=LR,
                min_factor=0.1,
            )
            for g in opt.param_groups:
                g["lr"] = lr
            gen = torch.Generator(device="cpu")
            gen.manual_seed(unit_dropout_seed(SEED, epoch, bi))
            keep = whole_unit_dropout(
                torch.as_tensor(data["items"][session]["unit_mask"], dtype=torch.bool),
                UNIT_DROPOUT,
                gen,
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
                raise FloatingPointError("non-finite source loss")
            loss.backward()
            slope_grad = model.temporal.slope_log.grad
            if slope_grad is None or not torch.isfinite(slope_grad).all():
                raise FloatingPointError("non-finite learned slope gradient")
            nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True)
            opt.step()
            ema.update_after_step(model)
            losses.append(float(loss.detach().cpu()))
            if smoke and step >= args.max_updates_smoke:
                break
        state = {
            "schema": CKPT_SCHEMA,
            "epoch": epoch,
            "global_step": step,
            "seed": SEED,
            "smoke": smoke,
            "raw_state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "ema": ema.state_dict(),
            "rng": rng_state(device),
            "source_hashes": source_hashes(),
            "data_contract": data["contract"],
            "static_identity_shape": [64, 16],
        }
        ckpt = dest / f"epoch_{epoch:03d}.pt"
        atomic_checkpoint(ckpt, state)
        row = {
            "status": "SMOKE" if smoke else "TRAINING",
            "epoch": epoch,
            "global_step": step,
            "epoch_update_count": len(losses),
            "train_mse": float(np.mean(losses)),
            "ema_updates": ema.n_updates,
        }
        append(dest / "metrics.jsonl", row)
        atom(dest / "heartbeat.json", row)
        if smoke:
            disk = torch.load(ckpt, map_location=device, weights_only=False)
            if (
                disk.get("schema") != CKPT_SCHEMA
                or disk.get("source_hashes") != source_hashes()
                or disk.get("data_contract") != data["contract"]
            ):
                raise RuntimeError("saved smoke checkpoint seal mismatch")
            restored = decoder(cfg, device)
            restored.load_state_dict(disk["raw_state_dict"])
            item = data["items"][data["sessions"][0]]
            x, _, v = batch_tensors(item, np.arange(1), device)
            parity = torch.equal(
                model.eval()(x, input_valid_mask=v),
                restored.eval()(x, input_valid_mask=v),
            )
            changed = not torch.equal(before, model.static_identity.detach())
            # This reference is made from the live training EMA shadow, before
            # the disk checkpoint is consulted for the EMA view.
            ema_reference = decoder(cfg, device)
            ema_reference.load_state_dict(model.state_dict())
            ema.apply_to(ema_reference)
            ema_loaded = decoder(cfg, device)
            _ema_load(ema_loaded, disk)
            ema_parity = bool(
                torch.equal(
                    ema_reference.eval()(x, input_valid_mask=v),
                    ema_loaded.eval()(x, input_valid_mask=v),
                )
            )
            # Full R100 offline/stream comparison, including the padded prefix.
            streamer = StaticRiftStreamDecoder(restored.eval())
            stream = None
            for ti in range(CONTEXT):
                stream = streamer.stream_step(x[:, ti], ["smoke"], valid_mask=v[:, ti])
            offline = restored(x, input_valid_mask=v)[0]
            finite = math.isfinite(row["train_mse"])
            stream_parity = bool(
                torch.allclose(stream[0], offline, atol=1e-5, rtol=1e-5)
            )
            held_report = score_surface(ema_loaded, load_heldout(), device, limit=1)
            grad_nonzero = bool(torch.count_nonzero(slope_grad).item())
            if (
                not (
                    finite
                    and changed
                    and parity
                    and ema_parity
                    and stream_parity
                    and grad_nonzero
                )
                or held_report["n_windows"] != 96
                or not held_report["partial"]
                or not math.isfinite(held_report["pooled_r2"])
            ):
                raise RuntimeError("M1 STATIC smoke receipt validation failed")
            receipt = {
                "schema": "m1_static_smoke_receipt_v1",
                "status": "COMPLETED",
                "formal_claim": False,
                "finite_loss": finite,
                "identity_updated": changed,
                "raw_checkpoint_parity": parity,
                "ema_checkpoint_parity": ema_parity,
                "stream_offline_r100_parity": stream_parity,
                "stream_max_abs": float((stream[0] - offline).abs().max().cpu()),
                "slope_grad_finite": True,
                "slope_grad_nonzero": grad_nonzero,
                "heldout_ho3_limited32": {
                    "partial": True,
                    "n_windows": held_report["n_windows"],
                    "pooled_r2": held_report["pooled_r2"],
                },
                "checkpoint": str(ckpt),
                "global_step": step,
                "runtime_seconds": time.monotonic() - started,
            }
            atom(dest / "smoke_receipt.json", receipt)
            return receipt
        if step != epoch * UPDATES or ema.n_updates != step:
            raise RuntimeError("formal step/EMA accounting drift")
    receipt = {
        "schema": "m1_static_train_receipt_v1",
        "status": "COMPLETED",
        "selection": {
            "checkpoint": "epoch_024.pt",
            "view": "EMA",
            "rule": "pending HO3 earliest-max pick; source_only",
        },
        "epochs": EPOCHS,
        "global_step": step,
        "runtime_seconds": time.monotonic() - started,
    }
    atom(dest / "train_receipt.json", receipt)
    return receipt


def run_score(args):
    run = args.dest.resolve()
    meta = json.loads((run / "run_meta.json").read_text())
    smoke = meta.get("status") == "SMOKE"
    receipt = json.loads(
        (run / ("smoke_receipt.json" if smoke else "train_receipt.json")).read_text()
    )
    if (
        receipt.get("status") != "COMPLETED"
        or (
            smoke
            and (not args.allow_smoke or not args.max_batches or args.max_batches < 1)
        )
        or (not smoke and meta.get("status") != "FORMAL")
    ):
        raise RuntimeError(
            "score requires completed formal run, or --allow-smoke with positive --max-batches"
        )
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be positive")
    if meta.get("schema") != SCHEMA or meta.get("source_hashes") != source_hashes():
        raise RuntimeError("run metadata schema/source seal mismatch")
    ckpt = run / ("epoch_001.pt" if smoke else "epoch_024.pt")
    state = torch.load(ckpt, map_location=args.device, weights_only=False)
    if (
        state.get("schema") != CKPT_SCHEMA
        or bool(state.get("smoke")) != smoke
        or state.get("source_hashes") != meta["source_hashes"]
        or state.get("data_contract") != meta["data_contract"]
        or (
            not smoke
            and (
                state.get("epoch") != 24
                or state.get("global_step") != EPOCHS * UPDATES
                or state.get("ema", {}).get("n_updates") != EPOCHS * UPDATES
            )
        )
    ):
        raise RuntimeError("fixed final checkpoint required")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    model = decoder(config_from_run_meta(meta, "m1"), device)
    _ema_load(model, state)
    held = load_heldout()
    report = score_surface(model, held, device, args.max_batches)
    out = {
        "schema": SCORE_SCHEMA,
        "status": "SMOKE_COMPLETED" if smoke else "COMPLETED",
        "formal_claim": not smoke,
        "partial": args.max_batches is not None,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha(ckpt),
        "view": "EMA",
        "selection": {
            "epoch": state["epoch"],
            "rule": "last-epoch sidecar; official pick is --stage pick",
        },
        "heldout_contract": held["contract"],
        "metrics": report,
    }
    atom(run / "score_receipt.json", out)
    return out


def run_pick(args):
    run = args.dest.resolve()
    meta = json.loads((run / "run_meta.json").read_text())
    receipt = json.loads((run / "train_receipt.json").read_text())
    now_hashes = source_hashes()
    skip = {str(Path(__file__))}
    locked = {k: v for k, v in now_hashes.items() if k not in skip}
    recorded = {k: v for k, v in (meta.get("source_hashes") or {}).items() if k not in skip}
    if (
        receipt.get("status") != "COMPLETED"
        or meta.get("status") != "FORMAL"
        or meta.get("schema") != SCHEMA
        or locked != recorded
    ):
        raise RuntimeError("pick requires a completed formal static run")
    if args.max_batches is not None or args.allow_smoke:
        raise RuntimeError("pick is formal full-surface only")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    model = decoder(config_from_run_meta(meta, "m1"), device)
    held = load_heldout()
    progress_path = run / "pick_progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.is_file()
        else {"completed": {}}
    )
    curve = progress.get("completed", {})
    for epoch in range(1, EPOCHS + 1):
        if str(epoch) in curve:
            continue
        ckpt = run / f"epoch_{epoch:03d}.pt"
        state = torch.load(ckpt, map_location=args.device, weights_only=False)
        if (
            state.get("schema") != CKPT_SCHEMA
            or bool(state.get("smoke"))
            or state.get("source_hashes") != meta["source_hashes"]
            or int(state.get("epoch", -1)) != epoch
        ):
            raise RuntimeError(f"static checkpoint drift at epoch {epoch}")
        _ema_load(model, state)
        report = score_surface(model, held, device)
        curve[str(epoch)] = {**report, "checkpoint_sha256": sha(ckpt)}
        atom(progress_path, {"completed": curve})
    values = {
        epoch: float(curve[str(epoch)]["equal_session_mean"])
        for epoch in range(1, EPOCHS + 1)
    }
    best = max(range(1, EPOCHS + 1), key=lambda epoch: (values[epoch], -epoch))
    out = {
        "schema": PICK_SCHEMA,
        "status": "COMPLETED",
        "formal_claim": True,
        "partial": False,
        "view": "EMA",
        "selection": {
            "epoch": best,
            "equal_session_mean": values[best],
            "rule": SELECTION_RULE,
        },
        "ema_by_epoch": curve,
        "heldout_contract": held["contract"],
        "official_test_used": False,
    }
    atom(run / "ho3_selection.json", out)
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_learnable_flags(p)
    p.set_defaults(ladder="default")
    p.add_argument("--stage", choices=("train", "score", "pick"), default="train")
    p.add_argument("--dest", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--proj-dim", type=int, default=PROJ_DIM)
    p.add_argument("--max-updates-smoke", type=int)
    p.add_argument("--allow-smoke", action="store_true")
    p.add_argument("--max-batches", type=int)
    return p


def main():
    a = build_parser().parse_args()
    if a.max_updates_smoke is not None and (
        a.max_updates_smoke < 1 or a.max_updates_smoke > UPDATES or a.stage != "train"
    ):
        raise ValueError(f"--max-updates-smoke must be 1..{UPDATES} and train-only")
    if a.dest is None:
        a.dest = RESULTS / (
            "smoke/m1_static_s42_final" if a.max_updates_smoke else "m1_static_s42"
        )
    if a.max_updates_smoke is not None and a.device == "cuda:0":
        a.device = "cpu"
    stage = {"train": run_train, "score": run_score, "pick": run_pick}[a.stage]
    print(json.dumps(stage(a), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
