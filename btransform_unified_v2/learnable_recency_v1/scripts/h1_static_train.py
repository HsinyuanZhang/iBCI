#!/usr/bin/env python3
"""H1 STATIC R300 learned-recency route.

This route intentionally has no TaskBank, C2 identity materialization, carrier,
or calibration support input.  It learns ``static_identity[176,16]`` from the
source-supervised raw H1 cache and uses the local public held-out-calibration
NWB as the HO-M3 development query surface.  Epoch pick is the same
earliest-max grouped-seven rule as FULL (``--stage pick``).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
V1 = ROOT.parent / "btransform_unified_v1"
SPINT = ROOT.parent / "SPINT-main"
for path in (PKG / "src", ROOT / "src", V1 / "src", V1 / "scripts", ROOT.parent, SPINT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from btransform_unified_v1 import adapters, h1_config  # noqa: E402
from btransform_unified_v1.c2_protocol import (
    HELDOUT_SESSION_TO_FALCON_KEY,
    HO_SELECTION_METRIC,
    grouped_session_metrics,
    select_epoch,
)  # noqa: E402
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.model import (
    unit_dropout_seed,
    whole_unit_dropout,
)  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402
from learnable_recency_v1.config import (
    add_learnable_flags,
    config_from_args,
    config_from_run_meta,
)  # noqa: E402
from learnable_recency_v1.static_model import (
    StaticLearnableRiftDecoder,
    StaticRiftStreamDecoder,
)  # noqa: E402
from learnable_recency_v1.wrap import (
    apply_group_lrs,
    split_optimizer_parameters,
)  # noqa: E402

CONTEXT, EPOCHS, UPDATES, SEED, BATCH, MICRO = 300, 32, 731, 42, 32, 32
SCALE = h1_config.TARGET_MULTIPLIER
RESULTS = PKG / "results"
OLD_SELECTION_RULE = (
    "fixed final EMA checkpoint: epoch 32; local HO labels never select epoch"
)
SELECTION_RULE = "HO-M3 grouped-seven, all32, earliest maximum (same as FULL)"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def endpoint_context(
    neural: np.ndarray, ends: np.ndarray, context: int = CONTEXT
) -> tuple[np.ndarray, np.ndarray]:
    """End-anchored raw contexts with explicit invalid left padding."""
    neural = np.ascontiguousarray(neural, np.float32)
    out = np.zeros((len(ends), context, neural.shape[1]), np.float32)
    valid = np.zeros((len(ends), context), np.bool_)
    for row, endpoint in enumerate(np.asarray(ends, np.int64)):
        lo = max(0, int(endpoint) - context + 1)
        width = int(endpoint) - lo + 1
        if width:
            out[row, context - width :] = neural[lo : int(endpoint) + 1]
            valid[row, context - width :] = True
    return out, valid


def _source_endpoints(session: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct source-cache access; source labels are training supervision only."""
    row = adapters._h1_source_cache()["train"][session]
    neural = np.ascontiguousarray(row["neural"], np.float32)
    ends = np.asarray(row["query_starts"], np.int64) + h1_config.FULL_WINDOW - 1
    eval_mask = np.asarray(row["eval_mask"], bool)
    ends = ends[(ends >= 0) & (ends < len(eval_mask))]
    ends = ends[eval_mask[ends]]
    y = np.ascontiguousarray(row["velocity"], np.float32)[ends]
    return neural, ends, y


def _load_data() -> dict[str, Any]:
    """Load only source cache neural/Y, never an existing bank loader."""
    sessions = list(h1_config.H1_ALL_SESSIONS)
    xs: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    ys: dict[str, np.ndarray] = {}
    ids: dict[str, np.ndarray] = {}
    for session in sessions:
        neural, ends, y = _source_endpoints(session)
        if neural.shape[1] != 176:
            raise RuntimeError(
                f"{session}: expected fixed H1 width 176, got {neural.shape}"
            )
        x, mask = endpoint_context(neural, ends)
        xs[session], valid[session], ys[session], ids[session] = x, mask, y, ends
    updates = sum((len(xs[s]) + BATCH - 1) // BATCH for s in sessions)
    if updates != UPDATES:
        raise RuntimeError(
            f"source direct endpoint recipe drift: {updates} != {UPDATES}"
        )
    digest = hashlib.sha256()
    for session in sessions:
        digest.update(session.encode())
        digest.update(ids[session].tobytes())
        digest.update(valid[session].tobytes())
    return {
        "sessions": sessions,
        "X": xs,
        "valid": valid,
        "y": ys,
        "ids": ids,
        "updates_per_epoch": updates,
        "endpoint_valid_sha256": digest.hexdigest(),
    }


def _load_ho() -> dict[str, Any]:
    """Local query scoring reads only NWB raw neural, labels, and score masks.

    There is deliberately no ``_h1_payload_arrays``, ``build_ho_calibration``,
    B2 materialization, or support construction in this route.
    """
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    root = SPINT / "data" / "000954" / "sub-HumanPitt-held-out-calib"
    xs: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    ys: dict[str, np.ndarray] = {}
    keys = []
    for session, key in HELDOUT_SESSION_TO_FALCON_KEY:
        neural, velocity, _change, score_mask = load_nwb(
            root / f"sub-HumanPitt-held-out-calib_{session}.nwb", FalconTask.h1
        )
        ends = np.flatnonzero(np.asarray(score_mask, bool)).astype(np.int64)
        x, mask = endpoint_context(np.asarray(neural, np.float32), ends)
        xs[key], valid[key], ys[key] = x, mask, np.asarray(velocity, np.float32)[ends]
        keys.append(key)
    return {"X": xs, "valid": valid, "y": ys, "keys": keys}


def _manifest() -> dict[str, str]:
    paths = [
        Path(__file__),
        V1 / "src/btransform_unified_v1/adapters.py",
        V1 / "src/btransform_unified_v1/h1_config.py",
        V1 / "src/btransform_unified_v1/ema.py",
        V1 / "src/btransform_unified_v1/model.py",
        V1 / "src/btransform_unified_v1/schedule.py",
        V1 / "src/btransform_unified_v1/c2_protocol.py",
        ROOT / "src/btransform_unified_v2/model.py",
        ROOT / "src/btransform_unified_v2/temporal.py",
        ROOT / "src/btransform_unified_v2/config.py",
        PKG / "src/learnable_recency_v1/static_model.py",
        PKG / "src/learnable_recency_v1/wrap.py",
        PKG / "src/learnable_recency_v1/temporal.py",
        PKG / "src/learnable_recency_v1/config.py",
    ]
    return {str(p): _sha(p) for p in paths}


def _json_config(cfg: Any) -> dict[str, Any]:
    """Normalize tuples before comparing the JSON metadata on resume."""
    return json.loads(json.dumps(cfg.__dict__, default=str))


def _checkpoint(
    path: Path,
    model: nn.Module,
    opt: Any,
    ema: DecoderEMA,
    epoch: int,
    step: int,
    rng: np.random.Generator,
    smoke: bool,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite static checkpoint: {path}")
    state = {
        "schema": "h1_static_learnable_train_v1",
        "variant": "static",
        "epoch": epoch,
        "global_step": step,
        "smoke": smoke,
        "raw_state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "ema": ema.state_dict(),
        "rng": rng.bit_generator.state,
        "torch_rng_cpu": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "torch_rng_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        ),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _stream_parity(
    model: StaticLearnableRiftDecoder,
    x: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    xb = torch.from_numpy(x[:1]).to(device)
    vm = torch.from_numpy(valid[:1]).to(device)
    with torch.inference_mode():
        offline = model(xb, input_valid_mask=vm)[0]
    stream = StaticRiftStreamDecoder(model)
    for i in range(xb.shape[1]):
        stream.stream_step(xb[:, i], ["parity"], valid_mask=vm[:, i])
    online = stream.predict("parity")
    if not torch.allclose(offline, online, rtol=3e-4, atol=3e-5):
        raise RuntimeError("static offline/stream parity drift")
    return {
        "max_abs": float((offline - online).abs().max().cpu()),
        "valid_bins": int(vm.sum().item()),
    }


def _save_load_prediction_parity(
    model: StaticLearnableRiftDecoder,
    recency_cfg: Any,
    checkpoint: Path,
    x: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
) -> float:
    """A checkpoint must reproduce a raw-model query prediction exactly enough."""
    model.eval()
    xb = torch.from_numpy(x[:2]).to(device)
    vm = torch.from_numpy(valid[:2]).to(device)
    with torch.inference_mode():
        expected = model(xb, input_valid_mask=vm).detach().cpu()
    restored = StaticLearnableRiftDecoder("h1", recency_cfg, seed=SEED).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    restored.load_state_dict(state["raw_state_dict"], strict=True)
    restored.eval()
    with torch.inference_mode():
        actual = restored(xb, input_valid_mask=vm).cpu()
    if not torch.allclose(expected, actual, rtol=2e-5, atol=2e-6):
        raise RuntimeError("static checkpoint save/load prediction parity drift")
    return float((expected - actual).abs().max())


def _score(
    model: StaticLearnableRiftDecoder,
    ema: DecoderEMA,
    ho: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    saved = {n: p.detach().clone() for n, p in model.named_parameters()}
    was = model.training
    try:
        ema.apply_to(model)
        model.eval()
        pred = {}
        masks = {}
        for key in ho["keys"]:
            pieces = []
            for off in range(0, len(ho["X"][key]), BATCH):
                xb = torch.from_numpy(ho["X"][key][off : off + BATCH]).to(device)
                vm = torch.from_numpy(ho["valid"][key][off : off + BATCH]).to(device)
                with torch.inference_mode():
                    pieces.append(
                        (model(xb, input_valid_mask=vm) / SCALE).cpu().numpy()
                    )
            pred[key] = np.concatenate(pieces)
            masks[key] = np.ones(len(pred[key]), bool)
            if not np.isfinite(pred[key]).all():
                raise FloatingPointError(
                    f"nonfinite static local query prediction for {key}"
                )
        report = grouped_session_metrics(
            pred, ho["y"], masks, HELDOUT_SESSION_TO_FALCON_KEY
        )
        required = ("r2_mean", "worst_session_r2", "r2_std_population")
        if any(not np.isfinite(float(report[name])) for name in required):
            raise FloatingPointError("nonfinite grouped static local query R2")
        if not report.get("per_session_r2") or any(
            not np.isfinite(float(value)) for value in report["per_session_r2"].values()
        ):
            raise FloatingPointError("nonfinite per-session static local query R2")
        return report
    finally:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(saved[n])
        model.train(was)


def _rebuild(meta: dict[str, Any], device: torch.device) -> StaticLearnableRiftDecoder:
    return StaticLearnableRiftDecoder(
        "h1", config_from_run_meta(meta, "h1"), seed=int(meta["seed"])
    ).to(device)


def _validate_recipe(args: Any, cfg: Any) -> None:
    if (
        args.seed != SEED
        or args.half_lives is not None
        or cfg.context_bins != CONTEXT
        or cfg.layers != 4
        or cfg.tier != "learned_slope"
        or cfg.ladder != "default"
        or not cfg.per_layer
        or cfg.learn_flat_heads
        or cfg.lr_multiplier != 1.0
        or cfg.cable_nw
    ):
        raise ValueError(
            "STATIC H1 recipe fixes R300/D4 learned_slope, default ladder, per-layer slopes, and two flat heads"
        )


def train(args: Any) -> dict[str, Any]:
    cfg = config_from_args(args, "h1")
    _validate_recipe(args, cfg)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data = _load_data()
    dest = args.dest
    smoke = args.max_updates_smoke is not None
    if dest.exists() and any(dest.iterdir()) and not args.resume:
        raise FileExistsError(f"fresh destination required: {dest}")
    model = StaticLearnableRiftDecoder("h1", cfg, seed=args.seed).to(args.device)
    if (
        tuple(model.static_identity.shape) != (176, 16)
        or tuple(model.static_carrier.shape) != (176, 4)
        or bool(model.static_carrier.any())
    ):
        raise RuntimeError("static identity/carrier invariant drift")
    ema = DecoderEMA(model, decay=0.9995)
    groups = split_optimizer_parameters(
        model, peak_lr=1e-4, weight_decay=0.01, lr_multiplier=cfg.lr_multiplier
    )
    opt = torch.optim.AdamW(
        groups, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )
    rng = np.random.default_rng(args.seed)
    step = 0
    first_epoch = 1
    initial_identity = model.static_identity.detach().cpu().clone()
    meta = {
        "schema": "h1_static_learnable_v1",
        "status": "SMOKE" if smoke else "FORMAL",
        "smoke": smoke,
        "variant": "STATIC",
        "task": "h1",
        "seed": args.seed,
        "context_bins": CONTEXT,
        "depth": 4,
        "proj_dim": 16,
        "attention_backend": "dense",
        "static_identity_shape": [176, 16],
        "static_carrier_shape": [176, 4],
        "static_carrier_literal_zero": True,
        "source_protocol": "direct adapters._h1_source_cache()['train'][session] neural/velocity; no TaskBank/E0/carrier/calibration support",
        "source_cache_path": str(h1_config.SOURCE_CACHE_PATH),
        "source_cache_sha256": _sha(h1_config.SOURCE_CACHE_PATH),
        "endpoint_valid_sha256": data["endpoint_valid_sha256"],
        "fixed_channel_index_assumption": "H1 raw 176 columns are fixed and all unit-mask columns are true; no channel remap",
        "input_valid_mask": "end-anchored R300 left padding is explicit invalid evidence",
        "selection_rule": SELECTION_RULE,
        "local_ho_surface": "held-out-calib NWB raw neural/labels/score masks only; local development query scoring",
        "epochs": 1 if smoke else EPOCHS,
        "updates_per_epoch": UPDATES,
        "batch": BATCH,
        "microbatch": MICRO,
        "ema": 0.9995,
        "unit_dropout": 0.1,
        "optimizer": {
            "name": "AdamW",
            "peak_lr": 1e-4,
            "floor_lr": 1e-5,
            "warmup_epochs": 1,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "clip_grad_norm": 1.0,
            "parameter_groups": "split_optimizer_parameters; learned slopes have zero weight decay",
        },
        "learnable_config": _json_config(cfg),
        "ladder": cfg.ladder_metadata(),
        "source_manifest_sha256": _manifest(),
        "official_test_used": False,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.resume:
        old = json.loads((dest / "run_meta.json").read_text())
        if (
            args.resume.resolve().parent != dest.resolve()
            or smoke
            or old.get("schema") != meta["schema"]
            or old.get("status") != "FORMAL"
            or old.get("source_manifest_sha256") != meta["source_manifest_sha256"]
            or old.get("source_cache_sha256") != meta["source_cache_sha256"]
            or old.get("endpoint_valid_sha256") != meta["endpoint_valid_sha256"]
            or old.get("learnable_config") != meta["learnable_config"]
        ):
            raise RuntimeError("resume metadata/protocol drift")
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        if (
            state.get("schema") != "h1_static_learnable_train_v1"
            or state.get("smoke")
            or state.get("variant") != "static"
            or int(state.get("epoch", -1)) * UPDATES
            != int(state.get("global_step", -1))
            or int(state.get("ema", {}).get("n_updates", -1))
            != int(state.get("global_step", -1))
            or set(
                (
                    "raw_state_dict",
                    "optimizer",
                    "rng",
                    "torch_rng_cpu",
                    "numpy_rng",
                    "python_rng",
                )
            )
            - set(state)
        ):
            raise RuntimeError(
                "resume requires own complete formal static epoch checkpoint"
            )
        model.load_state_dict(state["raw_state_dict"], strict=True)
        opt.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng_cpu"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if (
            str(args.device).startswith("cuda")
            and state.get("torch_rng_cuda") is not None
        ):
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in state["torch_rng_cuda"]]
            )
        step = int(state["global_step"])
        first_epoch = int(state["epoch"]) + 1
        meta = old
        if first_epoch > EPOCHS:
            raise RuntimeError(
                "resume checkpoint already completes the fixed epoch-32 run"
            )
    else:
        dest.mkdir(parents=True, exist_ok=True)
        _atomic(dest / "run_meta.json", meta)
    parity_pre = _stream_parity(
        model,
        data["X"][data["sessions"][0]],
        data["valid"][data["sessions"][0]],
        torch.device(args.device),
    )
    _atomic(dest / "runtime_parity_pre.json", parity_pre)
    started = time.monotonic()
    last_loss = None
    last_grad = None
    for epoch in range(first_epoch, EPOCHS + 1):
        model.train()
        losses = []
        order = list(data["sessions"])
        rng.shuffle(order)
        within = 0
        for session in order:
            indices = rng.permutation(len(data["X"][session]))
            for off in range(0, len(data["X"][session]), BATCH):
                take = indices[off : off + BATCH]
                step += 1
                within += 1
                apply_group_lrs(
                    opt,
                    warmup_cosine_lr(
                        step, EPOCHS * UPDATES, UPDATES, peak=1e-4, min_factor=0.1
                    ),
                )
                keep = whole_unit_dropout(
                    torch.ones(176, dtype=torch.bool),
                    p=0.1,
                    generator=torch.Generator().manual_seed(
                        unit_dropout_seed(args.seed, epoch, within - 1)
                    ),
                )
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for moff in range(0, len(take), MICRO):
                    subset = take[moff : moff + MICRO]
                    xb = torch.from_numpy(data["X"][session][subset]).to(args.device)
                    vm = torch.from_numpy(data["valid"][session][subset]).to(
                        args.device
                    )
                    yb = torch.from_numpy(data["y"][session][subset] * SCALE).to(
                        args.device
                    )
                    amp = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if str(args.device).startswith("cuda")
                        else contextlib.nullcontext()
                    )
                    with amp:
                        loss = nn.functional.mse_loss(
                            model(xb, dropout_keep=keep, input_valid_mask=vm).float(),
                            yb,
                        )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(
                            f"nonfinite loss epoch={epoch} step={step}"
                        )
                    (loss * len(subset) / len(take)).backward()
                    batch_loss += float(loss.detach()) * len(subset) / len(take)
                last_grad = float(
                    nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                )
                opt.step()
                ema.update_after_step(model)
                losses.append(batch_loss)
                last_loss = batch_loss
                if smoke and step >= args.max_updates_smoke:
                    break
            if smoke and step >= args.max_updates_smoke:
                break
        _checkpoint(
            dest / f"epoch_{epoch:03d}.pt", model, opt, ema, epoch, step, rng, smoke
        )
        row = {
            "event": "epoch",
            "epoch": epoch,
            "global_step": step,
            "train_mse": float(np.mean(losses)),
            "lr": float(opt.param_groups[0]["lr"]),
            "smoke": smoke,
        }
        with (dest / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        _atomic(dest / "heartbeat.json", row)
        if smoke:
            break
        if within != UPDATES or step != epoch * UPDATES:
            raise RuntimeError("formal 731-update schedule drift")
    parity_post = _stream_parity(
        model,
        data["X"][data["sessions"][0]],
        data["valid"][data["sessions"][0]],
        torch.device(args.device),
    )
    _atomic(dest / "runtime_parity_post.json", parity_post)
    changed = not torch.equal(initial_identity, model.static_identity.detach().cpu())
    if not changed:
        raise RuntimeError("STATIC smoke/training did not update static_identity")
    if smoke:
        checkpoint_parity = _save_load_prediction_parity(
            model,
            cfg,
            dest / "epoch_001.pt",
            data["X"][data["sessions"][0]],
            data["valid"][data["sessions"][0]],
            torch.device(args.device),
        )
        ho = _load_ho()
        for key in ho["keys"]:
            ho["X"][key] = ho["X"][key][:2]
            ho["valid"][key] = ho["valid"][key][:2]
            ho["y"][key] = ho["y"][key][:2]
        few_bin_report = _score(model, ema, ho, torch.device(args.device))
        query_counts = {str(key): int(len(ho["y"][key])) for key in ho["keys"]}
        result = {
            "status": "SMOKE_COMPLETED",
            "steps": step,
            "finite_loss": bool(last_loss is not None and np.isfinite(last_loss)),
            "finite_grad": bool(last_grad is not None and np.isfinite(last_grad)),
            "static_identity_updated": changed,
            "stream_parity": parity_post,
            "save_load_pred_max_abs": checkpoint_parity,
            "few_bins_local_query_score": {
                "query_counts_by_surface_key": query_counts,
                "surface_key_count": len(query_counts),
                "metric_group_count": len(few_bin_report["per_session_r2"]),
                "queries_per_surface_key": 2,
                "r2_mean": few_bin_report["r2_mean"],
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic(dest / "smoke_receipt.json", result)
        return result
    if step != EPOCHS * UPDATES:
        raise RuntimeError("formal total update drift")
    return score(args)


def _formal_meta(dest: Path) -> dict[str, Any]:
    meta = json.loads((dest / "run_meta.json").read_text())
    if meta.get("status") != "FORMAL" or meta.get("selection_rule") not in {
        OLD_SELECTION_RULE,
        SELECTION_RULE,
    }:
        raise RuntimeError("only completed formal STATIC run may score or pick")
    return meta


def score(args: Any) -> dict[str, Any]:
    meta = _formal_meta(args.dest)
    device = torch.device(args.device)
    model = _rebuild(meta, device)
    ema = DecoderEMA(model, decay=0.9995)
    state = torch.load(
        args.dest / "epoch_032.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(state["raw_state_dict"], strict=True)
    ema.load_state_dict(state["ema"])
    if (
        state.get("smoke")
        or state.get("variant") != "static"
        or state.get("epoch") != 32
        or state.get("global_step") != EPOCHS * UPDATES
        or int(state["ema"].get("n_updates", -1)) != EPOCHS * UPDATES
    ):
        raise RuntimeError(
            "final static EMA checkpoint is not the complete epoch-32 checkpoint"
        )
    if tuple(model.static_identity.shape) != (176, 16) or bool(
        model.static_carrier.any()
    ):
        raise RuntimeError("static final model invariant drift")
    report = _score(model, ema, _load_ho(), device)
    last = {
        "epoch": 32,
        "epoch_zero_based": 31,
        HO_SELECTION_METRIC: report["r2_mean"],
        "worst_session_r2": report["worst_session_r2"],
        "session_std_population": report["r2_std_population"],
        "per_session_r2": report["per_session_r2"],
    }
    _atomic(
        args.dest / "local_ho_static_report.json",
        {
            "status": "LOCAL_HO_REPORT_FINAL_EMA",
            "selected": last,
            "selection_rule": "last-epoch sidecar; official pick is --stage pick",
            "local_ho_labels_used_for_selection": False,
        },
    )
    if (args.dest / "ho_m3_selection.json").is_file():
        return json.loads((args.dest / "train_receipt.json").read_text())
    receipt = {
        "status": "COMPLETED",
        "epochs": EPOCHS,
        "updates": EPOCHS * UPDATES,
        "selected_epoch": 32,
        "selection_pending_pick": True,
        "official_test_used": False,
        "local_ho_labels_used_for_selection": False,
    }
    _atomic(args.dest / "train_receipt.json", receipt)
    return receipt


def pick(args: Any) -> dict[str, Any]:
    meta = _formal_meta(args.dest)
    device = torch.device(args.device)
    model = _rebuild(meta, device)
    ema = DecoderEMA(model, decay=0.9995)
    ho = _load_ho()
    progress_path = args.dest / "pick_progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.is_file()
        else {"completed": {}}
    )
    curve_map = {int(k): v for k, v in progress.get("completed", {}).items()}
    for epoch in range(1, EPOCHS + 1):
        if epoch in curve_map:
            continue
        state = torch.load(
            args.dest / f"epoch_{epoch:03d}.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(state["raw_state_dict"], strict=True)
        ema.load_state_dict(state["ema"])
        if (
            state.get("smoke")
            or state.get("variant") != "static"
            or int(state.get("epoch", -1)) != epoch
        ):
            raise RuntimeError(f"static checkpoint drift at epoch {epoch}")
        report = _score(model, ema, ho, device)
        curve_map[epoch] = {
            "epoch": epoch,
            "epoch_zero_based": epoch - 1,
            HO_SELECTION_METRIC: report["r2_mean"],
            "worst_session_r2": report["worst_session_r2"],
            "session_std_population": report["r2_std_population"],
            "per_session_r2": report["per_session_r2"],
        }
        _atomic(
            progress_path,
            {"completed": {str(k): curve_map[k] for k in sorted(curve_map)}},
        )
    curve = [curve_map[epoch] for epoch in range(1, EPOCHS + 1)]
    selected = select_epoch(curve)
    meta["selection_rule"] = SELECTION_RULE
    _atomic(args.dest / "run_meta.json", meta)
    _atomic(
        args.dest / "ho_m3_selection.json",
        {
            "status": "HO_M3_DEVELOPMENT_SELECTION",
            "selected": selected,
            "curve": curve,
            "selection_rule": SELECTION_RULE,
        },
    )
    receipt = {
        "status": "COMPLETED",
        "epochs": EPOCHS,
        "updates": EPOCHS * UPDATES,
        "selected_epoch": selected["epoch"],
        "selection_pending_pick": False,
        "official_test_used": False,
        "local_ho_labels_used_for_selection": True,
        "selection_rule": SELECTION_RULE,
    }
    _atomic(args.dest / "train_receipt.json", receipt)
    return {"status": "PICK_COMPLETED", "selected_epoch": selected["epoch"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_learnable_flags(parser)
    parser.set_defaults(ladder="default")
    parser.add_argument("--config", default="h1")
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("train", "score", "pick"), default="train")
    parser.add_argument("--max-updates-smoke", type=int)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if args.config != "h1" or args.seed != SEED:
        p.error("STATIC H1 formal recipe fixes config=h1 and seed=42")
    if args.smoke_steps is not None:
        args.max_updates_smoke = args.smoke_steps
    if args.max_updates_smoke is not None and args.max_updates_smoke < 1:
        p.error("positive smoke updates required")
    if args.dest is None:
        args.dest = (
            RESULTS
            / ("smoke/h1_static_s42" if args.max_updates_smoke else "h1_static_s42")
        ).resolve()
    else:
        args.dest = args.dest.resolve()
    if args.max_updates_smoke is not None and args.device == "cuda:0":
        args.device = "cpu"
    stage = {"train": train, "score": score, "pick": pick}[args.stage]
    print(json.dumps(stage(args), indent=2, default=str))


if __name__ == "__main__":
    main()
