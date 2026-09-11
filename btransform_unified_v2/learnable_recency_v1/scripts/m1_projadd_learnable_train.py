#!/usr/bin/env python3
"""M1 RIFT R100/D4/P16 proj_add learnable-recency trainer.

Mirrors ``btransform_unified_v2/scripts/rift_v1/m1_train.py`` (data pipeline,
HO scoring, receipts) and swaps the fixed-recency ``RiftDecoder`` for the
``LearnableRiftDecoder`` wrapper.  Does not edit any existing file; the rift
runner is loaded privately and only its helpers are reused.

Pairing: the formal reference is ``results/rift_v1/m1_r100_recency_s42_formal_v3``
(stock DEFAULT ladder).  Shared ``named_parameters`` of the learnable model stay
byte-equal to a same-seed stock ``RiftDecoder`` (only the ``recency_slopes``
buffer differs under the scaled M1 1/3 ladder); init-forward parity is checked
against a *same-ladder* fixed reference built via ``install_temporal``.
"""
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
V1 = WS / "btransform_unified_v1"
RIFT_SCRIPT = ROOT / "scripts/rift_v1/m1_train.py"
PAIRED_DEFAULT = ROOT / "results/rift_v1/m1_r100_recency_s42_formal_v3"
RESULTS = PKG / "results"

for candidate in (PKG / "src", ROOT / "src", V1 / "src", V1 / "scripts", WS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from btransform_unified_v1 import plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.config import add_learnable_flags, config_from_args, config_from_run_meta, dataset_config
from learnable_recency_v1.p8 import maybe_truncate_p8
from learnable_recency_v1.temporal import LearnableRecencyTemporal
from learnable_recency_v1.wrap import (
    LearnableRiftDecoder,
    LearnableRiftStreamDecoder,
    apply_group_lrs,
    assert_shared_byte_equal,
    install_temporal,
    new_parameter_names,
    recency_bias_snapshot,
    split_optimizer_parameters,
    trainable_new_parameter_count,
)

# Recipe constants must match scripts/rift_v1/m1_train.py exactly (asserted at load).
SEED, CONTEXT, PROJ_DIM, EPOCHS, BATCH, LR = 42, 100, 16, 24, 32, 1e-4
QUERY_PAD_BINS = CONTEXT - 1
HO = ("20121004", "20121017", "20121024")
SOURCE_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
EXPECTED_WINDOWS, UPDATES_PER_EPOCH = 213336, 6665

PAIRED_CELL = "M1-RIFT-R100-D4-P16-RECENCY-V1"
TRAIN_SCHEMA = "m1_projadd_learnable_train_v1"
CHECKPOINT_SCHEMA = "m1_projadd_learnable_epoch_checkpoint_v1"
PROGRESS_SCHEMA = "m1_projadd_learnable_score_progress_v1"
SCORE_SCHEMA = "m1_projadd_learnable_ho_calib_epoch_scan_v1"

_RIFT_MODULE = None


def rift_module():
    """Privately load the frozen rift_v1 M1 runner and bind its constants."""
    global _RIFT_MODULE
    if _RIFT_MODULE is None:
        spec = importlib.util.spec_from_file_location("_rift_v1_m1_train_private", RIFT_SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot privately load scripts/rift_v1/m1_train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in (
            "SEED", "CONTEXT", "PROJ_DIM", "EPOCHS", "BATCH", "LR", "QUERY_PAD_BINS",
            "UPDATES_PER_EPOCH", "EXPECTED_WINDOWS", "HO", "SOURCE_SESSIONS",
        ):
            if getattr(module, name, None) != globals()[name]:
                raise RuntimeError(f"rift_v1 m1_train constant {name} drifted against this runner")
        _RIFT_MODULE = module
    return _RIFT_MODULE


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return _sha_file(path)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def shared_init_sha(model: nn.Module, names: list[str]) -> str:
    digest = hashlib.sha256()
    named = dict(model.named_parameters())
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(named[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def learnable_decoder(device: torch.device, recency_cfg, proj_dim: int = PROJ_DIM) -> LearnableRiftDecoder:
    # P8 is a rank-8 truncation of the P16 build (the v1 operator requires a
    # multiple of 16); see learnable_recency_v1.p8.
    build_dim = 16 if proj_dim == 8 else proj_dim
    model = LearnableRiftDecoder("m1", recency_cfg, context_bins=CONTEXT, seed=SEED, proj_dim=build_dim).to(device)
    maybe_truncate_p8(model, proj_dim)
    model.temporal.set_attention_backend("local")
    expected = tuple(recency_cfg.temporal_config.windows)
    if tuple(model.temporal_config.windows) != expected:
        raise RuntimeError(f"M1 R100 D{recency_cfg.layers} windows drifted: {tuple(model.temporal_config.windows)} != {expected}")
    if recency_cfg.layers == 4 and expected != (25, 25, 25, 24):
        raise RuntimeError(f"M1 R100 D4 layer windows drifted: {expected}")
    if int(model.proj_dim) != proj_dim:
        raise RuntimeError(f"M1 proj_add decoder proj_dim drifted: {model.proj_dim} != {proj_dim}")
    return model


def stock_decoder(device: torch.device, layers: int, proj_dim: int = PROJ_DIM) -> RiftDecoder:
    """Same-seed stock RiftDecoder with the DEFAULT (unscaled) ladder."""
    build_dim = 16 if proj_dim == 8 else proj_dim
    model = RiftDecoder("m1", context_bins=CONTEXT, bias_mode="recency", seed=SEED, proj_dim=build_dim).to(device)
    maybe_truncate_p8(model, proj_dim)
    if layers != 4:
        install_temporal(model, dataset_config("m1", tier="fixed", layers=layers, ladder="default"), SEED)
    model.temporal.set_attention_backend("local")
    return model


def fixed_ladder_reference(device: torch.device, recency_cfg, proj_dim: int = PROJ_DIM) -> RiftDecoder:
    """Plain fixed-recency RiftDecoder on the *same* ladder as the learnable run."""
    build_dim = 16 if proj_dim == 8 else proj_dim
    model = RiftDecoder("m1", context_bins=CONTEXT, bias_mode="recency", seed=SEED, proj_dim=build_dim).to(device)
    maybe_truncate_p8(model, proj_dim)
    ladder_cfg = dataset_config(
        "m1", tier="fixed", layers=recency_cfg.layers, half_life_seconds=recency_cfg.half_life_seconds
    )
    install_temporal(model, ladder_cfg, SEED)
    model.temporal.set_attention_backend("local")
    return model


def paired_reference(path: Path) -> dict[str, Any]:
    """Validate the frozen m1_r100_recency_s42_formal_v3 reference and return its bindings."""
    meta = json.loads((path / "run_meta.json").read_text())
    receipt = json.loads((path / "train_receipt.json").read_text())
    expected_meta = {
        "schema": "m1_rift_train_v2", "status": "FORMAL", "cell": PAIRED_CELL, "task": "m1",
        "variant": "recency", "identity_interface": "proj_add", "seed": SEED, "context_bins": CONTEXT,
        "proj_dim": PROJ_DIM, "epochs": EPOCHS, "batch": BATCH, "depth": 4, "width": 256,
        "layer_windows": [25, 25, 25, 24], "attention_backend": "local",
        "updates_per_epoch": UPDATES_PER_EPOCH, "total_updates": EPOCHS * UPDATES_PER_EPOCH,
    }
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise RuntimeError(f"paired reference run_meta {key} mismatch: {meta.get(key)!r} != {value!r}")
    expected_receipt = {"status": "COMPLETED", "cell": PAIRED_CELL, "epochs": EPOCHS, "steps": EPOCHS * UPDATES_PER_EPOCH}
    for key, value in expected_receipt.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"paired reference train_receipt {key} mismatch: {receipt.get(key)!r} != {value!r}")
    summary: dict[str, Any] = {
        "path": str(path), "schema": meta["schema"], "cell": meta["cell"],
        "train_receipt_status": receipt["status"], "train_steps": int(receipt["steps"]),
        "initialization_sha256": meta["initialization_sha256"],
    }
    score_path = path / "score_receipt.json"
    ho_contract = None
    if score_path.is_file():
        score = json.loads(score_path.read_text())
        if score.get("status") == "COMPLETED" and score.get("schema") == "m1_rift_ho_calib_epoch_scan_v2":
            best = str(score["selection"]["epoch"])
            summary["ho3_selection"] = {
                "epoch": int(best),
                "rule": score["selection"]["rule"],
                "equal_session_mean": float(score["ema_by_epoch"][best]["equal_session_mean"]),
            }
            ho_contract = score.get("ho_contract")
    return {"summary": summary, "source_contract": meta["source_contract"], "source_hashes": meta["source_hashes"],
            "initialization_sha256": meta["initialization_sha256"], "ho_contract": ho_contract}


def assert_paired(
    model: nn.Module, device: torch.device, paired: Mapping[str, Any], recency_cfg, proj_dim: int = PROJ_DIM
) -> dict[str, Any]:
    stock = stock_decoder(device, recency_cfg.layers, proj_dim)
    shared = assert_shared_byte_equal(model, stock)
    extra = new_parameter_names(model)
    hashed = shared_init_sha(model, shared)
    # The paired formal reference stays P16; only a P16 D4 run can (and must)
    # reproduce its shared initialization SHA. Non-16 arms pair structurally
    # against the same-seed stock build above.
    if recency_cfg.layers == 4 and proj_dim == PROJ_DIM and hashed != paired["initialization_sha256"]:
        raise RuntimeError("learnable shared named-parameter SHA differs from paired recency reference")
    ladder_ref = fixed_ladder_reference(device, recency_cfg, proj_dim)
    z = torch.randn(1, CONTEXT, model.temporal_config.width, device=device)
    mask = torch.ones(1, CONTEXT, dtype=torch.bool, device=device)
    with torch.inference_mode():
        left = model.temporal(z, mask)
        right = ladder_ref.temporal(z, mask)
    if not torch.allclose(left, right, atol=1e-6, rtol=1e-6):
        raise RuntimeError("M1 proj_add learnable/same-ladder init forward drift")
    del stock, ladder_ref
    return {
        "shared_parameter_names": shared,
        "new_parameter_names": extra,
        "new_parameter_counts": {name: int(dict(model.named_parameters())[name].numel()) for name in extra},
        "trainable_new_parameter_count": trainable_new_parameter_count(model),
        "shared_initialization_sha256": hashed,
        "init_forward_max_abs": float((left - right).abs().max().cpu()),
        "ladder": recency_cfg.ladder_metadata(),
    }


def assert_full_stream_parity(model: nn.Module, material: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Full-window versus token-by-token parity on a left-padded W100 context."""
    from btransform_unified_v2.streaming import RiftStreamDecoder

    item = material[HO[0]]
    x = item["dataset"][0][0]
    raw = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(device)
    valid = rift_module().valid_mask_from_padded_starts((97,), device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            full = model(raw, item["bank"], input_valid_mask=valid)
        stream = LearnableRiftStreamDecoder(model) if isinstance(model.temporal, LearnableRecencyTemporal) else RiftStreamDecoder(model)
        streamed = None
        for offset in range(CONTEXT):
            streamed = stream.stream_step(raw[:, offset], item["bank"], ["m1-projadd-learnable-parity"], valid_mask=valid[:, offset])
        if streamed is None or not torch.allclose(full, streamed, rtol=2e-5, atol=2e-5):
            raise RuntimeError("M1 proj_add learnable full-window versus streaming parity failed")
        return {"status": "PASSED", "query_start": 97, "valid_bins": int(valid.sum())}
    finally:
        model.train(was_training)


def bias_snapshot(model: nn.Module, x: torch.Tensor, bank: Any, valid: torch.Tensor) -> dict[str, Any]:
    was = model.training
    model.eval()
    with torch.inference_mode():
        tokens = model.frontend_tokens(x[:1], bank)
        stats = recency_bias_snapshot(model.temporal, tokens, valid[:1])
    model.train(was)
    return stats


def _checkpoint_payload(model, optimizer, ema, *, epoch: int, step: int, smoke: bool, meta: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {"schema": CHECKPOINT_SCHEMA, "cell": meta["cell"], "epoch": epoch, "global_step": step, "smoke": smoke,
            "tier": meta["tier"], "bias_mode": meta["bias_mode"], "new_parameter_names": meta["new_parameter_names"],
            "config": {"seed": SEED, "context_bins": CONTEXT, "proj_dim": int(meta["proj_dim"]), "epochs": EPOCHS, "batch": BATCH,
                       "lr": LR, "bias_mode": meta["bias_mode"], "attention_backend": "local"},
            "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"],
            "shared_initialization_sha256": meta["initialization_pairing"]["shared_initialization_sha256"],
            "raw_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema.state_dict(),
            "rng": _rng_state(device)}


def _validate_checkpoint(path: Path, meta: Mapping[str, Any], *, expected_epoch: int | None = None) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"schema": CHECKPOINT_SCHEMA, "cell": meta["cell"], "smoke": False, "tier": meta["tier"]}
    if any(state.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"checkpoint contract mismatch: {path}")
    if expected_epoch is not None and int(state.get("epoch", 0)) != expected_epoch:
        raise RuntimeError(f"checkpoint epoch mismatch: {path}")
    config = state.get("config", {})
    if config != {"seed": SEED, "context_bins": CONTEXT, "proj_dim": int(meta["proj_dim"]), "epochs": EPOCHS, "batch": BATCH,
                  "lr": LR, "bias_mode": meta["bias_mode"], "attention_backend": "local"}:
        raise RuntimeError(f"checkpoint configuration mismatch: {path}")
    for key in ("source_hashes", "source_contract", "new_parameter_names"):
        if state.get(key) != meta.get(key):
            raise RuntimeError(f"checkpoint {key} mismatch: {path}")
    if state.get("shared_initialization_sha256") != meta["initialization_pairing"]["shared_initialization_sha256"]:
        raise RuntimeError(f"checkpoint shared initialization sha mismatch: {path}")
    return state


def rebuild_model(meta: Mapping[str, Any], device: torch.device) -> LearnableRiftDecoder:
    """Rebuild the learnable decoder for scoring from the run's recorded proj_dim."""
    recency_cfg = config_from_run_meta(meta, "m1")
    return learnable_decoder(device, recency_cfg, int(meta.get("proj_dim", PROJ_DIM)))


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    recency_cfg = config_from_args(args, "m1")
    proj_dim = int(args.proj_dim)
    smoke = args.max_updates_smoke is not None
    if not smoke and args.epochs != EPOCHS:
        raise ValueError("formal M1 proj_add learnable requires exactly 24 epochs")
    if args.resume is not None:
        raise RuntimeError("resume is not used in this drop; rerun into a fresh --dest")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dest = args.dest.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("new learnable destination must be empty")
    rift = rift_module()
    paired = paired_reference(args.paired_reference.resolve())
    dataset, sampler = rift.legacy.build_fullsession_face()
    banks, report = rift.legacy.build_fullsession_banks(dataset)
    contract = rift._source_contract(dataset, sampler, banks, report)
    if contract != paired["source_contract"]:
        raise RuntimeError("learnable source contract differs from paired recency reference")
    model = learnable_decoder(device, recency_cfg, proj_dim)
    init = assert_paired(model, device, paired, recency_cfg, proj_dim)
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    # Shared parameters keep the reference trainer's name-based no-decay
    # grouping (m1_train.py -> tfpd build_optimizer) so this arm's recipe is
    # identical to the paired formal reference; new recency params are a third
    # group with wd=0 and the tier's lr multiplier.
    new_names = set(new_parameter_names(model))
    shared_named = [(n, p) for n, p in model.named_parameters() if n not in new_names and p.requires_grad]
    extra_named = [(n, p) for n, p in model.named_parameters() if n in new_names and p.requires_grad]
    groups = rift.optimizer_factory.adamw_param_groups(shared_named, weight_decay=plan.WEIGHT_DECAY)
    if extra_named:
        groups.append({"params": [p for _, p in extra_named], "weight_decay": 0.0,
                       "lr_multiplier": recency_cfg.lr_multiplier})
    for group in groups:
        group.setdefault("lr_multiplier", 1.0)
    optimizer = torch.optim.AdamW(groups, lr=LR, betas=rift.optimizer_factory.plan.ADAM_BETAS,
                                  eps=rift.optimizer_factory.plan.ADAM_EPS)
    smoke_preflight = None
    material = None
    if smoke:
        material = rift._ho_material()
        ho = rift._ho_contract(material)
        if paired["ho_contract"] is not None and ho != paired["ho_contract"]:
            raise RuntimeError("learnable HO contract differs from paired recency score receipt")
        smoke_preflight = {
            "initialization_pairing": init,
            "ho_contract": ho,
            "streaming_parity": assert_full_stream_parity(model, material, device),
        }
    meta = {
        "schema": TRAIN_SCHEMA, "status": "SMOKE" if smoke else "FORMAL",
        "cell": f"M1-RIFT-R100-D{recency_cfg.layers}-P{proj_dim}-PROJADD-LEARNABLE-{recency_cfg.tier.upper()}-V1",
        "task": "m1", "identity_interface": "proj_add", "variant": "learnable_recency", "tier": recency_cfg.tier,
        "bias_mode": "recency" if recency_cfg.tier == "fixed" else f"learnable_{recency_cfg.tier}",
        "learnable_config": recency_cfg.__dict__,
        "new_parameter_names": init["new_parameter_names"],
        "new_parameter_counts": init["new_parameter_counts"],
        "trainable_new_parameter_count": init["trainable_new_parameter_count"],
        "seed": SEED, "sampler_seed": SEED, "context_bins": CONTEXT, "query_pad_bins": QUERY_PAD_BINS,
        "proj_dim": proj_dim, "layer_windows": list(recency_cfg.temporal_config.windows), "depth": recency_cfg.layers,
        "width": int(model.temporal_config.width), "ladder": recency_cfg.ladder_metadata(), "attention_backend": "local",
        "epochs": args.epochs, "batch": BATCH, "updates_per_epoch": UPDATES_PER_EPOCH,
        "total_updates": EPOCHS * UPDATES_PER_EPOCH,
        "optimizer": {"name": "AdamW", "betas": list(rift.optimizer_factory.plan.ADAM_BETAS),
                      "eps": rift.optimizer_factory.plan.ADAM_EPS, "weight_decay": plan.WEIGHT_DECAY,
                      "new_param_weight_decay": 0.0, "clip": 1.0,
                      "grouping": "reference name-based no-decay groups (shared) + wd=0 new-recency group"},
        "lr": {"peak": LR, "min": LR * plan.LR_MIN_FACTOR, "warmup_updates": UPDATES_PER_EPOCH,
               "new_param_multiplier": recency_cfg.lr_multiplier},
        "ema_decay": plan.EMA_DECAY, "unit_dropout": 0.1,
        "initialization_pairing": init,
        "paired_reference": paired["summary"],
        "source_train_only_for_gradients": True,
        "development_surface": "held-out-calib M10, post-training EMA scan only",
        "official_test_used": False,
        "source_hashes": dict(rift._source_hashes()),
        "runner_sha256": _sha_file(Path(__file__)),
        "source_contract": dict(contract),
        "launch": {"argv": sys.argv, "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "python_no_user_site": os.environ.get("PYTHONNOUSERSITE")},
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    _atomic_json(dest / "run_meta.json", meta)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=rift.legacy._collate, num_workers=0)
    started = time.monotonic()
    step, last_bias, last_loss = 0, None, None
    for epoch in range(1, args.epochs + 1):
        model.train(); losses: list[float] = []
        last_x = last_bank = last_valid = None
        for batch_id, (x, y, sessions) in enumerate(loader):
            if any(session != sessions[0] for session in sessions):
                raise RuntimeError("M1 source sampler produced mixed-session batch")
            step += 1
            lr = warmup_cosine_lr(step, total_steps=EPOCHS * UPDATES_PER_EPOCH, warmup_steps=UPDATES_PER_EPOCH,
                                  peak=LR, min_factor=plan.LR_MIN_FACTOR)
            apply_group_lrs(optimizer, lr)
            keep_rng = torch.Generator(device="cpu"); keep_rng.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
            keep = whole_unit_dropout(banks[sessions[0]].unit_mask, p=0.1, generator=keep_rng)
            valid = torch.ones((len(x), CONTEXT), dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                prediction = model(x.float().to(device), banks[sessions[0]], dropout_keep=keep, input_valid_mask=valid)
                loss = nn.functional.mse_loss(prediction.float(), y.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite learnable loss epoch={epoch} batch={batch_id}")
            loss.backward(); grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            optimizer.step(); ema.update_after_step(model)
            losses.append(float(loss.detach().cpu()))
            last_loss = losses[-1]
            last_x, last_bank, last_valid = x.float().to(device), banks[sessions[0]], valid
            if step == 1 or step % 100 == 0:
                _atomic_json(dest / "heartbeat.json", {"status": "SMOKE" if smoke else "TRAINING", "pid": os.getpid(),
                                                       "event": "step", "epoch": epoch, "global_step": step,
                                                       "loss": losses[-1], "lr": lr, "grad_norm": grad_norm,
                                                       "elapsed_seconds": time.monotonic() - started,
                                                       "utc": datetime.now(timezone.utc).isoformat()})
            if smoke and step >= args.max_updates_smoke:
                break
        last_bias = bias_snapshot(model, last_x, last_bank, last_valid)
        checkpoint = _checkpoint_payload(model, optimizer, ema, epoch=epoch, step=step, smoke=smoke, meta=meta, device=device)
        checkpoint_sha = _atomic_checkpoint(dest / f"epoch_{epoch:03d}.pt", checkpoint)
        row = {"status": "SMOKE" if smoke else "TRAINING", "event": "epoch", "epoch": epoch, "global_step": step,
               "train_mse": float(np.mean(losses)), "checkpoint_sha256": checkpoint_sha, "bias": last_bias,
               "utc": datetime.now(timezone.utc).isoformat()}
        _atomic_json(dest / "heartbeat.json", row); _append_jsonl(dest / "metrics.jsonl", row)
        if smoke:
            receipt = {
                "schema": TRAIN_SCHEMA, "status": "COMPLETED", "cell": meta["cell"], "tier": recency_cfg.tier,
                "steps": step, "finite_loss": last_loss is not None and bool(np.isfinite(last_loss)),
                "new_parameter_names": init["new_parameter_names"],
                "new_parameter_counts": init["new_parameter_counts"],
                "trainable_new_parameter_count": init["trainable_new_parameter_count"],
                "preflight": smoke_preflight,
                "postupdate": {"full_vs_stream": assert_full_stream_parity(model, material, device)},
                "bias": last_bias, "checkpoint_sha256": checkpoint_sha,
                "cuda_initialized": torch.cuda.is_initialized(),
            }
            _atomic_json(dest / "smoke_receipt.json", receipt)
            return {"status": "SMOKE_COMPLETED", "steps": step,
                    "trainable_new_parameter_count": init["trainable_new_parameter_count"]}
        if len(losses) != UPDATES_PER_EPOCH:
            raise RuntimeError(f"epoch {epoch} had {len(losses)} updates, expected {UPDATES_PER_EPOCH}")
    if step != EPOCHS * UPDATES_PER_EPOCH:
        raise RuntimeError(f"formal total updates {step} != {EPOCHS * UPDATES_PER_EPOCH}")
    _atomic_json(dest / "train_receipt.json", {"schema": TRAIN_SCHEMA, "status": "COMPLETED", "cell": meta["cell"],
                                               "tier": recency_cfg.tier, "epochs": EPOCHS, "steps": step,
                                               "paired_reference": meta["paired_reference"],
                                               "post_training_scoring_required": True})
    return {"status": "TRAIN_COMPLETED", "steps": step}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    dest = args.dest.resolve()
    meta = json.loads((dest / "run_meta.json").read_text())
    if meta.get("status") != "FORMAL" or meta.get("schema") != TRAIN_SCHEMA or meta.get("task") != "m1":
        raise RuntimeError("score requires matching formal M1 proj_add learnable run")
    recency_cfg = config_from_run_meta(meta, "m1")
    rift = rift_module()
    paired = paired_reference(args.paired_reference.resolve())
    if meta.get("paired_reference", {}).get("cell") != PAIRED_CELL:
        raise RuntimeError("run was not paired against the rift_v1 recency reference")
    if meta.get("paired_reference", {}).get("initialization_sha256") != paired["initialization_sha256"]:
        raise RuntimeError("paired recency reference initialization SHA changed since training")
    if meta.get("source_hashes") != rift._source_hashes():
        raise RuntimeError("score source hashes differ from the frozen rift_v1 pipeline")
    if meta.get("source_contract", {}).get("total_windows") != EXPECTED_WINDOWS:
        raise RuntimeError("score source contract mismatch")
    receipt = json.loads((dest / "train_receipt.json").read_text())
    required_receipt = {"status": "COMPLETED", "cell": meta["cell"], "epochs": EPOCHS, "steps": EPOCHS * UPDATES_PER_EPOCH}
    if any(receipt.get(key) != value for key, value in required_receipt.items()):
        raise RuntimeError("score requires a completed matching formal train receipt")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    model = rebuild_model(meta, device)
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    material = rift._ho_material()
    ho_contract = rift._ho_contract(material)
    progress_path = dest / "score_progress.json"
    initial_progress = {"schema": PROGRESS_SCHEMA, "cell": meta["cell"], "ho_contract": ho_contract, "completed": {}}
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else initial_progress
    if any(progress.get(key) != initial_progress[key] for key in ("schema", "cell", "ho_contract")):
        raise RuntimeError("score progress provenance mismatch")
    completed = progress.get("completed", {})
    if not isinstance(completed, dict) or not set(completed).issubset({str(epoch) for epoch in range(1, EPOCHS + 1)}):
        raise RuntimeError("malformed score progress")
    repeat: dict[str, Any] | None = None
    for epoch in range(1, EPOCHS + 1):
        path = dest / f"epoch_{epoch:03d}.pt"
        state = _validate_checkpoint(path, meta, expected_epoch=epoch)
        checkpoint_sha = _sha_file(path)
        model.load_state_dict(state["raw_state_dict"], strict=True)
        ema.load_state_dict(state["ema"])
        if repeat is None:
            repeat = rift._assert_ho_repeatable(model, ema, material, device)
            repeat["full_vs_stream_parity"] = assert_full_stream_parity(model, material, device)
        previous = completed.get(str(epoch))
        if previous is not None:
            if previous.get("checkpoint_sha256") != checkpoint_sha:
                raise RuntimeError(f"checkpoint hash drift after scored epoch {epoch}")
            rift._validate_scored_report(previous.get("ema_ho_calib", {}))
            continue
        report = rift._with_ema_eval(model, ema, lambda: rift._ho_score(model, material, device))
        rift._validate_scored_report(report)
        completed[str(epoch)] = {"checkpoint_sha256": checkpoint_sha, "ema_ho_calib": report}
        _atomic_json(progress_path, {**initial_progress, "status": "SCORING", "completed": completed,
                                     "last_completed_epoch": epoch, "repeatability": repeat})
        _atomic_json(dest / "heartbeat.json", {"status": "SCORING", "event": "ho_calib_score", "epoch": epoch,
                                               "completed_epochs": len(completed),
                                               "utc": datetime.now(timezone.utc).isoformat()})
    if set(completed) != {str(epoch) for epoch in range(1, EPOCHS + 1)}:
        raise RuntimeError("score scan did not produce exactly 24 epochs")
    scores = {epoch: row["ema_ho_calib"] for epoch, row in completed.items()}
    best = min(range(1, EPOCHS + 1), key=lambda epoch: (-scores[str(epoch)]["equal_session_mean"], epoch))
    score_receipt = {"schema": SCORE_SCHEMA, "status": "COMPLETED", "cell": meta["cell"], "tier": recency_cfg.tier,
                     "ema_by_epoch": scores,
                     "checkpoint_sha256_by_epoch": {epoch: completed[epoch]["checkpoint_sha256"] for epoch in sorted(completed, key=int)},
                     "selection": {"epoch": best, "rule": "earliest maximum equal-session mean EMA"},
                     "repeatability": repeat, "ho_contract": ho_contract,
                     "paired_reference": meta["paired_reference"], "official_test_used": False}
    _atomic_json(dest / "score_receipt.json", score_receipt)
    _atomic_json(dest / "ho_calib_epoch_scan.json", score_receipt)
    _atomic_json(dest / "heartbeat.json", {"status": "COMPLETED", "event": "score_complete", "epoch": best,
                                           "completed_epochs": EPOCHS,
                                           "utc": datetime.now(timezone.utc).isoformat()})
    return {"status": "SCORE_COMPLETED", "best_epoch": best}


def default_dest(args: argparse.Namespace) -> Path:
    """P16 keeps the historical default name; other proj_dims carry the run knob."""
    suffix = "" if int(args.layers) == 4 else f"_layers{int(args.layers)}"
    if args.proj_dim == PROJ_DIM:
        name = f"m1_projadd_{args.tier}_s{args.seed}{suffix}"
    else:
        name = f"m1_projadd_{args.tier}_{config_from_args(args, 'm1').ladder}_p{args.proj_dim}_s{args.seed}{suffix}"
    return (RESULTS / "smoke" / f"{name}_v1") if args.max_updates_smoke is not None else (RESULTS / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_learnable_flags(parser)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--stage", choices=("train", "score"), default="train")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--proj-dim", type=int, default=PROJ_DIM)
    parser.add_argument("--paired-reference", type=Path, default=PAIRED_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates-smoke", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > EPOCHS:
        parser.error("--epochs must be 1..24")
    if args.proj_dim <= 0:
        parser.error("proj_dim must be positive")
    if args.max_updates_smoke is not None and (args.max_updates_smoke < 1 or args.stage != "train"):
        parser.error("smoke requires positive updates and --stage train")
    if args.stage == "score" and args.max_updates_smoke is not None:
        parser.error("smoke is a train-stage-only mode")
    if args.dest is None:
        args.dest = default_dest(args)
    if args.max_updates_smoke is not None and args.device == "cuda:0":
        args.device = "cpu"
    result = run_train(args) if args.stage == "train" else run_score(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
