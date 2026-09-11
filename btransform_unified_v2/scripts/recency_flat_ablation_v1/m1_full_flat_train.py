#!/usr/bin/env python3
"""Isolated M1 FULL muscle-carrier flat-versus-recency ablation runner.

This file deliberately imports the frozen original FULL runner privately and reuses
its data, sampler, carrier, evaluation, parity, and metric helpers.  It never
changes the frozen module's globals or files.  The only model-level difference
from the paired recency formal run is the all-zero temporal recency-slope buffer.
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
ROOT = HERE.parents[1]  # btransform_unified_v2
WS = ROOT.parent
FROZEN_RUNNER = ROOT / "scripts/m1_muscle_r100_v1/train.py"
FROZEN_DIR = FROZEN_RUNNER.parent
MODEL_FILE = HERE / "m1_flat_model.py"
OUTPUT_ROOT = ROOT / "results/recency_flat_ablation_v1"
DEFAULT_REFERENCE = ROOT / "results/m1_muscle_r100_v1/formal_s42_gpu1"
DEFAULT_PACK = ROOT / "results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz"

# Keep the frozen runner's carrier facade ahead of this directory.  The private
# import is needed only to call its helpers; no attributes in it are reassigned.
for candidate in (FROZEN_DIR, ROOT / "src", ROOT.parent / "btransform_unified_v1/src", ROOT.parent / "btransform_unified_v1/scripts", WS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
spec = importlib.util.spec_from_file_location("_frozen_m1_full_runner", FROZEN_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot privately load frozen M1 FULL runner")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
import carrier as frozen_carrier  # resolves to FROZEN_DIR/carrier.py above
if Path(frozen_carrier.__file__).resolve() != (FROZEN_DIR / "carrier.py").resolve():
    raise RuntimeError("flat runner did not bind the frozen carrier facade")

from btransform_unified_v1 import plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from tfpd_exploration.src.m2_dual_track_v1 import training as optimizer_factory
from btransform_unified_v2.joint_m1_model import ARMS
from m1_flat_model import FlatJointM1ConcatDecoder

CELL = "M1-MUSCLE-R100-D4-JOINT-B3S-CONCAT-FULL-FLAT-V1"
SCHEMA = "m1_muscle_r100_full_flat_train_v1"
CHECKPOINT_SCHEMA = "m1_muscle_r100_full_flat_epoch_checkpoint_v1"
SCORE_SCHEMA = "m1_muscle_r100_full_flat_ho_calib_epoch_scan_v1"
PROGRESS_SCHEMA = "m1_muscle_r100_full_flat_score_progress_v1"
FROZEN_SAMPLER_SEED = frozen.SEED
CONTEXT, EPOCHS, BATCH, LR = frozen.CONTEXT, frozen.EPOCHS, frozen.BATCH, frozen.LR
UPDATES_PER_EPOCH, EXPECTED_WINDOWS = frozen.UPDATES_PER_EPOCH, frozen.EXPECTED_WINDOWS
HO = frozen.HO
SOURCE = frozen.SOURCE_SESSIONS


def _sha_file(path: Path) -> str:
    return frozen._sha_file(path)


def _atomic_json(path: Path, payload: Any) -> None:
    frozen._atomic_json(path, payload)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    return frozen._atomic_checkpoint(path, payload)


def _initialization_sha(model: nn.Module) -> str:
    """The paired SHA: named parameters only, intentionally excluding buffers."""
    return frozen._initialization_sha(model)


def _flat_slopes(model: nn.Module, *, stage: str) -> dict[str, Any]:
    slopes = model.temporal.recency_slopes.detach().cpu()
    if slopes.shape != (8,) or slopes.dtype != torch.float32:
        raise RuntimeError(f"{stage}: flat recency_slopes shape/dtype drift")
    if not bool(torch.isfinite(slopes).all()) or int(torch.count_nonzero(slopes)) != 0:
        raise RuntimeError(f"{stage}: flat recency_slopes must be finite zeros")
    array = slopes.contiguous().numpy()
    return {
        "state_dict_key": "temporal.recency_slopes",
        "shape": [8], "dtype": "float32", "zero_count": 8,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _source_hashes() -> dict[str, str]:
    """Frozen dependency binding plus both isolated flat sources."""
    values = dict(frozen._source_hashes())
    values[str(Path(__file__).resolve())] = _sha_file(Path(__file__).resolve())
    values[str(MODEL_FILE.resolve())] = _sha_file(MODEL_FILE.resolve())
    return values


def _carrier_binding(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Bind the exact official carrier, retaining its original receipt fields.

    The new-family source hashes are deliberately additional fields: they bind
    the ablation checkpoints without claiming that their code made the formal
    reference run.
    """
    carriers, original_binding = frozen._carrier_binding(path)
    binding = dict(original_binding)
    binding.pop("binding_sha256", None)
    binding["carrier_variant"] = "muscle_response16_svd4/global_rms"
    binding["original_full_runner_py_sha256"] = _sha_file(FROZEN_RUNNER)
    binding["full_flat_runner_py_sha256"] = _sha_file(Path(__file__).resolve())
    binding["flat_model_py_sha256"] = _sha_file(MODEL_FILE.resolve())
    binding["temporal_bias_mode"] = "flat"
    binding["binding_sha256"] = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return carriers, binding


def _decoder(device: torch.device, arm: str, seed: int) -> FlatJointM1ConcatDecoder:
    model = FlatJointM1ConcatDecoder(arm, seed=seed).to(device)
    model.temporal.set_attention_backend("local")
    if model.bias_mode != "flat" or model.temporal.config.bias_mode != "flat":
        raise RuntimeError("flat decoder construction drift")
    if tuple(model.temporal_config.windows) != (25, 25, 25, 24):
        raise RuntimeError("M1 R100 D4 layer windows drifted")
    _flat_slopes(model, stage="decoder construction")
    return model


def _read_json(path: Path) -> dict[str, Any]:
    return frozen._read_json(path)


def _paired_reference(path: Path, *, arm: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    root = path.resolve()
    meta = _read_json(root / "run_meta.json")
    train = _read_json(root / "train_receipt.json")
    score = _read_json(root / "score_receipt.json")
    if (meta.get("schema") != "m1_muscle_r100_train_v1" or meta.get("status") != "FORMAL" or
            meta.get("cell") != "M1-MUSCLE-R100-D4-JOINT-B3S-CONCAT-V1"):
        raise RuntimeError("paired reference must be the original completed formal M1 FULL run")
    if meta.get("arm") != arm or meta.get("seed") != 42 or meta.get("sampler_seed") != FROZEN_SAMPLER_SEED:
        raise RuntimeError("paired reference arm/seed/sampler mismatch")
    expected_optimizer = {"name": "AdamW", "weight_decay": plan.WEIGHT_DECAY, "clip": 1.0}
    expected_lr = {"peak": LR, "min": LR * plan.LR_MIN_FACTOR, "warmup_updates": UPDATES_PER_EPOCH}
    if (meta.get("epochs") != EPOCHS or meta.get("updates_per_epoch") != UPDATES_PER_EPOCH or
            meta.get("total_updates") != EPOCHS * UPDATES_PER_EPOCH or meta.get("batch") != BATCH or
            meta.get("context_bins") != CONTEXT or meta.get("layer_windows") != [25, 25, 25, 24] or
            meta.get("depth") != 4 or meta.get("width") != 256 or meta.get("attention_backend") != "local" or
            meta.get("optimizer") != expected_optimizer or meta.get("lr") != expected_lr or
            meta.get("ema_decay") != plan.EMA_DECAY or meta.get("unit_dropout") != 0.1):
        raise RuntimeError("paired reference optimization/configuration mismatch")
    if (train.get("status") != "COMPLETED" or train.get("epochs") != EPOCHS or
            train.get("steps") != EPOCHS * UPDATES_PER_EPOCH):
        raise RuntimeError("paired reference is not a completed 24-epoch formal train")
    if meta.get("source_hashes") != frozen._source_hashes():
        raise RuntimeError("paired reference frozen shared source hashes drifted")
    ref_binding = meta.get("carrier_binding")
    carrier_fields = ("carrier_pack_npz_sha256", "carrier_pack_receipt_sha256",
                      "carrier_pack_receipt_body", "fit_sha256", "carrier_py_sha256")
    if not isinstance(ref_binding, Mapping) or any(ref_binding.get(key) != binding.get(key) for key in carrier_fields):
        raise RuntimeError("paired reference carrier pack or fit does not match requested flat input")
    if meta.get("fit_sha256") != binding.get("fit_sha256") or train.get("fit_sha256") != binding.get("fit_sha256"):
        raise RuntimeError("paired reference fit binding mismatch")
    init_sha = meta.get("initialization_sha256")
    if not isinstance(init_sha, str) or len(init_sha) != 64:
        raise RuntimeError("paired reference lacks named-parameter initialization SHA")
    if score.get("status") != "COMPLETED" or score.get("official_test_used") is not False:
        raise RuntimeError("paired reference score receipt is not a completed HO-only scan")
    selection = score.get("selection")
    if (score.get("schema") != "m1_muscle_r100_ho_calib_epoch_scan_v1" or
            not isinstance(selection, Mapping) or selection.get("epoch") != 3 or
            selection.get("metric") != "channel_variance_weighted_r2" or
            score.get("checkpoint_sha256_by_epoch", {}).get("3") !=
            "f4b248b79addcb72df32c94c4f2887af6e4ddc4cf5269adf5cae03227a82227a"):
        raise RuntimeError("paired reference is not the recorded 582205 selected e3 source")
    return {
        "path": str(root), "run_meta_sha256": _sha_file(root / "run_meta.json"),
        "train_receipt_sha256": _sha_file(root / "train_receipt.json"),
        "score_receipt_sha256": _sha_file(root / "score_receipt.json"),
        "initialization_named_parameters_sha256": init_sha,
        "source_contract": meta.get("source_contract"), "ho_contract": score.get("ho_contract"),
        "carrier_binding": meta.get("carrier_binding"), "fit_sha256": meta.get("fit_sha256"),
        "mainline": "EvalAI submission 582205; original FULL formal e3 reference",
        "reference_selection": dict(selection),
        "reference_selected_checkpoint_sha256": score["checkpoint_sha256_by_epoch"]["3"],
    }


def _checkpoint_payload(model: nn.Module, optimizer: Any, ema: DecoderEMA, *, epoch: int,
                        step: int, smoke: bool, meta: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    slopes = _flat_slopes(model, stage=f"checkpoint epoch {epoch}")
    return {
        "schema": CHECKPOINT_SCHEMA, "cell": CELL, "epoch": epoch, "global_step": step, "smoke": smoke,
        "config": {"arm": meta["arm"], "seed": meta["seed"], "sampler_seed": FROZEN_SAMPLER_SEED,
                   "context_bins": CONTEXT, "proj_dim": None, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                   "bias_mode": "flat", "attention_backend": "local", "b3s": meta["b3s"]},
        "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"],
        "initialization_named_parameters_sha256": meta["initialization_named_parameters_sha256"],
        "flat_slopes": slopes, "seed_scope": meta["seed_scope"], "carrier_binding": meta["carrier_binding"],
        "fit_sha256": meta["carrier_binding"]["fit_sha256"], "paired_reference": meta["paired_reference"],
        "raw_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema.state_dict(),
        "rng": frozen._rng_state(device),
    }


def _validate_checkpoint(path: Path, meta: Mapping[str, Any], *, expected_epoch: int | None = None) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if {key: state.get(key) for key in ("schema", "cell", "smoke")} != {
        "schema": CHECKPOINT_SCHEMA, "cell": CELL, "smoke": False
    }:
        raise RuntimeError(f"checkpoint contract mismatch: {path}")
    epoch = int(state.get("epoch", 0))
    if expected_epoch is not None and epoch != expected_epoch:
        raise RuntimeError(f"checkpoint epoch mismatch: {path}")
    if epoch < 1 or int(state.get("global_step", -1)) != epoch * UPDATES_PER_EPOCH:
        raise RuntimeError(f"checkpoint step/epoch mismatch: {path}")
    expected_config = {"arm": meta["arm"], "seed": meta["seed"], "sampler_seed": FROZEN_SAMPLER_SEED,
                       "context_bins": CONTEXT, "proj_dim": None, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                       "bias_mode": "flat", "attention_backend": "local", "b3s": meta["b3s"]}
    if state.get("config") != expected_config:
        raise RuntimeError(f"checkpoint flat configuration mismatch: {path}")
    for key in ("source_hashes", "source_contract", "initialization_named_parameters_sha256", "flat_slopes",
                "seed_scope", "carrier_binding", "paired_reference"):
        if state.get(key) != meta.get(key):
            raise RuntimeError(f"checkpoint {key} mismatch: {path}")
    if state.get("fit_sha256") != meta["carrier_binding"]["fit_sha256"]:
        raise RuntimeError(f"checkpoint carrier fit SHA mismatch: {path}")
    raw = state.get("raw_state_dict")
    if not isinstance(raw, Mapping) or "temporal.recency_slopes" not in raw:
        raise RuntimeError(f"checkpoint lacks flat slope buffer: {path}")
    slopes = raw["temporal.recency_slopes"]
    if not torch.is_tensor(slopes) or slopes.shape != (8,) or slopes.dtype != torch.float32 or int(torch.count_nonzero(slopes)) != 0:
        raise RuntimeError(f"checkpoint has non-flat recency slopes: {path}")
    return state


def _init_parity_flat(dataset: Any, banks: Mapping[str, Any], source_calib: Mapping[str, Any],
                      device: torch.device, *, seed: int) -> dict[str, Any]:
    """Retain the old live-B3S/full-concat init test with both sides flat."""
    from btransform_unified_v2.concat_model import RiftConcatDecoder
    from btransform_unified_v2.joint_m1_model import ARM_D
    x, _y, sessions = frozen.legacy._collate([dataset[0], dataset[1]])
    session = sessions[0]
    if any(name != session for name in sessions):
        raise RuntimeError("flat init parity batch session drift")
    valid = torch.ones((len(x), CONTEXT), dtype=torch.bool, device=device)
    standalone = RiftConcatDecoder("m1", context_bins=CONTEXT, bias_mode="flat", seed=seed).to(device).eval()
    joint = FlatJointM1ConcatDecoder(ARM_D, seed=seed).to(device)
    joint.install_session_memory(banks, source_calib); joint.to(device); joint.eval()
    _flat_slopes(standalone, stage="standalone init parity")
    _flat_slopes(joint, stage="joint init parity")
    shared = dict(standalone.named_parameters())
    pairs = [(name, value) for name, value in joint.named_parameters() if name in shared and value.shape == shared[name].shape]
    with torch.inference_mode():
        left = standalone(x.to(device), banks[session], input_valid_mask=valid)
        right = joint(x.to(device), banks[session], input_valid_mask=valid)
    maximum = float((left - right).abs().max())
    if maximum >= 2e-6 or not all(torch.equal(value, shared[name]) for name, value in pairs):
        raise RuntimeError("flat full-forward concat init parity failed")
    return {"status": "PASSED", "comparison": "flat_concat_vs_live_flat_joint_d", "seed": seed,
            "session": session, "batch_size": len(x), "output_max_abs": maximum, "threshold": 2e-6,
            "shared_same_parameter_count": len(pairs), "shared_all_byte_equal": True,
            "standalone_slopes": _flat_slopes(standalone, stage="standalone init parity result"),
            "joint_slopes": _flat_slopes(joint, stage="joint init parity result")}


def _meta(args: argparse.Namespace, *, smoke: bool, contract: Mapping[str, Any], binding: Mapping[str, Any],
          paired: Mapping[str, Any], model: nn.Module) -> dict[str, Any]:
    init_sha = _initialization_sha(model)
    if init_sha != paired["initialization_named_parameters_sha256"]:
        raise RuntimeError("flat named-parameter initialization SHA differs from paired recency reference")
    slopes = _flat_slopes(model, stage="initialization")
    return {
        "schema": SCHEMA, "status": "SMOKE" if smoke else "FORMAL", "cell": CELL, "task": "m1",
        "arm": args.arm, "variant": "flat", "identity_interface": "live_b3s_concat",
        "identity_e0_dim": 100, "concat_token_width": 120, "seed": args.seed,
        "sampler_seed": FROZEN_SAMPLER_SEED, "seed_scope": {
            "training_seed": args.seed, "frozen_sampler_seed": FROZEN_SAMPLER_SEED,
            "sampler": "SessionBatchSampler shuffle=True, balance=False, reshuffle_each_epoch=False",
            "model_initialization": "FlatJointM1ConcatDecoder(seed=42); shared named parameters with recency",
            "unit_dropout": "unit_dropout_seed(42, epoch, batch_id)",
            "b3s_initialization": "fixed B3 Sfix e11 copied with zero side columns; parameters remain trainable",
        }, "context_bins": CONTEXT, "query_pad_bins": CONTEXT - 1, "layer_windows": [25, 25, 25, 24],
        "depth": 4, "width": 256, "proj_dim": None, "attention_backend": "local", "bias_mode": "flat",
        "flat_slopes": slopes, "initialization_named_parameters_sha256": init_sha,
        "epochs": args.epochs, "batch": BATCH, "updates_per_epoch": UPDATES_PER_EPOCH,
        "total_updates": EPOCHS * UPDATES_PER_EPOCH, "optimizer": {"name": "AdamW", "weight_decay": plan.WEIGHT_DECAY, "clip": 1.0},
        "lr": {"peak": LR, "min": LR * plan.LR_MIN_FACTOR, "warmup_updates": UPDATES_PER_EPOCH},
        "ema_decay": plan.EMA_DECAY, "unit_dropout": 0.1, "source_train_only_for_gradients": True,
        "development_surface": "visible HO3 calibration (3881 windows), post-training EMA1..24 scan only",
        "official_selection_metric": "equal-session mean channel-centered variance-weighted R2", "official_test_used": False,
        "source_hashes": _source_hashes(), "source_contract": dict(contract), "b3s": frozen._b3s_provenance(),
        "carrier_binding": dict(binding), "fit_sha256": binding["fit_sha256"], "paired_reference": dict(paired),
        "launch": {"argv": sys.argv, "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "python_no_user_site": os.environ.get("PYTHONNOUSERSITE")}, "utc": datetime.now(timezone.utc).isoformat(),
    }


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    smoke = args.max_updates_smoke is not None
    if args.seed != 42 or (not smoke and args.epochs != EPOCHS):
        raise ValueError("flat formal run requires seed42 and exactly 24 epochs")
    device = torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    dest = args.dest.resolve()
    if args.resume is None and dest.exists() and any(dest.iterdir()):
        raise FileExistsError("new flat destination must be empty")
    carriers, binding = _carrier_binding(args.carrier_pack)
    if binding["carrier_variant"] != "muscle_response16_svd4/global_rms":
        raise RuntimeError("flat ablation accepts only the frozen official muscle-response carrier pack")
    paired = _paired_reference(args.paired_reference, arm=args.arm, binding=binding)
    dataset, sampler = frozen.legacy.build_fullsession_face(); legacy_banks, legacy_report = frozen.legacy.build_fullsession_banks(dataset)
    banks = frozen._replace_carriers(legacy_banks, carriers, SOURCE)
    contract = frozen._source_contract(dataset, sampler, banks, {
        "legacy_frozen_activity_bank_report": legacy_report,
        "carrier_replacement": binding["carrier_variant"],
        "actual_carrier_metadata": "source_contract.bank_hashes + carrier_binding",
    })
    source_calib = frozen.m1_plan.calib_trials_from_dataset(dataset)
    contract["raw_m10_calib_sha256"] = {name: frozen._array_sha(value) for name, value in source_calib.items()}
    if contract != paired["source_contract"]:
        raise RuntimeError("flat source contract differs from paired recency reference")
    model = _decoder(device, args.arm, 42); model.install_session_memory(banks, source_calib); model.to(device)
    meta = _meta(args, smoke=smoke, contract=contract, binding=binding, paired=paired, model=model)
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    optimizer = optimizer_factory.build_optimizer(model.named_parameters(), lr=LR, weight_decay=plan.WEIGHT_DECAY)
    smoke_preflight: dict[str, Any] | None = None
    if smoke:
        material = frozen._ho_material(carriers); ho = frozen._ho_contract(material)
        if ho != paired["ho_contract"]:
            raise RuntimeError("flat HO contract differs from paired recency reference")
        model.install_session_memory({s: material[s]["bank"] for s in HO}, {s: material[s]["calib10"] for s in HO}); model.to(device)
        smoke_preflight = {"flat_init_parity": _init_parity_flat(dataset, banks, source_calib, device, seed=42),
                           "streaming_parity": frozen._assert_full_stream_parity(model, material, device),
                           "repeatable_eval": frozen._assert_ho_repeatable(model, ema, material, device),
                           "flat_slopes": _flat_slopes(model, stage="smoke preflight")}
    dest.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        _atomic_json(dest / "run_meta.json", meta); step, first_epoch = 0, 1
    else:
        existing = _read_json(dest / "run_meta.json")
        keys = ("cell", "arm", "seed", "sampler_seed", "source_hashes", "source_contract", "flat_slopes",
                "initialization_named_parameters_sha256", "seed_scope", "carrier_binding", "fit_sha256", "paired_reference")
        if existing.get("status") != "FORMAL" or any(existing.get(key) != meta[key] for key in keys):
            raise RuntimeError("flat resume metadata mismatch")
        if args.resume.resolve().parent != dest:
            raise RuntimeError("resume checkpoint must belong directly to --dest")
        state = _validate_checkpoint(args.resume.resolve(), existing)
        if int(state["epoch"]) >= EPOCHS:
            raise RuntimeError("epoch 24 is complete; run --stage score instead")
        model.load_state_dict(state["raw_state_dict"], strict=True); _flat_slopes(model, stage="resume load")
        optimizer.load_state_dict(state["optimizer"]); ema.load_state_dict(state["ema"]); frozen._restore_rng(state["rng"], device)
        step, first_epoch = int(state["global_step"]), int(state["epoch"]) + 1
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=frozen.legacy._collate, num_workers=0)
    started = time.monotonic()
    all_gradients_finite = True
    for epoch in range(first_epoch, args.epochs + 1):
        model.train(); losses: list[float] = []
        for batch_id, (x, y, sessions) in enumerate(loader):
            if any(session != sessions[0] for session in sessions):
                raise RuntimeError("M1 source sampler produced mixed-session batch")
            step += 1
            lr = frozen.warmup_cosine_lr(step, total_steps=EPOCHS * UPDATES_PER_EPOCH, warmup_steps=UPDATES_PER_EPOCH, peak=LR, min_factor=plan.LR_MIN_FACTOR)
            for group in optimizer.param_groups: group["lr"] = lr
            keep_rng = torch.Generator(device="cpu"); keep_rng.manual_seed(unit_dropout_seed(42, epoch, batch_id))
            keep = whole_unit_dropout(banks[sessions[0]].unit_mask, p=0.1, generator=keep_rng)
            valid = torch.ones((len(x), CONTEXT), dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                prediction = model(x.float().to(device), banks[sessions[0]], dropout_keep=keep, input_valid_mask=valid)
                loss = nn.functional.mse_loss(prediction.float(), y.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite flat loss epoch={epoch} batch={batch_id}")
            loss.backward(); grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            if smoke:
                all_gradients_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                                           for parameter in model.parameters())
                if not all_gradients_finite:
                    raise FloatingPointError("nonfinite flat gradient")
            optimizer.step(); ema.update_after_step(model); losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 100 == 0:
                _atomic_json(dest / "heartbeat.json", {"status": "TRAINING", "event": "step", "epoch": epoch, "global_step": step, "loss": losses[-1], "lr": lr, "grad_norm": grad_norm, "elapsed_seconds": time.monotonic() - started, "utc": datetime.now(timezone.utc).isoformat()})
            if smoke and step >= args.max_updates_smoke:
                break
        postupdate = None
        if smoke and step >= args.max_updates_smoke:
            postupdate = {
                "full_vs_stream_startup_truezero": frozen._assert_full_stream_parity(model, material, device),
                "repeatable_eval": frozen._assert_ho_repeatable(model, ema, material, device),
                "flat_slopes": _flat_slopes(model, stage="smoke post-update"),
            }
        checkpoint = _checkpoint_payload(model, optimizer, ema, epoch=epoch, step=step, smoke=smoke, meta=meta, device=device)
        checkpoint_sha = _atomic_checkpoint(dest / f"epoch_{epoch:03d}.pt", checkpoint)
        row = {"status": "SMOKE" if smoke else "TRAINING", "event": "epoch", "epoch": epoch, "global_step": step, "train_mse": float(np.mean(losses)), "checkpoint_sha256": checkpoint_sha, "elapsed_seconds": time.monotonic() - started, "utc": datetime.now(timezone.utc).isoformat()}
        _atomic_json(dest / "heartbeat.json", row); frozen._append_jsonl(dest / "metrics.jsonl", row)
        if smoke:
            _atomic_json(dest / "smoke_receipt.json", {"schema": SCHEMA, "status": "COMPLETED", "cell": CELL, "variant": "flat", "arm": args.arm, "seed": 42, "sampler_seed": FROZEN_SAMPLER_SEED, "steps": step, "checkpoint_sha256": checkpoint_sha, "flat_slopes": _flat_slopes(model, stage="smoke receipt"), "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "carrier_binding": meta["carrier_binding"], "paired_reference": meta["paired_reference"], "finite_loss": bool(np.isfinite(losses[-1])), "all_gradients_finite": all_gradients_finite, "preflight": smoke_preflight, "postupdate": postupdate})
            return {"status": "SMOKE_COMPLETED", "steps": step}
        if len(losses) != UPDATES_PER_EPOCH:
            raise RuntimeError(f"epoch {epoch} had {len(losses)} updates, expected {UPDATES_PER_EPOCH}")
    if step != EPOCHS * UPDATES_PER_EPOCH:
        raise RuntimeError(f"formal total updates {step} != {EPOCHS * UPDATES_PER_EPOCH}")
    _atomic_json(dest / "train_receipt.json", {"schema": SCHEMA, "status": "COMPLETED", "cell": CELL, "variant": "flat", "arm": args.arm, "seed": 42, "sampler_seed": FROZEN_SAMPLER_SEED, "epochs": EPOCHS, "steps": step, "elapsed_seconds": time.monotonic() - started, "runtime": {"device": str(device), "cpu_threads": args.cpu_threads, "torch_version": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "b3s": meta["b3s"], "carrier_binding": meta["carrier_binding"], "fit_sha256": meta["fit_sha256"], "flat_slopes": meta["flat_slopes"], "initialization_named_parameters_sha256": meta["initialization_named_parameters_sha256"], "paired_reference": meta["paired_reference"], "post_training_scoring_required": True})
    return {"status": "TRAIN_COMPLETED", "steps": step}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    dest = args.dest.resolve(); meta = _read_json(dest / "run_meta.json")
    if meta.get("status") != "FORMAL" or meta.get("schema") != SCHEMA or meta.get("cell") != CELL or meta.get("variant") != "flat" or meta.get("arm") != args.arm or meta.get("seed") != 42:
        raise RuntimeError("score requires matching formal flat M1 run")
    carriers, binding = _carrier_binding(args.carrier_pack)
    paired = _paired_reference(args.paired_reference, arm=args.arm, binding=binding)
    if binding != meta.get("carrier_binding") or paired != meta.get("paired_reference") or meta.get("source_hashes") != _source_hashes():
        raise RuntimeError("flat score provenance mismatch")
    receipt = _read_json(dest / "train_receipt.json")
    if receipt.get("status") != "COMPLETED" or receipt.get("schema") != SCHEMA or receipt.get("steps") != EPOCHS * UPDATES_PER_EPOCH:
        raise RuntimeError("score requires completed matching formal flat train receipt")
    torch.set_num_threads(args.cpu_threads); device = torch.device(args.device)
    material = frozen._ho_material(carriers); ho = frozen._ho_contract(material)
    if ho != meta.get("paired_reference", {}).get("ho_contract"):
        raise RuntimeError("flat HO contract differs from paired recency reference")
    model = _decoder(device, args.arm, 42); model.install_session_memory({s: material[s]["bank"] for s in HO}, {s: material[s]["calib10"] for s in HO}); model.to(device)
    if _initialization_sha(model) != meta.get("initialization_named_parameters_sha256"):
        raise RuntimeError("flat score initialization named-parameter SHA drift")
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    progress_path = dest / "score_progress.json"
    initial = {"schema": PROGRESS_SCHEMA, "cell": CELL, "variant": "flat", "arm": args.arm, "seed": 42, "sampler_seed": FROZEN_SAMPLER_SEED, "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "b3s": meta["b3s"], "ho_contract": ho, "carrier_binding": binding, "flat_slopes": meta["flat_slopes"], "paired_reference": paired, "completed": {}}
    progress = _read_json(progress_path) if progress_path.is_file() else initial
    if any(progress.get(key) != initial[key] for key in initial if key != "completed"):
        raise RuntimeError("flat score progress provenance mismatch")
    completed = progress.get("completed", {})
    if not isinstance(completed, dict) or not set(completed).issubset({str(epoch) for epoch in range(1, EPOCHS + 1)}):
        raise RuntimeError("malformed flat score progress")
    repeat: dict[str, Any] | None = None
    for epoch in range(1, EPOCHS + 1):
        path = dest / f"epoch_{epoch:03d}.pt"; state = _validate_checkpoint(path, meta, expected_epoch=epoch); checkpoint_sha = _sha_file(path)
        model.load_state_dict(state["raw_state_dict"], strict=True); _flat_slopes(model, stage=f"score checkpoint load epoch {epoch}")
        ema.load_state_dict(state["ema"])
        if repeat is None:
            repeat = frozen._assert_ho_repeatable(model, ema, material, device)
            repeat["full_vs_stream_parity"] = frozen._assert_full_stream_parity(model, material, device)
            repeat["flat_slopes"] = _flat_slopes(model, stage="score repeatability")
        previous = completed.get(str(epoch))
        if previous is not None:
            if previous.get("checkpoint_sha256") != checkpoint_sha:
                raise RuntimeError(f"checkpoint hash drift after scored epoch {epoch}")
            frozen._validate_scored_report(previous.get("ema_ho_calib", {}))
            artifact = Path(previous.get("prediction_artifact", {}).get("path", ""))
            if not artifact.is_file() or previous["prediction_artifact"].get("sha256") != _sha_file(artifact):
                raise RuntimeError(f"prediction artifact drift after scored epoch {epoch}")
            continue
        report = frozen._with_ema_eval(model, ema, lambda: frozen._ho_score(model, material, device, capture_predictions=True))
        _flat_slopes(model, stage=f"post-EMA score epoch {epoch}")
        arrays = report.pop("_prediction_arrays"); frozen._validate_scored_report(report)
        artifact = dest / f"ema_ho_epoch_{epoch:03d}_predictions.npz"
        if artifact.exists():
            raise FileExistsError(f"refusing to overwrite prediction artifact: {artifact}")
        np.savez_compressed(artifact, **arrays)
        completed[str(epoch)] = {"checkpoint_sha256": checkpoint_sha, "ema_ho_calib": report, "flat_slopes": _flat_slopes(model, stage=f"score artifact epoch {epoch}"), "prediction_artifact": {"path": str(artifact.resolve()), "sha256": _sha_file(artifact)}}
        _atomic_json(progress_path, {**initial, "status": "SCORING", "completed": completed, "last_completed_epoch": epoch, "repeatability": repeat})
        _atomic_json(dest / "heartbeat.json", {"status": "SCORING", "event": "ho_calib_score", "epoch": epoch, "completed_epochs": len(completed), "utc": datetime.now(timezone.utc).isoformat()})
    if set(completed) != {str(epoch) for epoch in range(1, EPOCHS + 1)}:
        raise RuntimeError("flat score scan did not produce exactly 24 epochs")
    scores = {epoch: row["ema_ho_calib"] for epoch, row in completed.items()}
    best = min(range(1, EPOCHS + 1), key=lambda epoch: (-scores[str(epoch)]["equal_session_mean_channel_variance_weighted_r2"], epoch))
    legacy_best = min(range(1, EPOCHS + 1), key=lambda epoch: (-scores[str(epoch)]["equal_session_mean"], epoch))
    result = {"schema": SCORE_SCHEMA, "status": "COMPLETED", "cell": CELL, "variant": "flat", "arm": args.arm, "seed": 42, "sampler_seed": FROZEN_SAMPLER_SEED, "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "b3s": meta["b3s"], "carrier_binding": binding, "fit_sha256": binding["fit_sha256"], "flat_slopes": meta["flat_slopes"], "initialization_named_parameters_sha256": meta["initialization_named_parameters_sha256"], "paired_reference": paired, "ema_by_epoch": scores, "checkpoint_sha256_by_epoch": {epoch: completed[epoch]["checkpoint_sha256"] for epoch in sorted(completed, key=int)}, "selection": {"epoch": best, "metric": "channel_variance_weighted_r2", "rule": "earliest maximum equal-session mean EMA on visible HO3 calibration"}, "legacy_selection": {"epoch": legacy_best, "metric": "legacy_flattened_r2", "rule": "earliest maximum equal-session mean EMA on visible HO3 calibration"}, "repeatability": repeat, "ho_contract": ho, "official_test_used": False}
    _atomic_json(dest / "score_receipt.json", result); _atomic_json(dest / "ho_calib_epoch_scan.json", result)
    _atomic_json(dest / "heartbeat.json", {"status": "COMPLETED", "event": "score_complete", "epoch": best, "completed_epochs": EPOCHS, "utc": datetime.now(timezone.utc).isoformat()})
    return {"status": "SCORE_COMPLETED", "best_epoch": best}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--arm", choices=ARMS, default="D_JOINT")
    parser.add_argument("--stage", choices=("train", "score"), default="train")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--carrier-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--paired-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=Path); parser.add_argument("--max-updates-smoke", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > EPOCHS:
        parser.error("--epochs must be 1..24")
    if args.max_updates_smoke is not None and (args.max_updates_smoke < 1 or args.stage != "train"):
        parser.error("smoke requires positive updates and --stage train")
    if args.max_updates_smoke is None and args.stage == "train" and args.epochs != EPOCHS:
        parser.error("formal flat M1 run requires exactly 24 epochs")
    if args.resume is not None and args.stage != "train":
        parser.error("--resume is valid only with --stage train")
    if args.dest is None:
        args.dest = OUTPUT_ROOT / (
            "root_smoke_m1_muscle_full_flat_s42" if args.max_updates_smoke is not None
            else "formal_m1_muscle_full_flat_s42"
        )
    print(json.dumps(run_train(args) if args.stage == "train" else run_score(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
