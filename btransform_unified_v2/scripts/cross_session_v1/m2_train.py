#!/usr/bin/env python3
"""Sealed M2 Z/B/D cross-session trainer and source-only epoch selector.

``train`` never opens EXT4.  It trains source-seven, evaluates every EMA on
source-minival only, and seals the earliest best source-minival epoch.  ``score``
then opens EXT4 once to evaluate that selected EMA and the predeclared e24
sensitivity endpoint.  No query target label can affect optimization or epoch
selection.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT.parent / "btransform_unified_v1" / "src", ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from btransform_unified_v1 import plan as v1_plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
from btransform_unified_v1.r2 import variance_weighted_r2
from scripts.rift_v1 import m2_concat_train as base
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan
from tfpd_exploration.src.m2_dual_track_v1 import sampler, training
from btransform_unified_v2.cross_session_m2_model import CrossSessionM2Decoder, ARMS, ARM_Z, ARM_B, ARM_D
from scripts.cross_session_v1 import m2_data

CELL = "M2-CROSS-SESSION-R50-D4-ZBD-M33-V1"
EPOCHS, BATCH = 24, 32
CACHE = ROOT.parent / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_hashes() -> dict[str, str]:
    paths = (Path(__file__), ROOT / "src/btransform_unified_v2/cross_session_m2_model.py",
             ROOT / "src/btransform_unified_v2/model.py", ROOT / "src/btransform_unified_v2/temporal.py",
             ROOT / "src/btransform_unified_v2/config.py", ROOT / "scripts/rift_v1/m2_concat_train.py",
             ROOT.parent / "tfpd_exploration/src/m2_hold_film_probe_v1/encoder.py",
             Path(old_plan.__file__), Path(sampler.__file__), Path(training.__file__), Path(m2_data.__file__),
             Path(v1_plan.__file__), Path(sys.modules[DecoderEMA.__module__].__file__))
    return {str(path): sha(path) for path in paths}


def cache_hashes(surface: str, banks: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Hash consumed query bytes; record calibration files by metadata only."""
    out: dict[str, Any] = {}
    for session in banks:
        root = CACHE / surface / session
        row = {name: sha(root / name) for name in ("X_store.npy", "target_store.npy", "eligible_starts.npy", "mapping.json")}
        row["calibration_artifact_metadata"] = {name: {"bytes": (root / name).stat().st_size}
                                                 for name in ("calib_activity.npy", "T.npy", "e0_u.pt")}
        row["calibration_bytes_read_by_model"] = arm != ARM_Z
        if arm in (ARM_B, ARM_D): row["calib_activity_sha256"] = sha(root / "calib_activity.npy")
        if arm == ARM_D: row["carrier_t4_sha256"] = sha(root / "T.npy")
        out[session] = row
    return out


def model_for(arm: str, seed: int, device: torch.device, banks: Mapping[str, Any], surface: str,
              encoder_init: str = "random") -> CrossSessionM2Decoder:
    model = CrossSessionM2Decoder(arm, seed=seed, encoder_init=encoder_init).to(device)
    model.install_session_memory(banks, CACHE / surface)
    return model


def valid(batch: Any, surface: str, pads: Mapping[str, int], device: torch.device) -> torch.Tensor:
    return base._batch_valid(batch, surface=surface, padding=pads, device=device)


def score_surface(model: nn.Module, dual: Mapping[str, Any], banks: Mapping[str, Any], pads: Mapping[str, int],
                  surface: str, device: torch.device) -> dict[str, Any]:
    model.eval(); rows: dict[str, Any] = {}; all_p: list[np.ndarray] = []; all_t: list[np.ndarray] = []
    for session, dual_bank in dual.items():
        preds: list[np.ndarray] = []; targets: list[np.ndarray] = []
        for offset in range(0, len(dual_bank.eligible_starts), BATCH):
            indices = list(range(offset, min(offset + BATCH, len(dual_bank.eligible_starts))))
            batch = sampler.batch_from_indices(dual_bank, indices, device=device, target_space=old_plan.SCORING_TARGET_SPACE)
            with torch.inference_mode():
                raw = model(batch.X, banks[session], input_valid_mask=valid(batch, surface, pads, device))
            preds.append(np.ascontiguousarray(raw.float().cpu().numpy() / old_plan.BEHAVIOR_SCALE, dtype=np.float32))
            targets.append(np.ascontiguousarray(batch.last_target.float().cpu().numpy(), dtype=np.float32))
        p, t = np.concatenate(preds), np.concatenate(targets)
        rows[session] = {"r2": float(variance_weighted_r2(t, p)), "window_count": int(len(t)),
                         "prediction_sha256": hashlib.sha256(p.tobytes()).hexdigest()}
        all_p.append(p); all_t.append(t)
    return {"per_session": rows, "equal_session_mean": float(np.mean([rows[s]["r2"] for s in sorted(rows)])),
            "pooled_r2": float(variance_weighted_r2(np.concatenate(all_t), np.concatenate(all_p))),
            "n_windows": int(sum(row["window_count"] for row in rows.values())), "partial": False}


def rng_state(device: torch.device) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def parameter_report(model: nn.Module) -> dict[str, int]:
    params = list(model.parameters())
    return {"total": int(sum(p.numel() for p in params)), "active": int(sum(p.numel() for p in params if p.requires_grad))}


def shared_init_report(seed: int, encoder_init: str) -> dict[str, Any]:
    models = {arm: CrossSessionM2Decoder(arm, seed=seed, encoder_init=encoder_init).cpu() for arm in ARMS}
    states = {arm: model.state_dict() for arm, model in models.items()}
    shared = sorted(set.intersection(*(set(x) for x in states.values())))
    unequal = [name for name in shared if not all(torch.equal(states[ARM_Z][name], states[arm][name]) for arm in ARMS[1:])]
    counts = {arm: parameter_report(model) for arm, model in models.items()}
    reference = max(v["active"] for v in counts.values())
    deltas = {arm: abs(v["active"] - reference) / reference for arm, v in counts.items()}
    if max(deltas.values()) > 0.05:
        raise RuntimeError(f"active parameter mismatch exceeds 5%: {counts}; baseline intentionally lacks encoder")
    if unequal:
        raise RuntimeError(f"shared initialization drift: {unequal[:5]}")
    encoder_equal = all(torch.equal(states[ARM_B][name], states[ARM_D][name])
                        for name in states[ARM_B] if name.startswith("encoder."))
    if not encoder_equal:
        raise RuntimeError("B/D random encoder initial states differ")
    return {"shared_state_keys": len(shared), "shared_state_equal": True, "parameter_counts": counts,
            "active_delta_from_max": deltas, "b_d_encoder_initial_state_equal": True}


def meta(args: argparse.Namespace, train_banks: Mapping[str, Any], split: Mapping[str, Any], updates: int) -> dict[str, Any]:
    return {"schema": "cross_session_m2_train_v1", "status": "FORMAL", "cell": CELL, "arm": args.arm,
            "seed": args.seed, "sampler_seed": int(split["manifest"]["sampler_seed"]), "epochs": EPOCHS, "updates_per_epoch": updates,
            "batch": BATCH, "context_bins": 50, "depth": 4, "attention_backend": "local",
            "identity_policy": {"Z_NONE": "no M33/E0/T4 read", "B_ACTIVITY_ONLY": "M33 identity; FiLM/direct carrier zero",
                                "D_JOINT": "M33 identity; T4 through FiLM and direct-token paths"}[args.arm],
            "encoder_init": args.encoder_init,
            "source_train_only_for_gradients": True, "epoch_selection": "earliest_best_source_train_last20pct_whole_trials_ema",
            "target_query_labels_used_for_selection": False, "target_query_labels_used_for_gradients": False,
            "ext4_use": "score stage only: selected source-trial-validation EMA and predeclared e24 sensitivity",
            "official_test_used": False, "evalai_opened": False, "split_contract": split,
            "source_hashes": source_hashes(), "cache_hashes": {"source_train": cache_hashes("source_train", train_banks, args.arm)}}


def assert_meta(run: Path, args: argparse.Namespace) -> dict[str, Any]:
    value = json.loads((run / "run_meta.json").read_text())
    if value.get("cell") != CELL or value.get("arm") != args.arm or int(value.get("seed", -1)) != args.seed:
        raise RuntimeError("run metadata arm/seed mismatch")
    if value.get("source_hashes") != source_hashes():
        raise RuntimeError("training source drift")
    return value


def save_resume(run: Path, epoch: int, step: int, model: nn.Module, opt: torch.optim.Optimizer, ema: DecoderEMA, device: torch.device) -> None:
    state = {"schema": "cross_session_m2_resume_v1", "cell": CELL, "epoch": epoch, "global_step": step,
             "model": model.state_dict(), "optimizer": opt.state_dict(), "ema": ema.state_dict(), "rng": rng_state(device)}
    temp = run / "resume_latest.pt.tmp"; torch.save(state, temp); temp.replace(run / "resume_latest.pt")


def tensor_dict_digest(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_ema_package(model: nn.Module, package_path: Path, device: torch.device) -> dict[str, Any]:
    package = torch.load(package_path, map_location="cpu", weights_only=True)
    named = dict(model.named_parameters())
    if set(package) != set(named):
        raise RuntimeError("EMA package keys do not exactly equal named parameters")
    if not all(torch.isfinite(value).all() for value in package.values()):
        raise RuntimeError("EMA package contains nonfinite parameter")
    raw_digest = tensor_dict_digest({name: value.detach() for name, value in named.items()})
    changed = 0
    with torch.no_grad():
        for name, value in named.items():
            incoming = package[name]
            if tuple(incoming.shape) != tuple(value.shape): raise RuntimeError(f"EMA shape drift: {name}")
            changed += int(not torch.equal(value.detach().cpu(), incoming))
            value.copy_(incoming.to(device=device, dtype=value.dtype))
    # State dict may contain aliases from the inherited frontend owner.  Accept
    # only aliases that share storage with one named parameter; any independent
    # persistent buffer would be an unsealed state path and is rejected.
    aliases: list[str] = []; fixed_buffers: dict[str, str] = {}
    for key, value in model.state_dict().items():
        if key in named: continue
        if any(value.data_ptr() == parameter.data_ptr() for parameter in named.values()): aliases.append(key); continue
        if key.endswith("temporal.recency_slopes") and value.ndim == 1 and torch.isfinite(value).all():
            fixed_buffers[key] = hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest(); continue
        raise RuntimeError(f"unexpected persistent non-parameter state: {key}")
    return {"path": str(package_path), "sha256": sha(package_path), "loaded_parameter_count": len(named),
            "ema_parameter_digest": tensor_dict_digest({name: value.detach() for name, value in named.items()}),
            "raw_parameter_digest_before_ema": raw_digest, "raw_vs_ema_changed_parameter_count": changed,
            "finite": True, "persistent_buffer_keys": fixed_buffers, "parameter_alias_state_keys": sorted(aliases)}


def load_resume(path: Path, model: nn.Module, opt: torch.optim.Optimizer, ema: DecoderEMA, device: torch.device) -> tuple[int, int]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("schema") != "cross_session_m2_resume_v1" or state.get("cell") != CELL:
        raise RuntimeError("resume contract mismatch")
    model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); ema.load_state_dict(state["ema"]); restore_rng(state["rng"], device)
    return int(state["epoch"]), int(state["global_step"])


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    if not args.resume:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    train_dual, val_dual, train_banks, split = m2_data.source_train_val_split(device)
    manifest = split["manifest"]; updates = training.count_updates(train_dual)
    if int(manifest.get("batch_size", -1)) != BATCH or updates < 1:
        raise RuntimeError("whole-trial source manifest/update contract drift")
    run = args.dest.resolve(); model = model_for(args.arm, args.seed, device, train_banks, "source_train", args.encoder_init)
    opt = training.build_optimizer(model.named_parameters(), lr=v1_plan.LR_PEAK, weight_decay=v1_plan.WEIGHT_DECAY)
    ema = DecoderEMA(model, decay=v1_plan.EMA_DECAY); run_meta = meta(args, train_banks, split, updates)
    if args.resume:
        if not (run / "run_meta.json").is_file(): raise RuntimeError("resume requires run_meta")
        if json.loads((run / "run_meta.json").read_text()) != run_meta: raise RuntimeError("resume metadata drift")
        last_epoch, step = load_resume(args.resume, model, opt, ema, device); start = last_epoch + 1
    else:
        if run.exists() and any(run.iterdir()): raise FileExistsError("new destination must be empty")
        atomic_json(run / "run_meta.json", run_meta); step = 0; start = 1
    # Keep the training RNG stream identical across Z/B/D after their unequal
    # constructors.  Encoder construction itself is inside fork_rng, but this
    # reset is an explicit receipt-worthy defense for dropout/sample parity.
    if not args.resume:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    pads = base._surface_padding("source_train", train_banks)
    progress_path = run / "source_trial_val_progress.json"
    curve: dict[str, Any] = (json.loads(progress_path.read_text())["completed"]
                             if args.resume and progress_path.is_file() else {})
    if args.resume and set(curve) != {str(epoch) for epoch in range(1, start)}:
        raise RuntimeError("resume source-minival curve is incomplete or mismatched")
    started = time.monotonic()
    for epoch in range(start, EPOCHS + 1):
        model.train(); losses: list[float] = []
        for batch_id, batch in enumerate(sampler.iter_manifest_batches(train_dual, manifest, epoch, device=device, target_space=old_plan.TRAINING_TARGET_SPACE)):
            step += 1; lr = warmup_cosine_lr(step, total_steps=EPOCHS * updates, warmup_steps=updates,
                                               peak=v1_plan.LR_PEAK, min_factor=v1_plan.LR_MIN_FACTOR)
            for group in opt.param_groups: group["lr"] = lr
            generator = torch.Generator(device="cpu"); generator.manual_seed(base.unit_dropout_seed(args.seed, epoch, batch_id))
            keep = whole_unit_dropout(batch.unit_mask, p=v1_plan.UNIT_DROPOUT, generator=generator)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = nn.functional.mse_loss(model(batch.X, train_banks[batch.session_id], dropout_keep=keep,
                                                   input_valid_mask=torch.ones((len(batch.window_ids), 50), dtype=torch.bool, device=device)).float(), batch.last_target)
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite source loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), v1_plan.GRAD_CLIP, error_if_nonfinite=True); opt.step(); ema.update_after_step(model); losses.append(float(loss))
            if step == 1 or step % 50 == 0:
                heartbeat = {"status": "TRAINING", "epoch": epoch, "global_step": step, "loss": losses[-1],
                             "elapsed_seconds": time.monotonic() - started}
                atomic_json(run / "heartbeat.json", heartbeat); print(json.dumps(heartbeat, sort_keys=True), flush=True)
        if step != epoch * updates: raise RuntimeError("epoch update-count drift")
        report = base._with_ema(model, ema, lambda: score_surface(model, val_dual, train_banks, pads, "source_train", device))
        curve[str(epoch)] = report
        package = {name: value.detach().cpu().clone() for name, value in ema.shadow.items()}
        torch.save(package, run / f"ema_epoch_{epoch:03d}.pt")
        save_resume(run, epoch, step, model, opt, ema, device)
        atomic_json(progress_path, {"schema": "cross_session_m2_source_trial_val_curve_v1", "completed": curve})
    chosen = min(((-float(row["equal_session_mean"]), int(epoch)) for epoch, row in curve.items()))[1]
    selected = torch.load(run / f"ema_epoch_{chosen:03d}.pt", weights_only=True)
    torch.save(selected, run / "selected_source_trial_val_ema.pt")
    receipt = {"schema": "cross_session_m2_train_receipt_v1", "status": "COMPLETED", "cell": CELL, "arm": args.arm,
               "seed": args.seed, "epochs": EPOCHS, "updates_per_epoch": updates, "global_step": step, "source_trial_val_ema_by_epoch": curve,
               "selection": {"rule": "earliest best source_train last20pct-whole-trial equal_session_mean", "epoch": chosen,
                             "equal_session_mean": curve[str(chosen)]["equal_session_mean"]},
               "target_query_labels_used_for_selection": False, "target_query_labels_used_for_gradients": False,
               "source_hashes": run_meta["source_hashes"], "cache_hashes": run_meta["cache_hashes"]}
    atomic_json(run / "train_receipt.json", receipt); return receipt


def score(args: argparse.Namespace) -> dict[str, Any]:
    run = args.dest.resolve(); run_meta = assert_meta(run, args)
    receipt = json.loads((run / "train_receipt.json").read_text())
    if receipt.get("status") != "COMPLETED" or receipt.get("target_query_labels_used_for_selection") is not False:
        raise RuntimeError("requires completed source-only selected run")
    device = torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    ext_dual, ext_banks = m2_data.ext4_query(device); pads = base._surface_padding("ext4", ext_banks)
    model = model_for(args.arm, args.seed, device, ext_banks, "ext4", str(run_meta["encoder_init"]))
    selected_load = load_ema_package(model, run / "selected_source_trial_val_ema.pt", device)
    selected_report = score_surface(model, ext_dual, ext_banks, pads, "ext4", device)
    # e24 sensitivity is predeclared and every epoch's EMA package is mandatory.
    e24_path = run / "ema_epoch_024.pt"
    if not e24_path.is_file(): raise RuntimeError("predeclared e24 EMA package is missing")
    e24_load = load_ema_package(model, e24_path, device); endpoint24 = score_surface(model, ext_dual, ext_banks, pads, "ext4", device)
    out = {"schema": "cross_session_m2_ext4_score_v1", "status": "COMPLETED", "cell": CELL, "arm": args.arm,
           "seed": args.seed, "source_trial_validation_selection": receipt["selection"], "selected_ema_load": selected_load, "selected_ema_ext4": selected_report,
           "e24_ema_load": e24_load,
           "predeclared_e24_ext4_sensitivity": endpoint24, "target_query_labels_used_for_selection": False,
           "target_query_labels_used_for_gradients": False, "official_test_used": False, "evalai_opened": False,
           "ext4_cache_hashes": cache_hashes("ext4", ext_banks, args.arm), "source_hashes": run_meta["source_hashes"]}
    atomic_json(run / "score_receipt.json", out); return out


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    train_dual, val_dual, train_banks, split = m2_data.source_train_val_split(device)
    if set(train_banks) & set(old_plan.EXT4_SESSIONS): raise RuntimeError("target session in source roster")
    evidence = {"source_roster": sorted(train_banks), "target_roster": sorted(old_plan.EXT4_SESSIONS),
                "rosters_disjoint": not bool(set(train_banks) & set(old_plan.EXT4_SESSIONS)),
                "source_trial_split_contract": split, "updates_per_epoch": training.count_updates(train_dual),
                "parameter_shared_init": shared_init_report(args.seed, args.encoder_init), "target_dataflow": "preflight/train do not load ext4; score-only opens ext4",
                "gradient_checks": {}}
    batch = next(sampler.iter_manifest_batches(train_dual, split["manifest"], 1, device=device, target_space=old_plan.TRAINING_TARGET_SPACE))
    for arm in ARMS:
        model = model_for(arm, args.seed, device, train_banks, "source_train", args.encoder_init).train()
        output = model(batch.X[:1], train_banks[batch.session_id]); output.sum().backward()
        encoder_grad = None if not hasattr(model, "encoder") else all(p.grad is not None for p in model.encoder.parameters())
        e0, carrier = model._identity([batch.session_id], device)
        if not bool(torch.isfinite(output).all()): raise RuntimeError(f"{arm}: nonfinite preflight output")
        evidence["gradient_checks"][arm] = {"finite_output": True, "encoder_grad": encoder_grad,
                                              "e0_nonzero": bool(torch.count_nonzero(e0)), "direct_carrier_nonzero": bool(torch.count_nonzero(carrier))}
    z, b, d = (evidence["gradient_checks"][arm] for arm in (ARM_Z, ARM_B, ARM_D))
    if z["e0_nonzero"] or z["direct_carrier_nonzero"] or not b["e0_nonzero"] or b["direct_carrier_nonzero"] or not d["e0_nonzero"] or not d["direct_carrier_nonzero"]:
        raise RuntimeError("arm information switch proof failed")
    if z["encoder_grad"] is not None or b["encoder_grad"] is not True or d["encoder_grad"] is not True:
        raise RuntimeError("arm gradient-path proof failed")
    # Actual output interventions, rather than only inspecting zero-valued
    # state: Z lacks a calibration buffer; B changes under calibration but has
    # no carrier buffer; D changes when its live carrier is perturbed.
    checks: dict[str, bool] = {}
    z_model = model_for(ARM_Z, args.seed, device, train_banks, "source_train", args.encoder_init).eval()
    with torch.inference_mode(): z0 = z_model(batch.X[:1], train_banks[batch.session_id])
    checks["z_has_no_calibration_buffer"] = not hasattr(z_model, f"calib_{batch.session_id.replace('-', '_')}")
    b_model = model_for(ARM_B, args.seed, device, train_banks, "source_train", args.encoder_init).eval(); key = batch.session_id.replace("-", "_")
    with torch.inference_mode(): b0 = b_model(batch.X[:1], train_banks[batch.session_id])
    with torch.no_grad(): getattr(b_model, f"calib_{key}").add_(1.0)
    with torch.inference_mode(): b1 = b_model(batch.X[:1], train_banks[batch.session_id])
    checks["b_calibration_perturbation_changes_output"] = not torch.equal(b0, b1)
    checks["b_has_no_carrier_buffer"] = not hasattr(b_model, f"carrier_{key}")
    d_model = model_for(ARM_D, args.seed, device, train_banks, "source_train", args.encoder_init).eval()
    with torch.inference_mode(): d0 = d_model(batch.X[:1], train_banks[batch.session_id])
    with torch.no_grad(): getattr(d_model, f"carrier_{key}").add_(1.0)
    with torch.inference_mode(): d1 = d_model(batch.X[:1], train_banks[batch.session_id])
    checks["d_carrier_perturbation_changes_output"] = not torch.equal(d0, d1)
    if not all(checks.values()): raise RuntimeError(f"preflight intervention proof failed: {checks}")
    evidence["actual_intervention_checks"] = checks
    return {"schema": "cross_session_m2_preflight_v1", "status": "COMPLETED", "cell": CELL, "arm": args.arm,
            "seed": args.seed, **evidence, "official_test_used": False, "evalai_opened": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="sealed cross-session M2 Z/B/D runner")
    parser.add_argument("--dest", type=Path, required=True); parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=(42, 43, 44), type=int, required=True); parser.add_argument("--stage", choices=("preflight", "train", "score"), required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--encoder-init", choices=("random",), default="random")
    args = parser.parse_args()
    if args.cpu_threads < 1: parser.error("cpu threads must be positive")
    if args.stage == "preflight": result = preflight(args)
    elif args.stage == "train": result = train(args)
    else: result = score(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
