"""Fixed 2x2 diagnostic: FLAT/ROUTE x extra-training/prefix-augmentation.

Starts from each arm's fixed epoch24 EMA, not a newly selected checkpoint.
The original 24-epoch experiment and its selections remain unchanged. Both
treatments receive identical extra updates; only training history masking
differs. All1011 minival queries remain the primary descriptive surface.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from . import finalize_pair as finalizer
from .cold_history import apply_prefix_dropout
from .decoder import make_paired_decoders
from tfpd_exploration.src.m2_b_small_stability_v1 import training
from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
from tfpd_exploration.src.m2_dual_track_v1 import data, plan, sampler, training as source_training
from tfpd_exploration.src.m2_dual_track_v1.champion import tensor_state_sha256

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tfpd_exploration/results/m2/family_v1/cold_history_2x2_phase_v1"
PROSPECTIVE = ROOT / "tfpd_exploration/results/m2/family_v1/cold_history_2x2_prospective_v1"
MANIFEST = ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/sampler/shuffled_batch_manifest_24.json"
FINAL_SHA = "3be4a096f98bd9f77726094501271e2836e37c6f61c06ce136ce72505488cf47"
CELLS = ("FLAT_CONTROL", "FLAT_PREFIX", "ROUTE_CONTROL", "ROUTE_PREFIX")
PROTOCOL = {
    "schema": "m2_reset_prefix_2x2_phase_v1", "epochs": 2, "updates_per_epoch_per_cell": 3165,
    "cells": list(CELLS), "initialization": "each arm fixed epoch24 plain EMA; control/prefix copies byte-identical",
    "architecture_change": False, "seed": 42, "batch": 32, "window": 50,
    "optimizer": "fresh AdamW, original parameter groups, wd0.01, constant lr1e-5, clip1",
    "ema": "fresh decay0.9995; first successful update copies raw",
    "prefix": "p0 control; p0.5 independently per row, observed length uniform integers1..49 else50; zero only missing left history",
    "sampler": "original source-train shuffled manifest epochs1,2; same rows/order/masks for every cell",
    "whole_unit_dropout": "original Bx96 p0.1 hash counter, same in all cells",
    "target": "unchanged native-times5 final-bin target; raw MSE",
    "primary": "fixed phase epoch2 EMA equal-session R2 on ALL1011; no epoch/treatment/model selection",
    "reported": "all cells, both epochs, RAW and EMA, pooled secondary; compare prefix minus matched extra-training control",
    "motivation_disclosure": "designed after source-minival startup diagnostic; not preregistered before those observations",
    "forbidden": ["hidden", "official", "ext4", "changing original picks", "automatic promotion"],
    "hard_budget_seconds": 3600,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def authority():
    receipt_path = finalizer.OUT_ROOT / "receipt.json"
    if finalizer.sha(receipt_path) != FINAL_SHA:
        raise RuntimeError("fixed completed24 finalizer receipt drift")
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "SOURCE_MINIVAL_SELECTION_DIAGNOSTIC_NOT_UNTOUCHED_GENERALIZATION":
        raise RuntimeError("completed24 finalizer required")
    initial = {}
    for arm in ("FLAT", "ROUTE"):
        key = f"{arm}_endpoint24_epoch_024"
        record = receipt["exports"][key]
        path = finalizer.OUT_ROOT / f"{key}_ema_state.pt"
        if str(path) != record["export_path"] or finalizer.sha(path) != record["export_sha256"]:
            raise RuntimeError("fixed epoch24 EMA export drift")
        initial[arm] = {"path": str(path), "sha256": finalizer.sha(path),
                        "initial_all1011_equal_session_r2": receipt["scores"][key]["equal_session_r2"]}
    model_source = finalizer._code_and_source_authority()
    if model_source != receipt["authority_post"]:
        raise RuntimeError("original model/minival source authority drift")
    pretrain = json.loads((finalizer.PAIR_ROOT / "pretrain_manifest.json").read_text())
    training_source = finalizer._verify_pretrain_recipe(pretrain)
    extra = {Path(module.__file__).name: finalizer.sha(Path(module.__file__)) for module in (source_training, sampler)}
    for name in ("cold_history.py", "cold_compare.py"):
        extra[name] = finalizer.sha(Path(__file__).with_name(name))
    return {"protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL), "finalizer_receipt_sha256": FINAL_SHA,
            "initial": initial, "model_source": model_source, "training_source": training_source,
            "extra_code_sha256": extra, "manifest_sha256": finalizer.sha(MANIFEST)}


def build_cells(bound, device):
    bases = dict(zip(("FLAT", "ROUTE"), make_paired_decoders(42), strict=True))
    models, optimizers, emas, initial_sha = {}, {}, {}, {}
    for arm, model in bases.items():
        state = torch.load(bound["initial"][arm]["path"], map_location="cpu", weights_only=True)
        if not all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()):
            raise RuntimeError("initial EMA state finite/dtype drift")
        model.load_state_dict(state, strict=True)
        for treatment in ("CONTROL", "PREFIX"):
            key = f"{arm}_{treatment}"
            models[key] = copy.deepcopy(model).to(device)
            initial_sha[key] = tensor_state_sha256(models[key].state_dict())
            optimizers[key] = training.make_optimizer(models[key].trainable_parameters().items())
            for group in optimizers[key].param_groups:
                group["lr"] = 1e-5
            emas[key] = DecoderEMA(models[key], decay=.9995)
        if initial_sha[f"{arm}_CONTROL"] != initial_sha[f"{arm}_PREFIX"]:
            raise RuntimeError("within-arm treatment initialization not byte-identical")
    return models, optimizers, emas, initial_sha


def step_cells(batch, models, optimizers, emas, *, epoch, batch_id):
    base_mask = batch.unit_mask if batch.unit_mask is not None else batch.bank.unit_mask
    if base_mask.ndim == 1:
        base_mask = base_mask.unsqueeze(0).expand(batch.X.shape[0], -1)
    keep = training.unit_dropout_mask(base_mask, seed=42, epoch=epoch, batch_id=batch_id).to(batch.X.device)
    full, _ = apply_prefix_dropout(batch.X, seed=42, epoch=epoch, batch_id=batch_id, probability=0)
    shortened, lengths = apply_prefix_dropout(batch.X, seed=42, epoch=epoch, batch_id=batch_id, probability=.5)
    if not torch.equal(shortened[:, -1], batch.X[:, -1]):
        raise RuntimeError("augmentation changed the observed current bin")
    row = {"cold_rows": int((lengths < 50).sum()), "rows": batch.X.shape[0], "loss": {}}
    for key in CELLS:
        model, optimizer = models[key], optimizers[key]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        value = shortened if key.endswith("PREFIX") else full
        prediction = model.forward_last(value, batch.bank, dropout_keep=keep)
        loss = torch.nn.functional.mse_loss(prediction.float(), batch.last_target.float())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite extra-training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        emas[key].update_after_step(model)
        row["loss"][key] = float(loss.detach())
    return row


def _device():
    if os.environ.get("M2_COLD_COMPARE_GO") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise RuntimeError("review GO and planned physical GPU0 required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    return torch.device("cuda:0")


def _data(device):
    banks = {s: data.load_session_bank("source_train", s, device=device) for s in plan.HELDIN_SESSIONS}
    if source_training.count_updates(banks) != 3165:
        raise RuntimeError("source sampler update cardinality drift")
    return banks, sampler.load_manifest(MANIFEST)


def _checkpoint(key, model, optimizer, ema, epoch, bound):
    state = training.TrainState(cell=key, seed=42, epoch=epoch,
                                global_step=epoch * 3165, batch_id=3165,
                                manifest_digest=bound["manifest_sha256"])
    payload = training.save_checkpoint(model, optimizer, ema, state)
    payload.update(schema="m2_reset_prefix_2x2_phase_checkpoint_v1", authority=bound)
    if payload["ema"]["n_updates"] != epoch * 3165:
        raise RuntimeError("phase EMA update count drift")
    return payload


def validate_phase_checkpoint(payload, model, key, epoch, bound):
    if (payload.get("schema") != "m2_reset_prefix_2x2_phase_checkpoint_v1" or payload.get("cell") != key
            or payload.get("epoch") != epoch or payload.get("global_step") != epoch * 3165
            or payload.get("batch_id") != 3165 or payload.get("seed") != 42
            or payload.get("manifest_digest") != bound["manifest_sha256"] or payload.get("authority") != bound):
        raise RuntimeError("phase checkpoint identity/protocol drift")
    ema = payload.get("ema", {})
    params, raw = model.trainable_parameters(), payload.get("raw_state_dict", {})
    if (ema.get("n_updates") != epoch * 3165 or ema.get("decay") != .9995
            or set(ema.get("shadow", {})) != set(params) or set(raw) != set(model.state_dict())):
        raise RuntimeError("phase checkpoint EMA/state metadata drift")
    for name, value in raw.items():
        if value.shape != model.state_dict()[name].shape or not bool(torch.isfinite(value).all()):
            raise RuntimeError("phase checkpoint raw tensor drift")
    for name, value in ema["shadow"].items():
        if value.shape != params[name].shape or not bool(torch.isfinite(value).all()):
            raise RuntimeError("phase checkpoint EMA tensor drift")


def atomic_archive(path, arrays):
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def seal_fixed_endpoints(models, metrics, bound, device):
    exports = {}
    for key in CELLS:
        path = OUT / key / "epoch_002.pt"
        if finalizer.sha(path) != metrics["cells"][key]["checkpoint_sha256"]:
            raise RuntimeError("fixed phase checkpoint hash drift")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        arm = key.split("_")[0]
        model = make_paired_decoders(42)[0 if arm == "FLAT" else 1]
        validate_phase_checkpoint(payload, model, key, 2, bound)
        export = finalizer.export_ema_strict(payload, model, OUT / key / "endpoint2_ema_state.pt")
        # Export helper strictly reloads a fresh model; inference below reads
        # the actual exported plain state, not an in-memory EMA approximation.
        fresh = make_paired_decoders(42)[0 if arm == "FLAT" else 1]
        fresh.load_state_dict(torch.load(export["export_path"], map_location="cpu", weights_only=True), strict=True)
        score, arrays = finalizer.score_source_minival(fresh, device)
        recorded = metrics["cells"][key]["scores"]["EMA"]
        for name in ("equal_session_r2", "pooled_r2"):
            if not np.isfinite(score[name]) or abs(score[name] - recorded[name]) > 1e-5:
                raise RuntimeError("fixed phase exported EMA score reproduction drift")
        original = OUT / key / "epoch_002_ema_native.npz"
        if finalizer.sha(original) != recorded["archive_sha256"]:
            raise RuntimeError("phase native reference archive drift")
        with np.load(original, allow_pickle=False) as reference:
            for name in ("target", "start", "session"):
                if not np.array_equal(arrays[name], reference[name]):
                    raise RuntimeError("fixed phase exported query identity drift")
            np.testing.assert_allclose(arrays["prediction"], reference["prediction"], atol=1e-5, rtol=1e-5)
        archive = OUT / key / "endpoint2_ema_replay_native.npz"
        atomic_archive(archive, arrays)
        exports[key] = {**export, "reproduced_score": score, "replay_archive_sha256": finalizer.sha(archive)}
    return exports


def smoke():
    path = PROSPECTIVE / "resource_smoke.json"
    if path.exists():
        raise FileExistsError(path)
    device, bound = _device(), authority()
    models, optimizers, emas, shared = build_cells(bound, device)
    banks, manifest = _data(device)
    timed, rows = [], []
    torch.cuda.reset_peak_memory_stats()
    for batch_id, batch in enumerate(source_training.epoch_batches_shuffled(banks, manifest, 1, device=device)):
        if batch_id == 8:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        rows.append(step_cells(batch, models, optimizers, emas, epoch=1, batch_id=batch_id))
        torch.cuda.synchronize()
        if batch_id >= 2:
            timed.append(time.perf_counter() - start)
    if len(rows) != 8 or any(ema.n_updates != 8 for ema in emas.values()):
        raise RuntimeError("disposable smoke did not complete8 four-cell batches")
    if authority() != bound:
        raise RuntimeError("smoke authority drift")
    mean = float(np.mean(timed))
    result = {"status": "PASS_DISPOSABLE_SOURCE_TRAIN_ONLY", "authority": bound, "initial_cell_sha256": shared,
        "warmup_batches": 2, "timed_batches": 6, "four_cell_batch_seconds": timed,
        "mean_four_cell_batch_seconds": mean, "estimated_twoepoch_train_only_seconds": mean * 3165 * 2,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "first_last": [rows[0], rows[-1]], "weights_saved": False, "minival_scores_computed": False,
        "integrity_validation_may_deserialize_existing_minival_target_arrays": True}
    finalizer._atomic_json(path, result)
    print(json.dumps({k: result[k] for k in ("status", "mean_four_cell_batch_seconds", "estimated_twoepoch_train_only_seconds", "peak_allocated_bytes")}))
    return result


def train(authorization):
    device, bound = _device(), authority()
    if OUT.exists():
        raise FileExistsError(OUT)
    auth = json.loads(Path(authorization).read_text())
    if auth.get("status") != "ROOT_REVIEW_GO" or auth.get("authority") != bound:
        raise RuntimeError("exact prospective root authorization required")
    smoke_path = PROSPECTIVE / "resource_smoke.json"
    resource = json.loads(smoke_path.read_text())
    if (auth.get("smoke_sha256") != finalizer.sha(smoke_path) or resource["authority"] != bound
            or resource["estimated_twoepoch_train_only_seconds"] > 2700):
        raise RuntimeError("resource smoke not bound or exceeds admitted train-only budget")
    models, optimizers, emas, shared = build_cells(bound, device)
    banks, manifest = _data(device)
    started = time.monotonic()
    finalizer._atomic_json(OUT / "pretrain_authority.json", {"authority": bound, "authorization_sha256": finalizer.sha(Path(authorization)), "initial_cell_sha256": shared})
    history = {}
    for epoch in (1, 2):
        losses = {key: 0. for key in CELLS}
        count, cold_count, source_rows = 0, 0, 0
        for batch_id, batch in enumerate(source_training.epoch_batches_shuffled(banks, manifest, epoch, device=device)):
            if time.monotonic() - started > PROTOCOL["hard_budget_seconds"]:
                raise RuntimeError("fixed one-hour diagnostic deadline reached")
            row = step_cells(batch, models, optimizers, emas, epoch=epoch, batch_id=batch_id)
            count += 1
            cold_count += row["cold_rows"]
            source_rows += row["rows"]
            for key in CELLS:
                losses[key] += row["loss"][key] * row["rows"]
            if batch_id % 50 == 0:
                finalizer._atomic_json(OUT / "live.json", {"status": "TRAINING", "epoch": epoch, "batch_id": batch_id,
                    "updates_per_cell": (epoch - 1) * 3165 + batch_id + 1, "elapsed_seconds": time.monotonic() - started, **row})
        if count != 3165:
            raise RuntimeError("incomplete phase epoch")
        metrics = {"epoch": epoch, "source_rows": source_rows, "prefix_rows": cold_count, "cells": {}}
        for key in CELLS:
            checkpoint = OUT / key / f"epoch_{epoch:03d}.pt"
            finalizer._atomic_torch(checkpoint, _checkpoint(key, models[key], optimizers[key], emas[key], epoch, bound))
            reports = {}
            for kind in ("RAW", "EMA"):
                fn = lambda model: finalizer.score_source_minival(model, device)
                score, arrays = fn(models[key]) if kind == "RAW" else emas[key].score_with_ema(models[key], fn)
                if not np.isfinite(score["equal_session_r2"]):
                    raise RuntimeError("nonfinite fixed-surface phase score")
                archive = OUT / key / f"epoch_{epoch:03d}_{kind.lower()}_native.npz"
                # Exact output targets/start order were verified against the
                # completed24 source authority before this training phase.
                atomic_archive(archive, arrays)
                reports[kind] = {**score, "archive_sha256": finalizer.sha(archive)}
            metrics["cells"][key] = {"checkpoint_sha256": finalizer.sha(checkpoint),
                "mean_raw_mse": losses[key] / source_rows, "scores": reports}
        if authority() != bound:
            raise RuntimeError("phase model/source/initial artifact authority drift")
        finalizer._atomic_json(OUT / f"epoch_{epoch:03d}_metrics.json", metrics)
        history[epoch] = metrics
        print(json.dumps({"epoch": epoch, "ema_equal_session": {k: v["scores"]["EMA"]["equal_session_r2"] for k, v in metrics["cells"].items()}}), flush=True)
    endpoint = history[2]["cells"]
    delta = {arm: endpoint[f"{arm}_PREFIX"]["scores"]["EMA"]["equal_session_r2"] - endpoint[f"{arm}_CONTROL"]["scores"]["EMA"]["equal_session_r2"] for arm in ("FLAT", "ROUTE")}
    exports = seal_fixed_endpoints(models, history[2], bound, device)
    if authority() != bound:
        raise RuntimeError("phase final authority drift")
    result = {"status": "COMPLETE_FIXED_ENDPOINT2_DIAGNOSTIC", "authority": bound, "history": history,
        "primary_prefix_minus_control": delta, "strict_fixed_endpoint_exports": exports, "elapsed_seconds": time.monotonic() - started,
        "selected_or_promoted_model": None, "old_experiment_modified": False}
    finalizer._atomic_json(OUT / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.smoke == args.train or (args.train and args.authorization is None):
        parser.error("choose --smoke or --train --authorization PATH")
    smoke() if args.smoke else train(args.authorization)
