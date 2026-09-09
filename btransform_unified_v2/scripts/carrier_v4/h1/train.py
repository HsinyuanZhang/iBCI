#!/usr/bin/env python3
"""Train H1 RIFT R300 recency on carrier-v4 C2-CAL-1 banks; select on HO-M3.

Recipe matches submission 582073: 13 source sessions, R300, 32 epochs, seed 42,
batch/microbatch 32, recency, dense attention.  Only the carrier estimator is
new.  Last-date 0.564 is not a selection or submission gate.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from common import OFFICIAL_HO_M3, atomic_json, setup_imports, sha_array, sha_file

setup_imports()

from btransform_unified_v1 import adapters, cal1_b2, h1_config  # noqa: E402
from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402
from btransform_unified_v1.c2_protocol import (  # noqa: E402
    C2_CYCLE,
    HELDOUT_SESSION_TO_FALCON_KEY,
    HO_SELECTION_METRIC,
    pick_m7_start,
    prefix_schedule,
    select_epoch,
)
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402
import h1_c2_cal1_b2_l200_p16 as b2  # noqa: E402
import h1_profiles  # noqa: E402
import common_estimator as core  # noqa: E402
import h1_train as ht  # noqa: E402
from build_banks import load_fit, materialize, ARMS  # noqa: E402

setup_imports()


def _register_rift_package() -> None:
    """Load RiftDecoder without executing v2 ``__init__`` (it imports M2)."""
    name = "btransform_unified_v2"
    src = Path(__file__).resolve().parents[3] / "src" / "btransform_unified_v2"
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "RiftDecoder", None) is not None:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(src)]
    pkg.__package__ = name
    pkg.__file__ = str(src / "__init__.py")
    sys.modules[name] = pkg


_register_rift_package()


CONTEXT = 300
EPOCHS = 32
SEED = 42
BATCH = 32
SCALE = h1_config.TARGET_MULTIPLIER
UPDATES_PER_EPOCH = 731


def _bank_implementation_sha256() -> dict[str, str]:
    """Hashes the exact builder stack that sealed a carrier bank receipt."""
    here = Path(__file__).resolve().parent
    paths = (here / "build_banks.py", here / "common.py", here.parent / "common_estimator.py")
    return {str(path.resolve()): sha_file(path) for path in paths}


def _model_code_sha256() -> dict[str, str]:
    """Bind checkpoints to the fusion adapter and the RIFT decoder implementation."""
    try:
        import fusion_model
    except ImportError as error:
        raise RuntimeError("carrier_v4/fusion_model.py is required for the selected fusion") from error
    # Include every decoder dependency already sealed by the RIFT helper, plus
    # this carrier trainer and its separate fusion implementation.
    code = ht.source_manifest()
    for path in (Path(__file__).resolve(), Path(fusion_model.__file__).resolve()):
        code[str(path)] = sha_file(path)
    return code


def _artifact_identity(banks_dir: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    receipt_path = banks_dir / "receipt.json"
    fit_json = Path(receipt["fit_json"]).resolve()
    if not fit_json.is_file() or sha_file(fit_json) != receipt.get("fit_json_sha256"):
        raise RuntimeError("carrier-v4 fit JSON hash drift")
    implementation = _bank_implementation_sha256()
    if receipt.get("implementation_sha256") != implementation:
        raise RuntimeError("carrier-v4 bank receipt implementation hash drift")
    return {
        "banks_receipt_sha256": sha_file(receipt_path),
        "fit_json_sha256": sha_file(fit_json),
        "bank_implementation_sha256": implementation,
        "model_code_sha256": _model_code_sha256(),
    }


def _require_identity(recorded: dict[str, Any], current: dict[str, Any], where: str) -> None:
    missing = [key for key in current if key not in recorded]
    if missing or any(recorded[key] != value for key, value in current.items()):
        detail = f"missing={','.join(missing)}" if missing else "hash mismatch"
        raise RuntimeError(f"{where} artifact identity mismatch ({detail})")


def _load_banks(banks_dir: Path, args: Any) -> tuple[Any, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    receipt_path = banks_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "h1_carrier_v4_latent_state3_ridge_intercept_27tag_v1" or receipt.get("status") != "BUILT":
        raise RuntimeError("carrier-v4 27-tag receipt is not sealed")
    if tuple(receipt.get("source_sessions", ())) != tuple(h1_config.H1_ALL_SESSIONS):
        raise RuntimeError("27-tag bank source roster is not official 13")
    if receipt.get("tag_count") != 27 or receipt.get("last_date_0564_used") is not False or receipt.get("information_arm") != args.information_arm:
        raise RuntimeError("27-tag bank contract drift")
    if sha_file(banks_dir / "banks_27.npz") != receipt["banks_27_sha256"]:
        raise RuntimeError("27-tag NPZ hash drift")
    identity = _artifact_identity(banks_dir, receipt)
    plan = load_fit(Path(receipt["fit_json"]))
    carriers: dict[str, np.ndarray] = {}
    with np.load(banks_dir / "banks_27.npz") as handle:
        for tag, row in receipt["tags"].items():
            t = np.ascontiguousarray(handle[f"T/{tag}"], np.float32)
            if sha_array(t) != row["T_sha256"]:
                raise RuntimeError(f"{tag}: T hash drift")
            carriers[tag] = t
            carriers[str(row["session"])] = t
            if row.get("falcon_key"):
                carriers[str(row["falcon_key"])] = t
    return plan, receipt, carriers, identity


def _deploy_prefix(record: Any, fit: Any, values: tuple[float, ...]) -> np.ndarray:
    """The common core, rather than a fixed M3 carrier bank, owns every prefix T."""
    rates, behavior, _trials = h1_profiles.support_blocks(record, values)
    carrier, _raw, _diag = core.deploy(fit, behavior, rates)
    return np.ascontiguousarray(carrier, np.float32)


def _build_model(args: Any, device: torch.device):
    """Defer the v4 fusion implementation to the separately owned module."""
    try:
        import fusion_model
    except ImportError as error:
        raise RuntimeError("carrier_v4/h1/fusion_model.py is required for the selected fusion") from error
    return fusion_model.build_h1(args.fusion, args.proj_dim, args.context_bins, "recency", SEED, args.attention_backend, device)


def build_cal1_signed(train_banks: dict[str, TaskBank], plan: Any, information_arm: str) -> dict[str, Any]:
    inventory = cal1_b2.load_v1_inventory()
    paths = h1_profiles._legacy().index_heldin_calib(b2.DATA_ROOT)
    records = {session: h1_profiles._legacy().load_record(paths[session]) for session in train_banks}
    banks: dict[tuple[str, int, int], TaskBank] = {}
    n_trials = {session: len(records[session].trial_values) for session in train_banks}
    start_lists = {session: tuple(inventory["starts"][session]) for session in train_banks}
    for session, base in train_banks.items():
        record = records[session]
        values = [float(v) for v in record.trial_values]
        for start in start_lists[session]:
            block7 = values[start : start + 7]
            m4_vals = tuple(values[start : start + 4])
            m3_vals = tuple(values[start : start + 3])
            if len(m4_vals) < 4:
                continue
            m4_hc = _deploy_prefix(record, plan, m4_vals)
            m3_hc = _deploy_prefix(record, plan, m3_vals)
            for budget in C2_CYCLE:
                if start + int(budget) > n_trials[session]:
                    continue
                activity = np.stack(
                    [h1_profiles._legacy().interpolate_trial_identity(record, float(v)) for v in block7[: int(budget)]],
                    axis=0,
                )
                carrier = m3_hc if int(budget) == 3 else m4_hc
                e0, hc = materialize(activity, carrier, information_arm)
                meta = dict(base.calibration_meta)
                meta.update(
                    {
                        "budget": int(budget),
                        "m7_start": int(start),
                        "trial_count": int(budget),
                        "array_sha256": array_sha256(e0),
                        "carrier_sha256": array_sha256(hc),
                        "estimator": "C2-CAL-1 B2 activity prefixes; latent_state3_ridge_intercept/per_column T (M3/M4 support)",
                    }
                )
                banks[(session, int(start), int(budget))] = dataclasses.replace(
                    base, E0=e0, carrier=hc, calibration_meta=meta
                )
        print(f"[carrier-v4 cal1] banks {session} n_starts={len(start_lists[session])}", flush=True)
    return {
        "banks": banks,
        "starts": start_lists,
        "n_trials": n_trials,
        "s_src": None,
        "plan_q": None,
        "plan_lambda": None,
        "carrier": "latent_state3_ridge_intercept/per_column",
    }


def build_ho_signed(context_bins: int, carriers: dict[str, np.ndarray], information_arm: str) -> dict[str, Any]:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    banks = {}
    xs = {}
    vm = {}
    ys = {}
    keys = []
    for session, key in HELDOUT_SESSION_TO_FALCON_KEY:
        path = b2.HO_DIR / f"sub-HumanPitt-held-out-calib_{session}.nwb"
        neural, velocity, _change, eval_mask = load_nwb(path, FalconTask.h1)
        ends = np.flatnonzero(np.asarray(eval_mask, bool)).astype(np.int64)
        x, m = ht.endpoint_context(np.asarray(neural, np.float32), ends, context_bins)
        activity, _old = adapters._h1_payload_arrays(session)
        carrier = np.ascontiguousarray(carriers[key], np.float32)
        e0, hc = materialize(activity, carrier, information_arm)
        banks[key] = TaskBank(
            session,
            e0,
            hc,
            np.ones(e0.shape[0], bool),
            x,
            np.asarray(velocity, np.float32)[ends],
            ends,
            {
                "shape": tuple(e0.shape),
                "trial_count": 3,
                "budget": 3,
                "estimator": "C2 HO-M3 activity + latent_state3_ridge_intercept/per_column T",
                "array_sha256": array_sha256(e0),
                "carrier_sha256": array_sha256(hc),
            },
        )
        xs[key], vm[key], ys[key] = x, m, banks[key].target_store
        keys.append(key)
    return {"context_bins": context_bins, "banks": banks, "X": xs, "valid": vm, "y": ys, "keys": keys}


def score_ho_m3(model, ema, ho, device) -> dict[str, Any]:
    return ht.score_ho_m3(model, ema, ho, device)


def compare_to_582073(selected: dict[str, Any]) -> dict[str, Any]:
    mean = float(selected[HO_SELECTION_METRIC])
    worst = float(selected["worst_session_r2"])
    base_mean = float(OFFICIAL_HO_M3[HO_SELECTION_METRIC])
    base_worst = float(OFFICIAL_HO_M3["worst_session_r2"])
    return {
        "baseline": OFFICIAL_HO_M3,
        "delta_mean": mean - base_mean,
        "delta_worst": worst - base_worst,
        "clearly_better": bool(mean >= base_mean + 0.01 and worst >= base_worst),
        "note": "pack only after a later explicit decision; last-date 0.564 is not a gate",
    }


def score_stage(args, carriers: dict[str, np.ndarray] | None = None, identity: dict[str, Any] | None = None) -> dict[str, Any]:
    dest = args.dest.resolve()
    meta = json.loads((dest / "run_meta.json").read_text())
    if (meta.get("schema") != "rift_h1_carrier_v4_r300_v1" or meta.get("status") != "FORMAL" or meta.get("variant") != args.variant
            or int(meta.get("context_bins", 0)) != args.context_bins
            or meta.get("fusion") != args.fusion or int(meta.get("proj_dim", 0)) != args.proj_dim or meta.get("information_arm") != args.information_arm):
        raise RuntimeError("dest is not a matching formal carrier-v4 RIFT run")
    if carriers is None or identity is None:
        _, _, carriers, identity = _load_banks(args.banks, args)
    _require_identity(meta, identity, "run metadata")
    backend = meta.get("attention_backend", "dense")
    device = torch.device(args.device)
    model = _build_model(args, device)
    ema = DecoderEMA(model, decay=0.9995)
    ho = build_ho_signed(args.context_bins, carriers, args.information_arm)
    curve = []
    for ep in range(1, int(meta["epochs"]) + 1):
        path = dest / f"epoch_{ep:03d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint for score stage: {path}")
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get("schema") != "rift_h1_carrier_v4_r300_v1" or state.get("smoke"):
            raise RuntimeError(f"checkpoint is not a formal carrier-v4 RIFT checkpoint: {path}")
        _require_identity(state, identity, f"checkpoint {path.name}")
        model.load_state_dict(state["raw_state_dict"])
        ema.load_state_dict(state["ema"])
        report = score_ho_m3(model, ema, ho, device)
        curve.append(
            {
                "epoch": ep,
                "epoch_zero_based": ep - 1,
                HO_SELECTION_METRIC: report["r2_mean"],
                "worst_session_r2": report["worst_session_r2"],
                "session_std_population": report["r2_std_population"],
                "per_session_r2": report["per_session_r2"],
            }
        )
        print(
            f"[carrier-v4 ho-m3] ep={ep}/{meta['epochs']} mean={report['r2_mean']:.6f} worst={report['worst_session_r2']:.6f}",
            flush=True,
        )
    selected = select_epoch(curve)
    comparison = compare_to_582073(selected)
    atomic_json(
        dest / "ho_m3_selection.json",
        {
            "status": "HO_M3_DEVELOPMENT_SELECTION",
            "selected": selected,
            "curve": curve,
            "versus_582073": comparison,
            "last_date_0564_used": False,
        },
    )
    atomic_json(
        dest / "train_receipt.json",
        {
            "status": "COMPLETED",
            "updates": int(meta["epochs"]) * UPDATES_PER_EPOCH,
            "epochs": int(meta["epochs"]),
            "selected_epoch": selected["epoch"],
            "official_test_used": False,
            "last_date_0564_used": False,
            "versus_582073": comparison,
        },
    )
    return {"status": "SCORE_COMPLETED", "selected_epoch": selected["epoch"], "versus_582073": comparison}


def run(args) -> dict[str, Any]:
    plan, banks_receipt, carriers, identity = _load_banks(args.banks, args)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    dest = args.dest.resolve()
    if args.resume:
        if not dest.is_dir() or not (dest / "run_meta.json").is_file():
            raise RuntimeError("--resume requires its existing formal run --dest")
        existing = json.loads((dest / "run_meta.json").read_text())
        if (
            existing.get("schema") != "rift_h1_carrier_v4_r300_v1"
            or existing.get("status") != "FORMAL"
            or existing.get("variant") != args.variant
            or int(existing.get("context_bins", 0)) != args.context_bins
            or existing.get("attention_backend", "dense") != args.attention_backend
            or int(existing.get("microbatch", 0)) != args.microbatch
            or existing.get("fusion") != args.fusion
            or int(existing.get("proj_dim", 0)) != args.proj_dim
            or existing.get("information_arm") != args.information_arm
        ):
            raise RuntimeError("--resume run metadata mismatch")
        _require_identity(existing, identity, "resume run metadata")
    elif dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"refusing nonempty non-resume result directory {dest}")
    train = ht.build_train(args.context_bins)
    cal1 = build_cal1_signed(train["banks"], plan, args.information_arm)
    contract_digests = ht._digest_train_contract(train, cal1)
    model = _build_model(args, device)
    init_sha = ht._sha_state(model)
    ema = DecoderEMA(model, decay=0.9995)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(SEED)
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "rift_h1_carrier_v4_r300_v1",
        "status": "SMOKE" if args.max_updates_smoke else "FORMAL",
        "variant": args.variant,
        "fusion": args.fusion,
        "proj_dim": args.proj_dim,
        "information_arm": args.information_arm,
        "context_bins": args.context_bins,
        "attention_backend": args.attention_backend,
        "layer_bins": list(ht.layer_bins(args.context_bins)),
        "raw_span": f"x[a-{args.context_bins - 1}:a] with input_valid_mask",
        "train_sessions": train["sessions"],
        "updates_per_epoch": 731,
        "epochs": args.epochs,
        "batch": 32,
        "microbatch": args.microbatch,
        "precision": "bf16 autocast on CUDA; fp32 CPU",
        "peak_lr": 1e-4,
        "floor_lr": 1e-5,
        "warmup_epochs": 1,
        "ema": 0.9995,
        "unit_dropout": 0.1,
        "seed": 42,
        "initialization_sha256": init_sha,
        "cal1": "C2-CAL-1 B2 activity prefixes + latent_state3_ridge_intercept/per_column T",
        **identity,
        "official_test_used": False,
        "last_date_0564_used": False,
        "selection_surface": "HO-M3 development only",
        "baseline_582073_ho_m3": OFFICIAL_HO_M3,
        "cold_start": "raw pre-session padding invalid via input_valid_mask",
        "pairing_digests": contract_digests,
        "source_manifest_sha256": ht.source_manifest(),
        "launch": {
            "argv": sys.argv,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        },
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if not args.resume:
        atomic_json(dest / "run_meta.json", meta)
    total = train["updates_per_epoch"] * args.epochs
    step = 0
    first_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("schema") != "rift_h1_carrier_v4_r300_v1" or state.get("smoke"):
            raise RuntimeError("resume requires a formal carrier-v4 H1-RIFT epoch checkpoint")
        _require_identity(state, identity, "resume checkpoint")
        if (
            state.get("variant") != args.variant
            or state.get("context_bins") != args.context_bins
            or state.get("attention_backend", "dense") != args.attention_backend
            or state.get("microbatch") != args.microbatch
            or state.get("fusion") != args.fusion
            or int(state.get("proj_dim", 0)) != args.proj_dim
            or state.get("information_arm") != args.information_arm
        ):
            raise RuntimeError("resume variant/context/backend/microbatch mismatch")
        model.load_state_dict(state["raw_state_dict"])
        opt.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng_cpu"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state.get("torch_rng_cuda") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_rng_cuda"]])
        step = int(state["global_step"])
        first_epoch = int(state["epoch"]) + 1
        if first_epoch > args.epochs:
            raise RuntimeError("resume checkpoint already reaches requested --epochs")
    started = time.monotonic()
    for ep in range(first_epoch, args.epochs + 1):
        model.train()
        losses = []
        schedule = prefix_schedule(ep - 1, train["updates_per_epoch"])
        order = list(train["sessions"])
        rng.shuffle(order)
        si = 0
        pairing = hashlib.sha256()
        for s in order:
            idx = rng.permutation(len(train["X"][s]))
            for off in range(0, len(idx), BATCH):
                take = idx[off : off + BATCH]
                budget = int(schedule[si])
                starts = cal1_b2.legal_starts(cal1["starts"][s], cal1["n_trials"][s], budget)
                bank = cal1["banks"][(s, pick_m7_start(s, epoch0=ep - 1, step=si, starts=starts), budget)]
                pairing.update(s.encode())
                pairing.update(np.asarray(train["ids"][s][take], np.int64).tobytes())
                pairing.update(str(bank.calibration_meta.get("array_sha256")).encode())
                step += 1
                lr = warmup_cosine_lr(step, total, train["updates_per_epoch"], peak=1e-4, min_factor=0.1)
                opt.param_groups[0]["lr"] = lr
                keep = whole_unit_dropout(
                    torch.from_numpy(bank.unit_mask.copy()),
                    p=0.1,
                    generator=torch.Generator().manual_seed(unit_dropout_seed(SEED, ep, si)),
                )
                pairing.update(keep.numpy().tobytes())
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for moff in range(0, len(take), args.microbatch):
                    mtake = take[moff : moff + args.microbatch]
                    xb = torch.from_numpy(train["X"][s][mtake]).to(device)
                    vm = torch.from_numpy(train["valid"][s][mtake]).to(device)
                    yb = torch.from_numpy(train["y"][s][mtake] * SCALE).to(device)
                    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
                    with amp:
                        pred = ht._forward(model, xb, bank, keep, vm)
                        raw_loss = nn.functional.mse_loss(pred.float(), yb)
                    if not bool(torch.isfinite(raw_loss)):
                        raise FloatingPointError(f"nonfinite loss at epoch={ep} step={step}")
                    (raw_loss * (len(mtake) / len(take))).backward()
                    batch_loss += float(raw_loss.detach()) * len(mtake) / len(take)
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
                opt.step()
                ema.update_after_step(model)
                losses.append(batch_loss)
                si += 1
                if step == 1 or step % 50 == 0:
                    progress = {
                        "event": "step",
                        "epoch": ep,
                        "step": step,
                        "loss": batch_loss,
                        "grad_norm": grad_norm,
                        "elapsed_seconds": time.monotonic() - started,
                        "cuda_peak_alloc_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
                        if device.type == "cuda"
                        else None,
                    }
                    atomic_json(dest / "heartbeat.json", progress)
                    print(
                        f"[carrier-v4 rift] {args.variant}] ep={ep}/{args.epochs} step={step}/{total} "
                        f"loss={batch_loss:.6f} grad={grad_norm:.3f} elapsed={progress['elapsed_seconds']:.0f}s",
                        flush=True,
                    )
                if args.max_updates_smoke and step >= args.max_updates_smoke:
                    break
            if args.max_updates_smoke and step >= args.max_updates_smoke:
                break
        ht._checkpoint(
            dest / f"epoch_{ep:03d}.pt",
            model,
            opt,
            ema,
            ep,
            step,
            rng,
            variant=args.variant,
            context_bins=args.context_bins,
            attention_backend=args.attention_backend,
            microbatch=args.microbatch,
            smoke=bool(args.max_updates_smoke),
        )
        # The historical checkpoint helper has no v4 fusion fields.  Bind them
        # immediately so a checkpoint cannot cross fusion/projection variants.
        checkpoint_path = dest / f"epoch_{ep:03d}.pt"
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_payload.update({
            "schema": "rift_h1_carrier_v4_r300_v1",
            "fusion": args.fusion,
            "proj_dim": args.proj_dim,
            "information_arm": args.information_arm,
            **identity,
        })
        torch.save(checkpoint_payload, checkpoint_path)
        row = {
            "event": "epoch",
            "epoch": ep,
            "global_step": step,
            "train_mse": float(np.mean(losses)),
            "lr": lr,
            "smoke": bool(args.max_updates_smoke),
            "endpoint_bank_mask_sequence_sha256": pairing.hexdigest(),
        }
        with (dest / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
        atomic_json(dest / "heartbeat.json", row)
        if args.max_updates_smoke:
            break
        if si != train["updates_per_epoch"]:
            raise RuntimeError(f"epoch {ep} updates {si} != {train['updates_per_epoch']}")
    if not args.max_updates_smoke:
        if step != args.epochs * UPDATES_PER_EPOCH:
            raise RuntimeError(f"formal update total {step} != {args.epochs * UPDATES_PER_EPOCH}")
        return score_stage(args, carriers, identity)
    return {
        "status": "SMOKE_COMPLETED",
        "steps": step,
        "initialization_sha256": init_sha,
        "banks_receipt_sha256": banks_receipt["banks_27_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", choices=("recency",), default="recency")
    parser.add_argument("--fusion", choices=("concat", "proj_add"), default="proj_add")
    parser.add_argument("--proj-dim", choices=(16, 32), type=int, default=16)
    parser.add_argument("--information-arm", choices=ARMS, default="full")
    parser.add_argument("--context-bins", type=int, choices=(300,), default=300)
    parser.add_argument("--attention-backend", choices=("dense",), default="dense")
    parser.add_argument("--stage", choices=("train", "score"), default="train")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--max-updates-smoke", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > EPOCHS or args.microbatch < 1 or args.microbatch > BATCH:
        parser.error("epochs must be 1..32 and microbatch must be 1..32")
    if not args.max_updates_smoke and args.epochs != EPOCHS:
        parser.error("formal carrier-v4 training requires exactly --epochs 32")
    if args.fusion == "concat" and args.proj_dim != 16:
        parser.error("concat fusion only supports --proj-dim 16")
    args.banks = args.banks.resolve()
    args.dest = args.dest.resolve()
    print(json.dumps(score_stage(args) if args.stage == "score" else run(args), indent=2))


if __name__ == "__main__":
    main()
