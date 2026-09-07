"""H1 L=200 P16, C2-CAL-1 B2: scheduled M7 blocks + M4/M3 carrier switch.

Same decoder/recipe as h1_c2protocol_l200_p16 (32ep, 1e-4, EMA, HO-M3 pick).
Does not overwrite the half-cycle dest or historical stage-2 / e24 roots.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
SPINT_MAIN = WORKSPACE_ROOT / "SPINT-main"
SCRIPTS = Path(__file__).resolve().parent
for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT), str(SCRIPTS), str(SPINT_MAIN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import adapters, h1_config, plan, receipts  # noqa: E402
from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402
from btransform_unified_v1 import cal1_b2  # noqa: E402
from btransform_unified_v1.c2_protocol import (  # noqa: E402
    C2_CYCLE,
    C2_TIE_BREAK,
    HELDOUT_SESSION_TO_FALCON_KEY,
    HO_SELECTION_METRIC,
    grouped_session_metrics,
    pick_m7_start,
    prefix_schedule,
    select_epoch,
)
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity  # noqa: E402
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402

import h1_stage2_full13_proj_add as s2  # noqa: E402
from src.data.h1_m4_eb_pilot import (  # noqa: E402
    fit_deployment_carrier,
    fit_frozen_carrier,
    index_heldin_calib,
    interpolate_trial_identity,
    load_record,
)

TRAIN_ENV_FLAG = "BTRANSFORM_H1_C2_CAL1_B2_TRAIN"
WINDOW = 200
PROJ_DIM = 16
EPOCHS = 32
SEED = plan.SEED
SCALE = h1_config.TARGET_MULTIPLIER
DATA_ROOT = SPINT_MAIN / "data" / "000954"
HO_DIR = DATA_ROOT / "sub-HumanPitt-held-out-calib"
BUDGET_SECONDS = 5.0 * 3600.0


def _seal(path: Path, payload: Any) -> str:
    return receipts.seal_json(path, payload)


def _materialize_e0(activity: np.ndarray, carrier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    materializer = adapters._h1_materializer()
    act_t = torch.from_numpy(np.ascontiguousarray(activity)).unsqueeze(0)
    car_t = torch.from_numpy(np.ascontiguousarray(carrier)).unsqueeze(0)
    e0, hc = materializer.materialize_bank(act_t, car_t)
    return (
        np.ascontiguousarray(e0.detach().cpu().numpy(), dtype=np.float32),
        np.ascontiguousarray(hc.detach().cpu().numpy(), dtype=np.float32),
    )


def build_cal1_banks(train_banks: dict[str, TaskBank]) -> dict[str, Any]:
    inventory = cal1_b2.load_v1_inventory()
    paths = index_heldin_calib(DATA_ROOT)
    records = {session: load_record(paths[session]) for session in train_banks}
    eb_plan = cal1_b2.rebuild_plan(records)
    denom = float(inventory["denominator"])
    banks: dict[tuple[str, int, int], TaskBank] = {}
    n_trials = {session: len(records[session].trial_values) for session in train_banks}
    start_lists = {session: tuple(inventory["starts"][session]) for session in train_banks}
    for session, base in train_banks.items():
        record = records[session]
        values = [float(v) for v in record.trial_values]
        for start in start_lists[session]:
            block7 = values[start : start + 7]
            m4_vals = tuple(values[start : start + 4])
            if len(m4_vals) < 4:
                continue
            m4_raw = fit_frozen_carrier(record, eb_plan, m4_vals)["carrier"]
            m4_hc = cal1_b2.normalize_carrier(m4_raw, denom)
            m3_vals = tuple(values[start : start + 3])
            m3_raw = fit_deployment_carrier(record, eb_plan, m3_vals)["carrier"]
            m3_hc = cal1_b2.normalize_carrier(m3_raw, denom)
            for budget in C2_CYCLE:
                if start + int(budget) > n_trials[session]:
                    continue
                activity = np.stack(
                    [interpolate_trial_identity(record, float(v)) for v in block7[: int(budget)]],
                    axis=0,
                )
                carrier = m3_hc if int(budget) == 3 else m4_hc
                e0, hc = _materialize_e0(activity, carrier)
                meta = dict(base.calibration_meta)
                meta.update(
                    {
                        "budget": int(budget),
                        "m7_start": int(start),
                        "trial_count": int(budget),
                        "array_sha256": array_sha256(e0),
                        "carrier_sha256": array_sha256(hc),
                        "estimator": (
                            "C2-CAL-1 B2: scheduled block identity prefix; "
                            "M4 carrier if M in {7,5,4}; fit_deployment_carrier if M=3"
                        ),
                    }
                )
                banks[(session, int(start), int(budget))] = dataclasses.replace(
                    base, E0=e0, carrier=hc, calibration_meta=meta
                )
        print(f"[c2cal1b2] banks {session} n_starts={len(start_lists[session])}", flush=True)
    return {
        "banks": banks,
        "starts": start_lists,
        "n_trials": n_trials,
        "s_src": inventory["s_src"],
        "plan_q": int(eb_plan.q),
        "plan_lambda": float(eb_plan.ridge_lambda),
    }


def build_train_only(window: int) -> dict[str, Any]:
    s2.PROJ_DIM = PROJ_DIM
    s2.IDENTITY_MODE = "proj_add"
    s2.WINDOW = window
    sessions = list(h1_config.H1_ALL_SESSIONS)
    banks, xs, ys, ids = {}, {}, {}, {}
    for session in sessions:
        bank, _parity = s2._rebank("train", session, window)
        banks[session] = bank
        xs[session] = bank.X_store
        ys[session] = bank.target_store
        ids[session] = bank.window_ids
    n_updates = int(sum(math.ceil(len(xs[s]) / s2.EFFECTIVE_BATCH) for s in sessions))
    plan.require(n_updates == s2.FORMAL12_UPDATES_PER_EPOCH, f"caliber drift {n_updates}")
    return {
        "window": window,
        "sessions": sessions,
        "banks": banks,
        "X": xs,
        "y": ys,
        "ids": ids,
        "updates_per_epoch": n_updates,
    }


def build_ho_faces(window: int) -> dict[str, Any]:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    banks, xs, ys, keys = {}, {}, {}, []
    for session, falcon_key in HELDOUT_SESSION_TO_FALCON_KEY:
        path = HO_DIR / f"sub-HumanPitt-held-out-calib_{session}.nwb"
        plan.require(path.is_file(), f"missing HO-calib {path}")
        neural, velocity, _change, eval_mask = load_nwb(path, FalconTask.h1)
        neural = np.ascontiguousarray(neural, dtype=np.float32)
        velocity = np.ascontiguousarray(velocity, dtype=np.float32)
        ends = np.flatnonzero(np.asarray(eval_mask, dtype=np.bool_)).astype(np.int64)
        activity, carrier = adapters._h1_payload_arrays(session)
        e0, hc = _materialize_e0(activity, carrier)
        x = adapters._h1_windows(neural, ends, window)
        unit_mask = np.ones((e0.shape[0],), dtype=np.bool_)
        bank = TaskBank(
            session_id=session,
            E0=e0,
            carrier=hc,
            unit_mask=unit_mask,
            X_store=x,
            target_store=np.ascontiguousarray(velocity[ends], dtype=np.float32),
            window_ids=ends,
            calibration_meta={
                "shape": tuple(e0.shape),
                "trial_count": 3,
                "estimator": "C2 HO-M3: payload earliest-M3 + C2 materializer",
                "array_sha256": array_sha256(e0),
                "budget": 3,
            },
        )
        banks[falcon_key] = bank
        xs[falcon_key] = x
        ys[falcon_key] = bank.target_store
        keys.append(falcon_key)
        print(f"[c2protocol] HO-M3 face {falcon_key} n={len(ends)}", flush=True)
    return {"banks": banks, "X": xs, "y": ys, "keys": keys, "window": window}


def _predict_key(model, bank: TaskBank, x: np.ndarray, device: torch.device) -> np.ndarray:
    chunks = []
    for off in range(0, len(x), s2.EVAL_BATCH):
        xb = torch.from_numpy(x[off : off + s2.EVAL_BATCH]).to(device)
        with torch.inference_mode():
            raw = model(xb, bank)
        chunks.append(raw.detach().cpu().numpy() / SCALE)
    return np.concatenate(chunks, axis=0)


def score_ho_m3(model, ema: DecoderEMA, ho: dict[str, Any], device: torch.device) -> dict[str, Any]:
    named = model.trainable_parameters()
    backup = {name: param.detach().clone() for name, param in named.items()}
    was = model.training
    try:
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(ema.shadow[name].to(device=param.device, dtype=param.dtype))
        model.eval()
        preds, targets, masks = {}, {}, {}
        for key in ho["keys"]:
            pred = _predict_key(model, ho["banks"][key], ho["X"][key], device)
            tgt = ho["y"][key]
            preds[key] = pred
            targets[key] = tgt
            masks[key] = np.ones(len(tgt), dtype=bool)
        return grouped_session_metrics(preds, targets, masks, HELDOUT_SESSION_TO_FALCON_KEY)
    finally:
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(backup[name].to(device=param.device, dtype=param.dtype))
        model.train(was)


def run_train(dest: Path) -> dict[str, Any]:
    s2.PROJ_DIM = PROJ_DIM
    s2.IDENTITY_MODE = "proj_add"
    s2.WINDOW = WINDOW
    s2.CUDA_PIN = "1"
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    train = build_train_only(WINDOW)
    cal1 = build_cal1_banks(train["banks"])
    cal1_banks = cal1["banks"]
    ho = build_ho_faces(WINDOW)
    n_updates = int(train["updates_per_epoch"])
    total_updates = n_updates * EPOCHS
    warmup_updates = n_updates * plan.WARMUP_EPOCHS

    geometry = s2._make_geometry(WINDOW)
    model = BTransformerUnifiedDecoderIdentity(
        geometry, seed=SEED, override_prefix=0, override_window=WINDOW, identity_mode="proj_add"
    ).to(device)
    plan.require(int(model.init_meta["token_in"]) == 20, "P16 token_in drift")
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-4, weight_decay=plan.WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8
    )
    micro = 32
    accum = s2.EFFECTIVE_BATCH // micro
    rng = np.random.default_rng(SEED)
    sessions = train["sessions"]

    _seal(
        dest / "run_meta.json",
        {
            "schema": "btransform_unified_v1_h1_c2_cal1_b2_l200_p16_meta",
            "window": WINDOW,
            "proj_dim": PROJ_DIM,
            "epochs": EPOCHS,
            "train_sessions": sessions,
            "train_session_count": 13,
            "cal1": "B2 scheduled M7 starts + M4/deploy-M3 carrier",
            "s_src": cal1["s_src"],
            "plan_q": cal1["plan_q"],
            "plan_lambda": cal1["plan_lambda"],
            "v1_starts": {s: list(cal1["starts"][s]) for s in sessions},
            "n_trials": {s: int(cal1["n_trials"][s]) for s in sessions},
            "legal_starts_by_budget": {
                s: {str(b): list(cal1_b2.legal_starts(cal1["starts"][s], cal1["n_trials"][s], b)) for b in C2_CYCLE}
                for s in sessions
            },
            "npz_sha_gate": False,
            "plan_rebuild": "q=12 lambda=10 same config; arrays need not match sealed V1 bytes",
            "prefix_cycle": list(C2_CYCLE),
            "interleaved_validation": False,
            "early_stop": False,
            "checkpoint_selection_during_train": False,
            "selection": "C2 HO-M3 after all 32 epochs",
            "selection_metric": HO_SELECTION_METRIC,
            "tie_break": list(C2_TIE_BREAK),
            "updates_per_epoch": n_updates,
            "utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    (dest / "PROGRESS_10MIN.md").write_text(
        "# C2-CAL-1 B2 L200 P16 (32ep, no mid-val)\n\n"
        "| ep | train_mse | ruling |\n|----|-----------|--------|\n",
        encoding="utf-8",
    )

    metrics_path = dest / "metrics.jsonl"
    started = time.monotonic()
    deadline = started + BUDGET_SECONDS
    global_step = 0
    train_mse_series: dict[int, float] = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_t0 = time.monotonic()
        prefixes = prefix_schedule(epoch - 1, n_updates)
        order = list(sessions)
        rng.shuffle(order)
        losses: list[float] = []
        batch_id = -1
        step_in_epoch = 0
        for session in order:
            X, y = train["X"][session], train["y"][session]
            idx = rng.permutation(len(X))
            for offset in range(0, len(idx), s2.EFFECTIVE_BATCH):
                if time.monotonic() >= deadline:
                    raise RuntimeError("c2protocol 5h budget hit")
                take = idx[offset : offset + s2.EFFECTIVE_BATCH]
                budget = int(prefixes[step_in_epoch])
                start_roster = cal1_b2.legal_starts(
                    cal1["starts"][session], cal1["n_trials"][session], budget
                )
                start = pick_m7_start(
                    session, epoch0=epoch - 1, step=step_in_epoch, starts=start_roster
                )
                bank = cal1_banks[(session, int(start), int(budget))]
                micro_losses = []
                for m_off in range(0, len(take), micro):
                    m_take = take[m_off : m_off + micro]
                    batch_id += 1
                    global_step += 1
                    lr = warmup_cosine_lr(
                        global_step,
                        total_steps=total_updates,
                        warmup_steps=warmup_updates,
                        peak=1.0e-4,
                        min_factor=plan.LR_MIN_FACTOR,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
                    keep = whole_unit_dropout(
                        torch.from_numpy(bank.unit_mask.copy()), p=plan.UNIT_DROPOUT, generator=generator
                    )
                    xb = torch.from_numpy(X[m_take]).to(device)
                    yb = torch.from_numpy(y[m_take] * SCALE).to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = model(xb, bank, dropout_keep=keep)
                        loss = nn.functional.mse_loss(pred.float(), yb) / accum
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    micro_losses.append(float(loss.detach().cpu()) * accum)
                nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.GRAD_CLIP)
                optimizer.step()
                ema.update_after_step(model)
                losses.extend(micro_losses)
                step_in_epoch += 1
        plan.require(step_in_epoch == n_updates, f"epoch {epoch} steps {step_in_epoch} != {n_updates}")
        train_mse_series[epoch] = float(np.mean(losses)) if losses else float("nan")
        row = {
            "event": "epoch",
            "epoch": epoch,
            "epoch_zero_based": epoch - 1,
            "train_mse": train_mse_series[epoch],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.monotonic() - epoch_t0,
            "global_step": global_step,
            "ema_updates": ema.n_updates,
            "prefix_cycle": list(C2_CYCLE),
            "scored_during_train": False,
            "unix": time.time(),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        dest.joinpath("heartbeat.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        with (dest / "PROGRESS_10MIN.md").open("a", encoding="utf-8") as prog:
            prog.write(f"| {epoch} | {row['train_mse']:.6f} | TRAIN_ONLY |\n")
        print(
            f"[c2cal1b2 L200 P16] ep{epoch}/{EPOCHS} loss={row['train_mse']:.5f} ({row['seconds']:.0f}s)",
            flush=True,
        )
        torch.save(
            {
                "schema": "btransform_unified_v1_h1_c2_cal1_b2_ckpt",
                "window": WINDOW,
                "proj_dim": PROJ_DIM,
                "epoch": epoch,
                "global_step": global_step,
                "seed": SEED,
                "raw_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "ema": ema.state_dict(),
            },
            dest / f"epoch_{epoch:03d}.pt",
        )

    curve = []
    for epoch in range(1, EPOCHS + 1):
        ckpt = torch.load(dest / f"epoch_{epoch:03d}.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["raw_state_dict"])
        ema.load_state_dict(ckpt["ema"])
        ho_metrics = score_ho_m3(model, ema, ho, device)
        ho_row = {
            "epoch": epoch,
            "epoch_zero_based": epoch - 1,
            HO_SELECTION_METRIC: ho_metrics["r2_mean"],
            "worst_session_r2": ho_metrics["worst_session_r2"],
            "session_std_population": ho_metrics["r2_std_population"],
            "per_session_r2": ho_metrics["per_session_r2"],
            "per_recording_r2": ho_metrics["per_recording_r2"],
            "path": str(dest / f"epoch_{epoch:03d}.pt"),
        }
        curve.append(ho_row)
        print(
            f"[c2cal1b2 HO-M3] e{epoch} {HO_SELECTION_METRIC}={ho_metrics['r2_mean']:.4f} "
            f"worst={ho_metrics['worst_session_r2']:.4f}",
            flush=True,
        )
    picked = select_epoch(curve)
    selection = {
        "schema": "btransform_unified_v1_h1_c2_cal1_b2_ho_m3_selection",
        "status": "SEALED_HO_M3_SELECTION",
        "surface": "HO-M3 development/model-selection; not untouched held-out generalization",
        "selection_metric": HO_SELECTION_METRIC,
        "tie_break": list(C2_TIE_BREAK),
        "selected": picked,
        "curve": curve,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    _seal(dest / "ho_m3_selection.json", selection)
    summary = {
        "schema": "btransform_unified_v1_h1_c2_cal1_b2_train_receipt",
        "status": "COMPLETED",
        "epochs": EPOCHS,
        "train_mse": train_mse_series,
        "picked_epoch": int(picked["epoch"]),
        "picked_ho_m3": float(picked[HO_SELECTION_METRIC]),
        "elapsed_s": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    _seal(dest / "train_receipt.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--cuda-visible", default="1", choices=["1"])
    args = parser.parse_args()
    if os.environ.get(TRAIN_ENV_FLAG) != "1":
        print(f"REFUSED: {TRAIN_ENV_FLAG}=1 required", file=sys.stderr)
        return 2
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.cuda_visible):
        print("REFUSED: CUDA_VISIBLE_DEVICES must be 1", file=sys.stderr)
        return 2
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    s2.CUDA_PIN = "1"
    s2.PROJ_DIM = PROJ_DIM
    s2.IDENTITY_MODE = "proj_add"
    s2.WINDOW = WINDOW
    pre = s2.gpu_preflight(dest / "preflight_gpu.json")
    if not pre["ok"]:
        _seal(dest / "BLOCKED.json", {"reason": "GPU1 not clean", "preflight": pre})
        return 2
    summary = run_train(dest)
    print(json.dumps({"status": summary["status"], "picked_epoch": summary["picked_epoch"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
