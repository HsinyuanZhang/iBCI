"""Small real-data overfit under the 20y train contract. New result root only."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import (
    _index_split,
    load_session_arrays,
    materialize_banks,
)
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import (
    H1TemporalFlatDecoder,
    adamw_param_groups,
)

from . import plan
from .metrics import report_pair


def _windows(neural: np.ndarray, last_bins: np.ndarray) -> np.ndarray:
    out = np.zeros((int(last_bins.size), plan.WINDOW, neural.shape[1]), dtype=np.float32)
    for i, last in enumerate(last_bins.tolist()):
        last = int(last)
        out[i] = neural[last + 1 - plan.WINDOW : last + 1]
    return out


def run_overfit(device: torch.device) -> dict:
    calib = _index_split("held-in-calib")
    name = plan.PROBE_SESSION
    arrays = load_session_arrays(calib[name], name, skip_first3=True)
    banks = materialize_banks({name: arrays})
    bank = banks[name]
    last = np.asarray(
        [start + plan.WINDOW - 1 for start in arrays.query_starts[: plan.PROBE_N]],
        dtype=np.int64,
    )
    if last.size != plan.PROBE_N:
        raise RuntimeError("not enough train probe windows")
    x = torch.as_tensor(_windows(arrays.neural, last), device=device)
    y = torch.as_tensor(arrays.velocity[last], device=device, dtype=torch.float32)
    target = y * plan.TRAIN_TARGET_SCALE
    gpu_bank = type(bank)(E0=bank.E0.to(device), T=bank.T.to(device), unit_mask=bank.unit_mask.to(device))
    model = H1TemporalFlatDecoder(seed=plan.SEED).to(device)
    model.train()
    opt = torch.optim.AdamW(adamw_param_groups(model, 0.0), lr=3.0e-3)
    history = []
    for step in range(1, plan.OVERFIT_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        pred = model.forward_last(x, gpu_bank)
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {1, 50, 100, 200, 400} or step == plan.OVERFIT_STEPS:
            history.append({"step": step, "mse_20y": float(loss.detach().item())})
    model.eval()
    with torch.inference_mode():
        pred = model.forward_last(x, gpu_bank)
        noisy = model.forward_last(x + 1.0, gpu_bank)
    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    report = {
        "schema": plan.SCHEMA + "_overfit",
        "revision": plan.REVISION,
        "session": name,
        "n_windows": plan.PROBE_N,
        "steps": plan.OVERFIT_STEPS,
        "train_target": "20y",
        "deploy_divisor": plan.DEPLOY_DIVISOR,
        "history": history,
        "train_units_pred_vs_20y": report_pair(pred_np, y_np * plan.TRAIN_TARGET_SCALE),
        "deploy_units_pred_over_20_vs_y": report_pair(pred_np / plan.DEPLOY_DIVISOR, y_np),
        "input_abs_delta_l2": float((noisy - pred).abs().square().sum().sqrt().item()),
        "updated": datetime.now(timezone.utc).isoformat(),
        "no_submit": True,
    }
    train = report["train_units_pred_vs_20y"]
    deploy = report["deploy_units_pred_over_20_vs_y"]
    # "Beats mean by 2e-5" is not a real fit. Require visible variance and R2.
    train_ok = (
        bool(train["beats_zero_r2"])
        and bool(train["beats_mean_r2"])
        and float(train["r2"]) >= 0.20
        and float(train["pred_std_over_target_std"] or 0.0) >= 0.20
        and float(report["input_abs_delta_l2"]) > 1.0e-3
    )
    deploy_ok = bool(deploy["beats_zero_r2"]) and bool(deploy["beats_mean_r2"]) and float(deploy["r2"]) >= 0.20
    report["overfit_beats_baselines_train_units"] = bool(train_ok)
    report["overfit_beats_baselines_deploy_units"] = bool(deploy_ok)
    report["status"] = "OVERFIT_OK" if train_ok and deploy_ok else "OVERFIT_WEAK"
    plan.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    dest = plan.RESULT_ROOT / "overfit_16win.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {"schema": plan.SCHEMA + "_overfit_ckpt", "raw_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
        plan.RESULT_ROOT / "overfit_16win.pt",
    )
    return report
