"""Real-data learning check for M1-TEMPORAL-v2. Does not assume an H1 /20 bug."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.h1_temporal_unit_v2.metrics import report_pair
from tfpd_exploration.src.m1_temporal_decoder_quick_product_v1.config import N_UNITS, PREDICTION_DIVISOR, SEED
from tfpd_exploration.src.m1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.m1_temporal_v2 import plan
from tfpd_exploration.src.m1_temporal_v2.data import build_source_only_datamodule, materialize_source_banks
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import M1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import (
    M1TemporalFlatDecoder,
    adamw_param_groups,
)


def _session_name(session: Any) -> str:
    text = session.decode("ascii") if isinstance(session, bytes) else str(session)
    return text if text.startswith("ses-") else str(session)


def run_learn_check(device: torch.device, *, steps: int = 200, n_windows: int = 16) -> dict[str, Any]:
    if float(M1_TEMPORAL.prediction_divisor) != 1.0 or float(PREDICTION_DIVISOR) != 1.0:
        raise RuntimeError("M1 contract drifted; this check must not silently inherit H1 /20")
    data = build_source_only_datamodule()
    if getattr(data, "target_path", None) is not None or data.val_heldin_dataset is not None:
        raise RuntimeError("learn-check opened outer query")
    banks = materialize_source_banks()
    loader = data.train_dataloader()
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    names: list[str] = []
    for batch in loader:
        neural, target, _calib, session, _carrier = batch[:5]
        name = _session_name(session[0] if not isinstance(session, str) else session)
        neural = neural.to(dtype=torch.float32)
        if neural.dim() == 4:
            neural = neural.squeeze(-1) if neural.size(-1) == 1 else neural.mean(-1)
        if neural.size(-1) != N_UNITS and neural.size(1) == N_UNITS:
            neural = neural.transpose(1, 2)
        target = target.to(dtype=torch.float32)
        if target.dim() == 3:
            target = target[:, -1, :]
        take = min(int(neural.size(0)), n_windows - sum(int(x.size(0)) for x in xs))
        xs.append(neural[:take])
        ys.append(target[:take])
        names.extend([name] * take)
        if sum(int(x.size(0)) for x in xs) >= n_windows:
            break
    x = torch.cat(xs, dim=0)[:n_windows].to(device)
    y = torch.cat(ys, dim=0)[:n_windows].to(device)
    # Use the first window's session bank; probe is source-only and session-pure enough for a smoke set.
    bank = banks[names[0]]
    gpu_bank = type(bank)(E0=bank.E0.to(device), T=bank.T.to(device), unit_mask=bank.unit_mask.to(device))
    model = M1TemporalFlatDecoder(seed=SEED).to(device)
    model.train()
    opt = torch.optim.AdamW(adamw_param_groups(model, 1.0e-2), lr=1.0e-3)
    ema = DecoderEMA(model)
    history = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        pred = model.forward_last(x, gpu_bank)
        loss = F.mse_loss(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError("M1 overfit loss is not finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update_after_step(model)
        if step in {1, 20, 50, 100, 200} or step == steps:
            history.append({"step": step, "mse_raw_y": float(loss.detach().item())})
    model.eval()
    with torch.inference_mode():
        raw = model.forward_last(x, gpu_bank)
        noisy = model.forward_last(x + 1.0, gpu_bank)
    ema_pred = ema.score_with_ema(model, lambda module: module.forward_last(x, gpu_bank))
    pred_np = raw.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    report = {
        "schema": "m1_temporal_v2_learn_check",
        "revision": plan.REVISION,
        "prediction_divisor": PREDICTION_DIVISOR,
        "assumes_h1_div20": False,
        "train_target": "raw EMG / covariate from source-only loader, no extra scale",
        "n_windows": int(x.size(0)),
        "steps": steps,
        "sessions_in_probe": sorted(set(names[: int(x.size(0))])),
        "history": history,
        "raw_vs_y": report_pair(pred_np, y_np),
        "ema_vs_y": report_pair(ema_pred.detach().cpu().numpy(), y_np),
        "input_abs_delta_l2": float((noisy - raw).abs().square().sum().sqrt().item()),
        "target_path_resolved": False,
        "updated": datetime.now(timezone.utc).isoformat(),
        "no_submit": True,
    }
    raw = report["raw_vs_y"]
    ok = (
        bool(raw["beats_zero_r2"])
        and bool(raw["beats_mean_r2"])
        and float(raw["r2"]) >= 0.20
        and float(raw["pred_std_over_target_std"] or 0.0) >= 0.20
        and float(report["input_abs_delta_l2"]) > 1.0e-3
    )
    report["status"] = "LEARN_OK" if ok else "LEARN_FAIL"
    dest = plan.RESULT_ROOT / "learn_check.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
