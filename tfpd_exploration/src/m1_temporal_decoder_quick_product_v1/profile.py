"""W=100 / N=64 cost probe. Must finish before any train ETA."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import torch

from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import M1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import (
    M1Bank,
    M1TemporalFlatDecoder,
    M1TemporalRouteDecoder,
    adamw_param_groups,
    count_m1_decoder_parameters,
    prove_zero_gate_equals_flat,
)

from .config import EFFECTIVE_BATCH, GRAD_CLIP, LR, PROFILE_PATH, RESULT_ROOT, SEED, WEIGHT_DECAY


def _bank(device: torch.device) -> M1Bank:
    g = torch.Generator(device="cpu").manual_seed(0)
    return M1Bank(
        E0=torch.randn(M1_TEMPORAL.n_units, M1_TEMPORAL.e0_dim, generator=g).to(device),
        T=torch.randn(M1_TEMPORAL.n_units, M1_TEMPORAL.hc_dim, generator=g).to(device),
        unit_mask=torch.ones(M1_TEMPORAL.n_units, dtype=torch.bool, device=device),
    )


def _step_seconds(arm: str, device: torch.device) -> float:
    if arm == "flat":
        model = M1TemporalFlatDecoder(seed=SEED).to(device)
    else:
        template = M1TemporalFlatDecoder(seed=SEED)
        model = M1TemporalRouteDecoder(seed=SEED, flat_template=template).to(device)
    model.train()
    opt = torch.optim.AdamW(adamw_param_groups(model, WEIGHT_DECAY), lr=LR)
    bank = _bank(device)
    x = torch.randn(EFFECTIVE_BATCH, M1_TEMPORAL.window, M1_TEMPORAL.n_units, device=device)
    y = torch.randn(EFFECTIVE_BATCH, M1_TEMPORAL.out_dim, device=device)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        pred = model.forward_last(x, bank)
        pred.square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    pred = model.forward_last(x, bank)
    torch.nn.functional.mse_loss(pred, y).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flat = M1TemporalFlatDecoder(seed=SEED)
    route = M1TemporalRouteDecoder(seed=SEED, flat_template=flat)
    cpu_bank = _bank(torch.device("cpu"))
    x = torch.randn(2, M1_TEMPORAL.window, M1_TEMPORAL.n_units)
    zero = prove_zero_gate_equals_flat(flat, route, x, cpu_bank)
    peak = None
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.zeros(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        flat_s = _step_seconds("flat", device)
        route_s = _step_seconds("route", device)
        peak = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    else:
        flat_s = _step_seconds("flat", device)
        route_s = _step_seconds("route", device)
    report = {
        "status": "PROFILE_OK",
        "schema": "m1_temporal_decoder_quick_product_v1_profile",
        "device": str(device),
        "window": M1_TEMPORAL.window,
        "n_units": M1_TEMPORAL.n_units,
        "e0_dim": M1_TEMPORAL.e0_dim,
        "out_dim": M1_TEMPORAL.out_dim,
        "effective_batch": EFFECTIVE_BATCH,
        "step_s_flat": flat_s,
        "step_s_route": route_s,
        "peak_mib": peak,
        "params": {
            "flat": count_m1_decoder_parameters(flat),
            "route": count_m1_decoder_parameters(route),
        },
        "zero_gate": zero,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    PROFILE_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
