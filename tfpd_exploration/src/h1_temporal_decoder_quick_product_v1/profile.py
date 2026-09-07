"""Real W=700 / N=176 cost profile. Must finish before any train ETA is claimed."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_calibration import load_frozen_c2_materializer
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import C2_CKPT_SHA256, GPU0_UUID, H1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import (
    H1Bank,
    H1TemporalFlatDecoder,
    H1TemporalRouteDecoder,
    adamw_param_groups,
    count_h1_decoder_parameters,
    prove_zero_gate_equals_flat,
)

from .config import EFFECTIVE_BATCH, RESULT_ROOT, SEED
from .streaming import probe_cpu_stream

PROFILE_PATH = RESULT_ROOT / "h1_cost_profile.json"


def _gpu0_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    return out


def assert_gpu0_only() -> str:
    uuid = _gpu0_uuid()
    if uuid != GPU0_UUID:
        raise RuntimeError(f"refusing to run: GPU0 uuid {uuid} != {GPU0_UUID}")
    visible = torch.cuda.current_device()
    if visible != 0:
        raise RuntimeError("CUDA current device is not 0")
    return uuid


def _peak_vram_mib() -> float:
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def _make_bank(device: torch.device) -> H1Bank:
    g = torch.Generator(device="cpu").manual_seed(SEED)
    e0 = torch.randn(H1_TEMPORAL.n_units, H1_TEMPORAL.e0_dim, generator=g)
    hc = torch.randn(H1_TEMPORAL.n_units, H1_TEMPORAL.hc_dim, generator=g)
    mask = torch.ones(H1_TEMPORAL.n_units, dtype=torch.bool)
    return H1Bank(E0=e0.to(device), T=hc.to(device), unit_mask=mask.to(device))


def _make_batch(batch: int, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch, H1_TEMPORAL.window, H1_TEMPORAL.n_units, generator=g)
    y = torch.randn(batch, H1_TEMPORAL.out_dim, generator=g)
    return x.to(device), y.to(device)


def _tile_grad_parity(
    device: torch.device,
    tile_size: int,
    microbatch: int,
) -> dict[str, float]:
    flat_a = H1TemporalFlatDecoder(seed=SEED).to(device).train()
    flat_b = H1TemporalFlatDecoder(seed=SEED).to(device).train()
    flat_b.load_state_dict(flat_a.state_dict())
    bank = _make_bank(device)
    x, y = _make_batch(microbatch, device, seed=7)
    x_a = x.detach().clone().requires_grad_(True)
    x_b = x.detach().clone().requires_grad_(True)
    flat_a.set_memory_policy(None, False)
    pred_a = flat_a.forward_last(x_a, bank)
    loss_a = torch.nn.functional.mse_loss(pred_a, y)
    loss_a.backward()
    flat_b.set_memory_policy(tile_size, False)
    pred_b = flat_b.forward_last(x_b, bank)
    loss_b = torch.nn.functional.mse_loss(pred_b, y)
    loss_b.backward()
    pred_delta = float((pred_a.detach() - pred_b.detach()).abs().max().item())
    input_delta = float((x_a.grad - x_b.grad).abs().max().item())
    worst_param = 0.0
    grads_a = {n: p.grad for n, p in flat_a.named_parameters() if p.grad is not None}
    for name, param in flat_b.named_parameters():
        if param.grad is None:
            continue
        worst_param = max(worst_param, float((grads_a[name] - param.grad).abs().max().item()))
    ok = pred_delta < 1e-5 and input_delta < 2e-4 and worst_param < 2e-4
    if not ok:
        raise RuntimeError(
            f"tile parity failed pred={pred_delta} input={input_delta} param={worst_param}"
        )
    return {
        "tile_size": float(tile_size),
        "pred_max_abs": pred_delta,
        "input_grad_max_abs": input_delta,
        "param_grad_max_abs": worst_param,
        "passed": 1.0,
    }


def _one_effective_update(
    model: torch.nn.Module,
    device: torch.device,
    *,
    microbatch: int,
    tile_size: int | None,
    use_checkpoint: bool,
    n_warmup: int = 1,
    n_timed: int = 2,
) -> dict[str, Any]:
    require_div = EFFECTIVE_BATCH % microbatch == 0
    if not require_div:
        raise ValueError("microbatch must divide effective batch 32")
    accum = EFFECTIVE_BATCH // microbatch
    model.to(device)
    model.train()
    model.set_memory_policy(tile_size, use_checkpoint)
    opt = torch.optim.AdamW(adamw_param_groups(model, 1e-2), lr=1e-4)
    bank = _make_bank(device)
    times: list[float] = []
    last_loss = None
    for trial in range(n_warmup + n_timed):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        losses = []
        for piece in range(accum):
            x, y = _make_batch(microbatch, device, seed=100 + trial * 32 + piece)
            pred = model.forward_last(x, bank)
            # mse_loss averages the microbatch; divide by accum so the step matches a 32-mean.
            loss = torch.nn.functional.mse_loss(pred, y) / accum
            loss.backward()
            losses.append(float(loss.detach().item()) * accum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        last_loss = float(np.mean(losses))
        if trial >= n_warmup:
            times.append(elapsed)
        peak = _peak_vram_mib()
    return {
        "microbatch": microbatch,
        "accum": accum,
        "tile_size": tile_size,
        "activation_checkpoint": use_checkpoint,
        "step_s_mean": float(np.mean(times)),
        "step_s_min": float(np.min(times)),
        "peak_vram_mib": peak,
        "last_loss": last_loss,
        "oom": False,
    }


def _try_update(model_factory, device, **kwargs) -> dict[str, Any]:
    model = model_factory()
    try:
        result = _one_effective_update(model, device, **kwargs)
        result["ok"] = True
        return result
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {
            "ok": False,
            "oom": True,
            "error": str(exc).split("\n")[0][:200],
            **{k: kwargs.get(k) for k in ("microbatch", "tile_size", "use_checkpoint")},
        }
    finally:
        del model
        torch.cuda.empty_cache()


def run_cost_profile() -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    uuid = assert_gpu0_only()
    device = torch.device("cuda:0")
    materializer = load_frozen_c2_materializer()
    flat = H1TemporalFlatDecoder(seed=SEED)
    route = H1TemporalRouteDecoder(seed=SEED, flat_template=flat)
    bank_cpu = _make_bank(torch.device("cpu"))
    x_small = torch.randn(2, H1_TEMPORAL.window, H1_TEMPORAL.n_units)
    zero_gate = prove_zero_gate_equals_flat(flat, route, x_small, bank_cpu)
    # Routing grads after a real update must be nonzero.
    route.train()
    pred = route.forward_last(x_small, bank_cpu)
    torch.nn.functional.mse_loss(pred, torch.zeros_like(pred)).backward()
    route_grad = {
        name: float(param.grad.detach().abs().sum().item())
        for name, param in route.named_parameters()
        if param.grad is not None and ("route_proj" in name or name.endswith("q_cal") or name.endswith(".g"))
    }
    if not route_grad or max(route_grad.values()) <= 0.0:
        raise RuntimeError("routing grads are zero after a real update")
    param_flat = count_h1_decoder_parameters(flat)
    param_route = count_h1_decoder_parameters(route)

    candidates = [
        {"microbatch": 8, "tile_size": None, "use_checkpoint": False},
        {"microbatch": 4, "tile_size": None, "use_checkpoint": False},
        {"microbatch": 4, "tile_size": 100, "use_checkpoint": False},
        {"microbatch": 4, "tile_size": 100, "use_checkpoint": True},
        {"microbatch": 2, "tile_size": 50, "use_checkpoint": True},
        {"microbatch": 1, "tile_size": 50, "use_checkpoint": True},
    ]
    attempts: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    tile_parity: dict[str, float] | None = None
    for spec in candidates:
        if spec["tile_size"] is not None and tile_parity is None:
            try:
                tile_parity = _tile_grad_parity(device, int(spec["tile_size"]), microbatch=2)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                tile_parity = _tile_grad_parity(torch.device("cpu"), int(spec["tile_size"]), microbatch=1)
        result = _try_update(
            lambda: H1TemporalFlatDecoder(seed=SEED),
            device,
            microbatch=spec["microbatch"],
            tile_size=spec["tile_size"],
            use_checkpoint=spec["use_checkpoint"],
        )
        attempts.append({**spec, **result})
        if result.get("ok"):
            chosen = result
            break

    if chosen is None:
        report = {
            "schema": "h1_cost_profile_v1",
            "status": "RED_W700_DOES_NOT_FIT",
            "gpu_uuid": uuid,
            "window": 700,
            "n_units": 176,
            "effective_batch": EFFECTIVE_BATCH,
            "attempts": attempts,
            "tile_parity": tile_parity,
            "c2_sha256": C2_CKPT_SHA256,
            "e0_is_fused_identity": True,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        PROFILE_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    # ROUTE one timed step with the chosen memory policy.
    route_result = _try_update(
        lambda: H1TemporalRouteDecoder(seed=SEED, flat_template=H1TemporalFlatDecoder(seed=SEED)),
        device,
        microbatch=chosen["microbatch"],
        tile_size=chosen["tile_size"],
        use_checkpoint=chosen["activation_checkpoint"],
        n_warmup=0,
        n_timed=1,
    )

    cpu_stream = probe_cpu_stream(flat.cpu(), bank_cpu, n_windows=24)
    step_s = float(chosen["step_s_mean"])
    # Conservative held-in window count until the real manifest is sealed.
    # 13 sessions × ~20 post-M3 trials × ~(2500-700)/4 windows ≈ 11.7k; C2 ran ~4.1k updates/epoch.
    estimated_updates_per_epoch = 3600
    gpu_h_12 = step_s * estimated_updates_per_epoch * 12 / 3600.0
    gpu_h_pair_12 = gpu_h_12 * 2.0

    report = {
        "schema": "h1_cost_profile_v1",
        "status": "PROFILE_OK",
        "gpu_uuid": uuid,
        "cuda_visible_devices": "0",
        "window": 700,
        "n_units": 176,
        "out_dim": 7,
        "pe_max_len": H1_TEMPORAL.pe_max_len,
        "effective_batch": EFFECTIVE_BATCH,
        "e0_dim": H1_TEMPORAL.e0_dim,
        "e0_is_fused_identity": True,
        "hc_dim": H1_TEMPORAL.hc_dim,
        "token_in": H1_TEMPORAL.token_in,
        "c2_sha256": materializer.checkpoint_sha256,
        "c2_decoder_weights_used_for_init": False,
        "identity_branch_trained": False,
        "zero_gate_equals_flat": zero_gate,
        "routing_grad_abs_sum": route_grad,
        "routing_grads_nonzero": True,
        "params_flat": param_flat,
        "params_route": param_route,
        "tile_parity": tile_parity,
        "chosen": chosen,
        "route_probe": route_result,
        "attempts": attempts,
        "cpu_stream": cpu_stream,
        "estimated_updates_per_epoch": estimated_updates_per_epoch,
        "estimated_updates_note": "placeholder until held-in stride-4 manifest is sealed; not a 2h identity-only ETA",
        "projected_gpu_h_flat_12ep": gpu_h_12,
        "projected_gpu_h_pair_12ep": gpu_h_pair_12,
        "projected_gpu_h_pair_4ep": gpu_h_pair_12 * (4.0 / 12.0),
        "identity_only_2h_eta_revoked": True,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    PROFILE_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_cost_profile()
    print(json.dumps({k: report[k] for k in report if k not in {"attempts", "routing_grad_abs_sum"}}, indent=2))
    print("profile_path", str(PROFILE_PATH))
    print("status", report["status"])
    if report["status"] != "PROFILE_OK":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
