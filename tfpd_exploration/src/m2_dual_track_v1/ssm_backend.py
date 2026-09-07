"""Mamba2 import boundary for m2_dual_track_v1 (B-track owner).

Never substitutes DiagSSM, Mamba1, or a handwritten scan. If the official
Mamba2 module cannot be imported and exercised, this module reports
ENGINEERING_BLOCKED and leaves B-TRANSFORMER as the usable path.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import plan

Mamba2 = None
InferenceParams = None
MAMBA_AVAILABLE = False
BLOCK_REASON = "ENGINEERING_BLOCKED: mamba_ssm.Mamba2 has not been probed in this process"
_PROBE: dict[str, Any] = {}

_ISOLATED_ENV = plan.active_run_root() / "env"
_ISOLATED_PYTHON = _ISOLATED_ENV / "bin" / "python"
_STAGE0_DIR = plan.active_run_root() / "stage0"


def _reason(msg: str) -> str:
    if msg.startswith("ENGINEERING_BLOCKED"):
        return msg
    return f"ENGINEERING_BLOCKED: {msg}"


class _LocalInferenceParams:
    """Minimal official-compatible cache holder.

    Avoids importing ``mamba_ssm.utils.generation`` (that module pulls
    ``transformers``). Fields match what ``Mamba2.forward`` / ``_get_states_from_cache``
    actually read. This is not a substitute SSM.
    """

    def __init__(self, max_seqlen: int, max_batch_size: int, seqlen_offset: int = 0) -> None:
        self.max_seqlen = int(max_seqlen)
        self.max_batch_size = int(max_batch_size)
        self.seqlen_offset = int(seqlen_offset)
        self.batch_size_offset = 0
        self.key_value_memory_dict: dict = {}
        self.lengths_per_sample = None


def _try_import() -> tuple[Any, Any, str | None]:
    import sys

    try:
        # ``import mamba_ssm`` executes package __init__, which also loads
        # generation.py → transformers. The mixer submodule is still the
        # official Mamba2 class; we only need that.
        from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2
    except Exception as exc:  # noqa: BLE001 — import surface is the product
        cached = sys.modules.get("mamba_ssm.modules.mamba2")
        if cached is not None and hasattr(cached, "Mamba2"):
            _Mamba2 = cached.Mamba2
        else:
            return None, None, _reason(
                f"mamba_ssm.modules.mamba2 import failed: {type(exc).__name__}: {exc}"
            )
    infer: Any = _LocalInferenceParams
    try:
        from mamba_ssm.utils.generation import InferenceParams as _Infer
        infer = _Infer
    except Exception:
        infer = _LocalInferenceParams
    return _Mamba2, infer, None


def probe_mamba_kernel(*, device: str | None = None) -> dict[str, Any]:
    """Import is not enough: require a real forward / backward / step.

    Returns a receipt. Sets module-level MAMBA_AVAILABLE only when the
    official Mamba2 mixer runs a nonzero-branch step on the requested device.
    """
    global Mamba2, InferenceParams, MAMBA_AVAILABLE, BLOCK_REASON, _PROBE
    receipt: dict[str, Any] = {
        "import_ok": False,
        "forward_ok": False,
        "backward_ok": False,
        "step_ok": False,
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    }
    cls, infer, err = _try_import()
    if err is not None:
        Mamba2 = None
        InferenceParams = None
        MAMBA_AVAILABLE = False
        BLOCK_REASON = err
        receipt["block_reason"] = err
        _PROBE = receipt
        return receipt
    Mamba2 = cls
    InferenceParams = infer
    receipt["import_ok"] = True
    receipt["mamba2_module"] = getattr(cls, "__module__", None)
    receipt["has_inference_params"] = infer is not None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    receipt["device"] = device
    if device.startswith("cuda") and not torch.cuda.is_available():
        err = _reason("CUDA requested but torch.cuda.is_available() is False")
        MAMBA_AVAILABLE = False
        BLOCK_REASON = err
        receipt["block_reason"] = err
        _PROBE = receipt
        return receipt

    try:
        torch.manual_seed(0)
        mixer = cls(
            d_model=plan.B_TEMPORAL_WIDTH,
            d_state=plan.B_MAMBA_D_STATE,
            d_conv=plan.B_MAMBA_D_CONV,
            expand=plan.B_MAMBA_EXPAND,
            headdim=plan.B_MAMBA_HEADDIM,
            ngroups=plan.B_MAMBA_NGROUPS,
            chunk_size=plan.B_MAMBA_CHUNK,
            layer_idx=0,
        )
        mixer = mixer.to(device)
        x = torch.randn(2, 8, plan.B_TEMPORAL_WIDTH, device=device, dtype=torch.float32)
        # Nonzero residual / dynamics branches: official A_log, D, dt_bias stay as constructed.
        y = mixer(x)
        receipt["forward_ok"] = bool(torch.isfinite(y).all().item())
        receipt["forward_shape"] = list(y.shape)
        loss = (y * y).mean()
        loss.backward()
        grad_ok = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in mixer.parameters()
            if p.requires_grad
        )
        receipt["backward_ok"] = bool(grad_ok)
        conv_state, ssm_state = mixer.allocate_inference_cache(2, 8)
        mixer.eval()
        with torch.no_grad():
            out_t, conv_state, ssm_state = mixer.step(x[:, :1], conv_state, ssm_state)
        receipt["step_ok"] = bool(torch.isfinite(out_t).all().item())
        receipt["step_shape"] = list(out_t.shape)
        tags = {
            "A_log_no_wd": bool(getattr(mixer.A_log, "_no_weight_decay", False)),
            "D_no_wd": bool(getattr(mixer.D, "_no_weight_decay", False)),
            "dt_bias_no_wd": bool(getattr(mixer.dt_bias, "_no_weight_decay", False)),
        }
        receipt["no_weight_decay_tags"] = tags
        if not (receipt["forward_ok"] and receipt["backward_ok"] and receipt["step_ok"]):
            raise RuntimeError("Mamba2 forward/backward/step produced non-finite or missing grads")
        MAMBA_AVAILABLE = True
        BLOCK_REASON = ""
        receipt["block_reason"] = None
    except Exception as exc:  # noqa: BLE001
        err = _reason(f"Mamba2 kernel probe failed on {device}: {type(exc).__name__}: {exc}")
        MAMBA_AVAILABLE = False
        BLOCK_REASON = err
        receipt["block_reason"] = err
        receipt["exception"] = f"{type(exc).__name__}: {exc}"
    _PROBE = receipt
    return receipt


# Probe at import time on CPU-or-CUDA of *this* interpreter. Isolated-env
# 3090 smokes are invoked from stage0_b via the dedicated venv, not by
# mutating the shared spint site-packages.
_import_probe = probe_mamba_kernel(device="cuda" if torch.cuda.is_available() else "cpu")
if not MAMBA_AVAILABLE and _import_probe.get("import_ok") is False:
    # Keep the import-time reason; GPU smoke may still succeed in the isolated env.
    pass


def isolated_python() -> Path | None:
    if _ISOLATED_PYTHON.is_file():
        return _ISOLATED_PYTHON
    return None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu1_idle() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, f"nvidia-smi failed: {exc}"
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        index, uuid, mem, util = parts[0], parts[1], parts[2], parts[3]
        if uuid == plan.GPU1_UUID or index == "1":
            mem_i = float(mem)
            util_i = float(util)
            idle = mem_i < 512.0 and util_i < 5.0
            return idle, f"GPU1 uuid={uuid} mem={mem_i}MiB util={util_i}%"
    return False, "GPU1 not listed by nvidia-smi"


def _run_isolated_kernel_smoke() -> dict[str, Any]:
    py = isolated_python()
    if py is None:
        return {"ran": False, "reason": "isolated env python not present"}
    idle, note = _gpu1_idle()
    if not idle:
        return {"ran": False, "reason": f"GPU1 not idle ({note}); skipped isolated kernel smoke"}
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONPATH"] = str(plan.repo_root())
    code = r"""
import os, json
os.environ['PYTHONNOUSERSITE'] = '1'
import tfpd_exploration.src.m2_dual_track_v1.ssm_backend as sb
rec = sb.probe_mamba_kernel(device='cuda')
rec['mamba_available'] = bool(sb.MAMBA_AVAILABLE)
rec['block_reason'] = sb.BLOCK_REASON or None
print(json.dumps(rec))
"""
    try:
        proc = subprocess.run(
            [str(py), "-c", code],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return {"ran": True, "ok": False, "reason": "isolated kernel smoke timed out (180s)"}
    out = proc.stdout.strip().splitlines()
    payload: dict[str, Any] = {
        "ran": True,
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-2000:],
        "gpu_note": note,
    }
    if out:
        try:
            payload["probe"] = json.loads(out[-1])
            payload["ok"] = bool(payload["probe"].get("forward_ok") and payload["probe"].get("backward_ok") and payload["probe"].get("step_ok"))
        except json.JSONDecodeError:
            payload["ok"] = False
            payload["stdout_tail"] = proc.stdout[-2000:]
    else:
        payload["ok"] = False
        payload["stdout_tail"] = proc.stdout[-2000:]
    return payload


def _max_err(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-8)
    return {
        "max_abs": float(diff.max().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def _run_transformer_contracts(device: str) -> dict[str, Any]:
    from .contracts import make_stub_bank
    from .decoders import (
        BTransformerDecoder,
        count_trainable_parameters,
        whole_unit_dropout,
    )

    torch.manual_seed(0)
    model = BTransformerDecoder(seed=plan.SEED_PRIMARY).to(device).eval()
    bank = make_stub_bank(num_units=plan.CHANNELS, seed=3, device=torch.device(device))
    x = torch.randn(2, plan.WINDOW, plan.CHANNELS, device=device)
    y = model.forward_last(x, bank, bank.unit_mask)
    out: dict[str, Any] = {
        "output_shape": list(y.shape),
        "training_target_space": model.training_target_space,
        "name": model.name,
        "finite": bool(torch.isfinite(y).all().item()),
    }
    perm = torch.randperm(plan.CHANNELS, device=device)
    bank_p = make_stub_bank(num_units=plan.CHANNELS, seed=3, device=torch.device(device))
    # Rebuild a permuted bank with the same tensors.
    from .contracts import SessionBank

    bank_p = SessionBank(
        session_id=bank.session_id,
        support_trial_ids=bank.support_trial_ids,
        raw_trial_ids=bank.raw_trial_ids,
        X_store=bank.X_store,
        target_store=bank.target_store,
        eligible_starts=bank.eligible_starts,
        E0=bank.E0[perm],
        T=bank.T[perm],
        unit_mask=bank.unit_mask[perm],
        provenance=dict(bank.provenance),
        frozen_u=None if bank.frozen_u is None else bank.frozen_u[:, perm],
    )
    y_p = model.forward_last(x[:, :, perm], bank_p, bank_p.unit_mask)
    perm_err = _max_err(y, y_p)
    out["permutation"] = {
        **perm_err,
        "pass": bool(torch.allclose(y, y_p, atol=plan.PERM_ATOL, rtol=plan.PERM_RTOL)),
    }
    hidden = model.forward_hidden(x, bank, bank.unit_mask)
    x_future = x.clone()
    x_future[:, 25:] = x_future[:, 25:] + 3.5
    hidden_f = model.forward_hidden(x_future, bank, bank.unit_mask)
    causal_err = _max_err(hidden[:, :25], hidden_f[:, :25])
    out["causality"] = {
        **causal_err,
        "pass": bool(torch.allclose(hidden[:, :25], hidden_f[:, :25], atol=plan.PERM_ATOL, rtol=plan.PERM_RTOL)),
    }
    rec_rows = []
    rec_pass = True
    max_abs = 0.0
    for length in plan.B_RECURRENCE_LENGTHS:
        xr = torch.randn(2, int(length), plan.CHANNELS, device=device) * 2.5
        xr_hi = xr * 25.0
        for tag, seq in (("unit", xr), ("highmag", xr_hi)):
            y_fwd = model.forward_hidden(seq, bank, bank.unit_mask)
            y_step = model.step_hidden(seq, bank, bank.unit_mask)
            y_chunk = model.chunk_hidden(seq, bank, bank.unit_mask, chunk_size=plan.B_MAMBA_CHUNK)
            model.reset_temporal_state()
            y_reset = model.forward_hidden(seq, bank, bank.unit_mask)
            e1 = _max_err(y_fwd, y_step)
            e2 = _max_err(y_fwd, y_chunk)
            e3 = _max_err(y_fwd, y_reset)
            finite = bool(torch.isfinite(y_fwd).all() and torch.isfinite(y_step).all() and torch.isfinite(y_chunk).all())
            ok = (
                finite
                and torch.allclose(y_fwd, y_step, atol=plan.PERM_ATOL, rtol=plan.PERM_RTOL)
                and torch.allclose(y_fwd, y_chunk, atol=plan.PERM_ATOL, rtol=plan.PERM_RTOL)
                and torch.allclose(y_fwd, y_reset, atol=plan.PERM_ATOL, rtol=plan.PERM_RTOL)
            )
            rec_pass = rec_pass and ok
            max_abs = max(max_abs, e1["max_abs"], e2["max_abs"], e3["max_abs"])
            rec_rows.append({"length": int(length), "branch": tag, "forward_vs_step": e1, "forward_vs_chunk": e2, "forward_vs_reset": e3, "finite": finite, "pass": ok})
    out["recurrence"] = {"pass": rec_pass, "max_abs": max_abs, "rows": rec_rows}
    dropped = whole_unit_dropout(torch.ones(1, 8, dtype=torch.bool, device=device), p=1.0)
    out["unit_dropout_keep_one"] = bool(int(dropped.sum().item()) == 1)
    out["param_count"] = count_trainable_parameters(model)
    out["pass"] = bool(
        y.shape == (2, 2)
        and model.training_target_space == plan.TRAINING_TARGET_SPACE
        and out["finite"]
        and out["permutation"]["pass"]
        and out["causality"]["pass"]
        and out["recurrence"]["pass"]
        and out["unit_dropout_keep_one"]
    )
    return out


def _timed_optimizer_steps(device: str) -> dict[str, Any]:
    from .contracts import make_stub_bank
    from .decoders import BTransformerDecoder, adamw_param_groups

    model = BTransformerDecoder(seed=plan.SEED_PRIMARY).to(device).train()
    bank = make_stub_bank(num_units=plan.CHANNELS, seed=11, device=torch.device(device))
    opt = torch.optim.AdamW(
        adamw_param_groups(model),
        lr=plan.B_LR,
        betas=plan.ADAM_BETAS,
        eps=plan.ADAM_EPS,
    )
    batch = 8 if device == "cpu" else plan.EFFECTIVE_BATCH
    x = torch.randn(batch, plan.WINDOW, plan.CHANNELS, device=device)
    target = torch.randn(batch, plan.OUT_DIM, device=device)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        pred = model.forward_last(x, bank, bank.unit_mask)
        (pred - target).pow(2).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), plan.B_GRAD_CLIP)
        opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        pred = model.forward_last(x, bank, bank.unit_mask)
        (pred - target).pow(2).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), plan.B_GRAD_CLIP)
        opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = None
    if device.startswith("cuda"):
        peak = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    return {
        "model": "B-TRANSFORMER",
        "device": device,
        "warmup_steps": 20,
        "timed_steps": 100,
        "batch": batch,
        "elapsed_s": elapsed,
        "examples_per_s": (100 * batch) / elapsed if elapsed > 0 else None,
        "peak_memory_mib": peak,
        "note": (
            "PREFLIGHT ONLY on one model/optimizer instance. "
            "Formal epoch1 must rebuild model, optimizer, scheduler, "
            "sampler/dropout RNG, and all recurrent state."
        ),
    }


def stage0_b(device: str = "cpu") -> dict:
    """Coordinator hook. Transformer contracts always run; Mamba is optional."""
    _STAGE0_DIR.mkdir(parents=True, exist_ok=True)
    in_process = dict(_PROBE)
    isolated = _run_isolated_kernel_smoke()
    mamba_ok = bool(MAMBA_AVAILABLE) or bool(isolated.get("ok"))
    mamba_reason = None if mamba_ok else (BLOCK_REASON or isolated.get("reason") or isolated.get("probe", {}).get("block_reason"))
    trans = _run_transformer_contracts(device)
    timing_device = device
    idle, gpu_note = _gpu1_idle()
    if device == "cpu" and idle and torch.cuda.is_available():
        # Do not remap devices here; timing on CPU is valid. GPU1 smoke is isolated.
        pass
    timing = _timed_optimizer_steps(timing_device)
    report = {
        "schema": plan.SCHEMA,
        "owner": "B",
        "started": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "mamba_available": mamba_ok,
        "block_reason": mamba_reason,
        "in_process_probe": in_process,
        "isolated_kernel_smoke": isolated,
        "gpu1_note": gpu_note,
        "transformer": trans,
        "timing_100_step": timing,
        "formal_epoch1_rebuild_required": True,
        "pass": bool(trans.get("pass")),
        "status": "PASS" if trans.get("pass") and mamba_ok else ("PARTIAL" if trans.get("pass") else "FAIL"),
    }
    if trans.get("pass") and not mamba_ok:
        report["status"] = "ENGINEERING_BLOCKED"
        report["engineering_blocked_arm"] = "B-MAMBA"
        report["usable_arm"] = "B-TRANSFORMER"
    dest = _STAGE0_DIR / "stage0_b.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "Mamba2",
    "InferenceParams",
    "MAMBA_AVAILABLE",
    "BLOCK_REASON",
    "probe_mamba_kernel",
    "stage0_b",
    "isolated_python",
]
