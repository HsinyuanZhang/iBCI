"""CPU latency bench for sealed M-F250 lr1e4 (L=250, proj_add).

Official-shaped: single thread, batch 1, per-bin stream. Does not touch CUDA.
Compares (1) naive full-window forward, (2) SPD-A1 static fold,
(3) exact-E = frontend window cache + last-query on the last temporal block.

Identity: B-transformer unified series, NOT SPINT. Not an EvalAI submit.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
import sys

for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import h1_config, plan  # noqa: E402
from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity  # noqa: E402

DEFAULT_CKPT = (
    PACKAGE_ROOT
    / "results/h1_matrix/M_F250_lr1e4_20260906T094633Z/epoch_024.pt"
)
BIN_MS = 20.0
WINDOW = 250
UNITS = 176
KERNEL = 5


def _force_cpu_single_thread() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CUDA visible; refuse GPU for this official-shaped CPU bench")


def _load_model(ckpt_path: Path, *, ema: bool) -> BTransformerUnifiedDecoderIdentity:
    geo = h1_config.h1_matrix_geometry(WINDOW, "proj_add")
    model = BTransformerUnifiedDecoderIdentity(
        geo, seed=42, override_prefix=0, identity_mode="proj_add", override_window=WINDOW
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["shadow"] if ema else ckpt["raw_state_dict"]
    named = model.trainable_parameters()
    missing = set(named) - set(state)
    extra = set(state) - set(named)
    if missing or extra:
        raise RuntimeError(f"state mismatch missing={sorted(missing)[:6]} extra={sorted(extra)[:6]}")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(state[name].detach().to(dtype=param.dtype))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _synthetic_bank() -> TaskBank:
    rng = np.random.default_rng(0)
    e0 = rng.standard_normal((UNITS, 700), dtype=np.float32)
    carrier = rng.standard_normal((UNITS, 4), dtype=np.float32)
    mask = np.ones((UNITS,), dtype=np.bool_)
    x = rng.standard_normal((1, WINDOW, UNITS), dtype=np.float32)
    y = rng.standard_normal((1, 7), dtype=np.float32)
    return TaskBank(
        session_id="latency-synth",
        E0=e0,
        carrier=carrier,
        unit_mask=mask,
        X_store=x,
        target_store=y,
        window_ids=np.array([0], dtype=np.int64),
        calibration_meta={
            "shape": [UNITS, 700],
            "trial_count": 3,
            "estimator": "latency_synth",
            "array_sha256": array_sha256(e0),
            "budget": 3,
        },
    )


class FrontendWindowCache:
    def __init__(self, frontend, bank: TaskBank, keep: torch.Tensor) -> None:
        self.frontend = frontend
        self.bank = bank
        self.keep = keep
        self.e0 = torch.from_numpy(np.ascontiguousarray(bank.E0, dtype=np.float32))
        self.carrier = torch.from_numpy(np.ascontiguousarray(bank.carrier, dtype=np.float32))
        self.raw: torch.Tensor | None = None
        self.fused: torch.Tensor | None = None

    def _run(self, raw: torch.Tensor) -> torch.Tensor:
        return self.frontend(raw, self.e0, self.carrier, self.keep)

    def rebuild(self, raw_window: torch.Tensor) -> torch.Tensor:
        z = self._run(raw_window)
        self.raw = raw_window.detach().clone()
        self.fused = z.detach().clone()
        return z

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        assert self.raw is not None and self.fused is not None
        raw = torch.cat((self.raw[:, 1:], next_bin), dim=1)
        left = self._run(raw[:, : KERNEL - 1])
        right = self._run(raw[:, -KERNEL:])[:, -1:]
        z = torch.cat((left, self.fused[:, KERNEL:], right), dim=1)
        self.raw = raw
        self.fused = z
        return z


def temporal_last(model: BTransformerUnifiedDecoderIdentity, fused: torch.Tensor) -> torch.Tensor:
    hidden = fused + model.temporal.pe[: fused.size(1)].unsqueeze(0).to(dtype=fused.dtype)
    blocks = model.temporal.blocks
    for block in blocks[:-1]:
        hidden = block(hidden)
    block = blocks[-1]
    normed = block.norm1(hidden)
    attn = block.attn
    batch, width, dim = normed.shape
    qkv = attn.qkv(normed).view(batch, width, 3, attn.n_heads, attn.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    _q, key, value = qkv.unbind(dim=0)
    query = _q[:, :, -1:, :]
    attn_out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    attn_out = attn.proj(attn_out.transpose(1, 2).contiguous().view(batch, 1, dim))
    last = hidden[:, -1:] + attn_out
    last = last + block.ffn(block.norm2(last))
    return model.readout(model.final_norm(last))[:, 0, :]


def _stats(ms: list[float]) -> dict[str, float]:
    ordered = sorted(ms)
    n = len(ordered)
    return {
        "n": n,
        "mean_ms": float(statistics.fmean(ordered)),
        "median_ms": float(statistics.median(ordered)),
        "p95_ms": float(ordered[max(0, int(0.95 * (n - 1)))]),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "normalized_latency_median": float(statistics.median(ordered) / BIN_MS),
        "normalized_latency_p95": float(ordered[max(0, int(0.95 * (n - 1)))] / BIN_MS),
    }


def _time_calls(fn, n: int) -> list[float]:
    out: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--dest", type=Path, default=None)
    args = parser.parse_args()
    _force_cpu_single_thread()

    model = _load_model(args.ckpt, ema=True)
    bank = _synthetic_bank()
    keep = torch.from_numpy(np.ascontiguousarray(bank.unit_mask, dtype=np.bool_)).unsqueeze(0)
    rng = np.random.default_rng(1)
    stream = torch.from_numpy(rng.standard_normal((1, WINDOW + args.warmup + args.steps, UNITS), dtype=np.float32))
    window0 = stream[:, :WINDOW].contiguous()
    bins = [stream[:, WINDOW + i : WINDOW + i + 1].contiguous() for i in range(args.warmup + args.steps)]

    static = model.bank_static_term(bank)

    with torch.inference_mode():
        naive0 = model.forward(window0, bank)
        folded0 = model.forward_static_folded(window0, bank, static)
        cache = FrontendWindowCache(model.frontend, bank, keep)
        exact0 = temporal_last(model, cache.rebuild(window0))
        parity = {
            "naive_vs_folded_max_abs": float((naive0 - folded0).abs().max()),
            "naive_vs_exacte_max_abs": float((naive0 - exact0).abs().max()),
        }

        def naive_step(w: torch.Tensor) -> None:
            model.forward(w, bank)

        def folded_step(w: torch.Tensor) -> None:
            model.forward_static_folded(w, bank, static)

        def run_stream(step_fn, label: str) -> dict[str, object]:
            w = window0.clone()
            for nxt in bins[: args.warmup]:
                w = torch.cat((w[:, 1:], nxt), dim=1)
                step_fn(w)
            timed: list[float] = []
            for nxt in bins[args.warmup :]:
                w = torch.cat((w[:, 1:], nxt), dim=1)
                timed.extend(_time_calls(lambda ww=w: step_fn(ww), 1))
            return {"label": label, **_stats(timed)}

        naive = run_stream(naive_step, "naive_full_window")
        folded = run_stream(folded_step, "static_folded_A1")

        cache = FrontendWindowCache(model.frontend, bank, keep)
        cache.rebuild(window0)
        for nxt in bins[: args.warmup]:
            temporal_last(model, cache.advance(nxt))
        exact_ms: list[float] = []
        for nxt in bins[args.warmup :]:
            exact_ms.extend(_time_calls(lambda n=nxt: temporal_last(model, cache.advance(n)), 1))
        exact = {"label": "exactE_frontend_cache_plus_lastQ", **_stats(exact_ms)}

    report = {
        "schema": "btransform_unified_v1_h1_mf250_latency_cpu_v1",
        "ckpt": str(args.ckpt),
        "weights": "EMA shadow from epoch_024",
        "device": "cpu",
        "threads": 1,
        "batch": 1,
        "window": WINDOW,
        "units": UNITS,
        "official_bin_ms": BIN_MS,
        "official_latency_note": "normalized_latency ~= compute_ms / 20; 0.7 ~= 14 ms/bin on official CPU (often slower than this host)",
        "parity": parity,
        "paths": {"naive": naive, "static_folded": folded, "exact_e": exact},
        "submit_gate_local": {
            "want_median_ms_lt": 10.0,
            "want_p95_ms_lt": 14.0,
            "naive_clears": naive["median_ms"] < 10.0,
            "exact_e_clears": exact["median_ms"] < 10.0,
        },
    }
    dest = args.dest or args.ckpt.parent / "latency_cpu_v1"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "receipt.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
