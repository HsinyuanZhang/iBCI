"""Public/source-development parity and R2 validation for M2 E-path execution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import contracts, data, plan
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime import constants as C
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.load_weights import load_pick

from .decoder import M2ExactOptimizedDecoder

OUT = C.REPO_ROOT / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/m2"
GATE_ATOL, GATE_RTOL = 1.0e-5, 1.0e-5


def _allowed(ref: np.ndarray) -> np.ndarray:
    return GATE_ATOL + GATE_RTOL * np.abs(ref)


def validate(kind: str, *, max_windows: int | None = None) -> dict[str, Any]:
    """Score all ext-4 source-development windows unless max_windows is explicit."""
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, meta = load_pick(kind, device="cpu")
    # Independent source-only constant comparator, fit solely on held-in
    # source-minival targets before the ext4 development sessions are opened.
    source_targets = [
        np.asarray(data.load_session_bank("source_minival", s, device="cpu").target_store, dtype=np.float32)
        for s in plan.HELDIN_SESSIONS
    ]
    source_mean = np.concatenate(source_targets, axis=0).mean(axis=0, keepdims=True)
    per_session: dict[str, Any] = {}
    all_ref: list[np.ndarray] = []
    all_opt: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    for session in plan.EXT4_SESSIONS:
        bank = data.load_session_bank("ext4", session, device="cpu")
        starts = np.asarray(bank.eligible_starts, dtype=np.int64)
        if max_windows is not None:
            starts = starts[:max_windows]
        # These starts are consecutive in the frozen ext4 stream.  Rebuild
        # once, then exercise the boundary-correct frontend cache per bin.
        raw = np.asarray(bank.X_store, dtype=np.float32)
        opt = M2ExactOptimizedDecoder(model, [bank])
        ref_rows: list[np.ndarray] = []
        opt_rows: list[np.ndarray] = []
        with torch.inference_mode():
            for ix, start in enumerate(starts):
                x = torch.from_numpy(np.array(raw[start : start + C.WINDOW], dtype=np.float32, copy=True)).unsqueeze(0)
                ref = model.forward_last(x, bank, bank.unit_mask).cpu().numpy()
                if ix == 0 or int(start) != int(starts[ix - 1]) + 1:
                    got = opt.rebuild(x)
                else:
                    got = opt.advance(x[:, -1:])
                ref_rows.append(ref)
                opt_rows.append(got.cpu().numpy())
        ref_np, opt_np = np.concatenate(ref_rows), np.concatenate(opt_rows)
        # target_store is already compacted to eligible_starts order.
        target = np.asarray(bank.target_store[: len(starts)], dtype=np.float32)
        # All models train in raw /5 target units; reproduce the official local
        # replay scale rather than silently comparing a differently scaled R2.
        ref_native, opt_native = ref_np / C.BEHAVIOR_SCALE, opt_np / C.BEHAVIOR_SCALE
        ref_r2 = float(contracts.variance_weighted_r2(target, ref_native))
        opt_r2 = float(contracts.variance_weighted_r2(target, opt_native))
        zero_r2 = float(contracts.variance_weighted_r2(target, np.zeros_like(target)))
        # A causal source-only constant baseline: mean from the session's
        # preceding eligible targets, never current/future target values.
        source_mean_pred = np.broadcast_to(source_mean, target.shape).copy()
        source_mean_r2 = float(contracts.variance_weighted_r2(target, source_mean_pred))
        delta = np.abs(opt_np - ref_np)
        if not bool(np.all(delta <= _allowed(ref_np))):
            raise RuntimeError(f"{kind}/{session}: element parity gate failed: max={delta.max()}")
        per_session[session] = {
            "windows": int(len(starts)), "max_abs_raw": float(delta.max()),
            "mean_abs_raw": float(delta.mean()), "reference_r2": ref_r2,
            "optimized_r2": opt_r2, "r2_abs_delta": abs(opt_r2 - ref_r2),
            "zero_baseline_r2": zero_r2, "first_target_constant_baseline_r2": source_mean_r2,
        }
        all_ref.append(ref_native); all_opt.append(opt_native); all_target.append(target)
    # Equal-session mean is the frozen visible-development selection metric.
    ref_mean = float(np.mean([v["reference_r2"] for v in per_session.values()]))
    opt_mean = float(np.mean([v["optimized_r2"] for v in per_session.values()]))
    if abs(opt_mean - ref_mean) > 1.0e-5:
        raise RuntimeError(f"{kind}: equal-session mean R2 delta {opt_mean-ref_mean} exceeds gate")
    result = {
        "schema": "m2_optimized_v2_validation_v1", "kind": kind,
        "surface": "ext4_source_development_only", "windows_limit": max_windows,
        "weight_sha256": meta["weight_sha256"], "view": meta["view"], "epoch": meta["epoch"],
        "per_session": per_session, "reference_equal_session_mean_r2": ref_mean,
        "optimized_equal_session_mean_r2": opt_mean, "mean_r2_abs_delta": abs(opt_mean-ref_mean),
        "element_gate": "abs(opt-ref) <= 1e-5 + 1e-5*abs(ref)",
        "r2_gate": "abs(mean_r2_delta) <= 1e-5", "passes": True,
        "ordinary_cross_window_temporal_kv_reuse": False,
        "frontend_cache": "recompute positions 0..3 and W-1 for k=5; retain old positions 5..W-1 as new 4..W-2",
        "last_query": "only final temporal layer query/FFN is sliced; first L-1 layers full",
        "independent_baselines": "zero and a constant mean fit only on source_minival held-in targets",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{kind}_source_dev_parity.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("small", "large"))
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(validate(args.kind, max_windows=args.max_windows), indent=2, sort_keys=True))
