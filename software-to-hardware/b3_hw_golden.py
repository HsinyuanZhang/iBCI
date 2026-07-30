#!/usr/bin/env python3
"""B3 EarlyPool IDEncoder — FP32 layered golden for HW/SW consistency checks.

Self-contained under software-to-hardware/. Pure-numpy forward mirrors
early_pool_encoder.EarlyPoolEncoder. Optional --torch-check uses that local module.

Examples (run inside this folder after copy):
  python b3_hw_golden.py --profile tiny --out runs/tiny_sw
  python b3_hw_golden.py --profile d64 --out runs/d64_sw --torch-check
  python b3_hw_golden.py --compare runs/tiny_sw runs/tiny_rtl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure local imports work no matter what cwd is used to launch the script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

PROFILES = {
    "tiny": dict(T=8, D=4, W=4, N=2, M=2, seed=0),
    "d64": dict(T=100, D=64, W=50, N=96, M=4, seed=0),
    "d128": dict(T=100, D=128, W=50, N=96, M=4, seed=0),
    "full_m33_d64": dict(T=100, D=64, W=50, N=96, M=33, seed=0),
}


@dataclass
class B3Shapes:
    T: int
    D: int
    W: int
    N: int
    M: int
    seed: int = 0


# ---------------------------------------------------------------------------
# Math (PyTorch Linear: y = x @ W.T + b, W shape [out, in])
# ---------------------------------------------------------------------------

def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x[..., in] -> [..., out]."""
    return x @ weight.T + bias


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class B3Weights:
    pre_w: np.ndarray  # [D, T]
    pre_b: np.ndarray  # [D]
    # B3: [D, D].  B3S/T4: [D, D + side_dim].
    post0_w: np.ndarray
    post0_b: np.ndarray  # [D]
    post1_w: np.ndarray  # [D, D]
    post1_b: np.ndarray  # [D]
    post2_w: np.ndarray  # [W, D]
    post2_b: np.ndarray  # [W]


def random_weights(shapes: B3Shapes, rng: np.random.Generator) -> B3Weights:
    """Xavier-like small random weights for reproducible HW bring-up."""
    def xavier(out_f: int, in_f: int) -> Tuple[np.ndarray, np.ndarray]:
        limit = np.sqrt(6.0 / (in_f + out_f))
        w = rng.uniform(-limit, limit, size=(out_f, in_f)).astype(np.float32)
        b = np.zeros((out_f,), dtype=np.float32)
        return w, b

    pre_w, pre_b = xavier(shapes.D, shapes.T)
    post0_w, post0_b = xavier(shapes.D, shapes.D)
    post1_w, post1_b = xavier(shapes.D, shapes.D)
    post2_w, post2_b = xavier(shapes.W, shapes.D)
    return B3Weights(pre_w, pre_b, post0_w, post0_b, post1_w, post1_b, post2_w, post2_b)


def random_calib(shapes: B3Shapes, rng: np.random.Generator) -> np.ndarray:
    """Calibration tensor [M, T, N], non-negative spike-like counts."""
    return rng.integers(0, 8, size=(shapes.M, shapes.T, shapes.N)).astype(np.float32)


# ---------------------------------------------------------------------------
# Layered forward — dumps every checkpoint
# ---------------------------------------------------------------------------

def forward_b3_layered(
    calib: np.ndarray,
    weights: B3Weights,
    side_features: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """calib: [M, T, N]. Returns named stages for HW compare.

    ``side_features`` extends the same golden to B3S/T4.  It is a per-unit
    matrix ``[N, side_dim]`` concatenated to the pooled activity immediately
    before ``post0``.  Plain B3 remains byte-for-byte compatible when it is
    omitted.
    """
    if calib.ndim != 3:
        raise ValueError(f"calib must be [M,T,N], got {calib.shape}")
    M, T, N = calib.shape
    D = int(weights.pre_w.shape[0])
    W = int(weights.post2_w.shape[0])
    if weights.pre_w.shape[1] != T:
        raise ValueError(f"pre_w in_features {weights.pre_w.shape[1]} != T={T}")

    # Per-trial features before pool: [M, N, D]
    feat_trials = np.zeros((M, N, D), dtype=np.float32)
    pre_linear_trials = np.zeros((M, N, D), dtype=np.float32)

    for m in range(M):
        # trial layout [T, N] -> per neuron time vector [T]
        for n in range(N):
            x = calib[m, :, n]  # [T]
            pre_lin = linear(x, weights.pre_w, weights.pre_b)
            pre_act = relu(pre_lin)
            pre_linear_trials[m, n] = pre_lin
            feat_trials[m, n] = pre_act

    sum_feat = feat_trials.sum(axis=0)  # [N, D]
    pooled_mean_feat = sum_feat / float(M)
    post0_in_features = int(weights.post0_w.shape[1])
    required_side_dim = post0_in_features - D
    if required_side_dim < 0:
        raise ValueError(
            f"post0_w in_features {post0_in_features} is smaller than pooled D={D}"
        )
    if required_side_dim == 0:
        if side_features is not None and side_features.shape[-1] != 0:
            raise ValueError("Plain B3 weights do not accept non-empty side_features")
        mean_feat = pooled_mean_feat
    else:
        if side_features is None:
            raise ValueError(
                f"post0_w requires side_dim={required_side_dim}, but side_features is missing"
            )
        side_features = np.asarray(side_features, dtype=np.float32)
        if side_features.shape != (N, required_side_dim):
            raise ValueError(
                f"side_features must be {(N, required_side_dim)}, got {side_features.shape}"
            )
        mean_feat = np.concatenate([pooled_mean_feat, side_features], axis=-1)

    post0_lin = linear(mean_feat, weights.post0_w, weights.post0_b)
    post0_act = relu(post0_lin)
    post1_lin = linear(post0_act, weights.post1_w, weights.post1_b)
    post1_act = relu(post1_lin)
    E = linear(post1_act, weights.post2_w, weights.post2_b)  # [N, W]

    # Streaming-equivalent SUM snapshots after each trial (for ctrl FSM debug)
    sum_after = np.zeros((M, N, D), dtype=np.float32)
    running = np.zeros((N, D), dtype=np.float32)
    for m in range(M):
        running = running + feat_trials[m]
        sum_after[m] = running

    return {
        "calib": calib.astype(np.float32),  # [M, T, N]
        "x_trial0_neuron0": calib[0, :, 0].astype(np.float32),
        "pre_linear": pre_linear_trials.astype(np.float32),  # [M, N, D]
        "feat": feat_trials.astype(np.float32),  # [M, N, D]  (== pre_relu)
        "sum_after_trial": sum_after.astype(np.float32),  # [M, N, D]
        "sum_feat": sum_feat.astype(np.float32),  # [N, D]
        "pooled_mean_feat": pooled_mean_feat.astype(np.float32),  # [N, D]
        "side_features": (
            np.empty((N, 0), dtype=np.float32)
            if side_features is None
            else side_features.astype(np.float32)
        ),
        "mean_feat": mean_feat.astype(np.float32),  # post0 input [N, D(+S)]
        "post0_linear": post0_lin.astype(np.float32),
        "post0_relu": post0_act.astype(np.float32),
        "post1_linear": post1_lin.astype(np.float32),
        "post1_relu": post1_act.astype(np.float32),
        "E": E.astype(np.float32),  # [N, W]
        # handy single-vector locals for first neuron / first trial
        "S0_x_m0_n0": calib[0, :, 0].astype(np.float32),
        "S1_pre_linear_m0_n0": pre_linear_trials[0, 0].astype(np.float32),
        "S2_feat_m0_n0": feat_trials[0, 0].astype(np.float32),
        "S4_mean_n0": mean_feat[0].astype(np.float32),
        "S5_post0_relu_n0": post0_act[0].astype(np.float32),
        "S6_post1_relu_n0": post1_act[0].astype(np.float32),
        "S7_E_n0": E[0].astype(np.float32),
    }


def streaming_forward_matches_batch(
    calib: np.ndarray,
    weights: B3Weights,
    side_features: np.ndarray | None = None,
) -> np.ndarray:
    """Explicit push_trial / finalize path; returns E[N,W]."""
    M, T, N = calib.shape
    D = weights.pre_w.shape[0]
    sum_feat = np.zeros((N, D), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            x = calib[m, :, n]
            sum_feat[n] += relu(linear(x, weights.pre_w, weights.pre_b))
    mean_feat = sum_feat / float(M)
    required_side_dim = int(weights.post0_w.shape[1]) - D
    if required_side_dim:
        if side_features is None or side_features.shape != (N, required_side_dim):
            raise ValueError(
                f"side_features must be {(N, required_side_dim)} for streaming B3S"
            )
        mean_feat = np.concatenate(
            [mean_feat, np.asarray(side_features, dtype=np.float32)], axis=-1
        )
    h = relu(linear(mean_feat, weights.post0_w, weights.post0_b))
    h = relu(linear(h, weights.post1_w, weights.post1_b))
    return linear(h, weights.post2_w, weights.post2_b)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

STAGE_KEYS = [
    "calib",
    "pre_linear",
    "feat",
    "sum_after_trial",
    "sum_feat",
    "mean_feat",
    "post0_linear",
    "post0_relu",
    "post1_linear",
    "post1_relu",
    "E",
    "S0_x_m0_n0",
    "S1_pre_linear_m0_n0",
    "S2_feat_m0_n0",
    "S4_mean_n0",
    "S5_post0_relu_n0",
    "S6_post1_relu_n0",
    "S7_E_n0",
]


def save_weights(out: Path, weights: B3Weights) -> None:
    wdir = out / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    for name, arr in asdict(weights).items():
        np.save(wdir / f"{name}.npy", arr)


def load_weights(wdir: Path) -> B3Weights:
    def ld(name: str) -> np.ndarray:
        return np.load(wdir / f"{name}.npy").astype(np.float32)

    return B3Weights(
        pre_w=ld("pre_w"),
        pre_b=ld("pre_b"),
        post0_w=ld("post0_w"),
        post0_b=ld("post0_b"),
        post1_w=ld("post1_w"),
        post1_b=ld("post1_b"),
        post2_w=ld("post2_w"),
        post2_b=ld("post2_b"),
    )


def save_run(out: Path, shapes: B3Shapes, weights: B3Weights, stages: Dict[str, np.ndarray]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_weights(out, weights)
    sdir = out / "stages"
    sdir.mkdir(exist_ok=True)
    for key in STAGE_KEYS:
        np.save(sdir / f"{key}.npy", stages[key])
    np.save(out / "E.npy", stages["E"])

    meta = {
        "variant": "B3_EarlyPool",
        "shapes": asdict(shapes),
        "stage_shapes": {k: list(stages[k].shape) for k in STAGE_KEYS},
        "notes": [
            "Linear convention: y = x @ W.T + b, W shape [out, in] (PyTorch nn.Linear).",
            "feat == ReLU(pre_linear); accumulate feat over M then mean; then post MLP.",
            "Local vectors S0..S7 are neuron0 (and trial0 where applicable) for quick RTL probes.",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Human-readable tiny dump for waveform / hand check
    lines = [
        f"B3 EarlyPool golden  T={shapes.T} D={shapes.D} W={shapes.W} N={shapes.N} M={shapes.M}",
        f"E shape {stages['E'].shape}  E[0,:min4]={stages['E'][0, : min(4, shapes.W)]}",
        f"S0_x_m0_n0={stages['S0_x_m0_n0']}",
        f"S1_pre_linear_m0_n0={stages['S1_pre_linear_m0_n0']}",
        f"S2_feat_m0_n0={stages['S2_feat_m0_n0']}",
        f"S7_E_n0={stages['S7_E_n0']}",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def max_abs_rel(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = np.maximum(np.abs(b.astype(np.float64)), 1e-12)
    return float(diff.max()), float((diff / denom).max())


def compare_dirs(ref_dir: Path, dut_dir: Path, atol: float, rtol: float) -> int:
    ref_stages = ref_dir / "stages"
    dut_stages = dut_dir / "stages"
    if not dut_stages.is_dir():
        # allow flat E.npy-only DUT
        dut_E = np.load(dut_dir / "E.npy")
        ref_E = np.load(ref_dir / "E.npy")
        mad, mrd = max_abs_rel(dut_E, ref_E)
        ok = mad <= atol or mrd <= rtol
        print(f"E-only compare: max_abs={mad:.6e} max_rel={mrd:.6e} {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    keys = sorted(p.stem for p in ref_stages.glob("*.npy"))
    failed: List[str] = []
    for key in keys:
        dut_path = dut_stages / f"{key}.npy"
        if not dut_path.exists():
            print(f"[SKIP] {key}: missing in DUT")
            continue
        ref = np.load(ref_stages / f"{key}.npy")
        dut = np.load(dut_path)
        if ref.shape != dut.shape:
            print(f"[FAIL] {key}: shape {dut.shape} != ref {ref.shape}")
            failed.append(key)
            continue
        mad, mrd = max_abs_rel(dut, ref)
        ok = bool(np.allclose(dut, ref, atol=atol, rtol=rtol))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {key}: shape={ref.shape} max_abs={mad:.6e} max_rel={mrd:.6e}")
        if not ok:
            failed.append(key)
    if failed:
        print(f"FAILED stages: {failed}")
        return 1
    print("All compared stages PASS")
    return 0


# ---------------------------------------------------------------------------
# Optional PyTorch cross-check
# ---------------------------------------------------------------------------

def torch_cross_check(shapes: B3Shapes, calib: np.ndarray, weights: B3Weights, stages: Dict[str, np.ndarray]) -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit(f"--torch-check requires torch: {exc}") from exc

    from early_pool_encoder import EarlyPoolEncoder

    enc = EarlyPoolEncoder(shapes.T, shapes.W, shapes.D, num_post_layers=3)
    with torch.no_grad():
        # load numpy weights into modules
        enc.pre_pool[0].weight.copy_(torch.from_numpy(weights.pre_w))
        enc.pre_pool[0].bias.copy_(torch.from_numpy(weights.pre_b))
        # post_pool: Linear, ReLU, Linear, ReLU, Linear
        linears = [m for m in enc.post_pool if isinstance(m, nn.Linear)]
        assert len(linears) == 3
        linears[0].weight.copy_(torch.from_numpy(weights.post0_w))
        linears[0].bias.copy_(torch.from_numpy(weights.post0_b))
        linears[1].weight.copy_(torch.from_numpy(weights.post1_w))
        linears[1].bias.copy_(torch.from_numpy(weights.post1_b))
        linears[2].weight.copy_(torch.from_numpy(weights.post2_w))
        linears[2].bias.copy_(torch.from_numpy(weights.post2_b))

        # calib to encoder: [B,M,T,N]
        t_calib = torch.from_numpy(calib).unsqueeze(0)
        E_t = enc.forward_batch(t_calib).squeeze(0).cpu().numpy()

    mad, mrd = max_abs_rel(stages["E"], E_t)
    ok = np.allclose(stages["E"], E_t, atol=1e-5, rtol=1e-5)
    print(f"torch EarlyPoolEncoder vs numpy: max_abs={mad:.6e} max_rel={mrd:.6e} {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)

    # streaming API path
    state = enc.reset_stream(1, shapes.N, t_calib.device, t_calib.dtype)
    with torch.no_grad():
        for m in range(shapes.M):
            state = enc.push_trial(state, t_calib[:, m])
        E_s = enc.finalize_identity(state).squeeze(0).cpu().numpy()
    ok2 = np.allclose(stages["E"], E_s, atol=1e-5, rtol=1e-5)
    mad2, mrd2 = max_abs_rel(stages["E"], E_s)
    print(f"torch streaming vs numpy: max_abs={mad2:.6e} max_rel={mrd2:.6e} {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_shapes(args: argparse.Namespace) -> B3Shapes:
    if args.profile:
        if args.profile not in PROFILES:
            raise SystemExit(f"Unknown profile {args.profile}; choose from {list(PROFILES)}")
        base = dict(PROFILES[args.profile])
    else:
        base = dict(PROFILES["tiny"])
    for key in ("T", "D", "W", "N", "M", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            base[key] = val
    return B3Shapes(**base)


def cmd_export(args: argparse.Namespace) -> int:
    shapes = parse_shapes(args)
    rng = np.random.default_rng(shapes.seed)

    if args.weights_dir:
        weights = load_weights(Path(args.weights_dir))
    else:
        weights = random_weights(shapes, rng)

    if args.calib:
        calib = np.load(args.calib).astype(np.float32)
        if calib.shape != (shapes.M, shapes.T, shapes.N):
            raise SystemExit(f"calib shape {calib.shape} != {(shapes.M, shapes.T, shapes.N)}")
    else:
        calib = random_calib(shapes, rng)

    stages = forward_b3_layered(calib, weights)
    E_stream = streaming_forward_matches_batch(calib, weights)
    if not np.allclose(stages["E"], E_stream, atol=1e-6):
        raise SystemExit("internal error: batch vs streaming mismatch")

    out = Path(args.out)
    save_run(out, shapes, weights, stages)
    print(f"Wrote golden to {out}")
    print((out / "summary.txt").read_text(encoding="utf-8"))

    if args.torch_check:
        torch_cross_check(shapes, calib, weights, stages)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=list(PROFILES), default="tiny")
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--W", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="runs/default_sw")
    parser.add_argument("--weights-dir", type=str, default=None, help="Load weights/*.npy instead of random")
    parser.add_argument("--calib", type=str, default=None, help="Load calib [M,T,N] .npy")
    parser.add_argument("--torch-check", action="store_true")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("REF_DIR", "DUT_DIR"),
        help="Compare DUT stage dumps against REF golden dir",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.compare:
        return compare_dirs(Path(args.compare[0]), Path(args.compare[1]), args.atol, args.rtol)
    return cmd_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
