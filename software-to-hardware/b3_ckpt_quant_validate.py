#!/usr/bin/env python3
"""Quantize a real B3 checkpoint and compare INT8 vs FP32 on actual calibration data.

This is the validation you should run before RTL — not random-weight smoke tests.

Examples:
  # Use repo checkpoint + Falcon data (paths relative to SPINT root)
  python b3_ckpt_quant_validate.py \\
    --ckpt ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt \\
    --exp-root ../streaming_calibration_exp \\
    --data-dir ../SPINT-main/data/000953 \\
    --loso-fold 0 --out runs/b3_d64_anchor_quant

  # Pre-exported calib [M,T,N]
  python b3_ckpt_quant_validate.py --ckpt best.ckpt --calib session_calib.npy --out runs/from_npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from b3_ckpt_loader import load_b3_weights_from_ckpt, load_hyperparams_from_ckpt
from b3_hw_golden import B3Shapes, forward_b3_layered, save_run as save_fp32_run
from b3_quant import (
    build_quant_bundle,
    compare_fp32_vs_int8,
    forward_b3_int8_layered,
    save_quant_run,
)


def _infer_shapes_from_weights(weights, M: int) -> B3Shapes:
    D, T = weights.pre_w.shape
    W = weights.post2_w.shape[0]
    return B3Shapes(T=int(T), D=int(D), W=int(W), N=96, M=int(M), seed=0)


def fetch_calib_sessions(
    exp_root: Path,
    data_dir: Path,
    *,
    validation_protocol: str = "minival",
    loso_fold: int = 0,
    calibration_n_trials: int = 33,
    max_sessions: int = 16,
    seed: int = 42,
) -> List[Tuple[str, np.ndarray]]:
    """Return list of (session_name, calib[M,T,N]) from val/test loaders."""
    import torch

    exp_root = exp_root.resolve()
    if not (exp_root / "src" / "data" / "falcon_datamodule.py").is_file():
        raise FileNotFoundError(f"Invalid --exp-root: {exp_root}")

    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))

    torch.manual_seed(seed)
    from src.data.falcon_datamodule import FalconDataModule

    dm = FalconDataModule(
        task="m2",
        data_dir=str(data_dir.resolve()),
        heldin_session_names=[""],
        batch_size=1,
        window_size=50,
        calibration_n_trials=calibration_n_trials,
        random_calibration=True,
        smooth_calibration=False,
        max_trial_length=100,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        validation_protocol=validation_protocol,
        loso_fold=loso_fold if validation_protocol == "loso" else None,
        rotation_id=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("test")

    out: List[Tuple[str, np.ndarray]] = []
    seen_sessions: set[str] = set()

    def collect(loader, split_name: str) -> None:
        if loader is None:
            return
        for batch_idx, batch in enumerate(loader):
            if len(out) >= max_sessions:
                break
            _neural, _behavior, calib, session_names = batch
            session = session_names[0]
            if session in seen_sessions:
                continue
            seen_sessions.add(session)
            arr = calib[0].detach().cpu().numpy().astype(np.float32)
            out.append((f"{split_name}:{session}", arr))

    collect(dm.val_dataloader(), "val_heldin")
    if hasattr(dm, "test_dataloader"):
        try:
            collect(dm.test_dataloader(), "test")
        except Exception:
            pass
    return out


def validate_one_session(
    name: str,
    calib: np.ndarray,
    weights,
    out_root: Path,
    recip_shift: int,
) -> Dict[str, Any]:
    M, T, N = calib.shape
    shapes = B3Shapes(T=T, D=weights.pre_w.shape[0], W=weights.post2_w.shape[0], N=N, M=M)

    fp_stages = forward_b3_layered(calib, weights)
    bundle = build_quant_bundle(weights, calib, shapes, fp_stages, recip_shift=recip_shift)
    int8_out = forward_b3_int8_layered(calib, bundle)
    report = compare_fp32_vs_int8(fp_stages, int8_out, bundle)

    safe = name.replace("/", "_").replace(":", "_")
    sess_dir = out_root / safe
    save_fp32_run(sess_dir / "fp32", shapes, weights, fp_stages)
    save_quant_run(sess_dir / "int8", bundle, int8_out, fp_stages, report)

    E_fp = fp_stages["E"]
    e8 = int8_out["E_int8_dequant"]
    e16 = int8_out["E_int16_dequant"]
    diff8 = np.abs(e8 - E_fp)
    diff16 = np.abs(e16 - E_fp)

    return {
        "session": name,
        "shapes": {"M": M, "T": T, "N": N, "D": shapes.D, "W": shapes.W},
        "E_fp32_std": float(E_fp.std()),
        "E_int8_dequant": report["E_int8_dequant"],
        "E_int16_dequant": report["E_int16_dequant"],
        "E_int8_cosine": float(
            np.dot(e8.ravel(), E_fp.ravel()) / (np.linalg.norm(e8) * np.linalg.norm(E_fp) + 1e-12)
        ),
        "E_int16_cosine": float(
            np.dot(e16.ravel(), E_fp.ravel()) / (np.linalg.norm(e16) * np.linalg.norm(E_fp) + 1e-12)
        ),
        "per_stage": {k: v for k, v in report.items() if k.startswith("stage_")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="B3 Lightning checkpoint (best.ckpt)")
    parser.add_argument("--calib", default=None, help="Single calib tensor [M,T,N] .npy")
    parser.add_argument("--exp-root", default=None, help="streaming_calibration_exp root for Falcon data")
    parser.add_argument("--data-dir", default=None, help="Falcon data dir, e.g. .../SPINT-main/data/000953")
    parser.add_argument("--validation-protocol", choices=["minival", "loso"], default="minival")
    parser.add_argument("--loso-fold", type=int, default=0)
    parser.add_argument("--calibration-n-trials", type=int, default=33)
    parser.add_argument("--max-sessions", type=int, default=16)
    parser.add_argument("--recip-shift", type=int, default=20)
    parser.add_argument("--out", type=str, default="runs/b3_ckpt_quant")
    parser.add_argument("--session-label", default="calib_npy", help="Label when using --calib")
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    weights = load_b3_weights_from_ckpt(ckpt)
    hp = load_hyperparams_from_ckpt(ckpt)
    print(f"Loaded B3 ckpt: {ckpt}")
    print(f"  variant={hp.get('variant')} hidden_dim={hp.get('hidden_dim')} trial_length={hp.get('trial_length')}")
    print(f"  pre_w {weights.pre_w.shape} post2_w {weights.post2_w.shape}")

    sessions: List[Tuple[str, np.ndarray]] = []
    if args.calib:
        calib = np.load(args.calib).astype(np.float32)
        sessions = [(args.session_label, calib)]
    elif args.exp_root and args.data_dir:
        sessions = fetch_calib_sessions(
            Path(args.exp_root),
            Path(args.data_dir),
            validation_protocol=args.validation_protocol,
            loso_fold=args.loso_fold,
            calibration_n_trials=args.calibration_n_trials,
            max_sessions=args.max_sessions,
        )
    else:
        raise SystemExit("Provide --calib [M,T,N].npy OR both --exp-root and --data-dir")

    if not sessions:
        raise SystemExit("No calibration sessions collected")

    results = []
    for name, calib in sessions:
        print(f"\n== Session {name} calib{calib.shape} ==")
        row = validate_one_session(name, calib, weights, out_root, args.recip_shift)
        results.append(row)
        e8 = row["E_int8_dequant"]
        e16 = row["E_int16_dequant"]
        print(
            f"  E vs FP32: int8 max_abs={e8['max_abs']:.4e} rmse={e8['rmse']:.4e} | "
            f"int16 max_abs={e16['max_abs']:.4e} rmse={e16['rmse']:.4e}"
        )
        print(f"  cosine: int8={row['E_int8_cosine']:.6f} int16={row['E_int16_cosine']:.6f}")

    # aggregate
    agg = {
        "ckpt": str(ckpt.resolve()),
        "hyperparams": hp,
        "num_sessions": len(results),
        "sessions": results,
        "aggregate": {
            "E_int8_max_abs_mean": float(np.mean([r["E_int8_dequant"]["max_abs"] for r in results])),
            "E_int8_max_abs_p95": float(np.percentile([r["E_int8_dequant"]["max_abs"] for r in results], 95)),
            "E_int8_rmse_mean": float(np.mean([r["E_int8_dequant"]["rmse"] for r in results])),
            "E_int16_max_abs_mean": float(np.mean([r["E_int16_dequant"]["max_abs"] for r in results])),
            "E_int16_rmse_mean": float(np.mean([r["E_int16_dequant"]["rmse"] for r in results])),
            "E_int8_cosine_mean": float(np.mean([r["E_int8_cosine"] for r in results])),
            "E_int16_cosine_mean": float(np.mean([r["E_int16_cosine"] for r in results])),
        },
        "quant_policy": "see B3_INT8_quantization_baseline.md",
    }
    (out_root / "ckpt_quant_report.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    lines = [
        f"B3 checkpoint INT8 validation",
        f"ckpt: {ckpt}",
        f"sessions: {len(results)}",
        f"E_int8  mean max_abs={agg['aggregate']['E_int8_max_abs_mean']:.4e} mean rmse={agg['aggregate']['E_int8_rmse_mean']:.4e}",
        f"E_int16 mean max_abs={agg['aggregate']['E_int16_max_abs_mean']:.4e} mean rmse={agg['aggregate']['E_int16_rmse_mean']:.4e}",
        f"cosine mean int8={agg['aggregate']['E_int8_cosine_mean']:.6f} int16={agg['aggregate']['E_int16_cosine_mean']:.6f}",
        "",
        "Per session:",
    ]
    for r in results:
        e8 = r["E_int8_dequant"]
        lines.append(f"  {r['session']}: int8 max_abs={e8['max_abs']:.4e} rmse={e8['rmse']:.4e}")
    (out_root / "ckpt_quant_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + (out_root / "ckpt_quant_summary.txt").read_text(encoding="utf-8"))
    print(f"Full report: {out_root / 'ckpt_quant_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
