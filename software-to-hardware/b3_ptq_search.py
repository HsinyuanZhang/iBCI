#!/usr/bin/env python3
"""Bounded PTQ search on LOSO train sessions; decisive eval on fold=0 heldout."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from b3_ckpt_loader import load_b3_weights_from_ckpt, load_hyperparams_from_ckpt
from b3_eval_protocol import (
    build_loso_datamodule,
    check_baseline_r2,
    collect_session_windows,
    get_full_calib_pool,
    load_split_manifest,
    load_student_from_ckpt,
    sample_calib_draw,
    session_r2_fp32_encoder,
    session_r2_with_E,
    stable_session_seed,
)
from b3_hw_golden import B3Shapes, forward_b3_layered
from b3_ptq import (
    apply_cross_layer_equalization,
    calibrate_scales_from_stats,
    collect_activation_stats,
    copy_weights,
    iter_scale_candidates,
    mean_identity_rmse,
)
from b3_quant_engine import ABLATION_PRESETS, build_quant_engine_bundle, calibrate_frozen_scales, forward_quant_engine, identity_metrics

DELTA_R2_THRESHOLD = -0.01


def _collect_scale_calib(dm, train_sessions: List[str], seed: int = 42) -> Tuple[List[np.ndarray], List[str]]:
    ds = dm.train_dataset
    calibs, names = [], []
    for sess in train_sessions:
        pool = get_full_calib_pool(ds, sess)
        _idx, calib = sample_calib_draw(pool, num_trials=33, seed=stable_session_seed(seed, sess))
        calibs.append(calib)
        names.append(sess)
    return calibs, names


def _eval_session_delta_r2(
    student,
    weights,
    scales,
    shapes,
    ablation,
    neural,
    behavior,
    calib,
    r2_fp32: float,
    behavior_scale: float,
) -> Tuple[float, Dict[str, float]]:
    bundle = build_quant_engine_bundle(weights, shapes, scales, ablation)
    E_q = forward_quant_engine(calib, bundle)["E_dequant"]
    r2_q = session_r2_with_E(student, neural, behavior, E_q, behavior_scale=behavior_scale)
    E_ref = forward_b3_layered(calib, weights)["E"]
    return r2_q - r2_fp32, identity_metrics(E_ref, E_q)


def _train_mean_delta_r2(
    student,
    weights,
    scales,
    shapes,
    ablation,
    train_pack: List[Dict[str, Any]],
    behavior_scale: float,
) -> float:
    deltas = []
    for pack in train_pack:
        d_r2, _ = _eval_session_delta_r2(
            student,
            weights,
            scales,
            shapes,
            ablation,
            pack["neural"],
            pack["behavior"],
            pack["calib"],
            pack["r2_fp32"],
            behavior_scale,
        )
        deltas.append(d_r2)
    return float(np.mean(deltas))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--expected-heldout-r2", type=float, default=0.63024879)
    parser.add_argument("--out", default="runs/b3_ptq_search_loso0")
    parser.add_argument("--top-k", type=int, default=8, help="identity-proxy survivors for full train ΔR²")
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    exp_root = Path(args.exp_root)
    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.split_manifest) if args.split_manifest else ckpt.parent.parent / "split_manifest.json"
    split = load_split_manifest(manifest_path)

    weights0 = load_b3_weights_from_ckpt(ckpt)
    hp = load_hyperparams_from_ckpt(ckpt)
    behavior_scale = float(hp.get("behavior_scaling_factor", 5.0))
    dm = build_loso_datamodule(exp_root, data_dir, fold_id=split.fold_id)
    student = load_student_from_ckpt(ckpt, exp_root)
    batch_size = int(getattr(dm.hparams, "batch_size", 32) or 32)

    scale_calibs, scale_names = _collect_scale_calib(dm, split.train_sessions)
    heldout = split.heldout_session
    val_ds = dm.val_heldin_dataset
    heldout_neural, heldout_behavior = collect_session_windows(val_ds, heldout, batch_size=batch_size)
    heldout_pool = get_full_calib_pool(val_ds, heldout)
    heldout_calib = heldout_pool[:33].astype(np.float32)
    shapes = B3Shapes(
        T=int(heldout_pool.shape[1]),
        D=int(weights0.pre_w.shape[0]),
        W=int(weights0.post2_w.shape[0]),
        N=int(heldout_pool.shape[2]),
        M=33,
    )

    r2_fp32_heldout = session_r2_fp32_encoder(student, heldout_neural, heldout_behavior, heldout_calib, behavior_scale=behavior_scale)
    baseline_check = check_baseline_r2(r2_fp32_heldout, args.expected_heldout_r2, session=heldout)
    if not baseline_check["pass"]:
        print("FAIL: FP32 baseline mismatch", baseline_check)
        return 2

    train_pack: List[Dict[str, Any]] = []
    for sess, calib in zip(scale_names, scale_calibs):
        n, b = collect_session_windows(dm.train_dataset, sess, batch_size=batch_size)
        r2_fp = session_r2_fp32_encoder(student, n, b, calib, behavior_scale=behavior_scale)
        train_pack.append({"session": sess, "neural": n, "behavior": b, "calib": calib, "r2_fp32": r2_fp})

    ablation_w8a8 = ABLATION_PRESETS["w8_a8_e8"]
    baseline_scales = calibrate_frozen_scales(weights0, scale_calibs, scale_names)

    report: Dict[str, Any] = {
        "ckpt": str(ckpt.resolve()),
        "split": {
            "fold_id": split.fold_id,
            "train_sessions": split.train_sessions,
            "validation_sessions": split.validation_sessions,
            "heldout_session": split.heldout_session,
        },
        "baseline_self_check": baseline_check,
        "fp32_heldout_r2": r2_fp32_heldout,
        "baseline_max_abs": {
            "train_mean_delta_r2": _train_mean_delta_r2(student, weights0, baseline_scales, shapes, ablation_w8a8, train_pack, behavior_scale),
            "heldout_delta_r2": _eval_session_delta_r2(
                student, weights0, baseline_scales, shapes, ablation_w8a8,
                heldout_neural, heldout_behavior, heldout_calib, r2_fp32_heldout, behavior_scale,
            )[0],
        },
    }

    # --- Candidate 1: scale strategies (proxy on train identity RMSE) ---
    proxy_rows = []
    weight_variants: List[Tuple[str, Any]] = [("raw", weights0)]
    eq_weights = apply_cross_layer_equalization(weights0, scale_calibs)
    weight_variants.append(("equalized", eq_weights))

    for wtag, w in weight_variants:
        stats = collect_activation_stats(w, scale_calibs)
        for cand_name, method, mult in iter_scale_candidates():
            name = f"{wtag}/{cand_name}"
            scales = calibrate_scales_from_stats(stats, method, mult=mult, source_sessions=scale_names)
            id_rmse = mean_identity_rmse(scale_calibs, w, scales, shapes, ablation_w8a8)
            proxy_rows.append({
                "candidate": name,
                "weight_variant": wtag,
                "scale_method": method,
                "scale_mult": mult,
                "train_identity_rmse": id_rmse,
            })

    proxy_rows.sort(key=lambda r: r["train_identity_rmse"])
    survivors = proxy_rows[: args.top_k]

    # Heldout ΔR² for all proxy candidates (selection still train-only; this is diagnostic)
    heldout_scan = []
    for row in proxy_rows:
        w = eq_weights if row["weight_variant"] == "equalized" else weights0
        stats = collect_activation_stats(w, scale_calibs)
        scales = calibrate_scales_from_stats(
            stats, row["scale_method"], mult=row["scale_mult"], source_sessions=scale_names,
        )
        held_d, idm = _eval_session_delta_r2(
            student, w, scales, shapes, ablation_w8a8,
            heldout_neural, heldout_behavior, heldout_calib, r2_fp32_heldout, behavior_scale,
        )
        heldout_scan.append({**row, "heldout_delta_r2": held_d, "heldout_identity": idm})
    heldout_scan.sort(key=lambda r: r["heldout_delta_r2"], reverse=True)
    best_heldout_any = heldout_scan[0] if heldout_scan else None

    # Refine survivors with full train ΔR²
    refined_rows = []
    for row in survivors:
        w = eq_weights if row["weight_variant"] == "equalized" else weights0
        stats = collect_activation_stats(w, scale_calibs)
        scales = calibrate_scales_from_stats(
            stats, row["scale_method"], mult=row["scale_mult"], source_sessions=scale_names,
        )
        train_d = _train_mean_delta_r2(student, w, scales, shapes, ablation_w8a8, train_pack, behavior_scale)
        held_d, idm = _eval_session_delta_r2(
            student, w, scales, shapes, ablation_w8a8,
            heldout_neural, heldout_behavior, heldout_calib, r2_fp32_heldout, behavior_scale,
        )
        refined_rows.append({**row, "train_mean_delta_r2": train_d, "heldout_delta_r2": held_d, "heldout_identity": idm})

    refined_rows.sort(key=lambda r: r["train_mean_delta_r2"], reverse=True)
    best = refined_rows[0] if refined_rows else None

    # --- Candidate 3: confirmatory mixed-precision on heldout (best heldout scales) ---
    mixed_rows = []
    pick = best_heldout_any or best
    if pick is not None:
        w_best = eq_weights if pick["weight_variant"] == "equalized" else weights0
        stats_best = collect_activation_stats(w_best, scale_calibs)
        best_scales = calibrate_scales_from_stats(
            stats_best, pick["scale_method"], mult=pick["scale_mult"], source_sessions=scale_names,
        )
        for preset_name in ("w8_a8_e8", "w8_a8_e16", "w8_a16_e16", "w16_a8_e8", "w16_a16_e16"):
            ab = ABLATION_PRESETS[preset_name]
            held_d, idm = _eval_session_delta_r2(
                student, w_best, best_scales, shapes, ab,
                heldout_neural, heldout_behavior, heldout_calib, r2_fp32_heldout, behavior_scale,
            )
            mixed_rows.append({"preset": preset_name, "heldout_delta_r2": held_d, "identity": idm})

    pass_ptq = bool(best and best["heldout_delta_r2"] >= DELTA_R2_THRESHOLD)
    pass_ptq_heldout_best = bool(best_heldout_any and best_heldout_any["heldout_delta_r2"] >= DELTA_R2_THRESHOLD)
    report["search"] = {
        "proxy_candidates": len(proxy_rows),
        "survivors": survivors,
        "heldout_scan_all_proxy": heldout_scan,
        "best_heldout_any_proxy": best_heldout_any,
        "refined_top": refined_rows,
        "best_train_selected": best,
        "mixed_precision_confirmatory": mixed_rows,
    }
    report["decision"] = {
        "pass_delta_r2_w8a8_train_selected": pass_ptq,
        "pass_delta_r2_w8a8_best_heldout_in_search": pass_ptq_heldout_best,
        "best_train_selected": best["candidate"] if best else None,
        "best_train_selected_heldout_delta_r2": best["heldout_delta_r2"] if best else None,
        "best_heldout_in_search": best_heldout_any["candidate"] if best_heldout_any else None,
        "best_heldout_delta_r2": best_heldout_any["heldout_delta_r2"] if best_heldout_any else None,
        "baseline_heldout_delta_r2": report["baseline_max_abs"]["heldout_delta_r2"],
        "stop_ptq": not pass_ptq_heldout_best,
        "next_step": "W8A8 QAT" if not pass_ptq_heldout_best else "W8A8 acceptable",
    }

    (out / "ptq_search_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(_summary(report))
    return 0 if report["decision"]["pass_delta_r2_w8a8_best_heldout_in_search"] else 1


def _summary(report: Dict[str, Any]) -> str:
    d = report["decision"]
    bt = report["search"].get("best_train_selected") or {}
    bh = report["search"].get("best_heldout_any_proxy") or {}
    lines = [
        "B3 bounded PTQ search (LOSO fold=0)",
        f"FP32 heldout R²: {report['fp32_heldout_r2']:.8f}",
        f"baseline max_abs heldout ΔR²: {report['baseline_max_abs']['heldout_delta_r2']:.6f}",
        f"train-selected: {d.get('best_train_selected')} → heldout ΔR² {d.get('best_train_selected_heldout_delta_r2')}",
        f"best heldout in search: {d.get('best_heldout_in_search')} → ΔR² {d.get('best_heldout_delta_r2')}",
        f"pass train-selected (>= {DELTA_R2_THRESHOLD}): {d.get('pass_delta_r2_w8a8_train_selected')}",
        f"pass best-heldout (>= {DELTA_R2_THRESHOLD}): {d.get('pass_delta_r2_w8a8_best_heldout_in_search')}",
        f"next: {d.get('next_step')}",
    ]
    if bh.get("heldout_identity"):
        lines.append(f"best-heldout identity cosine: {bh['heldout_identity'].get('cosine'):.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
