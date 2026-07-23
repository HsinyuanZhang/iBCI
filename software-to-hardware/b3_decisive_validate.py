#!/usr/bin/env python3
"""Corrected decisive B3 quantization validation (LOSO fold=0 matched)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from b3_ckpt_loader import load_b3_weights_from_ckpt, load_hyperparams_from_ckpt
from b3_eval_protocol import (
    BASELINE_R2_TOLERANCE,
    build_loso_datamodule,
    calib_sha256,
    check_baseline_r2,
    collect_session_windows,
    get_dataset_for_session,
    get_full_calib_pool,
    load_split_manifest,
    load_student_from_ckpt,
    sample_calib_draw,
    session_r2_fp32_encoder,
    session_r2_with_E,
    stable_session_seed,
)
from b3_hw_golden import B3Shapes, forward_b3_layered
from b3_quant_engine import (
    ABLATION_PRESETS,
    build_quant_engine_bundle,
    calibrate_frozen_scales,
    forward_quant_engine,
    identity_metrics,
)

EXPECTED_HELDOUT_R2 = {
    ("ses-2020-10-19-Run1", 0): 0.63024879,
}


def _collect_scale_calib(dm, train_sessions: List[str], seed: int = 42) -> tuple[list[np.ndarray], list[str]]:
    ds = dm.train_dataset
    calibs, names = [], []
    for sess in train_sessions:
        pool = get_full_calib_pool(ds, sess)
        _idx, calib = sample_calib_draw(pool, num_trials=33, seed=stable_session_seed(seed, sess))
        calibs.append(calib)
        names.append(sess)
    return calibs, names


def _aggregate_diag(diag_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    overflow = 0
    sat = 0
    acc_max = 0
    for d in diag_list:
        for section in (d.get("pre_pool", {}), *d.get("layers", {}).values()):
            if not isinstance(section, dict):
                continue
            overflow += int(section.get("acc_i32_overflow", 0))
            sat += int(section.get("saturation_count", 0))
            acc_max = max(acc_max, int(section.get("acc_i64_max", 0)))
        overflow += int(d.get("input", {}).get("clip_count", 0))
    return {"acc_i32_overflow_total": overflow, "activation_saturation_total": sat, "acc_i64_max": acc_max}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-manifest", default=None, help="defaults to <ckpt_dir>/../split_manifest.json")
    parser.add_argument("--expected-heldout-r2", type=float, default=0.63024879)
    parser.add_argument("--out", default="runs/b3_decisive_loso0")
    parser.add_argument("--calib-seeds", default="42,43,44,45,46")
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    exp_root = Path(args.exp_root)
    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.split_manifest) if args.split_manifest else ckpt.parent.parent / "split_manifest.json"
    split = load_split_manifest(manifest_path)
    seeds = [int(s) for s in args.calib_seeds.split(",") if s.strip()]

    weights = load_b3_weights_from_ckpt(ckpt)
    hp = load_hyperparams_from_ckpt(ckpt)
    dm = build_loso_datamodule(exp_root, data_dir, fold_id=split.fold_id)
    student = load_student_from_ckpt(ckpt, exp_root)
    batch_size = int(getattr(dm.hparams, "batch_size", 32) or 32)

    # --- scale calibration: 6 train sessions only (no heldout leakage) ---
    scale_calibs, scale_names = _collect_scale_calib(dm, split.train_sessions)
    frozen_scales = calibrate_frozen_scales(weights, scale_calibs, scale_names)

    heldout = split.heldout_session
    val_ds = get_dataset_for_session(dm, heldout, split="val")
    neural, behavior = collect_session_windows(val_ds, heldout, batch_size=batch_size)
    pool = get_full_calib_pool(val_ds, heldout)

    # Default val calib = first 33 trials (matches random_calibration=False)
    default_calib = pool[:33].astype(np.float32)
    shapes = B3Shapes(T=int(pool.shape[1]), D=int(weights.pre_w.shape[0]), W=int(weights.post2_w.shape[0]), N=int(pool.shape[2]), M=33)

    report: Dict[str, Any] = {
        "ckpt": str(ckpt.resolve()),
        "hyperparams": hp,
        "split": asdict_split(split),
        "frozen_scales": frozen_scales.__dict__,
        "scale_calibration_sessions": scale_names,
        "heldout_session": heldout,
    }

    # --- Step 0: FP32 baseline self-check (HARD FAIL) ---
    r2_fp32 = session_r2_fp32_encoder(student, neural, behavior, default_calib, behavior_scale=hp.get("behavior_scaling_factor", 5.0))
    baseline_check = check_baseline_r2(r2_fp32, args.expected_heldout_r2, session=heldout)
    report["baseline_self_check"] = baseline_check
    if not baseline_check["pass"]:
        report["decision"] = {"valid": False, "reason": "FP32 baseline mismatch; abort before ΔR²"}
        _write_report(out, report)
        print("FAIL: FP32 baseline mismatch", baseline_check)
        return 2

    E_fp32_ref = forward_b3_layered(default_calib, weights)["E"]

    # --- Exp 1/2/6: ablations on heldout with frozen scales ---
    ablation_rows = []
    heldout_diags = []
    for preset_name, ablation in ABLATION_PRESETS.items():
        bundle = build_quant_engine_bundle(weights, shapes, frozen_scales, ablation)
        qout = forward_quant_engine(default_calib, bundle)
        idm = identity_metrics(E_fp32_ref, qout["E_dequant"])
        r2_q = session_r2_with_E(student, neural, behavior, qout["E_dequant"], behavior_scale=hp.get("behavior_scaling_factor", 5.0))
        diag = _aggregate_diag([qout["diagnostics"]])
        heldout_diags.append({"preset": preset_name, **diag})
        ablation_rows.append({
            "preset": preset_name,
            "identity": idm,
            "r2": r2_q,
            "delta_r2": r2_q - r2_fp32,
            "diagnostics": diag,
        })

    report["experiments"] = {
        "ablation_heldout_frozen_scales": ablation_rows,
        "fp32_baseline_r2": r2_fp32,
    }

    # --- Exp 5: true multi-draw on heldout ---
    draw_rows = []
    prev_sha = None
    all_indices_distinct = True
    all_sha_distinct = True
    for seed in seeds:
        indices, calib_draw = sample_calib_draw(pool, num_trials=33, seed=seed)
        sha = calib_sha256(calib_draw)
        if prev_sha is not None and sha == prev_sha:
            all_sha_distinct = False
        prev_sha = sha
        bundle = build_quant_engine_bundle(weights, shapes, frozen_scales, ABLATION_PRESETS["w8_a8_e8"])
        qout = forward_quant_engine(calib_draw, bundle)
        r2_q = session_r2_with_E(student, neural, behavior, qout["E_dequant"], behavior_scale=hp.get("behavior_scaling_factor", 5.0))
        draw_rows.append({
            "seed": seed,
            "trial_indices": indices.tolist(),
            "calib_sha256": sha,
            "identity": identity_metrics(E_fp32_ref, qout["E_dequant"]),
            "r2": r2_q,
            "delta_r2": r2_q - r2_fp32,
            "diagnostics": _aggregate_diag([qout["diagnostics"]]),
        })
    indices_sets = [tuple(r["trial_indices"]) for r in draw_rows]
    if len(set(indices_sets)) < len(indices_sets):
        all_indices_distinct = False
    multi_draw_valid = all_indices_distinct and all_sha_distinct
    report["experiments"]["multi_draw_heldout"] = {
        "draws": draw_rows,
        "indices_all_distinct": all_indices_distinct,
        "sha256_all_distinct": all_sha_distinct,
        "valid": multi_draw_valid,
    }

    # --- Overflow/saturation across train sessions + heldout + ablations ---
    coverage = []
    targets = [(s, "train_scale_cal") for s in split.train_sessions] + [(heldout, "heldout")]
    for sess, role in targets:
        ds = dm.train_dataset if role == "train_scale_cal" else val_ds
        pool_s = get_full_calib_pool(ds, sess)
        calib_s = pool_s[:33]
        for preset_name in ("w8_a8_e8", "w8_a16_e16", "w16_a16_e16"):
            bundle = build_quant_engine_bundle(weights, shapes, frozen_scales, ABLATION_PRESETS[preset_name])
            qout = forward_quant_engine(calib_s, bundle)
            coverage.append({"session": sess, "role": role, "preset": preset_name, **_aggregate_diag([qout["diagnostics"]])})
    report["experiments"]["overflow_saturation_coverage"] = coverage

    # --- Optional minival robustness (7 sessions, not decisive) ---
    minival_rows = []
    for sess in split.train_sessions + [heldout]:
        ds = dm.train_dataset if sess in split.train_sessions else val_ds
        try:
            n_s, b_s = collect_session_windows(ds, sess, batch_size=batch_size)
        except ValueError:
            continue
        calib_s = get_full_calib_pool(ds, sess)[:33]
        bundle = build_quant_engine_bundle(weights, shapes, frozen_scales, ABLATION_PRESETS["w8_a8_e8"])
        qout = forward_quant_engine(calib_s, bundle)
        r2_q = session_r2_with_E(student, n_s, b_s, qout["E_dequant"], behavior_scale=hp.get("behavior_scaling_factor", 5.0))
        r2_fp = session_r2_fp32_encoder(student, n_s, b_s, calib_s, behavior_scale=hp.get("behavior_scaling_factor", 5.0))
        minival_rows.append({"session": sess, "r2_fp32": r2_fp, "r2_int8": r2_q, "delta_r2": r2_q - r2_fp})
    report["experiments"]["minival_robustness_w8a8"] = minival_rows

    w8 = next(r for r in ablation_rows if r["preset"] == "w8_a8_e8")
    report["decision"] = {
        "valid": baseline_check["pass"] and multi_draw_valid,
        "fp32_baseline_r2": r2_fp32,
        "heldout_delta_r2_w8a8": w8["delta_r2"],
        "heldout_delta_r2_w8a16": next(r for r in ablation_rows if r["preset"] == "w8_a16_e16")["delta_r2"],
        "pass_delta_r2_w8a8": w8["delta_r2"] >= -0.01,
        "overflow_zero_global": all(c["acc_i32_overflow_total"] == 0 for c in coverage),
        "multi_draw_valid": multi_draw_valid,
    }

    _write_report(out, report)
    print(_summary_text(report))
    return 0 if report["decision"]["pass_delta_r2_w8a8"] else 1


def asdict_split(split) -> Dict[str, Any]:
    return {
        "fold_id": split.fold_id,
        "train_sessions": split.train_sessions,
        "validation_sessions": split.validation_sessions,
        "heldout_session": split.heldout_session,
    }


def _write_report(out: Path, report: Dict[str, Any]) -> None:
    (out / "decisive_quant_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _summary_text(report: Dict[str, Any]) -> str:
    d = report["decision"]
    lines = [
        "B3 decisive validation (corrected)",
        f"heldout: {report['heldout_session']}",
        f"scale cal: {report['scale_calibration_sessions']}",
        f"FP32 baseline: {report['baseline_self_check']}",
        f"valid: {d.get('valid')}",
        f"ΔR² w8_a8_e8: {d.get('heldout_delta_r2_w8a8')}",
        f"pass (>=-0.01): {d.get('pass_delta_r2_w8a8')}",
        f"multi-draw valid: {d.get('multi_draw_valid')}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
