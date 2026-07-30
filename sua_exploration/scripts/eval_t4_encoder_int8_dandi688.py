#!/usr/bin/env python3
"""Train-only PTQ and fixed-validation evaluation for the SUA T4 encoder.

Scope is intentionally narrow:

* B3S/T4 identity encoder: W8A8, INT32 accumulator, integer requant.
* The four normalized T4 values share the post0-input A8 scale with pooled
  activity and are concatenated in the integer domain.
* Decoder remains FP32.  This script therefore reports
  ``T4 encoder INT8 + FP decoder`` and never claims full-model INT8.
* Scales and scale-candidate selection use the 27 training sessions only.
  The six validation sessions are evaluated once after selection.  The six
  formal-test NWBs in the strict manifest are names/receipts only and are
  never resolved or opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_SUA = _ROOT / "sua_exploration"
_SWHW = _ROOT / "software-to-hardware"
_SCE = _ROOT / "streaming_calibration_exp"
for path in (_SUA, _SWHW, _SCE, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from b3_ckpt_loader import load_b3_weights_from_ckpt
from b3_fake_quant import B3QATScales
from b3_hw_golden import B3Shapes, forward_b3_layered
from b3_ptq import (
    calibrate_scales_from_stats,
    collect_activation_stats,
)
from b3_qat_encoder import wrap_early_pool_with_qat
from b3_quant_engine import (
    ABLATION_PRESETS,
    FrozenActivationScales,
    build_quant_engine_bundle,
    forward_quant_engine,
    identity_metrics,
)
from dandi688_gradient_free_protocol import (
    select_calibration_trial_indices,
    sha256_file,
)
from eval_adaptation_dandi688 import (
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    attach_side_features,
    build_calib_trials_for_indices,
    load_session_with_trials,
    load_side_feature_stats_for_run_metadata,
)
from mc_maze.multisession_datamodule import (
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    session_name_from_path,
)
from select_gradient_free_protocol_dandi688 import (
    evaluate_session_configs,
    load_frozen_model,
)
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder


PTQ_DELTA_R2_THRESHOLD = -0.01
SATURATION_RATE_THRESHOLD = 0.005
SCALE_CANDIDATES = (
    ("max_abs", "max_abs", 1.0),
    ("p9999_x0p8", "p9999", 0.8),
    ("p9999", "p9999", 1.0),
    ("p9999_x1p2", "p9999", 1.2),
    ("mse_opt", "mse_opt", 1.0),
)


def _sha256_jsonable(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_metadata(run_dir: Path, metadata: dict, ckpt: Path) -> None:
    side = metadata.get("side_features") or {}
    training = metadata.get("training") or {}
    checks = {
        "completed": metadata.get("status") == "completed",
        "variant_b3s": metadata.get("variant") == "B3S",
        "side_group_t4": side.get("group") == "t4",
        "side_dim_4": side.get("side_dim") == 4,
        "side_pool_30": side.get("pool_size") == 30,
        "activity_calibration_30": training.get("calibration_n_trials") == 30,
        "formal_test_not_evaluated": metadata.get("held_out_test_evaluated") is False,
        "signal_view_sua": metadata.get("signal_view") == "sua",
        "strict_manifest_recorded": bool(metadata.get("train_val_manifest")),
        "checkpoint_belongs_to_run": ckpt.parent == run_dir / "epoch_ckpts",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"T4 INT8 provenance check failed: {failed}")
    manifest = Path(metadata["train_val_manifest"]).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if sha256_file(manifest) != metadata.get("train_val_manifest_sha256"):
        raise ValueError("train_val_manifest SHA-256 differs from run metadata")
    teacher = Path(metadata["teacher_checkpoint"]).resolve()
    if sha256_file(teacher) != metadata.get("teacher_sha256"):
        raise ValueError("teacher checkpoint SHA-256 differs from run metadata")


def _load_record(
    path: Path,
    *,
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    cache_dir: Path | None,
    side_config: tuple,
) -> dict:
    rec = load_session_with_trials(
        path,
        20,
        WINDOW_SIZE,
        30,
        TRIAL_LENGTH,
        PAD_VALUE,
        behavior_mean,
        behavior_std,
        cache_dir=cache_dir,
        signal_view="sua",
    )
    (
        side_group,
        waveform_group,
        side_pool,
        permutation_seed,
        side_mean,
        side_std,
    ) = side_config
    rec = attach_side_features(
        rec,
        path,
        side_feature_group=side_group,
        waveform_feature_group=waveform_group,
        pool_size=side_pool,
        permutation_seed=permutation_seed,
        mean=side_mean,
        std=side_std,
        cache_dir=cache_dir,
    )
    indices = select_calibration_trial_indices(rec["trials"], 30, 30, "first")
    rec["calib_trials"] = build_calib_trials_for_indices(rec, indices, 30)
    if rec["calib_trials"].shape != (30, 100, rec["n_units"]):
        raise ValueError(
            f"{rec['name']}: unexpected calib shape {rec['calib_trials'].shape}"
        )
    if rec["side_features"].shape != (rec["n_units"], 4):
        raise ValueError(
            f"{rec['name']}: unexpected T4 shape {rec['side_features'].shape}"
        )
    return rec


def _select_train_only_scales(
    weights,
    calibs: list[np.ndarray],
    sides: list[np.ndarray],
    session_names: list[str],
) -> tuple[FrozenActivationScales, list[dict]]:
    stats = collect_activation_stats(
        weights, calibs, side_feature_sessions=sides
    )
    rows: list[dict] = []
    shape_template = B3Shapes(T=100, D=64, W=50, N=1, M=30)
    for name, method, mult in SCALE_CANDIDATES:
        scales = calibrate_scales_from_stats(
            stats,
            method,
            mult=mult,
            source_sessions=session_names,
        )
        bundle = build_quant_engine_bundle(
            weights,
            shape_template,
            scales,
            ABLATION_PRESETS["w8_a8_e8"],
        )
        metrics = []
        for calib, side in zip(calibs, sides):
            ref = forward_b3_layered(
                calib, weights, side_features=side
            )["E"]
            pred = forward_quant_engine(
                calib, bundle, side_features=side
            )["E_dequant"]
            metrics.append(identity_metrics(ref, pred))
        rows.append(
            {
                "name": name,
                "method": method,
                "mult": mult,
                "train_identity_rmse_mean": float(
                    np.mean([item["rmse"] for item in metrics])
                ),
                "train_identity_cosine_mean": float(
                    np.mean([item["cosine"] for item in metrics])
                ),
                "scales": asdict(scales),
            }
        )
    rows.sort(key=lambda row: row["train_identity_rmse_mean"])
    selected = FrozenActivationScales(**rows[0]["scales"])
    return selected, rows


@torch.no_grad()
def _evaluate_records(
    model,
    records: list[dict],
    *,
    device: torch.device,
) -> dict[str, float]:
    per_session: dict[str, float] = {}
    for rec in records:
        values, _ = evaluate_session_configs(
            rec,
            [("first", 30)],
            30,
            model,
            device,
        )
        per_session[rec["name"]] = values["gradient_free_calibrated_first_n30"]
    return per_session


def _export_integer_package(
    out_dir: Path,
    weights,
    scales: FrozenActivationScales,
) -> dict:
    shapes = B3Shapes(T=100, D=64, W=50, N=1, M=30)
    bundle = build_quant_engine_bundle(
        weights, shapes, scales, ABLATION_PRESETS["w8_a8_e8"]
    )
    arrays: dict[str, np.ndarray] = {}
    layer_manifest: list[dict] = []
    for layer in bundle.layers:
        assert layer.w_q is not None
        arrays[f"{layer.name}.weight_int8"] = layer.w_q.astype(np.int8)
        arrays[f"{layer.name}.weight_scale_fp32"] = layer.w_scale.astype(np.float32)
        arrays[f"{layer.name}.bias_int32"] = layer.bias_i32.astype(np.int32)
        arrays[f"{layer.name}.requant_mult_int64"] = layer.requant.mult.astype(
            np.int64
        )
        arrays[f"{layer.name}.requant_shift_int32"] = np.asarray(
            [layer.requant.shift], dtype=np.int32
        )
        layer_manifest.append(
            {
                "name": layer.name,
                "weight_shape": list(layer.w_q.shape),
                "input_scale": layer.in_scale,
                "output_scale": layer.out_scale,
                "weight_bits": 8,
                "activation_bits": 8,
                "accumulator_bits": 32,
            }
        )
    package_path = out_dir / "encoder_int8_package.npz"
    np.savez_compressed(package_path, **arrays)
    return {
        "path": str(package_path.resolve()),
        "sha256": sha256_file(package_path),
        "layers": layer_manifest,
        "side_concat": {
            "side_dim": 4,
            "post0_input_dim": 68,
            "integer_domain_concat": True,
            "shared_scale": scales.mean_scale_i8,
        },
        "decoder_quantized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to epoch_ckpts/epoch_011.ckpt.",
    )
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ckpt = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else run_dir / "epoch_ckpts" / "epoch_011.ckpt"
    )
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    _validate_metadata(run_dir, metadata, ckpt)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(metadata["train_val_manifest"]).resolve()
    data_dir = Path(metadata["data_dir"]).resolve()
    cache_dir = (
        Path(metadata["cache_dir"]).resolve()
        if metadata.get("cache_dir")
        else None
    )
    train_files, val_files, sealed_test_names = load_frozen_train_val_manifest(
        manifest, data_dir
    )
    if len(train_files) != 27 or len(val_files) != 6 or len(sealed_test_names) != 6:
        raise ValueError("strict split must be exactly 27 train / 6 val / 6 sealed")
    behavior_mean, behavior_std = fit_behavior_stats(
        train_files, 20, cache_dir=cache_dir
    )
    side_config = load_side_feature_stats_for_run_metadata(
        metadata, train_files, cache_dir
    )
    if side_config is None or side_config[0] != "t4":
        raise ValueError("run metadata did not resolve a real T4 feature configuration")

    train_records = [
        _load_record(
            path,
            behavior_mean=behavior_mean,
            behavior_std=behavior_std,
            cache_dir=cache_dir,
            side_config=side_config,
        )
        for path in train_files
    ]
    val_records = [
        _load_record(
            path,
            behavior_mean=behavior_mean,
            behavior_std=behavior_std,
            cache_dir=cache_dir,
            side_config=side_config,
        )
        for path in val_files
    ]
    train_calibs = [record["calib_trials"] for record in train_records]
    train_sides = [record["side_features"] for record in train_records]
    train_names = [record["name"] for record in train_records]

    weights = load_b3_weights_from_ckpt(ckpt)
    if weights.post0_w.shape != (64, 68):
        raise ValueError(
            f"T4 post0 weight must be [64,68], got {weights.post0_w.shape}"
        )
    scales, candidate_rows = _select_train_only_scales(
        weights, train_calibs, train_sides, train_names
    )

    teacher = Path(metadata["teacher_checkpoint"]).resolve()
    model = load_frozen_model(ckpt, teacher, "B3S", device)
    base_encoder = model.student.id_encoder
    if not isinstance(base_encoder, SideFeatureEarlyPoolEncoder):
        raise TypeError(f"Expected SideFeatureEarlyPoolEncoder, got {type(base_encoder)}")
    if base_encoder.side_dim != 4 or base_encoder.post_pool[0].in_features != 68:
        raise ValueError("loaded checkpoint is not the required T4 B3S architecture")

    fp32_r2 = _evaluate_records(model, val_records, device=device)
    qat_scales = B3QATScales(
        input=scales.input_scale_i8,
        pre_out=scales.pre_out_scale_i8,
        mean=scales.mean_scale_i8,
        post0_out=scales.post0_out_scale_i8,
        post1_out=scales.post1_out_scale_i8,
        E=scales.E_int8_scale,
    )
    quant_encoder = wrap_early_pool_with_qat(
        base_encoder,
        qat_scales,
        num_trials=30,
        learnable_scales=False,
    ).to(device)
    quant_encoder.eval()
    model.student.id_encoder = quant_encoder
    int8_r2 = _evaluate_records(model, val_records, device=device)

    saturation: dict[str, list[float]] = {}
    overflow_count = 0
    exact_max_abs = 0.0
    shape_template = B3Shapes(T=100, D=64, W=50, N=1, M=30)
    bundle = build_quant_engine_bundle(
        weights, shape_template, scales, ABLATION_PRESETS["w8_a8_e8"]
    )
    for record in val_records:
        calib_t = torch.from_numpy(record["calib_trials"]).unsqueeze(0).to(device)
        side_t = torch.from_numpy(record["side_features"]).unsqueeze(0).to(device)
        diag = quant_encoder.compute_quant_diagnostics(
            calib_t, side_features=side_t
        )
        for edge, values in diag.items():
            if isinstance(values, dict) and "saturation_rate" in values:
                saturation.setdefault(edge, []).append(values["saturation_rate"])
        stages_t = quant_encoder.forward_integer_with_stages(
            calib_t, side_features=side_t
        )
        stages_np = forward_quant_engine(
            record["calib_trials"],
            bundle,
            side_features=record["side_features"],
        )
        exact_max_abs = max(
            exact_max_abs,
            float(
                np.max(
                    np.abs(
                        stages_t["E_dequant"].squeeze(0).cpu().numpy()
                        - stages_np["E_dequant"]
                    )
                )
            ),
        )
        overflow_count += int(
            stages_np["diagnostics"]["pre_pool"].get("acc_i32_overflow", 0)
        )
        overflow_count += sum(
            int(values.get("acc_i32_overflow", 0))
            for values in stages_np["diagnostics"]["layers"].values()
        )

    session_delta = {
        name: int8_r2[name] - fp32_r2[name] for name in sorted(fp32_r2)
    }
    mean_fp32 = float(np.mean(list(fp32_r2.values())))
    mean_int8 = float(np.mean(list(int8_r2.values())))
    mean_delta = mean_int8 - mean_fp32
    saturation_summary = {
        edge: {
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
        for edge, values in saturation.items()
    }
    max_saturation = max(
        (values["max"] for values in saturation_summary.values()), default=0.0
    )
    gates = {
        "validation_mean_delta_r2_ge_minus_0p01": (
            mean_delta >= PTQ_DELTA_R2_THRESHOLD
        ),
        "integer_matches_qat_exactly": exact_max_abs == 0.0,
        "int32_overflow_zero": overflow_count == 0,
        "max_edge_saturation_le_0p005": (
            max_saturation <= SATURATION_RATE_THRESHOLD
        ),
        "post0_is_real_68_to_64_int8": weights.post0_w.shape == (64, 68),
        "formal_test_unopened": True,
    }
    package = _export_integer_package(out_dir, weights, scales)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "T4/B3S identity encoder W8A8 + FP32 decoder",
        "decoder_quantized_in_this_run": False,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "run_metadata": str(metadata_path),
        "run_metadata_sha256": sha256_file(metadata_path),
        "teacher_sha256": metadata["teacher_sha256"],
        "train_val_manifest": str(manifest),
        "train_val_manifest_sha256": sha256_file(manifest),
        "protocol": {
            "scale_fit_sessions": train_names,
            "scale_fit_uses_behavior_labels": False,
            "validation_sessions": [session_name_from_path(path) for path in val_files],
            "sealed_formal_test_receipts": sealed_test_names,
            "formal_test_files_opened": False,
            "activity_calibration": "chronological first 30 rewarded trials",
            "t4_pool": "same chronological first 30 rewarded trials",
            "evaluation_windows": "trials[30:] only",
            "selection_mode": "first",
        },
        "scale_search": {
            "selection_source": "27 training sessions, identity RMSE only",
            "selected": candidate_rows[0]["name"],
            "candidates": candidate_rows,
            "selected_scales": asdict(scales),
            "selection_receipt_sha256": _sha256_jsonable(candidate_rows),
        },
        "r2": {
            "fp32_encoder": {
                "mean": mean_fp32,
                "per_session": fp32_r2,
            },
            "int8_encoder": {
                "mean": mean_int8,
                "per_session": int8_r2,
            },
            "delta_int8_minus_fp32": {
                "mean": mean_delta,
                "per_session": session_delta,
            },
        },
        "integer_alignment": {
            "max_abs_E": exact_max_abs,
            "int32_overflow_count": overflow_count,
        },
        "saturation": saturation_summary,
        "max_edge_saturation": max_saturation,
        "gates": gates,
        "ptq_pass": all(gates.values()),
        "next_step": "accept_encoder_ptq" if all(gates.values()) else "run_encoder_qat",
        "integer_package": package,
    }
    result_path = out_dir / "ptq_report.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fp32_mean_r2": mean_fp32,
                "int8_mean_r2": mean_int8,
                "delta_r2": mean_delta,
                "max_saturation": max_saturation,
                "overflow": overflow_count,
                "ptq_pass": payload["ptq_pass"],
                "next_step": payload["next_step"],
                "report": str(result_path),
            },
            indent=2,
        )
    )
    return 0 if payload["ptq_pass"] else 10


if __name__ == "__main__":
    raise SystemExit(main())
