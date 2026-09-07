#!/usr/bin/env python3
"""Source-fit, paired dual-view PTQ for one C1 shared-T4 checkpoint.

Only the B3S/T4 identity encoder is W8A8 with INT32 accumulation and integer
requantization.  The co-trained decoder remains byte-identical FP32.  A single
activation-scale set is fitted with equal SUA/pseudo-MUA view weight on the 27
source sessions; the six reused-development sessions are evaluated once in
both views after scale selection.  Formal sessions are never resolved.
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


ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
SWHW = ROOT / "software-to-hardware"
SCE = ROOT / "streaming_calibration_exp"
for path in (SUA, SWHW, SCE, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from b3_ckpt_loader import load_b3_weights_from_ckpt
from b3_fake_quant import B3QATScales
from b3_hw_golden import B3Shapes, forward_b3_layered
from b3_ptq import ActivationStats, calibrate_scales_from_stats
from b3_qat_encoder import wrap_early_pool_with_qat
from b3_quant_engine import (
    ABLATION_PRESETS,
    FrozenActivationScales,
    build_quant_engine_bundle,
    forward_quant_engine,
    identity_metrics,
)
from dandi688_gradient_free_protocol import select_calibration_trial_indices
from eval_adaptation_dandi688 import (
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    attach_side_features,
    build_calib_trials_for_indices,
    load_session_with_trials,
)
from mc_maze.multisession_datamodule import (
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    session_name_from_path,
)
from mc_maze.unit_side_features import (
    base_feature_group,
    fit_side_feature_stats,
    side_feature_stats_sha256,
)
from select_gradient_free_protocol_dandi688 import (
    evaluate_session_configs,
    load_frozen_model,
)
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
from write_t4_paired_view_c1_encoder_int8_prelaunch import (
    VIEWS,
    sha256_file,
    validate_receipt,
)


ACTIVITY_BUDGET = 30
T4_POOL = 50
EVALUATION_START = 50
DELTA_GATE = -0.01
SATURATION_GATE = 0.005
SCALE_CANDIDATES = (
    ("max_abs", "max_abs", 1.0),
    ("p9999_x0p8", "p9999", 0.8),
    ("p9999", "p9999", 1.0),
    ("p9999_x1p2", "p9999", 1.2),
    ("mse_opt", "mse_opt", 1.0),
)
MAX_SCALE_SAMPLES_PER_VIEW_PER_TENSOR = 262144


def _sha256_jsonable(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _rebase_repository_path(raw: str | Path) -> Path:
    """Map an isolated-stage absolute path onto this repository when needed."""
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()
    parts = path.parts
    if "sua_exploration" in parts:
        index = parts.index("sua_exploration")
        candidate = ROOT.joinpath(*parts[index:])
        return candidate.resolve()
    return path.resolve()


def _metadata_checks(metadata: dict[str, Any], seed: int) -> None:
    training = metadata.get("training") or {}
    checks = {
        "completed": metadata.get("status") == "completed",
        "seed": metadata.get("seed") == seed,
        "variant": metadata.get("variant") == "B3S",
        "paired": metadata.get("training_kind") == "shared_paired_view",
        "paired_signal": metadata.get("signal_view") == "paired_sua_pseudo_mua",
        "formal_sealed": metadata.get("held_out_test_evaluated") is False,
        "formal_sua_sealed": metadata.get("formal_sua_files_opened") is False,
        "q30": training.get("evaluation_forward_calibration_n") == ACTIVITY_BUDGET,
        "pool50": training.get("evaluation_pool_size") == T4_POOL,
        "start50": training.get("evaluation_start_trial") == EVALUATION_START,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"C1 INT8 metadata failed: {failed}")


def _view_feature_config(
    metadata: dict[str, Any], view: str, train_files: list[Path]
) -> tuple[str, str, int, int | None, np.ndarray, np.ndarray, Path, str]:
    views = metadata.get("view_configs") or {}
    config = views.get(view) or {}
    side = config.get("side_features") or {}
    if config.get("signal_view") != view:
        raise ValueError(f"C1 metadata view mismatch: {view}")
    if side.get("group") != "t4" or side.get("pool_size") != 50 or side.get("side_dim") != 4:
        raise ValueError(f"C1 {view} is not a four-value T4@50 view")
    cache_dir = _rebase_repository_path(str(config.get("cache_dir", "")))
    group = "t4"
    waveform_group = base_feature_group(group)
    mean, std = fit_side_feature_stats(
        train_files,
        feature_group=waveform_group,
        pool_size=50,
        cache_dir=cache_dir,
        bin_size_ms=20,
        window_size=WINDOW_SIZE,
        trial_result_filter="R",
        signal_view=view,
    )
    observed = side_feature_stats_sha256(mean, std)
    if observed != side.get("normalization_sha256"):
        raise ValueError(f"C1 {view} source-only T4 normalizer hash drift")
    return (
        group,
        waveform_group,
        50,
        side.get("permutation_seed"),
        mean,
        std,
        cache_dir,
        observed,
    )


def _load_record(
    path: Path,
    *,
    view: str,
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    feature_config: tuple[str, str, int, int | None, np.ndarray, np.ndarray, Path, str],
) -> dict[str, Any]:
    group, waveform, pool, permutation_seed, side_mean, side_std, cache_dir, _ = feature_config
    record = load_session_with_trials(
        path,
        20,
        WINDOW_SIZE,
        T4_POOL,
        TRIAL_LENGTH,
        PAD_VALUE,
        behavior_mean,
        behavior_std,
        cache_dir=cache_dir,
        signal_view=view,
    )
    record = attach_side_features(
        record,
        path,
        side_feature_group=group,
        waveform_feature_group=waveform,
        pool_size=pool,
        permutation_seed=permutation_seed,
        mean=side_mean,
        std=side_std,
        cache_dir=cache_dir,
    )
    indices = select_calibration_trial_indices(
        record["trials"], ACTIVITY_BUDGET, T4_POOL, "first"
    )
    record["calib_trials"] = build_calib_trials_for_indices(
        record, indices, ACTIVITY_BUDGET
    )
    expected_calib = (ACTIVITY_BUDGET, TRIAL_LENGTH, record["n_units"])
    if record["calib_trials"].shape != expected_calib:
        raise ValueError(f"{record['name']} {view} calib shape drift")
    if record["side_features"].shape != (record["n_units"], 4):
        raise ValueError(f"{record['name']} {view} side-feature shape drift")
    return record


def select_dual_view_source_scales(
    weights,
    calibs: list[np.ndarray],
    sides: list[np.ndarray],
    record_views: list[str],
    record_names: list[str],
) -> tuple[FrozenActivationScales, list[dict[str, Any]]]:
    if not (len(calibs) == len(sides) == len(record_views) == len(record_names) == 54):
        raise ValueError("scale fitting requires exactly 27 source sessions x two views")
    if record_views.count("sua") != 27 or record_views.count("pseudo_mua") != 27:
        raise ValueError("scale fitting views are not equally weighted")
    stats = collect_equal_view_activation_stats(
        weights, calibs, sides, record_views
    )
    rows: list[dict[str, Any]] = []
    shape_template = B3Shapes(T=100, D=64, W=50, N=1, M=30)
    for name, method, multiplier in SCALE_CANDIDATES:
        scales = calibrate_scales_from_stats(
            stats,
            method,
            mult=multiplier,
            source_sessions=record_names,
        )
        bundle = build_quant_engine_bundle(
            weights,
            shape_template,
            scales,
            ABLATION_PRESETS["w8_a8_e8"],
        )
        metrics: list[dict[str, float]] = []
        for calib, side in zip(calibs, sides):
            reference = forward_b3_layered(calib, weights, side_features=side)["E"]
            prediction = forward_quant_engine(
                calib, bundle, side_features=side
            )["E_dequant"]
            metrics.append(identity_metrics(reference, prediction))
        view_rmse = {
            view: float(
                np.mean(
                    [
                        metric["rmse"]
                        for metric, metric_view in zip(metrics, record_views)
                        if metric_view == view
                    ]
                )
            )
            for view in VIEWS
        }
        rows.append(
            {
                "name": name,
                "method": method,
                "multiplier": multiplier,
                "train_identity_rmse_equal_view_mean": float(np.mean(list(view_rmse.values()))),
                "train_identity_rmse_by_view": view_rmse,
                "train_identity_cosine_record_mean": float(
                    np.mean([metric["cosine"] for metric in metrics])
                ),
                "scales": asdict(scales),
            }
        )
    rows.sort(key=lambda row: row["train_identity_rmse_equal_view_mean"])
    return FrozenActivationScales(**rows[0]["scales"]), rows


def _activation_stats_for_records(
    weights,
    calibs: list[np.ndarray],
    sides: list[np.ndarray],
) -> ActivationStats:
    chunks: dict[str, list[np.ndarray]] = {
        "input": [],
        "pre_relu": [],
        "mean": [],
        "post0_relu": [],
        "post1_relu": [],
        "E": [],
    }
    for calib, side in zip(calibs, sides):
        stages = forward_b3_layered(calib, weights, side_features=side)
        chunks["input"].append(np.abs(calib).astype(np.float64).ravel())
        chunks["pre_relu"].append(
            np.maximum(stages["feat"], 0).astype(np.float64).ravel()
        )
        # post0 consumes non-negative pooled activity plus signed normalized T4.
        # Preserve both signs here; the generic plain-B3 helper historically
        # clips the whole concatenated tensor at zero and would miss a large
        # negative T4 tail.
        chunks["mean"].append(
            np.concatenate(
                [
                    stages["pooled_mean_feat"].astype(np.float64).ravel(),
                    np.asarray(side, dtype=np.float64).ravel(),
                ]
            )
        )
        chunks["post0_relu"].append(
            np.maximum(stages["post0_relu"], 0).astype(np.float64).ravel()
        )
        chunks["post1_relu"].append(
            np.maximum(stages["post1_relu"], 0).astype(np.float64).ravel()
        )
        chunks["E"].append(np.abs(stages["E"]).astype(np.float64).ravel())
    return ActivationStats(
        input_abs=np.concatenate(chunks["input"]),
        pre_relu=np.concatenate(chunks["pre_relu"]),
        mean=np.concatenate(chunks["mean"]),
        post0_relu=np.concatenate(chunks["post0_relu"]),
        post1_relu=np.concatenate(chunks["post1_relu"]),
        E_abs=np.concatenate(chunks["E"]),
    )


def _deterministic_equal_mass_pair(
    left: np.ndarray,
    right: np.ndarray,
    *,
    cap: int = MAX_SCALE_SAMPLES_PER_VIEW_PER_TENSOR,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size == 0 or right.size == 0:
        raise ValueError("dual-view activation statistic is empty")
    count = min(cap, left.size, right.size)

    def sample(values: np.ndarray) -> np.ndarray:
        if count == values.size:
            selected = values.copy()
        elif count == 1:
            selected = values[[0]].copy()
        else:
            indices = np.linspace(0, values.size - 1, num=count, dtype=np.int64)
            selected = values[indices].copy()
        # Retain the absolute extremum so max-abs remains conservative even
        # though percentile/MSE candidates use the bounded equal-mass sample.
        selected[0] = values[int(np.argmax(np.abs(values)))]
        return selected

    return np.concatenate([sample(left), sample(right)])


def collect_equal_view_activation_stats(
    weights,
    calibs: list[np.ndarray],
    sides: list[np.ndarray],
    record_views: list[str],
) -> ActivationStats:
    by_view: dict[str, ActivationStats] = {}
    for view in VIEWS:
        indices = [index for index, value in enumerate(record_views) if value == view]
        if len(indices) != 27:
            raise ValueError("equal-view activation stats require 27 records per view")
        by_view[view] = _activation_stats_for_records(
            weights,
            [calibs[index] for index in indices],
            [sides[index] for index in indices],
        )

    def pair(attribute: str) -> np.ndarray:
        return _deterministic_equal_mass_pair(
            getattr(by_view["sua"], attribute),
            getattr(by_view["pseudo_mua"], attribute),
        )

    return ActivationStats(
        input_abs=pair("input_abs"),
        pre_relu=pair("pre_relu"),
        mean=pair("mean"),
        post0_relu=pair("post0_relu"),
        post1_relu=pair("post1_relu"),
        E_abs=pair("E_abs"),
    )


@torch.no_grad()
def _evaluate_view(model, records: list[dict[str, Any]], device: torch.device) -> dict[str, float]:
    values: dict[str, float] = {}
    for record in records:
        session_result, _ = evaluate_session_configs(
            record,
            [("first", ACTIVITY_BUDGET)],
            EVALUATION_START,
            model,
            device,
        )
        values[record["name"]] = float(
            session_result["gradient_free_calibrated_first_n30"]
        )
    return values


def _integer_diagnostics(
    quant_encoder,
    weights,
    scales: FrozenActivationScales,
    records_by_view: dict[str, list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    bundle = build_quant_engine_bundle(
        weights,
        B3Shapes(T=100, D=64, W=50, N=1, M=30),
        scales,
        ABLATION_PRESETS["w8_a8_e8"],
    )
    by_view: dict[str, Any] = {}
    global_exact = 0.0
    global_overflow = 0
    global_max_saturation = 0.0
    for view in VIEWS:
        saturation: dict[str, list[float]] = {}
        exact_max = 0.0
        overflow = 0
        for record in records_by_view[view]:
            calib_t = torch.from_numpy(record["calib_trials"]).unsqueeze(0).to(device)
            side_t = torch.from_numpy(record["side_features"]).unsqueeze(0).to(device)
            diagnostics = quant_encoder.compute_quant_diagnostics(
                calib_t, side_features=side_t
            )
            for edge, values in diagnostics.items():
                if isinstance(values, dict) and "saturation_rate" in values:
                    saturation.setdefault(edge, []).append(float(values["saturation_rate"]))
            stages_t = quant_encoder.forward_integer_with_stages(
                calib_t, side_features=side_t
            )
            stages_np = forward_quant_engine(
                record["calib_trials"], bundle, side_features=record["side_features"]
            )
            exact_max = max(
                exact_max,
                float(
                    np.max(
                        np.abs(
                            stages_t["E_dequant"].squeeze(0).cpu().numpy()
                            - stages_np["E_dequant"]
                        )
                    )
                ),
            )
            overflow += int(
                stages_np["diagnostics"]["pre_pool"].get("acc_i32_overflow", 0)
            )
            overflow += sum(
                int(values.get("acc_i32_overflow", 0))
                for values in stages_np["diagnostics"]["layers"].values()
            )
        saturation_summary = {
            edge: {"max": float(np.max(values)), "mean": float(np.mean(values))}
            for edge, values in saturation.items()
        }
        max_saturation = max(
            (row["max"] for row in saturation_summary.values()), default=0.0
        )
        by_view[view] = {
            "integer_fake_quant_max_abs_E": exact_max,
            "int32_overflow_count": overflow,
            "saturation": saturation_summary,
            "max_edge_saturation": max_saturation,
        }
        global_exact = max(global_exact, exact_max)
        global_overflow += overflow
        global_max_saturation = max(global_max_saturation, max_saturation)
    return {
        "by_view": by_view,
        "integer_fake_quant_max_abs_E": global_exact,
        "int32_overflow_count": global_overflow,
        "max_edge_saturation": global_max_saturation,
    }


def _export_integer_package(
    out_dir: Path, weights, scales: FrozenActivationScales
) -> dict[str, Any]:
    bundle = build_quant_engine_bundle(
        weights,
        B3Shapes(T=100, D=64, W=50, N=1, M=30),
        scales,
        ABLATION_PRESETS["w8_a8_e8"],
    )
    arrays: dict[str, np.ndarray] = {}
    layers: list[dict[str, Any]] = []
    for layer in bundle.layers:
        if layer.w_q is None:
            raise ValueError(f"{layer.name} lacks integer weights")
        arrays[f"{layer.name}.weight_int8"] = layer.w_q.astype(np.int8)
        arrays[f"{layer.name}.weight_scale_fp32"] = layer.w_scale.astype(np.float32)
        arrays[f"{layer.name}.bias_int32"] = layer.bias_i32.astype(np.int32)
        arrays[f"{layer.name}.requant_mult_int64"] = layer.requant.mult.astype(np.int64)
        arrays[f"{layer.name}.requant_shift_int32"] = np.asarray(
            [layer.requant.shift], dtype=np.int32
        )
        layers.append(
            {
                "name": layer.name,
                "weight_shape": list(layer.w_q.shape),
                "weight_bits": 8,
                "activation_bits": 8,
                "accumulator_bits": 32,
                "integer_requant": True,
                "input_scale": layer.in_scale,
                "output_scale": layer.out_scale,
            }
        )
    path = out_dir / "encoder_int8_package.npz"
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "layers": layers,
        "one_package_serves_both_views": True,
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
    parser.add_argument("--prelaunch", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    metadata_path = args.run_metadata.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    seed = int(metadata["seed"])
    receipt = validate_receipt(
        args.prelaunch,
        seed=seed,
        checkpoint=checkpoint,
        run_metadata=metadata_path,
    )
    _metadata_checks(metadata, seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable PTQ directory: {out_dir}")
    out_dir.mkdir(parents=True)

    manifest = ROOT / receipt["strict_manifest"]["path"]
    data_dir = ROOT / receipt["data_dir"]["path"]
    train_files, val_files, sealed_test_names = load_frozen_train_val_manifest(
        manifest, data_dir
    )
    if (len(train_files), len(val_files), len(sealed_test_names)) != (27, 6, 6):
        raise ValueError("C1 INT8 strict split is not 27/6/6")
    if sealed_test_names != receipt["strict_manifest"]["sealed_formal_session_names"]:
        raise ValueError("formal name receipt drifted")

    feature_configs = {
        view: _view_feature_config(metadata, view, train_files) for view in VIEWS
    }
    behavior_mean, behavior_std = fit_behavior_stats(
        train_files, 20, cache_dir=feature_configs["sua"][6]
    )
    train_calibs: list[np.ndarray] = []
    train_sides: list[np.ndarray] = []
    train_views: list[str] = []
    train_record_names: list[str] = []
    for view in VIEWS:
        for path in train_files:
            record = _load_record(
                path,
                view=view,
                behavior_mean=behavior_mean,
                behavior_std=behavior_std,
                feature_config=feature_configs[view],
            )
            train_calibs.append(record["calib_trials"])
            train_sides.append(record["side_features"])
            train_views.append(view)
            train_record_names.append(f"{record['name']}::{view}")

    weights = load_b3_weights_from_ckpt(checkpoint)
    if weights.post0_w.shape != (64, 68):
        raise ValueError(f"C1 T4 post0 must be [64,68], got {weights.post0_w.shape}")
    scales, candidates = select_dual_view_source_scales(
        weights, train_calibs, train_sides, train_views, train_record_names
    )

    records_by_view = {
        view: [
            _load_record(
                path,
                view=view,
                behavior_mean=behavior_mean,
                behavior_std=behavior_std,
                feature_config=feature_configs[view],
            )
            for path in val_files
        ]
        for view in VIEWS
    }
    teacher = ROOT / receipt["teacher"]["path"]
    model = load_frozen_model(checkpoint, teacher, "B3S", device)
    encoder = model.student.id_encoder
    if not isinstance(encoder, SideFeatureEarlyPoolEncoder):
        raise TypeError(f"expected SideFeatureEarlyPoolEncoder, got {type(encoder)}")
    if encoder.side_dim != 4 or tuple(encoder.post_pool[0].weight.shape) != (64, 68):
        raise ValueError("loaded checkpoint is not shared C1 B3S/T4")
    decoder_before = _module_state_sha256(model.student.decoder)
    fp32 = {
        view: _evaluate_view(model, records_by_view[view], device) for view in VIEWS
    }

    qat_scales = B3QATScales(
        input=scales.input_scale_i8,
        pre_out=scales.pre_out_scale_i8,
        mean=scales.mean_scale_i8,
        post0_out=scales.post0_out_scale_i8,
        post1_out=scales.post1_out_scale_i8,
        E=scales.E_int8_scale,
    )
    quant_encoder = wrap_early_pool_with_qat(
        encoder, qat_scales, num_trials=ACTIVITY_BUDGET, learnable_scales=False
    ).to(device)
    quant_encoder.eval()
    model.student.id_encoder = quant_encoder
    int8 = {
        view: _evaluate_view(model, records_by_view[view], device) for view in VIEWS
    }
    decoder_after = _module_state_sha256(model.student.decoder)
    diagnostics = _integer_diagnostics(
        quant_encoder, weights, scales, records_by_view, device
    )

    view_results: dict[str, Any] = {}
    for view in VIEWS:
        sessions = sorted(fp32[view])
        delta = {name: int8[view][name] - fp32[view][name] for name in sessions}
        view_results[view] = {
            "fp32": {
                "mean_r2": float(np.mean(list(fp32[view].values()))),
                "per_session_r2": fp32[view],
            },
            "int8": {
                "mean_r2": float(np.mean(list(int8[view].values()))),
                "per_session_r2": int8[view],
            },
            "delta_int8_minus_fp32": {
                "mean_r2": float(np.mean(list(delta.values()))),
                "median_r2": float(np.median(list(delta.values()))),
                "min_r2": float(np.min(list(delta.values()))),
                "positive_session_count": int(sum(value > 0 for value in delta.values())),
                "per_session_r2": delta,
            },
        }

    gates = {
        "sua_seed_mean_delta_ge_minus_0p01": (
            view_results["sua"]["delta_int8_minus_fp32"]["mean_r2"] >= DELTA_GATE
        ),
        "pseudo_mua_seed_mean_delta_ge_minus_0p01": (
            view_results["pseudo_mua"]["delta_int8_minus_fp32"]["mean_r2"] >= DELTA_GATE
        ),
        "integer_fake_quant_exact": diagnostics["integer_fake_quant_max_abs_E"] == 0.0,
        "int32_overflow_zero": diagnostics["int32_overflow_count"] == 0,
        "max_edge_saturation_le_0p005": diagnostics["max_edge_saturation"] <= SATURATION_GATE,
        "decoder_state_byte_identical": decoder_before == decoder_after,
        "one_shared_scale_set_for_both_views": True,
        "formal_test_unopened": True,
    }
    package = _export_integer_package(out_dir, weights, scales)
    prelaunch_path = args.prelaunch.expanduser().resolve()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "ptq",
        "scope": "C1 shared B3S/T4 identity encoder W8A8 INT32 + FP32 decoder",
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_rule": "fixed_final_training_epoch_epoch_011_no_dev_argmax",
        "run_metadata": str(metadata_path),
        "run_metadata_sha256": sha256_file(metadata_path),
        "prelaunch": str(prelaunch_path),
        "prelaunch_sha256": sha256_file(prelaunch_path),
        "decoder_quantized_in_this_run": False,
        "decoder_precision": "FP32",
        "decoder_state_sha256_before": decoder_before,
        "decoder_state_sha256_after": decoder_after,
        "protocol": {
            "source_sessions": [session_name_from_path(path) for path in train_files],
            "development_sessions": [session_name_from_path(path) for path in val_files],
            "sealed_formal_session_names": sealed_test_names,
            "formal_test_files_opened": False,
            "views": list(VIEWS),
            "scale_fit_records": 54,
            "scale_fit_unique_source_sessions": 27,
            "scale_fit_records_by_view": {"sua": 27, "pseudo_mua": 27},
            "scale_fit_view_weight": {"sua": 0.5, "pseudo_mua": 0.5},
            "scale_statistic_equal_mass_per_view": True,
            "scale_statistic_max_samples_per_view_per_tensor": MAX_SCALE_SAMPLES_PER_VIEW_PER_TENSOR,
            "scale_statistic_sampling": "deterministic_uniform_stride_with_absolute_extremum_retained",
            "signed_t4_included_in_post0_input_scale_fit": True,
            "one_shared_scale_set_per_seed": True,
            "scale_selection_uses_behavior_targets": False,
            "validation_used_for_scale_selection": False,
            "validation_used_once_after_scale_selection": True,
            "activity_calibration_n": 30,
            "t4_label_rate_pool_n": 50,
            "evaluation_start_trial": 50,
            "selection_mode": "chronological_first",
            "normalizer_sha256_by_view": {
                view: feature_configs[view][7] for view in VIEWS
            },
        },
        "scale_search": {
            "objective": "mean_identity_rmse_equal_weight_over_27x2_source_view_records",
            "selected": candidates[0]["name"],
            "selected_scales": asdict(scales),
            "candidates": candidates,
            "receipt_sha256": _sha256_jsonable(candidates),
        },
        "view_results": view_results,
        "integer_diagnostics": diagnostics,
        "gates": gates,
        "ptq_pass": all(gates.values()),
        "next_step": "accept_encoder_ptq" if all(gates.values()) else "await_three_seed_ptq_then_qat_all_three",
        "integer_package": package,
        "formal_test_files_opened": False,
    }
    report = out_dir / "ptq_report.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": seed,
                "sua_delta": view_results["sua"]["delta_int8_minus_fp32"]["mean_r2"],
                "pseudo_mua_delta": view_results["pseudo_mua"]["delta_int8_minus_fp32"]["mean_r2"],
                "max_saturation": diagnostics["max_edge_saturation"],
                "overflow": diagnostics["int32_overflow_count"],
                "ptq_pass": payload["ptq_pass"],
                "report": str(report),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["ptq_pass"] else 10


if __name__ == "__main__":
    raise SystemExit(main())
