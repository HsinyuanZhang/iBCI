#!/usr/bin/env python3
"""Uniform source-only paired-view QAT for one C1 shared-T4 seed."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "sua_exploration",
    ROOT / "sua_exploration/scripts",
    ROOT / "software-to-hardware",
    ROOT / "streaming_calibration_exp",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from b3_fake_quant import B3QATScales
from b3_hw_golden import B3Shapes, B3Weights
from b3_qat_encoder import wrap_early_pool_with_qat
from b3_quant_engine import ABLATION_PRESETS, build_quant_engine_bundle, forward_quant_engine
from eval_t4_paired_view_c1_encoder_int8 import (
    EVALUATION_START,
    SATURATION_GATE,
    T4_POOL,
    VIEWS,
    _evaluate_view,
    _export_integer_package,
    _load_record,
    _module_state_sha256,
    _rebase_repository_path,
    _view_feature_config,
)
from mc_maze.multisession_datamodule import (
    Dandi688MultiSessionDataModule,
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    session_name_from_path,
)
from mc_maze.paired_view_c1 import PairedViewC1DataModule, validate_pair_batch
from mc_maze.unit_side_features import side_feature_stats_sha256
from select_gradient_free_protocol_dandi688 import load_frozen_model
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
from write_t4_paired_view_c1_encoder_qat_prelaunch import (
    SEEDS,
    sha256_file,
    validate_receipt,
)


QAT_EPOCHS = 8
BATCH_SIZE = 32
LR_WEIGHT = 1e-5
LR_SCALE = 1e-5
GRADIENT_CLIP = 1.0
SIDE_SCALE_FLOOR_MARGIN = 1.10
DELTA_GATE = -0.01


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _unpack(batch: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor]:
    if len(batch) != 5:
        raise ValueError("C1 QAT requires five-item T4 batches")
    neural, behavior, calib, sessions, side = batch
    return neural, behavior, calib, sessions, side


def _export_weights(encoder) -> B3Weights:
    def array(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32)

    return B3Weights(
        pre_w=array(encoder.pre_linear.weight),
        pre_b=array(encoder.pre_linear.bias),
        post0_w=array(encoder.post_linears[0].weight),
        post0_b=array(encoder.post_linears[0].bias),
        post1_w=array(encoder.post_linears[1].weight),
        post1_b=array(encoder.post_linears[1].bias),
        post2_w=array(encoder.post_linears[2].weight),
        post2_b=array(encoder.post_linears[2].bias),
    )


def source_side_scale_floor(paired_dm: PairedViewC1DataModule) -> tuple[float, dict[str, float]]:
    if paired_dm.train_dataset is None:
        raise RuntimeError("paired source dataset is not initialized")
    maxima: dict[str, float] = {}
    for view, dataset in (
        ("sua", paired_dm.train_dataset.sua_dataset),
        ("pseudo_mua", paired_dm.train_dataset.pseudo_mua_dataset),
    ):
        values = [
            float(np.max(np.abs(record.side_features)))
            for record in dataset.sessions.values()
            if record.side_features is not None
        ]
        if len(values) != 27:
            raise ValueError(f"{view} source-side floor did not see 27 sessions")
        maxima[view] = max(values)
    floor = SIDE_SCALE_FLOOR_MARGIN * max(maxima.values()) / 127.0
    return float(floor), maxima


def _project_mean_scale_floor(quant_encoder, floor: float) -> None:
    scale = quant_encoder.shared_scales.s_mean
    if not isinstance(scale.log_scale, torch.nn.Parameter):
        raise TypeError("C1 QAT mean scale is not learnable")
    floor_log = math.log(max(float(floor), 1e-8))
    if floor_log > float(scale.log_max):
        raise ValueError("source-side scale floor exceeds frozen 4x QAT scale bound")
    with torch.no_grad():
        scale.log_scale.clamp_(min=floor_log)


def _set_encoder_only_trainable(model, quant_encoder) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in quant_encoder.parameters():
        parameter.requires_grad = True
    model.student.freeze_decoder()
    model.student.decoder.eval()
    quant_encoder.train()


def _view_loss(
    *,
    model,
    quant_encoder,
    anchor_encoder,
    batch: Sequence[Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    neural, behavior, calib, _sessions, side = _unpack(batch)
    neural = neural.to(device)
    behavior = behavior.to(device)
    calib = calib.to(device)
    side = side.to(device)
    e_quant = quant_encoder.forward_batch(calib, side_features=side)
    y_quant = model.student.decode_with_identity(neural, e_quant)[:, -1:, :] / 5.0
    target = behavior[:, -1:, :]
    with torch.no_grad():
        e_anchor = anchor_encoder.forward_batch(calib, side_features=side)
        y_anchor = model.student.decode_with_identity(neural, e_anchor)[:, -1:, :] / 5.0
    task = F.mse_loss(y_quant, target)
    anchor_y = F.mse_loss(y_quant, y_anchor)
    anchor_e = F.mse_loss(e_quant, e_anchor) / e_anchor.pow(2).mean().clamp_min(1e-8)
    mean_scale = quant_encoder.shared_scales.s_mean.value()
    range_excess = torch.relu(side.abs() / (127.0 * mean_scale) - 1.0)
    side_range = range_excess.square().mean()
    total = task + 0.75 * anchor_y + 0.075 * anchor_e + 0.01 * side_range
    return total, {
        "task": float(task.detach()),
        "anchor_y": float(anchor_y.detach()),
        "anchor_E": float(anchor_e.detach()),
        "source_side_range": float(side_range.detach()),
        "total": float(total.detach()),
    }


def train_epoch(
    *,
    model,
    quant_encoder,
    anchor_encoder,
    loader,
    optimizer,
    device: torch.device,
    side_scale_floor: float,
) -> dict[str, float]:
    quant_encoder.train()
    model.student.decoder.eval()
    totals = {
        f"{view}_{key}": 0.0
        for view in VIEWS
        for key in ("task", "anchor_y", "anchor_E", "source_side_range", "total")
    }
    batches = 0
    for pair in loader:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("C1 QAT loader did not return a paired batch")
        sua_batch, pseudo_batch = pair
        validate_pair_batch(sua_batch, pseudo_batch)
        optimizer.zero_grad(set_to_none=True)
        for view, batch in (("sua", sua_batch), ("pseudo_mua", pseudo_batch)):
            loss, metrics = _view_loss(
                model=model,
                quant_encoder=quant_encoder,
                anchor_encoder=anchor_encoder,
                batch=batch,
                device=device,
            )
            (0.5 * loss).backward()
            for key, value in metrics.items():
                totals[f"{view}_{key}"] += value
        torch.nn.utils.clip_grad_norm_(quant_encoder.parameters(), GRADIENT_CLIP)
        optimizer.step()
        _project_mean_scale_floor(quant_encoder, side_scale_floor)
        batches += 1
    if batches == 0:
        raise RuntimeError("C1 QAT source loader produced no batches")
    return {key: value / batches for key, value in totals.items()}


@torch.no_grad()
def quant_diagnostics(
    quant_encoder,
    weights: B3Weights,
    scales,
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
    global_code_mismatch = 0
    global_code_max_abs = 0
    global_dequant_max_abs = 0.0
    global_overflow = 0
    global_saturation = 0.0
    for view in VIEWS:
        saturation: dict[str, list[float]] = {}
        code_mismatch = 0
        code_max_abs = 0
        dequant_max_abs = 0.0
        overflow = 0
        for record in records_by_view[view]:
            calib_t = torch.from_numpy(record["calib_trials"]).unsqueeze(0).to(device)
            side_t = torch.from_numpy(record["side_features"]).unsqueeze(0).to(device)
            diag = quant_encoder.compute_quant_diagnostics(calib_t, side_features=side_t)
            for edge, values in diag.items():
                if isinstance(values, dict) and "saturation_rate" in values:
                    saturation.setdefault(edge, []).append(float(values["saturation_rate"]))
            torch_stages = quant_encoder.forward_integer_with_stages(
                calib_t, side_features=side_t
            )
            numpy_stages = forward_quant_engine(
                record["calib_trials"], bundle, side_features=record["side_features"]
            )
            torch_code = np.rint(
                torch_stages["E_q"].squeeze(0).cpu().numpy()
            ).astype(np.int16)
            numpy_code = np.asarray(numpy_stages["E"], dtype=np.int16)
            code_delta = torch_code.astype(np.int32) - numpy_code.astype(np.int32)
            code_mismatch += int(np.count_nonzero(code_delta))
            code_max_abs = max(code_max_abs, int(np.max(np.abs(code_delta))))
            dequant_max_abs = max(
                dequant_max_abs,
                float(
                    np.max(
                        np.abs(
                            torch_stages["E_dequant"].squeeze(0).cpu().numpy()
                            - numpy_stages["E_dequant"]
                        )
                    )
                ),
            )
            overflow += int(
                numpy_stages["diagnostics"]["pre_pool"].get("acc_i32_overflow", 0)
            )
            overflow += sum(
                int(values.get("acc_i32_overflow", 0))
                for values in numpy_stages["diagnostics"]["layers"].values()
            )
        saturation_summary = {
            edge: {"max": float(np.max(values)), "mean": float(np.mean(values))}
            for edge, values in saturation.items()
        }
        max_saturation = max(
            (values["max"] for values in saturation_summary.values()), default=0.0
        )
        by_view[view] = {
            "integer_E_q_code_mismatch_count": code_mismatch,
            "integer_E_q_code_max_abs_difference": code_max_abs,
            "E_dequant_torch_minus_numpy_max_abs": dequant_max_abs,
            "int32_overflow_count": overflow,
            "max_edge_saturation": max_saturation,
            "saturation": saturation_summary,
        }
        global_code_mismatch += code_mismatch
        global_code_max_abs = max(global_code_max_abs, code_max_abs)
        global_dequant_max_abs = max(global_dequant_max_abs, dequant_max_abs)
        global_overflow += overflow
        global_saturation = max(global_saturation, max_saturation)
    return {
        "by_view": by_view,
        "integer_E_q_code_mismatch_count": global_code_mismatch,
        "integer_E_q_code_max_abs_difference": global_code_max_abs,
        "E_dequant_torch_minus_numpy_max_abs": global_dequant_max_abs,
        "int32_overflow_count": global_overflow,
        "max_edge_saturation": global_saturation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qat-prelaunch", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=list(SEEDS), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=QAT_EPOCHS)
    args = parser.parse_args()
    if args.epochs != QAT_EPOCHS:
        raise ValueError("C1 QAT epoch budget is frozen at 8")
    receipt_path = args.qat_prelaunch.expanduser().resolve()
    receipt = validate_receipt(receipt_path, seed=args.seed)
    bound = receipt["seed_inputs"][str(args.seed)]
    ptq_path = ROOT / bound["ptq_report"]["path"]
    ptq = _read_json(ptq_path)
    metadata_path = Path(bound["run_metadata"]["path"]).expanduser().resolve()
    checkpoint = Path(bound["checkpoint"]["path"]).expanduser().resolve()
    metadata = _read_json(metadata_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to reuse QAT output: {out_dir}")
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    ptq_prelaunch_path = ROOT / receipt["ptq_prelaunch"]["path"]
    ptq_prelaunch = _read_json(ptq_prelaunch_path)
    manifest = ROOT / ptq_prelaunch["strict_manifest"]["path"]
    data_dir = ROOT / ptq_prelaunch["data_dir"]["path"]
    teacher = ROOT / ptq_prelaunch["teacher"]["path"]
    train_files, val_files, sealed_names = load_frozen_train_val_manifest(manifest, data_dir)
    if (len(train_files), len(val_files), len(sealed_names)) != (27, 6, 6):
        raise ValueError("C1 QAT strict split is not 27/6/6")

    view_configs = {
        view: _view_feature_config(metadata, view, train_files) for view in VIEWS
    }
    behavior_mean, behavior_std = fit_behavior_stats(
        train_files, 20, cache_dir=view_configs["sua"][6]
    )
    val_records = {
        view: [
            _load_record(
                path,
                view=view,
                behavior_mean=behavior_mean,
                behavior_std=behavior_std,
                feature_config=view_configs[view],
            )
            for path in val_files
        ]
        for view in VIEWS
    }

    common_dm = dict(
        data_dir=str(data_dir),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=BATCH_SIZE,
        window_size=50,
        calibration_n_trials=30,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=4,
        random_calibration=False,
        seed=args.seed,
        max_units_exclusive=100,
        side_feature_group="t4",
        side_feature_pool_size=50,
        side_permutation_seed=None,
        train_val_manifest_path=str(manifest),
    )
    sua_dm = Dandi688MultiSessionDataModule(
        **common_dm, cache_dir=str(view_configs["sua"][6]), signal_view="sua"
    )
    pseudo_dm = Dandi688MultiSessionDataModule(
        **common_dm,
        cache_dir=str(view_configs["pseudo_mua"][6]),
        signal_view="pseudo_mua",
    )
    paired_dm = PairedViewC1DataModule(sua_dm, pseudo_dm)
    paired_dm.setup()
    if sua_dm.session_files["test"] or pseudo_dm.session_files["test"]:
        raise RuntimeError("C1 QAT resolved formal test paths")
    if paired_dm.session_splits["test"] != sealed_names:
        raise RuntimeError("C1 QAT formal name receipt drift")
    sua_stats = sua_dm._get_side_feature_stats()
    pseudo_stats = pseudo_dm._get_side_feature_stats()
    if side_feature_stats_sha256(*sua_stats) != view_configs["sua"][7]:
        raise ValueError("C1 QAT SUA normalizer drift")
    if side_feature_stats_sha256(*pseudo_stats) != view_configs["pseudo_mua"][7]:
        raise ValueError("C1 QAT pseudo-MUA normalizer drift")

    model = load_frozen_model(checkpoint, teacher, "B3S", device)
    base_encoder = model.student.id_encoder
    if not isinstance(base_encoder, SideFeatureEarlyPoolEncoder):
        raise TypeError(base_encoder)
    anchor_encoder = copy.deepcopy(base_encoder).to(device).eval()
    for parameter in anchor_encoder.parameters():
        parameter.requires_grad = False
    selected_scales = ptq["scale_search"]["selected_scales"]
    qat_scales = B3QATScales(
        input=float(selected_scales["input_scale_i8"]),
        pre_out=float(selected_scales["pre_out_scale_i8"]),
        mean=float(selected_scales["mean_scale_i8"]),
        post0_out=float(selected_scales["post0_out_scale_i8"]),
        post1_out=float(selected_scales["post1_out_scale_i8"]),
        E=float(selected_scales["E_int8_scale"]),
    )
    quant_encoder = wrap_early_pool_with_qat(
        base_encoder, qat_scales, num_trials=30, learnable_scales=True
    ).to(device)
    model.student.id_encoder = quant_encoder
    _set_encoder_only_trainable(model, quant_encoder)
    decoder_before = _module_state_sha256(model.student.decoder)
    side_floor, side_maxima = source_side_scale_floor(paired_dm)
    _project_mean_scale_floor(quant_encoder, side_floor)

    weight_parameters = [
        parameter
        for name, parameter in quant_encoder.named_parameters()
        if "shared_scales" not in name
    ]
    scale_parameters = quant_encoder.shared_scales.scale_parameters()
    optimizer = torch.optim.Adam(
        [
            {"params": weight_parameters, "lr": LR_WEIGHT},
            {"params": scale_parameters, "lr": LR_SCALE},
        ],
        weight_decay=0.0,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, QAT_EPOCHS + 1):
        metrics = train_epoch(
            model=model,
            quant_encoder=quant_encoder,
            anchor_encoder=anchor_encoder,
            loader=paired_dm.train_dataloader(),
            optimizer=optimizer,
            device=device,
            side_scale_floor=side_floor,
        )
        checkpoint_path = checkpoint_dir / f"qat_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "quant_encoder_state_dict": quant_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "qat_prelaunch_sha256": sha256_file(receipt_path),
            },
            checkpoint_path,
        )
        row = {
            "epoch": epoch,
            **metrics,
            "scales": quant_encoder.export_scales_dict(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    quant_encoder.eval()
    qat_r2 = {
        view: _evaluate_view(model, val_records[view], device) for view in VIEWS
    }
    view_results: dict[str, Any] = {}
    for view in VIEWS:
        fp32 = ptq["view_results"][view]["fp32"]["per_session_r2"]
        if set(fp32) != set(qat_r2[view]):
            raise ValueError(f"C1 QAT {view} FP32/QAT session mismatch")
        delta = {name: float(qat_r2[view][name] - fp32[name]) for name in sorted(fp32)}
        view_results[view] = {
            "fp32": {"mean_r2": float(np.mean(list(fp32.values()))), "per_session_r2": fp32},
            "qat_int8": {
                "mean_r2": float(np.mean(list(qat_r2[view].values()))),
                "per_session_r2": qat_r2[view],
            },
            "delta_int8_minus_fp32": {
                "mean_r2": float(np.mean(list(delta.values()))),
                "median_r2": float(np.median(list(delta.values()))),
                "min_r2": float(np.min(list(delta.values()))),
                "positive_session_count": int(sum(value > 0 for value in delta.values())),
                "per_session_r2": delta,
            },
        }

    weights = _export_weights(quant_encoder)
    source_tags = [
        f"{name}::{view}"
        for view in VIEWS
        for name in paired_dm.session_splits["train"]
    ]
    frozen_scales = quant_encoder.to_frozen_scales(source_tags)
    diagnostics = quant_diagnostics(
        quant_encoder, weights, frozen_scales, val_records, device
    )
    decoder_after = _module_state_sha256(model.student.decoder)
    code_gates = {
        "sua_seed_mean_delta_ge_minus_0p01": view_results["sua"][
            "delta_int8_minus_fp32"
        ]["mean_r2"]
        >= DELTA_GATE,
        "pseudo_mua_seed_mean_delta_ge_minus_0p01": view_results["pseudo_mua"][
            "delta_int8_minus_fp32"
        ]["mean_r2"]
        >= DELTA_GATE,
        "integer_E_q_codes_exact": diagnostics["integer_E_q_code_mismatch_count"] == 0,
        "int32_overflow_zero": diagnostics["int32_overflow_count"] == 0,
        "max_edge_saturation_le_0p005": diagnostics["max_edge_saturation"]
        <= SATURATION_GATE,
        "decoder_state_byte_identical": decoder_before == decoder_after,
        "formal_test_unopened": True,
        "fixed_epoch_final_only": True,
    }
    legacy_exact_gate = diagnostics["E_dequant_torch_minus_numpy_max_abs"] == 0.0
    strict_gates = {
        **code_gates,
        "legacy_E_dequant_float_max_abs_exact_zero": legacy_exact_gate,
    }
    package = _export_integer_package(out_dir, weights, frozen_scales)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "uniform_c1_paired_encoder_qat",
        "scope": "C1 shared B3S/T4 encoder QAT W8A8 INT32 + FP32 decoder",
        "seed": args.seed,
        "qat_prelaunch": str(receipt_path),
        "qat_prelaunch_sha256": sha256_file(receipt_path),
        "source_ptq_report": str(ptq_path),
        "source_ptq_report_sha256": sha256_file(ptq_path),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "run_metadata": str(metadata_path),
        "run_metadata_sha256": sha256_file(metadata_path),
        "decoder_quantized_in_this_run": False,
        "decoder_precision": "FP32",
        "decoder_state_sha256_before": decoder_before,
        "decoder_state_sha256_after": decoder_after,
        "protocol": {
            "training_sessions": paired_dm.session_splits["train"],
            "development_sessions": [session_name_from_path(path) for path in val_files],
            "sealed_formal_session_names": sealed_names,
            "formal_test_files_opened": False,
            "qat_uses_source_behavior_labels": True,
            "validation_used_for_training": False,
            "validation_used_for_epoch_selection": False,
            "fixed_epoch_budget": QAT_EPOCHS,
            "selected_checkpoint": "final_fixed_budget_epoch_008",
            "activity_calibration_n": 30,
            "t4_label_rate_pool_n": 50,
            "evaluation_start_trial": EVALUATION_START,
            "paired_view_loss_weights": {"sua": 0.5, "pseudo_mua": 0.5},
            "source_side_scale_floor_margin": SIDE_SCALE_FLOOR_MARGIN,
            "source_side_abs_max_by_view": side_maxima,
            "source_side_scale_floor": side_floor,
            "source_side_scale_floor_projected_each_step": True,
        },
        "training": {
            "weight_learning_rate": LR_WEIGHT,
            "scale_learning_rate": LR_SCALE,
            "gradient_clip": GRADIENT_CLIP,
            "loss": "0.5_per_view*(task+0.75*prediction_anchor+0.075*identity_anchor+0.01*source_side_range)",
            "history": history,
        },
        "final_scales": asdict(frozen_scales),
        "view_results": view_results,
        "integer_diagnostics": diagnostics,
        "code_parity_gates": code_gates,
        "strict_legacy_gates": strict_gates,
        "qat_code_parity_pass": all(code_gates.values()),
        "qat_strict_legacy_pass": all(strict_gates.values()),
        "ptq_frozen_failure_reclassified": False,
        "integer_package": package,
        "formal_test_files_opened": False,
    }
    report_path = out_dir / "qat_report.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "sua_delta": view_results["sua"]["delta_int8_minus_fp32"]["mean_r2"],
                "pseudo_mua_delta": view_results["pseudo_mua"]["delta_int8_minus_fp32"]["mean_r2"],
                "max_saturation": diagnostics["max_edge_saturation"],
                "integer_code_mismatches": diagnostics["integer_E_q_code_mismatch_count"],
                "dequant_float_max_abs": diagnostics["E_dequant_torch_minus_numpy_max_abs"],
                "qat_code_parity_pass": payload["qat_code_parity_pass"],
                "qat_strict_legacy_pass": payload["qat_strict_legacy_pass"],
                "report": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if payload["qat_strict_legacy_pass"] else 11


if __name__ == "__main__":
    raise SystemExit(main())
