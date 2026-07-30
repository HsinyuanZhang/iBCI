#!/usr/bin/env python3
"""Automatic encoder-only QAT fallback for a failed SUA T4 PTQ gate.

The decoder is frozen FP32.  Only the four B3S/T4 identity-encoder Linear
layers and their six shared activation scales are trained with the bit-exact
W8A8/INT32 STE path.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
_SUA = _ROOT / "sua_exploration"
_SWHW = _ROOT / "software-to-hardware"
_SCE = _ROOT / "streaming_calibration_exp"
for path in (_SUA, _SWHW, _SCE, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from b3_fake_quant import B3QATScales
from b3_hw_golden import B3Shapes, B3Weights
from b3_qat_encoder import wrap_early_pool_with_qat
from b3_quant_engine import (
    ABLATION_PRESETS,
    build_quant_engine_bundle,
    forward_quant_engine,
)
from dandi688_gradient_free_protocol import sha256_file
from eval_adaptation_dandi688 import load_side_feature_stats_for_run_metadata
from eval_t4_encoder_int8_dandi688 import (
    PTQ_DELTA_R2_THRESHOLD,
    SATURATION_RATE_THRESHOLD,
    _evaluate_records,
    _export_integer_package,
    _load_record,
    _validate_metadata,
)
from mc_maze.multisession_datamodule import (
    Dandi688MultiSessionDataModule,
    fit_behavior_stats,
    load_frozen_train_val_manifest,
)
from select_gradient_free_protocol_dandi688 import load_frozen_model
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder


def _unpack_batch(batch):
    if len(batch) != 5:
        raise ValueError(
            "T4 QAT requires (neural, behavior, calib, session, side_features)"
        )
    neural, behavior, calib, session, side = batch
    return neural, behavior, calib, session, side


def _export_weights(encoder) -> B3Weights:
    def array(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32)

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


def _set_trainable_encoder_only(model, quant_encoder) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in quant_encoder.parameters():
        parameter.requires_grad = True
    model.student.freeze_decoder()
    model.student.decoder.eval()
    quant_encoder.train()


def _train_epoch(
    *,
    model,
    quant_encoder,
    anchor_encoder,
    loader,
    optimizer,
    device: torch.device,
    gradient_clip: float,
) -> dict[str, float]:
    quant_encoder.train()
    model.student.decoder.eval()
    totals = {"loss": 0.0, "task": 0.0, "anchor_y": 0.0, "anchor_E": 0.0}
    batches = 0
    for batch in loader:
        neural, behavior, calib, _session, side = _unpack_batch(batch)
        neural = neural.to(device)
        behavior = behavior.to(device)
        calib = calib.to(device)
        side = side.to(device)

        optimizer.zero_grad(set_to_none=True)
        e_quant = quant_encoder.forward_batch(calib, side_features=side)
        y_quant = model.student.decode_with_identity(neural, e_quant)
        y_quant = y_quant[:, -1:, :] / 5.0
        target = behavior[:, -1:, :]

        with torch.no_grad():
            e_anchor = anchor_encoder.forward_batch(
                calib, side_features=side
            )
            y_anchor = model.student.decode_with_identity(neural, e_anchor)
            y_anchor = y_anchor[:, -1:, :] / 5.0

        loss_task = F.mse_loss(y_quant, target)
        loss_y = F.mse_loss(y_quant, y_anchor)
        loss_e = F.mse_loss(e_quant, e_anchor) / e_anchor.pow(2).mean().clamp_min(
            1e-8
        )
        loss = loss_task + 0.75 * loss_y + 0.075 * loss_e
        loss.backward()
        torch.nn.utils.clip_grad_norm_(quant_encoder.parameters(), gradient_clip)
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["task"] += float(loss_task.detach())
        totals["anchor_y"] += float(loss_y.detach())
        totals["anchor_E"] += float(loss_e.detach())
        batches += 1
    if batches == 0:
        raise RuntimeError("QAT train loader produced no full session-homogeneous batches")
    return {key: value / batches for key, value in totals.items()}


@torch.no_grad()
def _quant_diagnostics(
    quant_encoder,
    weights: B3Weights,
    scales,
    records: list[dict],
    device: torch.device,
) -> tuple[dict, int, float]:
    saturation: dict[str, list[float]] = {}
    overflow_count = 0
    exact_max_abs = 0.0
    shapes = B3Shapes(T=100, D=64, W=50, N=1, M=30)
    bundle = build_quant_engine_bundle(
        weights, shapes, scales, ABLATION_PRESETS["w8_a8_e8"]
    )
    for record in records:
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
            record["calib_trials"], bundle, side_features=record["side_features"]
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
    summary = {
        edge: {
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
        for edge, values in saturation.items()
    }
    return summary, overflow_count, exact_max_abs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--ptq_report", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr_weight", type=float, default=1e-5)
    parser.add_argument("--lr_scale", type=float, default=1e-5)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    run_dir = args.run_dir.expanduser().resolve()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ptq_path = args.ptq_report.expanduser().resolve()
    ptq = json.loads(ptq_path.read_text(encoding="utf-8"))
    if ptq.get("ptq_pass") is not False or ptq.get("next_step") != "run_encoder_qat":
        raise ValueError("QAT may start only from an explicit failed PTQ report")
    ckpt = Path(ptq["checkpoint"]).resolve()
    _validate_metadata(run_dir, metadata, ckpt)
    if sha256_file(ckpt) != ptq.get("checkpoint_sha256"):
        raise ValueError("PTQ checkpoint hash drifted before QAT")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    out_dir = args.out_dir.expanduser().resolve()
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    behavior_mean, behavior_std = fit_behavior_stats(
        train_files, 20, cache_dir=cache_dir
    )
    side_config = load_side_feature_stats_for_run_metadata(
        metadata, train_files, cache_dir
    )
    if side_config is None or side_config[0] != "t4":
        raise ValueError("QAT requires the real T4 side-feature configuration")
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

    dm = Dandi688MultiSessionDataModule(
        data_dir=str(data_dir),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=int((metadata.get("training") or {}).get("batch_size", 32)),
        window_size=50,
        calibration_n_trials=30,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=4,
        seed=int(metadata["seed"]),
        max_units_exclusive=100,
        cache_dir=str(cache_dir) if cache_dir else None,
        signal_view="sua",
        side_feature_group="t4",
        side_feature_pool_size=30,
        side_permutation_seed=None,
        train_val_manifest_path=str(manifest),
    )
    dm.setup("fit")
    if dm.session_splits["test"] != sealed_test_names or dm.session_files["test"]:
        raise RuntimeError("strict QAT datamodule attempted to resolve formal-test files")

    teacher = Path(metadata["teacher_checkpoint"]).resolve()
    model = load_frozen_model(ckpt, teacher, "B3S", device)
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
        base_encoder,
        qat_scales,
        num_trials=30,
        learnable_scales=True,
    ).to(device)
    model.student.id_encoder = quant_encoder
    _set_trainable_encoder_only(model, quant_encoder)

    weight_params = [
        parameter
        for name, parameter in quant_encoder.named_parameters()
        if "shared_scales" not in name
    ]
    scale_params = quant_encoder.shared_scales.scale_parameters()
    optimizer = torch.optim.Adam(
        [
            {"params": weight_params, "lr": args.lr_weight},
            {"params": scale_params, "lr": args.lr_scale},
        ],
        weight_decay=0.0,
    )
    torch.manual_seed(int(metadata["seed"]))
    np.random.seed(int(metadata["seed"]))

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        metrics = _train_epoch(
            model=model,
            quant_encoder=quant_encoder,
            anchor_encoder=anchor_encoder,
            loader=dm.train_dataloader(),
            optimizer=optimizer,
            device=device,
            gradient_clip=args.gradient_clip,
        )
        checkpoint_path = checkpoint_dir / f"qat_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "quant_encoder_state_dict": quant_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "source_checkpoint": str(ckpt),
                "source_checkpoint_sha256": sha256_file(ckpt),
                "fixed_epoch_budget": args.epochs,
            },
            checkpoint_path,
        )
        row = {
            "epoch": epoch,
            **metrics,
            "scales": quant_encoder.export_scales_dict(),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        history.append(row)
        print(json.dumps(row))

    # Fixed-budget final checkpoint only; validation is not used to choose a
    # QAT epoch or tune scales.
    quant_encoder.eval()
    int8_r2 = _evaluate_records(model, val_records, device=device)
    fp32_r2 = ptq["r2"]["fp32_encoder"]["per_session"]
    if set(fp32_r2) != set(int8_r2):
        raise ValueError("PTQ FP32 and QAT validation session sets differ")
    session_delta = {
        name: float(int8_r2[name] - fp32_r2[name]) for name in sorted(fp32_r2)
    }
    mean_fp32 = float(np.mean(list(fp32_r2.values())))
    mean_int8 = float(np.mean(list(int8_r2.values())))
    mean_delta = mean_int8 - mean_fp32

    weights = _export_weights(quant_encoder)
    frozen_scales = quant_encoder.to_frozen_scales(
        list(dm.session_splits["train"])
    )
    saturation, overflow_count, exact_max_abs = _quant_diagnostics(
        quant_encoder, weights, frozen_scales, val_records, device
    )
    max_saturation = max(
        (values["max"] for values in saturation.values()), default=0.0
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
        "fixed_epoch_final_only": True,
        "decoder_remained_frozen_fp32": True,
        "formal_test_unopened": True,
    }
    package = _export_integer_package(out_dir, weights, frozen_scales)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "T4/B3S identity encoder QAT W8A8 + frozen FP32 decoder",
        "decoder_quantized_in_this_run": False,
        "source_ptq_report": str(ptq_path),
        "source_ptq_report_sha256": sha256_file(ptq_path),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "train_val_manifest": str(manifest),
        "train_val_manifest_sha256": sha256_file(manifest),
        "protocol": {
            "training_sessions": dm.session_splits["train"],
            "validation_sessions": dm.session_splits["val"],
            "sealed_formal_test_receipts": sealed_test_names,
            "formal_test_files_opened": False,
            "qat_uses_training_behavior_labels": True,
            "validation_used_for_epoch_selection": False,
            "fixed_epoch_budget": args.epochs,
            "selected_checkpoint": "final fixed-budget epoch",
        },
        "training": {
            "lr_weight": args.lr_weight,
            "lr_scale": args.lr_scale,
            "gradient_clip": args.gradient_clip,
            "loss": "task + 0.75*FP-anchor prediction + 0.075*normalized E anchor",
            "history": history,
        },
        "final_scales": asdict(frozen_scales),
        "r2": {
            "fp32_encoder": {
                "mean": mean_fp32,
                "per_session": fp32_r2,
            },
            "qat_int8_encoder": {
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
        "saturation": saturation,
        "max_edge_saturation": max_saturation,
        "gates": gates,
        "qat_pass": all(gates.values()),
        "integer_package": package,
    }
    report_path = out_dir / "qat_report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fp32_mean_r2": mean_fp32,
                "qat_int8_mean_r2": mean_int8,
                "delta_r2": mean_delta,
                "max_saturation": max_saturation,
                "overflow": overflow_count,
                "qat_pass": payload["qat_pass"],
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0 if payload["qat_pass"] else 11


if __name__ == "__main__":
    raise SystemExit(main())
