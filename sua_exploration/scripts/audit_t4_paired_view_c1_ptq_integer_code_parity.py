#!/usr/bin/env python3
"""Append-only integer-code parity audit for the frozen C1 PTQ result.

The original PTQ gate compared dequantized floating-point tensors with a
zero-tolerance threshold.  This audit asks the hardware-relevant question
separately: do the PyTorch fake-quant path and the NumPy/RTL golden path emit
the same final signed INT8 identity codes?  It never changes or reclassifies
the frozen PTQ verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "sua_exploration",
    ROOT / "sua_exploration/scripts",
    ROOT / "software-to-hardware",
    ROOT / "streaming_calibration_exp",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from b3_ckpt_loader import load_b3_weights_from_ckpt
from b3_fake_quant import B3QATScales
from b3_hw_golden import B3Shapes
from b3_qat_encoder import wrap_early_pool_with_qat
from b3_quant_engine import (
    ABLATION_PRESETS,
    FrozenActivationScales,
    build_quant_engine_bundle,
    forward_quant_engine,
)
from eval_t4_paired_view_c1_encoder_int8 import (
    VIEWS,
    _load_record,
    _metadata_checks,
    _view_feature_config,
)
from mc_maze.multisession_datamodule import (
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    session_name_from_path,
)
from select_gradient_free_protocol_dandi688 import load_frozen_model
from write_t4_paired_view_c1_encoder_int8_prelaunch import (
    SEEDS,
    sha256_file,
    validate_receipt,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@torch.no_grad()
def _audit_seed(
    *,
    seed: int,
    report_path: Path,
    prelaunch_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    report = _read_json(report_path)
    expected = {
        "schema_version": 1,
        "method": "ptq",
        "scope": "C1 shared B3S/T4 identity encoder W8A8 INT32 + FP32 decoder",
        "seed": seed,
        "prelaunch_sha256": sha256_file(prelaunch_path),
        "ptq_pass": False,
        "formal_test_files_opened": False,
        "decoder_quantized_in_this_run": False,
        "decoder_precision": "FP32",
    }
    failed = [key for key, value in expected.items() if report.get(key) != value]
    if failed:
        raise ValueError(f"seed {seed} frozen PTQ identity drift: {failed}")
    package = report.get("integer_package") or {}
    package_path = Path(str(package.get("path", ""))).expanduser().resolve()
    if not package_path.is_file() or sha256_file(package_path) != package.get("sha256"):
        raise ValueError(f"seed {seed} integer package hash drift")

    checkpoint = Path(report["checkpoint"]).expanduser().resolve()
    metadata_path = Path(report["run_metadata"]).expanduser().resolve()
    if sha256_file(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError(f"seed {seed} checkpoint hash drift")
    if sha256_file(metadata_path) != report["run_metadata_sha256"]:
        raise ValueError(f"seed {seed} metadata hash drift")
    metadata = _read_json(metadata_path)
    _metadata_checks(metadata, seed)
    receipt = validate_receipt(
        prelaunch_path,
        seed=seed,
        checkpoint=checkpoint,
        run_metadata=metadata_path,
    )
    manifest = ROOT / receipt["strict_manifest"]["path"]
    data_dir = ROOT / receipt["data_dir"]["path"]
    teacher = ROOT / receipt["teacher"]["path"]
    train_files, val_files, sealed_names = load_frozen_train_val_manifest(
        manifest, data_dir
    )
    if (len(train_files), len(val_files), len(sealed_names)) != (27, 6, 6):
        raise ValueError("integer-code audit strict split is not 27/6/6")
    if sealed_names != receipt["strict_manifest"]["sealed_formal_session_names"]:
        raise ValueError("integer-code audit formal-name receipt drift")
    feature_configs = {
        view: _view_feature_config(metadata, view, train_files) for view in VIEWS
    }
    behavior_mean, behavior_std = fit_behavior_stats(
        train_files, 20, cache_dir=feature_configs["sua"][6]
    )
    records = {
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

    selected = report["scale_search"]["selected_scales"]
    scales = FrozenActivationScales(**selected)
    weights = load_b3_weights_from_ckpt(checkpoint)
    bundle = build_quant_engine_bundle(
        weights,
        B3Shapes(T=100, D=64, W=50, N=1, M=30),
        scales,
        ABLATION_PRESETS["w8_a8_e8"],
    )
    model = load_frozen_model(checkpoint, teacher, "B3S", device)
    qat_scales = B3QATScales(
        input=scales.input_scale_i8,
        pre_out=scales.pre_out_scale_i8,
        mean=scales.mean_scale_i8,
        post0_out=scales.post0_out_scale_i8,
        post1_out=scales.post1_out_scale_i8,
        E=scales.E_int8_scale,
    )
    quant_encoder = wrap_early_pool_with_qat(
        model.student.id_encoder,
        qat_scales,
        num_trials=30,
        learnable_scales=False,
    ).to(device)
    quant_encoder.eval()

    by_view: dict[str, Any] = {}
    total_mismatches = 0
    code_max_abs_difference = 0
    dequant_max_abs_difference = 0.0
    compared_codes = 0
    for view in VIEWS:
        rows: dict[str, Any] = {}
        view_mismatches = 0
        view_code_max = 0
        view_dequant_max = 0.0
        view_codes = 0
        for record in records[view]:
            calib = torch.from_numpy(record["calib_trials"]).unsqueeze(0).to(device)
            side = torch.from_numpy(record["side_features"]).unsqueeze(0).to(device)
            stages_t = quant_encoder.forward_integer_with_stages(
                calib, side_features=side
            )
            stages_np = forward_quant_engine(
                record["calib_trials"], bundle, side_features=record["side_features"]
            )
            torch_code = np.rint(
                stages_t["E_q"].squeeze(0).cpu().numpy()
            ).astype(np.int16)
            numpy_code = np.asarray(stages_np["E"], dtype=np.int16)
            if torch_code.shape != numpy_code.shape:
                raise ValueError(
                    f"seed {seed} {view} {record['name']} E_q shape mismatch"
                )
            delta = torch_code.astype(np.int32) - numpy_code.astype(np.int32)
            mismatches = int(np.count_nonzero(delta))
            code_max = int(np.max(np.abs(delta))) if delta.size else 0
            dequant_max = float(
                np.max(
                    np.abs(
                        stages_t["E_dequant"].squeeze(0).cpu().numpy()
                        - stages_np["E_dequant"]
                    )
                )
            )
            count = int(delta.size)
            rows[record["name"]] = {
                "compared_codes": count,
                "integer_E_q_code_mismatch_count": mismatches,
                "integer_E_q_code_max_abs_difference": code_max,
                "E_dequant_torch_minus_numpy_max_abs": dequant_max,
            }
            view_mismatches += mismatches
            view_code_max = max(view_code_max, code_max)
            view_dequant_max = max(view_dequant_max, dequant_max)
            view_codes += count
        by_view[view] = {
            "sessions": rows,
            "compared_codes": view_codes,
            "integer_E_q_code_mismatch_count": view_mismatches,
            "integer_E_q_code_max_abs_difference": view_code_max,
            "E_dequant_torch_minus_numpy_max_abs": view_dequant_max,
        }
        total_mismatches += view_mismatches
        code_max_abs_difference = max(code_max_abs_difference, view_code_max)
        dequant_max_abs_difference = max(dequant_max_abs_difference, view_dequant_max)
        compared_codes += view_codes
    reported_float = float(
        report["integer_diagnostics"]["integer_fake_quant_max_abs_E"]
    )
    return {
        "seed": seed,
        "ptq_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "integer_package": {"path": str(package_path), "sha256": sha256_file(package_path)},
        "selected_scales": asdict(scales),
        "development_sessions": [session_name_from_path(path) for path in val_files],
        "sealed_formal_session_names": sealed_names,
        "formal_test_files_opened": False,
        "by_view": by_view,
        "compared_codes": compared_codes,
        "integer_E_q_code_mismatch_count": total_mismatches,
        "integer_E_q_code_max_abs_difference": code_max_abs_difference,
        "integer_E_q_codes_exact": total_mismatches == 0,
        "E_dequant_torch_minus_numpy_max_abs": dequant_max_abs_difference,
        "reported_E_dequant_max_abs": reported_float,
        "reported_float_diagnostic_reproduced": abs(
            dequant_max_abs_difference - reported_float
        )
        <= 1e-12,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prelaunch", type=Path, required=True)
    parser.add_argument("--ptq-aggregate", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    prelaunch = args.prelaunch.expanduser().resolve()
    aggregate_path = args.ptq_aggregate.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite integer-code audit: {out}")
    aggregate = _read_json(aggregate_path)
    expected_aggregate = {
        "method": "uniform_three_seed_ptq",
        "seeds": list(SEEDS),
        "views": list(VIEWS),
        "prelaunch_sha256": sha256_file(prelaunch),
        "ptq_pass": False,
        "next_step": "trigger_uniform_three_seed_encoder_qat",
        "formal_test_files_opened": False,
    }
    failed = [
        key for key, value in expected_aggregate.items() if aggregate.get(key) != value
    ]
    if failed:
        raise ValueError(f"frozen PTQ aggregate identity drift: {failed}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seeds = {
        str(seed): _audit_seed(
            seed=seed,
            report_path=result_root / f"seed{seed}" / "ptq_report.json",
            prelaunch_path=prelaunch,
            device=device,
        )
        for seed in SEEDS
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "C1 frozen PTQ final INT8 identity-code parity audit",
        "source_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "prelaunch": {"path": str(prelaunch), "sha256": sha256_file(prelaunch)},
        "ptq_aggregate": {
            "path": str(aggregate_path),
            "sha256": sha256_file(aggregate_path),
        },
        "seeds": seeds,
        "compared_codes": sum(row["compared_codes"] for row in seeds.values()),
        "integer_E_q_code_mismatch_count": sum(
            row["integer_E_q_code_mismatch_count"] for row in seeds.values()
        ),
        "integer_E_q_code_max_abs_difference": max(
            row["integer_E_q_code_max_abs_difference"] for row in seeds.values()
        ),
        "integer_E_q_codes_exact": all(
            row["integer_E_q_codes_exact"] for row in seeds.values()
        ),
        "E_dequant_torch_minus_numpy_max_abs": max(
            row["E_dequant_torch_minus_numpy_max_abs"] for row in seeds.values()
        ),
        "formal_test_files_opened": False,
        "ptq_frozen_pass_remains": False,
        "ptq_frozen_failure_reclassified": False,
        "interpretation": (
            "Final INT8 code parity is a separate deployment diagnostic; the "
            "pre-registered PTQ saturation and zero-tolerance dequantized-float "
            "gates remain unchanged."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "sha256": sha256_file(out),
                "compared_codes": payload["compared_codes"],
                "integer_E_q_code_mismatch_count": payload[
                    "integer_E_q_code_mismatch_count"
                ],
                "integer_E_q_code_max_abs_difference": payload[
                    "integer_E_q_code_max_abs_difference"
                ],
                "E_dequant_torch_minus_numpy_max_abs": payload[
                    "E_dequant_torch_minus_numpy_max_abs"
                ],
                "ptq_frozen_failure_reclassified": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
