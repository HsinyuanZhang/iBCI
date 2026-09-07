#!/usr/bin/env python3
"""Fail-closed aggregate for uniform three-seed C1 paired encoder QAT."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_t4_paired_view_c1_encoder_int8 import summarize_delta_matrix
from write_t4_paired_view_c1_encoder_qat_prelaunch import (
    SEEDS,
    VIEWS,
    sha256_file,
    validate_receipt,
)


DELTA_GATE = -0.01
SATURATION_GATE = 0.005


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_package(report: dict[str, Any]) -> None:
    package = report.get("integer_package") or {}
    path = Path(str(package.get("path", ""))).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != package.get("sha256"):
        raise ValueError("QAT integer package hash drift")
    if package.get("one_package_serves_both_views") is not True:
        raise ValueError("QAT package is not shared across views")
    if package.get("decoder_quantized") is not False:
        raise ValueError("QAT package quantized the decoder")
    layers = package.get("layers") or []
    if [row.get("name") for row in layers] != ["pre_pool", "post0", "post1", "post2"]:
        raise ValueError("QAT package layer set drift")
    if any(
        row.get("weight_bits") != 8
        or row.get("activation_bits") != 8
        or row.get("accumulator_bits") != 32
        or row.get("integer_requant") is not True
        for row in layers
    ):
        raise ValueError("QAT package is not W8A8/INT32 integer-requant")


def load_seed_report(
    path: Path,
    *,
    seed: int,
    receipt: dict[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    report = _read_json(path)
    bound = receipt["seed_inputs"][str(seed)]
    expected = {
        "schema_version": 1,
        "method": "uniform_c1_paired_encoder_qat",
        "scope": "C1 shared B3S/T4 encoder QAT W8A8 INT32 + FP32 decoder",
        "seed": seed,
        "qat_prelaunch_sha256": sha256_file(receipt_path),
        "source_ptq_report_sha256": bound["ptq_report"]["sha256"],
        "source_checkpoint_sha256": bound["checkpoint"]["sha256"],
        "run_metadata_sha256": bound["run_metadata"]["sha256"],
        "decoder_quantized_in_this_run": False,
        "decoder_precision": "FP32",
        "ptq_frozen_failure_reclassified": False,
        "formal_test_files_opened": False,
    }
    failed = [key for key, value in expected.items() if report.get(key) != value]
    if failed:
        raise ValueError(f"seed {seed} QAT report provenance failed: {failed}")
    if report.get("decoder_state_sha256_before") != report.get("decoder_state_sha256_after"):
        raise ValueError(f"seed {seed} QAT changed decoder state")
    protocol = report.get("protocol") or {}
    ptq_prelaunch = _read_json(ROOT / receipt["ptq_prelaunch"]["path"])
    expected_source = ptq_prelaunch["strict_manifest"]["source_sessions"]
    expected_dev = ptq_prelaunch["strict_manifest"]["development_sessions"]
    expected_formal = ptq_prelaunch["strict_manifest"]["sealed_formal_session_names"]
    checks = {
        "source": protocol.get("training_sessions") == expected_source,
        "development": protocol.get("development_sessions") == expected_dev,
        "formal_names": protocol.get("sealed_formal_session_names") == expected_formal,
        "formal": protocol.get("formal_test_files_opened") is False,
        "source_labels": protocol.get("qat_uses_source_behavior_labels") is True,
        "no_dev_train": protocol.get("validation_used_for_training") is False,
        "no_dev_select": protocol.get("validation_used_for_epoch_selection") is False,
        "epochs": protocol.get("fixed_epoch_budget") == 8,
        "final": protocol.get("selected_checkpoint") == "final_fixed_budget_epoch_008",
        "budget": (
            protocol.get("activity_calibration_n"),
            protocol.get("t4_label_rate_pool_n"),
            protocol.get("evaluation_start_trial"),
        )
        == (30, 50, 50),
        "view_weights": protocol.get("paired_view_loss_weights")
        == {"sua": 0.5, "pseudo_mua": 0.5},
        "floor_margin": protocol.get("source_side_scale_floor_margin") == 1.10,
        "floor_projection": protocol.get("source_side_scale_floor_projected_each_step")
        is True,
    }
    failed_checks = [key for key, value in checks.items() if not value]
    if failed_checks:
        raise ValueError(f"seed {seed} QAT protocol failed: {failed_checks}")

    view_results = report.get("view_results") or {}
    code_gates: dict[str, bool] = {}
    for view in VIEWS:
        row = view_results.get(view) or {}
        fp32 = (row.get("fp32") or {}).get("per_session_r2") or {}
        qat = (row.get("qat_int8") or {}).get("per_session_r2") or {}
        delta = (row.get("delta_int8_minus_fp32") or {}).get("per_session_r2") or {}
        if list(sorted(fp32)) != list(sorted(expected_dev)) or set(qat) != set(fp32) or set(delta) != set(fp32):
            raise ValueError(f"seed {seed} QAT {view} session set drift")
        recomputed = {name: float(qat[name]) - float(fp32[name]) for name in fp32}
        if any(abs(recomputed[name] - float(delta[name])) > 1e-10 for name in fp32):
            raise ValueError(f"seed {seed} QAT {view} delta arithmetic drift")
        mean = float(np.mean(list(recomputed.values())))
        if abs(mean - float((row.get("delta_int8_minus_fp32") or {}).get("mean_r2"))) > 1e-10:
            raise ValueError(f"seed {seed} QAT {view} mean delta drift")
        code_gates[f"{view}_seed_mean_delta_ge_minus_0p01"] = mean >= DELTA_GATE
    diagnostics = report.get("integer_diagnostics") or {}
    code_gates.update(
        {
            "integer_E_q_codes_exact": diagnostics.get("integer_E_q_code_mismatch_count") == 0,
            "int32_overflow_zero": diagnostics.get("int32_overflow_count") == 0,
            "max_edge_saturation_le_0p005": float(
                diagnostics.get("max_edge_saturation", float("inf"))
            )
            <= SATURATION_GATE,
            "decoder_state_byte_identical": report.get("decoder_state_sha256_before")
            == report.get("decoder_state_sha256_after"),
            "formal_test_unopened": report.get("formal_test_files_opened") is False,
            "fixed_epoch_final_only": True,
        }
    )
    if code_gates != report.get("code_parity_gates"):
        raise ValueError(f"seed {seed} QAT code-parity gates cannot be recomputed")
    legacy = diagnostics.get("E_dequant_torch_minus_numpy_max_abs") == 0.0
    strict_gates = {**code_gates, "legacy_E_dequant_float_max_abs_exact_zero": legacy}
    if strict_gates != report.get("strict_legacy_gates"):
        raise ValueError(f"seed {seed} QAT strict gates cannot be recomputed")
    if report.get("qat_code_parity_pass") is not all(code_gates.values()):
        raise ValueError(f"seed {seed} QAT code pass bit drift")
    if report.get("qat_strict_legacy_pass") is not all(strict_gates.values()):
        raise ValueError(f"seed {seed} QAT strict pass bit drift")
    _validate_package(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qat-prelaunch", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.qat_prelaunch.expanduser().resolve()
    receipt = validate_receipt(receipt_path)
    result_root = args.result_root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite QAT aggregate: {out}")
    reports = {
        seed: load_seed_report(
            result_root / f"seed{seed}" / "qat_report.json",
            seed=seed,
            receipt=receipt,
            receipt_path=receipt_path,
        )
        for seed in SEEDS
    }
    ptq_prelaunch = _read_json(ROOT / receipt["ptq_prelaunch"]["path"])
    sessions = ptq_prelaunch["strict_manifest"]["development_sessions"]
    stats: dict[str, Any] = {}
    for index, view in enumerate(VIEWS):
        matrix = np.asarray(
            [
                [
                    reports[seed]["view_results"][view]["delta_int8_minus_fp32"]
                    ["per_session_r2"][session]
                    for session in sessions
                ]
                for seed in SEEDS
            ],
            dtype=np.float64,
        )
        stats[view] = summarize_delta_matrix(matrix, bootstrap_seed=20260815 + index)

    diagnostics = {
        "integer_E_q_code_mismatch_count": sum(
            int(reports[seed]["integer_diagnostics"]["integer_E_q_code_mismatch_count"])
            for seed in SEEDS
        ),
        "integer_E_q_code_max_abs_difference": max(
            int(reports[seed]["integer_diagnostics"]["integer_E_q_code_max_abs_difference"])
            for seed in SEEDS
        ),
        "E_dequant_torch_minus_numpy_max_abs": max(
            float(reports[seed]["integer_diagnostics"]["E_dequant_torch_minus_numpy_max_abs"])
            for seed in SEEDS
        ),
        "int32_overflow_count": sum(
            int(reports[seed]["integer_diagnostics"]["int32_overflow_count"])
            for seed in SEEDS
        ),
        "max_edge_saturation": max(
            float(reports[seed]["integer_diagnostics"]["max_edge_saturation"])
            for seed in SEEDS
        ),
    }
    code_gates = {
        "sua_overall_mean_delta_ge_minus_0p01": stats["sua"]["overall_mean_ge_minus_0p01"],
        "pseudo_mua_overall_mean_delta_ge_minus_0p01": stats["pseudo_mua"][
            "overall_mean_ge_minus_0p01"
        ],
        "sua_all_seed_means_ge_minus_0p01": stats["sua"]["all_seed_means_ge_minus_0p01"],
        "pseudo_mua_all_seed_means_ge_minus_0p01": stats["pseudo_mua"][
            "all_seed_means_ge_minus_0p01"
        ],
        "integer_E_q_codes_exact": diagnostics["integer_E_q_code_mismatch_count"] == 0,
        "int32_overflow_zero": diagnostics["int32_overflow_count"] == 0,
        "max_edge_saturation_le_0p005": diagnostics["max_edge_saturation"] <= SATURATION_GATE,
        "all_decoders_byte_identical_fp32": all(
            reports[seed]["decoder_state_sha256_before"]
            == reports[seed]["decoder_state_sha256_after"]
            and reports[seed]["decoder_quantized_in_this_run"] is False
            for seed in SEEDS
        ),
        "formal_test_unopened": all(
            reports[seed]["formal_test_files_opened"] is False for seed in SEEDS
        ),
        "uniform_qat_all_three_seeds": True,
    }
    strict_gates = {
        **code_gates,
        "legacy_E_dequant_float_max_abs_exact_zero": diagnostics[
            "E_dequant_torch_minus_numpy_max_abs"
        ]
        == 0.0,
    }
    code_pass = all(code_gates.values())
    strict_pass = all(strict_gates.values())
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "C1 shared B3S/T4 encoder QAT W8A8 INT32 + FP32 decoder",
        "method": "uniform_three_seed_c1_paired_encoder_qat",
        "seeds": list(SEEDS),
        "views": list(VIEWS),
        "sessions": sessions,
        "qat_prelaunch": str(receipt_path),
        "qat_prelaunch_sha256": sha256_file(receipt_path),
        "reports": {
            str(seed): {
                "path": str(result_root / f"seed{seed}" / "qat_report.json"),
                "sha256": sha256_file(result_root / f"seed{seed}" / "qat_report.json"),
                "package_sha256": reports[seed]["integer_package"]["sha256"],
            }
            for seed in SEEDS
        },
        "paired_int8_minus_fp32": stats,
        "integer_diagnostics": diagnostics,
        "code_parity_gates": code_gates,
        "strict_legacy_gates": strict_gates,
        "qat_code_parity_pass": code_pass,
        "qat_strict_legacy_pass": strict_pass,
        "ptq_frozen_failure_reclassified": False,
        "decoder_quantized_in_this_program": False,
        "decoder_precision": "FP32",
        "formal_test_files_opened": False,
        "disposition": (
            "strict_qat_pass"
            if strict_pass
            else "integer_code_deployment_pass_but_legacy_float_exact_gate_remains_failed"
            if code_pass
            else "qat_failed_accuracy_or_hardware_gate"
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if strict_pass else 11


if __name__ == "__main__":
    raise SystemExit(main())
