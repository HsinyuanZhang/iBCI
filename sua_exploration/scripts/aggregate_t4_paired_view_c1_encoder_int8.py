#!/usr/bin/env python3
"""Aggregate three paired-view C1 encoder-only PTQ reports.

The aggregate recomputes every accuracy and hardware gate from per-session
values.  A failure authorizes one uniform three-seed QAT follow-up; it never
mixes PTQ and QAT seeds and never opens formal data.
"""
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

from write_t4_paired_view_c1_encoder_int8_prelaunch import (
    SEEDS,
    VIEWS,
    sha256_file,
    validate_receipt,
)


DELTA_GATE = -0.01
SATURATION_GATE = 0.005


def summarize_delta_matrix(matrix: np.ndarray, *, bootstrap_seed: int) -> dict[str, Any]:
    if matrix.shape != (3, 6):
        raise ValueError(f"paired delta matrix must be [3,6], got {matrix.shape}")
    flat = matrix.reshape(-1)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(20000, dtype=np.float64)
    for index in range(bootstrap.size):
        bootstrap[index] = float(np.mean(rng.choice(flat, size=flat.size, replace=True)))
    seed_means = matrix.mean(axis=1)
    session_means = matrix.mean(axis=0)
    mean = float(flat.mean())
    standard_error = float(flat.std(ddof=1) / np.sqrt(flat.size))
    return {
        "mean_delta_r2": mean,
        "median_delta_r2": float(np.median(flat)),
        "min_delta_r2": float(np.min(flat)),
        "max_delta_r2": float(np.max(flat)),
        "positive_cell_count": int(np.sum(flat > 0)),
        "seed_mean_delta_r2": seed_means.tolist(),
        "session_mean_delta_r2": session_means.tolist(),
        "all_seed_means_ge_minus_0p01": bool(np.all(seed_means >= DELTA_GATE)),
        "overall_mean_ge_minus_0p01": mean >= DELTA_GATE,
        "paired_two_se_interval": [mean - 2.0 * standard_error, mean + 2.0 * standard_error],
        "cell_bootstrap_95_interval": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "inference_note": "descriptive reused-development paired statistics; not formal confirmation",
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_package(report: dict[str, Any], report_path: Path) -> None:
    package = report.get("integer_package") or {}
    path = Path(str(package.get("path", ""))).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != package.get("sha256"):
        raise ValueError(f"integer package missing/hash drift: {report_path}")
    if package.get("one_package_serves_both_views") is not True:
        raise ValueError("C1 PTQ package is not shared across views")
    if package.get("decoder_quantized") is not False:
        raise ValueError("C1 PTQ package claims a quantized decoder")
    layers = package.get("layers") or []
    if [row.get("name") for row in layers] != ["pre_pool", "post0", "post1", "post2"]:
        raise ValueError("C1 PTQ package layer list drifted")
    for row in layers:
        if (
            row.get("weight_bits") != 8
            or row.get("activation_bits") != 8
            or row.get("accumulator_bits") != 32
            or row.get("integer_requant") is not True
        ):
            raise ValueError("C1 PTQ package is not W8A8/INT32 integer-requant")


def load_seed_report(
    report_path: Path,
    *,
    seed: int,
    receipt: dict[str, Any],
    prelaunch_path: Path,
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    report = _read_json(report_path)
    bound = receipt["seed_artifacts"][str(seed)]
    expected = {
        "schema_version": 1,
        "method": "ptq",
        "scope": "C1 shared B3S/T4 identity encoder W8A8 INT32 + FP32 decoder",
        "seed": seed,
        "checkpoint_sha256": bound["checkpoint"]["sha256"],
        "run_metadata_sha256": bound["run_metadata"]["sha256"],
        "prelaunch_sha256": sha256_file(prelaunch_path),
        "decoder_quantized_in_this_run": False,
        "decoder_precision": "FP32",
        "formal_test_files_opened": False,
    }
    failed = [key for key, value in expected.items() if report.get(key) != value]
    if failed:
        raise ValueError(f"seed {seed} PTQ report provenance failed: {failed}")
    if report.get("decoder_state_sha256_before") != report.get("decoder_state_sha256_after"):
        raise ValueError(f"seed {seed} decoder state changed")
    protocol = report.get("protocol") or {}
    expected_sessions = receipt["strict_manifest"]["development_sessions"]
    checks = {
        "source_count": len(protocol.get("source_sessions") or []) == 27,
        "dev_names": protocol.get("development_sessions") == expected_sessions,
        "formal_names": protocol.get("sealed_formal_session_names")
        == receipt["strict_manifest"]["sealed_formal_session_names"],
        "formal": protocol.get("formal_test_files_opened") is False,
        "records": protocol.get("scale_fit_records") == 54,
        "unique_source": protocol.get("scale_fit_unique_source_sessions") == 27,
        "view_records": protocol.get("scale_fit_records_by_view")
        == {"sua": 27, "pseudo_mua": 27},
        "view_weight": protocol.get("scale_fit_view_weight")
        == {"sua": 0.5, "pseudo_mua": 0.5},
        "equal_mass": protocol.get("scale_statistic_equal_mass_per_view") is True,
        "scale_cap": protocol.get("scale_statistic_max_samples_per_view_per_tensor")
        == 262144,
        "signed_t4": protocol.get("signed_t4_included_in_post0_input_scale_fit") is True,
        "shared_scales": protocol.get("one_shared_scale_set_per_seed") is True,
        "no_target_scale_selection": protocol.get("scale_selection_uses_behavior_targets")
        is False,
        "no_val_scale_selection": protocol.get("validation_used_for_scale_selection") is False,
        "budget": (
            protocol.get("activity_calibration_n"),
            protocol.get("t4_label_rate_pool_n"),
            protocol.get("evaluation_start_trial"),
        )
        == (30, 50, 50),
    }
    failed_checks = [key for key, value in checks.items() if not value]
    if failed_checks:
        raise ValueError(f"seed {seed} PTQ protocol failed: {failed_checks}")

    diagnostics = report.get("integer_diagnostics") or {}
    view_results = report.get("view_results") or {}
    recomputed_gates: dict[str, bool] = {}
    for view in VIEWS:
        row = view_results.get(view) or {}
        fp32 = (row.get("fp32") or {}).get("per_session_r2") or {}
        int8 = (row.get("int8") or {}).get("per_session_r2") or {}
        delta = (row.get("delta_int8_minus_fp32") or {}).get("per_session_r2") or {}
        if list(sorted(fp32)) != list(sorted(expected_sessions)) or set(int8) != set(fp32) or set(delta) != set(fp32):
            raise ValueError(f"seed {seed} {view} session set drifted")
        recomputed = {name: float(int8[name]) - float(fp32[name]) for name in fp32}
        if any(abs(recomputed[name] - float(delta[name])) > 1e-10 for name in fp32):
            raise ValueError(f"seed {seed} {view} delta arithmetic drifted")
        mean_delta = float(np.mean(list(recomputed.values())))
        if abs(mean_delta - float((row.get("delta_int8_minus_fp32") or {}).get("mean_r2"))) > 1e-10:
            raise ValueError(f"seed {seed} {view} mean delta drifted")
        recomputed_gates[f"{view}_seed_mean_delta_ge_minus_0p01"] = mean_delta >= DELTA_GATE
    recomputed_gates.update(
        {
            "integer_fake_quant_exact": diagnostics.get("integer_fake_quant_max_abs_E") == 0.0,
            "int32_overflow_zero": diagnostics.get("int32_overflow_count") == 0,
            "max_edge_saturation_le_0p005": float(
                diagnostics.get("max_edge_saturation", float("inf"))
            )
            <= SATURATION_GATE,
            "decoder_state_byte_identical": report.get("decoder_state_sha256_before")
            == report.get("decoder_state_sha256_after"),
            "one_shared_scale_set_for_both_views": True,
            "formal_test_unopened": report.get("formal_test_files_opened") is False,
        }
    )
    reported_gates = report.get("gates") or {}
    # The evaluator uses a shortened `pseudo_mua_seed...` key, while the loop
    # above constructs the same exact spelling for both views.
    if recomputed_gates != reported_gates:
        raise ValueError(f"seed {seed} PTQ gates are not recomputable")
    if report.get("ptq_pass") is not all(recomputed_gates.values()):
        raise ValueError(f"seed {seed} PTQ pass bit is inconsistent")
    _validate_package(report, report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prelaunch", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prelaunch_path = args.prelaunch.expanduser().resolve()
    receipt = validate_receipt(prelaunch_path)
    result_root = args.result_root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable aggregate: {out}")

    reports = {
        seed: load_seed_report(
            result_root / f"seed{seed}" / "ptq_report.json",
            seed=seed,
            receipt=receipt,
            prelaunch_path=prelaunch_path,
        )
        for seed in SEEDS
    }
    sessions = receipt["strict_manifest"]["development_sessions"]
    paired_stats: dict[str, Any] = {}
    matrices: dict[str, np.ndarray] = {}
    for view_index, view in enumerate(VIEWS):
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
        matrices[view] = matrix
        paired_stats[view] = summarize_delta_matrix(
            matrix, bootstrap_seed=20260805 + view_index
        )

    max_saturation = max(
        float(reports[seed]["integer_diagnostics"]["max_edge_saturation"])
        for seed in SEEDS
    )
    overflow = sum(
        int(reports[seed]["integer_diagnostics"]["int32_overflow_count"])
        for seed in SEEDS
    )
    exact = max(
        float(reports[seed]["integer_diagnostics"]["integer_fake_quant_max_abs_E"])
        for seed in SEEDS
    )
    gates = {
        "sua_overall_mean_delta_ge_minus_0p01": paired_stats["sua"][
            "overall_mean_ge_minus_0p01"
        ],
        "pseudo_mua_overall_mean_delta_ge_minus_0p01": paired_stats["pseudo_mua"][
            "overall_mean_ge_minus_0p01"
        ],
        "sua_all_seed_means_ge_minus_0p01": paired_stats["sua"][
            "all_seed_means_ge_minus_0p01"
        ],
        "pseudo_mua_all_seed_means_ge_minus_0p01": paired_stats["pseudo_mua"][
            "all_seed_means_ge_minus_0p01"
        ],
        "integer_fake_quant_exact": exact == 0.0,
        "int32_overflow_zero": overflow == 0,
        "max_edge_saturation_le_0p005": max_saturation <= SATURATION_GATE,
        "all_seed_decoders_byte_identical_fp32": all(
            reports[seed]["decoder_quantized_in_this_run"] is False
            and reports[seed]["decoder_state_sha256_before"]
            == reports[seed]["decoder_state_sha256_after"]
            for seed in SEEDS
        ),
        "formal_test_unopened": all(
            reports[seed]["formal_test_files_opened"] is False for seed in SEEDS
        ),
    }
    ptq_pass = all(gates.values())
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "C1 shared B3S/T4 identity encoder W8A8 INT32 + FP32 decoder",
        "method": "uniform_three_seed_ptq",
        "seeds": list(SEEDS),
        "views": list(VIEWS),
        "sessions": sessions,
        "prelaunch": str(prelaunch_path),
        "prelaunch_sha256": sha256_file(prelaunch_path),
        "reports": {
            str(seed): {
                "path": str((result_root / f"seed{seed}" / "ptq_report.json").resolve()),
                "sha256": sha256_file(result_root / f"seed{seed}" / "ptq_report.json"),
                "package_sha256": reports[seed]["integer_package"]["sha256"],
            }
            for seed in SEEDS
        },
        "paired_int8_minus_fp32": paired_stats,
        "diagnostics": {
            "integer_fake_quant_max_abs_E": exact,
            "int32_overflow_count": overflow,
            "max_edge_saturation": max_saturation,
        },
        "decoder_quantized_in_this_program": False,
        "decoder_precision": "FP32",
        "formal_test_files_opened": False,
        "gates": gates,
        "ptq_pass": ptq_pass,
        "next_step": "accept_c1_shared_encoder_ptq" if ptq_pass else "trigger_uniform_three_seed_encoder_qat",
        "qat_triggered_by_this_aggregate": not ptq_pass,
        "mixed_ptq_qat_seed_aggregate_forbidden": True,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sua_delta": paired_stats["sua"]["mean_delta_r2"],
                "pseudo_mua_delta": paired_stats["pseudo_mua"]["mean_delta_r2"],
                "max_saturation": max_saturation,
                "overflow": overflow,
                "ptq_pass": ptq_pass,
                "next_step": payload["next_step"],
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0 if ptq_pass else 10


if __name__ == "__main__":
    raise SystemExit(main())
