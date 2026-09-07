#!/usr/bin/env python3
"""Write the conditional all-three-seed C1 encoder-QAT prelaunch.

The writer fails closed until the frozen three-seed PTQ aggregate explicitly
triggers QAT.  QAT is uniform across seeds, source-only for optimization, and
keeps the co-trained decoder byte-identical FP32.  It reports integer-code
parity separately from the legacy zero-tolerance dequantized-float diagnostic;
the latter is never retroactively changed to a PTQ pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
VIEWS = ("sua", "pseudo_mua")
PTQ_PRELAUNCH_REL = (
    "sua_exploration/results/t4_paired_view_c1_encoder_int8_prelaunch_v2_20260805/"
    "receipt.json"
)
PTQ_RESULT_ROOT_REL = (
    "sua_exploration/results/t4_paired_view_c1_encoder_int8_ptq_v1_20260805"
)
SOURCE_FILES = (
    "sua_exploration/scripts/write_t4_paired_view_c1_encoder_qat_prelaunch.py",
    "sua_exploration/scripts/train_t4_paired_view_c1_encoder_qat.py",
    "sua_exploration/scripts/aggregate_t4_paired_view_c1_encoder_qat.py",
    "sua_exploration/scripts/run_t4_paired_view_c1_encoder_qat.sh",
    "sua_exploration/scripts/eval_t4_paired_view_c1_encoder_int8.py",
    "sua_exploration/mc_maze/paired_view_c1.py",
    "software-to-hardware/b3_fake_quant.py",
    "software-to-hardware/b3_qat_encoder.py",
    "software-to-hardware/b3_quant_engine.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _row(path: Path, *, relative: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = str(path.relative_to(ROOT)) if relative else str(path)
    return {"path": value, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def validate_ptq_trigger(aggregate: dict[str, Any], ptq_prelaunch_sha: str) -> None:
    expected = {
        "schema_version": 1,
        "scope": "C1 shared B3S/T4 identity encoder W8A8 INT32 + FP32 decoder",
        "method": "uniform_three_seed_ptq",
        "seeds": list(SEEDS),
        "views": list(VIEWS),
        "prelaunch_sha256": ptq_prelaunch_sha,
        "decoder_quantized_in_this_program": False,
        "decoder_precision": "FP32",
        "formal_test_files_opened": False,
        "ptq_pass": False,
        "next_step": "trigger_uniform_three_seed_encoder_qat",
        "qat_triggered_by_this_aggregate": True,
        "mixed_ptq_qat_seed_aggregate_forbidden": True,
    }
    failed = [key for key, value in expected.items() if aggregate.get(key) != value]
    if failed:
        raise ValueError(f"three-seed PTQ aggregate does not authorize QAT: {failed}")
    gates = aggregate.get("gates") or {}
    if not gates or all(value is True for value in gates.values()):
        raise ValueError("PTQ aggregate claims failure without a failed frozen gate")


def build_receipt(ptq_aggregate_path: Path) -> dict[str, Any]:
    ptq_prelaunch_path = ROOT / PTQ_PRELAUNCH_REL
    ptq_prelaunch = _read_json(ptq_prelaunch_path)
    ptq_aggregate_path = ptq_aggregate_path.expanduser().resolve()
    ptq_aggregate = _read_json(ptq_aggregate_path)
    validate_ptq_trigger(ptq_aggregate, sha256_file(ptq_prelaunch_path))

    seed_inputs: dict[str, Any] = {}
    for seed in SEEDS:
        report_path = ROOT / PTQ_RESULT_ROOT_REL / f"seed{seed}" / "ptq_report.json"
        report = _read_json(report_path)
        if report.get("seed") != seed or report.get("method") != "ptq":
            raise ValueError(f"seed {seed} PTQ report identity drift")
        if report.get("formal_test_files_opened") is not False:
            raise ValueError(f"seed {seed} PTQ report opened formal data")
        if report.get("prelaunch_sha256") != sha256_file(ptq_prelaunch_path):
            raise ValueError(f"seed {seed} PTQ prelaunch binding drift")
        package = report.get("integer_package") or {}
        package_path = Path(str(package.get("path", ""))).expanduser().resolve()
        if not package_path.is_file() or sha256_file(package_path) != package.get("sha256"):
            raise ValueError(f"seed {seed} PTQ package drift")
        seed_inputs[str(seed)] = {
            "seed": seed,
            "ptq_report": _row(report_path),
            "ptq_package": _row(package_path, relative=False),
            "checkpoint": {
                "path": report["checkpoint"],
                "sha256": report["checkpoint_sha256"],
            },
            "run_metadata": {
                "path": report["run_metadata"],
                "sha256": report["run_metadata_sha256"],
            },
        }

    source_code = {relative: _row(ROOT / relative) for relative in SOURCE_FILES}
    return {
        "schema_version": 1,
        "status": "authorized_uniform_three_seed_c1_encoder_qat",
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "C1_shared_B3S_T4_encoder_QAT_W8A8_INT32_plus_FP32_decoder",
        "ptq_prelaunch": _row(ptq_prelaunch_path),
        "ptq_aggregate": _row(ptq_aggregate_path),
        "ptq_frozen_failure_is_not_reclassified": True,
        "seed_inputs": seed_inputs,
        "formal_test_files_opened": False,
        "c2_authorized": False,
        "frozen_protocol": {
            "seeds": list(SEEDS),
            "views": list(VIEWS),
            "method_uniform_across_seeds": True,
            "mixed_ptq_qat_seed_aggregate_forbidden": True,
            "fixed_epochs": 8,
            "batch_size": 32,
            "activity_calibration_n": 30,
            "t4_label_rate_pool_n": 50,
            "evaluation_start_trial": 50,
            "source_sessions": 27,
            "paired_source_view_loss_weights": {"sua": 0.5, "pseudo_mua": 0.5},
            "validation_used_for_training": False,
            "validation_used_for_epoch_selection": False,
            "final_fixed_epoch_only": True,
            "weight_learning_rate": 1e-5,
            "scale_learning_rate": 1e-5,
            "gradient_clip": 1.0,
            "source_side_scale_floor": "1.10*max_abs_source_T4/127",
            "source_side_scale_floor_projection_after_every_step": True,
            "loss": "0.5_per_view*(task+0.75*fp32_prediction_anchor+0.075*normalized_identity_anchor+0.01*source_side_range_penalty)",
            "decoder_precision": "FP32",
            "decoder_trainable": False,
            "formal_paths_resolved": False,
        },
        "frozen_accuracy_hardware_gates": {
            "each_seed_each_view_mean_delta_int8_minus_fp32_r2_min": -0.01,
            "each_view_overall_mean_delta_int8_minus_fp32_r2_min": -0.01,
            "max_edge_saturation": 0.005,
            "int32_overflow_count": 0,
            "integer_E_q_code_mismatch_count": 0,
            "decoder_state_byte_identical": True,
            "formal_test_files_opened": False,
        },
        "legacy_ptq_exact_diagnostic": {
            "metric": "E_dequant_torch_minus_numpy_max_abs",
            "frozen_threshold": 0.0,
            "reported_separately": True,
            "may_not_reclassify_the_failed_ptq_gate": True,
            "included_in_strict_legacy_qat_disposition": True,
        },
        "source_code": source_code,
    }


def _resolve_hashed(row: dict[str, Any]) -> Path:
    raw = Path(str(row.get("path", ""))).expanduser()
    path = raw if raw.is_absolute() else ROOT / raw
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != row.get("sha256"):
        raise ValueError(f"QAT receipt file drift: {path}")
    return path


def validate_receipt(
    receipt_path: Path,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _read_json(receipt_path)
    expected = {
        "schema_version": 1,
        "status": "authorized_uniform_three_seed_c1_encoder_qat",
        "scope": "C1_shared_B3S_T4_encoder_QAT_W8A8_INT32_plus_FP32_decoder",
        "ptq_frozen_failure_is_not_reclassified": True,
        "formal_test_files_opened": False,
        "c2_authorized": False,
    }
    failed = [key for key, value in expected.items() if receipt.get(key) != value]
    if failed:
        raise ValueError(f"C1 QAT prelaunch failed: {failed}")
    _resolve_hashed(receipt["ptq_prelaunch"])
    aggregate_path = _resolve_hashed(receipt["ptq_aggregate"])
    aggregate = _read_json(aggregate_path)
    validate_ptq_trigger(aggregate, receipt["ptq_prelaunch"]["sha256"])
    for row in (receipt.get("source_code") or {}).values():
        _resolve_hashed(row)
    if seed is not None:
        if seed not in SEEDS:
            raise ValueError(f"unexpected QAT seed: {seed}")
        bound = receipt["seed_inputs"][str(seed)]
        _resolve_hashed(bound["ptq_report"])
        _resolve_hashed(bound["ptq_package"])
        checkpoint = Path(bound["checkpoint"]["path"]).expanduser().resolve()
        metadata = Path(bound["run_metadata"]["path"]).expanduser().resolve()
        if not checkpoint.is_file() or sha256_file(checkpoint) != bound["checkpoint"]["sha256"]:
            raise ValueError(f"seed {seed} QAT checkpoint drift")
        if not metadata.is_file() or sha256_file(metadata) != bound["run_metadata"]["sha256"]:
            raise ValueError(f"seed {seed} QAT metadata drift")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptq-aggregate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite QAT prelaunch: {out}")
    receipt = build_receipt(args.ptq_aggregate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_receipt(out)
    print(json.dumps({"out": str(out), "sha256": sha256_file(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
