#!/usr/bin/env python3
"""Fail-closed aggregate for the three-seed SUA T4 encoder INT8 pipeline."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


SEEDS = (42, 43, 44)


def _validate_positive_source(source: dict) -> dict[str, float]:
    if source.get("formal_test_files_opened") is not False:
        raise ValueError("source FP32 aggregate must leave formal test unopened")
    contrasts = source.get("contrasts") or {}
    b0 = contrasts.get("t4_vs_original_spint_b0") or {}
    ts4 = contrasts.get("t4_vs_shuffled_label_ts4") or {}
    deltas = {
        "t4_minus_b0": float(b0["mean_paired_delta_r2"]),
        "t4_minus_ts4": float(ts4["mean_paired_delta_r2"]),
    }
    if not all(value > 0.0 for value in deltas.values()):
        raise ValueError(f"INT8 trigger requires two strictly positive FP32 deltas: {deltas}")
    protocol = source.get("protocol") or {}
    if (
        protocol.get("same_trial_count_and_prefix_for_all_arms") is not True
        or protocol.get("evaluation_backward_gradients") is not False
        or protocol.get("scored_epoch_window") != list(range(5, 13))
    ):
        raise ValueError("source FP32 protocol audit failed")
    return deltas


def _load_seed(seed_dir: Path, seed: int) -> dict:
    ptq_path = seed_dir / "ptq" / "ptq_report.json"
    if not ptq_path.is_file():
        raise FileNotFoundError(ptq_path)
    ptq = json.loads(ptq_path.read_text(encoding="utf-8"))
    if ptq.get("ptq_pass") is True:
        report = ptq
        report_path = ptq_path
        method = "ptq"
        r2_key = "int8_encoder"
        passed = True
    else:
        qat_path = seed_dir / "qat" / "qat_report.json"
        if not qat_path.is_file():
            raise FileNotFoundError(qat_path)
        report = json.loads(qat_path.read_text(encoding="utf-8"))
        report_path = qat_path
        method = "qat"
        r2_key = "qat_int8_encoder"
        passed = report.get("qat_pass") is True
    if not passed:
        raise ValueError(f"seed {seed}: neither PTQ nor QAT passed")
    if report.get("decoder_quantized_in_this_run") is not False:
        raise ValueError(f"seed {seed}: decoder scope drifted")
    protocol = report.get("protocol") or {}
    if protocol.get("formal_test_files_opened") is not False:
        raise ValueError(f"seed {seed}: formal test was not sealed")
    if len(protocol.get("validation_sessions", [])) != 6:
        raise ValueError(f"seed {seed}: expected six validation sessions")
    fp = report["r2"]["fp32_encoder"]
    quant = report["r2"][r2_key]
    delta = report["r2"]["delta_int8_minus_fp32"]
    observed = float(quant["mean"]) - float(fp["mean"])
    if abs(observed - float(delta["mean"])) > 1e-10:
        raise ValueError(f"seed {seed}: mean delta is internally inconsistent")
    return {
        "seed": seed,
        "method": method,
        "report": str(report_path.resolve()),
        "fp32_mean_r2": float(fp["mean"]),
        "int8_mean_r2": float(quant["mean"]),
        "delta_r2": float(delta["mean"]),
        "per_session_delta_r2": {
            key: float(value) for key, value in delta["per_session"].items()
        },
        "max_edge_saturation": float(report["max_edge_saturation"]),
        "int32_overflow_count": int(
            report["integer_alignment"]["int32_overflow_count"]
        ),
        "integer_max_abs_E": float(
            report["integer_alignment"]["max_abs_E"]
        ),
        "package": report["integer_package"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_fp32_aggregate", required=True, type=Path)
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    source_path = args.source_fp32_aggregate.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_deltas = _validate_positive_source(source)

    rows = [
        _load_seed(args.result_dir / f"seed{seed}", seed) for seed in SEEDS
    ]
    session_sets = [set(row["per_session_delta_r2"]) for row in rows]
    if any(sessions != session_sets[0] for sessions in session_sets[1:]):
        raise ValueError("INT8 seed reports have different validation sessions")
    sessions = sorted(session_sets[0])
    matrix = np.asarray(
        [
            [row["per_session_delta_r2"][session] for session in sessions]
            for row in rows
        ],
        dtype=np.float64,
    )
    seed_means = matrix.mean(axis=1)
    session_means = matrix.mean(axis=0)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "SUA T4/B3S identity encoder W8A8 + FP32 decoder",
        "decoder_quantized_in_this_run": False,
        "source_fp32_aggregate": str(source_path),
        "source_positive_trigger": source_deltas,
        "seeds": list(SEEDS),
        "sessions": sessions,
        "seed_reports": rows,
        "methods": {str(row["seed"]): row["method"] for row in rows},
        "mean_delta_int8_minus_fp32_r2": float(matrix.mean()),
        "per_seed_mean_delta_r2": {
            str(seed): float(value) for seed, value in zip(SEEDS, seed_means)
        },
        "per_session_mean_delta_r2": {
            session: float(value)
            for session, value in zip(sessions, session_means)
        },
        "max_edge_saturation": max(
            row["max_edge_saturation"] for row in rows
        ),
        "int32_overflow_count": sum(
            row["int32_overflow_count"] for row in rows
        ),
        "integer_max_abs_E": max(row["integer_max_abs_E"] for row in rows),
        "all_three_seed_quantization_pass": True,
        "formal_test_files_opened": False,
    }
    out = args.out or args.result_dir / "aggregate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
