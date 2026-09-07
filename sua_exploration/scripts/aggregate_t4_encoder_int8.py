#!/usr/bin/env python3
"""Fail-closed aggregate for the three-seed SUA T4 encoder INT8 pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from t4_encoder_int8_protocol import (
    ACTIVITY_BUDGET,
    EVALUATION_START_TRIAL,
    T4_LABEL_BUDGET,
    selected_seed_entry,
    validate_selection,
)


SEEDS = (42, 43, 44)
DELTA_GATE = -0.01
SATURATION_GATE = 0.005


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_integer_package_arrays(package_path: Path, layers: list[dict]) -> None:
    expected_weight_shapes = {
        "pre_pool": (64, 100),
        "post0": (64, 68),
        "post1": (64, 64),
        "post2": (50, 64),
    }
    suffixes = {
        "weight_int8": np.dtype(np.int8),
        "weight_scale_fp32": np.dtype(np.float32),
        "bias_int32": np.dtype(np.int32),
        "requant_mult_int64": np.dtype(np.int64),
        "requant_shift_int32": np.dtype(np.int32),
    }
    expected_keys = {
        f"{name}.{suffix}"
        for name in expected_weight_shapes
        for suffix in suffixes
    }
    with np.load(package_path, allow_pickle=False) as arrays:
        if set(arrays.files) != expected_keys:
            raise ValueError("integer package NPZ key set drifted")
        for layer in layers:
            name = layer["name"]
            weight_shape = expected_weight_shapes[name]
            if tuple(layer.get("weight_shape") or ()) != weight_shape:
                raise ValueError(f"integer package {name} manifest shape drifted")
            output_dim = weight_shape[0]
            expected_shapes = {
                "weight_int8": weight_shape,
                "weight_scale_fp32": (output_dim,),
                "bias_int32": (output_dim,),
                "requant_mult_int64": (output_dim,),
                "requant_shift_int32": (1,),
            }
            for suffix, dtype in suffixes.items():
                array = arrays[f"{name}.{suffix}"]
                if array.dtype != dtype or array.shape != expected_shapes[suffix]:
                    raise ValueError(
                        f"integer package {name}.{suffix} dtype/shape drifted"
                    )


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
    for name, row in (("t4_minus_b0", b0), ("t4_minus_ts4", ts4)):
        if row.get("passes_all_gates") is not True:
            raise ValueError(f"INT8 trigger requires {name} to pass every FP32 gate")
        gates = row.get("gates") or {}
        if not gates or not all(value is True for value in gates.values()):
            raise ValueError(f"INT8 trigger has incomplete/failed gates for {name}")
    if not all(value >= 0.03 for value in deltas.values()):
        raise ValueError(f"INT8 trigger requires two >=0.03 FP32 deltas: {deltas}")
    protocol = source.get("protocol") or {}
    if (
        protocol.get("same_trial_count_and_prefix_for_all_arms") is not True
        or protocol.get("evaluation_backward_gradients") is not False
        or protocol.get("scored_epoch_window") != list(range(5, 13))
        or protocol.get("seeds") != list(SEEDS)
        or len(protocol.get("sessions") or []) != 6
    ):
        raise ValueError("source FP32 protocol audit failed")
    return deltas


def _load_seed(
    seed_dir: Path,
    seed: int,
    *,
    selection_path: Path,
    selection: dict,
    expected_train_sessions: list[str],
    expected_sessions: list[str],
    expected_formal_sessions: list[str],
) -> dict:
    expected_train_sessions = sorted(expected_train_sessions)
    expected_sessions = sorted(expected_sessions)
    expected_formal_sessions = sorted(expected_formal_sessions)
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
    if report.get("schema_version") != 1 or report.get("seed") != seed:
        raise ValueError(f"seed {seed}: report schema/seed mismatch")
    gates = report.get("gates") or {}
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError(f"seed {seed}: report contains an incomplete/failed gate")
    if report.get("decoder_quantized_in_this_run") is not False:
        raise ValueError(f"seed {seed}: decoder scope drifted")
    if report.get("scope") not in {
        "T4/B3S identity encoder W8A8 + FP32 decoder",
        "T4/B3S identity encoder QAT W8A8 + frozen FP32 decoder",
    }:
        raise ValueError(f"seed {seed}: quantization scope drifted")
    if report.get("final_architecture_selection_sha256") != _sha256_file(selection_path):
        raise ValueError(f"seed {seed}: final-selection receipt drifted")
    protocol = report.get("protocol") or {}
    if protocol.get("formal_test_files_opened") is not False:
        raise ValueError(f"seed {seed}: formal test was not sealed")
    if len(protocol.get("validation_sessions", [])) != 6:
        raise ValueError(f"seed {seed}: expected six validation sessions")
    if sorted(protocol["validation_sessions"]) != expected_sessions:
        raise ValueError(f"seed {seed}: validation sessions drifted")
    if sorted(protocol.get("sealed_formal_test_receipts") or []) != expected_formal_sessions:
        raise ValueError(f"seed {seed}: formal-test name receipts drifted")
    if (
        protocol.get("activity_calibration_n") != ACTIVITY_BUDGET
        or protocol.get("t4_label_feature_pool_n") != T4_LABEL_BUDGET
        or protocol.get("evaluation_start_trial") != EVALUATION_START_TRIAL
    ):
        raise ValueError(f"seed {seed}: expected frozen 30/50/50 protocol")
    if method == "ptq":
        if sorted(protocol.get("scale_fit_sessions") or []) != expected_train_sessions:
            raise ValueError(f"seed {seed}: PTQ scale source sessions drifted")
        if protocol.get("validation_labels_used_for_scale_selection") is not False:
            raise ValueError(f"seed {seed}: validation influenced PTQ scale selection")
    else:
        if sorted(protocol.get("training_sessions") or []) != expected_train_sessions:
            raise ValueError(f"seed {seed}: QAT training sessions drifted")
        if protocol.get("fixed_epoch_budget") != 8:
            raise ValueError(f"seed {seed}: QAT budget drifted")
        if protocol.get("validation_used_for_epoch_selection") is not False:
            raise ValueError(f"seed {seed}: validation influenced QAT epoch selection")
    fp = report["r2"]["fp32_encoder"]
    quant = report["r2"][r2_key]
    delta = report["r2"]["delta_int8_minus_fp32"]
    if set(fp["per_session"]) != set(quant["per_session"]):
        raise ValueError(f"seed {seed}: FP32/INT8 session sets differ")
    if set(fp["per_session"]) != set(delta["per_session"]):
        raise ValueError(f"seed {seed}: delta session set differs")
    for session in fp["per_session"]:
        observed_session = (
            float(quant["per_session"][session])
            - float(fp["per_session"][session])
        )
        if abs(observed_session - float(delta["per_session"][session])) > 1e-10:
            raise ValueError(f"seed {seed}: {session} delta is internally inconsistent")
    observed = float(quant["mean"]) - float(fp["mean"])
    if abs(observed - float(delta["mean"])) > 1e-10:
        raise ValueError(f"seed {seed}: mean delta is internally inconsistent")
    if float(delta["mean"]) < DELTA_GATE:
        raise ValueError(f"seed {seed}: quantized R2 delta missed the frozen gate")
    if float(report["max_edge_saturation"]) > SATURATION_GATE:
        raise ValueError(f"seed {seed}: saturation missed the frozen gate")
    if int(report["integer_alignment"]["int32_overflow_count"]) != 0:
        raise ValueError(f"seed {seed}: INT32 overflow is nonzero")
    if float(report["integer_alignment"]["max_abs_E"]) != 0.0:
        raise ValueError(f"seed {seed}: STE/integer output is not exact")
    checkpoint = Path(report["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file() or _sha256_file(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError(f"seed {seed}: checkpoint receipt drifted")
    selected = selected_seed_entry(selection, seed)
    selected_checkpoint = Path(selected["checkpoint"]["path"])
    if not selected_checkpoint.is_absolute():
        selected_checkpoint = Path(__file__).resolve().parents[2] / selected_checkpoint
    if checkpoint != selected_checkpoint.resolve():
        raise ValueError(f"seed {seed}: checkpoint is not the final selected T4@50 anchor")
    package = report.get("integer_package") or {}
    package_path = Path(package.get("path", "")).expanduser().resolve()
    if not package_path.is_file() or _sha256_file(package_path) != package.get("sha256"):
        raise ValueError(f"seed {seed}: integer package receipt drifted")
    layers = package.get("layers") or []
    if [layer.get("name") for layer in layers] != ["pre_pool", "post0", "post1", "post2"]:
        raise ValueError(f"seed {seed}: integer package layer set drifted")
    if any(
        layer.get("weight_bits") != 8
        or layer.get("activation_bits") != 8
        or layer.get("accumulator_bits") != 32
        or layer.get("integer_requant") is not True
        for layer in layers
    ):
        raise ValueError(f"seed {seed}: integer package is not W8A8/INT32")
    _validate_integer_package_arrays(package_path, layers)
    side_concat = package.get("side_concat") or {}
    if (
        side_concat.get("side_dim") != 4
        or side_concat.get("post0_input_dim") != 68
        or side_concat.get("integer_domain_concat") is not True
        or package.get("decoder_quantized") is not False
    ):
        raise ValueError(f"seed {seed}: integer T4 concat/decoder scope drifted")
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
        "package": package,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_fp32_aggregate", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    source_path = args.source_fp32_aggregate.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_deltas = _validate_positive_source(source)

    selection_path = args.selection.expanduser().resolve()
    selection, _selection_source, selection_deltas = validate_selection(
        selection_path, source_fp32_path=source_path
    )
    if source_deltas != selection_deltas:
        raise ValueError("source FP32 and final-selection deltas differ")
    expected_sessions = sorted((source.get("protocol") or {}).get("sessions") or [])
    manifest_path = Path(selection["strict_manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parents[2] / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_splits = manifest["session_splits"]
    expected_train_sessions = sorted(manifest_splits["train"])
    expected_formal_sessions = sorted(manifest_splits["test"])
    if expected_sessions != sorted(manifest_splits["val"]):
        raise ValueError("source FP32 validation sessions differ from the strict manifest")

    rows = [
        _load_seed(
            args.result_dir / f"seed{seed}",
            seed,
            selection_path=selection_path,
            selection=selection,
            expected_train_sessions=expected_train_sessions,
            expected_sessions=expected_sessions,
            expected_formal_sessions=expected_formal_sessions,
        )
        for seed in SEEDS
    ]
    session_sets = [set(row["per_session_delta_r2"]) for row in rows]
    if any(sessions != session_sets[0] for sessions in session_sets[1:]):
        raise ValueError("INT8 seed reports have different validation sessions")
    sessions = sorted(session_sets[0])
    if sessions != expected_sessions:
        raise ValueError("INT8 validation sessions differ from the source FP32 aggregate")
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
        "source_fp32_aggregate_sha256": _sha256_file(source_path),
        "final_architecture_selection": str(selection_path),
        "final_architecture_selection_sha256": _sha256_file(selection_path),
        "selected_architecture": selection["selected_architecture"],
        "activity_calibration_n": ACTIVITY_BUDGET,
        "t4_label_feature_pool_n": T4_LABEL_BUDGET,
        "evaluation_start_trial": EVALUATION_START_TRIAL,
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
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable INT8 aggregate: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
