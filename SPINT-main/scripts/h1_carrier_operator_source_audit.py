#!/usr/bin/env python3
"""Gate-closed source-only numerical audit for the H1 carrier operator.

This is an audit runner, not a trainer or an evaluator.  It deliberately does
not import the active H1 data producer until an operator supplies the explicit
``--run-source-audit`` gate.  When opened, its data scope is exactly the eleven
``H1_M4_FOLD0_SOURCE`` recordings; fold-0 targets, held-out data, and formal
paths are rejected before any source loader is called.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_carrier_operator_candidate import (
    CanonicalFP64Accumulator,
    FrozenCarrierOperator,
    design_from_rates,
    fit_o1,
    projected_rates,
)


AUDIT_SCHEMA = "h1_carrier_operator_source_audit_v1"
AUDIT_STATUS = "SOURCE_ONLY_OPERATOR_AUDIT_COMPLETE"
FORBIDDEN_PATH_TOKENS = (
    "held-out", "heldout", "formal", "evalai", "minival", "private", "test_ecephys",
)
COMMON_QUANTITIES = (
    "carrier", "raw_carrier", "raw_rows", "beta", "G", "sigma2", "projected_variance", "weight",
)


class SourceAuditError(RuntimeError):
    """Raised for a gate, source-scope, or numerical-audit violation."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuditError(message)


def _active_h1_module() -> ModuleType:
    """Late import: default/no-gate use cannot initialize the active loader."""

    return importlib.import_module("src.data.h1_m4_eb_pilot")


def _resolve_safe_path(path: str | Path, *, label: str, require_000954_root: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve()
    lower = str(resolved).lower()
    _need(not any(token in lower for token in FORBIDDEN_PATH_TOKENS), f"{label} rejects non-source path {resolved}")
    if require_000954_root:
        _need(resolved.name == "000954", f"{label} must be the H1 000954 source root")
    return resolved


def _canonical_json_bytes(body: Mapping[str, Any]) -> bytes:
    return (json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _atomic_immutable_json(output: str | Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    """Create one receipt atomically; an existing receipt is never overwritten."""

    target = Path(output).expanduser().resolve()
    _need(not target.exists(), f"refusing to overwrite existing audit receipt {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(body)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        target.chmod(0o444)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise
    return target, hashlib.sha256(payload).hexdigest()


def _array_range(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    _need(array.size > 0 and np.isfinite(array).all(), "audit range received empty/non-finite array")
    return {
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "abs_max": float(np.abs(array).max()),
    }


def _max_abs(left: np.ndarray, right: np.ndarray, *, quantity: str) -> float:
    lhs, rhs = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    _need(lhs.shape == rhs.shape, f"{quantity}: comparison shape mismatch {lhs.shape} != {rhs.shape}")
    _need(np.isfinite(lhs).all() and np.isfinite(rhs).all(), f"{quantity}: non-finite comparison")
    return float(np.abs(lhs - rhs).max())


def _operator_from_plan(plan: Any) -> FrozenCarrierOperator:
    return FrozenCarrierOperator(
        mean=np.asarray(plan.mean, dtype=np.float64),
        scale=np.asarray(plan.scale, dtype=np.float64),
        pcs=np.asarray(plan.pcs[: int(plan.q)], dtype=np.float64),
        ridge_lambda=float(plan.ridge_lambda),
        U=np.asarray(plan.U, dtype=np.float64),
        mu=np.asarray(plan.mu, dtype=np.float64),
        tau2=float(plan.tau2),
    )


def _first_support(record: Any, *, support_trials: int) -> tuple[tuple[float, ...], np.ndarray, np.ndarray]:
    values = tuple(float(value) for value in record.trial_values[:support_trials])
    _need(len(values) == support_trials, f"{record.session_name}: missing first M4 support trials")
    trials = tuple(record.blocks_for(value) for value in values)
    _need(all(np.asarray(trial.rates).shape[0] >= 2 for trial in trials),
          f"{record.session_name}: first M4 support has underspecified trial")
    rates = np.concatenate([np.asarray(trial.rates, dtype=np.float64) for trial in trials], axis=0)
    labels = np.concatenate([np.asarray(trial.velocity, dtype=np.float64) for trial in trials], axis=0)
    _need(rates.shape[0] == labels.shape[0] and rates.shape[0] > 0, f"{record.session_name}: invalid M4 block rows")
    return values, rates, labels


def _all_record_rates(record: Any) -> np.ndarray:
    blocks = tuple(np.asarray(trial.rates, dtype=np.float64) for trial in record.trials if np.asarray(trial.rates).shape[0])
    _need(bool(blocks), f"{record.session_name}: no complete 100-ms source blocks")
    rates = np.concatenate(blocks, axis=0)
    _need(np.isfinite(rates).all(), f"{record.session_name}: source rates are not finite")
    return rates


def _firing_rate_descriptors(
    records: Mapping[str, Any], source_sessions: Sequence[str], *, block_seconds: float, projection_dim: int
) -> dict[str, Any]:
    _need(np.isfinite(block_seconds) and block_seconds > 0.0, "invalid source block duration")
    per_record: dict[str, Any] = {}
    means: list[np.ndarray] = []
    pooled: list[np.ndarray] = []
    for name in source_sessions:
        rates = _all_record_rates(records[name])
        mean_hz = rates.mean(axis=0)
        means.append(mean_hz)
        pooled.append(rates)
        per_record[name] = {
            "complete_100ms_blocks": int(rates.shape[0]),
            "per_channel_mean_hz": _array_range(mean_hz),
            "fraction_channels_below_10hz": float(np.mean(mean_hz < 10.0)),
        }
    macro_channel_hz = np.stack(means, axis=0).mean(axis=0)
    pooled_channel_hz = np.concatenate(pooled, axis=0).mean(axis=0)
    expected_spikes_per_block = float(macro_channel_hz.sum() * block_seconds)
    dense_pca_macs = int(macro_channel_hz.size * projection_dim)
    event_additions = float(expected_spikes_per_block * projection_dim)
    return {
        "per_record": per_record,
        "macro_over_11_recordings_per_channel_hz": _array_range(macro_channel_hz),
        "pooled_over_all_complete_blocks_per_channel_hz": _array_range(pooled_channel_hz),
        "fraction_channels_below_10hz_macro": float(np.mean(macro_channel_hz < 10.0)),
        "crossover_raw_operation_descriptor": {
            "estimator_block_seconds": float(block_seconds),
            "firing_rate_crossover_hz": float(1.0 / block_seconds),
            "expected_spikes_per_100ms_block_macro": expected_spikes_per_block,
            "dense_projection_macs_per_block": dense_pca_macs,
            "event_driven_additions_per_block_at_macro_rate": event_additions,
            "event_to_dense_operation_ratio": float(event_additions / dense_pca_macs),
            "interpretation": "raw operation count only; sparse-index and memory energy are not measured",
        },
    }


def _record_audit(active: ModuleType, record: Any, plan: Any, operator: FrozenCarrierOperator) -> dict[str, Any]:
    values, rates, labels = _first_support(record, support_trials=int(active.SUPPORT_TRIALS))
    literal = active.fit_frozen_carrier(record, plan, values)
    o1 = fit_o1(rates, labels, operator)
    o2 = CanonicalFP64Accumulator.from_rates_labels(rates, labels, operator).solve_carrier(operator)
    z = projected_rates(rates, operator)
    design = design_from_rates(rates, operator)
    dtd = design.T @ design
    dty = design.T @ labels
    regularizer = np.eye(design.shape[1], dtype=np.float64) * operator.ridge_lambda
    regularizer[0, 0] = 0.0
    system = dtd + regularizer
    condition = float(np.linalg.cond(system))
    _need(np.isfinite(condition), f"{record.session_name}: ridge system condition is not finite")
    comparisons: dict[str, dict[str, float]] = {}
    for candidate_name, candidate in (("o1", o1), ("o2", o2)):
        comparisons[candidate_name] = {
            quantity: _max_abs(candidate[quantity], literal[quantity], quantity=f"{record.session_name}:{candidate_name}:{quantity}")
            for quantity in COMMON_QUANTITIES
        }
    o1_o2 = {quantity: _max_abs(o1[quantity], o2[quantity], quantity=f"{record.session_name}:o1_o2:{quantity}")
             for quantity in COMMON_QUANTITIES}
    return {
        "support_trial_values": list(values),
        "n": int(design.shape[0]),
        "system_condition_2norm": condition,
        "ranges": {
            "rates_hz": _array_range(rates),
            "labels": _array_range(labels),
            "z": _array_range(z),
            "DtD": _array_range(dtd),
            "Dty": _array_range(dty),
            "beta": _array_range(np.asarray(literal["beta"])),
            "raw_rows": _array_range(np.asarray(literal["raw_rows"])),
            "projected_variance": _array_range(np.asarray(literal["projected_variance"])),
            "weight": _array_range(np.asarray(literal["weight"])),
        },
        "max_abs_vs_active_literal": comparisons,
        "max_abs_o1_vs_o2": o1_o2,
    }


def run_source_audit(
    *,
    data_dir: str | Path,
    raw_receipt: str | Path,
    eb_receipt: str | Path,
    output: str | Path,
    execute_source_audit: bool,
) -> dict[str, Any]:
    """Run the gated source-only numerical audit and write one atomic receipt."""

    _need(execute_source_audit, "refusing source/NWB access without explicit --run-source-audit")
    source_root = _resolve_safe_path(data_dir, label="data_dir", require_000954_root=True)
    raw_path = _resolve_safe_path(raw_receipt, label="raw_receipt")
    eb_path = _resolve_safe_path(eb_receipt, label="eb_receipt")
    output_path = _resolve_safe_path(output, label="output")
    _need(raw_path.is_file() and eb_path.is_file(), "frozen plan receipts must exist before source access")
    _need(not output_path.exists(), f"refusing to overwrite existing audit receipt {output_path}")
    active = _active_h1_module()
    source_sessions = tuple(active.H1_M4_FOLD0_SOURCE)
    _need(len(source_sessions) == 11 and len(set(source_sessions)) == 11, "expected exactly eleven H1 source recordings")
    records = active.load_source_records(source_root)
    _need(tuple(records) == source_sessions and set(records) == set(source_sessions),
          "source loader returned records outside the exact frozen eleven-recording scope")
    for name in source_sessions:
        record = records[name]
        _need(str(record.session_name) == name, f"source record key/session mismatch for {name}")
        _need(str(record.date) != str(active.FOLD0_DATE), f"fold-0 target leaked into source record {name}")
    plan = active.reconstruct_frozen_plan(records, raw_path, eb_path)
    _need(tuple(plan.source_sessions) == source_sessions, "reconstructed plan source scope drift")
    operator = _operator_from_plan(plan)
    per_record = {name: _record_audit(active, records[name], plan, operator) for name in source_sessions}
    firing = _firing_rate_descriptors(
        records, source_sessions, block_seconds=float(active.BLOCK_SECONDS), projection_dim=operator.projection_dim
    )
    body = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "scope": {
            "run_source_audit_explicit": True,
            "allowed_recordings": list(source_sessions),
            "source_recordings_opened": len(source_sessions),
            "fold0_target_recordings_opened": 0,
            "heldout_recordings_opened": 0,
            "formal_recordings_opened": 0,
            "gpu_or_trainer_constructed": False,
        },
        "source_binding": {
            "data_dir": str(source_root),
            "source_sessions": list(source_sessions),
            "plan_transform_sha256": str(plan.transform_sha256),
            "raw_receipt_sha256": str(plan.raw_receipt_sha256),
            "eb_receipt_sha256": str(plan.eb_receipt_sha256),
        },
        "operator": {
            "num_channels": operator.num_channels,
            "projection_dim": operator.projection_dim,
            "label_dim": operator.label_dim,
            "carrier_dim": operator.carrier_dim,
            "ridge_lambda": operator.ridge_lambda,
            "packed_o2_state_floats": CanonicalFP64Accumulator.zeros(operator.design_dim, operator.label_dim).packed_float_count,
        },
        "per_record": per_record,
        "source_firing_rate_and_crossover": firing,
        "comparison_quantities": list(COMMON_QUANTITIES),
        "numerical_contract": {
            "o1_rss": "raw_residual_sum_of_squares",
            "o2_rss": "sufficient_statistic_quadratic_form",
            "factorization": "one_cholesky_factor_per_fit_reused_for_beta_hat_trace_and_G",
            "floating_merge_order": "FP64 merge grouping is not bitwise associative; canonical row order is required for byte reproducibility",
        },
    }
    written, digest = _atomic_immutable_json(output_path, body)
    return {"status": AUDIT_STATUS, "receipt_path": str(written), "receipt_sha256": digest, "records": len(source_sessions)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--raw-receipt", required=True, type=Path)
    parser.add_argument("--eb-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-source-audit", action="store_true", help="explicitly permit the eleven source recordings")
    args = parser.parse_args(argv)
    result = run_source_audit(
        data_dir=args.data_dir,
        raw_receipt=args.raw_receipt,
        eb_receipt=args.eb_receipt,
        output=args.output,
        execute_source_audit=bool(args.run_source_audit),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
