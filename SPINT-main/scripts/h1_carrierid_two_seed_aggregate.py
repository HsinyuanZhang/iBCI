#!/usr/bin/env python3
"""Receipt-only two-seed aggregation for the H1 CarrierID H-S/H-C/H-C0 pilot.

This program intentionally reads only terminal-result JSON documents.  It does
not import a dataset, checkpoint, target loader, torch, or GPU component.  The
canonical seed-42 result is discovered from its Float64 terminal-receipt
schema, rather than from an accuracy constant.  A seed-43 result is discovered
by its future terminal-result schema.  Both must be present and fully
comparable before an immutable aggregate can be written.

``n=2`` here measures seed-to-seed dispersion only.  The reported sample
standard deviation and range are explicitly not confidence intervals and are
not inferential tests.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43)
ARMS = {
    "h_s_matched_spint": "base",
    "h_c_full": "full",
    "h_c0_separate_literal_zero": "zero",
}
EXPECTED_SESSIONS = 2
SEED42_SCHEMA = "h1_carrierid_h32_fold0_terminal_gate_v1"
SEED43_SCHEMA = "h1_carrierid_h32_seed43_fold0_terminal_gate_v1"
SEED42_STATUS = "PASS_H1_CARRIERID_H32_EXPANSION_AUTHORIZED"
SEED43_STATUSES = frozenset({
    "PASS_H1_CARRIERID_H32_SEED43_TERMINAL_EVALUATED",
    "STOP_H1_CARRIERID_H32_SEED43_TERMINAL_GATE_FAILED",
})
AGGREGATE_SCHEMA = "h1_carrierid_h32_two_seed_terminal_aggregate_v1"
AGGREGATE_STATUS_PASS = "PASS_H1_CARRIERID_H32_TWO_SEED_TERMINAL_AGGREGATE_COMPLETED_PASS"
AGGREGATE_STATUS_SEED43_GATE_FAILED = "COMPLETED_H1_CARRIERID_H32_TWO_SEED_AGGREGATE_SEED43_GATE_FAILED"
SEED43_NO_TARGET_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_H32_SEED43_THREE_ARM_TERMINAL_PREFLIGHT_NO_TARGET"
DEFAULT_SEED42_ROOT = ROOT / "pilot_artifacts/h1_carrierid"
DEFAULT_SEED43_ROOT = ROOT / "pilot_artifacts/h1_carrierid_seed43"


class TwoSeedAggregateError(RuntimeError):
    """A terminal result is missing, ambiguous, or fails comparability checks."""


@dataclass(frozen=True)
class TerminalInput:
    seed: int
    path: Path
    sha256: str
    receipt: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TwoSeedAggregateError(f"terminal result must be a JSON object: {path}")
    return value


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise TwoSeedAggregateError(message)


def _sha(value: Any, label: str) -> str:
    _need(isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value), f"{label} must be SHA-256")
    return value


def _finite(value: Any, label: str) -> float:
    _need(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def _seed42_candidate(receipt: Mapping[str, Any]) -> bool:
    """Identify the Float64 receipt, not the older float32 predecessor."""

    metric = receipt.get("metric_contract")
    return (
        receipt.get("schema") == SEED42_SCHEMA
        and receipt.get("status") == SEED42_STATUS
        and isinstance(metric, Mapping)
        and metric.get("r2_sse_tss_accumulator_dtype") == "float64"
    )


def _seed43_candidate(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("schema") == SEED43_SCHEMA and receipt.get("status") in SEED43_STATUSES


def discover_terminal_result(*, seed: int, root: Path) -> TerminalInput:
    """Find exactly one result JSON of the expected terminal-result schema."""

    _need(seed in SEEDS, f"unsupported seed: {seed}")
    _need(root.is_dir(), f"terminal-result discovery root does not exist: {root}")
    matches: list[tuple[Path, dict[str, Any]]] = []
    predicate = _seed42_candidate if seed == 42 else _seed43_candidate
    for path in sorted(root.rglob("*.json")):
        try:
            receipt = _json(path)
        # A result root can legitimately contain manifest arrays and unrelated
        # JSON; they are not candidates and must not make discovery brittle.
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TwoSeedAggregateError):
            continue
        if predicate(receipt):
            matches.append((path, receipt))
    _need(len(matches) == 1, f"seed{seed} needs exactly one discoverable terminal result; found {len(matches)} under {root}")
    path, receipt = matches[0]
    return TerminalInput(seed=seed, path=path.resolve(), sha256=_sha256(path), receipt=receipt)


def _validate_checkpoint_entry(entry: Any, *, key: str, expected_arm: str) -> dict[str, Any]:
    _need(isinstance(entry, Mapping), f"{key} checkpoint entry is missing")
    _sha(entry.get("sha256"), f"{key}.sha256")
    _sha(entry.get("config_sha256"), f"{key}.config_sha256")
    metadata = entry.get("metadata")
    _need(isinstance(metadata, Mapping), f"{key} checkpoint metadata is missing")
    _need(metadata.get("arm") == expected_arm, f"{key} checkpoint arm drift")
    _need(metadata.get("config_sha256") == entry.get("config_sha256"), f"{key} checkpoint/config SHA binding drift")
    _need(metadata.get("checkpoint_epoch_zero_based") == 49 and metadata.get("epochs_completed") == 50,
          f"{key} terminal epoch contract drift")
    _need(metadata.get("selected_by") == "fixed_terminal_epoch_no_selection", f"{key} checkpoint selection drift")
    return dict(metadata)


def _validate_metric(metric: Any, *, name: str, sessions: tuple[str, ...], target: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(metric, Mapping), f"{name} metric is missing")
    samples = metric.get("samples")
    _need(isinstance(samples, int) and samples > 0, f"{name} sample count is invalid")
    _finite(metric.get("pooled_r2"), f"{name} pooled R2")
    _need(metric.get("r2_accumulator_dtype") == "float64", f"{name} R2 accumulator is not float64")
    _need(metric.get("state_immutable") is True and metric.get("state_sha256_before") == metric.get("state_sha256_after"),
          f"{name} target forward changed model state")
    _need(metric.get("query_window_indices_sha256") == target.get("strict_query_window_indices_sha256"),
          f"{name} query window identity drift")
    per_session, session_samples = metric.get("per_session"), metric.get("session_samples")
    _need(isinstance(per_session, Mapping) and isinstance(session_samples, Mapping), f"{name} session metrics are missing")
    _need(tuple(per_session) == sessions and tuple(session_samples) == sessions, f"{name} session order/set drift")
    _need(sum(session_samples.values()) == samples, f"{name} session sample sum drift")
    for session in sessions:
        item = per_session[session]
        _need(isinstance(item, Mapping) and item.get("samples") == session_samples[session], f"{name}/{session} sample binding drift")
        _finite(item.get("r2"), f"{name}/{session} R2")
    return {
        "pooled_r2": float(metric["pooled_r2"]), "samples": samples,
        "session_samples": dict(session_samples), "query_window_indices_sha256": metric["query_window_indices_sha256"],
    }


def validate_terminal_result(terminal: TerminalInput) -> dict[str, Any]:
    """Validate result JSON evidence only; never dereference its checkpoints."""

    receipt, seed = terminal.receipt, terminal.seed
    if seed == 42:
        _need(_seed42_candidate(receipt), "seed42 result is not the sealed Float64 terminal receipt")
        # The sealed seed42 schema predates an explicit ``seed`` field.  Its
        # seed identity is therefore represented by the unique canonical
        # Float64 result schema/path and is cross-checked by source schedule
        # against the explicit seed43 result below.
        _need(receipt.get("seed") in (None, 42), "seed42 terminal receipt has inconsistent seed field")
    else:
        _need(_seed43_candidate(receipt), "seed43 result schema/status drift")
        _need(receipt.get("seed") == 43, "seed43 terminal receipt seed drift")
        no_target = receipt.get("no_target_preflight")
        _need(isinstance(no_target, Mapping), "seed43 no-target preflight binding is missing")
        _need(isinstance(no_target.get("path"), str) and bool(no_target.get("path")), "seed43 no-target preflight path is missing")
        _sha(no_target.get("sha256"), "seed43 no-target preflight SHA")
        _need(no_target.get("status") == SEED43_NO_TARGET_PREFLIGHT_STATUS,
              "seed43 no-target preflight status drift")
    _need(receipt.get("fold_date") == "19250101", f"seed{seed} fold date drift")
    _need(receipt.get("checkpoint_binding_completed_before_target_open") is True,
          f"seed{seed} lacks checkpoint-before-target evidence")
    metric_contract = receipt.get("metric_contract")
    _need(isinstance(metric_contract, Mapping) and metric_contract.get("prediction_dtype") == "float32"
          and metric_contract.get("r2_sse_tss_accumulator_dtype") == "float64",
          f"seed{seed} metric contract drift")
    scope = receipt.get("data_scope")
    _need(isinstance(scope, Mapping), f"seed{seed} data scope missing")
    for field in ("minival_opened", "heldout_opened", "formal_heldout_opened", "evalai_opened"):
        _need(scope.get(field) is False, f"seed{seed} terminal result opened forbidden scope: {field}")

    target = receipt.get("target")
    _need(isinstance(target, Mapping), f"seed{seed} target identity evidence missing")
    raw_sessions = target.get("sessions")
    _need(isinstance(raw_sessions, list) and len(raw_sessions) == EXPECTED_SESSIONS and len(set(raw_sessions)) == EXPECTED_SESSIONS
          and all(isinstance(name, str) and name for name in raw_sessions), f"seed{seed} target sessions drift")
    sessions = tuple(raw_sessions)
    target_files = target.get("files")
    _need(isinstance(target_files, Mapping) and tuple(target_files) == sessions, f"seed{seed} target file identity drift")
    for session in sessions:
        _sha(target_files[session], f"seed{seed} target file {session}")
    support_and_carrier = target.get("support_and_carrier_hashes")
    _need(isinstance(support_and_carrier, Mapping) and tuple(support_and_carrier) == sessions,
          f"seed{seed} target support/carrier identity drift")
    _sha(target.get("strict_query_window_indices_sha256"), f"seed{seed} query-window SHA")
    _need(target.get("all_query_histories_start_at_or_after_fifth_trial") is True
          and target.get("pooled_recordings_before_variance_weighted_r2") is True
          and target.get("remainder_preserved") is True, f"seed{seed} target metric scope drift")

    checkpoints = receipt.get("checkpoints")
    _need(isinstance(checkpoints, Mapping) and set(checkpoints) >= set(ARMS), f"seed{seed} required H-S/H-C/H-C0 checkpoints missing")
    metadata = {key: _validate_checkpoint_entry(checkpoints[key], key=key, expected_arm=arm) for key, arm in ARMS.items()}
    _need(metadata["h_s_matched_spint"].get("base_residual_literal_zero") is True
          and metadata["h_s_matched_spint"].get("residual_trainable") is False,
          f"seed{seed} H-S arm is not frozen base")
    for key, carrier_mode in (("h_c_full", "full"), ("h_c0_separate_literal_zero", "literal_zero_at_model_boundary")):
        meta = metadata[key]
        _need(meta.get("carrier_mode") == carrier_mode, f"seed{seed} {key} carrier mode drift")
        _need(meta.get("deployment_target_optimizer_steps") == 0 and meta.get("deployment_target_backward_steps") == 0,
              f"seed{seed} {key} target update metadata drift")
    _need(metadata["h_c_full"].get("carrierid_parameters") == metadata["h_c0_separate_literal_zero"].get("carrierid_parameters"),
          f"seed{seed} H-C/H-C0 carrier parameter mismatch")

    # These are terminal-receipt input bindings.  The aggregator intentionally
    # does not open the referenced checkpoint/config/source files, but it does
    # require their already-recorded SHA links to agree across all three arms.
    source = receipt.get("source_manifest") if seed == 42 else receipt.get("source")
    _need(isinstance(source, Mapping), f"seed{seed} source schedule evidence missing")
    source_manifest_sha = receipt.get("source_manifest_sha256") if seed == 42 else source.get("manifest_sha256")
    _sha(source_manifest_sha, f"seed{seed} source manifest SHA")
    for key, meta in metadata.items():
        _need(meta.get("source_manifest_sha256") == source_manifest_sha, f"seed{seed} {key} source-manifest SHA binding drift")

    metrics = receipt.get("metrics")
    _need(isinstance(metrics, Mapping) and isinstance(metrics.get("h_c_interventions"), Mapping), f"seed{seed} metrics missing")
    metric_rows = {
        "h_s": _validate_metric(metrics.get("h_s"), name=f"seed{seed}/H-S", sessions=sessions, target=target),
        "h_c": _validate_metric(metrics["h_c_interventions"].get("full"), name=f"seed{seed}/H-C", sessions=sessions, target=target),
        "h_c0": _validate_metric(metrics.get("h_c0"), name=f"seed{seed}/H-C0", sessions=sessions, target=target),
    }
    _need(len({row["samples"] for row in metric_rows.values()}) == 1, f"seed{seed} arm sample counts differ")
    _need(len({tuple(row["session_samples"].items()) for row in metric_rows.values()}) == 1, f"seed{seed} arm session samples differ")
    _need(len({row["query_window_indices_sha256"] for row in metric_rows.values()}) == 1, f"seed{seed} arm query windows differ")

    accounting = receipt.get("parameter_accounting")
    _need(isinstance(accounting, Mapping), f"seed{seed} parameter accounting missing")
    static = accounting.get("static_session_identity_encoder_parameters")
    whole = accounting.get("whole_model_parameters")
    _need(isinstance(static, Mapping) and isinstance(whole, Mapping), f"seed{seed} parameter counts missing")
    for group in (static, whole):
        _need(isinstance(group.get("h_s_spint"), int) and group["h_s_spint"] > 0
              and isinstance(group.get("h_c_h32"), int) and group["h_c_h32"] > 0,
              f"seed{seed} parameter count invalid")
    _need(static["h_c_h32"] == metadata["h_c_full"].get("carrierid_parameters"), f"seed{seed} H-C parameter accounting drift")
    _need(static["h_s_spint"] == metadata["h_c_full"].get("spint_identity_parameters"), f"seed{seed} H-S parameter accounting drift")
    _need(whole["h_c_h32"] == metadata["h_c_full"].get("whole_model_parameters"), f"seed{seed} H-C whole-model accounting drift")
    _need(whole["h_s_spint"] == metadata["h_c_full"].get("spint_whole_model_parameters"), f"seed{seed} H-S whole-model accounting drift")
    for group_name in ("target_session_updated_parameters", "target_session_optimizer_steps", "target_session_backward_steps"):
        values = accounting.get(group_name)
        _need(isinstance(values, Mapping) and all(values.get(arm) == 0 for arm in ("h_s", "h_c", "h_c0")),
              f"seed{seed} nonzero target update evidence: {group_name}")

    schedule = source.get("calibration_schedule_sha256")
    _sha(schedule, f"seed{seed} source calibration schedule")
    seed42_schedule = None
    if seed == 43:
        _need(source.get("schedule_is_new_for_seed43") is True, "seed43 terminal result does not prove a new source schedule")
        seed42_schedule = _sha(source.get("seed42_schedule_sha256"), "seed43 recorded seed42 calibration schedule")
    return {
        "seed": seed, "terminal": {"path": str(terminal.path), "sha256": terminal.sha256, "status": receipt["status"]},
        "sessions": list(sessions), "target_files": dict(target_files), "support_and_carrier_hashes": dict(support_and_carrier),
        "samples": metric_rows["h_c"]["samples"], "session_samples": metric_rows["h_c"]["session_samples"],
        "query_window_indices_sha256": metric_rows["h_c"]["query_window_indices_sha256"],
        "source_calibration_schedule_sha256": schedule,
        "seed42_schedule_sha256_recorded_by_seed43": seed42_schedule,
        "source_manifest_sha256": source_manifest_sha,
        "parameter_accounting": {"static": dict(static), "whole": dict(whole)},
        "r2": {name: row["pooled_r2"] for name, row in metric_rows.items()},
    }


def _cross_seed_validate(rows: Mapping[int, Mapping[str, Any]]) -> None:
    reference, replication = rows[42], rows[43]
    for field in ("sessions", "target_files", "support_and_carrier_hashes", "samples", "session_samples", "query_window_indices_sha256", "parameter_accounting"):
        _need(reference[field] == replication[field], f"seed42/seed43 comparability drift at {field}")
    _need(replication["seed42_schedule_sha256_recorded_by_seed43"] == reference["source_calibration_schedule_sha256"],
          "seed43 recorded seed42 schedule does not bind the sealed seed42 result")
    _need(reference["source_calibration_schedule_sha256"] != replication["source_calibration_schedule_sha256"],
          "seed43 source schedule is not independent from seed42")


def _sample_dispersion(values: list[float]) -> dict[str, Any]:
    _need(len(values) == 2 and all(math.isfinite(value) for value in values), "two-seed dispersion requires exactly two finite values")
    mean = sum(values) / 2.0
    return {
        "n_seeds": 2, "values_by_seed_order": values, "mean": mean,
        "sample_standard_deviation_ddof1": math.sqrt(sum((value - mean) ** 2 for value in values)),
        "range": max(values) - min(values), "positive_seed_count": sum(value > 0.0 for value in values),
        "interpretation": "n=2 seed dispersion only; not a confidence interval, p-value, or inferential estimate",
    }


def aggregate(*, seed42_root: Path, seed43_root: Path, output: Path) -> dict[str, Any]:
    """Write a formal immutable aggregate only after both result JSONs validate."""

    _need(not output.exists(), f"refusing to overwrite two-seed aggregate: {output}")
    terminal42 = discover_terminal_result(seed=42, root=seed42_root)
    terminal43 = discover_terminal_result(seed=43, root=seed43_root)
    rows = {42: validate_terminal_result(terminal42), 43: validate_terminal_result(terminal43)}
    _cross_seed_validate(rows)
    contrasts = {
        "h_c_minus_h_s": [rows[seed]["r2"]["h_c"] - rows[seed]["r2"]["h_s"] for seed in SEEDS],
        "h_c_minus_h_c0": [rows[seed]["r2"]["h_c"] - rows[seed]["r2"]["h_c0"] for seed in SEEDS],
    }
    all_gates_pass = all(rows[seed]["terminal"]["status"].startswith("PASS_") for seed in SEEDS)
    status = AGGREGATE_STATUS_PASS if all_gates_pass else AGGREGATE_STATUS_SEED43_GATE_FAILED
    payload = {
        "schema": AGGREGATE_SCHEMA, "status": status,
        "scope": "H1 held-in-calibration two-seed replication; no formal held-out or EvalAI endpoint",
        "statistics_limit": "n=2 reports seed dispersion only; it is not a confidence interval or inferential claim",
        "seeds": {str(seed): rows[seed] for seed in SEEDS},
        "contrasts": {name: _sample_dispersion(values) for name, values in contrasts.items()},
        "all_seed_terminal_gates_pass": all_gates_pass,
        "data_scope": {"nwb_opened_by_aggregator": False, "target_opened_by_aggregator": False,
                       "checkpoint_opened_by_aggregator": False, "cuda_constructed_by_aggregator": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    output.chmod(0o444)
    return {"status": status, "receipt_path": str(output), "receipt_sha256": _sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42-root", type=Path, default=DEFAULT_SEED42_ROOT)
    parser.add_argument("--seed43-root", type=Path, default=DEFAULT_SEED43_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(seed42_root=args.seed42_root, seed43_root=args.seed43_root, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
