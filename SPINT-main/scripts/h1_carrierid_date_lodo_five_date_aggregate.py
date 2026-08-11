#!/usr/bin/env python3
"""Fail-closed receipt-only aggregate for five H1 source-date LODO evaluations.

This program opens exactly five immutable JSON receipts at known canonical
paths—one completed, one-shot H-S/H-C held-source-date evaluation for each
confirmatory date.  It does not enumerate recordings, open an NWB, dereference
the receipt's checkpoint/config paths, construct a trainer, or import a GPU
library.  A missing, mutable, malformed, or non-comparable date receipt stops
before an aggregate is published.

The aggregate reports all datewise numbers even if a date is negative.  It is
therefore a source/date-screen completion record, not a significance test and
not a route selector: it cannot automatically choose or launch EST4 or CI64.
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
import uuid
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DATES = ("19250108", "19250113", "19250115", "19250119", "19250120")
EVALUATION_SCHEMA = "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1"
AGGREGATE_SCHEMA = "h1_carrierid_date_lodo_five_date_heldout_aggregate_v1"
AGGREGATE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_FIVE_DATE_SOURCE_DATE_SCREEN_COMPLETE_NO_ROUTE_SELECTED"
ROUTE_PREREQUISITE_STATUS = "source/date screen complete"
EVALUATION_DIR = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase2/terminal_evaluations"
DEFAULT_OUTPUT = (
    ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase2/"
    "H1_CARRIERID_DATE_LODO_FIVE_DATE_HELDOUT_AGGREGATE_ROUTE_PREREQUISITE_v1.json"
)


class FiveDateAggregateError(RuntimeError):
    """A five-date terminal receipt is missing or fails a frozen contract."""


@dataclass(frozen=True)
class TerminalReceipt:
    outer_date: str
    path: Path
    sha256: str
    body: Mapping[str, Any]


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise FiveDateAggregateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    _need(
        isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
        f"{label} must be a SHA-256",
    )
    return value


def _finite(value: Any, label: str) -> float:
    _need(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def canonical_evaluation_path(outer_date: str, *, evaluation_dir: Path = EVALUATION_DIR) -> Path:
    _need(outer_date in DATES, f"unsupported outer date: {outer_date}")
    return (Path(evaluation_dir).resolve() /
            f"H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_HS_HC_TERMINAL_EVALUATION_v1.json")


def _expected_status(outer_date: str) -> str:
    return f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_HS_HC_EVALUATED"


def _read_immutable_receipt(*, outer_date: str, evaluation_dir: Path) -> TerminalReceipt:
    """Read one known result receipt, never scanning a data/checkpoint tree."""

    path = canonical_evaluation_path(outer_date, evaluation_dir=evaluation_dir)
    _need(path.is_file() and not path.is_symlink(), f"{outer_date}: canonical terminal receipt is missing")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{outer_date}: terminal receipt must be mode 0444")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FiveDateAggregateError(f"{outer_date}: terminal receipt JSON is unreadable") from exc
    _need(isinstance(value, dict), f"{outer_date}: terminal receipt must be a JSON object")
    return TerminalReceipt(outer_date=outer_date, path=path, sha256=_sha256(path), body=value)


def _validate_checkpoint(row: Any, *, outer_date: str, arm: str) -> dict[str, Any]:
    _need(isinstance(row, Mapping), f"{outer_date}/{arm}: checkpoint receipt entry missing")
    _sha(row.get("sha256"), f"{outer_date}/{arm} checkpoint")
    _sha(row.get("config_sha256"), f"{outer_date}/{arm} config")
    metadata = row.get("metadata")
    _need(isinstance(metadata, Mapping), f"{outer_date}/{arm}: checkpoint metadata missing")
    _need(metadata.get("arm") == arm and metadata.get("outer_date") == outer_date,
          f"{outer_date}/{arm}: checkpoint arm/date binding drift")
    _need(metadata.get("config_sha256") == row.get("config_sha256"),
          f"{outer_date}/{arm}: checkpoint/config SHA binding drift")
    _need(metadata.get("checkpoint_epoch_zero_based") == 49 and metadata.get("epochs_completed") == 50,
          f"{outer_date}/{arm}: checkpoint terminal epoch binding drift")
    _need(metadata.get("selected_by") == "fixed_terminal_epoch_no_validation_or_target_selection",
          f"{outer_date}/{arm}: checkpoint selection contract drift")
    _need(metadata.get("target_optimizer_steps") == 0 and metadata.get("target_backward_steps") == 0,
          f"{outer_date}/{arm}: checkpoint metadata records nonzero target optimizer/backward steps")
    _need(metadata.get("checkpoint_warm_start") is False,
          f"{outer_date}/{arm}: checkpoint warm start is forbidden")
    for key in ("initial_state_sha256", "phase2_source_binding_sha256", "phase1_source_manifest_sha256",
                "phase1_preflight_sha256", "config_sha256"):
        _sha(metadata.get(key), f"{outer_date}/{arm} metadata.{key}")
    return dict(metadata)


def _validate_metric(
    metric: Any, *, outer_date: str, arm: str, sessions: tuple[str, ...], window_hash: str,
) -> dict[str, Any]:
    _need(isinstance(metric, Mapping), f"{outer_date}/{arm}: metric missing")
    samples = metric.get("samples")
    _need(isinstance(samples, int) and samples > 0, f"{outer_date}/{arm}: invalid sample count")
    pooled = _finite(metric.get("pooled_r2"), f"{outer_date}/{arm}: pooled R2")
    _need(metric.get("r2_accumulator_dtype") == "float64", f"{outer_date}/{arm}: R2 is not float64")
    _need(metric.get("state_immutable") is True and metric.get("state_sha256_before") == metric.get("state_sha256_after"),
          f"{outer_date}/{arm}: model state changed during target forward")
    _sha(metric.get("state_sha256_before"), f"{outer_date}/{arm} state SHA")
    _need(metric.get("query_window_indices_sha256") == window_hash,
          f"{outer_date}/{arm}: strict query-window hash drift")
    per_session = metric.get("per_session")
    _need(isinstance(per_session, Mapping) and tuple(per_session) == sessions,
          f"{outer_date}/{arm}: per-session order/set drift")
    session_samples = 0
    rendered: dict[str, Any] = {}
    for session in sessions:
        row = per_session[session]
        _need(isinstance(row, Mapping) and isinstance(row.get("samples"), int) and int(row["samples"]) > 0,
              f"{outer_date}/{arm}/{session}: invalid sample binding")
        session_samples += int(row["samples"])
        rendered[session] = {"samples": int(row["samples"]), "r2": _finite(row.get("r2"), f"{outer_date}/{arm}/{session}: R2")}
    _need(session_samples == samples, f"{outer_date}/{arm}: per-session samples do not sum to pooled samples")
    _need(isinstance(metric.get("last_batch_size"), int) and int(metric["last_batch_size"]) > 0,
          f"{outer_date}/{arm}: target remainder evidence missing")
    return {"pooled_r2": pooled, "samples": samples, "per_session": rendered,
            "query_window_indices_sha256": window_hash}


def validate_terminal_receipt(terminal: TerminalReceipt) -> dict[str, Any]:
    """Validate existing receipt evidence only; do not touch referenced files."""

    date, body = terminal.outer_date, terminal.body
    _need(body.get("schema") == EVALUATION_SCHEMA and body.get("status") == _expected_status(date),
          f"{date}: terminal evaluation schema/status drift")
    _need(body.get("outer_date") == date, f"{date}: receipt outer_date drift")
    _need(body.get("checkpoint_binding_completed_before_target_open") is True,
          f"{date}: missing checkpoint-before-target binding")
    one_shot = body.get("one_shot")
    _need(isinstance(one_shot, Mapping), f"{date}: one-shot evidence missing")
    _need(Path(str(one_shot.get("canonical_output_path", ""))).resolve() == terminal.path,
          f"{date}: receipt was not written to its canonical one-shot output")
    _need(one_shot.get("same_date_prior_terminal_evaluation_receipts") == 0,
          f"{date}: receipt does not prove first/only same-date target evaluation")

    checkpoints = body.get("checkpoints")
    _need(isinstance(checkpoints, Mapping) and set(checkpoints) == {"H-S", "H-C"},
          f"{date}: exact H-S/H-C checkpoint receipt entries required")
    hs_checkpoint = _validate_checkpoint(checkpoints["H-S"], outer_date=date, arm="H-S")
    hc_checkpoint = _validate_checkpoint(checkpoints["H-C"], outer_date=date, arm="H-C")
    for key in ("phase2_source_binding_sha256", "phase1_source_manifest_sha256", "phase1_preflight_sha256"):
        _need(hs_checkpoint[key] == hc_checkpoint[key], f"{date}: H-S/H-C source binding mismatch at {key}")

    updates = body.get("deployment_updates")
    _need(isinstance(updates, Mapping) and updates.get("optimizer_steps") == 0
          and updates.get("backward_steps") == 0 and updates.get("model_state_unchanged") is True,
          f"{date}: terminal receipt records nonzero deployment target updates")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping) and scope.get("formal_heldout_opened") is False
          and scope.get("minival_opened") is False and scope.get("evalai_opened") is False,
          f"{date}: terminal receipt opened a forbidden scope")

    target = body.get("target")
    _need(isinstance(target, Mapping), f"{date}: target identity receipt missing")
    strict = target.get("strict_dataset")
    _need(isinstance(strict, Mapping) and strict.get("outer_date") == date,
          f"{date}: strict target dataset binding missing")
    raw_sessions = strict.get("sessions")
    _need(isinstance(raw_sessions, list) and len(raw_sessions) > 0 and len(raw_sessions) == len(set(raw_sessions))
          and all(isinstance(value, str) and value for value in raw_sessions), f"{date}: strict target sessions malformed")
    sessions = tuple(raw_sessions)
    _need(target.get("sessions") == list(sessions), f"{date}: target/strict session ordering mismatch")
    files = target.get("files")
    _need(isinstance(files, Mapping) and tuple(files) == sessions, f"{date}: target file binding drift")
    for session in sessions:
        _sha(files[session], f"{date}/{session} target file")
    window_hash = _sha(strict.get("window_indices_sha256"), f"{date}: strict target query-window hash")
    _need(strict.get("all_query_histories_start_at_or_after_fifth_trial") is True
          and target.get("all_query_histories_start_at_or_after_fifth_trial") is True,
          f"{date}: strict post-support history contract drift")
    _need(strict.get("samples") is not None and int(strict["samples"]) > 0,
          f"{date}: strict target dataset sample count missing")

    metrics = body.get("metrics")
    _need(isinstance(metrics, Mapping), f"{date}: metrics missing")
    hs = _validate_metric(metrics.get("h_s"), outer_date=date, arm="H-S", sessions=sessions, window_hash=window_hash)
    hc = _validate_metric(metrics.get("h_c"), outer_date=date, arm="H-C", sessions=sessions, window_hash=window_hash)
    _need(hs["samples"] == hc["samples"] == int(strict["samples"]),
          f"{date}: H-S/H-C/strict target sample counts differ")
    delta = float(hc["pooled_r2"] - hs["pooled_r2"])
    _need(math.isclose(_finite(metrics.get("h_c_minus_h_s"), f"{date}: reported delta"), delta, rel_tol=0.0, abs_tol=1e-12),
          f"{date}: reported H-C minus H-S delta drift")
    source_manifest_sha = _sha(body.get("source_manifest_sha256"), f"{date}: source manifest SHA")
    _need(source_manifest_sha == hs_checkpoint["phase1_source_manifest_sha256"]
          == hc_checkpoint["phase1_source_manifest_sha256"],
          f"{date}: evaluator/checkpoint Phase-1 source-manifest binding mismatch")
    return {
        "outer_date": date,
        "receipt": {"path": str(terminal.path), "sha256": terminal.sha256,
                     "schema": EVALUATION_SCHEMA, "status": _expected_status(date)},
        "source_manifest_sha256": source_manifest_sha,
        "target": {"sessions": list(sessions), "files": dict(files), "query_window_indices_sha256": window_hash,
                   "samples": int(strict["samples"])},
        "checkpoints": {
            "h_s": {"sha256": checkpoints["H-S"]["sha256"], "config_sha256": checkpoints["H-S"]["config_sha256"],
                    "initial_state_sha256": hs_checkpoint["initial_state_sha256"]},
            "h_c": {"sha256": checkpoints["H-C"]["sha256"], "config_sha256": checkpoints["H-C"]["config_sha256"],
                    "initial_state_sha256": hc_checkpoint["initial_state_sha256"]},
        },
        "metrics": {"h_s": hs, "h_c": hc, "h_c_minus_h_s": delta},
    }


def _summary(values: list[float]) -> dict[str, float]:
    _need(len(values) == len(DATES) and all(math.isfinite(value) for value in values),
          "summary requires five finite datewise values")
    ordered = sorted(values)
    return {"mean": float(sum(values) / len(values)), "median": float(ordered[len(ordered) // 2])}


def _sign(value: float) -> str:
    return "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> tuple[Path, str]:
    """Publish a new receipt atomically without replacing any active output."""

    output = Path(path).resolve()
    _need(not output.exists() and not output.is_symlink() and not os.path.lexists(str(output)),
          f"refusing to overwrite aggregate output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FiveDateAggregateError(f"refusing to overwrite aggregate output: {output}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(stat.S_IMODE(output.stat().st_mode) == 0o444, "aggregate output publication lost mode 0444")
    return output, hashlib.sha256(encoded).hexdigest()


def aggregate(*, evaluation_dir: str | Path = EVALUATION_DIR, output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Aggregate only when every one of the five frozen date receipts is valid."""

    directory = Path(evaluation_dir).resolve()
    terminals = [_read_immutable_receipt(outer_date=date, evaluation_dir=directory) for date in DATES]
    rows = {terminal.outer_date: validate_terminal_receipt(terminal) for terminal in terminals}
    _need(tuple(rows) == DATES, "five-date aggregate requires each frozen date exactly once")
    hs_values = [float(rows[date]["metrics"]["h_s"]["pooled_r2"]) for date in DATES]
    hc_values = [float(rows[date]["metrics"]["h_c"]["pooled_r2"]) for date in DATES]
    deltas = [float(rows[date]["metrics"]["h_c_minus_h_s"]) for date in DATES]
    signs = {date: _sign(float(rows[date]["metrics"]["h_c_minus_h_s"])) for date in DATES}
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": AGGREGATE_STATUS,
        "scope": "receipt-only five-date held-source-date LODO aggregation; no recording/checkpoint/trainer/GPU access by this program",
        "statistics_limit": "five-date descriptive mean/median and sign pattern only; not a confidence interval, significance test, route selector, or paper endpoint",
        "required_outer_dates": list(DATES),
        "all_five_date_receipts_present_and_validated": True,
        "per_date": rows,
        "summary": {
            "h_c_pooled_r2": _summary(hc_values),
            "h_s_pooled_r2": _summary(hs_values),
            "h_c_minus_h_s_pooled_r2": _summary(deltas),
            "h_c_minus_h_s_sign_pattern": {
                "by_date": signs,
                "positive_date_count": sum(value == "positive" for value in signs.values()),
                "negative_date_count": sum(value == "negative" for value in signs.values()),
                "zero_date_count": sum(value == "zero" for value in signs.values()),
                "all_five_dates_reported": True,
            },
        },
        "route_prerequisite": {
            "prior_status": "MISSING_IMPLEMENTATION_FAIL_CLOSED",
            "status": ROUTE_PREREQUISITE_STATUS,
            "completion_rule": "only after all five immutable date receipts validate; no individual date sign can stop or pass this aggregation",
            "automatic_route_selection": "FORBIDDEN",
            "EST4": "NOT_SELECTED_OR_LAUNCHED_BY_THIS_RECEIPT",
            "CI64": "NOT_SELECTED_OR_LAUNCHED_BY_THIS_RECEIPT",
            "H64": "NOT_SELECTED_OR_LAUNCHED_BY_THIS_RECEIPT",
        },
        "aggregator_scope": {
            "nwb_opened_by_aggregator": False,
            "checkpoint_opened_by_aggregator": False,
            "trainer_constructed_or_launched": False,
            "gpu_constructed_or_launched": False,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
        },
    }
    path, digest = _write_immutable(Path(output), payload)
    return {"status": AGGREGATE_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(aggregate(evaluation_dir=args.evaluation_dir, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
