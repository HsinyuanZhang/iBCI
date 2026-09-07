#!/usr/bin/env python3
"""Publish the frozen H1 H-S/H-C five-date aggregate from legacy receipts.

The five terminal evaluator receipts were produced before the explicit
``checkpoint_binding_completed_before_target_open`` field was added to the
receipt contract.  They are immutable and must not be rewritten.  This
receipt-only adapter validates the old receipts' equivalent checkpoint and
deployment invariants, adds the missing assertion only to an in-memory copy,
then delegates all metric/target validation and aggregate construction to the
existing v1 implementation.  No target data, checkpoint, trainer, GPU, or
optimizer is opened or constructed by this program.

The resulting output intentionally uses the v1 canonical schema/path because
the already-frozen H-LS route gate consumes that schema.  The output records
the compatibility mode and every local input SHA so the legacy normalization
cannot be mistaken for a mutation of an evaluator receipt.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import stat
from typing import Any, Mapping

try:
    from scripts import h1_carrierid_date_lodo_five_date_aggregate as v1
    from scripts import h1_carrierid_date_lodo_five_date_aggregate_v2 as v2
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    import h1_carrierid_date_lodo_five_date_aggregate as v1
    import h1_carrierid_date_lodo_five_date_aggregate_v2 as v2


ROOT = v1.ROOT
DATES = v1.DATES
EVALUATION_DIR = v1.EVALUATION_DIR
DEFAULT_OUTPUT = v1.DEFAULT_OUTPUT
SCHEMA = v1.AGGREGATE_SCHEMA
STATUS = v1.AGGREGATE_STATUS


class LegacyCompatibilityError(RuntimeError):
    """A legacy receipt failed the frozen v1 contract or compatibility gate."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise LegacyCompatibilityError(message)


def _validate_equivalent_binding(body: Mapping[str, Any], *, date: str) -> None:
    """Require evidence that makes the legacy field omission safe to bridge."""

    _need("checkpoint_binding_completed_before_target_open" not in body,
          f"{date}: refusing compatibility mode because explicit binding field is present")
    checkpoints = body.get("checkpoints")
    _need(isinstance(checkpoints, Mapping) and set(checkpoints) == {"H-S", "H-C"},
          f"{date}: legacy receipt lacks exact paired checkpoints")
    for arm in ("H-S", "H-C"):
        row = checkpoints[arm]
        _need(isinstance(row, Mapping), f"{date}/{arm}: malformed legacy checkpoint row")
        metadata = row.get("metadata")
        _need(isinstance(metadata, Mapping), f"{date}/{arm}: legacy checkpoint metadata missing")
        _need(metadata.get("outer_date") == date and metadata.get("arm") == arm,
              f"{date}/{arm}: legacy checkpoint arm/date binding drift")
        _need(metadata.get("checkpoint_epoch_zero_based") == 49
              and metadata.get("epochs_completed") == 50
              and metadata.get("checkpoint_warm_start") is False
              and metadata.get("selected_by") == "fixed_terminal_epoch_no_validation_or_target_selection",
              f"{date}/{arm}: legacy checkpoint is not the fixed fresh e49 endpoint")
        _need(metadata.get("target_optimizer_steps") == 0
              and metadata.get("target_backward_steps") == 0,
              f"{date}/{arm}: legacy checkpoint records target updates")
    updates = body.get("deployment_updates")
    _need(isinstance(updates, Mapping)
          and updates.get("optimizer_steps") == 0
          and updates.get("backward_steps") == 0
          and updates.get("model_state_unchanged") is True,
          f"{date}: legacy deployment update evidence is not immutable")
    one_shot = body.get("one_shot")
    _need(isinstance(one_shot, Mapping)
          and one_shot.get("same_date_prior_terminal_evaluation_receipts") == 0,
          f"{date}: legacy receipt is not first/only same-date evaluation")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping)
          and scope.get("formal_heldout_opened") is False
          and scope.get("minival_opened") is False
          and scope.get("evalai_opened") is False,
          f"{date}: legacy receipt opened a forbidden scope")


def _legacy_terminal(date: str, path: Path) -> tuple[v1.TerminalReceipt, dict[str, Any]]:
    """Read one immutable local copy and synthesize the missing field in memory."""

    try:
        imported = v2._read_input_terminal(outer_date=date, path=path)
    except v2.FiveDateDescriptiveAggregateError as error:
        raise LegacyCompatibilityError(str(error)) from error
    original = imported.terminal.body
    _validate_equivalent_binding(original, date=date)
    bridged = copy.deepcopy(dict(original))
    bridged["checkpoint_binding_completed_before_target_open"] = True
    terminal = v1.TerminalReceipt(
        outer_date=date,
        path=imported.declared_canonical_path,
        sha256=imported.input_sha256,
        body=bridged,
    )
    return terminal, {
        "local_input_path": str(imported.input_path),
        "local_input_sha256": imported.input_sha256,
        "declared_canonical_output_path": str(imported.declared_canonical_path),
        "source_receipt_mutated": False,
        "synthesized_field": "checkpoint_binding_completed_before_target_open",
        "synthesized_value": True,
    }


def aggregate(*, evaluation_dir: str | Path = EVALUATION_DIR,
              output: str | Path = DEFAULT_OUTPUT,
              receipt_paths: Mapping[str, str | Path] | None = None) -> dict[str, Any]:
    """Validate all five immutable legacy receipts and publish one v1 aggregate."""

    output_path = Path(output).resolve()
    _need(not output_path.exists() and not output_path.is_symlink(),
          f"refusing to overwrite aggregate output: {output_path}")
    if receipt_paths is None:
        directory = Path(evaluation_dir).resolve()
        paths = {date: v1.canonical_evaluation_path(date, evaluation_dir=directory) for date in DATES}
    else:
        _need(set(receipt_paths) == set(DATES), "explicit receipt-path grid must contain exactly the five frozen dates")
        paths = {date: Path(receipt_paths[date]).resolve() for date in DATES}

    terminals: dict[str, v1.TerminalReceipt] = {}
    compatibility: dict[str, Any] = {}
    for date in DATES:
        terminal, bridge = _legacy_terminal(date, paths[date])
        terminals[date] = terminal
        compatibility[date] = bridge

    rows = {date: v1.validate_terminal_receipt(terminals[date]) for date in DATES}
    hs_values = [float(rows[date]["metrics"]["h_s"]["pooled_r2"]) for date in DATES]
    hc_values = [float(rows[date]["metrics"]["h_c"]["pooled_r2"]) for date in DATES]
    deltas = [float(rows[date]["metrics"]["h_c_minus_h_s"]) for date in DATES]
    signs = {date: v1._sign(float(rows[date]["metrics"]["h_c_minus_h_s"])) for date in DATES}
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "scope": "receipt-only five-date held-source-date LODO aggregation; legacy receipt compatibility bridge; no recording/checkpoint/trainer/GPU access",
        "statistics_limit": "five-date descriptive mean/median and sign pattern only; not a confidence interval, significance test, route selector, or paper endpoint",
        "required_outer_dates": list(DATES),
        "all_five_date_receipts_present_and_validated": True,
        "per_date": rows,
        "summary": {
            "h_c_pooled_r2": v1._summary(hc_values),
            "h_s_pooled_r2": v1._summary(hs_values),
            "h_c_minus_h_s_pooled_r2": v1._summary(deltas),
            "h_c_minus_h_s_sign_pattern": {
                "by_date": signs,
                "positive_date_count": sum(value == "positive" for value in signs.values()),
                "negative_date_count": sum(value == "negative" for value in signs.values()),
                "zero_date_count": sum(value == "zero" for value in signs.values()),
                "all_five_dates_reported": True,
            },
        },
        "legacy_receipt_compatibility": {
            "mode": "in_memory_bridge_only",
            "source_receipts_schema": v1.EVALUATION_SCHEMA,
            "source_receipts_unchanged": True,
            "explicit_binding_field_missing_from_source_receipts": True,
            "bridge_field_added_to_in_memory_copy_only": "checkpoint_binding_completed_before_target_open",
            "equivalent_evidence_checked": [
                "paired fresh fixed epoch-49 checkpoint metadata",
                "zero target optimizer/backward steps",
                "immutable deployment model state",
                "first/only same-date terminal evaluation",
                "formal/minival/EvalAI scopes closed",
            ],
            "inputs": compatibility,
        },
        "route_prerequisite": {
            "prior_status": "MISSING_IMPLEMENTATION_FAIL_CLOSED",
            "status": v1.ROUTE_PREREQUISITE_STATUS,
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
    try:
        path, digest = v1._write_immutable(output_path, payload)
    except v1.FiveDateAggregateError as error:
        raise LegacyCompatibilityError(str(error)) from error
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, "aggregate output publication lost mode 0444")
    return {"status": STATUS, "receipt_path": str(path), "receipt_sha256": digest,
            "outer_dates": len(DATES), "legacy_compatibility": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--receipt-path", action="append", default=[], metavar="DATE=PATH")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    explicit: dict[str, Path] | None = None
    if args.receipt_path:
        explicit = {}
        for item in args.receipt_path:
            date, separator, raw_path = str(item).partition("=")
            _need(bool(separator) and date in DATES and bool(raw_path),
                  "--receipt-path requires DATE=PATH for a frozen outer date")
            _need(date not in explicit, f"duplicate --receipt-path date: {date}")
            explicit[date] = Path(raw_path)
    print(json.dumps(aggregate(evaluation_dir=args.evaluation_dir, output=args.output,
                               receipt_paths=explicit), sort_keys=True))


if __name__ == "__main__":
    main()
