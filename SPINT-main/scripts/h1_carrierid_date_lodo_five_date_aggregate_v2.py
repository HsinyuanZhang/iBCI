#!/usr/bin/env python3
"""Receipt-only descriptive completion for five H1 date-LODO H-S/H-C pairs.

This is an offline extension of the existing v1 five-date finalizer, not a
producer or evaluator.  It reuses v1's full immutable receipt validation and
adds the reporting units that v1 intentionally omitted: every fixed target
recording, every outer date, and a paired outer-date summary with a standard
error and deterministic bootstrap.  The five outer dates—not individual
recordings nested within them—are the resampling/SE unit.

No target NWB/data/checkpoint/config path stored in a receipt is dereferenced.
The program does not import a model/data module/GPU library, launch a process,
or select a route from the observed effect.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import stat
from typing import Any, Mapping, Sequence

try:  # Direct `python scripts/...` and package/test execution.
    from scripts import h1_carrierid_date_lodo_five_date_aggregate as v1
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    import h1_carrierid_date_lodo_five_date_aggregate as v1


DATES = v1.DATES
EVALUATION_DIR = v1.EVALUATION_DIR
SCHEMA = "h1_carrierid_date_lodo_five_date_heldout_descriptive_paired_aggregate_v2"
STATUS = "PASS_H1_CARRIERID_DATE_LODO_FIVE_DATE_DESCRIPTIVE_PAIRED_AGGREGATE_V2_NO_ROUTE_SELECTED"
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 202608081
DEFAULT_OUTPUT = (
    v1.ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase2/"
    "H1_CARRIERID_DATE_LODO_FIVE_DATE_HELDOUT_DESCRIPTIVE_PAIRED_AGGREGATE_v2.json"
)

# This roster is deliberately code-static rather than discovered from an NWB
# or directory.  It is the frozen five-date confirmatory grid that the source
# plan enumerated.  Requiring it prevents a receipt that jointly omitted a
# recording from looking complete merely because both arms omitted it.
EXPECTED_RECORDINGS_BY_DATE: dict[str, tuple[str, ...]] = {
    "19250108": ("ses-19250108T110520", "ses-19250108T111022", "ses-19250108T111455"),
    "19250113": ("ses-19250113T120811", "ses-19250113T121303"),
    "19250115": ("ses-19250115T110633", "ses-19250115T111328"),
    "19250119": ("ses-19250119T113543", "ses-19250119T114045"),
    "19250120": ("ses-19250120T115044", "ses-19250120T115537"),
}


class FiveDateDescriptiveAggregateError(RuntimeError):
    """A required immutable H1 receipt/grid/statistical invariant failed."""


@dataclass(frozen=True)
class InputTerminal:
    """A local immutable receipt and the canonical path it declares on creation."""

    outer_date: str
    input_path: Path
    input_sha256: str
    declared_canonical_path: Path
    terminal: v1.TerminalReceipt


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise FiveDateDescriptiveAggregateError(message)


def _finite(value: Any, label: str) -> float:
    _need(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def _sign(value: float) -> str:
    return "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_input_terminal(*, outer_date: str, path: str | Path) -> InputTerminal:
    """Read a known immutable receipt, allowing an offline read-only copy.

    The evaluator embeds its original canonical output path.  A byte-identical
    local import cannot live at that remote path, so v1 validation receives the
    declared path while this wrapper separately records the actual local input
    path and byte SHA.  The original canonical filename remains mandatory.
    """

    input_path = Path(path).resolve()
    expected_name = v1.canonical_evaluation_path(outer_date).name
    _need(input_path.is_file() and not input_path.is_symlink(), f"{outer_date}: immutable held-out receipt is missing")
    _need(stat.S_IMODE(input_path.stat().st_mode) == 0o444, f"{outer_date}: held-out receipt must be mode 0444")
    _need(input_path.name == expected_name, f"{outer_date}: receipt filename is not the frozen canonical terminal name")
    try:
        body = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FiveDateDescriptiveAggregateError(f"{outer_date}: held-out receipt JSON is unreadable") from error
    _need(isinstance(body, dict), f"{outer_date}: held-out receipt must be a JSON object")
    one_shot = body.get("one_shot")
    _need(isinstance(one_shot, Mapping), f"{outer_date}: one-shot receipt section is absent")
    declared = Path(str(one_shot.get("canonical_output_path", ""))).resolve()
    _need(declared.name == expected_name, f"{outer_date}: declared canonical terminal filename drift")
    digest = _sha256(input_path)
    terminal = v1.TerminalReceipt(outer_date=outer_date, path=declared, sha256=digest, body=body)
    return InputTerminal(outer_date=outer_date, input_path=input_path, input_sha256=digest,
                         declared_canonical_path=declared, terminal=terminal)


def _resolve_receipt_paths(*, evaluation_dir: str | Path, receipt_paths: Mapping[str, str | Path] | None) -> dict[str, Path]:
    if receipt_paths is None:
        directory = Path(evaluation_dir).resolve()
        return {date: v1.canonical_evaluation_path(date, evaluation_dir=directory) for date in DATES}
    _need(set(receipt_paths) == set(DATES), "explicit receipt-path grid must contain exactly the five frozen dates")
    return {date: Path(receipt_paths[date]).resolve() for date in DATES}


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    _need(bool(sorted_values) and 0.0 <= probability <= 1.0, "invalid bootstrap quantile input")
    position = (len(sorted_values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    fraction = position - low
    return float(sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction)


def _paired_summary(values: Sequence[float]) -> dict[str, Any]:
    numbers = [_finite(value, "paired H-C minus H-S delta") for value in values]
    _need(len(numbers) == len(DATES), "paired summary requires exactly five outer-date deltas")
    mean = sum(numbers) / len(numbers)
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1))
    signs = [_sign(value) for value in numbers]
    return {
        "mean": float(mean), "median": _quantile(sorted(numbers), 0.5),
        "sample_standard_deviation": float(sample_std),
        "paired_standard_error": float(sample_std / math.sqrt(len(numbers))),
        "sign_counts": {
            "positive": signs.count("positive"), "negative": signs.count("negative"),
            "zero": signs.count("zero"), "total": len(signs),
        },
    }


def _paired_bootstrap(values: Sequence[float]) -> dict[str, Any]:
    numbers = tuple(_finite(value, "bootstrap paired delta") for value in values)
    _need(len(numbers) == len(DATES), "paired bootstrap requires all five outer dates")
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(numbers)
    means = sorted(sum(numbers[rng.randrange(count)] for _ in range(count)) / count for _ in range(BOOTSTRAP_DRAWS))
    return {
        "resampling_unit": "outer_date (recordings remain nested; they are not treated as independent)",
        "draws": BOOTSTRAP_DRAWS, "rng_seed": BOOTSTRAP_SEED,
        "quantile_method": "linear_order_statistic",
        "lower_95": _quantile(means, 0.025), "upper_95": _quantile(means, 0.975),
    }


def _validate_expected_grid(date: str, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Add fixed recording-level pairing checks to v1's date receipt contract."""

    expected = EXPECTED_RECORDINGS_BY_DATE.get(date)
    _need(expected is not None, f"{date}: no frozen expected-recording grid")
    target = row.get("target")
    _need(isinstance(target, Mapping), f"{date}: v1 target receipt is absent")
    sessions = tuple(target.get("sessions", ()))
    _need(sessions == expected, f"{date}: target recording grid is incomplete/reordered/extra; expected {expected}, got {sessions}")
    files = target.get("files")
    _need(isinstance(files, Mapping) and tuple(files) == expected,
          f"{date}: recording file grid does not exactly match frozen target roster")

    metrics = row.get("metrics")
    _need(isinstance(metrics, Mapping), f"{date}: v1 metric rows are absent")
    h_s, h_c = metrics.get("h_s"), metrics.get("h_c")
    _need(isinstance(h_s, Mapping) and isinstance(h_c, Mapping), f"{date}: H-S/H-C metric rows absent")
    hs_sessions, hc_sessions = h_s.get("per_session"), h_c.get("per_session")
    _need(isinstance(hs_sessions, Mapping) and isinstance(hc_sessions, Mapping), f"{date}: per-recording metrics absent")
    _need(tuple(hs_sessions) == expected and tuple(hc_sessions) == expected,
          f"{date}: H-S/H-C recording grid mismatch")

    recordings: list[dict[str, Any]] = []
    for session in expected:
        hs, hc = hs_sessions[session], hc_sessions[session]
        _need(isinstance(hs, Mapping) and isinstance(hc, Mapping), f"{date}/{session}: paired metric row missing")
        hs_samples, hc_samples = hs.get("samples"), hc.get("samples")
        _need(isinstance(hs_samples, int) and isinstance(hc_samples, int) and hs_samples > 0 and hc_samples > 0,
              f"{date}/{session}: invalid paired sample count")
        _need(hs_samples == hc_samples, f"{date}/{session}: H-S/H-C per-recording sample count mismatch")
        hs_r2, hc_r2 = _finite(hs.get("r2"), f"{date}/{session}: H-S R2"), _finite(hc.get("r2"), f"{date}/{session}: H-C R2")
        delta = hc_r2 - hs_r2
        recordings.append({
            "outer_date": date, "recording": session, "target_file_sha256": files[session],
            "samples": int(hs_samples), "h_s_r2": hs_r2, "h_c_r2": hc_r2,
            "h_c_minus_h_s": delta, "sign": _sign(delta),
        })
    _need(sum(item["samples"] for item in recordings) == h_s.get("samples") == h_c.get("samples"),
          f"{date}: paired recording samples do not sum to date total")
    return recordings


def aggregate(*, evaluation_dir: str | Path = EVALUATION_DIR, output: str | Path = DEFAULT_OUTPUT,
              receipt_paths: Mapping[str, str | Path] | None = None) -> dict[str, Any]:
    """Publish once only after the complete immutable five-date, 11-recording grid validates."""

    output_path = Path(output).resolve()
    _need(not output_path.exists() and not output_path.is_symlink(), f"refusing to overwrite aggregate output: {output_path}")
    paths = _resolve_receipt_paths(evaluation_dir=evaluation_dir, receipt_paths=receipt_paths)
    try:
        inputs = {date: _read_input_terminal(outer_date=date, path=paths[date]) for date in DATES}
        per_date = {date: v1.validate_terminal_receipt(inputs[date].terminal) for date in DATES}
    except v1.FiveDateAggregateError as error:
        raise FiveDateDescriptiveAggregateError(str(error)) from error
    _need(tuple(per_date) == DATES, "complete five-date receipt matrix is required")

    recordings_by_date = {date: _validate_expected_grid(date, per_date[date]) for date in DATES}
    all_recordings = [record for date in DATES for record in recordings_by_date[date]]
    expected_count = sum(len(EXPECTED_RECORDINGS_BY_DATE[date]) for date in DATES)
    _need(len(all_recordings) == expected_count == 11, "complete 11-recording grid is required")
    keys = [(item["outer_date"], item["recording"]) for item in all_recordings]
    _need(len(set(keys)) == len(keys), "duplicate recording/date row in paired aggregate")

    date_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for date in DATES:
        metrics = per_date[date]["metrics"]
        delta = _finite(metrics["h_c_minus_h_s"], f"{date}: date delta")
        deltas.append(delta)
        date_rows.append({
            "outer_date": date,
            "input_receipt": {
                "path": str(inputs[date].input_path), "sha256": inputs[date].input_sha256,
                "declared_canonical_output_path": str(inputs[date].declared_canonical_path),
            },
            "receipt": per_date[date]["receipt"],
            "source_manifest_sha256": per_date[date]["source_manifest_sha256"],
            "samples": int(per_date[date]["target"]["samples"]),
            "recording_count": len(recordings_by_date[date]),
            "h_s_pooled_r2": _finite(metrics["h_s"]["pooled_r2"], f"{date}: H-S pooled R2"),
            "h_c_pooled_r2": _finite(metrics["h_c"]["pooled_r2"], f"{date}: H-C pooled R2"),
            "h_c_minus_h_s": delta, "sign": _sign(delta),
            "recordings": recordings_by_date[date],
        })

    paired = _paired_summary(deltas)
    paired["paired_outer_date_bootstrap_95"] = _paired_bootstrap(deltas)
    payload = {
        "schema": SCHEMA, "status": STATUS,
        "scope": "receipt-only immutable held-source-date H1 H-S/H-C aggregation; no NWB/checkpoint/config/trainer/GPU access",
        "statistics_limit": "descriptive pairing by five frozen outer dates; bootstrap and paired SE use outer date, not nested recordings; not a formal-heldout claim, route selector, or significance declaration",
        "required_outer_dates": list(DATES),
        "expected_recordings_by_date": {date: list(EXPECTED_RECORDINGS_BY_DATE[date]) for date in DATES},
        "all_five_date_receipts_present_and_validated": True,
        "all_eleven_recording_pairs_present_and_validated": True,
        "per_recording": all_recordings,
        "per_date": date_rows,
        "overall_paired_h_c_minus_h_s": paired,
        "aggregator_scope": {
            "nwb_opened_by_aggregator": False, "checkpoint_opened_by_aggregator": False,
            "config_opened_by_aggregator": False, "trainer_constructed_or_launched": False,
            "gpu_constructed_or_launched": False, "target_optimizer_steps": 0, "target_backward_steps": 0,
            "effect_used_to_select_checkpoint_or_date": False, "effect_used_to_select_or_launch_route": False,
        },
    }
    try:
        path, digest = v1._write_immutable(output_path, payload)
    except v1.FiveDateAggregateError as error:
        raise FiveDateDescriptiveAggregateError(str(error)) from error
    return {"status": STATUS, "receipt_path": str(path), "receipt_sha256": digest,
            "outer_dates": len(DATES), "recording_pairs": len(all_recordings)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--receipt-path", action="append", default=[], metavar="DATE=PATH",
                        help="repeat exactly five times for offline copies from distinct immutable roots")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    explicit: dict[str, Path] | None = None
    if args.receipt_path:
        explicit = {}
        for item in args.receipt_path:
            date, separator, raw_path = str(item).partition("=")
            _need(bool(separator) and date in DATES and bool(raw_path), "--receipt-path requires DATE=PATH for a frozen outer date")
            _need(date not in explicit, f"duplicate --receipt-path date: {date}")
            explicit[date] = Path(raw_path)
    print(json.dumps(aggregate(evaluation_dir=args.evaluation_dir, output=args.output, receipt_paths=explicit), sort_keys=True))


if __name__ == "__main__":
    main()
