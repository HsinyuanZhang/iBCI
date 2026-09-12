"""Read a sealed DANDI688 final-score receipt and write a 19-cell results report.

This module has no data loader, training entry point, network access, or CUDA use.  It
only validates receipt/seal metadata and prediction-file digests already written by
``finalize.score_final``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

import numpy as np

from .common import sha256
from .final_access import FA_CELLS, FINAL_SCORE_CELLS, REQUIRED_CELLS, SEAL_SCHEMA, SEAL_STATUS


FULL_SEEDS = (42, 43, 44)
MAIN_NEURAL = ("full_sua", "full_pmua", "activity_sua", "activity_pmua", "raw_set_sua", "raw_set_pmua")
PAIR_ROWS = (("Full", "full_sua", "full_pmua"), ("ACT", "activity_sua", "activity_pmua"),
             ("Raw-set", "raw_set_sua", "raw_set_pmua"), ("WF-FSS", "wf_fss_sua", "wf_fss_pmua"))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _metrics(row: Mapping[str, Any], roster: tuple[str, ...], cell: str) -> dict[str, float]:
    result = row.get("result")
    metrics = result.get("metrics") if isinstance(result, Mapping) else None
    if not isinstance(metrics, Mapping) or metrics.get("n_sessions") != len(roster):
        raise ValueError(f"{cell} lacks complete final metrics")
    sessions = metrics.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != len(roster):
        raise ValueError(f"{cell} lacks all six final dates")
    values: dict[str, float] = {}
    for item in sessions:
        if not isinstance(item, Mapping) or not isinstance(item.get("session_id"), str):
            raise ValueError(f"{cell} has malformed final-session metric")
        session_id = item["session_id"]
        if session_id not in roster or session_id in values:
            raise ValueError(f"{cell} final-session coverage differs from receipt roster")
        values[session_id] = _finite(item.get("r2"), f"{cell}.{session_id}.r2")
    if tuple(values) != roster:
        raise ValueError(f"{cell} final-session ordering/coverage differs from receipt roster")
    calculated = mean(values.values())
    reported = _finite(metrics.get("mean_r2"), f"{cell}.mean_r2")
    if not math.isclose(reported, calculated, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{cell}.mean_r2 differs from equal-session mean")
    return values


def _verify_predictions(row: Mapping[str, Any], roster: tuple[str, ...], cell: str, *, complete: bool) -> None:
    info = row.get("predictions")
    if info is None:
        if complete:
            raise ValueError(f"{cell} scored result lacks a prediction NPZ receipt")
        return
    if not isinstance(info, Mapping) or not isinstance(info.get("path"), str) or not isinstance(info.get("sha256"), str):
        raise ValueError(f"{cell} prediction receipt is malformed")
    path = Path(info["path"])
    if not path.is_file() or sha256(path) != info["sha256"]:
        raise ValueError(f"{cell} prediction NPZ SHA-256 mismatch")
    sessions = info.get("sessions")
    if not isinstance(sessions, Mapping) or not set(sessions).issubset(set(roster)) or (complete and set(sessions) != set(roster)):
        raise ValueError(f"{cell} prediction receipt has invalid final-date coverage")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(sessions) or (complete and set(archive.files) != set(roster)):
            raise ValueError(f"{cell} prediction NPZ has invalid final-date coverage")


def _unavailable_reason(row: Mapping[str, Any], roster: tuple[str, ...], cell: str) -> str:
    """Keep either a pre-score reason or every final-time per-session failure."""
    top_level = row.get("reason")
    if isinstance(top_level, str) and top_level:
        return top_level
    result = row.get("result")
    per_session = result.get("per_session") if isinstance(result, Mapping) else None
    if not isinstance(per_session, Mapping) or set(per_session) != set(roster):
        raise ValueError(f"{cell} unavailable result lacks complete per-session reasons")
    reasons: list[str] = []
    for session in roster:
        detail = per_session.get(session)
        if not isinstance(detail, Mapping):
            raise ValueError(f"{cell}.{session} unavailable result is malformed")
        status, reason = detail.get("status"), detail.get("reason")
        if status == "SCORED":
            if not isinstance(detail.get("prediction_sha256"), str) or not detail["prediction_sha256"]:
                raise ValueError(f"{cell}.{session} scored partial result lacks prediction SHA-256")
            reasons.append(f"{session}: SCORED (prediction retained)")
        elif status == "UNAVAILABLE" or status is None:
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"{cell}.{session} unavailable result lacks a reason")
            reasons.append(f"{session}: UNAVAILABLE ({reason})")
        else:
            raise ValueError(f"{cell}.{session} has invalid unavailable-row status: {status!r}")
    return "; ".join(reasons)


def _validate(receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], dict[str, dict[str, float] | None]]:
    receipt_path = Path(receipt_path).resolve()
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "FINAL_SCORED":
        raise ValueError("receipt status must be FINAL_SCORED")
    seal_path = Path(receipt.get("seal_path", ""))
    if not seal_path.is_file():
        raise FileNotFoundError(f"seal_path is not a file: {seal_path}")
    seal = _read_json(seal_path)
    if seal.get("schema") != SEAL_SCHEMA or seal.get("status") != SEAL_STATUS:
        raise ValueError("receipt does not reference a frozen formal seal")
    if receipt.get("seal_sha256") != sha256(seal_path):
        raise ValueError("receipt seal SHA-256 mismatch")
    roster_value = receipt.get("roster")
    if not isinstance(roster_value, list) or len(roster_value) != 6 or not all(isinstance(item, str) for item in roster_value):
        raise ValueError("receipt roster must contain exactly six final dates")
    roster = tuple(roster_value)
    if len(set(roster)) != len(roster):
        raise ValueError("receipt roster has duplicate final dates")
    expected = set(FINAL_SCORE_CELLS)
    results = receipt.get("results")
    if not isinstance(results, Mapping) or set(results) != expected or len(results) != len(FINAL_SCORE_CELLS):
        raise ValueError("receipt must contain exactly FINAL_SCORE_CELLS")
    selections = {**dict(seal.get("cell_selections", {})), **dict(seal.get("supplemental_full_selections", {}))}
    if set(selections) != expected:
        raise ValueError("seal does not contain exactly FINAL_SCORE_CELLS")
    values: dict[str, dict[str, float] | None] = {}
    for cell in FINAL_SCORE_CELLS:
        row = results[cell]
        if not isinstance(row, Mapping):
            raise ValueError(f"{cell} final result is malformed")
        status = row.get("status")
        if status == "UNAVAILABLE":
            if cell not in FA_CELLS:
                raise ValueError(f"{cell} is not an allowed unavailable FA result")
            row["_report_reason"] = _unavailable_reason(row, roster, cell)
            _verify_predictions(row, roster, cell, complete=False)
            values[cell] = None
            continue
        if status != "SCORED":
            raise ValueError(f"{cell} has invalid final status: {status!r}")
        values[cell] = _metrics(row, roster, cell)
        _verify_predictions(row, roster, cell, complete=True)
    for cell in MAIN_NEURAL + ("full_sua_s43", "full_pmua_s43", "full_sua_s44", "full_pmua_s44"):
        if values[cell] is None:
            raise ValueError(f"{cell} must have finite complete neural final metrics")
    return receipt, seal, roster, values


def _cell_mean(values: dict[str, float] | None) -> float | None:
    return None if values is None else mean(values.values())


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def build_report(receipt_path: Path, dest: Path) -> dict[str, Path]:
    """Validate a completed final receipt and write a fresh, human-readable report."""
    receipt, _seal, roster, values = _validate(receipt_path)
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"report destination must be fresh: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    results = receipt["results"]
    main_rows: list[list[Any]] = []
    for cell in REQUIRED_CELLS:
        row, scores = results[cell], values[cell]
        if scores is None:
            main_rows.append([cell, "UNAVAILABLE", "NA", *("NA" for _ in roster), row["_report_reason"]])
        else:
            main_rows.append([cell, "SCORED", f"{_cell_mean(scores):.12g}", *(f"{scores[day]:.12g}" for day in roster), ""])
    main_csv = dest / "main_results.csv"
    _write_csv(main_csv, ["cell", "status", "mean_r2", *roster, "reason"], main_rows)

    seed_rows: list[list[Any]] = []
    summary: dict[str, Any] = {"roster": list(roster), "full": {}, "matched_sua_minus_pmua": {}}
    for rep in ("sua", "pmua"):
        per_seed: dict[int, dict[str, float]] = {}
        for seed in FULL_SEEDS:
            cell = f"full_{rep}" if seed == 42 else f"full_{rep}_s{seed}"
            scores = values[cell]
            assert scores is not None
            per_seed[seed] = scores
            seed_rows.append([rep, "seed", seed, f"{_cell_mean(scores):.12g}", *(f"{scores[day]:.12g}" for day in roster), "", ""])
        seed_means = [_cell_mean(per_seed[seed]) for seed in FULL_SEEDS]
        assert all(item is not None for item in seed_means)
        rep_mean, rep_sd = mean(seed_means), stdev(seed_means)
        summary["full"][rep] = {"seed_scores": {str(seed): {"mean_r2": _cell_mean(per_seed[seed]), "per_session_r2": per_seed[seed]} for seed in FULL_SEEDS},
                                "three_seed_mean_r2": rep_mean, "three_seed_sample_sd_r2": rep_sd}
        seed_rows.append([rep, "three_seed_summary", "42,43,44", "", *("" for _ in roster), f"{rep_mean:.12g}", f"{rep_sd:.12g}"])
    differences: dict[int, dict[str, float]] = {}
    for seed in FULL_SEEDS:
        sua = values["full_sua" if seed == 42 else f"full_sua_s{seed}"]
        pmua = values["full_pmua" if seed == 42 else f"full_pmua_s{seed}"]
        assert sua is not None and pmua is not None
        differences[seed] = {day: sua[day] - pmua[day] for day in roster}
        seed_rows.append(["sua_minus_pmua", "matched_seed_difference", seed, f"{mean(differences[seed].values()):.12g}", *(f"{differences[seed][day]:.12g}" for day in roster), "", ""])
    diff_means = [mean(differences[seed].values()) for seed in FULL_SEEDS]
    summary["matched_sua_minus_pmua"] = {"seed_scores": {str(seed): {"mean_r2_difference": mean(differences[seed].values()), "per_session_r2_difference": differences[seed]} for seed in FULL_SEEDS},
                                           "three_seed_mean_r2_difference": mean(diff_means), "three_seed_sample_sd_r2_difference": stdev(diff_means)}
    seed_rows.append(["sua_minus_pmua", "three_seed_summary", "42,43,44", "", *("" for _ in roster), f"{mean(diff_means):.12g}", f"{stdev(diff_means):.12g}"])
    full_csv, full_json = dest / "full_seed_summary.csv", dest / "full_seed_summary.json"
    _write_csv(full_csv, ["representation", "row_type", "seed", "mean_r2", *roster, "three_seed_mean_r2", "three_seed_sample_sd_r2_ddof1"], seed_rows)
    full_json.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    pair_rows: list[list[Any]] = []
    for label, sua_cell, pmua_cell in PAIR_ROWS:
        sua, pmua = values[sua_cell], values[pmua_cell]
        if sua is None or pmua is None:
            continue
        for day in roster:
            pair_rows.append([label, day, f"{sua[day]:.12g}", f"{pmua[day]:.12g}", f"{sua[day] - pmua[day]:.12g}"])
        pair_rows.append([label, "equal_session_mean", f"{mean(sua.values()):.12g}", f"{mean(pmua.values()):.12g}", f"{mean(sua.values()) - mean(pmua.values()):.12g}"])
    pair_csv = dest / "main_sua_pmua_paired_differences.csv"
    _write_csv(pair_csv, ["comparison", "session", "sua_r2", "pmua_r2", "sua_minus_pmua_r2"], pair_rows)

    main_markdown = "\n".join("| " + " | ".join(map(str, row)) + " |" for row in main_rows)
    report = dest / "report.md"
    report.write_text(
        "# DANDI688 final results\n\n"
        f"Validated receipt: `{Path(receipt_path).resolve()}`. The frozen score matrix contains all {len(FINAL_SCORE_CELLS)} cells and six final dates.\n\n"
        "## Main 15 cells\n\n"
        "| cell | status | mean R² | " + " | ".join(roster) + " | reason |\n"
        "| --- | --- | ---: | " + " | ".join("---:" for _ in roster) + " | --- |\n" + main_markdown + "\n\n"
        f"Full seed details and ddof=1 seed SD: [{full_csv.name}]({full_csv.name}) and [{full_json.name}]({full_json.name}). "
        f"Main paired SUA−PMUA daily differences: [{pair_csv.name}]({pair_csv.name}).\n\n"
        "The main neural table uses seed42. Full seed SD is conditional on the representation-specific shared seed42 encoder; it measures decoder-seed variability, not encoder-seed variability. Full−ACT is not a pure causal label effect because the arms differ in encoder training/freeze structure. WF-FSS uses dense velocity labels, whereas Full uses trial-angle information, so their label budgets differ. FA rows marked UNAVAILABLE retain their reason and are not averaged over a reduced date set.\n",
        encoding="utf-8")
    return {"report": report, "main_results": main_csv, "full_seed_summary_csv": full_csv,
            "full_seed_summary_json": full_json, "paired_differences": pair_csv}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    args = parser.parse_args()
    outputs = build_report(args.receipt, args.dest)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
