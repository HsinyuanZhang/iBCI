#!/usr/bin/env python3
"""Describe an already-completed M2 carrier-support stability ledger.

This script is deliberately a post-hoc descriptive summarizer.  It never
loads NWB data, constructs a carrier, loads a decoder, or scores a target.
The ten M33 entries are repeated canonical anchors, not ten independent
resamples or model seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_REPORT_SHA256 = "573711d9b144b00387e3de07f60c6c89990f81d1681d2a550eb3e394c4443bac"
BUDGETS = (8, 16, 25, 33)
PLOT_BUDGETS = (8, 16, 25)
SEEDS = tuple(range(101, 111))
EXT4 = (
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2",
    "ses-2020-11-18-Run1", "ses-2020-11-19-Run1",
)
SOURCE = (
    "ses-2020-10-19-Run1", "ses-2020-10-19-Run2",
    "ses-2020-10-20-Run1", "ses-2020-10-20-Run2",
    "ses-2020-10-27-Run1", "ses-2020-10-27-Run2",
    "ses-2020-10-28-Run1",
)
METRICS = (
    "relative_frobenius_vs_m33", "cosine_vs_m33", "carrier_norm_frobenius",
    "per_unit_l2_mean", "carrier_seconds", "direction_design_condition",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{label}: expected finite number")
    return value


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def check_record(record: Mapping[str, Any], session: str) -> None:
    budget, seed = record.get("budget_trials"), record.get("seed")
    require(budget in BUDGETS and seed in SEEDS, f"{session}: invalid budget/seed")
    ids = record.get("trial_ids")
    require(isinstance(ids, list) and len(ids) == budget and len(set(ids)) == budget and
            all(isinstance(item, int) and 0 <= item < 33 for item in ids),
            f"{session}/M{budget}/s{seed}: invalid support ids")
    require(isinstance(record.get("valid_production_carrier"), bool),
            f"{session}/M{budget}/s{seed}: missing validity flag")
    require(isinstance(record.get("direction_design_rank"), int) and
            isinstance(record.get("direction_coverage_count"), int) and
            isinstance(record.get("finite_direction_trials"), int) and
            isinstance(record.get("nonfinite_direction_trials"), int),
            f"{session}/M{budget}/s{seed}: missing rank/coverage/trial counts")
    require(record["finite_direction_trials"] + record["nonfinite_direction_trials"] == budget,
            f"{session}/M{budget}/s{seed}: direction trial count drift")
    finite(record.get("carrier_seconds"), f"{session}/M{budget}/s{seed}/carrier_seconds")
    if record["valid_production_carrier"]:
        require(isinstance(record.get("carrier_sha256"), str) and len(record["carrier_sha256"]) == 64 and
                isinstance(record.get("raw_carrier_sha256"), str) and len(record["raw_carrier_sha256"]) == 64 and
                isinstance(record.get("vector_key"), str) and isinstance(record.get("raw_vector_key"), str),
                f"{session}/M{budget}/s{seed}: incomplete valid carrier receipt")
        for metric in METRICS:
            finite(record.get(metric), f"{session}/M{budget}/s{seed}/{metric}")
        require(record["direction_design_rank"] == 3,
                f"{session}/M{budget}/s{seed}: valid carrier has wrong rank")
    else:
        require(isinstance(record.get("failure"), str) and record["failure"],
                f"{session}/M{budget}/s{seed}: invalid carrier lacks reason")


def failure_counts(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    invalid = [row for row in records if not row["valid_production_carrier"]]
    rank = [row for row in invalid if "rank" in str(row.get("failure", "")).lower()]
    coverage = [row for row in invalid if "directional trial" in str(row.get("failure", "")).lower() or
                "needs >=3" in str(row.get("failure", "")).lower()]
    unclassified = [row for row in invalid if row not in rank and row not in coverage]
    return {
        "invalid_total": len(invalid),
        "reported_failure_reason_counts": {
            "rank_reason_count": len(rank),
            "coverage_reason_count": len(coverage),
            "unclassified_failure_count": len(unclassified),
        },
    }


def session_budget_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in records if row["valid_production_carrier"]]
    result: dict[str, Any] = {
        "n_records": len(records), "n_valid": len(valid), "n_invalid": len(records) - len(valid),
        "valid_fraction": len(valid) / len(records), "failure_counts": failure_counts(records),
        "invalid_records": [dict(row) for row in records if not row["valid_production_carrier"]],
    }
    if valid:
        result["median_relative_frobenius_vs_m33"] = statistics.median(
            finite(row["relative_frobenius_vs_m33"], "relative_frobenius") for row in valid)
        result["median_cosine_vs_m33"] = statistics.median(
            finite(row["cosine_vs_m33"], "cosine") for row in valid)
    else:
        result["median_relative_frobenius_vs_m33"] = None
        result["median_cosine_vs_m33"] = None
    return result


def surface_summary(session_records: Mapping[str, list[Mapping[str, Any]]], sessions: tuple[str, ...]) -> dict[str, Any]:
    per_budget: dict[str, Any] = {}
    for budget in BUDGETS:
        by_session = {session: session_budget_summary(session_records[session][budget]) for session in sessions}
        all_records = [row for session in sessions for row in session_records[session][budget]]
        medians_f = [by_session[session]["median_relative_frobenius_vs_m33"] for session in sessions]
        medians_c = [by_session[session]["median_cosine_vs_m33"] for session in sessions]
        complete = all(value is not None for value in medians_f + medians_c)
        per_budget[str(budget)] = {
            "per_session": by_session,
            "n_sessions_total": len(sessions),
            "n_sessions_with_valid": sum(by_session[session]["n_valid"] > 0 for session in sessions),
            "group_n_valid_of_total": f"{sum(row['valid_production_carrier'] for row in all_records)}/{len(all_records)}",
            "group_n_valid": sum(row["valid_production_carrier"] for row in all_records),
            "group_n_total": len(all_records),
            "group_equal_session_mean_of_session_medians": {
                "relative_frobenius_vs_m33": statistics.mean(medians_f) if complete else None,
                "cosine_vs_m33": statistics.mean(medians_c) if complete else None,
                "defined_only_when_every_session_has_a_valid_resample": complete,
                "n_sessions_total": len(sessions),
                "n_sessions_with_valid": sum(by_session[session]["n_valid"] > 0 for session in sessions),
            },
            "group_failure_counts": failure_counts(all_records),
        }
    return {"sessions": list(sessions), "per_budget": per_budget}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True,
                        help="new output directory for summary.json, figure_data.json, PNG, and PDF")
    args = parser.parse_args()
    source, dest = args.input.resolve(), args.dest.resolve()
    if dest.exists():
        raise FileExistsError(f"destination must be fresh: {dest}")
    require(source.is_file(), f"missing input: {source}")
    source_sha = sha(source)
    require(source_sha == EXPECTED_REPORT_SHA256, "input SHA-256 is not the frozen support-stability report")
    report = json.loads(source.read_text(encoding="utf-8"))
    require(isinstance(report, Mapping) and report.get("schema") == "m2_carrier_support_stability_v2" and
            report.get("status") == "COMPLETED" and report.get("no_decoder_or_target_bp") is True,
            "unexpected report identity or non-read-only scope")
    protocol = report.get("protocol", {})
    require(protocol.get("budgets") == list(BUDGETS) and protocol.get("seeds") == list(SEEDS) and
            protocol.get("M33") == 33, "protocol budget/seed/M33 drift")
    raw_sessions = report.get("sessions", {})
    expected_keys = {f"ext4/{session}" for session in EXT4} | {f"source_train/{session}" for session in SOURCE}
    require(isinstance(raw_sessions, Mapping) and set(raw_sessions) == expected_keys,
            "expected exactly seven source and four ext4 sessions")

    parsed: dict[str, dict[str, dict[int, list[Mapping[str, Any]]]]] = {"ext4": {}, "source_train": {}}
    for key, payload in raw_sessions.items():
        surface, session = key.split("/", 1)
        require(isinstance(payload, Mapping) and isinstance(payload.get("runs"), list), f"{key}: missing runs")
        require(payload.get("m33_direct_cached_byte_equal") is True, f"{key}: M33 cached/direct byte mismatch")
        rows = payload["runs"]
        require(len(rows) == 40, f"{key}: expected 40 budget/seed records")
        pairs = [(row.get("budget_trials"), row.get("seed")) for row in rows if isinstance(row, Mapping)]
        require(len(pairs) == len(rows) and len(set(pairs)) == 40 and set(pairs) == {(b, s) for b in BUDGETS for s in SEEDS},
                f"{key}: budget/seed records are not unique and complete")
        by_budget = {budget: [] for budget in BUDGETS}
        for row in rows:
            require(isinstance(row, Mapping), f"{key}: non-object run")
            check_record(row, key)
            if row["budget_trials"] == 33:
                require(row["trial_ids"] == list(range(33)) and row["valid_production_carrier"] is True and
                        abs(finite(row["relative_frobenius_vs_m33"], "M33 relative Frobenius")) <= 1e-12 and
                        abs(finite(row["cosine_vs_m33"], "M33 cosine") - 1.0) <= 2e-6,
                        f"{key}: repeated M33 anchor drift")
            by_budget[row["budget_trials"]].append(row)
        parsed[surface][session] = by_budget

    ext4_summary = surface_summary(parsed["ext4"], EXT4)
    source_summary = surface_summary(parsed["source_train"], SOURCE)
    figure_data = {
        "schema": "m2_carrier_support_stability_descriptive_figure_data_v1",
        "post_hoc_descriptive_only": True,
        "source_report": str(source), "source_report_sha256": source_sha,
        "helper_source": str(Path(__file__).resolve()), "helper_source_sha256": sha(Path(__file__).resolve()),
        "not_training_seeds": True, "no_confidence_intervals_or_p_values": True,
        "m33_anchor": "M33 rows repeat canonical [0..32] and are anchors, not ten independent resamples",
        "ext4": ext4_summary, "source_train": source_summary,
        "records": {surface: {session: {str(budget): [dict(row) for row in budgets[budget]]
                                            for budget in BUDGETS}
                               for session, budgets in sessions.items()}
                    for surface, sessions in parsed.items()},
    }
    summary = {
        "schema": "m2_carrier_support_stability_descriptive_summary_v1",
        "status": "COMPLETED", "post_hoc_descriptive_only": True,
        "source_report": str(source), "source_report_sha256": source_sha,
        "helper_source": str(Path(__file__).resolve()), "helper_source_sha256": sha(Path(__file__).resolve()),
        "scope": {"carrier_recomputed": False, "training_run": False, "decoder_loaded": False,
                  "decoder_scored": False, "nwb_opened": False, "evalai_opened": False},
        "interpretation": "No correlation, regression, causal claim, confidence interval, p-value, or training-seed claim is made.",
        "m33_anchor": "Repeated canonical M33 is retained for audit only and excluded from independent-resample interpretation.",
        "ext4_main": ext4_summary, "source_train_supplement": source_summary,
    }
    dest.mkdir(parents=True)
    (dest / "figure_data.json").write_text(json.dumps(figure_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 3.7), constrained_layout=True)
    for session in EXT4:
        values = [ext4_summary["per_budget"][str(budget)]["per_session"][session]["median_relative_frobenius_vs_m33"]
                  for budget in PLOT_BUDGETS]
        left.plot(PLOT_BUDGETS, values, marker="o", label=session.replace("ses-", ""))
    left.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    left.annotate("M33 canonical anchor = 0", xy=(25, 0), xytext=(8.2, 0.08), fontsize=8)
    left.set(xlabel="support trials (M)", ylabel="session median relative Frobenius vs M33",
             title="EXT4 session summaries")
    left.set_xticks(PLOT_BUDGETS); left.legend(fontsize=7, title="session", title_fontsize=7)
    fractions = []
    labels = []
    for budget in PLOT_BUDGETS:
        block = ext4_summary["per_budget"][str(budget)]
        fractions.append(block["group_n_valid"] / block["group_n_total"])
        labels.append(f"M{budget}\n{block['group_n_valid_of_total']}")
    bars = right.bar(range(len(PLOT_BUDGETS)), fractions, color="#4C78A8")
    for bar, budget in zip(bars, PLOT_BUDGETS):
        counts = ext4_summary["per_budget"][str(budget)]["group_failure_counts"]
        right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .025,
                   f"{counts['invalid_total']} invalid", ha="center", va="bottom", fontsize=8)
    right.set(ylim=(0, 1.17), xticks=range(len(PLOT_BUDGETS)), xticklabels=labels,
              xlabel="support trials / valid of 40", ylabel="valid carrier proportion",
              title="EXT4 valid resamples (not model seeds)")
    for path in (dest / "support_stability_ext4.png", dest / "support_stability_ext4.pdf"):
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"status": "COMPLETED", "dest": str(dest), "source_report_sha256": source_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
