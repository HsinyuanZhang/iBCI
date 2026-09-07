#!/usr/bin/env python3
"""Fail-closed aggregate for the M2 M33 support/query replay correction.

This deliberately aggregates only four sessions eligible after the fixed M33
support prefix.  It never reads historical M33 metric CSVs as score inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


GROUPS = ("f0", "t4", "ts4")
CELLS = ("fold1_seed42", "fold1_seed43", "fold2_seed42")
CELL_META = {"fold1_seed42": (1, 42), "fold1_seed43": (1, 43), "fold2_seed42": (2, 42)}
INELIGIBLE = {"ses-2020-11-24-Run1", "ses-2020-11-24-Run2"}
M, WINDOW = 33, 50
SCREEN = "m2_m33_disjoint_replay_correction_v1"
PROTOCOL_SHA = "da4eb05db43401f2e314351e8882c6606cc4d020aed3ed45e860d52db9e59df1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eq(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(f"{name}: expected {expected!r}, found {observed!r}")


def read_scores(path: Path, expected_sessions: set[str]) -> dict[str, float]:
    with (path / "metrics_per_session.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "test_heldout"]
    if {row.get("session") for row in rows} != expected_sessions:
        raise ValueError(f"{path}: metrics do not contain exactly the four eligible sessions")
    out = {}
    for row in rows:
        eq(int(float(row["M"])), M, f"{path}/{row['session']} M")
        score = float(row["R2_variance_weighted"])
        if not math.isfinite(score):
            raise ValueError(f"{path}/{row['session']}: nonfinite R2")
        out[row["session"]] = score
    return out


def load_arm(protocol: dict, seal_list: dict, group: str, cell: str, eligible: set[str]) -> dict:
    fold, seed = CELL_META[cell]
    sealed = seal_list["sealed_artifacts"][group][cell]
    path = Path(sealed["path"]).resolve()
    if not path.is_dir():
        raise ValueError(f"{group}/{cell}: explicitly sealed artifact directory is missing")
    prov_path = path / "m33_disjoint_replay_correction_provenance.json"
    split_path = path / "split_manifest.json"
    if not prov_path.is_file() or not split_path.is_file():
        raise ValueError(f"{path}: missing correction provenance or split audit")
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    eq(sha256(prov_path), sealed["provenance_sha256"], f"{group}/{cell} explicit seal SHA")
    eq(provenance.get("protocol", {}).get("sha256"), PROTOCOL_SHA, f"{group}/{cell} protocol SHA")
    eq(provenance.get("group"), group, f"{group}/{cell} provenance group")
    eq(provenance.get("cell", {}).get("name"), cell, f"{group}/{cell} provenance cell")
    eq(provenance.get("uses_cpu_only"), True, f"{group}/{cell} CPU-only")
    eq(provenance.get("test_only"), True, f"{group}/{cell} test-only")
    eq(provenance.get("no_backward_optimizer_or_checkpoint_selection_on_heldout"), True, f"{group}/{cell} no-selection")
    frozen = protocol["frozen_source_arms"][group][cell]["checkpoint"]
    eq(provenance.get("frozen_source_checkpoint"), frozen, f"{group}/{cell} checkpoint binding")
    audit = provenance.get("query_window_audit", {})
    if set(audit) != set(protocol["session_eligibility"]["all_historical_heldout_sessions"]):
        raise ValueError(f"{group}/{cell}: audit does not cover historical six sessions")
    for session in INELIGIBLE:
        row = audit[session]
        eq(row.get("total_trials"), M, f"{group}/{cell}/{session} trials")
        eq(row.get("query_trials"), 0, f"{group}/{cell}/{session} zero query")
        eq(row.get("eligible_windows"), 0, f"{group}/{cell}/{session} zero windows")
        eq(row.get("ineligible_reason"), "zero_query_trials_after_chronological_support", f"{group}/{cell}/{session} reason")
    for session in eligible:
        row = audit[session]
        eq(row.get("support_trials"), M, f"{group}/{cell}/{session} support")
        eq(row.get("query_start_trial"), M, f"{group}/{cell}/{session} query start")
        eq(row.get("window_size"), WINDOW, f"{group}/{cell}/{session} window")
        eq(row.get("full_window_disjoint"), True, f"{group}/{cell}/{session} disjoint")
        if row.get("minimum_window_start_padded_bin") != row.get("raw_query_start_bin") + WINDOW - 1:
            raise ValueError(f"{group}/{cell}/{session}: query window includes support history")
    scores = read_scores(path, eligible)
    eq(provenance.get("eligible_scores"), scores, f"{group}/{cell} score receipt")
    return {"path": str(path), "scores": scores, "split": json.loads(split_path.read_text(encoding="utf-8"))}


def norm_payload(split: dict) -> dict:
    norm = split.get("native_t4_normalization")
    if not isinstance(norm, dict):
        raise ValueError("T4/TS4 normalization missing")
    return {key: norm[key] for key in ("mean", "std", "train_sessions")}


def report_delta(left: dict, right: dict, eligible: list[str]) -> dict:
    per_cell_per_session = {
        cell: {session: left[cell]["scores"][session] - right[cell]["scores"][session] for session in eligible}
        for cell in CELLS
    }
    per_cell_mean = {cell: float(np.mean(list(per_cell_per_session[cell].values()))) for cell in CELLS}
    per_session_cluster = {
        session: float(np.mean([per_cell_per_session[cell][session] for cell in CELLS])) for session in eligible
    }
    clusters = np.asarray([per_session_cluster[session] for session in eligible], dtype=float)
    rng = np.random.default_rng(20260801)
    boot = np.mean(rng.choice(clusters, size=(100000, len(clusters)), replace=True), axis=1)
    nonzero = bool(np.all(clusters != 0.0))
    exact_wilcoxon = float(wilcoxon(clusters, alternative="two-sided", method="exact").pvalue) if nonzero else None
    return {
        "per_cell_per_session": per_cell_per_session,
        "per_cell_equal_session_mean": per_cell_mean,
        "equal_cell_mean_delta_r2": float(np.mean(list(per_cell_mean.values()))),
        "four_session_cluster_means": per_session_cluster,
        "four_session_cluster_uncertainty": {
            "n_clusters": len(clusters),
            "bootstrap_seed": 20260801,
            "bootstrap_resamples": 100000,
            "bootstrap_percentile_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "cluster_sd": float(np.std(clusters, ddof=1)),
            "positive_clusters": int(np.sum(clusters > 0)),
            "two_sided_exact_sign_p": float(binomtest(int(np.sum(clusters > 0)), len(clusters), 0.5).pvalue),
            "two_sided_exact_wilcoxon_p": exact_wilcoxon,
            "wilcoxon_undefined_if_zero_cluster_delta": not nonzero,
            "minimum_attainable_two_sided_exact_wilcoxon_p_at_n4": 0.125,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seal-list", type=Path, required=True, help="immutable explicit path/SHA map of the nine sealed replays")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = root / f"sua_exploration/results/{SCREEN}/protocol_receipt.json"
    eq(sha256(protocol_path), PROTOCOL_SHA, "immutable protocol receipt SHA")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seal_list_path = args.seal_list.resolve()
    seal_list = json.loads(seal_list_path.read_text(encoding="utf-8"))
    eq(seal_list.get("protocol_sha256"), PROTOCOL_SHA, "seal list protocol SHA")
    if set(seal_list.get("sealed_artifacts", {})) != set(GROUPS):
        raise ValueError("seal list must name exactly F0/T4/TS4")
    for group in GROUPS:
        if set(seal_list["sealed_artifacts"].get(group, {})) != set(CELLS):
            raise ValueError(f"seal list must name exactly three cells for {group}")
    eligible = protocol["session_eligibility"]["eligible_sessions"]
    if len(eligible) != 4:
        raise ValueError("this correction is defined only for exactly four eligible sessions")
    arms = {group: {cell: load_arm(protocol, seal_list, group, cell, set(eligible)) for cell in CELLS} for group in GROUPS}
    for cell in CELLS:
        if norm_payload(arms["t4"][cell]["split"]) != norm_payload(arms["ts4"][cell]["split"]):
            raise ValueError(f"{cell}: T4/TS4 train normalization parity failed")
    comparisons = {
        "T4_minus_F0": report_delta(arms["t4"], arms["f0"], eligible),
        "T4_minus_TS4": report_delta(arms["t4"], arms["ts4"], eligible),
    }
    arm_summary = {
        group: {
            "per_cell_equal_session_mean_r2": {
                cell: float(np.mean(list(arms[group][cell]["scores"].values()))) for cell in CELLS
            },
        }
        for group in GROUPS
    }
    for summary in arm_summary.values():
        summary["equal_cell_mean_r2"] = float(np.mean(list(summary["per_cell_equal_session_mean_r2"].values())))
    payload = {
        "schema_version": 1,
        "purpose": "M2_M33_local_heldout_support_query_contamination_correction_aggregate",
        "name": "M33-eligible four-session subset",
        "protocol": {"path": str(protocol_path), "sha256": PROTOCOL_SHA},
        "explicit_seal_list": {"path": str(seal_list_path), "sha256": sha256(seal_list_path)},
        "historical_scores_not_used": True,
        "ineligible_zero_query_sessions": sorted(INELIGIBLE),
        "eligible_sessions": eligible,
        "aggregation": "equal session within cell, then equal cell across the three frozen cells",
        "arms": arm_summary,
        "paired_deltas_r2": comparisons,
        "significance_disclosure": "n=4 session clusters; exact two-sided Wilcoxon minimum p=0.125. This result cannot pass the withdrawn six-session p<=0.05 gate.",
        "label_information_disclosure": protocol["label_information"],
        "scope": "test-only CPU correction replay over four eligible local held-out-calibration sessions; not a full six-session M33 result or a hidden EvalAI result.",
    }
    out = args.out or root / f"sua_exploration/results/{SCREEN}/aggregate_heldout.json"
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {out}")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
