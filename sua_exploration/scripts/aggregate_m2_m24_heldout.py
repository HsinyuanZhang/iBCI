#!/usr/bin/env python3
"""Strict aggregate for M2 M24 chronological-disjoint, test-only held-out replays."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml
from scipy.stats import wilcoxon


GROUPS = {"f0": ("B3", "none"), "t4": ("B3S", "t4"), "k4": ("B3S", "k4"), "ks4": ("B3S", "ks4")}
SOURCE_SCREEN = "m2_m24_disjoint_source_v1"
SCREEN = "m2_m24_disjoint_heldout_v1"
EXPECTED_HELDOUT = {
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
}
M, WINDOW, FOLD, SEED = 24, 50, 1, 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def one(root: Path, pattern: str) -> Path:
    hits = sorted(path for path in root.glob(pattern) if path.is_dir())
    if len(hits) != 1:
        raise ValueError(f"expected exactly one {pattern}, found {hits}")
    return hits[0]


def eq(observed: object, expected: object, what: str) -> None:
    if observed != expected:
        raise ValueError(f"{what}: expected {expected!r}, found {observed!r}")


def read_scores(path: Path) -> dict[str, float]:
    rows = []
    with (path / "metrics_per_session.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "test_heldout":
                rows.append(row)
    observed = {row.get("session", "") for row in rows}
    if observed != EXPECTED_HELDOUT or len(rows) != len(EXPECTED_HELDOUT):
        raise ValueError(f"{path}: test_heldout metrics must contain exactly six expected sessions, found {sorted(observed)}")
    scores: dict[str, float] = {}
    for row in rows:
        eq(int(float(row.get("M", "nan"))), M, f"{path}/{row['session']} M")
        score = float(row["R2_variance_weighted"])
        if not math.isfinite(score):
            raise ValueError(f"{path}/{row['session']}: non-finite held-out R2")
        scores[row["session"]] = score
    return scores


def read_arm(root: Path, group: str, source: dict) -> dict:
    output_root = root / "streaming_calibration_exp/outputs/streaming_calibration"
    path = one(output_root, f"{SCREEN}_{group}_m2_f1_s42_*")
    required = ["resolved_config.yaml", "split_manifest.json", "checkpoint_manifest.json", "heldout_m24_provenance.json", "metrics_per_session.csv"]
    if missing := [name for name in required if not (path / name).is_file()]:
        raise ValueError(f"{path}: missing {missing}")
    cfg = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    data, model = cfg["data"], cfg["model"]
    variant, side = GROUPS[group]
    expected = {
        "data.task": "m2", "data.loso_fold": FOLD, "seed": SEED,
        "model.variant": variant, "data.side_feature_group": side,
        "data.calibration_n_trials": M, "data.random_calibration": False,
        "data.include_heldout_in_fit": False, "data.include_heldout_in_test": True,
        "data.query_start_trial": M, "train": False, "test": True,
    }
    observed = {
        "data.task": data.get("task"), "data.loso_fold": data.get("loso_fold"), "seed": cfg.get("seed"),
        "model.variant": model.get("variant"), "data.side_feature_group": data.get("side_feature_group"),
        "data.calibration_n_trials": data.get("calibration_n_trials"), "data.random_calibration": data.get("random_calibration"),
        "data.include_heldout_in_fit": data.get("include_heldout_in_fit"), "data.include_heldout_in_test": data.get("include_heldout_in_test"),
        "data.query_start_trial": data.get("query_start_trial"), "train": cfg.get("train"), "test": cfg.get("test"),
    }
    if bad := {key: {"expected": value, "observed": observed[key]} for key, value in expected.items() if observed[key] != value}:
        raise ValueError(f"{path}: test-only config contract mismatch {bad}")
    source_path = Path(source["path"]).resolve()
    if not source_path.is_file() or sha256(source_path) != source["sha256"]:
        raise ValueError(f"{group}: source checkpoint receipt does not verify at aggregation")
    eq(str(Path(str(cfg.get("ckpt_path"))).resolve()), str(source_path), f"{group} ckpt_path")
    source_artifact = source_path.parents[1]
    source_config_path = source_artifact / "resolved_config.yaml"
    if not source_config_path.is_file():
        raise ValueError(f"{group}: frozen source artifact lacks resolved config")
    source_cfg = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source_data, source_model = source_cfg["data"], source_cfg["model"]
    expected_data = copy.deepcopy(source_data)
    expected_data.update({"include_heldout_in_fit": False, "include_heldout_in_test": True, "query_start_trial": M})
    if data != expected_data:
        raise ValueError(f"{group}: held-out data mapping drifts from frozen source beyond the two permitted query overrides")
    if model != source_model:
        raise ValueError(f"{group}: held-out model mapping drifts from frozen source")
    checkpoint = json.loads((path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if checkpoint.get("source_checkpoint_sha256") != source["sha256"] or checkpoint.get("artifact_checkpoint_sha256") != source["sha256"]:
        raise ValueError(f"{group}: test-only checkpoint differs from frozen source")

    split = json.loads((path / "split_manifest.json").read_text(encoding="utf-8"))
    eq(split.get("heldout_evaluated_in_fit"), False, f"{group} heldout fit")
    eq(split.get("heldout_evaluated_in_test"), True, f"{group} heldout test")
    eq(split.get("query_start_trial"), M, f"{group} split query start")
    audit = split.get("heldout_query_window_audit")
    if not isinstance(audit, dict) or set(audit) != EXPECTED_HELDOUT:
        raise ValueError(f"{group}: missing exact held-out full-window audit")
    for session, row in audit.items():
        for key, value in {"support_trials": M, "query_start_trial": M, "window_size": WINDOW, "full_window_disjoint": True}.items():
            eq(row.get(key), value, f"{group}/{session} {key}")
        if int(row.get("query_trials", 0)) <= 0 or int(row.get("eligible_windows", 0)) <= 0:
            raise ValueError(f"{group}/{session}: empty eligible query")
        if row.get("minimum_window_start_padded_bin") != row.get("raw_query_start_bin", -WINDOW) + WINDOW - 1:
            raise ValueError(f"{group}/{session}: full temporal history overlaps support")
    if group in {"k4", "ks4"}:
        eq(split.get("k4_estimator", {}).get("calibration_trials"), M, f"{group} K4 M")
        eq(split.get("native_k4_normalization", {}).get("feature_group"), group, f"{group} normalization")
        k4_audit = split.get("heldout_k4_calibration_audit")
        if not isinstance(k4_audit, dict) or set(k4_audit) != EXPECTED_HELDOUT:
            raise ValueError(f"{group}: missing exact held-out K4 calibration audit")
        for session, row in k4_audit.items():
            if row.get("calibration_trials") != M or row.get("design_rank") != 3 or row.get("active_blocks", 0) <= 2:
                raise ValueError(f"{group}/{session}: invalid K4 fit audit")
            if not math.isfinite(float(row.get("design_condition", float("inf")))):
                raise ValueError(f"{group}/{session}: invalid K4 design condition")
            for key, value in {"raw_bin_ms": 20, "block_width_bins": 5, "behavior_lead_bins": 2, "max_trial_length_used": False}.items():
                eq(row.get(key), value, f"{group}/{session} {key}")
    if group == "t4":
        eq(split.get("native_t4_normalization", {}).get("feature_group"), "t4", "T4 normalization")

    provenance = json.loads((path / "heldout_m24_provenance.json").read_text(encoding="utf-8"))
    eq(provenance.get("group"), group, f"{group} provenance group")
    eq(provenance.get("test_only"), True, f"{group} provenance test_only")
    eq(provenance.get("hidden_evalai_evaluated"), False, f"{group} hidden EvalAI disclosure")
    eq(provenance.get("frozen_source_checkpoint", {}).get("sha256"), source["sha256"], f"{group} provenance source SHA")
    if set(provenance.get("heldout_sessions", [])) != EXPECTED_HELDOUT:
        raise ValueError(f"{group}: provenance held-out session drift")
    query_window_layout = {
        session: {key: audit[session][key] for key in ("eligible_windows", "raw_query_start_bin", "minimum_window_start_padded_bin", "query_trials")}
        for session in sorted(audit)
    }
    return {"path": str(path.resolve()), "scores": read_scores(path), "query_window_layout": query_window_layout}


def delta_report(left: dict[str, float], right: dict[str, float]) -> dict:
    deltas = {session: left[session] - right[session] for session in sorted(EXPECTED_HELDOUT)}
    values = list(deltas.values())
    if any(value == 0.0 for value in values):
        raise ValueError("exact Wilcoxon endpoint is undefined under zero paired deltas; do not silently switch methods")
    test = wilcoxon(values, alternative="two-sided", method="exact")
    return {
        "per_session": deltas, "mean": sum(values) / len(values),
        "positive_sessions": sum(value > 0 for value in values), "n_sessions": len(values),
        "wilcoxon_two_sided_exact_p": float(test.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    source_path = root / "sua_exploration/results" / SOURCE_SCREEN / "aggregate_internal.json"
    source_record = json.loads(source_path.read_text(encoding="utf-8"))
    if source_record.get("purpose") != "M2_M24_heldin_source_gate_for_disjoint_local_heldout_replay":
        raise ValueError("source internal aggregate has an unexpected purpose")
    if source_record.get("formal_heldout_evaluated") is not False or not source_record.get("gate", {}).get("all_three_pass"):
        raise ValueError("held-out aggregate requires a passed held-in-only source gate")
    cell = source_record.get("cell", {})
    eq((cell.get("task"), cell.get("fold"), cell.get("seed"), cell.get("M")), ("m2", FOLD, SEED, M), "source internal cell")
    source_arms = source_record.get("arms", {})
    if set(source_arms) != set(GROUPS):
        raise ValueError("source aggregate must retain exactly the four frozen arms")
    arms = {group: read_arm(root, group, source_arms[group]["checkpoint"]) for group in GROUPS}
    reference_layout = arms["f0"]["query_window_layout"]
    for group in ("t4", "k4", "ks4"):
        if arms[group]["query_window_layout"] != reference_layout:
            raise ValueError(f"{group}: held-out query-window layout drifts from F0; comparison is not paired")
    deltas = {
        "K4_minus_T4": delta_report(arms["k4"]["scores"], arms["t4"]["scores"]),
        "K4_minus_KS4": delta_report(arms["k4"]["scores"], arms["ks4"]["scores"]),
        "K4_minus_F0": delta_report(arms["k4"]["scores"], arms["f0"]["scores"]),
    }
    for arm in arms.values():
        arm["mean_r2"] = sum(arm.pop("scores").values()) / len(EXPECTED_HELDOUT)
    effect_size_pass = {name: report["mean"] >= 0.03 for name, report in deltas.items()}
    strong_consistency_pass = {
        name: report["positive_sessions"] == report["n_sessions"] and report["wilcoxon_two_sided_exact_p"] <= 0.05
        for name, report in deltas.items()
    }
    payload = {
        "schema_version": 1,
        "purpose": "M2_M24_local_heldout_test_only_chronological_disjoint_comparison",
        "formal_heldout_evaluated": True,
        "hidden_evalai_evaluated": False,
        "source_internal_gate": {"path": str(source_path.resolve()), "sha256": sha256(source_path)},
        "cell": {"task": "m2", "fold": FOLD, "seed": SEED, "calibration_trials": M, "window_size": WINDOW},
        "arms": arms, "paired_deltas_r2": deltas,
        "gate": {
            "effect_size_threshold": 0.03,
            "effect_size_pass": effect_size_pass,
            "strong_consistency_pass": strong_consistency_pass,
            "all_three_effect_size_pass": all(effect_size_pass.values()),
            "all_three_strong_consistency_pass": all(strong_consistency_pass.values()),
            "primary_effective": all(effect_size_pass.values()) and all(strong_consistency_pass.values()),
            "rule": "primary effective requires all session-mean deltas >= +0.03 R2 and, for each comparison, 6/6 positive session deltas with exact two-sided Wilcoxon p<=.05",
        },
        "label_information_disclosure": "All arms use 24 chronological support trials; T4 consumes target-direction labels while K4/KS4 consume continuous finger velocity, so K4 is not label-information matched to T4.",
        "scope": "local held-out-calibration-file chronological support/query replay; sessions were visible during development, not a hidden EvalAI/challenge test",
    }
    out = args.out or root / "sua_exploration/results" / SCREEN / "aggregate_heldout.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite aggregate {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"means": {name: report["mean"] for name, report in deltas.items()}, "primary_effective": payload["gate"]["primary_effective"]}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
