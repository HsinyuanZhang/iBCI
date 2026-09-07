#!/usr/bin/env python3
"""Fail-closed aggregate for the four held-in-only M2 M24 source arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


GROUPS = {"f0": ("B3", "none"), "t4": ("B3S", "t4"), "k4": ("B3S", "k4"), "ks4": ("B3S", "ks4")}
SCREEN = "m2_m24_disjoint_source_v1"
RECEIPT_SHA = "84af6055db3840b7f68bfeb74b61f9d028311343211eadba67b6d5a33345bdd3"
CORRECTION_SHA = "c7a038da879621d9352199b361dd1ebd980a9ca76e79855a61041a2e38ed15e2"
LEGACY_V1_SHA = "fb87b067ff8bf5b3d92ea755b5e4987e7c5041a700b52db4dc1e99a2d1a1fc68"
M, FOLD, SEED = 24, 1, 42
# Hydra omits this constructor-default field from the resolved F0/T4 configs;
# K4 declares the identical value explicitly because its fail-closed contract
# names it.  Compare the runtime-effective value, not YAML key presence.
DATA_CONSTRUCTOR_DEFAULTS = {"remove_calib_still_times": False}


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


def effective_data_value(data: dict, key: str):
    return data.get(key, DATA_CONSTRUCTOR_DEFAULTS.get(key))


def score(path: Path) -> float:
    for row in csv.DictReader((path / "metrics_summary.csv").open(encoding="utf-8")):
        if row.get("split") == "test_heldin":
            if int(float(row["M"])) != M:
                raise ValueError(f"{path}: held-in metric has wrong M")
            return float(row["R2_variance_weighted"])
    raise ValueError(f"{path}: missing test_heldin metric")


def read(path: Path, group: str) -> dict:
    required = ["resolved_config.yaml", "split_manifest.json", "run_metadata.json", "metrics_summary.csv", "checkpoint_manifest.json"]
    if missing := [name for name in required if not (path / name).is_file()]:
        raise ValueError(f"{path}: missing {missing}")
    config = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    split = json.loads((path / "split_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    data, model, trainer = config["data"], config["model"], config["trainer"]
    variant, side = GROUPS[group]
    expected = {
        "task": "m2", "fold": FOLD, "seed": SEED, "variant": variant, "side": side,
        "M": M, "random": False, "fit_heldout": False, "test_heldout": False,
        "query_start_trial": 0, "protocol": "loso", "epochs": 12,
    }
    observed = {
        "task": data.get("task"), "fold": data.get("loso_fold"), "seed": config.get("seed"),
        "variant": model.get("variant"), "side": data.get("side_feature_group"),
        "M": data.get("calibration_n_trials"), "random": data.get("random_calibration"),
        "fit_heldout": data.get("include_heldout_in_fit"), "test_heldout": data.get("include_heldout_in_test"),
        "query_start_trial": data.get("query_start_trial"), "protocol": data.get("validation_protocol"),
        "epochs": trainer.get("max_epochs"),
    }
    if bad := {key: {"expected": value, "observed": observed[key]} for key, value in expected.items() if observed[key] != value}:
        raise ValueError(f"{path}: config contract mismatch {bad}")
    if metadata.get("fold_id") != FOLD or metadata.get("seed") != SEED:
        raise ValueError(f"{path}: metadata fold/seed mismatch")
    if split.get("heldout_evaluated_in_fit") is not False or split.get("heldout_evaluated_in_test") is not False:
        raise ValueError(f"{path}: internal source accessed held-out")
    if split.get("query_start_trial") != 0:
        raise ValueError(f"{path}: internal source must not query-exclude minival")
    if group in {"k4", "ks4"}:
        estimator = split.get("k4_estimator", {})
        normalization = split.get("native_k4_normalization", {})
        if estimator.get("calibration_trials") != M or normalization.get("feature_group") != group:
            raise ValueError(f"{path}: invalid M24 K4 estimator/normalization receipt")
    if group == "t4" and split.get("native_t4_normalization", {}).get("feature_group") != "t4":
        raise ValueError(f"{path}: missing T4 train-only normalization receipt")
    checkpoint = json.loads((path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    ckpt_path = Path(checkpoint.get("artifact_checkpoint_path", ""))
    if not ckpt_path.is_file() or checkpoint.get("artifact_checkpoint_sha256") != sha256(ckpt_path):
        raise ValueError(f"{path}: source best checkpoint receipt mismatch")
    return {"path": str(path.resolve()), "score": score(path), "split": split, "config": config,
            "checkpoint": {"path": str(ckpt_path.resolve()), "sha256": sha256(ckpt_path)}}


def validate_launch_binding(root: Path, group: str) -> dict:
    """F0/T4 began before v2 existed; K4/KS4 must begin after it exists."""
    logs = sorted((root / "sua_exploration/results" / SCREEN / "logs").glob(f"{group}_f1_s42_gpu*.log"))
    if len(logs) != 1:
        raise ValueError(f"{group}: expected exactly one source-launch log, found {logs}")
    expected = LEGACY_V1_SHA if group in {"f0", "t4"} else RECEIPT_SHA
    text = logs[0].read_text(encoding="utf-8", errors="replace")
    if f"qualification_receipt_sha256={expected}" not in text:
        raise ValueError(f"{group}: launch log is not bound to its required qualification receipt")
    return {"launch_log": str(logs[0].resolve()), "receipt_sha256_at_launch": expected,
            "correction_receipt_sha256": CORRECTION_SHA,
            "interpretation": (
                "v1 had stale top-level M20 metadata; the immutable correction receipt verifies all seven M24 numeric rows "
                "are identical to v2 after all20_* -> all_support_* key mapping."
            )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = root / "sua_exploration/results/general_carrier_proxy_v1/audit_m2_m24_heldin_v2.json"
    correction = root / "sua_exploration/results/general_carrier_proxy_v1/audit_m2_m24_correction_receipt_v1.json"
    if sha256(receipt) != RECEIPT_SHA:
        raise ValueError("M24 qualification receipt SHA drift")
    if sha256(correction) != CORRECTION_SHA:
        raise ValueError("M24 qualification correction receipt SHA drift")
    record = json.loads(receipt.read_text(encoding="utf-8"))
    summary = record.get("summary", {})
    if record.get("protocol", {}).get("name") != "m24" or record.get("protocol", {}).get("fit_trials") != "0:12" or record.get("protocol", {}).get("prediction_trials") != "12:24" or record.get("formal_heldout_evaluated") is not False:
        raise ValueError("M24 receipt scope/protocol mismatch")
    if not (summary.get("all_7_sessions_beat_all_100_w_only_nulls") and summary.get("kreg_beats_baseline_sessions") == 7 and summary.get("kreg_beats_w_only_median_sessions") == 7 and summary.get("mean_kreg_over_baseline_ratio", 1.0) <= .95 and summary.get("mean_kreg_over_w_only_null_median_ratio", 1.0) <= .95 and min(summary.get("W_A_B_correlations", [0.0])) > .5 and summary.get("wilcoxon_two_sided_kreg_vs_baseline_mse", 1.0) <= .05 and summary.get("wilcoxon_two_sided_kreg_vs_w_only_median_mse", 1.0) <= .05):
        raise ValueError("M24 qualification receipt does not pass frozen full gate")
    if json.loads(correction.read_text(encoding="utf-8")).get("numeric_rows_verified_identical") is not True:
        raise ValueError("M24 correction receipt incomplete")
    output_root = root / "streaming_calibration_exp/outputs/streaming_calibration"
    arms = {group: read(one(output_root, f"{SCREEN}_{group}_m2_f1_s42_*"), group) for group in GROUPS}
    launch_bindings = {group: validate_launch_binding(root, group) for group in GROUPS}
    shared = ("teacher_ckpt_path", "freeze_decoder", "loss_mode", "lambda_y", "lambda_E", "behavior_scaling_factor", "window_size", "trial_length")
    t4_model = arms["t4"]["config"]["model"]
    t4_data = arms["t4"]["config"]["data"]
    preprocess_keys = (
        "window_size", "max_trial_length", "smooth_calibration", "standardize_covariates",
        "use_intertrials", "use_calib_intertrials", "trial_feature_type",
        "interpolate_trials", "interpolate_trials_kind", "remove_calib_still_times",
    )
    for group in ("f0", "k4", "ks4"):
        candidate_data = arms[group]["config"]["data"]
        if drift := {key: {"t4": effective_data_value(t4_data, key), group: effective_data_value(candidate_data, key)}
                     for key in preprocess_keys if effective_data_value(candidate_data, key) != effective_data_value(t4_data, key)}:
            raise ValueError(f"{group}: preprocessing parity drift against T4: {drift}")
    for group in ("k4", "ks4"):
        model = arms[group]["config"]["model"]
        if any(model.get(key) != t4_model.get(key) for key in shared):
            raise ValueError(f"{group}: T4 runtime parity drift")
        if arms[group]["split"]["train_sessions"] != arms["t4"]["split"]["train_sessions"]:
            raise ValueError(f"{group}: T4 train-session split drift")
    deltas = {"K4_minus_T4": arms["k4"]["score"] - arms["t4"]["score"],
              "K4_minus_KS4": arms["k4"]["score"] - arms["ks4"]["score"],
              "K4_minus_F0": arms["k4"]["score"] - arms["f0"]["score"]}
    for arm in arms.values():
        arm.pop("config")
        arm.pop("split")
    payload = {
        "schema_version": 1, "purpose": "M2_M24_heldin_source_gate_for_disjoint_local_heldout_replay",
        "formal_heldout_evaluated": False, "qualification_receipt": str(receipt.resolve()),
        "qualification_receipt_sha256": RECEIPT_SHA,
        "qualification_correction_receipt": str(correction.resolve()),
        "qualification_correction_receipt_sha256": CORRECTION_SHA,
        "source_launch_qualification_bindings": launch_bindings,
        "cell": {"task": "m2", "fold": FOLD, "seed": SEED, "M": M, "network": "fresh_B3/B3S"},
        "arms": arms, "paired_deltas_r2": deltas,
        "gate": {"threshold": 0.03, "all_three_pass": all(value >= .03 for value in deltas.values()),
                 "rule": "K4-T4, K4-KS4, K4-F0 all >= +0.03 R2 before a test-only local held-out replay."},
    }
    out = args.out or root / "sua_exploration/results" / SCREEN / "aggregate_internal.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite aggregate {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"deltas": deltas, "all_three_pass": payload["gate"]["all_three_pass"]}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
