#!/usr/bin/env python3
"""Fail-closed aggregation for the single frozen M2 K4/KS4 Gate-B cell."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


SCREEN_DEFAULT = "m2_k4_gate_b_v1"
GATE_A_SHA256 = "8767d7ce0fc852401273806a61492a607a2064a4546bbd1c21e25ce344cf4ec9"
FOLD, SEED, M = 1, 42, 33
REF_GROUPS = {"f0": ("B3", "none"), "t4": ("B3S", "t4"), "ts4": ("B3S", "ts4")}
NEW_GROUPS = {"k4": "k4", "ks4": "ks4"}
REFERENCE_SHA256 = {
    "f0": {
        "run_metadata": "9b932d83daaf8b2cbdc623b29f54db8408837cf9b370529429dd0f85df23a8e7",
        "resolved_config": "1eb291de18b82911ff16e57be50414c50ac3923e9f308e3e88f48900cc059092",
        "split_manifest": "129a646aa98494cb8e6ca60cb74cd3585c896d4fbb01d6342655846684cdb896",
    },
    "t4": {
        "run_metadata": "7cc2e8856146e4bc55880592380a73a02f74d7fbb0d2ed993f28c7105592504e",
        "resolved_config": "839b38f05e9050645d9d7b0525e721bd0fb035abaa28819021fa2f4de16c21dc",
        "split_manifest": "129a646aa98494cb8e6ca60cb74cd3585c896d4fbb01d6342655846684cdb896",
    },
    "ts4": {
        "run_metadata": "f2e8f4415c4bcb4a3a7f5aa7dfb69fe818eb50fe974d5c1821f4021fc681f0d0",
        "resolved_config": "9260a73cac8071dc58cfaaa5f35e94f915d1c8ddaacaff2c5afa4c5f26486c8d",
        "split_manifest": "129a646aa98494cb8e6ca60cb74cd3585c896d4fbb01d6342655846684cdb896",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gate_a(path: Path) -> dict:
    """Bind Gate B to the one frozen, successful development-only Gate A."""
    if sha256(path) != GATE_A_SHA256:
        raise ValueError("Gate-A artifact SHA256 drifted; refusing Gate-B aggregate")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("formal_heldout_evaluated") is not False:
        raise ValueError("Gate-A artifact must be development-only")
    if payload.get("summary", {}).get("stage1_candidate") is not True:
        raise ValueError("Gate-A artifact does not authorize a Gate-B GPU screen")
    return payload


def one_directory(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_dir())
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern}, found {matches}")
    return matches[0]


def read_score(path: Path) -> float:
    with (path / "metrics_summary.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "test_heldin":
                if int(row["M"]) != M:
                    raise ValueError(f"{path}: metric M must equal {M}")
                return float(row["R2_variance_weighted"])
    raise ValueError(f"{path}: missing test_heldin score")


def read_artifact(path: Path, *, group: str, new: bool) -> dict:
    required = ["resolved_config.yaml", "split_manifest.json", "run_metadata.json", "metrics_summary.csv"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"{path}: missing required files {missing}")
    config = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    split = json.loads((path / "split_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    data, model, trainer = config.get("data", {}), config.get("model", {}), config.get("trainer", {})
    expected_side = NEW_GROUPS[group] if new else REF_GROUPS[group][1]
    expected_variant = "B3S" if new else REF_GROUPS[group][0]
    expected = {
        "task": "m2", "fold": FOLD, "seed": SEED, "variant": expected_variant,
        "side": expected_side, "calibration": M, "random": False,
        "fit_heldout": False, "test_heldout": False, "protocol": "loso", "epochs": 12,
    }
    observed = {
        "task": data.get("task"), "fold": data.get("loso_fold"), "seed": config.get("seed"),
        "variant": model.get("variant"), "side": data.get("side_feature_group"),
        "calibration": data.get("calibration_n_trials"), "random": data.get("random_calibration"),
        "fit_heldout": data.get("include_heldout_in_fit"), "test_heldout": data.get("include_heldout_in_test"),
        "protocol": data.get("validation_protocol"), "epochs": trainer.get("max_epochs"),
    }
    bad = {key: {"expected": expected[key], "observed": observed[key]} for key in expected if expected[key] != observed[key]}
    if bad:
        raise ValueError(f"{path}: resolved-config contract mismatch: {bad}")
    if split.get("heldout_evaluated_in_fit") is not False or split.get("heldout_evaluated_in_test") is not False:
        raise ValueError(f"{path}: held-out scope was evaluated")
    if metadata.get("fold_id") != FOLD or metadata.get("seed") != SEED:
        raise ValueError(f"{path}: run metadata fold/seed mismatch")
    if new:
        if model.get("side_dim") != 4 or data.get("smooth_calibration") is not False:
            raise ValueError(f"{path}: K4 requires B3S side_dim=4 and raw unsmoothed calibration")
        if data.get("standardize_covariates") is not False or data.get("use_intertrials") is not True:
            raise ValueError(f"{path}: K4 raw-covariate contract mismatch")
        if data.get("max_trial_length") != 100 or data.get("interpolate_trials") is not True or data.get("interpolate_trials_kind") != "cubic":
            raise ValueError(f"{path}: online activity parity cap/interpolation contract mismatch")
        if model.get("freeze_decoder") is not True or model.get("loss_mode") != "task_plus_y_plus_E":
            raise ValueError(f"{path}: decoder/loss parity mismatch")
        if model.get("lambda_y") != 1.0 or model.get("lambda_E") != 0.1:
            raise ValueError(f"{path}: loss-coefficient parity mismatch")
        estimator = split.get("k4_estimator", {})
        if estimator.get("raw_full_trial_no_interpolation_no_max_trial_length_cap") is not True:
            raise ValueError(f"{path}: missing raw-full-trial K4 provenance")
    hashes = {
        "run_metadata": sha256(path / "run_metadata.json"),
        "resolved_config": sha256(path / "resolved_config.yaml"),
        "split_manifest": sha256(path / "split_manifest.json"),
    }
    if not new and hashes != REFERENCE_SHA256[group]:
        raise ValueError(f"{path}: frozen native_mua_t4_v1 reference SHA drift: {hashes}")
    return {
        "path": str(path.resolve()), "score": read_score(path),
        "run_metadata_sha256": hashes["run_metadata"],
        "resolved_config_sha256": hashes["resolved_config"],
        "split_manifest_sha256": hashes["split_manifest"],
        "split_manifest": split,
        "_resolved_config": config,
    }


def assert_split_parity(reference: dict, candidate: dict, *, group: str) -> None:
    """K4/KS4 must hold out precisely the existing F0/T4 validation session."""
    ref = reference["split_manifest"]
    got = candidate["split_manifest"]
    fields = ("validation_protocol", "fold_id", "train_sessions", "validation_sessions",
              "heldout_evaluated_in_fit", "heldout_evaluated_in_test")
    drift = {field: {"reference": ref.get(field), "candidate": got.get(field)}
             for field in fields if ref.get(field) != got.get(field)}
    if drift:
        raise ValueError(f"{group}: F0/T4 split parity drift: {drift}")


def assert_t4_runtime_parity(t4_config: dict, candidate_config: dict, *, group: str) -> None:
    """Forbid an architecture/loss/checkpoint-policy explanation of a K4 gap."""
    parity_paths = (
        ("model", "teacher_ckpt_path"), ("model", "freeze_decoder"),
        ("model", "loss_mode"), ("model", "lambda_y"), ("model", "lambda_E"),
        ("model", "behavior_scaling_factor"), ("model", "window_size"),
        ("model", "trial_length"), ("data", "window_size"),
        ("data", "max_trial_length"), ("data", "calibration_n_trials"),
        ("data", "random_calibration"), ("data", "interpolate_trials"),
        ("data", "interpolate_trials_kind"), ("trainer", "max_epochs"),
        ("callbacks", "early_stopping"), ("", "no_early_stopping"),
    )
    drift = {}
    for section, key in parity_paths:
        reference_value = t4_config.get(key) if not section else t4_config.get(section, {}).get(key)
        candidate_value = candidate_config.get(key) if not section else candidate_config.get(section, {}).get(key)
        if reference_value != candidate_value:
            drift[f"{section + '.' if section else ''}{key}"] = {
                "t4": reference_value, "candidate": candidate_value
            }
    if drift:
        raise ValueError(f"{group}: frozen T4 decoder/loss/activity/checkpoint parity drift: {drift}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-id", default=SCREEN_DEFAULT)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    gate_a = root / "sua_exploration/results/general_carrier_proxy_v1/audit_m2_heldin_v2.json"
    validate_gate_a(gate_a)
    output_root = root / "streaming_calibration_exp/outputs/streaming_calibration"
    refs = {
        group: read_artifact(
            one_directory(output_root, f"native_mua_t4_v1_{group}_m2_f1_s42_*"), group=group, new=False
        ) for group in REF_GROUPS
    }
    new = {
        group: read_artifact(
            one_directory(output_root, f"{args.screen_id}_{group}_m2_f1_s42_*"), group=group, new=True
        ) for group in NEW_GROUPS
    }
    # K4 may change only the side-feature construction.  Its decoder, teacher,
    # loss, online activity cap, and preprocessing must match the frozen T4
    # reference exactly; otherwise an apparent score gap is uninterpretable.
    t4_config = refs["t4"]["_resolved_config"]
    for group, record in new.items():
        candidate_config = record["_resolved_config"]
        assert_split_parity(refs["t4"], record, group=group)
        assert_split_parity(refs["f0"], record, group=group)
        assert_t4_runtime_parity(t4_config, candidate_config, group=group)
        record.pop("_resolved_config")
    for record in refs.values():
        record.pop("_resolved_config")
    k4 = new["k4"]["score"]
    deltas = {"K4_minus_T4": k4 - refs["t4"]["score"], "K4_minus_KS4": k4 - new["ks4"]["score"], "K4_minus_F0": k4 - refs["f0"]["score"]}
    pass_all = all(value >= 0.03 for value in deltas.values())
    marginal_shrink_eligible = (-0.03 <= deltas["K4_minus_T4"] < 0.03 and deltas["K4_minus_KS4"] >= 0.03)
    early_stop = (deltas["K4_minus_T4"] < -0.05 or deltas["K4_minus_KS4"] < 0.03)
    payload = {
        "schema_version": 1,
        "purpose": "frozen_M2_K4_KS4_Gate_B_internal_LOSO_fold1_seed42",
        "formal_heldout_evaluated": False,
        "gate_a_artifact": str(gate_a.resolve()), "gate_a_sha256": GATE_A_SHA256,
        "cell": {"task": "m2", "fold": FOLD, "seed": SEED, "M": M, "network": "fresh_B3S_side_dim4"},
        "references": refs, "new_arms": new, "paired_deltas_r2": deltas,
        "gate": {"threshold": 0.03, "all_three_pass": pass_all, "early_stop": early_stop,
                 "marginal_shrink_eligible": marginal_shrink_eligible,
                 "rule": "K4-T4, K4-KS4, and K4-F0 must each be >= +0.03 R2"},
        "exposure_disclosure": (
            "K4 uses raw full-trial contiguous blocks; legacy T4 trial sums cap valid_prefix at max_trial_length=100. "
            "This screen is not an equal-exposure claim; positive K4 needs the preregistered cap/full sensitivity follow-up."
        ),
    }
    out = args.out or root / "sua_exploration/results" / args.screen_id / "aggregate_seed42.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"deltas": deltas, "all_three_pass": pass_all, "early_stop": early_stop}, indent=2, sort_keys=True))
    print(out)


if __name__ == "__main__":
    main()
