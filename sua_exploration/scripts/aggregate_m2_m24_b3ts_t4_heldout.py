#!/usr/bin/env python3
"""Fail-closed local-heldout aggregate for the frozen M2 M24 B3TS screen."""
import csv
import hashlib
import json
import re
from pathlib import Path

import yaml
from scipy.stats import wilcoxon

R = Path(__file__).resolve().parents[2]
X = "m2_m24_b3ts_t4_v1"
S = {
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
}
GROUP = {"t4": ("B3S", "t4"), "b3ts": ("B3TS", "t4")}
PROTOCOL_SHA = "5779aaee90f1a4050b135803ef6a84bd2b556330ef00ed171ca8f071c6e2f286"
LABEL_BUDGET = "M24 neural activity plus trial-level target-direction labels only; no continuous velocity/K4 labels"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_from_nwb(path: Path) -> str:
    match = re.search(r"_(ses-[^_]+)_", path.name)
    if match is None:
        raise ValueError(f"unparseable held-out NWB name: {path.name}")
    return match.group(1)


def current_nwb_binding() -> list[dict[str, str]]:
    files = []
    for path in sorted((R / "SPINT-main/data/000953").rglob("*held-out-calib*.nwb")):
        files.append({"session": session_from_nwb(path), "path": str(path.resolve()), "sha256": sha(path)})
    if {item["session"] for item in files} != S:
        raise ValueError("current held-out NWB set")
    return files


def check_config(group: str, config: dict, checkpoint: Path) -> None:
    data, model = config["data"], config["model"]
    variant, side = GROUP[group]
    got = (
        data.get("task"), data.get("loso_fold"), config.get("seed"), model.get("variant"),
        data.get("side_feature_group"), data.get("calibration_n_trials"),
        data.get("random_calibration"), data.get("include_heldout_in_fit"),
        data.get("include_heldout_in_test"), data.get("query_start_trial"),
        config.get("train"), config.get("test"),
    )
    want = ("m2", 1, 42, variant, side, 24, False, False, True, 24, False, True)
    if got != want:
        raise ValueError(f"{group} heldout config contract: {got}")
    if str(Path(config["ckpt_path"]).resolve()) != str(checkpoint.resolve()):
        raise ValueError("resolved checkpoint path")


def check_windows(audit: dict) -> dict:
    if set(audit) != S:
        raise ValueError("six session audit")
    for session, value in audit.items():
        if not (
            value.get("support_trials") == value.get("query_start_trial") == 24
            and value.get("window_size") == 50
            and value.get("full_window_disjoint") is True
            and value.get("minimum_window_start_padded_bin") == value.get("raw_query_start_bin") + 49
        ):
            raise ValueError(f"window contract: {session}")
    return {
        name: (
            audit[name]["eligible_windows"], audit[name]["raw_query_start_bin"],
            audit[name]["minimum_window_start_padded_bin"], audit[name]["query_trials"],
        )
        for name in sorted(S)
    }


def arm(group: str, source_arm: dict, source_aggregate_sha: str) -> tuple[Path, dict, dict, dict]:
    hits = list((R / "streaming_calibration_exp/outputs/streaming_calibration").glob(
        f"{X}_heldout_{group}_m2_f1_s42_*"
    ))
    if len(hits) != 1:
        raise ValueError(f"expected exactly one heldout {group} artifact, got {hits}")
    path = hits[0]
    provenance_path = path / "heldout_b3ts_t4_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if (
        provenance.get("formal_heldout_evaluated") is not False
        or provenance.get("local_heldout_calib_evaluated") is not True
        or provenance.get("hidden_evalai_evaluated") is not False
    ):
        raise ValueError("scope")
    if provenance.get("group") != group:
        raise ValueError("provenance group")
    if provenance.get("frozen_protocol_receipt_sha256") != PROTOCOL_SHA:
        raise ValueError("protocol receipt provenance")
    if provenance.get("source_aggregate_sha256") != source_aggregate_sha:
        raise ValueError("source aggregate provenance")
    checkpoint = source_arm["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    if not checkpoint_path.is_file() or sha(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError("source checkpoint bytes")
    if provenance.get("source_checkpoint") != checkpoint:
        raise ValueError("source checkpoint provenance")
    config_path = path / "resolved_config.yaml"
    check_config(group, yaml.safe_load(config_path.read_text()), checkpoint_path)
    split = json.loads((path / "split_manifest.json").read_text())
    windows = check_windows(split.get("heldout_query_window_audit", {}))
    if provenance.get("query_contract") != split["heldout_query_window_audit"]:
        raise ValueError("provenance query contract")
    if provenance.get("label_budget") != LABEL_BUDGET:
        raise ValueError("label budget")
    if provenance.get("no_backward_optimizer_or_checkpoint_selection_on_heldout") is not True:
        raise ValueError("heldout optimizer contract")
    nwbs = current_nwb_binding()
    if provenance.get("six_calibration_nwbs") != nwbs:
        raise ValueError("heldout NWB provenance")
    rows = [row for row in csv.DictReader((path / "metrics_per_session.csv").open()) if row["split"] == "test_heldout"]
    scores = {row["session"]: float(row["R2_variance_weighted"]) for row in rows}
    if set(scores) != S:
        raise ValueError("scores")
    binding = {
        "artifact": str(path.resolve()),
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": sha(provenance_path),
        "resolved_config_path": str(config_path.resolve()),
        "resolved_config_sha256": sha(config_path),
        "source_checkpoint": checkpoint,
        "six_calibration_nwbs": nwbs,
        "label_budget": LABEL_BUDGET,
        "no_backward_optimizer_or_checkpoint_selection_on_heldout": True,
    }
    return path, scores, windows, binding


def main() -> None:
    source = R / f"sua_exploration/results/{X}/aggregate_source.json"
    receipt = R / f"sua_exploration/results/{X}/protocol_stage0_and_conditional_expansion.json"
    if not receipt.is_file() or sha(receipt) != PROTOCOL_SHA:
        raise ValueError("frozen protocol receipt")
    source_payload = json.loads(source.read_text())
    if source_payload.get("gate", {}).get("pass") is not True:
        raise ValueError("source gate")
    source_sha = sha(source)
    t4 = arm("t4", source_payload["arms"]["t4"], source_sha)
    b3ts = arm("b3ts", source_payload["arms"]["b3ts"], source_sha)
    if t4[2] != b3ts[2]:
        raise ValueError("unpaired windows")
    delta = {name: b3ts[1][name] - t4[1][name] for name in sorted(S)}
    values = list(delta.values())
    output = R / f"sua_exploration/results/{X}/aggregate_heldout.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({
        "local_heldout_calib_evaluated": True,
        "formal_heldout_evaluated": False,
        "hidden_evalai_evaluated": False,
        "scope": "local visible heldout-calib replay, not hidden EvalAI",
        "frozen_protocol_receipt": str(receipt.resolve()),
        "frozen_protocol_receipt_sha256": PROTOCOL_SHA,
        "conditional_expansion_must_inherit_protocol_receipt_sha256": PROTOCOL_SHA,
        "source_aggregate_sha256": source_sha,
        "arm_provenance_bindings": {"t4": t4[3], "b3ts": b3ts[3]},
        "per_session_r2": {"t4": t4[1], "b3ts": b3ts[1]},
        "query_contract": t4[2],
        "t4_mean": sum(t4[1].values()) / 6,
        "b3ts_mean": sum(b3ts[1].values()) / 6,
        "delta": delta,
        "mean_delta": sum(values) / 6,
        "positive_sessions": sum(value > 0 for value in values),
        "all_six_positive": sum(value > 0 for value in values) == 6,
        "exact_wilcoxon_two_sided_p": float(wilcoxon(values, method="exact").pvalue),
        "primary_pass": sum(values) / 6 >= .03 and sum(value > 0 for value in values) >= 5,
        "primary_rule": "mean>=.03 and >=5/6 positive",
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
