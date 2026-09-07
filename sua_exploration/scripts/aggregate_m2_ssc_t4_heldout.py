#!/usr/bin/env python3
"""Fail-closed one-shot aggregate for the frozen SSC-T4 local-heldout replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml
from scipy.stats import wilcoxon


PROTOCOL_SHA256 = "c769f39703f33e1e1f1f0c02a77d7cc9cd28356e1af2b51ec47fbc214aec24dc"
EXPECTED_SESSIONS = {
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path, arm: str, source_gate: dict) -> dict:
    required = ["resolved_config.yaml", "split_manifest.json", "metrics_per_session.csv", "checkpoint_manifest.json", "heldout_ssc_t4_provenance.json"]
    if missing := [item for item in required if not (path / item).is_file()]:
        raise ValueError(f"{path}: missing heldout artifact fields {missing}")
    config = yaml.safe_load((path / "resolved_config.yaml").read_text())
    split = json.loads((path / "split_manifest.json").read_text())
    checkpoint = json.loads((path / "checkpoint_manifest.json").read_text())
    provenance = json.loads((path / "heldout_ssc_t4_provenance.json").read_text())
    data, model = config["data"], config["model"]
    expected_weight = 0.0 if arm == "ordinary_t4" else 1.0
    expected_source = source_gate["arms"][arm]["checkpoint"]
    actual = Path(checkpoint.get("artifact_checkpoint_path", ""))
    if not actual.is_file() or checkpoint.get("artifact_checkpoint_sha256") != sha256(actual):
        raise ValueError(f"{path}: test checkpoint hash mismatch")
    if Path(str(config.get("ckpt_path", ""))).resolve() != Path(expected_source["path"]).resolve() or sha256(actual) != expected_source["sha256"]:
        raise ValueError(f"{path}: test does not use frozen source checkpoint")
    expected = {
        "task": "m2", "M": 24, "random": False, "fit": False, "test": True,
        "query": 24, "variant": "B3S", "side": "t4", "ssc": expected_weight,
        "train": False, "run_test": True,
    }
    observed = {
        "task": data.get("task"), "M": data.get("calibration_n_trials"), "random": data.get("random_calibration"),
        "fit": data.get("include_heldout_in_fit"), "test": data.get("include_heldout_in_test"),
        "query": data.get("query_start_trial"), "variant": model.get("variant"), "side": data.get("side_feature_group"),
        "ssc": model.get("ssc_t4_prediction_consistency_weight"), "train": config.get("train"), "run_test": config.get("test"),
    }
    if observed != expected:
        raise ValueError(f"{path}: test-only contract drift {observed}")
    if split.get("heldout_evaluated_in_fit") is not False or split.get("heldout_evaluated_in_test") is not True:
        raise ValueError(f"{path}: split manifest does not certify test-only heldout")
    if provenance.get("arm") != arm or provenance.get("protocol_receipt", {}).get("sha256") != PROTOCOL_SHA256:
        raise ValueError(f"{path}: missing or mismatched heldout provenance protocol binding")
    if provenance.get("source_gate", {}).get("sha256") != sha256(Path(source_gate["_path"])):
        raise ValueError(f"{path}: heldout provenance binds a different source gate")
    if provenance.get("frozen_source_student_checkpoint", {}).get("sha256") != expected_source["sha256"]:
        raise ValueError(f"{path}: heldout provenance binds a different student checkpoint")
    if provenance.get("source_student_selection") != source_gate["arms"][arm].get("checkpoint_selection"):
        raise ValueError(f"{path}: heldout provenance does not bind the frozen held-in student selector")
    audit = provenance.get("query_contract", {}).get("per_session_audit")
    if not isinstance(audit, dict) or set(audit) != EXPECTED_SESSIONS:
        raise ValueError(f"{path}: heldout provenance lacks exact six-session audit")
    for session, row in audit.items():
        if not isinstance(row, dict) or row.get("support_trials") != 24 or row.get("query_start_trial") != 24 or row.get("window_size") != 50 or row.get("full_window_disjoint") is not True:
            raise ValueError(f"{path}: {session} provenance support/query/window contract failed")
    nwb_rows = provenance.get("six_heldout_calibration_nwbs")
    if not isinstance(nwb_rows, list) or {row.get("session") for row in nwb_rows if isinstance(row, dict)} != EXPECTED_SESSIONS or len(nwb_rows) != 6:
        raise ValueError(f"{path}: provenance lacks exact six held-out NWB rows")
    if not all(isinstance(row, dict) and row.get("sha256") and int(row.get("size_bytes", 0)) > 0 for row in nwb_rows):
        raise ValueError(f"{path}: provenance held-out NWB fingerprints are incomplete")
    scores = {}
    for row in csv.DictReader((path / "metrics_per_session.csv").open(encoding="utf-8")):
        if row.get("split") == "test_heldout":
            score = float(row["R2_variance_weighted"])
            if not math.isfinite(score):
                raise ValueError(f"{path}: non-finite heldout score for {row.get('session')}")
            scores[row["session"]] = score
    if set(scores) != EXPECTED_SESSIONS:
        raise ValueError(f"{path}: expected six heldout session scores, got {len(scores)}")
    return {"path": str(path.resolve()), "scores": scores,
            "checkpoint": {"path": str(actual.resolve()), "sha256": sha256(actual)},
            "provenance": provenance}


def read_clean_spint_reference(path: Path, source_gate: dict) -> dict:
    reference = json.loads(path.read_text())
    if reference.get("role") != "clean_full_spint_third_reference_not_primary":
        raise ValueError("clean SPINT reference role is not third-reference-only")
    if reference.get("protocol_receipt", {}).get("sha256") != PROTOCOL_SHA256:
        raise ValueError("clean SPINT reference protocol receipt mismatch")
    if reference.get("source_gate", {}).get("sha256") != sha256(Path(source_gate["_path"])):
        raise ValueError("clean SPINT reference source gate mismatch")
    descriptor = reference.get("descriptor_contract", {})
    required = {"calibration_trials": 24, "query_start_trial": 24, "window_size_bins": 50,
                "side_feature_group": "none", "calibration_target_labels_used": False,
                "backward_gradients": False, "optimizer_updates": False, "checkpoint_selection": False}
    if any(descriptor.get(key) != value for key, value in required.items()):
        raise ValueError("clean SPINT reference violates no-label M24/query24 contract")
    scores = reference.get("per_session_r2", {})
    if not isinstance(scores, dict) or set(scores) != EXPECTED_SESSIONS:
        raise ValueError("clean SPINT reference lacks exact six session scores")
    try:
        numeric_scores = {session: float(value) for session, value in scores.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("clean SPINT reference has non-numeric session score") from exc
    if not all(math.isfinite(value) for value in numeric_scores.values()):
        raise ValueError("clean SPINT reference has non-finite session score")
    computed_mean = sum(numeric_scores.values()) / len(numeric_scores)
    try:
        recorded_mean = float(reference.get("mean_r2_equal_session"))
    except (TypeError, ValueError) as exc:
        raise ValueError("clean SPINT reference has no numeric equal-session mean") from exc
    if not math.isfinite(recorded_mean) or not math.isclose(recorded_mean, computed_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("clean SPINT reference equal-session mean does not match its six scores")
    audit = reference.get("query_window_audit")
    if not isinstance(audit, dict) or set(audit) != EXPECTED_SESSIONS:
        raise ValueError("clean SPINT reference query audit is incomplete")
    return {"path": str(path.resolve()), "sha256": sha256(path), "per_session_r2": numeric_scores,
            "mean_r2_equal_session": recorded_mean, "provenance": reference}


def equal_session_contrast(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, object]:
    """Six-session paired report; never used to decide the SSC primary gate."""
    sessions = sorted(EXPECTED_SESSIONS)
    deltas = [candidate[session] - reference[session] for session in sessions]
    return {
        "mean": sum(deltas) / len(deltas),
        "per_session": dict(zip(sessions, deltas)),
        "positive_sessions": sum(delta > 0 for delta in deltas),
        "exact_wilcoxon_two_sided_p": float(wilcoxon(deltas, alternative="two-sided", method="exact").pvalue),
        "decision_role": "third_reference_only_not_part_of_primary_gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--ssc", type=Path, required=True)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--clean-spint-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if sha256(args.protocol_receipt) != PROTOCOL_SHA256:
        raise ValueError("frozen heldout protocol receipt SHA drift")
    source_gate = json.loads(args.source_gate.read_text())
    source_gate["_path"] = str(args.source_gate.resolve())
    if source_gate.get("next_action") != "one_frozen_local_heldout_replay_required":
        raise ValueError("source catastrophic gate forbids heldout aggregate")
    ordinary, ssc = read(args.ordinary, "ordinary_t4", source_gate), read(args.ssc, "ssc_t4", source_gate)
    if set(ordinary["scores"]) != set(ssc["scores"]):
        raise ValueError("heldout arms do not share exactly the same sessions")
    for field in ("clean_teacher_receipt", "native_t4_normalization"):
        if ordinary["provenance"].get(field) != ssc["provenance"].get(field):
            raise ValueError(f"heldout arms do not share identical {field}")
    if ordinary["provenance"].get("six_heldout_calibration_nwbs") != ssc["provenance"].get("six_heldout_calibration_nwbs"):
        raise ValueError("heldout arms do not share identical NWB fingerprints")
    if ordinary["provenance"].get("support_contract") != ssc["provenance"].get("support_contract"):
        raise ValueError("heldout arms do not share an identical support contract")
    if ordinary["provenance"].get("query_contract") != ssc["provenance"].get("query_contract"):
        raise ValueError("heldout arms do not share an identical query audit")
    clean_spint = read_clean_spint_reference(args.clean_spint_reference.resolve(), source_gate)
    if clean_spint["provenance"].get("clean_teacher_receipt") != ordinary["provenance"].get("clean_teacher_receipt"):
        raise ValueError("clean SPINT reference uses a different teacher receipt")
    if clean_spint["provenance"].get("six_heldout_calibration_nwbs") != ordinary["provenance"].get("six_heldout_calibration_nwbs"):
        raise ValueError("clean SPINT reference uses different held-out NWBs")
    if clean_spint["provenance"].get("query_window_audit") != ordinary["provenance"].get("query_contract", {}).get("per_session_audit"):
        raise ValueError("clean SPINT reference has a different query-window audit")
    sessions = sorted(ordinary["scores"])
    deltas = [ssc["scores"][session] - ordinary["scores"][session] for session in sessions]
    exact_p = float(wilcoxon(deltas, alternative="two-sided", method="exact").pvalue)
    primary = sum(delta > 0 for delta in deltas) >= 5 and sum(deltas) / len(deltas) >= .03
    payload = {
        "schema_version": 1, "scope": "single_local_heldout_calibration_replay",
        "formal_heldout_evaluated": False, "local_heldout_calib_evaluated": True,
        "hidden_evalai_evaluated": False,
        "protocol_receipt": str(args.protocol_receipt.resolve()), "protocol_receipt_sha256": PROTOCOL_SHA256,
        "source_gate": str(args.source_gate.resolve()), "source_gate_sha256": sha256(args.source_gate),
        "arms": {"ordinary_t4": {key: value for key, value in ordinary.items() if key != "provenance"}, "ssc_t4": {key: value for key, value in ssc.items() if key != "provenance"}},
        "shared_provenance": {key: ordinary["provenance"][key] for key in ("clean_teacher_receipt", "native_t4_normalization", "six_heldout_calibration_nwbs", "support_contract", "query_contract")},
        "clean_spint_reference": {key: value for key, value in clean_spint.items() if key != "provenance"},
        "per_session_r2": {"ordinary_t4": ordinary["scores"], "ssc_t4": ssc["scores"]},
        "ssc_minus_ordinary": {"mean": sum(deltas) / len(deltas), "per_session": dict(zip(sessions, deltas)),
                                 "positive_sessions": sum(delta > 0 for delta in deltas), "exact_wilcoxon_two_sided_p": exact_p},
        "primary_rule": "mean SSC-T4 minus ordinary-T4 >= +0.03 R2 and >=5/6 positive sessions",
        "primary_effective": primary,
        "third_reference_contrasts": {
            "ordinary_t4_minus_clean_spint": equal_session_contrast(ordinary["scores"], clean_spint["per_session_r2"]),
            "ssc_t4_minus_clean_spint": equal_session_contrast(ssc["scores"], clean_spint["per_session_r2"]),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
