#!/usr/bin/env python3
"""Independent structural/arithmetic verifier for the H-SE5 terminal receipt.

This verifier deliberately does not import the evaluator that creates the
receipt.  It checks immutable artifact binding, embedded checkpoint metadata,
the sealed H-S/H-C reference, query/session accounting, state immutability,
and every terminal-gate arithmetic relation.  Prediction-level R2 is
recomputed separately because the receipt intentionally does not store all
8,965 prediction vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
SPINT_ROOT = ROOT / "SPINT-main"
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))
if str(SPINT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SPINT_ROOT))

from src.data.h1_sparse_event_source_snapshot import load_snapshot


SCHEMA = "h1_sparse_event_endpoint_hse5_fold0_terminal_v1"
CHECKPOINT_SCHEMA = "h1_sparse_event_endpoint_h32_fold0_checkpoint_v1"
QUERY_SHA = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
TARGET_SAMPLES = {
    "ses-19250101T111740": 6735,
    "ses-19250101T112404": 2230,
}
INTERVENTIONS = ("full", "zero", "row", "label")
PAIR_FIELDS = (
    "fold_date", "source_manifest_sha256", "normalizer_sha256",
    "source_cache_sha256", "normalized_cache_sha256", "basis_sha256",
    "source_hashes_sha256", "initial_state_sha256",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(first: Any, second: Any, label: str) -> None:
    need(math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1.0e-12),
         f"{label}: {first!r} != {second!r}")


def verify_checkpoint(entry: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    path = Path(entry["path"]).resolve()
    need(path.is_file() and file_sha(path) == entry["sha256"], f"{arm}: checkpoint SHA mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    need(checkpoint.get("epoch") == 49 and int(checkpoint.get("global_step", 0)) > 0,
         f"{arm}: checkpoint is not terminal epoch 49")
    need(isinstance(checkpoint.get("state_dict"), Mapping), f"{arm}: state_dict missing")
    embedded = checkpoint.get("h1_sparse_event_endpoint")
    need(embedded == entry["metadata"], f"{arm}: embedded metadata differs from receipt")
    need(embedded.get("schema") == CHECKPOINT_SCHEMA and embedded.get("arm") == arm,
         f"{arm}: checkpoint schema/arm mismatch")
    expected = {
        "fold_date": "19250101", "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50, "selected_by": "fixed_terminal_epoch_no_validation_selection",
        "carrier_dim": 5, "carrier_hidden_dim": 32, "carrier_trial_length": 1024,
        "carrier_identity_parameters": 58172, "target_session_optimizer_steps": 0,
        "target_session_backward_steps": 0,
    }
    for key, value in expected.items():
        need(embedded.get(key) == value, f"{arm}: metadata drift at {key}")
    need(embedded.get("carrier_mode") == (
        "correct_sparse_endpoint" if arm == "full" else "literal_zero5_at_model_boundary"
    ), f"{arm}: carrier mode mismatch")
    config = path.parents[2] / ".hydra/config.yaml"
    need(config.is_file() and file_sha(config) == embedded["config_sha256"],
         f"{arm}: resolved config SHA mismatch")
    return embedded


def verify_score(score: Mapping[str, Any], label: str) -> None:
    need(score["query_window_indices_sha256"] == QUERY_SHA, f"{label}: query SHA mismatch")
    need(score["samples"] == sum(TARGET_SAMPLES.values()), f"{label}: pooled sample count mismatch")
    need(score["batches"] == 281 and score["last_batch_size"] == 5,
         f"{label}: batch accounting mismatch")
    need(score["state_immutable"] is True and score["state_sha256_before"] == score["state_sha256_after"],
         f"{label}: model state changed during evaluation")
    need(set(score["per_session"]) == set(TARGET_SAMPLES), f"{label}: session roster mismatch")
    need(math.isfinite(float(score["pooled_r2"])), f"{label}: pooled R2 nonfinite")
    for session, count in TARGET_SAMPLES.items():
        row = score["per_session"][session]
        need(row["samples"] == count and math.isfinite(float(row["r2"])),
             f"{label}/{session}: invalid score")


def verify(path: Path) -> dict[str, Any]:
    path = path.resolve()
    need(path.is_file(), f"missing receipt: {path}")
    need(stat.S_IMODE(path.stat().st_mode) == 0o444, "terminal receipt is not immutable mode 0444")
    body = json.loads(path.read_text(encoding="utf-8"))
    need(body.get("schema") == SCHEMA and body.get("fold_date") == "19250101" and body.get("seed") == 42,
         "receipt identity mismatch")
    need(body.get("support_trials") == 4, "receipt is not the matched M4 cell")
    evaluator = body.get("evaluator")
    need(isinstance(evaluator, Mapping), "terminal evaluator binding missing")
    evaluator_path = Path(evaluator["path"]).resolve()
    need(evaluator_path.is_file() and file_sha(evaluator_path) == evaluator["sha256"],
         "terminal evaluator code SHA drift")
    snapshot_entry = body.get("source_snapshot")
    need(isinstance(snapshot_entry, Mapping), "terminal source snapshot binding missing")
    snapshot = load_snapshot(Path(snapshot_entry["receipt"]))
    need(snapshot.receipt_sha256 == snapshot_entry["receipt_sha256"] and
         snapshot.snapshot_sha256 == snapshot_entry["snapshot_sha256"] and
         snapshot.manifest_sha256 == snapshot_entry["manifest_sha256"] and
         snapshot.basis.basis_sha256 == snapshot_entry["basis_sha256"] and
         snapshot.normalizer.normalizer_sha256 == snapshot_entry["normalizer_sha256"],
         "terminal source snapshot binding drift")

    checkpoints = body["checkpoints"]
    full_meta = verify_checkpoint(checkpoints["full"], "full")
    zero_meta = verify_checkpoint(checkpoints["zero"], "zero")
    need(snapshot.manifest_sha256 == full_meta["source_manifest_sha256"] == zero_meta["source_manifest_sha256"],
         "source snapshot/checkpoint manifest drift")
    for field in PAIR_FIELDS:
        need(full_meta[field] == zero_meta[field] == checkpoints["pair_binding"][field],
             f"checkpoint pair drift at {field}")

    reference_entry = body["sealed_matched_references"]
    reference_path = Path(reference_entry["receipt"]).resolve()
    need(reference_path.is_file() and file_sha(reference_path) == reference_entry["receipt_sha256"],
         "sealed reference SHA mismatch")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    need(reference["target"]["strict_query_window_indices_sha256"] == QUERY_SHA,
         "sealed reference query SHA mismatch")
    close(reference["metrics"]["h_s"]["pooled_r2"], reference_entry["h_s_r2"], "sealed H-S")
    close(reference["metrics"]["h_c_interventions"]["full"]["pooled_r2"],
          reference_entry["dense_h_c_r2"], "sealed dense H-C")
    close(round(float(reference["metrics"]["h_c0"]["pooled_r2"]), 8),
          reference_entry["dense_h_c0_r2_rounded"], "sealed dense H-C0")
    need(reference_entry["query_window_indices_sha256"] == QUERY_SHA,
         "receipt/reference query SHA mismatch")

    target = body["target"]
    need(tuple(target["sessions"]) == tuple(TARGET_SAMPLES) and target["post_four_trial_query"] is True,
         "target roster/boundary mismatch")
    need(target["query_window_indices_sha256"] == QUERY_SHA, "target query SHA mismatch")
    supports = target["support_and_carrier_hashes"]
    need(set(supports) == set(TARGET_SAMPLES), "target support roster mismatch")
    for session, support in supports.items():
        need(len(support["trial_values"]) == 4 and math.isfinite(float(support["fifth_trial"])),
             f"{session}: support trial boundary mismatch")
        hashes = support["carrier_sha256"]
        need(set(hashes) == set(INTERVENTIONS) and all(len(value) == 64 for value in hashes.values()),
             f"{session}: intervention hash roster mismatch")
        need(len(set(hashes.values())) == 4, f"{session}: intervention carrier collision")

    metrics = body["metrics"]
    intervention_scores = metrics["hse5_same_checkpoint_interventions"]
    need(set(intervention_scores) == set(INTERVENTIONS), "intervention score roster mismatch")
    for name in INTERVENTIONS:
        verify_score(intervention_scores[name], f"H-SE5/{name}")
    verify_score(metrics["separately_trained_zero5"], "H-SE5-Z5")
    full_state = intervention_scores["full"]["state_sha256_before"]
    need(all(intervention_scores[name]["state_sha256_before"] == full_state for name in INTERVENTIONS),
         "same-checkpoint controls did not use one model state")

    candidate = float(intervention_scores["full"]["pooled_r2"])
    expected_margins = {
        "hse5_minus_separately_trained_zero5": candidate - float(metrics["separately_trained_zero5"]["pooled_r2"]),
        "hse5_minus_same_checkpoint_zero5": candidate - float(intervention_scores["zero"]["pooled_r2"]),
        "hse5_minus_same_checkpoint_row_shuffle": candidate - float(intervention_scores["row"]["pooled_r2"]),
        "hse5_minus_same_checkpoint_label_shuffle": candidate - float(intervention_scores["label"]["pooled_r2"]),
        "hse5_minus_sealed_hs": candidate - float(reference_entry["h_s_r2"]),
        "hse5_minus_sealed_dense_hc": candidate - float(reference_entry["dense_h_c_r2"]),
    }
    gate = body["gate"]
    for key, value in expected_margins.items():
        close(gate["margins"][key], value, key)
    expected_clauses = {
        "beats_separately_trained_zero5": expected_margins["hse5_minus_separately_trained_zero5"] > 0,
        "beats_same_checkpoint_zero5": expected_margins["hse5_minus_same_checkpoint_zero5"] > 0,
        "beats_same_checkpoint_row_shuffle": expected_margins["hse5_minus_same_checkpoint_row_shuffle"] > 0,
        "beats_same_checkpoint_label_shuffle": expected_margins["hse5_minus_same_checkpoint_label_shuffle"] > 0,
    }
    need(gate["clauses"] == expected_clauses and gate["pass"] == all(expected_clauses.values()),
         "gate arithmetic/status mismatch")
    need(body["status"] == ("PASS_HSE5_FIRST_CELL" if gate["pass"] else "STOP_HSE5_FIRST_CELL"),
         "terminal status mismatch")
    need(body["scope"] == {
        "public_held_in_calibration_nwbs_opened": 13,
        "minival_opened": False,
        "heldout_opened": False,
        "formal_opened": False,
        "evalai_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
    }, "scope mismatch")
    return {
        "status": "PASS",
        "receipt": str(path),
        "sha256": file_sha(path),
        "scientific_status": body["status"],
        "hse5_r2": candidate,
        "gate_pass": gate["pass"],
        "margins": expected_margins,
        "source_snapshot_receipt_sha256": snapshot.receipt_sha256,
        "source_snapshot_sha256": snapshot.snapshot_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.receipt)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        need(not output.exists(), f"refusing to overwrite audit artifact: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        output.chmod(0o444)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
