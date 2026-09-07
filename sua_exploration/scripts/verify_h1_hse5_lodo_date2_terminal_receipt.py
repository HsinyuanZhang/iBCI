#!/usr/bin/env python3
"""Structural/arithmetic verifier for a date-2 H-SE5 terminal receipt.

It does not import the terminal evaluator or replay predictions.  The latter
is intentionally delegated to ``recompute_h1_hse5_lodo_date2_r2.py`` so that
the receipt checker and prediction calculation have independent code paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


WORKSPACE = Path(__file__).resolve().parents[2]
SPINT = WORKSPACE / "SPINT-main"
for directory in (WORKSPACE, SPINT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch

from src.data.h1_sparse_event_source_snapshot_dated import load_snapshot


SCHEMA = "h1_hse5_lodo_date2_terminal_evaluation_v1"
CHECKPOINT_SCHEMA = "h1_sparse_event_endpoint_dated_h32_terminal_checkpoint_v1"
PROTOCOL = "h1_sparse_event_endpoint_q4_ridge3_lodo_m4_date2_v1"
OUTER_DATE = "19250108"
QUERY_SHA = "b0cd153750cb484af1237b7af1861600aca69a9b201244c22d140b42b1da7f6e"
TARGET_SAMPLES = {
    "ses-19250108T110520": 8330,
    "ses-19250108T111022": 2508,
    "ses-19250108T111455": 2269,
}
INTERVENTIONS = ("full", "zero", "row", "label")
PAIR_FIELDS = (
    "fold_date", "protocol", "source_manifest_sha256", "normalizer_sha256", "source_cache_sha256",
    "normalized_cache_sha256", "basis_sha256", "source_hashes_sha256", "initial_state_sha256",
    "source_snapshot_receipt_sha256", "source_snapshot_sha256",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact(left: Any, right: Any, label: str) -> None:
    need(math.isclose(float(left), float(right), abs_tol=1e-12, rel_tol=0.0), f"{label}: {left!r} != {right!r}")


def verify_checkpoint(entry: Mapping[str, Any], arm: str, snapshot: Any, training_source_manifest_sha256: str) -> Mapping[str, Any]:
    path = Path(entry["path"]).resolve()
    need(path.is_file() and sha256_file(path) == entry["sha256"], f"{arm}: checkpoint SHA mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("h1_sparse_event_endpoint_dated")
    need(checkpoint.get("epoch") == 49 and int(checkpoint.get("global_step", 0)) > 0,
         f"{arm}: nonterminal epoch")
    need(isinstance(metadata, Mapping) and metadata == entry["metadata"], f"{arm}: receipt/embedded metadata mismatch")
    expected = {
        "schema": CHECKPOINT_SCHEMA, "protocol": PROTOCOL, "fold_date": OUTER_DATE, "arm": arm,
        "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_selection",
        "carrier_mode": "correct_sparse_endpoint" if arm == "full" else "literal_zero5_at_model_boundary",
        "carrier_dim": 5, "carrier_hidden_dim": 32, "carrier_trial_length": 1024,
        "carrier_identity_parameters": 58172, "target_session_optimizer_steps": 0,
        "target_session_backward_steps": 0, "source_snapshot_receipt_sha256": snapshot.receipt_sha256,
        "source_snapshot_sha256": snapshot.snapshot_sha256, "source_manifest_sha256": training_source_manifest_sha256,
        "normalizer_sha256": snapshot.normalizer.normalizer_sha256, "basis_sha256": snapshot.basis.basis_sha256,
    }
    for key, value in expected.items():
        need(metadata.get(key) == value, f"{arm}: metadata drift at {key}")
    config = path.parents[2] / ".hydra/config.yaml"
    need(config.is_file() and sha256_file(config) == metadata["config_sha256"], f"{arm}: config SHA drift")
    return metadata


def verify_score(value: Mapping[str, Any], label: str) -> None:
    need(value["query_window_indices_sha256"] == QUERY_SHA, f"{label}: query SHA mismatch")
    need(value["samples"] == 13107 and value["batches"] == 410 and value["last_batch_size"] == 19 and value["batch_size"] == 32,
         f"{label}: target batch accounting mismatch")
    need(value["state_immutable"] is True and value["state_sha256_before"] == value["state_sha256_after"],
         f"{label}: model changed while scoring")
    need(set(value["per_session"]) == set(TARGET_SAMPLES) and math.isfinite(float(value["pooled_r2"])), f"{label}: score roster/R2 invalid")
    for session, samples in TARGET_SAMPLES.items():
        row = value["per_session"][session]
        need(row["samples"] == samples and math.isfinite(float(row["r2"])), f"{label}/{session}: invalid score")


def verify(receipt_path: Path) -> dict[str, Any]:
    receipt_path = receipt_path.resolve()
    need(receipt_path.is_file() and stat.S_IMODE(receipt_path.stat().st_mode) == 0o444,
         "terminal receipt must be a mode-0444 regular file")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    need(receipt.get("schema") == SCHEMA and receipt.get("outer_date") == OUTER_DATE and receipt.get("seed") == 42
         and receipt.get("support_trials") == 4, "terminal receipt identity mismatch")
    evaluator = receipt.get("evaluator")
    need(isinstance(evaluator, Mapping), "evaluator binding absent")
    evaluator_path = Path(evaluator["path"]).resolve()
    need(evaluator_path.is_file() and sha256_file(evaluator_path) == evaluator["sha256"], "terminal evaluator code drift")
    source_entry = receipt.get("source_snapshot")
    need(isinstance(source_entry, Mapping), "source snapshot binding absent")
    snapshot = load_snapshot(Path(source_entry["receipt"]))
    need(
        snapshot.receipt_sha256 == source_entry["receipt_sha256"]
        and snapshot.snapshot_sha256 == source_entry["snapshot_sha256"]
        and snapshot.manifest_sha256 == source_entry["manifest_sha256"]
        and snapshot.basis.basis_sha256 == source_entry["basis_sha256"]
        and snapshot.normalizer.normalizer_sha256 == source_entry["normalizer_sha256"],
        "immutable source snapshot binding drift",
    )
    training_source_manifest_sha256 = str(source_entry.get("training_source_manifest_sha256", ""))
    need(len(training_source_manifest_sha256) == 64, "training source manifest binding absent")
    checkpoints = receipt["checkpoints"]
    full_meta = verify_checkpoint(checkpoints["full"], "full", snapshot, training_source_manifest_sha256)
    zero_meta = verify_checkpoint(checkpoints["zero5"], "zero", snapshot, training_source_manifest_sha256)
    for field in PAIR_FIELDS:
        need(full_meta[field] == zero_meta[field] == checkpoints["pair_binding"][field], f"checkpoint pair drift at {field}")
    target = receipt["target"]
    need(tuple(target["sessions"]) == tuple(TARGET_SAMPLES) and target["query_window_indices_sha256"] == QUERY_SHA
         and target["post_four_trial_query"] is True and target["target_samples"] == TARGET_SAMPLES,
         "target contract drift")
    supports = target["support_and_carrier_hashes"]
    need(set(supports) == set(TARGET_SAMPLES), "support session roster mismatch")
    for session, item in supports.items():
        need(len(item["trial_values"]) == 4 and math.isfinite(float(item["fifth_trial"])) and int(item["query_first_bin"]) >= 0,
             f"{session}: support boundary mismatch")
        hashes = item["carrier_sha256"]
        need(set(hashes) == set(INTERVENTIONS) and len(set(hashes.values())) == 4 and all(len(value) == 64 for value in hashes.values()),
             f"{session}: carrier intervention hash mismatch")
    metrics = receipt["metrics"]
    same = metrics["full_same_checkpoint_interventions"]
    need(set(same) == set(INTERVENTIONS), "same-checkpoint intervention roster mismatch")
    for arm in INTERVENTIONS:
        verify_score(same[arm], f"Full/{arm}")
    independent = metrics["independently_trained_zero5"]
    verify_score(independent, "independent Zero5")
    full_state = same["full"]["state_sha256_before"]
    need(all(same[arm]["state_sha256_before"] == full_state for arm in INTERVENTIONS),
         "same-checkpoint controls use different model state")
    margins = receipt["gate"]["margins"]
    full_r2 = float(same["full"]["pooled_r2"])
    expected_margins = {
        "full_minus_independently_trained_zero5_pooled": full_r2 - float(independent["pooled_r2"]),
        "full_minus_same_checkpoint_zero5_pooled": full_r2 - float(same["zero"]["pooled_r2"]),
        "full_minus_same_checkpoint_row_shuffle_pooled": full_r2 - float(same["row"]["pooled_r2"]),
        "full_minus_same_checkpoint_endpoint_label_shuffle_pooled": full_r2 - float(same["label"]["pooled_r2"]),
    }
    for name, expected in expected_margins.items():
        exact(margins[name], expected, name)
    observed_recording_delta = receipt["gate"]["per_recording_full_minus_independently_trained_zero5"]
    expected_recording_delta = {
        session: float(same["full"]["per_session"][session]["r2"]) - float(independent["per_session"][session]["r2"])
        for session in TARGET_SAMPLES
    }
    for session, expected in expected_recording_delta.items():
        exact(observed_recording_delta[session], expected, f"per-recording delta {session}")
    expected_core = {
        "full_minus_independently_trained_zero5_pooled_positive": expected_margins["full_minus_independently_trained_zero5_pooled"] > 0.0,
        "all_three_recording_full_minus_independently_trained_zero5_positive": all(value > 0.0 for value in expected_recording_delta.values()),
    }
    expected_diagnostics = {
        "full_minus_same_checkpoint_zero5_pooled_positive": expected_margins["full_minus_same_checkpoint_zero5_pooled"] > 0.0,
        "full_minus_same_checkpoint_row_shuffle_pooled_positive": expected_margins["full_minus_same_checkpoint_row_shuffle_pooled"] > 0.0,
        "full_minus_same_checkpoint_endpoint_label_shuffle_pooled_positive": expected_margins["full_minus_same_checkpoint_endpoint_label_shuffle_pooled"] > 0.0,
    }
    gate = receipt["gate"]
    need(gate["predeclared_core"] == expected_core and gate["same_checkpoint_diagnostics"] == expected_diagnostics,
         "terminal gate arithmetic mismatch")
    need(gate["core_pass"] == all(expected_core.values()) and gate["diagnostic_pass"] == all(expected_diagnostics.values()),
         "terminal gate boolean mismatch")
    need(receipt["status"] == ("PASS_HSE5_LODO_DATE2_CORE_GATE" if gate["core_pass"] else "STOP_HSE5_LODO_DATE2_CORE_GATE"),
         "terminal status mismatch")
    need(receipt["scope"] == {
        "public_heldin_calibration_nwbs_opened": 13, "target_optimizer_steps": 0,
        "target_backward_steps": 0, "minival_opened": False, "heldout_opened": False,
        "formal_opened": False, "evalai_opened": False,
    }, "target evaluation scope drift")
    return {
        "schema": "h1_hse5_lodo_date2_terminal_receipt_structural_verifier_v1", "status": "PASS",
        "receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
        "scientific_status": receipt["status"], "core_gate_pass": gate["core_pass"],
        "diagnostic_pass": gate["diagnostic_pass"], "margins": expected_margins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    body = verify(args.receipt)
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        need(not output.exists(), f"refusing to overwrite verifier artifact: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        output.chmod(0o444)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
