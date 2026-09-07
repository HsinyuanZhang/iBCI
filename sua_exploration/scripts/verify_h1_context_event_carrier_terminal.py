#!/usr/bin/env python3
"""Fail-closed structural verifier for the Context Full terminal receipt.

This module deliberately does *not* import ``h1_context_event_carrier_evaluate``.
It validates a terminal execution after the fact: immutable receipt/source assets,
the current evaluator code binding, full and common-Zero checkpoints/configs, and
all reported control and outcome-gate arithmetic.  Prediction-level R2 is kept in
the separately implemented batch-29 audit.
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

import torch


ROOT = Path(__file__).resolve().parents[2]
SPINT = ROOT / "SPINT-main"
if str(SPINT) not in sys.path:
    sys.path.insert(0, str(SPINT))


SCHEMA = "h1_context_event_carrier_m4_fold0_terminal_v2"
PREFLIGHT_SCHEMA = "h1_context_event_carrier_m4_fold0_cpu_preflight_v2"
CHECKPOINT_SCHEMA = "h1_sparse_event_endpoint_h32_fold0_checkpoint_v1"
QUERY_SHA = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
TARGET_SAMPLES = {"ses-19250101T111740": 6735, "ses-19250101T112404": 2230}
INTERVENTIONS = ("full", "zero", "row", "label", "tag", "midpoint")
REQUIRED_CLAUSES = (
    "common_independent_zero", "same_checkpoint_zero", "row_shuffle",
    "endpoint_label_shuffle", "tag_shuffle", "label_two_of_two", "tag_two_of_two",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_json(path: Path, label: str) -> Mapping[str, Any]:
    path = path.resolve()
    need(path.is_file() and not path.is_symlink(), f"missing regular {label}: {path}")
    need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} is not immutable mode 0444")
    return json.loads(path.read_text(encoding="utf-8"))


def close(first: Any, second: Any, label: str, *, tolerance: float = 1.0e-12) -> None:
    need(math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=tolerance),
         f"{label}: {first!r} != {second!r}")


def _file_entry(entry: Mapping[str, Any], key: str, label: str) -> Path:
    path = Path(entry[key]).resolve()
    need(path.is_file() and not path.is_symlink(), f"{label}: missing regular file {path}")
    return path


def _config_for_checkpoint(entry: Mapping[str, Any], arm: str) -> Path:
    if "resolved_hydra_config" in entry:
        path = _file_entry(entry, "resolved_hydra_config", f"{arm} resolved config")
        need(sha256(path) == entry.get("resolved_hydra_config_sha256"), f"{arm}: receipt config SHA mismatch")
        return path
    checkpoint = _file_entry(entry, "path", f"{arm} checkpoint")
    path = checkpoint.parents[2] / ".hydra/config.yaml"
    need(path.is_file() and not path.is_symlink(), f"{arm}: resolved config is missing beside checkpoint")
    return path


def verify_checkpoint(entry: Mapping[str, Any], arm: str, *, expected_manifest: str, common_zero: bool) -> dict[str, Any]:
    path = _file_entry(entry, "path", f"{arm} checkpoint")
    need(sha256(path) == entry.get("sha256"), f"{arm}: checkpoint SHA mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("h1_sparse_event_endpoint")
    need(isinstance(metadata, Mapping) and metadata == entry.get("metadata"),
         f"{arm}: embedded metadata differs from receipt")
    need(checkpoint.get("epoch") == 49 and int(checkpoint.get("global_step", 0)) > 0,
         f"{arm}: checkpoint is not terminal epoch 49")
    expected = {
        "schema": CHECKPOINT_SCHEMA, "arm": arm, "fold_date": "19250101",
        "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_selection", "carrier_dim": 5,
        "carrier_hidden_dim": 32, "carrier_trial_length": 1024,
        "target_session_optimizer_steps": 0, "target_session_backward_steps": 0,
    }
    for key, value in expected.items():
        need(metadata.get(key) == value, f"{arm}: metadata drift at {key}")
    need(metadata.get("source_manifest_sha256") == expected_manifest,
         f"{arm}: context source manifest mismatch")
    config = _config_for_checkpoint(entry, arm)
    need(sha256(config) == metadata.get("config_sha256"), f"{arm}: checkpoint config SHA mismatch")
    if common_zero:
        need(metadata.get("carrier_mode") == "literal_zero5_at_model_boundary",
             "common H-SE5 Zero does not use literal zero boundary")
    else:
        need(metadata.get("carrier_mode") != "literal_zero5_at_model_boundary",
             "Context Full unexpectedly uses literal zero boundary")
    return {"checkpoint": str(path), "checkpoint_sha256": sha256(path),
            "config": str(config), "config_sha256": sha256(config), "metadata": dict(metadata)}


def verify_score(score: Mapping[str, Any], label: str) -> None:
    need(score.get("query_window_indices_sha256") == QUERY_SHA, f"{label}: query SHA mismatch")
    need(score.get("samples") == sum(TARGET_SAMPLES.values()), f"{label}: sample count mismatch")
    need(score.get("state_immutable") is True and score.get("state_sha256_before") == score.get("state_sha256_after"),
         f"{label}: model state changed")
    sessions = score.get("per_session")
    need(isinstance(sessions, Mapping) and set(sessions) == set(TARGET_SAMPLES), f"{label}: session roster mismatch")
    need(math.isfinite(float(score.get("pooled_r2"))), f"{label}: pooled R2 nonfinite")
    for name, count in TARGET_SAMPLES.items():
        row = sessions[name]
        need(row.get("samples") == count and math.isfinite(float(row.get("r2"))),
             f"{label}/{name}: invalid R2/sample accounting")


def verify(receipt_path: Path) -> dict[str, Any]:
    receipt_path = receipt_path.resolve()
    body = immutable_json(receipt_path, "terminal receipt")
    need(body.get("schema") == SCHEMA and body.get("fold_date") == "19250101" and body.get("seed") == 42,
         "terminal identity/schema mismatch")
    need(body.get("checkpoint_binding_completed_before_target_open") is True,
         "checkpoint binding was not completed before target open")

    evaluator = body.get("evaluator")
    need(isinstance(evaluator, Mapping), "current terminal evaluator SHA binding is missing")
    evaluator_path = _file_entry(evaluator, "path", "terminal evaluator")
    need(evaluator_path == (SPINT / "scripts/h1_context_event_carrier_evaluate.py").resolve(),
         "terminal evaluator path is not the Context evaluator")
    need(sha256(evaluator_path) == evaluator.get("sha256"), "terminal evaluator SHA drift")

    preflight_entry = body.get("common_zero_parity_preflight")
    need(isinstance(preflight_entry, Mapping), "v3 common-Zero preflight binding is missing")
    preflight_path = _file_entry(preflight_entry, "path", "v3 preflight")
    need(preflight_path.name == "H1_CONTEXT_SER_Q4_M4_FOLD0_PREFLIGHT_v3.json", "preflight is not v3")
    need(sha256(preflight_path) == preflight_entry.get("sha256"), "v3 preflight SHA mismatch")
    preflight = immutable_json(preflight_path, "v3 preflight")
    need(preflight.get("schema") == PREFLIGHT_SCHEMA and preflight.get("status") == "READY_CONTEXT_FULL_COMMON_HSE5_ZERO",
         "v3 preflight is not a ready common-Zero authorization")
    parity = preflight.get("noncarrier_zero_parity", {})
    need(parity.get("reusable_common_hse5_zero5") is True and all(parity.get("checks", {}).values()),
         "v3 preflight common-Zero checks are not all true")
    launch_config = preflight.get("configs", {}).get("context_full", {})
    launch_config_path = _file_entry(launch_config, "path", "v3 Context Full launch config")
    need(sha256(launch_config_path) == launch_config.get("sha256"), "v3 Context Full launch config SHA drift")

    snapshot_entry = body.get("source_snapshot")
    need(isinstance(snapshot_entry, Mapping), "v3 source snapshot binding is missing")
    snapshot_receipt = _file_entry(snapshot_entry, "receipt", "v3 source snapshot receipt")
    need(snapshot_receipt.name == "H1_CONTEXT_SER_Q4_FOLD0_SOURCE_v3.json", "source snapshot receipt is not v3")
    need(sha256(snapshot_receipt) == snapshot_entry.get("sha256"), "terminal source receipt SHA mismatch")
    snapshot = immutable_json(snapshot_receipt, "v3 source snapshot receipt")
    need(snapshot.get("schema") == "h1_context_event_source_snapshot_receipt_v1" and
         snapshot.get("snapshot_schema") == "h1_context_event_source_snapshot_v1", "v3 source snapshot receipt schema mismatch")
    snapshot_path = _file_entry(snapshot.get("snapshot", {}), "path", "v3 source snapshot")
    need(snapshot_path.name == "H1_CONTEXT_SER_Q4_FOLD0_SOURCE_v3.npz" and stat.S_IMODE(snapshot_path.stat().st_mode) == 0o444,
         "source snapshot asset is not immutable v3")
    need(sha256(snapshot_path) == snapshot.get("snapshot", {}).get("sha256"), "source snapshot asset SHA mismatch")
    need(preflight["source_snapshot"]["receipt_sha256"] == sha256(snapshot_receipt), "preflight/source receipt mismatch")
    need(preflight["source_snapshot"]["snapshot_sha256"] == sha256(snapshot_path), "preflight/source asset mismatch")
    manifest_sha = str(preflight["source_manifest_sha256"])
    need(snapshot.get("source_manifest_sha256") == manifest_sha and snapshot.get("expected_manifest_sha256") == manifest_sha,
         "v3 source snapshot receipt manifest binding mismatch")
    need(body.get("source_manifest_sha256") == manifest_sha, "terminal/preflight context manifest mismatch")
    need(body.get("source_manifest") == preflight.get("source_manifest"), "terminal context manifest is not the preflight manifest")

    checkpoints = body.get("checkpoints")
    need(isinstance(checkpoints, Mapping) and set(checkpoints) == {"full", "common_hse5_zero5"},
         "checkpoint roster mismatch")
    full = verify_checkpoint(checkpoints["full"], "full", expected_manifest=manifest_sha, common_zero=False)
    zero = verify_checkpoint(checkpoints["common_hse5_zero5"], "zero", expected_manifest=str(preflight["checkpoints"]["sealed_hse5_zero"]["metadata"]["source_manifest_sha256"]), common_zero=True)
    sealed_zero = preflight["checkpoints"]["sealed_hse5_zero"]
    need(zero["checkpoint_sha256"] == sealed_zero["sha256"] and zero["metadata"] == sealed_zero["metadata"],
         "common Zero does not equal sealed H-SE5 Zero checkpoint")

    sealed = body.get("sealed_hse5")
    need(isinstance(sealed, Mapping), "sealed H-SE5 reference is missing")
    sealed_path = _file_entry(sealed, "receipt", "sealed H-SE5 terminal")
    need(stat.S_IMODE(sealed_path.stat().st_mode) == 0o444 and sha256(sealed_path) == sealed.get("sha256"),
         "sealed H-SE5 terminal binding mismatch")
    sealed_body = immutable_json(sealed_path, "sealed H-SE5 terminal")
    hse_full = sealed_body["metrics"]["hse5_same_checkpoint_interventions"]["full"]
    close(sealed.get("full_pooled_r2"), hse_full["pooled_r2"], "sealed H-SE5 Full R2")
    need(hse_full["query_window_indices_sha256"] == QUERY_SHA, "sealed H-SE5 query SHA mismatch")

    target = body.get("target")
    need(isinstance(target, Mapping) and target.get("query_window_indices_sha256") == QUERY_SHA and target.get("post_four_trial_query") is True,
         "terminal target query boundary mismatch")
    supports = target.get("support_and_carrier_hashes")
    need(isinstance(supports, Mapping) and set(supports) == set(TARGET_SAMPLES), "target support roster mismatch")
    for name, support in supports.items():
        hashes = support.get("carrier_sha256", {})
        need(len(support.get("trial_values", [])) == 4 and math.isfinite(float(support.get("fifth_trial"))),
             f"{name}: target support boundary mismatch")
        need(set(hashes) == set(INTERVENTIONS) and len(set(hashes.values())) == len(INTERVENTIONS),
             f"{name}: context intervention carrier binding mismatch")

    metrics = body.get("metrics")
    scores = metrics.get("context_same_checkpoint_interventions", {})
    need(set(scores) == set(INTERVENTIONS), "same-checkpoint intervention score roster mismatch")
    for name in INTERVENTIONS:
        verify_score(scores[name], f"Context/{name}")
    common = metrics.get("common_independent_hse5_zero5")
    need(isinstance(common, Mapping), "common independent Zero score missing")
    verify_score(common, "Context/common-HSE5-Zero")
    full_state = scores["full"]["state_sha256_before"]
    need(all(scores[name]["state_sha256_before"] == full_state for name in INTERVENTIONS),
         "same-checkpoint controls do not bind one Full model state")

    candidate = float(scores["full"]["pooled_r2"])
    expected_margins = {
        "context_full_minus_sealed_hse5_full": candidate - float(sealed["full_pooled_r2"]),
        "full_minus_same_checkpoint_zero": candidate - float(scores["zero"]["pooled_r2"]),
        "full_minus_row_shuffle": candidate - float(scores["row"]["pooled_r2"]),
        "full_minus_endpoint_label_shuffle": candidate - float(scores["label"]["pooled_r2"]),
        "full_minus_tag_shuffle": candidate - float(scores["tag"]["pooled_r2"]),
        "context_full_minus_common_independent_zero5": candidate - float(common["pooled_r2"]),
    }
    for key, value in expected_margins.items():
        close(body["margins"][key], value, key)
    session_deltas = {name: float(scores["full"]["per_session"][name]["r2"]) - float(hse_full["per_session"][name]["r2"])
                      for name in TARGET_SAMPLES}
    need(sealed.get("per_recording_delta") == body["sealed_hse5"].get("per_recording_delta"), "unreachable sealed delta binding")
    for name, value in session_deltas.items():
        close(sealed["per_recording_delta"][name], value, f"H-SE5 session delta {name}")
    clauses = {
        "common_independent_zero": expected_margins["context_full_minus_common_independent_zero5"] > 0,
        "same_checkpoint_zero": expected_margins["full_minus_same_checkpoint_zero"] > 0,
        "row_shuffle": expected_margins["full_minus_row_shuffle"] > 0,
        "endpoint_label_shuffle": expected_margins["full_minus_endpoint_label_shuffle"] > 0,
        "tag_shuffle": expected_margins["full_minus_tag_shuffle"] > 0,
        "label_two_of_two": all(float(scores["full"]["per_session"][n]["r2"]) > float(scores["label"]["per_session"][n]["r2"]) for n in TARGET_SAMPLES),
        "tag_two_of_two": all(float(scores["full"]["per_session"][n]["r2"]) > float(scores["tag"]["per_session"][n]["r2"]) for n in TARGET_SAMPLES),
    }
    need(set(body["gate"].get("clauses", {})) == set(REQUIRED_CLAUSES) and body["gate"]["clauses"] == clauses,
         "control clause arithmetic mismatch")
    signs = all(value > 0 for value in session_deltas.values())
    controls = all(clauses.values())
    primary = expected_margins["context_full_minus_sealed_hse5_full"]
    expected_status = ("PASS_CONTEXT_MATERIAL" if controls and primary >= 0.010 and signs else
                       "SMALL_POSITIVE_NONMATERIAL" if controls and 0 < primary < 0.010 and signs else
                       "STOP_CONTEXT")
    need(body["gate"]["primary_material_translation"] == {"pooled_minimum": 0.010, "both_recordings_positive": signs},
         "primary gate arithmetic mismatch")
    need(body.get("status") == expected_status and body["gate"].get("pass") == (expected_status == "PASS_CONTEXT_MATERIAL"),
         "terminal scientific status mismatch")
    need(body.get("scope") == {"target_optimizer_steps": 0, "target_backward_steps": 0,
                                "dense_velocity_used_for_carrier": False}, "terminal scope mismatch")
    return {"status": "PASS", "receipt": str(receipt_path), "sha256": sha256(receipt_path),
            "scientific_status": expected_status, "context_full_r2": candidate,
            "common_hse5_zero_r2": float(common["pooled_r2"]), "margins": expected_margins,
            "v3_preflight_sha256": sha256(preflight_path), "v3_snapshot_receipt_sha256": sha256(snapshot_receipt),
            "v3_snapshot_sha256": sha256(snapshot_path), "evaluator_sha256": sha256(evaluator_path),
            "full_checkpoint": full, "common_zero_checkpoint": zero}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="immutable Context Full terminal receipt")
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
