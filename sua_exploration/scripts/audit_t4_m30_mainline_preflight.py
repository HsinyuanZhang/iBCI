#!/usr/bin/env python3
"""Fail-closed, CPU-only provenance audit for Experiment A's M30 reference arms.

This audit deliberately does *not* import torch, open an NWB file, load a checkpoint, or
invoke an evaluator.  It binds the nine reused-development result artifacts to their run
metadata, the strict manifest, teacher, and the scorer source that is presently available.
It emits an immutable receipt only when the requested output directory does not yet exist.

It is intentionally an eligibility audit, not a scorer rerun: a historical scorer trace can
only be accepted when it is already present in the qualified artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "sua_exploration/results/sua_spint_t4_mainline_fp32_v1"
CHECKPOINT_ROOT = REPO_ROOT / "sua_exploration/checkpoints"
MANIFEST_PATH = REPO_ROOT / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
TEACHER_PATH = REPO_ROOT / "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
EPOCH_SCORER_PATH = REPO_ROOT / "sua_exploration/scripts/eval_epoch_window_dandi688.py"
SESSION_SCORER_PATH = REPO_ROOT / "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py"

SEEDS = (42, 43, 44)
ARMS = ("B0", "T4", "TS4")
EXPECTED_ARTIFACT_SHA256 = {
    ("B0", 42): "24477be67d6855ccaadaf64f2651e949e16a2ff972afd821d881a2f6d3f325fd",
    ("B0", 43): "8cd4872c6419145a8b5c349207d42d7868ae80b4c72e607d33a00e9fd98031ee",
    ("B0", 44): "415dde1d40106ccb0d5481fa413dda123bc1e99f9a3a2464c48475befc0a5f90",
    ("T4", 42): "b8f659a46ad55eea766cbad1be70e1cc99df4c3c4c6c38863f2a5a5ee3104148",
    ("T4", 43): "e18a52a750b44f426ba1e39f3a8f791806f53228e5a8a69b73a491178b8a09b4",
    ("T4", 44): "704f5a40bb07e53dc3267d4ed70e434255c84cb9e9b60dcfcb8dd7dcecac3e64",
    ("TS4", 42): "0178e384eb8976b931b0fcc47ce53354e84500a65672ee51fd4d9c19ab41c841",
    ("TS4", 43): "ca146f795aa342560a1220c7a6e3218f4370f10493bbbb76f7e808a1591936ba",
    ("TS4", 44): "e697e870d40efe5357b9c3ea410eb286fdd2d537e0134007b2b33f19d6641f7d",
}
EXPECTED_TEACHER_SHA256 = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"
EXPECTED_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
EXPECTED_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
EXPECTED_EPOCHS = list(range(5, 13))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def run_stem(arm: str, seed: int) -> str:
    return f"{arm.lower()}_s{seed}"


def metadata_path_for(checkpoint_root: Path, arm: str, seed: int) -> Path:
    return checkpoint_root / f"sua_spint_t4_mainline_fp32_v1_{arm.lower()}_dandi688_co_s{seed}" / "run_metadata.json"


def artifact_path_for(result_root: Path, arm: str, seed: int) -> Path:
    return result_root / f"{arm.lower()}_s{seed}.json"


def require_equal(failures: list[str], label: str, actual: Any, expected: Any) -> None:
    add_failure(failures, actual == expected, f"{label}: expected {expected!r}, found {actual!r}")


def scorer_static_audit(epoch_scorer_path: Path, session_scorer_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Record the current source semantics without claiming they are historical execution."""
    failures: list[str] = []
    epoch_text = epoch_scorer_path.read_text(encoding="utf-8") if epoch_scorer_path.is_file() else ""
    session_text = session_scorer_path.read_text(encoding="utf-8") if session_scorer_path.is_file() else ""
    fixed_pool = re.search(r"^FIXED_POOL_SIZE\s*=\s*(\d+)\s*$", epoch_text, re.MULTILINE)
    fixed_calibration = re.search(r"^FIXED_CALIBRATION_N\s*=\s*(\d+)\s*$", epoch_text, re.MULTILINE)
    pool = int(fixed_pool.group(1)) if fixed_pool else None
    calibration_n = int(fixed_calibration.group(1)) if fixed_calibration else None
    scorer_has_trace_return = '"trial_selections": selections' in session_text
    scorer_uses_pool_for_score_start = "eval_trials = rec[\"trials\"][pool_size:]" in session_text
    epoch_source_ok = epoch_scorer_path.is_file()
    session_source_ok = session_scorer_path.is_file()
    add_failure(failures, epoch_source_ok and session_source_ok, "scorer source missing; scorer semantics cannot be audited")
    add_failure(failures, calibration_n == 30, f"current epoch scorer calibration_n is not M30: {calibration_n!r}")
    # This does not invalidate the source in general.  It prevents assigning a 30-pool score
    # to a historical artifact when the supplied current scorer instead uses 50.
    add_failure(
        failures,
        pool == 30,
        "current epoch scorer hard-codes FIXED_POOL_SIZE=50, conflicting with the M30 artifact pool_size=30",
    )
    add_failure(failures, scorer_uses_pool_for_score_start, "current session scorer no longer exposes score start as trials[pool_size:]")
    return {
        "epoch_scorer": {
            "path": str(epoch_scorer_path),
            "sha256": sha256_file(epoch_scorer_path) if epoch_source_ok else None,
            "fixed_calibration_n": calibration_n,
            "fixed_pool_size": pool,
        },
        "session_scorer": {
            "path": str(session_scorer_path),
            "sha256": sha256_file(session_scorer_path) if session_source_ok else None,
            "score_start_expression": "trials[pool_size:]" if scorer_uses_pool_for_score_start else None,
            "returns_trial_selections_when_run": scorer_has_trace_return,
        },
        "historical_trace_status": "unavailable_in_qualified_artifacts",
        "historical_actual_indices": None,
        "reason": (
            "The M30 artifacts record only aggregate R2 and no trial_selections/actual scorer trace. "
            "The current scorer cannot be used as historical proof because its fixed pool is 50, not 30."
        ),
    }, failures


def audit_m30_mainline(
    *,
    result_root: Path = RESULT_ROOT,
    checkpoint_root: Path = CHECKPOINT_ROOT,
    manifest_path: Path = MANIFEST_PATH,
    teacher_path: Path = TEACHER_PATH,
    epoch_scorer_path: Path = EPOCH_SCORER_PATH,
    session_scorer_path: Path = SESSION_SCORER_PATH,
) -> dict[str, Any]:
    """Return a complete receipt.  It never opens test data, NWB, or checkpoints."""
    failures: list[str] = []
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    teacher_sha = sha256_file(teacher_path) if teacher_path.is_file() else None
    manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else None
    add_failure(failures, teacher_sha == EXPECTED_TEACHER_SHA256, "teacher SHA-256 mismatch or teacher absent")
    add_failure(failures, manifest_sha == EXPECTED_MANIFEST_SHA256, "strict manifest SHA-256 mismatch or manifest absent")
    if manifest is None:
        failures.append("strict manifest cannot be parsed")
        expected_splits: dict[str, Any] = {}
    else:
        expected_splits = manifest.get("session_splits", {})
        require_equal(failures, "manifest purpose", manifest.get("purpose"), "Strict validation-only SUA manifest: only train and validation NWBs may be opened.")
        require_equal(failures, "manifest split_counts", manifest.get("split_counts"), [27, 6, 6])
        require_equal(failures, "manifest task", manifest.get("task"), "CO")

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            label = f"{arm}@{seed}"
            artifact_path = artifact_path_for(result_root, arm, seed)
            metadata_path = metadata_path_for(checkpoint_root, arm, seed)
            artifact = read_json(artifact_path) if artifact_path.is_file() else None
            metadata = read_json(metadata_path) if metadata_path.is_file() else None
            artifact_sha = sha256_file(artifact_path) if artifact_path.is_file() else None
            metadata_sha = sha256_file(metadata_path) if metadata_path.is_file() else None
            add_failure(failures, artifact_sha == EXPECTED_ARTIFACT_SHA256[(arm, seed)], f"{label}: artifact SHA-256 does not match protocol lock")
            if artifact is None or metadata is None:
                failures.append(f"{label}: artifact or run_metadata is missing")
                rows.append({"arm": arm, "seed": seed, "artifact_path": str(artifact_path), "run_metadata_path": str(metadata_path), "artifact_sha256": artifact_sha, "run_metadata_sha256": metadata_sha})
                continue

            require_equal(failures, f"{label}: artifact seed", artifact.get("seed"), seed)
            require_equal(failures, f"{label}: artifact signal_view", artifact.get("signal_view"), "sua")
            require_equal(failures, f"{label}: artifact no_test_files_evaluated", artifact.get("no_test_files_evaluated"), True)
            require_equal(failures, f"{label}: recorded run_metadata SHA", artifact.get("run_metadata_sha256"), metadata_sha)
            require_equal(failures, f"{label}: artifact teacher SHA", artifact.get("teacher_ckpt_sha256"), EXPECTED_TEACHER_SHA256)
            require_equal(failures, f"{label}: artifact manifest SHA", artifact.get("train_val_manifest_sha256"), EXPECTED_MANIFEST_SHA256)
            require_equal(failures, f"{label}: metadata status", metadata.get("status"), "completed")
            require_equal(failures, f"{label}: metadata seed", metadata.get("seed"), seed)
            require_equal(failures, f"{label}: metadata signal_view", metadata.get("signal_view"), "sua")
            require_equal(failures, f"{label}: metadata held_out_test_evaluated", metadata.get("held_out_test_evaluated"), False)
            require_equal(failures, f"{label}: metadata test file list", (metadata.get("session_files") or {}).get("test"), [])
            require_equal(failures, f"{label}: metadata teacher SHA", metadata.get("teacher_sha256"), EXPECTED_TEACHER_SHA256)
            require_equal(failures, f"{label}: metadata manifest SHA", metadata.get("train_val_manifest_sha256"), EXPECTED_MANIFEST_SHA256)
            require_equal(failures, f"{label}: metadata session split order", metadata.get("session_splits"), expected_splits)
            require_equal(failures, f"{label}: artifact session split order", artifact.get("session_splits"), expected_splits)
            training = metadata.get("training") or {}
            protocol = artifact.get("protocol") or {}
            for field, expected in (("calibration_n_trials", 30), ("max_epochs", 12), ("no_early_stopping", True), ("checkpoint_every_epoch", True)):
                require_equal(failures, f"{label}: training.{field}", training.get(field), expected)
            for field, expected in (("calibration_n", 30), ("evaluation_forward_calibration_n", 30), ("train_activity_calibration_n", 30), ("pool_size", 30), ("epoch_window", EXPECTED_EPOCHS), ("total_epochs", 12), ("selection_mode", "first")):
                require_equal(failures, f"{label}: protocol.{field}", protocol.get(field), expected)
            require_equal(failures, f"{label}: epoch list", artifact.get("epoch_list"), EXPECTED_EPOCHS)
            require_equal(failures, f"{label}: fixed checkpoint rule", artifact.get("checkpoint_selection_rule"), "pre_declared_fixed_epoch_window_no_argmax")
            side = metadata.get("side_features") or {}
            expected_group = {"B0": "none", "T4": "t4", "TS4": "ts4"}[arm]
            expected_side_dim = 0 if arm == "B0" else 4
            require_equal(failures, f"{label}: side feature group", side.get("group"), expected_group)
            require_equal(failures, f"{label}: side_dim", side.get("side_dim"), expected_side_dim)
            if arm in {"T4", "TS4"}:
                require_equal(failures, f"{label}: train-only normalizer SHA", side.get("normalization_sha256"), EXPECTED_NORMALIZER_SHA256)
                require_equal(failures, f"{label}: side feature pool", side.get("pool_size"), 30)
            if arm == "TS4":
                require_equal(failures, f"{label}: TS4 permutation seed", side.get("permutation_seed"), seed)
            if arm in {"T4", "TS4"}:
                # A new Experiment-A reference must use the exact chronological 30-trial
                # label/rate support.  Reward-filtered support is a different protocol.
                require_equal(failures, f"{label}: label/rate feature scope", artifact.get("calibration_feature_label_scope"), "chronological_trials[0:30]")

            rows.append({
                "arm": arm,
                "seed": seed,
                "artifact_path": str(artifact_path),
                "artifact_sha256": artifact_sha,
                "artifact_sha256_matches_protocol_lock": artifact_sha == EXPECTED_ARTIFACT_SHA256[(arm, seed)],
                "run_metadata_path": str(metadata_path),
                "run_metadata_sha256": metadata_sha,
                "run_metadata_sha256_matches_artifact": artifact.get("run_metadata_sha256") == metadata_sha,
                "teacher_path": metadata.get("teacher_checkpoint"),
                "teacher_sha256": metadata.get("teacher_sha256"),
                "manifest_path": metadata.get("train_val_manifest"),
                "manifest_sha256": metadata.get("train_val_manifest_sha256"),
                "side_feature_group": side.get("group"),
                "side_dim": side.get("side_dim"),
                "normalization_sha256": side.get("normalization_sha256"),
                "permutation_seed": side.get("permutation_seed"),
                "declared_activity_calibration_n": training.get("calibration_n_trials"),
                "declared_feature_pool_n": side.get("pool_size") if arm != "B0" else protocol.get("pool_size"),
                "declared_score_start_n": protocol.get("pool_size"),
                "label_feature_scope": artifact.get("calibration_feature_label_scope"),
                "epoch_window": artifact.get("epoch_list"),
                "no_test_files_evaluated": artifact.get("no_test_files_evaluated"),
                "held_out_test_evaluated": metadata.get("held_out_test_evaluated"),
            })

    scorer, scorer_failures = scorer_static_audit(epoch_scorer_path, session_scorer_path)
    failures.extend(scorer_failures)
    # The preflight contract requires actual indices, not an inferred range.  Do not infer
    # them from M30 metadata after the scorer source has diverged.
    failures.append("historical M30 scorer trace has no recorded support/feature/scored trial indices")
    unique_failures = list(dict.fromkeys(failures))
    return {
        "schema_version": 1,
        "audit_name": "t4_m30_experiment_a_cpu_preflight",
        "scope": "DANDI 000688 sub-C/CO reused-development evidence only; formal SUA sessions unopened",
        "implementation_scope": {
            "descriptor_implementations_added": [],
            "ph4": {
                "implemented": False,
                "authorized_by_this_receipt": False,
                "next_required_step": "separate descriptor-contract audit with a source-only phase normalizer",
            },
        },
        "execution": {
            "cpu_only": True,
            "nwb_files_opened": [],
            "checkpoint_files_loaded": [],
            "formal_files_opened": False,
            "training_started": False,
            "gpu_used": False,
        },
        "locks": {
            "teacher_path": str(teacher_path),
            "teacher_sha256": teacher_sha,
            "expected_teacher_sha256": EXPECTED_TEACHER_SHA256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "manifest_session_splits": expected_splits,
            "expected_t4_ts4_normalizer_sha256": EXPECTED_NORMALIZER_SHA256,
        },
        "artifacts": rows,
        "scorer_trace": scorer,
        "eligibility": {
            "status": "pass" if not unique_failures else "fail",
            "experiment_a_m30_reference_qualified": not unique_failures,
            "may_proceed_to_descriptor_implementation": not unique_failures,
            "blockers": unique_failures,
        },
    }


def markdown_receipt(receipt: dict[str, Any]) -> str:
    eligibility = receipt["eligibility"]
    lines = [
        "# Experiment A M30 CPU preflight receipt",
        "",
        f"**Status:** `{eligibility['status'].upper()}` — descriptor implementation is {'authorized' if eligibility['may_proceed_to_descriptor_implementation'] else 'not authorized'}.",
        "",
        "## Scope",
        "",
        receipt["scope"],
        "",
        "No NWB, checkpoint, formal SUA, M1/M2, EvalAI, training, or GPU path was opened or run.",
        "This receipt adds no descriptor implementation. In particular, PH4 is not implemented or authorized here; it requires a separate source-only phase-normalizer descriptor-contract step.",
        "",
        "## Verified immutable identities",
        "",
        f"- Teacher SHA-256: `{receipt['locks']['teacher_sha256']}`",
        f"- Strict manifest SHA-256: `{receipt['locks']['manifest_sha256']}`",
        f"- Shared T4/TS4 normalizer SHA-256: `{receipt['locks']['expected_t4_ts4_normalizer_sha256']}`",
        "",
        "| Arm | Seed | Artifact SHA-256 | run_metadata SHA-256 | Feature group / side dim |",
        "|---|---:|---|---|---|",
    ]
    for row in receipt["artifacts"]:
        lines.append(f"| {row['arm']} | {row['seed']} | `{row.get('artifact_sha256')}` | `{row.get('run_metadata_sha256')}` | `{row.get('side_feature_group')}` / `{row.get('side_dim')}` |")
    lines += ["", "## M30 scorer boundary", ""]
    scorer = receipt["scorer_trace"]
    lines += [
        f"- Qualified artifact declaration: activity support `30`, T4 label/rate pool `30`, score start `30`, epoch window `5–12`.",
        f"- Historical actual support/feature/scored indices: **{scorer['historical_trace_status']}**.",
        f"- Current epoch scorer fixed pool: `{scorer['epoch_scorer']['fixed_pool_size']}`; fixed calibration n: `{scorer['epoch_scorer']['fixed_calibration_n']}`.",
        f"- {scorer['reason']}",
        "",
        "## Blocking findings",
        "",
    ]
    lines.extend(f"- {item}" for item in eligibility["blockers"])
    lines += ["", "A passing receipt requires a versioned historical scorer trace for every qualified seed, with explicit support, feature-fit, and scored original trial indices, and a TS4 seed-44 feature scope matching chronological `trials[0:30]`. "]
    return "\n".join(lines) + "\n"


def write_immutable_receipt(receipt: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing receipt directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    timestamped = {**receipt, "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    json_path = output_dir / "receipt.json"
    markdown_path = output_dir / "RECEIPT.md"
    json_path.write_text(json.dumps(timestamped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_receipt(timestamped), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New, non-existent immutable receipt directory.")
    args = parser.parse_args()
    receipt = audit_m30_mainline()
    write_immutable_receipt(receipt, args.output_dir)
    print(f"Wrote {args.output_dir / 'receipt.json'}")
    print(f"M30 Experiment A reference status: {receipt['eligibility']['status'].upper()}")
    return 0 if receipt["eligibility"]["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
