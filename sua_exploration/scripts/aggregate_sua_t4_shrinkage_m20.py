#!/usr/bin/env python3
"""Fail-closed aligned-first gate and aggregate for the M_T4=20 W3 branch.

This module is deliberately separate from the M15 pilot.  M20 is only a
pre-registered contingency for the narrow pattern in which W3 improves over
ordinary M15 T4 but cannot quite reach the frozen M50 reference.  It must not
reuse M15 artifacts or relax any of their provenance checks.

The aligned arm is evaluated first.  A shuffled control may be launched only
when :func:`check_aligned_first_gate` returns ``control_permitted=True``.  The
module contains no formal-test entrypoint and validates that formal data remain
sealed in every accepted validation artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sua_confidence_film_t4_budget import (  # noqa: E402
    _per_session_epoch_mean,
    hierarchical_ci,
    summarize,
)
from aggregate_sua_t4_shrinkage import (  # noqa: E402
    EPOCHS,
    EXPECTED_SHRINKAGE_RECEIPT,
    EXPECTED_VAL_SESSIONS,
    _load,
    _require,
    _sha256_file,
    _sha256_hex,
    noninferiority_summary,
    validate_arm as validate_m15_or_reference_arm,
)


M20_ARMS = ("t4w3_m20", "ts4w3_m20")
EXPECTED_GROUP = {
    "t4w3_m20": "t4w3",
    "ts4w3_m20": "ts4w3",
}
M20_POOL = 20
EVAL_POOL = 50
ACTIVITY_CALIBRATION = 30
SEED42_ONLY = (42,)
NONINFERIORITY_MARGIN = 0.03


def _require_m20_arm(arm: str) -> None:
    if arm not in M20_ARMS:
        raise ValueError(f"M20 branch only permits {M20_ARMS}, got {arm!r}")


def validate_m20_arm(
    path: Path,
    *,
    arm: str,
    seed: int,
) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """Validate one complete M20 W3 validation artifact without exceptions.

    This intentionally spells out the M20 contract instead of parameterizing
    the published M15 validator.  It prevents a pool-15 artifact from being
    relabelled or silently consumed by the contingency branch.
    """

    _require_m20_arm(arm)
    if seed not in SEED42_ONLY:
        raise ValueError(f"M20 aligned-first branch is seed-42 only, got {seed}")

    payload = _load(path)
    protocol = payload.get("protocol") or {}
    _require(f"{path}: schema", payload.get("schema_version"), 1)
    _require(
        f"{path}: purpose",
        payload.get("purpose"),
        "epoch_window_deterministic_checkpoint_selection",
    )
    _require(f"{path}: variant", payload.get("variant"), "B3S")
    _require(f"{path}: seed", payload.get("seed"), seed)
    _require(f"{path}: task", payload.get("task"), "CO")
    _require(f"{path}: signal", payload.get("signal_view"), "sua")
    _require(f"{path}: split", payload.get("split_counts"), [27, 6, 6])
    _require(f"{path}: max units", payload.get("max_units_exclusive"), 100)
    _require(f"{path}: epoch list", payload.get("epoch_list"), EPOCHS)
    _require(
        f"{path}: checkpoint selection",
        payload.get("checkpoint_selection_rule"),
        "pre_declared_fixed_epoch_window_no_argmax",
    )
    _require(f"{path}: no test", payload.get("no_test_files_evaluated"), True)
    _require(f"{path}: no eval backward", payload.get("uses_backward_gradients"), False)
    _require(
        f"{path}: no eval label updates",
        payload.get("uses_behavior_labels_for_weight_updates"),
        False,
    )
    _require(
        f"{path}: labelled W3 support",
        payload.get("calibration_features_use_behavior_labels"),
        True,
    )
    _require(
        f"{path}: chronological trial selection is label-free",
        payload.get("calibration_trial_selection_uses_behavior_labels"),
        False,
    )
    _require(f"{path}: total epochs", protocol.get("total_epochs"), 12)
    _require(f"{path}: burn-in", protocol.get("burn_in_epochs"), 4)
    _require(f"{path}: selection mode", protocol.get("selection_mode"), "first")
    _require(
        f"{path}: activity calibration",
        protocol.get("calibration_n"),
        ACTIVITY_CALIBRATION,
    )
    _require(
        f"{path}: evaluation-forward calibration",
        protocol.get("evaluation_forward_calibration_n"),
        ACTIVITY_CALIBRATION,
    )
    _require(
        f"{path}: train activity calibration",
        protocol.get("train_activity_calibration_n"),
        ACTIVITY_CALIBRATION,
    )
    _require(f"{path}: common evaluation start", protocol.get("pool_size"), EVAL_POOL)
    _require(
        f"{path}: labelled feature budget",
        protocol.get("label_feature_calibration_n"),
        M20_POOL,
    )
    _require(f"{path}: epoch window", protocol.get("epoch_window"), EPOCHS)
    _require(
        f"{path}: chronological label scope",
        payload.get("calibration_feature_label_scope"),
        "chronological_rewarded_trials[0:20]",
    )
    per_epoch = payload.get("per_epoch")
    if not isinstance(per_epoch, dict) or set(per_epoch) != {
        str(epoch) for epoch in EPOCHS
    }:
        raise ValueError(f"{path}: per_epoch must contain exactly epochs 5..12")

    metadata_path = Path(payload.get("run_metadata_path", ""))
    if not metadata_path.is_file():
        raise ValueError(f"{path}: missing run metadata {metadata_path}")
    _require(
        f"{path}: metadata SHA",
        payload.get("run_metadata_sha256"),
        _sha256_file(metadata_path),
    )
    metadata = _load(metadata_path)
    _require(f"{path}: metadata schema", metadata.get("schema_version"), 1)
    _require(f"{path}: metadata variant", metadata.get("variant"), "B3S")
    _require(f"{path}: metadata seed", metadata.get("seed"), seed)
    _require(f"{path}: metadata task", metadata.get("task"), "CO")
    _require(f"{path}: metadata signal", metadata.get("signal_view"), "sua")
    _require(f"{path}: metadata split", metadata.get("split_counts"), [27, 6, 6])
    _require(f"{path}: metadata units", metadata.get("max_units_exclusive"), 100)
    side = metadata.get("side_features") or {}
    _require(f"{path}: side group", side.get("group"), EXPECTED_GROUP[arm])
    _require(f"{path}: feature version", side.get("feature_version"), 1)
    _require(f"{path}: side pool", side.get("pool_size"), M20_POOL)
    _require(f"{path}: side dimension", side.get("side_dim"), 4)
    _require(f"{path}: no electrode embedding", side.get("electrode_embed_dim"), 0)
    _require(f"{path}: no electrode vocabulary", side.get("num_electrodes"), 0)
    _require(
        f"{path}: no electrode relation",
        side.get("uses_equality_only_relation_membership"),
        False,
    )
    normalization_sha = _sha256_hex(
        f"{path}: normalization SHA", side.get("normalization_sha256")
    )
    _require(
        f"{path}: frozen shrinkage receipt",
        side.get("shrinkage"),
        EXPECTED_SHRINKAGE_RECEIPT,
    )
    expected_permutation = seed if arm == "ts4w3_m20" else None
    _require(
        f"{path}: permutation receipt", side.get("permutation_seed"), expected_permutation
    )

    training = metadata.get("training") or {}
    _require(
        f"{path}: train activity budget",
        training.get("calibration_n_trials"),
        ACTIVITY_CALIBRATION,
    )
    _require(f"{path}: epochs", training.get("max_epochs"), 12)
    _require(f"{path}: no early stop", training.get("no_early_stopping"), True)
    _require(
        f"{path}: checkpoint each epoch", training.get("checkpoint_every_epoch"), True
    )
    _require(f"{path}: batch size", training.get("batch_size"), 32)
    _require(f"{path}: loss", training.get("loss_mode"), "task_only")
    _require(f"{path}: identity", training.get("identity_mode"), "calibrated")
    _require(f"{path}: decoder trainable", training.get("freeze_decoder"), False)
    _require(
        f"{path}: encoder base trainable", training.get("freeze_encoder_base"), False
    )
    _require(f"{path}: completed", metadata.get("status"), "completed")
    _require(f"{path}: formal unopened", metadata.get("held_out_test_evaluated"), False)
    _require(f"{path}: fresh fit", metadata.get("encoder_warmstart_path"), None)
    _require(
        f"{path}: coupled decoder",
        (metadata.get("decoder_architecture") or {}).get("mode"),
        "coupled",
    )
    _require(
        f"{path}: no fixed slots", (metadata.get("fixed_slot") or {}).get("enabled"), False
    )
    session_splits = metadata.get("session_splits") or {}
    _require(
        f"{path}: exact validation sessions",
        session_splits.get("val"),
        EXPECTED_VAL_SESSIONS,
    )
    fit_loader = metadata.get("trainer_fit_validation_loader_contract") or {}
    _require(
        f"{path}: fit loader excludes formal sessions",
        fit_loader.get("formal_test_sessions_loaded_during_fit"),
        False,
    )
    _require(
        f"{path}: fit validation sessions",
        fit_loader.get("loader_0_sessions"),
        EXPECTED_VAL_SESSIONS,
    )
    session_files = metadata.get("session_files") or {}
    _require(f"{path}: no formal files opened", session_files.get("test"), [])

    manifest_path = Path(metadata.get("train_val_manifest", ""))
    teacher_path = Path(metadata.get("teacher_checkpoint", ""))
    if not manifest_path.is_file():
        raise ValueError(f"{path}: strict manifest is missing")
    if not teacher_path.is_file():
        raise ValueError(f"{path}: teacher checkpoint is missing")
    manifest_sha = _sha256_hex(
        f"{path}: manifest SHA", metadata.get("train_val_manifest_sha256")
    )
    teacher_sha = _sha256_hex(f"{path}: teacher SHA", metadata.get("teacher_sha256"))
    _require(f"{path}: manifest bytes", manifest_sha, _sha256_file(manifest_path))
    _require(f"{path}: teacher bytes", teacher_sha, _sha256_file(teacher_path))

    sessions, values = _per_session_epoch_mean(payload, path)
    _require(f"{path}: validation score sessions", sessions, EXPECTED_VAL_SESSIONS)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite validation score")
    per_epoch_mean = payload.get("per_epoch_mean_r2") or {}
    if set(per_epoch_mean) != {str(epoch) for epoch in EPOCHS}:
        raise ValueError(f"{path}: per_epoch_mean_r2 must contain exactly epochs 5..12")
    epoch_means = np.asarray(
        [float(per_epoch_mean[str(epoch)]) for epoch in EPOCHS], dtype=np.float64
    )
    if not np.isfinite(epoch_means).all():
        raise ValueError(f"{path}: non-finite epoch mean")
    variant_score = payload.get("variant_score")
    if not isinstance(variant_score, (int, float)) or not np.isclose(
        float(variant_score), float(epoch_means.mean()), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(f"{path}: variant_score does not match fixed epoch mean")
    return sessions, values, {
        "metadata_path": str(metadata_path.resolve()),
        "manifest_sha256": manifest_sha,
        "teacher_sha256": teacher_sha,
        "normalization_sha256": normalization_sha,
    }


def _load_reference(
    reference_dir: Path, seed: int, sessions: list[str]
) -> tuple[np.ndarray, dict[str, str], Path]:
    reference_path = reference_dir / f"t4m50_s{seed}.json"
    ref_sessions, reference, receipt = validate_m15_or_reference_arm(
        reference_path, arm="t4_m50", seed=seed
    )
    _require(
        f"{reference_path}: M50 reference session matrix", ref_sessions, sessions
    )
    return reference, receipt, reference_path


def _require_reference_provenance(
    m20_receipt: dict[str, str], reference_receipt: dict[str, str], *, seed: int
) -> None:
    _require(
        f"seed {seed}: teacher checkpoint drift against frozen T4@50",
        m20_receipt["teacher_sha256"],
        reference_receipt["teacher_sha256"],
    )
    _require(
        f"seed {seed}: strict manifest drift against frozen T4@50",
        m20_receipt["manifest_sha256"],
        reference_receipt["manifest_sha256"],
    )


def check_m15_selection_gate(
    m15_result_dir: Path, reference_dir: Path, *, seed: int = 42
) -> dict[str, Any]:
    """Read-only, frozen selection rule for whether M20 may be attempted.

    M20 is not a freely selectable extra budget.  This gate verifies the
    complete M15 aligned W3, ordinary M15 T4, and frozen same-seed M50 T4
    receipts before applying the pre-registered narrow interval:

    ``d50 < -0.03 and d15 >= +0.015 and d50 >= -0.05``.

    Here ``d50 = T4W3@15 - T4@50`` and
    ``d15 = T4W3@15 - ordinary T4@15`` over the exact common validation
    session matrix.  No file is written and no M20 artifact is read or
    created.
    """

    if seed not in SEED42_ONLY:
        raise ValueError(f"M20 aligned-first branch is seed-42 only, got {seed}")
    aligned_path = m15_result_dir / f"t4w3_m15_s{seed}.json"
    ordinary_path = m15_result_dir / f"t4_m15_s{seed}.json"
    sessions, aligned, aligned_receipt = validate_m15_or_reference_arm(
        aligned_path, arm="t4w3_m15", seed=seed
    )
    ordinary_sessions, ordinary, ordinary_receipt = validate_m15_or_reference_arm(
        ordinary_path, arm="t4_m15", seed=seed
    )
    _require(
        f"{ordinary_path}: ordinary M15 session matrix", ordinary_sessions, sessions
    )
    reference, reference_receipt, reference_path = _load_reference(
        reference_dir, seed, sessions
    )
    _require_reference_provenance(aligned_receipt, reference_receipt, seed=seed)
    _require_reference_provenance(ordinary_receipt, reference_receipt, seed=seed)

    d50 = float((aligned - reference).mean())
    d15 = float((aligned - ordinary).mean())
    permitted = bool(d50 < -0.03 and d15 >= 0.015 and d50 >= -0.05)
    return {
        "schema_version": 1,
        "purpose": "frozen_m15_to_m20_w3_selection_gate",
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "protocol": {
            "seed": seed,
            "M15_activity": ACTIVITY_CALIBRATION,
            "M15_T4": 15,
            "M15_common_evaluation_start": EVAL_POOL,
            "M20_selection_bounds": {
                "d50_strictly_below": -0.03,
                "d50_lower_inclusive": -0.05,
                "d15_lower_inclusive": 0.015,
            },
            "formal_test_evaluated": False,
        },
        "artifacts": {
            "t4w3_m15": str(aligned_path.resolve()),
            "t4_m15": str(ordinary_path.resolve()),
            "t4_m50": str(reference_path.resolve()),
        },
        "provenance": {
            "t4w3_m15": aligned_receipt,
            "t4_m15": ordinary_receipt,
            "t4_m50": reference_receipt,
        },
        "arm_mean_r2": {
            "t4w3_m15": float(aligned.mean()),
            "t4_m15": float(ordinary.mean()),
            "t4_m50": float(reference.mean()),
        },
        "d50_t4w3_m15_minus_t4_m50": d50,
        "d15_t4w3_m15_minus_t4_m15": d15,
        "m20_permitted": permitted,
        "formal_effectiveness_eligible": False,
        "formal_effectiveness_pass": False,
    }


def check_aligned_first_gate(
    result_dir: Path, reference_dir: Path, *, seed: int = 42
) -> dict[str, Any]:
    """Read-only M20 aligned gate; no control is touched by this function."""

    if seed not in SEED42_ONLY:
        raise ValueError(f"M20 aligned-first branch is seed-42 only, got {seed}")
    aligned_path = result_dir / f"t4w3_m20_s{seed}.json"
    sessions, aligned, aligned_receipt = validate_m20_arm(
        aligned_path, arm="t4w3_m20", seed=seed
    )
    reference, reference_receipt, reference_path = _load_reference(
        reference_dir, seed, sessions
    )
    _require_reference_provenance(aligned_receipt, reference_receipt, seed=seed)
    noninferiority = noninferiority_summary(
        aligned[np.newaxis, :],
        reference[np.newaxis, :],
        seeds=(seed,),
        sessions=sessions,
        margin=NONINFERIORITY_MARGIN,
    )
    return {
        "schema_version": 1,
        "purpose": "m20_w3_aligned_first_noninferiority_gate",
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "protocol": {
            "seed": seed,
            "M_activity": ACTIVITY_CALIBRATION,
            "M_T4": M20_POOL,
            "common_evaluation_start": EVAL_POOL,
            "epochs": 12,
            "scored_epoch_window": EPOCHS,
            "formal_test_evaluated": False,
        },
        "artifacts": {
            "t4w3_m20": str(aligned_path.resolve()),
            "t4_m50": str(reference_path.resolve()),
        },
        "provenance": {
            "t4w3_m20": aligned_receipt,
            "t4_m50": reference_receipt,
        },
        "arm_mean_r2": {
            "t4w3_m20": float(aligned.mean()),
            "t4_m50": float(reference.mean()),
        },
        "noninferiority_vs_t4_m50": noninferiority,
        "control_permitted": bool(noninferiority["stage0_within_margin"]),
        "formal_effectiveness_eligible": False,
        "formal_effectiveness_pass": False,
    }


def aggregate_m20(
    result_dir: Path,
    reference_dir: Path,
    *,
    m15_result_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate M20 only after both frozen gates permit its control.

    Re-running the M15 selection gate here is intentional.  A full M20
    aggregate must carry the immutable reason that M20 was selected, rather
    than relying on a prior shell invocation whose provenance would otherwise
    be absent from the final artifact.
    """

    m15_selection_gate = check_m15_selection_gate(
        m15_result_dir, reference_dir, seed=seed
    )
    if not m15_selection_gate["m20_permitted"]:
        raise ValueError(
            "frozen M15-to-M20 selection gate did not permit this M20 aggregate"
        )

    aligned_gate = check_aligned_first_gate(result_dir, reference_dir, seed=seed)
    if not aligned_gate["control_permitted"]:
        raise ValueError(
            "aligned-first noninferiority gate failed; ts4w3_m20 control is not permitted"
        )

    sessions, aligned, aligned_receipt = validate_m20_arm(
        result_dir / f"t4w3_m20_s{seed}.json", arm="t4w3_m20", seed=seed
    )
    shuffled_path = result_dir / f"ts4w3_m20_s{seed}.json"
    shuffled_sessions, shuffled, shuffled_receipt = validate_m20_arm(
        shuffled_path, arm="ts4w3_m20", seed=seed
    )
    _require(f"{shuffled_path}: validation session matrix", shuffled_sessions, sessions)
    _require(
        f"seed {seed}: aligned/shuffled M20 W3 normalization",
        aligned_receipt["normalization_sha256"],
        shuffled_receipt["normalization_sha256"],
    )
    _require(
        f"seed {seed}: teacher checkpoint drift across M20 W3 arms",
        aligned_receipt["teacher_sha256"],
        shuffled_receipt["teacher_sha256"],
    )
    _require(
        f"seed {seed}: strict manifest drift across M20 W3 arms",
        aligned_receipt["manifest_sha256"],
        shuffled_receipt["manifest_sha256"],
    )
    content = summarize(
        aligned[np.newaxis, :], shuffled[np.newaxis, :], seeds=(seed,), sessions=sessions
    )
    result = {
        "schema_version": 1,
        "purpose": "m20_w3_aligned_first_paired_control_aggregate",
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "protocol": {
            "seed": seed,
            "M_activity": ACTIVITY_CALIBRATION,
            "M_T4": M20_POOL,
            "common_evaluation_start": EVAL_POOL,
            "epochs": 12,
            "scored_epoch_window": EPOCHS,
            "formal_test_evaluated": False,
        },
        "m15_selection_gate": m15_selection_gate,
        "aligned_first_gate": aligned_gate,
        "artifacts": {
            "t4w3_m20": str((result_dir / f"t4w3_m20_s{seed}.json").resolve()),
            "ts4w3_m20": str(shuffled_path.resolve()),
        },
        "provenance": {
            "t4w3_m20": aligned_receipt,
            "ts4w3_m20": shuffled_receipt,
        },
        "arm_mean_r2": {
            "t4w3_m20": float(aligned.mean()),
            "ts4w3_m20": float(shuffled.mean()),
        },
        "t4w3_m20_vs_ts4w3_m20": content,
        "stage0_descriptive_pass": bool(
            aligned_gate["control_permitted"]
            and content["passes_stage0_descriptive_gates"]
        ),
        "formal_effectiveness_eligible": False,
        "formal_effectiveness_pass": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument(
        "--m15-result-dir",
        type=Path,
        help="M15 shrinkage screen consumed only by the frozen M15-to-M20 selection gate",
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aligned-only",
        action="store_true",
        help="read-only gate; exits nonzero when the shuffled control is not permitted",
    )
    parser.add_argument(
        "--m15-selection-only",
        action="store_true",
        help="read-only frozen M15 split; exits nonzero when M20 is not permitted",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="required for the two-arm aggregate; refuses to overwrite an existing file",
    )
    args = parser.parse_args()
    reference_dir = args.reference_dir.expanduser().resolve()
    if args.aligned_only and args.m15_selection_only:
        parser.error("--aligned-only and --m15-selection-only are mutually exclusive")
    if args.m15_selection_only:
        if args.out is not None or args.result_dir is not None:
            parser.error("--m15-selection-only is read-only and uses only --m15-result-dir")
        if args.m15_result_dir is None:
            parser.error("--m15-result-dir is required for --m15-selection-only")
        result = check_m15_selection_gate(
            args.m15_result_dir.expanduser().resolve(),
            reference_dir,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["m20_permitted"]:
            raise SystemExit(3)
        return
    if args.result_dir is None:
        parser.error("--result-dir is required unless --m15-selection-only is used")
    result_dir = args.result_dir.expanduser().resolve()
    if args.aligned_only:
        if args.out is not None:
            parser.error("--aligned-only is read-only and cannot be combined with --out")
        result = check_aligned_first_gate(result_dir, reference_dir, seed=args.seed)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["control_permitted"]:
            raise SystemExit(3)
        return
    if args.out is None:
        parser.error("--out is required for the two-arm M20 aggregate")
    if args.m15_result_dir is None:
        parser.error("--m15-result-dir is required for the two-arm M20 aggregate")
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite aggregate {out}")
    result = aggregate_m20(
        result_dir,
        reference_dir,
        m15_result_dir=args.m15_result_dir.expanduser().resolve(),
        seed=args.seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm_mean_r2": result["arm_mean_r2"],
                "stage0_pass": result["stage0_descriptive_pass"],
                "formal_effectiveness_pass": result["formal_effectiveness_pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
