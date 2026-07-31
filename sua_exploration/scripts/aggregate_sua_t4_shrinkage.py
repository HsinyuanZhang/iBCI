#!/usr/bin/env python3
"""Fail-closed aggregate for the M_T4=15 uncertainty-shrunk T4 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


PILOT_ARMS = ("t4_m15", "t4w3_m15", "ts4w3_m15")
EXPECTED_GROUP = {
    "t4_m15": "t4",
    "t4w3_m15": "t4w3",
    "ts4w3_m15": "ts4w3",
    "t4_m50": "t4",
}
EXPECTED_POOL = {
    "t4_m15": 15,
    "t4w3_m15": 15,
    "ts4w3_m15": 15,
    "t4_m50": 50,
}
EPOCHS = list(range(5, 13))
EXPECTED_VAL_SESSIONS = [
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
]
EXPECTED_SHRINKAGE_RECEIPT = {
    "family": "uncertainty_wiener_ac_modulation_only",
    "strength": 3.0,
    "intercept_b_shrunk": False,
    "modulation_m_recomputed_from_shrunk_ac": True,
    "selection_scope": (
        "fixed_from_train_only_nested_leave_one_session_out_audit"
    ),
}


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_hex(label: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label}: expected a SHA-256 hex receipt")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label}: expected a SHA-256 hex receipt") from exc
    return value


def _require(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    if not set(seeds).issubset({42, 43, 44}):
        raise argparse.ArgumentTypeError("seeds must be a subset of 42,43,44")
    return seeds


def validate_arm(
    path: Path,
    *,
    arm: str,
    seed: int,
) -> tuple[list[str], np.ndarray, dict[str, str]]:
    payload = _load(path)
    protocol = payload.get("protocol") or {}
    expected_pool = EXPECTED_POOL[arm]
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
        f"{path}: labelled T4 support",
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
    _require(f"{path}: activity calibration", protocol.get("calibration_n"), 30)
    _require(
        f"{path}: evaluation-forward calibration",
        protocol.get("evaluation_forward_calibration_n"),
        30,
    )
    _require(
        f"{path}: train activity calibration",
        protocol.get("train_activity_calibration_n"),
        30,
    )
    _require(f"{path}: common evaluation start", protocol.get("pool_size"), 50)
    _require(
        f"{path}: labelled feature budget",
        protocol.get("label_feature_calibration_n"),
        expected_pool,
    )
    _require(f"{path}: epoch window", protocol.get("epoch_window"), EPOCHS)
    _require(
        f"{path}: chronological label scope",
        payload.get("calibration_feature_label_scope"),
        f"chronological_rewarded_trials[0:{expected_pool}]",
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
    _require(f"{path}: side pool", side.get("pool_size"), expected_pool)
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
    if arm in {"t4w3_m15", "ts4w3_m15"}:
        _require(
            f"{path}: frozen shrinkage receipt",
            side.get("shrinkage"),
            EXPECTED_SHRINKAGE_RECEIPT,
        )
    elif "shrinkage" in side:
        raise ValueError(f"{path}: ordinary T4 must not record shrinkage")
    training = metadata.get("training") or {}
    _require(f"{path}: train activity budget", training.get("calibration_n_trials"), 30)
    _require(f"{path}: epochs", training.get("max_epochs"), 12)
    _require(f"{path}: no early stop", training.get("no_early_stopping"), True)
    _require(f"{path}: checkpoint each epoch", training.get("checkpoint_every_epoch"), True)
    _require(f"{path}: batch size", training.get("batch_size"), 32)
    _require(f"{path}: loss", training.get("loss_mode"), "task_only")
    _require(f"{path}: identity", training.get("identity_mode"), "calibrated")
    _require(f"{path}: decoder trainable", training.get("freeze_decoder"), False)
    if arm != "t4_m50":
        _require(
            f"{path}: encoder base trainable",
            training.get("freeze_encoder_base"),
            False,
        )
    _require(f"{path}: completed", metadata.get("status"), "completed")
    _require(f"{path}: formal unopened", metadata.get("held_out_test_evaluated"), False)
    _require(f"{path}: fresh fit", metadata.get("encoder_warmstart_path"), None)
    if arm != "t4_m50":
        _require(
            f"{path}: coupled decoder",
            (metadata.get("decoder_architecture") or {}).get("mode"),
            "coupled",
        )
    _require(
        f"{path}: no fixed slots",
        (metadata.get("fixed_slot") or {}).get("enabled"),
        False,
    )
    expected_permutation = seed if arm == "ts4w3_m15" else None
    _require(f"{path}: permutation receipt", side.get("permutation_seed"), expected_permutation)
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
    teacher_sha = _sha256_hex(
        f"{path}: teacher SHA", metadata.get("teacher_sha256")
    )
    _require(
        f"{path}: manifest bytes",
        manifest_sha,
        _sha256_file(manifest_path),
    )
    _require(
        f"{path}: teacher bytes",
        teacher_sha,
        _sha256_file(teacher_path),
    )

    sessions, values = _per_session_epoch_mean(payload, path)
    _require(f"{path}: validation score sessions", sessions, EXPECTED_VAL_SESSIONS)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite validation score")
    per_epoch_mean = payload.get("per_epoch_mean_r2") or {}
    if set(per_epoch_mean) != {str(epoch) for epoch in EPOCHS}:
        raise ValueError(f"{path}: per_epoch_mean_r2 must contain exactly epochs 5..12")
    epoch_means = np.asarray(
        [float(per_epoch_mean[str(epoch)]) for epoch in EPOCHS],
        dtype=np.float64,
    )
    if not np.isfinite(epoch_means).all():
        raise ValueError(f"{path}: non-finite epoch mean")
    variant_score = payload.get("variant_score")
    if not isinstance(variant_score, (int, float)) or not np.isclose(
        float(variant_score),
        float(epoch_means.mean()),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"{path}: variant_score does not match fixed epoch mean")
    return sessions, values, {
        "metadata_path": str(metadata_path.resolve()),
        "manifest_sha256": manifest_sha,
        "teacher_sha256": teacher_sha,
        "normalization_sha256": normalization_sha,
    }


def noninferiority_summary(
    treatment: np.ndarray,
    reference: np.ndarray,
    *,
    seeds: tuple[int, ...],
    sessions: list[str],
    margin: float = 0.03,
) -> dict:
    delta = treatment - reference
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    ci = hierarchical_ci(delta)
    stage0 = float(delta.mean()) >= -margin
    formal = bool(
        len(seeds) >= 3
        and stage0
        and ci[0] >= -margin
    )
    return {
        "margin_r2": margin,
        "mean_delta_r2": float(delta.mean()),
        "per_seed_mean_delta_r2": {
            str(seed): float(value) for seed, value in zip(seeds, seed_means)
        },
        "per_session_mean_delta_r2": {
            session: float(value)
            for session, value in zip(sessions, session_means)
        },
        "hierarchical_bootstrap_95ci": ci,
        "stage0_within_margin": stage0,
        "formal_within_margin": formal,
    }


def aggregate(
    result_dir: Path,
    reference_dir: Path,
    seeds: tuple[int, ...],
) -> dict:
    matrices: dict[str, np.ndarray] = {}
    artifacts: dict[str, dict[str, str]] = {
        arm: {} for arm in (*PILOT_ARMS, "t4_m50")
    }
    provenance: dict[str, dict[str, dict[str, str]]] = {
        arm: {} for arm in (*PILOT_ARMS, "t4_m50")
    }
    session_names: list[str] | None = None
    for arm in PILOT_ARMS:
        rows = []
        for seed in seeds:
            path = result_dir / f"{arm}_s{seed}.json"
            sessions, values, receipt = validate_arm(
                path,
                arm=arm,
                seed=seed,
            )
            if session_names is None:
                session_names = sessions
            elif sessions != session_names:
                raise ValueError(f"{path}: validation session matrix differs")
            rows.append(values)
            artifacts[arm][str(seed)] = str(path.resolve())
            provenance[arm][str(seed)] = receipt
        matrices[arm] = np.asarray(rows, dtype=np.float64)

    reference_rows = []
    for seed in seeds:
        path = reference_dir / f"t4m50_s{seed}.json"
        sessions, values, receipt = validate_arm(
            path,
            arm="t4_m50",
            seed=seed,
        )
        if sessions != session_names:
            raise ValueError(f"{path}: M50 reference session matrix differs")
        reference_rows.append(values)
        artifacts["t4_m50"][str(seed)] = str(path.resolve())
        provenance["t4_m50"][str(seed)] = receipt
    matrices["t4_m50"] = np.asarray(reference_rows, dtype=np.float64)
    assert session_names is not None

    for seed in seeds:
        seed_key = str(seed)
        teacher_hashes = {
            provenance[arm][seed_key]["teacher_sha256"]
            for arm in (*PILOT_ARMS, "t4_m50")
        }
        manifest_hashes = {
            provenance[arm][seed_key]["manifest_sha256"]
            for arm in (*PILOT_ARMS, "t4_m50")
        }
        if len(teacher_hashes) != 1:
            raise ValueError(f"seed {seed}: teacher checkpoint drift across arms")
        if len(manifest_hashes) != 1:
            raise ValueError(f"seed {seed}: strict manifest drift across arms")
        _require(
            f"seed {seed}: aligned/shuffled W3 normalization",
            provenance["t4w3_m15"][seed_key]["normalization_sha256"],
            provenance["ts4w3_m15"][seed_key]["normalization_sha256"],
        )

    mechanism = {
        "t4w3_m15_vs_t4_m15": summarize(
            matrices["t4w3_m15"],
            matrices["t4_m15"],
            seeds=seeds,
            sessions=session_names,
        ),
        "t4w3_m15_vs_ts4w3_m15": summarize(
            matrices["t4w3_m15"],
            matrices["ts4w3_m15"],
            seeds=seeds,
            sessions=session_names,
        ),
    }
    noninferiority = noninferiority_summary(
        matrices["t4w3_m15"],
        matrices["t4_m50"],
        seeds=seeds,
        sessions=session_names,
    )
    stage0 = bool(
        all(
            comparison["passes_stage0_descriptive_gates"]
            for comparison in mechanism.values()
        )
        and noninferiority["stage0_within_margin"]
    )
    formal = bool(
        all(
            comparison["passes_formal_effectiveness_gates"]
            for comparison in mechanism.values()
        )
        and noninferiority["formal_within_margin"]
    )
    if any(not math.isfinite(float(matrix.mean())) for matrix in matrices.values()):
        raise ValueError("non-finite arm score")
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "low_label_uncertainty_shrunk_t4_decoding_pilot",
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "protocol": {
            "M_activity": 30,
            "M_T4_pilot": 15,
            "M_T4_reference": 50,
            "common_evaluation_start": 50,
            "shrinkage": "wiener strength=3; frozen by train-only nested LOSO",
            "epochs": 12,
            "scored_epoch_window": EPOCHS,
            "seeds": list(seeds),
            "sessions": session_names,
            "formal_test_evaluated": False,
        },
        "artifacts": artifacts,
        "provenance": provenance,
        "arm_mean_r2": {
            arm: float(matrix.mean()) for arm, matrix in matrices.items()
        },
        "mechanism_contrasts": mechanism,
        "m15_shrink_vs_m50_t4_noninferiority": noninferiority,
        "stage0_descriptive_mechanism_and_label_reduction_pass": stage0,
        "formal_effectiveness_eligible": len(seeds) >= 3,
        "formal_effectiveness_pass": formal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(42,))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.result_dir.expanduser().resolve(),
        args.reference_dir.expanduser().resolve(),
        args.seeds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "arm_mean_r2": result["arm_mean_r2"],
                "stage0_pass": result[
                    "stage0_descriptive_mechanism_and_label_reduction_pass"
                ],
                "formal_effectiveness_pass": result[
                    "formal_effectiveness_pass"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
