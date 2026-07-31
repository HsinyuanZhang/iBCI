"""Fail-closed aggregate for fresh T4/B3T+T4/B3T+TS4 SUA runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ARMS = ("t4", "b3t_t4", "b3t_ts4")
EXPECTED = {
    "t4": ("B3S", "t4"),
    "b3t_t4": ("B3TS", "t4"),
    "b3t_ts4": ("B3TS", "ts4"),
}
EXPECTED_COST = {
    "t4": {
        "cost_source": "cycle_model_estimate",
        "parameter_count": 18_290,
        "mac_per_session": 13_033_472,
        "mac_per_trial": 409_600,
        "support_state_bytes": 16_384,
        "trial_buffer_bytes": 25_600,
        "peak_live_state_bytes": 41_984,
        "supports_bin_streaming": False,
        "requires_cubic_interpolation": True,
        "requires_divider": True,
        "requires_general_multiplier": True,
        "weight_bytes": 73_160,
        "variant": "B3S",
        "reference_shape": {
            "num_neurons": 64,
            "trial_length": 100,
            "num_trials": 30,
        },
    },
    "b3t_t4": {
        "cost_source": "cycle_model_estimate",
        "parameter_count": 12_658,
        "mac_per_session": 4_524_032,
        "mac_per_trial": 125_952,
        "support_state_bytes": 16_384,
        "trial_buffer_bytes": 3_072,
        "peak_live_state_bytes": 19_456,
        "supports_bin_streaming": True,
        "requires_cubic_interpolation": True,
        "requires_divider": True,
        "requires_general_multiplier": True,
        "weight_bytes": 50_632,
        "variant": "B3TS",
        "reference_shape": {
            "num_neurons": 64,
            "trial_length": 100,
            "num_trials": 30,
        },
    },
    "b3t_ts4": {
        "cost_source": "cycle_model_estimate",
        "parameter_count": 12_658,
        "mac_per_session": 4_524_032,
        "mac_per_trial": 125_952,
        "support_state_bytes": 16_384,
        "trial_buffer_bytes": 3_072,
        "peak_live_state_bytes": 19_456,
        "supports_bin_streaming": True,
        "requires_cubic_interpolation": True,
        "requires_divider": True,
        "requires_general_multiplier": True,
        "weight_bytes": 50_632,
        "variant": "B3TS",
        "reference_shape": {
            "num_neurons": 64,
            "trial_length": 100,
            "num_trials": 30,
        },
    },
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
PREDECLARED_SEEDS = (42, 43, 44)
NONINFERIORITY_MARGIN = -0.03


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    if any(seed not in PREDECLARED_SEEDS for seed in seeds):
        raise argparse.ArgumentTypeError(
            f"seeds must be a subset of {PREDECLARED_SEEDS}"
        )
    return seeds


def require(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def require_sha256(label: str, value) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label}: expected a SHA-256 hex receipt")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label}: expected a SHA-256 hex receipt") from exc
    return value


def finite_float(label: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a finite numeric value")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label}: expected a finite numeric value")
    return result


def load_cell(
    path: Path, arm: str, seed: int
) -> tuple[list[str], np.ndarray, dict]:
    """Strictly validate one B3T receipt and return its score matrix receipt.

    The returned receipt binds score values to the immutable metadata and data
    substrate inputs.  Both the aligned-first gate and the full aggregate use
    it for cross-arm provenance checks.
    """

    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError(f"{path}: JSON artifact must be an object")
    variant, side = EXPECTED[arm]
    protocol = artifact.get("protocol") or {}
    require(f"{path}: schema", artifact.get("schema_version"), 1)
    require(
        f"{path}: purpose",
        artifact.get("purpose"),
        "epoch_window_deterministic_checkpoint_selection",
    )
    require(
        f"{path}: evaluator",
        artifact.get("generated_by"),
        "eval_epoch_window_generic_dandi688.py",
    )
    require(f"{path}: artifact variant", artifact.get("variant"), variant)
    require(f"{path}: artifact seed", artifact.get("seed"), seed)
    require(f"{path}: task", artifact.get("task"), "CO")
    require(f"{path}: signal", artifact.get("signal_view"), "sua")
    require(f"{path}: split", artifact.get("split_counts"), [27, 6, 6])
    require(f"{path}: max units", artifact.get("max_units_exclusive"), 100)
    require(f"{path}: epoch list", artifact.get("epoch_list"), EPOCHS)
    require(
        f"{path}: checkpoint selection",
        artifact.get("checkpoint_selection_rule"),
        "pre_declared_fixed_epoch_window_no_argmax",
    )
    require(f"{path}: no formal test", artifact.get("no_test_files_evaluated"), True)
    require(f"{path}: eval backward", artifact.get("uses_backward_gradients"), False)
    require(
        f"{path}: eval behavior updates",
        artifact.get("uses_behavior_labels_for_weight_updates"),
        False,
    )
    require(
        f"{path}: label-derived calibration feature",
        artifact.get("calibration_features_use_behavior_labels"),
        True,
    )
    require(
        f"{path}: label-free chronological trial selection",
        artifact.get("calibration_trial_selection_uses_behavior_labels"),
        False,
    )
    require(
        f"{path}: label scope",
        artifact.get("calibration_feature_label_scope"),
        "chronological_rewarded_trials[0:30]",
    )
    require(f"{path}: total epochs", protocol.get("total_epochs"), 12)
    require(f"{path}: burn-in", protocol.get("burn_in_epochs"), 4)
    require(f"{path}: selection mode", protocol.get("selection_mode"), "first")
    require(f"{path}: calibration n", protocol.get("calibration_n"), 30)
    require(
        f"{path}: evaluation-forward calibration n",
        protocol.get("evaluation_forward_calibration_n"),
        30,
    )
    require(
        f"{path}: activity calibration n",
        protocol.get("train_activity_calibration_n"),
        30,
    )
    require(
        f"{path}: labelled calibration n",
        protocol.get("label_feature_calibration_n"),
        30,
    )
    require(f"{path}: common eval start", protocol.get("pool_size"), 30)
    require(f"{path}: epoch window", protocol.get("epoch_window"), EPOCHS)

    metadata_path = Path(artifact.get("run_metadata_path", "")).expanduser().resolve()
    if not metadata_path.is_file():
        raise ValueError(f"{path}: metadata path is missing: {metadata_path}")
    require(
        f"{path}: metadata SHA",
        sha256_file(metadata_path),
        artifact.get("run_metadata_sha256"),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata JSON must be an object")
    require(f"{path}: metadata schema", metadata.get("schema_version"), 1)
    require(f"{path}: metadata variant", metadata.get("variant"), variant)
    require(f"{path}: metadata seed", metadata.get("seed"), seed)
    require(f"{path}: metadata task", metadata.get("task"), "CO")
    require(f"{path}: metadata signal", metadata.get("signal_view"), "sua")
    require(f"{path}: metadata split", metadata.get("split_counts"), [27, 6, 6])
    require(f"{path}: metadata units", metadata.get("max_units_exclusive"), 100)
    run_dir = metadata_path.parent
    require(
        f"{path}: evaluator run directory",
        Path(artifact.get("run_dir", "")).expanduser().resolve(),
        run_dir,
    )
    require(
        f"{path}: metadata output directory",
        Path(metadata.get("output_dir", "")).expanduser().resolve(),
        run_dir,
    )
    data_dir = Path(metadata.get("data_dir", "")).expanduser().resolve()
    if not data_dir.is_dir():
        raise ValueError(f"{path}: data directory is missing: {data_dir}")
    require(
        f"{path}: evaluator data directory",
        Path(artifact.get("data_dir", "")).expanduser().resolve(),
        data_dir,
    )
    side_receipt = metadata.get("side_features") or {}
    require(f"{path}: side group", side_receipt.get("group"), side)
    require(f"{path}: feature version", side_receipt.get("feature_version"), 1)
    require(f"{path}: side pool", side_receipt.get("pool_size"), 30)
    require(f"{path}: side dimension", side_receipt.get("side_dim"), 4)
    require(
        f"{path}: no electrode embedding",
        side_receipt.get("electrode_embed_dim"),
        0,
    )
    require(
        f"{path}: no electrode vocabulary",
        side_receipt.get("num_electrodes"),
        0,
    )
    require(
        f"{path}: no electrode relation",
        side_receipt.get("uses_equality_only_relation_membership"),
        False,
    )
    normalization_sha = require_sha256(
        f"{path}: normalization SHA", side_receipt.get("normalization_sha256")
    )
    expected_permutation = seed if arm == "b3t_ts4" else None
    require(
        f"{path}: permutation receipt",
        side_receipt.get("permutation_seed"),
        expected_permutation,
    )
    training = metadata.get("training") or {}
    require(
        f"{path}: activity support",
        training.get("calibration_n_trials"),
        30,
    )
    require(f"{path}: fixed epochs", training.get("max_epochs"), 12)
    require(
        f"{path}: no early stopping",
        training.get("no_early_stopping"),
        True,
    )
    require(f"{path}: checkpoint every epoch", training.get("checkpoint_every_epoch"), True)
    require(f"{path}: batch size", training.get("batch_size"), 32)
    require(f"{path}: learning rate", training.get("learning_rate"), 1e-4)
    require(f"{path}: task-only loss", training.get("loss_mode"), "task_only")
    require(f"{path}: calibrated identity", training.get("identity_mode"), "calibrated")
    require(f"{path}: decoder trainable", training.get("freeze_decoder"), False)
    require(f"{path}: encoder base trainable", training.get("freeze_encoder_base"), False)
    require(f"{path}: deterministic training", training.get("deterministic"), True)
    require(f"{path}: trial length", training.get("trial_length"), 100)
    require(f"{path}: window size", training.get("window_size"), 50)
    require(
        f"{path}: decode last timestep",
        training.get("decode_last_timestep_only"),
        True,
    )
    require(f"{path}: task loss weight", training.get("lambda_y"), 1.0)
    require(f"{path}: embedding loss weight", training.get("lambda_E"), 0.1)
    epoch_ckpt_dir = Path(
        training.get("epoch_checkpoints_dir", "")
    ).expanduser().resolve()
    require(
        f"{path}: epoch checkpoint directory",
        epoch_ckpt_dir,
        run_dir / "epoch_ckpts",
    )
    require(f"{path}: completed", metadata.get("status"), "completed")
    require(f"{path}: training formal seal", metadata.get("held_out_test_evaluated"), False)
    require(f"{path}: no warm-start", metadata.get("encoder_warmstart_path"), None)
    require(
        f"{path}: coupled decoder",
        (metadata.get("decoder_architecture") or {}).get("mode"),
        "coupled",
    )
    require(
        f"{path}: no fixed slots",
        (metadata.get("fixed_slot") or {}).get("enabled"),
        False,
    )
    session_splits = metadata.get("session_splits") or {}
    require(
        f"{path}: exact validation sessions",
        session_splits.get("val"),
        EXPECTED_VAL_SESSIONS,
    )
    fit_loader = metadata.get("trainer_fit_validation_loader_contract") or {}
    require(
        f"{path}: fit loader excludes formal sessions",
        fit_loader.get("formal_test_sessions_loaded_during_fit"),
        False,
    )
    require(
        f"{path}: fit validation sessions",
        fit_loader.get("loader_0_sessions"),
        EXPECTED_VAL_SESSIONS,
    )
    session_files = metadata.get("session_files") or {}
    require(f"{path}: no formal files opened", session_files.get("test"), [])
    require(
        f"{path}: evaluator session splits",
        artifact.get("session_splits"),
        session_splits,
    )
    require(
        f"{path}: evaluator session unit counts",
        artifact.get("session_unit_counts"),
        metadata.get("session_unit_counts"),
    )

    teacher_path = Path(metadata.get("teacher_checkpoint", ""))
    manifest_path = Path(metadata.get("train_val_manifest", ""))
    if not teacher_path.is_file():
        raise ValueError(f"{path}: teacher checkpoint is missing")
    if not manifest_path.is_file():
        raise ValueError(f"{path}: strict manifest is missing")
    teacher_sha = require_sha256(
        f"{path}: teacher SHA", metadata.get("teacher_sha256")
    )
    manifest_sha = require_sha256(
        f"{path}: manifest SHA", metadata.get("train_val_manifest_sha256")
    )
    require(f"{path}: teacher bytes", teacher_sha, sha256_file(teacher_path))
    require(f"{path}: manifest bytes", manifest_sha, sha256_file(manifest_path))
    require(
        f"{path}: eval teacher path",
        Path(artifact.get("teacher_ckpt", "")).resolve(),
        teacher_path.resolve(),
    )
    require(
        f"{path}: eval manifest path",
        Path(artifact.get("train_val_manifest", "")).resolve(),
        manifest_path.resolve(),
    )
    require(
        f"{path}: eval teacher SHA", artifact.get("teacher_ckpt_sha256"), teacher_sha
    )
    require(
        f"{path}: eval manifest SHA",
        artifact.get("train_val_manifest_sha256"),
        manifest_sha,
    )

    cost = metadata.get("encoder_cost_profile_reference") or {}
    require(
        f"{path}: exact cost profile keys",
        set(cost),
        set(EXPECTED_COST[arm]),
    )
    for key, expected in EXPECTED_COST[arm].items():
        require(f"{path}: cost {key}", cost.get(key), expected)

    per_epoch = artifact.get("per_epoch")
    if not isinstance(per_epoch, dict) or set(per_epoch) != {
        str(epoch) for epoch in EPOCHS
    }:
        raise ValueError(f"{path}: per_epoch must contain exactly epochs 5..12")
    per_epoch_mean = artifact.get("per_epoch_mean_r2")
    if not isinstance(per_epoch_mean, dict) or set(per_epoch_mean) != {
        str(epoch) for epoch in EPOCHS
    }:
        raise ValueError(f"{path}: per_epoch_mean_r2 must contain exactly epochs 5..12")
    rows: list[list[float]] = []
    epoch_means: list[float] = []
    for epoch in EPOCHS:
        record = per_epoch[str(epoch)]
        if not isinstance(record, dict):
            raise ValueError(f"{path}: epoch {epoch} receipt must be an object")
        values = record.get("per_session_r2")
        if not isinstance(values, dict) or set(values) != set(EXPECTED_VAL_SESSIONS):
            raise ValueError(f"{path}: epoch {epoch} must have the exact validation sessions")
        row = [finite_float(f"{path}: epoch {epoch} {session}", values[session]) for session in EXPECTED_VAL_SESSIONS]
        mean_from_sessions = float(np.mean(row))
        recorded_mean = finite_float(
            f"{path}: epoch {epoch} mean", record.get("mean_r2")
        )
        reported_epoch_mean = finite_float(
            f"{path}: epoch {epoch} aggregate mean", per_epoch_mean[str(epoch)]
        )
        checkpoint_path = Path(
            record.get("checkpoint_path", "")
        ).expanduser().resolve()
        expected_checkpoint_path = epoch_ckpt_dir / f"epoch_{epoch - 1:03d}.ckpt"
        require(
            f"{path}: epoch {epoch} checkpoint path",
            checkpoint_path,
            expected_checkpoint_path,
        )
        if not checkpoint_path.is_file():
            raise ValueError(
                f"{path}: epoch {epoch} checkpoint is missing: {checkpoint_path}"
            )
        checkpoint_sha = require_sha256(
            f"{path}: epoch {epoch} checkpoint SHA",
            record.get("checkpoint_sha256"),
        )
        require(
            f"{path}: epoch {epoch} checkpoint bytes",
            checkpoint_sha,
            sha256_file(checkpoint_path),
        )
        if not np.isclose(recorded_mean, mean_from_sessions, rtol=0.0, atol=1e-12):
            raise ValueError(f"{path}: epoch {epoch} mean does not match sessions")
        if not np.isclose(reported_epoch_mean, recorded_mean, rtol=0.0, atol=1e-12):
            raise ValueError(f"{path}: epoch {epoch} aggregate mean does not match receipt")
        rows.append(row)
        epoch_means.append(recorded_mean)
    variant_score = finite_float(f"{path}: variant_score", artifact.get("variant_score"))
    expected_variant_score = float(np.mean(epoch_means))
    if not np.isclose(variant_score, expected_variant_score, rtol=0.0, atol=1e-12):
        raise ValueError(f"{path}: variant_score does not match fixed epoch mean")
    return EXPECTED_VAL_SESSIONS, np.asarray(rows, dtype=np.float64).mean(axis=0), {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": artifact.get("run_metadata_sha256"),
        "teacher_sha256": teacher_sha,
        "manifest_sha256": manifest_sha,
        "normalization_sha256": normalization_sha,
        "cost_profile": cost,
    }


def check_aligned_first_gate(result_dir: Path, *, seed: int = 42) -> dict:
    """Read-only gate for the B3T+TS4 content control.

    The control is meaningful only after a same-seed, fresh B3S/T4 baseline
    and B3TS/T4 arm have both completed under the fixed protocol. Loading both
    artifacts through ``load_cell`` validates their epoch windows, metadata
    receipts, cost profiles, and six validation sessions. This function never
    creates or changes a result, checkpoint, or log path.
    """

    if seed not in PREDECLARED_SEEDS:
        raise ValueError(
            f"B3T aligned-first seed must be in {PREDECLARED_SEEDS}, got {seed}"
        )

    result_dir = result_dir.expanduser().resolve()
    fresh_path = result_dir / f"t4_s{seed}.json"
    aligned_path = result_dir / f"b3t_t4_s{seed}.json"
    sessions, fresh, fresh_receipt = load_cell(fresh_path, "t4", seed)
    aligned_sessions, aligned, aligned_receipt = load_cell(
        aligned_path, "b3t_t4", seed
    )
    require(
        f"seed {seed}: fresh T4/B3T+T4 paired validation-session set",
        aligned_sessions,
        sessions,
    )
    for receipt_key in ("teacher_sha256", "manifest_sha256", "normalization_sha256"):
        require(
            f"seed {seed}: fresh T4/B3T+T4 {receipt_key}",
            aligned_receipt[receipt_key],
            fresh_receipt[receipt_key],
        )
    delta = aligned - fresh
    mean_delta = float(delta.mean())
    # The threshold is inclusive. ``isclose`` protects the intended inclusive
    # boundary from binary floating-point representation noise.
    within_margin = bool(
        mean_delta >= NONINFERIORITY_MARGIN
        or np.isclose(mean_delta, NONINFERIORITY_MARGIN, rtol=0.0, atol=1e-12)
    )
    return {
        "schema_version": 1,
        "purpose": "b3t_t4_aligned_first_noninferiority_gate",
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "protocol": {
            "seed": seed,
            "fresh_training_for_every_arm": True,
            "activity_calibration_n": 30,
            "t4_label_calibration_n": 30,
            "common_evaluation_start": 30,
            "epochs": 12,
            "scored_epoch_window": EPOCHS,
            "formal_test_evaluated": False,
        },
        "artifacts": {
            "fresh_t4": str(fresh_path),
            "b3t_t4": str(aligned_path),
        },
        "cost_profiles": {
            "fresh_t4": fresh_receipt["cost_profile"],
            "b3t_t4": aligned_receipt["cost_profile"],
        },
        "provenance": {
            "fresh_t4": fresh_receipt,
            "b3t_t4": aligned_receipt,
        },
        "arm_mean_r2": {
            "fresh_t4": float(fresh.mean()),
            "b3t_t4": float(aligned.mean()),
        },
        "b3t_t4_minus_fresh_t4": {
            "mean_delta_r2": mean_delta,
            "per_session_delta_r2": {
                session: float(value) for session, value in zip(sessions, delta)
            },
            "margin": NONINFERIORITY_MARGIN,
            "observed_mean_delta_at_least_margin": within_margin,
        },
        "control_permitted": within_margin,
        "formal_effectiveness_eligible": False,
        "formal_effectiveness_pass": False,
    }


def hierarchical_ci(delta: np.ndarray, draws: int = 50_000) -> list[float]:
    rng = np.random.default_rng(20260731)
    n_seed, n_session = delta.shape
    sampled = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        seeds = rng.integers(0, n_seed, n_seed)
        sessions = rng.integers(0, n_session, n_session)
        sampled[index] = delta[np.ix_(seeds, sessions)].mean()
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def summarize(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    seeds: tuple[int, ...],
    sessions: list[str],
) -> dict:
    delta = treatment - control
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    try:
        p_value = float(
            wilcoxon(
                session_means,
                alternative="two-sided",
                zero_method="wilcox",
                method="exact",
            ).pvalue
        )
    except ValueError:
        p_value = 1.0
    ci = hierarchical_ci(delta)
    stage0_gates = {
        "mean_delta_at_least_0p03": float(delta.mean()) >= 0.03,
        "all_observed_seed_means_positive": bool(np.all(seed_means > 0.0)),
        "all_six_session_means_positive": bool(np.all(session_means > 0.0)),
        "session_paired_exact_wilcoxon_two_sided_le_0p05": p_value <= 0.05,
    }
    gates = {
        **stage0_gates,
        "hierarchical_bootstrap_95ci_lower_positive": ci[0] > 0.0,
        "at_least_three_predeclared_seeds": len(seeds) >= 3,
    }
    return {
        "mean_paired_delta_r2": float(delta.mean()),
        "per_seed_mean_delta_r2": {
            str(seed): float(value) for seed, value in zip(seeds, seed_means)
        },
        "per_session_mean_delta_r2": {
            session: float(value) for session, value in zip(sessions, session_means)
        },
        "positive_seed_count": int((seed_means > 0.0).sum()),
        "positive_session_count": int((session_means > 0.0).sum()),
        "hierarchical_bootstrap_95ci": ci,
        "session_paired_exact_wilcoxon_two_sided_p": p_value,
        "stage0_descriptive_gates": stage0_gates,
        "passes_stage0_descriptive_gates": all(stage0_gates.values()),
        "strict_superiority_gates": gates,
        "passes_strict_superiority": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument(
        "--aligned-only",
        action="store_true",
        help=(
            "read-only one-seed gate; exits nonzero when the same-seed TS4 "
            "control is not permitted"
        ),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    result_dir = args.result_dir.expanduser().resolve()

    if args.aligned_only:
        if args.out is not None:
            parser.error("--aligned-only is read-only and cannot be combined with --out")
        if len(seeds) != 1:
            parser.error("--aligned-only requires exactly one predeclared seed")
        gate = check_aligned_first_gate(result_dir, seed=seeds[0])
        print(json.dumps(gate, indent=2, sort_keys=True))
        if not gate["control_permitted"]:
            raise SystemExit(3)
        return
    if args.out is None:
        parser.error("--out is required for the full three-arm aggregate")
    out_path = args.out.expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {out_path}")

    matrices: dict[str, np.ndarray] = {}
    artifacts: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    provenance: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
    costs: dict[str, dict] = {}
    session_names: list[str] | None = None
    shared_receipt: dict | None = None
    for arm in ARMS:
        rows: list[np.ndarray] = []
        for seed in seeds:
            path = result_dir / f"{arm}_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            sessions, values, receipt = load_cell(path, arm, seed)
            if session_names is None:
                session_names = sessions
            elif sessions != session_names:
                raise ValueError(f"{path}: paired validation-session set differs")
            if shared_receipt is None:
                shared_receipt = receipt
            else:
                for receipt_key in (
                    "teacher_sha256",
                    "manifest_sha256",
                    "normalization_sha256",
                ):
                    require(
                        f"{path}: cross-arm {receipt_key}",
                        receipt[receipt_key],
                        shared_receipt[receipt_key],
                    )
            cost = receipt["cost_profile"]
            if arm in costs and cost != costs[arm]:
                raise ValueError(f"{path}: encoder cost profile drifted across seeds")
            costs[arm] = cost
            rows.append(values)
            artifacts[arm][str(seed)] = str(path.resolve())
            provenance[arm][str(seed)] = receipt
        matrices[arm] = np.asarray(rows, dtype=np.float64)
    assert session_names is not None

    accuracy = summarize(
        matrices["b3t_t4"],
        matrices["t4"],
        seeds=seeds,
        sessions=session_names,
    )
    content = summarize(
        matrices["b3t_t4"],
        matrices["b3t_ts4"],
        seeds=seeds,
        sessions=session_names,
    )
    seed_delta = (matrices["b3t_t4"] - matrices["t4"]).mean(axis=1)
    if len(seeds) >= 2:
        seed_se = float(seed_delta.std(ddof=1) / np.sqrt(len(seed_delta)))
        lower_2se = float(seed_delta.mean() - 2.0 * seed_se)
    else:
        seed_se = None
        lower_2se = None
    parameter_reduction = 1.0 - (
        costs["b3t_t4"]["parameter_count"] / costs["t4"]["parameter_count"]
    )
    mac_reduction = 1.0 - (
        costs["b3t_t4"]["mac_per_session"] / costs["t4"]["mac_per_session"]
    )
    state_change = (
        costs["b3t_t4"]["support_state_bytes"]
        - costs["t4"]["support_state_bytes"]
    )
    efficiency_gates = {
        "at_least_three_predeclared_seeds": len(seeds) >= 3,
        "paired_seed_lower_2se_at_least_minus_0p03": (
            lower_2se is not None and lower_2se >= -0.03
        ),
        "parameter_reduction_at_least_25pct": parameter_reduction >= 0.25,
        "session_mac_reduction_at_least_25pct": mac_reduction >= 0.25,
        "no_support_state_increase": state_change <= 0,
        "t4_content_strict_gate": content["passes_strict_superiority"],
    }
    efficiency_pass = all(efficiency_gates.values())
    stage0_efficiency_gates = {
        "observed_mean_delta_at_least_minus_0p03": float(seed_delta.mean()) >= -0.03,
        "parameter_reduction_at_least_25pct": parameter_reduction >= 0.25,
        "session_mac_reduction_at_least_25pct": mac_reduction >= 0.25,
        "no_support_state_increase": state_change <= 0,
        "t4_content_stage0_gate": content["passes_stage0_descriptive_gates"],
    }
    stage0_efficiency_pass = all(stage0_efficiency_gates.values())
    stage0_candidate = (
        accuracy["passes_stage0_descriptive_gates"] or stage0_efficiency_pass
    )
    effective = accuracy["passes_strict_superiority"] or efficiency_pass

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "protocol": {
            "fresh_training_for_every_arm": True,
            "activity_calibration_n": 30,
            "t4_label_calibration_n": 30,
            "common_evaluation_start": 30,
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
        "cost_profiles": costs,
        "contrasts": {
            "b3t_t4_vs_fresh_t4": accuracy,
            "b3t_t4_vs_b3t_ts4": content,
        },
        "efficiency_noninferiority": {
            "mean_seed_paired_delta_r2": float(seed_delta.mean()),
            "seed_level_standard_error": seed_se,
            "paired_lower_2se": lower_2se,
            "margin": -0.03,
            "parameter_reduction_fraction": float(parameter_reduction),
            "session_mac_reduction_fraction": float(mac_reduction),
            "support_state_change_bytes": int(state_change),
            "gates": efficiency_gates,
            "passes_all_gates": efficiency_pass,
        },
        "stage0_efficiency_screen": {
            "gates": stage0_efficiency_gates,
            "passes_all_gates": stage0_efficiency_pass,
        },
        "stage0_candidate_for_multiseed_expansion": stage0_candidate,
        "effective_by_accuracy_superiority": accuracy["passes_strict_superiority"],
        "effective_by_deployment_efficiency": efficiency_pass,
        "overall_effective": effective,
        "formal_test_files_opened": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "arm_mean_r2": payload["arm_mean_r2"],
                "accuracy_delta": accuracy["mean_paired_delta_r2"],
                "content_delta": content["mean_paired_delta_r2"],
                "efficiency_noninferiority": payload["efficiency_noninferiority"],
                "overall_effective": effective,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
