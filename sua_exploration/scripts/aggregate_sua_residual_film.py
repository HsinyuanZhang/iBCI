#!/usr/bin/env python3
"""Fail-closed M50 aggregate for the frozen residual-only FiLM fallback.

Only ``residual_film``, ``residual_shuffle`` and ``residual_nofilm`` may be
new runs.  ``t4_continuation`` and ``film`` are immutable same-anchor
references.  In addition to provenance receipts, this verifier opens the
validation checkpoints and proves that every frozen decoder/substrate tensor
equals the selected ordinary-T4 anchor exactly.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sua_confidence_film_t4_budget import summarize


M_T4 = 50
M_ACTIVITY = 30
EPOCHS = tuple(range(5, 13))
ALLOWED_SEEDS = (42, 43, 44)
EXPECTED_VAL_SESSIONS = [
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
]
ARMS = ("t4_continuation", "film", "residual_film", "residual_shuffle", "residual_nofilm")
NEW_ARMS = ("residual_film", "residual_shuffle", "residual_nofilm")
EXPECTED = {
    "t4_continuation": ("B3S", "t4", 1),
    "film": ("B3SCF", "t4cf", 2),
    "residual_film": ("B3SCFR", "t4cf_residual", 2),
    "residual_shuffle": ("B3SCFRS", "t4cf_residual_shuffled", 2),
    "residual_nofilm": ("B3SCFRA", "t4cf_residual", 2),
}
HEAD_SUFFIXES = frozenset({
    "confidence_context.0.weight", "confidence_context.0.bias",
    "confidence_film.weight", "confidence_film.bias",
})
EXPECTED_HEAD_SHAPES = {
    "confidence_context.0.weight": (8, 6),
    "confidence_context.0.bias": (8,),
    "confidence_film.weight": (128, 8),
    "confidence_film.bias": (128,),
}
EXPECTED_HEAD_PARAMETERS = frozenset(f"id_encoder.{name}" for name in HEAD_SUFFIXES)
CONTRASTS = {
    "residual_film_vs_t4_continuation": "t4_continuation",
    "residual_film_vs_full_confidence_film": "film",
    "residual_film_vs_residual_shuffle": "residual_shuffle",
    "residual_film_vs_residual_nofilm": "residual_nofilm",
}
REQUIRED = (
    "residual_film_vs_t4_continuation",
    "residual_film_vs_residual_shuffle",
    "residual_film_vs_residual_nofilm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def require(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(ALLOWED_SEEDS):
        raise argparse.ArgumentTypeError("seeds must be a nonempty unique subset of 42,43,44")
    return seeds


def expected_checkpoint(run_dir: Path, epoch: int) -> Path:
    return run_dir / "epoch_ckpts" / f"epoch_{epoch - 1:03d}.ckpt"


def per_session_epoch_mean(payload: dict[str, Any], path: Path) -> tuple[list[str], np.ndarray]:
    per_epoch = payload.get("per_epoch") or {}
    if not isinstance(per_epoch, dict) or set(per_epoch) != {
        str(epoch) for epoch in EPOCHS
    }:
        raise ValueError(f"{path}: per_epoch must contain exactly epochs 5..12")
    names = EXPECTED_VAL_SESSIONS
    rows: list[list[float]] = []
    epoch_means: dict[int, float] = {}
    for epoch in EPOCHS:
        record = per_epoch.get(str(epoch))
        if not isinstance(record, dict):
            raise ValueError(f"{path}: missing epoch {epoch}")
        scores = record.get("per_session_r2")
        if not isinstance(scores, dict) or set(scores) != set(names):
            raise ValueError(
                f"{path}: epoch {epoch} must contain the exact validation sessions"
            )
        values = [float(scores[name]) for name in names]
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: epoch {epoch} contains non-finite R2")
        mean = float(np.mean(values))
        recorded_mean = record.get("mean_r2")
        if (
            isinstance(recorded_mean, bool)
            or not isinstance(recorded_mean, (int, float))
            or not np.isclose(float(recorded_mean), mean, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(f"{path}: epoch {epoch} mean receipt drifted")
        rows.append(values)
        epoch_means[epoch] = mean
    require(f"{path}: epoch list", payload.get("epoch_list"), list(EPOCHS))
    require(f"{path}: per-epoch mean receipt", payload.get("per_epoch_mean_r2"), {str(k): v for k, v in epoch_means.items()})
    expected_score = float(np.mean(list(epoch_means.values())))
    variant_score = payload.get("variant_score")
    if (
        isinstance(variant_score, bool)
        or not isinstance(variant_score, (int, float))
        or not np.isclose(
            float(variant_score), expected_score, rtol=0.0, atol=1e-12
        )
    ):
        raise ValueError(f"{path}: variant score drifted")
    return names, np.asarray(rows, dtype=np.float64).mean(axis=0)


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError(f"{path}: checkpoint lacks a tensor state_dict")
    return state


def verify_checkpoint_receipts(*, artifact: Path, payload: dict[str, Any], metadata: dict[str, Any]) -> Path:
    """Bind every scored epoch receipt to its actual file before scoring it."""
    run_dir = Path(payload.get("run_dir", "")).expanduser().resolve()
    metadata_run_dir = Path(metadata.get("output_dir", "")).expanduser().resolve()
    if not run_dir.is_dir() or run_dir != metadata_run_dir:
        raise ValueError(f"{artifact}: run directory provenance drift")
    for epoch in EPOCHS:
        record = (payload.get("per_epoch") or {}).get(str(epoch)) or {}
        checkpoint = expected_checkpoint(run_dir, epoch)
        checkpoint_receipt = Path(
            record.get("checkpoint_path", "")
        ).expanduser().resolve()
        require(
            f"{artifact}: epoch {epoch} checkpoint path",
            checkpoint_receipt,
            checkpoint.resolve(),
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        require(f"{artifact}: epoch {epoch} checkpoint SHA-256", record.get("checkpoint_sha256"), sha256(checkpoint))
    return run_dir


def verify_frozen_state(*, artifact: Path, payload: dict[str, Any], metadata: dict[str, Any], anchor: Path) -> None:
    """Prove every new arm checkpoint retained the selected T4 substrate."""
    anchor_state = checkpoint_state(anchor)
    anchor_decoder = {k: v for k, v in anchor_state.items() if k.startswith("student.decoder.")}
    anchor_encoder = {k: v for k, v in anchor_state.items() if k.startswith("student.id_encoder.")}
    if len(anchor_decoder) != 31 or len(anchor_encoder) != 8:
        raise ValueError(f"{artifact}: selected anchor has unexpected decoder/substrate key count")
    run_dir = verify_checkpoint_receipts(artifact=artifact, payload=payload, metadata=metadata)
    for epoch in EPOCHS:
        checkpoint = expected_checkpoint(run_dir, epoch)
        state = checkpoint_state(checkpoint)
        decoder = {k: v for k, v in state.items() if k.startswith("student.decoder.")}
        encoder = {k: v for k, v in state.items() if k.startswith("student.id_encoder.")}
        if set(decoder) != set(anchor_decoder):
            raise ValueError(f"{artifact}: epoch {epoch} decoder state keys drifted")
        if set(encoder) != set(anchor_encoder) | {f"student.{name}" for name in EXPECTED_HEAD_PARAMETERS}:
            raise ValueError(f"{artifact}: epoch {epoch} encoder state keys are not T4 plus four confidence-head tensors")
        for name, value in anchor_decoder.items():
            if not torch.equal(decoder[name], value):
                raise ValueError(f"{artifact}: epoch {epoch} frozen decoder tensor drifted: {name}")
        for name, value in anchor_encoder.items():
            if not torch.equal(encoder[name], value):
                raise ValueError(f"{artifact}: epoch {epoch} frozen T4 substrate tensor drifted: {name}")
        head_numel = 0
        for suffix, expected_shape in EXPECTED_HEAD_SHAPES.items():
            tensor = encoder[f"student.id_encoder.{suffix}"]
            require(
                f"{artifact}: epoch {epoch} head shape {suffix}",
                tuple(tensor.shape),
                expected_shape,
            )
            if not tensor.is_floating_point():
                raise ValueError(
                    f"{artifact}: epoch {epoch} head tensor is not floating point: {suffix}"
                )
            head_numel += tensor.numel()
        require(
            f"{artifact}: epoch {epoch} actual head parameter count",
            head_numel,
            1208,
        )


def validate_arm(path: Path, arm: str, seed: int, *, expected_sessions: list[str] | None, shared: dict[str, str]) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    variant, group, version = EXPECTED[arm]
    payload = load_json(path)
    require(f"{path}: schema", payload.get("schema_version"), 1)
    require(f"{path}: evaluator", payload.get("generated_by"), "eval_epoch_window_generic_dandi688.py")
    require(f"{path}: variant", payload.get("variant"), variant)
    require(f"{path}: seed", payload.get("seed"), seed)
    require(f"{path}: task", payload.get("task"), "CO")
    require(f"{path}: signal", payload.get("signal_view"), "sua")
    require(f"{path}: split", payload.get("split_counts"), [27, 6, 6])
    require(f"{path}: max units", payload.get("max_units_exclusive"), 100)
    require(f"{path}: no formal files", payload.get("no_test_files_evaluated"), True)
    require(f"{path}: no evaluator backward", payload.get("uses_backward_gradients"), False)
    require(f"{path}: no label updates", payload.get("uses_behavior_labels_for_weight_updates"), False)
    require(f"{path}: labels create side feature", payload.get("calibration_features_use_behavior_labels"), True)
    require(
        f"{path}: chronological selection is label-free",
        payload.get("calibration_trial_selection_uses_behavior_labels"),
        False,
    )
    require(f"{path}: label scope", payload.get("calibration_feature_label_scope"), "chronological_rewarded_trials[0:50]")
    protocol = payload.get("protocol") or {}
    for key, value in {
        "total_epochs": 12, "burn_in_epochs": 4, "selection_mode": "first",
        "calibration_n": M_ACTIVITY, "train_activity_calibration_n": M_ACTIVITY,
        "evaluation_forward_calibration_n": M_ACTIVITY, "label_feature_calibration_n": M_T4,
        "pool_size": M_T4, "epoch_window": list(EPOCHS),
    }.items():
        require(f"{path}: protocol {key}", protocol.get(key), value)
    require(f"{path}: fixed checkpoint rule", payload.get("checkpoint_selection_rule"), "pre_declared_fixed_epoch_window_no_argmax")
    metadata_path = Path(payload.get("run_metadata_path", ""))
    if not metadata_path.is_file():
        raise ValueError(f"{path}: run metadata is missing: {metadata_path}")
    require(f"{path}: metadata SHA-256", payload.get("run_metadata_sha256"), sha256(metadata_path))
    metadata = load_json(metadata_path)
    side = metadata.get("side_features") or {}
    training = metadata.get("training") or {}
    require(f"{path}: metadata status", metadata.get("status"), "completed")
    require(f"{path}: metadata schema", metadata.get("schema_version"), 1)
    require(f"{path}: metadata formal seal", metadata.get("held_out_test_evaluated"), False)
    require(f"{path}: metadata variant", metadata.get("variant"), variant)
    require(f"{path}: metadata seed", metadata.get("seed"), seed)
    require(f"{path}: side group", side.get("group"), group)
    require(f"{path}: side version", side.get("feature_version"), version)
    require(f"{path}: side pool", side.get("pool_size"), M_T4)
    require(f"{path}: side dimension", side.get("side_dim"), 4 if arm == "t4_continuation" else 6)
    require(f"{path}: no electrodes", side.get("electrode_embed_dim"), 0)
    require(f"{path}: no electrode vocabulary", side.get("num_electrodes"), 0)
    require(
        f"{path}: no electrode relation",
        side.get("uses_equality_only_relation_membership"),
        False,
    )
    for key, value in {
        "calibration_n_trials": M_ACTIVITY, "max_epochs": 12, "no_early_stopping": True,
        "checkpoint_every_epoch": True, "learning_rate": 1e-4, "batch_size": 32,
        "loss_mode": "task_only", "identity_mode": "calibrated", "deterministic": True,
        "trial_length": 100, "window_size": 50, "decode_last_timestep_only": True,
        "lambda_y": 1.0, "lambda_E": 0.1,
        "limit_train_batches": None, "limit_val_batches": None,
    }.items():
        require(f"{path}: training {key}", training.get(key), value)
    require(f"{path}: task-only decoder mode", (metadata.get("decoder_architecture") or {}).get("mode"), "coupled")
    require(f"{path}: fixed slots disabled", (metadata.get("fixed_slot") or {}).get("enabled"), False)
    require(f"{path}: exact task split", metadata.get("split_counts"), [27, 6, 6])
    require(f"{path}: exact max units", metadata.get("max_units_exclusive"), 100)
    require(f"{path}: metadata signal", metadata.get("signal_view"), "sua")
    require(f"{path}: metadata task", metadata.get("task"), "CO")
    for key in ("teacher_sha256", "train_val_manifest_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{path}: missing {key}")
        actual_path = Path(metadata["teacher_checkpoint"] if key == "teacher_sha256" else metadata["train_val_manifest"])
        if not actual_path.is_file() or sha256(actual_path) != value:
            raise ValueError(f"{path}: {key} provenance drift")
        require(f"{path}: evaluator {key}", payload.get("teacher_ckpt_sha256" if key == "teacher_sha256" else "train_val_manifest_sha256"), value)
        evaluator_path_key = (
            "teacher_ckpt" if key == "teacher_sha256" else "train_val_manifest"
        )
        require(
            f"{path}: evaluator {evaluator_path_key}",
            Path(payload.get(evaluator_path_key, "")).expanduser().resolve(),
            actual_path.expanduser().resolve(),
        )
        previous = shared.setdefault(key, value)
        require(f"{path}: shared {key}", value, previous)
    sessions, values = per_session_epoch_mean(payload, path)
    metadata_sessions = ((metadata.get("session_splits") or {}).get("val"))
    require(f"{path}: metadata validation sessions", metadata_sessions, sessions)
    require(f"{path}: evaluator validation sessions", ((payload.get("session_splits") or {}).get("val")), sessions)
    require(
        f"{path}: fixed validation sessions",
        sessions,
        EXPECTED_VAL_SESSIONS,
    )
    require(
        f"{path}: evaluator session unit counts",
        payload.get("session_unit_counts"),
        metadata.get("session_unit_counts"),
    )
    require(
        f"{path}: no formal files opened",
        (metadata.get("session_files") or {}).get("test"),
        [],
    )
    fit_loader = metadata.get("trainer_fit_validation_loader_contract") or {}
    require(
        f"{path}: fit excludes formal sessions",
        fit_loader.get("formal_test_sessions_loaded_during_fit"),
        False,
    )
    require(
        f"{path}: fit validation sessions",
        fit_loader.get("loader_0_sessions"),
        EXPECTED_VAL_SESSIONS,
    )
    if expected_sessions is not None:
        require(f"{path}: fixed validation sessions", sessions, expected_sessions)
    warmstart = metadata.get("encoder_warmstart_sha256")
    warmstart_text = metadata.get("encoder_warmstart_path")
    warmstart_path = Path(warmstart_text) if isinstance(warmstart_text, str) else Path()
    if not isinstance(warmstart, str) or len(warmstart) != 64 or warmstart_path.name != "epoch_011.ckpt" or not warmstart_path.is_file():
        raise ValueError(f"{path}: missing selected-T4 epoch_011 warm-start")
    require(f"{path}: warm-start SHA-256", warmstart, sha256(warmstart_path))
    previous = shared.setdefault(f"anchor:{seed}", warmstart)
    require(f"{path}: shared selected-T4 anchor", warmstart, previous)
    if arm in NEW_ARMS:
        require(f"{path}: frozen decoder", training.get("freeze_decoder"), True)
        require(f"{path}: frozen substrate", training.get("freeze_encoder_base"), True)
        receipt = metadata.get("confidence_film") or {}
        require(f"{path}: residual mask", receipt.get("confidence_mask"), [True, False])
        require(
            f"{path}: confidence input order",
            receipt.get("confidence_input_order"),
            ["log_residual_variance", "direction_geometry"],
        )
        require(f"{path}: additive control", receipt.get("additive_only"), arm == "residual_nofilm")
        require(f"{path}: six-wide parameter match", receipt.get("parameter_matched_six_wide_context"), True)
        require(f"{path}: receipt frozen decoder", receipt.get("freeze_decoder"), True)
        require(f"{path}: receipt frozen substrate", receipt.get("freeze_encoder_base"), True)
        require(f"{path}: exact optimizer tensors", set(receipt.get("optimizer_trainable_parameter_names") or []), EXPECTED_HEAD_PARAMETERS)
        require(f"{path}: exact optimizer parameter count", receipt.get("optimizer_trainable_parameter_count"), 1208)
        if arm == "residual_shuffle":
            require(f"{path}: residual permutation seed", side.get("permutation_seed"), seed)
        else:
            require(f"{path}: nonshuffle permutation seed", side.get("permutation_seed"), None)
        verify_frozen_state(artifact=path, payload=payload, metadata=metadata, anchor=warmstart_path)
    else:
        # The two immutable references were written before the runner began
        # recording these default-false flags.  ``None`` is therefore accepted
        # only for references; all new residual arms must receipt ``True``.
        if training.get("freeze_decoder") not in (None, False):
            raise ValueError(f"{path}: reference decoder unexpectedly frozen")
        if training.get("freeze_encoder_base") not in (None, False):
            raise ValueError(f"{path}: reference substrate unexpectedly frozen")
        verify_checkpoint_receipts(artifact=path, payload=payload, metadata=metadata)
    normalization = side.get("normalization_sha256")
    if not isinstance(normalization, str) or len(normalization) != 64:
        raise ValueError(f"{path}: missing side-feature normalization SHA")
    if arm != "t4_continuation":
        previous = shared.setdefault(f"t4c-normalization:{seed}", normalization)
        require(f"{path}: shared six-wide normalization", normalization, previous)
    return sessions, values, metadata


def aggregate(result_dir: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    matrices: dict[str, np.ndarray] = {}
    artifacts: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    expected_sessions: list[str] | None = None
    shared: dict[str, str] = {}
    parameter_counts: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        rows: list[np.ndarray] = []
        for seed in seeds:
            path = result_dir / f"{arm}_m50_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            sessions, values, metadata = validate_arm(path, arm, seed, expected_sessions=expected_sessions, shared=shared)
            if expected_sessions is None:
                expected_sessions = sessions
            profile = metadata.get("encoder_cost_profile_reference") or {}
            count = profile.get("parameter_count")
            if not isinstance(count, int):
                raise ValueError(f"{path}: missing encoder parameter receipt")
            parameter_counts.setdefault(str(seed), {})[arm] = count
            rows.append(values)
            artifacts[arm][str(seed)] = str(path.resolve())
        matrices[arm] = np.asarray(rows, dtype=np.float64)
    assert expected_sessions is not None
    for seed, counts in parameter_counts.items():
        expected = counts["film"]
        for arm in NEW_ARMS:
            require(f"seed {seed}: {arm} parameter match to full FiLM", counts[arm], expected)
    contrasts = {name: summarize(matrices["residual_film"], matrices[control], seeds=seeds, sessions=expected_sessions) for name, control in CONTRASTS.items()}
    stage0 = all(contrasts[name]["passes_stage0_descriptive_gates"] for name in REQUIRED)
    formal_eligible = tuple(seeds) == ALLOWED_SEEDS
    formal = formal_eligible and all(contrasts[name]["passes_formal_effectiveness_gates"] for name in REQUIRED)
    return {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "frozen_residual_only_confidence_film_m50_optimized_round",
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "protocol": {"M_activity": M_ACTIVITY, "M_T4": M_T4, "common_evaluation_start": M_T4, "epochs": 12, "scored_epoch_window": list(EPOCHS), "seeds": list(seeds), "sessions": expected_sessions, "formal_test_evaluated": False},
        "selected_t4_anchor_sha256_by_seed": {str(seed): shared[f"anchor:{seed}"] for seed in seeds},
        "shared_provenance": {key: value for key, value in shared.items() if not key.startswith("anchor:")},
        "artifacts": artifacts,
        "parameter_counts_by_seed": parameter_counts,
        "arm_mean_r2": {arm: float(values.mean()) for arm, values in matrices.items()},
        "contrasts": contrasts,
        "required_effectiveness_contrasts": list(REQUIRED),
        "stage0_descriptive_mechanism_pass": stage0,
        "formal_effectiveness_eligible": formal_eligible,
        "formal_effectiveness_pass": formal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=(42,))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate: {out}")
    result = aggregate(args.result_dir.expanduser().resolve(), args.seeds)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arm_mean_r2": result["arm_mean_r2"], "stage0_descriptive_mechanism_pass": result["stage0_descriptive_mechanism_pass"], "formal_effectiveness_pass": result["formal_effectiveness_pass"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
