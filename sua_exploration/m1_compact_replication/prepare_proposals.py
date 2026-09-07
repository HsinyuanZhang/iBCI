#!/usr/bin/env python3
"""Prepare score-free M1 compact-B3S GPU proposals without launching them.

The generated JSON binds two strictly conditional programs:

* fold0/seed42 fixed-e11 -> fixed-e23 continuation for the existing B0/B3S
  pair, using Lightning's optimizer/loop restoration; and
* fresh fixed-e11 B0/B3S cells for folds 1 and 2, allowed only after the
  continuation gate passes.

This module never imports CUDA, instantiates a data module, constructs a
Trainer, or invokes a subprocess.  It validates checkpoint and source-only
teacher receipts from disk and emits command argument vectors as inert data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
STREAM_ROOT = ROOT / "streaming_calibration_exp"
PYTHON = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
TRAIN = STREAM_ROOT / "src/train.py"

SCHEMA = "m1_compact_b3s_gpu_proposals_v1"
STATUS = "PASS_M1_COMPACT_B3S_PROPOSALS_PREPARED_NOT_LAUNCHED"
SEED = 42
EXPECTED_RESUME_EPOCH = 11
EXPECTED_GLOBAL_STEP = 59_412
EXPECTED_LIGHTNING = "2.6.5"

B0_RUN = ROOT / (
    "outputs/streaming_calibration/"
    "m1_version_b_hs_continuation_f0_s42_f0_s42_20260809_231214"
)
B3S_RUN = ROOT / (
    "outputs/streaming_calibration/"
    "m1_version_b_c0_f0_s42_f0_s42_20260809_222207"
)
FOLD0_AUTHORITY = ROOT / (
    "sua_exploration/m1_compact_replication/results/"
    "M1_COMPACT_B3S_F0_S42_FORWARD_AUTHORITY_v1.json"
)
FOLD0_E23_GATE_RECEIPT = ROOT / (
    "sua_exploration/m1_compact_replication/results/"
    "M1_COMPACT_B3S_F0_S42_E23_GATE_v1.json"
)
DEFAULT_OUTPUT = ROOT / (
    "sua_exploration/m1_compact_replication/proposals/"
    "M1_COMPACT_B3S_GPU_PROPOSALS_v1.json"
)

M1_SESSIONS = {
    0: ("ses-20120924", ("ses-20120926", "ses-20120927", "ses-20120928")),
    1: ("ses-20120926", ("ses-20120924", "ses-20120927", "ses-20120928")),
    2: ("ses-20120927", ("ses-20120924", "ses-20120926", "ses-20120928")),
}

TEACHERS = {
    1: {
        "checkpoint": STREAM_ROOT / (
            "logs/m1_afc4_source_decoder_fold1/runs/"
            "2026-08-06-16-20-17-515482_rid-m1_afc4_source_decoder_fold1_dev20_f1_s42/"
            "checkpoints/best_ckpt/epoch_019.ckpt"
        ),
        "checkpoint_sha256": "b15edc9f66ff9acded6b78fe0b7a2041b359f8f831db22f3ab87f6083e8f50f3",
        "manifest": STREAM_ROOT / (
            "logs/m1_afc4_source_decoder_fold1/runs/"
            "2026-08-06-16-20-17-515482_rid-m1_afc4_source_decoder_fold1_dev20_f1_s42/"
            "source_only_decoder_manifest.json"
        ),
        "manifest_sha256": "9d67b56bdc354d1a8b0b7bee83b6c5203e566adb982496289a0122723e130551",
    },
    2: {
        "checkpoint": STREAM_ROOT / (
            "logs/m1_afc4_source_decoder_fold2_remote/runs/remote_fold2_source_epoch019/"
            "checkpoints/best_ckpt/epoch_019.ckpt"
        ),
        "checkpoint_sha256": "925fba67a6a4338ee6e399751e80d53c5e74a22a7dd3ae0329d9c8da9e0291e2",
        "manifest": STREAM_ROOT / (
            "logs/m1_afc4_source_decoder_fold2_remote/runs/remote_fold2_source_epoch019/"
            "source_only_decoder_manifest.json"
        ),
        "manifest_sha256": "4dc53581280281326257c158a977b329195d0335443c658899f83a9a4410e4bb",
    },
}


class ProposalError(ValueError):
    """A fail-closed proposal prerequisite was not met."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ProposalError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(value, dict), f"{label} must be an object")
    return value


def _optimizer_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    groups = state.get("param_groups")
    slots = state.get("state")
    _need(isinstance(groups, list) and len(groups) == 1, "resume optimizer must have one param group")
    _need(isinstance(slots, dict) and len(slots) > 0, "resume optimizer state is empty")
    group = groups[0]
    parameters = group.get("params")
    _need(isinstance(parameters, list) and parameters, "resume optimizer param list is empty")
    _need(set(slots).issubset(set(parameters)), "optimizer state keys are outside its param group")
    _need(float(group.get("lr")) == 1.0e-4, "resume optimizer LR drift")
    _need(float(group.get("weight_decay")) == 0.0, "resume optimizer weight decay drift")
    steps: set[int] = set()
    for slot in slots.values():
        _need(isinstance(slot, Mapping), "optimizer slot is malformed")
        step = slot.get("step")
        if torch.is_tensor(step):
            step = step.detach().cpu().item()
        steps.add(int(step))
    _need(steps == {EXPECTED_GLOBAL_STEP}, f"optimizer step drift: {sorted(steps)}")
    return {
        "param_group_count": 1,
        "parameter_ids_in_group": len(parameters),
        "parameters_with_optimizer_state": len(slots),
        "lr": float(group["lr"]),
        "weight_decay": float(group["weight_decay"]),
        "optimizer_step": EXPECTED_GLOBAL_STEP,
    }


def validate_resume_checkpoint(path: Path, *, expected_variant: str) -> dict[str, Any]:
    """Validate the Lightning state required for an exact continuation."""

    _need(path.is_file() and not path.is_symlink(), f"resume checkpoint missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _need(payload.get("epoch") == EXPECTED_RESUME_EPOCH, "resume checkpoint is not epoch 11")
    _need(payload.get("global_step") == EXPECTED_GLOBAL_STEP, "resume global step drift")
    _need(payload.get("pytorch-lightning_version") == EXPECTED_LIGHTNING, "Lightning version drift")
    _need(isinstance(payload.get("state_dict"), Mapping) and payload["state_dict"], "state_dict missing")
    hyper = payload.get("hyper_parameters")
    _need(isinstance(hyper, Mapping) and hyper.get("variant") == expected_variant, "resume variant drift")
    optimizers = payload.get("optimizer_states")
    _need(isinstance(optimizers, list) and len(optimizers) == 1, "resume optimizer count drift")
    _need(payload.get("lr_schedulers") == [], "resume scheduler state must remain empty")
    loops = payload.get("loops")
    _need(isinstance(loops, Mapping), "resume loop state missing")
    fit = loops.get("fit_loop")
    _need(isinstance(fit, Mapping), "resume fit-loop state missing")
    batch_total = fit.get("epoch_loop.batch_progress", {}).get("total", {})
    _need(batch_total.get("completed") == EXPECTED_GLOBAL_STEP, "fit-loop batch progress drift")
    epoch_total = fit.get("epoch_progress", {}).get("total", {})
    _need(epoch_total.get("processed") == 12, "fit-loop epoch progress drift")
    callbacks = payload.get("callbacks")
    _need(isinstance(callbacks, Mapping) and len(callbacks) == 1, "checkpoint callback state drift")
    callback_key = next(iter(callbacks))
    _need("every_n_epochs': 12" in str(callback_key), "fixed-12 callback state missing")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "epoch": EXPECTED_RESUME_EPOCH,
        "global_step": EXPECTED_GLOBAL_STEP,
        "pytorch_lightning_version": EXPECTED_LIGHTNING,
        "optimizer": _optimizer_summary(optimizers[0]),
        "lr_schedulers": 0,
        "loop_state_present": True,
        "fixed_every_12_epoch_callback_state_present": True,
        "strict_resume_possible": True,
    }


def validate_source_only_teacher(fold: int, spec: Mapping[str, Any]) -> dict[str, Any]:
    target, sources = M1_SESSIONS[fold]
    checkpoint = Path(spec["checkpoint"])
    manifest_path = Path(spec["manifest"])
    _need(sha256_file(checkpoint) == spec["checkpoint_sha256"], f"fold{fold} teacher checkpoint SHA drift")
    _need(sha256_file(manifest_path) == spec["manifest_sha256"], f"fold{fold} teacher manifest SHA drift")
    manifest = read_json(manifest_path, label=f"fold{fold} teacher manifest")
    _need(manifest.get("task") == "m1", f"fold{fold} teacher task drift")
    _need(manifest.get("outer_fold") == fold, f"fold{fold} teacher fold drift")
    _need(manifest.get("outer_left_out") == target, f"fold{fold} teacher target drift")
    _need(manifest.get("source_only") is True, f"fold{fold} teacher is not source-only")
    _need(manifest.get("target_backpropagation") is False, f"fold{fold} teacher target backprop drift")
    for field in ("heldout_opened", "minival_opened", "formal", "evalai"):
        _need(manifest.get(field) is False, f"fold{fold} teacher {field} drift")
    _need(tuple(manifest.get("train_sessions", ())) == sources, f"fold{fold} teacher source list drift")
    _need(manifest.get("validation_sessions") == [], f"fold{fold} teacher validation must be empty")
    _need(target not in manifest.get("source_files", {}), f"fold{fold} teacher manifest includes target")
    _need(set(manifest.get("source_files", {})) == set(sources), f"fold{fold} source files drift")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _need(checkpoint_payload.get("epoch") == 19, f"fold{fold} teacher is not terminal epoch 19")
    return {
        "fold": fold,
        "outer_target": target,
        "source_sessions": list(sources),
        "outer_target_excluded_from_train_sessions": True,
        "outer_target_excluded_from_source_files": True,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": spec["checkpoint_sha256"], "epoch": 19},
        "manifest": {"path": str(manifest_path.resolve()), "sha256": spec["manifest_sha256"]},
    }


def _source_list_override(sources: Sequence[str]) -> str:
    return "data.source_session_names=[" + ",".join(sources) + "]"


def _train_argv(
    *, experiment: str, run_id: str, fold: int, sources: Sequence[str],
    teacher_checkpoint: Path | None = None, resume_checkpoint: Path | None = None,
    epochs: int = 12,
) -> list[str]:
    argv = [
        str(PYTHON), str(TRAIN), f"experiment={experiment}", f"run_id={run_id}",
        f"data.loso_fold={fold}", _source_list_override(sources), f"seed={SEED}",
        f"trainer.min_epochs={epochs}", f"trainer.max_epochs={epochs}",
        "trainer.limit_val_batches=0", "trainer.num_sanity_val_steps=0",
        "test=true",
    ]
    if teacher_checkpoint is not None:
        argv.append(f"model.teacher_ckpt_path={teacher_checkpoint.resolve()}")
    argv.append(f"ckpt_path={resume_checkpoint.resolve() if resume_checkpoint else 'null'}")
    return argv


def build_proposals() -> dict[str, Any]:
    b0_checkpoint = B0_RUN / "checkpoints/best.ckpt"
    b3s_checkpoint = B3S_RUN / "checkpoints/best.ckpt"
    resume = {
        "b0": validate_resume_checkpoint(b0_checkpoint, expected_variant="B0"),
        "b3s_zero4": validate_resume_checkpoint(b3s_checkpoint, expected_variant="B3S"),
    }
    source_teachers = {
        str(fold): validate_source_only_teacher(fold, spec)
        for fold, spec in TEACHERS.items()
    }
    target0, sources0 = M1_SESSIONS[0]
    continuation_commands = {
        "b0": _train_argv(
            experiment="m1_version_b_hs_continuation",
            run_id="m1_compact_b0_f0_s42_resume_e11_to_e23",
            fold=0, sources=sources0, resume_checkpoint=b0_checkpoint, epochs=24,
        ),
        "b3s_zero4": _train_argv(
            experiment="m1_version_b_c0",
            run_id="m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23",
            fold=0, sources=sources0, resume_checkpoint=b3s_checkpoint, epochs=24,
        ),
    }
    cross_cells: list[dict[str, Any]] = []
    for fold in (1, 2):
        target, sources = M1_SESSIONS[fold]
        teacher = Path(TEACHERS[fold]["checkpoint"])
        for arm, experiment in (
            ("b0", "m1_version_b_hs_continuation"),
            ("b3s_zero4", "m1_version_b_c0"),
        ):
            cross_cells.append({
                "fold": fold,
                "seed": SEED,
                "arm": arm,
                "outer_target": target,
                "source_sessions": list(sources),
                "fixed_terminal_epoch_zero_based": 11,
                "argv": _train_argv(
                    experiment=experiment,
                    run_id=f"m1_compact_{arm}_f{fold}_s42_fresh_e11",
                    fold=fold, sources=sources, teacher_checkpoint=teacher, epochs=12,
                ),
            })
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "execution": {
            "gpu_launched": False,
            "trainer_constructed_by_preparation": False,
            "datamodule_constructed_by_preparation": False,
            "cuda_imported_by_preparation": False,
            "commands_are_inert_argv_only": True,
        },
        "fold0_e23_continuation": {
            "name": "paired fixed-e11 to fixed-e23 strict continuation",
            "scope": {
                "fold": 0, "seed": SEED, "outer_target": target0,
                "source_sessions": list(sources0), "support_trials": [0, 10],
                "query_trials": [10, 210], "formal_or_minival_or_heldout": False,
            },
            "resume_validation": resume,
            "commands": continuation_commands,
            "evaluation": "one terminal e23 target query per arm; no intermediate target metric",
            "gate": {
                "metric": "B3S-Zero4 minus B0 pooled variance-weighted R2",
                "threshold": -0.03,
                "pass_if": "delta >= -0.03",
                "paired_same_ordered_query_required": True,
            },
            "required_before_launch": {
                "fold0_forward_authority": str(FOLD0_AUTHORITY.resolve()),
                "authority_status": "PASS_M1_COMPACT_B3S_FOLD0_FORWARD_ONLY_AUTHORITY",
                "strict_lightning_resume_from_both_checkpoints": True,
            },
            "contingency": {
                "if_any_restore_check_fails": "do not call this continuation; prepare a separately named fresh fixed-24 pair",
                "fresh24_is_not_equivalent_to_resume": True,
                "no_independent_from_scratch_claim_for_this_pair": True,
            },
            "launched": False,
        },
        "folds1_2_cross_session": {
            "name": "conditional fresh fixed-e11 precision replication",
            "source_only_teacher_validation": source_teachers,
            "cells": cross_cells,
            "required_gate_receipt_before_launch": {
                "path": str(FOLD0_E23_GATE_RECEIPT.resolve()),
                "required_status": "PASS_M1_COMPACT_B3S_F0_E23_NONINFERIORITY",
                "required_delta_minimum": -0.03,
            },
            "stop_rules": {
                "any_new_fold_delta_below_minus_0_03": "stop remaining expansion",
                "at_least_two_negative_deltas_across_folds_0_1_2": "stop and do not claim stable advantage",
                "no_extra_seed_or_fold_after_stop": True,
            },
            "claim_limit": "development source-LOSO replication only; not formal held-out superiority",
            "launched": False,
        },
        "metadata_correction": {
            "generic_teacher_seen_validation_session_default": None,
            "generic_teacher_exposure_attested": False,
            "teacher_exclusion_evidence_lives_in_fold_specific_source_only_manifests": True,
        },
    }
    result["canonical_content_sha256"] = canonical_sha256(result)
    return result


def write_immutable(path: Path, value: Mapping[str, Any]) -> str:
    output = path.resolve()
    _need(not output.exists() and not output.is_symlink(), f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, output)
        _need(stat.S_IMODE(output.stat().st_mode) == 0o444, "proposal output mode drift")
        return sha256_file(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_proposals()
    digest = write_immutable(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
