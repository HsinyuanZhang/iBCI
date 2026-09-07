"""Executable paired Stage-1/2 coordination; no CUDA action occurs on import."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import plan
from .core import require


class GPUAuthorizationRequired(PermissionError):
    pass


def stage_plan(stage: int, *, film_admitted: bool) -> dict[str, object]:
    require(stage in (1, 2), "only Stage 1/2 plans are defined")
    if stage == 1:
        return {"stage": "frozen_parent_deployment_ood", "epochs": 12, "seeds": list(plan.SEEDS), "estimator_arms": list(plan.STAGE1_ESTIMATOR_ARMS), "film_arms": list(plan.STAGE1_FILM_ARMS) if film_admitted else [], "parent_checkpoint": "epoch_011.ckpt"}
    return {"stage": "matched_sparse_label_training", "epochs": 12, "seeds": list(plan.SEEDS), "estimator_arms": list(plan.STAGE2_ESTIMATOR_ARMS), "film_arms": list(plan.STAGE2_FILM_ARMS) if film_admitted else [], "parent_checkpoint": "epoch_011.ckpt", "epoch_average_zero_based": list(plan.AVERAGE_EPOCHS_ZERO_BASED)}


def require_explicit_gpu_authorization(authorized: bool) -> None:
    if not authorized:
        raise GPUAuthorizationRequired("Stage-1/2 execution requires a separate explicit GPU authorization")


@dataclass(frozen=True)
class CellSpec:
    stage: int
    seed: int
    arm: str
    signal_view: str
    epoch_count: int
    parent_checkpoint: str
    head_learning_rate: float | None
    base_learning_rate: float | None


def stage2_estimator_contract(seed: int) -> dict[str, object]:
    """Frozen two-arm complete-student contract used by the coordinated executor."""
    require(seed in plan.SEEDS, "unsupported paired seed")
    return {"stage": 2, "branch": "estimator", "seed": seed, "arms": ["WHOLE-T4", "POST700-T4"], "epochs": plan.EPOCHS, "base_learning_rate": plan.STAGE2_BASE_LEARNING_RATE, "complete_student_trainable": True, "paired_rng_restore_before_each_arm": True, "average_epochs_zero_based": list(plan.AVERAGE_EPOCHS_ZERO_BASED), "query_start_trial": plan.QUERY_START_TRIAL}


def executable_cells(*, stage: int, film_admitted: bool, estimator_admitted: bool, stage1_film_open: bool, signal_view: str = "sua") -> tuple[CellSpec, ...]:
    """Gate arms before any model/CUDA construction; Stage-1 estimator is non-gating."""
    require(stage in (1, 2), "unsupported stage")
    if stage == 1:
        arms = list(plan.STAGE1_ESTIMATOR_ARMS)
        if film_admitted:
            arms.extend(arm for arm in plan.STAGE1_FILM_ARMS if arm != "WHOLE-NATIVE")
    else:
        arms = list(plan.STAGE2_ESTIMATOR_ARMS) if estimator_admitted else []
        if film_admitted and stage1_film_open:
            arms.extend(plan.STAGE2_FILM_ARMS)
    return tuple(CellSpec(stage=stage, seed=seed, arm=arm, signal_view=signal_view, epoch_count=plan.EPOCHS, parent_checkpoint=plan.PARENT_RELATIVE[seed], head_learning_rate=plan.FILM_HEAD_LEARNING_RATE if arm in {"EMPTY", "PHASE-R", "SE-T4", "ROW-SHUFFLE"} else None, base_learning_rate=plan.STAGE2_BASE_LEARNING_RATE if stage == 2 else None) for seed in plan.SEEDS for arm in arms)


def execute_cells(cells: Sequence[CellSpec], *, repo_root, result_roots: Mapping[tuple[int, str], object], explicitly_authorized: bool) -> list[object]:
    """Route-owned production dispatch; no caller-provided executable callback."""
    require_explicit_gpu_authorization(explicitly_authorized)
    require(cells and all(cell.epoch_count == plan.EPOCHS and cell.seed in plan.SEEDS for cell in cells), "cell contract drift")
    from .production import execute_gpu_cell
    return [execute_gpu_cell(repo_root, cell, result_roots[(cell.seed, cell.arm)]) for cell in cells]


def average_last_four_trainable_states(epoch_states: Sequence[Mapping[str, np.ndarray]], *, trainable_names: Sequence[str], frozen_reference: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    require(len(epoch_states) == plan.EPOCHS, "exactly 12 epoch states required")
    selected = [epoch_states[index] for index in plan.AVERAGE_EPOCHS_ZERO_BASED]
    names = set(trainable_names)
    require(names, "no trainable parameters")
    output: dict[str, np.ndarray] = {}
    for name, initial in frozen_reference.items():
        values = [np.asarray(state[name]) for state in selected]
        if name in names:
            require(all(np.issubdtype(value.dtype, np.floating) for value in values), f"trainable {name} is not floating")
            output[name] = np.asarray(np.mean(np.stack(values, axis=0, dtype=np.float64), axis=0), dtype=initial.dtype)
        else:
            require(all(np.array_equal(value, initial) for value in values), f"frozen state drift: {name}")
            output[name] = np.asarray(initial).copy()
    return output
