"""Train and score the contrast-only FiLM follow-up catalog."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m2_hold_film_probe_v1 import plan as probe_plan
from tfpd_exploration.src.m2_hold_film_probe_v1.physical import (
    _atomic_bytes,
    _atomic_json,
    install_film,
    score_arm,
    train_film,
)

from . import core, plan


def _mask() -> np.ndarray:
    return np.asarray(plan.MASK, dtype=np.float32)


def _ext(arm: dict[str, object]) -> dict[str, object]:
    return arm["summaries"]["external_official_query"]


def _clone_state(student: Any) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in student.id_encoder.state_dict().items()}


def execute(
    repo_root: Path,
    *,
    gpu_index: int = plan.DEFAULT_GPU_INDEX,
    batch_size: int = 1024,
) -> dict[str, object]:
    if isinstance(gpu_index, bool) or gpu_index < 0:
        raise core.ProbeError("gpu_index must be a non-negative integer")
    expected_cvd = str(gpu_index)
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == expected_cvd, "CUDA_VISIBLE_DEVICES mismatch")
    core.require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER mismatch")
    root = plan.result_root(repo_root)
    core.require(not root.exists(), f"result root already exists: {root}")
    core.require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    started = time.monotonic()

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    core.require(metadata["checkpoint_sha256"] == plan.CHECKPOINT_SHA256, "checkpoint drift")
    core.require(metadata["normalization_sha256"] == plan.NORMALIZATION_SHA256, "normalizer drift")
    student = model.student.to(device)
    base_b3s = student.id_encoder
    datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    core.require(datasets["within_post30"] is not None, "within dataset missing")
    core.require(datasets["external_official_query"] is not None, "external dataset missing")

    def progress(message: str) -> None:
        print(message, flush=True)

    install_film(student, device, film_input=plan.FILM_INPUT, seed=probe_plan.SEED)
    progress("scoring P0 native M33 T4-only")
    p0 = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        arm="p0_native_m33",
        horizon=plan.M33_HORIZON,
        t4_mode=plan.T4_MODE,
        contrast_mask=np.zeros(4, dtype=np.float32),
    )
    p0_mean = float(_ext(p0)["equal_session_mean"])
    core.require(
        abs(p0_mean - plan.P0_M33_EXTERNAL) <= 1.0e-6,
        f"P0 M33 drift {p0_mean} vs {plan.P0_M33_EXTERNAL}",
    )

    trained_arms: list[dict[str, object]] = []
    states: dict[str, dict[str, torch.Tensor]] = {}
    contrast_mask = _mask()
    for row in plan.CONFIGS:
        name = str(row["name"])
        progress(f"training {name}")
        student.id_encoder = base_b3s
        install_film(
            student,
            device,
            film_input=str(row["film_input"]),
            seed=int(row["seed"]),
        )
        training = train_film(
            student=student,
            train_dataset=datasets["within_post30"],
            device=device,
            shuffle=False,
            contrast_mask=contrast_mask,
            horizon=int(row["horizon"]),
            t4_mode=str(row["t4_mode"]),
            learning_rate=float(row["learning_rate"]),
            epochs=int(row["epochs"]),
            seed=int(row["seed"]),
            windows_per_session=int(row["windows_per_session"]),
        )
        labeled = score_arm(
            student=student,
            datasets=datasets,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            arm=name,
            horizon=int(row["horizon"]),
            t4_mode=str(row["t4_mode"]),
            contrast_mask=contrast_mask,
        )
        shuffled = score_arm(
            student=student,
            datasets=datasets,
            device=device,
            batch_size=batch_size,
            shuffle=True,
            arm=f"{name}_shuffle",
            horizon=int(row["horizon"]),
            t4_mode=str(row["t4_mode"]),
            contrast_mask=contrast_mask,
        )
        states[name] = _clone_state(student)
        arm = {
            "name": name,
            "config": row,
            "summaries": labeled["summaries"],
            "shuffle_summaries": shuffled["summaries"],
            "vs_p0": core.paired_contrast(
                labeled["session_maps"]["external_official_query"],
                p0["session_maps"]["external_official_query"],
            ),
            "training": training,
            "film_head_sha256": labeled["film_head_sha256"],
        }
        trained_arms.append(arm)
        progress(
            f"{name} external={float(_ext(labeled)['equal_session_mean']):.6f} "
            f"median={float(_ext(labeled)['equal_session_median']):.6f} "
            f"shuffle={float(shuffled['summaries']['external_official_query']['equal_session_mean']):.6f}"
        )

    submit = core.choose_best(p0_external=_ext(p0), arms=trained_arms)
    table = {
        str(arm["name"]): {
            "external_mean": float(arm["summaries"]["external_official_query"]["equal_session_mean"]),
            "external_median": float(arm["summaries"]["external_official_query"]["equal_session_median"]),
            "within_mean": float(arm["summaries"]["within_post30"]["equal_session_mean"]),
            "vs_p0": arm["vs_p0"],
            "shuffle_external_mean": float(
                arm["shuffle_summaries"]["external_official_query"]["equal_session_mean"]
            ),
            "passes_floor": core.passes_floor(arm, _ext(p0)),
            "config": arm["config"],
        }
        for arm in trained_arms
    }
    result = {
        "schema": plan.SCHEMA,
        "status": "TERMINAL",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "workorder": plan.WORKORDER_RELATIVE,
        "elapsed_seconds": float(time.monotonic() - started),
        "gpu": {
            "cuda_visible_devices": expected_cvd,
            "name": torch.cuda.get_device_name(0),
        },
        "references": {
            "prior_contrast_only_ep12_lr1e4_s42": plan.PRIOR_CONTRAST_ONLY_EXTERNAL,
            "winner_means_ep12_lr3e4_s42": plan.WINNER_T4_PLUS_CONTRAST_EXTERNAL,
        },
        "p0": {"summaries": p0["summaries"]},
        "arms": table,
        "submit": submit,
        "evalai_push": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    _atomic_json(root / "score.json", result)
    _atomic_json(
        root / "terminal.json",
        {
            "schema": plan.SCHEMA,
            "status": "TERMINAL",
            "submit": submit,
        },
    )
    buffer = __import__("io").BytesIO()
    torch.save(states, buffer)
    _atomic_bytes(root / "film_states.pt", buffer.getvalue())
    return result
