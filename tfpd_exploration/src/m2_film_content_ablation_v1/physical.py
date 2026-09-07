"""Run one paired seed of the official-grid M2 FiLM content ablation."""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m2_hold_film_probe_v1 import core as probe_core
from tfpd_exploration.src.m2_hold_film_probe_v1 import plan as probe_plan
from tfpd_exploration.src.m2_hold_film_probe_v1.physical import (
    _atomic_bytes,
    _atomic_json,
    _state_sha,
    install_film,
    score_arm,
    train_film,
)

from . import plan


class AblationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AblationError(message)


def _external(score: dict[str, Any]) -> dict[str, Any]:
    return score["summaries"]["external_official_query"]


def _head_state(student: Any) -> dict[str, torch.Tensor]:
    prefixes = ("contrast_context.", "contrast_film.")
    return {
        key: value.detach().cpu().clone()
        for key, value in student.id_encoder.state_dict().items()
        if key.startswith(prefixes)
    }


def _score_record(score: dict[str, Any], p0: dict[str, Any]) -> dict[str, Any]:
    return {
        "summaries": score["summaries"],
        "vs_p0_external": probe_core.paired_contrast(
            score["session_maps"]["external_official_query"],
            p0["session_maps"]["external_official_query"],
        ),
        "film_head_sha256": score["film_head_sha256"],
    }


def execute_seed(
    repo_root: Path,
    *,
    seed: int,
    gpu_index: int = plan.GPU_INDEX,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Train REAL, EMPTY and train-on-ROWSHUFFLE with paired RNG for one seed."""
    seed = int(seed)
    _require(seed in plan.SEEDS, f"seed must be one of {plan.SEEDS}")
    _require(int(gpu_index) == plan.GPU_INDEX, "this experiment is bound to physical GPU0")
    _require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 is required")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CUDA_VISIBLE_DEVICES must be 0")
    _require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER mismatch")

    root = plan.seed_root(repo_root, seed)
    _require(not root.exists(), f"immutable seed root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        root / "attempt.json",
        {
            "schema": plan.SCHEMA + "_attempt",
            "status": "STARTED",
            "seed": seed,
            "gpu_index": plan.GPU_INDEX,
            "official_config": plan.OFFICIAL_CONFIG_NAME,
            "arms": ["REAL", "EMPTY", "ROWSHUFFLE"],
        },
    )

    _require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    started = time.monotonic()

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    _require(metadata["checkpoint_sha256"] == plan.CHECKPOINT_SHA256, "checkpoint drift")
    _require(metadata["normalization_sha256"] == plan.NORMALIZATION_SHA256, "normalizer drift")
    student = model.student.to(device)
    base_encoder = student.id_encoder
    decoder_before = _state_sha(student.decoder.state_dict())
    base_before = _state_sha(base_encoder.state_dict())
    datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    _require(all(value is not None for value in datasets.values()), "required dataset missing")

    canonical_path = repo_root / probe_plan.RESULT_ROOT_RELATIVE / "film_states.pt"
    _require(canonical_path.is_file(), f"canonical P0 state missing: {canonical_path}")
    canonical_payload = torch.load(canonical_path, map_location="cpu", weights_only=False)
    _require("p0" in canonical_payload, "canonical film_states.pt lacks p0")
    canonical_p0 = {
        key: value.detach().cpu().clone() for key, value in canonical_payload["p0"].items()
    }

    def install_canonical() -> None:
        student.id_encoder = base_encoder
        install_film(student, device, film_input=plan.FILM_INPUT, seed=seed)
        student.id_encoder.load_state_dict(canonical_p0, strict=True)
        student.id_encoder.to(device)
        student.id_encoder.freeze_base_path()

    # One zero-init anchor.  All trained arms below are independently restored
    # from the same base encoder and initialized with the same within-seed seed.
    install_canonical()
    p0 = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        arm="P0_NATIVE_M33",
        horizon=plan.HORIZON,
        t4_mode=plan.T4_MODE,
        contrast_mask=np.asarray(plan.EMPTY_MASK, dtype=np.float32),
    )
    p0_mean = float(_external(p0)["equal_session_mean"])
    _require(abs(p0_mean - plan.P0_M33_EXTERNAL) <= 1.0e-6, "P0 anchor drift")

    specs = (
        ("REAL", False, np.asarray(plan.CONTRAST_MASK, dtype=np.float32)),
        ("EMPTY", False, np.asarray(plan.EMPTY_MASK, dtype=np.float32)),
        ("ROWSHUFFLE", True, np.asarray(plan.CONTRAST_MASK, dtype=np.float32)),
    )
    arms: dict[str, Any] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for name, shuffled, mask in specs:
        print(f"seed={seed} arm={name} install", flush=True)
        install_canonical()
        # train_film historically adds SHUFFLE_SEED to the sampling RNG for a
        # shuffled arm.  Offset its explicit seed so every arm uses the exact
        # same RandomState(seed) window/session schedule.
        train_seed = seed - probe_plan.SHUFFLE_SEED if shuffled else seed
        training = train_film(
            student=student,
            train_dataset=datasets["within_post30"],
            device=device,
            shuffle=shuffled,
            contrast_mask=mask,
            horizon=plan.HORIZON,
            t4_mode=plan.T4_MODE,
            learning_rate=plan.LEARNING_RATE,
            epochs=plan.EPOCHS,
            seed=train_seed,
            windows_per_session=plan.WINDOWS_PER_SESSION,
        )
        matched = score_arm(
            student=student,
            datasets=datasets,
            device=device,
            batch_size=batch_size,
            shuffle=shuffled,
            arm=name,
            horizon=plan.HORIZON,
            t4_mode=plan.T4_MODE,
            contrast_mask=mask,
        )
        record: dict[str, Any] = {
            "matched": _score_record(matched, p0),
            "training": training,
            "initialization_seed": seed,
            "effective_sampling_seed": seed,
            "train_helper_seed": train_seed,
            "profile_law": "fixed_row_shuffle" if shuffled else ("positive_zero" if name == "EMPTY" else "real"),
        }
        if name == "REAL":
            shuffled_score = score_arm(
                student=student,
                datasets=datasets,
                device=device,
                batch_size=batch_size,
                shuffle=True,
                arm="REAL_SCORE_ROWSHUFFLE",
                horizon=plan.HORIZON,
                t4_mode=plan.T4_MODE,
                contrast_mask=mask,
            )
            record["score_time_rowshuffle"] = _score_record(shuffled_score, p0)
        elif name == "ROWSHUFFLE":
            real_score = score_arm(
                student=student,
                datasets=datasets,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                arm="ROWSHUFFLE_TRAIN_REAL_SCORE",
                horizon=plan.HORIZON,
                t4_mode=plan.T4_MODE,
                contrast_mask=mask,
            )
            record["real_profile_diagnostic"] = _score_record(real_score, p0)
        states[name] = _head_state(student)
        arms[name] = record
        print(
            f"seed={seed} arm={name} external={float(_external(matched)['equal_session_mean']):.6f}",
            flush=True,
        )

    if seed == 42:
        canonical_real = float(
            arms["REAL"]["matched"]["summaries"]["external_official_query"]["equal_session_mean"]
        )
        _require(
            abs(canonical_real - plan.LOCAL_CANONICAL_REAL_SEED42)
            <= plan.LOCAL_REPRO_TOLERANCE,
            f"canonical REAL seed42 drift: {canonical_real} vs {plan.LOCAL_CANONICAL_REAL_SEED42}",
        )

    _require(_state_sha(student.decoder.state_dict()) == decoder_before, "decoder changed")
    _require(_state_sha(base_encoder.state_dict()) == base_before, "base encoder changed")
    result = {
        "schema": plan.SCHEMA,
        "status": "TERMINAL",
        "seed": seed,
        "official_config": {
            "name": plan.OFFICIAL_CONFIG_NAME,
            "film_input": plan.FILM_INPUT,
            "contrast_mask": list(plan.CONTRAST_MASK),
            "epochs": plan.EPOCHS,
            "learning_rate": plan.LEARNING_RATE,
            "windows_per_session": plan.WINDOWS_PER_SESSION,
            "horizon": plan.HORIZON,
        },
        "references": {
            "evalai_submission": plan.OFFICIAL_EVALAI_SUBMISSION,
            "evalai_heldout_r2": plan.OFFICIAL_EVALAI_HELDOUT_R2,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "normalization_sha256": metadata["normalization_sha256"],
            "canonical_p0_path": str(canonical_path.relative_to(repo_root)),
            "canonical_p0_full_state_sha256": _state_sha(canonical_p0),
        },
        "p0": {"summaries": p0["summaries"]},
        "arms": arms,
        "frozen_state": {
            "decoder_before_after_sha256": decoder_before,
            "base_encoder_before_after_sha256": base_before,
        },
        "gpu": {"physical_index": plan.GPU_INDEX, "name": torch.cuda.get_device_name(0)},
        "elapsed_seconds": float(time.monotonic() - started),
        "evalai_push": False,
    }
    _atomic_json(root / "score.json", result)
    buffer = io.BytesIO()
    torch.save(states, buffer)
    _atomic_bytes(root / "film_heads.pt", buffer.getvalue())
    _atomic_json(
        root / "terminal.json",
        {"schema": plan.SCHEMA, "status": "TERMINAL", "seed": seed, "evalai_push": False},
    )
    return result


def publish_failure(repo_root: Path, *, seed: int, exc: BaseException) -> None:
    root = plan.seed_root(repo_root, seed)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
    path = root / "failure.json"
    if path.exists():
        return
    _atomic_json(
        path,
        {
            "schema": plan.SCHEMA + "_failure",
            "status": "FAILED",
            "seed": int(seed),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
    )
