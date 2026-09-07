"""One paired seed of REAL/EMPTY/ROWSHUFFLE on MOVE-T4."""

from __future__ import annotations

import io
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
from tfpd_exploration.src.m2_movement_t4_ablation_v1.physical import (
    START_BIN,
    STOP_BIN,
    _fit_source_normalizer,
    _override,
)

from . import plan


class MoveFilmError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MoveFilmError(message)


def _head_state(student: Any) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in student.id_encoder.state_dict().items()
        if key.startswith(("contrast_context.", "contrast_film."))
    }


def _record(score: dict[str, Any], p0: dict[str, Any]) -> dict[str, Any]:
    return {
        "summaries": score["summaries"],
        "vs_move_p0_external": probe_core.paired_contrast(
            score["session_maps"]["external_official_query"],
            p0["session_maps"]["external_official_query"],
        ),
        "film_head_sha256": score["film_head_sha256"],
    }


def execute_seed(repo_root: Path, *, seed: int, batch_size: int = 1024) -> dict[str, Any]:
    seed = int(seed)
    _require(seed in plan.SEEDS, f"seed must be one of {plan.SEEDS}")
    _require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CUDA_VISIBLE_DEVICES must be 0")
    _require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER mismatch")
    root = plan.seed_root(repo_root, seed)
    _require(not root.exists(), f"immutable seed root exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    _atomic_json(root / "attempt.json", {"schema": plan.SCHEMA + "_attempt", "status": "STARTED", "seed": seed})
    _require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    started = time.monotonic()

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    student = model.student.to(device)
    base_encoder = student.id_encoder
    decoder_before = _state_sha(student.decoder.state_dict())
    base_before = _state_sha(base_encoder.state_dict())
    base_datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    _require(all(value is not None for value in base_datasets.values()), "dataset missing")
    mean, std, normalizer = _fit_source_normalizer(base_datasets["within_post30"])
    train, train_hashes = _override(base_datasets["within_post30"], mean, std)
    external, external_hashes = _override(base_datasets["external_official_query"], mean, std)
    datasets = {"within_post30": train, "external_official_query": external}

    canonical_path = repo_root / probe_plan.RESULT_ROOT_RELATIVE / "film_states.pt"
    _require(canonical_path.is_file(), "canonical P0 state missing")
    payload = torch.load(canonical_path, map_location="cpu", weights_only=False)
    _require("p0" in payload, "canonical p0 key missing")
    canonical = {key: value.detach().cpu().clone() for key, value in payload["p0"].items()}

    def install_canonical() -> None:
        student.id_encoder = base_encoder
        install_film(student, device, film_input=plan.FILM_INPUT, seed=seed)
        student.id_encoder.load_state_dict(canonical, strict=True)
        student.id_encoder.to(device)
        student.id_encoder.freeze_base_path()

    zero = np.asarray(plan.EMPTY_MASK, dtype=np.float32)
    install_canonical()
    p0 = score_arm(
        student=student, datasets=datasets, device=device, batch_size=batch_size,
        shuffle=False, arm="MOVE-P0", horizon=plan.HORIZON, t4_mode="native", contrast_mask=zero,
    )
    p0_mean = float(p0["summaries"]["external_official_query"]["equal_session_mean"])
    _require(abs(p0_mean - plan.MOVE_P0_EXTERNAL) <= plan.MOVE_P0_TOLERANCE, "MOVE-P0 drift")

    specs = (
        ("REAL", False, np.asarray(plan.CONTRAST_MASK, dtype=np.float32)),
        ("EMPTY", False, zero),
        ("ROWSHUFFLE", True, np.asarray(plan.CONTRAST_MASK, dtype=np.float32)),
    )
    arms: dict[str, Any] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for name, shuffled, mask in specs:
        print(f"MOVE seed={seed} arm={name}", flush=True)
        install_canonical()
        helper_seed = seed - probe_plan.SHUFFLE_SEED if shuffled else seed
        training = train_film(
            student=student, train_dataset=datasets["within_post30"], device=device,
            shuffle=shuffled, contrast_mask=mask, horizon=plan.HORIZON, t4_mode="native",
            learning_rate=plan.LEARNING_RATE, epochs=plan.EPOCHS, seed=helper_seed,
            windows_per_session=plan.WINDOWS_PER_SESSION,
        )
        score = score_arm(
            student=student, datasets=datasets, device=device, batch_size=batch_size,
            shuffle=shuffled, arm=name, horizon=plan.HORIZON, t4_mode="native", contrast_mask=mask,
        )
        arms[name] = {
            "matched": _record(score, p0),
            "training": training,
            "initialization": "canonical_probe_p0",
            "effective_sampling_seed": seed,
            "profile_law": "fixed_row_shuffle" if shuffled else ("positive_zero" if name == "EMPTY" else "real"),
        }
        states[name] = _head_state(student)
        print(f"MOVE seed={seed} arm={name} external={score['summaries']['external_official_query']['equal_session_mean']:.6f}", flush=True)

    _require(_state_sha(student.decoder.state_dict()) == decoder_before, "decoder changed")
    _require(_state_sha(base_encoder.state_dict()) == base_before, "base encoder changed")
    result = {
        "schema": plan.SCHEMA,
        "status": "TERMINAL",
        "seed": seed,
        "movement_window_bins": [START_BIN, STOP_BIN],
        "movement_window_ms": [START_BIN * 20, STOP_BIN * 20],
        "source_normalizer": normalizer,
        "side_sha256": {"within_post30": train_hashes, "external_official_query": external_hashes},
        "p0": {"summaries": p0["summaries"]},
        "arms": arms,
        "canonical_p0_sha256": _state_sha(canonical),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "frozen_state": {"decoder_sha256": decoder_before, "base_encoder_sha256": base_before},
        "elapsed_seconds": float(time.monotonic() - started),
        "evalai_push": False,
    }
    _atomic_json(root / "score.json", result)
    buffer = io.BytesIO(); torch.save(states, buffer); _atomic_bytes(root / "film_heads.pt", buffer.getvalue())
    _atomic_json(root / "terminal.json", {"schema": plan.SCHEMA, "status": "TERMINAL", "seed": seed, "evalai_push": False})
    return result


def publish_failure(repo_root: Path, *, seed: int, exc: BaseException) -> None:
    root = plan.seed_root(repo_root, seed); root.mkdir(parents=True, exist_ok=True)
    path = root / "failure.json"
    if not path.exists():
        _atomic_json(path, {"schema": plan.SCHEMA + "_failure", "status": "FAILED", "seed": int(seed), "type": type(exc).__name__, "message": str(exc)})

