"""Load frozen M2 T4, train hold-contrast FiLM only, score local held-out first-30."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import physical as screen_physical

from . import core, plan
from .coeffs import contrast_from_trial_rates, shuffle_contrast
from .encoder import HoldContrastFiLMEarlyPoolEncoder

ARM_P0 = "p0_zero_film_t4"
ARM_P1 = "p1_trained_film"
ARM_P1S = "p1_shuffle_score"
ARM_C2 = "c2_train_on_shuffle"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise core.ProbeError(f"refusing to overwrite {path}")
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise core.ProbeError(f"refusing to overwrite {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _post_linear_count(encoder: nn.Module) -> int:
    return sum(1 for child in encoder.post_pool if isinstance(child, nn.Linear))


def _tensor_sha(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return core.array_sha256(array)


def _state_sha(state: dict[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(_tensor_sha(state[key]).encode())
    return digest.hexdigest()


def install_film(
    student: Any,
    device: torch.device,
    *,
    film_input: str = "t4_plus_contrast",
    seed: int = plan.SEED,
) -> HoldContrastFiLMEarlyPoolEncoder:
    base = student.id_encoder
    core.require(getattr(base, "variant", None) == "B3S", "frozen encoder is not B3S")
    core.require(int(base.side_dim) == plan.T4_DIM, "frozen B3S side_dim is not 4")
    core.require(int(base.hidden_dim) == plan.HIDDEN_DIM, "frozen hidden_dim drift")
    core.require(int(base.trial_length) == plan.TRIAL_LENGTH, "frozen trial_length drift")
    core.require(int(base.window_size) == plan.WINDOW_SIZE, "frozen window_size drift")
    in_features = int(base.post_pool[0].in_features)
    core.require(in_features == plan.HIDDEN_DIM + plan.T4_DIM, "post_pool geometry is not 64+4")
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    film = HoldContrastFiLMEarlyPoolEncoder(
        trial_length=int(base.trial_length),
        window_size=int(base.window_size),
        hidden_dim=int(base.hidden_dim),
        side_dim=plan.SIDE_DIM,
        film_rank=plan.FILM_RANK,
        num_post_layers=_post_linear_count(base),
        film_input=film_input,
    )
    film.load_t4_state_dict({key: value.detach().cpu() for key, value in base.state_dict().items()})
    film = film.to(device)
    student.id_encoder = film
    student.freeze_decoder()
    film.freeze_base_path()
    trainable = film.trainable_parameter_names()
    core.require(
        trainable == (
            "contrast_context.0.weight",
            "contrast_context.0.bias",
            "contrast_film.weight",
            "contrast_film.bias",
        ),
        f"trainable set drift: {trainable}",
    )
    core.require(
        all(not parameter.requires_grad for parameter in student.decoder.parameters()),
        "decoder is not frozen",
    )
    core.require(
        all(
            not parameter.requires_grad
            for name, parameter in film.named_parameters()
            if not name.startswith(("contrast_context.", "contrast_film."))
        ),
        "B3S base path is not frozen",
    )
    n_train = sum(parameter.numel() for parameter in film.parameters() if parameter.requires_grad)
    expected_params = 1192 if film_input == "contrast_only" else 1224
    core.require(n_train == expected_params, f"FiLM parameter count drift: {n_train}")
    return film


def session_side(
    dataset: Any,
    session: str,
    *,
    shuffle: bool,
    contrast_mask: np.ndarray | None = None,
    horizon: int = plan.ACTIVITY_HORIZON,
    t4_mode: str = "ridge",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
    core.require(calibration.ndim == 3, "calibration must be [trials,time,channels]")
    core.require(calibration.shape[0] >= horizon, f"{session} lacks first-{horizon} trials")
    core.require(
        calibration.shape[1:] == (plan.TRIAL_LENGTH, plan.CHANNELS),
        "M2 calibration shape drift",
    )
    activity = np.ascontiguousarray(calibration[:horizon], dtype=np.float32)
    if t4_mode == "native":
        t4 = np.ascontiguousarray(
            dataset._native_t4_side_features(session, 0, horizon), dtype=np.float32
        )
        ridge_evidence = {
            "budget": int(horizon),
            "selection": "native_t4_first_m",
        }
    else:
        core.require(t4_mode == "ridge", "t4_mode must be ridge or native")
        selected = np.arange(horizon, dtype=np.int64)
        t4, ridge_evidence = screen_physical._ridge_side(dataset, session, selected)
    contrast = contrast_from_trial_rates(
        dataset.calib_trial_spike_sums[session],
        dataset.calib_trial_lengths[session],
        dataset.calib_trial_target_angles[session],
        horizon=horizon,
    )
    if shuffle:
        contrast = shuffle_contrast(contrast, seed=plan.SHUFFLE_SEED)
    if contrast_mask is not None:
        mask = np.asarray(contrast_mask, dtype=np.float32).reshape(plan.CONTRAST_DIM)
        core.require(mask.shape == (plan.CONTRAST_DIM,), "contrast mask dim drift")
        contrast = np.ascontiguousarray(contrast * mask[None, :], dtype=np.float32)
    side = np.ascontiguousarray(np.concatenate([t4, contrast], axis=1), dtype=np.float32)
    core.require(side.shape == (plan.CHANNELS, plan.SIDE_DIM), "side concat shape drift")
    core.require(np.isfinite(side).all(), "non-finite T4+contrast side")
    return activity, side, {
        "session": session,
        "horizon": int(horizon),
        "t4_mode": t4_mode,
        "activity_sha256": core.array_sha256(activity),
        "t4_sha256": core.array_sha256(t4),
        "contrast_sha256": core.array_sha256(contrast),
        "side_sha256": core.array_sha256(side),
        "shuffled": bool(shuffle),
        "contrast_mask": None if contrast_mask is None else np.asarray(contrast_mask, dtype=np.float32).tolist(),
        "ridge": ridge_evidence,
        "hold_count": int((~np.isfinite(
            np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)[:horizon]
        )).sum()),
        "reach_count": int(np.isfinite(
            np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)[:horizon]
        ).sum()),
    }


def _score_session(
    *,
    student: Any,
    dataset: Any,
    session: str,
    surface: str,
    device: torch.device,
    batch_size: int,
    shuffle: bool,
    contrast_mask: np.ndarray | None = None,
    horizon: int = plan.ACTIVITY_HORIZON,
    t4_mode: str = "ridge",
) -> dict[str, object]:
    activity, side, evidence = session_side(
        dataset,
        session,
        shuffle=shuffle,
        contrast_mask=contrast_mask,
        horizon=horizon,
        t4_mode=t4_mode,
    )
    starts = screen_physical._session_starts(dataset, session, surface)
    support = torch.from_numpy(activity).unsqueeze(0).to(device)
    side_tensor = torch.from_numpy(side).unsqueeze(0).to(device)
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        identity = student.compute_identity(support, side_features=side_tensor)
        core.require(tuple(identity.shape) == (1, plan.CHANNELS, plan.WINDOW_SIZE), "identity shape drift")
        for offset in range(0, starts.size, batch_size):
            chunk_starts = starts[offset : offset + batch_size]
            neural_np = np.stack(
                [dataset.neural_data[session][start : start + plan.WINDOW_SIZE] for start in chunk_starts],
                axis=0,
            ).astype(np.float32, copy=False)
            target_np = np.stack(
                [dataset.covariate_data[session][start + plan.WINDOW_SIZE - 1] for start in chunk_starts],
                axis=0,
            ).astype(np.float32, copy=False)
            neural = torch.from_numpy(neural_np).to(device)
            prediction, _ = student(neural, identity=identity)
            prediction_np = (
                prediction[:, -1, :].detach().cpu().numpy().astype(np.float32) / plan.BEHAVIOR_SCALE
            )
            predictions.append(prediction_np)
            targets.append(target_np)
    target = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
    prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import variance_weighted_r2

    return {
        "session": session,
        "surface": surface,
        "window_count": int(starts.size),
        "ordered_window_starts_sha256": core.array_sha256(starts),
        "target_sha256": core.array_sha256(target),
        "prediction_sha256": core.array_sha256(prediction),
        "r2": variance_weighted_r2(target, prediction),
        "carrier": evidence,
    }


def score_arm(
    *,
    student: Any,
    datasets: dict[str, Any],
    device: torch.device,
    batch_size: int,
    shuffle: bool,
    arm: str,
    contrast_mask: np.ndarray | None = None,
    horizon: int = plan.ACTIVITY_HORIZON,
    t4_mode: str = "ridge",
) -> dict[str, object]:
    student.eval()
    expected_counts = {
        "within_post30": plan.EXPECTED_WITHIN_SESSIONS,
        "external_official_query": plan.EXPECTED_EXTERNAL_SESSIONS,
    }
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    session_maps: dict[str, dict[str, float]] = {}
    for surface, dataset in datasets.items():
        sessions = sorted(dataset.calib_trialized_neural_features)
        core.require(len(sessions) == expected_counts[surface], f"{surface} session count drift")
        values: dict[str, float] = {}
        for session in sessions:
            row = _score_session(
                student=student,
                dataset=dataset,
                session=session,
                surface=surface,
                device=device,
                batch_size=batch_size,
                shuffle=shuffle,
                contrast_mask=contrast_mask,
                horizon=horizon,
                t4_mode=t4_mode,
            )
            row["arm"] = arm
            rows.append(row)
            values[session] = float(row["r2"])
        session_maps[surface] = values
        summaries[surface] = core.summarize_sessions(values)
    return {
        "arm": arm,
        "shuffle": bool(shuffle),
        "horizon": int(horizon),
        "t4_mode": t4_mode,
        "contrast_mask": None if contrast_mask is None else np.asarray(contrast_mask, dtype=np.float32).tolist(),
        "summaries": summaries,
        "session_maps": session_maps,
        "rows": rows,
        "film_state_sha256": _state_sha(student.id_encoder.state_dict()),
        "film_head_sha256": _state_sha(
            {
                key: value
                for key, value in student.id_encoder.state_dict().items()
                if key.startswith(("contrast_context.", "contrast_film."))
            }
        ),
    }


def train_film(
    *,
    student: Any,
    train_dataset: Any,
    device: torch.device,
    shuffle: bool,
    contrast_mask: np.ndarray | None = None,
    horizon: int = plan.ACTIVITY_HORIZON,
    t4_mode: str = "ridge",
    learning_rate: float | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    windows_per_session: int | None = None,
) -> dict[str, object]:
    film = student.id_encoder
    student.freeze_decoder()
    film.freeze_base_path()
    lr = plan.LEARNING_RATE if learning_rate is None else float(learning_rate)
    n_epochs = plan.EPOCHS if epochs is None else int(epochs)
    train_seed = plan.SEED if seed is None else int(seed)
    n_windows = plan.WINDOWS_PER_SESSION if windows_per_session is None else int(windows_per_session)
    core.require(n_epochs >= 1, "epochs must be >= 1")
    core.require(n_windows >= 1, "windows_per_session must be >= 1")
    core.require(lr > 0.0, "learning_rate must be positive")
    trainable = [parameter for parameter in film.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable,
        lr=lr,
        betas=plan.ADAM_BETAS,
        eps=plan.ADAM_EPS,
    )
    sessions = sorted(train_dataset.calib_trialized_neural_features)
    core.require(len(sessions) == plan.EXPECTED_WITHIN_SESSIONS, "held-in session count drift")
    packs: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    starts_map: dict[str, np.ndarray] = {}
    for session in sessions:
        packs[session] = session_side(
            train_dataset,
            session,
            shuffle=shuffle,
            contrast_mask=contrast_mask,
            horizon=horizon,
            t4_mode=t4_mode,
        )
        starts_map[session] = screen_physical._session_starts(
            train_dataset, session, "within_post30"
        )
    rng = np.random.RandomState(train_seed + (plan.SHUFFLE_SEED if shuffle else 0))
    student.train()
    history: list[dict[str, object]] = []
    updates = 0
    for epoch in range(n_epochs):
        epoch_losses: list[float] = []
        for session_index in rng.permutation(len(sessions)):
            session = sessions[int(session_index)]
            activity, side, _evidence = packs[session]
            support = torch.from_numpy(activity).unsqueeze(0).to(device)
            side_tensor = torch.from_numpy(side).unsqueeze(0).to(device)
            starts = starts_map[session]
            chosen = rng.choice(
                starts,
                size=n_windows,
                replace=starts.size < n_windows,
            )
            for offset in range(0, chosen.size, plan.BATCH_SIZE):
                chunk = np.ascontiguousarray(chosen[offset : offset + plan.BATCH_SIZE])
                neural_np = np.stack(
                    [train_dataset.neural_data[session][start : start + plan.WINDOW_SIZE] for start in chunk],
                    axis=0,
                ).astype(np.float32, copy=False)
                target_np = np.stack(
                    [
                        train_dataset.covariate_data[session][start + plan.WINDOW_SIZE - 1]
                        for start in chunk
                    ],
                    axis=0,
                ).astype(np.float32, copy=False)
                neural = torch.from_numpy(neural_np).to(device)
                target = torch.from_numpy(target_np).to(device)
                identity = student.compute_identity(support, side_features=side_tensor)
                prediction, _ = student(neural, identity=identity)
                loss = ((prediction[:, -1, :] / plan.BEHAVIOR_SCALE) - target).pow(2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
                updates += 1
        history.append(
            {
                "epoch": epoch,
                "mean_last_bin_mse": float(np.mean(epoch_losses)),
                "batch_count": len(epoch_losses),
            }
        )
        print(
            f"film train epoch {epoch} shuffle={shuffle} "
            f"mse={float(np.mean(epoch_losses)):.6f}",
            flush=True,
        )
    student.eval()
    return {
        "shuffle": bool(shuffle),
        "epochs": n_epochs,
        "windows_per_session": n_windows,
        "batch_size": plan.BATCH_SIZE,
        "learning_rate": lr,
        "seed": train_seed,
        "parameter_updates": updates,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
        "history": history,
        "film_state_sha256": _state_sha(film.state_dict()),
        "held_in_sessions": sessions,
        "horizon": int(horizon),
        "t4_mode": t4_mode,
        "contrast_mask": None if contrast_mask is None else np.asarray(contrast_mask, dtype=np.float32).tolist(),
        "session_contrast_sha256": {
            session: packs[session][2]["contrast_sha256"] for session in sessions
        },
    }


def execute(
    repo_root: Path,
    *,
    gpu_index: int = plan.DEFAULT_GPU_INDEX,
    batch_size: int = 1024,
) -> dict[str, object]:
    if isinstance(gpu_index, bool) or gpu_index < 0:
        raise core.ProbeError("gpu_index must be a non-negative integer")
    if isinstance(batch_size, bool) or batch_size < 1:
        raise core.ProbeError("batch_size must be a positive integer")
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
    install_film(student, device)
    zero_state = {key: value.detach().cpu().clone() for key, value in student.id_encoder.state_dict().items()}
    core.require(int(torch.count_nonzero(student.id_encoder.contrast_film.weight)) == 0, "P0 film weight not zero")
    core.require(int(torch.count_nonzero(student.id_encoder.contrast_film.bias)) == 0, "P0 film bias not zero")

    datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    core.require(datasets["within_post30"] is not None, "within dataset missing")
    core.require(datasets["external_official_query"] is not None, "external dataset missing")

    def _progress(message: str) -> None:
        print(message, flush=True)

    _progress("scoring P0 zero-init FiLM")
    p0 = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        arm=ARM_P0,
    )
    _progress(
        "P0 external mean "
        f"{p0['summaries']['external_official_query']['equal_session_mean']:.6f}"
    )
    _progress("training P1 FiLM on labeled hold-vs-reach contrast")
    p1_train = train_film(
        student=student,
        train_dataset=datasets["within_post30"],
        device=device,
        shuffle=False,
    )
    p1_state = {key: value.detach().cpu().clone() for key, value in student.id_encoder.state_dict().items()}
    _progress("scoring P1")
    p1 = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        arm=ARM_P1,
    )
    _progress("scoring P1-shuffle")
    p1s = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        arm=ARM_P1S,
    )
    student.id_encoder.load_state_dict(zero_state, strict=True)
    student.id_encoder.to(device)
    student.id_encoder.freeze_base_path()
    _progress("training C2 on shuffled contrast")
    c2_train = train_film(
        student=student,
        train_dataset=datasets["within_post30"],
        device=device,
        shuffle=True,
    )
    c2_state = {key: value.detach().cpu().clone() for key, value in student.id_encoder.state_dict().items()}
    _progress("scoring C2")
    c2 = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        arm=ARM_C2,
    )

    def _ext(arm: dict[str, object]) -> dict[str, object]:
        return arm["summaries"]["external_official_query"]

    p1_vs_p0 = core.paired_contrast(p1["session_maps"]["external_official_query"], p0["session_maps"]["external_official_query"])
    shuffle_vs_p0 = core.paired_contrast(
        p1s["session_maps"]["external_official_query"], p0["session_maps"]["external_official_query"]
    )
    c2_vs_p0 = core.paired_contrast(c2["session_maps"]["external_official_query"], p0["session_maps"]["external_official_query"])
    verdict = core.decide_verdict(
        p0_external=_ext(p0),
        p1_external=_ext(p1),
        p1_shuffle_external=_ext(p1s),
        c2_external=_ext(c2),
        p1_vs_p0=p1_vs_p0,
        shuffle_vs_p0=shuffle_vs_p0,
        c2_vs_p0=c2_vs_p0,
    )
    payload = {
        "schema": plan.SCHEMA,
        "status": "TERMINAL",
        "scientific_role": (
            "frozen_decoder_hold_vs_reach_film_probe__not_official_champion__not_tta"
        ),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "normalization_sha256": metadata["normalization_sha256"],
        "workorder": plan.WORKORDER_RELATIVE,
        "gpu": {
            "cuda_visible_devices": expected_cvd,
            "logical_device": "cuda:0",
            "name": torch.cuda.get_device_name(0),
            "score_batch_size": batch_size,
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "arms": {
            ARM_P0: {"summaries": p0["summaries"], "film_head_sha256": p0["film_head_sha256"]},
            ARM_P1: {"summaries": p1["summaries"], "film_head_sha256": p1["film_head_sha256"]},
            ARM_P1S: {"summaries": p1s["summaries"], "film_head_sha256": p1s["film_head_sha256"]},
            ARM_C2: {"summaries": c2["summaries"], "film_head_sha256": c2["film_head_sha256"]},
        },
        "contrasts_external": {
            "p1_minus_p0": p1_vs_p0,
            "p1_shuffle_minus_p0": shuffle_vs_p0,
            "c2_minus_p0": c2_vs_p0,
        },
        "training": {"p1": p1_train, "c2": c2_train},
        "sealed_ridge_static_m30_external": plan.SEALED_M30_EXTERNAL,
        "verdict": verdict,
        "rows": p0["rows"] + p1["rows"] + p1s["rows"] + c2["rows"],
    }
    root.mkdir(parents=True, exist_ok=False)
    _atomic_json(root / "score.json", payload)
    _atomic_json(
        root / "training.json",
        {"schema": plan.SCHEMA + "_training", "p1": p1_train, "c2": c2_train},
    )
    _atomic_json(root / "terminal.json", {"schema": plan.SCHEMA, "status": "TERMINAL", "verdict": verdict})
    buffer = __import__("io").BytesIO()
    torch.save({"p0": zero_state, "p1": p1_state, "c2": c2_state}, buffer)
    _atomic_bytes(root / "film_states.pt", buffer.getvalue())
    return payload
