"""Coordinated one-GPU runner for four DANDI CP-FiLM arms."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from . import core, data, plan


def _json_body(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _publish_bytes(root: Path, name: str, body: bytes) -> str:
    path = root / name
    sidecar = root / f"{name}.sha256"
    core.require(not path.exists() and not sidecar.exists(), f"refusing to overwrite {name}")
    digest = hashlib.sha256(body).hexdigest()
    for target, content in ((path, body), (sidecar, f"{digest}  {name}\n".encode())):
        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
            target.chmod(0o444)
        finally:
            temporary.unlink(missing_ok=True)
    return digest


def _publish_json(root: Path, name: str, payload: object) -> str:
    return _publish_bytes(root, name, _json_body(payload))


def _state_sha(module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(key.encode())
        digest.update(core.array_sha256(array).encode())
    return digest.hexdigest()


def _verify_static(repo_root: Path, seed: int) -> dict[str, str]:
    core.require(seed in plan.SEEDS, f"unsupported seed {seed}")
    checks = {
        plan.DESIGN_RELATIVE: plan.DESIGN_SHA256,
        plan.WORKORDER_RELATIVE: plan.WORKORDER_SHA256,
        plan.MANIFEST_RELATIVE: plan.MANIFEST_SHA256,
        plan.TEACHER_RELATIVE: plan.TEACHER_SHA256,
        plan.ANCHOR_RELATIVE[seed]: plan.ANCHOR_SHA256[seed],
    }
    for relative, expected in checks.items():
        path = repo_root / relative
        core.require(path.is_file(), f"missing frozen input {relative}")
        core.require(core.sha256_file(path) == expected, f"frozen input SHA drift {relative}")
    return checks


def _gpu0_attestation() -> dict[str, object]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = [item.strip() for item in output.split(",")]
    core.require(len(fields) == 4 and fields[0] == "0", "GPU0 attestation parse drift")
    core.require(fields[1] == plan.EXPECTED_GPU_UUID, f"GPU0 UUID drift: {fields[1]}")
    return {"physical_index": 0, "uuid": fields[1], "name": fields[2], "memory_total_mib": int(fields[3])}


def _prepare_student(repo_root: Path, seed: int, device):
    import torch
    from src.models.streaming_calibration_module import StreamingCalibrationLitModule

    module = StreamingCalibrationLitModule(
        task="mc_maze",
        variant="B3S",
        teacher_ckpt_path=str(repo_root / plan.TEACHER_RELATIVE),
        window_size=plan.WINDOW,
        trial_length=plan.TRIAL_LENGTH,
        id_hidden_dim=128,
        hidden_dim=plan.HIDDEN_DIM,
        pad_value=-1.0,
        freeze_decoder=True,
        freeze_encoder_base=False,
        loss_mode="task_only",
        lambda_y=1.0,
        lambda_E=0.1,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=plan.BEHAVIOR_SCALE,
        identity_mode="calibrated",
        side_dim=plan.T4_DIM,
        electrode_embed_dim=0,
        num_electrodes=0,
        encoder_warmstart_path=str(repo_root / plan.ANCHOR_RELATIVE[seed]),
        optimizer=partial(torch.optim.Adam, lr=plan.LEARNING_RATE, weight_decay=0.0),
        scheduler=None,
        compile=False,
    )
    module.setup("fit")
    core.require(module.student is not None, "student construction failed")
    student = module.student
    for parameter in student.parameters():
        parameter.requires_grad = False
    student.freeze_decoder()
    student.eval()
    student = student.to(device)
    base = student.id_encoder
    core.require(getattr(base, "variant", None) == "B3S", "anchor encoder is not B3S")
    core.require(base.side_dim == plan.T4_DIM and base.hidden_dim == plan.HIDDEN_DIM, "anchor geometry drift")
    core.require(base.trial_length == plan.TRIAL_LENGTH and base.window_size == plan.WINDOW, "anchor window drift")
    del module
    return student


def _session_cache(materials, student, device, seed: int):
    import torch

    cache = {}
    base = student.id_encoder
    for session, material in sorted(materials.items()):
        calibration = torch.from_numpy(np.asarray(material.record.calib_trials, dtype=np.float32)).unsqueeze(0).to(device)
        carrier = torch.from_numpy(np.asarray(material.record.side_features, dtype=np.float32)).unsqueeze(0).to(device)
        core.require(calibration.shape[1:] == (plan.ACTIVITY_SUPPORT, plan.TRIAL_LENGTH, carrier.shape[1]), f"{session}: support geometry drift")
        core.require(carrier.shape[-1] == plan.T4_DIM, f"{session}: T4 geometry drift")
        with torch.no_grad():
            encoded = base.pre_pool(calibration.permute(0, 1, 3, 2))
            mean_feature = encoded.mean(dim=1)
            native_identity = base.post_pool(torch.cat((mean_feature, carrier), dim=-1))
        shuffled10, permutation = data.shuffled_profile(material.profile10, session, seed)
        profiles = {
            "M10": torch.from_numpy(material.profile10).unsqueeze(0).to(device),
            "M30": torch.from_numpy(material.profile30).unsqueeze(0).to(device),
            "SHUFFLED10": torch.from_numpy(shuffled10).unsqueeze(0).to(device),
            "ZERO": torch.zeros((1, carrier.shape[1], plan.PROFILE_DIM), dtype=carrier.dtype, device=device),
        }
        cache[session] = {
            "mean_feature": mean_feature,
            "carrier": carrier,
            "native_identity": native_identity,
            "profiles": profiles,
            "shuffle_permutation": permutation,
        }
    return cache


def _arm_profile_key(arm: str) -> str:
    return {"CP10": "M10", "CP30": "M30", "SHUFFLE10": "SHUFFLED10", "EMPTY": "ZERO"}[arm]


def _batch_arrays(record, starts: np.ndarray):
    neural = np.stack([record.neural[int(start) : int(start) + plan.WINDOW] for start in starts]).astype(np.float32, copy=False)
    target = np.stack([record.behavior[int(start) + plan.WINDOW - 1] for start in starts]).astype(np.float32, copy=False)
    return np.ascontiguousarray(neural), np.ascontiguousarray(target)


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    residual = np.square(target64 - prediction64).sum(axis=0)
    total = np.square(target64 - target64.mean(axis=0)).sum(axis=0)
    core.require(bool(np.all(total > 0.0)), "R2 target variance degenerate")
    value = 1.0 - float(residual.sum() / total.sum())
    core.require(np.isfinite(value), "R2 nonfinite")
    return value


def _train(*, materials, cache, student, films, seed: int, device) -> dict[str, object]:
    import torch

    optimizers = {
        arm: torch.optim.Adam(films[arm].parameters(), lr=plan.LEARNING_RATE, weight_decay=0.0)
        for arm in plan.ARMS
    }
    first_identity_equal = {}
    first_prediction_equal = {}
    history = []
    gradient_steps = {arm: 0 for arm in plan.ARMS}
    train_sessions = sorted(session for session, material in materials.items() if material.split == "train")
    core.require(len(train_sessions) == plan.TRAIN_SESSIONS, "train roster drift")
    first_batch_done = False
    for epoch in range(plan.EPOCHS):
        epoch_losses = {arm: [] for arm in plan.ARMS}
        order_rng = np.random.Generator(np.random.PCG64(seed * 1000 + epoch))
        for session_index in order_rng.permutation(len(train_sessions)):
            session = train_sessions[int(session_index)]
            material = materials[session]
            starts = data.epoch_starts(material, seed, epoch)
            for offset in range(0, starts.size, plan.BATCH_SIZE):
                chunk = starts[offset : offset + plan.BATCH_SIZE]
                neural_np, target_np = _batch_arrays(material.record, chunk)
                neural = torch.from_numpy(neural_np).to(device)
                target = torch.from_numpy(target_np).to(device)
                state = cache[session]
                if not first_batch_done:
                    with torch.no_grad():
                        native_prediction = student.decode_with_identity(
                            neural, state["native_identity"].expand(neural.shape[0], -1, -1)
                        )[:, -1, :] / plan.BEHAVIOR_SCALE
                        for arm in plan.ARMS:
                            identity = core.film_identity(
                                student.id_encoder,
                                state["mean_feature"],
                                state["carrier"],
                                state["profiles"][_arm_profile_key(arm)],
                                films[arm],
                            )
                            prediction = student.decode_with_identity(neural, identity.expand(neural.shape[0], -1, -1))[:, -1, :] / plan.BEHAVIOR_SCALE
                            first_identity_equal[arm] = bool(torch.equal(identity, state["native_identity"]))
                            first_prediction_equal[arm] = bool(torch.equal(prediction, native_prediction))
                    core.require(all(first_identity_equal.values()) and all(first_prediction_equal.values()), "zero-init parity failed")
                    first_batch_done = True
                for arm in plan.ARMS:
                    film = films[arm]
                    film.train(True)
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    identity = core.film_identity(
                        student.id_encoder,
                        state["mean_feature"],
                        state["carrier"],
                        state["profiles"][_arm_profile_key(arm)],
                        film,
                    )
                    prediction = student.decode_with_identity(neural, identity.expand(neural.shape[0], -1, -1))[:, -1, :] / plan.BEHAVIOR_SCALE
                    loss = torch.mean(torch.square(prediction - target))
                    core.require(bool(torch.isfinite(loss)), f"{arm}: nonfinite loss")
                    loss.backward()
                    optimizer.step()
                    core.require(all(bool(torch.isfinite(parameter).all()) for parameter in film.parameters()), f"{arm}: nonfinite parameter")
                    epoch_losses[arm].append(float(loss.detach().cpu()))
                    gradient_steps[arm] += 1
        row = {
            "epoch": epoch + 1,
            "mean_last_bin_mse": {arm: float(np.mean(epoch_losses[arm])) for arm in plan.ARMS},
            "batch_count": {arm: len(epoch_losses[arm]) for arm in plan.ARMS},
        }
        history.append(row)
        print(f"epoch={epoch + 1} losses={row['mean_last_bin_mse']}", flush=True)
    core.require(all(value > 0 for value in gradient_steps.values()), "missing gradient steps")
    return {
        "history": history,
        "gradient_steps": gradient_steps,
        "first_identity_bitwise_equal": first_identity_equal,
        "first_prediction_bitwise_equal": first_prediction_equal,
        "final_film_state_sha256": {arm: _state_sha(films[arm]) for arm in plan.ARMS},
    }


def _evaluate(*, materials, cache, student, films, device) -> dict[str, object]:
    import torch

    eval_cells = {
        "CP10@M10": ("CP10", "M10"),
        "CP10@M30": ("CP10", "M30"),
        "CP30@M10": ("CP30", "M10"),
        "CP30@M30": ("CP30", "M30"),
        "SHUFFLE10@M10": ("SHUFFLE10", "SHUFFLED10"),
        "SHUFFLE10@REAL10": ("SHUFFLE10", "M10"),
        "EMPTY@ZERO": ("EMPTY", "ZERO"),
    }
    by_cell = {"NATIVE": {}}
    by_cell.update({name: {} for name in eval_cells})
    rows = []
    for film in films.values():
        film.eval()
    val_sessions = sorted(session for session, material in materials.items() if material.split == "val")
    core.require(len(val_sessions) == plan.VALIDATION_SESSIONS, "validation roster drift")
    for session in val_sessions:
        material = materials[session]
        state = cache[session]
        identities = {"NATIVE": state["native_identity"]}
        with torch.no_grad():
            for cell, (arm, profile_key) in eval_cells.items():
                identities[cell] = core.film_identity(
                    student.id_encoder,
                    state["mean_feature"],
                    state["carrier"],
                    state["profiles"][profile_key],
                    films[arm],
                )
            predictions = {name: [] for name in identities}
            targets = []
            for offset in range(0, material.q50_starts.size, 256):
                starts = material.q50_starts[offset : offset + 256]
                neural_np, target_np = _batch_arrays(material.record, starts)
                neural = torch.from_numpy(neural_np).to(device)
                targets.append(target_np)
                for name, identity in identities.items():
                    prediction = student.decode_with_identity(neural, identity.expand(neural.shape[0], -1, -1))[:, -1, :] / plan.BEHAVIOR_SCALE
                    predictions[name].append(prediction.detach().cpu().numpy().astype(np.float32))
        target = np.ascontiguousarray(np.concatenate(targets), dtype=np.float32)
        row = {
            "session": session,
            "window_count": int(target.shape[0]),
            "starts_sha256": core.array_sha256(material.q50_starts),
            "target_sha256": core.array_sha256(target),
            "cells": {},
        }
        for name in predictions:
            prediction = np.ascontiguousarray(np.concatenate(predictions[name]), dtype=np.float32)
            value = _r2(target, prediction)
            by_cell[name][session] = value
            row["cells"][name] = {"r2": value, "prediction_sha256": core.array_sha256(prediction)}
        rows.append(row)
    contrasts = {name: core.paired_summary(values, by_cell["NATIVE"]) for name, values in by_cell.items() if name != "NATIVE"}
    primary = {name: contrasts[name] for name in ("CP10@M10", "CP30@M30", "SHUFFLE10@M10", "EMPTY@ZERO")}
    return {
        "rows": rows,
        "per_session_r2": by_cell,
        "equal_session_mean_r2": {name: float(np.mean(list(values.values()))) for name, values in by_cell.items()},
        "contrasts_vs_native": contrasts,
        "decision": core.decide(primary),
    }


def execute(repo_root: Path, seed: int) -> dict[str, object]:
    repo_root = Path(repo_root).resolve()
    frozen = _verify_static(repo_root, seed)
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "launch requires isolated physical GPU0")
    core.require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "launch requires deterministic CUBLAS_WORKSPACE_CONFIG=:4096:8",
    )
    root = plan.result_root(repo_root, seed)
    core.require(root.parent.is_dir(), f"result parent missing: {root.parent}")
    core.require(not root.exists(), f"result root already exists: {root}")
    root.mkdir(mode=0o755)
    attempt_sha = _publish_json(
        root,
        "attempt.json",
        {
            "schema": f"{plan.SCHEMA}_attempt",
            "seed": seed,
            "frozen_inputs": frozen,
            "formal_test_files_opened": False,
            "gpu_initialized": False,
            "started_at_unix": time.time(),
        },
    )
    progress = {"stage": "attempt", "published": ["attempt.json"], "formal_test_files_opened": False}
    try:
        attestation = _gpu0_attestation()
        launch_sha = _publish_json(
            root,
            "launch.json",
            {
                "schema": f"{plan.SCHEMA}_launch",
                "attempt_sha256": attempt_sha,
                "seed": seed,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": attestation,
            },
        )
        progress.update(stage="launch", published=["attempt.json", "launch.json"])

        from .data import materialize, prepare_datamodule

        dm = prepare_datamodule(repo_root)
        materials = materialize(repo_root, dm)
        authority = {
            "schema": f"{plan.SCHEMA}_source_authority",
            "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "train_sessions": sorted(key for key, row in materials.items() if row.split == "train"),
            "validation_sessions": sorted(key for key, row in materials.items() if row.split == "val"),
            "formal_test_names_only": list(dm.session_splits["test"]),
            "formal_test_files_opened": False,
            "profiles": {
                key: {
                    "m10": row.evidence10,
                    "m30": row.evidence30,
                    "q50_window_count": int(row.q50_starts.size),
                    "q50_starts_sha256": core.array_sha256(row.q50_starts),
                    "t4_sha256": core.array_sha256(row.record.side_features),
                    "activity30_sha256": core.array_sha256(row.record.calib_trials),
                }
                for key, row in sorted(materials.items())
            },
        }
        authority_sha = _publish_json(root, "source_authority.json", authority)
        progress.update(stage="source_authority", published=["attempt.json", "launch.json", "source_authority.json"])

        import torch

        core.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "expected one visible CUDA device")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0")
        student = _prepare_student(repo_root, seed, device)
        base_state_before = _state_sha(student)
        template = core.build_film().to(device)
        films = {arm: copy.deepcopy(template).to(device) for arm in plan.ARMS}
        initial_states = {_state_sha(film) for film in films.values()}
        core.require(len(initial_states) == 1, "FiLM initial states are not identical")
        cache = _session_cache(materials, student, device, seed)
        progress["stage"] = "training"
        training = _train(materials=materials, cache=cache, student=student, films=films, seed=seed, device=device)
        base_state_after = _state_sha(student)
        core.require(base_state_after == base_state_before, "frozen substrate changed during training")

        checkpoint_receipts = {}
        for arm, film in films.items():
            buffer = tempfile.SpooledTemporaryFile(max_size=1 << 20)
            torch.save({"arm": arm, "seed": seed, "state_dict": film.state_dict()}, buffer)
            buffer.seek(0)
            body = buffer.read()
            name = f"film_{arm.lower()}.pt"
            checkpoint_receipts[arm] = {"path": name, "sha256": _publish_bytes(root, name, body), "state_sha256": _state_sha(film)}
        training_payload = {
            "schema": f"{plan.SCHEMA}_training",
            "attempt_sha256": attempt_sha,
            "source_authority_sha256": authority_sha,
            "seed": seed,
            "base_state_before_sha256": base_state_before,
            "base_state_after_sha256": base_state_after,
            "checkpoints": checkpoint_receipts,
            **training,
        }
        training_sha = _publish_json(root, "training.json", training_payload)
        progress.update(stage="scoring", published=["attempt.json", "launch.json", "source_authority.json", "training.json"])

        score = _evaluate(materials=materials, cache=cache, student=student, films=films, device=device)
        score_payload = {
            "schema": f"{plan.SCHEMA}_score",
            "attempt_sha256": attempt_sha,
            "source_authority_sha256": authority_sha,
            "training_sha256": training_sha,
            "seed": seed,
            **score,
        }
        score_sha = _publish_json(root, "score.json", score_payload)
        terminal = {
            "schema": f"{plan.SCHEMA}_terminal",
            "status": "PASS" if score["decision"]["seed_expansion_authorized"] else "STOP_NEGATIVE",
            "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "source_authority_sha256": authority_sha,
            "training_sha256": training_sha,
            "score_sha256": score_sha,
            "decision": score["decision"],
            "formal_test_files_opened": False,
            "target_session_updates": 0,
            "evalai_push": False,
        }
        _publish_json(root, "terminal.json", terminal)
        return terminal
    except BaseException as exc:
        if not (root / "terminal.json").exists() and not (root / "failure.json").exists():
            _publish_json(
                root,
                "failure.json",
                {
                    "schema": f"{plan.SCHEMA}_failure",
                    "attempt_sha256": attempt_sha,
                    "progress": progress,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "formal_test_files_opened": False,
                },
            )
        raise
